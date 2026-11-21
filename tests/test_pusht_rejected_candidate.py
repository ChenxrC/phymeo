"""CPU-only tests of experimental evidence capture, with no physics execution."""

import json
from contextlib import contextmanager
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.area_pressure.pusht_rejected_candidate import capture_rejected_candidate


class _Device:
    def __init__(self, keys=(6, 5)):
        n = len(keys)
        self.plan = SimpleNamespace(resident_native_contacts=True, nq=9, nv=8)
        self.mode, self.width, self.active = "cuda", 8, 1
        self.inside, self.synced, self.reads = False, False, []
        self.runtime = SimpleNamespace(sync=self._sync)
        self.tx = SimpleNamespace(use_runtime=self._use_runtime)
        self.status = np.array([0, 8][:n], dtype=np.int32)
        self.area_native_meta = np.array([[0, 1, 2, 1], [8, 24, 25, 0]][:n], np.int32)
        self.area_native_trace_meta = np.zeros((n, 25, 3), np.int32)
        self.area_native_trace_meta[:, 1:, 0] = 1
        self.area_native_trace = np.arange(n * 25 * 30, dtype=np.float32).reshape(n, 25, 30)
        self.solver_diagnostics = np.zeros((n, 10), np.float32)
        self.constraint_counts = np.tile([8, 8, 32], (n, 1)).astype(np.int32)
        self.friction_sizes = np.tile([8, 56, 24, 0], (n, 1)).astype(np.int32)
        self.friction_pack_status = np.zeros(n, np.int32)
        self.friction_vectors = np.arange(n * 8 * 8, dtype=np.float32).reshape(n, 8, 8)
        self.area_native_applied = np.ones((n, 8), np.float32)
        self.physical_acc = np.full((n, 8), 2, np.float32)
        self.q = [np.full((n, 9), b + 10, np.float32) for b in range(2)]
        self.v = [np.full((n, 8), b + 20, np.float32) for b in range(2)]
        self.time = [np.full((n, 2), b + 30, np.float32) for b in range(2)]

    @contextmanager
    def _use_runtime(self, runtime):
        assert runtime is self.runtime and not self.inside
        self.inside = True
        try:
            yield
        finally:
            self.inside = False

    def _sync(self):
        assert self.inside
        self.synced = True

    def _get(self, buffer):
        assert self.inside and self.synced
        self.reads.append(buffer)
        return buffer.copy()

    def download(self):
        pytest.fail("capture must use small targeted downloads, not the complete physical result")


def _batch(*owners):
    return SimpleNamespace(_buckets=owners, _lock=RLock(), _check=lambda: None)


def _owner(keys=(6, 5)):
    return SimpleNamespace(world_keys=keys, _device=_Device(keys))


def _bytes(device):
    return {
        name: value.tobytes()
        if isinstance(value, np.ndarray)
        else tuple(buffer.tobytes() for buffer in value)
        for name, value in vars(device).items()
        if isinstance(value, np.ndarray) or name in ("q", "v", "time")
    }


def test_rejected_capture_maps_worlds_bounds_trace_and_never_publishes():
    owner = _owner()
    device = owner._device
    before, active = _bytes(device), device.active
    result = capture_rejected_candidate(
        _batch(owner), RuntimeError("candidate rejected"), max_trace_rows=3
    )
    assert result["capture_complete"] and not result["capture_errors"]
    assert result["publication"] == "unpublished" and result["freshness"] == "unverified"
    left, right = result["lanes"]
    assert [(lane["world"], lane["lane"]) for lane in result["lanes"]] == [(6, 0), (5, 1)]
    assert left["status"] == 0 and right["status"] == 8
    assert "not_an_acceptance" in left["status_name"]
    for lane in (left, right):
        assert lane["publication"] == "unpublished" and lane["freshness"] == "unverified"
        assert lane["layout_interpretable"] and "candidate_q" not in lane and "qacc" not in lane
        assert len(lane["trace_columns"]) == 30
    assert left["trace_rows_retained"] == 2 and not left["trace_truncated"]
    assert right["trace_rows_retained"] == 3 and right["trace_truncated"]
    np.testing.assert_array_equal(right["area_native_trace"], device.area_native_trace[1, :3])
    np.testing.assert_array_equal(right["input_q"], device.q[active][1])
    np.testing.assert_array_equal(right["native_trial_physical_acc"], device.physical_acc[1])
    assert result["columns"]["area_native_trace_meta"][1] == "status_before_area_evaluation"
    assert _bytes(device) == before and device.active == active and not device.inside
    json.dumps(result, allow_nan=False)


def test_rejected_capture_preserves_original_error_when_one_download_fails(monkeypatch):
    owner = _owner()
    device, original, cause = owner._device, RuntimeError("original rejection"), ValueError("cause")
    get = device._get

    def fail_trace(buffer):
        if buffer is device.area_native_trace:
            raise OSError("diagnostic readback failed")
        return get(buffer)

    monkeypatch.setattr(device, "_get", fail_trace)
    with pytest.raises(RuntimeError) as caught:
        try:
            raise original from cause
        except RuntimeError as error:
            result = capture_rejected_candidate(_batch(owner), error)
            raise
    assert caught.value is original and caught.value.__cause__ is cause
    assert result["original_error"] == {"type": "RuntimeError", "message": "original rejection"}
    assert not result["capture_complete"]
    assert result["capture_errors"][0]["field"] == "area_native_trace"
    for lane in result["lanes"]:
        assert lane["availability"] == "partial" and not lane["layout_interpretable"]
        assert "area_native_meta" in lane and "area_native_trace" not in lane
    assert not device.inside
    json.dumps(result, allow_nan=False)


def test_rejected_capture_nonfinites_are_explicit_and_invalid_counts_are_not_trusted():
    owner = _owner()
    device = owner._device
    device.area_native_trace[1, 0, :3] = [np.nan, np.inf, -np.inf]
    device.area_native_meta[1, 2] = 200
    result = capture_rejected_candidate(_batch(owner), RuntimeError("failed"))
    lane = result["lanes"][1]
    assert lane["area_native_trace"][0][:3] == ["NaN", "+Infinity", "-Infinity"]
    assert lane["trace_rows_retained"] == 25 and lane["trace_calls_recorded"] == 200
    assert not lane["trace_count_valid"] and not lane["layout_interpretable"]
    json.dumps(result, allow_nan=False)


def test_rejected_capture_does_not_interpret_other_routes_or_unattempted_buckets():
    one, two = _owner((6,)), _owner((5,))
    two._device.plan.resident_native_contacts = False
    result = capture_rejected_candidate(_batch(one, two), ValueError("preflight failure"))
    assert result["capture_complete"]
    assert [(lane["world"], lane["bucket"]) for lane in result["lanes"]] == [(6, 0), (5, 1)]
    assert all(not lane["layout_interpretable"] for lane in result["lanes"])
    assert all(lane["freshness"] == "unverified" for lane in result["lanes"])
    assert "area_native_meta" not in result["lanes"][1]
    assert len(two._device.reads) == 1  # Only status belongs to the generic layout.


@pytest.mark.parametrize("failure", ["closed", "sync", "shape", "bound"])
def test_rejected_capture_marks_unavailable_evidence_without_replacing_failure(failure):
    owner = _owner()
    batch = _batch(owner)
    kwargs = {}

    def unavailable():
        raise RuntimeError("diagnostics unavailable")

    if failure == "closed":
        batch._check = unavailable
    elif failure == "sync":
        owner._device.runtime.sync = unavailable
    elif failure == "shape":
        owner._device.area_native_trace = np.empty((2, 25000, 30), np.float32)
    else:
        kwargs["max_trace_rows"] = 26
    original = ValueError("physics rejected")
    result = capture_rejected_candidate(batch, original, **kwargs)
    assert not result["capture_complete"] and result["capture_errors"]
    assert result["original_error"] == {"type": "ValueError", "message": "physics rejected"}
    if failure == "sync":
        assert all(lane["availability"] == "unavailable" for lane in result["lanes"])
        assert not owner._device.reads
    if failure == "shape":
        assert all(buffer is not owner._device.area_native_trace for buffer in owner._device.reads)
    json.dumps(result, allow_nan=False)


def test_rejected_capture_retains_evaluation_half_step_phases_without_claiming_validation():
    owner = _owner((6,))
    device = owner._device
    device.status[0] = 8  # The one-world fixture otherwise initializes lane 0 to status zero.
    device.area_native_meta[0] = (8, 3, 4, 0)
    device.area_native_trace_meta[0, :4, 0] = (0, 1, 2, 2)
    result = capture_rejected_candidate(_batch(owner), RuntimeError("candidate rejected"))
    lane = result["lanes"][0]
    assert [row[0] for row in lane["area_native_trace_meta"]] == [0, 1, 2, 2]
    assert result["columns"]["area_native_trace_meta"][0] == (
        "evaluation_phase_0_baseline_1_full_2_half"
    )
    assert result["columns"]["trace_phase_values"][2] == (
        "2: this evaluation uses a half update if not converged; half mode persists within this solve"
    )
    # This helper records the producer's values; it is not the restore validator
    # and does not certify the trigger, convergence, or freshness of this scratch.
    assert lane["publication"] == "unpublished" and lane["freshness"] == "unverified"
    assert lane["status"] == 8 and lane["area_native_meta"][3] == 0
