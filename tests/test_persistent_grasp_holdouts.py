"""Short protocol/IO tests only; no four-window grasp acceptance is run here.

UNEXECUTED by the author. Real protocol tests use at most one original open-state
physical step; remaining cases are explicit passive callback/file stand-ins.
"""

import gzip
import json
import sys
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

CHECK_NAMES = (
    "actual_bilateral_acquisition",
    "expected_holding_outcome",
    "actual_release",
    "primary_supported_after_release",
    "second_body_has_real_support",
    "actuator_caps",
    "free_body_impulses",
    "no_execution_failure",
)


def _driver():
    name = "test_persistent_grasp_holdouts_driver"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "examples" / "persistent_grasp_holdouts.py"
        spec = spec_from_file_location(name, path)
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def test_original_accepted_callback_io_error_keeps_real_step_and_owned_record():
    driver = _driver()
    original = driver._example("robot_force_acquisition")
    scene, controller, info = original.make_case("mimic-box")
    accepted, rejected = [], []
    with scene:

        def stop_after_accepted(record):
            driver._validate_record(record, "accepted")
            accepted.append(record)
            assert record["entry"]["reading"].step_number == 0
            assert record["accepted"]["reading"].step_number == 1
            assert record["proposal"] is not None
            expected = scene.state.rigid.position.copy()
            record["accepted"]["state"].rigid.position[:] = 123.0
            np.testing.assert_array_equal(scene.state.rigid.position, expected)
            raise OSError("passive archive failed after real acceptance")

        with pytest.raises(OSError, match="after real acceptance"):
            original.run_case(
                scene,
                controller,
                info,
                on_accepted=stop_after_accepted,
                on_rejected=rejected.append,
            )
        assert len(accepted) == 1 and rejected == []
        assert scene.observe().step_number == 1
        assert scene.observe().time_s == info["dt"]
        assert scene.observe().load_time_s == 0
        declarations = driver.declaration_arguments(info)
        assert declarations["pads"] == info["pads"]
        assert tuple(k.token for keys in declarations["station_keys"].values() for k in keys) == (
            101,
            107,
            131,
            149,
            211,
            239,
            263,
            307,
        )


@pytest.mark.parametrize("failure_at", ["step", "propose"])
def test_original_rejected_callback_preserves_request_and_full_retained_state(
    failure_at, monkeypatch
):
    driver = _driver()
    original = driver._example("robot_force_acquisition")
    scene, controller, info = original.make_case("mimic-box")
    accepted, rejected, requests = [], [], []
    with scene:

        def reject_step(dt, *, actions, **kwargs):
            assert dt == info["dt"]
            requests.append(actions[10])
            raise ValueError("frozen protocol rejection")

        def reject_propose(*args, **kwargs):
            raise ValueError("frozen proposal rejection")

        monkeypatch.setattr(scene, "step", reject_step)
        if failure_at == "propose":
            monkeypatch.setattr(controller, "propose", reject_propose)
        result = original.run_case(
            scene, controller, info, on_accepted=accepted.append, on_rejected=rejected.append
        )
        assert accepted == [] and len(rejected) == 1
        record = rejected[0]
        driver._validate_record(record, "retained")
        assert record["failure"]["type"] == "ValueError"
        assert record["failure"]["accepted_step"] == 0
        assert record["failure"]["accepted_time_s"] == 0
        assert record["failure"]["message"] == (
            "frozen protocol rejection" if failure_at == "step" else "frozen proposal rejection"
        )
        if failure_at == "step":
            assert len(requests) == 1
            assert driver._public(record["proposal"].command) == driver._public(requests[0])
        else:
            assert requests == [] and record["proposal"] is None
        assert scene.observe().step_number == 0
        assert result["accepted_steps"] == 0
        assert not result["checks"]["no_execution_failure"]
        assert driver._public(record["entry"]) == driver._public(record["retained"])


def test_rejected_callback_io_failure_is_not_reclassified_or_retried(monkeypatch):
    driver = _driver()
    original = driver._example("robot_force_acquisition")
    scene, controller, info = original.make_case("mimic-box")
    calls = []
    with scene:

        def reject(*args, **kwargs):
            calls.append("physical-attempt")
            raise ValueError("original physical error")

        def archive_failure(record):
            assert record["failure"]["message"] == "original physical error"
            calls.append("observer")
            raise OSError("independent rejected-observer error")

        monkeypatch.setattr(scene, "step", reject)
        with pytest.raises(OSError, match="rejected-observer"):
            original.run_case(scene, controller, info, on_rejected=archive_failure)
        assert calls == ["physical-attempt", "observer"]
        assert scene.observe().step_number == 0


def test_archive_roundtrip_retains_complete_public_arrays_and_forbids_reuse(tmp_path):
    driver = _driver()
    path = tmp_path / "owned.jsonl.gz"
    payload = dict(
        q=np.array([1 / 8, -1 / 16], dtype=np.float64),
        keys=((0, 101), (0, 307)),
        request=dict(feedforward=None, target_velocity=np.zeros(2)),
    )
    archive = driver.ObservationArchive(path)
    try:
        archive.write("input", payload)
        archive.write("result", dict(checks={name: True for name in CHECK_NAMES}, trace=[payload]))
        assert archive.records_written == 2
        assert archive.bytes_written == path.stat().st_size
    finally:
        archive.close()
    saved = _rows(path)
    assert [row["kind"] for row in saved] == ["input", "result"]
    assert saved[0]["value"] == driver._public(payload)
    assert saved[1]["value"] == driver._public(
        dict(checks={name: True for name in CHECK_NAMES}, trace=[payload])
    )
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        driver.ObservationArchive(path)
    assert path.read_bytes() == before


def test_archive_cap_keeps_pending_complete_record_and_reports_stop(tmp_path):
    driver = _driver()
    payload = dict(request=dict(target_q=[0.0, 0.012]), accepted_step=17)
    record_size = len(driver._member("accepted", payload))
    footer = 8192
    cap, reserve = 2 * record_size + footer, record_size + footer
    archive = driver.ObservationArchive(
        tmp_path / "capacity.jsonl.gz", max_bytes=cap, reserve_bytes=reserve, footer_bytes=footer
    )
    try:
        archive.write("accepted", payload)
        with pytest.raises(driver.ArchiveLimit, match="ordinary observation budget"):
            archive.write("accepted", payload)
        assert archive.records_written == 1
        assert archive.stop(dict(type="ArchiveLimit", message="stop at actual accepted step 17"))
        assert archive.bytes_written <= cap
        assert archive.bytes_written == archive.path.stat().st_size
    finally:
        archive.close()
    saved = _rows(archive.path)
    assert [row["kind"] for row in saved] == ["accepted", "accepted", "incomplete"]
    assert saved[0]["value"] == saved[1]["value"] == driver._public(payload)


def test_partial_write_restores_only_new_member_boundary(tmp_path):
    driver = _driver()
    archive = driver.ObservationArchive(tmp_path / "partial.jsonl.gz")
    archive.write("first", dict(step=1))
    first = archive.path.read_bytes()
    raw = archive._stream

    class FailOnce:
        failed = False

        def write(self, value):
            if not self.failed:
                self.failed = True
                raw.write(value[: len(value) // 2])
                raw.flush()
                raise OSError("partial new member")
            return raw.write(value)

        def __getattr__(self, name):
            return getattr(raw, name)

    archive._stream = FailOnce()
    try:
        with pytest.raises(OSError, match="partial new member"):
            archive.write("second", dict(step=2))
        assert archive.path.read_bytes() == first
        assert archive.prefix_intact
        assert archive.records_written == 1
        assert archive.stop(dict(type="OSError", message="preserved complete prefix"))
    finally:
        archive.close()
    assert [row["kind"] for row in _rows(archive.path)] == ["first", "second", "incomplete"]


@dataclass(frozen=True)
class _Reading:
    step_number: int
    time_s: float
    load_time_s: float | None


@dataclass(frozen=True)
class _History:
    active: tuple = ()
    ended: tuple = ()
    dropped_spring_energy_j: float = 0.0


@dataclass(frozen=True)
class _Receipt:
    material_motion_error_m: float = 0.0
    application_motion_error_m: float = 0.0
    material_application_error_m: float = 0.0
    energy_residual_j: float = 0.0
    actual_work_j: float = 0.0


def _snapshot(step):
    return dict(
        state=dict(q=np.array([0.0, 0.0])),
        reading=_Reading(step, step * 0.001, None if step == 0 else 0.0),
        geometry=dict(time_s=step * 0.001),
        history=_History(),
        receipt=_Receipt(),
        events=SimpleEvents(),
        last_frame=None,
    )


@dataclass(frozen=True)
class SimpleEvents:
    active: tuple = ()


def _record(accepted=True):
    result = dict(
        schema="grasp-protocol-observation-v1",
        stage="acquiring",
        dt_s=0.001,
        proposal=dict(command=dict(target_q=(0.0, 0.001), feedforward=None)),
        entry=_snapshot(0),
    )
    if accepted:
        result["accepted"] = _snapshot(1)
    else:
        result["retained"] = _snapshot(0)
        result["failure"] = dict(
            type="ValueError",
            message="real boundary rejection",
            accepted_step=0,
            accepted_time_s=0.0,
        )
    return result


@pytest.mark.parametrize("accepted", [True, False])
def test_driver_calls_original_protocol_once_and_preserves_its_gate_set(accepted, tmp_path):
    driver = _driver()
    calls, scene, controller, info = [], object(), object(), {"friction": 0.65}
    record = _record(accepted)
    original = dict(
        checks={name: True for name in CHECK_NAMES},
        trace=[dict(original="entire row")],
        events=[dict(original="entire event")],
        accepted_steps=int(accepted),
        execution_failure=None if accepted else record["failure"],
    )
    if not accepted:
        original["checks"]["no_execution_failure"] = False

    def run(actual_scene, actual_controller, actual_info, *, on_accepted, on_rejected):
        assert actual_scene is scene and actual_controller is controller and actual_info is info
        calls.append("original protocol")
        assert (on_accepted if accepted else on_rejected)(record) is None
        return original

    archive = driver.ObservationArchive(tmp_path / "protocol.jsonl.gz")
    try:
        summary = driver._run_protocol(
            SimpleNamespace(run_case=run), scene, controller, info, archive
        )
    finally:
        archive.close()
    assert calls == ["original protocol"]
    assert summary["original_result"] is original
    assert tuple(original["checks"]) == CHECK_NAMES
    assert summary["status"] == ("complete" if accepted else "physics_rejected")
    assert summary["accepted_callback_count"] == int(accepted)
    assert summary["rejected_callback_count"] == int(not accepted)
    saved = _rows(archive.path)
    assert saved[-1]["kind"] == "original_result"
    assert saved[-1]["value"] == driver._public(original)


def test_driver_observer_write_failure_does_not_retry_accepted_protocol(tmp_path, monkeypatch):
    driver = _driver()
    record, calls = _record(), []

    def run(scene, controller, info, *, on_accepted, on_rejected):
        calls.append("one accepted step")
        on_accepted(record)
        pytest.fail("observer failure must abort this consumer without another physical step")

    archive = driver.ObservationArchive(tmp_path / "observer-error.jsonl.gz")
    write = archive.write

    def fail_one(kind, value):
        if kind == "accepted":
            raise OSError("independent archive failure")
        return write(kind, value)

    monkeypatch.setattr(archive, "write", fail_one)
    try:
        summary = driver._run_protocol(
            SimpleNamespace(run_case=run), object(), object(), {"friction": 0.65}, archive
        )
    finally:
        archive.close()
    assert calls == ["one accepted step"]
    assert summary["status"] == "observation_incomplete"
    assert summary["original_result"] is None
    assert summary["failure"]["accepted_step"] == 1
    assert summary["failure"]["physical_state_not_rolled_back"]
    assert summary["failure"]["last_record_saved"]
    saved = _rows(archive.path)
    assert [row["kind"] for row in saved] == ["observer_failed_record", "incomplete"]
    assert saved[0]["value"] == driver._public(record)


@pytest.mark.parametrize("destination", ["relative", "existing"])
def test_driver_output_guard_runs_before_any_model_or_protocol(destination, tmp_path, monkeypatch):
    driver = _driver()

    def forbidden():
        pytest.fail("invalid output must fail before source or model loading")

    monkeypatch.setattr(driver, "_source_manifest", forbidden)
    path = Path("relative-grasp-output") if destination == "relative" else tmp_path
    with pytest.raises(ValueError if destination == "relative" else FileExistsError):
        driver.run_holdout("mimic-box", path)
    assert driver.CASES == ("mimic-box", "mimic-zero-friction", "mimic-overload", "panda-sphere")
    assert driver.KT == 200.0
    assert driver.MAX_COMPRESSED_BYTES == 256 * 1024 * 1024


def test_callback_schema_failure_preserves_original_error_and_raw_record(tmp_path):
    driver = _driver()
    malformed = dict(schema="not-the-frozen-schema", stage="acquiring", request="unchanged")

    def run(scene, controller, info, *, on_accepted, on_rejected):
        on_accepted(malformed)
        pytest.fail("invalid callback protocol cannot continue")

    archive = driver.ObservationArchive(tmp_path / "schema-error.jsonl.gz")
    try:
        summary = driver._run_protocol(
            SimpleNamespace(run_case=run), object(), object(), {"friction": 0.65}, archive
        )
    finally:
        archive.close()
    assert summary["status"] == "observation_incomplete"
    assert summary["failure"]["type"] == "ValueError"
    assert summary["failure"]["message"] == "unexpected original grasp callback protocol"
    assert summary["failure"]["accepted_step"] is None
    assert _rows(archive.path)[0]["value"] == driver._public(malformed)
