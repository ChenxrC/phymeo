"""COL2 protocol: no HOST search, explicit capacities, immutable/atomic output."""

import copy

import numpy as np
import pytest
import tyx as tx
from test_device_triangle_distance import assert_packet, new_batch, reference, saved
from test_device_triangle_distance import mode as mode


@pytest.mark.parametrize(
    "field", ("vertex_capacity", "face_capacity", "segment_capacity", "pair_capacity")
)
def test_exact_and_one_less_capacity(mode, field, tmp_path):
    data = reference().make_case("face_above")
    with new_batch(mode, data) as batch:
        with saved(tmp_path, field, mode, data, batch):
            assert_packet(data, batch.query(**data))
    count = {"vertex_capacity": 3, "face_capacity": 1, "segment_capacity": 1, "pair_capacity": 1}[
        field
    ]
    with pytest.raises(ValueError, match="capacity"):
        with new_batch(mode, data, **{field: count - 1}) as batch:
            batch.query(**data)


@pytest.mark.parametrize(
    "fault",
    (
        "vertex-id",
        "facet-id",
        "segment-id",
        "candidate",
        "candidate-duplicate",
        "radius",
        "finite",
        "dtype",
    ),
)
def test_preflight_rejects_without_launch(mode, fault):
    data = reference().make_case("shared_edge")
    if fault == "vertex-id":
        data["vertex_ids"][1] = data["vertex_ids"][0]
    if fault == "facet-id":
        data["facet_ids"][1] = data["facet_ids"][0]
    if fault == "segment-id":
        data["segment_ids"] = np.array((-1,), np.int64)
    if fault == "candidate":
        data["candidates"][0, 1] = 999
    if fault == "candidate-duplicate":
        data["candidates"][1] = data["candidates"][0]
    if fault == "radius":
        data["radii"][0] = -0.1
    if fault == "finite":
        data["vertices"][0, 0] = float("nan")
    if fault == "dtype":
        data["faces"] = data["faces"].astype(np.int32)
    with new_batch(mode, data) as batch:
        with pytest.raises(ValueError):
            batch.query(**data)
        assert batch.stats["launches"] == 0 and batch.last_packet is None


@pytest.mark.parametrize("fault", ("download", "ownership", "barycentric"))
def test_late_failure_preserves_packet_and_real_retry(mode, fault, tmp_path, monkeypatch):
    data = reference().make_case("face_above")
    with new_batch(mode, data) as batch:
        old = batch.query(**data)
        real = batch._download

        def fail(array, label):
            out = real(array, label)
            if label == "meta":
                if fault == "download":
                    raise RuntimeError("late download")
                if fault == "ownership":
                    out[0, 5] += 1
            if fault == "barycentric" and label == "summary":
                out[0, 3] = 2
            return out

        monkeypatch.setattr(batch, "_download", fail)
        with saved(tmp_path, fault, mode, data, batch) as record:
            with pytest.raises(RuntimeError):
                batch.query(**data)
            record["rejected_raw"] = batch.last_attempt_raw
            assert batch.last_packet is old and batch.stats["live_bytes"] == 0
            assert batch.stats["failed_queries"] == 1
            monkeypatch.setattr(batch, "_download", real)
            new = batch.query(**data)
            assert new is not old and new.input_binding == old.input_binding
            assert batch.stats["launches"] == 3
            assert_packet(data, new)
            record["retry"] = new


def test_snapshot_and_owner_binding(mode):
    data = reference().make_case("face_above")
    with new_batch(mode, data) as first, new_batch(mode, data) as second:
        packet = first.query(**data)
        with pytest.raises(ValueError, match="owner"):
            second.require_current(packet, **data)
        changed = copy.deepcopy(data)
        changed["vertices"][0, 2] = 0.125
        with pytest.raises(ValueError, match="binding"):
            first.require_current(packet, **changed)
        changed = copy.deepcopy(data)
        changed["snapshot_id"] = "next"
        with pytest.raises(ValueError, match="binding"):
            first.require_current(packet, **changed)
        current = first.query(**changed)
        with pytest.raises(ValueError, match="current"):
            first.require_current(packet, **data)
        assert current.snapshot_id == "next"
        with pytest.raises(ValueError):
            current.raw["meta"].setflags(write=True)
        with pytest.raises(TypeError):
            current.inputs["new"] = 1


def test_source_conversion_cannot_use_world_translation_as_scale(mode, tmp_path):
    data = reference().make_case("face_above")
    # Pre-execution correction: common translation is representable in a pair
    # frame. A distant small segment instead loses .25 relative to this triangle.
    data["segments"][:, :, 0] += 2**24
    with new_batch(mode, data) as batch:
        with pytest.raises(ValueError, match="precision"):
            batch.query(**data)
        assert batch.stats["launches"] == 0
        # Exactly representable but very distant data still cannot promise an
        # absolute distance error scaled only by the small local triangle.
        distant = reference().make_case("face_above")
        distant["snapshot_id"] = "distant-resolution"
        distant["segments"][:] = (2**24, 0, 0)
        with saved(tmp_path, "distant_resolution", mode, distant, batch) as record:
            packet = batch.query(**distant)
            record["packet"] = packet
            assert packet.rows[0].status == "CONDITIONING"
            assert packet.rows[0].normal is None


def test_no_host_distance_search(mode, monkeypatch, tmp_path):
    import phymeo.geometry._capsule_triangle_sweep as host

    def forbidden(*args, **kwargs):
        raise AssertionError("HOST closest-point search")

    monkeypatch.setattr(host, "segment_minimum", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    data = reference().make_case("parallel_face")
    with new_batch(mode, data) as batch:
        with saved(tmp_path, "no_host_search", mode, data, batch) as record:
            packet = batch.query(**data)
            assert packet.rows[0].status == "RESOLVED" and packet.rows[0].normal is not None
            record["parallel"] = packet
            coplanar = copy.deepcopy(data)
            coplanar["snapshot_id"] = "coplanar"
            coplanar["segments"][:, :, 2] = 0
            point = batch.query(**coplanar)
            assert_packet(coplanar, point)
            assert point.rows[0].status == "RESOLVED" and point.rows[0].normal is None
            record["coplanar"] = point
            band = copy.deepcopy(data)
            band["snapshot_id"] = "direction-band"
            band["segments"][:, :, 2] = 2**-20
            close = batch.query(**band)
            assert_packet(band, close)
            assert close.rows[0].status == "RESOLVED" and close.rows[0].normal is None
            assert close.rows[0].distance > 0
            record["direction_band"] = close
            assert batch.stats["completed_launches"] == 3


def test_empty_candidate_packet_has_no_launch(mode):
    data = reference().make_case("face_above")
    data["candidates"] = np.empty((0, 2), np.int64)
    with new_batch(mode, data, pair_capacity=1) as batch:
        packet = batch.query(**data)
        assert packet.rows == () and batch.stats["launches"] == 0
        assert batch.stats["live_bytes"] == 0
        batch.require_current(packet, **data)


def test_borrowed_runtime_survives_batch_close(mode):
    data = reference().make_case("face_above")
    runtime = tx.Runtime(backend="cuda" if mode == "cuda" else "cpu")
    try:
        with new_batch(mode, data, runtime=runtime) as batch:
            assert_packet(data, batch.query(**data))
        with new_batch(mode, data, runtime=runtime) as next_batch:
            assert_packet(data, next_batch.query(**data))
    finally:
        runtime.close()


class _FakeReleaseOwner:
    def __init__(self, name, events, phase=None):
        self.name, self.events, self.phase = name, events, phase
        self.calls = 0
        self.released = False

    def free(self):
        self.calls += 1
        self.events.append(self.name)
        if self.phase == "before":
            raise RuntimeError("authored before release")
        assert not self.released
        self.released = True
        if self.phase == "after":
            raise RuntimeError("authored after release")

    def close(self):
        self.free()


@pytest.mark.parametrize("phase", ("before", "after"))
@pytest.mark.parametrize("ownership", ("owned", "borrowed"))
def test_host_fake_release_retains_unknown_owner(phase, ownership, monkeypatch, tmp_path):
    import gc
    import json
    import weakref
    from types import SimpleNamespace

    from phymeo.backends.mujoco_tyx._resources import quarantined_resources

    def exercise():
        with monkeypatch.context() as patch:
            events = []
            runtime = _FakeReleaseOwner("runtime", events)
            patch.setattr(tx, "Runtime", lambda **kwargs: runtime)
            buffers = []

            def allocate(shape, dtype):
                del shape, dtype
                buffer = _FakeReleaseOwner(
                    f"buffer-{len(buffers)}", events, phase if len(buffers) == 1 else None
                )
                buffers.append(buffer)
                return buffer

            patch.setattr(tx, "cuda_empty", allocate)
            data = reference().make_case("face_above")
            extra = {} if ownership == "owned" else {"runtime": runtime}
            batch = new_batch("cuda", data, **extra)
            # HOST protocol sentinel, not a claimed physical packet.
            previous = SimpleNamespace(
                pair_origins_relative_m=np.empty((0, 3)),
                inputs={},
                uploaded={},
                conversion_deltas={},
                raw={},
            )
            batch._last_packet = previous
            resources = batch._new_query_resources()
            for _ in range(3):
                batch._allocate(resources, shape=(2, 3), dtype=np.float32)
            assert batch.stats["live_bytes"] == 72
            with pytest.raises(BaseExceptionGroup, match="UNKNOWN_RELEASE"):
                resources.close()
            assert resources.state == "UNKNOWN_RELEASE" and resources.runtime_pending
            assert resources.pending_buffer_count == 0
            assert len(resources.failures) == 1
            proxy = resources.failures[0].owner
            assert proxy.array is buffers[1]
            assert buffers[1].released is (phase == "after")
            assert batch.stats["live_bytes"] == 24
            assert events == ["buffer-2", "buffer-1", "buffer-0"]
            for operation in (lambda: batch.query(**data), batch.close, batch.close):
                with pytest.raises(BaseExceptionGroup, match="UNKNOWN_RELEASE"):
                    operation()
            assert all(buffer.calls == 1 for buffer in buffers)
            assert buffers[0].released and buffers[2].released
            assert runtime.calls == 0 and batch.last_packet is previous
            assert resources in quarantined_resources()
            assert resources._runtime.owner.runtime is runtime
            retained = weakref.ref(buffers[1])
            retained_runtime = weakref.ref(runtime)
            record = dict(
                phase=phase,
                ownership=ownership,
                events=list(events),
                live_unknown_bytes=24,
                ledger=resources.state,
                runtime_calls=runtime.calls,
                buffer_calls=[b.calls for b in buffers],
                side_effect_completed=buffers[1].released,
                old_packet_retained=True,
            )
            return retained, retained_runtime, record

    retained, retained_runtime, record = exercise()
    # Only the quarantined ledger may now retain the opaque owners.
    gc.collect()
    assert retained() is not None and retained_runtime() is not None
    (tmp_path / "fake-release.json").write_text(json.dumps(record), encoding="utf-8")


@pytest.mark.cuda
def test_cuda_post_free_error_is_sticky_and_does_not_publish(monkeypatch, tmp_path):
    from phymeo.backends.mujoco_tyx._resources import DeviceResources, quarantined_resources

    data = reference().make_case("face_above")
    batch = new_batch("cuda", data)
    actual_runtime = batch._runtime
    state = dict(calls=0, completed=False)
    captured = []
    original_add = DeviceResources.add_buffer
    selected = []
    try:
        old = batch.query(**data)

        def capture(resources, owner):
            if resources is batch._query_resources:
                captured.append(owner)
                if len(captured) == 4 and not selected:
                    selected.append(owner)
                    real_free = owner.array.free

                    def late_free():
                        state["calls"] += 1
                        real_free()
                        state["completed"] = True
                        raise RuntimeError("authored after actual CUDA free")

                    monkeypatch.setattr(owner.array, "free", late_free)
            return original_add(resources, owner)

        monkeypatch.setattr(DeviceResources, "add_buffer", capture)
        with saved(tmp_path, "cuda_release_unknown", "cuda", data, batch) as record:
            with pytest.raises(BaseExceptionGroup, match="UNKNOWN_RELEASE"):
                batch.query(**data)
            resources = batch._query_resources
            assert len(captured) == 8 and state["calls"] == 1 and state["completed"]
            assert batch.last_packet is old
            assert resources.state == "UNKNOWN_RELEASE" and resources.runtime_pending
            assert resources in quarantined_resources()
            assert resources.failures[0].owner is selected[0]
            assert all(owner.released for owner in captured if owner is not selected[0])
            assert not selected[0].released
            assert batch.stats["live_bytes"] == data["candidates"].shape[0] * 2 * 4
            assert not actual_runtime.closed
            for operation in (lambda: batch.query(**data), batch.close, batch.close):
                with pytest.raises(BaseExceptionGroup, match="UNKNOWN_RELEASE"):
                    operation()
            assert state["calls"] == 1 and not actual_runtime.closed
            record["release"] = dict(
                ledger=resources.state,
                calls=state["calls"],
                completed=state["completed"],
                siblings_released=7,
                unknown_owned_bytes=batch.stats["live_bytes"],
                old_packet_retained=True,
            )
    finally:
        # Test-only knowledge: the selected free returned normally before our
        # exception, and all independent siblings completed. No production path
        # infers this or retries the ambiguous free; its ledger remains sticky.
        if (
            state["completed"]
            and len(captured) == 8
            and all(owner.released for owner in captured if owner is not selected[0])
        ):
            actual_runtime.close()
        try:
            batch.close()
        except BaseExceptionGroup:
            pass
