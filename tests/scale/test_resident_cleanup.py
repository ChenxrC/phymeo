"""Release failure must retain the owner, callback and upstream dependencies."""

import pytest
from test_resident_publication import cache

from phymeo.backends._resident_cleanup import (
    ReleaseStack,
    ResidentCleanupError,
    RetryableReleaseError,
)
from phymeo.backends.tyx_world_batch import FreeSphere, TyxWorldBatch


def test_release_stack_stops_at_failed_dependency_and_does_not_repeat_success():
    stack = ReleaseStack()
    events = []
    failed = False

    def leaf():
        nonlocal failed
        events.append("leaf")
        if not failed:
            failed = True
            raise RetryableReleaseError("release failure")

    stack.callback(events.append, "runtime")
    stack.callback(leaf)
    stack.callback(events.append, "scratch")
    with pytest.raises(RuntimeError, match="release failure"):
        stack.close()
    assert events == ["scratch", "leaf"]
    stack.close()
    stack.close()
    assert events == ["scratch", "leaf", "leaf", "runtime"]


def fail_free(monkeypatch, buffer):
    """Fail before driver release; successful releases retain real CUDA behavior."""
    array_type = type(buffer)
    original = array_type.free
    calls = []
    enabled = [True]

    def release(value):
        calls.append(value)
        if value is buffer and enabled[0]:
            raise RetryableReleaseError("injected CUDA free failure")
        original(value)

    monkeypatch.setattr(array_type, "free", release)
    return enabled, calls


@pytest.mark.parametrize("count", [1, 8, 64])
def test_cuda_batch_retains_allocations_and_runtime_until_retry(count, tmp_path, monkeypatch):
    batch = TyxWorldBatch(
        [(w, FreeSphere(1, 0.125)) for w in range(count)], "cuda", cache_root=cache(tmp_path)
    )
    assert all(batch.step(1 / 1024, require_all=True).values())
    target = batch._states[0]
    before = batch.stats()
    enabled, calls = fail_free(monkeypatch, target)
    try:
        with pytest.raises(RuntimeError, match="CUDA free failure"):
            batch.close()
        stats = batch.stats()
        assert stats["closing"] and not stats["closed"]
        assert 0 < stats["device_owned_bytes"] < before["device_owned_bytes"]
        assert not batch._runtime.closed
        assert not target._freed
        for operation in (lambda: batch.step(1 / 1024), batch.snapshot, batch.reset):
            with pytest.raises(RuntimeError, match="closed or closing"):
                operation()
        enabled[0] = False
        batch.close()
        batch.close()
        assert batch.stats()["device_owned_bytes"] == 0
        assert batch.stats()["closed"] and batch._runtime.closed
        assert batch.stats()["synchronizations"] == before["synchronizations"]
        assert sum(value is target for value in calls) == 2
        successful = [id(value) for value in calls if value is not target]
        assert len(successful) == len(set(successful))
    finally:
        enabled[0] = False
        batch.close()


@pytest.mark.parametrize("kind", ["readback", "convex", "contact", "bundle"])
def test_cuda_dependent_failure_retains_parent_and_blocks_use(kind, tmp_path, monkeypatch):
    from test_resident_contact_bundle import _spec
    from test_resident_tree_contacts import H, _slider

    tree = _slider()[1] if kind != "readback" else FreeSphere(1, 0.125)
    batch = TyxWorldBatch([(7, tree)], "cuda", cache_root=cache(tmp_path))
    if kind == "readback":
        plan = batch.prepare_readback([7])
        plan.read()
        target, invoke = plan._rows, plan.read
        registry = batch._readback_plans
    elif kind == "convex":
        spec = _spec(batch, [7])
        broad = batch.prepare_collision(spec.model, spec.bindings)
        plan = batch.prepare_convex_query(broad)
        batch.convex_query(plan)
        target = next(value for value in plan._buffers.values() if hasattr(value, "free"))

        def invoke():
            return batch.convex_query(plan)

        registry = broad._convex_dependents
    else:
        spec = _spec(batch, [7])
        plan = (
            batch.prepare_contacts(spec.model, spec.bindings, spec.materials_by_pair)
            if kind == "contact"
            else batch.prepare_contact_bundle([spec])
        )
        batch.step_contacts(plan, H, require_all=True)
        # Bindings captured by the release stack are real contact scratch arrays.
        target = plan._contact_resources._callbacks[-1][1][0]

        def invoke():
            return batch.step_contacts(plan, H, require_all=True)

        registry = batch._collision_plans if kind == "contact" else batch._contact_bundles
    before = batch.stats()["device_owned_bytes"]
    enabled, _ = fail_free(monkeypatch, target)
    try:
        with pytest.raises(RuntimeError, match="CUDA free failure"):
            plan.close()
        assert plan in registry and not plan._closed and plan._closing
        assert not batch._runtime.closed and not target._freed
        with pytest.raises(RuntimeError, match="closed"):
            invoke()
        with pytest.raises(RuntimeError, match="CUDA free failure"):
            batch.close()
        assert not batch._runtime.closed and not batch.stats()["closed"]
        assert batch.stats()["device_owned_bytes"] > 0
        assert batch.stats()["device_owned_bytes"] <= before
        enabled[0] = False
        batch.close()
        assert plan._closed and plan not in registry
        assert batch._runtime.closed and batch.stats()["device_owned_bytes"] == 0
    finally:
        enabled[0] = False
        batch.close()


def test_failed_readback_construction_exposes_cleanup_owner(tmp_path, monkeypatch):
    from phymeo.backends._world_readback_plan import WorldReadbackPlan

    batch = TyxWorldBatch([(7, FreeSphere(1, 0.125))], "cuda", cache_root=cache(tmp_path))
    original = WorldReadbackPlan._allocate
    allocations = []
    switches = []

    def allocate(plan, *args):
        if allocations:
            raise MemoryError("injected allocation failure")
        result = original(plan, *args)
        allocations.append(result)
        switches.append(fail_free(monkeypatch, result)[0])
        return result

    monkeypatch.setattr(WorldReadbackPlan, "_allocate", allocate)
    try:
        with pytest.raises(ResidentCleanupError) as error:
            batch.prepare_readback([7])
        owner = error.value.cleanup_owner
        assert owner in batch._readback_plans and not owner._closed
        assert owner.stats()["device_owned_bytes"] == allocations[0].nbytes
        assert not batch._runtime.closed
        switches[0][0] = False
        owner.close()
        assert owner._closed and owner not in batch._readback_plans
        assert owner.stats()["device_owned_bytes"] == 0
        assert batch.step(1 / 1024, require_all=True) == {7: True}
    finally:
        for switch in switches:
            switch[0] = False
        batch.close()


def test_failed_recovery_requires_completion_before_free(tmp_path, monkeypatch):
    batch = TyxWorldBatch([(7, FreeSphere(1, 0.125))], "cuda", cache_root=cache(tmp_path))
    before = batch.stats()["device_owned_bytes"]
    original = batch._runtime.sync
    failing = [True]

    def sync():
        if failing[0]:
            raise RuntimeError("injected sync failure")
        original()

    monkeypatch.setattr(batch._runtime, "sync", sync)
    try:
        with pytest.raises(RuntimeError, match="sync failure"):
            batch.step(1 / 1024)
        assert batch._unavailable
        with pytest.raises(RuntimeError, match="sync failure"):
            batch.close()
        assert batch.stats()["device_owned_bytes"] == before
        assert not batch._runtime.closed
        failing[0] = False
        batch.close()
        assert batch._runtime.closed and batch.stats()["device_owned_bytes"] == 0
    finally:
        failing[0] = False
        batch.close()


def test_failed_nested_engine_construction_keeps_dependency(tmp_path, monkeypatch):
    from test_resident_collision import _bindings, _one_model

    from phymeo.backends.tyx_collision import TyxCollisionBatch

    batch = TyxWorldBatch([(7, FreeSphere(1, 0.125))], "cuda", cache_root=cache(tmp_path))
    original = TyxCollisionBatch._scratch
    allocated = []
    switches = []

    def allocate(engine, *args):
        if allocated:
            raise MemoryError("injected nested allocation failure")
        result = original(engine, *args)
        allocated.append(result)
        switches.append(fail_free(monkeypatch, result)[0])
        return result

    monkeypatch.setattr(TyxCollisionBatch, "_scratch", allocate)
    try:
        model, ball, wall, static = _one_model()
        with pytest.raises(ResidentCleanupError) as error:
            batch.prepare_collision(model, _bindings(ball, wall, static))
        plan = error.value.cleanup_owner
        engine = plan._cleanup_dependencies[0]
        assert plan in batch._collision_plans and plan._engine is None
        assert not engine._closed and engine.resident_device_bytes > 0
        assert not batch._runtime.closed and not allocated[0]._freed
        with pytest.raises(RuntimeError, match="CUDA free failure"):
            batch.close()
        switches[0][0] = False
        batch.close()
        assert engine._closed and plan._closed and batch._runtime.closed
        assert engine.resident_device_bytes == 0
    finally:
        for switch in switches:
            switch[0] = False
        batch.close()


def test_runtime_close_is_last_and_retry_does_not_replay_free(tmp_path, monkeypatch):
    import tyx as tx

    original = tx.Runtime.close
    calls = []
    failing = [True]
    target = [None]

    def close(runtime):
        if runtime is target[0]:
            calls.append(runtime)
            if failing[0]:
                raise RetryableReleaseError("injected runtime close failure")
        original(runtime)

    monkeypatch.setattr(tx.Runtime, "close", close)
    batch = TyxWorldBatch([(7, FreeSphere(1, 0.125))], "cuda", cache_root=cache(tmp_path))
    target[0] = batch._runtime
    try:
        batch.step(1 / 1024, require_all=True)
        with pytest.raises(RuntimeError, match="runtime close failure"):
            batch.close()
        assert batch.stats()["device_owned_bytes"] == 0
        assert batch.stats()["closing"] and not batch.stats()["closed"]
        assert len(batch._resources._callbacks) == 1
        failing[0] = False
        batch.close()
        batch.close()
        assert len(calls) == 2 and batch._runtime.closed
        assert batch.stats()["device_owned_bytes"] == 0
    finally:
        failing[0] = False
        batch.close()


def test_standalone_collision_rejects_access_after_partial_close(tmp_path, monkeypatch):
    import numpy as np

    from phymeo.backends.tyx_collision import TyxCollisionBatch

    engine = TyxCollisionBatch("cuda", np.array([0, 1], np.int32), [0], [], cache(tmp_path))
    enabled, _ = fail_free(monkeypatch, engine._envelopes)
    try:
        with pytest.raises(RuntimeError, match="CUDA free failure"):
            engine.close()
        assert engine.stats()["closing"] and not engine.stats()["closed"]
        calls = (
            lambda: engine.query(None),
            lambda: engine.query_device_bounds(trees=None, pending=None),
            lambda: engine.prepare_shape_bvh(None),
            lambda: engine.tree_hits(None, None, None, None, None, None),
            lambda: engine.heightfield_face_candidates(None, None, None, None),
            lambda: engine.entry_times([]),
            lambda: engine.hull_intervals(None, None, None, None),
        )
        for call in calls:
            with pytest.raises(RuntimeError, match="closed or closing"):
                call()
        enabled[0] = False
        engine.close()
        assert engine.stats()["closed"]
    finally:
        enabled[0] = False
        engine.close()
