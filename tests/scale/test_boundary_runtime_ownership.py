"""Borrowed query buffers must not outlive the shared resident runtime."""

import gc
import weakref

import pytest

from phymeo.backends import tyx_world_batch as batch_module
from phymeo.backends.mujoco_tyx._resources import DeviceResources
from phymeo.backends.tyx_world_batch import TyxWorldBatch


class Runtime:
    def __init__(self, events):
        self.events = events
        self.calls = 0

    def close(self):
        self.calls += 1
        self.events.append("runtime")


class Lease:
    def __init__(self, batch):
        self.batch = batch
        self.runtime = batch._runtime
        self.token = batch._register_boundary_runtime_lease(self)

    def close(self):
        self.batch._release_boundary_runtime_lease(self.token)


class Buffer:
    def __init__(self, events, name, error=None):
        self.events, self.name, self.error = events, name, error
        self.calls = 0

    def free(self):
        self.calls += 1
        self.events.append(self.name)
        if self.error is not None:
            raise self.error


class Plan:
    def __init__(self, batch, events, name, *, error=None, pending_once=False):
        self.batch = batch
        self.lease = Lease(batch)
        self.ledger = DeviceResources()
        self.ledger.set_runtime(self.lease)
        self.buffer = Buffer(events, name, error)
        self.ledger.add_buffer(self.buffer)
        self.pending_once = pending_once
        batch._boundary_plans.add(self)

    def close(self):
        if self.pending_once:
            self.pending_once = False
            raise RuntimeError("before ledger entry")
        try:
            self.ledger.close()
        except BaseException:
            if self.ledger.state == "UNKNOWN_RELEASE":
                self.batch._quarantine_boundary_runtime_lease(self.lease.token, self.ledger)
            raise
        self.batch._boundary_plans.discard(self)


@pytest.fixture
def owned(monkeypatch):
    # All owners are synthetic. No device free or project physics is fabricated.
    from phymeo.backends.mujoco_tyx import _resources

    monkeypatch.setattr(_resources, "_QUARANTINE", {})
    monkeypatch.setattr(batch_module, "_BOUNDARY_RUNTIME_QUARANTINE", {})
    events = []
    batch = TyxWorldBatch((), mode="interpret")
    batch._runtime = Runtime(events)
    batch._resources.callback(batch._runtime.close)
    batch._resources.callback(events.append, "parent-buffer")
    batch._counts["device_owned_bytes"] = 123
    return batch, events


def test_normal_close_releases_queries_before_parent_runtime_once(owned):
    batch, events = owned
    a, b = Plan(batch, events, "a"), Plan(batch, events, "b")
    batch.close()
    batch.close()
    assert set(events[:2]) == {"a", "b"}
    assert events[2:] == ["parent-buffer", "runtime"]
    assert a.buffer.calls == b.buffer.calls == batch._runtime.calls == 1
    assert not batch._boundary_runtime_leases and not batch._boundary_plans
    assert batch.stats()["device_owned_bytes"] == 0


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_unknown_release_keeps_parent_and_cleans_independent_siblings(owned, error_type):
    batch, events = owned
    original = error_type("opaque release")
    bad = Plan(batch, events, "bad", error=original)
    good = Plan(batch, events, "good")
    for _ in range(2):
        with pytest.raises(BaseExceptionGroup):
            batch.close()
    assert set(events) == {"bad", "good"}
    assert bad.buffer.calls == good.buffer.calls == 1
    assert bad.ledger.failures[0].error is original
    assert batch._runtime.calls == 0 and not batch._closed
    assert batch.stats()["device_owned_bytes"] == 123
    assert bad.lease.token in batch._boundary_runtime_leases
    with pytest.raises(RuntimeError, match="UNKNOWN_RELEASE"):
        batch.step(0.001)
    with pytest.raises(RuntimeError, match="UNKNOWN_RELEASE"):
        batch._release_boundary_runtime_lease(bad.lease.token)


def test_pending_outer_failure_allows_close_retry_but_not_state_use(owned):
    batch, events = owned
    plan = Plan(batch, events, "query", pending_once=True)
    with pytest.raises(BaseExceptionGroup, match="boundary"):
        batch.close()
    assert not events and plan.ledger.state == "OPEN"
    assert batch.stats()["device_owned_bytes"] == 123
    with pytest.raises(RuntimeError, match="cleanup pending"):
        batch.set_controls({})
    assert not batch_module._BOUNDARY_RUNTIME_QUARANTINE
    batch.close()
    assert events == ["query", "parent-buffer", "runtime"]
    assert plan.buffer.calls == 1


def test_unowned_pending_lease_cannot_be_erased_by_batch_close(owned):
    batch, events = owned
    lease = Lease(batch)
    with pytest.raises(RuntimeError, match="pending"):
        batch.close()
    assert not events and not batch._closed
    lease.close()
    batch.close()
    assert events == ["parent-buffer", "runtime"]


def test_wrong_runtime_duplicate_and_foreign_tokens_are_rejected(owned):
    batch, events = owned
    lease = Lease(batch)
    with pytest.raises(ValueError, match="already registered"):
        batch._register_boundary_runtime_lease(lease)
    wrong = object.__new__(Lease)
    wrong.runtime = Runtime(events)
    with pytest.raises(ValueError, match="runtime"):
        batch._register_boundary_runtime_lease(wrong)
    with pytest.raises(ValueError, match="token"):
        batch._release_boundary_runtime_lease(object())
    with pytest.raises(ValueError, match="UNKNOWN_RELEASE"):
        batch._quarantine_boundary_runtime_lease(lease.token, DeviceResources())
    lease.close()
    batch.close()


def test_unknown_query_keeps_batch_runtime_alive_after_caller_drops_them(owned):
    batch, events = owned
    plan = Plan(batch, events, "bad", error=RuntimeError("opaque release"))
    with pytest.raises(BaseExceptionGroup):
        plan.close()
    refs = [weakref.ref(x) for x in (batch, batch._runtime, plan.ledger, plan.buffer)]
    del batch, plan
    # Drop the fixture tuple as well; the global quarantine must retain the owner.
    owned = None
    gc.collect()
    assert all(ref() is not None for ref in refs)
    assert refs[0]() in batch_module._BOUNDARY_RUNTIME_QUARANTINE.values()
    assert events == ["bad"]


def test_boundary_loan_rejects_reentrant_mutation_and_close_then_expires(owned):
    batch, events = owned
    with batch._borrow_boundary_collision_state() as accepted:
        assert accepted.state_epoch == 0
        for mutate in (lambda: batch.step(0.001), lambda: batch.set_controls({}), batch.close):
            with pytest.raises(RuntimeError, match="boundary source loan"):
                mutate()
        lease = Lease(batch)
        lease.close()
    assert not events
    assert batch.step(0.001) == {}
    batch.close()
    assert events == ["parent-buffer", "runtime"]


def test_boundary_loan_expires_on_base_exception_without_poisoning_owner(owned):
    batch, _ = owned
    with pytest.raises(KeyboardInterrupt):
        with batch._borrow_boundary_collision_state():
            raise KeyboardInterrupt()
    assert batch.step(0.001) == {}
    batch.close()


def test_recursive_cleanup_preserves_parent_until_pending_plan_retry(owned):
    batch, events = owned
    plan = Plan(batch, events, "query")
    original = plan.close
    plan.close = batch.close
    with pytest.raises(BaseExceptionGroup):
        batch.close()
    assert not events
    plan.close = original
    batch.close()
    assert events == ["query", "parent-buffer", "runtime"]
