"""Release outcomes across the integrated resident and diagnostic owners."""

import gc
import weakref
from types import SimpleNamespace

import pytest

from phymeo.backends import _resident_cleanup as cleanup
from phymeo.backends._device_convex_query_trace import _TraceCapture
from phymeo.backends.tyx_world_batch import TyxWorldBatch


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_unknown_release_is_once_only_and_retains_owners(error_type, monkeypatch):
    monkeypatch.setattr(cleanup, "_UNKNOWN_RELEASE_STACKS", {})
    events = []

    class Owner:
        def free(self):
            events.append("free")
            raise error_type("opaque failure")

    owner = Owner()
    ref = weakref.ref(owner)
    stack = cleanup.ReleaseStack()
    stack.callback(events.append, "parent")
    stack.callback(owner.free)
    for _ in range(2):
        with pytest.raises(cleanup.UnknownReleaseError, match="UNKNOWN_RELEASE"):
            stack.close()
    assert events == ["free"]
    del owner, stack
    gc.collect()
    assert ref() is not None


@pytest.mark.parametrize("retryable", [True, False])
def test_trace_release_failure_defers_cleanup_and_keeps_parent(retryable, monkeypatch):
    monkeypatch.setattr(cleanup, "_UNKNOWN_RELEASE_STACKS", {})
    events = []
    plan = SimpleNamespace(_closing=False)
    trace = object.__new__(_TraceCapture)
    trace.plan = plan
    trace.resources = cleanup.ReleaseStack()
    trace.buffers = {"owned": object()}
    trace.released = trace.release_failed = False

    def release():
        events.append("release")
        if not retryable:
            raise RuntimeError("opaque")
        if len(events) == 1:
            raise cleanup.RetryableReleaseError("before release")

    trace.resources.callback(release)
    with pytest.raises(RuntimeError):
        with trace:
            trace.release()
    assert events == ["release"]  # __exit__ cannot silently retry.
    assert not trace.released and trace.buffers
    assert plan._closing and plan._cleanup_dependencies == [trace]
    if retryable:
        cleanup.close_dependencies(plan)
        assert trace.released and not trace.buffers
        assert not plan._cleanup_dependencies
        assert events == ["release", "release"]
    else:
        with pytest.raises(cleanup.UnknownReleaseError):
            cleanup.close_dependencies(plan)
        assert plan._cleanup_dependencies == [trace]
        assert events == ["release"]


def test_boundary_pending_then_prepared_plan_retry_keeps_parent():
    batch = TyxWorldBatch((), mode="interpret")
    events = []

    class Plan:
        def __init__(self, name, registry):
            self.name, self.registry, self.calls = name, registry, 0
            registry.add(self)

        def close(self):
            self.calls += 1
            if self.calls == 1:
                raise cleanup.RetryableReleaseError("before release")
            events.append(self.name)
            self.registry.remove(self)

    boundary = Plan("boundary", batch._boundary_plans)
    prepared = Plan("prepared", batch._control_plans)
    batch._resources.callback(events.append, "parent")
    with pytest.raises(BaseExceptionGroup):
        batch.close()
    assert not events and prepared.calls == 0
    with pytest.raises(RuntimeError, match="cleanup pending"):
        batch.step(0.01)
    with pytest.raises(cleanup.RetryableReleaseError):
        batch.close()
    assert events == ["boundary"] and boundary.calls == 2
    batch.close()
    batch.close()
    assert events == ["boundary", "prepared", "parent"]
    assert prepared.calls == 2 and batch._closed


def test_cleanup_reentry_and_late_registration_fail_closed(monkeypatch):
    monkeypatch.setattr(cleanup, "_UNKNOWN_RELEASE_STACKS", {})
    stack = cleanup.ReleaseStack()
    stack.callback(stack.close)
    with pytest.raises(cleanup.UnknownReleaseError) as failure:
        stack.close()
    assert "recursive" in str(failure.value.__cause__)
    with pytest.raises(RuntimeError, match="register"):
        stack.callback(lambda: None)
    with pytest.raises(cleanup.UnknownReleaseError):
        stack.close()
    empty = cleanup.ReleaseStack()
    empty.close()
    with pytest.raises(RuntimeError, match="register"):
        empty.callback(lambda: None)
