"""Release failures retain unknown owners without retrying native side effects."""

import gc
import json
import weakref

import pytest

from phymeo.backends.mujoco_tyx._resources import DeviceResources, quarantined_resources


class FakeOwner:
    def __init__(self, name, events, *, fault=None, error=None):
        self.name, self.events, self.fault = name, events, fault
        self.error = error if error is not None else RuntimeError(name)
        self.calls, self.released = 0, False

    def _release(self):
        self.calls += 1
        self.events.append(self.name)
        if self.fault == "before":
            raise self.error
        assert not self.released, "a release side effect must not happen twice"
        self.released = True
        if self.fault == "after":
            raise self.error

    def free(self):
        self._release()

    def close(self):
        self._release()


def test_successful_owners_close_in_dependency_order_once():
    events = []
    resources = DeviceResources()
    runtime = FakeOwner("runtime", events)
    buffers = [FakeOwner(str(i), events) for i in range(3)]
    resources.set_runtime(runtime)
    for owner in buffers:
        resources.add_buffer(owner)
    assert resources.state == "OPEN"
    resources.close()
    resources.close()
    assert events == ["2", "1", "0", "runtime"]
    assert resources.closed and resources.state == "CLOSED"
    assert resources.pending_buffer_count == 0 and not resources.runtime_pending
    assert resources.failures == ()
    assert all(owner.calls == 1 for owner in [*buffers, runtime])
    assert resources not in quarantined_resources()


@pytest.mark.parametrize("phase", ("before", "after"))
def test_opaque_buffer_failure_is_sticky_and_keeps_runtime(phase):
    events = []
    resources = DeviceResources()
    runtime = FakeOwner("runtime", events)
    buffers = [FakeOwner(str(i), events, fault=phase if i == 1 else None) for i in range(3)]
    resources.set_runtime(runtime)
    for owner in buffers:
        resources.add_buffer(owner)
    for _ in range(2):
        with pytest.raises(BaseExceptionGroup) as caught:
            resources.close()
        assert caught.value.exceptions == (buffers[1].error,)
    assert events == ["2", "1", "0"]
    assert runtime.calls == 0 and resources.runtime_pending
    assert resources.pending_buffer_count == 0
    assert resources.state == "UNKNOWN_RELEASE" and not resources.closed
    assert resources.failures[0].owner is buffers[1]
    assert resources.failures[0].kind == "buffer"
    assert resources.failures[0].error is buffers[1].error
    assert buffers[1].released is (phase == "after")
    assert all(owner.calls == 1 for owner in buffers)
    assert resources in quarantined_resources()


@pytest.mark.parametrize("phase", ("before", "after"))
def test_opaque_runtime_failure_never_repeats_release(phase):
    events = []
    resources = DeviceResources()
    runtime = FakeOwner("runtime", events, fault=phase)
    buffer = FakeOwner("buffer", events)
    resources.set_runtime(runtime)
    resources.add_buffer(buffer)
    for _ in range(2):
        with pytest.raises(BaseExceptionGroup) as caught:
            resources.close()
        assert caught.value.exceptions == (runtime.error,)
    assert events == ["buffer", "runtime"]
    assert resources.failures[0].owner is runtime
    assert resources.failures[0].kind == "runtime"
    assert resources.runtime_pending is False
    assert runtime.released is (phase == "after")
    assert resources.state == "UNKNOWN_RELEASE"
    assert resources in quarantined_resources()


def test_multiple_original_base_exceptions_and_successes_are_retained():
    events = []
    resources = DeviceResources()
    runtime = FakeOwner("runtime", events)
    a = FakeOwner("a", events, fault="after", error=KeyboardInterrupt("a"))
    b = FakeOwner("b", events)
    c = FakeOwner("c", events, fault="before", error=SystemExit("c"))
    resources.set_runtime(runtime)
    for owner in (a, b, c):
        resources.add_buffer(owner)
    with pytest.raises(BaseExceptionGroup) as caught:
        resources.close()
    assert caught.value.exceptions == (c.error, a.error)
    assert [failure.owner for failure in resources.failures] == [c, a]
    assert events == ["c", "b", "a"] and b.released
    assert runtime.calls == 0
    with pytest.raises(BaseExceptionGroup):
        resources.close()
    assert events == ["c", "b", "a"]


def test_quarantine_retains_unknown_owners_across_ordinary_gc():
    events = []
    resources = DeviceResources()
    runtime = FakeOwner("runtime", events)
    buffer = FakeOwner("buffer", events, fault="after")
    resources.set_runtime(runtime)
    resources.add_buffer(buffer)
    references = tuple(weakref.ref(item) for item in (resources, runtime, buffer))
    try:
        resources.close()
    except BaseExceptionGroup:
        pass
    del resources, runtime, buffer
    gc.collect()
    assert all(reference() is not None for reference in references)
    retained, retained_runtime, retained_buffer = (reference() for reference in references)
    assert retained in quarantined_resources()
    assert retained_runtime.calls == 0 and retained_buffer.calls == 1
    assert events == ["buffer"]


def test_failure_before_entering_ledger_can_retry_pending_owners():
    events = []
    resources = DeviceResources()
    runtime = FakeOwner("runtime", events)
    buffer = FakeOwner("buffer", events)
    resources.set_runtime(runtime)
    resources.add_buffer(buffer)

    def outer_close(fail):
        if fail:
            raise RuntimeError("before actual bank close")
        resources.close()

    with pytest.raises(RuntimeError, match="before actual"):
        outer_close(True)
    assert resources.state == "OPEN" and not events
    outer_close(False)
    outer_close(False)
    assert events == ["buffer", "runtime"]


def test_registration_and_recursive_close_do_not_hide_ownership_errors():
    events = []
    empty = DeviceResources()
    empty.close()
    assert empty.closed
    resources = DeviceResources()
    runtime = FakeOwner("runtime", events)
    buffer = FakeOwner("buffer", events)
    with pytest.raises(RuntimeError, match="runtime"):
        resources.add_buffer(buffer)
    resources.set_runtime(runtime)
    with pytest.raises(ValueError, match="runtime"):
        resources.set_runtime(runtime)
    resources.add_buffer(buffer)
    with pytest.raises(ValueError, match="duplicate"):
        resources.add_buffer(buffer)
    with pytest.raises(ValueError, match="runtime"):
        resources.add_buffer(runtime)
    resources.close()
    with pytest.raises(RuntimeError, match="closed"):
        resources.add_buffer(FakeOwner("late", events))

    recursive = DeviceResources()
    recursive.set_runtime(FakeOwner("runtime2", events))
    owner = FakeOwner("recursive", events)
    owner.free = recursive.close
    recursive.add_buffer(owner)
    with pytest.raises(BaseExceptionGroup) as caught:
        recursive.close()
    assert isinstance(caught.value.exceptions[0], RuntimeError)
    assert "recursive" in str(caught.value.exceptions[0])
    assert recursive.state == "UNKNOWN_RELEASE"


MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
FAMILIES = ("base", "tree", "floating")


def compiled(family):
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    root = '<body pos="0 0 .6">'
    if family == "base":
        joints = '<joint name="j" type="slide" axis="0 0 1"/>'
        bodies = joints + '<geom type="sphere" size=".08" mass="1"/>'
    elif family == "tree":
        bodies = (
            '<joint name="j" type="hinge" damping=".1"/>'
            '<geom type="sphere" pos=".2 0 0" size=".08" mass="1"/>'
        )
    else:
        bodies = (
            '<freejoint/><geom type="sphere" size=".08" mass="1"/>'
            '<body pos=".2 0 0"><joint name="j" type="hinge" damping=".1"/>'
            '<geom type="sphere" pos=".1 0 0" size=".04" mass=".2"/></body>'
        )
    xml = (
        '<mujoco><option timestep=".001" gravity="0 0 0"/>'
        "<worldbody>" + root + bodies + "</body></worldbody>"
        '<actuator><motor joint="j"/></actuator></mujoco>'
    )
    return compile_mujoco_xml(xml, require_asset=False)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("family", FAMILIES)
def test_actual_constructor_early_failure_closes_registered_resources(
    mode, family, monkeypatch, tmp_path
):
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime
    from phymeo.backends.mujoco_tyx._device import TyxDevice

    records = []
    real_set_runtime = DeviceResources.set_runtime
    real_buffer = TyxDevice._buffer
    calls = 0

    def capture(resources, runtime):
        real_set_runtime(resources, runtime)
        records.append((resources, runtime))

    def fail_buffer(bank, value):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise MemoryError("authored allocation entry failure")
        return real_buffer(bank, value)

    monkeypatch.setattr(DeviceResources, "set_runtime", capture)
    monkeypatch.setattr(TyxDevice, "_buffer", fail_buffer)
    with pytest.raises(MemoryError, match="authored allocation"):
        MuJoCoRuntime(compiled(family), mode=mode, world_keys=(7, 91))
    assert calls == 3 and len(records) == 1
    resources, runtime = records[0]
    assert resources.closed and runtime.closed
    assert resources.failures == () and resources.pending_buffer_count == 0
    assert not resources.runtime_pending and resources not in quarantined_resources()
    (tmp_path / "constructor-cleanup.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "family": family,
                "calls": calls,
                "resources_state": resources.state,
                "runtime_closed": runtime.closed,
                "failure": "authored allocation entry failure",
                "no_physics_step": True,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "mode,fault",
    (
        ("interpret", "runtime"),
        pytest.param("cuda", "runtime", marks=pytest.mark.cuda),
        pytest.param("cuda", "buffer", marks=pytest.mark.cuda),
    ),
)
@pytest.mark.parametrize("family", FAMILIES)
def test_actual_release_completed_then_error_is_not_retried(
    mode, fault, family, monkeypatch, tmp_path
):
    from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime

    state = {"armed": False, "completed": False, "calls": 0}
    captured = []
    original_set = DeviceResources.set_runtime
    original_add = DeviceResources.add_buffer
    record = {"mode": mode, "fault": fault, "family": family}
    runtime = None

    def capture(resources, actual_runtime):
        captured.append((resources, actual_runtime))
        if fault == "runtime":
            real_close = actual_runtime.close

            def late_close():
                state["calls"] += 1
                real_close()
                state["completed"] = True
                if state["armed"]:
                    raise RuntimeError("authored post-release runtime failure")

            monkeypatch.setattr(actual_runtime, "close", late_close)
        original_set(resources, actual_runtime)

    def capture_buffer(resources, buffer):
        if fault == "buffer" and not state.get("buffer_selected"):
            state["buffer_selected"] = True
            real_free = buffer.free

            def late_free():
                state["calls"] += 1
                real_free()
                state["completed"] = True
                if state["armed"]:
                    raise RuntimeError("authored post-release buffer failure")

            monkeypatch.setattr(buffer, "free", late_free)
        original_add(resources, buffer)

    monkeypatch.setattr(DeviceResources, "set_runtime", capture)
    monkeypatch.setattr(DeviceResources, "add_buffer", capture_buffer)
    try:
        runtime = MuJoCoRuntime(compiled(family), mode=mode, world_keys=(7, 91))
        runtime.set_controls({7: MuJoCoControl([0.2]), 91: MuJoCoControl([-0.15])})
        runtime.step(0.001)
        record["accepted_world_steps"] = runtime.counters["accepted_world_steps"]
        assert record["accepted_world_steps"] == 2
        assert len(captured) == 1
        resources, actual_runtime = captured[0]
        state["armed"] = True
        for _ in range(2):
            with pytest.raises(BaseExceptionGroup):
                runtime.close()
        assert state["completed"] and state["calls"] == 1
        assert resources.state == "UNKNOWN_RELEASE" and not resources.closed
        assert resources in quarantined_resources()
        assert len(resources.failures) == 1
        assert resources.failures[0].kind == fault
        if fault == "buffer":
            assert resources.runtime_pending and not actual_runtime.closed
            # This test knows its error happened AFTER real free returned. All
            # siblings also completed. Retire only its held runtime explicitly;
            # production cannot infer this phase and must retain UNKNOWN_RELEASE.
            actual_runtime.close()
        assert actual_runtime.closed
        record.update(
            ledger=resources.state,
            actual_runtime_closed=actual_runtime.closed,
            callback_calls=state["calls"],
            real_release_completed=state["completed"],
            quarantine_retained=True,
            automatic_recovery=False,
        )
    except BaseException as error:
        record["failure"] = repr(error)
        raise
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except BaseExceptionGroup:
                pass  # Preserve sticky error; no unsafe release retry in cleanup.
        if captured and state["completed"] and fault == "buffer":
            actual_runtime = captured[0][1]
            if not actual_runtime.closed:
                actual_runtime.close()  # Known authored post-return fault only.
        (tmp_path / "actual-release-cleanup.json").write_text(json.dumps(record), encoding="utf-8")
