"""Policy calls and actual physical work remain distinct across endpoint resets."""

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from phymeo.training import mujoco_tyx_evaluation as evaluation
from phymeo.training.mujoco_tyx_control_window import (
    MuJoCoTyxControlWindowRecord,
    MuJoCoTyxControlWindowSpec,
    MuJoCoTyxSubstepRecord,
)


@dataclass(frozen=True)
class State:
    tick: int


def counters():
    return {
        "owners": {"environment": 1, "runtime": 2, "workspace": 3},
        "frame": {},
        "counts": {
            **dict.fromkeys(evaluation.ENV_COUNTERS, 0),
            "runtime": dict.fromkeys(evaluation.RUNTIME_COUNTERS | evaluation.DEVICE_COUNTERS, 0),
            "task": dict.fromkeys(evaluation.TASK_COUNTERS, 0),
        },
    }


def transition(d, *, explicit=True, reset=True):
    keys = (7, 4096)
    steps = tuple(
        MuJoCoTyxSubstepRecord(
            i, 10 + i, 11 + i, dict.fromkeys(keys, True), {key: State(21 + i) for key in keys}, True
        )
        for i in range(d)
    )
    result = SimpleNamespace(
        sequence=1,
        world_keys=keys,
        accepted_worlds=dict.fromkeys(keys, True),
        states=steps[-1].states,
        output_frame=SimpleNamespace(ticks=(20 + d, 0 if reset else 20 + d), revision=11 + d),
    )
    if explicit:
        result.control_window = MuJoCoTyxControlWindowRecord(
            MuJoCoTyxControlWindowSpec(d).sha256, d, d, d, steps, None, "complete"
        )
    return result


def snapshots(d, reset=True):
    before = counters()
    after = deepcopy(before)
    after["counts"].update(transitions=1, physical_steps=d, resets=int(reset))
    after["counts"]["runtime"].update(completed_steps=d, accepted_world_steps=2 * d)
    return before, after


@pytest.mark.parametrize("d,explicit", ((1, False), (1, True), (2, True), (8, True)))
@pytest.mark.parametrize("reset", (False, True))
def test_actual_substeps_count_once_and_reset_only_at_endpoint(d, explicit, reset):
    before, after = snapshots(d, reset)
    result = evaluation.counter_accounting(
        before, after, (transition(d, explicit=explicit, reset=reset),)
    )
    events = result["receipt_events"]
    assert events["physical_batch_steps"] == d
    assert events["policy_transitions"] == 1
    assert events["accepted_world_steps"] == 2 * d
    assert events["reset_worlds"] == events["reset_calls"] == int(reset)
    if reset:
        assert events["resets"][0]["world_keys"] == (4096,)
    assert before == counters()


@pytest.mark.parametrize(
    "group,field,value",
    (
        ("environment", "physical_steps", 1),
        ("runtime", "completed_steps", 1),
        ("runtime", "accepted_world_steps", 2),
        ("environment", "resets", 4),
    ),
)
def test_policy_count_cannot_be_substituted_for_physical_or_reset_work(group, field, value):
    before, after = snapshots(4)
    target = after["counts"] if group == "environment" else after["counts"][group]
    target[field] = value
    with pytest.raises(ValueError, match="lifecycle counters"):
        evaluation.counter_accounting(before, after, (transition(4),))


@pytest.mark.parametrize(
    "fault", ("failed", "short", "world", "diagnostics", "revision", "tick", "endpoint")
)
def test_partial_or_forged_window_cannot_count_as_complete_work(fault):
    value = transition(2)
    window = value.control_window
    if fault == "failed":
        object.__setattr__(window, "phase", "diagnostic-failure")
    elif fault == "short":
        object.__setattr__(window, "completed", 1)
    elif fault == "world":
        object.__setattr__(window.substeps[0], "accepted_worlds", {7: True})
    elif fault == "diagnostics":
        object.__setattr__(window.substeps[0], "diagnostics_complete", False)
    elif fault == "revision":
        object.__setattr__(window.substeps[1], "entry_revision", 99)
    elif fault == "tick":
        object.__setattr__(window.substeps[0], "states", {7: State(20), 4096: State(21)})
    else:
        value.states = {7: State(99), 4096: State(22)}
    with pytest.raises(ValueError, match="accounting"):
        evaluation.counter_accounting(*snapshots(2), (value,))


def test_update_capture_keeps_same_partial_window_without_replaying():
    failed = RuntimeError("second substep failed after first accepted")
    failed.control_window = transition(2).control_window
    failed.action_accepted = True

    class Action:
        def detach(self):
            return self

        def clone(self):
            return self

    class Adapter:
        calls = 0

        def step(self, action):
            self.calls += 1
            raise failed

    adapter = Adapter()
    with evaluation.capture_adapter_steps(adapter) as attempts:
        with pytest.raises(RuntimeError) as caught:
            adapter.step(Action())
    assert caught.value is failed and adapter.calls == 1
    assert attempts[0]["control_window"] is failed.control_window
    wrapped = evaluation.UpdateCaptureError(failed, attempts, 0.1)
    assert wrapped.control_window is failed.control_window
    assert wrapped.attempts[0]["action_accepted"] is True
    assert "step" not in vars(adapter)
