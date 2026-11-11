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


@dataclass(frozen=True)
class ActionReceipt:
    source_revision: int
    revision: int


def counters():
    return {
        "owners": {"environment": 1, "runtime": 2, "workspace": 3},
        "frame": {"revision": 9, "epoch": 4, "ticks": [20, 20], "times": [0.02, 0.02]},
        "counts": {
            **dict.fromkeys(evaluation.ENV_COUNTERS, 0),
            "runtime": dict.fromkeys(evaluation.RUNTIME_COUNTERS | evaluation.DEVICE_COUNTERS, 0),
            "task": dict.fromkeys(evaluation.TASK_COUNTERS, 0),
        },
    }


def transition(d, *, explicit=True, reset=True, revision=9, tick=20, epoch=4, sequence=1):
    keys = (7, 4096)
    steps = tuple(
        MuJoCoTyxSubstepRecord(
            i,
            revision + 1 + i,
            revision + 2 + i,
            dict.fromkeys(keys, True),
            {key: State(tick + 1 + i) for key in keys},
            True,
        )
        for i in range(d)
    )
    result = SimpleNamespace(
        sequence=sequence,
        world_keys=keys,
        input_frame=SimpleNamespace(
            revision=revision, epoch=epoch, ticks=(tick, tick), times=(tick / 1000, tick / 1000)
        ),
        action_receipt=ActionReceipt(source_revision=revision, revision=revision + 1),
        accepted_worlds=dict.fromkeys(keys, True),
        states=steps[-1].states,
        output_frame=SimpleNamespace(
            ticks=(tick + d, 0 if reset else tick + d),
            times=((tick + d) / 1000, 0 if reset else (tick + d) / 1000),
            revision=revision + 1 + d + int(reset),
            epoch=epoch + 1 + int(reset),
        ),
    )
    if explicit:
        result.control_window = MuJoCoTyxControlWindowRecord(
            MuJoCoTyxControlWindowSpec(d).sha256, d, d, d, steps, result.action_receipt, "complete"
        )
    return result


def snapshots(d, reset=True):
    before = counters()
    after = deepcopy(before)
    after["counts"].update(transitions=1, physical_steps=d, resets=int(reset))
    after["counts"]["runtime"].update(completed_steps=d, accepted_world_steps=2 * d)
    after["frame"] = {
        "revision": 10 + d + int(reset),
        "epoch": 5 + int(reset),
        "ticks": [20 + d, 0 if reset else 20 + d],
        "times": [(20 + d) / 1000, 0 if reset else (20 + d) / 1000],
    }
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


@pytest.mark.parametrize(
    "fault",
    (
        "shift-revisions",
        "shift-ticks",
        "window-action",
        "transition-action",
        "input-frame",
        "output-revision",
        "before-frame",
        "after-frame",
    ),
)
def test_complete_window_must_bind_external_publications(fault):
    value = transition(2)
    before, after = snapshots(2)
    if fault == "shift-revisions":
        for sub in value.control_window.substeps:
            object.__setattr__(sub, "entry_revision", sub.entry_revision + 100)
            object.__setattr__(sub, "exit_revision", sub.exit_revision + 100)
    elif fault == "shift-ticks":
        for sub in value.control_window.substeps:
            object.__setattr__(
                sub, "states", {key: State(state.tick + 100) for key, state in sub.states.items()}
            )
        value.states = value.control_window.substeps[-1].states
        value.output_frame.ticks = (122, 0)
        after["frame"]["ticks"] = [122, 0]
    elif fault == "window-action":
        object.__setattr__(
            value.control_window, "action_receipt", ActionReceipt(source_revision=8, revision=9)
        )
    elif fault == "transition-action":
        value.action_receipt = ActionReceipt(source_revision=8, revision=10)
    elif fault == "input-frame":
        value.input_frame.revision += 1
    elif fault == "output-revision":
        value.output_frame.revision += 1
        after["frame"]["revision"] += 1
    elif fault == "before-frame":
        before["frame"]["epoch"] += 1
    else:
        after["frame"]["epoch"] += 1
    with pytest.raises(ValueError, match="accounting"):
        evaluation.counter_accounting(before, after, (value,))


@pytest.mark.parametrize("swap", (False, True))
def test_adjacent_policy_windows_preserve_publication_order(swap):
    first = transition(2, reset=False)
    second = transition(2, reset=False, revision=12, tick=22, epoch=5, sequence=2)
    before, after = snapshots(2, reset=False)
    after["counts"].update(transitions=2, physical_steps=4)
    after["counts"]["runtime"].update(completed_steps=4, accepted_world_steps=8)
    after["frame"] = {"revision": 15, "epoch": 6, "ticks": [24, 24], "times": [0.024, 0.024]}
    if swap:
        with pytest.raises(ValueError, match="accounting"):
            evaluation.counter_accounting(before, after, (second, first))
    else:
        result = evaluation.counter_accounting(before, after, (first, second))
        assert result["receipt_events"]["physical_batch_steps"] == 4
