"""Pure publication-protocol tests, not a synthetic numerical physics oracle.

AUTHORING_ONLY / UNEXECUTED. Real TYX contact and numeric rejection are covered
in scale/test_resident_env_contacts.py by the parent's serial execution queue.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pytest

from phymeo.training import PhysicsStepFailure, ResidentVectorEnv, VectorEnvStepError

KEYS = (4, 7)


@dataclass(frozen=True)
class LogicalState:
    tick: int


class PublicationBackend:
    """Advance logical publication clocks exactly as its declared result says."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.ticks = dict.fromkeys(KEYS, 0)
        self.controls = {}
        self.step_calls = []

    def snapshot(self, keys=None):
        return {key: LogicalState(self.ticks[key]) for key in KEYS if keys is None or key in keys}

    def set_controls(self, controls):
        self.controls = dict(controls)

    def step(self, h):
        self.step_calls.append(h)
        outcome = self.outcomes.pop(0)
        flags = outcome.accepted_worlds if isinstance(outcome, PhysicsStepFailure) else outcome
        if isinstance(flags, Mapping):
            for key, flag in flags.items():
                if key in self.ticks and isinstance(flag, (bool, np.bool_)) and flag:
                    self.ticks[key] += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def reset(self, keys):
        for key in keys:
            self.ticks[key] = 0


class Task:
    def __init__(self, *, fail_observer=False):
        self.observe_calls = self.evaluate_calls = 0
        self.fail_observer = fail_observer

    def controls(self, actions, keys):
        return {key: actions[i].copy() for i, key in enumerate(keys)}

    def observe(self, states, keys):
        self.observe_calls += 1
        return np.array([[states[key].tick] for key in keys], dtype=float)

    def evaluate(self, before, after, actions, keys):
        self.evaluate_calls += 1
        if self.fail_observer:
            raise RuntimeError("post-publication reward failure")
        return np.zeros(len(keys)), np.zeros(len(keys), dtype=bool)


def environment(outcomes, *, task=None, stepper=None):
    backend, task = PublicationBackend(outcomes), Task() if task is None else task
    env = ResidentVectorEnv(backend, task, KEYS, action_size=1, dt=1 / 64,
                            horizon=10, stepper=stepper)
    return backend, task, env


def assert_no_transition(task, env):
    assert task.evaluate_calls == 0
    assert task.observe_calls == 1
    np.testing.assert_array_equal(env._length, [0, 0])
    np.testing.assert_array_equal(env._episode_id, [0, 0])


@pytest.mark.parametrize("raised", (False, True), ids=("returned", "explicit-failure"))
def test_all_false_is_retryable_without_reward_or_episode_transition(raised):
    flags = dict.fromkeys(KEYS, False)
    outcome = PhysicsStepFailure("known rejection", accepted_worlds=flags) if raised else flags
    backend, task, env = environment([outcome, dict.fromkeys(KEYS, True)])
    with pytest.raises(VectorEnvStepError) as caught:
        env.step(np.zeros((2, 1)))
    error = caught.value
    assert dict(error.accepted_worlds) == flags
    assert error.acceptance_known and not error.physics_accepted and not env.needs_reset
    assert_no_transition(task, env)
    assert backend.ticks == {4: 0, 7: 0}
    with pytest.raises(TypeError):
        error.accepted_worlds[4] = True
    result = env.step(np.zeros((2, 1)))
    assert backend.ticks == {4: 1, 7: 1} and task.evaluate_calls == 1
    np.testing.assert_array_equal(result.episode_length, [1, 1])
    np.testing.assert_array_equal(result.episode_id, [0, 0])


@pytest.mark.parametrize("raised", (False, True), ids=("returned", "explicit-failure"))
def test_partial_publication_is_retained_but_never_emits_an_env_transition(raised):
    flags = {4: True, 7: False}
    outcome = PhysicsStepFailure("partial publication", accepted_worlds=flags) if raised else flags
    backend, task, env = environment([outcome, dict.fromkeys(KEYS, True)])
    with pytest.raises(VectorEnvStepError) as caught:
        env.step(np.zeros((2, 1)))
    assert dict(caught.value.accepted_worlds) == flags
    assert caught.value.physics_accepted and caught.value.acceptance_known and env.needs_reset
    assert backend.ticks == {4: 1, 7: 0}
    assert_no_transition(task, env)
    with pytest.raises(RuntimeError, match="reset all"):
        env.step(np.zeros((2, 1)))
    with pytest.raises(RuntimeError, match="reset all"):
        env.reset([0])
    assert backend.ticks == {4: 1, 7: 0} and len(backend.step_calls) == 1
    env.reset()
    result = env.step(np.zeros((2, 1)))
    np.testing.assert_array_equal(result.episode_id, [1, 1])
    np.testing.assert_array_equal(result.episode_length, [1, 1])


@pytest.mark.parametrize("outcome", (
    None, [(4, True), (7, True)], {4: True}, {4: True, 7: True, 99: False},
    {4: True, 7: 1}, {4: True, 7: "False"},
), ids=("missing-return", "not-a-map", "missing-world", "extra-world", "int-flag", "str-flag"))
def test_invalid_return_poison_is_conservative(outcome):
    _, task, env = environment([outcome])
    with pytest.raises(VectorEnvStepError) as caught:
        env.step(np.zeros((2, 1)))
    assert not caught.value.acceptance_known and caught.value.physics_accepted
    assert dict(caught.value.accepted_worlds) == {} and env.needs_reset
    assert_no_transition(task, env)


@pytest.mark.parametrize("failure", (
    RuntimeError("unknown publication timing"),
    PhysicsStepFailure("unknown backend timing", accepted_worlds={4: False, 7: False},
                       acceptance_known=False),
    PhysicsStepFailure("bad partial report", accepted_worlds={4: True}),
), ids=("unknown-exception", "explicit-unknown", "incomplete-exception-map"))
def test_unknown_or_incomplete_failure_needs_full_reset(failure):
    _, task, env = environment([failure])
    with pytest.raises(VectorEnvStepError) as caught:
        env.step(np.zeros((2, 1)))
    assert caught.value.physics_accepted and not caught.value.acceptance_known
    assert env.needs_reset and caught.value.__cause__ is not None
    assert_no_transition(task, env)


def test_failure_after_all_worlds_accepted_reports_true_mask_without_reward():
    failure = PhysicsStepFailure("late backend failure", accepted_worlds={7: True, 4: True})
    backend, task, env = environment([failure])
    with pytest.raises(VectorEnvStepError) as caught:
        env.step(np.zeros((2, 1)))
    assert backend.ticks == {4: 1, 7: 1}
    assert dict(caught.value.accepted_worlds) == {4: True, 7: True}
    assert caught.value.physics_accepted and caught.value.acceptance_known and env.needs_reset
    assert_no_transition(task, env)


def test_post_accept_observer_failure_keeps_complete_physical_provenance():
    backend, task, env = environment([{4: True, 7: True}], task=Task(fail_observer=True))
    with pytest.raises(VectorEnvStepError) as caught:
        env.step(np.zeros((2, 1)))
    assert backend.ticks == {4: 1, 7: 1} and task.evaluate_calls == 1
    assert dict(caught.value.accepted_worlds) == {4: True, 7: True}
    assert caught.value.physics_accepted and env.needs_reset
    np.testing.assert_array_equal(env._length, [0, 0])
    np.testing.assert_array_equal(env._episode_id, [0, 0])


def test_failure_owns_a_read_only_copy_of_its_boolean_mask():
    flags = {4: np.bool_(False), 7: True}
    failure = PhysicsStepFailure("partial", accepted_worlds=flags)
    flags[4] = True
    assert dict(failure.accepted_worlds) == {4: False, 7: True}
    assert failure.physics_accepted and failure.acceptance_known
    with pytest.raises(TypeError):
        failure.accepted_worlds[4] = True
    with pytest.raises(AttributeError):
        failure.accepted_worlds = {}
    with pytest.raises(TypeError, match="boolean"):
        PhysicsStepFailure("bad flags", accepted_worlds={4: 1})


def test_custom_stepper_is_called_once_and_noncallable_is_rejected():
    backend, task = PublicationBackend([{7: np.bool_(True), 4: np.bool_(True)}]), Task()
    calls = []

    def selected(h):
        calls.append(h)
        return backend.step(h)

    env = ResidentVectorEnv(backend, task, KEYS, action_size=1, dt=1 / 64,
                            horizon=10, stepper=selected)
    result = env.step(np.zeros((2, 1)))
    assert calls == [1 / 64] and backend.step_calls == calls
    np.testing.assert_array_equal(result.episode_length, [1, 1])
    with pytest.raises(TypeError, match="callable"):
        ResidentVectorEnv(backend, task, KEYS, action_size=1, dt=1 / 64,
                          horizon=10, stepper=0)

def test_reset_publication_then_readback_failure_poison_keeps_counters_and_cause():
    cause = ValueError("original reset readback detail")
    failure = RuntimeError("observer failed after physical reset")

    class ResetTask(Task):
        fail_reset_read = False

        def observe(self, states, keys):
            result = super().observe(states, keys)
            if self.fail_reset_read:
                raise failure from cause
            return result

    backend, task, env = environment([dict.fromkeys(KEYS, True), dict.fromkeys(KEYS, True)],
                                     task=ResetTask())
    env.step(np.zeros((2, 1)))
    assert backend.ticks == {4: 1, 7: 1}
    task.fail_reset_read = True
    with pytest.raises(RuntimeError, match="after physical reset") as caught:
        env.reset([0])
    assert caught.value is failure and caught.value.__cause__ is cause
    assert backend.ticks == {4: 0, 7: 1} and env.needs_reset
    np.testing.assert_array_equal(env._length, [1, 1])
    np.testing.assert_array_equal(env._episode_id, [0, 0])
    with pytest.raises(RuntimeError, match="reset all"):
        env.step(np.zeros((2, 1)))
    with pytest.raises(RuntimeError, match="reset all"):
        env.reset([0])
    assert len(backend.step_calls) == 1 and backend.ticks == {4: 0, 7: 1}
    task.fail_reset_read = False
    env.reset()
    assert not env.needs_reset and backend.ticks == {4: 0, 7: 0}
    result = env.step(np.zeros((2, 1)))
    np.testing.assert_array_equal(result.episode_length, [1, 1])
    np.testing.assert_array_equal(result.episode_id, [1, 1])
