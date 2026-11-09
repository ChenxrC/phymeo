"""Policy scheduling tests without substituting policy clocks for physical state."""

from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from phymeo.control.policy_runner import PolicyRunner, PolicyRunnerProtocolError


class Refused(RuntimeError):
    physics_accepted = False
    acceptance_known = True
    accepted_worlds = MappingProxyType({"left": False, "right": False})


class PartialFailure(RuntimeError):
    physics_accepted = True
    acceptance_known = True
    accepted_worlds = MappingProxyType({"left": True, "right": False})


class ToyEnv:
    world_keys = ("left", "right")
    action_size = 1

    def __init__(self):
        self.initial = np.array([[1.0], [10.0]])
        self.observation = self.initial.copy()
        self.episode_id = np.zeros(2, dtype=int)
        self.length = np.zeros(2, dtype=int)
        self.needs_reset = False
        self.reset_calls, self.actions = [], []
        self.failure = None
        self.terminate_at = None
        self.truncate = False
        self.bad_transition = self.bad_reset = False
        self.last_transition = None

    def reset(self, indices=None):
        selected = np.arange(2) if indices is None else np.asarray(indices)
        self.reset_calls.append(selected.copy())
        self.observation[selected] = self.initial[selected]
        self.length[selected] = 0
        self.episode_id[selected] += 1
        if self.bad_reset:
            raise RuntimeError("reset readback failed")
        self.needs_reset = False
        return self.observation.copy()

    def step(self, actions):
        self.actions.append(np.array(actions, copy=True))
        if isinstance(self.failure, PartialFailure):
            self.observation[0] += actions[0]
            self.needs_reset = True
            raise self.failure
        if self.failure is not None:
            raise self.failure
        self.observation += actions
        self.length += 1
        final = self.observation.copy()
        terminated = np.array([len(self.actions) == self.terminate_at, False])
        self.observation[terminated] = self.initial[terminated]
        self.length[terminated] = 0
        self.episode_id += terminated
        self.last_transition = SimpleNamespace(
            observation=self.observation.copy(),
            reward=-(final[:, 0] ** 2),
            terminated=np.zeros(2, dtype=bool) if self.truncate else terminated,
            truncated=terminated if self.truncate else np.zeros(2, dtype=bool),
            final_mask=terminated.copy(),
            final_observation=final,
            episode_id=self.episode_id.copy(),
            episode_length=self.length.copy(),
        )
        if self.bad_transition:
            self.last_transition.final_mask = np.zeros(2, dtype=int)
        return self.last_transition


def plus_one(observation, due_mask):
    return observation + 1


def assert_unchanged(before, after):
    assert np.array_equal(before.observation, after.observation)
    assert np.array_equal(before.physical_ticks, after.physical_ticks)
    assert np.array_equal(before.refresh_mask, after.refresh_mask)
    if before.held_actions is None:
        assert after.held_actions is None
    else:
        assert np.array_equal(before.held_actions, after.held_actions)


def test_construction_never_resets_and_initial_observation_is_owned():
    env = ToyEnv()
    runner = PolicyRunner(env, plus_one)
    assert env.reset_calls == []
    with pytest.raises(RuntimeError, match="initial_observation"):
        runner.step()
    assert env.actions == []
    runner.reset()
    assert len(env.reset_calls) == 1
    first = runner.step()
    assert np.array_equal(first.actions, [[2], [11]])
    initial = env.initial.copy()
    supplied = PolicyRunner(env, plus_one, initial_observation=initial)
    initial[:] = 999
    assert np.array_equal(supplied.snapshot().observation, env.initial)
    assert len(env.reset_calls) == 1


def test_hold_has_one_real_step_per_call_and_no_extra_policy_inference():
    env, calls = ToyEnv(), []

    def policy(observation, due):
        calls.append(due.copy())
        return observation + 1

    runner = PolicyRunner(env, policy, control_decimation=3, initial_observation=env.initial)
    results = [runner.step() for _ in range(6)]
    assert len(env.actions) == 6 and len(calls) == 2
    assert all(np.array_equal(row, [True, True]) for row in calls)
    assert all(np.array_equal(result.actions, [[2], [11]]) for result in results[:3])
    assert all(np.array_equal(result.actions, [[8], [44]]) for result in results[3:])
    assert np.array_equal(env.observation, [[31], [175]])
    assert np.array_equal(runner.snapshot().physical_ticks, [6, 6])


def test_selective_reset_refreshes_one_world_without_rescheduling_the_other():
    env, due_calls = ToyEnv(), []

    def policy(observation, due):
        due_calls.append(due.copy())
        return observation + 1

    runner = PolicyRunner(env, policy, control_decimation=3, initial_observation=env.initial)
    runner.step()
    runner.step()
    before_other = env.observation[1].copy()
    runner.reset(np.array([True, False]))
    assert np.array_equal(env.reset_calls[-1], [0])
    assert np.array_equal(env.observation[1], before_other)
    assert np.array_equal(runner.snapshot().physical_ticks, [0, 2])
    third = runner.step()
    assert np.array_equal(third.due_mask, [True, False])
    assert np.array_equal(third.actions, [[2], [11]])  # Ignore non-due proposed 33.
    fourth = runner.step()
    assert np.array_equal(fourth.due_mask, [False, True])
    assert np.array_equal(fourth.actions, [[2], [44]])
    assert np.array_equal(runner.snapshot().physical_ticks, [2, 4])


@pytest.mark.parametrize("truncated", (False, True))
def test_autoreset_uses_final_mask_and_preserves_terminal_observation(truncated):
    env = ToyEnv()
    env.terminate_at = 2
    env.truncate = truncated
    runner = PolicyRunner(env, plus_one, control_decimation=3, initial_observation=env.initial)
    runner.step()
    terminal = runner.step()
    assert terminal.transition.terminated[0] == (not truncated)
    assert terminal.transition.truncated[0] == truncated
    assert terminal.transition.final_observation[0, 0] == 5
    assert terminal.transition.observation[0, 0] == 1
    assert np.array_equal(runner.snapshot().physical_ticks, [0, 2])
    following = runner.step()
    assert np.array_equal(following.due_mask, [True, False])
    assert np.array_equal(following.actions, [[2], [11]])
    assert env.episode_id[0] == 1 and env.episode_id[1] == 0


@pytest.mark.parametrize(
    "bad",
    (
        np.zeros(2),
        np.array([[0.0], [np.nan]]),
        np.array([[0.0], [np.inf]]),
        np.ones((2, 1), dtype=complex),
        np.ones((2, 1), dtype=bool),
    ),
)
def test_invalid_callback_action_is_rejected_before_any_physics(bad):
    env = ToyEnv()
    runner = PolicyRunner(env, lambda observation, due: bad, initial_observation=env.initial)
    before = runner.snapshot()
    with pytest.raises(ValueError, match="policy actions"):
        runner.step()
    assert env.actions == [] and not runner.needs_reset
    assert_unchanged(before, runner.snapshot())
    runner.policy = plus_one
    assert np.array_equal(runner.step().actions, [[2], [11]])


def test_callback_inputs_actions_and_returned_transition_cannot_alias_mutable_data():
    env = ToyEnv()

    def policy(observation, due):
        with pytest.raises(ValueError):
            observation.flags.writeable = True
        with pytest.raises(ValueError):
            due[:] = False
        return observation + 1

    runner = PolicyRunner(env, policy, initial_observation=env.initial)
    result = runner.step()
    env.last_transition.observation[:] = 999
    assert np.array_equal(result.transition.observation, [[3], [21]])
    for array in (
        result.actions,
        result.due_mask,
        result.physical_ticks,
        result.transition.observation,
        result.transition.final_observation,
        result.transition.reward,
        result.transition.episode_id,
        runner.snapshot().physical_ticks,
        runner.snapshot().held_actions,
    ):
        with pytest.raises(ValueError):
            array.flags.writeable = True


def test_known_atomic_refusal_does_not_commit_due_actions_or_clocks_before_retry():
    env = ToyEnv()
    runner = PolicyRunner(env, plus_one, control_decimation=2, initial_observation=env.initial)
    runner.step()
    runner.step()
    before = runner.snapshot()
    env.failure = Refused("no worlds advanced")
    with pytest.raises(Refused) as caught:
        runner.step()
    assert caught.value is env.failure
    assert not caught.value.physics_accepted and not runner.needs_reset
    assert_unchanged(before, runner.snapshot())
    assert np.array_equal(env.observation, [[5], [32]])
    assert np.array_equal(env.actions[-1], [[6], [33]])
    assert np.array_equal(runner.snapshot().held_actions, [[2], [11]])
    env.failure = None
    retry = runner.step()
    assert np.array_equal(retry.actions, [[6], [33]])
    assert np.array_equal(retry.physical_ticks, [2, 2])


def test_partial_physics_is_not_hidden_and_requires_full_explicit_reset():
    env = ToyEnv()
    runner = PolicyRunner(env, plus_one, initial_observation=env.initial)
    before = runner.snapshot()
    env.failure = PartialFailure("left advanced, right refused")
    with pytest.raises(PartialFailure) as caught:
        runner.step()
    assert caught.value.accepted_worlds == {"left": True, "right": False}
    assert caught.value.physics_accepted and runner.needs_reset
    assert env.observation[0, 0] == 3 and env.observation[1, 0] == 10
    assert_unchanged(before, runner.snapshot())
    with pytest.raises(RuntimeError, match="reset all"):
        runner.step()
    with pytest.raises(RuntimeError, match="reset all"):
        runner.reset([0])
    assert env.reset_calls == []
    env.failure = None
    runner.reset()
    assert not runner.needs_reset
    assert np.array_equal(runner.step().actions, [[2], [11]])


def test_unknown_environment_exception_is_conservatively_reset_required():
    env = ToyEnv()
    runner = PolicyRunner(env, plus_one, initial_observation=env.initial)
    env.failure = RuntimeError("no acceptance contract")
    with pytest.raises(RuntimeError, match="no acceptance"):
        runner.step()
    assert runner.needs_reset and np.array_equal(runner.snapshot().physical_ticks, [0, 0])


def test_malformed_post_physics_transition_cannot_be_published_as_runner_success():
    env = ToyEnv()
    runner = PolicyRunner(env, plus_one, initial_observation=env.initial)
    before = runner.snapshot()
    env.bad_transition = True
    with pytest.raises(PolicyRunnerProtocolError) as caught:
        runner.step()
    assert caught.value.physics_accepted and runner.needs_reset
    assert np.array_equal(env.observation, [[3], [21]])
    assert_unchanged(before, runner.snapshot())
    with pytest.raises(RuntimeError, match="reset all"):
        runner.reset(np.array([False, True]))


def test_failed_reset_does_not_commit_runner_schedule_and_requires_full_reset():
    env = ToyEnv()
    runner = PolicyRunner(env, plus_one, initial_observation=env.initial)
    runner.step()
    before = runner.snapshot()
    env.bad_reset = True
    with pytest.raises(RuntimeError, match="reset readback"):
        runner.reset([0])
    assert_unchanged(before, runner.snapshot())
    assert runner.needs_reset
    env.bad_reset = False
    runner.reset()
    assert not runner.needs_reset
    assert np.array_equal(runner.snapshot().physical_ticks, [0, 0])


@pytest.mark.parametrize("decimation", (0, -1, 1.5, True))
def test_invalid_clock_configuration_does_not_reset_environment(decimation):
    env = ToyEnv()
    with pytest.raises(ValueError, match="control_decimation"):
        PolicyRunner(env, plus_one, control_decimation=decimation)
    assert env.reset_calls == [] and env.actions == []


def test_non_due_callback_rows_still_require_valid_full_batch_actions():
    env = ToyEnv()
    runner = PolicyRunner(env, plus_one, control_decimation=3, initial_observation=env.initial)
    runner.step()
    runner.step()
    runner.reset([0])
    before = runner.snapshot()

    def invalid(observation, due):
        return np.array([[2.0], [np.nan]])

    runner.policy = invalid
    with pytest.raises(ValueError, match="policy actions"):
        runner.step()
    assert len(env.actions) == 2
    assert_unchanged(before, runner.snapshot())


@pytest.mark.parametrize("selected", ([0, 0], [-1], [0.5], [True], [2]))
def test_invalid_reset_selection_is_rejected_before_environment_reset(selected):
    env = ToyEnv()
    runner = PolicyRunner(env, plus_one, initial_observation=env.initial)
    before = runner.snapshot()
    with pytest.raises(ValueError, match="reset"):
        runner.reset(selected)
    assert env.reset_calls == []
    assert_unchanged(before, runner.snapshot())
