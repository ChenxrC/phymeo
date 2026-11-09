"""Actual resident force control through the common vector-env/policy boundary."""

import json
from dataclasses import fields

import numpy as np
import pytest

import phymeo as ph
from phymeo.control.policy_runner import PolicyRunner
from phymeo.training import ResidentVectorEnv, VectorEnvStepError

KEYS = (11, 29)
H = 1 / 8


class ForceTask:
    def controls(self, actions, keys):
        return dict(zip(keys, actions, strict=True))

    def observe(self, states, keys):
        return np.array([(*states[key].position, *states[key].velocity) for key in keys])

    def evaluate(self, before, after, actions, keys):
        positions = np.array([after[key].position for key in keys])
        return -np.sum(positions**2, axis=1), positions[:, 0] > 1 / 16


def policy(observation, due_mask):
    # Direct forces: independent of any reference q/v injection.
    return np.array([[2.0, 0, 0], [0, -1 - observation[1, 1], 0]])


def batch_worlds():
    return [(key, ph.FreeSphere(1, 0.05, gravity=(0, 0, 0))) for key in KEYS]


def record_step(result):
    return {
        "actual_actions": result.actions.tolist(),
        "due_mask": result.due_mask.tolist(),
        "entry_physical_ticks": result.physical_ticks.tolist(),
        "transition": {
            field.name: getattr(result.transition, field.name).tolist()
            for field in fields(result.transition)
        },
    }


def test_two_world_physics_holding_autoreset_and_independent_policy_clock(tyx_mode, tmp_path):
    with ph.TyxWorldBatch(batch_worlds(), tyx_mode, cache_root=tmp_path) as batch:
        env = ResidentVectorEnv(batch, ForceTask(), KEYS, action_size=3, dt=H, horizon=20)
        runner = PolicyRunner(env, policy, control_decimation=3)
        runner.reset()
        results = [runner.step() for _ in range(4)]
        first, terminal, refreshed, scheduled = results
        assert np.array_equal(first.actions, [[2, 0, 0], [0, -1, 0]])
        assert first.transition.final_observation[0, 0] == 1 / 32
        assert first.transition.final_observation[0, 3] == 1 / 4
        assert terminal.transition.terminated[0]
        assert terminal.transition.final_observation[0, 0] == 3 / 32
        assert terminal.transition.observation[0, 0] == 0
        assert np.array_equal(refreshed.due_mask, [True, False])
        assert np.array_equal(refreshed.physical_ticks, [0, 2])
        assert refreshed.actions[1, 1] == -1
        assert np.array_equal(scheduled.due_mask, [False, True])
        assert scheduled.actions[1, 1] == -29 / 32
        actual = batch.snapshot()
        # Independent semi-implicit force/mass calculation, all binary rationals:
        # v_y(3)=-3/8; F_y(4)=-1-(-3/32)=-29/32.
        assert actual[KEYS[1]].velocity[1] == -125 / 256
        assert actual[KEYS[1]].position[1] == -317 / 2048
        assert actual[KEYS[1]].tick == 4
        assert actual[KEYS[0]].tick == 0  # World zero completed again on step four.
        assert np.array_equal(runner.snapshot().physical_ticks, [0, 4])
        (tmp_path / "policy-resident-two-world.json").write_text(
            json.dumps(
                {
                    "schema": "policy-resident-two-world-v1",
                    "mode": tyx_mode,
                    "dt_s": H,
                    "control_decimation": 3,
                    "world_keys": KEYS,
                    "steps": [record_step(result) for result in results],
                    "final_position": [actual[key].position for key in KEYS],
                    "final_velocity": [actual[key].velocity for key in KEYS],
                    "final_physics_ticks": [actual[key].tick for key in KEYS],
                },
                allow_nan=False,
            ),
            encoding="utf-8",
        )


def test_real_post_accept_observer_failure_cannot_advance_runner_or_hide_motion(tyx_mode, tmp_path):
    class BrokenTask(ForceTask):
        fail = True

        def evaluate(self, *args):
            if self.fail:
                raise RuntimeError("diagnostic reward failed after accepted force step")
            return super().evaluate(*args)

    with ph.TyxWorldBatch(batch_worlds(), tyx_mode, cache_root=tmp_path) as batch:
        task = BrokenTask()
        env = ResidentVectorEnv(batch, task, KEYS, action_size=3, dt=H, horizon=20)
        runner = PolicyRunner(
            env, policy, control_decimation=3, initial_observation=np.zeros((2, 6))
        )
        with pytest.raises(VectorEnvStepError) as caught:
            runner.step()
        assert caught.value.physics_accepted and env.needs_reset and runner.needs_reset
        advanced = batch.snapshot()
        assert advanced[KEYS[0]].position[0] == 1 / 32
        assert advanced[KEYS[1]].position[1] == -1 / 64
        assert advanced[KEYS[0]].tick == advanced[KEYS[1]].tick == 1
        assert np.array_equal(runner.snapshot().physical_ticks, [0, 0])
        assert runner.snapshot().held_actions is None
        with pytest.raises(RuntimeError, match="reset all"):
            runner.step()
        with pytest.raises(RuntimeError, match="reset all"):
            runner.reset([0])
        task.fail = False
        runner.reset()
        reset = batch.snapshot()
        assert all(state.tick == 0 for state in reset.values())
        assert np.array_equal(runner.snapshot().physical_ticks, [0, 0])
        retried = runner.step()
        assert np.array_equal(retried.due_mask, [True, True])
        assert retried.transition.final_observation[0, 0] == 1 / 32
