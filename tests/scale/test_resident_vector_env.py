"""Real resident physics, final observations and independent episode reset."""

import numpy as np
import pytest

import phymeo as ph
from phymeo.training import ResidentVectorEnv, VectorEnvStepError


class SphereTask:
    def controls(self, actions, keys):
        return dict(zip(keys, actions, strict=True))

    def observe(self, states, keys):
        return np.array([(*states[key].position, *states[key].velocity) for key in keys])

    def evaluate(self, before, after, actions, keys):
        x = np.array([after[key].position[0] for key in keys])
        return -(x**2), x > 1 / 128


def test_unlisted_world_cannot_advance_as_an_invisible_environment(tyx_mode, tmp_path):
    worlds = [(key, ph.FreeSphere(1, 0.05)) for key in (4, 7)]
    with ph.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path) as batch:
        with pytest.raises(ValueError, match="complete borrowed batch"):
            ResidentVectorEnv(batch, SphereTask(), (4,), action_size=3, dt=1 / 8, horizon=3)
        assert batch.snapshot()[7].tick == 0


@pytest.mark.parametrize("worlds", (1, 32))
def test_terminal_observation_and_selective_autoreset(worlds, tyx_mode, tmp_path):
    keys = tuple(range(10, 10 + worlds))
    with ph.TyxWorldBatch(
        [(key, ph.FreeSphere(1, 0.05, gravity=(0, 0, 0))) for key in keys],
        tyx_mode,
        cache_root=tmp_path,
    ) as batch:
        env = ResidentVectorEnv(batch, SphereTask(), keys, action_size=3, dt=1 / 8, horizon=3)
        actions = np.zeros((worlds, 3))
        actions[0, 0] = 1
        result = env.step(actions)
        assert result.terminated[0] and not result.truncated.any()
        assert result.final_observation[0, 0] == 1 / 64
        assert result.observation[0, 0] == 0
        assert result.episode_id[0] == 1 and not result.episode_id[1:].any()
        assert result.episode_length[0] == 0
        assert np.all(result.episode_length[1:] == 1)
        with pytest.raises(ValueError):
            result.observation.flags.writeable = True
        for _ in range(2):
            result = env.step(np.zeros((worlds, 3)))
        assert np.all(result.truncated[1:])
        assert not result.terminated.any()


def test_invalid_action_does_not_advance_physics(tyx_mode, tmp_path):
    with ph.TyxWorldBatch(
        [(4, ph.FreeSphere(1, 0.05, gravity=(0, 0, 0)))], tyx_mode, cache_root=tmp_path
    ) as batch:
        env = ResidentVectorEnv(batch, SphereTask(), (4,), action_size=3, dt=1 / 8, horizon=3)
        before = batch.snapshot()[4]
        with pytest.raises(ValueError, match="actions"):
            env.step(np.array([[np.nan, 0, 0]]))
        assert batch.snapshot()[4] == before


def test_post_accept_task_failure_requires_full_reset(tyx_mode, tmp_path):
    class BrokenTask(SphereTask):
        def evaluate(self, *args):
            raise RuntimeError("reward observer failed")

    with ph.TyxWorldBatch(
        [(4, ph.FreeSphere(1, 0.05, gravity=(0, 0, 0)))], tyx_mode, cache_root=tmp_path
    ) as batch:
        env = ResidentVectorEnv(batch, BrokenTask(), (4,), action_size=3, dt=1 / 8, horizon=3)
        with pytest.raises(VectorEnvStepError) as error:
            env.step(np.zeros((1, 3)))
        assert error.value.physics_accepted and env.needs_reset
        assert batch.snapshot()[4].tick == 1
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros((1, 3)))
        env.reset()
        assert not env.needs_reset and batch.snapshot()[4].tick == 0
