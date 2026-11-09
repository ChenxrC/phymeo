"""Optional RSL-RL 5.5.2 VecEnv boundary with explicit HOST tensor copies.

Import this module explicitly after installing optional training dependencies.
It is intentionally absent from phymeo.training's base exports. Physics and
resource ownership remain in ResidentVectorEnv and its borrowed backend.
"""

from importlib.metadata import version

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .resident_env import ResidentVectorEnv, VectorEnvStepError

SUPPORTED_RSL_RL = "5.5.2"


class RslRlVecEnv(VecEnv):
    """Fresh TensorDict observations, copied actions, and original failure masks.

    One 'policy' observation group is available to actor and critic. This is a
    synchronous NumPy/HOST bridge, even if tensors are explicitly moved to CUDA.
    Randomizing episode_length_buf without resetting physical worlds is rejected.
    """

    def __init__(self, env, *, device="cpu"):
        if not isinstance(env, ResidentVectorEnv):
            raise TypeError("RslRlVecEnv borrows a ResidentVectorEnv")
        installed = version("rsl-rl-lib")
        if installed != SUPPORTED_RSL_RL:
            raise RuntimeError(
                f"this adapter requires rsl-rl-lib=={SUPPORTED_RSL_RL}, got {installed}"
            )
        self.env, self.device = env, torch.device(device)
        self.num_envs, self.num_actions = len(env.world_keys), env.action_size
        # Same-package read-only metadata; these are never written into the env.
        horizons = np.array(env._horizon, dtype=np.int64, copy=True)
        self.max_episode_length = (
            int(horizons[0])
            if np.all(horizons == horizons[0])
            else self._tensor(horizons, torch.long)
        )
        self._episode_length = self._tensor(env._length, torch.long)
        self._obs_width = None
        self._poisoned = False
        self.cfg = {
            "adapter": "phymeo-rsl-rl-v1",
            "rsl_rl": installed,
            "physical_backend": type(env.batch).__name__,
            "host_tensor_copy": True,
            "world_keys": list(env.world_keys),
            "physical_dt_s": env.dt,
        }
        self.get_observations()

    @property
    def needs_reset(self):
        return self._poisoned or self.env.needs_reset

    @property
    def episode_length_buf(self):
        return self._episode_length.clone()

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        raise NotImplementedError(
            "use init_at_random_ep_len=False; clock-only randomization would not reset physics"
        )

    def _tensor(self, value, dtype):
        owned = np.array(value, copy=True)
        result = torch.tensor(owned, dtype=dtype, device=self.device)
        if result.is_floating_point() and not bool(torch.isfinite(result).all()):
            raise ValueError("observation/reward exceeds the finite tensor dtype range")
        return result

    def _observations(self, value):
        array = np.asarray(value)
        if array.ndim != 2 or array.shape[0] != self.num_envs or not np.isfinite(array).all():
            raise ValueError("RSL observations must be finite [world, feature] matrices")
        if self._obs_width is None:
            self._obs_width = array.shape[1]
        elif array.shape[1] != self._obs_width:
            raise ValueError("RSL observation feature width changed")
        return TensorDict(
            {"policy": self._tensor(array, torch.float32)},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def get_observations(self):
        if self.needs_reset:
            raise RuntimeError("reset all worlds before reading policy observations")
        states = self.env.batch.snapshot(self.env.world_keys)
        return self._observations(self.env.task.observe(states, self.env.world_keys))

    def reset(self, indices=None):
        if self._poisoned and indices is not None:
            if set(indices) != set(range(self.num_envs)):
                raise RuntimeError("reset all worlds after a post-physics tensor failure")
        observations = self.env.reset(indices)
        try:
            result = self._observations(observations)
            lengths = self._tensor(self.env._length, torch.long)
        except Exception:
            self._poisoned = True
            raise
        self._episode_length, self._poisoned = lengths, False
        return result

    def step(self, actions):
        if self.needs_reset:
            raise RuntimeError("reset all worlds before continuing")
        if not isinstance(actions, torch.Tensor) or not actions.is_floating_point():
            raise TypeError("RSL actions must be a floating-point Torch tensor")
        if tuple(actions.shape) != (self.num_envs, self.num_actions) or not bool(
            torch.isfinite(actions).all()
        ):
            raise ValueError("RSL actions must be finite [world, action] tensors")
        # Explicit device->HOST transfer followed by a detached owned NumPy copy.
        action = actions.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
        # Preserve the original immutable masks, cause and recovery requirement.
        transition = self.env.step(action)
        try:
            observation = self._observations(transition.observation)
            reward = self._tensor(transition.reward, torch.float32)
            terminated = self._tensor(transition.terminated, torch.bool)
            truncated = self._tensor(transition.truncated, torch.bool)
            lengths = self._tensor(transition.episode_length, torch.long)
            extras = {
                "time_outs": truncated,
                "terminated": terminated,
                "truncated": truncated.clone(),
                "final_observation": self._observations(transition.final_observation),
                "final_mask": self._tensor(transition.final_mask, torch.bool),
                "episode_id": self._tensor(transition.episode_id, torch.long),
                "episode_length": lengths.clone(),
            }
        except Exception as error:
            self._poisoned = True
            failure = VectorEnvStepError(
                "physics/env transition accepted; tensor conversion failed; reset all worlds",
                accepted_worlds=dict.fromkeys(self.env.world_keys, True),
            )
            failure.transition = transition
            raise failure from error
        self._episode_length = lengths
        return observation, reward, terminated | truncated, extras
