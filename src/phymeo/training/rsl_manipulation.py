"""Feed-forward RSL PPO over accepted CUDA manipulation transitions.

Policy actions are raw Gaussian coordinates. The environment applies tanh and
the physical scale/bias once. Each training row carries its own effective
action mask, including after minibatch shuffling.
"""

from importlib.metadata import version
from math import isfinite

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from torch.distributions import Normal

from .manipulation_env import HeterogeneousManipulationEnv
from .rsl_mujoco_tyx import SUPPORTED_RSL_RL, ppo_config


class MaskedGaussianActor(MLPModel):
    """RSL Gaussian actor restricted to the controllable coordinates of each row."""

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None or hidden_state is not None:
            raise ValueError("manipulation actor requires feed-forward observations")
        self._valid = obs["action_mask"]
        output = super().forward(obs, stochastic_output=stochastic_output)
        return torch.where(self._valid, output, 0.0)

    @property
    def output_distribution_params(self):
        # Canonical inactive Gaussians give exactly zero KL(old || new).
        return (
            torch.where(self._valid, self.output_mean, 0.0),
            torch.where(self._valid, self.output_std, 1.0),
        )

    def get_output_log_prob(self, outputs):
        dist = Normal(*self.output_distribution_params)
        scores = dist.log_prob(torch.where(self._valid, outputs, 0.0))
        return torch.where(self._valid, scores, 0.0).sum(-1)

    @property
    def output_entropy(self):
        entropy = Normal(*self.output_distribution_params).entropy()
        return torch.where(self._valid, entropy, 0.0).sum(-1)

    def as_jit(self):
        raise NotImplementedError("export requires a mask-aware actor signature")

    def as_onnx(self):
        raise NotImplementedError("export requires a mask-aware actor signature")


class ManipulationPpo:
    """One shared actor/critic, one upstream storage and optimizer for a borrowed Env.

    An exception during collection or update invalidates this training session.
    The environment keeps its own recovery protocol and physical commit history.
    """

    def __init__(self, env, *, steps=24):
        if not isinstance(env, HeterogeneousManipulationEnv) or not env.auto_reset:
            raise ValueError("requires an auto-reset heterogeneous manipulation Env")
        if version("rsl-rl-lib") != SUPPORTED_RSL_RL:
            raise RuntimeError("requires rsl-rl-lib==" + SUPPORTED_RSL_RL)
        if "area_policy_input" not in env.binding:
            raise ValueError("requires v2 area-object contact features")
        self.env = env
        self._mask = env.controllable_action_mask
        if not self._mask.any(dim=1).all().item():
            raise ValueError("each world must have a controllable action")
        self._frame = env.current_frame
        self._observation = self.pack(env.observation)
        cfg = ppo_config(steps=steps)
        cfg["actor"]["class_name"] = MaskedGaussianActor
        cfg["obs_groups"] = dict.fromkeys(
            ("actor", "critic"), ["state", "area_contact", "action_mask"]
        )
        self.algorithm = PPO.construct_algorithm(self._observation, env, cfg, str(env.device))
        self._steps = steps
        self._valid = True

    def pack(self, values):
        """Pack endpoint state and previous accepted contact forces in frozen world order."""
        parts = []
        for name in ("q", "v", "act"):
            mask = values[name + "_mask"].bool()
            parts.extend((torch.where(mask, values[name], 0.0), mask.to(torch.float32)))
        return TensorDict(
            {
                "state": torch.cat(parts, dim=1),
                "area_contact": values["area_object_contact"].flatten(1),
                "action_mask": self._mask.clone(),
            },
            batch_size=[self.env.num_envs],
            device=self.env.device,
        )

    def _check(self):
        if not self._valid or not self.env.is_current_frame(self._frame):
            raise RuntimeError(
                "training session is stale or interrupted; discard the pending rollout"
            )

    @torch.no_grad()
    def collect_step(self):
        """Store action-entry observations and credit a timeout at its pre-reset endpoint."""
        self._check()
        ppo = self.algorithm
        if ppo.storage.step >= self._steps:
            raise RuntimeError("rollout is full")
        self._valid = False
        actions = ppo.act(self._observation)
        tr = self.env.step(actions, world_keys=self.env.world_keys, expected_frame=self._frame)
        following = self.pack(tr.observation)
        # RSL's default time_outs bonus uses V(entry). Supply the actual terminal
        # V once; a done still cuts GAE across the reset, including on truncation.
        terminal_value = ppo.critic(self.pack(tr.final_observation)).squeeze(-1)
        reward = tr.reward + ppo.gamma * torch.where(tr.truncated, terminal_value, 0.0)
        ppo.process_env_step(following, reward, tr.final_mask, {})
        self._frame, self._observation = tr.output_frame, following
        self._valid = True
        return tr

    def update(self):
        """Compute GAE and perform the configured upstream PPO minibatch update."""
        self._check()
        if self.algorithm.storage.step != self._steps:
            raise RuntimeError("a complete rollout is required before update")
        self._valid = False
        with torch.no_grad():
            self.algorithm.compute_returns(self._observation)
        losses = self.algorithm.update()
        optimizer = self.algorithm.optimizer
        # Adam may commit a nonfinite value without raising. A failed check
        # leaves this session unusable; it cannot undo that optimizer step.
        tensors = [param for group in optimizer.param_groups for param in group["params"]]
        tensors.extend(
            value
            for state in optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value)
            and value.is_floating_point()
            and value.device == self.env.device
        )
        if (
            not all(isfinite(v) for v in losses.values())
            or not torch.stack([torch.isfinite(t.detach()).all() for t in tensors]).all().item()
        ):
            raise FloatingPointError("nonfinite PPO update; discard the learner")
        self._valid = True
        return losses
