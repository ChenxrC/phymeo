"""Feed-forward RSL PPO over accepted CUDA manipulation transitions.

Policy actions are raw Gaussian coordinates. The environment applies tanh and
the physical scale/bias once. Each training row carries its own effective
action mask, including after minibatch shuffling.
"""

from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from math import isfinite
from re import fullmatch
from threading import RLock

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from torch.distributions import Normal

from .manipulation_env import HeterogeneousManipulationEnv
from .manipulation_ppo_checkpoint import read_checkpoint, write_checkpoint
from .mujoco_tyx_checkpoint import (
    MuJoCoTyxTrainingRestoreError,
    RslPpoBoundaryState,
    checkpoint_platform,
)
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


class ManipulationPpo(RslPpoBoundaryState):
    """One shared actor/critic, one upstream storage and optimizer for a borrowed Env.

    An exception during collection or update invalidates this training session.
    The environment keeps its own recovery protocol and physical commit history.
    """

    def __init__(self, env, *, steps=24):
        if not isinstance(env, HeterogeneousManipulationEnv) or not env.auto_reset:
            raise ValueError("requires an auto-reset heterogeneous manipulation Env")
        if version("rsl-rl-lib") != SUPPORTED_RSL_RL:
            raise RuntimeError("requires rsl-rl-lib==" + SUPPORTED_RSL_RL)
        if "area_policy_input" in env.binding:
            self._contact_group, self._contact_value = "area_contact", "area_object_contact"
        elif "native_policy_input" in env.binding:
            self._contact_group, self._contact_value = "native_contact", "native_object_contact"
        else:
            raise ValueError("requires declared area or native object contact features")
        self.env = env
        self._mask = env.controllable_action_mask
        if not self._mask.any(dim=1).all().item():
            raise ValueError("each world must have a controllable action")
        self._frame = env.current_frame
        self._observation = self.pack(env.observation)
        cfg = ppo_config(steps=steps)
        cfg["actor"]["class_name"] = f"{__name__}.MaskedGaussianActor"
        cfg["obs_groups"] = dict.fromkeys(("actor", "critic"), list(self._observation.keys()))
        self._config = cfg
        self._lock = RLock()
        self._generators = {}
        self.recurrent = False
        self._completed = 0
        self.algorithm = self._new_algorithm()
        self._steps = steps
        self._valid = True

    def _new_algorithm(self):
        return PPO.construct_algorithm(
            self._observation, self.env, deepcopy(self._config), str(self.env.device)
        )

    @property
    def completed_updates(self):
        return self._completed

    def pack(self, values):
        """Pack endpoint state and previous accepted contact forces in frozen world order."""
        parts = []
        for name in ("q", "v", "act"):
            mask = values[name + "_mask"].bool()
            parts.extend((torch.where(mask, values[name], 0.0), mask.to(torch.float32)))
        features = {
            "state": torch.cat(parts, dim=1),
            self._contact_group: values[self._contact_value].flatten(1),
            "action_mask": self._mask.clone(),
        }
        if "instruction" in values:
            valid = values["visible_target_mask"].bool()
            features.update(
                instruction=values["instruction"],
                visible_targets=torch.where(
                    valid.unsqueeze(-1), values["visible_targets"], 0.0
                ).flatten(1),
                visible_target_mask=valid,
            )
        return TensorDict(features, batch_size=[self.env.num_envs], device=self.env.device)

    def _check(self):
        if not self._valid or not self.env.is_current_frame(self._frame):
            raise RuntimeError(
                "training session is stale or interrupted; discard the pending rollout"
            )

    @torch.no_grad()
    def collect_step(self):
        """Store action-entry observations and credit a timeout at its pre-reset endpoint."""
        with self._lock:
            return self._collect_step()

    def _collect_step(self):
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
        with self._lock:
            return self._update()

    def _update(self):
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
        self._completed += 1
        self._valid = True
        return losses

    def _checkpoint_binding(self):
        return {
            "config": self._config,
            "timeout_value": "pre-reset-terminal-v1",
            "platform": checkpoint_platform(self.env.device),
        }

    @staticmethod
    def _observation_hashes(observation):
        return {
            name: sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in observation.items()
        }

    def save_checkpoint(self, path):
        """Save a new file after a complete update. The caller owns the global RNGs."""
        with self._lock:
            self._check()
            self._empty()
            if not 1 <= self._completed <= 2**24:
                raise RuntimeError("checkpoint requires a completed PPO update")
            environment = self.env.checkpoint()
            self._check()
            algo = self.algorithm
            state = {
                "binding": self._checkpoint_binding(),
                "environment": environment.sha256,
                "updates": self._completed,
                "learner": algo.save(),
                "modes": {
                    key: {name: mod.training for name, mod in network.named_modules()}
                    for key, network in (("actor", algo.actor), ("critic", algo.critic))
                },
                "hidden": self._hidden(algo),
                "rng": self._capture_rng(),
                "observation": self._observation_hashes(self._observation),
            }
            write_checkpoint(path, environment, state)

    def restore_checkpoint(self, path, *, expected_revision):
        """Resume a saved update boundary in an exclusively borrowed, compatible environment."""
        with self._lock:
            self._check()
            self._empty()
            env = self.env
            revision = env.batch.checkpoint_revision
            if type(expected_revision) is not int or expected_revision != revision:
                raise ValueError("stale PPO restore revision")
            environment, state = read_checkpoint(path)
            fields = {
                "binding",
                "environment",
                "updates",
                "learner",
                "modes",
                "hidden",
                "rng",
                "observation",
            }
            if (
                type(state) is not dict
                or set(state) != fields
                or state["binding"] != self._checkpoint_binding()
            ):
                raise ValueError("PPO training configuration or checkpoint fields differ")
            if environment.sha256 != state["environment"]:
                raise ValueError("PPO environment binding differs")
            hashes = state["observation"]
            if (
                type(hashes) is not dict
                or set(hashes) != set(self._observation.keys())
                or not all(
                    type(value) is str and fullmatch("[0-9a-f]{64}", value)
                    for value in hashes.values()
                )
            ):
                raise ValueError("invalid PPO observation digest layout")
            updates = state["updates"]
            if type(updates) is not int or not 1 <= updates <= 2**24:
                raise ValueError("invalid completed update count")
            self._validate_rng(state["rng"])
            before_rng = self._capture_rng()
            try:
                candidate = self._new_algorithm()
                self._validate_learner(
                    candidate, state["learner"], state["hidden"], state["modes"], updates
                )
            finally:
                self._apply_rng(before_rng)

            try:
                receipt = env.restore(environment, expected_revision=revision)
                observation = self.pack(env.observation)
                if self._observation_hashes(observation) != state["observation"]:
                    raise ValueError("restored PPO observation differs from the saved boundary")
                frame = env.current_frame
                self._install_rng(state["rng"])
            except BaseException as primary:
                if env.batch.checkpoint_revision != revision:
                    self._valid = False
                    try:
                        self._apply_rng(before_rng)
                    except BaseException as rng_error:
                        primary.add_note(
                            f"Restoring the previous RNG state also failed: {rng_error!r}"
                        )
                    raise MuJoCoTyxTrainingRestoreError(
                        "physics restored; discard the interrupted manipulation learner",
                        physics_restored=True,
                    ) from primary
                raise
            self.algorithm = candidate
            self._frame, self._observation = frame, observation
            self._completed = updates
            return receipt
