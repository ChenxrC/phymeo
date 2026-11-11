"""Thin RSL5.5.2 boundary for the independent MuJoCo-TYX device VecEnv.

Import explicitly with the optional CUDA training profile. No legacy HOST
VecEnv, PPO implementation, learner restore or automatic recovery is wrapped.
"""

from importlib.metadata import version

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .mujoco_tyx_env import MuJoCoTyxEnvError, MuJoCoTyxVectorEnv

SUPPORTED_RSL_RL = "5.5.2"


class RslMuJoCoTyxVecEnv(VecEnv):
    def __init__(self, env):
        if not isinstance(env, MuJoCoTyxVectorEnv):
            raise TypeError("requires MuJoCoTyxVectorEnv")
        if version("rsl-rl-lib") != SUPPORTED_RSL_RL:
            raise RuntimeError("requires exactly rsl-rl-lib==5.5.2")
        self.env, self.device = env, env.device
        self.num_envs, self.num_actions = env.num_envs, env.num_actions
        self.max_episode_length = torch.tensor(env.horizons.copy(), device=self.device)
        self._poisoned = False
        self._step_busy = False
        self.last_transition = None
        self.cfg = {
            "adapter": "phymeo-mujoco-tyx-rsl-v1",
            "rsl_rl": SUPPORTED_RSL_RL,
            "world_keys": list(env.world_keys),
            "physical_dt_s": env.h,
            "decimation": env.decimation,
            "policy_observations": "device-task-copy-owned-CUDA",
            "actions": "device-native-CAS-with-complete-HOST-bit-audit",
            "physics_diagnostics": "complete-HOST-validation-retained",
            "timeout_bootstrap": "RSL5.5.2-entry-act-values",
        }
        if env.control_window is not None:
            self.cfg.update(
                policy_dt_s=env.policy_dt_s,
                control_window=env.control_window.binding(env.h),
            )

    @property
    def needs_recovery(self):
        return self._poisoned or self.env.needs_recovery

    @property
    def episode_length_buf(self):
        return torch.tensor(self.env.episode_length.copy(), device=self.device)

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        raise NotImplementedError(
            "use init_at_random_ep_len=False; clock-only randomization is not a physical episode"
        )

    def _ready(self):
        if self._step_busy:
            raise RuntimeError("an adapter control window is already in progress")
        if self.needs_recovery:
            raise RuntimeError(
                "learner requires explicit full recovery reset; discard pending rollout"
            )

    def _observations(self, value):
        if (
            value.device != self.device
            or value.dtype != torch.float32
            or tuple(value.shape) != (self.num_envs, self.env.task_spec.width)
        ):
            raise ValueError("task observation device/dtype/shape changed")
        return TensorDict({"policy": value}, batch_size=[self.num_envs], device=self.device)

    def get_observations(self):
        self._ready()
        return self._observations(self.env.get_observations())

    def step(self, actions):
        self._ready()
        self._step_busy = True
        try:
            return self._step(actions)
        finally:
            self._step_busy = False

    def _step(self, actions):
        transition = self.env.step(
            actions,
            world_keys=self.env.world_keys,
            expected_frame=self.env.current_frame,
            producer_stream=torch.cuda.current_stream(self.device),
        )
        try:
            observations = self._observations(transition.observation)
            final_observation = self._observations(transition.final_observation)
            extras = {
                "terminated": transition.terminated,
                "time_outs": transition.time_outs,
                "final_observation": final_observation,
                "final_mask": transition.final_mask,
                "episode_id": torch.tensor(transition.episode_id.copy(), device=self.device),
                "episode_length": torch.tensor(
                    transition.episode_length.copy(), device=self.device
                ),
                "episode_return": transition.episode_return,
                "transition_sequence": transition.sequence,
            }
            if self.env.control_window is not None:
                extras["control_window"] = transition.control_window
            done = transition.terminated | transition.time_outs
        except BaseException as error:
            self._poisoned = True
            raise MuJoCoTyxEnvError(
                "wrapper transition accepted; RSL publication failed",
                phase="adapter-publication",
                action_accepted=True,
                state_accepted=True,
                needs_recovery=True,
                action_receipt=transition.action_receipt,
                accepted_worlds=transition.accepted_worlds,
                transition=transition,
                control_window=getattr(transition, "control_window", None),
            ) from error
        self.last_transition = transition
        return observations, transition.reward, done, extras

    def restore_training_state(self, checkpoint, state, *, expected_revision):
        """Explicit composition seam; upstream storage must be empty at caller."""
        self._ready()
        try:
            receipt = self.env.restore_training_state(
                checkpoint, state, expected_revision=expected_revision
            )
        except BaseException:
            if self.env.needs_recovery:
                self._poisoned = True
            raise
        self.last_transition = None
        return receipt

    def invalidate_training_state(self, reason):
        self._poisoned = True
        self.env.invalidate_training_state(reason)

    def reset(self, indices=None):
        if self._step_busy:
            raise RuntimeError("an adapter control window is already in progress")
        if indices is None:
            keys = None
        else:
            values = tuple(indices)
            if any(type(i) is not int or not 0 <= i < self.num_envs for i in values) or (
                len(set(values)) != len(values)
            ):
                raise ValueError("reset indices must be distinct row ordinals")
            if self._poisoned and set(values) != set(range(self.num_envs)):
                raise RuntimeError("adapter recovery requires resetting all worlds")
            keys = tuple(self.env.world_keys[i] for i in values)
        value = self.env.reset(keys)
        try:
            result = self._observations(value)
        except Exception:
            self._poisoned = True
            raise
        self._poisoned = False
        return result


def ppo_config(*, steps=16, recurrent=False):
    """Fixed short-run protocol; not a tuned task-learning configuration."""
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be positive")
    if type(recurrent) is not bool:
        raise TypeError("recurrent must be bool")
    network = {
        "class_name": "rsl_rl.models.RNNModel" if recurrent else "rsl_rl.models.MLPModel",
        "hidden_dims": [32, 32],
        "activation": "tanh",
        "obs_normalization": False,
    }
    if recurrent:
        network.update(rnn_type="gru", rnn_hidden_dim=16, rnn_num_layers=1)
    actor = {
        **network,
        "distribution_cfg": {
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": 0.1,
            "std_type": "log",
            "learn_std": True,
        },
    }
    return {
        "num_steps_per_env": steps,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": actor,
        "critic": dict(network),
        "algorithm": {
            "class_name": "rsl_rl.algorithms.PPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "optimizer": "adam",
            "use_clipped_value_loss": True,
            "schedule": "fixed",
            "desired_kl": None,
            "normalize_advantage_per_mini_batch": False,
            "use_mixed_precision": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "multi_gpu": None,
        "torch_compile_mode": None,
    }


def make_ppo(env, *, steps=16, recurrent=False):
    """Construct original upstream models/storage/PPO with no algorithm override."""
    from rsl_rl.algorithms import PPO

    if not isinstance(env, RslMuJoCoTyxVecEnv):
        raise TypeError("requires the device RSL adapter")
    return PPO.construct_algorithm(
        env.get_observations(),
        env,
        ppo_config(steps=steps, recurrent=recurrent),
        str(env.device),
    )


def optimization_evidence(algorithm):
    """Explicit diagnostic HOST reads after an upstream update."""
    parameters = list(
        dict.fromkeys(list(algorithm.actor.parameters()) + list(algorithm.critic.parameters()))
    )
    gradients = [parameter.grad for parameter in parameters if parameter.requires_grad]
    if not gradients or any(gradient is None for gradient in gradients):
        raise RuntimeError("an optimized parameter has no gradient evidence")
    finite = all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
    steps = [
        int(state["step"].item()) for state in algorithm.optimizer.state.values() if "step" in state
    ]
    if not finite or not steps or len(set(steps)) != 1:
        raise RuntimeError("nonfinite gradients or inconsistent/absent optimizer steps")
    if not all(parameter.is_cuda for parameter in parameters):
        raise RuntimeError("upstream optimization parameter left CUDA")
    return {
        "finite_gradient": finite,
        "optimizer_step": steps[0],
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "gradient_norms": tuple(float(torch.linalg.vector_norm(g).item()) for g in gradients),
    }
