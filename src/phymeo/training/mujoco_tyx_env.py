"""Device-native task/action VecEnv; optional Torch dependency, explicit HOST audits.

The runtime is borrowed and must be mutated exclusively through this wrapper
during a rollout. Physics, policy/optimizer and wrapper metadata have separate
transactions. N2 deliberately does not expose a training restore API.
"""
import json
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
from types import MappingProxyType

import numpy as np


def _frozen(value, dtype):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _indices(values, name):
    result = tuple(values)
    if any(type(index) is not int or index < 0 for index in result) or len(set(result)) != len(result):
        raise ValueError(name + " requires distinct nonnegative native indices")
    return result


@dataclass(frozen=True, init=False)
class MuJoCoTyxActionSpec:
    """Policy columns: ctrl, qfrc, then (body, force/torque component).

    Native selected value = bias + scale * tanh(raw policy action).
    Other native values are zero. There is no implied motor/clipping policy.
    """
    ctrl_indices: tuple
    qfrc_indices: tuple
    xfrc_indices: tuple
    scale: np.ndarray
    bias: np.ndarray
    sha256: str

    def __init__(self, ctrl_indices=(), qfrc_indices=(), xfrc_indices=(), *, scale=None, bias=None):
        ctrl = _indices(ctrl_indices, "ctrl")
        qfrc = _indices(qfrc_indices, "qfrc")
        xfrc = tuple(tuple(pair) for pair in xfrc_indices)
        if any(len(pair) != 2 or any(type(i) is not int or i < 0 for i in pair)
               or pair[1] >= 6 for pair in xfrc) or len(set(xfrc)) != len(xfrc):
            raise ValueError("xfrc requires unique (body, component0..5) native addresses")
        width = len(ctrl) + len(qfrc) + len(xfrc)
        if not 0 < width <= 1024:
            raise ValueError("action width must be 1..1024")
        for name, value in (("ctrl_indices", ctrl), ("qfrc_indices", qfrc), ("xfrc_indices", xfrc)):
            object.__setattr__(self, name, value)
        for name, supplied, default in (("scale", scale, 1.0), ("bias", bias, 0.0)):
            with np.errstate(over="ignore", invalid="ignore"):
                value = np.asarray([default] * width if supplied is None else supplied, np.float32)
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError(name + " requires finite FP32 values for every action column")
            object.__setattr__(self, name, _frozen(value, np.float32))
        binding = json.dumps(
            {"schema": "mujoco-tyx-action-map-v1", "ctrl": ctrl, "qfrc": qfrc,
             "xfrc": xfrc, "squash": "tanh"}, sort_keys=True, separators=(",", ":"),
        ).encode()
        object.__setattr__(self, "sha256", sha256(
            binding + self.scale.tobytes() + self.bias.tobytes()
        ).hexdigest())

    @property
    def width(self):
        return self.scale.size


class MuJoCoTyxEnvError(RuntimeError):
    """No successful learner transition is implied by accepted physics."""
    def __init__(self, message, *, phase, action_accepted=False, state_accepted=False,
                 reset_accepted=False, action_receipt=None, accepted_worlds=None,
                 needs_recovery=False, transition=None):
        super().__init__(message)
        self.phase = phase
        self.action_accepted = bool(action_accepted)
        self.state_accepted = bool(state_accepted)
        self.reset_accepted = bool(reset_accepted)
        self.action_receipt = action_receipt
        self.accepted_worlds = MappingProxyType(dict(accepted_worlds or {}))
        self.needs_recovery = bool(needs_recovery)
        self.transition = transition


@dataclass(frozen=True)
class MuJoCoTyxTransition:
    sequence: int
    world_keys: tuple
    input_frame: object
    output_frame: object
    observation: object
    reward: object
    terminated: object
    time_outs: object
    final_observation: object
    final_mask: object
    episode_id: np.ndarray
    episode_length: np.ndarray
    episode_return: object
    action: object
    native_channels: tuple
    action_receipt: object
    accepted_worlds: object
    states: object


class MuJoCoTyxVectorEnv:
    """Synchronous CUDA VecEnv with borrowed runtime and owned output workspace.

    Returned tensors are fresh Torch-owned storage and never overwritten by a
    later step/reset. Caller mutation of its tensor does not mutate physics.
    HOST lengths and done-mask readback are explicit. A full reset is a fresh
    episode boundary, not learner checkpoint restoration.
    """
    def __init__(self, runtime, task_spec, action_spec, *, h, horizons):
        import torch
        from phymeo.backends.mujoco_task_workspace import MuJoCoTyxTaskSpec

        info = runtime.capabilities
        if info.get("implementation") != "tyx" or info.get("mode") != "cuda":
            raise ValueError("device VecEnv requires admitted MuJoCo-TYX CUDA runtime")
        if torch.version.cuda is None or not torch.cuda.is_available():
            raise RuntimeError("CUDA Torch is required; no HOST fallback")
        if type(task_spec) is not MuJoCoTyxTaskSpec or type(action_spec) is not MuJoCoTyxActionSpec:
            raise TypeError("immutable native task/action specs are required")
        if not np.isfinite(h) or float(h) != runtime.timestep:
            raise ValueError("h must equal the original compiled runtime timestep")
        self.runtime, self.task_spec, self.action_spec = runtime, task_spec, action_spec
        self.world_keys = tuple(runtime.world_keys)
        self.num_envs, self.num_actions = len(self.world_keys), action_spec.width
        self.h = float(h)
        values = tuple(horizons)
        if len(values) != self.num_envs or any(
            type(value) is not int or not 1 <= value <= np.iinfo(np.int32).max for value in values
        ):
            raise ValueError("horizons must be positive i32 counts in frozen world order")
        self.horizons = _frozen(values, np.int64)
        self._nu, self._nv, self._nbody = (info[key] for key in ("nu", "nv", "nbody"))
        if any(i >= self._nu for i in action_spec.ctrl_indices) or any(
            i >= self._nv for i in action_spec.qfrc_indices
        ) or any(body >= self._nbody for body, _ in action_spec.xfrc_indices):
            raise ValueError("action address exceeds the compiled native layout")
        self._lock, self._closed, self._needs_recovery = RLock(), False, False
        self._length = np.zeros(self.num_envs, np.int64)
        self._episode_id = np.zeros(self.num_envs, np.int64)
        self._sequence, self.last_transition, self.last_failure = 0, None, None
        self._counts = dict(transitions=0, physical_steps=0, resets=0,
                            termination_readback_bytes=0, action_finite_scalar_readbacks=0)
        self.workspace = runtime.prepare_task_workspace(task_spec)
        try:
            self.device = torch.device("cuda", self.workspace.binding["device"])
            self._scale = torch.tensor(action_spec.scale.copy(), device=self.device)
            self._bias = torch.tensor(action_spec.bias.copy(), device=self.device)
            self._returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self._frame, self._observation, _, _ = self._export()
            self.binding = MappingProxyType({
                "schema": "mujoco-tyx-vector-env-v1", "world_keys": self.world_keys,
                "task": task_spec.sha256, "action": action_spec.sha256,
                "task_source": self.workspace.binding, "h": self.h,
                "horizons": tuple(values), "decimation": 1,
                "runtime_ownership": "borrowed-exclusive-mutation",
                "physics_diagnostics": "complete-HOST-validation-retained",
                "episode_bookkeeping": "HOST-lengths-and-done-mask",
                "restore": "training-composition-not-provided-in-N2",
            })
        except BaseException:
            self.workspace.close()
            self._closed = True
            raise

    @property
    def current_frame(self):
        return self._frame

    @property
    def needs_recovery(self):
        return self._needs_recovery

    @property
    def episode_length(self):
        with self._lock:
            return _frozen(self._length, np.int64)

    @property
    def episode_id(self):
        with self._lock:
            return _frozen(self._episode_id, np.int64)

    @property
    def counters(self):
        with self._lock:
            return MappingProxyType({**self._counts,
                "runtime": self.runtime.counters, "task": self.workspace.counters})

    def _ready(self):
        if self._closed:
            raise RuntimeError("device VecEnv is closed")
        if self._needs_recovery:
            raise RuntimeError("device VecEnv requires explicit full recovery reset")
        if self.runtime.checkpoint_revision != self._frame.revision:
            self._needs_recovery = True
            raise RuntimeError("external runtime mutation requires recovery; exclusive owner violated")

    def _export(self):
        frame = self.workspace.refresh(expected_revision=self.runtime.checkpoint_revision)
        observation, reward, terminated = self.workspace.to_torch(frame)
        return frame, observation, reward, terminated

    def get_observations(self):
        with self._lock:
            self._ready()
            return self._observation.clone()

    def _preflight(self, actions, world_keys, frame, stream):
        import torch

        if tuple(world_keys) != self.world_keys:
            raise ValueError("world-order metadata differs from the frozen native row order")
        if frame is not self._frame:
            raise ValueError("stale or foreign task frame")
        if not isinstance(actions, torch.Tensor) or actions.device != self.device:
            raise ValueError("policy action must be a CUDA tensor on the runtime device")
        if actions.dtype != torch.float32 or not actions.is_contiguous():
            raise ValueError("policy action requires contiguous FP32")
        if tuple(actions.shape) != (self.num_envs, self.num_actions):
            raise ValueError("policy action shape differs from frozen native mapping")
        if not isinstance(stream, torch.cuda.Stream) or stream.device != self.device:
            raise ValueError("producer stream belongs to a different device")
        # The producer's current stream orders validation before mapping.
        self._counts["action_finite_scalar_readbacks"] += 1
        if not bool(torch.isfinite(actions).all().item()):
            raise ValueError("policy action must be finite")

    def _channels(self, actions):
        import torch

        spec = self.action_spec
        value = self._bias + self._scale * torch.tanh(actions.detach())
        ctrl = torch.zeros((self.num_envs, self._nu), dtype=torch.float32, device=self.device)
        qfrc = torch.zeros((self.num_envs, self._nv), dtype=torch.float32, device=self.device)
        xfrc = torch.zeros((self.num_envs, self._nbody, 6), dtype=torch.float32, device=self.device)
        offset = 0
        for indices, output in ((spec.ctrl_indices, ctrl), (spec.qfrc_indices, qfrc)):
            for index in indices:
                output[:, index] = value[:, offset]
                offset += 1
        for body, component in spec.xfrc_indices:
            xfrc[:, body, component] = value[:, offset]
            offset += 1
        return ctrl, qfrc, xfrc

    def _submit(self, channels, stream, revision):
        from tyx.torch import from_torch, stream_from_torch

        receipt = None
        try:
            with ExitStack() as borrowed:
                views = []
                for index, tensor in enumerate(channels):
                    if index == 0 and self._nu == 0:
                        views.append(None)
                    else:
                        view = from_torch(tensor, backend="cuda")
                        borrowed.callback(view.free)
                        views.append(view)
                producer = stream_from_torch(stream, backend="cuda")
                borrowed.callback(producer.close)
                receipt = self.runtime.set_controls_from_device(
                    views[0], qfrc_applied=views[1], xfrc_applied=views[2],
                    producer_stream=producer, expected_revision=revision,
                )
            return receipt
        except Exception as error:
            if receipt is not None:
                raise MuJoCoTyxEnvError(
                    "device action committed; borrowed-view cleanup failed",
                    phase="action-publication", action_accepted=True,
                    action_receipt=receipt, needs_recovery=True,
                ) from error
            raise

    def step(self, actions, *, world_keys, expected_frame, producer_stream=None):
        import torch

        with self._lock:
            self._ready()
            phase, action_accepted, state_accepted, reset_accepted = "action-preflight", False, False, False
            receipt, reset_revision = None, None
            accepted = dict.fromkeys(self.world_keys, False)
            input_frame = self._frame
            try:
                stream = (torch.cuda.current_stream(self.device)
                          if producer_stream is None else producer_stream)
                # Invalid streams must fail before entering torch.cuda.stream.
                if not isinstance(stream, torch.cuda.Stream) or stream.device != self.device:
                    raise ValueError("producer stream type/device")
                with torch.cuda.stream(stream), torch.no_grad():
                    self._preflight(actions, world_keys, expected_frame, stream)
                    saved_action = actions.detach().clone()
                    channels = self._channels(actions)
                    phase = "action-publication"
                    receipt = self._submit(channels, stream, input_frame.revision)
                    action_accepted = True
                phase = "physics"
                result = self.runtime.step(self.h)
                accepted = dict(result)
                state_accepted = accepted == dict.fromkeys(self.world_keys, True)
                if not state_accepted:
                    raise RuntimeError("physical batch did not accept every declared world")
                self._counts["physical_steps"] += 1
                phase = "physical-diagnostics"
                states = self.runtime.snapshot()
                phase = "task-export"
                final_frame, final_obs, reward, raw_terminated = self._export()
                phase = "episode-bookkeeping"
                terminated = raw_terminated.to(torch.bool)
                host_terminated = terminated.detach().cpu().numpy().copy()
                self._counts["termination_readback_bytes"] += host_terminated.nbytes
                lengths = self._length + 1
                timeout_host = (lengths >= self.horizons) & ~host_terminated
                done_host = host_terminated | timeout_host
                time_outs = torch.tensor(timeout_host, dtype=torch.bool, device=self.device)
                final_mask = terminated | time_outs
                final_observation = final_obs.clone()
                returns = self._returns + reward
                output_frame, observation = final_frame, final_obs
                ids = self._episode_id.copy()
                next_lengths, next_ids = lengths.copy(), ids.copy()
                next_returns = returns.clone()
                keys = tuple(key for key, done in zip(self.world_keys, done_host, strict=True) if done)
                if keys:
                    phase = "subset-reset"
                    reset_revision = self.runtime.checkpoint_revision
                    self.runtime.reset(keys)
                    reset_accepted = True
                    self._counts["resets"] += len(keys)
                    phase = "reset-task-export"
                    output_frame, observation, _, _ = self._export()
                    next_lengths[done_host] = 0
                    next_ids[done_host] += 1
                    next_returns[final_mask] = 0
                phase = "transition-publication"
                transition = MuJoCoTyxTransition(
                    self._sequence + 1, self.world_keys, input_frame, output_frame,
                    observation, reward, terminated, time_outs, final_observation, final_mask,
                    _frozen(ids, np.int64), _frozen(lengths, np.int64), returns,
                    saved_action, channels, receipt, MappingProxyType(accepted), states,
                )
                # Internal observation/return owners cannot be mutated via a returned transition.
                current_observation = observation.clone()
                torch.cuda.current_stream(self.device).synchronize()
                self._length, self._episode_id, self._returns = next_lengths, next_ids, next_returns
                self._frame, self._observation = output_frame, current_observation
                self._sequence, self.last_transition = transition.sequence, transition
                self._counts["transitions"] += 1
                return transition
            except Exception as error:
                if isinstance(error, MuJoCoTyxEnvError) and error.action_accepted:
                    action_accepted, receipt = True, error.action_receipt
                if reset_revision is not None and self.runtime.checkpoint_revision != reset_revision:
                    reset_accepted = True
                recovery = action_accepted or state_accepted or reset_accepted
                self._needs_recovery = self._needs_recovery or recovery
                failure = MuJoCoTyxEnvError(
                    "device rollout stopped at " + phase, phase=phase,
                    action_accepted=action_accepted, state_accepted=state_accepted,
                    reset_accepted=reset_accepted, action_receipt=receipt,
                    accepted_worlds=accepted, needs_recovery=self._needs_recovery,
                )
                self.last_failure = failure
                raise failure from error

    def reset(self, world_keys=None):
        """Explicit fresh episodes; full batch required after a partial transaction."""
        import torch

        with self._lock:
            if self._closed:
                raise RuntimeError("device VecEnv is closed")
            selected = self.world_keys if world_keys is None else tuple(world_keys)
            if len(set(selected)) != len(selected) or not set(selected) <= set(self.world_keys):
                raise ValueError("reset keys must be unique members of frozen world order")
            if self._needs_recovery and set(selected) != set(self.world_keys):
                raise RuntimeError("recovery requires resetting all worlds")
            if not selected:
                self._ready()
                return self._observation.clone()
            revision = self.runtime.checkpoint_revision
            try:
                self.runtime.reset(selected)
                frame, observation, _, _ = self._export()
                mask = np.array([key in selected for key in self.world_keys])
                lengths, ids = self._length.copy(), self._episode_id.copy()
                lengths[mask] = 0
                ids[mask] += 1
                returns = self._returns.clone()
                returns[torch.tensor(mask, device=self.device)] = 0
                torch.cuda.current_stream(self.device).synchronize()
            except Exception as error:
                changed = self.runtime.checkpoint_revision != revision
                self._needs_recovery = self._needs_recovery or changed
                failure = MuJoCoTyxEnvError(
                    "explicit reset failed; no training restoration claimed", phase="explicit-reset",
                    reset_accepted=changed, needs_recovery=self._needs_recovery,
                )
                self.last_failure = failure
                raise failure from error
            self._length, self._episode_id, self._returns = lengths, ids, returns
            self._frame, self._observation = frame, observation
            self._counts["resets"] += len(selected)
            self._needs_recovery, self.last_failure = False, None
            return observation.clone()

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self.workspace.close()

    def __enter__(self):
        with self._lock:
            self._ready()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
