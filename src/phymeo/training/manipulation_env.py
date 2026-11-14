"""CUDA heterogeneous manipulation lifecycle over the public batch protocol.

The batch is borrowed for exclusive mutation. Success evaluation is an explicit
HOST consumer of accepted-entry data; actor observations remain device copies.
This adapter does not implement language encoding, PPO or a control policy.
"""

import json
import math
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from threading import RLock
from types import MappingProxyType

import numpy as np

from .manipulation_observation import ManipulationSuccessReader, SuccessReaderCheckpoint
from .manipulation_tasks import Goal, Scene, SuccessHistory, SuccessResult, evaluate_goal
from .mujoco_tyx_env import MuJoCoTyxActionSpec, MuJoCoTyxEnvError

_INT64 = (1 << 63) - 1
_INT32 = (1 << 31) - 1
_UNSET = object()
_STATE_FIELDS = {
    "schema",
    "sequence",
    "goals",
    "episode_ids",
    "episode_lengths",
    "returns",
    "histories",
    "cursor_records",
    "pending_reset",
}
_HISTORY_FIELDS = {
    "binding",
    "tick",
    "time_s",
    "initial_bottom_m",
    "initial_xy",
    "lifted",
    "transported",
    "held_s",
    "holding",
}


def _require(condition, message):
    if not condition:
        raise ValueError("manipulation Env: " + message)


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _host(values):
    array = np.asarray(values, dtype=np.int64)
    return np.frombuffer(array.tobytes(), dtype=np.int64)


def _integer(value, low=0, high=_INT64):
    return type(value) is int and low <= value <= high


def _finite(value):
    return type(value) is float and math.isfinite(value)


def _hex(value, size):
    return type(value) is str and len(value) == size and all(c in "0123456789abcdef" for c in value)


def _history_binding(scene, goal, world, episode):
    target = next(target for target in scene.targets if target.key == goal.target_key)
    return (
        scene.physical_binding,
        goal.binding,
        (target.key, target.position, target.yaw, target.half_extents),
        world,
        episode,
    )


@dataclass(frozen=True)
class ManipulationTransition:
    sequence: int
    world_keys: tuple
    input_frame: object
    evaluation_frame: object
    output_frame: object
    observation: object
    final_observation: object
    reward: object
    terminated: object
    truncated: object
    final_mask: object
    episode_id: np.ndarray
    episode_length: np.ndarray
    episode_return: object
    action: object
    native_channels: tuple
    action_receipt: object
    accepted_worlds: object
    success_frames: object
    results: object
    goal_bindings: object


class HeterogeneousManipulationEnv:
    """One policy action per physical step, with independent world episodes."""

    def __init__(self, batch, scenes, *, goals, action_specs, horizons, auto_reset=True):
        import torch

        _require(
            batch.capabilities.get("implementation") == "tyx"
            and batch.capabilities.get("mode") == "cuda",
            "CUDA TYX batch required",
        )
        _require(
            torch.version.cuda is not None and torch.cuda.is_available(), "CUDA Torch required"
        )
        _require(type(auto_reset) is bool, "auto_reset must be bool")
        self.batch = batch
        self.world_keys = tuple(batch.world_keys)
        self.num_envs = len(self.world_keys)
        for name, values in (
            ("scenes", scenes),
            ("goals", goals),
            ("action_specs", action_specs),
            ("horizons", horizons),
        ):
            _require(
                isinstance(values, Mapping)
                and all(type(k) is int for k in values)
                and set(values) == set(self.world_keys),
                name + " must cover exactly the frozen world keys",
            )
        self._scenes, self._specs = dict(scenes), dict(action_specs)
        self._goals, self._horizons = dict(goals), dict(horizons)
        for key in self.world_keys:
            scene, spec, layout = scenes[key], action_specs[key], batch.layout[key]
            _require(type(scene) is Scene, "immutable Scene declaration required")
            self._validate_goal(key, goals[key])
            _require(type(spec) is MuJoCoTyxActionSpec, "immutable native action spec required")
            _require(
                not any(i >= layout.nu for i in spec.ctrl_indices)
                and not any(i >= layout.nv for i in spec.qfrc_indices)
                and not any(b >= layout.nbody for b, _ in spec.xfrc_indices),
                "action address exceeds world native layout",
            )
            _require(_integer(horizons[key], 1, _INT32), "horizon must be a positive i32 count")
        self.num_actions = max(spec.width for spec in action_specs.values())
        self.h = batch.timestep
        self._auto_reset = auto_reset
        self._lock, self._closed, self._needs_recovery = RLock(), False, False
        self._busy = False
        self.last_failure, self.last_transition = None, None
        self._sequence = 0
        self._episodes = dict.fromkeys(self.world_keys, 0)
        self._lengths = dict.fromkeys(self.world_keys, 0)
        self._returns = dict.fromkeys(self.world_keys, 0.0)
        self._history = dict.fromkeys(self.world_keys, None)
        self._pending_reset = dict.fromkeys(self.world_keys, False)
        self.workspace = batch.prepare_device_workspace(include_contacts=True, include_bodies=True)
        try:
            self.device = torch.device("cuda", self.workspace.binding["device"])
            self._dimensions = self.workspace.binding["dimensions"]
            self._has_ctrl = any(batch.layout[key].nu for key in self.world_keys)
            mask = np.zeros((self.num_envs, self.num_actions), dtype=np.bool_)
            for row, key in enumerate(self.world_keys):
                mask[row, : self._specs[key].width] = True
            self._action_mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
            self._scale = tuple(
                torch.tensor(self._specs[k].scale.copy(), device=self.device)
                for k in self.world_keys
            )
            self._bias = tuple(
                torch.tensor(self._specs[k].bias.copy(), device=self.device)
                for k in self.world_keys
            )
            self._reader = ManipulationSuccessReader(self.workspace, scenes)
            self._frame, self._observation = self._export()
            initial = self.workspace.readback(self._frame)["entry_meta"]
            _require(
                np.all(initial[:, 0] == 0)
                and np.all(initial[:, 2] == 0)
                and all(tick == 0 for tick in self._frame.ticks),
                "constructor requires empty tick-zero physics",
            )
            _require(
                not self._reader.read(self._frame, episodes=self._episodes),
                "initial entry must be absent",
            )
            self._binding = _plain(
                {
                    "schema": "heterogeneous-manipulation-env-v1",
                    "source": self.workspace.binding["source"],
                    "world_keys": self.world_keys,
                    "scenes": [scenes[k].physical_binding for k in self.world_keys],
                    "goal_catalogs": [
                        [g.binding for g in scenes[k].goals] for k in self.world_keys
                    ],
                    "actions": [action_specs[k].sha256 for k in self.world_keys],
                    "horizons": [horizons[k] for k in self.world_keys],
                    "auto_reset": auto_reset,
                    "h": self.h,
                }
            )
            torch.cuda.current_stream(self.device).synchronize()
        except BaseException as primary:
            self._closed = True
            try:
                self.workspace.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Env construction and workspace cleanup", [primary, cleanup]
                ) from None
            raise

    def _validate_goal(self, key, goal):
        _require(
            type(goal) is Goal and any(goal is candidate for candidate in self._scenes[key].goals),
            "goal must be an object from the world's Scene catalog",
        )

    def _open(self):
        if self._closed:
            raise RuntimeError("manipulation Env is closed")

    def _failure(
        self,
        message,
        *,
        phase,
        action=False,
        state=False,
        reset=False,
        receipt=None,
        accepted=None,
        recovery=False,
    ):
        self._needs_recovery = self._needs_recovery or recovery or action or state or reset
        failure = MuJoCoTyxEnvError(
            message,
            phase=phase,
            action_accepted=action,
            state_accepted=state,
            reset_accepted=reset,
            action_receipt=receipt,
            accepted_worlds=accepted,
            needs_recovery=self._needs_recovery,
        )
        self.last_failure = failure
        return failure

    def _ready(self, expected_frame=_UNSET, *, stepping=False):
        self._open()
        if self._busy:
            raise RuntimeError("an Env transaction is already active")
        if self._needs_recovery:
            raise MuJoCoTyxEnvError(
                "restore is required before further Env work", phase="recovery", needs_recovery=True
            )
        if self.batch.checkpoint_revision != self._frame.revision:
            raise self._failure(
                "external batch mutation requires paired restore",
                phase="frame-preflight",
                recovery=True,
            )
        if expected_frame is not _UNSET and expected_frame is not self._frame:
            raise self._failure(
                "stale or foreign Env frame", phase="frame-preflight", recovery=True
            )
        if stepping and any(self._pending_reset.values()):
            raise self._failure("finished worlds require explicit reset", phase="action-preflight")

    def _export(self):
        frame = self.workspace.refresh(expected_revision=self.batch.checkpoint_revision)
        values = dict(self.workspace.to_torch(frame))
        values["action_mask"] = self._action_mask.clone()
        return frame, values

    @staticmethod
    def _copies(values):
        return {name: value.clone() for name, value in values.items()}

    @property
    def binding(self):
        return json.loads(json.dumps(self._binding, allow_nan=False))

    @property
    def current_frame(self):
        return self._frame

    @property
    def auto_reset(self):
        return self._auto_reset

    @property
    def needs_recovery(self):
        return self._needs_recovery

    @property
    def observation(self):
        with self._lock:
            self._ready()
            return self._copies(self._observation)

    @property
    def action_mask(self):
        with self._lock:
            self._open()
            return self._action_mask.clone()

    @property
    def episode_ids(self):
        with self._lock:
            return MappingProxyType(dict(self._episodes))

    @property
    def episode_lengths(self):
        with self._lock:
            return MappingProxyType(dict(self._lengths))

    @property
    def returns(self):
        with self._lock:
            return MappingProxyType(dict(self._returns))

    @property
    def history(self):
        with self._lock:
            return MappingProxyType(dict(self._history))

    @property
    def goals(self):
        with self._lock:
            return MappingProxyType(dict(self._goals))

    @property
    def horizons(self):
        return MappingProxyType(dict(self._horizons))

    @property
    def pending_reset(self):
        with self._lock:
            return MappingProxyType(dict(self._pending_reset))

    @property
    def contexts(self):
        with self._lock:
            return MappingProxyType(
                {
                    key: MappingProxyType(
                        {
                            "instructions_zh": self._goals[key].instructions_zh,
                            "instructions_en": self._goals[key].instructions_en,
                            "object_label_zh": self._scenes[key].object_label_zh,
                            "object_label_en": self._scenes[key].object_label_en,
                            "targets": self._scenes[key].targets,
                        }
                    )
                    for key in self.world_keys
                }
            )

    def _actions(self, actions, world_keys, stream):
        import torch

        order = tuple(world_keys)
        _require(
            all(type(key) is int for key in order) and order == self.world_keys,
            "action world order differs",
        )
        _require(
            isinstance(stream, torch.cuda.Stream) and stream.device == self.device,
            "producer stream type/device",
        )
        _require(
            isinstance(actions, torch.Tensor)
            and actions.device == self.device
            and actions.dtype == torch.float32
            and actions.is_contiguous()
            and tuple(actions.shape) == (self.num_envs, self.num_actions),
            "actions require contiguous CUDA FP32 in frozen padded layout",
        )
        _require(
            bool(torch.isfinite(actions.masked_select(self._action_mask)).all().item()),
            "valid action columns must be finite",
        )
        saved = torch.where(self._action_mask, actions.detach(), torch.zeros_like(actions))
        ctrl = torch.zeros(
            (self.num_envs, self._dimensions["nu"]), dtype=torch.float32, device=self.device
        )
        qfrc = torch.zeros(
            (self.num_envs, self._dimensions["nv"]), dtype=torch.float32, device=self.device
        )
        xfrc = torch.zeros(
            (self.num_envs, self._dimensions["nbody"], 6), dtype=torch.float32, device=self.device
        )
        for row, key in enumerate(self.world_keys):
            spec = self._specs[key]
            values = self._bias[row] + self._scale[row] * torch.tanh(saved[row, : spec.width])
            offset = 0
            for indices, target in ((spec.ctrl_indices, ctrl), (spec.qfrc_indices, qfrc)):
                for index in indices:
                    target[row, index] = values[offset]
                    offset += 1
            for body, component in spec.xfrc_indices:
                xfrc[row, body, component] = values[offset]
                offset += 1
        return saved, (ctrl, qfrc, xfrc)

    def _submit(self, channels, stream, revision):
        from tyx.torch import from_torch, stream_from_torch

        receipt = None
        try:
            with ExitStack() as owners:
                views = []
                for i, value in enumerate(channels):
                    if i == 0 and not self._has_ctrl:
                        views.append(None)
                    else:
                        view = from_torch(value, backend="cuda")
                        owners.callback(view.free)
                        views.append(view)
                producer = stream_from_torch(stream, backend="cuda")
                owners.callback(producer.close)
                receipt = self.batch.set_controls_from_device(
                    views[0],
                    qfrc_applied=views[1],
                    xfrc_applied=views[2],
                    world_keys=self.world_keys,
                    expected_revision=revision,
                    producer_stream=producer,
                )
            return receipt
        except BaseException as error:
            if receipt is not None:
                raise self._failure(
                    "action accepted; borrowed view cleanup failed",
                    phase="action-publication",
                    action=True,
                    receipt=receipt,
                ) from error
            raise

    def step(self, actions, *, world_keys, expected_frame, producer_stream=None):
        import torch

        with self._lock:
            self._ready(expected_frame, stepping=True)
            phase = "action-preflight"
            action_accepted = state_accepted = reset_accepted = False
            receipt = None
            accepted = dict.fromkeys(self.world_keys, False)
            action_revision = self.batch.checkpoint_revision
            physics_revision = reset_revision = None
            input_frame = self._frame
            self._busy = True
            try:
                _require(
                    self._sequence < _INT64 and all(v < _INT32 for v in self._lengths.values()),
                    "episode/transition counter exhausted",
                )
                _require(
                    not self._auto_reset or all(v < _INT64 for v in self._episodes.values()),
                    "episode counter exhausted",
                )
                stream = (
                    torch.cuda.current_stream(self.device)
                    if producer_stream is None
                    else producer_stream
                )
                _require(
                    isinstance(stream, torch.cuda.Stream) and stream.device == self.device,
                    "producer stream type/device",
                )
                with torch.cuda.stream(stream), torch.no_grad():
                    saved, channels = self._actions(actions, world_keys, stream)
                    phase = "action-publication"
                    receipt = self._submit(channels, stream, input_frame.revision)
                    action_accepted = True
                phase = "physics"
                physics_revision = self.batch.checkpoint_revision
                accepted = dict(self.batch.step(self.h))
                state_accepted = any(accepted.values())
                _require(
                    accepted == dict.fromkeys(self.world_keys, True),
                    "physical batch did not accept all worlds",
                )
                phase = "task-export"
                evaluation_frame, final_values = self._export()
                phase = "success-evaluation"
                success_frames = self._reader.read(evaluation_frame, episodes=self._episodes)
                _require(
                    set(success_frames) == set(self.world_keys),
                    "one input sample per physical step required",
                )
                results, histories = {}, {}
                for key in self.world_keys:
                    result = evaluate_goal(
                        self._scenes[key], self._goals[key], success_frames[key], self._history[key]
                    )
                    _require(
                        type(result) is SuccessResult and type(result.success) is bool,
                        "invalid success evaluation result",
                    )
                    results[key], histories[key] = result, result.history
                lengths = {k: self._lengths[k] + 1 for k in self.world_keys}
                flags = {k: results[k].success for k in self.world_keys}
                timeouts = {
                    k: lengths[k] >= self._horizons[k] and not flags[k] for k in self.world_keys
                }
                done = {k: flags[k] or timeouts[k] for k in self.world_keys}
                returns = {k: self._returns[k] + float(flags[k]) for k in self.world_keys}
                reward = torch.tensor(
                    [float(flags[k]) for k in self.world_keys],
                    dtype=torch.float32,
                    device=self.device,
                )
                terminated = torch.tensor(
                    [flags[k] for k in self.world_keys], dtype=torch.bool, device=self.device
                )
                truncated = torch.tensor(
                    [timeouts[k] for k in self.world_keys], dtype=torch.bool, device=self.device
                )
                final_mask = terminated | truncated
                episode_return = torch.tensor(
                    [returns[k] for k in self.world_keys], dtype=torch.float32, device=self.device
                )
                next_episodes, next_lengths, next_returns = (
                    dict(self._episodes),
                    dict(lengths),
                    dict(returns),
                )
                next_history, pending = dict(histories), dict(done)
                output_frame, values = evaluation_frame, final_values
                selected = tuple(k for k in self.world_keys if done[k])
                if self._auto_reset and selected:
                    for key in selected:
                        next_episodes[key] += 1
                        next_lengths[key], next_returns[key], next_history[key], pending[key] = (
                            0,
                            0.0,
                            None,
                            False,
                        )
                    phase = "subset-reset"
                    reset_revision = self.batch.checkpoint_revision
                    self.batch.reset(selected)
                    reset_accepted = True
                    phase = "reset-task-export"
                    output_frame, values = self._export()
                    _require(
                        not self._reader.read(output_frame, episodes=next_episodes),
                        "reset must not invent input samples",
                    )
                phase = "transition-publication"
                transition = ManipulationTransition(
                    self._sequence + 1,
                    self.world_keys,
                    input_frame,
                    evaluation_frame,
                    output_frame,
                    self._copies(values),
                    self._copies(final_values),
                    reward,
                    terminated,
                    truncated,
                    final_mask,
                    _host([self._episodes[k] for k in self.world_keys]),
                    _host([lengths[k] for k in self.world_keys]),
                    episode_return,
                    saved,
                    channels,
                    receipt,
                    MappingProxyType(accepted),
                    success_frames,
                    MappingProxyType(results),
                    MappingProxyType({k: self._goals[k].binding for k in self.world_keys}),
                )
                current = self._copies(values)
                torch.cuda.current_stream(self.device).synchronize()
                self._episodes, self._lengths, self._returns = (
                    next_episodes,
                    next_lengths,
                    next_returns,
                )
                self._history, self._pending_reset = next_history, pending
                self._frame, self._observation = output_frame, current
                self._sequence, self.last_transition, self.last_failure = (
                    transition.sequence,
                    transition,
                    None,
                )
                return transition
            except BaseException as error:
                revision = self.batch.checkpoint_revision
                if isinstance(error, MuJoCoTyxEnvError) and error.action_accepted:
                    action_accepted, receipt = True, error.action_receipt
                action_accepted = action_accepted or revision != action_revision
                if (
                    physics_revision is not None
                    and phase == "physics"
                    and revision != physics_revision
                ):
                    state_accepted = True
                    accepted = dict.fromkeys(self.world_keys, True)
                reset_accepted = reset_accepted or (
                    reset_revision is not None and revision != reset_revision
                )
                raise self._failure(
                    "manipulation transition stopped at " + phase,
                    phase=phase,
                    action=action_accepted,
                    state=state_accepted,
                    reset=reset_accepted,
                    receipt=receipt,
                    accepted=accepted,
                ) from error
            finally:
                self._busy = False

    def reset(self, world_keys=None, *, goals=None, expected_frame):
        import torch

        with self._lock:
            self._ready(expected_frame)
            selected = self.world_keys if world_keys is None else tuple(world_keys)
            _require(
                all(type(key) is int for key in selected)
                and len(set(selected)) == len(selected)
                and set(selected) <= set(self.world_keys),
                "reset world selection",
            )
            if goals is not None:
                _require(
                    isinstance(goals, Mapping)
                    and all(type(k) is int for k in goals)
                    and set(goals) == set(selected),
                    "reset goals must cover exactly selected worlds",
                )
                for key in selected:
                    self._validate_goal(key, goals[key])
            _require(
                all(self._episodes[key] < _INT64 for key in selected), "episode counter exhausted"
            )
            if not selected:
                return self._copies(self._observation)
            next_goals = dict(self._goals)
            episodes, lengths, returns = (
                dict(self._episodes),
                dict(self._lengths),
                dict(self._returns),
            )
            histories, pending = dict(self._history), dict(self._pending_reset)
            for key in selected:
                if goals is not None:
                    next_goals[key] = goals[key]
                episodes[key] += 1
                lengths[key], returns[key], histories[key], pending[key] = 0, 0.0, None, False
            revision = self.batch.checkpoint_revision
            self._busy = True
            try:
                self.batch.reset(selected)
                frame, values = self._export()
                _require(
                    not self._reader.read(frame, episodes=episodes),
                    "reset must not invent input samples",
                )
                result, current = self._copies(values), self._copies(values)
                torch.cuda.current_stream(self.device).synchronize()
            except BaseException as error:
                raise self._failure(
                    "explicit manipulation reset failed",
                    phase="explicit-reset",
                    reset=self.batch.checkpoint_revision != revision,
                ) from error
            finally:
                self._busy = False
            self._goals, self._episodes, self._lengths, self._returns = (
                next_goals,
                episodes,
                lengths,
                returns,
            )
            self._history, self._pending_reset = histories, pending
            self._frame, self._observation = frame, current
            self.last_transition, self.last_failure = None, None
            return result

    def _state(self):
        return _plain(
            {
                "schema": "heterogeneous-manipulation-state-v1",
                "sequence": self._sequence,
                "goals": [self._goals[k].binding for k in self.world_keys],
                "episode_ids": [self._episodes[k] for k in self.world_keys],
                "episode_lengths": [self._lengths[k] for k in self.world_keys],
                "returns": [float(self._returns[k]) for k in self.world_keys],
                "histories": [
                    None if self._history[k] is None else asdict(self._history[k])
                    for k in self.world_keys
                ],
                "cursor_records": self._reader.checkpoint().records,
                "pending_reset": [self._pending_reset[k] for k in self.world_keys],
            }
        )

    def _validate_state(self, state):
        """Validate all semantic owners before requesting a physical restore."""
        _require(type(state) is dict and set(state) == _STATE_FIELDS, "checkpoint state fields")
        _require(
            state["schema"] == "heterogeneous-manipulation-state-v1", "checkpoint state schema"
        )
        sequence = state["sequence"]
        _require(_integer(sequence), "checkpoint sequence")
        for name in _STATE_FIELDS - {"schema", "sequence"}:
            _require(
                type(state[name]) is list and len(state[name]) == self.num_envs,
                "checkpoint row layout: " + name,
            )
        goals, episodes, lengths, returns, histories, pending, records = {}, {}, {}, {}, {}, {}, []
        for row, key in enumerate(self.world_keys):
            goal_hash = state["goals"][row]
            catalog = {goal.binding: goal for goal in self._scenes[key].goals}
            _require(_hex(goal_hash, 64) and goal_hash in catalog, "checkpoint goal catalog")
            goal = catalog[goal_hash]
            episode, length = state["episode_ids"][row], state["episode_lengths"][row]
            value, waiting = state["returns"][row], state["pending_reset"][row]
            _require(
                _integer(episode)
                and _integer(length, 0, self._horizons[key])
                and length <= sequence,
                "checkpoint episode/length",
            )
            _require(
                _finite(value) and value >= 0.0 and type(waiting) is bool,
                "checkpoint return/pending reset",
            )
            record = state["cursor_records"][row]
            _require(type(record) is list and len(record) == 6, "checkpoint cursor layout")
            world, cursor_episode, tick, time_s, digest, boundary = record
            _require(
                type(world) is int
                and world == key
                and _integer(cursor_episode)
                and cursor_episode == episode
                and _integer(tick, -1, _INT32)
                and tick == length - 1,
                "checkpoint cursor world/episode/tick",
            )
            _require(_finite(time_s) and time_s >= 0.0, "checkpoint cursor time")
            _require(
                type(boundary) is list
                and len(boundary) == 3
                and _hex(boundary[0], 32)
                and _integer(boundary[1])
                and type(boundary[2]) is str
                and boundary[2] in ("initial", "reset", "reset-states", "restore"),
                "checkpoint cursor boundary",
            )
            raw_history = state["histories"][row]
            success = False
            if length == 0:
                _require(
                    raw_history is None
                    and digest == ""
                    and type(digest) is str
                    and time_s == 0.0
                    and value == 0.0
                    and not waiting,
                    "empty checkpoint episode/history/cursor",
                )
                history = None
            else:
                _require(_hex(digest, 64), "checkpoint cursor fingerprint")
                _require(
                    type(raw_history) is dict and set(raw_history) == _HISTORY_FIELDS,
                    "checkpoint history fields",
                )
                _require(
                    _integer(raw_history["tick"], 0, _INT32)
                    and raw_history["tick"] == tick
                    and _finite(raw_history["time_s"])
                    and raw_history["time_s"] == time_s,
                    "checkpoint history tick/time",
                )
                expected = _history_binding(self._scenes[key], goal, key, episode)
                _require(
                    type(raw_history["binding"]) is list
                    and json.dumps(raw_history["binding"], allow_nan=False)
                    == json.dumps(_plain(expected), allow_nan=False),
                    "checkpoint history goal/scene/episode binding",
                )
                bottom, xy, held = (
                    raw_history[name] for name in ("initial_bottom_m", "initial_xy", "held_s")
                )
                _require(
                    _finite(bottom)
                    and -0.005 <= bottom <= 0.005
                    and type(xy) is list
                    and len(xy) == 2
                    and all(_finite(x) for x in xy),
                    "checkpoint history initial geometry",
                )
                _require(
                    all(
                        type(raw_history[name]) is bool
                        for name in ("lifted", "transported", "holding")
                    ),
                    "checkpoint history flags",
                )
                _require(
                    not raw_history["transported"] or raw_history["lifted"],
                    "checkpoint transport requires prior lift",
                )
                _require(
                    _finite(held)
                    and 0.0 <= held <= min(time_s + 2e-6, tick * (self.h + 2e-6))
                    and (raw_history["holding"] or held == 0.0),
                    "checkpoint history dwell",
                )
                history = SuccessHistory(
                    expected,
                    tick,
                    time_s,
                    bottom,
                    tuple(xy),
                    raw_history["lifted"],
                    raw_history["transported"],
                    held,
                    raw_history["holding"],
                )
                success = history.holding and history.held_s + 1e-12 >= goal.dwell_s
                _require(value == float(success), "checkpoint sparse return differs from history")
            done = success or length >= self._horizons[key]
            _require(
                waiting == (done and not self._auto_reset) and not (self._auto_reset and done),
                "checkpoint pending reset/history/horizon differs",
            )
            goals[key], episodes[key], lengths[key], returns[key] = goal, episode, length, value
            histories[key], pending[key] = history, waiting
            records.append((key, episode, tick, time_s, digest, tuple(boundary)))
        cursor = SuccessReaderCheckpoint(self._reader.binding, tuple(records))
        return sequence, goals, episodes, lengths, returns, histories, pending, cursor

    def checkpoint(self):
        """Capture physics and all semantic owners at the accepted Env boundary."""
        from .manipulation_checkpoint import capture_manipulation_checkpoint

        with self._lock:
            self._ready()
            state = self._state()
            self._validate_state(state)
            physics = self.batch.checkpoint()
            self._ready()
            return capture_manipulation_checkpoint(self._binding, physics, state)

    def restore(self, checkpoint, *, expected_revision):
        """Prevalidate metadata, then restore physics and publish paired owners.

        A failure after physical acceptance requires another explicit restore;
        neither the old observation nor a rolled-back physics state is claimed.
        """
        import torch

        from .manipulation_checkpoint import ManipulationEnvCheckpoint

        with self._lock:
            self._open()
            if self._busy:
                raise RuntimeError("an Env transaction is already active")
            _require(
                type(expected_revision) is int
                and expected_revision == self.batch.checkpoint_revision,
                "stale checkpoint destination revision",
            )
            _require(
                type(checkpoint) is ManipulationEnvCheckpoint, "immutable Env checkpoint required"
            )
            physics, state = checkpoint.read(self._binding)
            sequence, goals, episodes, lengths, returns, histories, pending, cursor = (
                self._validate_state(state)
            )
            reader = ManipulationSuccessReader(self.workspace, self._scenes)
            revision = self.batch.checkpoint_revision
            self._busy = True
            try:
                receipt = self.batch.restore(physics, expected_revision=expected_revision)
                frame, values = self._export()
                reader.restore(cursor, frame=frame)
                current = self._copies(values)
                torch.cuda.current_stream(self.device).synchronize()
            except BaseException as error:
                changed = self.batch.checkpoint_revision != revision
                if changed:
                    raise self._failure(
                        "physics restored; semantic publication failed",
                        phase="training-restore",
                        state=True,
                    ) from error
                raise
            finally:
                self._busy = False
            self._sequence = sequence
            self._goals, self._episodes, self._lengths, self._returns = (
                goals,
                episodes,
                lengths,
                returns,
            )
            self._history, self._pending_reset = histories, pending
            self._reader, self._frame, self._observation = reader, frame, current
            self._needs_recovery, self.last_failure, self.last_transition = False, None, None
            return receipt

    def close(self):
        with self._lock:
            if self._busy:
                raise RuntimeError("cannot close an active Env transaction")
            self._closed = True
            self.workspace.close()

    def __enter__(self):
        with self._lock:
            self._ready()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
