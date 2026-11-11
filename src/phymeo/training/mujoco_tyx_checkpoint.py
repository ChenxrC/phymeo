"""Exact update-boundary composition for the fixed N2 RSL5.5.2 protocol.

Explicit optional CUDA training module. No pickle, arbitrary learner import,
private physical bank, PPO reimplementation, or cross-platform exactness claim.
"""

import base64
import copy
import hashlib
import json
import math
import os
import random
import sys
import uuid
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from threading import RLock

import numpy as np
import torch
from tensordict import TensorDict

from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint

from .rsl_mujoco_tyx import RslMuJoCoTyxVecEnv, optimization_evidence, ppo_config

MAX_PHYSICS = 128 * 1024 * 1024
MAX_TRAINING = 64 * 1024 * 1024
MAX_MANIFEST = 1024 * 1024
MAX_ITEMS = 1_048_576
DTYPES = {"|b1", "|u1", "<i4", "<i8", "<u4", "<u8", "<f4", "<f8"}
SCHEMA = "mujoco-tyx-ppo-checkpoint-v1"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_clone(item) for item in value)
    return copy.deepcopy(value)


def _encode(value, depth=0):
    _require(depth <= 32, "training state depth")
    if isinstance(value, (np.ndarray, torch.Tensor)):
        kind = "torch" if isinstance(value, torch.Tensor) else "numpy"
        array = value.detach().cpu().numpy() if kind == "torch" else value
        _require(array.dtype.str in DTYPES and array.size <= MAX_ITEMS, "training array dtype/size")
        _require(np.isfinite(array).all(), "nonfinite training array")
        return {
            "$array": kind,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    if isinstance(value, Mapping):
        if all(type(key) is str and not key.startswith("$") for key in value):
            return {key: _encode(item, depth + 1) for key, item in value.items()}
        return {
            "$mapping": [
                [_encode(key, depth + 1), _encode(item, depth + 1)] for key, item in value.items()
            ]
        }
    if isinstance(value, tuple):
        return {"$tuple": [_encode(item, depth + 1) for item in value]}
    if isinstance(value, list):
        return [_encode(item, depth + 1) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        _require(not isinstance(value, float) or math.isfinite(value), "nonfinite scalar")
        return value
    raise TypeError("unsupported training state value: " + type(value).__name__)


def _decode(value, budget, depth=0):
    _require(depth <= 32, "training state depth")
    budget[0] += 1
    _require(budget[0] <= MAX_ITEMS, "training state item budget")
    if isinstance(value, dict):
        if "$array" in value:
            _require(set(value) == {"$array", "dtype", "shape", "data"}, "array schema")
            _require(
                value["$array"] in ("torch", "numpy") and value["dtype"] in DTYPES,
                "array representation/dtype",
            )
            shape = value["shape"]
            _require(type(shape) is list and len(shape) <= 4, "array dimensions")
            count = 1
            for size in shape:
                _require(type(size) is int and 0 <= size <= MAX_ITEMS, "array dimension")
                count *= size
                _require(count <= MAX_ITEMS, "array element limit")
            size = count * np.dtype(value["dtype"]).itemsize
            budget[1] += size
            _require(budget[1] <= MAX_TRAINING, "decoded array byte budget")
            _require(
                type(value["data"]) is str and len(value["data"]) == 4 * ((size + 2) // 3),
                "array byte length",
            )
            raw = base64.b64decode(value["data"], validate=True)
            _require(len(raw) == size, "array bytes")
            array = np.frombuffer(raw, dtype=value["dtype"]).reshape(shape).copy()
            _require(np.isfinite(array).all(), "nonfinite training array")
            return torch.from_numpy(array) if value["$array"] == "torch" else array
        if "$tuple" in value:
            _require(set(value) == {"$tuple"} and type(value["$tuple"]) is list, "tuple schema")
            return tuple(_decode(item, budget, depth + 1) for item in value["$tuple"])
        if "$mapping" in value:
            _require(
                set(value) == {"$mapping"} and type(value["$mapping"]) is list, "mapping schema"
            )
            result = {}
            for pair in value["$mapping"]:
                _require(type(pair) is list and len(pair) == 2, "mapping pair")
                key = _decode(pair[0], budget, depth + 1)
                _require(type(key) in (int, str) and key not in result, "mapping key")
                result[key] = _decode(pair[1], budget, depth + 1)
            return result
        _require(
            all(type(key) is str and not key.startswith("$") for key in value),
            "unknown training state tag",
        )
        return {key: _decode(item, budget, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, budget, depth + 1) for item in value]
    _require(value is None or type(value) in (str, bool, int, float), "scalar type")
    _require(type(value) is not float or math.isfinite(value), "nonfinite scalar")
    _require(type(value) is not int or abs(value) < 2**256, "integer bound")
    return value


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _json_read(raw):
    def invalid(value):
        raise ValueError("nonfinite JSON literal " + value)

    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)


def _read(path, limit):
    _require(path.is_file() and not path.is_symlink(), "checkpoint component file")
    _require(path.stat().st_size <= limit, "checkpoint component byte limit")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    _require(len(raw) <= limit, "checkpoint component grew beyond byte limit")
    return raw


def _file_info(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _write(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


class MuJoCoTyxTrainingRestoreError(RuntimeError):
    def __init__(self, message, *, physics_restored):
        super().__init__(message)
        self.physics_restored = bool(physics_restored)


class MuJoCoTyxPpoSession:
    """Single synchronous owner of the fixed upstream learner and registered RNGs.

    Borrowed adapter/runtime remain caller-owned. Do not concurrently mutate any
    registered generator, global RNG, learner, adapter or runtime outside this
    owner. In-progress rollout serialization and automatic recovery are excluded.
    """

    def __init__(self, adapter, *, steps=16, recurrent=False, generators=None):
        _require(isinstance(adapter, RslMuJoCoTyxVecEnv), "device RSL adapter required")
        _require(type(steps) is int and 1 <= steps <= 64, "steps must be 1..64")
        self.adapter, self.env = adapter, adapter.env
        self.steps, self.recurrent = steps, recurrent
        self.config = ppo_config(steps=steps, recurrent=recurrent)
        self._generators = dict(generators or {})
        _require(
            all(type(key) is str and key and len(key) <= 128 for key in self._generators),
            "registered generator names",
        )
        _require(
            len(self._generators) <= 32
            and len({id(item) for item in self._generators.values()}) == len(self._generators),
            "registered generator count/alias",
        )
        self._generator_binding = {
            key: self._rng_kind(item) for key, item in self._generators.items()
        }
        self._lock, self._phase, self._poisoned = RLock(), None, False
        self.next_iteration = 0
        with self.env.training_boundary():
            self.algorithm = self._construct_algorithm(self.env.get_observations())
            self.algorithm.train_mode()
            self._binding = self._current_binding()

    @property
    def needs_recovery(self):
        return self._poisoned or self.adapter.needs_recovery

    def _ready(self):
        if self.needs_recovery:
            raise RuntimeError("failed training composition requires recovery")
        if self._phase is not None:
            raise RuntimeError("training update/restore is in progress; boundary unavailable")

    def _construct_algorithm(self, observation):
        from rsl_rl.algorithms import PPO

        observations = TensorDict(
            {"policy": observation.to(self.env.device)},
            batch_size=[self.env.num_envs],
            device=self.env.device,
        )
        return PPO.construct_algorithm(
            observations, self.adapter, copy.deepcopy(self.config), str(self.env.device)
        )

    def _rng_kind(self, generator):
        if isinstance(generator, torch.Generator):
            _require(
                generator.device.type == "cpu" or generator.device == self.env.device,
                "registered Torch generator device",
            )
            return {"kind": "torch", "device": str(generator.device)}
        if isinstance(generator, np.random.Generator):
            return {"kind": "numpy", "bit_generator": type(generator.bit_generator).__name__}
        if isinstance(generator, random.Random):
            return {"kind": "python"}
        raise ValueError("only explicit Torch/NumPy/Python generators supported")

    def _capture_rng(self):
        named = {}
        for key, generator in self._generators.items():
            if isinstance(generator, torch.Generator):
                named[key] = generator.get_state().clone()
            elif isinstance(generator, np.random.Generator):
                named[key] = _clone(generator.bit_generator.state)
            else:
                named[key] = generator.getstate()
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": torch.cuda.get_rng_state(self.env.device).clone(),
            "named": named,
        }

    def _validate_rng(self, state):
        _require(
            type(state) is dict
            and set(state) == {"python", "numpy", "torch_cpu", "torch_cuda", "named"},
            "RNG state schema",
        )
        _require(
            type(state["named"]) is dict and set(state["named"]) == set(self._generators),
            "registered RNG binding",
        )
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        for device, name in (("cpu", "torch_cpu"), (self.env.device, "torch_cuda")):
            value = state[name]
            _require(
                isinstance(value, torch.Tensor)
                and value.device.type == "cpu"
                and value.dtype == torch.uint8
                and value.ndim == 1,
                "RNG byte layout",
            )
            torch.Generator(device=device).set_state(value)
        for key, generator in self._generators.items():
            value = state["named"][key]
            if isinstance(generator, torch.Generator):
                torch.Generator(device=generator.device).set_state(value)
            elif isinstance(generator, np.random.Generator):
                candidate = copy.deepcopy(generator)
                candidate.bit_generator.state = value
            else:
                random.Random().setstate(value)

    def _apply_rng(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        torch.cuda.set_rng_state(state["torch_cuda"], self.env.device)
        for key, generator in self._generators.items():
            value = state["named"][key]
            if isinstance(generator, torch.Generator):
                generator.set_state(value)
            elif isinstance(generator, np.random.Generator):
                generator.bit_generator.state = value
            else:
                generator.setstate(value)

    def _install_rng(self, state):
        """Post-CAS switch, separated for explicit late-failure evidence."""
        self._apply_rng(state)

    def _current_binding(self):
        properties = torch.cuda.get_device_properties(self.env.device)
        return {
            "schema": SCHEMA,
            "physics": _encode(self.env.runtime.checkpoint().binding),
            "environment": self.env.training_binding(),
            "config": copy.deepcopy(self.config),
            "generators": copy.deepcopy(self._generator_binding),
            "dependencies": {
                key: version(key)
                for key in ("rsl-rl-lib", "torch", "tensordict", "numpy", "mujoco", "tyx-compute")
            },
            "python": list(sys.version_info[:3]),
            "device": {
                "index": self.env.device.index,
                "name": properties.name,
                "capability": [properties.major, properties.minor],
                "torch_cuda": torch.version.cuda,
            },
            "determinism": {
                "algorithms": torch.are_deterministic_algorithms_enabled(),
                "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            },
        }

    def _empty(self):
        if self.algorithm.storage.step != 0:
            raise RuntimeError("nonempty rollout storage is not an update boundary")
        transition = vars(self.algorithm.transition)
        for key, value in transition.items():
            if key == "hidden_states":
                valid = (
                    isinstance(value, tuple) and len(value) == 2 and all(v is None for v in value)
                )
            else:
                valid = value is None
            if not valid:
                raise RuntimeError("pending upstream Transition is not a save boundary")

    def _hidden(self, algorithm):
        return {
            name: _clone(network.get_hidden_state())
            for name, network in (("actor", algorithm.actor), ("critic", algorithm.critic))
        }

    def training_state(self):
        """Owned diagnostic copy; excludes monotonic resource/revision identities."""
        with self._lock, self.env.training_boundary():
            self._ready()
            return {
                "env": self.env.capture_training_state(),
                "learner": _clone(self.algorithm.save()),
                "hidden": self._hidden(self.algorithm),
                "rng": self._capture_rng(),
                "modes": {
                    owner: {name: module.training for name, module in network.named_modules()}
                    for owner, network in (
                        ("actor", self.algorithm.actor),
                        ("critic", self.algorithm.critic),
                    )
                },
                "next_iteration": self.next_iteration,
            }

    def update(self):
        with self._lock, self.env.training_boundary():
            self._ready()
            self._empty()
            self._phase = "rollout"
            transitions, recurrent_masks = [], []
            try:
                observation = self.adapter.get_observations()
                with torch.inference_mode():
                    for _ in range(self.steps):
                        action = self.algorithm.act(observation)
                        before_hidden = self._hidden(self.algorithm) if self.recurrent else None
                        observation, reward, done, extras = self.adapter.step(action)
                        self.algorithm.process_env_step(observation, reward, done, extras)
                        transitions.append(self.adapter.last_transition)
                        if self.recurrent:
                            recurrent_masks.append(
                                {
                                    "before": before_hidden,
                                    "after": self._hidden(self.algorithm),
                                    "done": done.detach().cpu().clone(),
                                }
                            )
                    self.algorithm.compute_returns(observation)
                self._phase = "upstream-update"
                loss = self.algorithm.update()
                _require(
                    all(math.isfinite(float(value)) for value in loss.values()), "nonfinite loss"
                )
                evidence = optimization_evidence(self.algorithm)
                self._empty()
                torch.cuda.synchronize(self.env.device)
                iteration = self.next_iteration
                self.next_iteration += 1
                return {
                    "iteration": iteration,
                    "next_iteration": self.next_iteration,
                    "transitions": tuple(transitions),
                    "recurrent_masks": tuple(recurrent_masks),
                    "loss": _clone(loss),
                    "optimization": evidence,
                }
            except BaseException:
                self._poisoned = True
                raise
            finally:
                self._phase = None

    def _validate_learner(self, candidate, state, hidden, modes, iteration):
        _require(
            type(state) is dict
            and set(state) == {"actor_state_dict", "critic_state_dict", "optimizer_state_dict"},
            "learner schema",
        )
        for key, expected in (
            ("actor_state_dict", candidate.actor.state_dict()),
            ("critic_state_dict", candidate.critic.state_dict()),
        ):
            actual = state[key]
            _require(type(actual) is dict and set(actual) == set(expected), "policy state keys")
            for name, reference in expected.items():
                value = actual[name]
                _require(
                    isinstance(value, torch.Tensor)
                    and value.device.type == "cpu"
                    and value.dtype == reference.dtype
                    and value.shape == reference.shape,
                    "policy array layout: " + name,
                )
        optimizer = state["optimizer_state_dict"]
        _require(
            type(optimizer) is dict and set(optimizer) == {"state", "param_groups"}, "Adam schema"
        )
        baseline = candidate.optimizer.state_dict()
        groups = optimizer["param_groups"]
        _require(
            type(groups) is list and len(groups) == len(baseline["param_groups"]),
            "Adam group count",
        )
        ids, parameters = [], []
        for saved, expected, live in zip(
            groups, baseline["param_groups"], candidate.optimizer.param_groups, strict=True
        ):
            _require(type(saved) is dict and set(saved) == set(expected), "Adam group keys")
            _require(saved == expected, "fixed Adam configuration/parameter order")
            ids.extend(saved["params"])
            parameters.extend(live["params"])
        _require(
            type(optimizer["state"]) is dict and set(optimizer["state"]) == set(ids),
            "complete Adam state required",
        )
        for ident, parameter in zip(ids, parameters, strict=True):
            value = optimizer["state"][ident]
            _require(
                type(value) is dict and set(value) == {"step", "exp_avg", "exp_avg_sq"},
                "Adam parameter state keys",
            )
            step = value["step"]
            _require(
                isinstance(step, torch.Tensor)
                and step.device.type == "cpu"
                and step.shape == torch.Size([])
                and step.dtype == torch.float32
                and float(step.item()) == iteration,
                "Adam step/update counter",
            )
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = value[key]
                _require(
                    isinstance(tensor, torch.Tensor)
                    and tensor.device.type == "cpu"
                    and tensor.shape == parameter.shape
                    and tensor.dtype == parameter.dtype,
                    "Adam moment layout",
                )
            _require(bool((value["exp_avg_sq"] >= 0).all().item()), "negative Adam second moment")
        _require(type(hidden) is dict and set(hidden) == {"actor", "critic"}, "RNN state schema")
        _require(type(modes) is dict and set(modes) == {"actor", "critic"}, "module mode schema")
        candidate.load(state, load_cfg=None, strict=True)
        for name, network in (("actor", candidate.actor), ("critic", candidate.critic)):
            value = hidden[name]
            if self.recurrent:
                _require(
                    isinstance(value, torch.Tensor)
                    and value.device.type == "cpu"
                    and value.dtype == torch.float32
                    and tuple(value.shape) == (1, self.env.num_envs, 16),
                    "GRU hidden layout",
                )
                network.reset(hidden_state=value.to(self.env.device).clone())
            else:
                _require(value is None, "MLP hidden must be explicitly None")
            members = dict(network.named_modules())
            _require(
                type(modes[name]) is dict
                and set(modes[name]) == set(members)
                and all(type(flag) is bool for flag in modes[name].values()),
                "module modes",
            )
            for member, module in members.items():
                module.training = modes[name][member]

    def save_checkpoint(self, path):
        with self._lock, self.env.training_boundary():
            self._ready()
            self._empty()
            if not 1 <= self.next_iteration <= 2**24:
                raise RuntimeError("checkpoint requires a completed update")
            target = Path(path)
            if target.exists():
                raise FileExistsError(target)
            _require(target.parent.is_dir() and not target.is_symlink(), "checkpoint parent")
            _require(self._current_binding() == self._binding, "live training binding changed")
            torch.cuda.synchronize(self.env.device)
            physical = self.env.runtime.checkpoint()
            state = self.training_state()
            training = _json_bytes(_encode(state))
            physics = physical.to_bytes()
            _require(
                len(training) <= MAX_TRAINING and len(physics) <= MAX_PHYSICS,
                "checkpoint byte limit",
            )
            manifest = {
                "schema": SCHEMA,
                "binding": self._binding,
                "files": {
                    "physics.bin": _file_info(physics),
                    "training.json": _file_info(training),
                },
            }
            raw_manifest = _json_bytes(manifest)
            _require(len(raw_manifest) <= MAX_MANIFEST, "manifest byte limit")
            staging = target.parent / ("." + target.name + ".pending-" + uuid.uuid4().hex)
            self._phase = "checkpoint-write"
            try:
                staging.mkdir()
                _write(staging / "physics.bin", physics)
                _write(staging / "training.json", training)
                _write(staging / "manifest.json", raw_manifest)
                self._read_bundle(staging)
                if target.exists():
                    raise FileExistsError(target)
                # On the supported Windows profile rename refuses an existing destination.
                os.rename(staging, target)
            finally:
                self._phase = None
            return {
                "path": str(target),
                "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                "next_iteration": self.next_iteration,
                "source_revision": physical.source_revision,
            }

    def _read_bundle(self, path):
        _require(path.is_dir() and not path.is_symlink(), "checkpoint directory")
        _require(
            {item.name for item in path.iterdir()}
            == {"manifest.json", "physics.bin", "training.json"},
            "checkpoint filename set",
        )
        manifest = _json_read(_read(path / "manifest.json", MAX_MANIFEST))
        _require(
            type(manifest) is dict
            and set(manifest) == {"schema", "binding", "files"}
            and manifest["schema"] == SCHEMA,
            "manifest schema",
        )
        _require(manifest["binding"] == self._binding, "training/source/device/config binding")
        files = manifest["files"]
        _require(
            type(files) is dict and set(files) == {"physics.bin", "training.json"},
            "checkpoint fixed filenames",
        )
        data = {}
        for name, cap in (("physics.bin", MAX_PHYSICS), ("training.json", MAX_TRAINING)):
            info = files[name]
            _require(
                type(info) is dict
                and set(info) == {"bytes", "sha256"}
                and type(info["bytes"]) is int
                and 0 < info["bytes"] <= cap
                and type(info["sha256"]) is str
                and len(info["sha256"]) == 64,
                "component manifest schema",
            )
            data[name] = _read(path / name, cap)
            _require(_file_info(data[name]) == info, "checkpoint component hash/size")
        checkpoint = MuJoCoTyxCheckpoint.from_bytes(data["physics.bin"])
        _require(_encode(checkpoint.binding) == self._binding["physics"], "physical source binding")
        state = _decode(_json_read(data["training.json"]), [0, 0])
        _require(
            type(state) is dict
            and set(state) == {"env", "learner", "hidden", "rng", "modes", "next_iteration"},
            "training schema",
        )
        iteration = state["next_iteration"]
        _require(type(iteration) is int and 1 <= iteration <= 2**24, "completed iteration range")
        return checkpoint, state

    def restore_checkpoint(self, path, *, expected_revision):
        with self._lock, self.env.training_boundary():
            self._ready()
            self._empty()
            revision = self.env.runtime.checkpoint_revision
            _require(
                type(expected_revision) is int and revision == expected_revision,
                "stale training CAS revision",
            )
            _require(self._current_binding() == self._binding, "live training binding changed")
            before_rng = self._capture_rng()
            self._phase = "candidate"
            try:
                try:
                    checkpoint, state = self._read_bundle(Path(path))
                    self._validate_rng(state["rng"])
                    _require(type(state["env"]) is dict, "environment state schema")
                    observation = state["env"].get("observation")
                    _require(
                        isinstance(observation, torch.Tensor)
                        and observation.device.type == "cpu"
                        and observation.dtype == torch.float32
                        and tuple(observation.shape)
                        == (self.env.num_envs, self.env.task_spec.width),
                        "candidate observation",
                    )
                    candidate = self._construct_algorithm(observation)
                    self._validate_learner(
                        candidate,
                        state["learner"],
                        state["hidden"],
                        state["modes"],
                        state["next_iteration"],
                    )
                    torch.cuda.synchronize(self.env.device)
                finally:
                    # Candidate initialization must not advance the current owner's RNG.
                    self._apply_rng(before_rng)
                self._phase = "physics-CAS"
                try:
                    receipt = self.adapter.restore_training_state(
                        checkpoint, state["env"], expected_revision=expected_revision
                    )
                    self._phase = "RNG-publication"
                    self._install_rng(state["rng"])
                    self.algorithm = candidate
                    self.next_iteration = state["next_iteration"]
                except BaseException as error:
                    changed = self.env.runtime.checkpoint_revision != revision
                    if changed:
                        self._poisoned = True
                        self.adapter.invalidate_training_state(
                            "physics restored; composite training publication failed"
                        )
                        raise MuJoCoTyxTrainingRestoreError(
                            "physics restored; training session requires recovery",
                            physics_restored=True,
                        ) from error
                    raise
                return {
                    "next_iteration": self.next_iteration,
                    "source_revision": checkpoint.source_revision,
                    "destination_revision": self.env.runtime.checkpoint_revision,
                    "physics_receipt": receipt,
                }
            finally:
                self._phase = None
