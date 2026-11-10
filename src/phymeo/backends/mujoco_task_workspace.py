"""Explicit device task outputs from accepted MuJoCo-TYX native q/v.

Own output storage never aliases physical banks. HOST actions and complete
physical diagnostic validation retain their existing contracts.
"""

from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
import json
from threading import RLock
from types import MappingProxyType
from uuid import uuid4

import numpy as np

MAX_FEATURES = 256
MAX_WORLDS = 256


def _require(value, message):
    if not value:
        raise ValueError("MuJoCo TYX task workspace: " + message)


def _frozen(value, dtype):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _indices(values, name):
    result = tuple(values)
    _require(all(type(value) is int and 0 <= value <= np.iinfo(np.int32).max for value in result),
             name + " must contain nonnegative integer native indices")
    return result


def _parameter(value, default, width, name):
    supplied = np.full(width, default, dtype=np.float64) if value is None else np.asarray(value, dtype=np.float64)
    _require(supplied.shape == (width,) and np.isfinite(supplied).all(), name + " shape/finite")
    if name == "weight":
        _require(np.all(supplied >= 0), "quadratic reward weights must be nonnegative")
    with np.errstate(over="ignore", invalid="ignore"):
        stored = supplied.astype(np.float32)
    _require(np.isfinite(stored).all(), name + " cannot be represented as finite FP32")
    return _frozen(stored, np.float32)


@dataclass(frozen=True, init=False)
class MuJoCoTyxTaskSpec:
    """Features are q_indices followed by v_indices, in original native units.

    observation = scale * x + bias; reward = -sum(weight * (x-target)^2).
    Termination means any raw native feature is outside its inclusive bounds.
    Duplicate indices are allowed and explicitly retained in schema order.
    All numeric parameters are stored FP32 and exposed read-only.
    """

    q_indices: tuple
    v_indices: tuple
    scale: np.ndarray
    bias: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    sha256: str

    def __init__(self, q_indices=(), v_indices=(), *, scale=None, bias=None,
                 target=None, weight=None, lower=None, upper=None):
        qi, vi = _indices(q_indices, "q_indices"), _indices(v_indices, "v_indices")
        width = len(qi) + len(vi)
        _require(0 < width <= MAX_FEATURES, "feature count must be 1..256")
        object.__setattr__(self, "q_indices", qi)
        object.__setattr__(self, "v_indices", vi)
        for name, supplied, default in (
            ("scale", scale, 1.0), ("bias", bias, 0.0), ("target", target, 0.0),
            ("weight", weight, 0.0), ("lower", lower, -float(np.finfo(np.float32).max)),
            ("upper", upper, float(np.finfo(np.float32).max)),
        ):
            object.__setattr__(self, name, _parameter(supplied, default, width, name))
        _require(np.all(self.weight >= 0), "quadratic reward weights must be nonnegative")
        _require(np.all(self.lower <= self.upper), "lower bound must not exceed upper bound")
        metadata = json.dumps({"schema": "mujoco-tyx-native-task-v1", "q": qi, "v": vi},
                              sort_keys=True, separators=(",", ":")).encode()
        numeric = b"".join(getattr(self, name).tobytes() for name in
                           ("scale", "bias", "target", "weight", "lower", "upper"))
        object.__setattr__(self, "sha256", sha256(metadata + b"\n" + numeric).hexdigest())

    @property
    def width(self):
        return len(self.q_indices) + len(self.v_indices)


@dataclass(frozen=True)
class MuJoCoTyxTaskFrame:
    """Immutable publication metadata; arrays remain in owned workspace storage."""

    binding: object
    revision: int
    epoch: int
    ticks: tuple
    times: tuple


class MuJoCoTyxTaskFailure(RuntimeError):
    """Task output failed; its already-accepted physics source is unchanged."""

    physics_accepted = True
    physics_mutated = False
    workspace_published = False


class MuJoCoTyxTaskWorkspace:
    """Opt-in bounded outputs, no snapshot, CPU forward, policy or auto-reset.

    refresh is synchronous and source-locked. copy_to supports an explicit TYX
    CUDA consumer stream and synchronizes it before returning. to_torch copies
    D2D to Torch-owned storage and waits for that consumer copy to complete.
    readback is an explicitly named diagnostic HOST copy, never used by refresh.
    """

    def __init__(self, engine, spec, *, cache_root=None):
        from ._mujoco_tyx_runtime import MuJoCoTyxRuntime
        from .mujoco_tyx import _task_kernels

        _require(isinstance(engine, MuJoCoTyxRuntime), "requires MuJoCo-TYX runtime")
        _require(type(spec) is MuJoCoTyxTaskSpec, "requires immutable task spec")
        with engine._lock:
            engine._check_open()
            nworld, plan = len(engine.world_keys), engine._plan
            _require(nworld <= MAX_WORLDS, "at most 256 task worlds")
            _require(all(i < plan.nq for i in spec.q_indices), "q index outside native nq")
            _require(all(i < plan.nv for i in spec.v_indices), "v index outside native nv")
            self._engine, self._spec = engine, spec
            self._mode = engine._mode
            self._device_id = int(engine._device.runtime.device)
            self._count, self._width = nworld, spec.width
            self._tx = engine._device.tx
            self._kernel = _task_kernels.evaluate
            self._lock, self._closed = RLock(), False
            self._epoch, self._active, self._frame = 0, 0, None
            source = engine._compilation.source
            self._binding = MappingProxyType({
                "schema": "mujoco-tyx-task-workspace-v1", "workspace_owner": uuid4().hex,
                "runtime_owner": id(engine), "spec_sha256": spec.sha256,
                "source_sha256": sha256(source.primary_xml).hexdigest(),
                "archive_sha256": sha256(source.archive_bytes).hexdigest(),
                "model_sha256": sha256(source.model_binary).hexdigest(),
                "world_keys": engine.world_keys, "mode": self._mode, "device": self._device_id,
                "native_layout": (plan.nq, plan.nv), "output_layout": (nworld, spec.width),
                "precision": "FP32-device-native-features",
                "actions": "existing-HOST-control-and-receipt-path",
                "physics_diagnostics": "existing-complete-HOST-validation",
            })
            self._counts = dict(task_kernels=0, source_synchronizations=0,
                                task_status_downloaded_bytes=0, explicit_output_readback_bytes=0,
                                output_d2d_bytes=0, published_frames=0, failed_refreshes=0)
            self._resources = ExitStack()
            self._runtime = self._tx.Runtime(
                backend="cuda" if self._mode == "cuda" else "cpu", device=self._device_id,
                cache_root=engine._cache_root if cache_root is None else cache_root,
            )
            self._resources.callback(self._runtime.close)
            self._stream = None
            try:
                if self._mode == "cuda":
                    self._stream = self._tx.CudaStream(device=self._device_id)
                    self._resources.callback(self._stream.close)
                with self._tx.use_runtime(self._runtime):
                    ids = np.asarray([(0, i) for i in spec.q_indices] + [(1, i) for i in spec.v_indices], dtype=np.int32)
                    parameters = np.column_stack(tuple(getattr(spec, name) for name in
                        ("scale", "bias", "target", "weight", "lower", "upper")))
                    self._ids, self._parameters = self._buffer(ids), self._buffer(parameters)
                    self._observation = tuple(self._buffer(np.zeros((nworld, spec.width), np.float32)) for _ in range(2))
                    self._reward = tuple(self._buffer(np.zeros(nworld, np.float32)) for _ in range(2))
                    self._terminated = tuple(self._buffer(np.zeros(nworld, np.int32)) for _ in range(2))
                    self._status = self._buffer(np.zeros(nworld, np.int32))
            except BaseException:
                self._resources.close()
                self._closed = True
                raise

    def _buffer(self, value):
        if self._mode == "interpret":
            return value.copy()
        result = self._tx.CudaArray(value, device=self._device_id)
        self._resources.callback(result.free)
        return result

    @property
    def spec(self):
        return self._spec

    @property
    def binding(self):
        return self._binding

    @property
    def counters(self):
        with self._lock:
            return MappingProxyType(dict(self._counts))

    def _open(self):
        if self._closed:
            raise RuntimeError("task workspace is closed")

    def refresh(self, *, expected_revision):
        """Evaluate one accepted source revision without changing physics."""
        return self._engine._refresh_task_workspace(self, expected_revision)

    def _refresh_locked(self, device, revision, ticks, times):
        # The parent runtime lock is held for the complete source read.
        with self._lock:
            self._open()
            _require(int(device.runtime.device) == self._device_id and device.mode == self._mode,
                     "source backend/device changed")
            destination = 1 - self._active
            try:
                with self._tx.use_runtime(device.runtime):
                    device.runtime.sync()
                    device.counts["synchronizations"] += 1
                    self._counts["source_synchronizations"] += 1
                with self._tx.use_runtime(self._runtime):
                    options = {"mode": self._mode}
                    if self._stream is not None:
                        options["stream"] = self._stream
                    self._kernel(
                        self._count, self._width, self._ids, self._parameters,
                        device.q[device.active], device.v[device.active],
                        self._observation[destination], self._reward[destination],
                        self._terminated[destination], self._status, **options,
                    )
                    self._counts["task_kernels"] += 1
                    if self._stream is None:
                        self._runtime.sync()
                        status = self._status.copy()
                    else:
                        self._stream.synchronize()
                        status = self._status.to_numpy(stream=self._stream)
                        self._stream.synchronize()
                        self._counts["task_status_downloaded_bytes"] += status.nbytes
                if status.shape != (self._count,) or status.dtype != np.int32 or np.any(status != 0):
                    raise RuntimeError("task kernel rejected nonfinite feature/output: " + repr(status.tolist()))
                frame = MuJoCoTyxTaskFrame(self.binding, revision, self._epoch + 1, ticks, times)
            except Exception as error:
                self._counts["failed_refreshes"] += 1
                raise MuJoCoTyxTaskFailure("task outputs rejected; accepted physics unchanged") from error
            self._active, self._epoch, self._frame = destination, frame.epoch, frame
            self._counts["published_frames"] += 1
            return frame

    def _check_frame(self, frame):
        self._open()
        self._engine._check_open()
        _require(frame is self._frame and type(frame) is MuJoCoTyxTaskFrame,
                 "foreign or superseded task frame")
        _require(frame.revision == self._engine._checkpoint_revision,
                 "task frame source revision is stale")
        return (self._observation[self._active], self._reward[self._active],
                self._terminated[self._active])

    def readback(self, frame):
        """Explicit diagnostic HOST copies; does not expose device storage."""
        with self._engine._lock, self._lock:
            values = self._check_frame(frame)
            with self._tx.use_runtime(self._runtime):
                if self._mode == "interpret":
                    result = tuple(value.copy() for value in values)
                else:
                    result = tuple(value.to_numpy(stream=self._stream) for value in values)
                    self._stream.synchronize()
                    self._counts["explicit_output_readback_bytes"] += sum(value.nbytes for value in result)
            return tuple(_frozen(value, value.dtype) for value in result)

    def _overlaps(self, left, right):
        if self._mode == "interpret":
            return np.shares_memory(left, right)
        a, b = int(left.device_ptr), int(right.device_ptr)
        return a < b + right.nbytes and b < a + left.nbytes

    def copy_to(self, frame, observation, reward, terminated, *, stream=None):
        """Synchronous copies to caller-owned arrays, optionally on another stream."""
        with self._engine._lock, self._lock:
            values = self._check_frame(frame)
            targets = (observation, reward, terminated)
            shapes = ((self._count, self._width), (self._count,), (self._count,))
            dtypes = (np.dtype(np.float32), np.dtype(np.float32), np.dtype(np.int32))
            if self._mode == "interpret":
                _require(stream is None, "interpret copies cannot use a device stream")
                for target, shape, dtype in zip(targets, shapes, dtypes, strict=True):
                    _require(isinstance(target, np.ndarray) and target.shape == shape
                             and target.dtype == dtype and target.flags.c_contiguous
                             and target.flags.writeable, "output dtype/shape/storage")
            else:
                if stream is not None:
                    _require(isinstance(stream, self._tx.CudaStream)
                             and stream.device_ordinal == self._device_id, "consumer stream device")
                    stream._ensure_alive()
                for target, shape, dtype in zip(targets, shapes, dtypes, strict=True):
                    _require(isinstance(target, self._tx.CudaArray) and target.shape == shape
                             and np.dtype(target.host_dtype) == dtype
                             and target.device_ordinal == self._device_id, "output dtype/shape/device")
                    target._ensure_alive()
            device = self._engine._device
            physical = []
            for value in vars(device).values():
                items = value if isinstance(value, (tuple, list)) else (value,)
                for item in items:
                    if isinstance(item, np.ndarray if self._mode == "interpret" else self._tx.CudaArray):
                        physical.append(item)
            forbidden = (*physical, *self._observation, *self._reward, *self._terminated,
                         self._ids, self._parameters, self._status)
            for i, target in enumerate(targets):
                _require(not any(self._overlaps(target, other) for other in forbidden),
                         "output must not alias physical or workspace storage")
                _require(not any(self._overlaps(target, other) for other in targets[:i]),
                         "caller outputs overlap")
            with self._tx.use_runtime(self._runtime):
                for target, value in zip(targets, values, strict=True):
                    if self._mode == "interpret":
                        np.copyto(target, value)
                    else:
                        target.copy_from_device(value, stream=stream)
                        self._counts["output_d2d_bytes"] += value.nbytes
                if self._mode == "cuda":
                    if stream is None:
                        self._runtime.sync()
                    else:
                        stream.synchronize()

    def to_torch(self, frame):
        """Copy-only CUDA export, with consumer completion before returning.

        This method explicitly rejects CPU-only PyTorch, and never stages
        observations through HOST. Returned tensors have independent ownership.
        """
        with self._engine._lock, self._lock:
            values = self._check_frame(frame)
            if self._mode != "cuda":
                raise RuntimeError("Torch device export requires a CUDA workspace")
            try:
                import torch
                from tyx.torch import to_torch
            except ImportError as error:
                raise RuntimeError("CUDA PyTorch is required for task tensor export") from error
            if torch.version.cuda is None or not torch.cuda.is_available():
                raise RuntimeError("CUDA PyTorch is unavailable; no CPU fallback")
            consumer = torch.cuda.current_stream(device=self._device_id)
            try:
                with torch.cuda.device(self._device_id):
                    result = tuple(to_torch(value, copy=True, stream=self._stream) for value in values)
            finally:
                consumer.synchronize()
            self._counts["output_d2d_bytes"] += sum(value.nbytes for value in values)
            return result

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._stream is not None:
                    self._stream.synchronize()
                else:
                    self._runtime.sync()
            finally:
                self._resources.close()

    def __enter__(self):
        with self._lock:
            self._open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
