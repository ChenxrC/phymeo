"""Copy-owned native device controls, with canonical HOST audit and CAS publication.

No policy scaling, actuator clipping, physics or implicit partial-world writes.
CUDA subnormal rejection reads the original copied bits before FP32 arithmetic.
"""

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

import numpy as np

from .mujoco_runtime import MuJoCoControl


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class MuJoCoTyxDeviceControlReceipt:
    binding: object
    source_revision: int
    revision: int
    epoch: int
    controls: object


def _arrays(owner, tx, mode):
    found = []
    for value in vars(owner).values():
        items = value if isinstance(value, (tuple, list)) else (value,)
        for item in items:
            if isinstance(item, np.ndarray if mode == "interpret" else tx.CudaArray):
                if mode == "cuda":
                    try:
                        item._ensure_alive()
                    except Exception:
                        continue
                found.append(item)
    return found


def _overlap(left, right, mode):
    if mode == "interpret":
        return np.shares_memory(left, right)
    a, b = int(left.device_ptr), int(right.device_ptr)
    return a < b + right.nbytes and b < a + left.nbytes


def _preflight(engine, values, stream):
    tx, mode = engine._device.tx, engine._mode
    plan, count = engine._plan, len(engine.world_keys)
    ordinal = int(engine._device.runtime.device)
    shapes = ((count, plan.nu), (count, plan.nv), (count, engine._model.nbody, 6))
    _require((values[0] is None) == (plan.nu == 0), "ctrl is None exactly when native nu is zero")
    if mode == "interpret":
        _require(stream is None, "interpret cannot use a producer stream")
    elif stream is not None:
        _require(
            isinstance(stream, tx.CudaStream) and stream.device_ordinal == ordinal,
            "producer stream device/type",
        )
        stream._ensure_alive()
    supplied = []
    for value, shape in zip(values, shapes, strict=True):
        if value is None:
            continue
        if mode == "interpret":
            _require(
                isinstance(value, np.ndarray)
                and value.dtype == np.float32
                and value.shape == shape
                and value.flags.c_contiguous,
                "control dtype/shape/storage",
            )
        else:
            _require(
                isinstance(value, tx.CudaArray)
                and value.shape == shape
                and np.dtype(value.host_dtype) == np.float32
                and value.device_ordinal == ordinal,
                "control dtype/shape/device",
            )
            value._ensure_alive()
        supplied.append(value)
    forbidden = _arrays(engine._device, tx, mode)
    for owner in (engine._control_packet, *engine._control_retired):
        if owner is not None:
            forbidden.extend(_arrays(owner, tx, mode))
    for i, value in enumerate(supplied):
        _require(
            not any(_overlap(value, old, mode) for old in forbidden),
            "control input aliases physical/action storage",
        )
        _require(
            not any(_overlap(value, old, mode) for old in supplied[:i]),
            "supplied control channels overlap",
        )


class _DeviceControlPacket:
    """A staged owner becomes trusted only through the engine's single CAS commit."""

    def __init__(self, engine):
        self.engine, self.device_owner = engine, engine._device
        self.tx, self.mode = engine._device.tx, engine._mode
        self.ordinal = int(engine._device.runtime.device)
        self.plan, self.count, self.nbody = (
            engine._plan,
            len(engine.world_keys),
            engine._model.nbody,
        )
        self.epoch = engine._control_epoch + 1
        self.resources, self.buffers = [], []
        self.closed = False
        self.counts = dict(
            control_copy_d2d_bytes=0,
            control_audit_downloaded_bytes=0,
            control_status_downloaded_bytes=0,
            control_initial_uploaded_bytes=0,
            control_pack_kernels=0,
        )
        source = engine._compilation.source
        self.binding = MappingProxyType(
            {
                "schema": "mujoco-tyx-device-controls-v1",
                "runtime_owner": id(engine),
                "source_sha256": sha256(source.primary_xml).hexdigest(),
                "archive_sha256": sha256(source.archive_bytes).hexdigest(),
                "model_sha256": sha256(source.model_binary).hexdigest(),
                "world_keys": engine.world_keys,
                "mode": self.mode,
                "device": self.ordinal,
                "native_layout": (self.plan.nq, self.plan.nv, self.plan.nu, self.nbody),
                "epoch": self.epoch,
                "precision": "FP32-native-requested-controls",
                "world_order": "caller-declared-runtime-order-not-inferred",
            }
        )

    def _buffer(self, shape, dtype=np.float32):
        value = np.zeros(shape, dtype)
        if self.mode == "cuda":
            value = self.tx.CudaArray(value, device=self.ordinal)
            self.resources.append(value.free)
            self.counts["control_initial_uploaded_bytes"] += value.nbytes
        self.buffers.append(value)
        return value

    def _copy_native(self, destination, source):
        if source is None:
            return
        if self.mode == "interpret":
            np.copyto(destination, source)
        else:
            destination.copy_from_device(source, stream=self.stream)
            self.counts["control_copy_d2d_bytes"] += source.nbytes

    def _audit(self):
        if self.mode == "interpret":
            values = tuple(value.copy() for value in self.native)
        else:
            collected = []
            for value in self.native:
                collected.append(value.to_numpy(stream=self.stream))
                self.counts["control_audit_downloaded_bytes"] += value.nbytes
            self.stream.synchronize()
            values = tuple(collected)
        # Inspect raw input bits before any arithmetic can flush a subnormal.
        for value in values:
            words = value.view(np.uint32)
            exponent, mantissa = words & 0x7F800000, words & 0x007FFFFF
            _require(not np.any(exponent == 0x7F800000), "device action must be finite")
            _require(
                not np.any((exponent == 0) & (mantissa != 0)),
                "device action must be normal-or-zero FP32",
            )
        return values

    def _pack(self):
        from .mujoco_tyx._control_kernels import pack

        options = {"mode": self.mode}
        if self.stream is not None:
            options["stream"] = self.stream
        pack(
            self.count,
            self.plan.nu,
            self.plan.nv,
            self.nbody,
            max(8, self.plan.nv),
            *self.native,
            self.padded,
            self.status,
            **options,
        )
        self.counts["control_pack_kernels"] += 1
        if self.mode == "interpret":
            self.runtime.sync()
            status = self.status.copy()
        else:
            self.stream.synchronize()
            status = self.status.to_numpy(stream=self.stream)
            self.stream.synchronize()
            self.counts["control_status_downloaded_bytes"] += status.nbytes
        _require(
            status.shape == (self.count,) and status.dtype == np.int32 and not np.any(status),
            "device control pack rejected",
        )

    def prepare(self, values, producer_stream):
        self.runtime = self.tx.Runtime(
            backend="cuda" if self.mode == "cuda" else "cpu",
            device=self.ordinal,
            cache_root=self.engine._cache_root,
        )
        self.resources.append(self.runtime.close)
        self.stream = None
        if self.mode == "cuda":
            self.stream = self.tx.CudaStream(device=self.ordinal)
            self.resources.append(self.stream.close)
        with self.tx.use_runtime(self.runtime):
            if producer_stream is not None:
                producer_stream.synchronize()
            elif self.mode == "cuda":
                # No asynchronous producer may be hidden behind stream=None.
                self.runtime.sync()
            self.native = (
                self._buffer((self.count, max(1, self.plan.nu))),
                self._buffer((self.count, self.plan.nv)),
                self._buffer((self.count, self.nbody, 6)),
            )
            self.padded = self._buffer((self.count, max(8, self.plan.nv)))
            self.status = self._buffer((self.count,), np.int32)
            for destination, value in zip(self.native, values, strict=True):
                self._copy_native(destination, value)
            if self.stream is not None:
                self.stream.synchronize()
            audit = self._audit()
            self._pack()
        self.controls = {
            key: MuJoCoControl(
                audit[0][i, : self.plan.nu], qfrc_applied=audit[1][i], xfrc_applied=audit[2][i]
            )
            for i, key in enumerate(self.engine.world_keys)
        }

    def inputs(self, engine):
        _require(
            engine is self.engine and engine._device is self.device_owner,
            "device action packet owner changed",
        )
        _require(
            not self.closed and engine._control_packet is self,
            "device action packet is not the accepted owner",
        )
        _require(engine._control_epoch == self.epoch, "device action packet epoch changed")
        _require(engine._controls is self.controls, "device action canonical controls changed")
        return (*self.native[:1], self.padded, self.native[2], self.controls)

    def close(self):
        self.closed = True
        failures, errors = [], []
        stream = getattr(self, "stream", None)
        if stream is not None and self.resources:
            try:
                stream.synchronize()
            except BaseException as error:
                errors.append(error)
        # Keep the original owner list until all attempts have been made. A cleanup
        # exception cannot make another still-live resource disappear from ownership.
        for closer in reversed(self.resources):
            try:
                closer()
            except BaseException as error:
                failures.append(closer)
                errors.append(error)
        self.resources = list(reversed(failures))
        if errors:
            raise BaseExceptionGroup("device control packet cleanup failed", errors)


def _cleanup_retired(engine):
    """Never escape after a completed action/state publication."""
    try:
        remaining, errors = [], []
        for owner in engine._control_retired:
            try:
                owner.close()
            except BaseException as error:
                remaining.append(owner)
                errors.append(repr(error))
        engine._control_retired = remaining
        engine._control_cleanup_errors = tuple(errors)
    except BaseException:
        # The original list still owns all entries until the final replacement.
        # Leave it intact if cleanup bookkeeping itself cannot complete.
        return


def invalidate(engine):
    """Called only after a successful HOST-control/reset/restore publication."""
    old = engine._control_packet
    if old is not None:
        # Callers pre-build this list before their commit via retirement_list().
        engine._control_packet = None
    _cleanup_retired(engine)


def retirement_list(engine):
    return [
        *engine._control_retired,
        *(() if engine._control_packet is None else (engine._control_packet,)),
    ]


def submit(engine, values, expected_revision, producer_stream):
    _require(
        type(expected_revision) is int and expected_revision == engine._checkpoint_revision,
        "stale device action source revision",
    )
    _preflight(engine, values, producer_stream)
    packet = _DeviceControlPacket(engine)
    # Reserve ownership before any device allocation, so a failed staging cleanup
    # has a durable place even if subsequent preparation raises.
    failed_retirement = [*engine._control_retired, packet]
    try:
        packet.prepare(values, producer_stream)
        revision = engine._checkpoint_revision + 1
        receipt = MuJoCoTyxDeviceControlReceipt(
            packet.binding,
            expected_revision,
            revision,
            packet.epoch,
            MappingProxyType(dict(packet.controls)),
        )
        retired = retirement_list(engine)
        counts = dict(engine._extra_counts)
        for name, value in packet.counts.items():
            counts[name] = counts.get(name, 0) + value
        counts["device_control_submissions"] = counts.get("device_control_submissions", 0) + 1
    except BaseException:
        engine._control_retired = failed_retirement
        for name, value in packet.counts.items():
            engine._extra_counts[name] = engine._extra_counts.get(name, 0) + value
        _cleanup_retired(engine)
        raise
    # Nothing below can reject a submitted action. The old packet is retired,
    # and any cleanup fault remains visible separately from this accepted receipt.
    engine._controls, engine._control_packet = packet.controls, packet
    engine._control_epoch, engine._checkpoint_revision = packet.epoch, revision
    engine._control_retired, engine._extra_counts = retired, counts
    _cleanup_retired(engine)
    return receipt
