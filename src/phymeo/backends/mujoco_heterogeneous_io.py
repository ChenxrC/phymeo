"""Public, copy-owned heterogeneous native-state CUDA IO.

State columns retain native addresses, never task IDs. Endpoint integration
banks are distinct from entry-phase geometry/contact scratch. Physics and
control audits remain those of the resident runtimes.
"""

from contextlib import ExitStack
from dataclasses import dataclass
from itertools import accumulate
from threading import RLock
from types import MappingProxyType, SimpleNamespace
from uuid import uuid4

import numpy as np

from .area_friction_law import law_binding
from .mujoco_device_controls import _arrays, _overlap, _require
from .mujoco_runtime import _frozen, _world_keys


@dataclass(frozen=True)
class MuJoCoTyxHeterogeneousFrame:
    binding: object
    revision: int
    epoch: int
    ticks: tuple
    world_boundaries: tuple = ()
    area_catalogs: object = None


@dataclass(frozen=True, eq=False)
class MuJoCoTyxBodyPoseSnapshot:
    """Owned contiguous FP32 [body, 3, 4] rows of body-to-world [R | p in metres].

    For world row w and native body id b, use transforms[body_offsets[w] + b].
    Each interval contains exactly that world's bodies including its world body.
    The frame carries the source revision, observation epoch, ticks and boundaries.
    """

    frame: MuJoCoTyxHeterogeneousFrame
    body_offsets: tuple[int, ...]
    transforms: object

    @property
    def world_keys(self):
        return self.frame.binding["world_keys"]


@dataclass(frozen=True)
class MuJoCoTyxHeterogeneousControlReceipt:
    schema: str
    source_revision: int
    revision: int
    world_keys: tuple
    receipts: tuple


def _dimensions(batch):
    return {
        name: max(1, max(getattr(row, name) for row in batch.layout.values()))
        for name in ("nq", "nv", "nu", "na", "nbody")
    }


def _metadata(batch):
    result = {}
    for spec in batch._models:
        model = batch._buckets[
            batch.layout[next(w.world for w in batch._worlds if w.model_key == spec.key)].bucket
        ]._model
        fields = {
            name: _frozen(getattr(model, name), dtype=getattr(model, name).dtype)
            for name in (
                "jnt_type",
                "jnt_qposadr",
                "jnt_dofadr",
                "jnt_bodyid",
                "jnt_axis",
                "jnt_pos",
                "jnt_limited",
                "jnt_range",
                "body_parentid",
                "body_jntadr",
                "body_jntnum",
                "geom_bodyid",
                "body_pos",
                "body_quat",
                "body_mass",
                "body_inertia",
                "actuator_trnid",
                "actuator_trntype",
                "actuator_ctrllimited",
                "actuator_ctrlrange",
            )
        }
        fields["joint_names"] = tuple(model.joint(i).name for i in range(model.njnt))
        fields["body_names"] = tuple(model.body(i).name for i in range(model.nbody))
        fields["geom_names"] = tuple(model.geom(i).name for i in range(model.ngeom))
        fields["actuator_names"] = tuple(model.actuator(i).name for i in range(model.nu))
        result[spec.key] = MappingProxyType(fields)
    return MappingProxyType(result)


class MuJoCoTyxHeterogeneousWorkspace:
    """Double-buffered native q/v/act/clock, with authoritative validity masks.

    refresh never invokes snapshot, native forward or a CPU physics solver.
    Zero-width native fields use one zero sentinel column with a false mask.
    FREE q is world translation + WXYZ, FREE v world linear + local angular.
    """

    def __init__(
        self,
        batch,
        *,
        cache_root=None,
        include_bodies=False,
        include_contacts=False,
        include_area_pressure=False,
    ):
        from .mujoco_tyx._heterogeneous_kernels import scatter
        from .mujoco_tyx._resources import DeviceResources

        _require(type(include_bodies) is bool, "include_bodies must be bool")
        _require(type(include_contacts) is bool, "include_contacts must be bool")
        _require(type(include_area_pressure) is bool, "include_area_pressure must be bool")
        include_contacts = include_contacts or include_area_pressure
        self._area_observer = None
        self._entry_observer = None
        self._body_observer = None
        self._batch, self._lock, self._closed = batch, RLock(), False
        self._storage = []
        self._resources, self._dependencies = DeviceResources(), DeviceResources()
        self._frame, self._active, self._epoch = None, 0, 0
        self._refresh_failed = False
        self._counts = dict(
            refreshes=0,
            failed_refreshes=0,
            kernels=0,
            status_downloaded_bytes=0,
            explicit_readback_bytes=0,
            output_d2d_bytes=0,
            body_pose_snapshots=0,
            body_pose_layout_upload_bytes=0,
            allocated_payload_bytes=0,
        )
        with batch._transaction():
            self._tx = batch._buckets[0]._device.tx
            self._mode = batch._mode
            self._ordinal = int(batch._buckets[0]._device.runtime.device)
            _require(
                all(int(o._device.runtime.device) == self._ordinal for o in batch._buckets),
                "heterogeneous devices differ",
            )
            self._dimensions = MappingProxyType(_dimensions(batch))
            self._metadata = _metadata(batch)
            self._scatter = scatter
            self._binding = MappingProxyType(
                dict(
                    schema="mujoco-tyx-heterogeneous-native-state-v1",
                    workspace=uuid4().hex,
                    world_keys=batch.world_keys,
                    mode=self._mode,
                    device=self._ordinal,
                    dimensions=self._dimensions,
                    source=batch._binding,
                    phase="accepted-integration-endpoint",
                    free_q="world-translation,WXYZ",
                    free_v="world-linear,body-local-angular",
                    clock="FP32 high-minus-compensation",
                    padding="zero-with-int32-valid-mask",
                    contact_tokens=include_contacts,
                    area_pressure=include_area_pressure,
                    area_pair_names=MappingProxyType(
                        {spec.key: spec.area_pressure_pairs for spec in batch._models}
                    )
                    if include_area_pressure
                    else None,
                    area_capacity=max(
                        [
                            1,
                            *(
                                o._constraint_capacity
                                for o in batch._buckets
                                if o._area_pressure_pairs
                            ),
                        ]
                    )
                    if include_area_pressure
                    else 0,
                    area_schema=(
                        "mujoco-tyx-area-device-entry-v2"
                        if any(s.area_friction_laws for s in batch._models)
                        else "mujoco-tyx-area-device-entry-v1"
                    )
                    if include_area_pressure
                    else None,
                    **(
                        {
                            "area_laws": MappingProxyType(
                                {s.key: law_binding(s.area_friction_laws) for s in batch._models}
                            )
                        }
                        if include_area_pressure
                        and any(s.area_friction_laws for s in batch._models)
                        else {}
                    ),
                    area_force="on-compliant-A; B opposite" if include_area_pressure else None,
                    area_moment="common-world-origin; area_ids[...,5] validity"
                    if include_area_pressure
                    else None,
                    area_measure="little-endian binary64 low32,high32 in i32 words"
                    if include_area_pressure
                    else None,
                    entry_phase="accepted-step-input" if include_contacts else None,
                    entry_body_kinematics=include_contacts,
                    contact_frame="rows-of-contact-axes-in-world",
                    contact_wrench_world="on-geometry2-at-contact-point:force,couple",
                    contact_body_wrench="on-body1,body2-at-body-origin:world-force,moment",
                    entry_meta=(
                        "valid",
                        "input_tick",
                        "output_tick",
                        "detected",
                        "active",
                        "overflow",
                    ),
                    entry_contact_ids=(
                        "valid",
                        "active",
                        "geom1",
                        "geom2",
                        "body1",
                        "body2",
                        "feature",
                        "condim",
                    ),
                    entry_contact_data=(
                        "distance",
                        "point3",
                        "frame-rows9",
                        "contact-wrench6",
                        "include_margin",
                    ),
                    entry_contact_capacity=max(1, max(o._contact_capacity for o in batch._buckets))
                    if include_contacts
                    else 0,
                    endpoint_body_kinematics=include_bodies,
                    body_velocity="world-linear-and-angular-at-body-origin",
                    body_rotation="body-to-world-matrix",
                )
            )
            batch._workspaces.add(self)
            try:
                self._runtime = self._tx.Runtime(
                    backend="cuda" if self._mode == "cuda" else "cpu",
                    device=self._ordinal,
                    cache_root=batch._cache_root if cache_root is None else cache_root,
                )
                self._dependencies.set_runtime(self._runtime)
                self._resources.set_runtime(self._dependencies)
                self._stream = None
                if self._mode == "cuda":
                    self._stream = self._tx.CudaStream(device=self._ordinal)
                    self._dependencies.add_buffer(SimpleNamespace(free=self._stream.close))
                n = len(batch.world_keys)
                self._rows = tuple(
                    self._buffer(
                        np.asarray([batch.world_keys.index(k) for k in owner.world_keys], np.int32)
                    )
                    for owner in batch._buckets
                )
                self._status = tuple(
                    self._buffer(np.zeros(len(o.world_keys), np.int32)) for o in batch._buckets
                )
                self._banks = tuple(
                    {
                        name: self._buffer(np.zeros((n, width), np.float32))
                        for name, width in (
                            ("q", self._dimensions["nq"]),
                            ("v", self._dimensions["nv"]),
                            ("act", self._dimensions["na"]),
                            ("clock", 2),
                        )
                    }
                    for _ in range(2)
                )
                self._masks = {}
                for name, dim in (
                    ("q", "nq"),
                    ("v", "nv"),
                    ("act", "na"),
                    ("ctrl", "nu"),
                    ("body", "nbody"),
                ):
                    value = np.zeros((n, self._dimensions[dim]), np.int32)
                    for i, key in enumerate(batch.world_keys):
                        value[i, : getattr(batch.layout[key], dim)] = 1
                    self._masks[name + "_mask"] = self._buffer(value)
                if include_bodies:
                    from .mujoco_tyx._body_observation import BodyObserver

                    self._body_observer = BodyObserver(self)
                    counts = (batch.layout[k].nbody for k in batch.world_keys)
                    self._pose_offsets = tuple(accumulate(counts, initial=0))
                    _require(self._pose_offsets[-1] <= np.iinfo(np.int32).max, "body index overflow")
                    offsets = np.asarray(self._pose_offsets, dtype=np.int32)
                    self._pose_rows = self._buffer(offsets)
                    if self._mode == "cuda":
                        self._counts["body_pose_layout_upload_bytes"] = offsets.nbytes
                if include_contacts:
                    from .mujoco_tyx._entry_observation import EntryObserver

                    self._entry_observer = EntryObserver(self)
                if include_area_pressure:
                    from .mujoco_tyx._area_observation import AreaObserver

                    self._area_observer = AreaObserver(self)
            except BaseException as primary:
                try:
                    self.close()
                except BaseException as cleanup:
                    raise BaseExceptionGroup(
                        "workspace construction and cleanup", [primary, cleanup]
                    ) from None
                raise

    def _buffer(self, value):
        if self._mode == "cuda":
            value = self._tx.CudaArray(value, device=self._ordinal)
            self._resources.add_buffer(value)
        self._storage.append(value)
        self._counts["allocated_payload_bytes"] += value.nbytes
        return value

    @property
    def binding(self):
        return self._binding

    @property
    def model_metadata(self):
        return self._metadata

    @property
    def counters(self):
        with self._lock:
            return MappingProxyType(dict(self._counts))

    def _open(self):
        if self._closed:
            raise RuntimeError("heterogeneous workspace is closed")

    def refresh(self, *, expected_revision):
        batch = self._batch
        with batch._transaction(), self._lock:
            self._open()
            _require(
                type(expected_revision) is int and expected_revision == batch._revision,
                "stale heterogeneous observation revision",
            )
            destination = 1 - self._active
            try:
                for i, owner in enumerate(batch._buckets):
                    device = owner._device
                    _require(int(device.runtime.device) == self._ordinal, "source device changed")
                    with self._tx.use_runtime(device.runtime):
                        device.runtime.sync()
                    fields = (
                        ("q", device.q[device.active], owner._plan.nq),
                        ("v", device.v[device.active], owner._plan.nv),
                        (
                            "act",
                            device.act[device.active]
                            if getattr(device, "activation_enabled", False)
                            else device.q[device.active],
                            owner._model.na,
                        ),
                        ("clock", device.time[device.active], 2),
                    )
                    with self._tx.use_runtime(self._runtime):
                        options = dict(mode=self._mode)
                        if self._stream is not None:
                            options["stream"] = self._stream
                        for name, source, width in fields:
                            target = self._banks[destination][name]
                            self._scatter(
                                len(owner.world_keys),
                                width,
                                target.shape[1],
                                self._rows[i],
                                source,
                                target,
                                self._status[i],
                                **options,
                            )
                            self._counts["kernels"] += 1
                            if self._stream is None:
                                self._runtime.sync()
                                status = self._status[i].copy()
                            else:
                                self._stream.synchronize()
                                status = self._status[i].to_numpy(stream=self._stream)
                                self._stream.synchronize()
                                self._counts["status_downloaded_bytes"] += status.nbytes
                            _require(not np.any(status), "nonfinite accepted native state")
                    if self._body_observer is not None:
                        self._body_observer.refresh_bucket(i, owner, destination, options)
                    if self._entry_observer is not None:
                        self._entry_observer.refresh_bucket(i, owner, destination, options)
                    if self._area_observer is not None:
                        self._area_observer.refresh_bucket(i, owner, destination, options)
                frame = MuJoCoTyxHeterogeneousFrame(
                    self.binding,
                    batch._revision,
                    self._epoch + 1,
                    tuple(
                        batch._buckets[batch.layout[k].bucket]._ticks[k] for k in batch.world_keys
                    ),
                    tuple(batch._world_boundaries[k] for k in batch.world_keys),
                    self._area_observer.catalogs() if self._area_observer is not None else None,
                )
            except BaseException:
                self._counts["failed_refreshes"] += 1
                self._refresh_failed = True
                raise
            self._refresh_failed = False
            self._active, self._epoch, self._frame = destination, frame.epoch, frame
            self._counts["refreshes"] += 1
            return frame

    def _check_current(self, frame):
        self._open()
        self._batch._check()
        _require(
            type(frame) is MuJoCoTyxHeterogeneousFrame and frame is self._frame,
            "foreign or superseded heterogeneous frame",
        )
        _require(frame.revision == self._batch._revision, "stale heterogeneous frame")

    def accepted_frame_time(self, frame, world_key):
        """Read the already committed host clock, rejecting stale or foreign observations."""
        with self._batch._lock, self._lock:
            self._check_current(frame)
            _require(not self._refresh_failed, "new exposure requires a successful observation refresh")
            owner = self._batch._buckets[self._batch.layout[world_key].bucket]
            return float(owner._data[world_key].time)

    def _values(self, frame):
        self._check_current(frame)
        return {**self._banks[self._active], **self._masks}

    def readback(self, frame):
        """Explicit diagnostic HOST copy; never used to construct device output."""
        with self._batch._lock, self._lock:
            values = self._values(frame)
            with self._tx.use_runtime(self._runtime):
                result = {
                    k: v.copy() if self._mode == "interpret" else v.to_numpy(stream=self._stream)
                    for k, v in values.items()
                }
                if self._stream is not None:
                    self._stream.synchronize()
                    self._counts["explicit_readback_bytes"] += sum(
                        v.nbytes for v in result.values()
                    )
            return MappingProxyType({k: _frozen(v, dtype=v.dtype) for k, v in result.items()})

    def to_torch(self, frame):
        """Fresh Torch-owned CUDA storage via the shared TYX D2D bridge."""
        with self._batch._lock, self._lock:
            values = self._values(frame)
            if self._mode != "cuda":
                raise RuntimeError("Torch export requires CUDA; no HOST fallback")
            import torch
            from tyx.torch import to_torch

            if torch.version.cuda is None or not torch.cuda.is_available():
                raise RuntimeError("CUDA Torch is required")
            with torch.cuda.device(self._ordinal):
                consumer = torch.cuda.current_stream(self._ordinal)
                try:
                    result = {
                        k: to_torch(v, copy=True, stream=self._stream) for k, v in values.items()
                    }
                finally:
                    consumer.synchronize()
            self._counts["output_d2d_bytes"] += sum(v.nbytes for v in values.values())
            return MappingProxyType(result)

    def body_pose_snapshot(self, frame):
        """Pack only valid body transforms from a published include_bodies frame.

        CUDA writes directly into independent Torch CUDA storage on the caller's
        current stream and waits before releasing the source bank. Interpret mode
        returns an independent NumPy array. Both survive refresh, reset and close.
        """
        from .mujoco_tyx._body_observation_kernels import pack_body_transforms

        with self._batch._lock, self._lock:
            values = self._values(frame)
            _require(self._body_observer is not None, "body poses require include_bodies=True")
            args = (
                len(self._batch.world_keys),
                self._pose_rows,
                values["body_position"],
                values["body_rotation"],
            )
            shape = (self._pose_offsets[-1], 3, 4)
            with self._tx.use_runtime(self._runtime):
                if self._mode == "interpret":
                    transforms = np.empty(shape, np.float32)
                    pack_body_transforms(*args, transforms, mode="interpret")
                else:
                    import torch
                    from tyx.torch import from_torch, stream_from_torch

                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA Torch is required for a CUDA body pose snapshot")
                    with torch.cuda.device(self._ordinal), ExitStack() as resources:
                        transforms = torch.empty(shape, dtype=torch.float32, device=f"cuda:{self._ordinal}")
                        view = from_torch(transforms, backend="cuda")
                        resources.callback(view.free)
                        stream = stream_from_torch(transforms, backend="cuda")
                        resources.callback(stream.close)
                        try:
                            pack_body_transforms(*args, view, mode="cuda", stream=stream)
                        finally:
                            stream.synchronize()
                    self._counts["output_d2d_bytes"] += transforms.numel() * transforms.element_size()
            self._counts["kernels"] += 1
            self._counts["body_pose_snapshots"] += 1
            return MuJoCoTyxBodyPoseSnapshot(frame, self._pose_offsets, transforms)

    def close(self):
        with self._batch._lock, self._lock:
            self._closed = True
            if self._resources.closed:
                return
            # An unknown native release is sticky; never reuse its stream or
            # invoke its callback again. Shared ledgers retain all dependencies.
            if self._resources.state == "OPEN":
                stream = getattr(self, "_stream", None)
                if stream is not None:
                    stream.synchronize()
            self._resources.close()
            self._storage = []
            self._batch._workspaces.discard(self)

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if exc_value is not None:
                raise BaseExceptionGroup(
                    "workspace body and cleanup", [exc_value, cleanup]
                ) from None
            raise


def submit_controls(batch, values, world_keys, expected_revision, producer_stream):
    """Stage/audit every native packet, then commit all buckets together."""
    from . import mujoco_device_controls as controls

    with batch._transaction():
        _require(
            type(expected_revision) is int and expected_revision == batch._revision,
            "stale heterogeneous action revision",
        )
        _require(_world_keys(world_keys) == batch.world_keys, "device controls world order differs")
        tx, mode = batch._buckets[0]._device.tx, batch._mode
        ordinal = int(batch._buckets[0]._device.runtime.device)
        dimensions, count = _dimensions(batch), len(batch.world_keys)
        has_ctrl = any(row.nu for row in batch.layout.values())
        _require((values[0] is not None) == has_ctrl, "ctrl is None exactly when all nu are zero")
        if mode == "interpret":
            _require(producer_stream is None, "interpret cannot use producer stream")
        elif producer_stream is not None:
            _require(
                isinstance(producer_stream, tx.CudaStream)
                and producer_stream.device_ordinal == ordinal,
                "producer stream device/type",
            )
            producer_stream._ensure_alive()
        shapes = (
            (count, dimensions["nu"]),
            (count, dimensions["nv"]),
            (count, dimensions["nbody"], 6),
        )
        forbidden = []
        for owner in batch._buckets:
            _require(int(owner._device.runtime.device) == ordinal, "heterogeneous devices differ")
            forbidden.extend(_arrays(owner._device, tx, mode))
            for packet in (owner._control_packet, *owner._control_retired):
                if packet is not None:
                    forbidden.extend(_arrays(packet, tx, mode))
        for workspace in batch._workspaces:
            forbidden.extend(_arrays(workspace, tx, mode))
        supplied = []
        for value, shape in zip(values, shapes, strict=True):
            if value is None:
                continue
            if mode == "interpret":
                _require(
                    isinstance(value, np.ndarray)
                    and value.shape == shape
                    and value.dtype == np.float32
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
            _require(
                not any(_overlap(value, old, mode) for old in (*forbidden, *supplied)),
                "controls alias physical/workspace storage or overlap",
            )
            supplied.append(value)
        drafts = []
        try:
            for owner in batch._buckets:
                owner._check_open()
                drafts.append(
                    controls.prepare(
                        owner,
                        values,
                        owner._checkpoint_revision,
                        producer_stream,
                        source_rows=tuple(batch.world_keys.index(k) for k in owner.world_keys),
                    )
                )
            receipt = MuJoCoTyxHeterogeneousControlReceipt(
                "mujoco-tyx-heterogeneous-device-controls-v1",
                batch._revision,
                batch._revision + 1,
                batch.world_keys,
                tuple(d.receipt for d in drafts),
            )
            stats = {**batch._stats, "control_submissions": batch._stats["control_submissions"] + 1}
            batch._before_commit("device-controls", receipt)
            for owner, draft in zip(batch._buckets, drafts, strict=True):
                controls.validate(owner, draft)
        except BaseException:
            for owner, draft in zip(batch._buckets, drafts):
                controls.discard(owner, draft)
            batch._stats["control_refusals"] += 1
            raise
        for owner, draft in zip(batch._buckets, drafts, strict=True):
            controls.commit(owner, draft)
        batch._revision, batch._stats = receipt.revision, stats
        for owner in batch._buckets:
            controls._cleanup_retired(owner)
            try:
                batch._observe_owner(owner)
            except BaseException as error:
                batch._counter_errors += (repr(error),)
        return receipt
