"""Resident rigid source -> device BVH -> original finite triangle distance.

All physical storage is borrowed. Only complete source/capacity/readback and
normal resource cleanup permit a new immutable query publication.
"""

import hashlib
import math
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, fields, is_dataclass, replace
from numbers import Real
from types import MappingProxyType

import numpy as np

from ..geometry._collision_bvh import BoundsTree
from ..geometry._triangle_distance_dispatch import (
    BoundaryDistanceCapacityError,
    BoundaryDistanceGroup,
    BoundaryDistanceLimits,
    BoundaryDistancePair,
)
from ._device_triangle_distance import _BufferLease, validate_triangle_distance_rows
from ._resident_boundary_source import (
    borrow_boundary_source,
    compile_registration,
    freeze,
    owned,
    stored_interval,
    validate_source,
)
from ._tyx_collision_bvh import TreeBuffers
from .mujoco_tyx._resources import DeviceResources


def _nbytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, Mapping):
        return sum(_nbytes(v, seen) for v in value.values())
    if is_dataclass(value):
        return sum(_nbytes(getattr(value, f.name), seen) for f in fields(value))
    if isinstance(value, (tuple, list)):
        return sum(_nbytes(v, seen) for v in value)
    return 0


def _digest(value):
    digest = hashlib.sha256()

    def add(v):
        if isinstance(v, np.ndarray):
            digest.update(v.dtype.str.encode())
            digest.update(repr(v.shape).encode())
            digest.update(v.tobytes())
        elif isinstance(v, Mapping):
            for key, item in v.items():
                add(key)
                add(item)
        elif is_dataclass(v):
            digest.update(type(v).__qualname__.encode())
            for field in fields(v):
                add(field.name)
                add(getattr(v, field.name))
        elif isinstance(v, (tuple, list)):
            digest.update(str(len(v)).encode())
            for item in v:
                add(item)
        elif type(v) is object:
            digest.update(("identity:" + str(id(v))).encode())
        else:
            digest.update(repr(v).encode())
        digest.update(b"\0")

    add(value)
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class ResidentBoundaryNarrowPacket:
    schema: str
    source_binding: str
    surface: object
    revision: int
    facet_ids: np.ndarray
    segment_ids: np.ndarray
    candidates: np.ndarray
    origin_m: np.ndarray
    pair_origins_relative_m: np.ndarray
    packed: Mapping
    raw: Mapping
    rows: tuple


@dataclass(frozen=True, eq=False)
class ResidentBoundaryDistances:
    schema: str
    semantics: str
    source_kind: str
    owner: object
    registration: object
    sequence: int
    backend: str
    topology_generation: int
    state_epoch: int
    active: int
    world_keys: tuple
    ticks: np.ndarray
    margin: float
    limits: BoundaryDistanceLimits
    source_binding: str
    source: Mapping
    pairs: tuple
    groups: tuple
    required_candidates: int
    counters: Mapping


@dataclass(frozen=True, eq=False)
class ResidentBoundaryQueryAttempt:
    sequence: int
    status: str
    source: object
    counters: Mapping
    required_capacities: Mapping
    completed_groups: tuple
    raw_prefix: Mapping
    error: object


class _BoundaryRuntimeLease:
    def __init__(self, batch):
        self.batch = batch
        self.runtime = batch._runtime
        self.token = batch._register_boundary_runtime_lease(self)
        self.closed = False

    def close(self):
        if not self.closed:
            self.batch._release_boundary_runtime_lease(self.token)
            self.closed = True


class _BVHEngine:
    def __init__(self, plan, ledger, ordinal, view):
        self.plan, self.ledger, self.ordinal, self.view = plan, ledger, ordinal, view

    def _scratch(self, shape, dtype):
        return self.plan._allocate(self.ledger, shape=shape, dtype=dtype)

    def _spatial_upload(self, target, value):
        self.plan._upload(target, value)

    def _spatial_output(self, array):
        return self.plan._download(array, "group:" + str(self.ordinal) + "/bvh")

    def _launch_device(self, name, kernel, *args):
        self.plan._launch("group:" + str(self.ordinal) + "/" + name, kernel, *args)

    def _require_device_inputs(self, lease, *arrays):
        if lease is not self.view._lease:
            raise ValueError("BVH source lease differs")
        lease.require()
        self.plan._validate_source(self.view)
        if any(self.plan._array_owners.get(id(a)) is not self.ledger for a in arrays):
            raise ValueError("BVH requires current registered device buffers")


class ResidentBoundaryQueryPlan:
    def __init__(self, batch, model, bindings, *, limits, memory_limit_bytes):
        if type(limits) is not BoundaryDistanceLimits:
            raise TypeError("boundary limits require BoundaryDistanceLimits")
        if type(memory_limit_bytes) is not int or memory_limit_bytes <= 0:
            raise ValueError("boundary memory_limit_bytes requires a positive builtin int")
        self._batch = batch
        self._mode = batch._mode
        self._tx = batch._tx if hasattr(batch, "_tx") else None
        self._runtime = getattr(batch, "_runtime", None)
        self._limits = limits
        self._memory_limit = memory_limit_bytes
        self._closed = False
        self._blocked = None
        self._querying = False
        self._last_result = None
        self._last_attempt = None
        self._sequence = 0
        self._generation = 0
        self._persistent_resources = None
        self._query_resources = None
        self._array_owners = {}
        self._buffers = {}
        self._layouts = {}
        self._raw_prefix = {}
        self._launch_labels = []
        self._requirements = {}
        self._completed_groups = []
        self._counts = dict.fromkeys(
            (
                "allocations",
                "uploads",
                "uploaded_bytes",
                "downloads",
                "downloaded_bytes",
                "launches",
                "completed_launches",
                "synchronizations",
                "live_bytes",
                "peak_owned_bytes",
                "host_downloaded_bytes",
                "host_constant_bytes",
                "packet_array_bytes",
                "persistent_bytes",
                "prepass_required_bytes",
                "peak_required_bytes",
                "queries",
                "failed_queries",
            ),
            0,
        )
        with batch._lock:
            batch._check_open()
            compile_registration(self, model, bindings)
            self._templates, self._pair_template = self._groups()
            self._counts["persistent_bytes"] = sum(x.nbytes for x in self._constants.values())
            self._counts["host_constant_bytes"] = _nbytes((self._source_base, self._templates))
            self._check_memory(self._counts["persistent_bytes"])
            batch._boundary_plans.add(self)
            if self._shape_keys:
                with batch._device():
                    self._persistent_resources = self._new_resources()
                    try:
                        for name, value in self._constants.items():
                            target = self._allocate(
                                self._persistent_resources, shape=value.shape, dtype=value.dtype
                            )
                            self._upload(target, value)
                            self._buffers[name] = target
                    except BaseException as original:
                        try:
                            self.close()
                        except BaseException as cleanup:
                            raise BaseExceptionGroup(
                                "boundary construction and cleanup failed", [original, cleanup]
                            ) from original
                        raise
            batch._boundary_plans.add(self)

    @property
    def last_result(self):
        return self._last_result

    @property
    def last_attempt(self):
        return self._last_attempt

    def stats(self):
        with self._batch._lock:
            return MappingProxyType(
                dict(self._counts, memory_limit_bytes=self._memory_limit, closed=self._closed)
            )

    def _check_memory(self, size):
        if size > self._memory_limit:
            raise MemoryError(
                f"boundary query requires {size} owned bytes, limit {self._memory_limit}"
            )

    def _check_open(self):
        self._batch._check_open()
        if self._blocked is not None:
            raise RuntimeError("boundary query retains UNKNOWN_RELEASE") from self._blocked
        if self._closed:
            raise RuntimeError("boundary query is closed")
        if self._query_resources is not None and self._query_resources.state != "CLOSED":
            self._batch._boundary_close_pending = True
            raise RuntimeError("boundary query cleanup pending; only close may be retried")

    def _new_resources(self):
        ledger = DeviceResources()
        ledger.set_runtime(_BoundaryRuntimeLease(self._batch))
        return ledger

    def _new_query_resources(self):
        self._query_resources = self._new_resources()
        return self._query_resources

    def _close_ledger(self, ledger):
        if ledger is None:
            return
        try:
            ledger.close()
        except BaseException as error:
            self._batch._boundary_close_pending = True
            if ledger.state == "UNKNOWN_RELEASE":
                self._blocked = error
                lease = (
                    ledger._runtime.owner
                    if ledger._runtime is not None
                    else next(f.owner for f in ledger.failures if f.kind == "runtime")
                )
                self._batch._quarantine_boundary_runtime_lease(lease.token, ledger)
            raise
        for key, value in tuple(self._array_owners.items()):
            if value is ledger:
                del self._array_owners[key]

    def _close_query_resources(self):
        self._close_ledger(self._query_resources)

    def close(self):
        with self._batch._lock:
            if self._querying:
                raise RuntimeError("close during boundary source loan")
            if self._closed:
                return
            errors = []
            for ledger in (self._query_resources, self._persistent_resources):
                try:
                    self._close_ledger(ledger)
                except BaseException as error:
                    errors.append(error)
            if errors:
                raise BaseExceptionGroup("boundary cleanup UNKNOWN_RELEASE or pending", errors)
            self._buffers.clear()
            self._closed = True
            self._batch._boundary_plans.discard(self)

    def _allocate(self, ledger, *, shape, dtype):
        size = math.prod(shape) * np.dtype(dtype).itemsize
        if not size:
            raise ValueError("zero owned boundary allocation is not permitted")
        self._check_memory(self._counts["live_bytes"] + size)
        array = (
            self._tx.cuda_empty(shape, dtype) if self._mode == "cuda" else np.empty(shape, dtype)
        )
        self._counts["allocations"] += 1
        self._counts["live_bytes"] += size
        self._counts["peak_owned_bytes"] = max(
            self._counts["peak_owned_bytes"], self._counts["live_bytes"]
        )
        ledger.add_buffer(_BufferLease(array, self._mode == "cuda", size, self._counts))
        self._array_owners[id(array)] = ledger
        return array

    def _upload(self, target, value):
        if self._mode == "cuda":
            target.copy_from(value)
            self._counts["uploads"] += 1
            self._counts["uploaded_bytes"] += value.nbytes
        else:
            target[:] = value

    def _launch(self, label, kernel, *args):
        self._launch_labels.append(label)
        self._counts["launches"] += 1
        kernel(*args, mode=self._mode)
        if self._mode == "cuda":
            self._runtime.sync()
            self._counts["synchronizations"] += 1
        self._counts["completed_launches"] += 1

    def _download(self, array, label):
        value = array.to_numpy() if self._mode == "cuda" else array.copy()
        if self._mode == "cuda":
            self._counts["downloads"] += 1
            self._counts["downloaded_bytes"] += value.nbytes
        self._counts["host_downloaded_bytes"] += value.nbytes
        self._raw_prefix[label] = owned(value)
        return value

    def _borrow_source(self):
        return borrow_boundary_source(self)

    def _validate_source(self, view):
        return validate_source(self, view)

    def _groups(self):
        shapes = self._model.shapes
        pairs = []
        grouped = {}
        excluded = set(self._model.excluded)
        for i, a in enumerate(shapes):
            for j, b in enumerate(shapes[i + 1 :], start=i + 1):
                if a.key.world != b.key.world:
                    continue
                reason = None
                if a.key.body == b.key.body:
                    reason = "same_rigid_body"
                elif not (a.group & b.mask and b.group & a.mask):
                    reason = "mask"
                elif (a.key, b.key) in excluded:
                    reason = "excluded"
                if reason:
                    pairs.append(BoundaryDistancePair(a.key, b.key, "filtered", reason))
                    continue
                ar = a.kind in ("sphere", "capsule")
                br = b.kind in ("sphere", "capsule")
                if ar == br:
                    pairs.append(
                        BoundaryDistancePair(a.key, b.key, "not_applicable", "same_family")
                    )
                    continue
                surface, round_shape = (j, i) if ar else (i, j)
                group = grouped.setdefault(surface, [])
                group.append((round_shape, len(pairs)))
                pairs.append(None)
        templates = []
        for shape_index, items in sorted(grouped.items()):
            shape = shapes[shape_index]
            facets = (shape.vertex_ids,) if shape.kind == "triangle" else shape.triangles
            by_id = {value: i for i, value in enumerate(shape.vertex_ids)}
            faces = np.asarray([[by_id[x] for x in f] for f in facets], np.int32)
            items = tuple(sorted(items))
            templates.append(
                dict(
                    shape_index=shape_index,
                    surface=shape.key,
                    items=items,
                    facets=facets,
                    local_faces=owned(faces),
                    faces=owned(faces + self._offsets[shape_index]),
                    round_shapes=owned([i for i, _ in items], np.int32),
                )
            )
        return tuple(templates), tuple(pairs)

    def _requirements_for(self):
        return dict(
            groups=len(self._templates),
            vertices=max(
                (len(self._model.shapes[g["shape_index"]].vertex_ids) for g in self._templates),
                default=0,
            ),
            faces=max((len(g["facets"]) for g in self._templates), default=0),
            segments=max((len(g["items"]) for g in self._templates), default=0),
        )

    def _prepass_bytes(self):
        s, n = len(self._shape_keys), int(self._offsets[-1])
        result = self._counts["persistent_bytes"] + 64 * s + 36 * n
        for group in self._templates:
            f, r = len(group["facets"]), len(group["items"])
            result += 40 * f + 48 * (2 * f - 1) + (52 + 4 * ((f + 29) // 30)) * r
        return result

    def _gather(self, view, ledger):
        from ._tyx_boundary_source import gather_boundary_points

        s, n = len(self._shape_keys), int(self._offsets[-1])
        outputs = {}
        for name, shape, dtype in (
            ("poses", (s, 12), np.float32),
            ("stamps", (s, 4), np.int32),
            ("points", (n, 3), np.float32),
            ("point_bounds", (n, 6), np.float32),
        ):
            outputs[name] = self._allocate(ledger, shape=shape, dtype=dtype)
        for layout in sorted(set(map(int, self._shape_layouts))):
            source = view.sources[layout]
            self._validate_source(view)
            self._launch(
                "source/gather:" + str(layout),
                gather_boundary_points,
                s,
                layout,
                int(layout >= 2),
                self._buffers["shape_meta"],
                self._buffers["local"],
                self._buffers["local_bounds"],
                source.meta,
                source.fk,
                view.accepted.ticks,
                outputs["poses"],
                outputs["stamps"],
                outputs["points"],
                outputs["point_bounds"],
            )
        raw = {name: self._download(array, "source/" + name) for name, array in outputs.items()}
        ticks = self._download(view.accepted.ticks, "source/accepted_ticks")
        self._validate_gather(view, raw, ticks)
        return outputs, raw, ticks

    def _validate_gather(self, view, raw, ticks):
        s, n = len(self._shape_keys), int(self._offsets[-1])
        for name, shape, dtype in (
            ("poses", (s, 12), np.float32),
            ("stamps", (s, 4), np.int32),
            ("points", (n, 3), np.float32),
            ("point_bounds", (n, 6), np.float32),
        ):
            value = raw[name]
            if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise RuntimeError("invalid boundary gathered " + name)
        if ticks.shape != (len(view.accepted.bindings),) or ticks.dtype != np.int32:
            raise RuntimeError("invalid accepted source ticks")
        meta = self._constants["shape_meta"]
        if (
            np.any(raw["stamps"][:, 0] != 1)
            or not np.array_equal(raw["stamps"][:, 1], ticks[meta[:, 3]])
            or not np.array_equal(raw["stamps"][:, 2], meta[:, 3])
            or not np.array_equal(raw["stamps"][:, 3], meta[:, 1])
        ):
            raise RuntimeError("boundary gathered source stamps differ")
        points, bounds = raw["points"], raw["point_bounds"]
        if np.any(bounds[:, :3] > points) or np.any(points > bounds[:, 3:]):
            raise RuntimeError("boundary point bounds omit gathered material")
        material_deltas = np.empty((n, 3), np.float64)
        for index, binding in enumerate(self._bindings):
            begin, end = map(int, self._offsets[index : index + 2])
            original = self._source_base["authored_constants"]["points"][index]
            if self._shape_rows[index] < 0:
                expected = original - self._anchors[binding.key.world]
            else:
                pose = raw["poses"][index].astype(np.float64)
                expected = pose[:3] + original @ pose[3:].reshape(3, 3).T
            material_deltas[begin:end] = points[begin:end].astype(np.float64) - expected
        # Validation from actual stored pose + authored constants, never input to
        # a device kernel and never called an ideal-FP64 FK oracle.
        raw["material_deltas"] = material_deltas
        self._raw_prefix["source/material_reconstruction_delta"] = owned(material_deltas)
        self._validate_source(view)

    def _broad_group(self, index, template, view, ledger, global_buffers, pending):
        from ._tyx_boundary_source import boundary_leaf_bounds, boundary_round_bounds

        faces, rounds = template["faces"], template["round_shapes"]
        f, r = len(faces), len(rounds)
        arrays = {}
        for name, value in (("faces", faces), ("rounds", rounds)):
            arrays[name] = self._allocate(ledger, shape=value.shape, dtype=value.dtype)
            self._upload(arrays[name], value)
        arrays["leaf_bounds"] = self._allocate(ledger, shape=(f, 6), dtype=np.float32)
        arrays["query_bounds"] = self._allocate(ledger, shape=(r, 6), dtype=np.float32)
        prefix = "group:" + str(index) + "/"
        self._launch(
            prefix + "leaf_bounds",
            boundary_leaf_bounds,
            f,
            arrays["faces"],
            global_buffers["point_bounds"],
            arrays["leaf_bounds"],
        )
        self._launch(
            prefix + "round_bounds",
            boundary_round_bounds,
            r,
            self._margin_upper,
            arrays["rounds"],
            self._buffers["shape_meta"],
            self._buffers["radius_bounds"],
            global_buffers["point_bounds"],
            arrays["query_bounds"],
        )
        layout_key = (
            template["surface"],
            self._model.shapes[template["shape_index"]].revision,
            template["facets"],
        )
        if layout_key not in pending:
            # Topology only: no current geometry or HOST broad phase.
            pending[layout_key] = BoundsTree(
                np.zeros((f, 6), np.float32),
                (np.arange(f, dtype=np.int32),),
                np.zeros(f, dtype=bool),
            )
        tree = pending[layout_key]
        engine = _BVHEngine(self, ledger, index, view)
        buffers = TreeBuffers(engine, tree, r, False, device_inputs=True)
        params = np.tile(np.array((0, -1, 0), np.int32), (r, 1))
        raw = buffers.hits_device(
            tree,
            leaf_bounds=arrays["leaf_bounds"],
            query_bounds=arrays["query_bounds"],
            query_count=r,
            params=params,
            source_lease=view._lease,
        )
        leaves = self._download(arrays["leaf_bounds"], prefix + "leaf_bounds")
        boxes = self._download(arrays["query_bounds"], prefix + "query_bounds")
        if (
            not np.isfinite(leaves).all()
            or not np.isfinite(boxes).all()
            or np.any(leaves[:, :3] > leaves[:, 3:])
            or np.any(boxes[:, :3] > boxes[:, 3:])
        ):
            raise RuntimeError("invalid finite boundary AABB readback")
        words = (f + 29) // 30
        candidates = []
        for segment in range(r):
            for word in range(words):
                value = int(raw[segment, word])
                if value < 0 or value >= 1 << 30:
                    raise RuntimeError("invalid BVH bit word")
                for bit in range(30):
                    if value & (1 << bit):
                        facet = word * 30 + bit
                        if facet >= f:
                            raise RuntimeError("BVH facet tail bit")
                        candidates.append((segment, facet))
        return dict(
            template=template,
            arrays=arrays,
            leaves=leaves,
            boxes=boxes,
            bvh=raw,
            candidates=np.asarray(candidates, np.int64).reshape(-1, 2),
        )

    def _source_record(self, raw, ticks, view):
        points, axes = {}, {}
        for index, shape in enumerate(self._model.shapes):
            begin, end = map(int, self._offsets[index : index + 2])
            points[shape.key] = raw["points"][begin:end]
            if shape.kind in ("sphere", "capsule"):
                axes[shape.key] = raw["points"][[begin, end - 1]]
        return freeze(
            dict(
                self._source_base,
                gathered_points=points,
                gathered_axes=axes,
                stamps=dict(
                    rows=raw["stamps"],
                    poses=raw["poses"],
                    point_bounds=raw["point_bounds"],
                    accepted_ticks=ticks,
                    material_reconstruction_delta=raw["material_deltas"],
                ),
            )
        )

    def _pack_group(
        self, index, group, view, ledger, global_buffers, source_raw, binding, generation
    ):
        from ._tyx_boundary_source import pack_boundary_pairs
        from ._tyx_triangle_distance import triangle_distances

        template = group["template"]
        pairs = group["candidates"]
        p = len(pairs)
        prefix = "group:" + str(index) + "/"
        layouts = (
            ("selected", (p, 2), np.int32),
            ("geometry", (p, 5, 3), np.float32),
            ("radii", (p,), np.float32),
            ("scales", (p,), np.float32),
            ("owners", (p, 2), np.int32),
            ("frames", (p, 3), np.float32),
            ("pack_status", (p, 2), np.int32),
            ("candidates", (p, 18, 11), np.float32),
            ("valid", (p, 18), np.int32),
            ("summary", (p, 16), np.float32),
            ("meta", (p, 7), np.int32),
        )
        if p:
            arrays = {
                name: self._allocate(ledger, shape=shape, dtype=dtype)
                for name, shape, dtype in layouts
            }
            self._upload(arrays["selected"], pairs.astype(np.int32))
            self._validate_source(view)
            self._launch(
                prefix + "pack",
                pack_boundary_pairs,
                p,
                generation,
                arrays["selected"],
                group["arrays"]["faces"],
                group["arrays"]["rounds"],
                self._buffers["shape_meta"],
                self._buffers["radius_bounds"],
                global_buffers["points"],
                arrays["geometry"],
                arrays["radii"],
                arrays["scales"],
                arrays["owners"],
                arrays["frames"],
                arrays["pack_status"],
            )
            packed = {
                name: self._download(arrays[name], prefix + name)
                for name in ("geometry", "radii", "scales", "owners", "frames", "pack_status")
            }
            self._validate_pack(group, packed, source_raw, generation)
            self._launch(
                prefix + "narrow",
                triangle_distances,
                p,
                generation,
                arrays["geometry"],
                arrays["radii"],
                arrays["scales"],
                arrays["owners"],
                arrays["candidates"],
                arrays["valid"],
                arrays["summary"],
                arrays["meta"],
            )
            raw = {
                name: self._download(arrays[name], prefix + name)
                for name in ("candidates", "valid", "summary", "meta")
            }
        else:
            empty = {name: np.empty(shape, dtype) for name, shape, dtype in layouts}
            packed = {
                name: empty[name]
                for name in ("geometry", "radii", "scales", "owners", "frames", "pack_status")
            }
            raw = {name: empty[name] for name in ("candidates", "valid", "summary", "meta")}
        shape = self._model.shapes[template["shape_index"]]
        inputs = dict(
            candidates=pairs,
            vertex_ids=np.asarray(shape.vertex_ids, np.int64),
            faces=template["local_faces"],
        )
        rows, _ = validate_triangle_distance_rows(
            inputs,
            packed,
            raw,
            packed["frames"].astype(np.float64),
            packed["scales"],
            generation,
            {},
        )
        return ResidentBoundaryNarrowPacket(
            "phymeo-resident-triangle-distance-v1",
            binding,
            shape.key,
            shape.revision,
            owned(np.arange(len(template["facets"])), np.int64),
            owned(np.arange(len(template["items"])), np.int64),
            owned(pairs, np.int64),
            owned(self._anchors[shape.key.world]),
            owned(packed["frames"]),
            freeze(packed),
            freeze(raw),
            rows,
        )

    def _validate_pack(self, group, packed, source_raw, generation):
        pairs = group["candidates"]
        template = group["template"]
        p = len(pairs)
        for name, shape, dtype in (
            ("geometry", (p, 5, 3), np.float32),
            ("radii", (p,), np.float32),
            ("scales", (p,), np.float32),
            ("owners", (p, 2), np.int32),
            ("frames", (p, 3), np.float32),
            ("pack_status", (p, 2), np.int32),
        ):
            array = packed[name]
            if array.shape != shape or array.dtype != dtype or not np.isfinite(array).all():
                raise RuntimeError("invalid boundary packed " + name)
        if not np.array_equal(packed["owners"], pairs):
            raise RuntimeError("boundary packed original ownership differs")
        if np.any(packed["pack_status"][:, 0] != 1) or np.any(
            packed["pack_status"][:, 1] != generation
        ):
            raise RuntimeError("boundary device pack is invalid or stale")
        points = source_raw["points"].astype(np.float64)
        for index, (segment, facet) in enumerate(pairs):
            rows = template["faces"][facet]
            shape_index = int(template["round_shapes"][segment])
            begin, end = map(int, self._offsets[shape_index : shape_index + 2])
            frame = source_raw["points"][rows[0]]
            if not np.array_equal(packed["frames"][index], frame):
                raise RuntimeError("boundary pair anchor differs from gathered triangle")
            expected = np.vstack((points[rows], points[[begin, end - 1]])) - frame.astype(
                np.float64
            )
            length = float(packed["scales"][index])
            if length <= 0:
                raise RuntimeError("boundary local scale is not positive")
            # Checking already emitted geometry is not a HOST geometry producer.
            reference_length = max(
                *(np.linalg.norm(points[rows[(j + 1) % 3]] - points[rows[j]]) for j in range(3)),
                np.linalg.norm(points[end - 1] - points[begin]),
                float(self._constants["radius_bounds"][shape_index, 0]),
            )
            surface_deltas = source_raw["material_deltas"][rows]
            axis_deltas = source_raw["material_deltas"][[begin, end - 1]]
            if (
                np.max(np.linalg.norm(np.vstack((surface_deltas, axis_deltas)), axis=1))
                > 2e-4 * length
            ):
                raise RuntimeError(
                    "boundary gathered material precision exceeds pair-local witness gate"
                )
            if abs(length - reference_length) > 8 * 2**-24 * reference_length:
                raise RuntimeError("boundary local scale differs from original pair material")
            if (
                np.max(np.abs(packed["geometry"][index].astype(np.float64) - expected))
                > 8 * 2**-24 * length
            ):
                raise RuntimeError("boundary pair-relative source precision loses geometry")
            if packed["radii"][index] != self._constants["radius_bounds"][shape_index, 0]:
                raise RuntimeError("boundary radius binding differs")

    def _build_packet(self, view, source, groups, pairs, ticks, margin, total):
        return ResidentBoundaryDistances(
            "phymeo-resident-boundary-distances-v1",
            "threshold-limited-boundary-distance",
            "resident-rigid-device-v1",
            self._owner,
            self._registration,
            self._sequence + 1,
            self._mode,
            self._topology_generation,
            view.accepted.state_epoch,
            view.accepted.active,
            self._world_keys,
            owned(ticks, np.int32),
            margin,
            self._limits,
            self._binding,
            source,
            tuple(pairs),
            tuple(groups),
            total,
            freeze(dict(self._counts, launch_labels=tuple(self._launch_labels))),
        )

    def _attempt(self, status, source, error=None):
        return ResidentBoundaryQueryAttempt(
            self._generation,
            status,
            freeze(source),
            freeze(dict(self._counts, launch_labels=tuple(self._launch_labels))),
            freeze(self._requirements),
            tuple(self._completed_groups),
            freeze(self._raw_prefix),
            None if error is None else freeze(dict(type=type(error).__name__, message=str(error))),
        )

    def query(self, *, margin=0.0):
        with self._batch._lock:
            self._check_open()
            if isinstance(margin, (bool, np.bool_)) or not isinstance(margin, Real):
                raise ValueError("boundary margin requires a nonnegative finite scalar")
            margin = float(margin)
            if not math.isfinite(margin) or margin < 0:
                raise ValueError("boundary margin requires a nonnegative finite scalar")
            _, _, upper = stored_interval(np.asarray([margin]))
            self._margin_upper = upper[0]
            if self._generation >= 2**31 - 1:
                raise RuntimeError("boundary query generation exhausted")
            self._generation += 1
            self._raw_prefix = {}
            self._launch_labels = []
            self._completed_groups = []
            self._requirements = self._requirements_for()
            self._query_resources = None
            source_stamp = None
            self._querying = True
            try:
                # Hard implementation bounds always precede allocation. User limits
                # are checked after the complete in-domain candidate prepass.
                hard = BoundaryDistanceLimits()
                for name, count in self._requirements.items():
                    if count > getattr(hard, name):
                        raise BoundaryDistanceCapacityError(name, count, getattr(hard, name))
                prepass = self._prepass_bytes()
                self._counts["prepass_required_bytes"] = prepass
                self._counts["peak_required_bytes"] = prepass
                self._check_memory(prepass)
                with self._borrow_source() as view:
                    self._validate_source(view)
                    source_stamp = dict(
                        owner=view.owner,
                        registration=view.registration,
                        topology_generation=view.accepted.topology_generation,
                        state_epoch=view.accepted.state_epoch,
                        active=view.accepted.active,
                    )
                    pending = self._layouts.copy()
                    active = self._batch._device() if self._shape_keys else nullcontext()
                    original_error = None
                    try:
                        with active:
                            if self._shape_keys:
                                ledger = self._new_query_resources()
                                global_buffers, source_raw, accepted_ticks = self._gather(
                                    view, ledger
                                )
                            else:
                                ledger = None
                                global_buffers = {}
                                source_raw = dict(
                                    points=np.empty((0, 3), np.float32),
                                    point_bounds=np.empty((0, 6), np.float32),
                                    poses=np.empty((0, 12), np.float32),
                                    stamps=np.empty((0, 4), np.int32),
                                    material_deltas=np.empty((0, 3), np.float64),
                                )
                                accepted_ticks = np.empty(0, np.int32)
                            source = self._source_record(source_raw, accepted_ticks, view)
                            self._binding = _digest((source_stamp, margin, self._limits, source))
                            broad = []
                            for index, template in enumerate(self._templates):
                                self._validate_source(view)
                                broad.append(
                                    self._broad_group(
                                        index, template, view, ledger, global_buffers, pending
                                    )
                                )
                            total = sum(len(g["candidates"]) for g in broad)
                            self._requirements["candidates"] = total
                            peak = prepass + 1060 * total
                            self._counts["peak_required_bytes"] = peak
                            for name, count in self._requirements.items():
                                if count > getattr(self._limits, name):
                                    raise BoundaryDistanceCapacityError(
                                        name, count, getattr(self._limits, name)
                                    )
                            self._check_memory(peak)
                            groups = []
                            pairs = list(self._pair_template)
                            for index, g in enumerate(broad):
                                template = g["template"]
                                packet = self._pack_group(
                                    index,
                                    g,
                                    view,
                                    ledger,
                                    global_buffers,
                                    source_raw,
                                    self._binding,
                                    self._generation,
                                )
                                round_keys = tuple(
                                    self._shape_keys[i] for i, _ in template["items"]
                                )
                                round_ids = tuple(
                                    self._model.shapes[i].vertex_ids for i, _ in template["items"]
                                )
                                groups.append(
                                    BoundaryDistanceGroup(
                                        template["surface"],
                                        round_keys,
                                        round_ids,
                                        template["facets"],
                                        owned(g["candidates"]),
                                        owned(g["leaves"]),
                                        owned(g["boxes"]),
                                        owned(g["bvh"]),
                                        packet,
                                        freeze(dict(rows=len(packet.rows))),
                                    )
                                )
                                for segment, (shape_index, pair_slot) in enumerate(
                                    template["items"]
                                ):
                                    selected = tuple(
                                        i
                                        for i, row in enumerate(packet.rows)
                                        if row.segment_id == segment
                                    )
                                    facets = tuple(packet.rows[i].facet_id for i in selected)
                                    a, b = sorted(
                                        (template["surface"], self._shape_keys[shape_index])
                                    )
                                    pairs[pair_slot] = BoundaryDistancePair(
                                        a,
                                        b,
                                        "boundary_candidates"
                                        if selected
                                        else "no_boundary_candidate",
                                        "finite_boundary",
                                        index,
                                        segment,
                                        facets,
                                        selected,
                                    )
                                self._completed_groups.append(index)
                            self._validate_source(view)
                            ticks = [
                                int(accepted_ticks[view.accepted.slots[w]])
                                for w in self._world_keys
                            ]
                            packet = self._build_packet(
                                view, source, groups, pairs, ticks, margin, total
                            )
                    except BaseException as error:
                        original_error = error
                        raise
                    finally:
                        try:
                            self._close_query_resources()
                        except BaseException as cleanup:
                            if original_error is not None:
                                raise BaseExceptionGroup(
                                    "boundary query and cleanup failed", [original_error, cleanup]
                                ) from original_error
                            raise
                    self._validate_source(view)
                    # All allocations, independent cleanup and final same-bank checks
                    # completed. These assignments perform no device IO.
                    final_counts = dict(
                        self._counts,
                        queries=self._counts["queries"] + 1,
                        packet_array_bytes=_nbytes(packet),
                    )
                    final_evidence = freeze(
                        dict(final_counts, launch_labels=tuple(self._launch_labels))
                    )
                    packet = replace(packet, counters=final_evidence)
                    attempt = replace(
                        self._attempt("COMPLETED", source_stamp), counters=final_evidence
                    )
                    self._layouts = pending
                    self._last_result = packet
                    self._sequence = packet.sequence
                    self._counts.update(final_counts)
                    self._last_attempt = attempt
                    return packet
            except BaseException as error:
                self._counts["failed_queries"] += 1
                if self._query_resources is not None and self._query_resources.state != "CLOSED":
                    self._batch._boundary_close_pending = True
                self._last_attempt = self._attempt(
                    "UNKNOWN_RELEASE" if self._blocked is not None else "FAILED",
                    source_stamp,
                    error,
                )
                raise
            finally:
                self._querying = False

    def require_current(self, result):
        with self._batch._lock:
            self._check_open()
            if result is not self._last_result or result is None:
                raise ValueError("boundary evidence is not this plan's current publication")
            if (
                result.owner is not self._owner
                or result.registration is not self._registration
                or result.topology_generation != self._batch._topology_generation
                or result.state_epoch != self._batch._state_epoch
                or result.active != self._batch._active
            ):
                raise ValueError("boundary evidence has another or stale accepted source")
            stamp = dict(
                owner=result.owner,
                registration=result.registration,
                topology_generation=result.topology_generation,
                state_epoch=result.state_epoch,
                active=result.active,
            )
            if result.source_binding != _digest(
                (stamp, result.margin, result.limits, result.source)
            ):
                raise ValueError("boundary source binding differs")
            return result
