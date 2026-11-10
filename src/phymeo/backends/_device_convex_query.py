"""Owned current-pose device convex queries; no contact law or HOST fallback.

The only HOST geometry operation is one-time validation of the registered hull.
Accepted poses and every distance/witness come from the borrowed device bank.
"""

from collections import Counter
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionMotion

_STATUS = ("UNWRITTEN", "RESOLVED", "UNKNOWN", "UNSUPPORTED", "RESOURCE_LIMIT")
_RELATION = ("UNDETERMINED", "SEPARATED", "OVERLAP", "CONTACT_BAND")
_REASON = (
    "OK",
    "NEAR_ZERO",
    "NORMAL_NONUNIQUE",
    "CORE_CONTAINMENT",
    "GJK_LIMIT",
    "EPA_LIMIT",
    "DEGENERATE_SIMPLEX",
    "EPA_CAPACITY",
    "UNSUPPORTED_SHAPE",
    "SOURCE_PRECISION",
    "STALLED",
)
_PHASE = ("NONE", "ANALYTIC_ROUND", "GJK_CORE", "GJK_POLYTOPE", "EPA")
_PATCH_STATUS = ("ABSENT", "RESOLVED", "UNRESOLVED", "UNSUPPORTED", "RESOURCE_LIMIT")
_WIRE = {
    "meta": (np.int32, (16,)),
    "gap": (np.float32, (4,)),
    "normal": (np.float32, (3,)),
    "surface": (np.float32, (2, 3)),
    "material_ids": (np.int32, (2, 4)),
    "barycentric": (np.float32, (2, 4)),
    "skin_offset": (np.float32, (2, 3)),
}
_COUNTERS = (
    "allocations",
    "live_bytes",
    "peak_owned_bytes",
    "static_uploads",
    "dynamic_uploads",
    "uploaded_bytes",
    "downloads",
    "downloaded_bytes",
    "launches",
    "completed_launches",
    "synchronizations",
    "attempts",
    "queries",
    "failed_queries",
    "pose_gather_launches",
    "query_launches",
    "pair_uploads",
    "host_read_bytes",
    "host_upload_bytes",
)


def _owned(value, dtype=None):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _freeze(value):
    if isinstance(value, np.ndarray):
        return _owned(value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-bool integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return int(value)


def _canonical(pair):
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError("each convex query pair must contain two CollisionKeys")
    a, b = pair
    if not isinstance(a, CollisionKey) or not isinstance(b, CollisionKey):
        raise TypeError("convex query pairs require CollisionKeys")
    if a == b or a.world != b.world:
        raise ValueError("convex query requires distinct shapes in one world")
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True)
class ConvexQueryWitness:
    key: CollisionKey
    vertex_ids: tuple
    weights: tuple
    offset_m: tuple
    surface_relative_m: tuple
    valid: bool

    def __post_init__(self):
        for name in ("vertex_ids", "weights", "offset_m", "surface_relative_m"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class ConvexPatchSite:
    weight: float
    gap_m: float
    normal: tuple
    common_m: tuple
    first: ConvexQueryWitness
    second: ConvexQueryWitness

    def __post_init__(self):
        for name in ("normal", "common_m"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class ConvexQueryPatch:
    status: str
    dimension: int
    sites: tuple
    device_resolved: bool
    host_completed: bool
    skip_host_patch_search: bool

    def __post_init__(self):
        object.__setattr__(self, "sites", tuple(self.sites))


@dataclass(frozen=True)
class ConvexQueryRow:
    pair: tuple
    status: str
    relation: str
    reason: str
    phase: str
    gap_m: float
    gap_lower_m: float
    gap_upper_m: float
    scale_m: float
    origin_m: tuple
    normal: tuple
    gap_valid: bool
    normal_valid: bool
    witness_valid: bool
    first: ConvexQueryWitness
    second: ConvexQueryWitness
    gjk_used: int
    epa_used: int
    support_calls: int
    vertices_used: int
    faces_used: int
    patch: ConvexQueryPatch

    def __post_init__(self):
        for name in ("pair", "origin_m", "normal"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True, eq=False)
class ConvexQueryResult:
    owner: object
    registration: object
    topology_generation: int
    state_epoch: int
    active: int
    world_keys: tuple
    ticks: np.ndarray
    shape_keys: tuple
    shape_revisions: tuple
    pair_source: str
    requested_pairs: tuple
    request_row: tuple
    request_reversed: tuple
    rows: tuple
    raw: Mapping
    source: Mapping
    stats: Mapping
    attempt_tag: int
    patch: Mapping

    def __post_init__(self):
        for name in (
            "world_keys",
            "shape_keys",
            "shape_revisions",
            "requested_pairs",
            "request_row",
            "request_reversed",
            "rows",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "ticks", _owned(self.ticks, np.int32))
        for name in ("raw", "source", "stats", "patch"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


class ConvexQueryCapacityExceeded(ValueError):
    def __init__(self, required, capacity, per_world_required):
        self.required = int(required)
        self.capacity = int(capacity)
        self.per_world_required = MappingProxyType(dict(per_world_required))
        super().__init__(f"convex query needs {required} pair rows; capacity is {capacity}")


class ConvexQueryPlan:
    """An opt-in, bounded query tied to one broad plan and accepted-state owner."""

    def __init__(
        self,
        batch,
        broad,
        *,
        pairs=None,
        pair_capacity=None,
        gjk_iterations=64,
        epa_iterations=192,
        epa_vertex_capacity=64,
        epa_face_capacity=128,
        memory_limit_bytes=67108864,
        site_capacity=None,
    ):
        from ._resident_collision_plan import ResidentCollisionPlan
        from .tyx_world_batch import TyxWorldBatch

        if not isinstance(batch, TyxWorldBatch) or not isinstance(broad, ResidentCollisionPlan):
            raise TypeError("convex query requires a resident batch and collision plan")
        if broad._batch is not batch:
            raise ValueError("convex query broad phase belongs to another batch")
        self._batch, self._broad = batch, broad
        self._resources = ExitStack()
        self._counts = dict.fromkeys(_COUNTERS, 0)
        self._closed = False
        self._last_result = None
        self._attempt_tag = 0
        self._attempt_ticks = None
        self._buffers = {}
        self._kernels = None
        self._gjk = _integer(gjk_iterations, "gjk_iterations", 1, 64)
        self._epa = _integer(epa_iterations, "epa_iterations", 1, 192)
        self._vertices = _integer(epa_vertex_capacity, "epa_vertex_capacity", 4, 64)
        self._faces = _integer(epa_face_capacity, "epa_face_capacity", 4, 128)
        self._memory_limit = _integer(memory_limit_bytes, "memory_limit_bytes", 1, 2**63 - 1)
        self._site_capacity = (
            None if site_capacity is None else _integer(site_capacity, "site_capacity", 1, 4096)
        )
        self._patch_kernels = None
        with broad._borrow_convex_state() as view:
            self._owner, self._registration = view.owner, view.registration
            self._topology_generation = view.accepted.topology_generation
            self._shape_keys = tuple(view.shape_keys)
            self._shape_revisions = tuple(view.shape_revisions)
            self._allowed = tuple(sorted(view.allowed_pairs))
            self._requested = None if pairs is None else tuple(tuple(pair) for pair in pairs)
            if self._requested is not None:
                canonical = tuple(_canonical(pair) for pair in self._requested)
                if len(set(canonical)) != len(canonical):
                    raise ValueError("duplicate unordered explicit query pair")
                if any(pair not in self._allowed for pair in canonical):
                    raise ValueError("explicit pair is outside the filtered physical domain")
            self._capacity = (
                len(self._allowed)
                if pair_capacity is None
                else _integer(pair_capacity, "pair_capacity", 0, 32640)
            )
            self._tile = min(32, self._capacity)
            self._compile_source(view)
            layouts = self._layouts()
            owned_bytes = sum(
                int(np.prod(shape)) * np.dtype(dtype).itemsize
                for shape, dtype, _ in layouts.values()
            )
            if owned_bytes > self._memory_limit:
                raise MemoryError(f"convex query requires {owned_bytes} owned bytes")
            self._owned_layout = tuple(
                (name, shape, np.dtype(dtype).str) for name, (shape, dtype, _) in layouts.items()
            )
            try:
                if self._shape_keys:
                    from ._tyx_convex_query_kernels import make_convex_query_kernels

                    self._kernels = make_convex_query_kernels()
                    if self._site_capacity is not None:
                        from ._tyx_convex_patch_kernels import make_convex_patch_kernels

                        self._patch_kernels = make_convex_patch_kernels()
                for name, (shape, dtype, initial) in layouts.items():
                    if not self._shape_keys:
                        self._buffers[name] = np.full(
                            shape, -1 if name == "material_ids" else 0, dtype=dtype
                        )
                    elif not np.prod(shape):
                        self._buffers[name] = np.empty(shape, dtype=dtype)
                    else:
                        self._buffers[name] = self._allocate(shape, dtype, initial)
                broad._convex_dependents.add(self)
            except BaseException:
                self.close()
                raise

    @property
    def closed(self):
        return self._closed

    @property
    def last_result(self):
        return self._last_result

    @property
    def stats(self):
        with self._batch._lock:
            return MappingProxyType(
                dict(
                    self._counts,
                    closed=self._closed,
                    pair_capacity=self._capacity,
                    pair_tile=self._tile,
                    site_capacity=0 if self._site_capacity is None else self._site_capacity,
                )
            )

    def _compile_source(self, view):
        from ..core.capabilities import SolverFailure
        from ..geometry._convex_hull import HullShape
        from ._resident_collision_plan import ResidentStaticBinding
        from ._resident_convex_source import _DYNAMIC_BINDINGS, _LAYOUT_NAMES

        keys = tuple(view.world_keys)
        self._world_keys = keys
        origins = np.zeros((len(keys), 3), dtype=np.float64)
        for i, world in enumerate(keys):
            source = view.sources[view.world_layouts[i]]
            origins[i] = np.asarray(source.specs[world].anchor, dtype=np.float32).astype(np.float64)
        self._source_layouts = tuple(sorted(set(map(int, view.shape_layouts))))
        shape_index = {key: i for i, key in enumerate(self._shape_keys)}
        self._shape_index = shape_index
        self._world_index = {world: i for i, world in enumerate(keys)}
        offsets, ids, declared, uploaded, errors, frames = [0], [], [], [], [], []
        radii, stored_radii, radius_errors, kinds, tags, precision = [], [], [], [], [], []
        source_rows = np.asarray(view.shape_rows, dtype=np.int32)
        source_groups = np.asarray(view.shape_groups, dtype=np.int32)
        shape_world_rows = []
        for index, (shape, binding) in enumerate(
            zip(view.model.shapes, view.bindings, strict=True)
        ):
            if shape.key != self._shape_keys[index] or binding.key != shape.key:
                raise ValueError("convex source shape order does not match its binding")
            world_row = self._world_index[shape.key.world]
            shape_world_rows.append(world_row)
            if type(binding) in _DYNAMIC_BINDINGS:
                points = np.asarray(binding.local_points_m, dtype=np.float64)
                local = points
                frame = "body_local"
            elif type(binding) is ResidentStaticBinding:
                if binding.motion.moving:
                    raise ValueError("static convex source must have constant material coordinates")
                points = np.asarray(binding.motion.start, dtype=np.float64)
                local = points - origins[world_row]
                frame = "world_anchor_relative"
            else:
                raise ValueError("unsupported convex source binding")
            if points.shape != (len(shape.vertex_ids), 3) or not np.isfinite(points).all():
                raise ValueError("convex source material coordinates are invalid")
            kind = {"sphere": 0, "capsule": 1, "hull": 2}.get(shape.kind, 3)
            if kind == 2:
                try:
                    HullShape(shape, CollisionMotion(points))
                except SolverFailure as error:
                    # Capacity and incidence stay source rejections. There is
                    # no host completion that may drop vertices and continue.
                    raise ValueError(str(error)) from error
            if kind == 0 and len(points) != 1 or kind == 1 and len(points) != 2:
                raise ValueError("round convex source has invalid material arity")
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                stored = np.asarray(local, dtype=np.float32)
                radius = np.float32(shape.radius)
            if not np.isfinite(stored).all() or not np.isfinite(radius):
                raise ValueError("convex source cannot be represented by finite FP32 inputs")
            # A large reference that erases declared finite local offsets cannot
            # be repaired by silently inventing a different world geometry.
            lost = bool(
                np.max(np.abs((points - origins[world_row]) + origins[world_row] - points))
                > 1e-4 * max(1.0, float(np.max(np.abs(local))))
            )
            if type(binding) in _DYNAMIC_BINDINGS:
                lost = bool(
                    np.max(np.abs((origins[world_row] + points) - origins[world_row] - points))
                    > 1e-4 * max(1.0, float(np.max(np.abs(points))))
                )
            tiny = np.finfo(np.float32).tiny
            lost = lost or bool(np.any((stored != 0) & (np.abs(stored) < tiny)))
            lost = lost or bool(np.any((local != 0) & (stored == 0)))
            lost = lost or bool(radius != 0 and abs(radius) < tiny)
            lost = lost or bool(shape.radius != 0 and radius == 0)
            if kind == 2 and shape.radius != 0:
                kind = 3
            begin = offsets[-1]
            offsets.append(begin + len(points))
            if any(node < 0 or node > 2147483647 for node in shape.vertex_ids):
                raise ValueError("convex material IDs must fit the declared int32 wire")
            ids.extend(shape.vertex_ids)
            declared.extend(points)
            uploaded.extend(stored)
            errors.extend(stored.astype(np.float64) - local)
            frames.append(frame)
            radii.append(float(shape.radius))
            stored_radii.append(radius)
            radius_errors.append(float(radius) - float(shape.radius))
            kinds.append(shape.kind)
            tags.append((kind, begin, offsets[-1], view.accepted.slots[shape.key.world]))
            precision.append(int(lost))
        self._source = dict(
            schema="resident-convex-source-v1",
            model=view.model,
            bindings=tuple(view.bindings),
            world_keys=keys,
            world_origins_m=origins,
            shape_world_rows=np.asarray(shape_world_rows, dtype=np.int32),
            shape_keys=self._shape_keys,
            shape_revisions=self._shape_revisions,
            shape_kinds=tuple(kinds),
            shape_rows=source_rows,
            shape_groups=source_groups,
            shape_layouts=np.asarray(view.shape_layouts, dtype=np.int32),
            world_layouts=tuple(_LAYOUT_NAMES[item] for item in view.world_layouts),
            vertex_offsets=np.asarray(offsets, dtype=np.int32),
            vertex_ids=np.asarray(ids, dtype=np.int32),
            declared_local_points_m=np.asarray(declared, dtype=np.float64).reshape(-1, 3),
            upload_frame=tuple(frames),
            uploaded_local_points_m=np.asarray(uploaded, dtype=np.float32).reshape(-1, 3),
            point_quantization_error_m=np.asarray(errors, dtype=np.float64).reshape(-1, 3),
            declared_radius_m=np.asarray(radii, dtype=np.float64),
            uploaded_radius_m=np.asarray(stored_radii, dtype=np.float32),
            radius_quantization_error_m=np.asarray(radius_errors, dtype=np.float64),
            allowed_pairs=self._allowed,
        )
        self._tags = np.asarray(tags, dtype=np.int32).reshape(-1, 4)
        self._precision = np.asarray(precision, dtype=np.int32)

    def _layouts(self):
        s, v, p, t = (
            len(self._shape_keys),
            len(self._source["vertex_ids"]),
            self._capacity,
            self._tile,
        )
        static = {
            "tags": self._tags,
            "rows": self._source["shape_rows"],
            "groups": self._source["shape_groups"],
            "layouts": self._source["shape_layouts"],
            "offsets": self._source["vertex_offsets"],
            "points": self._source["uploaded_local_points_m"],
            "ids": self._source["vertex_ids"],
            "radius": self._source["uploaded_radius_m"],
            "precision": self._precision,
        }
        layouts = {name: (array.shape, array.dtype, array) for name, array in static.items()}
        scratch = {
            "pose": ((s, 12), np.float32),
            "stamp": ((s, 3), np.int32),
            "world_points": ((v, 3), np.float32),
            "pairs": ((p, 2), np.int32),
            "vertices": ((t, self._vertices, 9), np.float32),
            "vertex_slots": ((t, self._vertices, 2), np.int32),
            "faces": ((t, self._faces, 3), np.int32),
            "face_geometry": ((t, self._faces, 4), np.float32),
            "edges": ((t, self._faces * 3, 2), np.int32),
            "weights": ((t, 4), np.float32),
            "indices": ((t, 4), np.int32),
            "simplex_copy": ((t, 4, 9), np.float32),
            "simplex_slots": ((t, 4, 2), np.int32),
        }
        scratch.update({name: ((p, *tail), dtype) for name, (dtype, tail) in _WIRE.items()})
        if self._site_capacity is not None:
            sites = self._site_capacity
            scratch.update(
                {
                    "patch_meta": ((p, 8), np.int32),
                    "patch_weight": ((sites,), np.float32),
                    "patch_gap": ((sites,), np.float32),
                    "patch_normal": ((sites, 3), np.float32),
                    "patch_common": ((sites, 3), np.float32),
                    "patch_ids": ((sites, 2, 4), np.int32),
                    "patch_bary": ((sites, 2, 4), np.float32),
                    "patch_surface": ((sites, 2, 3), np.float32),
                    "patch_skin": ((sites, 2, 3), np.float32),
                    "patch_mcount": ((sites, 2), np.int32),
                    "patch_work": ((t, 32, 24), np.float32),
                    "patch_work_i": ((t, 32, 8), np.int32),
                }
            )
        layouts.update({name: (shape, dtype, None) for name, (shape, dtype) in scratch.items()})
        return layouts

    def _free(self, buffer, byte_count, device):
        try:
            if device:
                buffer.free()
        finally:
            self._counts["live_bytes"] -= byte_count
            if device:
                self._batch._counts["device_owned_bytes"] -= byte_count

    def _allocate(self, shape, dtype, initial=None):
        """Fault seam: a successful return already belongs to the ExitStack."""
        batch = self._batch
        device = batch._mode == "cuda"
        if device:
            result = (
                batch._tx.cuda_empty(shape, dtype)
                if initial is None
                else batch._tx.cuda_array(initial)
            )
        else:
            result = (
                np.empty(shape, dtype=dtype) if initial is None else np.array(initial, copy=True)
            )
        size = int(result.nbytes)
        self._resources.callback(self._free, result, size, device)
        self._counts["allocations"] += 1
        self._counts["live_bytes"] += size
        self._counts["peak_owned_bytes"] = max(
            self._counts["peak_owned_bytes"], self._counts["live_bytes"]
        )
        if device:
            batch._counts["device_allocations"] += 1
            batch._counts["device_owned_bytes"] += size
        if initial is not None:
            self._counts["static_uploads"] += int(device)
            self._counts["uploaded_bytes"] += int(device) * size
            self._counts["host_upload_bytes"] += size
        return result

    def _upload_pairs(self, pairs):
        data = np.full((self._capacity, 2), -1, dtype=np.int32)
        for i, (a, b) in enumerate(pairs):
            data[i] = self._shape_index[a], self._shape_index[b]
        if self._capacity:
            if self._batch._mode == "cuda":
                self._buffers["pairs"].copy_from(data)
                self._counts["dynamic_uploads"] += 1
                self._counts["pair_uploads"] += 1
                self._counts["uploaded_bytes"] += data.nbytes
            else:
                np.copyto(self._buffers["pairs"], data)
            self._counts["host_upload_bytes"] += data.nbytes

    def _read(self, buffer):
        if self._batch._mode == "cuda":
            value = buffer.to_numpy()
            self._counts["downloads"] += 1
            self._counts["downloaded_bytes"] += value.nbytes
        else:
            value = buffer.copy()
        self._counts["host_read_bytes"] += value.nbytes
        return value

    def _read_raw(self):
        """Fault seam: every raw copy is complete; validation has not yet run."""
        return {
            name: (
                self._read(self._buffers[name]) if self._shape_keys else self._buffers[name].copy()
            )
            if self._capacity
            else np.empty((0, *tail), dtype=dtype)
            for name, (dtype, tail) in _WIRE.items()
        }

    def _launch(self, kernel, category, *args):
        self._counts["launches"] += 1
        self._batch._launch(kernel, *args)
        self._counts["completed_launches"] += 1
        self._counts[category] += 1

    def _sync(self):
        self._batch._sync()
        self._counts["synchronizations"] += 1

    def _check_open(self):
        if self._closed:
            raise RuntimeError("convex query plan is closed")
        self._batch._check_open()
        self._broad._check_open()

    def close(self):
        with self._batch._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._resources.close()
            finally:
                self._buffers.clear()
                self._kernels = None
                self._patch_kernels = None
                self._broad._convex_dependents.discard(self)

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()

    def _validate(self, view, raw, pose, stamp, pairs):
        s, p, logical = len(self._shape_keys), self._capacity, len(pairs)
        if pose.shape != (s, 12) or pose.dtype != np.float32 or not np.isfinite(pose).all():
            raise RuntimeError("invalid convex pose readback")
        if stamp.shape != (s, 3) or stamp.dtype != np.int32:
            raise RuntimeError("invalid convex pose stamp")
        expected_ticks = {}
        for i, key in enumerate(self._shape_keys):
            slot = view.accepted.slots[key.world]
            if (
                stamp[i, 0] != 1
                or stamp[i, 1] < 0
                or stamp[i, 2] != slot
                or self._attempt_ticks is None
                or stamp[i, 1] != self._attempt_ticks[slot]
            ):
                raise RuntimeError("stale or invalid convex source stamp")
            if key.world in expected_ticks and expected_ticks[key.world] != int(stamp[i, 1]):
                raise RuntimeError("inconsistent same-world convex source ticks")
            expected_ticks[key.world] = int(stamp[i, 1])
        if set(raw) != set(_WIRE):
            raise RuntimeError("invalid convex raw field membership")
        for name, (dtype, tail) in _WIRE.items():
            value = raw[name]
            if (
                not isinstance(value, np.ndarray)
                or value.shape != (p, *tail)
                or value.dtype != dtype
            ):
                raise RuntimeError(f"invalid convex {name} shape or dtype")
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise RuntimeError(f"nonfinite convex {name} readback")
            fill = -1 if name == "material_ids" else 0
            if np.any(value[logical:] != fill):
                raise RuntimeError("invalid convex raw padding")
        for i, pair in enumerate(pairs):
            meta = raw["meta"][i]
            status, relation, reason, phase = map(int, meta[:4])
            if not (
                1 <= status < len(_STATUS)
                and 0 <= relation < len(_RELATION)
                and 0 <= reason < len(_REASON)
                and 0 <= phase < len(_PHASE)
            ):
                raise RuntimeError("invalid convex raw enum or unwritten logical row")
            if meta[12] != self._attempt_tag or meta[13] != i or np.any(meta[14:] != 0):
                raise RuntimeError("convex raw attempt or pair identity mismatch")
            if not (
                0 <= meta[4] <= self._gjk
                and 0 <= meta[5] <= self._epa
                and meta[6] >= 0
                and 0 <= meta[7] <= self._vertices
                and 0 <= meta[8] <= self._faces
                and 0 <= meta[9] <= 7
            ):
                raise RuntimeError("invalid convex raw resource accounting")
            bits = int(meta[9])
            gap, lo, hi, scale = map(float, raw["gap"][i])
            if scale < 1.0:
                raise RuntimeError("invalid convex geometric scale")
            if bits & 1:
                if lo > gap or gap > hi:
                    raise RuntimeError("invalid convex gap bracket")
            elif gap != 0 or lo != 0 or hi != 0:
                raise RuntimeError("invalid convex placeholder gap")
            if bits & 2:
                if abs(float(np.linalg.norm(raw["normal"][i])) - 1) > 2e-4:
                    raise RuntimeError("invalid convex normal")
            elif np.any(raw["normal"][i] != 0):
                raise RuntimeError("invalid convex placeholder normal")
            if status == 1 and bits != 7:
                raise RuntimeError("resolved convex row lacks its finite bilateral witness")
            for side, key in enumerate(pair):
                count = int(meta[10 + side])
                ids = raw["material_ids"][i, side]
                weights = raw["barycentric"][i, side]
                if not 0 <= count <= 4:
                    raise RuntimeError("invalid convex material support count")
                if np.any(ids[count:] != -1) or np.any(weights[count:] != 0):
                    raise RuntimeError("invalid convex material padding")
                registered = view.model.shapes[self._shape_index[key]].vertex_ids
                if (
                    any(int(item) not in registered for item in ids[:count])
                    or len(set(map(int, ids[:count]))) != count
                ):
                    raise RuntimeError(
                        "convex raw contains an unregistered or repeated material ID"
                    )
                if bits & 4:
                    if (
                        count == 0
                        or np.any(weights[:count] < 0)
                        or abs(float(sum(weights[:count])) - 1) > 64 * 2**-23
                    ):
                        raise RuntimeError("invalid convex material barycentric weights")
                    shape_index = self._shape_index[key]
                    start = int(self._source["vertex_offsets"][shape_index])
                    point = np.zeros(3, dtype=np.float64)
                    rotation = pose[shape_index, 3:].astype(np.float64).reshape(3, 3)
                    for vertex, weight in zip(ids[:count], weights[:count], strict=True):
                        slot = start + registered.index(int(vertex))
                        local = self._source["uploaded_local_points_m"][slot].astype(np.float64)
                        point += float(weight) * (
                            pose[shape_index, :3].astype(np.float64) + rotation @ local
                        )
                    point += raw["skin_offset"][i, side].astype(np.float64)
                    if np.linalg.norm(point - raw["surface"][i, side]) > 2e-4 * scale:
                        raise RuntimeError(
                            "convex material witness does not reconstruct its device surface"
                        )
                elif (
                    count
                    or np.any(weights)
                    or np.any(raw["surface"][i, side])
                    or np.any(raw["skin_offset"][i, side])
                ):
                    raise RuntimeError("invalid convex placeholder witness")
            if bits == 7:
                closure = (
                    raw["surface"][i, 1].astype(float)
                    - raw["surface"][i, 0]
                    - gap * raw["normal"][i]
                )
                if np.linalg.norm(closure) > 2e-4 * scale:
                    raise RuntimeError("convex surface/gap closure failed")
        return expected_ticks

    def _read_patch(self):
        names = (
            "patch_meta",
            "patch_weight",
            "patch_gap",
            "patch_normal",
            "patch_common",
            "patch_ids",
            "patch_bary",
            "patch_surface",
            "patch_skin",
            "patch_mcount",
        )
        return {name: self._read(self._buffers[name]) for name in names}

    def _patches(self, view, raw, pose, pairs, patch_raw):
        absent = ConvexQueryPatch("ABSENT", -1, (), False, False, False)
        if patch_raw is None:
            return tuple(absent for _ in pairs), {
                "schema": "resident-convex-patch-v1",
                "enabled": False,
            }
        meta = patch_raw["patch_meta"]
        if meta.shape != (self._capacity, 8) or meta.dtype != np.int32:
            raise RuntimeError("invalid convex patch meta")
        if np.any(meta[len(pairs) :] != 0):
            raise RuntimeError("invalid convex patch padding")
        for name, value in patch_raw.items():
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise RuntimeError(f"nonfinite convex {name} readback")
        built = []
        used = 0
        for index, pair in enumerate(pairs):
            code = int(meta[index, 0])
            if not 0 <= code < len(_PATCH_STATUS):
                raise RuntimeError("invalid convex patch status")
            status = _PATCH_STATUS[code]
            count = int(meta[index, 2])
            offset = int(meta[index, 6])
            if status != "RESOLVED":
                if count != 0 or offset != used or int(meta[index, 4]) or int(meta[index, 3]):
                    raise RuntimeError("invalid unresolved convex patch")
                built.append(ConvexQueryPatch(status, -1, (), False, False, False))
                continue
            if int(raw["meta"][index, 0]) != 1 or int(meta[index, 3]) or not int(meta[index, 4]):
                raise RuntimeError("resolved convex patch is inconsistent with its distance row")
            dimension = int(meta[index, 1])
            if dimension not in (0, 1, 2) or count < dimension + 1 or offset != used:
                raise RuntimeError("invalid convex patch layout")
            if used + count > self._site_capacity:
                raise RuntimeError("convex patch exceeds its site capacity")
            scale = float(raw["gap"][index, 3])
            sites = []
            weights = []
            for site in range(offset, offset + count):
                site_weight = float(patch_raw["patch_weight"][site])
                if site_weight <= 0.0:
                    raise RuntimeError("invalid convex patch site weight")
                weights.append(site_weight)
                site_normal = patch_raw["patch_normal"][site].astype(np.float64)
                if abs(float(np.linalg.norm(site_normal)) - 1.0) > 2e-4:
                    raise RuntimeError("invalid convex patch normal")
                witnesses = []
                for side, key in enumerate(pair):
                    mcount = int(patch_raw["patch_mcount"][site, side])
                    ids = patch_raw["patch_ids"][site, side]
                    bary = patch_raw["patch_bary"][site, side]
                    if (
                        not 1 <= mcount <= 4
                        or np.any(ids[mcount:] != -1)
                        or np.any(bary[mcount:] != 0)
                    ):
                        raise RuntimeError("invalid convex patch material padding")
                    if (
                        np.any(bary[:mcount] < 0)
                        or abs(float(bary[:mcount].sum()) - 1.0) > 64 * 2**-23
                    ):
                        raise RuntimeError("invalid convex patch material weights")
                    registered = view.model.shapes[self._shape_index[key]].vertex_ids
                    if (
                        any(int(item) not in registered for item in ids[:mcount])
                        or len(set(map(int, ids[:mcount]))) != mcount
                    ):
                        raise RuntimeError("convex patch contains an unregistered material ID")
                    shape_index = self._shape_index[key]
                    start = int(self._source["vertex_offsets"][shape_index])
                    point = np.zeros(3, dtype=np.float64)
                    rotation = pose[shape_index, 3:].astype(np.float64).reshape(3, 3)
                    for vertex, weight in zip(ids[:mcount], bary[:mcount], strict=True):
                        slot = start + registered.index(int(vertex))
                        local = self._source["uploaded_local_points_m"][slot].astype(np.float64)
                        point += float(weight) * (
                            pose[shape_index, :3].astype(np.float64) + rotation @ local
                        )
                    point += patch_raw["patch_skin"][site, side].astype(np.float64)
                    surface = patch_raw["patch_surface"][site, side]
                    if np.linalg.norm(point - surface) > 2e-4 * scale:
                        raise RuntimeError(
                            "convex patch material does not reconstruct its device surface"
                        )
                    witnesses.append(
                        ConvexQueryWitness(
                            key,
                            tuple(map(int, ids[:mcount])),
                            tuple(map(float, bary[:mcount])),
                            tuple(map(float, patch_raw["patch_skin"][site, side])),
                            tuple(map(float, surface)),
                            True,
                        )
                    )
                closure = witnesses[1].surface_relative_m
                closure = (
                    np.asarray(closure)
                    - witnesses[0].surface_relative_m
                    - float(patch_raw["patch_gap"][site]) * site_normal
                )
                if np.linalg.norm(closure) > 2e-4 * scale:
                    raise RuntimeError("convex patch surface/gap closure failed")
                common = patch_raw["patch_common"][site].astype(np.float64)
                midpoint = 0.5 * (
                    np.asarray(witnesses[0].surface_relative_m)
                    + np.asarray(witnesses[1].surface_relative_m)
                )
                if np.linalg.norm(common - midpoint) > 2e-4 * scale:
                    raise RuntimeError("convex patch common point is not the surface midpoint")
                sites.append(
                    ConvexPatchSite(
                        site_weight,
                        float(patch_raw["patch_gap"][site]),
                        tuple(map(float, site_normal)),
                        tuple(map(float, common)),
                        witnesses[0],
                        witnesses[1],
                    )
                )
            if abs(float(sum(weights)) - 1.0) > 64 * 2**-23:
                raise RuntimeError("invalid convex patch quadrature")
            built.append(ConvexQueryPatch("RESOLVED", dimension, tuple(sites), True, False, True))
            used += count
        if np.any(patch_raw["patch_ids"][used:] != -1):
            raise RuntimeError("invalid convex patch site padding")
        return tuple(built), {"schema": "resident-convex-patch-v1", "enabled": True, **patch_raw}

    def _build_result(
        self, view, raw, pose, stamp, pairs, attempt_delta, broadphase_delta, patch_raw
    ):
        """Fault seam: return a wholly owned record before either publication."""
        ticks = self._validate(view, raw, pose, stamp, pairs)
        patches, patch = self._patches(view, raw, pose, pairs, patch_raw)
        rows = []
        for i, pair in enumerate(pairs):
            meta, gap = raw["meta"][i], raw["gap"][i]
            bits = int(meta[9])
            witnesses = []
            for side, key in enumerate(pair):
                count = int(meta[10 + side])
                witnesses.append(
                    ConvexQueryWitness(
                        key,
                        tuple(map(int, raw["material_ids"][i, side, :count])),
                        tuple(map(float, raw["barycentric"][i, side, :count])),
                        tuple(map(float, raw["skin_offset"][i, side])),
                        tuple(map(float, raw["surface"][i, side])),
                        bool(bits & 4),
                    )
                )
            origin = self._source["world_origins_m"][self._world_index[pair[0].world]]
            rows.append(
                ConvexQueryRow(
                    pair,
                    _STATUS[int(meta[0])],
                    _RELATION[int(meta[1])],
                    _REASON[int(meta[2])],
                    _PHASE[int(meta[3])],
                    *map(float, gap),
                    tuple(map(float, origin)),
                    tuple(map(float, raw["normal"][i])),
                    bool(bits & 1),
                    bool(bits & 2),
                    bool(bits & 4),
                    *witnesses,
                    *map(int, meta[4:9]),
                    patches[i],
                )
            )
        requested = pairs if self._requested is None else self._requested
        index = {pair: i for i, pair in enumerate(pairs)}
        mapping = tuple(index[_canonical(pair)] for pair in requested)
        reverse = tuple(tuple(pair) != _canonical(pair) for pair in requested)
        source = dict(
            self._source,
            pose=pose,
            stamp=stamp,
            accepted_ticks=self._attempt_ticks,
            accepted_slots=dict(view.accepted.slots),
            owned_buffer_layout=self._owned_layout,
        )
        stats = {
            "attempt_delta": dict(attempt_delta),
            "broadphase_delta": dict(broadphase_delta),
            "logical_pairs": len(pairs),
            "padded_pairs": self._capacity,
            "mode": self._batch._mode,
        }
        return ConvexQueryResult(
            view.owner,
            view.registration,
            view.accepted.topology_generation,
            view.accepted.state_epoch,
            view.accepted.active,
            self._world_keys,
            np.asarray([ticks[world] for world in self._world_keys], dtype=np.int32),
            self._shape_keys,
            self._shape_revisions,
            "host_broadphase" if self._requested is None else "explicit",
            requested,
            mapping,
            reverse,
            tuple(rows),
            raw,
            source,
            stats,
            self._attempt_tag,
            patch,
        )

    def _stage_query(self, staged_broadphase=None):
        """Build owned geometry without publishing either query.

        Contact passes its just-staged broad result, pending layouts and entry
        bank. The public query path stages its own broad phase as before.
        """
        with self._batch._lock:
            self._check_open()
            before = dict(self._counts)
            self._counts["attempts"] += 1
            self._attempt_tag = self._attempt_tag % 2147483646 + 1
            try:
                with self._broad._borrow_convex_state() as view:
                    if (
                        view.owner is not self._owner
                        or view.registration is not self._registration
                        or view.accepted.topology_generation != self._topology_generation
                        or tuple(view.shape_keys) != self._shape_keys
                        or tuple(view.shape_revisions) != self._shape_revisions
                        or tuple(sorted(view.allowed_pairs)) != self._allowed
                        or not np.array_equal(view.shape_layouts, self._source["shape_layouts"])
                        or not np.array_equal(view.shape_rows, self._source["shape_rows"])
                        or not np.array_equal(view.shape_groups, self._source["shape_groups"])
                    ):
                        raise ValueError("convex plan source registration has changed")
                    broad_before = dict(self._broad.stats())
                    broad_result = pending = None
                    if staged_broadphase is not None:
                        if self._requested is not None:
                            raise ValueError("staged contact geometry cannot use explicit query pairs")
                        broad_result, pending, entry_bank = staged_broadphase
                        self._broad.require_current(broad_result)
                        if (entry_bank != view.accepted.active
                                or broad_result.owner is not view.owner
                                or broad_result.registration is not view.registration
                                or broad_result.state_epoch != view.accepted.state_epoch
                                or broad_result.topology_generation != view.accepted.topology_generation
                                or broad_result.shape_keys != tuple(view.shape_keys)
                                or broad_result.shape_revisions != tuple(view.shape_revisions)
                                or broad_result.registered_pairs != self._broad._registered_pairs
                                or broad_result.world_keys != tuple(view.world_keys)):
                            raise ValueError("staged broad phase differs from the entry source")
                        pairs = tuple(sorted(tuple(pair) for pair in broad_result.candidates))
                        if len(set(pairs)) != len(pairs) or any(pair != _canonical(pair) for pair in pairs):
                            raise ValueError("staged broad phase candidate membership differs")
                    elif self._requested is None:
                        broad_result, pending = view.stage_broadphase()
                        pairs = tuple(sorted(tuple(pair) for pair in broad_result.candidates))
                    else:
                        pairs = tuple(sorted(_canonical(pair) for pair in self._requested))
                    if any(pair not in self._allowed for pair in pairs):
                        raise RuntimeError("broad phase returned a forbidden convex pair")
                    if len(pairs) > self._capacity:
                        raise ConvexQueryCapacityExceeded(
                            len(pairs), self._capacity, Counter(pair[0].world for pair in pairs)
                        )
                    if self._shape_keys:
                        b = self._buffers
                        for layout in self._source_layouts:
                            source = view.sources[layout]
                            self._launch(
                                self._kernels["gather"][view.accepted.active],
                                "pose_gather_launches",
                                len(self._shape_keys),
                                layout,
                                int(layout >= 2),
                                b["layouts"],
                                b["tags"],
                                b["rows"],
                                b["groups"],
                                b["offsets"],
                                b["points"],
                                source.meta,
                                source.fk,
                                view.accepted.ticks,
                                b["pose"],
                                b["stamp"],
                                b["world_points"],
                            )
                        if self._capacity:
                            self._upload_pairs(pairs)
                            self._launch(
                                self._kernels["clear"],
                                "query_launches",
                                self._capacity,
                                *(b[name] for name in _WIRE),
                            )
                            for first in range(0, len(pairs), 32):
                                self._launch(
                                    self._kernels["query"],
                                    "query_launches",
                                    min(32, len(pairs) - first),
                                    first,
                                    self._attempt_tag,
                                    self._gjk,
                                    self._epa,
                                    self._vertices,
                                    self._faces,
                                    b["tags"],
                                    b["ids"],
                                    b["radius"],
                                    b["precision"],
                                    b["world_points"],
                                    b["pairs"],
                                    b["vertices"],
                                    b["vertex_slots"],
                                    b["faces"],
                                    b["face_geometry"],
                                    b["edges"],
                                    b["weights"],
                                    b["indices"],
                                    b["simplex_copy"],
                                    b["simplex_slots"],
                                    *(b[name] for name in _WIRE),
                                )
                            if self._patch_kernels is not None:
                                self._launch(
                                    self._patch_kernels["clear"],
                                    "query_launches",
                                    self._capacity,
                                    self._site_capacity,
                                    b["patch_meta"],
                                    b["patch_weight"],
                                    b["patch_gap"],
                                    b["patch_normal"],
                                    b["patch_common"],
                                    b["patch_ids"],
                                    b["patch_bary"],
                                    b["patch_surface"],
                                    b["patch_skin"],
                                    b["patch_mcount"],
                                )
                                base = 0
                                for index in range(len(pairs)):
                                    self._launch(
                                        self._patch_kernels["fill"],
                                        "query_launches",
                                        index,
                                        base,
                                        self._site_capacity,
                                        b["tags"],
                                        b["ids"],
                                        b["radius"],
                                        b["world_points"],
                                        b["pairs"],
                                        b["meta"],
                                        b["gap"],
                                        b["normal"],
                                        b["patch_work"],
                                        b["patch_work_i"],
                                        b["patch_meta"],
                                        b["patch_weight"],
                                        b["patch_gap"],
                                        b["patch_normal"],
                                        b["patch_common"],
                                        b["patch_ids"],
                                        b["patch_bary"],
                                        b["patch_surface"],
                                        b["patch_skin"],
                                        b["patch_mcount"],
                                    )
                                    self._sync()
                                    written = self._read(b["patch_meta"])
                                    base += (
                                        int(written[index, 2]) if int(written[index, 0]) == 1 else 0
                                    )
                        self._sync()
                        pose, stamp = self._read(b["pose"]), self._read(b["stamp"])
                        self._attempt_ticks = self._read(view.accepted.ticks)
                        if self._attempt_ticks.dtype != np.int32 or self._attempt_ticks.shape != (
                            len(view.accepted.world_keys),
                        ):
                            raise RuntimeError("invalid accepted tick readback")
                    else:
                        pose, stamp = np.empty((0, 12), np.float32), np.empty((0, 3), np.int32)
                        self._attempt_ticks = np.empty(0, dtype=np.int32)
                    raw = self._read_raw()
                    patch_raw = self._read_patch() if self._patch_kernels is not None else None
                    broad_after = dict(self._broad.stats())
                    broad_delta = {
                        key: value - broad_before[key]
                        for key, value in broad_after.items()
                        if type(value) in (int, float)
                        and type(broad_before.get(key)) in (int, float)
                    }
                    delta = {key: value - before[key] for key, value in self._counts.items()}
                    delta["queries"] = 1
                    result = self._build_result(
                        view, raw, pose, stamp, pairs, delta, broad_delta, patch_raw
                    )
                    if staged_broadphase is not None:
                        expected_ticks = dict(zip(broad_result.world_keys, broad_result.ticks, strict=True))
                        if any(int(result.ticks[i]) != int(expected_ticks[world])
                               for i, world in enumerate(result.world_keys)):
                            raise ValueError("staged broad phase ticks differ from convex entry")
                    return result, broad_result, pending
            except BaseException:
                self._counts["failed_queries"] += 1
                raise
            finally:
                self._attempt_ticks = None

    def _publish_staged(self, result):
        """Assignment-only publication; all geometry construction precedes swap."""
        self._last_result = result
        self._counts["queries"] += 1

    def query(self):
        with self._batch._lock:
            result, broad_result, pending = self._stage_query()
            if broad_result is not None:
                self._broad._publish_query(broad_result, pending)
            self._publish_staged(result)
            return result

    def require_current(self, result):
        with self._batch._lock:
            self._check_open()
            if not isinstance(result, ConvexQueryResult):
                raise TypeError("expected a ConvexQueryResult")
            with self._broad._borrow_convex_state() as view:
                if (
                    result.owner is not self._owner
                    or result.registration is not self._registration
                    or result.topology_generation != view.accepted.topology_generation
                    or result.state_epoch != view.accepted.state_epoch
                    or result.active != view.accepted.active
                    or result.shape_keys != tuple(view.shape_keys)
                    or result.shape_revisions != tuple(view.shape_revisions)
                ):
                    raise ValueError("convex result belongs to another or stale accepted state")
            for key, expected in self._source.items():
                actual = result.source.get(key)
                if isinstance(expected, np.ndarray):
                    valid = (isinstance(actual, np.ndarray) and actual.dtype == expected.dtype
                             and np.array_equal(actual, expected))
                elif key == "model":
                    valid = actual is expected
                elif key == "bindings":
                    valid = (isinstance(actual, tuple) and len(actual) == len(expected)
                             and all(a is b for a, b in zip(actual, expected, strict=True)))
                else:
                    valid = actual == expected
                if not valid:
                    raise ValueError("convex result immutable source binding differs: " + key)
            return result


def convex_query_for(batch, plan):
    if not isinstance(plan, ConvexQueryPlan) or plan._batch is not batch:
        raise ValueError("convex query plan belongs to another batch")
    return plan.query()
