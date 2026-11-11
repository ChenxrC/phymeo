"""Automatic current-snapshot boundary proximity; no solid/CCD/load semantics."""

import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from types import MappingProxyType

import numpy as np

from ..backends._device_triangle_distance import DeviceTriangleDistanceBatch
from ..constraints.collision_shapes import CollisionMotion
from ._collision_spatial import WORD_BITS


@dataclass(frozen=True)
class BoundaryDistanceLimits:
    groups: int = 32
    candidates: int = 8192
    vertices: int = 512
    faces: int = 128
    segments: int = 64

    def __post_init__(self):
        for name, ceiling in (
            ("groups", 32),
            ("candidates", 8192),
            ("vertices", 512),
            ("faces", 128),
            ("segments", 64),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= ceiling:
                raise ValueError(name + " must be a builtin integer within its hard capacity")


class BoundaryDistanceCapacityError(ValueError):
    def __init__(self, resource, required, limit):
        self.resource, self.required, self.limit = resource, int(required), int(limit)
        super().__init__(f"boundary {resource} capacity: required {required}, limit {limit}")


@dataclass(frozen=True)
class BoundaryDistancePair:
    first: object
    second: object
    status: str
    reason: str
    group: int | None = None
    segment_id: int | None = None
    facet_ids: tuple = ()
    row_indices: tuple = ()


@dataclass(frozen=True)
class BoundaryDistanceGroup:
    surface: object
    round_keys: tuple
    round_vertex_ids: tuple
    facet_vertices: tuple
    candidates: np.ndarray
    leaf_bounds: np.ndarray
    query_bounds: np.ndarray
    bvh_raw: np.ndarray
    packet: object
    narrow_counters: Mapping


@dataclass(frozen=True)
class BoundaryDistances:
    schema: str
    semantics: str
    owner: str
    sequence: int
    backend: str
    source_binding: str
    margin: float
    limits: BoundaryDistanceLimits
    source: Mapping
    pairs: tuple
    groups: tuple
    required_candidates: int


def _array(value):
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)


def _owned(value):
    if isinstance(value, np.ndarray):
        return _array(value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _owned(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_owned(item) for item in value)
    return value


def _key(key):
    return (key.world, key.body, key.shape)


def _margin(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("boundary margin must be a real scalar, not bool")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("boundary margin must be finite and nonnegative")
    return result


def _snapshot(workspace, motion, margin):
    if workspace._backend not in ("interpret", "cuda"):
        raise ValueError("boundary distances require the explicit interpret or CUDA path")
    if workspace._world_partitioned:
        raise ValueError("boundary dispatch requires a complete public CollisionModel")
    if not isinstance(motion, Mapping):
        raise TypeError("motion must map every original CollisionKey")
    paths = dict(motion)
    shapes = workspace._model.shapes
    if set(paths) != {shape.key for shape in shapes}:
        raise ValueError("motion keys must exactly cover the complete model")
    registrations = []
    h = hashlib.sha256(b"phymeo-boundary-snapshot-v1\0")
    for shape in shapes:
        if shape.kind not in ("sphere", "capsule", "triangle", "mesh"):
            raise ValueError("unsupported boundary shape family: " + shape.kind)
        if shape.kind in ("triangle", "mesh") and shape.radius != 0:
            raise ValueError("boundary surfaces require zero skin")
        path = paths[shape.key]
        if type(path) is not CollisionMotion or path.start.shape != (len(shape.vertex_ids), 3):
            raise ValueError("complete material-order CollisionMotion required")
        if path.turn is not None or path.body_orientation_xyzw is not None:
            raise ValueError(
                "boundary distances accept world vertices, not a turn/body orientation"
            )
        if path.end is not None and not np.array_equal(path.start, path.end):
            raise ValueError("boundary distances are current-pose only")
        registration = dict(
            key=_key(shape.key),
            kind=shape.kind,
            vertex_ids=shape.vertex_ids,
            triangles=shape.triangles,
            radius=shape.radius,
            revision=shape.revision,
            group=shape.group,
            mask=shape.mask,
            has_end=path.end is not None,
        )
        registrations.append(registration)
        h.update(json.dumps(registration, sort_keys=True, separators=(",", ":")).encode())
        h.update(path.start.tobytes())
        if path.end is not None:
            h.update(path.end.tobytes())
    exclusions = tuple((_key(a), _key(b)) for a, b in workspace._model.excluded)
    h.update(
        json.dumps(
            dict(excluded=exclusions, margin=margin), sort_keys=True, separators=(",", ":")
        ).encode()
    )
    h.update(workspace._pairs.tobytes())
    h.update(workspace._filtered_codes.tobytes())
    return (
        paths,
        h.hexdigest(),
        _owned(dict(registrations=registrations, excluded=exclusions, motion=paths)),
    )


def _bounds(points, expansion=Fraction(0)):
    # Original stored FP64 inputs, expanded exactly before directed FP32 storage.
    # This is a bounding-box pack, never a HOST closest-feature calculation.
    from .collision_pipeline import _outward

    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    return np.asarray(
        [_outward(Fraction(float(value)) - expansion, False) for value in lower]
        + [_outward(Fraction(float(value)) + expansion, True) for value in upper],
        np.float32,
    )


class BoundaryDistanceDispatcher:
    """One workspace owns an independent COL2 runtime and a committed aggregate.

    The snapshot binds HOST coordinates, not a resident device epoch or loan.
    Private earlier group packets may exist when a later group fails; only the
    whole aggregate and pending BVH layouts enter the workspace transaction.
    """

    def __init__(self, workspace):
        self._workspace = workspace
        self._owner = uuid.uuid4().hex
        self._last = None
        self._batch = None
        self._closed = False
        self._blocked = None
        self._last_attempt = None

    @property
    def last(self):
        with self._workspace._lock:
            return self._last

    @property
    def last_attempt(self):
        with self._workspace._lock:
            return self._last_attempt

    def _ready(self):
        if self._closed or self._workspace._closed:
            raise RuntimeError("boundary dispatcher/workspace is closed")
        if self._blocked is not None:
            raise RuntimeError(
                "boundary dispatcher retains UNKNOWN_RELEASE owners"
            ) from self._blocked

    def _publish(self, packet):
        self._last = packet

    def _groups(self, paths):
        ws = self._workspace
        shapes = ws._model.shapes
        pairs = []
        grouped = {}
        reasons = ("candidate", "different_world", "same_rigid_body", "mask", "excluded")
        for (i, j, _), code in zip(ws._pairs, ws._filtered_codes, strict=True):
            a, b = shapes[int(i)], shapes[int(j)]
            if code:
                pairs.append(BoundaryDistancePair(a.key, b.key, "filtered", reasons[int(code)]))
                continue
            a_round = a.kind in ("sphere", "capsule")
            b_round = b.kind in ("sphere", "capsule")
            if a_round == b_round:
                pairs.append(BoundaryDistancePair(a.key, b.key, "not_applicable", "same_family"))
                continue
            surface, round_shape = (b, a) if a_round else (a, b)
            slot = len(pairs)
            pairs.append(None)
            grouped.setdefault(surface.key, dict(shape=surface, rounds=[])).get("rounds").append(
                (round_shape, slot, a.key, b.key)
            )
        prepared = []
        for surface_key, group in sorted(grouped.items()):
            shape = group["shape"]
            items = sorted(group["rounds"], key=lambda row: row[0].key)
            faces = (shape.vertex_ids,) if shape.kind == "triangle" else shape.triangles
            by_id = {value: index for index, value in enumerate(shape.vertex_ids)}
            face_rows = np.array([[by_id[value] for value in face] for face in faces], np.int64)
            vertices = paths[surface_key].start
            segments = np.array(
                [
                    np.repeat(paths[s.key].start, 2, axis=0)
                    if s.kind == "sphere"
                    else paths[s.key].start
                    for s, _, _, _ in items
                ],
                np.float64,
            )
            prepared.append(
                dict(
                    shape=shape,
                    items=items,
                    facets=faces,
                    faces=face_rows,
                    vertices=vertices,
                    segments=segments,
                )
            )
        return pairs, prepared

    def query(self, motion, *, margin=0.0, limits=None):
        ws = self._workspace
        with ws._lock:
            context = ws.transaction() if ws._draft is None else nullcontext(ws)
            with context:
                self._ready()
                margin = _margin(margin)
                limits = BoundaryDistanceLimits() if limits is None else limits
                if type(limits) is not BoundaryDistanceLimits:
                    raise TypeError("limits require immutable BoundaryDistanceLimits")
                paths, binding, source = _snapshot(ws, motion, margin)
                pairs, prepared = self._groups(paths)
                requirements = dict(
                    groups=len(prepared),
                    vertices=max((len(g["vertices"]) for g in prepared), default=0),
                    faces=max((len(g["faces"]) for g in prepared), default=0),
                    segments=max((len(g["segments"]) for g in prepared), default=0),
                )
                self._last_attempt = _owned(
                    dict(source_binding=binding, requirements=requirements, groups=())
                )
                for name, count in requirements.items():
                    if count > getattr(limits, name):
                        raise BoundaryDistanceCapacityError(name, count, getattr(limits, name))
                pending = ws._draft.layouts.copy()
                total = 0
                evidence = []
                # Finish every device BVH query and capacity check before any narrow query.
                for group in prepared:
                    shape = group["shape"]
                    leaves = np.asarray(
                        [_bounds(group["vertices"][face]) for face in group["faces"]], np.float32
                    )
                    queries = np.asarray(
                        [
                            _bounds(axis, Fraction(s.radius) + Fraction(margin))
                            for axis, (s, _, _, _) in zip(
                                group["segments"], group["items"], strict=True
                            )
                        ],
                        np.float32,
                    )
                    layout_key = (
                        "mesh",
                        "boundary-distance",
                        shape.key,
                        shape.revision,
                        shape.vertex_ids,
                        group["facets"],
                    )
                    try:
                        raw = ws._trees.hits(pending, layout_key, leaves, queries, None, ws._engine)
                        candidates = []
                        for segment, words in enumerate(raw[:, :-2]):
                            for word_index, word in enumerate(words):
                                bits = int(word)
                                while bits:
                                    bit = bits & -bits
                                    facet = word_index * WORD_BITS + bit.bit_length() - 1
                                    if facet >= len(group["faces"]):
                                        raise RuntimeError(
                                            "BVH returned an out-of-range original facet"
                                        )
                                    candidates.append((segment, facet))
                                    bits -= bit
                    except BaseException as error:
                        self._last_attempt = _owned(
                            dict(
                                source_binding=binding,
                                requirements={**requirements, "candidates": total},
                                groups=evidence,
                                failed_bvh_surface=shape.key,
                                error=repr(error),
                            )
                        )
                        try:
                            ws._close_locked()
                        except BaseException as close_error:
                            raise BaseExceptionGroup(
                                "boundary BVH and cleanup failed", [error, close_error]
                            ) from error
                        raise
                    candidates = np.asarray(candidates, np.int64).reshape(-1, 2)
                    group.update(
                        leaves=leaves, queries=queries, raw=_array(raw), candidates=candidates
                    )
                    total += len(candidates)
                    evidence.append(
                        dict(
                            surface=shape.key,
                            leaf_bounds=leaves,
                            query_bounds=queries,
                            bvh_raw=raw,
                            candidates=candidates,
                            packet=None,
                        )
                    )
                    self._last_attempt = _owned(
                        dict(
                            source_binding=binding,
                            requirements={**requirements, "candidates": total},
                            groups=evidence,
                        )
                    )
                self._last_attempt = _owned(
                    dict(
                        source_binding=binding,
                        requirements={**requirements, "candidates": total},
                        groups=evidence,
                    )
                )
                if total > limits.candidates:
                    raise BoundaryDistanceCapacityError("candidates", total, limits.candidates)
                groups = []
                try:
                    if prepared and self._batch is None:
                        self._batch = DeviceTriangleDistanceBatch(
                            mode=ws._backend,
                            vertex_capacity=512,
                            face_capacity=128,
                            segment_capacity=64,
                            pair_capacity=8192,
                        )
                    for index, group in enumerate(prepared):
                        shape = group["shape"]
                        items = group["items"]
                        inputs = dict(
                            mesh_key="boundary:" + ":".join(map(str, _key(shape.key))),
                            revision=shape.revision,
                            snapshot_id=binding,
                            vertex_ids=np.asarray(shape.vertex_ids, np.int64),
                            facet_ids=np.arange(len(group["facets"]), dtype=np.int64),
                            faces=group["faces"],
                            vertices=group["vertices"],
                            segment_ids=np.arange(len(items), dtype=np.int64),
                            segments=group["segments"],
                            radii=np.asarray([s.radius for s, _, _, _ in items], np.float64),
                            candidates=group["candidates"],
                            origin_m=group["vertices"][0],
                        )
                        packet = self._batch.query(**inputs)
                        self._batch.require_current(packet, **inputs)
                        if len(packet.rows) != len(group["candidates"]) or not np.array_equal(
                            packet.inputs["candidates"], group["candidates"]
                        ):
                            raise RuntimeError(
                                "narrow phase changed complete BVH candidate membership"
                            )
                        groups.append(
                            BoundaryDistanceGroup(
                                shape.key,
                                tuple(s.key for s, _, _, _ in items),
                                tuple(s.vertex_ids for s, _, _, _ in items),
                                tuple(group["facets"]),
                                _array(group["candidates"]),
                                _array(group["leaves"]),
                                _array(group["queries"]),
                                group["raw"],
                                packet,
                                _owned(self._batch.stats),
                            )
                        )
                        evidence[index]["packet"] = packet
                        self._last_attempt = _owned(
                            dict(
                                source_binding=binding,
                                requirements={**requirements, "candidates": total},
                                groups=evidence,
                            )
                        )
                        for segment, (_, slot, first, second) in enumerate(items):
                            selected = tuple(
                                i
                                for i, (sid, _) in enumerate(group["candidates"])
                                if sid == segment
                            )
                            facets = tuple(int(group["candidates"][i, 1]) for i in selected)
                            status = "boundary_candidates" if selected else "no_boundary_candidate"
                            pairs[slot] = BoundaryDistancePair(
                                first,
                                second,
                                status,
                                "boundary_band",
                                index,
                                segment,
                                facets,
                                selected,
                            )
                except BaseException as error:
                    if self._batch is not None:
                        self._last_attempt = _owned(
                            dict(
                                source_binding=binding,
                                requirements={**requirements, "candidates": total},
                                groups=evidence,
                                failed_narrow_raw=self._batch.last_attempt_raw,
                                narrow_counters=self._batch.stats,
                                error=repr(error),
                            )
                        )
                        if self._batch.stats["live_bytes"]:
                            self._blocked = error
                    raise
                if any(pair is None for pair in pairs):
                    raise RuntimeError("incomplete public pair table")
                prior = ws._draft.publications.get(self._publish, self._last)
                result = BoundaryDistances(
                    "phymeo-boundary-distances-v1",
                    "threshold-limited-boundary-distance",
                    self._owner,
                    1 if prior is None else prior.sequence + 1,
                    ws._backend,
                    binding,
                    margin,
                    limits,
                    source,
                    tuple(pairs),
                    tuple(groups),
                    total,
                )
                ws._draft.layouts = pending
                ws._stage_publication(self._publish, result)
                return result

    def require_current(self, packet, motion, *, margin=0.0):
        with self._workspace._lock:
            self._ready()
            if type(packet) is not BoundaryDistances or packet.owner != self._owner:
                raise ValueError("boundary packet owner mismatch")
            if packet is not self._last:
                raise ValueError("boundary packet is not current (uncommitted or superseded)")
            if _snapshot(self._workspace, motion, _margin(margin))[1] != packet.source_binding:
                raise ValueError("boundary packet source binding mismatch")
            return None

    def close(self):
        with self._workspace._lock:
            if not self._closed:
                if self._batch is not None:
                    # Its ledger retains unknown owners and never retries opaque free.
                    try:
                        self._batch.close()
                    except BaseException as error:
                        self._blocked = error
                        raise
                self._closed = True
