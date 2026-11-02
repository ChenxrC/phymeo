"""Finite, mixed-family collision queries. Candidate coverage is independent of narrow support."""

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from threading import Lock

import numpy as np

from ..constraints.collision_shapes import (
    CollisionKey,
    CollisionModel,
    CollisionMotion,
    CollisionPolicy,
)
from ..core.capabilities import SolverFailure
from .surface_features import closest_point_triangle, closest_segment_segment

_REASONS = ("candidate", "different_world", "same_rigid_body", "mask", "excluded", "aabb")
_F32_MAX = Fraction(float(np.finfo(np.float32).max))
_F32_TINY = Fraction(float(np.finfo(np.float32).tiny))


@dataclass(frozen=True)
class CollisionBounds:
    key: CollisionKey
    revision: int
    lower: tuple[float, ...]
    upper: tuple[float, ...]


@dataclass(frozen=True)
class CollisionWitness:
    """One feasible material representative, with normal oriented from first to second."""

    gap_m: float
    normal: tuple[float, ...] | None
    first_point: tuple[float, ...] | None
    second_point: tuple[float, ...] | None
    first_weights: tuple[tuple[int, float], ...]
    second_weights: tuple[tuple[int, float], ...]
    source: str = "host-fp64"


@dataclass(frozen=True)
class CollisionPair:
    first: CollisionKey
    second: CollisionKey
    family: str
    candidate: bool
    status: str
    reason: str
    witness: CollisionWitness | None = None


@dataclass(frozen=True)
class CollisionDetection:
    status: str
    backend: str
    required_candidates: int
    pairs: tuple[CollisionPair, ...]
    bounds: tuple[CollisionBounds, ...]


def _outward(value, upper):
    if abs(value) > _F32_MAX:
        raise ValueError("collision bound exceeds finite FP32 storage")
    if 0 < abs(value) < _F32_TINY:
        if upper:
            return np.finfo(np.float32).tiny if value > 0 else 0.0
        return -np.finfo(np.float32).tiny if value < 0 else 0.0
    rounded = np.float32(float(value))
    exact = Fraction(float(rounded))
    if (upper and exact < value) or (not upper and exact > value):
        rounded = np.nextafter(rounded, np.float32(np.inf if upper else -np.inf))
    return rounded


def _envelopes(model, paths, margin):
    """Prepare outward FP32 per-vertex intervals from the original stored coordinates.

    A rotating vertex never extends farther along any axis than its original
    L1 distance to the pivot. This remains true throughout the arc, including
    for a moving pivot and multiple turns. The shape reduction happens on the
    selected backend.
    """
    packed = np.empty((sum(len(s.vertex_ids) for s in model.shapes), 6), np.float32)
    row = 0
    material_paths = {}
    for shape in model.shapes:
        path = paths[shape.key]
        if not isinstance(path, CollisionMotion) or path.start.shape != (len(shape.vertex_ids), 3):
            raise ValueError("motion must supply all registered vertices in material order")
        turn = path.turn
        if turn is not None and not shape.rigid:
            raise ValueError("rigid turn requires a rigid shape")
        pad = Fraction(shape.radius) + Fraction(margin) / 2
        translation = None
        for i, p in enumerate(path.start):
            start = tuple(map(Fraction, p))
            if turn is None:
                end = start if path.end is None else tuple(map(Fraction, path.end[i]))
                if shape.rigid:
                    delta = tuple(y - x for x, y in zip(start, end, strict=True))
                    if translation is not None and delta != translation:
                        raise ValueError(
                            "rigid motion needs a common translation or an explicit rigid turn"
                        )
                    translation = delta
                else:
                    vertex = (shape.key.world, shape.key.body, shape.vertex_ids[i])
                    if material_paths.setdefault(vertex, (start, end)) != (start, end):
                        raise ValueError("shared material vertex has inconsistent motion")
                low, high = start, end
                extent = pad
            else:
                origin = tuple(map(Fraction, turn.pivot_start))
                target = tuple(map(Fraction, turn.pivot_end))
                if turn.angle == 0:
                    low = start
                    high = tuple(p0 + q - o for p0, o, q in zip(start, origin, target, strict=True))
                    extent = pad
                else:
                    low, high = origin, target
                    extent = pad + sum(abs(x - y) for x, y in zip(start, origin, strict=True))
            for k in range(3):
                packed[row, k] = _outward(min(low[k], high[k]) - extent, False)
                packed[row, k + 3] = _outward(max(low[k], high[k]) + extent, True)
            row += 1
    return packed


def _host_query(offsets, tags, pairs, envelopes):
    bounds = np.empty((len(tags), 6), np.float32)
    for i, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        vertices = envelopes[lo:hi]
        bounds[i, :3] = vertices[:, :3].min(axis=0)
        bounds[i, 3:] = vertices[:, 3:].max(axis=0)
    flags = np.zeros(len(pairs), np.int32)
    for row, (a, b, excluded) in enumerate(pairs):
        if tags[a, 0] != tags[b, 0]:
            flags[row] = 1
        elif tags[a, 2] and tags[a, 1] == tags[b, 1]:
            flags[row] = 2
        elif not (tags[a, 3] & tags[b, 4] and tags[b, 3] & tags[a, 4]):
            flags[row] = 3
        elif excluded:
            flags[row] = 4
        elif np.any(bounds[a, :3] > bounds[b, 3:]) or np.any(bounds[b, :3] > bounds[a, 3:]):
            flags[row] = 5
    return bounds, flags


def _address(shape, weights):
    return tuple(
        sorted((node, float(w)) for node, w in zip(shape.vertex_ids, weights, strict=True))
    )


def _axis_pair(a, b, x, y):
    if len(x) == len(y) == 2 and not np.array_equal(x[0], x[1]) and not np.array_equal(y[0], y[1]):
        try:
            pair = closest_segment_segment(x, y)
        except SolverFailure:
            pass  # Conditioned FP64 query declined. Solve the same two closed axes below.
        else:
            return (
                pair.first,
                pair.second,
                pair.second - pair.first,
                _address(a, pair.first_weights),
                _address(b, pair.second_weights),
                "host-fp64",
            )

    # The squared distance between two closed axes is a convex quadratic on [0,1]^2.
    # Check its stationary point and each boundary minimum on the stored coordinates.
    p, p1, q, q1 = (
        np.array([Fraction(float(z)) for z in point], dtype=object)
        for point in (x[0], x[-1], y[0], y[-1])
    )
    u, v, w = p1 - p, q1 - q, p - q
    uu, uv, vv, uw, vw = u @ u, u @ v, v @ v, u @ w, v @ w
    half = Fraction(1, 2)

    def clamp(z):
        return min(1, max(0, z))

    choices = [(s, clamp((vw + s * uv) / vv) if vv else half) for s in ((0, 1) if uu else (half,))]
    choices.extend(
        (clamp((t * uv - uw) / uu) if uu else half, t) for t in ((0, 1) if vv else (half,))
    )
    determinant = uu * vv - uv * uv
    if determinant:
        s, t = (uv * vw - vv * uw) / determinant, (uu * vw - uv * uw) / determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            choices.append((s, t))

    def squared(pair):
        s, t = pair
        residual = w + s * u - t * v
        return residual @ residual

    s, t = min(choices, key=squared)
    first, second = p + s * u, q + t * v
    return (
        np.asarray(first, float),
        np.asarray(second, float),
        np.asarray(second - first, float),
        _address(a, (1,) if len(x) == 1 else (1 - s, s)),
        _address(b, (1,) if len(y) == 1 else (1 - t, t)),
        "host-rational-axis",
    )


def _static_witness(a, b, x, y):
    if a.kind in {"sphere", "capsule"} and b.kind in {"sphere", "capsule"}:
        p, q, delta, wa, wb, source = _axis_pair(a, b, x, y)
    elif {a.kind, b.kind} == {"sphere", "triangle"}:
        sphere_first = a.kind == "sphere"
        face = b if sphere_first else a
        if face.radius != 0:
            return None
        pair = closest_point_triangle(x[0], y) if sphere_first else closest_point_triangle(y[0], x)
        if sphere_first:
            p, q = pair.first, pair.second
            wa, wb = _address(a, (1.0,)), _address(b, pair.second_weights)
        else:
            p, q = pair.second, pair.first
            wa, wb = _address(a, pair.second_weights), _address(b, (1.0,))
        delta, source = q - p, "host-fp64"
    else:
        return None
    distance = float(np.hypot.reduce(delta))
    gap = distance - (a.radius + b.radius)
    if distance == 0:
        return CollisionWitness(gap, None, None, None, wa, wb, source)
    normal = delta / distance
    return CollisionWitness(
        gap,
        tuple(map(float, normal)),
        tuple(map(float, p + a.radius * normal)),
        tuple(map(float, q - b.radius * normal)),
        wa,
        wb,
        source,
    )


def _pair(a, b, ma, mb, code):
    family = "RR" if a.rigid and b.rigid else "RD" if a.rigid != b.rigid else "DD"
    fields = (a.key, b.key, family)
    if code:
        return CollisionPair(
            *fields, False, "separated" if code == 5 else "filtered", _REASONS[code]
        )
    if ma.moving or mb.moving:
        return CollisionPair(*fields, True, "unsupported", "swept_narrow_phase")
    try:
        witness = _static_witness(a, b, ma.start, mb.start)
    except SolverFailure as error:
        return CollisionPair(*fields, True, "unresolved", str(error))
    except ValueError as error:
        if str(error) not in {"degenerate triangle", "zero extent geometry"}:
            raise
        return CollisionPair(*fields, True, "unresolved", str(error))
    if witness is None:
        return CollisionPair(*fields, True, "unsupported", "shape_pair")
    gap = witness.gap_m
    status = "overlap" if gap < 0 else "touching" if gap == 0 else "separated"
    return CollisionPair(
        *fields,
        True,
        status,
        "nonunique_normal" if witness.normal is None else "narrow_phase",
        witness,
    )


class CollisionWorkspace:
    """One immutable model, one explicit broad-phase backend and one publication."""

    def __init__(self, model, backend="host", *, cache_root=None):
        if not isinstance(model, CollisionModel):
            raise TypeError("collision workspace requires a CollisionModel")
        if backend not in ("host", "interpret", "cuda"):
            raise ValueError("collision backend must be host, interpret or cuda")
        shapes = model.shapes
        excluded = set(model.excluded)
        self._offsets = np.array(
            [0, *np.cumsum([len(s.vertex_ids) for s in shapes], dtype=np.int32)], dtype=np.int32
        )
        self._tags = np.array(
            [(s.key.world, s.key.body, int(s.rigid), s.group, s.mask) for s in shapes],
            dtype=np.int32,
        ).reshape(-1, 5)
        self._pairs = np.array(
            [
                (i, j, int((shapes[i].key, shapes[j].key) in excluded))
                for i, j in combinations(range(len(shapes)), 2)
            ],
            dtype=np.int32,
        ).reshape(-1, 3)
        self._model, self._backend = model, backend
        self._lock = Lock()
        self._closed = False
        self._last = None
        self._engine = None
        if backend != "host":
            from ..backends.tyx_collision import TyxCollisionBatch

            self._engine = TyxCollisionBatch(
                backend, self._offsets, self._tags, self._pairs, cache_root
            )

    @property
    def last_result(self):
        return self._last

    def stats(self):
        with self._lock:
            return (
                {"backend": "host", "completed_launches": 0}
                if self._engine is None
                else self._engine.stats()
            )

    def _close_locked(self):
        if not self._closed:
            self._closed = True
            if self._engine is not None:
                self._engine.close()

    def close(self):
        with self._lock:
            self._close_locked()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("collision workspace is closed")
        return self

    def __exit__(self, *args):
        self.close()


def detect(model, motion, policy, workspace):
    """Query every registered pair. Never publish a partial or over-capacity result.

    Supported static narrow queries run on the bound FP64 source geometry.
    The selected host/tyx broad phase always supplies its own bounds and flags.
    """
    if not isinstance(workspace, CollisionWorkspace) or workspace._model is not model:
        raise ValueError("workspace is bound to a different collision model")
    if not isinstance(policy, CollisionPolicy):
        raise TypeError("query requires a CollisionPolicy")
    with workspace._lock:
        if workspace._closed:
            raise RuntimeError("collision workspace is closed")
        if not isinstance(motion, Mapping):
            raise TypeError("motion must be a mapping keyed by CollisionKey")
        paths = dict(motion)
        if set(paths) != {s.key for s in model.shapes}:
            raise ValueError("motion keys must exactly cover the collision model")
        envelopes = _envelopes(model, paths, policy.margin)
        try:
            if workspace._engine is None:
                bounds, flags = _host_query(
                    workspace._offsets, workspace._tags, workspace._pairs, envelopes
                )
            else:
                bounds, flags = workspace._engine.query(envelopes)
            if (
                bounds.shape != (len(model.shapes), 6)
                or flags.shape != (len(workspace._pairs),)
                or not np.isfinite(bounds).all()
                or np.any(bounds[:, :3] > bounds[:, 3:])
                or np.any(flags < 0)
                or np.any(flags >= len(_REASONS))
            ):
                raise SolverFailure("invalid collision broad-phase output")
        except BaseException:
            workspace._close_locked()
            raise
        count = int(np.count_nonzero(flags == 0))
        if count > policy.candidate_capacity:
            return CollisionDetection("capacity_exceeded", workspace._backend, count, (), ())
        rows = []
        for (i, j, _), code in zip(workspace._pairs, flags, strict=True):
            a, b = model.shapes[i], model.shapes[j]
            rows.append(_pair(a, b, paths[a.key], paths[b.key], int(code)))
        boxes = tuple(
            CollisionBounds(
                s.key, s.revision, tuple(map(float, row[:3])), tuple(map(float, row[3:]))
            )
            for s, row in zip(model.shapes, bounds, strict=True)
        )
        result = CollisionDetection("ok", workspace._backend, count, tuple(rows), boxes)
        workspace._last = result
        return result
