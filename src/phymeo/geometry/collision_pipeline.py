"""Mixed-family collision queries for bounded shapes and unbounded half-spaces."""

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
from ._capsule_triangle_sweep import capsule_face_query
from ._closed_mesh import ClosedMesh, MeshHullQuery
from ._convex_contact import HullPairQuery
from ._convex_hull import HullShape
from ._rotating_round_sweep import RotatingCapsuleQuery, rotating_round_query
from ._round_sweep import round_query, sweep_result
from ._sphere_triangle_sweep import sphere_face_query
from ._tetrahedral_solid import TetrahedralCellQuery, solid_body_pairs, volume_mesh_issue
from ._tetrahedral_surface import TetrahedralSurfaceQuery, exterior_faces, positive_cell
from ._triangle_pair_sweep import triangle_pair_query
from .collision_witness import (
    CollisionSweep,
    CollisionWitness,
    ConvexHullTrace,
    MeshContactTrace,
    TetrahedralBodyPair,
    TetrahedralFacetResult,
    TetrahedralSolidResult,
)
from .surface_features import closest_point_triangle

_REASONS = ("candidate", "different_world", "same_rigid_body", "mask", "excluded", "aabb")
_F32_MAX = Fraction(float(np.finfo(np.float32).max))
_F32_TINY = Fraction(float(np.finfo(np.float32).tiny))


@dataclass(frozen=True)
class CollisionBounds:
    """Outward box for a bounded shape; both endpoints are None for a half-space."""

    key: CollisionKey
    revision: int
    lower: tuple[float, ...] | None
    upper: tuple[float, ...] | None


@dataclass(frozen=True)
class CollisionPair:
    first: CollisionKey
    second: CollisionKey
    family: str
    candidate: bool
    status: str
    reason: str
    witness: CollisionWitness | None = None
    sweep: CollisionSweep | None = None
    facets: tuple[TetrahedralFacetResult, ...] = ()
    solid: TetrahedralSolidResult | None = None
    convex: ConvexHullTrace | None = None
    mesh: MeshContactTrace | None = None


@dataclass(frozen=True)
class CollisionDetection:
    status: str
    backend: str
    required_candidates: int
    pairs: tuple[CollisionPair, ...]
    bounds: tuple[CollisionBounds, ...]
    solid_pairs: tuple[TetrahedralBodyPair, ...] = ()


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
        if shape.kind == "plane":
            # Its row is storage padding only. Both broad phases bypass this AABB.
            packed[row] = 0
            row += 1
            continue
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
        elif tags[a, 5] or tags[b, 5]:
            continue
        elif np.any(bounds[a, :3] > bounds[b, 3:]) or np.any(bounds[b, :3] > bounds[a, 3:]):
            flags[row] = 5
    return bounds, flags


def _address(shape, weights):
    return tuple(
        sorted((node, float(w)) for node, w in zip(shape.vertex_ids, weights, strict=True))
    )


def _static_witness(a, b, x, y):
    if {a.kind, b.kind} != {"sphere", "triangle"}:
        return None
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
    delta = q - p
    distance = float(np.hypot.reduce(delta))
    gap = distance - (a.radius + b.radius)
    if distance == 0:
        return CollisionWitness(gap, None, None, None, wa, wb)
    normal = delta / distance
    return CollisionWitness(
        gap,
        tuple(map(float, normal)),
        tuple(map(float, p + a.radius * normal)),
        tuple(map(float, q - b.radius * normal)),
        wa,
        wb,
    )


@dataclass(frozen=True)
class _NarrowFailure:
    status: str
    reason: str
    features = ()


def _pair(a, b, ma, mb, code, analytic, seeds):
    family = "RR" if a.rigid and b.rigid else "RD" if a.rigid != b.rigid else "DD"
    fields = (a.key, b.key, family)
    if isinstance(analytic, _NarrowFailure):
        return CollisionPair(*fields, not code, analytic.status, analytic.reason)
    if code:
        return CollisionPair(
            *fields, False, "separated" if code == 5 else "filtered", _REASONS[code]
        )
    if analytic is not None:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if isinstance(analytic, MeshHullQuery):
                    status, reason, witness, sweep, mesh = analytic.result(
                        ma.moving or mb.moving, seeds
                    )
                    return CollisionPair(*fields, True, status, reason, witness, sweep, mesh=mesh)
                if isinstance(analytic, HullPairQuery):
                    status, reason, witness, sweep, convex = analytic.result(
                        ma.moving or mb.moving, seeds
                    )
                    return CollisionPair(
                        *fields, True, status, reason, witness, sweep, convex=convex
                    )
                if isinstance(analytic, TetrahedralCellQuery):
                    status, reason, witness, sweep, solid = analytic.result(
                        ma.moving or mb.moving, seeds
                    )
                    return CollisionPair(*fields, True, status, reason, witness, sweep, solid=solid)
                if isinstance(analytic, TetrahedralSurfaceQuery):
                    return CollisionPair(
                        *fields, True, *analytic.result(ma.moving or mb.moving, seeds)
                    )
                if ma.moving or mb.moving:
                    witness, sweep = (
                        analytic.sweep(seeds)
                        if isinstance(analytic, RotatingCapsuleQuery)
                        else sweep_result(analytic, seeds)
                    )
                    hit = sweep.first_t is not None
                    if not hit and witness.shared_only:
                        return CollisionPair(
                            *fields, True, "adjacent", "shared_material_only", witness, sweep
                        )
                    return CollisionPair(
                        *fields,
                        True,
                        "hit" if hit else "separated",
                        "first_contact" if hit else "swept_separated",
                        witness,
                        sweep,
                    )
                witness, signed = analytic.sample(Fraction(0))
        except SolverFailure as error:
            return CollisionPair(*fields, True, "unresolved", str(error))
        except (OverflowError, FloatingPointError):
            return CollisionPair(*fields, True, "unresolved", "witness_outside_fp64")
        if witness.shared_only:
            return CollisionPair(*fields, True, "adjacent", "shared_material_only", witness)
        if witness.shared_material_points and witness.intersection_points:
            return CollisionPair(
                *fields, True, "touching", "distinct_material_intersection", witness
            )
        status = "overlap" if signed < 0 else "touching" if signed == 0 else "separated"
        return CollisionPair(
            *fields,
            True,
            status,
            "nonunique_normal" if witness.normal is None else "narrow_phase",
            witness,
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


def _detect_pairs(workspace, paths, flags):
    rows = []
    shapes = workspace._model.shapes
    # Bound temporary exact feature storage independently of the model's pair capacity.
    valid = {}
    mesh_issues = {}
    hulls = {}
    meshes = {}

    def prepared_hull(shape, path):
        if shape.key not in hulls:
            try:
                hulls[shape.key] = HullShape(shape, path)
            except ValueError as error:
                hulls[shape.key] = _NarrowFailure("invalid_geometry", str(error))
            except SolverFailure as error:
                hulls[shape.key] = _NarrowFailure("unresolved", str(error))
        return hulls[shape.key]

    def prepared_mesh(shape, path):
        if shape.key not in meshes:
            try:
                meshes[shape.key] = ClosedMesh(shape, path)
            except ValueError as error:
                meshes[shape.key] = _NarrowFailure("invalid_geometry", str(error))
            except SolverFailure as error:
                meshes[shape.key] = _NarrowFailure("unresolved", str(error))
        return meshes[shape.key]

    def valid_body(shape):
        body = shape.key.world, shape.key.body
        if body not in valid:
            valid[body] = all(
                positive_cell(paths[k])
                for k in workspace.exterior_faces
                if (k.world, k.body) == body
            )
        return valid[body]

    def solid_issue(shape):
        body = shape.key.world, shape.key.body
        if body not in mesh_issues:
            if not valid_body(shape):
                mesh_issues[body] = ("invalid_geometry", "tetrahedron_nonpositive_on_path")
            else:
                cells = [
                    s
                    for s in shapes
                    if s.kind == "tetrahedron" and (s.key.world, s.key.body) == body
                ]
                try:
                    mesh_issues[body] = volume_mesh_issue(cells, paths)
                except SolverFailure as error:
                    mesh_issues[body] = ("unresolved", str(error))
        return mesh_issues[body]

    # A thick cloth/cell pair can contribute 4 outer faces x 8 skin triangles.
    batch_size = 4 if workspace.exterior_faces or any(s.kind == "hull" for s in shapes) else 128
    for offset in range(0, len(flags), batch_size):
        batch = []
        features = []
        for local, (i, j, _) in enumerate(workspace._pairs[offset : offset + batch_size]):
            a, b = shapes[i], shapes[j]
            ma, mb = paths[a.key], paths[b.key]
            code = int(flags[offset + local])
            analytic = None
            if {a.kind, b.kind} == {"mesh", "hull"} and code in (0, 5):
                mesh_first = a.kind == "mesh"
                ms, hs, mp, hp = (a, b, ma, mb) if mesh_first else (b, a, mb, ma)
                if ms.radius or hs.radius or any(p.turn and p.turn.angle for p in (ma, mb)):
                    if not code:
                        analytic = _NarrowFailure("unsupported", "mesh_rounding_or_rotation")
                else:
                    mesh, hull = prepared_mesh(ms, mp), prepared_hull(hs, hp)
                    if isinstance(mesh, _NarrowFailure):
                        analytic = mesh
                    elif isinstance(hull, _NarrowFailure):
                        analytic = hull
                    elif not code:
                        try:
                            analytic = MeshHullQuery(mesh, hull, mesh_first)
                        except SolverFailure as error:
                            analytic = _NarrowFailure("unresolved", str(error))
            elif a.kind == b.kind == "hull" and code in (0, 5):
                if (
                    a.radius
                    or b.radius
                    or any(p.turn is not None and p.turn.angle != 0 for p in (ma, mb))
                ):
                    if not code:
                        analytic = _NarrowFailure("unsupported", "convex_rounding_or_rotation")
                else:
                    first, second = prepared_hull(a, ma), prepared_hull(b, mb)
                    if isinstance(first, _NarrowFailure):
                        analytic = first
                    elif isinstance(second, _NarrowFailure):
                        analytic = second
                    elif not code:
                        try:
                            analytic = HullPairQuery(first, second)
                        except SolverFailure as error:
                            analytic = _NarrowFailure("unresolved", str(error))
            elif a.kind == b.kind == "tetrahedron" and a.key.body != b.key.body and code in (0, 5):
                issue = solid_issue(a) or solid_issue(b)
                if issue:
                    analytic = _NarrowFailure(*issue)
                elif a.radius or b.radius:
                    analytic = _NarrowFailure("unsupported", "tetrahedral_cell_offset")
                elif not code:
                    analytic = TetrahedralCellQuery(a, b, ma, mb, workspace.exterior_faces)
            elif (
                not code
                and "tetrahedron" in (a.kind, b.kind)
                and {a.kind, b.kind} & {"triangle", "sphere"}
            ):
                tet_first = a.kind == "tetrahedron"
                cell, other, mc, mo = (a, b, ma, mb) if tet_first else (b, a, mb, ma)
                analytic = TetrahedralSurfaceQuery(
                    cell,
                    other,
                    mc,
                    mo,
                    workspace.exterior_faces[cell.key],
                    tet_first,
                    valid_body(cell),
                )
            elif not code:
                analytic = (
                    round_query(a, b, ma, mb)
                    or rotating_round_query(a, b, ma, mb)
                    or sphere_face_query(a, b, ma, mb)
                    or capsule_face_query(a, b, ma, mb)
                    or triangle_pair_query(a, b, ma, mb)
                )
            begin = len(features)
            if analytic is not None and (ma.moving or mb.moving):
                features.extend(analytic.features)
            batch.append((a, b, ma, mb, code, analytic, begin, len(features)))
        if workspace._engine is None:
            seeds = [None] * len(features)
        else:
            try:
                seeds = workspace._engine.entry_times(features)
            except BaseException:
                workspace._close_locked()
                raise
        for a, b, ma, mb, code, analytic, start, end in batch:
            proposals = seeds[start:end]
            if isinstance(analytic, HullPairQuery):
                proposals = None
                if workspace._engine is not None:
                    try:
                        proposals = workspace._engine.hull_intervals(*analytic.device_inputs())
                    except BaseException:
                        workspace._close_locked()
                        raise
            if isinstance(analytic, MeshHullQuery):
                proposals = None
                if workspace._engine is not None:
                    try:
                        proposals = [
                            workspace._engine.hull_intervals(*part.device_inputs())
                            for part in analytic.parts
                        ]
                    except BaseException:
                        workspace._close_locked()
                        raise
            rows.append(_pair(a, b, ma, mb, code, analytic, proposals))
    return tuple(rows)


class CollisionWorkspace:
    """One immutable model, one explicit broad-phase backend and one publication."""

    def __init__(self, model, backend="host", *, cache_root=None):
        if not isinstance(model, CollisionModel):
            raise TypeError("collision workspace requires a CollisionModel")
        if backend not in ("host", "interpret", "cuda"):
            raise ValueError("collision backend must be host, interpret or cuda")
        shapes = model.shapes
        self._exterior_faces = exterior_faces(shapes)
        excluded = set(model.excluded)
        self._offsets = np.array(
            [0, *np.cumsum([len(s.vertex_ids) for s in shapes], dtype=np.int32)], dtype=np.int32
        )
        self._tags = np.array(
            [
                (s.key.world, s.key.body, int(s.rigid), s.group, s.mask, int(s.kind == "plane"))
                for s in shapes
            ],
            dtype=np.int32,
        ).reshape(-1, 6)
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
    def exterior_faces(self):
        """Outward vertex ID triples grouped by owning tetrahedron, after internal cancellation."""
        return self._exterior_faces

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

    The selected backend supplies the broad phase and, for supported swept
    features, entry proposals. Stored-coordinate feature tests bound the first event.
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
        rows = _detect_pairs(workspace, paths, flags)
        boxes = tuple(
            CollisionBounds(s.key, s.revision, None, None)
            if s.kind == "plane"
            else CollisionBounds(
                s.key, s.revision, tuple(map(float, row[:3])), tuple(map(float, row[3:]))
            )
            for s, row in zip(model.shapes, bounds, strict=True)
        )
        result = CollisionDetection(
            "ok",
            workspace._backend,
            count,
            rows,
            boxes,
            solid_body_pairs(model, paths, rows) if workspace.exterior_faces else (),
        )
        workspace._last = result
        return result
