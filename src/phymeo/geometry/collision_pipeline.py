"""Mixed-family collision queries for bounded shapes and unbounded half-spaces."""

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from itertools import combinations
from threading import RLock
from time import perf_counter_ns

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
from ._collision_bvh import CollisionTrees
from ._collision_spatial import SpatialPairs
from ._convex_contact import HullPairQuery
from ._convex_hull import HullShape
from ._curved_solids import CurvedContact, curved_bounds, curved_solid_contact
from ._heightfield import HeightfieldTrace, heightfield_sphere_contact
from ._hull_surface import HullSurfaceQuery
from ._mesh_mesh import MeshMeshQuery, MeshMeshTrace
from ._plate_contact import PlateSurfaceQuery, RoundedPlate
from ._rotating_capsule_face import RotatingCapsuleFaceQuery, rotating_capsule_face_query
from ._rotating_round_sweep import RotatingCapsuleQuery, rotating_round_query
from ._round_sweep import round_query, sweep_result
from ._sheet_pair_bounds import SheetPairBounds
from ._sphere_triangle_sweep import sphere_face_query
from ._tetrahedral_solid import TetrahedralCellQuery, solid_body_pairs, volume_mesh_issue
from ._tetrahedral_surface import TetrahedralSurfaceQuery, exterior_faces, positive_cell
from ._triangle_pair_sweep import triangle_pair_query
from .collision_witness import (
    CollisionSweep,
    CollisionWitness,
    ConvexHullTrace,
    HullSurfaceTrace,
    MaterialContactPoint,
    MeshContactTrace,
    PlateContactTrace,
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
    plate: PlateContactTrace | None = None
    hull_surface: HullSurfaceTrace | None = None
    heightfield: HeightfieldTrace | None = None
    curved: CurvedContact | None = None
    mesh_mesh: MeshMeshTrace | None = None


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
        if shape.heightfield is not None and not np.array_equal(
            path.start, shape.heightfield.world_vertices()
        ):
            raise ValueError("heightfield motion start must match registered world vertices")
        turn = path.turn
        if turn is not None and not shape.rigid:
            raise ValueError("rigid turn requires a rigid shape")
        if shape.kind == "plane":
            # Its row is storage padding only. Both broad phases bypass this AABB.
            packed[row] = 0
            row += 1
            continue
        if shape.curved is not None:
            low, high = curved_bounds(shape.curved, path)
            halo = Fraction(margin) / 2
            for k in range(3):
                packed[row, k] = _outward(low[k] - halo, False)
                packed[row, k + 3] = _outward(high[k] + halo, True)
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


def _host_bounds(offsets, envelopes):
    bounds = np.empty((len(offsets) - 1, 6), np.float32)
    for i, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        vertices = envelopes[lo:hi]
        bounds[i, :3] = vertices[:, :3].min(axis=0)
        bounds[i, 3:] = vertices[:, 3:].max(axis=0)
    return bounds


def _host_query(offsets, tags, pairs, envelopes):
    bounds = _host_bounds(offsets, envelopes)
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
                if isinstance(analytic, HullSurfaceQuery):
                    status, reason, witness, sweep, surface = analytic.result(
                        ma.moving or mb.moving, seeds
                    )
                    return CollisionPair(
                        *fields, True, status, reason, witness, sweep, hull_surface=surface
                    )
                if isinstance(analytic, PlateSurfaceQuery):
                    status, reason, witness, sweep, plate = analytic.result(
                        ma.moving or mb.moving, seeds
                    )
                    return CollisionPair(*fields, True, status, reason, witness, sweep, plate=plate)
                if isinstance(analytic, MeshMeshQuery):
                    status, reason, witness, sweep, trace = analytic.result(ma.moving or mb.moving)
                    return CollisionPair(
                        *fields, True, status, reason, witness, sweep, mesh_mesh=trace
                    )
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
                        if isinstance(analytic, (RotatingCapsuleQuery, RotatingCapsuleFaceQuery))
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


def _mesh_candidates(workspace, paths, flags, envelopes, bounds, pending):
    shapes = workspace._model.shapes
    groups = {}
    for slot in np.flatnonzero(flags == 0):
        i, j, _ = workspace._pairs[slot]
        if shapes[j].kind == "mesh":
            i, j = j, i
        if shapes[i].kind != "mesh" or shapes[j].kind != "hull":
            continue
        if any(
            s.radius or (paths[s.key].turn and paths[s.key].turn.angle)
            for s in (shapes[i], shapes[j])
        ):
            continue
        groups.setdefault(i, []).append((int(slot), j))
    selected = {}
    for i, partners in groups.items():
        shape = shapes[i]
        rows = workspace._mesh_rows[i]
        corners = envelopes[rows]
        leaves = np.column_stack((corners[:, :, :3].min(axis=1), corners[:, :, 3:].max(axis=1)))
        key = ("mesh", shape.key, shape.revision)
        hits = workspace._trees.facets(
            pending, key, leaves, bounds[[j for _, j in partners]], workspace._engine
        )
        selected.update((slot, row) for (slot, _), row in zip(partners, hits, strict=True))
    return selected


def _proved_sheet_pairs(workspace, paths, flags, margin):
    bounds = workspace._sheet_bounds
    if bounds is None:
        return {}
    rows = {}
    for slot, common in bounds.classify(paths, flags, margin).items():
        i, j, _ = workspace._pairs[slot]
        a, b = workspace._model.shapes[i], workspace._model.shapes[j]
        if not common:
            rows[slot] = CollisionPair(a.key, b.key, "DD", False, "separated", "sheet_projection")
            continue
        ma, mb = paths[a.key], paths[b.key]
        end = ma.start if ma.end is None else ma.end
        points = []
        for node in common:
            position = tuple(map(float, end[a.vertex_ids.index(node)]))
            weights_a = tuple((k, float(k == node)) for k in sorted(a.vertex_ids))
            weights_b = tuple((k, float(k == node)) for k in sorted(b.vertex_ids))
            points.append(MaterialContactPoint(position, weights_a, weights_b))
        points.sort(key=lambda p: (p.first_weights, p.second_weights))
        witness = CollisionWitness(
            0.0,
            None,
            None,
            None,
            (),
            (),
            "host-affine-triangle-pair",
            shared_material_points=tuple(points),
        )
        sweep = CollisionSweep(0.0, 0.0, None, None, None) if ma.moving or mb.moving else None
        rows[slot] = CollisionPair(
            a.key, b.key, "DD", True, "adjacent", "shared_material_only", witness, sweep
        )
    return rows


def _detect_pairs(
    workspace, paths, flags, selected, proximity=False, resolved=None, *, pair_start=0
):
    rows = []
    shapes = workspace._model.shapes
    resolved = {} if resolved is None else resolved
    # Bound temporary exact feature storage independently of the model's pair capacity.
    valid = {}
    mesh_issues = {}
    hulls = {}
    meshes = {}
    plates = {}

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
    batch_size = (
        4 if workspace.exterior_faces or any(s.kind in ("hull", "plate") for s in shapes) else 128
    )
    pairs = workspace._pairs[pair_start : pair_start + len(flags)]
    for offset in range(0, len(flags), batch_size):
        batch = []
        features = []
        for local, (i, j, _) in enumerate(pairs[offset : offset + batch_size]):
            a, b = shapes[i], shapes[j]
            ma, mb = paths[a.key], paths[b.key]
            code = int(flags[offset + local])
            slot = pair_start + offset + local
            ready = resolved.get(slot)
            if code == 0 and (a.curved is not None or b.curved is not None):
                family = "RR" if a.rigid and b.rigid else "RD"
                status, reason, witness, trace, sweep = curved_solid_contact(a, b, ma, mb)
                ready = CollisionPair(
                    a.key, b.key, family, True, status, reason, witness, sweep, curved=trace
                )
            elif code == 0 and (
                (a.kind == "heightfield" and b.kind in ("sphere", "rounded_particle"))
                or (b.kind == "heightfield" and a.kind in ("sphere", "rounded_particle"))
            ):
                terrain, sphere = (i, j) if a.kind == "heightfield" else (j, i)
                selector = (
                    partial(workspace._heightfield_faces, terrain, sphere)
                    if workspace._engine is not None
                    else None
                )
                status, reason, witness, chart, sweep = heightfield_sphere_contact(
                    a, b, ma, mb, face_selector=selector
                )
                family = "RR" if shapes[sphere].rigid else "RD"
                ready = CollisionPair(
                    a.key, b.key, family, True, status, reason, witness, sweep, heightfield=chart
                )
            if ready is not None:
                batch.append((a, b, ma, mb, code, ready, 0, 0))
                continue
            analytic = None
            if (
                "plate" in (a.kind, b.kind)
                and {a.kind, b.kind} & {"particle", "triangle"}
                and code in (0, 5)
            ):
                plate_first = a.kind == "plate"
                ps, other, mp, mc = (a, b, ma, mb) if plate_first else (b, a, mb, ma)
                if other.radius or (mp.turn is not None and mp.turn.angle):
                    if not code:
                        analytic = _NarrowFailure("unsupported", "plate_rotation_or_cloth_offset")
                else:
                    if ps.key not in plates:
                        try:
                            plates[ps.key] = RoundedPlate(ps, mp)
                        except ValueError as error:
                            plates[ps.key] = _NarrowFailure("invalid_geometry", str(error))
                    plate = plates[ps.key]
                    if isinstance(plate, _NarrowFailure):
                        analytic = plate
                    elif not code:
                        analytic = PlateSurfaceQuery(plate, other, mc, plate_first)
            elif {a.kind, b.kind} == {"hull", "triangle"} and code in (0, 5):
                hull_first = a.kind == "hull"
                hs, cs, hp, cp = (a, b, ma, mb) if hull_first else (b, a, mb, ma)
                if hs.radius or cs.radius or (hp.turn is not None and hp.turn.angle):
                    if not code:
                        analytic = _NarrowFailure("unsupported", "hull_rotation_or_surface_offset")
                else:
                    hull = prepared_hull(hs, hp)
                    if isinstance(hull, _NarrowFailure):
                        analytic = hull
                    elif not code:
                        analytic = HullSurfaceQuery(hs, hull, cs, cp, hull_first)
            elif a.kind == b.kind == "mesh" and code in (0, 5):
                if a.radius or b.radius or any(path.turn and path.turn.angle for path in (ma, mb)):
                    if not code:
                        analytic = _NarrowFailure("unsupported", "mesh_rounding_or_rotation")
                else:
                    first, second = prepared_mesh(a, ma), prepared_mesh(b, mb)
                    if isinstance(first, _NarrowFailure):
                        analytic = first
                    elif isinstance(second, _NarrowFailure):
                        analytic = second
                    elif not code:
                        analytic = MeshMeshQuery(first, second)
            elif {a.kind, b.kind} == {"mesh", "hull"} and code in (0, 5):
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
                            analytic = MeshHullQuery(mesh, hull, mesh_first, selected.get(slot))
                            workspace._mesh_work["mesh_facet_pairs_total"] += len(mesh.facets)
                            workspace._mesh_work["mesh_facet_pairs_tested"] += len(analytic.parts)
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
                            analytic = HullPairQuery(first, second, proximity=proximity)
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
                    or rotating_capsule_face_query(a, b, ma, mb)
                    or capsule_face_query(a, b, ma, mb)
                    or triangle_pair_query(a, b, ma, mb, proximity=proximity)
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
            if isinstance(analytic, CollisionPair):
                rows.append(analytic)
                continue
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


@dataclass
class _QueryDraft:
    layouts: dict
    result: CollisionDetection | None
    publications: dict = field(default_factory=dict)


class CollisionWorkspace:
    """One immutable model, one explicit broad-phase backend and one publication."""

    def __init__(
        self,
        model,
        backend="host",
        *,
        cache_root=None,
        broad_phase="all_pairs",
        mesh_midphase="all_facets",
        sheet_bounds=False,
    ):
        self._initialize(model, backend, cache_root, broad_phase, mesh_midphase, sheet_bounds, None)

    @classmethod
    def _from_world_pairs(cls, model, backend, pairs, *, cache_root=None):
        """Use an already partitioned host table for a single all-pairs backend batch."""
        if not (
            isinstance(pairs, np.ndarray)
            and pairs.dtype == np.int32
            and pairs.ndim == 2
            and pairs.shape[1] == 3
            and pairs.flags.c_contiguous
        ):
            raise ValueError("world pairs require a contiguous int32 array with three columns")
        instance = cls.__new__(cls)
        instance._initialize(model, backend, cache_root, "all_pairs", "all_facets", False, pairs)
        return instance

    def _initialize(
        self, model, backend, cache_root, broad_phase, mesh_midphase, sheet_bounds, pairs
    ):
        if not isinstance(model, CollisionModel):
            raise TypeError("collision workspace requires a CollisionModel")
        if backend not in ("host", "interpret", "cuda"):
            raise ValueError("collision backend must be host, interpret or cuda")
        if broad_phase not in ("all_pairs", "sap", "bvh"):
            raise ValueError("broad phase must be all_pairs, sap or bvh")
        if mesh_midphase not in ("all_facets", "bvh"):
            raise ValueError("mesh midphase must be all_facets or bvh")
        shapes = model.shapes
        if not isinstance(sheet_bounds, bool):
            raise TypeError("sheet_bounds must be a boolean")
        self._sheet_bounds = SheetPairBounds(shapes) if sheet_bounds else None
        self._exterior_faces = exterior_faces(shapes)
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
        self._world_partitioned = pairs is not None
        if pairs is None:
            excluded = set(model.excluded)
            pairs = np.array(
                [
                    (i, j, int((shapes[i].key, shapes[j].key) in excluded))
                    for i, j in combinations(range(len(shapes)), 2)
                ],
                dtype=np.int32,
            ).reshape(-1, 3)
        else:
            i, j, excluded = pairs.T
            if (
                np.any(i < 0)
                or np.any(j >= len(shapes))
                or np.any(i >= j)
                or np.any((excluded < 0) | (excluded > 1))
                or np.any(self._tags[i, 0] != self._tags[j, 0])
            ):
                raise ValueError("world pairs contain an invalid index, world or exclusion")
            pairs.flags.writeable = False
        self._pairs = pairs
        self._broad_phase = broad_phase
        self._spatial = (
            SpatialPairs(self._tags, self._pairs, broad_phase)
            if broad_phase != "all_pairs"
            else None
        )
        self._trees = CollisionTrees()
        self._mesh_midphase = mesh_midphase
        self._mesh_work = dict(mesh_facet_pairs_total=0, mesh_facet_pairs_tested=0)
        self._mesh_rows = {}
        if mesh_midphase == "bvh":
            for index, shape in enumerate(shapes):
                if shape.kind == "mesh":
                    by_id = {k: self._offsets[index] + j for j, k in enumerate(shape.vertex_ids)}
                    self._mesh_rows[index] = np.asarray(
                        [[by_id[k] for k in face] for face in shape.triangles], dtype=np.int32
                    )
        self._times = dict(envelope_ns=0, broad_ns=0, midphase_ns=0, narrow_ns=0)
        self._model, self._backend = model, backend
        self._lock = RLock()
        self._draft = None
        self._closed = False
        self._last = None
        self._engine = None
        if backend != "host":
            from ..backends.tyx_collision import TyxCollisionBatch

            self._engine = TyxCollisionBatch(
                backend, self._offsets, self._tags, self._pairs, cache_root, spatial=self._spatial
            )

    @property
    def exterior_faces(self):
        """Outward vertex ID triples grouped by owning tetrahedron, after internal cancellation."""
        return self._exterior_faces

    @property
    def last_result(self):
        with self._lock:
            return self._last

    @contextmanager
    def transaction(self):
        """Publish a group of geometry queries and associated physical frames on success.

        Nested groups join the enclosing commit. Failed work still contributes to
        diagnostics and a backend fault still closes the workspace.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("collision workspace is closed")
            parent = self._draft
            draft = (
                _QueryDraft(self._trees.layouts, self._last)
                if parent is None
                else _QueryDraft(parent.layouts, parent.result)
            )
            self._draft = draft
            try:
                yield self
                if self._closed:
                    raise RuntimeError("collision workspace closed during transaction")
                if parent is None:
                    self._trees.layouts, self._last = draft.layouts, draft.result
                    for publish, frame in draft.publications.items():
                        publish(frame)
                else:
                    parent.layouts, parent.result = draft.layouts, draft.result
                    parent.publications.update(draft.publications)
            finally:
                self._draft = parent

    def _stage_publication(self, publish, frame):
        self._draft.publications[publish] = frame

    def stats(self):
        with self._lock:
            base = (
                {"backend": "host", "completed_launches": 0}
                if self._engine is None
                else self._engine.stats()
            )
            return {
                **base,
                **self._times,
                **self._trees.counts,
                **self._mesh_work,
                **(
                    {
                        "sheet_pairs_adjacent": self._sheet_bounds.adjacent,
                        "sheet_pairs_far": self._sheet_bounds.far,
                        "sheet_pairs_fallback": self._sheet_bounds.fallback,
                    }
                    if self._sheet_bounds is not None
                    else {}
                ),
                "mesh_midphase": self._mesh_midphase,
                "broad_phase": self._broad_phase,
                "sap_window_pairs": self._spatial.window_pairs if self._spatial else 0,
                "sap_box_tests": self._spatial.box_tests if self._spatial else 0,
            }

    def _classify_world_envelopes(self, envelopes, *, pending=None):
        """Classify precisely this workspace's pair table without publishing results."""
        try:
            if self._engine is None and self._spatial is not None:
                space = self._spatial
                bounds = _host_bounds(self._offsets, envelopes)
                if space.mode == "bvh":
                    flags = space.tree_flags(bounds, self._trees, pending)
                else:
                    flags = space.decode_sap(space.host_sap(bounds, space.order(bounds)))
            elif self._engine is None:
                bounds, flags = _host_query(self._offsets, self._tags, self._pairs, envelopes)
            elif self._broad_phase == "bvh":
                bounds, flags = self._engine.query(envelopes, trees=self._trees, pending=pending)
            else:
                bounds, flags = self._engine.query(envelopes)
            if (
                bounds.shape != (len(self._model.shapes), 6)
                or flags.shape != (len(self._pairs),)
                or not np.isfinite(bounds).all()
                or np.any(bounds[:, :3] > bounds[:, 3:])
                or np.any(flags < 0)
                or np.any(flags >= len(_REASONS))
            ):
                raise SolverFailure("invalid collision broad-phase output")
            return bounds, flags
        except BaseException:
            self._close_locked()
            raise

    def _heightfield_faces(self, terrain, sphere):
        heights = self._model.shapes[terrain].heightfield.heights_yx
        try:
            faces = self._engine.heightfield_face_candidates(
                terrain, sphere, len(heights), len(heights[0])
            )
            if not isinstance(faces, tuple):
                raise RuntimeError("invalid heightfield candidate selection")
            return faces
        except BaseException:
            self._close_locked()
            raise

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


def _detection_result(model, paths, backend, count, rows, bounds):
    """Build public diagnostics using the caller's model and corresponding bounds slice."""
    boxes = tuple(
        CollisionBounds(s.key, s.revision, None, None)
        if s.kind == "plane"
        else CollisionBounds(
            s.key, s.revision, tuple(map(float, row[:3])), tuple(map(float, row[3:]))
        )
        for s, row in zip(model.shapes, bounds, strict=True)
    )
    return CollisionDetection(
        "ok",
        backend,
        count,
        rows,
        boxes,
        solid_body_pairs(model, paths, rows)
        if any(s.kind == "tetrahedron" for s in model.shapes)
        else (),
    )


def detect(model, motion, policy, workspace):
    """Query every registered pair. Never publish a partial or over-capacity result.

    The selected backend supplies the broad phase and, for supported swept
    features, entry proposals. Stored-coordinate feature tests bound the first event.
    """
    if not isinstance(workspace, CollisionWorkspace) or workspace._model is not model:
        raise ValueError("workspace is bound to a different collision model")
    if workspace._world_partitioned:
        raise ValueError("world-partitioned workspaces require a per-world detect query")
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
        stamp = perf_counter_ns()
        envelopes = _envelopes(model, paths, policy.margin)
        broad_start = perf_counter_ns()
        workspace._times["envelope_ns"] += broad_start - stamp
        draft = workspace._draft
        pending = (workspace._trees.layouts if draft is None else draft.layouts).copy()
        bounds, flags = workspace._classify_world_envelopes(envelopes, pending=pending)
        narrow_start = perf_counter_ns()
        workspace._times["broad_ns"] += narrow_start - broad_start
        resolved = _proved_sheet_pairs(workspace, paths, flags, policy.margin)
        count = int(np.count_nonzero(flags == 0)) - sum(not p.candidate for p in resolved.values())
        if count > policy.candidate_capacity:
            return CollisionDetection("capacity_exceeded", workspace._backend, count, (), ())
        selected = {}
        if workspace._mesh_midphase == "bvh":
            try:
                selected = _mesh_candidates(workspace, paths, flags, envelopes, bounds, pending)
            except BaseException:
                workspace._close_locked()
                raise
            after_mid = perf_counter_ns()
            workspace._times["midphase_ns"] += after_mid - narrow_start
            narrow_start = after_mid
        rows = _detect_pairs(workspace, paths, flags, selected, policy.proximity_patches, resolved)
        result = _detection_result(model, paths, workspace._backend, count, rows, bounds)
        workspace._times["narrow_ns"] += perf_counter_ns() - narrow_start
        if draft is None:
            workspace._trees.layouts, workspace._last = pending, result
        else:
            draft.layouts, draft.result = pending, result
        return result
