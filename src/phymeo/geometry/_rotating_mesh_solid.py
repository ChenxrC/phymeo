"""Closed-solid occupancy over the same bounded coaxial mesh arcs as the surface solver."""

from dataclasses import dataclass, replace
from fractions import Fraction as F
from math import inf, nextafter, ulp

from ..core.capabilities import SolverFailure
from ._closed_mesh import ClosedMesh, _Facet
from ._mesh_mesh import (
    MeshMeshPose,
    MeshMeshQuery,
    MeshMeshTrace,
    _representative,
    _support,
)
from ._rotation_angle import axis_length_bounds, rotate_enclosures
from ._round_sweep import _translation, _unit, _vector, _xyz
from ._triangle_pair_sweep import _EDGES, _cross3
from .collision_witness import CollisionSweep
from .mesh_arc_surface import MeshArcBoundaryEvent, _relative, mesh_arc_first_surface

_ZERO = F(0)
_MAX_ERROR_M = F(1, 10**6)


@dataclass(frozen=True)
class MeshArcSolidTrace(MeshMeshTrace):
    """Solid starts and ends plus the first actual boundary event, which can occur after time zero."""

    first_boundary: MeshArcBoundaryEvent | None
    end_certificate: str
    end_error_m: float


class _EndMesh(ClosedMesh):
    """Endpoint view of an already validated manifold, with its original oriented material faces."""

    def __init__(self, source, shape, path):
        travel = _translation(path)
        if path.turn and path.turn.angle:
            pivot = _vector(path.turn.pivot_start)
            axis = _vector(path.turn.axis)
            rotated = rotate_enclosures(
                [p - pivot for p in source.points.values()],
                axis,
                axis_length_bounds(axis),
                F(path.turn.angle),
            )
            self.points = {
                i: p + pivot + travel for i, (p, _) in zip(source.points, rotated, strict=True)
            }
            spatial = max(e for _, e in rotated)
        else:
            self.points = {i: p + travel for i, p in source.points.items()}
            spatial = _ZERO
        # The published coordinates and barycentric weights are FP64. The
        # latter multiply up to three coordinates on each owned triangle.
        scale = max(abs(x) for p in self.points.values() for x in p)
        self.error = spatial + 12 * F(ulp(float(scale)))
        if self.error > _MAX_ERROR_M:
            raise SolverFailure("mesh_arc_solid_world_resolution")
        self.step = _vector((0, 0, 0))
        self.facets = tuple(_Facet(shape, self.points, f.ids, self.step) for f in source.facets)


def _det_sign(a, b, c, d, error):
    """Sign of a 3D orientation for every allowed placement of these four vertices."""
    u, v, w = b - a, c - a, d - a
    det = _cross3(u, v) @ w
    lu, lv, lw = (sum(abs(x) for x in vector) for vector in (u, v, w))
    e = 2 * error
    bound = (lu + e) * (lv + e) * (lw + e) - lu * lv * lw
    return det, 1 if det > bound else -1 if det < -bound else 0


def _crossings(source, target, error):
    # An edge whose endpoints are strictly on opposite sides of the plane
    # crosses the open triangle if its three oriented edge tests have one sign.
    face = target.xyz
    heights = [_det_sign(*face, p, error) for p in source.xyz]
    for i, j in _EDGES:
        (h, a), (k, b) = heights[i], heights[j]
        if a * b != -1:
            continue
        p, q = source.xyz[i], source.xyz[j]
        signs = [_det_sign(p, q, face[k], face[later], error)[1] for k, later in _EDGES]
        if min(signs) != max(signs) or not signs[0]:
            continue
        t = h / (h - k)
        target_weights = target.weights(p + t * (q - p))
        if target_weights is None:
            raise SolverFailure("mesh_arc_solid_crossing_address")
        source_weights = tuple(sorted(((source.ids[i], 1 - t), (source.ids[j], t))))
        yield source_weights, target_weights


def _transverse_pose(query, error, previous_tests):
    contacts = {}
    count = previous_tests
    for i, j in query.facet_pairs(_ZERO, _ZERO):
        a, b = query.a.facets[i], query.b.facets[j]
        count += 1
        for wa, wb in _crossings(a, b, error):
            contacts[wa, wb] = None
        for wb, wa in _crossings(b, a, error):
            contacts[wa, wb] = None
    if not contacts:
        raise SolverFailure("mesh_arc_solid_endpoint_relation_unresolved")
    supports = tuple(_support(query.a, query.b, wa, wb, _ZERO) for wa, wb in sorted(contacts))
    return MeshMeshPose(
        "boundary_overlap", 0.0, None, (), supports, _representative(supports), count
    )


def _common_pose(start, ends, turn):
    # All features and distances are invariant under this common rigid motion.
    # Rebind stored weights to the moving world vertices on each side.
    axis = _vector(turn.axis)
    length = axis_length_bounds(axis)

    def world(w):
        pts = []
        for mesh, weights in zip(ends, (w.first_weights, w.second_weights), strict=True):
            pts.append(_xyz(sum(F(weight) * mesh.points[i] for i, weight in weights)))
        n = w.normal
        if n is not None:
            ((v, _),) = rotate_enclosures((_vector(n),), axis, length, F(turn.angle))
            n = _xyz(_unit(v)[0])
        return replace(
            w, first_point=pts[0], second_point=pts[1], normal=n, source="host-rotating-mesh-solid"
        )

    def supports(rows):
        return tuple(replace(s, witness=world(s.witness)) for s in rows)

    return replace(
        start,
        witness=world(start.witness),
        deepest=supports(start.deepest),
        boundary=supports(start.boundary),
        boundary_facet_pairs_tested=0,
    )


def _end_pose(shapes, paths, meshes, start, relative_angle):
    ends = tuple(_EndMesh(m, s, p) for m, s, p in zip(meshes, shapes, paths, strict=True))
    bound = sum(m.error for m in ends)
    if bound > _MAX_ERROR_M:
        raise SolverFailure("mesh_arc_solid_world_resolution")
    error = nextafter(float(bound), inf)
    if not relative_angle:
        turn = next(p.turn for p in paths if p.turn and p.turn.angle)
        return _common_pose(start, ends, turn), "fixed_relative_pose", error
    query = MeshMeshQuery(*ends)
    pose = query.sample(_ZERO)
    if F(pose.boundary_gap_m) > bound + 16 * F(ulp(pose.boundary_gap_m)):
        # Linear interpolation of each material vertex between the proposed
        # and the true endpoint stays within this displacement bound. The two
        # boundaries therefore cannot cross, preserving each winding number.
        if pose.relation in ("first_contains_second", "second_contains_first"):
            nearest = query._nearest(_ZERO)
            n = nearest.normal
            normal = None if n is None else tuple(-v for v in n)
            w = replace(
                nearest, gap_m=-nearest.gap_m, normal=normal, source="host-rotating-mesh-solid"
            )
            pose = replace(pose, witness=w, deepest=(), vertex_intrusion_m=None)
        return pose, "positive_boundary_clearance", error
    # Coincident proxy points or numerical penetration alone are insufficient
    # to establish a solid intersection of the real endpoint meshes.
    return (
        _transverse_pose(query, max(m.error for m in ends), pose.boundary_facet_pairs_tested),
        "strict_transverse_intersection",
        error,
    )


class RotatingMeshSolidQuery(MeshMeshQuery):
    def __init__(self, first_mesh, second_mesh, first, second, ma, mb):
        super().__init__(first_mesh, second_mesh)
        self.shapes = first, second
        self.motions = ma, mb

    def result(self, moving):
        relative = _relative(self.shapes, self.motions)
        if relative is None:
            return "unsupported", "mesh_arc_requires_one_relative_axis", None, None, None
        start = self.sample(_ZERO)
        surface = mesh_arc_first_surface(*self.shapes, *self.motions)
        if surface.status not in ("hit", "surface_clear"):
            return surface.status, surface.reason, None, None, None
        end, certificate, error = _end_pose(
            self.shapes, self.motions, (self.a, self.b), start, relative[3]
        )
        boundary = surface.event
        boundary_at_start = boundary is not None and boundary.interval == (0.0, 0.0)
        if bool(start.boundary) != boundary_at_start:
            raise SolverFailure("mesh_arc_solid_initial_boundary_inconsistent")
        if boundary is None and start.relation != end.relation:
            raise SolverFailure("mesh_arc_solid_clear_path_inconsistent")
        occupied = start.relation != "separated"
        if occupied:
            first, time, w = (0.0, 0.0), 0.0, start.witness
        elif boundary is not None:
            first, time, w = (
                boundary.interval,
                boundary.witness_time,
                _representative(boundary.supports),
            )
        else:
            first = time = w = None
        sweep = CollisionSweep(start.witness.gap_m, end.witness.gap_m, first, time, w)
        trace = MeshArcSolidTrace(
            start,
            end,
            (0, 1) if occupied else None,
            start if occupied else None,
            boundary,
            certificate,
            error,
        )
        status, reason = (
            ("hit", "mesh_arc_first_solid_contact")
            if first is not None
            else ("separated", "mesh_arc_solid_separated")
        )
        return status, reason, end.witness, sweep, trace
