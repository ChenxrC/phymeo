"""Solid occupancy of a rotating closed mesh and a finite sphere with an affine center."""

from dataclasses import dataclass, replace
from fractions import Fraction as F
from math import inf, nextafter, ulp

from ..constraints.collision_shapes import CollisionMotion
from ..core.capabilities import SolverFailure
from ._bounded_contact import first_bounded_contact
from ._mesh_mesh import _point_nearest
from ._mesh_round import MeshRoundEvent, MeshRoundQuery, MeshRoundTrace
from ._prism_pair_exact import sqrt_bounds
from ._rotation_angle import axis_length_bounds, rotate_enclosures
from ._round_sweep import _translation, _vector, _xyz
from .collision_witness import CollisionSweep

_ZERO = F(0)
_ONE = F(1)
_MAX_CENTER_SAMPLES = 512
_MAX_ANGLE = 32
_MAX_WORLD_ERROR = F(1, 10**6)


@dataclass(frozen=True)
class RotatingMeshSphereTrace(MeshRoundTrace):
    center_samples: int
    center_speed_upper_m: float


class RotatingMeshSphereQuery(MeshRoundQuery):
    """Run the point-distance search in the mesh's initial material coordinates."""

    def __init__(self, mesh, sphere, mesh_path, sphere_path, mesh_first):
        self.mesh = mesh
        self.sphere = sphere
        self.mesh_first = mesh_first
        self.radius = F(sphere.radius)
        turn = mesh_path.turn
        self.angle = F(turn.angle)
        if abs(self.angle) > _MAX_ANGLE:
            raise SolverFailure("mesh_sphere_arc_angle_capacity")
        self.axis = _vector(turn.axis)
        self.axis_length = axis_length_bounds(self.axis)
        self.pivot = _vector(turn.pivot_start)
        self.travel = _translation(mesh_path)
        self.center = _vector(sphere_path.start[0])
        self.sphere_travel = _translation(sphere_path)
        self.relative = self.center - self.pivot
        self.drift = self.sphere_travel - self.travel
        # The norm of the perpendicular offset is convex on the affine segment.
        # The inverse rotation contributes at most |angle| times its endpoint maximum.
        axis2 = self.axis @ self.axis
        reach = _ZERO
        for arm in (self.relative, self.relative + self.drift):
            perp = arm - self.axis * ((arm @ self.axis) / axis2)
            reach = max(reach, sqrt_bounds(perp @ perp, 80)[1])
        self.speed = sqrt_bounds(self.drift @ self.drift, 80)[1] + abs(self.angle) * reach
        self.samples = {}

    def _center_sample(self, t):
        if t not in self.samples:
            if len(self.samples) >= _MAX_CENTER_SAMPLES:
                raise SolverFailure("mesh_sphere_arc_center_capacity")
            arm = self.relative + t * self.drift
            ((rotated, error),) = rotate_enclosures(
                (arm,), self.axis, self.axis_length, -self.angle * t
            )
            center = self.pivot + rotated
            d2, _ = _point_nearest(self.mesh.facets, center)
            self.samples[t] = center, error, d2
        return self.samples[t]

    def _distance(self, t):
        _, error, squared = self._center_sample(t)
        return squared, error

    def _pose(self, t):
        center, error, best = self._center_sample(t)
        # Keep the original material facets, winding and stable vertex IDs. Only
        # three rich pose observations are needed; interval leaves use point distance.
        path = CollisionMotion([_xyz(center)])
        stored = _vector(path.start[0])
        rounding = sum(abs(x) for x in stored - center)
        if rounding:
            best, _ = _point_nearest(self.mesh.facets, stored)
            error += rounding
        if error:
            far = max((p - stored) @ (p - stored) for p in self.mesh.points.values())
            lo, hi = max(_ZERO, self.radius - error) ** 2, (self.radius + error) ** 2
            # Center winding changes only at the boundary. The nearest and
            # furthest radii delimit touching and containment of the entire mesh.
            if best <= error**2 or lo <= best <= hi or lo <= far <= hi:
                raise SolverFailure("mesh_sphere_arc_pose_relation_unresolved")
        local = MeshRoundQuery(self.mesh, self.sphere, path, self.mesh_first).sample(_ZERO)
        if not t:
            return local

        def world(w):
            point, offset = (
                (w.first_point, w.second_offset)
                if self.mesh_first
                else (w.second_point, w.first_offset)
            )
            vectors = [_vector(point) - self.pivot, _vector(offset)]
            if w.normal is not None:
                vectors.append(_vector(w.normal))
            values = rotate_enclosures(vectors, self.axis, self.axis_length, t * self.angle)
            shift = self.pivot + t * self.travel
            mesh_point = _xyz(shift + values[0][0])
            radial = _xyz(values[1][0])
            ball_point = _xyz(self.center + t * self.sphere_travel + values[1][0])
            precision = max(ulp(x) for x in (*point, *offset, *mesh_point, *ball_point))
            if error + max(values[0][1], values[1][1]) + 8 * F(precision) > _MAX_WORLD_ERROR:
                raise SolverFailure("mesh_sphere_arc_world_resolution")
            normal = _xyz(values[2][0]) if w.normal is not None else None
            if self.mesh_first:
                return replace(
                    w,
                    normal=normal,
                    first_point=mesh_point,
                    second_point=ball_point,
                    second_offset=radial,
                )
            return replace(
                w,
                normal=normal,
                first_point=ball_point,
                second_point=mesh_point,
                first_offset=radial,
            )

        supports = tuple(replace(s, witness=world(s.witness)) for s in local.supports)
        boundary = tuple(replace(s, witness=world(s.witness)) for s in local.boundary)
        return replace(local, witness=world(local.witness), supports=supports, boundary=boundary)

    def result(self, moving):
        start, end = self._pose(_ZERO), self._pose(_ONE)
        if start.relation != "separated":
            entry = _ZERO, _ZERO
        else:
            # From an outside and separated start, any solid occupation must first
            # reach the closed boundary at radius. Surface clearance alone cannot
            # certify a mesh containing the sphere at the start.
            entry = first_bounded_contact([(_ZERO, _ONE)], self._distance, self.speed, self.radius)
        event = None
        if entry is None:
            if end.relation != "separated":
                raise SolverFailure("mesh_sphere_arc_terminal_state_inconsistent")
            status, reason = "separated", "mesh_sphere_arc_swept_clear"
        else:
            lo, hi = entry
            if self.speed * (hi - lo) > _MAX_WORLD_ERROR:
                raise SolverFailure("mesh_sphere_arc_first_witness_resolution")
            pose = start if not hi else end if hi == 1 else self._pose(hi)
            if pose.relation == "separated":
                raise SolverFailure("mesh_sphere_arc_occupied_sample_unresolved")
            event = MeshRoundEvent((float(lo), float(hi)), float(hi), pose)
            status, reason = "hit", "mesh_sphere_arc_first_solid_contact"
        upper_speed = float(self.speed)
        if F(upper_speed) < self.speed:
            upper_speed = nextafter(upper_speed, inf)
        trace = RotatingMeshSphereTrace(start, end, event, len(self.samples), upper_speed)
        sweep = CollisionSweep(
            start.witness.gap_m,
            end.witness.gap_m,
            None if event is None else event.interval,
            None if event is None else event.witness_time,
            None if event is None else event.pose.witness,
        )
        return status, reason, end.witness, sweep, trace
