"""Solid contact along a closed mesh arc and an independently translating sphere or capsule."""

from dataclasses import dataclass, replace
from fractions import Fraction as F
from math import inf, nextafter, ulp

import numpy as np

from ..constraints.collision_shapes import CollisionMotion
from ..core.capabilities import SolverFailure
from ._bounded_contact import first_bounded_contact
from ._capsule_triangle_sweep import segment_minimum
from ._mesh_mesh import _point_nearest
from ._mesh_round import MeshRoundEvent, MeshRoundQuery, MeshRoundTrace
from ._prism_pair_exact import sqrt_bounds
from ._rotating_mesh_solid import _det_sign
from ._rotation_angle import axis_length_bounds, rotate_enclosures
from ._round_sweep import _translation, _vector, _xyz
from .collision_witness import CollisionSweep

_ZERO = F(0)
_ONE = F(1)
_MAX_SAMPLES = 512
_MAX_ANGLE = 32
_MAX_WORLD_ERROR = F(1, 10**6)


@dataclass(frozen=True)
class RotatingMeshSphereTrace(MeshRoundTrace):
    center_samples: int
    center_speed_upper_m: float


@dataclass(frozen=True)
class RotatingMeshCapsuleTrace(MeshRoundTrace):
    axis_samples: int
    axis_speed_upper_m: float
    observed_axis_error_upper_m: float


class RotatingMeshRoundQuery(MeshRoundQuery):
    """Use the full inverse-rotated spine in the original, validated material mesh."""

    def __init__(self, mesh, shape, mesh_path, round_path, mesh_first):
        self.mesh = mesh
        self.shape = shape
        self.mesh_first = mesh_first
        self.radius = F(shape.radius)
        self.prefix = "mesh_" + shape.kind + "_arc_"
        turn = mesh_path.turn
        self.angle = F(turn.angle)
        if abs(self.angle) > _MAX_ANGLE:
            raise SolverFailure(self.prefix + "angle_capacity")
        self.axis = _vector(turn.axis)
        self.axis_length = axis_length_bounds(self.axis)
        self.pivot = _vector(turn.pivot_start)
        self.travel = _translation(mesh_path)
        self.spine = tuple(_vector(p) for p in round_path.start)
        self.round_travel = _translation(round_path)
        self.relative = tuple(p - self.pivot for p in self.spine)
        self.drift = self.round_travel - self.travel
        # Each axis point and each affine offset is a convex combination of endpoints.
        # The maximum perpendicular reach thus occurs among these <=4 timed endpoints.
        axis2 = self.axis @ self.axis
        reach = _ZERO
        for v in self.relative:
            for arm in (v, v + self.drift):
                perp = arm - self.axis * ((arm @ self.axis) / axis2)
                reach = max(reach, sqrt_bounds(perp @ perp, 80)[1])
        self.speed = sqrt_bounds(self.drift @ self.drift, 80)[1] + abs(self.angle) * reach
        self.triangles = MeshRoundQuery(mesh, shape, round_path, mesh_first).triangles
        self.samples = {}
        self.observed_error = _ZERO

    def _minimum(self, ends):
        if len(ends) == 1:
            return _point_nearest(self.mesh.facets, ends[0])[0]
        a, b = ends
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        bounds = []
        for facet in self.mesh.facets:
            delta = np.maximum(0, np.maximum(facet.lo - hi, lo - facet.hi))
            bounds.append((delta @ delta, facet.ids))
        best = None
        for lower, ids in sorted(bounds):
            if best is not None and lower > best:
                break
            squared, _, _ = segment_minimum(a, b - a, self.triangles[ids])
            best = squared if best is None else min(best, squared)
        return best

    def _sample_axis(self, t):
        if t not in self.samples:
            if len(self.samples) >= _MAX_SAMPLES:
                name = "center_capacity" if len(self.spine) == 1 else "axis_capacity"
                raise SolverFailure(self.prefix + name)
            arms = tuple(v + t * self.drift for v in self.relative)
            rotated = rotate_enclosures(arms, self.axis, self.axis_length, -self.angle * t)
            ends = tuple(self.pivot + v for v, _ in rotated)
            error = max(e for _, e in rotated)
            self.samples[t] = ends, error, self._minimum(ends)
        return self.samples[t]

    def _distance(self, t):
        _, error, squared = self._sample_axis(t)
        return squared, error

    def _crosses_level_edge(self, ends, error, t):
        # Both material axes lie exactly in one conserved axial level. Strict
        # opposite 2D orientations prove an actual crossing even along a shared
        # mesh edge whose neighboring triangles are both on the same side.
        height = self.axis @ (self.spine[0] + t * self.drift)
        if self.axis @ (self.spine[1] + t * self.drift) != height:
            return False
        a, b = ends
        axis_norm = sum(abs(x) for x in self.axis)
        width = sum(abs(x) for x in b - a)
        for i, j in {edge for f in self.mesh.facets for edge in f.edges}:
            p, q = self.mesh.points[i], self.mesh.points[j]
            if self.axis @ p != height or self.axis @ q != height:
                continue
            edge = q - p
            side = np.cross(self.axis, edge)
            tol = error * sum(abs(x) for x in side)
            h0, h1 = side @ (a - p), side @ (b - p)
            if min(h0, h1) >= -tol or max(h0, h1) <= tol:
                continue
            normal = np.cross(self.axis, b - a)
            signs = []
            for vertex in (p, q):
                arm = vertex - a
                estimate = normal @ arm
                bound = axis_norm * error * (2 * sum(abs(x) for x in arm) + width + 2 * error)
                signs.append(1 if estimate > bound else -1 if estimate < -bound else 0)
            if signs[0] * signs[1] == -1:
                return True
        return False

    def _verified_axis_intersection(self, ends, error, t):
        # A strictly transverse crossing survives endpoint errors when both plane
        # heights and all three oriented edge-side tests retain their signs.
        if len(ends) != 2:
            return False
        a, b = ends
        for f in self.mesh.facets:
            n = f.face.normal
            heights = a @ n - f.face.plane, b @ n - f.face.plane
            bound = error * sum(abs(x) for x in n)
            edges = ((0, 1), (1, 2), (2, 0))
            if min(heights) < -bound and max(heights) > bound:
                signs = [_det_sign(f.xyz[i], f.xyz[j], a, b, error)[1] for i, j in edges]
                if signs[0] and min(signs) == max(signs):
                    return True
            if any(np.cross(n, self.axis)):
                continue
            # Axial projection is exactly invariant to the stored turn angle. A
            # material parameter on this plane can therefore be certified even
            # for a shaft that lies on it for the whole motion.
            h0, h1 = (n @ (p + t * self.drift) - f.face.plane for p in self.spine)
            if h0 == h1:
                if h0:
                    continue
                distance, low, high = segment_minimum(a, b - a, self.triangles[f.ids])
                if distance:
                    continue
                s = (low[0] + high[0]) / 2
            else:
                s = h0 / (h0 - h1)
                if not 0 <= s <= 1:
                    continue
            point = a + s * (b - a)
            for i, j in edges:
                inward = np.cross(n, f.xyz[j] - f.xyz[i])
                if inward @ (point - f.xyz[i]) <= error * sum(abs(x) for x in inward):
                    break
            else:
                return True
        return self._crosses_level_edge(ends, error, t)

    def _pose(self, t):
        ends, error, best = self._sample_axis(t)
        path = CollisionMotion([_xyz(v) for v in ends])
        stored = tuple(_vector(v) for v in path.start)
        rounding = max(sum(abs(x) for x in p - q) for p, q in zip(stored, ends, strict=True))
        if rounding:
            best = self._minimum(stored)
            error += rounding
        self.observed_error = max(self.observed_error, error)
        query = MeshRoundQuery(self.mesh, self.shape, path, self.mesh_first)
        if error:
            far = max(query._axis_distance(p, stored[0])[0] for p in self.mesh.points.values())
            lo, hi = max(_ZERO, self.radius - error) ** 2, (self.radius + error) ** 2
            # Distance to the spine is 1-Lipschitz in its Hausdorff displacement.
            # An uncertain crossing of the axis itself cannot establish its winding.
            near_axis = best <= error**2 and not (
                best == 0 and self._verified_axis_intersection(stored, error, t)
            )
            if near_axis or lo <= best <= hi or lo <= far <= hi:
                raise SolverFailure(self.prefix + "pose_relation_unresolved")
        local = query.sample(_ZERO)
        if not t:
            return local
        shift = self.pivot + t * self.travel
        actual = tuple(p + t * self.round_travel for p in self.spine)
        columns = rotate_enclosures(
            tuple(_vector(e) for e in np.eye(3)), self.axis, self.axis_length, t * self.angle
        )
        rotation = np.column_stack([col for col, _ in columns])
        column_error = np.array([e for _, e in columns], dtype=object)

        def world_point(point):
            v = _vector(point) - self.pivot
            mapped = _xyz(shift + rotation @ v)
            if (
                error + abs(v) @ column_error + 8 * F(max(ulp(x) for x in (*point, *mapped)))
                > _MAX_WORLD_ERROR
            ):
                raise SolverFailure(self.prefix + "world_resolution")
            return mapped

        def mesh_material(point, weights):
            # The stored barycentric coefficients also amplify any common pivot travel
            # if their floating sum differs from one. Reconstruct in the world chart.
            total = sum(F(w) for _, w in weights)
            arm = sum(F(w) * (self.mesh.points[i] - self.pivot) for i, w in weights)
            stored_point = world_point(point)
            mismatch = total * shift + rotation @ arm - _vector(stored_point)
            if sum(abs(x) for x in mismatch) + abs(arm) @ column_error > _MAX_WORLD_ERROR:
                raise SolverFailure(self.prefix + "material_address_resolution")
            return stored_point

        def world(w):
            mp, cp, offset, weights = (
                (w.first_point, w.second_point, w.second_offset, w.second_weights)
                if self.mesh_first
                else (w.second_point, w.first_point, w.first_offset, w.first_weights)
            )
            mesh_point = mesh_material(mp, w.first_weights if self.mesh_first else w.second_weights)
            radial = rotation @ _vector(offset)
            anchor = sum(
                F(weight) * actual[self.shape.vertex_ids.index(i)] for i, weight in weights
            )
            skin = _xyz(anchor + radial)
            if sum(abs(x) for x in _vector(skin) - _vector(world_point(cp))) > _MAX_WORLD_ERROR:
                raise SolverFailure(self.prefix + "material_address_resolution")
            normal = _xyz(rotation @ _vector(w.normal)) if w.normal is not None else None
            span = w.capsule_axis_span
            if span is not None:
                items = []
                for item in span:
                    core = _xyz(actual[0] + F(item.axis_parameter) * (actual[-1] - actual[0]))
                    if (
                        sum(abs(x) for x in _vector(core) - _vector(world_point(item.axis_point)))
                        > _MAX_WORLD_ERROR
                    ):
                        raise SolverFailure(self.prefix + "material_address_resolution")
                    items.append(
                        replace(
                            item,
                            axis_point=core,
                            face_point=mesh_material(item.face_point, item.face_weights),
                        )
                    )
                span = tuple(items)
            if self.mesh_first:
                return replace(
                    w,
                    normal=normal,
                    first_point=mesh_point,
                    second_point=skin,
                    second_offset=_xyz(radial),
                    capsule_axis_span=span,
                )
            return replace(
                w,
                normal=normal,
                first_point=skin,
                second_point=mesh_point,
                first_offset=_xyz(radial),
                capsule_axis_span=span,
            )

        return replace(
            local,
            witness=world(local.witness),
            supports=tuple(replace(s, witness=world(s.witness)) for s in local.supports),
            boundary=tuple(replace(s, witness=world(s.witness)) for s in local.boundary),
        )

    def result(self, moving):
        start, end = self._pose(_ZERO), self._pose(_ONE)
        if start.relation != "separated":
            entry = _ZERO, _ZERO
        else:
            # From separated solids any new occupation first meets the actual closed boundary.
            entry = first_bounded_contact([(_ZERO, _ONE)], self._distance, self.speed, self.radius)
        event = None
        if entry is None:
            if end.relation != "separated":
                raise SolverFailure(self.prefix + "terminal_state_inconsistent")
            status, reason = "separated", self.prefix + "swept_clear"
        else:
            lo, hi = entry
            if self.speed * (hi - lo) > _MAX_WORLD_ERROR:
                raise SolverFailure(self.prefix + "first_witness_resolution")
            pose = start if not hi else end if hi == 1 else self._pose(hi)
            if pose.relation == "separated":
                raise SolverFailure(self.prefix + "occupied_sample_unresolved")
            event = MeshRoundEvent((float(lo), float(hi)), float(hi), pose)
            status, reason = "hit", self.prefix + "first_solid_contact"
        upper_speed = float(self.speed)
        if F(upper_speed) < self.speed:
            upper_speed = nextafter(upper_speed, inf)
        if len(self.spine) == 1:
            trace = RotatingMeshSphereTrace(start, end, event, len(self.samples), upper_speed)
        else:
            upper_error = float(self.observed_error)
            if F(upper_error) < self.observed_error:
                upper_error = nextafter(upper_error, inf)
            trace = RotatingMeshCapsuleTrace(
                start, end, event, len(self.samples), upper_speed, upper_error
            )
        sweep = CollisionSweep(
            start.witness.gap_m,
            end.witness.gap_m,
            None if event is None else event.interval,
            None if event is None else event.witness_time,
            None if event is None else event.pose.witness,
        )
        return status, reason, end.witness, sweep, trace
