"""Closed rigid triangle mesh against the actual radius of a sphere or finite capsule."""

from dataclasses import dataclass, replace
from fractions import Fraction as F
from math import ulp

import numpy as np

from ..core.capabilities import SolverFailure
from ._capsule_triangle_sweep import _CapsuleFace, segment_minimum
from ._curved_solids import _root
from ._mesh_mesh import _point_nearest
from ._round_sweep import _GRID, _translation, _unit, _vector, _weights, _xyz
from ._sphere_triangle_sweep import _SphereFace
from ._tetrahedral_surface import _TriangleMotion
from ._triangle_pair_sweep import _FacePath, _Pose
from .collision_witness import CapsuleAxisPoint, CollisionSweep, CollisionWitness

_ZERO = F(0)
_ONE = F(1)
_ZERO_OFFSET = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class MeshRoundSupport:
    face: tuple[int, int, int]
    witness: CollisionWitness


@dataclass(frozen=True)
class MeshRoundPose:
    relation: str
    boundary_gap_m: float
    axis_gap_m: float
    axis_location: str
    supports: tuple[MeshRoundSupport, ...]
    boundary: tuple[MeshRoundSupport, ...]
    witness: CollisionWitness


@dataclass(frozen=True)
class MeshRoundEvent:
    interval: tuple[float, float]
    witness_time: float
    pose: MeshRoundPose


@dataclass(frozen=True)
class MeshRoundTrace:
    start: MeshRoundPose
    end: MeshRoundPose
    first_contact: MeshRoundEvent | None


def _gap(d2, radius):
    # Rationalization keeps the sign at nearly equal positive radii.
    return float((d2 - radius * radius) / (_root(d2) + radius))


def _material(face, weights, shift):
    # The witness and its published barycentric address must identify the same
    # point. Small world coordinates do not bound cancellation on a long face.
    rounding = np.array([F(float(w)) - w for w in weights], dtype=object)
    if any(rounding):
        displacement = rounding @ (face.xyz + shift)
        if sum(abs(x) for x in displacement) > F(1, 10**7):
            raise SolverFailure("mesh_round_material_address_resolution")
    return tuple(sorted((i, float(w)) for i, w in zip(face.ids, weights, strict=True) if w))


class MeshRoundQuery:
    features = ()

    def __init__(self, mesh, round_shape, round_motion, mesh_first, *, face_selector=None):
        self.mesh = mesh
        self.face_selector = face_selector
        self.round = round_shape
        self.path = round_motion
        self.mesh_first = mesh_first
        self.radius = F(round_shape.radius)
        self.start = _vector(round_motion.start[0])
        self.axis = _vector(round_motion.start[-1]) - self.start
        self.axis2 = self.axis @ self.axis
        self.drift = _translation(round_motion) - mesh.step
        self.triangles = (
            {f.ids: _Pose(_FacePath(f.shape, f.stationary), _ZERO) for f in mesh.facets}
            if round_shape.kind == "capsule"
            else {}
        )

    def _axis_distance(self, p, start):
        s = max(_ZERO, min(_ONE, ((p - start) @ self.axis) / self.axis2)) if self.axis2 else _ZERO
        delta = p - (start + s * self.axis)
        return delta @ delta, s

    def _nearest(self, face, start):
        if self.round.kind == "capsule":
            return segment_minimum(start, self.axis, self.triangles[face.ids])
        d2, addresses = _point_nearest((face,), start)
        address = dict(min(addresses))
        point = (_ZERO, tuple(address.get(i, _ZERO) for i in face.ids))
        return d2, point, point

    def _record(self, face, weights, s, start, shift, *, inside=False, span=None):
        p = np.asarray(weights, dtype=object) @ face.xyz
        axis = start + s * self.axis
        toward_axis = axis - p
        normal, _ = _unit(toward_axis)
        if normal is None:
            raise SolverFailure("mesh_round_radial_direction_unresolved")
        gap = _gap(toward_axis @ toward_axis, self.radius)
        radial = -self.round.radius * normal
        if inside:
            normal = -normal
            gap = -gap
        mesh_point = _xyz(p + shift)
        round_point = _xyz(np.asarray(axis + shift, dtype=float) + radial)
        if any(ulp(x) > 1e-7 for x in (*mesh_point, *round_point)):
            raise SolverFailure("mesh_round_world_resolution")
        mesh_weights = _material(face, weights, shift)
        round_weights = _weights(self.round, s)
        offsets = _ZERO_OFFSET, _xyz(radial)
        points = mesh_point, round_point
        materials = mesh_weights, round_weights
        if not self.mesh_first:
            normal = -normal
            points = points[::-1]
            materials = materials[::-1]
            offsets = offsets[::-1]
        witness = CollisionWitness(
            gap,
            _xyz(normal),
            *points,
            *materials,
            "host-closed-mesh-round",
            first_offset=offsets[0],
            second_offset=offsets[1],
            capsule_axis_span=span,
        )
        return MeshRoundSupport(face.ids, witness)

    def _nearest_support(self, face, low, high, start, shift, inside):
        s = (low[0] + high[0]) / 2
        w = tuple((F(x) + y) / 2 for x, y in zip(low[1], high[1], strict=True))
        span = None
        if self.round.kind == "capsule":
            span = tuple(
                CapsuleAxisPoint(
                    float(u),
                    _xyz(start + u * self.axis + shift),
                    _xyz(np.asarray(v, dtype=object) @ face.xyz + shift),
                    _material(face, v, shift),
                )
                for u, v in (low, high)
            )
        return self._record(face, w, s, start, shift, inside=inside, span=span)

    def _boundary_point(self, face, start, minimum, near, far, shift):
        d2, q, node = max(far)
        high = tuple(F(i == node) for i in face.ids)
        if d2 == self.radius**2:
            return self._record(face, high, q, start, shift)
        if minimum == self.radius**2:
            return self._record(face, near[1], near[0], start, shift)
        # The sublevel set of distance to a segment is convex. This material
        # segment runs from an interior point on the triangle to an exterior
        # vertex. Its boundary is on both the finite triangle and the round skin.
        base = np.array(near[1], dtype=object)
        delta = np.array(high, dtype=object) - base
        p, dp = base @ face.xyz, delta @ face.xyz
        speed = sum(abs(x) for x in dp)
        lo, hi = _ZERO, _ONE
        for _ in range(128):
            t = (lo + hi) / 2
            actual, s = self._axis_distance(p + t * dp, start)
            if actual == self.radius**2 or ((hi - lo) * speed <= F(1, 10**11) and actual):
                return self._record(face, base + t * delta, s, start, shift)
            if actual < self.radius**2:
                lo = t
            else:
                hi = t
        raise SolverFailure("mesh_round_boundary_witness_resolution")

    def sample(self, t):
        start = self.start + t * self.drift
        shift = t * self.mesh.step
        radius2 = self.radius**2
        distances = {i: self._axis_distance(p, start) for i, p in self.mesh.points.items()}
        furthest = max(d2 for d2, _ in distances.values())
        low, high = np.minimum(start, start + self.axis), np.maximum(start, start + self.axis)
        queue = []
        indices = (
            range(len(self.mesh.facets))
            if self.face_selector is None
            else self.face_selector.sample(
                start,
                start + self.axis,
                radius=self.radius,
                vertex_distance2={i: distance[0] for i, distance in distances.items()},
            )
        )
        for i in indices:
            f = self.mesh.facets[i]
            delta = np.maximum(0, np.maximum(f.lo - high, low - f.hi))
            queue.append((delta @ delta, i))
        best = None
        closest = []
        crossings = []
        for lower, i in sorted(queue):
            if best is not None and lower > max(best, radius2):
                break
            f = self.mesh.facets[i]
            d2, a, b = self._nearest(f, start)
            if best is None or d2 < best:
                best, closest = d2, []
            if d2 == best:
                closest.append((f, a, b))
            far = [(*distances[k], k) for k in f.ids]
            if d2 <= radius2 <= max(x[0] for x in far):
                crossings.append(self._boundary_point(f, start, d2, a, far, shift))
        if best is None:
            raise SolverFailure("mesh_round_empty_boundary")
        if best == furthest == radius2:
            raise SolverFailure("mesh_round_constant_boundary_unresolved")
        axis_location = (
            "on_boundary"
            if best == 0
            else "inside"
            if self.mesh.inside(start + self.axis / 2)
            else "outside"
        )
        axis_gap = _gap(best, self.radius)
        if furthest <= radius2:
            relation = "round_contains_mesh"
            supports = []
            for face in self.mesh.facets:
                for node in face.ids:
                    d2, s = distances[node]
                    if d2 == furthest:
                        weights = tuple(F(i == node) for i in face.ids)
                        supports.append(self._record(face, weights, s, start, shift))
            clearance = -_gap(furthest, self.radius)
        else:
            if best >= radius2 and axis_location == "inside":
                relation = "mesh_contains_round"
            elif best == radius2:
                relation = "touching"
            elif best > radius2:
                relation = "separated"
            else:
                relation = "boundary_overlap"
            clearance = axis_gap if best > radius2 else 0.0
            supports = (
                [
                    self._nearest_support(f, a, b, start, shift, relation == "mesh_contains_round")
                    for f, a, b in closest
                ]
                if best
                else crossings
            )
        if not supports:
            raise SolverFailure("mesh_round_material_support_unresolved")
        supports = tuple(
            sorted(
                supports, key=lambda s: (s.face, s.witness.first_weights, s.witness.second_weights)
            )
        )
        boundary = tuple(sorted(crossings, key=lambda s: s.face))
        witness = supports[0].witness
        if len({s.witness.normal for s in supports}) > 1:
            witness = replace(witness, normal=None)
        return MeshRoundPose(
            relation, clearance, axis_gap, axis_location, supports, boundary, witness
        )

    def _first(self):
        # In the mesh frame, the two endpoints of the finite spine translate
        # by the same vector. A radius-expanded sweep guards the exact features.
        ends = np.array(
            (
                self.start,
                self.start + self.axis,
                self.start + self.drift,
                self.start + self.axis + self.drift,
            )
        )
        lo, hi = ends.min(0) - self.radius, ends.max(0) + self.radius
        best = None
        indices = (
            range(len(self.mesh.facets))
            if self.face_selector is None
            else self.face_selector.translation(
                self.start,
                self.start + self.axis,
                self.drift,
                radius=self.radius,
            )
        )
        for i in indices:
            facet = self.mesh.facets[i]
            if any(facet.hi < lo) or any(facet.lo > hi):
                continue
            face_path = _TriangleMotion(facet.xyz, facet.xyz + facet.step, bool(any(facet.step)))
            query = (
                _SphereFace(facet.shape, self.round, face_path, self.path)
                if self.round.kind == "sphere"
                else _CapsuleFace(facet.shape, self.round, face_path, self.path)
            )
            if not query.features:
                raise SolverFailure("mesh_round_facet_sweep_unresolved")
            for feature in query.features:
                bracket = feature.entry(None)
                if bracket is not None and (best is None or bracket < best):
                    best = bracket
        return best

    def result(self, moving):
        start = self.sample(_ZERO)
        if not moving:
            status = start.relation if start.relation in ("separated", "touching") else "overlap"
            return status, start.relation, start.witness, None, MeshRoundTrace(start, start, None)
        end = self.sample(_ONE)
        entry = (0, 0) if start.relation != "separated" else self._first()
        if entry is None:
            if end.relation != "separated":
                raise SolverFailure("mesh_round_terminal_state_inconsistent")
            sweep = CollisionSweep(start.witness.gap_m, end.witness.gap_m, None, None, None)
            return (
                "separated",
                "mesh_round_swept_clear",
                end.witness,
                sweep,
                MeshRoundTrace(start, end, None),
            )
        bracket = entry[0] / _GRID, entry[1] / _GRID
        time = (bracket[0] + bracket[1]) / 2
        pose = start if time == 0 else end if time == 1 else self.sample(F(time))
        if time and (abs(pose.axis_gap_m) > 1e-6 or pose.boundary_gap_m > 1e-6):
            raise SolverFailure("mesh_round_first_witness_resolution")
        event = MeshRoundEvent(bracket, time, pose)
        sweep = CollisionSweep(start.witness.gap_m, end.witness.gap_m, bracket, time, pose.witness)
        return (
            "hit",
            "mesh_round_first_solid_contact",
            end.witness,
            sweep,
            MeshRoundTrace(start, end, event),
        )
