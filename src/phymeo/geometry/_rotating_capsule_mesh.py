"""Bounded solid contact for a turning finite capsule and a translating closed mesh.

Only the capsule rotates. The complete finite spine is enclosed in the mesh's
initial material frame; certified distance exclusions cover the whole time path.
Published witnesses describe a stored occupied sample, not a general MTD or a
response manifold. TYX may classify the swept pair, while this narrow phase is HOST.
"""

from dataclasses import dataclass, replace
from fractions import Fraction as F
from math import inf, nextafter, ulp

import numpy as np

from ..constraints.collision_shapes import CollisionMotion
from ..core.capabilities import SolverFailure
from ._bounded_contact import first_bounded_contact
from ._capsule_triangle_sweep import segment_minimum
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


def _upper_float(value):
    stored = float(value)
    return nextafter(stored, inf) if F(stored) < value else stored


@dataclass(frozen=True)
class RotatingCapsuleMeshTrace(MeshRoundTrace):
    axis_samples: int
    axis_speed_upper_m: float
    observed_axis_error_upper_m: float


class RotatingCapsuleMeshQuery(MeshRoundQuery):
    """One finite capsule arc, an affine pivot, and a nonrotating closed mesh.

    The caller supplies the once-validated ClosedMesh from its actual registered
    motion. Shape/world/path validation and broad-phase capacity remain upstream.
    The inherited type lets the ordinary mesh-round publication consume the same
    five-field result without a separate response path.
    """

    def __init__(self, mesh, capsule, mesh_path, capsule_path, mesh_first, *, face_selector=None):
        turn = capsule_path.turn
        if (
            capsule.kind != "capsule"
            or turn is None
            or not turn.angle
            or (mesh_path.turn is not None and mesh_path.turn.angle)
        ):
            raise ValueError("requires one turning capsule and a nonrotating unrounded mesh")
        self.mesh = mesh
        self.face_selector = face_selector
        self.face_candidates = None
        if face_selector is not None:
            from ._rotating_mesh_candidates import MeshRoundArcCandidates

            self.face_candidates = MeshRoundArcCandidates(mesh, face_selector)
        self.shape = capsule
        self.mesh_first = mesh_first
        self.radius = F(capsule.radius)
        self.prefix = "capsule_mesh_arc_"
        self.angle = F(turn.angle)
        if abs(self.angle) > _MAX_ANGLE:
            raise SolverFailure(self.prefix + "angle_capacity")
        self.axis = _vector(turn.axis)
        self.axis_length = axis_length_bounds(self.axis)
        self.pivot = _vector(turn.pivot_start)
        self.travel = _translation(mesh_path)
        self.drift = _translation(capsule_path) - self.travel
        self.spine = tuple(_vector(p) for p in capsule_path.start)
        self.arms = tuple(p - self.pivot for p in self.spine)
        # For every finite-axis parameter u, the perpendicular arm is a convex
        # combination of its endpoints. Its norm is bounded by the larger end.
        axis2 = self.axis @ self.axis
        reach = _ZERO
        for arm in self.arms:
            perpendicular = arm - self.axis * ((arm @ self.axis) / axis2)
            reach = max(reach, sqrt_bounds(perpendicular @ perpendicular, 80)[1])
        self.speed = sqrt_bounds(self.drift @ self.drift, 80)[1] + abs(self.angle) * reach
        self.triangles = MeshRoundQuery(mesh, capsule, capsule_path, mesh_first).triangles
        self.samples = {}
        self.observed_error = _ZERO

    def _minimum(self, ends):
        indices = (
            range(len(self.mesh.facets))
            if self.face_candidates is None
            else self.face_candidates.sample_axis(ends, radius=self.radius)
        )
        facets = tuple(self.mesh.facets[i] for i in indices)
        a, b = ends
        low, high = np.minimum(a, b), np.maximum(a, b)
        bounds = []
        for facet in facets:
            delta = np.maximum(0, np.maximum(facet.lo - high, low - facet.hi))
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
                raise SolverFailure(self.prefix + "axis_capacity")
            rotated = rotate_enclosures(self.arms, self.axis, self.axis_length, self.angle * t)
            ends = tuple(self.pivot + t * self.drift + point for point, _ in rotated)
            error = max(bound for _, bound in rotated)
            self.samples[t] = ends, error, self._minimum(ends)
        return self.samples[t]

    def _distance(self, t):
        _, error, squared = self._sample_axis(t)
        return squared, error

    def _verified_axis_crossing(self, ends, error):
        # A transverse crossing of the open triangle survives every allowed
        # endpoint displacement. Coplanar and near-edge ambiguity remains unknown.
        a, b = ends
        for facet in self.mesh.facets:
            normal = facet.face.normal
            heights = a @ normal - facet.face.plane, b @ normal - facet.face.plane
            bound = error * sum(abs(x) for x in normal)
            if min(heights) >= -bound or max(heights) <= bound:
                continue
            signs = [
                _det_sign(facet.xyz[i], facet.xyz[j], a, b, error)[1]
                for i, j in ((0, 1), (1, 2), (2, 0))
            ]
            if signs[0] and min(signs) == max(signs):
                return True
        return False

    def _pose(self, t):
        ends, error, best = self._sample_axis(t)
        path = CollisionMotion([_xyz(p) for p in ends])
        stored = tuple(_vector(p) for p in path.start)
        rounding = max(sum(abs(x) for x in p - q) for p, q in zip(stored, ends, strict=True))
        if rounding:
            best = self._minimum(stored)
            error += rounding
        self.observed_error = max(self.observed_error, error)
        query = MeshRoundQuery(
            self.mesh, self.shape, path, self.mesh_first, face_selector=self.face_selector
        )
        if error:
            far = max(query._axis_distance(p, stored[0])[0] for p in self.mesh.points.values())
            low, high = max(_ZERO, self.radius - error) ** 2, (self.radius + error) ** 2
            # Boundary distance alone does not determine solid containment. Do
            # not accept a changed winding, radius crossing, or full containment
            # classification when the enclosed actual axis could change it.
            near_axis = best <= error**2 and not (
                best == 0 and self._verified_axis_crossing(stored, error)
            )
            if near_axis or low <= best <= high or low <= far <= high:
                raise SolverFailure(self.prefix + "pose_relation_unresolved")
        local = query.sample(_ZERO)
        if not t:
            return local
        shift = t * self.travel
        actual = tuple(p + shift for p in stored)

        def world_point(point):
            mapped = _xyz(_vector(point) + shift)
            if error + 8 * F(max(ulp(x) for x in (*point, *mapped))) > _MAX_WORLD_ERROR:
                raise SolverFailure(self.prefix + "world_resolution")
            return mapped

        def mesh_material(point, weights):
            mapped = world_point(point)
            reconstructed = sum(F(w) * (self.mesh.points[i] + shift) for i, w in weights)
            if sum(abs(x) for x in reconstructed - _vector(mapped)) > _MAX_WORLD_ERROR:
                raise SolverFailure(self.prefix + "material_address_resolution")
            return mapped

        def world(witness):
            mesh_point, cap_point, weights, offset, mesh_weights = (
                (
                    witness.first_point,
                    witness.second_point,
                    witness.second_weights,
                    witness.second_offset,
                    witness.first_weights,
                )
                if self.mesh_first
                else (
                    witness.second_point,
                    witness.first_point,
                    witness.first_weights,
                    witness.first_offset,
                    witness.second_weights,
                )
            )
            mp = mesh_material(mesh_point, mesh_weights)
            radial = _vector(offset)
            anchor = sum(F(w) * actual[self.shape.vertex_ids.index(i)] for i, w in weights)
            skin = _xyz(anchor + radial)
            if (
                sum(abs(x) for x in _vector(skin) - _vector(world_point(cap_point)))
                > _MAX_WORLD_ERROR
            ):
                raise SolverFailure(self.prefix + "material_address_resolution")
            span = witness.capsule_axis_span
            if span is not None:
                items = []
                for item in span:
                    core = _xyz(actual[0] + F(item.axis_parameter) * (actual[1] - actual[0]))
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
            return replace(
                witness,
                first_point=mp if self.mesh_first else skin,
                second_point=skin if self.mesh_first else mp,
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
            # Only the true initial solid relation authorizes this boundary search.
            # Distance exclusion covers every preceding time cell, not a sample list.
            entry = first_bounded_contact([(_ZERO, _ONE)], self._distance, self.speed, self.radius)
        event = None
        if entry is None:
            if end.relation != "separated":
                raise SolverFailure(self.prefix + "terminal_state_inconsistent")
            status, reason = "separated", self.prefix + "swept_clear"
        else:
            low, high = entry
            if self.speed * (high - low) > _MAX_WORLD_ERROR:
                raise SolverFailure(self.prefix + "first_witness_resolution")
            pose = start if not high else end if high == 1 else self._pose(high)
            if pose.relation == "separated":
                raise SolverFailure(self.prefix + "occupied_sample_unresolved")
            event = MeshRoundEvent((float(low), float(high)), float(high), pose)
            status, reason = "hit", self.prefix + "first_solid_contact"
        trace = RotatingCapsuleMeshTrace(
            start,
            end,
            event,
            len(self.samples),
            _upper_float(self.speed),
            _upper_float(self.observed_error),
        )
        sweep = CollisionSweep(
            start.witness.gap_m,
            end.witness.gap_m,
            None if event is None else event.interval,
            None if event is None else event.witness_time,
            None if event is None else event.pose.witness,
        )
        return status, reason, end.witness, sweep, trace
