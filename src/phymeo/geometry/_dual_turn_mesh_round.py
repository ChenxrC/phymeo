"""Finite mesh-round contact along two independently turning rigid paths.

The validated mesh stays in its initial material chart. Each finite round-axis
point follows the complete composed rotation, with conservative distance and
speed bounds. Only this relative-path sampler and world material mapping are
new; the existing bounded search and exact finite feature evaluator are reused.
"""

from dataclasses import dataclass, replace
from fractions import Fraction as F
from math import ulp

import numpy as np

from ..constraints.collision_shapes import CollisionMotion
from ..core.capabilities import SolverFailure
from ._mesh_round import MeshRoundQuery, MeshRoundTrace
from ._prism_pair_exact import sqrt_bounds
from ._rotating_capsule_mesh import RotatingCapsuleMeshQuery, _upper_float
from ._rotating_mesh_sphere import RotatingMeshRoundQuery
from ._rotation_angle import axis_length_bounds, rotate_enclosures
from ._round_sweep import _translation, _vector, _xyz

_ZERO = F(0)
_MAX_SAMPLES = 512
_MAX_ANGLE = 32
_MAX_WORLD_ERROR = F(1, 10**6)


def _norm_upper(vector):
    return sqrt_bounds(vector @ vector, 80)[1]


@dataclass(frozen=True)
class DualTurnMeshRoundTrace(MeshRoundTrace):
    relative_samples: int
    relative_speed_upper_m: float
    observed_relative_error_upper_m: float
    observed_world_error_upper_m: float


class DualTurnMeshRoundQuery(RotatingMeshRoundQuery):
    """Two stored nonzero RigidTurns, independent world axes and affine pivots.

    The parent pipeline owns registered-path/world/filter/capacity validation and
    the prepared ClosedMesh. This object owns only one bounded pair query; no
    sampled geometry or publication is shared across queries or worlds.
    """

    def __init__(self, mesh, round_shape, mesh_path, round_path, mesh_first):
        mesh_turn, round_turn = mesh_path.turn, round_path.turn
        if (
            round_shape.kind not in ("sphere", "capsule")
            or mesh_turn is None
            or round_turn is None
            or not mesh_turn.angle
            or not round_turn.angle
        ):
            raise ValueError("dual-turn mesh-round requires two nonzero rigid turns")
        self.mesh = mesh
        self.shape = round_shape
        self.mesh_first = mesh_first
        self.radius = F(round_shape.radius)
        self.prefix = "dual_turn_mesh_round_"
        self.angle, self.round_angle = F(mesh_turn.angle), F(round_turn.angle)
        if max(abs(self.angle), abs(self.round_angle)) > _MAX_ANGLE:
            raise SolverFailure(self.prefix + "angle_capacity")
        self.axis, self.round_axis = _vector(mesh_turn.axis), _vector(round_turn.axis)
        self.axis_length = axis_length_bounds(self.axis)
        self.round_axis_length = axis_length_bounds(self.round_axis)
        self.pivot, self.round_pivot = (
            _vector(mesh_turn.pivot_start),
            _vector(round_turn.pivot_start),
        )
        self.travel, self.round_travel = _translation(mesh_path), _translation(round_path)
        self.spine = tuple(_vector(p) for p in round_path.start)
        self.arms = tuple(p - self.round_pivot for p in self.spine)
        self.separation = self.round_pivot - self.pivot
        self.drift = self.round_travel - self.travel
        axis2 = self.round_axis @ self.round_axis
        spin_reach = max(
            _norm_upper(arm - self.round_axis * ((arm @ self.round_axis) / axis2))
            for arm in self.arms
        )
        # Every finite-axis parameter is a convex combination of endpoints.
        # Its inner rotation preserves arm norm. The affine pivot separation's
        # maximum norm occurs at one of the two time endpoints. The outer
        # angular velocity can therefore use this complete, not sampled, reach.
        outer_reach = max(
            _norm_upper(self.separation), _norm_upper(self.separation + self.drift)
        ) + max(_norm_upper(arm) for arm in self.arms)
        self.speed = (
            _norm_upper(self.drift)
            + abs(self.round_angle) * spin_reach
            + abs(self.angle) * outer_reach
        )
        self.triangles = MeshRoundQuery(mesh, round_shape, round_path, mesh_first).triangles
        self.samples = {}
        self._world_round_samples = {}
        self.observed_error = _ZERO
        self.observed_world_error = _ZERO

    def _sample_axis(self, t):
        if t not in self.samples:
            if len(self.samples) >= _MAX_SAMPLES:
                raise SolverFailure(self.prefix + "sample_capacity")
            inner = rotate_enclosures(
                self.arms, self.round_axis, self.round_axis_length, self.round_angle * t
            )
            arms = tuple(self.separation + t * self.drift + p for p, _ in inner)
            outer = rotate_enclosures(arms, self.axis, self.axis_length, -self.angle * t)
            ends = tuple(self.pivot + p for p, _ in outer)
            # The true outer rotation is an isometry: inner displacement error
            # is transported without amplification, then outer proposal error is added.
            error = max(a + b for (_, a), (_, b) in zip(inner, outer, strict=True))
            self.observed_error = max(self.observed_error, error)
            self.samples[t] = ends, error, self._minimum(ends)
            self._world_round_samples[t] = tuple(
                (self.round_pivot + t * self.round_travel + p, e) for p, e in inner
            )
        return self.samples[t]

    def _world_bound(self, error, reason):
        if error > _MAX_WORLD_ERROR:
            raise SolverFailure(self.prefix + reason)
        self.observed_world_error = max(self.observed_world_error, error)

    def _pose(self, t):
        ends, error, best = self._sample_axis(t)
        path = CollisionMotion([_xyz(point) for point in ends])
        stored = tuple(_vector(point) for point in path.start)
        rounding = max(sum(abs(x) for x in a - b) for a, b in zip(stored, ends, strict=True))
        if rounding:
            best = self._minimum(stored)
            error += rounding
        self.observed_error = max(self.observed_error, error)
        query = MeshRoundQuery(self.mesh, self.shape, path, self.mesh_first)
        if error:
            far = max(query._axis_distance(p, stored[0])[0] for p in self.mesh.points.values())
            low, high = max(_ZERO, self.radius - error) ** 2, (self.radius + error) ** 2
            transverse = (
                len(stored) == 2
                and best == 0
                and RotatingCapsuleMeshQuery._verified_axis_crossing(self, stored, error)
            )
            # Do not infer winding/containment from boundary distance alone.
            # The reused crossing predicate uses only fixed facets and bounded
            # endpoint positions; no conserved-axis argument is valid here.
            if (best <= error**2 and not transverse) or low <= best <= high or low <= far <= high:
                raise SolverFailure(self.prefix + "pose_relation_unresolved")
        local = query.sample(_ZERO)
        if not t:
            return local
        shift = self.pivot + t * self.travel
        columns = rotate_enclosures(
            tuple(_vector(p) for p in np.eye(3)), self.axis, self.axis_length, self.angle * t
        )
        rotation = np.column_stack([p for p, _ in columns])
        column_error = np.array([e for _, e in columns], dtype=object)
        actual = self._world_round_samples[t]

        def world_point(point):
            arm = _vector(point) - self.pivot
            mapped = _xyz(shift + rotation @ arm)
            bound = error + abs(arm) @ column_error + 8 * F(max(ulp(x) for x in (*point, *mapped)))
            self._world_bound(bound, "world_resolution")
            return mapped

        def mesh_material(point, weights):
            mapped = world_point(point)
            total = sum(F(w) for _, w in weights)
            arm = sum(F(w) * (self.mesh.points[i] - self.pivot) for i, w in weights)
            difference = total * shift + rotation @ arm - _vector(mapped)
            self._world_bound(
                sum(abs(x) for x in difference) + abs(arm) @ column_error,
                "material_address_resolution",
            )
            return mapped

        def round_material(weights):
            anchor = sum(F(w) * actual[self.shape.vertex_ids.index(i)][0] for i, w in weights)
            uncertainty = sum(
                abs(F(w)) * actual[self.shape.vertex_ids.index(i)][1] for i, w in weights
            )
            return anchor, uncertainty

        def world(witness):
            mp, cp, mesh_weights, weights, offset = (
                (
                    witness.first_point,
                    witness.second_point,
                    witness.first_weights,
                    witness.second_weights,
                    witness.second_offset,
                )
                if self.mesh_first
                else (
                    witness.second_point,
                    witness.first_point,
                    witness.second_weights,
                    witness.first_weights,
                    witness.first_offset,
                )
            )
            mesh_point = mesh_material(mp, mesh_weights)
            radial_local = _vector(offset)
            radial = rotation @ radial_local
            anchor, anchor_error = round_material(weights)
            skin = _xyz(anchor + radial)
            # The round-side anchor comes from its own actual forward turn,
            # independently of the mesh-local proposal and inverse rotation.
            difference = _vector(skin) - _vector(world_point(cp))
            self._world_bound(
                sum(abs(x) for x in difference)
                + anchor_error
                + abs(radial_local) @ column_error
                + 8 * F(max(ulp(x) for x in skin)),
                "material_address_resolution",
            )
            normal = None if witness.normal is None else _xyz(rotation @ _vector(witness.normal))
            span = witness.capsule_axis_span
            if span is not None:
                points = []
                for item in span:
                    u = F(item.axis_parameter)
                    core = _xyz((1 - u) * actual[0][0] + u * actual[-1][0])
                    uncertainty = abs(1 - u) * actual[0][1] + abs(u) * actual[-1][1]
                    self._world_bound(
                        sum(abs(x) for x in _vector(core) - _vector(world_point(item.axis_point)))
                        + uncertainty
                        + 8 * F(max(ulp(x) for x in core)),
                        "material_address_resolution",
                    )
                    points.append(
                        replace(
                            item,
                            axis_point=core,
                            face_point=mesh_material(item.face_point, item.face_weights),
                        )
                    )
                span = tuple(points)
            return replace(
                witness,
                normal=normal,
                first_point=mesh_point if self.mesh_first else skin,
                second_point=skin if self.mesh_first else mesh_point,
                first_offset=witness.first_offset if self.mesh_first else _xyz(radial),
                second_offset=_xyz(radial) if self.mesh_first else witness.second_offset,
                capsule_axis_span=span,
            )

        return replace(
            local,
            witness=world(local.witness),
            supports=tuple(replace(s, witness=world(s.witness)) for s in local.supports),
            boundary=tuple(replace(s, witness=world(s.witness)) for s in local.boundary),
        )

    def result(self, moving):
        # Reuse the established whole-path search, actual-start solid semantics,
        # occupied sample gate, time/spatial budget and original five-field output.
        status, reason, endpoint, sweep, trace = super().result(moving)
        return (
            status,
            reason,
            endpoint,
            sweep,
            DualTurnMeshRoundTrace(
                trace.start,
                trace.end,
                trace.first_contact,
                len(self.samples),
                _upper_float(self.speed),
                _upper_float(self.observed_error),
                _upper_float(self.observed_world_error),
            ),
        )
