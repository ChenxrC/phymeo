"""Reusable body-contact and joint-stop loads on arbitrary fixed-base robot trees."""

from itertools import combinations
from types import MappingProxyType

import numpy as np

from ..constraints.robot_contact import (
    CollisionPlane,
    JointLimits,
    LinkCapsule,
    RobotContactMaterial,
)
from ..core.capabilities import SolverFailure
from ..core.robot import RobotInputs, RobotModel, validate_robot_step
from ..geometry.capsule import closest_segment_points
from .robot_dynamics import robot_com_kinematics


class RobotContactSystem:
    """Loads without state updates. No implicit parent/child exclusion.

    Same-link shapes are one rigid body; all other exclusions are named.
    Reactions use one common world point. Coincident centerlines fail.
    """

    def __init__(
        self, models, *, limits=None, capsules=(), planes=(), excluded_pairs=(), material=None
    ):
        models = dict(models)
        if not models or any(
            not isinstance(n, str) or not n or not isinstance(m, RobotModel)
            for n, m in models.items()
        ):
            raise ValueError("expected a named robot model set")
        self.models = MappingProxyType(models)
        self.limits = MappingProxyType(dict(limits or {}))
        for n, row in self.limits.items():
            if (
                n not in models
                or not isinstance(row, JointLimits)
                or row.lower.shape != (models[n].dof_count,)
            ):
                raise ValueError("unbound joint limits")
        self.capsules, self.planes = tuple(capsules), tuple(planes)
        if not all(isinstance(c, LinkCapsule) for c in self.capsules) or not all(
            isinstance(p, CollisionPlane) for p in self.planes
        ):
            raise TypeError("expected capsule and plane shapes")
        keys = [s.key for s in self.capsules + self.planes]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate collision shape key")
        for c in self.capsules:
            if c.domain not in models or c.link_index >= models[c.domain].dof_count:
                raise ValueError("unbound capsule")
        exclusions = set()
        for pair in excluded_pairs:
            if (
                isinstance(pair, str)
                or len(pair) != 2
                or pair[0] == pair[1]
                or any(k not in keys for k in pair)
            ):
                raise ValueError("invalid collision exclusion")
            key = tuple(sorted(pair))
            if key in exclusions:
                raise ValueError("duplicate collision exclusion")
            exclusions.add(key)
        self.excluded_pairs = tuple(sorted(exclusions))
        self.pairs = tuple(
            (a, b)
            for a, b in combinations(self.capsules, 2)
            if (a.domain, a.link_index) != (b.domain, b.link_index)
            and tuple(sorted((a.key, b.key))) not in exclusions
        )
        self.plane_pairs = tuple(
            (a, p)
            for a in self.capsules
            for p in self.planes
            if tuple(sorted((a.key, p.key))) not in exclusions
        )
        self.material = RobotContactMaterial() if material is None else material
        if not isinstance(self.material, RobotContactMaterial):
            raise TypeError("expected RobotContactMaterial")

    def signature(self):
        return dict(
            models=dict(self.models),
            limits=dict(self.limits),
            capsules=self.capsules,
            planes=self.planes,
            excluded_pairs=self.excluded_pairs,
            material=self.material,
        )

    def evaluate(self, states):
        """Public evaluation from a complete named RobotState mapping."""
        geo = {}
        for n, m in self.models.items():
            if n not in states:
                raise ValueError("missing robot state")
            validate_robot_step(m, states[n], RobotInputs.zeros(m.dof_count))
            g, jl, ja = robot_com_kinematics(m, states[n].q)
            geo[n] = (g.position, g.rotation, g.com_position, jl, ja)
        return self._evaluate(states, geo)

    def _evaluate(self, states, geo):
        # Private seam reuses the same geometry/Jacobians as inertial dynamics.
        efforts = {n: np.zeros(m.dof_count) for n, m in self.models.items()}
        stops = {n: row.evaluate(states[n]) for n, row in self.limits.items()}
        energy = 0.0
        for n, row in stops.items():
            efforts[n] += row["effort"]
            energy += row["potential_j"]
        bodies = {}
        for c in self.capsules:
            pos, rot = geo[c.domain][:2]
            i = c.link_index
            bodies[c.key] = (pos[i] + rot[i] @ c.start, pos[i] + rot[i] @ c.end)
        rows = []
        material = self.material

        def velocity(body, point):
            if body is None:
                return np.zeros(3)
            _, _, com, jl, ja = geo[body.domain][:5]
            i, v = body.link_index, states[body.domain].v
            return jl[i] @ v + np.cross(ja[i] @ v, point - com[i])

        def apply(body, point, force):
            if body is not None:
                _, _, com, jl, ja = geo[body.domain][:5]
                i = body.link_index
                efforts[body.domain] += jl[i].T @ force + ja[i].T @ np.cross(point - com[i], force)

        def contact(a, b, other_key, point, normal, gap):
            nonlocal energy
            relative = velocity(a, point) - velocity(b, point)
            rate = float(relative @ normal)
            depth = -gap
            normal_load = material.stiffness * depth + material.damping * max(-rate, 0.0)
            tangent = relative - rate * normal
            force = normal_load * normal - material.friction * normal_load * tangent / max(
                material.slip_speed_m_s, np.linalg.norm(tangent)
            )
            apply(a, point, force)
            apply(b, point, -force)
            potential = material.stiffness * depth * depth / 2
            energy += potential
            rows.append(
                dict(
                    a=a.key,
                    b=other_key,
                    domain_a=a.domain,
                    domain_b=None if b is None else b.domain,
                    point_m=point,
                    normal=normal,
                    gap_m=float(gap),
                    normal_speed_m_s=rate,
                    relative_velocity_m_s=relative,
                    normal_force_n=float(normal_load),
                    force_on_a_n=force,
                    force_on_b_n=-force,
                    potential_j=float(potential),
                )
            )

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                for a, b in self.pairs:
                    p0, p1 = bodies[a.key]
                    q0, q1 = bodies[b.key]
                    radius = a.radius_m + b.radius_m
                    if np.any(
                        np.maximum(np.minimum(p0, p1), np.minimum(q0, q1))
                        > np.minimum(np.maximum(p0, p1), np.maximum(q0, q1)) + radius
                    ):
                        continue
                    p, q = closest_segment_points(p0, p1, q0, q1)
                    length = float(np.linalg.norm(p - q))
                    gap = length - radius
                    if gap >= 0:
                        continue
                    if length <= np.finfo(float).eps * max(1.0, radius):
                        raise SolverFailure(
                            "coincident capsule centerlines have no resolved contact normal"
                        )
                    normal = (p - q) / length
                    point = (p - a.radius_m * normal + q + b.radius_m * normal) / 2
                    contact(a, b, b.key, point, normal, gap)
                for a, p in self.plane_pairs:
                    ends = bodies[a.key]
                    distances = [e @ p.normal - p.offset_m - a.radius_m for e in ends]
                    gap = min(distances)
                    if gap >= 0:
                        continue
                    center = (
                        (ends[0] + ends[1]) / 2
                        if distances[0] == distances[1]
                        else ends[int(distances[1] < distances[0])]
                    )
                    point = center - (a.radius_m + gap / 2) * p.normal
                    contact(a, None, p.key, point, p.normal, gap)
        except (FloatingPointError, OverflowError) as exc:
            raise SolverFailure(f"robot contact arithmetic failed: {exc}") from exc
        if not all(np.isfinite(e).all() for e in efforts.values()) or not np.isfinite(energy):
            raise SolverFailure("nonfinite robot contact load")
        return dict(
            efforts=efforts,
            contacts=rows,
            limits=stops,
            potential_j=energy,
            checked_pairs=len(self.pairs) + len(self.plane_pairs),
            penetration_m=max((-r["gap_m"] for r in rows), default=0.0),
            joint_violation=max(
                (float(np.max(r["violation"])) for r in stops.values()), default=0.0
            ),
        )
