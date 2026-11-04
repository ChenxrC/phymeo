"""Material area weighted contact between finite link capsules and deforming triangle faces."""

from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np

from ..backends.tyx_normal_layer import TyxNormalLayer
from ..constraints.collision_shapes import CollisionKey, CollisionMotion, CollisionShape
from ..core.capabilities import SolverFailure
from ..core.validation import finite_real
from ..dynamics.robot_kinematics import forward_kinematics
from ..geometry._capsule_triangle_sweep import capsule_material_sites
from ..geometry._static_capsule_face import static_capsule_face_sample
from .collision_loads import _host_law


@dataclass(frozen=True)
class ClothLinkMaterial:
    """Rounded cloth-middle collision proxy and per-reference-area normal layer, in SI units."""

    width_m: float
    stiffness_pa_per_m: float
    damping_s_per_m: float = 0.0
    friction: float = 0.0
    slip_speed_m_s: float = 0.01

    def __post_init__(self):
        for name in (
            "width_m",
            "stiffness_pa_per_m",
            "damping_s_per_m",
            "friction",
            "slip_speed_m_s",
        ):
            value = finite_real(getattr(self, name), name)
            if value < 0 or (
                value == 0 and name in ("width_m", "stiffness_pa_per_m", "slip_speed_m_s")
            ):
                raise ValueError("invalid cloth-link material calibration")
            object.__setattr__(self, name, value)


class ClothLinkContacts:
    def __init__(self, cloth_domain, surfaces, areas, robots, material, operator):
        if not isinstance(material, ClothLinkMaterial):
            raise TypeError("expected a cloth-link material")
        self.cloth_domain = cloth_domain
        self.faces = np.array([s.node_indices for s in surfaces], dtype=np.intp)
        self.rho = np.array([s.half_thickness_m for s in surfaces])
        self.face_names = tuple(s.key for s in surfaces)
        self.triangles = tuple(
            CollisionShape(CollisionKey(0, 0, i), "triangle", tuple(ids))
            for i, ids in enumerate(self.faces)
        )
        self.areas = np.array(areas, dtype=float, copy=True)
        for values in (self.faces, self.rho, self.areas):
            values.flags.writeable = False
        self.models = robots.models
        self.links = robots.capsules
        self.capsules = tuple(
            CollisionShape(CollisionKey(0, 1 + i, 0), "capsule", (0, 1), link.radius_m)
            for i, link in enumerate(self.links)
        )
        self.law = material
        if operator is not None and not isinstance(operator, TyxNormalLayer):
            raise TypeError("cloth-link operator must be a TyxNormalLayer")
        self.operator = operator
        self._phase_dispatch = ContextVar('cloth_link_phase_dispatch', default=None)

    @staticmethod
    def endpoints(link, geo):
        origins, rotations = geo[link.domain][:2]
        local = np.array([link.start, link.end])
        return local @ rotations[link.link_index].T + origins[link.link_index]

    @staticmethod
    def bounds_distance(ends, tri, cutoff=None):
        """Lower distance bound from boxes and a separating projection for diagonal shafts."""
        scale = np.maximum(abs(tri).max(axis=(1, 2)), max(1.0, abs(ends).max()))
        slack = 128 * np.finfo(float).eps
        low = ends.min(0) / scale[:, None]
        high = ends.max(0) / scale[:, None]
        gaps = np.maximum(low - tri.max(1) / scale[:, None], tri.min(1) / scale[:, None] - high)
        box = np.hypot.reduce(np.maximum(0.0, gaps), axis=1)
        with np.errstate(over="ignore"):
            bound = np.minimum(np.finfo(float).max, np.maximum(0.0, box - slack) * scale)
        near = np.ones(len(tri), dtype=bool) if cutoff is None else bound <= cutoff
        if not np.any(near):
            return bound

        unit_scale = scale[near, None]
        origin = ends[0] / unit_scale
        shaft = ends[1] / unit_scale - origin
        vertices = tri[near] / unit_scale[:, None, :] - origin[:, None, :]
        center = vertices.mean(1)
        length = np.hypot.reduce(shaft, axis=1)
        along = np.divide(
            shaft, length[:, None], out=np.zeros_like(shaft), where=length[:, None] > 0
        )
        t = np.clip(np.einsum("ij,ij->i", center, along), 0.0, length)
        direction = center - t[:, None] * along
        size = np.hypot.reduce(direction, axis=1)
        # Any unit direction gives a lower bound, irrespective of how well it
        # approximates the true closest normal. A zero vector adds no bound.
        n = np.divide(
            direction, size[:, None], out=np.zeros_like(direction), where=size[:, None] > slack
        )
        projected = np.einsum("fij,fj->fi", vertices, n)
        tip = np.einsum("ij,ij->i", shaft, n)
        gap = np.maximum(
            projected.min(1) - np.maximum(0.0, tip), np.minimum(0.0, tip) - projected.max(1)
        )
        with np.errstate(over="ignore"):
            directional = np.minimum(
                np.finfo(float).max, np.maximum(0.0, gap - slack) * scale[near]
            )
        bound[near] = np.maximum(bound[near], directional)
        return bound

    def _witness(self, cap, face, ends, tri):
        return static_capsule_face_sample(
            cap, self.triangles[face], CollisionMotion(ends), CollisionMotion(tri)
        )[0]

    def evaluate(self, state, geo, *, phase=None):
        cloth = state.domains[self.cloth_domain]
        tri = cloth.position[self.faces]
        forces = np.zeros_like(cloth.position)
        efforts = {k: np.zeros_like(state.domains[k].v) for k in self.models}
        groups = []
        rows = []
        candidates = 0
        depth = 0.0
        for link, cap in zip(self.links, self.capsules, strict=True):
            ends = self.endpoints(link, geo)
            cutoff = cap.radius + self.rho + self.law.width_m
            indices = np.flatnonzero(self.bounds_distance(ends, tri, cutoff) <= cutoff)
            candidates += len(indices)
            _, _, com, jl, ja = geo[link.domain][:5]
            step_v = state.domains[link.domain].v
            i = link.link_index
            for face in indices:
                w = self._witness(cap, face, ends, tri[face])
                gap = w.gap_m - self.rho[face]
                if gap >= self.law.width_m:
                    continue
                if w.normal is None:
                    raise SolverFailure(
                        f"cloth-link capsule axis intersects face {self.face_names[face]} on {link.key}"
                    )
                depth = max(depth, -gap)
                n = np.asarray(w.normal)
                sites = capsule_material_sites(cap, w, capsule_first=True)
                group = []
                for site in sites:
                    point = np.asarray(site.second_point)
                    arm = point - com[i]
                    ids, weights = zip(*site.second_weights, strict=True)
                    weights = np.array(weights)
                    robot_v = jl[i] @ step_v + np.cross(ja[i] @ step_v, arm)
                    relative = weights @ cloth.velocity[list(ids)] - robot_v
                    rows.append(
                        (
                            *n,
                            *relative,
                            self.law.width_m,
                            gap,
                            self.law.stiffness_pa_per_m * self.areas[face],
                            1 / len(sites),
                            self.law.damping_s_per_m,
                            self.law.friction,
                            self.law.slip_speed_m_s,
                        )
                    )
                    group.append((point, arm, ids, weights, relative, site.first_weights))
                groups.append((link, face, ends, n, gap, group))
        values = np.array(rows, dtype=float).reshape(-1, 13)
        dispatch = self._phase_dispatch.get()
        if dispatch is None:
            answer = _host_law(values) if self.operator is None else self.operator.evaluate(values)
        else:
            addresses = tuple(
                (link.domain, link.key, self.face_names[face], tuple(map(int, self.faces[face])), len(sites))
                for link, face, _, _, _, sites in groups
            )
            answer = dispatch(phase, values, addresses)
        if (
            answer.shape != (len(rows), 5)
            or not np.isfinite(answer).all()
            or np.any(answer[:, 1] < 0)
        ):
            raise SolverFailure("invalid cloth-link layer response")
        actual = iter(answer.astype(float, copy=False))
        records = []
        for link, face, ends, n, gap, group in groups:
            _, _, _, jl, ja = geo[link.domain][:5]
            sites = []
            for point, arm, ids, weights, relative, axis_weights in group:
                rate, fn, *components = next(actual)
                force = np.array(components)
                np.add.at(forces, list(ids), weights[:, None] * force)
                efforts[link.domain] -= jl[link.link_index].T @ force + ja[
                    link.link_index
                ].T @ np.cross(arm, force)
                sites.append(
                    dict(
                        point_m=point.copy(),
                        axis_parameter=dict(axis_weights)[1],
                        face_weights=tuple(zip(ids, map(float, weights), strict=True)),
                        normal_speed_m_s=float(rate),
                        relative_velocity_m_s=relative.copy(),
                        normal_force_n=float(fn),
                        force_on_cloth_n=force,
                        force_on_link_n=-force,
                        power_w=float(force @ relative),
                    )
                )
            records.append(
                dict(
                    link=link.key,
                    robot=link.domain,
                    face=self.face_names[face],
                    face_nodes=tuple(map(int, self.faces[face])),
                    face_positions_m=tri[face].copy(),
                    half_thickness_m=float(self.rho[face]),
                    axis_ends_m=ends.copy(),
                    normal=n,
                    gap_m=float(gap),
                    normal_force_n=sum(s["normal_force_n"] for s in sites),
                    power_w=sum(s["power_w"] for s in sites),
                    sites=sites,
                )
            )
        return forces, dict(
            cloth_links=records,
            cloth_link_efforts=efforts,
            cloth_link_penetration_m=float(depth),
            cloth_link_candidates=candidates,
            cloth_link_pairs=len(tri) * len(self.links),
        )

    def _axis_motion(self, link, first, last):
        # Triangle inequality for each ancestor's rotation or translation along the actual joint-coordinate segment.
        model = self.models[link.domain]
        q0, q1 = first.domains[link.domain].q, last.domains[link.domain].q
        radius = max(np.linalg.norm(link.start), np.linalg.norm(link.end))
        travel = 0.0
        i = link.link_index
        while i >= 0:
            change = abs(q1[i] - q0[i])
            if model.joint_types[i] == "revolute":
                travel += change * radius
            else:
                travel += change
                radius += max(abs(q0[i]), abs(q1[i]))
            radius += np.linalg.norm(model.joint_position[i])
            i = int(model.parents[i])
        return travel

    def _clear_axis_interval(self, index, face, first, last, e0, e1, travel, uncertainty, required):
        link, cap = self.links[index], self.capsules[index]
        q0, q1 = (state.domains[link.domain].q for state in (first, last))
        x0, x1 = (
            state.domains[self.cloth_domain].position[self.faces[face]] for state in (first, last)
        )

        def distance(ends, triangle):
            witness = self._witness(cap, face, ends, triangle)
            if witness.normal is None and required == 0:
                raise SolverFailure(
                    f"cloth-link axis intersects face {self.face_names[face]} on {link.key} during drift"
                )
            return witness.gap_m + cap.radius - required

        def covered(lo, hi, d0, d1, depth):
            # If an intersection existed, travel to it and away from it would
            # have to cover both endpoint distances within this time interval.
            if min(d0, d1) <= uncertainty:
                return False
            if d0 + d1 > travel * (hi - lo) + 2 * uncertainty:
                return True
            if depth == 8:
                return False
            t = (lo + hi) / 2
            pose = forward_kinematics(self.models[link.domain], q0 + t * (q1 - q0))
            ends = np.array([link.start, link.end]) @ pose.rotation[link.link_index].T
            ends += pose.position[link.link_index]
            dm = distance(ends, x0 + t * (x1 - x0))
            return covered(lo, t, d0, dm, depth + 1) and covered(t, hi, dm, d1, depth + 1)

        return covered(0.0, 1.0, distance(e0, x0), distance(e1, x1), 0)

    def check_path(self, first, last, geo0, geo1):
        """Require strictly positive axis-to-middle separation along this affine x/q segment."""
        blocked = self._blocked_pair(first, last, geo0, geo1)
        if blocked is not None:
            link, face = blocked
            raise SolverFailure(f"cloth-link axis path unresolved for {link} and face {face}")

    def approach_blocker(self, domain, first, last):
        """First structural pair without certified layer clearance on the proposed x/q segment."""
        if domain not in self.models:
            raise ValueError("unknown approaching robot")

        def geometry(state):
            pose = forward_kinematics(self.models[domain], state.domains[domain].q)
            return {domain: (pose.position, pose.rotation)}

        return self._blocked_pair(first, last, geometry(first), geometry(last), domain)

    def _blocked_pair(self, first, last, geo0, geo1, domain=None):
        tri0 = first.domains[self.cloth_domain].position[self.faces]
        tri1 = last.domains[self.cloth_domain].position[self.faces]
        cloth_travel = np.linalg.norm(tri1 - tri0, axis=2).max(1)
        for index, (link, cap) in enumerate(zip(self.links, self.capsules, strict=True)):
            if domain is not None and link.domain != domain:
                continue
            e0, e1 = self.endpoints(link, geo0), self.endpoints(link, geo1)
            travel = cloth_travel + self._axis_motion(link, first, last)
            scale = np.maximum(
                np.maximum(abs(tri0).max(axis=(1, 2)), abs(tri1).max(axis=(1, 2))),
                max(1.0, abs(e0).max(), abs(e1).max()),
            )
            uncertainty = 128 * np.finfo(float).eps * (scale + travel + cap.radius)
            required = (
                np.zeros_like(self.rho)
                if domain is None
                else cap.radius + self.rho + self.law.width_m
            )
            cutoff = travel + 2 * (uncertainty + required)
            possible = np.flatnonzero(
                self.bounds_distance(e0, tri0, cutoff) + self.bounds_distance(e1, tri1, cutoff)
                <= cutoff
            )
            for face in possible:
                if not self._clear_axis_interval(
                    index,
                    face,
                    first,
                    last,
                    e0,
                    e1,
                    travel[face],
                    uncertainty[face],
                    required[face],
                ):
                    return link.key, self.face_names[face]
        return None
