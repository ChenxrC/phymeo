"""Material area weighted contact between finite link capsules and deforming triangle faces."""

from dataclasses import dataclass

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionMotion, CollisionShape
from ..core.capabilities import SolverFailure
from ..core.validation import finite_real
from ..geometry._capsule_triangle_sweep import capsule_face_query, capsule_material_sites
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
        self.models = robots.models
        self.links = robots.capsules
        self.capsules = tuple(
            CollisionShape(CollisionKey(0, 1 + i, 0), "capsule", (0, 1), link.radius_m)
            for i, link in enumerate(self.links)
        )
        self.law = material
        if operator is not None and not callable(getattr(operator, "evaluate", None)):
            raise TypeError("cloth-link operator must evaluate world-space load rows")
        self.operator = operator

    @staticmethod
    def endpoints(link, geo):
        origins, rotations = geo[link.domain][:2]
        local = np.array([link.start, link.end])
        return local @ rotations[link.link_index].T + origins[link.link_index]

    @staticmethod
    def bounds_distance(ends, tri):
        separation = np.maximum(ends.min(0) - tri.max(1), tri.min(1) - ends.max(0))
        return np.linalg.norm(np.maximum(0.0, separation), axis=1)

    def _witness(self, cap, face, ends, tri):
        query = capsule_face_query(
            cap, self.triangles[face], CollisionMotion(ends), CollisionMotion(tri)
        )
        return query.sample(0)[0]

    def evaluate(self, state, geo):
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
            indices = np.flatnonzero(
                self.bounds_distance(ends, tri) <= cap.radius + self.rho + self.law.width_m
            )
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
                groups.append((link, face, n, gap, group))
        values = np.array(rows, dtype=float).reshape(-1, 13)
        answer = _host_law(values) if self.operator is None else self.operator.evaluate(values)
        if (
            answer.shape != (len(rows), 5)
            or not np.isfinite(answer).all()
            or np.any(answer[:, 1] < 0)
        ):
            raise SolverFailure("invalid cloth-link layer response")
        actual = iter(answer.astype(float, copy=False))
        records = []
        for link, face, n, gap, group in groups:
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

    def check_path(self, first, last, geo0, geo1):
        """Require strictly positive axis-to-middle separation along this affine x/q segment."""
        tri0 = first.domains[self.cloth_domain].position[self.faces]
        tri1 = last.domains[self.cloth_domain].position[self.faces]
        cloth_travel = np.linalg.norm(tri1 - tri0, axis=2).max(1)
        for link, cap in zip(self.links, self.capsules, strict=True):
            e0, e1 = self.endpoints(link, geo0), self.endpoints(link, geo1)
            travel = cloth_travel + self._axis_motion(link, first, last)
            scale = np.maximum(
                np.maximum(abs(tri0).max(axis=(1, 2)), abs(tri1).max(axis=(1, 2))),
                max(1.0, abs(e0).max(), abs(e1).max()),
            )
            uncertainty = 128 * np.finfo(float).eps * (scale + travel + cap.radius)
            possible = np.flatnonzero(
                np.maximum(self.bounds_distance(e0, tri0), self.bounds_distance(e1, tri1))
                <= travel + uncertainty
            )
            for face in possible:
                d0 = self._witness(cap, face, e0, tri0[face]).gap_m + cap.radius
                d1 = self._witness(cap, face, e1, tri1[face]).gap_m + cap.radius
                if max(d0, d1) <= travel[face] + uncertainty[face]:
                    raise SolverFailure(
                        f"cloth-link axis path unresolved for {link.key} and face {self.face_names[face]}"
                    )
