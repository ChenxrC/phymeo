"""Instantaneous reciprocal solid contacts between distinct real robot owners.

Finite sphere, capsule and convex-hull patches share one bilateral load map.
This sampler has no live state and never advances, projects or publishes a robot.
It uses full-coordinate COM Jacobians; the existing participant applies any
mimic projection and integrates each real tree once with its other loads.
"""

from dataclasses import dataclass
from itertools import combinations, product
from types import MappingProxyType

import numpy as np

from ..constraints.robot_contact import RobotContactMaterial
from ..solvers.contact_patch_loads import sample_patch_loads
from ._convex_owner_contacts import convex_owner_patch, patch_feature, validate_convex_shape
from ._robot_scene_contacts import _Records
from .robot_payload_scene import _frozen, _readonly


@dataclass(frozen=True, eq=False)
class RobotContactContribution:
    """One owner's full generalized load and world force/moment in N and N m.

    Moving-link moments are about each actual COM. The final field contains
    only direct contact on fixed-root colliders, about the root frame origin;
    it is not the total base reaction transmitted through constrained joints.
    """

    joint_effort_full: np.ndarray
    link_wrenches_world: np.ndarray
    fixed_root_contact_wrench: np.ndarray

    def __post_init__(self):
        joint, links, root = (
            np.asarray(value, dtype=float)
            for value in (
                self.joint_effort_full,
                self.link_wrenches_world,
                self.fixed_root_contact_wrench,
            )
        )
        if (
            joint.ndim != 1
            or links.ndim != 2
            or links.shape[1] != 6
            or root.shape != (6,)
            or not all(np.isfinite(value).all() for value in (joint, links, root))
        ):
            raise ValueError("invalid full-coordinate robot contact contribution")
        for name, value in (
            ("joint_effort_full", joint),
            ("link_wrenches_world", links),
            ("fixed_root_contact_wrench", root),
        ):
            object.__setattr__(self, name, _readonly(value))


@dataclass(frozen=True)
class RobotRobotLoads:
    """Owned per-robot loads and a single collection of bilateral material rows."""

    by_robot: object
    contacts: tuple
    details: tuple

    def __post_init__(self):
        object.__setattr__(self, "by_robot", MappingProxyType(dict(self.by_robot)))
        object.__setattr__(self, "contacts", tuple(self.contacts))
        object.__setattr__(self, "details", tuple(_frozen(row) for row in self.details))


@dataclass(frozen=True)
class _Collider:
    owner: tuple
    key: object
    owner_index: int
    link: str
    name: str


class RobotRobotContacts:
    """Compile convex collider pairs without prescribing their physical owner.

    Geometry comes from the same entry poses published by each participant;
    full-coordinate COM Jacobians and the original integration stay untouched.
    """

    def __init__(self, robots, pairs, model):
        self._robots = dict(sorted(robots.items()))
        self._shapes = {shape.key: shape for shape in model.shapes}
        if len(self._shapes) != len(model.shapes):
            raise ValueError("direct robot contacts require unique scene shape keys")
        compiled, occupied = {}, set()
        for owner, participant in self._robots.items():
            rows = participant.contact_view.collisions
            if len({(c.link_index, c.link_name, c.name) for c in rows}) != len(rows):
                raise ValueError(f"robot {owner}: duplicate link/shape identities")
            if set(participant.key_map) != {c.shape.key for c in rows}:
                raise ValueError(f"robot {owner}: collider key mapping does not match its asset")
            for c in rows:
                key = participant.key_map[c.shape.key]
                shape = self._shapes.get(key)
                if (
                    key.world != owner[0]
                    or key in occupied
                    or shape is None
                    or shape.kind != c.shape.kind
                    or shape.vertex_ids != c.shape.vertex_ids
                    or shape.radius != c.shape.radius
                    or not -1 <= c.link_index < participant.body_count
                ):
                    raise ValueError(
                        f"robot {owner}: collider mapping does not match the scene model"
                    )
                occupied.add(key)
                compiled[owner, c.shape.key] = _Collider(
                    owner, key, c.link_index, c.link_name, c.name
                )
        selected = []
        validated = set()
        for first, second in combinations(self._robots, 2):
            if first[0] != second[0]:
                raise ValueError("direct robot contacts require the same world")
            if (first, second) not in pairs:
                raise ValueError(f"missing explicit robot pair {first}, {second}")
            law = pairs[first, second]
            if law is None:
                continue
            if not isinstance(law, RobotContactMaterial):
                raise TypeError("direct robot contacts require RobotContactMaterial")
            for a, b in product(
                self._robots[first].contact_view.collisions,
                self._robots[second].contact_view.collisions,
            ):
                if any(c.shape.kind not in {"sphere", "capsule", "hull"} for c in (a, b)):
                    raise ValueError(
                        "unsupported direct contact: expected sphere, finite capsule or convex hull"
                    )
                for owner, c in ((first, a), (second, b)):
                    if (owner, c.shape.key) not in validated:
                        validate_convex_shape(c.shape, c.local_vertices)
                        validated.add((owner, c.shape.key))
                selected.append((compiled[first, a.shape.key], compiled[second, b.shape.key], law))
        self._pairs = tuple(selected)

    @staticmethod
    def _point_motion(collider, point, geometry, twists, root_reference):
        index = collider.owner_index
        if index < 0:
            return root_reference, np.zeros(3)
        center = geometry.com_position[index]
        twist = twists[index]
        return center, twist[:3] + np.cross(twist[3:], point - center)

    @staticmethod
    def _accumulate(collider, point, center, force, link_loads, root_loads):
        wrench = np.concatenate((force, np.cross(point - center, force)))
        if collider.owner_index < 0:
            root_loads[collider.owner] += wrench
        else:
            link_loads[collider.owner][collider.owner_index] += wrench

    def evaluate(self, terms, poses):
        if set(terms) != set(self._robots):
            raise ValueError("direct contact terms must cover exactly the registered robots")
        if not set(self._shapes).issubset(poses):
            raise ValueError("direct contact poses must cover the compiled collision model")
        geometry, twists, roots, link_loads, root_loads = {}, {}, {}, {}, {}
        for owner, participant in self._robots.items():
            physical, geo, linear, angular = terms[owner]
            n, dofs = participant.body_count, participant.velocity_count
            if (
                physical.v.shape != (dofs,)
                or linear.shape != (n, 3, dofs)
                or angular.shape != (n, 3, dofs)
                or geo.position.shape != (n, 3)
                or geo.rotation.shape != (n, 3, 3)
                or geo.com_position.shape != (n, 3)
            ):
                raise ValueError(f"robot {owner}: inconsistent full-coordinate contact terms")
            geometry[owner] = geo
            twists[owner] = np.column_stack((linear @ physical.v, angular @ physical.v))
            roots[owner] = np.asarray(participant.contact_view.root_xyz, dtype=float)
            link_loads[owner] = np.zeros((n, 6))
            root_loads[owner] = np.zeros(6)
        records = _Records(self._shapes, poses)
        for first, second, law in self._pairs:
            a, b = self._shapes[first.key], self._shapes[second.key]
            ma, mb = poses[first.key], poses[second.key]
            if any(m.end is not None or m.turn is not None for m in (ma, mb)):
                raise ValueError("instantaneous owner loads require the same entry poses")
            x, y = np.asarray(ma.start), np.asarray(mb.start)
            # Broad rejection only; every possible loaded pair uses finite geometry.
            if np.any(x.max(axis=0) + a.radius <= y.min(axis=0) - b.radius) or np.any(
                y.max(axis=0) + b.radius <= x.min(axis=0) - a.radius
            ):
                continue
            patch = convex_owner_patch(a, b, poses)
            if patch is None:
                continue
            ga, gb = geometry[first.owner], geometry[second.owner]
            first_motion = [
                self._point_motion(
                    first, np.asarray(s.load_point), ga, twists[first.owner], roots[first.owner]
                )
                for s in patch.sites
            ]
            second_motion = [
                self._point_motion(
                    second, np.asarray(s.load_point), gb, twists[second.owner], roots[second.owner]
                )
                for s in patch.sites
            ]
            loads = sample_patch_loads(
                patch,
                law,
                np.array([v for _, v in first_motion]),
                np.array([v for _, v in second_motion]),
            )
            for i, site in enumerate(patch.sites):
                fa, fb = loads["first_force"][i], loads["second_force"][i]
                if not np.any(fa):
                    continue
                w = site.witness
                point = np.asarray(site.load_point)
                ca, va = first_motion[i]
                cb, vb = second_motion[i]
                self._accumulate(first, point, ca, fa, link_loads, root_loads)
                self._accumulate(second, point, cb, fb, link_loads, root_loads)
                records.add(
                    first.key,
                    second.key,
                    dict(
                        normal_world=w.normal,
                        force_on_payload_N=fb,
                        link_surface_m=w.first_point,
                        payload_surface_m=w.second_point,
                        link_com_world_m=ca,
                        payload_com_world_m=cb,
                        point_world_m=point,
                        weight=site.weight,
                        normal_load_N=float(fb @ w.normal),
                        normal_rate_m_s=float((vb - va) @ w.normal),
                        gap_m=w.gap_m,
                        pair_power_W=float(fb @ (vb - va)),
                        first_weights=w.first_weights,
                        second_weights=w.second_weights,
                        feature=patch_feature(patch),
                    ),
                    law,
                    "robot-robot",
                )
        contributions = {}
        for owner, (_, _, linear, angular) in terms.items():
            wrenches = link_loads[owner]
            joint = np.einsum("nki,nk->i", linear, wrenches[:, :3]) + np.einsum(
                "nki,nk->i", angular, wrenches[:, 3:]
            )
            contributions[owner] = RobotContactContribution(joint, wrenches, root_loads[owner])
        return RobotRobotLoads(contributions, tuple(records.contacts), tuple(records.details))
