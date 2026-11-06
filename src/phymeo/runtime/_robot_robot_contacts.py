"""Instantaneous reciprocal solid contacts between distinct real robot owners.

Only sphere-sphere, sphere-finite-capsule and sphere-OBB pairs are supported.
This sampler has no live state and never advances, projects or publishes a robot.
It uses full-coordinate COM Jacobians; the existing participant applies any
mimic projection and integrates each real tree once with its other loads.
"""

from dataclasses import dataclass
from itertools import combinations, product
from types import MappingProxyType

import numpy as np

from ..constraints.robot_contact import RobotContactMaterial
from ..solvers.sampled_box_contact import sample_loads
from ..solvers.sphere_link_contact import SphereLinkContacts
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
            or links.shape != (len(joint), 6)
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
    solid: object


def _kind(collision):
    return "box" if collision.shape.kind == "hull" else collision.shape.kind


def _label(owner, collision):
    return f"robot {owner} {collision.link_name}/{collision.name} ({_kind(collision)})"


class RobotRobotContacts:
    """Compile every collider combination of each explicitly enabled tree pair.

    robots maps physical (world, body) addresses to RobotParticipant objects.
    pairs is the existing scene owner-pair material mapping, including explicit
    None exclusions. Pads are intentionally not consulted: a declared direct
    pair uses actual solid colliders, without an additional pad sample force.
    """

    def __init__(self, robots, pairs, model):
        self._robots = dict(sorted(robots.items()))
        self._shapes = {shape.key: shape for shape in model.shapes}
        if len(self._shapes) != len(model.shapes):
            raise ValueError("direct robot contacts require unique scene shape keys")
        specifications = []
        needed = {owner: set() for owner in self._robots}
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
                self._robots[first].tree.collisions, self._robots[second].tree.collisions,
            ):
                kinds = _kind(a), _kind(b)
                if "sphere" not in kinds or any(
                    kind not in {"sphere", "capsule", "box"} for kind in kinds
                ):
                    raise ValueError(
                        f"unsupported direct contact: {_label(first, a)} with {_label(second, b)}; "
                        "one collider must be a sphere and the other a sphere, finite capsule or OBB"
                    )
                needed[first].add(a.shape.key)
                needed[second].add(b.shape.key)
                specifications.append((first, a, second, b, law))
        compiled = {}
        occupied = set()
        for owner, participant in self._robots.items():
            tree = participant.tree
            original = {
                (row.link_index, row.link_name, row.name): row.shape.key for row in tree.collisions
            }
            if len(original) != len(tree.collisions):
                raise ValueError(f"robot {owner}: duplicate link/shape identities")
            if set(participant.key_map) != {row.shape.key for row in tree.collisions}:
                raise ValueError(f"robot {owner}: collider key mapping does not match its asset")
            for collision in tree.collisions:
                key = participant.key_map[collision.shape.key]
                shape = self._shapes.get(key)
                if (
                    key.world != owner[0]
                    or key in occupied
                    or shape is None
                    or shape.kind != collision.shape.kind
                    or shape.vertex_ids != collision.shape.vertex_ids
                    or shape.radius != collision.shape.radius
                ):
                    raise ValueError(f"robot {owner}: collider mapping does not match the scene model")
                occupied.add(key)
            if not needed[owner]:
                continue
            excluded = set(participant.key_map) - needed[owner]
            try:
                solids = SphereLinkContacts.from_asset(tree, exclude_keys=excluded).solids
            except ValueError as exc:
                raise ValueError(f"robot {owner}: invalid direct-contact collider: {exc}") from exc
            for solid in solids:
                source = original[solid.owner_index, solid.link, solid.shape]
                if solid.owner_index >= tree.model.dof_count:
                    raise ValueError(f"robot {owner} {solid.link}/{solid.shape}: invalid link owner")
                compiled[owner, source] = _Collider(owner, participant.key_map[source], solid)
        selected = []
        for first, a, second, b, law in specifications:
            ca, cb = compiled[first, a.shape.key], compiled[second, b.shape.key]
            # The existing finite-solid witness takes one sphere as its second
            # argument. Canonical record order is restored by _Records.add.
            if cb.solid.kind != "sphere":
                ca, cb = cb, ca
            selected.append((ca, cb, law))
        self._pairs = tuple(selected)

    @staticmethod
    def _world_point(collider, point, geo):
        index = collider.solid.owner_index
        return np.asarray(point) if index < 0 else geo.position[index] + geo.rotation[index] @ point

    @staticmethod
    def _point_motion(collider, point, geometry, twists, root_reference):
        index = collider.solid.owner_index
        if index < 0:
            return root_reference, np.zeros(3)
        center = geometry.com_position[index]
        twist = twists[index]
        return center, twist[:3] + np.cross(twist[3:], point - center)

    @staticmethod
    def _accumulate(collider, point, center, force, link_loads, root_loads):
        wrench = np.concatenate((force, np.cross(point - center, force)))
        index = collider.solid.owner_index
        if index < 0:
            root_loads[collider.owner] += wrench
        else:
            link_loads[collider.owner][index] += wrench

    def evaluate(self, terms, poses):
        """Sample one common entry state without changing any input or live scene.

        terms[owner] is (full RobotState, RobotKinematics, COM linear Jacobians,
        angular Jacobians); poses uses the same entry state and compiled keys.
        Returned full-coordinate efforts are sums of J_com.T F + J_omega.T M.
        """
        if set(terms) != set(self._robots):
            raise ValueError("direct contact terms must cover exactly the registered robots")
        if not set(self._shapes).issubset(poses):
            raise ValueError("direct contact poses must cover the compiled collision model")
        geometry, twists, roots, link_loads, root_loads = {}, {}, {}, {}, {}
        for owner, participant in self._robots.items():
            physical, geo, linear, angular = terms[owner]
            n = participant.tree.model.dof_count
            if (
                physical.v.shape != (n,)
                or linear.shape != (n, 3, n)
                or angular.shape != (n, 3, n)
                or geo.position.shape != (n, 3)
                or geo.rotation.shape != (n, 3, 3)
                or geo.com_position.shape != (n, 3)
            ):
                raise ValueError(f"robot {owner}: inconsistent full-coordinate contact terms")
            geometry[owner] = geo
            twists[owner] = np.column_stack((linear @ physical.v, angular @ physical.v))
            roots[owner] = np.asarray(participant.tree.root_xyz, dtype=float)
            link_loads[owner] = np.zeros((n, 6))
            root_loads[owner] = np.zeros(6)
        records = _Records(self._shapes, poses)
        for first, second, law in self._pairs:
            ga, gb = geometry[first.owner], geometry[second.owner]
            center = self._world_point(second, second.solid.vertices[0], gb)
            index = first.solid.owner_index
            local_center = (
                center if index < 0 else (center - ga.position[index]) @ ga.rotation[index]
            )
            try:
                witness = first.solid.witness(local_center, second.solid.radius)
            except ValueError as exc:
                raise ValueError(
                    f"robot {first.owner} {first.solid.link}/{first.solid.shape} with "
                    f"robot {second.owner} {second.solid.link}/{second.solid.shape}: {exc}"
                ) from exc
            if witness is None:
                continue
            gap, normal, surface, feature = witness
            if index >= 0:
                normal = ga.rotation[index] @ normal
            pa = self._world_point(first, surface, ga)
            pb = center - second.solid.radius * normal
            point = (pa + pb) / 2
            ca, va = self._point_motion(first, point, ga, twists[first.owner], roots[first.owner])
            cb, vb = self._point_motion(second, point, gb, twists[second.owner], roots[second.owner])
            relative = vb - va
            normal_load, forces = sample_loads(
                np.array([-gap]), normal[None, :], relative[None, :], law,
            )
            force = forces[0]
            self._accumulate(first, point, ca, -force, link_loads, root_loads)
            self._accumulate(second, point, cb, force, link_loads, root_loads)
            records.add(
                first.key,
                second.key,
                dict(
                    normal_world=normal,
                    force_on_payload_N=force,
                    link_surface_m=pa,
                    payload_surface_m=pb,
                    link_com_world_m=ca,
                    payload_com_world_m=cb,
                    point_world_m=point,
                    weight=1.0,
                    normal_load_N=normal_load[0],
                    normal_rate_m_s=float(relative @ normal),
                    gap_m=gap,
                    pair_power_W=float(force @ relative),
                    feature=feature,
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
