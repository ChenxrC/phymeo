"""Read the classic Franka Emika Panda URDF as an ordinary fixed-base RobotAsset."""

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from xml.etree import ElementTree as ET

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionMotion
from ..core.robot import robot_array
from ..core.validation import finite_real
from ..dynamics.robot_kinematics import forward_kinematics
from ..dynamics.rotation import matrix_quaternion, quaternion_matrix
from ..sensors.robot_frame import RobotSite
from ._robot_asset import RobotAsset, assemble_robot, make_collider
from ._robot_xml import _child, _children, _number
from .urdf import _joint_data, _link_collisions, _origin, _tree

PANDA_URDF = Path(__file__).with_name("panda_assets") / "panda.urdf"
_ARM = tuple(f"panda_link{i}" for i in range(1, 8))
_MOVING = (*_ARM, "panda_leftfinger", "panda_rightfinger")
_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8)) + (
    "panda_finger_joint1",
    "panda_finger_joint2",
)
_PARENTS = ("panda_link0", *_ARM[:-1], "panda_hand", "panda_hand")
_FIXED = {
    "panda_link8": ("panda_link7", "panda_joint8"),
    "panda_hand": ("panda_link8", "panda_hand_joint"),
    "panda_hand_tcp": ("panda_hand", "panda_hand_tcp_joint"),
}


def _rotation(quat):
    return quaternion_matrix(np.array([quat], dtype=float))[0]


def _origin_pose(element, name):
    xyz, xyzw = _origin(element, name)
    return xyz, _rotation(xyzw)


def _mass_component(link, name, position, orientation):
    inertial = _child(link, "inertial", name, required=False)
    if inertial is None:
        return None
    _children(inertial, {"mass", "origin", "inertia"}, name)
    com, rotation = _origin_pose(inertial, name)
    mass = _number(_child(inertial, "mass", name), "value", name)
    entry = _child(inertial, "inertia", name)
    xx, xy, xz, yy, yz, zz = [
        _number(entry, k, name) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    ]
    tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    moments = np.linalg.eigvalsh(tensor)
    if mass <= 0 or moments[0] <= 0 or moments[2] > moments[:2].sum() * (1 + 2e-14):
        raise ValueError(f"{name}: invalid mass or COM inertia")
    frame = orientation @ rotation
    return mass, position + orientation @ com, frame @ tensor @ frame.T


def _principal(parts):
    mass = sum(m for m, _, _ in parts)
    center = sum(m * p for m, p, _ in parts) / mass
    tensor = np.zeros((3, 3))
    for m, p, inertia in parts:
        d = p - center
        tensor += inertia + m * ((d @ d) * np.eye(3) - np.outer(d, d))
    if np.any(tensor - np.diag(tensor.diagonal())):
        moments, axes = np.linalg.eigh(tensor)
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1
    else:
        moments, axes = tensor.diagonal(), np.eye(3)
    return axes, (mass, axes.T @ center, moments)


def _moving_joint(joint, name):
    # Drives are supplied separately to Phymeo. Keep the original bounded joint data.
    _children(
        joint,
        {"parent", "child", "origin", "axis", "limit", "safety_controller", "dynamics", "mimic"},
        name,
    )
    motion = deepcopy(joint)
    for tag in ("safety_controller", "dynamics", "mimic"):
        optional = _child(motion, tag, name, required=False)
        if optional is not None:
            motion.remove(optional)
    mimic = joint.find("mimic")
    if name == _JOINTS[-1]:
        if (
            mimic is None
            or mimic.get("joint") != _JOINTS[-2]
            or float(mimic.get("multiplier", "1")) != 1
            or float(mimic.get("offset", "0")) != 0
        ):
            raise ValueError("Panda requires an equal-displacement, zero-offset right finger mimic")
    elif mimic is not None:
        raise ValueError(f"{name}: unsupported mimic")
    return _joint_data(motion, name)


def _proxy_capsules(colliders, name, owner):
    # The upstream URDF expresses each capsule as a cylinder and two closed end spheres.
    # They must be geometrically the same union before one collision shape can represent it.
    if len(colliders) % 3:
        raise ValueError(f"{name}: incomplete capsule proxy")
    result = []
    for i in range(0, len(colliders), 3):
        barrel, first, second = colliders[i : i + 3]
        if [c.shape.kind for c in (barrel, first, second)] != ["cylinder", "sphere", "sphere"]:
            raise ValueError(f"{name}: capsule proxy needs a cylinder and two spheres")
        radius, half = barrel.shape.curved.dimensions_m
        center = np.array(barrel.origin_xyz)
        q = np.array(barrel.origin_xyzw)
        offset = _rotation(q)[:, 2] * half
        a, b = np.array(first.origin_xyz) - center, np.array(second.origin_xyz) - center
        if (
            first.shape.radius != radius
            or second.shape.radius != radius
            or not (
                np.linalg.norm(a + b) <= 1e-12
                and min(np.linalg.norm(a - offset), np.linalg.norm(a + offset)) <= 1e-12
            )
        ):
            raise ValueError(f"{name}: sphere ends disagree with the capsule barrel")
        result.append(
            make_collider(
                f"{name}_{i // 3}",
                name,
                owner,
                i // 3,
                center,
                q,
                "capsule",
                np.array([radius, half]),
            )
        )
    return result


@dataclass(frozen=True, eq=False, kw_only=True)
class PandaAsset(RobotAsset):
    """Seven arm coordinates and two physical sliders; sites refer to the original URDF frames.

    The native moving frames use COM principal axes. Public collider vertices remain in their
    named URDF link frame; sites carry the transform into native coordinates for IK/point ports.
    Finger positions are independent dynamic coordinates. configuration maps the single nominal
    opening command to a matching pair of slider positions.
    """

    sites: Mapping[str, RobotSite]

    def configuration(self, arm_q, opening_m):
        arm = np.asarray(arm_q)
        if arm.dtype.kind not in "fiu":
            raise ValueError("Panda arm coordinates must be real numbers")
        arm = robot_array(arm.astype(np.float64), "arm_q", ndim=1)
        width = finite_real(opening_m, "opening_m")
        if arm.shape != (7,):
            raise ValueError("Panda needs seven arm joint coordinates")
        q = np.r_[arm, width / 2, width / 2]
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError(
                "Panda joint configuration or commanded opening is outside URDF limits"
            )
        return q

    def finger_opening(self, q):
        """Read the sum of the physical finger slider coordinates without enforcing symmetry."""
        values = robot_array(q, "q", ndim=1)
        if values.shape != (9,):
            raise ValueError("Panda needs seven arm and two finger coordinates")
        return float(values[-2] + values[-1])

    def authored_to_model_points(self, name, points_m):
        """Map authored local points to a native moving link, or to world for the fixed base."""
        site = self.sites[name]
        points = np.asarray(points_m, dtype=float)
        if points.ndim not in (1, 2) or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("authored points must be finite 3-vectors")
        return np.asarray(site.local_position_m) + points @ _rotation(site.local_orientation_xyzw).T

    def _pose(self, name, kinematics):
        site = self.sites[name]
        p, r = np.asarray(site.local_position_m), _rotation(site.local_orientation_xyzw)
        i = site.link_index
        if i is None:
            return p, r
        return kinematics.position[i] + kinematics.rotation[i] @ p, kinematics.rotation[i] @ r

    def authored_link_pose(self, name, q):
        """Return the named URDF link or pad site's world position and orientation matrix."""
        return self._pose(name, forward_kinematics(self.model, q))

    def collision_motion(self, q, q_end=None):
        fk = forward_kinematics(self.model, q)
        poses = {name: self._pose(name, fk) for name in {c.link_name for c in self.collisions}}
        paths = {}
        for collider in self.collisions:
            p, r = poses[collider.link_name]
            points = p + np.asarray(collider.local_vertices) @ r.T
            orientation = matrix_quaternion(r) if collider.shape.curved is not None else None
            paths[collider.shape.key] = CollisionMotion(points, body_orientation_xyzw=orientation)
        if q_end is not None:
            from ._robot_joint_motion import joint_interval_motion

            return joint_interval_motion(self, q, q_end, fk, paths)
        return paths


def load_panda(path=None, *, base_xyz=(0, 0, 0), base_xyzw=(0, 0, 0, 1), gravity=(0, 0, -9.81)):
    """Read the bundled classic Panda or a local URDF with the same named topology/hand profile.

    Optional *_sc fixed links contain nominal sphere/cylinder proxy triples. The finger pad
    boxes use the shipped names, with each inward surface normal along the finger's local Y
    axis. Visuals are excluded. CAD mesh/package URIs and Gazebo actuator plugins are outside
    the packaged model. URDF passive drive tags do not add loads to RobotModel.
    """
    path = PANDA_URDF if path is None else Path(path)
    robot = ET.parse(path).getroot()
    if robot.tag != "robot" or robot.get("name") != "panda":
        raise ValueError("expected the classic Panda robot URDF")
    _children(robot, {"link", "joint", "material"}, "Panda")
    root, links, order, incoming = _tree(robot)
    if root != "panda_link0" or not set((*_MOVING, *_FIXED)).issubset(links):
        raise ValueError("URDF lacks the named classic Panda arm, fixed hand or finger links")
    base = RobotSite(None, base_xyz, base_xyzw)
    base_r = _rotation(base.local_orientation_xyzw)
    frames = {root: (-1, np.zeros(3), np.eye(3))}
    joints = [None] * 9
    parents = [-1] * 9
    masses = [[] for _ in _MOVING]
    for name in order:
        parent, jname, joint = incoming[name]
        owner, parent_pos, parent_r = frames[parent]
        if name in _MOVING:
            index = _MOVING.index(name)
            kind, offset, orientation, axis, bounds = _moving_joint(joint, jname)
            if (
                jname != _JOINTS[index]
                or parent != _PARENTS[index]
                or kind != ("revolute" if index < 7 else "prismatic")
            ):
                raise ValueError(f"unexpected Panda joint topology at {name}")
            joints[index] = (
                kind,
                parent_pos + parent_r @ offset,
                parent_r @ _rotation(orientation),
                axis,
                bounds,
            )
            parents[index] = owner
            frames[name] = (index, np.zeros(3), np.eye(3))
        else:
            expected = _FIXED.get(name)
            if name.endswith("_sc"):
                expected = (name[:-3], name + "_joint")
            if (parent, jname) != expected or joint.get("type") != "fixed":
                raise ValueError(f"unexpected fixed Panda link or parent at {name}")
            _children(joint, {"parent", "child", "origin"}, jname)
            offset, r = _origin_pose(joint, jname)
            frames[name] = owner, parent_pos + parent_r @ offset, parent_r @ r
        owner, position, rotation = frames[name]
        component = _mass_component(links[name], name, position, rotation)
        if name in _MOVING and component is None:
            raise ValueError(f"{name}: missing moving-body inertial data")
        if component is not None and owner >= 0:
            masses[owner].append(component)
    _mass_component(links[root], root, np.zeros(3), np.eye(3))
    axes, bodies = zip(*(_principal(parts) for parts in masses), strict=True)
    native = []
    for i, (kind, xyz, orientation, axis, limits) in enumerate(joints):
        p = parents[i]
        before = base_r if p == -1 else axes[p].T
        point = before @ xyz + (np.asarray(base.local_position_m) if p == -1 else 0)
        native.append(
            (
                kind,
                point,
                matrix_quaternion(before @ orientation @ axes[i]),
                axes[i].T @ axis,
                limits,
            )
        )

    sites, colliders = {}, []
    body_shapes = defaultdict(list)
    for name, (owner, pos, r) in frames.items():
        if owner == -1:
            site = RobotSite(
                None,
                np.asarray(base.local_position_m) + base_r @ pos,
                matrix_quaternion(base_r @ r),
            )
        else:
            site = RobotSite(owner, axes[owner].T @ pos, matrix_quaternion(axes[owner].T @ r))
        sites[name] = site
        raw = _link_collisions(links[name], name, owner, path)
        if name.endswith("_sc"):
            raw = _proxy_capsules(raw, name, owner)
        for collider in raw:
            key = CollisionKey(0, owner + 1, len(body_shapes[owner]))
            collider = replace(collider, shape=replace(collider.shape, key=key))
            colliders.append(collider)
            body_shapes[owner].append(key)
            if name in _MOVING[-2:] and collider.name == name + "_pad":
                element = links[name].find(f"collision[@name='{collider.name}']/geometry/box")
                if element is None:
                    raise ValueError("Panda rubber pad must have box geometry")
                size_y = float(element.attrib["size"].split()[1])
                inward = -1 if name == _MOVING[-2] else 1
                rp = _rotation(collider.origin_xyzw)
                inner = np.array(collider.origin_xyz) + rp @ [0, inward * size_y / 2, 0]
                rs = _rotation(site.local_orientation_xyzw)
                sites[name.replace("finger", "pad")] = RobotSite(
                    owner, np.array(site.local_position_m) + rs @ inner, matrix_quaternion(rs @ rp)
                )
    if not {"panda_leftpad", "panda_rightpad"}.issubset(sites):
        raise ValueError("both named Panda finger pads are required")
    excluded = [
        (a, b)
        for i, parent in enumerate(parents)
        for a in body_shapes[parent]
        for b in body_shapes[i]
    ]
    asset = assemble_robot(
        "panda",
        root,
        _MOVING,
        parents,
        _JOINTS,
        native,
        bodies,
        colliders,
        gravity,
        root_xyz=base.local_position_m,
        root_xyzw=base.local_orientation_xyzw,
        excluded=excluded,
    )
    return PandaAsset(**vars(asset), sites=MappingProxyType(sites))
