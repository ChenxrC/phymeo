"""Local URDF assets with one physical free root and ordinary scalar-joint children.

The explicit loader preserves authored link frames and root COM inertia. It
shares the existing strict URDF geometry and scalar-joint parser. Fixed-joint
folding is explicit; mimic and a new root principal-axis frame remain unsupported.
"""

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from ..constraints.collision_shapes import CollisionModel, CollisionMotion
from ..core.rigid import RigidBodyModel
from ..dynamics.floating_robot import FloatingRobotModel, FloatingRobotState, floating_kinematics
from ..dynamics.rotation import matrix_quaternion
from ._robot_asset import LinkCollision, assemble_robot
from ._robot_xml import _attributes, _children, _name
from .urdf import _inertial, _joint_data, _link_collisions, _tree


@dataclass(frozen=True, eq=False)
class FloatingRobotAsset:
    """Real massive root plus its scalar children, with no dummy coordinates.

    link_names/joint_names list the children in scalar-joint order; root_link is
    separate. Collision link indices retain the file-import convention (-1 for
    the root, i for child i), while FloatingRobotModel body indices are 0 and i+1.
    Full q is world root-origin XYZ, body-to-world XYZW, then child coordinates.
    The root body's axes must be principal inertia axes; COM may be eccentric.
    """

    name: str
    root_link: str
    model: FloatingRobotModel
    link_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    effort_limit: tuple[float | None, ...]
    velocity_limit: tuple[float | None, ...]
    collisions: tuple[LinkCollision, ...]
    collision_model: CollisionModel

    def collision_motion(self, q, q_end=None):
        """Read true static full-tree poses; a floating interval is not a chord."""
        if q_end is not None:
            raise ValueError("floating asset interval collision motion is not implemented")
        state = FloatingRobotState(q, np.zeros(6 + len(self.joint_names)))
        geometry = floating_kinematics(self.model, state)
        result = {}
        for collider in self.collisions:
            body = collider.link_index + 1
            points = (
                geometry.position[body]
                + np.asarray(collider.local_vertices) @ geometry.rotation[body].T
            )
            orientation = (
                None
                if collider.shape.curved is None
                else matrix_quaternion(geometry.rotation[body])
            )
            result[collider.shape.key] = CollisionMotion(points, body_orientation_xyzw=orientation)
        return result


def load_floating_urdf(
    path: str | Path, *, gravity=(0.0, 0.0, -9.81), collapse_fixed_joints=False
) -> FloatingRobotAsset:
    """Load local URDF 1.0 with positive massive root and bounded scalar children.

    Root-only assets are valid. All COM inertia tensors must be diagonal after
    rotating the authored inertial frame into its link frame. Legal unsupported
    fixed joints require collapse_fixed_joints=True; mimic still fails explicitly.
    Fixed members then contribute complete group mass/inertia and material frames.
    The ordinary fixed-base loader is unchanged.
    Lengths/masses/inertia are SI. Visual elements do not become collision shapes.
    """
    if not isinstance(collapse_fixed_joints, (bool, np.bool_)):
        raise ValueError("collapse_fixed_joints must be True or False")
    try:
        robot = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid URDF XML in {path}: {exc}") from exc
    if robot.tag != "robot":
        raise ValueError("URDF root element must be <robot>")
    _attributes(robot, {"name", "version"}, "robot")
    if robot.get("version", "1.0") != "1.0":
        raise ValueError("floating URDF requires version 1.0")
    name = _name(robot, "name", "robot")
    _children(robot, {"link", "joint", "material"}, f"robot {name!r}")
    joints = robot.findall("joint")
    for joint in joints:
        label = _name(joint, "name", "joint")
        if joint.get("type") == "fixed" and not collapse_fixed_joints:
            raise ValueError(f"joint {label!r}: floating fixed-joint folding is not implemented")
        if joint.find("mimic") is not None:
            raise ValueError(f"joint {label!r}: floating mimic is not implemented")
    if joints:
        root, links, order, inbound = _tree(robot)
    else:
        entries = robot.findall("link")
        if len(entries) != 1:
            raise ValueError("floating URDF requires exactly one connected root")
        root = _name(entries[0], "name", "link")
        _attributes(entries[0], {"name"}, f"link {root!r}")
        _children(entries[0], {"inertial", "collision", "visual"}, f"link {root!r}")
        links, order, inbound = {root: entries[0]}, [], {}
    if collapse_fixed_joints:
        from ._floating_fixed_asset import folded_floating_robot

        return folded_floating_robot(name, root, links, order, inbound, gravity, path)
    mass, center, inertia = _inertial(links[root], root)
    root_model = RigidBodyModel(
        np.array([1.0 / mass]), inertia[None, :], np.asarray(gravity, dtype=float)
    )
    colliders = _link_collisions(links[root], root, -1, path)
    if not order:
        return FloatingRobotAsset(
            name,
            root,
            FloatingRobotModel(root_model, center),
            (),
            (),
            (),
            (),
            (),
            (),
            tuple(colliders),
            CollisionModel(tuple(row.shape for row in colliders)),
        )
    indices = {root: -1, **{link: i for i, link in enumerate(order)}}
    parents, names, data, bodies = [], [], [], []
    for link in order:
        parent, joint_name, joint = inbound[link]
        parents.append(indices[parent])
        names.append(joint_name)
        data.append(_joint_data(joint, joint_name))
        bodies.append(_inertial(links[link], link))
        colliders.extend(_link_collisions(links[link], link, indices[link], path))
    child_asset = assemble_robot(
        name, root, order, parents, names, data, bodies, colliders, gravity
    )
    return FloatingRobotAsset(
        name,
        root,
        FloatingRobotModel(root_model, center, child_asset.model),
        child_asset.link_names,
        child_asset.joint_names,
        child_asset.lower,
        child_asset.upper,
        child_asset.effort_limit,
        child_asset.velocity_limit,
        child_asset.collisions,
        child_asset.collision_model,
    )
