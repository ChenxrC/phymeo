"""Explicit-inertia floating MJCF hinge trees, retaining principal-frame transforms.

This importer admits one free root and one limited hinge per massive child.
Visual meshes are ignored only when their collision masks are both zero.
"""

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionModel, CollisionShape
from ..core.rigid import RigidBodyModel
from ..dynamics.floating_robot import FloatingRobotModel
from ..dynamics.rotation import matrix_quaternion, quaternion_matrix
from ._robot_asset import LinkCollision, assemble_robot, make_collider, unit_axis
from .floating_urdf import FloatingRobotAsset


def _numbers(text, length, name):
    value = np.fromstring(text, sep=" ")
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f"invalid {name}: expected {length} finite values")
    return value


def _rotation(element):
    if "euler" in element.attrib or "axisangle" in element.attrib:
        raise ValueError("floating MJCF requires explicit quaternions")
    quat = _numbers(element.get("quat", "1 0 0 0"), 4, "quaternion")
    if np.linalg.norm(quat) == 0:
        raise ValueError("zero quaternion")
    quat /= np.linalg.norm(quat)
    return quaternion_matrix(quat[[1, 2, 3, 0]][None])[0]


@dataclass(frozen=True, eq=False)
class FloatingMJCFAsset(FloatingRobotAsset):
    """Child model frames are principal axes; these rotations map them to MJCF axes."""

    principal_to_authored_xyzw: tuple[tuple[float, ...], ...]


def load_floating_mjcf(path, *, gravity=(0.0, 0.0, -9.81)):
    document = ET.parse(Path(path)).getroot()
    compiler = document.find("compiler")
    if compiler is None or compiler.get("angle", "degree") != "radian":
        raise ValueError("floating MJCF requires compiler angle=radian")
    defaults = {"": {}}

    def default_tree(element, inherited):
        attrs = dict(inherited)
        geom = element.find("geom")
        if geom is not None:
            attrs.update(geom.attrib)
        defaults[element.get("class", "")] = attrs
        for child in element.findall("default"):
            default_tree(child, attrs)

    for entry in document.findall("default"):
        default_tree(entry, {})
    roots = [body for world in document.findall("worldbody") for body in world.findall("body")]
    if len(roots) != 1 or len(roots[0].findall("freejoint")) != 1:
        raise ValueError("floating MJCF needs exactly one root with one freejoint")
    root = roots[0]
    if document.find("equality") is not None or document.find("tendon") is not None:
        raise ValueError("MJCF equality and tendon mechanisms require a different importer")
    for joint in document.iter("joint"):
        for name in ("damping", "armature", "frictionloss", "stiffness", "ref"):
            if float(joint.get(name, "0")) != 0:
                raise ValueError(f"nonzero MJCF joint {name} needs an actuator/dynamics adapter")
    motors = {row.get("joint"): row for row in document.findall("actuator/motor")}
    colliders, links, parents, joints, joint_data, bodies, frames = [], [], [], [], [], [], []

    def inertia(body):
        entry = body.find("inertial")
        if entry is None or "diaginertia" not in entry.attrib:
            raise ValueError("each body needs explicit mass and principal diaginertia")
        mass = float(entry.get("mass", "nan"))
        diagonal = _numbers(entry.get("diaginertia", ""), 3, "diaginertia")
        center = _numbers(entry.get("pos", "0 0 0"), 3, "inertial position")
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("body mass must be positive")
        return mass, center, diagonal, _rotation(entry)

    root_mass, root_com, root_inertia, root_frame = inertia(root)

    def visit(body, parent, parent_frame, inherited_class):
        is_root = body is root
        frame = np.eye(3) if is_root else inertia(body)[3]
        index = -1 if is_root else len(links)
        name = body.get("name")
        if not name:
            raise ValueError("each body needs a name")
        childclass = body.get("childclass", inherited_class)
        if not is_root:
            entries = body.findall("joint")
            if len(entries) != 1 or entries[0].get("type", "hinge") != "hinge":
                raise ValueError("each child requires exactly one hinge")
            joint = entries[0]
            if np.any(_numbers(joint.get("pos", "0 0 0"), 3, "joint position")):
                raise ValueError("nonzero local joint anchors are not implemented")
            joint_name = joint.get("name")
            motor = motors.get(joint_name)
            if not joint_name or motor is None:
                raise ValueError("each moving joint requires a named motor")
            limits = _numbers(joint.get("range", ""), 2, "joint range")
            control = _numbers(motor.get("ctrlrange", ""), 2, "motor control range")
            if limits[0] >= limits[1] or control[0] >= 0 or control[1] <= 0:
                raise ValueError("invalid joint/motor range")
            gear = _numbers(motor.get("gear", "1"), 1, "motor gear")[0]
            if gear != 1 or not np.isclose(-control[0], control[1]):
                raise ValueError("only unit-gear symmetric torque motors are supported")
            position = parent_frame.T @ _numbers(body.get("pos", "0 0 0"), 3, "body position")
            orientation = matrix_quaternion(parent_frame.T @ _rotation(body) @ frame)
            axis = frame.T @ unit_axis(
                _numbers(joint.get("axis", "0 0 1"), 3, "joint axis"), joint_name
            )
            mass, center, diagonal, _ = inertia(body)
            links.append(name)
            parents.append(parent)
            joints.append(joint_name)
            bodies.append((mass, frame.T @ center, diagonal))
            joint_data.append(
                ("revolute", position, orientation, axis, (*limits, control[1], None))
            )
            frames.append(tuple(matrix_quaternion(frame)))
        for ordinal, geom in enumerate(body.findall("geom")):
            attrs = {**defaults.get(geom.get("class", childclass), {}), **geom.attrib}
            if attrs.get("contype", "1") == "0" and attrs.get("conaffinity", "1") == "0":
                continue
            kind = attrs.get("type", "sphere")
            label = attrs.get("name", f"{name}-collision-{ordinal}")
            if kind == "capsule" and "fromto" in attrs:
                ends = _numbers(attrs["fromto"], 6, "capsule endpoints").reshape(2, 3) @ frame
                radius = _numbers(attrs.get("size", ""), 1, "capsule radius")[0]
                shape = CollisionShape(
                    CollisionKey(0, index + 1, ordinal), "capsule", (0, 1), radius
                )
                colliders.append(
                    LinkCollision(
                        label, name, index, (0, 0, 0), (0, 0, 0, 1), shape, tuple(map(tuple, ends))
                    )
                )
            elif kind in ("sphere", "box", "capsule"):
                sizes = {"sphere": 1, "box": 3, "capsule": 2}
                size = _numbers(attrs.get("size", ""), sizes[kind], "geometry size")
                element = ET.Element("geom", attrs)
                pos = frame.T @ _numbers(attrs.get("pos", "0 0 0"), 3, "geometry position")
                quat = matrix_quaternion(frame.T @ _rotation(element))
                colliders.append(make_collider(label, name, index, ordinal, pos, quat, kind, size))
            else:
                raise ValueError(f"unsupported physical MJCF geometry: {kind}")
        for child in body.findall("body"):
            visit(child, index, frame, childclass)

    visit(root, -1, np.eye(3), root.get("childclass", ""))
    if not links or len(set(joints)) != len(joints) or len(set(links)) != len(links):
        raise ValueError("expected a nonempty tree with unique body and joint names")
    child_asset = assemble_robot(
        document.get("model", "floating-mjcf"),
        root.get("name"),
        links,
        parents,
        joints,
        joint_data,
        bodies,
        colliders,
        gravity,
    )
    model = FloatingRobotModel(
        RigidBodyModel(
            np.array([1 / root_mass]), root_inertia[None], np.array(gravity, dtype=float)
        ),
        root_com,
        child_asset.model,
        root_inertia_frame_xyzw=matrix_quaternion(root_frame),
    )
    return FloatingMJCFAsset(
        child_asset.name,
        child_asset.root_link,
        model,
        child_asset.link_names,
        child_asset.joint_names,
        child_asset.lower,
        child_asset.upper,
        child_asset.effort_limit,
        child_asset.velocity_limit,
        tuple(colliders),
        CollisionModel(tuple(row.shape for row in colliders)),
        tuple(frames),
    )
