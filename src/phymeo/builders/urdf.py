"""Local URDF import for massive scalar links connected to a fixed world frame."""

from collections import deque
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from ._robot_asset import LinkCollision as LinkCollision
from ._robot_asset import RobotAsset, assemble_robot, make_collider, principal_inertia, unit_axis
from ._robot_mesh import load_obj_mesh
from ._robot_xml import _attributes, _child, _children, _name, _number, _numbers
from ._urdf_frames import FoldedURDFAsset as FoldedURDFAsset


def _origin(element, where):
    origin = _child(element, "origin", where, required=False)
    if origin is None:
        return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
    if set(origin.attrib) - {"xyz", "rpy"} or len(origin):
        raise ValueError(f"{where} supports origin xyz and rpy only")
    xyz = _numbers(origin, "xyz", 3, where, (0, 0, 0))
    half = _numbers(origin, "rpy", 3, where, (0, 0, 0)) / 2
    sr, sp, sy = np.sin(half)
    cr, cp, cy = np.cos(half)
    # URDF roll, pitch, yaw are extrinsic rotations: Rz(yaw) Ry(pitch) Rx(roll).
    quat = np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )
    return xyz, quat


def _inertial(link, name, *, root=False):
    where = f"link {name!r} inertial"
    entry = _child(link, "inertial", where, required=not root)
    if entry is None:
        return None
    _attributes(entry, set(), where)
    _children(entry, {"mass", "inertia", "origin"}, where)
    mass_element = _child(entry, "mass", where)
    _attributes(mass_element, {"value"}, where)
    _children(mass_element, set(), where)
    mass = _number(mass_element, "value", where)
    com, orientation = _origin(entry, where)
    entry = _child(entry, "inertia", where)
    _attributes(entry, {"ixx", "ixy", "ixz", "iyy", "iyz", "izz"}, where)
    _children(entry, set(), where)
    xx, xy, xz, yy, yz, zz = (
        _number(entry, key, where) for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    )
    tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    return principal_inertia(mass, com, orientation, tensor, where, root=root)


def _joint_data(joint, name):
    where = f"joint {name!r}"
    kind = _name(joint, "type", where)
    if kind not in {"revolute", "prismatic"}:
        raise ValueError(f"{where}: unsupported joint type {kind!r}")
    _attributes(joint, {"name", "type"}, where)
    _children(joint, {"parent", "child", "origin", "axis", "limit"}, where)
    position, orientation = _origin(joint, where)
    element = _child(joint, "axis", where, required=False)
    if element is None:
        axis = np.array([1.0, 0, 0])
    else:
        _attributes(element, {"xyz"}, where + " axis")
        _children(element, set(), where + " axis")
        axis = _numbers(element, "xyz", 3, where + " axis")
    axis = unit_axis(axis, where)
    limit = _child(joint, "limit", where)
    keys = ("lower", "upper", "effort", "velocity")
    _attributes(limit, set(keys), where + " limit")
    _children(limit, set(), where + " limit")
    bounds = tuple(_number(limit, key, where + " limit") for key in keys)
    if bounds[0] >= bounds[1] or min(bounds[2:]) < 0:
        raise ValueError(f"{where} has an invalid limit range, effort or velocity")
    return kind, position, orientation, axis, bounds


def _tree(robot):
    links = {}
    for link in robot.findall("link"):
        name = _name(link, "name", "link")
        if name in links:
            raise ValueError(f"duplicate link {name!r}")
        _attributes(link, {"name"}, f"link {name!r}")
        _children(link, {"inertial", "collision", "visual"}, f"link {name!r}")
        links[name] = link
    children = {name: [] for name in links}
    inbound = {}
    names = set()
    for joint in robot.findall("joint"):
        name = _name(joint, "name", "joint")
        if name in names:
            raise ValueError(f"duplicate joint {name!r}")
        names.add(name)
        where = f"joint {name!r}"
        ends = []
        for side in ("parent", "child"):
            endpoint = _child(joint, side, where)
            _attributes(endpoint, {"link"}, f"{where} {side}")
            _children(endpoint, set(), f"{where} {side}")
            ends.append(_name(endpoint, "link", f"{where} {side}"))
        parent, child = ends
        for node in (parent, child):
            if node not in links:
                raise ValueError(f"{where} references unknown link {node!r}")
        if child in inbound or parent == child:
            raise ValueError(f"{where} gives child {child!r} multiple parents or a self cycle")
        inbound[child] = (parent, name, joint)
        children[parent].append(child)
    roots = links.keys() - inbound.keys()
    if len(roots) != 1:
        raise ValueError("fixed-base URDF requires exactly one connected root")
    root = next(iter(roots))
    order = []
    queue = deque(children[root])
    while queue:
        child = queue.popleft()
        order.append(child)
        queue.extend(children[child])
    if len(order) != len(links) - 1:
        raise ValueError("URDF contains a disconnected cycle")
    if not order:
        raise ValueError("RobotModel requires at least one moving link")
    return root, links, order, inbound


def _link_collisions(link, link_name, link_index, xml_path):
    result = []
    names = set()
    for ordinal, collision in enumerate(link.findall("collision")):
        name = collision.get("name", f"collision_{ordinal}").strip()
        where = f"link {link_name!r} collision {name!r}"
        if not name or name in names:
            raise ValueError(f"{where} has an empty or duplicate name")
        names.add(name)
        _attributes(collision, {"name"}, where)
        _children(collision, {"origin", "geometry"}, where)
        origin, orientation = _origin(collision, where)
        geometry = _child(collision, "geometry", where)
        _attributes(geometry, set(), where + " geometry")
        if len(geometry) != 1:
            raise ValueError(f"{where} geometry needs exactly one shape")
        primitive = geometry[0]
        _children(primitive, set(), where + " geometry")
        if primitive.tag == "mesh":
            _attributes(primitive, {"filename", "scale"}, where)
            size = load_obj_mesh(
                xml_path,
                _name(primitive, "filename", where),
                _numbers(primitive, "scale", 3, where, (1.0, 1.0, 1.0)),
                where=where,
            )
        elif primitive.tag == "sphere":
            _attributes(primitive, {"radius"}, where)
            size = _numbers(primitive, "radius", 1, where)
        elif primitive.tag == "cylinder":
            _attributes(primitive, {"radius", "length"}, where)
            size = np.array(
                [_number(primitive, "radius", where), 0.5 * _number(primitive, "length", where)]
            )
        elif primitive.tag == "box":
            _attributes(primitive, {"size"}, where)
            size = _numbers(primitive, "size", 3, where) / 2
        else:
            raise ValueError(f"{where}: unsupported collision geometry <{primitive.tag}>")
        result.append(
            make_collider(
                name, link_name, link_index, ordinal, origin, orientation, primitive.tag, size
            )
        )
    return result


def load_urdf(
    path: str | Path, *, gravity=(0.0, 0.0, -9.81), collapse_fixed_joints=None
) -> RobotAsset:
    """Read a local, fixed-base URDF 1.0 tree with bounded revolute/prismatic scalar joints.

    With the default setting a tree containing fixed joints groups their inertial
    and collision components with the nearest moving ancestor. The result exposes
    every authored link through sites and authored_link_pose. Its moving bodies use
    their aggregate COM and principal axes. Explicit True also selects these frames
    for an all-scalar tree. False keeps the strict scalar-only importer, which needs
    each COM inertia to be diagonal in the original moving link. Existing scalar-only
    assets use that strict path by default. Static-root inertia is checked and omitted
    from dynamics. An absent moving joint axis defaults to +X.
    Spheres, boxes, Z-axis cylinders and local closed OBJ meshes retain their link and transform.
    Visuals and global display materials are ignored. Other joint/shape types
    and semantic elements (e.g. mimic or joint dynamics) raise ValueError.
    """
    try:
        robot = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid URDF XML in {path}: {exc}") from exc
    if robot.tag != "robot":
        raise ValueError("URDF root element must be <robot>")
    _attributes(robot, {"name", "version"}, "robot")
    if robot.get("version", "1.0") != "1.0":
        raise ValueError(f"Unsupported URDF version {robot.get('version')!r}, expected 1.0")
    name = _name(robot, "name", "robot")
    _children(robot, {"link", "joint", "material"}, f"robot {name!r}")
    root, links, order, inbound = _tree(robot)
    if collapse_fixed_joints is not None and not isinstance(
        collapse_fixed_joints, (bool, np.bool_)
    ):
        raise ValueError("collapse_fixed_joints must be True, False or None")
    fold = (
        any(joint.get("type") == "fixed" for _, _, joint in inbound.values())
        if collapse_fixed_joints is None
        else collapse_fixed_joints
    )
    if fold:
        from ._urdf_fixed import folded_robot

        return folded_robot(name, root, links, order, inbound, gravity, path)
    indices = {root: -1, **{link: i for i, link in enumerate(order)}}
    _inertial(links[root], root, root=True)
    colliders = _link_collisions(links[root], root, -1, path)
    parents, joints, joint_data, bodies = [], [], [], []
    for link_name in order:
        parent, joint_name, joint = inbound[link_name]
        parents.append(indices[parent])
        joints.append(joint_name)
        joint_data.append(_joint_data(joint, joint_name))
        bodies.append(_inertial(links[link_name], link_name))
        colliders.extend(_link_collisions(links[link_name], link_name, indices[link_name], path))
    return assemble_robot(
        name, root, order, parents, joints, joint_data, bodies, colliders, gravity
    )
