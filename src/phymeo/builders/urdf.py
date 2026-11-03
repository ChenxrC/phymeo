"""Local URDF import for massive scalar links connected to a fixed world frame."""

from collections import deque
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionModel, CollisionMotion, CollisionShape
from ..core.robot import RobotModel
from ..dynamics.robot_kinematics import forward_kinematics
from ..dynamics.rotation import quaternion_matrix


@dataclass(frozen=True)
class LinkCollision:
    """A named collision element. Local vertices are expressed in the URDF link frame."""

    name: str
    link_name: str
    link_index: int
    origin_xyz: tuple[float, ...]
    origin_xyzw: tuple[float, ...]
    shape: CollisionShape
    local_vertices: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, eq=False)
class RobotAsset:
    """Moving links and joints use the model DOF order. Link index -1 is the fixed root.

    Limits retain URDF units (rad or m, N·m or N, rad/s or m/s). They are asset
    metadata and may be passed to a controller or physical stop with explicit gains.
    The collision model assigns body 0 to the base and i+1 to moving link i.
    """

    name: str
    root_link: str
    model: RobotModel
    link_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    effort_limit: tuple[float, ...]
    velocity_limit: tuple[float, ...]
    collisions: tuple[LinkCollision, ...]
    collision_model: CollisionModel

    def collision_motion(self, q):
        """Build a static pose for each shape at q, ready for the public detect query."""
        geometry = forward_kinematics(self.model, q)
        paths = {}
        for collider in self.collisions:
            points = np.asarray(collider.local_vertices)
            link = collider.link_index
            if link >= 0:
                points = geometry.position[link] + points @ geometry.rotation[link].T
            paths[collider.shape.key] = CollisionMotion(points)
        return paths


def _name(element, attribute, where):
    value = element.get(attribute, "").strip()
    if not value:
        raise ValueError(f"{where} requires a nonempty {attribute}")
    return value


def _attributes(element, allowed, where):
    extra = set(element.attrib) - allowed
    if extra:
        raise ValueError(f"{where} contains unsupported attribute(s) {sorted(extra)}")


def _children(element, allowed, where):
    for child in element:
        if child.tag not in allowed:
            raise ValueError(f"{where} contains unsupported <{child.tag}>")


def _child(element, tag, where, *, required=True):
    items = element.findall(tag)
    if len(items) > 1 or (required and not items):
        raise ValueError(f"{where} requires exactly one <{tag}>")
    return items[0] if items else None


def _numbers(element, key, count, where, default=None):
    if key not in element.attrib:
        if default is not None:
            return np.array(default, dtype=float)
        raise ValueError(f"{where} requires {key}")
    try:
        values = np.array([float(s) for s in element.attrib[key].split()])
    except ValueError as exc:
        raise ValueError(f"{where} has invalid {key}") from exc
    if len(values) != count or not np.isfinite(values).all():
        raise ValueError(f"{where} requires {count} finite value(s) for {key}")
    return values


def _number(element, key, where):
    return float(_numbers(element, key, 1, where)[0])


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
    quat = np.array([
        sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy,
    ])
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
    if root and mass == 0 and not np.any(tensor):
        return None
    scale = np.max(np.abs(tensor))
    if mass <= 0 or scale == 0:
        raise ValueError(f"{where} needs positive mass and inertia")
    rotation = quaternion_matrix(orientation[None, :])[0]
    # Scale first so the roundoff budget is relative to authored inertia.
    normalized = rotation @ (tensor / scale) @ rotation.T
    diagonal = normalized.diagonal()
    off_diagonal = normalized - np.diag(diagonal)
    budget = 32 * np.finfo(float).eps if np.any(orientation[:3]) else 0.0
    if np.max(np.abs(off_diagonal)) > budget:
        raise ValueError(f"{where} has non-diagonal inertia in the link frame")
    if np.any(diagonal <= 0) or 2 * np.max(diagonal) > np.sum(diagonal) + 8 * np.finfo(float).eps:
        raise ValueError(f"{where} violates positive principal inertia or triangle inequalities")
    with np.errstate(over="ignore"):
        inertia = diagonal * scale
    if not np.isfinite(inertia).all() or np.any(inertia <= 0):
        raise ValueError(f"{where} is outside the representable inertia range")
    return mass, com, inertia


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
    scale = np.max(np.abs(axis))
    if scale == 0:
        raise ValueError(f"{where} has a zero axis")
    axis = axis / scale
    axis /= np.hypot.reduce(axis)
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


def _link_collisions(link, link_name, link_index):
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
        if primitive.tag == "sphere":
            _attributes(primitive, {"radius"}, where)
            radius = _number(primitive, "radius", where)
            if radius <= 0:
                raise ValueError(f"{where} requires a positive sphere radius")
            kind = "sphere"
            vertices = origin[None, :]
        elif primitive.tag == "box":
            _attributes(primitive, {"size"}, where)
            size = _numbers(primitive, "size", 3, where)
            if np.any(size <= 0):
                raise ValueError(f"{where} requires positive box size")
            kind, radius = "hull", 0.0
            corners = np.array(list(product((-0.5, 0.5), repeat=3))) * size
            vertices = origin + corners @ quaternion_matrix(orientation[None, :])[0].T
        else:
            raise ValueError(f"{where}: unsupported collision geometry <{primitive.tag}>")
        if not np.isfinite(vertices).all():
            raise ValueError(f"{where} has unrepresentable local vertices")
        key = CollisionKey(0, link_index + 1, ordinal)
        shape = CollisionShape(key, kind, tuple(range(len(vertices))), radius)
        result.append(LinkCollision(
            name, link_name, link_index, tuple(origin), tuple(orientation), shape,
            tuple(tuple(row) for row in vertices),
        ))
    return result


def load_urdf(path: str | Path, *, gravity=(0.0, 0.0, -9.81)) -> RobotAsset:
    """Read a local, fixed-base URDF 1.0 tree with bounded revolute/prismatic scalar joints.

    An absent joint axis defaults to +X. Moving links need positive mass and a
    COM inertia diagonal after rotating the URDF inertial frame into the link.
    Optional fixed-root inertia is validated and omitted from moving dynamics.
    Spheres and boxes retain their authored link, name and collision transform.
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
    indices = {root: -1, **{link: i for i, link in enumerate(order)}}
    _inertial(links[root], root, root=True)
    colliders = _link_collisions(links[root], root, -1)
    parents, joints, joint_data, bodies = [], [], [], []
    for link_name in order:
        parent, joint_name, joint = inbound[link_name]
        parents.append(indices[parent])
        joints.append(joint_name)
        joint_data.append(_joint_data(joint, joint_name))
        bodies.append(_inertial(links[link_name], link_name))
        colliders.extend(_link_collisions(links[link_name], link_name, indices[link_name]))
    kinds, positions, orientations, axes, limits = zip(*joint_data, strict=True)
    masses, com, inertia = zip(*bodies, strict=True)
    lower, upper, effort, velocity = zip(*limits, strict=True)
    model = RobotModel(
        parents=np.array(parents), joint_types=kinds, joint_position=np.array(positions),
        joint_orientation=np.array(orientations), joint_axis=np.array(axes), mass=np.array(masses),
        com=np.array(com), inertia_diagonal=np.array(inertia), gravity=np.asarray(gravity, dtype=float),
    )
    return RobotAsset(
        name, root, model, tuple(order), tuple(joints), lower, upper, effort, velocity,
        tuple(colliders), CollisionModel(tuple(c.shape for c in colliders)),
    )
