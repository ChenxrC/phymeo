"""Import a local MJCF fixed-base tree with one limited scalar joint per moving body."""

from collections import deque
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from ..dynamics.rotation import quaternion_matrix
from ._robot_asset import RobotAsset, assemble_robot, make_collider, principal_inertia, unit_axis
from ._robot_mesh import mjcf_mesh_assets
from ._robot_xml import _attributes, _child, _children, _name, _number, _numbers


def _multiply(left, right):
    """Composition of two active rotations in native XYZW storage."""
    vector = left[3] * right[:3] + right[3] * left[:3] + np.cross(left[:3], right[:3])
    return np.append(vector, left[3] * right[3] - left[:3] @ right[:3])


def _pose(element, where, degrees, sequence, *, require_pos=False):
    position = _numbers(element, "pos", 3, where, None if require_pos else (0, 0, 0))
    if "quat" in element.attrib:
        if "euler" in element.attrib:
            raise ValueError(f"{where} cannot author both quat and euler")
        quat = _numbers(element, "quat", 4, where)[[1, 2, 3, 0]]
        scale = np.max(np.abs(quat))
        if scale == 0:
            raise ValueError(f"{where} has a zero quaternion")
        quat /= scale
        return position, quat / np.hypot.reduce(quat)
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    if "euler" in element.attrib:
        angles = _numbers(element, "euler", 3, where)
        if degrees:
            angles = np.deg2rad(angles)
        for axis, half_angle in zip(sequence, angles / 2, strict=True):
            turn = np.zeros(4)
            turn["xyz".index(axis.lower())] = np.sin(half_angle)
            turn[3] = np.cos(half_angle)
            # Lowercase axes rotate with the frame. Uppercase axes remain fixed.
            quat = _multiply(quat, turn) if axis.islower() else _multiply(turn, quat)
    return position, quat


def _compiler(robot):
    compiler = _child(robot, "compiler", "mujoco", required=False)
    if compiler is None:
        return True, "xyz", True
    _attributes(
        compiler, {"angle", "eulerseq", "inertiafromgeom", "autolimits", "meshdir"}, "compiler"
    )
    _children(compiler, set(), "compiler")
    angle = compiler.get("angle", "degree")
    if angle not in {"degree", "radian"}:
        raise ValueError(f"unsupported compiler angle {angle!r}")
    sequence = compiler.get("eulerseq", "xyz")
    if len(sequence) != 3 or any(axis not in "xyzXYZ" for axis in sequence):
        raise ValueError(f"invalid compiler eulerseq {sequence!r}")
    if compiler.get("inertiafromgeom", "auto") not in {"false", "auto"}:
        raise ValueError("compiler inertiafromgeom requires explicit body inertias")
    automatic = compiler.get("autolimits", "true")
    if automatic not in {"true", "false"}:
        raise ValueError(f"invalid compiler autolimits {automatic!r}")
    return angle == "degree", sequence, automatic == "true"


def _body_inertia(body, name, degrees, sequence, *, root=False):
    where = f"body {name!r} inertial"
    entry = _child(body, "inertial", where, required=not root)
    if entry is None:
        return None
    _attributes(entry, {"pos", "quat", "euler", "mass", "diaginertia", "fullinertia"}, where)
    _children(entry, set(), where)
    mass = _number(entry, "mass", where)
    com, frame = _pose(entry, where, degrees, sequence, require_pos=True)
    if ("diaginertia" in entry.attrib) == ("fullinertia" in entry.attrib):
        raise ValueError(f"{where} requires one of diaginertia and fullinertia")
    if "diaginertia" in entry.attrib:
        tensor = np.diag(_numbers(entry, "diaginertia", 3, where))
    else:
        if "euler" in entry.attrib or "quat" in entry.attrib:
            raise ValueError(
                f"{where}: fullinertia uses the body frame and cannot specify orientation"
            )
        xx, yy, zz, xy, xz, yz = _numbers(entry, "fullinertia", 6, where)
        tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    return principal_inertia(mass, com, frame, tensor, where, root=root)


def _joint(joint, name, degrees, automatic):
    where = f"joint {name!r}"
    _attributes(joint, {"name", "type", "axis", "pos", "ref", "limited", "range"}, where)
    _children(joint, set(), where)
    kind = joint.get("type", "hinge")
    if kind not in {"hinge", "slide"}:
        raise ValueError(f"{where}: unsupported MJCF joint type {kind!r}")
    if np.any(_numbers(joint, "pos", 3, where, (0, 0, 0))):
        raise ValueError(f"{where} has a nonzero body-local joint anchor pos")
    if "ref" in joint.attrib and _number(joint, "ref", where) != 0:
        raise ValueError(f"{where} supports ref=0 only")
    limited = joint.get("limited", "auto")
    if limited != "true" and not (limited == "auto" and automatic):
        raise ValueError(f"{where} requires a bounded scalar joint, found limited={limited!r}")
    bounds = _numbers(joint, "range", 2, where)
    if kind == "hinge" and degrees:
        bounds = np.deg2rad(bounds)
    if bounds[0] >= bounds[1]:
        raise ValueError(f"{where} needs an increasing range")
    axis = unit_axis(_numbers(joint, "axis", 3, where, (0, 0, 1)), where)
    return (
        "revolute" if kind == "hinge" else "prismatic",
        axis,
        (float(bounds[0]), float(bounds[1]), None, None),
    )


def _fromto(element, kind, where):
    """Read a cylinder/capsule segment in body coordinates, in metres."""
    if kind not in {"cylinder", "capsule"}:
        raise ValueError(f"{where} supports fromto only on a cylinder or capsule")
    if any(key in element.attrib for key in ("pos", "quat", "euler")):
        raise ValueError(f"{where} cannot combine fromto with an explicit pose")
    start, end = _numbers(element, "fromto", 6, where).reshape(2, 3)
    # Work with half vectors before adding/subtracting distant end coordinates.
    center = 0.5 * start + 0.5 * end
    half_axis = 0.5 * start - 0.5 * end
    with np.errstate(over="ignore"):
        length = np.hypot.reduce(half_axis)
    if not np.isfinite(length) or length == 0:
        raise ValueError(f"{where} fromto needs a finite positive segment length")
    z = half_axis / length
    lateral = np.hypot(z[0], z[1])
    if lateral == 0:
        rotation = np.array([0.0, 0.0, 0.0, 1.0]) if z[2] > 0 else np.array([1.0, 0.0, 0.0, 0.0])
    else:
        half_angle = 0.5 * np.arctan2(lateral, z[2])
        rotation = np.array(
            [
                -z[1] / lateral * np.sin(half_angle),
                z[0] / lateral * np.sin(half_angle),
                0.0,
                np.cos(half_angle),
            ]
        )
    # The segment supplies the axial dimension even if an optional size[1] was authored.
    count = len(element.get("size", "").split())
    if count not in (1, 2):
        raise ValueError(f"{where} fromto needs size containing a radius and optional half-length")
    supplied = _numbers(element, "size", count, where)
    if np.any(supplied <= 0):
        raise ValueError(f"{where} needs positive size values")
    return center, rotation, np.array([supplied[0], length])


def _colliders(body, link, index, degrees, sequence, named_geoms, resolve_mesh):
    colliders = []
    names = set()
    for ordinal, geom in enumerate(body.findall("geom")):
        name = geom.get("name", f"geom_{ordinal}").strip()
        where = f"body {link!r} geom {name!r}"
        if not name or name in names:
            raise ValueError(f"{where} has an empty or duplicate name")
        names.add(name)
        if "name" in geom.attrib:
            if name in named_geoms:
                raise ValueError(f"duplicate MJCF geom name {name!r}")
            named_geoms.add(name)
        try:
            masks = int(geom.get("contype", "1")), int(geom.get("conaffinity", "1"))
        except ValueError as exc:
            raise ValueError(f"{where} has invalid contype/conaffinity") from exc
        if masks == (0, 0):
            # Every moving body has explicit inertia, so this render-only geom
            # contributes neither contact nor mass to the imported robot.
            continue
        if masks != (1, 1):
            raise ValueError(f"{where} has unsupported contype/conaffinity masks {masks}")
        _attributes(
            geom,
            {
                "name",
                "pos",
                "quat",
                "euler",
                "fromto",
                "mesh",
                "type",
                "size",
                "contype",
                "conaffinity",
                "rgba",
            },
            where,
        )
        _children(geom, set(), where)
        kind = geom.get("type", "mesh" if "mesh" in geom.attrib else "sphere")
        if "mesh" in geom.attrib and kind != "mesh":
            raise ValueError(f"{where} fitting a primitive to a mesh is unsupported")
        if kind == "mesh":
            if "fromto" in geom.attrib:
                raise ValueError(f"{where} mesh cannot use fromto")
            if "size" in geom.attrib:
                count = len(geom.get("size").split())
                if not 1 <= count <= 3:
                    raise ValueError(f"{where} mesh size needs one to three finite numbers")
                _numbers(geom, "size", count, where)
            position, rotation = _pose(geom, where, degrees, sequence)
            size = resolve_mesh(_name(geom, "mesh", where), where)
            colliders.append(
                make_collider(name, link, index, ordinal, position, rotation, kind, size)
            )
            continue
        counts = {"sphere": 1, "box": 3, "cylinder": 2, "capsule": 2, "ellipsoid": 3}
        if kind not in counts:
            raise ValueError(f"{where} uses unsupported primitive {kind!r}")
        if "fromto" in geom.attrib:
            position, rotation, size = _fromto(geom, kind, where)
        else:
            size = _numbers(geom, "size", counts[kind], where)
            position, rotation = _pose(geom, where, degrees, sequence)
        colliders.append(make_collider(name, link, index, ordinal, position, rotation, kind, size))
    return colliders


def load_mjcf(path: str | Path, *, gravity=None) -> RobotAsset:
    """Load an explicitly inertial MJCF robot with a named, welded root.

    Each moving body has one finite hinge or slide with a zero local pivot and
    reference coordinate. Positions and slides are metres. Hinge limits and
    Euler angles follow compiler angle and eulerseq. Quaternions use MJCF WXYZ.
    Colliders support spheres, boxes, cylinders, capsules, ellipsoids and named local OBJ meshes.
    A cylinder or capsule may specify its axis with fromto. Geoms with
    contype=conaffinity=0 supply no colliders. Immediate moving
    parent/child pairs are excluded as in MJCF; the world-welded root is exempt.

    Missing actuator force and speed limits are returned as None. Unsupported
    semantic elements (e.g. defaults, free joints, actuators and primitive fitting)
    raise ValueError. The gravity keyword overrides the authored option vector.
    """
    try:
        robot = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid MJCF XML in {path}: {exc}") from exc
    if robot.tag != "mujoco":
        raise ValueError("MJCF root element must be <mujoco>")
    _attributes(robot, {"model"}, "mujoco")
    _children(robot, {"compiler", "option", "worldbody", "asset"}, "mujoco")
    name = _name(robot, "model", "mujoco") if "model" in robot.attrib else "MuJoCo Model"
    degrees, sequence, automatic = _compiler(robot)
    resolve_mesh = mjcf_mesh_assets(robot, path)
    option = _child(robot, "option", "mujoco", required=False)
    default_gravity = (0, 0, -9.81)
    if option is not None:
        _attributes(option, {"gravity"}, "option")
        _children(option, set(), "option")
        default_gravity = _numbers(option, "gravity", 3, "option", default_gravity)
    worldbody = _child(robot, "worldbody", "mujoco")
    _attributes(worldbody, set(), "worldbody")
    _children(worldbody, {"body"}, "worldbody")
    base = _child(worldbody, "body", "worldbody")
    root = _name(base, "name", "root body")
    root_xyz, root_quat = _pose(base, f"root body {root!r}", degrees, sequence)
    root_rotation = quaternion_matrix(root_quat[None, :])[0]
    order, parents, joints, joint_data, bodies = [], [], [], [], []
    colliders, excluded = [], []
    shapes_by_link = {}
    link_names, joint_names, named_geoms = set(), set(), set()
    queue = deque([(base, -1)])
    while queue:
        body, parent = queue.popleft()
        link = _name(body, "name", "body")
        where = f"body {link!r}"
        if link in link_names:
            raise ValueError(f"duplicate {where}")
        link_names.add(link)
        _attributes(body, {"name", "pos", "quat", "euler"}, where)
        _children(body, {"geom", "inertial", "body", "joint"}, where)
        base_link = body is base
        inertial = _body_inertia(body, link, degrees, sequence, root=base_link)
        if base_link:
            if body.find("joint") is not None:
                raise ValueError(f"fixed root {where} must not have a joint")
            index = -1
        else:
            joint = _child(body, "joint", where)
            joint_name = _name(joint, "name", where + " joint")
            if joint_name in joint_names:
                raise ValueError(f"duplicate joint {joint_name!r}")
            joint_names.add(joint_name)
            kind, axis, limits = _joint(joint, joint_name, degrees, automatic)
            position, rotation = _pose(body, where, degrees, sequence)
            if parent == -1:
                # RobotModel attaches root joints to identity world. Compose the
                # welded MJCF base transform only for direct children of base.
                position = root_xyz + root_rotation @ position
                rotation = _multiply(root_quat, rotation)
            index = len(order)
            order.append(link)
            parents.append(parent)
            joints.append(joint_name)
            joint_data.append((kind, position, rotation, axis, limits))
            bodies.append(inertial)
        local_shapes = _colliders(body, link, index, degrees, sequence, named_geoms, resolve_mesh)
        if parent >= 0:
            excluded.extend(
                (p.shape.key, c.shape.key) for p in shapes_by_link[parent] for c in local_shapes
            )
        shapes_by_link[index] = local_shapes
        colliders.extend(local_shapes)
        queue.extend((child, index) for child in body.findall("body"))
    if not order:
        raise ValueError("MJCF RobotModel requires at least one moving link")
    return assemble_robot(
        name,
        root,
        order,
        parents,
        joints,
        joint_data,
        bodies,
        colliders,
        default_gravity if gravity is None else gravity,
        root_xyz=root_xyz,
        root_xyzw=root_quat,
        excluded=excluded,
    )
