"""Combine local fixed URDF links into the nearest dynamic parent, preserving their authored frames."""

from collections import defaultdict
from dataclasses import replace
from types import MappingProxyType

import numpy as np

from ..constraints.collision_shapes import CollisionKey
from ..dynamics.rotation import matrix_quaternion
from ..sensors.robot_frame import RobotSite
from ._robot_asset import assemble_robot
from ._robot_xml import _attributes, _child, _children, _number
from ._urdf_frames import FoldedURDFAsset, _rotation
from .urdf import _joint_data, _link_collisions, _origin


def _origin_pose(element, where):
    xyz, xyzw = _origin(element, where)
    return xyz, _rotation(xyzw)


def _mass_component(link, name, position, orientation, *, allow_massless=False):
    """One mass, COM position and COM inertia in the current rigid group's authored frame."""
    inertial = _child(link, "inertial", name, required=False)
    if inertial is None:
        return None
    where = f"link {name!r} inertial"
    _attributes(inertial, set(), where)
    _children(inertial, {"mass", "origin", "inertia"}, where)
    com, rotation = _origin_pose(inertial, where)
    entry = _child(inertial, "mass", where)
    _attributes(entry, {"value"}, where)
    _children(entry, set(), where)
    mass = _number(entry, "value", where)
    entry = _child(inertial, "inertia", where)
    fields = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    _attributes(entry, set(fields), where)
    _children(entry, set(), where)
    xx, xy, xz, yy, yz, zz = [_number(entry, k, where) for k in fields]
    tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    if allow_massless and mass == 0 and not np.any(tensor):
        return None
    scale = np.max(np.abs(tensor))
    if mass <= 0 or scale == 0:
        raise ValueError(f"{name}: invalid mass or COM inertia")
    moments = np.linalg.eigvalsh(tensor / scale)
    if moments[0] <= 0 or moments[2] > moments[:2].sum() * (1 + 2e-14):
        raise ValueError(f"{name}: invalid mass or COM inertia")
    frame = orientation @ rotation
    return mass, position + orientation @ com, frame @ tensor @ frame.T


def _principal(parts, name="rigid group"):
    if not parts:
        raise ValueError(f"{name}: moving rigid group has no positive mass and inertia")
    mass = sum(m for m, _, _ in parts)
    center = sum(m * p for m, p, _ in parts) / mass
    tensor = np.zeros((3, 3))
    for m, p, inertia in parts:
        d = p - center
        tensor += inertia + m * ((d @ d) * np.eye(3) - np.outer(d, d))
    if not np.isfinite(mass) or not np.isfinite(tensor).all() or not np.isfinite(center).all():
        raise ValueError(f"{name}: rigid group mass or inertia is outside FP64 range")
    if np.any(tensor - np.diag(tensor.diagonal())):
        moments, axes = np.linalg.eigh(tensor)
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1
    else:
        moments, axes = tensor.diagonal(), np.eye(3)
    return axes, (mass, axes.T @ center, moments)


def folded_robot(name, root, links, order, incoming, gravity, path):
    # A frame stores its owner DOF and the authored link-to-owner transform.
    frames = {root: (-1, np.zeros(3), np.eye(3))}
    moving, parents, joints, data, components = [], [], [], [], []
    for child in order:
        parent, jname, joint = incoming[child]
        owner, base, rotation = frames[parent]
        if joint.get("type") == "fixed":
            where = f"fixed joint {jname!r}"
            _attributes(joint, {"name", "type"}, where)
            _children(joint, {"parent", "child", "origin"}, where)
            offset, basis = _origin_pose(joint, where)
            frame = owner, base + rotation @ offset, rotation @ basis
        else:
            kind, offset, quat, axis, bounds = _joint_data(joint, jname)
            parents.append(owner)
            joints.append(jname)
            moving.append(child)
            data.append((kind, base + rotation @ offset, rotation @ _rotation(quat), axis, bounds))
            components.append([])
            frame = len(moving) - 1, np.zeros(3), np.eye(3)
        frames[child] = frame
    if not moving:
        raise ValueError("a fixed URDF tree needs at least one active scalar joint")
    for link, (owner, p, R) in frames.items():
        component = _mass_component(links[link], link, p, R, allow_massless=True)
        if owner >= 0 and component is not None:
            components[owner].append(component)
    axes, bodies = zip(
        *(_principal(parts, link) for link, parts in zip(moving, components, strict=True)),
        strict=True,
    )
    native = []
    for i, (kind, position, R, axis, bounds) in enumerate(data):
        parent = parents[i]
        before = np.eye(3) if parent < 0 else axes[parent].T
        native.append(
            (
                kind,
                before @ position,
                matrix_quaternion(before @ R @ axes[i]),
                axes[i].T @ axis,
                bounds,
            )
        )
    sites, colliders, ordinal = {}, [], defaultdict(int)
    for link, (owner, p, R) in frames.items():
        basis = np.eye(3) if owner < 0 else axes[owner].T
        sites[link] = RobotSite(
            None if owner < 0 else owner, basis @ p, matrix_quaternion(basis @ R)
        )
        for col in _link_collisions(links[link], link, owner, path):
            key = CollisionKey(0, owner + 1, ordinal[owner])
            ordinal[owner] += 1
            colliders.append(replace(col, shape=replace(col.shape, key=key)))
    asset = assemble_robot(name, root, moving, parents, joints, native, bodies, colliders, gravity)
    return FoldedURDFAsset(**vars(asset), sites=MappingProxyType(sites))
