"""Opt-in fixed URDF tooling on one physical free root and scalar children.

This compiler only assembles rigid groups and authored frames. It does not
integrate, constrain, weld, or advance a scene. Root axes keep their existing
public meaning; only moving child groups may use a new principal-axis frame.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionModel, CollisionMotion
from ..core.rigid import RigidBodyModel
from ..dynamics.floating_robot import FloatingRobotModel, FloatingRobotState, floating_kinematics
from ..dynamics.rotation import matrix_quaternion
from ._robot_asset import assemble_robot, principal_inertia
from ._robot_xml import _attributes, _children
from ._urdf_fixed import _mass_component, _origin_pose, _principal
from ._urdf_frames import _rotation
from .floating_urdf import FloatingRobotAsset
from .urdf import _joint_data, _link_collisions


def _vector(value, width, name):
    raw = np.asarray(value)
    if raw.shape != (width,) or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real {width}-vector")
    out = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return tuple(float(item) for item in out)


@dataclass(frozen=True)
class FloatingFixedSite:
    """An authored link frame relative to a real body origin, including root 0."""

    body_index: int
    local_position_m: tuple = (0.0, 0.0, 0.0)
    local_orientation_xyzw: tuple = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self):
        index = self.body_index
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral) or index < 0:
            raise ValueError("floating site body_index must be a nonnegative integer")
        position = _vector(self.local_position_m, 3, "floating site position")
        rotation = _vector(self.local_orientation_xyzw, 4, "floating site orientation")
        if abs(np.hypot.reduce(rotation) - 1.0) > 1e-10:
            raise ValueError("floating site orientation must be a unit XYZW quaternion")
        object.__setattr__(self, "body_index", int(index))
        object.__setattr__(self, "local_position_m", position)
        object.__setattr__(self, "local_orientation_xyzw", rotation)


@dataclass(frozen=True, eq=False, kw_only=True)
class FoldedFloatingRobotAsset(FloatingRobotAsset):
    """Keep authored collision coordinates alongside the real grouped model.

    sites use physical body indices (0 root, i+1 child). Collision link_index
    remains the asset convention (-1 root, i child); local_vertices are authored
    material coordinates, not silently converted geometry.
    """

    sites: Mapping[str, FloatingFixedSite]

    def __post_init__(self):
        if not isinstance(self.model, FloatingRobotModel):
            raise TypeError("folded floating asset needs a FloatingRobotModel")
        if not isinstance(self.sites, Mapping):
            raise TypeError("floating authored sites must be a mapping")
        child_count = self.model.body_count - 1
        if len(self.link_names) != child_count or len(self.joint_names) != child_count:
            raise ValueError("floating body and joint metadata counts must agree")
        sites = {}
        for name, site in self.sites.items():
            if not isinstance(name, str) or not name:
                raise ValueError("floating authored site names must be nonempty strings")
            if not isinstance(site, FloatingFixedSite):
                raise TypeError("floating authored sites must be FloatingFixedSite values")
            if site.body_index >= self.model.body_count:
                raise ValueError("floating authored site owner is outside the body list")
            sites[name] = FloatingFixedSite(
                site.body_index, site.local_position_m, site.local_orientation_xyzw
            )
        for body, name in enumerate((self.root_link, *self.link_names)):
            if name not in sites or sites[name].body_index != body:
                raise ValueError("floating root/child authored frame has the wrong physical owner")
        root_site = sites[self.root_link]
        if (
            any(root_site.local_position_m)
            or any(root_site.local_orientation_xyzw[:3])
        ):
            raise ValueError("floating root frame must retain its original origin and axes")
        collisions = tuple(self.collisions)
        keys = tuple(col.shape.key for col in collisions)
        model_shapes = {shape.key: shape for shape in self.collision_model.shapes}
        if len(set(keys)) != len(keys) or set(keys) != set(model_shapes):
            raise ValueError("floating collider keys must be unique and match the collision model")
        for col in collisions:
            site = sites.get(col.link_name)
            if site is None or site.body_index != col.link_index + 1:
                raise ValueError("floating collider and authored material site have different owners")
            if col.shape.key.world != 0 or col.shape.key.body != site.body_index:
                raise ValueError("floating asset shape key must identify its physical body group")
            if model_shapes.get(col.shape.key) != col.shape:
                raise ValueError("floating collision model and authored shape disagree")
        object.__setattr__(self, "collisions", collisions)
        object.__setattr__(self, "sites", MappingProxyType(sites))

    def authored_to_model_points(self, name, points_m):
        """Map original material to its group's local origin, never to world."""
        site = self.sites[name]
        raw = np.asarray(points_m)
        if (
            raw.ndim not in (1, 2)
            or raw.shape[-1] != 3
            or raw.dtype.kind not in "fiu"
            or not np.isfinite(raw).all()
        ):
            raise ValueError("authored material points must be finite real 3-vectors")
        points = np.asarray(raw, dtype=np.float64)
        with np.errstate(over="raise", invalid="raise"):
            try:
                result = np.asarray(site.local_position_m) + points @ _rotation(
                    site.local_orientation_xyzw
                ).T
            except FloatingPointError as exc:
                raise ValueError("authored material transform exceeds FP64 range") from exc
        if not np.isfinite(result).all():
            raise ValueError("authored material transform exceeds FP64 range")
        return result

    def _geometry(self, q):
        state = FloatingRobotState(q, np.zeros(6 + len(self.joint_names)))
        return floating_kinematics(self.model, state)

    def _pose(self, name, geometry):
        site = self.sites[name]
        body = site.body_index
        position = geometry.position[body] + geometry.rotation[body] @ site.local_position_m
        rotation = geometry.rotation[body] @ _rotation(site.local_orientation_xyzw)
        return position, rotation

    def authored_link_pose(self, name, q):
        """Read the actual world pose of the original named link."""
        return self._pose(name, self._geometry(q))

    def collision_motion(self, q, q_end=None):
        """Read one actual pose. No chord approximates a floating interval."""
        if q_end is not None:
            raise ValueError("floating asset interval collision motion is not implemented")
        geometry = self._geometry(q)
        poses = {col.link_name: self._pose(col.link_name, geometry) for col in self.collisions}
        result = {}
        for col in self.collisions:
            p, r = poses[col.link_name]
            points = p + np.asarray(col.local_vertices) @ r.T
            orientation = matrix_quaternion(r) if col.shape.curved is not None else None
            result[col.shape.key] = CollisionMotion(points, body_orientation_xyzw=orientation)
        return result


def _root_group(parts, root):
    """Aggregate about COM in original root axes; reject genuine products."""
    if not parts:
        raise ValueError(f"{root}: root rigid group has no positive mass and inertia")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mass = sum(m for m, _, _ in parts)
            center = sum(m * point for m, point, _ in parts) / mass
            tensor = np.zeros((3, 3))
            for m, point, inertia in parts:
                delta = point - center
                tensor += inertia + m * (
                    (delta @ delta) * np.eye(3) - np.outer(delta, delta)
                )
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{root}: root group mass or inertia exceeds FP64 range") from exc
    if not np.isfinite(mass) or not np.isfinite(center).all() or not np.isfinite(tensor).all():
        raise ValueError(f"{root}: root group mass or inertia exceeds FP64 range")
    scale = np.max(np.abs(tensor))
    if scale <= 0:
        raise ValueError(f"{root}: root group needs positive inertia")
    normalized = tensor / scale
    # Same relative 32-epsilon rotation budget as principal_inertia. Here the
    # rotation and parallel-axis sums have already been applied per component.
    products = normalized - np.diag(normalized.diagonal())
    if np.max(np.abs(products)) > 32 * np.finfo(float).eps:
        raise ValueError(f"{root}: non-diagonal root group inertia is not supported")
    return principal_inertia(
        mass, center, np.array([0.0, 0.0, 0.0, 1.0]),
        np.diag(tensor.diagonal()), f"root group {root!r}",
    )


def folded_floating_robot(name, root, links, order, incoming, gravity, path):
    """Compile validated URDF nodes into physical groups without dropping root mass."""
    frames = {root: (-1, np.zeros(3), np.eye(3))}
    moving, parents, joints, data = [], [], [], []
    components = defaultdict(list)
    for child in order:
        parent, joint_name, joint = incoming[child]
        owner, base, rotation = frames[parent]
        if joint.find("mimic") is not None:
            raise ValueError(f"joint {joint_name!r}: floating mimic is not implemented")
        if joint.get("type") == "fixed":
            where = f"fixed joint {joint_name!r}"
            _attributes(joint, {"name", "type"}, where)
            _children(joint, {"parent", "child", "origin"}, where)
            offset, basis = _origin_pose(joint, where)
            frame = owner, base + rotation @ offset, rotation @ basis
        else:
            kind, offset, quat, axis, bounds = _joint_data(joint, joint_name)
            parents.append(owner)
            joints.append(joint_name)
            moving.append(child)
            data.append((kind, base + rotation @ offset, rotation @ _rotation(quat), axis, bounds))
            frame = len(moving) - 1, np.zeros(3), np.eye(3)
        frames[child] = frame
    for link, (owner, position, rotation) in frames.items():
        value = _mass_component(links[link], link, position, rotation, allow_massless=True)
        if value is not None:
            components[owner].append(value)
    root_mass, root_com, root_inertia = _root_group(components[-1], root)
    axes, bodies = [], []
    for owner, link in enumerate(moving):
        frame, body = _principal(components[owner], link)
        axes.append(frame)
        bodies.append(body)
    native = []
    for i, (kind, position, rotation, axis, bounds) in enumerate(data):
        parent = parents[i]
        before = np.eye(3) if parent < 0 else axes[parent].T
        native.append(
            (kind, before @ position, matrix_quaternion(before @ rotation @ axes[i]),
             axes[i].T @ axis, bounds)
        )
    sites, colliders, ordinal = {}, [], defaultdict(int)
    for link, (owner, position, rotation) in frames.items():
        basis = np.eye(3) if owner < 0 else axes[owner].T
        sites[link] = FloatingFixedSite(
            owner + 1, basis @ position, matrix_quaternion(basis @ rotation)
        )
        for col in _link_collisions(links[link], link, owner, path):
            key = CollisionKey(0, owner + 1, ordinal[owner])
            ordinal[owner] += 1
            colliders.append(replace(col, shape=replace(col.shape, key=key)))
    child = None
    if moving:
        child = assemble_robot(
            name, root, moving, parents, joints, native, bodies, colliders, gravity
        )
    model = FloatingRobotModel(
        RigidBodyModel(
            np.array([1 / root_mass]), root_inertia[None, :], np.asarray(gravity, dtype=float)
        ),
        root_com,
        None if child is None else child.model,
    )
    return FoldedFloatingRobotAsset(
        name=name,
        root_link=root,
        model=model,
        link_names=tuple(moving),
        joint_names=tuple(joints),
        lower=() if child is None else child.lower,
        upper=() if child is None else child.upper,
        effort_limit=() if child is None else child.effort_limit,
        velocity_limit=() if child is None else child.velocity_limit,
        collisions=tuple(colliders),
        collision_model=CollisionModel(tuple(col.shape for col in colliders)),
        sites=sites,
    )
