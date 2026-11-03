"""Immutable robot asset geometry shared by the local file importers."""

from dataclasses import dataclass
from itertools import product

import numpy as np

from ..constraints.collision_shapes import (
    CollisionKey,
    CollisionModel,
    CollisionMotion,
    CollisionShape,
)
from ..constraints.curved_solids import CurvedSolid
from ..core.robot import RobotModel
from ..dynamics.robot_kinematics import forward_kinematics
from ..dynamics.rotation import matrix_quaternion, quaternion_matrix


@dataclass(frozen=True)
class LinkCollision:
    """A named collision element. Local vertices are expressed in the authored link frame."""

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

    Limits retain physical units (rad or m, N·m or N, rad/s or m/s). They are asset
    metadata. An absent effort or velocity bound is None. Finite bounds may
    be passed to a controller or physical stop with explicit gains.
    The collision model assigns body 0 to the base and i+1 to moving link i.
    """

    name: str
    root_link: str
    model: RobotModel
    link_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    effort_limit: tuple[float | None, ...]
    velocity_limit: tuple[float | None, ...]
    collisions: tuple[LinkCollision, ...]
    collision_model: CollisionModel
    root_xyz: tuple[float, ...] = (0.0, 0.0, 0.0)
    root_xyzw: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)

    def collision_motion(self, q, q_end=None):
        """Build static shape poses or rigid paths over the given straight joint-coordinate interval."""
        geometry = forward_kinematics(self.model, q)
        root_rotation = quaternion_matrix(np.array([self.root_xyzw]))[0]
        paths = {}
        for collider in self.collisions:
            points = np.asarray(collider.local_vertices)
            link = collider.link_index
            if link >= 0:
                points = geometry.position[link] + points @ geometry.rotation[link].T
            else:
                points = np.asarray(self.root_xyz) + points @ root_rotation.T
            orientation = None
            if collider.shape.curved is not None:
                orientation = (
                    matrix_quaternion(geometry.rotation[link]) if link >= 0 else self.root_xyzw
                )
            paths[collider.shape.key] = CollisionMotion(points, body_orientation_xyzw=orientation)
        if q_end is not None:
            from ._robot_joint_motion import joint_interval_motion

            return joint_interval_motion(self, q, q_end, geometry, paths)
        return paths


def principal_inertia(mass, com, orientation, tensor, where, *, root=False):
    """Express an authored COM inertia in the link frame without losing products."""
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


def unit_axis(axis, where):
    scale = np.max(np.abs(axis))
    if scale == 0:
        raise ValueError(f"{where} has a zero axis")
    axis = axis / scale
    return axis / np.hypot.reduce(axis)


def make_collider(name, link_name, link_index, ordinal, position, orientation, kind, size):
    """Register a finite primitive in the authored link frame, with SI half-lengths."""
    where = f"link {link_name!r} collision {name!r}"
    if kind != "mesh" and np.any(size <= 0):
        raise ValueError(
            f"{where} requires positive {kind} {'radius' if kind == 'sphere' else 'size'}"
        )
    curved = None
    triangles = ()
    if kind == "mesh":
        native_kind, radius = "mesh", 0.0
        triangles = size.triangles
        vertices = (
            position + np.asarray(size.vertices) @ quaternion_matrix(orientation[None, :])[0].T
        )
    elif kind == "sphere":
        native_kind, radius = "sphere", float(size[0])
        vertices = position[None, :]
    elif kind in ("cylinder", "ellipsoid"):
        native_kind, radius = kind, 0.0
        curved = CurvedSolid(kind, tuple(size), tuple(orientation))
        vertices = position[None, :]
    elif kind == "capsule":
        if len(size) != 2:
            raise ValueError(f"{where} requires a radius and an axis half-length")
        native_kind, radius = kind, float(size[0])
        half_axis = size[1] * quaternion_matrix(orientation[None, :])[0, :, 2]
        vertices = np.array([position - half_axis, position + half_axis])
    elif kind == "box":
        native_kind, radius = "hull", 0.0
        corners = np.array(list(product((-1.0, 1.0), repeat=3))) * size
        vertices = position + corners @ quaternion_matrix(orientation[None, :])[0].T
    else:
        raise ValueError(f"{where} has unsupported collision shape {kind!r}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{where} has unrepresentable local vertices")
    key = CollisionKey(0, link_index + 1, ordinal)
    try:
        shape = CollisionShape(
            key,
            native_kind,
            tuple(range(len(vertices))),
            radius,
            triangles=triangles,
            curved=curved,
        )
        if kind == "mesh":
            from ..geometry._closed_mesh import ClosedMesh

            ClosedMesh(shape, CollisionMotion(vertices))
    except ValueError as error:
        if kind != "mesh":
            raise
        raise ValueError(f"{where}: {error}") from error
    return LinkCollision(
        name,
        link_name,
        link_index,
        tuple(position),
        tuple(orientation),
        shape,
        tuple(tuple(row) for row in vertices),
    )


def assemble_robot(
    name,
    root,
    order,
    parents,
    joints,
    joint_data,
    bodies,
    colliders,
    gravity,
    *,
    root_xyz=(0.0, 0.0, 0.0),
    root_xyzw=(0.0, 0.0, 0.0, 1.0),
    excluded=(),
):
    kinds, positions, orientations, axes, limits = zip(*joint_data, strict=True)
    masses, com, inertia = zip(*bodies, strict=True)
    lower, upper, effort, velocity = zip(*limits, strict=True)
    model = RobotModel(
        parents=np.array(parents),
        joint_types=kinds,
        joint_position=np.array(positions),
        joint_orientation=np.array(orientations),
        joint_axis=np.array(axes),
        mass=np.array(masses),
        com=np.array(com),
        inertia_diagonal=np.array(inertia),
        gravity=np.asarray(gravity, dtype=float),
    )
    return RobotAsset(
        name,
        root,
        model,
        tuple(order),
        tuple(joints),
        lower,
        upper,
        effort,
        velocity,
        tuple(colliders),
        CollisionModel(tuple(c.shape for c in colliders), tuple(excluded)),
        tuple(root_xyz),
        tuple(root_xyzw),
    )
