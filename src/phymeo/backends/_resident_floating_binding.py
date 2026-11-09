"""Original finite material attached to a real floating-tree body frame.

Bindings do not evaluate current HOST kinematics. Static q=0 geometry is used
only for registration; every query consumes the accepted device body frames.
"""

from dataclasses import dataclass, field, replace
from numbers import Integral

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionMotion, CollisionShape
from ..dynamics.rotation import quaternion_matrix
from ._resident_floating_tree import FloatingTreeWorldSpec, floating_model_token
from ._resident_tree_binding import _outward_positive, _points, _token
from ._resident_tree_world import _owned
from ._world_step_layout import as_fp32


@dataclass(frozen=True, eq=False)
class ResidentFloatingTreeBinding:
    """An explicit body-local installation; from_asset additionally proves source ownership."""

    key: CollisionKey
    world: int
    group_index: int
    local_points_m: np.ndarray
    topology: tuple | None = field(kw_only=True)
    _source_shape: CollisionShape | None = field(default=None, init=False, repr=False)
    _model_token: tuple | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.key, CollisionKey):
            raise TypeError("floating binding requires a CollisionKey")
        if (
            isinstance(self.world, (bool, np.bool_))
            or not isinstance(self.world, Integral)
            or int(self.world) != self.key.world
        ):
            raise ValueError("floating binding world must match its key")
        if (
            isinstance(self.group_index, (bool, np.bool_))
            or not isinstance(self.group_index, Integral)
            or not 0 <= self.group_index <= 32
        ):
            raise ValueError("floating binding group must be a body index in [0,33)")
        object.__setattr__(self, "world", int(self.world))
        object.__setattr__(self, "group_index", int(self.group_index))
        object.__setattr__(self, "local_points_m", _points(self.local_points_m))
        if self.topology is not None:
            if not isinstance(self.topology, tuple):
                raise ValueError("floating binding topology must be a checkpoint tuple")
            object.__setattr__(self, "topology", _token(self.topology))

    @classmethod
    def from_asset(cls, asset, source_key, *, key, world):
        from ..builders._floating_fixed_asset import FoldedFloatingRobotAsset
        from ..builders.floating_urdf import FloatingRobotAsset

        if not isinstance(asset, FloatingRobotAsset):
            raise TypeError("floating source must be an ordinary FloatingRobotAsset")
        if not isinstance(source_key, CollisionKey):
            raise TypeError("floating source key must be a CollisionKey")
        colliders = [c for c in asset.collisions if c.shape.key == source_key]
        shapes = [s for s in asset.collision_model.shapes if s.key == source_key]
        if len(colliders) != 1 or len(shapes) != 1 or colliders[0].shape != shapes[0]:
            raise ValueError(
                "floating source must identify exactly one complete registered collider"
            )
        collider = colliders[0]
        if collider.shape.kind not in ("sphere", "capsule", "hull"):
            raise ValueError("floating source supports finite sphere, capsule or hull only")
        group = collider.link_index + 1
        points = np.asarray(collider.local_vertices, dtype=float)
        if isinstance(asset, FoldedFloatingRobotAsset):
            site = asset.sites.get(collider.link_name)
            if site is None or site.body_index != group:
                raise ValueError("floating authored site and collider owner disagree")
            rotation = quaternion_matrix(np.asarray([site.local_orientation_xyzw], dtype=float))[0]
            points = np.asarray(site.local_position_m, dtype=float) + points @ rotation.T
        if not 0 <= group < asset.model.body_count:
            raise ValueError("floating source owner is outside the actual body list")
        result = cls(key, world, group, points, topology=None)
        object.__setattr__(result, "_source_shape", collider.shape)
        object.__setattr__(result, "_model_token", floating_model_token(asset.model))
        return result


@dataclass(frozen=True, eq=False)
class FloatingBindingCompiled:
    group_index: int
    zero_motion: CollisionMotion
    points_m: np.ndarray
    geometry: np.ndarray


def _zero_pose(spec, group, points):
    """Static authored model at identity root orientation and all child q=0."""
    positions = np.zeros((spec.b, 3))
    frames = np.repeat(np.eye(3)[None], spec.b, axis=0)
    token = spec.model_token[6]
    offsets = np.zeros((spec.b, 3))
    if spec.n:
        parents = token[0]
        offsets[1:] = np.asarray(token[2], dtype=float)
        rotations = quaternion_matrix(np.asarray(token[3], dtype=float))
        for i in range(spec.n):
            body, parent = i + 1, parents[i] + 1
            positions[body] = positions[parent] + frames[parent] @ offsets[body]
            frames[body] = frames[parent] @ rotations[i]
    world = spec.anchor + positions[group] + points @ frames[group].T
    reach, depth, body = 0.0, 0, group
    while body >= 0:
        reach += float(np.sum(abs(offsets[body])))
        depth += 1
        body = int(spec.parents[body])
    if not np.isfinite(world).all() or not np.isfinite(reach):
        raise ValueError("floating zero pose or path cannot form finite geometry")
    return CollisionMotion(world), reach, depth


def compile_floating_tree_binding(shape, binding, spec):
    """Pack original points[V,3] and radius/local/path/depth geometry[4]."""
    if (
        not isinstance(shape, CollisionShape)
        or not isinstance(binding, ResidentFloatingTreeBinding)
        or not isinstance(spec, FloatingTreeWorldSpec)
    ):
        raise TypeError("floating binding compilation needs shape, binding and owned spec")
    if shape.key != binding.key or shape.key.world != binding.world:
        raise ValueError("floating shape key/world differs from its binding")
    if shape.kind not in ("sphere", "capsule", "hull") or not shape.rigid:
        raise ValueError("floating resident shape must be a finite rigid sphere, capsule or hull")
    if binding.group_index >= spec.b:
        raise ValueError("floating group is outside this physical model")
    if len(shape.vertex_ids) != len(binding.local_points_m):
        raise ValueError("floating binding must retain every original material vertex")
    if binding._source_shape is not None:
        if binding._model_token != spec.model_token:
            raise ValueError("floating source model differs from the owned physical model")
        if shape != replace(binding._source_shape, key=binding.key):
            raise ValueError("floating source shape IDs, skin, revision or content changed")
    elif binding.topology is None or binding.topology != spec.topology:
        raise ValueError("floating topology differs from its owned physical model")
    points = binding.local_points_m
    motion, reach, depth = _zero_pose(spec, binding.group_index, points)
    local_reach = float(np.max(np.sum(abs(points), axis=1)))
    geometry = _owned(
        [
            _outward_positive(shape.radius, "floating skin"),
            _outward_positive(local_reach, "floating local reach"),
            _outward_positive(reach, "floating path reach"),
            np.float32(depth),
        ]
    )
    return FloatingBindingCompiled(
        binding.group_index,
        motion,
        _owned(as_fp32(points, "floating body-local material", points.shape)),
        geometry,
    )
