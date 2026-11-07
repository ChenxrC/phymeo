"""Owned material bindings for real accepted fixed-tree link frames.

Direct bindings declare group-local installation. ``from_asset`` also retains
original model/shape identity and converts folded authored points exactly once.
No current-state HOST FK is used to produce a query's device bounds.
"""

from dataclasses import dataclass, field, replace
from numbers import Integral

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionMotion, CollisionShape
from ..dynamics.rotation import quaternion_matrix
from ._resident_tree_world import MAX_TREE_JOINTS, TreeWorldSpec, _owned, tree_model_token
from ._world_step_layout import as_fp32


def _points(value):
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != 3 or not len(raw) or raw.dtype.kind not in "iuf":
        raise ValueError("tree local material points must be a nonempty real [V,3] array")
    with np.errstate(over="ignore", invalid="ignore"):
        points = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(points).all():
        raise ValueError("tree local material points must be finite")
    return _owned(points, np.float64)


def _token(value):
    if isinstance(value, tuple):
        return tuple(_token(item) for item in value)
    if isinstance(value, str):
        return value
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return float(value)
    raise ValueError("tree topology must be an immutable finite model declaration")


@dataclass(frozen=True, eq=False)
class ResidentTreeBinding:
    """One original shape in a real moving group's local principal frame.

    ``topology`` is the complete checkpoint topology for explicit installation;
    from_asset supplies a separate original FP64 model and source-shape proof.
    No group is inferred from a collision body's integer ID.
    """
    key: CollisionKey
    world: int
    group_index: int
    local_points_m: np.ndarray
    topology: tuple | None = field(kw_only=True)
    _source_shape: CollisionShape | None = field(default=None, init=False, repr=False)
    _model_token: tuple | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.key, CollisionKey):
            raise TypeError("tree binding requires a CollisionKey")
        if isinstance(self.world, (bool, np.bool_)) or not isinstance(self.world, Integral) or int(self.world) != self.key.world:
            raise ValueError("tree binding world must match its key")
        group = self.group_index
        if isinstance(group, (bool, np.bool_)) or not isinstance(group, Integral) or not 0 <= group < MAX_TREE_JOINTS:
            raise ValueError("tree binding group must be a moving group index in [0,16)")
        object.__setattr__(self, "world", int(self.world))
        object.__setattr__(self, "group_index", int(group))
        object.__setattr__(self, "local_points_m", _points(self.local_points_m))
        if self.topology is not None:
            if not isinstance(self.topology, tuple):
                raise ValueError("tree binding topology must be a checkpoint tuple")
            object.__setattr__(self, "topology", _token(self.topology))

    @classmethod
    def from_asset(cls, asset, source_key, *, key, world):
        """Retain a registered ordinary/folded asset's actual owner and material.

        The fixed root is a static binding, not an artificial active group. Mimic
        and floating assets have different physical coordinates and are rejected.
        """
        from ..builders._robot_asset import RobotAsset
        from ..builders._urdf_frames import FoldedURDFAsset
        from ..core.robot import RobotModel

        if not isinstance(asset, RobotAsset) or not isinstance(asset.model, RobotModel):
            raise TypeError("tree source must be an ordinary fixed-root RobotAsset")
        if not isinstance(source_key, CollisionKey):
            raise TypeError("tree source key must be a CollisionKey")
        colliders = [c for c in asset.collisions if c.shape.key == source_key]
        if len(colliders) != 1:
            raise ValueError("tree source key must identify exactly one collider")
        collider = colliders[0]
        shapes = [s for s in asset.collision_model.shapes if s.key == source_key]
        if len(shapes) != 1 or shapes[0] != collider.shape:
            raise ValueError("tree source shape disagrees with its material registration")
        if collider.shape.kind not in ("sphere", "capsule", "hull"):
            raise ValueError("tree source supports only finite sphere, capsule or hull")
        local = np.asarray(collider.local_vertices, dtype=np.float64)
        if isinstance(asset, FoldedURDFAsset):
            site = asset.sites.get(collider.link_name)
            if site is None:
                raise ValueError("folded tree source lacks its authored owner site")
            group = site.link_index
            rotation = quaternion_matrix(np.asarray([site.local_orientation_xyzw], dtype=float))[0]
            local = np.asarray(site.local_position_m, dtype=float) + local @ rotation.T
        else:
            group = collider.link_index
        if group is None or group < 0 or group >= asset.model.dof_count:
            raise ValueError("tree source must belong to a moving group, not the fixed root")
        result = cls(key, world, group, local, topology=None)
        object.__setattr__(result, "_source_shape", collider.shape)
        object.__setattr__(result, "_model_token", tree_model_token(asset.model))
        return result


@dataclass(frozen=True, eq=False)
class TreeBindingCompiled:
    group_index: int
    zero_motion: CollisionMotion
    points_m: np.ndarray
    geometry: np.ndarray


def _outward_positive(value, name):
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"tree {name} must be finite and nonnegative")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rounded = np.float32(value)
        if float(rounded) < value:
            rounded = np.nextafter(rounded, np.float32(np.inf))
        # CUDA normal math may flush positive subnormals. Bound coefficients are
        # conservative metadata, so use the smallest normal positive upper bound.
        if value > 0 and rounded < np.finfo(np.float32).tiny:
            rounded = np.float32(np.finfo(np.float32).tiny)
    if not np.isfinite(rounded):
        raise ValueError(f"tree {name} cannot form finite FP32 geometry")
    return rounded


def _zero_pose_and_path(spec, group, points):
    """Static q=0 initialization only, retaining the original FP64 declaration."""
    parents = spec.model_token[0]
    offsets = np.asarray(spec.model_token[2], dtype=np.float64)
    rotations = quaternion_matrix(np.asarray(spec.model_token[3], dtype=np.float64))
    anchor = offsets[0]
    positions = np.zeros((spec.n, 3))
    frames = np.empty((spec.n, 3, 3))
    for index in range(group + 1):
        parent = parents[index]
        if parent < 0:
            positions[index] = offsets[index] - anchor
            frames[index] = rotations[index]
        else:
            positions[index] = positions[parent] + frames[parent] @ offsets[index]
            frames[index] = frames[parent] @ rotations[index]
    points_world = anchor + positions[group] + points @ frames[group].T
    depth, reach, index = 0, 0.0, group
    while index >= 0:
        offset = offsets[index] - anchor if parents[index] < 0 else offsets[index]
        reach += float(np.sum(np.abs(offset)))
        depth += 1
        index = parents[index]
    if not np.isfinite(points_world).all() or not np.isfinite(reach):
        raise ValueError("tree zero pose or path cannot be represented by finite geometry")
    return CollisionMotion(points_world), reach, depth


def compile_tree_binding(shape, binding, spec):
    """Compile owned points[V,3] and geometry[4] before any device allocation.

    geometry = radius_upper, local_reach_upper, fixed_path_reach_upper, depth.
    Per-query actual prismatic travel is added on device from the accepted bank.
    """
    if not isinstance(shape, CollisionShape) or not isinstance(binding, ResidentTreeBinding) or not isinstance(spec, TreeWorldSpec):
        raise TypeError("tree compilation requires a shape, binding and owned tree spec")
    if shape.key != binding.key or shape.key.world != binding.world:
        raise ValueError("tree shape key/world differs from its binding")
    if shape.kind not in ("sphere", "capsule", "hull") or not shape.rigid:
        raise ValueError("tree shape must be a finite sphere, capsule or hull")
    if binding.group_index >= spec.n:
        raise ValueError("tree binding group is outside its actual model")
    if len(shape.vertex_ids) != len(binding.local_points_m):
        raise ValueError("tree binding must retain every original shape material vertex")
    if binding._source_shape is not None:
        if binding._model_token != spec.model_token:
            raise ValueError("tree source model differs from the owned physical model")
        if shape != replace(binding._source_shape, key=binding.key):
            raise ValueError("tree source shape IDs, radius, revision or content changed")
    elif binding.topology is None or binding.topology != spec.topology:
        raise ValueError("tree binding topology differs from its owned physical model")
    points = binding.local_points_m
    motion, path_reach, depth = _zero_pose_and_path(spec, binding.group_index, points)
    with np.errstate(over="ignore", invalid="ignore"):
        local_reach = float(np.max(np.sum(np.abs(points), axis=1)))
    geometry = _owned([_outward_positive(shape.radius, "skin radius"),
        _outward_positive(local_reach, "local reach"),
        _outward_positive(path_reach, "fixed path reach"), np.float32(depth)])
    packed = _owned(as_fp32(points, "tree group-local material points", points.shape))
    return TreeBindingCompiled(binding.group_index, motion, packed, geometry)
