"""Owned geometry declarations for accepted single-prismatic resident worlds.

This module imports neither the resident plan nor its physical world layout.
A direct binding is an explicit installation declaration. Only from_asset also
verifies a supported asset's real moving group and original shape registration.
"""

from dataclasses import dataclass, field, replace
from numbers import Integral

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionMotion, CollisionShape
from ..dynamics.rotation import quaternion_matrix

_FRAME_ROUNDOFF = 32 * np.finfo(np.float64).eps


def _finite_vector(value, length, name):
    raw = np.asarray(value)
    if raw.shape != (length,) or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real vector of length {length}")
    with np.errstate(over="ignore", invalid="ignore"):
        copied = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(copied).all():
        raise ValueError(f"{name} must be finite FP64 values")
    return copied


def _unit_axis(value):
    axis = _finite_vector(value, 3, "slider local axis")
    scale = float(np.max(np.abs(axis)))
    if scale == 0:
        raise ValueError("slider local axis must be nonzero")
    axis /= scale
    axis /= np.linalg.norm(axis)
    return tuple(float(x) for x in axis)


def _unit_quaternion(value):
    q = _finite_vector(value, 4, "slider orientation XYZW")
    norm = float(np.hypot.reduce(q))
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-10:
        raise ValueError("slider orientation must be a unit XYZW quaternion")
    q /= norm
    # q and -q encode exactly the same rotation, including the w=0 half-turn.
    for component in (q[3], *q[:3]):
        if component != 0:
            if component < 0:
                q = -q
            break
    return tuple(float(x) for x in q)


@dataclass(frozen=True)
class SliderGeometryFrame:
    """Zero joint origin [m], unit child-to-world XYZW and normalized local axis.

    Metadata is owned, immutable and independent of any mutable RobotModel array.
    This frame does not transform zero_motion a second time.
    """

    origin_m: tuple
    orientation_xyzw: tuple
    axis_local: tuple

    def __post_init__(self):
        origin = tuple(float(x) for x in _finite_vector(self.origin_m, 3, "slider origin"))
        object.__setattr__(self, "origin_m", origin)
        object.__setattr__(self, "orientation_xyzw", _unit_quaternion(self.orientation_xyzw))
        object.__setattr__(self, "axis_local", _unit_axis(self.axis_local))


def _owned_single_pose(motion):
    if not isinstance(motion, CollisionMotion):
        raise TypeError("slider zero geometry must be a CollisionMotion")
    if motion.end is not None or motion.turn is not None:
        raise ValueError("slider zero geometry requires one pose, not an interval or turn")
    # CollisionMotion owns bytes-backed FP64 vertices and an immutable quaternion.
    return CollisionMotion(motion.start, body_orientation_xyzw=motion.body_orientation_xyzw)


@dataclass(frozen=True, eq=False)
class ResidentSliderBinding:
    """One finite shape translated from its true q=0 WORLD pose by accepted q.

    origin/orientation/axis describe the real model joint, not the collider COM
    or its local geometry frame. The owning batch validates this declaration
    against its independently owned physical frame before allocating device data.
    """

    key: CollisionKey
    world: int
    zero_motion: CollisionMotion
    origin_m: tuple = field(kw_only=True)
    orientation_xyzw: tuple = field(kw_only=True)
    axis_local: tuple = field(kw_only=True)
    _frame: SliderGeometryFrame = field(init=False, repr=False)
    _source_shape: CollisionShape | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.key, CollisionKey):
            raise TypeError("resident slider key must be a CollisionKey")
        if (
            isinstance(self.world, (bool, np.bool_))
            or not isinstance(self.world, Integral)
            or int(self.world) != self.key.world
        ):
            raise ValueError("slider binding world must match its collision key")
        frame = SliderGeometryFrame(self.origin_m, self.orientation_xyzw, self.axis_local)
        object.__setattr__(self, "world", int(self.world))
        object.__setattr__(self, "_frame", frame)
        for name in ("origin_m", "orientation_xyzw", "axis_local"):
            object.__setattr__(self, name, getattr(frame, name))
        object.__setattr__(self, "zero_motion", _owned_single_pose(self.zero_motion))

    @property
    def frame(self):
        return self._frame

    @classmethod
    def from_asset(cls, asset, source_key, *, key, world):
        """Copy a genuine single-prismatic moving collider's public q=0 geometry.

        The caller must register the original shape with only its key remapped.
        Fixed root geometry is static. Folded tools use the named site's true
        moving group; the original authored link number is not a physical index.
        """
        from ..builders._robot_asset import RobotAsset
        from ..builders._urdf_frames import FoldedURDFAsset
        from ..core.robot import RobotModel

        if not isinstance(asset, RobotAsset) or not isinstance(asset.model, RobotModel):
            raise TypeError("slider asset must expose an ordinary fixed-tree RobotAsset")
        model = asset.model
        if model.joint_types != ("prismatic",) or tuple(model.parents) != (-1,):
            raise ValueError("resident slider asset requires exactly one root prismatic joint")
        if not isinstance(source_key, CollisionKey):
            raise TypeError("source collider key must be a CollisionKey")
        colliders = [c for c in asset.collisions if c.shape.key == source_key]
        if len(colliders) != 1:
            raise ValueError("source key must identify exactly one asset collider")
        collider = colliders[0]
        if isinstance(asset, FoldedURDFAsset):
            site = asset.sites.get(collider.link_name)
            if site is None:
                raise ValueError("folded collider has no authored site owner")
            group = site.link_index
        else:
            group = collider.link_index
        if group != 0:
            raise ValueError("source collider must belong to active group 0, not the fixed root")
        registered = [s for s in asset.collision_model.shapes if s.key == source_key]
        if len(registered) != 1 or registered[0] != collider.shape:
            raise ValueError("source collider does not match the asset shape registration")
        paths = asset.collision_motion(np.zeros(1, dtype=np.float64))
        if source_key not in paths:
            raise ValueError("asset zero pose is missing its registered source collider")
        result = cls(
            key,
            world,
            paths[source_key],
            origin_m=model.joint_position[0],
            orientation_xyzw=model.joint_orientation[0],
            axis_local=model.joint_axis[0],
        )
        object.__setattr__(result, "_source_shape", collider.shape)
        return result


def _axis_intervals(axis):
    """FP32 lower/upper intervals, including normal bounds around FTZ inputs."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        nearest = np.array(axis, dtype=np.float32, copy=True)
        lower, upper = nearest.copy(), nearest.copy()
        for i, value in enumerate(axis):
            if nearest[i] > value:
                lower[i] = np.nextafter(nearest[i], np.float32(-np.inf))
            if nearest[i] < value:
                upper[i] = np.nextafter(nearest[i], np.float32(np.inf))
            if 0 < abs(float(lower[i])) < np.finfo(np.float32).tiny:
                lower[i] = -np.finfo(np.float32).tiny if lower[i] < 0 else 0.0
            if 0 < abs(float(upper[i])) < np.finfo(np.float32).tiny:
                upper[i] = np.finfo(np.float32).tiny if upper[i] > 0 else 0.0
    intervals = np.concatenate((lower, upper))
    if not np.isfinite(intervals).all():
        raise ValueError("actual slider world axis cannot have finite FP32 bounds")
    return np.frombuffer(intervals.tobytes(order="C"), dtype=np.float32)


def compile_slider_binding(shape, binding, owned_frame, owned_axis):
    """Validate ownership and return (owned zero_motion, readonly FP32 axis[6]).

    Parent code computes the zero-pose envelope through its existing geometry
    route. Dynamic bounds use only the owner's actual FP64 physical world axis;
    the caller's axis is never substituted for it.
    """
    if not isinstance(shape, CollisionShape) or not isinstance(binding, ResidentSliderBinding):
        raise TypeError("expected a collision shape and ResidentSliderBinding")
    if shape.key != binding.key or shape.key.world != binding.world:
        raise ValueError("slider shape key and world must match its binding")
    if shape.kind == "plane" or not shape.rigid:
        raise ValueError("resident slider geometry must be a finite rigid shape")
    if len(shape.vertex_ids) != len(binding.zero_motion.start):
        raise ValueError("slider zero pose must contain every registered material vertex")
    if not isinstance(owned_frame, SliderGeometryFrame):
        raise ValueError("actual slider owner is missing its geometry frame")
    if binding.origin_m != owned_frame.origin_m:
        raise ValueError("slider binding origin differs from its actual owned frame")
    for name in ("orientation_xyzw", "axis_local"):
        if not np.allclose(
            getattr(binding, name), getattr(owned_frame, name), rtol=0, atol=_FRAME_ROUNDOFF
        ):
            raise ValueError(f"slider binding {name} differs from its actual owned frame")
    axis = _finite_vector(owned_axis, 3, "actual slider world axis")
    expected = quaternion_matrix(np.array([owned_frame.orientation_xyzw]))[0] @ np.asarray(
        owned_frame.axis_local
    )
    if not np.allclose(axis, expected, rtol=0, atol=_FRAME_ROUNDOFF):
        raise ValueError("actual slider world axis disagrees with its owned geometry frame")
    if binding._source_shape is not None:
        expected_shape = replace(binding._source_shape, key=binding.key)
        if shape != expected_shape:
            raise ValueError("target shape changed source material IDs, revision or shape content")
    return binding.zero_motion, _axis_intervals(axis)
