"""Owned geometry bindings for accepted single-revolute resident worlds.

A direct constructor declares an installation. ``from_asset`` additionally
checks the ordinary asset's real moving group and original shape registration.
The engineering bounds budget is intentionally not an all-real interval proof.
"""

from dataclasses import dataclass, field, replace
from numbers import Integral

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionMotion, CollisionShape
from ..dynamics.rotation import quaternion_matrix

_FRAME_ROUNDOFF = 32 * np.finfo(np.float64).eps
_TRIG_ABSOLUTE_BUDGET = 2.0**-20


def _finite_vector(value, length, name):
    raw = np.asarray(value)
    if raw.shape != (length,) or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real vector of length {length}")
    with np.errstate(over="ignore", invalid="ignore"):
        copied = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(copied).all():
        raise ValueError(f"{name} must contain finite FP64 values")
    return copied


def _unit_axis(value):
    axis = _finite_vector(value, 3, "revolute local axis")
    scale = float(np.max(np.abs(axis)))
    if scale == 0:
        raise ValueError("revolute local axis must be nonzero")
    axis /= scale
    axis /= np.linalg.norm(axis)
    return tuple(float(x) for x in axis)


def _unit_quaternion(value):
    q = _finite_vector(value, 4, "revolute orientation XYZW")
    norm = float(np.hypot.reduce(q))
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-10:
        raise ValueError("revolute orientation must be a unit XYZW quaternion")
    q /= norm
    for component in (q[3], *q[:3]):
        if component != 0:
            if component < 0:
                q = -q
            break
    return tuple(float(x) for x in q)


@dataclass(frozen=True)
class RevoluteGeometryFrame:
    """Owned joint pivot [m], zero child-to-world XYZW and unit local axis.

    These describe the actual physical joint. Zero geometry is already in the
    world frame; its local collider pose must not be applied a second time.
    """

    origin_m: tuple
    orientation_xyzw: tuple
    axis_local: tuple

    def __post_init__(self):
        object.__setattr__(
            self,
            "origin_m",
            tuple(float(x) for x in _finite_vector(self.origin_m, 3, "revolute origin")),
        )
        object.__setattr__(self, "orientation_xyzw", _unit_quaternion(self.orientation_xyzw))
        object.__setattr__(self, "axis_local", _unit_axis(self.axis_local))


def _owned_pose(motion):
    if not isinstance(motion, CollisionMotion):
        raise TypeError("revolute zero geometry must be a CollisionMotion")
    if motion.end is not None or motion.turn is not None:
        raise ValueError("revolute zero geometry requires one pose, not a moving interval")
    return CollisionMotion(motion.start, body_orientation_xyzw=motion.body_orientation_xyzw)


@dataclass(frozen=True, eq=False)
class ResidentRevoluteBinding:
    """Rotate one original finite shape from its true q=0 WORLD material pose.

    ``world`` names its real resident body. Frame metadata is checked against
    that owner's independently compiled model before device allocation.
    """

    key: CollisionKey
    world: int
    zero_motion: CollisionMotion
    origin_m: tuple = field(kw_only=True)
    orientation_xyzw: tuple = field(kw_only=True)
    axis_local: tuple = field(kw_only=True)
    _frame: RevoluteGeometryFrame = field(init=False, repr=False)
    _source_shape: CollisionShape | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.key, CollisionKey):
            raise TypeError("resident revolute key must be a CollisionKey")
        if (
            isinstance(self.world, (bool, np.bool_))
            or not isinstance(self.world, Integral)
            or int(self.world) != self.key.world
        ):
            raise ValueError("revolute binding world must match its collision key")
        frame = RevoluteGeometryFrame(self.origin_m, self.orientation_xyzw, self.axis_local)
        object.__setattr__(self, "world", int(self.world))
        object.__setattr__(self, "_frame", frame)
        for name in ("origin_m", "orientation_xyzw", "axis_local"):
            object.__setattr__(self, name, getattr(frame, name))
        object.__setattr__(self, "zero_motion", _owned_pose(self.zero_motion))

    @property
    def frame(self):
        return self._frame

    @classmethod
    def from_asset(cls, asset, source_key, *, key, world):
        """Bind an ordinary one-R moving group, retaining its complete shape.

        A folded tool uses its authored site's real group, not its original
        link number. A fixed root remains a static participant.
        """
        from ..builders._robot_asset import RobotAsset
        from ..builders._urdf_frames import FoldedURDFAsset
        from ..core.robot import RobotModel

        if not isinstance(asset, RobotAsset) or not isinstance(asset.model, RobotModel):
            raise TypeError("revolute asset must expose an ordinary fixed-tree RobotAsset")
        model = asset.model
        if model.joint_types != ("revolute",) or tuple(model.parents) != (-1,):
            raise ValueError("resident revolute asset requires exactly one root revolute joint")
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
        if collider.shape.kind not in ("sphere", "capsule", "hull"):
            raise ValueError(
                "resident revolute source supports only finite sphere, capsule or hull"
            )
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


def _readonly_fp32(value, name):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.array(value, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} cannot be represented by finite FP32 geometry")
    return np.frombuffer(result.tobytes(order="C"), dtype=np.float32).reshape(result.shape)


def compile_revolute_binding(shape, binding, owned_frame, owned_axis):
    """Return owned zero pose, readonly points[V,3], readonly geometry[8].

    Points are original world material vertices minus the actual pivot in FP64
    before FP32 conversion. Geometry is pivotXYZ, actual-axisXYZ, radius and a
    fixed engineering padding [m]. The owner axis is never replaced by a caller
    declaration. CUDA sinf/cosf use normal math; this budget is not a formal proof.
    """
    if not isinstance(shape, CollisionShape) or not isinstance(binding, ResidentRevoluteBinding):
        raise TypeError("expected a collision shape and ResidentRevoluteBinding")
    if shape.key != binding.key or shape.key.world != binding.world:
        raise ValueError("revolute shape key and world must match its binding")
    if shape.kind not in ("sphere", "capsule", "hull") or not shape.rigid:
        raise ValueError("resident revolute geometry supports finite sphere, capsule or hull")
    if len(shape.vertex_ids) != len(binding.zero_motion.start):
        raise ValueError("revolute zero pose must contain every registered material vertex")
    if not isinstance(owned_frame, RevoluteGeometryFrame):
        raise ValueError("actual revolute owner is missing its geometry frame")
    if binding.origin_m != owned_frame.origin_m:
        raise ValueError("revolute binding origin differs from its actual owned frame")
    for name in ("orientation_xyzw", "axis_local"):
        if not np.allclose(
            getattr(binding, name), getattr(owned_frame, name), rtol=0, atol=_FRAME_ROUNDOFF
        ):
            raise ValueError(f"revolute binding {name} differs from its actual owned frame")
    axis = _finite_vector(owned_axis, 3, "actual revolute world axis")
    expected = quaternion_matrix(np.array([owned_frame.orientation_xyzw]))[0] @ np.asarray(
        owned_frame.axis_local
    )
    if not np.allclose(axis, expected, rtol=0, atol=_FRAME_ROUNDOFF):
        raise ValueError("actual revolute world axis disagrees with its owned geometry frame")
    if binding._source_shape is not None and shape != replace(
        binding._source_shape, key=binding.key
    ):
        raise ValueError("target shape changed source material IDs, revision or shape content")
    origin = np.asarray(owned_frame.origin_m)
    points = binding.zero_motion.start
    with np.errstate(over="ignore", invalid="ignore"):
        relative = points - origin
        length = float(np.max(np.hypot.reduce(relative, axis=1))) + shape.radius
        scale = max(1.0, float(np.max(np.abs(origin))), float(np.max(np.abs(points))), shape.radius)
        padding = 64 * np.finfo(np.float32).eps * scale + 8 * _TRIG_ABSOLUTE_BUDGET * length
    if not np.isfinite(relative).all() or not np.isfinite(padding) or padding <= 0:
        raise ValueError("revolute geometry cannot form finite engineering bounds")
    packed_points = _readonly_fp32(relative, "revolute relative points")
    # Round the positive budget outward, including conversion of radius/frame.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        padding32 = np.float32(padding)
        if float(padding32) < padding:
            padding32 = np.nextafter(padding32, np.float32(np.inf))
        radius32 = np.float32(shape.radius)
        if float(radius32) < shape.radius:
            radius32 = np.nextafter(radius32, np.float32(np.inf))
    geometry = _readonly_fp32((*origin, *axis, radius32, padding32), "revolute geometry")
    return binding.zero_motion, packed_points, geometry
