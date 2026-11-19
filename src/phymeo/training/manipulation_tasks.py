"""Authored manipulation scenes and a HOST reference success oracle.

This module neither compiles nor advances physics and claims no task success.
Goals are evaluator truth, never task-ID actor features. All visible candidates
remain in Scene.targets; natural-language strings never replace numeric poses.
SuccessFrame values must come from one accepted step's INPUT dynamics/contact
phase, in metres, seconds, newtons and world-frame velocities; quaternions WXYZ.
"""

import json
import math
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import combinations, product

from ..geometry._prism_clip import hull as _hull
from ..geometry._prism_clip import intersect as _convex_intersection

H = 0.001


def _vector(value, length, name):
    result = tuple(float(x) for x in value)
    if len(result) != length or not all(math.isfinite(x) for x in result):
        raise ValueError(name + " must contain finite numeric components")
    return result


@dataclass(frozen=True)
class Primitive:
    kind: str
    size: tuple
    position: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        widths = {"sphere": 1, "capsule": 2, "box": 3}
        if self.kind not in widths:
            raise ValueError("only sphere/capsule/box object primitives are declared")
        object.__setattr__(self, "size", _vector(self.size, widths[self.kind], "size"))
        object.__setattr__(self, "position", _vector(self.position, 3, "position"))
        if min(self.size) <= 0:
            raise ValueError("primitive dimensions must be positive")


@dataclass(frozen=True)
class Target:
    key: str
    label_zh: str
    label_en: str
    position: tuple
    yaw: float
    half_extents: tuple = (0.085, 0.085)

    def __post_init__(self):
        for field in ("key", "label_zh", "label_en"):
            if type(getattr(self, field)) is not str or not getattr(self, field).strip():
                raise ValueError("target identifiers and labels must be nonempty strings")
        object.__setattr__(self, "yaw", float(self.yaw))
        object.__setattr__(self, "position", _vector(self.position, 3, "target position"))
        object.__setattr__(self, "half_extents", _vector(self.half_extents, 2, "target extents"))
        if not math.isfinite(self.yaw) or min(self.half_extents) <= 0:
            raise ValueError("finite target yaw and positive extents required")


@dataclass(frozen=True)
class Goal:
    """Task kind/object/target bindings are evaluator truth, not actor inputs."""

    kind: str
    object_key: str
    target_key: str
    instructions_zh: tuple
    instructions_en: tuple
    lift_m: float = 0.03
    transport_m: float = 0.03
    speed_m_s: float = 0.03
    angular_speed_rad_s: float = 0.15
    dwell_s: float = 0.2
    position_tolerance_m: float = 0.015
    yaw_tolerance_rad: float = 0.12
    tilt_tolerance_rad: float = 0.1
    footprint_iou_min: float = 0.9
    support_weight_tolerance: float = 0.2

    def __post_init__(self):
        if self.kind not in ("grasp", "place", "pusht"):
            raise ValueError("unknown manipulation predicate")
        for field in ("instructions_zh", "instructions_en"):
            values = tuple(getattr(self, field))
            if len(values) < 2 or any(not isinstance(s, str) or not s.strip() for s in values):
                raise ValueError("each language needs an original and a paraphrase")
            object.__setattr__(self, field, values)
        for field, value in asdict(self).items():
            if field.endswith(("_m", "_s", "_rad")) or field in (
                "footprint_iou_min",
                "support_weight_tolerance",
            ):
                object.__setattr__(self, field, float(value))
        thresholds = tuple(v for k, v in asdict(self).items() if k.endswith(("_m", "_s", "_rad")))
        if any(not math.isfinite(v) or v <= 0 for v in thresholds):
            raise ValueError("predicate thresholds must be finite and positive")
        if not 0 < self.footprint_iou_min <= 1 or not 0 < self.support_weight_tolerance < 1:
            raise ValueError("invalid overlap/support threshold")

    @property
    def binding(self):
        return sha256(
            json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()


@dataclass(frozen=True)
class Scene:
    key: str
    robot_kinds: str
    mjcf: str
    object_shapes: tuple
    object_mass_kg: float
    initial_object_position: tuple
    targets: tuple
    goals: tuple
    default_goal: int
    robot_joint_names: tuple
    actuator_names: tuple
    finger_geom_names: tuple
    object_key: str = "object"
    support_geom_name: str = "support"
    support_top_m: float = 0.0
    timestep_s: float = H
    object_label_zh: str = "物体"
    object_label_en: str = "object"
    initial_object_wxyz: tuple = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self):
        for field in (
            "key",
            "mjcf",
            "object_key",
            "support_geom_name",
            "object_label_zh",
            "object_label_en",
        ):
            value = getattr(self, field)
            if type(value) is not str or not value.strip():
                raise ValueError("scene identifiers, labels and XML must be nonempty strings")
        if type(self.robot_kinds) is not str or any(kind not in "RP" for kind in self.robot_kinds):
            raise ValueError("robot_kinds must describe scalar R/P joints")
        for field, expected in (("object_shapes", Primitive), ("targets", Target), ("goals", Goal)):
            values = tuple(getattr(self, field))
            if not values or any(type(value) is not expected for value in values):
                raise ValueError(field + " requires nonempty typed declarations")
            object.__setattr__(self, field, values)
        for field in ("robot_joint_names", "actuator_names", "finger_geom_names"):
            values = tuple(getattr(self, field))
            if any(type(value) is not str or not value.strip() for value in values):
                raise ValueError(field + " requires nonempty names")
            if len(set(values)) != len(values):
                raise ValueError(field + " requires unique names")
            object.__setattr__(self, field, values)
        if len(self.finger_geom_names) not in (0, 2):
            raise ValueError("scene declares no fingers or exactly two contact fingers")
        if not self.robot_kinds and not self.finger_geom_names:
            raise ValueError("a scene needs an arm or a standalone two-finger gripper")
        count = len(self.robot_joint_names)
        if count != len(self.robot_kinds) + len(self.finger_geom_names):
            raise ValueError("robot joint names must cover the arm and declared fingers")
        if len(self.actuator_names) != count or count + 6 > 35 or count + 7 > 45:
            raise ValueError("actuators and complete FREE coordinates must fit the declared layout")
        object.__setattr__(
            self,
            "initial_object_position",
            _vector(self.initial_object_position, 3, "initial position"),
        )
        object.__setattr__(
            self, "initial_object_wxyz", _vector(self.initial_object_wxyz, 4, "initial WXYZ")
        )
        if abs(sum(q * q for q in self.initial_object_wxyz) - 1) > 2e-5:
            raise ValueError("initial object orientation must be unit WXYZ")
        for field in ("object_mass_kg", "support_top_m", "timestep_s"):
            value = float(getattr(self, field))
            if not math.isfinite(value) or (field != "support_top_m" and value <= 0):
                raise ValueError("finite scene scalars and positive mass/timestep required")
            object.__setattr__(self, field, value)
        target_keys = {target.key for target in self.targets}
        if len(target_keys) != len(self.targets):
            raise ValueError("target keys must be unique")
        for goal in self.goals:
            if type(goal.target_key) is not str or goal.target_key not in target_keys:
                raise ValueError("every goal must name a declared visible target")
            if (
                type(goal.kind) is not str
                or type(goal.object_key) is not str
                or goal.object_key != self.object_key
            ):
                raise ValueError("every goal must name the declared object")
            if goal.kind in ("grasp", "place") and len(self.finger_geom_names) != 2:
                raise ValueError("grasp/place predicates require two declared contact fingers")
            if goal.kind == "pusht" and (
                len(self.object_shapes) > 2 or any(s.kind != "box" for s in self.object_shapes)
            ):
                raise ValueError("PushT footprint requires one or two actual box components")
        if type(self.default_goal) is not int or not 0 <= self.default_goal < len(self.goals):
            raise ValueError("default_goal must index the declared goals")

    @property
    def physical_binding(self):
        """Bind all owned declarations, including metadata not encoded by XML."""
        metadata = asdict(self)
        metadata["mjcf"] = self.source_sha256
        return sha256(
            json.dumps(metadata, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        ).hexdigest()

    @property
    def object_geom_names(self):
        return tuple(f"object-{i}" for i in range(len(self.object_shapes)))

    @property
    def source_sha256(self):
        return sha256(self.mjcf.encode()).hexdigest()


@dataclass(frozen=True)
class SuccessFrame:
    world: int
    episode: int
    tick: int
    time_s: float
    object_position: tuple
    object_wxyz: tuple
    linear_velocity: tuple
    angular_velocity: tuple
    finger_normal_forces_n: tuple
    support_up_force_n: float
    phase: str = "accepted-entry"
    object_key: str = "object"

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in (self.world, self.episode, self.tick)):
            raise ValueError("world/episode/tick require nonnegative integer routing metadata")
        if self.phase != "accepted-entry":
            raise ValueError("pose, velocities and contact loads must share accepted-entry phase")
        for field, size in (
            ("object_position", 3),
            ("object_wxyz", 4),
            ("linear_velocity", 3),
            ("angular_velocity", 3),
            ("finger_normal_forces_n", 2),
        ):
            object.__setattr__(self, field, _vector(getattr(self, field), size, field))
        if abs(sum(q * q for q in self.object_wxyz) - 1) > 2e-5:
            raise ValueError("object orientation must be a unit WXYZ quaternion")
        for field in ("time_s", "support_up_force_n"):
            object.__setattr__(self, field, float(getattr(self, field)))
        if not isinstance(self.object_key, str) or not self.object_key:
            raise ValueError("a named object binding is required")
        if not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError("invalid physical time")
        if not math.isfinite(self.support_up_force_n) or self.support_up_force_n < 0:
            raise ValueError("support load must be an observed nonnegative upward force")
        if min(self.finger_normal_forces_n) < 0:
            raise ValueError("finger loads must be nonnegative contact normal forces")


@dataclass(frozen=True)
class SuccessHistory:
    binding: tuple
    tick: int
    time_s: float
    initial_bottom_m: float
    initial_xy: tuple
    lifted: bool = False
    transported: bool = False
    held_s: float = 0.0
    holding: bool = False


@dataclass(frozen=True)
class SuccessResult:
    success: bool
    history: SuccessHistory
    metrics: tuple


def _rotation(q):
    w, x, y, z = q
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def _transform(point, position, rotation):
    return tuple(position[i] + sum(rotation[i][j] * point[j] for j in range(3)) for i in range(3))


def _bounds(shapes, position, rotation):
    lower, upper = [math.inf] * 3, [-math.inf] * 3
    for shape in shapes:
        center = _transform(shape.position, position, rotation)
        for axis in range(3):
            if shape.kind == "box":
                extent = sum(abs(rotation[axis][j]) * shape.size[j] for j in range(3))
            elif shape.kind == "capsule":
                extent = shape.size[0] + shape.size[1] * abs(rotation[axis][2])
            else:
                extent = shape.size[0]
            lower[axis] = min(lower[axis], center[axis] - extent)
            upper[axis] = max(upper[axis], center[axis] + extent)
    return tuple(lower), tuple(upper)


def _intersection(subject, clip):
    if len(subject) < 3 or len(clip) < 3:
        return ()
    return _convex_intersection(subject, clip)


def _area(poly):
    return abs(sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(poly, poly[1:] + poly[:1]))) / 2


def _union_area(polygons):
    # At most two authored object boxes and two target boxes: full inclusion-
    # exclusion preserves overlapping projections, even under a small tilt.
    total = 0.0
    for count in range(1, len(polygons) + 1):
        for selected in combinations(polygons, count):
            intersection = selected[0]
            for other in selected[1:]:
                intersection = _intersection(intersection, other)
            total += (-1 if count % 2 == 0 else 1) * _area(intersection)
    return max(0.0, total)


def footprint_iou(shapes, position, wxyz, target):
    """Exact polygon-union XY projection of the declared boxes, not their AABB."""
    if not 1 <= len(shapes) <= 2 or any(s.kind != "box" for s in shapes):
        raise ValueError("footprint oracle supports one or two actual box components")
    position = _vector(position, 3, "object position")
    wxyz = _vector(wxyz, 4, "object WXYZ")
    if abs(sum(q * q for q in wxyz) - 1) > 2e-5:
        raise ValueError("footprint requires a unit WXYZ quaternion")
    actual_rotation = _rotation(wxyz)
    desired_rotation = _rotation((math.cos(target.yaw / 2), 0, 0, math.sin(target.yaw / 2)))
    groups = []
    for center, rotation in ((position, actual_rotation), (target.position, desired_rotation)):
        polygons = []
        for shape in shapes:
            vertices = [
                tuple(shape.position[i] + sign[i] * shape.size[i] for i in range(3))
                for sign in product((-1, 1), repeat=3)
            ]
            polygons.append(_hull([_transform(v, center, rotation)[:2] for v in vertices]))
        groups.append(polygons)
    a, b = groups
    union = _union_area(a + b)
    overlap = _union_area(a) + _union_area(b) - union
    return min(1.0, max(0.0, overlap / union)) if union > 0 else 0.0


def evaluate_goal(scene, goal, frame, history=None):
    """Pure HOST oracle. Supply EVERY accepted physical input sample, no skips.

    finger_normal_forces_n are the two named finger/object sums; support force
    is only the named support/object upward sum. Controls or total actuator
    forces are not substitutes. History belongs in the episode checkpoint.
    """
    if (
        goal not in scene.goals
        or goal.object_key != scene.object_key
        or frame.object_key != goal.object_key
    ):
        raise ValueError("goal is not bound to this scene/object")
    target = next(t for t in scene.targets if t.key == goal.target_key)
    target_binding = (target.key, target.position, target.yaw, target.half_extents)
    binding = (scene.physical_binding, goal.binding, target_binding, frame.world, frame.episode)
    rotation = _rotation(frame.object_wxyz)
    lower, upper = _bounds(scene.object_shapes, frame.object_position, rotation)
    bottom = lower[2] - scene.support_top_m
    if history is None:
        if frame.tick != 0:
            raise ValueError("new episode history must begin at tick 0")
        if not -0.005 <= bottom <= 0.005:
            raise ValueError("episode history must begin with the object at the support surface")
        history = SuccessHistory(
            binding, frame.tick, frame.time_s, bottom, frame.object_position[:2]
        )
        delta = 0.0
    else:
        delta = frame.time_s - history.time_s
        if history.binding != binding or frame.tick != history.tick + 1:
            raise ValueError("history crossed scene/goal/world/episode or skipped a physical tick")
        if abs(delta - scene.timestep_s) > 2e-6:
            raise ValueError("history must advance by exactly one physical timestep")
    weight = scene.object_mass_kg * 9.81
    gripping = min(frame.finger_normal_forces_n) > 0.05 * weight
    released = max(frame.finger_normal_forces_n) < 0.02 * weight
    unsupported = frame.support_up_force_n < 0.05 * weight
    supported = abs(frame.support_up_force_n / weight - 1) <= goal.support_weight_tolerance
    elevated = bottom >= goal.lift_m and bottom - history.initial_bottom_m >= goal.lift_m
    lifted = history.lifted or (elevated and gripping and unsupported)
    distance = math.dist(frame.object_position[:2], history.initial_xy)
    transported = history.transported or (
        lifted and elevated and gripping and distance >= goal.transport_m
    )
    speed = math.sqrt(sum(x * x for x in frame.linear_velocity))
    angular_speed = math.sqrt(sum(x * x for x in frame.angular_velocity))
    slow = speed <= goal.speed_m_s and angular_speed <= goal.angular_speed_rad_s
    yaw = math.atan2(rotation[1][0], rotation[0][0])
    yaw_error = abs(math.atan2(math.sin(yaw - target.yaw), math.cos(yaw - target.yaw)))
    tilt = math.acos(max(-1.0, min(1.0, rotation[2][2])))
    position_error = math.dist(frame.object_position[:2], target.position[:2])
    iou = 0.0
    if goal.kind == "grasp":
        candidate = elevated and gripping and unsupported and slow
    elif goal.kind == "place":
        # Target-local exact projected extrema of all primitive geometry.
        target_rotation = _rotation((math.cos(-target.yaw / 2), 0, 0, math.sin(-target.yaw / 2)))
        relative = tuple(frame.object_position[i] - target.position[i] for i in range(3))
        local_position = _transform(relative, (0, 0, 0), target_rotation)
        local_rotation = tuple(
            tuple(sum(target_rotation[i][k] * rotation[k][j] for k in range(3)) for j in range(3))
            for i in range(3)
        )
        lo, hi = _bounds(scene.object_shapes, local_position, local_rotation)
        contained = all(
            -target.half_extents[i] <= lo[i] and hi[i] <= target.half_extents[i] for i in (0, 1)
        )
        candidate = (
            lifted
            and transported
            and released
            and supported
            and slow
            and contained
            and abs(bottom) <= 0.005
        )
    else:
        iou = footprint_iou(scene.object_shapes, frame.object_position, frame.object_wxyz, target)
        candidate = (
            iou >= goal.footprint_iou_min
            and yaw_error <= goal.yaw_tolerance_rad
            and tilt <= goal.tilt_tolerance_rad
            and position_error <= goal.position_tolerance_m
            and supported
            and slow
            and abs(bottom) <= 0.005
        )
    held = history.held_s + delta if candidate and history.holding else 0.0
    state = SuccessHistory(
        binding,
        frame.tick,
        frame.time_s,
        history.initial_bottom_m,
        history.initial_xy,
        lifted,
        transported,
        held,
        candidate,
    )
    metrics = (
        ("clearance_m", bottom),
        ("speed_m_s", speed),
        ("angular_speed_rad_s", angular_speed),
        ("position_error_m", position_error),
        ("yaw_error_rad", yaw_error),
        ("footprint_iou", iou),
        ("held_s", held),
    )
    return SuccessResult(candidate and held + 1e-12 >= goal.dwell_s, state, metrics)


def _numbers(values):
    return " ".join(format(value, ".12g") for value in values)


def _goals(object_zh, object_en, targets, pushing):
    result = []
    for target in targets:
        if pushing:
            result.append(
                Goal(
                    "pusht",
                    "object",
                    target.key,
                    (
                        f"将{object_zh}推到{target.label_zh}，并对齐朝向。",
                        f"推动{object_zh}，让它与{target.label_zh}重合。",
                    ),
                    (
                        f"Push the {object_en} into the {target.label_en} and align its orientation.",
                        f"Move the {object_en} by pushing until it matches the {target.label_en}.",
                    ),
                )
            )
        else:
            if not result:
                result.append(
                    Goal(
                        "grasp",
                        "object",
                        target.key,
                        (
                            f"抓起{object_zh}，让它离开桌面并保持稳定。",
                            f"把{object_zh}夹起，稳稳保持悬空。",
                        ),
                        (
                            f"Pick up the {object_en} and hold it above the table.",
                            f"Grasp and lift the {object_en}, then keep it steady in the air.",
                        ),
                    )
                )
            result.append(
                Goal(
                    "place",
                    "object",
                    target.key,
                    (
                        f"把{object_zh}放到{target.label_zh}，然后松开夹爪。",
                        f"先抓起{object_zh}，再放稳在{target.label_zh}并松手。",
                    ),
                    (
                        f"Pick up the {object_en}, place it in the {target.label_en}, then release it.",
                        f"Lift the {object_en}, move it to the {target.label_en}, and let go once it is placed.",
                    ),
                )
            )
    return tuple(result)


def manipulation_scenes():
    """Eight unqualified authored models; caller owns compilation and validation.

    Collision bits: robot=1 (including self), object=2, support=4. Robot/object,
    robot/support and object/support are enabled; MuJoCo parent/weld filters
    still apply. Targets are noncolliding MuJoCo sites. Only robot joints
    have motors; each object is a separate full FREE body, with native inertia
    inferred from explicit geom masses. No equality constraints or trajectories.
    """

    def box(size, position=(0, 0, 0)):
        return Primitive("box", size, position)

    rows = (
        (
            "grasp-sphere",
            "RRPR",
            "红球",
            "red ball",
            (Primitive("sphere", (0.025,)),),
            0.08,
            "1 .2 .2 1",
            0,
        ),
        (
            "grasp-box",
            "RRRR",
            "蓝盒",
            "blue box",
            (box((0.03, 0.025, 0.03)),),
            0.12,
            ".2 .4 1 1",
            0,
        ),
        (
            "place-block",
            "RRRRR",
            "绿色长盒",
            "green rectangular block",
            (box((0.04, 0.025, 0.025)),),
            0.14,
            ".2 .8 .3 1",
            1,
        ),
        (
            "place-capsule",
            "RRRRRR",
            "黄色胶囊状物体",
            "yellow capsule",
            (Primitive("capsule", (0.022, 0.03)),),
            0.12,
            "1 .8 .2 1",
            2,
        ),
        (
            "place-compound",
            "RRRRRRR",
            "橙色阶梯块",
            "orange stepped block",
            (box((0.03, 0.025, 0.015)), box((0.015, 0.025, 0.015), (0.015, 0, 0.03))),
            0.15,
            "1 .5 .1 1",
            3,
        ),
        (
            "pusht-planar",
            "RR",
            "标准T形块",
            "standard T block",
            (box((0.018, 0.045, 0.015), (0, -0.025, 0)), box((0.065, 0.018, 0.015), (0, 0.038, 0))),
            0.16,
            ".6 .3 .9 1",
            0,
        ),
        (
            "pusht-redundant",
            "RRR",
            "宽T形块",
            "wide T block",
            (box((0.022, 0.045, 0.015), (0, -0.025, 0)), box((0.075, 0.018, 0.015), (0, 0.038, 0))),
            0.18,
            ".2 .8 .8 1",
            1,
        ),
        (
            "pusht-cartesian",
            "PPPR",
            "窄T形块",
            "narrow T block",
            (box((0.014, 0.045, 0.015), (0, -0.025, 0)), box((0.055, 0.018, 0.015), (0, 0.038, 0))),
            0.14,
            ".9 .4 .6 1",
            2,
        ),
    )
    scenes = []
    for key, kinds, zh, en, shapes, mass, color, default in rows:
        pushing = key.startswith("pusht")
        low, _ = _bounds(shapes, (0, 0, 0), _rotation((1, 0, 0, 0)))
        initial = (0.52 if pushing else 0.4, 0.0, -low[2] + 0.001)
        targets = tuple(
            Target(
                name,
                label_zh,
                label_en,
                (x, y, initial[2]),
                yaw,
                (0.085, 0.085) if pushing else (0.06, 0.06),
            )
            for name, label_zh, label_en, x, y, yaw in (
                (
                    "left",
                    "左侧蓝色目标区",
                    "left blue target",
                    0.55 if pushing else 0.34,
                    0.13 if pushing else 0.15,
                    0.2 if pushing else 0,
                ),
                (
                    "right",
                    "右侧黄色目标区",
                    "right yellow target",
                    0.55 if pushing else 0.34,
                    -0.13 if pushing else -0.15,
                    -0.25 if pushing else 0,
                ),
                (
                    "middle",
                    "中间白色目标区",
                    "middle white target",
                    0.58 if pushing else 0.30,
                    0,
                    0,
                ),
            )
        )
        lengths = [
            (0.6 if kinds in ("RR", "RRR") and pushing else 0.48 if pushing else 0.52) / len(kinds)
        ] * len(kinds)
        joint_names = tuple(f"joint{i}" for i in range(len(kinds))) + (
            () if pushing else ("left-finger", "right-finger")
        )
        actuators = tuple(f"motor-{name}" for name in joint_names)
        fingers = () if pushing else ("left-pad", "right-pad")
        tool = '<body name="tool" pos="' + _numbers((lengths[-1], 0, 0)) + '">'
        if pushing:
            tool += (
                '<geom name="pusher" type="capsule" size=".018 .035" pos="0 0 -.16" mass=".05"/>'
            )
        else:
            tool += (
                '<geom name="wrist" type="capsule" fromto="0 0 0 0 0 -.09" size=".016" mass=".05"/>'
            )
            for side, sign in (("left", 1), ("right", -1)):
                tool += (
                    f'<body name="{side}-finger-body" pos="0 {sign * 0.07} -.14">'
                    f'<joint name="{side}-finger" type="slide" axis="0 {-sign} 0" limited="true" range="0 .06"/>'
                    f'<geom name="{side}-pad" type="box" size=".012 .009 .04" mass=".04"/></body>'
                )
        tool += "</body>"
        content = tool
        for index in reversed(range(len(kinds))):
            kind = kinds[index]
            if pushing:
                axis = ("1 0 0", "0 1 0", "0 0 1")[index] if kind == "P" else "0 0 1"
            else:
                axis = (
                    "0 0 1"
                    if kind == "P" or index == 0 or kinds == "RRPR"
                    else ("0 1 0" if index < 3 else "1 0 0" if index % 2 else "0 0 1")
                )
            pos = (
                ((-0.08 if pushing else -0.12), 0, 0.22 if pushing else 0.24)
                if index == 0
                else (lengths[index - 1], 0, 0)
            )
            yaw = 0.0
            if pushing and kinds == "RR":
                yaw = (0.6, -1.2)[index]
            elif pushing and kinds == "RRR":
                yaw = (0.8, -1.6, 0.8)[index]
            interval = "-.15 .15" if kind == "P" else "-2.6 2.6"
            content = (
                f'<body name="link{index}" pos="{_numbers(pos)}" quat="{_numbers((math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))}">'
                f'<joint name="joint{index}" type="{"slide" if kind == "P" else "hinge"}" axis="{axis}" limited="true" range="{interval}"/>'
                f'<geom name="link{index}-shape" type="capsule" fromto="0 0 0 {lengths[index]} 0 0" size=".018" mass=".18"/>'
                + content
                + "</body>"
            )
        object_xml = (
            f'<body name="object" pos="{_numbers(initial)}"><freejoint name="object-free"/>'
        )
        for index, shape in enumerate(shapes):
            object_xml += (
                f'<geom name="object-{index}" type="{shape.kind}" size="{_numbers(shape.size)}" '
                f'pos="{_numbers(shape.position)}" mass="{mass / len(shapes)}" contype="2" conaffinity="5" rgba="{color}"/>'
            )
        object_xml += "</body>"
        markers = ""
        for target in targets:
            tint = {"left": ".2 .45 1 .25", "right": "1 .8 .1 .25", "middle": ".95 .95 .95 .25"}[
                target.key
            ]
            quaternion = (math.cos(target.yaw / 2), 0, 0, math.sin(target.yaw / 2))
            outlines = shapes if pushing else (Primitive("box", (*target.half_extents, 0.0005)),)
            for index, shape in enumerate(outlines):
                center = _transform(shape.position, target.position, _rotation(quaternion))
                markers += (
                    f'<site name="target-{target.key}-{index}" type="box" '
                    f'pos="{_numbers((center[0], center[1], 0.001))}" quat="{_numbers(quaternion)}" '
                    f'size="{_numbers((*shape.size[:2], 0.0005))}" rgba="{tint}"/>'
                )
        motors = "".join(
            f'<motor name="{actor}" joint="{joint}" gear="1" ctrllimited="true" ctrlrange="-20 20" forcelimited="true" forcerange="-20 20"/>'
            for actor, joint in zip(actuators, joint_names)
        )
        xml = (
            '<mujoco model="' + key + '"><compiler angle="radian" fusestatic="false"/>'
            '<option timestep=".001" gravity="0 0 -9.81" integrator="Euler" cone="pyramidal" solver="Newton" iterations="128" tolerance="1e-10"/>'
            '<default><joint damping=".2" armature=".003"/><geom contype="1" conaffinity="7" condim="3" friction=".7 .01 .001" solref=".02 1" solimp=".8 .9 .002 .5 2"/></default>'
            '<worldbody><geom name="support" type="plane" size="1 1 .05" contype="4" conaffinity="3"/>'
            + content
            + object_xml
            + markers
            + "</worldbody><actuator>"
            + motors
            + "</actuator></mujoco>"
        )
        scenes.append(
            Scene(
                key,
                kinds,
                xml,
                shapes,
                mass,
                initial,
                targets,
                _goals(zh, en, targets, pushing),
                default,
                joint_names,
                actuators,
                fingers,
                object_label_zh=zh,
                object_label_en=en,
            )
        )
    return tuple(scenes)
