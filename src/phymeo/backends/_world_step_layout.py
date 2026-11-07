"""Owned HOST bindings and checkpoints for the supported resident TYX world laws."""

from dataclasses import dataclass
from math import fsum
from numbers import Integral

import numpy as np

from ..core.robot import RobotInputs, validate_robot_step
from ..core.scene import SceneInputs, SceneState
from ..dynamics.rotation import quaternion_matrix
from ..solvers.driven_robot import DrivenRobotProgram
from ._resident_slider_binding import SliderGeometryFrame

# State columns are xyz (or scalar q), velocity, last applied effort and delayed packet.
STATE_WIDTH = 12
# Per-world constants are inverse mass, xyz gravity, kp, kd, effort limit, radius and mass.
PROPERTY_WIDTH = 9
MAX_TICK = int(np.iinfo(np.int32).max)


def as_fp32(value, name, shape):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape != shape:
        raise ValueError(f"{name} must be a real array of shape {shape}")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.array(raw, dtype=np.float32, order="C", copy=True)
        lost = (raw != 0) & (np.abs(result) < np.finfo(np.float32).tiny)
    if not np.isfinite(result).all() or np.any(lost):
        raise ValueError(f"{name} must be finite and representable as normal FP32 or zero")
    return result


def world_key(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError("world key must be a nonnegative integer")
    return int(value)


def _values(row):
    return tuple(float(x) for x in row)


@dataclass(frozen=True)
class FreeSphere:
    """Freely translating solid sphere under gravity and a world-space force at its COM.

    Initial spin is zero. Gravity is uniform. Contact response and torques are outside
    this law. Radius is also used to observe the lowest point of the actual sphere.
    """

    mass: float
    radius: float
    position: tuple = (0.0, 0.0, 0.0)
    velocity: tuple = (0.0, 0.0, 0.0)
    gravity: tuple = (0.0, 0.0, -9.81)

    def __post_init__(self):
        for name in ("mass", "radius"):
            value = float(as_fp32(getattr(self, name), name, ()))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("position", "velocity", "gravity"):
            object.__setattr__(self, name, _values(as_fp32(getattr(self, name), name, (3,))))


@dataclass(frozen=True)
class SpherePlaneWorld(FreeSphere):
    """A translating sphere above its own fixed horizontal (+Z) frictionless half-space.

    Restitution acts on incoming free velocity exceeding twice the external normal
    velocity change in this step. Slower contacts settle. A sphere outside the
    plane's end-step candidate receives no impulse. Radius and height are in m.
    """

    plane_height: float = 0.0
    restitution: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        height = float(as_fp32(self.plane_height, "plane height", ()))
        e = float(as_fp32(self.restitution, "normal restitution", ()))
        if not 0 <= float(self.restitution) <= 1:
            raise ValueError("normal restitution must lie in [0, 1]")
        object.__setattr__(self, "plane_height", height)
        object.__setattr__(self, "restitution", e)

    def center_stop(self):
        """Lowest supported FP32 center on or above the actual sphere-plane contact."""
        level = as_fp32(self.plane_height + self.radius, "sphere contact height", ())
        if fsum((float(level), -self.plane_height, -self.radius)) < 0:
            with np.errstate(over="ignore", under="ignore"):
                level = np.nextafter(level, np.float32(np.inf))
        return float(as_fp32(level, "outward sphere contact height", ()))


@dataclass(frozen=True)
class PrismaticSlider:
    """A fixed-base, one-link DrivenRobotProgram and its initial physical/history state."""

    program: DrivenRobotProgram
    state: SceneState


@dataclass(frozen=True)
class WorldCheckpoint:
    """Accepted physical state, delayed packet and held input, keyed to a world and its law."""

    world: int
    topology: tuple
    tick: int
    position: tuple
    velocity: tuple
    applied_effort: tuple
    history: tuple
    control: tuple


@dataclass(frozen=True)
class SpherePlaneCheckpoint(WorldCheckpoint):
    """Last accepted upward plane impulse in N s, separate from the held external force."""

    normal_impulse_n_s: float


@dataclass(frozen=True)
class WorldBinding:
    world: int
    kind: int  # 0 sphere, 1/2 slider, 3/4 revolute; even positive kinds delay one step
    domain: str
    axis: tuple
    topology: tuple
    plane_height: float | None = None
    slider_frame: SliderGeometryFrame | None = None
    rotor_frame: object | None = None
    tree_spec: object | None = None

    @property
    def width(self):
        if self.tree_spec is not None:
            raise ValueError("tree worlds require typed tree kinematics")
        return 1 if self.kind else 3

    @property
    def command_width(self):
        return 10 if self.kind in (3, 4) else 4

    def packet(self, value):
        if self.tree_spec is not None:
            raise ValueError("tree controls require the tree storage packet")
        if self.kind in (3, 4):
            from ._resident_revolute_world import revolute_packet

            return revolute_packet(value, self.domain)
        if not self.kind:
            return (*as_fp32(value, "sphere COM force", (3,)), 0.0)
        if not isinstance(value, SceneInputs) or set(value.domains) != {self.domain}:
            raise ValueError("slider requires inputs from its bound drive domain")
        action = value.domains[self.domain]
        if not isinstance(action, RobotInputs) or not hasattr(action, "target"):
            raise ValueError("slider requires a target, velocity and feedforward drive packet")
        target = [
            as_fp32(getattr(action, name), name, (1,))[0]
            for name in ("target", "target_velocity", "feedforward")
        ]
        effort = as_fp32(action.effort, "generalized load", (1,))[0]
        force = as_fp32(action.force, "world COM force", (1, 3))[0]
        as_fp32(action.torque, "world COM torque", (1, 3))
        # A pure prismatic joint has zero angular Jacobian. Only axial force does work.
        return as_fp32([*target, float(effort) + np.dot(self.axis, force)], "slider command", (4,))

    def checkpoint(self, row, tick, control):
        if self.tree_spec is not None:
            raise ValueError("tree checkpoint requires its complete tree bank")
        n = self.width
        args = (
            self.world,
            self.topology,
            int(tick),
            _values(row[:n]),
            _values(row[3 : 3 + n]),
            _values(row[6 : 6 + n]),
            _values(row[9:12]) if self.kind in (2, 4) else (),
            _values(control[: self.command_width if self.kind else 3]),
        )

        if self.plane_height is not None:
            return SpherePlaneCheckpoint(*args, float(row[9]))
        return WorldCheckpoint(*args)

    def unpack(self, point):
        if self.tree_spec is not None:
            raise ValueError("tree restore requires its complete tree bank")
        if (
            not isinstance(point, WorldCheckpoint)
            or point.world != self.world
            or point.topology != self.topology
        ):
            raise ValueError("checkpoint belongs to a different world or topology")
        if (
            isinstance(point.tick, (bool, np.bool_))
            or not isinstance(point.tick, Integral)
            or not 0 <= point.tick <= MAX_TICK
        ):
            raise ValueError("checkpoint tick is outside the i32 range")
        state = np.zeros(STATE_WIDTH, np.float32)
        for start, value in ((0, point.position), (3, point.velocity), (6, point.applied_effort)):
            state[start : start + self.width] = as_fp32(value, "checkpoint state", (self.width,))
        if self.kind in (3, 4):
            from ._resident_revolute_world import validate_revolute_angle

            validate_revolute_angle(state[0], "checkpoint angle")
        count = 3 if self.kind in (2, 4) else 0
        state[9 : 9 + count] = as_fp32(point.history, "checkpoint history", (count,))
        if self.plane_height is not None:
            if not isinstance(point, SpherePlaneCheckpoint):
                raise ValueError("sphere-plane checkpoint requires a normal impulse")
            impulse = as_fp32(point.normal_impulse_n_s, "normal impulse", ())
            if impulse < 0:
                raise ValueError("normal impulse must be nonnegative")
            state[9] = impulse
        n = self.command_width if self.kind else 3
        controls = np.zeros(self.command_width, np.float32)
        controls[:n] = as_fp32(point.control, "checkpoint held control", (n,))
        return state, int(point.tick), controls


def bind_world(world, spec):
    from ._resident_revolute_world import RevoluteRotor, compile_revolute_world
    from ._resident_tree_world import FixedRobotTree, compile_tree_world

    if isinstance(spec, FixedRobotTree):
        tree = compile_tree_world(spec)
        binding = WorldBinding(world, tree.kind, tree.domain, (), tree.topology, tree_spec=tree)
        return (
            binding,
            np.zeros(PROPERTY_WIDTH, np.float32),
            np.zeros(STATE_WIDTH, np.float32),
            np.zeros(4, np.float32),
        )

    state = np.zeros(STATE_WIDTH, np.float32)
    command = np.zeros(4, np.float32)
    axis, domain = (), ""
    plane_height = None
    slider_frame = None
    rotor_frame = None
    if isinstance(spec, FreeSphere):
        kind = 0
        plane = (0, 0, 0)
        if isinstance(spec, SpherePlaneWorld):
            plane = (spec.center_stop(), spec.restitution, 1)
            plane_height = spec.plane_height
        props = (1 / spec.mass, *spec.gravity, *plane, spec.radius, spec.mass)
        state[:3], state[3:6] = spec.position, spec.velocity
    elif isinstance(spec, PrismaticSlider):
        p = spec.program
        if (
            not isinstance(p, DrivenRobotProgram)
            or p.model.joint_types != ("prismatic",)
            or p.limits is not None
        ):
            raise ValueError(
                "resident slider requires one fixed-base prismatic joint without stops"
            )
        if p.delay_steps not in (0, 1):
            raise ValueError("resident slider supports zero or one substep of command delay")
        initial, domain = spec.state, p.domain
        if not isinstance(initial, SceneState) or set(initial.domains) != {domain}:
            raise ValueError("slider state must match its drive domain")
        joint = initial.domains[domain]
        validate_robot_step(p.model, joint, RobotInputs.zeros(1))
        history_key = domain + "/drive_queue"
        if set(initial.history) != ({history_key} if p.delay_steps else set()):
            raise ValueError("slider state requires its drive history")
        if p.delay_steps:
            state[9:12] = as_fp32(
                initial.history[history_key], "initial delay queue", (1, 3, 1)
            ).ravel()
        state[0] = as_fp32(joint.q, "initial q", (1,))[0]
        state[3] = as_fp32(joint.v, "initial qd", (1,))[0]
        command[0] = state[0]
        local_axis = p.model.joint_axis[0]
        axis = _values(
            quaternion_matrix(p.model.joint_orientation)[0]
            @ (local_axis / np.linalg.norm(local_axis))
        )
        slider_frame = SliderGeometryFrame(
            p.model.joint_position[0], p.model.joint_orientation[0], local_axis
        )
        mass = float(p.model.mass[0])
        axial_gravity = np.dot(axis, p.model.gravity)
        props = (1 / mass, axial_gravity, 0, 0, p.kp[0], p.kd[0], p.effort_limit[0], 0, mass)
        kind = 1 + p.delay_steps
    elif isinstance(spec, RevoluteRotor):
        kind, domain, axis, rotor_frame, props, state, command = compile_revolute_world(spec)
    else:
        raise TypeError("expected FreeSphere, PrismaticSlider, RevoluteRotor or FixedRobotTree")
    width = 16 if kind in (3, 4) else PROPERTY_WIDTH
    constants = as_fp32(props, "world topology", (width,))
    topology = (kind, domain, axis, _values(constants))
    if plane_height is not None:
        topology += ("horizontal-sphere-plane-v1", plane_height)
    if rotor_frame is not None:
        topology += (
            "single-revolute-v1",
            tuple(rotor_frame.origin_m),
            tuple(rotor_frame.orientation_xyzw),
            tuple(rotor_frame.axis_local),
        )
    binding = WorldBinding(
        world, kind, domain, axis, topology, plane_height, slider_frame, rotor_frame
    )
    return binding, constants, state, command
