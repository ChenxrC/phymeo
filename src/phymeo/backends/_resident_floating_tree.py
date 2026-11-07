"""Owned ordinary floating-tree declarations for the existing resident batch.

Only static model/input compilation occurs here. Pose-dependent mechanics,
FK and integration belong to the device kernels, never a HOST fallback.
"""

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from ..core.rigid import RigidBodyModel
from ..core.robot import RobotModel
from ..dynamics.floating_robot import FloatingRobotModel, FloatingRobotState
from ..dynamics.rotation import quaternion_matrix
from ..solvers.floating_tree import FloatingRobotInputs
from ._resident_tree_world import _nested, _owned, tree_model_token
from ._world_step_layout import as_fp32, world_key

MAX_FLOATING_JOINTS = 16
FLOATING_COORDINATE_LIMIT = 4096.0


def _integer(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not lower <= value <= upper:
        raise ValueError(f"floating {name} must be an integer in [{lower},{upper}]")
    return int(value)


def _freeze_fields(instance, names):
    for name in names:
        value = getattr(instance, name)
        object.__setattr__(instance, name, _owned(value, value.dtype))
    return instance


def _model_copy(model):
    if not isinstance(model, FloatingRobotModel):
        raise TypeError("FloatingRobotTree needs an ordinary FloatingRobotModel")
    root = RigidBodyModel(model.root.inverse_mass, model.root.inertia_diagonal, model.root.gravity)
    _freeze_fields(root, ("inverse_mass", "inertia_diagonal", "gravity"))
    children = model.children
    if children is not None:
        children = RobotModel(children.parents, children.joint_types, children.joint_position,
                              children.joint_orientation, children.joint_axis, children.mass,
                              children.com, children.inertia_diagonal, children.gravity)
        _freeze_fields(children, ("parents", "joint_position", "joint_orientation", "joint_axis",
                                  "mass", "com", "inertia_diagonal", "gravity"))
        if not np.array_equal(children.gravity, root.gravity):
            raise ValueError("floating root and children must share world gravity")
    owned = FloatingRobotModel(root, model.root_com_m, children,
                               root_inertia_frame_xyzw=model.root_inertia_frame_xyzw)
    # Preserve the original validated declaration token: do not repeatedly round
    # an already normalized principal frame when copying or compiling.
    object.__setattr__(owned, "root_inertia_frame_xyzw", _owned(model.root_inertia_frame_xyzw, np.float64))
    return _freeze_fields(owned, ("root_com_m", "root_inertia_frame_xyzw"))


def floating_model_token(model):
    """Original FP64 physical identity, independent of initial pose and controls."""
    if not isinstance(model, FloatingRobotModel):
        raise TypeError("floating model identity needs FloatingRobotModel")
    return ("floating_model_v1", _nested(model.root.inverse_mass),
            _nested(model.root.inertia_diagonal), _nested(model.root.gravity),
            _nested(model.root_com_m), _nested(model.root_inertia_frame_xyzw),
            () if model.children is None else tree_model_token(model.children))


def _coordinate(value, name, shape):
    result = as_fp32(value, name, shape)
    if np.any(abs(result) > FLOATING_COORDINATE_LIMIT):
        raise ValueError(f"{name} exceeds the floating coordinate domain +/-4096")
    return result


def _quat(value):
    out = as_fp32(value, "floating orientation", (4,))
    if abs(float(out.astype(float) @ out.astype(float)) - 1.0) > 1e-5:
        raise ValueError("floating orientation must be a unit XYZW quaternion")
    return out


@dataclass(frozen=True, eq=False)
class FloatingTreeInputs:
    """Owned n-by-3 drive and instantaneous full physical COM loads."""

    drive: np.ndarray
    external: FloatingRobotInputs

    def __post_init__(self):
        raw = np.asarray(self.drive)
        if raw.ndim != 2 or raw.shape[1] != 3 or not 0 <= len(raw) <= MAX_FLOATING_JOINTS:
            raise ValueError("floating drive must have shape (n,3), 0<=n<=16")
        n = len(raw)
        if not isinstance(self.external, FloatingRobotInputs):
            raise TypeError("floating external must be FloatingRobotInputs")
        values = []
        for name, shape in (("effort", (n,)), ("force", (n + 1, 3)), ("torque", (n + 1, 3))):
            value = as_fp32(getattr(self.external, name), "floating " + name, shape)
            values.append(_owned(value, np.float64))
        external = FloatingRobotInputs(*values)
        _freeze_fields(external, ("effort", "force", "torque"))
        object.__setattr__(self, "drive", _owned(as_fp32(raw, "floating drive", (n, 3))))
        object.__setattr__(self, "external", external)


@dataclass(frozen=True, eq=False)
class FloatingRobotTree:
    """One real free root and an ordinary R/P tree; not a separate runtime."""

    model: FloatingRobotModel
    state: FloatingRobotState
    kp: np.ndarray = field(kw_only=True)
    kd: np.ndarray = field(kw_only=True)
    effort_limit: np.ndarray = field(kw_only=True)
    delay_steps: int = field(default=0, kw_only=True)
    control_max_iterations: int = field(default=512, kw_only=True)

    def __post_init__(self):
        model = _model_copy(self.model)
        n = model.body_count - 1
        if not 0 <= n <= MAX_FLOATING_JOINTS:
            raise ValueError("floating scalar joint count must be in [0,16]")
        if not isinstance(self.state, FloatingRobotState):
            raise TypeError("floating initial state must be FloatingRobotState")
        state = FloatingRobotState(self.state.q, self.state.v)
        if state.q.shape != (n + 7,) or state.v.shape != (n + 6,):
            raise ValueError("floating state must have Q=n+7 and D=n+6 entries")
        _freeze_fields(state, ("q", "v"))
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "state", state)
        for name in ("kp", "kd", "effort_limit"):
            value = as_fp32(getattr(self, name), "floating " + name, (n,))
            if np.any(value < 0):
                raise ValueError("floating gains and effort caps must be nonnegative")
            object.__setattr__(self, name, _owned(value))
        object.__setattr__(self, "delay_steps", _integer(self.delay_steps, "delay", 0, 1))
        object.__setattr__(self, "control_max_iterations",
                           _integer(self.control_max_iterations, "iteration budget", 1, 4096))

    def inputs(self, drive, external=None):
        n = self.model.body_count - 1
        if external is None:
            external = FloatingRobotInputs.zeros(n)
        packet = FloatingTreeInputs(drive, external)
        if len(packet.drive) != n:
            raise ValueError("floating drive size differs from this model")
        return packet


@dataclass(frozen=True)
class FloatingTreeCheckpoint:
    world: int
    topology: tuple
    tick: int
    position: tuple
    velocity: tuple
    applied_effort: tuple
    history: tuple
    control: tuple
    generalized_applied: tuple
    local_origin_m: tuple


@dataclass(frozen=True, eq=False)
class FloatingTreeKinematics:
    tick: int
    joint_types: tuple
    position: np.ndarray
    velocity: np.ndarray
    link_position_m: np.ndarray
    link_rotation: np.ndarray
    com_position_m: np.ndarray
    linear_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray
    joint_count: int
    body_count: int
    velocity_count: int
    configuration_count: int

    def __post_init__(self):
        for name in ("position", "velocity", "link_position_m", "link_rotation",
                     "com_position_m", "linear_velocity_m_s", "angular_velocity_rad_s"):
            dtype = np.float64 if name in ("position", "link_position_m", "com_position_m") else np.float32
            object.__setattr__(self, name, _owned(getattr(self, name), dtype))


@dataclass(frozen=True, eq=False)
class FloatingTreeDiagnostic:
    entry_tick: int
    exit_tick: int
    dt_s: float
    entry_position: np.ndarray
    entry_velocity: np.ndarray
    linear_jacobian: np.ndarray
    angular_jacobian: np.ndarray
    mass: np.ndarray
    bias: np.ndarray
    external: np.ndarray
    drive: np.ndarray
    acceleration: np.ndarray
    iterations: int
    control_residual: float
    equation_residual: float

    def __post_init__(self):
        for name in ("entry_position", "entry_velocity", "linear_jacobian", "angular_jacobian",
                     "mass", "bias", "external", "drive", "acceleration"):
            object.__setattr__(self, name, _owned(getattr(self, name),
                               np.float64 if name == "entry_position" else np.float32))


@dataclass(frozen=True, eq=False)
class FloatingTreeWorldSpec:
    n: int
    delay: int
    control_max_iterations: int
    topology: tuple
    model_token: tuple
    joint_types: tuple
    parents: np.ndarray
    kinds: np.ndarray
    offsets: np.ndarray
    zero_rotation: np.ndarray
    local_axis: np.ndarray
    mass: np.ndarray
    com: np.ndarray
    inertia: np.ndarray
    inertia_frame: np.ndarray
    gravity: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    cap: np.ndarray
    anchor: np.ndarray
    initial_root: np.ndarray
    initial_state: np.ndarray
    initial_drive: np.ndarray
    initial_loads: np.ndarray
    initial_effort: np.ndarray
    initial_generalized: np.ndarray

    @property
    def domain(self):
        return "floating_tree"

    @property
    def kind(self):
        return 9 + self.delay

    @property
    def b(self):
        return self.n + 1

    @property
    def d(self):
        return self.n + 6

    @property
    def q_count(self):
        return self.n + 7

    def checkpoint(self, world, root_row, joint_row, generalized_row, tick, control):
        drive, effort, loads = floating_packet(control, self.n)
        position = np.r_[self.anchor + np.asarray(root_row[:3], dtype=float), root_row[3:7], joint_row[:self.n, 0]]
        velocity = np.r_[root_row[7:13], joint_row[:self.n, 1]]
        return FloatingTreeCheckpoint(world_key(world), self.topology, int(tick), _nested(position),
            _nested(velocity), _nested(joint_row[:self.n, 2]),
            _nested(joint_row[:self.n, 4:7]) if self.delay else (),
            (_nested(drive), _nested(effort), _nested(loads[:, :3]), _nested(loads[:, 3:])),
            _nested(generalized_row[:self.d]), _nested(root_row[:3]))

    def restore_row(self, checkpoint, world):
        if not isinstance(checkpoint, FloatingTreeCheckpoint) or world_key(checkpoint.world) != world:
            raise ValueError("floating checkpoint world or state kind differs")
        if checkpoint.topology != self.topology:
            raise ValueError("floating checkpoint topology/model differs")
        tick = _integer(checkpoint.tick, "checkpoint tick", 0, 2147483647)
        q = np.asarray(checkpoint.position)
        if q.dtype.kind not in "iuf" or q.shape != (self.q_count,) or not np.isfinite(q).all():
            raise ValueError("floating checkpoint configuration has invalid Q")
        v = as_fp32(checkpoint.velocity, "restored floating velocity", (self.d,))
        root = np.zeros(13, np.float32)
        root[:3] = _coordinate(checkpoint.local_origin_m, "restored floating local origin", (3,))
        if not np.array_equal(q[:3], self.anchor + root[:3].astype(np.float64)):
            raise ValueError("floating world origin disagrees with its complete local checkpoint")
        root[3:7], root[7:13] = _quat(q[3:7]), v[:6]
        row = np.zeros((max(1, self.n), 7), np.float32)
        row[:self.n, 0] = _coordinate(q[7:], "restored floating joints", (self.n,))
        row[:self.n, 1] = v[6:]
        row[:self.n, 2] = as_fp32(checkpoint.applied_effort, "restored floating drive", (self.n,))
        applied = as_fp32(checkpoint.generalized_applied, "restored floating generalized load", (self.d,))
        row[:self.n, 3] = applied[6:]
        if self.delay:
            history = np.asarray(checkpoint.history)
            if self.n == 0 and history.shape == (0,):
                history = history.reshape(0, 3)
            row[:self.n, 4:7] = as_fp32(history, "restored floating queue", (self.n, 3))
        elif checkpoint.history != ():
            raise ValueError("undelayed floating checkpoint cannot contain a queue")
        if not isinstance(checkpoint.control, tuple) or len(checkpoint.control) != 4:
            raise ValueError("floating checkpoint control must contain all four blocks")
        drive, effort, force, torque = checkpoint.control
        drive = np.asarray(drive, dtype=float)
        if self.n == 0 and drive.shape == (0,):
            drive = drive.reshape(0, 3)
        if drive.shape != (self.n, 3):
            raise ValueError("restored floating drive must retain its (n,3) block")
        packet = FloatingTreeInputs(drive, FloatingRobotInputs(np.asarray(effort, dtype=float),
                                   np.asarray(force, dtype=float), np.asarray(torque, dtype=float)))
        floating_packet(packet, self.n)
        return _owned(root), _owned(row), _owned(applied), packet, tick


def floating_packet(packet, n):
    if not isinstance(packet, FloatingTreeInputs):
        raise TypeError("floating control must be FloatingTreeInputs")
    drive = as_fp32(packet.drive, "floating drive", (n, 3))
    effort = as_fp32(packet.external.effort, "floating effort", (n,))
    force = as_fp32(packet.external.force, "floating COM force", (n + 1, 3))
    torque = as_fp32(packet.external.torque, "floating COM torque", (n + 1, 3))
    return _owned(drive), _owned(effort), _owned(np.column_stack((force, torque)))


def compile_floating_tree_world(spec):
    if not isinstance(spec, FloatingRobotTree):
        raise TypeError("expected FloatingRobotTree")
    model, state = _model_copy(spec.model), spec.state
    n, b = model.body_count - 1, model.body_count
    parents, kinds = np.full(b, -1, np.int32), np.full(b, 2, np.int32)
    offsets, axes = np.zeros((b, 3)), np.zeros((b, 3))
    zero = np.repeat(np.eye(3)[None], b, axis=0)
    frames = zero.copy()
    mass, com, inertia = np.empty(b), np.empty((b, 3)), np.empty((b, 3))
    mass[0], com[0], inertia[0] = 1 / model.root.inverse_mass[0], model.root_com_m, model.root.inertia_diagonal[0]
    frames[0] = quaternion_matrix(model.root_inertia_frame_xyzw[None])[0]
    types = ()
    if n:
        child = model.children
        types = tuple(child.joint_types)
        parents[1:], kinds[1:] = child.parents + 1, [0 if t == "revolute" else 1 for t in types]
        offsets[1:], axes[1:] = child.joint_position, child.joint_axis
        zero[1:] = quaternion_matrix(child.joint_orientation)
        mass[1:], com[1:], inertia[1:] = child.mass, child.com, child.inertia_diagonal
    anchor = _owned(state.q[:3], np.float64)
    as_fp32(anchor, "floating world anchor", (3,))
    root = np.zeros(13, np.float32)
    root[3:7] = _quat(state.q[3:7])
    root[7:13] = as_fp32(state.v[:6], "floating root velocity", (6,))
    row = np.zeros((max(1, n), 7), np.float32)
    row[:n, 0] = _coordinate(state.q[7:], "initial floating joints", (n,))
    row[:n, 1] = as_fp32(state.v[6:], "initial floating rates", (n,))
    drive = np.zeros((n, 3), np.float32)
    drive[:, 0] = row[:n, 0]
    if spec.delay_steps:
        row[:n, 4:7] = drive
    token = floating_model_token(model)
    topology = ("floating_tree_v1", token, _nested(anchor), *(_nested(x) for x in
                (spec.kp, spec.kd, spec.effort_limit)), spec.delay_steps, spec.control_max_iterations)
    def packed(value, name):
        return _owned(as_fp32(value, "floating " + name, value.shape))
    return FloatingTreeWorldSpec(n, spec.delay_steps, spec.control_max_iterations, topology,
        token, types, _owned(parents, np.int32), _owned(kinds, np.int32), packed(offsets, "offsets"),
        packed(zero, "joint rotations"), packed(axes, "joint axes"), packed(mass, "mass"),
        packed(com, "COM"), packed(inertia, "inertia"), packed(frames, "inertia frames"),
        packed(model.root.gravity, "gravity"), _owned(spec.kp), _owned(spec.kd), _owned(spec.effort_limit),
        anchor, _owned(root), _owned(row), _owned(drive), _owned(np.zeros((b, 6))),
        _owned(np.zeros(n)), _owned(np.zeros(n + 6)))
