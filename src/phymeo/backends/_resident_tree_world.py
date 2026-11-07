"""Owned ordinary fixed-tree specifications for the existing TYX world batch.

Compilation is static/initial-state only. No current FK, mass solve, device
runtime, control advance, or accepted-state publication occurs in this module.
"""

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..core.robot import RobotInputs, RobotModel, RobotState, validate_robot_step
from ..core.scene import SceneInputs, SceneState
from ..dynamics.rotation import quaternion_matrix
from ..solvers.driven_robot import DrivenRobotProgram
from ._world_step_layout import as_fp32, world_key

MAX_TREE_JOINTS = 16
TREE_COORDINATE_LIMIT = 4096.0
TREE_STATE_WIDTH = 7
TREE_COMMAND_WIDTH = 10
TREE_FK_WIDTH = 18


def _owned(value, dtype=np.float32):
    a = np.asarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(order="C"), dtype=dtype).reshape(a.shape)


def _nested(value):
    a = np.asarray(value)
    if a.ndim == 1:
        return tuple(float(x) for x in a)
    return tuple(_nested(x) for x in a)


def tree_model_token(model):
    """Full authored physical model identity, independent of controls and assets."""
    if not isinstance(model, RobotModel):
        raise TypeError("resident tree needs an ordinary RobotModel")
    return (
        tuple(int(x) for x in model.parents),
        tuple(model.joint_types),
        *(
            _nested(getattr(model, name))
            for name in (
                "joint_position",
                "joint_orientation",
                "joint_axis",
                "mass",
                "com",
                "inertia_diagonal",
                "gravity",
            )
        ),
    )


def _coordinate(value, name, n):
    a = as_fp32(value, name, (n,))
    if np.any(np.abs(a) > TREE_COORDINATE_LIMIT):
        raise ValueError(f"{name} exceeds the resident tree coordinate domain +/-4096")
    return a


@dataclass(frozen=True)
class FixedRobotTree:
    """One full coupled R/P tree, with zero/one physical-substep drive delay.

    Limits, mimic coordinates and floating bases require different physical laws
    and are rejected. This is a world specification, not a runtime of its own.
    """

    program: DrivenRobotProgram
    state: SceneState
    control_max_iterations: int = 512

    def __post_init__(self):
        if not isinstance(self.program, DrivenRobotProgram):
            raise TypeError("FixedRobotTree requires an ordinary DrivenRobotProgram")
        if not isinstance(self.state, SceneState):
            raise TypeError("FixedRobotTree requires the program's complete SceneState")
        budget = self.control_max_iterations
        if (
            isinstance(budget, (bool, np.bool_))
            or not isinstance(budget, Integral)
            or not 1 <= budget <= 4096
        ):
            raise ValueError("tree control iterations must be an integer in [1,4096]")
        object.__setattr__(self, "control_max_iterations", int(budget))


@dataclass(frozen=True)
class TreeWorldCheckpoint:
    world: int
    topology: tuple
    tick: int
    position: tuple
    velocity: tuple
    applied_effort: tuple
    history: tuple
    control: tuple
    generalized_applied: tuple


@dataclass(frozen=True, eq=False)
class TreeKinematics:
    tick: int
    joint_types: tuple
    position: np.ndarray
    velocity: np.ndarray
    link_position_m: np.ndarray
    link_rotation: np.ndarray
    com_position_m: np.ndarray

    def __post_init__(self):
        for name in ("position", "velocity", "link_position_m", "link_rotation", "com_position_m"):
            object.__setattr__(self, name, _owned(getattr(self, name)))


@dataclass(frozen=True, eq=False)
class TreeDiagnostic:
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
        for name in (
            "entry_position",
            "entry_velocity",
            "linear_jacobian",
            "angular_jacobian",
            "mass",
            "bias",
            "external",
            "drive",
            "acceleration",
        ):
            object.__setattr__(self, name, _owned(getattr(self, name)))


@dataclass(frozen=True, eq=False)
class TreeWorldSpec:
    domain: str
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
    gravity: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    cap: np.ndarray
    anchor: np.ndarray
    initial_state: np.ndarray
    initial_command: np.ndarray

    @property
    def kind(self):
        return 5 + self.delay

    def checkpoint(self, world, row, tick, held):
        n = self.n
        row = np.asarray(row)
        return TreeWorldCheckpoint(
            world_key(world),
            self.topology,
            int(tick),
            _nested(row[:n, 0]),
            _nested(row[:n, 1]),
            _nested(row[:n, 2]),
            _nested(row[:n, 4:7]) if self.delay else (),
            _nested(np.asarray(held)[:n]),
            _nested(row[:n, 3]),
        )

    def restore_row(self, checkpoint, world):
        if not isinstance(checkpoint, TreeWorldCheckpoint) or checkpoint.world != world:
            raise ValueError("tree checkpoint belongs to a different world or state kind")
        if checkpoint.topology != self.topology:
            raise ValueError("tree checkpoint topology/model does not match")
        t = checkpoint.tick
        if (
            isinstance(t, (bool, np.bool_))
            or not isinstance(t, Integral)
            or not 0 <= t <= 2147483647
        ):
            raise ValueError("tree checkpoint tick is outside the i32 domain")
        row = np.zeros((self.n, TREE_STATE_WIDTH), np.float32)
        row[:, 0] = _coordinate(checkpoint.position, "restored tree coordinate", self.n)
        for column, name in ((1, "velocity"), (2, "applied_effort"), (3, "generalized_applied")):
            row[:, column] = as_fp32(getattr(checkpoint, name), "restored tree " + name, (self.n,))
        if self.delay:
            row[:, 4:7] = as_fp32(checkpoint.history, "restored tree history", (self.n, 3))
        elif checkpoint.history != ():
            raise ValueError("undelayed tree checkpoint cannot contain a history queue")
        held = as_fp32(checkpoint.control, "restored tree control", (self.n, TREE_COMMAND_WIDTH))
        return row, held, int(t)


def tree_packet(inputs, domain, n):
    """Own the full current packet; only the first three columns are delayed."""
    if not isinstance(inputs, SceneInputs) or set(inputs.domains) != {domain}:
        raise ValueError("tree command must use its bound drive domain")
    action = inputs.domains[domain]
    if not isinstance(action, RobotInputs) or not all(
        hasattr(action, name) for name in ("target", "target_velocity", "feedforward")
    ):
        raise TypeError("tree command needs target/rate/feedforward and RobotInputs loads")
    out = np.empty((n, TREE_COMMAND_WIDTH), np.float32)
    for column, name in enumerate(("target", "target_velocity", "feedforward", "effort")):
        out[:, column] = as_fp32(getattr(action, name), "tree " + name, (n,))
    out[:, 4:7] = as_fp32(action.force, "tree COM force", (n, 3))
    out[:, 7:10] = as_fp32(action.torque, "tree COM torque", (n, 3))
    return out


def compile_tree_physical(model):
    """Own the physical N-row layout without a drive or coordinate policy.

    Both ordinary and affine worlds retain this complete mass/geometry model.
    This function only compiles static data; it never creates a program/runtime,
    evaluates current dynamics or changes accepted state.
    """
    if not isinstance(model, RobotModel):
        raise TypeError("resident tree requires an ordinary RobotModel")
    model = RobotModel(
        model.parents,
        model.joint_types,
        model.joint_position,
        model.joint_orientation,
        model.joint_axis,
        model.mass,
        model.com,
        model.inertia_diagonal,
        model.gravity,
    )
    n = model.dof_count
    if not 1 <= n <= MAX_TREE_JOINTS:
        raise ValueError("resident tree joint count must be in [1,16]")
    # Every root retains its real installation relative to one fixed anchor.
    anchor64 = np.array(model.joint_position[0], copy=True)
    offsets = np.array(model.joint_position, copy=True)
    offsets[model.parents < 0] -= anchor64
    return MappingProxyType(
        dict(
            n=n,
            model_token=tree_model_token(model),
            joint_types=model.joint_types,
            parents=_owned(model.parents, np.int32),
            kinds=_owned([0 if k == "revolute" else 1 for k in model.joint_types], np.int32),
            offsets=_owned(as_fp32(offsets, "tree offsets relative to anchor", (n, 3))),
            zero_rotation=_owned(
                as_fp32(
                    quaternion_matrix(model.joint_orientation), "tree zero rotations", (n, 3, 3)
                )
            ),
            local_axis=_owned(as_fp32(model.joint_axis, "tree axes", (n, 3))),
            mass=_owned(as_fp32(model.mass, "tree masses", (n,))),
            com=_owned(as_fp32(model.com, "tree COM", (n, 3))),
            inertia=_owned(as_fp32(model.inertia_diagonal, "tree inertia", (n, 3))),
            gravity=_owned(as_fp32(model.gravity, "tree gravity", (3,))),
            anchor=_owned(as_fp32(anchor64, "tree anchor", (3,))),
        )
    )


def compile_tree_world(spec):
    if not isinstance(spec, FixedRobotTree):
        raise TypeError("expected FixedRobotTree")
    program, initial = spec.program, spec.state
    physical = dict(compile_tree_physical(program.model))
    n = physical.pop("n")
    model_token = physical.pop("model_token")
    joint_types = physical.pop("joint_types")
    if program.delay_steps not in (0, 1):
        raise ValueError("resident tree supports delay 0 or 1 only")
    if program.limits is not None:
        raise ValueError("resident tree joint stops/limits are not supported")
    domain = program.domain
    if set(initial.domains) != {domain} or not isinstance(initial.domains[domain], RobotState):
        raise ValueError("resident tree needs its complete ordinary SceneState")
    state = initial.domains[domain]
    validate_robot_step(program.model, state, RobotInputs.zeros(n))
    row = np.zeros((n, TREE_STATE_WIDTH), np.float32)
    row[:, 0] = _coordinate(state.q, "initial tree coordinate", n)
    row[:, 1] = as_fp32(state.v, "initial tree velocity", (n,))
    queue_name = domain + "/drive_queue"
    if set(initial.history) != ({queue_name} if program.delay_steps else set()):
        raise ValueError("initial tree state has an incompatible history queue")
    if program.delay_steps:
        row[:, 4:7] = as_fp32(initial.history[queue_name], "initial tree history", (1, 3, n))[0].T
    command = np.zeros((n, TREE_COMMAND_WIDTH), np.float32)
    command[:, 0] = row[:, 0]
    gains = [
        as_fp32(getattr(program, name), "tree " + name, (n,))
        for name in ("kp", "kd", "effort_limit")
    ]
    if any(np.any(value < 0) for value in gains):
        raise ValueError("tree gains and effort caps must be nonnegative")
    topology = (
        "fixed_tree_v1",
        domain,
        n,
        model_token,
        *(_nested(x) for x in gains),
        int(program.delay_steps),
        spec.control_max_iterations,
    )
    return TreeWorldSpec(
        domain,
        n,
        int(program.delay_steps),
        spec.control_max_iterations,
        topology,
        model_token,
        joint_types,
        **physical,
        kp=_owned(gains[0]),
        kd=_owned(gains[1]),
        cap=_owned(gains[2]),
        initial_state=_owned(row),
        initial_command=_owned(command),
    )
