"""Compile a real one-revolute robot into the existing resident world layout.

No runtime or device allocation lives here. The batch owns checkpointing,
commands, delayed state, accepted/candidate buffers and publication.
"""

from dataclasses import dataclass

import numpy as np

from ..core.robot import RobotInputs, RobotModel, RobotState, validate_robot_step
from ..core.scene import SceneInputs, SceneState
from ..dynamics.rotation import quaternion_matrix
from ..solvers.driven_robot import DrivenRobotProgram
from ._resident_revolute_binding import RevoluteGeometryFrame
from ._world_step_layout import as_fp32

REVOLUTE_ANGLE_LIMIT_RAD = 4096.0
REVOLUTE_PROPERTY_WIDTH = 16
REVOLUTE_COMMAND_WIDTH = 10


@dataclass(frozen=True)
class RevoluteRotor:
    """One ordinary fixed-base revolute DrivenRobotProgram and its initial state.

    A positive physical axis inertia, offset COM and installation come from the
    model. Supports zero/one-substep command delay, no compliant joint stops.
    q is in radians, qd in radians/s, actuator/generalized efforts in N m.
    """

    program: DrivenRobotProgram
    state: SceneState


def validate_revolute_angle(value, name):
    """Return the checked FP32 scalar angle in the supported stored domain.

    The inclusive range concerns the actual stored angle, without wrapping.
    Nonfinite, nonrepresentable and nonzero subnormal values follow the existing
    resident as_fp32 contract. Targets are controls, not accepted physical q.
    """
    angle = as_fp32(value, name, ())
    if abs(float(angle)) > REVOLUTE_ANGLE_LIMIT_RAD:
        raise ValueError(f"{name} lies outside the resident angle domain [-4096, 4096] rad")
    return np.float32(angle)


def revolute_packet(value, domain):
    """Copy target/rate/ff, generalized effort, world COM force and COM torque.

    The last seven columns remain instantaneous independent loads. They are not
    projected at HOST packet construction and never enter the delayed triplet.
    """
    if not isinstance(value, SceneInputs) or set(value.domains) != {domain}:
        raise ValueError("revolute rotor requires inputs from its bound drive domain")
    action = value.domains[domain]
    if not isinstance(action, RobotInputs) or not all(hasattr(action, name) for name in ("target", "target_velocity", "feedforward")):
        raise ValueError("revolute rotor requires a target, velocity and feedforward drive packet")
    drive = [as_fp32(getattr(action, name), name, (1,))[0]
             for name in ("target", "target_velocity", "feedforward")]
    effort = as_fp32(action.effort, "revolute generalized load", (1,))[0]
    force = as_fp32(action.force, "revolute world COM force", (1, 3))[0]
    torque = as_fp32(action.torque, "revolute world COM torque", (1, 3))[0]
    return as_fp32([*drive, effort, *force, *torque], "revolute command", (REVOLUTE_COMMAND_WIDTH,))


def _owned_model(model):
    if not isinstance(model, RobotModel):
        raise TypeError("resident revolute requires an ordinary RobotModel")
    # Revalidate and detach: public RobotModel arrays can originate outside this
    # batch and must not remain live property storage after compilation.
    return RobotModel(model.parents, model.joint_types, model.joint_position,
                      model.joint_orientation, model.joint_axis, model.mass, model.com,
                      model.inertia_diagonal, model.gravity)


def compile_revolute_world(spec):
    """Return kind, domain, axis tuple, frame, props[16], state[12], command[10].

    props = invI, gravityXYZ, kp,kd,cap,unused,I, mass, zero_COM_armXYZ,
    axis_worldXYZ. The two geometry directions use the original frame/pivot.
    All arrays are new HOST storage ready for the owner's initial device upload.
    """
    if not isinstance(spec, RevoluteRotor):
        raise TypeError("expected RevoluteRotor")
    program = spec.program
    if not isinstance(program, DrivenRobotProgram):
        raise TypeError("resident revolute requires DrivenRobotProgram")
    model = _owned_model(program.model)
    if model.joint_types != ("revolute",) or tuple(model.parents) != (-1,) or program.limits is not None:
        raise ValueError("resident revolute requires one root revolute joint without stops")
    if program.delay_steps not in (0, 1):
        raise ValueError("resident revolute supports zero or one substep of command delay")
    initial, domain = spec.state, program.domain
    if not isinstance(initial, SceneState) or set(initial.domains) != {domain}:
        raise ValueError("revolute state must match its drive domain")
    physical = initial.domains[domain]
    if not isinstance(physical, RobotState):
        raise TypeError("resident revolute state requires a RobotState domain")
    validate_robot_step(model, physical, RobotInputs.zeros(1))
    queue_name = domain + "/drive_queue"
    if set(initial.history) != ({queue_name} if program.delay_steps else set()):
        raise ValueError("revolute state requires exactly its drive history")
    frame = RevoluteGeometryFrame(model.joint_position[0], model.joint_orientation[0], model.joint_axis[0])
    rotation = quaternion_matrix(np.array([frame.orientation_xyzw]))[0]
    local_axis = np.asarray(frame.axis_local)
    axis = rotation @ local_axis
    com = model.com[0]
    mass = float(model.mass[0])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        cross = np.cross(local_axis, com)
        inertia = float(np.dot(local_axis * model.inertia_diagonal[0], local_axis) + mass * np.dot(cross, cross))
        arm = rotation @ com
    if not np.isfinite(inertia) or inertia <= 0:
        raise ValueError("revolute axis inertia must be finite and positive")
    gains = [as_fp32(getattr(program, name), name, (1,))[0]
             for name in ("kp", "kd", "effort_limit")]
    if any(value < 0 for value in gains):
        raise ValueError("revolute gains and effort limit must be nonnegative")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        constants = as_fp32([1 / inertia, *model.gravity, *gains, 0.0, inertia, mass, *arm, *axis],
                            "revolute world topology", (REVOLUTE_PROPERTY_WIDTH,))
    if constants[0] <= 0 or constants[8] <= 0 or constants[9] <= 0:
        raise ValueError("revolute mass and axis inertia must remain positive in FP32")
    state = np.zeros(12, dtype=np.float32)
    state[0] = validate_revolute_angle(physical.q[0], "initial revolute q")
    state[3] = as_fp32(physical.v, "initial revolute qd", (1,))[0]
    if program.delay_steps:
        state[9:12] = as_fp32(initial.history[queue_name], "initial revolute delay queue", (1, 3, 1)).ravel()
    command = np.zeros(REVOLUTE_COMMAND_WIDTH, dtype=np.float32)
    command[0] = state[0]
    return (3 + program.delay_steps, domain, tuple(float(x) for x in axis), frame,
            constants, state, command)
