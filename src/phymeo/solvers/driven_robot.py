"""Fixed-base robot dynamics with compliant stops and an implicit, delayed PD command."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..constraints.robot_contact import JointLimits
from ..control.bounded_pd import solve_bounded_pd
from ..core.robot import RobotInputs, RobotModel, RobotState, robot_array, validate_robot_step
from ..core.scene import (
    DomainBinding,
    HistoryField,
    SceneInputs,
    SceneModel,
    SceneState,
    StepSettings,
)
from ..dynamics.robot_dynamics import _numerical, bias_forces, generalized_forces, mass_matrix
from ..runtime.program import ProgramCapabilities
from ..runtime.registry import StateFamily, StateRegistry
from ..runtime.scene_stepper import SceneStepper
from .robot import RobotWorkspace


def _vector(value, name, n):
    array = robot_array(value, name, ndim=1)
    if array.shape != (n,):
        raise ValueError(f"{name} needs one value per robot DOF")
    return array


@dataclass(eq=False)
class _DriveInputs(RobotInputs):
    target: np.ndarray
    target_velocity: np.ndarray
    feedforward: np.ndarray

    def __post_init__(self):
        super().__post_init__()
        n = len(self.effort)
        for name in ("target", "target_velocity", "feedforward"):
            setattr(self, name, _vector(getattr(self, name), name, n))


class DrivenRobotProgram:
    """HOST semi-implicit physical substeps with separately reported actuator and stop impulses.

    Each substep samples the target position, target velocity and feedforward as one packet.
    A delay d uses the packet sampled d physical substeps earlier. By default the initial
    history holds the starting position with zero target velocity and feedforward. Feedback,
    compliant stops and independent external loads always use the current physical state.

    The bounded PD solves for the end-of-substep free velocity including stop and external
    loads. Only the actuator effort is clipped. The held actuator effort and stop effort are
    applied to the robot integrator, which publishes the actual q and v and advances both the
    physical state and command queue transactionally.
    """

    capabilities = ProgramCapabilities(
        "numpy",
        {"float64"},
        {"driven_robot"},
        {"fixed_base_tree", "joint_stops", "delayed_pd"},
        history="explicit",
    )

    def __init__(self, model, *, kp, kd, effort_limit, limits=None, delay_steps=0, domain="robot"):
        if not isinstance(model, RobotModel):
            raise TypeError("DrivenRobotProgram requires RobotModel")
        if (
            isinstance(delay_steps, (bool, np.bool_))
            or not isinstance(delay_steps, Integral)
            or delay_steps < 0
        ):
            raise ValueError("delay_steps must be a nonnegative physical substep count")
        n = model.dof_count
        if limits is not None and (
            not isinstance(limits, JointLimits) or limits.lower.shape != (n,)
        ):
            raise ValueError("joint limits must match the robot DOFs")
        self.model, self.domain, self.limits = model, domain, limits
        self.delay_steps = int(delay_steps)
        for name, values in (("kp", kp), ("kd", kd), ("effort_limit", effort_limit)):
            array = _vector(values, name, n)
            if np.any(array < 0):
                raise ValueError(f"{name} must be nonnegative")
            array.flags.writeable = False
            setattr(self, name, array)
        self.scene = SceneModel((DomainBinding(domain, "driven_robot", model),))
        self._queue_name = domain + "/drive_queue"
        self._queue_shape = (self.delay_steps, 3, n)
        self.history_fields = (
            (HistoryField(self._queue_name, self._queue_shape, np.float64),)
            if self.delay_steps
            else ()
        )

    def problem_signature(self):
        return dict(
            law="robot-held-stop-delayed-implicit-pd-v1",
            model=self.model,
            domain=self.domain,
            limits=self.limits,
            kp=self.kp,
            kd=self.kd,
            effort_limit=self.effort_limit,
            delay_steps=self.delay_steps,
        )

    def initial_state(self, state, *, prehistory=None):
        """Return an owned state. Optional prehistory has shape (delay, 3, DOFs), oldest first."""
        validate_robot_step(self.model, state, RobotInputs.zeros(self.model.dof_count))
        history = {}
        if prehistory is not None:
            queue = robot_array(prehistory, "prehistory", ndim=3)
            if queue.shape != self._queue_shape:
                raise ValueError(f"prehistory must have shape {self._queue_shape}")
        elif self.delay_steps:
            queue = np.zeros(self._queue_shape)
            queue[:, 0, :] = state.q
        if self.delay_steps:
            history[self._queue_name] = queue
        return SceneState({self.domain: RobotState(state.q, state.v)}, history)

    def inputs(self, target, target_velocity=None, *, feedforward=None, loads=None):
        """Hold a command over the outer step; loads are independent, undelayed RobotInputs."""
        n = self.model.dof_count
        if loads is None:
            loads = RobotInputs.zeros(n)
        if not isinstance(loads, RobotInputs) or loads.effort.shape != (n,):
            raise ValueError("loads must be RobotInputs matching the robot")
        command = _DriveInputs(
            loads.effort,
            loads.force,
            loads.torque,
            target,
            np.zeros(n) if target_velocity is None else target_velocity,
            np.zeros(n) if feedforward is None else feedforward,
        )
        return SceneInputs({self.domain: command})

    def stepper(self, settings=None, *, verifier=None):
        registry = StateRegistry(defaults=False)
        registry.register(
            StateFamily(
                "driven_robot",
                RobotModel,
                RobotState,
                _DriveInputs,
                ("q", "v"),
                ("effort", "force", "torque", "target", "target_velocity", "feedforward"),
                "q",
                validate_robot_step,
            )
        )
        return SceneStepper(
            self.scene,
            self,
            StepSettings(substeps=1) if settings is None else settings,
            registry=registry,
            verifier=verifier,
        )

    def create_workspace(self, scene, state, inputs, h, settings, context):
        self.scene.assert_unchanged()
        if scene.identity != self.scene.identity:
            raise ValueError("driven robot program requires its bound scene")
        return _DrivenWorkspace(self, state, inputs.domains[self.domain], h)

    def close(self):
        pass


class _DrivenWorkspace(RobotWorkspace):
    def __init__(self, program, state, inputs, h):
        super().__init__(program.model, program.domain, state.domains[program.domain], inputs, h)
        self.program = program
        incoming = np.stack((inputs.target, inputs.target_velocity, inputs.feedforward))
        self.history = {}
        if program.delay_steps:
            queue = state.history[program._queue_name]
            self.packet = queue[0]
            self.history[program._queue_name] = np.concatenate((queue[1:], incoming[None, :, :]))
        else:
            self.packet = incoming

    @_numerical
    def predict(self):
        p = self.program
        state = self.initial
        if p.limits is None:
            self.stop = dict(
                effort=np.zeros_like(state.q), violation=np.zeros_like(state.q), potential_j=0.0
            )
        else:
            self.stop = p.limits.evaluate(state)
        external = generalized_forces(self.model, state, self.inputs)
        effective_bias = bias_forces(self.model, state) - external - self.stop["effort"]
        self.drive = solve_bounded_pd(
            mass_matrix(self.model, state.q),
            effective_bias,
            state.q,
            state.v,
            self.packet[0],
            self.packet[1],
            p.kp,
            p.kd,
            p.effort_limit,
            self.h,
            feedforward=self.packet[2],
        )
        self.inputs = RobotInputs(
            self.inputs.effort + self.stop["effort"] + self.drive.effort,
            self.inputs.force,
            self.inputs.torque,
        )
        super().predict()

    def result(self):
        return SceneState(super().result().domains, self.history)

    def diagnostics(self):
        return {
            **super().diagnostics(),
            "applied_target": self.packet[0],
            "applied_target_velocity": self.packet[1],
            "applied_feedforward": self.packet[2],
            "actuator_effort": self.drive.effort,
            "actuator_impulse": self.h * self.drive.effort,
            "stop_effort": self.stop["effort"],
            "stop_impulse": self.h * self.stop["effort"],
            "stop_violation": self.stop["violation"],
            "stop_initial_potential_j": self.stop["potential_j"],
        }
