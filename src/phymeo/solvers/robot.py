"""Unconstrained robot stages; SceneStepper alone owns substeps and commit."""

import numpy as np

from ..core.robot import RobotInputs, RobotModel, RobotState, validate_robot_step
from ..core.scene import (
    BackendContext,
    DomainBinding,
    SceneModel,
    SceneState,
    StageMetrics,
    StepSettings,
)
from ..dynamics.robot_dynamics import (
    _finite,
    _numerical,
    bias_forces,
    forward_dynamics,
    generalized_forces,
    mass_matrix,
)
from ..dynamics.robot_kinematics import forward_kinematics
from ..runtime.program import ProgramCapabilities
from ..runtime.registry import StateRegistry
from ..runtime.robot_family import robot_family
from ..runtime.scene_stepper import SceneStepper


@_numerical
def robot_energy(model, state):
    """Kinetic and gravitational potential energy in joules, in that order."""
    validate_robot_step(model, state, RobotInputs.zeros(model.dof_count))
    kinetic = state.v @ mass_matrix(model, state.q) @ state.v / 2
    com = forward_kinematics(model, state.q).com_position
    potential = -np.sum(model.mass[:, None] * com * model.gravity)
    return np.array([kinetic, potential])


class RobotProgram:
    capabilities = ProgramCapabilities(
        "numpy", {"float64"}, {"robot"}, {"fixed_base_tree", "free_dynamics"}
    )
    history_fields = ()

    def __init__(self, model, domain="robot"):
        if not isinstance(model, RobotModel):
            raise TypeError("RobotProgram requires RobotModel")
        self.model, self.domain = model, domain
        self.scene = SceneModel((DomainBinding(domain, "robot", model),))

    def problem_signature(self):
        return {
            "law": "fixed-base-world-inertial-loads-semi-implicit-v1",
            "model": self.model,
            "domain": self.domain,
        }

    def create_workspace(self, scene, state, inputs, h, settings, context):
        self.scene.assert_unchanged()
        if scene.identity != self.scene.identity:
            raise ValueError("robot program must use its bound model and domain")
        return RobotWorkspace(
            self.model, self.domain, state.domains[self.domain], inputs.domains[self.domain], h
        )

    def close(self):
        pass


class RobotWorkspace:
    """Per-substep numerical state only; no persistent physical caches or history."""

    def __init__(self, model, domain, state, inputs, h):
        self.model, self.domain, self.initial, self.inputs, self.h = model, domain, state, inputs, h
        self.candidate = None

    @_numerical
    def predict(self):
        if self.candidate is not None:
            raise RuntimeError("robot workspace prediction already performed")
        self.applied = generalized_forces(self.model, self.initial, self.inputs)
        self.bias = bias_forces(self.model, self.initial)
        self.acceleration = forward_dynamics(self.model, self.initial, self.inputs)
        mass = mass_matrix(self.model, self.initial.q)
        rhs = _finite(self.applied - self.bias)
        residual = _finite(mass @ self.acceleration - rhs)
        scale = max(
            1.0,
            np.linalg.norm(rhs, ord=np.inf),
            np.linalg.norm(mass, ord=np.inf) * np.linalg.norm(self.acceleration, ord=np.inf),
        )
        self.equation_residual = float(np.linalg.norm(residual, ord=np.inf) / _finite(scale))
        velocity = _finite(self.initial.v + self.h * self.acceleration)
        configuration = _finite(self.initial.q + self.h * velocity)
        self.candidate = RobotState(configuration, velocity)

    def position_sweep(self):
        pass

    def position_metrics(self):
        return StageMetrics()

    def prepare_velocity(self):
        pass

    def velocity_sweep(self):
        pass

    def velocity_metrics(self):
        return StageMetrics({"equation": self.equation_residual})

    def result(self):
        if self.candidate is None:
            raise RuntimeError("robot prediction has not been performed")
        return SceneState({self.domain: self.candidate})

    def diagnostics(self):
        kinetic, potential = robot_energy(self.model, self.candidate)
        return {
            "acceleration": self.acceleration,
            "applied": self.applied,
            "bias": self.bias,
            "equation_residual": self.equation_residual,
            "kinetic_energy": float(kinetic),
            "potential_energy": float(potential),
        }

    def close(self):
        pass


def make_robot_runtime(model, *, domain="robot", settings=None):
    program = RobotProgram(model, domain)
    registry = StateRegistry(defaults=False)
    registry.register(robot_family())
    return SceneStepper(
        program.scene,
        program,
        StepSettings(substeps=1) if settings is None else settings,
        BackendContext(),
        registry=registry,
    )
