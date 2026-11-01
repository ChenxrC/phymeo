"""Robot-only consumer of the same compliant loads and common-motion integrator."""

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.scene import DomainBinding, SceneModel, SceneState, StageMetrics
from ..dynamics.robot_contact import RobotContactSystem
from ..dynamics.robot_dynamics import robot_dynamics_terms
from ..runtime.program import ProgramCapabilities
from ..runtime.registry import StateRegistry
from ..runtime.robot_family import robot_family
from ..runtime.scene_stepper import SceneStepper
from ._compliant_motion import (
    advance_state,
    check_robot_contact,
    finite_accelerations,
    optional_budget,
)


class CompliantRobotContactProgram:
    capabilities = ProgramCapabilities(
        "numpy", {"float64"}, {"robot"}, {"compliant_contact", "joint_stops"}
    )
    history_fields = ()

    def __init__(self, contact, *, max_penetration_m=None, max_joint_violation=None):
        if not isinstance(contact, RobotContactSystem):
            raise TypeError("expected RobotContactSystem")
        self.contact = contact
        self.scene = SceneModel(
            tuple(DomainBinding(n, "robot", m) for n, m in contact.models.items())
        )
        self.max_penetration = optional_budget(max_penetration_m, "max_penetration_m")
        self.max_violation = optional_budget(max_joint_violation, "max_joint_violation")

    def problem_signature(self):
        return dict(
            law="robot-contact-midpoint-v1",
            contact=self.contact.signature(),
            max_penetration_m=self.max_penetration,
            max_joint_violation=self.max_violation,
        )

    def stepper(self, settings=None):
        registry = StateRegistry(defaults=False)
        registry.register(robot_family())
        return SceneStepper(self.scene, self, settings, registry=registry)

    def create_workspace(self, scene, state, inputs, h, settings, context):
        if scene.identity != self.scene.identity:
            raise ValueError("different robot contact scene")
        return _Workspace(self, state, inputs, h)

    def accelerations(self, state, inputs):
        terms, geo = {}, {}
        for n, m in self.contact.models.items():
            g, jl, ja, mass, bias = robot_dynamics_terms(m, state.domains[n])
            terms[n] = mass, bias, jl, ja
            geo[n] = g.position, g.rotation, g.com_position, jl, ja
        report = self.contact._evaluate(state.domains, geo)
        accel = {}
        for n, (mass, bias, jl, ja) in terms.items():
            f = inputs.domains[n]
            rhs = (
                f.effort
                + np.einsum("nki,nk->i", jl, f.force)
                + np.einsum("nki,nk->i", ja, f.torque)
                + report["efforts"][n]
                - bias
            )
            try:
                accel[n] = np.linalg.solve(mass, rhs)
            except np.linalg.LinAlgError as exc:
                raise SolverFailure("singular robot inertia") from exc
        finite_accelerations(accel)
        return accel, report

    def close(self):
        pass


class _Workspace:
    def __init__(self, program, state, inputs, h):
        self.program, self.state, self.inputs, self.h = program, state, inputs, h

    def predict(self):
        p = self.program
        a0, r0 = p.accelerations(self.state, self.inputs)
        mid = advance_state(self.state, a0, self.h / 2, midpoint=True)
        a, rm = p.accelerations(mid, self.inputs)
        self.state = advance_state(self.state, a, self.h, midpoint=False)
        re = p.contact.evaluate(self.state.domains)
        check_robot_contact((r0, rm, re), p.max_penetration, p.max_violation)
        self.penetration = max(r["penetration_m"] for r in (r0, rm, re))
        self.report = dict(
            robot_contact=rm,
            sampled_penetration_m=self.penetration,
            sampled_joint_violation=max(r["joint_violation"] for r in (r0, rm, re)),
            force_sample="substep-midpoint",
        )

    def position_sweep(self):
        pass

    def position_metrics(self):
        return StageMetrics(penetration=self.penetration)

    def prepare_velocity(self):
        pass

    def velocity_sweep(self):
        pass

    def velocity_metrics(self):
        return StageMetrics()

    def result(self):
        return SceneState(self.state.domains)

    def diagnostics(self):
        return self.report

    def close(self):
        pass
