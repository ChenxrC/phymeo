"""Explicit bounded StVK stages under the generic scene transaction owner."""

import numpy as np

from ..core.scene import DomainBinding, SceneModel, SceneState, StageMetrics, StepSettings
from ..core.state import ParticleState
from ..core.tetrahedral import TetrahedralModel
from ..materials.stvk import _finite, deformation_gradient, elastic_energy, elastic_forces
from ..runtime.program import ProgramCapabilities
from ..runtime.registry import StateRegistry
from ..runtime.scene_stepper import SceneStepper
from ..runtime.tetrahedral_family import tetrahedral_family


class TetrahedralProgram:
    capabilities = ProgramCapabilities(
        "numpy", {"float64"}, {"tetrahedra"}, {"stvk", "lumped_mass", "bounded_deformation"}
    )
    history_fields = ()

    def __init__(self, model, domain="soft"):
        if not isinstance(model, TetrahedralModel):
            raise TypeError("TetrahedralProgram requires TetrahedralModel")
        self.model, self.domain = model, domain
        self.scene = SceneModel((DomainBinding(domain, "tetrahedra", model),))

    def problem_signature(self):
        return {
            "law": "bounded-stvk-explicit-semi-implicit-v1",
            "model": self.model,
            "domain": self.domain,
        }

    def create_workspace(self, scene, state, inputs, h, settings, context):
        self.scene.assert_unchanged()
        if scene.identity != self.scene.identity:
            raise ValueError("solid program must use its bound model and domain")
        return TetrahedralWorkspace(
            self.model,
            self.domain,
            state.domains[self.domain],
            inputs.domains[self.domain],
            h,
            settings,
        )

    def close(self):
        pass


class TetrahedralWorkspace:
    def __init__(self, model, domain, state, inputs, h, settings):
        self.model, self.domain, self.initial, self.inputs = model, domain, state, inputs
        self.h, self.settings = h, settings
        self.candidate, self.velocity_ready = None, False

    def predict(self):
        if self.candidate is not None:
            raise RuntimeError("solid prediction already performed")
        model = self.model
        self.internal_force = elastic_forces(model, self.initial.position)
        total = _finite(self.internal_force + self.inputs.force)
        self.acceleration = np.zeros_like(total)
        dynamic = ~model.fixed
        self.acceleration[dynamic] = _finite(
            model.gravity + model.inverse_mass[dynamic, None] * total[dynamic]
        )
        velocity = self.initial.velocity.copy()
        velocity[dynamic] += self.h * self.acceleration[dynamic]
        velocity[model.fixed] = 0
        position = self.initial.position.copy()
        position[dynamic] += self.h * velocity[dynamic]
        self.candidate = ParticleState(_finite(position), _finite(velocity))
        self.predicted = self.candidate.copy()
        self.support_force = np.zeros_like(total)
        self.support_force[model.fixed] = _finite(
            -total[model.fixed] - model.mass[model.fixed, None] * model.gravity
        )
        velocity_error = (velocity - self.initial.velocity) - self.h * self.acceleration
        position_error = (position - self.initial.position) - self.h * velocity
        self.integration_residual = float(
            max(
                np.max(np.abs(velocity_error), initial=0) / self.settings.reference_speed,
                np.max(np.abs(position_error), initial=0) / self.settings.reference_length,
            )
        )

    def position_sweep(self):
        pass

    def position_metrics(self):
        deformation_gradient(self.model, self.candidate.position)
        return StageMetrics({"prediction_equation": self.integration_residual})

    def prepare_velocity(self):
        if self.velocity_ready:
            raise RuntimeError("solid velocity already prepared")
        dynamic = ~self.model.fixed
        self.candidate.velocity[dynamic] = _finite(
            self.predicted.velocity[dynamic]
            + (self.candidate.position[dynamic] - self.predicted.position[dynamic]) / self.h
        )
        self.candidate.velocity[self.model.fixed] = 0
        self.velocity_ready = True

    def velocity_sweep(self):
        pass

    def velocity_metrics(self):
        return StageMetrics()

    def result(self):
        if self.candidate is None:
            raise RuntimeError("solid prediction has not been performed")
        return SceneState({self.domain: self.candidate})

    def diagnostics(self):
        gradient = deformation_gradient(self.model, self.candidate.position)
        singular = np.linalg.svd(gradient, compute_uv=False)
        kinetic = _finite(np.sum(self.model.mass[:, None] * self.candidate.velocity**2) / 2)
        potential = _finite(
            -np.sum(self.model.mass[:, None] * self.candidate.position * self.model.gravity)
        )
        return {
            "internal_force": self.internal_force,
            "support_force": self.support_force,
            "acceleration": self.acceleration,
            "integration_residual": self.integration_residual,
            "elastic_energy": elastic_energy(self.model, self.candidate.position),
            "kinetic_energy": float(kinetic),
            "potential_energy": float(potential),
            "min_det_f": float(np.min(np.linalg.det(gradient))),
            "min_singular_value": float(singular.min()),
            "max_singular_value": float(singular.max()),
        }

    def close(self):
        pass


def make_tetrahedral_runtime(model, *, domain="soft", settings=None):
    program = TetrahedralProgram(model, domain)
    registry = StateRegistry(defaults=False)
    registry.register(tetrahedral_family())
    return SceneStepper(
        program.scene,
        program,
        StepSettings(substeps=1) if settings is None else settings,
        registry=registry,
    )
