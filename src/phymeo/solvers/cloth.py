"""Single-domain cloth stages using the existing serial distance mathematics."""

import numpy as np

from ..constraints.distance import DistanceConstraints, distance_residual, project_distances
from ..core.capabilities import SolverFailure
from ..core.model import ParticleModel
from ..core.scene import DomainBinding, SceneModel, SceneState, StageMetrics, StepSettings
from ..dynamics.integrate import predict
from ..runtime.program import ProgramCapabilities
from ..runtime.scene_stepper import SceneStepper


class ClothProgram:
    capabilities = ProgramCapabilities("numpy", {"float64"}, {"particles"}, {"distance", "cloth"})
    history_fields = ()

    def __init__(self, model, constraints, domain="cloth"):
        if not isinstance(model, ParticleModel) or model.inverse_mass.dtype != np.float64:
            raise TypeError("ClothProgram requires a float64 ParticleModel")
        if constraints is not None:
            if not isinstance(constraints, DistanceConstraints):
                raise TypeError("cloth requires DistanceConstraints or None")
            if constraints.rest_length.dtype != np.float64 or np.any(
                constraints.indices >= model.particle_count
            ):
                raise ValueError("cloth constraints must match model count and precision")
        self.model, self.constraints, self.domain = model, constraints, domain
        self.scene = SceneModel((DomainBinding(domain, "particles", model),))

    def problem_signature(self):
        return {
            "law": "cloth-xpbd-distance-predicted-velocity-v1",
            "model": self.model,
            "ordered_constraints": self.constraints,
            "domain": self.domain,
        }

    def create_workspace(self, scene, state, inputs, h, settings, context):
        self.scene.assert_unchanged()
        if scene.identity != self.scene.identity:
            raise ValueError("cloth program must use its bound model and domain")
        return ClothWorkspace(
            self.model,
            self.constraints,
            self.domain,
            state.domains[self.domain],
            inputs.domains[self.domain],
            h,
            settings,
        )

    def close(self):
        pass


class ClothWorkspace:
    def __init__(self, model, constraints, domain, state, inputs, h, settings):
        self.model, self.constraints, self.domain = model, constraints, domain
        self.initial, self.inputs, self.h, self.settings = state, inputs, h, settings
        self.candidate, self.velocity_ready = None, False

    def predict(self):
        if self.candidate is not None:
            raise RuntimeError("cloth prediction already performed")
        self.candidate = self.initial.copy()
        if not predict(self.model, self.candidate, self.inputs.force, self.h):
            raise SolverFailure("cloth free prediction failed")
        self.predicted = self.candidate.copy()
        count = 0 if self.constraints is None else len(self.constraints.indices)
        self.alpha, self.multipliers = np.zeros(count), np.zeros(count)
        if self.constraints is not None:
            np.divide(
                self.constraints.compliance,
                self.h,
                out=self.alpha,
                where=self.constraints.compliance > 0,
            )
            np.divide(self.alpha, self.h, out=self.alpha)
            if not np.isfinite(self.alpha).all():
                raise SolverFailure("cloth compliance is not representable at this step")

    def position_sweep(self):
        if self.constraints is not None:
            project_distances(
                self.candidate.position,
                self.model.inverse_mass,
                self.constraints,
                self.alpha,
                self.multipliers,
            )

    def position_metrics(self):
        if self.constraints is None:
            return StageMetrics()
        residual, geometric = distance_residual(
            self.candidate.position,
            self.constraints,
            self.alpha,
            self.multipliers,
            self.settings.reference_length,
        )
        return StageMetrics({"distance": residual}, geometric_error=geometric)

    def prepare_velocity(self):
        if self.velocity_ready:
            raise RuntimeError("cloth velocity already prepared")
        dynamic = self.model.inverse_mass > 0
        self.candidate.velocity[dynamic] = (
            self.predicted.velocity[dynamic]
            + (self.candidate.position[dynamic] - self.predicted.position[dynamic]) / self.h
        )
        self.candidate.velocity[~dynamic] = 0
        if not np.isfinite(self.candidate.velocity).all():
            raise SolverFailure("cloth velocity reconstruction failed")
        self.velocity_ready = True

    def velocity_sweep(self):
        pass

    def velocity_metrics(self):
        return StageMetrics()

    def result(self):
        if self.candidate is None:
            raise RuntimeError("cloth prediction has not been performed")
        return SceneState({self.domain: self.candidate})

    def diagnostics(self):
        return {"distance_multipliers": self.multipliers}

    def close(self):
        pass


def make_cloth_runtime(model, constraints, *, domain="cloth", settings=None):
    program = ClothProgram(model, constraints, domain)
    return SceneStepper(
        program.scene, program, StepSettings(substeps=1) if settings is None else settings
    )
