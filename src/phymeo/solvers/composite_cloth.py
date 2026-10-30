"""One all-dynamic particle owner for ordered distance and bending material."""

import numpy as np

from ..constraints.bending import bending_residual, project_bending
from ..constraints.cloth_material import ClothMaterial, CompositeMaterialAuditSettings, CompositeMaterialBudgets
from ..constraints.distance import distance_residual, project_distances
from ..constraints.material_trace import MaterialBatch, CompositeMaterialSweep, CompositeMaterialPath
from ..core.capabilities import SolverFailure
from ..core.model import ParticleModel
from ..core.scene import DomainBinding, SceneModel, SceneState, StageMetrics, fingerprint
from ..core.validation import finite_real
from ..dynamics.integrate import predict
from ..runtime.program import ProgramCapabilities
from ..validation.composite_material_path import CompositeMaterialPathLedger
from ..validation.composite_material_energy import CompositeMaterialEnergyLedger


def bending_wire(rows):
    result = np.empty((len(rows), 45))
    for k, row in enumerate(rows):
        result[k, :12] = row.positions_before.ravel()
        result[k, 12:24] = row.positions_after.ravel()
        result[k, 24:36] = row.gradient.ravel()
        result[k, 36:] = (row.theta_before, row.theta_after, row.constraint_before,
                          row.constraint_after, row.lambda_before, row.lambda_after,
                          row.delta_lambda, row.alpha, row.denominator)
    return result


class CompositeClothProgram:
    capabilities = ProgramCapabilities("numpy", {"float64"}, {"particles"}, {"cloth", "distance", "bending"})
    history_fields = ()

    def __init__(self, model, material, domain="cloth", *, audit_settings=None, budgets):
        if not isinstance(model, ParticleModel) or not isinstance(material, ClothMaterial):
            raise TypeError("composite requires ParticleModel and ClothMaterial")
        if not isinstance(budgets, CompositeMaterialBudgets):
            raise TypeError("explicit composite physical budgets required")
        material.validate_model(model)
        self.model, self.material, self.domain = model, ClothMaterial(material.distance, material.bending), domain
        self.audit_settings = CompositeMaterialAuditSettings() if audit_settings is None else audit_settings
        if not isinstance(self.audit_settings, CompositeMaterialAuditSettings):
            raise TypeError("expected CompositeMaterialAuditSettings")
        self.budgets = budgets
        self.scene = SceneModel((DomainBinding(domain, "particles", model),))

    def problem_signature(self):
        return dict(law="composite-cloth-distance-bending-v1", model=self.model, material=self.material,
                    domain=self.domain, audit_settings=self.audit_settings, budgets=self.budgets)

    def create_workspace(self, scene, state, inputs, h, settings, context):
        self.scene.assert_unchanged()
        if scene.identity != self.scene.identity or context.backend != self.capabilities.backend or context.dtype != np.float64:
            raise ValueError("composite bound scene/backend mismatch")
        return self.workspace(self.domain, state.domains[self.domain], inputs.domains[self.domain], h, settings)

    def workspace(self, domain, state, inputs, h, settings):
        return CompositeClothWorkspace(self, domain, state, inputs, h, settings)

    def close(self):
        pass


class CompositeClothWorkspace:
    def __init__(self, program, domain, state, inputs, h, settings):
        self.program, self.model, self.material, self.domain = program, program.model, program.material, domain
        self.initial, self.inputs = state.copy(), type(inputs)(inputs.force.copy())
        self.h, self.settings = finite_real(h, "substep"), settings
        if self.h <= 0:
            raise ValueError("substep must be positive")
        self.candidate, self.predicted, self.velocity_ready = None, None, False
        self.frames, self.alphas, self.multipliers = [], {}, {}
        self.terminal_pre = None
        self.terminal_requests = np.zeros_like(state.position)
        self.prediction_launches = self.reconstruction_launches = 0
        self.closed = False
        self.ledger = CompositeMaterialPathLedger(self.model, self.material, self.initial, self.inputs,
            self.h, settings, audit_settings=program.audit_settings, backend=program.capabilities.backend)
        self.energy_ledger = CompositeMaterialEnergyLedger(self.model, self.material, self.initial,
            self.inputs, self.h, policy=program.audit_settings, budgets=program.budgets, tolerance=settings.tolerance)

    def _ready(self):
        if self.closed or self.candidate is None:
            raise RuntimeError("composite workspace is closed or not predicted")

    def _predict(self):
        trial = self.initial.copy()
        if not predict(self.model, trial, self.inputs.force, self.h):
            raise SolverFailure("composite prediction failed")
        return trial

    def _alpha(self, compliance):
        return (compliance / self.h) / self.h

    def predict(self):
        if self.closed or self.candidate is not None:
            raise RuntimeError("composite prediction must occur exactly once")
        trial = self._predict()
        alphas = {}
        for name in self.material.order:
            law = getattr(self.material, name)
            alphas[name] = np.zeros(0) if law is None else np.asarray(self._alpha(law.compliance), dtype=np.float64)
            if not np.isfinite(alphas[name]).all():
                raise SolverFailure("composite alpha is not representable")
        self.ledger.domain(self.initial.position, trial.position)
        self.candidate, self.predicted = trial, trial.copy()
        self.alphas = alphas
        self.multipliers = {name: np.zeros(self.material.count(name)) for name in self.material.order}

    def _block(self, name, x, multipliers):
        law, count = getattr(self.material, name), self.material.count(name)
        before, old_lambda = x.copy(), multipliers.copy()
        if name == "distance":
            rows = np.empty((count, 18))
            if law is not None:
                project_distances(x, self.model.inverse_mass, law, self.alphas[name], multipliers, trace=rows)
        else:
            rows = np.empty((0, 45)) if law is None else bending_wire(project_bending(
                x, self.model.inverse_mass, law, self.alphas[name], multipliers))
        ids = np.empty((0, 2 if name == "distance" else 4), np.int64) if law is None else law.indices
        rest = np.zeros(0) if law is None else (law.rest_length if name == "distance" else law.rest_angle)
        return MaterialBatch(name, ids, before, before, x, x, old_lambda, old_lambda, multipliers,
            multipliers, self.model.inverse_mass, rest, self.alphas[name], rows, np.zeros(count, np.int64))

    def position_sweep(self):
        self._ready()
        if self.velocity_ready:
            raise RuntimeError("material projection after reconstruction")
        x = self.candidate.position.copy()
        values = {name: value.copy() for name, value in self.multipliers.items()}
        blocks = tuple(self._block(name, x, values[name]) for name in self.material.order)
        sweep = CompositeMaterialSweep(len(self.frames), blocks)
        self.ledger.audit_sweep(sweep, self.candidate.position, self.multipliers, self.alphas)
        # Publish a complete sweep only after its independent domain/path gates.
        self.candidate.position = x
        self.multipliers = values
        self.frames.append(sweep)

    def position_metrics(self):
        self._ready()
        residuals, geometric, angular = {}, 0.0, 0.0
        if self.material.distance is not None:
            residuals["distance"], geometric = distance_residual(self.candidate.position,
                self.material.distance, self.alphas["distance"], self.multipliers["distance"], self.settings.reference_length)
        if self.material.bending is not None:
            residuals["bending"], angular = bending_residual(self.candidate.position,
                self.material.bending, self.alphas["bending"], self.multipliers["bending"], self.program.audit_settings.reference_angle_rad)
        self.angular_error_rad = angular
        return StageMetrics(residuals, geometric_error=geometric)

    def _reconstruct(self):
        return self.predicted.velocity + (self.candidate.position - self.predicted.position) / self.h

    def prepare_velocity(self):
        self._ready()
        if self.velocity_ready:
            raise RuntimeError("composite velocity already reconstructed")
        velocity = self._reconstruct()
        if not np.isfinite(velocity).all():
            raise SolverFailure("composite velocity reconstruction failed")
        self.candidate.velocity = velocity
        self.velocity_ready = True

    def velocity_sweep(self):
        self._ready()

    def velocity_metrics(self):
        self._ready()
        return StageMetrics()

    def _path(self):
        return CompositeMaterialPath(self.material.identity, fingerprint(self.program.audit_settings),
            fingerprint(self.inputs), fingerprint(self.initial), self.program.capabilities.backend,
            self.h, self.predicted.position, self.predicted.velocity, self.alphas["distance"],
            self.alphas["bending"], tuple(self.frames), self.prediction_launches, self.reconstruction_launches)

    def _audit(self):
        self._ready()
        if not self.velocity_ready:
            raise RuntimeError("cannot publish before velocity reconstruction")
        base = self.candidate if self.terminal_pre is None else self.terminal_pre
        data = dict(material_path=self._path(), distance_multipliers=self.multipliers["distance"],
                    bending_multipliers=self.multipliers["bending"], angular_error_rad=getattr(self, "angular_error_rad", 0.0))
        data["material_audit"] = self.ledger.require(base, self.inputs, np.zeros_like(base.position),
            data, sweeps=len(self.frames))
        data["physical_audit"] = self.energy_ledger.require(base)
        if self.terminal_pre is not None:
            if not np.array_equal(self.candidate.position, base.position):
                raise SolverFailure("terminal configuration mutation")
            error = self.candidate.velocity - base.velocity - self.model.inverse_mass[:, None] * self.terminal_requests
            if np.max(np.abs(error), initial=0) / self.settings.reference_speed > self.settings.tolerance:
                raise SolverFailure("terminal velocity mutation or unrecorded impulse")
            data["terminal_pre"] = base.copy()
            data["terminal_requests"] = self.terminal_requests.copy()
        return data

    def result(self):
        self._audit()
        return SceneState({self.domain: self.candidate.copy()})

    def diagnostics(self):
        return self._audit()

    def close(self):
        self.closed = True
