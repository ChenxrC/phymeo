"""Actual FP32 composite execution with FP64 owned storage and independent audits."""

import numpy as np

from ..constraints.material_trace import MaterialBatch
from ..core.scene import BackendContext
from ..core.state import ParticleState
from ..coupling.composite_cloth import CompositeClothParticipant, CompositeParticipantWorkspace
from ..runtime.program import ProgramCapabilities
from ..runtime.registry import StateRegistry
from ..solvers.composite_cloth import CompositeClothProgram, CompositeClothWorkspace
from .tyx_bending import TyxBendingMath
from .tyx_math import as_compute
from .tyx_participants import _NodePort


class TyxCompositeClothProgram(CompositeClothProgram):
    def __init__(self, model, material, mode, domain="cloth", *, audit_settings=None, budgets, cache_root=None):
        if mode not in TyxBendingMath.MODES:
            raise ValueError("explicit supported native mode required")
        super().__init__(model, material, domain, audit_settings=audit_settings, budgets=budgets)
        self.mode, self.cache_root = mode, cache_root
        self.capabilities = ProgramCapabilities("tyx:" + mode, {"float64"}, {"particles"}, self.capabilities.features)

    def problem_signature(self):
        return dict(super().problem_signature(), backend="tyx:" + self.mode,
                    compute_dtype="float32", storage_dtype="float64")

    def workspace(self, domain, state, inputs, h, settings):
        return TyxCompositeClothWorkspace(self, domain, state, inputs, h, settings)


class TyxCompositeClothParticipant(CompositeClothParticipant):
    def __init__(self, model, material, mode, *, audit_settings=None, budgets, cache_root=None):
        self.program = TyxCompositeClothProgram(model, material, mode,
            audit_settings=audit_settings, budgets=budgets, cache_root=cache_root)
        self.family = StateRegistry().get("particles")

    def create_workspace(self, domain, state, inputs, h, settings, context):
        if context.backend != self.capabilities.backend or context.dtype != np.float64:
            raise ValueError("native composite context mismatch")
        state, inputs = self.family.snapshot(self.model, state, inputs, context)
        return TyxCompositeParticipantWorkspace(self.program.workspace(domain, state, inputs, h, settings), None)


class TyxCompositeParticipantWorkspace(CompositeParticipantWorkspace):
    def point_port(self, indices, local_points=None):
        self.native._ready()
        return _NodePort(self.native, indices, local_points)


class TyxCompositeClothWorkspace(CompositeClothWorkspace):
    def __init__(self, program, domain, state, inputs, h, settings):
        super().__init__(program, domain, state, inputs, h, settings)
        self.family = StateRegistry().get("particles")
        self.context = BackendContext("tyx:" + program.mode)
        self.engine = TyxBendingMath(program.mode, cache_root=program.cache_root)

    def _predict(self):
        before = self.engine.stats()
        converted = as_compute(self.initial.position, "prediction position").astype(np.float64)
        self.ledger.domain(self.initial.position, converted)
        x, v = self.engine.predict(self.initial.position, self.initial.velocity, self.inputs.force,
                                  self.model.inverse_mass, self.model.gravity, self.h)
        self.ledger.domain(converted, x)
        self.prediction_launches = self.engine.stats()["completed_launches"] - before["completed_launches"]
        return ParticleState(x, v)

    def _alpha(self, compliance):
        first = self.engine.axpy(np.zeros_like(compliance), compliance, 1 / self.h)
        return self.engine.axpy(np.zeros_like(first), first, 1 / self.h).astype(np.float64)

    def _block(self, name, x, multipliers):
        law = getattr(self.material, name)
        count = self.material.count(name)
        if not count:
            return super()._block(name, x, multipliers)
        if name == "bending":
            after, result, batch = self.engine.project_bending(x, self.model.inverse_mass, law,
                                                             self.alphas[name], multipliers)
        else:
            before_x, before_lambda = x.copy(), multipliers.copy()
            rows = np.empty((count, 18))
            before = self.engine.stats()
            after, result = self.engine.project_distances(x, self.model.inverse_mass, law,
                                                          self.alphas[name], multipliers, trace=rows)
            stats = self.engine.stats()
            cast = lambda value: as_compute(value, "distance boundary").astype(np.float64)
            batch = MaterialBatch(name, law.indices, before_x, cast(before_x), after, after,
                before_lambda, cast(before_lambda), result, result, cast(self.model.inverse_mass),
                cast(law.rest_length), cast(self.alphas[name]), rows, np.zeros(count, np.int64),
                stats["launches"]-before["launches"], stats["completed_launches"]-before["completed_launches"])
        x[:] = after
        multipliers[:] = result
        return batch

    def _reconstruct(self):
        before = self.engine.stats()
        # Reconstruction itself converts coordinates too; this boundary is certified.
        converted = as_compute(self.candidate.position, "reconstruction position").astype(np.float64)
        self.ledger.domain(self.candidate.position, converted)
        value = self.engine.reconstruct(self.predicted.position, self.predicted.velocity,
                                        self.candidate.position, self.model.inverse_mass, self.h)
        self.reconstruction_launches = self.engine.stats()["completed_launches"] - before["completed_launches"]
        return value

    def diagnostics(self):
        return dict(super().diagnostics(), execution=self.engine.stats())

    def close(self):
        try:
            self.engine.close()
        finally:
            super().close()
