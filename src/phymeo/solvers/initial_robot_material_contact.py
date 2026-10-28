"""Initial common-point material contact with native replay and physical budgets."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from ..constraints.distance import DistanceConstraints
from ..core.capabilities import SolverFailure
from ..core.scene import StageMetrics
from ..core.validation import finite_real
from ..runtime.program import own_diagnostics
from ..validation.material_path import MaterialPathLedger
from ._material_contact_balance import MaterialContactBalance
from .initial_robot_node_contact import InitialRobotNodeContactProgram, _InitialPairWorkspace


@dataclass(frozen=True)
class MaterialContactBudgets:
    """Explicit absolute ceilings per substep, in J and N m s respectively."""

    max_energy_gain_j: float
    max_material_angular_defect_nms: float

    def __post_init__(self):
        for name in ("max_energy_gain_j", "max_material_angular_defect_nms"):
            value = finite_real(getattr(self, name), name)
            if value <= 0:
                raise ValueError("physical budgets must be strictly positive")
            object.__setattr__(self, name, value)


class InitialRobotMaterialContactProgram(InitialRobotNodeContactProgram):
    """Closed initial contacts only; neither automatic collision generation nor CCD."""

    def __init__(
        self,
        scene,
        base_program,
        contacts,
        *,
        materials,
        budgets,
        robot_domain="robot",
        registry=None,
        inner_settings=None,
        relaxation=1.0,
        reference_angular_speed=1.0,
    ):
        super().__init__(
            scene,
            base_program,
            contacts,
            robot_domain=robot_domain,
            registry=registry,
            inner_settings=inner_settings,
            relaxation=relaxation,
            reference_angular_speed=reference_angular_speed,
        )
        if not isinstance(budgets, MaterialContactBudgets):
            raise TypeError("explicit MaterialContactBudgets required")
        if not isinstance(materials, Mapping):
            raise TypeError("materials must be an explicit mapping")
        selected = {c.node_domain for c in self.contacts}
        if set(materials) != selected:
            raise ValueError("materials must match exactly the selected node domains")
        models = {d.name: d.model for d in scene.domains}
        owned = {}
        for name, material in materials.items():
            if models[name].inverse_mass.dtype != np.float64:
                raise TypeError("material contacts require FP64 particle storage")
            if material is not None:
                if (
                    not isinstance(material, DistanceConstraints)
                    or material.rest_length.dtype != np.float64
                ):
                    raise TypeError("expected FP64 DistanceConstraints or None")
                if np.any(material.indices >= models[name].particle_count):
                    raise ValueError("material edge index outside domain")
                material = replace(material)
            owned[name] = material
        self.materials = MappingProxyType(owned)
        self.budgets = replace(budgets)

    @property
    def capabilities(self):
        return replace(
            self.base.capabilities,
            features=self.base.capabilities.features | {"initial_robot_material_replay"},
            dtypes=frozenset({"float64"}),
        )

    def problem_signature(self):
        result = super().problem_signature()
        result.update(
            law="initial-common-point-robot-distance-material-replay-v1",
            materials=self.materials,
            budgets=self.budgets,
        )
        return result

    def _make_workspace(self, state, inputs, h, settings, inner, context):
        return _MaterialPairWorkspace(self, state, inputs, h, settings, inner, context)


class _MaterialPairWorkspace(_InitialPairWorkspace):
    def __init__(self, *args):
        super().__init__(*args)
        self._sweeps = 0
        self._best = None
        self._refinement_done = False

    def position_sweep(self):
        self._sweeps += 1
        super().position_sweep()

    def _restore_best(self):
        (_, self.impulses, self.trial, self.native_audit, self.material_reports) = self._best
        self._refinement_done = True

    def position_metrics(self):
        metrics = self._audit()[0]
        admissible = metrics.max_residual <= self.settings.tolerance
        if self._refinement_done:
            return metrics
        # A tolerance-level positive velocity bias accumulates in repeated
        # closed-contact windows. Refine the *same* law with remaining GS
        # iterations, retaining an entire independently admissible native trial.
        # No gap feedback, position patch, altered input law or tolerance.
        score = max(metrics.residuals["normal"], metrics.residuals["friction"])
        if admissible and (self._best is None or score < self._best[0]):
            self._best = (
                score,
                self.impulses.copy(),
                self.trial,
                self.native_audit,
                self.material_reports,
            )
            if score == 0 or self._sweeps >= self.settings.position_iterations:
                self._refinement_done = True
                return metrics
        elif self._best is not None:
            self._restore_best()
            return self._audit()[0]
        if self._best is not None:
            # This is an iteration-status residual, not a physical error. The
            # physical budget and original admissibility thresholds are intact.
            residual = dict(metrics.residuals)
            residual["iteration_refinement_pending"] = 2 * self.settings.tolerance
            return StageMetrics(residual, penetration=metrics.penetration)
        return metrics

    def _selected_domains(self):
        return tuple(sorted(self.program.materials))

    def _make_node_ledgers(self, state, inputs, h, settings, context):
        self.balance = MaterialContactBalance(
            self.models, self.program.materials, state, inputs, self.domain, h, self.program.budgets
        )
        self.material_reports = {}
        return {
            name: MaterialPathLedger(
                self.models[name],
                self.program.materials[name],
                state.domains[name],
                inputs.domains[name],
                h,
                settings,
                context,
            )
            for name in self.node_domains
        }

    def _audit_nodes(self, trial, node, audit):
        substeps = trial.result.report.substeps
        if len(substeps) != 1:
            raise SolverFailure("material replay requires one native substep")
        sub = substeps[0]
        participants = sub.diagnostics.get("participants", {})
        reports = {}
        for name, ledger in self.node_ledgers.items():
            report = ledger.require(
                trial.result.state.domains[name],
                trial.effective_inputs.domains[name],
                np.asarray(node[name], dtype=object).reshape(-1, 3),
                participants.get(name),
                sweeps=sub.position_iterations,
            )
            reports[name] = report
            # MaterialPathLedger already enforces exact equations at original tol.
            audit["material_" + name] = report["max_residual"]
        self.material_reports = reports

    def _remember_free_energy(self, state):
        # The old K-free bound is deliberately exclusive to the free-node law.
        pass

    def _energy_audit(self, state, magnitude, residual):
        metrics, self.physical_report, change = self.balance.require_metrics(
            state, self.ledger, self._loads()[2], self.settings.tolerance
        )
        residual.update(metrics)
        return change

    def diagnostics(self):
        source = super().diagnostics()
        contact = source.pop("initial_robot_node_contact")
        contact.pop("frozen_total_kinetic_change_from_free")
        contact.update(
            material_reports=self.material_reports, physical_budgets=self.physical_report
        )
        source["initial_robot_material_contact"] = contact
        return own_diagnostics(source)
