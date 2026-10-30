"""Exactly one actual held/terminal replay and selected-pre evidence binding."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.scene import SceneState, fingerprint
from ..runtime.load_replay import SceneLoadReplay
from ..runtime.program import SceneSubstepReport, own_diagnostics
from ..runtime.terminal_impulse import PointVelocityImpulses, TerminalImpulseProgram
from ..validation.material_path import MaterialPathLedger
from ..validation.cloth_self_mapping import SelfContactMappingLedger


def observational(value):
    """R5: only the precise typed substep report's timings are observational."""
    if type(value) is SceneSubstepReport:
        return observational({entry.name: getattr(value, entry.name) for entry in fields(value) if entry.name != "timings"})
    if isinstance(value, Mapping):
        return {key: observational(item) for key, item in value.items()}
    if is_dataclass(value):
        return {"type": type(value).__module__+"."+type(value).__qualname__,
                "fields": {entry.name: observational(getattr(value, entry.name)) for entry in fields(value)}}
    if isinstance(value, (tuple, list)):
        return tuple(observational(item) for item in value)
    return value


def require_selected_pre(actual, selected):
    for key in ("pre_impulse_state", "pre_impulse_diagnostics", "pre_position_iterations",
                "pre_velocity_iterations", "pre_position_metrics", "pre_velocity_metrics"):
        if fingerprint(observational(actual["terminal"][key])) != fingerprint(observational(selected["terminal"][key])):
            raise SolverFailure("terminal replay changed complete selected pre evidence: "+key)
    if fingerprint(actual["trial"].effective_inputs) != fingerprint(selected["trial"].effective_inputs):
        raise SolverFailure("terminal replay changed selected represented inputs")


class SelfContactReplay:
    def __init__(self, program, initial, inputs, h, settings, context):
        self.program, self.h, self.settings, self.context = program, h, settings, context
        self.initial, self.inputs = program.registry.snapshot(program.scene, initial, inputs, context, program.history_fields)
        self.original_token = fingerprint((self.initial, self.inputs))
        self.base_state = SceneState(self.initial.domains, {f.name: self.initial.history[f.name] for f in program.base.history_fields})
        self.count = 0
        self.material = MaterialPathLedger(program.model, program.materials, self.initial.domains[program.domain],
                                           self.inputs.domains[program.domain], h, program.inner_settings, context)
        self.mapping = SelfContactMappingLedger(program.model, proof_limits=program.limits.geometry_limits)

    def evaluate(self, held_plan, terminal_plan=None):
        program, domain = self.program, self.program.domain
        if self.count >= program.limits.max_outer_replays:
            raise SolverFailure("outer replay budget exhausted")
        self.count += 1
        requests = ()
        if terminal_plan is not None:
            ids = np.flatnonzero(np.any(terminal_plan.node_impulses != 0, axis=1)).astype(np.int64)
            if len(ids):
                requests = (PointVelocityImpulses(domain, ids, terminal_plan.node_impulses[ids]),)
        adapter = TerminalImpulseProgram(program.base, requests, registry=program.registry)
        trial = None
        try:
            with SceneLoadReplay(program.scene, adapter, self.base_state, self.inputs, self.h,
                                 channels={domain: ("force",)}, settings=program.inner_settings,
                                 context=self.context, registry=program.registry) as replay:
                trial = replay.evaluate({domain: {"force": held_plan.node_impulses}})
            if not trial.result.report.accepted:
                raise SolverFailure("actual self-contact replay rejected: "+str(trial.result.report.reason))
            substep = trial.result.report.substeps[0]
            terminal = substep.diagnostics["terminal_impulse"]
            pre = terminal["pre_impulse_state"]
            path = terminal["pre_impulse_diagnostics"]["participants"][domain]
            material = self.material.require(pre.domains[domain], trial.effective_inputs.domains[domain],
                                             held_plan.node_impulses.astype(object), path,
                                             sweeps=terminal["pre_position_iterations"])
            held_receipt = dict(phase="held", plan_token=held_plan.binding_token,
                                represented_nodes=trial.integrated_loads[domain]["force"], h=self.h,
                                original_force=self.inputs.domains[domain].force,
                                effective_force=trial.effective_inputs.domains[domain].force,
                                effective_inputs=trial.effective_inputs.domains[domain],
                                predicted_velocity=path["material_path"]["predicted_velocity"],
                                gravity=program.model.gravity, backend=self.context.backend,
                                native_execution=path.get("execution", {}), material_scope=material["scope"],
                                pre_diagnostics=path, sweeps=terminal["pre_position_iterations"])
            held_map = self.mapping.require(held_plan, original_positions=self.initial.domains[domain].position,
                                            original_velocity=self.initial.domains[domain].velocity,
                                            actual_replay_or_terminal_receipt=held_receipt, budgets=program.budgets)
            terminal_receipt = terminal_map = None
            if terminal_plan is not None:
                represented = np.zeros_like(terminal_plan.node_impulses)
                for batch in terminal["batches"]:
                    if batch.domain != domain or batch.local_points is not None:
                        raise SolverFailure("unexpected terminal physical domain/local point")
                    np.add.at(represented, batch.indices, batch.impulses)
                terminal_receipt = dict(phase="terminal", plan_token=terminal_plan.binding_token,
                                        represented_nodes=represented, final_velocity=trial.result.state.domains[domain].velocity,
                                        backend=self.context.backend, applications=terminal["applications"],
                                        native_execution=substep.diagnostics["participants"][domain].get("execution", {}),
                                        native_delta=terminal["post_execution_delta"].get(domain, {}), batches=terminal["batches"])
                terminal_map = self.mapping.require(terminal_plan, original_positions=pre.domains[domain].position,
                                                     original_velocity=pre.domains[domain].velocity,
                                                     actual_replay_or_terminal_receipt=terminal_receipt, budgets=program.budgets)
            record = dict(trial=trial, terminal=terminal, material=material,
                          held_plan=held_plan, terminal_plan=terminal_plan,
                          material_path=path["material_path"], held_receipt=held_receipt,
                          terminal_receipt=terminal_receipt, held_mapping=held_map, terminal_mapping=terminal_map)
            self.record(dict(kind="replay", index=self.count, accepted=True, **record))
            return record
        except BaseException as error:
            self.record(dict(kind="replay", index=self.count, accepted=False, reason=str(error), trial=trial, held_plan=held_plan, terminal_plan=terminal_plan, requests=requests))
            raise
        finally:
            if fingerprint((self.initial, self.inputs)) != self.original_token:
                raise SolverFailure("self replay mutated original state/history/input snapshot")

    def record(self, receipt):
        self.program._attempt_receipts = (*self.program.last_attempt_receipts, own_diagnostics(receipt))
