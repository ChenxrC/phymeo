"""Composite material participant: one state owner and explicit terminal impulses."""

import numpy as np

from ..core.capabilities import SolverFailure
from ..runtime.registry import StateRegistry
from ..solvers.composite_cloth import CompositeClothProgram
from .participant import NodeParticipantWorkspace
from .ports import ParticlePointPort


class CompositeClothParticipant:
    point_role = "node"

    def __init__(self, model, material, *, audit_settings=None, budgets):
        self.program = CompositeClothProgram(model, material, audit_settings=audit_settings, budgets=budgets)
        self.family = StateRegistry().get("particles")

    @property
    def model(self):
        return self.program.model

    @property
    def material(self):
        return self.program.material

    @property
    def capabilities(self):
        return self.program.capabilities

    def problem_signature(self):
        return self.program.problem_signature()

    def create_workspace(self, domain, state, inputs, h, settings, context):
        if context.backend != self.capabilities.backend or context.dtype != np.float64:
            raise ValueError("composite participant context mismatch")
        state, inputs = self.family.snapshot(self.model, state, inputs, context)
        return CompositeParticipantWorkspace(self.program.workspace(domain, state, inputs, h, settings), ParticlePointPort)


class CompositeParticipantWorkspace(NodeParticipantWorkspace):
    def apply_position_load(self, indices, local_points, loads):
        raise SolverFailure("B2-M forbids unrecorded interleaved configuration loads")

    def apply_velocity_impulses(self, indices, local_points, impulses):
        self.native._ready()
        if not self.native.velocity_ready:
            raise RuntimeError("terminal impulse requires reconstructed velocity")
        port = self.point_port(indices, local_points)
        impulses = np.asarray(impulses)
        if impulses.dtype != np.float64 or impulses.shape != (len(port.indices), 3) or not np.isfinite(impulses).all():
            raise ValueError("terminal impulses require finite FP64 (m,3)")
        candidate = port.validate_candidate(port.apply_impulses(impulses))
        grouped = np.zeros_like(self.native.terminal_requests)
        # Aggregate every request on the physical node before kinetic accounting.
        for node in sorted(set(map(int, port.indices))):
            from math import fsum
            for c in range(3):
                grouped[node, c] = fsum(float(impulses[k, c]) for k, i in enumerate(port.indices) if i == node)
        if self.native.terminal_pre is None:
            self.native.terminal_pre = self.native.candidate.copy()
        self.native.candidate = candidate
        self.native.terminal_requests += grouped
