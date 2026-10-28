"""Stored-value physical budgets, separate from native equation certification."""

from dataclasses import replace

from ..core.state import ParticleState, StepInputs
from ..runtime.program import own_diagnostics
from ..validation._material_path_math import (
    ZERO,
    cross,
    display,
    display_array,
    elastic,
    interval_float,
    matrix,
    q,
)
from ._robot_replay_ledger import upper_float


class MaterialContactBalance:
    """Frozen robot energy and per-domain material angular defect; never integrate."""

    def __init__(self, models, materials, state, inputs, robot_domain, h, budgets):
        self.materials = materials
        self.models = {name: replace(models[name]) for name in materials}
        self.initial = {
            name: ParticleState(
                state.domains[name].position.copy(), state.domains[name].velocity.copy()
            )
            for name in materials
        }
        self.original = {name: StepInputs(inputs.domains[name].force.copy()) for name in materials}
        self.robot_domain = robot_domain
        self.robot_q0 = tuple(map(q, state.domains[robot_domain].q))
        self.robot_v0 = state.domains[robot_domain].v.copy()
        self.h, self.budgets = q(h), budgets

    def require_metrics(self, final, robot_ledger, requests, tolerance):
        r0 = robot_ledger.kinetic(self.robot_v0)
        r1 = robot_ledger.kinetic(final.domains[self.robot_domain].v)
        e0, e1 = [r0, r0], [r1, r1]
        work = sum(
            (
                f / self.h * (q(x) - before)
                for f, x, before in zip(
                    robot_ledger.free_integral,
                    final.domains[self.robot_domain].q,
                    self.robot_q0,
                    strict=True,
                )
            ),
            ZERO,
        )
        defects = {}
        for name, model in self.models.items():
            x0, x1, v0, v1 = map(
                matrix,
                (
                    self.initial[name].position,
                    final.domains[name].position,
                    self.initial[name].velocity,
                    final.domains[name].velocity,
                ),
            )
            force = matrix(self.original[name].force)
            angular = [ZERO] * 3
            for i, inverse in enumerate(model.inverse_mass):
                mass = 1 / q(inverse)
                expected = [
                    self.h * (mass * q(model.gravity[k]) + force[i][k]) + q(requests[name][i][k])
                    for k in range(3)
                ]
                before = cross(x0[i], [mass * v for v in v0[i]])
                after = cross(x1[i], [mass * v for v in v1[i]])
                external = cross(x0[i], expected)
                for k in range(3):
                    first = mass * (v0[i][k] ** 2 / 2 - q(model.gravity[k]) * x0[i][k])
                    last = mass * (v1[i][k] ** 2 / 2 - q(model.gravity[k]) * x1[i][k])
                    for j in range(2):
                        e0[j] += first
                        e1[j] += last
                    work += force[i][k] * (x1[i][k] - x0[i][k])
                    angular[k] += after[k] - before[k] - external[k]
            u0 = elastic(model, self.materials[name], x0)
            u1 = elastic(model, self.materials[name], x1)
            for j in range(2):
                e0[j] += u0[j]
                e1[j] += u1[j]
            defects[name] = angular
        change = (e1[0] - e0[1] - work, e1[1] - e0[0] - work)
        positive = max(change[1], ZERO)
        angular = sum((abs(v) for row in defects.values() for v in row), ZERO)
        # Normalize exact physical bounds for StageMetrics, without changing the
        # fixed physical ceilings or using rounded diagnostic values as gates.
        metrics = dict(
            frozen_model_energy=upper_float(
                q(tolerance) * positive / q(self.budgets.max_energy_gain_j)
            ),
            material_angular_defect=upper_float(
                q(tolerance) * angular / q(self.budgets.max_material_angular_defect_nms)
            ),
        )
        report = own_diagnostics(
            dict(
                frozen_model_energy=dict(
                    scope="frozen-M0-energy-original-endpoint-work-v1",
                    before_j=interval_float(e0),
                    after_j=interval_float(e1),
                    original_input_work_j=display(work),
                    change_minus_work_j=interval_float(change),
                    positive_gain_upper_j=upper_float(positive),
                    limit_j=self.budgets.max_energy_gain_j,
                ),
                material_angular_defect=dict(
                    scope="particle-initial-external-moment-defect-v1",
                    by_domain_nms={name: display_array(row) for name, row in defects.items()},
                    l1_sum_nms=upper_float(angular),
                    limit_nms=self.budgets.max_material_angular_defect_nms,
                ),
            )
        )
        return metrics, report, change[1]
