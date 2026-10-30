"""Independent original-input energy, impulse and NUM005 angular accounting."""

import numpy as np

from ..constraints.material_trace import CompositeExternalImpulses, EXTERNAL_PURPOSE
from ..core.capabilities import SolverFailure
from ..core.scene import fingerprint
from ._material_path_math import q, matrix, interval_float
from .bending_bounds import Interval, cross, dot, elastic_bounds


class CompositeMaterialEnergyLedger:
    def __init__(self, model, material, initial, original_inputs, h, *, policy, budgets, tolerance):
        material.validate_model(model)
        self.model, self.material = model, material
        self.initial, self.original = initial.copy(), type(original_inputs)(original_inputs.force.copy())
        self.h, self.policy, self.budgets, self.tolerance = q(h), policy, budgets, q(tolerance)
        if self.h <= 0 or self.tolerance < 0:
            raise ValueError("invalid physical audit time/tolerance")
        self.material_id, self.model_id = material.identity, fingerprint(model)

    def _energy(self, state):
        x, v = matrix(state.position), matrix(state.velocity)
        kinetic = sum((dot(row, row) / (2 * q(w)) for row, w in zip(v, self.model.inverse_mass, strict=True)), q(0))
        gravity = -sum((dot(row, tuple(map(q, self.model.gravity))) / q(w)
                        for row, w in zip(x, self.model.inverse_mass, strict=True)), q(0))
        elastic = elastic_bounds(state.position, self.material, policy=self.policy)
        return kinetic, gravity, elastic, kinetic + gravity + elastic

    def require(self, final, *, selected_pre=None, external=None, held_requests=None,
                terminal_requests=None, effective_inputs=None, internal_boundary=False):
        if self.material.identity != self.material_id or fingerprint(self.model) != self.model_id:
            raise SolverFailure("physical ledger material identity changed")
        n = self.model.particle_count
        held = np.zeros((n, 3)) if held_requests is None else np.asarray(held_requests)
        terminal = np.zeros((n, 3)) if terminal_requests is None else np.asarray(terminal_requests)
        for array in (held, terminal):
            if array.dtype != np.float64 or array.shape != (n, 3) or not np.isfinite(array).all():
                raise SolverFailure("invalid external impulse request")
        requested = bool(np.any(held) or np.any(terminal))
        if requested and external is None:
            raise SolverFailure("explicit impulse requests require an external transfer receipt")
        pre = final if selected_pre is None else selected_pre
        if external is not None:
            if (internal_boundary or not isinstance(external, CompositeExternalImpulses)
                    or external.purpose != EXTERNAL_PURPOSE or q(external.duration) != self.h
                    or external.original_input_fingerprint != fingerprint(self.original)
                    or external.selected_pre_fingerprint != fingerprint(pre)
                    or external.final_state_fingerprint != fingerprint(final)
                    or not np.array_equal(external.held, held)
                    or not np.array_equal(external.terminal, terminal)):
                raise SolverFailure("stale/mismatched or internal external-power receipt")
        if not np.array_equal(pre.position, final.position):
            raise SolverFailure("terminal impulse changed configuration")
        effective = self.original if effective_inputs is None else effective_inputs
        force_error = q(0)
        terminal_error = q(0)
        for i, inverse in enumerate(self.model.inverse_mass):
            for c in range(3):
                force_error = max(force_error, abs((q(effective.force[i, c]) - q(self.original.force[i, c]))
                                                   * self.h - q(held[i, c])) * q(inverse))
                terminal_error = max(terminal_error, abs(q(final.velocity[i, c]) - q(pre.velocity[i, c])
                                                        - q(inverse) * q(terminal[i, c])))
        if max(force_error, terminal_error) > self.tolerance:
            raise SolverFailure("actual represented external load/terminal response mismatch")
        x0, x1, vp = matrix(self.initial.position), matrix(final.position), matrix(pre.velocity)
        v0, v1 = matrix(self.initial.velocity), matrix(final.velocity)
        held_q, terminal_q = matrix(held), matrix(terminal)
        work = sum((q(self.original.force[i, c]) * (x1[i][c] - x0[i][c])
                    for i in range(n) for c in range(3)), q(0))
        held_work = sum((held_q[i][c] * (x1[i][c] - x0[i][c]) / self.h
                         for i in range(n) for c in range(3)), q(0))
        terminal_work = sum((dot(terminal_q[i], vp[i]) + q(self.model.inverse_mass[i])
                             * dot(terminal_q[i], terminal_q[i]) / 2 for i in range(n)), q(0))
        before, after = self._energy(self.initial), self._energy(final)
        raw = after[3] - before[3] - work
        unexplained = raw - held_work - terminal_work
        if unexplained.width > self.tolerance / 64:
            raise SolverFailure("energy proof enclosure budget exhausted")
        angular, linear = [q(0)] * 3, [q(0)] * 3
        for i, inverse in enumerate(self.model.inverse_mass):
            w = q(inverse)
            original_impulse = [self.h * (q(self.original.force[i, c]) + q(self.model.gravity[c]) / w)
                                + held_q[i][c] for c in range(3)]
            a0, a1 = cross(x0[i], [v / w for v in v0[i]]), cross(x1[i], [v / w for v in v1[i]])
            ext, term = cross(x0[i], original_impulse), cross(x1[i], terminal_q[i])
            for c in range(3):
                angular[c] += a1[c] - a0[c] - ext[c] - term[c]
                linear[c] += (v1[i][c] - v0[i][c]) / w - original_impulse[c] - terminal_q[i][c]
        angular_l1 = sum(map(abs, angular), q(0))
        if unexplained.high > q(self.budgets.max_energy_gain_j):
            raise SolverFailure("composite physical energy gain budget exceeded")
        if angular_l1 > q(self.budgets.max_angular_defect_nms):
            raise SolverFailure("composite NUM005 angular defect budget exceeded")
        return dict(kinetic_before=float(before[0]), kinetic_after=float(after[0]),
                    gravity_before=float(before[1]), gravity_after=float(after[1]),
                    raw_energy_defect=raw.floats(), external_transfer_work=float(held_work + terminal_work),
                    held_transfer_work=float(held_work), terminal_transfer_work=float(terminal_work),
                    original_input_work=float(work), unexplained_defect=unexplained.floats(),
                    positive_energy_gain=interval_float((max(q(0), unexplained.high),) * 2)[1],
                    angular_defect=tuple(interval_float((v, v)) for v in angular),
                    angular_l1=interval_float((angular_l1, angular_l1))[1],
                    linear_defect=tuple(interval_float((v, v)) for v in linear),
                    input_response_error=float(force_error), terminal_response_error=float(terminal_error),
                    elastic_before=before[2].floats(), elastic_after=after[2].floats(),
                    frozen_power=material_power_bounds(self.model, self.material, final, policy=self.policy))


def require_material_window(reports, budgets):
    """Positive-part per-step energy and component/domain L1; no cancellation."""
    energy = angular = q(0)
    for report in reports:
        e = max(q(0), q(report["unexplained_defect"][1]))
        a = sum((max(abs(q(low)), abs(q(high))) for low, high in report["angular_defect"]), q(0))
        if e > q(budgets.max_energy_gain_j) or a > q(budgets.max_angular_defect_nms):
            raise SolverFailure("composite per-step physical window limit")
        energy += e
        angular += a
    if energy > q(budgets.max_energy_gain_j) or angular > q(budgets.max_angular_defect_nms):
        raise SolverFailure("composite whole-window physical budget exceeded")
    return dict(positive_energy_sum=interval_float((energy, energy))[1],
                angular_l1_sum=interval_float((angular, angular))[1])


def material_power_bounds(model, material, state, *, policy):
    """Independent frozen-state conservative force, moment and dU/dt bounds."""
    from .bending_bounds import bending_bounds, sqrt_interval
    force = [[Interval.point(0) for _ in range(3)] for _ in state.position]
    if material.distance is not None:
        for ids, rest, compliance in zip(material.distance.indices, material.distance.rest_length,
                                          material.distance.compliance, strict=True):
            if compliance == 0:
                continue
            i, j = map(int, ids)
            delta = [q(state.position[i, c]) - q(state.position[j, c]) for c in range(3)]
            length = sqrt_interval(dot(delta, delta), policy.sqrt_bits)
            if length.low == 0:
                if rest != 0:
                    return dict(status="undefined-distance-cusp")
                continue
            coefficient = -(length - q(rest)) / q(compliance)
            for c in range(3):
                contribution = coefficient * delta[c] / length
                force[i][c] += contribution
                force[j][c] -= contribution
    if material.bending is not None:
        for ids, rest, compliance in zip(material.bending.indices, material.bending.rest_angle,
                                          material.bending.compliance, strict=True):
            bounded = bending_bounds(state.position[ids], rest, compliance, policy=policy)
            for local, node in enumerate(ids):
                for c in range(3):
                    force[node][c] -= bounded.constraint / q(compliance) * bounded.gradient[local][c]
    resultant = [sum((row[c] for row in force), Interval.point(0)) for c in range(3)]
    moments = [cross(tuple(map(q, x)), f) for x, f in zip(state.position, force, strict=True)]
    moment = [sum((row[c] for row in moments), Interval.point(0)) for c in range(3)]
    power = sum((dot(f, tuple(map(q, v))) for f, v in zip(force, state.velocity, strict=True)), Interval.point(0))
    return dict(status="bounded-smooth", force=tuple(tuple(v.floats() for v in row) for row in force),
                resultant=tuple(v.floats() for v in resultant), moment=tuple(v.floats() for v in moment),
                conservative_power=power.floats(), elastic_rate=(-power).floats())


