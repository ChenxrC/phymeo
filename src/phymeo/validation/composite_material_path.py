"""Independent replay of complete composite observations; no state repair."""

import numpy as np

from ..constraints.material_trace import CompositeMaterialPath
from ..core.capabilities import SolverFailure
from ..core.scene import fingerprint
from ._material_path_math import q, matrix, compute
from .bending_bounds import Interval, bending_bounds, cross, dot, sqrt_interval
from .bending_domain import certify_bending_segment


class CompositeMaterialPathLedger:
    def __init__(self, model, material, initial, original_inputs, h, settings, *, audit_settings,
                 backend="numpy"):
        material.validate_model(model)
        self.model, self.material, self.initial = model, material, initial.copy()
        self.original = type(original_inputs)(original_inputs.force.copy())
        self.h, self.settings, self.policy, self.backend = q(h), settings, audit_settings, backend
        self.material_id, self.policy_id = material.identity, fingerprint(audit_settings)
        self.model_id = fingerprint(model)
        self.tau = q(settings.tolerance)
        self._domains, self._bounds = {}, {}
        if self.h <= 0 or backend not in ("numpy", "tyx:interpret", "tyx:cpu_cpp", "tyx:cuda"):
            raise ValueError("unsupported composite audit context")

    def _equal(self, actual, expected, name):
        if not np.array_equal(actual, expected):
            raise SolverFailure(f"composite path {name} mismatch")

    def _error(self, actual, ideal, scale=1):
        ideal = ideal if isinstance(ideal, Interval) else Interval.point(ideal)
        scale = q(scale)
        if scale <= 0 or ideal.width / scale > self.tau / 64:
            raise SolverFailure("composite independent enclosure width exhausted")
        error = (q(actual) - ideal).magnitude / scale
        if error > self.tau:
            raise SolverFailure("composite independent equation mismatch")
        return error

    def domain(self, before, after):
        key = fingerprint((before, after))
        if key not in self._domains:
            self._domains[key] = certify_bending_segment(before, after, self.material, policy=self.policy).require()
        return self._domains[key]

    def bounds(self, nodes, k):
        key = fingerprint((nodes, k))
        if key not in self._bounds:
            law = self.material.bending
            self._bounds[key] = bending_bounds(nodes, law.rest_angle[k], law.compliance[k], policy=self.policy)
        return self._bounds[key]

    def audit_sweep(self, sweep, position, multipliers, alphas):
        """Validate a private full sweep before its owner publishes any part."""
        current = position.copy()
        values = {name: multipliers[name].copy() for name in self.material.order}
        budget = [[q(0)] * 3 for _ in current]
        equation_budget = {name: [q(0)] * self.material.count(name) for name in self.material.order}
        lambda_budget = {name: [q(0)] * self.material.count(name) for name in self.material.order}
        loads = [[q(0)] * 3 for _ in current]
        native = self.backend != "numpy"
        for batch in sweep.blocks:
            name, law = batch.kind, getattr(self.material, batch.kind)
            count = self.material.count(name)
            ids = np.empty((0, 2 if name == "distance" else 4), np.int64) if law is None else law.indices
            rest = np.zeros(0) if law is None else (law.rest_length if name == "distance" else law.rest_angle)
            convert = compute if native and count else lambda x: np.array(x, copy=True)
            self._equal(batch.node_indices, ids, "ordered identities")
            self._equal(batch.positions_in, current, "block input")
            self._equal(batch.lambda_in, values[name], "lambda input")
            self._equal(batch.compute_start, convert(current), "position conversion")
            if native and count:
                self._equal(batch.rows, compute(batch.rows), "native FP32 observation storage")
            self._equal(batch.lambda_compute_start, convert(values[name]), "lambda conversion")
            self._equal(batch.compute_weights, convert(self.model.inverse_mass), "mass conversion")
            self._equal(batch.compute_rest, convert(rest), "rest conversion")
            self._equal(batch.compute_alpha, convert(alphas[name]), "alpha conversion")
            required_launches = int(native and count > 0)
            if (batch.launches, batch.completed_launches) != (required_launches, required_launches):
                raise SolverFailure("missing or forged operation-specific material launch")
            self.domain(current, batch.compute_start)
            for i in range(len(current)):
                for c in range(3):
                    budget[i][c] += abs(q(batch.compute_start[i, c]) - q(current[i, c]))
            current = batch.compute_start.copy()
            values[name] = batch.lambda_compute_start.copy()
            for k, node_ids in enumerate(ids):
                row = batch.rows[k]
                size = len(node_ids) * 3
                self._equal(row[:size].reshape((-1, 3)), current[node_ids], "row before")
                before = current.copy()
                weights = tuple(q(batch.compute_weights[i]) for i in node_ids)
                original_weights = tuple(q(self.model.inverse_mass[i]) for i in node_ids)
                alpha = q(batch.compute_alpha[k])
                ideal_alpha = q(law.compliance[k]) / self.h**2
                if ideal_alpha == 0:
                    if alpha != 0:
                        raise SolverFailure("hard distance acquired compliance")
                else:
                    self._error(alpha, ideal_alpha, ideal_alpha)
                if name == "distance":
                    lb, la, delta = map(q, (row[12], row[13], row[17]))
                    d = tuple(q(row[c]) - q(row[c + 3]) for c in range(3))
                    length = sqrt_interval(dot(d, d), self.policy.sqrt_bits)
                    gradient = tuple(Interval.point(int(c == 0)) if length.high == 0 else Interval.point(d[c]) / length for c in range(3))
                    gradients = (gradient, tuple(-g for g in gradient))
                    observed = (tuple(map(q, row[14:17])), tuple(-q(v) for v in row[14:17]))
                    constraint = length - q(law.rest_length[k])
                    denominator = sum(weights) + alpha
                    ideal_denominator = Interval.point(sum(original_weights) + ideal_alpha)
                    equation = constraint + ideal_alpha * lb + ideal_denominator * delta
                    scale = max(q(law.rest_length[k]), q(self.settings.reference_length))
                    self._error(0, equation, scale)
                else:
                    lb, la, delta = map(q, (row[40], row[41], row[42]))
                    bounded = self.bounds(current[node_ids], k)
                    gradients = bounded.gradient
                    observed = tuple(tuple(map(q, row[24+3*i:27+3*i])) for i in range(4))
                    scale = max(abs(q(law.rest_angle[k])), q(self.policy.reference_angle_rad))
                    self._error(row[36], bounded.angle, scale)
                    self._error(row[38], bounded.constraint, scale)
                    self._error(row[43], alpha, alpha)
                    denominator = sum((weights[i] * sum((g.square() for g in gradients[i]), Interval.point(0)) for i in range(4)), Interval.point(alpha))
                    self._error(row[44], denominator, denominator.low)
                    ideal_denominator = sum((original_weights[i] * sum((g.square() for g in gradients[i]), Interval.point(0)) for i in range(4)), Interval.point(ideal_alpha))
                    equation = bounded.constraint + ideal_alpha * lb + ideal_denominator * delta
                    self._error(0, equation, scale)
                if lb != q(values[name][k]):
                    raise SolverFailure("lambda row path mismatch")
                lambda_error = abs(la - lb - delta) + abs(q(batch.lambda_compute_start[k]) - q(batch.lambda_in[k]))
                equation_budget[name][k] += equation.magnitude / scale
                lambda_budget[name][k] += ideal_denominator.high * lambda_error / scale
                for i, node in enumerate(node_ids):
                    for c in range(3):
                        self._error(observed[i][c], gradients[i][c], 1 / q(self.settings.reference_length))
                        actual_after = q(row[size + i * 3 + c])
                        ideal_after = q(current[node, c]) + original_weights[i] * delta * observed[i][c]
                        gradient_error = (observed[i][c] - gradients[i][c]).magnitude
                        budget[node][c] += abs(actual_after - ideal_after) + original_weights[i] * (
                            abs(delta) * gradient_error + equation.magnitude / ideal_denominator.low * gradients[i][c].magnitude
                            + lambda_error * abs(observed[i][c]))
                        loads[node][c] += delta * observed[i][c]
                current[node_ids] = row[size:2*size].reshape((-1, 3))
                values[name][k] = float(la)
                self.domain(before, current)
                if name == "bending":
                    after = self.bounds(current[node_ids], k)
                    self._error(row[37], after.angle, scale)
                    self._error(row[39], after.constraint, scale)
                    if abs(q(row[37]) - q(row[36])) >= q(np.pi):
                        raise SolverFailure("explicit observed angle branch jump")
            self._equal(batch.compute_end, current, "raw block output")
            self._equal(batch.lambda_compute_end, values[name], "raw lambda output")
            self._equal(batch.positions_out, current, "published block output")
            self._equal(batch.lambda_out, values[name], "published lambda output")
            self.domain(batch.compute_end, batch.positions_out)
        projection_error = max((v / (self.h * q(self.settings.reference_speed)) for row in budget for v in row), default=q(0))
        if max([projection_error, *(v for rows in equation_budget.values() for v in rows),
                *(v for rows in lambda_budget.values() for v in rows)]) > self.tau:
            raise SolverFailure("composite cumulative projection/conversion error exhausted")
        return current, values, (budget, equation_budget, lambda_budget), loads

    def require(self, candidate, effective_inputs, expected_impulses, diagnostics, *, sweeps):
        try:
            return self._require(candidate, effective_inputs, expected_impulses, diagnostics, sweeps=sweeps)
        except (TypeError, ValueError, KeyError, IndexError, ArithmeticError) as error:
            raise SolverFailure(f"invalid composite material evidence: {error}") from error

    def _require(self, candidate, effective_inputs, expected_impulses, diagnostics, *, sweeps):
        path = diagnostics.get("material_path")
        if not isinstance(path, CompositeMaterialPath):
            raise SolverFailure("complete typed composite path required")
        if (fingerprint(self.model) != self.model_id or self.material.identity != self.material_id or path.material_fingerprint != self.material_id
                or path.policy_fingerprint != self.policy_id or path.backend != self.backend
                or path.initial_fingerprint != fingerprint(self.initial)
                or path.input_fingerprint != fingerprint(effective_inputs) or q(path.duration) != self.h
                or len(path.sweeps) != sweeps or sweeps < 1):
            raise SolverFailure("composite path binding or sweep count mismatch")
        expected = np.asarray(expected_impulses)
        if expected.shape != self.initial.position.shape or expected.dtype != np.float64:
            raise SolverFailure("invalid expected held requests")
        native = self.backend != "numpy"
        launch = int(native and self.model.particle_count > 0)
        if (path.prediction_launches, path.reconstruction_launches) != (launch, launch):
            raise SolverFailure("prediction/reconstruction launch evidence mismatch")
        error = q(0)
        for i, w in enumerate(self.model.inverse_mass):
            for c in range(3):
                self._error((q(effective_inputs.force[i, c]) - q(self.original.force[i, c])) * self.h,
                            q(expected[i, c]), 1 / q(w))
                vp = q(self.initial.velocity[i, c]) + self.h * (q(self.model.gravity[c]) + q(w) * q(effective_inputs.force[i, c]))
                xp = q(self.initial.position[i, c]) + self.h * q(path.predicted_velocity[i, c])
                error = max(error, self._error(path.predicted_velocity[i, c], vp, self.settings.reference_speed),
                            self._error(path.predicted_position[i, c], xp, self.h * q(self.settings.reference_speed)))
        self.domain(self.initial.position, path.predicted_position)
        position = path.predicted_position.copy()
        values = {name: np.zeros(self.material.count(name)) for name in self.material.order}
        alphas = {name: getattr(path, name + "_alpha") for name in self.material.order}
        total_budget = [[q(0)] * 3 for _ in position]
        total_equation = {name: [q(0)] * self.material.count(name) for name in self.material.order}
        total_lambda = {name: [q(0)] * self.material.count(name) for name in self.material.order}
        load = [[q(0)] * 3 for _ in position]
        for sweep in path.sweeps:
            position, values, local, contribution = self.audit_sweep(sweep, position, values, alphas)
            for name in self.material.order:
                for k in range(self.material.count(name)):
                    total_equation[name][k] += local[1][name][k]
                    total_lambda[name][k] += local[2][name][k]
            for i in range(len(position)):
                for c in range(3):
                    load[i][c] += contribution[i][c]
                    total_budget[i][c] += local[0][i][c]
        cumulative = max((v / (self.h * q(self.settings.reference_speed)) for row in total_budget for v in row), default=q(0))
        if max([cumulative, *(v for rows in total_equation.values() for v in rows),
                *(v for rows in total_lambda.values() for v in rows)]) > self.tau:
            raise SolverFailure("composite complete-path roundoff budget exhausted")
        self._equal(position, candidate.position, "final configuration")
        for name in self.material.order:
            self._equal(values[name], diagnostics.get(name + "_multipliers"), "final lambda")
        for i in range(len(position)):
            for c in range(3):
                ideal_v = q(path.predicted_velocity[i, c]) + (q(position[i, c]) - q(path.predicted_position[i, c])) / self.h
                error = max(error, self._error(candidate.velocity[i, c], ideal_v, self.settings.reference_speed))
        for name in self.material.order:
            law = getattr(self.material, name)
            if law is None:
                continue
            for k, ids in enumerate(law.indices):
                if name == "bending":
                    constraint = self.bounds(position[ids], k).constraint
                    scale = max(abs(q(law.rest_angle[k])), q(self.policy.reference_angle_rad))
                else:
                    a, b = matrix(position[ids])
                    d = [x-y for x, y in zip(a, b, strict=True)]
                    constraint = sqrt_interval(dot(d, d), self.policy.sqrt_bits) - q(law.rest_length[k])
                    scale = max(q(law.rest_length[k]), q(self.settings.reference_length))
                self._error(0, constraint + q(law.compliance[k]) / self.h**2 * q(values[name][k]), scale)
        physical = self.domain(self.initial.position, position)
        return dict(max_equation_error=float(error), cumulative_projection_error=float(cumulative),
                    material_configuration_load=np.array([[float(v) for v in row] for row in load]).reshape((-1, 3)),
                    numerical_domain_certificates=tuple(self._domains.values()), physical_affine_domain=physical,
                    ordered_material_work=ordered_work(path, self.model, self.initial))


def ordered_work(path, model, initial):
    """Exact stored-value kinetic work, material impulse and NUM005 mechanism.

    Numerical configuration corrections are bookkeeping increments, not physical
    time evolution. Conversion and reconstruction discrepancies stay separate.
    """
    h = q(path.duration)
    predicted_x, predicted_v, x0 = matrix(path.predicted_position), matrix(path.predicted_velocity), matrix(initial.position)
    contributions = []
    for sweep in path.sweeps:
        for batch in sweep.blocks:
            converted_work = q(0)
            for i, w in enumerate(model.inverse_mass):
                u = [predicted_v[i][c] + (q(batch.positions_in[i, c])-predicted_x[i][c]) / h for c in range(3)]
                du = [(q(batch.compute_start[i, c])-q(batch.positions_in[i, c])) / h for c in range(3)]
                converted_work += (dot(u, du) + dot(du, du) / 2) / q(w)
            contributions.append(dict(kind=batch.kind, sweep=sweep.sweep_index,
                                      conversion_kinetic_work=float(converted_work), rows=[]))
            for ids, row in zip(batch.node_indices, batch.rows, strict=True):
                size = len(ids) * 3
                before = row[:size].reshape((-1, 3))
                after = row[size:2*size].reshape((-1, 3))
                if batch.kind == "bending":
                    gradients = row[24:36].reshape((4, 3))
                    delta = q(row[42])
                else:
                    gradients = np.array([row[14:17], -row[14:17]])
                    delta = q(row[17])
                kinetic = impulse_work = q(0)
                local_moment, discrete_moment = [q(0)]*3, [q(0)]*3
                for local, node in enumerate(ids):
                    x = tuple(map(q, before[local]))
                    u = [predicted_v[node][c] + (x[c]-predicted_x[node][c]) / h for c in range(3)]
                    du = [(q(after[local, c])-x[c]) / h for c in range(3)]
                    impulse = [delta*q(g)/h for g in gradients[local]]
                    kinetic += (dot(u, du) + dot(du, du)/2) / q(model.inverse_mass[node])
                    impulse_work += dot(impulse, [u[c]+du[c]/2 for c in range(3)])
                    moment = cross(x, impulse)
                    mechanism = cross([x0[node][c]-x[c] for c in range(3)], impulse)
                    for c in range(3):
                        local_moment[c] += moment[c]
                        discrete_moment[c] += mechanism[c]
                contributions[-1]["rows"].append(dict(kinetic_change=float(kinetic),
                    material_impulse_work=float(impulse_work), arithmetic_work_discrepancy=float(kinetic-impulse_work),
                    local_moment=tuple(map(float, local_moment)), discrete_angular_contribution=tuple(map(float, discrete_moment))))
    return tuple(contributions)





