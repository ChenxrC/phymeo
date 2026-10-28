"""Exact audits of stored canonical robot coefficients; never produce a candidate."""

from fractions import Fraction

import numpy as np

from ..core.capabilities import SolverFailure
from ..dynamics.robot_dynamics import bias_forces, mass_matrix
from ..dynamics.robot_kinematics import point_jacobians


def rational(value):
    return Fraction(float(value))


def dot(left, right):
    return sum((rational(a) * rational(b) for a, b in zip(left, right, strict=True)), Fraction(0))


def exact_float(value):
    result = float(value)
    if not np.isfinite(result) or (value != 0 and result == 0):
        raise SolverFailure("robot audit value is not representable")
    return result


def upper_float(value):
    """Enclose a nonnegative exact residual when passing it to float metrics."""
    result = float(value)
    if value < 0 or not np.isfinite(result):
        raise SolverFailure("invalid exact robot residual")
    if rational(result) < value:
        result = float(np.nextafter(result, np.inf))
    if not np.isfinite(result):
        raise SolverFailure("robot residual overflow")
    return result


class FrozenRobotLedger:
    def __init__(self, model, state, inputs, h, settings, reference_angular_speed):
        raw = mass_matrix(model, state.q)
        self.mass = raw * 0.5 + raw.T * 0.5
        self.n, self.h = model.dof_count, rational(h)
        self.q0, self.v0 = list(map(rational, state.q)), list(map(rational, state.v))
        self.before = list(map(rational, inputs.effort))
        self.scales = [
            rational(reference_angular_speed if kind == "revolute" else settings.reference_speed)
            for kind in model.joint_types
        ]
        self.matrix = [[rational(x) for x in row] for row in self.mass]
        # Exact LDL proves positivity/symmetry of the stored matrix used below.
        self.lower = [[Fraction(i == j) for j in range(self.n)] for i in range(self.n)]
        self.diagonal = []
        for j in range(self.n):
            pivot = self.matrix[j][j] - sum(
                self.lower[j][k] ** 2 * self.diagonal[k] for k in range(j)
            )
            if pivot <= 0:
                raise SolverFailure("canonical frozen robot mass is not positive definite")
            self.diagonal.append(pivot)
            for i in range(j + 1, self.n):
                self.lower[i][j] = (
                    self.matrix[i][j]
                    - sum(self.lower[i][k] * self.lower[j][k] * self.diagonal[k] for k in range(j))
                ) / pivot
        columns = [self.solve([Fraction(i == j) for i in range(self.n)]) for j in range(self.n)]
        self.inverse = np.array(
            [[exact_float(columns[j][i]) for j in range(self.n)] for i in range(self.n)]
        )
        linear, angular = point_jacobians(model, state.q, np.arange(self.n), model.com)
        bias = bias_forces(model, state)
        self.free_integral = []
        for j in range(self.n):
            applied = self.before[j] + sum(
                (
                    dot(linear[i, :, j], inputs.force[i]) + dot(angular[i, :, j], inputs.torque[i])
                    for i in range(self.n)
                ),
                Fraction(0),
            )
            self.free_integral.append(self.h * (applied - rational(bias[j])))

    def solve(self, rhs):
        y = []
        for i in range(self.n):
            y.append(rhs[i] - sum(self.lower[i][k] * y[k] for k in range(i)))
        result = [Fraction(0)] * self.n
        for i in reversed(range(self.n)):
            result[i] = y[i] / self.diagonal[i] - sum(
                self.lower[k][i] * result[k] for k in range(i + 1, self.n)
            )
        return result

    def joint_error(self, velocity_error):
        return max(
            (abs(v) / scale for v, scale in zip(velocity_error, self.scales, strict=True)),
            default=Fraction(0),
        )

    def input_integral(self, effective):
        return [
            (rational(after) - before) * self.h
            for after, before in zip(effective.effort, self.before, strict=True)
        ]

    def trial_audit(self, state, effective, expected):
        actual = self.input_integral(effective)
        dv = [rational(v) - initial for v, initial in zip(state.v, self.v0, strict=True)]
        dynamics = [
            sum(self.matrix[i][j] * dv[j] for j in range(self.n))
            - self.free_integral[i]
            - actual[i]
            for i in range(self.n)
        ]
        motion = [
            (rational(q) - initial) / self.h - rational(v)
            for q, initial, v in zip(state.q, self.q0, state.v, strict=True)
        ]
        return dict(
            dynamics=self.joint_error(self.solve(dynamics)),
            input_integral=self.joint_error(
                self.solve([a - b for a, b in zip(actual, expected, strict=True)])
            ),
            configuration=self.joint_error(motion),
        )

    def mapping_error(self, effective, expected):
        actual = self.input_integral(effective)
        return self.joint_error(self.solve([a - b for a, b in zip(actual, expected, strict=True)]))

    def kinetic(self, velocity):
        v = list(map(rational, velocity))
        return (
            sum(
                (v[i] * self.matrix[i][j] * v[j] for i in range(self.n) for j in range(self.n)),
                Fraction(0),
            )
            / 2
        )
