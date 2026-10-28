"""Independent native XPBD path equations, not contact or conservation acceptance."""

from collections.abc import Mapping
from dataclasses import replace
from numbers import Integral

import numpy as np

from ..constraints.distance import DistanceConstraints
from ..constraints.distance_trace import COLUMNS
from ..core.capabilities import SolverFailure
from ..core.model import ParticleModel
from ..core.scene import BackendContext, StepSettings
from ..core.state import ParticleState, StepInputs
from ..core.validation import validate_step
from ..runtime.program import own_diagnostics
from ..solvers._robot_replay_ledger import upper_float
from ._material_path_math import (
    ZERO,
    array,
    bookkeeping,
    compute,
    direction_bounds,
    display_array,
    matrix,
    maxabs,
    norm_bounds,
    q,
    zero_matrix,
)


class MaterialPathLedger:
    """Require recorded discrete equations without changing the native candidate.

    Physical energy/angular-momentum budgets and coupled contact laws remain the
    caller's responsibility. Inputs, parameters and the initial state are owned.
    """

    def __init__(
        self, model, constraints, initial, original_inputs, h, settings, context=BackendContext()
    ):
        if not isinstance(settings, StepSettings) or settings.tolerance <= 0:
            raise ValueError("material ledger requires positive StepSettings tolerance")
        if (
            not isinstance(context, BackendContext)
            or context.dtype != np.float64
            or context.backend not in ("numpy", "tyx:interpret", "tyx:cpu_cpp", "tyx:cuda")
        ):
            raise ValueError("unsupported material ledger context")
        if not isinstance(model, ParticleModel) or model.inverse_mass.dtype != np.float64:
            raise TypeError("material ledger requires an FP64 ParticleModel")
        initial, force, step = validate_step(model, initial, original_inputs, h)
        if np.any(initial.velocity[model.inverse_mass == 0] != 0):
            raise ValueError("material ledger fixed nodes must start stationary")
        if constraints is not None:
            if (
                not isinstance(constraints, DistanceConstraints)
                or constraints.rest_length.dtype != np.float64
            ):
                raise TypeError("material ledger requires FP64 distance constraints")
            if np.any(constraints.indices >= model.particle_count):
                raise ValueError("material ledger edge index outside domain")
            constraints = replace(constraints)
        self._model, self._constraints = replace(model), constraints
        self._initial, self._original = initial, StepInputs(force)
        self._h, self._settings, self._context = q(step), settings, context
        self._count = model.particle_count
        self._edges = 0 if constraints is None else len(constraints.indices)
        self._speed, self._tol = q(settings.reference_speed), q(settings.tolerance)

    def require(self, candidate, effective_inputs, expected_impulses, diagnostics, *, sweeps):
        try:
            return self._require(
                candidate, effective_inputs, expected_impulses, diagnostics, sweeps
            )
        except SolverFailure:
            raise
        except (TypeError, ValueError, KeyError, IndexError, ArithmeticError) as error:
            raise SolverFailure(f"invalid material ledger evidence: {error}") from error

    def _require(self, candidate, effective, expected, diagnostics, sweeps):
        n, m, h, model = self._count, self._edges, self._h, self._model
        if not isinstance(candidate, ParticleState) or not isinstance(effective, StepInputs):
            raise SolverFailure("material candidate and inputs must have particle types")
        for value, name in (
            (candidate.position, "candidate position"),
            (candidate.velocity, "candidate velocity"),
            (effective.force, "effective force"),
        ):
            array(value, (n, 3), name)
        raw = np.asarray(expected, dtype=object)
        if raw.shape != (n, 3):
            raise SolverFailure("material request shape mismatch")
        expected = matrix(raw)
        if not isinstance(diagnostics, Mapping) or "material_path" not in diagnostics:
            raise SolverFailure("missing material path")
        path = diagnostics["material_path"]
        if (
            not isinstance(path, Mapping)
            or path.get("schema") != "xpbd-distance-path-v1"
            or tuple(path.get("columns", ())) != COLUMNS
            or q(path["h"]) != h
        ):
            raise SolverFailure("material path schema or interval mismatch")
        frames = path["frames"]
        if (
            isinstance(sweeps, (bool, np.bool_))
            or not isinstance(sweeps, Integral)
            or not 1 <= sweeps <= self._settings.position_iterations
            or not isinstance(frames, tuple)
            or len(frames) != sweeps
        ):
            raise SolverFailure("material path sweep schedule mismatch")
        px = array(path["predicted_position"], (n, 3), "prediction position")
        pv = array(path["predicted_velocity"], (n, 3), "prediction velocity")
        alpha = array(path["alpha"], (m,), "alpha")
        final_lambda = array(diagnostics["distance_multipliers"], (m,), "multipliers")
        is_device = self._context.backend != "numpy"
        if is_device:
            execution = diagnostics["execution"]
            if not isinstance(execution, Mapping):
                raise SolverFailure("material execution must be a mapping")
            launches = execution.get("completed_launches", 0)
            if (
                diagnostics.get("compute_dtype") != "float32"
                or execution.get("mode") != self._context.backend[4:]
                or execution.get("compute_dtype") != "float32"
                or isinstance(launches, (bool, np.bool_))
                or not isinstance(launches, Integral)
                or launches <= 0
            ):
                raise SolverFailure("material execution contract mismatch")
        residual = {
            name: ZERO
            for name in (
                "input",
                "prediction_velocity",
                "prediction_position",
                "alpha",
                "edge_equation",
                "lambda_update",
                "direction",
                "projection",
                "rounding_budget",
                "velocity",
                "constitutive",
            )
        }

        def check(name, value):
            residual[name] = max(residual[name], value)
            if value > self._tol:
                raise SolverFailure(f"material {name} exceeds original tolerance")

        for i, inverse in enumerate(model.inverse_mass):
            w = q(inverse)
            for k in range(3):
                added = h * (q(effective.force[i, k]) - q(self._original.force[i, k]))
                if w == 0:
                    if (
                        expected[i][k] != 0
                        or added != 0
                        or px[i, k] != self._initial.position[i, k]
                        or pv[i, k] != 0
                        or candidate.velocity[i, k] != 0
                    ):
                        raise SolverFailure("material fixed-node boundary violated")
                else:
                    check("input", abs(added - expected[i][k]) * w / self._speed)
                    vp = q(self._initial.velocity[i, k]) + h * (
                        q(model.gravity[k]) + w * q(effective.force[i, k])
                    )
                    check("prediction_velocity", abs(q(pv[i, k]) - vp) / self._speed)
                    check(
                        "prediction_position",
                        abs(q(px[i, k]) - q(self._initial.position[i, k]) - h * q(pv[i, k]))
                        / (h * self._speed),
                    )
        host = px.copy()
        multipliers = np.zeros(m)
        load, conversion, rounding, restoration, budget = (zero_matrix(n) for _ in range(5))
        edge_error = [ZERO] * m
        lambda_error = [ZERO] * m
        constraints = self._constraints
        for e in range(m):
            i, j = constraints.indices[e]
            exact = q(constraints.compliance[e]) / (h * h)
            scale = max(exact, q(model.inverse_mass[i]) + q(model.inverse_mass[j]))
            if scale == 0:
                if alpha[e] != 0:
                    raise SolverFailure("zero denominator has nonzero alpha")
            else:
                check("alpha", abs(q(alpha[e]) - exact) / scale)
        for frame in frames:
            array(frame, (m, 18), "frame")
            if is_device and not np.array_equal(frame, compute(frame)):
                raise SolverFailure("device material frame is not stored FP32")
            working = compute(host) if is_device and m else host.copy()
            for i in range(n):
                for k in range(3):
                    dx = q(working[i, k]) - q(host[i, k])
                    conversion[i][k] += dx
                    budget[i][k] += abs(dx)
                    check("rounding_budget", budget[i][k] / (h * self._speed))
            for e, (i, j) in enumerate(constraints.indices if constraints is not None else ()):
                row = frame[e]
                if (
                    not np.array_equal(row[:3], working[i])
                    or not np.array_equal(row[3:6], working[j])
                    or row[12] != multipliers[e]
                ):
                    raise SolverFailure("material path continuity mismatch")
                wi, wj = q(model.inverse_mass[i]), q(model.inverse_mass[j])
                a = q(constraints.compliance[e]) / (h * h)
                denominator = wi + wj + a
                dl = q(row[17])
                direction = [q(v) for v in row[14:17]]
                if denominator == 0:
                    if (
                        dl != 0
                        or any(direction)
                        or row[13] != row[12]
                        or not np.array_equal(row[:6], row[6:12])
                    ):
                        raise SolverFailure("invalid skipped material projection")
                else:
                    lengths, normal = direction_bounds(
                        [q(row[k]) - q(row[k + 3]) for k in range(3)]
                    )
                    rest = q(constraints.rest_length[e])
                    le = max(rest, q(self._settings.reference_length))
                    constant = -rest + a * q(row[12]) + denominator * dl
                    equation = maxabs((lengths[0] + constant, lengths[1] + constant))
                    edge_error[e] += equation / le
                    lambda_error[e] += denominator * abs(q(row[13]) - q(row[12]) - dl) / le
                    check("edge_equation", edge_error[e])
                    check("lambda_update", lambda_error[e])
                    for k in range(3):
                        derror = maxabs((direction[k] - normal[k][0], direction[k] - normal[k][1]))
                        check("direction", derror)
                        for node, weight, before, after, sign in (
                            (i, wi, k, k + 6, 1),
                            (j, wj, k + 3, k + 9, -1),
                        ):
                            actual = q(row[after]) - q(row[before])
                            ideal = sign * weight * dl * direction[k]
                            error = actual - ideal
                            if weight == 0 and actual != 0:
                                raise SolverFailure("material projection moved a fixed node")
                            rounding[node][k] += error
                            budget[node][k] += abs(error) + weight * (
                                abs(dl) * derror + equation / denominator
                            )
                            check("projection", abs(error) / (h * self._speed))
                            check("rounding_budget", budget[node][k] / (h * self._speed))
                for k in range(3):
                    load[i][k] += dl * direction[k]
                    load[j][k] -= dl * direction[k]
                working[i], working[j], multipliers[e] = row[6:9], row[9:12], row[13]
            for i, inverse in enumerate(model.inverse_mass):
                if inverse == 0:
                    for k in range(3):
                        restoration[i][k] += q(host[i, k]) - q(working[i, k])
                    working[i] = host[i]
            host = working
        if not np.array_equal(candidate.position, host) or not np.array_equal(
            final_lambda, multipliers
        ):
            raise SolverFailure("unrecorded material final position or multiplier")
        for i, inverse in enumerate(model.inverse_mass):
            if inverse > 0:
                for k in range(3):
                    check(
                        "velocity",
                        abs(
                            q(candidate.velocity[i, k])
                            - q(pv[i, k])
                            - (q(candidate.position[i, k]) - q(px[i, k])) / h
                        )
                        / self._speed,
                    )
        for e, (i, j) in enumerate(constraints.indices if constraints is not None else ()):
            lengths = norm_bounds([q(host[i, k]) - q(host[j, k]) for k in range(3)])
            offset = -q(constraints.rest_length[e]) + q(constraints.compliance[e]) / (h * h) * q(
                final_lambda[e]
            )
            check(
                "constitutive",
                maxabs((lengths[0] + offset, lengths[1] + offset))
                / max(q(constraints.rest_length[e]), q(self._settings.reference_length)),
            )
        quantities = bookkeeping(
            model,
            constraints,
            self._initial,
            candidate,
            effective,
            self._original,
            expected,
            load,
            h,
        )
        return own_diagnostics(
            dict(
                scope="distance-material-equations-v1",
                residuals={k: upper_float(v) for k, v in residual.items()},
                max_residual=upper_float(max(residual.values())),
                conversion_displacement=display_array(conversion),
                projection_rounding_displacement=display_array(rounding),
                fixed_restoration_displacement=display_array(restoration),
                **quantities,
            )
        )
