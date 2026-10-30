"""Independent exact world-coordinate reader for represented native evidence."""

import math
import struct
from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from ..constraints.material_representation import (
    CODEC_BITS,
    SCHEMA,
    NativeCompositeRepresentation,
    RepresentationBoundary,
    RepresentedMaterialBatch,
    RepresentedMaterialPath,
    RepresentedMaterialSweep,
)
from ..core.capabilities import SolverFailure
from ..core.scene import fingerprint
from ._material_path_math import compute, matrix, q
from .bending_bounds import Interval, bending_bounds, dot, sqrt_interval
from .bending_domain import certify_bending_segment
from .composite_material_path import ordered_work as _world_ordered_work


def _bounded(value):
    value = value if isinstance(value, Fraction) else Fraction(float(value))
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > CODEC_BITS:
        raise SolverFailure("independent representation codec resources exhausted")
    return value


def _neighbors(word, bits):
    fmt, integer = (">f", ">I") if bits == 32 else (">d", ">Q")
    raw = struct.unpack(integer, struct.pack(fmt, abs(float(word))))[0]
    center = Fraction(abs(float(word)))
    precision, emin = (24, -126) if bits == 32 else (53, -1022)
    if raw == 0:
        spacing = Fraction(2) ** (emin - precision + 1)
        return -spacing, spacing, raw
    lower = Fraction(struct.unpack(fmt, struct.pack(integer, raw - 1))[0])
    higher_float = struct.unpack(fmt, struct.pack(integer, raw + 1))[0]
    higher = Fraction(higher_float) if math.isfinite(higher_float) else center + (center - lower)
    if word < 0:
        return -higher, -lower, raw
    return lower, higher, raw


def verify_rounding(source, word, bits):
    """Independent cell/bracket/tie-parity proof; does not call the producer codec."""
    source = _bounded(source)
    if bits not in (32, 64) or not math.isfinite(float(word)):
        raise SolverFailure("invalid represented rounded word")
    if bits == 32:
        try:
            if struct.unpack(">f", struct.pack(">f", float(word)))[0] != word:
                raise SolverFailure("word is not actual FP32")
        except OverflowError as error:
            raise SolverFailure("native word overflow") from error
        if source and (not word or abs(float(word)) < 2.0**-126):
            raise SolverFailure("unsupported native subnormal/zero")
    if not word and math.copysign(1.0, word) < 0:
        raise SolverFailure("computed zero must be positive")
    lower, higher, raw = _neighbors(word, bits)
    center = Fraction(float(word))
    left, right = (center + lower) / 2, (center + higher) / 2
    if not left <= source <= right or ((source == left or source == right) and raw % 2):
        raise SolverFailure("independent representation rounding mismatch")
    return center - source


def _rounded32(source):
    source = _bounded(source)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        center = np.float32(float(source))
        candidates = (
            center,
            np.nextafter(center, np.float32(-np.inf)),
            np.nextafter(center, np.float32(np.inf)),
        )
    for candidate in candidates:
        if candidate == 0:
            candidate = np.float32(0)
        try:
            verify_rounding(source, float(candidate), 32)
            return candidate
        except SolverFailure:
            pass
    raise SolverFailure("independent native encoding unsupported")


def _decode_world(reference, displacement, worlds=None, witnesses=None):
    reference, displacement = np.asarray(reference), np.asarray(displacement)
    if (
        reference.dtype != np.float64
        or displacement.dtype != np.float32
        or reference.shape != displacement.shape
    ):
        raise SolverFailure("invalid represented decode shape/dtype")
    output = np.empty(reference.shape, np.float64)
    if witnesses is not None and len(witnesses) != reference.size:
        raise SolverFailure("incomplete decoder witnesses")
    for k, (r, d) in enumerate(zip(reference.flat, displacement.flat, strict=True)):
        if d and abs(float(d)) < 2.0**-126:
            raise SolverFailure("unsupported displacement subnormal")
        source = _bounded(q(r) + q(d))
        value = float(source)
        if value == 0:
            value = 0.0
        verify_rounding(source, value, 64)
        output.flat[k] = value
        if witnesses is not None:
            witness = witnesses[k]
            if (
                witness.bits != 64
                or (witness.numerator, witness.denominator)
                != (source.numerator, source.denominator)
                or witness.word != value
            ):
                raise SolverFailure("decoder witness binding mismatch")
    if worlds is not None and (
        output.dtype != worlds.dtype
        or output.shape != worlds.shape
        or output.tobytes() != worlds.tobytes()
    ):
        raise SolverFailure("derived world decode mismatch")
    return output


def projection_charge(reference, before, after, product, weight, delta, gradient):
    """Return exact conservative charge and direct residual, never float acceptance."""
    r, b, a, t, w, dl, g = map(q, (reference, before, after, product, weight, delta, gradient))
    xb, xa = float(_bounded(r + b)), float(_bounded(r + a))
    rho = verify_rounding(r + b, xb, 64)
    rho_after = verify_rounding(r + a, xa, 64)
    addition, multiplication = a - b - t, t - w * dl * g
    direct_signed = q(xa) - q(xb) - w * dl * g
    if direct_signed != addition + multiplication + rho_after - rho:
        raise SolverFailure("represented projection identity mismatch")
    charge = abs(addition) + abs(multiplication) + abs(rho_after) + abs(rho)
    if abs(direct_signed) > charge:
        raise SolverFailure("represented charge failed to bound world residual")
    return charge, abs(direct_signed), (addition, multiplication, rho, rho_after)


def _directed(reference, ids, kind):
    vectors = 1 if kind == "distance" else 3
    result = np.empty((len(ids), vectors, 3), np.float32)
    sources = []
    for k, row in enumerate(ids):
        pairs = (
            ((row[0], row[1]),)
            if kind == "distance"
            else ((row[1], row[0]), (row[2], row[0]), (row[3], row[1]))
        )
        for j, (a, b) in enumerate(pairs):
            for c in range(3):
                value = _bounded(q(reference[a, c]) - q(reference[b, c]))
                sources.append(value)
                result[k, j, c] = _rounded32(value)
    return result, tuple(sources)


def _check_geometry(batch, k, displacement, table, actual):
    ids = batch.node_indices[k]
    pairs = (
        ((ids[0], ids[1]),)
        if batch.kind == "distance"
        else ((ids[1], ids[0]), (ids[2], ids[0]), (ids[3], ids[1]))
    )
    for j, (a, b) in enumerate(pairs):
        for c in range(3):
            difference = _rounded32(q(displacement[a, c]) - q(displacement[b, c]))
            verify_rounding(q(table[k, j, c]) + q(difference), actual[j, c], 32)


def _represented_work(path, model, initial):
    # This internal world view is explicitly derived, never a native/v1 receipt.
    sweeps = []
    for sweep in path.sweeps:
        blocks = []
        for batch in sweep.blocks:
            size = batch.node_indices.shape[1] * 3
            world_rows = batch.rows.astype(np.float64)
            world_rows[:, :size] = batch.world_before.reshape((-1, size))
            world_rows[:, size : 2 * size] = batch.world_after.reshape((-1, size))
            blocks.append(
                SimpleNamespace(
                    kind=batch.kind,
                    node_indices=batch.node_indices,
                    positions_in=batch.positions_in,
                    compute_start=batch.positions_in,
                    rows=world_rows,
                )
            )
        sweeps.append(SimpleNamespace(sweep_index=sweep.sweep_index, blocks=blocks))
    return _world_ordered_work(
        SimpleNamespace(
            duration=path.duration,
            predicted_position=path.predicted_position,
            predicted_velocity=path.predicted_velocity,
            sweeps=sweeps,
        ),
        model,
        initial,
    )


class RepresentedMaterialPathLedger:
    def __init__(
        self,
        model,
        material,
        initial,
        original_inputs,
        h,
        settings,
        *,
        audit_settings,
        backend="tyx:interpret",
        representation=None,
        domain="cloth",
    ):
        material.validate_model(model)
        self.model, self.material, self.initial = model, material, initial.copy()
        self.original = type(original_inputs)(original_inputs.force.copy())
        self.h, self.settings, self.policy, self.backend = q(h), settings, audit_settings, backend
        self.material_id, self.policy_id = material.identity, fingerprint(audit_settings)
        self.model_id = fingerprint(model)
        if not isinstance(representation, NativeCompositeRepresentation):
            raise TypeError("represented ledger requires explicit policy")
        self.representation, self.domain_name = representation, domain
        self.reference = self.initial.position.copy()
        self.reference_id = fingerprint(
            (
                self.reference,
                self.material.identity,
                representation,
                domain,
                float(h),
                fingerprint(self.initial),
                fingerprint(self.original),
            )
        )
        self.tau = q(settings.tolerance)
        self._domains, self._bounds = {}, {}
        if self.h <= 0 or backend not in ("tyx:interpret", "tyx:cpu_cpp", "tyx:cuda"):
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
            self._domains[key] = certify_bending_segment(
                before, after, self.material, policy=self.policy
            ).require()
        return self._domains[key]

    def bounds(self, nodes, k):
        key = fingerprint((nodes, k))
        if key not in self._bounds:
            law = self.material.bending
            self._bounds[key] = bending_bounds(
                nodes, law.rest_angle[k], law.compliance[k], policy=self.policy
            )
        return self._bounds[key]

    def audit_sweep(self, sweep, position, multipliers, alphas, *, displacement):
        domains, bounds = self._domains.copy(), self._bounds.copy()
        try:
            return self._audit_sweep(
                sweep, position, multipliers, alphas, displacement=displacement
            )
        except BaseException:
            self._domains, self._bounds = domains, bounds
            raise

    def _audit_sweep(self, sweep, position, multipliers, alphas, *, displacement):
        """Validate a private full sweep before its owner publishes any part."""
        if not isinstance(sweep, RepresentedMaterialSweep):
            raise SolverFailure("typed represented sweep required")
        encoded = np.array(displacement, copy=True)
        current = position.copy()
        _decode_world(self.reference, encoded, current)
        values = {name: multipliers[name].copy() for name in self.material.order}
        budget = [[q(0)] * 3 for _ in current]
        equation_budget = {name: [q(0)] * self.material.count(name) for name in self.material.order}
        lambda_budget = {name: [q(0)] * self.material.count(name) for name in self.material.order}
        loads = [[q(0)] * 3 for _ in current]
        row_charges = []
        for batch in sweep.blocks:
            name, law = batch.kind, getattr(self.material, batch.kind)
            count = self.material.count(name)
            ids = (
                np.empty((0, 2 if name == "distance" else 4), np.int64)
                if law is None
                else law.indices
            )
            rest = (
                np.zeros(0)
                if law is None
                else (law.rest_length if name == "distance" else law.rest_angle)
            )
            if (
                not isinstance(batch, RepresentedMaterialBatch)
                or batch.reference_fingerprint != self.reference_id
            ):
                raise SolverFailure("represented batch/reference identity mismatch")
            self._same(batch.node_indices, ids, "ordered identities")
            self._same(batch.displacement_in, encoded, "encoded block input")
            _decode_world(self.reference, encoded, batch.positions_in)
            self._equal(batch.positions_in, current, "block input")
            self._equal(batch.lambda_in, values[name], "lambda input")
            self._equal(batch.compute_weights, compute(self.model.inverse_mass), "mass conversion")
            self._equal(batch.compute_rest, compute(rest), "rest conversion")
            self._equal(batch.compute_alpha, compute(alphas[name]), "alpha conversion")
            required_launches = int(count > 0)
            if (batch.launches, batch.completed_launches) != (required_launches, required_launches):
                raise SolverFailure("missing or forged operation-specific represented launch")
            reference_vectors, _ = _directed(self.reference, ids, name)
            self._same(batch.reference_vectors, reference_vectors, "actual reference vectors")
            values[name] = batch.lambda_in.astype(np.float64)
            for k, node_ids in enumerate(ids):
                raw = batch.rows[k]
                size = len(node_ids) * 3
                raw_before = raw[:size].reshape((-1, 3))
                raw_after = raw[size : 2 * size].reshape((-1, 3))
                self._same(raw_before, encoded[node_ids], "native row before")
                world_before = _decode_world(
                    self.reference[node_ids],
                    raw_before,
                    batch.world_before[k],
                    batch.decode_before[k * size : (k + 1) * size],
                )
                world_after = _decode_world(
                    self.reference[node_ids],
                    raw_after,
                    batch.world_after[k],
                    batch.decode_after[k * size : (k + 1) * size],
                )
                _check_geometry(batch, k, encoded, reference_vectors, batch.geometry_before[k])
                row = raw.astype(np.float64)
                row[:size], row[size : 2 * size] = world_before.ravel(), world_after.ravel()
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
                    gradient = tuple(
                        Interval.point(int(c == 0))
                        if length.high == 0
                        else Interval.point(d[c]) / length
                        for c in range(3)
                    )
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
                    observed = tuple(tuple(map(q, row[24 + 3 * i : 27 + 3 * i])) for i in range(4))
                    scale = max(abs(q(law.rest_angle[k])), q(self.policy.reference_angle_rad))
                    self._error(row[36], bounded.angle, scale)
                    self._error(row[38], bounded.constraint, scale)
                    self._error(row[43], alpha, alpha)
                    denominator = sum(
                        (
                            weights[i] * sum((g.square() for g in gradients[i]), Interval.point(0))
                            for i in range(4)
                        ),
                        Interval.point(alpha),
                    )
                    self._error(row[44], denominator, denominator.low)
                    ideal_denominator = sum(
                        (
                            original_weights[i]
                            * sum((g.square() for g in gradients[i]), Interval.point(0))
                            for i in range(4)
                        ),
                        Interval.point(ideal_alpha),
                    )
                    equation = bounded.constraint + ideal_alpha * lb + ideal_denominator * delta
                    self._error(0, equation, scale)
                if lb != q(values[name][k]):
                    raise SolverFailure("lambda row path mismatch")
                lambda_error = abs(la - lb - delta)
                verify_rounding(lb + delta, float(la), 32)
                equation_budget[name][k] += equation.magnitude / scale
                lambda_budget[name][k] += ideal_denominator.high * lambda_error / scale
                for i, node in enumerate(node_ids):
                    for c in range(3):
                        self._error(
                            observed[i][c], gradients[i][c], 1 / q(self.settings.reference_length)
                        )
                        actual_after = q(row[size + i * 3 + c])
                        ideal_after = (
                            q(current[node, c]) + original_weights[i] * delta * observed[i][c]
                        )
                        gradient_error = (observed[i][c] - gradients[i][c]).magnitude
                        product = batch.products[k, i, c]
                        first_product = _rounded32(weights[i] * delta)
                        verify_rounding(q(first_product) * observed[i][c], product, 32)
                        verify_rounding(q(raw_before[i, c]) + q(product), raw_after[i, c], 32)
                        charge, direct, terms = projection_charge(
                            self.reference[node, c],
                            raw_before[i, c],
                            raw_after[i, c],
                            product,
                            self.model.inverse_mass[node],
                            delta,
                            observed[i][c],
                        )
                        if direct != abs(actual_after - ideal_after):
                            raise SolverFailure("direct represented world residual mismatch")
                        row_charges.append(
                            dict(
                                sweep=sweep.sweep_index,
                                kind=name,
                                row=k,
                                node=int(node),
                                component=c,
                                charge=(charge.numerator, charge.denominator),
                                direct=(direct.numerator, direct.denominator),
                                terms=tuple((term.numerator, term.denominator) for term in terms),
                            )
                        )
                        budget[node][c] += charge + original_weights[i] * (
                            abs(delta) * gradient_error
                            + equation.magnitude / ideal_denominator.low * gradients[i][c].magnitude
                            + lambda_error * abs(observed[i][c])
                        )
                        loads[node][c] += delta * observed[i][c]
                current[node_ids] = row[size : 2 * size].reshape((-1, 3))
                encoded[node_ids] = raw_after
                _check_geometry(batch, k, encoded, reference_vectors, batch.geometry_after[k])
                values[name][k] = float(la)
                self.domain(before, current)
                if name == "bending":
                    after = self.bounds(current[node_ids], k)
                    self._error(row[37], after.angle, scale)
                    self._error(row[39], after.constraint, scale)
                    if abs(q(row[37]) - q(row[36])) >= q(np.pi):
                        raise SolverFailure("explicit observed angle branch jump")
            self._same(batch.displacement_out, encoded, "encoded block output")
            _decode_world(self.reference, encoded, batch.positions_out)
            self._equal(batch.positions_out, current, "published decoded block output")
            self._equal(batch.lambda_out, values[name], "raw lambda output")
        projection_error = max(
            (v / (self.h * q(self.settings.reference_speed)) for row in budget for v in row),
            default=q(0),
        )
        if (
            max(
                [
                    projection_error,
                    *(v for rows in equation_budget.values() for v in rows),
                    *(v for rows in lambda_budget.values() for v in rows),
                ]
            )
            > self.tau
        ):
            raise SolverFailure("composite cumulative projection/conversion error exhausted")
        return current, values, (budget, equation_budget, lambda_budget, tuple(row_charges)), loads

    @staticmethod
    def _same(actual, expected, name):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if (
            actual.dtype != expected.dtype
            or actual.shape != expected.shape
            or actual.tobytes() != expected.tobytes()
        ):
            raise SolverFailure("represented " + name + " byte binding mismatch")

    def _boundary(self, boundary, name, sources, shape):
        if (
            not isinstance(boundary, RepresentationBoundary)
            or boundary.name != name
            or boundary.computed.shape != shape
        ):
            raise SolverFailure("representation boundary sequence/shape mismatch")
        sources = tuple(_bounded(v) for v in sources)
        if len(sources) != len(boundary.witnesses):
            raise SolverFailure("representation boundary witness count mismatch")
        for source, witness, word in zip(
            sources, boundary.witnesses, boundary.computed.flat, strict=True
        ):
            if (witness.numerator, witness.denominator, witness.bits, witness.word) != (
                source.numerator,
                source.denominator,
                32,
                float(word),
            ):
                raise SolverFailure("representation boundary original binding mismatch")
            verify_rounding(source, float(word), 32)
        return boundary.computed

    def _operation_binding(self, operation, name, inputs, outputs, launches):
        if operation.name != name or (operation.launches, operation.completed_launches) != (
            launches,
            launches,
        ):
            raise SolverFailure("represented operation-specific launch mismatch")
        if len(operation.inputs) != len(inputs) or len(operation.outputs) != len(outputs):
            raise SolverFailure("represented operation operands missing")
        for a, b in zip(
            operation.inputs + operation.outputs, tuple(inputs) + tuple(outputs), strict=True
        ):
            self._same(a, b, "operation operand")

    def require(self, candidate, effective_inputs, expected_impulses, diagnostics, *, sweeps):
        domains, bounds = self._domains.copy(), self._bounds.copy()
        try:
            return self._require(
                candidate, effective_inputs, expected_impulses, diagnostics, sweeps=sweeps
            )
        except (TypeError, ValueError, KeyError, IndexError, ArithmeticError) as error:
            self._domains, self._bounds = domains, bounds
            raise SolverFailure(f"invalid represented material evidence: {error}") from error
        except BaseException:
            self._domains, self._bounds = domains, bounds
            raise

    def _require(self, candidate, effective_inputs, expected_impulses, diagnostics, *, sweeps):
        path = diagnostics.get("material_path")
        if not isinstance(path, RepresentedMaterialPath) or path.schema != SCHEMA:
            raise SolverFailure("complete typed v2 represented path required")
        if (
            fingerprint(self.model) != self.model_id
            or self.material.identity != self.material_id
            or path.material_fingerprint != self.material_id
            or path.policy_fingerprint != self.policy_id
            or path.backend != self.backend
            or path.initial_fingerprint != fingerprint(self.initial)
            or path.input_fingerprint != fingerprint(effective_inputs)
            or q(path.duration) != self.h
            or path.representation != self.representation
            or path.domain != self.domain_name
            or path.reference_fingerprint != self.reference_id
            or len(path.sweeps) != sweeps
            or sweeps < 1
        ):
            raise SolverFailure("represented path identity/policy/trial binding mismatch")
        self._same(path.reference, self.reference, "immutable trial reference")
        self._same(diagnostics.get("represented_reference"), self.reference, "hidden reference")
        self._same(
            diagnostics.get("represented_displacement"),
            path.final_displacement,
            "hidden displacement",
        )
        expected = np.asarray(expected_impulses)
        if expected.shape != self.initial.position.shape or expected.dtype != np.float64:
            raise SolverFailure("invalid expected held requests")
        boundary_values = {}
        specifications = []
        for name, original in (
            ("predict/velocity", self.initial.velocity),
            ("predict/force", effective_inputs.force),
            ("predict/weights", self.model.inverse_mass),
            ("predict/gravity", self.model.gravity),
            ("predict/h", np.asarray(float(self.h))),
        ):
            specifications.append(
                (name, tuple(q(v) for v in np.asarray(original).flat), np.asarray(original).shape)
            )
        specifications.append(("alpha/inverse_h", (1 / self.h,), ()))
        for name in self.material.order:
            law = getattr(self.material, name)
            ids = (
                np.empty((0, 2 if name == "distance" else 4), np.int64)
                if law is None
                else law.indices
            )
            rest = (
                np.zeros(0)
                if law is None
                else (law.rest_length if name == "distance" else law.rest_angle)
            )
            compliance = np.zeros(0) if law is None else law.compliance
            specifications.append((name + "/rest", tuple(map(q, rest)), rest.shape))
            specifications.append(
                (name + "/compliance", tuple(map(q, compliance)), compliance.shape)
            )
            vectors, sources = _directed(self.reference, ids, name)
            specifications.append(("reference/" + name, sources, vectors.shape))
        if len(path.compute_boundaries) != len(specifications):
            raise SolverFailure("missing represented compute boundary")
        for boundary, (name, sources, shape) in zip(
            path.compute_boundaries, specifications, strict=True
        ):
            boundary_values[name] = self._boundary(boundary, name, sources, shape)
        launch = int(self.model.particle_count > 0)
        inputs = tuple(
            boundary_values[name]
            for name in (
                "predict/velocity",
                "predict/force",
                "predict/weights",
                "predict/gravity",
                "predict/h",
            )
        )
        vp32 = path.predicted_velocity.astype(np.float32)
        self._equal(vp32.astype(np.float64), path.predicted_velocity, "native predicted velocity")
        self._operation_binding(
            path.prediction_operation,
            "predict",
            inputs,
            (path.predicted_displacement, vp32),
            launch,
        )
        velocity, force, weights, gravity, h_native = inputs
        for i in range(len(velocity)):
            for c in range(3):
                acceleration = _rounded32(
                    q(gravity[c]) + q(_rounded32(q(weights[i]) * q(force[i, c])))
                )
                predicted = _rounded32(
                    q(velocity[i, c]) + q(_rounded32(q(float(h_native)) * q(acceleration)))
                )
                self._equal(vp32[i, c], predicted, "actual prediction arithmetic")
                verify_rounding(
                    q(float(h_native)) * q(predicted), path.predicted_displacement[i, c], 32
                )
        _decode_world(self.reference, path.predicted_displacement, path.predicted_position)
        self.domain(self.initial.position, path.predicted_position)
        alpha_operations = iter(path.alpha_operations)
        for name in self.material.order:
            actual = getattr(path, name + "_alpha")
            values = boundary_values[name + "/compliance"]
            inverse = boundary_values["alpha/inverse_h"]
            if self.material.count(name):
                for phase in range(2):
                    operation = next(alpha_operations, None)
                    if operation is None or len(operation.outputs) != 1:
                        raise SolverFailure("missing represented alpha operation")
                    result = operation.outputs[0]
                    self._operation_binding(
                        operation,
                        name + "/alpha/" + str(phase),
                        (np.zeros_like(values), values, inverse),
                        (result,),
                        1,
                    )
                    if result.shape != values.shape:
                        raise SolverFailure("represented alpha shape mismatch")
                    for before, after in zip(values, result, strict=True):
                        verify_rounding(q(before) * q(float(inverse)), after, 32)
                    values = result
            self._same(actual, values, "native alpha")
        if next(alpha_operations, None) is not None:
            raise SolverFailure("extra represented alpha operation")
        self._operation_binding(
            path.reconstruction_operation,
            "reconstruct",
            (path.predicted_displacement, vp32, path.final_displacement, h_native),
            (path.reconstructed_velocity,),
            launch,
        )
        for i in range(len(velocity)):
            for c in range(3):
                difference = _rounded32(
                    q(path.final_displacement[i, c]) - q(path.predicted_displacement[i, c])
                )
                quotient = _rounded32(q(difference) / q(float(h_native)))
                verify_rounding(q(vp32[i, c]) + q(quotient), path.reconstructed_velocity[i, c], 32)
        self._equal(
            candidate.velocity,
            path.reconstructed_velocity.astype(np.float64),
            "published native velocity",
        )
        error = q(0)
        for i, w in enumerate(self.model.inverse_mass):
            for c in range(3):
                self._error(
                    (q(effective_inputs.force[i, c]) - q(self.original.force[i, c])) * self.h,
                    q(expected[i, c]),
                    1 / q(w),
                )
                vp = q(self.initial.velocity[i, c]) + self.h * (
                    q(self.model.gravity[c]) + q(w) * q(effective_inputs.force[i, c])
                )
                xp = q(self.initial.position[i, c]) + self.h * q(path.predicted_velocity[i, c])
                error = max(
                    error,
                    self._error(path.predicted_velocity[i, c], vp, self.settings.reference_speed),
                    self._error(
                        path.predicted_position[i, c], xp, self.h * q(self.settings.reference_speed)
                    ),
                )
        self.domain(self.initial.position, path.predicted_position)
        position = path.predicted_position.copy()
        values = {name: np.zeros(self.material.count(name)) for name in self.material.order}
        alphas = {name: getattr(path, name + "_alpha") for name in self.material.order}
        total_budget = [[q(0)] * 3 for _ in position]
        total_equation = {name: [q(0)] * self.material.count(name) for name in self.material.order}
        total_lambda = {name: [q(0)] * self.material.count(name) for name in self.material.order}
        load = [[q(0)] * 3 for _ in position]
        row_charges = []
        encoded = path.predicted_displacement.copy()
        for sweep in path.sweeps:
            position, values, local, contribution = self.audit_sweep(
                sweep, position, values, alphas, displacement=encoded
            )
            encoded = sweep.blocks[-1].displacement_out.copy()
            row_charges.extend(local[3])
            for name in self.material.order:
                for k in range(self.material.count(name)):
                    total_equation[name][k] += local[1][name][k]
                    total_lambda[name][k] += local[2][name][k]
            for i in range(len(position)):
                for c in range(3):
                    load[i][c] += contribution[i][c]
                    total_budget[i][c] += local[0][i][c]
        cumulative = max(
            (v / (self.h * q(self.settings.reference_speed)) for row in total_budget for v in row),
            default=q(0),
        )
        if (
            max(
                [
                    cumulative,
                    *(v for rows in total_equation.values() for v in rows),
                    *(v for rows in total_lambda.values() for v in rows),
                ]
            )
            > self.tau
        ):
            raise SolverFailure("composite complete-path roundoff budget exhausted")
        self._same(encoded, path.final_displacement, "complete-path final displacement")
        _decode_world(self.reference, encoded, candidate.position)
        self._equal(position, candidate.position, "final configuration")
        for name in self.material.order:
            self._equal(values[name], diagnostics.get(name + "_multipliers"), "final lambda")
        for i in range(len(position)):
            for c in range(3):
                ideal_v = (
                    q(path.predicted_velocity[i, c])
                    + (q(position[i, c]) - q(path.predicted_position[i, c])) / self.h
                )
                error = max(
                    error,
                    self._error(candidate.velocity[i, c], ideal_v, self.settings.reference_speed),
                )
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
                    d = [x - y for x, y in zip(a, b, strict=True)]
                    constraint = sqrt_interval(dot(d, d), self.policy.sqrt_bits) - q(
                        law.rest_length[k]
                    )
                    scale = max(q(law.rest_length[k]), q(self.settings.reference_length))
                self._error(
                    0, constraint + q(law.compliance[k]) / self.h**2 * q(values[name][k]), scale
                )
        physical = self.domain(self.initial.position, position)
        reference_errors = {}
        for boundary in path.compute_boundaries:
            if boundary.name.startswith("reference/"):
                errors = [
                    q(word) - Fraction(w.numerator, w.denominator)
                    for word, w in zip(boundary.computed.flat, boundary.witnesses, strict=True)
                ]
                reference_errors[boundary.name] = tuple(
                    (e.numerator, e.denominator) for e in errors
                )
        decoder_errors = {}
        for name, world, d in (
            ("prediction", path.predicted_position, path.predicted_displacement),
            ("publication", candidate.position, path.final_displacement),
        ):
            errors = [
                q(x) - q(r) - q(offset)
                for x, r, offset in zip(world.flat, self.reference.flat, d.flat, strict=True)
            ]
            decoder_errors[name] = tuple((e.numerator, e.denominator) for e in errors)
        return dict(
            represented_row_charges=tuple(row_charges),
            reference_vector_conversion=reference_errors,
            decoder_residuals=decoder_errors,
            max_equation_error=float(error),
            cumulative_projection_error=float(cumulative),
            material_configuration_load=np.array([[float(v) for v in row] for row in load]).reshape(
                (-1, 3)
            ),
            numerical_domain_certificates=tuple(self._domains.values()),
            physical_affine_domain=physical,
            ordered_material_work=_represented_work(path, self.model, self.initial),
        )

    def audit_terminal(self, operations, before, after):
        if not any(operation.name == "terminal_apply" for operation in operations):
            raise SolverFailure("terminal native application receipt missing")
        current = np.asarray(before, np.float32).copy()
        response = None
        for operation in operations:
            if operation.name == "terminal_response":
                if len(operation.inputs) != 4 or len(operation.outputs) != 1:
                    raise SolverFailure("terminal response evidence missing")
                start, ids, loads, weights = operation.inputs
                if (
                    ids.dtype != np.int32
                    or ids.ndim != 1
                    or loads.shape != (len(ids), 3)
                    or np.any(ids < 0)
                    or np.any(ids >= len(current))
                ):
                    raise SolverFailure("invalid terminal identities")
                self._same(start, np.zeros_like(current), "terminal response origin")
                self._equal(weights, compute(self.model.inverse_mass), "terminal mass")
                result = np.zeros_like(current)
                for k, node in enumerate(ids):
                    for c in range(3):
                        product = _rounded32(q(weights[node]) * q(loads[k, c]))
                        result[node, c] = _rounded32(q(result[node, c]) + q(product))
                self._operation_binding(
                    operation,
                    "terminal_response",
                    (start, ids, loads, weights),
                    (result,),
                    int(len(ids) > 0),
                )
                response = result
            elif operation.name == "terminal_apply":
                if response is None or len(operation.inputs) != 3 or len(operation.outputs) != 1:
                    raise SolverFailure("terminal apply lacks response")
                x, delta, factor = operation.inputs
                self._same(x, current, "terminal selected pre velocity")
                self._same(delta, response, "terminal applied response")
                if factor.shape != () or float(factor) != 1:
                    raise SolverFailure("terminal scale changed")
                output = np.empty_like(current)
                for index in np.ndindex(current.shape):
                    output[index] = _rounded32(q(current[index]) + q(response[index]))
                self._operation_binding(
                    operation,
                    "terminal_apply",
                    (current, response, factor),
                    (output,),
                    int(current.size > 0),
                )
                current, response = output, None
            else:
                raise SolverFailure("unknown terminal operation")
        self._equal(current.astype(np.float64), after, "complete terminal velocity")
