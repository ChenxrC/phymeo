"""Audit saved ordered-distance FP32 transitions without running a solver.

The input is one accepted D1-style step: immutable entry arrays, the stored
binary32 step size, predictor, complete ordered trace for exactly sweeps,
and accepted endpoint. This module imports no Phymeo/TYX code. It checks the
recorded primitive arithmetic, not the distance constitutive law that produced
each recorded direction and delta; the independent material oracle remains
necessary. Missing rows or sweeps fail the declared shape/count binding; reordered
nonidentical transitions fail the state/lambda chains.

For each free node (inverse mass w > 0), the exact rational identity is

    (v_final - v_entry)/w = h*force + h*gravity/w + J + Ep + Ex + Ev,

where J is the original sum of signed recorded n*delta/h, Ep is predictor
rounding, Ex is ordered position-update rounding, and Ev is velocity
reconstruction rounding. Fixed reactions remain -h*force-J. None of these
definitions replaces J with a measured displacement or claims exact physical
conservation. The raw physical imbalance is retained separately.

Every bound is formed from independently expected rounded values BEFORE
comparing the corresponding saved output. U(z) is a conservative half-ULP of
the expected binary32 result z. Under the checked normal-or-zero domain:

    Bp = [h*(U(wF)+U(g+RN(wF))) + U(h*a) + U(v_entry+RN(h*a))]/w
    Bx = sum [abs(n)*U(w*delta) + U(RN(w*delta)*n) + U(x_next)]/(w*h)
    Bv = [U(x_final-x_pred)/h + U(RN(x_final-x_pred)/h) + U(v_final)]/w

RN is applied at each displayed primitive. Signed subtraction uses addition
of the negated binary32 operand. All error/bound comparisons and the closing
identity use exact rational representations of stored binary32 values. Returned
arrays are owned float64 diagnostic representations, not new physical state.
No bound is fitted to the observed residual. Unsupported nonzero subnormal
inputs/products/quotients/addition results and nonfinite values are rejected.
"""

import math
from fractions import Fraction

import numpy as np

_SCHEMA = "deformable-fp32-impulse-audit-v1"
_METHOD = "a-priori-binary32-half-ulp/exact-rational-v1"
_TINY = float.fromhex("0x1.0p-126")
_ZERO = Fraction(0)


def _array(name, value, shape):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} must be a stored float32 ndarray")
    if value.shape != shape:
        raise ValueError(f"{name} has shape {value.shape}; expected {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains a nonfinite value")
    magnitude = np.abs(value)
    if np.any((magnitude != 0) & (magnitude < _TINY)):
        raise ValueError(f"{name} contains unsupported subnormal data")
    return value


def _same(name, actual, expected):
    actual_array = np.asarray(actual, dtype=np.float32)
    expected_array = np.asarray(expected, dtype=np.float32)
    if actual_array.shape != expected_array.shape:
        raise ValueError(f"{name}: stored and expected shapes differ")
    if actual_array.tobytes(order="C") != expected_array.tobytes(order="C"):
        raise ValueError(f"{name}: saved value does not match the FP32 transition")


def _rn(operation, left, right):
    """Separate scalar binary32 primitives, including the original domain guards."""
    a = np.float32(left)
    b = np.float32(right)
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        if operation == "add":
            result = np.float32(a + b)
        elif operation == "mul":
            result = np.float32(a * b)
        elif operation == "div":
            if b == 0:
                raise ArithmeticError("division by zero")
            result = np.float32(a / b)
        else:
            raise ValueError("unknown primitive")
    magnitude = abs(float(result))
    if not math.isfinite(float(result)):
        raise ArithmeticError(f"nonfinite FP32 {operation}")
    if magnitude != 0 and magnitude < _TINY:
        raise ArithmeticError(f"subnormal FP32 {operation}")
    if operation == "mul" and a != 0 and b != 0 and magnitude < _TINY:
        raise ArithmeticError("unsupported FP32 multiplication underflow")
    if operation == "div" and a != 0 and magnitude < _TINY:
        raise ArithmeticError("unsupported FP32 division underflow")
    return result


def _half_ulp(value):
    """Upper half-spacing at the independently expected normal FP32 result."""
    magnitude = abs(float(value))
    if magnitude == 0:
        # Every permitted operation producing zero is exact: add cancels two
        # binary32 operands; a nonzero multiply/divide underflow is rejected.
        return _ZERO
    _, exponent = math.frexp(magnitude)
    power = exponent - 25
    if power >= 0:
        return Fraction(1 << power)
    return Fraction(1, 1 << -power)


def _matrix(count):
    return [[_ZERO for _ in range(3)] for _ in range(count)]


def _output(values, *, upper=False):
    result = np.array([[float(x) for x in row] for row in values], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ArithmeticError("diagnostic conversion exceeds finite float64")
    if upper:
        # Preserve a conservative exported bound if exact rational -> float64
        # rounds downward. This is representation, not a fitted physical gate.
        for node, row in enumerate(values):
            for axis, exact_value in enumerate(row):
                if Fraction.from_float(float(result[node, axis])) < exact_value:
                    result[node, axis] = np.nextafter(result[node, axis], np.inf)
        if not np.isfinite(result).all():
            raise ArithmeticError("outward bound conversion exceeds finite float64")
    return result


def _bounded(name, error, bound):
    if bound < 0 or abs(error) > bound:
        raise ArithmeticError(f"{name}: exact rounding error exceeds the a-priori bound")


def audit_deformable_impulse(
    *,
    input_position,
    input_velocity,
    force,
    inverse_mass,
    gravity,
    h,
    predicted_position,
    predicted_velocity,
    trace,
    edges,
    final_position,
    final_velocity,
    sweeps,
):
    """Verify one saved step and return per-node impulses, remainders and bounds.

    sweeps is required and must equal the actual recorded sweep count.
    Each trace frame has shape (len(edges), 18), with columns pre_i, pre_j,
    post_i, post_j, lambda_before, lambda_after, direction[3], delta.
    Node/edge order is the authored model order; no rows are sorted or dropped.
    This function does not modify any input and performs no project import.
    """
    if not isinstance(input_position, np.ndarray) or input_position.ndim != 2:
        raise TypeError("input_position must be a stored (N, 3) ndarray")
    count = input_position.shape[0]
    if count < 1 or input_position.shape[1] != 3:
        raise ValueError("input_position must have nonempty shape (N, 3)")
    shape = (count, 3)
    x0 = _array("input_position", input_position, shape)
    v0 = _array("input_velocity", input_velocity, shape)
    loads = _array("force", force, shape)
    weights = _array("inverse_mass", inverse_mass, (count,))
    g = _array("gravity", gravity, (3,))
    saved_px = _array("predicted_position", predicted_position, shape)
    saved_pv = _array("predicted_velocity", predicted_velocity, shape)
    saved_fx = _array("final_position", final_position, shape)
    saved_fv = _array("final_velocity", final_velocity, shape)
    if np.any(weights < 0):
        raise ValueError("inverse_mass must be nonnegative")
    if isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, float, np.floating)):
        raise TypeError("h must be the stored scalar step size")
    if not math.isfinite(float(h)) or float(h) <= 0:
        raise ValueError("h must be finite and positive")
    with np.errstate(over="ignore", under="ignore"):
        step = np.float32(h)
    if not math.isfinite(float(step)) or float(step) != float(h) or float(step) < _TINY:
        raise ValueError("h must be exactly representable normal float32")
    if type(sweeps) is not int or sweeps < 0:
        raise TypeError("sweeps must be a nonnegative builtin int")
    if not isinstance(trace, (tuple, list)) or len(trace) != sweeps:
        raise ValueError("trace must contain exactly the recorded number of sweeps")
    if (
        not isinstance(edges, np.ndarray)
        or edges.dtype.kind not in "iu"
        or edges.ndim != 2
        or edges.shape[1] != 2
    ):
        raise TypeError("edges must be an integer ndarray with shape (E, 2)")
    edge_count = edges.shape[0]
    if np.any(edges < 0) or np.any(edges >= count):
        raise ValueError("edge endpoint is outside the node layout")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("self edges are not supported")
    frames = tuple(
        _array(f"trace[{index}]", frame, (edge_count, 18)) for index, frame in enumerate(trace)
    )
    # Local conversion cache retains no owner beyond this audit call.
    rationals = {}

    def exact(value):
        key = float(value)
        if key not in rationals:
            rationals[key] = Fraction.from_float(key)
        return rationals[key]

    hq = exact(step)
    command = _matrix(count)
    predictor_error = _matrix(count)
    row_error = _matrix(count)
    reconstruction_error = _matrix(count)
    predictor_bound = _matrix(count)
    row_bound = _matrix(count)
    reconstruction_bound = _matrix(count)
    expected_px = np.empty(shape, dtype=np.float32)
    expected_pv = np.empty(shape, dtype=np.float32)

    for node in range(count):
        w = weights[node]
        for axis in range(3):
            if w > 0:
                weighted_force = _rn("mul", w, loads[node, axis])
                acceleration = _rn("add", g[axis], weighted_force)
                dv = _rn("mul", step, acceleration)
                velocity = _rn("add", v0[node, axis], dv)
                dx = _rn("mul", step, velocity)
                position = _rn("add", x0[node, axis], dx)
                bound = (
                    hq * (_half_ulp(weighted_force) + _half_ulp(acceleration))
                    + _half_ulp(dv)
                    + _half_ulp(velocity)
                ) / exact(w)
                error = (
                    (exact(velocity) - exact(v0[node, axis])) / exact(w)
                    - hq * exact(loads[node, axis])
                    - hq * exact(g[axis]) / exact(w)
                )
                _bounded(f"predictor[{node},{axis}]", error, bound)
                predictor_error[node][axis] = error
                predictor_bound[node][axis] = bound
            else:
                # A saved fixed input velocity is required to represent a fixed
                # node rather than an unaccounted instantaneous momentum reset.
                if v0[node, axis] != 0:
                    raise ValueError("fixed node has nonzero input velocity")
                velocity = np.float32(0)
                position = x0[node, axis]
            expected_pv[node, axis] = velocity
            expected_px[node, axis] = position
    _same("predicted_position", saved_px, expected_px)
    _same("predicted_velocity", saved_pv, expected_pv)

    current = expected_px.copy()
    multipliers = np.zeros(edge_count, dtype=np.float32)
    for sweep_index, frame in enumerate(frames):
        for edge_index, (left_index, right_index) in enumerate(edges):
            i = int(left_index)
            j = int(right_index)
            row = frame[edge_index]
            prefix = f"trace[{sweep_index}][{edge_index}]"
            _same(prefix + ".pre_i", row[:3], current[i])
            _same(prefix + ".pre_j", row[3:6], current[j])
            _same(prefix + ".lambda_before", row[12], multipliers[edge_index])
            delta = row[17]
            direction = row[14:17]
            skipped = bool(np.all(direction == 0))
            if skipped:
                if delta != 0:
                    raise ValueError(prefix + ": skipped row has nonzero delta")
                next_multiplier = multipliers[edge_index]
            else:
                next_multiplier = _rn("add", multipliers[edge_index], delta)
            expected_posts = []
            for node, sign in ((i, 1), (j, -1)):
                previous = current[node].copy()
                next_position = previous.copy()
                for axis in range(3):
                    exact_command = sign * exact(direction[axis]) * exact(delta) / hq
                    command[node][axis] += exact_command
                    if weights[node] > 0 and not skipped:
                        wd = _rn("mul", weights[node], delta)
                        movement = _rn("mul", wd, direction[axis])
                        signed_movement = movement if sign == 1 else np.float32(-movement)
                        next_position[axis] = _rn("add", previous[axis], signed_movement)
                        bound = (
                            abs(exact(direction[axis])) * _half_ulp(wd)
                            + _half_ulp(movement)
                            + _half_ulp(next_position[axis])
                        ) / (exact(weights[node]) * hq)
                        error = (exact(next_position[axis]) - exact(previous[axis])) / (
                            exact(weights[node]) * hq
                        ) - exact_command
                        _bounded(prefix + f".position[{node},{axis}]", error, bound)
                        row_error[node][axis] += error
                        row_bound[node][axis] += bound
                expected_posts.append(next_position)
            # Both independent endpoint computations and all a-priori bounds
            # exist before either saved endpoint is compared or accepted.
            _same(prefix + ".post_i", row[6:9], expected_posts[0])
            _same(prefix + ".post_j", row[9:12], expected_posts[1])
            _same(prefix + ".lambda_after", row[13], next_multiplier)
            current[i] = expected_posts[0]
            current[j] = expected_posts[1]
            multipliers[edge_index] = next_multiplier
    _same("final_position (including every unvisited node)", saved_fx, current)

    expected_fv = np.zeros(shape, dtype=np.float32)
    for node in range(count):
        if weights[node] > 0:
            for axis in range(3):
                displacement = _rn("add", current[node, axis], np.float32(-expected_px[node, axis]))
                correction = _rn("div", displacement, step)
                velocity = _rn("add", expected_pv[node, axis], correction)
                bound = (
                    _half_ulp(displacement) / hq + _half_ulp(correction) + _half_ulp(velocity)
                ) / exact(weights[node])
                error = (exact(velocity) - exact(expected_pv[node, axis])) / exact(
                    weights[node]
                ) - (exact(current[node, axis]) - exact(expected_px[node, axis])) / (
                    exact(weights[node]) * hq
                )
                _bounded(f"reconstruction[{node},{axis}]", error, bound)
                reconstruction_error[node][axis] = error
                reconstruction_bound[node][axis] = bound
                expected_fv[node, axis] = velocity
    _same("final_velocity", saved_fv, expected_fv)

    actual = _matrix(count)
    external = _matrix(count)
    gravity_impulse = _matrix(count)
    free_material = _matrix(count)
    raw = _matrix(count)
    remainder = _matrix(count)
    total_bound = _matrix(count)
    closure = _matrix(count)
    fixed_reaction = _matrix(count)
    for node in range(count):
        for axis in range(3):
            if weights[node] > 0:
                actual[node][axis] = (exact(saved_fv[node, axis]) - exact(v0[node, axis])) / exact(
                    weights[node]
                )
                external[node][axis] = hq * exact(loads[node, axis])
                gravity_impulse[node][axis] = hq * exact(g[axis]) / exact(weights[node])
                free_material[node][axis] = command[node][axis]
                raw[node][axis] = (
                    actual[node][axis]
                    - external[node][axis]
                    - gravity_impulse[node][axis]
                    - command[node][axis]
                )
                remainder[node][axis] = (
                    predictor_error[node][axis]
                    + row_error[node][axis]
                    + reconstruction_error[node][axis]
                )
                total_bound[node][axis] = (
                    predictor_bound[node][axis]
                    + row_bound[node][axis]
                    + reconstruction_bound[node][axis]
                )
                _bounded(
                    f"aggregate[{node},{axis}]", remainder[node][axis], total_bound[node][axis]
                )
                closure[node][axis] = raw[node][axis] - remainder[node][axis]
                if closure[node][axis] != 0:
                    raise ArithmeticError("complete saved chain does not close exactly")
            else:
                fixed_reaction[node][axis] = -hq * exact(loads[node, axis]) - command[node][axis]

    return {
        "schema": _SCHEMA,
        "bound_method": _METHOD,
        "physical_conservation_claim": False,
        "constitutive_law_recomputed": False,
        "node_count": count,
        "edge_count": edge_count,
        "sweeps": sweeps,
        "h": float(step),
        "actual_impulse": _output(actual),
        "external_impulse": _output(external),
        "gravity_impulse": _output(gravity_impulse),
        "material_impulse": _output(free_material),
        "commanded_material_impulse": _output(command),
        "fixed_reaction_impulse": _output(fixed_reaction),
        "raw_imbalance": _output(raw),
        "raw_impulse_imbalance": _output(raw),
        "predictor_rounding_impulse": _output(predictor_error),
        "position_rounding_impulse": _output(row_error),
        "reconstruction_rounding_impulse": _output(reconstruction_error),
        "predictor_bound": _output(predictor_bound, upper=True),
        "position_bound": _output(row_bound, upper=True),
        "reconstruction_bound": _output(reconstruction_bound, upper=True),
        "total_remainder": _output(remainder),
        "remainder_bound": _output(total_bound, upper=True),
        "closure_residual": _output(closure),
        "bounded_remainder": _output(closure),
    }
