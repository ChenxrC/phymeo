"""Affine projection/expansion around the original resident full-tree kernels.

No FK, mass, bias, geometry, energy or bounded-PD algorithm is duplicated here.
One lane owns a real N/K mechanism. Candidate writes are never publication.
"""

import tyx as tx

from ._tyx_numeric import _add, _div, _mul


def affine_project(
    count: tx.i32,
    full_meta: tx.ndarray(tx.i32, 2),
    reduced_meta: tx.ndarray(tx.i32, 2),
    matrix: tx.ndarray(tx.f32, 3),
    state: tx.ndarray(tx.f32, 3),
    full_mass: tx.ndarray(tx.f32, 3),
    full_vector: tx.ndarray(tx.f32, 3),
    full_linear: tx.ndarray(tx.f32, 4),
    full_angular: tx.ndarray(tx.f32, 4),
    mass: tx.ndarray(tx.f32, 3),
    vector: tx.ndarray(tx.f32, 3),
    linear: tx.ndarray(tx.f32, 4),
    angular: tx.ndarray(tx.f32, 4),
):
    for w in tx.range(count):
        n = full_meta[w, 1]
        k = reduced_meta[w, 1]
        for a in range(k):
            for column in tx.static_range(14):
                vector[w, a, column] = 0.0
            vector[w, a, 4] = state[w, a, 0]
            vector[w, a, 5] = state[w, a, 1]
            bias = 0.0
            external = 0.0
            for i in range(n):
                bias = _add(bias, _mul(matrix[w, i, a], full_vector[w, i, 0]))
                external = _add(external, _mul(matrix[w, i, a], full_vector[w, i, 1]))
            vector[w, a, 0] = bias
            vector[w, a, 1] = external
            for b in range(k):
                value = 0.0
                for i in range(n):
                    left = matrix[w, i, a]
                    if left != 0.0:
                        for j in range(n):
                            right = matrix[w, j, b]
                            if right != 0.0:
                                value = _add(value, _mul(_mul(left, full_mass[w, i, j]), right))
                mass[w, a, b] = value
            for body in range(n):
                for axis in tx.static_range(3):
                    jv = 0.0
                    jw = 0.0
                    for i in range(n):
                        jv = _add(jv, _mul(full_linear[w, body, axis, i], matrix[w, i, a]))
                        jw = _add(jw, _mul(full_angular[w, body, axis, i], matrix[w, i, a]))
                    linear[w, body, axis, a] = jv
                    angular[w, body, axis, a] = jw


def affine_expand(
    count: tx.i32,
    full_meta: tx.ndarray(tx.i32, 2),
    reduced_meta: tx.ndarray(tx.i32, 2),
    matrix: tx.ndarray(tx.f32, 3),
    offset: tx.ndarray(tx.f32, 2),
    independent: tx.ndarray(tx.i32, 2),
    row_indices: tx.ndarray(tx.i32, 3),
    row_values: tx.ndarray(tx.f32, 3),
    reverse_rows: tx.ndarray(tx.i32, 2),
    full_limits: tx.ndarray(tx.f32, 3),
    limits: tx.ndarray(tx.f32, 3),
    full_state: tx.ndarray(tx.f32, 3),
    state: tx.ndarray(tx.f32, 3),
    full_candidate: tx.ndarray(tx.f32, 3),
    candidate: tx.ndarray(tx.f32, 3),
    full_mass: tx.ndarray(tx.f32, 3),
    full_vector: tx.ndarray(tx.f32, 3),
    vector: tx.ndarray(tx.f32, 3),
    reaction: tx.ndarray(tx.f32, 3),
    row_receipt: tx.ndarray(tx.f32, 3),
    info: tx.ndarray(tx.i32, 2),
    metrics: tx.ndarray(tx.f32, 2),
):
    for w in tx.range(count):
        n = full_meta[w, 1]
        k = reduced_meta[w, 1]
        r = n - k
        code = info[w, 2]
        for j in tx.static_range(3):
            metrics[w, j] = 0.0
        for i in range(n):
            for j in tx.static_range(7):
                full_candidate[w, i, j] = full_state[w, i, j]
            for j in tx.static_range(3):
                reaction[w, i, j] = 0.0
        for row in range(r):
            row_receipt[w, row, 0] = 0.0
            row_receipt[w, row, 1] = 0.0
        if code == 0:
            for a in range(k):
                q = candidate[w, a, 0]
                v = candidate[w, a, 1]
                if q < limits[w, a, 0] or q > limits[w, a, 1] or tx.abs(v) > limits[w, a, 2]:
                    code = 5
            for i in range(n):
                q = offset[w, i]
                v = 0.0
                acceleration = 0.0
                motor = 0.0
                for a in range(k):
                    q = _add(q, _mul(matrix[w, i, a], candidate[w, a, 0]))
                    v = _add(v, _mul(matrix[w, i, a], candidate[w, a, 1]))
                    acceleration = _add(acceleration, _mul(matrix[w, i, a], vector[w, a, 3]))
                    if i == independent[w, a]:
                        motor = vector[w, a, 2]
                full_candidate[w, i, 0] = q
                full_candidate[w, i, 1] = v
                full_candidate[w, i, 2] = motor
                full_candidate[w, i, 3] = _add(motor, full_vector[w, i, 1])
                for j in tx.static_range(3):
                    full_candidate[w, i, 4 + j] = 0.0
                full_vector[w, i, 2] = motor
                full_vector[w, i, 3] = acceleration
                if (
                    q < full_limits[w, i, 0]
                    or q > full_limits[w, i, 1]
                    or tx.abs(q) > 4096.0
                    or tx.abs(v) > full_limits[w, i, 2]
                ):
                    code = 5
                for j in tx.static_range(7):
                    if not tx.isfinite(full_candidate[w, i, j]):
                        code = 2
                if not tx.isfinite(acceleration):
                    code = 2
            # Full M(P*a)+b-E*tau-e, not a reduced residual masquerading as full.
            full_scale = 1.0
            for i in range(n):
                inertia = 0.0
                for j in range(n):
                    inertia = _add(inertia, _mul(full_mass[w, i, j], full_vector[w, j, 3]))
                value = _add(
                    _add(inertia, full_vector[w, i, 0]),
                    -_add(full_vector[w, i, 2], full_vector[w, i, 1]),
                )
                reaction[w, i, 0] = value
                reaction[w, i, 1] = value
                full_scale = tx.max(full_scale, tx.abs(inertia))
                for j in tx.static_range(3):
                    full_scale = tx.max(full_scale, tx.abs(full_vector[w, i, j]))
                full_scale = tx.max(full_scale, tx.abs(value))
                if not tx.isfinite(value) or not tx.isfinite(inertia):
                    code = 2
            # Undo original rows in reverse topological order, preserving a
            # joint which both follows and drives another physical coordinate.
            for order in range(r):
                row = reverse_rows[w, order]
                driver = row_indices[w, row, 0]
                follower = row_indices[w, row, 1]
                a = row_values[w, row, 0]
                value = reaction[w, follower, 1]
                row_receipt[w, row, 0] = value
                relative = _add(full_vector[w, follower, 5], -_mul(a, full_vector[w, driver, 5]))
                row_receipt[w, row, 1] = _mul(value, relative)
                reaction[w, follower, 1] = _add(reaction[w, follower, 1], -value)
                reaction[w, driver, 1] = _add(reaction[w, driver, 1], _mul(a, value))
                reaction[w, follower, 2] = _add(reaction[w, follower, 2], value)
                reaction[w, driver, 2] = _add(reaction[w, driver, 2], -_mul(a, value))
                if not tx.isfinite(value) or not tx.isfinite(row_receipt[w, row, 1]):
                    code = 2
            full_error = 0.0
            power = 0.0
            power_scale = 1.0
            for i in range(n):
                error = _add(reaction[w, i, 0], -reaction[w, i, 2])
                full_error = tx.max(full_error, tx.abs(error))
                term = _mul(reaction[w, i, 0], full_vector[w, i, 5])
                power = _add(power, term)
                power_scale = _add(power_scale, tx.abs(term))
                if not tx.isfinite(error) or not tx.isfinite(term):
                    code = 2
            full_residual = _div(full_error, full_scale)
            metrics[w, 0] = full_residual
            metrics[w, 1] = power
            if (
                not tx.isfinite(full_scale)
                or not tx.isfinite(full_residual)
                or not tx.isfinite(power)
                or not tx.isfinite(power_scale)
            ):
                code = 2
            elif code == 0 and full_residual > 2e-5:
                code = 8
            elif code == 0 and tx.abs(power) > _mul(2e-5, power_scale):
                code = 7
            # Original row offsets, velocities and accelerations, both accepted
            # entry and candidate. No repair or projection of the actual state.
            row_error = 0.0
            for row in range(r):
                driver = row_indices[w, row, 0]
                follower = row_indices[w, row, 1]
                a = row_values[w, row, 0]
                for channel in tx.static_range(5):
                    left = full_candidate[w, follower, 0]
                    right = _mul(a, full_candidate[w, driver, 0])
                    offset_value = row_values[w, row, 1]
                    if channel == 1:
                        left = full_candidate[w, follower, 1]
                        right = _mul(a, full_candidate[w, driver, 1])
                        offset_value = 0.0
                    elif channel == 2:
                        left = full_vector[w, follower, 3]
                        right = _mul(a, full_vector[w, driver, 3])
                        offset_value = 0.0
                    elif channel == 3:
                        left = full_state[w, follower, 0]
                        right = _mul(a, full_state[w, driver, 0])
                    elif channel == 4:
                        left = full_state[w, follower, 1]
                        right = _mul(a, full_state[w, driver, 1])
                        offset_value = 0.0
                    error = _add(_add(left, -right), -offset_value)
                    scale = tx.max(
                        1.0, tx.max(tx.abs(left), tx.max(tx.abs(right), tx.abs(offset_value)))
                    )
                    residual = _div(tx.abs(error), scale)
                    row_error = tx.max(row_error, residual)
                    if not tx.isfinite(residual) or not tx.isfinite(scale):
                        code = 2
                    elif code == 0 and residual > 2e-5:
                        code = 7
            metrics[w, 2] = row_error
        info[w, 2] = code
        if code != 0:
            for i in range(n):
                for j in tx.static_range(7):
                    full_candidate[w, i, j] = full_state[w, i, j]


def affine_retain(
    count: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    state: tx.ndarray(tx.f32, 3),
    candidate: tx.ndarray(tx.f32, 3),
    accepted: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        slot = meta[w, 0]
        if accepted[slot] != 1:
            for i in range(meta[w, 1]):
                for j in tx.static_range(7):
                    candidate[w, i, j] = state[w, i, j]


def affine_validate_state(
    count: tx.i32,
    full_meta: tx.ndarray(tx.i32, 2),
    reduced_meta: tx.ndarray(tx.i32, 2),
    matrix: tx.ndarray(tx.f32, 3),
    offset: tx.ndarray(tx.f32, 2),
    independent: tx.ndarray(tx.i32, 2),
    full_limits: tx.ndarray(tx.f32, 3),
    limits: tx.ndarray(tx.f32, 3),
    full: tx.ndarray(tx.f32, 3),
    reduced: tx.ndarray(tx.f32, 3),
    flags: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        n = full_meta[w, 1]
        k = reduced_meta[w, 1]
        good = 1
        for a in range(k):
            q = reduced[w, a, 0]
            v = reduced[w, a, 1]
            if (
                tx.abs(q) > 4096.0
                or q < limits[w, a, 0]
                or q > limits[w, a, 1]
                or tx.abs(v) > limits[w, a, 2]
            ):
                good = 0
            for j in tx.static_range(7):
                if not tx.isfinite(reduced[w, a, j]):
                    good = 0
            projected_load = 0.0
            for i in range(n):
                projected_load = _add(projected_load, _mul(matrix[w, i, a], full[w, i, 3]))
            scale = tx.max(1.0, tx.max(tx.abs(projected_load), tx.abs(reduced[w, a, 3])))
            if not tx.isfinite(projected_load) or tx.abs(
                _add(projected_load, -reduced[w, a, 3])
            ) > _mul(2e-5, scale):
                good = 0
        for i in range(n):
            expected_q = offset[w, i]
            expected_v = 0.0
            expected_motor = 0.0
            for a in range(k):
                expected_q = _add(expected_q, _mul(matrix[w, i, a], reduced[w, a, 0]))
                expected_v = _add(expected_v, _mul(matrix[w, i, a], reduced[w, a, 1]))
                if i == independent[w, a]:
                    expected_motor = reduced[w, a, 2]
            q = full[w, i, 0]
            v = full[w, i, 1]
            if (
                tx.abs(q) > 4096.0
                or q < full_limits[w, i, 0]
                or q > full_limits[w, i, 1]
                or tx.abs(v) > full_limits[w, i, 2]
            ):
                good = 0
            for j in tx.static_range(7):
                if not tx.isfinite(full[w, i, j]):
                    good = 0
            for j in tx.static_range(3):
                expected = expected_q
                if j == 1:
                    expected = expected_v
                elif j == 2:
                    expected = expected_motor
                actual = full[w, i, j]
                scale = tx.max(1.0, tx.max(tx.abs(actual), tx.abs(expected)))
                if not tx.isfinite(expected) or tx.abs(_add(actual, -expected)) > _mul(2e-5, scale):
                    good = 0
        flags[w] = good


def make_affine_kernels():
    """Separate handles for both parent active-bank directions; normal math."""
    functions = {
        "project": affine_project,
        "expand": affine_expand,
        "retain": affine_retain,
        "validate_state": affine_validate_state,
    }
    return {
        name: (tx.kernel(function), tx.kernel(function)) for name, function in functions.items()
    }
