"""Constant floating affine coordinates around the sole physical floating solve.

No mechanics, bounded PD, quaternion integration, FK, energy or publication is
implemented here. One lane owns the full/reduced maps of one real world.
"""

import tyx as tx

from ._tyx_floating_numeric import _add, _div, _mul


def floating_affine_project(
    count: tx.i32,
    full_meta: tx.ndarray(tx.i32, 2),
    reduced_meta: tx.ndarray(tx.i32, 2),
    projection: tx.ndarray(tx.f32, 3),
    state: tx.ndarray(tx.f32, 3),
    full_mass: tx.ndarray(tx.f32, 3),
    full_vector: tx.ndarray(tx.f32, 3),
    full_linear: tx.ndarray(tx.f32, 4),
    full_angular: tx.ndarray(tx.f32, 4),
    mass: tx.ndarray(tx.f32, 3),
    vector: tx.ndarray(tx.f32, 3),
    linear: tx.ndarray(tx.f32, 4),
    angular: tx.ndarray(tx.f32, 4),
    flags: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        n = full_meta[w, 1]
        k = reduced_meta[w, 1]
        df = n + 6
        dr = k + 6
        code = 0
        # Validate even components eliminated by the constant map. A reduced
        # finite result cannot excuse nonfinite physical mechanics.
        for i in range(df):
            for channel in tx.static_range(2):
                if not tx.isfinite(full_vector[w, i, channel]):
                    code = 2
            if not tx.isfinite(full_vector[w, i, 5]):
                code = 2
            for j in range(df):
                if not tx.isfinite(full_mass[w, i, j]):
                    code = 2
            for body in range(n + 1):
                for axis in tx.static_range(3):
                    if not tx.isfinite(full_linear[w, body, axis, i]) or not tx.isfinite(full_angular[w, body, axis, i]):
                        code = 2
        for a in range(dr):
            for column in tx.static_range(14):
                vector[w, a, column] = 0.0
            if a < 6:
                vector[w, a, 4] = full_vector[w, a, 4]
                vector[w, a, 5] = full_vector[w, a, 5]
            else:
                vector[w, a, 4] = state[w, a - 6, 0]
                vector[w, a, 5] = state[w, a - 6, 1]
            bias = 0.0
            external = 0.0
            for i in range(df):
                bias = _add(bias, _mul(projection[w, i, a], full_vector[w, i, 0]))
                external = _add(external, _mul(projection[w, i, a], full_vector[w, i, 1]))
            vector[w, a, 0] = bias
            vector[w, a, 1] = external
            if not tx.isfinite(bias) or not tx.isfinite(external):
                code = 2
            for b in range(dr):
                value = 0.0
                for i in range(df):
                    left = projection[w, i, a]
                    if left != 0.0:
                        for j in range(df):
                            right = projection[w, j, b]
                            if right != 0.0:
                                value = _add(value, _mul(_mul(left, full_mass[w, i, j]), right))
                mass[w, a, b] = value
                if not tx.isfinite(value):
                    code = 2
            for body in range(n + 1):
                for axis in tx.static_range(3):
                    jv = 0.0
                    jw = 0.0
                    for i in range(df):
                        jv = _add(jv, _mul(full_linear[w, body, axis, i], projection[w, i, a]))
                        jw = _add(jw, _mul(full_angular[w, body, axis, i], projection[w, i, a]))
                    linear[w, body, axis, a] = jv
                    angular[w, body, axis, a] = jw
                    if not tx.isfinite(jv) or not tx.isfinite(jw):
                        code = 2
        # The shared solve initializes info. Keep this independent failure mask
        # until expansion joins it to the same world's physical status.
        flags[w] = code


def floating_affine_expand(
    count: tx.i32,
    full_meta: tx.ndarray(tx.i32, 2),
    reduced_meta: tx.ndarray(tx.i32, 2),
    matrix: tx.ndarray(tx.f32, 3),
    projection: tx.ndarray(tx.f32, 3),
    offset: tx.ndarray(tx.f32, 2),
    independent: tx.ndarray(tx.i32, 2),
    row_indices: tx.ndarray(tx.i32, 3),
    row_values: tx.ndarray(tx.f32, 3),
    reverse_rows: tx.ndarray(tx.i32, 2),
    full_limits: tx.ndarray(tx.f32, 3),
    limits: tx.ndarray(tx.f32, 3),
    full_state: tx.ndarray(tx.f32, 3),
    full_candidate: tx.ndarray(tx.f32, 3),
    candidate: tx.ndarray(tx.f32, 3),
    full_applied: tx.ndarray(tx.f32, 2),
    next_full_applied: tx.ndarray(tx.f32, 2),
    full_mass: tx.ndarray(tx.f32, 3),
    full_vector: tx.ndarray(tx.f32, 3),
    vector: tx.ndarray(tx.f32, 3),
    reaction: tx.ndarray(tx.f32, 3),
    row_receipt: tx.ndarray(tx.f32, 3),
    info: tx.ndarray(tx.i32, 2),
    metrics: tx.ndarray(tx.f32, 2),
    projection_flags: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        n = full_meta[w, 1]
        k = reduced_meta[w, 1]
        df = n + 6
        dr = k + 6
        r = n - k
        code = info[w, 2]
        if code == 0 and projection_flags[w] != 0:
            code = projection_flags[w]
        for j in tx.static_range(4):
            metrics[w, j] = 0.0
        for i in range(n):
            for j in tx.static_range(7):
                full_candidate[w, i, j] = full_state[w, i, j]
        for i in range(df):
            next_full_applied[w, i] = full_applied[w, i]
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
                for channel in tx.static_range(7):
                    if not tx.isfinite(candidate[w, a, channel]):
                        code = 2
            # Root has no motor. Its once-integrated pose/velocity already lives
            # in the real physical destination root bank, untouched here.
            for i in tx.static_range(6):
                full_vector[w, i, 2] = 0.0
                full_vector[w, i, 3] = vector[w, i, 3]
                next_full_applied[w, i] = full_vector[w, i, 1]
            for i in range(n):
                q = offset[w, i]
                v = 0.0
                acceleration = 0.0
                motor = 0.0
                for a in range(k):
                    q = _add(q, _mul(matrix[w, i, a], candidate[w, a, 0]))
                    v = _add(v, _mul(matrix[w, i, a], candidate[w, a, 1]))
                    acceleration = _add(acceleration, _mul(matrix[w, i, a], vector[w, a + 6, 3]))
                    if i == independent[w, a]:
                        motor = vector[w, a + 6, 2]
                applied = _add(motor, full_vector[w, i + 6, 1])
                full_candidate[w, i, 0] = q
                full_candidate[w, i, 1] = v
                full_candidate[w, i, 2] = motor
                full_candidate[w, i, 3] = applied
                for channel in tx.static_range(3):
                    full_candidate[w, i, 4 + channel] = 0.0
                full_vector[w, i + 6, 2] = motor
                full_vector[w, i + 6, 3] = acceleration
                next_full_applied[w, i + 6] = applied
                if q < full_limits[w, i, 0] or q > full_limits[w, i, 1] or tx.abs(q) > 4096.0 or tx.abs(v) > full_limits[w, i, 2]:
                    code = 5
                for channel in tx.static_range(7):
                    if not tx.isfinite(full_candidate[w, i, channel]):
                        code = 2
                if not tx.isfinite(acceleration):
                    code = 2
            # Full M*S*a+b-e-E*tau, including all six real root equations.
            full_scale = 1.0
            for i in range(df):
                inertia = 0.0
                for j in range(df):
                    inertia = _add(inertia, _mul(full_mass[w, i, j], full_vector[w, j, 3]))
                value = _add(_add(inertia, full_vector[w, i, 0]), -_add(full_vector[w, i, 2], full_vector[w, i, 1]))
                reaction[w, i, 0] = value
                reaction[w, i, 1] = value
                full_scale = tx.max(full_scale, tx.abs(inertia))
                for channel in tx.static_range(3):
                    full_scale = tx.max(full_scale, tx.abs(full_vector[w, i, channel]))
                full_scale = tx.max(full_scale, tx.abs(value))
                if not tx.isfinite(value) or not tx.isfinite(inertia) or not tx.isfinite(next_full_applied[w, i]):
                    code = 2
            row_power_sum = 0.0
            for order in range(r):
                row = reverse_rows[w, order]
                driver = row_indices[w, row, 0] + 6
                follower = row_indices[w, row, 1] + 6
                ratio = row_values[w, row, 0]
                value = reaction[w, follower, 1]
                row_receipt[w, row, 0] = value
                relative = _add(full_vector[w, follower, 5], -_mul(ratio, full_vector[w, driver, 5]))
                power = _mul(value, relative)
                row_receipt[w, row, 1] = power
                row_power_sum = _add(row_power_sum, power)
                row_scale = tx.max(1.0, tx.max(tx.abs(_mul(value, full_vector[w, follower, 5])), tx.abs(_mul(_mul(ratio, value), full_vector[w, driver, 5]))))
                reaction[w, follower, 1] = _add(reaction[w, follower, 1], -value)
                reaction[w, driver, 1] = _add(reaction[w, driver, 1], _mul(ratio, value))
                reaction[w, follower, 2] = _add(reaction[w, follower, 2], value)
                reaction[w, driver, 2] = _add(reaction[w, driver, 2], -_mul(ratio, value))
                if not tx.isfinite(value) or not tx.isfinite(power) or not tx.isfinite(row_scale):
                    code = 2
                elif code == 0 and tx.abs(power) > _mul(2e-5, row_scale):
                    code = 7
            full_error = 0.0
            power = 0.0
            power_scale = 1.0
            for i in range(df):
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
            if not tx.isfinite(full_scale) or not tx.isfinite(full_residual) or not tx.isfinite(power) or not tx.isfinite(power_scale) or not tx.isfinite(row_power_sum):
                code = 2
            elif code == 0 and full_residual > 2e-5:
                code = 8
            elif code == 0 and (tx.abs(power) > _mul(2e-5, power_scale) or tx.abs(_add(power, -row_power_sum)) > _mul(2e-5, power_scale)):
                code = 7
            projected_error = 0.0
            for a in range(dr):
                value = 0.0
                column_scale = 0.0
                for i in range(df):
                    value = _add(value, _mul(projection[w, i, a], reaction[w, i, 0]))
                    column_scale = _add(column_scale, tx.abs(projection[w, i, a]))
                scale = _mul(full_scale, tx.max(1.0, column_scale))
                residual = _div(tx.abs(value), scale)
                projected_error = tx.max(projected_error, residual)
                if not tx.isfinite(value) or not tx.isfinite(scale) or not tx.isfinite(residual):
                    code = 2
                elif code == 0 and residual > 2e-5:
                    code = 7
            metrics[w, 3] = projected_error
            row_error = 0.0
            for row in range(r):
                driver = row_indices[w, row, 0]
                follower = row_indices[w, row, 1]
                ratio = row_values[w, row, 0]
                for channel in tx.static_range(5):
                    left = full_candidate[w, follower, 0]
                    right = _mul(ratio, full_candidate[w, driver, 0])
                    intercept = row_values[w, row, 1]
                    if channel == 1:
                        left = full_candidate[w, follower, 1]
                        right = _mul(ratio, full_candidate[w, driver, 1])
                        intercept = 0.0
                    elif channel == 2:
                        left = full_vector[w, follower + 6, 3]
                        right = _mul(ratio, full_vector[w, driver + 6, 3])
                        intercept = 0.0
                    elif channel == 3:
                        left = full_state[w, follower, 0]
                        right = _mul(ratio, full_state[w, driver, 0])
                    elif channel == 4:
                        left = full_state[w, follower, 1]
                        right = _mul(ratio, full_state[w, driver, 1])
                        intercept = 0.0
                    error = _add(_add(left, -right), -intercept)
                    scale = tx.max(1.0, tx.max(tx.abs(left), tx.max(tx.abs(right), tx.abs(intercept))))
                    residual = _div(tx.abs(error), scale)
                    row_error = tx.max(row_error, residual)
                    if not tx.isfinite(residual) or not tx.isfinite(scale):
                        code = 2
                    elif code == 0 and residual > 2e-5:
                        code = 7
            metrics[w, 2] = row_error
        info[w, 2] = code
        # Full finalize owns root/FK/twist/energy/applied retention. Keeping the
        # full joint candidate here avoids needless invalid geometry evaluation.
        if code != 0:
            for i in range(n):
                for channel in tx.static_range(7):
                    full_candidate[w, i, channel] = full_state[w, i, channel]
            for i in range(df):
                next_full_applied[w, i] = full_applied[w, i]


def floating_affine_retain(
    count: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    state: tx.ndarray(tx.f32, 3),
    candidate: tx.ndarray(tx.f32, 3),
    applied: tx.ndarray(tx.f32, 2),
    next_applied: tx.ndarray(tx.f32, 2),
    accepted: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        k = meta[w, 1]
        if accepted[meta[w, 0]] != 1:
            for i in range(k):
                for channel in tx.static_range(7):
                    candidate[w, i, channel] = state[w, i, channel]
            for i in range(k + 6):
                next_applied[w, i] = applied[w, i]


def floating_affine_restore(
    count: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    mask: tx.ndarray(tx.i32, 1),
    restore_state: tx.ndarray(tx.f32, 3),
    restore_applied: tx.ndarray(tx.f32, 2),
    state: tx.ndarray(tx.f32, 3),
    candidate: tx.ndarray(tx.f32, 3),
    applied: tx.ndarray(tx.f32, 2),
    next_applied: tx.ndarray(tx.f32, 2),
):
    for w in tx.range(count):
        k = meta[w, 1]
        for i in range(k):
            for channel in tx.static_range(7):
                value = state[w, i, channel]
                if mask[w] == 1:
                    value = restore_state[w, i, channel]
                candidate[w, i, channel] = value
        for i in range(k + 6):
            value = applied[w, i]
            if mask[w] == 1:
                value = restore_applied[w, i]
            next_applied[w, i] = value


def floating_affine_validate_state(
    count: tx.i32,
    full_meta: tx.ndarray(tx.i32, 2),
    meta: tx.ndarray(tx.i32, 2),
    matrix: tx.ndarray(tx.f32, 3),
    projection: tx.ndarray(tx.f32, 3),
    offset: tx.ndarray(tx.f32, 2),
    independent: tx.ndarray(tx.i32, 2),
    row_indices: tx.ndarray(tx.i32, 3),
    row_values: tx.ndarray(tx.f32, 3),
    full_limits: tx.ndarray(tx.f32, 3),
    limits: tx.ndarray(tx.f32, 3),
    full: tx.ndarray(tx.f32, 3),
    state: tx.ndarray(tx.f32, 3),
    full_applied: tx.ndarray(tx.f32, 2),
    applied: tx.ndarray(tx.f32, 2),
    flags: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        n = full_meta[w, 1]
        k = meta[w, 1]
        good = 1
        for a in range(k):
            q = state[w, a, 0]
            v = state[w, a, 1]
            if tx.abs(q) > 4096.0 or q < limits[w, a, 0] or q > limits[w, a, 1] or tx.abs(v) > limits[w, a, 2]:
                good = 0
            for channel in tx.static_range(7):
                if not tx.isfinite(state[w, a, channel]):
                    good = 0
            expected = applied[w, a + 6]
            scale = tx.max(1.0, tx.max(tx.abs(expected), tx.abs(state[w, a, 3])))
            if not tx.isfinite(expected) or tx.abs(_add(expected, -state[w, a, 3])) > _mul(2e-5, scale):
                good = 0
        for a in range(k + 6):
            projected = 0.0
            for i in range(n + 6):
                projected = _add(projected, _mul(projection[w, i, a], full_applied[w, i]))
                if not tx.isfinite(full_applied[w, i]):
                    good = 0
            scale = tx.max(1.0, tx.max(tx.abs(projected), tx.abs(applied[w, a])))
            if not tx.isfinite(projected) or not tx.isfinite(applied[w, a]) or tx.abs(_add(projected, -applied[w, a])) > _mul(2e-5, scale):
                good = 0
        for i in range(n):
            q = full[w, i, 0]
            v = full[w, i, 1]
            if tx.abs(q) > 4096.0 or q < full_limits[w, i, 0] or q > full_limits[w, i, 1] or tx.abs(v) > full_limits[w, i, 2]:
                good = 0
            expected_q = offset[w, i]
            expected_v = 0.0
            expected_motor = 0.0
            for a in range(k):
                expected_q = _add(expected_q, _mul(matrix[w, i, a], state[w, a, 0]))
                expected_v = _add(expected_v, _mul(matrix[w, i, a], state[w, a, 1]))
                if i == independent[w, a]:
                    expected_motor = state[w, a, 2]
            for channel in tx.static_range(7):
                if not tx.isfinite(full[w, i, channel]):
                    good = 0
            for channel in tx.static_range(4):
                expected = expected_q
                if channel == 1:
                    expected = expected_v
                elif channel == 2:
                    expected = expected_motor
                elif channel == 3:
                    expected = full_applied[w, i + 6]
                actual = full[w, i, channel]
                scale = tx.max(1.0, tx.max(tx.abs(actual), tx.abs(expected)))
                if not tx.isfinite(expected) or tx.abs(_add(actual, -expected)) > _mul(2e-5, scale):
                    good = 0
            for channel in tx.static_range(3):
                if full[w, i, channel + 4] != 0.0:
                    good = 0
        for row in range(n - k):
            driver = row_indices[w, row, 0]
            follower = row_indices[w, row, 1]
            ratio = row_values[w, row, 0]
            for channel in tx.static_range(2):
                intercept = 0.0
                if channel == 0:
                    intercept = row_values[w, row, 1]
                left = full[w, follower, channel]
                right = _mul(ratio, full[w, driver, channel])
                scale = tx.max(1.0, tx.max(tx.abs(left), tx.max(tx.abs(right), tx.abs(intercept))))
                residual = _div(tx.abs(_add(_add(left, -right), -intercept)), scale)
                if not tx.isfinite(residual) or residual > 2e-5:
                    good = 0
        flags[w] = good


def make_floating_affine_kernels():
    """Separate handles for the two original active-bank directions."""
    functions = {
        "project": floating_affine_project,
        "expand": floating_affine_expand,
        "retain": floating_affine_retain,
        "restore": floating_affine_restore,
        "validate_state": floating_affine_validate_state,
    }
    return {name: (tx.kernel(function), tx.kernel(function)) for name, function in functions.items()}
