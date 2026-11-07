"""Complete floating-root mechanics on the original batch's TYX device.

Body, generalized velocity and motor dimensions are independent. No root
coordinate is represented by a scalar joint or a motor. All writes are staged;
the batch owner alone swaps accepted banks after every status/receipt check.
"""

import tyx as tx

from ._tyx_floating_numeric import _add, _div, _mul


def floating_geometry(
    count: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    parents: tx.ndarray(tx.i32, 2),
    kinds: tx.ndarray(tx.i32, 2),
    properties: tx.ndarray(tx.f32, 3),
    zero: tx.ndarray(tx.f32, 3),
    root: tx.ndarray(tx.f32, 2),
    state: tx.ndarray(tx.f32, 3),
    fk: tx.ndarray(tx.f32, 3),
    frame: tx.ndarray(tx.f32, 3),
    flags: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        n = meta[w, 1]
        flags[w] = 1
        norm = 0.0
        for a in tx.static_range(4):
            norm = _add(norm, _mul(root[w, 3 + a], root[w, 3 + a]))
        if not tx.isfinite(norm) or norm <= 0.0 or tx.abs(_add(norm, -1.0)) > 1e-5:
            flags[w] = 0
            norm = 1.0
        norm = tx.sqrt(norm)
        # Root rotation is always derived from the actual stored quaternion.
        for a in tx.static_range(3):
            fk[w, 0, a] = root[w, a]
            fk[w, 0, 12 + a] = 0.0
            if not tx.isfinite(root[w, a]) or tx.abs(root[w, a]) > 4096.0:
                flags[w] = 0
            for b in tx.static_range(3):
                qa = _div(root[w, 3 + a], norm)
                qb = _div(root[w, 3 + b], norm)
                qw = _div(root[w, 6], norm)
                value = _mul(2.0, _mul(qa, qb))
                if a == b:
                    total = 0.0
                    for k in tx.static_range(3):
                        if k != a:
                            qk = _div(root[w, 3 + k], norm)
                            total = _add(total, _mul(qk, qk))
                    value = _add(1.0, -_mul(2.0, total))
                elif (a + 1) % 3 == b:
                    value = _add(value, -_mul(2.0, _mul(qw, _div(root[w, 3 + (a + 2) % 3], norm))))
                else:
                    value = _add(value, _mul(2.0, _mul(qw, _div(root[w, 3 + (a + 1) % 3], norm))))
                fk[w, 0, 3 + a * 3 + b] = value
        for body in range(1, n + 1):
            parent = parents[w, body]
            q = state[w, body - 1, 0]
            if not tx.isfinite(q) or tx.abs(q) > 4096.0:
                flags[w] = 0
            for a in tx.static_range(3):
                coordinate = fk[w, parent, a]
                for b in tx.static_range(3):
                    coordinate = _add(
                        coordinate, _mul(fk[w, parent, 3 + a * 3 + b], properties[w, body, b])
                    )
                    value = 0.0
                    for k in tx.static_range(3):
                        value = _add(
                            value, _mul(fk[w, parent, 3 + a * 3 + k], zero[w, body, k * 3 + b])
                        )
                    frame[w, body, a * 3 + b] = value
                fk[w, body, a] = coordinate
                axis = 0.0
                for b in tx.static_range(3):
                    axis = _add(axis, _mul(frame[w, body, a * 3 + b], properties[w, body, 3 + b]))
                fk[w, body, 12 + a] = axis
            cosine = 1.0
            sine = 0.0
            if kinds[w, body] == 0:
                cosine = tx.cos(q)
                sine = tx.sin(q)
            for a in tx.static_range(3):
                if kinds[w, body] == 1:
                    fk[w, body, a] = _add(fk[w, body, a], _mul(fk[w, body, 12 + a], q))
                for b in tx.static_range(3):
                    value = frame[w, body, a * 3 + b]
                    if kinds[w, body] == 0:
                        value = 0.0
                        for k in tx.static_range(3):
                            turn = _mul(
                                _add(1.0, -cosine),
                                _mul(properties[w, body, 3 + k], properties[w, body, 3 + b]),
                            )
                            if k == b:
                                turn = _add(turn, cosine)
                            elif (k + 1) % 3 == b:
                                turn = _add(turn, -_mul(sine, properties[w, body, 3 + (k + 2) % 3]))
                            else:
                                turn = _add(turn, _mul(sine, properties[w, body, 3 + (k + 1) % 3]))
                            value = _add(value, _mul(frame[w, body, a * 3 + k], turn))
                    fk[w, body, 3 + a * 3 + b] = value
        for body in range(n + 1):
            for a in tx.static_range(3):
                point = fk[w, body, a]
                for b in tx.static_range(3):
                    point = _add(
                        point, _mul(fk[w, body, 3 + a * 3 + b], properties[w, body, 6 + b])
                    )
                fk[w, body, 15 + a] = point
            for a in tx.static_range(18):
                if not tx.isfinite(fk[w, body, a]):
                    flags[w] = 0


def floating_motion_energy(
    count: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    parents: tx.ndarray(tx.i32, 2),
    kinds: tx.ndarray(tx.i32, 2),
    properties: tx.ndarray(tx.f32, 3),
    principal: tx.ndarray(tx.f32, 3),
    root: tx.ndarray(tx.f32, 2),
    state: tx.ndarray(tx.f32, 3),
    fk: tx.ndarray(tx.f32, 3),
    origins: tx.ndarray(tx.f32, 3),
    twist: tx.ndarray(tx.f32, 3),
    energy: tx.ndarray(tx.f32, 1),
    flags: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        n = meta[w, 1]
        total = 0.0
        for a in tx.static_range(3):
            j = (a + 1) % 3
            k = (a + 2) % 3
            cj = _add(fk[w, 0, 15 + j], -fk[w, 0, j])
            ck = _add(fk[w, 0, 15 + k], -fk[w, 0, k])
            twist[w, 0, a] = root[w, 7 + a]
            twist[w, 0, 3 + a] = root[w, 10 + a]
            origins[w, 0, a] = _add(
                root[w, 7 + a], -_add(_mul(root[w, 10 + j], ck), -_mul(root[w, 10 + k], cj))
            )
        for body in range(1, n + 1):
            parent = parents[w, body]
            rate = state[w, body - 1, 1]
            for a in tx.static_range(3):
                j = (a + 1) % 3
                k = (a + 2) % 3
                rj = _add(fk[w, body, j], -fk[w, parent, j])
                rk = _add(fk[w, body, k], -fk[w, parent, k])
                speed = _add(
                    origins[w, parent, a],
                    _add(_mul(twist[w, parent, 3 + j], rk), -_mul(twist[w, parent, 3 + k], rj)),
                )
                omega = twist[w, parent, 3 + a]
                if kinds[w, body] == 0:
                    omega = _add(omega, _mul(fk[w, body, 12 + a], rate))
                else:
                    speed = _add(speed, _mul(fk[w, body, 12 + a], rate))
                origins[w, body, a] = speed
                twist[w, body, 3 + a] = omega
            for a in tx.static_range(3):
                j = (a + 1) % 3
                k = (a + 2) % 3
                cj = _add(fk[w, body, 15 + j], -fk[w, body, j])
                ck = _add(fk[w, body, 15 + k], -fk[w, body, k])
                twist[w, body, a] = _add(
                    origins[w, body, a],
                    _add(_mul(twist[w, body, 3 + j], ck), -_mul(twist[w, body, 3 + k], cj)),
                )
        for body in range(n + 1):
            for a in tx.static_range(3):
                speed = twist[w, body, a]
                local_spin = 0.0
                for b in tx.static_range(3):
                    rotation = 0.0
                    for k in tx.static_range(3):
                        rotation = _add(
                            rotation,
                            _mul(fk[w, body, 3 + b * 3 + k], principal[w, body, k * 3 + a]),
                        )
                    local_spin = _add(local_spin, _mul(rotation, twist[w, body, 3 + b]))
                # Nonnegative kinetic-energy terms use native FP32 products.
                # Weight first avoids rejecting a tiny v*v before a large mass
                # can scale it. An underflowed final energy contribution is zero;
                # overflow/NaN remain visible to the final finite/positive gate.
                linear_energy = (properties[w, body, 12] * speed) * speed * 0.5
                angular_energy = (properties[w, body, 9 + a] * local_spin) * local_spin * 0.5
                total = total + linear_energy + angular_energy
            for a in tx.static_range(6):
                if not tx.isfinite(twist[w, body, a]):
                    flags[w] = 0
        energy[w] = total
        if not tx.isfinite(total) or total < 0.0:
            flags[w] = 0


def floating_mechanics(
    count: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    parents: tx.ndarray(tx.i32, 2),
    kinds: tx.ndarray(tx.i32, 2),
    properties: tx.ndarray(tx.f32, 3),
    principal: tx.ndarray(tx.f32, 3),
    gravity: tx.ndarray(tx.f32, 2),
    effort: tx.ndarray(tx.f32, 2),
    loads: tx.ndarray(tx.f32, 3),
    root: tx.ndarray(tx.f32, 2),
    state: tx.ndarray(tx.f32, 3),
    fk: tx.ndarray(tx.f32, 3),
    linear: tx.ndarray(tx.f32, 4),
    angular: tx.ndarray(tx.f32, 4),
    spatial: tx.ndarray(tx.f32, 3),
    inertial: tx.ndarray(tx.f32, 3),
    mass: tx.ndarray(tx.f32, 3),
    vector: tx.ndarray(tx.f32, 3),
    entry_pose: tx.ndarray(tx.f32, 2),
):
    for w in tx.range(count):
        n = meta[w, 1]
        d = n + 6
        for a in tx.static_range(7):
            entry_pose[w, a] = root[w, a]
        for body in range(n + 1):
            for column in range(d):
                for a in tx.static_range(3):
                    linear[w, body, a, column] = 0.0
                    angular[w, body, a, column] = 0.0
                    if column == a:
                        linear[w, body, a, column] = 1.0
                    if column == 3 + a:
                        angular[w, body, a, column] = 1.0
                    if column >= 3 and column < 6:
                        j = (a + 1) % 3
                        k = (a + 2) % 3
                        if column == 3 + j:
                            linear[w, body, a, column] = _add(
                                fk[w, body, 15 + k], -fk[w, 0, 15 + k]
                            )
                        if column == 3 + k:
                            linear[w, body, a, column] = -_add(
                                fk[w, body, 15 + j], -fk[w, 0, 15 + j]
                            )
            ancestor = body
            for depth in range(n):
                if ancestor > 0:
                    column = 5 + ancestor
                    for a in tx.static_range(3):
                        value = fk[w, ancestor, 12 + a]
                        if kinds[w, ancestor] == 0:
                            j = (a + 1) % 3
                            k = (a + 2) % 3
                            rj = _add(fk[w, body, 15 + j], -fk[w, ancestor, j])
                            rk = _add(fk[w, body, 15 + k], -fk[w, ancestor, k])
                            value = _add(
                                _mul(fk[w, ancestor, 12 + j], rk),
                                -_mul(fk[w, ancestor, 12 + k], rj),
                            )
                            angular[w, body, a, column] = fk[w, ancestor, 12 + a]
                        linear[w, body, a, column] = value
                    ancestor = parents[w, ancestor]
            for a in tx.static_range(3):
                for b in tx.static_range(3):
                    value = 0.0
                    for k in tx.static_range(3):
                        value = _add(
                            value, _mul(fk[w, body, 3 + a * 3 + k], principal[w, body, k * 3 + b])
                        )
                    inertial[w, body, a * 3 + b] = value
        # Analytic zero-generalized-acceleration recurrence. Root COM a=0;
        # its eccentric authored origin has a nonzero centripetal bias.
        root_dot = 0.0
        root_square = 0.0
        for a in tx.static_range(3):
            center = _add(fk[w, 0, 15 + a], -fk[w, 0, a])
            root_dot = _add(root_dot, _mul(root[w, 10 + a], center))
            root_square = _add(root_square, _mul(root[w, 10 + a], root[w, 10 + a]))
        for body in range(n + 1):
            if body == 0:
                for a in tx.static_range(3):
                    center = _add(fk[w, 0, 15 + a], -fk[w, 0, a])
                    spatial[w, 0, a] = root[w, 10 + a]
                    spatial[w, 0, 3 + a] = 0.0
                    spatial[w, 0, 6 + a] = _add(
                        _mul(center, root_square), -_mul(root[w, 10 + a], root_dot)
                    )
            else:
                parent = parents[w, body]
                dot_wr = 0.0
                square_w = 0.0
                for a in tx.static_range(3):
                    arm = _add(fk[w, body, a], -fk[w, parent, a])
                    dot_wr = _add(dot_wr, _mul(spatial[w, parent, a], arm))
                    square_w = _add(square_w, _mul(spatial[w, parent, a], spatial[w, parent, a]))
                for a in tx.static_range(3):
                    j = (a + 1) % 3
                    k = (a + 2) % 3
                    ra = _add(fk[w, body, a], -fk[w, parent, a])
                    rj = _add(fk[w, body, j], -fk[w, parent, j])
                    rk = _add(fk[w, body, k], -fk[w, parent, k])
                    wa = spatial[w, parent, a]
                    cross_axis = _mul(
                        _add(
                            _mul(spatial[w, parent, j], fk[w, body, 12 + k]),
                            -_mul(spatial[w, parent, k], fk[w, body, 12 + j]),
                        ),
                        state[w, body - 1, 1],
                    )
                    spatial[w, body, a] = wa
                    spatial[w, body, 3 + a] = spatial[w, parent, 3 + a]
                    oa = _add(
                        spatial[w, parent, 6 + a],
                        _add(
                            _mul(spatial[w, parent, 3 + j], rk),
                            -_mul(spatial[w, parent, 3 + k], rj),
                        ),
                    )
                    oa = _add(oa, _add(_mul(wa, dot_wr), -_mul(ra, square_w)))
                    if kinds[w, body] == 0:
                        spatial[w, body, a] = _add(
                            wa, _mul(fk[w, body, 12 + a], state[w, body - 1, 1])
                        )
                        spatial[w, body, 3 + a] = _add(spatial[w, parent, 3 + a], cross_axis)
                    else:
                        oa = _add(oa, _mul(2.0, cross_axis))
                    spatial[w, body, 6 + a] = oa
            # Principal-frame gyro avoids a spurious isotropic torque from
            # the rounded world rotation's R R^T error at high angular speed.
            # Complete all three local components before their cross products.
            for b in tx.static_range(3):
                local_spin = 0.0
                for k in tx.static_range(3):
                    local_spin = _add(
                        local_spin, _mul(inertial[w, body, k * 3 + b], spatial[w, body, k])
                    )
                spatial[w, body, 9 + b] = local_spin
            dot_wc = 0.0
            square_w = 0.0
            for a in tx.static_range(3):
                center = _add(fk[w, body, 15 + a], -fk[w, body, a])
                dot_wc = _add(dot_wc, _mul(spatial[w, body, a], center))
                square_w = _add(square_w, _mul(spatial[w, body, a], spatial[w, body, a]))
                angular_force = 0.0
                for b in tx.static_range(3):
                    local_alpha = 0.0
                    for k in tx.static_range(3):
                        local_alpha = _add(
                            local_alpha, _mul(inertial[w, body, k * 3 + b], spatial[w, body, 3 + k])
                        )
                    j = (b + 1) % 3
                    k = (b + 2) % 3
                    local_gyro = _mul(
                        _add(properties[w, body, 9 + k], -properties[w, body, 9 + j]),
                        _mul(spatial[w, body, 9 + j], spatial[w, body, 9 + k]),
                    )
                    angular_force = _add(
                        angular_force,
                        _mul(
                            inertial[w, body, a * 3 + b],
                            _add(_mul(properties[w, body, 9 + b], local_alpha), local_gyro),
                        ),
                    )
                spatial[w, body, 15 + a] = angular_force
            for a in tx.static_range(3):
                j = (a + 1) % 3
                k = (a + 2) % 3
                ca = _add(fk[w, body, 15 + a], -fk[w, body, a])
                cj = _add(fk[w, body, 15 + j], -fk[w, body, j])
                ck = _add(fk[w, body, 15 + k], -fk[w, body, k])
                acceleration = _add(
                    spatial[w, body, 6 + a],
                    _add(_mul(spatial[w, body, 3 + j], ck), -_mul(spatial[w, body, 3 + k], cj)),
                )
                acceleration = _add(
                    acceleration, _add(_mul(spatial[w, body, a], dot_wc), -_mul(ca, square_w))
                )
                if body == 0:
                    acceleration = 0.0
                spatial[w, body, 12 + a] = _mul(
                    properties[w, body, 12], _add(acceleration, -gravity[w, a])
                )
        for a in range(d):
            for b in range(a, d):
                value = 0.0
                for body in range(n + 1):
                    for k in tx.static_range(3):
                        value = _add(
                            value,
                            _mul(
                                properties[w, body, 12],
                                _mul(linear[w, body, k, a], linear[w, body, k, b]),
                            ),
                        )
                        la = 0.0
                        lb = 0.0
                        for j in tx.static_range(3):
                            la = _add(
                                la, _mul(inertial[w, body, j * 3 + k], angular[w, body, j, a])
                            )
                            lb = _add(
                                lb, _mul(inertial[w, body, j * 3 + k], angular[w, body, j, b])
                            )
                        value = _add(value, _mul(properties[w, body, 9 + k], _mul(la, lb)))
                mass[w, a, b] = value
                mass[w, b, a] = value
            bias = 0.0
            external = 0.0
            if a >= 6:
                external = effort[w, a - 6]
            for body in range(n + 1):
                for k in tx.static_range(3):
                    bias = _add(
                        bias,
                        _add(
                            _mul(linear[w, body, k, a], spatial[w, body, 12 + k]),
                            _mul(angular[w, body, k, a], spatial[w, body, 15 + k]),
                        ),
                    )
                    external = _add(
                        external,
                        _add(
                            _mul(linear[w, body, k, a], loads[w, body, k]),
                            _mul(angular[w, body, k, a], loads[w, body, 3 + k]),
                        ),
                    )
            vector[w, a, 0] = bias
            vector[w, a, 1] = external
            vector[w, a, 4] = 0.0
            if a < 6:
                vector[w, a, 5] = root[w, 7 + a]
            else:
                vector[w, a, 4] = state[w, a - 6, 0]
                vector[w, a, 5] = state[w, a - 6, 1]


def floating_solve(
    count: tx.i32,
    h: tx.f32,
    meta: tx.ndarray(tx.i32, 2),
    properties: tx.ndarray(tx.f32, 3),
    commands: tx.ndarray(tx.f32, 3),
    root: tx.ndarray(tx.f32, 2),
    state: tx.ndarray(tx.f32, 3),
    fk: tx.ndarray(tx.f32, 3),
    applied: tx.ndarray(tx.f32, 2),
    tick: tx.ndarray(tx.i32, 1),
    next_root: tx.ndarray(tx.f32, 2),
    candidate: tx.ndarray(tx.f32, 3),
    next_applied: tx.ndarray(tx.f32, 2),
    mass: tx.ndarray(tx.f32, 3),
    factor: tx.ndarray(tx.f32, 3),
    mobility: tx.ndarray(tx.f32, 3),
    vector: tx.ndarray(tx.f32, 3),
    info: tx.ndarray(tx.i32, 2),
    diag: tx.ndarray(tx.f32, 2),
):
    for w in tx.range(count):
        n = meta[w, 1]
        d = n + 6
        slot = meta[w, 0]
        code = 0
        info[w, 0] = tick[slot]
        info[w, 1] = tick[slot]
        info[w, 3] = 0
        diag[w, 0] = h
        diag[w, 1] = 0.0
        diag[w, 2] = 0.0
        for i in tx.static_range(13):
            next_root[w, i] = root[w, i]
        for i in range(n):
            for k in tx.static_range(7):
                candidate[w, i, k] = state[w, i, k]
        for i in range(d):
            next_applied[w, i] = applied[w, i]
            vector[w, i, 2] = 0.0
            for j in range(d):
                factor[w, i, j] = 0.0
                mobility[w, i, j] = 0.0
        # Full D Cholesky/inverse. Only the true n motors are optimized below.
        for i in range(d):
            for j in range(i + 1):
                value = mass[w, i, j]
                for k in range(j):
                    value = _add(value, -_mul(factor[w, i, k], factor[w, j, k]))
                if i == j:
                    if not tx.isfinite(value) or value <= 0.0:
                        if code == 0:
                            code = 1
                        factor[w, i, j] = 1.0
                    else:
                        factor[w, i, j] = tx.sqrt(value)
                else:
                    factor[w, i, j] = _div(value, factor[w, j, j])
                    if not tx.isfinite(factor[w, i, j]) and code == 0:
                        code = 2
        for column in range(d):
            for i in range(d):
                value = 0.0
                if i == column:
                    value = 1.0
                for k in range(i):
                    value = _add(value, -_mul(factor[w, i, k], vector[w, k, 10]))
                vector[w, i, 10] = _div(value, factor[w, i, i])
            for i in range(d - 1, -1, -1):
                value = vector[w, i, 10]
                for k in range(i + 1, d):
                    value = _add(value, -_mul(factor[w, k, i], mobility[w, k, column]))
                mobility[w, i, column] = _div(value, factor[w, i, i])
                if not tx.isfinite(mobility[w, i, column]) and code == 0:
                    code = 2
        for i in range(d):
            response = 0.0
            for j in range(d):
                response = _add(
                    response, _mul(mobility[w, i, j], _add(vector[w, j, 1], -vector[w, j, 0]))
                )
            vector[w, i, 6] = _add(vector[w, i, 5], _mul(h, response))
            if (
                not tx.isfinite(vector[w, i, 6])
                or not tx.isfinite(vector[w, i, 0])
                or not tx.isfinite(vector[w, i, 1])
            ):
                if code == 0:
                    code = 2
        active = 0
        for motor in range(n):
            i = motor + 6
            body = motor + 1
            target = commands[w, motor, 0]
            rate = commands[w, motor, 1]
            ff = commands[w, motor, 2]
            if meta[w, 2] == 1:
                target = state[w, motor, 4]
                rate = state[w, motor, 5]
                ff = state[w, motor, 6]
            vector[w, i, 11] = target
            vector[w, i, 12] = rate
            vector[w, i, 13] = ff
            a = _add(
                _add(
                    _mul(properties[w, body, 13], _add(target, -state[w, motor, 0])),
                    _mul(properties[w, body, 14], rate),
                ),
                ff,
            )
            damping = _add(_mul(h, properties[w, body, 13]), properties[w, body, 14])
            vector[w, i, 7] = a
            vector[w, i, 8] = damping
            if damping == 0.0:
                vector[w, i, 2] = tx.min(
                    properties[w, body, 15], tx.max(-properties[w, body, 15], a)
                )
            else:
                active = active + 1
            if (not tx.isfinite(a) or not tx.isfinite(damping)) and code == 0:
                code = 2
        solved = 0
        eligible = 0
        sweep_limit = meta[w, 3]
        if active == 0:
            sweep_limit = 1
        # Full inverse's motor block equals inverse(Schur(root6)). Fixed effort
        # motors remain in every cross sum. There are no fake root PD rows.
        for iteration in range(sweep_limit):
            if code == 0 and solved == 0:
                for motor in range(n):
                    i = motor + 6
                    body = motor + 1
                    damping = vector[w, i, 8]
                    if damping > 0.0:
                        response = 0.0
                        for other in range(n):
                            response = _add(
                                response, _mul(mobility[w, i, other + 6], vector[w, other + 6, 2])
                            )
                        speed = _add(vector[w, i, 6], _mul(h, response))
                        denominator = _add(1.0, _mul(_mul(damping, h), mobility[w, i, i]))
                        numerator = _add(
                            _add(vector[w, i, 7], -_mul(damping, speed)), -vector[w, i, 2]
                        )
                        if (
                            not tx.isfinite(denominator)
                            or denominator <= 0.0
                            or not tx.isfinite(numerator)
                        ):
                            code = 2
                        value = _add(vector[w, i, 2], _div(numerator, denominator))
                        if not tx.isfinite(value):
                            code = 2
                        vector[w, i, 2] = tx.min(
                            properties[w, body, 15], tx.max(-properties[w, body, 15], value)
                        )
                for i in range(d):
                    response = 0.0
                    for motor in range(n):
                        response = _add(
                            response, _mul(mobility[w, i, motor + 6], vector[w, motor + 6, 2])
                        )
                    vector[w, i, 9] = _add(vector[w, i, 6], _mul(h, response))
                eligible = 1
                refined = 1
                absolute_residual = 0.0
                for motor in range(n):
                    i = motor + 6
                    body = motor + 1
                    speed = vector[w, i, 9]
                    target = vector[w, i, 11]
                    rate = vector[w, i, 12]
                    ff = vector[w, i, 13]
                    kp = properties[w, body, 13]
                    kd = properties[w, body, 14]
                    cap = properties[w, body, 15]
                    tau = vector[w, i, 2]
                    nextq = _add(state[w, motor, 0], _mul(h, speed))
                    raw = _add(
                        _add(
                            _mul(kp, _add(_add(target, -state[w, motor, 0]), -_mul(h, speed))),
                            _mul(kd, _add(rate, -speed)),
                        ),
                        ff,
                    )
                    rounded = _add(
                        _add(_mul(kp, _add(target, -nextq)), _mul(kd, _add(rate, -speed))), ff
                    )
                    if (
                        not tx.isfinite(raw)
                        or not tx.isfinite(rounded)
                        or not tx.isfinite(speed)
                        or not tx.isfinite(tau)
                    ):
                        code = 2
                    expected = tx.min(cap, tx.max(-cap, raw))
                    expected_q = tx.min(cap, tx.max(-cap, rounded))
                    residual = tx.max(tx.abs(_add(tau, -expected)), tx.abs(_add(tau, -expected_q)))
                    scale = tx.max(
                        1.0, tx.max(tx.abs(tau), tx.max(tx.abs(expected), tx.abs(expected_q)))
                    )
                    absolute_residual = tx.max(absolute_residual, residual)
                    if residual > _mul(5e-5, scale):
                        eligible = 0
                    if residual > _mul(1e-6, scale):
                        refined = 0
                max_delta = 0.0
                max_rhs = 0.0
                matrix_norm = 0.0
                equation_error = 0.0
                for i in range(d):
                    delta = _add(vector[w, i, 9], -vector[w, i, 5])
                    max_delta = tx.max(max_delta, tx.abs(delta))
                    rhs = _mul(h, _add(_add(vector[w, i, 2], vector[w, i, 1]), -vector[w, i, 0]))
                    max_rhs = tx.max(max_rhs, tx.abs(rhs))
                    equation = -rhs
                    row_norm = 0.0
                    for j in range(d):
                        row_norm = _add(row_norm, tx.abs(mass[w, i, j]))
                        equation = _add(
                            equation, _mul(mass[w, i, j], _add(vector[w, j, 9], -vector[w, j, 5]))
                        )
                    matrix_norm = tx.max(matrix_norm, row_norm)
                    equation_error = tx.max(equation_error, tx.abs(equation))
                    if (
                        not tx.isfinite(equation)
                        or not tx.isfinite(row_norm)
                        or not tx.isfinite(rhs)
                        or not tx.isfinite(vector[w, i, 9])
                    ):
                        code = 2
                equation_scale = tx.max(1.0, tx.max(_mul(matrix_norm, max_delta), max_rhs))
                equation_residual = _div(equation_error, equation_scale)
                if not tx.isfinite(equation_scale) or not tx.isfinite(equation_residual):
                    code = 2
                elif equation_residual > 2e-5:
                    code = 3
                diag[w, 1] = absolute_residual
                diag[w, 2] = equation_residual
                if active > 0:
                    info[w, 3] = iteration + 1
                if refined == 1 and eligible == 1 and code == 0:
                    solved = 1
        if code == 0 and eligible == 0:
            code = 4
        if code == 0:
            for i in range(d):
                acceleration = 0.0
                for j in range(d):
                    acceleration = _add(
                        acceleration,
                        _mul(
                            mobility[w, i, j],
                            _add(_add(vector[w, j, 2], vector[w, j, 1]), -vector[w, j, 0]),
                        ),
                    )
                vector[w, i, 3] = acceleration
                next_applied[w, i] = _add(vector[w, i, 1], vector[w, i, 2])
                if not tx.isfinite(acceleration) or not tx.isfinite(next_applied[w, i]):
                    code = 2
            for motor in range(n):
                speed = vector[w, motor + 6, 9]
                q = _add(state[w, motor, 0], _mul(h, speed))
                candidate[w, motor, 0] = q
                candidate[w, motor, 1] = speed
                candidate[w, motor, 2] = vector[w, motor + 6, 2]
                candidate[w, motor, 3] = next_applied[w, motor + 6]
                if meta[w, 2] == 1:
                    for a in tx.static_range(3):
                        candidate[w, motor, 4 + a] = commands[w, motor, a]
                if not tx.isfinite(q) or tx.abs(q) > 4096.0:
                    code = 5
            for a in tx.static_range(6):
                next_root[w, 7 + a] = vector[w, a, 9]
            # Left multiplication by exp(h*world_omega); no Euler angles.
            rx = _mul(h, vector[w, 3, 9])
            ry = _mul(h, vector[w, 4, 9])
            rz = _mul(h, vector[w, 5, 9])
            scale = tx.max(tx.abs(rx), tx.max(tx.abs(ry), tx.abs(rz)))
            dx = 0.0
            dy = 0.0
            dz = 0.0
            cosine = 1.0
            if not tx.isfinite(scale):
                code = 5
            elif scale > 0.0:
                sx = _div(rx, scale)
                sy = _div(ry, scale)
                sz = _div(rz, scale)
                length = tx.sqrt(_add(_add(_mul(sx, sx), _mul(sy, sy)), _mul(sz, sz)))
                angle = _mul(scale, length)
                if not tx.isfinite(angle) or angle > 3.141592653589793:
                    code = 5
                else:
                    sine = tx.sin(_mul(0.5, angle))
                    cosine = tx.cos(_mul(0.5, angle))
                    dx = _mul(_div(sx, length), sine)
                    dy = _mul(_div(sy, length), sine)
                    dz = _mul(_div(sz, length), sine)
            qx = root[w, 3]
            qy = root[w, 4]
            qz = root[w, 5]
            qw = root[w, 6]
            nx = _add(_add(_add(_mul(cosine, qx), _mul(qw, dx)), _mul(dy, qz)), -_mul(dz, qy))
            ny = _add(_add(_add(_mul(cosine, qy), _mul(qw, dy)), _mul(dz, qx)), -_mul(dx, qz))
            nz = _add(_add(_add(_mul(cosine, qz), _mul(qw, dz)), _mul(dx, qy)), -_mul(dy, qx))
            nw = _add(_add(_add(_mul(cosine, qw), -_mul(dx, qx)), -_mul(dy, qy)), -_mul(dz, qz))
            norm = tx.sqrt(_add(_add(_add(_mul(nx, nx), _mul(ny, ny)), _mul(nz, nz)), _mul(nw, nw)))
            if not tx.isfinite(norm) or norm <= 0.0:
                code = 5
                norm = 1.0
            next_root[w, 3] = _div(nx, norm)
            next_root[w, 4] = _div(ny, norm)
            next_root[w, 5] = _div(nz, norm)
            next_root[w, 6] = _div(nw, norm)
            alignment = 0.0
            for a in tx.static_range(4):
                alignment = _add(alignment, _mul(next_root[w, 3 + a], root[w, 3 + a]))
            if alignment < 0.0:
                for a in tx.static_range(4):
                    next_root[w, 3 + a] = -next_root[w, 3 + a]
            for a in tx.static_range(3):
                offset = 0.0
                for b in tx.static_range(3):
                    qa = next_root[w, 3 + a]
                    qb = next_root[w, 3 + b]
                    value = _mul(2.0, _mul(qa, qb))
                    if a == b:
                        total = 0.0
                        for k in tx.static_range(3):
                            if k != a:
                                total = _add(total, _mul(next_root[w, 3 + k], next_root[w, 3 + k]))
                        value = _add(1.0, -_mul(2.0, total))
                    elif (a + 1) % 3 == b:
                        value = _add(
                            value, -_mul(2.0, _mul(next_root[w, 6], next_root[w, 3 + (a + 2) % 3]))
                        )
                    else:
                        value = _add(
                            value, _mul(2.0, _mul(next_root[w, 6], next_root[w, 3 + (a + 1) % 3]))
                        )
                    offset = _add(offset, _mul(value, properties[w, 0, 6 + b]))
                old_offset = _add(fk[w, 0, 15 + a], -fk[w, 0, a])
                coordinate = _add(
                    root[w, a], _add(_mul(h, vector[w, a, 9]), _add(old_offset, -offset))
                )
                next_root[w, a] = coordinate
                if not tx.isfinite(coordinate) or tx.abs(coordinate) > 4096.0:
                    code = 5
            for a in tx.static_range(13):
                if not tx.isfinite(next_root[w, a]):
                    code = 2
            if tick[slot] >= 2147483647:
                code = 5
        info[w, 2] = code
        if code != 0:
            for a in tx.static_range(13):
                next_root[w, a] = root[w, a]
            for motor in range(n):
                for a in tx.static_range(7):
                    candidate[w, motor, a] = state[w, motor, a]
            for a in range(d):
                next_applied[w, a] = applied[w, a]


def floating_finalize(
    count: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    root: tx.ndarray(tx.f32, 2),
    next_root: tx.ndarray(tx.f32, 2),
    state: tx.ndarray(tx.f32, 3),
    candidate: tx.ndarray(tx.f32, 3),
    applied: tx.ndarray(tx.f32, 2),
    next_applied: tx.ndarray(tx.f32, 2),
    fk: tx.ndarray(tx.f32, 3),
    next_fk: tx.ndarray(tx.f32, 3),
    twist: tx.ndarray(tx.f32, 3),
    next_twist: tx.ndarray(tx.f32, 3),
    energy: tx.ndarray(tx.f32, 1),
    next_energy: tx.ndarray(tx.f32, 1),
    tick: tx.ndarray(tx.i32, 1),
    next_tick: tx.ndarray(tx.i32, 1),
    accepted: tx.ndarray(tx.i32, 1),
    info: tx.ndarray(tx.i32, 2),
    flags: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        n = meta[w, 1]
        slot = meta[w, 0]
        if flags[w] != 1 and info[w, 2] == 0:
            info[w, 2] = 6
        if info[w, 2] == 0:
            next_tick[slot] = tick[slot] + 1
            accepted[slot] = 1
        else:
            for a in tx.static_range(13):
                next_root[w, a] = root[w, a]
            for motor in range(n):
                for a in tx.static_range(7):
                    candidate[w, motor, a] = state[w, motor, a]
            for a in range(n + 6):
                next_applied[w, a] = applied[w, a]
            for body in range(n + 1):
                for a in tx.static_range(18):
                    next_fk[w, body, a] = fk[w, body, a]
                for a in tx.static_range(6):
                    next_twist[w, body, a] = twist[w, body, a]
            next_energy[w] = energy[w]
            next_tick[slot] = tick[slot]
            accepted[slot] = 0
        info[w, 1] = next_tick[slot]


def floating_restore(
    count: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    mask: tx.ndarray(tx.i32, 1),
    values_root: tx.ndarray(tx.f32, 2),
    values_state: tx.ndarray(tx.f32, 3),
    values_applied: tx.ndarray(tx.f32, 2),
    values_tick: tx.ndarray(tx.i32, 1),
    root: tx.ndarray(tx.f32, 2),
    next_root: tx.ndarray(tx.f32, 2),
    state: tx.ndarray(tx.f32, 3),
    candidate: tx.ndarray(tx.f32, 3),
    applied: tx.ndarray(tx.f32, 2),
    next_applied: tx.ndarray(tx.f32, 2),
    tick: tx.ndarray(tx.i32, 1),
    next_tick: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        n = meta[w, 1]
        slot = meta[w, 0]
        for a in tx.static_range(13):
            value = root[w, a]
            if mask[w] == 1:
                value = values_root[w, a]
            next_root[w, a] = value
        for motor in range(n):
            for a in tx.static_range(7):
                value = state[w, motor, a]
                if mask[w] == 1:
                    value = values_state[w, motor, a]
                candidate[w, motor, a] = value
        for a in range(n + 6):
            value = applied[w, a]
            if mask[w] == 1:
                value = values_applied[w, a]
            next_applied[w, a] = value
        next_tick[slot] = tick[slot]
        if mask[w] == 1:
            next_tick[slot] = values_tick[w]


def make_floating_tree_kernels():
    functions = {
        "geometry": floating_geometry,
        "energy": floating_motion_energy,
        "mechanics": floating_mechanics,
        "solve": floating_solve,
        "finalize": floating_finalize,
        "restore": floating_restore,
    }
    return {
        name: (tx.kernel(function), tx.kernel(function)) for name, function in functions.items()
    }
