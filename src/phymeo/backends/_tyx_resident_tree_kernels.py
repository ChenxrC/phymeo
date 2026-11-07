"""Full fixed-root R/P tree mechanics in persistent TYX arrays.

Each device lane owns one heterogeneous tree. All loops use its real n<=16;
there is no asset-specialized topology, HOST dynamics rescue or per-joint model.
The owner alone publishes after all passes and status reads succeed.
"""

import tyx as tx

from ._tyx_numeric import _add, _div, _mul


def tree_geometry(
    count: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    parents: tx.ndarray(tx.i32, 2),
    kinds: tx.ndarray(tx.i32, 2),
    properties: tx.ndarray(tx.f32, 3),
    zero: tx.ndarray(tx.f32, 3),
    state: tx.ndarray(tx.f32, 3),
    fk: tx.ndarray(tx.f32, 3),
    frame: tx.ndarray(tx.f32, 3),
    flags: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        n = meta[w, 1]
        flags[w] = 1
        for i in range(n):
            parent = parents[w, i]
            q = state[w, i, 0]
            if not tx.isfinite(q) or tx.abs(q) > 4096.0:
                flags[w] = 0
            for a in tx.static_range(3):
                coordinate = 0.0
                if parent >= 0:
                    coordinate = fk[w, parent, a]
                for b in tx.static_range(3):
                    pr = 0.0
                    if parent >= 0:
                        pr = fk[w, parent, 3 + a * 3 + b]
                    elif a == b:
                        pr = 1.0
                    coordinate = _add(coordinate, _mul(pr, properties[w, i, b]))
                fk[w, i, a] = coordinate
                for b in tx.static_range(3):
                    value = 0.0
                    for k in tx.static_range(3):
                        pr = 0.0
                        if parent >= 0:
                            pr = fk[w, parent, 3 + a * 3 + k]
                        elif a == k:
                            pr = 1.0
                        value = _add(value, _mul(pr, zero[w, i, k * 3 + b]))
                    frame[w, i, a * 3 + b] = value
                value = 0.0
                for k in tx.static_range(3):
                    value = _add(value, _mul(frame[w, i, a * 3 + k], properties[w, i, 3 + k]))
                fk[w, i, 12 + a] = value
            cosine = 1.0
            sine = 0.0
            if kinds[w, i] == 0:
                cosine = tx.cos(q)
                sine = tx.sin(q)
            for a in tx.static_range(3):
                if kinds[w, i] == 1:
                    fk[w, i, a] = _add(fk[w, i, a], _mul(fk[w, i, 12 + a], q))
                for b in tx.static_range(3):
                    value = frame[w, i, a * 3 + b]
                    if kinds[w, i] == 0:
                        value = 0.0
                        for k in tx.static_range(3):
                            turn = _mul(
                                _add(1.0, -cosine),
                                _mul(properties[w, i, 3 + k], properties[w, i, 3 + b]),
                            )
                            if k == b:
                                turn = _add(turn, cosine)
                            elif (k + 1) % 3 == b:
                                turn = _add(turn, -_mul(sine, properties[w, i, 3 + (k + 2) % 3]))
                            else:
                                turn = _add(turn, _mul(sine, properties[w, i, 3 + (k + 1) % 3]))
                            value = _add(value, _mul(frame[w, i, a * 3 + k], turn))
                    fk[w, i, 3 + a * 3 + b] = value
                point = fk[w, i, a]
                for b in tx.static_range(3):
                    point = _add(point, _mul(fk[w, i, 3 + a * 3 + b], properties[w, i, 6 + b]))
                fk[w, i, 15 + a] = point
            for a in tx.static_range(18):
                if not tx.isfinite(fk[w, i, a]):
                    flags[w] = 0


def tree_mechanics(
    count: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    parents: tx.ndarray(tx.i32, 2),
    kinds: tx.ndarray(tx.i32, 2),
    properties: tx.ndarray(tx.f32, 3),
    gravity: tx.ndarray(tx.f32, 2),
    commands: tx.ndarray(tx.f32, 3),
    state: tx.ndarray(tx.f32, 3),
    fk: tx.ndarray(tx.f32, 3),
    linear: tx.ndarray(tx.f32, 4),
    angular: tx.ndarray(tx.f32, 4),
    spatial: tx.ndarray(tx.f32, 3),
    mass: tx.ndarray(tx.f32, 3),
    vector: tx.ndarray(tx.f32, 3),
):
    for w in tx.range(count):
        n = meta[w, 1]
        # Jacobians refer to each true group COM at this same accepted q.
        for body in range(n):
            for column in range(n):
                for a in tx.static_range(3):
                    linear[w, body, a, column] = 0.0
                    angular[w, body, a, column] = 0.0
            ancestor = body
            for depth in range(n):
                if ancestor >= 0:
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
                            angular[w, body, a, ancestor] = fk[w, ancestor, 12 + a]
                        linear[w, body, a, ancestor] = value
                    ancestor = parents[w, ancestor]
        # Ordered exact velocity-bias recurrence, qdd=0, including Coriolis and
        # world I*alpha + omega x I*omega; no finite differences.
        for i in range(n):
            parent = parents[w, i]
            dot_wr = 0.0
            square_w = 0.0
            if parent >= 0:
                for a in tx.static_range(3):
                    arm = _add(fk[w, i, a], -fk[w, parent, a])
                    dot_wr = _add(dot_wr, _mul(spatial[w, parent, a], arm))
                    square_w = _add(square_w, _mul(spatial[w, parent, a], spatial[w, parent, a]))
            for a in tx.static_range(3):
                j = (a + 1) % 3
                k = (a + 2) % 3
                wa = 0.0
                wj = 0.0
                wk = 0.0
                aa = 0.0
                aj = 0.0
                ak = 0.0
                oa = 0.0
                ra = fk[w, i, a]
                rj = fk[w, i, j]
                rk = fk[w, i, k]
                if parent >= 0:
                    wa = spatial[w, parent, a]
                    wj = spatial[w, parent, j]
                    wk = spatial[w, parent, k]
                    aa = spatial[w, parent, 3 + a]
                    aj = spatial[w, parent, 3 + j]
                    ak = spatial[w, parent, 3 + k]
                    oa = spatial[w, parent, 6 + a]
                    ra = _add(ra, -fk[w, parent, a])
                    rj = _add(rj, -fk[w, parent, j])
                    rk = _add(rk, -fk[w, parent, k])
                cross_axis = _mul(
                    _add(_mul(wj, fk[w, i, 12 + k]), -_mul(wk, fk[w, i, 12 + j])), state[w, i, 1]
                )
                spatial[w, i, a] = wa
                spatial[w, i, 3 + a] = aa
                oa = _add(oa, _add(_mul(aj, rk), -_mul(ak, rj)))
                oa = _add(oa, _add(_mul(wa, dot_wr), -_mul(ra, square_w)))
                if kinds[w, i] == 0:
                    spatial[w, i, a] = _add(wa, _mul(fk[w, i, 12 + a], state[w, i, 1]))
                    spatial[w, i, 3 + a] = _add(aa, cross_axis)
                else:
                    oa = _add(oa, _mul(2.0, cross_axis))
                spatial[w, i, 6 + a] = oa
            dot_wc = 0.0
            square_w = 0.0
            for a in tx.static_range(3):
                ca = _add(fk[w, i, 15 + a], -fk[w, i, a])
                dot_wc = _add(dot_wc, _mul(spatial[w, i, a], ca))
                square_w = _add(square_w, _mul(spatial[w, i, a], spatial[w, i, a]))
                momentum = 0.0
                angular_force = 0.0
                for b in tx.static_range(3):
                    lw = 0.0
                    la = 0.0
                    for k in tx.static_range(3):
                        lw = _add(lw, _mul(fk[w, i, 3 + k * 3 + b], spatial[w, i, k]))
                        la = _add(la, _mul(fk[w, i, 3 + k * 3 + b], spatial[w, i, 3 + k]))
                    momentum = _add(
                        momentum, _mul(fk[w, i, 3 + a * 3 + b], _mul(properties[w, i, 9 + b], lw))
                    )
                    angular_force = _add(
                        angular_force,
                        _mul(fk[w, i, 3 + a * 3 + b], _mul(properties[w, i, 9 + b], la)),
                    )
                spatial[w, i, 9 + a] = momentum
                spatial[w, i, 15 + a] = angular_force
            for a in tx.static_range(3):
                j = (a + 1) % 3
                k = (a + 2) % 3
                ca = _add(fk[w, i, 15 + a], -fk[w, i, a])
                cj = _add(fk[w, i, 15 + j], -fk[w, i, j])
                ck = _add(fk[w, i, 15 + k], -fk[w, i, k])
                acceleration = _add(
                    spatial[w, i, 6 + a],
                    _add(_mul(spatial[w, i, 3 + j], ck), -_mul(spatial[w, i, 3 + k], cj)),
                )
                acceleration = _add(
                    acceleration, _add(_mul(spatial[w, i, a], dot_wc), -_mul(ca, square_w))
                )
                spatial[w, i, 12 + a] = _mul(
                    properties[w, i, 12], _add(acceleration, -gravity[w, a])
                )
                spatial[w, i, 15 + a] = _add(
                    spatial[w, i, 15 + a],
                    _add(
                        _mul(spatial[w, i, j], spatial[w, i, 9 + k]),
                        -_mul(spatial[w, i, k], spatial[w, i, 9 + j]),
                    ),
                )
        for a in range(n):
            for b in range(a, n):
                value = 0.0
                for link in range(n):
                    for k in tx.static_range(3):
                        value = _add(
                            value,
                            _mul(
                                properties[w, link, 12],
                                _mul(linear[w, link, k, a], linear[w, link, k, b]),
                            ),
                        )
                        la = 0.0
                        lb = 0.0
                        for j in tx.static_range(3):
                            la = _add(la, _mul(fk[w, link, 3 + j * 3 + k], angular[w, link, j, a]))
                            lb = _add(lb, _mul(fk[w, link, 3 + j * 3 + k], angular[w, link, j, b]))
                        value = _add(value, _mul(properties[w, link, 9 + k], _mul(la, lb)))
                mass[w, a, b] = value
                mass[w, b, a] = value
            bias = 0.0
            external = commands[w, a, 3]
            for link in range(n):
                for k in tx.static_range(3):
                    bias = _add(
                        bias,
                        _add(
                            _mul(linear[w, link, k, a], spatial[w, link, 12 + k]),
                            _mul(angular[w, link, k, a], spatial[w, link, 15 + k]),
                        ),
                    )
                    external = _add(
                        external,
                        _add(
                            _mul(linear[w, link, k, a], commands[w, link, 4 + k]),
                            _mul(angular[w, link, k, a], commands[w, link, 7 + k]),
                        ),
                    )
            vector[w, a, 0] = bias
            vector[w, a, 1] = external
            vector[w, a, 4] = state[w, a, 0]
            vector[w, a, 5] = state[w, a, 1]


def tree_solve(
    count: tx.i32,
    h: tx.f32,
    meta: tx.ndarray(tx.i32, 2),
    properties: tx.ndarray(tx.f32, 3),
    commands: tx.ndarray(tx.f32, 3),
    state: tx.ndarray(tx.f32, 3),
    tick: tx.ndarray(tx.i32, 1),
    candidate: tx.ndarray(tx.f32, 3),
    mass: tx.ndarray(tx.f32, 3),
    factor: tx.ndarray(tx.f32, 3),
    mobility: tx.ndarray(tx.f32, 3),
    vector: tx.ndarray(tx.f32, 3),
    info: tx.ndarray(tx.i32, 2),
    diag: tx.ndarray(tx.f32, 2),
):
    for w in tx.range(count):
        n = meta[w, 1]
        slot = meta[w, 0]
        code = 0
        info[w, 0] = tick[slot]
        info[w, 1] = tick[slot]
        info[w, 3] = 0
        diag[w, 0] = h
        diag[w, 1] = 0.0
        diag[w, 2] = 0.0
        for i in range(n):
            for k in tx.static_range(7):
                candidate[w, i, k] = state[w, i, k]
            for j in range(n):
                factor[w, i, j] = 0.0
                mobility[w, i, j] = 0.0
        # Full Cholesky. There is no diagonal shift, eigen clipping or fallback.
        for i in range(n):
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
        # Solve all RHS columns; shared/off-diagonal response stays physical.
        for column in range(n):
            for i in range(n):
                value = 0.0
                if i == column:
                    value = 1.0
                for k in range(i):
                    value = _add(value, -_mul(factor[w, i, k], vector[w, k, 10]))
                vector[w, i, 10] = _div(value, factor[w, i, i])
            for i in range(n - 1, -1, -1):
                value = vector[w, i, 10]
                for k in range(i + 1, n):
                    value = _add(value, -_mul(factor[w, k, i], mobility[w, k, column]))
                mobility[w, i, column] = _div(value, factor[w, i, i])
                if not tx.isfinite(mobility[w, i, column]) and code == 0:
                    code = 2
        active = 0
        for i in range(n):
            free = state[w, i, 1]
            response = 0.0
            for j in range(n):
                response = _add(
                    response, _mul(mobility[w, i, j], _add(vector[w, j, 1], -vector[w, j, 0]))
                )
            free = _add(free, _mul(h, response))
            vector[w, i, 6] = free
            target = commands[w, i, 0]
            rate = commands[w, i, 1]
            ff = commands[w, i, 2]
            if meta[w, 2] == 1:
                target = state[w, i, 4]
                rate = state[w, i, 5]
                ff = state[w, i, 6]
            vector[w, i, 11] = target
            vector[w, i, 12] = rate
            vector[w, i, 13] = ff
            a = _add(
                _add(
                    _mul(properties[w, i, 13], _add(target, -state[w, i, 0])),
                    _mul(properties[w, i, 14], rate),
                ),
                ff,
            )
            d = _add(_mul(h, properties[w, i, 13]), properties[w, i, 14])
            vector[w, i, 7] = a
            vector[w, i, 8] = d
            vector[w, i, 2] = 0.0
            if d == 0.0:
                vector[w, i, 2] = tx.min(properties[w, i, 15], tx.max(-properties[w, i, 15], a))
            else:
                active = active + 1
            if (
                not tx.isfinite(a)
                or not tx.isfinite(d)
                or not tx.isfinite(free)
                or not tx.isfinite(vector[w, i, 0])
                or not tx.isfinite(vector[w, i, 1])
            ) and code == 0:
                code = 2
        solved = 0
        eligible = 0
        # Projected coordinate method for the full SPD box QP. The scalar update
        # below is algebraically the original diag(1/d)+h*inverse(M) system;
        # fixed d=0 rows participate in every cross-mobility sum.
        sweep_limit = meta[w, 3]
        if active == 0:
            sweep_limit = 1
        for iteration in range(sweep_limit):
            if code == 0 and solved == 0:
                for i in range(n):
                    d = vector[w, i, 8]
                    if d > 0.0:
                        speed = vector[w, i, 6]
                        response = 0.0
                        for j in range(n):
                            response = _add(response, _mul(mobility[w, i, j], vector[w, j, 2]))
                        speed = _add(speed, _mul(h, response))
                        denominator = _add(1.0, _mul(_mul(d, h), mobility[w, i, i]))
                        numerator = _add(_add(vector[w, i, 7], -_mul(d, speed)), -vector[w, i, 2])
                        if (
                            not tx.isfinite(denominator)
                            or denominator <= 0.0
                            or not tx.isfinite(numerator)
                        ):
                            code = 2
                        correction = _div(numerator, denominator)
                        value = _add(vector[w, i, 2], correction)
                        if not tx.isfinite(value):
                            code = 2
                        vector[w, i, 2] = tx.min(
                            properties[w, i, 15], tx.max(-properties[w, i, 15], value)
                        )
                for i in range(n):
                    response = 0.0
                    for j in range(n):
                        response = _add(response, _mul(mobility[w, i, j], vector[w, j, 2]))
                    vector[w, i, 9] = _add(vector[w, i, 6], _mul(h, response))
                eligible = 1
                refined = 1
                absolute_residual = 0.0
                max_delta = 0.0
                max_rhs = 0.0
                matrix_norm = 0.0
                equation_error = 0.0
                for i in range(n):
                    speed = vector[w, i, 9]
                    target = vector[w, i, 11]
                    rate = vector[w, i, 12]
                    ff = vector[w, i, 13]
                    kp = properties[w, i, 13]
                    kd = properties[w, i, 14]
                    cap = properties[w, i, 15]
                    tau = vector[w, i, 2]
                    nextq = _add(state[w, i, 0], _mul(h, speed))
                    # Check both stated expression and rounded candidate state.
                    raw = _add(
                        _add(
                            _mul(kp, _add(_add(target, -state[w, i, 0]), -_mul(h, speed))),
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
                    # Internal refinement target is stricter, but final resource
                    # exhaustion still uses only the public 5e-5 physical gate.
                    if residual > _mul(1e-6, scale):
                        refined = 0
                    delta = _add(speed, -state[w, i, 1])
                    max_delta = tx.max(max_delta, tx.abs(delta))
                    rhs = _mul(h, _add(_add(tau, vector[w, i, 1]), -vector[w, i, 0]))
                    max_rhs = tx.max(max_rhs, tx.abs(rhs))
                    equation = -rhs
                    row_norm = 0.0
                    for j in range(n):
                        row_norm = _add(row_norm, tx.abs(mass[w, i, j]))
                        equation = _add(
                            equation, _mul(mass[w, i, j], _add(vector[w, j, 9], -state[w, j, 1]))
                        )
                    matrix_norm = tx.max(matrix_norm, row_norm)
                    equation_error = tx.max(equation_error, tx.abs(equation))
                    if (
                        not tx.isfinite(equation)
                        or not tx.isfinite(row_norm)
                        or not tx.isfinite(rhs)
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
            for i in range(n):
                speed = vector[w, i, 9]
                q = _add(state[w, i, 0], _mul(h, speed))
                candidate[w, i, 0] = q
                candidate[w, i, 1] = speed
                candidate[w, i, 2] = vector[w, i, 2]
                candidate[w, i, 3] = _add(vector[w, i, 1], vector[w, i, 2])
                acceleration = 0.0
                for j in range(n):
                    acceleration = _add(
                        acceleration,
                        _mul(
                            mobility[w, i, j],
                            _add(_add(vector[w, j, 2], vector[w, j, 1]), -vector[w, j, 0]),
                        ),
                    )
                vector[w, i, 3] = acceleration
                if meta[w, 2] == 1:
                    for j in tx.static_range(3):
                        candidate[w, i, 4 + j] = commands[w, i, j]
                if not tx.isfinite(q) or tx.abs(q) > 4096.0:
                    code = 5
                for j in tx.static_range(7):
                    if not tx.isfinite(candidate[w, i, j]):
                        code = 2
                if not tx.isfinite(vector[w, i, 3]):
                    code = 2
            if tick[slot] >= 2147483647:
                code = 5
        info[w, 2] = code
        if code != 0:
            for i in range(n):
                for j in tx.static_range(7):
                    candidate[w, i, j] = state[w, i, j]


def tree_energy(
    count: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    parents: tx.ndarray(tx.i32, 2),
    kinds: tx.ndarray(tx.i32, 2),
    properties: tx.ndarray(tx.f32, 3),
    state: tx.ndarray(tx.f32, 3),
    fk: tx.ndarray(tx.f32, 3),
    work: tx.ndarray(tx.f32, 3),
    energy: tx.ndarray(tx.f32, 1),
    flags: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        n = meta[w, 1]
        total = 0.0
        for body in range(n):
            for a in tx.static_range(6):
                work[w, body, a] = 0.0
            ancestor = body
            for depth in range(n):
                if ancestor >= 0:
                    for a in tx.static_range(3):
                        j = (a + 1) % 3
                        k = (a + 2) % 3
                        point_speed = fk[w, ancestor, 12 + a]
                        if kinds[w, ancestor] == 0:
                            rj = _add(fk[w, body, 15 + j], -fk[w, ancestor, j])
                            rk = _add(fk[w, body, 15 + k], -fk[w, ancestor, k])
                            point_speed = _add(
                                _mul(fk[w, ancestor, 12 + j], rk),
                                -_mul(fk[w, ancestor, 12 + k], rj),
                            )
                            work[w, body, 3 + a] = _add(
                                work[w, body, 3 + a],
                                _mul(fk[w, ancestor, 12 + a], state[w, ancestor, 1]),
                            )
                        work[w, body, a] = _add(
                            work[w, body, a], _mul(point_speed, state[w, ancestor, 1])
                        )
                    ancestor = parents[w, ancestor]
            for a in tx.static_range(3):
                local_omega = 0.0
                for b in tx.static_range(3):
                    local_omega = _add(
                        local_omega, _mul(fk[w, body, 3 + b * 3 + a], work[w, body, 3 + b])
                    )
                linear_term = _mul(
                    properties[w, body, 12], _mul(work[w, body, a], work[w, body, a])
                )
                angular_term = _mul(properties[w, body, 9 + a], _mul(local_omega, local_omega))
                total = _add(total, _mul(0.5, _add(linear_term, angular_term)))
        energy[w] = total
        if not tx.isfinite(total) or total < 0.0:
            flags[w] = 0


def tree_finalize(
    count: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    state: tx.ndarray(tx.f32, 3),
    candidate: tx.ndarray(tx.f32, 3),
    fk: tx.ndarray(tx.f32, 3),
    next_fk: tx.ndarray(tx.f32, 3),
    energy: tx.ndarray(tx.f32, 1),
    next_energy: tx.ndarray(tx.f32, 1),
    tick: tx.ndarray(tx.i32, 1),
    next_tick: tx.ndarray(tx.i32, 1),
    accepted: tx.ndarray(tx.i32, 1),
    info: tx.ndarray(tx.i32, 2),
    flags: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        slot = meta[w, 0]
        n = meta[w, 1]
        if flags[w] != 1 and info[w, 2] == 0:
            info[w, 2] = 6
        if info[w, 2] == 0:
            next_tick[slot] = tick[slot] + 1
            accepted[slot] = 1
            info[w, 1] = next_tick[slot]
        else:
            for i in range(n):
                for j in tx.static_range(7):
                    candidate[w, i, j] = state[w, i, j]
                for j in tx.static_range(18):
                    next_fk[w, i, j] = fk[w, i, j]
            next_energy[w] = energy[w]
            next_tick[slot] = tick[slot]
            accepted[slot] = 0
            info[w, 1] = tick[slot]


def tree_restore(
    count: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    mask: tx.ndarray(tx.i32, 1),
    values: tx.ndarray(tx.f32, 3),
    values_tick: tx.ndarray(tx.i32, 1),
    source: tx.ndarray(tx.f32, 3),
    target: tx.ndarray(tx.f32, 3),
    tick: tx.ndarray(tx.i32, 1),
    next_tick: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        slot = meta[w, 0]
        n = meta[w, 1]
        for i in range(n):
            for j in tx.static_range(7):
                value = source[w, i, j]
                if mask[w] == 1:
                    value = values[w, i, j]
                target[w, i, j] = value
        next_tick[slot] = tick[slot]
        if mask[w] == 1:
            next_tick[slot] = values_tick[w]


def make_tree_kernels():
    """Independent normal-math handles for the parent's two buffer directions."""
    functions = {
        "geometry": tree_geometry,
        "mechanics": tree_mechanics,
        "solve": tree_solve,
        "energy": tree_energy,
        "finalize": tree_finalize,
        "restore": tree_restore,
    }
    return {
        name: (tx.kernel(function), tx.kernel(function)) for name, function in functions.items()
    }
