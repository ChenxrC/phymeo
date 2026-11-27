# Copyright 2026 Phymeo contributors
# Apache-2.0; see LICENSE and NOTICE in this package.
"""Bounded primal acceleration solve; no integration or geometry generation.

The objective uses physical M, native elliptic contact regularization and native
equality, friction-loss and unilateral scalar costs. Newton/bisection is our own
algorithm, not a claim of reproducing native iteration order or stopping rules.
"""

import tyx as tx


@tx.kernel
def center(
    worlds: tx.i32,
    nv: tx.i32,
    capacity: tx.i32,
    scalar_capacity: tx.i32,
    counts: tx.ndarray(tx.i32, 1),
    scalar_counts: tx.ndarray(tx.i32, 1),
    smooth: tx.ndarray(tx.f32, 2),
    smooth_low: tx.ndarray(tx.f32, 2),
    jac: tx.ndarray(tx.f32, 3),
    aref: tx.ndarray(tx.f32, 2),
    sj: tx.ndarray(tx.f32, 3),
    sa: tx.ndarray(tx.f32, 2),
    centered_aref: tx.ndarray(tx.f32, 2),
    centered_sa: tx.ndarray(tx.f32, 2),
    acc: tx.ndarray(tx.f32, 2),
    acc_low: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(worlds):
        if status[w] == 0:
            for i in range(nv):
                acc[w, i] = 0.0
                acc_low[w, i] = 0.0
                if not tx.isfinite(smooth_low[w, i]):
                    status[w] = 8
            for c in range(counts[w]):
                row = w * capacity + c
                for k in tx.static_range(3):
                    hi = aref[row, k]
                    lo = 0.0
                    for i in range(nv):
                        p = tx.mul_rn(-jac[row, k, i], smooth[w, i])
                        pe = (
                            tx.fma(-jac[row, k, i], smooth[w, i], -p)
                            - jac[row, k, i] * smooth_low[w, i]
                        )
                        total = hi + p
                        z = total - hi
                        lo = lo + ((hi - (total - z)) + (p - z)) + pe
                        hi = total
                    centered_aref[row, k] = hi + lo
            for c in range(scalar_counts[w]):
                hi = sa[w, c]
                lo = 0.0
                for i in range(nv):
                    p = tx.mul_rn(-sj[w, c, i], smooth[w, i])
                    pe = tx.fma(-sj[w, c, i], smooth[w, i], -p) - sj[w, c, i] * smooth_low[w, i]
                    total = hi + p
                    z = total - hi
                    lo = lo + ((hi - (total - z)) + (p - z)) + pe
                    hi = total
                centered_sa[w, c] = hi + lo


@tx.kernel
def recover(
    worlds: tx.i32,
    nv: tx.i32,
    smooth: tx.ndarray(tx.f32, 2),
    smooth_low: tx.ndarray(tx.f32, 2),
    correction: tx.ndarray(tx.f32, 2),
    correction_low: tx.ndarray(tx.f32, 2),
    acc: tx.ndarray(tx.f32, 2),
    acc_low: tx.ndarray(tx.f32, 2),
    base: tx.ndarray(tx.f32, 2),
    counts: tx.ndarray(tx.i32, 1),
    result_counts: tx.ndarray(tx.i32, 1),
):
    for row in tx.range(worlds * nv):
        w = row // nv
        i = row % nv
        acc[w, i] = smooth[w, i] + correction[w, i]
        z = acc[w, i] - smooth[w, i]
        acc_low[w, i] = (
            (smooth[w, i] - (acc[w, i] - z))
            + (correction[w, i] - z)
            + smooth_low[w, i]
            + correction_low[w, i]
        )
        base[w, i] = smooth[w, i]
        if i == 0:
            result_counts[w] = counts[w]


@tx.kernel
def initialize(
    worlds: tx.i32,
    nv: tx.i32,
    capacity: tx.i32,
    scalar_capacity: tx.i32,
    counts: tx.ndarray(tx.i32, 1),
    scalar_counts: tx.ndarray(tx.i32, 1),
    mass: tx.ndarray(tx.f32, 3),
    smooth: tx.ndarray(tx.f32, 2),
    jac: tx.ndarray(tx.f32, 3),
    aref: tx.ndarray(tx.f32, 2),
    eligible: tx.ndarray(tx.i32, 1),
    sj: tx.ndarray(tx.f32, 3),
    sa: tx.ndarray(tx.f32, 2),
    sd: tx.ndarray(tx.f32, 2),
    sl: tx.ndarray(tx.f32, 2),
    sk: tx.ndarray(tx.i32, 2),
    factor: tx.ndarray(tx.f32, 3),
    scale: tx.ndarray(tx.f32, 2),
    acc: tx.ndarray(tx.f32, 2),
    iterations: tx.ndarray(tx.i32, 1),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(worlds):
        status[w] = 0
        iterations[w] = 0
        if (
            counts[w] < 0
            or counts[w] > capacity
            or scalar_counts[w] < 0
            or scalar_counts[w] > scalar_capacity
        ):
            status[w] = 8
        for i in range(nv):
            acc[w, i] = smooth[w, i]
            if (
                not tx.isfinite(smooth[w, i])
                or not tx.isfinite(mass[w, i, i])
                or mass[w, i, i] <= 0.0
            ):
                status[w] = 8
            scale[w, i] = 1.0 / tx.sqrt(tx.max(mass[w, i, i], 1e-30))
            for j in range(nv):
                if not tx.isfinite(mass[w, i, j]) or tx.abs(
                    mass[w, i, j] - mass[w, j, i]
                ) > 2e-6 * tx.max(1e-12, tx.max(tx.abs(mass[w, i, j]), tx.abs(mass[w, j, i]))):
                    status[w] = 8
        for i in range(nv):
            for j in range(i + 1):
                value = mass[w, i, j] * scale[w, i] * scale[w, j]
                for k in range(j):
                    value = value - factor[w, i, k] * factor[w, j, k]
                if i == j:
                    if not tx.isfinite(value) or value <= 0.0:
                        status[w] = 8
                    factor[w, i, j] = tx.sqrt(tx.max(value, 1e-30))
                else:
                    factor[w, i, j] = value / factor[w, j, j]
        if status[w] == 0:
            for c in range(counts[w]):
                row = w * capacity + c
                if eligible[row] != 1:
                    status[w] = 8
                for k in tx.static_range(3):
                    if not tx.isfinite(aref[row, k]):
                        status[w] = 8
                    for i in range(nv):
                        if not tx.isfinite(jac[row, k, i]):
                            status[w] = 8
            for c in range(scalar_counts[w]):
                if (
                    (sk[w, c] != 0 and sk[w, c] != 1 and sk[w, c] != 3 and sk[w, c] != 4)
                    or not tx.isfinite(sd[w, c])
                    or sd[w, c] <= 0.0
                    or not tx.isfinite(sa[w, c])
                    or not tx.isfinite(sl[w, c])
                    or sl[w, c] < 0.0
                ):
                    status[w] = 8
                for i in range(nv):
                    if not tx.isfinite(sj[w, c, i]):
                        status[w] = 8


@tx.kernel
def finalize(
    worlds: tx.i32, tolerance: tx.f32, balance: tx.ndarray(tx.f32, 1), status: tx.ndarray(tx.i32, 1)
):
    for w in tx.range(worlds):
        if status[w] == 0:
            status[w] = 7
            if tx.isfinite(balance[w]) and balance[w] <= tolerance:
                status[w] = 1


@tx.func
def scalar_force(r: tx.f32, d: tx.f32, loss: tx.f32, kind: tx.i32) -> tx.f32:
    return (
        tx.max(-loss, tx.min(loss, -d * r))
        if kind == 1
        else (tx.max(0.0, -d * r) if kind == 3 or kind == 4 else -d * r)
    )


@tx.func
def scalar_hessian(r: tx.f32, d: tx.f32, loss: tx.f32, kind: tx.i32) -> tx.f32:
    return (
        0.0
        if ((kind == 1 and tx.abs(d * r) > loss) or ((kind == 3 or kind == 4) and r >= 0.0))
        else d
    )


@tx.func
def scalar_cost(r: tx.f32, d: tx.f32, loss: tx.f32, kind: tx.i32) -> tx.f32:
    return (
        (loss * tx.abs(r) - 0.5 * loss * loss / d)
        if kind == 1 and tx.abs(d * r) > loss
        else (0.0 if (kind == 3 or kind == 4) and r >= 0.0 else 0.5 * d * r * r)
    )


@tx.func
def cone_cost(n: tx.f32, t: tx.f32, mu: tx.f32, d0: tx.f32, quadratic: tx.f32) -> tx.f32:
    return (
        0.0
        if n >= mu * t
        else (
            quadratic
            if mu * n + t <= 0.0
            else 0.5 * d0 / (mu * mu * (1.0 + mu * mu)) * (n - mu * t) * (n - mu * t)
        )
    )


@tx.func
def elliptic_cost(
    r0: tx.f32,
    r1: tx.f32,
    r2: tx.f32,
    d0: tx.f32,
    d1: tx.f32,
    d2: tx.f32,
    f1: tx.f32,
    f2: tx.f32,
    mu: tx.f32,
) -> tx.f32:
    return cone_cost(
        r0 * mu,
        tx.sqrt((r1 * f1) * (r1 * f1) + (r2 * f2) * (r2 * f2)),
        mu,
        d0,
        0.5 * (d0 * r0 * r0 + d1 * r1 * r1 + d2 * r2 * r2),
    )


@tx.func
def cone_derivative(
    n: tx.f32, t: tx.f32, dn: tx.f32, udotu: tx.f32, mu: tx.f32, d0: tx.f32, quadratic: tx.f32
) -> tx.f32:
    return (
        0.0
        if n >= mu * t
        else (
            quadratic
            if mu * n + t <= 0.0
            else d0
            / (mu * mu * (1.0 + mu * mu))
            * (n - mu * t)
            * (dn - mu * udotu / tx.max(t, 1e-30))
        )
    )


@tx.func
def elliptic_derivative(
    r0: tx.f32,
    r1: tx.f32,
    r2: tx.f32,
    p0: tx.f32,
    p1: tx.f32,
    p2: tx.f32,
    d0: tx.f32,
    d1: tx.f32,
    d2: tx.f32,
    f1: tx.f32,
    f2: tx.f32,
    mu: tx.f32,
) -> tx.f32:
    return cone_derivative(
        r0 * mu,
        tx.sqrt((r1 * f1) * (r1 * f1) + (r2 * f2) * (r2 * f2)),
        p0 * mu,
        r1 * f1 * p1 * f1 + r2 * f2 * p2 * f2,
        mu,
        d0,
        d0 * r0 * p0 + d1 * r1 * p1 + d2 * r2 * p2,
    )


@tx.kernel
def residuals(
    worlds: tx.i32,
    nv: tx.i32,
    capacity: tx.i32,
    counts: tx.ndarray(tx.i32, 1),
    jac: tx.ndarray(tx.f32, 3),
    aref: tx.ndarray(tx.f32, 2),
    acc: tx.ndarray(tx.f32, 2),
    acc_low: tx.ndarray(tx.f32, 2),
    residual: tx.ndarray(tx.f32, 2),
):
    for row in tx.range(worlds * capacity):
        w = row // capacity
        if counts[w] >= 0 and counts[w] <= capacity and row % capacity < counts[w]:
            for k in tx.static_range(3):
                value = -aref[row, k]
                low = 0.0
                for i in range(nv):
                    p = tx.mul_rn(jac[row, k, i], acc[w, i])
                    pe = tx.fma(jac[row, k, i], acc[w, i], -p) + jac[row, k, i] * acc_low[w, i]
                    total = value + p
                    z = total - value
                    low = low + (value - (total - z)) + (p - z) + pe
                    value = total
                residual[row, k] = value + low


@tx.kernel
def assemble(
    worlds: tx.i32,
    nv: tx.i32,
    capacity: tx.i32,
    scalar_capacity: tx.i32,
    counts: tx.ndarray(tx.i32, 1),
    scalar_counts: tx.ndarray(tx.i32, 1),
    mass: tx.ndarray(tx.f32, 3),
    smooth: tx.ndarray(tx.f32, 2),
    jac: tx.ndarray(tx.f32, 3),
    force: tx.ndarray(tx.f32, 2),
    hessian: tx.ndarray(tx.f32, 3),
    cost: tx.ndarray(tx.f32, 1),
    response_status: tx.ndarray(tx.i32, 1),
    sj: tx.ndarray(tx.f32, 3),
    sa: tx.ndarray(tx.f32, 2),
    sd: tx.ndarray(tx.f32, 2),
    sl: tx.ndarray(tx.f32, 2),
    sk: tx.ndarray(tx.i32, 2),
    acc: tx.ndarray(tx.f32, 2),
    acc_low: tx.ndarray(tx.f32, 2),
    matrix: tx.ndarray(tx.f32, 3),
    gradient: tx.ndarray(tx.f32, 2),
    objective: tx.ndarray(tx.f32, 1),
    balance: tx.ndarray(tx.f32, 1),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(worlds):
        if status[w] == 0:
            value = 0.0
            max_balance = 0.0
            for i in range(nv):
                g = 0.0
                g_low = 0.0
                scale = 1.0
                for j in range(nv):
                    delta = acc[w, j] - smooth[w, j]
                    term = tx.mul_rn(mass[w, i, j], delta)
                    total = g + term
                    z = total - g
                    g_low = (
                        g_low + (g - (total - z)) + (term - z) + tx.fma(mass[w, i, j], delta, -term)
                    ) + mass[w, i, j] * acc_low[w, j]
                    g = total
                    scale = scale + tx.abs(term)
                    value = value + 0.5 * (acc[w, i] - smooth[w, i]) * term
                    matrix[w, i, j] = mass[w, i, j]
                for c in range(counts[w]):
                    row = w * capacity + c
                    if response_status[row] != 0:
                        status[w] = 2
                    for k in tx.static_range(3):
                        term = tx.mul_rn(-jac[row, k, i], force[row, k])
                        total = g + term
                        z = total - g
                        g_low = (
                            g_low
                            + (g - (total - z))
                            + (term - z)
                            + tx.fma(-jac[row, k, i], force[row, k], -term)
                        )
                        g = total
                        scale = scale + tx.abs(term)
                    for j in range(nv):
                        h = 0.0
                        for k in tx.static_range(3):
                            for axis in tx.static_range(3):
                                h = h + jac[row, k, i] * hessian[row, k, axis] * jac[row, axis, j]
                        matrix[w, i, j] = matrix[w, i, j] + h
                for c in range(scalar_counts[w]):
                    r = -sa[w, c]
                    low = 0.0
                    for j in range(nv):
                        p = tx.mul_rn(sj[w, c, j], acc[w, j])
                        pe = tx.fma(sj[w, c, j], acc[w, j], -p) + sj[w, c, j] * acc_low[w, j]
                        total = r + p
                        z = total - r
                        low = low + (r - (total - z)) + (p - z) + pe
                        r = total
                    r = r + low
                    sf = scalar_force(r, sd[w, c], sl[w, c], sk[w, c])
                    sh = scalar_hessian(r, sd[w, c], sl[w, c], sk[w, c])
                    term = tx.mul_rn(-sj[w, c, i], sf)
                    total = g + term
                    z = total - g
                    g_low = g_low + (g - (total - z)) + (term - z) + tx.fma(-sj[w, c, i], sf, -term)
                    g = total
                    scale = scale + tx.abs(sj[w, c, i] * sf)
                    for j in range(nv):
                        matrix[w, i, j] = matrix[w, i, j] + sj[w, c, i] * sh * sj[w, c, j]
                g = g + g_low
                gradient[w, i] = g
                max_balance = tx.max(max_balance, tx.abs(g) / scale)
            for c in range(counts[w]):
                value = value + cost[w * capacity + c]
            for c in range(scalar_counts[w]):
                r = -sa[w, c]
                for j in range(nv):
                    r = r + sj[w, c, j] * acc[w, j] + sj[w, c, j] * acc_low[w, j]
                sc = scalar_cost(r, sd[w, c], sl[w, c], sk[w, c])
                value = value + sc
            objective[w] = value
            balance[w] = max_balance
            if not tx.isfinite(value) or not tx.isfinite(max_balance):
                status[w] = 3


@tx.kernel
def newton(
    worlds: tx.i32,
    nv: tx.i32,
    capacity: tx.i32,
    tolerance: tx.f32,
    counts: tx.ndarray(tx.i32, 1),
    scalar_counts: tx.ndarray(tx.i32, 1),
    mass: tx.ndarray(tx.f32, 3),
    smooth: tx.ndarray(tx.f32, 2),
    jac: tx.ndarray(tx.f32, 3),
    aref: tx.ndarray(tx.f32, 2),
    d: tx.ndarray(tx.f32, 2),
    friction: tx.ndarray(tx.f32, 2),
    mu: tx.ndarray(tx.f32, 1),
    sj: tx.ndarray(tx.f32, 3),
    sa: tx.ndarray(tx.f32, 2),
    sd: tx.ndarray(tx.f32, 2),
    sl: tx.ndarray(tx.f32, 2),
    sk: tx.ndarray(tx.i32, 2),
    acc: tx.ndarray(tx.f32, 2),
    acc_low: tx.ndarray(tx.f32, 2),
    matrix: tx.ndarray(tx.f32, 3),
    gradient: tx.ndarray(tx.f32, 2),
    objective: tx.ndarray(tx.f32, 1),
    balance: tx.ndarray(tx.f32, 1),
    factor: tx.ndarray(tx.f32, 3),
    scale: tx.ndarray(tx.f32, 2),
    direction: tx.ndarray(tx.f32, 2),
    trial: tx.ndarray(tx.f32, 2),
    trial_low: tx.ndarray(tx.f32, 2),
    trial_residual: tx.ndarray(tx.f32, 2),
    iterations: tx.ndarray(tx.i32, 1),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(worlds):
        if status[w] == 0 and balance[w] <= tolerance:
            status[w] = 1
        if status[w] == 0 and balance[w] > tolerance:
            iterations[w] = iterations[w] + 1
            for i in range(nv):
                if not tx.isfinite(matrix[w, i, i]) or matrix[w, i, i] <= 0.0:
                    status[w] = 4
                scale[w, i] = 1.0 / tx.sqrt(tx.max(matrix[w, i, i], 1e-30))
            for i in range(nv):
                for j in range(i + 1):
                    value = matrix[w, i, j] * scale[w, i] * scale[w, j]
                    for k in range(j):
                        value = value - factor[w, i, k] * factor[w, j, k]
                    if i == j:
                        if not tx.isfinite(value) or value <= 0.0:
                            status[w] = 4
                        factor[w, i, j] = tx.sqrt(tx.max(value, 1e-30))
                    else:
                        factor[w, i, j] = value / factor[w, j, j]
            if status[w] == 0:
                for i in range(nv):
                    value = -gradient[w, i] * scale[w, i]
                    for j in range(i):
                        value = value - factor[w, i, j] * direction[w, j]
                    direction[w, i] = value / factor[w, i, i]
                for ii in range(nv):
                    i = nv - 1 - ii
                    value = direction[w, i]
                    for j in range(i + 1, nv):
                        value = value - factor[w, j, i] * direction[w, j]
                    direction[w, i] = value / factor[w, i, i]
                slope = 0.0
                for i in range(nv):
                    direction[w, i] = direction[w, i] * scale[w, i]
                    slope = slope + gradient[w, i] * direction[w, i]
                if not tx.isfinite(slope) or slope >= 0.0:
                    status[w] = 5
                else:
                    accepted = 0
                    alpha = 1.0
                    lower = 0.0
                    upper = 1.0
                    for attempt in range(32):
                        if accepted == 0:
                            for i in range(nv):
                                p = tx.mul_rn(alpha, direction[w, i])
                                pe = tx.fma(alpha, direction[w, i], -p)
                                trial[w, i] = acc[w, i] + p
                                z = trial[w, i] - acc[w, i]
                                trial_low[w, i] = (
                                    (acc[w, i] - (trial[w, i] - z)) + (p - z) + acc_low[w, i] + pe
                                )
                            value = 0.0
                            for i in range(nv):
                                for j in range(nv):
                                    value = (
                                        value
                                        + direction[w, i]
                                        * mass[w, i, j]
                                        * (trial[w, j] - smooth[w, j])
                                        + direction[w, i] * mass[w, i, j] * trial_low[w, j]
                                    )
                            for c in range(counts[w]):
                                row = w * capacity + c
                                for axis in tx.static_range(3):
                                    r = -aref[row, axis]
                                    low = 0.0
                                    for i in range(nv):
                                        p = tx.mul_rn(jac[row, axis, i], trial[w, i])
                                        pe = (
                                            tx.fma(jac[row, axis, i], trial[w, i], -p)
                                            + jac[row, axis, i] * trial_low[w, i]
                                        )
                                        total = r + p
                                        z = total - r
                                        low = low + (r - (total - z)) + (p - z) + pe
                                        r = total
                                    trial_residual[row, axis] = r + low
                                r0 = trial_residual[row, 0]
                                r1 = trial_residual[row, 1]
                                r2 = trial_residual[row, 2]
                                p0 = 0.0
                                p1 = 0.0
                                p2 = 0.0
                                for i in range(nv):
                                    p0 = p0 + jac[row, 0, i] * direction[w, i]
                                    p1 = p1 + jac[row, 1, i] * direction[w, i]
                                    p2 = p2 + jac[row, 2, i] * direction[w, i]
                                value = value + elliptic_derivative(
                                    r0,
                                    r1,
                                    r2,
                                    p0,
                                    p1,
                                    p2,
                                    d[row, 0],
                                    d[row, 1],
                                    d[row, 2],
                                    friction[row, 0],
                                    friction[row, 1],
                                    mu[row],
                                )
                            for c in range(scalar_counts[w]):
                                r = -sa[w, c]
                                p = 0.0
                                low = 0.0
                                for j in range(nv):
                                    product = tx.mul_rn(sj[w, c, j], trial[w, j])
                                    pe = (
                                        tx.fma(sj[w, c, j], trial[w, j], -product)
                                        + sj[w, c, j] * trial_low[w, j]
                                    )
                                    total = r + product
                                    z = total - r
                                    low = low + (r - (total - z)) + (product - z) + pe
                                    r = total
                                    p = p + sj[w, c, j] * direction[w, j]
                                r = r + low
                                value = value - scalar_force(r, sd[w, c], sl[w, c], sk[w, c]) * p
                            if not tx.isfinite(value):
                                status[w] = 6
                            else:
                                if value <= 0.0:
                                    lower = alpha
                                else:
                                    upper = alpha
                                if (attempt == 0 and value <= 0.0) or tx.abs(value) <= -0.1 * slope:
                                    accepted = 1
                                    for i in range(nv):
                                        acc[w, i] = trial[w, i]
                                        acc_low[w, i] = trial_low[w, i]
                            alpha = 0.5 * (lower + upper)
                    if accepted == 0:
                        status[w] = 6
