# Copyright 2021 DeepMind Technologies Limited
# Copyright 2026 Phymeo contributors
# Apache-2.0; scalar laws follow MuJoCo 3.3.7 engine_core_constraint.c.
"""Original polynomial joint equality, dry friction and scalar joint limits."""

import tyx as tx

from ._two_lane import add_hi, add_lo, mul_hi, mul_lo


@tx.kernel
def rows(
    worlds: tx.i32,
    nv: tx.i32,
    candidates: tx.i32,
    capacity: tx.i32,
    neq: tx.i32,
    ids: tx.ndarray(tx.i32, 2),
    data: tx.ndarray(tx.f32, 3),
    h: tx.f32,
    q: tx.ndarray(tx.f32, 3),
    v: tx.ndarray(tx.f32, 2),
    active: tx.ndarray(tx.i32, 2),
    counts: tx.ndarray(tx.i32, 1),
    sj: tx.ndarray(tx.f32, 3),
    sa: tx.ndarray(tx.f32, 2),
    sd: tx.ndarray(tx.f32, 2),
    sl: tx.ndarray(tx.f32, 2),
    sk: tx.ndarray(tx.i32, 2),
    owner: tx.ndarray(tx.i32, 2),
    position: tx.ndarray(tx.f32, 2),
    margin: tx.ndarray(tx.f32, 2),
    velocity: tx.ndarray(tx.f32, 2),
    regularization: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(worlds):
        counts[w] = 0
        for e in range(neq):
            if active[w, e] != 0 and active[w, e] != 1:
                status[w] = 3
        for r in range(capacity):
            sa[w, r] = 0.0
            sd[w, r] = 0.0
            sl[w, r] = 0.0
            sk[w, r] = 0
            owner[w, r] = -1
            position[w, r] = 0.0
            margin[w, r] = 0.0
            velocity[w, r] = 0.0
            regularization[w, r] = 0.0
            for d in range(nv):
                sj[w, r, d] = 0.0
        for c in range(candidates):
            kind = ids[c, 0]
            obj = ids[c, 1]
            qa = ids[c, 2]
            da = ids[c, 3]
            qb = ids[c, 4]
            db = ids[c, 5]
            enabled = 1
            ph = 0.0
            pl = 0.0
            derivative = 0.0
            sign = 1.0
            if kind == 0:
                enabled = active[w, obj]
                ph = add_hi(q[w, qa, 0], q[w, qa, 1], -data[c, 5, 0], -data[c, 5, 1])
                pl = add_lo(q[w, qa, 0], q[w, qa, 1], -data[c, 5, 0], -data[c, 5, 1])
                polyh = data[c, 0, 0]
                polyl = data[c, 0, 1]
                if qb >= 0:
                    xh = add_hi(q[w, qb, 0], q[w, qb, 1], -data[c, 6, 0], -data[c, 6, 1])
                    xl = add_lo(q[w, qb, 0], q[w, qb, 1], -data[c, 6, 0], -data[c, 6, 1])
                    polyh = data[c, 4, 0]
                    polyl = data[c, 4, 1]
                    derivative = 4.0 * (polyh + polyl)
                    for reverse in range(4):
                        k = 3 - reverse
                        mh = mul_hi(polyh, polyl, xh, xl)
                        ml = mul_lo(polyh, polyl, xh, xl)
                        polyh = add_hi(mh, ml, data[c, k, 0], data[c, k, 1])
                        polyl = add_lo(mh, ml, data[c, k, 0], data[c, k, 1])
                        if k > 0:
                            derivative = derivative * (xh + xl) + tx.cast_f32(k) * (
                                data[c, k, 0] + data[c, k, 1]
                            )
                nh = add_hi(ph, pl, -polyh, -polyl)
                nl = add_lo(ph, pl, -polyh, -polyl)
                ph = nh
                pl = nl
            elif kind == 3:
                sign = data[c, 9, 0]
                ph = sign * add_hi(q[w, qa, 0], q[w, qa, 1], -data[c, 8, 0], -data[c, 8, 1])
                pl = sign * add_lo(q[w, qa, 0], q[w, qa, 1], -data[c, 8, 0], -data[c, 8, 1])
                distance = add_hi(ph, pl, -data[c, 7, 0], -data[c, 7, 1])
                if distance >= 0.0:
                    enabled = 0
            if enabled == 1:
                r = counts[w]
                counts[w] = r + 1
                if r >= capacity:
                    status[w] = 3
                else:
                    sj[w, r, da] = sign
                    if kind == 0 and db >= 0:
                        sj[w, r, db] = -derivative
                    vel = 0.0
                    for d in range(nv):
                        vel = vel + sj[w, r, d] * v[w, d]
                    pos = add_hi(ph, pl, -data[c, 7, 0], -data[c, 7, 1])
                    pos = pos + add_lo(ph, pl, -data[c, 7, 0], -data[c, 7, 1])
                    minimum = 1e-15
                    dmin = tx.min(0.9999, tx.max(0.0001, data[c, 14, 0]))
                    dmax = tx.min(0.9999, tx.max(0.0001, data[c, 15, 0]))
                    width = tx.max(0.0, data[c, 16, 0])
                    midpoint = tx.min(0.9999, tx.max(0.0001, data[c, 17, 0]))
                    power = tx.max(1.0, data[c, 18, 0])
                    imp = 0.5 * (dmin + dmax)
                    if dmin != dmax and width > minimum:
                        x = tx.abs(pos / width)
                        imp = dmin
                        if x >= 1.0:
                            imp = dmax
                        elif x > 0.0:
                            y = x
                            if power != 1.0:
                                if x <= midpoint:
                                    y = tx.pow(x, power) / tx.pow(midpoint, power - 1.0)
                                else:
                                    y = 1.0 - tx.pow(1.0 - x, power) / tx.pow(
                                        1.0 - midpoint, power - 1.0
                                    )
                            imp = dmin + y * (dmax - dmin)
                    ref0 = data[c, 12, 0]
                    ref1 = data[c, 13, 0]
                    if ref0 > 0.0:
                        ref0 = tx.max(ref0, 2.0 * h)
                    stiffness = 0.0
                    if kind != 1:
                        stiffness = -ref0 / tx.max(minimum, dmax * dmax)
                        if ref0 > 0.0:
                            stiffness = 1.0 / tx.max(
                                minimum, dmax * dmax * ref0 * ref0 * ref1 * ref1
                            )
                    damping = -ref1 / tx.max(minimum, dmax)
                    if ref1 > 0.0:
                        damping = 2.0 / tx.max(minimum, dmax * ref0)
                    R = tx.max(minimum, (1.0 - imp) * data[c, 10, 0] / imp)
                    sa[w, r] = -damping * vel - stiffness * imp * pos
                    sd[w, r] = 1.0 / R
                    sl[w, r] = data[c, 11, 0]
                    sk[w, r] = kind
                    owner[w, r] = obj
                    position[w, r] = ph + pl
                    margin[w, r] = data[c, 7, 0] + data[c, 7, 1]
                    velocity[w, r] = vel
                    regularization[w, r] = R
                    if (
                        not tx.isfinite(sa[w, r])
                        or not tx.isfinite(sd[w, r])
                        or not tx.isfinite(vel)
                        or sd[w, r] <= 0.0
                    ):
                        status[w] = 3
