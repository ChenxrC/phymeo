# Copyright 2021 DeepMind Technologies Limited
# Copyright 2026 Phymeo contributors
# Apache-2.0. IMPLICITFAST/advance semantics: MuJoCo 3.3.7 engine_forward.c,
# engine_derivative.c (Copyright 2022 DeepMind), engine_util_spatial.c.
"""Bounded GPU implicit system, original force assembly and candidate state."""

import tyx as tx

from ._elliptic_dynamics_kernels import scalar_force
from ._two_lane import add_hi, add_lo, div_hi, div_lo, mul_hi, mul_lo, sqrt_hi, sqrt_lo


@tx.kernel
def system(
    worlds: tx.i32,
    nv: tx.i32,
    hh: tx.f32,
    hl: tx.f32,
    derivative: tx.ndarray(tx.f32, 2),
    mass: tx.ndarray(tx.f32, 3),
    mass_low: tx.ndarray(tx.f32, 3),
    force: tx.ndarray(tx.f32, 2),
    force_low: tx.ndarray(tx.f32, 2),
    modified: tx.ndarray(tx.f32, 3),
    modified_low: tx.ndarray(tx.f32, 3),
    rhs: tx.ndarray(tx.f32, 2),
    rhs_low: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(worlds):
        status[w] = 0
        for i in range(nv):
            rhs[w, i] = force[w, i]
            rhs_low[w, i] = force_low[w, i]
            if not tx.isfinite(force[w, i]) or not tx.isfinite(force_low[w, i]):
                status[w] = 1
            for j in range(nv):
                ah = mass[w, i, j]
                al = mass_low[w, i, j]
                if i == j:
                    dh = -mul_hi(hh, hl, derivative[i, 0], derivative[i, 1])
                    dl = -mul_lo(hh, hl, derivative[i, 0], derivative[i, 1])
                    nh = add_hi(ah, al, dh, dl)
                    nl = add_lo(ah, al, dh, dl)
                    ah = nh
                    al = nl
                modified[w, i, j] = ah
                modified_low[w, i, j] = al


@tx.kernel
def coupled_force(
    worlds: tx.i32,
    nv: tx.i32,
    cap: tx.i32,
    scalar_cap: tx.i32,
    counts: tx.ndarray(tx.i32, 1),
    result_counts: tx.ndarray(tx.i32, 1),
    jac: tx.ndarray(tx.f32, 3),
    force: tx.ndarray(tx.f32, 2),
    scalar_counts: tx.ndarray(tx.i32, 1),
    sj: tx.ndarray(tx.f32, 3),
    sa: tx.ndarray(tx.f32, 2),
    sd: tx.ndarray(tx.f32, 2),
    sl: tx.ndarray(tx.f32, 2),
    sk: tx.ndarray(tx.i32, 2),
    acc: tx.ndarray(tx.f32, 2),
    acc_low: tx.ndarray(tx.f32, 2),
    smooth_rhs: tx.ndarray(tx.f32, 2),
    smooth_rhs_low: tx.ndarray(tx.f32, 2),
    total: tx.ndarray(tx.f32, 2),
    total_low: tx.ndarray(tx.f32, 2),
):
    for w in tx.range(worlds):
        n = counts[w]
        ns = scalar_counts[w]
        good = n >= 0 and n <= cap and n == result_counts[w] and ns >= 0 and ns <= scalar_cap
        for i in range(nv):
            ah = smooth_rhs[w, i]
            al = smooth_rhs_low[w, i]
            if good:
                for c in range(n):
                    row = w * cap + c
                    for k in range(3):
                        mh = mul_hi(jac[row, k, i], 0.0, force[row, k], 0.0)
                        ml = mul_lo(jac[row, k, i], 0.0, force[row, k], 0.0)
                        nh = add_hi(ah, al, mh, ml)
                        nl = add_lo(ah, al, mh, ml)
                        ah = nh
                        al = nl
                for c in range(ns):
                    rh = -sa[w, c]
                    rl = 0.0
                    for j in range(nv):
                        mh = mul_hi(sj[w, c, j], 0.0, acc[w, j], acc_low[w, j])
                        ml = mul_lo(sj[w, c, j], 0.0, acc[w, j], acc_low[w, j])
                        nh = add_hi(rh, rl, mh, ml)
                        nl = add_lo(rh, rl, mh, ml)
                        rh = nh
                        rl = nl
                    sf = scalar_force(rh + rl, sd[w, c], sl[w, c], sk[w, c])
                    mh = mul_hi(sj[w, c, i], 0.0, sf, 0.0)
                    ml = mul_lo(sj[w, c, i], 0.0, sf, 0.0)
                    nh = add_hi(ah, al, mh, ml)
                    nl = add_lo(ah, al, mh, ml)
                    ah = nh
                    al = nl
            else:
                # Explicit invalid marker propagates through system admission.
                ah = tx.sqrt(-1.0)
            total[w, i] = ah
            total_low[w, i] = al


@tx.kernel
def advance(
    worlds: tx.i32,
    nq: tx.i32,
    nv: tx.i32,
    nj: tx.i32,
    hh: tx.f32,
    hl: tx.f32,
    joints: tx.ndarray(tx.i32, 2),
    products: tx.ndarray(tx.i32, 3),
    q: tx.ndarray(tx.f32, 3),
    v: tx.ndarray(tx.f32, 2),
    clock: tx.ndarray(tx.f32, 2),
    acc: tx.ndarray(tx.f32, 2),
    acc_low: tx.ndarray(tx.f32, 2),
    nextq: tx.ndarray(tx.f32, 3),
    nextv: tx.ndarray(tx.f32, 2),
    nexttime: tx.ndarray(tx.f32, 2),
    scratch: tx.ndarray(tx.f32, 3),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(worlds):
        for i in range(nv):
            dh = mul_hi(hh, hl, acc[w, i], acc_low[w, i])
            dl = mul_lo(hh, hl, acc[w, i], acc_low[w, i])
            nextv[w, i] = add_hi(v[w, i], 0.0, dh, dl)
            if not tx.isfinite(v[w, i]) or not tx.isfinite(nextv[w, i]):
                status[w] = 4
        for i in range(nq):
            nextq[w, i, 0] = q[w, i, 0]
            nextq[w, i, 1] = q[w, i, 1]
        for j in range(nj):
            kind = joints[j, 0]
            qa = joints[j, 1]
            da = joints[j, 2]
            linear = 1
            if kind == 0:
                linear = 3
            for i in range(linear):
                dh = mul_hi(hh, hl, nextv[w, da + i], 0.0)
                dl = mul_lo(hh, hl, nextv[w, da + i], 0.0)
                nextq[w, qa + i, 0] = add_hi(q[w, qa + i, 0], q[w, qa + i, 1], dh, dl)
                nextq[w, qa + i, 1] = add_lo(q[w, qa + i, 0], q[w, qa + i, 1], dh, dl)
            if kind == 0:
                # cos(x) and sinc(x), x=|omega|*h/2; no singular zero-spin division.
                xh = 0.0
                xl = 0.0
                for i in range(3):
                    wh = mul_hi(nextv[w, da + 3 + i], 0.0, hh * 0.5, hl * 0.5)
                    wl = mul_lo(nextv[w, da + 3 + i], 0.0, hh * 0.5, hl * 0.5)
                    scratch[w, i + 4, 0] = wh
                    scratch[w, i + 4, 1] = wl
                    mh = mul_hi(wh, wl, wh, wl)
                    ml = mul_lo(wh, wl, wh, wl)
                    nh = add_hi(xh, xl, mh, ml)
                    nl = add_lo(xh, xl, mh, ml)
                    xh = nh
                    xl = nl
                if not tx.isfinite(xh) or xh + xl > 0.25:
                    status[w] = 4
                    xh = 0.0
                    xl = 0.0
                ch = 1.0
                cl = 0.0
                sh = 1.0
                sl = 0.0
                cth = 1.0
                ctl = 0.0
                sth = 1.0
                stl = 0.0
                for k in range(1, 11):
                    mh = -mul_hi(cth, ctl, xh, xl)
                    ml = -mul_lo(cth, ctl, xh, xl)
                    den = tx.cast_f32((2 * k - 1) * (2 * k))
                    cth = div_hi(mh, ml, den, 0.0)
                    ctl = div_lo(mh, ml, den, 0.0)
                    nh = add_hi(ch, cl, cth, ctl)
                    nl = add_lo(ch, cl, cth, ctl)
                    ch = nh
                    cl = nl
                    mh = -mul_hi(sth, stl, xh, xl)
                    ml = -mul_lo(sth, stl, xh, xl)
                    den = tx.cast_f32((2 * k) * (2 * k + 1))
                    sth = div_hi(mh, ml, den, 0.0)
                    stl = div_lo(mh, ml, den, 0.0)
                    nh = add_hi(sh, sl, sth, stl)
                    nl = add_lo(sh, sl, sth, stl)
                    sh = nh
                    sl = nl
                scratch[w, 7, 0] = ch
                scratch[w, 7, 1] = cl
                for i in range(3):
                    nh = mul_hi(scratch[w, i + 4, 0], scratch[w, i + 4, 1], sh, sl)
                    nl = mul_lo(scratch[w, i + 4, 0], scratch[w, i + 4, 1], sh, sl)
                    scratch[w, i + 8, 0] = nh
                    scratch[w, i + 8, 1] = nl
                nh = 0.0
                nl = 0.0
                for i in range(4):
                    mh = mul_hi(
                        q[w, qa + 3 + i, 0],
                        q[w, qa + 3 + i, 1],
                        q[w, qa + 3 + i, 0],
                        q[w, qa + 3 + i, 1],
                    )
                    ml = mul_lo(
                        q[w, qa + 3 + i, 0],
                        q[w, qa + 3 + i, 1],
                        q[w, qa + 3 + i, 0],
                        q[w, qa + 3 + i, 1],
                    )
                    ah = add_hi(nh, nl, mh, ml)
                    al = add_lo(nh, nl, mh, ml)
                    nh = ah
                    nl = al
                if not tx.isfinite(nh) or nh < 1e-20:
                    status[w] = 4
                    nh = 1.0
                    nl = 0.0
                lh = sqrt_hi(nh, nl)
                ll = sqrt_lo(nh, nl)
                for i in range(4):
                    scratch[w, i, 0] = div_hi(q[w, qa + 3 + i, 0], q[w, qa + 3 + i, 1], lh, ll)
                    scratch[w, i, 1] = div_lo(q[w, qa + 3 + i, 0], q[w, qa + 3 + i, 1], lh, ll)
                for i in range(4):
                    ah = 0.0
                    al = 0.0
                    for k in range(4):
                        u = products[i, k, 0]
                        t = products[i, k, 1] + 7
                        sign = tx.cast_f32(products[i, k, 2])
                        mh = sign * mul_hi(
                            scratch[w, u, 0], scratch[w, u, 1], scratch[w, t, 0], scratch[w, t, 1]
                        )
                        ml = sign * mul_lo(
                            scratch[w, u, 0], scratch[w, u, 1], scratch[w, t, 0], scratch[w, t, 1]
                        )
                        nh = add_hi(ah, al, mh, ml)
                        nl = add_lo(ah, al, mh, ml)
                        ah = nh
                        al = nl
                    nextq[w, qa + 3 + i, 0] = ah
                    nextq[w, qa + 3 + i, 1] = al
        nexttime[w, 0] = add_hi(clock[w, 0], clock[w, 1], hh, hl)
        nexttime[w, 1] = add_lo(clock[w, 0], clock[w, 1], hh, hl)
        for i in range(nq):
            if not tx.isfinite(nextq[w, i, 0]) or not tx.isfinite(nextq[w, i, 1]):
                status[w] = 4
        if not tx.isfinite(nexttime[w, 0]) or not tx.isfinite(nexttime[w, 1]):
            status[w] = 4
