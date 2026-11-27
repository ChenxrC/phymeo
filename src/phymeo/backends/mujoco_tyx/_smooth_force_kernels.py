# Copyright 2021 DeepMind Technologies Limited
# Copyright 2026 Phymeo contributors
# Apache-2.0; force semantics: MuJoCo 3.3.7 engine_forward.c/engine_passive.c.
"""Scalar-joint forces and refined physical mass solve, without integration."""

import tyx as tx

from ._two_lane import add_hi, add_lo, mul_hi, mul_lo


@tx.kernel
def forces(
    worlds: tx.i32,
    nv: tx.i32,
    nb: tx.i32,
    nu: tx.i32,
    actors: tx.ndarray(tx.i32, 2),
    params: tx.ndarray(tx.f32, 3),
    dofs: tx.ndarray(tx.i32, 2),
    passive_params: tx.ndarray(tx.f32, 3),
    q: tx.ndarray(tx.f32, 3),
    v: tx.ndarray(tx.f32, 2),
    jac: tx.ndarray(tx.f32, 4),
    jac_low: tx.ndarray(tx.f32, 4),
    bias: tx.ndarray(tx.f32, 2),
    bias_low: tx.ndarray(tx.f32, 2),
    ctrl: tx.ndarray(tx.f32, 3),
    applied: tx.ndarray(tx.f32, 3),
    body_force: tx.ndarray(tx.f32, 4),
    actuator_force: tx.ndarray(tx.f32, 2),
    actuator: tx.ndarray(tx.f32, 2),
    actuator_low: tx.ndarray(tx.f32, 2),
    passive: tx.ndarray(tx.f32, 2),
    external: tx.ndarray(tx.f32, 2),
    rhs: tx.ndarray(tx.f32, 2),
    rhs_low: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(worlds):
        status[w] = 0
        # World-body forces have no generalized effect, but malformed input
        # must still be rejected in the resident API as in the host API.
        for b in range(nb):
            for k in range(6):
                if not tx.isfinite(body_force[w, b, k, 0]) or not tx.isfinite(
                    body_force[w, b, k, 1]
                ):
                    status[w] = 1
        for d in range(nv):
            actuator[w, d] = 0.0
            actuator_low[w, d] = 0.0
        for a in range(nu):
            qa = actors[a, 0]
            da = actors[a, 1]
            ch = ctrl[w, a, 0]
            cl = ctrl[w, a, 1]
            if not tx.isfinite(ch) or not tx.isfinite(cl):
                status[w] = 1
            # Clamp in the original order: control, per-actuator force,
            # transmission sum, then total force at the joint.
            for side in range(2):
                boundh = params[a, 5 + side, 0]
                boundl = params[a, 5 + side, 1]
                delta = add_hi(ch, cl, -boundh, -boundl)
                if actors[a, 2] != 0 and (
                    (side == 0 and delta < 0.0) or (side == 1 and delta > 0.0)
                ):
                    ch = boundh
                    cl = boundl
            fh = mul_hi(params[a, 1, 0], params[a, 1, 1], ch, cl)
            fl = mul_lo(params[a, 1, 0], params[a, 1, 1], ch, cl)
            nh = add_hi(fh, fl, params[a, 2, 0], params[a, 2, 1])
            nl = add_lo(fh, fl, params[a, 2, 0], params[a, 2, 1])
            fh = nh
            fl = nl
            for term in range(2):
                xh = q[w, qa, 0]
                xl = q[w, qa, 1]
                if term == 1:
                    xh = v[w, da]
                    xl = 0.0
                mh = mul_hi(params[a, 0, 0], params[a, 0, 1], xh, xl)
                ml = mul_lo(params[a, 0, 0], params[a, 0, 1], xh, xl)
                ph = mul_hi(params[a, 3 + term, 0], params[a, 3 + term, 1], mh, ml)
                pl = mul_lo(params[a, 3 + term, 0], params[a, 3 + term, 1], mh, ml)
                nh = add_hi(fh, fl, ph, pl)
                nl = add_lo(fh, fl, ph, pl)
                fh = nh
                fl = nl
            for side in range(2):
                bh = params[a, 7 + side, 0]
                bl = params[a, 7 + side, 1]
                delta = add_hi(fh, fl, -bh, -bl)
                if actors[a, 3] != 0 and (
                    (side == 0 and delta < 0.0) or (side == 1 and delta > 0.0)
                ):
                    fh = bh
                    fl = bl
            actuator_force[w, a] = fh + fl
            mh = mul_hi(params[a, 0, 0], params[a, 0, 1], fh, fl)
            ml = mul_lo(params[a, 0, 0], params[a, 0, 1], fh, fl)
            nh = add_hi(actuator[w, da], actuator_low[w, da], mh, ml)
            nl = add_lo(actuator[w, da], actuator_low[w, da], mh, ml)
            actuator[w, da] = nh
            actuator_low[w, da] = nl
        for d in range(nv):
            for side in range(2):
                bh = passive_params[d, 3 + side, 0]
                bl = passive_params[d, 3 + side, 1]
                delta = add_hi(actuator[w, d], actuator_low[w, d], -bh, -bl)
                if dofs[d, 1] != 0 and ((side == 0 and delta < 0.0) or (side == 1 and delta > 0.0)):
                    actuator[w, d] = bh
                    actuator_low[w, d] = bl
            ph = -mul_hi(passive_params[d, 0, 0], passive_params[d, 0, 1], v[w, d], 0.0)
            pl = -mul_lo(passive_params[d, 0, 0], passive_params[d, 0, 1], v[w, d], 0.0)
            qa = dofs[d, 0]
            if qa >= 0:
                dh = add_hi(
                    q[w, qa, 0], q[w, qa, 1], -passive_params[d, 2, 0], -passive_params[d, 2, 1]
                )
                dl = add_lo(
                    q[w, qa, 0], q[w, qa, 1], -passive_params[d, 2, 0], -passive_params[d, 2, 1]
                )
                mh = -mul_hi(dh, dl, passive_params[d, 1, 0], passive_params[d, 1, 1])
                ml = -mul_lo(dh, dl, passive_params[d, 1, 0], passive_params[d, 1, 1])
                nh = add_hi(ph, pl, mh, ml)
                nl = add_lo(ph, pl, mh, ml)
                ph = nh
                pl = nl
            passive[w, d] = ph + pl
            eh = applied[w, d, 0]
            el = applied[w, d, 1]
            if not tx.isfinite(eh) or not tx.isfinite(el):
                status[w] = 1
            for b in range(1, nb):
                for k in range(6):
                    mh = mul_hi(
                        jac[w, b, k, d],
                        jac_low[w, b, k, d],
                        body_force[w, b, k, 0],
                        body_force[w, b, k, 1],
                    )
                    ml = mul_lo(
                        jac[w, b, k, d],
                        jac_low[w, b, k, d],
                        body_force[w, b, k, 0],
                        body_force[w, b, k, 1],
                    )
                    nh = add_hi(eh, el, mh, ml)
                    nl = add_lo(eh, el, mh, ml)
                    eh = nh
                    el = nl
            external[w, d] = eh + el
            fh = add_hi(ph, pl, -bias[w, d], -bias_low[w, d])
            fl = add_lo(ph, pl, -bias[w, d], -bias_low[w, d])
            nh = add_hi(fh, fl, eh, el)
            nl = add_lo(fh, fl, eh, el)
            rhs[w, d] = add_hi(nh, nl, actuator[w, d], actuator_low[w, d])
            rhs_low[w, d] = add_lo(nh, nl, actuator[w, d], actuator_low[w, d])
            if not tx.isfinite(rhs[w, d]) or not tx.isfinite(rhs_low[w, d]):
                status[w] = 1


@tx.kernel
def smooth_solve(
    worlds: tx.i32,
    nv: tx.i32,
    source_mass: tx.ndarray(tx.f32, 3),
    source_mass_low: tx.ndarray(tx.f32, 3),
    rhs: tx.ndarray(tx.f32, 2),
    rhs_low: tx.ndarray(tx.f32, 2),
    mass: tx.ndarray(tx.f32, 3),
    mass_low: tx.ndarray(tx.f32, 3),
    factor: tx.ndarray(tx.f32, 3),
    forward: tx.ndarray(tx.f32, 2),
    correction: tx.ndarray(tx.f32, 2),
    smooth: tx.ndarray(tx.f32, 2),
    smooth_low: tx.ndarray(tx.f32, 2),
    balance: tx.ndarray(tx.f32, 1),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(worlds):
        for i in range(nv):
            smooth[w, i] = 0.0
            smooth_low[w, i] = 0.0
            for j in range(nv):
                mass[w, i, j] = source_mass[w, i, j]
                mass_low[w, i, j] = source_mass_low[w, i, j]
                factor[w, i, j] = 0.0
                if (
                    not tx.isfinite(source_mass[w, i, j])
                    or not tx.isfinite(source_mass_low[w, i, j])
                    or tx.abs(source_mass[w, i, j] - source_mass[w, j, i])
                    > 2e-6 * (1.0 + tx.abs(source_mass[w, i, j]))
                ):
                    status[w] = 2
        for i in range(nv):
            for j in range(i + 1):
                value = mass[w, i, j]
                for k in range(j):
                    value = value - factor[w, i, k] * factor[w, j, k]
                if i == j:
                    if not tx.isfinite(value) or value <= 0.0:
                        status[w] = 2
                        value = 1.0
                    factor[w, i, j] = tx.sqrt(value)
                else:
                    factor[w, i, j] = value / factor[w, j, j]
        # Compensated residual refinement solves the same physical M;
        # no damping diagonal or tolerance relaxation is introduced.
        for iteration in range(4):
            worst = 0.0
            for i in range(nv):
                rh = rhs[w, i]
                rl = rhs_low[w, i]
                scale = 1.0 + tx.abs(rh + rl)
                for j in range(nv):
                    mh = mul_hi(mass[w, i, j], mass_low[w, i, j], smooth[w, j], smooth_low[w, j])
                    ml = mul_lo(mass[w, i, j], mass_low[w, i, j], smooth[w, j], smooth_low[w, j])
                    nh = add_hi(rh, rl, -mh, -ml)
                    nl = add_lo(rh, rl, -mh, -ml)
                    rh = nh
                    rl = nl
                    scale = scale + tx.abs(mh + ml)
                worst = tx.max(worst, tx.abs(rh + rl) / scale)
                forward[w, i] = rh + rl
            balance[w] = worst
            if iteration < 3:
                for i in range(nv):
                    value = forward[w, i]
                    for j in range(i):
                        value = value - factor[w, i, j] * forward[w, j]
                    forward[w, i] = value / factor[w, i, i]
                for reverse in range(nv):
                    i = nv - 1 - reverse
                    value = forward[w, i]
                    for j in range(i + 1, nv):
                        value = value - factor[w, j, i] * correction[w, j]
                    correction[w, i] = value / factor[w, i, i]
                for i in range(nv):
                    nh = add_hi(smooth[w, i], smooth_low[w, i], correction[w, i], 0.0)
                    nl = add_lo(smooth[w, i], smooth_low[w, i], correction[w, i], 0.0)
                    smooth[w, i] = nh
                    smooth_low[w, i] = nl
        if not tx.isfinite(balance[w]) or balance[w] > 2e-6:
            status[w] = 2
        for i in range(nv):
            if not tx.isfinite(smooth[w, i]) or not tx.isfinite(smooth_low[w, i]):
                status[w] = 2
