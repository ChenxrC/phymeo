# Copyright 2026 Phymeo contributors
# Apache-2.0; physical J^T I J and world-coordinate inverse dynamics.
"""Retain low lanes through COM Jacobians, physical M and gravity projection.

Velocity-dependent body wrenches come from the ordinary FP32 RNE kernel.
There is no state-dependent precision switch or native substitution.
"""

import tyx as tx

from ._two_lane import add_hi, add_lo, mul_hi, mul_lo


@tx.kernel
def precise_terms(
    worlds: tx.i32,
    nb: tx.i32,
    nv: tx.i32,
    bodies: tx.ndarray(tx.i32, 2),
    dofs: tx.ndarray(tx.i32, 2),
    ancestor: tx.ndarray(tx.i32, 2),
    inertia: tx.ndarray(tx.f32, 3),
    inertia_low: tx.ndarray(tx.f32, 3),
    weight: tx.ndarray(tx.f32, 1),
    weight_low: tx.ndarray(tx.f32, 1),
    armature: tx.ndarray(tx.f32, 1),
    armature_low: tx.ndarray(tx.f32, 1),
    gravity: tx.ndarray(tx.f32, 1),
    gravity_low: tx.ndarray(tx.f32, 1),
    rotation: tx.ndarray(tx.f32, 4),
    axes: tx.ndarray(tx.f32, 3),
    axes_low: tx.ndarray(tx.f32, 3),
    anchors: tx.ndarray(tx.f32, 4),
    com: tx.ndarray(tx.f32, 3),
    com_low: tx.ndarray(tx.f32, 3),
    coriolis: tx.ndarray(tx.f32, 3),
    jac: tx.ndarray(tx.f32, 4),
    jac_low: tx.ndarray(tx.f32, 4),
    iw: tx.ndarray(tx.f32, 4),
    iw_low: tx.ndarray(tx.f32, 4),
    mass: tx.ndarray(tx.f32, 3),
    mass_low: tx.ndarray(tx.f32, 3),
    bias: tx.ndarray(tx.f32, 2),
    bias_low: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(worlds):
        for b in range(nb):
            node = bodies[b, 0]
            for d in range(nv):
                for i in range(3):
                    j = (i + 1) % 3
                    k = (i + 2) % 3
                    lh = 0.0
                    ll = 0.0
                    ah = 0.0
                    al = 0.0
                    if ancestor[b, d] != 0:
                        if dofs[d, 0] != 0:
                            rjh = add_hi(
                                com[w, b, j],
                                com_low[w, b, j],
                                -anchors[w, d, j, 0],
                                -anchors[w, d, j, 1],
                            )
                            rjl = add_lo(
                                com[w, b, j],
                                com_low[w, b, j],
                                -anchors[w, d, j, 0],
                                -anchors[w, d, j, 1],
                            )
                            rkh = add_hi(
                                com[w, b, k],
                                com_low[w, b, k],
                                -anchors[w, d, k, 0],
                                -anchors[w, d, k, 1],
                            )
                            rkl = add_lo(
                                com[w, b, k],
                                com_low[w, b, k],
                                -anchors[w, d, k, 0],
                                -anchors[w, d, k, 1],
                            )
                            xh = mul_hi(axes[w, d, j], axes_low[w, d, j], rkh, rkl)
                            xl = mul_lo(axes[w, d, j], axes_low[w, d, j], rkh, rkl)
                            yh = mul_hi(axes[w, d, k], axes_low[w, d, k], rjh, rjl)
                            yl = mul_lo(axes[w, d, k], axes_low[w, d, k], rjh, rjl)
                            lh = add_hi(xh, xl, -yh, -yl)
                            ll = add_lo(xh, xl, -yh, -yl)
                            ah = axes[w, d, i]
                            al = axes_low[w, d, i]
                        else:
                            lh = axes[w, d, i]
                            ll = axes_low[w, d, i]
                    jac[w, b, i, d] = lh
                    jac_low[w, b, i, d] = ll
                    jac[w, b, i + 3, d] = ah
                    jac_low[w, b, i + 3, d] = al
            for i in range(3):
                for j in range(3):
                    sh = 0.0
                    sl = 0.0
                    for a in range(3):
                        for c in range(3):
                            xh = mul_hi(
                                rotation[w, node, 3 * i + a, 0],
                                rotation[w, node, 3 * i + a, 1],
                                inertia[b, a, c],
                                inertia_low[b, a, c],
                            )
                            xl = mul_lo(
                                rotation[w, node, 3 * i + a, 0],
                                rotation[w, node, 3 * i + a, 1],
                                inertia[b, a, c],
                                inertia_low[b, a, c],
                            )
                            yh = mul_hi(
                                xh,
                                xl,
                                rotation[w, node, 3 * j + c, 0],
                                rotation[w, node, 3 * j + c, 1],
                            )
                            yl = mul_lo(
                                xh,
                                xl,
                                rotation[w, node, 3 * j + c, 0],
                                rotation[w, node, 3 * j + c, 1],
                            )
                            nh = add_hi(sh, sl, yh, yl)
                            nl = add_lo(sh, sl, yh, yl)
                            sh = nh
                            sl = nl
                    iw[w, b, i, j] = sh
                    iw_low[w, b, i, j] = sl
        for i in range(nv):
            sh = 0.0
            sl = 0.0
            for b in range(1, nb):
                for k in range(6):
                    fh = coriolis[w, b, k]
                    fl = 0.0
                    if k < 3:
                        gh = -mul_hi(weight[b], weight_low[b], gravity[k], gravity_low[k])
                        gl = -mul_lo(weight[b], weight_low[b], gravity[k], gravity_low[k])
                        fh = add_hi(coriolis[w, b, k], 0.0, gh, gl)
                        fl = add_lo(coriolis[w, b, k], 0.0, gh, gl)
                    xh = mul_hi(jac[w, b, k, i], jac_low[w, b, k, i], fh, fl)
                    xl = mul_lo(jac[w, b, k, i], jac_low[w, b, k, i], fh, fl)
                    nh = add_hi(sh, sl, xh, xl)
                    nl = add_lo(sh, sl, xh, xl)
                    sh = nh
                    sl = nl
            bias[w, i] = sh
            bias_low[w, i] = sl
            if not tx.isfinite(sh) or not tx.isfinite(sl):
                status[w] = 1
            for j in range(i, nv):
                sh = 0.0
                sl = 0.0
                if i == j:
                    sh = armature[i]
                    sl = armature_low[i]
                for b in range(1, nb):
                    for k in range(3):
                        xh = mul_hi(weight[b], weight_low[b], jac[w, b, k, i], jac_low[w, b, k, i])
                        xl = mul_lo(weight[b], weight_low[b], jac[w, b, k, i], jac_low[w, b, k, i])
                        yh = mul_hi(xh, xl, jac[w, b, k, j], jac_low[w, b, k, j])
                        yl = mul_lo(xh, xl, jac[w, b, k, j], jac_low[w, b, k, j])
                        nh = add_hi(sh, sl, yh, yl)
                        nl = add_lo(sh, sl, yh, yl)
                        sh = nh
                        sl = nl
                        for a in range(3):
                            xh = mul_hi(
                                jac[w, b, k + 3, i],
                                jac_low[w, b, k + 3, i],
                                iw[w, b, k, a],
                                iw_low[w, b, k, a],
                            )
                            xl = mul_lo(
                                jac[w, b, k + 3, i],
                                jac_low[w, b, k + 3, i],
                                iw[w, b, k, a],
                                iw_low[w, b, k, a],
                            )
                            yh = mul_hi(xh, xl, jac[w, b, a + 3, j], jac_low[w, b, a + 3, j])
                            yl = mul_lo(xh, xl, jac[w, b, a + 3, j], jac_low[w, b, a + 3, j])
                            nh = add_hi(sh, sl, yh, yl)
                            nl = add_lo(sh, sl, yh, yl)
                            sh = nh
                            sl = nl
                mass[w, i, j] = sh
                mass[w, j, i] = sh
                mass_low[w, i, j] = sl
                mass_low[w, j, i] = sl
                if not tx.isfinite(sh) or not tx.isfinite(sl):
                    status[w] = 1
