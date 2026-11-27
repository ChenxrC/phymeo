# Copyright 2026 Phymeo contributors
# Licensed under Apache License 2.0; see LICENSE and NOTICE in this package.
"""Compensated GPU transform tree. No state integration or CPU pose fallback."""

import tyx as tx

from ._two_lane import add_hi, add_lo, div_hi, div_lo, mul_hi, mul_lo, sqrt_hi, sqrt_lo


@tx.kernel
def poses(
    worlds: tx.i32,
    nodes: tx.i32,
    nq: tx.i32,
    ids: tx.ndarray(tx.i32, 2),
    local: tx.ndarray(tx.f32, 3),
    joint: tx.ndarray(tx.f32, 3),
    products: tx.ndarray(tx.i32, 3),
    matrix: tx.ndarray(tx.i32, 3),
    q: tx.ndarray(tx.f32, 3),
    sampled_qpos: tx.ndarray(tx.f32, 3),
    mocap: tx.ndarray(tx.f32, 4),
    state: tx.ndarray(tx.f32, 4),
    rotation: tx.ndarray(tx.f32, 4),
    scratch: tx.ndarray(tx.f32, 3),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(worlds):
        status[w] = 0
        for i in range(nq):
            sampled_qpos[w, i, 0] = q[w, i, 0]
            sampled_qpos[w, i, 1] = q[w, i, 1]
        for n in range(nodes):
            parent = ids[n, 0]
            kind = ids[n, 1]
            index = ids[n, 2]
            for k in range(7):
                scratch[w, k, 0] = local[n, k, 0]
                scratch[w, k, 1] = local[n, k, 1]
            if kind == 3 or kind == 4:
                for k in range(7):
                    if kind == 3:
                        scratch[w, k, 0] = q[w, index + k, 0]
                        scratch[w, k, 1] = q[w, index + k, 1]
                    else:
                        scratch[w, k, 0] = mocap[w, index, k, 0]
                        scratch[w, k, 1] = mocap[w, index, k, 1]
            if kind == 1 or kind == 2:
                dh = add_hi(q[w, index, 0], q[w, index, 1], -joint[n, 3, 0], -joint[n, 3, 1])
                dl = add_lo(q[w, index, 0], q[w, index, 1], -joint[n, 3, 0], -joint[n, 3, 1])
                if kind == 2:
                    for k in range(3):
                        scratch[w, k, 0] = mul_hi(dh, dl, joint[n, k, 0], joint[n, k, 1])
                        scratch[w, k, 1] = mul_lo(dh, dl, joint[n, k, 0], joint[n, k, 1])
                else:
                    # Explicit software domain; never clip an unsupported state.
                    if tx.abs(dh + dl) > 32.0:
                        status[w] = 2
                    xh = dh * 0.5
                    xl = dl * 0.5
                    quadrant = 0
                    for reduction in range(16):
                        ph = 0.0
                        pl = 0.0
                        if xh + xl > 0.7853981633974483:
                            ph = -1.5707963705062866
                            pl = 4.371138828673793e-8
                            quadrant = quadrant + 1
                        elif xh + xl < -0.7853981633974483:
                            ph = 1.5707963705062866
                            pl = -4.371138828673793e-8
                            quadrant = quadrant - 1
                        nh = add_hi(xh, xl, ph, pl)
                        nl = add_lo(xh, xl, ph, pl)
                        xh = nh
                        xl = nl
                    xxh = -mul_hi(xh, xl, xh, xl)
                    xxl = -mul_lo(xh, xl, xh, xl)
                    sh = xh
                    sl = xl
                    ch = 1.0
                    cl = 0.0
                    sth = xh
                    stl = xl
                    cth = 1.0
                    ctl = 0.0
                    for t in range(1, 11):
                        mh = mul_hi(sth, stl, xxh, xxl)
                        ml = mul_lo(sth, stl, xxh, xxl)
                        den = tx.cast_f32((2 * t) * (2 * t + 1))
                        sth = div_hi(mh, ml, den, 0.0)
                        stl = div_lo(mh, ml, den, 0.0)
                        mh = mul_hi(cth, ctl, xxh, xxl)
                        ml = mul_lo(cth, ctl, xxh, xxl)
                        den = tx.cast_f32((2 * t - 1) * (2 * t))
                        cth = div_hi(mh, ml, den, 0.0)
                        ctl = div_lo(mh, ml, den, 0.0)
                        nh = add_hi(sh, sl, sth, stl)
                        nl = add_lo(sh, sl, sth, stl)
                        sh = nh
                        sl = nl
                        nh = add_hi(ch, cl, cth, ctl)
                        nl = add_lo(ch, cl, cth, ctl)
                        ch = nh
                        cl = nl
                    quadrant = (quadrant + 32) % 4
                    if quadrant == 1 or quadrant == 3:
                        nh = sh
                        nl = sl
                        sh = ch
                        sl = cl
                        ch = -nh
                        cl = -nl
                    if quadrant >= 2:
                        sh = -sh
                        sl = -sl
                        ch = -ch
                        cl = -cl
                    scratch[w, 3, 0] = ch
                    scratch[w, 3, 1] = cl
                    for k in range(3):
                        scratch[w, k + 4, 0] = mul_hi(sh, sl, joint[n, k, 0], joint[n, k, 1])
                        scratch[w, k + 4, 1] = mul_lo(sh, sl, joint[n, k, 0], joint[n, k, 1])
            for k in range(7):
                if not tx.isfinite(scratch[w, k, 0]) or not tx.isfinite(scratch[w, k, 1]):
                    status[w] = 1
            for i in range(3):
                ah = scratch[w, i, 0]
                al = scratch[w, i, 1]
                if parent >= 0:
                    ah = state[w, parent, i, 0]
                    al = state[w, parent, i, 1]
                    for j in range(3):
                        mh = mul_hi(
                            rotation[w, parent, 3 * i + j, 0],
                            rotation[w, parent, 3 * i + j, 1],
                            scratch[w, j, 0],
                            scratch[w, j, 1],
                        )
                        ml = mul_lo(
                            rotation[w, parent, 3 * i + j, 0],
                            rotation[w, parent, 3 * i + j, 1],
                            scratch[w, j, 0],
                            scratch[w, j, 1],
                        )
                        nh = add_hi(ah, al, mh, ml)
                        nl = add_lo(ah, al, mh, ml)
                        ah = nh
                        al = nl
                state[w, n, i, 0] = ah
                state[w, n, i, 1] = al
            for i in range(4):
                ah = scratch[w, i + 3, 0]
                al = scratch[w, i + 3, 1]
                if parent >= 0:
                    ah = 0.0
                    al = 0.0
                    for j in range(4):
                        u = products[i, j, 0] + 3
                        v = products[i, j, 1] + 3
                        sign = tx.cast_f32(products[i, j, 2])
                        mh = sign * mul_hi(
                            state[w, parent, u, 0],
                            state[w, parent, u, 1],
                            scratch[w, v, 0],
                            scratch[w, v, 1],
                        )
                        ml = sign * mul_lo(
                            state[w, parent, u, 0],
                            state[w, parent, u, 1],
                            scratch[w, v, 0],
                            scratch[w, v, 1],
                        )
                        nh = add_hi(ah, al, mh, ml)
                        nl = add_lo(ah, al, mh, ml)
                        ah = nh
                        al = nl
                state[w, n, i + 3, 0] = ah
                state[w, n, i + 3, 1] = al
            ah = 0.0
            al = 0.0
            for i in range(3, 7):
                mh = mul_hi(
                    state[w, n, i, 0], state[w, n, i, 1], state[w, n, i, 0], state[w, n, i, 1]
                )
                ml = mul_lo(
                    state[w, n, i, 0], state[w, n, i, 1], state[w, n, i, 0], state[w, n, i, 1]
                )
                nh = add_hi(ah, al, mh, ml)
                nl = add_lo(ah, al, mh, ml)
                ah = nh
                al = nl
            if not tx.isfinite(ah) or ah < 1e-20:
                status[w] = 1
                ah = 1.0
                al = 0.0
            lh = sqrt_hi(ah, al)
            ll = sqrt_lo(ah, al)
            for i in range(3, 7):
                nh = div_hi(state[w, n, i, 0], state[w, n, i, 1], lh, ll)
                nl = div_lo(state[w, n, i, 0], state[w, n, i, 1], lh, ll)
                state[w, n, i, 0] = nh
                state[w, n, i, 1] = nl
            for i in range(9):
                ah = 0.0
                al = 0.0
                for j in range(4):
                    u = matrix[i, j, 0] + 3
                    v = matrix[i, j, 1] + 3
                    sign = tx.cast_f32(matrix[i, j, 2])
                    mh = sign * mul_hi(
                        state[w, n, u, 0], state[w, n, u, 1], state[w, n, v, 0], state[w, n, v, 1]
                    )
                    ml = sign * mul_lo(
                        state[w, n, u, 0], state[w, n, u, 1], state[w, n, v, 0], state[w, n, v, 1]
                    )
                    nh = add_hi(ah, al, mh, ml)
                    nl = add_lo(ah, al, mh, ml)
                    ah = nh
                    al = nl
                rotation[w, n, i, 0] = ah
                rotation[w, n, i, 1] = al
                if not tx.isfinite(ah) or not tx.isfinite(al):
                    status[w] = 1
            for i in range(3):
                if not tx.isfinite(state[w, n, i, 0]) or not tx.isfinite(state[w, n, i, 1]):
                    status[w] = 1


@tx.kernel
def gather(
    count: tx.i32,
    ngeom: tx.i32,
    nbody: tx.i32,
    nv: tx.i32,
    body_nodes: tx.ndarray(tx.i32, 2),
    geom_nodes: tx.ndarray(tx.i32, 1),
    dofs: tx.ndarray(tx.i32, 2),
    axes_local: tx.ndarray(tx.f32, 3),
    state: tx.ndarray(tx.f32, 4),
    rot: tx.ndarray(tx.f32, 4),
    pos: tx.ndarray(tx.f32, 3),
    pos_low: tx.ndarray(tx.f32, 3),
    mat: tx.ndarray(tx.f32, 4),
    mat_low: tx.ndarray(tx.f32, 4),
    com: tx.ndarray(tx.f32, 3),
    com_low: tx.ndarray(tx.f32, 3),
    axes: tx.ndarray(tx.f32, 3),
    axes_low: tx.ndarray(tx.f32, 3),
    anchors: tx.ndarray(tx.f32, 4),
):
    for w in tx.range(count):
        for g in range(ngeom):
            node = geom_nodes[g]
            for i in range(3):
                pos[w, g, i] = state[w, node, i, 0]
                pos_low[w, g, i] = state[w, node, i, 1]
                for j in range(3):
                    mat[w, g, i, j] = rot[w, node, 3 * i + j, 0]
                    mat_low[w, g, i, j] = rot[w, node, 3 * i + j, 1]
        for b in range(nbody):
            for i in range(3):
                com[w, b, i] = state[w, body_nodes[b, 1], i, 0]
                com_low[w, b, i] = state[w, body_nodes[b, 1], i, 1]
        for d in range(nv):
            node = dofs[d, 2]
            for i in range(3):
                value = 0.0
                low = 0.0
                for j in range(3):
                    mh = mul_hi(
                        rot[w, node, 3 * i + j, 0],
                        rot[w, node, 3 * i + j, 1],
                        axes_local[d, j, 0],
                        axes_local[d, j, 3],
                    )
                    ml = mul_lo(
                        rot[w, node, 3 * i + j, 0],
                        rot[w, node, 3 * i + j, 1],
                        axes_local[d, j, 0],
                        axes_local[d, j, 3],
                    )
                    nh = add_hi(value, low, mh, ml)
                    nl = add_lo(value, low, mh, ml)
                    value = nh
                    low = nl
                axes[w, d, i] = value
                axes_low[w, d, i] = low
                ah = state[w, node, i, 0]
                al = state[w, node, i, 1]
                for j in range(3):
                    mh = mul_hi(
                        rot[w, node, 3 * i + j, 0],
                        rot[w, node, 3 * i + j, 1],
                        axes_local[d, j, 1],
                        axes_local[d, j, 2],
                    )
                    ml = mul_lo(
                        rot[w, node, 3 * i + j, 0],
                        rot[w, node, 3 * i + j, 1],
                        axes_local[d, j, 1],
                        axes_local[d, j, 2],
                    )
                    nh = add_hi(ah, al, mh, ml)
                    nl = add_lo(ah, al, mh, ml)
                    ah = nh
                    al = nl
                anchors[w, d, i, 0] = ah
                anchors[w, d, i, 1] = al
