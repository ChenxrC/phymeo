# Copyright 2026 Phymeo contributors. Apache-2.0; see LICENSE and NOTICE.
# Frame, material and impedance semantics follow MuJoCo 3.3.7.
"""Device assembly for zero-margin single-point convex and native box contacts."""

import tyx as tx


@tx.kernel
def prepare_boxes(
    worlds: tx.i32,
    nb: tx.i32,
    capacity: tx.i32,
    box_map: tx.ndarray(tx.i32, 1),
    positions: tx.ndarray(tx.f32, 3),
    rotations: tx.ndarray(tx.f32, 4),
    motion: tx.ndarray(tx.f32, 4),
    box_rotation: tx.ndarray(tx.f32, 4),
    counts: tx.ndarray(tx.i32, 1),
    eligible: tx.ndarray(tx.i32, 1),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(worlds):
        counts[w] = 0
        status[w] = 0
        for row in range(capacity):
            eligible[w * capacity + row] = 0
        for b in range(nb):
            g = box_map[b]
            for i in range(3):
                motion[w, b, 0, i] = positions[w, g, i]
                for j in range(3):
                    box_rotation[w, b, i, j] = rotations[w, g, i, j]


@tx.kernel
def compact_contacts(
    worlds: tx.i32,
    ngeom: tx.i32,
    qcap: tx.i32,
    cap: tx.i32,
    boxslots: tx.i32,
    kinds: tx.ndarray(tx.i32, 2),
    qcounts: tx.ndarray(tx.i32, 1),
    qpairs: tx.ndarray(tx.i32, 2),
    gap: tx.ndarray(tx.f32, 2),
    normal: tx.ndarray(tx.f32, 2),
    surface: tx.ndarray(tx.f32, 3),
    box_slot_pairs: tx.ndarray(tx.i32, 2),
    raw_boxes: tx.ndarray(tx.f32, 3),
    pairs: tx.ndarray(tx.i32, 2),
    geometry: tx.ndarray(tx.f32, 2),
    counts: tx.ndarray(tx.i32, 1),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(worlds):
        n = 0
        for i in range(qcounts[w]):
            src = w * qcap + i
            a = qpairs[src, 0] % ngeom
            b = qpairs[src, 1] % ngeom
            if (kinds[a, 0] != 6 or kinds[b, 0] != 6) and gap[src, 0] < 0.0:
                if n < cap:
                    dst = w * cap + n
                    pairs[dst, 0] = a
                    pairs[dst, 1] = b
                    geometry[dst, 0] = gap[src, 0]
                    for k in range(3):
                        geometry[dst, 1 + k] = 0.5 * (surface[src, 0, k] + surface[src, 1, k])
                        geometry[dst, 4 + k] = normal[src, k]
                n = n + 1
        for i in range(boxslots):
            if raw_boxes[w, i, 0] == 1.0 and raw_boxes[w, i, 1] < 0.0:
                if n < cap:
                    dst = w * cap + n
                    pairs[dst, 0] = box_slot_pairs[i, 0]
                    pairs[dst, 1] = box_slot_pairs[i, 1]
                    for k in range(7):
                        geometry[dst, k] = raw_boxes[w, i, 1 + k]
                n = n + 1
        counts[w] = tx.min(n, cap)
        if n > cap:
            status[w] = 1


@tx.kernel
def check_motion(
    worlds: tx.i32,
    nb: tx.i32,
    nv: tx.i32,
    com: tx.ndarray(tx.f32, 3),
    bj: tx.ndarray(tx.f32, 4),
    qvel: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(worlds):
        for b in range(nb):
            for i in range(3):
                if not tx.isfinite(com[w, b, i]):
                    status[w] = 2
            for i in range(6):
                for j in range(nv):
                    if not tx.isfinite(bj[w, b, i, j]):
                        status[w] = 2
        for j in range(nv):
            if not tx.isfinite(qvel[w, j]):
                status[w] = 2


@tx.kernel
def assemble_contacts(
    total: tx.i32,
    cap: tx.i32,
    nv: tx.i32,
    h: tx.f32,
    impratio: tx.f32,
    minimum: tx.f32,
    minimp: tx.f32,
    maximp: tx.f32,
    minmu: tx.f32,
    counts: tx.ndarray(tx.i32, 1),
    kinds: tx.ndarray(tx.i32, 2),
    params: tx.ndarray(tx.f32, 2),
    pairs: tx.ndarray(tx.i32, 2),
    geometry: tx.ndarray(tx.f32, 2),
    com: tx.ndarray(tx.f32, 3),
    bj: tx.ndarray(tx.f32, 4),
    qvel: tx.ndarray(tx.f32, 2),
    frame: tx.ndarray(tx.f32, 3),
    jac: tx.ndarray(tx.f32, 3),
    aref: tx.ndarray(tx.f32, 2),
    D: tx.ndarray(tx.f32, 2),
    R: tx.ndarray(tx.f32, 2),
    friction: tx.ndarray(tx.f32, 2),
    mu: tx.ndarray(tx.f32, 1),
    eligible: tx.ndarray(tx.i32, 1),
    law: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for r in tx.range(total):
        w = r // cap
        if r % cap < counts[w] and status[w] == 0:
            a = pairs[r, 0]
            b = pairs[r, 1]
            ba = kinds[a, 2]
            bb = kinds[b, 2]
            length = 0.0
            for k in range(3):
                length = length + geometry[r, 4 + k] * geometry[r, 4 + k]
            length = tx.sqrt(length)
            for k in range(3):
                frame[r, 0, k] = geometry[r, 4 + k] / tx.max(minimum, length)
            axis = 2
            if frame[r, 0, 1] > -0.5 and frame[r, 0, 1] < 0.5:
                axis = 1
            length = 0.0
            for k in range(3):
                seed = 0.0
                if k == axis:
                    seed = 1.0
                val = seed - frame[r, 0, k] * frame[r, 0, axis]
                frame[r, 1, k] = val
                length = length + val * val
            length = tx.sqrt(length)
            for k in range(3):
                frame[r, 1, k] = frame[r, 1, k] / tx.max(minimum, length)
            for k in range(3):
                j = (k + 1) % 3
                axis2 = (k + 2) % 3
                frame[r, 2, k] = (
                    frame[r, 0, j] * frame[r, 1, axis2] - frame[r, 0, axis2] * frame[r, 1, j]
                )
            s1 = params[a, 0]
            s2 = params[b, 0]
            mix = 0.5
            if s1 >= minimum or s2 >= minimum:
                mix = s1 / tx.max(minimum, s1 + s2)
                if s1 < minimum:
                    mix = 0.0
                elif s2 < minimum:
                    mix = 1.0
            priority = kinds[a, 1] - kinds[b, 1]
            if priority > 0:
                mix = 1.0
            elif priority < 0:
                mix = 0.0
            for j in range(7):
                value = mix * params[a, 1 + j] + (1.0 - mix) * params[b, 1 + j]
                if j < 2 and priority == 0 and (params[a, 1] <= 0.0 or params[b, 1] <= 0.0):
                    value = tx.min(params[a, 1 + j], params[b, 1 + j])
                law[r, j] = value
            tc = law[r, 0]
            dr = law[r, 1]
            if (tc > 0.0 and dr <= 0.0) or (tc <= 0.0 and dr > 0.0):
                tc = 0.02
                dr = 1.0
            if tc > 0.0:
                tc = tx.max(tc, 2.0 * h)
            lo = tx.min(maximp, tx.max(minimp, law[r, 2]))
            hi = tx.min(maximp, tx.max(minimp, law[r, 3]))
            width = tx.max(0.0, law[r, 4])
            mid = tx.min(maximp, tx.max(minimp, law[r, 5]))
            power = tx.max(1.0, law[r, 6])
            imp = 0.5 * (lo + hi)
            if lo != hi and width > minimum:
                x = tx.abs(geometry[r, 0] / width)
                y = x
                if x >= 1.0:
                    y = 1.0
                elif x <= 0.0:
                    y = 0.0
                elif power != 1.0:
                    if x <= mid:
                        y = tx.pow(x, power) / tx.pow(mid, power - 1.0)
                    else:
                        y = 1.0 - tx.pow(1.0 - x, power) / tx.pow(1.0 - mid, power - 1.0)
                imp = lo + y * (hi - lo)
            K = -tc / tx.max(minimum, hi * hi)
            B = -dr / tx.max(minimum, hi)
            if tc > 0.0:
                K = 1.0 / tx.max(minimum, hi * hi * tc * tc * dr * dr)
            if dr > 0.0:
                B = 2.0 / tx.max(minimum, hi * tc)
            fri = tx.max(params[a, 8], params[b, 8])
            if priority > 0:
                fri = params[a, 8]
            elif priority < 0:
                fri = params[b, 8]
            fri = tx.max(minmu, fri)
            friction[r, 0] = fri
            friction[r, 1] = fri
            r0 = tx.max(minimum, (1.0 - imp) * (params[a, 11] + params[b, 11]) / imp)
            rt = r0 / tx.max(minimum, impratio)
            mu[r] = fri * tx.sqrt(rt / r0)
            law[r, 7] = imp
            law[r, 8] = K
            law[r, 9] = B
            for row in range(3):
                vel = 0.0
                for col in range(nv):
                    value = 0.0
                    for k in range(3):
                        j = (k + 1) % 3
                        axis2 = (k + 2) % 3
                        va = (
                            bj[w, ba, k, col]
                            + bj[w, ba, 3 + j, col] * (geometry[r, 1 + axis2] - com[w, ba, axis2])
                            - bj[w, ba, 3 + axis2, col] * (geometry[r, 1 + j] - com[w, ba, j])
                        )
                        vb = (
                            bj[w, bb, k, col]
                            + bj[w, bb, 3 + j, col] * (geometry[r, 1 + axis2] - com[w, bb, axis2])
                            - bj[w, bb, 3 + axis2, col] * (geometry[r, 1 + j] - com[w, bb, j])
                        )
                        value = value + frame[r, row, k] * (vb - va)
                    jac[r, row, col] = value
                    vel = vel + value * qvel[w, col]
                rr = rt
                ref = -B * vel
                if row == 0:
                    rr = r0
                    ref = ref - K * imp * geometry[r, 0]
                R[r, row] = rr
                D[r, row] = 1.0 / rr
                aref[r, row] = ref
                if not tx.isfinite(ref) or not tx.isfinite(D[r, row]):
                    tx.atomic_add(status[w], 2)
            eligible[r] = 1
