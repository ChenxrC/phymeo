# Copyright 2025 The Newton Developers
# Copyright 2026 Phymeo contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# use only in compliance with the License. A copy is in LICENSE in this package.
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
#
# Modified TYX port of mujoco-warp v0.0.2: constraint._update_efc_row,
# collision_primitive_core.plane_sphere, forward.euler and math.quat_integrate.
# Scalar normal QP is solved directly for the admitted single-row domain.

"""Actual TYX FP32 forward/constraint/Euler computation; no Warp or NumPy solver."""

import tyx as tx


def advance(
    count: tx.i32,
    kind: tx.i32,
    nq: tx.i32,
    nv: tx.i32,
    nu: tx.i32,
    nbody: tx.i32,
    root_body: tx.i32,
    has_pair: tx.i32,
    contact_capacity: tx.i32,
    constraint_capacity: tx.i32,
    h: tx.f32,
    matrices: tx.ndarray(tx.f32, 3),
    vectors: tx.ndarray(tx.f32, 2),
    body_jac: tx.ndarray(tx.f32, 3),
    point_jac: tx.ndarray(tx.f32, 2),
    law: tx.ndarray(tx.f32, 1),
    act_ids: tx.ndarray(tx.i32, 2),
    act_values: tx.ndarray(tx.f32, 2),
    controls: tx.ndarray(tx.f32, 2),
    applied: tx.ndarray(tx.f32, 2),
    body_force: tx.ndarray(tx.f32, 3),
    q: tx.ndarray(tx.f32, 2),
    v: tx.ndarray(tx.f32, 2),
    clock: tx.ndarray(tx.f32, 1),
    qnext: tx.ndarray(tx.f32, 2),
    vnext: tx.ndarray(tx.f32, 2),
    tnext: tx.ndarray(tx.f32, 1),
    force: tx.ndarray(tx.f32, 2),
    smooth_acc: tx.ndarray(tx.f32, 2),
    acceleration: tx.ndarray(tx.f32, 2),
    integration_acc: tx.ndarray(tx.f32, 2),
    actuator: tx.ndarray(tx.f32, 2),
    constraint: tx.ndarray(tx.f32, 2),
    rows: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        status[w] = 0
        for i in tx.static_range(8):
            qnext[w, i] = q[w, i]
            vnext[w, i] = v[w, i]
            force[w, i] = 0.0
            actuator[w, i] = 0.0
            constraint[w, i] = 0.0
            acceleration[w, i] = 0.0
            integration_acc[w, i] = 0.0
        for i in tx.static_range(20):
            rows[w, i] = 0.0
        for i in range(nv):
            force[w, i] = applied[w, i] - vectors[0, i] - vectors[1, i] * v[w, i]
            if kind == 0:
                force[w, i] = force[w, i] - (vectors[2, i] * (q[w, i] - vectors[3, i]))
                if q[w, i] <= vectors[7, i] or q[w, i] >= vectors[8, i]:
                    status[w] = 3
                for b in range(nbody):
                    for xyz in tx.static_range(3):
                        force[w, i] = force[w, i] + (body_jac[b, xyz, i] * body_force[w, b, xyz])
        if kind == 1:
            for xyz in tx.static_range(3):
                force[w, xyz] = force[w, xyz] + (body_force[w, root_body, xyz])
            qw = q[w, 3]
            qx = q[w, 4]
            qy = q[w, 5]
            qz = q[w, 6]
            txw = body_force[w, root_body, 3]
            tyw = body_force[w, root_body, 4]
            tzw = body_force[w, root_body, 5]
            force[w, 3] = force[w, 3] + (
                (1.0 - 2.0 * (qy * qy + qz * qz)) * txw
                + 2.0 * (qx * qy + qw * qz) * tyw
                + 2.0 * (qx * qz - qw * qy) * tzw
            )
            force[w, 4] = force[w, 4] + (
                2.0 * (qx * qy - qw * qz) * txw
                + (1.0 - 2.0 * (qx * qx + qz * qz)) * tyw
                + 2.0 * (qy * qz + qw * qx) * tzw
            )
            force[w, 5] = force[w, 5] + (
                2.0 * (qx * qz + qw * qy) * txw
                + 2.0 * (qy * qz - qw * qx) * tyw
                + (1.0 - 2.0 * (qx * qx + qy * qy)) * tzw
            )
        for a in range(nu):
            if act_ids[a, 2] == 0:
                qa = act_ids[a, 0]
                va = act_ids[a, 1]
                gear = act_values[a, 0]
                ctrl = tx.min(act_values[a, 6], tx.max(act_values[a, 5], controls[w, a]))
                value = act_values[a, 1] * ctrl + act_values[a, 2]
                value = value + (act_values[a, 3] * gear * q[w, qa])
                value = value + (act_values[a, 4] * gear * v[w, va])
                value = tx.min(act_values[a, 8], tx.max(act_values[a, 7], value))
                actuator[w, va] = actuator[w, va] + (gear * value)
        for i in range(nv):
            force[w, i] = force[w, i] + (actuator[w, i])
        for i in range(nv):
            smooth_acc[w, i] = 0.0
            for j in range(nv):
                smooth_acc[w, i] = smooth_acc[w, i] + (matrices[0, i, j] * force[w, j])
            acceleration[w, i] = smooth_acc[w, i]
        if has_pair == 1:
            cx = law[6]
            cy = law[7]
            cz = law[8]
            if kind == 1:
                cx = q[w, 0]
                cy = q[w, 1]
                cz = q[w, 2]
            else:
                for j in range(nv):
                    shift = q[w, j] - vectors[4, j]
                    cx = cx + (point_jac[0, j] * shift)
                    cy = cy + (point_jac[1, j] * shift)
                    cz = cz + (point_jac[2, j] * shift)
            distance = (
                (cx - law[0]) * law[3] + (cy - law[1]) * law[4] + (cz - law[2]) * law[5] - law[9]
            )
            rows[w, 2] = distance
            rows[w, 9] = cx - law[3] * (law[9] + 0.5 * distance)
            rows[w, 10] = cy - law[4] * (law[9] + 0.5 * distance)
            rows[w, 11] = cz - law[5] * (law[9] + 0.5 * distance)
            if distance <= law[10]:
                rows[w, 0] = 1.0
                if contact_capacity < 1:
                    status[w] = 1
                position = distance - (law[10] - law[11])
                if position < 0.0:
                    rows[w, 1] = 1.0
                    if constraint_capacity < 1:
                        status[w] = 1
                    velocity = 0.0
                    free_acc = 0.0
                    for j in range(nv):
                        velocity = velocity + (vectors[5, j] * v[w, j])
                        free_acc = free_acc + (vectors[5, j] * smooth_acc[w, j])
                    dmin = tx.min(law[23], tx.max(law[22], law[16]))
                    dmax = tx.min(law[23], tx.max(law[22], law[17]))
                    width = tx.max(0.0, law[18])
                    midpoint = tx.min(law[23], tx.max(law[22], law[19]))
                    power = tx.max(1.0, law[20])
                    impedance = dmax
                    # CPU MuJoCo getimpedance also admits decreasing impedance
                    # and width=0 (the flat average); Warp 0.0.2 differs here.
                    if dmin == dmax or width <= law[21]:
                        impedance = 0.5 * (dmin + dmax)
                    else:
                        x = tx.abs(position) / width
                        if x < 1.0:
                            y = 0.0
                            if x < midpoint:
                                y = tx.pow(x, power) / tx.pow(midpoint, power - 1.0)
                            else:
                                y = 1.0 - tx.pow(1.0 - x, power) / tx.pow(
                                    1.0 - midpoint, power - 1.0
                                )
                            impedance = dmin + y * (dmax - dmin)
                    stiffness = 0.0
                    damping = 0.0
                    if law[14] > 0.0:
                        timeconst = tx.max(law[14], 2.0 * h)
                        stiffness = 1.0 / tx.max(
                            law[21], dmax * dmax * timeconst * timeconst * law[15] * law[15]
                        )
                        damping = 2.0 / tx.max(law[21], dmax * timeconst)
                    else:
                        stiffness = -law[14] / (dmax * dmax)
                        damping = -law[15] / dmax
                    regularization = tx.max(law[12] * (1.0 - impedance) / impedance, law[21])
                    aref = -stiffness * impedance * position - damping * velocity
                    normal_force = tx.max(0.0, (aref - free_acc) / (law[13] + regularization))
                    rows[w, 3] = normal_force
                    rows[w, 4] = regularization
                    rows[w, 5] = 1.0 / regularization
                    rows[w, 6] = aref
                    rows[w, 7] = velocity
                    rows[w, 8] = impedance
                    for j in range(nv):
                        constraint[w, j] = vectors[5, j] * normal_force
                        acceleration[w, j] = acceleration[w, j] + (vectors[6, j] * normal_force)
        for i in range(nv):
            integration_acc[w, i] = 0.0
            for j in range(nv):
                integration_acc[w, i] = integration_acc[w, i] + (
                    matrices[1, i, j] * acceleration[w, j]
                )
            vnext[w, i] = v[w, i] + h * integration_acc[w, i]
        if kind == 0:
            for i in range(nq):
                qnext[w, i] = q[w, i] + h * vnext[w, i]
                if qnext[w, i] <= vectors[7, i] or qnext[w, i] >= vectors[8, i]:
                    status[w] = 3
        else:
            for i in tx.static_range(3):
                qnext[w, i] = q[w, i] + h * vnext[w, i]
            qw = q[w, 3]
            qx = q[w, 4]
            qy = q[w, 5]
            qz = q[w, 6]
            length = tx.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            qw = qw / length
            qx = qx / length
            qy = qy / length
            qz = qz / length
            wx = vnext[w, 3]
            wy = vnext[w, 4]
            wz = vnext[w, 5]
            norm = tx.sqrt(wx * wx + wy * wy + wz * wz)
            scalar = 1.0
            scale = 0.0
            if norm > 0.0:
                scalar = tx.cos(0.5 * h * norm)
                scale = tx.sin(0.5 * h * norm) / norm
            dx = scale * wx
            dy = scale * wy
            dz = scale * wz
            nw = qw * scalar - qx * dx - qy * dy - qz * dz
            nx = qw * dx + qx * scalar + qy * dz - qz * dy
            ny = qw * dy - qx * dz + qy * scalar + qz * dx
            nz = qw * dz + qx * dy - qy * dx + qz * scalar
            length = tx.sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
            qnext[w, 3] = nw / length
            qnext[w, 4] = nx / length
            qnext[w, 5] = ny / length
            qnext[w, 6] = nz / length
        tnext[w] = clock[w] + h
        if not tx.isfinite(tnext[w]) or tnext[w] <= clock[w]:
            status[w] = 2
        for i in range(nq):
            if not tx.isfinite(qnext[w, i]):
                status[w] = 2
        for i in range(nv):
            if not tx.isfinite(vnext[w, i]) or not tx.isfinite(acceleration[w, i]):
                status[w] = 2
        for i in tx.static_range(12):
            if not tx.isfinite(rows[w, i]):
                status[w] = 2


advance_other = tx.kernel(advance)
advance = tx.kernel(advance)
