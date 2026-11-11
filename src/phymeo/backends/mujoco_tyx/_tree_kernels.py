# Copyright 2025 The Newton Developers
# Copyright 2026 Phymeo contributors
# Licensed under Apache License 2.0; see LICENSE and NOTICE in this package.
# TYX world-coordinate rewrite of smooth.py kinematics/com_vel/crb/rne and
# passive.py scalar spring/damper semantics from mujoco-warp v0.0.2.

"""Every-step fixed native-tree kinematics, inertia and bias on actual TYX operators."""

import tyx as tx


def geometry(
    count: tx.i32,
    nbody: tx.i32,
    nv: tx.i32,
    parents: tx.ndarray(tx.i32, 1),
    body_ranges: tx.ndarray(tx.i32, 2),
    body_data: tx.ndarray(tx.f32, 2),
    joints: tx.ndarray(tx.i32, 2),
    joint_data: tx.ndarray(tx.f32, 2),
    vectors: tx.ndarray(tx.f32, 2),
    q: tx.ndarray(tx.f32, 2),
    v: tx.ndarray(tx.f32, 2),
    motion: tx.ndarray(tx.f32, 4),
    rotation: tx.ndarray(tx.f32, 4),
    axes: tx.ndarray(tx.f32, 3),
    anchors: tx.ndarray(tx.f32, 3),
    work: tx.ndarray(tx.f32, 3),
    work_rotation: tx.ndarray(tx.f32, 3),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        status[w] = 0
        for d in range(nv):
            if not tx.isfinite(v[w, d]):
                status[w] = 2
        for b in range(nbody):
            for slot in range(7):
                for k in range(3):
                    motion[w, b, slot, k] = 0.0
            for i in range(3):
                for j in range(3):
                    rotation[w, b, i, j] = 0.0
                    if i == j:
                        rotation[w, b, i, j] = 1.0
            if b > 0:
                parent = parents[b]
                for i in range(3):
                    work[w, 0, i] = 0.0
                    for j in range(3):
                        work[w, 0, i] = work[w, 0, i] + rotation[w, parent, i, j] * body_data[b, j]
                        rotation[w, b, i, j] = 0.0
                        for k in range(3):
                            rotation[w, b, i, j] = (
                                rotation[w, b, i, j]
                                + rotation[w, parent, i, k] * body_data[b, 3 + 3 * k + j]
                            )
                    motion[w, b, 0, i] = motion[w, parent, 0, i] + work[w, 0, i]
                    motion[w, b, 2, i] = motion[w, parent, 2, i]
                    motion[w, b, 4, i] = motion[w, parent, 4, i]
                for i in range(3):
                    j = (i + 1) % 3
                    k = (i + 2) % 3
                    work[w, 1, i] = (
                        motion[w, parent, 2, j] * work[w, 0, k]
                        - motion[w, parent, 2, k] * work[w, 0, j]
                    )
                for i in range(3):
                    j = (i + 1) % 3
                    k = (i + 2) % 3
                    motion[w, b, 1, i] = motion[w, parent, 1, i] + work[w, 1, i]
                    motion[w, b, 3, i] = (
                        motion[w, parent, 3, i]
                        + motion[w, parent, 4, j] * work[w, 0, k]
                        - motion[w, parent, 4, k] * work[w, 0, j]
                        + motion[w, parent, 2, j] * work[w, 1, k]
                        - motion[w, parent, 2, k] * work[w, 1, j]
                    )
                start = body_ranges[b, 0]
                number = body_ranges[b, 1]
                for offset in range(number):
                    joint = start + offset
                    qa = joints[joint, 1]
                    dof = joints[joint, 2]
                    if joints[joint, 0] != 2:
                        if q[w, qa] <= vectors[7, dof] or q[w, qa] >= vectors[8, dof]:
                            status[w] = 3
                        if not tx.isfinite(q[w, qa]):
                            status[w] = 2
                    displacement = q[w, qa] - joint_data[joint, 6]
                    rate = v[w, dof]
                    for i in range(3):
                        axes[w, dof, i] = 0.0
                        work[w, 0, i] = 0.0
                        for j in range(3):
                            axes[w, dof, i] = (
                                axes[w, dof, i] + rotation[w, b, i, j] * joint_data[joint, j]
                            )
                            work[w, 0, i] = (
                                work[w, 0, i] + rotation[w, b, i, j] * joint_data[joint, 3 + j]
                            )
                        anchors[w, dof, i] = motion[w, b, 0, i] + work[w, 0, i]
                    if joints[joint, 0] == 0:
                        for i in range(3):
                            work[w, 0, i] = axes[w, dof, i] * displacement
                        for i in range(3):
                            j = (i + 1) % 3
                            k = (i + 2) % 3
                            work[w, 1, i] = (
                                motion[w, b, 2, j] * work[w, 0, k]
                                - motion[w, b, 2, k] * work[w, 0, j]
                            )
                        for i in range(3):
                            j = (i + 1) % 3
                            k = (i + 2) % 3
                            motion[w, b, 0, i] = motion[w, b, 0, i] + work[w, 0, i]
                            motion[w, b, 1, i] = (
                                motion[w, b, 1, i] + work[w, 1, i] + axes[w, dof, i] * rate
                            )
                            motion[w, b, 3, i] = (
                                motion[w, b, 3, i]
                                + motion[w, b, 4, j] * work[w, 0, k]
                                - motion[w, b, 4, k] * work[w, 0, j]
                                + motion[w, b, 2, j] * work[w, 1, k]
                                - motion[w, b, 2, k] * work[w, 1, j]
                                + 2.0
                                * rate
                                * (
                                    motion[w, b, 2, j] * axes[w, dof, k]
                                    - motion[w, b, 2, k] * axes[w, dof, j]
                                )
                            )
                    else:
                        # Old origin -> anchored point, then new body origin.
                        for i in range(3):
                            j = (i + 1) % 3
                            k = (i + 2) % 3
                            work[w, 6, i] = (
                                motion[w, b, 2, j] * work[w, 0, k]
                                - motion[w, b, 2, k] * work[w, 0, j]
                            )
                        for i in range(3):
                            j = (i + 1) % 3
                            k = (i + 2) % 3
                            work[w, 1, i] = motion[w, b, 1, i] + work[w, 6, i]
                            work[w, 2, i] = (
                                motion[w, b, 3, i]
                                + motion[w, b, 4, j] * work[w, 0, k]
                                - motion[w, b, 4, k] * work[w, 0, j]
                                + motion[w, b, 2, j] * work[w, 6, k]
                                - motion[w, b, 2, k] * work[w, 6, j]
                            )
                            work[w, 3, i] = motion[w, b, 2, i] + axes[w, dof, i] * rate
                            work[w, 4, i] = motion[w, b, 4, i] + rate * (
                                motion[w, b, 2, j] * axes[w, dof, k]
                                - motion[w, b, 2, k] * axes[w, dof, j]
                            )
                        if joints[joint, 0] == 2:
                            # Native ball qvel is local angular velocity. All three
                            # axes are taken from the same post-joint frame below.
                            qw = q[w, qa]
                            qx = q[w, qa + 1]
                            qy = q[w, qa + 2]
                            qz = q[w, qa + 3]
                            square = qw * qw + qx * qx + qy * qy + qz * qz
                            length = tx.sqrt(square)
                            if (
                                not tx.isfinite(length)
                                or length <= 0.0
                                or tx.abs(square - 1.0) > 0.00000095367431640625
                            ):
                                status[w] = 2
                                length = 1.0
                            qw = qw / length
                            qx = qx / length
                            qy = qy / length
                            qz = qz / length
                            r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
                            r01 = 2.0 * (qx * qy - qw * qz)
                            r02 = 2.0 * (qx * qz + qw * qy)
                            r10 = 2.0 * (qx * qy + qw * qz)
                            r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
                            r12 = 2.0 * (qy * qz - qw * qx)
                            r20 = 2.0 * (qx * qz - qw * qy)
                            r21 = 2.0 * (qy * qz + qw * qx)
                            r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
                            for i in range(3):
                                work_rotation[w, i, 0] = rotation[w, b, i, 0] * r00 + rotation[w, b, i, 1] * r10 + rotation[w, b, i, 2] * r20
                                work_rotation[w, i, 1] = rotation[w, b, i, 0] * r01 + rotation[w, b, i, 1] * r11 + rotation[w, b, i, 2] * r21
                                work_rotation[w, i, 2] = rotation[w, b, i, 0] * r02 + rotation[w, b, i, 1] * r12 + rotation[w, b, i, 2] * r22
                        else:
                            cosine = tx.cos(displacement)
                            sine = tx.sin(displacement)
                            for i in range(3):
                                for j in range(3):
                                    work_rotation[w, i, j] = 0.0
                                    for k in range(3):
                                        value = (
                                            (1.0 - cosine) * joint_data[joint, k] * joint_data[joint, j]
                                        )
                                        if k == j:
                                            value = value + cosine
                                        else:
                                            axis = 3 - k - j
                                            if (k + 1) % 3 == j:
                                                value = value - sine * joint_data[joint, axis]
                                            else:
                                                value = value + sine * joint_data[joint, axis]
                                        work_rotation[w, i, j] = (
                                            work_rotation[w, i, j] + rotation[w, b, i, k] * value
                                        )
                        for i in range(3):
                            work[w, 5, i] = 0.0
                            for j in range(3):
                                rotation[w, b, i, j] = work_rotation[w, i, j]
                                work[w, 5, i] = (
                                    work[w, 5, i]
                                    - work_rotation[w, i, j] * joint_data[joint, 3 + j]
                                )
                        if joints[joint, 0] == 2:
                            for i in range(3):
                                work[w, 7, i] = 0.0
                                for column in range(3):
                                    axes[w, dof + column, i] = rotation[w, b, i, column]
                                    anchors[w, dof + column, i] = anchors[w, dof, i]
                                    work[w, 7, i] = work[w, 7, i] + rotation[w, b, i, column] * v[w, dof + column]
                            for i in range(3):
                                j = (i + 1) % 3
                                k = (i + 2) % 3
                                work[w, 3, i] = motion[w, b, 2, i] + work[w, 7, i]
                                work[w, 4, i] = motion[w, b, 4, i] + motion[w, b, 2, j] * work[w, 7, k] - motion[w, b, 2, k] * work[w, 7, j]
                        for i in range(3):
                            j = (i + 1) % 3
                            k = (i + 2) % 3
                            work[w, 6, i] = (
                                work[w, 3, j] * work[w, 5, k] - work[w, 3, k] * work[w, 5, j]
                            )
                        for i in range(3):
                            j = (i + 1) % 3
                            k = (i + 2) % 3
                            motion[w, b, 0, i] = anchors[w, dof, i] + work[w, 5, i]
                            motion[w, b, 1, i] = work[w, 1, i] + work[w, 6, i]
                            motion[w, b, 3, i] = (
                                work[w, 2, i]
                                + work[w, 4, j] * work[w, 5, k]
                                - work[w, 4, k] * work[w, 5, j]
                                + work[w, 3, j] * work[w, 6, k]
                                - work[w, 3, k] * work[w, 6, j]
                            )
                            motion[w, b, 2, i] = work[w, 3, i]
                            motion[w, b, 4, i] = work[w, 4, i]
        for b in range(nbody):
            for i in range(3):
                work[w, 0, i] = 0.0
                for j in range(3):
                    work[w, 0, i] = work[w, 0, i] + rotation[w, b, i, j] * body_data[b, 12 + j]
                motion[w, b, 5, i] = motion[w, b, 0, i] + work[w, 0, i]
            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3
                work[w, 1, i] = (
                    motion[w, b, 2, j] * work[w, 0, k] - motion[w, b, 2, k] * work[w, 0, j]
                )
            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3
                motion[w, b, 6, i] = (
                    motion[w, b, 3, i]
                    + motion[w, b, 4, j] * work[w, 0, k]
                    - motion[w, b, 4, k] * work[w, 0, j]
                    + motion[w, b, 2, j] * work[w, 1, k]
                    - motion[w, b, 2, k] * work[w, 1, j]
                )
                if not tx.isfinite(motion[w, b, 5, i]) or not tx.isfinite(motion[w, b, 6, i]):
                    status[w] = 2


geometry_other = tx.kernel(geometry)
geometry = tx.kernel(geometry)


def mechanics(
    count: tx.i32,
    nbody: tx.i32,
    nv: tx.i32,
    nu: tx.i32,
    body_data: tx.ndarray(tx.f32, 2),
    dofs: tx.ndarray(tx.i32, 2),
    ancestors: tx.ndarray(tx.i32, 2),
    vectors: tx.ndarray(tx.f32, 2),
    gravity: tx.ndarray(tx.f32, 1),
    act_ids: tx.ndarray(tx.i32, 2),
    act_values: tx.ndarray(tx.f32, 2),
    controls: tx.ndarray(tx.f32, 2),
    applied_input: tx.ndarray(tx.f32, 2),
    body_force: tx.ndarray(tx.f32, 3),
    q: tx.ndarray(tx.f32, 2),
    v: tx.ndarray(tx.f32, 2),
    motion: tx.ndarray(tx.f32, 4),
    rotation: tx.ndarray(tx.f32, 4),
    axes: tx.ndarray(tx.f32, 3),
    anchors: tx.ndarray(tx.f32, 3),
    inertia: tx.ndarray(tx.f32, 4),
    jacobian: tx.ndarray(tx.f32, 4),
    mass: tx.ndarray(tx.f32, 3),
    bias: tx.ndarray(tx.f32, 2),
    passive: tx.ndarray(tx.f32, 2),
    applied: tx.ndarray(tx.f32, 2),
    actuator: tx.ndarray(tx.f32, 2),
    work: tx.ndarray(tx.f32, 3),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        for i in range(8):
            bias[w, i] = 0.0
            passive[w, i] = 0.0
            applied[w, i] = 0.0
            actuator[w, i] = 0.0
            for j in range(8):
                mass[w, i, j] = 0.0
        for i in range(nv):
            mass[w, i, i] = vectors[5, i]
            passive[w, i] = -vectors[1, i] * v[w, i]
            qa = dofs[i, 1]
            if qa >= 0:
                passive[w, i] = passive[w, i] - vectors[2, i] * (q[w, qa] - vectors[3, i])
            applied[w, i] = applied_input[w, i]
        for a in range(nu):
            value = 0.0
            qa = act_ids[a, 0]
            dof = act_ids[a, 1]
            gear = act_values[a, 0]
            if act_ids[a, 2] == 0:
                ctrl = tx.min(act_values[a, 6], tx.max(act_values[a, 5], controls[w, a]))
                value = (
                    act_values[a, 1] * ctrl
                    + act_values[a, 2]
                    + act_values[a, 3] * gear * q[w, qa]
                    + act_values[a, 4] * gear * v[w, dof]
                )
            # CPU 3.3.7 clamps all actuator forces, including disabled-group zero.
            value = tx.min(act_values[a, 8], tx.max(act_values[a, 7], value))
            actuator[w, dof] = actuator[w, dof] + gear * value
        for b in range(nbody):
            for i in range(3):
                for j in range(3):
                    inertia[w, b, i, j] = 0.0
                    for r in range(3):
                        for s in range(3):
                            inertia[w, b, i, j] = (
                                inertia[w, b, i, j]
                                + rotation[w, b, i, r]
                                * body_data[b, 16 + 3 * r + s]
                                * rotation[w, b, j, s]
                            )
            for i in range(3):
                work[w, 0, i] = 0.0
                work[w, 1, i] = 0.0
                for j in range(3):
                    work[w, 0, i] = work[w, 0, i] + inertia[w, b, i, j] * motion[w, b, 2, j]
                    work[w, 1, i] = work[w, 1, i] + inertia[w, b, i, j] * motion[w, b, 4, j]
            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3
                work[w, 2, i] = (
                    work[w, 1, i]
                    + motion[w, b, 2, j] * work[w, 0, k]
                    - motion[w, b, 2, k] * work[w, 0, j]
                )
            for d in range(nv):
                for i in range(6):
                    jacobian[w, b, i, d] = 0.0
                if ancestors[b, d] == 1:
                    for i in range(3):
                        j = (i + 1) % 3
                        k = (i + 2) % 3
                        if dofs[d, 0] == 1:
                            jacobian[w, b, i, d] = axes[w, d, j] * (
                                motion[w, b, 5, k] - anchors[w, d, k]
                            ) - axes[w, d, k] * (motion[w, b, 5, j] - anchors[w, d, j])
                            jacobian[w, b, 3 + i, d] = axes[w, d, i]
                        else:
                            jacobian[w, b, i, d] = axes[w, d, i]
                for i in range(3):
                    bias[w, d] = (
                        bias[w, d]
                        + jacobian[w, b, i, d]
                        * body_data[b, 15]
                        * (motion[w, b, 6, i] - gravity[i])
                        + jacobian[w, b, 3 + i, d] * work[w, 2, i]
                    )
                    applied[w, d] = (
                        applied[w, d]
                        + jacobian[w, b, i, d] * body_force[w, b, i]
                        + jacobian[w, b, 3 + i, d] * body_force[w, b, 3 + i]
                    )
            for i in range(nv):
                for j in range(i + 1):
                    value = 0.0
                    for k in range(3):
                        value = (
                            value + body_data[b, 15] * jacobian[w, b, k, i] * jacobian[w, b, k, j]
                        )
                        for r in range(3):
                            value = (
                                value
                                + jacobian[w, b, 3 + k, i]
                                * inertia[w, b, k, r]
                                * jacobian[w, b, 3 + r, j]
                            )
                    mass[w, i, j] = mass[w, i, j] + value
                    mass[w, j, i] = mass[w, i, j]
        for i in range(nv):
            applied[w, i] = applied[w, i] + actuator[w, i]
            if (
                not tx.isfinite(bias[w, i])
                or not tx.isfinite(passive[w, i])
                or not tx.isfinite(applied[w, i])
            ):
                status[w] = 2
            for j in range(nv):
                if not tx.isfinite(mass[w, i, j]):
                    status[w] = 2


mechanics_other = tx.kernel(mechanics)
mechanics = tx.kernel(mechanics)


def solve(
    count: tx.i32,
    nv: tx.i32,
    nq: tx.i32,
    njnt: tx.i32,
    h: tx.f32,
    joints: tx.ndarray(tx.i32, 2),
    vectors: tx.ndarray(tx.f32, 2),
    mass: tx.ndarray(tx.f32, 3),
    bias: tx.ndarray(tx.f32, 2),
    passive: tx.ndarray(tx.f32, 2),
    applied: tx.ndarray(tx.f32, 2),
    q: tx.ndarray(tx.f32, 2),
    v: tx.ndarray(tx.f32, 2),
    clock: tx.ndarray(tx.f32, 2),
    qnext: tx.ndarray(tx.f32, 2),
    vnext: tx.ndarray(tx.f32, 2),
    tnext: tx.ndarray(tx.f32, 2),
    factor: tx.ndarray(tx.f32, 3),
    rhs: tx.ndarray(tx.f32, 2),
    forward: tx.ndarray(tx.f32, 2),
    acceleration: tx.ndarray(tx.f32, 2),
    integration_acc: tx.ndarray(tx.f32, 2),
    constraint: tx.ndarray(tx.f32, 2),
    rows: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        for i in range(8):
            qnext[w, i] = q[w, i]
            vnext[w, i] = v[w, i]
            acceleration[w, i] = 0.0
            integration_acc[w, i] = 0.0
            constraint[w, i] = 0.0
            rhs[w, i] = applied[w, i] + passive[w, i] - bias[w, i]
        for i in range(8, nq):
            qnext[w, i] = q[w, i]
        for i in range(20):
            rows[w, i] = 0.0
        for system in range(2):
            for i in range(8):
                forward[w, i] = 0.0
                for j in range(8):
                    factor[w, i, j] = 0.0
            for i in range(nv):
                for j in range(i + 1):
                    value = mass[w, i, j]
                    if system == 1 and i == j:
                        value = value + h * vectors[1, i]
                    for k in range(j):
                        value = value - factor[w, i, k] * factor[w, j, k]
                    if i == j:
                        if not tx.isfinite(value) or value <= 0.0:
                            status[w] = 4
                            factor[w, i, j] = 1.0
                        else:
                            factor[w, i, j] = tx.sqrt(value)
                    else:
                        factor[w, i, j] = value / factor[w, j, j]
            for i in range(nv):
                value = rhs[w, i]
                for j in range(i):
                    value = value - factor[w, i, j] * forward[w, j]
                forward[w, i] = value / factor[w, i, i]
            for reverse in range(nv):
                i = nv - 1 - reverse
                value = forward[w, i]
                for j in range(i + 1, nv):
                    if system == 0:
                        value = value - factor[w, j, i] * acceleration[w, j]
                    else:
                        value = value - factor[w, j, i] * integration_acc[w, j]
                if system == 0:
                    acceleration[w, i] = value / factor[w, i, i]
                else:
                    integration_acc[w, i] = value / factor[w, i, i]
        for i in range(nv):
            vnext[w, i] = v[w, i] + h * integration_acc[w, i]
            if (
                not tx.isfinite(vnext[w, i])
                or not tx.isfinite(acceleration[w, i])
                or not tx.isfinite(integration_acc[w, i])
            ):
                status[w] = 2
        for joint in range(njnt):
            qa = joints[joint, 1]
            dof = joints[joint, 2]
            if joints[joint, 0] != 2:
                qnext[w, qa] = q[w, qa] + h * vnext[w, dof]
                if qnext[w, qa] <= vectors[7, dof] or qnext[w, qa] >= vectors[8, dof]:
                    status[w] = 3
                if not tx.isfinite(qnext[w, qa]):
                    status[w] = 2
            else:
                # Native local angular velocity integrates by right multiplication.
                qw = q[w, qa]
                qx = q[w, qa + 1]
                qy = q[w, qa + 2]
                qz = q[w, qa + 3]
                length = tx.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
                if not tx.isfinite(length) or length <= 0.0:
                    status[w] = 2
                    length = 1.0
                qw = qw / length
                qx = qx / length
                qy = qy / length
                qz = qz / length
                wx = vnext[w, dof]
                wy = vnext[w, dof + 1]
                wz = vnext[w, dof + 2]
                speed = tx.sqrt(wx * wx + wy * wy + wz * wz)
                dw = 1.0
                scale = 0.0
                if not tx.isfinite(speed):
                    status[w] = 2
                elif speed > 0.0:
                    angle = 0.5 * h * speed
                    dw = tx.cos(angle)
                    scale = tx.sin(angle) / speed
                dx = scale * wx
                dy = scale * wy
                dz = scale * wz
                ow = qw * dw - qx * dx - qy * dy - qz * dz
                ox = qw * dx + qx * dw + qy * dz - qz * dy
                oy = qw * dy - qx * dz + qy * dw + qz * dx
                oz = qw * dz + qx * dy - qy * dx + qz * dw
                out_length = tx.sqrt(ow * ow + ox * ox + oy * oy + oz * oz)
                if not tx.isfinite(out_length) or out_length <= 0.0:
                    status[w] = 2
                    out_length = 1.0
                qnext[w, qa] = ow / out_length
                qnext[w, qa + 1] = ox / out_length
                qnext[w, qa + 2] = oy / out_length
                qnext[w, qa + 3] = oz / out_length
                for component in range(4):
                    if not tx.isfinite(qnext[w, qa + component]):
                        status[w] = 2
        # Kahan clock: keep the rounding compensation in the transaction buffer.
        increment = h - clock[w, 1]
        total = clock[w, 0] + increment
        compensation = (total - clock[w, 0]) - increment
        tnext[w, 0] = total
        tnext[w, 1] = compensation
        if not tx.isfinite(total) or not tx.isfinite(compensation) or total <= clock[w, 0]:
            status[w] = 2


solve_other = tx.kernel(solve)
solve = tx.kernel(solve)
