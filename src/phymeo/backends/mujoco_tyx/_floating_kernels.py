# Copyright 2025 The Newton Developers
# Copyright 2026 Phymeo contributors
# Licensed under Apache License 2.0; see LICENSE and NOTICE in this package.
# TYX world-coordinate rewrite of smooth.py kinematics/com_vel/crb/rne and
# passive.py scalar spring/damper semantics from mujoco-warp v0.0.2.

"""Every-step native free-root tree kinematics, inertia and bias on actual TYX operators."""

import tyx as tx


def geometry(
    count: tx.i32,
    nbody: tx.i32,
    nv: tx.i32,
    root: tx.i32,
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
        for d in range(6, nv):
            if q[w, d + 1] <= vectors[7, d] or q[w, d + 1] >= vectors[8, d]:
                status[w] = 3
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
                if b == root:
                    # Native free qpos is the body origin in world axes and WXYZ.
                    # Normalize a scratch copy; never overwrite the input state bank.
                    qw = q[w, 3]
                    qx = q[w, 4]
                    qy = q[w, 5]
                    qz = q[w, 6]
                    length = tx.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
                    if not tx.isfinite(length) or length <= 0.0:
                        status[w] = 2
                        length = 1.0
                    qw = qw / length
                    qx = qx / length
                    qy = qy / length
                    qz = qz / length
                    rotation[w, b, 0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
                    rotation[w, b, 0, 1] = 2.0 * (qx * qy - qw * qz)
                    rotation[w, b, 0, 2] = 2.0 * (qx * qz + qw * qy)
                    rotation[w, b, 1, 0] = 2.0 * (qx * qy + qw * qz)
                    rotation[w, b, 1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
                    rotation[w, b, 1, 2] = 2.0 * (qy * qz - qw * qx)
                    rotation[w, b, 2, 0] = 2.0 * (qx * qz - qw * qy)
                    rotation[w, b, 2, 1] = 2.0 * (qy * qz + qw * qx)
                    rotation[w, b, 2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)
                    for i in range(3):
                        motion[w, b, 0, i] = q[w, i]
                        motion[w, b, 1, i] = v[w, i]
                        motion[w, b, 2, i] = 0.0
                        motion[w, b, 3, i] = 0.0
                        motion[w, b, 4, i] = 0.0
                        for j in range(3):
                            motion[w, b, 2, i] = (
                                motion[w, b, 2, i] + rotation[w, b, i, j] * v[w, 3 + j]
                            )
                            axes[w, j, i] = 0.0
                            if i == j:
                                axes[w, j, i] = 1.0
                            axes[w, 3 + j, i] = rotation[w, b, i, j]
                            anchors[w, j, i] = q[w, i]
                            anchors[w, 3 + j, i] = q[w, i]
                    # qddot=0 => origin acceleration=0 and alpha=R(omega x omega)=0.
                    # Descendant and COM centripetal/transport terms remain below.
                start = body_ranges[b, 0]
                number = body_ranges[b, 1]
                for offset in range(number):
                    joint = start + offset
                    qa = joints[joint, 1]
                    dof = joints[joint, 2]
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
    joints: tx.ndarray(tx.i32, 2),
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
        for i in range(nv):
            bias[w, i] = 0.0
            passive[w, i] = 0.0
            applied[w, i] = 0.0
            actuator[w, i] = 0.0
            for j in range(nv):
                mass[w, i, j] = 0.0
        for i in range(nv):
            mass[w, i, i] = vectors[5, i]
            passive[w, i] = -vectors[1, i] * v[w, i]
            if i >= 6:
                passive[w, i] = passive[w, i] - vectors[2, i] * (q[w, i + 1] - vectors[3, i])
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
                        if joints[d, 0] == 1:
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
    h: tx.f32,
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
        for i in range(nv + 1):
            qnext[w, i] = q[w, i]
        for i in range(nv):
            vnext[w, i] = v[w, i]
            acceleration[w, i] = 0.0
            integration_acc[w, i] = 0.0
            constraint[w, i] = 0.0
            rhs[w, i] = applied[w, i] + passive[w, i] - bias[w, i]
        for i in range(20):
            rows[w, i] = 0.0
        for system in range(2):
            for i in range(nv):
                forward[w, i] = 0.0
                for j in range(nv):
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
            if i >= 6:
                qnext[w, i + 1] = q[w, i + 1] + h * vnext[w, i]
                if qnext[w, i + 1] <= vectors[7, i] or qnext[w, i + 1] >= vectors[8, i]:
                    status[w] = 3
            if (
                not tx.isfinite(vnext[w, i])
                or not tx.isfinite(acceleration[w, i])
                or not tx.isfinite(integration_acc[w, i])
            ):
                status[w] = 2
        for i in range(3):
            qnext[w, i] = q[w, i] + h * vnext[w, i]
        qw = q[w, 3]
        qx = q[w, 4]
        qy = q[w, 5]
        qz = q[w, 6]
        length = tx.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if not tx.isfinite(length) or length <= 0.0:
            status[w] = 2
            length = 1.0
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
        # Right multiplication: increment is in native root-local angular axes.
        nw = qw * scalar - qx * dx - qy * dy - qz * dz
        nx = qw * dx + qx * scalar + qy * dz - qz * dy
        ny = qw * dy - qx * dz + qy * scalar + qz * dx
        nz = qw * dz + qx * dy - qy * dx + qz * scalar
        length = tx.sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
        if not tx.isfinite(length) or length <= 0.0:
            status[w] = 2
            length = 1.0
        qnext[w, 3] = nw / length
        qnext[w, 4] = nx / length
        qnext[w, 5] = ny / length
        qnext[w, 6] = nz / length
        for i in range(nv + 1):
            if not tx.isfinite(qnext[w, i]):
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
