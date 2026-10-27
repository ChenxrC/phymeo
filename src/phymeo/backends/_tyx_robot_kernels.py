"""Ordered world-tree mechanics, independently expressed in public tyx kernels."""

import tyx as tx

from ._tyx_numeric import _add, _mul


@tx.kernel
def robot_geometry(
    n: tx.i32,
    parents: tx.ndarray(tx.i32, 1),
    kinds: tx.ndarray(tx.i32, 1),
    offsets: tx.ndarray(tx.f32, 2),
    zero: tx.ndarray(tx.f32, 3),
    local_axis: tx.ndarray(tx.f32, 2),
    com: tx.ndarray(tx.f32, 2),
    q: tx.ndarray(tx.f32, 1),
    position: tx.ndarray(tx.f32, 2),
    rotation: tx.ndarray(tx.f32, 3),
    axis: tx.ndarray(tx.f32, 2),
    com_position: tx.ndarray(tx.f32, 2),
    frame: tx.ndarray(tx.f32, 3),
):
    for lane in tx.range(1):
        for i in range(n):
            parent = parents[i]
            for a in tx.static_range(3):
                coordinate = 0.0
                if parent >= 0:
                    coordinate = position[parent, a]
                for b in tx.static_range(3):
                    pr = 0.0
                    if parent >= 0:
                        pr = rotation[parent, a, b]
                    elif a == b:
                        pr = 1.0
                    coordinate = _add(coordinate, _mul(pr, offsets[i, b]))
                position[i, a] = coordinate
                for b in tx.static_range(3):
                    value = 0.0
                    for k in tx.static_range(3):
                        pr = 0.0
                        if parent >= 0:
                            pr = rotation[parent, a, k]
                        elif a == k:
                            pr = 1.0
                        value = _add(value, _mul(pr, zero[i, k, b]))
                    frame[i, a, b] = value
                value = 0.0
                for k in tx.static_range(3):
                    value = _add(value, _mul(frame[i, a, k], local_axis[i, k]))
                axis[i, a] = value
            cosine = tx.cos(q[i])
            sine = tx.sin(q[i])
            for a in tx.static_range(3):
                if kinds[i] == 1:
                    position[i, a] = _add(position[i, a], _mul(axis[i, a], q[i]))
                for b in tx.static_range(3):
                    value = frame[i, a, b]
                    if kinds[i] == 0:
                        value = 0.0
                        for k in tx.static_range(3):
                            turn = _mul(
                                _add(1.0, -cosine), _mul(local_axis[i, k], local_axis[i, b])
                            )
                            if k == b:
                                turn = _add(turn, cosine)
                            elif (k + 1) % 3 == b:
                                turn = _add(turn, -_mul(sine, local_axis[i, (k + 2) % 3]))
                            else:
                                turn = _add(turn, _mul(sine, local_axis[i, (k + 1) % 3]))
                            value = _add(value, _mul(frame[i, a, k], turn))
                    rotation[i, a, b] = value
                value = position[i, a]
                for k in tx.static_range(3):
                    value = _add(value, _mul(rotation[i, a, k], com[i, k]))
                com_position[i, a] = value


@tx.kernel
def robot_jacobians(
    n: tx.i32,
    count: tx.i32,
    parents: tx.ndarray(tx.i32, 1),
    kinds: tx.ndarray(tx.i32, 1),
    position: tx.ndarray(tx.f32, 2),
    rotation: tx.ndarray(tx.f32, 3),
    axis: tx.ndarray(tx.f32, 2),
    indices: tx.ndarray(tx.i32, 1),
    local: tx.ndarray(tx.f32, 2),
    linear: tx.ndarray(tx.f32, 3),
    angular: tx.ndarray(tx.f32, 3),
):
    for row in tx.range(count):
        link = indices[row]
        ancestor = link
        for depth in range(n):
            if ancestor >= 0:
                for a in tx.static_range(3):
                    value = axis[ancestor, a]
                    if kinds[ancestor] == 0:
                        j = (a + 1) % 3
                        k = (a + 2) % 3
                        rj = _add(position[link, j], -position[ancestor, j])
                        rk = _add(position[link, k], -position[ancestor, k])
                        for b in tx.static_range(3):
                            rj = _add(rj, _mul(rotation[link, j, b], local[row, b]))
                            rk = _add(rk, _mul(rotation[link, k, b], local[row, b]))
                        value = _add(_mul(axis[ancestor, j], rk), -_mul(axis[ancestor, k], rj))
                        angular[row, a, ancestor] = axis[ancestor, a]
                    linear[row, a, ancestor] = value
                ancestor = parents[ancestor]


@tx.kernel
def robot_mass(
    n: tx.i32,
    mass: tx.ndarray(tx.f32, 1),
    inertia: tx.ndarray(tx.f32, 2),
    rotation: tx.ndarray(tx.f32, 3),
    linear: tx.ndarray(tx.f32, 3),
    angular: tx.ndarray(tx.f32, 3),
    output: tx.ndarray(tx.f32, 2),
):
    for a in tx.range(n):
        for b in range(n):
            value = 0.0
            for link in range(n):
                for k in tx.static_range(3):
                    value = _add(
                        value, _mul(mass[link], _mul(linear[link, k, a], linear[link, k, b]))
                    )
                    local_a = 0.0
                    local_b = 0.0
                    for j in tx.static_range(3):
                        local_a = _add(local_a, _mul(rotation[link, j, k], angular[link, j, a]))
                        local_b = _add(local_b, _mul(rotation[link, j, k], angular[link, j, b]))
                    value = _add(value, _mul(inertia[link, k], _mul(local_a, local_b)))
            output[a, b] = value


@tx.kernel
def robot_inertial_loads(
    n: tx.i32,
    parents: tx.ndarray(tx.i32, 1),
    kinds: tx.ndarray(tx.i32, 1),
    position: tx.ndarray(tx.f32, 2),
    rotation: tx.ndarray(tx.f32, 3),
    axis: tx.ndarray(tx.f32, 2),
    com: tx.ndarray(tx.f32, 2),
    mass: tx.ndarray(tx.f32, 1),
    inertia: tx.ndarray(tx.f32, 2),
    gravity: tx.ndarray(tx.f32, 1),
    velocity: tx.ndarray(tx.f32, 1),
    omega: tx.ndarray(tx.f32, 2),
    alpha: tx.ndarray(tx.f32, 2),
    origin_acc: tx.ndarray(tx.f32, 2),
    momentum: tx.ndarray(tx.f32, 2),
    force: tx.ndarray(tx.f32, 2),
    torque: tx.ndarray(tx.f32, 2),
):
    for lane in tx.range(1):
        for i in range(n):
            parent = parents[i]
            dot_wr = 0.0
            square_w = 0.0
            if parent >= 0:
                for a in tx.static_range(3):
                    arm = _add(position[i, a], -position[parent, a])
                    dot_wr = _add(dot_wr, _mul(omega[parent, a], arm))
                    square_w = _add(square_w, _mul(omega[parent, a], omega[parent, a]))
            for a in tx.static_range(3):
                j = (a + 1) % 3
                k = (a + 2) % 3
                wa = 0.0
                wj = 0.0
                wk = 0.0
                aa = 0.0
                aj = 0.0
                ak = 0.0
                oa = 0.0
                ra = position[i, a]
                rj = position[i, j]
                rk = position[i, k]
                if parent >= 0:
                    wa = omega[parent, a]
                    wj = omega[parent, j]
                    wk = omega[parent, k]
                    aa = alpha[parent, a]
                    aj = alpha[parent, j]
                    ak = alpha[parent, k]
                    oa = origin_acc[parent, a]
                    ra = _add(ra, -position[parent, a])
                    rj = _add(rj, -position[parent, j])
                    rk = _add(rk, -position[parent, k])
                cross_axis = _mul(_add(_mul(wj, axis[i, k]), -_mul(wk, axis[i, j])), velocity[i])
                omega[i, a] = wa
                alpha[i, a] = aa
                oa = _add(oa, _add(_mul(aj, rk), -_mul(ak, rj)))
                oa = _add(oa, _add(_mul(wa, dot_wr), -_mul(ra, square_w)))
                if kinds[i] == 0:
                    omega[i, a] = _add(wa, _mul(axis[i, a], velocity[i]))
                    alpha[i, a] = _add(aa, cross_axis)
                else:
                    oa = _add(oa, _mul(2.0, cross_axis))
                origin_acc[i, a] = oa
            dot_wc = 0.0
            square_w = 0.0
            for a in tx.static_range(3):
                ca = 0.0
                for b in tx.static_range(3):
                    ca = _add(ca, _mul(rotation[i, a, b], com[i, b]))
                dot_wc = _add(dot_wc, _mul(omega[i, a], ca))
                square_w = _add(square_w, _mul(omega[i, a], omega[i, a]))
                la = 0.0
                ia = 0.0
                for b in tx.static_range(3):
                    local_w = 0.0
                    local_a = 0.0
                    for k in tx.static_range(3):
                        local_w = _add(local_w, _mul(rotation[i, k, b], omega[i, k]))
                        local_a = _add(local_a, _mul(rotation[i, k, b], alpha[i, k]))
                    la = _add(la, _mul(rotation[i, a, b], _mul(inertia[i, b], local_w)))
                    ia = _add(ia, _mul(rotation[i, a, b], _mul(inertia[i, b], local_a)))
                momentum[i, a] = la
                torque[i, a] = ia
            for a in tx.static_range(3):
                j = (a + 1) % 3
                k = (a + 2) % 3
                ca = 0.0
                cj = 0.0
                ck = 0.0
                for b in tx.static_range(3):
                    ca = _add(ca, _mul(rotation[i, a, b], com[i, b]))
                    cj = _add(cj, _mul(rotation[i, j, b], com[i, b]))
                    ck = _add(ck, _mul(rotation[i, k, b], com[i, b]))
                acceleration = _add(
                    origin_acc[i, a], _add(_mul(alpha[i, j], ck), -_mul(alpha[i, k], cj))
                )
                acceleration = _add(
                    acceleration, _add(_mul(omega[i, a], dot_wc), -_mul(ca, square_w))
                )
                force[i, a] = _mul(mass[i], _add(acceleration, -gravity[a]))
                torque[i, a] = _add(
                    torque[i, a],
                    _add(_mul(omega[i, j], momentum[i, k]), -_mul(omega[i, k], momentum[i, j])),
                )


@tx.kernel
def robot_project_loads(
    n: tx.i32,
    linear: tx.ndarray(tx.f32, 3),
    angular: tx.ndarray(tx.f32, 3),
    effort: tx.ndarray(tx.f32, 1),
    force: tx.ndarray(tx.f32, 2),
    torque: tx.ndarray(tx.f32, 2),
    out: tx.ndarray(tx.f32, 1),
):
    for column in tx.range(n):
        value = effort[column]
        for link in range(n):
            for a in tx.static_range(3):
                value = _add(value, _mul(linear[link, a, column], force[link, a]))
                value = _add(value, _mul(angular[link, a, column], torque[link, a]))
        out[column] = value
