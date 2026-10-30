"""Actual ordered FP32 reference-displacement material operations."""

import tyx as tx

from ._tyx_numeric import _add, _div, _mul


@tx.kernel
def project_bending_represented(
    count: tx.i32,
    hinges: tx.ndarray(tx.i32, 2),
    rest: tx.ndarray(tx.f32, 1),
    alpha: tx.ndarray(tx.f32, 1),
    weights: tx.ndarray(tx.f32, 1),
    x: tx.ndarray(tx.f32, 2),
    multipliers: tx.ndarray(tx.f32, 1),
    trace: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
    failed: tx.ndarray(tx.i32, 1),
    work: tx.ndarray(tx.f32, 2),
    reference_vectors: tx.ndarray(tx.f32, 3),
    products: tx.ndarray(tx.f32, 3),
    geometry_before: tx.ndarray(tx.f32, 3),
    geometry_after: tx.ndarray(tx.f32, 3),
):
    for lane in tx.range(1):
        for k in range(count):
            if failed[0] == 0:
                a = hinges[k, 0]
                b = hinges[k, 1]
                c = hinges[k, 2]
                d = hinges[k, 3]
                trace[k, 40] = multipliers[k]
                trace[k, 43] = alpha[k]
                for phase in range(2):
                    if failed[0] == 0:
                        left_scale = 0.0
                        right_scale = 0.0
                        for axis in tx.static_range(3):
                            e = _add(reference_vectors[k, 0, axis], _add(x[b, axis], -x[a, axis]))
                            u = _add(reference_vectors[k, 1, axis], _add(x[c, axis], -x[a, axis]))
                            v = _add(reference_vectors[k, 2, axis], _add(x[d, axis], -x[b, axis]))
                            if phase == 0:
                                geometry_before[k, 0, axis] = e
                                geometry_before[k, 1, axis] = u
                                geometry_before[k, 2, axis] = v
                            else:
                                geometry_after[k, 0, axis] = e
                                geometry_after[k, 1, axis] = u
                                geometry_after[k, 2, axis] = v
                            work[0, axis] = e
                            work[1, axis] = u
                            work[2, axis] = -e
                            work[3, axis] = v
                            left_scale = tx.max(left_scale, tx.max(tx.abs(e), tx.abs(u)))
                            right_scale = tx.max(right_scale, tx.max(tx.abs(e), tx.abs(v)))
                        if left_scale <= 0.0 or right_scale <= 0.0:
                            failed[0] = k + 1
                            status[k] = 1 + phase
                        else:
                            for axis in tx.static_range(3):
                                work[0, axis] = _div(work[0, axis], left_scale)
                                work[1, axis] = _div(work[1, axis], left_scale)
                                work[2, axis] = _div(work[2, axis], right_scale)
                                work[3, axis] = _div(work[3, axis], right_scale)
                            l2 = 0.0
                            r2 = 0.0
                            e2 = 0.0
                            er2 = 0.0
                            eu = 0.0
                            ev = 0.0
                            for axis in tx.static_range(3):
                                j = (axis + 1) % 3
                                t = (axis + 2) % 3
                                nl = _add(
                                    _mul(work[0, j], work[1, t]), -_mul(work[0, t], work[1, j])
                                )
                                nr = _add(
                                    _mul(work[2, j], work[3, t]), -_mul(work[2, t], work[3, j])
                                )
                                work[4, axis] = nl
                                work[5, axis] = nr
                                l2 = _add(l2, _mul(nl, nl))
                                r2 = _add(r2, _mul(nr, nr))
                                e2 = _add(e2, _mul(work[0, axis], work[0, axis]))
                                er2 = _add(er2, _mul(work[2, axis], work[2, axis]))
                                eu = _add(eu, _mul(work[0, axis], work[1, axis]))
                                ev = _add(ev, -_mul(work[2, axis], work[3, axis]))
                            la = tx.sqrt(l2)
                            ra = tx.sqrt(r2)
                            edge_unit = tx.sqrt(e2)
                            if la <= 1.0e-10 or ra <= 1.0e-10 or edge_unit <= 0.0:
                                failed[0] = k + 1
                                status[k] = 1 + phase
                            else:
                                edge = _mul(left_scale, edge_unit)
                                for axis in tx.static_range(3):
                                    work[4, axis] = _div(work[4, axis], la)
                                    work[5, axis] = _div(work[5, axis], ra)
                                sine = 0.0
                                cosine = 0.0
                                for axis in tx.static_range(3):
                                    j = (axis + 1) % 3
                                    t = (axis + 2) % 3
                                    component = _add(
                                        _mul(work[4, j], work[5, t]), -_mul(work[4, t], work[5, j])
                                    )
                                    sine = _add(
                                        sine, _mul(_div(work[0, axis], edge_unit), component)
                                    )
                                    cosine = _add(cosine, _mul(work[4, axis], work[5, axis]))
                                theta = tx.atan2(sine, cosine)
                                constraint = _add(theta, -rest[k])
                                trace[k, 36 + phase] = theta
                                trace[k, 38 + phase] = constraint
                                if (
                                    tx.abs(theta) >= 3.141592653589793
                                    or tx.abs(constraint) >= 3.141592653589793
                                ):
                                    failed[0] = k + 1
                                    status[k] = 3 + phase
                                elif phase == 0:
                                    tc = _div(eu, e2)
                                    td = _add(1.0, _div(ev, er2))
                                    denominator = alpha[k]
                                    for axis in tx.static_range(3):
                                        gc = -_mul(
                                            _div(_div(edge, left_scale), la),
                                            _div(work[4, axis], left_scale),
                                        )
                                        gd = -_mul(
                                            _div(_div(edge, right_scale), ra),
                                            _div(work[5, axis], right_scale),
                                        )
                                        ga = _add(
                                            _mul(_add(tc, -1.0), gc), _mul(_add(td, -1.0), gd)
                                        )
                                        gb = _add(-_mul(tc, gc), -_mul(td, gd))
                                        trace[k, 24 + axis] = ga
                                        trace[k, 27 + axis] = gb
                                        trace[k, 30 + axis] = gc
                                        trace[k, 33 + axis] = gd
                                    for node in tx.static_range(4):
                                        index = hinges[k, node]
                                        for axis in tx.static_range(3):
                                            g = trace[k, 24 + 3 * node + axis]
                                            denominator = _add(
                                                denominator, _mul(weights[index], _mul(g, g))
                                            )
                                            trace[k, 3 * node + axis] = x[index, axis]
                                    trace[k, 44] = denominator
                                    if denominator <= 0.0:
                                        failed[0] = k + 1
                                        status[k] = 5
                                    else:
                                        delta = _div(
                                            -_add(constraint, _mul(alpha[k], multipliers[k])),
                                            denominator,
                                        )
                                        trace[k, 42] = delta
                                        multipliers[k] = _add(multipliers[k], delta)
                                        trace[k, 41] = multipliers[k]
                                        for node in tx.static_range(4):
                                            index = hinges[k, node]
                                            for axis in tx.static_range(3):
                                                g = trace[k, 24 + 3 * node + axis]
                                                products[k, node, axis] = _mul(_mul(weights[index], delta), g)
                                                x[index, axis] = _add(
                                                    x[index, axis],
                                                    products[k, node, axis],
                                                )
                                                trace[k, 12 + 3 * node + axis] = x[index, axis]
                                elif tx.abs(_add(theta, -trace[k, 36])) >= 3.141592653589793:
                                    failed[0] = k + 1
                                    status[k] = 6

                for field in tx.static_range(45):
                    if trace[k, field] == 0.0:
                        trace[k, field] = 0.0
                if multipliers[k] == 0.0:
                    multipliers[k] = 0.0
                for node in tx.static_range(4):
                    for axis in tx.static_range(3):
                        if x[hinges[k, node], axis] == 0.0:
                            x[hinges[k, node], axis] = 0.0
                        if products[k, node, axis] == 0.0:
                            products[k, node, axis] = 0.0
                for vector in tx.static_range(3):
                    for axis in tx.static_range(3):
                        if geometry_before[k, vector, axis] == 0.0:
                            geometry_before[k, vector, axis] = 0.0
                        if geometry_after[k, vector, axis] == 0.0:
                            geometry_after[k, vector, axis] = 0.0

@tx.kernel
def predict_represented(
    count: tx.i32,
    h: tx.f32,
    velocity: tx.ndarray(tx.f32, 2),
    force: tx.ndarray(tx.f32, 2),
    weights: tx.ndarray(tx.f32, 1),
    gravity: tx.ndarray(tx.f32, 1),
    displacement: tx.ndarray(tx.f32, 2),
    predicted_velocity: tx.ndarray(tx.f32, 2),
):
    for i in tx.range(count):
        for c in tx.static_range(3):
            acceleration = _add(gravity[c], _mul(weights[i], force[i, c]))
            predicted_velocity[i, c] = _add(velocity[i, c], _mul(h, acceleration))
            if predicted_velocity[i, c] == 0.0:
                predicted_velocity[i, c] = 0.0
            displacement[i, c] = _mul(h, predicted_velocity[i, c])
            if displacement[i, c] == 0.0:
                displacement[i, c] = 0.0


@tx.kernel
def reconstruct_represented(
    count: tx.i32,
    h: tx.f32,
    predicted_d: tx.ndarray(tx.f32, 2),
    predicted_v: tx.ndarray(tx.f32, 2),
    displacement: tx.ndarray(tx.f32, 2),
    velocity: tx.ndarray(tx.f32, 2),
):
    for i in tx.range(count):
        for c in tx.static_range(3):
            velocity[i, c] = _add(predicted_v[i, c], _div(_add(displacement[i, c], -predicted_d[i, c]), h))
            if velocity[i, c] == 0.0:
                velocity[i, c] = 0.0


@tx.kernel
def project_distance_represented(
    count: tx.i32,
    indices: tx.ndarray(tx.i32, 2),
    rest: tx.ndarray(tx.f32, 1),
    alpha: tx.ndarray(tx.f32, 1),
    weights: tx.ndarray(tx.f32, 1),
    displacement: tx.ndarray(tx.f32, 2),
    multipliers: tx.ndarray(tx.f32, 1),
    trace: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
    failed: tx.ndarray(tx.i32, 1),
    reference_vectors: tx.ndarray(tx.f32, 3),
    products: tx.ndarray(tx.f32, 3),
    geometry_before: tx.ndarray(tx.f32, 3),
    geometry_after: tx.ndarray(tx.f32, 3),
):
    for lane in tx.range(1):
        for k in range(count):
            if failed[0] == 0:
                i = indices[k, 0]
                j = indices[k, 1]
                for c in tx.static_range(3):
                    trace[k, c] = displacement[i, c]
                    trace[k, c + 3] = displacement[j, c]
                    geometry_before[k, 0, c] = _add(reference_vectors[k, 0, c], _add(displacement[i, c], -displacement[j, c]))
                dx = geometry_before[k, 0, 0]
                dy = geometry_before[k, 0, 1]
                dz = geometry_before[k, 0, 2]
                scale = tx.max(tx.abs(dx), tx.max(tx.abs(dy), tx.abs(dz)))
                length = 0.0
                nx = 1.0
                ny = 0.0
                nz = 0.0
                if scale > 0.0:
                    sx = _div(dx, scale)
                    sy = _div(dy, scale)
                    sz = _div(dz, scale)
                    unit = tx.sqrt(_add(_add(_mul(sx, sx), _mul(sy, sy)), _mul(sz, sz)))
                    length = _mul(scale, unit)
                    nx = _div(sx, unit)
                    ny = _div(sy, unit)
                    nz = _div(sz, unit)
                trace[k, 12] = multipliers[k]
                denominator = _add(_add(weights[i], weights[j]), alpha[k])
                if denominator <= 0.0 or not tx.isfinite(denominator):
                    status[k] = 1
                    failed[0] = k + 1
                else:
                    residual = _add(_add(length, -rest[k]), _mul(alpha[k], multipliers[k]))
                    delta = _div(-residual, denominator)
                    trace[k, 17] = delta
                    multipliers[k] = _add(multipliers[k], delta)
                    trace[k, 13] = multipliers[k]
                    for c in tx.static_range(3):
                        g = nx
                        if c == 1:
                            g = ny
                        elif c == 2:
                            g = nz
                        trace[k, 14 + c] = g
                        products[k, 0, c] = _mul(_mul(weights[i], delta), g)
                        products[k, 1, c] = _mul(_mul(weights[j], delta), -g)
                        displacement[i, c] = _add(displacement[i, c], products[k, 0, c])
                        displacement[j, c] = _add(displacement[j, c], products[k, 1, c])
                        trace[k, c + 6] = displacement[i, c]
                        trace[k, c + 9] = displacement[j, c]
                        geometry_after[k, 0, c] = _add(reference_vectors[k, 0, c], _add(displacement[i, c], -displacement[j, c]))
                        if not tx.isfinite(displacement[i, c]) or not tx.isfinite(displacement[j, c]):
                            status[k] = 2
                            failed[0] = k + 1
                for field in tx.static_range(18):
                    if trace[k, field] == 0.0:
                        trace[k, field] = 0.0
                if multipliers[k] == 0.0:
                    multipliers[k] = 0.0
                for node in tx.static_range(2):
                    for c in tx.static_range(3):
                        if displacement[indices[k, node], c] == 0.0:
                            displacement[indices[k, node], c] = 0.0
                        if products[k, node, c] == 0.0:
                            products[k, node, c] = 0.0
                for c in tx.static_range(3):
                    if geometry_before[k, 0, c] == 0.0:
                        geometry_before[k, 0, c] = 0.0
                    if geometry_after[k, 0, c] == 0.0:
                        geometry_after[k, 0, c] = 0.0