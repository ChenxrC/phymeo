"""Actual ordered FP32 signed-dihedral projection and 45-field observations."""

import tyx as tx

from ._tyx_numeric import _add, _div, _mul


@tx.kernel
def project_bending_traced(
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
                            e = _add(x[b, axis], -x[a, axis])
                            u = _add(x[c, axis], -x[a, axis])
                            v = _add(x[d, axis], -x[b, axis])
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
                                                x[index, axis] = _add(
                                                    x[index, axis],
                                                    _mul(_mul(weights[index], delta), g),
                                                )
                                                trace[k, 12 + 3 * node + axis] = x[index, axis]
                                elif tx.abs(_add(theta, -trace[k, 36])) >= 3.141592653589793:
                                    failed[0] = k + 1
                                    status[k] = 6
