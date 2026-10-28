"""Opt-in device observations of the original ordered distance arithmetic.

Keep arithmetic synchronized with project_node_distances. Bitwise parity tests
protect the unchanged default path; observations never feed the solve.
"""

import tyx as tx

from ._tyx_numeric import _add, _div, _mul


@tx.kernel
def project_node_distances_traced(
    count: tx.i32,
    edges: tx.ndarray(tx.i32, 2),
    rest: tx.ndarray(tx.f32, 1),
    alpha: tx.ndarray(tx.f32, 1),
    weights: tx.ndarray(tx.f32, 1),
    x: tx.ndarray(tx.f32, 2),
    multipliers: tx.ndarray(tx.f32, 1),
    trace: tx.ndarray(tx.f32, 2),
):
    for lane in tx.range(1):
        for k in range(count):
            i = edges[k, 0]
            j = edges[k, 1]
            for a in tx.static_range(3):
                trace[k, a] = x[i, a]
                trace[k, a + 3] = x[j, a]
            trace[k, 12] = multipliers[k]
            trace[k, 17] = 0.0
            dx = _add(x[i, 0], -x[j, 0])
            dy = _add(x[i, 1], -x[j, 1])
            dz = _add(x[i, 2], -x[j, 2])
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
            trace[k, 14] = nx
            trace[k, 15] = ny
            trace[k, 16] = nz
            denominator = _add(_add(weights[i], weights[j]), alpha[k])
            if denominator > 0.0:
                residual = _add(_add(length, -rest[k]), _mul(alpha[k], multipliers[k]))
                delta = _div(-residual, denominator)
                trace[k, 17] = delta
                multipliers[k] = _add(multipliers[k], delta)
                for a in tx.static_range(3):
                    direction = nx
                    if a == 1:
                        direction = ny
                    elif a == 2:
                        direction = nz
                    if weights[i] > 0.0:
                        x[i, a] = _add(x[i, a], _mul(_mul(weights[i], delta), direction))
                    if weights[j] > 0.0:
                        x[j, a] = _add(x[j, a], -_mul(_mul(weights[j], delta), direction))
            for a in tx.static_range(3):
                trace[k, a + 6] = x[i, a]
                trace[k, a + 9] = x[j, a]
            trace[k, 13] = multipliers[k]
