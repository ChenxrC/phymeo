"""Explicit ordered axial velocity dissipation; never an accepted-bank filter."""

import tyx as tx

from ._tyx_numeric import _add, _div, _mul


@tx.kernel
def copy_dissipation_velocity(
    n: tx.i32,
    velocity: tx.ndarray(tx.f32, 2),
    saved_velocity: tx.ndarray(tx.f32, 2),
):
    for i in tx.range(n):
        for a in tx.static_range(3):
            saved_velocity[i, a] = velocity[i, a]


@tx.kernel
def dissipate_distances(
    count: tx.i32,
    h: tx.f32,
    edges: tx.ndarray(tx.i32, 2),
    coefficients: tx.ndarray(tx.f32, 1),
    weights: tx.ndarray(tx.f32, 1),
    x: tx.ndarray(tx.f32, 2),
    v: tx.ndarray(tx.f32, 2),
    trace: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    # One lane owns the complete declared order. Prior failure is sticky.
    # Native indices and allocated shapes are statically checked by the owner.
    for lane in tx.range(1):
        if count < 0 or count > 1536 or not tx.isfinite(h) or h <= 0.0:
            status[0] = 1
        for k in range(count):
            if status[0] == 0:
                i = edges[k, 0]
                j = edges[k, 1]
                wi = weights[i]
                wj = weights[j]
                c = coefficients[k]
                for a in tx.static_range(32):
                    trace[k, a] = 0.0
                trace[k, 18] = h
                trace[k, 19] = c
                trace[k, 20] = wi
                trace[k, 21] = wj
                trace[k, 23] = 1.0
                for a in tx.static_range(3):
                    trace[k, a + 6] = v[i, a]
                    trace[k, a + 9] = v[j, a]
                    trace[k, a + 12] = v[i, a]
                    trace[k, a + 15] = v[j, a]
                    if not tx.isfinite(v[i, a]) or not tx.isfinite(v[j, a]):
                        status[0] = 1
                    if (wi == 0.0 and v[i, a] != 0.0) or (wj == 0.0 and v[j, a] != 0.0):
                        status[0] = 3
                if not tx.isfinite(c) or c < 0.0:
                    status[0] = 1
                if not tx.isfinite(wi) or not tx.isfinite(wj) or wi < 0.0 or wj < 0.0:
                    status[0] = 3
                if status[0] == 0 and c > 0.0:
                    dx = _add(x[i, 0], -x[j, 0])
                    dy = _add(x[i, 1], -x[j, 1])
                    dz = _add(x[i, 2], -x[j, 2])
                    scale = tx.max(tx.abs(dx), tx.max(tx.abs(dy), tx.abs(dz)))
                    if not tx.isfinite(dx) or not tx.isfinite(dy) or not tx.isfinite(dz):
                        status[0] = 1
                    elif scale <= 0.0:
                        status[0] = 2
                    else:
                        sx = _div(dx, scale)
                        sy = _div(dy, scale)
                        sz = _div(dz, scale)
                        unit = tx.sqrt(_add(_add(_mul(sx, sx), _mul(sy, sy)), _mul(sz, sz)))
                        length = _mul(scale, unit)
                        nx = _div(sx, unit)
                        ny = _div(sy, unit)
                        nz = _div(sz, unit)
                        g2 = _add(_add(_mul(nx, nx), _mul(ny, ny)), _mul(nz, nz))
                        dvx = _add(v[i, 0], -v[j, 0])
                        dvy = _add(v[i, 1], -v[j, 1])
                        dvz = _add(v[i, 2], -v[j, 2])
                        rate = _add(_add(_mul(nx, dvx), _mul(ny, dvy)), _mul(nz, dvz))
                        b = _mul(h, c)
                        total_weight = _add(wi, wj)
                        denominator = _add(1.0, _mul(_mul(b, total_weight), g2))
                        impulse = _div(-_mul(b, rate), denominator)
                        if (
                            not tx.isfinite(length)
                            or length <= 0.0
                            or not tx.isfinite(g2)
                            or g2 <= 0.0
                            or not tx.isfinite(rate)
                            or not tx.isfinite(b)
                            or not tx.isfinite(denominator)
                            or denominator < 1.0
                            or not tx.isfinite(impulse)
                        ):
                            status[0] = 1
                        else:
                            trace[k, 0] = nx
                            trace[k, 1] = ny
                            trace[k, 2] = nz
                            trace[k, 3] = rate
                            trace[k, 4] = impulse
                            trace[k, 22] = length
                            trace[k, 23] = denominator
                            trace[k, 27] = g2
                            for a in tx.static_range(3):
                                direction = nx
                                if a == 1:
                                    direction = ny
                                elif a == 2:
                                    direction = nz
                                requested = _mul(impulse, direction)
                                trace[k, 24 + a] = requested
                                if wi > 0.0:
                                    v[i, a] = _add(v[i, a], _mul(_mul(wi, impulse), direction))
                                if wj > 0.0:
                                    v[j, a] = _add(v[j, a], -_mul(_mul(wj, impulse), direction))
                                trace[k, 12 + a] = v[i, a]
                                trace[k, 15 + a] = v[j, a]
                            dvx = _add(v[i, 0], -v[j, 0])
                            dvy = _add(v[i, 1], -v[j, 1])
                            dvz = _add(v[i, 2], -v[j, 2])
                            trace[k, 5] = _add(_add(_mul(nx, dvx), _mul(ny, dvy)), _mul(nz, dvz))
                            for a in tx.static_range(32):
                                if not tx.isfinite(trace[k, a]):
                                    status[0] = 1
