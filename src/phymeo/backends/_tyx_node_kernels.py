"""Ordered FP32 node updates and bounded StVK material arithmetic."""

import tyx as tx

from ._tyx_numeric import _add, _div, _mul


@tx.func
def _bounded_material_product(a: tx.f32, b: tx.f32) -> tx.f32:
    # Used only in material matrix and energy reductions, whose worst-case propagated
    # absolute loss is independently bounded from the static model on the host.
    return (
        0.0
        if tx.abs(a) > 0.0
        and tx.abs(a) < 1.0
        and tx.abs(b) > 0.0
        and tx.abs(b) < 1.0
        and tx.abs(a) < 1.1754943508222875e-38 / tx.abs(b)
        else _mul(a, b)
    )


@tx.kernel
def reconstruct_nodes(
    n: tx.i32,
    h: tx.f32,
    predicted_x: tx.ndarray(tx.f32, 2),
    predicted_v: tx.ndarray(tx.f32, 2),
    x: tx.ndarray(tx.f32, 2),
    weights: tx.ndarray(tx.f32, 1),
    velocity: tx.ndarray(tx.f32, 2),
):
    for i in tx.range(n):
        for a in tx.static_range(3):
            value = 0.0
            if weights[i] > 0.0:
                value = _add(predicted_v[i, a], _div(_add(x[i, a], -predicted_x[i, a]), h))
            velocity[i, a] = value


@tx.kernel
def apply_node_loads(
    count: tx.i32,
    indices: tx.ndarray(tx.i32, 1),
    loads: tx.ndarray(tx.f32, 2),
    weights: tx.ndarray(tx.f32, 1),
    x: tx.ndarray(tx.f32, 2),
):
    for lane in tx.range(1):
        for k in range(count):
            i = indices[k]
            if weights[i] > 0.0:
                for a in tx.static_range(3):
                    x[i, a] = _add(x[i, a], _mul(weights[i], loads[k, a]))


@tx.kernel
def project_node_distances(
    count: tx.i32,
    edges: tx.ndarray(tx.i32, 2),
    rest: tx.ndarray(tx.f32, 1),
    alpha: tx.ndarray(tx.f32, 1),
    weights: tx.ndarray(tx.f32, 1),
    x: tx.ndarray(tx.f32, 2),
    multipliers: tx.ndarray(tx.f32, 1),
):
    for lane in tx.range(1):
        for k in range(count):
            i = edges[k, 0]
            j = edges[k, 1]
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
            denominator = _add(_add(weights[i], weights[j]), alpha[k])
            if denominator > 0.0:
                residual = _add(_add(length, -rest[k]), _mul(alpha[k], multipliers[k]))
                delta = _div(-residual, denominator)
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


@tx.kernel
def stvk_material(
    count: tx.i32,
    mu: tx.f32,
    lam: tx.f32,
    bulk: tx.f32,
    cells: tx.ndarray(tx.i32, 2),
    inverse: tx.ndarray(tx.f32, 3),
    volume: tx.ndarray(tx.f32, 1),
    x: tx.ndarray(tx.f32, 2),
    gradient: tx.ndarray(tx.f32, 3),
    strain: tx.ndarray(tx.f32, 3),
    stress: tx.ndarray(tx.f32, 3),
    piola: tx.ndarray(tx.f32, 3),
    force: tx.ndarray(tx.f32, 2),
    energy: tx.ndarray(tx.f32, 1),
):
    # One lane owns the complete ordered assembly, including shared vertices.
    for lane in tx.range(1):
        total_energy = 0.0
        for e in range(count):
            for a in range(3):
                for b in range(3):
                    value = 0.0
                    for k in range(3):
                        edge = _add(x[cells[e, k + 1], a], -x[cells[e, 0], a])
                        value = _add(value, _bounded_material_product(edge, inverse[e, k, b]))
                    gradient[e, a, b] = value
            for a in range(3):
                for b in range(3):
                    value = 0.0
                    for k in range(3):
                        value = _add(
                            value, _bounded_material_product(gradient[e, k, a], gradient[e, k, b])
                        )
                    if a == b:
                        value = _add(value, -1.0)
                    strain[e, a, b] = _mul(0.5, value)
            trace = _add(_add(strain[e, 0, 0], strain[e, 1, 1]), strain[e, 2, 2])
            square = 0.0
            for a in range(3):
                for b in range(3):
                    value = strain[e, a, b]
                    if lam < 0.0 and a == b:
                        first = _add(value, -strain[e, (a + 1) % 3, (a + 1) % 3])
                        second = _add(value, -strain[e, (a + 2) % 3, (a + 2) % 3])
                        value = _div(_add(first, second), 3.0)
                    square = _add(square, _bounded_material_product(value, value))
                    stress_value = _mul(_mul(2.0, mu), value)
                    if a == b:
                        coefficient = lam
                        if lam < 0.0:
                            coefficient = bulk
                        stress_value = _add(stress_value, _mul(coefficient, trace))
                    stress[e, a, b] = stress_value
            coefficient = lam
            if lam < 0.0:
                coefficient = bulk
            density = _add(
                _mul(mu, square),
                _mul(_mul(0.5, coefficient), _bounded_material_product(trace, trace)),
            )
            total_energy = _add(total_energy, _mul(volume[e], density))
            for a in range(3):
                for b in range(3):
                    value = 0.0
                    for k in range(3):
                        value = _add(
                            value, _bounded_material_product(gradient[e, a, k], stress[e, k, b])
                        )
                    piola[e, a, b] = value
            for a in range(3):
                origin_force = 0.0
                for b in range(3):
                    value = 0.0
                    for k in range(3):
                        value = _add(
                            value, _bounded_material_product(piola[e, a, k], inverse[e, b, k])
                        )
                    value = -_mul(volume[e], value)
                    i = cells[e, b + 1]
                    force[i, a] = _add(force[i, a], value)
                    origin_force = _add(origin_force, -value)
                i = cells[e, 0]
                force[i, a] = _add(force[i, a], origin_force)
        energy[0] = total_energy
