"""Read endpoint geometry and contract qdot with ancestor axes on device.

No step-entry dynamics Jacobian is reused, and no geometry or physics is mutated.
"""

import tyx as tx

from ._tyx_numeric import _add, _mul


def read_sphere_bodies(
    count: tx.i32,
    width: tx.i32,
    rows: tx.ndarray(tx.i32, 2),
    columns: tx.ndarray(tx.i32, 2),
    state: tx.ndarray(tx.f32, 2),
    ticks: tx.ndarray(tx.i32, 1),
    output: tx.ndarray(tx.f32, 2),
    output_ticks: tx.ndarray(tx.i32, 1),
):
    for i in tx.range(count):
        w = rows[i, 0]
        for col in range(width):
            code = columns[col, 1]
            value = 0.0
            if code < 6:
                value = state[w, code % 3]
            elif code < 15:
                if (code - 6) % 4 == 0:
                    value = 1.0
            elif code < 21:
                value = state[w, 3 + code % 3]
            output[i, col] = value
        output_ticks[i] = ticks[rows[i, 1]]


def read_tree_bodies(
    count: tx.i32,
    width: tx.i32,
    rows: tx.ndarray(tx.i32, 2),
    columns: tx.ndarray(tx.i32, 2),
    state: tx.ndarray(tx.f32, 3),
    ticks: tx.ndarray(tx.i32, 1),
    meta: tx.ndarray(tx.i32, 2),
    parents: tx.ndarray(tx.i32, 2),
    kinds: tx.ndarray(tx.i32, 2),
    anchor: tx.ndarray(tx.f32, 2),
    fk: tx.ndarray(tx.f32, 3),
    output: tx.ndarray(tx.f32, 2),
    output_ticks: tx.ndarray(tx.i32, 1),
):
    for i in tx.range(count):
        w = rows[i, 0]
        for col in range(width):
            body = columns[col, 0]
            code = columns[col, 1]
            axis = code % 3
            value = 0.0
            if code < 6:
                value = anchor[w, axis]
                offset = axis
                if code >= 3:
                    offset += 15
                value = _add(value, fk[w, body, offset])
            elif code < 15:
                value = fk[w, body, code - 3]
            else:
                ancestor = body
                for depth in range(meta[w, 1]):
                    if ancestor >= 0:
                        component = 0.0
                        if code >= 21:
                            if kinds[w, ancestor] == 0:
                                component = fk[w, ancestor, 12 + axis]
                        elif kinds[w, ancestor] == 1:
                            component = fk[w, ancestor, 12 + axis]
                        else:
                            a = (axis + 1) % 3
                            b = (axis + 2) % 3
                            point_offset = 0
                            if code >= 18:
                                point_offset = 15
                            ra = _add(fk[w, body, point_offset + a], -fk[w, ancestor, a])
                            rb = _add(fk[w, body, point_offset + b], -fk[w, ancestor, b])
                            component = _add(
                                _mul(fk[w, ancestor, 12 + a], rb),
                                -_mul(fk[w, ancestor, 12 + b], ra),
                            )
                        value = _add(value, _mul(component, state[w, ancestor, 1]))
                        ancestor = parents[w, ancestor]
            output[i, col] = value
        output_ticks[i] = ticks[rows[i, 1]]


def make_body_read_kernels(tree):
    function = read_tree_bodies if tree else read_sphere_bodies
    return tx.kernel(function), tx.kernel(function)
