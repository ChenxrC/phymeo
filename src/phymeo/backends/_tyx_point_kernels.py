"""World/local attachment transforms in the selected numerical backend."""

import tyx as tx

from ._tyx_numeric import _add, _mul


@tx.kernel
def transform_points(
    count: tx.i32,
    inverse: tx.i32,
    indices: tx.ndarray(tx.i32, 1),
    origins: tx.ndarray(tx.f32, 2),
    rotations: tx.ndarray(tx.f32, 3),
    points: tx.ndarray(tx.f32, 2),
    result: tx.ndarray(tx.f32, 2),
):
    for row in tx.range(count):
        link = indices[row]
        for a in tx.static_range(3):
            value = 0.0
            if inverse == 0:
                value = origins[link, a]
            for b in tx.static_range(3):
                entry = rotations[link, a, b]
                coordinate = points[row, b]
                if inverse != 0:
                    entry = rotations[link, b, a]
                    coordinate = _add(coordinate, -origins[link, b])
                value = _add(value, _mul(entry, coordinate))
            result[row, a] = value
