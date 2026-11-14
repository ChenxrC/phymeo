"""Shape-driven transport between native buckets and padded world rows.

Only accepted integration banks are exported. Padding is not read as data.
"""

import tyx as tx


def scatter(
    count: tx.i32,
    width: tx.i32,
    out_width: tx.i32,
    rows: tx.ndarray(tx.i32, 1),
    source: tx.ndarray(tx.f32, 2),
    target: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for i in tx.range(count):
        status[i] = 0
        row = rows[i]
        for j in range(out_width):
            value = 0.0
            if j < width:
                value = source[i, j]
                if not tx.isfinite(value):
                    status[i] = 1
            target[row, j] = value


def gather(
    count: tx.i32,
    width: tx.i32,
    rows: tx.ndarray(tx.i32, 1),
    source: tx.ndarray(tx.f32, 2),
    target: tx.ndarray(tx.f32, 2),
):
    for i in tx.range(count):
        for j in range(width):
            target[i, j] = source[rows[i], j]


def gather_body(
    count: tx.i32,
    bodies: tx.i32,
    rows: tx.ndarray(tx.i32, 1),
    source: tx.ndarray(tx.f32, 3),
    target: tx.ndarray(tx.f32, 3),
):
    for i in tx.range(count):
        for b in range(bodies):
            for j in range(6):
                target[i, b, j] = source[rows[i], b, j]


scatter = tx.kernel(scatter)
gather = tx.kernel(gather)
gather_body = tx.kernel(gather_body)
