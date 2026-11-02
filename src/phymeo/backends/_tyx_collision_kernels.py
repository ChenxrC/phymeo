"""Finite broad-phase kernels. One deterministic output slot per shape or pair."""

import tyx as tx


@tx.kernel
def reduce_bounds(
    count: tx.i32,
    offsets: tx.ndarray(tx.i32, 1),
    envelopes: tx.ndarray(tx.f32, 2),
    bounds: tx.ndarray(tx.f32, 2),
):
    for i in tx.range(count):
        first = offsets[i]
        last = offsets[i + 1]
        for k in tx.static_range(3):
            low = envelopes[first, k]
            high = envelopes[first, k + 3]
            for v in range(first + 1, last):
                if envelopes[v, k] < low:
                    low = envelopes[v, k]
                if envelopes[v, k + 3] > high:
                    high = envelopes[v, k + 3]
            bounds[i, k] = low
            bounds[i, k + 3] = high


@tx.kernel
def classify_pairs(
    count: tx.i32,
    pairs: tx.ndarray(tx.i32, 2),
    tags: tx.ndarray(tx.i32, 2),
    bounds: tx.ndarray(tx.f32, 2),
    result: tx.ndarray(tx.i32, 1),
):
    # Codes correspond to _REASONS in collision_pipeline.
    for row in tx.range(count):
        a = pairs[row, 0]
        b = pairs[row, 1]
        code = 0
        if tags[a, 0] != tags[b, 0]:
            code = 1
        elif tags[a, 2] != 0 and tags[a, 1] == tags[b, 1]:
            code = 2
        elif (tags[a, 3] & tags[b, 4]) == 0 or (tags[b, 3] & tags[a, 4]) == 0:
            code = 3
        elif pairs[row, 2] != 0:
            code = 4
        else:
            for k in tx.static_range(3):
                if bounds[a, k] > bounds[b, k + 3] or bounds[b, k] > bounds[a, k + 3]:
                    code = 5
        result[row] = code
