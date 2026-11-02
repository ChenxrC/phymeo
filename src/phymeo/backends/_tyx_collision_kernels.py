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
        elif tags[a, 5] != 0 or tags[b, 5] != 0:
            code = 0  # Unbounded half-space. Its anchor box cannot separate the pair.
        else:
            for k in tx.static_range(3):
                if bounds[a, k] > bounds[b, k + 3] or bounds[b, k] > bounds[a, k + 3]:
                    code = 5
        result[row] = code


@tx.kernel
def round_entry_times(count: tx.i32, rows: tx.ndarray(tx.f32, 2), entry: tx.ndarray(tx.f32, 1)):
    # First entry of a convex quadratic on a closed feature interval in FP32.
    for i in tx.range(count):
        lo = rows[i, 0]
        hi = rows[i, 1]
        a = rows[i, 2]
        b = rows[i, 3]
        c = rows[i, 4]
        answer = -1.0
        if (a * lo + b) * lo + c <= 0.0:
            answer = lo
        else:
            minimum = hi
            if a > 0.0:
                minimum = tx.max(lo, tx.min(hi, -0.5 * b / a))
            if (a * minimum + b) * minimum + c <= 0.0:
                before = lo
                after = minimum
                for _step in range(24):
                    middle = 0.5 * (before + after)
                    if (a * middle + b) * middle + c <= 0.0:
                        after = middle
                    else:
                        before = middle
                answer = after
        entry[i] = answer


@tx.kernel
def surface_entry_times(count: tx.i32, rows: tx.ndarray(tx.f32, 2), entry: tx.ndarray(tx.f32, 1)):
    # Earliest sampled feasible time of a vertex, edge or face. The exact
    # adjudicator also checks intervals between samples and repeated roots.
    for i in tx.range(count):
        answer = -1.0
        for tick in range(17):
            if answer < 0.0:
                t = tick * 0.0625
                feasible = 1
                for condition in tx.static_range(4):
                    base = condition * 7
                    val = rows[i, base + 6]
                    for k in tx.static_range(6):
                        val = val * t + rows[i, base + 5 - k]
                    if val < 0.0:
                        feasible = 0
                if feasible != 0:
                    answer = t
        entry[i] = answer
