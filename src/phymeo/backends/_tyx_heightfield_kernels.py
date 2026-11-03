"""Device-side screening of the two material triangles in every registered heightfield cell."""

import tyx as tx


@tx.kernel
def heightfield_masks(
    cells: tx.i32,
    columns: tx.i32,
    field_start: tx.i32,
    ball: tx.i32,
    generation: tx.i32,
    envelopes: tx.ndarray(tx.f32, 2),
    bounds: tx.ndarray(tx.f32, 2),
    output: tx.ndarray(tx.i32, 2),
):
    # One writer per cell. The shared diagonal is SW–NE; triangle 0 also owns SE,
    # triangle 1 NW. All six input bounds already enclose the stored FP64 geometry.
    for cell in tx.range(cells):
        row = cell // (columns - 1)
        sw = field_start + row * columns + cell % (columns - 1)
        se = sw + 1
        nw = sw + columns
        ne = nw + 1
        low_bit = 1
        high_bit = 2
        for k in tx.static_range(3):
            diagonal_lo = tx.min(envelopes[sw, k], envelopes[ne, k])
            diagonal_hi = tx.max(envelopes[sw, k + 3], envelopes[ne, k + 3])
            qlo = bounds[ball, k]
            qhi = bounds[ball, k + 3]
            if (
                tx.min(diagonal_lo, envelopes[se, k]) > qhi
                or tx.max(diagonal_hi, envelopes[se, k + 3]) < qlo
            ):
                low_bit = 0
            if (
                tx.min(diagonal_lo, envelopes[nw, k]) > qhi
                or tx.max(diagonal_hi, envelopes[nw, k + 3]) < qlo
            ):
                high_bit = 0
        output[cell, 0] = generation
        output[cell, 1] = low_bit + high_bit
