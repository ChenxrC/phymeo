"""Spatial index traversal on packed, outward-rounded boxes."""

import tyx as tx


@tx.kernel
def sap_candidates(
    count: tx.i32,
    words: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    allowed: tx.ndarray(tx.i32, 2),
    order: tx.ndarray(tx.i32, 2),
    boxes: tx.ndarray(tx.f32, 2),
    output: tx.ndarray(tx.i32, 2),
):
    # One source row owns its words. Plane intervals precede all bounded intervals.
    for pos in tx.range(count):
        src = order[pos, 0]
        axis = order[pos, 1]
        for w in range(words):
            output[src, w] = 0
        output[src, words] = 0
        output[src, words + 1] = 0
        for later in range(pos + 1, count):
            dst = order[later, 0]
            if meta[dst, 0] != meta[src, 0]:
                break
            if meta[src, 1] == 0 and boxes[dst, axis] > boxes[src, axis + 3]:
                break
            output[src, words] = output[src, words] + 1
            word = dst // 30
            mask = meta[dst, 2]
            if (allowed[src, word] & mask) != 0:
                hit = 1
                if meta[src, 1] == 0 and meta[dst, 1] == 0:
                    output[src, words + 1] = output[src, words + 1] + 1
                    for k in tx.static_range(3):
                        if boxes[src, k] > boxes[dst, k + 3] or boxes[dst, k] > boxes[src, k + 3]:
                            hit = 0
                if hit != 0:
                    output[src, word] = output[src, word] | mask
