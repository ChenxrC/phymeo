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


@tx.kernel
def refit_boxes(
    count: tx.i32,
    nodes: tx.ndarray(tx.i32, 2),
    order: tx.ndarray(tx.i32, 1),
    leaves: tx.ndarray(tx.f32, 2),
    boxes: tx.ndarray(tx.f32, 2),
):
    # Descendants are contiguous in the leaf permutation. All node reductions
    # read current leaves directly, so one parallel launch needs no tree barriers.
    for i in tx.range(count):
        begin = nodes[i, 0]
        end = nodes[i, 1]
        for axis in tx.static_range(3):
            lo = leaves[order[begin], axis]
            hi = leaves[order[begin], axis + 3]
            for j in range(begin + 1, end):
                leaf = order[j]
                lo = tx.min(lo, leaves[leaf, axis])
                hi = tx.max(hi, leaves[leaf, axis + 3])
            boxes[i, axis] = lo
            boxes[i, axis + 3] = hi


@tx.kernel
def bvh_candidates(
    count: tx.i32,
    words: tx.i32,
    use_masks: tx.i32,
    nodes: tx.ndarray(tx.i32, 2),
    boxes: tx.ndarray(tx.f32, 2),
    queries: tx.ndarray(tx.f32, 2),
    params: tx.ndarray(tx.i32, 2),
    allowed: tx.ndarray(tx.i32, 2),
    out: tx.ndarray(tx.i32, 2),
):
    # Last slot owns the escape cursor. Row storage lets serial break discard
    # inactive subtrees without a loop-carried scalar across a break in tyx.
    for i in tx.range(count):
        root = params[i, 0]
        stop = nodes[root, 2]
        for w in range(words + 2):
            out[i, w] = 0
        out[i, words + 2] = root
        for step in range(stop - root):
            node = out[i, words + 2]
            if node >= stop:
                break
            out[i, words] = out[i, words] + 1
            largest = nodes[node, 3]
            bit = nodes[node, 5]
            visit = 1
            if largest <= params[i, 1]:
                visit = 0
            if bit != 0 and use_masks != 0:
                if (allowed[i, largest // 30] & bit) == 0:
                    visit = 0
            if visit != 0 and params[i, 2] == 0 and nodes[node, 4] == 0:
                if bit != 0:
                    out[i, words + 1] = out[i, words + 1] + 1
                for k in tx.static_range(3):
                    if boxes[node, k] > queries[i, k + 3] or queries[i, k] > boxes[node, k + 3]:
                        visit = 0
            if visit != 0 and bit != 0:
                word = largest // 30
                out[i, word] = out[i, word] | bit
            if visit == 0 or bit != 0:
                out[i, words + 2] = nodes[node, 2]
            else:
                out[i, words + 2] = node + 1
