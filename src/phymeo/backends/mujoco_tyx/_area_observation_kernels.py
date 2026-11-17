"""Scatter accepted area records into the shared heterogeneous observation bank."""

import tyx as tx


@tx.kernel
def scatter(
    count: tx.i32,
    capacity: tx.i32,
    out_capacity: tx.i32,
    npair: tx.i32,
    nbody: tx.i32,
    rows: tx.ndarray(tx.i32, 1),
    body_ids: tx.ndarray(tx.i32, 2),
    entry_meta: tx.ndarray(tx.i32, 2),
    source_meta: tx.ndarray(tx.i32, 2),
    source_ids: tx.ndarray(tx.i32, 3),
    source_data: tx.ndarray(tx.f32, 3),
    source_words: tx.ndarray(tx.i32, 3),
    meta: tx.ndarray(tx.i32, 2),
    ids: tx.ndarray(tx.i32, 3),
    data: tx.ndarray(tx.f32, 3),
    words: tx.ndarray(tx.i32, 3),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        status[w] = 0
        row = rows[w]
        valid = source_meta[w, 0]
        size = source_meta[w, 1]
        positive = source_meta[w, 2]
        if valid != 0 and valid != 1:
            status[w] = 2
        if valid != entry_meta[w, 0] or source_meta[w, 3] != 0:
            status[w] = 2
        if size < 0 or size > capacity or positive < 0 or positive > size:
            status[w] = 2
        if valid == 0 and (size != 0 or positive != 0):
            status[w] = 2
        for j in range(4):
            meta[row, j] = source_meta[w, j]
        seen = 0
        for c in range(out_capacity):
            for j in range(6):
                ids[row, c, j] = 0
            for j in range(15):
                data[row, c, j] = 0.0
            for j in range(2):
                words[row, c, j] = 0
            if c < capacity:
                used = valid == 1 and c < size
                for j in range(6):
                    value = source_ids[w, c, j]
                    ids[row, c, j] = value
                    if not used and value != 0:
                        status[w] = 2
                for j in range(15):
                    value = source_data[w, c, j]
                    data[row, c, j] = value
                    if not tx.isfinite(value) or (not used and value != 0.0):
                        status[w] = 2
                for j in range(2):
                    value = source_words[w, c, j]
                    words[row, c, j] = value
                    if not used and value != 0:
                        status[w] = 2
                if used:
                    if source_ids[w, c, 0] != 1 or source_ids[w, c, 2] != c:
                        status[w] = 2
                    pair = source_ids[w, c, 1]
                    ba = source_ids[w, c, 3]
                    bb = source_ids[w, c, 4]
                    moment = source_ids[w, c, 5]
                    if pair < 0 or pair >= npair:
                        status[w] = 2
                    elif ba != body_ids[pair, 0] or bb != body_ids[pair, 1]:
                        status[w] = 2
                    if ba < 0 or ba >= nbody or bb < 0 or bb >= nbody or ba == bb:
                        status[w] = 2
                    if moment != 0 and moment != 1:
                        status[w] = 2
                    if moment == 0:
                        for j in range(12, 15):
                            if source_data[w, c, j] != 0.0:
                                status[w] = 2
                    if source_data[w, c, 6] <= 0.0 or source_data[w, c, 8] < 0.0:
                        status[w] = 2
                    if source_data[w, c, 8] > 0.0:
                        seen = seen + 1
        if seen != positive:
            status[w] = 2
