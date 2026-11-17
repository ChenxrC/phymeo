"""Stage area observations from actual device inputs and solved rows."""

import tyx as tx


@tx.kernel
def stage(
    count: tx.i32,
    cap: tx.i32,
    npair: tx.i32,
    sizes: tx.ndarray(tx.i32, 1),
    pair_index: tx.ndarray(tx.i32, 2),
    body_ids: tx.ndarray(tx.i32, 2),
    samples: tx.ndarray(tx.f32, 3),
    measures: tx.ndarray(tx.i32, 3),
    counts: tx.ndarray(tx.i32, 2),
    row_ids: tx.ndarray(tx.i32, 3),
    row_data: tx.ndarray(tx.f32, 3),
    entry_meta: tx.ndarray(tx.i32, 2),
    meta: tx.ndarray(tx.i32, 2),
    ids: tx.ndarray(tx.i32, 3),
    data: tx.ndarray(tx.f32, 3),
    words: tx.ndarray(tx.i32, 3),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        for j in range(4):
            meta[w, j] = 0
        for c in range(cap):
            for j in range(6):
                ids[w, c, j] = 0
            for j in range(15):
                data[w, c, j] = 0.0
            for j in range(2):
                words[w, c, j] = 0
        size = sizes[w]
        total = counts[w, 2]
        start = 0
        if total < 0 or total > cap or size < 0 or size > total:
            if status[w] == 0:
                status[w] = 1
        if status[w] == 0:
            start = total - size
            meta[w, 0] = entry_meta[w, 0]
            meta[w, 1] = size
            for c in range(size):
                pair = pair_index[w, c]
                r = start + c
                if pair < 0 or pair >= npair:
                    status[w] = 2
                elif row_ids[w, r, 0] != 7 or row_ids[w, r, 1] != c:
                    status[w] = 2
                else:
                    ids[w, c, 0] = 1
                    ids[w, c, 1] = pair
                    ids[w, c, 2] = c
                    ids[w, c, 3] = body_ids[pair, 0]
                    ids[w, c, 4] = body_ids[pair, 1]
                    for j in range(8):
                        data[w, c, j] = samples[w, c, j]
                        if not tx.isfinite(samples[w, c, j]):
                            status[w] = 2
                    force = row_data[w, r, 9]
                    data[w, c, 8] = force
                    if not tx.isfinite(force) or force < 0.0:
                        status[w] = 2
                    if force > 0.0:
                        meta[w, 2] = meta[w, 2] + 1
                    for j in range(3):
                        data[w, c, 9 + j] = samples[w, c, 3 + j] * force
                    ids[w, c, 5] = 1
                    for i in range(3):
                        j = (i + 1) % 3
                        k = (i + 2) % 3
                        left = samples[w, c, j] * data[w, c, 9 + k]
                        right = samples[w, c, k] * data[w, c, 9 + j]
                        # Bound each product before subtraction: CUDA fusion cannot
                        # change this conservative validity rule. No physics gate.
                        if not tx.isfinite(left) or not tx.isfinite(right):
                            ids[w, c, 5] = 0
                        elif (
                            tx.abs(left) > 4.253529586511731e37
                            or tx.abs(right) > 4.253529586511731e37
                        ):
                            ids[w, c, 5] = 0
                        else:
                            data[w, c, 12 + i] = left - right
                    if ids[w, c, 5] == 0:
                        for i in range(3):
                            data[w, c, 12 + i] = 0.0
                    for j in range(2):
                        words[w, c, j] = measures[w, c, j]
                    for j in range(9, 15):
                        if not tx.isfinite(data[w, c, j]):
                            status[w] = 2
