"""Device-only force application and version-2 accepted area record staging."""

import tyx as tx


@tx.kernel
def apply_load(
    count: tx.i32,
    nv: tx.i32,
    width: tx.i32,
    h: tx.f32,
    pack_status: tx.ndarray(tx.i32, 1),
    solver_meta: tx.ndarray(tx.i32, 2),
    solver_vectors: tx.ndarray(tx.f32, 3),
    input_v: tx.ndarray(tx.f32, 2),
    applied: tx.ndarray(tx.f32, 2),
    generalized: tx.ndarray(tx.f32, 2),
    total_applied: tx.ndarray(tx.f32, 2),
    smooth_acc: tx.ndarray(tx.f32, 2),
    candidate_v: tx.ndarray(tx.f32, 2),
    compensated_total: tx.i32,
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        if status[w] == 0 and (pack_status[w] != 0 or solver_meta[w, 0] != 0):
            status[w] = 8
        for d in range(width):
            generalized[w, d] = 0.0
            candidate_v[w, d] = input_v[w, d]
            total_applied[w, d] = applied[w, d]
            smooth_acc[w, d] = 0.0
            if status[w] == 0 and d < nv:
                candidate_v[w, d] = solver_vectors[w, 1, d]
                value = solver_vectors[w, 7, d]
                generalized[w, d] = value
                if compensated_total != 0:
                    # The coupled solver summed external and individual contact
                    # terms before rounding, and kept the joint load separately.
                    total_applied[w, d] = solver_vectors[w, 4, d]
                else:
                    total_applied[w, d] = applied[w, d] + value
                smooth_acc[w, d] = (solver_vectors[w, 0, d] - input_v[w, d]) / h
                if not tx.isfinite(value) or not tx.isfinite(total_applied[w, d]):
                    status[w] = 8
                if not tx.isfinite(smooth_acc[w, d]):
                    status[w] = 8


@tx.kernel
def store_native_gradients(
    count: tx.i32,
    nv: tx.i32,
    capacity: tx.i32,
    counts: tx.ndarray(tx.i32, 2),
    acceleration: tx.ndarray(tx.f32, 2),
    rows: tx.ndarray(tx.f32, 3),
    jacobian: tx.ndarray(tx.f32, 3),
):
    for w in tx.range(count):
        for r in range(tx.min(capacity, counts[w, 2])):
            projected = 0.0
            for d in range(nv):
                projected = projected + jacobian[w, r, d] * acceleration[w, d]
            rows[w, r, 10] = projected + rows[w, r, 4] * rows[w, r, 9] - rows[w, r, 3]


@tx.kernel
def stage(
    count: tx.i32,
    nv: tx.i32,
    cap: tx.i32,
    npair: tx.i32,
    sizes: tx.ndarray(tx.i32, 1),
    pair_index: tx.ndarray(tx.i32, 2),
    body_ids: tx.ndarray(tx.i32, 2),
    measures: tx.ndarray(tx.i32, 3),
    samples: tx.ndarray(tx.f32, 3),
    ja: tx.ndarray(tx.f32, 4),
    jb: tx.ndarray(tx.f32, 4),
    contact: tx.ndarray(tx.f32, 3),
    actual_v: tx.ndarray(tx.f32, 2),
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
        for s in range(cap):
            for j in range(6):
                ids[w, s, j] = 0
            for j in range(23):
                data[w, s, j] = 0.0
            for j in range(2):
                words[w, s, j] = 0
        ns = sizes[w]
        if ns < 0 or ns > cap or ns > 128:
            if status[w] == 0:
                status[w] = 8
        if status[w] == 0:
            if entry_meta[w, 0] != 1:
                status[w] = 8
        if status[w] == 0:
            meta[w, 0] = 2
            meta[w, 1] = ns
            for s in range(ns):
                pair = pair_index[w, s]
                if pair < 0 or pair >= npair:
                    status[w] = 8
                else:
                    ids[w, s, 0] = 1
                    ids[w, s, 1] = pair
                    ids[w, s, 2] = s
                    ids[w, s, 3] = body_ids[pair, 0]
                    ids[w, s, 4] = body_ids[pair, 1]
                    ids[w, s, 5] = 1
                    for j in range(8):
                        data[w, s, j] = samples[w, s, j]
                    fn = contact[w, s, 0]
                    data[w, s, 8] = fn
                    if fn < 0.0:
                        status[w] = 8
                    if fn > 0.0:
                        meta[w, 2] = meta[w, 2] + 1
                    for j in range(3):
                        data[w, s, 9 + j] = contact[w, s, 9 + j]
                        data[w, s, 12 + j] = contact[w, s, 12 + j]
                        data[w, s, 15 + j] = contact[w, s, 6 + j]
                        relative = 0.0
                        for d in range(nv):
                            relative = relative + (ja[w, s, j, d] - jb[w, s, j, d]) * actual_v[w, d]
                        data[w, s, 18 + j] = relative
                    data[w, s, 21] = samples[w, s, 8]
                    data[w, s, 22] = samples[w, s, 9]
                    for j in range(2):
                        words[w, s, j] = measures[w, s, j]
                    for j in range(23):
                        if not tx.isfinite(data[w, s, j]):
                            status[w] = 8
