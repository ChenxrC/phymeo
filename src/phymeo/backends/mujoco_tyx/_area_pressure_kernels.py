"""New elastic area rows; native solref/solimp equations are not reused."""

import tyx as tx


@tx.kernel
def append_rows(
    count: tx.i32,
    nv: tx.i32,
    capacity: tx.i32,
    h: tx.f32,
    body_ids: tx.ndarray(tx.i32, 2),
    samples: tx.ndarray(tx.f32, 3),
    pair_index: tx.ndarray(tx.i32, 2),
    sizes: tx.ndarray(tx.i32, 1),
    motion: tx.ndarray(tx.f32, 4),
    body_j: tx.ndarray(tx.f32, 4),
    v: tx.ndarray(tx.f32, 2),
    mass: tx.ndarray(tx.f32, 3),
    vectors: tx.ndarray(tx.f32, 2),
    response_mass: tx.ndarray(tx.f32, 3),
    ids: tx.ndarray(tx.i32, 3),
    data: tx.ndarray(tx.f32, 3),
    jac: tx.ndarray(tx.f32, 3),
    counts: tx.ndarray(tx.i32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        for i in range(nv):
            for j in range(nv):
                value = mass[w, i, j]
                if i == j:
                    value = value + h * vectors[1, i]
                response_mass[w, i, j] = value
        start = counts[w, 2]
        size = sizes[w]
        counts[w, 2] = start + size
        if size < 0 or start < 0 or start + size > capacity:
            status[w] = 1
        if status[w] == 0:
            for sample in range(size):
                r = start + sample
                pair = pair_index[w, sample]
                ba = body_ids[pair, 0]
                bb = body_ids[pair, 1]
                k = samples[w, sample, 6]
                phi = samples[w, sample, 7]
                velocity = 0.0
                for d in range(nv):
                    value = 0.0
                    for i in range(3):
                        j = (i + 1) % 3
                        axis2 = (i + 2) % 3
                        ja = (
                            body_j[w, ba, i, d]
                            + body_j[w, ba, 3 + j, d]
                            * (samples[w, sample, axis2] - motion[w, ba, 5, axis2])
                            - body_j[w, ba, 3 + axis2, d]
                            * (samples[w, sample, j] - motion[w, ba, 5, j])
                        )
                        jb = (
                            body_j[w, bb, i, d]
                            + body_j[w, bb, 3 + j, d]
                            * (samples[w, sample, axis2] - motion[w, bb, 5, axis2])
                            - body_j[w, bb, 3 + axis2, d]
                            * (samples[w, sample, j] - motion[w, bb, 5, j])
                        )
                        value = value + samples[w, sample, 3 + i] * (ja - jb)
                    jac[w, r, d] = value
                    velocity = velocity + value * v[w, d]
                ids[w, r, 0] = 7
                ids[w, r, 1] = sample
                for i in range(2, 5):
                    ids[w, r, i] = -1
                # Native assembly cleared the entire storage before appending.
                reg = 1.0 / (h * h * k)
                aref = -phi / (h * h) - velocity / h
                data[w, r, 0] = phi
                data[w, r, 2] = velocity
                data[w, r, 3] = aref
                data[w, r, 4] = reg
                data[w, r, 5] = h * h * k
                data[w, r, 11] = 1.0
                if (
                    not tx.isfinite(reg)
                    or reg <= 0.0
                    or not tx.isfinite(aref)
                    or not tx.isfinite(velocity)
                    or not tx.isfinite(k)
                    or k <= 0.0
                ):
                    status[w] = 2
