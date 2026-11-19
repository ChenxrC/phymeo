"""New elastic area rows; native solref/solimp equations are not reused."""

import tyx as tx


@tx.kernel
def append_rows(
    count: tx.i32,
    nv: tx.i32,
    capacity: tx.i32,
    h: tx.f32,
    scalar_damped_contacts: tx.i32,
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
                if scalar_damped_contacts != 0:
                    # Native rows use physical acceleration. Multiply the area
                    # equation by A/M so its force still acts through M unchanged.
                    scalar_mass = mass[w, 0, 0]
                    if scalar_mass > 0.0:
                        scale = response_mass[w, 0, 0] / scalar_mass
                        aref = scale * aref
                        reg = scale * reg
                        data[w, r, 3] = aref
                        data[w, r, 4] = reg
                        data[w, r, 5] = h * h * k / scale
                    else:
                        status[w] = 2
                if (
                    not tx.isfinite(reg)
                    or reg <= 0.0
                    or not tx.isfinite(aref)
                    or not tx.isfinite(velocity)
                    or not tx.isfinite(k)
                    or k <= 0.0
                ):
                    status[w] = 2


@tx.kernel
def publish_scalar_area_rows(
    count: tx.i32,
    h: tx.f32,
    samples: tx.ndarray(tx.f32, 3),
    sizes: tx.ndarray(tx.i32, 1),
    counts: tx.ndarray(tx.i32, 2),
    jac: tx.ndarray(tx.f32, 3),
    data: tx.ndarray(tx.f32, 3),
    integration_acc: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        if status[w] == 0:
            for sample in range(sizes[w]):
                r = counts[w, 2] - sizes[w] + sample
                compliance_inverse = h * h * samples[w, sample, 6]
                reg = 1.0 / compliance_inverse
                aref = -data[w, r, 0] / (h * h) - data[w, r, 2] / h
                grad = (jac[w, r, 0] * integration_acc[w, 0] - aref) + reg * data[w, r, 9]
                data[w, r, 3] = aref
                data[w, r, 4] = reg
                data[w, r, 5] = compliance_inverse
                data[w, r, 10] = grad
                if not tx.isfinite(grad):
                    status[w] = 2
