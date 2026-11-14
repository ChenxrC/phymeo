"""Device transport and world-wrench derivation for accepted-entry observations."""

import tyx as tx


def scatter(
    count: tx.i32,
    nq: tx.i32,
    nv: tx.i32,
    na: tx.i32,
    nbody: tx.i32,
    ngeom: tx.i32,
    contact_capacity: tx.i32,
    identity_count: tx.i32,
    out_nq: tx.i32,
    out_nv: tx.i32,
    out_na: tx.i32,
    out_contacts: tx.i32,
    rows: tx.ndarray(tx.i32, 1),
    identities: tx.ndarray(tx.i32, 2),
    source_q: tx.ndarray(tx.f32, 2),
    source_v: tx.ndarray(tx.f32, 2),
    source_act: tx.ndarray(tx.f32, 2),
    source_clock: tx.ndarray(tx.f32, 2),
    source_meta: tx.ndarray(tx.i32, 2),
    source_ids: tx.ndarray(tx.i32, 3),
    source_data: tx.ndarray(tx.f32, 3),
    current_q: tx.ndarray(tx.f32, 2),
    current_v: tx.ndarray(tx.f32, 2),
    current_clock: tx.ndarray(tx.f32, 2),
    safe_q: tx.ndarray(tx.f32, 2),
    safe_v: tx.ndarray(tx.f32, 2),
    q_out: tx.ndarray(tx.f32, 2),
    v_out: tx.ndarray(tx.f32, 2),
    act_out: tx.ndarray(tx.f32, 2),
    clock_out: tx.ndarray(tx.f32, 2),
    meta_out: tx.ndarray(tx.i32, 2),
    ids_out: tx.ndarray(tx.i32, 3),
    data_out: tx.ndarray(tx.f32, 3),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        status[w] = 0
        row = rows[w]
        valid = source_meta[w, 0]
        before = source_meta[w, 1]
        after = source_meta[w, 2]
        detected = source_meta[w, 3]
        active_count = source_meta[w, 4]
        if (valid != 0 and valid != 1) or after < 0 or source_meta[w, 5] != 0:
            status[w] = 2
        for k in range(6):
            meta_out[row, k] = 0
        # A restored legacy capsule can lack a record at a nonzero accepted tick.
        meta_out[row, 2] = after
        if valid == 1:
            if before < 0 or before >= 2147483647:
                status[w] = 2
            elif before + 1 != after:
                status[w] = 2
            if (
                detected < 0
                or detected > contact_capacity
                or active_count < 0
                or active_count > detected
            ):
                status[w] = 2
            for k in range(6):
                meta_out[row, k] = source_meta[w, k]
        elif before != 0 or detected != 0 or active_count != 0:
            status[w] = 2
        qwidth = nq
        if qwidth < 8:
            qwidth = 8
        vwidth = nv
        if vwidth < 8:
            vwidth = 8
        awidth = na
        if awidth < 1:
            awidth = 1
        for k in range(out_nq):
            q_out[row, k] = 0.0
        for k in range(out_nv):
            v_out[row, k] = 0.0
        for k in range(out_na):
            act_out[row, k] = 0.0
        for k in range(qwidth):
            value = source_q[w, k]
            if not tx.isfinite(value):
                status[w] = 2
            if (valid != 1 or k >= nq) and value != 0.0:
                status[w] = 2
            safe = current_q[w, k]
            if valid == 1:
                safe = value
                if k < nq:
                    q_out[row, k] = value
            safe_q[w, k] = safe
            if not tx.isfinite(safe):
                status[w] = 2
        for k in range(vwidth):
            value = source_v[w, k]
            if not tx.isfinite(value):
                status[w] = 2
            if (valid != 1 or k >= nv) and value != 0.0:
                status[w] = 2
            safe = current_v[w, k]
            if valid == 1:
                safe = value
                if k < nv:
                    v_out[row, k] = value
            safe_v[w, k] = safe
            if not tx.isfinite(safe):
                status[w] = 2
        for k in range(awidth):
            value = source_act[w, k]
            if not tx.isfinite(value):
                status[w] = 2
            if (valid != 1 or k >= na) and value != 0.0:
                status[w] = 2
            if valid == 1 and k < na:
                act_out[row, k] = value
        for k in range(4):
            value = source_clock[w, k]
            if not tx.isfinite(value):
                status[w] = 2
            clock_out[row, k] = 0.0
            if valid == 1:
                clock_out[row, k] = value
                if k >= 2:
                    if value != current_clock[w, k - 2]:
                        status[w] = 2
            elif value != 0.0:
                status[w] = 2
        storage = contact_capacity
        if storage < 1:
            storage = 1
        active_seen = 0
        for c in range(out_contacts):
            for k in range(8):
                ids_out[row, c, k] = 0
            for k in range(20):
                data_out[row, c, k] = 0.0
            if c < storage:
                for k in range(8):
                    value_id = source_ids[w, c, k]
                    if valid == 1 and c < detected:
                        ids_out[row, c, k] = value_id
                    elif value_id != 0:
                        status[w] = 2
                for k in range(20):
                    value = source_data[w, c, k]
                    if not tx.isfinite(value):
                        status[w] = 2
                    if valid == 1 and c < detected:
                        data_out[row, c, k] = value
                    elif value != 0.0:
                        status[w] = 2
                if valid == 1 and c < detected:
                    active = source_ids[w, c, 1]
                    geom1 = source_ids[w, c, 2]
                    geom2 = source_ids[w, c, 3]
                    body1 = source_ids[w, c, 4]
                    body2 = source_ids[w, c, 5]
                    feature = source_ids[w, c, 6]
                    dim = source_ids[w, c, 7]
                    if (
                        source_ids[w, c, 0] != 1
                        or (active != 0 and active != 1)
                        or geom1 < 0
                        or geom1 >= ngeom
                        or geom2 < 0
                        or geom2 >= ngeom
                        or body1 < 0
                        or body1 >= nbody
                        or body2 < 0
                        or body2 >= nbody
                        or feature < 0
                        or (dim != 1 and dim != 3 and dim != 4 and dim != 6)
                    ):
                        status[w] = 2
                    if active == 1:
                        active_seen = active_seen + 1
                    found = 0
                    for slot in range(identity_count):
                        same = 1
                        for k in range(6):
                            if identities[slot, k] != source_ids[w, c, 2 + k]:
                                same = 0
                        if same == 1:
                            found = 1
                    if found == 0:
                        status[w] = 2
                    for k in range(6):
                        if (active == 0 or k >= dim) and source_data[w, c, 13 + k] != 0.0:
                            status[w] = 2
        if valid == 1 and active_seen != active_count:
            status[w] = 2


def finish(
    count: tx.i32,
    nbody: tx.i32,
    body_width: tx.i32,
    contact_width: tx.i32,
    rows: tx.ndarray(tx.i32, 1),
    meta: tx.ndarray(tx.i32, 2),
    ids: tx.ndarray(tx.i32, 3),
    data: tx.ndarray(tx.f32, 3),
    position: tx.ndarray(tx.f32, 3),
    rotation: tx.ndarray(tx.f32, 4),
    linear: tx.ndarray(tx.f32, 3),
    angular: tx.ndarray(tx.f32, 3),
    body_mask: tx.ndarray(tx.i32, 2),
    world_wrench: tx.ndarray(tx.f32, 3),
    body_wrench: tx.ndarray(tx.f32, 4),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        row = rows[w]
        valid = meta[row, 0]
        for b in range(body_width):
            keep = 0
            if valid == 1 and b < nbody:
                keep = 1
            body_mask[row, b] = keep
            for i in range(3):
                if keep == 0:
                    position[row, b, i] = 0.0
                    linear[row, b, i] = 0.0
                    angular[row, b, i] = 0.0
                if (
                    not tx.isfinite(position[row, b, i])
                    or not tx.isfinite(linear[row, b, i])
                    or not tx.isfinite(angular[row, b, i])
                ):
                    status[w] = 2
                for j in range(3):
                    if keep == 0:
                        rotation[row, b, i, j] = 0.0
                    if not tx.isfinite(rotation[row, b, i, j]):
                        status[w] = 2
        for c in range(contact_width):
            for k in range(6):
                world_wrench[row, c, k] = 0.0
                body_wrench[row, c, 0, k] = 0.0
                body_wrench[row, c, 1, k] = 0.0
            if valid == 1 and ids[row, c, 0] == 1 and ids[row, c, 1] == 1:
                b1 = ids[row, c, 4]
                b2 = ids[row, c, 5]
                if b1 < 0 or b1 >= nbody or b2 < 0 or b2 >= nbody:
                    status[w] = 2
                else:
                    # Frame rows are world axes: transpose maps contact components.
                    for i in range(3):
                        force = 0.0
                        couple = 0.0
                        for j in range(3):
                            force = force + data[row, c, 4 + 3 * j + i] * data[row, c, 13 + j]
                            couple = couple + data[row, c, 4 + 3 * j + i] * data[row, c, 16 + j]
                        world_wrench[row, c, i] = force
                        world_wrench[row, c, 3 + i] = couple
                        body_wrench[row, c, 0, i] = -force
                        body_wrench[row, c, 1, i] = force
                    for i in range(3):
                        j = (i + 1) % 3
                        k = (i + 2) % 3
                        r1j = data[row, c, 1 + j] - position[row, b1, j]
                        r1k = data[row, c, 1 + k] - position[row, b1, k]
                        r2j = data[row, c, 1 + j] - position[row, b2, j]
                        r2k = data[row, c, 1 + k] - position[row, b2, k]
                        body_wrench[row, c, 0, 3 + i] = (
                            -world_wrench[row, c, 3 + i]
                            - r1j * world_wrench[row, c, k]
                            + r1k * world_wrench[row, c, j]
                        )
                        body_wrench[row, c, 1, 3 + i] = (
                            world_wrench[row, c, 3 + i]
                            + r2j * world_wrench[row, c, k]
                            - r2k * world_wrench[row, c, j]
                        )
            for k in range(6):
                if (
                    not tx.isfinite(world_wrench[row, c, k])
                    or not tx.isfinite(body_wrench[row, c, 0, k])
                    or not tx.isfinite(body_wrench[row, c, 1, k])
                ):
                    status[w] = 2


scatter = tx.kernel(scatter)
finish = tx.kernel(finish)
