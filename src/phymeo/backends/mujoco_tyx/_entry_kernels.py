"""Device-only staging for the accepted-entry native/contact ABI.

These kernels never flip device.active or clear a producer error. Contact rows
are copied in native producer order; six-dimensional T4 wrenches are preserved.
"""

import tyx as tx


def state(
    count: tx.i32,
    nq: tx.i32,
    nv: tx.i32,
    na: tx.i32,
    contact_storage: tx.i32,
    q: tx.ndarray(tx.f32, 2),
    v: tx.ndarray(tx.f32, 2),
    act: tx.ndarray(tx.f32, 2),
    old_clock: tx.ndarray(tx.f32, 2),
    new_clock: tx.ndarray(tx.f32, 2),
    old_meta: tx.ndarray(tx.i32, 2),
    entry_q: tx.ndarray(tx.f32, 2),
    entry_v: tx.ndarray(tx.f32, 2),
    entry_act: tx.ndarray(tx.f32, 2),
    entry_clock: tx.ndarray(tx.f32, 2),
    meta: tx.ndarray(tx.i32, 2),
    ids: tx.ndarray(tx.i32, 3),
    data: tx.ndarray(tx.f32, 3),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        for k in range(6):
            meta[w, k] = 0
        if status[w] == 1:
            meta[w, 5] = 1
        tick = old_meta[w, 2]
        meta[w, 1] = tick
        meta[w, 2] = tick
        if tick < 0 or tick >= 2147483647:
            if status[w] == 0:
                status[w] = 2
        else:
            # Add only after the bound check, so the i32 counter never wraps.
            meta[w, 2] = tick + 1
        for k in range(nq):
            value = q[w, k]
            entry_q[w, k] = value
            if not tx.isfinite(value) and status[w] == 0:
                status[w] = 2
        for k in range(nv):
            value = v[w, k]
            entry_v[w, k] = value
            if not tx.isfinite(value) and status[w] == 0:
                status[w] = 2
        activation_storage = na
        if activation_storage < 1:
            activation_storage = 1
        for k in range(activation_storage):
            value = 0.0
            if k < na:
                value = act[w, k]
            entry_act[w, k] = value
            if not tx.isfinite(value) and status[w] == 0:
                status[w] = 2
        for k in range(2):
            value = old_clock[w, k]
            entry_clock[w, k] = value
            if not tx.isfinite(value) and status[w] == 0:
                status[w] = 2
            value = new_clock[w, k]
            entry_clock[w, 2 + k] = value
            if not tx.isfinite(value) and status[w] == 0:
                status[w] = 2
        for c in range(contact_storage):
            for k in range(8):
                ids[w, c, k] = 0
            for k in range(20):
                data[w, c, k] = 0.0
        if status[w] == 0:
            # An accepted smooth/no-contact step is a record, not missing data.
            meta[w, 0] = 1


def legacy(
    count: tx.i32,
    contact_capacity: tx.i32,
    rows: tx.ndarray(tx.f32, 2),
    law: tx.ndarray(tx.f32, 1),
    pair_ids: tx.ndarray(tx.i32, 1),
    frame: tx.ndarray(tx.f32, 2),
    meta: tx.ndarray(tx.i32, 2),
    ids: tx.ndarray(tx.i32, 3),
    data: tx.ndarray(tx.f32, 3),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        present = 0
        active = 0
        if rows[w, 0] == 1.0:
            present = 1
        if rows[w, 1] == 1.0:
            active = 1
        if (
            (rows[w, 0] != 0.0 and rows[w, 0] != 1.0)
            or (rows[w, 1] != 0.0 and rows[w, 1] != 1.0)
            or active > present
        ):
            if status[w] == 0:
                status[w] = 2
        meta[w, 3] = present
        meta[w, 4] = active
        if present > contact_capacity:
            meta[w, 5] = 1
            if status[w] == 0:
                status[w] = 1
        if present == 1 and contact_capacity > 0:
            ids[w, 0, 0] = 1
            ids[w, 0, 1] = active
            for k in range(4):
                ids[w, 0, 2 + k] = pair_ids[k]
            ids[w, 0, 6] = 0
            ids[w, 0, 7] = 1
            data[w, 0, 0] = rows[w, 2]
            for i in range(3):
                data[w, 0, 1 + i] = rows[w, 9 + i]
                for j in range(3):
                    data[w, 0, 4 + 3 * i + j] = frame[i, j]
            data[w, 0, 13] = rows[w, 3]
            data[w, 0, 19] = law[10] - law[11]
            for k in range(20):
                if not tx.isfinite(data[w, 0, k]) and status[w] == 0:
                    status[w] = 2
        if status[w] != 0:
            meta[w, 0] = 0


def normal(
    count: tx.i32,
    nslot: tx.i32,
    ngeom: tx.i32,
    contact_capacity: tx.i32,
    slot_ids: tx.ndarray(tx.i32, 2),
    geom_ids: tx.ndarray(tx.i32, 2),
    rows: tx.ndarray(tx.f32, 3),
    meta: tx.ndarray(tx.i32, 2),
    ids: tx.ndarray(tx.i32, 3),
    data: tx.ndarray(tx.f32, 3),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        detected = 0
        active_count = 0
        # Capacity limits compact output, never traversal of candidate slots.
        for s in range(nslot):
            active = 0
            if rows[w, s, 1] == 1.0:
                active = 1
            if (
                (rows[w, s, 0] != 0.0 and rows[w, s, 0] != 1.0)
                or (rows[w, s, 1] != 0.0 and rows[w, s, 1] != 1.0)
                or (active == 1 and rows[w, s, 0] != 1.0)
            ):
                if status[w] == 0:
                    status[w] = 2
            if rows[w, s, 0] == 1.0:
                c = detected
                detected = detected + 1
                active_count = active_count + active
                ga = slot_ids[s, 2]
                gb = slot_ids[s, 3]
                if ga < 0 or ga >= ngeom or gb < 0 or gb >= ngeom:
                    if status[w] == 0:
                        status[w] = 2
                elif c < contact_capacity:
                    ids[w, c, 0] = 1
                    ids[w, c, 1] = active
                    ids[w, c, 2] = ga
                    ids[w, c, 3] = gb
                    ids[w, c, 4] = geom_ids[ga, 0]
                    ids[w, c, 5] = geom_ids[gb, 0]
                    ids[w, c, 6] = slot_ids[s, 1]
                    ids[w, c, 7] = 1
                    data[w, c, 0] = rows[w, s, 2]
                    for i in range(3):
                        data[w, c, 1 + i] = rows[w, s, 9 + i]
                    for k in range(9):
                        data[w, c, 4 + k] = rows[w, s, 12 + k]
                    data[w, c, 13] = rows[w, s, 3]
                    data[w, c, 19] = rows[w, s, 21]
                    for k in range(20):
                        if not tx.isfinite(data[w, c, k]) and status[w] == 0:
                            status[w] = 2
        meta[w, 3] = detected
        meta[w, 4] = active_count
        if detected > contact_capacity:
            meta[w, 5] = 1
            if status[w] == 0:
                status[w] = 1
        if status[w] != 0:
            meta[w, 0] = 0


def constraint(
    count: tx.i32,
    nslot: tx.i32,
    ngeom: tx.i32,
    npair: tx.i32,
    contact_capacity: tx.i32,
    slot_ids: tx.ndarray(tx.i32, 2),
    geom_ids: tx.ndarray(tx.i32, 2),
    pair_condim: tx.ndarray(tx.i32, 1),
    rows: tx.ndarray(tx.f32, 3),
    counts: tx.ndarray(tx.i32, 2),
    contact_map: tx.ndarray(tx.i32, 3),
    wrench: tx.ndarray(tx.f32, 3),
    meta: tx.ndarray(tx.i32, 2),
    ids: tx.ndarray(tx.i32, 3),
    data: tx.ndarray(tx.f32, 3),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        detected = counts[w, 0]
        active_count = counts[w, 1]
        meta[w, 3] = detected
        meta[w, 4] = active_count
        if detected > contact_capacity:
            meta[w, 5] = 1
            if status[w] == 0:
                status[w] = 1
        if detected < 0 or detected > nslot or active_count < 0 or active_count > detected:
            if status[w] == 0:
                status[w] = 2
        active_seen = 0
        # Contact counts/maps exclude joint limits and DOF dry-friction rows.
        for c in range(contact_capacity):
            if c < detected:
                s = contact_map[w, c, 0]
                if s < 0 or s >= nslot:
                    if status[w] == 0:
                        status[w] = 2
                else:
                    pair = slot_ids[s, 0]
                    ga = slot_ids[s, 2]
                    gb = slot_ids[s, 3]
                    active = 0
                    if contact_map[w, c, 2] > 0:
                        active = 1
                    active_seen = active_seen + active
                    if (
                        pair < 0
                        or pair >= npair
                        or ga < 0
                        or ga >= ngeom
                        or gb < 0
                        or gb >= ngeom
                        or rows[w, s, 0] != 1.0
                        or (active == 1 and rows[w, s, 1] != 1.0)
                        or (active == 0 and rows[w, s, 1] != 0.0)
                    ):
                        if status[w] == 0:
                            status[w] = 2
                    else:
                        dim = pair_condim[pair]
                        if dim != 1 and dim != 3 and dim != 4 and dim != 6:
                            if status[w] == 0:
                                status[w] = 2
                        ids[w, c, 0] = 1
                        ids[w, c, 1] = active
                        ids[w, c, 2] = ga
                        ids[w, c, 3] = gb
                        ids[w, c, 4] = geom_ids[ga, 0]
                        ids[w, c, 5] = geom_ids[gb, 0]
                        ids[w, c, 6] = slot_ids[s, 1]
                        ids[w, c, 7] = dim
                        data[w, c, 0] = rows[w, s, 2]
                        for i in range(3):
                            data[w, c, 1 + i] = rows[w, s, 9 + i]
                        for k in range(9):
                            data[w, c, 4 + k] = rows[w, s, 12 + k]
                        for k in range(6):
                            data[w, c, 13 + k] = wrench[w, c, k]
                        data[w, c, 19] = rows[w, s, 21]
                        for k in range(20):
                            if not tx.isfinite(data[w, c, k]) and status[w] == 0:
                                status[w] = 2
        if detected <= contact_capacity and active_seen != active_count:
            if status[w] == 0:
                status[w] = 2
        if status[w] != 0:
            meta[w, 0] = 0


state = tx.kernel(state)
legacy = tx.kernel(legacy)
normal = tx.kernel(normal)
constraint = tx.kernel(constraint)
