"""FP32 topology proposals only; exact HOST boundary checks decide publication."""

import tyx as tx


@tx.kernel
def clip_topology(
    count: tx.i32,
    coefficients: tx.ndarray(tx.f32, 3),
    work: tx.ndarray(tx.f32, 4),
    tight: tx.ndarray(tx.i32, 4),
    meta: tx.ndarray(tx.i32, 2),
    active: tx.ndarray(tx.i32, 3),
):
    for lane in tx.range(count):
        status = 1
        size = 3
        bank = 0
        for vertex in range(8):
            for j in range(7):
                active[lane, vertex, j] = 0
                for slot in range(2):
                    work[lane, slot, vertex, j] = 0.0
                    tight[lane, slot, vertex, j] = 0
        for vertex in range(3):
            for j in range(7):
                value = 0.0
                if j < 3:
                    if j == vertex:
                        value = 1.0
                else:
                    value = coefficients[lane, j - 3, vertex]
                work[lane, 0, vertex, j] = value
                if value == 0.0:
                    tight[lane, 0, vertex, j] = 1
                if not tx.isfinite(value):
                    status = 5
        for plane in range(3, 7):
            if status == 1 and size > 0:
                dest = 1 - bank
                produced = 0
                for vertex in range(size):
                    previous = vertex - 1
                    if previous < 0:
                        previous = size - 1
                    da = work[lane, bank, previous, plane]
                    db = work[lane, bank, vertex, plane]
                    if (da != 0.0 and tx.abs(da) < 1e-6) or (db != 0.0 and tx.abs(db) < 1e-6):
                        status = 3
                    if (da < 0.0 and db > 0.0) or (db < 0.0 and da > 0.0):
                        if produced >= 8:
                            status = 4
                        else:
                            t = da / (da - db)
                            if not tx.isfinite(t) or t <= 0.0 or t >= 1.0:
                                status = 3
                            for j in range(7):
                                is_tight = 0
                                if j == plane or (
                                    tight[lane, bank, previous, j] == 1
                                    and tight[lane, bank, vertex, j] == 1
                                ):
                                    is_tight = 1
                                a = work[lane, bank, previous, j]
                                b = work[lane, bank, vertex, j]
                                value = a + t * (b - a)
                                if is_tight == 1:
                                    value = 0.0
                                if not tx.isfinite(value):
                                    status = 5
                                work[lane, dest, produced, j] = value
                                tight[lane, dest, produced, j] = is_tight
                            produced = produced + 1
                    if db >= 0.0:
                        if produced >= 8:
                            status = 4
                        else:
                            for j in range(7):
                                work[lane, dest, produced, j] = work[lane, bank, vertex, j]
                                is_tight = tight[lane, bank, vertex, j]
                                if j == plane and db == 0.0:
                                    is_tight = 1
                                tight[lane, dest, produced, j] = is_tight
                            produced = produced + 1
                size = produced
                bank = dest
                if size > 7:
                    status = 4
        if status == 1 and size < 3:
            status = 2
        meta[lane, 0] = status
        meta[lane, 1] = size
        if status == 1:
            for vertex in range(size):
                for j in range(7):
                    active[lane, vertex, j] = tight[lane, bank, vertex, j]
