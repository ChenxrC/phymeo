# Copyright 2021 DeepMind Technologies Limited
# Copyright 2016 Svetoslav Kolev
# Copyright 2026 Phymeo contributors
# Apache-2.0; see LICENSE and NOTICE.
# Translation of MuJoCo 3.3.7 primitive/box collision and driver postprocessing.

"""Bounded native-order capsule/capsule, capsule/box and box/box manifolds."""

import tyx as tx


@tx.func
def _bit(code: tx.i32, axis: tx.i32) -> tx.i32:
    return (code // (1 + axis * (axis + 1) // 2)) % 2


def extended_contacts(
    count: tx.i32,
    ngeom: tx.i32,
    npair: tx.i32,
    nslot: tx.i32,
    geom_ids: tx.ndarray(tx.i32, 2),
    geom_data: tx.ndarray(tx.f32, 2),
    pair_ids: tx.ndarray(tx.i32, 2),
    pair_law: tx.ndarray(tx.f32, 2),
    motion: tx.ndarray(tx.f32, 4),
    rotation: tx.ndarray(tx.f32, 4),
    geom_world: tx.ndarray(tx.f32, 3),
    raw_rows: tx.ndarray(tx.f32, 3),
    extended_scratch: tx.ndarray(tx.f32, 3),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        for row in range(nslot):
            for k in range(11):
                raw_rows[w, row, k] = 0.0
        for g in range(ngeom):
            body = geom_ids[g, 0]
            for i in range(3):
                value = motion[w, body, 0, i]
                for k in range(3):
                    value = value + rotation[w, body, i, k] * geom_data[g, k]
                geom_world[w, g, i] = value
                if not tx.isfinite(value):
                    status[w] = 2
                for j in range(3):
                    value = 0.0
                    for k in range(3):
                        value = value + rotation[w, body, i, k] * geom_data[g, 3 + 3 * k + j]
                    geom_world[w, g, 3 + 3 * i + j] = value
                    if not tx.isfinite(value):
                        status[w] = 2
        for pair in range(npair):
            ga = pair_ids[pair, 0]
            gb = pair_ids[pair, 1]
            ta = geom_ids[ga, 1]
            tb = geom_ids[gb, 1]
            start = pair_ids[pair, 2]
            slots = pair_ids[pair, 3]
            margin = pair_law[pair, 0]
            minimum = pair_law[pair, 10]
            n = 0
            if ta >= 3:
                for scratch_row in range(64):
                    for scratch_column in range(3):
                        extended_scratch[w, scratch_row, scratch_column] = 0.0
            if ta == 3 and tb == 3:
                # Scaled segment axes and native sequentially clamped line solution.
                ma = 0.0
                mb = 0.0
                mc = 0.0
                u = 0.0
                v = 0.0
                for i in range(3):
                    a = geom_world[w, ga, 5 + 3 * i] * geom_data[ga, 13]
                    b = geom_world[w, gb, 5 + 3 * i] * geom_data[gb, 13]
                    d = geom_world[w, ga, i] - geom_world[w, gb, i]
                    extended_scratch[w, 0, i] = a
                    extended_scratch[w, 1, i] = b
                    ma = ma + a * a
                    mb = mb - a * b
                    mc = mc + b * b
                    u = u - a * d
                    v = v + b * d
                # Equivalent Gram determinant and line numerators without
                # subtracting nearly equal squared dot products in FP32.
                det = 0.0
                numerator1 = 0.0
                numerator2 = 0.0
                for i in range(3):
                    j = (i + 1) % 3
                    k = (i + 2) % 3
                    cross = (
                        extended_scratch[w, 0, j] * extended_scratch[w, 1, k]
                        - extended_scratch[w, 0, k] * extended_scratch[w, 1, j]
                    )
                    dj = geom_world[w, ga, j] - geom_world[w, gb, j]
                    dk = geom_world[w, ga, k] - geom_world[w, gb, k]
                    det = det + cross * cross
                    numerator1 = numerator1 + cross * (
                        extended_scratch[w, 1, j] * dk - extended_scratch[w, 1, k] * dj
                    )
                    numerator2 = numerator2 + cross * (
                        extended_scratch[w, 0, j] * dk - extended_scratch[w, 0, k] * dj
                    )
                attempts = 1
                if tx.abs(det) < minimum:
                    attempts = 4
                if ma <= 0.0 or mc <= 0.0 or not tx.isfinite(det):
                    status[w] = 2
                    attempts = 0
                for attempt in range(attempts):
                    if n < 2:
                        x1 = 0.0
                        x2 = 0.0
                        if attempts == 1:
                            x1 = numerator1 / det
                            x2 = numerator2 / det
                            if x1 > 1.0:
                                x1 = 1.0
                                x2 = (v - mb) / mc
                            elif x1 < -1.0:
                                x1 = -1.0
                                x2 = (v + mb) / mc
                            if x2 > 1.0:
                                x2 = 1.0
                                x1 = tx.max(-1.0, tx.min(1.0, (u - mb) / ma))
                            elif x2 < -1.0:
                                x2 = -1.0
                                x1 = tx.max(-1.0, tx.min(1.0, (u + mb) / ma))
                        elif attempt < 2:
                            x1 = 1.0 - 2.0 * attempt
                            x2 = tx.max(-1.0, tx.min(1.0, (v - mb * x1) / mc))
                        else:
                            x2 = 5.0 - 2.0 * attempt
                            x1 = tx.max(-1.0, tx.min(1.0, (u - mb * x2) / ma))
                        length2 = 0.0
                        for i in range(3):
                            pa = geom_world[w, ga, i] + x1 * extended_scratch[w, 0, i]
                            pb = geom_world[w, gb, i] + x2 * extended_scratch[w, 1, i]
                            extended_scratch[w, 2, i] = pa
                            extended_scratch[w, 3, i] = pb - pa
                            length2 = length2 + (pb - pa) * (pb - pa)
                        radius = geom_data[ga, 12] + geom_data[gb, 12]
                        if length2 <= (margin + radius) * (margin + radius):
                            length = tx.sqrt(length2)
                            distance = length - radius
                            if length < minimum:
                                length2 = 0.0
                                for i in range(3):
                                    j = (i + 1) % 3
                                    k = (i + 2) % 3
                                    value = (
                                        geom_world[w, ga, 5 + 3 * j] * geom_world[w, gb, 5 + 3 * k]
                                        - geom_world[w, ga, 5 + 3 * k]
                                        * geom_world[w, gb, 5 + 3 * j]
                                    )
                                    extended_scratch[w, 3, i] = value
                                    length2 = length2 + value * value
                                length = tx.sqrt(length2)
                                if length < minimum:
                                    length = 1.0
                                    extended_scratch[w, 3, 0] = 1.0
                                    extended_scratch[w, 3, 1] = 0.0
                                    extended_scratch[w, 3, 2] = 0.0
                            if n < slots:
                                row = start + n
                                raw_rows[w, row, 0] = 1.0
                                raw_rows[w, row, 1] = distance
                                for i in range(3):
                                    normal = extended_scratch[w, 3, i] / length
                                    raw_rows[w, row, 5 + i] = normal
                                    raw_rows[w, row, 2 + i] = extended_scratch[w, 2, i] + normal * (
                                        geom_data[ga, 12] + 0.5 * distance
                                    )
                                n = n + 1
                            else:
                                status[w] = 1
            elif ta == 3 and tb == 6:
                # Capsule in box frame: CPU j-axis outer / corner inner tie order.
                axisdir = 0
                for i in range(3):
                    pos = 0.0
                    axis_component = 0.0
                    for j in range(3):
                        pos = pos + geom_world[w, gb, 3 + 3 * j + i] * (
                            geom_world[w, ga, j] - geom_world[w, gb, j]
                        )
                        axis_component = (
                            axis_component
                            + geom_world[w, gb, 3 + 3 * j + i] * geom_world[w, ga, 5 + 3 * j]
                        )
                    extended_scratch[w, 0, i] = pos
                    extended_scratch[w, 1, i] = axis_component
                    extended_scratch[w, 2, i] = axis_component * geom_data[ga, 13]
                    if axis_component * geom_data[ga, 13] > 0.0:
                        axisdir = axisdir + 1 + i * (i + 1) // 2
                bestdist = margin + 2.0 * (
                    geom_data[ga, 12]
                    + geom_data[ga, 13]
                    + geom_data[gb, 12]
                    + geom_data[gb, 13]
                    + geom_data[gb, 14]
                )
                best = 0.0
                bestbox = 0.0
                cltype = -4
                clface = -1
                clcorner = 0
                cledge = 0
                for endpoint in range(2):
                    sign = -1.0 + 2.0 * endpoint
                    outside = 0
                    face = -1
                    distance2 = 0.0
                    for j in range(3):
                        tip = extended_scratch[w, 0, j] + sign * extended_scratch[w, 2, j]
                        limited = tx.max(-geom_data[gb, 12 + j], tx.min(geom_data[gb, 12 + j], tip))
                        if tip < -geom_data[gb, 12 + j] or tip > geom_data[gb, 12 + j]:
                            outside = outside + 1
                            face = j
                        distance2 = distance2 + (limited - tip) * (limited - tip)
                    if outside <= 1 and distance2 < bestdist:
                        bestdist = distance2
                        best = sign
                        cltype = -3 + 2 * endpoint
                        clface = face
                for j in range(3):
                    for corner in range(8):
                        if _bit(corner, j) == 0:
                            dot = 0.0
                            for k in range(3):
                                value = (2.0 * _bit(corner, k) - 1.0) * geom_data[gb, 12 + k]
                                if k == j:
                                    value = 0.0
                                value = value - extended_scratch[w, 0, k]
                                extended_scratch[w, 3, k] = value
                                dot = dot + extended_scratch[w, 2, k] * value
                            ma = geom_data[gb, 12 + j] * geom_data[gb, 12 + j]
                            mb = -geom_data[gb, 12 + j] * extended_scratch[w, 2, j]
                            mc = geom_data[ga, 13] * geom_data[ga, 13]
                            u = -geom_data[gb, 12 + j] * extended_scratch[w, 3, j]
                            v = dot
                            det = ma * mc - mb * mb
                            if ma <= 0.0 or mc <= 0.0 or not tx.isfinite(det):
                                status[w] = 2
                            if tx.abs(det) >= minimum:
                                x1 = (mc * u - mb * v) / det
                                x2 = (ma * v - mb * u) / det
                                s1 = 1
                                s2 = 1
                                if x1 > 1.0:
                                    x1 = 1.0
                                    s1 = 2
                                    x2 = (v - mb) / mc
                                elif x1 < -1.0:
                                    x1 = -1.0
                                    s1 = 0
                                    x2 = (v + mb) / mc
                                if x2 > 1.0:
                                    x2 = 1.0
                                    s2 = 2
                                    x1 = (u - mb) / ma
                                    if x1 > 1.0:
                                        x1 = 1.0
                                        s1 = 2
                                    elif x1 < -1.0:
                                        x1 = -1.0
                                        s1 = 0
                                elif x2 < -1.0:
                                    x2 = -1.0
                                    s2 = 0
                                    x1 = (u + mb) / ma
                                    if x1 > 1.0:
                                        x1 = 1.0
                                        s1 = 2
                                    elif x1 < -1.0:
                                        x1 = -1.0
                                        s1 = 0
                                distance2 = 0.0
                                for k in range(3):
                                    value = (
                                        extended_scratch[w, 3, k] - extended_scratch[w, 2, k] * x2
                                    )
                                    if k == j:
                                        value = value + geom_data[gb, 12 + j] * x1
                                    distance2 = distance2 + value * value
                                if distance2 < bestdist - minimum:
                                    bestdist = distance2
                                    best = x2
                                    bestbox = x1
                                    cltype = s1 * 3 + s2
                                    clcorner = corner + (1 + j * (j + 1) // 2) * (cltype // 6)
                                    cledge = j
                # Native diagnostic-only block is omitted: it has no live outputs.
                second = -4.0
                relation = 0
                for i in range(3):
                    if _bit(axisdir, i) != _bit(clcorner, i):
                        relation = relation + 1 + i * (i + 1) // 2
                if cltype >= 0 and cltype // 3 != 1:
                    if relation != 0 and relation != 7:
                        mul = 1.0
                        de = 1.0 - best
                        dp = 1.0 + best
                        if relation == 3 or relation == 5 or relation == 6:
                            mul = -1.0
                            relation = 7 - relation
                            dp = 1.0 - best
                            de = 1.0 + best
                        ax = 0
                        if relation == 2:
                            ax = 1
                        elif relation == 4:
                            ax = 2
                        ax1 = (ax + 1) % 3
                        ax2 = (ax + 2) % 3
                        if extended_scratch[w, 1, ax] * extended_scratch[w, 1, ax] > 0.5:
                            second = de
                            den = tx.abs(extended_scratch[w, 2, ax])
                            if den > 0.0:
                                second = tx.min(second, 2.0 * geom_data[gb, 12 + ax] / den)
                            second = second * mul
                        else:
                            second = dp
                            for k in range(2):
                                axis = ax1
                                if k == 1:
                                    axis = ax2
                                den = tx.abs(extended_scratch[w, 2, axis])
                                if den > 0.0:
                                    second = tx.min(second, 2.0 * geom_data[gb, 12 + axis] / den)
                            second = -second * mul
                elif cltype >= 0:
                    if _bit(relation, cledge) != 0:
                        relation = relation - (1 + cledge * (cledge + 1) // 2)
                    if relation == 1 or relation == 2 or relation == 4:
                        ax = cledge
                        ax1 = (ax + 1) % 3
                        ax2 = (ax + 2) % 3
                        if tx.abs(extended_scratch[w, 1, ax1]) > tx.abs(
                            extended_scratch[w, 1, ax2]
                        ):
                            ax1 = ax2
                        ax2 = 3 - ax - ax1
                        mul = -1.0
                        second = 1.0 + best
                        if _bit(relation, ax2) != 0:
                            mul = 1.0
                            second = 1.0 - best
                        den = tx.abs(extended_scratch[w, 2, ax2])
                        if den > 0.0:
                            second = tx.min(second, 2.0 * geom_data[gb, 12 + ax2] / den)
                        e2 = 1.0 + bestbox
                        if _bit(axisdir, ax) == _bit(relation, ax2):
                            e2 = 1.0 - bestbox
                        den = tx.abs(extended_scratch[w, 2, ax])
                        if den > 0.0:
                            second = tx.min(second, geom_data[gb, 12 + ax] * e2 / den)
                        second = second * mul
                elif cltype != -4 and clface != -1:
                    mul = -1.0
                    if cltype == -3:
                        mul = 1.0
                    second = 2.0
                    for i in range(3):
                        if i != clface and extended_scratch[w, 2, i] != 0.0:
                            tip = extended_scratch[w, 0, i] - extended_scratch[w, 2, i] * mul
                            for side in range(2):
                                value = (
                                    ((1.0 - 2.0 * side) * geom_data[gb, 12 + i] - tip)
                                    / extended_scratch[w, 2, i]
                                    * mul
                                )
                                if value > 0.0:
                                    second = tx.min(second, value)
                    second = second * mul
                attempts = 0
                if cltype != -4:
                    attempts = 1
                    if second > -3.0:
                        attempts = 2
                for attempt in range(attempts):
                    segment = best
                    if attempt == 1:
                        segment = segment + second
                    length2 = 0.0
                    for i in range(3):
                        center = extended_scratch[w, 0, i] + segment * extended_scratch[w, 2, i]
                        closest = tx.max(
                            -geom_data[gb, 12 + i], tx.min(geom_data[gb, 12 + i], center)
                        )
                        extended_scratch[w, 4, i] = center
                        extended_scratch[w, 5, i] = closest - center
                        length2 = length2 + (closest - center) * (closest - center)
                    length = tx.sqrt(length2)
                    radius = geom_data[ga, 12]
                    if length - radius <= margin:
                        distance = length - radius
                        if length <= minimum:
                            closest = 2.0 * (
                                geom_data[gb, 12] + geom_data[gb, 13] + geom_data[gb, 14]
                            )
                            for face in range(6):
                                axis = face // 2
                                sign = 1.0 - 2.0 * (face % 2)
                                value = tx.abs(
                                    geom_data[gb, 12 + axis] + sign * extended_scratch[w, 4, axis]
                                )
                                if closest > value:
                                    closest = value
                                    for i in range(3):
                                        extended_scratch[w, 5, i] = 0.0
                                    extended_scratch[w, 5, axis] = sign
                            distance = -closest - radius
                            for i in range(3):
                                extended_scratch[w, 6, i] = (
                                    extended_scratch[w, 4, i]
                                    + extended_scratch[w, 5, i] * (radius - closest) * 0.5
                                )
                        else:
                            for i in range(3):
                                extended_scratch[w, 5, i] = extended_scratch[w, 5, i] / length
                                extended_scratch[w, 6, i] = extended_scratch[
                                    w, 4, i
                                ] + extended_scratch[w, 5, i] * (radius + 0.5 * distance)
                        if n < slots:
                            row = start + n
                            raw_rows[w, row, 0] = 1.0
                            raw_rows[w, row, 1] = distance
                            for i in range(3):
                                point = geom_world[w, gb, i]
                                normal = 0.0
                                for j in range(3):
                                    point = (
                                        point
                                        + geom_world[w, gb, 3 + 3 * i + j]
                                        * extended_scratch[w, 6, j]
                                    )
                                    normal = (
                                        normal
                                        + geom_world[w, gb, 3 + 3 * i + j]
                                        * extended_scratch[w, 5, j]
                                    )
                                raw_rows[w, row, 2 + i] = point
                                raw_rows[w, row, 5 + i] = normal
                            n = n + 1
                        else:
                            status[w] = 1

            elif ta == 6 and tb == 6:
                # Native fifteen-axis selection. Scratch 0:3 is R_a.T R_b.
                separated = 0
                code = -1
                cle1 = 0
                cle2 = 0
                inverse = 0
                penetration = margin
                for i in range(3):
                    penetration = penetration + 3.0 * (
                        geom_data[ga, 12 + i] + geom_data[gb, 12 + i]
                    )
                    pa = 0.0
                    pb = 0.0
                    for k in range(3):
                        delta = geom_world[w, gb, k] - geom_world[w, ga, k]
                        pa = pa + geom_world[w, ga, 3 + 3 * k + i] * delta
                        pb = pb - geom_world[w, gb, 3 + 3 * k + i] * delta
                    extended_scratch[w, 3, i] = pa
                    extended_scratch[w, 4, i] = pb
                    for j in range(3):
                        value = 0.0
                        for k in range(3):
                            value = (
                                value
                                + geom_world[w, ga, 3 + 3 * k + i]
                                * geom_world[w, gb, 3 + 3 * k + j]
                            )
                        extended_scratch[w, i, j] = value
                if not tx.isfinite(penetration):
                    status[w] = 2
                for i in range(3):
                    c1 = -tx.abs(extended_scratch[w, 3, i]) + geom_data[ga, 12 + i]
                    c2 = -tx.abs(extended_scratch[w, 4, i]) + geom_data[gb, 12 + i]
                    for j in range(3):
                        c1 = c1 + tx.abs(extended_scratch[w, i, j]) * geom_data[gb, 12 + j]
                        c2 = c2 + tx.abs(extended_scratch[w, j, i]) * geom_data[ga, 12 + j]
                    if not tx.isfinite(c1) or not tx.isfinite(c2):
                        status[w] = 2
                    if c1 < -margin or c2 < -margin:
                        separated = 1
                    if c1 < penetration:
                        penetration = c1
                        code = i
                        if extended_scratch[w, 3, i] < 0.0:
                            code = code + 3
                    if c2 < penetration:
                        penetration = c2
                        code = i + 6
                        if extended_scratch[w, 4, i] < 0.0:
                            code = code + 3
                for i in range(3):
                    for j in range(3):
                        length2 = 0.0
                        for k in range(3):
                            value = 0.0
                            if k == (i + 1) % 3:
                                value = -extended_scratch[w, (i + 2) % 3, j]
                            elif k == (i + 2) % 3:
                                value = extended_scratch[w, (i + 1) % 3, j]
                            extended_scratch[w, 5, k] = value
                            length2 = length2 + value * value
                        length = tx.sqrt(length2)
                        if length >= minimum:
                            center = 0.0
                            c3 = 0.0
                            for k in range(3):
                                value = extended_scratch[w, 5, k] / length
                                extended_scratch[w, 5, k] = value
                                center = center + extended_scratch[w, 3, k] * value
                                if k != i:
                                    c3 = c3 + geom_data[ga, 12 + k] * tx.abs(value)
                                if k != j:
                                    c3 = (
                                        c3
                                        + geom_data[gb, 12 + k]
                                        * tx.abs(extended_scratch[w, i, 3 - k - j])
                                        / length
                                    )
                            c3 = c3 - tx.abs(center)
                            if not tx.isfinite(c3):
                                status[w] = 2
                            if c3 < -margin:
                                separated = 1
                            # An exactly shared box axis makes this cross axis
                            # an already-tested face axis. Preserve the face
                            # manifold: FP32 roundoff must not turn a redundant
                            # axis into the different native edge-depth branch.
                            redundant = 0
                            for shared in range(3):
                                if shared != j and tx.abs(extended_scratch[w, i, shared]) == 1.0:
                                    aligned = 1
                                    for other in range(3):
                                        if other != i and extended_scratch[w, other, shared] != 0.0:
                                            aligned = 0
                                        if other != shared and extended_scratch[w, i, other] != 0.0:
                                            aligned = 0
                                    if aligned == 1:
                                        redundant = 1
                                if shared != i and tx.abs(extended_scratch[w, shared, j]) == 1.0:
                                    aligned = 1
                                    for other in range(3):
                                        if other != shared and extended_scratch[w, other, j] != 0.0:
                                            aligned = 0
                                        if other != j and extended_scratch[w, shared, other] != 0.0:
                                            aligned = 0
                                    if aligned == 1:
                                        redundant = 1
                            if (
                                redundant == 0 or penetration < 0.0
                            ) and c3 - penetration < -penetration * 1.0e-12:
                                penetration = c3
                                cle1 = 0
                                cle2 = 0
                                for k in range(3):
                                    if k != i:
                                        if (extended_scratch[w, 5, k] > 0.0) != (center < 0.0):
                                            cle1 = cle1 + 1 + k * (k + 1) // 2
                                    if k != j:
                                        first = (extended_scratch[w, i, 3 - k - j] > 0.0) != (
                                            center < 0.0
                                        )
                                        if first != ((k - j + 3) % 3 == 1):
                                            cle2 = cle2 + 1 + k * (k + 1) // 2
                                    extended_scratch[w, 15, k] = extended_scratch[w, 5, k]
                                code = 12 + 3 * i + j
                                inverse = 0
                                if center < 0.0:
                                    inverse = 1
                if separated == 0 and code >= 0:
                    # Choose native reference face, then rotate it onto local +Z.
                    face = code % 6
                    ref = ga
                    incident = gb
                    q1 = 0
                    q2 = 0
                    ax1 = 0
                    ax2 = 0
                    if code < 12:
                        if code >= 6:
                            ref = gb
                            incident = ga
                    else:
                        q1 = (code - 12) // 3
                        q2 = (code - 12) % 3
                        ax1 = 1
                        ax2 = 2
                        pax1 = 1
                        pax2 = 2
                        if q2 == 1:
                            ax1 = 0
                        elif q2 == 2:
                            ax2 = 0
                        if q1 == 1:
                            pax1 = 0
                        elif q1 == 2:
                            pax2 = 0
                        if tx.abs(extended_scratch[w, q1, ax1]) < tx.abs(
                            extended_scratch[w, q1, ax2]
                        ):
                            ax1 = ax2
                            ax2 = 3 - q2 - ax1
                        if tx.abs(extended_scratch[w, pax1, q2]) < tx.abs(
                            extended_scratch[w, pax2, q2]
                        ):
                            pax1 = pax2
                            pax2 = 3 - q1 - pax1
                        face = pax2
                        if _bit(cle1, pax2) == 0:
                            face = face + 3
                    for i in range(3):
                        for j in range(3):
                            extended_scratch[w, 6 + i, j] = 0.0
                    if face == 0:
                        extended_scratch[w, 6, 2] = -1.0
                        extended_scratch[w, 7, 1] = 1.0
                        extended_scratch[w, 8, 0] = 1.0
                    elif face == 1:
                        extended_scratch[w, 6, 0] = 1.0
                        extended_scratch[w, 7, 2] = -1.0
                        extended_scratch[w, 8, 1] = 1.0
                    elif face == 2:
                        extended_scratch[w, 6, 0] = 1.0
                        extended_scratch[w, 7, 1] = 1.0
                        extended_scratch[w, 8, 2] = 1.0
                    elif face == 3:
                        extended_scratch[w, 6, 2] = 1.0
                        extended_scratch[w, 7, 1] = 1.0
                        extended_scratch[w, 8, 0] = -1.0
                    elif face == 4:
                        extended_scratch[w, 6, 0] = 1.0
                        extended_scratch[w, 7, 2] = 1.0
                        extended_scratch[w, 8, 1] = -1.0
                    else:
                        extended_scratch[w, 6, 0] = -1.0
                        extended_scratch[w, 7, 1] = 1.0
                        extended_scratch[w, 8, 2] = -1.0
                    for i in range(3):
                        pos = 0.0
                        size = 0.0
                        rnormal = 0.0
                        for k in range(3):
                            relative = extended_scratch[w, 3, k]
                            if ref == gb:
                                relative = extended_scratch[w, 4, k]
                            pos = pos + extended_scratch[w, 6 + i, k] * relative
                            size = size + extended_scratch[w, 6 + i, k] * geom_data[ref, 12 + k]
                            if code >= 12:
                                rnormal = (
                                    rnormal
                                    + extended_scratch[w, 6 + i, k] * extended_scratch[w, 15, k]
                                )
                        extended_scratch[w, 12, i] = pos
                        extended_scratch[w, 13, i] = tx.abs(size)
                        extended_scratch[w, 14, i] = geom_data[incident, 12 + i]
                        extended_scratch[w, 16, i] = rnormal
                        extended_scratch[w, 47, i] = geom_world[w, ref, i]
                        for j in range(3):
                            value = 0.0
                            world = 0.0
                            for k in range(3):
                                relative = extended_scratch[w, k, j]
                                if ref == gb:
                                    relative = extended_scratch[w, j, k]
                                value = value + extended_scratch[w, 6 + i, k] * relative
                                world = (
                                    world
                                    + geom_world[w, ref, 3 + 3 * i + k]
                                    * extended_scratch[w, 6 + j, k]
                                )
                            extended_scratch[w, 9 + i, j] = value
                            extended_scratch[w, 44 + i, j] = world
                    lx = extended_scratch[w, 13, 0]
                    ly = extended_scratch[w, 13, 1]
                    hz = extended_scratch[w, 13, 2]
                    extended_scratch[w, 12, 2] = extended_scratch[w, 12, 2] - hz
                    if code < 12:
                        clcorner = 0
                        for i in range(3):
                            if extended_scratch[w, 11, i] < 0.0:
                                clcorner = clcorner + 1 + i * (i + 1) // 2
                        for k in range(3):
                            value = extended_scratch[w, 12, k]
                            for i in range(3):
                                value = value + extended_scratch[w, 9 + k, i] * extended_scratch[
                                    w, 14, i
                                ] * (2.0 * _bit(clcorner, i) - 1.0)
                            extended_scratch[w, 17, k] = value
                            extended_scratch[w, 18, k] = 0.0
                            extended_scratch[w, 19, k] = 0.0
                        directions = 0
                        for i in range(3):
                            if tx.abs(extended_scratch[w, 11, i]) < 0.5:
                                if directions < 2:
                                    for k in range(3):
                                        extended_scratch[w, 18 + directions, k] = (
                                            extended_scratch[w, 9 + k, i]
                                            * extended_scratch[w, 14, i]
                                            * (2.0 - 4.0 * _bit(clcorner, i))
                                        )
                                    directions = directions + 1
                                else:
                                    status[w] = 1
                        linecount = 0
                        if directions == 1:
                            linecount = 1
                        elif directions == 2:
                            linecount = 4
                        for line in range(linecount):
                            for k in range(3):
                                base = extended_scratch[w, 17, k]
                                direction = extended_scratch[w, 18, k]
                                if line == 1 or line == 2:
                                    direction = extended_scratch[w, 19, k]
                                if line == 2:
                                    base = base + extended_scratch[w, 18, k]
                                elif line == 3:
                                    base = base + extended_scratch[w, 19, k]
                                extended_scratch[w, 28 + line, k] = base
                                extended_scratch[w, 32 + line, k] = direction
                            for q in range(2):
                                a = extended_scratch[w, 28 + line, q]
                                b = extended_scratch[w, 32 + line, q]
                                c = extended_scratch[w, 28 + line, 1 - q]
                                d = extended_scratch[w, 32 + line, 1 - q]
                                if tx.abs(b) > minimum:
                                    for side in range(2):
                                        boundary = extended_scratch[w, 13, q] * (-1.0 + 2.0 * side)
                                        t = (boundary - a) / b
                                        other = c + d * t
                                        if (
                                            t >= 0.0
                                            and t <= 1.0
                                            and tx.abs(other) <= extended_scratch[w, 13, 1 - q]
                                        ):
                                            if n < slots and n < 24:
                                                for k in range(3):
                                                    raw_rows[w, start + n, 2 + k] = (
                                                        extended_scratch[w, 28 + line, k]
                                                        + extended_scratch[w, 32 + line, k] * t
                                                    )
                                                n = n + 1
                                            else:
                                                status[w] = 1
                        a = extended_scratch[w, 18, 0]
                        b = extended_scratch[w, 19, 0]
                        c = extended_scratch[w, 18, 1]
                        d = extended_scratch[w, 19, 1]
                        det = a * d - b * c
                        if directions == 2:
                            if det == 0.0:
                                status[w] = 2
                            else:
                                for corner in range(4):
                                    x = lx * (-1.0 + 2.0 * (corner // 2))
                                    y = ly * (-1.0 + 2.0 * (corner % 2))
                                    dx = x - extended_scratch[w, 17, 0]
                                    dy = y - extended_scratch[w, 17, 1]
                                    u = (dx * d - dy * b) / det
                                    v = (dy * a - dx * c) / det
                                    if u > 0.0 and u < 1.0 and v > 0.0 and v < 1.0:
                                        if n < slots and n < 24:
                                            raw_rows[w, start + n, 2] = x
                                            raw_rows[w, start + n, 3] = y
                                            raw_rows[w, start + n, 4] = (
                                                extended_scratch[w, 17, 2]
                                                + u * extended_scratch[w, 18, 2]
                                                + v * extended_scratch[w, 19, 2]
                                            )
                                            n = n + 1
                                        else:
                                            status[w] = 1
                        corners = 1 + directions
                        if directions == 2:
                            corners = 4
                        for corner in range(corners):
                            for k in range(3):
                                value = extended_scratch[w, 17, k]
                                if corner == 1 or corner == 3:
                                    value = value + extended_scratch[w, 18, k]
                                if corner == 2 or corner == 3:
                                    value = value + extended_scratch[w, 19, k]
                                extended_scratch[w, 48, k] = value
                            inside = 1
                            if corner != 0:
                                if (
                                    tx.abs(extended_scratch[w, 48, 0]) >= lx
                                    or tx.abs(extended_scratch[w, 48, 1]) >= ly
                                ):
                                    inside = 0
                            if inside == 1:
                                if n < slots and n < 24:
                                    for k in range(3):
                                        raw_rows[w, start + n, 2 + k] = extended_scratch[w, 48, k]
                                    n = n + 1
                                else:
                                    status[w] = 1
                        for i in range(n):
                            depth = raw_rows[w, start + i, 4]
                            if depth <= margin:
                                # MuJoCo 3.3.7 face branch publishes HALF depth.
                                raw_rows[w, start + i, 0] = 1.0
                                raw_rows[w, start + i, 1] = 0.5 * depth
                                raw_rows[w, start + i, 4] = 0.5 * depth
                        for i in range(3):
                            extended_scratch[w, 16, i] = 0.0
                        extended_scratch[w, 16, 2] = 1.0
                        if ref == gb:
                            extended_scratch[w, 16, 2] = -1.0

                    else:
                        # Edge-selected face: four original and projected vertices.
                        for vertex in range(4):
                            sign1 = 2.0 * _bit(cle2, ax1) - 1.0
                            if vertex >= 2:
                                sign1 = -sign1
                            sign2 = 2.0 * _bit(cle2, ax2) - 1.0
                            signq = 1.0 - 2.0 * (vertex % 2)
                            for i in range(3):
                                value = (
                                    extended_scratch[w, 12, i]
                                    + extended_scratch[w, 9 + i, ax1]
                                    * geom_data[gb, 12 + ax1]
                                    * sign1
                                    + extended_scratch[w, 9 + i, ax2]
                                    * geom_data[gb, 12 + ax2]
                                    * sign2
                                    + extended_scratch[w, 9 + i, q2]
                                    * geom_data[gb, 12 + q2]
                                    * signq
                                )
                                extended_scratch[w, 20 + vertex, i] = value
                        rnz = extended_scratch[w, 16, 2]
                        if tx.abs(rnz) >= minimum:
                            innorm = 1.0 / rnz
                            if inverse == 1:
                                innorm = -innorm
                            for vertex in range(4):
                                scale = -extended_scratch[w, 20 + vertex, 2] / rnz
                                for i in range(3):
                                    extended_scratch[w, 24 + vertex, i] = (
                                        extended_scratch[w, 20 + vertex, i]
                                        + extended_scratch[w, 16, i] * scale
                                    )
                            for i in range(3):
                                extended_scratch[w, 17, i] = extended_scratch[w, 24, i]
                                extended_scratch[w, 18, i] = (
                                    extended_scratch[w, 25, i] - extended_scratch[w, 24, i]
                                )
                                extended_scratch[w, 19, i] = (
                                    extended_scratch[w, 26, i] - extended_scratch[w, 24, i]
                                )
                            for line in range(4):
                                for i in range(3):
                                    base = extended_scratch[w, 24, i]
                                    direction = (
                                        extended_scratch[w, 25, i] - extended_scratch[w, 24, i]
                                    )
                                    original = extended_scratch[w, 20, i]
                                    original_direction = (
                                        extended_scratch[w, 21, i] - extended_scratch[w, 20, i]
                                    )
                                    if line == 1 or line == 2:
                                        direction = (
                                            extended_scratch[w, 26, i] - extended_scratch[w, 24, i]
                                        )
                                        original_direction = (
                                            extended_scratch[w, 22, i] - extended_scratch[w, 20, i]
                                        )
                                    if line == 2:
                                        base = extended_scratch[w, 25, i]
                                        original = extended_scratch[w, 21, i]
                                    elif line == 3:
                                        base = extended_scratch[w, 26, i]
                                        original = extended_scratch[w, 22, i]
                                    extended_scratch[w, 28 + line, i] = base
                                    extended_scratch[w, 32 + line, i] = direction
                                    extended_scratch[w, 36 + line, i] = original
                                    extended_scratch[w, 40 + line, i] = original_direction
                                for q in range(2):
                                    a = extended_scratch[w, 28 + line, q]
                                    b = extended_scratch[w, 32 + line, q]
                                    c = extended_scratch[w, 28 + line, 1 - q]
                                    d = extended_scratch[w, 32 + line, 1 - q]
                                    if tx.abs(b) > minimum:
                                        for side in range(2):
                                            boundary = extended_scratch[w, 13, q] * (
                                                -1.0 + 2.0 * side
                                            )
                                            t = (boundary - a) / b
                                            other = c + d * t
                                            depth = (
                                                extended_scratch[w, 36 + line, 2]
                                                + extended_scratch[w, 40 + line, 2] * t
                                            ) * innorm
                                            if (
                                                t >= 0.0
                                                and t <= 1.0
                                                and tx.abs(other) <= extended_scratch[w, 13, 1 - q]
                                                and depth <= margin
                                            ):
                                                if n < slots and n < 24:
                                                    row = start + n
                                                    raw_rows[w, row, 0] = 1.0
                                                    raw_rows[w, row, 1] = depth
                                                    for i in range(3):
                                                        value = (
                                                            0.5 * extended_scratch[w, 36 + line, i]
                                                            + 0.5
                                                            * t
                                                            * extended_scratch[w, 40 + line, i]
                                                        )
                                                        if i == q:
                                                            value = value + 0.5 * boundary
                                                        elif i == 1 - q:
                                                            value = value + 0.5 * other
                                                        raw_rows[w, row, 2 + i] = value
                                                    n = n + 1
                                                else:
                                                    status[w] = 1
                            nl = n
                            a = extended_scratch[w, 18, 0]
                            b = extended_scratch[w, 19, 0]
                            c = extended_scratch[w, 18, 1]
                            d = extended_scratch[w, 19, 1]
                            denominator = a * d - b * c
                            for corner in range(4):
                                # The fixed CPU source reuses c1 after each distance test.
                                x = lx * (-1.0 + 2.0 * (corner // 2))
                                y = ly * (-1.0 + 2.0 * (corner % 2))
                                dx = x - extended_scratch[w, 17, 0]
                                dy = y - extended_scratch[w, 17, 1]
                                if denominator == 0.0:
                                    status[w] = 2
                                else:
                                    u = (dx * d - dy * b) / denominator
                                    v = (dy * a - dx * c) / denominator
                                    accept = 1
                                    if nl == 0:
                                        if (u < 0.0 or u > 1.0) and (v < 0.0 or v > 1.0):
                                            accept = 0
                                    else:
                                        if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
                                            accept = 0
                                    if accept == 1:
                                        u = tx.max(0.0, tx.min(1.0, u))
                                        v = tx.max(0.0, tx.min(1.0, v))
                                        distance2 = 0.0
                                        for i in range(3):
                                            value = (
                                                extended_scratch[w, 20, i] * (1.0 - u - v)
                                                + extended_scratch[w, 21, i] * u
                                                + extended_scratch[w, 22, i] * v
                                            )
                                            extended_scratch[w, 48, i] = value
                                            target = 0.0
                                            if i == 0:
                                                target = x
                                            elif i == 1:
                                                target = y
                                            distance2 = distance2 + (target - value) * (
                                                target - value
                                            )
                                        denominator = distance2
                                        if (
                                            extended_scratch[w, 48, 2] <= 0.0
                                            or distance2 <= margin * margin
                                        ):
                                            if n < slots and n < 24:
                                                row = start + n
                                                depth = tx.sqrt(distance2)
                                                if extended_scratch[w, 48, 2] < 0.0:
                                                    depth = -depth
                                                raw_rows[w, row, 0] = 1.0
                                                raw_rows[w, row, 1] = depth
                                                raw_rows[w, row, 2] = 0.5 * (
                                                    x + extended_scratch[w, 48, 0]
                                                )
                                                raw_rows[w, row, 3] = 0.5 * (
                                                    y + extended_scratch[w, 48, 1]
                                                )
                                                raw_rows[w, row, 4] = (
                                                    0.5 * extended_scratch[w, 48, 2]
                                                )
                                                n = n + 1
                                            else:
                                                status[w] = 1
                            nf = n
                            for vertex in range(4):
                                x = extended_scratch[w, 24 + vertex, 0]
                                y = extended_scratch[w, 24 + vertex, 1]
                                accept = 1
                                if nl == 0:
                                    if nf != 0:
                                        if (x < -lx or x > lx) and (y < -ly or y > ly):
                                            accept = 0
                                else:
                                    if x < -lx or x > lx or y < -ly or y > ly:
                                        accept = 0
                                if accept == 1:
                                    distance2 = 0.0
                                    for i in range(2):
                                        projected = extended_scratch[w, 24 + vertex, i]
                                        limited = tx.max(
                                            -extended_scratch[w, 13, i],
                                            tx.min(extended_scratch[w, 13, i], projected),
                                        )
                                        distance2 = distance2 + (projected - limited) * (
                                            projected - limited
                                        )
                                        extended_scratch[w, 48, i] = 0.5 * (
                                            limited + extended_scratch[w, 20 + vertex, i]
                                        )
                                    z = extended_scratch[w, 20 + vertex, 2]
                                    distance2 = distance2 + z * innorm * z * innorm
                                    if z <= 0.0 or distance2 <= margin * margin:
                                        if n < slots and n < 24:
                                            row = start + n
                                            depth = tx.sqrt(distance2)
                                            if z < 0.0:
                                                depth = -depth
                                            raw_rows[w, row, 0] = 1.0
                                            raw_rows[w, row, 1] = depth
                                            raw_rows[w, row, 2] = extended_scratch[w, 48, 0]
                                            raw_rows[w, row, 3] = extended_scratch[w, 48, 1]
                                            raw_rows[w, row, 4] = 0.5 * z
                                            n = n + 1
                                        else:
                                            status[w] = 1
                        if inverse == 1:
                            for i in range(3):
                                extended_scratch[w, 16, i] = -extended_scratch[w, 16, i]
                    # Transform all selected candidates, then native driver postprocess.
                    for candidate in range(n):
                        row = start + candidate
                        for field in range(5):
                            if not tx.isfinite(raw_rows[w, row, field]):
                                status[w] = 2
                        for i in range(3):
                            extended_scratch[w, 48, i] = raw_rows[w, row, 2 + i]
                        extended_scratch[w, 48, 2] = extended_scratch[w, 48, 2] + hz
                        for i in range(3):
                            point = extended_scratch[w, 47, i]
                            normal = 0.0
                            for j in range(3):
                                point = (
                                    point
                                    + extended_scratch[w, 44 + i, j] * extended_scratch[w, 48, j]
                                )
                                normal = (
                                    normal
                                    + extended_scratch[w, 44 + i, j] * extended_scratch[w, 16, j]
                                )
                            raw_rows[w, row, 2 + i] = point
                            raw_rows[w, row, 5 + i] = normal
                        out1 = 0
                        out2 = 0
                        for which in range(2):
                            g = ga
                            if which == 1:
                                g = gb
                            inside = 1
                            outside = 0
                            for i in range(3):
                                local = 0.0
                                for j in range(3):
                                    local = local + geom_world[w, g, 3 + 3 * j + i] * (
                                        raw_rows[w, row, 2 + j] - geom_world[w, g, j]
                                    )
                                size = geom_data[g, 12 + i] + margin
                                if tx.abs(local) > 1.01 * size:
                                    outside = 1
                                if tx.abs(local) >= size / 1.01:
                                    inside = 0
                            result = 0
                            if outside == 1:
                                result = 1
                            elif inside == 1:
                                result = -1
                            if which == 0:
                                out1 = result
                            else:
                                out2 = result
                        if (out1 == 1 and out2 != -1) or (out2 == 1 and out1 != -1):
                            raw_rows[w, row, 0] = 0.0
                    # Exact equality removes the EARLIER live duplicate, preserving order.
                    for i in range(n):
                        if raw_rows[w, start + i, 0] == 1.0:
                            duplicate = 0
                            for j in range(i + 1, n):
                                if duplicate == 0 and raw_rows[w, start + j, 0] == 1.0:
                                    same = 1
                                    for k in range(3):
                                        if (
                                            raw_rows[w, start + i, 2 + k]
                                            != raw_rows[w, start + j, 2 + k]
                                        ):
                                            same = 0
                                    if same == 1:
                                        duplicate = 1
                            if duplicate == 1:
                                raw_rows[w, start + i, 0] = 0.0
                    kept = 0
                    for i in range(n):
                        if raw_rows[w, start + i, 0] == 1.0:
                            if kept != i:
                                for k in range(11):
                                    raw_rows[w, start + kept, k] = raw_rows[w, start + i, k]
                            kept = kept + 1
                    for i in range(kept, slots):
                        for k in range(11):
                            raw_rows[w, start + i, k] = 0.0
            # Every published primitive value must be finite. Fail without hiding it.
            if ta >= 3:
                for feature in range(slots):
                    row = start + feature
                    for k in range(11):
                        if not tx.isfinite(raw_rows[w, row, k]):
                            status[w] = 2
                for scratch_row in range(64):
                    for scratch_column in range(3):
                        if not tx.isfinite(extended_scratch[w, scratch_row, scratch_column]):
                            status[w] = 2


extended_contacts_other = tx.kernel(extended_contacts)
extended_contacts = tx.kernel(extended_contacts)
