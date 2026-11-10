# Copyright 2021 DeepMind Technologies Limited
# Copyright 2016 Svetoslav Kolev
# Copyright 2025 The Newton Developers
# Copyright 2026 Phymeo contributors
# Apache-2.0; see LICENSE and NOTICE. TYX translation of MuJoCo 3.3.7
# engine_collision_primitive.c/engine_collision_box.c and MuJoCo-Warp 0.0.2
# contact/constraint field formulas; CPU manifold and degenerate rules prevail.

"""Six native primitive pair manifolds and current complete normal Jacobians."""

import tyx as tx


def contacts(
    count: tx.i32,
    nv: tx.i32,
    ngeom: tx.i32,
    npair: tx.i32,
    nslot: tx.i32,
    contact_capacity: tx.i32,
    constraint_capacity: tx.i32,
    h: tx.f32,
    geom_ids: tx.ndarray(tx.i32, 2),
    geom_data: tx.ndarray(tx.f32, 2),
    pair_ids: tx.ndarray(tx.i32, 2),
    pair_law: tx.ndarray(tx.f32, 2),
    motion: tx.ndarray(tx.f32, 4),
    rotation: tx.ndarray(tx.f32, 4),
    jacobian: tx.ndarray(tx.f32, 4),
    velocity: tx.ndarray(tx.f32, 2),
    geom_world: tx.ndarray(tx.f32, 3),
    rows: tx.ndarray(tx.f32, 3),
    normal_jacobian: tx.ndarray(tx.f32, 3),
    scratch: tx.ndarray(tx.f32, 3),
    diagnostics: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        present_count = 0
        active_count = 0
        for r in range(nslot):
            for k in range(32):
                rows[w, r, k] = 0.0
            for d in range(nv):
                normal_jacobian[w, r, d] = 0.0
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
        for p in range(npair):
            ga = pair_ids[p, 0]
            gb = pair_ids[p, 1]
            ba = geom_ids[ga, 0]
            bb = geom_ids[gb, 0]
            ta = geom_ids[ga, 1]
            tb = geom_ids[gb, 1]
            margin = pair_law[p, 0]
            gap = pair_law[p, 1]
            minimum = pair_law[p, 10]
            plane_box_count = 0
            for feature in range(pair_ids[p, 3]):
                row = pair_ids[p, 2] + feature
                for i in range(3):
                    scratch[w, 0, i] = geom_world[w, ga, i]
                    scratch[w, 1, i] = geom_world[w, gb, i]
                    scratch[w, 2, i] = 0.0
                    scratch[w, 3, i] = 0.0
                    scratch[w, 4, i] = 0.0
                distance = 0.0
                eligible = 1
                if ta == 0:
                    # Plane is first by native type order; native normal is its Z axis.
                    for i in range(3):
                        scratch[w, 2, i] = geom_world[w, ga, 5 + 3 * i]
                    radius = 0.0
                    if tb == 2 or tb == 3:
                        radius = geom_data[gb, 12]
                    if tb == 3:
                        sign = 1.0
                        if feature == 1:
                            sign = -1.0
                        for i in range(3):
                            axis = geom_world[w, gb, 5 + 3 * i]
                            scratch[w, 1, i] = scratch[w, 1, i] + sign * geom_data[gb, 13] * axis
                            scratch[w, 4, i] = axis
                    local_distance = 0.0
                    if tb == 6:
                        # Keep native corner identity; choose first four qualifying corners.
                        for j in range(3):
                            sign = 1.0
                            if (feature // (1 << j)) % 2 == 0:
                                sign = -1.0
                            scratch[w, 5, j] = sign * geom_data[gb, 12 + j]
                        for i in range(3):
                            corner = 0.0
                            for j in range(3):
                                corner = corner + geom_world[w, gb, 3 + 3 * i + j] * scratch[w, 5, j]
                            scratch[w, 1, i] = scratch[w, 1, i] + corner
                            local_distance = local_distance + scratch[w, 2, i] * corner
                        if local_distance > 0.0 or plane_box_count >= 4:
                            eligible = 0
                    distance = -radius
                    for i in range(3):
                        distance = distance + (scratch[w, 1, i] - scratch[w, 0, i]) * scratch[w, 2, i]
                    for i in range(3):
                        scratch[w, 3, i] = scratch[w, 1, i] - (radius + 0.5 * distance) * scratch[w, 2, i]
                    if tb == 6 and eligible == 1 and distance <= margin:
                        plane_box_count = plane_box_count + 1
                elif tb == 2 or tb == 3:
                    # Native sphere/capsule projection has no artificial denominator epsilon.
                    if tb == 3:
                        projection = 0.0
                        for i in range(3):
                            projection = projection + (scratch[w, 0, i] - scratch[w, 1, i]) * geom_world[w, gb, 5 + 3 * i]
                        projection = tx.max(-geom_data[gb, 13], tx.min(geom_data[gb, 13], projection))
                        for i in range(3):
                            scratch[w, 1, i] = scratch[w, 1, i] + projection * geom_world[w, gb, 5 + 3 * i]
                    length2 = 0.0
                    for i in range(3):
                        value = scratch[w, 1, i] - scratch[w, 0, i]
                        scratch[w, 2, i] = value
                        length2 = length2 + value * value
                    length = tx.sqrt(length2)
                    distance = length - geom_data[ga, 12] - geom_data[gb, 12]
                    if length < minimum:
                        length2 = 0.0
                        for i in range(3):
                            j = (i + 1) % 3
                            k = (i + 2) % 3
                            value = geom_world[w, ga, 5 + 3 * j] * geom_world[w, gb, 5 + 3 * k] - geom_world[w, ga, 5 + 3 * k] * geom_world[w, gb, 5 + 3 * j]
                            scratch[w, 2, i] = value
                            length2 = length2 + value * value
                        length = tx.sqrt(length2)
                    if length < minimum:
                        for i in range(3):
                            scratch[w, 2, i] = 0.0
                        scratch[w, 2, 0] = 1.0
                    else:
                        for i in range(3):
                            scratch[w, 2, i] = scratch[w, 2, i] / length
                    for i in range(3):
                        scratch[w, 3, i] = scratch[w, 0, i] + (geom_data[ga, 12] + 0.5 * distance) * scratch[w, 2, i]
                else:
                    # Sphere-box: closest feature outside; first nearest native face inside.
                    length2 = 0.0
                    for i in range(3):
                        local = 0.0
                        for j in range(3):
                            local = local + geom_world[w, gb, 3 + 3 * j + i] * (scratch[w, 0, j] - scratch[w, 1, j])
                        scratch[w, 5, i] = local
                        clamp = tx.max(-geom_data[gb, 12 + i], tx.min(geom_data[gb, 12 + i], local))
                        scratch[w, 6, i] = clamp
                        scratch[w, 7, i] = clamp - local
                        length2 = length2 + (clamp - local) * (clamp - local)
                    length = tx.sqrt(length2)
                    if length <= minimum:
                        closest = 2.0 * (geom_data[gb, 12] + geom_data[gb, 13] + geom_data[gb, 14])
                        nearest_face = 0
                        for face in range(6):
                            sign = -1.0
                            if face % 2 == 1:
                                sign = 1.0
                            candidate = tx.abs(sign * geom_data[gb, 12 + face // 2] - scratch[w, 5, face // 2])
                            if candidate < closest:
                                closest = candidate
                                nearest_face = face
                        for i in range(3):
                            scratch[w, 7, i] = 0.0
                        sign = 1.0
                        if nearest_face % 2 == 1:
                            sign = -1.0
                        scratch[w, 7, nearest_face // 2] = sign
                        distance = -closest - geom_data[ga, 12]
                        for i in range(3):
                            scratch[w, 6, i] = scratch[w, 5, i] + scratch[w, 7, i] * (geom_data[ga, 12] - closest) * 0.5
                    else:
                        distance = length - geom_data[ga, 12]
                        for i in range(3):
                            scratch[w, 7, i] = scratch[w, 7, i] / length
                            scratch[w, 6, i] = 0.5 * (scratch[w, 6, i] + scratch[w, 5, i] + geom_data[ga, 12] * scratch[w, 7, i])
                    for i in range(3):
                        scratch[w, 2, i] = 0.0
                        scratch[w, 3, i] = scratch[w, 1, i]
                        for j in range(3):
                            scratch[w, 2, i] = scratch[w, 2, i] + geom_world[w, gb, 3 + 3 * i + j] * scratch[w, 7, j]
                            scratch[w, 3, i] = scratch[w, 3, i] + geom_world[w, gb, 3 + 3 * i + j] * scratch[w, 6, j]
                if not tx.isfinite(distance):
                    status[w] = 2
                if eligible == 1 and distance <= margin:
                    present_count = present_count + 1
                    rows[w, row, 0] = 1.0
                    rows[w, row, 2] = distance
                    rows[w, row, 21] = margin - gap
                    rows[w, row, 22] = margin
                    rows[w, row, 23] = gap
                    # Exact native frame construction, including capsule-axis tangent seed.
                    length2 = 0.0
                    tangent2 = 0.0
                    for i in range(3):
                        length2 = length2 + scratch[w, 2, i] * scratch[w, 2, i]
                        tangent2 = tangent2 + scratch[w, 4, i] * scratch[w, 4, i]
                    length = tx.sqrt(length2)
                    if not tx.isfinite(length) or length < minimum:
                        status[w] = 2
                        length = 1.0
                    for i in range(3):
                        scratch[w, 2, i] = scratch[w, 2, i] / length
                    if tangent2 < 0.25:
                        for i in range(3):
                            scratch[w, 4, i] = 0.0
                        if -0.5 < scratch[w, 2, 1] and scratch[w, 2, 1] < 0.5:
                            scratch[w, 4, 1] = 1.0
                        else:
                            scratch[w, 4, 2] = 1.0
                    projection = 0.0
                    for i in range(3):
                        projection = projection + scratch[w, 2, i] * scratch[w, 4, i]
                    length2 = 0.0
                    for i in range(3):
                        scratch[w, 4, i] = scratch[w, 4, i] - projection * scratch[w, 2, i]
                        length2 = length2 + scratch[w, 4, i] * scratch[w, 4, i]
                    length = tx.sqrt(length2)
                    if length < minimum:
                        for i in range(3):
                            scratch[w, 4, i] = 0.0
                        scratch[w, 4, 0] = 1.0
                    else:
                        for i in range(3):
                            scratch[w, 4, i] = scratch[w, 4, i] / length
                    for i in range(3):
                        j = (i + 1) % 3
                        k = (i + 2) % 3
                        rows[w, row, 9 + i] = scratch[w, 3, i]
                        rows[w, row, 12 + i] = scratch[w, 2, i]
                        rows[w, row, 15 + i] = scratch[w, 4, i]
                        rows[w, row, 18 + i] = scratch[w, 2, j] * scratch[w, 4, k] - scratch[w, 2, k] * scratch[w, 4, j]
                    row_velocity = 0.0
                    for d in range(nv):
                        value = 0.0
                        for i in range(3):
                            j = (i + 1) % 3
                            k = (i + 2) % 3
                            ja = jacobian[w, ba, i, d] + jacobian[w, ba, 3 + j, d] * (scratch[w, 3, k] - motion[w, ba, 5, k]) - jacobian[w, ba, 3 + k, d] * (scratch[w, 3, j] - motion[w, ba, 5, j])
                            jb = jacobian[w, bb, i, d] + jacobian[w, bb, 3 + j, d] * (scratch[w, 3, k] - motion[w, bb, 5, k]) - jacobian[w, bb, 3 + k, d] * (scratch[w, 3, j] - motion[w, bb, 5, j])
                            value = value + scratch[w, 2, i] * (jb - ja)
                        normal_jacobian[w, row, d] = value
                        row_velocity = row_velocity + value * velocity[w, d]
                    rows[w, row, 7] = row_velocity
                    position = distance - (margin - gap)
                    if position < 0.0:
                        active_count = active_count + 1
                        rows[w, row, 1] = 1.0
                        dmin = tx.max(pair_law[p, 11], tx.min(pair_law[p, 12], pair_law[p, 5]))
                        dmax = tx.max(pair_law[p, 11], tx.min(pair_law[p, 12], pair_law[p, 6]))
                        width = tx.max(0.0, pair_law[p, 7])
                        midpoint = tx.max(pair_law[p, 11], tx.min(pair_law[p, 12], pair_law[p, 8]))
                        power = tx.max(1.0, pair_law[p, 9])
                        impedance = dmax
                        if dmin == dmax or width <= minimum:
                            impedance = 0.5 * (dmin + dmax)
                        else:
                            x = tx.abs(position) / width
                            if x < 1.0:
                                y = 0.0
                                if x < midpoint:
                                    y = tx.pow(x, power) / tx.pow(midpoint, power - 1.0)
                                else:
                                    y = 1.0 - tx.pow(1.0 - x, power) / tx.pow(1.0 - midpoint, power - 1.0)
                                impedance = dmin + y * (dmax - dmin)
                        stiffness = 0.0
                        damping = 0.0
                        if pair_law[p, 3] > 0.0:
                            timeconst = tx.max(pair_law[p, 3], 2.0 * h)
                            stiffness = 1.0 / tx.max(minimum, dmax * dmax * timeconst * timeconst * pair_law[p, 4] * pair_law[p, 4])
                            damping = 2.0 / tx.max(minimum, dmax * timeconst)
                        else:
                            stiffness = -pair_law[p, 3] / (dmax * dmax)
                            damping = -pair_law[p, 4] / dmax
                        regularization = tx.max(minimum, pair_law[p, 2] * (1.0 - impedance) / impedance)
                        rows[w, row, 4] = regularization
                        rows[w, row, 5] = 1.0 / regularization
                        rows[w, row, 6] = -stiffness * impedance * position - damping * row_velocity
                        rows[w, row, 8] = impedance
                    for k in range(24):
                        if not tx.isfinite(rows[w, row, k]):
                            status[w] = 2
                    for d in range(nv):
                        if not tx.isfinite(normal_jacobian[w, row, d]):
                            status[w] = 2
        diagnostics[w, 4] = present_count
        diagnostics[w, 5] = active_count
        if present_count > contact_capacity or active_count > constraint_capacity:
            status[w] = 1


contacts_other = tx.kernel(contacts)
contacts = tx.kernel(contacts)
