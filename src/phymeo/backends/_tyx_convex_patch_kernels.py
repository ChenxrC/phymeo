"""Device patches for certified points, one parallel segment, and one coincident face.

Each lane owns one pair and reads the distance kernel's resolved normal. A point
is two sphere surfaces, or one sphere against a capsule axis, a hull vertex, or a
flat support face. The sphere may be either shape. A capsule endpoint against
one hull edge is also a point. Two capsule axes that are not parallel are a point.
A segment is the overlap of two parallel finite axes, or the part of one
capsule axis that projects into a single flat support face. The flat face may
be either shape. An area is the
shared support face when the two convex faces coincide, or the inner face
when one flat support polygon lies inside the other. Partially intersecting convex support faces use bounded device polygon clipping, with full original material barycentrics on both sides.
Anything else stays unresolved and writes no sites. No load is solved and no
nearest face is searched on the host.
"""

import tyx as tx

from ._tyx_convex_support import dot3


def clear_convex_patch(
    pairs: tx.i32,
    sites: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    weight: tx.ndarray(tx.f32, 1),
    gap: tx.ndarray(tx.f32, 1),
    normal: tx.ndarray(tx.f32, 2),
    common: tx.ndarray(tx.f32, 2),
    material_ids: tx.ndarray(tx.i32, 3),
    barycentric: tx.ndarray(tx.f32, 3),
    surface: tx.ndarray(tx.f32, 3),
    skin: tx.ndarray(tx.f32, 3),
    material_count: tx.ndarray(tx.i32, 2),
):
    for pair in tx.range(pairs):
        for column in tx.static_range(8):
            meta[pair, column] = 0
    for site in tx.range(sites):
        weight[site] = 0.0
        gap[site] = 0.0
        for axis in tx.static_range(3):
            normal[site, axis] = 0.0
            common[site, axis] = 0.0
        for side in tx.static_range(2):
            material_count[site, side] = 0
            for slot in tx.static_range(4):
                material_ids[site, side, slot] = -1
                barycentric[site, side, slot] = 0.0
            for axis in tx.static_range(3):
                surface[site, side, axis] = 0.0
                skin[site, side, axis] = 0.0


def fill_convex_patch(
    pair_index: tx.i32,
    site_base: tx.i32,
    site_capacity: tx.i32,
    tags: tx.ndarray(tx.i32, 2),
    original_ids: tx.ndarray(tx.i32, 1),
    radii: tx.ndarray(tx.f32, 1),
    points: tx.ndarray(tx.f32, 2),
    pairs: tx.ndarray(tx.i32, 2),
    distance_meta: tx.ndarray(tx.i32, 2),
    distance_gap: tx.ndarray(tx.f32, 2),
    distance_normal: tx.ndarray(tx.f32, 2),
    work: tx.ndarray(tx.f32, 3),
    work_i: tx.ndarray(tx.i32, 3),
    meta: tx.ndarray(tx.i32, 2),
    weight: tx.ndarray(tx.f32, 1),
    gap: tx.ndarray(tx.f32, 1),
    normal: tx.ndarray(tx.f32, 2),
    common: tx.ndarray(tx.f32, 2),
    material_ids: tx.ndarray(tx.i32, 3),
    barycentric: tx.ndarray(tx.f32, 3),
    surface: tx.ndarray(tx.f32, 3),
    skin: tx.ndarray(tx.f32, 3),
    material_count: tx.ndarray(tx.i32, 2),
):
    for lane in tx.range(1):
        out = pair_index
        shape_a = pairs[out, 0]
        shape_b = pairs[out, 1]
        a0 = tags[shape_a, 1]
        a1 = tags[shape_a, 2]
        b0 = tags[shape_b, 1]
        b1 = tags[shape_b, 2]
        kind_a = tags[shape_a, 0]
        kind_b = tags[shape_b, 0]
        radius_a = radii[shape_a]
        radius_b = radii[shape_b]
        nx = distance_normal[out, 0]
        ny = distance_normal[out, 1]
        nz = distance_normal[out, 2]
        query_gap = distance_gap[out, 0]
        scale = distance_gap[out, 3]
        status = 2
        dimension = -1
        count = 0
        if distance_meta[out, 0] == 1 and scale >= 1.0:
            if kind_a == 0 and kind_b == 0 and a1 == a0 + 1 and b1 == b0 + 1:
                ax = points[a0, 0]
                ay = points[a0, 1]
                az = points[a0, 2]
                bx = points[b0, 0]
                by = points[b0, 1]
                bz = points[b0, 2]
                s0x = ax + radius_a * nx
                s0y = ay + radius_a * ny
                s0z = az + radius_a * nz
                s1x = bx - radius_b * nx
                s1y = by - radius_b * ny
                s1z = bz - radius_b * nz
                signed = dot3(s1x - s0x, s1y - s0y, s1z - s0z, nx, ny, nz)
                if tx.abs(signed - query_gap) <= 1e-4 * scale and site_base + 1 <= site_capacity:
                    status = 1
                    dimension = 0
                    count = 1
                    weight[site_base] = 1.0
                    gap[site_base] = signed
                    normal[site_base, 0] = nx
                    normal[site_base, 1] = ny
                    normal[site_base, 2] = nz
                    common[site_base, 0] = (s0x + s1x) * 0.5
                    common[site_base, 1] = (s0y + s1y) * 0.5
                    common[site_base, 2] = (s0z + s1z) * 0.5
                    material_count[site_base, 0] = 1
                    material_count[site_base, 1] = 1
                    material_ids[site_base, 0, 0] = original_ids[a0]
                    material_ids[site_base, 0, 1] = -1
                    material_ids[site_base, 1, 0] = original_ids[b0]
                    material_ids[site_base, 1, 1] = -1
                    barycentric[site_base, 0, 0] = 1.0
                    barycentric[site_base, 0, 1] = 0.0
                    barycentric[site_base, 1, 0] = 1.0
                    barycentric[site_base, 1, 1] = 0.0
                    surface[site_base, 0, 0] = s0x
                    surface[site_base, 0, 1] = s0y
                    surface[site_base, 0, 2] = s0z
                    surface[site_base, 1, 0] = s1x
                    surface[site_base, 1, 1] = s1y
                    surface[site_base, 1, 2] = s1z
                    skin[site_base, 0, 0] = radius_a * nx
                    skin[site_base, 0, 1] = radius_a * ny
                    skin[site_base, 0, 2] = radius_a * nz
                    skin[site_base, 1, 0] = -radius_b * nx
                    skin[site_base, 1, 1] = -radius_b * ny
                    skin[site_base, 1, 2] = -radius_b * nz
                elif site_base + 1 > site_capacity:
                    status = 4
            elif (
                kind_a == 0
                and a1 == a0 + 1
                and (
                    (kind_b == 1 and b1 == b0 + 2) or (kind_b == 2 and radius_b == 0.0 and b1 > b0)
                )
            ):
                ax = points[a0, 0]
                ay = points[a0, 1]
                az = points[a0, 2]
                s0x = ax + radius_a * nx
                s0y = ay + radius_a * ny
                s0z = az + radius_a * nz
                s1x = 0.0
                s1y = 0.0
                s1z = 0.0
                ready = 0
                packed = 0
                if kind_b == 1:
                    bxx = points[b0 + 1, 0] - points[b0, 0]
                    bxy = points[b0 + 1, 1] - points[b0, 1]
                    bxz = points[b0 + 1, 2] - points[b0, 2]
                    denom = dot3(bxx, bxy, bxz, bxx, bxy, bxz)
                    parameter = 0.0
                    if denom > 1e-12:
                        parameter = (
                            dot3(
                                ax - points[b0, 0],
                                ay - points[b0, 1],
                                az - points[b0, 2],
                                bxx,
                                bxy,
                                bxz,
                            )
                            / denom
                        )
                    if parameter < 0.0:
                        parameter = 0.0
                    if parameter > 1.0:
                        parameter = 1.0
                    c1x = points[b0, 0] + parameter * bxx
                    c1y = points[b0, 1] + parameter * bxy
                    c1z = points[b0, 2] + parameter * bxz
                    s1x = c1x - radius_b * nx
                    s1y = c1y - radius_b * ny
                    s1z = c1z - radius_b * nz
                    first_weight = 1.0 - parameter
                    second_weight = parameter
                    if first_weight < 1e-5:
                        first_weight = 0.0
                    if second_weight < 1e-5:
                        second_weight = 0.0
                    total = first_weight + second_weight
                    if total > 0.0:
                        first_weight = first_weight / total
                        second_weight = second_weight / total
                        if first_weight > 0.0:
                            work_i[lane, packed, 1] = b0
                            work[lane, packed, 5] = first_weight
                            packed = packed + 1
                        if second_weight > 0.0:
                            work_i[lane, packed, 1] = b0 + 1
                            work[lane, packed, 5] = second_weight
                            packed = packed + 1
                        ready = 1
                else:
                    best = 1.0e30
                    for index in range(b0, b1):
                        height = dot3(
                            points[index, 0], points[index, 1], points[index, 2], nx, ny, nz
                        )
                        if height < best:
                            best = height
                    face_count = 0
                    for index in range(b0, b1):
                        height = dot3(
                            points[index, 0], points[index, 1], points[index, 2], nx, ny, nz
                        )
                        if height - best <= 2e-4 * scale and face_count < 8:
                            work[lane, face_count, 0] = points[index, 0]
                            work[lane, face_count, 1] = points[index, 1]
                            work[lane, face_count, 2] = points[index, 2]
                            work[lane, face_count, 4] = 0.0
                            work_i[lane, face_count, 0] = index
                            face_count = face_count + 1
                        elif height - best <= 2e-4 * scale:
                            face_count = 9
                    if face_count >= 3 and face_count <= 8:
                        absolute_x = tx.abs(nx)
                        absolute_y = tx.abs(ny)
                        absolute_z = tx.abs(nz)
                        tangent_x = 0.0
                        tangent_y = 0.0
                        tangent_z = 0.0
                        if absolute_x <= absolute_y and absolute_x <= absolute_z:
                            tangent_y = -nz
                            tangent_z = ny
                        elif absolute_y <= absolute_z:
                            tangent_x = -nz
                            tangent_z = nx
                        else:
                            tangent_x = ny
                            tangent_y = -nx
                        tangent_length = tx.sqrt(
                            dot3(tangent_x, tangent_y, tangent_z, tangent_x, tangent_y, tangent_z)
                        )
                        if tangent_length > 1e-8:
                            tangent_x = tangent_x / tangent_length
                            tangent_y = tangent_y / tangent_length
                            tangent_z = tangent_z / tangent_length
                            bitangent_x = ny * tangent_z - nz * tangent_y
                            bitangent_y = nz * tangent_x - nx * tangent_z
                            bitangent_z = nx * tangent_y - ny * tangent_x
                            center_h = dot3(ax, ay, az, nx, ny, nz)
                            qx = ax + (best - center_h) * nx
                            qy = ay + (best - center_h) * ny
                            qz = az + (best - center_h) * nz
                            qu = dot3(qx, qy, qz, tangent_x, tangent_y, tangent_z)
                            qv = dot3(qx, qy, qz, bitangent_x, bitangent_y, bitangent_z)
                            center_u = 0.0
                            center_v = 0.0
                            count_f = 0.0
                            for slot in tx.static_range(8):
                                if slot < face_count:
                                    uu = dot3(
                                        work[lane, slot, 0],
                                        work[lane, slot, 1],
                                        work[lane, slot, 2],
                                        tangent_x,
                                        tangent_y,
                                        tangent_z,
                                    )
                                    vv = dot3(
                                        work[lane, slot, 0],
                                        work[lane, slot, 1],
                                        work[lane, slot, 2],
                                        bitangent_x,
                                        bitangent_y,
                                        bitangent_z,
                                    )
                                    work[lane, slot, 6] = uu
                                    work[lane, slot, 7] = vv
                                    center_u = center_u + uu
                                    center_v = center_v + vv
                                    count_f = count_f + 1.0
                            center_u = center_u / count_f
                            center_v = center_v / count_f
                            qu = qu - center_u
                            qv = qv - center_v
                            for slot in tx.static_range(8):
                                if slot < face_count:
                                    work[lane, slot, 6] = work[lane, slot, 6] - center_u
                                    work[lane, slot, 7] = work[lane, slot, 7] - center_v
                            for _pass in tx.static_range(7):
                                for slot in tx.static_range(7):
                                    if slot + 1 < face_count:
                                        left_u = work[lane, slot, 6]
                                        left_v = work[lane, slot, 7]
                                        right_u = work[lane, slot + 1, 6]
                                        right_v = work[lane, slot + 1, 7]
                                        left_negative = 0
                                        right_negative = 0
                                        if left_v < 0.0 or (left_v == 0.0 and left_u < 0.0):
                                            left_negative = 1
                                        if right_v < 0.0 or (right_v == 0.0 and right_u < 0.0):
                                            right_negative = 1
                                        swap = 0
                                        if left_negative != right_negative:
                                            if left_negative == 0:
                                                swap = 1
                                        else:
                                            cross = left_u * right_v - left_v * right_u
                                            if cross < 0.0:
                                                swap = 1
                                        if swap == 1:
                                            for column in tx.static_range(8):
                                                held = work[lane, slot, column]
                                                work[lane, slot, column] = work[
                                                    lane, slot + 1, column
                                                ]
                                                work[lane, slot + 1, column] = held
                                            held_index = work_i[lane, slot, 0]
                                            work_i[lane, slot, 0] = work_i[lane, slot + 1, 0]
                                            work_i[lane, slot + 1, 0] = held_index
                            chosen = 0
                            for tri in tx.static_range(6):
                                second = tri + 1
                                third = tri + 2
                                if chosen == 0 and third < face_count:
                                    u0 = work[lane, 0, 6]
                                    v0 = work[lane, 0, 7]
                                    u1 = work[lane, second, 6]
                                    v1 = work[lane, second, 7]
                                    u2 = work[lane, third, 6]
                                    v2 = work[lane, third, 7]
                                    denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
                                    if tx.abs(denom) > 1e-12:
                                        alpha = (
                                            (v1 - v2) * (qu - u2) + (u2 - u1) * (qv - v2)
                                        ) / denom
                                        beta = (
                                            (v2 - v0) * (qu - u2) + (u0 - u2) * (qv - v2)
                                        ) / denom
                                        gamma = 1.0 - alpha - beta
                                        if alpha >= -1e-5 and beta >= -1e-5 and gamma >= -1e-5:
                                            if alpha < 1e-5:
                                                alpha = 0.0
                                            if beta < 1e-5:
                                                beta = 0.0
                                            if gamma < 1e-5:
                                                gamma = 0.0
                                            if alpha < 0.0:
                                                alpha = 0.0
                                            if beta < 0.0:
                                                beta = 0.0
                                            if gamma < 0.0:
                                                gamma = 0.0
                                            total = alpha + beta + gamma
                                            if total > 0.0:
                                                work[lane, 0, 4] = alpha / total
                                                work[lane, second, 4] = beta / total
                                                work[lane, third, 4] = gamma / total
                                                chosen = 1
                            if chosen == 1:
                                rx = 0.0
                                ry = 0.0
                                rz = 0.0
                                used = 0
                                for slot in tx.static_range(8):
                                    if slot < face_count and work[lane, slot, 4] > 0.0:
                                        rx = rx + work[lane, slot, 4] * work[lane, slot, 0]
                                        ry = ry + work[lane, slot, 4] * work[lane, slot, 1]
                                        rz = rz + work[lane, slot, 4] * work[lane, slot, 2]
                                        used = used + 1
                                error = dot3(rx - qx, ry - qy, rz - qz, rx - qx, ry - qy, rz - qz)
                                limit = 2e-4 * scale
                                if used >= 1 and used <= 4 and error <= limit * limit:
                                    s1x = qx
                                    s1y = qy
                                    s1z = qz
                                    for slot in tx.static_range(8):
                                        if (
                                            slot < face_count
                                            and work[lane, slot, 4] > 0.0
                                            and packed < 4
                                        ):
                                            work_i[lane, packed, 1] = work_i[lane, slot, 0]
                                            work[lane, packed, 5] = work[lane, slot, 4]
                                            packed = packed + 1
                                    ready = 1
                signed = dot3(s1x - s0x, s1y - s0y, s1z - s0z, nx, ny, nz)
                if (
                    ready == 1
                    and tx.abs(signed - query_gap) <= 1e-4 * scale
                    and packed >= 1
                    and site_base + 1 <= site_capacity
                ):
                    site = site_base
                    weight[site] = 1.0
                    gap[site] = signed
                    normal[site, 0] = nx
                    normal[site, 1] = ny
                    normal[site, 2] = nz
                    common[site, 0] = (s0x + s1x) * 0.5
                    common[site, 1] = (s0y + s1y) * 0.5
                    common[site, 2] = (s0z + s1z) * 0.5
                    material_count[site, 0] = 1
                    material_count[site, 1] = packed
                    material_ids[site, 0, 0] = original_ids[a0]
                    barycentric[site, 0, 0] = 1.0
                    for slot in tx.static_range(4):
                        if slot < packed:
                            material_ids[site, 1, slot] = original_ids[work_i[lane, slot, 1]]
                            barycentric[site, 1, slot] = work[lane, slot, 5]
                    surface[site, 0, 0] = s0x
                    surface[site, 0, 1] = s0y
                    surface[site, 0, 2] = s0z
                    surface[site, 1, 0] = s1x
                    surface[site, 1, 1] = s1y
                    surface[site, 1, 2] = s1z
                    skin[site, 0, 0] = radius_a * nx
                    skin[site, 0, 1] = radius_a * ny
                    skin[site, 0, 2] = radius_a * nz
                    skin[site, 1, 0] = -radius_b * nx
                    skin[site, 1, 1] = -radius_b * ny
                    skin[site, 1, 2] = -radius_b * nz
                    status = 1
                    dimension = 0
                    count = 1
                elif ready == 1 and site_base + 1 > site_capacity:
                    status = 4
            elif kind_a == 1 and kind_b == 0 and a1 == a0 + 2 and b1 == b0 + 1:
                bx = points[b0, 0]
                by = points[b0, 1]
                bz = points[b0, 2]
                axx = points[a0 + 1, 0] - points[a0, 0]
                axy = points[a0 + 1, 1] - points[a0, 1]
                axz = points[a0 + 1, 2] - points[a0, 2]
                denom = dot3(axx, axy, axz, axx, axy, axz)
                parameter = 0.0
                if denom > 1e-12:
                    parameter = (
                        dot3(
                            bx - points[a0, 0],
                            by - points[a0, 1],
                            bz - points[a0, 2],
                            axx,
                            axy,
                            axz,
                        )
                        / denom
                    )
                if parameter < 0.0:
                    parameter = 0.0
                if parameter > 1.0:
                    parameter = 1.0
                c0x = points[a0, 0] + parameter * axx
                c0y = points[a0, 1] + parameter * axy
                c0z = points[a0, 2] + parameter * axz
                s0x = c0x + radius_a * nx
                s0y = c0y + radius_a * ny
                s0z = c0z + radius_a * nz
                s1x = bx - radius_b * nx
                s1y = by - radius_b * ny
                s1z = bz - radius_b * nz
                first_weight = 1.0 - parameter
                second_weight = parameter
                if first_weight < 1e-5:
                    first_weight = 0.0
                if second_weight < 1e-5:
                    second_weight = 0.0
                total = first_weight + second_weight
                packed = 0
                if total > 0.0:
                    first_weight = first_weight / total
                    second_weight = second_weight / total
                    if first_weight > 0.0:
                        work_i[lane, packed, 1] = a0
                        work[lane, packed, 5] = first_weight
                        packed = packed + 1
                    if second_weight > 0.0:
                        work_i[lane, packed, 1] = a0 + 1
                        work[lane, packed, 5] = second_weight
                        packed = packed + 1
                signed = dot3(s1x - s0x, s1y - s0y, s1z - s0z, nx, ny, nz)
                if (
                    packed >= 1
                    and tx.abs(signed - query_gap) <= 1e-4 * scale
                    and site_base + 1 <= site_capacity
                ):
                    site = site_base
                    weight[site] = 1.0
                    gap[site] = signed
                    normal[site, 0] = nx
                    normal[site, 1] = ny
                    normal[site, 2] = nz
                    common[site, 0] = (s0x + s1x) * 0.5
                    common[site, 1] = (s0y + s1y) * 0.5
                    common[site, 2] = (s0z + s1z) * 0.5
                    material_count[site, 0] = packed
                    material_count[site, 1] = 1
                    material_ids[site, 1, 0] = original_ids[b0]
                    barycentric[site, 1, 0] = 1.0
                    for slot in tx.static_range(4):
                        if slot < packed:
                            material_ids[site, 0, slot] = original_ids[work_i[lane, slot, 1]]
                            barycentric[site, 0, slot] = work[lane, slot, 5]
                    surface[site, 0, 0] = s0x
                    surface[site, 0, 1] = s0y
                    surface[site, 0, 2] = s0z
                    surface[site, 1, 0] = s1x
                    surface[site, 1, 1] = s1y
                    surface[site, 1, 2] = s1z
                    skin[site, 0, 0] = radius_a * nx
                    skin[site, 0, 1] = radius_a * ny
                    skin[site, 0, 2] = radius_a * nz
                    skin[site, 1, 0] = -radius_b * nx
                    skin[site, 1, 1] = -radius_b * ny
                    skin[site, 1, 2] = -radius_b * nz
                    status = 1
                    dimension = 0
                    count = 1
                elif (
                    packed >= 1
                    and tx.abs(signed - query_gap) <= 1e-4 * scale
                    and site_base + 1 > site_capacity
                ):
                    status = 4
            elif kind_a == 1 and kind_b == 1 and a1 == a0 + 2 and b1 == b0 + 2:
                axx = points[a0 + 1, 0] - points[a0, 0]
                axy = points[a0 + 1, 1] - points[a0, 1]
                axz = points[a0 + 1, 2] - points[a0, 2]
                bxx = points[b0 + 1, 0] - points[b0, 0]
                bxy = points[b0 + 1, 1] - points[b0, 1]
                bxz = points[b0 + 1, 2] - points[b0, 2]
                cross = tx.sqrt(
                    dot3(
                        axy * bxz - axz * bxy,
                        axz * bxx - axx * bxz,
                        axx * bxy - axy * bxx,
                        axy * bxz - axz * bxy,
                        axz * bxx - axx * bxz,
                        axx * bxy - axy * bxx,
                    )
                )
                length_a = tx.sqrt(dot3(axx, axy, axz, axx, axy, axz))
                length_b = tx.sqrt(dot3(bxx, bxy, bxz, bxx, bxy, bxz))
                if cross <= 1e-4 * length_a * length_b and length_a > 1e-8 and length_b > 1e-8:
                    denom = dot3(axx, axy, axz, axx, axy, axz)
                    t0 = (
                        dot3(
                            points[b0, 0] - points[a0, 0],
                            points[b0, 1] - points[a0, 1],
                            points[b0, 2] - points[a0, 2],
                            axx,
                            axy,
                            axz,
                        )
                        / denom
                    )
                    t1 = (
                        dot3(
                            points[b0 + 1, 0] - points[a0, 0],
                            points[b0 + 1, 1] - points[a0, 1],
                            points[b0 + 1, 2] - points[a0, 2],
                            axx,
                            axy,
                            axz,
                        )
                        / denom
                    )
                    low = tx.max(0.0, tx.min(t0, t1))
                    high = tx.min(1.0, tx.max(t0, t1))
                    if high - low > 1e-5 and site_base + 2 <= site_capacity:
                        ready = 1
                        for end in tx.static_range(2):
                            parameter = low
                            if end == 1:
                                parameter = high
                            c0x = points[a0, 0] + parameter * axx
                            c0y = points[a0, 1] + parameter * axy
                            c0z = points[a0, 2] + parameter * axz
                            offset_x = points[b0, 0] - points[a0, 0]
                            offset_y = points[b0, 1] - points[a0, 1]
                            offset_z = points[b0, 2] - points[a0, 2]
                            along = dot3(offset_x, offset_y, offset_z, axx, axy, axz) / denom
                            offset_x = offset_x - along * axx
                            offset_y = offset_y - along * axy
                            offset_z = offset_z - along * axz
                            c1x = c0x + offset_x
                            c1y = c0y + offset_y
                            c1z = c0z + offset_z
                            s0x = c0x + radius_a * nx
                            s0y = c0y + radius_a * ny
                            s0z = c0z + radius_a * nz
                            s1x = c1x - radius_b * nx
                            s1y = c1y - radius_b * ny
                            s1z = c1z - radius_b * nz
                            signed = dot3(s1x - s0x, s1y - s0y, s1z - s0z, nx, ny, nz)
                            if tx.abs(signed - query_gap) > 1e-4 * scale:
                                ready = 0
                            bdenom = dot3(bxx, bxy, bxz, bxx, bxy, bxz)
                            bparameter = (
                                dot3(
                                    c1x - points[b0, 0],
                                    c1y - points[b0, 1],
                                    c1z - points[b0, 2],
                                    bxx,
                                    bxy,
                                    bxz,
                                )
                                / bdenom
                            )
                            if bparameter < -1e-4 or bparameter > 1.0 + 1e-4:
                                ready = 0
                            bparameter = tx.max(0.0, tx.min(1.0, bparameter))
                            if ready == 1:
                                site = site_base + end
                                weight[site] = 0.5
                                gap[site] = signed
                                normal[site, 0] = nx
                                normal[site, 1] = ny
                                normal[site, 2] = nz
                                common[site, 0] = (s0x + s1x) * 0.5
                                common[site, 1] = (s0y + s1y) * 0.5
                                common[site, 2] = (s0z + s1z) * 0.5
                                material_count[site, 0] = 2
                                material_count[site, 1] = 2
                                material_ids[site, 0, 0] = original_ids[a0]
                                material_ids[site, 0, 1] = original_ids[a0 + 1]
                                material_ids[site, 1, 0] = original_ids[b0]
                                material_ids[site, 1, 1] = original_ids[b0 + 1]
                                barycentric[site, 0, 0] = 1.0 - parameter
                                barycentric[site, 0, 1] = parameter
                                barycentric[site, 1, 0] = 1.0 - bparameter
                                barycentric[site, 1, 1] = bparameter
                                surface[site, 0, 0] = s0x
                                surface[site, 0, 1] = s0y
                                surface[site, 0, 2] = s0z
                                surface[site, 1, 0] = s1x
                                surface[site, 1, 1] = s1y
                                surface[site, 1, 2] = s1z
                                skin[site, 0, 0] = radius_a * nx
                                skin[site, 0, 1] = radius_a * ny
                                skin[site, 0, 2] = radius_a * nz
                                skin[site, 1, 0] = -radius_b * nx
                                skin[site, 1, 1] = -radius_b * ny
                                skin[site, 1, 2] = -radius_b * nz
                        if ready == 1:
                            status = 1
                            dimension = 1
                            count = 2
                    elif site_base + 2 > site_capacity:
                        status = 4
                if cross > 1e-4 * length_a * length_b and length_a > 1e-8 and length_b > 1e-8:
                    rx = points[a0, 0] - points[b0, 0]
                    ry = points[a0, 1] - points[b0, 1]
                    rz = points[a0, 2] - points[b0, 2]
                    aa = dot3(axx, axy, axz, axx, axy, axz)
                    vv = dot3(bxx, bxy, bxz, bxx, bxy, bxz)
                    bb = dot3(axx, axy, axz, bxx, bxy, bxz)
                    cc = dot3(axx, axy, axz, rx, ry, rz)
                    ff = dot3(bxx, bxy, bxz, rx, ry, rz)
                    sa = 0.5
                    sb = 0.5
                    if aa > 0.0 and vv > 0.0:
                        det = aa * vv - bb * bb
                        if det > 1e-7 * aa * vv:
                            sa = (bb * ff - cc * vv) / det
                            if sa < 0.0:
                                sa = 0.0
                            if sa > 1.0:
                                sa = 1.0
                        sb = (bb * sa + ff) / vv
                        if sb < 0.0:
                            sb = 0.0
                            sa = -cc / aa
                            if sa < 0.0:
                                sa = 0.0
                            if sa > 1.0:
                                sa = 1.0
                        elif sb > 1.0:
                            sb = 1.0
                            sa = (bb - cc) / aa
                            if sa < 0.0:
                                sa = 0.0
                            if sa > 1.0:
                                sa = 1.0
                    cx = points[a0, 0] + sa * axx
                    cy = points[a0, 1] + sa * axy
                    cz = points[a0, 2] + sa * axz
                    hx = points[b0, 0] + sb * bxx
                    hy = points[b0, 1] + sb * bxy
                    hz = points[b0, 2] + sb * bxz
                    s0x = cx + radius_a * nx
                    s0y = cy + radius_a * ny
                    s0z = cz + radius_a * nz
                    s1x = hx - radius_b * nx
                    s1y = hy - radius_b * ny
                    s1z = hz - radius_b * nz
                    signed = dot3(s1x - s0x, s1y - s0y, s1z - s0z, nx, ny, nz)
                    aw0 = 1.0 - sa
                    aw1 = sa
                    if aw0 < 1e-5:
                        aw0 = 0.0
                    if aw1 < 1e-5:
                        aw1 = 0.0
                    atotal = aw0 + aw1
                    packed_a = 0
                    if atotal > 0.0:
                        aw0 = aw0 / atotal
                        aw1 = aw1 / atotal
                        if aw0 > 0.0:
                            work_i[lane, packed_a, 1] = a0
                            work[lane, packed_a, 5] = aw0
                            packed_a = packed_a + 1
                        if aw1 > 0.0:
                            work_i[lane, packed_a, 1] = a0 + 1
                            work[lane, packed_a, 5] = aw1
                            packed_a = packed_a + 1
                    bw0 = 1.0 - sb
                    bw1 = sb
                    if bw0 < 1e-5:
                        bw0 = 0.0
                    if bw1 < 1e-5:
                        bw1 = 0.0
                    btotal = bw0 + bw1
                    packed_b = 0
                    if btotal > 0.0:
                        bw0 = bw0 / btotal
                        bw1 = bw1 / btotal
                        if bw0 > 0.0:
                            work_i[lane, packed_b, 2] = b0
                            work[lane, packed_b, 3] = bw0
                            packed_b = packed_b + 1
                        if bw1 > 0.0:
                            work_i[lane, packed_b, 2] = b0 + 1
                            work[lane, packed_b, 3] = bw1
                            packed_b = packed_b + 1
                    if (
                        packed_a >= 1
                        and packed_b >= 1
                        and tx.abs(signed - query_gap) <= 1e-4 * scale
                        and site_base + 1 <= site_capacity
                    ):
                        site = site_base
                        weight[site] = 1.0
                        gap[site] = signed
                        normal[site, 0] = nx
                        normal[site, 1] = ny
                        normal[site, 2] = nz
                        common[site, 0] = (s0x + s1x) * 0.5
                        common[site, 1] = (s0y + s1y) * 0.5
                        common[site, 2] = (s0z + s1z) * 0.5
                        material_count[site, 0] = packed_a
                        material_count[site, 1] = packed_b
                        for slot in tx.static_range(4):
                            if slot < packed_a:
                                material_ids[site, 0, slot] = original_ids[work_i[lane, slot, 1]]
                                barycentric[site, 0, slot] = work[lane, slot, 5]
                            if slot < packed_b:
                                material_ids[site, 1, slot] = original_ids[work_i[lane, slot, 2]]
                                barycentric[site, 1, slot] = work[lane, slot, 3]
                        surface[site, 0, 0] = s0x
                        surface[site, 0, 1] = s0y
                        surface[site, 0, 2] = s0z
                        surface[site, 1, 0] = s1x
                        surface[site, 1, 1] = s1y
                        surface[site, 1, 2] = s1z
                        skin[site, 0, 0] = radius_a * nx
                        skin[site, 0, 1] = radius_a * ny
                        skin[site, 0, 2] = radius_a * nz
                        skin[site, 1, 0] = -radius_b * nx
                        skin[site, 1, 1] = -radius_b * ny
                        skin[site, 1, 2] = -radius_b * nz
                        status = 1
                        dimension = 0
                        count = 1
                    elif (
                        packed_a >= 1
                        and packed_b >= 1
                        and tx.abs(signed - query_gap) <= 1e-4 * scale
                        and site_base + 1 > site_capacity
                    ):
                        status = 4
            elif kind_a == 1 and kind_b == 2 and radius_b == 0.0 and a1 == a0 + 2 and b1 > b0:
                best = 1.0e30
                for index in range(b0, b1):
                    height = dot3(points[index, 0], points[index, 1], points[index, 2], nx, ny, nz)
                    if height < best:
                        best = height
                face_count = 0
                for index in range(b0, b1):
                    height = dot3(points[index, 0], points[index, 1], points[index, 2], nx, ny, nz)
                    if height - best <= 2e-4 * scale and face_count < 8:
                        work[lane, face_count, 0] = points[index, 0]
                        work[lane, face_count, 1] = points[index, 1]
                        work[lane, face_count, 2] = points[index, 2]
                        work[lane, face_count, 4] = 0.0
                        work_i[lane, face_count, 0] = index
                        face_count = face_count + 1
                    elif height - best <= 2e-4 * scale:
                        face_count = 9
                if face_count >= 3 and face_count <= 8:
                    absolute_x = tx.abs(nx)
                    absolute_y = tx.abs(ny)
                    absolute_z = tx.abs(nz)
                    tangent_x = 0.0
                    tangent_y = 0.0
                    tangent_z = 0.0
                    if absolute_x <= absolute_y and absolute_x <= absolute_z:
                        tangent_y = -nz
                        tangent_z = ny
                    elif absolute_y <= absolute_z:
                        tangent_x = -nz
                        tangent_z = nx
                    else:
                        tangent_x = ny
                        tangent_y = -nx
                    tangent_length = tx.sqrt(
                        dot3(tangent_x, tangent_y, tangent_z, tangent_x, tangent_y, tangent_z)
                    )
                    if tangent_length > 1e-8:
                        tangent_x = tangent_x / tangent_length
                        tangent_y = tangent_y / tangent_length
                        tangent_z = tangent_z / tangent_length
                        bitangent_x = ny * tangent_z - nz * tangent_y
                        bitangent_y = nz * tangent_x - nx * tangent_z
                        bitangent_z = nx * tangent_y - ny * tangent_x
                        center_u = 0.0
                        center_v = 0.0
                        count_f = 0.0
                        for slot in tx.static_range(8):
                            if slot < face_count:
                                uu = dot3(
                                    work[lane, slot, 0],
                                    work[lane, slot, 1],
                                    work[lane, slot, 2],
                                    tangent_x,
                                    tangent_y,
                                    tangent_z,
                                )
                                vv = dot3(
                                    work[lane, slot, 0],
                                    work[lane, slot, 1],
                                    work[lane, slot, 2],
                                    bitangent_x,
                                    bitangent_y,
                                    bitangent_z,
                                )
                                work[lane, slot, 6] = uu
                                work[lane, slot, 7] = vv
                                center_u = center_u + uu
                                center_v = center_v + vv
                                count_f = count_f + 1.0
                        center_u = center_u / count_f
                        center_v = center_v / count_f
                        for slot in tx.static_range(8):
                            if slot < face_count:
                                work[lane, slot, 6] = work[lane, slot, 6] - center_u
                                work[lane, slot, 7] = work[lane, slot, 7] - center_v
                        for _pass in tx.static_range(7):
                            for slot in tx.static_range(7):
                                if slot + 1 < face_count:
                                    left_u = work[lane, slot, 6]
                                    left_v = work[lane, slot, 7]
                                    right_u = work[lane, slot + 1, 6]
                                    right_v = work[lane, slot + 1, 7]
                                    left_negative = 0
                                    right_negative = 0
                                    if left_v < 0.0 or (left_v == 0.0 and left_u < 0.0):
                                        left_negative = 1
                                    if right_v < 0.0 or (right_v == 0.0 and right_u < 0.0):
                                        right_negative = 1
                                    swap = 0
                                    if left_negative != right_negative:
                                        if left_negative == 0:
                                            swap = 1
                                    else:
                                        cross = left_u * right_v - left_v * right_u
                                        if cross < 0.0:
                                            swap = 1
                                    if swap == 1:
                                        for column in tx.static_range(8):
                                            held = work[lane, slot, column]
                                            work[lane, slot, column] = work[lane, slot + 1, column]
                                            work[lane, slot + 1, column] = held
                                        held_index = work_i[lane, slot, 0]
                                        work_i[lane, slot, 0] = work_i[lane, slot + 1, 0]
                                        work_i[lane, slot + 1, 0] = held_index
                        p0x = points[a0, 0]
                        p0y = points[a0, 1]
                        p0z = points[a0, 2]
                        p1x = points[a0 + 1, 0]
                        p1y = points[a0 + 1, 1]
                        p1z = points[a0 + 1, 2]
                        h0 = dot3(p0x, p0y, p0z, nx, ny, nz)
                        h1 = dot3(p1x, p1y, p1z, nx, ny, nz)
                        q0x = p0x + (best - h0) * nx
                        q0y = p0y + (best - h0) * ny
                        q0z = p0z + (best - h0) * nz
                        q1x = p1x + (best - h1) * nx
                        q1y = p1y + (best - h1) * ny
                        q1z = p1z + (best - h1) * nz
                        au = dot3(q0x, q0y, q0z, tangent_x, tangent_y, tangent_z) - center_u
                        av = dot3(q0x, q0y, q0z, bitangent_x, bitangent_y, bitangent_z) - center_v
                        bu = dot3(q1x, q1y, q1z, tangent_x, tangent_y, tangent_z) - center_u
                        bv = dot3(q1x, q1y, q1z, bitangent_x, bitangent_y, bitangent_z) - center_v
                        t_low = 0.0
                        t_high = 1.0
                        inside = 1
                        band = 1e-5 * scale
                        for edge in tx.static_range(8):
                            if edge < face_count and inside == 1:
                                nxt = edge + 1
                                if nxt == face_count:
                                    nxt = 0
                                e0u = work[lane, edge, 6]
                                e0v = work[lane, edge, 7]
                                e1u = work[lane, nxt, 6]
                                e1v = work[lane, nxt, 7]
                                ex = e1u - e0u
                                ey = e1v - e0v
                                c0 = ex * (av - e0v) - ey * (au - e0u)
                                c1 = ex * (bv - av) - ey * (bu - au)
                                if c1 > 1e-8:
                                    bound = (-band - c0) / c1
                                    if bound > t_low:
                                        t_low = bound
                                elif c1 < -1e-8:
                                    bound = (-band - c0) / c1
                                    if bound < t_high:
                                        t_high = bound
                                elif c0 < -band:
                                    inside = 0
                        if t_low < 0.0:
                            t_low = 0.0
                        if t_high > 1.0:
                            t_high = 1.0
                        ready_ends = 0
                        if inside == 1 and t_high - t_low > 1e-5:
                            for end in tx.static_range(2):
                                parameter = t_low
                                if end == 1:
                                    parameter = t_high
                                cx = p0x + parameter * (p1x - p0x)
                                cy = p0y + parameter * (p1y - p0y)
                                cz = p0z + parameter * (p1z - p0z)
                                s0x = cx + radius_a * nx
                                s0y = cy + radius_a * ny
                                s0z = cz + radius_a * nz
                                core_h = dot3(cx, cy, cz, nx, ny, nz)
                                s1x = cx + (best - core_h) * nx
                                s1y = cy + (best - core_h) * ny
                                s1z = cz + (best - core_h) * nz
                                signed = dot3(s1x - s0x, s1y - s0y, s1z - s0z, nx, ny, nz)
                                qu = dot3(s1x, s1y, s1z, tangent_x, tangent_y, tangent_z) - center_u
                                qv = (
                                    dot3(s1x, s1y, s1z, bitangent_x, bitangent_y, bitangent_z)
                                    - center_v
                                )
                                for slot in tx.static_range(8):
                                    work[lane, slot, 4] = 0.0
                                chosen = 0
                                for tri in tx.static_range(6):
                                    second = tri + 1
                                    third = tri + 2
                                    if chosen == 0 and third < face_count:
                                        u0 = work[lane, 0, 6]
                                        v0 = work[lane, 0, 7]
                                        u1 = work[lane, second, 6]
                                        v1 = work[lane, second, 7]
                                        u2 = work[lane, third, 6]
                                        v2 = work[lane, third, 7]
                                        denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
                                        if tx.abs(denom) > 1e-12:
                                            alpha = (
                                                (v1 - v2) * (qu - u2) + (u2 - u1) * (qv - v2)
                                            ) / denom
                                            beta = (
                                                (v2 - v0) * (qu - u2) + (u0 - u2) * (qv - v2)
                                            ) / denom
                                            gamma = 1.0 - alpha - beta
                                            if alpha >= -1e-5 and beta >= -1e-5 and gamma >= -1e-5:
                                                if alpha < 1e-5:
                                                    alpha = 0.0
                                                if beta < 1e-5:
                                                    beta = 0.0
                                                if gamma < 1e-5:
                                                    gamma = 0.0
                                                if alpha < 0.0:
                                                    alpha = 0.0
                                                if beta < 0.0:
                                                    beta = 0.0
                                                if gamma < 0.0:
                                                    gamma = 0.0
                                                total = alpha + beta + gamma
                                                if total > 0.0:
                                                    work[lane, 0, 4] = alpha / total
                                                    work[lane, second, 4] = beta / total
                                                    work[lane, third, 4] = gamma / total
                                                    chosen = 1
                                if chosen == 1 and tx.abs(signed - query_gap) <= 1e-4 * scale:
                                    rx = 0.0
                                    ry = 0.0
                                    rz = 0.0
                                    used = 0
                                    for slot in tx.static_range(8):
                                        if slot < face_count and work[lane, slot, 4] > 0.0:
                                            rx = rx + work[lane, slot, 4] * work[lane, slot, 0]
                                            ry = ry + work[lane, slot, 4] * work[lane, slot, 1]
                                            rz = rz + work[lane, slot, 4] * work[lane, slot, 2]
                                            used = used + 1
                                    error = dot3(
                                        rx - s1x, ry - s1y, rz - s1z, rx - s1x, ry - s1y, rz - s1z
                                    )
                                    limit = 2e-4 * scale
                                    if used >= 1 and used <= 4 and error <= limit * limit:
                                        packed = 0
                                        for slot in tx.static_range(8):
                                            if (
                                                slot < face_count
                                                and work[lane, slot, 4] > 0.0
                                                and packed < 4
                                            ):
                                                if end == 0:
                                                    work_i[lane, packed, 1] = work_i[lane, slot, 0]
                                                    work[lane, packed, 5] = work[lane, slot, 4]
                                                else:
                                                    work_i[lane, packed, 2] = work_i[lane, slot, 0]
                                                    work[lane, packed, 3] = work[lane, slot, 4]
                                                packed = packed + 1
                                        work[lane, end, 8] = parameter
                                        work[lane, end, 9] = s0x
                                        work[lane, end, 10] = s0y
                                        work[lane, end, 11] = s0z
                                        work[lane, end, 12] = s1x
                                        work[lane, end, 13] = s1y
                                        work[lane, end, 14] = s1z
                                        work[lane, end, 15] = signed
                                        work_i[lane, end, 7] = packed
                                        ready_ends = ready_ends + 1
                        if ready_ends == 2 and site_base + 2 <= site_capacity:
                            for end in tx.static_range(2):
                                site = site_base + end
                                parameter = work[lane, end, 8]
                                mcount = work_i[lane, end, 7]
                                weight[site] = 0.5
                                gap[site] = work[lane, end, 15]
                                normal[site, 0] = nx
                                normal[site, 1] = ny
                                normal[site, 2] = nz
                                common[site, 0] = (work[lane, end, 9] + work[lane, end, 12]) * 0.5
                                common[site, 1] = (work[lane, end, 10] + work[lane, end, 13]) * 0.5
                                common[site, 2] = (work[lane, end, 11] + work[lane, end, 14]) * 0.5
                                material_count[site, 0] = 2
                                material_count[site, 1] = mcount
                                material_ids[site, 0, 0] = original_ids[a0]
                                material_ids[site, 0, 1] = original_ids[a0 + 1]
                                barycentric[site, 0, 0] = 1.0 - parameter
                                barycentric[site, 0, 1] = parameter
                                for slot in tx.static_range(4):
                                    if slot < mcount:
                                        if end == 0:
                                            material_ids[site, 1, slot] = original_ids[
                                                work_i[lane, slot, 1]
                                            ]
                                            barycentric[site, 1, slot] = work[lane, slot, 5]
                                        else:
                                            material_ids[site, 1, slot] = original_ids[
                                                work_i[lane, slot, 2]
                                            ]
                                            barycentric[site, 1, slot] = work[lane, slot, 3]
                                surface[site, 0, 0] = work[lane, end, 9]
                                surface[site, 0, 1] = work[lane, end, 10]
                                surface[site, 0, 2] = work[lane, end, 11]
                                surface[site, 1, 0] = work[lane, end, 12]
                                surface[site, 1, 1] = work[lane, end, 13]
                                surface[site, 1, 2] = work[lane, end, 14]
                                skin[site, 0, 0] = radius_a * nx
                                skin[site, 0, 1] = radius_a * ny
                                skin[site, 0, 2] = radius_a * nz
                                skin[site, 1, 0] = 0.0
                                skin[site, 1, 1] = 0.0
                                skin[site, 1, 2] = 0.0
                            status = 1
                            dimension = 1
                            count = 2
                        elif ready_ends == 2 and site_base + 2 > site_capacity:
                            status = 4
                if face_count == 2:
                    p0x = points[a0, 0]
                    p0y = points[a0, 1]
                    p0z = points[a0, 2]
                    p1x = points[a0 + 1, 0]
                    p1y = points[a0 + 1, 1]
                    p1z = points[a0 + 1, 2]
                    e0x = work[lane, 0, 0]
                    e0y = work[lane, 0, 1]
                    e0z = work[lane, 0, 2]
                    e1x = work[lane, 1, 0]
                    e1y = work[lane, 1, 1]
                    e1z = work[lane, 1, 2]
                    ux = p1x - p0x
                    uy = p1y - p0y
                    uz = p1z - p0z
                    vx = e1x - e0x
                    vy = e1y - e0y
                    vz = e1z - e0z
                    wx = p0x - e0x
                    wy = p0y - e0y
                    wz = p0z - e0z
                    aa = dot3(ux, uy, uz, ux, uy, uz)
                    bb = dot3(ux, uy, uz, vx, vy, vz)
                    cc = dot3(vx, vy, vz, vx, vy, vz)
                    dd = dot3(ux, uy, uz, wx, wy, wz)
                    ee = dot3(vx, vy, vz, wx, wy, wz)
                    det = aa * cc - bb * bb
                    axis_t = 0.0
                    edge_t = 0.0
                    if det > 1e-12 and aa > 1e-12 and cc > 1e-12:
                        axis_t = (bb * ee - cc * dd) / det
                        edge_t = (aa * ee - bb * dd) / det
                    if axis_t < 0.0:
                        axis_t = 0.0
                    if axis_t > 1.0:
                        axis_t = 1.0
                    if cc > 1e-12:
                        edge_t = (
                            dot3(
                                p0x + axis_t * ux - e0x,
                                p0y + axis_t * uy - e0y,
                                p0z + axis_t * uz - e0z,
                                vx,
                                vy,
                                vz,
                            )
                            / cc
                        )
                    if edge_t < 0.0:
                        edge_t = 0.0
                    if edge_t > 1.0:
                        edge_t = 1.0
                    if aa > 1e-12:
                        axis_t = (
                            dot3(
                                e0x + edge_t * vx - p0x,
                                e0y + edge_t * vy - p0y,
                                e0z + edge_t * vz - p0z,
                                ux,
                                uy,
                                uz,
                            )
                            / aa
                        )
                    if axis_t < 0.0:
                        axis_t = 0.0
                    if axis_t > 1.0:
                        axis_t = 1.0
                    cx = p0x + axis_t * ux
                    cy = p0y + axis_t * uy
                    cz = p0z + axis_t * uz
                    hx = e0x + edge_t * vx
                    hy = e0y + edge_t * vy
                    hz = e0z + edge_t * vz
                    s0x = cx + radius_a * nx
                    s0y = cy + radius_a * ny
                    s0z = cz + radius_a * nz
                    s1x = hx
                    s1y = hy
                    s1z = hz
                    signed = dot3(s1x - s0x, s1y - s0y, s1z - s0z, nx, ny, nz)
                    cross_x = uy * vz - uz * vy
                    cross_y = uz * vx - ux * vz
                    cross_z = ux * vy - uy * vx
                    cross_len = tx.sqrt(dot3(cross_x, cross_y, cross_z, cross_x, cross_y, cross_z))
                    skew = 0
                    if (
                        tx.sqrt(aa) > 1e-8
                        and tx.sqrt(cc) > 1e-8
                        and cross_len > 1e-4 * tx.sqrt(aa) * tx.sqrt(cc)
                    ):
                        skew = 1
                    first_weight = 1.0 - axis_t
                    second_weight = axis_t
                    if first_weight < 1e-5:
                        first_weight = 0.0
                    if second_weight < 1e-5:
                        second_weight = 0.0
                    total = first_weight + second_weight
                    packed = 0
                    if total > 0.0:
                        first_weight = first_weight / total
                        second_weight = second_weight / total
                        if first_weight > 0.0:
                            work_i[lane, packed, 1] = a0
                            work[lane, packed, 5] = first_weight
                            packed = packed + 1
                        if second_weight > 0.0:
                            work_i[lane, packed, 1] = a0 + 1
                            work[lane, packed, 5] = second_weight
                            packed = packed + 1
                    edge0 = 1.0 - edge_t
                    edge1 = edge_t
                    if edge0 < 1e-5:
                        edge0 = 0.0
                    if edge1 < 1e-5:
                        edge1 = 0.0
                    edge_total = edge0 + edge1
                    edge_count = 0
                    if edge_total > 0.0:
                        edge0 = edge0 / edge_total
                        edge1 = edge1 / edge_total
                        if edge0 > 0.0:
                            work_i[lane, edge_count, 2] = work_i[lane, 0, 0]
                            work[lane, edge_count, 3] = edge0
                            edge_count = edge_count + 1
                        if edge1 > 0.0:
                            work_i[lane, edge_count, 2] = work_i[lane, 1, 0]
                            work[lane, edge_count, 3] = edge1
                            edge_count = edge_count + 1
                    if (
                        skew == 1
                        and packed >= 1
                        and edge_count >= 1
                        and tx.abs(signed - query_gap) <= 1e-4 * scale
                        and site_base + 1 <= site_capacity
                    ):
                        site = site_base
                        weight[site] = 1.0
                        gap[site] = signed
                        normal[site, 0] = nx
                        normal[site, 1] = ny
                        normal[site, 2] = nz
                        common[site, 0] = (s0x + s1x) * 0.5
                        common[site, 1] = (s0y + s1y) * 0.5
                        common[site, 2] = (s0z + s1z) * 0.5
                        material_count[site, 0] = packed
                        material_count[site, 1] = edge_count
                        for slot in tx.static_range(4):
                            if slot < packed:
                                material_ids[site, 0, slot] = original_ids[work_i[lane, slot, 1]]
                                barycentric[site, 0, slot] = work[lane, slot, 5]
                            if slot < edge_count:
                                material_ids[site, 1, slot] = original_ids[work_i[lane, slot, 2]]
                                barycentric[site, 1, slot] = work[lane, slot, 3]
                        surface[site, 0, 0] = s0x
                        surface[site, 0, 1] = s0y
                        surface[site, 0, 2] = s0z
                        surface[site, 1, 0] = s1x
                        surface[site, 1, 1] = s1y
                        surface[site, 1, 2] = s1z
                        skin[site, 0, 0] = radius_a * nx
                        skin[site, 0, 1] = radius_a * ny
                        skin[site, 0, 2] = radius_a * nz
                        skin[site, 1, 0] = 0.0
                        skin[site, 1, 1] = 0.0
                        skin[site, 1, 2] = 0.0
                        status = 1
                        dimension = 0
                        count = 1
                    elif (
                        skew == 1
                        and packed >= 1
                        and edge_count >= 1
                        and tx.abs(signed - query_gap) <= 1e-4 * scale
                        and site_base + 1 > site_capacity
                    ):
                        status = 4
            elif kind_a == 2 and kind_b == 1 and radius_a == 0.0 and a1 > a0 and b1 == b0 + 2:
                best = -1.0e30
                for index in range(a0, a1):
                    height = dot3(points[index, 0], points[index, 1], points[index, 2], nx, ny, nz)
                    if height > best:
                        best = height
                face_count = 0
                for index in range(a0, a1):
                    height = dot3(points[index, 0], points[index, 1], points[index, 2], nx, ny, nz)
                    if best - height <= 2e-4 * scale and face_count < 8:
                        work[lane, face_count, 0] = points[index, 0]
                        work[lane, face_count, 1] = points[index, 1]
                        work[lane, face_count, 2] = points[index, 2]
                        work[lane, face_count, 4] = 0.0
                        work_i[lane, face_count, 0] = index
                        face_count = face_count + 1
                    elif best - height <= 2e-4 * scale:
                        face_count = 9
                if face_count >= 3 and face_count <= 8:
                    absolute_x = tx.abs(nx)
                    absolute_y = tx.abs(ny)
                    absolute_z = tx.abs(nz)
                    tangent_x = 0.0
                    tangent_y = 0.0
                    tangent_z = 0.0
                    if absolute_x <= absolute_y and absolute_x <= absolute_z:
                        tangent_y = -nz
                        tangent_z = ny
                    elif absolute_y <= absolute_z:
                        tangent_x = -nz
                        tangent_z = nx
                    else:
                        tangent_x = ny
                        tangent_y = -nx
                    tangent_length = tx.sqrt(
                        dot3(tangent_x, tangent_y, tangent_z, tangent_x, tangent_y, tangent_z)
                    )
                    if tangent_length > 1e-8:
                        tangent_x = tangent_x / tangent_length
                        tangent_y = tangent_y / tangent_length
                        tangent_z = tangent_z / tangent_length
                        bitangent_x = ny * tangent_z - nz * tangent_y
                        bitangent_y = nz * tangent_x - nx * tangent_z
                        bitangent_z = nx * tangent_y - ny * tangent_x
                        center_u = 0.0
                        center_v = 0.0
                        count_f = 0.0
                        for slot in tx.static_range(8):
                            if slot < face_count:
                                uu = dot3(
                                    work[lane, slot, 0],
                                    work[lane, slot, 1],
                                    work[lane, slot, 2],
                                    tangent_x,
                                    tangent_y,
                                    tangent_z,
                                )
                                vv = dot3(
                                    work[lane, slot, 0],
                                    work[lane, slot, 1],
                                    work[lane, slot, 2],
                                    bitangent_x,
                                    bitangent_y,
                                    bitangent_z,
                                )
                                work[lane, slot, 6] = uu
                                work[lane, slot, 7] = vv
                                center_u = center_u + uu
                                center_v = center_v + vv
                                count_f = count_f + 1.0
                        center_u = center_u / count_f
                        center_v = center_v / count_f
                        for slot in tx.static_range(8):
                            if slot < face_count:
                                work[lane, slot, 6] = work[lane, slot, 6] - center_u
                                work[lane, slot, 7] = work[lane, slot, 7] - center_v
                        for _pass in tx.static_range(7):
                            for slot in tx.static_range(7):
                                if slot + 1 < face_count:
                                    left_u = work[lane, slot, 6]
                                    left_v = work[lane, slot, 7]
                                    right_u = work[lane, slot + 1, 6]
                                    right_v = work[lane, slot + 1, 7]
                                    left_negative = 0
                                    right_negative = 0
                                    if left_v < 0.0 or (left_v == 0.0 and left_u < 0.0):
                                        left_negative = 1
                                    if right_v < 0.0 or (right_v == 0.0 and right_u < 0.0):
                                        right_negative = 1
                                    swap = 0
                                    if left_negative != right_negative:
                                        if left_negative == 0:
                                            swap = 1
                                    else:
                                        cross = left_u * right_v - left_v * right_u
                                        if cross < 0.0:
                                            swap = 1
                                    if swap == 1:
                                        for column in tx.static_range(8):
                                            held = work[lane, slot, column]
                                            work[lane, slot, column] = work[lane, slot + 1, column]
                                            work[lane, slot + 1, column] = held
                                        held_index = work_i[lane, slot, 0]
                                        work_i[lane, slot, 0] = work_i[lane, slot + 1, 0]
                                        work_i[lane, slot + 1, 0] = held_index
                        p0x = points[b0, 0]
                        p0y = points[b0, 1]
                        p0z = points[b0, 2]
                        p1x = points[b0 + 1, 0]
                        p1y = points[b0 + 1, 1]
                        p1z = points[b0 + 1, 2]
                        h0 = dot3(p0x, p0y, p0z, nx, ny, nz)
                        h1 = dot3(p1x, p1y, p1z, nx, ny, nz)
                        q0x = p0x + (best - h0) * nx
                        q0y = p0y + (best - h0) * ny
                        q0z = p0z + (best - h0) * nz
                        q1x = p1x + (best - h1) * nx
                        q1y = p1y + (best - h1) * ny
                        q1z = p1z + (best - h1) * nz
                        au = dot3(q0x, q0y, q0z, tangent_x, tangent_y, tangent_z) - center_u
                        av = dot3(q0x, q0y, q0z, bitangent_x, bitangent_y, bitangent_z) - center_v
                        bu = dot3(q1x, q1y, q1z, tangent_x, tangent_y, tangent_z) - center_u
                        bv = dot3(q1x, q1y, q1z, bitangent_x, bitangent_y, bitangent_z) - center_v
                        t_low = 0.0
                        t_high = 1.0
                        inside = 1
                        band = 1e-5 * scale
                        for edge in tx.static_range(8):
                            if edge < face_count and inside == 1:
                                nxt = edge + 1
                                if nxt == face_count:
                                    nxt = 0
                                e0u = work[lane, edge, 6]
                                e0v = work[lane, edge, 7]
                                e1u = work[lane, nxt, 6]
                                e1v = work[lane, nxt, 7]
                                ex = e1u - e0u
                                ey = e1v - e0v
                                c0 = ex * (av - e0v) - ey * (au - e0u)
                                c1 = ex * (bv - av) - ey * (bu - au)
                                if c1 > 1e-8:
                                    bound = (-band - c0) / c1
                                    if bound > t_low:
                                        t_low = bound
                                elif c1 < -1e-8:
                                    bound = (-band - c0) / c1
                                    if bound < t_high:
                                        t_high = bound
                                elif c0 < -band:
                                    inside = 0
                        if t_low < 0.0:
                            t_low = 0.0
                        if t_high > 1.0:
                            t_high = 1.0
                        ready_ends = 0
                        if inside == 1 and t_high - t_low > 1e-5:
                            for end in tx.static_range(2):
                                parameter = t_low
                                if end == 1:
                                    parameter = t_high
                                cx = p0x + parameter * (p1x - p0x)
                                cy = p0y + parameter * (p1y - p0y)
                                cz = p0z + parameter * (p1z - p0z)
                                core_h = dot3(cx, cy, cz, nx, ny, nz)
                                s0x = cx + (best - core_h) * nx
                                s0y = cy + (best - core_h) * ny
                                s0z = cz + (best - core_h) * nz
                                s1x = cx - radius_b * nx
                                s1y = cy - radius_b * ny
                                s1z = cz - radius_b * nz
                                signed = dot3(s1x - s0x, s1y - s0y, s1z - s0z, nx, ny, nz)
                                qu = dot3(s0x, s0y, s0z, tangent_x, tangent_y, tangent_z) - center_u
                                qv = (
                                    dot3(s0x, s0y, s0z, bitangent_x, bitangent_y, bitangent_z)
                                    - center_v
                                )
                                for slot in tx.static_range(8):
                                    work[lane, slot, 4] = 0.0
                                chosen = 0
                                for tri in tx.static_range(6):
                                    second = tri + 1
                                    third = tri + 2
                                    if chosen == 0 and third < face_count:
                                        u0 = work[lane, 0, 6]
                                        v0 = work[lane, 0, 7]
                                        u1 = work[lane, second, 6]
                                        v1 = work[lane, second, 7]
                                        u2 = work[lane, third, 6]
                                        v2 = work[lane, third, 7]
                                        denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
                                        if tx.abs(denom) > 1e-12:
                                            alpha = (
                                                (v1 - v2) * (qu - u2) + (u2 - u1) * (qv - v2)
                                            ) / denom
                                            beta = (
                                                (v2 - v0) * (qu - u2) + (u0 - u2) * (qv - v2)
                                            ) / denom
                                            gamma = 1.0 - alpha - beta
                                            if alpha >= -1e-5 and beta >= -1e-5 and gamma >= -1e-5:
                                                if alpha < 1e-5:
                                                    alpha = 0.0
                                                if beta < 1e-5:
                                                    beta = 0.0
                                                if gamma < 1e-5:
                                                    gamma = 0.0
                                                if alpha < 0.0:
                                                    alpha = 0.0
                                                if beta < 0.0:
                                                    beta = 0.0
                                                if gamma < 0.0:
                                                    gamma = 0.0
                                                total = alpha + beta + gamma
                                                if total > 0.0:
                                                    work[lane, 0, 4] = alpha / total
                                                    work[lane, second, 4] = beta / total
                                                    work[lane, third, 4] = gamma / total
                                                    chosen = 1
                                if chosen == 1 and tx.abs(signed - query_gap) <= 1e-4 * scale:
                                    rx = 0.0
                                    ry = 0.0
                                    rz = 0.0
                                    used = 0
                                    for slot in tx.static_range(8):
                                        if slot < face_count and work[lane, slot, 4] > 0.0:
                                            rx = rx + work[lane, slot, 4] * work[lane, slot, 0]
                                            ry = ry + work[lane, slot, 4] * work[lane, slot, 1]
                                            rz = rz + work[lane, slot, 4] * work[lane, slot, 2]
                                            used = used + 1
                                    error = dot3(
                                        rx - s0x, ry - s0y, rz - s0z, rx - s0x, ry - s0y, rz - s0z
                                    )
                                    limit = 2e-4 * scale
                                    if used >= 1 and used <= 4 and error <= limit * limit:
                                        packed = 0
                                        for slot in tx.static_range(8):
                                            if (
                                                slot < face_count
                                                and work[lane, slot, 4] > 0.0
                                                and packed < 4
                                            ):
                                                if end == 0:
                                                    work_i[lane, packed, 1] = work_i[lane, slot, 0]
                                                    work[lane, packed, 5] = work[lane, slot, 4]
                                                else:
                                                    work_i[lane, packed, 2] = work_i[lane, slot, 0]
                                                    work[lane, packed, 3] = work[lane, slot, 4]
                                                packed = packed + 1
                                        work[lane, end, 8] = parameter
                                        work[lane, end, 9] = s0x
                                        work[lane, end, 10] = s0y
                                        work[lane, end, 11] = s0z
                                        work[lane, end, 12] = s1x
                                        work[lane, end, 13] = s1y
                                        work[lane, end, 14] = s1z
                                        work[lane, end, 15] = signed
                                        work_i[lane, end, 7] = packed
                                        ready_ends = ready_ends + 1
                        if ready_ends == 2 and site_base + 2 <= site_capacity:
                            for end in tx.static_range(2):
                                site = site_base + end
                                parameter = work[lane, end, 8]
                                mcount = work_i[lane, end, 7]
                                weight[site] = 0.5
                                gap[site] = work[lane, end, 15]
                                normal[site, 0] = nx
                                normal[site, 1] = ny
                                normal[site, 2] = nz
                                common[site, 0] = (work[lane, end, 9] + work[lane, end, 12]) * 0.5
                                common[site, 1] = (work[lane, end, 10] + work[lane, end, 13]) * 0.5
                                common[site, 2] = (work[lane, end, 11] + work[lane, end, 14]) * 0.5
                                material_count[site, 0] = mcount
                                material_count[site, 1] = 2
                                material_ids[site, 1, 0] = original_ids[b0]
                                material_ids[site, 1, 1] = original_ids[b0 + 1]
                                barycentric[site, 1, 0] = 1.0 - parameter
                                barycentric[site, 1, 1] = parameter
                                for slot in tx.static_range(4):
                                    if slot < mcount:
                                        if end == 0:
                                            material_ids[site, 0, slot] = original_ids[
                                                work_i[lane, slot, 1]
                                            ]
                                            barycentric[site, 0, slot] = work[lane, slot, 5]
                                        else:
                                            material_ids[site, 0, slot] = original_ids[
                                                work_i[lane, slot, 2]
                                            ]
                                            barycentric[site, 0, slot] = work[lane, slot, 3]
                                surface[site, 0, 0] = work[lane, end, 9]
                                surface[site, 0, 1] = work[lane, end, 10]
                                surface[site, 0, 2] = work[lane, end, 11]
                                surface[site, 1, 0] = work[lane, end, 12]
                                surface[site, 1, 1] = work[lane, end, 13]
                                surface[site, 1, 2] = work[lane, end, 14]
                                skin[site, 0, 0] = 0.0
                                skin[site, 0, 1] = 0.0
                                skin[site, 0, 2] = 0.0
                                skin[site, 1, 0] = -radius_b * nx
                                skin[site, 1, 1] = -radius_b * ny
                                skin[site, 1, 2] = -radius_b * nz
                            status = 1
                            dimension = 1
                            count = 2
                        elif ready_ends == 2 and site_base + 2 > site_capacity:
                            status = 4
            elif kind_a == 2 and kind_b == 0 and radius_a == 0.0 and a1 > a0 and b1 == b0 + 1:
                bx = points[b0, 0]
                by = points[b0, 1]
                bz = points[b0, 2]
                best = -1.0e30
                for index in range(a0, a1):
                    height = dot3(points[index, 0], points[index, 1], points[index, 2], nx, ny, nz)
                    if height > best:
                        best = height
                face_count = 0
                for index in range(a0, a1):
                    height = dot3(points[index, 0], points[index, 1], points[index, 2], nx, ny, nz)
                    if best - height <= 2e-4 * scale and face_count < 8:
                        work[lane, face_count, 0] = points[index, 0]
                        work[lane, face_count, 1] = points[index, 1]
                        work[lane, face_count, 2] = points[index, 2]
                        work[lane, face_count, 4] = 0.0
                        work_i[lane, face_count, 0] = index
                        face_count = face_count + 1
                    elif best - height <= 2e-4 * scale:
                        face_count = 9
                ready = 0
                packed = 0
                s0x = 0.0
                s0y = 0.0
                s0z = 0.0
                s1x = bx - radius_b * nx
                s1y = by - radius_b * ny
                s1z = bz - radius_b * nz
                if face_count == 1:
                    vx = work[lane, 0, 0]
                    vy = work[lane, 0, 1]
                    vz = work[lane, 0, 2]
                    center_h = dot3(bx, by, bz, nx, ny, nz)
                    qx = bx + (best - center_h) * nx
                    qy = by + (best - center_h) * ny
                    qz = bz + (best - center_h) * nz
                    limit = 2e-4 * scale
                    error = dot3(qx - vx, qy - vy, qz - vz, qx - vx, qy - vy, qz - vz)
                    if error <= limit * limit:
                        s0x = vx
                        s0y = vy
                        s0z = vz
                        work_i[lane, 0, 1] = work_i[lane, 0, 0]
                        work[lane, 0, 5] = 1.0
                        packed = 1
                        ready = 1
                elif face_count >= 3 and face_count <= 8:
                    absolute_x = tx.abs(nx)
                    absolute_y = tx.abs(ny)
                    absolute_z = tx.abs(nz)
                    tangent_x = 0.0
                    tangent_y = 0.0
                    tangent_z = 0.0
                    if absolute_x <= absolute_y and absolute_x <= absolute_z:
                        tangent_y = -nz
                        tangent_z = ny
                    elif absolute_y <= absolute_z:
                        tangent_x = -nz
                        tangent_z = nx
                    else:
                        tangent_x = ny
                        tangent_y = -nx
                    tangent_length = tx.sqrt(
                        dot3(tangent_x, tangent_y, tangent_z, tangent_x, tangent_y, tangent_z)
                    )
                    if tangent_length > 1e-8:
                        tangent_x = tangent_x / tangent_length
                        tangent_y = tangent_y / tangent_length
                        tangent_z = tangent_z / tangent_length
                        bitangent_x = ny * tangent_z - nz * tangent_y
                        bitangent_y = nz * tangent_x - nx * tangent_z
                        bitangent_z = nx * tangent_y - ny * tangent_x
                        center_h = dot3(bx, by, bz, nx, ny, nz)
                        qx = bx + (best - center_h) * nx
                        qy = by + (best - center_h) * ny
                        qz = bz + (best - center_h) * nz
                        qu = dot3(qx, qy, qz, tangent_x, tangent_y, tangent_z)
                        qv = dot3(qx, qy, qz, bitangent_x, bitangent_y, bitangent_z)
                        center_u = 0.0
                        center_v = 0.0
                        count_f = 0.0
                        for slot in tx.static_range(8):
                            if slot < face_count:
                                uu = dot3(
                                    work[lane, slot, 0],
                                    work[lane, slot, 1],
                                    work[lane, slot, 2],
                                    tangent_x,
                                    tangent_y,
                                    tangent_z,
                                )
                                vv = dot3(
                                    work[lane, slot, 0],
                                    work[lane, slot, 1],
                                    work[lane, slot, 2],
                                    bitangent_x,
                                    bitangent_y,
                                    bitangent_z,
                                )
                                work[lane, slot, 6] = uu
                                work[lane, slot, 7] = vv
                                center_u = center_u + uu
                                center_v = center_v + vv
                                count_f = count_f + 1.0
                        center_u = center_u / count_f
                        center_v = center_v / count_f
                        qu = qu - center_u
                        qv = qv - center_v
                        for slot in tx.static_range(8):
                            if slot < face_count:
                                work[lane, slot, 6] = work[lane, slot, 6] - center_u
                                work[lane, slot, 7] = work[lane, slot, 7] - center_v
                        for _pass in tx.static_range(7):
                            for slot in tx.static_range(7):
                                if slot + 1 < face_count:
                                    left_u = work[lane, slot, 6]
                                    left_v = work[lane, slot, 7]
                                    right_u = work[lane, slot + 1, 6]
                                    right_v = work[lane, slot + 1, 7]
                                    left_negative = 0
                                    right_negative = 0
                                    if left_v < 0.0 or (left_v == 0.0 and left_u < 0.0):
                                        left_negative = 1
                                    if right_v < 0.0 or (right_v == 0.0 and right_u < 0.0):
                                        right_negative = 1
                                    swap = 0
                                    if left_negative != right_negative:
                                        if left_negative == 0:
                                            swap = 1
                                    else:
                                        cross = left_u * right_v - left_v * right_u
                                        if cross < 0.0:
                                            swap = 1
                                    if swap == 1:
                                        for column in tx.static_range(8):
                                            held = work[lane, slot, column]
                                            work[lane, slot, column] = work[lane, slot + 1, column]
                                            work[lane, slot + 1, column] = held
                                        held_index = work_i[lane, slot, 0]
                                        work_i[lane, slot, 0] = work_i[lane, slot + 1, 0]
                                        work_i[lane, slot + 1, 0] = held_index
                        chosen = 0
                        for tri in tx.static_range(6):
                            second = tri + 1
                            third = tri + 2
                            if chosen == 0 and third < face_count:
                                u0 = work[lane, 0, 6]
                                v0 = work[lane, 0, 7]
                                u1 = work[lane, second, 6]
                                v1 = work[lane, second, 7]
                                u2 = work[lane, third, 6]
                                v2 = work[lane, third, 7]
                                denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
                                if tx.abs(denom) > 1e-12:
                                    alpha = ((v1 - v2) * (qu - u2) + (u2 - u1) * (qv - v2)) / denom
                                    beta = ((v2 - v0) * (qu - u2) + (u0 - u2) * (qv - v2)) / denom
                                    gamma = 1.0 - alpha - beta
                                    if alpha >= -1e-5 and beta >= -1e-5 and gamma >= -1e-5:
                                        if alpha < 1e-5:
                                            alpha = 0.0
                                        if beta < 1e-5:
                                            beta = 0.0
                                        if gamma < 1e-5:
                                            gamma = 0.0
                                        if alpha < 0.0:
                                            alpha = 0.0
                                        if beta < 0.0:
                                            beta = 0.0
                                        if gamma < 0.0:
                                            gamma = 0.0
                                        total = alpha + beta + gamma
                                        if total > 0.0:
                                            work[lane, 0, 4] = alpha / total
                                            work[lane, second, 4] = beta / total
                                            work[lane, third, 4] = gamma / total
                                            chosen = 1
                        if chosen == 1:
                            rx = 0.0
                            ry = 0.0
                            rz = 0.0
                            used = 0
                            for slot in tx.static_range(8):
                                if slot < face_count and work[lane, slot, 4] > 0.0:
                                    rx = rx + work[lane, slot, 4] * work[lane, slot, 0]
                                    ry = ry + work[lane, slot, 4] * work[lane, slot, 1]
                                    rz = rz + work[lane, slot, 4] * work[lane, slot, 2]
                                    used = used + 1
                            error = dot3(rx - qx, ry - qy, rz - qz, rx - qx, ry - qy, rz - qz)
                            limit = 2e-4 * scale
                            if used >= 1 and used <= 4 and error <= limit * limit:
                                s0x = qx
                                s0y = qy
                                s0z = qz
                                for slot in tx.static_range(8):
                                    if (
                                        slot < face_count
                                        and work[lane, slot, 4] > 0.0
                                        and packed < 4
                                    ):
                                        work_i[lane, packed, 1] = work_i[lane, slot, 0]
                                        work[lane, packed, 5] = work[lane, slot, 4]
                                        packed = packed + 1
                                ready = 1
                signed = dot3(s1x - s0x, s1y - s0y, s1z - s0z, nx, ny, nz)
                if (
                    ready == 1
                    and packed >= 1
                    and tx.abs(signed - query_gap) <= 1e-4 * scale
                    and site_base + 1 <= site_capacity
                ):
                    site = site_base
                    weight[site] = 1.0
                    gap[site] = signed
                    normal[site, 0] = nx
                    normal[site, 1] = ny
                    normal[site, 2] = nz
                    common[site, 0] = (s0x + s1x) * 0.5
                    common[site, 1] = (s0y + s1y) * 0.5
                    common[site, 2] = (s0z + s1z) * 0.5
                    material_count[site, 0] = packed
                    material_count[site, 1] = 1
                    material_ids[site, 1, 0] = original_ids[b0]
                    barycentric[site, 1, 0] = 1.0
                    for slot in tx.static_range(4):
                        if slot < packed:
                            material_ids[site, 0, slot] = original_ids[work_i[lane, slot, 1]]
                            barycentric[site, 0, slot] = work[lane, slot, 5]
                    surface[site, 0, 0] = s0x
                    surface[site, 0, 1] = s0y
                    surface[site, 0, 2] = s0z
                    surface[site, 1, 0] = s1x
                    surface[site, 1, 1] = s1y
                    surface[site, 1, 2] = s1z
                    skin[site, 0, 0] = 0.0
                    skin[site, 0, 1] = 0.0
                    skin[site, 0, 2] = 0.0
                    skin[site, 1, 0] = -radius_b * nx
                    skin[site, 1, 1] = -radius_b * ny
                    skin[site, 1, 2] = -radius_b * nz
                    status = 1
                    dimension = 0
                    count = 1
                elif (
                    ready == 1
                    and packed >= 1
                    and tx.abs(signed - query_gap) <= 1e-4 * scale
                    and site_base + 1 > site_capacity
                ):
                    status = 4
            elif (
                kind_a == 2
                and kind_b == 2
                and radius_a == 0.0
                and radius_b == 0.0
                and a1 > a0
                and b1 > b0
            ):
                best = -1.0e30
                for index in range(a0, a1):
                    height = dot3(points[index, 0], points[index, 1], points[index, 2], nx, ny, nz)
                    if height > best:
                        best = height
                face_count = 0
                for index in range(a0, a1):
                    height = dot3(points[index, 0], points[index, 1], points[index, 2], nx, ny, nz)
                    if best - height <= 1e-5 * scale and face_count < 4:
                        work[lane, face_count, 0] = points[index, 0]
                        work[lane, face_count, 1] = points[index, 1]
                        work[lane, face_count, 2] = points[index, 2]
                        work_i[lane, face_count, 0] = index
                        face_count = face_count + 1
                    elif best - height <= 1e-5 * scale:
                        face_count = 5
                other_best = 1.0e30
                for index in range(b0, b1):
                    height = dot3(points[index, 0], points[index, 1], points[index, 2], nx, ny, nz)
                    if height < other_best:
                        other_best = height
                other_count = 0
                for index in range(b0, b1):
                    height = dot3(points[index, 0], points[index, 1], points[index, 2], nx, ny, nz)
                    if height - other_best <= 1e-5 * scale and other_count < 4:
                        work[lane, 4 + other_count, 0] = points[index, 0]
                        work[lane, 4 + other_count, 1] = points[index, 1]
                        work[lane, 4 + other_count, 2] = points[index, 2]
                        work_i[lane, 4 + other_count, 0] = index
                        other_count = other_count + 1
                    elif height - other_best <= 1e-5 * scale:
                        other_count = 5
                if face_count == 4 and other_count == 4 and site_base + 4 <= site_capacity:
                    absolute_x = tx.abs(nx)
                    absolute_y = tx.abs(ny)
                    absolute_z = tx.abs(nz)
                    tangent_x = 0.0
                    tangent_y = 0.0
                    tangent_z = 0.0
                    if absolute_x <= absolute_y and absolute_x <= absolute_z:
                        tangent_y = -nz
                        tangent_z = ny
                    elif absolute_y <= absolute_z:
                        tangent_x = -nz
                        tangent_z = nx
                    else:
                        tangent_x = ny
                        tangent_y = -nx
                    tangent_length = tx.sqrt(
                        dot3(tangent_x, tangent_y, tangent_z, tangent_x, tangent_y, tangent_z)
                    )
                    if tangent_length > 1e-8:
                        tangent_x = tangent_x / tangent_length
                        tangent_y = tangent_y / tangent_length
                        tangent_z = tangent_z / tangent_length
                        bitangent_x = ny * tangent_z - nz * tangent_y
                        bitangent_y = nz * tangent_x - nx * tangent_z
                        bitangent_z = nx * tangent_y - ny * tangent_x
                        for slot in tx.static_range(8):
                            work[lane, slot, 6] = dot3(
                                work[lane, slot, 0],
                                work[lane, slot, 1],
                                work[lane, slot, 2],
                                tangent_x,
                                tangent_y,
                                tangent_z,
                            )
                            work[lane, slot, 7] = dot3(
                                work[lane, slot, 0],
                                work[lane, slot, 1],
                                work[lane, slot, 2],
                                bitangent_x,
                                bitangent_y,
                                bitangent_z,
                            )
                        for _pass in tx.static_range(3):
                            for slot in tx.static_range(3):
                                left_u = work[lane, slot, 6]
                                left_v = work[lane, slot, 7]
                                right_u = work[lane, slot + 1, 6]
                                right_v = work[lane, slot + 1, 7]
                                left_negative = 0
                                right_negative = 0
                                if left_v < 0.0 or (left_v == 0.0 and left_u < 0.0):
                                    left_negative = 1
                                if right_v < 0.0 or (right_v == 0.0 and right_u < 0.0):
                                    right_negative = 1
                                swap = 0
                                if left_negative != right_negative:
                                    if left_negative == 0:
                                        swap = 1
                                else:
                                    cross = left_u * right_v - left_v * right_u
                                    if cross < 0.0:
                                        swap = 1
                                if swap == 1:
                                    for column in tx.static_range(8):
                                        held = work[lane, slot, column]
                                        work[lane, slot, column] = work[lane, slot + 1, column]
                                        work[lane, slot + 1, column] = held
                                    held_index = work_i[lane, slot, 0]
                                    work_i[lane, slot, 0] = work_i[lane, slot + 1, 0]
                                    work_i[lane, slot + 1, 0] = held_index
                        center_u = 0.0
                        center_v = 0.0
                        for slot in tx.static_range(4):
                            center_u = center_u + work[lane, slot, 6] * 0.25
                            center_v = center_v + work[lane, slot, 7] * 0.25
                        for slot in tx.static_range(4):
                            work[lane, slot, 6] = work[lane, slot, 6] - center_u
                            work[lane, slot, 7] = work[lane, slot, 7] - center_v
                        for _pass in tx.static_range(3):
                            for slot in tx.static_range(3):
                                left_u = work[lane, slot, 6]
                                left_v = work[lane, slot, 7]
                                right_u = work[lane, slot + 1, 6]
                                right_v = work[lane, slot + 1, 7]
                                left_negative = 0
                                right_negative = 0
                                if left_v < 0.0 or (left_v == 0.0 and left_u < 0.0):
                                    left_negative = 1
                                if right_v < 0.0 or (right_v == 0.0 and right_u < 0.0):
                                    right_negative = 1
                                swap = 0
                                if left_negative != right_negative:
                                    if left_negative == 0:
                                        swap = 1
                                else:
                                    cross = left_u * right_v - left_v * right_u
                                    if cross < 0.0:
                                        swap = 1
                                if swap == 1:
                                    for column in tx.static_range(8):
                                        held = work[lane, slot, column]
                                        work[lane, slot, column] = work[lane, slot + 1, column]
                                        work[lane, slot + 1, column] = held
                                    held_index = work_i[lane, slot, 0]
                                    work_i[lane, slot, 0] = work_i[lane, slot + 1, 0]
                                    work_i[lane, slot + 1, 0] = held_index
                        matched = 0
                        for slot in tx.static_range(4):
                            best_distance = 1.0e30
                            best_other = -1
                            for other in tx.static_range(4):
                                du = work[lane, slot, 6] + center_u - work[lane, 4 + other, 6]
                                dv = work[lane, slot, 7] + center_v - work[lane, 4 + other, 7]
                                distance = du * du + dv * dv
                                if distance < best_distance:
                                    best_distance = distance
                                    best_other = other
                            limit = 1e-4 * scale
                            if best_distance <= limit * limit:
                                work_i[lane, slot, 1] = best_other
                                matched = matched + 1
                            else:
                                work_i[lane, slot, 1] = -1
                        if matched == 4:
                            center_x = 0.0
                            center_y = 0.0
                            center_z = 0.0
                            for slot in tx.static_range(4):
                                center_x = center_x + work[lane, slot, 0] * 0.25
                                center_y = center_y + work[lane, slot, 1] * 0.25
                                center_z = center_z + work[lane, slot, 2] * 0.25
                            area_sum = 0.0
                            for slot in tx.static_range(4):
                                nxt = 0
                                if slot < 3:
                                    nxt = slot + 1
                                ux = work[lane, slot, 0] - center_x
                                uy = work[lane, slot, 1] - center_y
                                uz = work[lane, slot, 2] - center_z
                                vx = work[lane, nxt, 0] - center_x
                                vy = work[lane, nxt, 1] - center_y
                                vz = work[lane, nxt, 2] - center_z
                                cross_x = uy * vz - uz * vy
                                cross_y = uz * vx - ux * vz
                                cross_z = ux * vy - uy * vx
                                work[lane, slot, 8] = tx.abs(
                                    dot3(nx, ny, nz, cross_x, cross_y, cross_z)
                                )
                                area_sum = area_sum + work[lane, slot, 8]
                            if area_sum > 1e-12:
                                ready = 1
                                for slot in tx.static_range(4):
                                    previous = 3
                                    if slot > 0:
                                        previous = slot - 1
                                    site_weight = (
                                        work[lane, slot, 8] + work[lane, previous, 8]
                                    ) / (3.0 * area_sum) + 1.0 / 12.0
                                    other = work_i[lane, slot, 1]
                                    s0x = work[lane, slot, 0]
                                    s0y = work[lane, slot, 1]
                                    s0z = work[lane, slot, 2]
                                    s1x = work[lane, 4 + other, 0]
                                    s1y = work[lane, 4 + other, 1]
                                    s1z = work[lane, 4 + other, 2]
                                    signed = dot3(s1x - s0x, s1y - s0y, s1z - s0z, nx, ny, nz)
                                    if tx.abs(signed - query_gap) > 1e-4 * scale:
                                        ready = 0
                                    if ready == 1:
                                        site = site_base + slot
                                        first_id = original_ids[work_i[lane, slot, 0]]
                                        second_id = original_ids[work_i[lane, 4 + other, 0]]
                                        weight[site] = site_weight
                                        gap[site] = signed
                                        normal[site, 0] = nx
                                        normal[site, 1] = ny
                                        normal[site, 2] = nz
                                        common[site, 0] = (s0x + s1x) * 0.5
                                        common[site, 1] = (s0y + s1y) * 0.5
                                        common[site, 2] = (s0z + s1z) * 0.5
                                        material_count[site, 0] = 1
                                        material_count[site, 1] = 1
                                        material_ids[site, 0, 0] = first_id
                                        material_ids[site, 0, 1] = -1
                                        material_ids[site, 1, 0] = second_id
                                        material_ids[site, 1, 1] = -1
                                        barycentric[site, 0, 0] = 1.0
                                        barycentric[site, 0, 1] = 0.0
                                        barycentric[site, 1, 0] = 1.0
                                        barycentric[site, 1, 1] = 0.0
                                        surface[site, 0, 0] = s0x
                                        surface[site, 0, 1] = s0y
                                        surface[site, 0, 2] = s0z
                                        surface[site, 1, 0] = s1x
                                        surface[site, 1, 1] = s1y
                                        surface[site, 1, 2] = s1z
                                        skin[site, 0, 0] = 0.0
                                        skin[site, 0, 1] = 0.0
                                        skin[site, 0, 2] = 0.0
                                        skin[site, 1, 0] = 0.0
                                        skin[site, 1, 1] = 0.0
                                        skin[site, 1, 2] = 0.0
                                if ready == 1:
                                    status = 1
                                    dimension = 2
                                    count = 4
                elif face_count == 4 and other_count == 4 and site_base + 4 > site_capacity:
                    status = 4
                if status == 2:
                    best = -1.0e30
                    for index in range(a0, a1):
                        height = dot3(
                            points[index, 0], points[index, 1], points[index, 2], nx, ny, nz
                        )
                        if height > best:
                            best = height
                    face_count = 0
                    for index in range(a0, a1):
                        height = dot3(
                            points[index, 0], points[index, 1], points[index, 2], nx, ny, nz
                        )
                        if best - height <= 2e-4 * scale and face_count < 8:
                            work[lane, face_count, 0] = points[index, 0]
                            work[lane, face_count, 1] = points[index, 1]
                            work[lane, face_count, 2] = points[index, 2]
                            work_i[lane, face_count, 0] = index
                            face_count = face_count + 1
                        elif best - height <= 2e-4 * scale:
                            face_count = 9
                    other_best = 1.0e30
                    for index in range(b0, b1):
                        height = dot3(
                            points[index, 0], points[index, 1], points[index, 2], nx, ny, nz
                        )
                        if height < other_best:
                            other_best = height
                    other_count = 0
                    for index in range(b0, b1):
                        height = dot3(
                            points[index, 0], points[index, 1], points[index, 2], nx, ny, nz
                        )
                        if height - other_best <= 2e-4 * scale and other_count < 8:
                            work[lane, 8 + other_count, 0] = points[index, 0]
                            work[lane, 8 + other_count, 1] = points[index, 1]
                            work[lane, 8 + other_count, 2] = points[index, 2]
                            work_i[lane, 8 + other_count, 0] = index
                            other_count = other_count + 1
                        elif height - other_best <= 2e-4 * scale:
                            other_count = 9
                    if (
                        face_count >= 3
                        and face_count <= 8
                        and other_count >= 3
                        and other_count <= 8
                    ):
                        absolute_x = tx.abs(nx)
                        absolute_y = tx.abs(ny)
                        absolute_z = tx.abs(nz)
                        tangent_x = 0.0
                        tangent_y = 0.0
                        tangent_z = 0.0
                        if absolute_x <= absolute_y and absolute_x <= absolute_z:
                            tangent_y = -nz
                            tangent_z = ny
                        elif absolute_y <= absolute_z:
                            tangent_x = -nz
                            tangent_z = nx
                        else:
                            tangent_x = ny
                            tangent_y = -nx
                        tangent_length = tx.sqrt(
                            dot3(tangent_x, tangent_y, tangent_z, tangent_x, tangent_y, tangent_z)
                        )
                        if tangent_length > 1e-8:
                            tangent_x = tangent_x / tangent_length
                            tangent_y = tangent_y / tangent_length
                            tangent_z = tangent_z / tangent_length
                            bitangent_x = ny * tangent_z - nz * tangent_y
                            bitangent_y = nz * tangent_x - nx * tangent_z
                            bitangent_z = nx * tangent_y - ny * tangent_x
                            center_u0 = 0.0
                            center_v0 = 0.0
                            center_u1 = 0.0
                            center_v1 = 0.0
                            for which in tx.static_range(2):
                                poly_base = 0
                                poly_count = face_count
                                if which == 1:
                                    poly_base = 8
                                    poly_count = other_count
                                center_u = 0.0
                                center_v = 0.0
                                count_f = 0.0
                                for slot in tx.static_range(8):
                                    if slot < poly_count:
                                        uu = dot3(
                                            work[lane, poly_base + slot, 0],
                                            work[lane, poly_base + slot, 1],
                                            work[lane, poly_base + slot, 2],
                                            tangent_x,
                                            tangent_y,
                                            tangent_z,
                                        )
                                        vv = dot3(
                                            work[lane, poly_base + slot, 0],
                                            work[lane, poly_base + slot, 1],
                                            work[lane, poly_base + slot, 2],
                                            bitangent_x,
                                            bitangent_y,
                                            bitangent_z,
                                        )
                                        work[lane, poly_base + slot, 3] = uu
                                        work[lane, poly_base + slot, 4] = vv
                                        work[lane, poly_base + slot, 6] = uu
                                        work[lane, poly_base + slot, 7] = vv
                                        center_u = center_u + uu
                                        center_v = center_v + vv
                                        count_f = count_f + 1.0
                                center_u = center_u / count_f
                                center_v = center_v / count_f
                                if which == 0:
                                    center_u0 = center_u
                                    center_v0 = center_v
                                else:
                                    center_u1 = center_u
                                    center_v1 = center_v
                                for slot in tx.static_range(8):
                                    if slot < poly_count:
                                        work[lane, poly_base + slot, 6] = (
                                            work[lane, poly_base + slot, 6] - center_u
                                        )
                                        work[lane, poly_base + slot, 7] = (
                                            work[lane, poly_base + slot, 7] - center_v
                                        )
                                for _pass in tx.static_range(7):
                                    for slot in tx.static_range(7):
                                        if slot + 1 < poly_count:
                                            left_u = work[lane, poly_base + slot, 6]
                                            left_v = work[lane, poly_base + slot, 7]
                                            right_u = work[lane, poly_base + slot + 1, 6]
                                            right_v = work[lane, poly_base + slot + 1, 7]
                                            left_negative = 0
                                            right_negative = 0
                                            if left_v < 0.0 or (left_v == 0.0 and left_u < 0.0):
                                                left_negative = 1
                                            if right_v < 0.0 or (right_v == 0.0 and right_u < 0.0):
                                                right_negative = 1
                                            swap = 0
                                            if left_negative != right_negative:
                                                if left_negative == 0:
                                                    swap = 1
                                            else:
                                                cross = left_u * right_v - left_v * right_u
                                                if cross < 0.0:
                                                    swap = 1
                                            if swap == 1:
                                                for column in tx.static_range(8):
                                                    held = work[lane, poly_base + slot, column]
                                                    work[lane, poly_base + slot, column] = work[
                                                        lane, poly_base + slot + 1, column
                                                    ]
                                                    work[lane, poly_base + slot + 1, column] = held
                                                held_index = work_i[lane, poly_base + slot, 0]
                                                work_i[lane, poly_base + slot, 0] = work_i[
                                                    lane, poly_base + slot + 1, 0
                                                ]
                                                work_i[lane, poly_base + slot + 1, 0] = held_index
                            a_inside = 1
                            b_inside = 1
                            band = 1e-4 * scale
                            for slot in tx.static_range(8):
                                if slot < face_count and a_inside == 1:
                                    uu = work[lane, slot, 3]
                                    vv = work[lane, slot, 4]
                                    for edge in tx.static_range(8):
                                        if edge < other_count and a_inside == 1:
                                            nxt = edge + 1
                                            if nxt == other_count:
                                                nxt = 0
                                            ex = work[lane, 8 + nxt, 3] - work[lane, 8 + edge, 3]
                                            ey = work[lane, 8 + nxt, 4] - work[lane, 8 + edge, 4]
                                            side = ex * (vv - work[lane, 8 + edge, 4]) - ey * (
                                                uu - work[lane, 8 + edge, 3]
                                            )
                                            if side < -band:
                                                a_inside = 0
                            for slot in tx.static_range(8):
                                if slot < other_count and b_inside == 1:
                                    uu = work[lane, 8 + slot, 3]
                                    vv = work[lane, 8 + slot, 4]
                                    for edge in tx.static_range(8):
                                        if edge < face_count and b_inside == 1:
                                            nxt = edge + 1
                                            if nxt == face_count:
                                                nxt = 0
                                            ex = work[lane, nxt, 3] - work[lane, edge, 3]
                                            ey = work[lane, nxt, 4] - work[lane, edge, 4]
                                            side = ex * (vv - work[lane, edge, 4]) - ey * (
                                                uu - work[lane, edge, 3]
                                            )
                                            if side < -band:
                                                b_inside = 0
                            emit = 0
                            emit_b = 0
                            if b_inside == 1 and a_inside == 0:
                                emit = 1
                                emit_b = 1
                            elif a_inside == 1:
                                emit = 1
                            if emit == 1:
                                inner_base = 0
                                outer_base = 8
                                inner_count = face_count
                                outer_count = other_count
                                outer_u = center_u1
                                outer_v = center_v1
                                plane = other_best
                                if emit_b == 1:
                                    inner_base = 8
                                    outer_base = 0
                                    inner_count = other_count
                                    outer_count = face_count
                                    outer_u = center_u0
                                    outer_v = center_v0
                                    plane = best
                                ready_verts = 0
                                for vert in tx.static_range(8):
                                    if vert < inner_count:
                                        for slot in tx.static_range(8):
                                            if slot < outer_count:
                                                work[lane, outer_base + slot, 5] = 0.0
                                        iu = work[lane, inner_base + vert, 3]
                                        iv = work[lane, inner_base + vert, 4]
                                        best_distance = 1.0e30
                                        best_slot = 0
                                        for slot in tx.static_range(8):
                                            if slot < outer_count:
                                                du = iu - work[lane, outer_base + slot, 3]
                                                dv = iv - work[lane, outer_base + slot, 4]
                                                distance = du * du + dv * dv
                                                if distance < best_distance:
                                                    best_distance = distance
                                                    best_slot = slot
                                        limit = 1e-4 * scale
                                        chosen = 0
                                        if best_distance <= limit * limit:
                                            work[lane, outer_base + best_slot, 5] = 1.0
                                            chosen = 1
                                        else:
                                            qu = iu - outer_u
                                            qv = iv - outer_v
                                            for tri in tx.static_range(6):
                                                second = tri + 1
                                                third = tri + 2
                                                if chosen == 0 and third < outer_count:
                                                    u0 = work[lane, outer_base, 6]
                                                    v0 = work[lane, outer_base, 7]
                                                    u1 = work[lane, outer_base + second, 6]
                                                    v1 = work[lane, outer_base + second, 7]
                                                    u2 = work[lane, outer_base + third, 6]
                                                    v2 = work[lane, outer_base + third, 7]
                                                    denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (
                                                        v0 - v2
                                                    )
                                                    if tx.abs(denom) > 1e-12:
                                                        alpha = (
                                                            (v1 - v2) * (qu - u2)
                                                            + (u2 - u1) * (qv - v2)
                                                        ) / denom
                                                        beta = (
                                                            (v2 - v0) * (qu - u2)
                                                            + (u0 - u2) * (qv - v2)
                                                        ) / denom
                                                        gamma = 1.0 - alpha - beta
                                                        if (
                                                            alpha >= -1e-5
                                                            and beta >= -1e-5
                                                            and gamma >= -1e-5
                                                        ):
                                                            if alpha < 1e-5:
                                                                alpha = 0.0
                                                            if beta < 1e-5:
                                                                beta = 0.0
                                                            if gamma < 1e-5:
                                                                gamma = 0.0
                                                            if alpha < 0.0:
                                                                alpha = 0.0
                                                            if beta < 0.0:
                                                                beta = 0.0
                                                            if gamma < 0.0:
                                                                gamma = 0.0
                                                            total = alpha + beta + gamma
                                                            if total > 0.0:
                                                                work[lane, outer_base, 5] = (
                                                                    alpha / total
                                                                )
                                                                work[
                                                                    lane, outer_base + second, 5
                                                                ] = beta / total
                                                                work[
                                                                    lane, outer_base + third, 5
                                                                ] = gamma / total
                                                                chosen = 1
                                        if chosen == 1:
                                            px = work[lane, inner_base + vert, 0]
                                            py = work[lane, inner_base + vert, 1]
                                            pz = work[lane, inner_base + vert, 2]
                                            core_h = dot3(px, py, pz, nx, ny, nz)
                                            hx = px + (plane - core_h) * nx
                                            hy = py + (plane - core_h) * ny
                                            hz = pz + (plane - core_h) * nz
                                            rx = 0.0
                                            ry = 0.0
                                            rz = 0.0
                                            used = 0
                                            for slot in tx.static_range(8):
                                                if (
                                                    slot < outer_count
                                                    and work[lane, outer_base + slot, 5] > 0.0
                                                ):
                                                    rx = (
                                                        rx
                                                        + work[lane, outer_base + slot, 5]
                                                        * work[lane, outer_base + slot, 0]
                                                    )
                                                    ry = (
                                                        ry
                                                        + work[lane, outer_base + slot, 5]
                                                        * work[lane, outer_base + slot, 1]
                                                    )
                                                    rz = (
                                                        rz
                                                        + work[lane, outer_base + slot, 5]
                                                        * work[lane, outer_base + slot, 2]
                                                    )
                                                    used = used + 1
                                            error = dot3(
                                                rx - hx, ry - hy, rz - hz, rx - hx, ry - hy, rz - hz
                                            )
                                            bound = 2e-4 * scale
                                            s0x = rx
                                            s0y = ry
                                            s0z = rz
                                            s1x = px
                                            s1y = py
                                            s1z = pz
                                            if emit_b == 0:
                                                s0x = px
                                                s0y = py
                                                s0z = pz
                                                s1x = rx
                                                s1y = ry
                                                s1z = rz
                                            signed = dot3(
                                                s1x - s0x, s1y - s0y, s1z - s0z, nx, ny, nz
                                            )
                                            if (
                                                used >= 1
                                                and used <= 4
                                                and error <= bound * bound
                                                and tx.abs(signed - query_gap) <= 1e-4 * scale
                                            ):
                                                packed = 0
                                                for slot in tx.static_range(8):
                                                    if (
                                                        slot < outer_count
                                                        and work[lane, outer_base + slot, 5] > 0.0
                                                        and packed < 4
                                                    ):
                                                        if packed == 0:
                                                            work_i[lane, inner_base + vert, 2] = (
                                                                work_i[lane, outer_base + slot, 0]
                                                            )
                                                            work[lane, inner_base + vert, 10] = (
                                                                work[lane, outer_base + slot, 5]
                                                            )
                                                        elif packed == 1:
                                                            work_i[lane, inner_base + vert, 3] = (
                                                                work_i[lane, outer_base + slot, 0]
                                                            )
                                                            work[lane, inner_base + vert, 11] = (
                                                                work[lane, outer_base + slot, 5]
                                                            )
                                                        elif packed == 2:
                                                            work_i[lane, inner_base + vert, 4] = (
                                                                work_i[lane, outer_base + slot, 0]
                                                            )
                                                            work[lane, inner_base + vert, 12] = (
                                                                work[lane, outer_base + slot, 5]
                                                            )
                                                        else:
                                                            work_i[lane, inner_base + vert, 5] = (
                                                                work_i[lane, outer_base + slot, 0]
                                                            )
                                                            work[lane, inner_base + vert, 13] = (
                                                                work[lane, outer_base + slot, 5]
                                                            )
                                                        packed = packed + 1
                                                work[lane, inner_base + vert, 5] = rx
                                                work[lane, inner_base + vert, 14] = ry
                                                work[lane, inner_base + vert, 15] = rz
                                                work_i[lane, inner_base + vert, 1] = packed
                                                ready_verts = ready_verts + 1
                                if ready_verts == inner_count:
                                    center_x = 0.0
                                    center_y = 0.0
                                    center_z = 0.0
                                    count_f = 0.0
                                    for slot in tx.static_range(8):
                                        if slot < inner_count:
                                            center_x = center_x + work[lane, inner_base + slot, 0]
                                            center_y = center_y + work[lane, inner_base + slot, 1]
                                            center_z = center_z + work[lane, inner_base + slot, 2]
                                            count_f = count_f + 1.0
                                    center_x = center_x / count_f
                                    center_y = center_y / count_f
                                    center_z = center_z / count_f
                                    area_sum = 0.0
                                    for slot in tx.static_range(8):
                                        if slot < inner_count:
                                            nxt = slot + 1
                                            if nxt == inner_count:
                                                nxt = 0
                                            ux = work[lane, inner_base + slot, 0] - center_x
                                            uy = work[lane, inner_base + slot, 1] - center_y
                                            uz = work[lane, inner_base + slot, 2] - center_z
                                            vx = work[lane, inner_base + nxt, 0] - center_x
                                            vy = work[lane, inner_base + nxt, 1] - center_y
                                            vz = work[lane, inner_base + nxt, 2] - center_z
                                            cross_x = uy * vz - uz * vy
                                            cross_y = uz * vx - ux * vz
                                            cross_z = ux * vy - uy * vx
                                            work[lane, inner_base + slot, 8] = tx.abs(
                                                dot3(nx, ny, nz, cross_x, cross_y, cross_z)
                                            )
                                            area_sum = area_sum + work[lane, inner_base + slot, 8]
                                    if (
                                        area_sum > 1e-12
                                        and site_base + inner_count <= site_capacity
                                    ):
                                        for slot in tx.static_range(8):
                                            if slot < inner_count:
                                                previous = slot - 1
                                                if slot == 0:
                                                    previous = inner_count - 1
                                                work[lane, inner_base + slot, 9] = (
                                                    work[lane, inner_base + slot, 8]
                                                    + work[lane, inner_base + previous, 8]
                                                ) / (3.0 * area_sum) + 1.0 / (3.0 * count_f)
                                        for slot in tx.static_range(8):
                                            if slot < inner_count:
                                                site = site_base + slot
                                                px = work[lane, inner_base + slot, 0]
                                                py = work[lane, inner_base + slot, 1]
                                                pz = work[lane, inner_base + slot, 2]
                                                ox = work[lane, inner_base + slot, 5]
                                                oy = work[lane, inner_base + slot, 14]
                                                oz = work[lane, inner_base + slot, 15]
                                                s0x = ox
                                                s0y = oy
                                                s0z = oz
                                                s1x = px
                                                s1y = py
                                                s1z = pz
                                                if emit_b == 0:
                                                    s0x = px
                                                    s0y = py
                                                    s0z = pz
                                                    s1x = ox
                                                    s1y = oy
                                                    s1z = oz
                                                signed = dot3(
                                                    s1x - s0x, s1y - s0y, s1z - s0z, nx, ny, nz
                                                )
                                                mcount = work_i[lane, inner_base + slot, 1]
                                                weight[site] = work[lane, inner_base + slot, 9]
                                                gap[site] = signed
                                                normal[site, 0] = nx
                                                normal[site, 1] = ny
                                                normal[site, 2] = nz
                                                common[site, 0] = (s0x + s1x) * 0.5
                                                common[site, 1] = (s0y + s1y) * 0.5
                                                common[site, 2] = (s0z + s1z) * 0.5
                                                if emit_b == 1:
                                                    material_count[site, 1] = 1
                                                    material_ids[site, 1, 0] = original_ids[
                                                        work_i[lane, inner_base + slot, 0]
                                                    ]
                                                    barycentric[site, 1, 0] = 1.0
                                                    material_count[site, 0] = mcount
                                                else:
                                                    material_count[site, 0] = 1
                                                    material_ids[site, 0, 0] = original_ids[
                                                        work_i[lane, inner_base + slot, 0]
                                                    ]
                                                    barycentric[site, 0, 0] = 1.0
                                                    material_count[site, 1] = mcount
                                                if mcount > 0:
                                                    if emit_b == 1:
                                                        material_ids[site, 0, 0] = original_ids[
                                                            work_i[lane, inner_base + slot, 2]
                                                        ]
                                                        barycentric[site, 0, 0] = work[
                                                            lane, inner_base + slot, 10
                                                        ]
                                                    else:
                                                        material_ids[site, 1, 0] = original_ids[
                                                            work_i[lane, inner_base + slot, 2]
                                                        ]
                                                        barycentric[site, 1, 0] = work[
                                                            lane, inner_base + slot, 10
                                                        ]
                                                if mcount > 1:
                                                    if emit_b == 1:
                                                        material_ids[site, 0, 1] = original_ids[
                                                            work_i[lane, inner_base + slot, 3]
                                                        ]
                                                        barycentric[site, 0, 1] = work[
                                                            lane, inner_base + slot, 11
                                                        ]
                                                    else:
                                                        material_ids[site, 1, 1] = original_ids[
                                                            work_i[lane, inner_base + slot, 3]
                                                        ]
                                                        barycentric[site, 1, 1] = work[
                                                            lane, inner_base + slot, 11
                                                        ]
                                                if mcount > 2:
                                                    if emit_b == 1:
                                                        material_ids[site, 0, 2] = original_ids[
                                                            work_i[lane, inner_base + slot, 4]
                                                        ]
                                                        barycentric[site, 0, 2] = work[
                                                            lane, inner_base + slot, 12
                                                        ]
                                                    else:
                                                        material_ids[site, 1, 2] = original_ids[
                                                            work_i[lane, inner_base + slot, 4]
                                                        ]
                                                        barycentric[site, 1, 2] = work[
                                                            lane, inner_base + slot, 12
                                                        ]
                                                if mcount > 3:
                                                    if emit_b == 1:
                                                        material_ids[site, 0, 3] = original_ids[
                                                            work_i[lane, inner_base + slot, 5]
                                                        ]
                                                        barycentric[site, 0, 3] = work[
                                                            lane, inner_base + slot, 13
                                                        ]
                                                    else:
                                                        material_ids[site, 1, 3] = original_ids[
                                                            work_i[lane, inner_base + slot, 5]
                                                        ]
                                                        barycentric[site, 1, 3] = work[
                                                            lane, inner_base + slot, 13
                                                        ]
                                                surface[site, 0, 0] = s0x
                                                surface[site, 0, 1] = s0y
                                                surface[site, 0, 2] = s0z
                                                surface[site, 1, 0] = s1x
                                                surface[site, 1, 1] = s1y
                                                surface[site, 1, 2] = s1z
                                                skin[site, 0, 0] = 0.0
                                                skin[site, 0, 1] = 0.0
                                                skin[site, 0, 2] = 0.0
                                                skin[site, 1, 0] = 0.0
                                                skin[site, 1, 1] = 0.0
                                                skin[site, 1, 2] = 0.0
                                        status = 1
                                        dimension = 2
                                        count = inner_count
                                    elif (
                                        area_sum > 1e-12 and site_base + inner_count > site_capacity
                                    ):
                                        status = 4
                            if emit == 0:
                                # Two finite convex support polygons: clip A by every closed B edge.
                                # Rows 0..15 retain original material vertices. Rows 16..31 are
                                # bounded intersection scratch; no host face search or device append.
                                clip_count = face_count
                                clip_ok = 1
                                for clip_i in tx.static_range(16):
                                    for clip_column in tx.static_range(24):
                                        work[lane, 16 + clip_i, clip_column] = 0.0
                                    for clip_column in tx.static_range(8):
                                        work_i[lane, 16 + clip_i, clip_column] = -1
                                    if clip_i < clip_count:
                                        work[lane, 16 + clip_i, 0] = work[lane, clip_i, 3]
                                        work[lane, 16 + clip_i, 1] = work[lane, clip_i, 4]
                                for clip_edge in tx.static_range(8):
                                    if clip_edge < other_count and clip_ok == 1 and clip_count >= 3:
                                        clip_next = clip_edge + 1
                                        if clip_next == other_count:
                                            clip_next = 0
                                        clip_ax = work[lane, 8 + clip_edge, 3]
                                        clip_ay = work[lane, 8 + clip_edge, 4]
                                        clip_ex = work[lane, 8 + clip_next, 3] - clip_ax
                                        clip_ey = work[lane, 8 + clip_next, 4] - clip_ay
                                        clip_written = 0
                                        for clip_i in tx.static_range(16):
                                            if clip_i < clip_count and clip_ok == 1:
                                                clip_previous = clip_i - 1
                                                if clip_i == 0:
                                                    clip_previous = clip_count - 1
                                                clip_px = work[lane, 16 + clip_previous, 0]
                                                clip_py = work[lane, 16 + clip_previous, 1]
                                                clip_cx = work[lane, 16 + clip_i, 0]
                                                clip_cy = work[lane, 16 + clip_i, 1]
                                                clip_ps = clip_ex * (
                                                    clip_py - clip_ay
                                                ) - clip_ey * (clip_px - clip_ax)
                                                clip_cs = clip_ex * (
                                                    clip_cy - clip_ay
                                                ) - clip_ey * (clip_cx - clip_ax)
                                                if (clip_ps < 0.0 and clip_cs >= 0.0) or (
                                                    clip_ps >= 0.0 and clip_cs < 0.0
                                                ):
                                                    clip_t = clip_ps / (clip_ps - clip_cs)
                                                    clip_x = clip_px + clip_t * (clip_cx - clip_px)
                                                    clip_y = clip_py + clip_t * (clip_cy - clip_py)
                                                    clip_duplicate = 0
                                                    if clip_written > 0:
                                                        if (
                                                            clip_x
                                                            == work[lane, 16 + clip_written - 1, 2]
                                                            and clip_y
                                                            == work[lane, 16 + clip_written - 1, 3]
                                                        ):
                                                            clip_duplicate = 1
                                                    if clip_duplicate == 0:
                                                        if clip_written < 16:
                                                            work[lane, 16 + clip_written, 2] = (
                                                                clip_x
                                                            )
                                                            work[lane, 16 + clip_written, 3] = (
                                                                clip_y
                                                            )
                                                            clip_written = clip_written + 1
                                                        else:
                                                            clip_ok = 0
                                                if clip_cs >= 0.0 and clip_ok == 1:
                                                    clip_duplicate = 0
                                                    if clip_written > 0:
                                                        if (
                                                            clip_cx
                                                            == work[lane, 16 + clip_written - 1, 2]
                                                            and clip_cy
                                                            == work[lane, 16 + clip_written - 1, 3]
                                                        ):
                                                            clip_duplicate = 1
                                                    if clip_duplicate == 0:
                                                        if clip_written < 16:
                                                            work[lane, 16 + clip_written, 2] = (
                                                                clip_cx
                                                            )
                                                            work[lane, 16 + clip_written, 3] = (
                                                                clip_cy
                                                            )
                                                            clip_written = clip_written + 1
                                                        else:
                                                            clip_ok = 0
                                        if clip_written > 1:
                                            if (
                                                work[lane, 16, 2]
                                                == work[lane, 16 + clip_written - 1, 2]
                                                and work[lane, 16, 3]
                                                == work[lane, 16 + clip_written - 1, 3]
                                            ):
                                                clip_written = clip_written - 1
                                        clip_count = clip_written
                                        for clip_i in tx.static_range(16):
                                            if clip_i < clip_count:
                                                work[lane, 16 + clip_i, 0] = work[
                                                    lane, 16 + clip_i, 2
                                                ]
                                                work[lane, 16 + clip_i, 1] = work[
                                                    lane, 16 + clip_i, 3
                                                ]
                                if clip_ok == 0:
                                    status = 4
                                if clip_ok == 1 and clip_count >= 3:
                                    clip_ready = 0
                                    clip_cu = 0.0
                                    clip_cv = 0.0
                                    clip_nf = 0.0
                                    for clip_i in tx.static_range(16):
                                        if clip_i < clip_count:
                                            clip_cu = clip_cu + work[lane, 16 + clip_i, 0]
                                            clip_cv = clip_cv + work[lane, 16 + clip_i, 1]
                                            clip_nf = clip_nf + 1.0
                                            for clip_side in tx.static_range(2):
                                                clip_base = 0
                                                clip_n = face_count
                                                if clip_side == 1:
                                                    clip_base = 8
                                                    clip_n = other_count
                                                clip_u = work[lane, 16 + clip_i, 0]
                                                clip_v = work[lane, 16 + clip_i, 1]
                                                clip_found = 0
                                                for clip_j in tx.static_range(8):
                                                    if clip_j < clip_n:
                                                        work[lane, clip_base + clip_j, 5] = 0.0
                                                for clip_tri in tx.static_range(6):
                                                    clip_j = clip_tri + 1
                                                    clip_k = clip_tri + 2
                                                    if clip_found == 0 and clip_k < clip_n:
                                                        clip_u0 = work[lane, clip_base, 3]
                                                        clip_v0 = work[lane, clip_base, 4]
                                                        clip_u1 = work[lane, clip_base + clip_j, 3]
                                                        clip_v1 = work[lane, clip_base + clip_j, 4]
                                                        clip_u2 = work[lane, clip_base + clip_k, 3]
                                                        clip_v2 = work[lane, clip_base + clip_k, 4]
                                                        clip_den = (clip_v1 - clip_v2) * (
                                                            clip_u0 - clip_u2
                                                        ) + (clip_u2 - clip_u1) * (
                                                            clip_v0 - clip_v2
                                                        )
                                                        if tx.abs(clip_den) > 1e-12:
                                                            clip_alpha = (
                                                                (clip_v1 - clip_v2)
                                                                * (clip_u - clip_u2)
                                                                + (clip_u2 - clip_u1)
                                                                * (clip_v - clip_v2)
                                                            ) / clip_den
                                                            clip_beta = (
                                                                (clip_v2 - clip_v0)
                                                                * (clip_u - clip_u2)
                                                                + (clip_u0 - clip_u2)
                                                                * (clip_v - clip_v2)
                                                            ) / clip_den
                                                            clip_gamma = (
                                                                1.0 - clip_alpha - clip_beta
                                                            )
                                                            if (
                                                                clip_alpha >= -1e-5
                                                                and clip_beta >= -1e-5
                                                                and clip_gamma >= -1e-5
                                                            ):
                                                                clip_alpha = tx.max(0.0, clip_alpha)
                                                                clip_beta = tx.max(0.0, clip_beta)
                                                                clip_gamma = tx.max(0.0, clip_gamma)
                                                                clip_total = (
                                                                    clip_alpha
                                                                    + clip_beta
                                                                    + clip_gamma
                                                                )
                                                                work[lane, clip_base, 5] = (
                                                                    clip_alpha / clip_total
                                                                )
                                                                work[
                                                                    lane, clip_base + clip_j, 5
                                                                ] = clip_beta / clip_total
                                                                work[
                                                                    lane, clip_base + clip_k, 5
                                                                ] = clip_gamma / clip_total
                                                                clip_found = 1
                                                if clip_found == 1:
                                                    clip_packed = 0
                                                    clip_rx = 0.0
                                                    clip_ry = 0.0
                                                    clip_rz = 0.0
                                                    for clip_j in tx.static_range(8):
                                                        if (
                                                            clip_j < clip_n
                                                            and work[lane, clip_base + clip_j, 5]
                                                            > 0.0
                                                        ):
                                                            clip_w = work[
                                                                lane, clip_base + clip_j, 5
                                                            ]
                                                            clip_rx = (
                                                                clip_rx
                                                                + clip_w
                                                                * work[lane, clip_base + clip_j, 0]
                                                            )
                                                            clip_ry = (
                                                                clip_ry
                                                                + clip_w
                                                                * work[lane, clip_base + clip_j, 1]
                                                            )
                                                            clip_rz = (
                                                                clip_rz
                                                                + clip_w
                                                                * work[lane, clip_base + clip_j, 2]
                                                            )
                                                            if clip_packed < 3:
                                                                work_i[
                                                                    lane,
                                                                    16 + clip_i,
                                                                    3 * clip_side + clip_packed,
                                                                ] = work_i[
                                                                    lane, clip_base + clip_j, 0
                                                                ]
                                                                work[
                                                                    lane,
                                                                    16 + clip_i,
                                                                    5 + 3 * clip_side + clip_packed,
                                                                ] = clip_w
                                                            clip_packed = clip_packed + 1
                                                    clip_eu = (
                                                        dot3(
                                                            clip_rx,
                                                            clip_ry,
                                                            clip_rz,
                                                            tangent_x,
                                                            tangent_y,
                                                            tangent_z,
                                                        )
                                                        - clip_u
                                                    )
                                                    clip_ev = (
                                                        dot3(
                                                            clip_rx,
                                                            clip_ry,
                                                            clip_rz,
                                                            bitangent_x,
                                                            bitangent_y,
                                                            bitangent_z,
                                                        )
                                                        - clip_v
                                                    )
                                                    if (
                                                        clip_packed >= 1
                                                        and clip_packed <= 3
                                                        and clip_eu * clip_eu + clip_ev * clip_ev
                                                        <= 4e-8 * scale * scale
                                                    ):
                                                        work_i[lane, 16 + clip_i, 6 + clip_side] = (
                                                            clip_packed
                                                        )
                                                        work[
                                                            lane, 16 + clip_i, 11 + 3 * clip_side
                                                        ] = clip_rx
                                                        work[
                                                            lane, 16 + clip_i, 12 + 3 * clip_side
                                                        ] = clip_ry
                                                        work[
                                                            lane, 16 + clip_i, 13 + 3 * clip_side
                                                        ] = clip_rz
                                                        clip_ready = clip_ready + 1
                                    clip_cu = clip_cu / clip_nf
                                    clip_cv = clip_cv / clip_nf
                                    clip_area = 0.0
                                    for clip_i in tx.static_range(16):
                                        if clip_i < clip_count:
                                            clip_next = clip_i + 1
                                            if clip_next == clip_count:
                                                clip_next = 0
                                            clip_ux = work[lane, 16 + clip_i, 0] - clip_cu
                                            clip_uy = work[lane, 16 + clip_i, 1] - clip_cv
                                            clip_vx = work[lane, 16 + clip_next, 0] - clip_cu
                                            clip_vy = work[lane, 16 + clip_next, 1] - clip_cv
                                            clip_twice = clip_ux * clip_vy - clip_uy * clip_vx
                                            if clip_twice < 0.0:
                                                clip_ready = -100
                                            work[lane, 16 + clip_i, 17] = clip_twice
                                            clip_area = clip_area + clip_twice
                                            clip_dx = (
                                                work[lane, 16 + clip_i, 14]
                                                - work[lane, 16 + clip_i, 11]
                                            )
                                            clip_dy = (
                                                work[lane, 16 + clip_i, 15]
                                                - work[lane, 16 + clip_i, 12]
                                            )
                                            clip_dz = (
                                                work[lane, 16 + clip_i, 16]
                                                - work[lane, 16 + clip_i, 13]
                                            )
                                            clip_signed = dot3(
                                                clip_dx, clip_dy, clip_dz, nx, ny, nz
                                            )
                                            if tx.abs(clip_signed - query_gap) > 1e-4 * scale:
                                                clip_ready = -100
                                    if clip_ready == 2 * clip_count and clip_area > 1e-12:
                                        if site_base + clip_count > site_capacity:
                                            status = 4
                                        else:
                                            for clip_i in tx.static_range(16):
                                                if clip_i < clip_count:
                                                    clip_previous = clip_i - 1
                                                    if clip_i == 0:
                                                        clip_previous = clip_count - 1
                                                    clip_site = site_base + clip_i
                                                    weight[clip_site] = (
                                                        work[lane, 16 + clip_i, 17]
                                                        + work[lane, 16 + clip_previous, 17]
                                                    ) / (3.0 * clip_area) + 1.0 / (3.0 * clip_nf)
                                                    clip_dx = (
                                                        work[lane, 16 + clip_i, 14]
                                                        - work[lane, 16 + clip_i, 11]
                                                    )
                                                    clip_dy = (
                                                        work[lane, 16 + clip_i, 15]
                                                        - work[lane, 16 + clip_i, 12]
                                                    )
                                                    clip_dz = (
                                                        work[lane, 16 + clip_i, 16]
                                                        - work[lane, 16 + clip_i, 13]
                                                    )
                                                    gap[clip_site] = dot3(
                                                        clip_dx, clip_dy, clip_dz, nx, ny, nz
                                                    )
                                                    normal[clip_site, 0] = nx
                                                    normal[clip_site, 1] = ny
                                                    normal[clip_site, 2] = nz
                                                    for clip_axis in tx.static_range(3):
                                                        common[clip_site, clip_axis] = 0.5 * (
                                                            work[lane, 16 + clip_i, 11 + clip_axis]
                                                            + work[
                                                                lane, 16 + clip_i, 14 + clip_axis
                                                            ]
                                                        )
                                                    for clip_side in tx.static_range(2):
                                                        clip_mcount = work_i[
                                                            lane, 16 + clip_i, 6 + clip_side
                                                        ]
                                                        material_count[clip_site, clip_side] = (
                                                            clip_mcount
                                                        )
                                                        for clip_j in tx.static_range(3):
                                                            if clip_j < clip_mcount:
                                                                material_ids[
                                                                    clip_site, clip_side, clip_j
                                                                ] = original_ids[
                                                                    work_i[
                                                                        lane,
                                                                        16 + clip_i,
                                                                        3 * clip_side + clip_j,
                                                                    ]
                                                                ]
                                                                barycentric[
                                                                    clip_site, clip_side, clip_j
                                                                ] = work[
                                                                    lane,
                                                                    16 + clip_i,
                                                                    5 + 3 * clip_side + clip_j,
                                                                ]
                                                        for clip_axis in tx.static_range(3):
                                                            surface[
                                                                clip_site, clip_side, clip_axis
                                                            ] = work[
                                                                lane,
                                                                16 + clip_i,
                                                                11 + 3 * clip_side + clip_axis,
                                                            ]
                                                            skin[
                                                                clip_site, clip_side, clip_axis
                                                            ] = 0.0
                                            status = 1
                                            dimension = 2
                                            count = clip_count

        meta[out, 0] = status
        meta[out, 1] = dimension
        meta[out, 2] = count
        meta[out, 3] = 0
        meta[out, 4] = 0
        if status == 1:
            meta[out, 4] = 1
        meta[out, 5] = 0
        meta[out, 6] = site_base
        meta[out, 7] = 0


def make_convex_patch_kernels():
    return {"clear": tx.kernel(clear_convex_patch), "fill": tx.kernel(fill_convex_patch)}
