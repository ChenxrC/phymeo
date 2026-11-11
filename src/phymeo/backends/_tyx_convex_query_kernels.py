"""Accepted-device pose gather and bounded bilateral convex distance kernels.

One lane owns one pair. At most 32 lanes share plan scratch at a time. Round
cores are finite segments, general convex cores use the same support scan, and
all material weights refer to actual original vertex IDs. No response is solved.
"""

import tyx as tx

from ._tyx_convex_support import clamp01, det3, dot3
from ._tyx_convex_support_family import (
    curved_support_x,
    curved_support_y,
    curved_support_z,
    support_ball_depth_width,
    support_ball_normal_bound,
)


def gather_convex_pose(
    count: tx.i32,
    selected_layout: tx.i32,
    body_extra: tx.i32,
    layouts: tx.ndarray(tx.i32, 1),
    tags: tx.ndarray(tx.i32, 2),
    rows: tx.ndarray(tx.i32, 1),
    groups: tx.ndarray(tx.i32, 1),
    offsets: tx.ndarray(tx.i32, 1),
    local: tx.ndarray(tx.f32, 2),
    tree_meta: tx.ndarray(tx.i32, 2),
    fk: tx.ndarray(tx.f32, 3),
    ticks: tx.ndarray(tx.i32, 1),
    pose: tx.ndarray(tx.f32, 2),
    stamp: tx.ndarray(tx.i32, 2),
    points: tx.ndarray(tx.f32, 2),
    asset_rotation: tx.ndarray(tx.f32, 3),
    shape_rotation: tx.ndarray(tx.f32, 3),
):
    for shape in tx.range(count):
        if layouts[shape] == selected_layout:
            row = rows[shape]
            group = groups[shape]
            slot = tags[shape, 3]
            stamp[shape, 0] = 1
            stamp[shape, 1] = ticks[slot]
            stamp[shape, 2] = slot
            for c in tx.static_range(12):
                pose[shape, c] = 0.0
            for a in tx.static_range(3):
                pose[shape, 3 + 4 * a] = 1.0
            if row >= 0:
                # Floating metadata counts child joints; FK includes root body 0.
                # Both affine sources already loan their expanded physical FK.
                if (
                    group < 0
                    or group >= tree_meta[row, 1] + body_extra
                    or slot != tree_meta[row, 0]
                ):
                    stamp[shape, 0] = 0
                else:
                    for c in tx.static_range(12):
                        pose[shape, c] = fk[row, group, c]
            for c in tx.static_range(12):
                if not tx.isfinite(pose[shape, c]):
                    stamp[shape, 0] = 0
            for axis in tx.static_range(3):
                for column in tx.static_range(3):
                    entry = 0.0
                    for inner in tx.static_range(3):
                        entry = tx.add_rn(
                            entry,
                            tx.mul_rn(
                                pose[shape, 3 + 3 * axis + inner],
                                asset_rotation[shape, inner, column],
                            ),
                        )
                    shape_rotation[shape, axis, column] = entry
                    if not tx.isfinite(entry):
                        stamp[shape, 0] = 0
            for i in range(offsets[shape], offsets[shape + 1]):
                for a in tx.static_range(3):
                    value = pose[shape, a]
                    for b in tx.static_range(3):
                        value = tx.add_rn(value, tx.mul_rn(pose[shape, 3 + a * 3 + b], local[i, b]))
                    points[i, a] = value
                    if not tx.isfinite(value):
                        stamp[shape, 0] = 0


def clear_convex_output(
    count: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    gap: tx.ndarray(tx.f32, 2),
    normal: tx.ndarray(tx.f32, 2),
    surface: tx.ndarray(tx.f32, 3),
    material_ids: tx.ndarray(tx.i32, 3),
    barycentric: tx.ndarray(tx.f32, 3),
    skin_offset: tx.ndarray(tx.f32, 3),
):
    for i in tx.range(count):
        for c in tx.static_range(16):
            meta[i, c] = 0
        for c in tx.static_range(4):
            gap[i, c] = 0.0
        for a in tx.static_range(3):
            normal[i, a] = 0.0
        for side in tx.static_range(2):
            for c in tx.static_range(4):
                material_ids[i, side, c] = -1
                barycentric[i, side, c] = 0.0
            for a in tx.static_range(3):
                surface[i, side, a] = 0.0
                skin_offset[i, side, a] = 0.0


def clear_convex_trace(
    count: tx.i32,
    history: tx.i32,
    trace_f: tx.ndarray(tx.f32, 3),
    trace_i: tx.ndarray(tx.i32, 3),
):
    for flat in tx.range(count):
        selected = flat // history
        iteration = flat % history
        for field in range(416):
            trace_f[selected, iteration, field] = 0.0
        for field in range(96):
            trace_i[selected, iteration, field] = 0


def query_convex_pairs(
    count: tx.i32,
    first_pair: tx.i32,
    attempt: tx.i32,
    gjk_budget: tx.i32,
    epa_budget: tx.i32,
    vertex_capacity: tx.i32,
    face_capacity: tx.i32,
    tags: tx.ndarray(tx.i32, 2),
    original_ids: tx.ndarray(tx.i32, 1),
    radii: tx.ndarray(tx.f32, 1),
    precision: tx.ndarray(tx.i32, 1),
    points: tx.ndarray(tx.f32, 2),
    dimensions: tx.ndarray(tx.f32, 2),
    shape_rotation: tx.ndarray(tx.f32, 3),
    support_points: tx.ndarray(tx.f32, 3),
    pairs: tx.ndarray(tx.i32, 2),
    vertices: tx.ndarray(tx.f32, 3),
    vertex_slots: tx.ndarray(tx.i32, 3),
    faces: tx.ndarray(tx.i32, 3),
    face_geometry: tx.ndarray(tx.f32, 3),
    edges: tx.ndarray(tx.i32, 3),
    weights: tx.ndarray(tx.f32, 2),
    indices: tx.ndarray(tx.i32, 2),
    simplex_copy: tx.ndarray(tx.f32, 3),
    simplex_slots: tx.ndarray(tx.i32, 3),
    meta: tx.ndarray(tx.i32, 2),
    gap: tx.ndarray(tx.f32, 2),
    normal: tx.ndarray(tx.f32, 2),
    surface: tx.ndarray(tx.f32, 3),
    material_ids: tx.ndarray(tx.i32, 3),
    barycentric: tx.ndarray(tx.f32, 3),
    skin_offset: tx.ndarray(tx.f32, 3),
    trace_count: tx.i32,
    trace_pairs: tx.ndarray(tx.i32, 2),
    trace_f: tx.ndarray(tx.f32, 3),
    trace_i: tx.ndarray(tx.i32, 3),
):
    for lane in tx.range(count):
        out = first_pair + lane
        a = pairs[out, 0]
        b = pairs[out, 1]
        trace_slot = -1
        for trace_candidate in range(trace_count):
            if trace_pairs[trace_candidate, 0] == a and trace_pairs[trace_candidate, 1] == b:
                trace_slot = trace_candidate
        a0 = tags[a, 1]
        a1 = tags[a, 2]
        b0 = tags[b, 1]
        b1 = tags[b, 2]
        ka = tags[a, 0]
        kb = tags[b, 0]
        full_surface = 0
        if ka >= 4 or kb >= 4:
            full_surface = 1
        ra = radii[a]
        rb = radii[b]
        scale = 1.0
        acx = 0.0
        acy = 0.0
        acz = 0.0
        bcx = 0.0
        bcy = 0.0
        bcz = 0.0
        diameter = 0.0
        for i in range(a0, a1):
            acx = acx + points[i, 0] / tx.cast_f32(a1 - a0)
            acy = acy + points[i, 1] / tx.cast_f32(a1 - a0)
            acz = acz + points[i, 2] / tx.cast_f32(a1 - a0)
            for j in range(a0, i):
                x = points[i, 0] - points[j, 0]
                y = points[i, 1] - points[j, 1]
                z = points[i, 2] - points[j, 2]
                diameter = tx.max(diameter, tx.sqrt(dot3(x, y, z, x, y, z)))
        if ka == 4 or ka == 5:
            diameter = 2.0 * (dimensions[a, 0] + dimensions[a, 1])
        elif ka == 6:
            diameter = 2.0 * tx.max(dimensions[a, 0], tx.max(dimensions[a, 1], dimensions[a, 2]))
        scale = tx.max(scale, diameter + 2.0 * ra)
        diameter = 0.0
        for i in range(b0, b1):
            bcx = bcx + points[i, 0] / tx.cast_f32(b1 - b0)
            bcy = bcy + points[i, 1] / tx.cast_f32(b1 - b0)
            bcz = bcz + points[i, 2] / tx.cast_f32(b1 - b0)
            for j in range(b0, i):
                x = points[i, 0] - points[j, 0]
                y = points[i, 1] - points[j, 1]
                z = points[i, 2] - points[j, 2]
                diameter = tx.max(diameter, tx.sqrt(dot3(x, y, z, x, y, z)))
        if kb == 4 or kb == 5:
            diameter = 2.0 * (dimensions[b, 0] + dimensions[b, 1])
        elif kb == 6:
            diameter = 2.0 * tx.max(dimensions[b, 0], tx.max(dimensions[b, 1], dimensions[b, 2]))
        scale = tx.max(scale, diameter + 2.0 * rb)
        dx = bcx - acx
        dy = bcy - acy
        dz = bcz - acz
        scale = tx.max(scale, tx.sqrt(dot3(dx, dy, dz, dx, dy, dz)))
        gap[out, 3] = scale
        curved_precision = 0
        for side in tx.static_range(2):
            shape = a
            if side == 1:
                shape = b
            if tags[shape, 0] >= 4:
                for axis in tx.static_range(3):
                    if (
                        dimensions[shape, axis] > 0.0
                        and dimensions[shape, axis] / scale < 1.0842021724855044e-19
                    ):
                        curved_precision = 1
        if full_surface != 0:
            # Full support already includes the round side's skin.
            ra = 0.0
            rb = 0.0
        rolling_radius = 0.0
        ball_peak_vertices = 0
        if full_surface != 0 and precision[a] == 0 and precision[b] == 0 and curved_precision == 0:
            for ball_side in tx.static_range(2):
                ball_shape = a
                if ball_side == 1:
                    ball_shape = b
                if tags[ball_shape, 0] < 2:
                    rolling_radius = rolling_radius + radii[ball_shape] / scale
                elif tags[ball_shape, 0] == 6:
                    # Gershgorin bound on the stored frame's Gram defect.
                    # No exact-orthogonality assumption or repaired frame.
                    frame_defect = 0.0
                    for gram_row in tx.static_range(3):
                        defect_row = 0.0
                        for gram_col in tx.static_range(3):
                            entry = 0.0
                            for gram_axis in tx.static_range(3):
                                entry = entry + (
                                    shape_rotation[ball_shape, gram_axis, gram_row]
                                    * shape_rotation[ball_shape, gram_axis, gram_col]
                                )
                            if gram_row == gram_col:
                                entry = entry - 1.0
                            defect_row = defect_row + tx.abs(entry)
                        frame_defect = tx.max(frame_defect, defect_row)
                    frame_defect = frame_defect + 3.814697265625e-6
                    if frame_defect < 0.25:
                        min_axis = (
                            tx.min(
                                dimensions[ball_shape, 0],
                                tx.min(dimensions[ball_shape, 1], dimensions[ball_shape, 2]),
                            )
                            / scale
                        )
                        max_axis = (
                            tx.max(
                                dimensions[ball_shape, 0],
                                tx.max(dimensions[ball_shape, 1], dimensions[ball_shape, 2]),
                            )
                            / scale
                        )
                        rolling_radius = rolling_radius + (
                            min_axis
                            * (min_axis / max_axis)
                            * (1.0 - frame_defect)
                            / tx.sqrt(1.0 + frame_defect)
                            * 0.9999990463256836
                        )

        meta[out, 12] = attempt
        meta[out, 13] = out
        status = 0
        ox = 0.0
        oy = 0.0
        oz = 0.0
        px = 0.0
        py = 0.0
        pz = 0.0
        relation = 0
        reason = 0
        phase = 0
        valid = 0
        gjk_used = 0
        epa_used = 0
        support_calls = 0
        vertex_count = 0
        face_count = 0
        active_count = 0
        signed_gap = 0.0
        lower_gap = 0.0
        upper_gap = 0.0
        nx = 0.0
        ny = 0.0
        nz = 0.0
        for k in tx.static_range(4):
            weights[lane, k] = 0.0
            indices[lane, k] = 0
        if precision[a] != 0 or precision[b] != 0 or curved_precision != 0:
            status = 3
            reason = 9
        elif ka == 3 or kb == 3 or ka < 0 or kb < 0 or ka > 6 or kb > 6:
            status = 3
            reason = 8
        elif ka < 2 and kb < 2:
            # The closest points of two finite segments; zero-length axes retain
            # both distinct material endpoints at parameter one half.
            phase = 1
            ax = points[a0, 0]
            ay = points[a0, 1]
            az = points[a0, 2]
            bx = points[b0, 0]
            by = points[b0, 1]
            bz = points[b0, 2]
            ux = points[a1 - 1, 0] - ax
            uy = points[a1 - 1, 1] - ay
            uz = points[a1 - 1, 2] - az
            vx = points[b1 - 1, 0] - bx
            vy = points[b1 - 1, 1] - by
            vz = points[b1 - 1, 2] - bz
            rx = ax - bx
            ry = ay - by
            rz = az - bz
            aa = dot3(ux, uy, uz, ux, uy, uz)
            ee = dot3(vx, vy, vz, vx, vy, vz)
            bb = dot3(ux, uy, uz, vx, vy, vz)
            cc = dot3(ux, uy, uz, rx, ry, rz)
            ff = dot3(vx, vy, vz, rx, ry, rz)
            sa = 0.5
            sb = 0.5
            if aa > 0.0 and ee > 0.0:
                denominator = aa * ee - bb * bb
                if denominator > 1e-7 * aa * ee:
                    sa = clamp01((bb * ff - cc * ee) / denominator)
                sb = (bb * sa + ff) / ee
                if sb < 0.0:
                    sb = 0.0
                    sa = clamp01(-cc / aa)
                elif sb > 1.0:
                    sb = 1.0
                    sa = clamp01((bb - cc) / aa)
            elif aa > 0.0:
                sa = clamp01(-cc / aa)
            elif ee > 0.0:
                sb = clamp01(ff / ee)
            pax = ax + sa * ux
            pay = ay + sa * uy
            paz = az + sa * uz
            pbx = bx + sb * vx
            pby = by + sb * vy
            pbz = bz + sb * vz
            dx = pbx - pax
            dy = pby - pay
            dz = pbz - paz
            distance = tx.sqrt(dot3(dx, dy, dz, dx, dy, dz))
            signed_gap = distance - ra - rb
            lower_gap = signed_gap
            upper_gap = signed_gap
            valid = 1
            if distance <= 1e-7 * scale:
                status = 2
                reason = 2
            else:
                status = 1
                valid = 7
                nx = dx / distance
                ny = dy / distance
                nz = dz / distance
                # Cartesian-product interpolation is a bilateral simplex with at
                # most four members and reconstructs each finite segment point.
                vertex_count = 4
                active_count = 4
                for k in tx.static_range(4):
                    ai = a0
                    bi = b0
                    wa = 1.0 - sa
                    wb = 1.0 - sb
                    if (k & 1) != 0:
                        ai = a1 - 1
                        wa = sa
                    if (k & 2) != 0:
                        bi = b1 - 1
                        wb = sb
                    vertex_slots[lane, k, 0] = ai
                    vertex_slots[lane, k, 1] = bi
                    weights[lane, k] = wa * wb
                    indices[lane, k] = k
                    for c in tx.static_range(3):
                        vertices[lane, k, 3 + c] = points[ai, c]
                        vertices[lane, k, 6 + c] = points[bi, c]
        else:
            phase = 3
            if full_surface != 0:
                phase = 5
            elif ka < 2 or kb < 2:
                phase = 2
            simplex_n = 0
            cx = 0.0
            cy = 0.0
            cz = 0.0
            closest_square = 0.0
            need_epa = 0
            if dot3(dx, dy, dz, dx, dy, dz) == 0.0:
                dx = 1.0
            # A common pair-local reference prevents large common translations
            # from entering the Minkowski subtraction or simplex conditioning.
            ox = points[a0, 0]
            oy = points[a0, 1]
            oz = points[a0, 2]
            for iteration in range(gjk_budget):
                if status == 0 and need_epa == 0:
                    if trace_slot >= 0:
                        trace_i[trace_slot, iteration, 0] = 1
                        trace_i[trace_slot, iteration, 1] = attempt
                        trace_i[trace_slot, iteration, 2] = out
                        trace_i[trace_slot, iteration, 3] = iteration
                        trace_i[trace_slot, iteration, 4] = simplex_n
                        trace_i[trace_slot, iteration, 6] = -1
                        trace_i[trace_slot, iteration, 22] = full_surface
                        trace_f[trace_slot, iteration, 0] = cx
                        trace_f[trace_slot, iteration, 1] = cy
                        trace_f[trace_slot, iteration, 2] = cz
                        trace_f[trace_slot, iteration, 3] = closest_square
                        trace_f[trace_slot, iteration, 8] = dx
                        trace_f[trace_slot, iteration, 9] = dy
                        trace_f[trace_slot, iteration, 10] = dz
                        trace_f[trace_slot, iteration, 30] = scale
                        trace_f[trace_slot, iteration, 412] = ox
                        trace_f[trace_slot, iteration, 413] = oy
                        trace_f[trace_slot, iteration, 414] = oz
                        trace_f[trace_slot, iteration, 415] = rolling_radius
                        for trace_vertex in range(simplex_n):
                            for trace_field in tx.static_range(9):
                                trace_f[trace_slot, iteration, 32+9*trace_vertex+trace_field] = vertices[lane, trace_vertex, trace_field]
                            trace_f[trace_slot, iteration, 68+trace_vertex] = weights[lane, trace_vertex]
                            trace_i[trace_slot, iteration, 32+trace_vertex] = indices[lane, trace_vertex]
                            for trace_side in tx.static_range(2):
                                trace_i[trace_slot, iteration, 24+2*trace_vertex+trace_side] = vertex_slots[lane, trace_vertex, trace_side]
                    ia = a0
                    ib = b0
                    best_a = dot3(
                        points[a0, 0] - ox, points[a0, 1] - oy, points[a0, 2] - oz, dx, dy, dz
                    )
                    best_b = dot3(
                        points[b0, 0] - ox, points[b0, 1] - oy, points[b0, 2] - oz, dx, dy, dz
                    )
                    for j in range(a0 + 1, a1):
                        value = dot3(
                            points[j, 0] - ox, points[j, 1] - oy, points[j, 2] - oz, dx, dy, dz
                        )
                        if value > best_a or (
                            value == best_a and original_ids[j] < original_ids[ia]
                        ):
                            best_a = value
                            ia = j
                    for j in range(b0 + 1, b1):
                        value = dot3(
                            points[j, 0] - ox, points[j, 1] - oy, points[j, 2] - oz, dx, dy, dz
                        )
                        if value < best_b or (
                            value == best_b and original_ids[j] < original_ids[ib]
                        ):
                            best_b = value
                            ib = j
                    # All three support sites share this full-surface adapter.
                    for support_side in tx.static_range(2):
                        support_shape = a
                        support_slot = ia
                        direction_sign = 1.0
                        if support_side == 1:
                            support_shape = b
                            support_slot = ib
                            direction_sign = -1.0
                        offset_x = 0.0
                        offset_y = 0.0
                        offset_z = 0.0
                        if full_surface != 0:
                            direction_scale = tx.max(tx.abs(dx), tx.max(tx.abs(dy), tx.abs(dz)))
                            if direction_scale == 0.0:
                                direction_scale = 1.0
                            direction_x = direction_sign * dx / direction_scale
                            direction_y = direction_sign * dy / direction_scale
                            direction_z = direction_sign * dz / direction_scale
                            if tags[support_shape, 0] >= 4:
                                local_x = dot3(
                                    shape_rotation[support_shape, 0, 0],
                                    shape_rotation[support_shape, 1, 0],
                                    shape_rotation[support_shape, 2, 0],
                                    direction_x,
                                    direction_y,
                                    direction_z,
                                )
                                local_y = dot3(
                                    shape_rotation[support_shape, 0, 1],
                                    shape_rotation[support_shape, 1, 1],
                                    shape_rotation[support_shape, 2, 1],
                                    direction_x,
                                    direction_y,
                                    direction_z,
                                )
                                local_z = dot3(
                                    shape_rotation[support_shape, 0, 2],
                                    shape_rotation[support_shape, 1, 2],
                                    shape_rotation[support_shape, 2, 2],
                                    direction_x,
                                    direction_y,
                                    direction_z,
                                )
                                if trace_slot >= 0:
                                    trace_f[trace_slot, iteration, 24+3*support_side] = local_x
                                    trace_f[trace_slot, iteration, 25+3*support_side] = local_y
                                    trace_f[trace_slot, iteration, 26+3*support_side] = local_z
                                dim_x = dimensions[support_shape, 0] / scale
                                dim_y = dimensions[support_shape, 1] / scale
                                dim_z = dimensions[support_shape, 2] / scale
                                point_x = curved_support_x(
                                    tags[support_shape, 0],
                                    local_x,
                                    local_y,
                                    local_z,
                                    dim_x,
                                    dim_y,
                                    dim_z,
                                )
                                point_y = curved_support_y(
                                    tags[support_shape, 0],
                                    local_x,
                                    local_y,
                                    local_z,
                                    dim_x,
                                    dim_y,
                                    dim_z,
                                )
                                point_z = curved_support_z(
                                    tags[support_shape, 0],
                                    local_x,
                                    local_y,
                                    local_z,
                                    dim_x,
                                    dim_y,
                                    dim_z,
                                )
                                offset_x = scale * dot3(
                                    shape_rotation[support_shape, 0, 0],
                                    shape_rotation[support_shape, 0, 1],
                                    shape_rotation[support_shape, 0, 2],
                                    point_x,
                                    point_y,
                                    point_z,
                                )
                                offset_y = scale * dot3(
                                    shape_rotation[support_shape, 1, 0],
                                    shape_rotation[support_shape, 1, 1],
                                    shape_rotation[support_shape, 1, 2],
                                    point_x,
                                    point_y,
                                    point_z,
                                )
                                offset_z = scale * dot3(
                                    shape_rotation[support_shape, 2, 0],
                                    shape_rotation[support_shape, 2, 1],
                                    shape_rotation[support_shape, 2, 2],
                                    point_x,
                                    point_y,
                                    point_z,
                                )
                            else:
                                direction_length = tx.sqrt(
                                    dot3(
                                        direction_x,
                                        direction_y,
                                        direction_z,
                                        direction_x,
                                        direction_y,
                                        direction_z,
                                    )
                                )
                                if direction_length > 0.0:
                                    offset_x = radii[support_shape] * direction_x / direction_length
                                    offset_y = radii[support_shape] * direction_y / direction_length
                                    offset_z = radii[support_shape] * direction_z / direction_length
                        for support_axis in tx.static_range(3):
                            support_points[lane, support_side, support_axis] = points[
                                support_slot, support_axis
                            ]
                        if full_surface != 0:
                            support_points[lane, support_side, 0] = (
                                support_points[lane, support_side, 0] - ox
                            ) + offset_x
                            support_points[lane, support_side, 1] = (
                                support_points[lane, support_side, 1] - oy
                            ) + offset_y
                            support_points[lane, support_side, 2] = (
                                support_points[lane, support_side, 2] - oz
                            ) + offset_z
                    support_calls = support_calls + 1
                    gjk_used = iteration + 1
                    if full_surface != 0:
                        px = (support_points[lane, 0, 0] - support_points[lane, 1, 0]) / scale
                        py = (support_points[lane, 0, 1] - support_points[lane, 1, 1]) / scale
                        pz = (support_points[lane, 0, 2] - support_points[lane, 1, 2]) / scale
                    else:
                        px = (
                            (support_points[lane, 0, 0] - ox) - (support_points[lane, 1, 0] - ox)
                        ) / scale
                        py = (
                            (support_points[lane, 0, 1] - oy) - (support_points[lane, 1, 1] - oy)
                        ) / scale
                        pz = (
                            (support_points[lane, 0, 2] - oz) - (support_points[lane, 1, 2] - oz)
                        ) / scale
                    bound = 0.0
                    if closest_square > 0.0:
                        bound = tx.max(0.0, dot3(cx, cy, cz, px, py, pz) / tx.sqrt(closest_square))
                    if trace_slot >= 0:
                        trace_f[trace_slot, iteration, 11] = bound
                        trace_f[trace_slot, iteration, 18] = px
                        trace_f[trace_slot, iteration, 19] = py
                        trace_f[trace_slot, iteration, 20] = pz
                        # New expression is trace-only, never a solver gate.
                        trace_f[trace_slot, iteration, 23] = dot3(cx, cy, cz, cx-px, cy-py, cz-pz)
                        trace_i[trace_slot, iteration, 16] = ia
                        trace_i[trace_slot, iteration, 17] = ib
                        trace_i[trace_slot, iteration, 18] = original_ids[ia]
                        trace_i[trace_slot, iteration, 19] = original_ids[ib]
                        for trace_side in tx.static_range(2):
                            for trace_axis in tx.static_range(3):
                                trace_f[trace_slot, iteration, 12+3*trace_side+trace_axis] = support_points[lane, trace_side, trace_axis]
                    # A global tangent-ball lower bound can certify a smooth
                    # penetration without tessellating its entire curved cap.
                    if full_surface != 0 and status == 0 and rolling_radius > 0.0:
                        ball_direction_scale = tx.max(tx.abs(dx), tx.max(tx.abs(dy), tx.abs(dz)))
                        if ball_direction_scale > 0.0:
                            ball_x = dx / ball_direction_scale
                            ball_y = dy / ball_direction_scale
                            ball_z = dz / ball_direction_scale
                            ball_length = tx.sqrt(
                                dot3(ball_x, ball_y, ball_z, ball_x, ball_y, ball_z)
                            )
                            ball_x = ball_x / ball_length
                            ball_y = ball_y / ball_length
                            ball_z = ball_z / ball_length
                            ball_reach = dot3(ball_x, ball_y, ball_z, px, py, pz)
                            ball_tx = px - ball_reach * ball_x
                            ball_ty = py - ball_reach * ball_y
                            ball_tz = pz - ball_reach * ball_z
                            ball_square = dot3(ball_tx, ball_ty, ball_tz, ball_tx, ball_ty, ball_tz)
                            ball_rounding = rolling_radius
                            for ball_support_side in tx.static_range(2):
                                ball_support_x = support_points[lane, ball_support_side, 0] / scale
                                ball_support_y = support_points[lane, ball_support_side, 1] / scale
                                ball_support_z = support_points[lane, ball_support_side, 2] / scale
                                ball_rounding = ball_rounding + tx.sqrt(
                                    dot3(
                                        ball_support_x,
                                        ball_support_y,
                                        ball_support_z,
                                        ball_support_x,
                                        ball_support_y,
                                        ball_support_z,
                                    )
                                )
                            ball_rounding = ball_rounding * 2.384185791015625e-7
                            ball_width = support_ball_depth_width(
                                ball_reach, ball_square, rolling_radius, ball_rounding
                            )
                            ball_normal_error = support_ball_normal_bound(
                                ball_reach, ball_square, rolling_radius, ball_rounding
                            )
                            if (
                                ball_reach + ball_rounding - ball_width > 0.0
                                and ball_width <= 8e-7
                                and ball_normal_error <= 1e-3
                                and tx.sqrt(ball_square) + 2.0 * ball_rounding <= 2e-4
                            ):
                                status = 1
                                valid = 7
                                phase = 6
                                nx = ball_x
                                ny = ball_y
                                nz = ball_z
                                signed_gap = -ball_reach * scale
                                lower_gap = -(ball_reach + ball_rounding) * scale
                                upper_gap = -(ball_reach + ball_rounding - ball_width) * scale
                                ball_peak_vertices = 1
                                if simplex_n > ball_peak_vertices:
                                    ball_peak_vertices = simplex_n
                                active_count = 1
                                simplex_n = 1
                                vertices[lane, 0, 0] = px
                                vertices[lane, 0, 1] = py
                                vertices[lane, 0, 2] = pz
                                vertex_slots[lane, 0, 0] = ia
                                vertex_slots[lane, 0, 1] = ib
                                for ball_axis in tx.static_range(3):
                                    vertices[lane, 0, 3 + ball_axis] = support_points[
                                        lane, 0, ball_axis
                                    ]
                                    vertices[lane, 0, 6 + ball_axis] = support_points[
                                        lane, 1, ball_axis
                                    ]
                                for ball_slot in tx.static_range(4):
                                    weights[lane, ball_slot] = 0.0
                                    indices[lane, ball_slot] = 0
                                weights[lane, 0] = 1.0
                                if trace_slot >= 0:
                                    trace_i[trace_slot, iteration, 9] = 7
                                    trace_i[trace_slot, iteration, 7] = 1

                    if status == 0:
                        duplicate = 0
                        for k in range(simplex_n):
                            ex = px - vertices[lane, k, 0]
                            ey = py - vertices[lane, k, 1]
                            ez = pz - vertices[lane, k, 2]
                            if dot3(ex, ey, ez, ex, ey, ez) <= 1e-16:
                                duplicate = 1
                        if trace_slot >= 0:
                            trace_i[trace_slot, iteration, 6] = duplicate
                        if simplex_n < 4 and duplicate == 0:
                            vertex_slots[lane, simplex_n, 0] = ia
                            vertex_slots[lane, simplex_n, 1] = ib
                            vertices[lane, simplex_n, 0] = px
                            vertices[lane, simplex_n, 1] = py
                            vertices[lane, simplex_n, 2] = pz
                            for c in tx.static_range(3):
                                vertices[lane, simplex_n, 3 + c] = support_points[lane, 0, c]
                                vertices[lane, simplex_n, 6 + c] = support_points[lane, 1, c]
                            simplex_n = simplex_n + 1
                            if trace_slot >= 0:
                                trace_i[trace_slot, iteration, 20] = 1
                        if trace_slot >= 0:
                            trace_i[trace_slot, iteration, 21] = simplex_n
                        # Enumerate the finite simplex's nonempty faces. Gram solves
                        # are at most 3x3; no approximate support point is fabricated.
                        best_square = 1e30
                        best_count = 0
                        for mask in range(1, 16):
                            # TYX has no shift operator. simplex_n is at most 4,
                            # so these powers of two are the same subset limit.
                            limit = 1
                            if simplex_n > 0:
                                limit = 2
                            if simplex_n > 1:
                                limit = 4
                            if simplex_n > 2:
                                limit = 8
                            if simplex_n > 3:
                                limit = 16
                            if mask < limit:
                                count_s = 0
                                i0 = 0
                                i1 = 0
                                i2 = 0
                                i3 = 0
                                for k in tx.static_range(4):
                                    bit = 1
                                    if k == 1:
                                        bit = 2
                                    elif k == 2:
                                        bit = 4
                                    elif k == 3:
                                        bit = 8
                                    if (mask & bit) != 0:
                                        if count_s == 0:
                                            i0 = k
                                        elif count_s == 1:
                                            i1 = k
                                        elif count_s == 2:
                                            i2 = k
                                        else:
                                            i3 = k
                                        count_s = count_s + 1
                                x0 = vertices[lane, i0, 0]
                                y0 = vertices[lane, i0, 1]
                                z0 = vertices[lane, i0, 2]
                                ex = vertices[lane, i1, 0] - x0
                                ey = vertices[lane, i1, 1] - y0
                                ez = vertices[lane, i1, 2] - z0
                                fx = vertices[lane, i2, 0] - x0
                                fy = vertices[lane, i2, 1] - y0
                                fz = vertices[lane, i2, 2] - z0
                                gx = vertices[lane, i3, 0] - x0
                                gy = vertices[lane, i3, 1] - y0
                                gz = vertices[lane, i3, 2] - z0
                                w0 = 1.0
                                w1 = 0.0
                                w2 = 0.0
                                w3 = 0.0
                                usable = 1
                                ee = dot3(ex, ey, ez, ex, ey, ez)
                                ef = dot3(ex, ey, ez, fx, fy, fz)
                                ff = dot3(fx, fy, fz, fx, fy, fz)
                                re = -dot3(x0, y0, z0, ex, ey, ez)
                                rf = -dot3(x0, y0, z0, fx, fy, fz)
                                if trace_slot >= 0:
                                    trace_f[trace_slot, iteration, 112+20*(mask-1)] = ee
                                    trace_f[trace_slot, iteration, 113+20*(mask-1)] = ef
                                    trace_f[trace_slot, iteration, 114+20*(mask-1)] = ff
                                    trace_f[trace_slot, iteration, 115+20*(mask-1)] = re
                                    trace_f[trace_slot, iteration, 116+20*(mask-1)] = rf
                                if count_s == 2:
                                    if ee <= 1e-20:
                                        usable = 0
                                    else:
                                        w1 = re / ee
                                        w0 = 1.0 - w1
                                elif count_s == 3:
                                    denominator = ee * ff - ef * ef
                                    if trace_slot >= 0:
                                        trace_f[trace_slot, iteration, 117+20*(mask-1)] = denominator
                                    if denominator <= 1e-12 * ee * ff:
                                        usable = 0
                                    else:
                                        w1 = (re * ff - rf * ef) / denominator
                                        w2 = (rf * ee - re * ef) / denominator
                                        w0 = 1.0 - w1 - w2
                                elif count_s == 4:
                                    determinant = det3(ex, ey, ez, fx, fy, fz, gx, gy, gz)
                                    if trace_slot >= 0:
                                        trace_f[trace_slot, iteration, 118+20*(mask-1)] = determinant
                                    if tx.abs(determinant) <= 1e-12:
                                        usable = 0
                                    else:
                                        w1 = (
                                            det3(-x0, -y0, -z0, fx, fy, fz, gx, gy, gz)
                                            / determinant
                                        )
                                        w2 = (
                                            det3(ex, ey, ez, -x0, -y0, -z0, gx, gy, gz)
                                            / determinant
                                        )
                                        w3 = (
                                            det3(ex, ey, ez, fx, fy, fz, -x0, -y0, -z0)
                                            / determinant
                                        )
                                        w0 = 1.0 - w1 - w2 - w3
                                if trace_slot >= 0:
                                    trace_f[trace_slot, iteration, 119+20*(mask-1)] = w0
                                    trace_f[trace_slot, iteration, 120+20*(mask-1)] = w1
                                    trace_f[trace_slot, iteration, 121+20*(mask-1)] = w2
                                    trace_f[trace_slot, iteration, 122+20*(mask-1)] = w3
                                if w0 < -1e-6 or w1 < -1e-6 or w2 < -1e-6 or w3 < -1e-6:
                                    usable = 0
                                if trace_slot >= 0:
                                    trace_i[trace_slot, iteration, 48+3*(mask-1)] = count_s
                                    trace_i[trace_slot, iteration, 49+3*(mask-1)] = usable
                                if usable != 0:
                                    w0 = tx.max(0.0, w0)
                                    w1 = tx.max(0.0, w1)
                                    w2 = tx.max(0.0, w2)
                                    w3 = tx.max(0.0, w3)
                                    total = w0 + w1 + w2 + w3
                                    w0 = w0 / total
                                    w1 = w1 / total
                                    w2 = w2 / total
                                    w3 = w3 / total
                                    qx = (
                                        w0 * x0
                                        + w1 * vertices[lane, i1, 0]
                                        + w2 * vertices[lane, i2, 0]
                                        + w3 * vertices[lane, i3, 0]
                                    )
                                    qy = (
                                        w0 * y0
                                        + w1 * vertices[lane, i1, 1]
                                        + w2 * vertices[lane, i2, 1]
                                        + w3 * vertices[lane, i3, 1]
                                    )
                                    qz = (
                                        w0 * z0
                                        + w1 * vertices[lane, i1, 2]
                                        + w2 * vertices[lane, i2, 2]
                                        + w3 * vertices[lane, i3, 2]
                                    )
                                    square = dot3(qx, qy, qz, qx, qy, qz)
                                    # A smooth support can improve the tangential
                                    # residual while the dominant normal square
                                    # rounds to the same FP32 value. Compare the
                                    # difference of squares without subtracting
                                    # two already-rounded squared norms on that tie.
                                    better = square < best_square
                                    if full_surface != 0 and square == best_square:
                                        better = (
                                            dot3(
                                                qx - cx,
                                                qy - cy,
                                                qz - cz,
                                                qx + cx,
                                                qy + cy,
                                                qz + cz,
                                            )
                                            < 0.0
                                        )
                                    if trace_slot >= 0:
                                        trace_f[trace_slot, iteration, 123+20*(mask-1)] = qx
                                        trace_f[trace_slot, iteration, 124+20*(mask-1)] = qy
                                        trace_f[trace_slot, iteration, 125+20*(mask-1)] = qz
                                        trace_f[trace_slot, iteration, 126+20*(mask-1)] = square
                                        # Diagnostic expression is not reused by the tie branch.
                                        trace_f[trace_slot, iteration, 127+20*(mask-1)] = dot3(qx-cx, qy-cy, qz-cz, qx+cx, qy+cy, qz+cz)
                                        trace_f[trace_slot, iteration, 128+20*(mask-1)] = best_square
                                        trace_f[trace_slot, iteration, 129+20*(mask-1)] = cx
                                        trace_f[trace_slot, iteration, 130+20*(mask-1)] = cy
                                        trace_f[trace_slot, iteration, 131+20*(mask-1)] = cz
                                        if better:
                                            trace_i[trace_slot, iteration, 50+3*(mask-1)] = 1
                                        if full_surface != 0 and square == best_square:
                                            trace_i[trace_slot, iteration, 50+3*(mask-1)] = trace_i[trace_slot, iteration, 50+3*(mask-1)] + 2
                                    if better:
                                        best_square = square
                                        best_count = count_s
                                        if trace_slot >= 0:
                                            trace_i[trace_slot, iteration, 7] = count_s
                                            trace_i[trace_slot, iteration, 8] = mask
                                        cx = qx
                                        cy = qy
                                        cz = qz
                                        indices[lane, 0] = i0
                                        indices[lane, 1] = i1
                                        indices[lane, 2] = i2
                                        indices[lane, 3] = i3
                                        weights[lane, 0] = w0
                                        weights[lane, 1] = w1
                                        weights[lane, 2] = w2
                                        weights[lane, 3] = w3
                        if best_count == 0:
                            status = 2
                            reason = 6
                            if trace_slot >= 0:
                                trace_i[trace_slot, iteration, 9] = 6
                        else:
                            closest_square = best_square
                            distance = tx.sqrt(best_square)
                            if trace_slot >= 0:
                                trace_f[trace_slot, iteration, 21] = distance
                                trace_f[trace_slot, iteration, 22] = distance-bound
                            if distance <= 8e-7 or (
                                distance <= 1e-5 and ((ka >= 2 and kb >= 2) or full_surface != 0)
                            ):
                                # A tilted shallow overlap can stop just outside the
                                # origin. Polytopes within 1e-5 go to EPA instead of
                                # publishing that residual as separation. A round
                                # core inside that same plane tolerance is contained:
                                # the leftover is not a surface gap.
                                if full_surface == 0 and (ka < 2 or kb < 2):
                                    status = 3
                                    reason = 3
                                    if trace_slot >= 0:
                                        trace_i[trace_slot, iteration, 9] = 5
                                else:
                                    need_epa = 1
                                    if trace_slot >= 0:
                                        trace_i[trace_slot, iteration, 9] = 4
                            elif distance - bound <= 8e-7 or duplicate != 0:
                                # A repeated support is already a simplex vertex, so
                                # another iteration cannot add one. Float32 can miss
                                # the plane certificate even when this closest point
                                # is already the separating distance.
                                status = 1
                                valid = 7
                                nx = -cx / distance
                                ny = -cy / distance
                                nz = -cz / distance
                                signed_gap = distance * scale - ra - rb
                                lower_gap = tx.min(signed_gap, bound * scale - ra - rb)
                                upper_gap = signed_gap
                                active_count = best_count
                                if trace_slot >= 0:
                                    trace_i[trace_slot, iteration, 9] = 3
                                    if distance-bound <= 8e-7:
                                        trace_i[trace_slot, iteration, 9] = 2
                            # Compact only after retaining the complete closest face.
                            for k in tx.static_range(4):
                                selected = indices[lane, k]
                                for c in tx.static_range(9):
                                    simplex_copy[lane, k, c] = vertices[lane, selected, c]
                                for side in tx.static_range(2):
                                    simplex_slots[lane, k, side] = vertex_slots[
                                        lane, selected, side
                                    ]
                            for k in tx.static_range(4):
                                for c in tx.static_range(9):
                                    vertices[lane, k, c] = simplex_copy[lane, k, c]
                                for side in tx.static_range(2):
                                    vertex_slots[lane, k, side] = simplex_slots[lane, k, side]
                                indices[lane, k] = k
                            simplex_n = best_count
                            dx = -cx
                            dy = -cy
                            dz = -cz
                    if trace_slot >= 0:
                        trace_i[trace_slot, iteration, 5] = simplex_n
                        if trace_i[trace_slot, iteration, 9] == 0:
                            trace_i[trace_slot, iteration, 9] = 1
                        trace_i[trace_slot, iteration, 10] = status
                        trace_i[trace_slot, iteration, 11] = reason
                        trace_i[trace_slot, iteration, 12] = phase
                        trace_i[trace_slot, iteration, 13] = need_epa
                        trace_i[trace_slot, iteration, 14] = gjk_used
                        trace_i[trace_slot, iteration, 15] = support_calls
                        trace_f[trace_slot, iteration, 4] = cx
                        trace_f[trace_slot, iteration, 5] = cy
                        trace_f[trace_slot, iteration, 6] = cz
                        trace_f[trace_slot, iteration, 7] = closest_square
                        for trace_vertex in range(simplex_n):
                            for trace_field in tx.static_range(9):
                                trace_f[trace_slot, iteration, 72+9*trace_vertex+trace_field] = vertices[lane, trace_vertex, trace_field]
                            trace_f[trace_slot, iteration, 108+trace_vertex] = weights[lane, trace_vertex]
                            trace_i[trace_slot, iteration, 44+trace_vertex] = indices[lane, trace_vertex]
                            for trace_side in tx.static_range(2):
                                trace_i[trace_slot, iteration, 36+2*trace_vertex+trace_side] = vertex_slots[lane, trace_vertex, trace_side]
            vertex_count = simplex_n
            if ball_peak_vertices > vertex_count:
                vertex_count = ball_peak_vertices
            if status == 0 and need_epa == 0:
                status = 4
                reason = 4
                if trace_slot >= 0 and gjk_used > 0:
                    trace_i[trace_slot, gjk_used-1, 9] = 8
                    trace_i[trace_slot, gjk_used-1, 10] = status
                    trace_i[trace_slot, gjk_used-1, 11] = reason
            if need_epa != 0 and status == 0:
                phase = 4
                # A lower-dimensional intersection simplex is not an EPA
                # polytope. Add a fixed, shape-independent set of directions,
                # retaining real bilateral support vertices and IDs only.
                for seed in range(14):
                    if status == 0:
                        dx = 0.0
                        dy = 0.0
                        dz = 0.0
                        if seed < 6:
                            sign = 1.0
                            if (seed & 1) != 0:
                                sign = -1.0
                            if seed < 2:
                                dx = sign
                            elif seed < 4:
                                dy = sign
                            else:
                                dz = sign
                        else:
                            code = seed - 6
                            dx = 1.0
                            dy = 1.0
                            dz = 1.0
                            if (code & 1) != 0:
                                dx = -1.0
                            if (code & 2) != 0:
                                dy = -1.0
                            if (code & 4) != 0:
                                dz = -1.0
                        ia = a0
                        ib = b0
                        best_a = dot3(
                            points[a0, 0] - ox, points[a0, 1] - oy, points[a0, 2] - oz, dx, dy, dz
                        )
                        best_b = dot3(
                            points[b0, 0] - ox, points[b0, 1] - oy, points[b0, 2] - oz, dx, dy, dz
                        )
                        for j in range(a0 + 1, a1):
                            value = dot3(
                                points[j, 0] - ox, points[j, 1] - oy, points[j, 2] - oz, dx, dy, dz
                            )
                            if value > best_a or (
                                value == best_a and original_ids[j] < original_ids[ia]
                            ):
                                best_a = value
                                ia = j
                        for j in range(b0 + 1, b1):
                            value = dot3(
                                points[j, 0] - ox, points[j, 1] - oy, points[j, 2] - oz, dx, dy, dz
                            )
                            if value < best_b or (
                                value == best_b and original_ids[j] < original_ids[ib]
                            ):
                                best_b = value
                                ib = j
                        # All three support sites share this full-surface adapter.
                        for support_side in tx.static_range(2):
                            support_shape = a
                            support_slot = ia
                            direction_sign = 1.0
                            if support_side == 1:
                                support_shape = b
                                support_slot = ib
                                direction_sign = -1.0
                            offset_x = 0.0
                            offset_y = 0.0
                            offset_z = 0.0
                            if full_surface != 0:
                                direction_scale = tx.max(tx.abs(dx), tx.max(tx.abs(dy), tx.abs(dz)))
                                if direction_scale == 0.0:
                                    direction_scale = 1.0
                                direction_x = direction_sign * dx / direction_scale
                                direction_y = direction_sign * dy / direction_scale
                                direction_z = direction_sign * dz / direction_scale
                                if tags[support_shape, 0] >= 4:
                                    local_x = dot3(
                                        shape_rotation[support_shape, 0, 0],
                                        shape_rotation[support_shape, 1, 0],
                                        shape_rotation[support_shape, 2, 0],
                                        direction_x,
                                        direction_y,
                                        direction_z,
                                    )
                                    local_y = dot3(
                                        shape_rotation[support_shape, 0, 1],
                                        shape_rotation[support_shape, 1, 1],
                                        shape_rotation[support_shape, 2, 1],
                                        direction_x,
                                        direction_y,
                                        direction_z,
                                    )
                                    local_z = dot3(
                                        shape_rotation[support_shape, 0, 2],
                                        shape_rotation[support_shape, 1, 2],
                                        shape_rotation[support_shape, 2, 2],
                                        direction_x,
                                        direction_y,
                                        direction_z,
                                    )
                                    dim_x = dimensions[support_shape, 0] / scale
                                    dim_y = dimensions[support_shape, 1] / scale
                                    dim_z = dimensions[support_shape, 2] / scale
                                    point_x = curved_support_x(
                                        tags[support_shape, 0],
                                        local_x,
                                        local_y,
                                        local_z,
                                        dim_x,
                                        dim_y,
                                        dim_z,
                                    )
                                    point_y = curved_support_y(
                                        tags[support_shape, 0],
                                        local_x,
                                        local_y,
                                        local_z,
                                        dim_x,
                                        dim_y,
                                        dim_z,
                                    )
                                    point_z = curved_support_z(
                                        tags[support_shape, 0],
                                        local_x,
                                        local_y,
                                        local_z,
                                        dim_x,
                                        dim_y,
                                        dim_z,
                                    )
                                    offset_x = scale * dot3(
                                        shape_rotation[support_shape, 0, 0],
                                        shape_rotation[support_shape, 0, 1],
                                        shape_rotation[support_shape, 0, 2],
                                        point_x,
                                        point_y,
                                        point_z,
                                    )
                                    offset_y = scale * dot3(
                                        shape_rotation[support_shape, 1, 0],
                                        shape_rotation[support_shape, 1, 1],
                                        shape_rotation[support_shape, 1, 2],
                                        point_x,
                                        point_y,
                                        point_z,
                                    )
                                    offset_z = scale * dot3(
                                        shape_rotation[support_shape, 2, 0],
                                        shape_rotation[support_shape, 2, 1],
                                        shape_rotation[support_shape, 2, 2],
                                        point_x,
                                        point_y,
                                        point_z,
                                    )
                                else:
                                    direction_length = tx.sqrt(
                                        dot3(
                                            direction_x,
                                            direction_y,
                                            direction_z,
                                            direction_x,
                                            direction_y,
                                            direction_z,
                                        )
                                    )
                                    if direction_length > 0.0:
                                        offset_x = (
                                            radii[support_shape] * direction_x / direction_length
                                        )
                                        offset_y = (
                                            radii[support_shape] * direction_y / direction_length
                                        )
                                        offset_z = (
                                            radii[support_shape] * direction_z / direction_length
                                        )
                            for support_axis in tx.static_range(3):
                                support_points[lane, support_side, support_axis] = points[
                                    support_slot, support_axis
                                ]
                            if full_surface != 0:
                                support_points[lane, support_side, 0] = (
                                    support_points[lane, support_side, 0] - ox
                                ) + offset_x
                                support_points[lane, support_side, 1] = (
                                    support_points[lane, support_side, 1] - oy
                                ) + offset_y
                                support_points[lane, support_side, 2] = (
                                    support_points[lane, support_side, 2] - oz
                                ) + offset_z
                        support_calls = support_calls + 1
                        if full_surface != 0:
                            px = (support_points[lane, 0, 0] - support_points[lane, 1, 0]) / scale
                            py = (support_points[lane, 0, 1] - support_points[lane, 1, 1]) / scale
                            pz = (support_points[lane, 0, 2] - support_points[lane, 1, 2]) / scale
                        else:
                            px = (
                                (support_points[lane, 0, 0] - ox)
                                - (support_points[lane, 1, 0] - ox)
                            ) / scale
                            py = (
                                (support_points[lane, 0, 1] - oy)
                                - (support_points[lane, 1, 1] - oy)
                            ) / scale
                            pz = (
                                (support_points[lane, 0, 2] - oz)
                                - (support_points[lane, 1, 2] - oz)
                            ) / scale
                        duplicate = 0
                        for k in range(vertex_count):
                            ex = px - vertices[lane, k, 0]
                            ey = py - vertices[lane, k, 1]
                            ez = pz - vertices[lane, k, 2]
                            if dot3(ex, ey, ez, ex, ey, ez) <= 1e-16:
                                duplicate = 1
                        if duplicate == 0:
                            if vertex_count == vertex_capacity:
                                status = 4
                                reason = 7
                            else:
                                vertex_slots[lane, vertex_count, 0] = ia
                                vertex_slots[lane, vertex_count, 1] = ib
                                vertices[lane, vertex_count, 0] = px
                                vertices[lane, vertex_count, 1] = py
                                vertices[lane, vertex_count, 2] = pz
                                for c in tx.static_range(3):
                                    vertices[lane, vertex_count, 3 + c] = support_points[lane, 0, c]
                                    vertices[lane, vertex_count, 6 + c] = support_points[lane, 1, c]
                                vertex_count = vertex_count + 1
                found = 0
                # Boundary-containing tetrahedra are useful for a true contact
                # band. EPA expands their zero-height faces with actual support.
                for i0 in range(vertex_count):
                    for i1 in range(i0 + 1, vertex_count):
                        for i2 in range(i1 + 1, vertex_count):
                            for i3 in range(i2 + 1, vertex_count):
                                if found == 0 and status == 0:
                                    x0 = vertices[lane, i0, 0]
                                    y0 = vertices[lane, i0, 1]
                                    z0 = vertices[lane, i0, 2]
                                    ex = vertices[lane, i1, 0] - x0
                                    ey = vertices[lane, i1, 1] - y0
                                    ez = vertices[lane, i1, 2] - z0
                                    fx = vertices[lane, i2, 0] - x0
                                    fy = vertices[lane, i2, 1] - y0
                                    fz = vertices[lane, i2, 2] - z0
                                    gx = vertices[lane, i3, 0] - x0
                                    gy = vertices[lane, i3, 1] - y0
                                    gz = vertices[lane, i3, 2] - z0
                                    determinant = det3(ex, ey, ez, fx, fy, fz, gx, gy, gz)
                                    if tx.abs(determinant) > 1e-12:
                                        w1 = (
                                            det3(-x0, -y0, -z0, fx, fy, fz, gx, gy, gz)
                                            / determinant
                                        )
                                        w2 = (
                                            det3(ex, ey, ez, -x0, -y0, -z0, gx, gy, gz)
                                            / determinant
                                        )
                                        w3 = (
                                            det3(ex, ey, ez, fx, fy, fz, -x0, -y0, -z0)
                                            / determinant
                                        )
                                        w0 = 1.0 - w1 - w2 - w3
                                        if (
                                            w0 >= -1e-6
                                            and w1 >= -1e-6
                                            and w2 >= -1e-6
                                            and w3 >= -1e-6
                                        ):
                                            indices[lane, 0] = i0
                                            indices[lane, 1] = i1
                                            indices[lane, 2] = i2
                                            indices[lane, 3] = i3
                                            found = 1
                ix = 0.0
                iy = 0.0
                iz = 0.0
                if status == 0 and found == 0:
                    status = 2
                    reason = 6
                if status == 0:
                    for k in tx.static_range(4):
                        selected = indices[lane, k]
                        ix = ix + 0.25 * vertices[lane, selected, 0]
                        iy = iy + 0.25 * vertices[lane, selected, 1]
                        iz = iz + 0.25 * vertices[lane, selected, 2]
                        for c in tx.static_range(9):
                            simplex_copy[lane, k, c] = vertices[lane, selected, c]
                        for side in tx.static_range(2):
                            simplex_slots[lane, k, side] = vertex_slots[lane, selected, side]
                    for k in tx.static_range(4):
                        for c in tx.static_range(9):
                            vertices[lane, k, c] = simplex_copy[lane, k, c]
                        for side in tx.static_range(2):
                            vertex_slots[lane, k, side] = simplex_slots[lane, k, side]
                    vertex_count = 4
                    for face in range(face_capacity):
                        faces[lane, face, 0] = -1
                        faces[lane, face, 1] = -1
                        faces[lane, face, 2] = -1
                    for face in tx.static_range(4):
                        c0 = (face + 1) % 4
                        c1 = (face + 2) % 4
                        c2 = (face + 3) % 4
                        x0 = vertices[lane, c0, 0]
                        y0 = vertices[lane, c0, 1]
                        z0 = vertices[lane, c0, 2]
                        ex = vertices[lane, c1, 0] - x0
                        ey = vertices[lane, c1, 1] - y0
                        ez = vertices[lane, c1, 2] - z0
                        fx = vertices[lane, c2, 0] - x0
                        fy = vertices[lane, c2, 1] - y0
                        fz = vertices[lane, c2, 2] - z0
                        dx = ey * fz - ez * fy
                        dy = ez * fx - ex * fz
                        dz = ex * fy - ey * fx
                        length = tx.sqrt(dot3(dx, dy, dz, dx, dy, dz))
                        if length <= 1e-12:
                            status = 2
                            reason = 6
                        else:
                            dx = dx / length
                            dy = dy / length
                            dz = dz / length
                            if dot3(dx, dy, dz, x0 - ix, y0 - iy, z0 - iz) < 0.0:
                                swap = c1
                                c1 = c2
                                c2 = swap
                                dx = -dx
                                dy = -dy
                                dz = -dz
                            faces[lane, face, 0] = c0
                            faces[lane, face, 1] = c1
                            faces[lane, face, 2] = c2
                            face_geometry[lane, face, 0] = dx
                            face_geometry[lane, face, 1] = dy
                            face_geometry[lane, face, 2] = dz
                            face_geometry[lane, face, 3] = dot3(dx, dy, dz, x0, y0, z0)
                    face_count = 4
                for iteration in range(epa_budget):
                    if status == 0:
                        epa_used = iteration + 1
                        closest_face = -1
                        depth = 1e30
                        for face in range(face_capacity):
                            # A backward plane can be slightly negative. Clamping
                            # it to zero makes it win over the real nearest face.
                            face_depth = face_geometry[lane, face, 3]
                            if (
                                faces[lane, face, 0] >= 0
                                and face_depth > -1e-4
                                and face_depth < depth
                            ):
                                closest_face = face
                                depth = face_depth
                        if closest_face < 0:
                            status = 2
                            reason = 6
                        else:
                            dx = face_geometry[lane, closest_face, 0]
                            dy = face_geometry[lane, closest_face, 1]
                            dz = face_geometry[lane, closest_face, 2]
                            ia = a0
                            ib = b0
                            best_a = dot3(
                                points[a0, 0] - ox,
                                points[a0, 1] - oy,
                                points[a0, 2] - oz,
                                dx,
                                dy,
                                dz,
                            )
                            best_b = dot3(
                                points[b0, 0] - ox,
                                points[b0, 1] - oy,
                                points[b0, 2] - oz,
                                dx,
                                dy,
                                dz,
                            )
                            for j in range(a0 + 1, a1):
                                value = dot3(
                                    points[j, 0] - ox,
                                    points[j, 1] - oy,
                                    points[j, 2] - oz,
                                    dx,
                                    dy,
                                    dz,
                                )
                                if value > best_a or (
                                    value == best_a and original_ids[j] < original_ids[ia]
                                ):
                                    best_a = value
                                    ia = j
                            for j in range(b0 + 1, b1):
                                value = dot3(
                                    points[j, 0] - ox,
                                    points[j, 1] - oy,
                                    points[j, 2] - oz,
                                    dx,
                                    dy,
                                    dz,
                                )
                                if value < best_b or (
                                    value == best_b and original_ids[j] < original_ids[ib]
                                ):
                                    best_b = value
                                    ib = j
                            # All three support sites share this full-surface adapter.
                            for support_side in tx.static_range(2):
                                support_shape = a
                                support_slot = ia
                                direction_sign = 1.0
                                if support_side == 1:
                                    support_shape = b
                                    support_slot = ib
                                    direction_sign = -1.0
                                offset_x = 0.0
                                offset_y = 0.0
                                offset_z = 0.0
                                if full_surface != 0:
                                    direction_scale = tx.max(
                                        tx.abs(dx), tx.max(tx.abs(dy), tx.abs(dz))
                                    )
                                    if direction_scale == 0.0:
                                        direction_scale = 1.0
                                    direction_x = direction_sign * dx / direction_scale
                                    direction_y = direction_sign * dy / direction_scale
                                    direction_z = direction_sign * dz / direction_scale
                                    if tags[support_shape, 0] >= 4:
                                        local_x = dot3(
                                            shape_rotation[support_shape, 0, 0],
                                            shape_rotation[support_shape, 1, 0],
                                            shape_rotation[support_shape, 2, 0],
                                            direction_x,
                                            direction_y,
                                            direction_z,
                                        )
                                        local_y = dot3(
                                            shape_rotation[support_shape, 0, 1],
                                            shape_rotation[support_shape, 1, 1],
                                            shape_rotation[support_shape, 2, 1],
                                            direction_x,
                                            direction_y,
                                            direction_z,
                                        )
                                        local_z = dot3(
                                            shape_rotation[support_shape, 0, 2],
                                            shape_rotation[support_shape, 1, 2],
                                            shape_rotation[support_shape, 2, 2],
                                            direction_x,
                                            direction_y,
                                            direction_z,
                                        )
                                        dim_x = dimensions[support_shape, 0] / scale
                                        dim_y = dimensions[support_shape, 1] / scale
                                        dim_z = dimensions[support_shape, 2] / scale
                                        point_x = curved_support_x(
                                            tags[support_shape, 0],
                                            local_x,
                                            local_y,
                                            local_z,
                                            dim_x,
                                            dim_y,
                                            dim_z,
                                        )
                                        point_y = curved_support_y(
                                            tags[support_shape, 0],
                                            local_x,
                                            local_y,
                                            local_z,
                                            dim_x,
                                            dim_y,
                                            dim_z,
                                        )
                                        point_z = curved_support_z(
                                            tags[support_shape, 0],
                                            local_x,
                                            local_y,
                                            local_z,
                                            dim_x,
                                            dim_y,
                                            dim_z,
                                        )
                                        offset_x = scale * dot3(
                                            shape_rotation[support_shape, 0, 0],
                                            shape_rotation[support_shape, 0, 1],
                                            shape_rotation[support_shape, 0, 2],
                                            point_x,
                                            point_y,
                                            point_z,
                                        )
                                        offset_y = scale * dot3(
                                            shape_rotation[support_shape, 1, 0],
                                            shape_rotation[support_shape, 1, 1],
                                            shape_rotation[support_shape, 1, 2],
                                            point_x,
                                            point_y,
                                            point_z,
                                        )
                                        offset_z = scale * dot3(
                                            shape_rotation[support_shape, 2, 0],
                                            shape_rotation[support_shape, 2, 1],
                                            shape_rotation[support_shape, 2, 2],
                                            point_x,
                                            point_y,
                                            point_z,
                                        )
                                    else:
                                        direction_length = tx.sqrt(
                                            dot3(
                                                direction_x,
                                                direction_y,
                                                direction_z,
                                                direction_x,
                                                direction_y,
                                                direction_z,
                                            )
                                        )
                                        if direction_length > 0.0:
                                            offset_x = (
                                                radii[support_shape]
                                                * direction_x
                                                / direction_length
                                            )
                                            offset_y = (
                                                radii[support_shape]
                                                * direction_y
                                                / direction_length
                                            )
                                            offset_z = (
                                                radii[support_shape]
                                                * direction_z
                                                / direction_length
                                            )
                                for support_axis in tx.static_range(3):
                                    support_points[lane, support_side, support_axis] = points[
                                        support_slot, support_axis
                                    ]
                                if full_surface != 0:
                                    support_points[lane, support_side, 0] = (
                                        support_points[lane, support_side, 0] - ox
                                    ) + offset_x
                                    support_points[lane, support_side, 1] = (
                                        support_points[lane, support_side, 1] - oy
                                    ) + offset_y
                                    support_points[lane, support_side, 2] = (
                                        support_points[lane, support_side, 2] - oz
                                    ) + offset_z
                            support_calls = support_calls + 1
                            if full_surface != 0:
                                px = (
                                    support_points[lane, 0, 0] - support_points[lane, 1, 0]
                                ) / scale
                                py = (
                                    support_points[lane, 0, 1] - support_points[lane, 1, 1]
                                ) / scale
                                pz = (
                                    support_points[lane, 0, 2] - support_points[lane, 1, 2]
                                ) / scale
                            else:
                                px = (
                                    (support_points[lane, 0, 0] - ox)
                                    - (support_points[lane, 1, 0] - ox)
                                ) / scale
                                py = (
                                    (support_points[lane, 0, 1] - oy)
                                    - (support_points[lane, 1, 1] - oy)
                                ) / scale
                                pz = (
                                    (support_points[lane, 0, 2] - oz)
                                    - (support_points[lane, 1, 2] - oz)
                                ) / scale
                            reach = dot3(dx, dy, dz, px, py, pz)
                            duplicate = 0
                            for k in range(vertex_count):
                                ex = px - vertices[lane, k, 0]
                                ey = py - vertices[lane, k, 1]
                                ez = pz - vertices[lane, k, 2]
                                if dot3(ex, ey, ez, ex, ey, ez) <= 1e-16:
                                    duplicate = 1
                            ball_accepted = 0
                            if full_surface != 0 and rolling_radius > 0.0:
                                ball_length = tx.sqrt(dot3(dx, dy, dz, dx, dy, dz))
                                if ball_length > 0.0:
                                    ball_x = dx / ball_length
                                    ball_y = dy / ball_length
                                    ball_z = dz / ball_length
                                    ball_reach = dot3(ball_x, ball_y, ball_z, px, py, pz)
                                    ball_tx = px - ball_reach * ball_x
                                    ball_ty = py - ball_reach * ball_y
                                    ball_tz = pz - ball_reach * ball_z
                                    ball_square = dot3(
                                        ball_tx, ball_ty, ball_tz, ball_tx, ball_ty, ball_tz
                                    )
                                    ball_rounding = rolling_radius
                                    for ball_support_side in tx.static_range(2):
                                        ball_support_x = (
                                            support_points[lane, ball_support_side, 0] / scale
                                        )
                                        ball_support_y = (
                                            support_points[lane, ball_support_side, 1] / scale
                                        )
                                        ball_support_z = (
                                            support_points[lane, ball_support_side, 2] / scale
                                        )
                                        ball_rounding = ball_rounding + tx.sqrt(
                                            dot3(
                                                ball_support_x,
                                                ball_support_y,
                                                ball_support_z,
                                                ball_support_x,
                                                ball_support_y,
                                                ball_support_z,
                                            )
                                        )
                                    ball_rounding = ball_rounding * 2.384185791015625e-7
                                    ball_width = support_ball_depth_width(
                                        ball_reach, ball_square, rolling_radius, ball_rounding
                                    )
                                    ball_normal_error = support_ball_normal_bound(
                                        ball_reach, ball_square, rolling_radius, ball_rounding
                                    )
                                    if (
                                        ball_reach + ball_rounding - ball_width > 0.0
                                        and ball_width <= 8e-7
                                        and ball_normal_error <= 1e-3
                                        and tx.sqrt(ball_square) + 2.0 * ball_rounding <= 2e-4
                                    ):
                                        ball_accepted = 1
                                        status = 1
                                        valid = 7
                                        phase = 6
                                        nx = ball_x
                                        ny = ball_y
                                        nz = ball_z
                                        signed_gap = -ball_reach * scale
                                        lower_gap = -(ball_reach + ball_rounding) * scale
                                        upper_gap = (
                                            -(ball_reach + ball_rounding - ball_width) * scale
                                        )
                                        active_count = 1
                                        # Reuse one owned slot; retain peak EPA
                                        # vertex/face counts already consumed.
                                        vertices[lane, 0, 0] = px
                                        vertices[lane, 0, 1] = py
                                        vertices[lane, 0, 2] = pz
                                        vertex_slots[lane, 0, 0] = ia
                                        vertex_slots[lane, 0, 1] = ib
                                        for ball_axis in tx.static_range(3):
                                            vertices[lane, 0, 3 + ball_axis] = support_points[
                                                lane, 0, ball_axis
                                            ]
                                            vertices[lane, 0, 6 + ball_axis] = support_points[
                                                lane, 1, ball_axis
                                            ]
                                        for ball_slot in tx.static_range(4):
                                            weights[lane, ball_slot] = 0.0
                                            indices[lane, ball_slot] = 0
                                        weights[lane, 0] = 1.0

                            if ball_accepted == 0 and reach - depth <= 8e-7:
                                # A polygonal face may be triangulated. Locate
                                # the actual coplanar triangle owning its foot,
                                # not arbitrary barycentrics on the first facet.
                                chosen = 0
                                multiple = 0
                                for face in range(face_capacity):
                                    if faces[lane, face, 0] >= 0:
                                        cosine = dot3(
                                            dx,
                                            dy,
                                            dz,
                                            face_geometry[lane, face, 0],
                                            face_geometry[lane, face, 1],
                                            face_geometry[lane, face, 2],
                                        )
                                        if (
                                            tx.abs(face_geometry[lane, face, 3] - depth) <= 4e-7
                                            and cosine < 0.9999
                                        ):
                                            multiple = 1
                                        if (
                                            chosen == 0
                                            and cosine > 0.9999
                                            and tx.abs(face_geometry[lane, face, 3] - depth) <= 2e-6
                                        ):
                                            i0 = faces[lane, face, 0]
                                            i1 = faces[lane, face, 1]
                                            i2 = faces[lane, face, 2]
                                            x0 = vertices[lane, i0, 0]
                                            y0 = vertices[lane, i0, 1]
                                            z0 = vertices[lane, i0, 2]
                                            ex = vertices[lane, i1, 0] - x0
                                            ey = vertices[lane, i1, 1] - y0
                                            ez = vertices[lane, i1, 2] - z0
                                            fx = vertices[lane, i2, 0] - x0
                                            fy = vertices[lane, i2, 1] - y0
                                            fz = vertices[lane, i2, 2] - z0
                                            ee = dot3(ex, ey, ez, ex, ey, ez)
                                            ff = dot3(fx, fy, fz, fx, fy, fz)
                                            ef = dot3(ex, ey, ez, fx, fy, fz)
                                            re = dot3(
                                                depth * dx - x0,
                                                depth * dy - y0,
                                                depth * dz - z0,
                                                ex,
                                                ey,
                                                ez,
                                            )
                                            rf = dot3(
                                                depth * dx - x0,
                                                depth * dy - y0,
                                                depth * dz - z0,
                                                fx,
                                                fy,
                                                fz,
                                            )
                                            denominator = ee * ff - ef * ef
                                            if denominator > 1e-12 * ee * ff:
                                                w1 = (re * ff - rf * ef) / denominator
                                                w2 = (rf * ee - re * ef) / denominator
                                                w0 = 1.0 - w1 - w2
                                                if w0 >= -1e-6 and w1 >= -1e-6 and w2 >= -1e-6:
                                                    w0 = tx.max(0.0, w0)
                                                    w1 = tx.max(0.0, w1)
                                                    w2 = tx.max(0.0, w2)
                                                    total = w0 + w1 + w2
                                                    weights[lane, 0] = w0 / total
                                                    weights[lane, 1] = w1 / total
                                                    weights[lane, 2] = w2 / total
                                                    weights[lane, 3] = 0.0
                                                    indices[lane, 0] = i0
                                                    indices[lane, 1] = i1
                                                    indices[lane, 2] = i2
                                                    indices[lane, 3] = 0
                                                    active_count = 3
                                                    chosen = 1
                                signed_gap = -depth * scale
                                lower_gap = tx.min(signed_gap, -reach * scale)
                                upper_gap = signed_gap
                                valid = 1
                                if chosen == 0:
                                    status = 2
                                    reason = 6
                                elif multiple != 0:
                                    status = 2
                                    reason = 2
                                else:
                                    status = 1
                                    valid = 7
                                    nx = dx
                                    ny = dy
                                    nz = dz
                            elif ball_accepted == 0 and duplicate != 0 and reach - depth > 1e-3:
                                # The repeated vertex lies well outside this plane,
                                # so the plane is not a face of the convex hull.
                                faces[lane, closest_face, 0] = -1
                                faces[lane, closest_face, 1] = -1
                                faces[lane, closest_face, 2] = -1
                                face_count = face_count - 1
                                if face_count <= 0:
                                    status = 2
                                    reason = 10
                            elif ball_accepted == 0 and duplicate != 0:
                                status = 2
                                reason = 10
                            elif ball_accepted == 0 and vertex_count >= vertex_capacity:
                                status = 4
                                reason = 7
                            elif ball_accepted == 0:
                                new_vertex = vertex_count
                                vertices[lane, new_vertex, 0] = px
                                vertices[lane, new_vertex, 1] = py
                                vertices[lane, new_vertex, 2] = pz
                                vertex_slots[lane, new_vertex, 0] = ia
                                vertex_slots[lane, new_vertex, 1] = ib
                                for c in tx.static_range(3):
                                    vertices[lane, new_vertex, 3 + c] = support_points[lane, 0, c]
                                    vertices[lane, new_vertex, 6 + c] = support_points[lane, 1, c]
                                vertex_count = vertex_count + 1
                                edge_count = 0
                                visible_count = 0
                                for face in range(face_capacity):
                                    if faces[lane, face, 0] >= 0:
                                        visibility = (
                                            dot3(
                                                face_geometry[lane, face, 0],
                                                face_geometry[lane, face, 1],
                                                face_geometry[lane, face, 2],
                                                px,
                                                py,
                                                pz,
                                            )
                                            - face_geometry[lane, face, 3]
                                        )
                                        if visibility > 1e-7:
                                            visible_count = visible_count + 1
                                            for edge in tx.static_range(3):
                                                v0 = faces[lane, face, edge]
                                                v1 = faces[lane, face, (edge + 1) % 3]
                                                opposite = -1
                                                for e in range(edge_count):
                                                    if (
                                                        edges[lane, e, 0] == v1
                                                        and edges[lane, e, 1] == v0
                                                    ):
                                                        opposite = e
                                                if opposite >= 0:
                                                    edges[lane, opposite, 0] = -1
                                                    edges[lane, opposite, 1] = -1
                                                else:
                                                    edges[lane, edge_count, 0] = v0
                                                    edges[lane, edge_count, 1] = v1
                                                    edge_count = edge_count + 1
                                            faces[lane, face, 0] = -1
                                            faces[lane, face, 1] = -1
                                            faces[lane, face, 2] = -1
                                            face_count = face_count - 1
                                if visible_count == 0:
                                    status = 2
                                    reason = 10
                                for edge in range(edge_count):
                                    if edges[lane, edge, 0] >= 0 and status == 0:
                                        c0 = edges[lane, edge, 0]
                                        c1 = edges[lane, edge, 1]
                                        c2 = new_vertex
                                        x0 = vertices[lane, c0, 0]
                                        y0 = vertices[lane, c0, 1]
                                        z0 = vertices[lane, c0, 2]
                                        ex = vertices[lane, c1, 0] - x0
                                        ey = vertices[lane, c1, 1] - y0
                                        ez = vertices[lane, c1, 2] - z0
                                        fx = px - x0
                                        fy = py - y0
                                        fz = pz - z0
                                        ux = ey * fz - ez * fy
                                        uy = ez * fx - ex * fz
                                        uz = ex * fy - ey * fx
                                        length = tx.sqrt(dot3(ux, uy, uz, ux, uy, uz))
                                        if length <= 1e-12:
                                            status = 2
                                            reason = 6
                                        else:
                                            ux = ux / length
                                            uy = uy / length
                                            uz = uz / length
                                            if dot3(ux, uy, uz, x0 - ix, y0 - iy, z0 - iz) < 0.0:
                                                swap = c0
                                                c0 = c1
                                                c1 = swap
                                                ux = -ux
                                                uy = -uy
                                                uz = -uz
                                            target = -1
                                            for face in range(face_capacity):
                                                if target < 0 and faces[lane, face, 0] < 0:
                                                    target = face
                                            if target < 0:
                                                status = 4
                                                reason = 7
                                            else:
                                                faces[lane, target, 0] = c0
                                                faces[lane, target, 1] = c1
                                                faces[lane, target, 2] = c2
                                                face_geometry[lane, target, 0] = ux
                                                face_geometry[lane, target, 1] = uy
                                                face_geometry[lane, target, 2] = uz
                                                face_geometry[lane, target, 3] = dot3(
                                                    ux, uy, uz, x0, y0, z0
                                                )
                                                face_count = face_count + 1
                if status == 0:
                    status = 4
                    reason = 5
        if (valid & 1) != 0:
            gap[out, 0] = signed_gap
            gap[out, 1] = lower_gap
            gap[out, 2] = upper_gap
            relation = 3
            if signed_gap > 1e-4 * scale:
                relation = 1
            elif signed_gap < -1e-4 * scale:
                relation = 2
            if relation == 3 and reason == 0:
                reason = 1
        if (valid & 2) != 0:
            normal[out, 0] = nx
            normal[out, 1] = ny
            normal[out, 2] = nz
        if (valid & 4) != 0:
            # Coalesce repeated material owners on each side, retaining exact
            # IDs. At most four simplex members contribute to either shape.
            for side in tx.static_range(2):
                used = 0
                core_x = 0.0
                core_y = 0.0
                core_z = 0.0
                for k in range(active_count):
                    weight = weights[lane, k]
                    selected = indices[lane, k]
                    if weight > 0.0:
                        slot = vertex_slots[lane, selected, side]
                        node = original_ids[slot]
                        if full_surface != 0:
                            core_x = core_x + weight * (points[slot, 0] - ox)
                            core_y = core_y + weight * (points[slot, 1] - oy)
                            core_z = core_z + weight * (points[slot, 2] - oz)
                        target = -1
                        for j in range(used):
                            if material_ids[out, side, j] == node:
                                target = j
                        if target < 0:
                            target = used
                            material_ids[out, side, target] = node
                            used = used + 1
                        barycentric[out, side, target] = barycentric[out, side, target] + weight
                        for c in tx.static_range(3):
                            surface[out, side, c] = (
                                surface[out, side, c]
                                + weight * vertices[lane, selected, 3 + 3 * side + c]
                            )
                # Order by material ID, independently of registration order.
                for sweep in tx.static_range(4):
                    for j in tx.static_range(3):
                        if (
                            j + 1 < used
                            and material_ids[out, side, j] > material_ids[out, side, j + 1]
                        ):
                            old_id = material_ids[out, side, j]
                            material_ids[out, side, j] = material_ids[out, side, j + 1]
                            material_ids[out, side, j + 1] = old_id
                            old_weight = barycentric[out, side, j]
                            barycentric[out, side, j] = barycentric[out, side, j + 1]
                            barycentric[out, side, j + 1] = old_weight
                meta[out, 10 + side] = used
                radius = ra
                if side == 1:
                    radius = -rb
                if full_surface != 0:
                    skin_offset[out, side, 0] = surface[out, side, 0] - core_x
                    skin_offset[out, side, 1] = surface[out, side, 1] - core_y
                    skin_offset[out, side, 2] = surface[out, side, 2] - core_z
                else:
                    skin_offset[out, side, 0] = radius * nx
                    skin_offset[out, side, 1] = radius * ny
                    skin_offset[out, side, 2] = radius * nz
                    for c in tx.static_range(3):
                        surface[out, side, c] = surface[out, side, c] + skin_offset[out, side, c]
        meta[out, 0] = status
        meta[out, 1] = relation
        meta[out, 2] = reason
        meta[out, 3] = phase
        meta[out, 4] = gjk_used
        meta[out, 5] = epa_used
        meta[out, 6] = support_calls
        meta[out, 7] = vertex_count
        meta[out, 8] = face_count
        meta[out, 9] = valid


def make_convex_query_kernels():
    """Plan-local binding handles; all execution uses its borrowed batch runtime."""
    return {
        "gather": (tx.kernel(gather_convex_pose), tx.kernel(gather_convex_pose)),
        "clear": tx.kernel(clear_convex_output),
        "trace_clear": tx.kernel(clear_convex_trace),
        "query": tx.kernel(query_convex_pairs),
    }
