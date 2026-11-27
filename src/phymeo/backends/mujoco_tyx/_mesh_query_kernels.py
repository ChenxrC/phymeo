"""Device geometry preparation/compaction for original convex mesh queries."""

import tyx as tx


@tx.func
def sum_error(a: tx.f32, b: tx.f32, total: tx.f32) -> tx.f32:
    return tx.add_rn(
        tx.sub_rn(a, tx.sub_rn(total, tx.sub_rn(total, a))),
        tx.sub_rn(b, tx.sub_rn(total, a)),
    )


@tx.kernel
def clear_counts(worlds: tx.i32, counts: tx.ndarray(tx.i32, 1)):
    for i in tx.range(worlds + 1):
        counts[i] = 0


@tx.kernel
def validate_response_input(
    worlds: tx.i32,
    capacity: tx.i32,
    tolerance: tx.f32,
    counts: tx.ndarray(tx.i32, 1),
    meta: tx.ndarray(tx.i32, 2),
    gap: tx.ndarray(tx.f32, 2),
    normal: tx.ndarray(tx.f32, 2),
    surface: tx.ndarray(tx.f32, 3),
):
    for flat in tx.range(worlds * capacity):
        world = flat // capacity
        if flat % capacity < counts[world]:
            invalid = 0
            if meta[flat, 0] != 1 or meta[flat, 9] != 7:
                invalid = 1
            if (
                not tx.isfinite(gap[flat, 0])
                or not tx.isfinite(gap[flat, 1])
                or not tx.isfinite(gap[flat, 2])
            ):
                invalid = 1
            if (
                gap[flat, 1] > gap[flat, 0]
                or gap[flat, 0] > gap[flat, 2]
                or gap[flat, 2] - gap[flat, 1] > tolerance
            ):
                invalid = 1
            square = 0.0
            closure = 0.0
            for axis in tx.static_range(3):
                n = normal[flat, axis]
                delta = surface[flat, 1, axis] - surface[flat, 0, axis] - gap[flat, 0] * n
                square = square + n * n
                closure = closure + delta * delta
                if not tx.isfinite(n) or not tx.isfinite(delta):
                    invalid = 1
            if tx.abs(square - 1.0) > 1e-4 or closure > tolerance * tolerance:
                invalid = 1
            if invalid != 0:
                tx.atomic_add(counts[worlds], 1)


@tx.kernel
def transform_meshes(
    total: tx.i32,
    ngeom: tx.i32,
    nvertex: tx.i32,
    worlds: tx.i32,
    offsets: tx.ndarray(tx.i32, 1),
    local: tx.ndarray(tx.f32, 2),
    positions: tx.ndarray(tx.f32, 3),
    rotations: tx.ndarray(tx.f32, 4),
    positions_low: tx.ndarray(tx.f32, 3),
    rotations_low: tx.ndarray(tx.f32, 4),
    points: tx.ndarray(tx.f32, 2),
    bounds: tx.ndarray(tx.f32, 2),
    counts: tx.ndarray(tx.i32, 1),
):
    for flat in tx.range(total):
        world = flat // ngeom
        geom = flat % ngeom
        valid = 1
        for axis in tx.static_range(3):
            if not tx.isfinite(positions[world, geom, axis]):
                valid = 0
            plow = positions_low[world, geom, axis]
            if (
                not tx.isfinite(plow)
                or tx.abs(plow) > 1.1920929e-7 * tx.abs(positions[world, geom, axis]) + 1e-38
            ):
                valid = 0
            for column in tx.static_range(3):
                if not tx.isfinite(rotations[world, geom, axis, column]):
                    valid = 0
                rlow = rotations_low[world, geom, axis, column]
                if (
                    not tx.isfinite(rlow)
                    or tx.abs(rlow)
                    > 1.1920929e-7 * tx.abs(rotations[world, geom, axis, column]) + 1e-38
                ):
                    valid = 0
                gram = 0.0
                for row in tx.static_range(3):
                    gram = (
                        gram
                        + rotations[world, geom, row, axis] * rotations[world, geom, row, column]
                    )
                target = 0.0
                if axis == column:
                    target = 1.0
                if tx.abs(gram - target) > 0.0001:
                    valid = 0
        determinant = (
            rotations[world, geom, 0, 0]
            * (
                rotations[world, geom, 1, 1] * rotations[world, geom, 2, 2]
                - rotations[world, geom, 1, 2] * rotations[world, geom, 2, 1]
            )
            - rotations[world, geom, 0, 1]
            * (
                rotations[world, geom, 1, 0] * rotations[world, geom, 2, 2]
                - rotations[world, geom, 1, 2] * rotations[world, geom, 2, 0]
            )
            + rotations[world, geom, 0, 2]
            * (
                rotations[world, geom, 1, 0] * rotations[world, geom, 2, 1]
                - rotations[world, geom, 1, 1] * rotations[world, geom, 2, 0]
            )
        )
        if determinant <= 0.0 or not tx.isfinite(determinant):
            valid = 0
        for axis in tx.static_range(3):
            low = 3.402823e38
            high = -3.402823e38
            for vertex in range(offsets[geom], offsets[geom + 1]):
                value = positions[world, geom, axis]
                error = positions_low[world, geom, axis]
                for column in tx.static_range(3):
                    rh = rotations[world, geom, axis, column]
                    vl = local[vertex, column]
                    prod = tx.mul_rn(rh, vl)
                    perr = tx.fma(rh, vl, -prod) + rotations_low[world, geom, axis, column] * vl
                    total_value = tx.add_rn(value, prod)
                    error = error + sum_error(value, prod, total_value) + perr
                    value = total_value
                value = tx.add_rn(value, error)
                points[world * nvertex + vertex, axis] = value
                if not tx.isfinite(value):
                    valid = 0
                low = tx.min(low, value)
                high = tx.max(high, value)
            # Conservative broad-phase padding only; narrow phase uses original vertices.
            pad = 0.000001 + 0.000001 * tx.max(tx.abs(low), tx.abs(high))
            bounds[flat, axis] = low - pad
            bounds[flat, axis + 3] = high + pad
        if valid == 0:
            tx.atomic_add(counts[worlds], 1)


@tx.kernel
def compact_mesh_pairs(
    total: tx.i32,
    npair: tx.i32,
    ngeom: tx.i32,
    capacity: tx.i32,
    pairs: tx.ndarray(tx.i32, 2),
    margins: tx.ndarray(tx.f32, 1),
    bounds: tx.ndarray(tx.f32, 2),
    counts: tx.ndarray(tx.i32, 1),
    output: tx.ndarray(tx.i32, 2),
):
    for flat in tx.range(total):
        world = flat // npair
        pair = flat % npair
        a = pairs[pair, 0]
        b = pairs[pair, 1]
        sa = world * ngeom + a
        sb = world * ngeom + b
        margin = tx.max(margins[a], margins[b])
        overlap = 1
        for axis in tx.static_range(3):
            if (
                bounds[sa, axis] > bounds[sb, axis + 3] + margin
                or bounds[sb, axis] > bounds[sa, axis + 3] + margin
            ):
                overlap = 0
        if overlap == 1:
            slot = tx.atomic_add(counts[world], 1)
            if slot < capacity:
                output[world * capacity + slot, 0] = sa
                output[world * capacity + slot, 1] = sb


@tx.kernel
def pack_local_pairs(
    count: tx.i32,
    first: tx.i32,
    ngeom: tx.i32,
    nvertex: tx.i32,
    max_vertices: tx.i32,
    offsets: tx.ndarray(tx.i32, 1),
    local: tx.ndarray(tx.f32, 2),
    positions: tx.ndarray(tx.f32, 3),
    rotations: tx.ndarray(tx.f32, 4),
    positions_low: tx.ndarray(tx.f32, 3),
    rotations_low: tx.ndarray(tx.f32, 4),
    bounds: tx.ndarray(tx.f32, 2),
    pairs: tx.ndarray(tx.i32, 2),
    tags: tx.ndarray(tx.i32, 2),
    ids: tx.ndarray(tx.i32, 1),
    packed_points: tx.ndarray(tx.f32, 2),
    packed_pairs: tx.ndarray(tx.i32, 2),
    frames: tx.ndarray(tx.f32, 2),
):
    for lane in tx.range(count):
        a = pairs[first + lane, 0]
        b = pairs[first + lane, 1]
        length = 0.0
        for axis in tx.static_range(3):
            low = tx.min(bounds[a, axis], bounds[b, axis])
            high = tx.max(bounds[a, axis + 3], bounds[b, axis + 3])
            frames[lane, axis] = 0.5 * low + 0.5 * high
            length = tx.max(length, high - low)
        # Broad-phase padding ensures positive length for every valid source.
        frames[lane, 3] = length
        for side in tx.static_range(2):
            shape = a
            if side == 1:
                shape = b
            geom = shape % ngeom
            world = shape // ngeom
            target = lane * 2 + side
            begin = target * max_vertices
            n = offsets[geom + 1] - offsets[geom]
            tags[target, 0] = 2
            tags[target, 1] = begin
            tags[target, 2] = begin + n
            tags[target, 3] = world
            packed_pairs[first + lane, side] = target
            for i in range(n):
                source = world * nvertex + offsets[geom] + i
                ids[begin + i] = source
                for axis in tx.static_range(3):
                    origin = -frames[lane, axis]
                    ph = positions[world, geom, axis]
                    value = tx.add_rn(ph, origin)
                    error = positions_low[world, geom, axis] + sum_error(ph, origin, value)
                    for column in tx.static_range(3):
                        rh = rotations[world, geom, axis, column]
                        vl = local[offsets[geom] + i, column]
                        prod = tx.mul_rn(rh, vl)
                        perr = tx.fma(rh, vl, -prod) + rotations_low[world, geom, axis, column] * vl
                        total_value = tx.add_rn(value, prod)
                        error = error + sum_error(value, prod, total_value) + perr
                        value = total_value
                    quotient = value / length
                    qerror = (tx.fma(-quotient, length, value) + error) / length
                    high_part = tx.add_rn(quotient, qerror)
                    packed_points[begin + i, axis] = high_part


@tx.kernel
def unpack_local_pairs(
    count: tx.i32,
    first: tx.i32,
    frames: tx.ndarray(tx.f32, 2),
    meta: tx.ndarray(tx.i32, 2),
    gap: tx.ndarray(tx.f32, 2),
    surface: tx.ndarray(tx.f32, 3),
    skin_offset: tx.ndarray(tx.f32, 3),
):
    for lane in tx.range(count):
        out = first + lane
        length = frames[lane, 3]
        for component in tx.static_range(4):
            gap[out, component] = gap[out, component] * length
        if (meta[out, 9] & 4) != 0:
            for side in tx.static_range(2):
                for axis in tx.static_range(3):
                    surface[out, side, axis] = (
                        frames[lane, axis] + surface[out, side, axis] * length
                    )
                    skin_offset[out, side, axis] = skin_offset[out, side, axis] * length
