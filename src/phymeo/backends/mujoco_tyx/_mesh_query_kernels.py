"""Device geometry preparation/compaction for original convex mesh queries."""

import tyx as tx


@tx.kernel
def clear_counts(worlds: tx.i32, counts: tx.ndarray(tx.i32, 1)):
    for i in tx.range(worlds + 1):
        counts[i] = 0


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
            for column in tx.static_range(3):
                if not tx.isfinite(rotations[world, geom, axis, column]):
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
                for column in tx.static_range(3):
                    value = value + rotations[world, geom, axis, column] * local[vertex, column]
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
    points: tx.ndarray(tx.f32, 2),
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
            frames[lane, axis] = low
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
                    packed_points[begin + i, axis] = (
                        points[source, axis] - frames[lane, axis]
                    ) / length


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
