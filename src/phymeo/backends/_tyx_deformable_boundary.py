"""Gather accepted node coordinates; no integration or collision mathematics."""

import tyx as tx


@tx.kernel
def gather_deformable_points(
    count: tx.i32,
    selected_world: tx.i32,
    node_count: tx.i32,
    shape_meta: tx.ndarray(tx.i32, 2),
    node_indices: tx.ndarray(tx.i32, 1),
    local: tx.ndarray(tx.f32, 2),
    local_bounds: tx.ndarray(tx.f32, 2),
    position: tx.ndarray(tx.f32, 2),
    tick: tx.ndarray(tx.i32, 1),
    points: tx.ndarray(tx.f32, 2),
    point_bounds: tx.ndarray(tx.f32, 2),
    stamps: tx.ndarray(tx.i32, 2),
    ticks: tx.ndarray(tx.i32, 1),
):
    for lane in tx.range(count + 1):
        if lane == count:
            ticks[selected_world] = tick[0]
        else:
            if shape_meta[lane, 0] == selected_world:
                row = shape_meta[lane, 1]
                begin = shape_meta[lane, 4]
                end = shape_meta[lane, 5]
                valid = 1
                for point in range(begin, end):
                    index = node_indices[point]
                    if row >= 0 and (index < 0 or index >= node_count):
                        valid = 0
                    for a in tx.static_range(3):
                        value = local[point, a]
                        lower = local_bounds[point, a]
                        upper = local_bounds[point, a + 3]
                        if row >= 0 and index >= 0 and index < node_count:
                            value = position[index, a]
                            lower = value
                            upper = value
                        if not tx.isfinite(value) or not tx.isfinite(lower) or not tx.isfinite(upper):
                            valid = 0
                        points[point, a] = value
                        point_bounds[point, a] = lower
                        point_bounds[point, a + 3] = upper
                stamps[lane, 0] = valid
                stamps[lane, 1] = tick[0]
                stamps[lane, 2] = selected_world
                stamps[lane, 3] = row
