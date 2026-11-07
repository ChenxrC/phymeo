"""Finite-shape engineering bounds from one accepted persistent tree FK bank.

This pass consumes only original group-local material and the borrowed device
state/FK. It writes the existing BVH bounds directly, without a HOST pose bridge.
The predeclared scale/depth padding is an engineering envelope, not a formal
all-real sin/cos interval proof. No temporal sweep or contact response is implied.
"""

import tyx as tx

from ._tyx_resident_revolute_bounds import _lower, _upper


def resident_tree_bounds(
    count: tx.i32,
    rows: tx.ndarray(tx.i32, 1),
    groups: tx.ndarray(tx.i32, 1),
    offsets: tx.ndarray(tx.i32, 1),
    points: tx.ndarray(tx.f32, 2),
    geometry: tx.ndarray(tx.f32, 2),
    meta: tx.ndarray(tx.i32, 2),
    parents: tx.ndarray(tx.i32, 2),
    kinds: tx.ndarray(tx.i32, 2),
    anchor: tx.ndarray(tx.f32, 2),
    state: tx.ndarray(tx.f32, 3),
    fk: tx.ndarray(tx.f32, 3),
    ticks: tx.ndarray(tx.i32, 1),
    bounds: tx.ndarray(tx.f32, 2),
    stamp: tx.ndarray(tx.i32, 2),
):
    for shape in tx.range(count):
        row = rows[shape]
        if row >= 0:
            group = groups[shape]
            n = meta[row, 1]
            slot = meta[row, 0]
            first = offsets[shape]
            last = offsets[shape + 1]
            stamp[shape, 0] = 1
            stamp[shape, 1] = ticks[slot]
            if group < 0 or group >= n or n < 1 or n > 16 or first >= last:
                stamp[shape, 0] = 0
            else:
                radius = geometry[shape, 0]
                local_reach = geometry[shape, 1]
                path_reach = geometry[shape, 2]
                declared_depth = geometry[shape, 3]
                for column in tx.static_range(4):
                    if not tx.isfinite(geometry[shape, column]) or geometry[shape, column] < 0.0:
                        stamp[shape, 0] = 0
                length = _upper(tx.add_rn(_upper(tx.add_rn(local_reach, path_reach)), radius))
                ancestor = group
                depth = 0
                for walk in range(n):
                    if ancestor >= 0:
                        q = state[row, ancestor, 0]
                        if not tx.isfinite(q) or tx.abs(q) > 4096.0:
                            stamp[shape, 0] = 0
                        if kinds[row, ancestor] == 1:
                            length = _upper(tx.add_rn(length, tx.abs(q)))
                        ancestor = parents[row, ancestor]
                        depth = depth + 1
                if ancestor >= 0 or declared_depth != tx.cast_f32(depth):
                    stamp[shape, 0] = 0
                scale = tx.max(1.0, length)
                for axis in tx.static_range(3):
                    if not tx.isfinite(anchor[row, axis]):
                        stamp[shape, 0] = 0
                    scale = tx.max(scale, tx.abs(anchor[row, axis]))
                layers = tx.add_rn(declared_depth, 1.0)
                # Exactly the frozen casebook budget, rounded outward at each
                # nonnegative accumulation; current P travel is read above.
                base_error = _upper(tx.mul_rn(7.62939453125e-6, scale))
                composition_error = _upper(tx.mul_rn(tx.mul_rn(1.52587890625e-5, layers), length))
                trigonometric_error = _upper(tx.mul_rn(tx.mul_rn(1.52587890625e-5, layers), length))
                ftz_error = _upper(tx.mul_rn(3.76158192263132e-37, layers))
                padding = _upper(
                    tx.add_rn(
                        _upper(tx.add_rn(base_error, composition_error)),
                        _upper(tx.add_rn(trigonometric_error, ftz_error)),
                    )
                )
                skin = _upper(tx.add_rn(radius, padding))
                if (
                    not tx.isfinite(length)
                    or not tx.isfinite(padding)
                    or not tx.isfinite(skin)
                    or padding <= 0.0
                ):
                    stamp[shape, 0] = 0
                for axis in tx.static_range(3):
                    low = 0.0
                    high = 0.0
                    for vertex in range(first, last):
                        point = fk[row, group, axis]
                        for local_axis in tx.static_range(3):
                            point = tx.add_rn(
                                point,
                                tx.mul_rn(
                                    fk[row, group, 3 + axis * 3 + local_axis],
                                    points[vertex, local_axis],
                                ),
                            )
                        point = tx.add_rn(anchor[row, axis], point)
                        if not tx.isfinite(point):
                            stamp[shape, 0] = 0
                        if vertex == first:
                            low = point
                            high = point
                        else:
                            low = tx.min(low, point)
                            high = tx.max(high, point)
                    lo = _lower(tx.sub_rn(low, skin))
                    hi = _upper(tx.add_rn(high, skin))
                    bounds[shape, axis] = lo
                    bounds[shape, axis + 3] = hi
                    if not tx.isfinite(lo) or not tx.isfinite(hi) or lo > hi:
                        stamp[shape, 0] = 0


def make_tree_bounds_kernels():
    """Independent handles selected with the parent's accepted active bank."""
    return tx.kernel(resident_tree_bounds), tx.kernel(resident_tree_bounds)
