"""Finite material bounds directly from the accepted floating FK and root banks.

Outward engineering padding includes one real root rotation, local translation,
all ancestor slide travel, original material reach and anchor rounding. These
are current-pose bounds, not CCD or a formal all-real trigonometric certificate.
"""

import tyx as tx

from ._tyx_resident_revolute_bounds import _lower, _upper


def resident_floating_bounds(
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
    root: tx.ndarray(tx.f32, 2),
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
            if n < 0 or n > 16 or group < 0 or group > n or first >= last:
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
                for axis in tx.static_range(3):
                    translation = root[row, axis]
                    if not tx.isfinite(translation) or tx.abs(translation) > 4096.0:
                        stamp[shape, 0] = 0
                    length = _upper(tx.add_rn(length, tx.abs(translation)))
                ancestor = group
                depth = 0
                for walk in range(n + 1):
                    if ancestor >= 0:
                        if ancestor > 0:
                            q = state[row, ancestor - 1, 0]
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
                # depth includes the physical quaternion root; the extra unit
                # covers the final point transform/anchor addition, as in C4.
                layers = tx.add_rn(declared_depth, 1.0)
                base_error = _upper(tx.mul_rn(7.62939453125e-6, scale))
                composition_error = _upper(tx.mul_rn(tx.mul_rn(1.52587890625e-5, layers), length))
                trig_error = _upper(tx.mul_rn(tx.mul_rn(1.52587890625e-5, layers), length))
                ftz_error = _upper(tx.mul_rn(3.76158192263132e-37, layers))
                padding = _upper(
                    tx.add_rn(
                        _upper(tx.add_rn(base_error, composition_error)),
                        _upper(tx.add_rn(trig_error, ftz_error)),
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


def make_floating_bounds_kernels():
    return tx.kernel(resident_floating_bounds), tx.kernel(resident_floating_bounds)
