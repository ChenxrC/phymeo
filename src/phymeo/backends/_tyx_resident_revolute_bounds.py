"""Engineering AABBs of actual accepted single-revolute finite material poses.

Uses ordinary sin/cos, a predeclared scale-dependent padding, and outward RN
endpoint operations. These require engineering validation, not formal interval claims.
Only explicitly registered rotor rows are written; no HOST state bridge exists.
"""

import tyx as tx


@tx.func
def _lower(value: tx.f32) -> tx.f32:
    return tx.sub_rn(value, tx.max(1.1754943508222875e-38, tx.mul_rn(tx.abs(value), 2.384185791015625e-7)))


@tx.func
def _upper(value: tx.f32) -> tx.f32:
    return tx.add_rn(value, tx.max(1.1754943508222875e-38, tx.mul_rn(tx.abs(value), 2.384185791015625e-7)))


def resident_revolute_bounds(
    count: tx.i32,
    slots: tx.ndarray(tx.i32, 1),
    offsets: tx.ndarray(tx.i32, 1),
    zero_relative_points: tx.ndarray(tx.f32, 2),
    geometry: tx.ndarray(tx.f32, 2),
    state: tx.ndarray(tx.f32, 2),
    tick: tx.ndarray(tx.i32, 1),
    bounds: tx.ndarray(tx.f32, 2),
    stamp: tx.ndarray(tx.i32, 2),
):
    for i in tx.range(count):
        slot = slots[i]
        if slot >= 0:
            q = state[slot, 0]
            first = offsets[i]
            last = offsets[i + 1]
            radius = geometry[i, 6]
            padding = geometry[i, 7]
            stamp[i, 0] = 1
            stamp[i, 1] = tick[slot]
            if (
                not tx.isfinite(q)
                or tx.abs(q) > 4096.0
                or first >= last
                or not tx.isfinite(radius)
                or radius < 0.0
                or not tx.isfinite(padding)
                or padding <= 0.0
            ):
                stamp[i, 0] = 0
            cosine = tx.cos(q)
            sine = tx.sin(q)
            remaining = tx.sub_rn(1.0, cosine)
            skin = _upper(tx.add_rn(radius, padding))
            if not tx.isfinite(cosine) or not tx.isfinite(sine) or not tx.isfinite(skin):
                stamp[i, 0] = 0
            for j in tx.static_range(3):
                low = 0.0
                high = 0.0
                pivot = geometry[i, j]
                if not tx.isfinite(pivot) or not tx.isfinite(geometry[i, 3 + j]):
                    stamp[i, 0] = 0
                for vertex in range(first, last):
                    dot = 0.0
                    for k in tx.static_range(3):
                        dot = tx.add_rn(dot, tx.mul_rn(geometry[i, 3 + k], zero_relative_points[vertex, k]))
                    cross = tx.sub_rn(
                        tx.mul_rn(geometry[i, 3 + (j + 1) % 3], zero_relative_points[vertex, (j + 2) % 3]),
                        tx.mul_rn(geometry[i, 3 + (j + 2) % 3], zero_relative_points[vertex, (j + 1) % 3]),
                    )
                    turned = tx.add_rn(
                        tx.add_rn(tx.mul_rn(cosine, zero_relative_points[vertex, j]), tx.mul_rn(sine, cross)),
                        tx.mul_rn(tx.mul_rn(remaining, dot), geometry[i, 3 + j]),
                    )
                    point = tx.add_rn(pivot, turned)
                    if not tx.isfinite(point):
                        stamp[i, 0] = 0
                    if vertex == first:
                        low = point
                        high = point
                    else:
                        low = tx.min(low, point)
                        high = tx.max(high, point)
                lo = _lower(tx.sub_rn(low, skin))
                hi = _upper(tx.add_rn(high, skin))
                bounds[i, j] = lo
                bounds[i, j + 3] = hi
                if not tx.isfinite(lo) or not tx.isfinite(hi) or lo > hi:
                    stamp[i, 0] = 0


def make_resident_revolute_bounds_kernels():
    """Independent query handles for the owner's two accepted state buffers."""
    return tx.kernel(resident_revolute_bounds), tx.kernel(resident_revolute_bounds)

