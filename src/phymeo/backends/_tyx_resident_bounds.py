"""Conservative FP32 sphere AABBs from the current resident accepted buffer."""

import tyx as tx


@tx.func
def _lower(value: tx.f32) -> tx.f32:
    return tx.sub_rn(
        value, tx.max(1.1754943508222875e-38, tx.mul_rn(tx.abs(value), 2.384185791015625e-7))
    )


@tx.func
def _upper(value: tx.f32) -> tx.f32:
    return tx.add_rn(
        value, tx.max(1.1754943508222875e-38, tx.mul_rn(tx.abs(value), 2.384185791015625e-7))
    )


def resident_sphere_bounds(
    count: tx.i32,
    slots: tx.ndarray(tx.i32, 1),
    static_bounds: tx.ndarray(tx.f32, 2),
    state: tx.ndarray(tx.f32, 2),
    tick: tx.ndarray(tx.i32, 1),
    props: tx.ndarray(tx.f32, 2),
    bounds: tx.ndarray(tx.f32, 2),
    stamp: tx.ndarray(tx.i32, 2),
):
    # Two eps of each rounded endpoint cover initial addition and final outward
    # rounding. A normal-minimum floor also covers FTZ cancellation. Explicit RN
    # intrinsics prevent multiply/add contraction from shrinking this margin.
    # Bounds can be conservative false positives; overflow rejects the query.
    for i in tx.range(count):
        slot = slots[i]
        stamp[i, 0] = 1
        stamp[i, 1] = -1
        if slot < 0:
            for axis in tx.static_range(6):
                bounds[i, axis] = static_bounds[i, axis]
        else:
            radius = props[slot, 7]
            stamp[i, 1] = tick[slot]
            if not tx.isfinite(radius) or radius <= 0.0:
                stamp[i, 0] = 0
            for axis in tx.static_range(3):
                center = state[slot, axis]
                lo = _lower(tx.sub_rn(center, radius))
                hi = _upper(tx.add_rn(center, radius))
                bounds[i, axis] = lo
                bounds[i, axis + 3] = hi
                if not tx.isfinite(center) or not tx.isfinite(lo) or not tx.isfinite(hi) or lo > hi:
                    stamp[i, 0] = 0


def make_resident_bounds_kernels():
    """A separate binding handle per actual accepted-buffer direction and plan."""
    return tx.kernel(resident_sphere_bounds), tx.kernel(resident_sphere_bounds)
