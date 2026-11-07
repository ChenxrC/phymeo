"""Conservative translated finite-shape bounds from accepted resident slider q."""

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


def resident_slider_bounds(
    count: tx.i32,
    slots: tx.ndarray(tx.i32, 1),
    zero_bounds: tx.ndarray(tx.f32, 2),
    axis_bounds: tx.ndarray(tx.f32, 2),
    state: tx.ndarray(tx.f32, 2),
    tick: tx.ndarray(tx.i32, 1),
    bounds: tx.ndarray(tx.f32, 2),
    stamp: tx.ndarray(tx.i32, 2),
):
    # The first sphere/static pass is already complete in the same borrowed
    # accepted view. Leave every non-slider row, including its stamp, untouched.
    for i in tx.range(count):
        slot = slots[i]
        if slot >= 0:
            q = state[slot, 0]
            stamp[i, 0] = 1
            stamp[i, 1] = tick[slot]
            if not tx.isfinite(q):
                stamp[i, 0] = 0
            for component in tx.static_range(3):
                a_lo = axis_bounds[i, component]
                a_hi = axis_bounds[i, component + 3]
                z_lo = zero_bounds[i, component]
                z_hi = zero_bounds[i, component + 3]
                lo_axis = a_lo
                hi_axis = a_hi
                if q < 0.0:
                    lo_axis = a_hi
                    hi_axis = a_lo
                p_lo = _lower(tx.mul_rn(q, lo_axis))
                p_hi = _upper(tx.mul_rn(q, hi_axis))
                # Expand the product BEFORE adding the zero geometry. Its error
                # depends on the large operands even if their sum nearly cancels.
                # RN operations prohibit FMA contraction. Minimum-normal floors
                # cover rounded/subnormal/FTZ intermediates in both operations.
                lo = _lower(tx.add_rn(z_lo, p_lo))
                hi = _upper(tx.add_rn(z_hi, p_hi))
                bounds[i, component] = lo
                bounds[i, component + 3] = hi
                if (
                    not tx.isfinite(a_lo)
                    or not tx.isfinite(a_hi)
                    or not tx.isfinite(z_lo)
                    or not tx.isfinite(z_hi)
                    or not tx.isfinite(lo)
                    or not tx.isfinite(hi)
                    or a_lo > a_hi
                    or z_lo > z_hi
                    or lo > hi
                ):
                    stamp[i, 0] = 0


def make_resident_slider_bounds_kernels():
    """Return two independent handles for the owner's two accepted buffers."""
    return tx.kernel(resident_slider_bounds), tx.kernel(resident_slider_bounds)
