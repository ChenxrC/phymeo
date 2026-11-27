"""Small scalar building blocks for the bounded device convex query.

TYX's public pure-function API takes a single return expression. Support scans,
finite-simplex ownership and EPA topology consequently live in the query kernel,
not in a hidden HOST search or an unsupported imperative device function.
"""

import tyx as tx


@tx.func
def difference_of_products(a: tx.f32, b: tx.f32, c: tx.f32, d: tx.f32) -> tx.f32:
    # Recover the rounded second product before subtracting two nearly equal
    # products. Explicit FMA preserves thin-simplex cross products in FP32.
    return tx.sub_rn(tx.fma(a, b, -tx.mul_rn(c, d)), tx.fma(c, d, -tx.mul_rn(c, d)))


@tx.func
def dot3(ax: tx.f32, ay: tx.f32, az: tx.f32, bx: tx.f32, by: tx.f32, bz: tx.f32) -> tx.f32:
    return tx.add_rn(tx.add_rn(tx.mul_rn(ax, bx), tx.mul_rn(ay, by)), tx.mul_rn(az, bz))


@tx.func
def det3(
    ax: tx.f32,
    ay: tx.f32,
    az: tx.f32,
    bx: tx.f32,
    by: tx.f32,
    bz: tx.f32,
    cx: tx.f32,
    cy: tx.f32,
    cz: tx.f32,
) -> tx.f32:
    return tx.fma(
        ax,
        difference_of_products(by, cz, bz, cy),
        tx.fma(
            ay, difference_of_products(bz, cx, bx, cz), az * difference_of_products(bx, cy, by, cx)
        ),
    )


@tx.func
def clamp01(value: tx.f32) -> tx.f32:
    return tx.max(0.0, tx.min(1.0, value))
