"""Small scalar building blocks for the bounded device convex query.

TYX's public pure-function API takes a single return expression. Support scans,
finite-simplex ownership and EPA topology consequently live in the query kernel,
not in a hidden HOST search or an unsupported imperative device function.
"""

import tyx as tx


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
    return ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)


@tx.func
def clamp01(value: tx.f32) -> tx.f32:
    return tx.max(0.0, tx.min(1.0, value))
