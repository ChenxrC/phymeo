"""Single-expression TYX support components for dimension-normalized curved solids.

The caller supplies a nonzero direction and positive admitted dimensions. Query
kernels normalize length/direction before calling these expressions. Exact zero
denominators use one only to keep invalid/zero helper inputs finite; query
admission and direction handling own their explicit status.
"""

import tyx as tx


@tx.func
def curved_support_x(
    kind: tx.i32, dx: tx.f32, dy: tx.f32, dz: tx.f32, dim0: tx.f32, dim1: tx.f32, dim2: tx.f32
) -> tx.f32:
    return (
        dim0
        * dim0
        * dx
        / (
            tx.sqrt(
                (dim0 * dx) * (dim0 * dx) + (dim1 * dy) * (dim1 * dy) + (dim2 * dz) * (dim2 * dz)
            )
            if (dim0 * dx) * (dim0 * dx) + (dim1 * dy) * (dim1 * dy) + (dim2 * dz) * (dim2 * dz)
            > 0.0
            else 1.0
        )
        if kind == 6
        else 0.0
        if kind == 5 and 2.0 * dim1 * dz >= dim0 * tx.sqrt(dx * dx + dy * dy)
        else dim0 * dx / (tx.sqrt(dx * dx + dy * dy) if dx * dx + dy * dy > 0.0 else 1.0)
    )


@tx.func
def curved_support_y(
    kind: tx.i32, dx: tx.f32, dy: tx.f32, dz: tx.f32, dim0: tx.f32, dim1: tx.f32, dim2: tx.f32
) -> tx.f32:
    return (
        dim1
        * dim1
        * dy
        / (
            tx.sqrt(
                (dim0 * dx) * (dim0 * dx) + (dim1 * dy) * (dim1 * dy) + (dim2 * dz) * (dim2 * dz)
            )
            if (dim0 * dx) * (dim0 * dx) + (dim1 * dy) * (dim1 * dy) + (dim2 * dz) * (dim2 * dz)
            > 0.0
            else 1.0
        )
        if kind == 6
        else 0.0
        if kind == 5 and 2.0 * dim1 * dz >= dim0 * tx.sqrt(dx * dx + dy * dy)
        else dim0 * dy / (tx.sqrt(dx * dx + dy * dy) if dx * dx + dy * dy > 0.0 else 1.0)
    )


@tx.func
def curved_support_z(
    kind: tx.i32, dx: tx.f32, dy: tx.f32, dz: tx.f32, dim0: tx.f32, dim1: tx.f32, dim2: tx.f32
) -> tx.f32:
    return (
        dim2
        * dim2
        * dz
        / (
            tx.sqrt(
                (dim0 * dx) * (dim0 * dx) + (dim1 * dy) * (dim1 * dy) + (dim2 * dz) * (dim2 * dz)
            )
            if (dim0 * dx) * (dim0 * dx) + (dim1 * dy) * (dim1 * dy) + (dim2 * dz) * (dim2 * dz)
            > 0.0
            else 1.0
        )
        if kind == 6
        else dim1
        if kind == 5 and 2.0 * dim1 * dz >= dim0 * tx.sqrt(dx * dx + dy * dy)
        else -dim1
        if kind == 5
        else dim1
        if dz > 0.0
        else -dim1
        if dz < 0.0
        else 0.0
    )


@tx.func
def support_ball_depth_width(
    reach: tx.f32, tangent_square: tx.f32, radius: tx.f32, rounding: tx.f32
) -> tx.f32:
    return (
        2.0 * rounding
        + (tx.sqrt(tx.max(0.0, tangent_square)) + 2.0 * rounding)
        * (tx.sqrt(tx.max(0.0, tangent_square)) + 2.0 * rounding)
        / (2.0 * (radius - reach - rounding))
        if radius - reach > rounding
        else 1e30
    )


@tx.func
def support_ball_normal_bound(
    reach: tx.f32, tangent_square: tx.f32, radius: tx.f32, rounding: tx.f32
) -> tx.f32:
    return (
        2.0 * (tx.sqrt(tx.max(0.0, tangent_square)) + 2.0 * rounding)
        / (radius - reach - rounding)
        if radius - reach > rounding
        else 1e30
    )
