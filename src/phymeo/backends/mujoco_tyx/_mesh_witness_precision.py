"""Original support-feature witnesses, evaluated entirely on the device.

Two-lane coordinates/axes classify the support set before solving for material
weights. A unique vertex/face or nonparallel edge/edge point is reconstructed;
nonunique manifolds retain the existing certified witness. No native contact
point, per-pose host geometry or contact margin is used.
"""

import tyx as tx

from .._tyx_convex_support import difference_of_products
from ._mesh_query_kernels import sum_error


@tx.kernel
def refine_witnesses(
    count: tx.i32,
    first: tx.i32,
    tags: tx.ndarray(tx.i32, 2),
    points: tx.ndarray(tx.f32, 2),
    points_low: tx.ndarray(tx.f32, 2),
    ids: tx.ndarray(tx.i32, 1),
    frames: tx.ndarray(tx.f32, 2),
    feature_axis: tx.ndarray(tx.f32, 3),
    corrected: tx.ndarray(tx.i32, 1),
    projections: tx.ndarray(tx.f32, 4),
    active: tx.ndarray(tx.i32, 3),
    vectors: tx.ndarray(tx.f32, 3),
    selected: tx.ndarray(tx.i32, 3),
    selected_weights: tx.ndarray(tx.f32, 3),
    receipt: tx.ndarray(tx.i32, 2),
    meta: tx.ndarray(tx.i32, 2),
    gap: tx.ndarray(tx.f32, 2),
    normal: tx.ndarray(tx.f32, 2),
    surface: tx.ndarray(tx.f32, 3),
    surface_low: tx.ndarray(tx.f32, 3),
    material: tx.ndarray(tx.i32, 3),
    weights: tx.ndarray(tx.f32, 3),
    skin: tx.ndarray(tx.f32, 3),
):
    for lane in tx.range(count):
        out = first + lane
        # Status: 0 inapplicable; 1 not uniquely reconstructed, retained;
        # 2 unique reconstructed;
        # -1 unique feature failed its material/closure certificate (fail closed).
        for k in range(3):
            receipt[out, k] = 0
            surface_low[out, 0, k] = 0.0
            surface_low[out, 1, k] = 0.0
        if corrected[out] == 2:
            square = 0.0
            for axis in range(3):
                square = square + (feature_axis[out, axis, 0] * feature_axis[out, axis, 0])
            # Arithmetic envelope for two-lane FP32 dot products. Coordinates
            # are pair-centered and normalized, NOT a physical contact margin.
            tol = 128.0 * 1.4210854715202004e-14 * tx.sqrt(square)
            for side in range(2):
                begin = tags[2 * lane + side, 1]
                end = tags[2 * lane + side, 2]
                best_hi = -3.402823e38
                best_lo = 0.0
                direction = 1.0
                if side == 1:
                    direction = -1.0
                for vertex in range(begin, end):
                    hi = 0.0
                    lo = 0.0
                    for axis in range(3):
                        ph = points[vertex, axis]
                        pl = points_low[vertex, axis]
                        ah = direction * feature_axis[out, axis, 0]
                        al = direction * feature_axis[out, axis, 1]
                        prod = tx.mul_rn(ph, ah)
                        err = tx.fma(ph, ah, -prod) + ph * al + pl * ah + pl * al
                        total = tx.add_rn(hi, prod)
                        lo = lo + (sum_error(hi, prod, total) + err)
                        hi = total
                    projections[lane, side, vertex - begin, 0] = hi
                    projections[lane, side, vertex - begin, 1] = lo
                    if (hi - best_hi) + (lo - best_lo) > 0.0:
                        best_hi = hi
                        best_lo = lo
                used = 0
                for vertex in range(begin, end):
                    hi = projections[lane, side, vertex - begin, 0]
                    lo = projections[lane, side, vertex - begin, 1]
                    if (best_hi - hi) + (best_lo - lo) <= tol:
                        active[lane, side, used] = vertex
                        used = used + (1)
                receipt[out, side + 1] = used
                for k in range(4):
                    selected[lane, side, k] = -1
                    selected_weights[lane, side, k] = 0.0
            na = receipt[out, 1]
            nb = receipt[out, 2]
            unique = 0
            found = 0
            if na == 2 and nb == 2:
                # Intersect A's edge with B's edge translated by -gap*n.
                a = active[lane, 0, 0]
                a1 = active[lane, 0, 1]
                b = active[lane, 1, 0]
                b1 = active[lane, 1, 1]
                for axis in range(3):
                    vectors[lane, 0, axis] = (points[a1, axis] - points[a, axis]) + (
                        points_low[a1, axis] - points_low[a, axis]
                    )
                    vectors[lane, 1, axis] = (points[b1, axis] - points[b, axis]) + (
                        points_low[b1, axis] - points_low[b, axis]
                    )
                    vectors[lane, 2, axis] = (
                        (points[b, axis] - points[a, axis])
                        + (points_low[b, axis] - points_low[a, axis])
                    ) - gap[out, 0] * normal[out, axis]
                denominator = 0.0
                numerator_u = 0.0
                numerator_v = 0.0
                ee = 0.0
                ff = 0.0
                for axis in range(3):
                    j = (axis + 1) % 3
                    k = (axis + 2) % 3
                    h = difference_of_products(
                        vectors[lane, 0, j],
                        vectors[lane, 1, k],
                        vectors[lane, 0, k],
                        vectors[lane, 1, j],
                    )
                    ru = difference_of_products(
                        vectors[lane, 2, j],
                        vectors[lane, 1, k],
                        vectors[lane, 2, k],
                        vectors[lane, 1, j],
                    )
                    rv = difference_of_products(
                        vectors[lane, 2, j],
                        vectors[lane, 0, k],
                        vectors[lane, 2, k],
                        vectors[lane, 0, j],
                    )
                    denominator = denominator + (h * h)
                    numerator_u = numerator_u + (ru * h)
                    numerator_v = numerator_v + (rv * h)
                    ee = ee + (vectors[lane, 0, axis] * vectors[lane, 0, axis])
                    ff = ff + (vectors[lane, 1, axis] * vectors[lane, 1, axis])
                if denominator > 0.0:
                    # A nonzero but ill-conditioned intersection is not a
                    # nonunique manifold: refuse response instead of guessing.
                    unique = 1
                if denominator > 1e-12 * ee * ff:
                    unique = 1
                    u = numerator_u / denominator
                    v = numerator_v / denominator
                    if u >= -2e-6 and u <= 1.000002 and v >= -2e-6 and v <= 1.000002:
                        u = tx.min(1.0, tx.max(0.0, u))
                        v = tx.min(1.0, tx.max(0.0, v))
                        selected[lane, 0, 0] = a
                        selected[lane, 0, 1] = a1
                        selected[lane, 1, 0] = b
                        selected[lane, 1, 1] = b1
                        selected_weights[lane, 0, 0] = 1.0 - u
                        selected_weights[lane, 0, 1] = u
                        selected_weights[lane, 1, 0] = 1.0 - v
                        selected_weights[lane, 1, 1] = v
                        found = 1
            elif na == 1 or nb == 1:
                unique = 1
                side = 0
                if nb == 1:
                    side = 1
                other = 1 - side
                vertex = active[lane, side, 0]
                selected[lane, side, 0] = vertex
                selected_weights[lane, side, 0] = 1.0
                n = receipt[out, other + 1]
                sign = 1.0
                if side == 1:
                    sign = -1.0
                if n == 1:
                    selected[lane, other, 0] = active[lane, other, 0]
                    selected_weights[lane, other, 0] = 1.0
                    found = 1
                elif n >= 3:
                    # All candidate triangles from the small original support
                    # face. Select the first containing the translated vertex.
                    for i in range(n - 2):
                        for j in range(i + 1, n - 1):
                            for k in range(j + 1, n):
                                if found == 0:
                                    p = active[lane, other, i]
                                    q = active[lane, other, j]
                                    r = active[lane, other, k]
                                    for axis in range(3):
                                        vectors[lane, 0, axis] = (
                                            points[q, axis] - points[p, axis]
                                        ) + (points_low[q, axis] - points_low[p, axis])
                                        vectors[lane, 1, axis] = (
                                            points[r, axis] - points[p, axis]
                                        ) + (points_low[r, axis] - points_low[p, axis])
                                        vectors[lane, 2, axis] = (
                                            (points[vertex, axis] - points[p, axis])
                                            + (points_low[vertex, axis] - points_low[p, axis])
                                        ) + sign * gap[out, 0] * normal[out, axis]
                                    denominator = 0.0
                                    nu = 0.0
                                    nv = 0.0
                                    for axis in range(3):
                                        x = (axis + 1) % 3
                                        y = (axis + 2) % 3
                                        h = difference_of_products(
                                            vectors[lane, 0, x],
                                            vectors[lane, 1, y],
                                            vectors[lane, 0, y],
                                            vectors[lane, 1, x],
                                        )
                                        cu = difference_of_products(
                                            vectors[lane, 2, x],
                                            vectors[lane, 1, y],
                                            vectors[lane, 2, y],
                                            vectors[lane, 1, x],
                                        )
                                        cv = difference_of_products(
                                            vectors[lane, 0, x],
                                            vectors[lane, 2, y],
                                            vectors[lane, 0, y],
                                            vectors[lane, 2, x],
                                        )
                                        denominator = denominator + (h * h)
                                        nu = nu + (cu * h)
                                        nv = nv + (cv * h)
                                    if denominator > 1e-24:
                                        u = nu / denominator
                                        v = nv / denominator
                                        if u >= -2e-6 and v >= -2e-6 and u + v <= 1.000002:
                                            u = tx.max(0.0, u)
                                            v = tx.max(0.0, v)
                                            scale = tx.max(1.0, u + v)
                                            u = u / scale
                                            v = v / scale
                                            selected[lane, other, 0] = p
                                            selected[lane, other, 1] = q
                                            selected[lane, other, 2] = r
                                            selected_weights[lane, other, 0] = tx.max(
                                                0.0, 1.0 - u - v
                                            )
                                            selected_weights[lane, other, 1] = u
                                            selected_weights[lane, other, 2] = v
                                            found = 1
                elif n == 2:
                    p = active[lane, other, 0]
                    q = active[lane, other, 1]
                    numerator = 0.0
                    denominator = 0.0
                    for axis in range(3):
                        e = (points[q, axis] - points[p, axis]) + (
                            points_low[q, axis] - points_low[p, axis]
                        )
                        r = (
                            (points[vertex, axis] - points[p, axis])
                            + (points_low[vertex, axis] - points_low[p, axis])
                        ) + sign * gap[out, 0] * normal[out, axis]
                        numerator = numerator + (r * e)
                        denominator = denominator + (e * e)
                    if denominator > 1e-24:
                        u = numerator / denominator
                        if u >= -2e-6 and u <= 1.000002:
                            u = tx.min(1.0, tx.max(0.0, u))
                            selected[lane, other, 0] = p
                            selected[lane, other, 1] = q
                            selected_weights[lane, other, 0] = 1.0 - u
                            selected_weights[lane, other, 1] = u
                            found = 1
            receipt[out, 0] = 1
            if unique == 1:
                receipt[out, 0] = -1
                if found == 1:
                    closure = 0.0
                    for side in range(2):
                        pivot = selected[lane, side, 0]
                        for axis in range(3):
                            value = points[pivot, axis]
                            low = points_low[pivot, axis]
                            for k in range(1, 4):
                                vertex = selected[lane, side, k]
                                if vertex >= 0:
                                    delta = tx.sub_rn(points[vertex, axis], points[pivot, axis])
                                    delta_low = (
                                        sum_error(points[vertex, axis], -points[pivot, axis], delta)
                                        + points_low[vertex, axis]
                                        - points_low[pivot, axis]
                                    )
                                    weight = selected_weights[lane, side, k]
                                    prod = tx.mul_rn(weight, delta)
                                    pe = tx.fma(weight, delta, -prod) + weight * delta_low
                                    total = tx.add_rn(value, prod)
                                    low = low + sum_error(value, prod, total) + pe
                                    value = total
                            total = tx.add_rn(value, low)
                            vectors[lane, side, axis] = total
                            surface_low[out, side, axis] = sum_error(value, low, total)
                    for axis in range(3):
                        d = (
                            vectors[lane, 1, axis]
                            - vectors[lane, 0, axis]
                            - gap[out, 0] * normal[out, axis]
                        )
                        closure = closure + (d * d)
                    if (
                        tx.isfinite(closure)
                        and closure * frames[lane, 3] * frames[lane, 3] <= 2.5e-11
                    ):
                        receipt[out, 0] = 2
                        for side in range(2):
                            used = 0
                            for k in range(4):
                                vertex = selected[lane, side, k]
                                material[out, side, k] = -1
                                if vertex >= 0:
                                    used = used + (1)
                                    material[out, side, k] = ids[vertex]
                                weights[out, side, k] = selected_weights[lane, side, k]
                            meta[out, 10 + side] = used
                            for axis in range(3):
                                surface[out, side, axis] = vectors[lane, side, axis]
                                skin[out, side, axis] = 0.0
                if receipt[out, 0] == -1:
                    meta[out, 9] = 0
