"""Bounded finite-feature refinement of small convex mesh pairs, on device.

Every vertex triple/edge is retained. No simplified hull or CPU fallback.
Only uncertain small hulls enter this path; normal GJK/EPA stays the fast path.
"""

import tyx as tx

from .._tyx_convex_support import clamp01, difference_of_products, dot3


@tx.kernel
def refine(
    count: tx.i32,
    first: tx.i32,
    tolerance_m: tx.f32,
    tags: tx.ndarray(tx.i32, 2),
    points: tx.ndarray(tx.f32, 2),
    ids: tx.ndarray(tx.i32, 1),
    frames: tx.ndarray(tx.f32, 2),
    meta: tx.ndarray(tx.i32, 2),
    gap: tx.ndarray(tx.f32, 2),
    normal: tx.ndarray(tx.f32, 2),
    surface: tx.ndarray(tx.f32, 3),
    material: tx.ndarray(tx.i32, 3),
    weights: tx.ndarray(tx.f32, 3),
    skin: tx.ndarray(tx.f32, 3),
):
    for lane in tx.range(count):
        out = first + lane
        a0 = tags[2 * lane, 1]
        a1 = tags[2 * lane, 2]
        b0 = tags[2 * lane + 1, 1]
        b1 = tags[2 * lane + 1, 2]
        tol = tolerance_m / frames[lane, 3]
        if meta[out, 0] != 1 and a1 - a0 <= 16 and b1 - b0 <= 16:
            best_gap = -3.402823e38
            nx = 0.0
            ny = 0.0
            nz = 0.0
            axis_count = 0
            # Complete convex SAT axes: every triangle normal and every edge
            # cross product. Interior triangles/edges cannot invalidate a bound.
            for family in range(3):
                start = a0
                end = a1
                if family == 1:
                    start = b0
                    end = b1
                for i in range(start, end):
                    for j in range(i + 1, end):
                        kstart = j + 1
                        kend = end
                        if family == 2:
                            kstart = b0
                            kend = b1
                        for k in range(kstart, kend):
                            lstart = k
                            lend = k + 1
                            if family == 2:
                                lstart = k + 1
                                lend = b1
                            for last in range(lstart, lend):
                                ex = points[j, 0] - points[i, 0]
                                ey = points[j, 1] - points[i, 1]
                                ez = points[j, 2] - points[i, 2]
                                fx = points[k, 0] - points[i, 0]
                                fy = points[k, 1] - points[i, 1]
                                fz = points[k, 2] - points[i, 2]
                                if family == 2:
                                    fx = points[last, 0] - points[k, 0]
                                    fy = points[last, 1] - points[k, 1]
                                    fz = points[last, 2] - points[k, 2]
                                x = difference_of_products(ey, fz, ez, fy)
                                y = difference_of_products(ez, fx, ex, fz)
                                z = difference_of_products(ex, fy, ey, fx)
                                length = tx.sqrt(dot3(x, y, z, x, y, z))
                                if length > 1e-15:
                                    x = x / length
                                    y = y / length
                                    z = z / length
                                    amin = 3.402823e38
                                    amax = -3.402823e38
                                    bmin = 3.402823e38
                                    bmax = -3.402823e38
                                    for v in range(a0, a1):
                                        value = dot3(
                                            points[v, 0], points[v, 1], points[v, 2], x, y, z
                                        )
                                        amin = tx.min(amin, value)
                                        amax = tx.max(amax, value)
                                    for v in range(b0, b1):
                                        value = dot3(
                                            points[v, 0], points[v, 1], points[v, 2], x, y, z
                                        )
                                        bmin = tx.min(bmin, value)
                                        bmax = tx.max(bmax, value)
                                    candidate = bmin - amax
                                    if amin - bmax > candidate:
                                        candidate = amin - bmax
                                        x = -x
                                        y = -y
                                        z = -z
                                    if candidate > best_gap:
                                        best_gap = candidate
                                        nx = x
                                        ny = y
                                        nz = z
                                    axis_count = axis_count + 1
            shift = tx.max(0.0, -best_gap)
            sx = shift * nx
            sy = shift * ny
            sz = shift * nz
            best = 3.402823e38
            ax = 0.0
            ay = 0.0
            az = 0.0
            bx = 0.0
            by = 0.0
            bz = 0.0
            for side in tx.static_range(2):
                for field in tx.static_range(4):
                    material[out, side, field] = -1
                    weights[out, side, field] = 0.0
            # Vertex-to-triangle candidates on both sides, in the separated
            # configuration (translate B by its SAT escape vector if overlapping).
            for side in range(2):
                v0 = a0
                v1 = a1
                f0 = b0
                f1 = b1
                sign = 1.0
                if side == 1:
                    v0 = b0
                    v1 = b1
                    f0 = a0
                    f1 = a1
                    sign = -1.0
                for v in range(v0, v1):
                    for i in range(f0, f1):
                        for j in range(i + 1, f1):
                            for k in range(j + 1, f1):
                                ex = points[j, 0] - points[i, 0]
                                ey = points[j, 1] - points[i, 1]
                                ez = points[j, 2] - points[i, 2]
                                fx = points[k, 0] - points[i, 0]
                                fy = points[k, 1] - points[i, 1]
                                fz = points[k, 2] - points[i, 2]
                                x = difference_of_products(ey, fz, ez, fy)
                                y = difference_of_products(ez, fx, ex, fz)
                                z = difference_of_products(ex, fy, ey, fx)
                                area = dot3(x, y, z, x, y, z)
                                if area > 1e-30:
                                    rx = points[v, 0] - points[i, 0] - sign * sx
                                    ry = points[v, 1] - points[i, 1] - sign * sy
                                    rz = points[v, 2] - points[i, 2] - sign * sz
                                    u = (
                                        dot3(
                                            difference_of_products(ry, fz, rz, fy),
                                            difference_of_products(rz, fx, rx, fz),
                                            difference_of_products(rx, fy, ry, fx),
                                            x,
                                            y,
                                            z,
                                        )
                                        / area
                                    )
                                    t = (
                                        dot3(
                                            difference_of_products(ey, rz, ez, ry),
                                            difference_of_products(ez, rx, ex, rz),
                                            difference_of_products(ex, ry, ey, rx),
                                            x,
                                            y,
                                            z,
                                        )
                                        / area
                                    )
                                    if u >= 0.0 and t >= 0.0 and u + t <= 1.0:
                                        px = tx.fma(t, fx, tx.fma(u, ex, points[i, 0]))
                                        py = tx.fma(t, fy, tx.fma(u, ey, points[i, 1]))
                                        pz = tx.fma(t, fz, tx.fma(u, ez, points[i, 2]))
                                        dx = points[v, 0] - px - sign * sx
                                        dy = points[v, 1] - py - sign * sy
                                        dz = points[v, 2] - pz - sign * sz
                                        square = dot3(dx, dy, dz, dx, dy, dz)
                                        if square < best:
                                            best = square
                                            ax = points[v, 0]
                                            ay = points[v, 1]
                                            az = points[v, 2]
                                            bx = px
                                            by = py
                                            bz = pz
                                            if side == 1:
                                                ax = px
                                                ay = py
                                                az = pz
                                                bx = points[v, 0]
                                                by = points[v, 1]
                                                bz = points[v, 2]
                                            for s in tx.static_range(2):
                                                for f in tx.static_range(4):
                                                    material[out, s, f] = -1
                                                    weights[out, s, f] = 0.0
                                            material[out, side, 0] = ids[v]
                                            weights[out, side, 0] = 1.0
                                            material[out, 1 - side, 0] = ids[i]
                                            material[out, 1 - side, 1] = ids[j]
                                            material[out, 1 - side, 2] = ids[k]
                                            weights[out, 1 - side, 0] = 1.0 - u - t
                                            weights[out, 1 - side, 1] = u
                                            weights[out, 1 - side, 2] = t
            # Edge-to-edge candidates include all endpoint boundary cases.
            for i in range(a0, a1):
                for j in range(i + 1, a1):
                    for k in range(b0, b1):
                        for last in range(k + 1, b1):
                            ex = points[j, 0] - points[i, 0]
                            ey = points[j, 1] - points[i, 1]
                            ez = points[j, 2] - points[i, 2]
                            fx = points[last, 0] - points[k, 0]
                            fy = points[last, 1] - points[k, 1]
                            fz = points[last, 2] - points[k, 2]
                            rx = points[i, 0] - points[k, 0] - sx
                            ry = points[i, 1] - points[k, 1] - sy
                            rz = points[i, 2] - points[k, 2] - sz
                            ee = dot3(ex, ey, ez, ex, ey, ez)
                            ff = dot3(fx, fy, fz, fx, fy, fz)
                            ef = dot3(ex, ey, ez, fx, fy, fz)
                            er = dot3(ex, ey, ez, rx, ry, rz)
                            fr = dot3(fx, fy, fz, rx, ry, rz)
                            if ee > 1e-30 and ff > 1e-30:
                                x = difference_of_products(ey, fz, ez, fy)
                                y = difference_of_products(ez, fx, ex, fz)
                                z = difference_of_products(ex, fy, ey, fx)
                                denominator = dot3(x, y, z, x, y, z)
                                u = clamp01(-er / ee)
                                if denominator > 1e-20 * ee * ff:
                                    u = clamp01(
                                        difference_of_products(ef, fr, er, ff) / denominator
                                    )
                                t = (ef * u + fr) / ff
                                if t < 0.0:
                                    t = 0.0
                                    u = clamp01(-er / ee)
                                elif t > 1.0:
                                    t = 1.0
                                    u = clamp01((ef - er) / ee)
                                pax = tx.fma(u, ex, points[i, 0])
                                pay = tx.fma(u, ey, points[i, 1])
                                paz = tx.fma(u, ez, points[i, 2])
                                pbx = tx.fma(t, fx, points[k, 0])
                                pby = tx.fma(t, fy, points[k, 1])
                                pbz = tx.fma(t, fz, points[k, 2])
                                dx = pbx + sx - pax
                                dy = pby + sy - pay
                                dz = pbz + sz - paz
                                square = dot3(dx, dy, dz, dx, dy, dz)
                                if square < best:
                                    best = square
                                    ax = pax
                                    ay = pay
                                    az = paz
                                    bx = pbx
                                    by = pby
                                    bz = pbz
                                    for s in tx.static_range(2):
                                        for f in tx.static_range(4):
                                            material[out, s, f] = -1
                                            weights[out, s, f] = 0.0
                                    material[out, 0, 0] = ids[i]
                                    material[out, 0, 1] = ids[j]
                                    material[out, 1, 0] = ids[k]
                                    material[out, 1, 1] = ids[last]
                                    weights[out, 0, 0] = 1.0 - u
                                    weights[out, 0, 1] = u
                                    weights[out, 1, 0] = 1.0 - t
                                    weights[out, 1, 1] = t
            distance = tx.sqrt(best)
            value = best_gap
            if best_gap > 0.0 and distance > tol:
                nx = (bx - ax) / distance
                ny = (by - ay) / distance
                nz = (bz - az) / distance
                value = distance
            low = 3.402823e38
            high = -3.402823e38
            for v in range(a0, a1):
                high = tx.max(high, dot3(points[v, 0], points[v, 1], points[v, 2], nx, ny, nz))
            for v in range(b0, b1):
                low = tx.min(low, dot3(points[v, 0], points[v, 1], points[v, 2], nx, ny, nz))
            plane = low - high
            dx = bx - ax - value * nx
            dy = by - ay - value * ny
            dz = bz - az - value * nz
            closure = tx.sqrt(dot3(dx, dy, dz, dx, dy, dz))
            meta[out, 14] = axis_count
            if axis_count > 0 and tx.abs(plane - value) <= tol and closure <= tol:
                meta[out, 0] = 1
                meta[out, 1] = 3
                if value > tol:
                    meta[out, 1] = 1
                elif value < -tol:
                    meta[out, 1] = 2
                meta[out, 2] = 0
                meta[out, 3] = 7
                meta[out, 9] = 7
                gap[out, 0] = value
                gap[out, 1] = tx.min(plane, value)
                gap[out, 2] = tx.max(plane, value)
                normal[out, 0] = nx
                normal[out, 1] = ny
                normal[out, 2] = nz
                surface[out, 0, 0] = ax
                surface[out, 0, 1] = ay
                surface[out, 0, 2] = az
                surface[out, 1, 0] = bx
                surface[out, 1, 1] = by
                surface[out, 1, 2] = bz
                for s in tx.static_range(2):
                    used = 0
                    for f in tx.static_range(4):
                        if material[out, s, f] >= 0:
                            used = used + 1
                    meta[out, 10 + s] = used
                    for axis in tx.static_range(3):
                        skin[out, s, axis] = 0.0
            else:
                # No output with failed bounds is eligible for force response.
                meta[out, 9] = 0
