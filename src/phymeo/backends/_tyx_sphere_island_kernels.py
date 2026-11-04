"""Discrete end-step geometry and frictionless, mass-weighted sphere island constraints."""

import tyx as tx

from ._tyx_numeric import _add, _mul


@tx.func
def _length(x: tx.f32, y: tx.f32, z: tx.f32) -> tx.f32:
    return tx.sqrt(x * x + y * y + z * z)


@tx.kernel
def free_motion(
    count: tx.i32,
    h: tx.f32,
    props: tx.ndarray(tx.f32, 2),
    initial: tx.ndarray(tx.f32, 2),
    forces: tx.ndarray(tx.f32, 2),
    staged: tx.ndarray(tx.f32, 2),
):
    for i in tx.range(count):
        for k in tx.static_range(3):
            accel = _add(props[i, 3 + k], _mul(props[i, 0], forces[i, k]))
            v = _add(initial[i, 3 + k], _mul(h, accel))
            staged[i, 3 + k] = v
            staged[i, k] = _add(initial[i, k], _mul(h, v))


@tx.kernel
def candidate_rows(
    count: tx.i32,
    sides: tx.ndarray(tx.i32, 2),
    props: tx.ndarray(tx.f32, 2),
    initial: tx.ndarray(tx.f32, 2),
    free: tx.ndarray(tx.f32, 2),
    rows: tx.ndarray(tx.f32, 2),
    issues: tx.ndarray(tx.i32, 1),
):
    for p in tx.range(count):
        a = sides[p, 0]
        b = sides[p, 1]
        issue = 0
        nx = 0.0
        ny = 0.0
        nz = 1.0
        support = props[b, 6]
        g0 = initial[b, 2] - support
        g1 = free[b, 2] - support
        if a >= 0:
            x0 = initial[b, 0] - initial[a, 0]
            y0 = initial[b, 1] - initial[a, 1]
            z0 = initial[b, 2] - initial[a, 2]
            x1 = free[b, 0] - free[a, 0]
            y1 = free[b, 1] - free[a, 1]
            z1 = free[b, 2] - free[a, 2]
            l0 = _length(x0, y0, z0)
            l1 = _length(x1, y1, z1)
            support = _add(props[a, 2], props[b, 2])
            g0 = l0 - support
            g1 = l1 - support
            if not tx.isfinite(l0) or not tx.isfinite(l1) or not tx.isfinite(support):
                issue = 6
            elif l0 == 0.0 or l1 == 0.0:
                issue = 2
            else:
                nx = x1 / l1
                ny = y1 / l1
                nz = z1 / l1
                dx = x1 - x0
                dy = y1 - y0
                dz = z1 - z0
                den = dx * dx + dy * dy + dz * dz
                along = x0 * dx + y0 * dy + z0 * dz
                facing = x0 * x1 + y0 * y1 + z0 * z1
                if not tx.isfinite(den) or not tx.isfinite(along) or not tx.isfinite(facing):
                    issue = 6
                elif den > 0.0 and along < 0.0 and -along < den:
                    t = -along / den
                    closest = _length(x0 + t * dx, y0 + t * dy, z0 + t * dz)
                    # End-step constraints cannot resolve a pass-through or a flipped normal.
                    # An FP32 guard around the segment minimum also rejects uncertain grazes.
                    scale = tx.max(l0, tx.max(l1, support))
                    if closest <= support + 0.000002 * scale:
                        if (g0 > 0.0 and g1 > 0.0) or facing <= 0.0:
                            issue = 3
        rows[p, 0] = nx
        rows[p, 1] = ny
        rows[p, 2] = nz
        rows[p, 3] = support
        rows[p, 4] = 1.0 if g1 <= 0.0 else 0.0
        rows[p, 5] = 0.0
        rows[p, 6] = 0.0
        rows[p, 7] = g1
        rows[p, 8] = g0
        issues[p] = issue


@tx.kernel
def solve_worlds(
    count: tx.i32,
    sweeps: tx.i32,
    pos_tol: tx.f32,
    vel_tol: tx.f32,
    bodies: tx.ndarray(tx.i32, 1),
    offsets: tx.ndarray(tx.i32, 1),
    sides: tx.ndarray(tx.i32, 2),
    props: tx.ndarray(tx.f32, 2),
    restitution: tx.ndarray(tx.f32, 1),
    capacity: tx.ndarray(tx.i32, 1),
    original: tx.ndarray(tx.f32, 2),
    staged: tx.ndarray(tx.f32, 2),
    rows: tx.ndarray(tx.f32, 2),
    issues: tx.ndarray(tx.i32, 1),
    status: tx.ndarray(tx.i32, 2),
    residual: tx.ndarray(tx.f32, 2),
):
    # A world is one independent GPU lane. Inner loops visit its canonical local pairs.
    for world in tx.range(count):
        begin = offsets[world]
        end = offsets[world + 1]
        code = 0
        required = 0
        pos_error = 0.0
        vel_error = 0.0
        for p in range(begin, end):
            if issues[p] != 0 and code == 0:
                code = issues[p]
        if code == 0:
            # Separate predicted centers against the local supporting half-spaces.
            # This correction changes positions only. Rows that become closed during
            # another pair's correction are retained for the final true-surface audit.
            for iteration in range(sweeps):
                for p in range(begin, end):
                    a = sides[p, 0]
                    b = sides[p, 1]
                    inverse = props[b, 0]
                    gap = -rows[p, 3]
                    if a >= 0:
                        inverse = inverse + props[a, 0]
                    for k in tx.static_range(3):
                        coord = staged[b, k]
                        if a >= 0:
                            coord = coord - staged[a, k]
                        gap = gap + rows[p, k] * coord
                    if gap <= 0.0:
                        rows[p, 4] = 1.0
                    if gap < 0.0:
                        delta = -gap / inverse
                        for k in tx.static_range(3):
                            push = delta * rows[p, k]
                            staged[b, k] = staged[b, k] + props[b, 0] * push
                            if a >= 0:
                                staged[a, k] = staged[a, k] - props[a, 0] * push
            # Refresh directions on the actual projected spheres, even when the starting
            # tangent planes were oblique. An unsupported changed contact chart rejects.
            for p in range(begin, end):
                a = sides[p, 0]
                b = sides[p, 1]
                gap = staged[b, 2] - props[b, 6]
                if a >= 0:
                    x = staged[b, 0] - staged[a, 0]
                    y = staged[b, 1] - staged[a, 1]
                    z = staged[b, 2] - staged[a, 2]
                    length = _length(x, y, z)
                    if length == 0.0 or not tx.isfinite(length):
                        code = 2
                    else:
                        rows[p, 0] = x / length
                        rows[p, 1] = y / length
                        rows[p, 2] = z / length
                    gap = length - rows[p, 3]
                if not tx.isfinite(gap):
                    code = 6
                pos_error = tx.max(pos_error, -gap)
                if rows[p, 4] > 0.0:
                    required = required + 1
                    if gap > pos_tol:
                        code = 7
                    free_rate = 0.0
                    old_rate = 0.0
                    for k in tx.static_range(3):
                        u = staged[b, k + 3]
                        u0 = original[b, k + 3]
                        if a >= 0:
                            u = u - staged[a, k + 3]
                            u0 = u0 - original[a, k + 3]
                        free_rate = free_rate + rows[p, k] * u
                        old_rate = old_rate + rows[p, k] * u0
                    if -free_rate > 2.0 * tx.abs(free_rate - old_rate):
                        rows[p, 6] = _mul(-free_rate, restitution[world])
            if required > capacity[world]:
                code = 1
            elif code == 0 and pos_error > pos_tol:
                code = 4
        if code == 0:
            for iteration in range(sweeps):
                for p in range(begin, end):
                    if rows[p, 4] > 0.0:
                        a = sides[p, 0]
                        b = sides[p, 1]
                        inverse = props[b, 0]
                        if a >= 0:
                            inverse = inverse + props[a, 0]
                        speed = 0.0
                        for k in tx.static_range(3):
                            u = staged[b, k + 3]
                            if a >= 0:
                                u = u - staged[a, k + 3]
                            speed = speed + rows[p, k] * u
                        old = rows[p, 5]
                        total = tx.max(0.0, old + (rows[p, 6] - speed) / inverse)
                        rows[p, 5] = total
                        delta = total - old
                        for k in tx.static_range(3):
                            push = rows[p, k] * delta
                            staged[b, k + 3] = staged[b, k + 3] + props[b, 0] * push
                            if a >= 0:
                                staged[a, k + 3] = staged[a, k + 3] - props[a, 0] * push
            for p in range(begin, end):
                if rows[p, 4] > 0.0:
                    a = sides[p, 0]
                    b = sides[p, 1]
                    inverse = props[b, 0]
                    if a >= 0:
                        inverse = inverse + props[a, 0]
                    speed = 0.0
                    for k in tx.static_range(3):
                        u = staged[b, k + 3]
                        if a >= 0:
                            u = u - staged[a, k + 3]
                        speed = speed + rows[p, k] * u
                    lam = rows[p, 5]
                    projected = tx.max(0.0, lam + (rows[p, 6] - speed) / inverse)
                    err = tx.abs(lam - projected) * inverse
                    vel_error = tx.max(vel_error, err)
                    if not tx.isfinite(lam) or not tx.isfinite(err) or not tx.isfinite(rows[p, 6]):
                        code = 6
            if code == 0 and vel_error > vel_tol:
                code = 5
        for b in range(bodies[world], bodies[world + 1]):
            for k in tx.static_range(6):
                if not tx.isfinite(staged[b, k]):
                    code = 6
        status[world, 0] = code
        status[world, 1] = required
        residual[world, 0] = pos_error
        residual[world, 1] = vel_error
