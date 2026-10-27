"""Device contact sweeps: one serial worker for all pairs sharing rigid bodies."""

import tyx as tx

from ._tyx_numeric import _add, _div, _mul


@tx.kernel
def project_contacts(
    particle_count: tx.i32,
    body_count: tx.i32,
    particle: tx.ndarray(tx.f32, 2),
    body: tx.ndarray(tx.f32, 2),
    wp: tx.ndarray(tx.f32, 1),
    wb: tx.ndarray(tx.f32, 1),
    radii: tx.ndarray(tx.f32, 1),
    multipliers: tx.ndarray(tx.f32, 1),
    loads: tx.ndarray(tx.f32, 2),
    failed: tx.ndarray(tx.i32, 1),
):
    # Only one CUDA worker enters this loop. Each pair reads the live body pose
    # left by the preceding pair, preserving deterministic Gauss-Seidel order.
    for worker in tx.range(1):
        for k in range(particle_count * body_count):
            i = k // body_count
            j = k % body_count
            for axis in tx.static_range(3):
                loads[k, axis] = 0.0
            dx = _add(particle[i, 0], -body[j, 0])
            dy = _add(particle[i, 1], -body[j, 1])
            dz = _add(particle[i, 2], -body[j, 2])
            if not tx.isfinite(dx) or not tx.isfinite(dy) or not tx.isfinite(dz):
                failed[k] = 1
            scale = tx.max(tx.abs(dx), tx.max(tx.abs(dy), tx.abs(dz)))
            distance = 0.0
            nx = 1.0
            ny = 0.0
            nz = 0.0
            if scale > 0.0:
                sx = _div(dx, scale)
                sy = _div(dy, scale)
                sz = _div(dz, scale)
                norm = tx.sqrt(_add(_add(_mul(sx, sx), _mul(sy, sy)), _mul(sz, sz)))
                distance = _mul(scale, norm)
                nx = _div(sx, norm)
                ny = _div(sy, norm)
                nz = _div(sz, norm)
            gap = _add(distance, -radii[j])
            weight = _add(wp[i], wb[j])
            if not tx.isfinite(gap) or not tx.isfinite(weight):
                failed[k] = 1
            elif weight > 0.0:
                proposal = _add(multipliers[k], -_div(gap, weight))
                if not tx.isfinite(proposal):
                    failed[k] = 1
                next_lambda = tx.max(0.0, proposal)
                change = _add(next_lambda, -multipliers[k])
                lx = _mul(change, nx)
                ly = _mul(change, ny)
                lz = _mul(change, nz)
                multipliers[k] = next_lambda
                loads[k, 0] = lx
                loads[k, 1] = ly
                loads[k, 2] = lz
                for axis in tx.static_range(3):
                    particle[i, axis] = _add(particle[i, axis], _mul(wp[i], loads[k, axis]))
                    body[j, axis] = _add(body[j, axis], -_mul(wb[j], loads[k, axis]))
                    if (
                        not tx.isfinite(particle[i, axis])
                        or not tx.isfinite(body[j, axis])
                        or not tx.isfinite(loads[k, axis])
                    ):
                        failed[k] = 1
                if not tx.isfinite(next_lambda):
                    failed[k] = 1


@tx.kernel
def contact_geometry(
    particle_count: tx.i32,
    body_count: tx.i32,
    particle: tx.ndarray(tx.f32, 2),
    body: tx.ndarray(tx.f32, 2),
    radii: tx.ndarray(tx.f32, 1),
    normals: tx.ndarray(tx.f32, 2),
    gaps: tx.ndarray(tx.f32, 1),
    failed: tx.ndarray(tx.i32, 1),
):
    # Geometry has no shared writers and is frozen throughout velocity sweeps.
    for k in tx.range(particle_count * body_count):
        i = k // body_count
        j = k % body_count
        dx = _add(particle[i, 0], -body[j, 0])
        dy = _add(particle[i, 1], -body[j, 1])
        dz = _add(particle[i, 2], -body[j, 2])
        if not tx.isfinite(dx) or not tx.isfinite(dy) or not tx.isfinite(dz):
            failed[k] = 1
        scale = tx.max(tx.abs(dx), tx.max(tx.abs(dy), tx.abs(dz)))
        distance = 0.0
        nx = 1.0
        ny = 0.0
        nz = 0.0
        if scale > 0.0:
            sx = _div(dx, scale)
            sy = _div(dy, scale)
            sz = _div(dz, scale)
            norm = tx.sqrt(_add(_add(_mul(sx, sx), _mul(sy, sy)), _mul(sz, sz)))
            distance = _mul(scale, norm)
            nx = _div(sx, norm)
            ny = _div(sy, norm)
            nz = _div(sz, norm)
        normals[k, 0] = nx
        normals[k, 1] = ny
        normals[k, 2] = nz
        gaps[k] = _add(distance, -radii[j])
        if (
            not tx.isfinite(gaps[k])
            or not tx.isfinite(nx)
            or not tx.isfinite(ny)
            or not tx.isfinite(nz)
        ):
            failed[k] = 1


@tx.kernel
def project_contact_velocities(
    particle_count: tx.i32,
    body_count: tx.i32,
    contact_band: tx.f32,
    particle: tx.ndarray(tx.f32, 2),
    body: tx.ndarray(tx.f32, 2),
    wp: tx.ndarray(tx.f32, 1),
    wb: tx.ndarray(tx.f32, 1),
    normals: tx.ndarray(tx.f32, 2),
    gaps: tx.ndarray(tx.f32, 1),
    impulses: tx.ndarray(tx.f32, 1),
    loads: tx.ndarray(tx.f32, 2),
    failed: tx.ndarray(tx.i32, 1),
):
    for worker in tx.range(1):
        for k in range(particle_count * body_count):
            i = k // body_count
            j = k % body_count
            for axis in tx.static_range(3):
                loads[k, axis] = 0.0
            weight = _add(wp[i], wb[j])
            if not tx.isfinite(weight):
                failed[k] = 1
            elif weight > 0.0 and (gaps[k] <= contact_band or impulses[k] > 0.0):
                next_impulse = 0.0
                if gaps[k] <= contact_band:
                    vx = _add(particle[i, 0], -body[j, 0])
                    vy = _add(particle[i, 1], -body[j, 1])
                    vz = _add(particle[i, 2], -body[j, 2])
                    vn = _add(
                        _add(_mul(vx, normals[k, 0]), _mul(vy, normals[k, 1])),
                        _mul(vz, normals[k, 2]),
                    )
                    proposal = _add(impulses[k], -_div(vn, weight))
                    if not tx.isfinite(proposal):
                        failed[k] = 1
                    next_impulse = tx.max(0.0, proposal)
                change = _add(next_impulse, -impulses[k])
                impulses[k] = next_impulse
                for axis in tx.static_range(3):
                    loads[k, axis] = _mul(change, normals[k, axis])
                    particle[i, axis] = _add(particle[i, axis], _mul(wp[i], loads[k, axis]))
                    body[j, axis] = _add(body[j, axis], -_mul(wb[j], loads[k, axis]))
                    if (
                        not tx.isfinite(particle[i, axis])
                        or not tx.isfinite(body[j, axis])
                        or not tx.isfinite(loads[k, axis])
                    ):
                        failed[k] = 1
                if not tx.isfinite(next_impulse):
                    failed[k] = 1


@tx.kernel
def reconstruct_velocity(
    count: tx.i32,
    h: tx.f32,
    weights: tx.ndarray(tx.f32, 1),
    initial: tx.ndarray(tx.f32, 2),
    predicted: tx.ndarray(tx.f32, 2),
    corrected: tx.ndarray(tx.f32, 2),
    velocity: tx.ndarray(tx.f32, 2),
    failed: tx.ndarray(tx.i32, 1),
):
    for i in tx.range(count):
        changed = (
            predicted[i, 0] != corrected[i, 0]
            or predicted[i, 1] != corrected[i, 1]
            or predicted[i, 2] != corrected[i, 2]
        )
        if changed and weights[i] > 0.0:
            for axis in tx.static_range(3):
                velocity[i, axis] = _div(_add(corrected[i, axis], -initial[i, axis]), h)
                if not tx.isfinite(velocity[i, axis]):
                    failed[i] = 1
