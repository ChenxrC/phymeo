"""Imported only when the optional tyx solver is constructed."""

import tyx as tx


@tx.kernel
def integrate_particles(
    n: tx.i32,
    dt: tx.f32,
    position: tx.ndarray(tx.f32, 2),
    velocity: tx.ndarray(tx.f32, 2),
    force: tx.ndarray(tx.f32, 2),
    inverse_mass: tx.ndarray(tx.f32, 1),
    gravity: tx.ndarray(tx.f32, 1),
    next_position: tx.ndarray(tx.f32, 2),
    next_velocity: tx.ndarray(tx.f32, 2),
):
    for i in tx.range(n):
        for axis in tx.static_range(3):
            if inverse_mass[i] > 0.0:
                acceleration = gravity[axis] + inverse_mass[i] * force[i, axis]
                value = velocity[i, axis] + dt * acceleration
                next_velocity[i, axis] = value
                next_position[i, axis] = position[i, axis] + dt * value
            else:
                next_velocity[i, axis] = 0.0
                next_position[i, axis] = position[i, axis]


@tx.kernel
def project_distance_color(
    count: tx.i32,
    offset: tx.i32,
    order: tx.ndarray(tx.i32, 1),
    edges: tx.ndarray(tx.i32, 2),
    rest: tx.ndarray(tx.f32, 1),
    alpha: tx.ndarray(tx.f32, 1),
    inverse_mass: tx.ndarray(tx.f32, 1),
    position: tx.ndarray(tx.f32, 2),
    multipliers: tx.ndarray(tx.f32, 1),
    failed: tx.ndarray(tx.i32, 1),
):
    # The host coloring guarantees that these parallel iterations never share a vertex.
    for item in tx.range(count):
        k = order[offset + item]
        i = edges[k, 0]
        j = edges[k, 1]
        dx = position[i, 0] - position[j, 0]
        dy = position[i, 1] - position[j, 1]
        dz = position[i, 2] - position[j, 2]
        scale = tx.max(tx.abs(dx), tx.max(tx.abs(dy), tx.abs(dz)))
        length = 0.0
        nx = 1.0
        ny = 0.0
        nz = 0.0
        if scale > 0.0:
            sx = dx / scale
            sy = dy / scale
            sz = dz / scale
            unit_length = tx.sqrt(sx * sx + sy * sy + sz * sz)
            length = scale * unit_length
            nx = sx / unit_length
            ny = sy / unit_length
            nz = sz / unit_length
        wi = inverse_mass[i]
        wj = inverse_mass[j]
        denominator = wi + wj + alpha[k]
        if not tx.isfinite(length) or not tx.isfinite(denominator):
            failed[k] = 1
        elif denominator > 0.0:
            delta_lambda = (-(length - rest[k]) - alpha[k] * multipliers[k]) / denominator
            next_lambda = multipliers[k] + delta_lambda
            if not tx.isfinite(delta_lambda) or not tx.isfinite(next_lambda):
                failed[k] = 1
            else:
                multipliers[k] = next_lambda
                for axis in tx.static_range(3):
                    direction = nx
                    if axis == 1:
                        direction = ny
                    elif axis == 2:
                        direction = nz
                    if wi > 0.0:
                        value_i = position[i, axis] + wi * delta_lambda * direction
                        if not tx.isfinite(value_i):
                            failed[k] = 1
                        position[i, axis] = value_i
                    if wj > 0.0:
                        value_j = position[j, axis] - wj * delta_lambda * direction
                        if not tx.isfinite(value_j):
                            failed[k] = 1
                        position[j, axis] = value_j


@tx.kernel
def project_y_plane(
    n: tx.i32,
    height: tx.f32,
    inverse_mass: tx.ndarray(tx.f32, 1),
    position: tx.ndarray(tx.f32, 2),
):
    for i in tx.range(n):
        if inverse_mass[i] > 0.0:
            position[i, 1] = tx.max(position[i, 1], height)


@tx.kernel
def reconstruct_particle_velocity(
    n: tx.i32,
    dt: tx.f32,
    has_plane: tx.i32,
    height: tx.f32,
    inverse_mass: tx.ndarray(tx.f32, 1),
    original: tx.ndarray(tx.f32, 2),
    position: tx.ndarray(tx.f32, 2),
    velocity: tx.ndarray(tx.f32, 2),
):
    for i in tx.range(n):
        for axis in tx.static_range(3):
            value = 0.0
            if inverse_mass[i] > 0.0:
                value = (position[i, axis] - original[i, axis]) / dt
                if has_plane != 0 and axis == 1 and position[i, 1] <= height and tx.isfinite(value):
                    value = tx.max(value, 0.0)
            velocity[i, axis] = value
