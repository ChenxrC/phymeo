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
