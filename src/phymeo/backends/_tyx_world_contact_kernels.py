"""World-local material velocities, transposed contact loads, and midpoint increments."""

import tyx as tx


@tx.kernel
def material_velocities(
    count: tx.i32,
    geometry: tx.ndarray(tx.f32, 2),
    velocity: tx.ndarray(tx.f32, 3),
    axis: tx.ndarray(tx.f32, 2),
    rows: tx.ndarray(tx.f32, 2),
):
    for s in tx.range(count):
        world = s // 2
        for col in tx.static_range(13):
            rows[s, col] = geometry[s, col]
        for j in tx.static_range(3):
            v = -axis[world, j] * velocity[world, 3, 0]
            for node in tx.static_range(3):
                v = v + geometry[s, 3 + node] * velocity[world, node, j]
            rows[s, 3 + j] = v


@tx.kernel
def map_and_advance(
    count: tx.i32,
    stage: tx.i32,
    h: tx.f32,
    axis: tx.ndarray(tx.f32, 2),
    inverse_mass: tx.ndarray(tx.f32, 2),
    geometry: tx.ndarray(tx.f32, 2),
    forces: tx.ndarray(tx.f32, 2),
    base_vel: tx.ndarray(tx.f32, 3),
    other_forces: tx.ndarray(tx.f32, 3),
    enabled: tx.ndarray(tx.i32, 1),
    mapped: tx.ndarray(tx.f32, 3),
    mid_velocity: tx.ndarray(tx.f32, 3),
    dx: tx.ndarray(tx.f32, 3),
    dv: tx.ndarray(tx.f32, 3),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        valid = enabled[w]
        if stage == 2:
            valid = status[w]
        for node in tx.static_range(4):
            for k in tx.static_range(3):
                f = 0.0
                first = 2 * w
                second = first + 1
                if node < 3:
                    wa = geometry[first, 3 + node]
                    wb = geometry[second, 3 + node]
                    fa = forces[first, 2 + k]
                    fb = forces[second, 2 + k]
                    f = wa * fa
                    if geometry[second, 9] > 0.0:
                        # Equivalent two-site transpose. Accumulating the common force
                        # on the sum of weights preserves ties under fused multiply-add.
                        mean_force = 0.5 * fa + 0.5 * fb
                        half_difference = 0.5 * fa - 0.5 * fb
                        f = (wa + wb) * mean_force + (wa - wb) * half_difference
                elif k == 0:
                    for a in tx.static_range(3):
                        f = f - axis[w, a] * (forces[first, 2 + a] + forces[second, 2 + a])
                mapped[w, node, k] = f
                velocity = base_vel[w, node, k]
                acceleration = inverse_mass[w, node] * (f + other_forces[w, node, k])
                if stage == 1:
                    mid_velocity[w, node, k] = velocity + 0.5 * h * acceleration
                    if not tx.isfinite(mid_velocity[w, node, k]):
                        valid = 0
                elif stage == 2:
                    change = h * acceleration
                    dv[w, node, k] = change
                    dx[w, node, k] = h * (velocity + 0.5 * change)
                    if (not tx.isfinite(change)) or (not tx.isfinite(dx[w, node, k])):
                        valid = 0
                if not tx.isfinite(f):
                    valid = 0
        status[w] = valid
