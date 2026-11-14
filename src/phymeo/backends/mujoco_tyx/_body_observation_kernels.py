"""Pack shared native FK results into padded, phase-bound body observations."""

import tyx as tx


def pack_bodies(
    count: tx.i32,
    nbody: tx.i32,
    width: tx.i32,
    rows: tx.ndarray(tx.i32, 1),
    motion: tx.ndarray(tx.f32, 4),
    rotation: tx.ndarray(tx.f32, 4),
    position_out: tx.ndarray(tx.f32, 3),
    rotation_out: tx.ndarray(tx.f32, 4),
    linear_out: tx.ndarray(tx.f32, 3),
    angular_out: tx.ndarray(tx.f32, 3),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        row = rows[w]
        for b in range(width):
            for i in range(3):
                p = 0.0
                linear = 0.0
                angular = 0.0
                if b < nbody:
                    p = motion[w, b, 0, i]
                    linear = motion[w, b, 1, i]
                    angular = motion[w, b, 2, i]
                if not tx.isfinite(p) or not tx.isfinite(linear) or not tx.isfinite(angular):
                    status[w] = 4
                position_out[row, b, i] = p
                linear_out[row, b, i] = linear
                angular_out[row, b, i] = angular
                for j in range(3):
                    value = 0.0
                    if b < nbody:
                        value = rotation[w, b, i, j]
                    if not tx.isfinite(value):
                        status[w] = 4
                    rotation_out[row, b, i, j] = value


pack_bodies = tx.kernel(pack_bodies)
