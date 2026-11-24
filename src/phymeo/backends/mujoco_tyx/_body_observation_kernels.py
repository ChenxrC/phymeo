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


def pack_body_transforms(
    count: tx.i32,
    offsets: tx.ndarray(tx.i32, 1),
    position: tx.ndarray(tx.f32, 3),
    rotation: tx.ndarray(tx.f32, 4),
    transforms: tx.ndarray(tx.f32, 3),
):
    for world in tx.range(count):
        start = offsets[world]
        for body in range(offsets[world + 1] - start):
            for i in range(3):
                for j in range(3):
                    transforms[start + body, i, j] = rotation[world, body, i, j]
                transforms[start + body, i, 3] = position[world, body, i]


pack_body_transforms = tx.kernel(pack_body_transforms)
