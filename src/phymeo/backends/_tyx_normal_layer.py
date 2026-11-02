"""Per-site normal load. One output row has one owner on both tyx targets."""

import tyx as tx


@tx.kernel
def normal_layer_rows(count: tx.i32, sites: tx.ndarray(tx.f32, 2), out: tx.ndarray(tx.f32, 2)):
    for i in tx.range(count):
        rate = sites[i, 0] * sites[i, 3] + sites[i, 1] * sites[i, 4] + sites[i, 2] * sites[i, 5]
        compression = tx.max(0.0, sites[i, 6] - sites[i, 7])
        damping = tx.max(0.0, 1.0 - sites[i, 10] * rate)
        magnitude = sites[i, 8] * compression * sites[i, 9] * damping
        out[i, 0] = rate
        out[i, 1] = magnitude
        for j in tx.static_range(3):
            out[i, 2 + j] = magnitude * sites[i, j]
        if sites[i, 11] > 0.0:
            vx = sites[i, 3] - rate * sites[i, 0]
            vy = sites[i, 4] - rate * sites[i, 1]
            vz = sites[i, 5] - rate * sites[i, 2]
            # Normalize before squaring, including the regularization scale.
            scale = tx.max(sites[i, 12], tx.max(tx.abs(vx), tx.max(tx.abs(vy), tx.abs(vz))))
            x = vx / scale
            y = vy / scale
            z = vz / scale
            factor = (
                sites[i, 11]
                * magnitude
                / tx.max(sites[i, 12] / scale, tx.sqrt(x * x + y * y + z * z))
            )
            out[i, 2] = out[i, 2] - factor * x
            out[i, 3] = out[i, 3] - factor * y
            out[i, 4] = out[i, 4] - factor * z
