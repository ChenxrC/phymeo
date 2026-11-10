"""Transport-only padding and finite guards; no dynamics or action clipping."""
import tyx as tx


def pack(count: tx.i32, nu: tx.i32, nv: tx.i32, nbody: tx.i32, width: tx.i32,
         controls: tx.ndarray(tx.f32, 2), forces: tx.ndarray(tx.f32, 2),
         body_force: tx.ndarray(tx.f32, 3), padded_force: tx.ndarray(tx.f32, 2),
         status: tx.ndarray(tx.i32, 1)):
    for w in tx.range(count):
        status[w] = 0
        for i in range(nu):
            if not tx.isfinite(controls[w, i]):
                status[w] = 1
        for i in range(width):
            padded_force[w, i] = 0.0
            if i < nv:
                value = forces[w, i]
                padded_force[w, i] = value
                if not tx.isfinite(value):
                    status[w] = 1
        for b in range(nbody):
            for i in range(6):
                if not tx.isfinite(body_force[w, b, i]):
                    status[w] = 1


pack = tx.kernel(pack)
