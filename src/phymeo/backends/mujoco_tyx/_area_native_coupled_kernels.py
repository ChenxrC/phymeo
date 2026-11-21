"""Frozen-entry area load coupling around the physical-M native working set.

The outer iteration owns no q/clock integration. Its separate diagnostics record
actual working-set calls, including the zero-area baseline, not Newton steps.
"""

import tyx as tx


@tx.kernel
def initialize(
    count: tx.i32,
    nv: tx.i32,
    width: tx.i32,
    packed_width: tx.i32,
    sample_capacity: tx.i32,
    row_capacity: tx.i32,
    h: tx.f32,
    pack_status: tx.ndarray(tx.i32, 1),
    sizes: tx.ndarray(tx.i32, 2),
    counts: tx.ndarray(tx.i32, 2),
    mass: tx.ndarray(tx.f32, 3),
    inputs: tx.ndarray(tx.f32, 3),
    applied: tx.ndarray(tx.f32, 2),
    passive: tx.ndarray(tx.f32, 2),
    bias: tx.ndarray(tx.f32, 2),
    matrices: tx.ndarray(tx.f32, 4),
    vectors: tx.ndarray(tx.f32, 3),
    contact: tx.ndarray(tx.f32, 3),
    old_meta: tx.ndarray(tx.i32, 2),
    old_scalars: tx.ndarray(tx.f32, 2),
    old_trace_meta: tx.ndarray(tx.i32, 3),
    old_trace: tx.ndarray(tx.f32, 3),
    trial_applied: tx.ndarray(tx.f32, 2),
    meta: tx.ndarray(tx.i32, 2),
    trace_meta: tx.ndarray(tx.i32, 3),
    trace: tx.ndarray(tx.f32, 3),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        for i in range(4):
            meta[w, i] = 0
        for i in range(5):
            old_meta[w, i] = 0
        for i in range(3):
            old_scalars[w, i] = 0.0
        for r in range(290):
            for i in range(4):
                old_trace_meta[w, r, i] = 0
            for i in range(3 + 2 * packed_width):
                old_trace[w, r, i] = 0.0
        for r in range(25):
            for i in range(3):
                trace_meta[w, r, i] = 0
            for i in range(14 + 2 * packed_width):
                trace[w, r, i] = 0.0
        for s in range(sample_capacity):
            for i in range(17):
                contact[w, s, i] = 0.0
        for bank in range(4):
            for i in range(packed_width):
                for j in range(packed_width):
                    matrices[w, bank, i, j] = 0.0
        for bank in range(8):
            for i in range(packed_width):
                vectors[w, bank, i] = 0.0
        for i in range(width):
            trial_applied[w, i] = applied[w, i]
        if status[w] == 0:
            if (
                pack_status[w] != 0
                or sizes[w, 0] != nv
                or nv < 6
                or nv > 8
                or nv > packed_width
                or sizes[w, 1] < 0
                or sizes[w, 1] > sample_capacity
                or counts[w, 2] < 0
                or counts[w, 2] > row_capacity
                or counts[w, 2] + sizes[w, 1] > row_capacity
                or not tx.isfinite(h)
                or h <= 0.0
            ):
                status[w] = 8
        # These are the new outer budget and zero line-search count. The old
        # Newton metadata above stays empty because its solver was not called.
        sizes[w, 2] = 24
        sizes[w, 3] = 0
        if status[w] == 0:
            for i in range(nv):
                for j in range(nv):
                    value = mass[w, i, j]
                    if i == j:
                        value = value + h * inputs[w, 0, i]
                    matrices[w, 0, i, j] = value
            for i in range(nv):
                for j in range(i + 1):
                    value = matrices[w, 0, i, j]
                    for k in range(j):
                        value = value - matrices[w, 1, i, k] * matrices[w, 1, j, k]
                    if i == j:
                        if not tx.isfinite(value) or value <= 0.0:
                            status[w] = 4
                            value = 1.0
                        matrices[w, 1, i, j] = tx.sqrt(value)
                    else:
                        matrices[w, 1, i, j] = value / matrices[w, 1, j, j]
                    if not tx.isfinite(matrices[w, 1, i, j]):
                        status[w] = 4
        if status[w] == 0:
            # Preserve A^-1(source applied + passive - bias) before each native
            # solve overwrites its own physical-M smooth-acceleration scratch.
            # Vector bank 0 has this route-specific acceleration meaning.
            for i in range(nv):
                value = applied[w, i] + passive[w, i] - bias[w, i]
                for j in range(i):
                    value = value - matrices[w, 1, i, j] * vectors[w, 6, j]
                vectors[w, 6, i] = value / matrices[w, 1, i, i]
            for reverse in range(nv):
                i = nv - 1 - reverse
                value = vectors[w, 6, i]
                for j in range(i + 1, nv):
                    value = value - matrices[w, 1, j, i] * vectors[w, 0, j]
                vectors[w, 0, i] = value / matrices[w, 1, i, i]
                if not tx.isfinite(vectors[w, 0, i]):
                    status[w] = 2
        meta[w, 0] = status[w]


@tx.kernel
def evaluate(
    count: tx.i32,
    nv: tx.i32,
    packed_width: tx.i32,
    h: tx.f32,
    iteration: tx.i32,
    sizes: tx.ndarray(tx.i32, 2),
    inputs: tx.ndarray(tx.f32, 3),
    samples: tx.ndarray(tx.f32, 3),
    ja: tx.ndarray(tx.f32, 4),
    jb: tx.ndarray(tx.f32, 4),
    matrices: tx.ndarray(tx.f32, 4),
    vectors: tx.ndarray(tx.f32, 3),
    contact: tx.ndarray(tx.f32, 3),
    applied: tx.ndarray(tx.f32, 2),
    passive: tx.ndarray(tx.f32, 2),
    bias: tx.ndarray(tx.f32, 2),
    native_total_applied: tx.ndarray(tx.f32, 2),
    native_diagnostics: tx.ndarray(tx.f32, 2),
    trial_applied: tx.ndarray(tx.f32, 2),
    meta: tx.ndarray(tx.i32, 2),
    trace_meta: tx.ndarray(tx.i32, 3),
    trace: tx.ndarray(tx.f32, 3),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        meta[w, 1] = iteration
        meta[w, 2] = iteration + 1
        trace_meta[w, iteration, 0] = 0
        if iteration > 0:
            trace_meta[w, iteration, 0] = 1
            if trace_meta[w, iteration - 1, 0] >= 2:
                trace_meta[w, iteration, 0] = 2
        trace_meta[w, iteration, 1] = status[w]
        for i in range(10):
            trace[w, iteration, i] = native_diagnostics[w, i]
        if status[w] == 0:
            # This is the same rounded physical RHS used by the native solve.
            # Its A response is mathematically A^-1 M a_phys, without a second
            # rounded M*a multiplication or a q integration inside the loop.
            for i in range(nv):
                value = native_total_applied[w, i] + passive[w, i] - bias[w, i]
                for j in range(i):
                    value = value - matrices[w, 1, i, j] * vectors[w, 6, j]
                vectors[w, 6, i] = value / matrices[w, 1, i, i]
            for reverse in range(nv):
                i = nv - 1 - reverse
                value = vectors[w, 6, i]
                for j in range(i + 1, nv):
                    value = value - matrices[w, 1, j, i] * vectors[w, 2, j]
                vectors[w, 2, i] = value / matrices[w, 1, i, i]
            for i in range(nv):
                value = inputs[w, 1, i] + h * vectors[w, 2, i]
                vectors[w, 2, i] = value
                vectors[w, 1, i] = value
                vectors[w, 5, i] = 0.0
                if not tx.isfinite(value):
                    status[w] = 2
        if status[w] == 0:
            # Same packed-normal projection, regularized tangential law, point
            # force layout and world-origin moment as the live Newton kernel.
            for sample in range(sizes[w, 1]):
                nx = samples[w, sample, 3]
                ny = samples[w, sample, 4]
                nz = samples[w, sample, 5]
                nn = nx * nx + ny * ny + nz * nz
                k = samples[w, sample, 6]
                phi = samples[w, sample, 7]
                mu = samples[w, sample, 8]
                slip = samples[w, sample, 9]
                ux = 0.0
                uy = 0.0
                uz = 0.0
                for j in range(nv):
                    ux = ux + (ja[w, sample, 0, j] - jb[w, sample, 0, j]) * vectors[w, 2, j]
                    uy = uy + (ja[w, sample, 1, j] - jb[w, sample, 1, j]) * vectors[w, 2, j]
                    uz = uz + (ja[w, sample, 2, j] - jb[w, sample, 2, j]) * vectors[w, 2, j]
                vn = nx * ux + ny * uy + nz * uz
                cx = uy * nz - uz * ny
                cy = uz * nx - ux * nz
                cz = ux * ny - uy * nx
                vx = (ny * cz - nz * cy) / nn
                vy = (nz * cx - nx * cz) / nn
                vz = (nx * cy - ny * cx) / nn
                gap = phi + h * vn
                fn = tx.max(0.0, -k * gap)
                speed2 = vx * vx + vy * vy + vz * vz + slip * slip
                if not tx.isfinite(speed2) or speed2 <= 0.0:
                    status[w] = 8
                    speed2 = 1.0
                speed = tx.sqrt(speed2)
                ftx = -mu * fn * (vx / speed)
                fty = -mu * fn * (vy / speed)
                ftz = -mu * fn * (vz / speed)
                fx = fn * nx + ftx
                fy = fn * ny + fty
                fz = fn * nz + ftz
                px = samples[w, sample, 0]
                py = samples[w, sample, 1]
                pz = samples[w, sample, 2]
                contact[w, sample, 0] = fn
                contact[w, sample, 1] = vn
                contact[w, sample, 2] = gap
                contact[w, sample, 3] = vx
                contact[w, sample, 4] = vy
                contact[w, sample, 5] = vz
                contact[w, sample, 6] = ftx
                contact[w, sample, 7] = fty
                contact[w, sample, 8] = ftz
                contact[w, sample, 9] = fx
                contact[w, sample, 10] = fy
                contact[w, sample, 11] = fz
                contact[w, sample, 12] = py * fz - pz * fy
                contact[w, sample, 13] = pz * fx - px * fz
                contact[w, sample, 14] = px * fy - py * fx
                contact[w, sample, 15] = fn * vn
                contact[w, sample, 16] = ftx * vx + fty * vy + ftz * vz
                for component in range(17):
                    if not tx.isfinite(contact[w, sample, component]):
                        status[w] = 8
                for i in range(nv):
                    gx = ja[w, sample, 0, i] - jb[w, sample, 0, i]
                    gy = ja[w, sample, 1, i] - jb[w, sample, 1, i]
                    gz = ja[w, sample, 2, i] - jb[w, sample, 2, i]
                    vectors[w, 5, i] = vectors[w, 5, i] + gx * fx + gy * fy + gz * fz
        if status[w] == 0:
            load_scaled = 0.0
            load_absolute = 0.0
            velocity_scaled = 0.0
            velocity_absolute = 0.0
            for i in range(nv):
                delta = vectors[w, 5, i] - vectors[w, 4, i]
                vectors[w, 3, i] = delta
                load_scale = 1.0 + tx.max(tx.abs(vectors[w, 5, i]), tx.abs(vectors[w, 4, i]))
                load_scaled = tx.max(load_scaled, tx.abs(delta) / load_scale)
                load_absolute = tx.max(load_absolute, tx.abs(delta))
                value = h * delta
                for j in range(i):
                    value = value - matrices[w, 1, i, j] * vectors[w, 6, j]
                vectors[w, 6, i] = value / matrices[w, 1, i, i]
            for reverse in range(nv):
                i = nv - 1 - reverse
                value = vectors[w, 6, i]
                for j in range(i + 1, nv):
                    value = value - matrices[w, 1, j, i] * vectors[w, 7, j]
                value = value / matrices[w, 1, i, i]
                vectors[w, 7, i] = value
                velocity_scale = 1.0 + tx.max(tx.abs(inputs[w, 1, i]), tx.abs(vectors[w, 1, i]))
                velocity_scaled = tx.max(velocity_scaled, tx.abs(value) / velocity_scale)
                velocity_absolute = tx.max(velocity_absolute, tx.abs(value))
                if not tx.isfinite(value) or not tx.isfinite(vectors[w, 5, i]):
                    status[w] = 8
            trace[w, iteration, 10] = load_scaled
            trace[w, iteration, 11] = velocity_scaled
            trace[w, iteration, 12] = load_absolute
            trace[w, iteration, 13] = velocity_absolute
            for i in range(nv):
                trace[w, iteration, 14 + i] = vectors[w, 1, i]
                trace[w, iteration, 14 + packed_width + i] = vectors[w, 3, i]
            converged = 0
            if status[w] == 0 and load_scaled <= 1e-6 and velocity_scaled <= 1e-7:
                converged = 1
            meta[w, 3] = converged
            trace_meta[w, iteration, 2] = converged
            # Converged lanes retain the load actually solved. While other lanes
            # continue they repeat the same native call, with a real trace row.
            if converged == 0 and iteration < 24 and status[w] == 0:
                # Halve the next update on a slow reversal. When a damped
                # trial still reverses, use a quarter step for this update.
                # A later trial without a fresh reversal returns to half.
                if iteration >= 2 and load_scaled >= 0.5 * trace[w, iteration - 1, 10]:
                    for i in range(nv):
                        delta = vectors[w, 3, i]
                        previous = trace[w, iteration - 1, 14 + packed_width + i]
                        if tx.abs(delta) == load_absolute and (
                            (delta > 0.0 and previous < 0.0) or (delta < 0.0 and previous > 0.0)
                        ):
                            trace_meta[w, iteration, 0] = 2
                            if trace_meta[w, iteration - 1, 0] >= 2:
                                trace_meta[w, iteration, 0] = 3
                for i in range(nv):
                    if trace_meta[w, iteration, 0] == 3:
                        vectors[w, 4, i] = vectors[w, 4, i] + 0.25 * vectors[w, 3, i]
                    elif trace_meta[w, iteration, 0] == 2:
                        vectors[w, 4, i] = vectors[w, 4, i] + 0.5 * vectors[w, 3, i]
                    else:
                        vectors[w, 4, i] = vectors[w, 5, i]
                    trial_applied[w, i] = applied[w, i] + vectors[w, 4, i]
                    if not tx.isfinite(trial_applied[w, i]):
                        status[w] = 8
        meta[w, 0] = status[w]


@tx.kernel
def finish(
    count: tx.i32,
    nv: tx.i32,
    width: tx.i32,
    h: tx.f32,
    inputs: tx.ndarray(tx.f32, 3),
    matrices: tx.ndarray(tx.f32, 4),
    vectors: tx.ndarray(tx.f32, 3),
    input_v: tx.ndarray(tx.f32, 2),
    applied: tx.ndarray(tx.f32, 2),
    passive: tx.ndarray(tx.f32, 2),
    bias: tx.ndarray(tx.f32, 2),
    native_constraint: tx.ndarray(tx.f32, 2),
    total_applied: tx.ndarray(tx.f32, 2),
    smooth_acc: tx.ndarray(tx.f32, 2),
    candidate_v: tx.ndarray(tx.f32, 2),
    meta: tx.ndarray(tx.i32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        if status[w] == 0 and meta[w, 3] != 1:
            status[w] = 8
        for i in range(width):
            candidate_v[w, i] = input_v[w, i]
            if status[w] == 0 and i < nv:
                value = native_constraint[w, i] + vectors[w, 5, i]
                native_constraint[w, i] = value
                vectors[w, 7, i] = value
                total_applied[w, i] = applied[w, i] + value
                vectors[w, 4, i] = total_applied[w, i]
                candidate_v[w, i] = vectors[w, 1, i]
                smooth_acc[w, i] = vectors[w, 0, i]
                if not tx.isfinite(value) or not tx.isfinite(total_applied[w, i]):
                    status[w] = 8
        if status[w] == 0:
            # Check the actual final rounded force representation, not merely
            # Q(final v)-Q(trial v). The existing native tail consumes exactly
            # total_applied and keeps the independently obtained common v.
            for i in range(nv):
                value = -h * (total_applied[w, i] + passive[w, i] - bias[w, i])
                for j in range(nv):
                    value = value + matrices[w, 0, i, j] * (candidate_v[w, j] - inputs[w, 1, j])
                for j in range(i):
                    value = value - matrices[w, 1, i, j] * vectors[w, 6, j]
                vectors[w, 6, i] = value / matrices[w, 1, i, i]
            for reverse in range(nv):
                i = nv - 1 - reverse
                value = vectors[w, 6, i]
                for j in range(i + 1, nv):
                    value = value - matrices[w, 1, j, i] * vectors[w, 3, j]
                value = value / matrices[w, 1, i, i]
                vectors[w, 3, i] = value
                if not tx.isfinite(value) or tx.abs(value) > 1e-5 * (
                    1.0 + tx.abs(candidate_v[w, i])
                ):
                    status[w] = 8
        meta[w, 0] = status[w]
