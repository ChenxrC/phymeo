"""Bounded FP32 shared-DOF final-normal friction; no runtime/type7 state calls."""

import tyx as tx

from .mujoco_tyx._area_friction_layout import FRICTION_FIELDS, scratch_layout


@tx.kernel
def _solve_batch(
    capacity: tx.i32,
    count: tx.i32,
    sizes: tx.ndarray(tx.i32, 2),
    mass: tx.ndarray(tx.f32, 3),
    inputs: tx.ndarray(tx.f32, 3),
    steps: tx.ndarray(tx.f32, 1),
    samples: tx.ndarray(tx.f32, 3),
    ja: tx.ndarray(tx.f32, 4),
    jb: tx.ndarray(tx.f32, 4),
    matrices: tx.ndarray(tx.f32, 4),
    vectors: tx.ndarray(tx.f32, 3),
    contact: tx.ndarray(tx.f32, 3),
    meta: tx.ndarray(tx.i32, 2),
    scalars: tx.ndarray(tx.f32, 2),
    trace_meta: tx.ndarray(tx.i32, 3),
    trace: tx.ndarray(tx.f32, 3),
    require_velocity_agreement: tx.i32,
    packed_width: tx.i32,
    row_capacity: tx.i32,
    native_physical_acceleration: tx.i32,
    rows: tx.ndarray(tx.f32, 3),
    row_j: tx.ndarray(tx.f32, 3),
    counts: tx.ndarray(tx.i32, 2),
):
    # Each lane owns an independent complete problem. All fixed padding is reset.
    # Matrices: A, Cholesky(A), actual residual Jacobian, pivot elimination copy.
    # Vectors: free, accepted, candidate, residual, direction, rhs, forward, JTF.
    # meta: status, accepted updates, trace count, line-search trials, pivot swaps.
    for w in tx.range(capacity):
        for i in range(5):
            meta[w, i] = 0
        for i in range(3):
            scalars[w, i] = 0.0
        for bank in range(4):
            for i in range(packed_width):
                for j in range(packed_width):
                    matrices[w, bank, i, j] = 0.0
        for bank in range(8):
            for i in range(packed_width):
                vectors[w, bank, i] = 0.0
        for sample in range(128):
            for i in range(17):
                contact[w, sample, i] = 0.0
        for row in range(290):
            for i in range(4):
                trace_meta[w, row, i] = 0
            for i in range(3 + 2 * packed_width):
                trace[w, row, i] = 0.0
        if w < count:
            nv = sizes[w, 0]
            ns = sizes[w, 1]
            budget = sizes[w, 2]
            searches = sizes[w, 3]
            h = steps[w]
            nr = 0
            if row_capacity > 0:
                nr = counts[w, 2]
                if nr < 0 or nr > row_capacity:
                    meta[w, 0] = 1
                    nr = 0
                if native_physical_acceleration != 0:
                    # The admitted damped contact route is a single scalar,
                    # with all actual compact native rows bounded together.
                    if nv != 1 or nr > 4 or nr + ns > row_capacity:
                        meta[w, 0] = 1
                        nr = 0
                elif nr > 2 or nv > 2:
                    meta[w, 0] = 1
                    nr = 0
            if (
                nv < 1
                or nv > packed_width
                or ns < 0
                or ns > 128
                or budget < 0
                or budget > 24
                or searches < 0
                or searches > 12
            ):
                meta[w, 0] = 1
                nv = 0
                ns = 0
            if not tx.isfinite(h) or h <= 0.0:
                meta[w, 0] = 1
            if meta[w, 0] == 0:
                for i in range(nv):
                    rhs = h * inputs[w, 2, i]
                    for j in range(nv):
                        value = mass[w, i, j]
                        rhs = rhs + value * inputs[w, 1, j]
                        if i == j:
                            value = value + h * inputs[w, 0, i]
                        matrices[w, 0, i, j] = value
                        if not tx.isfinite(value):
                            meta[w, 0] = 2
                    vectors[w, 5, i] = rhs
                    if not tx.isfinite(rhs):
                        meta[w, 0] = 2
                if native_physical_acceleration != 0:
                    physical_mass = mass[w, 0, 0]
                    damping = inputs[w, 0, 0]
                    if (
                        not tx.isfinite(physical_mass)
                        or physical_mass <= 0.0
                        or not tx.isfinite(damping)
                        or damping < 0.0
                    ):
                        meta[w, 0] = 2
                for i in range(nv):
                    for j in range(i + 1):
                        value = matrices[w, 0, i, j]
                        for k in range(j):
                            value = value - matrices[w, 1, i, k] * matrices[w, 1, j, k]
                        if i == j:
                            if not tx.isfinite(value) or value <= 0.0:
                                meta[w, 0] = 2
                                value = 1.0
                            matrices[w, 1, i, j] = tx.sqrt(value)
                        else:
                            matrices[w, 1, i, j] = value / matrices[w, 1, j, j]
                            if not tx.isfinite(matrices[w, 1, i, j]):
                                meta[w, 0] = 2
                if meta[w, 0] == 0:
                    for i in range(nv):
                        value = vectors[w, 5, i]
                        for j in range(i):
                            value = value - matrices[w, 1, i, j] * vectors[w, 6, j]
                        vectors[w, 6, i] = value / matrices[w, 1, i, i]
                    for reverse in range(nv):
                        i = nv - 1 - reverse
                        value = vectors[w, 6, i]
                        for j in range(i + 1, nv):
                            value = value - matrices[w, 1, j, i] * vectors[w, 0, j]
                        value = value / matrices[w, 1, i, i]
                        vectors[w, 0, i] = value
                        vectors[w, 1, i] = value
                        vectors[w, 2, i] = value
                        if not tx.isfinite(value):
                            meta[w, 0] = 2
                    norm_free2 = 0.0
                    for i in range(nv):
                        value = 0.0
                        for j in range(i, nv):
                            value = value + matrices[w, 1, j, i] * vectors[w, 0, j]
                        norm_free2 = norm_free2 + value * value
                    if not tx.isfinite(norm_free2) or norm_free2 < 0.0:
                        meta[w, 0] = 2
                        norm_free2 = 0.0
                    scalars[w, 2] = tx.sqrt(norm_free2)
                    scalars[w, 1] = 1e-5 * (1.0 + scalars[w, 2])
            done = 0
            trial = 0
            backtrack = 0
            alpha = 0.0
            search_norm = 0.0
            # Initial + 24*12 actual evaluations. The extra trace slot holds
            # a final witness copied from the accepted evaluation, not a solve.
            for evaluation in range(289):
                if done == 0 and meta[w, 0] == 0:
                    for i in range(nv):
                        vectors[w, 7, i] = 0.0
                        for j in range(nv):
                            matrices[w, 2, i, j] = matrices[w, 0, i, j]
                    for sample in range(ns):
                        nx = samples[w, sample, 3]
                        ny = samples[w, sample, 4]
                        nz = samples[w, sample, 5]
                        nn = nx * nx + ny * ny + nz * nz
                        k = samples[w, sample, 6]
                        phi = samples[w, sample, 7]
                        mu = samples[w, sample, 8]
                        vs = samples[w, sample, 9]
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
                        fn = -k * gap
                        if fn < 0.0:
                            fn = 0.0
                        speed2 = vx * vx + vy * vy + vz * vz + vs * vs
                        if not tx.isfinite(speed2) or speed2 <= 0.0:
                            meta[w, 0] = 2
                            speed2 = 1.0
                        speed = tx.sqrt(speed2)
                        ex = vx / speed
                        ey = vy / speed
                        ez = vz / speed
                        ftx = -mu * fn * ex
                        fty = -mu * fn * ey
                        ftz = -mu * fn * ez
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
                                meta[w, 0] = 2
                        for i in range(nv):
                            gx = ja[w, sample, 0, i] - jb[w, sample, 0, i]
                            gy = ja[w, sample, 1, i] - jb[w, sample, 1, i]
                            gz = ja[w, sample, 2, i] - jb[w, sample, 2, i]
                            vectors[w, 7, i] = vectors[w, 7, i] + gx * fx + gy * fy + gz * fz
                            for j in range(nv):
                                dx = ja[w, sample, 0, j] - jb[w, sample, 0, j]
                                dy = ja[w, sample, 1, j] - jb[w, sample, 1, j]
                                dz = ja[w, sample, 2, j] - jb[w, sample, 2, j]
                                gn = nx * dx + ny * dy + nz * dz
                                dfn = 0.0
                                if gap < 0.0:
                                    dfn = -k * h * gn
                                qx = dx - nx * gn / nn
                                qy = dy - ny * gn / nn
                                qz = dz - nz * gn / nn
                                dot = ex * qx + ey * qy + ez * qz
                                dtx = (qx - ex * dot) / speed
                                dty = (qy - ey * dot) / speed
                                dtz = (qz - ez * dot) / speed
                                dfx = (nx - mu * ex) * dfn - mu * fn * dtx
                                dfy = (ny - mu * ey) * dfn - mu * fn * dty
                                dfz = (nz - mu * ez) * dfn - mu * fn * dtz
                                matrices[w, 2, i, j] = matrices[w, 2, i, j] - h * (
                                    gx * dfx + gy * dfy + gz * dfz
                                )
                    # Eliminate the scalar native contact rows at this candidate's
                    # complete area load Q(v), rather than recovering their force
                    # through the quantized (v-v0)/h difference. On each active
                    # face: (M + sum J^2/R)*a = b + Q(v) + sum J*aref/R.
                    # Only the new unilateral contact route uses this reduction.
                    scalar_area_load = vectors[w, 7, 0]
                    native_rhs = 0.0
                    native_effective_mass = 1.0
                    active_face = 0
                    if native_physical_acceleration != 0 and nr > 0:
                        native_rhs = (
                            inputs[w, 2, 0] - inputs[w, 0, 0] * inputs[w, 1, 0] + vectors[w, 7, 0]
                        )
                        for r in range(nr):
                            if (
                                not tx.isfinite(rows[w, r, 4])
                                or rows[w, r, 4] <= 0.0
                                or not tx.isfinite(rows[w, r, 3])
                                or not tx.isfinite(row_j[w, r, 0])
                                or rows[w, r, 7] != 0.0
                                or rows[w, r, 12] != 0.0
                            ):
                                meta[w, 0] = 2
                        selected = 0
                        native_effective_mass = mass[w, 0, 0]
                        # At most four rows: all 16 faces, including all-open.
                        # Extra upper bits for smaller row counts are harmless.
                        for face in range(16):
                            if selected == 0 and meta[w, 0] == 0:
                                denominator = mass[w, 0, 0]
                                numerator = native_rhs
                                bits = face
                                for r in range(nr):
                                    if bits % 2 == 1:
                                        j_over_r = row_j[w, r, 0] / rows[w, r, 4]
                                        denominator = denominator + row_j[w, r, 0] * j_over_r
                                        numerator = numerator + rows[w, r, 3] * j_over_r
                                    bits = bits // 2
                                if (
                                    tx.isfinite(denominator)
                                    and denominator > 0.0
                                    and tx.isfinite(numerator)
                                ):
                                    candidate_acceleration = numerator / denominator
                                    face_ok = 1
                                    if not tx.isfinite(candidate_acceleration):
                                        face_ok = 0
                                    bits = face
                                    for r in range(nr):
                                        slack = (
                                            rows[w, r, 3] - row_j[w, r, 0] * candidate_acceleration
                                        )
                                        if not tx.isfinite(slack):
                                            face_ok = 0
                                        if (bits % 2 == 1 and slack < 0.0) or (
                                            bits % 2 == 0 and slack > 0.0
                                        ):
                                            face_ok = 0
                                        bits = bits // 2
                                    if face_ok == 1:
                                        selected = 1
                                        active_face = face
                                        native_effective_mass = denominator
                        if selected == 0 and meta[w, 0] == 0:
                            meta[w, 0] = 4
                        # da/dv = Q'(v)/H; d(Q+J'f)/dv = M*Q'(v)/H.
                        # Preserve every area normal/tangent derivative already
                        # assembled above, including their cross terms.
                        matrices[w, 2, 0, 0] = matrices[w, 0, 0, 0] + (
                            matrices[w, 2, 0, 0] - matrices[w, 0, 0, 0]
                        ) * (mass[w, 0, 0] / native_effective_mass)
                    # All native rows use the same candidate's complete area load.
                    native_ok = 1
                    for r in range(nr):
                        reg = rows[w, r, 4]
                        if not tx.isfinite(reg) or reg <= 0.0:
                            meta[w, 0] = 2
                        else:
                            raw = 0.0
                            if native_physical_acceleration != 0:
                                # Expand H*(aref_i - J_i*a) before division:
                                # the row's own large J_i^2*aref_i/R_i terms
                                # cancel analytically, preserving its small force.
                                ji = row_j[w, r, 0]
                                reference = rows[w, r, 3]
                                dual = mass[w, 0, 0] * reference - ji * native_rhs
                                bits = active_face
                                for k in range(nr):
                                    if bits % 2 == 1 and k != r:
                                        jk = row_j[w, k, 0]
                                        dual = dual + (jk / rows[w, k, 4]) * (
                                            jk * reference - ji * rows[w, k, 3]
                                        )
                                    bits = bits // 2
                                raw = dual / (reg * native_effective_mass)
                            else:
                                projected = 0.0
                                for i in range(nv):
                                    projected = projected + (
                                        row_j[w, r, i] * (vectors[w, 2, i] - inputs[w, 1, i]) / h
                                    )
                                raw = (rows[w, r, 3] - projected) / reg
                            lower = rows[w, r, 7]
                            upper = rows[w, r, 8]
                            has_upper = rows[w, r, 12] == 1.0
                            reaction = tx.max(lower, raw)
                            if has_upper:
                                reaction = tx.min(upper, reaction)
                            if not tx.isfinite(raw):
                                meta[w, 0] = 2
                            rows[w, r, 9] = reaction
                            for i in range(nv):
                                ji = row_j[w, r, i]
                                vectors[w, 7, i] = vectors[w, 7, i] + ji * reaction
                                if (
                                    native_physical_acceleration == 0
                                    and raw > lower
                                    and (not has_upper or raw < upper)
                                ):
                                    for j in range(nv):
                                        matrices[w, 2, i, j] = (
                                            matrices[w, 2, i, j] + ji * row_j[w, r, j] / reg
                                        )
                    if nr > 0:
                        if native_physical_acceleration != 0:
                            # Correct rounding of the analytical face solution in
                            # the published force coordinates. Each diagonal is
                            # the exact scalar Schur diagonal R_i + J_i^2/M.
                            # Reassemble the rounded forces in publication order.
                            for sweep in range(2):
                                for r in range(nr):
                                    physical = (
                                        inputs[w, 2, 0]
                                        - inputs[w, 0, 0] * inputs[w, 1, 0]
                                        + vectors[w, 7, 0]
                                    ) / mass[w, 0, 0]
                                    ji = row_j[w, r, 0]
                                    old_force = rows[w, r, 9]
                                    gradient = (
                                        rows[w, r, 4] * old_force - rows[w, r, 3] + ji * physical
                                    )
                                    defect = tx.abs(gradient)
                                    if old_force == 0.0:
                                        defect = tx.max(0.0, -gradient)
                                    if defect > 1e-5 * tx.max(
                                        1.0, tx.abs(rows[w, r, 4] * old_force)
                                    ):
                                        diagonal = rows[w, r, 4] + ji * ji / mass[w, 0, 0]
                                        rows[w, r, 9] = tx.max(0.0, old_force - gradient / diagonal)
                                        load = scalar_area_load
                                        for k in range(nr):
                                            load = load + row_j[w, k, 0] * rows[w, k, 9]
                                        vectors[w, 7, 0] = load
                            # Independent force-based KKT witness using M, not A.
                            # Pack adds D*v0 to its integration RHS; remove that
                            # compensation to recover the original physical load.
                            vectors[w, 5, 0] = (
                                inputs[w, 2, 0]
                                - inputs[w, 0, 0] * inputs[w, 1, 0]
                                + vectors[w, 7, 0]
                            ) / mass[w, 0, 0]
                            if not tx.isfinite(vectors[w, 5, 0]):
                                meta[w, 0] = 2
                        else:
                            # Legacy joint-only rows have D=0 and A=M.
                            for i in range(nv):
                                value = inputs[w, 2, i] + vectors[w, 7, i]
                                for j in range(i):
                                    value = value - matrices[w, 1, i, j] * vectors[w, 5, j]
                                vectors[w, 5, i] = value / matrices[w, 1, i, i]
                            for reverse in range(nv):
                                i = nv - 1 - reverse
                                value = vectors[w, 5, i]
                                for j in range(i + 1, nv):
                                    value = value - matrices[w, 1, j, i] * vectors[w, 5, j]
                                vectors[w, 5, i] = value / matrices[w, 1, i, i]
                        for r in range(nr):
                            reaction = rows[w, r, 9]
                            compliance_term = rows[w, r, 4] * reaction
                            gradient = compliance_term - rows[w, r, 3]
                            for i in range(nv):
                                gradient = gradient + row_j[w, r, i] * vectors[w, 5, i]
                            violation = tx.abs(gradient)
                            if reaction == rows[w, r, 7]:
                                violation = tx.max(0.0, -gradient)
                            elif rows[w, r, 12] == 1.0 and reaction == rows[w, r, 8]:
                                violation = tx.max(0.0, gradient)
                            if not tx.isfinite(gradient) or violation > 1e-5 * tx.max(
                                1.0, tx.abs(compliance_term)
                            ):
                                native_ok = 0
                    norm2 = 0.0
                    for i in range(nv):
                        value = -h * vectors[w, 7, i]
                        for j in range(nv):
                            value = value + matrices[w, 0, i, j] * (
                                vectors[w, 2, j] - vectors[w, 0, j]
                            )
                            if not tx.isfinite(matrices[w, 2, i, j]):
                                meta[w, 0] = 2
                        vectors[w, 3, i] = value
                        for j in range(i):
                            value = value - matrices[w, 1, i, j] * vectors[w, 6, j]
                        value = value / matrices[w, 1, i, i]
                        vectors[w, 6, i] = value
                        norm2 = norm2 + value * value
                    norm = 0.0
                    if not tx.isfinite(norm2) or norm2 < 0.0:
                        meta[w, 0] = 2
                    else:
                        norm = tx.sqrt(norm2)
                    scalars[w, 0] = norm
                    trace_meta[w, evaluation, 0] = meta[w, 1] + trial
                    trace_meta[w, evaluation, 1] = backtrack
                    trace_meta[w, evaluation, 2] = trial
                    trace_meta[w, evaluation, 3] = -1
                    trace[w, evaluation, 0] = alpha
                    trace[w, evaluation, 1] = norm
                    trace[w, evaluation, 2] = scalars[w, 1]
                    for i in range(nv):
                        trace[w, evaluation, 3 + i] = vectors[w, 2, i]
                        trace[w, evaluation, 3 + packed_width + i] = vectors[w, 3, i]
                    meta[w, 2] = evaluation + 1
                    if trial == 1:
                        meta[w, 3] = meta[w, 3] + 1
                    # The mass norm alone can hide velocity error in a small
                    # rotational inertia. Live integration reuses the stored Q,
                    # so also bound the correction A^-1 residual before stopping.
                    # Bank 5 is scratch here and is overwritten by the Newton RHS.
                    velocity_ok = 1
                    if require_velocity_agreement != 0:
                        for reverse in range(nv):
                            i = nv - 1 - reverse
                            value = vectors[w, 6, i]
                            for j in range(i + 1, nv):
                                value = value - matrices[w, 1, j, i] * vectors[w, 5, j]
                            value = value / matrices[w, 1, i, i]
                            vectors[w, 5, i] = value
                            if not tx.isfinite(value):
                                meta[w, 0] = 2
                            if tx.abs(value) > 1e-5 * (1.0 + tx.abs(vectors[w, 2, i])):
                                velocity_ok = 0
                    if native_physical_acceleration != 0 and nr > 0:
                        # Condensation can mask a velocity error at stiff native
                        # edges. Recheck the original momentum equation using
                        # the forces implied directly by the output velocity.
                        acceleration_at_v = (
                            (vectors[w, 2, 0] - inputs[w, 1, 0])
                            / h
                            * (matrices[w, 0, 0, 0] / mass[w, 0, 0])
                        )
                        uncondensed = vectors[w, 3, 0]
                        for r in range(nr):
                            ji = row_j[w, r, 0]
                            at_v = tx.max(
                                0.0, (rows[w, r, 3] - ji * acceleration_at_v) / rows[w, r, 4]
                            )
                            uncondensed = uncondensed - h * ji * (at_v - rows[w, r, 9])
                        if (
                            not tx.isfinite(uncondensed)
                            or tx.abs(uncondensed) > matrices[w, 1, 0, 0] * scalars[w, 1]
                        ):
                            velocity_ok = 0
                    accepted = 0
                    if meta[w, 0] == 0:
                        if norm <= scalars[w, 1] and velocity_ok == 1 and native_ok == 1:
                            accepted = 1
                            done = 1
                            trace_meta[w, evaluation, 3] = 2
                        elif trial == 0 or norm <= (1.0 - 1e-4 * alpha) * search_norm:
                            accepted = 1
                            trace_meta[w, evaluation, 3] = 1
                        else:
                            trace_meta[w, evaluation, 3] = 0
                    if accepted == 1:
                        for i in range(nv):
                            vectors[w, 1, i] = vectors[w, 2, i]
                        if trial == 1:
                            meta[w, 1] = meta[w, 1] + 1
                        if done == 1:
                            for j in range(3 + 2 * packed_width):
                                trace[w, evaluation + 1, j] = trace[w, evaluation, j]
                            for j in range(4):
                                trace_meta[w, evaluation + 1, j] = trace_meta[w, evaluation, j]
                            trace_meta[w, evaluation + 1, 2] = 2
                            meta[w, 2] = evaluation + 2
                        elif meta[w, 1] >= budget:
                            meta[w, 0] = 3
                        elif searches == 0:
                            meta[w, 0] = 5
                        else:
                            search_norm = norm
                            for i in range(nv):
                                vectors[w, 5, i] = -vectors[w, 3, i]
                                for j in range(nv):
                                    matrices[w, 3, i, j] = matrices[w, 2, i, j]
                            # General nonsymmetric elimination with partial row
                            # pivoting. Do not replace this with Cholesky.
                            for column in range(nv):
                                pivot = column
                                largest = tx.abs(matrices[w, 3, column, column])
                                for row in range(column + 1, nv):
                                    value = tx.abs(matrices[w, 3, row, column])
                                    if value > largest:
                                        largest = value
                                        pivot = row
                                if not tx.isfinite(largest) or largest <= 0.0:
                                    meta[w, 0] = 4
                                if meta[w, 0] == 0:
                                    if pivot != column:
                                        for j in range(nv):
                                            temp = matrices[w, 3, column, j]
                                            matrices[w, 3, column, j] = matrices[w, 3, pivot, j]
                                            matrices[w, 3, pivot, j] = temp
                                        temp = vectors[w, 5, column]
                                        vectors[w, 5, column] = vectors[w, 5, pivot]
                                        vectors[w, 5, pivot] = temp
                                        meta[w, 4] = meta[w, 4] + 1
                                    for row in range(column + 1, nv):
                                        factor = (
                                            matrices[w, 3, row, column]
                                            / matrices[w, 3, column, column]
                                        )
                                        matrices[w, 3, row, column] = 0.0
                                        for j in range(column + 1, nv):
                                            matrices[w, 3, row, j] = (
                                                matrices[w, 3, row, j]
                                                - factor * matrices[w, 3, column, j]
                                            )
                                        vectors[w, 5, row] = (
                                            vectors[w, 5, row] - factor * vectors[w, 5, column]
                                        )
                            if meta[w, 0] == 0:
                                for reverse in range(nv):
                                    i = nv - 1 - reverse
                                    value = vectors[w, 5, i]
                                    for j in range(i + 1, nv):
                                        value = value - matrices[w, 3, i, j] * vectors[w, 4, j]
                                    vectors[w, 4, i] = value / matrices[w, 3, i, i]
                                    if not tx.isfinite(vectors[w, 4, i]):
                                        meta[w, 0] = 4
                            alpha = 1.0
                            backtrack = 0
                            trial = 1
                            for i in range(nv):
                                vectors[w, 2, i] = vectors[w, 1, i] + vectors[w, 4, i]
                    elif meta[w, 0] == 0:
                        backtrack = backtrack + 1
                        if backtrack >= searches:
                            meta[w, 0] = 5
                        else:
                            alpha = alpha * 0.5
                            for i in range(nv):
                                vectors[w, 2, i] = vectors[w, 1, i] + alpha * vectors[w, 4, i]
            if meta[w, 0] == 0 and done == 0:
                meta[w, 0] = 3


def solve_batch(
    capacity,
    count,
    sizes,
    mass,
    *buffers,
    require_velocity_agreement=False,
    native_rows=None,
    native_physical_acceleration=False,
    **launch,
):
    """Keep the standalone ABI; live owners additionally bound A^-1 residual.

    ``native_physical_acceleration`` admits up to four actual native rows on
    one damped scalar. Each candidate's area load is used to eliminate the
    native unilateral rows in physical acceleration space, without modifying
    native R, aref, or force units. The owner
    supplies all compact rows and enforces its combined area/native capacity.
    The default retains the existing undamped joint-row route.
    """
    if type(require_velocity_agreement) is not bool:
        raise ValueError("require_velocity_agreement must be bool")
    if type(native_physical_acceleration) is not bool:
        raise ValueError("native_physical_acceleration must be bool")
    if native_physical_acceleration and native_rows is None:
        raise ValueError("physical native acceleration requires native row storage")
    if len(mass.shape) != 3 or mass.shape[1] not in (6, 8):
        raise ValueError("invalid packed area-friction mass layout")
    width = mass.shape[1]
    layout = scratch_layout(capacity, width)
    for name, array in zip(FRICTION_FIELDS, (sizes, mass, *buffers), strict=True):
        if array.shape != layout[name][0]:
            raise ValueError(f"invalid packed area-friction {name} layout")
    # Standalone launches do not read the placeholder bindings.
    rows, jacobian, counts = mass, mass, sizes
    row_capacity = 0
    if native_rows is not None:
        rows, jacobian, counts = native_rows
        if len(rows.shape) != 3 or rows.shape[0] != capacity or rows.shape[2] != 14:
            raise ValueError("invalid native constraint row storage")
        row_capacity = rows.shape[1]
        if (
            row_capacity < 1
            or len(jacobian.shape) != 3
            or jacobian.shape[:2] != rows.shape[:2]
            or jacobian.shape[2] < width
            or counts.shape != (capacity, 3)
        ):
            raise ValueError("invalid native constraint Jacobian or count storage")
    return _solve_batch(
        capacity,
        count,
        sizes,
        mass,
        *buffers,
        int(require_velocity_agreement),
        width,
        row_capacity,
        int(native_physical_acceleration),
        rows,
        jacobian,
        counts,
        **launch,
    )
