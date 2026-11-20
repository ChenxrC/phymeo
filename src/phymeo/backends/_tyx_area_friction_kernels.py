"""Bounded FP32 shared-DOF final-normal friction; no runtime/type7 state calls."""

import tyx as tx

from .mujoco_tyx._area_friction_layout import FRICTION_FIELDS, scratch_layout


@tx.func
def _sum_error(a: tx.f32, b: tx.f32, rounded: tx.f32) -> tx.f32:
    return tx.add_rn(
        tx.sub_rn(a, tx.sub_rn(rounded, tx.sub_rn(rounded, a))),
        tx.sub_rn(b, tx.sub_rn(rounded, a)),
    )


@tx.func
def _add_product(total: tx.vec2f, a: tx.f32, b: tx.f32) -> tx.vec2f:
    return tx.vec2f(
        tx.add_rn(total[0], tx.mul_rn(a, b)),
        tx.add_rn(
            total[1],
            tx.add_rn(
                _sum_error(total[0], tx.mul_rn(a, b), tx.add_rn(total[0], tx.mul_rn(a, b))),
                tx.fma(a, b, -tx.mul_rn(a, b)),
            ),
        ),
    )


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
    sample_capacity: tx.i32,
    row_capacity: tx.i32,
    native_physical_acceleration: tx.i32,
    native_limit_count: tx.i32,
    rows: tx.ndarray(tx.f32, 3),
    row_j: tx.ndarray(tx.f32, 3),
    counts: tx.ndarray(tx.i32, 2),
    source_applied: tx.ndarray(tx.f32, 2),
    source_passive: tx.ndarray(tx.f32, 2),
    source_bias: tx.ndarray(tx.f32, 2),
):
    # Each lane owns an independent complete problem. All fixed padding is reset.
    # Matrices: A, Cholesky(A), actual residual Jacobian, pivot elimination copy.
    # Vectors: free, accepted, candidate, residual, direction, rhs, forward, JTF.
    # On mode-4 success ONLY, direction becomes compensated applied+JTF (bank 4).
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
        for sample in range(sample_capacity):
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
            live_source_mode = (
                native_physical_acceleration == 3 or native_physical_acceleration == 4
            )
            coupled_mode = native_physical_acceleration == 2 or live_source_mode
            # Independent physical-M LDL factors for modes 2/3/4. Never reuse A's
            # Cholesky factor for native acceleration or Schur diagonals.
            m00 = 1.0
            m01 = 0.0
            m11 = 1.0
            ml10 = 0.0
            ms11 = 1.0
            published_l00 = 1.0
            published_l10 = 0.0
            published_l11 = 1.0
            if row_capacity > 0:
                nr = counts[w, 2]
                if nr < 0 or nr > row_capacity:
                    meta[w, 0] = 1
                    nr = 0
                if native_physical_acceleration == 1:
                    # The admitted damped contact route is a single scalar,
                    # with all actual compact native rows bounded together.
                    if nv != 1 or nr > 4 or nr + ns > row_capacity:
                        meta[w, 0] = 1
                        nr = 0
                elif native_physical_acceleration == 2:
                    if nv != 2 or nr > 4 or nr + ns > row_capacity:
                        meta[w, 0] = 1
                        nr = 0
                elif native_physical_acceleration == 3:
                    # Four contact edges plus one side per admitted joint limit.
                    # The owner supplies the declared count and guarantees each
                    # limit's margins cannot overlap. Default count 1 keeps the
                    # five-row gate; count 2 explicitly admits at most six.
                    if (
                        nv != 2
                        or native_limit_count < 1
                        or native_limit_count > 2
                        or nr > 4 + native_limit_count
                        or nr + ns > row_capacity
                    ):
                        meta[w, 0] = 1
                        nr = 0
                elif native_physical_acceleration == 4:
                    # Two declared condim-3 supports, no joint limits: all eight
                    # compact native edges share the same candidate area load.
                    # This separate budget never expands the older mode gates.
                    if nv != 2 or nr > 8 or nr + ns > row_capacity:
                        meta[w, 0] = 1
                        nr = 0
                elif nr > 2 or nv > 2:
                    meta[w, 0] = 1
                    nr = 0
            if (
                nv < 1
                or nv > packed_width
                or ns < 0
                or ns > sample_capacity
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
                if native_physical_acceleration == 1:
                    physical_mass = mass[w, 0, 0]
                    damping = inputs[w, 0, 0]
                    if (
                        not tx.isfinite(physical_mass)
                        or physical_mass <= 0.0
                        or not tx.isfinite(damping)
                        or damping < 0.0
                    ):
                        meta[w, 0] = 2
                elif coupled_mode:
                    m00 = mass[w, 0, 0]
                    m01 = mass[w, 0, 1]
                    m11 = mass[w, 1, 1]
                    if (
                        not tx.isfinite(m00)
                        or m00 <= 0.0
                        or not tx.isfinite(m01)
                        or m01 != mass[w, 1, 0]
                        or not tx.isfinite(m11)
                    ):
                        meta[w, 0] = 2
                    else:
                        ml10 = m01 / m00
                        ms11 = m11 - m01 * ml10
                        if not tx.isfinite(ml10) or not tx.isfinite(ms11) or ms11 <= 0.0:
                            meta[w, 0] = 2
                    for i in range(2):
                        if not tx.isfinite(inputs[w, 0, i]) or inputs[w, 0, i] < 0.0:
                            meta[w, 0] = 2
                    if live_source_mode and meta[w, 0] == 0:
                        # Match the native tree's physical-M factor operation by
                        # operation. Its final qacc does not use our LDL factor.
                        published_l00 = tx.sqrt(m00)
                        published_l10 = mass[w, 1, 0] / published_l00
                        value = m11 - published_l10 * published_l10
                        if not tx.isfinite(value) or value <= 0.0:
                            meta[w, 0] = 2
                        else:
                            published_l11 = tx.sqrt(value)
                        for i in range(2):
                            if (
                                not tx.isfinite(source_applied[w, i])
                                or not tx.isfinite(source_passive[w, i])
                                or not tx.isfinite(source_bias[w, i])
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
                    area_sum0 = tx.vec2f(0.0, 0.0)
                    area_sum1 = tx.vec2f(0.0, 0.0)
                    base_sum0 = tx.vec2f(0.0, 0.0)
                    base_sum1 = tx.vec2f(0.0, 0.0)
                    face_a0 = 0.0
                    face_a1 = 0.0
                    net_applied0 = 0.0
                    net_applied1 = 0.0
                    if native_physical_acceleration == 4:
                        # Reuse the SAME rounded point forces and point Jacobians.
                        # Preserve product and addition residuals until the net
                        # applied+constraint sum, not just until the large Q sum.
                        for sample in range(ns):
                            for axis in range(3):
                                force = contact[w, sample, 9 + axis]
                                j0 = ja[w, sample, axis, 0] - jb[w, sample, axis, 0]
                                j1 = ja[w, sample, axis, 1] - jb[w, sample, axis, 1]
                                area_sum0 = _add_product(area_sum0, j0, force)
                                area_sum1 = _add_product(area_sum1, j1, force)
                        vectors[w, 7, 0] = tx.add_rn(area_sum0[0], area_sum0[1])
                        vectors[w, 7, 1] = tx.add_rn(area_sum1[0], area_sum1[1])
                        base_sum0 = _add_product(area_sum0, 1.0, source_applied[w, 0])
                        base_sum1 = _add_product(area_sum1, 1.0, source_applied[w, 1])
                        base_sum0 = _add_product(base_sum0, 1.0, source_passive[w, 0])
                        base_sum1 = _add_product(base_sum1, 1.0, source_passive[w, 1])
                        base_sum0 = _add_product(base_sum0, -1.0, source_bias[w, 0])
                        base_sum1 = _add_product(base_sum1, -1.0, source_bias[w, 1])
                        # Also define the no-native-row face without a different
                        # force or integration convention.
                        numerator_sum = _add_product(base_sum1, -ml10, base_sum0[0])
                        numerator_sum = _add_product(numerator_sum, -ml10, base_sum0[1])
                        face_a1 = tx.div_rn(tx.add_rn(numerator_sum[0], numerator_sum[1]), ms11)
                        numerator_sum = _add_product(base_sum0, -m01, face_a1)
                        face_a0 = tx.div_rn(tx.add_rn(numerator_sum[0], numerator_sum[1]), m00)
                    scalar_area_load = vectors[w, 7, 0]
                    coupled_area_load = 0.0
                    if coupled_mode:
                        coupled_area_load = vectors[w, 7, 1]
                    native_rhs = 0.0
                    native_effective_mass = 1.0
                    active_face = 0
                    if native_physical_acceleration == 1 and nr > 0:
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
                    elif coupled_mode and nr > 0:
                        # On each bounded face, H=M+sum J_r J_r^T/R_r
                        # is SPD. Solve in physical acceleration space with an
                        # explicit 2x2 LDL factor; A=M+hD remains the integrator.
                        b0 = inputs[w, 2, 0] - inputs[w, 0, 0] * inputs[w, 1, 0]
                        b1 = inputs[w, 2, 1] - inputs[w, 0, 1] * inputs[w, 1, 1]
                        if live_source_mode:
                            # Do not reverse the lossy packed +D*v0 compensation.
                            b0 = source_applied[w, 0] + source_passive[w, 0] - source_bias[w, 0]
                            b1 = source_applied[w, 1] + source_passive[w, 1] - source_bias[w, 1]
                        coupled_rhs0 = b0 + scalar_area_load
                        coupled_rhs1 = b1 + coupled_area_load
                        for r in range(nr):
                            if (
                                not tx.isfinite(rows[w, r, 4])
                                or rows[w, r, 4] <= 0.0
                                or not tx.isfinite(rows[w, r, 3])
                                or not tx.isfinite(row_j[w, r, 0])
                                or not tx.isfinite(row_j[w, r, 1])
                                or rows[w, r, 7] != 0.0
                                or rows[w, r, 12] != 0.0
                            ):
                                meta[w, 0] = 2
                        selected = 0
                        chosen_h00 = m00
                        chosen_h01 = m01
                        chosen_l10 = ml10
                        chosen_s11 = ms11
                        # Preserve mode 2's loop. Modes 3/4 enumerate only actual
                        # compact rows: 2^nr masks, including all-inactive; <=32
                        # for one limit, <=64 for two, <=256 for two supports.
                        # The selected guard skips later face solves. Worst-case
                        # mode 4 costs 256 tiny SPD solves and 2*8*256 row visits
                        # per candidate, followed by bounded force corrections.
                        face_count = 16
                        if live_source_mode:
                            face_count = 1
                            for r in range(nr):
                                face_count = face_count * 2
                        for face in range(face_count):
                            if selected == 0 and meta[w, 0] == 0:
                                h00 = m00
                                h01 = m01
                                h11 = m11
                                c0 = coupled_rhs0
                                c1 = coupled_rhs1
                                hsum00 = tx.vec2f(m00, 0.0)
                                hsum01 = tx.vec2f(m01, 0.0)
                                hsum11 = tx.vec2f(m11, 0.0)
                                csum0 = base_sum0
                                csum1 = base_sum1
                                bits = face
                                for r in range(nr):
                                    if bits % 2 == 1:
                                        j0 = row_j[w, r, 0]
                                        j1 = row_j[w, r, 1]
                                        reg = rows[w, r, 4]
                                        h00 = h00 + j0 * (j0 / reg)
                                        h01 = h01 + j0 * (j1 / reg)
                                        h11 = h11 + j1 * (j1 / reg)
                                        c0 = c0 + j0 * (rows[w, r, 3] / reg)
                                        c1 = c1 + j1 * (rows[w, r, 3] / reg)
                                        if native_physical_acceleration == 4:
                                            # Recover the rounded quotient's low
                                            # part before accumulating j*aref/R.
                                            q0 = tx.div_rn(j0, reg)
                                            q1 = tx.div_rn(j1, reg)
                                            qr = tx.div_rn(rows[w, r, 3], reg)
                                            e0 = tx.div_rn(tx.fma(-q0, reg, j0), reg)
                                            e1 = tx.div_rn(tx.fma(-q1, reg, j1), reg)
                                            er = tx.div_rn(tx.fma(-qr, reg, rows[w, r, 3]), reg)
                                            hsum00 = _add_product(hsum00, j0, q0)
                                            hsum00 = _add_product(hsum00, j0, e0)
                                            hsum01 = _add_product(hsum01, j0, q1)
                                            hsum01 = _add_product(hsum01, j0, e1)
                                            hsum11 = _add_product(hsum11, j1, q1)
                                            hsum11 = _add_product(hsum11, j1, e1)
                                            csum0 = _add_product(csum0, j0, qr)
                                            csum0 = _add_product(csum0, j0, er)
                                            csum1 = _add_product(csum1, j1, qr)
                                            csum1 = _add_product(csum1, j1, er)
                                    bits = bits // 2
                                if native_physical_acceleration == 4:
                                    h00 = tx.add_rn(hsum00[0], hsum00[1])
                                    h01 = tx.add_rn(hsum01[0], hsum01[1])
                                    h11 = tx.add_rn(hsum11[0], hsum11[1])
                                    c0 = tx.add_rn(csum0[0], csum0[1])
                                    c1 = tx.add_rn(csum1[0], csum1[1])
                                if tx.isfinite(h00) and h00 > 0.0:
                                    l10 = h01 / h00
                                    s11 = h11 - h01 * l10
                                    if native_physical_acceleration == 4:
                                        s11 = tx.fma(-h01, l10, h11)
                                    if tx.isfinite(s11) and s11 > 0.0:
                                        a1 = (c1 - l10 * c0) / s11
                                        a0 = (c0 - h01 * a1) / h00
                                        if native_physical_acceleration == 4:
                                            numerator_sum = _add_product(csum1, -l10, csum0[0])
                                            numerator_sum = _add_product(
                                                numerator_sum, -l10, csum0[1]
                                            )
                                            a1 = tx.div_rn(
                                                tx.add_rn(numerator_sum[0], numerator_sum[1]), s11
                                            )
                                            numerator_sum = _add_product(csum0, -h01, a1)
                                            a0 = tx.div_rn(
                                                tx.add_rn(numerator_sum[0], numerator_sum[1]), h00
                                            )
                                        face_ok = 1
                                        if not tx.isfinite(a0) or not tx.isfinite(a1):
                                            face_ok = 0
                                        bits = face
                                        for r in range(nr):
                                            slack = rows[w, r, 3] - (
                                                row_j[w, r, 0] * a0 + row_j[w, r, 1] * a1
                                            )
                                            if native_physical_acceleration == 4:
                                                slack_sum = tx.vec2f(rows[w, r, 3], 0.0)
                                                slack_sum = _add_product(
                                                    slack_sum, -row_j[w, r, 0], a0
                                                )
                                                slack_sum = _add_product(
                                                    slack_sum, -row_j[w, r, 1], a1
                                                )
                                                slack = tx.add_rn(slack_sum[0], slack_sum[1])
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
                                            chosen_h00 = h00
                                            chosen_h01 = h01
                                            chosen_l10 = l10
                                            chosen_s11 = s11
                                            face_a0 = a0
                                            face_a1 = a1
                        if selected == 0 and meta[w, 0] == 0:
                            meta[w, 0] = 4
                        if meta[w, 0] == 0:
                            # Reconstruct each active force after excluding its
                            # OWN rank-one update. The denominator R+J K^-1 J
                            # avoids subtracting two self-amplified row loads.
                            bits = active_face
                            for r in range(nr):
                                rows[w, r, 9] = 0.0
                                if native_physical_acceleration == 4:
                                    # All forces come from this single face a,
                                    # never independent row-specific solves.
                                    slack_sum = tx.vec2f(rows[w, r, 3], 0.0)
                                    slack_sum = _add_product(slack_sum, -row_j[w, r, 0], face_a0)
                                    slack_sum = _add_product(slack_sum, -row_j[w, r, 1], face_a1)
                                    raw = tx.div_rn(
                                        tx.add_rn(slack_sum[0], slack_sum[1]), rows[w, r, 4]
                                    )
                                    if not tx.isfinite(raw):
                                        meta[w, 0] = 2
                                    rows[w, r, 9] = tx.max(0.0, raw)
                                elif bits % 2 == 1:
                                    k00 = m00
                                    k01 = m01
                                    k11 = m11
                                    c0 = coupled_rhs0
                                    c1 = coupled_rhs1
                                    other_bits = active_face
                                    for k in range(nr):
                                        if other_bits % 2 == 1 and k != r:
                                            j0 = row_j[w, k, 0]
                                            j1 = row_j[w, k, 1]
                                            reg = rows[w, k, 4]
                                            k00 = k00 + j0 * (j0 / reg)
                                            k01 = k01 + j0 * (j1 / reg)
                                            k11 = k11 + j1 * (j1 / reg)
                                            c0 = c0 + j0 * (rows[w, k, 3] / reg)
                                            c1 = c1 + j1 * (rows[w, k, 3] / reg)
                                        other_bits = other_bits // 2
                                    if not tx.isfinite(k00) or k00 <= 0.0:
                                        meta[w, 0] = 4
                                    else:
                                        kl10 = k01 / k00
                                        ks11 = k11 - k01 * kl10
                                        if not tx.isfinite(ks11) or ks11 <= 0.0:
                                            meta[w, 0] = 4
                                        else:
                                            a1 = (c1 - kl10 * c0) / ks11
                                            a0 = (c0 - k01 * a1) / k00
                                            j0 = row_j[w, r, 0]
                                            j1 = row_j[w, r, 1]
                                            z1 = (j1 - kl10 * j0) / ks11
                                            z0 = (j0 - k01 * z1) / k00
                                            diagonal = rows[w, r, 4] + j0 * z0 + j1 * z1
                                            raw = 0.0
                                            if not tx.isfinite(diagonal) or diagonal <= 0.0:
                                                meta[w, 0] = 4
                                            else:
                                                raw = (
                                                    rows[w, r, 3] - (j0 * a0 + j1 * a1)
                                                ) / diagonal
                                                if not tx.isfinite(raw):
                                                    meta[w, 0] = 2
                                            rows[w, r, 9] = tx.max(0.0, raw)
                                bits = bits // 2
                            # Columns of -h Qarea' are already in (Jacobian-A).
                            # Apply H^-1 FIRST, then M: never reverse this order
                            # or commute the generally nonsymmetric area term.
                            for j in range(2):
                                t0 = matrices[w, 2, 0, j] - matrices[w, 0, 0, j]
                                t1 = matrices[w, 2, 1, j] - matrices[w, 0, 1, j]
                                z1 = (t1 - chosen_l10 * t0) / chosen_s11
                                z0 = (t0 - chosen_h01 * z1) / chosen_h00
                                matrices[w, 2, 0, j] = matrices[w, 0, 0, j] + m00 * z0 + m01 * z1
                                matrices[w, 2, 1, j] = matrices[w, 0, 1, j] + m01 * z0 + m11 * z1
                    # All native rows use the same candidate's complete area load.
                    native_ok = 1
                    for r in range(nr):
                        reg = rows[w, r, 4]
                        if not tx.isfinite(reg) or reg <= 0.0:
                            meta[w, 0] = 2
                        else:
                            raw = 0.0
                            if native_physical_acceleration == 1:
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
                            elif coupled_mode:
                                # Computed in force coordinates by the coupled
                                # physical-M face elimination above.
                                raw = rows[w, r, 9]
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
                    if native_physical_acceleration == 4:
                        # Evaluate the actual stored point/row forces together.
                        # Q and (applied+Q) are independently rounded from the
                        # same compensated sum; never add applied to rounded Q.
                        # Probe zero is the unchanged force vector. A correction
                        # changes exactly one positive row by one FP32 neighbor.
                        polish_active = 1
                        polished_score = 0.0
                        for sweep in range(4):
                            if polish_active == 1:
                                base_score = 0.0
                                best_score = 0.0
                                best_row = -1
                                best_force = 0.0
                                best_q0 = 0.0
                                best_q1 = 0.0
                                best_net0 = 0.0
                                best_net1 = 0.0
                                for probe in range(1 + 2 * nr):
                                    if probe == 0 or base_score > 1.0:
                                        probe_row = -1
                                        probe_force = 0.0
                                        probe_valid = 1
                                        if probe > 0:
                                            probe_row = (probe - 1) // 2
                                            old_force = rows[w, probe_row, 9]
                                            if old_force <= 0.0:
                                                probe_valid = 0
                                            else:
                                                # f*2^-24 lies in [half,one) ULP.
                                                # A half-ULP tie can return the same
                                                # value. Double the offset in that
                                                # direction to reach its neighbor.
                                                spacing = tx.max(
                                                    1.401298464324817e-45,
                                                    tx.mul_rn(old_force, 5.960464477539063e-8),
                                                )
                                                if probe % 2 == 1:
                                                    probe_force = tx.sub_rn(old_force, spacing)
                                                    if probe_force == old_force:
                                                        probe_force = tx.sub_rn(
                                                            old_force, tx.add_rn(spacing, spacing)
                                                        )
                                                else:
                                                    probe_force = tx.add_rn(old_force, spacing)
                                                    if probe_force == old_force:
                                                        probe_force = tx.add_rn(
                                                            old_force, tx.add_rn(spacing, spacing)
                                                        )
                                                if (
                                                    not tx.isfinite(probe_force)
                                                    or probe_force < 0.0
                                                    or probe_force == old_force
                                                ):
                                                    probe_valid = 0
                                        if probe_valid == 1:
                                            load_sum0 = area_sum0
                                            load_sum1 = area_sum1
                                            for r in range(nr):
                                                force = rows[w, r, 9]
                                                if r == probe_row:
                                                    force = probe_force
                                                load_sum0 = _add_product(
                                                    load_sum0, row_j[w, r, 0], force
                                                )
                                                load_sum1 = _add_product(
                                                    load_sum1, row_j[w, r, 1], force
                                                )
                                            q0 = tx.add_rn(load_sum0[0], load_sum0[1])
                                            q1 = tx.add_rn(load_sum1[0], load_sum1[1])
                                            net_sum0 = _add_product(
                                                load_sum0, 1.0, source_applied[w, 0]
                                            )
                                            net_sum1 = _add_product(
                                                load_sum1, 1.0, source_applied[w, 1]
                                            )
                                            net0 = tx.add_rn(net_sum0[0], net_sum0[1])
                                            net1 = tx.add_rn(net_sum1[0], net_sum1[1])
                                            # Exactly the native tail after the new
                                            # rounded net-applied publication point.
                                            f0 = net0 + source_passive[w, 0]
                                            f0 = f0 - source_bias[w, 0]
                                            f1 = net1 + source_passive[w, 1]
                                            f1 = f1 - source_bias[w, 1]
                                            forward0 = f0 / published_l00
                                            forward1 = (
                                                f1 - published_l10 * forward0
                                            ) / published_l11
                                            a1 = forward1 / published_l11
                                            a0 = (forward0 - published_l10 * a1) / published_l00
                                            score = 0.0
                                            if (
                                                not tx.isfinite(q0)
                                                or not tx.isfinite(q1)
                                                or not tx.isfinite(a0)
                                                or not tx.isfinite(a1)
                                            ):
                                                probe_valid = 0
                                            for r in range(nr):
                                                force = rows[w, r, 9]
                                                if r == probe_row:
                                                    force = probe_force
                                                defect_sum = tx.vec2f(-rows[w, r, 3], 0.0)
                                                defect_sum = _add_product(
                                                    defect_sum, row_j[w, r, 0], a0
                                                )
                                                defect_sum = _add_product(
                                                    defect_sum, row_j[w, r, 1], a1
                                                )
                                                defect_sum = _add_product(
                                                    defect_sum, rows[w, r, 4], force
                                                )
                                                gradient = tx.add_rn(defect_sum[0], defect_sum[1])
                                                violation = tx.abs(gradient)
                                                if force == 0.0:
                                                    violation = tx.max(0.0, -gradient)
                                                score = tx.max(
                                                    score,
                                                    violation
                                                    / (
                                                        1e-5
                                                        * tx.max(1.0, tx.abs(rows[w, r, 4] * force))
                                                    ),
                                                )
                                                score = tx.max(
                                                    score,
                                                    -gradient
                                                    / (2e-5 * (1.0 + tx.abs(rows[w, r, 3]))),
                                                )
                                                if not tx.isfinite(gradient):
                                                    probe_valid = 0
                                            if probe == 0:
                                                base_score = score
                                                if probe_valid == 0:
                                                    meta[w, 0] = 2
                                            if probe_valid == 1 and (
                                                probe == 0 or score < best_score
                                            ):
                                                best_score = score
                                                best_row = probe_row
                                                best_force = probe_force
                                                best_q0 = q0
                                                best_q1 = q1
                                                best_net0 = net0
                                                best_net1 = net1
                                if best_row >= 0:
                                    rows[w, best_row, 9] = best_force
                                vectors[w, 7, 0] = best_q0
                                vectors[w, 7, 1] = best_q1
                                net_applied0 = best_net0
                                net_applied1 = best_net1
                                polished_score = best_score
                                if best_score <= 1.0 or best_row < 0:
                                    polish_active = 0
                        if polished_score > 1.0:
                            native_ok = 0
                    if nr > 0:
                        if native_physical_acceleration == 1:
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
                        elif coupled_mode:
                            # Bounded corrections in stored force coordinates.
                            # Use the physical Schur diagonal R+J M^-1 J; rebuild
                            # both generalized components in publication order.
                            force_sweeps = 2
                            if live_source_mode:
                                force_sweeps = 8
                            if native_physical_acceleration == 4:
                                # Only the bounded neighbor search above may alter
                                # a mode-4 face force, using its true net load.
                                force_sweeps = 0
                            for sweep in range(force_sweeps):
                                for r in range(nr):
                                    f0 = (
                                        inputs[w, 2, 0]
                                        - inputs[w, 0, 0] * inputs[w, 1, 0]
                                        + vectors[w, 7, 0]
                                    )
                                    f1 = (
                                        inputs[w, 2, 1]
                                        - inputs[w, 0, 1] * inputs[w, 1, 1]
                                        + vectors[w, 7, 1]
                                    )
                                    a1 = (f1 - ml10 * f0) / ms11
                                    a0 = (f0 - m01 * a1) / m00
                                    if live_source_mode:
                                        # apply_load stores (applied+Q), then the
                                        # native tail adds passive and subtracts
                                        # bias. Preserve those FP32 boundaries.
                                        f0 = source_applied[w, 0] + vectors[w, 7, 0]
                                        f0 = f0 + source_passive[w, 0]
                                        f0 = f0 - source_bias[w, 0]
                                        f1 = source_applied[w, 1] + vectors[w, 7, 1]
                                        f1 = f1 + source_passive[w, 1]
                                        f1 = f1 - source_bias[w, 1]
                                        forward0 = f0 / published_l00
                                        forward1 = (f1 - published_l10 * forward0) / published_l11
                                        a1 = forward1 / published_l11
                                        a0 = (forward0 - published_l10 * a1) / published_l00
                                    j0 = row_j[w, r, 0]
                                    j1 = row_j[w, r, 1]
                                    old_force = rows[w, r, 9]
                                    gradient = (
                                        rows[w, r, 4] * old_force
                                        - rows[w, r, 3]
                                        + j0 * a0
                                        + j1 * a1
                                    )
                                    if live_source_mode:
                                        projected = j0 * a0 + j1 * a1
                                        gradient = (
                                            projected + rows[w, r, 4] * old_force - rows[w, r, 3]
                                        )
                                    defect = tx.abs(gradient)
                                    if old_force == 0.0:
                                        defect = tx.max(0.0, -gradient)
                                    needs_correction = defect > 1e-5 * tx.max(
                                        1.0, tx.abs(rows[w, r, 4] * old_force)
                                    )
                                    if live_source_mode and gradient < (
                                        -2e-5 * (1.0 + tx.abs(rows[w, r, 3]))
                                    ):
                                        needs_correction = True
                                    if needs_correction:
                                        z1 = (j1 - ml10 * j0) / ms11
                                        z0 = (j0 - m01 * z1) / m00
                                        diagonal = rows[w, r, 4] + j0 * z0 + j1 * z1
                                        if not tx.isfinite(diagonal) or diagonal <= 0.0:
                                            meta[w, 0] = 4
                                        else:
                                            force = tx.max(0.0, old_force - gradient / diagonal)
                                            if (
                                                live_source_mode
                                                and force == old_force
                                                and old_force > 0.0
                                            ):
                                                # If the Newton correction is
                                                # below one force ULP, move in
                                                # ITS direction; never add an
                                                # unconditional positive bias.
                                                step = 1.1920928955078125e-7 * old_force
                                                if gradient < 0.0:
                                                    force = old_force + step
                                                elif gradient > 0.0:
                                                    force = tx.max(0.0, old_force - step)
                                            if not tx.isfinite(force):
                                                meta[w, 0] = 2
                                            rows[w, r, 9] = force
                                            load0 = scalar_area_load
                                            load1 = coupled_area_load
                                            for k in range(nr):
                                                load0 = load0 + row_j[w, k, 0] * rows[w, k, 9]
                                                load1 = load1 + row_j[w, k, 1] * rows[w, k, 9]
                                            vectors[w, 7, 0] = load0
                                            vectors[w, 7, 1] = load1
                            # KKT below sees M^-1 times the ACTUAL rounded forces,
                            # not the acceleration returned by the face solve.
                            f0 = (
                                inputs[w, 2, 0]
                                - inputs[w, 0, 0] * inputs[w, 1, 0]
                                + vectors[w, 7, 0]
                            )
                            f1 = (
                                inputs[w, 2, 1]
                                - inputs[w, 0, 1] * inputs[w, 1, 1]
                                + vectors[w, 7, 1]
                            )
                            vectors[w, 5, 1] = (f1 - ml10 * f0) / ms11
                            vectors[w, 5, 0] = (f0 - m01 * vectors[w, 5, 1]) / m00
                            if live_source_mode:
                                # Final eligibility uses the same RHS grouping
                                # and Cholesky solves as the published qacc.
                                f0 = source_applied[w, 0] + vectors[w, 7, 0]
                                if native_physical_acceleration == 4:
                                    f0 = net_applied0
                                f0 = f0 + source_passive[w, 0]
                                f0 = f0 - source_bias[w, 0]
                                f1 = source_applied[w, 1] + vectors[w, 7, 1]
                                if native_physical_acceleration == 4:
                                    f1 = net_applied1
                                f1 = f1 + source_passive[w, 1]
                                f1 = f1 - source_bias[w, 1]
                                forward0 = f0 / published_l00
                                forward1 = (f1 - published_l10 * forward0) / published_l11
                                vectors[w, 5, 1] = forward1 / published_l11
                                vectors[w, 5, 0] = (
                                    forward0 - published_l10 * vectors[w, 5, 1]
                                ) / published_l00
                            if not tx.isfinite(vectors[w, 5, 0]) or not tx.isfinite(
                                vectors[w, 5, 1]
                            ):
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
                            if live_source_mode:
                                projected = 0.0
                                for i in range(nv):
                                    projected = projected + row_j[w, r, i] * vectors[w, 5, i]
                                gradient = projected + compliance_term - rows[w, r, 3]
                            violation = tx.abs(gradient)
                            if reaction == rows[w, r, 7]:
                                violation = tx.max(0.0, -gradient)
                            elif rows[w, r, 12] == 1.0 and reaction == rows[w, r, 8]:
                                violation = tx.max(0.0, gradient)
                            if not tx.isfinite(gradient) or violation > 1e-5 * tx.max(
                                1.0, tx.abs(compliance_term)
                            ):
                                native_ok = 0
                            if live_source_mode and gradient < (
                                -2e-5 * (1.0 + tx.abs(rows[w, r, 3]))
                            ):
                                native_ok = 0
                    actual_residual0 = 0.0
                    actual_residual1 = 0.0
                    norm2 = 0.0
                    for i in range(nv):
                        value = -h * vectors[w, 7, i]
                        if coupled_mode:
                            # The original momentum RHS, independent of the
                            # rounded free-velocity solve used to initialize v.
                            value = -h * (
                                inputs[w, 2, i]
                                - inputs[w, 0, i] * inputs[w, 1, i]
                                + vectors[w, 7, i]
                            )
                            if live_source_mode:
                                physical_load = source_applied[w, i] + vectors[w, 7, i]
                                physical_load = physical_load + source_passive[w, i]
                                physical_load = physical_load - source_bias[w, i]
                                value = -h * physical_load
                        for j in range(nv):
                            if coupled_mode:
                                value = value + matrices[w, 0, i, j] * (
                                    vectors[w, 2, j] - inputs[w, 1, j]
                                )
                            else:
                                value = value + matrices[w, 0, i, j] * (
                                    vectors[w, 2, j] - vectors[w, 0, j]
                                )
                            if not tx.isfinite(matrices[w, 2, i, j]):
                                meta[w, 0] = 2
                        if native_physical_acceleration == 4:
                            # Newton follows the stable face equation M*a_face,
                            # not a cancellation-prone reconstructed rounded Q.
                            # The actual rounded publication gets its OWN gates.
                            momentum = tx.vec2f(0.0, 0.0)
                            face_load = tx.vec2f(0.0, 0.0)
                            for j in range(nv):
                                momentum = _add_product(
                                    momentum,
                                    matrices[w, 0, i, j],
                                    vectors[w, 2, j] - inputs[w, 1, j],
                                )
                            face_load = _add_product(face_load, mass[w, i, 0], face_a0)
                            face_load = _add_product(face_load, mass[w, i, 1], face_a1)
                            residual = _add_product(momentum, -h, face_load[0])
                            residual = _add_product(residual, -h, face_load[1])
                            value = tx.add_rn(residual[0], residual[1])
                            physical_load = net_applied0
                            if i == 1:
                                physical_load = net_applied1
                            physical_load = physical_load + source_passive[w, i]
                            physical_load = physical_load - source_bias[w, i]
                            actual = _add_product(momentum, -h, physical_load)
                            if i == 0:
                                actual_residual0 = tx.add_rn(actual[0], actual[1])
                            else:
                                actual_residual1 = tx.add_rn(actual[0], actual[1])
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
                    if native_physical_acceleration == 4:
                        # Preserve the old actual-load norm and per-component
                        # A^-1 residual bounds independently of the face norm.
                        u0 = actual_residual0 / matrices[w, 1, 0, 0]
                        u1 = (actual_residual1 - matrices[w, 1, 1, 0] * u0) / matrices[w, 1, 1, 1]
                        actual_norm2 = u0 * u0 + u1 * u1
                        correction1 = u1 / matrices[w, 1, 1, 1]
                        correction0 = (u0 - matrices[w, 1, 1, 0] * correction1) / matrices[
                            w, 1, 0, 0
                        ]
                        if (
                            not tx.isfinite(actual_norm2)
                            or actual_norm2 > scalars[w, 1] * scalars[w, 1]
                            or not tx.isfinite(correction0)
                            or not tx.isfinite(correction1)
                            or tx.abs(correction0) > 1e-5 * (1.0 + tx.abs(vectors[w, 2, 0]))
                            or tx.abs(correction1) > 1e-5 * (1.0 + tx.abs(vectors[w, 2, 1]))
                        ):
                            velocity_ok = 0
                    if native_physical_acceleration == 1 and nr > 0:
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
                    elif coupled_mode and nr > 0:
                        # Physical acceleration implied by the two output DOFs:
                        # a_motion=M^-1 A (v-v0)/h, in that matrix order.
                        dv0 = (vectors[w, 2, 0] - inputs[w, 1, 0]) / h
                        dv1 = (vectors[w, 2, 1] - inputs[w, 1, 1]) / h
                        f0 = matrices[w, 0, 0, 0] * dv0 + matrices[w, 0, 0, 1] * dv1
                        f1 = matrices[w, 0, 1, 0] * dv0 + matrices[w, 0, 1, 1] * dv1
                        a1 = (f1 - ml10 * f0) / ms11
                        a0 = (f0 - m01 * a1) / m00
                        uncondensed0 = vectors[w, 3, 0]
                        uncondensed1 = vectors[w, 3, 1]
                        if native_physical_acceleration == 4:
                            uncondensed0 = actual_residual0
                            uncondensed1 = actual_residual1
                        for r in range(nr):
                            j0 = row_j[w, r, 0]
                            j1 = row_j[w, r, 1]
                            at_v = tx.max(
                                0.0, (rows[w, r, 3] - (j0 * a0 + j1 * a1)) / rows[w, r, 4]
                            )
                            if native_physical_acceleration == 4:
                                slack_sum = tx.vec2f(rows[w, r, 3], 0.0)
                                slack_sum = _add_product(slack_sum, -j0, a0)
                                slack_sum = _add_product(slack_sum, -j1, a1)
                                at_v = tx.max(
                                    0.0,
                                    tx.div_rn(tx.add_rn(slack_sum[0], slack_sum[1]), rows[w, r, 4]),
                                )
                            force_delta = at_v - rows[w, r, 9]
                            # Same force-unit envelope as the native receipt's
                            # physical-M force law; row cancellation cannot hide it.
                            if not tx.isfinite(force_delta) or tx.abs(force_delta) > (
                                0.002 + 0.002 * tx.abs(at_v)
                            ):
                                velocity_ok = 0
                            uncondensed0 = uncondensed0 - h * j0 * force_delta
                            uncondensed1 = uncondensed1 - h * j1 * force_delta
                        # Check the original equation in the same A^-1 norm as
                        # the condensed residual, plus its actual velocity error.
                        u0 = uncondensed0 / matrices[w, 1, 0, 0]
                        u1 = (uncondensed1 - matrices[w, 1, 1, 0] * u0) / matrices[w, 1, 1, 1]
                        original_norm2 = u0 * u0 + u1 * u1
                        correction1 = u1 / matrices[w, 1, 1, 1]
                        correction0 = (u0 - matrices[w, 1, 1, 0] * correction1) / matrices[
                            w, 1, 0, 0
                        ]
                        if (
                            not tx.isfinite(a0)
                            or not tx.isfinite(a1)
                            or not tx.isfinite(original_norm2)
                            or original_norm2 > scalars[w, 1] * scalars[w, 1]
                            or not tx.isfinite(correction0)
                            or not tx.isfinite(correction1)
                            or tx.abs(correction0) > 1e-5 * (1.0 + tx.abs(vectors[w, 2, 0]))
                            or tx.abs(correction1) > 1e-5 * (1.0 + tx.abs(vectors[w, 2, 1]))
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
                            if native_physical_acceleration == 4:
                                # Bank 4 has no remaining Newton-direction role.
                                # Publish the SAME net used by all actual gates;
                                # passive and bias remain owned by the native tail.
                                vectors[w, 4, 0] = net_applied0
                                vectors[w, 4, 1] = net_applied1
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
    sample_capacity=128,
    native_rows=None,
    native_physical_acceleration=0,
    native_source_forces=None,
    native_limit_count=1,
    **launch,
):
    """Keep the standalone ABI; live owners additionally bound A^-1 residual.

    ``native_physical_acceleration`` is 0 for the legacy joint route, 1 for
    one damped scalar, 2 for two coupled damped DOFs, or 3 for the same two
    DOFs with one or two joint limits. Modes 1/2 admit up to four native rows.
    Mode 3's explicit ``native_limit_count`` is 1 (default) or 2, with at most
    one active side per declared limit: four contact edges plus that count,
    at most five/six rows and 32/64 actual active-set masks respectively.
    Mode 4 independently admits two condim-3 supports without limits, up to
    eight native rows and 256 actual masks. All rows participate in one physical
    solve with the same candidate area load. Its worst-case face search is
    O(nr * 2**nr) per candidate; finding a face skips all later face solves.
    Mode 4 follows a compensated face residual while separately checking actual
    rounded loads and motion. On success, vector bank 4 stores compensated
    applied+constraint (excluding passive/bias); bank 7 still stores constraint
    alone. Its owner must publish bank 4 directly, never add applied to bank 7.
    Each candidate's area load eliminates the
    native unilateral rows in physical acceleration space, without modifying
    native R, aref, or force units. The owner
    supplies all compact rows and enforces its combined area/native capacity.
    The default retains the existing undamped joint-row route.

    Modes 3/4 require ``native_source_forces=(applied, passive, bias)`` from
    the same live entry. They are read-only inputs for force correction and
    eligibility against the physical acceleration produced by the native tail.
    """
    if type(require_velocity_agreement) is not bool:
        raise ValueError("require_velocity_agreement must be bool")
    if type(native_physical_acceleration) not in (
        int,
        bool,
    ) or native_physical_acceleration not in (0, 1, 2, 3, 4):
        raise ValueError("native_physical_acceleration must be mode 0, 1, 2, 3, or 4")
    if native_physical_acceleration and native_rows is None:
        raise ValueError("physical native acceleration requires native row storage")
    if native_physical_acceleration in (3, 4) and native_source_forces is None:
        raise ValueError(
            f"mode {native_physical_acceleration} requires live native applied/passive/bias buffers"
        )
    if native_physical_acceleration not in (3, 4) and native_source_forces is not None:
        raise ValueError("native_source_forces is supported only in mode 3 or 4")
    if type(native_limit_count) is not int or native_limit_count not in (1, 2):
        raise ValueError("native_limit_count must be integer 1 or 2")
    if native_physical_acceleration != 3 and native_limit_count != 1:
        raise ValueError("nondefault native_limit_count is supported only in mode 3")
    if len(mass.shape) != 3 or mass.shape[1] not in (6, 8):
        raise ValueError("invalid packed area-friction mass layout")
    width = mass.shape[1]
    layout = scratch_layout(capacity, width, sample_capacity=sample_capacity)
    for name, array in zip(FRICTION_FIELDS, (sizes, mass, *buffers), strict=True):
        if array.shape != layout[name][0]:
            raise ValueError(f"invalid packed area-friction {name} layout")
    # The rank-2 FP32 scalars buffer fills the three unused source slots in older modes.
    scalars = buffers[FRICTION_FIELDS.index("scalars") - 2]
    source_applied = source_passive = source_bias = scalars
    if native_source_forces is not None:
        source_applied, source_passive, source_bias = native_source_forces
        for source in (source_applied, source_passive, source_bias):
            if len(source.shape) != 2 or source.shape[0] != capacity or source.shape[1] < width:
                raise ValueError("invalid native applied/passive/bias storage")
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
        sample_capacity,
        row_capacity,
        int(native_physical_acceleration),
        native_limit_count,
        rows,
        jacobian,
        counts,
        source_applied,
        source_passive,
        source_bias,
        **launch,
    )
