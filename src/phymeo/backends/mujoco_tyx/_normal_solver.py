# Copyright 2025 The Newton Developers
# Copyright 2026 Phymeo contributors
# Apache-2.0; see LICENSE and NOTICE.
# The native frictionless objective/efc fields follow MuJoCo-Warp 0.0.2.
# The bounded dual active-set algorithm below is a new TYX implementation,
# not the native Newton/CG iteration, line search or warm-start algorithm.

"""Joint normal QP with physical M; integration remains the existing Euler solve."""

import tyx as tx


@tx.kernel
def solve(
    count: tx.i32,
    nv: tx.i32,
    nslot: tx.i32,
    budget: tx.i32,
    tolerance: tx.f32,
    mass: tx.ndarray(tx.f32, 3),
    bias: tx.ndarray(tx.f32, 2),
    passive: tx.ndarray(tx.f32, 2),
    applied: tx.ndarray(tx.f32, 2),
    jacobian: tx.ndarray(tx.f32, 3),
    rows: tx.ndarray(tx.f32, 3),
    factor: tx.ndarray(tx.f32, 3),
    mass_work: tx.ndarray(tx.f32, 2),
    smooth: tx.ndarray(tx.f32, 2),
    response: tx.ndarray(tx.f32, 3),
    dual: tx.ndarray(tx.f32, 3),
    linear: tx.ndarray(tx.f32, 2),
    force: tx.ndarray(tx.f32, 2),
    gradient: tx.ndarray(tx.f32, 2),
    working: tx.ndarray(tx.i32, 2),
    face_factor: tx.ndarray(tx.f32, 3),
    face_forward: tx.ndarray(tx.f32, 2),
    trial: tx.ndarray(tx.f32, 2),
    constraint: tx.ndarray(tx.f32, 2),
    total_applied: tx.ndarray(tx.f32, 2),
    diagnostics: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        for d in range(nv):
            smooth[w, d] = 0.0
            constraint[w, d] = 0.0
            total_applied[w, d] = applied[w, d]
        for r in range(nslot):
            force[w, r] = 0.0
            linear[w, r] = 0.0
            gradient[w, r] = 0.0
            working[w, r] = 0
            trial[w, r] = 0.0
            for d in range(nv):
                response[w, r, d] = 0.0
            for s in range(nslot):
                dual[w, r, s] = 0.0
        diagnostics[w, 0] = 0.0
        diagnostics[w, 1] = 0.0
        diagnostics[w, 2] = 0.0
        diagnostics[w, 3] = 0.0
        diagnostics[w, 6] = 0.0
        diagnostics[w, 7] = budget
        if status[w] == 0:
            # One physical M factorization; never substitute the Euler M+hD matrix.
            for i in range(nv):
                for j in range(i + 1):
                    value = mass[w, i, j]
                    for k in range(j):
                        value = value - factor[w, i, k] * factor[w, j, k]
                    if i == j:
                        if not tx.isfinite(value) or value <= 0.0:
                            status[w] = 4
                            value = 1.0
                        factor[w, i, j] = tx.sqrt(value)
                    else:
                        factor[w, i, j] = value / factor[w, j, j]
            for i in range(nv):
                value = applied[w, i] + passive[w, i] - bias[w, i]
                for j in range(i):
                    value = value - factor[w, i, j] * mass_work[w, j]
                mass_work[w, i] = value / factor[w, i, i]
            for reverse in range(nv):
                i = nv - 1 - reverse
                value = mass_work[w, i]
                for j in range(i + 1, nv):
                    value = value - factor[w, j, i] * smooth[w, j]
                smooth[w, i] = value / factor[w, i, i]
                if not tx.isfinite(smooth[w, i]):
                    status[w] = 2
            for r in range(nslot):
                if rows[w, r, 0] == 1.0:
                    for i in range(nv):
                        value = jacobian[w, r, i]
                        for j in range(i):
                            value = value - factor[w, i, j] * mass_work[w, j]
                        mass_work[w, i] = value / factor[w, i, i]
                    for reverse in range(nv):
                        i = nv - 1 - reverse
                        value = mass_work[w, i]
                        for j in range(i + 1, nv):
                            value = value - factor[w, j, i] * response[w, r, j]
                        response[w, r, i] = value / factor[w, i, i]
                    value = 0.0
                    for i in range(nv):
                        value = value + jacobian[w, r, i] * smooth[w, i]
                    rows[w, r, 24] = value
                    linear[w, r] = value - rows[w, r, 6]
            for r in range(nslot):
                for s in range(r + 1):
                    value = 0.0
                    for d in range(nv):
                        value = value + jacobian[w, r, d] * response[w, s, d]
                    if r == s:
                        value = value + rows[w, r, 4]
                    dual[w, r, s] = value
                    dual[w, s, r] = value
                    if not tx.isfinite(value):
                        status[w] = 2
            # Feasible-face active set. All contact rows share this same QP.
            # A blocking face change consumes a solve just like an accepted face.
            done = 0
            refine_face = 0
            for iteration in range(budget):
                if done == 0 and status[w] == 0:
                    residual = 0.0
                    scale = 1.0
                    entering = -1
                    lowest = 0.0
                    for r in range(nslot):
                        if rows[w, r, 1] == 1.0:
                            value = linear[w, r]
                            row_scale = tx.abs(linear[w, r])
                            for s in range(nslot):
                                value = value + dual[w, r, s] * force[w, s]
                                row_scale = row_scale + tx.abs(dual[w, r, s] * force[w, s])
                            if not tx.isfinite(row_scale):
                                status[w] = 2
                            gradient[w, r] = value
                            violation = tx.max(0.0, -value)
                            if force[w, r] > 0.0:
                                violation = tx.abs(value)
                            residual = tx.max(residual, violation)
                            scale = tx.max(scale, row_scale)
                            if working[w, r] == 0 and value < lowest:
                                lowest = value
                                entering = r
                    if residual <= tolerance * scale:
                        done = 1
                    else:
                        if refine_face == 0:
                            if entering >= 0:
                                working[w, entering] = 1
                            else:
                                # No improving inactive coordinate and stationarity still fails.
                                status[w] = 5
                        if status[w] == 0:
                            diagnostics[w, 0] = diagnostics[w, 0] + 1.0
                            for i in range(nslot):
                                for j in range(i + 1):
                                    value = 0.0
                                    if working[w, i] == 1 and working[w, j] == 1:
                                        value = dual[w, i, j]
                                    elif i == j:
                                        value = 1.0
                                    for k in range(j):
                                        value = value - face_factor[w, i, k] * face_factor[w, j, k]
                                    if i == j:
                                        if not tx.isfinite(value) or value <= 0.0:
                                            status[w] = 6
                                            value = 1.0
                                        face_factor[w, i, j] = tx.sqrt(value)
                                    else:
                                        face_factor[w, i, j] = value / face_factor[w, j, j]
                            for i in range(nslot):
                                value = 0.0
                                if working[w, i] == 1:
                                    value = -linear[w, i]
                                for j in range(i):
                                    value = value - face_factor[w, i, j] * face_forward[w, j]
                                face_forward[w, i] = value / face_factor[w, i, i]
                            for reverse in range(nslot):
                                i = nslot - 1 - reverse
                                value = face_forward[w, i]
                                for j in range(i + 1, nslot):
                                    value = value - face_factor[w, j, i] * trial[w, j]
                                trial[w, i] = value / face_factor[w, i, i]
                                if not tx.isfinite(trial[w, i]):
                                    status[w] = 2
                            alpha = 1.0
                            blocking = -1
                            for r in range(nslot):
                                if working[w, r] == 1 and trial[w, r] <= 0.0:
                                    denominator = force[w, r] - trial[w, r]
                                    candidate = 0.0
                                    if denominator > 0.0:
                                        candidate = force[w, r] / denominator
                                    if candidate < alpha or blocking < 0:
                                        alpha = candidate
                                        blocking = r
                            if blocking >= 0:
                                for r in range(nslot):
                                    force[w, r] = tx.max(
                                        0.0, force[w, r] + alpha * (trial[w, r] - force[w, r])
                                    )
                                    if force[w, r] == 0.0:
                                        working[w, r] = 0
                                force[w, blocking] = 0.0
                                working[w, blocking] = 0
                                refine_face = 1
                            else:
                                for r in range(nslot):
                                    force[w, r] = trial[w, r]
                                refine_face = 0
            # Recheck the final candidate, including the last permitted face solve.
            residual = 0.0
            scale = 1.0
            min_force = 0.0
            first_force = 1
            for r in range(nslot):
                value = 0.0
                if rows[w, r, 1] == 1.0:
                    value = linear[w, r]
                    row_scale = tx.abs(linear[w, r])
                    for s in range(nslot):
                        value = value + dual[w, r, s] * force[w, s]
                        row_scale = row_scale + tx.abs(dual[w, r, s] * force[w, s])
                    violation = tx.max(0.0, -value)
                    if force[w, r] > 0.0:
                        violation = tx.abs(value)
                    residual = tx.max(residual, violation)
                    scale = tx.max(scale, row_scale)
                    if not tx.isfinite(row_scale):
                        status[w] = 2
                if first_force == 1:
                    min_force = force[w, r]
                    first_force = 0
                else:
                    min_force = tx.min(min_force, force[w, r])
                gradient[w, r] = value
                rows[w, r, 3] = force[w, r]
                rows[w, r, 25] = value
                rows[w, r, 26] = force[w, r] * value
                if (
                    not tx.isfinite(value)
                    or not tx.isfinite(force[w, r])
                    or not tx.isfinite(rows[w, r, 26])
                ):
                    status[w] = 2
                for d in range(nv):
                    constraint[w, d] = constraint[w, d] + jacobian[w, r, d] * force[w, r]
            diagnostics[w, 1] = residual / scale
            diagnostics[w, 2] = residual
            diagnostics[w, 3] = min_force
            if status[w] == 0:
                if diagnostics[w, 1] <= tolerance and min_force >= 0.0:
                    diagnostics[w, 6] = 1.0
                else:
                    status[w] = 5
            for d in range(nv):
                total_applied[w, d] = applied[w, d] + constraint[w, d]
                if not tx.isfinite(constraint[w, d]) or not tx.isfinite(total_applied[w, d]):
                    status[w] = 2
