# Copyright 2026 Phymeo contributors
# Apache-2.0; see LICENSE and NOTICE.
# MuJoCo 3.3.7 defines the assembled box objective. This is an independent
# bounded active-face algorithm, not native Newton/CG or warm-start iteration.

"""Mixed box constraints, using physical-M primal elimination on each face."""

import tyx as tx


@tx.kernel
def solve(
    count: tx.i32,
    nv: tx.i32,
    efc_capacity: tx.i32,
    budget: tx.i32,
    tolerance: tx.f32,
    mass: tx.ndarray(tx.f32, 3),
    bias: tx.ndarray(tx.f32, 2),
    passive: tx.ndarray(tx.f32, 2),
    applied: tx.ndarray(tx.f32, 2),
    efc_data: tx.ndarray(tx.f32, 3),
    efc_J: tx.ndarray(tx.f32, 3),
    counts: tx.ndarray(tx.i32, 2),
    physical_factor: tx.ndarray(tx.f32, 3),
    mass_work: tx.ndarray(tx.f32, 2),
    smooth_acc: tx.ndarray(tx.f32, 2),
    face_matrix: tx.ndarray(tx.f32, 3),
    face_factor: tx.ndarray(tx.f32, 3),
    face_rhs: tx.ndarray(tx.f32, 2),
    face_forward: tx.ndarray(tx.f32, 2),
    trial_acc: tx.ndarray(tx.f32, 2),
    trial_force: tx.ndarray(tx.f32, 2),
    working: tx.ndarray(tx.i32, 2),
    previous_working: tx.ndarray(tx.i32, 2),
    previous_force: tx.ndarray(tx.f32, 2),
    constraint: tx.ndarray(tx.f32, 2),
    total_applied: tx.ndarray(tx.f32, 2),
    physical_acc: tx.ndarray(tx.f32, 2),
    diagnostics: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        width = 8
        if nv > 8 and nv <= 35:
            width = nv
        row_storage = efc_capacity
        if row_storage < 1:
            row_storage = 1
        if row_storage > 512:
            row_storage = 512
        for i in range(width):
            mass_work[w, i] = 0.0
            smooth_acc[w, i] = 0.0
            face_rhs[w, i] = 0.0
            face_forward[w, i] = 0.0
            trial_acc[w, i] = 0.0
            constraint[w, i] = 0.0
            total_applied[w, i] = 0.0
            physical_acc[w, i] = 0.0
            for j in range(width):
                physical_factor[w, i, j] = 0.0
                face_matrix[w, i, j] = 0.0
                face_factor[w, i, j] = 0.0
        for r in range(row_storage):
            efc_data[w, r, 9] = 0.0
            efc_data[w, r, 10] = 0.0
            trial_force[w, r] = 0.0
            working[w, r] = 0
            previous_working[w, r] = 0
            previous_force[w, r] = 0.0
        for d in range(10):
            diagnostics[w, d] = 0.0
        diagnostics[w, 7] = budget
        nrow = counts[w, 2]
        diagnostics[w, 4] = nrow
        if status[w] == 0:
            if (
                nv < 1
                or nv > 35
                or efc_capacity < 0
                or efc_capacity > 512
                or budget < 1
                or budget > 512
                or not tx.isfinite(tolerance)
                or tolerance <= 0.0
            ):
                status[w] = 2
            if nrow < 0 or nrow > efc_capacity or counts[w, 0] > 128:
                status[w] = 1
            if counts[w, 0] < 0 or counts[w, 1] < 0 or counts[w, 1] > counts[w, 0]:
                status[w] = 2
        if nrow < 0 or nrow > efc_capacity:
            nrow = 0
        if status[w] == 0:
            # Only actual compact rows participate. All unused output is cleared.
            for r in range(nrow):
                for d in range(14):
                    if not tx.isfinite(efc_data[w, r, d]):
                        status[w] = 2
                lower = efc_data[w, r, 7]
                upper = efc_data[w, r, 8]
                upper_mask = efc_data[w, r, 12]
                if (
                    efc_data[w, r, 4] <= 0.0
                    or efc_data[w, r, 5] <= 0.0
                    or efc_data[w, r, 11] != 1.0
                    or (upper_mask != 0.0 and upper_mask != 1.0)
                    or lower > 0.0
                ):
                    status[w] = 2
                if upper_mask == 1.0:
                    if upper < 0.0 or upper <= lower:
                        status[w] = 2
                inverse_r = 0.0
                if efc_data[w, r, 4] > 0.0 and tx.isfinite(efc_data[w, r, 4]):
                    inverse_r = 1.0 / efc_data[w, r, 4]
                if not tx.isfinite(inverse_r) or inverse_r <= 0.0:
                    status[w] = 2
                for d in range(nv):
                    if not tx.isfinite(efc_J[w, r, d]):
                        status[w] = 2
                # Native dry friction has lower<0<upper, hence starts free.
                if lower < 0.0:
                    working[w, r] = 1
                    if upper_mask == 1.0 and upper == 0.0:
                        working[w, r] = 2
            for i in range(nv):
                if (
                    not tx.isfinite(applied[w, i])
                    or not tx.isfinite(passive[w, i])
                    or not tx.isfinite(bias[w, i])
                ):
                    status[w] = 2
                for j in range(nv):
                    if not tx.isfinite(mass[w, i, j]):
                        status[w] = 2
        if status[w] == 0:
            # This factor is exclusively physical M, not the Euler M+h*damping.
            for i in range(nv):
                for j in range(i + 1):
                    value = mass[w, i, j]
                    for k in range(j):
                        value = value - physical_factor[w, i, k] * physical_factor[w, j, k]
                    if i == j:
                        if not tx.isfinite(value) or value <= 0.0:
                            status[w] = 4
                            value = 1.0
                        physical_factor[w, i, j] = tx.sqrt(value)
                    else:
                        if not tx.isfinite(value):
                            status[w] = 4
                            value = 0.0
                        physical_factor[w, i, j] = value / physical_factor[w, j, j]
                        if not tx.isfinite(physical_factor[w, i, j]):
                            status[w] = 4
            if status[w] == 0:
                for i in range(nv):
                    value = applied[w, i] + passive[w, i] - bias[w, i]
                    for j in range(i):
                        value = value - physical_factor[w, i, j] * mass_work[w, j]
                    mass_work[w, i] = value / physical_factor[w, i, i]
                for reverse in range(nv):
                    i = nv - 1 - reverse
                    value = mass_work[w, i]
                    for j in range(i + 1, nv):
                        value = value - physical_factor[w, j, i] * smooth_acc[w, j]
                    smooth_acc[w, i] = value / physical_factor[w, i, i]
                    if not tx.isfinite(smooth_acc[w, i]):
                        status[w] = 2
        done = 0
        refine_face = 0
        stalled = 0
        iteration_limit = budget
        if budget < 1 or budget > 512:
            iteration_limit = 0
        # The final iteration is a check only: exactly budget face solves allowed.
        for iteration in range(iteration_limit + 1):
            if done == 0 and status[w] == 0:
                # Independently derive a from current stored f using physical M.
                # Never trust the last face's trial_acc for the KKT certificate.
                for i in range(nv):
                    value = 0.0
                    for r in range(nrow):
                        value = value + efc_J[w, r, i] * efc_data[w, r, 9]
                    constraint[w, i] = value
                    total_applied[w, i] = applied[w, i] + value
                    if (
                        not tx.isfinite(value)
                        or not tx.isfinite(total_applied[w, i])
                    ):
                        status[w] = 2
                for i in range(nv):
                    value = total_applied[w, i] + passive[w, i] - bias[w, i]
                    for j in range(i):
                        value = value - physical_factor[w, i, j] * mass_work[w, j]
                    mass_work[w, i] = value / physical_factor[w, i, i]
                for reverse in range(nv):
                    i = nv - 1 - reverse
                    value = mass_work[w, i]
                    for j in range(i + 1, nv):
                        value = value - physical_factor[w, j, i] * physical_acc[w, j]
                    physical_acc[w, i] = value / physical_factor[w, i, i]
                    if not tx.isfinite(physical_acc[w, i]):
                        status[w] = 2
                residual = 0.0
                scale = 1.0
                bound_violation = 0.0
                free_count = 0
                for r in range(nrow):
                    value = -efc_data[w, r, 3]
                    row_scale = tx.abs(efc_data[w, r, 3])
                    for d in range(nv):
                        term = efc_J[w, r, d] * physical_acc[w, d]
                        value = value + term
                        row_scale = row_scale + tx.abs(term)
                    term = efc_data[w, r, 4] * efc_data[w, r, 9]
                    value = value + term
                    row_scale = row_scale + tx.abs(term)
                    efc_data[w, r, 10] = value
                    lower = efc_data[w, r, 7]
                    upper = efc_data[w, r, 8]
                    upper_mask = efc_data[w, r, 12]
                    force_value = efc_data[w, r, 9]
                    violation = tx.abs(value)
                    if force_value == lower:
                        violation = tx.max(0.0, -value)
                    elif upper_mask == 1.0 and force_value == upper:
                        violation = tx.max(0.0, value)
                    residual = tx.max(residual, violation)
                    scale = tx.max(scale, row_scale)
                    if force_value < lower:
                        bound_violation = tx.max(bound_violation, lower - force_value)
                    if upper_mask == 1.0 and force_value > upper:
                        bound_violation = tx.max(bound_violation, force_value - upper)
                    if working[w, r] == 1:
                        free_count = free_count + 1
                    if (
                        not tx.isfinite(value)
                        or not tx.isfinite(row_scale)
                        or not tx.isfinite(force_value)
                    ):
                        status[w] = 2
                mass_residual = 0.0
                mass_scale = 1.0
                for i in range(nv):
                    rhs_value = applied[w, i] + passive[w, i] - bias[w, i]
                    value = -rhs_value - constraint[w, i]
                    row_scale = tx.abs(rhs_value) + tx.abs(constraint[w, i])
                    for j in range(nv):
                        term = mass[w, i, j] * physical_acc[w, j]
                        value = value + term
                        row_scale = row_scale + tx.abs(term)
                    mass_residual = tx.max(mass_residual, tx.abs(value))
                    mass_scale = tx.max(mass_scale, row_scale)
                    if not tx.isfinite(value) or not tx.isfinite(row_scale):
                        status[w] = 2
                diagnostics[w, 1] = residual / scale
                diagnostics[w, 2] = residual
                diagnostics[w, 3] = bound_violation
                diagnostics[w, 5] = free_count
                diagnostics[w, 8] = mass_residual / mass_scale
                diagnostics[w, 9] = scale
                if status[w] == 0:
                    if bound_violation > 0.0 or diagnostics[w, 8] > tolerance:
                        status[w] = 5
                    elif diagnostics[w, 1] <= tolerance:
                        done = 1
                        diagnostics[w, 6] = 1.0
                    elif stalled == 1:
                        status[w] = 5
                    elif iteration == budget:
                        status[w] = 7
                if done == 0 and status[w] == 0:
                    for r in range(nrow):
                        previous_working[w, r] = working[w, r]
                        previous_force[w, r] = efc_data[w, r, 9]
                    # Block release, only after completing the prior feasible face.
                    # Internal blocking repairs cannot immediately re-add the hits.
                    if refine_face == 0:
                        for r in range(nrow):
                            if (
                                working[w, r] == 0
                                and efc_data[w, r, 10] < 0.0
                            ):
                                working[w, r] = 1
                            elif (
                                working[w, r] == 2
                                and efc_data[w, r, 10] > 0.0
                            ):
                                working[w, r] = 1
                    # H=M+J_F^T D_F J_F, rhs=rhs0+J_F^T D_F aref_F+J_B^T f_B.
                    for i in range(nv):
                        face_rhs[w, i] = applied[w, i] + passive[w, i] - bias[w, i]
                        for j in range(i + 1):
                            face_matrix[w, i, j] = mass[w, i, j]
                    for r in range(nrow):
                        if working[w, r] == 1:
                            inverse_r = 1.0 / efc_data[w, r, 4]
                            for i in range(nv):
                                weighted_j = efc_J[w, r, i] * inverse_r
                                face_rhs[w, i] = (
                                    face_rhs[w, i] + weighted_j * efc_data[w, r, 3]
                                )
                                for j in range(i + 1):
                                    face_matrix[w, i, j] = (
                                        face_matrix[w, i, j] + weighted_j * efc_J[w, r, j]
                                    )
                        else:
                            for i in range(nv):
                                face_rhs[w, i] = (
                                    face_rhs[w, i] + efc_J[w, r, i] * efc_data[w, r, 9]
                                )
                    for i in range(nv):
                        if not tx.isfinite(face_rhs[w, i]):
                            status[w] = 2
                        for j in range(i + 1):
                            face_matrix[w, j, i] = face_matrix[w, i, j]
                            if not tx.isfinite(face_matrix[w, i, j]):
                                status[w] = 2
                    if status[w] == 0:
                        diagnostics[w, 0] = diagnostics[w, 0] + 1.0
                        for i in range(nv):
                            for j in range(i + 1):
                                value = face_matrix[w, i, j]
                                for k in range(j):
                                    value = value - face_factor[w, i, k] * face_factor[w, j, k]
                                if i == j:
                                    if not tx.isfinite(value) or value <= 0.0:
                                        status[w] = 6
                                        value = 1.0
                                    face_factor[w, i, j] = tx.sqrt(value)
                                else:
                                    if not tx.isfinite(value):
                                        status[w] = 6
                                        value = 0.0
                                    face_factor[w, i, j] = value / face_factor[w, j, j]
                                    if not tx.isfinite(face_factor[w, i, j]):
                                        status[w] = 6
                    if status[w] == 0:
                        for i in range(nv):
                            value = face_rhs[w, i]
                            for j in range(i):
                                value = value - face_factor[w, i, j] * face_forward[w, j]
                            face_forward[w, i] = value / face_factor[w, i, i]
                        for reverse in range(nv):
                            i = nv - 1 - reverse
                            value = face_forward[w, i]
                            for j in range(i + 1, nv):
                                value = value - face_factor[w, j, i] * trial_acc[w, j]
                            trial_acc[w, i] = value / face_factor[w, i, i]
                            if not tx.isfinite(trial_acc[w, i]):
                                status[w] = 2
                        for r in range(nrow):
                            value = efc_data[w, r, 9]
                            if working[w, r] == 1:
                                value = efc_data[w, r, 3]
                                for i in range(nv):
                                    value = value - efc_J[w, r, i] * trial_acc[w, i]
                                value = value / efc_data[w, r, 4]
                            trial_force[w, r] = value
                            if not tx.isfinite(value):
                                status[w] = 2
                    if status[w] == 0:
                        alpha = 1.0
                        blocking = 0
                        for r in range(nrow):
                            if working[w, r] == 1:
                                lower = efc_data[w, r, 7]
                                upper = efc_data[w, r, 8]
                                upper_mask = efc_data[w, r, 12]
                                force_value = efc_data[w, r, 9]
                                candidate = 1.0
                                if trial_force[w, r] < lower:
                                    denominator = force_value - trial_force[w, r]
                                    if not tx.isfinite(denominator) or denominator <= 0.0:
                                        status[w] = 2
                                    else:
                                        candidate = (force_value - lower) / denominator
                                    blocking = 1
                                elif upper_mask == 1.0 and trial_force[w, r] > upper:
                                    denominator = trial_force[w, r] - force_value
                                    if not tx.isfinite(denominator) or denominator <= 0.0:
                                        status[w] = 2
                                    else:
                                        candidate = (upper - force_value) / denominator
                                    blocking = 1
                                if (
                                    not tx.isfinite(candidate)
                                    or candidate < 0.0
                                    or candidate > 1.0
                                ):
                                    status[w] = 2
                                alpha = tx.min(alpha, candidate)
                        if status[w] == 0:
                            for r in range(nrow):
                                if working[w, r] == 1:
                                    lower = efc_data[w, r, 7]
                                    upper = efc_data[w, r, 8]
                                    upper_mask = efc_data[w, r, 12]
                                    force_value = previous_force[w, r]
                                    candidate = 1.0
                                    bound_side = 1
                                    if trial_force[w, r] < lower:
                                        denominator = force_value - trial_force[w, r]
                                        candidate = (force_value - lower) / denominator
                                        bound_side = 0
                                    elif upper_mask == 1.0 and trial_force[w, r] > upper:
                                        denominator = trial_force[w, r] - force_value
                                        candidate = (upper - force_value) / denominator
                                        bound_side = 2
                                    if bound_side != 1 and candidate == alpha:
                                        working[w, r] = bound_side
                                        value = lower
                                        if bound_side == 2:
                                            value = upper
                                    else:
                                        value = trial_force[w, r]
                                        if alpha < 1.0:
                                            value = (
                                                (1.0 - alpha) * force_value
                                                + alpha * trial_force[w, r]
                                            )
                                    efc_data[w, r, 9] = value
                                    if not tx.isfinite(value):
                                        status[w] = 2
                            refine_face = blocking
                            changed = 0
                            for r in range(nrow):
                                if (
                                    working[w, r] != previous_working[w, r]
                                    or efc_data[w, r, 9] != previous_force[w, r]
                                ):
                                    changed = 1
                            if changed == 0:
                                stalled = 1
        # Successful outputs were rebuilt from stored forces in the check pass.
        # On failure the caller rejects this entire attempt, including all worlds.
