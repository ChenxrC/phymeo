# Copyright 2021 DeepMind Technologies Limited
# Copyright 2025 The Newton Developers
# Copyright 2026 Phymeo contributors
# Apache-2.0; see LICENSE and NOTICE.
# Native row equations follow MuJoCo 3.3.7 engine_core_constraint.c.
# Device organization also references MuJoCo-Warp 0.0.2 constraint.py.

"""Typed native joint/contact rows for one shared bounded-force solve."""

import tyx as tx


@tx.func
def _power(value: tx.f32, exponent: tx.f32) -> tx.f32:
    result = value
    if exponent == 2.0:
        result = value * value
    elif exponent != 1.0:
        result = tx.pow(value, exponent)
    return result


@tx.func
def _impedance(
    position: tx.f32,
    dmin: tx.f32,
    dmax: tx.f32,
    width: tx.f32,
    midpoint: tx.f32,
    power: tx.f32,
    minimum: tx.f32,
) -> tx.f32:
    # Endpoints are individually clamped by the caller, never sorted.
    result = 0.5 * (dmin + dmax)
    if dmin != dmax and width > minimum:
        x = tx.abs(position / width)
        result = dmin
        if x >= 1.0:
            result = dmax
        elif x > 0.0:
            y = x
            valid = 1
            if power != 1.0:
                denominator = _power(midpoint, power - 1.0)
                if x > midpoint:
                    denominator = _power(1.0 - midpoint, power - 1.0)
                if not tx.isfinite(denominator) or denominator <= 0.0:
                    valid = 0
                elif x <= midpoint:
                    y = (1.0 / denominator) * _power(x, power)
                else:
                    y = 1.0 - (1.0 / denominator) * _power(1.0 - x, power)
            result = 0.0
            if valid == 1:
                result = dmin + y * (dmax - dmin)
            # Zero is outside the admitted impedance interval: the caller
            # reports status 2 rather than repairing an underflowed power.
    return result


def assemble(
    count: tx.i32,
    nv: tx.i32,
    nslot: tx.i32,
    nfriction: tx.i32,
    nlimit: tx.i32,
    contact_capacity: tx.i32,
    efc_capacity: tx.i32,
    h: tx.f32,
    impratio: tx.f32,
    slot_ids: tx.ndarray(tx.i32, 2),
    geom_ids: tx.ndarray(tx.i32, 2),
    pair_condim: tx.ndarray(tx.i32, 1),
    pair_friction: tx.ndarray(tx.f32, 2),
    pair_law: tx.ndarray(tx.f32, 2),
    friction_ids: tx.ndarray(tx.i32, 1),
    friction_law: tx.ndarray(tx.f32, 2),
    limit_ids: tx.ndarray(tx.i32, 2),
    limit_law: tx.ndarray(tx.f32, 2),
    q: tx.ndarray(tx.f32, 2),
    v: tx.ndarray(tx.f32, 2),
    motion: tx.ndarray(tx.f32, 4),
    body_jacobian: tx.ndarray(tx.f32, 4),
    normal_rows: tx.ndarray(tx.f32, 3),
    efc_ids: tx.ndarray(tx.i32, 3),
    efc_data: tx.ndarray(tx.f32, 3),
    efc_J: tx.ndarray(tx.f32, 3),
    contact_map: tx.ndarray(tx.i32, 3),
    contact_J: tx.ndarray(tx.f32, 4),
    counts: tx.ndarray(tx.i32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        width = nv
        if width < 8:
            width = 8
        for r in range(efc_capacity):
            for k in range(5):
                efc_ids[w, r, k] = -1
            for k in range(14):
                efc_data[w, r, k] = 0.0
            for d in range(width):
                efc_J[w, r, d] = 0.0
        for c in range(contact_capacity):
            for k in range(3):
                contact_map[w, c, k] = -1
            contact_map[w, c, 2] = 0
            for k in range(6):
                for d in range(width):
                    contact_J[w, c, k, d] = 0.0
        for k in range(3):
            counts[w, k] = 0
        required = 0
        detected = 0
        active_contacts = 0
        # An upstream geometry capacity error still permits complete counting.
        # Other upstream failures leave clean, non-publishable row buffers.
        if status[w] == 0 or status[w] == 1:
            for f in range(nfriction):
                dof = friction_ids[f]
                loss = friction_law[f, 0]
                if not tx.isfinite(loss) or loss <= 0.0 or dof < 0 or dof >= nv:
                    status[w] = 2
                else:
                    r = required
                    required = required + 1
                    if r < efc_capacity:
                        efc_ids[w, r, 0] = 1
                        efc_ids[w, r, 1] = dof
                        efc_ids[w, r, 4] = dof
                        efc_data[w, r, 2] = v[w, dof]
                        efc_data[w, r, 7] = -loss
                        efc_data[w, r, 8] = loss
                        efc_data[w, r, 11] = 1.0
                        efc_data[w, r, 12] = 1.0
                        efc_J[w, r, dof] = 1.0
            for joint in range(nlimit):
                native_joint = limit_ids[joint, 0]
                qa = limit_ids[joint, 1]
                dof = limit_ids[joint, 2]
                margin = limit_law[joint, 2]
                if dof < 0 or dof >= nv or not tx.isfinite(margin):
                    status[w] = 2
                else:
                    value = q[w, qa]
                    if not tx.isfinite(value):
                        status[w] = 2
                    for side in range(2):
                        sign = 1.0
                        distance = value - limit_law[joint, 0]
                        if side == 1:
                            sign = -1.0
                            distance = limit_law[joint, 1] - value
                        if not tx.isfinite(distance):
                            status[w] = 2
                        elif distance < margin:
                            r = required
                            required = required + 1
                            if r < efc_capacity:
                                efc_ids[w, r, 0] = 3
                                efc_ids[w, r, 1] = native_joint
                                efc_ids[w, r, 2] = 2 * side - 1
                                efc_ids[w, r, 4] = dof
                                efc_data[w, r, 0] = distance
                                efc_data[w, r, 1] = margin
                                efc_data[w, r, 2] = sign * v[w, dof]
                                efc_data[w, r, 11] = 1.0
                                efc_J[w, r, dof] = sign
            for s in range(nslot):
                if normal_rows[w, s, 0] == 1.0:
                    c = detected
                    detected = detected + 1
                    pair = slot_ids[s, 0]
                    ga = slot_ids[s, 2]
                    gb = slot_ids[s, 3]
                    ba = geom_ids[ga, 0]
                    bb = geom_ids[gb, 0]
                    dim = pair_condim[pair]
                    distance = normal_rows[w, s, 2]
                    margin = pair_law[pair, 0] - pair_law[pair, 1]
                    if (
                        not tx.isfinite(distance)
                        or not tx.isfinite(margin)
                        or (dim != 1 and dim != 3 and dim != 4 and dim != 6)
                    ):
                        status[w] = 2
                    if c < contact_capacity:
                        contact_map[w, c, 0] = s
                        for d in range(nv):
                            for component in range(3):
                                linear = 0.0
                                angular = 0.0
                                for i in range(3):
                                    j = (i + 1) % 3
                                    k = (i + 2) % 3
                                    ja = (
                                        body_jacobian[w, ba, i, d]
                                        + body_jacobian[w, ba, 3 + j, d]
                                        * (normal_rows[w, s, 9 + k] - motion[w, ba, 5, k])
                                        - body_jacobian[w, ba, 3 + k, d]
                                        * (normal_rows[w, s, 9 + j] - motion[w, ba, 5, j])
                                    )
                                    jb = (
                                        body_jacobian[w, bb, i, d]
                                        + body_jacobian[w, bb, 3 + j, d]
                                        * (normal_rows[w, s, 9 + k] - motion[w, bb, 5, k])
                                        - body_jacobian[w, bb, 3 + k, d]
                                        * (normal_rows[w, s, 9 + j] - motion[w, bb, 5, j])
                                    )
                                    axis = normal_rows[w, s, 12 + 3 * component + i]
                                    linear = linear + axis * (jb - ja)
                                    angular = angular + axis * (
                                        body_jacobian[w, bb, 3 + i, d]
                                        - body_jacobian[w, ba, 3 + i, d]
                                    )
                                contact_J[w, c, component, d] = linear
                                contact_J[w, c, 3 + component, d] = angular
                                if not tx.isfinite(linear) or not tx.isfinite(angular):
                                    status[w] = 2
                    if distance < margin:
                        active_contacts = active_contacts + 1
                        edges = 1
                        row_type = 5
                        if dim > 1:
                            edges = 2 * (dim - 1)
                            row_type = 6
                        if c < contact_capacity:
                            contact_map[w, c, 1] = required
                            contact_map[w, c, 2] = edges
                        for edge in range(edges):
                            r = required
                            required = required + 1
                            if r < efc_capacity and c < contact_capacity:
                                efc_ids[w, r, 0] = row_type
                                efc_ids[w, r, 1] = c
                                efc_ids[w, r, 2] = edge
                                efc_ids[w, r, 3] = s
                                efc_data[w, r, 0] = distance
                                efc_data[w, r, 1] = margin
                                efc_data[w, r, 11] = 1.0
                                velocity = 0.0
                                for d in range(nv):
                                    value = contact_J[w, c, 0, d]
                                    if dim > 1:
                                        component = edge // 2 + 1
                                        sign = 1.0
                                        if edge % 2 == 1:
                                            sign = -1.0
                                        value = value + (
                                            sign
                                            * pair_friction[pair, component - 1]
                                            * contact_J[w, c, component, d]
                                        )
                                    efc_J[w, r, d] = value
                                    velocity = velocity + value * v[w, d]
                                efc_data[w, r, 2] = velocity
            counts[w, 0] = detected
            counts[w, 1] = active_contacts
            counts[w, 2] = required
            # Parameter computation is separate from ordered compaction so the
            # same CPU law applies to all row kinds without parameter scratch.
            for r in range(efc_capacity):
                if efc_data[w, r, 11] == 1.0:
                    row_type = efc_ids[w, r, 0]
                    owner = efc_ids[w, r, 1]
                    minimum = 1.0e-15
                    minimp = 0.0001
                    maximp = 0.9999
                    invweight = 0.0
                    ref0 = 0.0
                    ref1 = 0.0
                    raw_dmin = 0.0
                    raw_dmax = 0.0
                    raw_width = 0.0
                    raw_midpoint = 0.0
                    raw_power = 0.0
                    found = 0
                    mu0 = 0.0
                    if row_type == 1:
                        for f in range(nfriction):
                            if friction_ids[f] == owner:
                                invweight = friction_law[f, 1]
                                ref0 = friction_law[f, 2]
                                ref1 = friction_law[f, 3]
                                raw_dmin = friction_law[f, 4]
                                raw_dmax = friction_law[f, 5]
                                raw_width = friction_law[f, 6]
                                raw_midpoint = friction_law[f, 7]
                                raw_power = friction_law[f, 8]
                                found = found + 1
                    elif row_type == 3:
                        for joint in range(nlimit):
                            if limit_ids[joint, 0] == owner:
                                invweight = limit_law[joint, 3]
                                ref0 = limit_law[joint, 4]
                                ref1 = limit_law[joint, 5]
                                raw_dmin = limit_law[joint, 6]
                                raw_dmax = limit_law[joint, 7]
                                raw_width = limit_law[joint, 8]
                                raw_midpoint = limit_law[joint, 9]
                                raw_power = limit_law[joint, 10]
                                found = found + 1
                    else:
                        s = efc_ids[w, r, 3]
                        pair = slot_ids[s, 0]
                        invweight = pair_law[pair, 2]
                        ref0 = pair_law[pair, 3]
                        ref1 = pair_law[pair, 4]
                        raw_dmin = pair_law[pair, 5]
                        raw_dmax = pair_law[pair, 6]
                        raw_width = pair_law[pair, 7]
                        raw_midpoint = pair_law[pair, 8]
                        raw_power = pair_law[pair, 9]
                        minimum = pair_law[pair, 10]
                        minimp = pair_law[pair, 11]
                        maximp = pair_law[pair, 12]
                        mu0 = pair_friction[pair, 0]
                        found = 1
                    dmin = tx.min(maximp, tx.max(minimp, raw_dmin))
                    dmax = tx.min(maximp, tx.max(minimp, raw_dmax))
                    law_width = tx.max(0.0, raw_width)
                    midpoint = tx.min(maximp, tx.max(minimp, raw_midpoint))
                    power = tx.max(1.0, raw_power)
                    position = efc_data[w, r, 0] - efc_data[w, r, 1]
                    impedance = _impedance(
                        position, dmin, dmax, law_width, midpoint, power, minimum
                    )
                    if (
                        found != 1
                        or not tx.isfinite(impedance)
                        or impedance <= 0.0
                        or impedance >= 1.0
                    ):
                        status[w] = 2
                    else:
                        stiffness = 0.0
                        damping = 0.0
                        if ref0 > 0.0:
                            ref0 = tx.max(ref0, 2.0 * h)
                        if row_type != 1:
                            if ref0 > 0.0:
                                stiffness = 1.0 / tx.max(
                                    minimum, dmax * dmax * ref0 * ref0 * ref1 * ref1
                                )
                            else:
                                stiffness = -ref0 / tx.max(minimum, dmax * dmax)
                        if ref1 > 0.0:
                            damping = 2.0 / tx.max(minimum, dmax * ref0)
                        else:
                            damping = -ref1 / tx.max(minimum, dmax)
                        approximate = invweight
                        if row_type == 6:
                            approximate = invweight + mu0 * mu0 * invweight
                        regularization = tx.max(
                            minimum, (1.0 - impedance) * approximate / impedance
                        )
                        if not tx.isfinite(regularization) or regularization <= 0.0:
                            status[w] = 2
                        else:
                            if row_type == 6:
                                tangent_r = regularization / tx.max(minimum, impratio)
                                scaled_mu = mu0 * tx.sqrt(tangent_r / regularization)
                                regularization = 2.0 * scaled_mu * scaled_mu * regularization
                            if not tx.isfinite(regularization) or regularization <= 0.0:
                                status[w] = 2
                            else:
                                efc_data[w, r, 3] = (
                                    -damping * efc_data[w, r, 2]
                                    - stiffness * impedance * position
                                )
                                efc_data[w, r, 4] = regularization
                                efc_data[w, r, 5] = 1.0 / regularization
                                efc_data[w, r, 6] = impedance
                                efc_data[w, r, 13] = (
                                    regularization * impedance / (1.0 - impedance)
                                )
                    for k in range(14):
                        if not tx.isfinite(efc_data[w, r, k]):
                            status[w] = 2
                    for d in range(nv):
                        if not tx.isfinite(efc_J[w, r, d]):
                            status[w] = 2
            if detected > contact_capacity or required > efc_capacity:
                status[w] = 1


assemble_other = tx.kernel(assemble)
assemble = tx.kernel(assemble)


@tx.kernel
def decode(
    count: tx.i32,
    nv: tx.i32,
    contact_capacity: tx.i32,
    efc_capacity: tx.i32,
    slot_ids: tx.ndarray(tx.i32, 2),
    pair_condim: tx.ndarray(tx.i32, 1),
    pair_friction: tx.ndarray(tx.f32, 2),
    contact_map: tx.ndarray(tx.i32, 3),
    counts: tx.ndarray(tx.i32, 2),
    efc_data: tx.ndarray(tx.f32, 3),
    contact_wrench: tx.ndarray(tx.f32, 3),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        for c in range(contact_capacity):
            for k in range(6):
                contact_wrench[w, c, k] = 0.0
        if status[w] == 0:
            if counts[w, 0] > contact_capacity or counts[w, 2] > efc_capacity:
                status[w] = 1
            else:
                active = 0
                for c in range(contact_capacity):
                    if c < counts[w, 0]:
                        s = contact_map[w, c, 0]
                        first = contact_map[w, c, 1]
                        edges = contact_map[w, c, 2]
                        if s < 0:
                            status[w] = 2
                        elif edges > 0:
                            active = active + 1
                            pair = slot_ids[s, 0]
                            dim = pair_condim[pair]
                            expected = 1
                            if dim > 1:
                                expected = 2 * (dim - 1)
                            if (
                                edges != expected
                                or first < 0
                                or first + edges > counts[w, 2]
                                or first + edges > efc_capacity
                            ):
                                status[w] = 2
                            else:
                                normal_force = 0.0
                                for edge in range(edges):
                                    force = efc_data[w, first + edge, 9]
                                    normal_force = normal_force + force
                                    if not tx.isfinite(force):
                                        status[w] = 2
                                contact_wrench[w, c, 0] = normal_force
                                if dim > 1:
                                    for component in range(1, dim):
                                        plus = efc_data[w, first + 2 * (component - 1), 9]
                                        minus = efc_data[w, first + 2 * (component - 1) + 1, 9]
                                        contact_wrench[w, c, component] = (
                                            pair_friction[pair, component - 1] * (plus - minus)
                                        )
                        elif first != -1:
                            status[w] = 2
                        for k in range(6):
                            if not tx.isfinite(contact_wrench[w, c, k]):
                                status[w] = 2
                if active != counts[w, 1]:
                    status[w] = 2
