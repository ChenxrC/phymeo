"""Independent entry-geometry checks for floating native/area contact.

Plane features have explicit geometric witnesses. Other retained pairs need
a separating direction at their native detection margin on this route.
"""

import numpy as np

_EPS = float(np.finfo(np.float32).eps)


def _require(ok, detail):
    if not ok:
        raise ValueError("resident mixed contact geometry: " + detail)


def _support_radius(kind, rotation, size, direction):
    if kind == 2:
        return size[0]
    if kind == 3:
        return size[0] + size[1] * abs(rotation[:, 2] @ direction)
    if kind == 6:
        return size @ abs(rotation.T @ direction)
    raise ValueError("unsupported finite shape in mixed contact geometry")


def _separated(a, b, margin, tolerance):
    # Each projected radius bounds the complete shape. The list of axes can
    # be incomplete: failure to prove separation rejects the candidate.
    ka, pa, ra, sa = a
    kb, pb, rb, sb = b
    center_delta = pb - pa
    axes = [center_delta, *ra.T, *rb.T, *np.cross(ra.T[:, None], rb.T).reshape(-1, 3)]
    for axis in axes:
        length = np.linalg.norm(axis)
        if length <= 64 * _EPS:
            continue
        unit = axis / length
        clearance = (
            abs(center_delta @ unit)
            - _support_radius(ka, ra, sa, unit)
            - _support_radius(kb, rb, sb, unit)
        )
        if clearance > max(0.0, margin) + tolerance:
            return True
    return False


def _check_frame(frame, normal, axis, minimum):
    n = normal / np.linalg.norm(normal)
    if axis is None:
        seeds = [np.eye(3)[1 if abs(n[1]) < 0.5 else 2]]
        if abs(abs(n[1]) - 0.5) < 32 * _EPS:
            seeds = np.eye(3)[1:]
    else:
        seeds = [axis]
    for seed in seeds:
        tangent = seed - n * (n @ seed)
        length = np.linalg.norm(tangent)
        if length < minimum:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            if length <= 64 * _EPS:
                # An ill-conditioned tangent cannot authenticate a friction frame.
                continue
            tangent /= length
        expected = np.stack([n, tangent, np.cross(n, tangent)])
        if np.allclose(frame, expected, rtol=0, atol=64 * _EPS):
            return
    raise ValueError("resident mixed contact geometry: source tangent frame differs")


def validate_geometry(plan, raw_candidate_rows, dynamics):
    """Bind every retained slot to both immutable local shapes and entry poses."""
    rows = np.asarray(raw_candidate_rows)
    _require(rows.shape == (max(1, plan.nslot), 32) and np.isfinite(rows).all(), "candidate rows")
    _require(
        np.isin(rows[:, :2], (0, 1)).all() and not np.any(rows[:, 1] > rows[:, 0]),
        "candidate flags",
    )
    poses = []
    scales = []
    for ids, local in zip(plan.geom_ids, plan.geom_data, strict=True):
        body, kind = map(int, ids[:2])
        origin = dynamics.body_position[body]
        rotation = dynamics.body_rotation[body]
        local = local.astype(float)
        pos = origin + rotation @ local[:3]
        geom_rotation = rotation @ local[3:12].reshape(3, 3)
        poses.append((kind, pos, geom_rotation, local[12:15]))
        # Includes cancellation between the body origin and a geom-local offset.
        scales.append(
            np.linalg.norm(origin) + np.linalg.norm(local[:3]) + np.linalg.norm(local[12:15])
        )
    for pair in range(plan.npair):
        ga, gb, first, count = map(int, plan.pair_ids[pair])
        a, b = poses[ga], poses[gb]
        kind, pos, rotation, size = b
        margin, gap = map(float, plan.pair_law[pair, :2])
        include = float(np.float32(margin - gap))
        tol = (
            64 * _EPS * (scales[ga] + scales[gb] + abs(margin) + abs(gap))
            + np.finfo(np.float32).tiny
        )
        selected = 0
        for feature in range(count):
            slot = first + feature
            _require(
                tuple(plan.slot_ids[slot]) == (pair, feature, ga, gb), "material slot identity"
            )
        if a[0] != 0:
            _require(
                _separated(a, b, margin, tol) and not rows[first : first + count].any(),
                "non-plane candidate requires a strict separating witness",
            )
            continue
        plane_origin, n = a[1], a[2][:, 2]
        _require(kind in (2, 3, 6) and count == {2: 1, 3: 2, 6: 8}[kind], "plane feature count")
        for feature, row in enumerate(rows[first : first + count]):
            offset = np.zeros(3)
            radius = size[0]
            seed = None
            if kind == 3:
                seed = rotation[:, 2]
                offset = (1 if feature == 0 else -1) * size[1] * seed
            elif kind == 6:
                signs = np.array([1 if feature & (1 << j) else -1 for j in range(3)])
                offset = rotation @ (size * signs)
                radius = 0
            lower, upper = True, True
            if kind == 6:
                eligible = offset @ n
                lower = selected < 4 and eligible < -tol
                upper = selected < 4 and eligible <= tol
            feature_center = pos + offset
            distance = (feature_center - plane_origin) @ n - radius
            detected = bool(row[0])
            if not detected:
                _require(
                    not row.any() and not (lower and distance < margin - tol),
                    "missing source feature",
                )
                continue
            _require(
                upper and distance <= margin + tol and abs(row[2] - distance) <= tol, "source gap"
            )
            selected += 1
            _require(
                (distance <= include + tol if row[1] else distance >= include - tol)
                and tuple(row[21:24]) == (include, margin, gap),
                "source activity",
            )
            point = feature_center - (radius + 0.5 * distance) * n
            _require(np.max(abs(row[9:12] - point)) <= tol, "source contact point")
            _check_frame(row[12:21].reshape(3, 3), n, seed, float(plan.pair_law[pair, 10]))
