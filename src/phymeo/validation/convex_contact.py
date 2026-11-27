# Copyright 2026 Phymeo contributors. Apache-2.0; see LICENSE and NOTICE.
"""Independent host certificates for zero-skin convex contact witnesses.

Explicit validation only: no CUDA kernel, production fallback, collision search,
or learning admission. Unique support intersections are reconstructed from the
original vertices, not from the tested midpoint. A nonunique witness certifies
membership/closure of a choice, never uniqueness or a global minimum depth.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContactCertificate:
    pair: tuple
    point: tuple
    normal: tuple
    gap: float
    unique: bool
    kind: str
    surface_error: float
    point_error: float
    closure_error: float
    support_error: float


def _finite(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name}: finite shape {shape} required")
    return a


def _unit(v):
    size = np.linalg.norm(v)
    if not np.isfinite(size) or size <= 1e-15:
        raise ValueError("degenerate support feature")
    return v / size


def certify_contact(
    vertices,
    offsets,
    positions,
    rotations,
    pair,
    material_ids,
    weights,
    surfaces,
    normal,
    gap,
    receipt,
    *,
    geometry_atol=2e-9,
    normal_atol=1e-6,
):
    """Validate a captured witness against independent world transforms.

    IDs refer to the full immutable original catalog. Tolerances are validation
    accuracy, never contact activation margins. No gap is clipped to zero. The
    selected original support features must themselves establish the normal,
    gap and unique intersection for a receipt claiming uniqueness.
    """
    p = np.asarray(vertices, dtype=np.float64)
    off = np.asarray(offsets)
    ids = np.asarray(material_ids)
    pair = np.asarray(pair)
    receipt = np.asarray(receipt)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or not np.isfinite(p).all()
        or off.ndim != 1
        or off.dtype.kind not in "iu"
        or len(off) < 3
        or off[0] != 0
        or off[-1] != len(p)
        or np.any(np.diff(off) < 4)
        or pair.shape != (2,)
        or pair.dtype.kind not in "iu"
        or np.any(pair < 0)
        or np.any(pair >= len(off) - 1)
        or pair[0] == pair[1]
        or ids.shape != (2, 4)
        or ids.dtype.kind not in "iu"
        or receipt.shape != (3,)
        or receipt.dtype.kind not in "iu"
        or receipt[0] not in (1, 2)
    ):
        raise ValueError("invalid catalog, pair, material or receipt schema")
    if (
        not np.isfinite([geometry_atol, normal_atol, gap]).all()
        or min(geometry_atol, normal_atol) <= 0
    ):
        raise ValueError("finite gap and positive tolerances required")
    pos = _finite(positions, (len(off) - 1, 3), "positions")
    rot = _finite(rotations, (len(off) - 1, 3, 3), "rotations")
    w = _finite(weights, (2, 4), "weights")
    surface = _finite(surfaces, (2, 3), "surfaces")
    n = _finite(normal, (3,), "normal")
    if abs(np.linalg.norm(n) - 1) > normal_atol:
        raise ValueError("nonunit normal")
    n = _unit(n)
    world, selected, reconstructed = [], [], []
    for side, g in enumerate(pair):
        if np.max(np.abs(rot[g].T @ rot[g] - np.eye(3))) > 1e-10 or np.linalg.det(rot[g]) < 0:
            raise ValueError("independent proper rotation required")
        used = ids[side] >= 0
        ix = ids[side, used]
        if (
            not len(ix)
            or len(set(ix)) != len(ix)
            or np.any(ids[side, ~used] != -1)
            or np.any(ix < off[g])
            or np.any(ix >= off[g + 1])
            or np.any(w[side] < 0)
            or np.any(w[side, ~used] != 0)
            or abs(w[side].sum() - 1) > 2e-7
        ):
            raise ValueError("material ownership or nonnegative normalized weights")
        world.append(p[off[g] : off[g + 1]] @ rot[g].T + pos[g])
        feature = p[ix] @ rot[g].T + pos[g]
        selected.append(feature)
        # Match the documented affine material representation. Weight rounding
        # affects the partition sum; no renormalization of tested data occurs.
        reconstructed.append(feature[0] + w[side, used][1:] @ (feature[1:] - feature[0]))
    reconstructed = np.asarray(reconstructed)
    surface_error = float(np.max(np.linalg.norm(reconstructed - surface, axis=1)))
    if surface_error > geometry_atol:
        raise ValueError("surface/material reconstruction mismatch")

    unique = receipt[0] == 2
    a, b = selected
    kind = "nonunique_choice"
    independent = reconstructed.copy()
    condition = 1.0
    if unique:
        if tuple(receipt[1:]) == (2, 2) and len(a) == len(b) == 2:
            e, f = a[1] - a[0], b[1] - b[0]
            cross = np.cross(e, f)
            if cross @ cross <= 1e-12 * (e @ e) * (f @ f):
                raise ValueError("ill-conditioned unique edge intersection")
            axis = _unit(cross)
            condition = np.sqrt((e @ e) * (f @ f) / (cross @ cross))
            if axis @ n < 0:
                axis = -axis
            u, v, _ = np.linalg.solve(np.column_stack((e, -f, axis)), b[0] - a[0])
            if min(u, v) < 0 or max(u, v) > 1:
                raise ValueError("intersection outside original edges")
            independent = np.array([a[0] + u * e, b[0] + v * f])
            kind = "unique_edge_edge"
        elif min(receipt[1:]) == 1:
            side = 0 if receipt[1] == 1 else 1
            face, vertex = selected[1 - side], selected[side]
            if len(vertex) != 1 or len(face) < 3:
                raise ValueError("unsupported unique vertex feature")
            e, f = face[1] - face[0], face[2] - face[0]
            cross = np.cross(e, f)
            if cross @ cross <= 1e-12 * (e @ e) * (f @ f):
                raise ValueError("ill-conditioned unique face")
            axis = _unit(cross)
            condition = np.sqrt((e @ e) * (f @ f) / (cross @ cross))
            if axis @ n < 0:
                axis = -axis
            projected = vertex[0] - ((vertex[0] - face[0]) @ axis) * axis
            uv, _, rank, _ = np.linalg.lstsq(
                (face[1:3] - face[0]).T, projected - face[0], rcond=None
            )
            if rank != 2 or min(uv) < -1e-12 or uv.sum() > 1 + 1e-12:
                raise ValueError("projected vertex outside original triangle")
            independent[side], independent[1 - side] = vertex[0], projected
            kind = "unique_vertex_face"
        else:
            raise ValueError("unsupported unique support topology")
        if np.linalg.norm(axis - n) > normal_atol:
            raise ValueError("normal disagrees with original features")
    else:
        # A nonunique choice still has to be on both support faces and close.
        # It is not replaced by an arbitrary convenient or native point.
        if min(receipt[1:]) < 2:
            raise ValueError("invalid nonunique topology")
        axis = n

    projections = [x @ axis for x in world]
    limits = [projections[0].max(), projections[1].min()]
    exact_gap = float(limits[1] - limits[0])
    support_error = max(float(np.max(np.abs(selected[s] @ axis - limits[s]))) for s in range(2))
    # The independent axis establishes which complete original support features
    # are present, rather than trusting the claimed vertex counts.
    # Binary64 arithmetic envelope, scaled by geometry and cross conditioning.
    # A fixed world-length threshold can incorrectly merge distinct features.
    arithmetic = (
        128 * np.finfo(float).eps * max(1.0, max(np.abs(x).max() for x in world)) * condition
    )
    active_counts = [
        int(np.count_nonzero(np.abs(projections[s] - limits[s]) <= arithmetic)) for s in range(2)
    ]
    if unique and tuple(active_counts) != tuple(receipt[1:]):
        raise ValueError("support feature count/uniqueness mismatch")
    closure_error = float(np.linalg.norm(surface[1] - surface[0] - gap * n))
    point_error = float(np.linalg.norm(independent.mean(0) - surface.mean(0)))
    if max(support_error, abs(exact_gap - gap), closure_error, point_error) > geometry_atol:
        raise ValueError("support, signed gap, closure or independent point mismatch")
    return ContactCertificate(
        tuple(map(int, pair)),
        tuple(independent.mean(0)),
        tuple(axis),
        exact_gap,
        bool(unique),
        kind,
        surface_error,
        point_error,
        closure_error,
        support_error,
    )


def match_contact_pairs(actual, expected):
    """One-to-one single-point convex pairing; duplicates/multicontact reject."""

    def keys(values):
        a = np.asarray(values)
        if a.ndim != 2 or a.shape[1] != 2 or a.dtype.kind not in "iu" or np.any(a < 0):
            raise ValueError("integer contact pair matrix required")
        result = list(map(tuple, a.tolist()))
        if any(x == y for x, y in result) or len(set(result)) != len(result):
            raise ValueError(
                "duplicate/degenerate contact pair; multicontact needs separate contract"
            )
        return result

    left, right = keys(actual), keys(expected)
    if set(left) != set(right):
        raise ValueError(
            f"contact set mismatch: extra={set(left) - set(right)}, missing={set(right) - set(left)}"
        )
    lookup = {pair: i for i, pair in enumerate(right)}
    return np.asarray([lookup[pair] for pair in left], dtype=np.int64)
