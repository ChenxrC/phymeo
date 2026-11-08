"""Independent K0 finite geometry declarations and witness checks.

NumPy + standard library only. No Phymeo, TYX, Newton, or production geometry
imports. Expected values come from the declared finite sets and literal planes.
AUTHORING ONLY / UNEXECUTED; this is not a native Newton driver.
"""

from itertools import product

import numpy as np

FAMILIES = ("SS", "SC", "SB", "SH", "CC", "CB", "CH", "BB", "BH", "HH")
GAPS = {"sep": 0.02, "touch": 0.0, "overlap": -0.01}
U32 = 2.0**-23
CYCLE = np.array(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
TILT = np.array(((0.6, 0.0, 0.8), (0.0, 1.0, 0.0), (-0.8, 0.0, 0.6)))


def primitive(kind):
    if kind == "S":
        return np.array([[0.0, 0.0, 0.0]]), 0.05, 0.05
    if kind == "C":
        return np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]), 0.05, 0.05
    if kind == "B":
        return np.array(list(product((-0.6, 0.6), (-0.1, 0.1), (-0.1, 0.1)))), 0.0, 0.1
    if kind == "H":
        ring = ((-0.5, -0.1), (0.5, -0.1), (0.6, 0.0), (0.3, 0.1), (-0.5, 0.1))
        return np.array([(x, y, z) for x, y in ring for z in (-0.1, 0.1)]), 0.0, 0.1
    raise ValueError("undeclared finite family")


def family_case(family, gap, *, world=17):
    if family not in FAMILIES:
        raise ValueError("undeclared pair family")
    first, ra, ea = primitive(family[0])
    second, rb, eb = primitive(family[1])
    shift = np.array([0.0, 0.0, ea + eb + float(gap)])
    return {
        "family": family,
        "kinds": tuple(family),
        "keys": ((world, 1, 0), (world, 2, 0)),
        "ids": (tuple(range(100, 100 + len(first))), tuple(range(200, 200 + len(second)))),
        "local": (first, second),
        "positions": (np.zeros(3), shift),
        "rotations": (np.eye(3), np.eye(3)),
        "radii": (ra, rb),
        "points": (first.copy(), second + shift),
        "gap_m": float(gap),
        "normal": np.array([0.0, 0.0, 1.0]),
        "requested_reverse": False,
        "registration_reverse": False,
        "oracle": "parallel finite support planes with nonempty projected intersection",
    }


def diameter(points, radius):
    points = np.asarray(points, dtype=float)
    return max(float(np.linalg.norm(a - b)) for a in points for b in points) + 2 * radius


def scale(case):
    return max(
        1.0,
        *(diameter(points, radius) for points, radius in zip(case["points"], case["radii"], strict=True)),
        float(np.linalg.norm(np.asarray(case["positions"][1]) - case["positions"][0])),
    )


def parallel_oracle(case, points=None, radii=None):
    """Independent support planes, applicable to the declared Q01/Q03 cases.

    The caller supplies the original or independently reconstructed uploaded
    points. No returned query gap/normal/feature selects the reference planes.
    """
    xyz = case["points"] if points is None else points
    radii = case["radii"] if radii is None else radii
    normal = np.asarray(case["normal"], dtype=float)
    upper = max(np.asarray(xyz[0]) @ normal) + radii[0]
    lower = min(np.asarray(xyz[1]) @ normal) - radii[1]
    return {"gap_m": float(lower - upper), "normal": normal.copy(), "first_plane_m": float(upper), "second_plane_m": float(lower)}


def segment_parameter(point, a, b):
    edge = np.asarray(b) - a
    square = float(edge @ edge)
    if square == 0:
        return 0.5
    return min(1.0, max(0.0, float((np.asarray(point) - a) @ edge) / square))


def finite_material_check(points, radius, ids, witness, normal, side, *, tolerance_m):
    """Check reconstruction and the independently directed support surface."""
    points = np.asarray(points, dtype=float)
    weights = np.asarray(witness.weights, dtype=float)
    used = tuple(int(x) for x in witness.vertex_ids)
    assert witness.valid and len(used) == len(weights) and 1 <= len(used) <= 4
    assert len(set(used)) == len(used) and set(used) <= set(ids)
    assert np.isfinite(weights).all() and np.all(weights >= 0)
    assert abs(float(weights.sum()) - 1.0) <= 64 * U32
    by_id = dict(zip(ids, points, strict=True))
    core = sum((weight * by_id[node] for node, weight in zip(used, weights, strict=True)), np.zeros(3))
    offset = np.asarray(witness.offset_m, dtype=float)
    surface = np.asarray(witness.surface_relative_m, dtype=float)
    assert offset.shape == surface.shape == (3,) and np.isfinite(offset).all() and np.isfinite(surface).all()
    assert np.linalg.norm(surface - core - offset) <= tolerance_m
    expected_offset = (1.0 if side == 0 else -1.0) * radius * normal
    assert np.linalg.norm(offset - expected_offset) <= tolerance_m
    if radius == 0:
        assert np.linalg.norm(offset) <= tolerance_m
    projected = points @ normal
    support = max(projected) + radius if side == 0 else min(projected) - radius
    assert abs(float(surface @ normal) - support) <= tolerance_m
    return {"reconstruction_error_m": float(np.linalg.norm(surface - core - offset)), "support_error_m": abs(float(surface @ normal) - support), "core_m": core}


def literal_parallel_precondition(case):
    oracle = parallel_oracle(case)
    assert abs(oracle["gap_m"] - case["gap_m"]) <= 1e-12
    assert np.array_equal(oracle["normal"], case["normal"])
    return oracle