"""M1 independent literals, affine kinematics and finite-distance reference.

No Phymeo/TYX imports. This file is authored before the new producer exists.
The Fraction oracle is the existing independent COL2 reference, never a product.
"""

import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np

LAYOUTS = ("ordinary", "affine", "floating", "floating_affine")
WORLDS = (83, 17, 107, 31)
H = 1 / 1024
RADIUS = 1 / 8
MARGIN = 1 / 2
ASSET_Q = ((0, 0, 0, 1), (0.5, 0.5, 0.5, 0.5), (0, 0.6, 0, 0.8), (0, 0, 1, 0))
TRIANGLE = ((0, 0, 0), (0, 2, 0), (0, 0, 2))
ROUND = (-1 / 4, 1 / 2, 1 / 2)
IDS = (11, 37, 91)
U = 2**-24


@lru_cache(maxsize=1)
def distance_reference():
    path = Path(__file__).with_name("device_triangle_distance_reference.py")
    spec = importlib.util.spec_from_file_location("m1_independent_col2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def pose(checkpoint, layout, which):
    """Closed-form original two sibling sliders; no simulator FK."""
    q = np.asarray(
        checkpoint.full_position if "affine" in layout else checkpoint.position,
        dtype=np.float64,
    )
    if layout.startswith("floating"):
        r, t, scalar, extra = rotation(q[3:7]), q[:3], q[7:], 1
    else:
        r, t, scalar, extra = np.eye(3), np.zeros(3), q, 0
    index = 0 if which == "driver" else 1
    offset = -1 / 4 if index == 0 else 1 / 4
    return t + r @ np.array((offset + scalar[index], 0, 0)), r, index + extra


def local_round(kind):
    first = np.array(ROUND, dtype=np.float64)
    return first[None] if kind == "sphere" else np.array((first, first + (1 / 4, 0, 0)))


def world_points(checkpoint, layout, which, local):
    position, frame, _ = pose(checkpoint, layout, which)
    return position + np.asarray(local, dtype=np.float64) @ frame.T


def expected_moving(checkpoint, layout, static_triangle=None):
    triangle = (
        world_points(checkpoint, layout, "driver", TRIANGLE)
        if static_triangle is None
        else np.asarray(static_triangle)
    )
    return triangle, {
        kind: world_points(checkpoint, layout, "follower", local_round(kind))
        for kind in ("sphere", "capsule")
    }


def memory_layout(shapes, points, groups, candidate_counts):
    """Independent, fixed S0 formula; groups are (facets, round_count)."""
    if len(groups) != len(candidate_counts):
        raise ValueError("one candidate count per group")
    persistent = 32 * shapes + 36 * points
    global_query = 64 * shapes + 36 * points
    group_bytes = []
    for faces, rounds in groups:
        if faces <= 0 or rounds <= 0:
            raise ValueError("only nonempty eligible groups are allocated")
        nodes, words = 2 * faces - 1, (faces + 29) // 30
        group_bytes.append(40 * faces + 48 * nodes + (52 + 4 * words) * rounds)
    prepass = persistent + global_query + sum(group_bytes)
    narrow = 1060 * sum(candidate_counts)
    return dict(
        persistent_bytes=persistent,
        query_global_bytes=global_query,
        group_bytes=tuple(group_bytes),
        prepass_required_bytes=prepass,
        narrow_bytes=narrow,
        peak_required_bytes=prepass + narrow,
    )


def plain_flat(case, kind):
    triangle = np.array(((0, 0, 0), (2, 0, 0), (0, 2, 0)), dtype=np.float64)
    first = {
        "front": (0.5, 0.5, 0.25),
        "back": (0.5, 0.5, -0.25),
        "edge": (1, -0.25, 0.25),
        "vertex": (-0.25, -0.25, 0.25),
        "outside": (4, 4, 0.25),
        "through": (0.5, 0.5, 0),
    }[case]
    points = np.array((first,), dtype=np.float64)
    if kind == "capsule":
        points = np.repeat(points, 2, axis=0)
        if case == "through":
            points[:, 2] = (-0.25, 0.25)
        else:
            points[1, 2] += -0.25 if case == "back" else 0.25
    return triangle, points, 0.01 if case == "outside" else 0.5


def mesh(case):
    if case == "concave":
        outline = ((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2))
        points = np.array([(x, y, z) for z in (0, 1) for x, y in outline], dtype=np.float64)
        n = 6
        caps = ((0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5))
        faces = list(caps) + [tuple(v + n for v in face) for face in caps]
        center, margin = (1.25, 1.25, 0.5), 0.2
    else:
        points = np.array(
            (
                (0, 0, 0),
                (2, 0, 0),
                (2, 2, 0),
                (0, 2, 0),
                (0, 0, 2),
                (2, 0, 2),
                (2, 2, 2),
                (0, 2, 2),
            ),
            dtype=np.float64,
        )
        n = 4
        faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6)]
        center, margin = ((1, 1, 1), 1.0) if case == "closed_inside" else ((1, 1, 0.25), 0.2)
    for i in range(n):
        j = (i + 1) % n
        faces.extend(((i, j, j + n), (i, j + n, i + n)))
    return points, np.asarray(faces, np.int64), np.asarray(center), margin


def axis(points):
    value = np.asarray(points, dtype=np.float64)
    return np.repeat(value, 2, axis=0) if len(value) == 1 else value


def aabb_candidates(vertices, faces, axes, radii, margin):
    """Independent complete finite AABB pair set; not a product BVH."""
    result = []
    for si, segment in enumerate(axes):
        lo = np.min(segment, axis=0) - radii[si] - margin
        hi = np.max(segment, axis=0) + radii[si] + margin
        for fi, ids in enumerate(faces):
            tri = vertices[list(ids)]
            if np.all(np.min(tri, axis=0) <= hi) and np.all(lo <= np.max(tri, axis=0)):
                result.append((si, fi))
    return tuple(result)


def assert_row(row, triangle, segment, radius, origin, *, state="RESOLVED"):
    """Independent exact oracle plus material address reconstruction."""
    ref = distance_reference()
    tri = np.asarray(triangle, dtype=np.float64)
    seg = axis(segment)
    length = ref.local_scale(tri, seg, radius)
    assert row.status == state
    if state != "RESOLVED":
        assert row.normal is None
        return
    ideal = ref.oracle(tri, seg)
    distance = float(ideal["distance2"]) ** 0.5
    assert abs(row.distance - distance) <= 1e-4 * length
    assert abs(row.gap - (distance - radius)) <= 1e-4 * length
    bary = np.asarray(row.barycentric)
    assert np.min(bary) >= 0 and abs(float(sum(bary)) - 1) <= 8 * U
    assert 0 <= row.axis_parameter <= 1
    ap = seg[0] + row.axis_parameter * (seg[1] - seg[0])
    # Affine reconstruction in the independently authored triangle frame.
    # Do not amplify the admitted FP32 sum residual by a common world translation.
    tp = tri[0] + bary @ (tri - tri[0])
    np.testing.assert_allclose(
        np.asarray(row.axis_point_relative_m) + origin, ap, atol=2e-4 * length, rtol=0
    )
    np.testing.assert_allclose(
        np.asarray(row.triangle_point_relative_m) + origin, tp, atol=2e-4 * length, rtol=0
    )
    assert abs(float(np.linalg.norm(tp - ap)) - distance) <= 2e-4 * length
    if distance <= 32 * U * length:
        assert row.normal is None
    else:
        expected = np.asarray(ideal["witnesses"][0][4], float) - np.asarray(
            ideal["witnesses"][0][3], float
        )
        assert np.linalg.norm(np.asarray(row.normal) - expected / distance) <= 2e-3
