"""COL1a independent literals and inscribed-ball depth certificates.

NumPy/stdlib only: no Phymeo, TYX, Newton, GJK or EPA implementation is called.
The negative oracle applies to the declared geometry, not arbitrary axial overlap.
"""

from fractions import Fraction
from itertools import combinations_with_replacement, product

import numpy as np

KINDS = ("S", "C", "B", "H", "Y", "N", "E")
PAIRS = tuple("".join(pair) for pair in combinations_with_replacement(KINDS, 2))
NEW_PAIRS = tuple(pair for pair in PAIRS if "Y" in pair or "N" in pair or "E" in pair)
GAPS = {"sep": 1 / 32, "touch": 0.0, "overlap": -1 / 64}
CYCLE_Q = (0.5, 0.5, 0.5, 0.5)
TILT_Q = (0.0, 0.6, 0.0, 0.8)
DIRECTIONS = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
    (1, 1, 0),
    (1, -1, 0),
    (0, 1, 1),
    (0, 1, -1),
    (1, 0, 1),
    (1, 0, -1),
    (2, 3, 6),
)
RING = np.array(((-1 / 4, -1 / 8), (1 / 4, -1 / 8), (3 / 8, 0), (1 / 8, 1 / 4), (-1 / 4, 1 / 8)))


def rotation(quaternion):
    x, y, z, w = np.asarray(quaternion, float)
    norm = x * x + y * y + z * z + w * w
    assert abs(norm - 1) < 1e-12
    return np.array(
        (
            (
                1 - 2 * (y * y + z * z) / norm,
                2 * (x * y - z * w) / norm,
                2 * (x * z + y * w) / norm,
            ),
            (
                2 * (x * y + z * w) / norm,
                1 - 2 * (x * x + z * z) / norm,
                2 * (y * z - x * w) / norm,
            ),
            (
                2 * (x * z - y * w) / norm,
                2 * (y * z + x * w) / norm,
                1 - 2 * (x * x + y * y) / norm,
            ),
        )
    )


def primitive(kind):
    if kind == "S":
        return np.zeros((1, 3)), 1 / 8, (), 1 / 8
    if kind == "C":
        return np.array(((-1 / 4, 0, 0), (1 / 4, 0, 0))), 1 / 16, (), 1 / 16
    if kind == "B":
        return (
            np.array(list(product((-3 / 8, 3 / 8), (-1 / 4, 1 / 4), (-1 / 8, 1 / 8)))),
            0.0,
            (),
            1 / 8,
        )
    if kind == "H":
        return np.array([(x, y, z) for x, y in RING for z in (-1 / 8, 1 / 8)]), 0.0, (), 1 / 8
    dimensions = {"Y": (1 / 8, 1 / 4), "N": (3 / 16, 1 / 4), "E": (1 / 4, 3 / 16, 1 / 8)}[kind]
    return np.zeros((1, 3)), 0.0, dimensions, dimensions[-1]


def support(kind, direction):
    d = np.asarray(direction, float)
    norm = float(np.linalg.norm(d))
    if norm == 0:
        raise ValueError("zero direction has no selected support")
    points, radius, dims, _ = primitive(kind)
    if kind not in "YNE":
        return points[int(np.argmax(points @ d))] + radius * d / norm
    if kind == "E":
        axes = np.asarray(dims)
        return axes * axes * d / np.linalg.norm(axes * d)
    radial = float(np.linalg.norm(d[:2]))
    if kind == "N" and 2 * dims[1] * d[2] >= dims[0] * radial:
        return np.array((0.0, 0.0, dims[1]))
    xy = np.zeros(2) if radial == 0 else dims[0] * d[:2] / radial
    height = -dims[1] if kind == "N" else dims[1] * np.sign(d[2])
    return np.r_[xy, height]


def set_residual(kind, point):
    """Nonnegative metres outside the declared closed solid, not a product oracle."""
    p = np.asarray(point, float)
    xyz, radius, dims, _ = primitive(kind)
    if kind == "S":
        return max(0.0, np.linalg.norm(p) - radius)
    if kind == "C":
        q = np.array((np.clip(p[0], -1 / 4, 1 / 4), 0.0, 0.0))
        return max(0.0, np.linalg.norm(p - q) - radius)
    if kind == "B":
        return max(0.0, float(np.max(abs(p) - np.array((3 / 8, 1 / 4, 1 / 8)))))
    if kind == "H":
        violations = [abs(p[2]) - 1 / 8]
        for first, second in zip(RING, np.roll(RING, -1, axis=0), strict=True):
            edge = second - first
            violations.append(
                -(edge[0] * (p[1] - first[1]) - edge[1] * (p[0] - first[0])) / np.linalg.norm(edge)
            )
        return max(0.0, *violations)
    if kind == "E":
        return max(0.0, (np.linalg.norm(p / np.asarray(dims)) - 1) * max(dims))
    radial = np.linalg.norm(p[:2])
    if kind == "Y":
        return max(0.0, radial - dims[0], abs(p[2]) - dims[1])
    side = radial + dims[0] * p[2] / (2 * dims[1]) - dims[0] / 2
    return max(0.0, -p[2] - dims[1], side / np.sqrt(1 + (dims[0] / (2 * dims[1])) ** 2))


def diameter(kind):
    points, radius, dims, _ = primitive(kind)
    if kind in "YN":
        return 2 * sum(dims)
    if kind == "E":
        return 2 * max(dims)
    return max(np.linalg.norm(a - b) for a in points for b in points) + 2 * radius


def case(family, phase, *, cycle=False, oblique=False, world=17):
    assert len(family) == 2 and all(k in KINDS for k in family)
    bodies = [primitive(k) for k in family]
    n = np.array((2 / 7, 3 / 7, 6 / 7)) if oblique else np.array((0.0, 0.0, 1.0))
    gap = GAPS[phase]
    first_support, second_support = support(family[0], n), support(family[1], -n)
    # Axis cases use the axis contact point even if a face support routine selects a corner.
    shift = (
        first_support - second_support + gap * n
        if oblique
        else np.array((0.0, 0.0, bodies[0][3] + bodies[1][3] + gap))
    )
    quaternion = CYCLE_Q if cycle else (0.0, 0.0, 0.0, 1.0)
    frame = rotation(quaternion)
    translation = np.array((4.0, -2.0, 1.0)) if cycle else np.zeros(3)
    centers = (translation, translation + frame @ shift)
    return dict(
        family=family,
        kinds=tuple(family),
        phase=phase,
        placement_m=gap,
        expected_gap_m=gap,
        normal=frame @ n,
        world=world,
        keys=((world, 11, 3), (world, 29, 7)),
        ids=tuple(
            tuple(range(101 + 100 * side, 101 + 100 * side + len(body[0])))
            for side, body in enumerate(bodies)
        ),
        local=tuple(body[0].copy() for body in bodies),
        centers=centers,
        frames=(frame.copy(), frame.copy()),
        points=tuple(
            body[0] @ frame.T + center for body, center in zip(bodies, centers, strict=True)
        ),
        radii=tuple(body[1] for body in bodies),
        dimensions=tuple(body[2] for body in bodies),
        body_quaternion=quaternion,
        static_quaternion=quaternion,
        requested_reverse=cycle,
        oblique=oblique,
    )


def scale(data):
    # Match the frozen kernel's vertex-mean centre convention for finite H.
    centers = tuple(np.mean(points, axis=0) for points in data["points"])
    return max(
        1.0, *(diameter(kind) for kind in data["kinds"]), np.linalg.norm(centers[1] - centers[0])
    )


def depth_certificate(data):
    """Exact dyadic inscribed-ball lower bound, independent supporting-plane upper bound.

    Every G01 second shape is Y/N/E. Let delta=-placement. A contains its
    top-axis support point. B contains a bottom-tangent ball radius R>delta.
    Thus A-B contains ball(R, (delta-R)ez), itself containing ball(delta,0).
    The +ez support is delta: min translation depth is exactly delta and the
    minimizing normal is unique. This is NOT a general placement=depth claim.
    """
    assert data["phase"] == "overlap" and not data["oblique"]
    a, b = data["kinds"]
    assert b in "YNE"
    delta = -Fraction(data["placement_m"])
    R = Fraction(1, 8) if b == "E" else Fraction(1, 32)
    _, _, dimensions, height = primitive(b)
    h = Fraction(height)
    assert 0 < delta < R
    if b == "Y":
        assert R <= Fraction(dimensions[0]) and R <= h
    elif b == "N":
        r = Fraction(dimensions[0])
        # Distance from axis centre (-h+R) to every cone side >= R.
        assert (r * (2 * h - R)) ** 2 >= R * R * ((2 * h) ** 2 + r * r)
    else:
        assert R == h and all(Fraction(v) >= R for v in dimensions)
    top = np.array((0.0, 0.0, primitive(a)[3]))
    assert set_residual(a, top) <= 1e-14
    assert abs(float(support(a, (0, 0, 1))[2]) - top[2]) < 1e-14
    assert abs(float(support(b, (0, 0, -1))[2]) + height) < 1e-14
    return dict(
        kind="inscribed-ball-and-support-plane-v1",
        first_top=top,
        second_ball_radius_m=float(R),
        second_ball_center_local_m=(0.0, 0.0, float(-h + R)),
        minkowski_ball_center_m=(0.0, 0.0, float(delta - R)),
        lower_depth_m=float(delta),
        upper_depth_m=float(delta),
        unique_normal_local=(0.0, 0.0, 1.0),
        strict_interior_overlap=True,
    )


def oracle(data):
    if data["phase"] == "overlap":
        proof = depth_certificate(data)
        gap = -proof["lower_depth_m"]
    else:
        proof = dict(kind="supporting-plane-separated-point-v1", strict_interior_overlap=False)
        gap = data["placement_m"]
    return dict(gap_m=gap, normal=np.asarray(data["normal"]), proof=proof, scale_m=scale(data))


def witness_check(data, side, witness, origin, normal, tolerance):
    used, weights = tuple(witness.vertex_ids), np.asarray(witness.weights)
    ids = data["ids"][side]
    assert witness.valid and len(used) == len(weights) and 1 <= len(used) <= 4
    assert len(set(used)) == len(used) and set(used) <= set(ids)
    assert np.all(weights >= 0) and abs(float(weights.sum()) - 1) <= 64 * 2**-23
    core = sum(
        (w * data["points"][side][ids.index(i)] for i, w in zip(used, weights, strict=True)),
        np.zeros(3),
    )
    world_surface = np.asarray(witness.surface_relative_m) + origin
    assert np.linalg.norm(world_surface - core - np.asarray(witness.offset_m)) <= tolerance
    kind, frame, center = data["kinds"][side], data["frames"][side], data["centers"][side]
    local = frame.T @ (world_surface - center)
    d = (1.0 if side == 0 else -1.0) * frame.T @ normal
    expected_support = float(support(kind, d) @ d)
    support_error = abs(float(local @ d) - expected_support)
    containment = set_residual(kind, local)
    assert support_error <= tolerance and containment <= tolerance
    if kind in "YNE":
        assert used == ids and len(used) == 1 and abs(float(weights[0]) - 1) <= 64 * 2**-23
    return dict(
        reconstruction_error_m=float(
            np.linalg.norm(world_surface - core - np.asarray(witness.offset_m))
        ),
        support_error_m=support_error,
        containment_error_m=containment,
    )
