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
        *(
            diameter(points, radius)
            for points, radius in zip(case["points"], case["radii"], strict=True)
        ),
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
    return {
        "gap_m": float(lower - upper),
        "normal": normal.copy(),
        "first_plane_m": float(upper),
        "second_plane_m": float(lower),
    }


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
    core = sum(
        (weight * by_id[node] for node, weight in zip(used, weights, strict=True)), np.zeros(3)
    )
    offset = np.asarray(witness.offset_m, dtype=float)
    surface = np.asarray(witness.surface_relative_m, dtype=float)
    assert (
        offset.shape == surface.shape == (3,)
        and np.isfinite(offset).all()
        and np.isfinite(surface).all()
    )
    assert np.linalg.norm(surface - core - offset) <= tolerance_m
    expected_offset = (1.0 if side == 0 else -1.0) * radius * normal
    assert np.linalg.norm(offset - expected_offset) <= tolerance_m
    if radius == 0:
        assert np.linalg.norm(offset) <= tolerance_m
    projected = points @ normal
    support = max(projected) + radius if side == 0 else min(projected) - radius
    assert abs(float(surface @ normal) - support) <= tolerance_m
    return {
        "reconstruction_error_m": float(np.linalg.norm(surface - core - offset)),
        "support_error_m": abs(float(surface @ normal) - support),
        "core_m": core,
    }


def literal_parallel_precondition(case):
    oracle = parallel_oracle(case)
    assert abs(oracle["gap_m"] - case["gap_m"]) <= 1e-12
    assert np.array_equal(oracle["normal"], case["normal"])
    return oracle


def transformed(case, variant):
    result = {key: value for key, value in case.items()}
    rotation = (
        CYCLE
        if variant in ("cycle", "cycle_reverse_ids")
        else TILT
        if variant == "tilt"
        else np.eye(3)
    )
    shift = (
        np.array([2.0, -3.0, 0.25])
        if variant in ("cycle", "cycle_reverse_ids", "tilt")
        else np.zeros(3)
    )
    result["positions"] = tuple(rotation @ position + shift for position in case["positions"])
    result["rotations"] = tuple(rotation @ frame for frame in case["rotations"])
    result["points"] = tuple(points @ rotation.T + shift for points in case["points"])
    result["normal"] = rotation @ case["normal"]
    result["requested_reverse"] = variant in ("reverse", "cycle_reverse_ids")
    result["registration_reverse"] = variant in ("permute", "cycle_reverse_ids")
    result["transform"] = {"variant": variant, "rotation": rotation, "translation": shift}
    if variant == "permute":
        result["local"] = tuple(points[::-1].copy() for points in case["local"])
        result["points"] = tuple(points[::-1].copy() for points in result["points"])
        result["ids"] = tuple(tuple(reversed(ids)) for ids in case["ids"])
    if variant == "cycle_reverse_ids":
        result["keys"] = ((17, 71, 29), (17, 13, 7))
        result["ids"] = tuple(
            tuple(2**31 - 1 - 101 * body - 7 * i for i in range(len(points)))
            for body, points in enumerate(case["local"], 1)
        )
    return result


def finite_case(name):
    descriptions = {
        "end_box": ("CB", -0.01),
        "cap_side": ("SC", -0.01),
        "cap_end": ("SC", -0.01),
        "cap_oblique": ("SC", 0.025),
        "cc_parallel": ("CC", -0.01),
        "cc_skew": ("CC", -0.01),
        "cc_end_end": ("CC", np.sqrt(2.0) / 10 - 0.1),
        "h_slant": ("SH", 0.08 * np.sqrt(2.0) - 0.05),
        "h_vertex": ("SH", 0.05),
    }
    family, gap = descriptions[name]
    base = family_case(family, gap)
    if name == "end_box":
        points = (base["points"][0] + (1.2, 0, 0), base["points"][1])
        kinds = ("C", "B")
        normal = np.array([-5.0, 0.0, 2.0]) / np.sqrt(29.0)
        gap = np.sqrt(29.0) / 50 - 0.05
    elif name.startswith("cap_"):
        center = {
            "cap_side": (0, 0, 0.09),
            "cap_end": (0.59, 0, 0),
            "cap_oblique": (0.575, 0, 0.1),
        }[name]
        points = (primitive("C")[0], np.array([center], dtype=float))
        kinds = ("C", "S")
        normal = np.array(
            {"cap_side": (0, 0, 1), "cap_end": (1, 0, 0), "cap_oblique": (0.6, 0, 0.8)}[name],
            dtype=float,
        )
    elif name.startswith("cc_"):
        other = {
            "cc_parallel": ((-0.25, 0, 0.09), (0.75, 0, 0.09)),
            "cc_skew": ((0, -0.5, 0.09), (0, 0.5, 0.09)),
            "cc_end_end": ((0.6, 0.1, 0), (0.6, 1.1, 0)),
        }[name]
        points = (primitive("C")[0], np.array(other, dtype=float))
        kinds = ("C", "C")
        normal = (
            np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
            if name == "cc_end_end"
            else np.array([0.0, 0.0, 1.0])
        )
    else:
        center = (0.63, -0.13, 0) if name == "h_slant" else (0.68, 0, 0.16)
        points = (primitive("H")[0], np.array([center], dtype=float))
        kinds = ("H", "S")
        normal = (
            np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
            if name == "h_slant"
            else np.array([0.8, 0.0, 0.6])
        )
    base.update(
        family="".join(kinds),
        kinds=kinds,
        points=points,
        local=tuple(point.copy() for point in points),
        positions=(np.zeros(3), np.zeros(3)),
        rotations=(np.eye(3), np.eye(3)),
        ids=tuple(
            tuple(range(100 * side, 100 * side + len(point)))
            for side, point in enumerate(points, 1)
        ),
        radii=tuple(primitive(kind)[1] for kind in kinds),
        gap_m=float(gap),
        normal=normal,
        oracle="finite segment/face/vertex literal",
        finite_case=name,
    )
    return base


def point_segment(point, a, b):
    return np.asarray(a) + segment_parameter(point, a, b) * (np.asarray(b) - a)


def segment_segment(a, b, c, d):
    """Four endpoint projections plus an interior stationary solution."""
    a, b, c, d = (np.asarray(x, dtype=float) for x in (a, b, c, d))
    candidates = [
        (a, point_segment(a, c, d)),
        (b, point_segment(b, c, d)),
        (point_segment(c, a, b), c),
        (point_segment(d, a, b), d),
    ]
    u, v, w = b - a, d - c, a - c
    aa, bb, cc, dd, ee = float(u @ u), float(u @ v), float(v @ v), float(u @ w), float(v @ w)
    determinant = aa * cc - bb * bb
    if determinant > 1e-26 * max(1.0, aa * cc):
        s, t = (bb * ee - cc * dd) / determinant, (aa * ee - bb * dd) / determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append((a + s * u, c + t * v))
    return min(candidates, key=lambda pair: float((pair[1] - pair[0]) @ (pair[1] - pair[0])))


def supporting_triangles(points):
    """Exhaustive independent FP64 hull surface; no GJK/EPA or product calls."""
    from itertools import combinations

    xyz = np.asarray(points, dtype=float)
    triangles = []
    for i, j, k in combinations(range(len(xyz)), 3):
        normal = np.cross(xyz[j] - xyz[i], xyz[k] - xyz[i])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-14:
            continue
        normal /= norm
        distance = (xyz - xyz[i]) @ normal
        if float(distance.max()) <= 1e-12:
            triangles.append((i, j, k))
        elif float(distance.min()) >= -1e-12:
            triangles.append((i, k, j))
    if not triangles:
        raise ValueError("independent finite hull is not full dimensional")
    return triangles


def point_triangle(point, a, b, c):
    point = np.asarray(point)
    candidates = [
        point_segment(point, a, b),
        point_segment(point, b, c),
        point_segment(point, c, a),
    ]
    u, v, w = b - a, c - a, point - a
    aa, ab, bb = float(u @ u), float(u @ v), float(v @ v)
    det = aa * bb - ab * ab
    if det > 1e-28:
        s = (bb * float(w @ u) - ab * float(w @ v)) / det
        t = (aa * float(w @ v) - ab * float(w @ u)) / det
        if s >= 0 and t >= 0 and s + t <= 1:
            candidates.append(a + s * u + t * v)
    return min(candidates, key=lambda x: float((point - x) @ (point - x)))


def point_inside(point, xyz, triangles):
    for i, j, k in triangles:
        normal = np.cross(xyz[j] - xyz[i], xyz[k] - xyz[i])
        if float((point - xyz[i]) @ normal) > 1e-12 * float(np.linalg.norm(normal)):
            return False
    return True


def core_hull_closest(core, xyz):
    triangles = supporting_triangles(xyz)
    candidates = []
    for p in core:
        if point_inside(p, xyz, triangles):
            return p.copy(), p.copy()
        for i, j, k in triangles:
            candidates.append((p, point_triangle(p, xyz[i], xyz[j], xyz[k])))
    if len(core) == 2:
        a, b = core
        for i, j, k in triangles:
            ta, tb, tc = xyz[i], xyz[j], xyz[k]
            normal = np.cross(tb - ta, tc - ta)
            denominator = float((b - a) @ normal)
            if denominator != 0:
                t = float((ta - a) @ normal) / denominator
                if 0 <= t <= 1:
                    hit = a + t * (b - a)
                    if np.linalg.norm(point_triangle(hit, ta, tb, tc) - hit) <= 1e-12:
                        return hit, hit.copy()
            for u, v in ((ta, tb), (tb, tc), (tc, ta)):
                candidates.append(segment_segment(a, b, u, v))
    return min(candidates, key=lambda pair: float((pair[1] - pair[0]) @ (pair[1] - pair[0])))


def polytope_oracle(first, second):
    ta, tb = supporting_triangles(first), supporting_triangles(second)

    def features(xyz, triangles):
        edges = sorted(
            {tuple(sorted(edge)) for i, j, k in triangles for edge in ((i, j), (j, k), (k, i))}
        )
        normals = [np.cross(xyz[j] - xyz[i], xyz[k] - xyz[i]) for i, j, k in triangles]
        return edges, normals

    ea, na = features(first, ta)
    eb, nb = features(second, tb)
    axes = [
        *na,
        *nb,
        *(np.cross(first[j] - first[i], second[end] - second[k]) for i, j in ea for k, end in eb),
    ]
    penetration = None
    separated = False
    for axis in axes:
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-14:
            continue
        normal = axis / norm
        a, b = first @ normal, second @ normal
        forward, backward = float(a.max() - b.min()), float(b.max() - a.min())
        if forward < 0 or backward < 0:
            separated = True
        for depth, direction in ((forward, normal), (backward, -normal)):
            if penetration is None or depth < penetration[0]:
                penetration = (depth, direction)
    if not separated:
        return {
            "gap_m": -penetration[0],
            "normal": penetration[1],
            "method": "full finite polytope SAT face/edge axes",
        }
    candidates = []
    for p in first:
        for i, j, k in tb:
            candidates.append((p, point_triangle(p, second[i], second[j], second[k])))
    for p in second:
        for i, j, k in ta:
            candidates.append((point_triangle(p, first[i], first[j], first[k]), p))
    for i, j in ea:
        for k, end in eb:
            candidates.append(segment_segment(first[i], first[j], second[k], second[end]))
    a, b = min(candidates, key=lambda pair: float((pair[1] - pair[0]) @ (pair[1] - pair[0])))
    delta = b - a
    length = float(np.linalg.norm(delta))
    return {
        "gap_m": length,
        "normal": delta / length,
        "method": "exhaustive finite vertex/face and edge/edge distances",
    }


def finite_oracle(kinds, points, radii):
    first, second = (np.asarray(p, dtype=float) for p in points)
    round_a, round_b = kinds[0] in ("S", "C"), kinds[1] in ("S", "C")
    if not round_a and not round_b:
        return polytope_oracle(first, second)
    if round_a and round_b:
        if len(first) == len(second) == 1:
            a, b = first[0], second[0]
        elif len(first) == 1:
            a, b = first[0], point_segment(first[0], *second)
        elif len(second) == 1:
            a, b = point_segment(second[0], *first), second[0]
        else:
            a, b = segment_segment(*first, *second)
    elif round_a:
        a, b = core_hull_closest(first, second)
    else:
        b, a = core_hull_closest(second, first)
    delta = b - a
    length = float(np.linalg.norm(delta))
    if length == 0:
        return {"gap_m": None, "normal": None, "core_containment_or_coincidence": True}
    return {
        "gap_m": length - sum(radii),
        "normal": delta / length,
        "first_core_m": a,
        "second_core_m": b,
        "method": "independent finite core metric projection plus skin",
    }


def stored_oracle(case, source):
    """Reconstruct actual uploaded geometry, never use returned query features."""
    points, radii = [], []
    keys = tuple(tuple((key.world, key.body, key.shape)) for key in source["shape_keys"])
    offsets = np.asarray(source["vertex_offsets"])
    for key in case["keys"]:
        row = keys.index(tuple(key))
        local = np.asarray(
            source["uploaded_local_points_m"][offsets[row] : offsets[row + 1]], dtype=float
        )
        pose = np.asarray(source["pose"][row], dtype=float)
        origin = np.asarray(source["world_origins_m"][source["shape_world_rows"][row]], dtype=float)
        points.append(origin + pose[:3] + local @ pose[3:].reshape(3, 3).T)
        radii.append(float(source["uploaded_radius_m"][row]))
    result = finite_oracle(case["kinds"], points, radii)
    return {**result, "points_m": tuple(points), "radii_m": tuple(radii)}


DOMAIN_IDS = (
    "coincident_ss",
    "deep_center",
    "deep_offset",
    "identical_hh",
    "zero_axis",
    "flat_h_far",
    "duplicate_xyz_far",
    "bad_address-repeat_id",
    "bad_address-missing_point",
    "nonfinite-nan",
    "nonfinite-inf",
    "rounded_h",
    "h_capacity-v64",
    "h_capacity-v65",
    "origin-rebase",
    "origin-loss",
)


def domain_case(name):
    if name == "origin-rebase":
        case = family_case("SH", 0.02)
        shift = np.array([2**20, -(2**20), 2**20], dtype=float)
        case["positions"] = tuple(position + shift for position in case["positions"])
        case["points"] = tuple(points + shift for points in case["points"])
        return case
    if name == "origin-loss":
        case = family_case("SB", 0.02)
        case.update(kinds=("B", "S"), family="BS")
        local = (primitive("B")[0], primitive("S")[0])
        origin = np.array([2**60, 0.0, 0.0])
        positions = (origin, origin + (0, 0, 0.17))
        case.update(
            local=local,
            positions=positions,
            points=tuple(p + x for p, x in zip(local, positions, strict=True)),
            radii=(0.0, 0.05),
            ids=(tuple(range(100, 108)), (200,)),
            oracle="declared B local x half-width lost in FP64 world expansion; SOURCE_PRECISION required",
        )
        return case
    if name in ("coincident_ss", "identical_hh"):
        family = "SS" if name == "coincident_ss" else "HH"
        case = family_case(family, -0.01)
        case["points"] = tuple(primitive(kind)[0] for kind in family)
        case["local"] = tuple(points.copy() for points in case["points"])
        case["positions"] = (np.zeros(3), np.zeros(3))
        return case
    if name in ("deep_center", "deep_offset"):
        case = family_case("SB", -0.01)
        first = np.array([[0.0, 0.0, 0.02 if name == "deep_offset" else 0.0]])
        second = primitive("B")[0]
        case.update(
            points=(first, second),
            local=(first.copy(), second.copy()),
            positions=(np.zeros(3), np.zeros(3)),
            expected_exit_depth_m=0.13 if name == "deep_offset" else 0.15,
        )
        return case
    if name == "zero_axis":
        case = finite_case("cap_side")
        points = (np.zeros((2, 3)), np.array([[0.0, 0.0, 0.12]]))
        case.update(points=points, local=tuple(p.copy() for p in points), gap_m=0.02)
        return case
    case = finite_case("h_vertex")
    first = primitive("H")[0].copy()
    center = np.array([[0.0, 0.0, 0.17]])
    if name == "flat_h_far":
        first = np.array(list(product((-0.5, 0.5), (-0.1, 0.1), (0.0,))))
        center = np.array([[10.0, 0.0, 0.0]])
    elif name == "duplicate_xyz_far":
        first[-1] = first[0]
        center = np.array([[10.0, 0.0, 0.0]])
    elif name.startswith("nonfinite-"):
        first[0, 0] = float("nan") if name.endswith("nan") else float("inf")
    elif name.startswith("h_capacity-"):
        first = np.array(
            [
                (0.5 * np.cos(2 * np.pi * j / 32), 0.5 * np.sin(2 * np.pi * j / 32), z)
                for j in range(32)
                for z in (-0.1, 0.1)
            ]
        )
        if name.endswith("v65"):
            first = np.vstack((first, (0.0, 0.0, 0.1)))
    ids = tuple(range(100, 100 + len(first)))
    if name == "bad_address-repeat_id":
        ids = (*ids[:-1], ids[0])
    elif name == "bad_address-missing_point":
        first = first[:-1]
    case.update(
        points=(first, center),
        local=(first.copy(), center.copy()),
        ids=(ids, (200,)),
        positions=(np.zeros(3), np.zeros(3)),
        radii=(0.01 if name == "rounded_h" else 0.0, 0.05),
        gap_m=0.02,
        normal=np.array([0.0, 0.0, 1.0]),
    )
    return case
