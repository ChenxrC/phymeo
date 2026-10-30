"""Exact symmetric geometry for two legacy finite-thickness triangle prisms.

Every maximal SAT tie is evidence. This module makes no friction-normal choice
and grants no material-incidence exemption.
"""

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256

import numpy as np

from phymeo.geometry._prism_pair_exact import (
    PairField,
    PrismPairProofLimits,
    SignUnresolved,
    compare_gaps,
    normalized_bounds,
)


def _array(value, shape, dtype, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must be an ndarray with dtype {np.dtype(dtype)}")
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return np.array(value, copy=True, order="C")


@dataclass(frozen=True, init=False)
class PrismPairIdentity:
    domain: str
    _face_bytes: bytes
    _node_bytes: bytes

    def __init__(self, domain, face_ids, node_ids):
        if not isinstance(domain, str) or not domain or domain.strip() != domain:
            raise ValueError("domain must be a nonempty unpadded string")
        faces = _array(face_ids, (2,), np.int64, "face_ids")
        nodes = _array(node_ids, (2, 3), np.int64, "node_ids")
        if faces[0] == faces[1]:
            raise ValueError("face IDs must be distinct")
        if np.any(nodes < 0) or any(len(set(map(int, row))) != 3 for row in nodes):
            raise ValueError("each oriented node triple needs three distinct nonnegative IDs")
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "_face_bytes", faces.tobytes())
        object.__setattr__(self, "_node_bytes", nodes.tobytes())

    @property
    def face_ids(self):
        return np.frombuffer(self._face_bytes, dtype=np.int64)

    @property
    def node_ids(self):
        return np.frombuffer(self._node_bytes, dtype=np.int64).reshape(2, 3)


@dataclass(frozen=True)
class ProofObligation:
    stage: str
    reason: str
    remaining: tuple = ()


@dataclass(frozen=True)
class PrismPairPoint:
    point_a: tuple
    point_b: tuple
    material_a: tuple
    material_b: tuple
    key: tuple


@dataclass(frozen=True)
class PrismPairAxis:
    canonical_direction: tuple
    direction: tuple
    norm_squared: Fraction
    numerator: object
    support_a: tuple
    support_b: tuple
    key: tuple
    points: tuple = ()

    @property
    def gap_bounds(self):
        return normalized_bounds(self.numerator, self.norm_squared)


@dataclass(frozen=True)
class PrismPairManifold:
    status: str
    fingerprint: str
    caller_fingerprint: str
    identity_token: tuple
    caller_to_canonical: tuple
    axes: tuple
    maximal_axes: tuple
    vertices: tuple
    exact_gap_bounds: tuple | None
    sat_complete: bool
    obligation: ProofObligation | None = None


@dataclass(frozen=True)
class _PairInput:
    identity_token: tuple
    caller_to_canonical: tuple
    canonical_to_caller: tuple
    vertex_permutations: tuple
    triangles: tuple
    thickness: tuple
    fingerprint: str
    caller_fingerprint: str


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _scale(a, s):
    return tuple(x * s for x in a)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _fraction_array(array):
    return tuple(tuple(Fraction.from_float(float(x)) for x in row) for row in array)


def _digest(value):
    return sha256(repr(value).encode("utf-8")).hexdigest()


def _prepare(x_a, x_b, identity, thickness, limits):
    if not isinstance(identity, PrismPairIdentity):
        raise TypeError("identity must be PrismPairIdentity")
    if not isinstance(limits, PrismPairProofLimits):
        raise TypeError("proof_limits must be PrismPairProofLimits")
    arrays = [_array(x, (3, 3), np.float64, name) for x, name in ((x_a, "x_a"), (x_b, "x_b"))]
    rho = _array(thickness, (2,), np.float64, "half_thickness_m")
    if np.any(rho <= 0):
        raise ValueError("half thickness must be positive")
    for i, node in enumerate(identity.node_ids[0]):
        for j, other in enumerate(identity.node_ids[1]):
            if node == other and not np.array_equal(arrays[0][i], arrays[1][j]):
                raise ValueError("shared node IDs require exactly equal stored coordinates")
    original = tuple(_fraction_array(x) for x in arrays)
    original_rho = tuple(Fraction.from_float(float(r)) for r in rho)
    order = tuple(sorted(range(2), key=lambda i: int(identity.face_ids[i])))
    permutations, triples = [], []
    for side in order:
        ids = tuple(map(int, identity.node_ids[side]))
        start = min(range(3), key=lambda i: ids[i:] + ids[:i])
        permutation = tuple((start + i) % 3 for i in range(3))
        permutations.append(permutation)
        triples.append(tuple(ids[i] for i in permutation))
    triangles = tuple(tuple(original[side][i] for i in perm) for side, perm in zip(order, permutations, strict=True))
    canonical_rho = tuple(original_rho[side] for side in order)
    token = (identity.domain, tuple(int(identity.face_ids[i]) for i in order), tuple(triples))
    caller_order = tuple(order.index(i) for i in range(2))
    fingerprint = _digest(("prism-pair-static-v1", token, triangles, canonical_rho, limits.token))
    caller_fingerprint = _digest((fingerprint, caller_order, tuple(permutations)))
    return _PairInput(token, caller_order, order, tuple(permutations), triangles, canonical_rho, fingerprint, caller_fingerprint)


def _directions(triangles):
    edges, faces = [], []
    for x in triangles:
        e, h = _sub(x[1], x[0]), _sub(x[2], x[0])
        c = _cross(e, h)
        edge = (e, h, _sub(h, e), c)
        edges.append(edge)
        faces.extend((c,) + tuple(_cross(v, c) for v in edge[:3]))
    axes = faces + [_cross(a, b) for a in edges[0] for b in edges[1]]
    canonical = set()
    for axis in axes:
        first = next((x for x in axis if x), None)
        if first is not None:
            canonical.add(tuple(x / first for x in axis))
    return tuple(sorted(d for axis in canonical for d in (axis, _scale(axis, -1))))


def _geometry(triangles, thickness, limits):
    areas = tuple(_cross(_sub(x[1], x[0]), _sub(x[2], x[0])) for x in triangles)
    squares = tuple(_dot(c, c) for c in areas)
    if not all(squares):
        raise ValueError("exact zero triangle area")
    field = PairField(*squares, limits)
    normals = tuple(tuple(field.scalar(v) / field.sqrt(c) for v in area) for area, c in zip(areas, squares, strict=True))
    vertices = tuple(tuple(_add(tuple(field.scalar(v) for v in point), _scale(normal, sign * rho)) for point in triangle for sign in (-1, 1)) for triangle, normal, rho in zip(triangles, normals, thickness, strict=True))
    return field, normals, vertices


def _support(vertices, direction):
    values = tuple(_dot(vertex, direction) for vertex in vertices)
    low, high = min(values), max(values)
    return low, high, tuple(i for i, x in enumerate(values) if x == low), tuple(i for i, x in enumerate(values) if x == high)


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _hull(points):
    points = sorted(set(points))
    if len(points) < 3:
        return tuple(points)
    lower, upper = [], []
    for target, source in ((lower, points), (upper, reversed(points))):
        for p in source:
            while len(target) >= 2 and _orient(target[-2], target[-1], p) <= 0:
                target.pop()
            target.append(p)
    return tuple(lower[:-1] + upper[:-1])


def _edges(polygon):
    if len(polygon) == 1:
        return ((polygon[0], polygon[0]),)
    if len(polygon) == 2:
        return ((polygon[0], polygon[1]),)
    return tuple(zip(polygon, polygon[1:] + polygon[:1], strict=True))


def _inside(p, polygon):
    if len(polygon) == 1:
        return p == polygon[0]
    if len(polygon) == 2:
        a, b = polygon
        return _orient(a, b, p) == 0 and all(min(a[i], b[i]) <= p[i] <= max(a[i], b[i]) for i in range(2))
    return all(_orient(a, b, p) >= 0 for a, b in _edges(polygon))


def _intersection(a, b):
    points = [p for p in a if _inside(p, b)] + [p for p in b if _inside(p, a)]
    for p, q in _edges(a):
        for r, s in _edges(b):
            u, v, delta = _sub(q, p), _sub(s, r), _sub(r, p)
            det = u[0] * v[1] - u[1] * v[0]
            if det == 0:
                continue
            t = (delta[0] * v[1] - delta[1] * v[0]) / det
            z = (delta[0] * u[1] - delta[1] * u[0]) / det
            if 0 <= t <= 1 and 0 <= z <= 1:
                points.append(_add(p, _scale(u, t)))
    return _hull(points)


def _clip(a, b, direction):
    q = _dot(direction, direction)
    drop = next(i for i, x in enumerate(direction) if x)
    keep = tuple(i for i in range(3) if i != drop)
    projected = []
    for points in (a, b):
        projected.append(_hull(tuple(tuple((_sub(p, _scale(direction, _dot(p, direction) / q)))[i] for i in keep) for p in points)))
    intersection = _intersection(*projected)
    supports = (_dot(a[0], direction), _dot(b[0], direction))
    result = []
    for point in intersection:
        perpendicular = [None, None, None]
        for i, value in zip(keep, point, strict=True):
            perpendicular[i] = value
        perpendicular[drop] = -sum(direction[i] * perpendicular[i] for i in keep) / direction[drop]
        result.append(tuple(_add(perpendicular, _scale(direction, support / q)) for support in supports))
    return tuple(result)


def _material(point, triangle, normal, rho):
    e, h = _sub(triangle[1], triangle[0]), _sub(triangle[2], triangle[0])
    offset = _dot(normal, _sub(point, triangle[0]))
    delta = _sub(_sub(point, _scale(normal, offset)), triangle[0])
    ee, eh, hh = _dot(e, e), _dot(e, h), _dot(h, h)
    determinant = ee * hh - eh * eh
    de, dh = _dot(delta, e), _dot(delta, h)
    b1, b2 = (de * hh - dh * eh) / determinant, (dh * ee - de * eh) / determinant
    weights = (1 - b1 - b2, b1, b2)
    reconstruction = tuple(sum(weights[i] * triangle[i][j] for i in range(3)) + offset * normal[j] for j in range(3))
    if any(b < 0 for b in weights) or sum(weights) != 1 or not -rho <= offset <= rho or reconstruction != point:
        raise ArithmeticError("clipped point failed exact material reconstruction")
    return weights + (offset,)


def _caller_material(material, permutation):
    weights = [None] * 3
    for i, original in enumerate(permutation):
        weights[original] = material[i]
    return tuple(weights) + material[3:]


def _caller_support(indices, permutation):
    return tuple(sorted(2 * permutation[i // 2] + i % 2 for i in indices))


def _manifold_exact(prepared, triangles, limits):
    """Internal rational-time entry; never converts times/coordinates to float."""
    axes, maximal, caller_vertices = [], [], ()
    sat_complete, complete_gap = False, None
    fingerprint = _digest(("prism-pair-static-v1", prepared.identity_token, triangles, prepared.thickness, limits.token))
    caller_fingerprint = _digest((fingerprint, prepared.caller_to_canonical, prepared.vertex_permutations))

    def result(status, gap=None, obligation=None):
        return PrismPairManifold(status, fingerprint, caller_fingerprint, prepared.identity_token, prepared.caller_to_canonical, tuple(axes), tuple(maximal), caller_vertices, complete_gap if gap is None else gap, sat_complete, obligation)

    try:
        _, normals, vertices = _geometry(triangles, prepared.thickness, limits)
        sides = [None, None]
        for canonical, caller in enumerate(prepared.canonical_to_caller):
            row = [None] * 6
            for i, vertex in enumerate(vertices[canonical]):
                row[2 * prepared.vertex_permutations[canonical][i // 2] + i % 2] = vertex
            sides[caller] = tuple(row)
        caller_vertices = tuple(sides)
        for direction in _directions(triangles):
            low, _, support_a, _ = _support(vertices[0], direction)
            _, high, _, support_b = _support(vertices[1], direction)
            supports = [_caller_support(support_a, prepared.vertex_permutations[0]), _caller_support(support_b, prepared.vertex_permutations[1])]
            caller_supports = tuple(supports[i] for i in prepared.caller_to_canonical)
            axis = PrismPairAxis(direction, direction if prepared.caller_to_canonical[0] == 0 else _scale(direction, -1), _dot(direction, direction), low - high, *caller_supports, (prepared.identity_token, direction))
            axes.append(axis)
            if not maximal:
                maximal = [axis]
            else:
                comparison = compare_gaps(axis.numerator, axis.norm_squared, maximal[0].numerator, maximal[0].norm_squared)
                if comparison > 0:
                    maximal = [axis]
                elif comparison == 0:
                    maximal.append(axis)
        complete_gap = maximal[0].gap_bounds
        sat_complete = True
        completed = []
        for axis in maximal:
            d = axis.canonical_direction
            _, _, ia, _ = _support(vertices[0], d)
            _, _, _, ib = _support(vertices[1], d)
            clips = _clip(tuple(vertices[0][i] for i in ia), tuple(vertices[1][i] for i in ib), d)
            if not clips:
                return result("unresolved", obligation=ProofObligation("clipping", "selected support projections have no proved nonempty clip", (axis.key,)))
            points = []
            for pa, pb in clips:
                materials = tuple(_material(p, x, n, rho) for p, x, n, rho in zip((pa, pb), triangles, normals, prepared.thickness, strict=True))
                material_key = tuple(tuple(v.token for v in material) for material in materials)
                caller_materials = tuple(_caller_material(materials[i], prepared.vertex_permutations[i]) for i in prepared.caller_to_canonical)
                caller_points = tuple((pa, pb)[i] for i in prepared.caller_to_canonical)
                points.append(PrismPairPoint(*caller_points, *caller_materials, (axis.key, material_key)))
            completed.append(PrismPairAxis(axis.canonical_direction, axis.direction, axis.norm_squared, axis.numerator, axis.support_a, axis.support_b, axis.key, tuple(points)))
        maximal = completed
        sign = maximal[0].numerator.sign()
        return result({1: "separated", 0: "touching", -1: "penetrating"}[sign], maximal[0].gap_bounds)
    except SignUnresolved as error:
        return result("unresolved", obligation=ProofObligation("sign", str(error), (error.value_token, error.bits)))
    except (ArithmeticError, ValueError) as error:
        return result("unresolved", obligation=ProofObligation("geometry", str(error)))


def prism_prism_manifold(x_a, x_b, *, identity, half_thickness_m, proof_limits=PrismPairProofLimits()):
    """Complete static SAT, all maximal ties, and both exact material extrema.

    Input coordinates are finite FP64 (3,3), thickness is positive FP64 (2,).
    Signed SAT gap is metres, not a Euclidean closest-distance claim. Material
    endpoints contain three dimensionless weights and one signed offset in m.
    Caller vertex reorder requires the corresponding identity row reorder.
    """
    prepared = _prepare(x_a, x_b, identity, half_thickness_m, proof_limits)
    return _manifold_exact(prepared, prepared.triangles, proof_limits)
