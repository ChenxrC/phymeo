"""Exact extrema, immutable material ownership and fixed-support continuation."""

from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction as F

import numpy as np

from ..constraints.cloth_self_response import owned
from ..core.capabilities import SolverFailure
from ._prism_pair_exact import PairNumber, PairField, square_root
from .prism_prism_manifold import PrismPairIdentity

DENOMINATOR = 1 << 52
SCHEMA = "self-material-owner-v1"


def q(value):
    return value if isinstance(value, F) else F.from_float(float(value))


def dot(a, b):
    return sum(x*y for x, y in zip(a, b, strict=True))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b, strict=True))


def proof_data(value):
    """Explicit rational/root tokens accepted by the runtime fingerprint codec."""
    if isinstance(value, F):
        return (value.numerator, value.denominator)
    if isinstance(value, PairNumber):
        return value.token
    if isinstance(value, PrismPairIdentity):
        return (value.domain, value.face_ids, value.node_ids)
    if is_dataclass(value):
        return {entry.name: proof_data(getattr(value, entry.name)) for entry in fields(value)}
    if isinstance(value, dict):
        return {key: proof_data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(proof_data(item) for item in value)
    return value


def _fraction(value):
    if any(value.coefficients[1:]):
        raise SolverFailure("expected rational source coordinate")
    return value.coefficients[0]


def _quantize(values):
    floors = []
    for value in values:
        if value.sign() < 0 or (value-1).sign() > 0:
            raise SolverFailure("ideal material address outside simplex")
        lo, hi = 0, DENOMINATOR
        while lo < hi:
            mid = (lo+hi+1)//2
            if (value-F(mid, DENOMINATOR)).sign() >= 0:
                lo = mid
            else:
                hi = mid-1
        floors.append(lo)
    remaining = DENOMINATOR-sum(floors)
    if remaining not in (0, 1, 2):
        raise SolverFailure("ideal barycentric sum is not one")
    ranks = sorted(range(3), key=lambda i: (values[i]-F(floors[i], DENOMINATOR), -i), reverse=True)
    for i in ranks[:remaining]:
        floors[i] += 1
    return tuple(floors)


def _rounded(value, error):
    bits = value.field.limits.initial_sqrt_bits
    while True:
        lo, hi = value.bounds(bits)
        result = float((lo+hi)/2)
        if max(abs(q(result)-lo), abs(q(result)-hi)) <= error:
            return result
        if bits == value.field.limits.max_sqrt_bits:
            raise SolverFailure("material representation enclosure exhausted")
        bits = min(bits*2, value.field.limits.max_sqrt_bits)


def ideal_attachment(vertices, counts, offset):
    """Reconstruct an actual address in the pair field, from original prism vertices."""
    x = [tuple(_fraction((vertices[2*i][j]+vertices[2*i+1][j])/2) for j in range(3)) for i in range(3)]
    area = cross(sub(x[1], x[0]), sub(x[2], x[0]))
    field = vertices[0][0].field
    norm = field.sqrt(dot(area, area))
    return tuple(sum(F(counts[i], DENOMINATOR)*x[i][j] for i in range(3)) + q(offset)*area[j]/norm for j in range(3))


@dataclass(frozen=True)
class SelfContactAnchors:
    identities: tuple
    manifolds: tuple
    owner_tokens: tuple
    triangles: np.ndarray
    counts: np.ndarray
    weights: np.ndarray
    offsets: np.ndarray
    pairs: np.ndarray
    pair_index: np.ndarray
    normals: np.ndarray
    directions: tuple
    active: np.ndarray
    generations: np.ndarray
    ideal_points: tuple

    def __post_init__(self):
        k = len(self.owner_tokens)
        if len(set(self.owner_tokens)) != k:
            raise ValueError("duplicate material owner")
        for name, shape, dtype in (
            ("triangles", (2*k, 3), np.int64), ("counts", (2*k, 3), np.int64),
            ("weights", (2*k, 3), np.float64), ("offsets", (2*k,), np.float64),
            ("pairs", (k, 2), np.int64), ("pair_index", (k,), np.int64),
            ("normals", (k, 3), np.float64), ("active", (k,), bool), ("generations", (k,), np.int64),
        ):
            object.__setattr__(self, name, owned(getattr(self, name), shape, dtype, name))
        if np.any(self.counts < 0) or np.any(np.sum(self.counts, axis=1) != DENOMINATOR):
            raise ValueError("invalid dyadic material address")


def make_self_contact_anchors(manifolds, *, identities, history, limits, budgets):
    manifolds, identities = tuple(manifolds), tuple(identities)
    if len(manifolds) != len(identities):
        raise ValueError("one explicit identity per manifold required")
    if len(manifolds) > limits.max_contacts:
        raise SolverFailure("pair budget exhausted")
    rows, counts, offsets, owners, pair_index, normals, directions, points, generations = ([] for _ in range(9))
    seen_pairs = set()
    for pair, (manifold, identity) in enumerate(zip(manifolds, identities, strict=True)):
        canonical = tuple(sorted(map(int, identity.face_ids)))
        if (identity.domain, canonical) in seen_pairs:
            raise ValueError("duplicate face-pair capacity")
        seen_pairs.add((identity.domain, canonical))
        oriented = []
        for side in np.argsort(identity.face_ids):
            ids = tuple(map(int, identity.node_ids[side]))
            start = min(range(3), key=lambda i: ids[i:]+ids[:i])
            oriented.append(ids[start:]+ids[:start])
        if manifold.identity_token != (identity.domain, canonical, tuple(oriented)):
            raise ValueError("manifold oriented identity does not match caller")
        if manifold.exact_gap_bounds is None or not manifold.sat_complete:
            raise SolverFailure("unknown static separation")
        if manifold.exact_gap_bounds[0] > 0:
            # A positive retained SAT witness may be unloaded even if clipping failed.
            continue
        if manifold.status == "unresolved" or not manifold.sat_complete or not manifold.maximal_axes:
            raise SolverFailure("loaded manifold clipping/normal is unresolved")
        if manifold.exact_gap_bounds[1] < -q(budgets.penetration_m):
            raise SolverFailure("initial prism penetration exceeds budget")
        first = manifold.maximal_axes[0]
        d = first.direction
        if any(any(cross(d, axis.direction)) or dot(d, axis.direction) <= 0 for axis in manifold.maximal_axes):
            raise SolverFailure("loaded maximal SAT ties do not expose a unique oriented ray")
        complete = {}
        for axis in manifold.maximal_axes:
            if not axis.points:
                raise SolverFailure("missing complete clipping extrema")
            for point in axis.points:
                key = tuple(tuple(v.token for v in mat) for mat in (point.material_a, point.material_b))
                complete[key] = point
        if len(complete) > limits.max_anchors_per_pair:
            raise SolverFailure("anchor budget exhausted")
        for point in complete.values():
            materials = (point.material_a, point.material_b)
            cs = tuple(_quantize(mat[:3]) for mat in materials)
            ss = tuple(_rounded(mat[3], q(budgets.anchor_error_m)/128) for mat in materials)
            actual = tuple(ideal_attachment(manifold.vertices[side], cs[side], ss[side]) for side in range(2))
            for side in range(2):
                extrusion = tuple((manifold.vertices[side][1][j]-manifold.vertices[side][0][j])/2 for j in range(3))
                rho = square_root(_fraction(dot(extrusion, extrusion)))
                if rho is None or abs(q(ss[side])) > rho:
                    raise SolverFailure("represented signed offset left the finite prism")
                support = first.support_a if side == 0 else first.support_b
                reference = dot(manifold.vertices[side][support[0]], d)
                if dot(actual[side], d) != reference:
                    raise SolverFailure("represented address left the exact selected support")
                ideal = point.point_a if side == 0 else point.point_b
                for a, b in zip(actual[side], ideal, strict=True):
                    lo, hi = (a-b).bounds(limits.geometry_limits.max_sqrt_bits)
                    if max(abs(lo), abs(hi)) > q(budgets.anchor_error_m)/3:
                        raise SolverFailure("quantized material extreme error exceeds ceiling")
            endpoints = []
            for side in range(2):
                ids = tuple(map(int, identity.node_ids[side]))
                start = min(range(3), key=lambda i: ids[i:]+ids[:i])
                endpoints.append((int(identity.face_ids[side]), ids[start:]+ids[:start], cs[side][start:]+cs[side][:start], int(np.array(ss[side]).view(np.int64))))
            material_token = (SCHEMA, identity.domain, tuple(sorted(endpoints)))
            generation = int(history.get(material_token, history.get("next_generation", 0)))
            if generation < 0 or generation >= np.iinfo(np.int64).max:
                raise ValueError("invalid material birth generation")
            token = material_token + (generation,)
            if token in owners:
                raise SolverFailure("quantization merged distinct ideal extrema")
            owners.append(token)
            rows.extend(identity.node_ids)
            counts.extend(cs)
            offsets.extend(ss)
            pair_index.append(pair)
            nd = np.array(d, dtype=np.float64)
            normals.append(nd/np.linalg.norm(nd))
            directions.append(d)
            points.append((point.point_a, point.point_b))
            generations.append(generation)
    k = len(owners)
    cs = np.asarray(counts, dtype=np.int64).reshape(2*k, 3)
    return SelfContactAnchors(identities, manifolds, tuple(owners), np.asarray(rows, dtype=np.int64).reshape(2*k, 3), cs,
                             cs.astype(np.float64)/DENOMINATOR, np.asarray(offsets, dtype=np.float64),
                             np.arange(2*k, dtype=np.int64).reshape(k, 2), np.asarray(pair_index, dtype=np.int64),
                             np.asarray(normals, dtype=np.float64).reshape(k, 3), tuple(directions),
                             np.ones(k, dtype=bool), np.asarray(generations, dtype=np.int64), tuple(points))


def area_path(start, end):
    a, b = [tuple(tuple(map(q, row)) for row in x) for x in (start, end)]
    e, f = sub(a[1], a[0]), sub(a[2], a[0])
    de, df = sub(sub(b[1], b[0]), e), sub(sub(b[2], b[0]), f)
    return (cross(e, f), tuple(x+y for x, y in zip(cross(e, df), cross(de, f), strict=True)), cross(de, df))


def require_area_path(start, end, *, constant_normal=False):
    coefficients = area_path(start, end)
    direction = coefficients[0]
    if not dot(direction, direction):
        raise SolverFailure("zero initial face area")
    polynomial = tuple(dot(c, direction) for c in coefficients)
    values = [polynomial[0], sum(polynomial)]
    if polynomial[2] > 0:
        t = -polynomial[1]/(2*polynomial[2])
        if 0 < t < 1:
            values.append(polynomial[0]+polynomial[1]*t+polynomial[2]*t*t)
    if min(values) <= 0:
        raise SolverFailure("face path has no positive oriented-area proof")
    if constant_normal and any(any(cross(c, direction)) for c in coefficients):
        raise SolverFailure("charged held continuation has rotating face normal")
    return proof_data(coefficients)


def require_fixed_support(anchors, start_positions, end_positions, loaded):
    """Affine support equalities plus positive constant normals prove whole path."""
    receipts = []
    for row, charge in enumerate(loaded):
        if not charge:
            continue
        d = anchors.directions[row]
        manifold = anchors.manifolds[int(anchors.pair_index[row])]
        for side, attachment in enumerate(anchors.pairs[row]):
            ids = anchors.triangles[attachment]
            start, end = start_positions[ids], end_positions[ids]
            receipts.append(require_area_path(start, end, constant_normal=True))
            x = tuple(tuple(map(q, v)) for v in start)
            area = cross(sub(x[1], x[0]), sub(x[2], x[0]))
            field = PairField(dot(area, area), dot(area, area))
            normal = tuple(field.scalar(v)/field.sqrt(dot(area, area)) for v in area)
            support = manifold.maximal_axes[0].support_a if side == 0 else manifold.maximal_axes[0].support_b
            material = anchors.counts[attachment]
            offset = q(anchors.offsets[attachment])
            # Initial support class binds top/bottom and all supporting material vertices.
            for endpoint in (start, end):
                values = [dot(tuple(map(q, point)), d) for point in endpoint]
                chosen = min(values) if side == 0 else max(values)
                for index, count in enumerate(material):
                    if count and values[index] != chosen:
                        raise SolverFailure("held material endpoint left its finite support")
                if {i//2 for i in support} != {i for i, v in enumerate(values) if v == chosen}:
                    raise SolverFailure("charged support feature changed along path")
            if dot(normal, d).sign() and any((i % 2 == 0) != (offset < 0) for i in support):
                raise SolverFailure("held thickness side changed")
    return tuple(receipts)
