"""Exact sign cells for positive-distance affine material features on normalized time [0,1]."""

from dataclasses import dataclass
from fractions import Fraction as F

from ._sweep_polynomial import (
    Roots,
    _gcd,
    _primitive,
    _sign,
    add,
    cross,
    difference,
    dot,
    face_coordinates,
    mul,
    sub,
)

ZERO, ONE = F(0), F(1)


class RootLimit(Exception):
    pass


class RootBudget:
    def __init__(self, limit):
        self.limit = limit
        self.used = 0

    def tick(self):
        if self.used >= self.limit:
            raise RootLimit("root_capacity")
        self.used += 1


@dataclass(frozen=True)
class Cut:
    lo: F
    hi: F
    zeros: frozenset = frozenset()


def _point_segment(p, a, b, r2):
    u = difference(b, a)
    w = difference(p, a)
    den = dot(u, u)
    along = dot(w, u)
    return (add(mul(sub(r2, dot(w, w)), den), mul(along, along)), (along, sub(den, along)), (den,))


def _point_point(p, q, r2):
    w = difference(p, q)
    return sub(r2, dot(w, w)), (), ()


def _raw_clauses(kind, a, b, radius):
    r2 = (radius * radius,) if radius else ()
    if kind == "VT":
        p = a[0]
        for q in b:
            yield _point_point(p, q, r2)
        for i, j in ((0, 1), (1, 2), (2, 0)):
            yield _point_segment(p, b[i], b[j], r2)
        u, v = difference(b[1], b[0]), difference(b[2], b[0])
        normal = cross(u, v)
        area2 = dot(normal, normal)
        height, weights = face_coordinates(difference(b[0], p), u, v, normal, area2)
        yield sub(mul(r2, area2), mul(height, height)), weights, (area2,)
    else:
        for p in a:
            for q in b:
                yield _point_point(p, q, r2)
            yield _point_segment(p, *b, r2)
        for q in b:
            yield _point_segment(q, *a, r2)
        u, v, w = difference(a[1], a[0]), difference(b[1], b[0]), difference(a[0], b[0])
        normal = cross(u, v)
        det = dot(normal, normal)
        uu, vv, uv = dot(u, u), dot(v, v), dot(u, v)
        uw, vw = dot(u, w), dot(v, w)
        sn = sub(mul(uv, vw), mul(vv, uw))
        tn = sub(mul(uu, vw), mul(uv, uw))
        height = dot(w, normal)
        # At an instant of parallelism the denominator vanishes. At that
        # instant the four finite endpoint-edge cases cover the full minimum.
        yield sub(mul(r2, det), mul(height, height)), (sn, sub(det, sn), tn, sub(det, tn)), (det,)


def _polynomials(kind, a, b, radius):
    clauses = []
    for clearance, limits, strict in _raw_clauses(kind, a, b, radius):
        c = _primitive(clearance)
        limits = tuple(_primitive(p) for p in limits)
        strict = tuple(_primitive(p) for p in strict)
        if len(c) == 1 and c[0] < 0 or any(len(p) == 1 and p[0] < 0 for p in limits):
            continue
        if any(not p or len(p) == 1 and p[0] <= 0 for p in strict):
            continue
        clauses.append(
            (c, tuple(p for p in limits if len(p) > 1), tuple(p for p in strict if len(p) > 1))
        )
    return tuple(clauses)


def _relation(clauses, sign):
    boundary = False
    for distance, limits, strict in clauses:
        if any(sign(q) < 0 for q in limits) or any(sign(q) <= 0 for q in strict):
            continue
        c = sign(distance)
        if c > 0:
            return "inside"
        if c == 0:
            boundary = True
    return "boundary" if boundary else "outside"


def _cuts(clauses, bits, budget):
    polynomials = {p for f, lim, pos in clauses for p in (f, *lim, *pos) if len(p) > 1}
    roots = {}
    for p in sorted(polynomials):
        key = p if p[-1] > 0 else tuple(-x for x in p)
        if key not in roots:
            budget.tick()
            roots[key] = Roots(key)
    cut = []
    gcds = {}

    def same_group(active, lo, hi):
        first = active[0][0]
        for p, count in active[1:]:
            if count != 1:
                return False
            if (first, p) not in gcds:
                gcds[first, p] = Roots(_gcd(first, p))
            if not gcds[first, p].count(lo, hi):
                return False
        return True

    def visit(lo, hi, active, depth):
        if not active:
            return
        budget.tick()
        single = all(n == 1 for _, n in active)
        zeros = frozenset(k for k, _ in active)
        if single and all(_sign(p, hi) == 0 for p, _ in active):
            cut.append(Cut(hi, hi, zeros))
            return
        if depth >= bits and single and same_group(active, lo, hi):
            cut.append(Cut(lo, hi, zeros))
            return
        if depth >= 51:
            raise RootLimit("root_precision")
        mid = (lo + hi) / 2
        first = []
        last = []
        for key, n in active:
            left = roots[key].count(lo, mid)
            if left:
                first.append((key, left))
            if n > left:
                last.append((key, n - left))
        visit(lo, mid, first, depth + 1)
        visit(mid, hi, last, depth + 1)

    active = [(p, n) for p, r in roots.items() if (n := r.count(ZERO, ONE))]
    visit(ZERO, ONE, active, 0)
    cut.insert(0, Cut(ZERO, ZERO))
    if cut[-1].lo != ONE:
        cut.append(Cut(ONE, ONE))

    def refine(c):
        if c.lo == c.hi:
            return c
        if c.hi - c.lo <= F(1, 1 << 51):
            raise RootLimit("root_precision")
        budget.tick()
        mid = (c.lo + c.hi) / 2
        p = next(iter(c.zeros))
        if _sign(p, mid) == 0:
            return Cut(mid, mid, c.zeros)
        return Cut(c.lo, mid, c.zeros) if roots[p].count(c.lo, mid) else Cut(mid, c.hi, c.zeros)

    # A rational point strictly between each two algebraic event enclosures
    # certifies its entire open sign cell, including very short entry windows.
    for i in range(len(cut) - 1):
        while cut[i].hi >= cut[i + 1].lo:
            before = (cut[i], cut[i + 1])
            cut[i], cut[i + 1] = refine(cut[i]), refine(cut[i + 1])
            if before == (cut[i], cut[i + 1]):
                raise RootLimit("root_precision")
    return cut


def locate_threshold(kind, a, b, radius, bits, budget):
    """First closed membership and infimum of strict membership with a real interior witness."""
    clauses = _polynomials(kind, a, b, radius)

    def at(t):
        return _relation(clauses, lambda p: _sign(p, t))

    start, end = at(ZERO), at(ONE)
    initial = Cut(ZERO, ZERO)
    if start == "inside":
        return start, end, (initial, ZERO), (initial, ZERO)
    closed = (initial, ZERO) if start == "boundary" else None
    opened = None
    if not clauses:
        return start, end, closed, opened
    cuts = _cuts(clauses, bits, budget)
    for i, c in enumerate(cuts):
        t = (c.lo + c.hi) / 2
        if c.lo == c.hi:
            relation = at(t)
        else:

            def algebraic_sign(p):
                key = p if not p or p[-1] > 0 else tuple(-x for x in p)
                return 0 if key in c.zeros else _sign(p, t)

            relation = _relation(clauses, algebraic_sign)
        if relation != "outside" and closed is None:
            closed = c, t
        if i + 1 < len(cuts):
            next_lo = cuts[i + 1].lo
            # For a nonrational root, the upper bracket endpoint is already
            # on the right. For a dyadic root choose a nearby actual instant.
            interior = c.hi if c.lo != c.hi else c.hi + min(F(1, 1 << bits), (next_lo - c.hi) / 2)
            if at(interior) == "inside":
                if relation == "outside" or closed is None:
                    raise RootLimit("feature_relation_unresolved")
                opened = c, interior
                break
    return start, end, closed, opened
