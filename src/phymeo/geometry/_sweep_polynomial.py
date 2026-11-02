"""Closed low-degree polynomial conditions for continuously moving features.

Coefficients are in ascending powers of normalized time. Each sign test uses
the stored geometry, independently of the FP32 device proposal.
"""

from fractions import Fraction as F
from math import ceil, floor, gcd, lcm

from ..core.capabilities import SolverFailure
from ._round_sweep import _GRID, _Feature


def _trim(p):
    end = len(p)
    while end and p[end - 1] == 0:
        end -= 1
    return tuple(p[:end])


def add(p, q):
    return _trim(
        [
            (p[i] if i < len(p) else 0) + (q[i] if i < len(q) else 0)
            for i in range(max(len(p), len(q)))
        ]
    )


def sub(p, q):
    return add(p, tuple(-v for v in q))


def mul(p, q):
    if not p or not q:
        return ()
    out = [0] * (len(p) + len(q) - 1)
    for i, x in enumerate(p):
        for j, y in enumerate(q):
            out[i + j] += x * y
    return _trim(out)


def value(p, x):
    out = 0
    for coefficient in reversed(p):
        out = out * x + coefficient
    return out


def _primitive(p):
    p = _trim(p)
    if not p:
        return ()
    den = lcm(*(v.denominator for v in p))
    z = [int(v * den) for v in p]
    common = gcd(*z)
    return tuple(v // common for v in z)


def _division(p, q):
    # Exact Euclidean division, with a positive rescaling only in the caller.
    r = list(p)
    quotient = [0] * max(0, len(p) - len(q) + 1)
    while r and len(r) >= len(q):
        shift = len(r) - len(q)
        c = F(r[-1], q[-1])
        quotient[shift] = c
        for i, qi in enumerate(q):
            r[shift + i] -= c * qi
        while r and not r[-1]:
            r.pop()
    return _trim(quotient), tuple(r)


def _gcd(p, q):
    while q:
        p, q = q, _primitive(_division(p, q)[1])
    return p


def _sign(p, t):
    # Homogeneous integer Horner evaluation avoids intermediate rational divisions.
    if not p:
        return 0
    num, den = t.numerator, t.denominator
    x, scale = p[-1], 1
    for c in reversed(p[:-1]):
        scale *= den
        x = x * num + c * scale
    return (x > 0) - (x < 0)


class Roots:
    def __init__(self, p):
        self.p = _primitive(p)
        self.chain = ()
        if len(self.p) < 2:
            return
        derivative = tuple(i * c for i, c in enumerate(self.p) if i)
        squarefree = _primitive(_division(self.p, _gcd(self.p, derivative))[0])
        chain = [squarefree, _primitive(tuple(i * c for i, c in enumerate(squarefree) if i))]
        while len(chain[-1]) > 1:
            rem = _division(chain[-2], chain[-1])[1]
            chain.append(_primitive(tuple(-c for c in rem)))
        self.chain = tuple(chain)

    def _variations(self, t):
        signs = [s for p in self.chain if (s := _sign(p, t))]
        return sum(a != b for a, b in zip(signs, signs[1:]))

    def count(self, lo, hi):
        """Distinct roots in (lo, hi], including a root at the right endpoint."""
        return self._variations(lo) - self._variations(hi) if self.chain else 0

    def cell(self, lo, hi):
        if _sign(self.p, hi) == 0:
            return floor(hi * _GRID), ceil(hi * _GRID)
        left, right = floor(lo * _GRID), ceil(hi * _GRID)
        while right - left > 1:
            mid = (left + right) // 2
            t = F(mid, _GRID)
            if t >= hi or (t > lo and self.count(lo, t)):
                right = mid
            else:
                left = mid
        t = F(right, _GRID)
        return (right, right) if lo < t <= hi and _sign(self.p, t) == 0 else (left, right)


def _first_feasible(source, conditions, end):
    common = {}

    def search(lo, hi, n, depth):
        if not n:
            return None
        if depth >= 160:
            raise SolverFailure("surface_root_isolation_limit")
        uncertain = False
        for other in conditions:
            if other is source:
                continue
            if not other.count(lo, hi):
                if _sign(other.p, hi) < 0:
                    return None
            elif n == 1:
                if other not in common:
                    common[other] = Roots(_gcd(source.p, other.p))
                if not common[other].count(lo, hi):
                    uncertain = True
        if n == 1:
            if _sign(source.p, hi) == 0:
                return source.cell(lo, hi) if all(_sign(g.p, hi) >= 0 for g in conditions) else None
            if not uncertain:
                return source.cell(lo, hi)
        mid = (lo + hi) / 2
        left = source.count(lo, mid)
        return search(lo, mid, left, depth + 1) or search(mid, hi, n - left, depth + 1)

    return search(F(0), end, source.count(F(0), end), 0)


def first_entry(conditions, proposal):
    """First t in [0,1] where all polynomials are nonnegative, or None."""
    if all(not p or p[0] >= 0 for p in conditions):
        return 0, 0
    if any(len(p) == 1 and p[0] < 0 for p in conditions):
        return None

    # Translation and fixed directions reduce to a convex quadratic over an
    # interval of linear barycentric inequalities, exactly the round-pair case.
    clearance, *limits = conditions
    if (
        len(clearance) <= 3
        and (len(clearance) < 3 or clearance[2] <= 0)
        and all(len(g) <= 2 for g in limits)
    ):
        lo, hi = F(0), F(1)
        for g in limits:
            if len(g) == 2:
                root = -F(g[0], g[1])
                if g[1] > 0:
                    lo = max(lo, root)
                else:
                    hi = min(hi, root)
        c, b, a = (*clearance, 0, 0, 0)[:3]
        return _Feature(lo, hi, -a, -b, -c).entry(proposal) if lo <= hi else None

    end = F(1)
    if proposal is not None and 0 <= proposal <= 1:
        seed = F(proposal)
        if all(value(p, seed) >= 0 for p in conditions):
            end = seed
    # Starting outside a closed intersection, its first point must be a root
    # of at least one nonconstant boundary polynomial. Common or repeated roots
    # are eligible, even when no sampled time lies inside the contact set.
    roots = [Roots(p) for p in conditions if len(p) > 1]
    bounds = [bracket for s in roots if (bracket := _first_feasible(s, roots, end)) is not None]
    return (min(x[0] for x in bounds), min(x[1] for x in bounds)) if bounds else None
