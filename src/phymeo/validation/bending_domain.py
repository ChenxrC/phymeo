"""Exact Bernstein certificates for affine numerical and physical hinge paths.

These certificates prove domain support along the named affine path. They do
not assert collision clearance or a continuous solution of the dynamics ODE.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import comb

import numpy as np

from ..constraints.cloth_material import CompositeMaterialAuditSettings
from ..core.capabilities import SolverFailure
from ..core.scene import fingerprint
from ._material_path_math import q
from .bending_bounds import Interval, atan2_bounds, cross, dot, pi_bounds, sqrt_interval, subtract


class Polynomial:
    def __init__(self, coefficients):
        values = list(map(q, coefficients))
        while len(values) > 1 and values[-1] == 0:
            values.pop()
        self.values = tuple(values)

    def __add__(self, other):
        other = polynomial(other)
        a, b = self.values, other.values
        return Polynomial([(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
                           for i in range(max(len(a), len(b)))])

    __radd__ = __add__

    def __neg__(self):
        return Polynomial([-v for v in self.values])

    def __sub__(self, other):
        return self + -polynomial(other)

    def __mul__(self, other):
        other = polynomial(other)
        result = [Fraction(0)] * (len(self.values) + len(other.values) - 1)
        for i, a in enumerate(self.values):
            for j, b in enumerate(other.values):
                result[i + j] += a * b
        return Polynomial(result)

    __rmul__ = __mul__

    def bounds(self, left, right):
        """Convex hull of exact Bernstein coefficients after affine restriction."""
        degree = len(self.values) - 1
        width = right - left
        power = [sum((self.values[j] * comb(j, k) * left**(j-k) * width**k
                      for j in range(k, degree + 1)), Fraction(0))
                 for k in range(degree + 1)]
        bernstein = [sum((power[k] * Fraction(comb(i, k), comb(degree, k))
                          for k in range(i + 1)), Fraction(0))
                     for i in range(degree + 1)]
        return Interval(min(bernstein), max(bernstein))


def polynomial(value):
    return value if isinstance(value, Polynomial) else Polynomial((value,))


@dataclass(frozen=True)
class BendingDomainCertificate:
    status: str
    path_fingerprint: str
    policy_fingerprint: str
    partition: tuple
    unresolved: tuple
    processed_intervals: int
    reason: str

    def require(self):
        if self.status != "certified":
            raise SolverFailure(f"bending continuous domain {self.status}: {self.reason}")
        return self


def _obligations(start, end, material):
    result = []
    if material.bending is None:
        return result
    for ids, rest in zip(material.bending.indices, material.bending.rest_angle, strict=True):
        x = tuple(tuple(Polynomial((q(a), q(b)-q(a))) for a, b in zip(u, v, strict=True))
                  for u, v in zip(start[ids], end[ids], strict=True))
        e, u, v = (subtract(x[k], x[0]) for k in (1, 2, 3))
        right = subtract(x[3], x[1])
        nl, nr = cross(e, u), cross(v, e)
        e2 = dot(e, e)
        a = dot(e, u) * dot(e, v) - e2 * dot(u, v)
        determinant = dot(e, cross(u, v))
        faces = []
        for normal, edges in ((nl, (*e, *u)), (nr, (*e, *right))):
            area2 = dot(normal, normal)
            # max(component)^4 is certified by all six separate inequalities.
            for component in edges:
                faces.append(area2 - q(1e-10)**2 * component * component * component * component)
        result.append((tuple(faces), e2, a, determinant, q(rest)))
    return result


def certify_bending_segment(start, end, material, *, policy=None):
    policy = CompositeMaterialAuditSettings() if policy is None else policy
    start, end = np.asarray(start), np.asarray(end)
    if (start.dtype != np.float64 or end.dtype != np.float64 or start.shape != end.shape
            or start.ndim != 2 or start.shape[1] != 3
            or not np.isfinite(start).all() or not np.isfinite(end).all()):
        raise ValueError("domain paths require two finite FP64 (n,3) endpoints")
    identity = fingerprint((start, end, material))
    policy_id = fingerprint(policy)
    obligations = _obligations(start, end, material)
    pi = pi_bounds(policy.sqrt_bits, policy.angle_terms)
    pending = [(Fraction(0), Fraction(1), 0)]
    accepted, unresolved, processed = [], [], 0
    reason = "complete strict affine domain partition"
    status = "certified"
    while pending:
        left, right, depth = pending.pop()
        if processed >= policy.max_domain_intervals:
            unresolved.extend((str(a), str(b)) for a, b, _ in [(left, right, depth), *pending])
            status, reason = "unknown", "domain interval budget exhausted"
            break
        processed += 1
        safe = True
        for faces, edge2, cosine, determinant, rest in obligations:
            bounds = [p.bounds(left, right) for p in faces]
            if any(b.high <= 0 for b in bounds):
                status, reason = "rejected", "proved unsupported face on a path cell"
                unresolved.append((str(left), str(right)))
                break
            if any(b.low <= 0 for b in bounds):
                safe = False
                break
            try:
                sine = -sqrt_interval(edge2.bounds(left, right), policy.sqrt_bits) * determinant.bounds(left, right)
                theta = atan2_bounds(sine, cosine.bounds(left, right), policy)
                if not (-pi.low < theta.low <= theta.high < pi.low
                        and (theta - rest).magnitude < pi.low):
                    safe = False
                    break
            except SolverFailure:
                safe = False
                break
        if status == "rejected":
            unresolved.extend((str(a), str(b)) for a, b, _ in pending)
            break
        if safe:
            accepted.append((str(left), str(right)))
        elif depth >= policy.max_domain_depth:
            status, reason = "unknown", "domain depth exhausted or strict boundary unresolved"
            unresolved.append((str(left), str(right)))
            unresolved.extend((str(a), str(b)) for a, b, _ in pending)
            break
        else:
            middle = (left + right) / 2
            pending.extend(((middle, right, depth + 1), (left, middle, depth + 1)))
    return BendingDomainCertificate(status, identity, policy_id, tuple(accepted), tuple(unresolved),
                                    processed, reason)

