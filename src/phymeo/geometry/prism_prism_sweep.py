"""Bounded continuous certificates for affine nodes and two curved normals.

Times are exact rational fractions of the positive duration h. Normal offsets
are not replaced by endpoint chords. Unknown or incomplete coverage is unsafe.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite

import numpy as np

from phymeo.geometry._prism_pair_exact import PrismPairProofLimits, SignUnresolved, sqrt_bounds
from phymeo.geometry.prism_prism_manifold import (
    ProofObligation,
    _add,
    _array,
    _cross,
    _digest,
    _directions,
    _dot,
    _geometry,
    _manifold_exact,
    _prepare,
    _scale,
    _sub,
    _support,
)


@dataclass(frozen=True)
class PrismAreaBound:
    coefficients: tuple
    projection_direction: tuple
    projection_minimum: Fraction
    norm_lower: Fraction
    derivative_norm_squared_upper: Fraction
    second_derivative_norm_upper: Fraction
    curvature_upper: Fraction


@dataclass(frozen=True)
class PrismSweepCell:
    start: Fraction
    end: Fraction
    depth: int
    canonical_direction: tuple
    direction: tuple
    area_a: PrismAreaBound
    area_b: PrismAreaBound
    endpoint_gap_lowers: tuple
    normal_contributions: tuple
    curvature_upper: Fraction
    unnormalized_lower: Fraction
    gap_lower: Fraction


@dataclass(frozen=True)
class PrismPendingCell:
    start: Fraction
    end: Fraction
    depth: int
    reason: str


@dataclass(frozen=True)
class PrismPrismSweep:
    status: str
    fingerprint: str
    caller_fingerprint: str
    identity_token: tuple
    caller_to_canonical: tuple
    area_coefficients: tuple
    segments: tuple
    pending: tuple
    checked_cells: int
    witness_time: Fraction | None
    witness_manifold: object | None
    obligation: ProofObligation | None


@dataclass(frozen=True)
class _SweepInput:
    prepared: object
    endpoints: tuple
    areas: tuple
    duration: Fraction
    budget: Fraction
    fingerprint: str
    caller_fingerprint: str


def _scalar(value, name, positive):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a finite real scalar")
    value = float(value)
    if not isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} is outside its admissible range")
    return Fraction.from_float(value)


def _area_polynomial(start, end):
    e, f = _sub(start[1], start[0]), _sub(start[2], start[0])
    de = _sub(_sub(end[1], end[0]), e)
    df = _sub(_sub(end[2], end[0]), f)
    return _cross(e, f), _add(_cross(de, f), _cross(e, df)), _cross(de, df)


def _value(coefficients, t):
    return _add(_add(coefficients[0], _scale(coefficients[1], t)), _scale(coefficients[2], t*t))


def _polynomial_minimum(coefficients, start, end):
    c0, c1, c2 = coefficients
    times = [start, end]
    if c2:
        stationary = -c1 / (2*c2)
        if start < stationary < end:
            times.append(stationary)
    return min(c0 + c1*t + c2*t*t for t in times)


def _area_bound(coefficients, start, end, limits):
    direction = _value(coefficients, (start + end)/2)
    squared = _dot(direction, direction)
    if not squared:
        return None
    minimum = _polynomial_minimum(tuple(_dot(c, direction) for c in coefficients), start, end)
    if minimum <= 0:
        return None
    rmin = minimum / sqrt_bounds(squared, limits.initial_sqrt_bits)[1]
    da = _add(coefficients[1], _scale(coefficients[2], 2*start))
    db = _add(coefficients[1], _scale(coefficients[2], 2*end))
    derivative_squared = max(_dot(da, da), _dot(db, db))
    second = _scale(coefficients[2], 2)
    second_upper = sqrt_bounds(_dot(second, second), limits.initial_sqrt_bits)[1]
    constant_normal = all(not any(_cross(c, direction)) for c in coefficients)
    curvature = Fraction(0) if constant_normal else second_upper/rmin + 3*derivative_squared/(rmin*rmin)
    return PrismAreaBound(coefficients, direction, minimum, rmin, derivative_squared, second_upper, curvature)


def _prepare_sweep(path_a, path_b, identity, thickness, h, budget, limits):
    paths = tuple(_array(path, (2, 3, 3), np.float64, name) for path, name in ((path_a, "path_a"), (path_b, "path_b")))
    start = _prepare(paths[0][0], paths[1][0], identity, thickness, limits)
    end = _prepare(paths[0][1], paths[1][1], identity, thickness, limits)
    duration, penetration = _scalar(h, "h", True), _scalar(budget, "penetration_budget_m", False)
    endpoints = tuple((a, b) for a, b in zip(start.triangles, end.triangles, strict=True))
    areas = tuple(_area_polynomial(a, b) for a, b in endpoints)
    fingerprint = _digest(("prism-pair-affine-nodes-v1", start.identity_token, endpoints, start.thickness, duration, penetration, limits.token))
    caller_fingerprint = _digest((fingerprint, start.caller_to_canonical, start.vertex_permutations))
    return _SweepInput(start, endpoints, areas, duration, penetration, fingerprint, caller_fingerprint)


def _triangles_at(source, time):
    return tuple(tuple(_add(a, _scale(_sub(b, a), time)) for a, b in zip(start, end, strict=True)) for start, end in source.endpoints)


def _endpoint_lower(source, time, direction, limits):
    _, _, vertices = _geometry(_triangles_at(source, time), source.prepared.thickness, limits)
    low, _, _, _ = _support(vertices[0], direction)
    _, high, _, _ = _support(vertices[1], direction)
    return (low - high).signed_bounds()[1][0]


def _cell_bound(source, start, end, depth, direction, limits):
    areas = tuple(_area_bound(coefficients, start, end, limits) for coefficients in source.areas)
    if any(area is None for area in areas):
        return None
    norm_squared = _dot(direction, direction)
    if norm_squared <= 0:
        return None
    norm_low, norm_high = sqrt_bounds(norm_squared, limits.initial_sqrt_bits)
    if norm_low <= 0:
        return None
    endpoints = tuple(_endpoint_lower(source, time, direction, limits) for time in (start, end))
    contributions = tuple(rho*area.curvature_upper for rho, area in zip(source.prepared.thickness, areas, strict=True))
    curvature = norm_high * sum(contributions)
    lower = min(endpoints) - curvature*(end-start)**2/8
    if lower + source.budget*norm_low < 0:
        return None
    normalized = lower / (norm_high if lower >= 0 else norm_low)
    caller_direction = direction if source.prepared.caller_to_canonical[0] == 0 else _scale(direction, -1)
    return PrismSweepCell(start, end, depth, direction, caller_direction, *areas, endpoints, contributions, curvature, lower, normalized)


def prism_prism_sweep(path_a, path_b, *, identity, half_thickness_m, h, penetration_budget_m, proof_limits=PrismPairProofLimits()):
    """Certify gap >= -budget over a complete rational partition, or reject.

    Each path has FP64 shape (2,3,3). h is seconds; the budget is metres. A
    violation requires a complete SAT upper bound at a valid rational time.
    An unresolved result retains every remaining interval, including siblings.
    """
    source = _prepare_sweep(path_a, path_b, identity, half_thickness_m, h, penetration_budget_m, proof_limits)
    stack = [(Fraction(0), Fraction(1), 0)]
    segments, checked = [], 0

    def finish(status, remaining=(), reason="", witness=None, manifold=None):
        pending = tuple(PrismPendingCell(a, b, depth, reason) for a, b, depth in sorted(remaining))
        obligation = None if status == "certified" else ProofObligation("continuous", reason, tuple((p.start, p.end) for p in pending))
        return PrismPrismSweep(status, source.fingerprint, source.caller_fingerprint, source.prepared.identity_token, source.prepared.caller_to_canonical, source.areas, tuple(sorted(segments, key=lambda item: item.start)), pending, checked, witness, manifold, obligation)

    while stack:
        if checked >= proof_limits.max_time_cells:
            return finish("unresolved", stack, "time-cell budget exhausted")
        start, end, depth = stack.pop()
        checked += 1
        times = (start, (start+end)/2, end)
        candidates = set()
        reason = "no fixed-axis lower bound established"
        for time in times:
            if any(not any(_value(coefficients, time)) for coefficients in source.areas):
                return finish("unresolved", stack + [(start, end, depth)], "exact area degeneracy at rational time", time)
            triangles = _triangles_at(source, time)
            sample = _manifold_exact(source.prepared, triangles, proof_limits)
            if sample.sat_complete and sample.exact_gap_bounds[1] < -source.budget:
                return finish("violation", stack + [(start, end, depth)], "complete SAT upper gap is below negative budget", time, sample)
            candidates.update(_directions(triangles))
        accepted = None
        for direction in sorted(candidates):
            try:
                cell = _cell_bound(source, start, end, depth, direction, proof_limits)
            except (SignUnresolved, ArithmeticError, ValueError) as error:
                reason = f"bounded support proof unresolved: {error}"
                continue
            if cell is not None and (accepted is None or cell.gap_lower > accepted.gap_lower):
                accepted = cell
        if accepted is not None:
            segments.append(accepted)
            continue
        if depth >= proof_limits.max_time_depth:
            return finish("unresolved", stack + [(start, end, depth)], f"time-depth budget exhausted: {reason}")
        midpoint = (start+end)/2
        stack.extend(((midpoint, end, depth+1), (start, midpoint, depth+1)))
    return finish("certified")


def verify_prism_prism_sweep(certificate, path_a, path_b, *, identity, half_thickness_m, h, penetration_budget_m, proof_limits=PrismPairProofLimits()):
    """Recompute a safety receipt from the bound inputs; fail closed on forgery.

    This does not call the sweep producer. Each supplied fixed direction is
    independently checked against the actual path, both normal contributions,
    and a complete gap-free partition. Non-certified evidence returns False.
    """
    try:
        if not isinstance(certificate, PrismPrismSweep) or certificate.status != "certified":
            return False
        source = _prepare_sweep(path_a, path_b, identity, half_thickness_m, h, penetration_budget_m, proof_limits)
        if (certificate.fingerprint, certificate.caller_fingerprint, certificate.identity_token, certificate.caller_to_canonical, certificate.area_coefficients) != (source.fingerprint, source.caller_fingerprint, source.prepared.identity_token, source.prepared.caller_to_canonical, source.areas):
            return False
        if type(certificate.segments) is not tuple or type(certificate.pending) is not tuple:
            return False
        if certificate.pending or certificate.obligation is not None or certificate.witness_time is not None or certificate.witness_manifold is not None:
            return False
        if type(certificate.checked_cells) is not int or certificate.checked_cells != 2*len(certificate.segments)-1 or certificate.checked_cells > proof_limits.max_time_cells:
            return False
        cursor = Fraction(0)
        for cell in certificate.segments:
            if not isinstance(cell, PrismSweepCell) or not isinstance(cell.start, Fraction) or not isinstance(cell.end, Fraction):
                return False
            if cell.start != cursor or not cell.start < cell.end <= 1:
                return False
            if type(cell.depth) is not int or not 0 <= cell.depth <= proof_limits.max_time_depth:
                return False
            if cell.end-cell.start != Fraction(1, 1 << cell.depth) or (cell.start*(1 << cell.depth)).denominator != 1:
                return False
            if len(cell.canonical_direction) != 3 or any(not isinstance(v, Fraction) for v in cell.canonical_direction):
                return False
            # Recompute every quantity, including BOTH curvature terms. Neither
            # an alleged lower bound nor a preconstructed AreaBound is trusted.
            recomputed = _cell_bound(source, cell.start, cell.end, cell.depth, cell.canonical_direction, proof_limits)
            if recomputed is None or recomputed != cell or cell.gap_lower < -source.budget:
                return False
            cursor = cell.end
        return cursor == 1
    except (TypeError, ValueError, ArithmeticError, AttributeError, OverflowError):
        return False
