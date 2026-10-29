"""Complete SAT and exact support-feature clipping for a triangular prism."""

from dataclasses import dataclass
from fractions import Fraction as F
from functools import cmp_to_key

import numpy as np

from ..constraints.robot_contact_patch import RobotContactPatch
from ._patch_interval import PatchCurve, validate_domain
from ._prism_clip import clip_support, material
from ._prism_exact import Quadratic as Q
from ._prism_exact import (
    add,
    compare_gaps,
    cross,
    dot,
    normalized_bounds,
    outward_float,
    rational,
    scale,
    sub,
    vector,
)


def validate_prismatic(model):
    if validate_domain(model) != "prismatic" or any(
        np.count_nonzero(a) != 1 or np.max(np.abs(a)) != 1 for a in model.joint_axis
    ):
        raise ValueError("thick geometry requires coordinate local prismatic axes")


def own(array):
    value = np.array(array, dtype=float, copy=True)
    value.flags.writeable = False
    return value


@dataclass(frozen=True)
class PrismPatchPoint:
    prism_point: np.ndarray
    robot_point: np.ndarray
    robot_local: np.ndarray
    material: tuple
    prism_feature: tuple
    patch_feature: tuple
    exact_prism_point: tuple
    exact_robot_point: tuple


@dataclass(frozen=True)
class PrismAxisEvidence:
    direction: tuple
    numerator: Q
    norm_squared: Q
    prism_support: tuple
    patch_support: tuple
    patch_face: bool


@dataclass(frozen=True)
class PrismPatchManifold:
    status: str
    normal: np.ndarray
    points: tuple
    axes: tuple
    gap_bounds: tuple
    exact_gap_bounds: tuple
    exact_vertices: tuple
    exact_patch_vertices: tuple
    selected_axis: PrismAxisEvidence


def exact_geometry(model, q, collider, positions, rho):
    validate_prismatic(model)
    if not isinstance(collider, RobotContactPatch) or collider.link_index >= model.dof_count:
        raise ValueError("valid RobotContactPatch required")
    x = tuple(vector(row) for row in positions)
    if len(x) != 3 or any(len(p) != 3 for p in x):
        raise ValueError("triangle positions require (3,3)")
    a, b = sub(x[1], x[0]), sub(x[2], x[0])
    c = cross(a, b)
    c2 = dot(c, c)
    if not c2 or c2.b:
        raise ValueError("degenerate triangle")
    root = Q(0, 1, c2.a)
    n = scale(c, 1 / root)
    av = tuple(add(p, scale(n, side * rho)) for side in (-1, 1) for p in x)
    # The public PatchCurve computes exact rational all-prismatic FK for local coordinate axes.
    raw = PatchCurve(model, np.array(q, float), np.array(q, float), collider).at(F(0))
    # Preserve Fraction configurations, including dyadic interior CCD witness times.
    if any(isinstance(v, F) for v in q):
        curve = PatchCurve(model, np.zeros(model.dof_count), np.zeros(model.dof_count), collider)
        raw = curve._prismatic(tuple(rational(v) for v in q))
    if any(lo != hi for p in raw for lo, hi in p):
        raise ValueError("nonrational prismatic FK")
    bv = tuple(tuple(Q(lo) for lo, hi in p) for p in raw)
    ae = (a, b, sub(x[2], x[1]), n)
    be = tuple(sub(bv[(i + 1) % len(bv)], p) for i, p in enumerate(bv))
    pn = cross(be[0], be[1])
    raw_axes = [(c, False), (pn, True)] + [(cross(e, n), False) for e in ae[:3]]
    raw_axes += [(cross(e, f), False) for e in ae for f in be]
    axes = {}
    for d, face in raw_axes:
        first = next((v for v in d if v), None)
        if first is None:
            continue
        key = scale(d, 1 / first)
        axes[key] = axes.get(key, False) or face
    evidence = []
    for key, face in axes.items():
        for d in (key, scale(key, -1)):
            ap = tuple(dot(d, p) for p in av)
            bp = tuple(dot(d, p) for p in bv)
            alo, bhi = min(ap), max(bp)
            evidence.append(
                PrismAxisEvidence(
                    d,
                    alo - bhi,
                    dot(d, d),
                    tuple(i for i, v in enumerate(ap) if v == alo),
                    tuple(i for i, v in enumerate(bp) if v == bhi),
                    face,
                )
            )
    return x, n, av, bv, tuple(evidence)


def triangle_prism_patch_manifold(
    model, q, collider, triangle_positions, *, half_thickness_m, geometric_budget_m
):
    rho, budget = rational(half_thickness_m), rational(geometric_budget_m)
    if rho <= 0 or budget < 0:
        raise ValueError("positive thickness and nonnegative budget required")
    x, n, av, bv, axes = exact_geometry(model, q, collider, triangle_positions, rho)

    def compare(a, b):
        order = compare_gaps(a.numerator, a.norm_squared, b.numerator, b.norm_squared)
        if order:
            return -order
        if a.patch_face != b.patch_face:
            return -1 if a.patch_face else 1
        return (a.direction > b.direction) - (a.direction < b.direction)

    ranked = sorted(axes, key=cmp_to_key(compare))
    best = ranked[0]
    # Only exact maximal ties can select a response axis.
    tied = [
        e
        for e in ranked
        if compare_gaps(e.numerator, e.norm_squared, best.numerator, best.norm_squared) == 0
    ]
    clipped = ()
    for selected in tied:
        clipped = clip_support(
            tuple(av[i] for i in selected.prism_support),
            tuple(bv[i] for i in selected.patch_support),
            selected.direction,
        )
        if clipped:
            best = selected
            break
    return _assemble(collider, x, n, av, bv, axes, best, clipped, budget)


def _assemble(collider, x, n, av, bv, axes, best, clipped, budget):
    bounds = normalized_bounds(best.numerator, best.norm_squared)
    status = "open" if bounds[0] > budget else "contact"
    if not clipped and status == "contact":
        raise ValueError("intersecting SAT support has no valid clipping")
    # Recover patch local coordinate by exact affine interpolation in its planar basis.
    local = tuple(vector(p) for p in collider.local_vertices)
    be1, be2 = sub(bv[1], bv[0]), sub(bv[2], bv[0])
    aa, ab, bb = dot(be1, be1), dot(be1, be2), dot(be2, be2)
    den = aa * bb - ab * ab
    points = []
    for ra, rb in clipped:
        r = sub(rb, bv[0])
        u = (dot(r, be1) * bb - dot(r, be2) * ab) / den
        v = (dot(r, be2) * aa - dot(r, be1) * ab) / den
        lp = add(
            local[0], add(scale(sub(local[1], local[0]), u), scale(sub(local[2], local[0]), v))
        )
        mat = tuple(t.a if not t.b else t for t in material(ra, x, n))
        points.append(
            PrismPatchPoint(
                own(ra), own(rb), own(lp), mat, best.prism_support, best.patch_support, ra, rb
            )
        )
    normal = own(best.direction)
    normal = own(normal / np.linalg.norm(normal))
    return PrismPatchManifold(
        status, normal, tuple(points), axes, outward_float(bounds), bounds, av, bv, best
    )


def prefer_initial_axis(
    manifold, initial_direction, collider, half_thickness_m, geometric_budget_m
):
    """Preserve a fixed initial axis only after valid exact patch-face priority."""
    if manifold.selected_axis.patch_face:
        return manifold
    initial = tuple(Q.coerce(v) for v in initial_direction)
    best = manifold.selected_axis
    for axis in sorted(manifold.axes, key=lambda a: a.direction):
        if any(cross(axis.direction, initial)) or dot(axis.direction, initial) <= 0:
            continue
        if compare_gaps(axis.numerator, axis.norm_squared, best.numerator, best.norm_squared) != 0:
            continue
        av, bv = manifold.exact_vertices, manifold.exact_patch_vertices
        clipped = clip_support(
            tuple(av[i] for i in axis.prism_support),
            tuple(bv[i] for i in axis.patch_support),
            axis.direction,
        )
        if not clipped:
            continue
        rho = rational(half_thickness_m)
        x = tuple(scale(add(av[i], av[i + 3]), F(1, 2)) for i in range(3))
        n = scale(sub(av[3], av[0]), 1 / (2 * rho))
        return _assemble(
            collider, x, n, av, bv, manifold.axes, axis, clipped, rational(geometric_budget_m)
        )
    return manifold
