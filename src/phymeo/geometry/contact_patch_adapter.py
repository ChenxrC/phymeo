"""Instantaneous finite convex patches, independent of owners and integration.

Inputs are the actual world-space material vertices at one instant. A bare
CollisionPair is insufficient for this operation: it does not retain arbitrary
sample poses or the capsule endpoints needed to reconstruct a closest interval.
No CCD or time interpolation is implied by this API.
"""

from dataclasses import dataclass, replace
from fractions import Fraction as F
from numbers import Integral

import numpy as np

from ..constraints.collision_shapes import CollisionMotion, CollisionShape
from ._capsule_triangle_sweep import segment_minimum
from ._convex_contact import HullPairQuery
from ._convex_hull import HullShape
from ._round_sweep import _AxisPair, _unit, _vector, _weights, _xyz
from ._triangle_pair_sweep import _FacePath, _Pose
from .collision_witness import CollisionWitness

_SOURCE = "host-instantaneous-convex-patch"


def _vec(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("contact point/normal must be a finite 3-vector")
    return a


def _address(value):
    result = tuple((i, float(w)) for i, w in value)
    if (
        not result
        or any(
            isinstance(i, bool)
            or not isinstance(i, Integral)
            or i < 0
            or not np.isfinite(w)
            or w < 0
            for i, w in result
        )
        or len({i for i, _ in result}) != len(result)
        or abs(sum(w for _, w in result) - 1) > 1e-12
    ):
        raise ValueError("invalid contact material weights")
    return tuple(sorted(result))


@dataclass(frozen=True)
class ContactSite:
    witness: CollisionWitness
    load_point: tuple[float, float, float]
    weight: float

    def __post_init__(self):
        w = self.witness
        if not isinstance(w, CollisionWitness) or np.ndim(w.gap_m) != 0 or not np.isfinite(w.gap_m):
            raise ValueError("finite material witness required")
        n, a, b, p = (_vec(x) for x in (w.normal, w.first_point, w.second_point, self.load_point))
        if abs(np.linalg.norm(n) - 1) > 1e-10:
            raise ValueError("contact normal must be unit")
        gap = float(w.gap_m)
        local_scale = max(1.0, np.linalg.norm(b - a), abs(gap))
        # Coordinate subtraction only contributes rounding along the normal.
        # A far-away perpendicular coordinate must not turn separation into load.
        eps = np.finfo(float).eps
        gap_rounding = 8 * eps * (abs(n) @ (abs(a) + abs(b)) + np.linalg.norm(b - a))
        if abs((b - a) @ n - gap) > 1e-10 * local_scale + gap_rounding:
            raise ValueError("witness local gap differs from its material points")
        midpoint = a / 2 + b / 2
        point_rounding = 8 * eps * (abs(a) + abs(b) + abs(p))
        if np.any(abs(p - midpoint) > 1e-10 * local_scale + point_rounding):
            raise ValueError("both owners must use the common midpoint load point")
        weight = float(self.weight)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("positive quadrature weight required")
        object.__setattr__(
            self,
            "witness",
            replace(
                w,
                gap_m=gap,
                normal=_xyz(n),
                first_point=_xyz(a),
                second_point=_xyz(b),
                first_weights=_address(w.first_weights),
                second_weights=_address(w.second_weights),
                first_offset=None if w.first_offset is None else _xyz(_vec(w.first_offset)),
                second_offset=None if w.second_offset is None else _xyz(_vec(w.second_offset)),
            ),
        )
        object.__setattr__(self, "load_point", _xyz(p))
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True)
class ContactPatchSample:
    sites: tuple[ContactSite, ...]
    dimension: int
    source: str

    def __post_init__(self):
        sites = tuple(self.sites)
        if (
            self.dimension not in (0, 1, 2)
            or not sites
            or any(not isinstance(s, ContactSite) for s in sites)
            or len(sites) < self.dimension + 1
            or abs(sum(s.weight for s in sites) - 1) > 1e-12
            or not isinstance(self.source, str)
            or not self.source
        ):
            raise ValueError("invalid normalized contact patch")
        first = np.asarray(sites[0].witness.normal)
        if any(np.linalg.norm(np.asarray(s.witness.normal) - first) > 1e-10 for s in sites):
            raise ValueError("multiple contact normals require a separate explicit patch")
        object.__setattr__(self, "sites", sites)


def _sites(witnesses, dimension):
    if dimension < 2:
        weights = np.full(len(witnesses), 1 / len(witnesses))
    else:
        points = np.array([w.first_point for w in witnesses])
        center = points.mean(axis=0)
        areas = np.array(
            [
                np.linalg.norm(np.cross(a - center, b - center))
                for a, b in zip(points, np.roll(points, -1, axis=0), strict=True)
            ]
        )
        if not np.isfinite(areas).all() or areas.sum() <= 0:
            raise ValueError("degenerate contact polygon")
        # A triangle fan around the vertex mean integrates every affine field.
        # Distribute each fan centroid's contribution back to the original vertices.
        weights = (areas + np.roll(areas, 1)) / (3 * areas.sum()) + 1 / (3 * len(points))
    sites = tuple(
        ContactSite(w, _xyz((np.asarray(w.first_point) + w.second_point) / 2), float(q))
        for w, q in zip(witnesses, weights, strict=True)
    )
    return ContactPatchSample(sites, dimension, _SOURCE)


def _round_patch(a, b, ma, mb):
    witness, _ = _AxisPair(a, b, ma, mb).sample(F(0))
    if witness.normal is None:
        raise ValueError("coincident axes have an ambiguous contact normal")
    n = np.asarray(witness.normal)
    span = witness.axis_parameter_span
    if span is None:
        return _sites(
            (replace(witness, first_offset=_xyz(a.radius * n), second_offset=_xyz(-b.radius * n)),),
            0,
        )
    parameters = (span[0],) if span[0] == span[1] else span
    witnesses = []
    for s, t in parameters:
        x = (1 - s) * ma.start[0] + s * ma.start[-1]
        y = (1 - t) * mb.start[0] + t * mb.start[-1]
        witnesses.append(
            replace(
                witness,
                first_point=_xyz(x + a.radius * n),
                second_point=_xyz(y - b.radius * n),
                first_weights=_weights(a, F(s)),
                second_weights=_weights(b, F(t)),
                first_offset=_xyz(a.radius * n),
                second_offset=_xyz(-b.radius * n),
            )
        )
    return _sites(tuple(witnesses), len(witnesses) - 1)


def _hull_patch(a, b, ma, mb):
    if a.radius or b.radius:
        raise ValueError("rounded hull pressure patches are unsupported")
    pose = HullPairQuery(HullShape(a, ma), HullShape(b, mb), proximity=True).sample(F(0))
    patches = pose.patches
    if not patches:
        patch = getattr(pose.witness, "proximity_patch", None)
        if patch is None:
            if pose.witness.normal is None:
                raise ValueError("ambiguous hull separation normal")
            return _sites((pose.witness,), 0)
    elif len(patches) != 1:
        raise ValueError("ambiguous multiple hull contact normals")
    else:
        patch = patches[0]
    witnesses = []
    normal = np.asarray(patch.normal)
    for point in patch.points:
        gap = float((np.asarray(point.second_point) - point.first_point) @ normal)
        witnesses.append(
            CollisionWitness(
                gap,
                patch.normal,
                point.first_point,
                point.second_point,
                point.first_weights,
                point.second_weights,
                _SOURCE,
            )
        )
    return _sites(tuple(witnesses), patch.dimension)


def _axis_intersects(hull, start, axis):
    low, high = F(0), F(1)
    for face in hull.faces:
        height = face.normal @ start - face.plane
        rate = face.normal @ axis
        if rate > 0:
            high = min(high, -height / rate)
        elif rate < 0:
            low = max(low, -height / rate)
        elif height > 0:
            return False
        if low > high:
            return False
    return True


def _hull_round_patch(a, b, ma, mb):
    round_first = a.kind != "hull"
    rounded, hs, mr, mh = (a, b, ma, mb) if round_first else (b, a, mb, ma)
    if hs.radius:
        raise ValueError("rounded hull pressure patches are unsupported")
    hull = HullShape(hs, mh)
    start, axis = _vector(mr.start[0]), _vector(mr.start[-1]) - _vector(mr.start[0])
    if _axis_intersects(hull, start, axis):
        raise ValueError("round axis inside hull has an unsupported deep contact normal")
    best, extrema = None, []
    for face in hull.faces:
        for j in range(1, len(face.ids) - 1):
            ids = (face.ids[0], face.ids[j], face.ids[j + 1])
            tri_shape = CollisionShape(hs.key, "triangle", ids)
            tri_path = CollisionMotion([_xyz(hull.points[i]) for i in ids])
            tri = _Pose(_FacePath(tri_shape, tri_path), F(0))
            squared, low, high = segment_minimum(start, axis, tri)
            if best is None or squared < best:
                best, extrema = squared, []
            if squared == best:
                for parameter, weights in (low, high):
                    material = tuple(sorted(zip(ids, map(float, weights), strict=True)))
                    face_point = np.asarray(weights, dtype=object) @ tri.p
                    extrema.append((parameter, material, face_point))
    if not extrema:
        raise ValueError("finite hull-round closest set unresolved")
    extrema.sort(key=lambda x: (x[0], x[1]))
    first, last = extrema[0], extrema[-1]
    chosen = (first,) if not any(axis) or first[0] == last[0] else (first, last)
    witnesses = []
    for parameter, weights, face_point in chosen:
        center = start + parameter * axis
        direction, distance = _unit(face_point - center)
        if direction is None:
            raise ValueError("ambiguous hull-round normal")
        surface = np.asarray(center, float) + rounded.radius * direction
        gap = float((best - F(rounded.radius) ** 2) / F(distance + rounded.radius))
        round_weights = _weights(rounded, parameter)
        if round_first:
            sides = (_xyz(surface), _xyz(face_point), round_weights, weights)
            normal, offsets = direction, (_xyz(rounded.radius * direction), None)
        else:
            sides = (_xyz(face_point), _xyz(surface), weights, round_weights)
            normal, offsets = -direction, (None, _xyz(rounded.radius * direction))
        witnesses.append(
            CollisionWitness(
                gap,
                _xyz(normal),
                *sides,
                _SOURCE,
                first_offset=offsets[0],
                second_offset=offsets[1],
            )
        )
    return _sites(tuple(witnesses), len(witnesses) - 1)


def contact_patch_sample(first, second, first_points, second_points):
    """Sample a single current pose; actual vertices follow original material IDs.

    Supports spheres, finite capsules and unrounded full-dimensional hulls.
    A true OBB is a hull specialization. Near-surface contacts require a unique
    normal; coincident axes, deep round-axis containment and multi-normal hull
    ties are explicit failures, not a substitute AABB or chosen arbitrary force.
    """
    for shape in (first, second):
        if not isinstance(shape, CollisionShape) or shape.kind not in ("sphere", "capsule", "hull"):
            raise ValueError("expected a supported convex registration")
    if first.key.world != second.key.world:
        raise ValueError("contact owners must share a world")
    if first.key.body == second.key.body:
        raise ValueError("distinct rigid owners required")
    ma, mb = CollisionMotion(first_points), CollisionMotion(second_points)
    if any(len(p.start) != len(s.vertex_ids) for s, p in ((first, ma), (second, mb))):
        raise ValueError("pose must cover every registered material vertex")
    if first.kind == second.kind == "hull":
        return _hull_patch(first, second, ma, mb)
    if "hull" not in (first.kind, second.kind):
        return _round_patch(first, second, ma, mb)
    return _hull_round_patch(first, second, ma, mb)
