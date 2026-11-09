"""Finite round link against an oriented box with unique material separation witnesses."""

from fractions import Fraction as F

import numpy as np

from .sphere_link_contact import _RigidLinkContacts


class _AxisInsideBox(ValueError):
    pass


def _closest_span(start, end, half):
    # Split the convex distance-squared polynomial where the clamp changes.
    # Compare the represented local dyadics exactly so a small tilt does not
    # turn a unique contact into a fictitious equal-load interval.
    a, b, h = ([F(float(v)) for v in row] for row in (start, end, half))
    direction = [v - u for u, v in zip(a, b, strict=True)]
    cuts = {F(0), F(1)}
    for x, d, extent in zip(a, direction, h, strict=True):
        if d:
            cuts.update(u for v in (-extent, extent) if 0 < (u := (v - x) / d) < 1)
    cuts = sorted(cuts)
    best = None
    first, last = F(0), F(0)
    for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
        mid = (lo + hi) / 2
        aa = bb = 0
        for x, d, extent in zip(a, direction, h, strict=True):
            pm = x + d * mid
            if abs(pm) > extent:
                side = extent if pm > 0 else -extent
                aa += d * d
                bb += d * (x - side)
        u = max(lo, min(hi, -bb / aa)) if aa else lo
        v = u if aa else hi
        dsq = sum(
            max(abs(x + d * u) - extent, 0) ** 2
            for x, d, extent in zip(a, direction, h, strict=True)
        )
        if best is None or dsq < best:
            best, first, last = dsq, u, v
        elif dsq == best:
            first, last = min(first, u), max(last, v)
    if not best:
        raise _AxisInsideBox(
            "capsule or sphere axis intersects the payload box; unique exterior separation unavailable"
        )
    if not any(direction):
        last = first = F(0)
    if first != last and float(first) == float(last):
        raise ValueError("distinct capsule material parameters are below output resolution")
    return (float(first),) if first == last else (float(first), float(last))


class BoxLinkContacts(_RigidLinkContacts):
    """Free OBB versus finite capsules or spheres with a unique separation normal."""

    def __init__(self, solids):
        super().__init__(solids)
        if any(s.kind not in {"capsule", "sphere"} for s in self.solids):
            raise ValueError(
                "box payload response requires round link solids (sphere or finite capsule)"
            )

    def evaluate(
        self,
        positions,
        rotations,
        centers_of_mass,
        twists,
        center,
        half_extents,
        velocity,
        omega,
        material,
        *,
        payload_rotation,
        fixed_reference_m=(0.0, 0.0, 0.0),
    ):
        half, R = np.asarray(half_extents, dtype=float), np.asarray(payload_rotation, dtype=float)
        if (
            half.shape != (3,)
            or not np.isfinite(half).all()
            or np.any(half <= 0)
            or R.shape != (3, 3)
            or not np.isfinite(R).all()
            or not np.allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-9)
            or np.linalg.det(R) <= 0
        ):
            raise ValueError("expected positive box half-extents and a proper payload orientation")
        return self._evaluate(
            positions,
            rotations,
            centers_of_mass,
            twists,
            center,
            (half, R),
            velocity,
            omega,
            material,
            fixed_reference_m=fixed_reference_m,
        )

    @staticmethod
    def _witnesses(solid, positions, rotations, center, geometry):
        half, R = geometry
        i = solid.owner_index
        world = positions[i] + solid.vertices @ rotations[i].T if i >= 0 else solid.vertices
        for w in box_capsule_witnesses(center, R, half, world, solid.radius, capsule_first=True):
            u = w["axis_parameter"]
            feature = (
                "sphere"
                if solid.kind == "sphere"
                else "end0"
                if u == 0
                else "end1"
                if u == 1
                else "shaft"
            )
            yield (
                w["gap_m"],
                w["normal"],
                w["first_surface"],
                w["second_surface"],
                feature,
                w["weight"],
                {"axis_parameter": u, "axis_point_m": w["axis_point"].tolist()},
            )


def _interior_span(start, end, half):
    """Exact nearest facet of box minus segment, including finite support clipping.

    The axis intersects the box, so zero belongs to this zonotope. Its generators
    are the three box axes and the segment: pairwise cross products give every
    facet ray. A tied escape direction remains unsupported, rather than choosing
    an arbitrary impulse. Radius adds the same distance to every facet.
    """
    a, b, h = ([F(float(v)) for v in row] for row in (start, end, half))
    d = [v - u for u, v in zip(a, b, strict=True)]
    axes = [(F(1), F(0), F(0)), (F(0), F(1), F(0)), (F(0), F(0), F(1))]
    cross = [(F(0), -d[2], d[1]), (d[2], F(0), -d[0]), (-d[1], d[0], F(0))]
    rays = set()
    for ray in axes + cross:
        if not any(ray):
            continue
        scale = abs(next(v for v in ray if v))
        for sign in (-1, 1):
            rays.add(tuple(sign * v / scale for v in ray))
    best, winners = None, []
    for w in sorted(rays):
        norm2 = sum(v * v for v in w)
        da, db = (sum(v * x for v, x in zip(w, p, strict=True)) for p in (a, b))
        height = sum(x * abs(v) for x, v in zip(h, w, strict=True)) - min(da, db)
        if height < 0:
            raise ValueError("axis/box interior separation predicate inconsistent")
        score = height * height / norm2
        if best is None or score < best:
            best, winners = score, [(w, norm2, height, da, db)]
        elif score == best:
            winners.append((w, norm2, height, da, db))
    if len(winners) != 1:
        raise ValueError("capsule axis inside box has ambiguous minimum separation normals")
    w, norm2, height, da, db = winners[0]
    lo, hi = (F(0), F(0)) if da < db else (F(1), F(1)) if db < da else (F(0), F(1))
    shift = [height * v / norm2 for v in w]
    for x, delta, extent, offset in zip(a, d, h, shift, strict=True):
        x += offset
        if delta:
            lower, upper = sorted(((-extent - x) / delta, (extent - x) / delta))
            lo, hi = max(lo, lower), min(hi, upper)
        elif abs(x) > extent:
            raise ValueError("axis/box interior facet has no finite material witness")
    if lo > hi:
        raise ValueError("axis/box interior facet has no finite material witness")
    if not any(d):
        lo = hi = F(0)
    if lo != hi and float(lo) == float(hi):
        raise ValueError("distinct capsule material parameters are below output resolution")
    us = (lo,) if lo == hi else (lo, hi)
    direction = np.array([float(v) for v in w]) / np.sqrt(float(norm2))
    distance = float(height) / np.sqrt(float(norm2))
    sites = [
        (float(u), np.array([float(x + u * v + s) for x, v, s in zip(a, d, shift, strict=True)]))
        for u in us
    ]
    return direction, distance, sites


def box_capsule_witnesses(center, R, half, world, radius, *, capsule_first):
    """Yield ordered material witnesses. Normal points from the first participant to the second.

    One or two world axis points describe a sphere or finite capsule. Axis parameters always
    refer to those supplied endpoints. The OBB rotation maps local coordinates to world.
    """
    center, R, half, world = (np.asarray(a, dtype=float) for a in (center, R, half, world))
    if (
        center.shape != (3,)
        or R.shape != (3, 3)
        or half.shape != (3,)
        or world.shape not in ((1, 3), (2, 3))
        or not all(np.isfinite(a).all() for a in (center, R, half, world))
        or not np.isfinite(radius)
        or radius <= 0
        or np.any(half <= 0)
        or not np.allclose(R @ R.T, np.eye(3), atol=1e-9, rtol=0)
        or np.linalg.det(R) <= 0
    ):
        raise ValueError("invalid box/capsule geometry")
    local = (world - center) @ R
    a, b = local[0], local[-1]
    separation = np.maximum(np.minimum(a, b) - half, -half - np.maximum(a, b))
    if np.linalg.norm(np.maximum(separation, 0)) >= radius:
        return
    try:
        us = _closest_span(a, b, half)
    except _AxisInsideBox:
        escape, distance, sites = _interior_span(a, b, half)
        normal = -(R @ escape)
        for u, closest in sites:
            line = (1 - u) * world[0] + u * world[-1]
            surface = line + radius * normal
            skin = center + R @ closest
            yield dict(
                gap_m=-(radius + distance),
                normal=normal if capsule_first else -normal,
                first_surface=surface if capsule_first else skin,
                second_surface=skin if capsule_first else surface,
                axis_parameter=u,
                weight=1 / len(sites),
                axis_point=line,
            )
        return
    for u in us:
        axis = (1 - u) * a + u * b
        closest = np.clip(axis, -half, half)
        delta = closest - axis
        distance = np.linalg.norm(delta)
        if distance == 0:
            raise ValueError("capsule axis and box material witnesses cannot be distinguished")
        gap = distance - radius
        if gap >= 0:
            continue
        normal = R @ (delta / distance)
        skin = center + R @ closest
        # Bind the link witness directly to the two original finite endpoints.
        line = (1 - u) * world[0] + u * world[-1]
        surface = line + radius * normal
        yield dict(
            gap_m=gap,
            normal=normal if capsule_first else -normal,
            first_surface=surface if capsule_first else skin,
            second_surface=skin if capsule_first else surface,
            axis_parameter=u,
            weight=1 / len(us),
            axis_point=line,
        )
