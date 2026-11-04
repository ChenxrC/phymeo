"""Finite round link against an oriented box, for capsule spines outside the box solid."""

from fractions import Fraction as F

import numpy as np

from .sphere_link_contact import _RigidLinkContacts


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
        raise ValueError(
            "capsule or sphere axis intersects the payload box; unique exterior separation unavailable"
        )
    if not any(direction):
        last = first = F(0)
    if first != last and float(first) == float(last):
        raise ValueError("distinct capsule material parameters are below output resolution")
    return (float(first),) if first == last else (float(first), float(last))


class BoxLinkContacts(_RigidLinkContacts):
    """Free OBB versus finite capsules or spheres with a unique exterior separation normal."""

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
    us = _closest_span(a, b, half)
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
