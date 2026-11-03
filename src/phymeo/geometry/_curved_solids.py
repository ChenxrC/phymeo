"""Analytic cylinder/sphere and curved-solid/half-space queries."""

from dataclasses import dataclass
from fractions import Fraction as F

import numpy as np

from ..constraints.collision_shapes import CollisionKey
from ._prism_pair_exact import sqrt_bounds
from ._round_sweep import _translation, _vector, _xyz
from .collision_witness import CollisionWitness


@dataclass(frozen=True)
class CurvedEvent:
    """A contact time bracket and both material witnesses evaluated at witness_time."""

    interval: tuple[float, float]
    witness_time: float
    witness: CollisionWitness


@dataclass(frozen=True)
class CurvedContact:
    """Boundary features owned by one curved solid, in world coordinates.

    points_m gives the unique support, segment endpoints, representative
    diameters of a disk/circle, or representatives of equally close features.
    The cone's base-disk center is listed first. A cylinder rim can have
    a unique closest point and a whole cone of normals.
    """

    owner: CollisionKey
    features: tuple[str, ...]
    locus: str
    points_m: tuple[tuple[float, ...], ...]
    first_contact: CurvedEvent | None = None
    last_contact: CurvedEvent | None = None


def _rotation(asset):
    # Dividing by |q|² retains exact orthogonality for the stored binary
    # quaternion, including around a support tie or a tangency.
    x, y, z, w = map(F, asset.orientation_xyzw)
    s = 2 / (x * x + y * y + z * z + w * w)
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=object,
    )


def _root(squared):
    lo, hi = sqrt_bounds(squared, 96)
    return (lo + hi) / 2


def _minus_root(value, squared, root):
    """Signed value - sqrt(squared), rationalized to preserve a near-zero sign."""
    if value <= 0:
        return value - root
    return (value * value - squared) / (value + root)


def _support_upper(kind, dimensions, direction):
    if kind == "ellipsoid":
        scaled = dimensions * direction
        return sqrt_bounds(scaled @ scaled, 96)[1]
    radius, half_height = dimensions
    radial = radius * sqrt_bounds(direction[:2] @ direction[:2], 96)[1]
    axial = half_height * direction[2]
    if kind == "cone":
        return max(axial, -axial + radial)
    return abs(axial) + radial


def curved_bounds(asset, path):
    """Outward rational world box before the pipeline adds its broad-phase margin."""
    dims = _vector(asset.dimensions_m)
    start = _vector(path.start[0])
    turn = path.turn
    if turn is not None and turn.angle:
        pivot = _vector(turn.pivot_start)
        end = _vector(turn.pivot_end)
        radius = max(dims) if asset.kind == "ellipsoid" else sqrt_bounds(dims @ dims, 96)[1]
        # The whole solid remains inside this sphere throughout the screw path.
        reach = sum(abs(start - pivot)) + radius
        return np.minimum(pivot, end) - reach, np.maximum(pivot, end) + reach
    end = start + _translation(path)
    frame = _rotation(asset)
    upper = [_support_upper(asset.kind, dims, axis) for axis in frame]
    lower = [_support_upper(asset.kind, dims, -axis) for axis in frame]
    return np.minimum(start, end) - lower, np.maximum(start, end) + upper


def _plane_support(kind, dims, n, center_height):
    """Return minimum support points and their signed, unnormalized plane height."""
    if kind == "ellipsoid":
        scaled = dims * n
        squared = scaled @ scaled
        support = _root(squared)
        point = -dims * scaled / support
        return ("surface",), "point", (point,), _minus_root(center_height, squared, support)

    radius, h = dims
    nz = n[2]
    tangent = n[:2]
    squared = tangent @ tangent
    length = _root(squared)
    if kind == "cylinder":
        cap = "bottom_cap" if nz > 0 else "top_cap"
        z = -h if nz > 0 else h
        if not squared:
            disk = tuple(
                _vector((x, y, z))
                for x, y in ((0, 0), (radius, 0), (-radius, 0), (0, radius), (0, -radius))
            )
            return (cap,), "disk", disk, center_height + z * nz
        xy = -radius * tangent / length
        height = _minus_root(
            center_height - h * abs(nz), radius * radius * squared, radius * length
        )
        if nz == 0:
            return ("side",), "segment", (_vector((*xy, -h)), _vector((*xy, h))), height
        return ("side", cap), "point", (_vector((*xy, z)),), height
    apex = _vector((0, 0, h))
    # The cone is the convex hull of its apex and base disk. Compare their
    # support heights without rounding sqrt(nx² + ny²) before testing the tie.
    rim_sq = radius * radius * squared
    base_minus_tip = _minus_root(-2 * h * nz, rim_sq, radius * length)
    tip_height = center_height + h * nz
    if base_minus_tip > 0:
        return ("apex",), "point", (apex,), tip_height
    if not squared:
        base = tuple(
            _vector((dx, dy, -h))
            for dx, dy in (
                (0, 0),
                (radius, 0),
                (-radius, 0),
                (0, radius),
                (0, -radius),
            )
        )
        return ("base_disk",), "disk", base, center_height - h * nz
    rim = _vector((*(-radius * tangent / length), -h))
    if not base_minus_tip:
        return ("apex", "base_rim"), "segment", (apex, rim), tip_height
    gap = _minus_root(center_height - h * nz, rim_sq, radius * length)
    return ("base_rim",), "point", (rim,), gap


def _cylinder_sphere(dims, p, ball):
    """Find the cylinder boundary nearest to the sphere center in cylinder coordinates."""
    r, h = dims
    z = p[2]
    radial_sq = p[:2] @ p[:2]
    rho = _root(radial_sq)
    dr = -_minus_root(r, radial_sq, rho)
    dz = abs(z) - h
    cap_sign = 1 if z >= 0 else -1
    cap = "top_cap" if z >= 0 else "bottom_cap"
    cap_normal = _vector((0, 0, cap_sign))
    radial = _vector((p[0] / rho, p[1] / rho, 0)) if rho else None
    if dr > 0 or dz > 0:
        near = p.copy()
        if dr >= 0:
            near[:2] = r * radial[:2]
        if dz >= 0:
            near[2] = cap_sign * h
        if dr > 0 and dz > 0:
            distance = _root(dr * dr + dz * dz)
            normal = (dr * radial + dz * cap_normal) / distance
            # (rho-r)² + dz² - ball², rationalized before deciding contact.
            bias = radial_sq + r * r + dz * dz - ball * ball
            delta = _minus_root(bias, 4 * r * r * radial_sq, 2 * r * rho)
            gap = delta / (distance + ball)
        elif dr > 0:
            normal = radial
            gap = -_minus_root(r + ball, radial_sq, rho)
        else:
            normal = cap_normal
            gap = dz - ball
        features = (("side",) if dr >= 0 else ()) + ((cap,) if dz >= 0 else ())
        return features, "point", (near,), normal, gap

    if dr == dz == 0:
        return ("side", cap), "normal_cone", (p,), None, -ball
    # Interior: keep every closest surface. Comparing rho against r+dz
    # by its square distinguishes a real tie from a rounded one.
    side_minus_cap = -_minus_root(r + dz, radial_sq, rho)
    features, points = [], []
    normal = None
    if side_minus_cap >= 0:
        features.append("side")
        if rho:
            points.append(_vector((*(r * radial[:2]), z)))
            normal = radial
        else:
            points.extend(
                _vector((dx, dy, z))
                for dx, dy in (
                    (r, 0),
                    (-r, 0),
                    (0, r),
                    (0, -r),
                )
            )
    if side_minus_cap <= 0:
        for sign, name in ((-1, "bottom_cap"), (1, "top_cap")):
            if sign * z >= 0:
                features.append(name)
                points.append(_vector((*p[:2], sign * h)))
                normal = _vector((0, 0, sign))
    gap = -_minus_root(r + ball, radial_sq, rho) if side_minus_cap > 0 else dz - ball
    if len(points) == 1:
        return tuple(features), "point", tuple(points), normal, gap
    locus = "circle" if features == ["side"] else "multiple"
    return tuple(features), locus, tuple(points), None, gap


def _static(curved, other, m_curved, m_other, curved_first, *, frame=None):
    asset = curved.curved
    dims = _vector(asset.dimensions_m)
    center = _vector(m_curved.start[0])
    anchor = _vector(m_other.start[0])
    if frame is None:
        frame = _rotation(asset)
    if other.kind == "plane":
        plane_n = _vector(other.normal)
        norm = _root(plane_n @ plane_n)
        n = -plane_n / norm  # outward from the curved solid toward the half-space
        features, locus, local, height = _plane_support(
            asset.kind, dims, frame.T @ plane_n, (center - anchor) @ plane_n
        )
        gap = height / norm
    else:
        features, locus, local, local_n, gap = _cylinder_sphere(
            dims, frame.T @ (anchor - center), F(other.radius)
        )
        n = None if local_n is None else frame @ local_n
    offsets = [frame @ point for point in local]
    points = tuple(_xyz(center + offset) for offset in offsets)
    trace = CurvedContact(curved.key, features, locus, points)
    cw, ow = ((curved.vertex_ids[0], 1.0),), ((other.vertex_ids[0], 1.0),)
    cp = op = co = oo = None
    if n is not None:
        offset = offsets[0]
        if other.kind == "plane":
            # Project the actual support vector. Even an irrational ellipsoid support
            # then has an exactly planar counterpart before rounding to world FP64.
            relative = center - anchor + offset
            other_offset = relative - plane_n * ((relative @ plane_n) / (plane_n @ plane_n))
        else:
            other_offset = -F(other.radius) * n
        cp, op = points[0], _xyz(anchor + other_offset)
        co, oo = _xyz(offset), _xyz(other_offset)
    if not curved_first:
        cp, op, cw, ow, co, oo = op, cp, ow, cw, oo, co
        if n is not None:
            n = -n
    published_gap = float(gap)
    if gap and not published_gap:
        raise OverflowError("curved gap cannot be represented in FP64")
    witness = CollisionWitness(
        published_gap,
        None if n is None else _xyz(n),
        cp,
        op,
        cw,
        ow,
        "host-analytic-curved-solid",
        first_offset=co,
        second_offset=oo,
    )
    status = "separated" if gap > 0 else "overlap" if gap < 0 else "touching"
    return status, "nonunique_normal" if n is None else "analytic_curved_solid", witness, trace


def curved_solid_contact(a, b, ma, mb):
    """Return endpoint geometry and any whole-path contact bracket for a supported pair."""
    curved_first = a.curved is not None
    curved, other, mc, mo = (a, b, ma, mb) if curved_first else (b, a, mb, ma)
    if (curved.kind, other.kind) not in (
        ("cylinder", "sphere"),
        ("cylinder", "plane"),
        ("cone", "plane"),
        ("ellipsoid", "plane"),
    ):
        return "unsupported", "curved_shape_pair", None, None, None
    try:
        if ma.moving or mb.moving:
            if curved.kind == "ellipsoid" and other.kind == "plane":
                from ._rotating_ellipsoid_plane import rotating_ellipsoid_plane

                return rotating_ellipsoid_plane(curved, other, mc, mo, curved_first)
            return "unsupported", "curved_sweep", None, None, None
        return (*_static(curved, other, mc, mo, curved_first), None)
    except OverflowError:
        return "unresolved", "witness_outside_fp64", None, None, None
