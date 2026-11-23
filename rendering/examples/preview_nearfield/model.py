"""Distance-weighted local diffuse transport for motion previews, in metres."""
from math import isfinite


def local_weight(distance, radius):
    if not (isfinite(distance) and distance >= 0 and isfinite(radius) and radius > 0):
        raise ValueError("expected a finite nonnegative hit distance and positive radius")
    t = min(distance / radius, 1.)
    return (1. - t) ** 2 * (1. + 2. * t)


def mix_hit(sky, hit, distance, radius):
    """Convex mixture for one incident direction before cosine integration."""
    strength = local_weight(distance, radius)
    return tuple(a + strength * (b - a) for a, b in zip(sky, hit, strict=True))
