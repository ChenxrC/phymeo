"""Exact projected convex feature intersection and prism material recovery."""

from ._prism_exact import Quadratic as Q
from ._prism_exact import add, dot, scale, sub


def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def hull(points):
    points = sorted(set(points))
    if len(points) <= 2:
        return tuple(points)
    halves = []
    for values in (points, list(reversed(points))):
        out = []
        for p in values:
            while len(out) > 1 and orient(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        halves.append(out[:-1])
    return tuple(halves[0] + halves[1])


def edges(poly):
    if len(poly) < 2:
        return ()
    return (
        tuple(zip(poly, poly[1:] + poly[:1], strict=True))
        if len(poly) > 2
        else ((poly[0], poly[1]),)
    )


def inside(p, poly):
    if len(poly) == 1:
        return p == poly[0]
    if len(poly) == 2:
        return orient(poly[0], poly[1], p) == 0 and all(
            min(a, b) <= x <= max(a, b) for x, a, b in zip(p, *poly, strict=True)
        )
    return all(orient(a, b, p) >= 0 for a, b in edges(poly))


def intersect(a, b):
    found = [p for p in a if inside(p, b)] + [p for p in b if inside(p, a)]
    for p, q in edges(a):
        for r, s in edges(b):
            u = (q[0] - p[0], q[1] - p[1])
            v = (s[0] - r[0], s[1] - r[1])
            det = u[0] * v[1] - u[1] * v[0]
            if not det:
                continue
            w = (r[0] - p[0], r[1] - p[1])
            t = (w[0] * v[1] - w[1] * v[0]) / det
            z = (w[0] * u[1] - w[1] * u[0]) / det
            if 0 <= t <= 1 and 0 <= z <= 1:
                found.append((p[0] + t * u[0], p[1] + t * u[1]))
    return hull(found)


def clip_support(a, b, d):
    d2 = dot(d, d)
    dropped = next(i for i, x in enumerate(d) if x)
    kept = [i for i in range(3) if i != dropped]

    def project(p):
        y = sub(p, scale(d, dot(d, p) / d2))
        return tuple(y[i] for i in kept)

    poly = intersect(hull([project(p) for p in a]), hull([project(p) for p in b]))
    result = []
    for p in poly:
        y = [Q(0)] * 3
        for i, x in zip(kept, p, strict=True):
            y[i] = x
        y[dropped] = -sum((d[i] * y[i] for i in kept), Q(0)) / d[dropped]
        ra = add(y, scale(d, (dot(d, a[0]) - dot(d, y)) / d2))
        rb = add(y, scale(d, (dot(d, b[0]) - dot(d, y)) / d2))
        result.append((ra, rb))
    return tuple(result)


def material(point, x, n):
    s = dot(n, sub(point, x[0]))
    mid = sub(point, scale(n, s))
    a, b, r = sub(x[1], x[0]), sub(x[2], x[0]), sub(mid, x[0])
    aa, ab, bb = dot(a, a), dot(a, b), dot(b, b)
    den = aa * bb - ab * ab
    b1 = (dot(r, a) * bb - dot(r, b) * ab) / den
    b2 = (dot(r, b) * aa - dot(r, a) * ab) / den
    return 1 - b1 - b2, b1, b2, s
