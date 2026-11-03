"""Independent oracle: clip material-space triangles against the fixture's rectangular solid pieces."""

from fractions import Fraction as F


def clipped_fractions(vertices, boxes):
    vertices = [tuple(map(F, v)) for v in vertices]
    inside = boundary = F(0)
    for low, high in boxes:
        poly = [(F(0), F(0)), (F(1), F(0)), (F(0), F(1))]
        on_exposed_face = False
        for k in range(3):
            for level, sign in ((F(low[k]), 1), (F(high[k]), -1)):
                d = [sign * (v[k] - level) for v in vertices]
                if not any(d):
                    # These fixtures only lie coplanar on an exposed horizontal
                    # face, never on an internal divider between the U's boxes.
                    on_exposed_face = True
                if not poly:
                    break
                result = []
                for p, q in zip(poly, poly[1:] + poly[:1], strict=True):
                    h0 = d[0] + (d[1] - d[0]) * p[0] + (d[2] - d[0]) * p[1]
                    h1 = d[0] + (d[1] - d[0]) * q[0] + (d[2] - d[0]) * q[1]
                    if h0 >= 0:
                        result.append(p)
                    if h0 * h1 < 0:
                        s = h0 / (h0 - h1)
                        result.append(tuple(x + s * (y - x) for x, y in zip(p, q, strict=True)))
                poly = result
        twice_area = abs(
            sum(p[0] * q[1] - p[1] * q[0] for p, q in zip(poly, poly[1:] + poly[:1], strict=True))
        )
        if on_exposed_face:
            boundary += twice_area
        else:
            inside += twice_area
    return inside, boundary, 1 - inside - boundary
