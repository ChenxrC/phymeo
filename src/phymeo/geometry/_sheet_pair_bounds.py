"""Exact separating-plane checks for the ordinary pairs of a single affine triangle sheet."""

from itertools import combinations
from math import isqrt


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]


def _unit_proposal(v):
    # This direction is only a proposal. All accepted signs and distances use
    # the original dyadic coordinates, including ties at the layer width.
    length = isqrt(_dot(v, v))
    return tuple(x * (1 << 20) // length for x in v) if length else (0, 0, 0)


def _edges(face):
    return _sub(face[1], face[0]), _sub(face[2], face[1]), _sub(face[0], face[2])


def _regular(a, b):
    e0, u0 = _sub(a[1], a[0]), _sub(a[2], a[0])
    e1, u1 = _sub(b[1], b[0]), _sub(b[2], b[0])
    coefficients = (_cross(e0, u0), _add(_cross(e0, u1), _cross(e1, u0)), _cross(e1, u1))
    # Any fixed direction with strictly one-signed Bernstein coefficients
    # certifies the entire face. Its start normal need not suit a wide turn.
    if any(min(values) > 0 or max(values) < 0 for values in zip(*coefficients, strict=True)):
        return True
    normal = coefficients[0]
    return all(_dot(normal, area) > 0 for area in coefficients)


def _opposite(first, second):
    return (min(first) > 0 and max(second) < 0) or (max(first) < 0 and min(second) > 0)


def _vertex_only(a, b, common, point):
    u = [[_sub(point[k][t], point[common][t]) for k in a if k != common] for t in (0, 1)]
    v = [[_sub(point[k][t], point[common][t]) for k in b if k != common] for t in (0, 1)]
    for us, vs in zip(u, v, strict=True):
        d = _sub(_add(*map(_unit_proposal, us)), _add(*map(_unit_proposal, vs)))
        if _opposite(
            [_dot(d, x) for side in u for x in side], [_dot(d, x) for side in v for x in side]
        ):
            return True
    return False


def _edge_only(a, b, common, point):
    i, j = common
    opposite_a = next(node for node in a if node not in common)
    opposite_b = next(node for node in b if node not in common)
    e = [_sub(point[j][t], point[i][t]) for t in (0, 1)]
    u = [_sub(point[opposite_a][t], point[i][t]) for t in (0, 1)]
    v = [_sub(point[opposite_b][t], point[i][t]) for t in (0, 1)]
    for t in (0, 1):
        ref = _sub(_unit_proposal(_cross(e[t], v[t])), _unit_proposal(_cross(e[t], u[t])))
        n0, n1 = (_cross(edge, ref) for edge in e)

        # The moving plane always contains the entire common edge. Opposite
        # strict Bernstein signs keep both triangles in opposite open sides.
        def coefficients(w):
            return _dot(n0, w[0]), _dot(n0, w[1]) + _dot(n1, w[0]), _dot(n1, w[1])

        if _opposite(coefficients(u), coefficients(v)):
            return True
    return False


def _directions(a, b):
    yield _sub(_add(_add(b[0], b[1]), b[2]), _add(_add(a[0], a[1]), a[2]))
    ae, be = _edges(a), _edges(b)
    an, bn = _cross(ae[0], ae[1]), _cross(be[0], be[1])
    yield an
    yield bn
    for edge in ae:
        yield _cross(an, edge)
    for edge in be:
        yield _cross(bn, edge)
    for e in ae:
        for h in be:
            yield _cross(e, h)


def _far(first, second, margin):
    for a, b in zip(first, second, strict=True):
        for d in _directions(a, b):
            width2 = margin * margin * _dot(d, d)
            gaps = []
            for x, y in zip(first, second, strict=True):
                ax = [_dot(p, d) for p in x]
                bx = [_dot(p, d) for p in y]
                gaps.append((min(bx) - max(ax), min(ax) - max(bx)))
            for side in zip(*gaps, strict=True):
                gap = min(side)
                if gap > 0 and gap * gap > width2:
                    return True
    return False


class SheetPairBounds:
    def __init__(self, shapes):
        if len({(s.key.world, s.key.body) for s in shapes}) != 1 or any(
            s.kind != "triangle" or s.radius for s in shapes
        ):
            raise ValueError("sheet bounds require one zero-thickness triangle sheet")
        self.shapes = shapes
        self.pairs = tuple(
            (i, j, tuple(sorted(set(shapes[i].vertex_ids) & set(shapes[j].vertex_ids))))
            for i, j in combinations(range(len(shapes)), 2)
        )
        self.adjacent = self.far = self.fallback = 0

    def _dyadic_points(self, paths, margin):
        points = {}
        numerator, den = margin.as_integer_ratio()
        shift = den.bit_length() - 1
        for shape in self.shapes:
            path = paths[shape.key]
            if path.turn is not None:
                return None
            end = path.start if path.end is None else path.end
            for k, p, q in zip(shape.vertex_ids, path.start, end, strict=True):
                rows = (tuple(map(float, p)), tuple(map(float, q)))
                if k in points:
                    if points[k][0] != rows:
                        return None
                    continue
                ratios = tuple(tuple(x.as_integer_ratio() for x in r) for r in rows)
                shift = max(shift, max(d.bit_length() - 1 for row in ratios for _, d in row))
                points[k] = (rows, ratios)

        def convert(n, d):
            return n << (shift - (d.bit_length() - 1))

        exact = {
            k: tuple(tuple(convert(n, d) for n, d in row) for row in ratios)
            for k, (_, ratios) in points.items()
        }
        origin = next(iter(exact.values()))[0]
        exact = {k: tuple(_sub(p, origin) for p in ends) for k, ends in exact.items()}
        width = convert(numerator, den)
        if width.bit_length() > 384 or any(
            v.bit_length() > 384 for ends in exact.values() for row in ends for v in row
        ):
            return None
        return exact, width

    def classify(self, paths, flags, margin):
        data = self._dyadic_points(paths, margin)
        resolved = {}
        if data is not None:
            p, width = data
            faces = [
                tuple(tuple(p[k][t] for k in s.vertex_ids) for t in (0, 1)) for s in self.shapes
            ]
            regular = [_regular(*triangle) for triangle in faces]
            for slot, (i, j, common) in enumerate(self.pairs):
                if flags[slot] or not regular[i] or not regular[j]:
                    continue
                a, b = self.shapes[i].vertex_ids, self.shapes[j].vertex_ids
                if len(common) == 1:
                    proved = _vertex_only(a, b, common[0], p)
                elif len(common) == 2:
                    proved = _edge_only(a, b, common, p)
                elif not common:
                    proved = _far(faces[i], faces[j], width)
                else:
                    proved = False
                if proved:
                    resolved[slot] = common
        ordinary = sum(bool(ids) for ids in resolved.values())
        self.adjacent += ordinary
        self.far += len(resolved) - ordinary
        self.fallback += int((flags == 0).sum()) - len(resolved)
        return resolved
