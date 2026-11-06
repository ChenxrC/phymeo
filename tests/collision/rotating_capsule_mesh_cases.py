"""Frozen independent inputs for capsule rotation against a closed finite mesh.

This module imports no Phymeo geometry implementation. Distances and rotations
used by the reference checks are independent of the new continuous producer.
"""

from dataclasses import dataclass
from math import acos, cos, pi, sin

import numpy as np

import phymeo as ph

BOX_FACES = (
    (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
    (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
    (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
)


def box(lower, upper):
    return tuple((x, y, z) for x in (lower[0], upper[0])
                 for y in (lower[1], upper[1]) for z in (lower[2], upper[2])), BOX_FACES


@dataclass(frozen=True)
class Case:
    name: str
    vertices: tuple
    triangles: tuple
    endpoints: tuple
    radius: float
    angle: float
    axis: tuple = (0.0, 0.0, 1.0)
    pivot_start: tuple = (0.0, 0.0, 0.0)
    pivot_end: tuple = (0.0, 0.0, 0.0)
    mesh_step: tuple = (0.0, 0.0, 0.0)


B01 = Case(
    "B01-orbit-shaft", *box((1.0, -1 / 8, -1 / 4), (2.0, 1 / 8, 1 / 4)),
    ((0.0, 3 / 2, -1.0), (0.0, 3 / 2, 1.0)), 1 / 8, -pi,
)


def public_case(case, *, world=0, reverse=False, reorder=False, capsule_world=None):
    """Construct only the existing public registrations and actual stored motions."""
    km = ph.CollisionKey(world, int(reverse), 0)
    kc = ph.CollisionKey(world if capsule_world is None else capsule_world, int(not reverse), 0)
    ids = tuple(range(len(case.vertices)))
    permutation = tuple(reversed(ids)) if reorder else ids
    mesh = ph.CollisionShape(km, "mesh", permutation, triangles=case.triangles)
    cap = ph.CollisionShape(kc, "capsule", (41, 93), case.radius)
    vertices = np.array([case.vertices[i] for i in permutation], dtype=float)
    end = None if not any(case.mesh_step) else vertices + case.mesh_step
    motion = ph.CollisionMotion(
        case.endpoints,
        turn=ph.RigidTurn(case.pivot_start, case.pivot_end, case.axis, case.angle),
    )
    return ph.CollisionModel((cap, mesh)), {
        km: ph.CollisionMotion(vertices, end), kc: motion
    }, mesh, cap


def stored_pose(path, time):
    """Independent ordinary Rodrigues evaluation of the *stored* public motion."""
    points = np.asarray(path.start)
    if path.turn is None:
        return points.copy() if path.end is None else points + time * (path.end - points)
    turn = path.turn
    axis = np.asarray(turn.axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    pivot = np.asarray(turn.pivot_start)
    arms = points - pivot
    theta = time * turn.angle
    return (pivot + time * (np.asarray(turn.pivot_end) - pivot)
            + cos(theta) * arms + sin(theta) * np.cross(axis, arms)
            + (1 - cos(theta)) * np.outer(arms @ axis, axis))


def b01_first(stored_angle):
    # y_axis=3*cos(alpha)/2 first reaches y_face+r=1/4; x=sqrt(35)/4 is inside.
    return acos(1 / 6) / abs(stored_angle)


def b01_face_gap(stored_angle, time):
    # Used only where the independently checked x coordinate is inside [1,2].
    return 1.5 * cos(abs(stored_angle) * time) - 0.25


def u_mesh():
    """A closed U made from seven cells, with shared interior faces removed.

    The solid is {|x| in [1,2], y in [-2,2]} union {x in [-1,1], y in [-2,-1]},
    extruded over z in [-1/4,1/4]. Its notch contains the complete radius-1/4 orbit.
    """
    cuts = (-2.0, -1.0, 1.0, 2.0)
    occupied = {(i, j) for i in range(3) for j in range(3) if i != 1 or j == 0}
    vertices, faces, ids = [], [], {}
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1), None, None)
    for i, j in sorted(occupied):
        local, _ = box((cuts[i], cuts[j], -0.25), (cuts[i+1], cuts[j+1], 0.25))
        for side, delta in enumerate(neighbors):
            if delta is not None and (i+delta[0], j+delta[1]) in occupied:
                continue
            for face in BOX_FACES[2*side:2*side+2]:
                row = []
                for node in face:
                    point = local[node]
                    if point not in ids:
                        ids[point] = len(vertices)
                        vertices.append(point)
                    row.append(ids[point])
                faces.append(tuple(row))
    return tuple(vertices), tuple(faces)


# B02 is a true finite-axis clearance control, not the producer's observed label.
B02 = Case("B02-radial-clear", B01.vertices, B01.triangles,
           ((0.0, 9/4, -1.0), (0.0, 9/4, 1.0)), 1/8, -pi)
B03 = Case("B03-concave-notch", *u_mesh(),
           ((0.0, 1/4, -1.0), (0.0, 1/4, 1.0)), 1/8, -2*pi)
B04 = Case("B04-turning-endcap", *box((-2.0, 1.0, -0.25), (2.0, 1.25, 0.25)),
           ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), 1/8, 3.0)
B05_INSIDE = Case("B05-initial-containment", *box((-2.0,)*3, (2.0,)*3),
                  ((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)), 1/8, 1.25)
B05_OPEN = Case("B05-initial-touch-opening", B04.vertices, B04.triangles,
                ((0.0, 0.0, 0.0), (0.0, 7/8, 0.0)), 1/8, 1.0)
B05_CLOSE = Case("B05-initial-touch-closing", B04.vertices, B04.triangles,
                 ((0.0, 0.0, 0.0), (0.5, 7/8, 0.0)), 1/8, 0.25)
B05_TANGENT = Case("B05-isolated-tangency", *box((0.25, 0.25, -0.125), (0.75, 1.0, 0.125)),
                   B01.endpoints, 0.25, -1.5)
B06_FINITE = Case("B06-infinite-line-false-positive", B01.vertices, B01.triangles,
                  ((0.0, 1.5, 0.625), (0.0, 1.5, 0.875)), 1/8, -pi)


def transform(case, matrix, offset=(0.0, 0.0, 0.0), *, name=None):
    """Covariance of positions and axial vectors; reflections reverse face winding."""
    from dataclasses import replace

    matrix = np.asarray(matrix, dtype=float)
    offset = np.asarray(offset, dtype=float)
    sign = 1 if np.linalg.det(matrix) > 0 else -1

    def points(values):
        return tuple(tuple(matrix @ np.asarray(p) + offset) for p in values)

    faces = case.triangles if sign > 0 else tuple((a, c, b) for a, b, c in case.triangles)
    return replace(case, name=case.name if name is None else name,
        vertices=points(case.vertices), triangles=faces, endpoints=points(case.endpoints),
        pivot_start=points((case.pivot_start,))[0], pivot_end=points((case.pivot_end,))[0],
        axis=tuple(sign * matrix @ case.axis), mesh_step=tuple(matrix @ case.mesh_step))


def b04_oblique_moving():
    """A true changing axis with independent pivot drift, all fixed before execution."""
    from dataclasses import replace

    basis = np.array(((1.0, 0.0, 0.0), (0.0, 3/5, -4/5), (0.0, 4/5, 3/5)))
    common = np.array((1/8, -1/16, 3/32))
    result = transform(B04, basis, (1/4, -1/2, 3/8), name="B04-oblique-moving-pivot")
    return replace(result, mesh_step=tuple(common),
        pivot_end=tuple(np.asarray(result.pivot_start) + common + basis @ (0, 1/16, 0)))


# Independent exact finite-feature minimization on the stored sample coordinates.
# It enumerates endpoints, triangle face interiors, edges, and segment-plane
# intersections. No Phymeo closest-feature or root implementation is imported.
def _r(value):
    from fractions import Fraction

    return tuple(Fraction(float(x)) for x in value)


def _add(a, b):
    return tuple(x+y for x, y in zip(a, b, strict=True))


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b, strict=True))


def _scale(a, s):
    return tuple(x*s for x in a)


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b, strict=True))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _clamp(value):
    return min(1, max(0, value))


def _barycentric(point, tri):
    a, b, c = tri
    u, v, d = _sub(b, a), _sub(c, a), _sub(point, a)
    aa, ab, bb = _dot(u, u), _dot(u, v), _dot(v, v)
    den = aa*bb-ab*ab
    if not den:
        raise AssertionError("independent oracle requires nondegenerate triangles")
    s, t = (bb*_dot(d, u)-ab*_dot(d, v))/den, (aa*_dot(d, v)-ab*_dot(d, u))/den
    return (1-s-t, s, t)


def _point_triangle(point, tri):
    candidates = []
    weights = _barycentric(point, tri)
    if min(weights) >= 0:
        projected = tuple(sum(w*p[k] for w, p in zip(weights, tri, strict=True)) for k in range(3))
        delta = _sub(point, projected)
        candidates.append((_dot(delta, delta), weights))
    for i, j in ((0, 1), (1, 2), (2, 0)):
        edge = _sub(tri[j], tri[i])
        t = _clamp(_dot(_sub(point, tri[i]), edge)/_dot(edge, edge))
        delta = _sub(point, _add(tri[i], _scale(edge, t)))
        w = [0, 0, 0]
        w[i], w[j] = 1-t, t
        candidates.append((_dot(delta, delta), tuple(w)))
    return min(candidates)


def _segment_segment(a, b, c, d):
    u, v, w = _sub(b, a), _sub(d, c), _sub(a, c)
    aa, ab, bb, aw, bw = _dot(u, u), _dot(u, v), _dot(v, v), _dot(u, w), _dot(v, w)
    candidates = []
    for s in (0, 1):
        candidates.append((s, _clamp((bw+s*ab)/bb) if bb else 0))
    for t in (0, 1):
        candidates.append((_clamp((t*ab-aw)/aa) if aa else 0, t))
    den = aa*bb-ab*ab
    if den:
        s, t = (ab*bw-bb*aw)/den, (aa*bw-ab*aw)/den
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append((s, t))
    values = []
    for s, t in candidates:
        delta = _sub(_add(a, _scale(u, s)), _add(c, _scale(v, t)))
        values.append((_dot(delta, delta), s, t))
    return min(values)


def segment_triangle_reference(endpoints, triangle):
    a, b = map(_r, endpoints)
    tri = tuple(map(_r, triangle))
    candidates = []
    for s, point in ((0, a), (1, b)):
        square, weights = _point_triangle(point, tri)
        candidates.append((square, s, weights))
    for i, j in ((0, 1), (1, 2), (2, 0)):
        square, s, t = _segment_segment(a, b, tri[i], tri[j])
        weights = [0, 0, 0]
        weights[i], weights[j] = 1-t, t
        candidates.append((square, s, tuple(weights)))
    normal = _cross(_sub(tri[1], tri[0]), _sub(tri[2], tri[0]))
    den = _dot(normal, _sub(b, a))
    if den:
        s = _dot(normal, _sub(tri[0], a))/den
        if 0 <= s <= 1:
            weights = _barycentric(_add(a, _scale(_sub(b, a), s)), tri)
            if min(weights) >= 0:
                candidates.append((0, s, weights))
    return min(candidates)


def mesh_distance_reference(mesh, capsule, paths, time):
    vertices = dict(zip(mesh.vertex_ids, stored_pose(paths[mesh.key], time), strict=True))
    ends = stored_pose(paths[capsule.key], time)
    rows = [(segment_triangle_reference(ends, [vertices[i] for i in face]), face) for face in mesh.triangles]
    return min(rows)


def first_endcap_reference(mesh, capsule, paths):
    """Independent numerical bracket for B04's unique first monotone face approach.

    The authored relative upper endpoint is sin(3t)+d*t, d=0 or 1/16.
    On [1/4,1/2] its derivative stays positive and x lies in the finite face.
    Stored FP64 Rodrigues poses are fed to independent exact finite-feature QPs;
    this is a 2^-40 reference interval, not the producer's interval proof.
    """
    lo, hi = 1/4, 1/2
    radius2 = capsule.radius**2
    assert mesh_distance_reference(mesh, capsule, paths, lo)[0][0] > radius2
    assert mesh_distance_reference(mesh, capsule, paths, hi)[0][0] < radius2
    while hi-lo > 2**-40:
        mid = (lo+hi)/2
        if mesh_distance_reference(mesh, capsule, paths, mid)[0][0] > radius2:
            lo = mid
        else:
            hi = mid
    return lo, hi


def closed_oriented_reference(vertices, triangles):
    """Exact edge incidence and positive signed volume, independent of ClosedMesh."""
    edge = {}
    volume6 = 0
    for face in triangles:
        assert len(set(face)) == 3
        a, b, c = (_r(vertices[i]) for i in face)
        assert any(_cross(_sub(b, a), _sub(c, a)))
        volume6 += _dot(a, _cross(b, c))
        for i, j in zip(face, face[1:]+face[:1], strict=True):
            edge.setdefault(tuple(sorted((i, j))), []).append((i, j))
    assert all(len(values) == 2 and values[0] == values[1][::-1] for values in edge.values())
    assert volume6 > 0
    return volume6/6
