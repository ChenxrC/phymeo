"""Static convex material features using support witnesses on the Minkowski difference."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

import numpy as np

from ..core.capabilities import SolverFailure
from ._convex_hull import HullShape
from ._curved_solids import _rotation
from ._prism_pair_exact import sqrt_bounds
from ._round_sweep import _vector, _xyz
from ._triangle_pair_sweep import _FacePath
from .collision_witness import CollisionWitness

_TOL_M = 2e-7
_MAX_GJK = 64
_MAX_EPA = 192
_VERTEX_SUPPORT = frozenset(
    ("sphere", "capsule", "hull", "triangle", "particle", "rounded_particle")
)
_KINDS = _VERTEX_SUPPORT | {"cylinder", "cone", "ellipsoid"}


@dataclass(frozen=True)
class ConvexSupportTrace:
    """Numerical signed-gap range in metres and work from this HOST query."""

    phase: str
    gap_lower_m: float
    gap_upper_m: float
    support_calls: int
    iterations: int
    time_cells: int = 0
    static_samples: int = 1
    pose_error_upper_m: float = 0.0


@dataclass
class _SupportVertex:
    point: np.ndarray
    first: np.ndarray
    second: np.ndarray
    first_id: int
    second_id: int


class _Body:
    def __init__(self, shape, path, origin):
        self.kind = shape.kind
        order = np.argsort(shape.vertex_ids)
        self.points = path.start[order] - origin
        self.ids = np.array(shape.vertex_ids)[order]
        self.radius = shape.radius
        self.dims = ()
        if shape.curved is not None:
            self.frame = _rotation(shape.curved, path).astype(np.float64)
            self.dims = shape.curved.dimensions_m
        if shape.kind == "triangle" and not _FacePath(shape, path).valid:
            raise ValueError("degenerate_material_triangle")
        if shape.kind == "hull":
            HullShape(shape, path)  # Enforce the same full-dimensional registration as hull/hull.
        self.extent = float(np.max(np.abs(self.points))) + max((*self.dims, self.radius))

    def support(self, direction):
        if self.kind in _VERTEX_SUPPORT:
            i = int(np.argmax(self.points @ direction))
            return self.points[i] + self.radius * direction, i
        local = self.frame.T @ direction
        if self.kind == "ellipsoid":
            dims = np.asarray(self.dims)
            scaled = dims * local
            point = dims * (scaled / np.linalg.norm(scaled))
        else:
            radius, half_height = self.dims
            radial = np.hypot(*local[:2])
            if self.kind == "cone" and 2 * half_height * local[2] >= radius * radial:
                point = np.array([0.0, 0.0, half_height])
            else:
                xy = np.zeros(2) if radial == 0 else radius * local[:2] / radial
                z = -half_height if self.kind == "cone" else half_height * np.sign(local[2])
                point = np.array([*xy, z])
        return self.points[0] + self.frame @ point, 0

    def material(self, indices, weights, point):
        coefficients = np.zeros(len(self.points))
        np.add.at(coefficients, indices, weights)
        address = tuple(
            (int(key), float(weight))
            for key, weight in zip(self.ids, coefficients, strict=True)
            if weight > 0
        )
        offset = (
            None
            if self.kind in ("triangle", "hull", "particle")
            else tuple(map(float, point - coefficients @ self.points))
        )
        return address, offset


def _closest(vertices):
    """Project the origin onto the simplex, retaining its active barycentric owners."""
    best = None
    for count in range(1, min(4, len(vertices)) + 1):
        for subset in combinations(vertices, count):
            coordinates = np.array([v.point for v in subset])
            if count == 1:
                weights = np.ones(1)
            else:
                edges = (coordinates[1:] - coordinates[0]).T
                values, _, rank, _ = np.linalg.lstsq(edges, -coordinates[0], rcond=None)
                if rank < count - 1:
                    continue
                weights = np.r_[1.0 - sum(values), values]
                if np.min(weights) < -2e-13:
                    continue
                weights = np.maximum(weights, 0.0)
                weights /= sum(weights)
            point = weights @ coordinates
            if count == 3 and np.all(weights > 0):
                # Close to contact, subtractive cancellation in a barycentric sum
                # can destroy the normal. The interior foot is perpendicular to
                # the actual triangle plane, even at vanishing separation.
                normal = np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
                point = normal * ((normal @ coordinates[0]) / (normal @ normal))
            elif count == 4 and np.all(weights > 0):
                point = np.zeros(3)
            distance = float(point @ point)
            if best is None or distance < best[0]:
                best = distance, subset, weights, point
    return best[1:]


@dataclass
class _Face:
    indices: tuple
    normal: np.ndarray
    height: float


class _Polytope:
    def __init__(self, vertices, eps):
        self.vertices = []
        self.faces = []
        self.eps = eps
        points = np.array([v.point for v in vertices])
        i = 0
        j = int(np.argmax(np.linalg.norm(points - points[i], axis=1)))
        edge = points[j] - points[i]
        k = int(np.argmax(np.linalg.norm(np.cross(edge, points - points[i]), axis=1)))
        normal = np.cross(edge, points[k] - points[i])
        length = np.linalg.norm(normal)
        if length == 0:
            raise SolverFailure("support_seed_dimension")
        heights = np.abs((points - points[i]) @ (normal / length))
        m = int(np.argmax(heights))
        if heights[m] <= eps:
            raise SolverFailure("support_seed_dimension")
        initial = (i, j, k, m)
        self.vertices.extend(vertices[n] for n in initial)
        self.inside = np.mean(points[list(initial)], axis=0)
        self.faces = [self._face(*face) for face in combinations(range(4), 3)]
        for index, vertex in enumerate(vertices):
            if index not in initial:
                self.extend(vertex)

    def _face(self, i, j, k):
        a, b, c = (self.vertices[q].point for q in (i, j, k))
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length <= self.eps * max(np.linalg.norm(b - a), np.linalg.norm(c - a)):
            raise SolverFailure("support_degenerate_horizon")
        normal /= length
        if normal @ (a - self.inside) < 0:
            normal = -normal
            j, k = k, j
        return _Face((i, j, k), normal, float(normal @ a))

    def extend(self, vertex):
        horizon, keep = {}, []
        for face in self.faces:
            a, b, c = face.indices
            if face.normal @ (vertex.point - self.vertices[a].point) <= self.eps:
                keep.append(face)
                continue
            for i, j in ((a, b), (b, c), (c, a)):
                if (j, i) in horizon:
                    del horizon[j, i]
                else:
                    horizon[i, j] = True
        if not horizon:
            return False
        index = len(self.vertices)
        self.vertices.append(vertex)
        self.faces = keep + [self._face(i, j, index) for i, j in horizon]
        return True

    def foot(self, face):
        foot = face.height * face.normal
        best = None
        # The supporting plane can be split into several coplanar triangles.
        for candidate in self.faces:
            if abs(candidate.height - face.height) > self.eps * 8 or (
                np.linalg.norm(candidate.normal - face.normal) > 1e-7
            ):
                continue
            tri = [self.vertices[i] for i in candidate.indices]
            vertices, weights, point = _closest(tri)
            error = float(np.linalg.norm(point - foot))
            if best is None or error < best[0]:
                best = (error, vertices, weights)
        if best is None or best[0] > _TOL_M:
            raise SolverFailure("support_witness_not_on_nearest_facet")
        return best[1:]


class _Query:
    def __init__(self, a, b, ma, mb):
        self.origin = ma.start[0]
        self.a, self.b = _Body(a, ma, self.origin), _Body(b, mb, self.origin)
        extent = max(1.0, self.a.extent, self.b.extent)
        self.eps = 128 * np.finfo(float).eps * extent
        if self.eps > _TOL_M / 16:
            raise SolverFailure("support_coordinate_resolution")
        self.calls = 0

    def support(self, direction):
        direction = direction / np.linalg.norm(direction)
        a, i = self.a.support(direction)
        b, j = self.b.support(-direction)
        self.calls += 1
        return _SupportVertex(a - b, a, b, i, j)

    def finish(self, vertices, weights, normal, lo, hi, phase, iterations):
        pa = weights @ np.array([v.first for v in vertices])
        pb = weights @ np.array([v.second for v in vertices])
        wa, oa = self.a.material([v.first_id for v in vertices], weights, pa)
        wb, ob = self.b.material([v.second_id for v in vertices], weights, pb)
        gap = float(normal @ (pb - pa))
        lo, hi = min(gap, lo) - self.eps, max(gap, hi) + self.eps
        points = np.array([self.origin + pa, self.origin + pb])
        if not np.isfinite(points).all() or np.spacing(abs(points)).max() > _TOL_M:
            raise SolverFailure("support_world_witness_resolution")
        status = "separated" if lo > 0 else "overlap" if hi < 0 else "touching"
        witness = CollisionWitness(
            gap,
            tuple(map(float, normal)),
            tuple(map(float, points[0])),
            tuple(map(float, points[1])),
            wa,
            wb,
            "host-convex-support",
            first_offset=oa,
            second_offset=ob,
        )
        return (
            status,
            "bounded_support_estimate",
            witness,
            ConvexSupportTrace(phase, lo, hi, self.calls, iterations),
        )

    def expand(self, simplex, weights, approach, previous):
        directions = [approach, -approach, *np.eye(3), *-np.eye(3)]
        vertices = list(simplex)
        for axis in directions:
            vertex = self.support(axis)
            reach = float(axis @ vertex.point)
            if abs(reach) <= self.eps:
                upper = float(np.linalg.norm(weights @ np.array([v.point for v in simplex])))
                return self.finish(simplex, weights, axis, -reach, upper, "support_plane", previous)
            if all(np.linalg.norm(vertex.point - v.point) > self.eps for v in vertices):
                vertices.append(vertex)
        poly = _Polytope(vertices, self.eps)
        for iteration in range(1, _MAX_EPA + 1):
            face = min(poly.faces, key=lambda f: f.height)
            n, depth = face.normal, face.height
            outer = self.support(n)
            reach = float(n @ outer.point)
            if depth >= -self.eps and reach - depth <= _TOL_M:
                subset, bary = poly.foot(face)
                return self.finish(
                    subset, bary, n, -reach, -depth, "expanding_polytope", previous + iteration
                )
            if reach <= self.eps and depth < 0:
                # A plane supporting the original bodies reaches the origin and
                # the active simplex supplies the matching near-coincident points.
                gap = float(np.linalg.norm(weights @ np.array([v.point for v in simplex])))
                return self.finish(
                    simplex, weights, n, -reach, gap, "support_plane", previous + iteration
                )
            if not poly.extend(outer):
                raise SolverFailure("support_expansion_stalled")
        face = min(poly.faces, key=lambda f: f.height)
        if face.height > self.eps:
            # The inner polytope already contains the origin strictly. Its inradius
            # and the original body's support bound penetration even when convergence
            # to a unique, accurate contact witness is costly.
            reach = float(face.normal @ self.support(face.normal).point)
            trace = ConvexSupportTrace(
                "overlap_bracket",
                -reach - self.eps,
                -face.height + self.eps,
                self.calls,
                previous + _MAX_EPA,
            )
            return "overlap", "support_depth_unresolved", None, trace
        raise SolverFailure("support_expansion_capacity")

    def result(self):
        approach = np.mean(self.b.points, axis=0) - np.mean(self.a.points, axis=0)
        if np.linalg.norm(approach) <= self.eps:
            approach = np.array([1.0, 0.0, 0.0])
        approach /= np.linalg.norm(approach)
        simplex = [self.support(approach)]
        for iteration in range(1, _MAX_GJK + 1):
            vertices, weights, point = _closest(simplex)
            upper = float(np.linalg.norm(point))
            if upper <= self.eps:
                return self.expand(vertices, weights, approach, iteration)
            normal = -point / upper
            vertex = self.support(normal)
            lower = -float(normal @ vertex.point)
            # A supporting plane bounds the signed gap from below even if it
            # crosses zero. The simplex gives an upper bound in either case.
            if upper - lower <= _TOL_M:
                return self.finish(
                    vertices, weights, normal, lower, upper, "simplex_distance", iteration
                )
            if any(np.linalg.norm(vertex.point - v.point) <= self.eps for v in vertices):
                raise SolverFailure("support_distance_stalled")
            simplex = [*vertices, vertex]
        raise SolverFailure("support_distance_capacity")


def _contained_cone_sphere(a, b, ma, mb):
    """Signed distance to the nearer base or lateral surface in the cone's meridian."""
    cone_first = a.kind == "cone"
    cone, ball, mc, ms = (a, b, ma, mb) if cone_first else (b, a, mb, ma)
    r, h = _vector(cone.curved.dimensions_m)
    frame = _rotation(cone.curved, mc)
    cone_center, center = _vector(mc.start[0]), _vector(ms.start[0])
    x, y, z = frame.T @ (center - cone_center)
    if not -h < z < h:
        return None
    radial = sqrt_bounds(x * x + y * y, 96)
    side_length = sqrt_bounds(r * r + 4 * h * h, 96)
    clearance = r * (h - z)
    if clearance <= 2 * h * radial[1]:
        return None
    # For an interior point the perpendicular foot lies on the finite sloping
    # segment, and the vertical projection is in the base disk.
    side_lo = (clearance - 2 * h * radial[1]) / side_length[1]
    side_hi = (clearance - 2 * h * radial[0]) / side_length[0]
    base = h + z
    lo, hi = min(base, side_lo), min(base, side_hi)
    depth = (lo + hi) / 2
    radius = Fraction(ball.radius)
    n = None
    if base < side_lo:
        n = -frame[:, 2]
    elif side_hi < base and radial[0]:
        length = sum(side_length) / 2
        scale = 2 * h / (sum(radial) / 2)
        n = frame @ np.array([scale * x, scale * y, r], dtype=object) / length
    pa = pb = oa = ob = None
    wa, wb = ((cone.vertex_ids[0], 1.0),), ((ball.vertex_ids[0], 1.0),)
    if n is not None:
        cone_offset, ball_offset = center - cone_center + depth * n, -radius * n
        pa, pb = _xyz(cone_center + cone_offset), _xyz(center + ball_offset)
        if np.spacing(np.abs((pa, pb))).max() > _TOL_M:
            raise SolverFailure("support_world_witness_resolution")
        oa, ob = _xyz(cone_offset), _xyz(ball_offset)
    if not cone_first:
        pa, pb, wa, wb, oa, ob = pb, pa, wb, wa, ob, oa
        if n is not None:
            n = -n
    gap = float(-depth - radius)
    if not np.isfinite(gap) or gap >= 0:
        raise SolverFailure("support_coordinate_resolution")
    witness = CollisionWitness(
        gap,
        None if n is None else _xyz(n),
        pa,
        pb,
        wa,
        wb,
        "host-cone-interior",
        first_offset=oa,
        second_offset=ob,
    )
    trace = ConvexSupportTrace(
        "cone_meridian",
        np.nextafter(float(-hi - radius), -np.inf),
        np.nextafter(float(-lo - radius), np.inf),
        0,
        0,
    )
    return "overlap", "cone_interior", witness, trace


def static_convex_support(a, b, ma, mb):
    """Return one bounded-distance observation for two registered, finite convex bodies."""
    if a.kind not in _KINDS or b.kind not in _KINDS or ma.moving or mb.moving:
        return None
    if any(shape.kind == "triangle" and shape.radius for shape in (a, b)):
        return "unsupported", "material_triangle_thickness", None, None
    if any(shape.kind == "hull" and shape.radius for shape in (a, b)):
        return "unsupported", "convex_hull_offset", None, None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if {a.kind, b.kind} == {"cone", "sphere"}:
                interior = _contained_cone_sphere(a, b, ma, mb)
                if interior is not None:
                    return interior
            return _Query(a, b, ma, mb).result()
    except ValueError as error:
        return "invalid_geometry", str(error), None, None
    except (SolverFailure, FloatingPointError, np.linalg.LinAlgError, OverflowError) as error:
        return "unresolved", str(error), None, None
