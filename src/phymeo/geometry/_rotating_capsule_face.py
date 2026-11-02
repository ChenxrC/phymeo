"""A finite, rotating capsule swept against an independently affine triangle."""

from fractions import Fraction as F
from math import ceil, inf, nextafter

import numpy as np

from ..constraints.collision_shapes import CollisionMotion, CollisionShape
from ..core.capabilities import SolverFailure
from ._bounded_contact import first_bounded_contact
from ._capsule_triangle_sweep import segment_minimum, segment_witness
from ._rotation_angle import axis_length_bounds, rotate_enclosures
from ._round_sweep import _GRID, _translation, _vector
from ._sphere_triangle_sweep import _SphereFace
from ._tetrahedral_surface import _TriangleMotion
from ._triangle_pair_sweep import _FacePath, _Pose
from .collision_witness import CollisionSweep

_MAX_SECTORS = 64


def _length_bound(v):
    return sum(abs(x) for x in v)


def _outward_radius(r):
    value = float(r)
    return nextafter(value, inf) if F(value) < r else value


class RotatingCapsuleFaceQuery:
    def __init__(self, cap, face, mp, mf, face_first):
        self.cap, self.face_first = cap, face_first
        self.face = _FacePath(face, mf)
        turn = mp.turn
        self.angle = F(turn.angle)
        self.axis = _vector(turn.axis)
        self.length = axis_length_bounds(self.axis)
        self.pivot = _vector(turn.pivot_start)
        self.travel = _translation(mp)
        self.ends = tuple(_vector(p) - self.pivot for p in mp.start)
        self.radius = F(cap.radius)
        self.features = ()
        self.sectors = []
        self.issue = None
        count = ceil(abs(self.angle))
        if not self.face.valid:
            self.issue = "triangle_degenerate_on_path"
        elif count > _MAX_SECTORS:
            self.issue = "rotation_face_sector_capacity"
        if self.issue:
            return
        starts = self.face.start - self.pivot
        rates = self.face.step - self.travel
        self.speed = abs(self.angle) * max(map(_length_bound, self.ends)) + max(
            map(_length_bound, rates)
        )
        # Two balls cover the two closed halves of the rotating segment. The
        # quarter-point drifts over a sector, while its enclosing query is fixed
        # in the moving pivot frame. Only proven clear sectors may be discarded.
        edge = self.ends[1] - self.ends[0]
        quarters = tuple(self.ends[0] + s * edge for s in (F(1, 4), F(3, 4)))
        features = []
        for i in range(count):
            lo, hi = F(i, count), F(i + 1, count)
            motion = _TriangleMotion(starts + lo * rates, starts + hi * rates, True)
            centers = rotate_enclosures(
                quarters, self.axis, self.length, self.angle * (lo + hi) / 2
            )
            begin = len(features)
            for original, (center, error) in zip(quarters, centers, strict=True):
                stored = np.asarray(center, float)
                radius = (
                    self.radius
                    + _length_bound(edge) / 4
                    + error
                    + _length_bound(_vector(stored) - center)
                    + abs(self.angle) * (hi - lo) * _length_bound(original) / 2
                )
                ball = CollisionShape(
                    cap.key, "sphere", (cap.vertex_ids[0],), _outward_radius(radius)
                )
                leaf = _SphereFace(face, ball, motion, CollisionMotion([stored]))
                if not leaf.features:
                    self.issue = "rotation_face_outer_geometry_unresolved"
                    return
                features.extend(leaf.features)
            self.sectors.append((lo, hi, begin, len(features)))
        self.features = tuple(features)

    def _geometry(self, t):
        (first, e0), (last, e1) = rotate_enclosures(
            self.ends, self.axis, self.length, self.angle * t
        )
        start = first + self.pivot + t * self.travel
        axis = last - first
        tri = _Pose(self.face, t)
        nearest = segment_minimum(start, axis, tri)
        return start, axis, tri, nearest, max(e0, e1)

    def _distance(self, t):
        _, _, _, nearest, error = self._geometry(t)
        return nearest[0], error

    def sample(self, t):
        start, axis, tri, nearest, _ = self._geometry(t)
        return segment_witness(
            self.cap, self.face_first, start, axis, tri, nearest, "host-rotating-capsule-face"
        )

    def sweep(self, seeds):
        if self.issue:
            raise SolverFailure(self.issue)
        start, initial = self.sample(F(0))
        end, _ = self.sample(F(1))
        if initial <= 0:
            return end, CollisionSweep(start.gap_m, end.gap_m, (0.0, 0.0), 0.0, start)
        intervals = []
        for lo, hi, begin, last in self.sectors:
            entries = (
                f.entry(seed)
                for f, seed in zip(self.features[begin:last], seeds[begin:last], strict=True)
            )
            tick = min((bound[0] for bound in entries if bound is not None), default=None)
            if tick is not None:
                intervals.append((lo + (hi - lo) * F(tick, _GRID), hi))
        bracket = first_bounded_contact(intervals, self._distance, self.speed, self.radius)
        if bracket is None:
            return end, CollisionSweep(start.gap_m, end.gap_m, None, None, None)
        lo, hi = bracket
        # Publish the witnessed occupied sample, even if the earliest event
        # lies slightly earlier in the bounded interval.
        lower = float(lo)
        if F(lower) > lo:
            lower = nextafter(lower, -inf)
        upper = float(hi)
        if F(upper) < hi:
            upper = nextafter(upper, inf)
        sample_time = F(upper)
        point, axis, tri, nearest, error = self._geometry(sample_time)
        if self.radius < error or nearest[0] > (self.radius - error) ** 2:
            raise SolverFailure("rotation_contact_unresolved")
        witness, _ = segment_witness(
            self.cap, self.face_first, point, axis, tri, nearest, "host-rotating-capsule-face"
        )
        return end, CollisionSweep(start.gap_m, end.gap_m, (lower, upper), upper, witness)


def rotating_capsule_face_query(a, b, ma, mb):
    if {a.kind, b.kind} != {"capsule", "triangle"}:
        return None
    face_first = a.kind == "triangle"
    cap, face, mp, mf = (b, a, mb, ma) if face_first else (a, b, ma, mb)
    if face.radius or mp.turn is None or not mp.turn.angle:
        return None
    return RotatingCapsuleFaceQuery(cap, face, mp, mf, face_first)
