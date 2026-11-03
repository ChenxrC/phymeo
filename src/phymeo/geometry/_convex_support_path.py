"""Swept support planes for translating convex bodies and affine material triangles."""

from dataclasses import replace
from fractions import Fraction
from typing import NamedTuple

import numpy as np

from ..constraints.collision_shapes import CollisionMotion
from ..core.capabilities import SolverFailure
from ..dynamics.rotation import advance_world_orientation
from ._convex_support import _KINDS, ConvexSupportTrace, static_convex_support
from ._curved_solids import _rotation, _support_upper
from ._prism_pair_exact import sqrt_bounds
from ._rotation_angle import axis_length_bounds, rotate_enclosures
from ._round_sweep import _translation, _vector
from ._triangle_pair_sweep import _FacePath
from .collision_witness import CollisionSweep, CollisionWitness

_MAX_CELLS = 256
_DEPTH = 18


class _AffineSupport:
    def __init__(self, shape, motion):
        self.shape = shape
        self.body_orientation = motion.body_orientation_xyzw
        self.start = np.array([_vector(row) for row in motion.start], dtype=object)
        if motion.end is None:
            self.delta = np.broadcast_to(_translation(motion), self.start.shape)
        else:
            self.delta = np.array([_vector(row) for row in motion.end], dtype=object) - self.start
        self.axes = None if shape.curved is None else _rotation(shape.curved, motion)
        self.dims = None if shape.curved is None else _vector(shape.curved.dimensions_m)
        self.cache = {}

    def pose(self, t):
        exact = self.start + t * self.delta
        points = exact.astype(float)
        errors = np.array([_vector(row) for row in points], dtype=object) - exact
        error = max(sum(abs(x) for x in row) for row in errors)
        return CollisionMotion(points, body_orientation_xyzw=self.body_orientation), error

    def upper(self, direction, t):
        # The maximum of affine vertex projections is convex in time.
        # Store start and velocity per direction rather than a sampled interval.
        rows = self.cache.get(direction)
        if rows is None:
            n = _vector(direction)
            offset = Fraction(0)
            if self.axes is not None:
                offset = _support_upper(self.shape.kind, self.dims, self.axes.T @ n)
            elif self.shape.radius:
                offset = Fraction(self.shape.radius) * sqrt_bounds(n @ n, 96)[1]
            rows = self.start @ n + offset, self.delta @ n
            self.cache[direction] = rows
        starts, speeds = rows
        return max(starts + t * speeds)

    def interval_upper(self, direction, lo, hi):
        return self.upper(direction, lo), self.upper(direction, hi)


class _RigidSupport(_AffineSupport):
    """A finite body rotating about an independently translating world pivot."""

    def __init__(self, shape, motion):
        super().__init__(shape, motion)
        self.turn = motion.turn
        self.angle = Fraction(self.turn.angle)
        self.pivot = _vector(self.turn.pivot_start)
        self.travel = _vector(self.turn.pivot_end) - self.pivot
        self.relative = self.start - self.pivot
        self.axis = _vector(self.turn.axis)
        self.axis_length = axis_length_bounds(self.axis)
        self.extent = Fraction(shape.radius)
        if self.dims is not None:
            self.extent = (
                max(self.dims)
                if shape.kind == "ellipsoid"
                else sqrt_bounds(self.dims @ self.dims, 96)[1]
            )
        self.reach = max(sum(abs(x) for x in row) for row in self.relative) + self.extent
        self.rotation_cache = {}

    def _rotated(self, t):
        if t not in self.rotation_cache:
            items = rotate_enclosures(self.relative, self.axis, self.axis_length, t * self.angle)
            points = np.array([v for v, _ in items], dtype=object)
            error = max(e for _, e in items)
            axes = None
            if self.axes is not None:
                columns = rotate_enclosures(
                    self.axes.T, self.axis, self.axis_length, t * self.angle
                )
                axes = np.column_stack([v for v, _ in columns])
                # Every material point is inside the local ball of radius extent.
                error += self.extent * sum(e for _, e in columns)
            self.rotation_cache[t] = points, axes, error
        return self.rotation_cache[t]

    def pose(self, t):
        if not t:
            return super().pose(t)
        relative, axes, error = self._rotated(t)
        exact = relative + self.pivot + t * self.travel
        points = exact.astype(float)
        rounding = np.array([_vector(row) for row in points], dtype=object) - exact
        error += max(sum(abs(x) for x in row) for row in rounding)
        orientation = None
        if axes is not None:
            entry = (0.0, 0.0, 0.0, 1.0) if self.body_orientation is None else self.body_orientation
            spin = np.asarray(self.turn.axis) * self.turn.angle / np.linalg.norm(self.turn.axis)
            orientation = advance_world_orientation(np.array([entry]), spin[None, :], float(t))[0]
        pose = CollisionMotion(points, body_orientation_xyzw=orientation)
        if axes is not None:
            # Compare the actual stored quaternion's frame with the bounded Rodrigues
            # frame before using the static solver's signed gap at this sample.
            difference = _rotation(self.shape.curved, pose) - axes
            error += self.extent * sum(abs(x) for x in difference.flat)
        return pose, error

    def interval_upper(self, direction, lo, hi):
        mid = (lo + hi) / 2
        relative, axes, error = self._rotated(mid)
        n = _vector(direction)
        norm = sqrt_bounds(n @ n, 96)[1]
        height = max(relative @ n)
        if axes is not None:
            height += _support_upper(self.shape.kind, self.dims, axes.T @ n)
        else:
            height += Fraction(self.shape.radius) * norm
        # Arc length bounds every point's displacement from the midpoint orientation.
        radius = self.reach * min(Fraction(2), abs(self.angle) * (hi - lo) / 2)
        support = height + (radius + error) * norm + self.pivot @ n
        speed = self.travel @ n
        return support + lo * speed, support + hi * speed


_MAX_POSE_ERROR = Fraction(1, 5_000_000)


class _Sample(NamedTuple):
    status: str
    reason: str
    witness: CollisionWitness | None
    trace: ConvexSupportTrace | None
    pose_error: Fraction


def _end_trace(sample, peak_error, cells, count):
    trace = sample.trace
    lower, upper = trace.gap_lower_m, trace.gap_upper_m
    if sample.pose_error:
        lower = np.nextafter(float(Fraction(lower) - sample.pose_error), -np.inf)
        upper = np.nextafter(float(Fraction(upper) + sample.pose_error), np.inf)
    return replace(
        trace,
        gap_lower_m=lower,
        gap_upper_m=upper,
        time_cells=cells,
        static_samples=count,
        pose_error_upper_m=np.nextafter(float(peak_error), np.inf) if peak_error else 0.0,
    )


class _Window:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.observed = {}
        self.cells = 0

    def sample(self, t):
        if t not in self.observed:
            pa, ea = self.a.pose(t)
            pb, eb = self.b.pose(t)
            status, reason, witness, trace = static_convex_support(
                self.a.shape, self.b.shape, pa, pb
            )
            self.observed[t] = _Sample(status, reason, witness, trace, ea + eb)
        return self.observed[t]

    def clear_on(self, lo, hi, directions):
        for direction in directions:
            if direction is None:
                continue
            forward = tuple(float(x) for x in direction)
            backward = tuple(-x for x in forward)
            a = self.a.interval_upper(forward, lo, hi)
            b = self.b.interval_upper(backward, lo, hi)
            if all(x + y < 0 for x, y in zip(a, b, strict=True)):
                return True
        return False

    def run(self):
        left, right = Fraction(0), Fraction(1)
        start, end = self.sample(left), self.sample(right)
        for sample in (start, end):
            if sample.status == "invalid_geometry":
                return sample.status, sample.reason, None, None, None
        if start.witness is None or end.witness is None:
            return "unresolved", "support_path_endpoint_witness", end.witness, end.trace, None
        w0, w1 = start.witness, end.witness
        axes = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

        def finish(status, reason, interval=None, t=None):
            error = max(s.pose_error for s in self.observed.values())
            trace = _end_trace(end, error, self.cells, len(self.observed))
            if error > _MAX_POSE_ERROR:
                return "unresolved", "support_sample_pose_resolution", w1, trace, None
            if status == "unresolved":
                return status, reason, w1, trace, None
            entry = None if t is None else self.sample(t).witness
            sweep = CollisionSweep(
                w0.gap_m, w1.gap_m, interval, None if t is None else float(t), entry
            )
            return status, reason, w1, trace, sweep

        def inside(sample):
            # A strictly negative upper estimate must dominate rounding the affine pose.
            trace = sample.trace
            return trace is not None and trace.gap_upper_m < -sample.pose_error

        if inside(start):
            return finish("hit", "support_path_start_overlapped", (0.0, 0.0), left)
        pending = [(left, right, 0)]
        earliest = None
        while pending:
            lo, hi, depth = pending.pop()
            self.cells += 1
            if self.cells > _MAX_CELLS:
                return finish("unresolved", "support_path_cell_capacity")
            sl, sr = self.sample(lo), self.sample(hi)
            proposals = [o.witness.normal for o in (sl, sr) if o.witness is not None]
            if self.clear_on(lo, hi, (*proposals, *axes)):
                continue
            if depth == _DEPTH:
                earliest = lo if earliest is None else earliest
                if inside(sr) and sr.witness is not None:
                    if hi - earliest > Fraction(1, 1024):
                        return finish("unresolved", "support_path_first_event_ambiguous")
                    return finish("hit", "bounded_support_path", (float(earliest), float(hi)), hi)
                continue
            mid = (lo + hi) / 2
            sm = self.sample(mid)
            if sm.witness and self.clear_on(lo, hi, (sm.witness.normal,)):
                continue
            pending.extend(((mid, hi, depth + 1), (lo, mid, depth + 1)))
        if earliest is not None:
            return finish("unresolved", "support_path_grazing_or_unresolved")
        return finish("separated", "swept_support_planes")


def _turn_clear(window):
    bodies = window.a, window.b
    allowance = sum(
        body.reach * min(Fraction(2), abs(body.angle))
        for body in bodies
        if isinstance(body, _RigidSupport)
    )
    start, end = window.sample(Fraction(0)), window.sample(Fraction(1))
    if not all(s.status == "separated" and s.witness is not None for s in (start, end)):
        return None
    error = max(start.pose_error, end.pose_error)
    if error > _MAX_POSE_ERROR:
        return None
    for proposed in (start.witness.normal, end.witness.normal, *np.eye(3), *-np.eye(3)):
        direction = tuple(map(float, proposed))
        reverse = tuple(-x for x in direction)
        n = _vector(direction)
        pad = allowance * sqrt_bounds(n @ n, 96)[1]
        if all(
            bodies[0].upper(direction, t) + bodies[1].upper(reverse, t) + pad < 0 for t in (0, 1)
        ):
            sweep = CollisionSweep(start.witness.gap_m, end.witness.gap_m, None, None, None)
            return (
                "separated",
                "rotational_support_clearance",
                end.witness,
                _end_trace(end, error, 1, 2),
                sweep,
            )
    return None


def affine_convex_contact(a, b, ma, mb):
    """Return a full positive clearance cover or a bounded first-overlap observation."""
    if not (ma.moving or mb.moving) or any(s.kind not in _KINDS for s in (a, b)):
        return None
    for shape, motion in ((a, ma), (b, mb)):
        if shape.kind in ("triangle", "hull") and shape.radius:
            return "unsupported", "support_path_material_offset", None, None, None
        if shape.kind == "triangle" and not _FacePath(shape, motion).valid:
            return "invalid_geometry", "degenerate_material_triangle_path", None, None, None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if any(m.turn and abs(m.turn.angle) > 2 * np.pi for m in (ma, mb)):
                return None
            bodies = [
                (_RigidSupport if m.turn and m.turn.angle else _AffineSupport)(s, m)
                for s, m in ((a, ma), (b, mb))
            ]
            window = _Window(*bodies)
            if any(isinstance(body, _RigidSupport) for body in bodies):
                clear = _turn_clear(window)
                if clear is not None:
                    return clear
            return window.run()
    except (SolverFailure, FloatingPointError, OverflowError):
        return "unresolved", "support_path_coordinate_resolution", None, None, None
