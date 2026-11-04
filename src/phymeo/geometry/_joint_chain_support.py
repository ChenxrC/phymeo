"""Bound convex link vertices along an affine path in arbitrary scalar joint coordinates."""

from fractions import Fraction

import numpy as np

from ..constraints.collision_shapes import CollisionMotion
from ..core.capabilities import SolverFailure
from ._curved_solids import _quaternion_frame
from ._prism_pair_exact import sqrt_bounds
from ._rotation_angle import axis_length_bounds, rotate_enclosures
from ._round_sweep import _vector

_SCALE = 1 << 80


class JointChainSupport:
    """Compose actual joint transforms tip-to-root and bound displacement on each time cell."""

    def __init__(self, model, chain, shape, local_vertices, q0, q1):
        if not 1 <= len(chain) <= 64 or len(local_vertices) > 64:
            raise SolverFailure("compound tool path exceeds its joint or vertex budget")
        self.shape = shape
        self.vertices = tuple(_vector(row) for row in local_vertices)
        reach = max(sum(abs(x) for x in point) for point in self.vertices)
        self.speed = Fraction(0)
        self.joints = []
        self.cache = {}
        for j in reversed(chain):
            start = Fraction(float(q0[j]))
            end = Fraction(float(q1[j]))
            delta = end - start
            kind = model.joint_types[j]
            offset = _vector(model.joint_position[j])
            axis = _vector(model.joint_axis[j])
            length = axis_length_bounds(axis)
            rotation = _quaternion_frame(model.joint_orientation[j])
            if kind == "revolute":
                if abs(delta) > float(2 * np.pi):
                    raise SolverFailure("compound joint arc exceeds one turn per checked segment")
                self.speed += abs(delta) * reach
            else:
                self.speed += abs(delta)
                reach += max(abs(start), abs(end))
            reach += sum(abs(x) for x in offset)
            self.joints.append((kind, start, delta, offset, axis, length, rotation))

    def _points(self, t):
        if t in self.cache:
            return self.cache[t]
        points = self.vertices
        error = Fraction(0)
        for kind, start, delta, offset, axis, length, rotation in self.joints:
            q = start + delta * t
            if kind == "revolute":
                rotated = rotate_enclosures(points, axis, length, q)
                error += max(e for _, e in rotated)
                moved = (v for v, _ in rotated)
            else:
                low, high = length
                shift = q * axis / high
                error += abs(q) * (1 - low / high)
                moved = (p + shift for p in points)
            values = [rotation @ p + offset for p in moved]
            # Bound bit growth after each orthogonal frame change. The L1
            # rounding error bounds Euclidean position error at the next joint.
            points = tuple(
                np.array([Fraction(round(x * _SCALE), _SCALE) for x in v], dtype=object)
                for v in values
            )
            error += max(
                sum(abs(x - y) for x, y in zip(old, new, strict=True))
                for old, new in zip(values, points, strict=True)
            )
        self.cache[t] = np.array(points, dtype=object), error
        return self.cache[t]

    def pose(self, t):
        points, error = self._points(t)
        world = points.astype(float)
        stored = np.array([_vector(row) for row in world], dtype=object)
        conversion = max(sum(abs(x) for x in row) for row in stored - points)
        return CollisionMotion(world), error + conversion

    def interval_upper(self, direction, lo, hi):
        # v = sum_j omega_j x (p - pivot_j) + sum_k slide_speed_k axis_k.
        # The radius at each pivot includes all downstream moving offsets.
        n = _vector(direction)
        mid = (lo + hi) / 2
        points, error = self._points(mid)
        height = max(points @ n)
        bound = height + sqrt_bounds(n @ n, 96)[1] * (error + self.speed * (hi - lo) / 2)
        return bound, bound
