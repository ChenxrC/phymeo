"""Rational FK enclosures; no assumptions about libm rounding."""

from fractions import Fraction as F
from functools import lru_cache
from math import factorial

import numpy as np

from ..constraints.robot_finite_patch_collider import _cross, _dot
from ..core.robot import RobotModel
from ..solvers._contact_replay_audit import sqrt_bounds

ZERO = F(0)


def interval(x):
    # Exact internal path samples are Fractions, not new FP64 endpoints.
    # Rounding them through float can invalidate both primal upper and dual
    # lower bounds. Stored floating inputs still denote their exact FP value.
    value = x if isinstance(x, F) else F(float(x))
    return (value, value)


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def neg(a):
    return -a[1], -a[0]


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    # Optimize only the immutable rational endpoint representation. Legacy
    # numeric types, iterable shapes and their errors retain the original path.
    if (
        type(a) is tuple
        and type(b) is tuple
        and len(a) == len(b) == 2
        and all(type(x) is F for x in a + b)
    ):
        if a[0] == a[1]:
            if b[0] == b[1]:
                value = a[0] * b[0]
                return value, value
            values = (a[0] * b[0], a[0] * b[1])
            return min(values), max(values)
        if b[0] == b[1]:
            values = (a[0] * b[0], a[1] * b[0])
            return min(values), max(values)
    values = [x * y for x in a for y in b]
    return min(values), max(values)


def scale(a, b):
    return mul(a, (b, b))


def divide(a, b):
    if b[0] <= 0 <= b[1]:
        raise ValueError("interval division through zero")
    return mul(a, (1 / b[1], 1 / b[0]))


def idot(a, b):
    out = (ZERO, ZERO)
    for x, y in zip(a, b, strict=True):
        out = add(out, mul(x, y))
    return out


def matvec(a, b):
    return tuple(idot(row, b) for row in a)


def matmul(a, b):
    return tuple(tuple(idot(row, col) for col in zip(*b, strict=True)) for row in a)


def exact_frame(q):
    x, y, z, w = map(lambda v: F(float(v)), q)
    n = x * x + y * y + z * z + w * w
    return tuple(
        tuple((v, v) for v in row)
        for row in (
            (1 - 2 * (y * y + z * z) / n, 2 * (x * y - z * w) / n, 2 * (x * z + y * w) / n),
            (2 * (x * y + z * w) / n, 1 - 2 * (x * x + z * z) / n, 2 * (y * z - x * w) / n),
            (2 * (x * z - y * w) / n, 2 * (y * z + x * w) / n, 1 - 2 * (x * x + y * y) / n),
        )
    )


def validate_domain(model):
    if not isinstance(model, RobotModel) or model.mass.dtype != np.float64:
        raise TypeError("finite FK requires FP64 RobotModel")
    if all(k == "prismatic" for k in model.joint_types):
        return "prismatic"
    if (
        model.dof_count == 1
        and model.joint_types == ("revolute",)
        and model.parents[0] == -1
        and np.all(model.com == 0)
        and np.count_nonzero(model.joint_axis[0]) == 1
        and np.max(np.abs(model.joint_axis[0])) == 1
    ):
        return "rotor"
    raise ValueError(
        "finite FK supports all-prismatic trees or a single COM0 coordinate-axis rotor"
    )


@lru_cache(maxsize=8192)
def trig(angle):
    if abs(angle) > 4:
        raise ValueError("rotor q must remain in [-4,4]")
    # Degree 159/158 Taylor polynomials, each with its own explicit
    # Lagrange remainder. Every operation is rational.
    sine = sum(((-1) ** k * angle ** (2 * k + 1) / factorial(2 * k + 1) for k in range(80)), ZERO)
    cosine = sum(((-1) ** k * angle ** (2 * k) / factorial(2 * k) for k in range(80)), ZERO)
    sr = abs(angle) ** 160 / factorial(160)
    cr = abs(angle) ** 159 / factorial(159)
    return (sine - sr, sine + sr), (cosine - cr, cosine + cr)


class PatchCurve:
    def __init__(self, model, q0, q1, collider):
        self.kind = validate_domain(model)
        self.model, self.collider = model, collider
        self.q0 = tuple(F(float(x)) for x in q0)
        self.q1 = tuple(F(float(x)) for x in q1)
        self.delta = tuple(b - a for a, b in zip(self.q0, self.q1, strict=True))
        if self.kind == "rotor":
            if any(abs(q) > 4 for q in self.q0 + self.q1):
                raise ValueError("rotor q must remain in [-4,4]")
            frame = exact_frame(model.joint_orientation[0])
            frame = tuple(tuple(v[0] for v in row) for row in frame)
            axis = tuple(F(float(x)) for x in model.joint_axis[0])
            self.rotor = []
            vertices = tuple(tuple(F(float(x)) for x in row) for row in collider.local_vertices)
            normal = _cross(
                tuple(b - a for a, b in zip(vertices[0], vertices[1], strict=True)),
                tuple(b - a for a, b in zip(vertices[0], vertices[2], strict=True)),
            )
            axial = tuple(_dot(axis, normal) * a for a in axis)
            radial = tuple(x - y for x, y in zip(normal, axial, strict=True))
            tangent = _cross(axis, normal)
            self.plane = (
                tuple(_dot(row, axial) for row in frame),
                tuple(_dot(row, radial) for row in frame),
                tuple(_dot(row, tangent) for row in frame),
                _dot(normal, vertices[0]),
                sqrt_bounds(_dot(normal, normal))[1],
            )
            for raw in collider.local_vertices:
                v = tuple(F(float(x)) for x in raw)
                axial = tuple(_dot(axis, v) * a for a in axis)
                radial = tuple(x - y for x, y in zip(v, axial, strict=True))
                tangent = _cross(axis, v)
                c = tuple(
                    F(float(model.joint_position[0, k])) + _dot(frame[k], axial) for k in range(3)
                )
                a = tuple(_dot(row, radial) for row in frame)
                b = tuple(_dot(row, tangent) for row in frame)
                self.rotor.append((c, a, b))
        else:
            left = self._prismatic(self.q0)
            self.ends = (left, left if self.q0 == self.q1 else self._prismatic(self.q1))

    def _prismatic(self, configuration):
        model = self.model
        positions = []
        frames = []
        identity = tuple(tuple(interval(i == j) for j in range(3)) for i in range(3))
        for i, parent in enumerate(model.parents):
            frame0 = identity if parent == -1 else frames[parent]
            p0 = ((ZERO, ZERO),) * 3 if parent == -1 else positions[parent]
            frame = matmul(frame0, exact_frame(model.joint_orientation[i]))
            local = tuple(F(float(x)) for x in model.joint_axis[i])
            norm = sqrt_bounds(_dot(local, local))
            axis = matvec(frame, tuple(divide((x, x), norm) for x in local))
            off = matvec(frame0, tuple(interval(x) for x in model.joint_position[i]))
            positions.append(
                tuple(
                    add(add(p, o), scale(a, configuration[i]))
                    for p, o, a in zip(p0, off, axis, strict=True)
                )
            )
            frames.append(frame)
        i = self.collider.link_index
        return tuple(
            tuple(
                add(p, v)
                for p, v in zip(
                    positions[i], matvec(frames[i], tuple(interval(x) for x in row)), strict=True
                )
            )
            for row in self.collider.local_vertices
        )

    @lru_cache(maxsize=8192)
    def at(self, t):
        if self.kind == "rotor":
            sine, cosine = trig(self.q0[0] + self.delta[0] * t)
            return tuple(
                tuple(
                    add((c[k], c[k]), add(scale(cosine, a[k]), scale(sine, b[k]))) for k in range(3)
                )
                for c, a, b in self.rotor
            )
        return tuple(
            tuple(add(scale(a, 1 - t), scale(b, t)) for a, b in zip(v, w, strict=True))
            for v, w in zip(*self.ends, strict=True)
        )

    def projected_curvature(self, d):
        if self.kind != "rotor":
            return (ZERO,) * len(self.collider.local_vertices)
        return tuple(
            self.delta[0] ** 2 * sqrt_bounds(_dot(d, a) ** 2 + _dot(d, b) ** 2)[1]
            for _, a, b in self.rotor
        )

    def plane_lower(self, x0, x1, start, end):
        """Moving-plane signed distance, a lower bound to any finite subset.

        f(t)=(a0+a1*t)cos(q(t))+(b0+b1*t)sin(q(t))+c0+c1*t.
        Endpoint Taylor intervals plus an explicit bound on |f''| certify
        the whole interval. No finite projection or face classification is
        assumed; the fixed-direction proof remains available at edges.
        """
        if self.kind != "rotor":
            return ZERO
        nc, na, nb, offset, norm = self.plane
        origin = tuple(F(float(x)) for x in self.model.joint_position[0])
        r0 = tuple(x - y for x, y in zip(x0, origin, strict=True))
        rd = tuple(b - a for a, b in zip(x0, x1, strict=True))
        a0, a1, b0, b1, c0, c1 = (
            _dot(na, r0),
            _dot(na, rd),
            _dot(nb, r0),
            _dot(nb, rd),
            _dot(nc, r0) - offset,
            _dot(nc, rd),
        )

        def value(t):
            sine, cosine = trig(self.q0[0] + self.delta[0] * t)
            return add(
                (c0 + c1 * t, c0 + c1 * t),
                add(scale(cosine, a0 + a1 * t), scale(sine, b0 + b1 * t)),
            )

        left, right = value(start), value(end)
        amplitude = max(
            sqrt_bounds((a0 + a1 * t) ** 2 + (b0 + b1 * t) ** 2)[1] for t in (start, end)
        )
        derivative = sqrt_bounds(a1 * a1 + b1 * b1)[1]
        curvature = self.delta[0] ** 2 * amplitude + 2 * abs(self.delta[0]) * derivative
        loss = curvature * (end - start) ** 2 / 8
        positive = min(left[0], right[0]) - loss
        negative = min(-left[1], -right[1]) - loss
        return max(ZERO, positive, negative) / norm
