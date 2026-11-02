"""Material coordinates inside a triangle's represented finite thickness."""

import numpy as np

from ..core.capabilities import SolverFailure
from .offset_port import TriangleOffsetPointPort


class TriangleColumnMap:
    """Interpolate material points and their full Jacobians between verified shell endpoints.

    The signed fraction is a material address. Keep it fixed when computing
    velocity and transposed load at this pose, even if the next query selects
    a different closest address.
    """

    def __init__(self, base, triangles, weights, half_thickness):
        rho = np.asarray(half_thickness)
        if np.any(rho < 0):
            raise ValueError("column half thickness cannot be negative")
        minus = TriangleOffsetPointPort(base, triangles, weights, -rho)
        plus = TriangleOffsetPointPort(base, triangles, weights, rho)
        ids = plus.triangles
        w = plus.weights
        self._rho = plus.offsets
        nodes = base.positions()[ids]
        self._velocity = base.velocities()[ids]
        self._arms = nodes - nodes[:, 0:1, :]
        self._origin = nodes[:, 0]
        self._middle = (
            self._origin + w[:, 1, None] * self._arms[:, 1] + w[:, 2, None] * self._arms[:, 2]
        )
        self._bary = w[:, :, None, None] * np.eye(3)
        self._x0, self._x1 = minus.positions(), plus.positions()
        self._j0, self._j1 = minus.jacobians, plus.jacobians
        self._dx = (self._x1 - self._x0) * 0.5
        self._dj = (self._j1 - self._j0) * 0.5
        self._scale = np.maximum(np.max(abs(self._arms), axis=(1, 2)), self._rho)

    def sample(self, fraction):
        a = np.asarray(fraction, dtype=float)
        if a.shape != self._rho.shape or not np.isfinite(a).all() or np.any(abs(a) > 1):
            raise ValueError("material column fractions must be in [-1,1]")
        # Centered form retains a small off-diagonal director term even if
        # (1+a)/2 rounds to exactly 1/2. The two endpoints are the original ports.
        displacement = a[:, None] * self._dx
        points = self._middle + displacement
        jac = self._bary + a[:, None, None, None] * self._dj
        for value, x, j in ((-1.0, self._x0, self._j0), (1.0, self._x1, self._j1)):
            at = a == value
            points[at] = x[at]
            jac[at] = j[at]
        local_error = np.max(abs((points - self._middle) - displacement), axis=1)
        # Check the published position against the represented full column length.
        if np.any(local_error > 1e-10 * self._rho):
            raise SolverFailure("material interpolation is unresolved at the stored world point")
        translation = np.max(abs(jac.sum(axis=1) - np.eye(3)), axis=(1, 2))
        moment = np.cross(self._arms[:, :, None, :], jac).sum(axis=1)
        expected = np.cross((points - self._origin)[:, None, :], np.eye(3)[None, :, :])
        moment_error = np.max(abs(moment - expected), axis=(1, 2))
        velocity = np.sum(jac * self._velocity[:, :, None, :], axis=(1, 3))
        if (
            not np.isfinite(points).all()
            or not np.isfinite(jac).all()
            or not np.isfinite(velocity).all()
            or not np.isfinite(moment_error).all()
            or np.any(translation > 1e-10)
            or np.any(moment_error > 1e-10 * self._scale)
        ):
            raise SolverFailure("material column fails the actual-point kinematic moment check")
        return points, velocity, jac
