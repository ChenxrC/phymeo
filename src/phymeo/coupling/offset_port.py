"""Fixed-configuration material points offset from a triangle's middle surface."""

from math import fsum

import numpy as np

from ..core._arrays import float_array
from ..core.capabilities import SolverFailure
from .candidate_audit import PointCandidateAudit
from .point_port import _group_sum
from .ports import PointPort, _numerical
from .surface_port import _array, _result


def _skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


class TriangleOffsetPointPort:
    """Map a signed face-normal offset using its complete block Jacobian.

    Geometry and maps run in host FP64; the immutable base retains its actual
    execution backend. This is a material attachment at fixed configuration,
    not a closest-point query, thickness side selector or time integrator.
    Rebuild the port whenever the configuration or material attachment changes.
    """

    @_numerical
    def __init__(self, base: PointPort, triangles, weights, offsets):
        if base.dtype != np.dtype(np.float64):
            raise TypeError("offset ports require float64 base state storage")
        positions = _array(base.positions(), "base positions")
        if positions.shape[1] != 3:
            raise ValueError("base positions must have three columns")
        triangles = np.asarray(triangles)
        if triangles.dtype.kind not in "iu":
            raise TypeError("triangles must use integer indices")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("triangles must have shape (k, 3)")
        if np.any(triangles < 0) or np.any(triangles >= len(positions)):
            raise ValueError("triangle index outside base point port")
        triangles = np.array(triangles, dtype=np.int64, copy=True)
        k = len(triangles)
        weights = _array(weights, "weights", (k, 3))
        offsets = float_array(offsets, "offsets", ndim=1)
        if offsets.dtype != np.dtype(np.float64):
            raise TypeError("offsets must use float64 storage")
        if offsets.shape != (k,):
            raise ValueError("offsets must have shape (k,)")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        for row in weights:
            total = fsum(row)
            if abs(total - 1) > 1e-12:
                raise ValueError("weights must sum to one")
            row /= total
            row[0] = 1 - row[1] - row[2]
            if row[0] < 0:
                raise ValueError("normalized weights must remain non-negative")

        points = np.empty((k, 3))
        blocks = np.empty((k, 3, 3, 3))
        eye = np.eye(3)
        for j, (indices, bary, offset) in enumerate(zip(triangles, weights, offsets)):
            nodes = positions[indices]
            edges = nodes[1:] - nodes[0]
            scale = float(np.max(np.abs(edges)))
            if scale == 0:
                raise SolverFailure("degenerate triangle material attachment")
            a, b = edges / scale
            cross = np.cross(a, b)
            area = float(np.linalg.norm(cross))
            if area <= 1e-10:
                raise SolverFailure("degenerate or ill-conditioned triangle material attachment")
            normal = cross / area
            normal_derivative = (offset / scale) * (eye - np.outer(normal, normal)) / area
            cross_derivatives = (_skew(b - a), -_skew(b), _skew(a))
            for i in range(3):
                blocks[j, i] = bary[i] * eye + normal_derivative @ cross_derivatives[i]
            middle = nodes[0] + bary[1] * edges[0] + bary[2] * edges[1]
            displacement = offset * normal
            points[j] = middle + displacement
            # The moment budget scales with the face, so it alone cannot detect
            # a small thickness lost in world-coordinate rounding.
            if offset != 0 and np.max(np.abs((points[j] - middle) - displacement)) > (
                1e-10 * abs(offset)
            ):
                raise SolverFailure("normal offset is not representable at the stored world point")
            if not np.isfinite(blocks[j]).all() or not np.isfinite(points[j]).all():
                raise SolverFailure("offset attachment is not representable")
            translation_error = np.max(np.abs(np.sum(blocks[j], axis=0) - eye))
            moment = _skew(edges[0]) @ blocks[j, 1].T + _skew(edges[1]) @ blocks[j, 2].T
            moment_error = np.max(np.abs(moment - _skew(points[j] - nodes[0])))
            if (
                not np.isfinite(moment_error)
                or translation_error > 1e-10
                or (moment_error > 1e-10 * max(scale, abs(offset)))
            ):
                raise SolverFailure("offset attachment fails actual-point kinematic moment check")

        for array in (triangles, weights, offsets, points, blocks):
            array.flags.writeable = False
        self._base = base
        self._base_shape = positions.shape
        self._shape = (k, 3)
        self._triangles = triangles
        self._weights = weights
        self._offsets = offsets
        self._points = points
        self._blocks = blocks

    @property
    def domain(self):
        return self._base.domain

    @property
    def dtype(self):
        return self._base.dtype

    @property
    def triangles(self):
        return self._triangles.copy()

    @property
    def weights(self):
        return self._weights.copy()

    @property
    def offsets(self):
        return self._offsets.copy()

    @property
    def jacobians(self):
        return self._blocks.copy()

    def snapshot(self):
        return self._base.snapshot()

    def validate_candidate(self, state):
        return self._base.validate_candidate(state)

    def positions(self):
        return self._points.copy()

    def _map(self, values):
        nodes = _result(values, self._base_shape)[self._triangles]
        # Explicit products honor NumPy's underflow/overflow traps; einsum can
        # silently return zero for nonzero subnormal products.
        products = self._blocks * nodes[:, :, None, :]
        return _result(np.sum(products, axis=(1, 3)), self._shape)

    def _loads(self, impulses):
        impulses = _array(impulses, "impulses", self._shape)
        loads = np.sum(self._blocks * impulses[:, None, :, None], axis=2)
        if not np.isfinite(loads).all():
            raise SolverFailure("offset attachment load is not representable")
        return _result(
            _group_sum(self._triangles.reshape(-1), loads.reshape(-1, 3), self._base_shape[0]),
            self._base_shape,
        )

    @_numerical
    def velocities(self):
        return self._map(self._base.velocities())

    @_numerical
    def response(self, impulses):
        return self._map(self._base.response(self._loads(impulses)))

    @_numerical
    def apply_impulses(self, impulses):
        return self.validate_candidate(self._base.apply_impulses(self._loads(impulses)))

    @_numerical
    def candidate_audit(self, state):
        audit = getattr(self._base, "candidate_audit", None)
        if not callable(audit):
            raise SolverFailure("base port lacks candidate audit")
        result = audit(state)
        return PointCandidateAudit(
            self._map(result.initial_velocities),
            self._map(result.candidate_velocities),
            result.energy_change,
        )
