"""Host FP64 surface maps over immutable, backend-owned point responses."""

import numpy as np

from ..core._arrays import float_array
from ..core.capabilities import SolverFailure
from .point_port import _group_sum
from .ports import PointPort, _numerical


def _array(value, name, shape=None):
    result = float_array(value, name, ndim=2)
    if result.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must use float64 storage")
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return result


def _result(value, shape):
    if not np.isfinite(value).all():
        raise SolverFailure("surface point mapping is not representable in FP64")
    return _array(value, "point result", shape)


class WeightedPointPort:
    """Barycentric world points with full response B R B.T.

    Maps run on the host; the supplied PointPort retains its actual backend.
    The base must satisfy the immutable snapshot contract of PointPort.
    """

    def __init__(self, base: PointPort, weights):
        if base.dtype != np.dtype(np.float64):
            raise TypeError("surface ports require float64 base state storage")
        positions = _array(base.positions(), "base positions")
        if positions.shape[1] != 3:
            raise ValueError("base positions must have three columns")
        weights = _array(weights, "weights")
        if weights.shape[1] != len(positions):
            raise ValueError("weights must have one column per base point")
        with np.errstate(over="ignore", invalid="ignore"):
            sums = np.sum(weights, axis=1)
        if np.any(weights < 0) or not np.all(np.abs(sums - 1) <= 1e-12):
            raise ValueError("weights must be non-negative with each row summing to one")
        weights.flags.writeable = False
        self._weights = weights
        self._base = base
        self._base_shape = positions.shape
        self._shape = (len(weights), 3)

    @property
    def domain(self):
        return self._base.domain

    @property
    def dtype(self):
        return self._base.dtype

    @property
    def weights(self):
        return self._weights.copy()

    def snapshot(self):
        return self._base.snapshot()

    def validate_candidate(self, state):
        return self._base.validate_candidate(state)

    def _map(self, values):
        return _result(self._weights @ _result(values, self._base_shape), self._shape)

    def _loads(self, impulses):
        impulses = _array(impulses, "impulses", self._shape)
        return _result(self._weights.T @ impulses, self._base_shape)

    @_numerical
    def positions(self):
        return self._map(self._base.positions())

    @_numerical
    def velocities(self):
        return self._map(self._base.velocities())

    @_numerical
    def response(self, impulses):
        return self._map(self._base.response(self._loads(impulses)))

    @_numerical
    def apply_impulses(self, impulses):
        return self.validate_candidate(self._base.apply_impulses(self._loads(impulses)))


class PointPairOperator:
    """Relative point response D R D.T within one state domain.

    Each pair receives +p at its first point and -p at its second point.
    All loads are accumulated before requesting one bottom-level candidate.
    Relative coordinates deliberately do not implement the world PointPort API.
    """

    def __init__(self, port: PointPort, pairs):
        if port.dtype != np.dtype(np.float64):
            raise TypeError("point pairs require float64 base state storage")
        positions = _array(port.positions(), "base positions")
        if positions.shape[1] != 3:
            raise ValueError("base positions must have three columns")
        pairs = np.asarray(pairs)
        if pairs.dtype.kind not in "iu":
            raise TypeError("pairs must use integer indices")
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape (k, 2)")
        if np.any(pairs < 0) or np.any(pairs >= len(positions)):
            raise ValueError("pair index outside point port")
        self._pairs = np.array(pairs, dtype=np.int64, copy=True)
        self._pairs.flags.writeable = False
        self._port = port
        self._base_shape = positions.shape
        self._shape = (len(pairs), 3)

    @property
    def domain(self):
        return self._port.domain

    @property
    def dtype(self):
        return self._port.dtype

    @property
    def pairs(self):
        return self._pairs.copy()

    def snapshot(self):
        return self._port.snapshot()

    def validate_candidate(self, state):
        return self._port.validate_candidate(state)

    def _relative(self, values):
        values = _result(values, self._base_shape)
        return _result(values[self._pairs[:, 0]] - values[self._pairs[:, 1]], self._shape)

    def _loads(self, impulses):
        impulses = _array(impulses, "impulses", self._shape)
        # Identical endpoints contribute exactly zero, even for very large loads.
        active = self._pairs[:, 0] != self._pairs[:, 1]
        indices = self._pairs[active].T.reshape(-1)
        values = np.concatenate((impulses[active], -impulses[active]))
        return _result(_group_sum(indices, values, self._base_shape[0]), self._base_shape)

    @_numerical
    def relative_positions(self):
        return self._relative(self._port.positions())

    @_numerical
    def relative_velocities(self):
        return self._relative(self._port.velocities())

    @_numerical
    def response(self, impulses):
        return self._relative(self._port.response(self._loads(impulses)))

    @_numerical
    def apply_impulses(self, impulses):
        return self.validate_candidate(self._port.apply_impulses(self._loads(impulses)))
