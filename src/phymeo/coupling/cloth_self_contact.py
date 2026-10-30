"""One-base self-contact attachments and immutable nodal-request audit receipts."""

from dataclasses import dataclass, field
from math import fsum

import numpy as np

from ..core.capabilities import SolverFailure
from .domain_graph import MultiDomainPointOperator
from .offset_port import TriangleOffsetPointPort
from .ports import PointPort


def _indices(value, name, columns):
    result = np.asarray(value)
    if result.dtype.kind not in "iu":
        raise TypeError(f"{name} must use integer indices")
    if result.ndim != 2 or result.shape[1] != columns:
        raise ValueError(f"{name} must have shape (k, {columns})")
    if np.any(result < 0) or np.any(result > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must contain non-negative int64 indices")
    return np.array(result, dtype=np.int64, copy=True, order="C")


def _read_only_copy(value):
    result = value.copy(order="C")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, eq=False, init=False)
class SelfContactBinding:
    """Fixed-configuration attachment receipt over one physical PointPort.

    All pair endpoints name rows of one TriangleOffsetPointPort. The existing
    graph sums signed attachment loads before its single base application; this
    receipt has no second response, apply method, or state owner. Rebuild after
    geometry or material attachments change. Pair order and orientation are
    preserved; an impulse is +p at its first row and -p at its second row.

    Exposed arrays are independent read-only copies, including the actual
    weights accepted/normalized by the public offset port. node_impulses returns
    requested loads in the supplied base point-row order, not a claim of actual
    application or of collision, momentum, energy, or full-step acceptance.
    """

    operator: MultiDomainPointOperator
    _triangles: np.ndarray = field(repr=False)
    _weights: np.ndarray = field(repr=False)
    _offsets: np.ndarray = field(repr=False)
    _pairs: np.ndarray = field(repr=False)
    _world_points: np.ndarray = field(repr=False)
    _jacobians: np.ndarray = field(repr=False)
    _node_count: int = field(repr=False)
    _domain: str = field(repr=False)

    def __init__(self, base: PointPort, triangles, weights, offsets, pairs):
        faces = _indices(triangles, "triangles", 3)
        edges = _indices(pairs, "pairs", 2)
        if np.any(edges >= len(faces)):
            raise ValueError("pair index outside attachment rows")
        positions = np.asarray(base.positions())
        if positions.dtype != np.dtype(np.float64):
            raise TypeError("base positions must use float64 storage")
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("base positions must have shape (n, 3)")
        if not np.isfinite(positions).all():
            raise ValueError("base positions must contain only finite values")
        # The public offset port validates geometry, weights, offsets and dtype.
        # Every attachment is wrapped together, including repeated face rows.
        port = TriangleOffsetPointPort(base, faces, weights, offsets)
        domain = base.domain
        operator = MultiDomainPointOperator(
            {domain: port},
            [(domain, int(left)) for left in edges[:, 0]],
            [(domain, int(right)) for right in edges[:, 1]],
        )
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "_node_count", len(positions))
        object.__setattr__(self, "_domain", domain)
        for name, value in (
            ("triangles", port.triangles),
            ("weights", port.weights),
            ("offsets", port.offsets),
            ("pairs", edges),
            ("world_points", port.positions()),
            ("jacobians", port.jacobians),
        ):
            object.__setattr__(self, f"_{name}", _read_only_copy(value))

    @property
    def triangles(self):
        return _read_only_copy(self._triangles)

    @property
    def weights(self):
        return _read_only_copy(self._weights)

    @property
    def offsets(self):
        return _read_only_copy(self._offsets)

    @property
    def pairs(self):
        return _read_only_copy(self._pairs)

    @property
    def world_points(self):
        return _read_only_copy(self._world_points)

    @property
    def jacobians(self):
        return _read_only_copy(self._jacobians)

    def node_impulses(self, impulses):
        """Return owned read-only FP64 (n,3) aggregate requested impulses [N*s].

        This uses the public graph's signed attachment aggregation and the
        recorded J.T blocks. It does not call base.response or apply_impulses.
        Shared-node cancellation and off-diagonal response are retained by the
        actual operator, which remains the only application route.
        """
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
                attachments = self.operator.domain_impulses(impulses)[self._domain]
                # Same block products/reduction as the public offset map. Sum
                # signed row requests before multiplication, including exact
                # cancellation of identical endpoints with very large impulses.
                products = self._jacobians * attachments[:, None, :, None]
                blocks = np.sum(products, axis=2)
                if not np.isfinite(blocks).all():
                    raise FloatingPointError("non-finite attachment request")
                nodes = self._triangles.reshape(-1)
                requests = blocks.reshape(-1, 3)
                total = np.zeros((self._node_count, 3), dtype=np.float64)
                for node in np.unique(nodes):
                    rows = requests[nodes == node]
                    total[node] = [fsum(rows[:, axis]) for axis in range(3)]
                if not np.isfinite(total).all():
                    raise FloatingPointError("non-finite aggregate nodal request")
        except (FloatingPointError, OverflowError) as error:
            raise SolverFailure(
                f"self-contact nodal request is not representable: {error}"
            ) from error
        total.flags.writeable = False
        return total


def make_self_contact_operator(base: PointPort, triangles, weights, offsets, pairs):
    """Bind all attachments once to a same-domain MultiDomainPointOperator.

    Attachment triangles index the supplied base's public point rows. pairs
    index the attachment rows, not topology faces. Repeated attachments and
    reversed/identical pair endpoints retain the existing graph semantics.
    This factory performs no narrow phase, thickness side selection, friction,
    CCD, state advancement, or physics acceptance.
    """
    return SelfContactBinding(base, triangles, weights, offsets, pairs)
