"""Owned oriented triangle with positive geometric half thickness."""

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from ..core.scene import _name


@dataclass(frozen=True)
class TrianglePrismSurface:
    key: str
    domain: str
    node_indices: tuple
    half_thickness_m: float = field(kw_only=True)

    def __post_init__(self):
        for label, value in (("surface key", self.key), ("surface domain", self.domain)):
            if type(value) is not str:
                raise TypeError(label + " must be a builtin str")
            _name(value, label)
        indices = tuple(self.node_indices)
        if len(indices) != 3 or any(
            isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in indices
        ):
            raise TypeError("node_indices require three nonbool integers")
        if min(indices) < 0 or len(set(indices)) != 3:
            raise ValueError("node_indices require distinct nonnegative indices")
        if (
            type(self.half_thickness_m) is not float
            or not np.isfinite(self.half_thickness_m)
            or self.half_thickness_m <= 0
        ):
            raise ValueError("half_thickness_m must be a finite positive builtin float")
        object.__setattr__(self, "node_indices", tuple(int(i) for i in indices))
