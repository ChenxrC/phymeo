"""Independent NumPy reference with the device's stable distance-color ordering."""

import numpy as np

from ..constraints.coloring import color_edges
from ..constraints.distance import DistanceConstraints
from .coupled_xpbd import CoupledXPBDSolver


class ColoredCoupledXPBDSolver(CoupledXPBDSolver):
    """Reorder independent edges by color; retain the NumPy numerical reference.

    FP32 state storage still uses the reference solver's FP64 geometry and
    multipliers. This measures device arithmetic error under the same scheduling,
    rather than attributing a different iteration order to floating-point error.
    """

    def __post_init__(self):
        super().__post_init__()
        constraints = self.constraints
        indices = np.empty((0, 2), np.int64) if constraints is None else constraints.indices
        colors = color_edges(indices)
        object.__setattr__(self, "_colors", colors)
        if constraints is not None:
            order = colors.order
            object.__setattr__(
                self,
                "constraints",
                DistanceConstraints(
                    constraints.indices[order],
                    constraints.rest_length[order],
                    constraints.compliance[order],
                ),
            )

    @property
    def color_count(self):
        return len(self._colors.offsets) - 1
