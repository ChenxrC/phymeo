"""An infinite static y-up plane for the first frictionless contact baseline."""

from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class Plane:
    """Admissible particle positions satisfy y >= height (SI metres)."""

    height: float = 0.0
    normal: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def __post_init__(self):
        if isinstance(self.height, (bool, np.bool_)) or not isinstance(self.height, Real):
            raise ValueError("plane height must be a finite real scalar")
        if not np.isfinite(self.height):
            raise ValueError("plane height must be finite")
        normal = np.asarray(self.normal)
        if normal.shape != (3,) or not np.array_equal(normal, [0.0, 1.0, 0.0]):
            raise ValueError("only normal=(0, 1, 0) is supported")
        object.__setattr__(self, "height", float(self.height))
        object.__setattr__(self, "normal", (0.0, 1.0, 0.0))
