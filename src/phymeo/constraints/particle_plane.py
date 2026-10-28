"""Owned particle-to-static-plane contact topology and coefficients."""

from dataclasses import dataclass

import numpy as np

from ..core._arrays import float_array
from ..core.scene import _name
from .friction import _norm


def _values(value, name, shape):
    value = float_array(value, name, ndim=len(shape))
    if value.dtype != np.float64 or value.shape != shape:
        raise ValueError(f"{name} requires float64 shape {shape}")
    return value.copy()


@dataclass(frozen=True, eq=False)
class ParticlePlaneContacts:
    """n.x-offset >= 0; friction uses the same static and dynamic coefficient."""

    domain: str
    indices: np.ndarray
    normals: np.ndarray
    offsets: np.ndarray
    friction: np.ndarray

    def __post_init__(self):
        _name(self.domain, "domain")
        indices = np.asarray(self.indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise TypeError("indices must be a one-dimensional integer array")
        if np.any(indices < 0) or np.any(indices > np.iinfo(np.int64).max):
            raise ValueError("indices must be nonnegative int64 values")
        indices = np.array(indices, dtype=np.int64, copy=True)
        count = len(indices)
        normals = _values(self.normals, "normals", (count, 3))
        offsets = _values(self.offsets, "offsets", (count,))
        friction = _values(self.friction, "friction", (count,))
        if np.any(friction < 0):
            raise ValueError("friction must be nonnegative")
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            for index, normal in enumerate(normals):
                length = _norm(normal)
                if abs(length - 1) > 1e-12:
                    raise ValueError("plane normals must have unit length")
                normals[index] /= length
                offsets[index] /= length
        for name, value in (
            ("indices", indices),
            ("normals", normals),
            ("offsets", offsets),
            ("friction", friction),
        ):
            value.flags.writeable = False
            object.__setattr__(self, name, value)
