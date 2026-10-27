"""Conforming rectangular blocks split into six positively oriented tetrahedra/cell."""

from itertools import permutations
from numbers import Integral

import numpy as np

from ..core.state import ParticleState
from ..core.tetrahedral import TetrahedralModel
from ..core.validation import finite_real


def tetrahedral_block(
    origin, size, *, shape=(3, 2, 3), density=200.0, young_modulus=500.0, poisson_ratio=0.3
):
    shape = tuple(shape)
    if len(shape) != 3 or any(
        isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 2 for n in shape
    ):
        raise ValueError("block shape requires three integers >=2")
    origin = np.array([finite_real(value, "origin") for value in origin])
    size = np.array([finite_real(value, "size") for value in size])
    if origin.shape != (3,) or size.shape != (3,) or np.any(size <= 0):
        raise ValueError("block requires three finite origin coordinates and positive extents")
    if any(n > np.iinfo(np.int64).max for n in shape):
        raise ValueError("block shape exceeds int64 indexing")
    axes = [origin[i] + np.linspace(0.0, size[i], shape[i]) for i in range(3)]
    rest = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    def index(vertex):
        x, y, z = vertex
        return (x * shape[1] + y) * shape[2] + z

    cells = []
    for base in np.ndindex(*(n - 1 for n in shape)):
        for order in permutations(range(3)):
            vertex, vertices = np.array(base), [index(base)]
            for axis in order:
                vertex[axis] += 1
                vertices.append(index(vertex))
            inversions = sum(order[i] > order[j] for i in range(3) for j in range(i + 1, 3))
            if inversions % 2:
                vertices[1], vertices[2] = vertices[2], vertices[1]
            cells.append(vertices)
    fixed = np.indices(shape)[1].reshape(-1) == 0
    model = TetrahedralModel(
        rest,
        np.array(cells, dtype=np.int64),
        density,
        young_modulus,
        poisson_ratio,
        fixed,
        np.zeros(3),
    )
    return model, ParticleState(model.rest_position, np.zeros_like(model.rest_position))
