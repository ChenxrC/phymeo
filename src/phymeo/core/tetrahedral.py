"""Owned FP64 linear tetrahedra with full physical mass and fixed-node weights."""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ._arrays import float_array
from .capabilities import SolverFailure
from .validation import finite_real


def tetra_array(value, name, *, ndim, columns=None):
    array = float_array(value, name, ndim=ndim, columns=columns)
    if array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name}: tetrahedral reference requires float64")
    return array


@dataclass(frozen=True, eq=False)
class TetrahedralModel:
    rest_position: np.ndarray
    tetrahedra: np.ndarray
    density: float
    young_modulus: float
    poisson_ratio: float
    fixed: np.ndarray
    gravity: np.ndarray
    rest_inverse: np.ndarray = field(init=False)
    volume: np.ndarray = field(init=False)
    lame_lambda: float = field(init=False)
    lame_mu: float = field(init=False)
    mass: np.ndarray = field(init=False)
    inverse_mass: np.ndarray = field(init=False)

    def __post_init__(self):
        rest = tetra_array(self.rest_position, "rest_position", ndim=2, columns=3)
        gravity = tetra_array(self.gravity, "gravity", ndim=1)
        if len(rest) < 4 or gravity.shape != (3,):
            raise ValueError("tetrahedra need at least four nodes and 3D gravity")
        cells = np.asarray(self.tetrahedra)
        if cells.ndim != 2 or cells.shape[1] != 4 or cells.dtype.kind not in "iu":
            raise TypeError("tetrahedra must be an (m,4) integer array")
        if len(cells) < 1 or np.any(cells < 0) or np.any(cells >= len(rest)):
            raise ValueError("tetrahedra must be nonempty with valid node indices")
        cells = np.array(cells, dtype=np.int64, copy=True)
        unordered = np.sort(cells, axis=1)
        if np.any(np.diff(unordered, axis=1) == 0):
            raise ValueError("each tetrahedron must have four different nodes")
        if len(np.unique(unordered, axis=0)) != len(cells):
            raise ValueError("duplicate tetrahedra")
        if len(np.unique(cells)) != len(rest):
            raise ValueError("every node must belong to a tetrahedron")
        fixed = np.asarray(self.fixed)
        if fixed.dtype != np.dtype(bool) or fixed.shape != (len(rest),):
            raise TypeError("fixed must be one boolean per node")
        fixed = fixed.copy()
        density, young, nu = (
            finite_real(getattr(self, name), name)
            for name in ("density", "young_modulus", "poisson_ratio")
        )
        if density <= 0 or young <= 0 or not -1 < nu < 0.5:
            raise ValueError("material requires density/E>0 and -1<nu<.5")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
                points = rest[cells]
                edges = np.stack(
                    [points[:, j] - points[:, i] for i, j in combinations(range(4), 2)], axis=1
                )
                scale = np.max(np.hypot.reduce(edges, axis=2), axis=1)
                if np.any(scale <= 0):
                    raise ValueError("degenerate reference tetrahedron")
                dm = (points[:, 1:] - points[:, :1]).transpose(0, 2, 1)
                normalized = dm / scale[:, None, None]
                determinant = np.linalg.det(normalized)
                if not np.isfinite(determinant).all():
                    raise FloatingPointError("non-finite reference determinant")
                if np.any(determinant <= 1e-12):
                    raise ValueError("reference tetrahedra must be positive and nondegenerate")
                volume = ((determinant / 6 * scale) * scale) * scale
                inverse = np.linalg.inv(normalized) / scale[:, None, None]
                mu = young / (2 * (1 + nu))
                lam = young * (nu / ((1 + nu) * (1 - 2 * nu)))
                mass = np.zeros(len(rest))
                np.add.at(mass, cells.reshape(-1), np.repeat(density * volume / 4, 4))
                inverse_mass = np.zeros_like(mass)
                inverse_mass[~fixed] = 1 / mass[~fixed]
                if (
                    not all(
                        np.isfinite(value).all()
                        for value in (volume, inverse, mu, lam, mass, inverse_mass)
                    )
                    or np.any(mass <= 0)
                    or np.any(volume <= 0)
                    or mu <= 0
                ):
                    raise FloatingPointError("unrepresentable material coefficients or mass")
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            raise SolverFailure(f"tetrahedral reference is not representable: {error}") from error
        for name, value in (
            ("rest_position", rest),
            ("tetrahedra", cells),
            ("fixed", fixed),
            ("gravity", gravity),
            ("rest_inverse", inverse),
            ("volume", volume),
            ("mass", mass),
            ("inverse_mass", inverse_mass),
        ):
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        for name, value in (
            ("density", density),
            ("young_modulus", young),
            ("poisson_ratio", nu),
            ("lame_lambda", lam),
            ("lame_mu", mu),
        ):
            object.__setattr__(self, name, float(value))

    @property
    def particle_count(self):
        return len(self.rest_position)
