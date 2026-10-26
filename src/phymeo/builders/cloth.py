"""Triangulated XZ patches with area masses and explicit stretch/shear edges."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from ..constraints.distance import DistanceConstraints
from ..core.model import ParticleModel
from ..core.state import ParticleState


@dataclass(frozen=True, eq=False)
class ClothPatch:
    """Builder result: immutable topology/parameters and an owned mutable initial state."""

    model: ParticleModel
    initial_state: ParticleState
    constraints: DistanceConstraints
    triangles: np.ndarray
    vertex_mass: np.ndarray
    structural_edge_count: int


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    negative, zero = value < 0, value == 0
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} is not representable") from error
    if not np.isfinite(value) or negative or (positive and zero):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")
    if not zero and value == 0:
        raise ValueError(f"{name} is not representable")
    return value


def cloth_grid(
    rows: int,
    columns: int,
    *,
    spacing=0.1,
    density=1.0,
    structural_compliance=1e-5,
    shear_compliance=1e-5,
    pinned=(),
    origin=(0.0, 1.0, 0.0),
    dtype=np.float64,
) -> ClothPatch:
    """Build a +Y-facing patch; spacing is in metres and density in kg/m^2.

    Vertices are numbered row*columns+column. Structural edges join grid neighbors;
    shear edges join both cell diagonals. Compliance [m/N] is per edge, not a
    calibrated continuum constitutive law. No bending or self-collision is implied.
    Mass equals one third of each incident triangle's represented area mass.
    """
    for name, value in [("rows", rows), ("columns", columns)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    rows, columns = int(rows), int(columns)
    count = rows * columns
    if count > np.iinfo(np.int64).max:
        raise ValueError("grid exceeds int64 indexing")
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("dtype must be float32 or float64")
    spacing, density = (
        _number(spacing, "spacing", positive=True),
        _number(density, "density", positive=True),
    )
    structural = _number(structural_compliance, "structural_compliance")
    shear = _number(shear_compliance, "shear_compliance")
    origin = np.asarray(origin)
    if np.iscomplexobj(origin):
        raise ValueError("origin must be a finite real three-vector")
    origin = np.asarray(origin, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin must be a finite three-vector")
    pins = []
    for index in pinned:
        if (
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, Integral)
            or index < 0
            or index >= count
        ):
            raise ValueError("pinned indices must be integers within the grid")
        pins.append(int(index))
    ids = np.arange(count, dtype=np.int64)
    grid = ids.reshape(rows, columns)
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        position = np.empty((count, 3), dtype=dtype)
        position[:, 0] = origin[0] + spacing * (ids % columns)
        position[:, 1] = origin[1]
        position[:, 2] = origin[2] + spacing * (ids // columns)
        if not np.isfinite(position).all():
            raise ValueError("grid positions are not representable in the requested dtype")
        a = grid[:-1, :-1].ravel()
        b, c, d = a + 1, a + columns, a + columns + 1
        triangles = np.empty((2 * len(a), 3), dtype=np.int64)
        triangles[0::2] = np.column_stack([a, c, b])
        triangles[1::2] = np.column_stack([b, c, d])
        face = position[triangles].astype(np.float64)
        area = 0.5 * np.hypot.reduce(
            np.cross(face[:, 1] - face[:, 0], face[:, 2] - face[:, 0]), axis=1
        )
        if not np.isfinite(area).all() or np.any(area <= 0):
            raise ValueError("grid geometry is degenerate or outside the numerical range")
        mass64 = np.zeros(count, dtype=np.float64)
        np.add.at(mass64, triangles.ravel(), np.repeat(area * density / 3, 3))
        vertex_mass = mass64.astype(dtype)
        if not np.isfinite(vertex_mass).all() or np.any(vertex_mass <= 0):
            raise ValueError("positive vertex masses are not representable")
        inverse_mass = np.zeros(count, dtype=dtype)
        dynamic = np.ones(count, dtype=bool)
        dynamic[pins] = False
        np.divide(1.0, vertex_mass, out=inverse_mass, where=dynamic)
        if not np.isfinite(inverse_mass).all() or np.any(inverse_mass[dynamic] <= 0):
            raise ValueError("positive inverse masses are not representable")
        horizontal = np.column_stack([grid[:, :-1].ravel(), grid[:, 1:].ravel()])
        vertical = np.column_stack([grid[:-1, :].ravel(), grid[1:, :].ravel()])
        diagonal = np.empty((2 * len(a), 2), dtype=np.int64)
        diagonal[0::2] = np.column_stack([a, d])
        diagonal[1::2] = np.column_stack([b, c])
        edges = np.vstack([horizontal, vertical, diagonal])
        structural_count = len(horizontal) + len(vertical)
        geometry = position.astype(np.float64)
        rest = np.hypot.reduce(geometry[edges[:, 0]] - geometry[edges[:, 1]], axis=1).astype(dtype)
        if not np.isfinite(rest).all() or np.any(rest <= 0):
            raise ValueError("positive edge lengths are not representable")
        values = np.array([structural, shear], dtype=dtype)
        if not np.isfinite(values).all() or np.any(
            (values == 0) & (np.array([structural, shear]) > 0)
        ):
            raise ValueError("compliance is not representable in the requested dtype")
        compliance = np.empty(len(edges), dtype=dtype)
        compliance[:structural_count], compliance[structural_count:] = values
    triangles.flags.writeable = False
    vertex_mass.flags.writeable = False
    return ClothPatch(
        ParticleModel(inverse_mass),
        ParticleState(position, np.zeros_like(position)),
        DistanceConstraints(edges, rest, compliance),
        triangles,
        vertex_mass,
        structural_count,
    )
