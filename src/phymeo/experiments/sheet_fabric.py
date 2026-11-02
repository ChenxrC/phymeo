"""Discrete fabric presets and a regular material grid used by cloth experiments."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..constraints.bending import BendingConstraints, bending_angle_gradient
from ..constraints.cloth_material import ClothMaterial
from ..constraints.distance import DistanceConstraints
from ..core.validation import finite_real

WIDTH_M = 0.12
HEIGHT_M = 0.05
MASS_KG = 0.048
HALF_THICKNESS_M = 0.0015
DRAPE_GRID = (7, 3)


def checked_grid(grid):
    columns, rows = grid
    if any(
        isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 2
        for n in (columns, rows)
    ):
        raise ValueError("sheet grid needs at least two integer nodes on each axis")
    return int(columns), int(rows)


@dataclass(frozen=True)
class SheetFabric:
    """Per-edge stretch compliance (m/N), per-hinge bend compliance (1/(N m))."""

    name: str
    stretch_compliance: float
    bend_compliance: float
    suggested_grid: tuple[int, int] = DRAPE_GRID

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("fabric needs a name")
        for field in ("stretch_compliance", "bend_compliance"):
            value = finite_real(getattr(self, field), field)
            if value <= 0:
                raise ValueError(f"{field} must be positive")
            object.__setattr__(self, field, value)
        object.__setattr__(self, "suggested_grid", checked_grid(self.suggested_grid))

    def record(self, grid):
        return dict(
            name=self.name,
            grid=list(grid),
            stretch_compliance_m_per_n=self.stretch_compliance,
            bend_compliance_per_nm=self.bend_compliance,
            mass_kg=MASS_KG,
            half_thickness_m=HALF_THICKNESS_M,
        )


REFERENCE = SheetFabric("reference", 0.001, 20.0, (2, 3))
SOFT = SheetFabric("soft", 0.001, 2000.0)


def fabric_preset(fabric):
    if isinstance(fabric, SheetFabric):
        return fabric
    if isinstance(fabric, str):
        for spec in (REFERENCE, SOFT):
            if fabric == spec.name:
                return spec
    raise ValueError("fabric must be 'reference', 'soft', or a SheetFabric")


def sheet_grid(grid):
    columns, rows = checked_grid(grid)
    u = np.linspace(-WIDTH_M / 2, WIDTH_M / 2, columns)
    v = np.linspace(0.0, HEIGHT_M, rows)
    uv = np.array([(a, b) for b in v for a in u])
    faces = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            i = row * columns + column
            faces.extend(((i, i + 1, i + columns + 1), (i, i + columns + 1, i + columns)))
    return uv, tuple(faces)


def material_for_sheet(positions, faces, fabric):
    """Build distance and signed-dihedral rows from one rest mesh."""
    sides = {}
    for a, b, c in faces:
        for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
            sides.setdefault(tuple(sorted((i, j))), []).append((i, j, k))
    edges = np.array(sorted(sides))
    hinges = []
    for incidents in sides.values():
        if len(incidents) == 1:
            continue
        if len(incidents) != 2:
            raise ValueError("nonmanifold sheet edge")
        (a, b, c), (j, i, d) = incidents
        if (a, b) != (i, j):
            raise ValueError("inconsistent sheet winding")
        hinges.append((a, b, c, d))
    bending = None
    if hinges:
        hinge = np.array(hinges)
        rest = np.array([bending_angle_gradient(positions[r])[0] for r in hinge])
        bending = BendingConstraints(hinge, rest, np.full(len(hinge), fabric.bend_compliance))
    return ClothMaterial(
        DistanceConstraints(
            edges,
            np.linalg.norm(positions[edges[:, 1]] - positions[edges[:, 0]], axis=1),
            np.full(len(edges), fabric.stretch_compliance),
        ),
        bending,
    )
