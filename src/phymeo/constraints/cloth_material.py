"""Owned additive distance/bending configuration; all units are SI."""

from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np

from ..core.scene import fingerprint
from ..core.validation import finite_real
from .bending import BendingConstraints
from .distance import DistanceConstraints


@dataclass(frozen=True, eq=False)
class ClothMaterial:
    distance: DistanceConstraints | None = None
    bending: BendingConstraints | None = None

    def __post_init__(self):
        for name, kind in (("distance", DistanceConstraints), ("bending", BendingConstraints)):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, kind):
                    raise TypeError(f"{name} requires {kind.__name__}")
                if value.compliance.dtype != np.float64:
                    raise TypeError("composite material requires FP64 configuration")
                object.__setattr__(self, name, replace(value))

    @property
    def identity(self):
        return fingerprint(self)

    @property
    def order(self):
        return ("distance", "bending")

    def count(self, kind):
        value = getattr(self, kind)
        return 0 if value is None else len(value.indices)

    def validate_model(self, model):
        if model.inverse_mass.dtype != np.float64 or np.any(model.inverse_mass <= 0):
            raise ValueError("composite physical scope requires all-dynamic FP64 particles")
        for name in self.order:
            law = getattr(self, name)
            if law is not None and np.any(law.indices >= model.particle_count):
                raise ValueError("material indices exceed the bound particle model")


@dataclass(frozen=True)
class CompositeMaterialAuditSettings:
    reference_angle_rad: float = 1.0
    sqrt_bits: int = 256
    angle_terms: int = 64
    max_domain_intervals: int = 4096
    max_domain_depth: int = 32

    def __post_init__(self):
        angle = finite_real(self.reference_angle_rad, "reference_angle_rad")
        if angle <= 0:
            raise ValueError("reference angle must be positive")
        object.__setattr__(self, "reference_angle_rad", angle)
        for name in ("sqrt_bits", "angle_terms", "max_domain_intervals", "max_domain_depth"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True)
class CompositeMaterialBudgets:
    max_energy_gain_j: float
    max_angular_defect_nms: float

    def __post_init__(self):
        for name in ("max_energy_gain_j", "max_angular_defect_nms"):
            value = finite_real(getattr(self, name), name)
            if value <= 0:
                raise ValueError("physical budgets must be positive")
            object.__setattr__(self, name, value)
