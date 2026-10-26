"""Solver contracts shared by mathematical and future learned candidates."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Protocol

import numpy as np

from .model import ParticleModel
from .result import StepResult
from .state import ParticleState, StepInputs


@dataclass(frozen=True)
class SolverCapabilities:
    backend: str
    dtypes: frozenset[str]
    features: frozenset[str]
    state_kind: str = "particles"
    gradient: str = "unsupported"
    history: str = "none"

    def __post_init__(self):
        object.__setattr__(self, "dtypes", frozenset(self.dtypes))
        object.__setattr__(self, "features", frozenset(self.features))


@dataclass(frozen=True)
class ProblemSignature:
    """Physical identity, excluding backend, tolerances and iteration budget.

    Model and input parameters are shared separately by the Stepper. Distance
    entries contain (i, j, rest_length, compliance); plane contains (height, law).
    """

    semantics: str = "particle-force-gravity-v1"
    distance: tuple[tuple[int, int, float, float], ...] = ()
    plane: tuple[float, str] | None = None

    def __post_init__(self):
        if not isinstance(self.semantics, str) or not self.semantics.strip():
            raise ValueError("semantics must be a non-empty version identifier")
        distance = []
        for row in self.distance:
            if len(row) != 4:
                raise ValueError("distance identity must contain (i, j, rest_length, compliance)")
            i, j, rest, compliance = row
            for index in (i, j):
                if (
                    isinstance(index, (bool, np.bool_))
                    or not isinstance(index, Integral)
                    or index < 0
                    or index > np.iinfo(np.int64).max
                ):
                    raise ValueError("distance indices must be non-negative int64 values")
            if i == j:
                raise ValueError("distance identity must connect different particles")
            rest = _finite_scalar(rest, "rest_length")
            compliance = _finite_scalar(compliance, "compliance")
            if rest < 0 or compliance < 0:
                raise ValueError("rest_length and compliance must be non-negative")
            distance.append((int(i), int(j), rest, compliance))
        object.__setattr__(self, "distance", tuple(distance))
        if self.plane is not None:
            if len(self.plane) != 2:
                raise ValueError("plane identity must contain (height, law)")
            height, law = self.plane
            height = _finite_scalar(height, "plane height")
            if not isinstance(law, str) or not law.strip():
                raise ValueError("plane law must be a non-empty version identifier")
            object.__setattr__(self, "plane", (height, law))


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} is not representable in FP64") from error
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class SolverFailure(Exception):
    """An expected domain/execution rejection that permits a configured fallback."""


class Solver(Protocol):
    @property
    def capabilities(self) -> SolverCapabilities: ...

    def problem_signature(self) -> ProblemSignature: ...

    def step(
        self, model: ParticleModel, state_in: ParticleState, inputs: StepInputs, dt: float
    ) -> StepResult: ...
