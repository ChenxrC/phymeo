"""Solver contracts shared by mathematical and future learned candidates."""

from dataclasses import dataclass
from typing import Protocol

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


class SolverFailure(Exception):
    """An expected domain/execution rejection that permits a configured fallback."""


class Solver(Protocol):
    @property
    def capabilities(self) -> SolverCapabilities: ...

    def problem_signature(self) -> ProblemSignature: ...

    def step(
        self, model: ParticleModel, state_in: ParticleState, inputs: StepInputs, dt: float
    ) -> StepResult: ...
