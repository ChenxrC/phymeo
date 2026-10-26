"""Observable solver outcomes; numerical failure is separate from invalid input."""

from dataclasses import dataclass
from typing import Generic, TypeVar

StateT = TypeVar("StateT")


@dataclass(frozen=True)
class StepReport:
    """Residual is normalized; geometric error is separately reported in metres."""

    accepted: bool
    iterations: int = 0
    max_residual: float = 0.0
    max_constraint_error: float = 0.0
    reason: str | None = None
    backend: str = "numpy"
    fallback_used: bool = False
    primary_failure: str | None = None


@dataclass(frozen=True)
class StepResult(Generic[StateT]):
    state: StateT
    report: StepReport
