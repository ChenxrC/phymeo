"""Independent physical validation, separate from a provider's self-report."""

from dataclasses import dataclass
from typing import Protocol, TypeVar

from .capabilities import ProblemSignature


@dataclass(frozen=True)
class VerificationReport:
    """Dimensionless physical residual and geometric position error in metres."""

    accepted: bool
    max_residual: float = 0.0
    max_constraint_error: float = 0.0
    reason: str | None = None


ModelT = TypeVar("ModelT", contravariant=True)
StateT = TypeVar("StateT", contravariant=True)
InputsT = TypeVar("InputsT", contravariant=True)


class CandidateVerifier(Protocol[ModelT, StateT, InputsT]):
    """Trusted physics logic receives owned attempt inputs and candidate copies."""

    @property
    def state_kind(self) -> str: ...

    def problem_signature(self) -> ProblemSignature: ...

    def verify(
        self, model: ModelT, state_in: StateT, inputs: InputsT, dt: float, candidate: StateT
    ) -> VerificationReport: ...
