"""Unconstrained particle dynamics baseline."""

from ..core.capabilities import ProblemSignature, SolverCapabilities
from ..core.model import ParticleModel
from ..core.result import StepReport, StepResult
from ..core.state import ParticleState, StepInputs
from ..core.validation import validate_step
from ..dynamics.integrate import predict


class SemiImplicitSolver:
    """Advance v first, then x using the new v; fixed particles have zero velocity."""

    capabilities = SolverCapabilities(
        backend="numpy",
        dtypes=frozenset(["float32", "float64"]),
        features=frozenset(["particles", "free_dynamics"]),
    )

    def problem_signature(self) -> ProblemSignature:
        return ProblemSignature()

    def step(
        self, model: ParticleModel, state_in: ParticleState, inputs: StepInputs, dt: float
    ) -> StepResult:
        candidate, force, h = validate_step(model, state_in, inputs, dt)
        if not predict(model, candidate, force, h):
            return StepResult(
                ParticleState.copy(state_in),
                StepReport(accepted=False, reason="non-finite integration result"),
            )
        return StepResult(candidate, StepReport(accepted=True))
