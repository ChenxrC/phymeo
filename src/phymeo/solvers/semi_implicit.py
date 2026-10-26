"""Unconstrained particle dynamics baseline."""

from ..core.model import ParticleModel
from ..core.result import StepReport, StepResult
from ..core.state import ParticleState, StepInputs
from ..dynamics.integrate import predict, validate_step


class SemiImplicitSolver:
    """Advance v first, then x using the new v; fixed particles have zero velocity."""

    def step(
        self, model: ParticleModel, state_in: ParticleState, inputs: StepInputs, dt: float
    ) -> StepResult:
        candidate, force, h = validate_step(model, state_in, inputs, dt)
        if not predict(model, candidate, force, h):
            return StepResult(
                state_in.copy(),
                StepReport(accepted=False, reason="non-finite integration result"),
            )
        return StepResult(candidate, StepReport(accepted=True))
