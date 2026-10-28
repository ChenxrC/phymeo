"""Exact audits of free-node native replay; never generate replacement states."""

from fractions import Fraction

from ..core.capabilities import SolverFailure
from ._robot_replay_ledger import rational


class FreeParticleLedger:
    def __init__(self, model, state, inputs, h, settings):
        if any(d <= 0 for d in model.inverse_mass):
            raise ValueError("selected free-node domains must have positive inverse masses")
        self.model, self.initial, self.original = model, state, inputs
        self.h, self.scale = rational(h), rational(settings.reference_speed)

    def audit(self, state, effective, expected):
        residual = dict(node_dynamics=Fraction(0), node_motion=Fraction(0), node_input=Fraction(0))
        for i, inverse in enumerate(self.model.inverse_mass):
            d = rational(inverse)
            for k in range(3):
                actual = self.h * (
                    rational(effective.force[i, k]) - rational(self.original.force[i, k])
                )
                free = self.h * (
                    rational(self.model.gravity[k]) + d * rational(self.original.force[i, k])
                )
                dv = rational(state.velocity[i, k]) - rational(self.initial.velocity[i, k])
                motion = (
                    rational(state.position[i, k]) - rational(self.initial.position[i, k])
                ) / self.h - rational(state.velocity[i, k])
                residual["node_dynamics"] = max(
                    residual["node_dynamics"], abs(dv - free - d * expected[i][k]) / self.scale
                )
                residual["node_input"] = max(
                    residual["node_input"], abs(actual - expected[i][k]) * d / self.scale
                )
                residual["node_motion"] = max(residual["node_motion"], abs(motion) / self.scale)
        return residual

    def kinetic(self, state):
        return sum(
            (
                rational(v) ** 2 / (2 * rational(d))
                for d, row in zip(self.model.inverse_mass, state.velocity, strict=True)
                for v in row
            ),
            Fraction(0),
        )

    def require(self, state, effective, expected, tolerance):
        result = self.audit(state, effective, expected)
        if max(result.values()) > rational(tolerance):
            raise SolverFailure(
                "native selected nodes do not satisfy free dynamics/input/motion tolerance"
            )
        return result
