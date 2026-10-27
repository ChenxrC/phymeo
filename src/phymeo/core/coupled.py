"""Owned mixed states and explicit whole-frame/substep outcomes."""

from dataclasses import dataclass

from .coupling import ImpulseLedger
from .rigid import RigidBodyInputs, RigidBodyState
from .state import ParticleState, StepInputs


@dataclass(eq=False)
class CoupledState:
    particles: ParticleState
    rigid: RigidBodyState

    def __post_init__(self):
        if not isinstance(self.particles, ParticleState) or not isinstance(
            self.rigid, RigidBodyState
        ):
            raise TypeError("coupled state requires particle and rigid states")
        self.particles = ParticleState(self.particles.position, self.particles.velocity)
        self.rigid = RigidBodyState(
            self.rigid.position,
            self.rigid.orientation,
            self.rigid.velocity,
            self.rigid.angular_velocity,
        )
        if self.particles.position.dtype != self.rigid.position.dtype:
            raise ValueError("coupled state families must have the same dtype")


@dataclass(eq=False)
class CoupledInputs:
    particles: StepInputs
    rigid: RigidBodyInputs

    def __post_init__(self):
        if not isinstance(self.particles, StepInputs) or not isinstance(
            self.rigid, RigidBodyInputs
        ):
            raise TypeError("coupled inputs require particle force and rigid force/torque")
        self.particles = StepInputs(self.particles.force)
        self.rigid = RigidBodyInputs(self.rigid.force, self.rigid.torque)
        if self.particles.force.dtype != self.rigid.force.dtype:
            raise ValueError("coupled input families must have the same dtype")

    @classmethod
    def zeros(cls, particle_count, body_count, *, dtype="float64"):
        return cls(
            StepInputs.zeros(particle_count, dtype=dtype),
            RigidBodyInputs.zeros(body_count, dtype=dtype),
        )


@dataclass(frozen=True)
class CoupledSubstepReport:
    accepted: bool
    iterations: int = 0
    velocity_iterations: int = 0
    distance_residual: float = 0.0
    contact_residual: float = 0.0
    velocity_residual: float = 0.0
    geometric_error: float = 0.0
    penetration: float = 0.0
    position_impulses: ImpulseLedger | None = None
    velocity_impulses: ImpulseLedger | None = None
    reason: str | None = None


@dataclass(frozen=True)
class CoupledStepReport:
    """Attempt diagnostics: no substep is committed unless accepted is True.

    completed_substeps counts accepted computations before a possible failure;
    the returned state still rolls back the whole frame when accepted=False.
    """

    accepted: bool
    completed_substeps: int
    substeps: tuple[CoupledSubstepReport, ...] = ()
    reason: str | None = None
    backend: str = "numpy"

    @property
    def iterations(self):
        return sum(item.iterations for item in self.substeps)

    @property
    def distance_residual(self):
        return max((item.distance_residual for item in self.substeps), default=0.0)

    @property
    def contact_residual(self):
        return max((item.contact_residual for item in self.substeps), default=0.0)

    @property
    def velocity_residual(self):
        return max((item.velocity_residual for item in self.substeps), default=0.0)

    @property
    def max_residual(self):
        return max(self.distance_residual, self.contact_residual, self.velocity_residual)


@dataclass(frozen=True)
class CoupledStepResult:
    """Producer-owned final state and optional independent accepted snapshots.

    Direct dataclass construction retains supplied state references. The solver
    supplies canonical copies; failed frames expose no partial substep snapshots.
    """

    state: CoupledState
    report: CoupledStepReport
    substep_states: tuple[CoupledState, ...] = ()
