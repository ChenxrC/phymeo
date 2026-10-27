"""Explicit physical models and independently implemented simulation solvers."""

from .backends.tyx import TyxSemiImplicitSolver
from .backends.tyx_coupled import TyxCoupledXPBDSolver
from .backends.tyx_rigid import TyxRigidSemiImplicitSolver
from .backends.tyx_xpbd import TyxXPBDSolver
from .builders.cloth import ClothPatch, cloth_grid
from .constraints.distance import DistanceConstraints
from .core.capabilities import ProblemSignature, SolverCapabilities, SolverFailure
from .core.coupled import CoupledInputs, CoupledState, CoupledStepReport, CoupledStepResult
from .core.coupling import ImpulseExchangeResult, ImpulseLedger
from .core.model import ParticleModel
from .core.result import StepReport, StepResult
from .core.rigid import RigidBodyInputs, RigidBodyModel, RigidBodyState
from .core.state import ParticleState, StepInputs
from .core.verification import CandidateVerifier, VerificationReport
from .coupling.point_port import ParticleRigidPorts
from .geometry.particle_sphere import ParticleSphereGeometry, particle_sphere_geometry
from .geometry.plane import Plane
from .geometry.sphere import SphereShapes
from .runtime.stepper import Stepper
from .solvers.coupled_xpbd import CoupledXPBDSolver
from .solvers.rigid_semi_implicit import RigidSemiImplicitSolver
from .solvers.rigid_sphere_plane import RigidSpherePlaneSolver
from .solvers.semi_implicit import SemiImplicitSolver
from .solvers.xpbd import XPBDSolver
from .validation.free_particle import FreeParticleVerifier

__version__ = "0.1.0a0"

__all__ = [
    "ParticleModel",
    "ParticleState",
    "StepInputs",
    "StepReport",
    "StepResult",
    "CoupledState",
    "CoupledInputs",
    "CoupledStepReport",
    "CoupledStepResult",
    "CoupledXPBDSolver",
    "SemiImplicitSolver",
    "DistanceConstraints",
    "Plane",
    "XPBDSolver",
    "TyxSemiImplicitSolver",
    "TyxRigidSemiImplicitSolver",
    "TyxCoupledXPBDSolver",
    "TyxXPBDSolver",
    "ProblemSignature",
    "SolverCapabilities",
    "SolverFailure",
    "Stepper",
    "CandidateVerifier",
    "VerificationReport",
    "FreeParticleVerifier",
    "ParticleRigidPorts",
    "ImpulseLedger",
    "ImpulseExchangeResult",
    "RigidBodyModel",
    "RigidBodyState",
    "RigidBodyInputs",
    "RigidSemiImplicitSolver",
    "SphereShapes",
    "ParticleSphereGeometry",
    "particle_sphere_geometry",
    "RigidSpherePlaneSolver",
    "ClothPatch",
    "cloth_grid",
]
