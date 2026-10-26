"""Explicit physical models and independently implemented simulation solvers."""

from .backends.tyx import TyxSemiImplicitSolver
from .backends.tyx_xpbd import TyxXPBDSolver
from .builders.cloth import ClothPatch, cloth_grid
from .constraints.distance import DistanceConstraints
from .core.capabilities import ProblemSignature, SolverCapabilities, SolverFailure
from .core.model import ParticleModel
from .core.result import StepReport, StepResult
from .core.rigid import RigidBodyInputs, RigidBodyModel, RigidBodyState
from .core.state import ParticleState, StepInputs
from .geometry.plane import Plane
from .geometry.sphere import SphereShapes
from .runtime.stepper import Stepper
from .solvers.rigid_semi_implicit import RigidSemiImplicitSolver
from .solvers.rigid_sphere_plane import RigidSpherePlaneSolver
from .solvers.semi_implicit import SemiImplicitSolver
from .solvers.xpbd import XPBDSolver

__version__ = "0.1.0a0"

__all__ = [
    "ParticleModel",
    "ParticleState",
    "StepInputs",
    "StepReport",
    "StepResult",
    "SemiImplicitSolver",
    "DistanceConstraints",
    "Plane",
    "XPBDSolver",
    "TyxSemiImplicitSolver",
    "TyxXPBDSolver",
    "ProblemSignature",
    "SolverCapabilities",
    "SolverFailure",
    "Stepper",
    "RigidBodyModel",
    "RigidBodyState",
    "RigidBodyInputs",
    "RigidSemiImplicitSolver",
    "SphereShapes",
    "RigidSpherePlaneSolver",
    "ClothPatch",
    "cloth_grid",
]
