"""Explicit physical models and independently implemented simulation solvers."""

from .backends.tyx import TyxSemiImplicitSolver
from .backends.tyx_articulated import (
    TyxArticulatedCouplingProgram,
    make_tyx_articulated_runtime,
)
from .backends.tyx_coupled import TyxCoupledXPBDSolver
from .backends.tyx_participants import (
    TyxClothParticipant,
    TyxRobotParticipant,
    TyxTetrahedralParticipant,
)
from .backends.tyx_rigid import TyxRigidSemiImplicitSolver
from .backends.tyx_xpbd import TyxXPBDSolver
from .builders.cloth import ClothPatch, cloth_grid
from .builders.robot import pendulum_chain
from .builders.tetrahedral import tetrahedral_block
from .constraints.contact_impulse import ContactImpulseResult, solve_contact_impulses
from .constraints.distance import DistanceConstraints
from .constraints.friction import FrictionImpulseResult, solve_friction_disk
from .constraints.node_robot import NodeRobotContact
from .constraints.particle_plane import ParticlePlaneContacts
from .constraints.robot_node_plane import RobotNodePlaneContacts
from .constraints.robot_plane import RobotPlaneContacts
from .control.bounded_pd import BoundedPDResult, solve_bounded_pd
from .core.capabilities import ProblemSignature, SolverCapabilities, SolverFailure
from .core.coupled import CoupledInputs, CoupledState, CoupledStepReport, CoupledStepResult
from .core.coupling import ImpulseExchangeResult, ImpulseLedger
from .core.model import ParticleModel
from .core.result import StepReport, StepResult
from .core.rigid import RigidBodyInputs, RigidBodyModel, RigidBodyState
from .core.robot import RobotInputs, RobotModel, RobotState
from .core.scene import (
    BackendContext,
    DomainBinding,
    HistoryField,
    SceneInputs,
    SceneModel,
    SceneState,
    StageMetrics,
    StepSettings,
)
from .core.state import ParticleState, StepInputs
from .core.tetrahedral import TetrahedralModel
from .core.verification import CandidateVerifier, VerificationReport
from .coupling.candidate_audit import PointCandidateAudit
from .coupling.domain_graph import MultiDomainPointOperator
from .coupling.offset_port import TriangleOffsetPointPort
from .coupling.participant import (
    ClothParticipant,
    ParticipantWorkspace,
    RobotParticipant,
    TetrahedralParticipant,
)
from .coupling.point_port import ParticleRigidPorts
from .coupling.ports import ParticlePointPort, PointPort, RigidPointPort, exchange_impulses
from .coupling.robot_port import RobotPointPort
from .coupling.surface_port import PointPairOperator, WeightedPointPort
from .coupling.tetrahedral_port import TetrahedralPointPort
from .geometry.feature_distance_bound import FeatureDistanceBounds, feature_distance_bounds
from .geometry.linear_sweep import FeatureSweepResult, linear_feature_sweep
from .geometry.particle_sphere import ParticleSphereGeometry, particle_sphere_geometry
from .geometry.plane import Plane
from .geometry.robot_sphere import RobotSphereAttachments
from .geometry.sphere import SphereShapes
from .geometry.surface_features import (
    ClosestFeaturePair,
    closest_point_triangle,
    closest_segment_segment,
)
from .runtime.load_replay import ReplayTrial, SceneLoadReplay
from .runtime.program import (
    ProgramCapabilities,
    SceneProgram,
    SceneStepReport,
    SceneStepResult,
    SceneSubstepReport,
    SceneVerifier,
    SubstepWorkspace,
    TerminalPointImpulseWorkspace,
)
from .runtime.registry import StateFamily, StateRegistry
from .runtime.scene_stepper import SceneStepper
from .runtime.stepper import Stepper
from .runtime.terminal_impulse import PointVelocityImpulses, TerminalImpulseProgram
from .solvers.articulated_coupling import ArticulatedCouplingProgram, make_articulated_runtime
from .solvers.cloth import ClothProgram, make_cloth_runtime
from .solvers.coupled_program import (
    CoupledRuntimeAdapter,
    NumpyCoupledProgram,
    make_coupled_runtime,
)
from .solvers.coupled_xpbd import CoupledXPBDSolver
from .solvers.free_scene import FreeSceneProgram
from .solvers.initial_robot_material_contact import (
    InitialRobotMaterialContactProgram,
    MaterialContactBudgets,
)
from .solvers.initial_robot_node_contact import InitialRobotNodeContactProgram
from .solvers.particle_plane_contact import ParticlePlaneContactProgram
from .solvers.rigid_semi_implicit import RigidSemiImplicitSolver
from .solvers.rigid_sphere_plane import RigidSpherePlaneSolver
from .solvers.robot import RobotProgram, make_robot_runtime
from .solvers.robot_plane_contact import RobotPlaneContactProgram
from .solvers.semi_implicit import SemiImplicitSolver
from .solvers.tetrahedral import TetrahedralProgram, make_tetrahedral_runtime
from .solvers.xpbd import XPBDSolver
from .validation.free_particle import FreeParticleVerifier

__version__ = "0.1.0a0"

from .validation.material_path import MaterialPathLedger
from .validation.terminal_impulse import TerminalImpulseLedger

__all__ = [
    "PointVelocityImpulses",
    "TerminalImpulseProgram",
    "TerminalImpulseLedger",
    "TerminalPointImpulseWorkspace",
    "FeatureSweepResult",
    "linear_feature_sweep",
    "FeatureDistanceBounds",
    "feature_distance_bounds",
    "MaterialPathLedger",
    "ParticlePlaneContacts",
    "ParticlePlaneContactProgram",
    "RobotPlaneContactProgram",
    "RobotNodePlaneContacts",
    "InitialRobotNodeContactProgram",
    "InitialRobotMaterialContactProgram",
    "MaterialContactBudgets",
    "RobotPlaneContacts",
    "SceneLoadReplay",
    "ReplayTrial",
    "ContactImpulseResult",
    "solve_contact_impulses",
    "MultiDomainPointOperator",
    "FrictionImpulseResult",
    "solve_friction_disk",
    "ClosestFeaturePair",
    "closest_point_triangle",
    "closest_segment_segment",
    "TriangleOffsetPointPort",
    "WeightedPointPort",
    "PointPairOperator",
    "TyxRobotParticipant",
    "TyxClothParticipant",
    "TyxTetrahedralParticipant",
    "TyxArticulatedCouplingProgram",
    "make_tyx_articulated_runtime",
    "RobotParticipant",
    "ClothParticipant",
    "TetrahedralParticipant",
    "ParticipantWorkspace",
    "RobotSphereAttachments",
    "NodeRobotContact",
    "ArticulatedCouplingProgram",
    "make_articulated_runtime",
    "TetrahedralModel",
    "TetrahedralProgram",
    "TetrahedralPointPort",
    "ClothProgram",
    "make_tetrahedral_runtime",
    "make_cloth_runtime",
    "tetrahedral_block",
    "RobotModel",
    "RobotState",
    "RobotInputs",
    "RobotPointPort",
    "RobotProgram",
    "make_robot_runtime",
    "pendulum_chain",
    "BackendContext",
    "DomainBinding",
    "HistoryField",
    "SceneInputs",
    "SceneModel",
    "SceneState",
    "StageMetrics",
    "StepSettings",
    "ProgramCapabilities",
    "SceneProgram",
    "SubstepWorkspace",
    "SceneVerifier",
    "SceneStepReport",
    "SceneStepResult",
    "SceneSubstepReport",
    "StateFamily",
    "StateRegistry",
    "SceneStepper",
    "FreeSceneProgram",
    "NumpyCoupledProgram",
    "CoupledRuntimeAdapter",
    "make_coupled_runtime",
    "PointPort",
    "PointCandidateAudit",
    "BoundedPDResult",
    "solve_bounded_pd",
    "ParticlePointPort",
    "RigidPointPort",
    "exchange_impulses",
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
