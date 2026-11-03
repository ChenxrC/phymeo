"""Explicit physical models and independently implemented simulation solvers."""

from .backends.tyx import TyxSemiImplicitSolver
from .backends.tyx_articulated import (
    TyxArticulatedCouplingProgram,
    make_tyx_articulated_runtime,
)
from .backends.tyx_coupled import TyxCoupledXPBDSolver
from .backends.tyx_normal_layer import TyxNormalLayer
from .backends.tyx_participants import (
    TyxClothParticipant,
    TyxRobotParticipant,
    TyxTetrahedralParticipant,
)
from .backends.tyx_rigid import TyxRigidSemiImplicitSolver
from .backends.tyx_sdf import TyxSdfSampler
from .backends.tyx_world_batch import FreeSphere, PrismaticSlider, TyxWorldBatch, WorldCheckpoint
from .backends.tyx_world_contact import (
    ClothLinkWorld,
    TyxWorldContactBatch,
    WorldContactResult,
    WorldContactState,
)
from .backends.tyx_xpbd import TyxXPBDSolver
from .builders.cloth import ClothPatch, cloth_grid
from .builders.mjcf import load_mjcf
from .builders.robot import pendulum_chain
from .builders.tetrahedral import tetrahedral_block
from .builders.urdf import load_urdf
from .builders.usd import load_usd
from .constraints.collision_shapes import (
    CollisionKey,
    CollisionModel,
    CollisionMotion,
    CollisionPolicy,
    CollisionShape,
    RigidTurn,
)
from .constraints.contact_impulse import ContactImpulseResult, solve_contact_impulses
from .constraints.curved_solids import CurvedSolid
from .constraints.distance import DistanceConstraints
from .constraints.friction import FrictionImpulseResult, solve_friction_disk
from .constraints.heightfield import Heightfield
from .constraints.node_robot import NodeRobotContact
from .constraints.particle_plane import ParticlePlaneContacts
from .constraints.robot_contact import (
    CollisionPlane,
    JointLimits,
    LinkCapsule,
    RobotContactMaterial,
)
from .constraints.robot_finite_patch_collider import RobotFinitePatchCollider
from .constraints.robot_node_plane import RobotNodePlaneContacts
from .constraints.robot_plane import RobotPlaneContacts
from .constraints.robot_plane_collider import RobotPlaneCollider
from .control.bounded_pd import BoundedPDResult, solve_bounded_pd
from .control.point_ik import PointIKResult, solve_point_ik
from .control.site_ik import MultiSiteIKResult, SiteIKResidual, SiteIKTarget, solve_multi_site_ik
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
from .coupling.floating_port import FloatingRobotPointPort
from .coupling.witness_impulses import solve_witness_impulses
from .coupling.ports import ParticlePointPort, PointPort, RigidPointPort, exchange_impulses
from .coupling.robot_port import RobotPointPort
from .coupling.surface_port import PointPairOperator, WeightedPointPort
from .coupling.tetrahedral_port import TetrahedralPointPort
from .dynamics.floating_robot import (
    FloatingKinematics,
    FloatingRobotModel,
    FloatingRobotState,
    floating_kinematics,
    floating_point_jacobians,
)
from .dynamics.robot_contact import RobotContactSystem
from .geometry._convex_support import ConvexSupportTrace
from .geometry._curved_solids import CurvedContact, CurvedEvent
from .geometry._heightfield import HeightfieldEvent, HeightfieldFace, HeightfieldTrace
from .geometry._mesh_mesh import MeshMeshPose, MeshMeshSupport, MeshMeshTrace
from .geometry._mesh_round import MeshRoundEvent, MeshRoundPose, MeshRoundSupport, MeshRoundTrace
from .geometry.batched_collision import WorldCollisionWorkspace
from .geometry.collision_pipeline import (
    CollisionBounds,
    CollisionDetection,
    CollisionPair,
    CollisionSweep,
    CollisionWitness,
    CollisionWorkspace,
    detect,
)
from .geometry.collision_witness import (
    CapsuleAxisPoint,
    ConvexContactPatch,
    ConvexContactPoint,
    ConvexContactPose,
    ConvexHullTrace,
    HullSurfacePose,
    HullSurfaceTrace,
    MaterialContactPoint,
    MeshContactPose,
    MeshContactTrace,
    PlateContactPose,
    PlateContactTrace,
    TetrahedralBodyPair,
    TetrahedralBoundaryContact,
    TetrahedralBoundaryTrace,
    TetrahedralFacetResult,
    TetrahedralOccupancy,
    TetrahedralSolidResult,
)
from .geometry.feature_distance_bound import FeatureDistanceBounds, feature_distance_bounds
from .geometry.linear_sweep import FeatureSweepResult, linear_feature_sweep
from .geometry.mesh_arc_surface import (
    MeshArcBoundaryEvent,
    MeshArcSupport,
    MeshArcSurfaceResult,
    mesh_arc_first_surface,
)
from .geometry.paged_collision import (
    CollisionPage,
    CollisionPageCapacityError,
    CollisionPageLimits,
    PagedWorldCollisionWorkspace,
)
from .geometry.particle_sphere import ParticleSphereGeometry, particle_sphere_geometry
from .geometry.plane import Plane
from .geometry.robot_patch_sweep import RobotPatchSweepResult, robot_patch_sweep
from .geometry.robot_sphere import RobotSphereAttachments
from .geometry.rotating_mesh import (
    MeshArcCandidates,
    MeshArcFacePair,
    MeshArcWindow,
    mesh_arc_face_candidates,
)
from .geometry.sdf_grid import SdfSample, SignedDistanceGrid, bake_mesh_sdf
from .geometry.sdf_material import (
    SdfContactBatch,
    SdfFieldPose,
    SdfMaterialResult,
    SdfMaterialSample,
    sample_sdf_contacts,
)
from .geometry.sphere import SphereShapes
from .geometry.surface_features import (
    ClosestFeaturePair,
    closest_point_triangle,
    closest_segment_segment,
)
from .runtime.finite_contact_terminal import FiniteContactTerminalProgram
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
from .sensors import (
    ContactSelector,
    ContactWrenchReading,
    RelativeSiteMeasurement,
    RobotIMUReading,
    RobotSite,
    measure_cloth_link_wrench,
    measure_contact_wrench,
    measure_relative_site,
    measure_robot_imu,
)
from .solvers.articulated_coupling import ArticulatedCouplingProgram, make_articulated_runtime
from .solvers.cloth import ClothProgram, make_cloth_runtime
from .solvers.cloth_link_contact import ClothLinkMaterial
from .solvers.collision_loads import (
    CollisionLoadSystem,
    NormalLayer,
    ParticleAttachment,
    RigidAttachment,
)
from .solvers.compliant_cloth_contact import CompliantClothContactProgram, CompliantPatch
from .solvers.compliant_robot_contact import CompliantRobotContactProgram
from .solvers.coupled_program import (
    CoupledRuntimeAdapter,
    NumpyCoupledProgram,
    make_coupled_runtime,
)
from .solvers.coupled_xpbd import CoupledXPBDSolver
from .solvers.driven_robot import DrivenRobotProgram
from .solvers.dynamic_robot_finite_contact import DynamicRobotFiniteContactProgram
from .solvers.dynamic_robot_friction_contact import DynamicRobotFrictionContactProgram
from .solvers.dynamic_robot_normal_contact import DynamicRobotNormalContactProgram
from .solvers.floating_tree import (
    FloatingDynamicsTerms,
    FloatingRobotInputs,
    floating_dynamics_terms,
    step_floating_robot,
)
from .solvers.free_root import step_free_root
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
from .solvers.robot_closure import (
    RobotClosureReaction,
    RobotClosureStep,
    RobotPointClosure,
    step_robot_closures,
)
from .solvers.robot_plane_contact import RobotPlaneContactProgram
from .solvers.semi_implicit import SemiImplicitSolver
from .solvers.tetrahedral import TetrahedralProgram, make_tetrahedral_runtime
from .solvers.xpbd import XPBDSolver
from .validation.finite_contact_terminal import FiniteContactTerminalLedger
from .validation.free_particle import FreeParticleVerifier

__version__ = "0.1.0a0"

from .constraints.robot_contact_patch import RobotContactPatch
from .constraints.triangle_prism_surface import TrianglePrismSurface
from .geometry.prism_patch_manifold import PrismPatchManifold, triangle_prism_patch_manifold
from .geometry.prism_patch_sweep import PrismPatchSweepResult, triangle_prism_patch_sweep
from .solvers.dynamic_robot_thick_contact import DynamicRobotThickContactProgram
from .validation.material_path import MaterialPathLedger
from .validation.terminal_impulse import TerminalImpulseLedger
from .validation.thick_contact_mapping import ThickContactMappingLedger

__all__ = [
    "RobotPlaneCollider",
    "DynamicRobotNormalContactProgram",
    "DynamicRobotFrictionContactProgram",
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
    "CompliantClothContactProgram",
    "CompliantPatch",
    "ClothLinkMaterial",
    "PointIKResult",
    "JointLimits",
    "CollisionPlane",
    "LinkCapsule",
    "RobotContactMaterial",
    "RobotContactSystem",
    "CompliantRobotContactProgram",
    "solve_point_ik",
    "MultiSiteIKResult",
    "SiteIKResidual",
    "SiteIKTarget",
    "solve_multi_site_ik",
    "make_tetrahedral_runtime",
    "make_cloth_runtime",
    "tetrahedral_block",
    "FreeSphere",
    "PrismaticSlider",
    "TyxWorldBatch",
    "WorldCheckpoint",
    "ClothLinkWorld",
    "TyxWorldContactBatch",
    "WorldContactResult",
    "WorldContactState",
    "RobotPointClosure",
    "RobotClosureReaction",
    "RobotClosureStep",
    "step_robot_closures",
    "FloatingKinematics",
    "FloatingRobotModel",
    "FloatingRobotState",
    "floating_kinematics",
    "floating_point_jacobians",
    "FloatingDynamicsTerms",
    "FloatingRobotInputs",
    "floating_dynamics_terms",
    "step_floating_robot",
    "step_free_root",
    "RobotModel",
    "RobotState",
    "RobotInputs",
    "RobotPointPort",
    "FloatingRobotPointPort",
    "solve_witness_impulses",
    "RobotSite",
    "RelativeSiteMeasurement",
    "measure_relative_site",
    "ContactSelector",
    "ContactWrenchReading",
    "measure_cloth_link_wrench",
    "measure_contact_wrench",
    "RobotIMUReading",
    "measure_robot_imu",
    "RobotProgram",
    "DrivenRobotProgram",
    "make_robot_runtime",
    "pendulum_chain",
    "load_urdf",
    "load_mjcf",
    "load_usd",
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


__all__ += ["RobotFinitePatchCollider", "RobotPatchSweepResult", "robot_patch_sweep"]

__all__ += ["FiniteContactTerminalProgram", "FiniteContactTerminalLedger"]

__all__ += ["DynamicRobotFiniteContactProgram"]

__all__ += [
    "TrianglePrismSurface",
    "RobotContactPatch",
    "PrismPatchManifold",
    "triangle_prism_patch_manifold",
]

__all__ += ["PrismPatchSweepResult", "triangle_prism_patch_sweep"]

__all__ += ["DynamicRobotThickContactProgram"]

__all__ += ["ThickContactMappingLedger"]

__all__ += [
    "CollisionBounds",
    "CollisionDetection",
    "CollisionKey",
    "CollisionLoadSystem",
    "TyxNormalLayer",
    "NormalLayer",
    "ParticleAttachment",
    "RigidAttachment",
    "CollisionModel",
    "CollisionMotion",
    "CollisionPair",
    "CollisionPolicy",
    "CollisionShape",
    "CurvedSolid",
    "SdfSample",
    "SdfContactBatch",
    "SdfFieldPose",
    "SdfMaterialResult",
    "SdfMaterialSample",
    "sample_sdf_contacts",
    "SignedDistanceGrid",
    "TyxSdfSampler",
    "bake_mesh_sdf",
    "CurvedContact",
    "CurvedEvent",
    "ConvexSupportTrace",
    "CollisionSweep",
    "CollisionWitness",
    "CollisionWorkspace",
    "WorldCollisionWorkspace",
    "CollisionPage",
    "CollisionPageCapacityError",
    "CollisionPageLimits",
    "PagedWorldCollisionWorkspace",
    "Heightfield",
    "HeightfieldEvent",
    "HeightfieldFace",
    "HeightfieldTrace",
    "ConvexContactPatch",
    "ConvexContactPoint",
    "ConvexContactPose",
    "ConvexHullTrace",
    "HullSurfacePose",
    "HullSurfaceTrace",
    "MaterialContactPoint",
    "MeshContactPose",
    "MeshContactTrace",
    "MeshMeshPose",
    "MeshMeshSupport",
    "MeshMeshTrace",
    "MeshRoundEvent",
    "MeshRoundPose",
    "MeshRoundSupport",
    "MeshRoundTrace",
    "MeshArcCandidates",
    "MeshArcFacePair",
    "MeshArcWindow",
    "mesh_arc_face_candidates",
    "MeshArcSupport",
    "MeshArcBoundaryEvent",
    "MeshArcSurfaceResult",
    "mesh_arc_first_surface",
    "PlateContactPose",
    "PlateContactTrace",
    "CapsuleAxisPoint",
    "TetrahedralFacetResult",
    "TetrahedralBodyPair",
    "TetrahedralBoundaryContact",
    "TetrahedralBoundaryTrace",
    "TetrahedralOccupancy",
    "TetrahedralSolidResult",
    "RigidTurn",
    "detect",
]
