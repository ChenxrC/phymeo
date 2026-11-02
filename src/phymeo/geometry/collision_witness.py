"""World-space geometry and material addresses from a collision query."""

from dataclasses import dataclass
from fractions import Fraction

from ..constraints.collision_shapes import CollisionKey


@dataclass(frozen=True)
class MaterialContactPoint:
    """A world point on both surfaces, addressed separately on each pair owner."""

    point: tuple[float, ...]
    first_weights: tuple[tuple[int, float], ...]
    second_weights: tuple[tuple[int, float], ...]
    first_height_m: float | None = None
    second_height_m: float | None = None


@dataclass(frozen=True)
class CapsuleAxisPoint:
    """One extreme closest pair in the capsule parameter and the triangle's material chart."""

    axis_parameter: float
    axis_point: tuple[float, ...]
    face_point: tuple[float, ...]
    face_weights: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class CollisionWitness:
    """One representative with the normal oriented from the first shape to the second."""

    gap_m: float
    normal: tuple[float, ...] | None
    first_point: tuple[float, ...] | None
    second_point: tuple[float, ...] | None
    first_weights: tuple[tuple[int, float], ...]
    second_weights: tuple[tuple[int, float], ...]
    source: str = "host-fp64"
    # Endpoints of the complete closest (first, second) axis-parameter interval
    # for two nonzero capsules. A unique minimum repeats the same pair.
    axis_parameter_span: tuple[tuple[float, float], tuple[float, float]] | None = None
    # World-frame tangent displacement from the moving anchor of an infinite plane.
    first_offset: tuple[float, ...] | None = None
    second_offset: tuple[float, ...] | None = None
    # The convex hull of both groups is the sampled triangle intersection.
    # Only intersection_points have distinct material preimages.
    intersection_points: tuple[MaterialContactPoint, ...] = ()
    shared_material_points: tuple[MaterialContactPoint, ...] = ()

    # Signed material height along the middle triangle's oriented normal.
    first_height_m: float | None = None
    second_height_m: float | None = None

    # Ordered along the capsule, regardless of the shape-pair order.
    capsule_axis_span: tuple[CapsuleAxisPoint, CapsuleAxisPoint] | None = None

    @property
    def shared_only(self):
        return bool(self.shared_material_points) and not self.intersection_points


@dataclass(frozen=True)
class CollisionSweep:
    """Closed first-contact time bracket on [0,1]; pair.witness is the endpoint witness."""

    start_gap_m: float
    end_gap_m: float
    first_t: tuple[float, float] | None
    witness_time: float | None
    witness: CollisionWitness | None
    # For an isolated incident hit the representable sample time can have only
    # adjacency. These separate material points are evaluated at that same time.
    feature_witness: CollisionWitness | None = None


@dataclass(frozen=True)
class TetrahedralFacetResult:
    """One outward cell face. An exact event is distinct from the sweep's sampled pose."""

    face_ids: tuple[int, int, int]
    witness: CollisionWitness
    sweep: CollisionSweep | None
    event_time_ratio: tuple[int, int] | None
    event_witness: CollisionWitness | None


@dataclass(frozen=True)
class TetrahedralOccupancy:
    """Intersection of filled cells or their conforming union at one pose."""

    relation: str
    volume_ratio: tuple[int, int]
    owners: tuple[CollisionKey, CollisionKey] | None
    witness: MaterialContactPoint | None

    @property
    def volume_m3(self):
        return float(Fraction(*self.volume_ratio))


@dataclass(frozen=True)
class TetrahedralBoundaryContact:
    """A representative pair of exterior faces in original outward material order."""

    first_face: tuple[int, int, int]
    second_face: tuple[int, int, int]
    witness: CollisionWitness


@dataclass(frozen=True)
class TetrahedralBoundaryTrace:
    start: TetrahedralBoundaryContact
    end: TetrahedralBoundaryContact
    first_t: tuple[float, float] | None
    witness_time: float | None
    witness: TetrahedralBoundaryContact | None
    event_time_ratio: tuple[int, int] | None
    event_witness: TetrahedralBoundaryContact | None


@dataclass(frozen=True)
class TetrahedralSolidResult:
    start: TetrahedralOccupancy
    end: TetrahedralOccupancy
    boundary: TetrahedralBoundaryTrace | None


@dataclass(frozen=True)
class TetrahedralBodyPair:
    world: int
    first_body: int
    second_body: int
    status: str
    start: TetrahedralOccupancy | None = None
    end: TetrahedralOccupancy | None = None
    first_t: tuple[float, float] | None = None
    boundary_first_t: tuple[float, float] | None = None


@dataclass(frozen=True)
class ConvexContactPoint:
    """Paired material support points, coincident when the solids just touch."""

    first_point: tuple[float, ...]
    second_point: tuple[float, ...]
    first_weights: tuple[tuple[int, float], ...]
    second_weights: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class ConvexContactPatch:
    normal: tuple[float, ...]
    first_support: tuple[int, ...]
    second_support: tuple[int, ...]
    dimension: int
    area_m2: float
    points: tuple[ConvexContactPoint, ...]


@dataclass(frozen=True)
class ProximityWitness(CollisionWitness):
    """Complete finite closest patch when explicitly requested for a separated pair."""

    proximity_patch: ConvexContactPatch | None = None


@dataclass(frozen=True)
class ConvexContactPose:
    relation: str
    witness: CollisionWitness
    patches: tuple[ConvexContactPatch, ...]


@dataclass(frozen=True)
class ConvexHullTrace:
    """Start/end geometry and both exact endpoints of the closed contact window."""

    start: ConvexContactPose
    end: ConvexContactPose
    first_time_ratio: tuple[int, int] | None
    last_time_ratio: tuple[int, int] | None
    first: ConvexContactPose | None
    last: ConvexContactPose | None
    axis_count: int


@dataclass(frozen=True)
class MeshContactPose:
    """Filled-solid relation and actual exterior-boundary geometry at a single time."""

    relation: str
    boundary_gap_m: float
    witness: CollisionWitness
    facets: tuple
    contacts: tuple[MaterialContactPoint, ...]


@dataclass(frozen=True)
class MeshContactTrace:
    start: MeshContactPose
    end: MeshContactPose
    first_ratio: tuple[int, int] | None
    entry: MeshContactPose | None


@dataclass(frozen=True)
class PlateContactPose:
    """The closest plate feature and complete support patch when its normal is defined."""

    relation: str
    feature: str
    witness: CollisionWitness
    patch: ConvexContactPatch | None


@dataclass(frozen=True)
class PlateContactTrace:
    start: PlateContactPose
    end: PlateContactPose
    entry: PlateContactPose | None


@dataclass(frozen=True)
class HullSurfacePose:
    """Occupied material section of a cloth triangle in a convex solid."""

    relation: str
    area_m2: float
    section: tuple[MaterialContactPoint, ...]
    boundary_faces: tuple
    support_normals: tuple[tuple[float, ...], ...]
    witness: CollisionWitness


@dataclass(frozen=True)
class HullSurfaceTrace:
    start: HullSurfacePose
    end: HullSurfacePose
    entry: HullSurfacePose | None
    exact_first_ratio: tuple[int, int] | None
