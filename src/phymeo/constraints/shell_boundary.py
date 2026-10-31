"""Owned weak-boundary proof data. No topology or numerical proof is performed here."""

import json
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from hashlib import sha256

import numpy as np

from .conforming_shell import ShellProofLimits


def freeze(value, depth=0):
    """Bound container traversal before copying; preserve exact numeric types."""
    if depth > 32:
        raise ValueError("nested evidence depth")
    if type(value) in (tuple, list):
        if len(value) > 8192:
            raise ValueError("nested evidence shape")
        return tuple(freeze(v, depth + 1) for v in value)
    if type(value) is np.ndarray:
        if value.dtype not in (np.dtype("float64"), np.dtype("int64")) or value.size > 65536:
            raise TypeError("bounded FP64/int64 arrays required")
        if not np.isfinite(value).all():
            raise ValueError("nonfinite evidence")
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if is_dataclass(value) or type(value) in (str, int, Fraction) or value is None:
        return value
    raise TypeError("evidence must use exact immutable values; no coercive IDs")


def canonical_data(value):
    if is_dataclass(value):
        return (
            type(value).__name__,
            tuple((f.name, canonical_data(getattr(value, f.name))) for f in fields(value)),
        )
    if type(value) is np.ndarray:
        return (str(value.dtype), value.shape, value.tobytes().hex())
    if type(value) is Fraction:
        return (value.numerator, value.denominator)
    if type(value) in (tuple, list):
        return tuple(canonical_data(v) for v in value)
    return value


def digest(value):
    return sha256(
        json.dumps(canonical_data(value), separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def identity(value):
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("invalid reference identity")


def integer(value, minimum=0, maximum=2**63 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise TypeError("bounded built-in integer required")


def vertex(value):
    if type(value) not in (tuple, list) or len(value) != 2:
        raise TypeError("lifted material vertex required")
    integer(value[0])
    if type(value[1]) is not int or value[1] not in (-1, 1):
        raise TypeError("exact lifted sign required")
    return tuple(value)


def tetra(value):
    if type(value) not in (tuple, list) or len(value) != 2:
        raise TypeError("tetra identity required")
    integer(value[0])
    integer(value[1], 0, 2)
    return tuple(value)


def coefficients(value):
    if type(value) not in (tuple, list) or len(value) not in (1, 2, 4, 8, 16):
        raise TypeError("bounded coefficient-mask tuple required")
    if any(type(q) is not Fraction for q in value):
        raise TypeError("normalized Fraction coefficients required")
    if any(max(abs(q.numerator).bit_length(), q.denominator.bit_length()) > 4096 for q in value):
        raise ValueError("coefficient bit ceiling")
    return tuple(value)


class OwnedRecord:
    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name, freeze(getattr(self, field.name)))


@dataclass(frozen=True)
class BoundaryProofLimits:
    strict_limits: ShellProofLimits = ShellProofLimits()
    max_field_rank: int = 4
    max_polynomial_degree: int = 8
    max_expression_nodes: int = 4096
    max_material_strata: int = 4096
    max_clusters: int = 8
    max_cluster_vertices: int = 16
    max_cluster_boundary_triangles: int = 32
    max_face_polygon_vertices: int = 16
    max_fan_planes: int = 64
    max_support_descents: int = 3
    max_factor_exponent: int = 8

    def __post_init__(self):
        if type(self.strict_limits) is not ShellProofLimits:
            raise TypeError("ShellProofLimits required")
        self.strict_limits.__post_init__()
        for f in fields(self):
            if f.name != "strict_limits":
                integer(getattr(self, f.name), 1, f.default)


@dataclass(frozen=True)
class BoundaryFaceHint(OwnedRecord):
    triangles: tuple
    cycle: tuple
    drop_axis: int

    def __post_init__(self):
        super().__post_init__()
        integer(self.drop_axis, 0, 2)
        if not 1 <= len(self.triangles) <= 128 or not 3 <= len(self.cycle) <= 128:
            raise ValueError("bounded complete face hint required")
        for f in self.triangles:
            if len(f) != 3:
                raise ValueError("triangle cardinality")
            for v in f:
                vertex(v)
        for v in self.cycle:
            vertex(v)


@dataclass(frozen=True)
class ClusterHint(OwnedRecord):
    cluster_id: int
    tetra_ids: tuple
    faces: tuple

    def __post_init__(self):
        super().__post_init__()
        integer(self.cluster_id)
        if not 1 <= len(self.tetra_ids) <= 128 or not 1 <= len(self.faces) <= 256:
            raise ValueError("cluster hint shape")
        for t in self.tetra_ids:
            tetra(t)
        if any(type(f) is not BoundaryFaceHint for f in self.faces):
            raise TypeError("face hint type")


@dataclass(frozen=True)
class SupportHint(OwnedRecord):
    normal: tuple
    anchor: tuple

    def __post_init__(self):
        super().__post_init__()
        vertex(self.anchor)
        if len(self.normal) != 3:
            raise ValueError("three-component normal required")
        for c in self.normal:
            coefficients(c)


@dataclass(frozen=True)
class ClusterPairHint(OwnedRecord):
    clusters: tuple
    supports: tuple
    matching: tuple

    def __post_init__(self):
        super().__post_init__()
        if len(self.clusters) != 2 or not 1 <= len(self.supports) <= 3 or len(self.matching) > 32:
            raise ValueError("pair hint shape")
        for i in self.clusters:
            integer(i)
        if any(type(s) is not SupportHint for s in self.supports):
            raise TypeError("support hint type")
        for pair in self.matching:
            if len(pair) != 2:
                raise ValueError("matching cardinality")
            vertex(pair[0])
            vertex(pair[1])


@dataclass(frozen=True)
class BoundaryWitnessHint(OwnedRecord):
    generators: tuple
    clusters: tuple
    pairs: tuple
    exterior: tuple

    def __post_init__(self):
        super().__post_init__()
        if len(self.generators) > 4 or not 1 <= len(self.clusters) <= 8 or len(self.pairs) > 28:
            raise ValueError("witness hint shape")
        if any(type(v) is not Fraction or v <= 0 for v in self.generators):
            raise TypeError("positive rational generators required")
        if any(type(v) is not ClusterHint for v in self.clusters):
            raise TypeError("cluster hint type")
        if any(type(v) is not ClusterPairHint for v in self.pairs):
            raise TypeError("pair hint type")
        if len(self.exterior) > 4096:
            raise ValueError("exterior shape")
        for s, direction in self.exterior:
            if not 1 <= len(s) <= 3 or len(direction) != 3:
                raise ValueError("exterior shape")
            for v in s:
                vertex(v)
            for c in direction:
                coefficients(c)


@dataclass(frozen=True)
class ExactBoundaryAddress(OwnedRecord):
    reference_id: str
    generation: int
    minimal_simplex: tuple
    weights: tuple
    representatives: tuple
    generators: tuple = ()

    def __post_init__(self):
        super().__post_init__()
        identity(self.reference_id)
        integer(self.generation)
        if not 1 <= len(self.minimal_simplex) <= 4 or len(self.weights) != len(
            self.minimal_simplex
        ):
            raise ValueError("address shape")
        if len(self.representatives) > 128 or len(self.generators) > 4:
            raise ValueError("address bounded representatives/basis")
        for v in self.minimal_simplex:
            vertex(v)
        for c in self.weights:
            coefficients(c)
        for t, weights in self.representatives:
            tetra(t)
            if len(weights) != 4:
                raise ValueError("four exact tetra weights required")
            for c in weights:
                coefficients(c)
        if any(type(q) is not Fraction or q <= 0 for q in self.generators):
            raise TypeError("positive exact address generators required")


@dataclass(frozen=True)
class BoundaryFieldWitness(OwnedRecord):
    generators: tuple
    radicands: tuple
    dependencies: tuple
    coefficient_order: tuple
    sqrt_bits: int


@dataclass(frozen=True)
class BoundaryTopology(OwnedRecord):
    nodes: tuple
    faces: tuple
    tetrahedra: tuple
    boundary_facets: tuple
    components: tuple
    strata: tuple
    reference_facts: tuple


@dataclass(frozen=True)
class BoundaryLaw(OwnedRecord):
    areas: tuple
    stars: tuple
    squared_norms: tuple
    normals: tuple
    gates: tuple


@dataclass(frozen=True)
class BoundaryPlaneEvidence(OwnedRecord):
    vertices: tuple
    triangles: tuple
    cycle: tuple
    drop_axis: int
    origin: tuple
    normal: tuple
    supports: tuple
    orientations: tuple
    edge_tests: tuple


@dataclass(frozen=True)
class ConvexClusterBoundaryWitness(OwnedRecord):
    cluster_id: int
    tetra_ids: tuple
    boundary_facets: tuple
    planes: tuple
    determinants: tuple
    links: tuple


@dataclass(frozen=True)
class BoundarySupportEvidence(OwnedRecord):
    normal: tuple
    anchor: tuple
    before_a: tuple
    before_b: tuple
    values_a: tuple
    values_b: tuple
    zero_a: tuple
    zero_b: tuple


@dataclass(frozen=True)
class MatchingIntersectionWitness(OwnedRecord):
    clusters: tuple
    supports: tuple
    matching: tuple
    intersection_simplices: tuple
    classification: str
    oriented_incidence: tuple = ()


@dataclass(frozen=True)
class BoundaryPairWitness(OwnedRecord):
    tetra_pair: tuple
    shared_reference_simplex: tuple
    cluster_pair: tuple
    intersection_simplices: tuple
    classification: str


@dataclass(frozen=True)
class BoundaryFanWitness(OwnedRecord):
    material_stratum: tuple
    incident_tetrahedra: tuple
    true_boundary_facets: tuple
    occupied_cones: tuple
    exterior_direction: tuple
    exterior_facts: tuple


@dataclass(frozen=True)
class BoundaryContactStratum(OwnedRecord):
    owners: tuple
    exact_vertices: tuple
    subfaces: tuple
    boundary_stars: tuple
    support_ties: tuple
    occupied_cones: tuple


@dataclass(frozen=True)
class BoundaryPointBasis(OwnedRecord):
    vertex: tuple
    node_ids: tuple
    exact_point: tuple
    exact_jacobian: tuple
    point: object
    jacobian: object
    representatives: tuple


@dataclass(frozen=True)
class ShellBoundaryAdmission(OwnedRecord):
    reference_id: str
    original_input_binding: str
    policy: BoundaryProofLimits
    original_positions: np.ndarray
    field: BoundaryFieldWitness
    topology: BoundaryTopology
    law: BoundaryLaw
    clusters: tuple
    cluster_pairs: tuple
    all_pairs: tuple
    full_strata_and_fans: tuple
    contacts: tuple
    basis: tuple
    operations: int
    producer_operations: int
    checker_operations: int
    peak_fraction_bits: int
    schema: str = "shell-boundary-v1"
    status: str = "weak_material_geometry_certified"
    precision: str = "ideal-from-stored-fp64"
    proof_kind: str = "convex-cluster-matching-v1"


@dataclass(frozen=True)
class ShellExactBoundaryPoint(OwnedRecord):
    reference_id: str
    original_input_binding: str
    policy: BoundaryProofLimits
    original_positions: np.ndarray
    address: ExactBoundaryAddress
    field: BoundaryFieldWitness
    node_ids: tuple
    exact_point: tuple
    exact_jacobian: tuple
    point: object
    jacobian: object
    reference_facts: tuple
    law: BoundaryLaw
    operations: int
    producer_operations: int
    checker_operations: int
    peak_fraction_bits: int
    schema: str = "shell-exact-boundary-point-v1"
    status: str = "exact_boundary_map_only"
    precision: str = "ideal-from-stored-fp64"


@dataclass(frozen=True)
class BoundaryTimeCell(OwnedRecord):
    span: tuple
    depth: int
    children: tuple
    polynomial_facts: tuple
    sqrt_bits: int


@dataclass(frozen=True)
class BoundarySegmentEvidence(OwnedRecord):
    original_segment: np.ndarray
    duration: Fraction
    star_polynomials: tuple
    cross_identities: tuple
    positive_dot_bounds: tuple
    component_translations: tuple
    lifted_paths: tuple
    plane_paths: tuple
    geometry_paths: tuple
    pair_paths: tuple
    cells: tuple
    endpoint_points: tuple
    endpoint_contacts: tuple
    map_functions: tuple
    sqrt_bits: int


@dataclass(frozen=True)
class ShellBoundarySweep(OwnedRecord):
    reference_id: str
    original_input_binding: str
    policy: BoundaryProofLimits
    durations: tuple
    base: ShellBoundaryAdmission
    segments: tuple
    operations: int
    producer_operations: int
    checker_operations: int
    created_cells: int
    peak_fraction_bits: int
    schema: str = "shell-boundary-sweep-v1"
    status: str = "weak_material_path_certified"
    precision: str = "ideal-from-stored-fp64"
    proof_kind: str = "convex-cluster-matching-v1"
    path_kind: str = "constant-derived-director-v1"


@dataclass(frozen=True)
class BoundaryVerificationReceipt(OwnedRecord):
    verification_kind: str
    reference_id: str
    original_input_binding: str
    verified_certificate_binding: str
    policy: BoundaryProofLimits
    status: str
    precision: str
    operations: int
    created_cells: int
    peak_fraction_bits: int
    schema: str = "shell-boundary-verification-v1"

    def __post_init__(self):
        super().__post_init__()
        if self.verification_kind not in ("static", "sweep", "point"):
            raise ValueError("verification kind")
        identity(self.reference_id)
        identity(self.original_input_binding)
        identity(self.verified_certificate_binding)
        if type(self.policy) is not BoundaryProofLimits:
            raise TypeError("immutable verification policy")
        self.policy.__post_init__()
        expected = {
            "static": "weak_material_geometry_certified",
            "sweep": "weak_material_path_certified",
            "point": "exact_boundary_map_only",
        }
        if (
            self.status != expected[self.verification_kind]
            or self.precision != "ideal-from-stored-fp64"
        ):
            raise ValueError("verification scope")
        integer(self.operations, 1, self.policy.strict_limits.max_scalar_operations)
        integer(self.created_cells, 0, self.policy.strict_limits.max_time_cells)
        integer(self.peak_fraction_bits, 1, self.policy.strict_limits.max_fraction_bits)

    def __bool__(self):
        return True
