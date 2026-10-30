"""Owned conforming simplicial material identity; construction does not embed it."""

from dataclasses import dataclass, fields
from fractions import Fraction
from hashlib import sha256
import json

import numpy as np


class ShellUnresolved(ValueError):
    """A bounded rejected obligation, never a partially accepted geometry/state."""

    def __init__(self, category, detail=""):
        self.category = str(category)[:80]
        self.detail = str(detail)[:256]
        super().__init__(f"{self.category}: {self.detail}")


@dataclass(frozen=True)
class ShellDirectorLaw:
    kind: str = "raw-area-star-director-v1"
    projected_ratio_min: Fraction = Fraction(1, 4)
    shape_policy: str = "invariant-1e-10-v1"

    def __post_init__(self):
        if (self.kind != "raw-area-star-director-v1"
                or type(self.projected_ratio_min) is not Fraction
                or self.projected_ratio_min != Fraction(1, 4)
                or self.shape_policy != "invariant-1e-10-v1"):
            raise ValueError("unsupported shell director law")


@dataclass(frozen=True)
class ShellProofLimits:
    max_nodes: int = 64
    max_faces: int = 40
    max_tetrahedra: int = 128
    max_star_faces: int = 16
    max_pairs: int = 8192
    max_time_cells: int = 4096
    max_time_depth: int = 32
    max_fraction_bits: int = 4096
    max_scalar_operations: int = 2000000
    initial_sqrt_bits: int = 64
    max_sqrt_bits: int = 4096

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int:
                raise TypeError(f"{field.name} must be a built-in integer")
            if not 0 < value <= field.default:
                raise ValueError(f"{field.name} exceeds its positive hard ceiling")
        if self.initial_sqrt_bits > self.max_sqrt_bits:
            raise ValueError("initial sqrt precision exceeds maximum")


@dataclass(frozen=True)
class ShellMaterialAddress:
    reference_id: str
    tetra_id: tuple
    counts: tuple
    generation: int

    def __post_init__(self):
        if (type(self.reference_id) is not str or len(self.reference_id) != 64
                or any(c not in "0123456789abcdef" for c in self.reference_id)):
            raise ValueError("invalid reference identity")
        if not isinstance(self.tetra_id, (tuple, list)) or len(self.tetra_id) != 2:
            raise TypeError("tetra_id must have two exact integers")
        if any(type(v) is not int or v < 0 for v in self.tetra_id):
            raise TypeError("invalid tetra_id")
        if self.tetra_id[1] > 2:
            raise ValueError("invalid staircase index")
        if not isinstance(self.counts, (tuple, list)) or len(self.counts) != 4:
            raise TypeError("counts must have four exact integers")
        if any(type(v) is not int for v in self.counts):
            raise TypeError("counts must be built-in integers, not bool")
        if any(v < 0 or v > 2**52 for v in self.counts) or sum(self.counts) != 2**52:
            raise ValueError("counts must be nonnegative and sum to 2^52")
        if type(self.generation) is not int or self.generation < 0 or self.generation.bit_length() > 63:
            raise ValueError("invalid generation")
        object.__setattr__(self, "tetra_id", tuple(self.tetra_id))
        object.__setattr__(self, "counts", tuple(self.counts))


@dataclass(frozen=True)
class ShellTetrahedron:
    tetra_id: tuple
    vertices: tuple


@dataclass(frozen=True)
class ShellBoundaryFacet:
    vertices: tuple
    owner: tuple


def _cyclic(values):
    values = tuple(values)
    return min(values[i:] + values[:i] for i in range(len(values)))


def _parity(values):
    return -1 if sum(values[i] > values[j] for i in range(len(values)) for j in range(i + 1, len(values))) % 2 else 1


def _owned(array):
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _digest(value):
    return sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _array(array, dtype, shape, name):
    if type(array) is not np.ndarray or array.dtype != np.dtype(dtype) or array.shape != shape:
        raise TypeError(f"{name} must be an ndarray of exact dtype {dtype} and shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} is nonfinite")


def _topology(node_ids, face_ids, oriented_nodes, limits):
    nodes = tuple(sorted(int(v) for v in node_ids))
    faces = tuple(sorted((int(i), tuple(int(v) for v in face)) for i, face in zip(face_ids, oriented_nodes, strict=True)))
    known = set(nodes)
    seen = set()
    stars = {i: [] for i in nodes}
    edges = {}
    neighbors = {i: set() for i in nodes}
    for face_id, face in faces:
        if len(set(face)) != 3 or not set(face) <= known:
            raise ShellUnresolved("invalid_reference_topology", "unknown/repeated face node")
        key = tuple(sorted(face))
        if key in seen:
            raise ShellUnresolved("invalid_reference_topology", "duplicate unoriented face")
        seen.add(key)
        for node in face:
            stars[node].append(face_id)
            neighbors[node].update(set(face) - {node})
        for a, b in zip(face, face[1:] + face[:1], strict=True):
            entries = edges.setdefault(tuple(sorted((a, b))), [])
            entries.append((a, b))
            if len(entries) > 2:
                raise ShellUnresolved("invalid_reference_topology", "nonmanifold edge")
    for entries in edges.values():
        if len(entries) == 2 and entries[0] != entries[1][::-1]:
            raise ShellUnresolved("invalid_reference_topology", "inconsistent winding")
    for node in nodes:
        if not stars[node]:
            raise ShellUnresolved("invalid_reference_topology", "unused node")
        if len(stars[node]) > limits.max_star_faces:
            raise ShellUnresolved("shape_budget", "star faces")
        link = {}
        for _, face in faces:
            if node not in face:
                continue
            a, b = (v for v in face if v != node)
            link.setdefault(a, set()).add(b)
            link.setdefault(b, set()).add(a)
        visited, pending = set(), [next(iter(link))]
        while pending:
            v = pending.pop()
            if v not in visited:
                visited.add(v)
                pending.extend(link[v] - visited)
        boundary = any(node in edge and len(entries) == 1 for edge, entries in edges.items())
        degrees = sorted(len(v) for v in link.values())
        correct = (degrees.count(1) == 2 and all(v in (1, 2) for v in degrees)) if boundary else all(v == 2 for v in degrees)
        if len(visited) != len(link) or not correct:
            raise ShellUnresolved("invalid_reference_topology", "vertex link is not one interval/cycle")
    components = []
    unseen = set(nodes)
    while unseen:
        pending, component = [min(unseen)], set()
        while pending:
            node = pending.pop()
            if node not in component:
                component.add(node)
                pending.extend(neighbors[node] - component)
        unseen.difference_update(component)
        components.append(tuple(sorted(component)))
    cycles = []
    outgoing = {}
    for entries in edges.values():
        if len(entries) == 1:
            a, b = entries[0]
            if a in outgoing:
                raise ShellUnresolved("invalid_reference_topology", "branched boundary")
            outgoing[a] = b
    remaining = set(outgoing)
    while remaining:
        start = min(remaining)
        cycle, node = [], start
        while node not in cycle:
            if node not in outgoing:
                raise ShellUnresolved("invalid_reference_topology", "open boundary chain")
            cycle.append(node)
            node = outgoing[node]
        if node != start:
            raise ShellUnresolved("invalid_reference_topology", "boundary repeats")
        remaining.difference_update(cycle)
        cycles.append(_cyclic(cycle))
    tetrahedra = []
    facet_entries = {}
    for face_id, face in faces:
        i, j, k = sorted(face)
        epsilon = _parity(face)
        triples = (((i, -1), (j, -1), (k, -1), (k, 1)),
                   ((i, -1), (j, -1), (j, 1), (k, 1)),
                   ((i, -1), (i, 1), (j, 1), (k, 1)))
        for stair, vertices in enumerate(triples):
            if epsilon * (1, -1, 1)[stair] < 0:
                vertices = vertices[:2] + (vertices[3], vertices[2])
            tet = ShellTetrahedron((face_id, stair), vertices)
            tetrahedra.append(tet)
            for omitted in range(4):
                facet = tuple(v for index, v in enumerate(vertices) if index != omitted)
                if omitted % 2:
                    facet = (facet[0], facet[2], facet[1])
                facet_entries.setdefault(tuple(sorted(facet)), []).append((_cyclic(facet), tet.tetra_id))
    boundary_facets = []
    for key, entries in sorted(facet_entries.items()):
        if len(entries) == 1:
            boundary_facets.append(ShellBoundaryFacet(*entries[0]))
        elif len(entries) != 2 or _parity(entries[0][0]) == _parity(entries[1][0]):
            raise ShellUnresolved("nonconforming_product", "internal facets do not cancel")
    return tuple(tetrahedra), tuple(boundary_facets), tuple(components), tuple(sorted(cycles))


@dataclass(frozen=True, init=False)
class ConformingShellReference:
    domain: str
    node_ids: np.ndarray
    face_ids: np.ndarray
    oriented_nodes: np.ndarray
    reference_positions: np.ndarray
    half_thickness_m: np.ndarray
    law: ShellDirectorLaw
    limits: ShellProofLimits
    tetrahedra: tuple
    boundary_facets: tuple
    components: tuple
    boundary_cycles: tuple
    reference_id: str
    input_fingerprint: str

    def __init__(self, domain, node_ids, face_ids, oriented_nodes, reference_positions, half_thickness_m, *, law=ShellDirectorLaw(), limits=ShellProofLimits()):
        if type(domain) is not str or not domain or len(domain.encode("utf-8")) > 256:
            raise ValueError("invalid shell domain")
        if type(law) is not ShellDirectorLaw or type(limits) is not ShellProofLimits:
            raise TypeError("owned shell law/limits required")
        law.__post_init__()
        limits.__post_init__()
        if type(node_ids) is not np.ndarray or node_ids.ndim != 1 or type(face_ids) is not np.ndarray or face_ids.ndim != 1:
            raise TypeError("rank-one node/face ID arrays required")
        n, f = len(node_ids), len(face_ids)
        if not 3 <= n <= limits.max_nodes or not 1 <= f <= limits.max_faces or 3*f > limits.max_tetrahedra or (3*f)*(3*f-1)//2 > limits.max_pairs:
            raise ShellUnresolved("shape_budget", "nodes/faces/tetrahedra/pairs")
        for value, dtype, shape, name in ((node_ids, "int64", (n,), "node_ids"), (face_ids, "int64", (f,), "face_ids"), (oriented_nodes, "int64", (f, 3), "oriented_nodes"), (reference_positions, "float64", (n, 3), "reference_positions"), (half_thickness_m, "float64", (n,), "half_thickness_m")):
            _array(value, dtype, shape, name)
        if np.any(node_ids < 0) or np.any(face_ids < 0) or len(set(node_ids.tolist())) != n or len(set(face_ids.tolist())) != f:
            raise ValueError("IDs must be distinct nonnegative int64")
        if np.any(half_thickness_m <= 0):
            raise ValueError("half thickness must be positive")
        tetrahedra, facets, components, cycles = _topology(node_ids, face_ids, oriented_nodes, limits)
        canonical_nodes = sorted((int(node), reference_positions[row].tobytes().hex(), half_thickness_m[row:row+1].tobytes().hex()) for row, node in enumerate(node_ids))
        canonical_faces = sorted((int(face_id), _cyclic(tuple(int(v) for v in face))) for face_id, face in zip(face_ids, oriented_nodes, strict=True))
        identity = _digest(("conforming-shell-reference-v1", domain, canonical_nodes, canonical_faces, law.kind, (1, 4), law.shape_policy))
        order = _digest((identity, node_ids.tobytes().hex(), face_ids.tobytes().hex(), oriented_nodes.tobytes().hex()))
        values = dict(domain=domain, node_ids=_owned(node_ids), face_ids=_owned(face_ids), oriented_nodes=_owned(oriented_nodes), reference_positions=_owned(reference_positions), half_thickness_m=_owned(half_thickness_m), law=law, limits=limits, tetrahedra=tetrahedra, boundary_facets=facets, components=components, boundary_cycles=cycles, reference_id=identity, input_fingerprint=order)
        for name, value in values.items():
            object.__setattr__(self, name, value)