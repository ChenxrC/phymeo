"""Complete analytic raw-area-star director maps, with owned FP64 proposals."""

from dataclasses import dataclass, fields
from fractions import Fraction as F
from hashlib import sha256
import json

import numpy as np

from ..constraints.conforming_shell import (
    ConformingShellReference,
    ShellMaterialAddress,
    ShellProofLimits,
    ShellUnresolved,
)
from ._shell_intervals import Arithmetic, Publication


@dataclass(frozen=True)
class DomainWitness:
    faces: tuple
    stars: tuple
    corners: tuple


@dataclass(frozen=True)
class ShellDirectorGeometry:
    schema: str
    precision: str
    status: str
    reference_id: str
    binding: str
    policy: ShellProofLimits
    bits: int
    node_ids: tuple
    face_ids: tuple
    areas: tuple
    sums: tuple
    directors: tuple
    offsets: tuple
    lifted: tuple
    domain: DomainWitness
    director_publication: Publication
    lifted_publication: Publication
    operations: int


@dataclass(frozen=True)
class ShellMaterialPoint:
    schema: str
    precision: str
    status: str
    reference_id: str
    binding: str
    policy: ShellProofLimits
    bits: int
    canonical_address: tuple
    generation: int
    node_ids: tuple
    point: Publication
    jacobian: Publication
    domain: DomainWitness
    operations: int


def _binding(kind, reference, array, limits, *, h=None, address=None):
    policy = tuple((field.name, getattr(limits, field.name)) for field in fields(limits))
    data = (kind, reference.reference_id, reference.input_fingerprint, array.dtype.str, array.shape, array.tobytes(order="C").hex(), policy, None if h is None else (h.numerator, h.denominator), None if address is None else (address.reference_id, address.tetra_id, address.counts, address.generation), "ideal-from-stored-fp64")
    return sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def _positions(reference, positions, limits, *, path=False):
    if type(reference) is not ConformingShellReference:
        raise TypeError("owned ConformingShellReference required")
    if type(limits) is not ShellProofLimits:
        raise TypeError("ShellProofLimits required")
    limits.__post_init__()
    n, f = len(reference.node_ids), len(reference.face_ids)
    if n > limits.max_nodes or f > limits.max_faces or len(reference.tetrahedra) > limits.max_tetrahedra:
        raise ShellUnresolved("shape_budget", "reference exceeds call policy")
    shape = (2, n, 3) if path else (n, 3)
    if type(positions) is not np.ndarray or positions.dtype != np.dtype("float64") or positions.shape != shape or not np.isfinite(positions).all():
        raise ShellUnresolved("invalid_input", "finite exact-FP64 state/path shape required")
    return np.frombuffer(positions.tobytes(), dtype=np.float64).reshape(shape)


def _path_polynomials(reference, path, arithmetic):
    result = {}
    for row, node in enumerate(reference.node_ids):
        vector = []
        for coordinate in range(3):
            start = arithmetic.input_float(path[0, row, coordinate])
            end = arithmetic.input_float(path[1, row, coordinate])
            vector.append(arithmetic.poly((start, arithmetic.qsub(end, start))))
        result[int(node)] = tuple(vector)
    return result


def _constant_polynomials(reference, positions, arithmetic):
    return {int(node): tuple((arithmetic.input_float(v),) for v in positions[row]) for row, node in enumerate(reference.node_ids)}


def _face_polynomials(reference, positions, arithmetic):
    faces = tuple(sorted((int(i), tuple(int(v) for v in face)) for i, face in zip(reference.face_ids, reference.oriented_nodes, strict=True)))
    areas, sums, h_vectors, edges = {}, {}, {}, {}
    zero = ((F(0),),) * 3
    for node in positions:
        sums[node] = zero
        h_vectors[node] = {}
    for face_id, (a, b, c) in faces:
        ab = arithmetic.poly_vsub(positions[b], positions[a])
        ac = arithmetic.poly_vsub(positions[c], positions[a])
        bc = arithmetic.poly_vsub(positions[c], positions[b])
        area = arithmetic.poly_cross(ab, ac)
        areas[face_id] = area
        edges[face_id] = (ab, ac, bc)
        vectors = {a: bc, b: tuple(tuple(-v for v in p) for p in ac), c: ab}
        for node in (a, b, c):
            sums[node] = arithmetic.poly_vadd(sums[node], area)
            for neighbor, vector in vectors.items():
                h_vectors[node][neighbor] = arithmetic.poly_vadd(h_vectors[node].get(neighbor, zero), vector)
    return faces, areas, sums, h_vectors, edges


def _evaluate(reference, polynomials, left, right, arithmetic):
    faces, area_polys, sum_polys, h_polys, edges = _face_polynomials(reference, polynomials, arithmetic)
    threshold = arithmetic.input_float(1e-10)
    areas, area_norms, face_witnesses = {}, {}, []
    for face_id, face in faces:
        area = arithmetic.poly_vector_bounds(area_polys[face_id], left, right)
        square = arithmetic.polynomial_bounds(arithmetic.poly_dot(area_polys[face_id], area_polys[face_id]), left, right)
        if square.lo <= 0:
            category = "degenerate_face" if square.hi == 0 else "law_domain_unknown"
            raise ShellUnresolved(category, str(face_id))
        norm = arithmetic.sqrt(square)
        maximum = max(arithmetic.polynomial_bounds(arithmetic.poly_dot(edge, edge), left, right).hi for edge in edges[face_id])
        if norm.lo <= arithmetic.qmul(threshold, maximum):
            raise ShellUnresolved("law_domain_unknown", "strict invariant face shape")
        areas[face_id], area_norms[face_id] = area, norm
        face_witnesses.append((face_id, norm, maximum))
    rho = {int(node): arithmetic.input_float(reference.half_thickness_m[row]) for row, node in enumerate(reference.node_ids)}
    sums, normals, offsets, star_witnesses, corner_witnesses = {}, {}, {}, [], []
    for node in sorted(polynomials):
        vector = arithmetic.poly_vector_bounds(sum_polys[node], left, right)
        square = arithmetic.polynomial_bounds(arithmetic.poly_dot(sum_polys[node], sum_polys[node]), left, right)
        if square.lo <= 0:
            category = "vanishing_director" if square.hi == 0 else "director_domain_unknown"
            raise ShellUnresolved(category, str(node))
        norm = arithmetic.sqrt(square)
        incident = tuple(face_id for face_id, face in faces if node in face)
        if len(incident) > arithmetic.limits.max_star_faces:
            raise ShellUnresolved("shape_budget", "star")
        sum_norms = arithmetic.sum(area_norms[face_id] for face_id in incident)
        if norm.lo <= arithmetic.qmul(threshold, sum_norms.hi):
            raise ShellUnresolved("director_domain_unknown", "strict invariant star shape")
        normal = tuple(arithmetic.div(value, norm) for value in vector)
        for face_id in incident:
            dot = arithmetic.polynomial_bounds(arithmetic.poly_dot(sum_polys[node], area_polys[face_id]), left, right)
            if dot.lo <= 0:
                raise ShellUnresolved("projected_support_unknown", "dot sign")
            ratio = arithmetic.div(dot, arithmetic.mul(norm, area_norms[face_id]))
            if ratio.lo < F(1, 4):
                raise ShellUnresolved("projected_support_unknown", "corner projection")
            corner_witnesses.append((node, face_id, ratio))
        sums[node], normals[node] = vector, normal
        offsets[node] = arithmetic.scale(arithmetic.exact(rho[node]), normal)
        star_witnesses.append((node, norm, sum_norms))
    positions = {node: arithmetic.poly_vector_bounds(poly, left, right) for node, poly in polynomials.items()}
    lifted = {(node, sign): arithmetic.vadd(positions[node], arithmetic.scale(arithmetic.exact(sign), offsets[node])) for node in sorted(polynomials) for sign in (-1, 1)}
    return dict(faces=faces, positions=positions, areas=areas, sums=sums, normals=normals, offsets=offsets, lifted=lifted, h_polys=h_polys, sum_polys=sum_polys, norms={row[0]: row[1] for row in star_witnesses}, rho=rho, domain=DomainWitness(tuple(face_witnesses), tuple(star_witnesses), tuple(corner_witnesses)))


def _canonical_address(reference, address, arithmetic):
    if type(address) is not ShellMaterialAddress:
        raise TypeError("ShellMaterialAddress required")
    address.__post_init__()
    if address.reference_id != reference.reference_id:
        raise ShellUnresolved("address_identity_mismatch", "reference")
    found = tuple(t for t in reference.tetrahedra if t.tetra_id == address.tetra_id)
    if len(found) != 1:
        raise ShellUnresolved("address_identity_mismatch", "tetrahedron")
    tet = found[0]
    weights = tuple(arithmetic.qdiv(F(count), F(2**52)) for count in address.counts)
    canonical = tuple(sorted((vertex, weight) for vertex, weight in zip(tet.vertices, weights, strict=True) if weight))
    beta, gamma = {}, {}
    for (node, sign), weight in canonical:
        beta[node] = arithmetic.qadd(beta.get(node, F(0)), weight)
        gamma[node] = arithmetic.qadd(gamma.get(node, F(0)), arithmetic.qmul(F(sign), weight))
    return canonical, beta, gamma


def _point_map(reference, address, evaluated, arithmetic):
    canonical, beta, gamma = _canonical_address(reference, address, arithmetic)
    node_ids = set(beta)
    for node, weight in gamma.items():
        if weight:
            node_ids.update(evaluated["h_polys"][node])
    node_ids = tuple(sorted(node_ids))
    point = (arithmetic.exact(0),) * 3
    for node, weight in beta.items():
        point = arithmetic.vadd(point, arithmetic.scale(arithmetic.exact(weight), evaluated["positions"][node]))
    for node, weight in gamma.items():
        if weight:
            point = arithmetic.vadd(point, arithmetic.scale(arithmetic.exact(weight), evaluated["offsets"][node]))
    blocks = {}
    for neighbor in node_ids:
        blocks[neighbor] = [[arithmetic.exact(beta.get(neighbor, F(0)) if i == j else 0) for j in range(3)] for i in range(3)]
    for node, weight in gamma.items():
        if not weight:
            continue
        normal = evaluated["normals"][node]
        projection = tuple(tuple(arithmetic.sub(arithmetic.exact(1 if i == j else 0), arithmetic.mul(normal[i], normal[j])) for j in range(3)) for i in range(3))
        scale = arithmetic.div(arithmetic.exact(arithmetic.qmul(weight, evaluated["rho"][node])), evaluated["norms"][node])
        for neighbor, vector_poly in evaluated["h_polys"][node].items():
            vector = arithmetic.poly_vector_bounds(vector_poly, F(0), F(0))
            derivative = arithmetic.matmul(projection, arithmetic.cross_matrix(vector))
            for i in range(3):
                for j in range(3):
                    blocks[neighbor][i][j] = arithmetic.add(blocks[neighbor][i][j], arithmetic.mul(scale, derivative[i][j]))
    flat = tuple(value for node in node_ids for row in blocks[node] for value in row)
    return canonical, node_ids, arithmetic.publish(point, (3,)), arithmetic.publish(flat, (len(node_ids), 3, 3))


def _geometry_receipt(reference, positions, arithmetic, polynomials=None):
    if polynomials is None:
        polynomials = _constant_polynomials(reference, positions, arithmetic)
    evaluated = _evaluate(reference, polynomials, F(0), F(0), arithmetic)
    nodes = tuple(sorted(polynomials))
    face_ids = tuple(face_id for face_id, _ in evaluated["faces"])
    lifted = tuple(evaluated["lifted"][(node, sign)] for node in nodes for sign in (-1, 1))
    directors = tuple(evaluated["normals"][node] for node in nodes)
    dpub = arithmetic.publish(tuple(v for vector in directors for v in vector), (len(nodes), 3))
    lpub = arithmetic.publish(tuple(v for vector in lifted for v in vector), (len(nodes), 2, 3))
    return ShellDirectorGeometry("conforming-shell-directors-v1", "ideal-from-stored-fp64", "director_geometry_only", reference.reference_id, _binding("directors", reference, positions, arithmetic.limits), arithmetic.limits, arithmetic.bits, nodes, face_ids, tuple(evaluated["areas"][face] for face in face_ids), tuple(evaluated["sums"][node] for node in nodes), directors, tuple(evaluated["offsets"][node] for node in nodes), lifted, evaluated["domain"], dpub, lpub, arithmetic.operations)


def shell_director_geometry(reference, positions, *, limits=ShellProofLimits()):
    positions = _positions(reference, positions, limits)
    arithmetic = Arithmetic(limits)
    polynomials = _constant_polynomials(reference, positions, arithmetic)
    while True:
        try:
            return _geometry_receipt(reference, positions, arithmetic, polynomials)
        except ShellUnresolved as error:
            if error.category not in ("publication_error", "division_interval_contains_zero") or arithmetic.bits == limits.max_sqrt_bits:
                raise
            arithmetic.set_bits(min(2 * arithmetic.bits, limits.max_sqrt_bits))

def shell_material_point(reference, positions, address, *, limits=ShellProofLimits()):
    positions = _positions(reference, positions, limits)
    arithmetic = Arithmetic(limits)
    polynomials = _constant_polynomials(reference, positions, arithmetic)
    while True:
        try:
            evaluated = _evaluate(reference, polynomials, F(0), F(0), arithmetic)
            canonical, nodes, point, jacobian = _point_map(reference, address, evaluated, arithmetic)
            return ShellMaterialPoint("conforming-shell-point-v1", "ideal-from-stored-fp64", "director_geometry_only", reference.reference_id, _binding("point", reference, positions, limits, address=address), limits, arithmetic.bits, canonical, address.generation, nodes, point, jacobian, evaluated["domain"], arithmetic.operations)
        except ShellUnresolved as error:
            if error.category not in ("publication_error", "division_interval_contains_zero") or arithmetic.bits == limits.max_sqrt_bits:
                raise
            arithmetic.set_bits(min(2 * arithmetic.bits, limits.max_sqrt_bits))