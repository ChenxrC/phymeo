"""Independent original-input checking; no producer topology/map/proof calls."""

import json
from dataclasses import fields, is_dataclass
from fractions import Fraction as F
from functools import wraps
from hashlib import sha256
from math import isfinite

import numpy as np

from ..constraints.conforming_shell import (
    ConformingShellReference,
    ShellBoundaryFacet,
    ShellDirectorLaw,
    ShellMaterialAddress,
    ShellProofLimits,
    ShellTetrahedron,
    ShellUnresolved,
)
from ..geometry._shell_intervals import Arithmetic, Interval, Publication
from ..geometry.shell_directors import DomainWitness, ShellDirectorGeometry, ShellMaterialPoint
from ..geometry.shell_embedding import (
    CellWitness,
    ComponentWitness,
    RankOneCharge,
    ReferenceChart,
    ReferenceGradient,
    ReferenceWitness,
    SeparatorWitness,
    ShellEmbeddingCertificate,
    TetrahedronWitness,
)
from ..geometry.shell_sweep import ShellSweepCertificate, TimeCell


def _checked_verification(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ShellUnresolved:
            raise
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise ShellUnresolved("certificate_rejected", str(error)) from error

    return checked


def _reject(detail):
    raise ShellUnresolved("certificate_rejected", detail)


def _cyclic(values):
    return min(tuple(values[j:]) + tuple(values[:j]) for j in range(len(values)))


def _orientation(values):
    count = 0
    for i, value in enumerate(values):
        count += sum(other < value for other in values[i + 1 :])
    return -1 if count % 2 else 1


def _hash(data):
    return sha256(json.dumps(data, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _immutable_array(array, shape, dtype):
    if (
        type(array) is not np.ndarray
        or array.shape != shape
        or array.dtype != np.dtype(dtype)
        or not np.isfinite(array).all()
        or array.flags.writeable
    ):
        _reject("owned reference/proposal array shape, dtype or finiteness")
    owner = array
    for _ in range(8):
        if type(owner) is bytes:
            return
        if type(owner) is not np.ndarray:
            break
        owner = owner.base
    _reject("array does not have immutable byte backing")


def _original(reference, limits, arithmetic):
    if type(reference) is not ConformingShellReference or type(limits) is not ShellProofLimits:
        _reject("reference/policy type")
    limits.__post_init__()
    if type(reference.law) is not ShellDirectorLaw:
        _reject("director law type")
    reference.law.__post_init__()
    if (
        type(reference.node_ids) is not np.ndarray
        or reference.node_ids.ndim != 1
        or type(reference.face_ids) is not np.ndarray
        or reference.face_ids.ndim != 1
    ):
        _reject("ID array rank")
    n, f = len(reference.node_ids), len(reference.face_ids)
    if (
        not 3 <= n <= limits.max_nodes
        or not 1 <= f <= limits.max_faces
        or 3 * f > limits.max_tetrahedra
        or (3 * f) * (3 * f - 1) // 2 > limits.max_pairs
    ):
        _reject("reference hard shape limits")
    for array, shape, dtype in (
        (reference.node_ids, (n,), "int64"),
        (reference.face_ids, (f,), "int64"),
        (reference.oriented_nodes, (f, 3), "int64"),
        (reference.reference_positions, (n, 3), "float64"),
        (reference.half_thickness_m, (n,), "float64"),
    ):
        _immutable_array(array, shape, dtype)
    nodes = tuple(int(v) for v in reference.node_ids)
    face_ids = tuple(int(v) for v in reference.face_ids)
    if (
        len(set(nodes)) != n
        or len(set(face_ids)) != f
        or min(nodes) < 0
        or min(face_ids) < 0
        or np.any(reference.half_thickness_m <= 0)
    ):
        _reject("IDs/thickness")
    if (
        type(reference.domain) is not str
        or not reference.domain
        or len(reference.domain.encode("utf-8")) > 256
    ):
        _reject("domain")
    faces = tuple(
        sorted(
            (face_id, tuple(int(v) for v in row))
            for face_id, row in zip(face_ids, reference.oriented_nodes, strict=True)
        )
    )
    adjacency = {node: set() for node in nodes}
    stars = {node: [] for node in nodes}
    oriented_edges = {}
    unoriented = set()
    for face_id, face in faces:
        if len(set(face)) != 3 or not set(face) <= set(nodes) or tuple(sorted(face)) in unoriented:
            _reject("original face identities")
        unoriented.add(tuple(sorted(face)))
        for node in face:
            stars[node].append(face_id)
            adjacency[node].update(set(face) - {node})
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            oriented_edges.setdefault(tuple(sorted((a, b))), []).append((a, b))
    boundary_edges = []
    for values in oriented_edges.values():
        if len(values) == 1:
            boundary_edges.extend(values)
        elif len(values) != 2 or values[0][0] != values[1][1] or values[0][1] != values[1][0]:
            _reject("manifold edge/winding")
    for node in nodes:
        if not 1 <= len(stars[node]) <= limits.max_star_faces:
            _reject("unused node/star limit")
        link_edges = [tuple(v for v in face if v != node) for _, face in faces if node in face]
        link_nodes = set(v for edge in link_edges for v in edge)
        connected = {min(link_nodes)}
        while True:
            expanded = (
                connected
                | {b for a, b in link_edges if a in connected}
                | {a for a, b in link_edges if b in connected}
            )
            if expanded == connected:
                break
            connected = expanded
        degrees = {v: sum(v in edge for edge in link_edges) for v in link_nodes}
        on_boundary = any(node in edge for edge in boundary_edges)
        required_ends = 2 if on_boundary else 0
        if (
            connected != link_nodes
            or sum(value == 1 for value in degrees.values()) != required_ends
            or any(value not in (1, 2) for value in degrees.values())
        ):
            _reject("vertex link is not one interval/cycle")
    components, remaining = [], set(nodes)
    while remaining:
        group = {min(remaining)}
        while True:
            expanded = group | set().union(*(adjacency[v] for v in group))
            if expanded == group:
                break
            group = expanded
        components.append(tuple(sorted(group)))
        remaining -= group
    cycles = []
    edge_set = set(boundary_edges)
    if len(edge_set) != len(boundary_edges):
        _reject("duplicate boundary edge")
    while edge_set:
        start = min(v for edge in edge_set for v in edge)
        chain, node = [], start
        while not chain or node != start:
            if node in chain:
                _reject("boundary self-link")
            outgoing = [edge for edge in edge_set if edge[0] == node]
            if len(outgoing) != 1:
                _reject("boundary successor")
            chain.append(node)
            edge_set.remove(outgoing[0])
            node = outgoing[0][1]
        cycles.append(_cyclic(tuple(chain)))
    tetrahedra = []
    boundary_coefficients = {}
    facet_owners = {}
    for face_id, face in faces:
        a, b, c = sorted(face)
        candidates = (
            ((a, -1), (b, -1), (c, -1), (c, 1)),
            ((a, -1), (b, -1), (b, 1), (c, 1)),
            ((a, -1), (a, 1), (b, 1), (c, 1)),
        )
        for number, original in enumerate(candidates):
            vertices = original
            if _orientation(face) != (1, -1, 1)[number]:
                vertices = (original[0], original[1], original[3], original[2])
            tet = ShellTetrahedron((face_id, number), vertices)
            tetrahedra.append(tet)
            for omitted in range(4):
                facet = vertices[:omitted] + vertices[omitted + 1 :]
                if omitted % 2:
                    facet = (facet[1], facet[0], facet[2])
                key = tuple(sorted(facet))
                boundary_coefficients[key] = boundary_coefficients.get(key, 0) + _orientation(facet)
                facet_owners.setdefault(key, []).append((_cyclic(facet), tet.tetra_id))
    boundary = []
    for key in sorted(facet_owners):
        owners = facet_owners[key]
        if len(owners) == 1:
            boundary.append(ShellBoundaryFacet(*owners[0]))
        elif len(owners) != 2 or boundary_coefficients[key] != 0:
            _reject("product facets do not cancel")
    topo = (tuple(tetrahedra), tuple(boundary), tuple(components), tuple(sorted(cycles)))
    if topo != (
        reference.tetrahedra,
        reference.boundary_facets,
        reference.components,
        reference.boundary_cycles,
    ):
        _reject("stored topology/facets differ from independent oriented chain")
    canonical_nodes = sorted(
        (
            node,
            reference.reference_positions[row].tobytes().hex(),
            reference.half_thickness_m[row : row + 1].tobytes().hex(),
        )
        for row, node in enumerate(nodes)
    )
    canonical_faces = sorted((i, _cyclic(face)) for i, face in faces)
    identity = _hash(
        (
            "conforming-shell-reference-v1",
            reference.domain,
            canonical_nodes,
            canonical_faces,
            reference.law.kind,
            (1, 4),
            reference.law.shape_policy,
        )
    )
    order = _hash(
        (
            identity,
            reference.node_ids.tobytes().hex(),
            reference.face_ids.tobytes().hex(),
            reference.oriented_nodes.tobytes().hex(),
        )
    )
    if identity != reference.reference_id or order != reference.input_fingerprint:
        _reject("reference canonical/order binding")
    coords = {
        node: tuple(arithmetic.input_float(v) for v in reference.reference_positions[row])
        for row, node in enumerate(nodes)
    }
    rho = {
        node: arithmetic.input_float(reference.half_thickness_m[row])
        for row, node in enumerate(nodes)
    }
    return dict(
        nodes=nodes,
        faces=faces,
        stars=stars,
        components=tuple(components),
        cycles=tuple(sorted(cycles)),
        tetrahedra=tuple(tetrahedra),
        boundary=tuple(boundary),
        coordinates=coords,
        rho=rho,
    )


def _input_array(array, n, path=False):
    shape = (2, n, 3) if path else (n, 3)
    if (
        type(array) is not np.ndarray
        or array.dtype != np.dtype("float64")
        or array.shape != shape
        or not np.isfinite(array).all()
    ):
        _reject("original state/path type, shape or finite values")
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(shape)


def _binding(kind, ref, array, limits, h=None, address=None):
    policy = tuple((field.name, getattr(limits, field.name)) for field in fields(limits))
    return _hash(
        (
            kind,
            ref.reference_id,
            ref.input_fingerprint,
            array.dtype.str,
            array.shape,
            array.tobytes(order="C").hex(),
            policy,
            None if h is None else (h.numerator, h.denominator),
            None
            if address is None
            else (address.reference_id, address.tetra_id, address.counts, address.generation),
            "ideal-from-stored-fp64",
        )
    )


def _contained(claim, independent, arithmetic):
    if type(independent) is Interval:
        arithmetic.interval(claim)
        if claim.lo > independent.lo or claim.hi < independent.hi:
            _reject("claimed ideal interval does not contain independent enclosure")
    elif type(independent) is F:
        arithmetic.check(claim)
        if claim != independent:
            _reject("exact scalar witness differs")
    elif type(independent) is tuple:
        if type(claim) is not tuple or len(claim) != len(independent):
            _reject("nested witness shape/coverage")
        for a, b in zip(claim, independent, strict=True):
            _contained(a, b, arithmetic)
    elif is_dataclass(independent):
        if type(claim) is not type(independent):
            _reject("witness schema type")
        for field in fields(independent):
            _contained(getattr(claim, field.name), getattr(independent, field.name), arithmetic)
    elif type(claim) is not type(independent) or claim != independent:
        _reject("exact identity/coverage field differs")


def _publication(publication, intervals, shape, arithmetic):
    if (
        type(publication) is not Publication
        or publication.shape != shape
        or len(publication.errors) != len(intervals)
    ):
        _reject("publication schema/shape")
    _immutable_array(publication.values, shape, "float64")
    _contained(publication.intervals, tuple(intervals), arithmetic)
    for bound, stored, error in zip(
        publication.intervals, publication.values.flat, publication.errors, strict=True
    ):
        arithmetic.check(error)
        midpoint = arithmetic.qdiv(arithmetic.qadd(bound.lo, bound.hi), F(2))
        if not isfinite(float(midpoint)) or float(stored) != float(midpoint):
            _reject("FP64 proposal is not its declared rational midpoint conversion")
        value = arithmetic.input_float(stored)
        required = max(
            arithmetic.qabs(arithmetic.qsub(value, bound.lo)),
            arithmetic.qabs(arithmetic.qsub(value, bound.hi)),
        )
        if error < required or error >= F(1, 2**40):
            _reject("publication storage error")


def _header(receipt, expected_type, schema, status, reference, binding, limits):
    if (
        type(receipt) is not expected_type
        or receipt.schema != schema
        or receipt.status != status
        or receipt.precision != "ideal-from-stored-fp64"
    ):
        _reject("receipt schema/status/precision")
    if (
        receipt.reference_id != reference.reference_id
        or receipt.binding != binding
        or type(receipt.policy) is not ShellProofLimits
        or receipt.policy != limits
    ):
        _reject("original input/identity/policy binding")
    if (
        type(receipt.operations) is not int
        or not 0 <= receipt.operations <= limits.max_scalar_operations
    ):
        _reject("reported work range")


def _polynomials(original, array, arithmetic, path=False):
    result = {}
    for row, node in enumerate(original["nodes"]):
        values = []
        for coordinate in range(3):
            if path:
                first = arithmetic.input_float(array[0, row, coordinate])
                difference = arithmetic.qsub(
                    arithmetic.input_float(array[1, row, coordinate]), first
                )
                values.append(arithmetic.poly((first, difference)))
            else:
                values.append((arithmetic.input_float(array[row, coordinate]),))
        result[node] = tuple(values)
    return result


def _derive(original, polynomials, left, right, arithmetic):
    zero = ((F(0),),) * 3
    raw, edge_sets = {}, {}
    sums = {node: zero for node in original["nodes"]}
    differentials = {node: {} for node in original["nodes"]}
    for face_id, face in original["faces"]:
        a, b, c = face
        e1 = arithmetic.poly_vsub(polynomials[b], polynomials[a])
        e2 = arithmetic.poly_vsub(polynomials[c], polynomials[a])
        e3 = arithmetic.poly_vsub(polynomials[c], polynomials[b])
        raw[face_id] = arithmetic.poly_cross(e1, e2)
        edge_sets[face_id] = (e1, e2, e3)
        # Coefficients of the cross-product matrix are summed exactly before
        # bounding; every face contributes to each of its three incident stars.
        coefficient = (e3, tuple(tuple(arithmetic.qneg(v) for v in p) for p in e2), e1)
        for star in face:
            sums[star] = arithmetic.poly_vadd(sums[star], raw[face_id])
            for neighbor, vector in zip(face, coefficient, strict=True):
                differentials[star][neighbor] = arithmetic.poly_vadd(
                    differentials[star].get(neighbor, zero), vector
                )
    face_values, face_lengths, face_checks = {}, {}, []
    cutoff = arithmetic.input_float(1e-10)
    for face_id, _ in original["faces"]:
        norm2 = arithmetic.polynomial_bounds(
            arithmetic.poly_dot(raw[face_id], raw[face_id]), left, right
        )
        if norm2.lo <= 0:
            _reject("independent face nondegeneracy")
        length = arithmetic.sqrt(norm2)
        longest2 = max(
            arithmetic.polynomial_bounds(arithmetic.poly_dot(edge, edge), left, right).hi
            for edge in edge_sets[face_id]
        )
        if length.lo <= arithmetic.qmul(cutoff, longest2):
            _reject("independent strict face conditioning")
        face_values[face_id] = arithmetic.poly_vector_bounds(raw[face_id], left, right)
        face_lengths[face_id] = length
        face_checks.append((face_id, length, longest2))
    sum_values, normals, offsets, lengths = {}, {}, {}, {}
    star_checks, corner_checks = [], []
    for node in sorted(original["nodes"]):
        squared = arithmetic.polynomial_bounds(
            arithmetic.poly_dot(sums[node], sums[node]), left, right
        )
        if squared.lo <= 0:
            _reject("independent director nondegeneracy")
        length = arithmetic.sqrt(squared)
        sum_face_lengths = arithmetic.sum(
            face_lengths[face_id] for face_id in original["stars"][node]
        )
        if length.lo <= arithmetic.qmul(cutoff, sum_face_lengths.hi):
            _reject("independent strict star conditioning")
        sum_values[node] = arithmetic.poly_vector_bounds(sums[node], left, right)
        normals[node] = tuple(arithmetic.div(value, length) for value in sum_values[node])
        offsets[node] = arithmetic.scale(arithmetic.exact(original["rho"][node]), normals[node])
        lengths[node] = length
        star_checks.append((node, length, sum_face_lengths))
        for face_id in original["stars"][node]:
            signed = arithmetic.polynomial_bounds(
                arithmetic.poly_dot(sums[node], raw[face_id]), left, right
            )
            if signed.lo <= 0:
                _reject("independent projected dot sign")
            ratio = arithmetic.div(signed, arithmetic.mul(length, face_lengths[face_id]))
            if ratio.lo < F(1, 4):
                _reject("independent corner projection")
            corner_checks.append((node, face_id, ratio))
    positions = {
        node: arithmetic.poly_vector_bounds(polynomials[node], left, right)
        for node in original["nodes"]
    }
    return dict(
        areas=face_values,
        sums=sum_values,
        normals=normals,
        offsets=offsets,
        lengths=lengths,
        positions=positions,
        sum_polys=sums,
        differentials=differentials,
        domain=DomainWitness(tuple(face_checks), tuple(star_checks), tuple(corner_checks)),
    )


def _point_formula(original, reference, address, evaluated, arithmetic):
    if type(address) is not ShellMaterialAddress:
        _reject("address type")
    address.__post_init__()
    if address.reference_id != reference.reference_id:
        _reject("address reference")
    matching = [tet for tet in original["tetrahedra"] if tet.tetra_id == address.tetra_id]
    if len(matching) != 1:
        _reject("address tetrahedron")
    entries = []
    for vertex, count in zip(matching[0].vertices, address.counts, strict=True):
        if count:
            entries.append((vertex, arithmetic.qdiv(F(count), F(2**52))))
    canonical = tuple(sorted(entries))
    beta, gamma = {}, {}
    for (node, sign), weight in canonical:
        beta[node] = arithmetic.qadd(beta.get(node, F(0)), weight)
        gamma[node] = arithmetic.qadd(gamma.get(node, F(0)), arithmetic.qmul(F(sign), weight))
    neighbors = set(beta)
    for node, gamma_i in gamma.items():
        if gamma_i:
            neighbors.update(evaluated["differentials"][node])
    neighbors = tuple(sorted(neighbors))
    point = (arithmetic.exact(0),) * 3
    for node, weight in beta.items():
        point = arithmetic.vadd(
            point, arithmetic.scale(arithmetic.exact(weight), evaluated["positions"][node])
        )
    for node, weight in gamma.items():
        if weight:
            point = arithmetic.vadd(
                point, arithmetic.scale(arithmetic.exact(weight), evaluated["offsets"][node])
            )
    blocks = {
        node: [
            [arithmetic.exact(beta.get(node, F(0)) if i == j else 0) for j in range(3)]
            for i in range(3)
        ]
        for node in neighbors
    }
    for star, weight in gamma.items():
        if not weight:
            continue
        normal = evaluated["normals"][star]
        projector = tuple(
            tuple(
                arithmetic.sub(arithmetic.exact(int(i == j)), arithmetic.mul(normal[i], normal[j]))
                for j in range(3)
            )
            for i in range(3)
        )
        factor = arithmetic.div(
            arithmetic.exact(arithmetic.qmul(weight, original["rho"][star])),
            evaluated["lengths"][star],
        )
        for node in sorted(evaluated["differentials"][star]):
            h = arithmetic.poly_vector_bounds(evaluated["differentials"][star][node], F(0), F(0))
            matrix = arithmetic.cross_matrix(h)
            # Reconstruct each contraction directly; no reported J participates.
            for row in range(3):
                for column in range(3):
                    contraction = arithmetic.sum(
                        arithmetic.mul(projector[row][k], matrix[k][column]) for k in range(3)
                    )
                    blocks[node][row][column] = arithmetic.add(
                        blocks[node][row][column], arithmetic.mul(factor, contraction)
                    )
    jacobian = tuple(value for node in neighbors for row in blocks[node] for value in row)
    return canonical, neighbors, point, jacobian


def _geometry_fields(receipt, reference, positions, original, arithmetic):
    limits = arithmetic.limits
    _header(
        receipt,
        ShellDirectorGeometry,
        "conforming-shell-directors-v1",
        "director_geometry_only",
        reference,
        _binding("directors", reference, positions, limits),
        limits,
    )
    arithmetic.set_bits(receipt.bits)
    evaluated = _derive(
        original, _polynomials(original, positions, arithmetic), F(0), F(0), arithmetic
    )
    nodes = tuple(sorted(original["nodes"]))
    face_ids = tuple(i for i, _ in original["faces"])
    _contained(receipt.node_ids, nodes, arithmetic)
    _contained(receipt.face_ids, face_ids, arithmetic)
    for name, expected in (
        ("areas", tuple(evaluated["areas"][i] for i in face_ids)),
        ("sums", tuple(evaluated["sums"][i] for i in nodes)),
        ("directors", tuple(evaluated["normals"][i] for i in nodes)),
        ("offsets", tuple(evaluated["offsets"][i] for i in nodes)),
    ):
        _contained(getattr(receipt, name), expected, arithmetic)
    lifted = tuple(
        arithmetic.vadd(
            evaluated["positions"][node],
            arithmetic.scale(arithmetic.exact(sign), evaluated["offsets"][node]),
        )
        for node in nodes
        for sign in (-1, 1)
    )
    _contained(receipt.lifted, lifted, arithmetic)
    _contained(receipt.domain, evaluated["domain"], arithmetic)
    _publication(
        receipt.director_publication,
        tuple(v for node in nodes for v in evaluated["normals"][node]),
        (len(nodes), 3),
        arithmetic,
    )
    _publication(
        receipt.lifted_publication,
        tuple(v for vector in lifted for v in vector),
        (len(nodes), 2, 3),
        arithmetic,
    )
    return True


@_checked_verification
def verify_shell_director_geometry(receipt, reference, positions, *, limits=ShellProofLimits()):
    arithmetic = Arithmetic(limits)
    original = _original(reference, limits, arithmetic)
    positions = _input_array(positions, len(original["nodes"]))
    return _geometry_fields(receipt, reference, positions, original, arithmetic)


@_checked_verification
def verify_shell_material_point(
    receipt, reference, positions, address, *, limits=ShellProofLimits()
):
    arithmetic = Arithmetic(limits)
    original = _original(reference, limits, arithmetic)
    positions = _input_array(positions, len(original["nodes"]))
    if type(address) is not ShellMaterialAddress:
        _reject("address type")
    address.__post_init__()
    _header(
        receipt,
        ShellMaterialPoint,
        "conforming-shell-point-v1",
        "director_geometry_only",
        reference,
        _binding("point", reference, positions, limits, address=address),
        limits,
    )
    arithmetic.set_bits(receipt.bits)
    evaluated = _derive(
        original, _polynomials(original, positions, arithmetic), F(0), F(0), arithmetic
    )
    canonical, nodes, point, jacobian = _point_formula(
        original, reference, address, evaluated, arithmetic
    )
    _contained(receipt.canonical_address, canonical, arithmetic)
    _contained(receipt.generation, address.generation, arithmetic)
    _contained(receipt.node_ids, nodes, arithmetic)
    _contained(receipt.domain, evaluated["domain"], arithmetic)
    _publication(receipt.point, point, (3,), arithmetic)
    _publication(receipt.jacobian, jacobian, (len(nodes), 3, 3), arithmetic)
    # Exact covariance/work identities are justified by the independently rebuilt
    # analytic law and complete star incidence, never by a residual containing 0.
    return True


def _side(a, b, p, arithmetic):
    return arithmetic.qcross2(arithmetic.qvsub(b, a), arithmetic.qvsub(p, a))


def _between(point, first, second, arithmetic):
    return _side(first, second, point, arithmetic) == 0 and all(
        min(a, b) <= p <= max(a, b) for p, a, b in zip(point, first, second, strict=True)
    )


def _segment_points(a, b, c, d, arithmetic):
    ab, cd = arithmetic.qvsub(b, a), arithmetic.qvsub(d, c)
    denominator = arithmetic.qcross2(ab, cd)
    if denominator == 0:
        if _side(a, b, c, arithmetic) != 0:
            return ()
        return tuple(
            sorted(
                {
                    p
                    for p in (a, b, c, d)
                    if _between(p, a, b, arithmetic) and _between(p, c, d, arithmetic)
                }
            )
        )
    ca = arithmetic.qvsub(c, a)
    t = arithmetic.qdiv(arithmetic.qcross2(ca, cd), denominator)
    s = arithmetic.qdiv(arithmetic.qcross2(ca, ab), denominator)
    if t < 0 or t > 1 or s < 0 or s > 1:
        return ()
    return (tuple(arithmetic.qadd(x, arithmetic.qmul(t, y)) for x, y in zip(a, ab, strict=True)),)


def _closed_triangle_points(first, second, arithmetic):
    # Independent finite intersection enumeration, NOT the producer's polygon clip.
    # Every intersection extreme is an included triangle vertex or edge crossing;
    # collinear endpoints retain zero-area point/segment intersections.
    def inside(point, triangle):
        return all(
            _side(triangle[i], triangle[(i + 1) % 3], point, arithmetic) >= 0 for i in range(3)
        )

    points = {p for p in first if inside(p, second)} | {p for p in second if inside(p, first)}
    for i in range(3):
        for j in range(3):
            points.update(
                _segment_points(
                    first[i], first[(i + 1) % 3], second[j], second[(j + 1) % 3], arithmetic
                )
            )
    return tuple(sorted(points))


def _reference_charts(original, arithmetic):
    charts = []
    coordinates = original["coordinates"]
    for component in original["components"]:
        faces = tuple(
            (face_id, face) for face_id, face in original["faces"] if face[0] in component
        )
        a, b, c = faces[0][1]
        normal = arithmetic.qcross(
            arithmetic.qvsub(coordinates[b], coordinates[a]),
            arithmetic.qvsub(coordinates[c], coordinates[a]),
        )
        if normal == (F(0),) * 3 or len({original["rho"][node] for node in component}) != 1:
            _reject("independent reference chart normal/constant thickness")
        for node in component:
            if (
                arithmetic.qdot(
                    normal, arithmetic.qvsub(coordinates[node], coordinates[min(component)])
                )
                != 0
            ):
                _reject("independent reference planarity")
        for _, face in faces:
            area = arithmetic.qcross(
                arithmetic.qvsub(coordinates[face[1]], coordinates[face[0]]),
                arithmetic.qvsub(coordinates[face[2]], coordinates[face[0]]),
            )
            if arithmetic.qcross(normal, area) != (F(0),) * 3 or arithmetic.qdot(normal, area) <= 0:
                _reject("independent reference parallel-positive normals")
        axis = next(i for i, value in enumerate(normal) if value)
        u, v = (axis + 1) % 3, (axis + 2) % 3
        if normal[axis] < 0:
            u, v = v, u
        xy = {node: (coordinates[node][u], coordinates[node][v]) for node in component}
        cycles = [cycle for cycle in original["cycles"] if cycle[0] in component]
        edges = {tuple(sorted((face[i], face[(i + 1) % 3]))) for _, face in faces for i in range(3)}
        euler = arithmetic.qsub(arithmetic.qadd(F(len(component)), F(len(faces))), F(len(edges)))
        if len(cycles) != 1 or euler != 1:
            _reject("independent reference disk coverage")
        boundary = cycles[0]
        polygon = tuple(xy[node] for node in boundary)
        if len(set(polygon)) != len(polygon):
            _reject("geometric boundary vertices repeat")
        for i in range(len(boundary)):
            a, b, c = polygon[i], polygon[(i + 1) % len(polygon)], polygon[(i + 2) % len(polygon)]
            if a == b:
                _reject("zero boundary edge")
            turn = _side(a, b, c, arithmetic)
            if turn < 0 or (
                turn == 0 and arithmetic.qdot(arithmetic.qvsub(b, a), arithmetic.qvsub(c, b)) <= 0
            ):
                _reject("convex boundary/backtracking")
            if any(_side(a, b, xy[node], arithmetic) < 0 for node in component):
                _reject("reference triangle outside boundary")
            for j in range(i + 1, len(boundary)):
                shared = {boundary[i], boundary[(i + 1) % len(boundary)]} & {
                    boundary[j],
                    boundary[(j + 1) % len(boundary)],
                }
                crossing = _segment_points(
                    a, b, polygon[j], polygon[(j + 1) % len(polygon)], arithmetic
                )
                if set(crossing) != {xy[node] for node in shared}:
                    _reject("reference boundary not simple")
        area = F(0)
        for i in range(len(polygon)):
            area = arithmetic.qadd(
                area, arithmetic.qcross2(polygon[i], polygon[(i + 1) % len(polygon)])
            )
        total = F(0)
        for index, (_, face) in enumerate(faces):
            triangle = tuple(xy[node] for node in face)
            positive = _side(*triangle, arithmetic)
            if positive <= 0:
                _reject("reference triangle not positively oriented")
            total = arithmetic.qadd(total, positive)
            for _, other in faces[index + 1 :]:
                intersection = _closed_triangle_points(
                    triangle, tuple(xy[node] for node in other), arithmetic
                )
                shared = tuple(sorted(set(face) & set(other)))
                if len(shared) == 0:
                    allowed = not intersection
                elif len(shared) == 1:
                    allowed = bool(intersection) and all(p == xy[shared[0]] for p in intersection)
                elif len(shared) == 2:
                    allowed = bool(intersection) and all(
                        _between(p, xy[shared[0]], xy[shared[1]], arithmetic) for p in intersection
                    )
                else:
                    allowed = False
                if not allowed:
                    _reject("closed reference intersection is not shared material simplex")
        if area <= 0 or total != area:
            _reject("complete convex reference area coverage")
        charts.append(
            ReferenceChart(
                component, tuple(i for i, _ in faces), boundary, (axis, u, v), normal, polygon, area
            )
        )
    return tuple(charts)


def _pair_incidence(original):
    pairs = []
    for i, left in enumerate(original["tetrahedra"]):
        for right in original["tetrahedra"][i + 1 :]:
            shared = tuple(v for v in sorted(left.vertices) if v in right.vertices)
            pairs.append((left.tetra_id, right.tetra_id, shared))
    return tuple(pairs)


def _support_bound(nodes, axis, origin, polynomials, offsets, left, right, arithmetic):
    ranges = []
    for node in nodes:
        projected_poly = (F(0),)
        relative = arithmetic.poly_vsub(polynomials[node], origin)
        for coordinate, coefficient in enumerate(axis):
            projected_poly = arithmetic.poly_add(
                projected_poly, arithmetic.poly_mul(relative[coordinate], (coefficient,))
            )
        middle = arithmetic.polynomial_bounds(projected_poly, left, right)
        arm = arithmetic.dot(tuple(arithmetic.exact(v) for v in axis), offsets[node])
        ranges.extend((arithmetic.sub(middle, arm), arithmetic.add(middle, arm)))
    return Interval(min(value.lo for value in ranges), max(value.hi for value in ranges))


def _check_separators(claims, charts, polynomials, offsets, left, right, arithmetic):
    components = tuple(chart.nodes for chart in charts)
    expected_pairs = {(a, b) for index, a in enumerate(components) for b in components[index + 1 :]}
    if type(claims) is not tuple or len(claims) != len(expected_pairs):
        _reject("component separator coverage")
    seen = set()
    for claim in claims:
        if (
            type(claim) is not SeparatorWitness
            or type(claim.left) is not tuple
            or type(claim.right) is not tuple
            or claim.left not in components
            or claim.right not in components
            or claim.left == claim.right
        ):
            _reject("component separator identity")
        key = tuple(sorted((claim.left, claim.right)))
        if key not in expected_pairs or key in seen:
            _reject("duplicate/unknown component pair")
        seen.add(key)
        if type(claim.axis) is not tuple or len(claim.axis) != 3:
            _reject("separator axis shape")
        for entry in claim.axis:
            arithmetic.check(entry)
        if claim.axis == (F(0),) * 3:
            _reject("zero separator axis")
        anchor = polynomials[min(claim.left)]
        a = _support_bound(
            claim.left, claim.axis, anchor, polynomials, offsets, left, right, arithmetic
        )
        b = _support_bound(
            claim.right, claim.axis, anchor, polynomials, offsets, left, right, arithmetic
        )
        _contained(claim.left_support, a, arithmetic)
        _contained(claim.right_support, b, arithmetic)
        if not (
            claim.left_support.hi < claim.right_support.lo
            or claim.right_support.hi < claim.left_support.lo
        ):
            _reject("strict full-component separation not proved")
    if seen != expected_pairs:
        _reject("missing component pair")


def _reference_check(claim, original, arithmetic):
    if type(claim) is not ReferenceWitness:
        _reject("reference witness type")
    charts = _reference_charts(original, arithmetic)
    _contained(claim.charts, charts, arithmetic)
    _contained(claim.boundary_facets, original["boundary"], arithmetic)
    _contained(claim.pair_incidence, _pair_incidence(original), arithmetic)
    polynomials = {
        node: tuple((value,) for value in coordinate)
        for node, coordinate in original["coordinates"].items()
    }
    reference_law = _derive(original, polynomials, F(0), F(0), arithmetic)
    unit = {}
    for chart in charts:
        vector = tuple(arithmetic.exact(v) for v in chart.normal)
        norm = arithmetic.norm(vector)
        normalized = tuple(arithmetic.div(v, norm) for v in vector)
        for node in chart.nodes:
            unit[node] = normalized
    gradients = []
    for tet in original["tetrahedra"]:
        first_node, first_sign = tet.vertices[0]
        edge_columns = []
        for node, sign in tet.vertices[1:]:
            middle = tuple(
                arithmetic.exact(value)
                for value in arithmetic.qvsub(
                    original["coordinates"][node], original["coordinates"][first_node]
                )
            )
            signed_thickness = arithmetic.qsub(
                arithmetic.qmul(F(sign), original["rho"][node]),
                arithmetic.qmul(F(first_sign), original["rho"][first_node]),
            )
            edge_columns.append(
                arithmetic.vadd(
                    middle, arithmetic.scale(arithmetic.exact(signed_thickness), unit[first_node])
                )
            )
        inverse, determinant = arithmetic.inverse3(tuple(zip(*edge_columns, strict=True)))
        if determinant.lo <= 0:
            _reject("reference tetrahedron invertibility/orientation")
        grad0 = tuple(
            arithmetic.neg(arithmetic.sum(row[coordinate] for row in inverse))
            for coordinate in range(3)
        )
        gradients.append(
            ReferenceGradient(tet.tetra_id, tet.vertices, (grad0,) + inverse, determinant)
        )
    _contained(claim.gradients, tuple(gradients), arithmetic)
    _check_separators(
        claim.separators, charts, polynomials, reference_law["offsets"], F(0), F(0), arithmetic
    )
    return charts, tuple(gradients), polynomials, reference_law


def _checked_tetra(
    original,
    tet,
    gradient,
    reference_polys,
    reference_law,
    polynomials,
    evaluated,
    component,
    left,
    right,
    arithmetic,
):
    first_node, first_sign = tet.vertices[0]
    columns = []
    for node, sign in tet.vertices[1:]:
        exact_middle_edge = arithmetic.poly_vsub(polynomials[node], polynomials[first_node])
        middle = arithmetic.poly_vector_bounds(exact_middle_edge, left, right)
        if evaluated["sum_polys"][node] == evaluated["sum_polys"][first_node]:
            coefficient = arithmetic.qsub(
                arithmetic.qmul(F(sign), original["rho"][node]),
                arithmetic.qmul(F(first_sign), original["rho"][first_node]),
            )
            offset_edge = arithmetic.scale(
                arithmetic.exact(coefficient), evaluated["normals"][first_node]
            )
        else:
            offset_edge = arithmetic.vsub(
                arithmetic.scale(arithmetic.exact(sign), evaluated["offsets"][node]),
                arithmetic.scale(arithmetic.exact(first_sign), evaluated["offsets"][first_node]),
            )
        columns.append(arithmetic.vadd(middle, offset_edge))
    determinant = arithmetic.determinant(tuple(zip(*columns, strict=True)))
    if determinant.lo <= 0:
        _reject("independent current tetra positive orientation")
    zero = (arithmetic.exact(0),) * 3
    beta, gamma = {}, {}
    for (node, sign), row in zip(tet.vertices, gradient.lambda_gradients, strict=True):
        beta[node] = arithmetic.vadd(beta.get(node, zero), row)
        gamma[node] = arithmetic.vadd(
            gamma.get(node, zero), arithmetic.scale(arithmetic.exact(sign), row)
        )
    anchor = min(component)
    common_translation = arithmetic.poly_vsub(polynomials[anchor], reference_polys[anchor])
    charges, total = [], F(0)
    for node in sorted(beta):
        displacement = arithmetic.poly_vsub(
            arithmetic.poly_vsub(polynomials[node], reference_polys[node]), common_translation
        )
        delta_x = arithmetic.poly_vector_bounds(displacement, left, right)
        if evaluated["sum_polys"][node] == reference_law["sum_polys"][node]:
            delta_a = zero
        else:
            delta_a = arithmetic.vsub(evaluated["offsets"][node], reference_law["offsets"][node])
        nx, na = arithmetic.norm(delta_x), arithmetic.norm(delta_a)
        nb, ng = arithmetic.norm(beta[node]), arithmetic.norm(gamma[node])
        tx, ta = arithmetic.qmul(nx.hi, nb.hi), arithmetic.qmul(na.hi, ng.hi)
        total = arithmetic.qadd(total, arithmetic.qadd(tx, ta))
        charges.append(RankOneCharge(node, nx, na, nb, ng, tx, ta))
    return TetrahedronWitness(
        tet.tetra_id,
        determinant,
        tuple(sorted(beta.items())),
        tuple(sorted(gamma.items())),
        tuple(charges),
        total,
    )


def _tetra_enclosure(claim, independent, arithmetic):
    if type(claim) is not TetrahedronWitness or claim.tetra_id != independent.tetra_id:
        _reject("tetra identity")
    _contained(claim.determinant, independent.determinant, arithmetic)
    if claim.determinant.lo <= 0:
        _reject("claimed positive tetra determinant")
    _contained(claim.beta_gradients, independent.beta_gradients, arithmetic)
    _contained(claim.gamma_gradients, independent.gamma_gradients, arithmetic)
    if type(claim.charges) is not tuple or len(claim.charges) != len(independent.charges):
        _reject("positive charge membership")
    total = F(0)
    for proposed, actual in zip(claim.charges, independent.charges, strict=True):
        if type(proposed) is not RankOneCharge or proposed.node_id != actual.node_id:
            _reject("rank-one charge identity")
        for name in ("middle_norm", "director_norm", "beta_norm", "gamma_norm"):
            _contained(getattr(proposed, name), getattr(actual, name), arithmetic)
        arithmetic.check(proposed.middle_term)
        arithmetic.check(proposed.director_term)
        if proposed.middle_term < arithmetic.qmul(
            proposed.middle_norm.hi, proposed.beta_norm.hi
        ) or proposed.director_term < arithmetic.qmul(
            proposed.director_norm.hi, proposed.gamma_norm.hi
        ):
            _reject("rank-one norm product undercharged")
        if (
            proposed.middle_term < actual.middle_term
            or proposed.director_term < actual.director_term
            or proposed.middle_term < 0
            or proposed.director_term < 0
        ):
            _reject("positive rank-one charges omitted/cancelled")
        total = arithmetic.qadd(
            total, arithmetic.qadd(proposed.middle_term, proposed.director_term)
        )
    arithmetic.check(claim.k)
    if claim.k < total or claim.k < independent.k or claim.k >= 1:
        _reject("tetra rank-one upper bound is not strict")


def _cell_check(
    claim,
    original,
    charts,
    gradients,
    reference_polys,
    reference_law,
    polynomials,
    left,
    right,
    arithmetic,
):
    if type(claim) is not CellWitness or claim.start != left or claim.end != right:
        _reject("cell endpoint/schema")
    arithmetic.check(claim.start)
    arithmetic.check(claim.end)
    arithmetic.set_bits(claim.bits)
    evaluated = _derive(original, polynomials, left, right, arithmetic)
    _contained(claim.domain, evaluated["domain"], arithmetic)
    _contained(claim.pair_incidence, _pair_incidence(original), arithmetic)
    if type(claim.tetrahedra) is not tuple or len(claim.tetrahedra) != len(original["tetrahedra"]):
        _reject("complete tetra coverage")
    node_component = {node: chart.nodes for chart in charts for node in chart.nodes}
    gradient_map = {entry.tetra_id: entry for entry in gradients}
    for proposed, tet in zip(claim.tetrahedra, original["tetrahedra"], strict=True):
        derived = _checked_tetra(
            original,
            tet,
            gradient_map[tet.tetra_id],
            reference_polys,
            reference_law,
            polynomials,
            evaluated,
            node_component[tet.vertices[0][0]],
            left,
            right,
            arithmetic,
        )
        _tetra_enclosure(proposed, derived, arithmetic)
    if type(claim.components) is not tuple or len(claim.components) != len(charts):
        _reject("complete component coverage")
    for proposed, chart in zip(claim.components, charts, strict=True):
        membership = tuple(
            tet.tetra_id for tet in original["tetrahedra"] if tet.vertices[0][0] in chart.nodes
        )
        if (
            type(proposed) is not ComponentWitness
            or proposed.nodes != chart.nodes
            or proposed.tetra_ids != membership
        ):
            _reject("component tetra membership")
        arithmetic.check(proposed.k)
        maximum = max(tet.k for tet in claim.tetrahedra if tet.tetra_id in membership)
        if not maximum <= proposed.k < 1:
            _reject("strict component bound")
    _check_separators(
        claim.separators, charts, polynomials, evaluated["offsets"], left, right, arithmetic
    )


@_checked_verification
def verify_shell_embedding(certificate, reference, positions, *, limits=ShellProofLimits()):
    arithmetic = Arithmetic(limits)
    original = _original(reference, limits, arithmetic)
    positions = _input_array(positions, len(original["nodes"]))
    _header(
        certificate,
        ShellEmbeddingCertificate,
        "conforming-shell-embedding-v1",
        "strict_geometry_certified",
        reference,
        _binding("embedding", reference, positions, limits),
        limits,
    )
    charts, gradients, reference_polys, reference_law = _reference_check(
        certificate.reference, original, arithmetic
    )
    polynomials = _polynomials(original, positions, arithmetic)
    _cell_check(
        certificate.cell,
        original,
        charts,
        gradients,
        reference_polys,
        reference_law,
        polynomials,
        F(0),
        F(1),
        arithmetic,
    )
    _geometry_fields(certificate.geometry, reference, positions, original, arithmetic)
    return True


def _partition(certificate, arithmetic):
    if (
        type(certificate.tree) is not tuple
        or not 1 <= len(certificate.tree) <= arithmetic.limits.max_time_cells
    ):
        _reject("partition tree size")
    nodes, starts = {}, {}
    for cell in certificate.tree:
        if type(cell) is not TimeCell or type(cell.accepted) is not bool:
            _reject("partition tree node type")
        arithmetic.check(cell.start)
        arithmetic.check(cell.end)
        arithmetic.visit_cell(cell.depth)
        if not 0 <= cell.start < cell.end <= 1 or (cell.start, cell.end) in nodes:
            _reject("partition span/duplicate")
        if (cell.depth, cell.start) in starts:
            _reject("duplicate start at the same tree depth")
        nodes[(cell.start, cell.end)] = cell
        starts[(cell.depth, cell.start)] = cell
    pending = [(F(0), F(1), 0)]
    visited, leaves = set(), []
    while pending:
        left, right, depth = pending.pop()
        key = (left, right)
        if key not in nodes or key in visited or nodes[key].depth != depth:
            _reject("partition tree coverage/depth")
        visited.add(key)
        node = nodes[key]
        if node.accepted:
            leaves.append(key)
        else:
            # The producer bisects; a verifier may admit ANY exact rational
            # binary split proved by the complete child spans and depths.
            first_child = starts.get((depth + 1, left))
            if first_child is None or not left < first_child.end < right:
                _reject("nonleaf has no proper first child span")
            split = first_child.end
            second_child = nodes.get((split, right))
            if second_child is None or second_child.depth != depth + 1:
                _reject("nonleaf is missing its exact complementary sibling")
            pending.extend(((split, right, depth + 1), (left, split, depth + 1)))
    if visited != set(nodes):
        _reject("unused tree node/pending sibling")
    if type(certificate.cells) is not tuple or len(certificate.cells) != len(leaves):
        _reject("complete accepted leaf coverage")
    claimed = []
    for cell in certificate.cells:
        if type(cell) is not CellWitness:
            _reject("accepted cell schema")
        arithmetic.check(cell.start)
        arithmetic.check(cell.end)
        claimed.append((cell.start, cell.end))
    if len(set(claimed)) != len(claimed) or set(claimed) != set(leaves):
        _reject("missing/extra/overlapping time cell")
    ordered = sorted(leaves)
    if (
        ordered[0][0] != 0
        or ordered[-1][1] != 1
        or any(first[1] != second[0] for first, second in zip(ordered, ordered[1:]))
    ):
        _reject("gap-free exact partition")


@_checked_verification
def verify_shell_sweep(certificate, reference, path, *, h, limits=ShellProofLimits()):
    arithmetic = Arithmetic(limits)
    original = _original(reference, limits, arithmetic)
    path = _input_array(path, len(original["nodes"]), path=True)
    duration = arithmetic.input_float(h)
    if duration <= 0:
        _reject("positive h")
    _header(
        certificate,
        ShellSweepCertificate,
        "conforming-shell-sweep-v1",
        "strict_geometry_certified",
        reference,
        _binding("sweep", reference, path, limits, h=duration),
        limits,
    )
    arithmetic.check(certificate.h)
    if certificate.h != duration:
        _reject("duration binding")
    _partition(certificate, arithmetic)
    charts, gradients, reference_polys, reference_law = _reference_check(
        certificate.reference, original, arithmetic
    )
    polynomials = _polynomials(original, path, arithmetic, path=True)
    for cell in sorted(certificate.cells, key=lambda value: value.start):
        _cell_check(
            cell,
            original,
            charts,
            gradients,
            reference_polys,
            reference_law,
            polynomials,
            cell.start,
            cell.end,
            arithmetic,
        )
    if type(certificate.endpoint_geometry) is not tuple or len(certificate.endpoint_geometry) != 2:
        _reject("complete endpoint publication coverage")
    for index, endpoint in enumerate(certificate.endpoint_geometry):
        _geometry_fields(endpoint, reference, path[index], original, arithmetic)
    return True
