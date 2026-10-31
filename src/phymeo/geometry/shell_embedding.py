"""Bounded global PL strict embedding over independently checked convex charts."""

from dataclasses import dataclass
from fractions import Fraction as F

from ..constraints.conforming_shell import ShellProofLimits, ShellUnresolved, _topology
from ._shell_intervals import Arithmetic, Interval
from .shell_directors import (
    DomainWitness,
    ShellDirectorGeometry,
    _binding,
    _constant_polynomials,
    _evaluate,
    _geometry_receipt,
    _positions,
)


@dataclass(frozen=True)
class ReferenceChart:
    nodes: tuple
    face_ids: tuple
    boundary: tuple
    axis: tuple
    normal: tuple
    polygon: tuple
    twice_area: F


@dataclass(frozen=True)
class ReferenceGradient:
    tetra_id: tuple
    vertices: tuple
    lambda_gradients: tuple
    determinant: Interval


@dataclass(frozen=True)
class SeparatorWitness:
    left: tuple
    right: tuple
    axis: tuple
    left_support: Interval
    right_support: Interval


@dataclass(frozen=True)
class ReferenceWitness:
    charts: tuple
    gradients: tuple
    separators: tuple
    boundary_facets: tuple
    pair_incidence: tuple


@dataclass(frozen=True)
class RankOneCharge:
    node_id: int
    middle_norm: Interval
    director_norm: Interval
    beta_norm: Interval
    gamma_norm: Interval
    middle_term: F
    director_term: F


@dataclass(frozen=True)
class TetrahedronWitness:
    tetra_id: tuple
    determinant: Interval
    beta_gradients: tuple
    gamma_gradients: tuple
    charges: tuple
    k: F


@dataclass(frozen=True)
class ComponentWitness:
    nodes: tuple
    tetra_ids: tuple
    k: F


@dataclass(frozen=True)
class CellWitness:
    start: F
    end: F
    bits: int
    domain: DomainWitness
    tetrahedra: tuple
    components: tuple
    separators: tuple
    pair_incidence: tuple

    @property
    def k_max(self):
        return max(component.k for component in self.components)


@dataclass(frozen=True)
class ShellEmbeddingCertificate:
    schema: str
    precision: str
    status: str
    reference_id: str
    binding: str
    policy: ShellProofLimits
    reference: ReferenceWitness
    cell: CellWitness
    geometry: ShellDirectorGeometry
    operations: int


def _orient(a, b, c, arithmetic):
    return arithmetic.qcross2(arithmetic.qvsub(b, a), arithmetic.qvsub(c, a))


def _on_segment(p, a, b, arithmetic):
    return _orient(a, b, p, arithmetic) == 0 and all(
        min(x, y) <= z <= max(x, y) for z, x, y in zip(p, a, b, strict=True)
    )


def _segment_intersection(a, b, c, d, arithmetic):
    ab, cd, ac = arithmetic.qvsub(b, a), arithmetic.qvsub(d, c), arithmetic.qvsub(c, a)
    determinant = arithmetic.qcross2(ab, cd)
    if determinant:
        t = arithmetic.qdiv(arithmetic.qcross2(ac, cd), determinant)
        u = arithmetic.qdiv(arithmetic.qcross2(ac, ab), determinant)
        if 0 <= t <= 1 and 0 <= u <= 1:
            return (
                tuple(
                    arithmetic.qadd(x, arithmetic.qmul(t, v)) for x, v in zip(a, ab, strict=True)
                ),
            )
        return ()
    if arithmetic.qcross2(ac, ab):
        return ()
    return tuple(
        sorted(
            set(
                p
                for p in (a, b, c, d)
                if _on_segment(p, a, b, arithmetic) and _on_segment(p, c, d, arithmetic)
            )
        )
    )


def _triangle_intersection(first, second, arithmetic):
    polygon = tuple(first)
    for a, b in zip(second, second[1:] + second[:1], strict=True):
        if not polygon:
            return ()
        output = []
        previous = polygon[-1]
        previous_side = _orient(a, b, previous, arithmetic)
        for current in polygon:
            current_side = _orient(a, b, current, arithmetic)
            if (previous_side < 0 <= current_side) or (current_side < 0 <= previous_side):
                denominator = arithmetic.qsub(previous_side, current_side)
                weight = arithmetic.qdiv(previous_side, denominator)
                edge = arithmetic.qvsub(current, previous)
                output.append(
                    tuple(
                        arithmetic.qadd(x, arithmetic.qmul(weight, delta))
                        for x, delta in zip(previous, edge, strict=True)
                    )
                )
            if current_side >= 0:
                output.append(current)
            previous, previous_side = current, current_side
        cleaned = []
        for point in output:
            if not cleaned or point != cleaned[-1]:
                cleaned.append(point)
        if len(cleaned) > 1 and cleaned[-1] == cleaned[0]:
            cleaned.pop()
        polygon = tuple(cleaned)
    return polygon


def _chart(reference, component, coordinates, arithmetic):
    faces = tuple(
        sorted(
            (int(i), tuple(int(v) for v in face))
            for i, face in zip(reference.face_ids, reference.oriented_nodes, strict=True)
            if int(face[0]) in component
        )
    )
    first = faces[0][1]
    normal = arithmetic.qcross(
        arithmetic.qvsub(coordinates[first[1]], coordinates[first[0]]),
        arithmetic.qvsub(coordinates[first[2]], coordinates[first[0]]),
    )
    if normal == (F(0),) * 3:
        raise ShellUnresolved("unsupported_reference_chart", "zero reference face")
    rho_by_node = {
        int(node): arithmetic.input_float(reference.half_thickness_m[row])
        for row, node in enumerate(reference.node_ids)
    }
    if len({rho_by_node[node] for node in component}) != 1:
        raise ShellUnresolved("unsupported_reference_chart", "nonuniform component thickness")
    origin = coordinates[min(component)]
    for node in component:
        if arithmetic.qdot(normal, arithmetic.qvsub(coordinates[node], origin)):
            raise ShellUnresolved("unsupported_reference_chart", "nonplanar component")
    for _, face in faces:
        area = arithmetic.qcross(
            arithmetic.qvsub(coordinates[face[1]], coordinates[face[0]]),
            arithmetic.qvsub(coordinates[face[2]], coordinates[face[0]]),
        )
        if arithmetic.qcross(normal, area) != (F(0),) * 3 or arithmetic.qdot(normal, area) <= 0:
            raise ShellUnresolved("unsupported_reference_chart", "reference oriented face normals")
    axis = next(index for index in range(3) if normal[index])
    u, v = (axis + 1) % 3, (axis + 2) % 3
    if normal[axis] < 0:
        u, v = v, u
    projected = {node: (coordinates[node][u], coordinates[node][v]) for node in component}
    cycles = tuple(cycle for cycle in reference.boundary_cycles if cycle[0] in component)
    edges = {tuple(sorted((face[i], face[(i + 1) % 3]))) for _, face in faces for i in range(3)}
    euler = arithmetic.qsub(arithmetic.qadd(F(len(component)), F(len(faces))), F(len(edges)))
    if len(cycles) != 1 or euler != 1:
        raise ShellUnresolved(
            "unsupported_reference_chart", "component is not one triangulated disk"
        )
    boundary = cycles[0]
    polygon = tuple(projected[node] for node in boundary)
    if len(set(polygon)) != len(polygon):
        raise ShellUnresolved("unsupported_reference_chart", "repeated geometric boundary vertex")
    for index, a in enumerate(polygon):
        b, c = polygon[(index + 1) % len(polygon)], polygon[(index + 2) % len(polygon)]
        if a == b:
            raise ShellUnresolved("unsupported_reference_chart", "zero boundary edge")
        turn = _orient(a, b, c, arithmetic)
        if turn < 0 or (
            turn == 0 and arithmetic.qdot(arithmetic.qvsub(b, a), arithmetic.qvsub(c, b)) <= 0
        ):
            raise ShellUnresolved("unsupported_reference_chart", "nonconvex/backtracking boundary")
        for node in component:
            if _orient(a, b, projected[node], arithmetic) < 0:
                raise ShellUnresolved(
                    "unsupported_reference_chart", "triangle outside closed convex boundary"
                )
        for other in range(index + 1, len(polygon)):
            c0, d0 = polygon[other], polygon[(other + 1) % len(polygon)]
            intersection = _segment_intersection(a, b, c0, d0, arithmetic)
            shared = set((boundary[index], boundary[(index + 1) % len(boundary)])) & set(
                (boundary[other], boundary[(other + 1) % len(boundary)])
            )
            expected = {projected[node] for node in shared}
            if set(intersection) != expected:
                raise ShellUnresolved("unsupported_reference_chart", "boundary is not simple")
    polygon_area = F(0)
    for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        polygon_area = arithmetic.qadd(polygon_area, arithmetic.qcross2(a, b))
    if polygon_area <= 0:
        raise ShellUnresolved("unsupported_reference_chart", "nonpositive boundary area")
    total = F(0)
    for index, (_, face) in enumerate(faces):
        triangle = tuple(projected[node] for node in face)
        area = _orient(*triangle, arithmetic)
        if area <= 0:
            raise ShellUnresolved("unsupported_reference_chart", "nonpositive projected triangle")
        total = arithmetic.qadd(total, area)
        for _, other in faces[index + 1 :]:
            intersection = _triangle_intersection(
                triangle, tuple(projected[node] for node in other), arithmetic
            )
            shared = tuple(sorted(set(face) & set(other)))
            if not shared:
                correct = not intersection
            elif len(shared) == 1:
                correct = bool(intersection) and all(
                    p == projected[shared[0]] for p in intersection
                )
            elif len(shared) == 2:
                correct = bool(intersection) and all(
                    _on_segment(p, projected[shared[0]], projected[shared[1]], arithmetic)
                    for p in intersection
                )
            else:
                correct = False
            if not correct:
                raise ShellUnresolved(
                    "unsupported_reference_chart",
                    "closed triangle intersection differs from material incidence",
                )
    if total != polygon_area:
        raise ShellUnresolved(
            "unsupported_reference_chart", "triangle area does not cover boundary"
        )
    return ReferenceChart(
        component,
        tuple(face_id for face_id, _ in faces),
        boundary,
        (axis, u, v),
        normal,
        polygon,
        polygon_area,
    )


def _incidence(reference):
    return tuple(
        (first.tetra_id, second.tetra_id, tuple(sorted(set(first.vertices) & set(second.vertices))))
        for index, first in enumerate(reference.tetrahedra)
        for second in reference.tetrahedra[index + 1 :]
    )


def _support(component, axis, origin_poly, polynomials, offsets, left, right, arithmetic):
    axes = tuple((value,) for value in axis)
    projections = []
    for node in component:
        relative = arithmetic.poly_vsub(polynomials[node], origin_poly)
        middle = arithmetic.polynomial_bounds(arithmetic.poly_dot(relative, axes), left, right)
        arm = arithmetic.dot(tuple(arithmetic.exact(v) for v in axis), offsets[node])
        projections.extend((arithmetic.sub(middle, arm), arithmetic.add(middle, arm)))
    return Interval(min(value.lo for value in projections), max(value.hi for value in projections))


def _separations(reference, charts, polynomials, offsets, left, right, arithmetic):
    witnesses = []
    midpoint = arithmetic.qdiv(arithmetic.qadd(left, right), F(2))
    for index, first in enumerate(charts):
        for second in charts[index + 1 :]:
            candidates = [
                (F(1), F(0), F(0)),
                (F(0), F(1), F(0)),
                (F(0), F(0), F(1)),
                first.normal,
                second.normal,
            ]
            means = []
            for chart in (first, second):
                mean = []
                for coordinate in range(3):
                    total = F(0)
                    for node in chart.nodes:
                        value = arithmetic.polynomial_bounds(
                            polynomials[node][coordinate], midpoint, midpoint
                        )
                        total = arithmetic.qadd(total, value.lo)
                    mean.append(arithmetic.qdiv(total, F(len(chart.nodes))))
                means.append(tuple(mean))
            candidates.append(arithmetic.qvsub(means[1], means[0]))
            found = None
            for axis in dict.fromkeys(candidates):
                if axis == (F(0),) * 3:
                    continue
                origin = polynomials[min(first.nodes)]
                a = _support(
                    first.nodes, axis, origin, polynomials, offsets, left, right, arithmetic
                )
                b = _support(
                    second.nodes, axis, origin, polynomials, offsets, left, right, arithmetic
                )
                if a.hi < b.lo or b.hi < a.lo:
                    found = SeparatorWitness(first.nodes, second.nodes, axis, a, b)
                    break
            if found is None:
                raise ShellUnresolved(
                    "embedding_not_certified", "no strict full-component support separator"
                )
            witnesses.append(found)
    return tuple(witnesses)


def _reference_proof(reference, arithmetic):
    topology = _topology(
        reference.node_ids, reference.face_ids, reference.oriented_nodes, arithmetic.limits
    )
    if topology != (
        reference.tetrahedra,
        reference.boundary_facets,
        reference.components,
        reference.boundary_cycles,
    ):
        raise ShellUnresolved(
            "invalid_reference_topology", "stored topology differs from recomputation"
        )
    if (
        len(reference.tetrahedra) * (len(reference.tetrahedra) - 1) // 2
        > arithmetic.limits.max_pairs
    ):
        raise ShellUnresolved("shape_budget", "all tetra pairs")
    polynomials = _constant_polynomials(reference, reference.reference_positions, arithmetic)
    coordinates = {node: tuple(poly[0] for poly in vector) for node, vector in polynomials.items()}
    charts = tuple(
        _chart(reference, component, coordinates, arithmetic) for component in reference.components
    )
    evaluated = _evaluate(reference, polynomials, F(0), F(0), arithmetic)
    # The exact parallel-positive reference normal proof permits one shared unit
    # expression per component, rather than interval-subtracting equal directors.
    units, rho = {}, evaluated["rho"]
    for chart in charts:
        vector = tuple(arithmetic.exact(v) for v in chart.normal)
        length = arithmetic.norm(vector)
        unit = tuple(arithmetic.div(v, length) for v in vector)
        for node in chart.nodes:
            units[node] = unit
    gradients = []
    for tet in reference.tetrahedra:
        node0, sign0 = tet.vertices[0]
        edges = []
        for node, sign in tet.vertices[1:]:
            middle = tuple(
                arithmetic.exact(v) for v in arithmetic.qvsub(coordinates[node], coordinates[node0])
            )
            coefficient = arithmetic.qsub(
                arithmetic.qmul(F(sign), rho[node]), arithmetic.qmul(F(sign0), rho[node0])
            )
            edges.append(
                arithmetic.vadd(
                    middle, arithmetic.scale(arithmetic.exact(coefficient), units[node0])
                )
            )
        inverse, determinant = arithmetic.inverse3(tuple(zip(*edges, strict=True)))
        if determinant.lo <= 0:
            raise ShellUnresolved(
                "unsupported_reference_chart", "reference tetra orientation not positive"
            )
        first_gradient = tuple(
            arithmetic.neg(arithmetic.sum(row[c] for row in inverse)) for c in range(3)
        )
        gradients.append(
            ReferenceGradient(tet.tetra_id, tet.vertices, (first_gradient,) + inverse, determinant)
        )
    separators = _separations(
        reference, charts, polynomials, evaluated["offsets"], F(0), F(0), arithmetic
    )
    witness = ReferenceWitness(
        charts, tuple(gradients), separators, reference.boundary_facets, _incidence(reference)
    )
    return witness, polynomials, evaluated


def _current_determinant(tet, polynomials, evaluated, left, right, arithmetic):
    first_node, first_sign = tet.vertices[0]
    edges = []
    for node, sign in tet.vertices[1:]:
        middle = arithmetic.poly_vector_bounds(
            arithmetic.poly_vsub(polynomials[node], polynomials[first_node]), left, right
        )
        if evaluated["sum_polys"][node] == evaluated["sum_polys"][first_node]:
            coefficient = arithmetic.qsub(
                arithmetic.qmul(F(sign), evaluated["rho"][node]),
                arithmetic.qmul(F(first_sign), evaluated["rho"][first_node]),
            )
            arm = arithmetic.scale(arithmetic.exact(coefficient), evaluated["normals"][first_node])
        else:
            arm = arithmetic.vsub(
                arithmetic.scale(arithmetic.exact(sign), evaluated["offsets"][node]),
                arithmetic.scale(arithmetic.exact(first_sign), evaluated["offsets"][first_node]),
            )
        edges.append(arithmetic.vadd(middle, arm))
    determinant = arithmetic.determinant(tuple(zip(*edges, strict=True)))
    if determinant.hi <= 0:
        raise ShellUnresolved("inverted_or_degenerate_tetra", str(tet.tetra_id))
    if determinant.lo <= 0:
        raise ShellUnresolved("tetra_orientation_unknown", str(tet.tetra_id))
    return determinant


def _cell_proof(
    reference,
    reference_witness,
    reference_polys,
    reference_evaluated,
    polynomials,
    left,
    right,
    arithmetic,
):
    evaluated = _evaluate(reference, polynomials, left, right, arithmetic)
    gradients = {value.tetra_id: value for value in reference_witness.gradients}
    component_by_node = {
        node: chart.nodes for chart in reference_witness.charts for node in chart.nodes
    }
    tetra_witnesses = []
    for tet in reference.tetrahedra:
        determinant = _current_determinant(tet, polynomials, evaluated, left, right, arithmetic)
        reference_gradient = gradients[tet.tetra_id]
        beta, gamma = {}, {}
        zero = (arithmetic.exact(0),) * 3
        for (node, sign), gradient in zip(
            tet.vertices, reference_gradient.lambda_gradients, strict=True
        ):
            beta[node] = arithmetic.vadd(beta.get(node, zero), gradient)
            gamma[node] = arithmetic.vadd(
                gamma.get(node, zero), arithmetic.scale(arithmetic.exact(sign), gradient)
            )
        component = component_by_node[tet.vertices[0][0]]
        anchor = min(component)
        anchor_displacement = arithmetic.poly_vsub(polynomials[anchor], reference_polys[anchor])
        charges, k = [], F(0)
        for node in sorted(beta):
            relative_displacement = arithmetic.poly_vsub(
                arithmetic.poly_vsub(polynomials[node], reference_polys[node]), anchor_displacement
            )
            middle_delta = arithmetic.poly_vector_bounds(relative_displacement, left, right)
            if evaluated["sum_polys"][node] == reference_evaluated["sum_polys"][node]:
                director_delta = zero
            else:
                director_delta = arithmetic.vsub(
                    evaluated["offsets"][node], reference_evaluated["offsets"][node]
                )
            middle_norm, director_norm = (
                arithmetic.norm(middle_delta),
                arithmetic.norm(director_delta),
            )
            beta_norm, gamma_norm = arithmetic.norm(beta[node]), arithmetic.norm(gamma[node])
            middle_term = arithmetic.qmul(middle_norm.hi, beta_norm.hi)
            director_term = arithmetic.qmul(director_norm.hi, gamma_norm.hi)
            k = arithmetic.qadd(k, arithmetic.qadd(middle_term, director_term))
            charges.append(
                RankOneCharge(
                    node,
                    middle_norm,
                    director_norm,
                    beta_norm,
                    gamma_norm,
                    middle_term,
                    director_term,
                )
            )
        tetra_witnesses.append(
            TetrahedronWitness(
                tet.tetra_id,
                determinant,
                tuple(sorted(beta.items())),
                tuple(sorted(gamma.items())),
                tuple(charges),
                k,
            )
        )
    components = []
    for chart in reference_witness.charts:
        included = tuple(
            t.tetra_id for t in reference.tetrahedra if t.vertices[0][0] in chart.nodes
        )
        maximum = max(t.k for t in tetra_witnesses if t.tetra_id in included)
        if maximum >= 1:
            raise ShellUnresolved(
                "embedding_not_certified", "strict positive rank-one bound is not below one"
            )
        components.append(ComponentWitness(chart.nodes, included, maximum))
    separators = _separations(
        reference,
        reference_witness.charts,
        polynomials,
        evaluated["offsets"],
        left,
        right,
        arithmetic,
    )
    return CellWitness(
        left,
        right,
        arithmetic.bits,
        evaluated["domain"],
        tuple(tetra_witnesses),
        tuple(components),
        separators,
        _incidence(reference),
    )


def certify_shell_embedding(reference, positions, *, limits=ShellProofLimits()):
    positions = _positions(reference, positions, limits)
    arithmetic = Arithmetic(limits)
    reference_witness, reference_polys, reference_evaluated = _reference_proof(
        reference, arithmetic
    )
    polynomials = _constant_polynomials(reference, positions, arithmetic)
    cell = _cell_proof(
        reference,
        reference_witness,
        reference_polys,
        reference_evaluated,
        polynomials,
        F(0),
        F(1),
        arithmetic,
    )
    geometry = _geometry_receipt(reference, positions, arithmetic, polynomials)
    return ShellEmbeddingCertificate(
        "conforming-shell-embedding-v1",
        "ideal-from-stored-fp64",
        "strict_geometry_certified",
        reference.reference_id,
        _binding("embedding", reference, positions, limits),
        limits,
        reference_witness,
        cell,
        geometry,
        arithmetic.operations,
    )
