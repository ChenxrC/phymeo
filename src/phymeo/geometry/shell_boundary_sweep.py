"""Closed affine-middle path proof with actual constant-derived-director identities."""

from dataclasses import replace
from fractions import Fraction as F
from itertools import combinations

import numpy as np

from ..constraints.conforming_shell import ConformingShellReference, ShellUnresolved
from ..constraints.shell_boundary import (
    BoundaryProofLimits,
    BoundarySegmentEvidence,
    BoundaryTimeCell,
    ShellBoundarySweep,
    digest,
)
from ._shell_contact_field import Meter
from ._shell_intervals import Publication
from .shell_boundary import _build_static, _coverage, _matching_incidence, _pair


def _fail(category, detail):
    raise ShellUnresolved(category, detail)


def _input_segments(reference, segments, durations, limits):
    if type(reference) is not ConformingShellReference or type(limits) is not BoundaryProofLimits:
        raise TypeError("original public reference and bounded policy required")
    limits.__post_init__()
    if type(segments) not in (tuple, list) or type(durations) not in (tuple, list):
        _fail("invalid_shape_or_identity", "bounded segment and duration tuples")
    if not 1 <= len(segments) <= limits.strict_limits.max_time_cells or len(segments) != len(
        durations
    ):
        _fail("incomplete_time_cover", "segment/duration cover")
    n = len(reference.node_ids)
    if not 3 <= n <= limits.strict_limits.max_nodes:
        _fail("shape_budget", "original path node shape")
    result = []
    for index, segment in enumerate(segments):
        if (
            type(segment) is not np.ndarray
            or segment.dtype != np.dtype("float64")
            or segment.shape != (2, n, 3)
        ):
            _fail("invalid_shape_or_identity", "actual FP64[2,N,3] affine segment")
        if not np.isfinite(segment).all():
            _fail("invalid_shape_or_identity", "finite actual middle path")
        if index and segment[0].tobytes() != segments[index - 1][1].tobytes():
            _fail("branch_or_endpoint_mismatch", "exact original join")
        result.append(np.frombuffer(segment.tobytes(), dtype=np.float64).reshape(segment.shape))
    return tuple(result)


def _qvector_sub(m, left, right):
    return tuple(m.poly_sub(p, q) for p, q in zip(left, right, strict=True))


def _qvector_add(m, left, right):
    return tuple(m.poly_add(p, q) for p, q in zip(left, right, strict=True))


def _actual_polynomials(original, segment, m, order):
    x = {}
    displacements = {}
    for row, node in enumerate(order):
        start = tuple(m.input_float(v) for v in segment[0, row])
        end = tuple(m.input_float(v) for v in segment[1, row])
        delta = m.qvsub(end, start)
        displacements[node] = delta
        x[node] = tuple(m.poly((p, v)) for p, v in zip(start, delta, strict=True))
    stars = {node: ((F(0),),) * 3 for node in original["nodes"]}
    h = {
        i: {
            j: ((F(0),),) * 3
            for j in sorted({v for _, f in original["faces"] if i in f for v in f})
        }
        for i in original["nodes"]
    }
    faces = {}
    for fid, (i, j, k) in original["faces"]:
        e1 = _qvector_sub(m, x[j], x[i])
        e2 = _qvector_sub(m, x[k], x[i])
        e3 = _qvector_sub(m, x[k], x[j])
        cross = m.poly_cross(e1, e2)
        faces[fid] = (e1, e2, e3, cross)
        derivative = (_qvector_sub(m, e2, e1), tuple(tuple(m.qneg(v) for v in p) for p in e2), e1)
        for owner in (i, j, k):
            stars[owner] = _qvector_add(m, stars[owner], cross)
            for node, value in zip((i, j, k), derivative, strict=True):
                h[owner][node] = _qvector_add(m, h[owner][node], value)
    return x, displacements, faces, stars, h


def _kp(a, qpoly):
    return a.poly(tuple(a.rational(v) for v in qpoly))


def _poly_vector_sub(a, x, y):
    return tuple(a.poly_sub(p, q) for p, q in zip(x, y, strict=True))


def _poly_cross(a, x, y):
    return tuple(
        a.poly_sub(a.poly_mul(x[j], y[k]), a.poly_mul(x[k], y[j]))
        for j, k in ((1, 2), (2, 0), (0, 1))
    )


def _poly_dot(a, x, y):
    value = (a.zero,)
    for p, q in zip(x, y, strict=True):
        value = a.poly_add(value, a.poly_mul(p, q))
    return value


def _lifted_polynomials(original, x, normals, a):
    points = {}
    for node in original["nodes"]:
        offset = tuple(a.scale(original["rho"][node], v) for v in normals[node])
        for sign in (-1, 1):
            row = []
            for component in range(3):
                p = list(_kp(a, x[node][component]))
                p[0] = (a.add if sign == 1 else a.sub)(p[0], offset[component])
                row.append(a.poly(tuple(p)))
            points[node, sign] = tuple(row)
    return points


def _director_identity(original, stars, base_raw, a):
    m = a.meter
    identities = []
    positive = []
    scales = {}
    constraints = []
    for node in original["nodes"]:
        polynomial = stars[node]
        coefficients = tuple(
            tuple(p[k] if k < len(p) else F(0) for p in polynomial) for k in range(3)
        )
        first = coefficients[0]
        if first == (F(0),) * 3:
            _fail("zero_star", "actual segment start")
        cross = tuple(m.qcross(c, first) for c in coefficients)
        if any(v for row in cross for v in row):
            _fail("unsupported_constant_director", "actual quadratic star is not collinear")
        dot = m.poly(tuple(m.qdot(c, first) for c in coefficients))
        bound = m.polynomial_bounds(dot, F(0), F(1))
        # Constant-base compatibility is separately established from actual coefficients.
        base = base_raw[2][node]
        if any(m.qcross(first, base)) or m.qdot(first, base) <= 0:
            _fail("branch_or_endpoint_mismatch", "positive constant director root branch")
        scalar = m.poly(tuple(m.qdiv(m.qdot(c, base), base_raw[3][node]) for c in coefficients))
        scales[node] = scalar
        identities.append((node, cross))
        positive.append((node, dot, bound))
        constraints.append((("director-positive", node), _kp(a, dot), "positive"))
    return tuple(identities), tuple(positive), scales, tuple(constraints)


def _translations(original, x, displacements, faces, stars, h, base_raw, a):
    result = []
    m = a.meter
    for component in original["components"]:
        first = component[0]
        delta = displacements[first]
        if not all(displacements[node] == delta for node in component):
            continue
        start_delta = tuple(
            m.qsub(x[first][j][0], original["positions"][first][j]) for j in range(3)
        )
        for node in component:
            actual = tuple(m.qsub(x[node][j][0], original["positions"][node][j]) for j in range(3))
            if actual != start_delta:
                _fail("branch_or_endpoint_mismatch", "component translation from original base")
        local_faces = tuple(fid for fid, face in original["faces"] if face[0] in component)
        for fid in local_faces:
            e1, e2, e3, cross = faces[fid]
            if any(len(p) != 1 for vector in (e1, e2, e3, cross) for p in vector):
                _fail("unsupported_constant_director", "translation edge/face coefficient identity")
            if (
                tuple(p[0] for p in e1) != base_raw[1][fid][0]
                or tuple(p[0] for p in e2) != base_raw[1][fid][1]
            ):
                _fail("branch_or_endpoint_mismatch", "complete original edge transport")
        for node in component:
            if (
                any(len(p) != 1 for p in stars[node])
                or tuple(p[0] for p in stars[node]) != base_raw[2][node]
            ):
                _fail("branch_or_endpoint_mismatch", "complete original star transport")
            if any(len(p) != 1 for vector in h[node].values() for p in vector):
                _fail("branch_or_endpoint_mismatch", "complete H transport")
        result.append(
            (
                component,
                start_delta,
                delta,
                local_faces,
                tuple((i, tuple(sorted(h[i].items()))) for i in component),
            )
        )
    return tuple(result)


def _geometry_polynomials(original, faces, stars, y, base, hint, a):
    # ID-only preflight bounds the entire prospective list before polynomial work.
    count = 10 * len(original["faces"]) + len(original["nodes"]) + len(original["tets"])
    count += sum(
        len(p.supports) + len(p.triangles) + len(p.edge_tests)
        for c in base.clusters
        for p in c.planes
    )
    count += sum(len(f.occupied_cones) for f in base.full_strata_and_fans if f.exterior_direction)
    if count > 8192:
        _fail("shape_budget", "whole geometry predicate list")
    m = a.meter
    eps2 = m.qmul(m.input_float(1e-10), m.input_float(1e-10))
    area2 = {}
    star2 = {}
    constraints = []
    for fid, face in original["faces"]:
        e1, e2, e3, c = faces[fid]
        c2 = m.poly_dot(c, c)
        area2[fid] = c2
        constraints.append((("face-area", fid), _kp(a, c2), "positive"))
        for index, edge in enumerate((e1, e2, e3)):
            squared = m.poly_dot(edge, edge)
            scaled = tuple(m.qmul(eps2, v) for v in m.poly_mul(squared, squared))
            constraints.append(
                (("face-quality", fid, index), _kp(a, m.poly_sub(c2, scaled)), "positive")
            )
    for node in original["nodes"]:
        squared = m.poly_dot(stars[node], stars[node])
        star2[node] = squared
        total = (F(0),)
        for fid in original["stars"][node]:
            total = m.poly_add(total, area2[fid])
        factor = m.qmul(eps2, F(len(original["stars"][node])))
        margin = m.poly_sub(squared, tuple(m.qmul(factor, v) for v in total))
        constraints.append((("star-quality", node), _kp(a, margin), "positive"))
    for fid, face in original["faces"]:
        for node in face:
            dot = m.poly_dot(stars[node], faces[fid][3])
            margin = m.poly_sub(
                m.poly_mul(dot, dot),
                tuple(m.qmul(F(1, 16), v) for v in m.poly_mul(star2[node], area2[fid])),
            )
            constraints.append((("projection-dot", fid, node), _kp(a, dot), "positive"))
            constraints.append((("projection-ratio", fid, node), _kp(a, margin), "nonnegative"))
    planes = {}
    for cluster in base.clusters:
        for index, plane in enumerate(cluster.planes):
            t = plane.triangles[0]
            # Recover the original oriented facet, not sorted material orientation.
            oriented = next(f for f in cluster.boundary_facets if tuple(sorted(f)) == t)
            p, q, r = (y[v] for v in oriented)
            normal = _poly_cross(a, _poly_vector_sub(a, q, p), _poly_vector_sub(a, r, p))
            planes[cluster.cluster_id, index] = normal
            vertices = tuple(
                sorted({v for tid in cluster.tetra_ids for v in original["tets"][tid]})
            )
            for vertex in vertices:
                value = _poly_dot(a, normal, _poly_vector_sub(a, y[vertex], p))
                relation = "zero" if vertex in plane.vertices else "negative"
                constraints.append(
                    (("cluster-support", cluster.cluster_id, index, vertex), value, relation)
                )
            axes = tuple(i for i in range(3) if i != plane.drop_axis)
            for triangle in plane.triangles:
                oriented = next(f for f in cluster.boundary_facets if tuple(sorted(f)) == triangle)
                p0, p1, p2 = (y[v] for v in oriented)
                u, v = _poly_vector_sub(a, p1, p0), _poly_vector_sub(a, p2, p0)
                area = a.poly_sub(
                    a.poly_mul(u[axes[0]], v[axes[1]]), a.poly_mul(u[axes[1]], v[axes[0]])
                )
                target = a.sign(next(row[1] for row in plane.orientations if row[0] == triangle))
                constraints.append(
                    (
                        ("face-orientation", cluster.cluster_id, index, triangle),
                        area,
                        "positive" if target > 0 else "negative",
                    )
                )
            for first, second, vertex in (row[0] for row in plane.edge_tests):
                u = _poly_vector_sub(a, y[second], y[first])
                v = _poly_vector_sub(a, y[vertex], y[first])
                area = a.poly_sub(
                    a.poly_mul(u[axes[0]], v[axes[1]]), a.poly_mul(u[axes[1]], v[axes[0]])
                )
                initial = next(
                    row[1] for row in plane.edge_tests if row[0] == (first, second, vertex)
                )
                sign = a.sign(initial)
                constraints.append(
                    (
                        ("face-edge", cluster.cluster_id, index, first, second, vertex),
                        area,
                        "zero" if sign == 0 else "positive" if sign > 0 else "negative",
                    )
                )
        for tid in cluster.tetra_ids:
            t = original["tets"][tid]
            p, q, r, s = (y[v] for v in t)
            determinant = _poly_dot(
                a,
                _poly_vector_sub(a, q, p),
                _poly_cross(a, _poly_vector_sub(a, r, p), _poly_vector_sub(a, s, p)),
            )
            constraints.append((("tetra", tid), determinant, "positive"))
    for fan in base.full_strata_and_fans:
        if not fan.exterior_direction:
            continue
        direction = tuple((a.element(v),) for v in fan.exterior_direction)
        for cid, indices in fan.occupied_cones:
            # The actual positive base witness is named; prove it throughout the cell.
            choice = next(
                index
                for owner, index in (row[0] for row in fan.exterior_facts)
                if owner == cid
                and a.sign(next(row[1] for row in fan.exterior_facts if row[0] == (owner, index)))
                > 0
            )
            constraints.append(
                (
                    ("exterior", fan.material_stratum, cid, choice),
                    _poly_dot(a, planes[cid, choice], direction),
                    "positive",
                )
            )
    plane_paths = tuple(
        (cluster.cluster_id, index, y[plane.vertices[0]], planes[cluster.cluster_id, index])
        for cluster in base.clusters
        for index, plane in enumerate(cluster.planes)
    )
    return tuple(constraints), plane_paths


def _pair_polynomials(original, base, hint, y, a, transported):
    if transported:
        return tuple((p.clusters, "component_translation", p) for p in base.cluster_pairs), ()
    clusters = {c.cluster_id: c for c in base.clusters}
    boundary = {
        tuple(sorted(s))
        for f in original["boundary"]
        for n in range(1, 4)
        for s in combinations(f, n)
    }
    records = []
    constraints = []
    for proposal in sorted(hint.pairs, key=lambda p: p.clusters):
        left, right = proposal.clusters
        if len(proposal.supports) > a.limits.max_support_descents:
            _fail("shape_budget", "path support descent cap")
        active_a = tuple(sorted({v for p in clusters[left].planes for v in p.vertices}))
        active_b = tuple(sorted({v for p in clusters[right].planes for v in p.vertices}))
        steps = []
        axes = []
        empty = False
        for index, support in enumerate(proposal.supports):
            normal = tuple(a.element(q) for q in support.normal)
            if not any(q != a.zero for q in normal) or support.anchor not in active_a:
                _fail("unsupported_matching", "path support normal/active anchor")
            if len(axes) == 1 and all(v == a.zero for v in a.cross(axes[0], normal)):
                _fail("unsupported_matching", "path support rank")
            if len(axes) == 2 and a.determinant(axes[0], axes[1], normal) == a.zero:
                _fail("unsupported_matching", "path third support rank")
            axes.append(normal)
            npoly = tuple((v,) for v in normal)
            groups = []
            for side, vertices in enumerate((active_a, active_b)):
                values = []
                for vertex in vertices:
                    polynomial = _poly_dot(
                        a, npoly, _poly_vector_sub(a, y[vertex], y[support.anchor])
                    )
                    if len(polynomial) > 2:
                        _fail("unsupported_witness", "constant support of affine lifted paths")
                    label = ("pair", proposal.clusters, index, side, vertex)
                    constraints.append(
                        (label, polynomial, "nonpositive" if side == 0 else "nonnegative")
                    )
                    values.append((vertex, polynomial))
                groups.append(tuple(values))
            zero_a = tuple(v for v, p in groups[0] if p == (a.zero,))
            zero_b = tuple(v for v, p in groups[1] if p == (a.zero,))
            steps.append(
                (normal, support.anchor, active_a, active_b, groups[0], groups[1], zero_a, zero_b)
            )
            if not zero_a or not zero_b:
                empty = True
                break
            active_a, active_b = zero_a, zero_b
        if empty:
            records.append((proposal.clusters, "open_empty", tuple(steps), (), ()))
            continue
        mapping = dict(proposal.matching)
        if (
            len(mapping) != len(proposal.matching)
            or set(mapping) != set(active_a)
            or set(mapping.values()) != set(active_b)
            or len(set(mapping.values())) != len(mapping)
        ):
            _fail("unsupported_matching", "persistent full matching bijection")
        for v, w in sorted(mapping.items()):
            if any(a.poly_sub(p, q) != (a.zero,) for p, q in zip(y[v], y[w], strict=True)):
                _fail("unsupported_matching", "persistent exact matching identity")
        complexes = []
        for cid, vertices in ((left, active_a), (right, active_b)):
            complexes.append(
                {
                    tuple(sorted(s))
                    for f in clusters[cid].boundary_facets
                    for n in range(1, 4)
                    for s in combinations(f, n)
                    if set(s) <= set(vertices)
                }
            )
        if {tuple(sorted(mapping[v] for v in s)) for s in complexes[0]} != complexes[1]:
            _fail("unsupported_matching", "persistent entire complex matching")
        pieces = []
        for s in sorted(complexes[0]):
            t = tuple(sorted(mapping[v] for v in s))
            if all(mapping[v] == v for v in s):
                label = "material_incidence"
            elif s in boundary and t in boundary:
                label = "true_boundary_contact"
            else:
                _fail("forbidden_interior_identification", "persistent material preimages")
            pieces.append((s, t, label))
        records.append(
            (
                proposal.clusters,
                "persistent_matching",
                tuple(steps),
                tuple(sorted(mapping.items())),
                tuple(pieces),
                _matching_incidence(clusters[left], clusters[right], mapping, sorted(complexes[0])),
            )
        )
    return tuple(records), tuple(constraints)


def _map_functions(original, base, scales, h, roots, normals, a, constant):
    if constant:
        return tuple(
            (
                b.vertex,
                b.node_ids,
                tuple(
                    tuple(tuple(((q,), (a.one,)) for q in row) for row in matrix)
                    for matrix in b.exact_jacobian
                ),
            )
            for b in base.basis
        ), ()
    m = a.meter
    records = []
    constraints = []
    for node in original["nodes"]:
        denominator = _kp(a, scales[node])
        constraints.append((("map-denominator", node), denominator, "positive"))
        factor = a.scale(original["rho"][node], a.inverse(roots[node]))
        d = tuple(
            tuple(
                a.mul(
                    factor, a.sub(a.rational(F(i == j)), a.mul(normals[node][i], normals[node][j]))
                )
                for j in range(3)
            )
            for i in range(3)
        )
        offsets = {}
        for neighbor, v in h[node].items():
            x, y, z = v

            def minus(p):
                return tuple(m.qneg(c) for c in p)

            zero = (F(0),)
            skew = ((zero, minus(z), y), (z, zero, minus(x)), (minus(y), x, zero))
            rows = []
            for row in range(3):
                columns = []
                for column in range(3):
                    value = (a.zero,)
                    for inner in range(3):
                        term = a.poly(tuple(a.scale(c, d[row][inner]) for c in skew[inner][column]))
                        value = a.poly_add(value, term)
                    columns.append(value)
                rows.append(tuple(columns))
            offsets[neighbor] = tuple(rows)
        for sign in (-1, 1):
            blocks = []
            for neighbor in h[node]:
                matrix = []
                for row in range(3):
                    columns = []
                    for column in range(3):
                        identity = denominator if node == neighbor and row == column else (a.zero,)
                        numerator = (a.poly_add if sign == 1 else a.poly_sub)(
                            identity, offsets[neighbor][row][column]
                        )
                        columns.append(a.rational_function(numerator, denominator))
                    matrix.append(tuple(columns))
                blocks.append(tuple(matrix))
            records.append(((node, sign), tuple(h[node]), tuple(blocks)))
    return tuple(records), tuple(constraints)


def _predicate_holds(polynomial, relation, bound, a):
    if relation == "zero":
        return polynomial == (a.zero,)
    if relation == "positive":
        return bound.lo > 0
    if relation == "negative":
        return bound.hi < 0
    if relation == "nonnegative":
        return bound.lo >= 0
    if relation == "nonpositive":
        return bound.hi <= 0
    _fail("unsupported_witness", "unknown polynomial relation")


def _cells(constraints, a):
    """Root was charged on segment entry; every created child is charged here."""
    if len(constraints) > 8192:
        _fail("shape_budget", "bounded whole-cell evidence")
    cells = [None]

    def prove(index, left, right, depth):
        facts = []
        accepted = True
        for key, polynomial, relation in constraints:
            polynomial = a.poly(polynomial)
            if relation == "zero" and polynomial != (a.zero,):
                _fail("unsupported_witness", "a residual interval is not an identity")
            bound = a.polynomial_bounds(polynomial, left, right)
            facts.append((key, polynomial, relation, bound))
            if not _predicate_holds(polynomial, relation, bound, a):
                accepted = False
                if relation in ("positive", "negative"):
                    for endpoint in (left, right):
                        sign = a.sign(a.poly_evaluate(polynomial, endpoint))
                        if (relation == "positive" and sign <= 0) or (
                            relation == "negative" and sign >= 0
                        ):
                            _fail("law_domain_unknown", "nonpositive strict closed endpoint")
        if accepted:
            cells[index] = BoundaryTimeCell((left, right), depth, (), tuple(facts), a.meter.bits)
            return
        if depth >= a.limits.strict_limits.max_time_depth:
            _fail("unresolved_event_time", "closed polynomial sign depth")
        middle = a.meter.qdiv(a.meter.qadd(left, right), F(2))
        children = (len(cells), len(cells) + 1)
        a.meter.visit_cell(depth + 1)
        a.meter.visit_cell(depth + 1)
        cells.extend((None, None))
        cells[index] = BoundaryTimeCell((left, right), depth, children, tuple(facts), a.meter.bits)
        prove(children[0], left, middle, depth + 1)
        prove(children[1], middle, right, depth + 1)

    prove(0, F(0), F(1), 0)
    return tuple(cells)


def _endpoint_basis(base, endpoint, order, original, normals, functions, a, time, constant):
    lookup = {vertex: (ids, matrices) for vertex, ids, matrices in functions}
    actual = {
        node: tuple(a.meter.input_float(q) for q in endpoint[row]) for row, node in enumerate(order)
    }
    offsets = {
        node: tuple(a.scale(original["rho"][node], q) for q in normals[node]) for node in order
    }
    result = []
    for b in base.basis:
        node, sign = b.vertex
        point = tuple(
            (a.add if sign == 1 else a.sub)(a.rational(q), offset)
            for q, offset in zip(actual[node], offsets[node], strict=True)
        )
        publication = a.publish(point, (3,))
        if constant:
            exact = b.exact_jacobian
            jacobian = Publication(
                b.jacobian.intervals, b.jacobian.values, b.jacobian.errors, b.jacobian.shape
            )
        else:
            ids, matrices = lookup[b.vertex]
            if ids != b.node_ids:
                _fail("incomplete_material_cover", "whole-path full-star IDs")
            exact = tuple(
                tuple(
                    tuple(a.div(a.poly_evaluate(p, time), a.poly_evaluate(q, time)) for p, q in row)
                    for row in matrix
                )
                for matrix in matrices
            )
            jacobian = a.publish(
                tuple(v for matrix in exact for row in matrix for v in row), (len(ids), 3, 3)
            )
        result.append(
            replace(
                b, exact_point=point, exact_jacobian=exact, point=publication, jacobian=jacobian
            )
        )
    return tuple(result)


def certify_shell_boundary_sweep(
    reference, segments, *, durations, witness_hint, limits=BoundaryProofLimits()
):
    paths = _input_segments(reference, segments, durations, limits)
    meter = Meter(limits.strict_limits)
    exact_durations = tuple(meter.input_float(v) for v in durations)
    if any(v <= 0 for v in exact_durations):
        _fail("incomplete_time_cover", "positive durations")
    total = F(0)
    for duration in exact_durations:
        total = meter.qadd(total, duration)
    base, context = _build_static(reference, paths[0][0], witness_hint, limits, meter)
    original, base_raw, a, roots, normals, points = context
    clusters = {c.cluster_id: c for c in base.clusters}
    boundary = {
        tuple(sorted(s))
        for f in original["boundary"]
        for n in range(1, 4)
        for s in combinations(f, n)
    }
    order = tuple(map(int, reference.node_ids))
    previous_basis = base.basis
    previous_contacts = (base.cluster_pairs, base.all_pairs, base.contacts)
    receipts = []
    for path, duration in zip(paths, exact_durations, strict=True):
        meter.visit_cell(0)
        x, displacements, faces, stars, h = _actual_polynomials(original, path, meter, order)
        cross, positive, scales, director_constraints = _director_identity(
            original, stars, base_raw, a
        )
        translation = _translations(original, x, displacements, faces, stars, h, base_raw, a)
        constant = len(translation) == len(original["components"])
        global_translation = constant and len({(v[1], v[2]) for v in translation}) == 1
        y = _lifted_polynomials(original, x, normals, a)
        if constant:
            geometry = ()
            plane_paths = tuple(
                (cluster.cluster_id, index, y[plane.vertices[0]], tuple((v,) for v in plane.normal))
                for cluster in base.clusters
                for index, plane in enumerate(cluster.planes)
            )
        else:
            geometry, plane_paths = _geometry_polynomials(
                original, faces, stars, y, base, witness_hint, a
            )
        pair_paths, pair_constraints = _pair_polynomials(
            original, base, witness_hint, y, a, global_translation
        )
        functions, map_constraints = _map_functions(
            original, base, scales, h, roots, normals, a, constant
        )
        constraints = director_constraints + geometry + pair_constraints + map_constraints
        cells = _cells(constraints, a)
        # Exact joins authorize immutable previous endpoint reuse; they do not prove the path.
        for b in previous_basis:
            evaluated = tuple(p[0] for p in y[b.vertex])
            if evaluated != b.exact_point:
                _fail("branch_or_endpoint_mismatch", "owned start endpoint")
        end_basis = _endpoint_basis(
            base, path[1], order, original, normals, functions, a, F(1), constant
        )
        end_points = {b.vertex: b.exact_point for b in end_basis}
        if global_translation:
            end_pairs = base.cluster_pairs
        else:
            end_pairs = tuple(
                _pair(p, clusters, end_points, boundary, a)
                for p in sorted(witness_hint.pairs, key=lambda p: p.clusters)
            )
        all_pairs, contacts = _coverage(
            original, clusters, end_pairs, base.full_strata_and_fans, end_points
        )
        end_contacts = (end_pairs, all_pairs, contacts)
        receipts.append(
            BoundarySegmentEvidence(
                path,
                duration,
                tuple(sorted(stars.items())),
                cross,
                positive,
                translation,
                tuple(sorted(y.items())),
                plane_paths,
                geometry,
                pair_paths,
                cells,
                (previous_basis, end_basis),
                (previous_contacts, end_contacts),
                functions,
                meter.bits,
            )
        )
        previous_basis, previous_contacts = end_basis, end_contacts
    before = meter.operations
    binding = digest(
        (
            "sweep",
            reference.reference_id,
            reference.node_ids,
            reference.face_ids,
            reference.oriented_nodes,
            paths,
            exact_durations,
            limits,
            witness_hint,
        )
    )
    certificate = ShellBoundarySweep(
        reference.reference_id,
        binding,
        limits,
        exact_durations,
        base,
        tuple(receipts),
        before,
        before,
        0,
        meter.created_cells,
        meter.peak_fraction_bits,
    )
    from ..validation.shell_boundary import _check_sweep

    if not _check_sweep(
        certificate, reference, paths, durations, witness_hint, limits, meter, pending=True
    ):
        _fail("independent_verification_failed", "same-counter whole-path verifier")
    return replace(
        certificate,
        operations=meter.operations,
        producer_operations=before,
        checker_operations=meter.operations - before,
        created_cells=meter.created_cells,
        peak_fraction_bits=meter.peak_fraction_bits,
    )
