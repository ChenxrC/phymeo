"""Actual S2-L phase, geometry and internal energy/angular acceptance."""

from fractions import Fraction as F

import numpy as np

from ..constraints.cloth_self_response import owned
from ..core.capabilities import SolverFailure
from ..geometry._prism_pair_exact import sqrt_bounds
from ..geometry.cloth_self_anchors import q, dot, cross, sub, proof_data, require_area_path, require_fixed_support
from ..geometry.prism_prism_manifold import prism_prism_manifold
from ..geometry.cloth_self_candidates import candidate_self_pairs
from ..geometry.prism_prism_sweep import verify_prism_prism_sweep


def upper(value):
    result = float(value)
    if q(result) < value:
        result = float(np.nextafter(result, np.inf))
    if not np.isfinite(result):
        raise SolverFailure("physical ledger overflow")
    return result


def _elastic(material, positions, budget):
    low = high = F(0)
    if material is None:
        return low, high
    for ids, rest, compliance in zip(material.indices, material.rest_length, material.compliance, strict=True):
        if compliance == 0:
            continue
        delta = sub(tuple(map(q, positions[ids[0]])), tuple(map(q, positions[ids[1]])))
        bits = 64
        while True:
            a, b = sqrt_bounds(dot(delta, delta), bits)
            a, b = a-q(rest), b-q(rest)
            lo = F(0) if a <= 0 <= b else min(a*a, b*b)
            hi = max(a*a, b*b)
            lo, hi = lo/(2*q(compliance)), hi/(2*q(compliance))
            if hi-lo <= budget/(128*max(1, len(material.indices))):
                break
            if bits == 4096:
                raise SolverFailure("elastic energy enclosure exhausted")
            bits = min(2*bits, 4096)
        low, high = low+lo, high+hi
    return low, high


def kinetic(model, velocity):
    return sum(q(v)**2/(2*q(w)) for row, w in zip(velocity, model.inverse_mass, strict=True) for v in row)


def phase_residual(data, *, phase, law, h, budgets):
    world, normals, velocity, gap, scale = (data[name] for name in ("world", "normals", "velocity", "gaps", "scale"))
    maximum = F(0)
    rows = []
    for i, (p, n, u, g, k) in enumerate(zip(world, normals, velocity, gap, scale, strict=True)):
        p, n, u = tuple(map(q, p)), tuple(map(q, n)), tuple(map(q, u))
        normal = dot(n, p)
        if normal < 0:
            raise SolverFailure("negative actual represented phase normal impulse")
        un = dot(n, u)
        tangent = sub(p, tuple(normal*v for v in n))
        slip = sub(u, tuple(un*v for v in n))
        capacity = normal
        if phase == "held" and not data["initial_closed"][i]:
            capacity = F(0)
        if phase == "terminal" and not data["initial_closed"][i]:
            capacity += q(data["held_normal"][i])
        radius = q(law.mu_dynamic)*capacity
        nl = q(g)/q(h) if phase == "held" else un
        residual = max(F(0), -nl, abs(min(q(k)*normal, nl)))
        tl, th = sqrt_bounds(dot(tangent, tangent), 128)
        sl, sh = sqrt_bounds(dot(slip, slip), 128)
        residual = max(residual, max(F(0), th-radius)*q(k))
        if radius == 0:
            residual = max(residual, th*q(k))
        elif sl > q(budgets.velocity_mps)/64:
            # Interval enclosure of t + radius * slip/|slip|; no single-angle choice.
            for a, b in zip(tangent, slip, strict=True):
                ends = (a+radius*b/sl, a+radius*b/sh)
                residual = max(residual, *(abs(v)*q(k) for v in ends))
        maximum = max(maximum, residual)
        rows.append(dict(normal_ns=float(normal), capacity_ns=float(radius), normal_speed_mps=float(un), residual_mps=upper(residual)))
    return upper(maximum), tuple(rows)


class SelfContactResponseLedger:
    def __init__(self, model, topology, *, law, material, proof_limits, inner_settings, context):
        self.model, self.topology = model, topology
        self.law, self.material, self.proof_limits = law, material, proof_limits
        self.inner_settings, self.context = inner_settings, context

    def require(self, original, selected_pre, final, *, original_inputs, h, geometry,
                anchors, held, terminal, material_evidence, budgets):
        domain = self.topology.domain
        a, b, c = (state.domains[domain] for state in (original, selected_pre, final))
        if not np.array_equal(b.position, c.position):
            raise SolverFailure("terminal impulse changed positions")
        if not material_evidence or material_evidence.get("schema") != "xpbd-distance-path-v1":
            raise SolverFailure("unbound or unsupported material path evidence")
        from .material_path import MaterialPathLedger
        from .cloth_self_mapping import SelfContactMappingLedger, verify_capacity_witness
        hr = held["actual_receipt"]
        tr = terminal["actual_receipt"]
        if (not np.array_equal(hr["original_force"], original_inputs.domains[domain].force)
                or not np.array_equal(hr["gravity"], self.model.gravity) or q(hr["h"]) != q(h)
                or not np.array_equal(hr["predicted_velocity"], material_evidence["predicted_velocity"])
                or not np.array_equal(tr["final_velocity"], c.velocity)):
            raise SolverFailure("capacity source snapshots do not match actual selected path")
        MaterialPathLedger(self.model, self.material, a, original_inputs.domains[domain], h,
                           self.inner_settings, self.context).require(b, hr["effective_inputs"],
                           held["plan"].node_impulses.astype(object), hr["pre_diagnostics"], sweeps=hr["sweeps"])
        residual_reports = []
        for phase, data, phase_before in (("held", held, a), ("terminal", terminal, b)):
            if not np.array_equal(data["world"], data["plan"].world_impulses):
                raise SolverFailure("phase proposal differs from its bound actual plan")
            receipt = data["actual_receipt"]
            SelfContactMappingLedger(self.model, proof_limits=self.proof_limits).require(data["plan"],
                original_positions=phase_before.position, original_velocity=phase_before.velocity,
                actual_replay_or_terminal_receipt=receipt, budgets=budgets)
            verify_capacity_witness(data["witness"], data["plan"], receipt, self.model.inverse_mass, phase_before.velocity, budgets, original_positions=phase_before.position, proof_limits=self.proof_limits)
            residual_reports.append(_residual_global(data["witness"], phase_before, b, self.model, phase, h, budgets))
        faces = self.topology.triangles
        areas = tuple(require_area_path(a.position[face], b.position[face]) for face in faces)
        expected_pairs = {tuple(map(int, row)) for row in candidate_self_pairs(self.topology, a.position, b.position)}
        supplied_pairs = [tuple(sorted(map(int, item["identity"].face_ids))) for item in geometry]
        if len(set(supplied_pairs)) != len(supplied_pairs) or set(supplied_pairs) != expected_pairs:
            raise SolverFailure("geometry receipt omitted or duplicated a swept candidate")
        certificates = []
        for item in geometry:
            identity = item["identity"]
            if identity.domain != domain or not np.array_equal(item["thickness"], self.topology.half_thickness_m[identity.face_ids]):
                raise SolverFailure("sweep domain/thickness does not match frozen topology")
            for face, nodes in zip(identity.face_ids, identity.node_ids, strict=True):
                expected = tuple(map(int, self.topology.triangles[face]))
                actual = tuple(map(int, nodes))
                if actual not in (expected, expected[1:]+expected[:1], expected[2:]+expected[:2]):
                    raise SolverFailure("sweep oriented node identity does not match topology")
            args = dict(identity=item["identity"], half_thickness_m=item["thickness"], h=h,
                        penetration_budget_m=budgets.penetration_m, proof_limits=self.proof_limits)
            if not verify_prism_prism_sweep(item["certificate"], item["path_a"], item["path_b"], **args):
                raise SolverFailure("independent two-normal sweep verification failed")
            for side, key in enumerate(("path_a", "path_b")):
                ids = item["identity"].node_ids[side]
                if not np.array_equal(item[key], np.stack((a.position[ids], b.position[ids]))):
                    raise SolverFailure("geometry certificate is stale for actual path")
            manifold = prism_prism_manifold(item["path_a"][1], item["path_b"][1],
                                           identity=item["identity"], half_thickness_m=item["thickness"],
                                           proof_limits=self.proof_limits)
            if manifold.exact_gap_bounds is None or manifold.exact_gap_bounds[0] < -q(budgets.penetration_m):
                raise SolverFailure("actual final whole-prism penetration is not certified")
            pair_rows = [i for i, pair in enumerate(anchors.pair_index) if tuple(sorted(map(int, anchors.identities[int(pair)].face_ids))) == tuple(sorted(map(int, item["identity"].face_ids)))]
            if pair_rows and any(np.any(held["world"][i] != 0) or np.any(terminal["world"][i] != 0) for i in pair_rows):
                if manifold.status == "unresolved" or not manifold.maximal_axes or any(not axis.points for axis in manifold.maximal_axes):
                    raise SolverFailure("loaded final complete clipping evidence is unresolved")
                direction = anchors.directions[pair_rows[0]]
                if any(any(cross(direction, axis.direction)) or dot(direction, axis.direction) <= 0 for axis in manifold.maximal_axes):
                    raise SolverFailure("loaded final ray differs from owned ray")
                original_manifold = anchors.manifolds[int(anchors.pair_index[pair_rows[0]])]
                address_keys = lambda m: {tuple(tuple(v.token for v in material) for material in (point.material_a, point.material_b)) for axis in m.maximal_axes for point in axis.points}
                if address_keys(manifold) != address_keys(original_manifold):
                    raise SolverFailure("loaded complete ideal extrema changed material support")
            certificates.append(proof_data(item["certificate"]))
        if len({token[:-1] for token in anchors.owner_tokens}) != len(anchors.owner_tokens):
            raise SolverFailure("duplicate material capacity under another birth generation")
        for row, pair in enumerate(anchors.pair_index):
            identity = anchors.identities[int(pair)]
            original_manifold = prism_prism_manifold(a.position[identity.node_ids[0]], a.position[identity.node_ids[1]],
                identity=identity, half_thickness_m=self.topology.half_thickness_m[identity.face_ids], proof_limits=self.proof_limits)
            if original_manifold.exact_gap_bounds is None:
                raise SolverFailure("initial capacity phase is unresolved")
            expected_closed = original_manifold.exact_gap_bounds[1] <= 0
            if any(bool(data["initial_closed"][row]) != expected_closed for data in (held, terminal)):
                raise SolverFailure("capacity initial/open phase binding changed")
        loaded = np.any(held["world"] != 0, axis=1)
        continuation = require_fixed_support(anchors, a.position, b.position, loaded)
        hr, hrows = _exact_phase(anchors, b, held, held, phase="held", model=self.model, law=self.law, h=h, budgets=budgets, limits=self.proof_limits)
        tr, trows = _exact_phase(anchors, c, terminal, held, phase="terminal", model=self.model, law=self.law, h=h, budgets=budgets, limits=self.proof_limits)
        if max(hr, tr) > budgets.velocity_mps:
            raise SolverFailure("actual held/terminal Signorini or Coulomb law failed")
        ellh, ellt = held["represented_nodes"], terminal["represented_nodes"]
        exact_h, exact_t = (_witness_values(data["witness"], "ell_exact") for data in (held, terminal))
        n = len(self.model.inverse_mass)
        owned(ellh, (n, 3), name="actual held nodes")
        owned(ellt, (n, 3), name="actual terminal nodes")
        kh = kinetic(self.model, b.velocity)
        k0, k1 = kinetic(self.model, a.velocity), kinetic(self.model, c.velocity)
        u0 = _elastic(self.material, a.position, q(budgets.energy_j))
        u1 = _elastic(self.material, c.position, q(budgets.energy_j))
        original_work = held_work = terminal_linear = terminal_quadratic = F(0)
        angular = [F(0)]*3
        linear = [F(0)]*3
        held_moment, terminal_moment = [F(0)]*3, [F(0)]*3
        for i, inverse in enumerate(self.model.inverse_mass):
            w = q(inverse)
            x0, x1, v0, vp, v1 = (tuple(map(q, value[i])) for value in (a.position, c.position, a.velocity, b.velocity, c.velocity))
            fh, ft = tuple(exact_h[3*i:3*i+3]), tuple(exact_t[3*i:3*i+3])
            external = tuple(q(original_inputs.domains[domain].force[i, j])+q(self.model.gravity[j])/w for j in range(3))
            displacement = sub(x1, x0)
            original_work += dot(external, displacement)
            held_work += dot(fh, displacement)/q(h)
            terminal_linear += dot(ft, vp)
            terminal_quadratic += w*dot(ft, ft)/2
            oldmoment = cross(x0, tuple(v/w for v in v0))
            newmoment = cross(x1, tuple(v/w for v in v1))
            externalmoment = cross(x0, tuple(q(h)*f for f in external))
            mh, mt = cross(x0, fh), cross(x1, ft)
            for j in range(3):
                angular[j] += newmoment[j]-oldmoment[j]-externalmoment[j]
                linear[j] += (v1[j]-v0[j])/w-q(h)*external[j]
                held_moment[j] += mh[j]
                terminal_moment[j] += mt[j]
        terminal_identity = abs(k1-kh-terminal_linear-terminal_quadratic)
        if terminal_identity > q(budgets.energy_j):
            raise SolverFailure("actual all-node terminal kinetic identity failed")
        change = (k1-k0+u1[0]-u0[1]-original_work, k1-k0+u1[1]-u0[0]-original_work)
        positive = max(F(0), change[1])
        angular_defect = sum(map(abs, angular))
        # Contact work is INTERNAL: it is reported, never subtracted here.
        if positive > q(budgets.energy_j) or angular_defect > q(budgets.angular_impulse_nms):
            raise SolverFailure("original-input physical energy/angular budget exceeded")
        if max(map(abs, linear), default=F(0)) > q(budgets.impulse_ns):
            raise SolverFailure("internal self-contact created resultant impulse")
        # Both maps use one constant-normal affine path, so actual point work is
        # independently paired with the endpoint chords, not credited externally.
        point_work = F(0)
        for impulse, before, after in zip(held["world"], held["points_initial"], held["points_final"], strict=True):
            delta = sub(sub(tuple(map(q, after[0])), tuple(map(q, after[1]))), sub(tuple(map(q, before[0])), tuple(map(q, before[1]))))
            point_work += dot(tuple(map(q, impulse)), delta)/q(h)
        work_defect = abs(held_work-point_work)
        if work_defect > q(budgets.energy_j):
            raise SolverFailure("held material chord versus represented nodal work failed")
        return dict(capacity_residuals=tuple(residual_reports), held_rows=hrows, terminal_rows=trows, held_residual_mps=hr, terminal_residual_mps=tr,
                    kinetic_change_j=float(k1-k0), elastic_change_j=(float(u1[0]-u0[1]), upper(u1[1]-u0[0])),
                    original_external_work_j=float(original_work), held_internal_work_j=float(held_work),
                    held_point_work_defect_j=upper(work_defect), terminal_linear_j=float(terminal_linear),
                    terminal_quadratic_j=float(terminal_quadratic), terminal_identity_defect_j=upper(terminal_identity),
                    positive_energy_upper_j=upper(positive), angular_absolute_nms=upper(angular_defect),
                    angular_vector_nms=np.array(angular, dtype=float), held_moment_nms=np.array(held_moment, dtype=float),
                    terminal_moment_nms=np.array(terminal_moment, dtype=float), area_proofs=areas,
                    continuation=continuation, sweep_certificates=tuple(certificates),
                    exact_physics=proof_data((change, tuple(angular), terminal_identity, held_work)))


def _interval(value, bits):
    return value.bounds(bits) if hasattr(value, "bounds") else (q(value), q(value))


def _iadd(a, b):
    return a[0]+b[0], a[1]+b[1]


def _imul(a, b):
    values = tuple(x*y for x in a for y in b)
    return min(values), max(values)


def _idiv(a, b):
    if b[0] <= 0:
        raise SolverFailure("independent phase normalization is unresolved")
    return _imul(a, (1/b[1], 1/b[0]))


def _norm_interval(vector, bits):
    square = dot(vector, vector)
    lo, hi = _interval(square, bits)
    if hi < 0:
        raise SolverFailure("negative squared velocity norm")
    return sqrt_bounds(max(F(0), lo), bits)[0], sqrt_bounds(max(F(0), hi), bits)[1]


def _maxabs(interval):
    return max(abs(interval[0]), abs(interval[1]))


def _extreme_velocity(anchors, row, state, model, limits):
    from .cloth_self_mapping import independent_pair_maps
    pair = anchors.pairs[row]
    triangles = anchors.triangles[pair]
    jac, points = independent_pair_maps(state.position, triangles, anchors.weights[pair], anchors.offsets[pair], limits)
    manifold = anchors.manifolds[int(anchors.pair_index[row])]
    ideal_point = anchors.ideal_points[row]
    extreme = next((point for axis in manifold.maximal_axes for point in axis.points
                    if (point.point_a, point.point_b) == ideal_point), None)
    if extreme is None:
        raise SolverFailure("owned ideal clipping extreme is missing")
    materials = (extreme.material_a, extreme.material_b)
    ideal_j, _ = independent_pair_maps(state.position, triangles,
                                      tuple(mat[:3] for mat in materials), tuple(mat[3] for mat in materials), limits)
    def relative(maps):
        return tuple(sum((1 if side == 0 else -1)*maps[side][i][component][axis]*q(state.velocity[node, axis])
                         for side in range(2) for i, node in enumerate(triangles[side]) for axis in range(3))
                     for component in range(3))
    # Aggregate endpoints before every response square, including any shared node.
    combined = {}
    for side in range(2):
        for i, node in enumerate(triangles[side]):
            key = int(node)
            block = combined.setdefault(key, [[jac[0][0][0][0].field.scalar(0) for _ in range(3)] for _ in range(3)])
            for component in range(3):
                for axis in range(3):
                    block[component][axis] += (1 if side == 0 else -1)*jac[side][i][component][axis]
    response = tuple(tuple(sum(q(model.inverse_mass[node])*sum(block[i][axis]*block[j][axis] for axis in range(3))
                                for node, block in combined.items()) for j in range(3)) for i in range(3))
    return relative(jac), relative(ideal_j), points, response, (jac, ideal_j, triangles)


def _witness_values(witness, key):
    return tuple(F(numerator, denominator) for numerator, denominator in witness[key])


def _residual_global(witness, before, selected, model, phase, h, budgets):
    r = _witness_values(witness, "residual_exact")
    e = _witness_values(witness, "residual_uncertainty_exact")
    ell = _witness_values(witness, "ell_exact")
    resultant = [F(0)]*3
    resultant_error = [F(0)]*3
    moment = [F(0)]*3
    moment_error = [F(0)]*3
    virtual_bound = physical_bound = F(0)
    for i, inverse in enumerate(model.inverse_mass):
        row, error = r[3*i:3*i+3], e[3*i:3*i+3]
        x, v = tuple(map(q, before.position[i])), tuple(map(q, before.velocity[i]))
        m = cross(x, row)
        for axis in range(3):
            resultant[axis] += row[axis]
            resultant_error[axis] += error[axis]
            moment[axis] += m[axis]
            other = ((axis+1)%3, (axis+2)%3)
            moment_error[axis] += abs(x[other[0]])*error[other[1]]+abs(x[other[1]])*error[other[0]]
            bound = abs(row[axis])+error[axis]
            virtual_bound += bound*abs(v[axis])
            if phase == "held":
                physical_bound += bound*abs(q(selected.position[i, axis])-x[axis])/q(h)
            else:
                physical_bound += bound*(abs(v[axis])+q(inverse)*abs(ell[3*i+axis])+q(inverse)*bound/2)
    net = max((abs(a)+b for a, b in zip(resultant, resultant_error, strict=True)), default=F(0))
    angular = sum(abs(a)+b for a, b in zip(moment, moment_error, strict=True))
    if net > q(budgets.impulse_ns) or angular > q(budgets.angular_impulse_nms) or max(virtual_bound, physical_bound) > q(budgets.energy_j):
        raise SolverFailure("capacity residual resultant/couple/work exceeds original budgets")
    return dict(phase=phase, resultant_bound_ns=upper(net), couple_bound_nms=upper(angular),
                virtual_work_bound_j=upper(virtual_bound), physical_work_bound_j=upper(physical_bound),
                residual_is_removed=False, exact=proof_data((tuple(resultant), tuple(moment), physical_bound)))


def _residual_speed(maps, triangles, witness, model, bits):
    r = _witness_values(witness, "residual_exact")
    e = _witness_values(witness, "residual_uncertainty_exact")
    maximum = F(0)
    for jacobian in maps:
        for component in range(3):
            nominal = jacobian[0][0][0][0].field.scalar(0)
            bound = F(0)
            for side in range(2):
                for i, node in enumerate(triangles[side]):
                    for axis in range(3):
                        factor = (1 if side == 0 else -1)*jacobian[side][i][component][axis]*q(model.inverse_mass[node])
                        nominal += factor*r[3*node+axis]
                        bound += _maxabs(_interval(factor, bits))*e[3*node+axis]
            maximum = max(maximum, _maxabs(_interval(nominal, bits))+bound)
    return maximum


def _exact_phase(anchors, state, data, held_data, *, phase, model, law, h, budgets, limits):
    """R2 robust phase audit of a certification witness, never requested pressure.

    Each coordinate uncertainty retains its common native source. The full
    enclosure is tested once through capacity/KKT; remainder effects are separate
    diagnostics/gates, not external-work credits or additional capacity.
    """
    maximum, reports = F(0), []
    effective = _witness_values(data["witness"], "effective_exact")
    effective_error = _witness_values(data["witness"], "uncertainty_exact")
    held_effective = _witness_values(held_data["witness"], "effective_exact")
    held_error = _witness_values(held_data["witness"], "uncertainty_exact")
    for row, direction in enumerate(anchors.directions):
        u, ideal_u, points, response, map_data = _extreme_velocity(anchors, row, state, model, limits)
        jac, ideal_j, triangles = map_data
        d, dd = tuple(map(q, direction)), dot(direction, direction)
        p, pe = effective[3*row:3*row+3], effective_error[3*row:3*row+3]
        ph, he = held_effective[3*row:3*row+3], held_error[3*row:3*row+3]
        pd, pd_error = dot(p, d), dot(pe, tuple(map(abs, d)))
        phd, phd_error = dot(ph, d), dot(he, tuple(map(abs, d)))
        tangent = sub(p, tuple(pd*v/dd for v in d))
        tangent_error = tuple(sum(abs(F(int(i == j))-d[i]*d[j]/dd)*pe[j] for j in range(3)) for i in range(3))
        normal_k = sum(d[i]*response[i][j]*d[j] for i in range(3) for j in range(3))/dd
        if normal_k.sign() <= 0:
            raise SolverFailure("independent normal response is not positive")
        tangential_u = sub(u, tuple(dot(d, u)*v/dd for v in d))
        ideal_tangent_u = sub(ideal_u, tuple(dot(d, ideal_u)*v/dd for v in d))
        bits = limits.initial_sqrt_bits
        while True:
            norm_d = sqrt_bounds(dd, bits)
            pressure = _idiv((pd-pd_error, pd+pd_error), norm_d)
            held_pressure = _idiv((phd-phd_error, phd+phd_error), norm_d)
            if pressure[0] < 0 or held_pressure[0] < 0:
                raise SolverFailure("negative normal LOWER bound leaves pressure uncertainty unresolved")
            cap = pressure
            if phase == "held" and not data["initial_closed"][row]:
                cap = (F(0), F(0))
            if phase == "terminal" and not data["initial_closed"][row]:
                cap = _iadd(cap, held_pressure)
            radius = _imul(cap, (q(law.mu_dynamic), q(law.mu_dynamic)))
            gap_speed = _idiv(_interval(dot(d, sub(points[0], points[1])), bits), (q(h)*norm_d[0], q(h)*norm_d[1]))
            normal_speed = _idiv(_interval(dot(d, u), bits), norm_d)
            ideal_normal_speed = _idiv(_interval(dot(d, ideal_u), bits), norm_d)
            nl = gap_speed if phase == "held" else normal_speed
            k = _interval(normal_k, bits)
            scale = max(sum(_maxabs(_interval(response[i][j], bits)) for j in range(3)) for i in range(3))
            impulse_speed = _imul(k, pressure)
            complement = (min(impulse_speed[0], nl[0]), min(impulse_speed[1], nl[1]))
            residual = max(F(0), -nl[0], _maxabs(complement))
            tangent_intervals = tuple((v-e, v+e) for v, e in zip(tangent, tangent_error, strict=True))
            upper_square = sum(_maxabs(interval)**2 for interval in tangent_intervals)
            lengths_upper = sqrt_bounds(upper_square, bits)[1]
            slips, ideal_slips = _norm_interval(tangential_u, bits), _norm_interval(ideal_tangent_u, bits)
            residual = max(residual, max(F(0), lengths_upper-radius[0])*scale)
            kkt_rounding = F(0)
            if radius[1] == 0:
                residual = max(residual, lengths_upper*scale)
            elif slips[0] > q(budgets.velocity_mps)/64:
                for t, v in zip(tangent_intervals, tangential_u, strict=True):
                    quotient = _idiv(_interval(v, bits), slips)
                    kkt = _iadd(t, _imul(radius, quotient))
                    residual = max(residual, _maxabs(kkt)*scale)
                    # Native enclosure width is physical uncertainty already
                    # included above; only arithmetic enclosure is charged here.
                    kkt_rounding += (quotient[1]-quotient[0])*max(map(abs, radius))*scale
            else:
                residual = max(residual, slips[1], ideal_slips[1])
            ideal_error = max((_maxabs(_interval(a-b, bits)) for a, b in zip(ideal_u, u, strict=True)), default=F(0))
            reported_error = max((_maxabs(_interval(value-q(reported), bits)) for value, reported in zip(u, data["velocity"][row], strict=True)), default=F(0))
            remainder_speed = _residual_speed((jac, ideal_j), triangles, data["witness"], model, bits)
            if phase == "terminal":
                residual = max(residual, -ideal_normal_speed[0])
            rounding = (nl[1]-nl[0])+(normal_speed[1]-normal_speed[0])+(ideal_normal_speed[1]-ideal_normal_speed[0])+kkt_rounding
            rounding += slips[1]-slips[0]+ideal_slips[1]-ideal_slips[0]
            rounding += (sqrt_bounds(upper_square, bits)[1]-sqrt_bounds(upper_square, bits)[0])*scale
            rounding += sum(_interval(response[i][j], bits)[1]-_interval(response[i][j], bits)[0] for i in range(3) for j in range(3))*sum(abs(v)+e for v, e in zip(p, pe, strict=True))
            if rounding <= q(budgets.velocity_mps)/64:
                break
            if bits == limits.max_sqrt_bits:
                raise SolverFailure("independent phase interval budget exhausted")
            bits = min(2*bits, limits.max_sqrt_bits)
        if max(ideal_error, reported_error) > q(budgets.velocity_mps)/64:
            raise SolverFailure("actual owner versus ideal-extreme velocity enclosure exceeded")
        if remainder_speed > q(budgets.velocity_mps):
            raise SolverFailure("off-range residual affects actual/ideal extrema beyond phase budget")
        pressure_uncertainty = require_witness_phase_uncertainty(data["witness"], row, scale, budgets.velocity_mps)
        charged = residual+ideal_error+rounding
        maximum = max(maximum, charged)
        reports.append(dict(normal_ns=float(sum(pressure)/2), normal_lower_ns=float(pressure[0]), normal_upper_ns=upper(pressure[1]),
                            capacity_ns=float(sum(radius)/2), capacity_lower_ns=float(radius[0]), capacity_upper_ns=upper(radius[1]),
                            normal_speed_mps=float(sum(normal_speed)/2), residual_mps=upper(charged),
                            ideal_velocity_error_mps=upper(ideal_error), enclosure_mps=upper(rounding),
                            remainder_speed_bound_mps=upper(remainder_speed), pressure_uncertainty_bound_mps=upper(pressure_uncertainty), independent_response_scale=float(scale),
                            sqrt_bits=bits, pressure_kind="mass-weighted-certification-witness"))
    return upper(maximum), tuple(reports)


def require_witness_phase_uncertainty(witness, row, response_scale, velocity_budget):
    """A separate sufficient bound, not an additional disk allowance.

    The same enclosure is used in the robust disk/KKT calculation. This gate
    prevents an ill-conditioned inverse from magnifying a small native residual
    beyond the original phase speed budget even before selecting a regime.
    """
    error = _witness_values(witness, "uncertainty_exact")[3*row:3*row+3]
    bound = sum(error)*q(response_scale)
    if bound > q(velocity_budget):
        raise SolverFailure("inverse-response amplification exhausts phase/capacity uncertainty budget")
    return bound
