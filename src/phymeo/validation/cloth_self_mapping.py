"""Independent exact normal-offset Jacobians and aggregate mapping receipts."""

from fractions import Fraction as F

import numpy as np

from ..constraints.cloth_self_response import owned, SelfContactLoadPlan
from ..core.capabilities import SolverFailure
from ..core.scene import fingerprint
from ..geometry._prism_pair_exact import PairField, PairNumber, PrismPairProofLimits
from ..geometry.cloth_self_anchors import q, dot, cross, sub, proof_data


def _bound(value, budget):
    bits = value.field.limits.initial_sqrt_bits
    while True:
        lo, hi = value.bounds(bits)
        if hi-lo <= budget/64:
            return lo, hi
        if bits == value.field.limits.max_sqrt_bits:
            raise SolverFailure("independent mapping enclosure exhausted")
        bits = min(2*bits, value.field.limits.max_sqrt_bits)


def transport(value, field):
    if not isinstance(value, PairNumber):
        return field.scalar(q(value))
    result = field.scalar(0)
    try:
        for mask, coefficient in enumerate(value.coefficients):
            if coefficient == 0:
                continue
            term = field.scalar(coefficient)
            for index, root in enumerate(value.field.roots):
                if mask & (1 << index):
                    term *= field.sqrt(root)
            result += term
    except ValueError as error:
        raise SolverFailure("material radical field transport is unresolved") from error
    return result


def independent_pair_maps(positions, triangles, weights, offsets, limits=PrismPairProofLimits()):
    """Direct differential of c=(x1-x0) cross (x2-x0), independent of port J."""
    x = [tuple(tuple(map(q, row)) for row in positions[ids]) for ids in triangles]
    areas = [cross(sub(t[1], t[0]), sub(t[2], t[0])) for t in x]
    squares = [dot(c, c) for c in areas]
    if not all(squares):
        raise SolverFailure("degenerate mapping triangle")
    field = PairField(*squares, limits)
    maps, points = [], []
    for t, c, square, b, s in zip(x, areas, squares, weights, offsets, strict=True):
        b = tuple(transport(value, field) for value in b)
        s = transport(s, field)
        norm = field.sqrt(square)
        e, f = sub(t[1], t[0]), sub(t[2], t[0])
        right = (sub(e, f), f, tuple(-v for v in e))
        jacobian = []
        for i in range(3):
            columns = [cross(tuple(F(int(j == k)) for j in range(3)), right[i]) for k in range(3)]
            jacobian.append(tuple(tuple(b[i]*int(j == k) + s*(columns[k][j]-c[j]*dot(c, columns[k])/square)/norm for k in range(3)) for j in range(3)))
        maps.append(tuple(jacobian))
        points.append(tuple(sum(b[i]*t[i][j] for i in range(3)) + s*c[j]/norm for j in range(3)))
    return tuple(maps), tuple(points)


class SelfContactMappingLedger:
    def __init__(self, model, *, proof_limits=PrismPairProofLimits()):
        inverse = owned(model.inverse_mass, name="inverse mass")
        if inverse.ndim != 1 or np.any(inverse <= 0):
            raise ValueError("mapping ledger requires positive dynamic masses")
        self.inverse_mass = inverse
        self.proof_limits = proof_limits

    def require(self, plan, *, original_positions, original_velocity,
                actual_replay_or_terminal_receipt, budgets):
        if not isinstance(plan, SelfContactLoadPlan):
            raise TypeError("owned self-contact load plan required")
        n = len(self.inverse_mass)
        x = owned(original_positions, (n, 3), name="mapping positions")
        velocity = owned(original_velocity, (n, 3), name="mapping velocity")
        data = {name: getattr(plan, name) for name in ("phase", "domain", "triangles", "weights", "offsets", "pairs", "jacobians", "world_points", "world_impulses", "node_impulses", "raw_response")}
        if fingerprint(data) != plan.binding_token:
            raise SolverFailure("load plan binding changed")
        if np.any(plan.triangles >= n) or plan.node_impulses.shape != (n, 3):
            raise SolverFailure("load plan node binding mismatch")
        exact = [[F(0)]*3 for _ in range(n)]
        ideal_bounds = [[[F(0), F(0)] for _ in range(3)] for _ in range(n)]
        couple = [F(0)]*3
        virtual_point = F(0)
        j_error = F(0)
        endpoint_error = F(0)
        for pair, impulse in zip(plan.pairs, plan.world_impulses, strict=True):
            maps, points = independent_pair_maps(x, plan.triangles[pair], plan.weights[pair], plan.offsets[pair], self.proof_limits)
            endpoints = []
            for side, attachment in enumerate(pair):
                sign = 1 if side == 0 else -1
                endpoints.append(tuple(map(q, plan.world_points[attachment])))
                for component in range(3):
                    lo, hi = _bound(points[side][component]-q(plan.world_points[attachment, component]), q(budgets.anchor_error_m))
                    endpoint_error = max(endpoint_error, abs(lo), abs(hi))
                for i, node in enumerate(plan.triangles[attachment]):
                    for axis in range(3):
                        stored = sum(q(plan.jacobians[attachment, i, component, axis])*q(impulse[component])*sign for component in range(3))
                        exact[node][axis] += stored
                        ideal = sum(maps[side][i][component][axis]*q(impulse[component])*sign for component in range(3))
                        lo, hi = _bound(ideal, q(budgets.impulse_ns)/max(1, 6*len(plan.pairs)))
                        ideal_bounds[node][axis][0] += lo
                        ideal_bounds[node][axis][1] += hi
                        virtual_point += stored*q(velocity[node, axis])
                        for component in range(3):
                            delta = maps[side][i][component][axis]-q(plan.jacobians[attachment, i, component, axis])
                            a, b = _bound(delta, F(1, 10**12))
                            j_error = max(j_error, abs(a), abs(b))
            moment = cross(sub(endpoints[0], endpoints[1]), tuple(map(q, impulse)))
            couple = [a+b for a, b in zip(couple, moment, strict=True)]
        error = F(0)
        representation = F(0)
        receipt = actual_replay_or_terminal_receipt
        if receipt["phase"] != plan.phase or receipt["plan_token"] != plan.binding_token:
            raise SolverFailure("actual mapping receipt phase/path mismatch")
        represented = owned(receipt["represented_nodes"], (n, 3), name="actual represented nodes")
        net, mapped_moment = [F(0)]*3, [F(0)]*3
        virtual_nodes = F(0)
        for i in range(n):
            for axis in range(3):
                requested = q(plan.node_impulses[i, axis])
                error = max(error, abs(requested-exact[i][axis]), *(abs(requested-v) for v in ideal_bounds[i][axis]))
                representation = max(representation, abs(q(represented[i, axis])-requested))
                net[axis] += requested
                virtual_nodes += requested*q(velocity[i, axis])
            moment = cross(tuple(map(q, x[i])), tuple(map(q, plan.node_impulses[i])))
            mapped_moment = [a+b for a, b in zip(mapped_moment, moment, strict=True)]
        moment_error = sum(abs(a-b) for a, b in zip(mapped_moment, couple, strict=True))
        work_error = abs(virtual_nodes-virtual_point)
        if endpoint_error > q(budgets.anchor_error_m) or j_error > F(1, 10**10):
            raise SolverFailure("independent ideal versus stored attachment map mismatch")
        if max(error, representation, *map(abs, net)) > q(budgets.impulse_ns):
            raise SolverFailure("aggregate requested/ideal/represented impulse mismatch")
        if moment_error > q(budgets.angular_impulse_nms) or work_error > q(budgets.energy_j):
            raise SolverFailure("mapping moment or virtual-work identity failed")
        actual_response_error = F(0)
        if plan.phase == "terminal":
            final = owned(receipt["final_velocity"], (n, 3), name="terminal actual velocity")
            for i in range(n):
                for axis in range(3):
                    defect = q(final[i, axis])-q(velocity[i, axis])-q(self.inverse_mass[i])*q(represented[i, axis])
                    actual_response_error = max(actual_response_error, abs(defect))
            if actual_response_error > q(budgets.velocity_mps):
                raise SolverFailure("actual native terminal response disagrees with aggregate request")
        witness = capacity_witness(plan, receipt, self.inverse_mass, velocity, budgets, original_positions=x, proof_limits=self.proof_limits)
        return dict(phase=plan.phase, plan_token=plan.binding_token, capacity_witness=witness, ideal_map_error_ns=float(error),
                    stored_j_error=float(j_error), endpoint_error_m=float(endpoint_error),
                    represented_error_ns=float(representation), virtual_work_error_j=float(work_error),
                    resultant_ns=np.array(net, dtype=float), actual_couple_nms=np.array(couple, dtype=float),
                    moment_error_nms=float(moment_error), actual_response_error_mps=float(actual_response_error),
                    exact=proof_data((tuple(net), tuple(couple), moment_error, work_error)))


def stored_difference_map(plan, node_count):
    rows = [[F(0)]*(3*node_count) for _ in range(3*len(plan.pairs))]
    for pair_index, pair in enumerate(plan.pairs):
        for side, attachment in enumerate(pair):
            for i, node in enumerate(plan.triangles[attachment]):
                for component in range(3):
                    for axis in range(3):
                        rows[3*pair_index+component][3*int(node)+axis] += (1 if side == 0 else -1)*q(plan.jacobians[attachment, i, component, axis])
    return rows


def capacity_decomposition(b, inverse_mass, ell, uncertainty):
    """Exact mass-weighted certification decomposition; never alters a state.

    b is the full stored difference map [dimensionless], ell [N*s] and its
    nonnegative coordinate enclosure. All shared-node terms survive the Gram.
    """
    from ..constraints.cloth_self_response import rational_response_inverse
    weights = [q(w) for w in inverse_mass for _ in range(3)]
    if any(w <= 0 for w in weights):
        raise ValueError("capacity decomposition requires positive inverse masses")
    ell, uncertainty = list(map(q, ell)), list(map(q, uncertainty))
    if len(ell) != len(weights) or len(uncertainty) != len(weights) or any(v < 0 for v in uncertainty):
        raise ValueError("capacity nodal impulse/uncertainty shape mismatch")
    b = [[q(value) for value in row] for row in b]
    if any(len(row) != len(weights) for row in b):
        raise ValueError("capacity difference-map shape mismatch")
    a = [[sum(x*y*w for x, y, w in zip(left, right, weights, strict=True)) for right in b] for left in b]
    inverse, rank = rational_response_inverse(a)
    transform = [[sum(inverse[i][j]*b[j][node] for j in range(len(b)))*weights[node]
                  for node in range(len(weights))] for i in range(len(b))]
    effective = [sum(v*p for v, p in zip(row, ell, strict=True)) for row in transform]
    effective_error = [sum(abs(v)*e for v, e in zip(row, uncertainty, strict=True)) for row in transform]
    remainder = [ell[node]-sum(b[row][node]*effective[row] for row in range(len(b))) for node in range(len(weights))]
    residual_error = [sum(abs(F(int(node == other))-sum(b[row][node]*transform[row][other] for row in range(len(b))))*uncertainty[other]
                          for other in range(len(weights))) for node in range(len(weights))]
    return dict(effective=effective, effective_error=effective_error, residual=remainder,
                residual_error=residual_error, response=a, transform=transform, rank=rank)


def actual_phase_impulse(plan, receipt, inverse_mass, original_velocity):
    """R2: reconstruct exact held subtraction or actual terminal momentum change."""
    if receipt["phase"] != plan.phase or receipt["plan_token"] != plan.binding_token:
        raise SolverFailure("phase capacity receipt binding changed")
    n = len(inverse_mass)
    backend = receipt["backend"]
    if backend not in ("numpy", "tyx:interpret", "tyx:cuda", "tyx:cpu_cpp"):
        raise SolverFailure("unknown phase native representation")
    if backend != "numpy":
        execution = receipt["native_execution"]
        if execution.get("mode") != backend[4:] or execution.get("compute_dtype") != "float32" or execution.get("completed_launches", 0) <= 0 or execution.get("completed_launches") != execution.get("launches"):
            raise SolverFailure("missing actual native phase conversion evidence")
        values = receipt["predicted_velocity"] if plan.phase == "held" else receipt["final_velocity"]
        if not np.array_equal(values, values.astype(np.float32).astype(np.float64)):
            raise SolverFailure("native phase velocity is not actual FP32 promotion")
    if plan.phase == "held":
        if receipt.get("material_scope") != "distance-material-equations-v1":
            raise SolverFailure("capacity lacks mandatory native material-path certification")
        original = owned(receipt["original_force"], (n, 3), name="original force")
        effective = owned(receipt["effective_force"], (n, 3), name="actual effective force")
        predicted = owned(receipt["predicted_velocity"], (n, 3), name="actual predicted velocity")
        gravity = owned(receipt["gravity"], (3,), name="original gravity")
        h = q(receipt["h"])
        if h <= 0:
            raise SolverFailure("capacity duration must be positive")
        ell, error = [], []
        for i, inverse in enumerate(inverse_mass):
            w = q(inverse)
            for axis in range(3):
                ell.append(h*(q(effective[i, axis])-q(original[i, axis])))
                e_v = q(predicted[i, axis])-q(original_velocity[i, axis])-h*(q(gravity[axis])+w*q(effective[i, axis]))
                error.append(abs(e_v)/w)
    else:
        if backend != "numpy":
            delta = receipt["native_delta"]
            expected = 2 if np.any(plan.node_impulses != 0) else 0
            if delta.get("completed_launches") != expected or delta.get("launches") != expected:
                raise SolverFailure("missing operation-specific actual native terminal receipt")
        if receipt.get("applications") != 1:
            raise SolverFailure("capacity terminal application count mismatch")
        final = owned(receipt["final_velocity"], (n, 3), name="actual terminal velocity")
        ell = [(q(final[i, axis])-q(original_velocity[i, axis]))/q(inverse_mass[i]) for i in range(n) for axis in range(3)]
        error = [F(0)]*len(ell)
    return ell, error


def capacity_witness(plan, receipt, inverse_mass, original_velocity, budgets, *, original_positions, proof_limits=PrismPairProofLimits()):
    b = stored_difference_map(plan, len(inverse_mass))
    ell, error = actual_phase_impulse(plan, receipt, inverse_mass, original_velocity)
    observation = receipt["represented_nodes"].ravel()
    request = plan.node_impulses.ravel()
    delivery_error = max((abs(value-q(expected)) for value, expected in zip(ell, request, strict=True)), default=F(0))
    observation_error = max((abs(value-q(observed)) for value, observed in zip(ell, observation, strict=True)), default=F(0))
    if max(delivery_error, observation_error) > q(budgets.impulse_ns):
        raise SolverFailure("actual momentum/input subtraction exceeds original impulse-map ceiling")
    proof = capacity_decomposition(b, inverse_mass, ell, error)
    total_error, ideal_error, contraction = _ideal_capacity_error(plan, original_positions, proof, inverse_mass, error, budgets, proof_limits)
    proof = capacity_decomposition(b, inverse_mass, ell, total_error)
    delta = [a-q(p) for a, p in zip(proof["effective"], plan.world_impulses.ravel(), strict=True)]
    raw = plan.raw_response
    if raw.shape != (len(b), len(b)):
        raise SolverFailure("raw capacity response shape mismatch")
    scale = max((abs(v) for row in proof["response"] for v in row), default=F(0))
    raw_error = max((abs(q(raw[i, j])-proof["response"][i][j]) for i in range(len(b)) for j in range(len(b))), default=F(0))
    if raw_error > F(1, 10**12)*scale:
        raise SolverFailure("raw response versus exact stored-map Gram mismatch")
    residual_bound = [abs(v)+e for v, e in zip(proof["residual"], proof["residual_error"], strict=True)]
    if max(residual_bound, default=F(0)) > q(budgets.impulse_ns):
        raise SolverFailure("unrepresented nodal remainder exceeds mapping ceiling")
    token = lambda value: (value.numerator, value.denominator)
    return dict(schema="mass-weighted-capacity-witness-v1", phase=plan.phase, plan_token=plan.binding_token,
                b_token=fingerprint(proof_data(tuple(tuple(row) for row in b))),
                effective_exact=tuple(map(token, proof["effective"])), uncertainty_exact=tuple(map(token, proof["effective_error"])),
                delta_exact=tuple(map(token, delta)), residual_exact=tuple(map(token, proof["residual"])),
                residual_uncertainty_exact=tuple(map(token, proof["residual_error"])), ell_exact=tuple(map(token, ell)),
                native_uncertainty_exact=tuple(map(token, error)), ideal_map_uncertainty_exact=tuple(map(token, ideal_error)),
                ideal_map_contraction=token(contraction), rank=proof["rank"],
                raw_gram_error=float(raw_error), delivery_error_ns=float(delivery_error),
                observation_error_ns=float(observation_error), decomposition_is_physical_ownership=False)


def verify_capacity_witness(witness, plan, receipt, inverse_mass, original_velocity, budgets, *, original_positions, proof_limits=PrismPairProofLimits()):
    expected = capacity_witness(plan, receipt, inverse_mass, original_velocity, budgets, original_positions=original_positions, proof_limits=proof_limits)
    if fingerprint(witness) != fingerprint(expected):
        raise SolverFailure("phase capacity witness is not bound to actual delivery")
    return expected


def _ideal_capacity_error(plan, positions, proof, inverse_mass, native_error, budgets, proof_limits):
    """Outward map error and a finite contraction bound for ideal capacity.

    Pair-local radicals are enclosed before accumulation. The actual stored-map
    decomposition remains unchanged; this bounds its interpretation uncertainty.
    """
    width, height = 3*len(inverse_mass), 3*len(plan.pairs)
    gamma = [[F(0)]*height for _ in range(width)]
    round_charge = F(0)
    for pair_index, pair in enumerate(plan.pairs):
        ideal, _ = independent_pair_maps(positions, plan.triangles[pair], plan.weights[pair], plan.offsets[pair], proof_limits)
        differences = {}
        for side, attachment in enumerate(pair):
            for i, node in enumerate(plan.triangles[attachment]):
                for component in range(3):
                    for axis in range(3):
                        key = (3*int(node)+axis, 3*pair_index+component)
                        delta = (1 if side == 0 else -1)*(ideal[side][i][component][axis]-q(plan.jacobians[attachment, i, component, axis]))
                        differences[key] = differences.get(key, delta.field.scalar(0))+delta
        for (node, component), delta in differences.items():
            lo, hi = delta.bounds(delta.field.limits.max_sqrt_bits)
            gamma[node][component] = max(abs(lo), abs(hi))
            round_charge += (hi-lo)*abs(proof["effective"][component])
    if round_charge > q(budgets.impulse_ns)/64:
        raise SolverFailure("ideal capacity map enclosure exceeds reserved impulse budget")
    transform = proof["transform"]
    feedback = [[sum(abs(transform[i][node])*gamma[node][j] for node in range(width)) for j in range(height)] for i in range(height)]
    contraction = max((sum(row) for row in feedback), default=F(0))
    if contraction >= 1:
        raise SolverFailure("ideal/stored capacity map uncertainty has no contraction witness")
    base = [proof["effective_error"][i]+sum(feedback[i][j]*abs(proof["effective"][j]) for j in range(height)) for i in range(height)]
    uniform = max(base, default=F(0))/(1-contraction)
    ideal_error = [sum(gamma[node][j]*(abs(proof["effective"][j])+uniform) for j in range(height)) for node in range(width)]
    total = [a+b for a, b in zip(native_error, ideal_error, strict=True)]
    return total, ideal_error, contraction
