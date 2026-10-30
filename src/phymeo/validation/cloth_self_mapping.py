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
        data = {name: getattr(plan, name) for name in ("phase", "domain", "triangles", "weights", "offsets", "pairs", "jacobians", "world_points", "world_impulses", "node_impulses")}
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
        return dict(phase=plan.phase, plan_token=plan.binding_token, ideal_map_error_ns=float(error),
                    stored_j_error=float(j_error), endpoint_error_m=float(endpoint_error),
                    represented_error_ns=float(representation), virtual_work_error_j=float(work_error),
                    resultant_ns=np.array(net, dtype=float), actual_couple_nms=np.array(couple, dtype=float),
                    moment_error_nms=float(moment_error), actual_response_error_mps=float(actual_response_error),
                    exact=proof_data((tuple(net), tuple(couple), moment_error, work_error)))
