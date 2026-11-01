"""Independent finite-chart checks for the first physical SC2 layer event."""

from fractions import Fraction as Q

import numpy as np
from check_p65_common_motion import (
    LEAVES,
    bounds,
    derive,
    integral,
    material_box,
    outward,
    power,
    scale,
    sub,
    test,
)
from p65_common_motion import DEGREE, GRID

from phymeo.validation.bending_bounds import Interval as I
from phymeo.validation.bending_bounds import cross, dot, sqrt_interval

WIDTH = Q(1, 500)
AREA_K = Q(500, 6)


def evaluate(p, s):
    y = 0
    for c in reversed(p):
        y = s * y + c
    return y


def state_polynomials(claim, q0, v0, model, h):
    # Reconstruct the trajectory directly from signed force words and the actual inlet.
    mat = claim["material"]
    test(
        mat.shape == (5, 13) and mat.dtype == np.int64 and claim["grid"] == GRID, "material schema"
    )
    test(not np.any(claim["rows"]) and not np.any(claim["pressure"]), "nonzero prebirth pressure")
    test(tuple(claim["owners"]) == model.owners, "missing or permuted material address")
    test(
        np.array_equal(claim["mass"], model.mass) and not np.any(claim["external"]), "mass or load"
    )
    forces = [power(np.pad(mat[:, i], (0, DEGREE + 1 - len(mat)))) for i in range(13)]
    vel = [
        integral(scale(f, h / Q(float(m))), v)
        for f, m, v in zip(forces, model.mass, v0, strict=True)
    ]
    pos = [integral(scale(v, h), x) for v, x in zip(vel, q0, strict=True)]
    test(
        all(derive(x) == scale(v, h) for x, v in zip(pos, vel, strict=True)), "derivative mismatch"
    )
    return forces, pos, vel


def difference(a, b):
    return [x - y for x, y in zip(a, b, strict=True)]


def face_data(x, v, ids, rho, *, jacobian=False):
    i, j, k = map(int, ids)
    xx = [x[3 * n : 3 * n + 3] for n in (i, j, k)]
    vv = [v[3 * n : 3 * n + 3] for n in (i, j, k)]
    e, u = difference(xx[1], xx[0]), difference(xx[2], xx[0])
    de, du = difference(vv[1], vv[0]), difference(vv[2], vv[0])
    c = cross(e, u)
    test(c[2].low > Q(1, 2), "face orientation")
    test(max(a.magnitude for row in xx for a in row[:2]) + rho < 2, "finite patch")
    r = [a / c[2] for a in c[:2]]
    ss = sum(a.square() for a in r)
    nr = sqrt_interval(1 + ss, 96)
    shift = (rho * ss / (nr * (1 + nr))).rounded(96)

    def differential(dc):
        dr = [(dc[z] - r[z] * dc[2]) / c[2] for z in range(2)]
        return (rho * dot(r, dr) / ((1 + ss) * nr)).rounded(96)

    dc = [a + b for a, b in zip(cross(de, u), cross(e, du), strict=True)]
    rate_shift = differential(dc)
    jac = [I.point(0) for _ in range(13)]
    if jacobian:
        for n in (i, j, k):
            for axis in range(3):
                dx = [int(z == axis) for z in range(3)]
                eu = [(int(n == j) - int(n == i)) * b for b in dx]
                uu = [(int(n == k) - int(n == i)) * b for b in dx]
                d = [a + b for a, b in zip(cross(eu, u), cross(e, uu), strict=True)]
                jac[3 * n + axis] = differential(d)
        jac[12] = I.point(-1)
    ans = []
    for n in (i, j, k):
        gap = x[3 * n + 2] - x[12] + shift
        rate = v[3 * n + 2] - v[12] + rate_shift
        row = jac.copy()
        row[3 * n + 2] += 1
        ans.append((gap, rate, row))
    return ans


def geometry(pos, vel, model, *, jacobian=False):
    return [
        p
        for f in model.faces
        for p in face_data(pos, vel, f, Q(float(model.rho)), jacobian=jacobian)
    ]


def squared_guard(pos, s, f, node, rho):
    x = [evaluate(z, s) for z in pos]
    xx = [x[3 * int(n) : 3 * int(n) + 3] for n in f]
    c = cross(difference(xx[1], xx[0]), difference(xx[2], xx[0]))
    w = x[3 * int(node) + 2] - x[12] + rho - WIDTH
    test(w > 0 and c[2] > 0, "squared guard sign ambiguity")
    return rho * rho * c[2] * c[2] - w * w * dot(c, c)


def isolate(pos, model, depth):
    rho = Q(float(model.rho))
    out = []
    cost = 0
    for face in model.faces:
        for node in face:
            lo, hi = Q(0), Q(1)

            def sign(s):
                return squared_guard(pos, s, face, node, rho)

            test(sign(lo) < 0 < sign(hi), "unbracketed free-extension event")
            cost += 1
            for _ in range(depth):
                mid = (lo + hi) / 2
                a = sign(mid)
                cost += 2
                if a == 0:
                    lo = hi = mid
                    break
                if a < 0:
                    lo = mid
                else:
                    hi = mid
            out.append((lo, hi))
    order = sorted(range(len(out)), key=lambda i: out[i][0])
    i, j = order[:2]
    test(out[i][1] < out[j][0], "earliest tie unresolved")
    return out, i, (out[i][1] + out[j][0]) / 2, cost


def verify_free(claim, q0, v0, model, h, br, first, end):
    forces, pos, vel = state_polynomials(claim, q0, v0, model, h)
    expected = isolate(pos, model, 32)
    test((br, first, end) == expected[:3], "wrong earliest address, root or endpoint")
    jerr, berr = [Q(0)] * 13, [Q(0)] * 13
    # Integrate defects on the physical free prefix; prove monotonicity on the longer root search.
    rho = Q(float(model.rho))
    edges = [[sub(pos[3 * i + j], pos[j]) for j in range(3)] for i in (1, 2, 3)]
    max_rate = None
    for leaf in range(LEAVES):
        low, high = Q(leaf, LEAVES), Q(leaf + 1, LEAVES)
        x = [bounds(p)[leaf] for p in pos]
        v = [bounds(p)[leaf] for p in vel]
        for g, rate, _ in geometry(x, v, model):
            # w = middle_z - platform + rho-width is positive throughout; normal tilt < rho here.
            test(g.low > WIDTH - rho / 2 and rate.high < -Q(1, 10), "monotone unsquared guard")
            max_rate = rate.high if max_rate is None else max(rate.high, max_rate)
        test(
            min(x[3 * int(n) + 2].low - x[12].high + rho - WIDTH for n in model.faces.ravel()) > 0,
            "unsquared factor on whole leaf",
        )
        span = h * max(Q(0), min(br[first][1], high) - low)
        if span:
            field, _ = material_box(*[[bounds(p)[leaf] for p in row] for row in edges], model.rest)
            for i in range(13):
                law = field[i // 3][i % 3] if i < 12 else I.point(0)
                err = (bounds(forces[i])[leaf] - law).magnitude
                jerr[i] += span * err
                berr[i] += span * h * (end - low) * err
    return (
        forces,
        pos,
        vel,
        jerr,
        berr,
        dict(
            roots_s_over_h=[[str(left), str(right)] for left, right in br],
            first=model.owners[first],
            event_seconds=[str(t * h) for t in br[first]],
            latest_closing_rate_upper=outward(I.point(max_rate))[1],
            positive_unsquared_factors=True,
            free_zero_until_root=True,
            instantaneous_impulse_ns=0,
            free_min_hard_gap_m=str(WIDTH),
            time_cells=expected[3] + LEAVES,
        ),
    )


def hull(a, b):
    return I(min(a.low, b.low), max(a.high, b.high))


def physical_bridge(claim, q0, v0, model, h, br, first, end, ramp):
    forces, pos, vel, jerr, berr, cert = verify_free(claim, q0, v0, model, h, br, first, end)
    lo, hi = br[first]
    dm = h * (end - (lo + hi) / 2)
    duration = I(h * (end - hi), h * (end - lo))
    amplitude, j, b = (Q(ramp[key]) for key in ("slope", "J", "B"))
    row = list(map(Q, ramp["map"]))
    test(
        len(row) == 13
        and row[-1] == -1
        and duration.low > 0
        and amplitude > 0
        and j == amplitude * dm**2 / 2
        and b == amplitude * dm**3 / 6,
        "invalid incident force moments",
    )
    # Solve the two moment equations for a ramp on xi=(time-root)/duration.
    # P/(time-root) is linear in xi. Enclose both endpoint coefficients jointly.
    ratio = I.point(dm) / duration
    scale0 = amplitude * ratio.square()
    slope = hull(scale0 * (4 * ratio - 3), scale0 * (3 - 2 * ratio))
    test(slope.low > 0, "ramp positivity")
    masses = [Q(float(m)) for m in model.mass]
    xx = [evaluate(p, I(lo, end)) + row[i] / masses[i] * I(0, b) for i, p in enumerate(pos)]
    vv = [evaluate(p, I(lo, end)) + row[i] / masses[i] * I(0, j) for i, p in enumerate(vel)]
    qp = [evaluate(p, end) + row[i] * b / masses[i] for i, p in enumerate(pos)]
    vp = [evaluate(p, end) + row[i] * j / masses[i] for i, p in enumerate(vel)]
    xyz = [I.point(x) for x in qp]
    ports = geometry(xx, vv, model, jacobian=True)
    endports = geometry(xyz, list(map(I.point, vp)), model)
    test(max(z.high for _, z, _ in ports) < -Q(1, 10), "bridge is not strictly closing")
    test(
        all(g.low > WIDTH for i, (g, _, _) in enumerate(endports) if i != first),
        "another owner entered",
    )
    mingap = min(g.low for g, _, _ in ports)
    test(mingap > 0 and endports[first][0].high < WIDTH, "unproved hard clearance or compression")
    # The exact root has g=width. Integrating the bounded actual rate from that root
    # bounds (width-g)/(time-root), also at its continuous positive-time limit.
    rate = ports[first][1]
    law_slope = AREA_K * (-rate) * (1 - 1000 * rate)
    schange = (slope - law_slope).magnitude
    rel = schange / law_slope.low
    test(rel <= Q(1, 100), "positive incident law failed")
    force_mismatch = schange * duration.high
    je = force_mismatch * duration.high / 2
    be = force_mismatch * duration.high**2 / 6
    test(je <= Q(1, 10**10) and be / min(masses) <= Q(1, 10**10), "owner moments exceed limit")
    pos_edges = [difference(xx[3 * i : 3 * i + 3], xx[:3]) for i in (1, 2, 3)]
    field, _ = material_box(*pos_edges, model.rest)
    for i, m in enumerate(masses):
        law = field[i // 3][i % 3] if i < 12 else I.point(0)
        materr = (evaluate(forces[i], I(lo, end)) - law).magnitude
        true_map = ports[first][2][i]
        jerr[i] += (
            duration.high * materr + (row[i] - true_map).magnitude * j + true_map.magnitude * je
        )
        berr[i] += (
            duration.high**2 * materr + (row[i] - true_map).magnitude * b + true_map.magnitude * be
        )
    verr = max(3 * x / m for x, m in zip(jerr, masses, strict=True))
    qerr = max(3 * x / m for x, m in zip(berr, masses, strict=True))
    test(verr <= Q(2, 10**6) and qerr <= Q(1, 10**8), "material and mapped force defects exceeded")

    def elastic(state):
        y = list(map(I.point, state))
        return material_box(
            *[difference(y[3 * i : 3 * i + 3], y[:3]) for i in (1, 2, 3)], model.rest
        )[1]

    kinetic = sum(m * (new**2 - old**2) / 2 for m, new, old in zip(masses, vp, v0, strict=True))
    compression = WIDTH - endports[first][0]
    elastic_layer = AREA_K * compression.square() / 2
    loss = AREA_K * 1000 * (-rate).high**3 * duration.high**2 / 2
    energy = (kinetic + elastic(qp) - elastic(q0) + elastic_layer).high + loss
    angular = [Q(0)] * 3
    for i in range(4):
        k = 3 * i
        out = cross(qp[k : k + 3], vp[k : k + 3])
        inp = cross(q0[k : k + 3], v0[k : k + 3])
        for d in range(3):
            angular[d] += masses[k] * (out[d] - inp[d])
    # Actual reaction arm lies strictly within the finite square. Both planar components contribute.
    ang = sum(map(abs, angular)) + 4 * j
    test(
        max(0, energy) <= Q(1, 10**8) and ang <= Q(1, 10**8),
        "joint original energy or paired angular",
    )
    cert.update(
        bridge_end_s=str(end * h),
        only_loaded_owner=model.owners[first],
        incident_relative_error_upper=outward(I.point(rel))[1],
        end_force_n=outward(duration * scale0 * (3 - 2 * ratio)),
        end_hard_gap_m=outward(endports[first][0]),
        hard_gap_lower_m=outward(I.point(mingap))[0],
        J_ns=str(j),
        B_ns2=str(b),
        impulse_error_upper_ns=outward(I.point(je))[1],
        time_moment_error_over_min_mass_m=outward(I.point(be / min(masses)))[1],
        node_velocity_defect_m_s=outward(I.point(verr))[1],
        node_position_defect_m=outward(I.point(qerr))[1],
        positive_energy_upper_j=outward(I.point(max(energy, 0)))[1],
        paired_angular_upper_nms=outward(I.point(ang))[1],
        original_velocity_retained=True,
        event_point_impulse=0,
        owner_charges=claim["owner_charges"] + 6,
    )
    return qp, vp, cert
