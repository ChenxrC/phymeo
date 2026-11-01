"""Whole-cell checks for a single strictly closing SC-L1 birth on a jointly forced path."""

from fractions import Fraction as Q
from hashlib import sha256

import numpy as np
from audit_p65_sc2_event_moments import direct_gap
from check_p65_common_motion import (
    ZERO,
    add,
    derive,
    integral,
    material_box,
    outward,
    power,
    scale,
    sub,
    times,
)
from check_p65_sc2_common_entry import AREA_K, WIDTH, difference, evaluate, geometry, hull
from p65_common_motion import DEGREE, GRID, checkpoint, need, unpack
from trace_p65_sc2_common_window import poly_range, positive

from phymeo.validation.bending_bounds import Interval as I
from phymeo.validation.bending_bounds import cross, dot, sqrt_interval

LEAVES = 64


def word_polys(a):
    return power(np.pad(a, (0, DEGREE + 1 - len(a))))


def moment(p, lo, hi):
    anti = integral(p)
    return evaluate(anti, hi) - evaluate(anti, lo)


class Chart:
    def __init__(self, claim, model, q, v, h):
        need(
            Q(claim["h"]) == h and claim["grid"] == GRID and tuple(claim["owners"]) == model.owners,
            "duration, grid or owner mismatch",
        )
        fields = [claim[k] for k in ("material", "pressure", "rows")]
        need(
            [f.shape for f in fields] == [(5, 13), (5, 6), (5, 6, 13)]
            and all(f.dtype == np.int64 for f in fields)
            and np.array_equal(claim["mass"], model.mass)
            and np.array_equal(claim["external"], model.external)
            and not np.any(model.external),
            "field shape, original mass or load",
        )
        need(
            claim["force_calls"] == 13
            and claim["owner_charges"] == 78
            and len(claim["iterations"]) == 3,
            "collocation ledger",
        )
        need(
            np.array_equal(claim["iterations"][0]["q"][0], np.asarray(q, float))
            and np.array_equal(claim["iterations"][0]["v"][0], np.asarray(v, float)),
            "chart inlet",
        )
        mat, ps, rows = fields
        self.mat = [word_polys(mat[:, i]) for i in range(13)]
        self.ps = [word_polys(ps[:, a]) for a in range(6)]
        self.rows = [[word_polys(rows[:, a, i]) for i in range(13)] for a in range(6)]
        self.v = []
        self.x = []
        for i, m in enumerate(model.mass):
            f = self.mat[i]
            for row in self.rows:
                f = add(f, row[i])
            vp = integral(scale(f, h / Q(float(m))), v[i])
            self.v.append(vp)
            self.x.append(integral(scale(vp, h), q[i]))
        need(
            all(derive(x) == scale(vp, h) for x, vp in zip(self.x, self.v, strict=True)),
            "velocity is not the derivative of the represented path",
        )
        self.h = h

    def at(self, s):
        return ([evaluate(z, s) for z in self.x], [evaluate(z, s) for z in self.v])


def packet_hash(p):
    return sha256(p.tobytes()).hexdigest()


def options(x, v, model):
    return geometry(list(map(I.point, x)), list(map(I.point, v)), model)


def shift_and_arm(x, v, face, rho):
    ids = list(map(int, face))
    xx = [x[3 * i : 3 * i + 3] for i in ids]
    vv = [v[3 * i : 3 * i + 3] for i in ids]
    e, u = difference(xx[1], xx[0]), difference(xx[2], xx[0])
    ep, up = difference(vv[1], vv[0]), difference(vv[2], vv[0])
    c = cross(e, u)
    dc = [a + b for a, b in zip(cross(ep, u), cross(e, up), strict=True)]
    need(c[2].low > Q(1, 2), "oriented face")
    r = [t / c[2] for t in c[:2]]
    rp = [(dc[j] - r[j] * dc[2]) / c[2] for j in range(2)]
    ss = sum(z.square() for z in r)
    rt = sqrt_interval(1 + ss, 96)
    return (
        (rho * ss / (rt * (1 + rt))).rounded(96),
        (rho * dot(r, rp) / (rt * (1 + ss))).rounded(96),
        [(rho * z / rt).rounded(96) for z in r],
    )


def onset(claim, mask, chart, model):
    idx = int(claim["owner"])
    lo, hi, b = map(Q, (claim["lo"], claim["hi"], claim["join"]))
    need(
        0 <= idx < 6
        and not mask[idx]
        and 0 < lo < hi < b < 1
        and hi - lo == Q(1, 2**32)
        and b == Q((hi * 4096).__ceil__() + 1, 4096),
        "event identity, bracket or join",
    )
    need(
        direct_gap(chart.x, lo, idx, model).low > WIDTH
        and direct_gap(chart.x, hi, idx, model).high < WIDTH,
        "unsquared event sign",
    )
    h = chart.h
    d = h * (b - (lo + hi) / 2)
    duration = I(h * (b - hi), h * (b - lo))
    j, back, slope = map(Q, (claim["J"], claim["B"], claim["slope"]))
    row = list(map(Q, claim["map"]))
    need(
        len(row) == 13
        and row[-1] == -1
        and slope > 0
        and j == Q(round(slope * d * d / 2 * (1 << 112)), 1 << 112)
        and j > 0
        and back == j * d / 3,
        "invalid bridge moments",
    )
    k = 2 * j / d**2
    ratio = I.point(d) / duration
    k0 = k * ratio.square()
    density_slope = hull(k0 * (4 * ratio - 3), k0 * (3 - 2 * ratio))
    need(density_slope.low > 0, "bridge positivity")
    masses = list(map(lambda z: Q(float(z)), model.mass))
    dx = [r / m * I(0, back) for r, m in zip(row, masses, strict=True)]
    dv = [r / m * I(0, j) for r, m in zip(row, masses, strict=True)]
    xx = [poly_range(p, lo, b) + z for p, z in zip(chart.x, dx, strict=True)]
    vv = [poly_range(p, lo, b) + z for p, z in zip(chart.v, dv, strict=True)]
    ports = geometry(xx, vv, model, jacobian=True)
    rate = ports[idx][1]
    need(
        max(z.high for _, z, _ in ports) < 0 and min(g.low for g, _, _ in ports) > 0,
        "bridge geometry",
    )
    law_slope = AREA_K * (-rate) * (1 - 1000 * rate)
    err = (density_slope - law_slope).magnitude
    relative = err / law_slope.low
    need(relative <= Q(1, 100), "bridge relative law")
    x, v = chart.at(b)
    x = [z + r * back / m for z, r, m in zip(x, row, masses, strict=True)]
    v = [z + r * j / m for z, r, m in zip(v, row, masses, strict=True)]
    end = options(x, v, model)
    need(end[idx][0].high < WIDTH, "birth did not enter")
    for i, taken in enumerate(mask):
        if i != idx and not taken:
            need(end[i][0].low > WIDTH, "unisolated earlier or simultaneous event")
    return dict(
        idx=idx,
        lo=lo,
        hi=hi,
        b=b,
        x=x,
        v=v,
        dx=dx,
        dv=dv,
        j=j,
        back=back,
        relative=relative,
        law_j=err * duration.high**2 / 2,
        law_b=err * duration.high**3 / 6,
        ports=ports,
        row=row,
        loss=AREA_K * 1000 * (-rate).high**3 * duration.high**2 / 2,
        start_force_n=0,
    )


def blank():
    return dict(
        je=[Q(0)] * 6,
        be=[Q(0)] * 6,
        relative=[Q(0)] * 6,
        j=[Q(0)] * 6,
        b=[Q(0)] * 6,
        node_j=[Q(0)] * 13,
        node_b=[Q(0)] * 13,
        torque=[I.point(0) for _ in range(3)],
        loss=I.point(0),
        mingap=None,
        rate_high=None,
        leaves=0,
    )


def audit_chart(curve, model, mask, end, tail, acc, bridge=None, *, closing=True, leaves=LEAVES):
    h = curve.h
    for i, taken in enumerate(mask):
        if not taken:
            need(
                curve.ps[i] == ZERO and all(p == ZERO for p in curve.rows[i]),
                "force outside active branch",
            )
    edges = [[sub(curve.x[3 * i + j], curve.x[j]) for j in range(3)] for i in (1, 2, 3)]
    need(1 <= leaves <= 4096, "proof leaf limit")
    closing_prefix = [True] * 6
    for leaf in range(leaves):
        lo = end * leaf / leaves
        hi = end * (leaf + 1) / leaves
        span = h * (hi - lo)
        remaining = tail + h * (end - lo)
        acc["leaves"] += 1
        moving = bridge is not None and hi >= bridge["lo"]
        dx = bridge["dx"] if moving else [I.point(0)] * 13
        dv = bridge["dv"] if moving else [I.point(0)] * 13
        x = [poly_range(z, lo, hi) + d for z, d in zip(curve.x, dx, strict=True)]
        v = [poly_range(z, lo, hi) + d for z, d in zip(curve.v, dv, strict=True)]
        ports = geometry(x, v, model, jacobian=True)
        highest = max(z.high for _, z, _ in ports)
        need(highest < (0 if closing else Q(1, 1000)), "unresolved exit or damping event")
        closing_prefix = [
            old and speed.high < 0 for old, (_, speed, _) in zip(closing_prefix, ports, strict=True)
        ]
        acc["rate_high"] = highest if acc["rate_high"] is None else max(highest, acc["rate_high"])
        gap = min(g.low for g, _, _ in ports)
        need(gap > 0, "hard penetration")
        acc["mingap"] = gap if acc["mingap"] is None else min(gap, acc["mingap"])
        material, _ = material_box(
            *[
                [poly_range(z, lo, hi) + dx[3 * n + j] - dx[j] for j, z in enumerate(edge)]
                for n, edge in enumerate(edges, 1)
            ],
            model.rest,
        )
        for i, p in enumerate(curve.mat):
            expected = material[i // 3][i % 3] if i < 12 else I.point(0)
            err = (poly_range(p, lo, hi) - expected).magnitude
            acc["node_j"][i] += span * err
            acc["node_b"][i] += span * remaining * err
        for face_no, ids in enumerate(model.faces):
            shift, rateshift, arm = shift_and_arm(x, v, ids, Q(float(model.rho)))
            for corner, n in enumerate(ids):
                a = 3 * face_no + corner
                if not mask[a]:
                    continue
                base = sub(curve.x[3 * n + 2], curve.x[12])
                speed = sub(curve.v[3 * n + 2], curve.v[12])
                actual_shift = shift + dx[3 * n + 2] - dx[12]
                actual_rate = rateshift + dv[3 * n + 2] - dv[12]
                nominal_shift = (actual_shift.low + actual_shift.high) / 2
                nominal_rate = (actual_rate.low + actual_rate.high) / 2
                cp = sub((WIDTH - nominal_shift,), base)
                fac = sub((1 - 1000 * nominal_rate,), scale(speed, 1000))
                comp = poly_range(cp, lo, hi) - (actual_shift - nominal_shift)
                # Monotone closing also supplies a bound at the left end of the whole chart.
                start_gap = direct_gap(curve.x, Q(0), a, model)
                minimum = max(comp.low, WIDTH - start_gap.high) if closing_prefix[a] else comp.low
                factor = 1 - 1000 * ports[a][1]
                need(minimum > 0 and factor.low > 0, "loaded branch is not active")
                p = curve.ps[a]
                pj = h * moment(p, lo, hi)
                pb = h * (tail + h * end) * moment(p, lo, hi) - h * h * moment(
                    times((0, 1), p), lo, hi
                )
                pg = poly_range(p, lo, hi)
                need(pg.low >= 0, "negative represented force")
                errp = sub(p, scale(times(cp, fac), AREA_K))
                spread = (actual_shift - nominal_shift).magnitude
                drift = (actual_rate - nominal_rate).magnitude
                err = poly_range(errp, lo, hi).magnitude + AREA_K * (
                    spread * poly_range(fac, lo, hi).magnitude + comp.magnitude * 1000 * drift
                )
                if "law_lower_abs" in acc:
                    nominal = poly_range(errp, lo, hi)
                    radius = AREA_K * (
                        spread * poly_range(fac, lo, hi).magnitude + 1000 * comp.magnitude * drift
                    )
                    floor = max(Q(0), nominal.low - radius, -nominal.high - radius)
                    acc["law_lower_abs"][a] += span * floor
                    acc["law_signed"][a] += span * (nominal + I(-radius, radius))
                acc["relative"][a] = max(acc["relative"][a], err / (AREA_K * minimum * factor.low))
                acc["je"][a] += span * err
                acc["be"][a] += span * remaining * err
                acc["j"][a] += pj
                acc["b"][a] += pb
                for i, ideal in enumerate(ports[a][2]):
                    center = (ideal.low + ideal.high) / 2
                    diff = poly_range(sub(curve.rows[a][i], scale(p, center)), lo, hi).magnitude
                    mismatch = (
                        diff + (ideal - center).magnitude * pg.magnitude + ideal.magnitude * err
                    )
                    acc["node_j"][i] += span * mismatch
                    acc["node_b"][i] += span * remaining * mismatch
                u = add(speed, (nominal_rate,))
                visc = AREA_K * 1000 * h * moment(times(cp, times(u, u)), lo, hi)
                du = poly_range(u, lo, hi).magnitude
                loss_error = (
                    span
                    * AREA_K
                    * 1000
                    * (spread * du**2 + comp.magnitude * drift * (2 * du + drift))
                )
                acc["loss"] += I(max(Q(0), visc - loss_error), visc + loss_error)
                for component, axis, sign in ((0, 1, 1), (1, 0, -1)):
                    arm_integral = h * moment(times(curve.x[3 * n + axis], p), lo, hi)
                    acc["torque"][component] += sign * (
                        arm_integral + pj * (dx[3 * n + axis] - arm[axis])
                    )


def energy(x, v, model):
    xx = list(map(I.point, x))
    _, mat = material_box(
        *[difference(xx[3 * n : 3 * n + 3], xx[:3]) for n in (1, 2, 3)], model.rest
    )
    layer = sum(AREA_K * positive(WIDTH - g).square() / 2 for g, _, _ in options(x, v, model))
    kinetic = sum(Q(float(m)) * z * z / 2 for m, z in zip(model.mass, v, strict=True))
    return kinetic + mat + layer


def angular(x, v, model):
    out = [Q(0)] * 3
    for i in range(4):
        k = 3 * i
        for j, z in enumerate(cross(x[k : k + 3], v[k : k + 3])):
            out[j] += Q(float(model.mass[k])) * z
    return out


def verify(model, original, source, proposal):
    inlet = source["input"]
    q, v, t = unpack(inlet, model.signature)
    h = source["h"]
    need(
        proposal["binding"] == original and proposal["input_sha"] == packet_hash(inlet),
        "different input",
    )
    left, right = proposal["left"], proposal["right"]
    need(left["binding"] == original, "left claim binding")
    mask = tuple(proposal["active"])
    ports = options(q, v, model)
    need(
        len(mask) == 6
        and all(type(z) is bool for z in mask)
        and all(
            (g.high < WIDTH if active else g.low > WIDTH) and r.high < 0
            for (g, r, _), active in zip(ports, mask, strict=True)
        ),
        "original active set",
    )
    prior = Chart(left, model, q, v, h)
    for i, on in enumerate(mask):
        if not on:
            need(
                prior.ps[i] == ZERO and all(z == ZERO for z in prior.rows[i]),
                "nonzero before event",
            )
    br = onset(proposal["ramp"], mask, prior, model)
    b, idx = br["b"], br["idx"]
    join = checkpoint(br["x"], br["v"], t + h * b, model.signature)
    need(
        packet_hash(join) == proposal["join_sha"] and right["binding"] == packet_hash(join),
        "algebraic event splice",
    )
    after = Chart(right, model, br["x"], br["v"], h * (1 - b))
    x1, v1 = after.at(Q(1))
    out = checkpoint(x1, v1, t + h, model.signature)
    need(packet_hash(out) == proposal["end_sha"], "old or forged terminal checkpoint")
    state = blank()
    audit_chart(prior, model, mask, b, h * (1 - b), state, br)
    next_mask = list(mask)
    next_mask[idx] = True
    audit_chart(after, model, next_mask, Q(1), Q(0), state)
    for i, taken in enumerate(next_mask):
        if not taken:
            need(options(x1, v1, model)[i][0].low > WIDTH, "unhandled second event")
    # All moments below use the target cell end as their common backward time origin.
    tail = h * (1 - b)
    state["relative"][idx] = max(state["relative"][idx], br["relative"])
    state["je"][idx] += br["law_j"]
    state["be"][idx] += br["law_b"] + tail * br["law_j"]
    J, B = br["j"], br["back"] + tail * br["j"]
    state["j"][idx] += J
    state["b"][idx] += B
    state["loss"] += I(0, br["loss"])
    for j in (0, 1):
        state["torque"][j] += I(-2 * J, 2 * J)
    for i, m in enumerate(model.mass):
        row = br["row"][i]
        ideal = br["ports"][idx][2][i]
        state["node_j"][i] += (row - ideal).magnitude * J + ideal.magnitude * br["law_j"]
        state["node_b"][i] += (row - ideal).magnitude * B + ideal.magnitude * (
            br["law_b"] + tail * br["law_j"]
        )
        jj, bb = row * J, row * B
        for chart, end, offset in ((prior, b, tail), (after, Q(1), Q(0))):
            wt = sub((chart.h * end + offset,), (0, chart.h))
            f = chart.mat[i]
            for mapped in chart.rows:
                f = add(f, mapped[i])
            jj += chart.h * moment(f, 0, end)
            bb += chart.h * moment(times(wt, f), 0, end)
        mass = Q(float(m))
        need(
            v[i] + jj / mass == v1[i] and q[i] + h * v[i] + bb / mass == x1[i],
            "full two-moment endpoint",
        )
    m0 = min(map(lambda m: Q(float(m)), model.mass))
    vel = max(3 * z / Q(float(m)) for z, m in zip(state["node_j"], model.mass, strict=True))
    pos = max(3 * z / Q(float(m)) for z, m in zip(state["node_b"], model.mass, strict=True))
    E = energy(x1, v1, model) - energy(q, v, model) + state["loss"]
    a0, a1 = angular(q, v, model), angular(x1, v1, model)
    ang = sum(
        (end - start - torque).magnitude
        for end, start, torque in zip(a1, a0, state["torque"], strict=True)
    )
    mom = [Q(0)] * 3
    for i, m in enumerate(model.mass):
        mom[i % 3 if i < 12 else 2] += Q(float(m)) * (v1[i] - v[i])
    # Read the old witness only after the independent new trajectory is fully reconstructed.
    old_sample = (source["witness_time"] - t) / h
    need(
        0 <= old_sample < br["lo"] and direct_gap(prior.x, old_sample, idx, model).low > WIDTH,
        "old witness not outside on the new path",
    )
    sample = direct_gap(prior.x, old_sample, idx, model)
    local_calls = left["owner_charges"] + right["owner_charges"] + 12
    consumed = source["prefix_charges"] + local_calls
    cells = state["leaves"] + 1 + 2 + 1 + 2 * 32
    metrics = dict(
        relative_positive=max(state["relative"]),
        owner_impulse_ns=max(state["je"]),
        owner_backward_over_mass_m=max(state["be"]) / m0,
        node_velocity_m_s=vel,
        node_position_m=pos,
        positive_energy_j=max(Q(0), E.high),
        paired_angular_nms=ang,
        linear_ns=sum(map(abs, mom)),
    )
    limits = dict(
        relative_positive=Q(1, 100),
        owner_impulse_ns=Q(1, 10**10),
        owner_backward_over_mass_m=Q(1, 10**10),
        node_velocity_m_s=Q(2, 10**6),
        node_position_m=Q(1, 10**8),
        positive_energy_j=Q(1, 10**8),
        paired_angular_nms=Q(1, 10**8),
        linear_ns=Q(1, 10**10),
    )
    gates = {k: z <= limits[k] for k, z in metrics.items()}
    gates.update(
        whole_cell_hard=state["mingap"] > 0,
        owner_budget=consumed <= 512,
        geometry_budget=source["prefix_geometry"] + cells <= 4096,
        unique_birth_and_no_exit_or_damping_cut=state["rate_high"] < 0,
        exact_two_moments=True,
    )
    return out, dict(
        accepted=all(gates.values()),
        gates=gates,
        upper={k: outward(I.point(z))[1] for k, z in metrics.items()},
        owners=list(model.owners),
        all_original_owners=True,
        model=model.signature,
        binding=original,
        same_inlet=True,
        inlet=packet_hash(inlet),
        join=packet_hash(join),
        new_endpoint=packet_hash(out),
        start_s=str(t),
        stop_s=str(t + h),
        event_owner=model.owners[idx],
        event_s=outward(I(t + h * br["lo"], t + h * br["hi"])),
        join_s=str(t + h * b),
        early_witness=dict(
            owner=model.owners[idx],
            time_s=float(source["witness_time"]),
            force_n=0,
            gap_m=outward(sample),
        ),
        minimum_hard_gap_m=outward(I.point(state["mingap"]))[0],
        charges=dict(
            inherited=source["prefix_charges"],
            new=local_calls,
            total=consumed,
            remaining=512 - consumed,
        ),
        proof_cells=dict(new=cells, inherited=source["prefix_geometry"], depth=32),
        J_ns=list(map(str, state["j"])),
        B_ns2=list(map(str, state["b"])),
        J_error_ns=[outward(I.point(z))[1] for z in state["je"]],
        B_error_over_mass_m=[outward(I.point(z / m0))[1] for z in state["be"]],
        relative_error=[outward(I.point(z))[1] for z in state["relative"]],
        energy_defect_j=outward(E),
        viscous_loss_j=outward(state["loss"]),
        platform_delta_v_m_s=float(v1[-1] - v[-1]),
        platform_x_m=float(x1[-1]),
    )
