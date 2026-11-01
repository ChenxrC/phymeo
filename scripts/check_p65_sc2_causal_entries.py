"""Causal piecewise forces on a shared four-node SC2 trajectory."""

from fractions import Fraction as Q

import numpy as np
from check_p65_common_motion import add, derive, integral, outward, poly, scale, times
from check_p65_sc2_common_entry import AREA_K, WIDTH, difference, evaluate, geometry
from check_p65_sc2_event_cell import (
    angular,
    audit_chart,
    blank,
    energy,
    moment,
    options,
    packet_hash,
    word_polys,
)
from p65_common_motion import GRID, checkpoint, need, unpack
from trace_p65_sc2_common_window import poly_range

from phymeo.validation.bending_bounds import Interval as I
from phymeo.validation.bending_bounds import cross, sqrt_interval

H = Q(1, 64)
DEPTH = 32
GRID_ROOT = 1 << DEPTH


def arc_horizon(model, q, v):
    ports = options(q, v, model)
    mask = []
    forecasts = []
    for i, (gap, rate, _) in enumerate(ports):
        need(rate.high < 0 and (gap.high < WIDTH or gap.low > WIDTH), "unresolved inlet branch")
        mask.append(gap.high < WIDTH)
        if not mask[-1]:
            forecasts.append(((gap.high - WIDTH) / (-rate.high), i))
    need(forecasts, "no approaching outside address")
    earliest = min(forecasts)[0]
    group = [(s, i) for s, i in forecasts if s <= 8 * earliest]
    target = 2 * max(s for s, _ in group)
    step = H
    need(target <= step, "forecast beyond original step")
    while step / 2 >= target:
        step /= 2
        need(step >= H / 2**32, "forecast selector depth")
    return step, mask


class Curve:
    """Integrate each represented paired force with its own identity and the original inertia."""

    def __init__(self, claim, model, q, v, h, mode, degree):
        aff = mode == "affine"
        need(
            mode in ("affine", "chebyshev", "active-newton", "slope-newton"),
            "unknown curve basis or solver",
        )
        if mode == "slope-newton":
            need(
                degree == 20 and claim.get("schema") == "p65-slope-newton-curve-v1",
                "wrong slope schedule",
            )
            calls = 1 + 3 * 3 + 2 * 13
        elif mode == "active-newton":
            need(
                degree == 13 and claim.get("schema") == "p65-active-newton-curve-v1",
                "wrong Newton schedule",
            )
            calls = 1 + 3 * 3 + 2 * degree
        else:
            calls = 2 if aff else 1 + 3 * degree
        count = 2 if aff else degree + 1
        need(
            tuple(claim["owners"]) == model.owners and Q(claim["h"]) == h and claim["grid"] == GRID,
            "owner order, duration or grid",
        )
        need(
            np.array_equal(claim["mass"], model.mass)
            and np.array_equal(claim["external"], model.external)
            and not np.any(model.external),
            "original load or mass",
        )
        need(
            claim["force_calls"] == calls and claim["owner_charges"] == 6 * calls,
            "field evaluation ledger",
        )
        fields = [claim[k] for k in ("material", "pressure", "rows")]
        need(
            [f.shape for f in fields] == [(count, 13), (count, 6), (count, 6, 13)]
            and all(f.dtype == np.int64 for f in fields),
            "force schema",
        )

        def decode(words):
            if not aff:
                return word_polys(words)
            a, b = map(int, words)
            return poly((Q(a, GRID), Q(b - a, GRID)))

        mat, ps, rows = fields
        self.mat = [decode(mat[:, i]) for i in range(13)]
        self.ps = [decode(ps[:, i]) for i in range(6)]
        self.rows = [[decode(rows[:, a, i]) for i in range(13)] for a in range(6)]
        self.x = []
        self.v = []
        self.h = h
        for i, m in enumerate(model.mass):
            force = self.mat[i]
            for row in self.rows:
                force = add(force, row[i])
            vel = integral(scale(force, h / Q(float(m))), v[i])
            self.v.append(vel)
            self.x.append(integral(scale(vel, h), q[i]))
            need(derive(self.x[-1]) == scale(vel, h), "non-authoritative velocity")


class Ramp:
    """The two rational moments fix the positive density on the true root-to-end duration."""

    def __init__(self, data, model, h, end=Q(1)):
        self.idx = data["index"]
        need(
            type(self.idx) is int
            and 0 <= self.idx < 6
            and data["owner"] == model.owners[self.idx]
            and data["before"] == 0,
            "event address or nonzero prebirth pressure",
        )
        self.lo, self.hi = map(Q, (data["lo"], data["hi"]))
        self.end = end
        need(
            0 < self.lo < self.hi < end
            and self.hi - self.lo == Q(1, GRID_ROOT)
            and (self.lo * GRID_ROOT).denominator == 1,
            "unresolved or malformed root",
        )
        self.h = h
        self.j, self.b = map(Q, (data["J"], data["B"]))
        slope = Q(data["slope"])
        self.row = list(map(Q, data["map"]))
        self.mass = list(map(lambda m: Q(float(m)), model.mass))
        dm = h * (end - (self.lo + self.hi) / 2)
        self.duration = I(h * (end - self.hi), h * (end - self.lo))
        need(
            slope > 0
            and len(self.row) == 13
            and self.row[-1] == -1
            and self.j > 0
            and self.j == Q(round(slope * dm**2 / 2 * (1 << 112)), 1 << 112)
            and self.b == self.j * dm / 3,
            "invalid event moments or map",
        )
        ratio = I.point(dm) / self.duration
        k = 2 * self.j / dm**2 * ratio.square()
        self.a = (k * (4 * ratio - 3)).rounded(160)
        self.c = (6 * k * (1 - ratio)).rounded(160)
        late = (k * (3 - 2 * ratio)).rounded(160)
        self.slope = I(min(self.a.low, late.low), max(self.a.high, late.high))
        need(self.slope.low > 0, "nonpositive incident ramp")

    def moments_at(self, s):
        if s <= self.lo:
            return I.point(0), I.point(0)
        if s == self.end:
            return I.point(self.j), I.point(self.b)
        need(0 <= s <= self.end, "time outside event packet")
        tau = I(max(Q(0), self.h * (s - self.hi)), self.h * (s - self.lo))
        t2 = tau.square()
        jj = self.a * t2 / 2 + self.c * t2 * tau / (3 * self.duration)
        bb = self.a * t2 * tau / 6 + self.c * t2.square() / (12 * self.duration)
        return (
            I(max(Q(0), jj.low), min(self.j, jj.high)),
            I(max(Q(0), bb.low), min(self.b, bb.high)),
        )


def state_at(curve, ramps, s):
    x = [I.point(evaluate(p, s)) for p in curve.x]
    v = [I.point(evaluate(p, s)) for p in curve.v]
    for ramp in ramps:
        j, b = ramp.moments_at(s)
        for i, (r, m) in enumerate(zip(ramp.row, ramp.mass, strict=True)):
            x[i] += r / m * b
            v[i] += r / m * j
    return [z.rounded(160) for z in x], [z.rounded(160) for z in v]


def director_gap(x, model, idx):
    face = model.faces[idx // 3]
    pts = [x[3 * n : 3 * n + 3] for n in face]
    c = cross(difference(pts[1], pts[0]), difference(pts[2], pts[0]))
    need(c[2].low > 0, "unsquared normal orientation")
    length = sqrt_interval(sum(z.square() for z in c), 160)
    rho = Q(float(model.rho))
    return (pts[idx % 3][2] - x[12] + rho - rho * c[2] / length).rounded(160)


def require_order(curve, model, mask, ramps):
    remaining = set(i for i, on in enumerate(mask) if not on)
    previous = []
    for ramp in ramps:
        need(
            ramp.idx in remaining and (not previous or previous[-1].hi < ramp.lo),
            "overlapping, reordered or duplicate births",
        )
        # Brackets must straddle the unsquared gap before installing this owner's force.
        left, _ = state_at(curve, previous, ramp.lo)
        right, _ = state_at(curve, previous, ramp.hi)
        need(
            director_gap(left, model, ramp.idx).low > WIDTH
            and director_gap(right, model, ramp.idx).high < WIDTH,
            "causal root signs",
        )
        remaining.remove(ramp.idx)
        need(
            all(director_gap(right, model, k).low > WIDTH for k in remaining),
            "not the earliest event",
        )
        previous.append(ramp)
    return remaining


def cluster_audit(curve, model, mask, ramps, end, acc, *, leaves=64):
    # A common enclosure includes every earlier-event continuation and the final path.
    dx = [I.point(0) for _ in range(13)]
    dv = dx.copy()
    for r in ramps:
        for i, (a, m) in enumerate(zip(r.row, r.mass, strict=True)):
            dx[i] += a / m * I(0, r.b)
            dv[i] += a / m * I(0, r.j)
    x = [poly_range(z, 0, end) + d for z, d in zip(curve.x, dx, strict=True)]
    v = [poly_range(z, 0, end) + d for z, d in zip(curve.v, dv, strict=True)]
    outer = geometry(x, v, model, jacobian=True)
    need(
        max(z.high for _, z, _ in outer) < 0 and min(g.low for g, _, _ in outer) > 0,
        "unresolved branch change, damping cut, or hard gap",
    )
    audit_chart(curve, model, mask, end, Q(0), acc, dict(lo=0, dx=dx, dv=dv), leaves=leaves)
    events = []
    for r in ramps:
        rx = [poly_range(z, r.lo, end) + d for z, d in zip(curve.x, dx, strict=True)]
        rv = [poly_range(z, r.lo, end) + d for z, d in zip(curve.v, dv, strict=True)]
        _, rate, map_box = geometry(rx, rv, model, jacobian=True)[r.idx]
        material_slope = AREA_K * (-rate) * (1 - 1000 * rate)
        err = (r.slope - material_slope).magnitude
        rel = err / material_slope.low
        je = err * r.duration.high**2 / 2
        be = err * r.duration.high**3 / 6
        acc["relative"][r.idx] = max(acc["relative"][r.idx], rel)
        acc["je"][r.idx] += je
        acc["be"][r.idx] += be
        acc["j"][r.idx] += r.j
        acc["b"][r.idx] += r.b
        acc["loss"] += I(0, AREA_K * 1000 * (-rate).high**3 * r.duration.high**2 / 2)
        for i in (0, 1):
            acc["torque"][i] += I(-2 * r.j, 2 * r.j)
        for i, d in enumerate(map_box):
            acc["node_j"][i] += (r.row[i] - d).magnitude * r.j + d.magnitude * je
            acc["node_b"][i] += (r.row[i] - d).magnitude * r.b + d.magnitude * be
        events.append(
            dict(
                owner=model.owners[r.idx],
                index=r.idx,
                root_unit=[str(r.lo), str(r.hi)],
                prebirth_force_n=0,
                point_impulse_ns=0,
                J_ns=str(r.j),
                B_ns2=str(r.b),
                positive_time_relative_upper=outward(I.point(rel))[1],
                final_pressure_n=outward(r.duration * (r.a + r.c)),
            )
        )
    return events


def represented_end(curve, model, q, v, ramps, end):
    x = []
    vel = []
    for i, mi in enumerate(model.mass):
        m = Q(float(mi))
        f = curve.mat[i]
        for row in curve.rows:
            f = add(f, row[i])
        j = curve.h * moment(f, 0, end) + sum(r.row[i] * r.j for r in ramps)
        b = curve.h**2 * moment(times((end, -1), f), 0, end) + sum(r.row[i] * r.b for r in ramps)
        x.append(q[i] + curve.h * end * v[i] + b / m)
        vel.append(v[i] + j / m)
        need(
            x[-1] == evaluate(curve.x[i], end) + sum(r.row[i] * r.b / m for r in ramps)
            and vel[-1] == evaluate(curve.v[i], end) + sum(r.row[i] * r.j / m for r in ramps),
            "two-moment endpoint",
        )
    return x, vel


def peak_box(curve, model, ramps, end, t, leaves):
    # Bound the constitutive law on the actual combined motion, including switches.
    bounds = []
    for k in range(leaves):
        lo, hi = end * k / leaves, end * (k + 1) / leaves
        dx = [I.point(0) for _ in range(13)]
        dv = dx.copy()
        for r in ramps:
            jl, bl = r.moments_at(lo)
            jr, br = r.moments_at(hi)
            for j, a in enumerate(r.row):
                dx[j] += a / r.mass[j] * I(bl.low, br.high)
                dv[j] += a / r.mass[j] * I(jl.low, jr.high)
        x = [poly_range(z, lo, hi) + d for z, d in zip(curve.x, dx, strict=True)]
        v = [poly_range(z, lo, hi) + d for z, d in zip(curve.v, dv, strict=True)]

        def law(pos, vel):
            return sum(
                AREA_K
                * I(max(Q(0), WIDTH - g.high), max(Q(0), WIDTH - g.low))
                * I(max(Q(0), 1 - 1000 * z.high), max(Q(0), 1 - 1000 * z.low))
                for g, z, _ in geometry(pos, vel, model)
            )

        val = law(x, v)
        pt = law(*state_at(curve, ramps, (lo + hi) / 2))
        bounds.append((t + curve.h * lo, t + curve.h * hi, pt.low, val.high))
    lower = max(z[2] for z in bounds)
    possible = [z for z in bounds if z[3] >= lower]
    return dict(
        force_n=outward(I(lower, max(z[3] for z in possible))),
        time_s=outward(I(possible[0][0], possible[-1][1])),
        leaves=leaves,
    )


def first_totals(entry, model, packet, qinit, vinit):
    q, v, t = unpack(packet, model.signature)
    i = model.owners.index(entry["first"])
    j = [Q(0)] * 6
    b = j.copy()
    j[i] = Q(entry["impulse_error_upper_ns"])
    b[i] = Q(entry["time_moment_error_over_min_mass_m"])
    linear = [Q(0)] * 3
    for n, m in enumerate(model.mass):
        linear[n % 3 if n < 12 else 2] += Q(float(m)) * (v[n] - vinit[n])
    return dict(
        j=j,
        b=b,
        pos=Q(entry["node_position_defect_m"]),
        vel=Q(entry["node_velocity_defect_m_s"]),
        energy=Q(entry["positive_energy_upper_j"]),
        angular=Q(entry["paired_angular_upper_nms"]),
        linear=sum(map(abs, linear)),
        rel=Q(entry["incident_relative_error_upper"]),
    )


def finish(source, model, data, curve, ramps, mask, end, acc, events, extra_cells):
    packet = source["input"]
    q, v, t = unpack(packet, model.signature)
    x, u = represented_end(curve, model, q, v, ramps, end)
    stop = t + curve.h * end
    out = checkpoint(x, u, stop, model.signature)
    need(packet_hash(out) == data["endpoint"], "wrong endpoint moments")
    endports = options(x, u, model)
    remaining = set(i for i, z in enumerate(mask) if not z) - {r.idx for r in ramps}
    need(all(endports[i][0].low > WIDTH for i in remaining), "unreported later birth")
    need(
        all(g.high < WIDTH for i, (g, z, _) in enumerate(endports) if i not in remaining),
        "loaded owner is outside at endpoint",
    )
    dt = stop - t
    m0 = Q(float(min(model.mass)))
    E = energy(x, u, model) - energy(q, v, model) + acc["loss"]
    a0, a1 = angular(q, v, model), angular(x, u, model)
    angular_err = sum(
        (last - first - torque).magnitude
        for first, last, torque in zip(a0, a1, acc["torque"], strict=True)
    )
    lin = [Q(0)] * 3
    for i, m in enumerate(model.mass):
        lin[i % 3 if i < 12 else 2] += Q(float(m)) * (u[i] - v[i])
    vel = max(3 * z / Q(float(m)) for z, m in zip(acc["node_j"], model.mass, strict=True))
    pos = max(3 * z / Q(float(m)) for z, m in zip(acc["node_b"], model.mass, strict=True))
    prev = source["totals"]
    total = dict(
        j=[a + b for a, b in zip(prev["j"], acc["je"], strict=True)],
        b=[
            old + dt * err / m0 + local / m0
            for old, err, local in zip(prev["b"], prev["j"], acc["be"], strict=True)
        ],
        vel=prev["vel"] + vel,
        pos=prev["pos"] + dt * prev["vel"] + pos,
        energy=prev["energy"] + max(Q(0), E.high),
        angular=prev["angular"] + angular_err,
        linear=prev["linear"] + sum(map(abs, lin)),
        rel=max(prev["rel"], max(acc["relative"])),
    )
    metrics = dict(
        relative_positive=total["rel"],
        owner_impulse_ns=max(total["j"]),
        owner_backward_over_mass_m=max(total["b"]),
        node_velocity_m_s=total["vel"],
        node_position_m=total["pos"],
        positive_energy_j=total["energy"],
        paired_angular_nms=total["angular"],
        linear_ns=total["linear"],
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
    peak = peak_box(curve, model, ramps, end, t, 64)
    count = source["prefix_charges"] + data["charges"]
    cells = (
        source["prefix_cells"] + data["search_cells"] + acc["leaves"] + extra_cells + peak["leaves"]
    )
    gates = {key: value <= limits[key] for key, value in metrics.items()}
    gates.update(
        whole_path_hard=acc["mingap"] > 0,
        exact_two_moments=True,
        owner_budget=count <= 512,
        cell_budget=cells <= 4096,
    )
    for event in events:
        lo, hi = map(Q, event["root_unit"])
        event["seconds"] = outward(I(t + curve.h * lo, t + curve.h * hi))
    return out, dict(
        accepted=all(gates.values()),
        gates=gates,
        owners=list(model.owners),
        all_original_owners=True,
        inlet=packet_hash(packet),
        endpoint=packet_hash(out),
        start_s=str(t),
        stop_s=str(stop),
        binding=source["binding"],
        model=model.signature,
        events=events,
        distinct_and_ordered=True,
        local_relative=[outward(I.point(z))[1] for z in acc["relative"]],
        local_J_ns=list(map(str, acc["j"])),
        local_B_ns2=list(map(str, acc["b"])),
        local_J_error_upper_ns=[outward(I.point(z))[1] for z in acc["je"]],
        local_B_error_over_mass_m=[outward(I.point(z / m0))[1] for z in acc["be"]],
        upper={k: outward(I.point(z))[1] for k, z in metrics.items()},
        cumulative_J_error_upper_ns=[outward(I.point(z))[1] for z in total["j"]],
        cumulative_B_error_over_mass_m=[outward(I.point(z))[1] for z in total["b"]],
        hard_gap_lower_m=outward(I.point(acc["mingap"]))[0],
        local_energy_j=outward(E),
        local_angular_upper_nms=outward(I.point(angular_err))[1],
        charges=dict(
            inherited=source["prefix_charges"],
            new=data["charges"],
            total=count,
            remaining=512 - count,
        ),
        cells=dict(
            new=cells - source["prefix_cells"], inherited=source["prefix_cells"], total=cells
        ),
        peak=peak,
        terminal_gaps_m=[outward(g) for g, _, _ in endports],
        terminal_rates_m_s=[outward(z) for _, z, _ in endports],
    )


def verify_cluster(model, source, data):
    q, v, t = unpack(source["input"], model.signature)
    h, mask = arc_horizon(model, q, v)
    need(
        h == source["h"] == Q(data["h"])
        and t + h <= H
        and data["binding"] == source["binding"]
        and data["input_sha"] == packet_hash(source["input"])
        and data["active"] == mask,
        "wrong input or active set",
    )
    curve = Curve(data["chart"], model, q, v, h, "affine", 1)
    need(data["chart"]["binding"] == source["binding"], "foreign background")
    ramps = [Ramp(r, model, h) for r in data["ramps"]]
    need(1 <= len(ramps) <= 6 and data["charges"] == 6 * (3 + len(ramps)), "causal owner ledger")
    count = 0
    for ring in data["rounds"]:
        count += sum(1 + (2 * DEPTH if p["bracket"] is not None else 0) for p in ring)
    need(
        len(data["rounds"]) == len(ramps) + 1 and count == data["search_cells"],
        "isolation work ledger",
    )
    remaining = require_order(curve, model, mask, ramps)
    acc = blank()
    events = cluster_audit(curve, model, mask, ramps, Q(1), acc)
    endx, _ = state_at(curve, ramps, Q(1))
    need(all(director_gap(endx, model, i).low > WIDTH for i in remaining), "skipped final event")
    return finish(source, model, data, curve, ramps, mask, Q(1), acc, events, 1 + len(ramps) * 2)
