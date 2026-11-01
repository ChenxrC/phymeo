"""Continue the SC2 entry packet and enclose the complete dense HOST reference trajectory."""

import argparse
import json
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path

import numpy as np
from check_p65_common_motion import integral, outward, scale
from check_p65_sc2_common_entry import AREA_K, WIDTH, evaluate, geometry
from p65_common_motion import checkpoint, need, unpack
from probe_p65_common_motion import saved
from probe_p65_joint_readback import save
from probe_p65_sc2_common_entry import SPEC, original
from verify_p65_contact_layer_screen import sha

from phymeo.validation.bending_bounds import Interval as I

H = Q(1, 64)
CELLS = 16
LEAVES = 16


def stage_fields(model, q, v):
    state = model.forces(q, v)
    return dict(
        q=q.copy(), v=v.copy(), material=state.material, rows=state.rows, pressure=state.pressures
    )


def stage_acc(frame, model):
    return (frame["material"] + frame["rows"].sum(axis=0) + model.external) / model.mass


def proposal_cell(q, v, dt, model):
    qo, vo = np.asarray(q, float), np.asarray(v, float)
    rows = []
    x, u = qo, vo
    for factor in (0.5, 0.5, 1.0, 0.0):
        f = stage_fields(model, x, u)
        rows.append(f)
        a = stage_acc(f, model)
        x, u = qo + float(dt) * factor * u, vo + float(dt) * factor * a
    return rows


def fractions(a):
    return [Q(float(z)) for z in a]


def step_moments(raw, model, q, v, dt):
    fs = []
    for frame in raw:
        mat = fractions(frame["material"])
        mapped = [fractions(row) for row in frame["rows"]]
        fs.append(
            [
                m + sum(row[i] for row in mapped) + Q(float(model.external[i]))
                for i, m in enumerate(mat)
            ]
        )
    q1, v1 = [], []
    for i, m in enumerate(map(lambda a: Q(float(a)), model.mass)):
        a, b, c, d = (f[i] for f in fs)
        j = dt * (a + 2 * b + 2 * c + d) / 6
        back = dt * dt * (a + b + c) / 6
        v1.append(v[i] + j / m)
        q1.append(q[i] + dt * v[i] + back / m)
    return q1, v1


def trace(root):
    model, _, _, bind = original()
    try:
        first = saved(root / "entry.json")
        prev = (root / "bridge.bin").read_bytes()
        need(
            first["input_binding"] == bind
            and sha256(prev).hexdigest() == first["checkpoint_sha256"],
            "bridge binding",
        )
        for k in range(8):
            q, v, t = unpack(np.frombuffer(prev, dtype="<i4"), model.signature)
            interval = ((k + 1) * H - t) / CELLS
            need(interval > 0 and (k == 0 or t == k * H), "time continuity")
            cells = []
            for _ in range(CELLS):
                raw = proposal_cell(q, v, interval, model)
                q, v = step_moments(raw, model, q, v, interval)
                cells.append(raw)
            packet = checkpoint(q, v, (k + 1) * H, model.signature).tobytes()
            charge = 4 * CELLS * len(model.owners) + (first["owner_charges"] if k == 0 else 0)
            need(charge <= 512, "original-step owner budget")
            save(
                root / f"sc2-step-{k}.json.gz",
                dict(
                    k=k,
                    time=str(t),
                    dt=str(interval),
                    model=model.signature,
                    binding=bind,
                    owners=model.owners,
                    in_sha=sha256(prev).hexdigest(),
                    out_sha=sha256(packet).hexdigest(),
                    charges=charge,
                    cells=cells,
                ),
            )
            with (root / f"sc2-step-{k}.bin").open("xb") as f:
                f.write(packet)
            prev = packet
            print("saved original step", k, "charges", charge)
    finally:
        model.parts[3].close()


def quadratic(values):
    a, b, c, d = values
    return (a, 2 * (b + c) - 3 * a - d, 2 * (a + d - b - c))


def cubic_force_curves(raw, model):
    # Separate material fields, all six world pressure rows and original external force.
    # Use an independent interpolation of the stage values and integrate both primitives.
    fp = []
    for i in range(13):
        seq = []
        for f in raw:
            seq.append(
                Q(float(f["material"][i]))
                + sum(Q(float(rr[i])) for rr in f["rows"])
                + Q(float(model.external[i]))
            )
        fp.append(quadratic(seq))
    pressure = [quadratic([Q(float(f["pressure"][i])) for f in raw]) for i in range(6)]
    return fp, pressure


def positive(x):
    return I(max(Q(0), x.low), max(Q(0), x.high))


def point_bounds(p, s):
    return I.point(evaluate(p, s))


def poly_range(p, lo, hi):
    # On a short subinterval, Taylor coefficients at the left endpoint wrap far
    # less than Horner on the global monomial interval. Each enclosure is outward.
    from math import comb

    degree = len(p) - 1
    result = I.point(evaluate(p, lo))
    width = hi - lo
    for k in range(1, degree + 1):
        d = sum(p[j] * comb(j, k) * lo ** (j - k) for j in range(k, degree + 1)) * width**k
        result += I(min(0, d), max(0, d))
    return result.rounded(96)


def total_pressure(ports):
    return [AREA_K * positive(WIDTH - gap) * positive(1 - 1000 * rate) for gap, rate, _ in ports]


def cell_observations(q, v, dt, raw, model, time):
    ff, pressure = cubic_force_curves(raw, model)
    vel = [
        integral(scale(f, dt / Q(float(m))), u) for f, m, u in zip(ff, model.mass, v, strict=True)
    ]
    pos = [integral(scale(vp, dt), x) for vp, x in zip(vel, q, strict=True)]
    observations = []
    for leaf in range(LEAVES):
        lo, hi = Q(leaf, LEAVES), Q(leaf + 1, LEAVES)
        region = geometry(
            [poly_range(p, lo, hi) for p in pos], [poly_range(p, lo, hi) for p in vel], model
        )
        law = total_pressure(region)
        prs = [poly_range(p, lo, hi) for p in pressure]
        mid = (lo + hi) / 2
        at_mid = geometry(
            [point_bounds(p, mid) for p in pos], [point_bounds(p, mid) for p in vel], model
        )
        law_mid = sum(total_pressure(at_mid))
        represented = sum(evaluate(p, mid) for p in pressure)
        row = dict(
            start=float(time + dt * lo),
            end=float(time + dt * hi),
            sample_t=float(time + dt * mid),
            gap=[outward(g) for g, _, _ in region],
            rate=[outward(z) for _, z, _ in region],
            law=outward(sum(law)),
            law_sample=outward(law_mid),
            represented=outward(sum(prs)),
            represented_sample=outward(I.point(represented)),
            negative_or_outside_witness=[],
        )
        for i, ((g, _, _), p) in enumerate(zip(at_mid, pressure, strict=True)):
            pf = evaluate(p, mid)
            if pf < 0 or (g.low > WIDTH and pf != 0):
                row["negative_or_outside_witness"].append(
                    dict(owner=model.owners[i], p_n=float(pf), gap_m=outward(g))
                )
        observations.append(row)
    end_q, end_v = [sum(p) for p in pos], [sum(p) for p in vel]
    return end_q, end_v, observations


def extrema(rows, key, sample):
    low = max(row[sample][0] for row in rows)
    high = max(row[key][1] for row in rows)
    possible = [r for r in rows if r[key][1] >= low]
    return dict(
        force_n=[low, high],
        time_s=[possible[0]["start"], possible[-1]["end"]],
        candidate_cells=len(possible),
    )


def audit(root):
    model, _, _, bind = original()
    try:
        entry = saved(root / "entry.json")
        prev = (root / "bridge.bin").read_bytes()
        need(
            sha256(prev).hexdigest() == entry["checkpoint_sha256"]
            and bind == entry["input_binding"],
            "foreign bridge checkpoint",
        )
        rows = []
        boundaries = []
        files = []
        for k in range(8):
            path = root / f"sc2-step-{k}.json.gz"
            src = saved(path)
            q, v, t = unpack(np.frombuffer(prev, dtype="<i4"), model.signature)
            dt = Q(src["dt"])
            raw_cells = src["cells"]
            expected_charge = 4 * CELLS * 6 + (entry["owner_charges"] if k == 0 else 0)
            need(
                src["k"] == k
                and src["model"] == model.signature
                and src["binding"] == bind
                and src["in_sha"] == sha256(prev).hexdigest()
                and src["owners"] == list(model.owners)
                and Q(src["time"]) == t
                and t + dt * CELLS == (k + 1) * H
                and len(raw_cells) == CELLS
                and src["charges"] == expected_charge <= 512,
                "missing step, owner, time, charge or binding",
            )
            for raw in raw_cells:
                need(
                    len(raw) == 4
                    and np.array_equal(raw[0]["q"], np.asarray(q, float))
                    and np.array_equal(raw[0]["v"], np.asarray(v, float)),
                    "lost incoming state",
                )
                for stage in raw:
                    need(
                        stage["material"].shape == (13,)
                        and stage["rows"].shape == (6, 13)
                        and stage["pressure"].shape == (6,)
                        and np.min(stage["pressure"]) >= 0,
                        "stage field schema",
                    )
                q, v, obs = cell_observations(q, v, dt, raw, model, t)
                rows.extend(obs)
                t += dt
            prev = (root / f"sc2-step-{k}.bin").read_bytes()
            need(
                sha256(prev).hexdigest() == src["out_sha"]
                and unpack(np.frombuffer(prev, dtype="<i4"), model.signature) == (q, v, t),
                "two moment readback mismatch",
            )
            boundaries.append(sha256(prev).hexdigest())
            files.append(sha256(path.read_bytes()).hexdigest())
        end_ports = geometry(list(map(I.point, q)), list(map(I.point, v)), model)
        min_gap = min(g[0] for row in rows for g in row["gap"])
        need(min_gap > 0, "reference hard gap unresolved")
        peak = extrema(rows, "law", "law_sample")
        delivered = extrema(rows, "represented", "represented_sample")
        duration_min = Q(entry["bridge_end_s"]) - Q(entry["event_seconds"][1])
        ramp_max = 18 * Q(entry["J_ns"]) / duration_min + 60 * Q(entry["B_ns2"]) / duration_min**2
        prefix_peak_upper = ramp_max / (1 - Q(entry["incident_relative_error_upper"]))
        need(
            prefix_peak_upper < Q(peak["force_n"][0]) and peak["time_s"][1] < rows[-1]["start"],
            "peak outside verified comparison",
        )
        for i in range(6):
            need(any(r["gap"][i][1] < float(WIDTH) for r in rows), "missing material entry")
        first_full = next(
            i for i, r in enumerate(rows) if all(z[1] < float(WIDTH) for z in r["gap"])
        )
        all_inside_after = all(
            all(z[1] < float(WIDTH) for z in r["gap"]) for r in rows[first_full:]
        )
        always_at_least_one = all(any(z[1] < float(WIDTH) for z in r["gap"]) for r in rows)
        witnesses = [(r["sample_t"], z) for r in rows for z in r["negative_or_outside_witness"]]
        end_values = total_pressure(end_ports)
        end_law = outward(sum(end_values))
        owner_exit = []
        for i, owner in enumerate(model.owners):
            inside = next(k for k, r in enumerate(rows) if r["gap"][i][1] < float(WIDTH))
            outside = next((r for r in rows[inside + 1 :] if r["gap"][i][0] > float(WIDTH)), None)
            owner_exit.append(
                dict(
                    owner=owner,
                    inside_by_s=rows[inside]["end"],
                    later_outside_by_s=outside["end"] if outside else None,
                )
            )
        need(end_law[1] < peak["force_n"][0], "unloading unverified")
        summary = dict(
            scope="finite HOST reference, original SC2 nominal h only",
            protocol_sha256_lf=sha(SPEC, normalize=True),
            model=model.signature,
            input_binding=bind,
            entry_sha=sha256((root / "entry.json").read_bytes()).hexdigest(),
            bridge_checkpoint=entry["checkpoint_sha256"],
            eight_boundary_sha256=boundaries,
            eight_stage_files_sha256=files,
            certified_entry=entry["first"],
            peak_law=peak,
            peak_represented=delivered,
            min_gap_lower_m=min_gap,
            end_gap=[outward(g) for g, _, _ in end_ports],
            end_rate=[outward(z) for _, z, _ in end_ports],
            end_law_n=end_law,
            end_owner_law_n=[outward(z) for z in end_values],
            owner_exit_observations=owner_exit,
            prefix_peak_upper_n=float(prefix_peak_upper),
            end_platform_speed_m_s=float(v[-1]),
            all_six_inside_by_s=rows[first_full]["end"],
            all_six_remain_inside=all_inside_after,
            one_or_more_inside_entire_continuation=always_at_least_one,
            positive_unloading_drop_n_lower=peak["force_n"][0] - end_law[1],
            full_law_certified=False,
            sample_spurious_or_negative_count=len(witnesses),
            first_spurious_or_negative=witnesses[0] if witnesses else None,
            strongest_spurious_or_negative=max(witnesses, key=lambda p: abs(p[1]["p_n"]))
            if witnesses
            else None,
            time_cells_per_outer_step=CELLS * LEAVES,
            charges=[468] + [384] * 7,
        )
        save(root / "window-bounds.json.gz", rows)
        save(root / "window.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        model.parts[3].close()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True, type=Path)
    cli.add_argument("--phase", required=True, choices=("window", "audit"))
    args = cli.parse_args()
    (trace if args.phase == "window" else audit)(args.root)
