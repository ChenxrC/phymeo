"""Compare fixed scalar step paths with the archived SC-L1 reference, without rerunning it."""

import argparse
import csv
import json
import math
from collections import defaultdict
from fractions import Fraction as Q
from math import comb
from pathlib import Path

from verify_p65_contact_layer_screen import INPUT, INPUT_SHA, sha

ROOT = Path(__file__).resolve().parents[1]
PLAN = INPUT.with_name("p65-contact-layer-intrastep-protocol.json")
PLAN_SHA = "d65d0d6138a2f25fa10f64a6dd3dc76f57a8cbeb1364fcc03f70e2f430048e87"


def product(a, b):
    """Bernstein coefficients on the unit interval, computed with exact rationals."""
    m, n = len(a) - 1, len(b) - 1
    result = [Q(0)] * (m + n + 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i + j] += x * y * Q(comb(m, i) * comb(n, j), comb(m + n, i + j))
    return result


def peak_range(coeff, left, right, tolerance, budget, depth_cap):
    """Bound max(0,polynomial) and return a time attaining the reported lower bound."""
    best, time = max(0, coeff[0]), left
    if coeff[-1] > best:
        best, time = coeff[-1], right
    pending = [(coeff, left, right, 0)]
    upper = best
    cells = 1
    while pending:
        p, a, b, depth = pending.pop()
        maximum = max(0, *p)
        if maximum <= best + tolerance:
            upper = max(upper, maximum)
            continue
        if depth >= depth_cap or cells + 2 > budget:
            raise ArithmeticError("force peak enclosure exhausted")
        levels = [p]
        while len(levels[-1]) > 1:
            levels.append([(x + y) / 2 for x, y in zip(levels[-1], levels[-1][1:])])
        mid = (a + b) / 2
        if levels[-1][0] > best:
            best, time = levels[-1][0], mid
        pending.extend(
            [
                ([r[-1] for r in reversed(levels)], mid, b, depth + 1),
                ([r[0] for r in levels], a, mid, depth + 1),
            ]
        )
        cells += 2
    return best, max(best, upper), time, cells


def check_piece(piece, width, k, d, M, external, plan, remaining_cells):
    a, b = Q(piece["t0"]), Q(piece["t1"])
    h = b - a
    g0, g1 = Q(piece["g0"]), Q(piece["g1"])
    v0, ve = (Q(piece["y0"]) - 1) / d, (Q(piece["ye"]) - 1) / d
    j_motion = external * h - M * (ve - v0)
    b_motion = external * h * h / 2 + M * (g1 - g0 + h * v0)
    if piece["phase"] == "free":
        if min(g0, g1) < width or external or v0 != ve:
            raise ArithmeticError("incorrect free segment")
        return dict(
            j=(0, 0),
            b=(0, 0),
            diss=(0, 0),
            j_motion=j_motion,
            b_motion=b_motion,
            prefix=0,
            energy_positive=0,
            gap_lower=min(g0, g1),
            peak=(0, 0, a),
            cells=0,
            inferred_min=0,
            negative_impulse=0,
            middle_velocity_error=0,
        )

    # The Hermite path uses published gaps and endpoint velocities. Do not take
    # the producer's middle coefficient as the derivative of the stored path.
    z0, z3 = width - g0, width - g1
    z = [z0, z0 + h * v0 / 3, z3 - h * ve / 3, z3]
    velocities = [3 * (v - u) / h for u, v in zip(z, z[1:])]
    y = [1 + d * v for v in velocities]
    gap_lower = width - max(z)
    if min(z) < 0 or gap_lower < 0:
        raise ArithmeticError("could not enclose active path between layer edge and hard core")
    raw = [k * x for x in product(z, y)]
    # If the rate factor changes sign, actual force is max(raw,0). Bound the
    # missing positive part and retain its full effect on impulse, work and moment.
    clipping = k * max(z) * max(0, -min(y))
    n = len(raw) - 1
    j = h * sum(raw) / (n + 1)
    moment = h * h * sum(p * (n + 1 - i) for i, p in enumerate(raw)) / ((n + 1) * (n + 2))
    j_bounds = max(0, j), max(0, j + clipping * h)
    b_bounds = max(0, moment), max(0, moment + clipping * h * h / 2)
    loss_coeff = product(z, product(velocities, velocities))
    raw_loss = h * k * d * sum(loss_coeff) / len(loss_coeff)
    diss = max(0, raw_loss - h * clipping * max(map(abs, velocities))), raw_loss
    defect_upper = (
        M * (ve * ve - v0 * v0) / 2 + k * (z3 * z3 - z0 * z0) / 2 - external * (z3 - z0) + diss[1]
    )

    # Inertial force inferred from the actual polynomial acceleration is linear.
    # It is compared with the constitutive traction; neither is repaired.
    left = external - 2 * M * (velocities[1] - velocities[0]) / h
    right = external - 2 * M * (velocities[2] - velocities[1]) / h
    residual = [p - (left + (right - left) * Q(i, n)) for i, p in enumerate(raw)]
    primitive = [Q(0)]
    for value in residual:
        primitive.append(primitive[-1] + h * value / (n + 1))
    prefix = max(map(abs, primitive)) + h * clipping
    lo, hi = sorted((left, right))
    if lo >= 0:
        negative = Q(0)
    elif hi <= 0:
        negative = -h * (lo + hi) / 2
    else:
        negative = h * lo * lo / (2 * (hi - lo))
    if remaining_cells < 1:
        raise ArithmeticError("outer peak budget exhausted")
    peak = peak_range(
        raw,
        a,
        b,
        Q(str(plan["limits"]["peak_enclosure_width_n"])),
        remaining_cells,
        plan["limits"]["peak_bound_depth"],
    )
    return dict(
        j=j_bounds,
        b=b_bounds,
        diss=diss,
        j_motion=j_motion,
        b_motion=b_motion,
        prefix=prefix,
        energy_positive=max(0, defect_upper),
        gap_lower=gap_lower,
        peak=peak[:3],
        cells=peak[3],
        inferred_min=lo,
        negative_impulse=negative,
        middle_velocity_error=abs(y[1] - Q(piece["yb"])) / d,
    )


def combine_step(pieces, start, h, constants, plan):
    width, k, d, M, external = constants
    js, bs, loss = [Q(0), Q(0)], [Q(0), Q(0)], [Q(0), Q(0)]
    delivered, weighted, prefix, energy, negative = Q(0), Q(0), Q(0), Q(0), Q(0)
    inferred, gap = None, None
    peak_low, peak_high, peak_time = Q(0), Q(0), start
    cells = calls = 0
    active_time = Q(0)
    first_active = None
    middle_error = Q(0)
    for piece in pieces:
        local_end = Q(piece["t1"])
        if piece["phase"] == "active":
            active_time += local_end - Q(piece["t0"])
            if first_active is None:
                first_active = Q(piece["t0"])
        row = check_piece(
            piece,
            width,
            k,
            d,
            M,
            external,
            plan,
            plan["limits"]["peak_bound_cells_per_outer_step"] - cells,
        )
        prefix = max(prefix, max(abs(js[0] - delivered), abs(js[1] - delivered)) + row["prefix"])
        delivered += row["j_motion"]
        shift = start + h - local_end
        weighted += row["b_motion"] + shift * row["j_motion"]
        for i in range(2):
            js[i] += row["j"][i]
            bs[i] += row["b"][i] + shift * row["j"][i]
            loss[i] += row["diss"][i]
        energy += row["energy_positive"]
        negative += row["negative_impulse"]
        inferred = row["inferred_min"] if inferred is None else min(inferred, row["inferred_min"])
        gap = row["gap_lower"] if gap is None else min(gap, row["gap_lower"])
        if row["peak"][0] > peak_low:
            peak_low, peak_time = row["peak"][0], row["peak"][2]
        peak_high = max(peak_high, row["peak"][1])
        middle_error = max(middle_error, row["middle_velocity_error"])
        cells += row["cells"]
        calls += piece["evaluations"]
    exact = dict(
        candidate_j_ns=delivered,
        candidate_b_ns2=weighted,
        law_j_low_ns=js[0],
        law_j_high_ns=js[1],
        law_b_low_ns2=bs[0],
        law_b_high_ns2=bs[1],
        law_j_max_difference_ns=max(abs(x - delivered) for x in js),
        law_b_max_difference_over_M_m=max(abs(x - weighted) for x in bs) / M,
        prefix_momentum_bound_ns=prefix,
        positive_energy_defect_bound_j=energy,
        loss_low_j=loss[0],
        loss_high_j=loss[1],
        candidate_gap_lower_m=gap,
        candidate_peak_low_n=peak_low,
        candidate_peak_high_n=peak_high,
        candidate_peak_time_s=peak_time,
        stage_velocity_representation_error_m_s=middle_error,
        inferred_force_min_n=inferred,
        negative_inferred_impulse_ns=negative,
        end_gap_m=pieces[-1]["g1"],
        end_speed_m_s=(Q(pieces[-1]["ye"]) - 1) / d,
    )
    return {
        **{key: float(v) for key, v in exact.items()},
        "pieces": len(pieces),
        "peak_cells": cells,
        "residual_evaluations": calls,
        "candidate_active_s": float(active_time),
        "first_active_time_s": float(first_active) if first_active is not None else None,
    }


def read_reference(summary, material):
    result = {}
    directory = ROOT / summary["raw_run_relative"] / "reference"
    for record in summary["reference_traces"]:
        path = directory / record["fine_csv"]
        if sha(path) != record["fine_sha256"]:
            raise ValueError(f"reference bytes changed: {path.name}")
        with path.open(newline="", encoding="utf-8") as stream:
            frames = [{key: float(x) for key, x in row.items()} for row in csv.DictReader(stream)]
        if len(frames) != record["fine_frames"] or any(
            f["time_s"] != i / 16384 for i, f in enumerate(frames)
        ):
            raise ValueError("reference grid changed")
        bracket = None
        mass = Q(record["mass_kg"])
        if record["case"] == "impact":
            width, k, d = material

            def derivative_sign(f):
                x = max(0.0, float(width) - f["hard_gap_m"])
                return f["closing_speed_m_s"] - float(d * k / (2 * mass)) * x * x

            brackets = [
                (a["time_s"], b["time_s"])
                for a, b in zip(frames, frames[1:])
                if derivative_sign(a) > 0 >= derivative_sign(b)
            ]
            if len(brackets) != 1:
                raise ValueError("unbracketed or ambiguous impact peak")
            bracket = brackets[0]
        result[record["case"], mass] = (frames, bracket, record["analytical_peaks"])
    return result


def reference_step(frames, bracket, analytic, start, h, M, ext, entry, resolution, width):
    first, last = int(start * 16384), int((start + h) * 16384)
    sample = frames[first : last + 1]
    a, b = sample[0], sample[-1]
    end = float(start + h)
    j = b["impulse_ns"] - a["impulse_ns"]
    v0, v1 = a["closing_speed_m_s"], b["closing_speed_m_s"]
    motion = ext * float(h) - M * (v1 - v0)
    moment = ext * float(h) ** 2 / 2 + M * (b["hard_gap_m"] - a["hard_gap_m"] + float(h) * v0)
    points = [(f["time_s"], f["normal_force_n"]) for f in sample]
    if entry is not None and float(start) < entry < end:
        points = sorted([*points, (entry, 0.0)])
    fine_j, fine_b = [], []
    for (ta, fa), (tb, fb) in zip(points, points[1:]):
        dt = tb - ta
        part = dt * (fa + fb) / 2
        fine_j.append(part)
        fine_b.append((end - tb) * part + dt * dt * (2 * fa + fb) / 6)
    interp_j, interp_b = math.fsum(fine_j), math.fsum(fine_b)
    fpeak = max(sample, key=lambda f: f["normal_force_n"])
    peak, time = fpeak["normal_force_n"], fpeak["time_s"]
    peak_source = "fine samples"
    if bracket and float(start) <= bracket[0] and bracket[1] <= end:
        peak = analytic["force_n"]
        time = sum(bracket) / 2
        peak_source = "analytic amplitude, fine time bracket"
    minimum = min(f["hard_gap_m"] for f in sample)
    gap_source = "fine sample"
    if analytic and v0 >= 0 >= v1:
        minimum = width - analytic["compression_m"]
        gap_source = "analytic compression maximum"
    return dict(
        ref_min_gap_m=minimum,
        ref_gap_source=gap_source,
        ref_j_ns=j,
        ref_b_ns2=moment,
        ref_mean_n=j / float(h),
        ref_peak_n=peak,
        ref_peak_time_s=time,
        ref_peak_source=peak_source,
        ref_centroid_s=end - moment / j if j > resolution else None,
        ref_active_s=max(0.0, end - max(float(start), entry)) if entry is not None else float(h),
        ref_end_gap_m=b["hard_gap_m"],
        ref_end_speed_m_s=v1,
        ref_end_force_n=b["normal_force_n"],
        ref_loss_j=b["dissipation_j"] - a["dissipation_j"],
        right_rectangle_j_ns=float(h) * b["normal_force_n"],
        endpoint_trapezoid_j_ns=float(h) * (a["normal_force_n"] + b["normal_force_n"]) / 2,
        uniform_exact_j_gap_error_m=(float(h) * j / 2 - moment) / M,
        ref_momentum_j_error_ns=abs(j - motion),
        fine_force_integral_j_error_ns=abs(interp_j - j),
        fine_force_moment_error_per_h_ns=abs(interp_b - moment) / float(h),
    )


def audit_run(run, group, reference, fixture, material, plan):
    case, mass, h = run["case"], Q(run["m"]), Q(run["h"])
    M = 2 * mass
    ext = mass / 4 if case == "support" else Q(0)
    frames, bracket, analytic = reference
    width, k, d = material
    c = fixture["controls"]
    entry = (
        None
        if case == "support"
        else float((Q(c["sc2"]["gap_initial_m"]) - width) / Q(c["sc2"]["closing_speed_m_s"]))
    )
    gates = plan["gates"]
    rows = []
    initial_g = 0.0 if case == "support" else float(Q(c["sc2"]["gap_initial_m"]))
    initial_y = 1.0 if case == "support" else float(1 + d * Q(c["sc2"]["closing_speed_m_s"]))
    prev_g, prev_y = initial_g, initial_y
    for i in range(int(Q(c["window_s"]) / h)):
        start = i * h
        ref = reference_step(
            frames,
            bracket,
            analytic,
            start,
            h,
            float(M),
            float(ext),
            entry,
            gates["step_impulse_vs_reference_ns"],
            float(width),
        )
        row = dict(
            case=case,
            mass_kg=str(mass),
            h_s=str(h),
            step=i,
            start_s=float(start),
            end_s=float(start + h),
            **ref,
        )
        if (
            ref["ref_momentum_j_error_ns"] > 1e-10
            or ref["fine_force_integral_j_error_ns"] > gates["step_impulse_vs_reference_ns"]
            or ref["fine_force_moment_error_per_h_ns"] > gates["step_impulse_vs_reference_ns"]
        ):
            raise ValueError(f"inconsistent reference integrals at {case}, {mass}, {h}, {i}")
        pieces = group.get(i, [])
        if not run["passed"] and i >= run["failed_step"]:
            row["candidate_state"] = "unfinished"
            rows.append(row)
            continue
        t = start
        for p in pieces:
            if Q(p["t0"]) != t or Q(p["t1"]) <= t or p["g0"] != prev_g or p["y0"] != prev_y:
                raise ValueError("discontinuous candidate path")
            t, prev_g, prev_y = Q(p["t1"]), p["g1"], p["ye"]
        if t != start + h:
            raise ValueError("missing candidate time coverage")
        try:
            result = combine_step(pieces, start, h, (Q(float(width)), k, d, M, ext), plan)
        except ArithmeticError as error:
            row["candidate_state"] = str(error)
        else:
            row["candidate_state"] = "observed"
            row.update(result)
            row["j_error_ns"] = abs(row["candidate_j_ns"] - ref["ref_j_ns"])
            row["b_error_over_M_m"] = abs(row["candidate_b_ns2"] - ref["ref_b_ns2"]) / float(M)
            row["peak_error_n"] = max(
                abs(row[key] - ref["ref_peak_n"])
                for key in ("candidate_peak_low_n", "candidate_peak_high_n")
            )
            row["gap_error_m"] = abs(row["end_gap_m"] - ref["ref_end_gap_m"])
            row["speed_error_m_s"] = abs(row["end_speed_m_s"] - ref["ref_end_speed_m_s"])
        rows.append(row)
    if not run["passed"] or any(row["candidate_state"] != "observed" for row in rows):
        return rows, dict(
            case=case,
            mass_kg=str(mass),
            h_s=str(h),
            passed=False,
            reason=run.get("reason")
            or next(r["candidate_state"] for r in rows if r["candidate_state"] != "observed"),
        )
    pairs = {
        "j_error_ns": "step_impulse_vs_reference_ns",
        "b_error_over_M_m": "step_moment_error_divided_by_mass_m",
        "peak_error_n": "step_peak_vs_reference_n",
        "gap_error_m": "endpoint_gap_error_m",
        "speed_error_m_s": "endpoint_speed_error_m_s",
        "law_j_max_difference_ns": "law_impulse_vs_delivered_ns",
        "law_b_max_difference_over_M_m": "law_moment_error_divided_by_mass_m",
        "prefix_momentum_bound_ns": "prefix_momentum_defect_ns",
    }
    maxima = {key: max(r[key] for r in rows) for key in pairs}
    checks = {key: maxima[key] <= gates[limit] for key, limit in pairs.items()}
    positive = sum(r["positive_energy_defect_bound_j"] for r in rows)
    window_error = abs(math.fsum(r["candidate_j_ns"] - r["ref_j_ns"] for r in rows))
    checks["window_work"] = positive <= gates["positive_work_defect_per_window_j"]
    checks["window_impulse"] = window_error <= gates["window_impulse_vs_reference_ns"]
    checks["core"] = min(r["candidate_gap_lower_m"] for r in rows) >= 0
    checks["nonnegative_step_impulse"] = all(r["candidate_j_ns"] >= 0 for r in rows)
    counts = [r["residual_evaluations"] for r in rows]
    if (
        sum(counts) != run["total_evaluations"]
        or max(counts) != run["max_step_evaluations"]
        or len(group) != len(rows)
    ):
        raise ValueError("candidate resource or step receipt mismatch")
    checks["budget"] = max(counts) <= plan["limits"]["residual_evaluations_per_outer_step"]
    checks["entry_and_active_duration"] = (
        max(abs(r["candidate_active_s"] - r["ref_active_s"]) for r in rows)
        <= gates["impact_peak_time_error_s"]
    )
    peakrow = max(rows, key=lambda r: r["candidate_peak_low_n"])
    timing = None
    if bracket:
        timing = max(
            0.0,
            bracket[0] - peakrow["candidate_peak_time_s"],
            peakrow["candidate_peak_time_s"] - bracket[1],
        )
        checks["impact_peak_time"] = timing <= gates["impact_peak_time_error_s"]
    return rows, dict(
        case=case,
        mass_kg=str(mass),
        h_s=str(h),
        passed=all(checks.values()),
        checks=checks,
        worst=maxima,
        worst_steps={k: max(rows, key=lambda r: r[k])["step"] for k in pairs},
        window_impulse_error_ns=window_error,
        positive_window_work_bound_j=positive,
        peak_time_distance_from_reference_bracket_s=timing,
        ref_peak_bracket_s=bracket,
        min_inferred_force_n=min(r["inferred_force_min_n"] for r in rows),
        negative_inferred_impulse_ns=math.fsum(r["negative_inferred_impulse_ns"] for r in rows),
        strict_inferred_sign=all(r["inferred_force_min_n"] >= 0 for r in rows),
        pieces=sum(r["pieces"] for r in rows),
        max_residual_evaluations=max(r["residual_evaluations"] for r in rows),
        max_peak_cells=max(r["peak_cells"] for r in rows),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if sha(PLAN, normalize=True) != PLAN_SHA or sha(INPUT, normalize=True) != INPUT_SHA:
        raise ValueError("registered input or protocol changed")
    fixture, plan = json.loads(INPUT.read_bytes()), json.loads(PLAN.read_bytes())
    prior = json.loads((INPUT.parent / plan["prior_summary"]).read_bytes())
    material = tuple(Q(prior["nominal"][key]) for key in ("delta", "k", "d"))
    refs = read_reference(prior, material)
    args.output_dir.mkdir(exist_ok=False)
    producer = json.loads((args.candidate_dir / "producer.json").read_bytes())
    file = args.candidate_dir / "pieces.jsonl"
    if (
        producer["protocol_sha256_lf"] != PLAN_SHA
        or sha(file) != producer["pieces_sha256"]
        or (
            sha(Path(__file__).with_name("probe_p65_layer_moments.py"), normalize=True)
            != producer["source_sha256_lf"]
        )
    ):
        raise ValueError("unbound candidate source, inputs or output")
    groups = defaultdict(lambda: defaultdict(list))
    for line in file.read_text(encoding="utf8").splitlines():
        p = json.loads(line)
        groups[p["case"], Q(p["m"]), Q(p["h"])][p["step"]].append(p)
    expected = [
        (case, Q(m), Q(h))
        for m in fixture["controls"]["trace_masses_kg"]
        for case in ("support", "impact")
        for h in fixture["controls"]["report_h_s"]
    ]
    keys = [(r["case"], Q(r["m"]), Q(r["h"])) for r in producer["runs"]]
    if keys != expected or set(groups) - set(expected):
        raise ValueError("unexpected case membership or order")
    steps, runs = [], []
    for run, key in zip(producer["runs"], keys, strict=True):
        rows, summary = audit_run(run, groups[key], refs[key[:2]], fixture, material, plan)
        steps.extend(rows)
        runs.append(summary)
    out = args.output_dir / "steps.csv"
    with out.open("x", newline="", encoding="utf8") as stream:
        fields = list(dict.fromkeys(key for r in steps for key in r))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(steps)
    ok = all(r["passed"] for r in runs)
    result = dict(
        schema="p65-scalar-intrastep-audit-v1",
        observable_gates_passed=ok,
        strict_inferred_sign_all=all(r.get("strict_inferred_sign", False) for r in runs),
        scope=plan["scope"],
        protocol_sha256_lf=PLAN_SHA,
        reference_fixture_sha256_lf=INPUT_SHA,
        producer_sha256_lf=producer["source_sha256_lf"],
        piece_file_sha256=producer["pieces_sha256"],
        audit_sha256_lf=sha(Path(__file__), normalize=True),
        step_rows=len(steps),
        step_csv_sha256=sha(out),
        runs=runs,
    )
    (args.output_dir / "audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    for r in runs:
        print(
            r["case"],
            r["mass_kg"],
            r["h_s"],
            "PASS"
            if r["passed"]
            else r.get("reason") or [k for k, v in r["checks"].items() if not v],
        )
    print("Pointwise inertial sign:", result["strict_inferred_sign_all"], "Rows:", len(steps))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
