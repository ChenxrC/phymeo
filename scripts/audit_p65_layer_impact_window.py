"""Bind the accepted impact prefix to six new intervals and audit the full SC-L1 path."""

import argparse
import csv
import json
import math
from fractions import Fraction as Q
from pathlib import Path

from audit_p65_layer_entry import kinematics, material_piece, peak_boxes, restrict
from audit_p65_layer_joint_publication import need, parse_state, reject, state_moments
from verify_p65_contact_layer_screen import INPUT, INPUT_SHA, sha

PLAN = INPUT.with_name("p65-contact-layer-impact-window-protocol.json")
PLAN_SHA = "efb8347aae20d4bccc8207251794f8f3ee50ae88ccc168e3b2bbed809e1c176d"


def rounded_interval(value):
    return Q(math.nextafter(value, -math.inf)), Q(math.nextafter(value, math.inf))


def same_seed(expected, actual):
    need(expected == actual, "wrong trajectory predecessor")


def continuity_files(folder, seed, hashes):
    expected = {"states.jsonl", "forces.jsonl", "pieces.jsonl", "end.state.json"}
    need(set(hashes) == expected, "trace inventory")
    for file, digest in hashes.items():
        need(sha(folder / file) == digest, "bound trace bytes")
    raw = (folder / "states.jsonl").read_bytes().splitlines(keepends=True)
    same_seed(seed, raw[0])
    same_seed((folder / "end.state.json").read_bytes(), raw[-1])
    return raw


def stepsize_fixture(fixture, case):
    ctl = fixture["controls"]
    m, h, T = map(Q, (case["m_kg"], case["h_s"], case["T_s"]))
    need(
        T / h == 8
        and T == Q(ctl["window_s"])
        and str(h) in ctl["report_h_s"]
        and str(m) in ctl["trace_masses_kg"]
        and Q(ctl["sc2"]["compressive_load_n"]) == 0
        and Q(ctl["sc2"]["gap_initial_m"]) == Q(case["g0_m"])
        and Q(ctl["sc2"]["closing_speed_m_s"]) == Q(case["closing_speed_m_s"]),
        "original controls",
    )
    return 2 * m, h, T


def read_force(m):
    p = [Q(float.fromhex(w)) for w in m["words"]]
    need(
        len(p) == 8
        and min(p) > 0
        and all(float(z).hex() == w for z, w in zip(p, m["words"], strict=True)),
        "positive force words",
    )
    return p


def tail_interval(folder, seed, origin, step, h, M, mat, rate_max, cfg, limits, gates, ref, plan):
    K, width, d = mat
    idx = json.loads((folder / "index.json").read_bytes())
    need(
        idx["step"] == step
        and idx["origin"] == origin
        and idx["event"] is None
        and idx["protocol_sha256_lf"] == PLAN_SHA
        and idx["kernel_sha256_lf"] == plan["kernel_sha256_lf"]
        and idx["driver_sha256_lf"]
        == sha(Path(__file__).with_name("probe_p65_layer_impact_window.py"), normalize=True),
        "continuation binding",
    )
    raw = continuity_files(folder, seed, idx["hashes"])
    decoded = [parse_state(frame, cfg) for frame in raw]
    states = [s for s, _ in decoded]
    n = math.ceil(
        8 * float(h) * max(float(K * d * width / M), math.sqrt(float(K * (1 + d * rate_max) / M)))
    )
    sizes = [2] * (n // 2) + ([1] if n % 2 else [])
    rows = [json.loads(b) for b in (folder / "pieces.jsonl").read_bytes().splitlines()]
    macros = [json.loads(b) for b in (folder / "forces.jsonl").read_bytes().splitlines()]
    need(
        len(rows) == n == idx["parts"]
        and len(macros) == len(sizes) == idx["macros"]
        and len(raw) == n + 1
        and states[0][0] == step * h
        and states[-1][0] == (step + 1) * h,
        "gap-free coverage",
    )
    J = B = dj = db = prefix = positive = loss = relative = Q(0)
    low_gap = high_gap = states[0][1]
    min_rate = states[0][2]
    boxes = []
    budget = [0]
    row = 0
    data = []
    last = None
    for mi, (m, size) in enumerate(zip(macros, sizes, strict=True)):
        need(
            m["phase"] == "active"
            and m["parts"] == size
            and Q(m["dt"]) == size * h / n
            and Q(m["t"]) == states[row][0],
            "macro span",
        )
        parent = read_force(m)
        for side in range(size):
            s, e = states[row : row + 2]
            r = rows[row]
            duration = e[0] - s[0]
            need(duration == h / n and (r["macro"], r["side"]) == (mi, side), "piece label or time")
            p = restrict(parent, size, side)
            g, v = kinematics(s, e, p, M, d)
            y = [1 + d * u for u in v]
            need(min(g) > 0 and max(g) < width and max(v) < 0 and min(y) > 0, "opening active tail")
            j, b = state_moments(s, e, M, d)
            need(
                j == Q(r["j"]) and b == Q(r["b"]) and 0 < b < duration * j,
                "exact piece moment cone",
            )
            for amount in (j, b):
                need(
                    max(amount.numerator.bit_length(), amount.denominator.bit_length())
                    <= cfg["bits_per_exported_integer"],
                    "moment width",
                )
            local_j, local_b, pre, pos, rel, diss = material_piece(
                g, v, p, K, width, d, M, duration
            )
            shift = (step + 1) * h - e[0]
            J += j
            B += b + shift * j
            prefix = max(prefix, abs(dj) + pre)
            dj += local_j
            db += local_b + shift * local_j
            positive += pos
            loss += diss
            relative = max(relative, rel)
            low_gap = min(low_gap, *g)
            high_gap = max(high_gap, *g)
            min_rate = min(min_rate, *y)
            boxes.extend(peak_boxes(p, s[0], e[0], budget, limits))
            data.append((s[0], p))
            last = (s, e, p)
            row += 1
    need((J, B) == state_moments(states[0], states[-1], M, d) and 0 < B < h * J, "outer readback")
    counts = idx["work"]
    expected = dict(
        force_charges=8 * len(sizes),
        convolution_terms=36 * len(sizes),
        conversion_terms=36 * len(sizes),
        half_subdivision_means=56 * (n // 2),
        method="series7",
    )
    need(
        counts == expected and counts["force_charges"] <= cfg["force_evaluations_per_outer_step"],
        "execution budget",
    )
    plo = max(z[0] for z in boxes)
    phi = max(z[1] for z in boxes)
    need(phi - plo <= Q(str(limits["peak_enclosure_width_n"])), "peak enclosure")
    possible = [z for z in boxes if z[1] >= plo]
    ta = min(z[2] for z in possible)
    tb = max(z[3] for z in possible)
    rp = Q(ref["ref_peak_n"])
    rt = Q(ref["ref_peak_time_s"])
    errors = dict(
        step_peak_vs_reference_n=max(abs(plo - rp), abs(phi - rp)),
        step_impulse_vs_reference_ns=abs(J - Q(ref["ref_j_ns"])),
        step_moment_error_divided_by_mass_m=abs(B - Q(ref["ref_b_ns2"])) / M,
        endpoint_gap_error_m=abs(states[-1][1] - Q(ref["ref_end_gap_m"])),
        endpoint_speed_error_m_s=abs((states[-1][2] - 1) / d - Q(ref["ref_end_speed_m_s"])),
        impact_peak_time_error_s=max(abs(ta - rt), abs(tb - rt)),
        law_impulse_vs_delivered_ns=abs(dj),
        law_moment_error_divided_by_mass_m=abs(db) / M,
        prefix_momentum_defect_ns=prefix,
        positive_work_defect_per_window_j=positive,
    )
    failed = [key for key, val in errors.items() if val > gates[key]]
    if relative > Q(1, 100):
        failed.append("relative_constitutive_path")
    result = dict(
        step=step,
        origin=origin,
        seed_sha256=idx["seed_sha256"],
        end_sha256=idx["hashes"]["end.state.json"],
        parts=n,
        work=counts,
        peak_cells=budget[0],
        max_wire_bytes=max(map(len, raw)),
        max_remainder_bits=max(b for _, b in decoded),
        peak_n=list(map(float, (plo, phi))),
        peak_time_s=list(map(float, (ta, tb))),
        end_force_n=float(last[2][-1]),
        end_gap_m=float(states[-1][1]),
        end_closing_speed_m_s=float((states[-1][2] - 1) / d),
        min_gap_lower_m=float(low_gap),
        max_gap_upper_m=float(high_gap),
        min_rate_factor=float(min_rate),
        exact_j_ns=str(J),
        exact_b_ns2=str(B),
        j_ns=float(J),
        b_ns2=float(B),
        loss_j=float(loss),
        positive_only_j=float(positive),
        law_relative_bound=float(relative),
        errors={key: float(val) for key, val in errors.items()},
        failed_gates=failed,
        trace_hashes=idx["hashes"],
    )
    return result, dict(
        J=J,
        B=B,
        dj=dj,
        db=db,
        prefix=prefix,
        positive=positive,
        loss=loss,
        refj=errors["step_impulse_vs_reference_ns"],
        states=(states[0], states[-1]),
        end_force=last[2][-1],
        peak_hi=phi,
        first=(states[0], states[1], data[0][1]),
        last=last,
        forces=data,
    )


def locate_turn(s, e, p, M, d, limits, used):
    g, v = kinematics(s, e, p, M, d)
    need(v[0] >= 0 >= v[-1] and min(p) > 0, "compression reversal bracket")
    a, b = s[0], e[0]
    cells = 1
    depth = 0
    if v[0] == 0 or v[-1] == 0:
        i = 0 if v[0] == 0 else -1
        return ((a, a) if i == 0 else (b, b)), (g[i], g[i]), cells
    while b - a > Q(1, 2**30):
        need(
            depth < limits["peak_bound_depth"]
            and used + cells + 2 <= limits["peak_bound_cells_per_outer_step"],
            "reversal search budget",
        )
        left = restrict(v, 2, 0)
        mid = (a + b) / 2
        side = int(left[-1] > 0)
        cells += 2
        depth += 1
        if left[-1] == 0:
            point = restrict(g, 2, 0)[-1]
            return (mid, mid), (point, point), cells
        a, b = (mid, b) if side else (a, mid)
        v = restrict(v, 2, 1) if side else left
        g = restrict(g, 2, side)
    return (a, b), (min(g), max(g)), cells


def prefix_phases(path, row, M, d, cfg, limits):
    # Only the new reversal and post-reversal pressure check reads this saved interval.
    for file in ("states.jsonl", "forces.jsonl", "pieces.jsonl"):
        need(sha(path / file) == row["trace_hashes"][file], "reversal input binding")
    states = [
        parse_state(b, cfg)[0]
        for b in (path / "states.jsonl").read_bytes().splitlines(keepends=True)
    ]
    macros = [json.loads(b) for b in (path / "forces.jsonl").read_bytes().splitlines()]
    pieces = [json.loads(b) for b in (path / "pieces.jsonl").read_bytes().splitlines()]
    crossing = [i for i, (s, e) in enumerate(zip(states, states[1:])) if s[2] > 1 >= e[2]]
    need(len(crossing) == 1 and states[-1][2] < 1, "unique observed reversal interval")
    start = crossing[0]
    post = []
    for i in range(start, len(pieces)):
        r = pieces[i]
        m = macros[r["macro"]]
        p = restrict(read_force(m), m["parts"], r["side"])
        post.append((states[i][0], p))
    time, gap, cells = locate_turn(
        states[start], states[start + 1], post[0][1], M, d, limits, row["peak_cells"]
    )
    return time, gap, cells, post


def log_unit(x):
    # For 1 <= x <= 2 use 16 positive terms of 2*atanh((x-1)/(x+1)).
    z = (x - 1) / (x + 1)
    power = z
    lower = Q(0)
    for k in range(16):
        lower += 2 * power / (2 * k + 1)
        power *= z * z
    return lower, lower + 2 * power / (33 * (1 - z * z))


def log_ratio(r):
    need(r >= 1, "log ratio domain")
    n = r.numerator.bit_length() - r.denominator.bit_length()
    if r < 2**n:
        n -= 1
    scale = 2**40
    mant = r / 2**n
    a = (mant * scale).__floor__()
    b = (mant * scale).__ceil__()
    l2, u2 = log_unit(Q(2))
    return n * l2 + log_unit(Q(a, scale))[0], n * u2 + log_unit(Q(b, scale))[1]


def terminal_oracle(state, y_in, M, K, width, d):
    _, gap, y = state
    x = width - gap

    def invariant(t):
        a, b = log_ratio(y_in / t)
        shift = t - y_in + d * d * K * x * x / (2 * M)
        return a + shift, b + shift

    lower, upper = Q(99, 100) * y, Q(101, 100) * y
    need(
        0 < lower < upper < 1 and invariant(lower)[0] > 0 and invariant(upper)[1] < 0,
        "analytic unloading branch root",
    )
    residual = invariant(y)
    bound = max(map(abs, residual)) / (1 - upper)
    need(bound < 1, "terminal log error")
    return dict(
        rate_factor=float(y),
        ideal_rate_at_same_gap_bracket=list(map(float, (lower, upper))),
        phase_residual=list(map(float, residual)),
        relative_rate_error_bound=float(bound / (1 - bound)),
        material_force_n=float(K * x * y),
    )


def fine_reversal(trace, fine, turn, gap, width, limit):
    path = INPUT.parents[2] / trace["raw_run_relative"] / "reference" / fine["fine_csv"]
    need(sha(path) == fine["fine_sha256"], "fine comparison input")
    with path.open(newline="", encoding="utf8") as file:
        frames = list(csv.DictReader(file))
    brackets = [
        (Q(a["time_s"]), Q(b["time_s"]))
        for a, b in zip(frames, frames[1:])
        if Q(a["closing_speed_m_s"]) > 0 >= Q(b["closing_speed_m_s"])
    ]
    ideal_gap = width - Q(str(fine["analytical_peaks"]["compression_m"]))
    need(
        len(brackets) == 1 and brackets[0][0] <= turn[0] <= turn[1] <= brackets[0][1],
        "turn vs fine time bracket",
    )
    need(max(abs(g - ideal_gap) for g in gap) <= limit, "minimum gap vs analytic compression")
    return list(map(float, brackets[0])), float(ideal_gap)


def phase_shape(forces):
    increases = []
    unproved = 0
    prev = None
    total = largest = Q(0)
    for t, p in forces:
        if any(a < b for a, b in zip(p, p[1:])):
            unproved += 1
        if prev is not None and p[0] > prev:
            jump = p[0] - prev
            total += jump
            largest = max(largest, jump)
            increases.append(t)
        prev = p[-1]
    return dict(
        piece_count=len(forces),
        intrapiece_decrease_unproved=unproved,
        upward_knot_count=len(increases),
        total_upward_jump_n=float(total),
        max_upward_jump_n=float(largest),
        first_upward_knot_s=float(increases[0]) if increases else None,
        published_force_nonincreasing_certified=(unproved == 0 and not increases),
    )


def faults(last, seed7, mat, M, B, unshifted):
    cases = json.loads(INPUT.with_name("p65-contact-layer-publication-results.json").read_bytes())[
        "cases"
    ]
    isolated = next(c["witness"]["inlet"] for c in cases if c["id"] == "h64-step7")
    other = (json.dumps(isolated, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    need(json.loads(seed7)["t"] == isolated["t"], "uninformative splice time")
    s, e, p = last
    return dict(
        independently_seeded_same_time=reject(
            "splice", "wrong trajectory predecessor", lambda: same_seed(seed7, other)
        ),
        last_force_zeroed=reject(
            "tail cutoff", "velocity readback", lambda: kinematics(s, e, [Q(0)] * len(p), M, mat[2])
        ),
        false_release=reject(
            "release",
            "still inside contact layer",
            lambda: need(e[1] >= mat[1], "still inside contact layer"),
        ),
        unshifted_time_moments=reject(
            "lever arm",
            "wrong window time moment",
            lambda: need(unshifted == B, "wrong window time moment"),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    need(
        sha(PLAN, normalize=True) == PLAN_SHA and sha(INPUT, normalize=True) == INPUT_SHA,
        "registered inputs",
    )
    plan = json.loads(PLAN.read_bytes())
    fixture = json.loads(INPUT.read_bytes())
    M, h, T = stepsize_fixture(fixture, plan["case"])
    pfile = INPUT.with_name(plan["prefix_result"])
    need(sha(pfile, normalize=True) == plan["prefix_result_sha256_lf"], "prefix report")
    pref = json.loads(pfile.read_bytes())
    archive = INPUT.parents[2] / pref["raw_directory"] / "series7"
    prior = pref["series7"]
    need(
        prior["passed"]
        and prior["structural_pass"]
        and len(prior["intervals"]) == 2
        and all(
            not r["failed_gates"] and r["exact_inertial_path"] and r["strict_nonnegative"]
            for r in prior["intervals"]
        ),
        "accepted prefix",
    )
    plan0 = INPUT.with_name(plan["gates_from"])
    refindex = json.loads(INPUT.with_name(plan["reference_index"]).read_bytes())
    need(sha(plan0, normalize=True) == refindex["protocol_sha256_lf"], "inherited budget")
    protocol = json.loads(plan0.read_bytes())
    limits = protocol["limits"]
    gates = {k: Q(str(v)) for k, v in protocol["gates"].items()}
    wire = json.loads(INPUT.with_name(plan["wire_from"]).read_bytes())["wire_limits"]
    trace = json.loads(INPUT.with_name(plan["reference_fine"]).read_bytes())
    fine = next(
        r for r in trace["reference_traces"] if r["case"] == "impact" and r["mass_kg"] == "1/2"
    )
    mat = tuple(Q(trace["nominal"][k]) for k in ("k", "delta", "d"))
    K, width, d = mat
    table = INPUT.with_name(refindex["step_table"]["file"])
    need(sha(table, normalize=True) == refindex["step_table"]["sha256_lf"], "reference identity")
    with table.open(newline="", encoding="utf8") as file:
        refs = [
            r
            for r in csv.DictReader(file)
            if r["case"] == "impact" and r["mass_kg"] == "1/2" and r["h_s"] == "1/64"
        ]
    need([int(r["step"]) for r in refs] == list(range(8)), "full reference membership")
    with (archive / "entry" / "states.jsonl").open("rb") as file:
        start = parse_state(file.readline(wire["bytes_per_motion_packet"] + 1), wire)[0]
    u = Q(plan["case"]["closing_speed_m_s"])
    need(start == (0, Q(plan["case"]["g0_m"]), 1 + d * u), "original outside inlet")
    first_end = (archive / "entry" / "end.state.json").read_bytes()
    with (archive / "successor" / "states.jsonl").open("rb") as file:
        same_seed(first_end, file.readline(wire["bytes_per_motion_packet"] + 1))
    seed = (archive / "successor" / "end.state.json").read_bytes()
    need(
        sha(archive / "successor" / "end.state.json")
        == plan["resume_sha256"]
        == prior["intervals"][1]["end_sha256"]
        and sha(archive / "entry" / "end.state.json") == prior["intervals"][0]["end_sha256"],
        "accepted endpoints",
    )
    source = plan["resume_sha256"]
    entry = (start[1] - width) / u
    need(Q(prior["intervals"][0]["entry_s"]) == entry, "analytic ballistic entry")
    turn, gap, cells, post = prefix_phases(
        archive / "successor", prior["intervals"][1], M, d, wire, limits
    )
    ft, ideal_gap = fine_reversal(trace, fine, turn, gap, width, gates["endpoint_gap_error_m"])
    # Exact prefix moments. One ulp on each side bounds any rounded signed receipt metric.
    J = B = unshifted = ref_err = hist_lo = hist_hi = mom_lo = mom_hi = Q(0)
    loss_lo = loss_hi = positive = pre = Q(0)
    for n, r in enumerate(prior["intervals"]):
        j, b = Q(r["exact_j_ns"]), Q(r["exact_b_ns2"])
        shift = T - (n + 1) * h
        J += j
        B += b + shift * j
        unshifted += b
        ref_err += abs(j - Q(refs[n]["ref_j_ns"]))
        pre = max(
            pre,
            max(abs(hist_lo), abs(hist_hi))
            + rounded_interval(r["errors"]["prefix_momentum_defect_ns"])[1],
        )
        jl, jh = rounded_interval(r["signed_law_j_ns"])
        bl, bh = rounded_interval(r["signed_law_b_ns2"])
        hist_lo += jl
        hist_hi += jh
        mom_lo += bl + shift * jl
        mom_hi += bh + shift * jh
        a, b = rounded_interval(r["dissipation_j"])
        loss_lo += a
        loss_hi += b
        positive += rounded_interval(r["positive_work_j"])[1]
    pk = prior["intervals"][0]
    peak_lo = rounded_interval(pk["peak_n"][0])[0]
    peak_hi = rounded_interval(pk["peak_n"][1])[1]
    peak_time = (
        rounded_interval(pk["peak_time_s"][0])[0],
        rounded_interval(pk["peak_time_s"][1])[1],
    )
    need(
        rounded_interval(prior["intervals"][1]["peak_n"][1])[1] < peak_lo
        and peak_time[1] < turn[0],
        "prefix phase order",
    )
    deadline = Q(trace["continuous_certificate"]["impact_half_initial_momentum_delivered_by_s"])
    need(h <= deadline and Q(pk["exact_j_ns"]) >= M * u / 2, "incident momentum delivery")
    details = []
    tail = []
    for n in range(2, 8):
        origin = (
            pref["raw_directory"] + "/series7/successor/end.state.json"
            if n == 2
            else f"step-{n - 1}/end.state.json"
        )
        folder = args.root / f"step-{n}"
        row, x = tail_interval(
            folder,
            seed,
            origin,
            n,
            h,
            M,
            mat,
            Q(fixture["rate_range_m_s"][1]),
            wire,
            limits,
            gates,
            refs[n],
            plan,
        )
        need(
            row["seed_sha256"] == source and x["peak_hi"] < peak_lo,
            "seed hash or later global peak",
        )
        source = row["end_sha256"]
        seed = (folder / "end.state.json").read_bytes()
        shift = T - (n + 1) * h
        J += x["J"]
        B += x["B"] + shift * x["J"]
        unshifted += x["B"]
        ref_err += x["refj"]
        pre = max(pre, max(abs(hist_lo), abs(hist_hi)) + x["prefix"])
        hist_lo += x["dj"]
        hist_hi += x["dj"]
        mom_lo += x["db"] + shift * x["dj"]
        mom_hi += x["db"] + shift * x["dj"]
        loss_lo += x["loss"]
        loss_hi += x["loss"]
        positive += x["positive"]
        post.extend(x["forces"])
        details.append(row)
        tail.append(x)
    end = tail[-1]["states"][1]
    endv = (end[2] - 1) / d
    need(
        end[0] == T and state_moments(start, end, M, d) == (J, B) and 0 < B < T * J,
        "full window moment identities",
    )
    Et = M * endv**2 / 2 + K * (width - end[1]) ** 2 / 2
    E0 = M * u * u / 2
    deficit = (Et - E0 + loss_lo, Et - E0 + loss_hi)
    oracle = terminal_oracle(end, start[2], M, K, width, d)
    controls = faults(
        tail[-1]["last"],
        (args.root / "step-6" / "end.state.json").read_bytes(),
        mat,
        M,
        B,
        unshifted,
    )
    checks = dict(
        window_reference_impulse=ref_err <= gates["window_impulse_vs_reference_ns"],
        window_positive_work=positive <= gates["positive_work_defect_per_window_j"],
        window_prefix_momentum=pre <= gates["prefix_momentum_defect_ns"],
        window_law_impulse=max(abs(hist_lo), abs(hist_hi)) <= gates["law_impulse_vs_delivered_ns"],
        window_law_time_moment=max(abs(mom_lo), abs(mom_hi)) / M
        <= gates["law_moment_error_divided_by_mass_m"],
    )
    summary = dict(
        time_s=str(T),
        entry_s=str(entry),
        receipt_prefix_steps=2,
        new_steps=6,
        cold_links=7,
        total_pieces=sum(r["parts"] for r in prior["intervals"] + details),
        peak_n=list(map(float, (peak_lo, peak_hi))),
        peak_time_s=list(map(float, peak_time)),
        peak_time_reference_s=pk["ref_time_s"],
        rebound_time_s=list(map(float, turn)),
        independent_turn_bracket_s=ft,
        global_min_gap_m=list(map(float, gap)),
        analytic_ref_min_gap_m=ideal_gap,
        max_compression_m=[float(width - gap[1]), float(width - gap[0])],
        turn_search_cells=cells,
        delivered_half_incident_momentum_by_s=float(h),
        final_gap_m=float(end[1]),
        final_compression_m=float(width - end[1]),
        final_closing_speed_m_s=float(endv),
        final_delivered_force_n=float(tail[-1]["end_force"]),
        exited_layer=False,
        post_turn_pressure=phase_shape(post),
        terminal_oracle=oracle,
        external_work_j=0,
        initial_kinetic_j=float(E0),
        final_kinetic_j=float(M * endv**2 / 2),
        final_elastic_j=float(K * (width - end[1]) ** 2 / 2),
        dissipation_j=list(map(float, (loss_lo, loss_hi))),
        signed_energy_defect_j=list(map(float, deficit)),
        sum_positive_piece_defects_upper_j=float(positive),
        sum_abs_reference_impulse_errors_ns=float(ref_err),
        prefix_momentum_bound_ns=float(pre),
        signed_window_law_impulse_error_ns=list(map(float, (hist_lo, hist_hi))),
        signed_window_law_time_moment_error_ns2=list(map(float, (mom_lo, mom_hi))),
        exact_contact_impulse_ns=str(J),
        contact_impulse_ns=float(J),
        exact_contact_time_moment_ns2=str(B),
        contact_time_moment_ns2=float(B),
        unshifted_time_moment_error_ns2=float(unshifted - B),
        reference_final=fine["final"],
    )
    passed = all(checks.values()) and all(not r["failed_gates"] for r in details)
    result = dict(
        schema="sc-l1-continuous-impact-audit-v1",
        passed=passed,
        checks=checks,
        summary=summary,
        negative_controls=controls,
        prefix_sha256_lf=plan["prefix_result_sha256_lf"],
        prefix_steps=prior["intervals"],
        new_steps=details,
        protocol_sha256_lf=PLAN_SHA,
        source_sha256_lf=sha(Path(__file__), normalize=True),
    )
    with (args.root / "audit.json").open("x", encoding="utf8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(
        "impact window",
        passed,
        "entry",
        entry,
        "peak",
        summary["peak_n"],
        "turn",
        summary["rebound_time_s"],
        "end force",
        summary["final_delivered_force_n"],
        "J",
        float(J),
        "defects+",
        float(positive),
    )
    print("failed", [(r["step"], r["failed_gates"]) for r in details if r["failed_gates"]], checks)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
