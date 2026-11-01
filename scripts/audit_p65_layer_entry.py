"""Audit delivered scalar impact pressure, its exact entry and its continuous maximum."""

import argparse
import csv
import json
import math
from fractions import Fraction as Q
from math import comb
from pathlib import Path

from audit_p65_layer_intrastep import product
from audit_p65_layer_joint_publication import need, parse_state, reject, state_moments
from verify_p65_contact_layer_screen import INPUT, INPUT_SHA, sha

PLAN = INPUT.with_name("p65-contact-layer-entry-protocol.json")
PLAN_SHA = "9fc803c5468982c580dac55bb567dd755bf16132242208bb967b5d2221052067"


def restrict(controls, parts, side):
    if parts == 1:
        return controls
    source = controls if side == 0 else controls[::-1]
    half = [sum(comb(i, k) * source[k] for k in range(i + 1)) / 2**i for i in range(len(source))]
    return half if side == 0 else half[::-1]


def peak_boxes(p, a, b, budget, limits):
    """Enclose all maximizers, including one-sided values at force publication knots."""
    stack = [(p, a, b, 0)]
    boxes = []
    while stack:
        c, t0, t1, dep = stack.pop()
        budget[0] += 1
        need(budget[0] <= limits["peak_bound_cells_per_outer_step"], "peak cell budget")
        if min(c) == max(c):
            boxes.append((c[0], c[0], t0, t1))
        elif all(x <= y for x, y in zip(c, c[1:])):
            boxes.append((c[-1], c[-1], t1, t1))
        elif all(x >= y for x, y in zip(c, c[1:])):
            boxes.append((c[0], c[0], t0, t0))
        elif max(c) - max(c[0], c[-1]) <= Q(str(limits["peak_enclosure_width_n"])) and t1 - t0 <= Q(
            1, 131072
        ):
            boxes.append((max(c[0], c[-1]), max(c), t0, t1))
        else:
            need(dep < limits["peak_bound_depth"], "peak depth")
            mid = (t0 + t1) / 2
            stack.extend(
                [(restrict(c, 2, 1), mid, t1, dep + 1), (restrict(c, 2, 0), t0, mid, dep + 1)]
            )
    return boxes


def free_path(s, e, p, event, width, closing, d):
    need(all(z == 0 for z in p), "free load")
    need(e[0] == event and e[0] > s[0], "event time")
    need(e[1] == width and s[1] > width, "event gap")
    need(s[2] == e[2] == 1 + d * closing and s[1] - (e[0] - s[0]) * closing == e[1], "free flight")


def kinematics(s, e, p, M, d):
    h = e[0] - s[0]
    n = len(p)
    need(h > 0 and min(p) >= 0, "time or pressure sign")
    v = [(s[2] - 1) / d]
    for f in p:
        v.append(v[-1] - h * f / (M * n))
    g = [s[1]]
    for u in v:
        g.append(g[-1] - h * u / (n + 1))
    need(1 + d * v[-1] == e[2], "velocity readback")
    need(g[-1] == e[1], "position readback")
    need([-(u - t) * M * n / h for t, u in zip(v, v[1:])] == p, "inertial force")
    return g, v


def material_piece(g, v, p, k, width, d, M, h):
    x = [width - z for z in g]
    y = [1 + d * u for u in v]
    need(min(g) >= 0 and min(x) >= 0 and all(a > 0 for a in x[1:]) and min(y) > 0, "active path")
    law = [k * z for z in product(x, y)]
    ext = product(p, [Q(1)] * (len(law) - len(p) + 1))
    diff = [a - b for a, b in zip(law, ext, strict=True)]
    if x[0] == 0:
        need(law[0] == ext[0] == 0 and min(law[1:]) > 0, "zero-force entry")
        relative = max(abs(r) / f for r, f in zip(diff[1:], law[1:], strict=True))
    else:
        relative = max(map(abs, diff)) / min(law)
    primitive = [Q(0)]
    for z in diff:
        primitive.append(primitive[-1] + h * z / len(diff))
    # Integrate the primitive across the segment for the time-weighted defect.
    db = h * sum(primitive) / len(primitive)
    dissip = product(x, product(v, v))
    loss = h * k * d * sum(dissip) / len(dissip)
    energy = M * (v[-1] ** 2 - v[0] ** 2) / 2 + k * (x[-1] ** 2 - x[0] ** 2) / 2 + loss
    return primitive[-1], db, max(map(abs, primitive)), max(0, energy), relative, loss


def audit_interval(
    folder, method, phase, seed, begin, h, fixture, layer, protocol, cfg, gates, ref, bracket
):
    M = 2 * Q("1/2")
    k, width, d = layer
    c = fixture["controls"]["sc2"]
    speed = Q(c["closing_speed_m_s"])
    entry = (Q(c["gap_initial_m"]) - width) / speed
    has_free = phase == "entry"
    span = h - entry if has_free else h
    rate = max(
        float(k * d * width / M),
        math.sqrt(float(k * (1 + d * Q(fixture["rate_range_m_s"][1])) / M)),
    )
    n = max(1, math.ceil(8 * float(span) * rate))
    dt = span / n
    idx = json.loads((folder / "index.json").read_bytes())
    need(
        idx["method"] == method
        and idx["phase"] == phase
        and idx["protocol_sha256_lf"] == PLAN_SHA
        and idx["source_sha256_lf"]
        == sha(Path(__file__).with_name("probe_p65_layer_entry.py"), normalize=True),
        "producer binding",
    )
    need(
        idx["origin"]
        == ("original scalar impact inlet" if has_free else method + "/entry/end.state.json"),
        "origin",
    )
    for f, digest in idx["hashes"].items():
        need(sha(folder / f) == digest, "trace digest")
    raw = (folder / "states.jsonl").read_bytes().splitlines(keepends=True)
    need(raw[0] == seed and raw[-1] == (folder / "end.state.json").read_bytes(), "cold readback")
    states = [parse_state(b, cfg)[0] for b in raw]
    macros = [json.loads(s) for s in (folder / "forces.jsonl").read_bytes().splitlines()]
    rows = [json.loads(s) for s in (folder / "pieces.jsonl").read_bytes().splitlines()]
    need(
        len(rows) == n + int(has_free) == idx["parts"]
        and len(states) == len(rows) + 1
        and states[0][0] == begin
        and states[-1][0] == begin + h,
        "full outer coverage",
    )
    expected_sizes = ([1] if has_free else []) + (
        [1] * n if method == "affine" else [2] * (n // 2) + ([1] if n % 2 else [])
    )
    need(
        [m["parts"] for m in macros] == expected_sizes and len(macros) == idx["macros"],
        "macro coverage",
    )
    J = B = dj = db = prefix = work = relative = loss = jump = Q(0)
    budget = [0]
    boxes = []
    core = states[0][1]
    prev_f = None
    rowidx = 0
    first_bad = None
    first_active = None
    for mi, m in enumerate(macros):
        free = has_free and mi == 0
        need(m["phase"] == ("free" if free else "active"), "wrong event ordering")
        size = m["parts"]
        controls = [Q(float.fromhex(word)) for word in m["words"]]
        need(
            all(float(z).hex() == w for z, w in zip(controls, m["words"], strict=True))
            and len(controls) == (1 if free else 2 if method == "affine" else 8)
            and min(controls) >= 0,
            "force record",
        )
        need(
            Q(m["t"]) == states[rowidx][0] and Q(m["dt"]) == (entry if free else size * dt),
            "macro time",
        )
        for side in range(size):
            s, e = states[rowidx : rowidx + 2]
            r = rows[rowidx]
            need(r["macro"] == mi and r["side"] == side, "piece membership")
            length = e[0] - s[0]
            need(length == (entry if free else dt), "piece partition")
            p = restrict(controls, size, side)
            g, v = kinematics(s, e, p, M, d)
            j, b = state_moments(s, e, M, d)
            need(
                (Q(r["j"]), Q(r["b"])) == (j, b) and 0 <= b <= length * j, "reported moment or cone"
            )
            for z in (j, b):
                need(
                    max(z.numerator.bit_length(), z.denominator.bit_length())
                    <= cfg["bits_per_exported_integer"],
                    "moment width",
                )
            core = min(core, *g)
            if free:
                free_path(s, e, p, entry, width, speed, d)
                need(j == b == 0, "event impulse")
            else:
                if first_active is None:
                    first_active = (s, e, p)
                    if has_free:
                        need(s[0] == entry and s[1] == width and p[0] == 0, "layer onset")
                localj, localb, pre, pos, rel, diss = material_piece(
                    g, v, p, k, width, d, M, length
                )
                loss += diss
                prefix = max(prefix, abs(dj) + pre)
                dj += localj
                db += localb + (begin + h - e[0]) * localj
                work += pos
                relative = max(relative, rel)
                if first_bad is None and (
                    abs(dj) > gates["law_impulse_vs_delivered_ns"]
                    or work > gates["positive_work_defect_per_window_j"]
                ):
                    first_bad = dict(
                        piece=rowidx,
                        t0=str(s[0]),
                        t1=str(e[0]),
                        cumulative_law_j_ns=float(dj),
                        positive_work_j=float(work),
                    )
            boxes.extend(peak_boxes(p, s[0], e[0], budget, protocol["limits"]))
            if prev_f is not None:
                jump = max(jump, abs(prev_f - p[0]))
            prev_f = p[-1]
            J += j
            B += b + (begin + h - e[0]) * j
            rowidx += 1
    need(
        (J, B) == state_moments(states[0], states[-1], M, d) and 0 < B < h * J and core >= 0,
        "outer strict delivery",
    )
    active_macros = len(macros) - int(has_free)
    counts = idx["work"]
    expected = dict(
        force_charges=(3 if method == "affine" else 8) * active_macros,
        convolution_terms=0 if method == "affine" else 36 * active_macros,
        conversion_terms=0 if method == "affine" else 36 * active_macros,
        half_subdivision_means=0 if method == "affine" else 56 * (n // 2),
        method=method,
    )
    need(
        counts == expected and counts["force_charges"] <= cfg["force_evaluations_per_outer_step"],
        "work budget",
    )
    lo = max(b[0] for b in boxes)
    hi = max(b[1] for b in boxes)
    candidates = [b for b in boxes if b[1] >= lo]
    tlo = min(b[2] for b in candidates)
    thi = max(b[3] for b in candidates)
    need(hi - lo <= Q(str(protocol["limits"]["peak_enclosure_width_n"])), "peak value enclosure")
    refpeak = Q(ref["ref_peak_n"])
    errors = dict(
        step_peak_vs_reference_n=max(abs(lo - refpeak), abs(hi - refpeak)),
        step_impulse_vs_reference_ns=abs(J - Q(ref["ref_j_ns"])),
        step_moment_error_divided_by_mass_m=abs(B - Q(ref["ref_b_ns2"])) / M,
        endpoint_gap_error_m=abs(states[-1][1] - Q(ref["ref_end_gap_m"])),
        endpoint_speed_error_m_s=abs((states[-1][2] - 1) / d - Q(ref["ref_end_speed_m_s"])),
        impact_peak_time_error_s=max(Q(0), bracket[0] - tlo, thi - bracket[1]),
        law_impulse_vs_delivered_ns=abs(dj),
        law_moment_error_divided_by_mass_m=abs(db) / M,
        prefix_momentum_defect_ns=prefix,
        positive_work_defect_per_window_j=work,
    )
    failures = [k for k, val in errors.items() if val > gates[k]]
    if relative > Q(1, 100):
        failures.append("relative_constitutive_path")
    details = dict(
        phase=phase,
        entry_s=str(entry) if has_free else None,
        active_s=str(span),
        parts=len(rows),
        force_macros=active_macros,
        work=counts,
        max_wire_bytes=max(map(len, raw)),
        peak_cells=budget[0],
        peak_n=[float(lo), float(hi)],
        peak_time_s=[float(tlo), float(thi)],
        ref_peak_n=float(refpeak),
        ref_time_s=list(map(float, bracket)),
        j_ns=float(J),
        b_ns2=float(B),
        exact_j_ns=str(J),
        exact_b_ns2=str(B),
        ref_j_ns=float(ref["ref_j_ns"]),
        centroid_s=float(begin + h - B / J),
        min_gap_lower_m=float(core),
        end_gap_m=float(states[-1][1]),
        end_speed_m_s=float((states[-1][2] - 1) / d),
        peak_relative_law_bound=float(relative),
        worst_knot_jump_n=float(jump),
        dissipation_j=float(loss),
        positive_work_j=float(work),
        signed_law_j_ns=float(dj),
        signed_law_b_ns2=float(db),
        errors={k: float(v) for k, v in errors.items()},
        failed_gates=failures,
        first_prefix_or_work_threshold=first_bad,
        exact_inertial_path=True,
        strict_nonnegative=True,
        end_sha256=idx["hashes"]["end.state.json"],
        trace_hashes=idx["hashes"],
    )
    return details, (work, abs(J - Q(ref["ref_j_ns"])), dj, prefix), first_active


def controls_negative(states, active, M, d, entry, width, speed):
    start, incident = states
    s, e, p = active
    need(p != p[::-1], "uninformative timing control")
    drops = (incident[0], Q(float(incident[1])), Q(float(incident[2])))
    return dict(
        free_load=reject(
            "free load",
            "free load",
            lambda: free_path(start, incident, [Q(1)], entry, width, speed, d),
        ),
        shifted_event=reject(
            "event time",
            "event time",
            lambda: free_path(
                start, (incident[0] + Q(1, 16384), *incident[1:]), [Q(0)], entry, width, speed, d
            ),
        ),
        dropped_gap_remainder=reject(
            "drop", "event gap", lambda: free_path(start, drops, [Q(0)], entry, width, speed, d)
        ),
        swapped_timing=reject(
            "timing", "position readback", lambda: kinematics(s, e, p[::-1], M, d)
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", choices=("affine", "series7"), required=True)
    args = parser.parse_args()
    need(
        sha(PLAN, normalize=True) == PLAN_SHA and sha(INPUT, normalize=True) == INPUT_SHA,
        "changed protocol",
    )
    plan = json.loads(PLAN.read_bytes())
    fixture = json.loads(INPUT.read_bytes())
    prior = json.loads(INPUT.with_name(plan["reference_index"]).read_bytes())
    old = INPUT.with_name(plan["gates_from"])
    need(sha(old, normalize=True) == prior["protocol_sha256_lf"], "changed inherited gates")
    protocol = json.loads(old.read_bytes())
    gates = {k: Q(str(v)) for k, v in protocol["gates"].items()}
    cfg = json.loads(INPUT.with_name(plan["wire_from"]).read_bytes())["wire_limits"]
    mat = json.loads(INPUT.with_name(prior["reference_summary"]).read_bytes())["nominal"]
    layer = tuple(Q(mat[k]) for k in ("k", "delta", "d"))
    table = INPUT.with_name(prior["step_table"]["file"])
    need(sha(table, normalize=True) == prior["step_table"]["sha256_lf"], "changed reference CSV")
    with table.open(newline="", encoding="utf8") as f:
        refs = [
            a
            for a in csv.DictReader(f)
            if a["case"] == "impact"
            and a["h_s"] == "1/64"
            and a["mass_kg"] == "1/2"
            and a["step"] in ("0", "1")
        ]
    need([r["step"] for r in refs] == ["0", "1"], "reference membership")
    bracket = next(
        r["reference_peak_time_bracket_s"]
        for r in prior["runs"]
        if r["case"] == "impact" and r["h_s"] == "1/64" and r["m_kg"] == "1/2"
    )
    folder = args.root / args.method
    with (folder / "entry" / "states.jsonl").open("rb") as f:
        seed = f.readline()
        event = f.readline()
    parsed = [parse_state(b, cfg)[0] for b in (seed, event)]
    ctl = fixture["controls"]["sc2"]
    g0, u = Q(ctl["gap_initial_m"]), Q(ctl["closing_speed_m_s"])
    need(parsed[0] == (0, g0, 1 + layer[2] * u), "original inlet")
    h = Q("1/64")
    details = []
    sums = []
    for step, phase in enumerate(("entry", "successor")):
        if step:
            seed = (folder / "entry" / "end.state.json").read_bytes()
        target = tuple(map(Q, bracket)) if step == 0 else (h, h)
        row, totals, first = audit_interval(
            folder / phase,
            args.method,
            phase,
            seed,
            step * h,
            h,
            fixture,
            layer,
            protocol,
            cfg,
            gates,
            refs[step],
            target,
        )
        details.append(row)
        sums.append(totals)
        if not step:
            faults = controls_negative(
                parsed, first, Q(1), layer[2], (g0 - layer[1]) / u, layer[1], u
            )
    work = sum(u[0] for u in sums)
    jerr = sum(u[1] for u in sums)
    prefix = max(sums[0][3], abs(sums[0][2]) + sums[1][3])
    combined = {
        k: float(v)
        for k, v in dict(
            positive_work_j=work,
            absolute_step_reference_impulse_errors_ns=jerr,
            prefix_momentum_ns=prefix,
        ).items()
    }
    pair_fail = [
        key
        for key, val, limit in (
            ("pair_work", work, gates["positive_work_defect_per_window_j"]),
            ("pair_impulse", jerr, gates["window_impulse_vs_reference_ns"]),
            ("pair_prefix", prefix, gates["prefix_momentum_defect_ns"]),
        )
        if val > limit
    ]
    passed = not any(r["failed_gates"] for r in details) and not pair_fail
    result = dict(
        schema="sc-l1-entry-peak-audit-v1",
        protocol_sha256_lf=PLAN_SHA,
        method=args.method,
        passed=passed,
        structural_pass=True,
        comparison_eligible=args.method == "affine" and not passed,
        source_sha256_lf=sha(Path(__file__), normalize=True),
        negative_controls=faults,
        intervals=details,
        pair=combined,
        pair_failed_gates=pair_fail,
    )
    with (folder / "audit.json").open("x", encoding="utf8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    for r in details:
        print(
            args.method,
            r["phase"],
            "peak",
            r["peak_n"],
            "J",
            r["j_ns"],
            "failures",
            r["failed_gates"],
        )
    print("pair", pair_fail, "passed", passed)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
