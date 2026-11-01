"""Check uninterrupted support, imposed work, and preloaded energy from persisted data."""

import argparse
import csv
import json
import math
from fractions import Fraction as Q
from pathlib import Path

from audit_p65_layer_entry import peak_boxes, restrict
from audit_p65_layer_intrastep import product
from audit_p65_layer_joint_publication import need, parse_state, reject, state_moments
from verify_p65_contact_layer_screen import INPUT, INPUT_SHA, sha

PLAN = INPUT.with_name("p65-contact-layer-support-protocol.json")
PLAN_SHA = "7e9a5854d41d26bfc7bb7adfe86caf64d09cded47122bf92db9d3e897d7bbf96"


def average(coeff):
    return sum(coeff) / len(coeff)


def motion(s, e, p, M, d, F):
    h = e[0] - s[0]
    need(h > 0 and min(p) > 0, "nonpositive duration or support force")
    v = [(s[2] - 1) / d]
    for contact in p:
        v.append(v[-1] + h * (F - contact) / (M * len(p)))
    g = [s[1]]
    for u in v:
        g.append(g[-1] - h * u / len(v))
    need(1 + d * v[-1] == e[2], "velocity readback")
    need(g[-1] == e[1], "position readback")
    need(
        all(F - M * len(p) * (u - v0) / h == f for u, v0, f in zip(v[1:], v[:-1], p, strict=True)),
        "inertial contact force",
    )
    return g, v


def imposed_work(ledger, F, v, g, h):
    integrated = F * h * average(v)
    need(integrated == F * (g[0] - g[-1]) == ledger, "signed external work")
    return integrated


def preload(claim, K, width):
    need(claim == K * width**2 / 2, "initial elastic storage")


def piece_metrics(s, e, p, r, M, F, layer, T, first, bits):
    K, width, d = layer
    h = e[0] - s[0]
    g, v = motion(s, e, p, M, d, F)
    x = [width - z for z in g]
    y = [1 + d * u for u in v]
    need(
        min(g) >= 0
        and min(x) >= width - T / d > 0
        and max(x) <= width
        and min(y) > 0
        and max(v) <= 0,
        "support domain",
    )
    if first:
        need(
            g[:2] == [0, 0] and min(g[2:]) > 0 and p[0] == K * width and (p[0] - F) / M > 0,
            "preloaded outward path",
        )
    else:
        need(min(g) > 0, "return to the core")
    floor = F * (1 - T / (d * width))
    need(min(p) >= floor > 0, "support load floor")
    j, b = state_moments(s, e, M, d)
    j += F * h
    b += F * h * h / 2
    need((Q(r["j"]), Q(r["b"])) == (j, b) and 0 < b < h * j, "contact moments")
    need(
        (Q(r["external_n"]), Q(r["external_j"]), Q(r["external_b"])) == (F, F * h, F * h * h / 2),
        "external load binding",
    )
    for k in ("j", "b", "external_n", "external_j", "external_b", "external_work"):
        value = Q(r[k])
        need(
            str(value) == r[k]
            and max(value.numerator.bit_length(), value.denominator.bit_length()) <= bits,
            "published ratio",
        )
    W = imposed_work(Q(r["external_work"]), F, v, g, h)
    contact_work = -h * average(product(p, v))
    deltaK = M * (v[-1] ** 2 - v[0] ** 2) / 2
    need(deltaK == W + contact_work, "separate external and internal work")
    law = [K * z for z in product(x, y)]
    elevated = product(p, [Q(1)] * (len(law) - len(p) + 1))
    residual = [a - b for a, b in zip(law, elevated, strict=True)]
    relative = max(map(abs, residual)) / min(law)
    primitive = [Q(0)]
    for z in residual:
        primitive.append(primitive[-1] + h * z / len(residual))
    loss = h * K * d * average(product(x, product(v, v)))
    deltaU = K * (x[-1] ** 2 - x[0] ** 2) / 2
    defect = deltaK + deltaU + loss - W
    need(loss >= 0 and defect == h * average(product(residual, v)), "material energy identity")
    return dict(
        j=j,
        b=b,
        W=W,
        contact_work=contact_work,
        loss=loss,
        defect=defect,
        positive=max(0, defect),
        dj=primitive[-1],
        db=h * average(primitive),
        prefix=max(map(abs, primitive)),
        relative=relative,
        min_g=min(g),
        max_g=max(g),
        min_x=min(x),
        min_y=min(y),
        v=v,
        g=g,
    )


def enclosing(boxes):
    return max(b[0] for b in boxes), max(b[1] for b in boxes)


def audit_step(folder, seed, no, h, T, M, F, layer, fixture, gates, limits, wire, ref):
    K, width, d = layer
    idx = json.loads((folder / "index.json").read_bytes())
    origin = "support initial g=v=0" if no == 0 else f"step-{no - 1}/end.state.json"
    need(
        idx["step"] == no
        and idx["origin"] == origin
        and idx["protocol_sha256_lf"] == PLAN_SHA
        and idx["source_sha256_lf"]
        == sha(Path(__file__).with_name("probe_p65_layer_support.py"), normalize=True),
        "source and inlet binding",
    )
    need(
        set(idx["hashes"]) == {"states.jsonl", "forces.jsonl", "pieces.jsonl", "end.state.json"},
        "evidence inventory",
    )
    for file, digest in idx["hashes"].items():
        need(sha(folder / file) == digest, "trace digest")
    raw = (folder / "states.jsonl").read_bytes().splitlines(keepends=True)
    need(
        raw[0] == seed and raw[-1] == (folder / "end.state.json").read_bytes(),
        "cold input or output",
    )
    decoded = [parse_state(frame, wire) for frame in raw]
    states = [a for a, _ in decoded]
    records = [json.loads(b) for b in (folder / "pieces.jsonl").read_bytes().splitlines()]
    macros = [json.loads(b) for b in (folder / "forces.jsonl").read_bytes().splitlines()]
    rate = max(
        float(K * d * width / M),
        math.sqrt(float(K * (1 + d * Q(fixture["rate_range_m_s"][1])) / M)),
    )
    n = math.ceil(8 * float(h) * rate)
    sizes = [2] * (n // 2) + ([1] if n % 2 else [])
    need(
        len(records) == n == idx["parts"]
        and len(raw) == n + 1
        and len(macros) == len(sizes) == idx["force_macros"]
        and states[0][0] == no * h
        and states[-1][0] == (no + 1) * h,
        "original grid coverage",
    )
    if no == 0:
        need(states[0] == (0, 0, 1), "only the original inlet is allowed")
        preload(Q(idx["initial_elastic_j"]), K, width)
    for a, at in ((states[0], "initial"), (states[-1], "final")):
        need(
            Q(idx[at + "_elastic_j"]) == K * (width - a[1]) ** 2 / 2
            and Q(idx[at + "_kinetic_j"]) == M * ((a[2] - 1) / d) ** 2 / 2,
            "published energy",
        )
    J = B = W = contact = loss = positive = defect = dj = db = prefix = relative = Q(0)
    gap_lo = gap_hi = states[0][1]
    min_x = width - gap_lo
    boxes = []
    negative_boxes = []
    budget = [0]
    row = 0
    sample = None
    last_p = None
    max_jump = Q(0)
    for mi, (m, size) in enumerate(zip(macros, sizes, strict=True)):
        words = m["words"]
        controls = [Q(float.fromhex(w)) for w in words]
        need(
            len(controls) == 8
            and min(controls) > 0
            and all(float(z).hex() == w for z, w in zip(controls, words, strict=True)),
            "traction words",
        )
        need(
            m["parts"] == size and Q(m["t"]) == states[row][0] and Q(m["dt"]) == size * h / n,
            "polynomial interval",
        )
        for side in range(size):
            s, e = states[row : row + 2]
            r = records[row]
            need(e[0] - s[0] == h / n and r["macro"] == mi and r["side"] == side, "piece partition")
            p = restrict(controls, size, side)
            u = piece_metrics(
                s, e, p, r, M, F, layer, T, no == 0 and row == 0, wire["bits_per_exported_integer"]
            )
            if row == 0:
                sample = (s, e, p, u)
            after = (no + 1) * h - e[0]
            J += u["j"]
            B += u["b"] + after * u["j"]
            prefix = max(prefix, abs(dj) + u["prefix"])
            dj += u["dj"]
            db += u["db"] + after * u["dj"]
            W += u["W"]
            contact += u["contact_work"]
            loss += u["loss"]
            positive += u["positive"]
            defect += u["defect"]
            gap_lo = min(gap_lo, u["min_g"])
            gap_hi = max(gap_hi, u["max_g"])
            min_x = min(min_x, u["min_x"])
            relative = max(relative, u["relative"])
            if last_p is not None:
                max_jump = max(max_jump, abs(p[0] - last_p))
            last_p = p[-1]
            boxes.extend(peak_boxes(p, s[0], e[0], budget, limits))
            negative_boxes.extend(peak_boxes([-f for f in p], s[0], e[0], budget, limits))
            row += 1
    hi = enclosing(boxes)
    neg = enclosing(negative_boxes)
    lo = (-neg[1], -neg[0])
    need(
        hi[1] - hi[0] <= Q(str(limits["peak_enclosure_width_n"]))
        and lo[1] - lo[0] <= Q(str(limits["peak_enclosure_width_n"])),
        "force extrema enclosures",
    )
    j0, b0 = state_moments(states[0], states[-1], M, d)
    need((J, B) == (j0 + F * h, b0 + F * h * h / 2) and 0 < B < h * J, "outer forced delivery")
    need(W == F * (states[0][1] - states[-1][1]) and W < 0, "outer external work")
    ua, ub = (Q(idx[k]) for k in ("initial_elastic_j", "final_elastic_j"))
    ka, kb = (Q(idx[k]) for k in ("initial_kinetic_j", "final_kinetic_j"))
    need(
        defect == kb - ka + ub - ua + loss - W and contact + W == kb - ka,
        "outer energy telescoping",
    )
    actual = idx["work"]
    expect = dict(
        method="loaded-series7",
        force_charges=8 * len(sizes),
        convolution_terms=36 * len(sizes),
        conversion_terms=36 * len(sizes),
        half_subdivision_means=56 * (n // 2),
    )
    need(
        actual == expect and actual["force_charges"] <= wire["force_evaluations_per_outer_step"],
        "charged work",
    )
    rp = Q(ref["ref_peak_n"])
    errors = dict(
        step_peak_vs_reference_n=max(abs(hi[0] - rp), abs(hi[1] - rp)),
        step_impulse_vs_reference_ns=abs(J - Q(ref["ref_j_ns"])),
        step_moment_error_divided_by_mass_m=abs(B - Q(ref["ref_b_ns2"])) / M,
        endpoint_gap_error_m=abs(states[-1][1] - Q(ref["ref_end_gap_m"])),
        endpoint_speed_error_m_s=abs((states[-1][2] - 1) / d - Q(ref["ref_end_speed_m_s"])),
        law_impulse_vs_delivered_ns=abs(dj),
        law_moment_error_divided_by_mass_m=abs(db) / M,
        prefix_momentum_defect_ns=prefix,
        positive_work_defect_per_window_j=positive,
    )
    failures = [key for key, val in errors.items() if val > gates[key]]
    if relative > Q(1, 100):
        failures.append("relative_constitutive_path")
    metrics = dict(
        step=no,
        parts=n,
        work=actual,
        peak_and_min_cells=budget[0],
        max_wire_bytes=max(map(len, raw)),
        max_remainder_bits=max(b for _, b in decoded),
        hard_gap_lower_m=float(gap_lo),
        hard_gap_upper_m=float(gap_hi),
        min_compression_m=float(min_x),
        force_min_n=list(map(float, lo)),
        force_max_n=list(map(float, hi)),
        force_end_n=float(last_p),
        end_gap_m=float(states[-1][1]),
        end_speed_m_s=float((states[-1][2] - 1) / d),
        contact_impulse_ns=float(J),
        contact_time_moment_ns2=float(B),
        external_impulse_ns=float(F * h),
        external_time_moment_ns2=float(F * h * h / 2),
        initial_elastic_j=float(ua),
        final_elastic_j=float(ub),
        initial_kinetic_j=float(ka),
        final_kinetic_j=float(kb),
        external_work_j=float(W),
        internal_contact_work_j=float(contact),
        dissipation_j=float(loss),
        signed_energy_defect_j=float(defect),
        positive_energy_defect_j=float(positive),
        relative_law_bound=float(relative),
        worst_knot_jump_n=float(max_jump),
        errors={k: float(v) for k, v in errors.items()},
        failed_gates=failures,
        origin=origin,
        hashes=idx["hashes"],
    )
    exact = dict(
        J=J,
        B=B,
        W=W,
        contact=contact,
        D=loss,
        positive=positive,
        defect=defect,
        dj=dj,
        db=db,
        prefix=prefix,
        gap_lo=gap_lo,
        gap_hi=gap_hi,
        min_x=min_x,
        lo=lo,
        hi=hi,
        s=states[0],
        e=states[-1],
        end_p=last_p,
        ref_j_error=errors["step_impulse_vs_reference_ns"],
    )
    return metrics, exact, sample


def check_faults(first, resumed, M, d, K, width, F):
    s, e, p, m = first
    h = e[0] - s[0]
    actual = m["W"]
    need(actual < 0 and m["contact_work"] > 0, "uninformative work control")

    def work(value):
        return imposed_work(value, F, m["v"], m["g"], h)

    rs, re, rp, _ = resumed
    rounded = (rs[0], Q(float(rs[1])), Q(float(rs[2])))
    need(rounded != rs, "no remainder at the cold boundary")
    try:
        motion(rounded, re, rp, M, d, F)
    except ValueError as exc:
        reason = str(exc)
        need(reason in ("position readback", "velocity readback"), "irrelevant readback rejection")
    else:
        raise ValueError("dropping cold remainder was not rejected")
    return dict(
        missing_force=reject("unforced", "velocity readback", lambda: motion(s, e, p, M, d, 0)),
        missing_work=reject("omitted work", "signed external work", lambda: work(0)),
        reversed_work=reject("reversed work", "signed external work", lambda: work(-actual)),
        internal_credit=reject(
            "internal credit", "signed external work", lambda: work(actual + m["contact_work"])
        ),
        lost_preload=reject("preload", "initial elastic storage", lambda: preload(0, K, width)),
        lost_resumed_remainder=reason,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    need(
        sha(PLAN, normalize=True) == PLAN_SHA and sha(INPUT, normalize=True) == INPUT_SHA,
        "registered inputs changed",
    )
    plan = json.loads(PLAN.read_bytes())
    old = json.loads(INPUT.with_name(plan["reference_index"]).read_bytes())
    prior = INPUT.with_name(plan["gates_from"])
    need(sha(prior, normalize=True) == old["protocol_sha256_lf"], "inherited limits")
    previous = json.loads(prior.read_bytes())
    gates = {k: Q(str(v)) for k, v in previous["gates"].items()}
    wire = json.loads(INPUT.with_name(plan["wire_from"]).read_bytes())["wire_limits"]
    reference = json.loads(INPUT.with_name(plan["reference_fine_summary"]).read_bytes())
    layer = tuple(Q(reference["nominal"][k]) for k in ("k", "delta", "d"))
    K, width, d = layer
    table = INPUT.with_name(old["step_table"]["file"])
    need(sha(table, normalize=True) == old["step_table"]["sha256_lf"], "archived reference table")
    with table.open(newline="", encoding="utf8") as file:
        refs = [
            r
            for r in csv.DictReader(file)
            if r["case"] == "support" and r["mass_kg"] == "1/2" and r["h_s"] == "1/64"
        ]
    fine = next(
        r for r in reference["reference_traces"] if r["case"] == "support" and r["mass_kg"] == "1/2"
    )
    fixture = json.loads(INPUT.read_bytes())
    ctl = fixture["controls"]
    h, T = Q("1/64"), Q(ctl["window_s"])
    M, F = 2 * Q("1/2"), Q("1/2") / 4
    count = 8
    need(
        count * h == T
        and ctl["sc1"]["reduced_mass_kg"] == "2m"
        and ctl["sc1"]["compressive_load_n"] == "m/4"
        and [int(r["step"]) for r in refs] == list(range(count)),
        "complete support identities",
    )
    with (args.root / "step-0" / "states.jsonl").open("rb") as file:
        seed = file.readline(wire["bytes_per_motion_packet"] + 1)
    need(parse_state(seed, wire)[0] == (0, 0, 1), "initial gap or speed")
    details = []
    full = []
    samples = []
    prefix = history_j = history_b = Q(0)
    knot_jump = Q(0)
    for no in range(count):
        folder = args.root / f"step-{no}"
        record, u, sample = audit_step(
            folder, seed, no, h, T, M, F, layer, fixture, gates, previous["limits"], wire, refs[no]
        )
        seed = (folder / "end.state.json").read_bytes()
        if full:
            need(u["s"] == full[-1]["e"], "reset at outer boundary")
            knot_jump = max(knot_jump, abs(sample[2][0] - full[-1]["end_p"]))
        prefix = max(prefix, abs(history_j) + u["prefix"])
        history_j += u["dj"]
        history_b += u["db"] + (T - (no + 1) * h) * u["dj"]
        full.append(u)
        details.append(record)
        if no <= 1:
            samples.append(sample)
    a, e = full[0]["s"], full[-1]["e"]
    J = sum(r["J"] for r in full)
    B = sum(r["B"] + (T - (i + 1) * h) * r["J"] for i, r in enumerate(full))
    basej, baseb = state_moments(a, e, M, d)
    need((J, B) == (basej + F * T, baseb + F * T**2 / 2), "full-window forced moments")
    W = sum(r["W"] for r in full)
    D = sum(r["D"] for r in full)
    kinetic = M * ((e[2] - 1) / d) ** 2 / 2
    U0 = K * (width - a[1]) ** 2 / 2
    UT = K * (width - e[1]) ** 2 / 2
    preload(U0, K, width)
    defect = kinetic + UT - U0 + D - W
    need(
        defect == sum(r["defect"] for r in full) and W == F * (a[1] - e[1]) < 0,
        "signed energy or external work binding",
    )
    positive = sum(r["positive"] for r in full)
    ref_err = sum(r["ref_j_error"] for r in full)
    checks = {
        "noncancelling_energy": positive <= gates["positive_work_defect_per_window_j"],
        "sum_abs_reference_impulse_errors": ref_err <= gates["window_impulse_vs_reference_ns"],
        "full_window_prefix_momentum": prefix <= gates["prefix_momentum_defect_ns"],
    }
    force_lo = [min(r["lo"][k] for r in full) for k in range(2)]
    force_hi = [max(r["hi"][k] for r in full) for k in range(2)]
    need(
        force_lo[0] >= F * (1 - T / (d * width)) and min(r["gap_lo"] for r in full) == 0,
        "window support",
    )
    faults = check_faults(*samples, M, d, K, width, F)
    endv = (e[2] - 1) / d
    summary = dict(
        time_s=str(T),
        h_s=str(h),
        cold_boundaries=count - 1,
        force_floor_n=float(F * (1 - T / (d * width))),
        force_min_n=list(map(float, force_lo)),
        force_max_n=list(map(float, force_hi)),
        force_min_fraction_of_external=float(force_lo[0] / F),
        initial_separating_acceleration_m_s2=float((K * width - F) / M),
        minimum_hard_gap_m=0,
        max_hard_gap_bound_m=float(max(r["gap_hi"] for r in full)),
        min_compression_m=float(min(r["min_x"] for r in full)),
        final_gap_m=float(e[1]),
        final_closing_speed_m_s=float(endv),
        final_force_n=float(full[-1]["end_p"]),
        exact_contact_impulse_ns=str(J),
        contact_impulse_ns=float(J),
        external_impulse_ns=float(F * T),
        contact_time_moment_ns2=float(B),
        external_time_moment_ns2=float(F * T * T / 2),
        final_momentum_kg_m_s=float(M * endv),
        initial_elastic_j=float(U0),
        final_elastic_j=float(UT),
        kinetic_change_j=float(kinetic),
        external_work_j=float(W),
        internal_contact_work_j=float(sum(r["contact"] for r in full)),
        dissipation_j=float(D),
        signed_energy_defect_j=float(defect),
        sum_positive_piece_defects_j=float(positive),
        signed_window_law_impulse_error_ns=float(history_j),
        signed_window_law_time_moment_error_ns2=float(history_b),
        prefix_momentum_bound_ns=float(prefix),
        sum_abs_reference_impulse_errors_ns=float(ref_err),
        max_relative_law_error_bound=max(r["relative_law_bound"] for r in details),
        max_outer_knot_jump_n=float(knot_jump),
        max_wire_bytes=max(r["max_wire_bytes"] for r in details),
        max_remainder_bits=max(r["max_remainder_bits"] for r in details),
        reference_final=fine["final"],
        sampled_reference_min_force_n=fine["fine_min_force_n"],
    )
    passed = all(checks.values()) and all(not r["failed_gates"] for r in details)
    result = dict(
        schema="sc-l1-loaded-support-audit-v1",
        passed=passed,
        protocol_sha256_lf=PLAN_SHA,
        source_sha256_lf=sha(Path(__file__), normalize=True),
        checks=checks,
        summary=summary,
        steps=details,
        negative_controls=faults,
    )
    with (args.root / "audit.json").open("x", encoding="utf8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(
        "eight steps, support force",
        summary["force_min_n"],
        summary["force_max_n"],
        "N, positive work defect",
        float(positive),
        "J",
    )
    print(
        "failed gates",
        [(r["step"], r["failed_gates"]) for r in details if r["failed_gates"]],
        checks,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
