"""Independently recover path, pressure and both moments from saved public frames."""

import argparse
import csv
import json
import math
from fractions import Fraction as Q
from pathlib import Path

from audit_p65_layer_intrastep import product
from verify_p65_contact_layer_screen import INPUT, INPUT_SHA, sha

PLAN = INPUT.with_name("p65-contact-layer-publication-protocol.json")
PLAN_SHA = "58f93ed30dc2733cfb5814428442718fb0ade03c15e26d4d43cdf6d6dcc617af"


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def parse_state(data, limits):
    need(len(data) <= limits["bytes_per_motion_packet"], "oversize state")
    p = json.loads(data)
    need(set(p) == {"schema", "t", "gap", "y"} and p["schema"] == "sc-l1-motion-v1", "state schema")
    t = Q(p["t"])
    need(str(t) == p["t"], "noncanonical time")
    values = []
    bits = 0
    for k in ("gap", "y"):
        word, rem = p[k]
        f = float.fromhex(word)
        r = Q(rem)
        n = max(r.numerator.bit_length(), r.denominator.bit_length())
        exact = Q(f) + r
        need(math.isfinite(f) and float(exact).hex() == word and str(r) == rem, "state projection")
        need(n <= limits["bits_per_exported_integer"], "state remainder limit")
        bits = max(bits, n)
        values.append(exact)
    return (t, *values), bits


def state_moments(s, e, M, d):
    t, g, y = s
    te, ge, ye = e
    u = (y - 1) / d
    return M * (y - ye) / d, M * (ge - g + (te - t) * u)


def check_delivery(s, e, L, R, j, b, M, d):
    h = e[0] - s[0]
    need(h > 0, "time order")
    need(min(L, R) >= 0, "negative pressure")
    need((j, b) == (h * (L + R) / 2, h * h * (2 * L + R) / 6), "force moments")
    need((j, b) == state_moments(s, e, M, d), "motion moments")
    need(j >= 0 and 0 <= b <= h * j, "outside moment cone")


def check_path(s, e, L, R, j, b, layer, M):
    k, width, d = layer
    check_delivery(s, e, L, R, j, b, M, d)
    h = e[0] - s[0]
    va, ve = (s[2] - 1) / d, (e[2] - 1) / d
    xa, xe = width - s[1], width - e[1]
    x = [xa, xa + h * va / 3, xe - h * ve / 3, xe]
    vel = [3 * (v - u) / h for u, v in zip(x, x[1:])]
    y = [1 + d * v for v in vel]
    need(width - max(x) >= 0 and min(x) > 0 and min(y) > 0, "active path or hard core")
    need(
        (-2 * M * (vel[1] - vel[0]) / h, -2 * M * (vel[2] - vel[1]) / h) == (L, R),
        "inertial pressure",
    )
    law = [k * p for p in product(x, y)]
    degree = len(law) - 1
    residual = [p - L - (R - L) * Q(i, degree) for i, p in enumerate(law)]
    relative = max(map(abs, residual)) / min(law)
    need(relative <= Q(1, 100), "relative constitutive path bound")
    primitive = [Q(0)]
    for coeff in residual:
        primitive.append(primitive[-1] + h * coeff / (degree + 1))
    bdiff = (
        h
        * h
        * sum((degree + 1 - i) * v for i, v in enumerate(residual))
        / ((degree + 1) * (degree + 2))
    )
    integrand = product(x, product(vel, vel))
    diss = h * k * d * sum(integrand) / len(integrand)
    defect = M * (ve * ve - va * va) / 2 + k * (xe * xe - xa * xa) / 2 + diss
    return dict(
        relative=relative,
        dj=primitive[-1],
        db=bdiff,
        prefix=max(map(abs, primitive)),
        positive=max(0, defect),
        core=width - max(x),
    )


def collapse_words(state):
    t, g, y = state
    return t, Q(float(g)), Q(float(y))


def reject(label, expected, fn):
    try:
        fn()
    except ValueError as error:
        need(str(error) == expected, label + " reached unrelated rejection")
        return str(error)
    raise ValueError(label + " unexpectedly accepted")


def audit_step(directory, seed, begin, h, mat, M, config, ref, gates, run):
    raw = (directory / "states.jsonl").read_bytes().splitlines(keepends=True)
    records = [json.loads(line) for line in (directory / "loads.jsonl").read_bytes().splitlines()]
    bits = 0
    states = []
    for frame in raw:
        obj, used = parse_state(frame, config)
        states.append(obj)
        bits = max(bits, used)
    need(
        raw[0] == seed and raw[-1] == (directory / "end.state.json").read_bytes(),
        "persisted seed/end",
    )
    need(states[0][0] == begin and states[-1][0] == begin + h, "outer times")
    for file, key in (
        ("states.jsonl", "states_sha256"),
        ("loads.jsonl", "loads_sha256"),
        ("end.state.json", "end_sha256"),
    ):
        need(sha(directory / file) == run[key], "unbound evidence")
    need(
        len(states) == len(records) + 1 and len(records) == run["parts"] and records,
        "piece coverage",
    )
    k, width, d = mat
    J = B = errj = errb = prefix = work = ratio_max = peak = Q(0)
    core = width
    word_pieces = calls = 0
    first = None
    for i, (s, e, p) in enumerate(zip(states[:-1], states[1:], records, strict=True)):
        dt = e[0] - s[0]
        need(dt == h / len(records), "partition")
        L, R = (Q(float.fromhex(p[key])) for key in ("p0", "p1"))
        j, b = Q(p["j"]), Q(p["b"])
        report = check_path(s, e, L, R, j, b, mat, M)
        if i == 0:
            first = (s, e, L, R, j, b)
        calls += p["calls"]
        need(p["calls"] == 3, "piece law charge")
        word_pieces += int(state_moments(collapse_words(s), collapse_words(e), M, d) != (j, b))
        J += j
        B += b + (begin + h - e[0]) * j
        prefix = max(prefix, abs(errj) + report["prefix"])
        errj += report["dj"]
        errb += report["db"] + (begin + h - e[0]) * report["dj"]
        work += report["positive"]
        ratio_max = max(ratio_max, report["relative"])
        peak = max(peak, L, R)
        core = min(core, report["core"])
    need(
        (J, B) == state_moments(states[0], states[-1], M, d) and J > 0 and 0 < B < h * J,
        "outer exact delivery",
    )
    need(
        calls == run["calls"] and calls <= config["force_evaluations_per_outer_step"],
        "outer law charge",
    )
    need(
        abs(errj) <= gates["law_impulse_vs_delivered_ns"]
        and abs(errb) / M <= gates["law_moment_error_divided_by_mass_m"]
        and prefix <= gates["prefix_momentum_defect_ns"]
        and work <= gates["positive_work_defect_per_window_j"],
        "original law/work gates",
    )
    observed = None
    if ref is not None:
        target = {
            "step_impulse_vs_reference_ns": J - Q(ref["ref_j_ns"]),
            "step_moment_error_divided_by_mass_m": (B - Q(ref["ref_b_ns2"])) / M,
            "step_peak_vs_reference_n": peak - Q(ref["ref_peak_n"]),
            "endpoint_gap_error_m": states[-1][1] - Q(ref["ref_end_gap_m"]),
            "endpoint_speed_error_m_s": (states[-1][2] - 1) / d - Q(ref["ref_end_speed_m_s"]),
        }
        need(all(abs(v) <= gates[key] for key, v in target.items()), "original reference gates")
        observed = {k: float(v) for k, v in target.items()}
    wordj, wordb = state_moments(collapse_words(states[0]), collapse_words(states[-1]), M, d)
    return dict(
        start=str(begin),
        end=str(begin + h),
        parts=len(records),
        calls=calls,
        j_ns=str(J),
        b_ns2=str(B),
        j_decimal_ns=float(J),
        b_decimal_ns2=float(B),
        hard_gap_lower_m=float(core),
        max_relative_law_error_bound=float(ratio_max),
        positive_work_j=float(work),
        prefix_ns=float(prefix),
        max_wire_bits=bits,
        max_wire_bytes=max(map(len, raw)),
        exact_momentum_and_position=True,
        word_only_piece_mismatches=word_pieces,
        word_only_delta_j_ns=float(wordj - J),
        word_only_delta_b_ns2=float(wordb - B),
        word_only_outer_matches=(wordj, wordb) == (J, B),
        reference_errors=observed,
        inlet=json.loads(raw[0]),
        outlet=json.loads(raw[-1]),
        end_sha256=run["end_sha256"],
    ), first


def faults(failed, first, M, d):
    s, e, L, R, j, b = first
    h = e[0] - s[0]
    oldj, oldb = state_moments(
        (Q(failed["t0"]), Q(failed["g0"]), Q(failed["y0"])),
        (Q(failed["t1"]), Q(failed["g1"]), Q(failed["ye"])),
        M,
        d,
    )
    oh = Q(failed["h"])
    need(
        (oldj, oldb) == (Q(failed["j_ns"]), Q(failed["b_ns2"]))
        and not (oldj >= 0 and 0 <= oldb <= oh * oldj),
        "original witness does not reject",
    )
    need(L != R and L > 0, "uninformative pressure tamper control")
    return dict(
        old_endpoint="outside moment cone",
        dropped_remainders=reject(
            "missing carries",
            "motion moments",
            lambda: check_delivery(collapse_words(s), collapse_words(e), L, R, j, b, M, d),
        ),
        same_impulse_other_timing=reject(
            "swapped pressure",
            "motion moments",
            lambda: check_delivery(s, e, R, L, j, h * h * (2 * R + L) / 6, M, d),
        ),
        negative_force=reject(
            "negative load", "negative pressure", lambda: check_delivery(s, e, -L, R, j, b, M, d)
        ),
        reversed_time=reject(
            "time order", "time order", lambda: check_delivery(e, s, L, R, j, b, M, d)
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--group", choices=("primary", "remaining"), required=True)
    args = parser.parse_args()
    need(
        sha(PLAN, normalize=True) == PLAN_SHA and sha(INPUT, normalize=True) == INPUT_SHA,
        "protocol changed",
    )
    plan = json.loads(PLAN.read_bytes())
    old = json.loads(INPUT.with_name(plan["old_results"]).read_bytes())
    oldplan = INPUT.with_name(plan["old_protocol"])
    need(sha(oldplan, normalize=True) == old["protocol_sha256_lf"], "prior gates changed")
    gates = {k: Q(str(v)) for k, v in json.loads(oldplan.read_bytes())["gates"].items()}
    calibration = json.loads(INPUT.with_name(old["reference_summary"]).read_bytes())["nominal"]
    mat = tuple(Q(calibration[k]) for k in ("k", "delta", "d"))
    fixture = json.loads(INPUT.read_bytes())
    controls = fixture["controls"]
    M = 2 * Q("1/2")
    cfg = plan["wire_limits"]
    table = INPUT.with_name(old["step_table"]["file"])
    need(sha(table, normalize=True) == old["step_table"]["sha256_lf"], "reference table changed")
    with table.open(newline="", encoding="utf8") as stream:
        ref = {
            (row["h_s"], int(row["step"])): row
            for row in csv.DictReader(stream)
            if row["case"] == "impact" and row["mass_kg"] == "1/2"
        }
    failed = {(r["h"], r["step"]): r for r in old["outer_failures"]}
    ids = [plan["primary"]] if args.group == "primary" else plan["others"]
    expected = [f"h{Q(grid).denominator}-step{step}" for grid, step in ids]
    root = args.root / args.group
    indexes = {}
    for mode in ("witness", "successor"):
        index = json.loads((root / mode / "index.json").read_bytes())
        need(
            index["group"] == args.group
            and index["mode"] == mode
            and index["protocol_sha256_lf"] == PLAN_SHA
            and index["source_sha256_lf"]
            == sha(Path(__file__).with_name("probe_p65_layer_joint_publication.py"), normalize=True)
            and [r["id"] for r in index["runs"]] == expected,
            "source or run membership",
        )
        indexes[mode] = index["runs"]
    results = []
    for i, (grid, step) in enumerate(ids):
        ident = expected[i]
        h = Q(grid)
        row = failed[grid, step]
        witness_dir = root / "witness" / ident
        with (witness_dir / "states.jsonl").open("rb") as file:
            seed = file.readline(cfg["bytes_per_motion_packet"] + 1)
        initial, _ = parse_state(seed, cfg)
        need(initial == (step * h, Q(row["g0"]), Q(row["y0"])), "original inlet changed")
        pair = {}
        for mode in ("witness", "successor"):
            n = step + int(mode == "successor")
            if mode == "successor":
                seed = (witness_dir / "end.state.json").read_bytes()
                need(
                    indexes[mode][i]["origin"]
                    == witness_dir.relative_to(args.root).as_posix() + "/end.state.json",
                    "cold origin",
                )
            reference = ref[grid, n] if (n + 1) * h <= Q(controls["window_s"]) else None
            record, first = audit_step(
                root / mode / ident, seed, n * h, h, mat, M, cfg, reference, gates, indexes[mode][i]
            )
            if mode == "witness":
                negatives = faults(row, first, M, mat[2])
            pair[mode] = record
        bound = sum(r["positive_work_j"] for r in pair.values())
        need(bound <= float(gates["positive_work_defect_per_window_j"]), "pair work budget")
        jerr = sum(
            abs(r["reference_errors"]["step_impulse_vs_reference_ns"])
            for r in pair.values()
            if r["reference_errors"]
        )
        need(
            jerr <= float(gates["window_impulse_vs_reference_ns"]), "pair reference impulse budget"
        )
        results.append(dict(id=ident, negative_controls=negatives, **pair))
    out = args.root / (args.group + "-audit.json")
    with out.open("x", encoding="utf8") as stream:
        json.dump(
            dict(
                schema="sc-l1-joint-publication-audit-v1",
                passed=True,
                protocol_sha256_lf=PLAN_SHA,
                audit_source_sha256_lf=sha(Path(__file__), normalize=True),
                runs=results,
            ),
            stream,
            indent=2,
        )
        stream.write("\n")
    print(f"{args.group}: {len(results)} original witnesses with cold successors verified")


if __name__ == "__main__":
    main()
