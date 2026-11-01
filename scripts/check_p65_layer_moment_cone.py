"""Test whether stored scalar endpoints admit any nonnegative force with the same moments."""

import argparse
import csv
import json
from collections import Counter
from fractions import Fraction as Q
from itertools import groupby
from pathlib import Path

from verify_p65_contact_layer_screen import sha


def inspect(piece):
    mass = 2 * Q(piece["m"])
    applied = mass / 8 if piece["case"] == "support" else 0
    h = Q(piece["t1"]) - Q(piece["t0"])
    v0, vb, ve = ((Q(piece[key]) - 1) / 1000 for key in ("y0", "yb", "ye"))
    g0, g1 = map(Q, (piece["g0"], piece["g1"]))
    free_end = g0 - h * (v0 + vb + ve) / 3
    publication = g1 - free_end
    j = applied * h - mass * (ve - v0)
    raw_b = applied * h * h / 2 + mass * (free_end - g0 + h * v0)
    b = applied * h * h / 2 + mass * (g1 - g0 + h * v0)
    raw_left = applied - 2 * mass * (vb - v0) / h
    raw_right = applied - 2 * mass * (ve - vb) / h
    physical_vb = 3 * (g0 - g1) / h - v0 - ve
    left = applied - 2 * mass * (physical_vb - v0) / h
    right = applied - 2 * mass * (ve - physical_vb) / h
    change = 6 * mass * publication / h**2
    if (left, right, b) != (raw_left + change, raw_right - change, raw_b + mass * publication):
        raise ValueError("endpoint publication difference does not explain the coefficients")
    before = not (j >= 0 and 0 <= raw_b <= h * j)
    after = not (j >= 0 and 0 <= b <= h * j)
    return dict(
        case=piece["case"],
        mass=piece["m"],
        grid=piece["h"],
        step=piece["step"],
        t0=piece["t0"],
        t1=piece["t1"],
        phase=piece["phase"],
        gap_publication_m=publication,
        j_ns=j,
        raw_b_ns2=raw_b,
        b_ns2=b,
        hj_ns2=h * j,
        raw_min_n=min(raw_left, raw_right),
        published_min_n=min(left, right),
        raw_outside_cone=before,
        published_outside_cone=after,
        raw_negative_force=min(raw_left, raw_right) < 0,
        published_negative_force=min(left, right) < 0,
        negative_delivered_j=j < 0,
        constant_velocity=(v0 == vb == ve),
        zero_j_nonzero_b=(j == 0 and b != 0),
    )


def inspect_outer(pieces, steps_csv, sign_dir, output, base):
    sign = json.loads((sign_dir / "sign.json").read_bytes())
    failed_file = sign_dir / "sign-failures.jsonl"
    if (
        sha(failed_file) != sign["failures_sha256"]
        or sign["pieces_sha256"] != base["piece_file_sha256"]
        or sha(steps_csv) != base["step_csv_sha256"]
    ):
        raise ValueError("outer inputs differ from the piece and step audits")
    failed_pieces = Counter()
    for line in failed_file.read_text(encoding="utf8").splitlines():
        row = json.loads(line)
        if row["published_outside_cone"]:
            failed_pieces[row["case"], row["mass"], row["grid"], row["step"]] += 1
    if failed_pieces.total() != sign["counts"]["published_outside_cone"]:
        raise ValueError("piece-failure membership differs")
    counts = Counter()
    failures = []
    seen = set()

    def key(p):
        return p["case"], p["m"], p["h"], p["step"]

    with (
        pieces.open(encoding="utf8") as data,
        steps_csv.open(encoding="utf8", newline="") as csv_file,
    ):
        groups = groupby(map(json.loads, data), key)
        for (identity, group), prior in zip(groups, csv.DictReader(csv_file), strict=True):
            case, m, grid, step = identity
            if (
                identity != (prior["case"], prior["mass_kg"], prior["h_s"], int(prior["step"]))
                or identity in seen
            ):
                raise ValueError("unexpected outer-step identity")
            seen.add(identity)
            segment = list(group)
            start, end = segment[0], segment[-1]
            h = Q(grid)
            if Q(start["t0"]) != step * h or Q(end["t1"]) != (step + 1) * h:
                raise ValueError("outer endpoint time differs")
            M = 2 * Q(m)
            applied = M / 8 if case == "support" else 0
            v0 = (Q(start["y0"]) - 1) / 1000
            v1 = (Q(end["ye"]) - 1) / 1000
            j = applied * h - M * (v1 - v0)
            b = applied * h * h / 2 + M * (Q(end["g1"]) - Q(start["g0"]) + h * v0)
            if float(j) != float(prior["candidate_j_ns"]) or float(b) != float(
                prior["candidate_b_ns2"]
            ):
                raise ValueError("endpoint moments disagree with independent piece sums")
            outside = not (j >= 0 and 0 <= b <= h * j)
            inner = failed_pieces[identity]
            counts.update(
                outer_steps=1,
                outside=int(outside),
                zero_j_nonzero_b=int(j == 0 and b != 0),
                contains_failed_piece=int(inner > 0),
                inner_only=int(inner > 0 and not outside),
            )
            if outside:
                failures.append(
                    dict(
                        case=case,
                        m=m,
                        h=grid,
                        step=step,
                        t0=start["t0"],
                        t1=end["t1"],
                        g0=start["g0"],
                        g1=end["g1"],
                        y0=start["y0"],
                        ye=end["ye"],
                        j_ns=j,
                        b_ns2=b,
                        hj_ns2=h * j,
                        failed_pieces=inner,
                    )
                )
    if len(seen) != base["step_rows"] or set(failed_pieces) - seen:
        raise ValueError("incomplete outer-step inventory")
    result = dict(
        schema="p65-sc-l1-outer-moment-cone-v1",
        counts=counts,
        failures=failures,
        pieces_sha256=base["piece_file_sha256"],
        step_csv_sha256=base["step_csv_sha256"],
        sign_sha256=sha(sign_dir / "sign.json"),
        source_lf_sha256=sha(Path(__file__), normalize=True),
    )
    (output / "outer.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf8"
    )
    print(json.dumps(counts, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--outer-from-sign-dir", type=Path, help="check original steps using the saved piece audit"
    )
    args = parser.parse_args()
    base = json.loads((args.audit_dir / "audit.json").read_bytes())
    pieces = args.candidate_dir / "pieces.jsonl"
    if sha(pieces) != base["piece_file_sha256"]:
        raise ValueError("candidate digest changed")
    args.output_dir.mkdir(exist_ok=False)
    if args.outer_from_sign_dir is not None:
        inspect_outer(
            pieces, args.audit_dir / "steps.csv", args.outer_from_sign_dir, args.output_dir, base
        )
        return
    counts = Counter()
    by_run = {}
    worst = None
    first = None
    with (
        pieces.open(encoding="utf8") as stream,
        (args.output_dir / "sign-failures.jsonl").open("x", encoding="utf8") as out,
    ):
        for line in stream:
            piece = json.loads(line)
            row = inspect(piece)
            key = (row["case"], row["mass"], row["grid"])
            flags = {k: int(v) for k, v in row.items() if isinstance(v, bool)}
            flags["pieces"] = 1
            flags["introduced_cone_failure"] = int(
                row["published_outside_cone"] and not row["raw_outside_cone"]
            )
            flags["introduced_negative_force"] = int(
                row["published_negative_force"] and not row["raw_negative_force"]
            )
            counts.update(flags)
            by_run.setdefault(key, Counter()).update(flags)
            if (
                row["published_negative_force"]
                or row["published_outside_cone"]
                or row["raw_negative_force"]
            ):
                out.write(json.dumps(row, default=str, allow_nan=False) + "\n")
                if first is None or Q(row["t0"]) < Q(first["t0"]):
                    first = row
                if worst is None or row["published_min_n"] < worst["published_min_n"]:
                    worst = {**row, "producer_fields": piece}
    for run in base["runs"]:
        counts_run = by_run[run["case"], run["mass_kg"], run["h_s"]]
        if (
            run["strict_inferred_sign"] != (counts_run["published_negative_force"] == 0)
            or run["pieces"] != counts_run["pieces"]
        ):
            raise ValueError("independent sign result differs from original step audit")
    result = dict(
        schema="p65-sc-l1-endpoint-moment-cone-v1",
        counts=counts,
        runs=[dict(case=k[0], m=k[1], h=k[2], counts=v) for k, v in by_run.items()],
        first=first,
        worst=worst,
        source_lf_sha256=sha(Path(__file__), normalize=True),
        pieces_sha256=base["piece_file_sha256"],
        failures_sha256=sha(args.output_dir / "sign-failures.jsonl"),
    )
    (args.output_dir / "sign.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf8"
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
