"""Test whether stored scalar endpoints admit any nonnegative force with the same moments."""

import argparse
import json
from collections import Counter
from fractions import Fraction as Q
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads((args.audit_dir / "audit.json").read_bytes())
    pieces = args.candidate_dir / "pieces.jsonl"
    if sha(pieces) != base["piece_file_sha256"]:
        raise ValueError("candidate digest changed")
    args.output_dir.mkdir(exist_ok=False)
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
