"""Classify genuinely matching supported trajectories separately from engineering qualification."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np


def unpack(path):
    report = json.loads(path.read_text(encoding="utf8"))
    guard = json.loads(path.with_name("final.json").read_text(encoding="utf-8-sig"))
    assert guard["raw_exit"] == 0
    return report


def native_load(report, scene):
    row = report["runs"][scene]
    worlds = row["rows"]
    measures = {}
    for w in worlds:
        key = str(w["key"])
        if report["engine"] == "phymeo":
            series = (
                np.array([sample[key]["normal_impulse_n_s"] for sample in row["native_per_step"]])
                / row["dt_s"]
            )
            active = [int(i + 1) for i, n in enumerate(series) if n > 1e-5]
        else:
            series = np.array(
                [
                    sum(f["force_on_sphere_n"][2] for f in sample[key])
                    for sample in row["native_per_step"]
                ]
            )
            active = [int(i + 1) for i, sample in enumerate(row["native_per_step"]) if sample[key]]
        measures[key] = dict(
            first_native_row_or_load=active[0] if active else None,
            force_z_n=series.tolist(),
            last_eight_mean_n=float(np.mean(series[-8:])),
        )
    return measures


def qualify(paths):
    entries = {}
    for path in paths:
        r = unpack(path)
        assert r["phase"] == "qualify"
        key = (r["device"], r["engine"])
        assert key not in entries
        entries[key] = (path, r)
    assert set(entries) == {(d, e) for d in ("cpu", "cuda:0") for e in ("phymeo", "newton")}
    comparisons = {}
    for dev in ("cpu", "cuda:0"):
        pp, pr = entries[dev, "phymeo"]
        npth, nr = entries[dev, "newton"]
        for scene in ("mixed", "isolated", "support"):
            p, n = (r["runs"][scene] for r in (pr, nr))
            assert p["rows"] == n["rows"]
            a = np.load(pp.with_name(p["trajectory"]), allow_pickle=False)
            b = np.load(npth.with_name(n["trajectory"]), allow_pickle=False)
            assert (
                a.shape == b.shape
                and a.dtype == b.dtype == np.float32
                and np.isfinite(a).all()
                and np.isfinite(b).all()
            )
            exact = np.all(a.view(np.uint32) == b.view(np.uint32), axis=2)
            mismatch = np.argwhere(~exact)
            delta = np.abs(a.astype(float) - b)
            native = {e: native_load(r, scene) for e, r in (("phymeo", pr), ("newton", nr))}
            load_ok = all(
                abs(native[e][str(w["key"])]["last_eight_mean_n"] - 16) <= 0.16
                for w in p["rows"]
                if w["role"] == "support"
                for e in native
            )
            world_differences = {}
            for col, w in enumerate(p["rows"]):
                different = np.flatnonzero(~exact[:, col])
                first = int(different[0]) if different.size else None
                world_differences[w["key"]] = dict(
                    role=w["role"],
                    mismatching_ticks=int(different.size),
                    first_tick=first,
                    first_ph=None if first is None else a[first, col].tolist(),
                    first_nt=None if first is None else b[first, col].tolist(),
                )
            comparisons[f"{dev}-{scene}"] = dict(
                full_raw_sha256=dict(
                    phymeo=sha256(a.tobytes()).hexdigest(), newton=sha256(b.tobytes()).hexdigest()
                ),
                world_differences=world_differences,
                same_bytes=not len(mismatch),
                mismatch_world_ticks=len(mismatch),
                first_mismatch=None if not len(mismatch) else [int(i) for i in mismatch[0]],
                max_errors=dict(
                    position_m=float(delta[:, :, :3].max()),
                    velocity_m_s=float(delta[:, :, 3:6].max()),
                    gap_m=float(delta[:, :, 6].max()),
                    impulse_n_s=float(delta[:, :, 7].max()),
                ),
                individual_gates={"phymeo": p["physics"], "newton": n["physics"]},
                native=native,
                both_functional=p["all_functional_gates"] and n["all_functional_gates"] and load_ok,
                support_native_force_check=load_ok,
            )
    w32 = comparisons["cuda:0-support"]
    return dict(
        w32_functional_eligible=w32["both_functional"],
        w32_same_output_eligible=w32["same_bytes"] and w32["both_functional"],
        comparisons=comparisons,
    )


def formal(paths, qualification):
    groups = {}
    for file in paths:
        r = unpack(file)
        assert r["device"] == "cuda:0" and r["phase"] == "measure"
        row = r["runs"]["support"]
        assert row["final_tick"] == 209 and row["all_functional_gates"]
        assert row["verified_world_steps"] == 5120 and len(row["samples"]) == 10
        raw = np.load(file.with_name(row["trajectory"]), allow_pickle=False)
        assert raw.shape == (210, 32, 8)
        groups.setdefault(r["engine"], []).append((row, raw))
    assert {k: len(v) for k, v in groups.items()} == {"newton": 3, "phymeo": 3}
    rounds = []
    reference = groups["phymeo"][0][1]
    for (p, a), (n, b) in zip(groups["phymeo"], groups["newton"], strict=True):
        same = bool(
            np.array_equal(a.view(np.uint32), b.view(np.uint32))
            and np.array_equal(reference.view(np.uint32), a.view(np.uint32))
        )
        prior = qualification["comparisons"]["cuda:0-support"]["full_raw_sha256"]
        assert sha256(a[:33].tobytes()).hexdigest() == prior["phymeo"]
        assert sha256(b[:33].tobytes()).hexdigest() == prior["newton"]
        assert p["rows"] == n["rows"]
        metrics = {
            k: {x: v[x] for x in ("p50_ms", "p95_ms", "world_steps_s", "cold", "setup_ns")}
            for k, v in (("phymeo", p), ("newton", n))
        }
        rounds.append(
            dict(
                all_values_same_bits=same,
                max_abs_state_diff=float(np.max(abs(a.astype(float) - b))),
                **metrics,
            )
        )
    fair = qualification["w32_same_output_eligible"] and all(
        r["all_values_same_bits"] for r in rounds
    )
    if fair:
        for r in rounds:
            r["newton_over_phymeo_throughput"] = (
                r["newton"]["world_steps_s"] / r["phymeo"]["world_steps_s"]
            )
    return dict(matched_output_timing=fair, rounds=rounds)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--reports", nargs="+", type=Path, required=True)
    cli.add_argument("--eligibility", type=Path)
    cli.add_argument("--out", type=Path, required=True)
    a = cli.parse_args()
    if a.out.exists():
        cli.error("fresh comparison required")
    result = (
        qualify(a.reports)
        if a.eligibility is None
        else formal(a.reports, json.loads(a.eligibility.read_text()))
    )
    with a.out.open("x", encoding="utf8") as f:
        json.dump(result, f, allow_nan=False, indent=2)
    if "comparisons" in result:
        print(
            {
                k: (v["both_functional"], v["same_bytes"], v["max_errors"])
                for k, v in result["comparisons"].items()
            }
        )
    else:
        print(
            result["matched_output_timing"],
            [(r["phymeo"]["p50_ms"], r["newton"]["p50_ms"]) for r in result["rounds"]],
        )


if __name__ == "__main__":
    main()
