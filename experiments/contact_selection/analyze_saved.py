"""Inspect saved Kamino trajectories once; no engine or model imports."""

import argparse
import gzip
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def analyze(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError("analysis output must be new")
    summary = read(root / "summary.json")
    reports = []
    for case in summary["cases"]:
        timings = {"loaded": [], "unloaded": []}
        totals = {"loaded": [], "unloaded": []}
        frames = {
            f["tick"]: f for f in read(root / case["frames"])["frames"] if f["contact_data_valid"]
        }
        matched, count = 0, 0
        with gzip.open(root / case["file"], "rt", encoding="utf-8") as stream:
            for count, line in enumerate(stream, start=1):
                r = json.loads(line)
                assert r["input_tick"] == count - 1 and r["output_tick"] == count
                assert r["phase"] == "step-input-contact"
                assert r["raw_contacts"]["overflow"] == 0
                assert all(s["converged"] for s in r["solver_status"])
                if r["input_tick"] in frames:
                    f = frames[r["input_tick"]]
                    assert f["body_pose"] == r["entry"]["body_q"][0]
                    assert f["contact_force_on_box"] == r["contact_force_on_box"]
                    assert f["input_force"] == r["external_body_wrench"][0][:3]
                    assert f["input_torque"] == r["external_body_wrench"][0][3:]
                    matched += 1
                if r["input_time"] >= 0.1:
                    name = "loaded" if r["active_contacts"] else "unloaded"
                    t = r["timing"]
                    timings[name].append(t["step_s"] * 1000)
                    totals[name].append(
                        (t["step_s"] + t["input_preparation_s"] + t["observation_s"]) * 1000
                    )
        assert count == case["accepted_steps"] and matched == len(frames)
        reports.append(
            {
                "id": case["id"],
                "steps": count,
                "matching_contact_frames": matched,
                "phase_costs": {
                    name: {
                        "samples": len(v),
                        "step_mean_ms": statistics.mean(v) if v else None,
                        "step_median_ms": statistics.median(v) if v else None,
                        "step_with_required_reads_mean_ms": statistics.mean(totals[name])
                        if v
                        else None,
                    }
                    for name, v in timings.items()
                },
            }
        )
    output.write_text(
        json.dumps({"source": str(root), "mode": "saved_records_only", "cases": reports}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(reports))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    analyze(a.input, a.output)
