"""Diagnose the frozen nine N4a runs without rerunning policies or physics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from phymeo.training.learning_diagnostics import diagnose_seed, require, strict_json

# Predeclared closed three-seed summaries, read before any payload analysis.
ANCHORS = (
    ("J", (14011, 14012, 14013), "50-n4-summary-j",
     "50020100fe49b7d6220f766281fcb72c159a810a2d09af2e048661fdeb2e4153"),
    ("S", (14111, 14112, 14113), "51-n4-summary-s",
     "60a27613183f0ec96b92e288694dc02a005b62a6388d9765fc7c52f08c6e128d"),
    ("R", (14211, 14212, 14213), "52-n4-summary-r",
     "bdbf77e984eb0ddbd27f8364110b7aedcdaef7a79b15c5bdab1af86339a2d808"),
)


def _anchor(root, task, seeds, name, expected_hash):
    path = root / name / "result" / "summary.json"
    require(path.is_file() and path.stat().st_size <= 4*1024**2, "bounded original anchor missing")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_hash, "original summary anchor hash")
    value = strict_json(raw)
    require(value.get("schema") == "mujoco-tyx-n4a-three-seed-summary-v1"
            and value.get("task") == task and value.get("seeds") == list(seeds)
            and set(value.get("source_bindings", {})) == {str(seed) for seed in seeds},
            "exact three-seed identity")
    require(value["protocol_binding"]["name"] == task, "task protocol identity")
    return value


def _compare(learned, reference):
    q = np.asarray(learned["q_cost_per_world_joint"])
    v = np.asarray(learned["v_cost_per_world_joint"])
    qr = np.asarray(reference["q_cost_per_world_joint"])
    vr = np.asarray(reference["v_cost_per_world_joint"])
    require(q.shape == qr.shape and v.shape == vr.shape
            and learned["world_keys"] == reference["world_keys"], "comparison world/coordinate layout")
    return {
        "cost_difference": learned["cost"] - reference["cost"],
        "per_world_cost_difference": (
            np.asarray(learned["per_world_cost"]) - reference["per_world_cost"] ).tolist(),
        "q_cost_difference_per_world_joint": (q - qr).tolist(),
        "v_cost_difference_per_world_joint": (v - vr).tolist(),
        "quarter_cost_difference_per_world": (
            np.asarray(learned["quarter_world_cost"]) - reference["quarter_world_cost"]).tolist(),
        "scope": "same fixed held-out states; saved behavior difference, not causal attribution",
    }


def diagnose_n4a(evidence_root, output):
    root, destination = Path(evidence_root).resolve(), Path(output).resolve()
    require(root.is_dir() and not destination.exists(), "original evidence root/fresh output required")
    destination.mkdir(parents=True, exist_ok=False)
    pending = destination / "summary.json.pending"
    phase = "anchor-preflight"
    try:
        # All nine identities/budgets are fixed before opening the first gzip.
        selected, anchors = [], {}
        for task, seeds, name, digest in ANCHORS:
            anchor = _anchor(root, task, seeds, name, digest)
            anchors[task] = {"path": str(root/name/"result"/"summary.json"),
                             "sha256": digest, "learning_outcome_unchanged": anchor["outcome"]}
            for seed in seeds:
                binding = anchor["source_bindings"][str(seed)]
                directory = Path(binding["directory"]).resolve()
                require(directory.is_relative_to(root), "bound seed outside declared evidence root")
                selected.append((task, seed, directory, binding, anchor["protocol_binding"]))
        require(len({str(item[2]) for item in selected}) == 9, "duplicate seed directory")
        runs, common = [], None
        for task, seed, directory, binding, protocol in selected:
            phase = f"stream-{task}-{seed}"
            run = diagnose_seed(directory, binding, expected_task=task, expected_seed=seed)
            require(run["protocol_binding"] == protocol and len(run["world_keys"]) == 8,
                    "run differs from fixed task anchor")
            signature = {key: run[key] for key in ("source_files", "dependencies", "ppo_config")}
            if common is None:
                common = signature
            require(signature == common, "nine-run source/dependency/learner configuration mismatch")
            runs.append(run)
        comparisons = {}
        for task, seeds, _, _ in ANCHORS:
            task_runs = [run for run in runs if run["task"] == task]
            require([run["seed"] for run in task_runs] == list(seeds), "complete fixed seed order")
            zero = task_runs[0]["phases"].get("zero")
            require(zero is not None and all("zero" not in run["phases"] for run in task_runs[1:]),
                    "one complete first-seed zero diagnostic required")
            comparisons[task] = [{
                "seed": run["seed"],
                "learned_minus_untrained": _compare(run["phases"]["learned"],
                                                    run["phases"]["untrained"]),
                "learned_minus_zero": _compare(run["phases"]["learned"], zero),
            } for run in task_runs]
        result = {
            "schema": "mujoco-tyx-n4b-saved-diagnosis-v1",
            "status": "PASS_SAVED_DIAGNOSIS_ONLY",
            "anchors": anchors, "runs": runs, "comparisons": comparisons,
            "diagnostic_scope": {
                "objective": "original q and velocity terms, native endpoint states/rewards",
                "actions": "saved raw action and actual ctrl; 95pct is a diagnostic bin, not a gate",
                "updates": "actual post-clip parameter norms/Adam/loss/model tensors only",
                "timeouts": "actual 256-step resets; no fabricated value/return/bootstrap estimates",
                "support": "S positive control uses actual last-step contact forces",
                "prohibited": "no policy forward, retraining, simulation, target/budget/tolerance change",
            },
            "source_sha256": {
                str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                str(Path(diagnose_seed.__code__.co_filename).resolve()):
                    hashlib.sha256(Path(diagnose_seed.__code__.co_filename).read_bytes()).hexdigest(),
            },
        }
        phase = "diagnosis-publication"
        raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        with pending.open("xb") as stream:
            stream.write(raw)
        require(hashlib.sha256(pending.read_bytes()).digest() == hashlib.sha256(raw).digest(),
                "diagnosis write readback")
        pending.rename(destination/"summary.json")
        return result
    except Exception as error:
        if pending.exists():
            pending.unlink()
        # A rejected input never produces a successful summary or partial metric file.
        with (destination/"failure.json").open("x", encoding="utf-8") as stream:
            json.dump({"status": "SAVED_DIAGNOSIS_REJECTED", "phase": phase,
                       "error_type": type(error).__name__, "error": str(error)},
                      stream, indent=2, allow_nan=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    result = diagnose_n4a(options.evidence_root, options.output)
    print(json.dumps({"status": result["status"], "runs": len(result["runs"])}))


if __name__ == "__main__":
    main()
