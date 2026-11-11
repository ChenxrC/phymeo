"""One fixed nine-policy response experiment; no new learning or physics."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from phymeo.training.learning_diagnostics import require, strict_json
from phymeo.training.policy_response_diagnostics import evaluate_pair, read_policy_pair

ANCHORS = (
    ("J", (14011, 14012, 14013), "50-n4-summary-j", "50020100fe49b7d6220f766281fcb72c159a810a2d09af2e048661fdeb2e4153"),
    ("S", (14111, 14112, 14113), "51-n4-summary-s", "60a27613183f0ec96b92e288694dc02a005b62a6388d9765fc7c52f08c6e128d"),
    ("R", (14211, 14212, 14213), "52-n4-summary-r", "bdbf77e984eb0ddbd27f8364110b7aedcdaef7a79b15c5bdab1af86339a2d808"),
)


def _encode(value):
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject and np.isfinite(value).all(), "finite output array")
        raw = value.tobytes(order="C")
        return {"dtype": value.dtype.str, "shape": list(value.shape),
                "bytes_base64": base64.b64encode(raw).decode("ascii"),
                "sha256": hashlib.sha256(raw).hexdigest(), "values": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    return value


def _anchor(root, task, seeds, name, digest):
    path = root / name / "result" / "summary.json"
    require(path.is_file() and path.stat().st_size <= 4 * 1024**2, "bounded original task anchor")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, "original three-seed summary hash")
    value = strict_json(raw)
    require(value.get("schema") == "mujoco-tyx-n4a-three-seed-summary-v1"
            and value.get("task") == task and value.get("seeds") == list(seeds)
            and value["protocol_binding"]["name"] == task
            and set(value.get("source_bindings", {})) == {str(seed) for seed in seeds},
            "fixed task/three-seed identity")
    return value


def diagnose_fixed_policies(evidence_root, output, *, device="cuda:0"):
    root, destination = Path(evidence_root).resolve(), Path(output).resolve()
    require(root.is_dir() and not destination.exists(), "evidence root/fresh output required")
    destination.mkdir(parents=True, exist_ok=False)
    pending = destination / "summary.json.pending"
    phase = "predeclare-nine-identities"
    try:
        selected, anchors = [], {}
        for task, seeds, name, digest in ANCHORS:
            anchor = _anchor(root, task, seeds, name, digest)
            anchors[task] = {"path": str(root / name / "result" / "summary.json"),
                             "sha256": digest, "learning_outcome_unchanged": anchor["outcome"]}
            for seed in seeds:
                binding = anchor["source_bindings"][str(seed)]
                directory = Path(binding["directory"]).resolve()
                require(directory.is_relative_to(root), "bound seed outside original evidence root")
                selected.append((task, seed, directory, binding, anchor["protocol_binding"]))
        require(len({str(item[2]) for item in selected}) == 9, "nine unique seed directories")
        pairs, common = [], None
        for task, seed, directory, binding, protocol in selected:
            phase = f"extract-{task}-{seed}"
            pair = read_policy_pair(directory, binding, expected_task=task, expected_seed=seed)
            require(pair["protocol"] == protocol, "policy differs from anchored task protocol")
            signature = {name: pair[name] for name in ("source_files", "dependencies", "ppo_config")}
            if common is None:
                common = signature
            require(signature == common, "nine-run source/dependency/config mismatch")
            pairs.append(pair)
        runs = []
        for pair in pairs:
            phase = f"response-{pair['task']}-{pair['seed']}"
            run = evaluate_pair(pair, device=device)
            run["interpretation"] = {
                "J": "uncoupled direct-force restoring direction is diagnostic, not learning acceptance",
                "S": "nonzero equilibrium control is legal; no new contact or learning test",
                "R": "full coupled response is descriptive, not a per-joint torque-sign gate",
            }[pair["task"]]
            runs.append(run)
        result = {
            "schema": "mujoco-tyx-n4c-fixed-policy-response-v1",
            "status": "PASS_FIXED_POLICY_RESPONSE_ONLY", "runs": runs, "anchors": anchors,
            "fixed_grid": "original eight starts, target/v0, each q +/-qscale/4, each v +/-vscale/4",
            "execution": {"training_updates": 0, "physics_steps": 0,
                          "feature_kernel_calls": 9, "deterministic_actor_calls": 18},
            "scope": "new finite input inference on original saved models; no new learning gate or causal proof",
            "source_sha256": {
                str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                str(Path(read_policy_pair.__code__.co_filename).resolve()): hashlib.sha256(
                    Path(read_policy_pair.__code__.co_filename).read_bytes()).hexdigest(),
            },
        }
        phase = "atomic-response-publication"
        raw = (json.dumps(_encode(result), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        with pending.open("xb") as stream:
            stream.write(raw)
        require(pending.read_bytes() == raw, "response output readback")
        pending.rename(destination / "summary.json")
        return result
    except Exception as error:
        if pending.exists():
            pending.unlink()
        with (destination / "failure.json").open("x", encoding="utf-8") as stream:
            json.dump({"status": "FAILED_POLICY_RESPONSE_NOT_PUBLISHED", "phase": phase,
                       "error_type": type(error).__name__, "error": str(error),
                       "scope": "original evidence unchanged; no optimizer/physical step"},
                      stream, indent=2, allow_nan=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = diagnose_fixed_policies(args.evidence_root, args.output, device=args.device)
    print(json.dumps({"status": result["status"], "runs": len(result["runs"])}))


if __name__ == "__main__":
    main()
