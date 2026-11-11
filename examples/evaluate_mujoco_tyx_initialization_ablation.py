"""One predeclared J initialization ablation, reusing the original learning loop."""

import argparse
import hashlib
import importlib.util
import json
import math
import traceback
from importlib.metadata import version
from pathlib import Path

import numpy as np

from phymeo.training import mujoco_tyx_evaluation as evidence
from phymeo.training.learning_diagnostics import require, strict_json
from phymeo.training.policy_initialization import center_initial_mean
from phymeo.training.policy_response_diagnostics import _features_cuda, _preserved_execution_state

SEEDS = (14011, 14012, 14013)
ANCHOR = "50-n4-summary-j/result/summary.json"
ANCHOR_SHA256 = "50020100fe49b7d6220f766281fcb72c159a810a2d09af2e048661fdeb2e4153"


def learning_api():
    path = Path(__file__).with_name("evaluate_mujoco_tyx_learning.py")
    spec = importlib.util.spec_from_file_location("n4d_original_learning_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path, expected=None):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 4 * 1024**2, "bounded original metadata")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(expected is None or digest == expected, "original metadata hash")
    return strict_json(raw), digest


def load_originals(evidence_root):
    root = Path(evidence_root).resolve()
    anchor, digest = _read_json(root / ANCHOR, ANCHOR_SHA256)
    require(
        anchor.get("task") == "J"
        and anchor.get("seeds") == list(SEEDS)
        and anchor["outcome"]["classification"] == "FIXED_BUDGET_NOT_LEARNED",
        "original J outcome/identities",
    )
    require(
        set(anchor["source_bindings"]) == {str(seed) for seed in SEEDS},
        "complete original seed bindings",
    )
    originals, bindings = {}, {}
    for seed in SEEDS:
        bound = anchor["source_bindings"][str(seed)]
        directory = Path(bound["directory"])
        value, summary_hash = _read_json(directory / "summary.json", bound["summary_sha256"])
        publication, _ = _read_json(directory / "publication.json")
        manifest, manifest_hash = _read_json(directory / "manifest.json")
        require(
            publication["summary_sha256"] == summary_hash
            and value["evidence"] == bound["evidence"],
            "original publication/evidence binding",
        )
        require(
            value["task"] == "J"
            and value["seed"] == seed
            and value["status"] == "COMPLETE_N4A_SEED"
            and value["updates"] == 128
            and value["accepted_world_steps"] == 32768
            and value["worlds"] == 8,
            "original complete J budget",
        )
        require(value["protocol_binding"] == anchor["protocol_binding"], "original fixed task")
        for key in (
            "task",
            "seed",
            "protocol_binding",
            "ppo_config",
            "source_files",
            "dependencies",
        ):
            require(value[key] == manifest[key], "original manifest binding: " + key)
        for phase in ("baseline", "learned"):
            require(
                value[phase]["world_steps"] == 2048 and math.isfinite(value[phase]["cost"]),
                "complete original held-out evaluation",
            )
        originals[seed] = value
        bindings[str(seed)] = {
            "directory": str(directory.resolve()),
            "summary_sha256": summary_hash,
            "manifest_sha256": manifest_hash,
            "evidence": bound["evidence"],
        }
    return {
        "anchor_sha256": digest,
        "anchor_path": str(root / ANCHOR),
        "source_bindings": bindings,
        "scope": "accepted metadata only; old traces not re-audited",
    }, originals


def target_features(protocol, env):
    """Original TYX feature computation at a declared native target; no physics."""
    import torch

    from phymeo.backends.mujoco_tyx import _task_kernels

    width = len(protocol["q_target"])
    q = np.asarray([protocol["q_target"]], np.float32)
    v = np.zeros((1, width), np.float32)
    grid = {"q": q, "v": v, "labels": ("declared-target-q-v0",)}
    revision, frame = env.runtime.checkpoint_revision, env.current_frame
    with _preserved_execution_state(torch) as flags:
        result = _features_cuda(protocol, grid, env.device.index)
    require(
        env.runtime.checkpoint_revision == revision and env.current_frame is frame,
        "target feature generation changed physics/task publication",
    )
    require(
        result["spec_sha256"] == env.task_spec.sha256,
        "actual target feature spec differs from live task",
    )
    result.update(
        native_q=q,
        native_v=v,
        input_origin="declared original task target and zero velocity",
        computation_flags=dict(flags),
        source_revision=revision,
        feature_source={
            "path": str(Path(_task_kernels.__file__).resolve()),
            "sha256": hashlib.sha256(Path(_task_kernels.__file__).read_bytes()).hexdigest(),
        },
    )
    return result


def make_initializer(api, seed, original, output):
    def initialize(session, env, name, supplied_seed):
        import torch
        from tensordict import TensorDict

        require(
            name == "J" and supplied_seed == seed and seed in SEEDS, "fixed initializer identity"
        )
        require(session.next_iteration == 0 and env.num_envs == 8, "fresh eight-world session")
        require(
            api.protocol_binding(name) == original["protocol_binding"], "unchanged original task"
        )
        require(
            json.loads(json.dumps(session.config)) == original["ppo_config"],
            "unchanged original PPO config",
        )
        require(
            all(version(key) == value for key, value in original["dependencies"].items()),
            "original dependency versions",
        )
        require(
            evidence.parameter_hash(session.algorithm) == original["parameter_hash_before"],
            "same original seed actor/critic initialization",
        )
        features = target_features(original["protocol_binding"], env)
        with _preserved_execution_state(torch):
            observation = TensorDict(
                {"policy": torch.from_numpy(features["observation"].copy()).to(env.device)},
                batch_size=[1],
                device=env.device,
            )
        metadata = {
            "task": name,
            "seed": seed,
            "features": features,
            "original_parameter_hash": original["parameter_hash_before"],
            "protocol_binding": original["protocol_binding"],
            "scope": "one initial mean-head bias edit; no task/physics/optimizer step",
            "source_files": {
                str(Path(__file__).resolve()): hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest()
            },
        }
        try:
            receipt = center_initial_mean(
                session.algorithm.actor,
                observation,
                critic=session.algorithm.critic,
                optimizer=session.algorithm.optimizer,
                expected_policy_hash=original["parameter_hash_before"],
            )
        except Exception as error:
            detail = {
                "metadata": metadata,
                "error_type": type(error).__name__,
                "error": str(error),
                "receipt": getattr(error, "receipt", None),
                "rolled_back": getattr(error, "rolled_back", None),
            }
            evidence.json_write(
                Path(output) / ("initializer-failure-" + str(seed) + ".json"),
                evidence.encode(detail),
            )
            raise
        return {"metadata": metadata, "operation": receipt}

    return initialize


def run_ablation(evidence_root, output):
    bindings, originals = load_originals(evidence_root)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    api = learning_api()
    evidence.json_write(
        destination / "protocol.json",
        {
            "schema": "n4d-j-initial-mean-centering-v1",
            "seeds": list(SEEDS),
            "task": "J",
            "updates": 128,
            "rollout": 32,
            "worlds": 8,
            "held_out": 256,
            "factor": "initial-final-mean-linear-bias-only",
            "original_bindings": bindings,
            "original_classification": "FIXED_BUDGET_NOT_LEARNED",
            "stop_rule": "no expanded budget/seed/grid/parameter search after outcomes",
        },
    )
    rows, complete = [], []
    for seed in SEEDS:
        directory = destination / ("J-" + str(seed))
        old = originals[seed]
        row = {
            "seed": seed,
            "original_baseline_cost": old["baseline"]["cost"],
            "original_final_cost": old["learned"]["cost"],
            "original_zero_cost": originals[SEEDS[0]]["zero_action_diagnostic"]["cost"],
        }
        try:
            result = api.run_learning(
                "J", seed, directory, initializer=make_initializer(api, seed, old, destination)
            )
            require("initializer" in result, "original run did not publish initializer receipt")
            row.update(
                status="COMPLETE",
                baseline_cost=result["baseline"]["cost"],
                final_cost=result["learned"]["cost"],
                directory=str(directory.resolve()),
                initialization=result["initializer"],
            )
            complete.append(directory)
        except Exception as error:
            row.update(
                status="FAILED_RETAINED",
                error_type=type(error).__name__,
                error=str(error),
                traceback=traceback.format_exc(),
                directory=str(directory.resolve()),
            )
        rows.append(row)
        evidence.json_write(destination / ("result-" + str(seed) + ".json"), row)
    if len(complete) == len(SEEDS):
        aggregate = api.summarize_task("J", complete, destination / "three-seed")
        classification = aggregate["outcome"]["classification"]
    else:
        aggregate, classification = None, "INCOMPLETE_ABLATION_RETAINED"
    result = {
        "schema": "n4d-j-ablation-comparison-v1",
        "classification": classification,
        "original_classification": "FIXED_BUDGET_NOT_LEARNED",
        "original_bindings": bindings,
        "rows": rows,
        "variant_three_seed": aggregate,
        "scope": "variant gate unchanged; compare absolute costs, not ratio alone",
        "newton_performance_claim": False,
    }
    evidence.json_write(destination / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_ablation(args.evidence_root, args.output)
    print(json.dumps({"classification": result["classification"], "seeds": list(SEEDS)}))
    if result["classification"] != "PASS_FIXED_LEARNING_GATE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
