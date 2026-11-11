"""Paired full/integration observation through real, bounded CUDA PPO updates.

The physics checks are identical; endpoint-only diagnostic outputs differ.
This is an explicit-output-contract measurement, not a Newton speed comparison.
"""

import argparse
import importlib.util
import json
import traceback
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from phymeo.training import mujoco_tyx_evaluation as evidence

STATE_FIELDS = (
    "world",
    "q",
    "v",
    "act",
    "ctrl",
    "time",
    "tick",
    "qfrc_applied",
    "xfrc_applied",
    "held_control",
    "last_step",
    "coordinate_system",
)
TRANSITION_FIELDS = (
    "sequence",
    "world_keys",
    "observation",
    "reward",
    "terminated",
    "time_outs",
    "final_observation",
    "final_mask",
    "episode_id",
    "episode_length",
    "episode_return",
    "action",
    "native_channels",
    "accepted_worlds",
)


def canonical(value):
    """Map order is not state; numeric representation and sequence order are."""
    if isinstance(value, dict):
        if set(value) == {"mapping"}:
            pairs = [[canonical(k), canonical(v)] for k, v in value["mapping"]]
            return {"mapping": sorted(pairs, key=lambda pair: json.dumps(pair[0], sort_keys=True))}
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    return value


def digest(value):
    encoded = canonical(evidence.encode(value))
    return sha256(
        json.dumps(encoded, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def common_update(record):
    """Explicit comparison view; both original complete records are also saved."""
    transitions = []
    for entry in record["transitions"]:
        selected = {name: getattr(entry, name) for name in TRANSITION_FIELDS}
        selected["states"] = {
            key: {name: getattr(state, name) for name in STATE_FIELDS}
            for key, state in entry.states.items()
        }
        for frame_name in ("input_frame", "output_frame"):
            frame = getattr(entry, frame_name)
            selected[frame_name] = {
                name: getattr(frame, name) for name in ("revision", "epoch", "ticks", "times")
            }
        transitions.append(selected)
    return {
        "transitions": tuple(transitions),
        **{
            key: record[key]
            for key in (
                "iteration",
                "next_iteration",
                "loss",
                "optimization",
                "recurrent_masks",
                "step_requests",
            )
        },
    }


def learner_state(session):
    return {
        "actor": session.algorithm.actor.state_dict(),
        "critic": session.algorithm.critic.state_dict(),
        "optimizer": session.algorithm.optimizer.state_dict(),
    }


def task_api():
    spec = importlib.util.spec_from_file_location(
        "paired_observation_learning_api",
        Path(__file__).with_name("evaluate_mujoco_tyx_learning.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_pair(name, worlds, output):
    from phymeo.training.mujoco_tyx_checkpoint import MuJoCoTyxPpoSession
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxVectorEnv
    from phymeo.training.rsl_mujoco_tyx import RslMuJoCoTyxVecEnv

    if name not in ("J", "S", "R") or worlds not in (8, 128, 256):
        raise ValueError("predeclared J/S/R and W8/128/256 only")
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    api = task_api()
    tasks = api.tasks()
    torch = tasks.require_cuda()
    item = tasks.task(name)
    seed = item.seeds[0]
    order = ("full", "integration") if worlds != 128 else ("integration", "full")
    rows, reference = {}, None
    evidence.json_write(
        target / "protocol.json",
        {
            "schema": "paired-observation-ppo-v1",
            "task": name,
            "worlds": worlds,
            "seed": seed,
            "order": order,
            "warmup_updates": 1,
            "measured_updates": 2,
            "steps_per_update": 16,
            "original_task": api.protocol_binding(name),
            "comparison": "all common physical fields, transitions, policy inputs, actor/critic/Adam/RNG",
            "different_outputs": "full includes endpoint geometry/contact/sensor/energy; integration does not",
            "newton_performance_comparison": False,
            "source_files": {
                **api._sources(),
                str(Path(__file__).resolve()): sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        },
    )
    try:
        for mode in order:
            writer = evidence.EvidenceWriter(target / (mode + ".jsonl.gz"))
            signatures, timings, checks = [], [], []
            start_cold = perf_counter()
            try:
                with ExitStack() as owners:
                    base = owners.enter_context(
                        tasks.make_environment(name, count=worlds, scale=True)
                    )
                    runtime = base.runtime
                    # Release the unused task workspace, retaining its borrowed runtime.
                    base.close()
                    env = owners.enter_context(
                        MuJoCoTyxVectorEnv(
                            runtime,
                            base.task_spec,
                            base.action_spec,
                            h=base.h,
                            horizons=tuple(int(x) for x in base.horizons),
                            observation_mode=mode,
                        )
                    )
                    tasks.seed_all(seed)
                    session = MuJoCoTyxPpoSession(RslMuJoCoTyxVecEnv(env), steps=16)
                    initial = learner_state(session)
                    initial_hash = digest(initial)
                    writer.write("initial_policy", initial)
                    initial_observation = env.get_observations()
                    observation_hash = digest(initial_observation)
                    writer.write("initial_observation", initial_observation)
                    writer.write(
                        "bindings",
                        {
                            "environment": env.binding,
                            "config": session.config,
                            "runtime": runtime.capabilities,
                            "mode": mode,
                        },
                    )
                    cold_seconds = perf_counter() - start_cold
                    measured_start = None
                    for iteration in range(3):
                        if iteration == 1:
                            measured_start = perf_counter()
                            write_start = writer.write_seconds
                        record, timing = evidence.measure_update(session, observe=False)
                        checked = evidence.validate_update(
                            record,
                            env,
                            expected_iteration=iteration,
                            cost=item.cost(),
                            writer=writer,
                        )
                        if len(record["transitions"]) != 16:
                            raise ValueError("rollout count differs")
                        learned = learner_state(session)
                        rng = {
                            "cpu": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state(env.device),
                        }
                        signatures.append(
                            {
                                "iteration": iteration,
                                "common": digest(common_update(record)),
                                "learner": digest(learned),
                                "rng": digest(rng),
                            }
                        )
                        writer.write("learner_and_rng", {"learner": learned, "rng": rng})
                        writer.write("measurement", timing)
                        if iteration:
                            timings.append(timing)
                            checks.append(checked)
                    if session.next_iteration != 3:
                        raise ValueError("actual complete updates differ")
                    counters = evidence.plain_counters(env.counters)
                writer.close()
                # Includes complete measured updates, their evidence and owner close.
                wall = perf_counter() - measured_start
                row = {
                    "initial_policy_hash": initial_hash,
                    "signatures": signatures,
                    "initial_observation_hash": observation_hash,
                    "cold_environment_learner_s": cold_seconds,
                    "measured_updates": timings,
                    "checks": checks,
                    "measured_world_steps": 32 * worlds,
                    "compute_and_mandatory_diagnostics_s": sum(
                        entry["compute_and_mandatory_diagnostics_s"] for entry in timings
                    ),
                    "loop_to_trace_close_s": wall,
                    "evidence_write_s": writer.write_seconds - write_start,
                    "transitions_per_s_with_trace": 32 * worlds / wall,
                    "counter_end": counters,
                    "evidence": writer.receipt(),
                }
                evidence.json_write(target / (mode + "-summary.json"), row)
                rows[mode] = row
                comparable = {
                    "initial": initial_hash,
                    "observation": observation_hash,
                    "updates": signatures,
                }
                if reference is None:
                    reference = comparable
                elif comparable != reference:
                    raise AssertionError(
                        "paired physics/learning/optimizer/RNG differs; inspect saved signatures"
                    )
            except evidence.UpdateCaptureError as error:
                writer.write("failed_update_attempts", error.attempts)
                writer.write("failed_update_record", error.completed_record)
                raise
            finally:
                writer.close()
        summary = {
            "status": "PASS_PAIRED_PPO_OBSERVATION",
            "task": name,
            "worlds": worlds,
            "seed": seed,
            "order": order,
            "modes": rows,
            "measured_pairs": 2,
            "warmup_pairs": 1,
            "physics_and_learner_identical": True,
            "learning_effectiveness_claim": False,
            "different_observation_outputs": True,
            "newton_performance_claim": False,
        }
        evidence.json_write(target / "summary.json", summary)
        return summary
    except BaseException as error:
        evidence.json_write(
            target / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "completed_modes": list(rows),
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("J", "S", "R"))
    parser.add_argument("--worlds", required=True, type=int, choices=(8, 128, 256))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = run_pair(args.task, args.worlds, args.output)
    print(json.dumps({key: summary[key] for key in ("status", "task", "worlds")}))


if __name__ == "__main__":
    main()
