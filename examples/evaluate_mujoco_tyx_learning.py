"""Frozen N4a preflight, learning, held-out evaluation and end-to-end scale entry.

Use the qualified CUDA profile. Each command owns a fresh output directory.
No command silently repeats preflight or replaces a failed seed/cell.
"""

import argparse
import importlib.util
import json
import math
import platform
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
from importlib import metadata
from pathlib import Path

import numpy as np

from phymeo.training import mujoco_tyx_evaluation as evaluation


def tasks():
    name = "phymeo_n4a_frozen_tasks"
    if name not in sys.modules:
        path = Path(__file__).with_name("mujoco_tyx_learning_tasks.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


def protocol_binding(name):
    definitions = tasks()
    value = asdict(definitions.task(name))
    value["xml_sha256"] = sha256(value["xml"].encode()).hexdigest()
    value.update(
        schema="mujoco-tyx-n4a-protocol-v1",
        training_templates=definitions.templates(name, held_out=False),
        held_out_templates=definitions.templates(name, held_out=True),
        updates=128,
        rollout=32,
        learning_worlds=8,
        horizon=256,
        action_squash="tanh",
        decimation=1,
        evaluation="all-256-pre-reset-states-deterministic-actor",
        zero_baseline="first-predeclared-seed-only-diagnostic",
        quality_gate={"max_ratio": 1.05, "at_least_two_ratio": 0.90, "median_ratio": 0.80},
    )
    # Canonical JSON types allow exact comparison with independently loaded summaries.
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _sources():
    paths = {
        Path(__file__).resolve(),
        Path(tasks().__file__).resolve(),
        Path(evaluation.__file__).resolve(),
    }
    # Bind the same full package source set regardless of optional modules loaded
    # by earlier independent CLI/pytest cases in the process.
    import phymeo

    paths.update(Path(phymeo.__file__).resolve().parent.rglob("*.py"))
    return {str(path): sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def _manifest(env, name, seed, phase):
    import torch

    versions = {
        key: metadata.version(key)
        for key in ("numpy", "torch", "rsl-rl-lib", "tensordict", "mujoco", "tyx-compute")
    }
    return {
        "schema": "mujoco-tyx-n4a-run-v1",
        "phase": phase,
        "task": name,
        "seed": seed,
        "protocol_binding": protocol_binding(name),
        "source_files": _sources(),
        "dependencies": versions,
        "python": platform.python_version(),
        "device": {
            "name": torch.cuda.get_device_name(env.device),
            "index": env.device.index,
            "torch_cuda": torch.version.cuda,
        },
        "world_keys": list(env.world_keys),
        "env_binding": evaluation.encode(env.binding),
        "capabilities": evaluation.encode(env.runtime.capabilities),
        "counter_start": evaluation.plain_counters(env.counters),
        "measurement_coverage": evaluation.COVERAGE,
    }


@contextmanager
def _output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    writer = evaluation.EvidenceWriter(path / "trace.jsonl.gz")
    context = {"path": path, "writer": writer, "phase": "construct", "env": None}
    try:
        yield context
    except BaseException as error:
        detail = {
            "status": "FAIL_N4A_RETAINED",
            "phase": context["phase"],
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        for key in (
            "phase",
            "action_accepted",
            "state_accepted",
            "needs_recovery",
            "accepted_worlds",
        ):
            if hasattr(error, key):
                detail["error_" + key] = evaluation.encode(getattr(error, key))
        if isinstance(error, evaluation.UpdateCaptureError):
            writer.write("failed_update_complete_attempts", error.attempts)
            detail["failed_update_compute_s"] = error.compute_seconds
            detail["original_error_type"] = error.original_type
        env = context["env"]
        if env is not None:
            try:
                writer.write("failure.last_transition", env.last_transition)
                detail["counter_end"] = evaluation.plain_counters(env.counters)
            except Exception as recording_error:
                detail["secondary_recording_error"] = repr(recording_error)
        try:
            writer.write("failure", detail)
        finally:
            writer.close()
        detail["evidence"] = writer.receipt()
        evaluation.json_write(path / "failure.json", detail)
        raise
    finally:
        writer.close()


def _finish(context, result):
    writer = context["writer"]
    writer.write("completion", {"status": result["status"], "task": result["task"]})
    writer.close()
    hash_started = time.perf_counter()
    result["evidence"] = writer.receipt()
    result["final_evidence_hash_io_s"] = time.perf_counter() - hash_started
    if "scale_write_start" in context:
        timing = result["timing"]
        timing["evidence_write_s"] = writer.write_seconds - context["scale_write_start"]
        timing["transitions_per_s_with_full_trace_delivery"] = result["measured_world_steps"] / (
            timing["compute_and_mandatory_diagnostics_s"]
            + timing["post_update_independent_audit_s"]
            + timing["evidence_write_s"]
        )
        timing.pop("transitions_per_s_with_current_audit_and_write", None)
    summary_started = time.perf_counter()
    evaluation.json_write(context["path"] / "summary.json", result)
    evaluation.json_write(
        context["path"] / "publication.json",
        {
            "summary_encode_write_s": time.perf_counter() - summary_started,
            "scope": "summary publication and final file hash are outside training/trace throughput",
            "summary_sha256": sha256((context["path"] / "summary.json").read_bytes()).hexdigest(),
        },
    )
    return result


class SupportInspection:
    """Read actual force rows used in the accepted step, not endpoint CPU contacts."""

    def __init__(self, count):
        self.seen = np.zeros(count, bool)
        self.positive = np.zeros(count, np.int64)
        self.total = np.zeros(count, np.float64)
        self.tail_steps = 0
        self.peak = 0.0
        self.max_geometric_overlap = 0.0

    def __call__(self, tick, transition):
        rows = []
        for i, key in enumerate(transition.world_keys):
            state = transition.states[key]
            receipt = state.last_step
            if receipt is None or receipt.backend != "mujoco-tyx-cuda":
                raise ValueError("support inspection requires actual TYX step force receipt")
            contacts = tuple(row for row in receipt.contacts if set(row.geometry_ids) == {0, 1})
            positive = [row for row in contacts if row.active and row.normal_force > 0]
            load = float(sum(row.normal_force for row in positive))
            if not math.isfinite(load) or len(positive) > 4:
                raise ValueError(
                    "support force/count does not match declared four-row native model"
                )
            self.seen[i] |= bool(positive)
            self.peak = max(self.peak, load)
            geometric_in, geometric_out = (
                float(receipt.input_q[0]) - 0.001,
                float(state.q[0]) - 0.001,
            )
            self.max_geometric_overlap = max(
                self.max_geometric_overlap, -geometric_in, -geometric_out
            )
            if tick >= 65:
                self.positive[i] += int(len(positive) == 4)
                self.total[i] += load
            rows.append(
                {
                    "world_key": key,
                    "positive_rows": len(positive),
                    "normal_force_N": load,
                    "native_distances_m": tuple(row.distance for row in contacts),
                    "geometric_input_gap_m": geometric_in,
                    "geometric_output_gap_m": geometric_out,
                    "input_time": receipt.input_time,
                    "output_time": receipt.output_time,
                    "contact_time": receipt.contact_time,
                }
            )
        if tick >= 65:
            self.tail_steps += 1
        return rows

    def summary(self, *, require_support):
        result = {
            "contact_seen": self.seen.tolist(),
            "tail_steps": self.tail_steps,
            "peak_normal_force_N": self.peak,
            "maximum_geometric_overlap_m": self.max_geometric_overlap,
        }
        if self.tail_steps:
            fraction, mean = self.positive / self.tail_steps, self.total / self.tail_steps
            result.update(
                four_positive_fraction=fraction.tolist(),
                mean_normal_force_N=mean.tolist(),
                mean_weight_ratio=(mean / (2 * 9.81)).tolist(),
            )
            if require_support and (
                np.any(fraction < 0.95)
                or np.any(mean < 0.5 * 2 * 9.81)
                or np.any(mean > 3 * 2 * 9.81)
            ):
                raise ValueError("fixed continuous-support evaluation gate failed")
        elif require_support:
            raise ValueError("support tail is absent")
        return result


def run_preflight(name, output):
    definitions = tasks()
    item = definitions.task(name)
    with _output(output) as context:
        definitions.seed_all(item.seeds[0])
        with definitions.make_environment(name) as env:
            context.update(env=env, phase="fixed-first-32-step-preflight")
            manifest = _manifest(env, name, item.seeds[0], "preflight")
            evaluation.json_write(context["path"] / "manifest.json", manifest)
            inspect = SupportInspection(env.num_envs) if name == "S" else None
            torch = definitions.require_cuda()

            def zero(_):
                return torch.zeros((env.num_envs, env.num_actions), device=env.device)

            outcome = evaluation.evaluate_window(
                env,
                zero,
                item.cost(),
                steps=32,
                writer=context["writer"],
                label="preflight",
                inspect=inspect,
            )
            support = {} if inspect is None else inspect.summary(require_support=False)
            if inspect is not None and not all(support["contact_seen"]):
                raise ValueError(
                    "fixed support preflight did not detect actual positive force in every world"
                )
            result = {
                "status": "PASS_N4A_PREFLIGHT",
                "task": name,
                "worlds": 8,
                "steps": 32,
                "accepted_world_steps": outcome["world_steps"],
                "objective": outcome,
                "counter_end": evaluation.plain_counters(env.counters),
                **support,
            }
        context["env"] = None
        return _finish(context, result)


def _held_out(actor, name, writer, label, *, zero=False):
    definitions = tasks()
    item = definitions.task(name)
    torch = definitions.require_cuda()
    from tensordict import TensorDict

    with evaluation.preserved_rng():
        modes = tuple((module, module.training) for module in actor.modules())
        try:
            actor.eval()
            with definitions.make_environment(name, held_out=True) as env:
                inspect = SupportInspection(env.num_envs) if name == "S" else None

                def action(observation):
                    if zero:
                        return torch.zeros((env.num_envs, env.num_actions), device=env.device)
                    return actor(
                        TensorDict(
                            {"policy": observation}, batch_size=[env.num_envs], device=env.device
                        ),
                        stochastic_output=False,
                    )

                result = evaluation.evaluate_window(
                    env, action, item.cost(), steps=256, writer=writer, label=label, inspect=inspect
                )
                if inspect is not None:
                    result["support"] = inspect.summary(require_support=not zero)
                result["counter_end"] = evaluation.plain_counters(env.counters)
                return result
        finally:
            for module, mode in modes:
                module.training = mode


def _session(env):
    from phymeo.training.mujoco_tyx_checkpoint import MuJoCoTyxPpoSession
    from phymeo.training.rsl_mujoco_tyx import RslMuJoCoTyxVecEnv

    return MuJoCoTyxPpoSession(RslMuJoCoTyxVecEnv(env), steps=32)


def run_learning(name, seed, output):
    definitions = tasks()
    item = definitions.task(name)
    if type(seed) is not int or seed not in item.seeds:
        raise ValueError("seed is not in the predeclared three-seed task")
    with _output(output) as context:
        definitions.seed_all(seed)
        started = time.perf_counter()
        with definitions.make_environment(name) as env:
            context.update(env=env, phase="construct-original-upstream-PPO")
            session = _session(env)
            manifest = _manifest(env, name, seed, "learn")
            manifest["ppo_config"] = session.config
            evaluation.json_write(context["path"] / "manifest.json", manifest)
            writer = context["writer"]
            writer.write(
                "initial_policy",
                {
                    "actor": session.algorithm.actor.state_dict(),
                    "critic": session.algorithm.critic.state_dict(),
                },
            )
            initial_hash = evaluation.parameter_hash(session.algorithm)
            context["phase"] = "untrained-deterministic-held-out-256"
            baseline = _held_out(session.algorithm.actor, name, writer, "untrained")
            zero = None
            if seed == item.seeds[0]:
                context["phase"] = "zero-action-diagnostic-held-out-256"
                try:
                    zero = _held_out(session.algorithm.actor, name, writer, "zero", zero=True)
                    zero["status"] = "COMPLETE_DIAGNOSTIC"
                except Exception as error:
                    zero = {
                        "status": "FAIL_DIAGNOSTIC",
                        "cost": None,
                        "world_steps": None,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "scope": "complete available prefix retained; no shortened cost",
                    }
                    writer.write("zero.failure", zero)
            if evaluation.parameter_hash(session.algorithm) != initial_hash:
                raise ValueError("evaluation mutated actor/critic parameters")
            updates, accepted = [], 0
            for index in range(definitions.UPDATES):
                context["phase"] = "upstream-update-" + str(index)
                before = evaluation.parameter_hash(session.algorithm)
                record, measurement = evaluation.measure_update(session)
                if len(record["transitions"]) != definitions.ROLLOUT:
                    raise ValueError("rollout length differs from fixed budget")
                checked = evaluation.validate_update(
                    record, env, expected_iteration=index, cost=item.cost(), writer=writer
                )
                after = evaluation.parameter_hash(session.algorithm)
                if before == after:
                    raise ValueError("actual update left every actor/critic parameter unchanged")
                row = {
                    "iteration": index,
                    "parameter_hash_before": before,
                    "parameter_hash_after": after,
                    **checked,
                    "measurement": measurement,
                }
                updates.append(row)
                accepted += checked["accepted_world_steps"]
                writer.write("update_checks", row)
            if accepted != 32768 or session.next_iteration != 128:
                raise ValueError("complete frozen learning budget was not consumed")
            final_hash = evaluation.parameter_hash(session.algorithm)
            context["phase"] = "trained-deterministic-held-out-256"
            learned = _held_out(session.algorithm.actor, name, writer, "learned")
            writer.write(
                "final_policy_and_adam",
                {
                    "actor": session.algorithm.actor.state_dict(),
                    "critic": session.algorithm.critic.state_dict(),
                    "optimizer": session.algorithm.optimizer.state_dict(),
                    "next_iteration": session.next_iteration,
                },
            )
            result = {
                "status": "COMPLETE_N4A_SEED",
                "learning_quality": "THREE_SEED_SUMMARY_REQUIRED",
                "task": name,
                "seed": seed,
                "updates": 128,
                "worlds": 8,
                "accepted_world_steps": accepted,
                "protocol_binding": protocol_binding(name),
                "ppo_config": session.config,
                "baseline": baseline,
                "learned": learned,
                "zero_action_diagnostic": zero,
                "parameter_hash_before": initial_hash,
                "parameter_hash_after": final_hash,
                "update_evidence": updates,
                "counter_end": evaluation.plain_counters(env.counters),
                "elapsed_including_evaluation_and_evidence_s": time.perf_counter() - started,
                "source_files": manifest["source_files"],
                "dependencies": manifest["dependencies"],
            }
        context["env"] = None
        return _finish(context, result)


def run_scale(name, seed, worlds, output):
    definitions = tasks()
    item = definitions.task(name)
    if seed != item.seeds[0] or type(worlds) is not int or worlds not in definitions.SCALE_WORLDS:
        raise ValueError("scale cell must use the fixed task seed and declared world count")
    with _output(output) as context:
        definitions.seed_all(seed)
        cold_started = time.perf_counter()
        with definitions.make_environment(name, count=worlds, scale=True) as env:
            context.update(env=env, phase="scale-construction")
            session = _session(env)
            definitions.require_cuda().cuda.synchronize(env.device)
            cold_seconds = time.perf_counter() - cold_started
            manifest = _manifest(env, name, seed, "scale")
            manifest["ppo_config"] = session.config
            evaluation.json_write(context["path"] / "manifest.json", manifest)
            writer = context["writer"]
            context["phase"] = "one-warmup-with-bounded-observer"
            record, warmup = evaluation.measure_update(session, observe=True)
            if len(record["transitions"]) != 32:
                raise ValueError("warmup length changed")
            warmup_checks = evaluation.validate_update(
                record, env, expected_iteration=0, cost=item.cost(), writer=writer
            )
            writer.write("warmup_measurement", warmup)
            warmup_write_seconds = writer.write_seconds
            context["scale_write_start"] = warmup_write_seconds
            measured, accepted = [], 0
            for index in range(1, 4):
                context["phase"] = "unprofiled-measured-update-" + str(index)
                record, timing = evaluation.measure_update(session, observe=False)
                if len(record["transitions"]) != 32:
                    raise ValueError("measured rollout length changed")
                audit_started = time.perf_counter()
                write_before = writer.write_seconds
                checked = evaluation.validate_update(
                    record, env, expected_iteration=index, cost=item.cost(), writer=writer
                )
                extra_audit_seconds = (
                    time.perf_counter() - audit_started - (writer.write_seconds - write_before)
                )
                row = {
                    "iteration": index,
                    **checked,
                    **timing,
                    "post_update_independent_audit_s": max(0.0, extra_audit_seconds),
                }
                writer.write("measured_update_checks", row)
                accepted += checked["accepted_world_steps"]
                measured.append(row)
            if accepted != 96 * worlds or session.next_iteration != 4:
                raise ValueError("complete scale budget was not consumed")
            compute = sum(row["compute_and_mandatory_diagnostics_s"] for row in measured)
            audits = sum(row["post_update_independent_audit_s"] for row in measured)
            write_seconds = writer.write_seconds - warmup_write_seconds
            result = {
                "status": "PASS_N4A_SCALE_CELL",
                "task": name,
                "seed": seed,
                "worlds": worlds,
                "protocol_binding": protocol_binding(name),
                "ppo_config": session.config,
                "cold_construction_s": cold_seconds,
                "warmup": warmup,
                "warmup_checks": warmup_checks,
                "measured": measured,
                "measured_world_steps": accepted,
                "timing": {
                    "compute_and_mandatory_diagnostics_s": compute,
                    "post_update_independent_audit_s": audits,
                    "evidence_write_s": write_seconds,
                    "transitions_per_s_compute_and_diagnostics": accepted / compute,
                    "transitions_per_s_with_current_audit_and_write": accepted
                    / (compute + audits + write_seconds),
                },
                "coverage": evaluation.COVERAGE,
                "counter_end": evaluation.plain_counters(env.counters),
                "source_files": manifest["source_files"],
                "dependencies": manifest["dependencies"],
            }
        context["env"] = None
        result = _finish(context, result)
        # Full final gzip close/write is reported separately in the evidence receipt.
        # The measured-window numerator excludes construction, warmup and their evidence.
        return result


def _read_summary(directory):
    directory = Path(directory)
    path = directory / "summary.json"
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("bounded complete seed summary required")
    raw = path.read_bytes()
    summary = json.loads(raw)
    for filename, maximum in (("manifest.json", 4 * 1024 * 1024), ("publication.json", 16384)):
        sibling = directory / filename
        if not sibling.is_file() or sibling.stat().st_size > maximum:
            raise ValueError("complete bounded run binding missing")
    manifest = json.loads((directory / "manifest.json").read_bytes())
    publication = json.loads((directory / "publication.json").read_bytes())
    if publication.get("summary_sha256") != sha256(raw).hexdigest():
        raise ValueError("summary publication hash differs")
    if manifest.get("phase") != "learn" or any(
        summary.get(key) != manifest.get(key)
        for key in (
            "task",
            "seed",
            "protocol_binding",
            "ppo_config",
            "source_files",
            "dependencies",
        )
    ):
        raise ValueError("summary and original run manifest differ")
    receipt = summary.get("evidence", {})
    if receipt.get("path") != "trace.jsonl.gz":
        raise ValueError("unexpected evidence path")
    payload = directory / receipt["path"]
    if not payload.is_file() or payload.stat().st_size != receipt.get("compressed_bytes"):
        raise ValueError("complete bound evidence file missing")
    digest = sha256()
    with payload.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != receipt.get("compressed_sha256"):
        raise ValueError("seed evidence hash differs")
    return summary, {
        "directory": str(directory.resolve()),
        "summary_sha256": sha256(raw).hexdigest(),
        "evidence": receipt,
    }


def summarize_task(name, inputs, output):
    """Pure saved-value aggregation: no policy, CUDA or physical replay."""
    item = tasks().task(name)
    if len(inputs) != 3 or len({str(Path(path).resolve()) for path in inputs}) != 3:
        raise ValueError("three distinct complete seed directories required")
    rows, bindings = {}, {}
    expected = protocol_binding(name)
    for path in inputs:
        value, binding = _read_summary(path)
        seed = value.get("seed")
        if seed not in item.seeds or seed in rows or value.get("task") != name:
            raise ValueError("missing, duplicate or wrong-task predeclared seed")
        if value.get("status") != "COMPLETE_N4A_SEED" or value.get("protocol_binding") != expected:
            raise ValueError("incomplete or changed learning contract")
        if (
            value.get("updates") != 128
            or value.get("accepted_world_steps") != 32768
            or value.get("worlds") != 8
        ):
            raise ValueError("learning budget differs")
        if any(value.get(key, {}).get("world_steps") != 2048 for key in ("baseline", "learned")):
            raise ValueError("complete held-out windows required")
        evidence = value.get("update_evidence", ())
        if len(evidence) != 128 or any(
            row.get("iteration") != i
            or row.get("optimizer_step") != i + 1
            or row.get("accepted_world_steps") != 256
            for i, row in enumerate(evidence)
        ):
            raise ValueError("full update sequence/Adam evidence required")
        if value["parameter_hash_before"] == value["parameter_hash_after"]:
            raise ValueError("unchanged learner")
        rows[seed], bindings[seed] = value, binding
    first = rows[item.seeds[0]]
    for value in rows.values():
        if any(value[key] != first[key] for key in ("ppo_config", "source_files", "dependencies")):
            raise ValueError("source/config/dependency binding differs across seeds")
    zero = first.get("zero_action_diagnostic") or {}
    if (
        zero.get("status") not in ("COMPLETE_DIAGNOSTIC", "FAIL_DIAGNOSTIC")
        or any(rows[seed].get("zero_action_diagnostic") is not None for seed in item.seeds[1:])
        or (zero["status"] == "COMPLETE_DIAGNOSTIC" and zero.get("world_steps") != 2048)
        or (zero["status"] == "FAIL_DIAGNOSTIC" and zero.get("cost") is not None)
    ):
        raise ValueError("exactly one bound first-seed zero-action diagnostic required")
    baseline = [rows[seed]["baseline"]["cost"] for seed in item.seeds]
    learned = [rows[seed]["learned"]["cost"] for seed in item.seeds]
    result = {
        "schema": "mujoco-tyx-n4a-three-seed-summary-v1",
        "task": name,
        "seeds": list(item.seeds),
        "protocol_binding": expected,
        "outcome": evaluation.learning_outcome(baseline, learned),
        "zero_action_diagnostic": zero,
        "learned_vs_zero": None
        if zero["status"] == "FAIL_DIAGNOSTIC"
        else {
            "cost_difference": [value - zero["cost"] for value in learned],
            "lower_cost": [value < zero["cost"] for value in learned],
            "acceptance_gate": False,
        },
        "source_bindings": bindings,
        "scope": "saved-fixed-budget-quality-no-new-physical-execution",
    }
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    evaluation.json_write(path / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", required=True, choices=("preflight", "learn", "scale", "summarize")
    )
    parser.add_argument("--task", required=True, choices=("J", "S", "R"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--worlds", type=int)
    parser.add_argument("--inputs", nargs=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "preflight":
        if args.seed is not None or args.worlds is not None or args.inputs is not None:
            parser.error("preflight has fixed seed/worlds and no saved inputs")
        result = run_preflight(args.task, args.output)
    elif args.phase == "learn":
        if args.seed is None or args.worlds is not None or args.inputs is not None:
            parser.error("learn requires only --seed in addition to task/output")
        result = run_learning(args.task, args.seed, args.output)
    elif args.phase == "scale":
        if args.seed is None or args.worlds is None or args.inputs is not None:
            parser.error("scale requires --seed and --worlds")
        result = run_scale(args.task, args.seed, args.worlds, args.output)
    else:
        if args.inputs is None or args.seed is not None or args.worlds is not None:
            parser.error("summarize requires exactly three --inputs")
        result = summarize_task(args.task, args.inputs, args.output)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "task": args.task,
                "output": str(args.output.resolve()),
                "status": result.get("status", result.get("outcome", {}).get("classification")),
            }
        )
    )
    if args.phase == "summarize" and not result["outcome"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
