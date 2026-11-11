"""Paired full/integration observation through real, bounded CUDA PPO updates.

The physics checks are identical; endpoint-only diagnostic outputs differ.
This is an explicit-output-contract measurement, not a Newton speed comparison.
"""

import argparse
import importlib.util
import json
import traceback
from contextlib import ExitStack, contextmanager, nullcontext
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
                **{name: getattr(frame, name) for name in ("revision", "epoch", "ticks", "times")},
                "binding": stable_task_binding(frame.binding),
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


TASK_BINDING_FIELDS = frozenset(
    (
        "schema",
        "workspace_owner",
        "runtime_owner",
        "spec_sha256",
        "source_sha256",
        "archive_sha256",
        "model_sha256",
        "world_keys",
        "mode",
        "device",
        "native_layout",
        "output_layout",
        "precision",
        "actions",
        "physics_diagnostics",
    )
)


def stable_task_binding(binding):
    values = dict(binding)
    if not TASK_BINDING_FIELDS <= values.keys():
        raise ValueError("incomplete task frame binding")
    return {
        key: value
        for key, value in values.items()
        if key not in ("workspace_owner", "runtime_owner")
    }


def stable_environment_binding(binding):
    values = dict(binding)
    if values.get("state_observation") not in ("full", "integration"):
        raise ValueError("unexpected environment observation binding")
    values.pop("state_observation")
    values["task_source"] = stable_task_binding(values["task_source"])
    return values


def summary_only(value):
    """Reject native/Torch/receipt owners at the per-mode return boundary."""
    return json.loads(json.dumps(value, allow_nan=False))


def process_memory():
    """Current Windows process commit and RSS; not exclusive MuJoCo memory."""
    import ctypes
    import os
    from ctypes import wintypes

    if os.name != "nt":
        return {"status": "UNKNOWN", "reason": "Windows process counters unavailable"}

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
                "PrivateUsage",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    values = Counters()
    values.cb = ctypes.sizeof(values)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(values), values.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "status": "OBSERVED",
        "private_commit_bytes": int(values.PrivateUsage),
        "rss_bytes": int(values.WorkingSetSize),
    }


class MemoryObservation:
    """Bounded S256 diagnostic. Never retains a native object or invokes physics."""

    def __init__(self, target, worlds, *, process_sample=process_memory):
        self.target, self.worlds, self.process_sample = Path(target), worlds, process_sample
        self.mode, self.step, self.step_copies = None, 0, 0
        self.copy_calls, self.copy_failures, self.events = 0, 0, []
        self.maximum_copies = 2 * 48 * worlds + 2 * worlds + 64
        self.maximum_events = 512
        self.native_ranges, self.files = {}, []

    def event(self, stage, *, native=None, **fields):
        if len(self.events) >= self.maximum_events:
            raise RuntimeError("fixed memory diagnostic event bound exceeded")
        try:
            process = self.process_sample()
        except Exception as error:
            process = {"status": "UNKNOWN", "reason": type(error).__name__ + ": " + str(error)}
        self.events.append(
            {
                "stage": stage,
                "mode": self.mode,
                "step": self.step,
                "copy_ordinal": self.copy_calls,
                "copy_in_step": self.step_copies,
                "process": process,
                **fields,
                **({"native": dict(native)} if native is not None else {}),
            }
        )

    def begin_mode(self, mode):
        self.mode, self.step, self.step_copies = mode, 0, 0
        self.event("construct-enter")

    def begin_step(self):
        self.step += 1
        self.step_copies = 0
        if self.step > 48:
            raise RuntimeError("fixed memory diagnostic step bound exceeded")
        if self.step == 1:
            self.event("first-physics-step-enter")

    def end_step(self, *, failed=False):
        if self.step == 1 or failed:
            self.event("physics-step-failed" if failed else "first-physics-step-complete")

    @contextmanager
    def observe_steps(self, adapter):
        sentinel = object()
        previous = vars(adapter).get("step", sentinel)
        original = adapter.step

        def step(actions):
            self.begin_step()
            try:
                result = original(actions)
            except BaseException:
                self.end_step(failed=True)
                raise
            self.end_step()
            return result

        adapter.step = step
        try:
            yield
        finally:
            if previous is sentinel:
                del adapter.step
            else:
                adapter.step = previous

    @contextmanager
    def install(self, module, data_type):
        from types import SimpleNamespace

        original = module.copy
        observer = self

        def copied(value):
            if not isinstance(value, data_type):
                return original.copy(value)
            observer.copy_calls += 1
            observer.step_copies += 1
            if observer.copy_calls > observer.maximum_copies:
                raise RuntimeError("fixed memory diagnostic copy bound exceeded")
            # model is a documented public MjData property; no private bank read.
            native = {
                "data_narena_bytes": int(value.narena),
                "data_nbuffer_bytes": int(value.nbuffer),
                "data_maxuse_arena_bytes": int(value.maxuse_arena),
                "model_narena_bytes": int(value.model.narena),
                "model_nbuffer_bytes": int(value.model.nbuffer),
            }
            for name, number in native.items():
                bounds = observer.native_ranges.setdefault(name, [number, number])
                bounds[0], bounds[1] = min(bounds[0], number), max(bounds[1], number)
            if observer.step_copies == 1:
                observer.event("copy-first-before", native=native)
            try:
                result = original.copy(value)
            except BaseException as error:
                observer.copy_failures += 1
                observer.event(
                    "copy-failed", native=native, error_type=type(error).__name__, error=str(error)
                )
                raise
            if observer.step_copies == observer.worlds:
                observer.event("copy-last-complete", native=native)
            return result

        # The actual module has only copy.copy call sites. Do not modify the
        # shared stdlib copy module object, any physical bank or native value.
        module.copy = SimpleNamespace(copy=copied)
        try:
            yield
        finally:
            module.copy = original

    def snapshot(self):
        return summary_only(
            {
                "schema": "bounded-native-allocation-observation-v1",
                "process_metric_scope": "whole-process-not-exclusive-native",
                "copy_scope": "TYX runtime copy.copy only; full snapshot copies are not intercepted",
                "model_scope": "actual copied MjData.model, not a fresh or dummy model",
                "copy_calls": self.copy_calls,
                "copy_failures": self.copy_failures,
                "maximum_copies": self.maximum_copies,
                "maximum_events": self.maximum_events,
                "native_ranges": self.native_ranges,
                "events": self.events,
                "previous_snapshots": list(self.files),
                "performance_eligible": False,
                "leak_causality": "NOT_ESTABLISHED",
            }
        )

    def flush(self):
        name = "memory-observation-" + str(len(self.files) + 1).zfill(3) + ".json"
        evidence.json_write(self.target / name, self.snapshot())
        self.files.append(name)


def _training_factories():
    from phymeo.training.mujoco_tyx_checkpoint import MuJoCoTyxPpoSession
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxVectorEnv
    from phymeo.training.rsl_mujoco_tyx import RslMuJoCoTyxVecEnv

    return MuJoCoTyxVectorEnv, RslMuJoCoTyxVecEnv, MuJoCoTyxPpoSession


@contextmanager
def _mode_owners(memory):
    with ExitStack() as owners:
        try:
            yield owners
        finally:
            if memory is not None:
                memory.event("mode-close-enter")


def _run_mode(name, worlds, mode, target, tasks, torch, seed, *, memory=None):
    """All mode-specific owners/rollouts remain local to this one call."""
    make_env, make_adapter, make_session = _training_factories()
    item = tasks.task(name)
    writer = evidence.EvidenceWriter(target / (mode + ".jsonl.gz"))
    signatures, timings, checks = [], [], []
    start_cold = perf_counter()
    diagnostic = nullcontext()
    if memory is not None:
        import mujoco

        from phymeo.backends import _mujoco_tyx_runtime

        diagnostic = memory.install(_mujoco_tyx_runtime, mujoco.MjData)
    try:
        with diagnostic:
            if memory is not None:
                memory.begin_mode(mode)
            try:
                with _mode_owners(memory) as owners:
                    base = owners.enter_context(
                        tasks.make_environment(name, count=worlds, scale=True)
                    )
                    runtime = base.runtime
                    base.close()
                    env = owners.enter_context(
                        make_env(
                            runtime,
                            base.task_spec,
                            base.action_spec,
                            h=base.h,
                            horizons=tuple(int(x) for x in base.horizons),
                            observation_mode=mode,
                        )
                    )
                    if memory is not None:
                        model = runtime.compilation.mj_model
                        memory.event(
                            "construct-complete",
                            compiler_model={
                                "narena_bytes": int(model.narena),
                                "nbuffer_bytes": int(model.nbuffer),
                                "scope": "public compilation model; runtime model observed on actual data copy",
                            },
                        )
                    tasks.seed_all(seed)
                    adapter = make_adapter(env)
                    if memory is not None:
                        owners.enter_context(memory.observe_steps(adapter))
                    session = make_session(adapter, steps=16)
                    initial = learner_state(session)
                    initial_hash = digest(initial)
                    writer.write("initial_policy", initial)
                    initial_observation = env.get_observations()
                    observation_hash = digest(initial_observation)
                    stable_binding = stable_environment_binding(env.binding)
                    binding_hash = digest(stable_binding)
                    writer.write(
                        "bindings",
                        {
                            "environment": env.binding,
                            "config": session.config,
                            "runtime": runtime.capabilities,
                            "mode": mode,
                            "stable_comparison_binding": stable_binding,
                        },
                    )
                    writer.write("initial_observation", initial_observation)
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
            finally:
                if memory is not None:
                    # Owners above were all attempted by ExitStack before this.
                    memory.event("mode-close-after-owner-attempts")
                    memory.flush()
        writer.close()
        wall = perf_counter() - measured_start
        row = {
            "initial_policy_hash": initial_hash,
            "signatures": signatures,
            "initial_observation_hash": observation_hash,
            "stable_binding_hash": binding_hash,
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
            "performance_eligible": memory is None,
        }
        evidence.json_write(target / (mode + "-summary.json"), row)
        return summary_only(
            {
                "summary": row,
                "comparable": {
                    "initial": initial_hash,
                    "observation": observation_hash,
                    "binding": binding_hash,
                    "updates": signatures,
                },
            }
        )
    except evidence.UpdateCaptureError as error:
        writer.write("failed_update_attempts", error.attempts)
        writer.write("failed_update_record", error.completed_record)
        raise
    finally:
        writer.close()


def run_pair(name, worlds, output, *, memory_diagnostic=False):
    if name not in ("J", "S", "R") or worlds not in (8, 128, 256):
        raise ValueError("predeclared J/S/R and W8/128/256 only")
    if memory_diagnostic and (name, worlds) != ("S", 256):
        raise ValueError("bounded memory diagnostic is restricted to the failed S256 case")
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    api = task_api()
    tasks = api.tasks()
    torch = tasks.require_cuda()
    seed = tasks.task(name).seeds[0]
    order = ("full", "integration") if worlds != 128 else ("integration", "full")
    rows, reference = {}, None
    memory = MemoryObservation(target, worlds) if memory_diagnostic else None
    evidence.json_write(
        target / "protocol.json",
        {
            "schema": "paired-observation-ppo-v2",
            "task": name,
            "worlds": worlds,
            "seed": seed,
            "order": order,
            "warmup_updates": 1,
            "measured_updates": 2,
            "steps_per_update": 16,
            "original_task": api.protocol_binding(name),
            "comparison": "common physics/transitions/policy inputs, stable bindings, actor/critic/Adam, Torch CPU/current CUDA RNG",
            "different_outputs": "full includes endpoint geometry/contact/sensor/energy; integration does not",
            "excluded_identity": ["workspace_owner", "runtime_owner"],
            "newton_performance_comparison": False,
            "memory_diagnostic": memory_diagnostic,
            "performance_eligible": not memory_diagnostic,
            "source_files": {
                **api._sources(),
                str(Path(__file__).resolve()): sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        },
    )
    try:
        for mode in order:
            mode_result = _run_mode(name, worlds, mode, target, tasks, torch, seed, memory=memory)
            if memory is not None:
                memory.event("mode-function-returned")
                memory.flush()
            rows[mode] = mode_result["summary"]
            comparable = mode_result["comparable"]
            if reference is None:
                reference = comparable
            elif comparable != reference:
                raise AssertionError(
                    "paired physics/learning/optimizer/Torch RNG differs; inspect saved signatures"
                )
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
            "rng_comparison": "Torch CPU and current CUDA device only",
            "memory_diagnostic": memory_diagnostic,
            "performance_eligible": not memory_diagnostic,
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
    parser.add_argument("--memory-diagnostic", action="store_true")
    args = parser.parse_args()
    summary = run_pair(
        args.task, args.worlds, args.output, memory_diagnostic=args.memory_diagnostic
    )
    print(json.dumps({key: summary[key] for key in ("status", "task", "worlds")}))


if __name__ == "__main__":
    main()
