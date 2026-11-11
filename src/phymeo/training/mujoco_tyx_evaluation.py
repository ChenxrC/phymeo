"""Optional, task-independent evaluation and honest full-update measurements.

No learning algorithm or physical solver is implemented here. Array encoding is
lossless evidence, not checkpoint deserialization. CUDA imports remain lazy.
"""

import cProfile
import gzip
import json
import math
import random
import sys
import time
from base64 import b64encode
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class QuadraticCost:
    q_target: tuple
    q_scale: tuple
    v_scale: tuple
    velocity_weight: float

    def __post_init__(self):
        for name in ("q_target", "q_scale", "v_scale"):
            value = tuple(float(x) for x in getattr(self, name))
            if not value or not all(math.isfinite(x) for x in value):
                raise ValueError("finite nonempty cost coordinates required")
            object.__setattr__(self, name, value)
        if (
            len(self.q_target) != len(self.q_scale)
            or any(x <= 0 for x in self.q_scale + self.v_scale)
            or not math.isfinite(self.velocity_weight)
            or self.velocity_weight < 0
        ):
            raise ValueError("positive scales and nonnegative velocity weight required")


def host_array(value):
    if isinstance(value, np.ndarray):
        return value
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def native_cost(states, world_keys, cost):
    """Independent FP64 objective on actual pre-reset published native states."""
    if set(states) != set(world_keys):
        raise ValueError("complete declared worlds required")
    values = []
    for key in world_keys:
        q, v = np.asarray(states[key].q, np.float64), np.asarray(states[key].v, np.float64)
        if q.shape != (len(cost.q_target),) or v.shape != (len(cost.v_scale),):
            raise ValueError("native coordinate dimensions differ from objective")
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError("finite native coordinates required")
        value = np.mean(((q - cost.q_target) / cost.q_scale) ** 2)
        value += cost.velocity_weight * np.mean((v / cost.v_scale) ** 2)
        if not math.isfinite(float(value)):
            raise ValueError("finite native objective required")
        values.append(float(value))
    return np.asarray(values, np.float64)


def learning_outcome(baseline, learned):
    """Fixed three-seed gate; never pick a best seed or divide by zero."""
    b, learned_costs = tuple(map(float, baseline)), tuple(map(float, learned))
    if len(b) != 3 or len(learned_costs) != 3:
        raise ValueError("exactly three predeclared seeds required")
    result = {"baseline": b, "learned": learned_costs, "ratios": None, "passed": False}
    if not all(math.isfinite(x) and x >= 0 for x in b + learned_costs):
        result["classification"] = "NONFINITE_OR_NEGATIVE_COST"
    elif any(x == 0 for x in b):
        result["classification"] = "BASELINE_DEGENERATE"
    else:
        ratios = tuple(y / x for x, y in zip(b, learned_costs, strict=True))
        median = sorted(ratios)[1]
        passed = max(ratios) <= 1.05 and sum(x <= 0.90 for x in ratios) >= 2 and median <= 0.80
        result.update(
            ratios=ratios,
            median_ratio=median,
            passed=passed,
            classification="PASS_FIXED_LEARNING_GATE" if passed else "FIXED_BUDGET_NOT_LEARNED",
        )
    return result


def check_window_transition(transition, world_keys, tick, steps):
    if type(tick) is not int or not 1 <= tick <= steps:
        raise ValueError("invalid complete-window tick")
    if set(transition.states) != set(world_keys) or dict(
        transition.accepted_worlds
    ) != dict.fromkeys(world_keys, True):
        raise ValueError("physical world missing or refused")
    terminated, timeout = (
        host_array(getattr(transition, key)) for key in ("terminated", "time_outs")
    )
    if terminated.shape != (len(world_keys),) or timeout.shape != terminated.shape:
        raise ValueError("done-mask layout")
    if terminated.any() or (tick < steps and timeout.any()):
        raise ValueError("early termination cannot shorten the evaluation cost")


def counter_delta(before, after):
    if set(before) != set(after):
        raise ValueError("counter schema changed inside measurement")
    output = {}
    for key in before:
        a, b = before[key], after[key]
        if isinstance(a, Mapping):
            output[key] = counter_delta(a, b)
        else:
            if (
                not isinstance(a, (int, np.integer))
                or not isinstance(b, (int, np.integer))
                or b < a
            ):
                raise ValueError("counter is not monotonic integer: " + str(key))
            output[key] = int(b - a)
    return output


def encode(value):
    """Complete typed lossless payload; rejects unknown object representations."""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object array cannot be serialized as evidence")
        array = np.ascontiguousarray(value)
        raw = array.tobytes()
        return {
            "array_dtype": array.dtype.str,
            "shape": list(array.shape),
            "bytes_base64": b64encode(raw).decode("ascii"),
            "sha256": sha256(raw).hexdigest(),
        }
    if isinstance(value, np.generic):
        return encode(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": type(value).__module__ + "." + type(value).__qualname__,
            "fields": {field.name: encode(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, Mapping):
        # Native world IDs remain typed keys; do not collapse int5 and string'5'.
        return {"mapping": [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {"sequence_type": type(value).__name__, "items": [encode(x) for x in value]}
    if isinstance(value, bytes):
        return {
            "bytes_base64": b64encode(value).decode("ascii"),
            "sha256": sha256(value).hexdigest(),
        }
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": repr(value)}
    raise TypeError("unrecorded evidence type: " + type(value).__qualname__)


def json_write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


class EvidenceWriter:
    """One fresh gzip stream, complete transition payloads, no temporary plain copy."""

    def __init__(self, path):
        self.path = Path(path)
        self._raw = self.path.open("xb")
        self._zip = gzip.GzipFile(fileobj=self._raw, mode="wb", mtime=0)
        self._hash, self.logical_bytes, self.records = sha256(), 0, 0
        self.write_seconds, self.closed = 0.0, False

    def write(self, kind, payload):
        started = time.perf_counter()
        raw = (
            json.dumps(
                {"ordinal": self.records, "kind": kind, "payload": encode(payload)},
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode()
        self._zip.write(raw)
        self._hash.update(raw)
        self.logical_bytes += len(raw)
        self.records += 1
        self.write_seconds += time.perf_counter() - started

    def close(self):
        if not self.closed:
            started = time.perf_counter()
            try:
                self._zip.close()
            finally:
                self._raw.close()
                self.closed = True
                self.write_seconds += time.perf_counter() - started

    def receipt(self):
        if not self.closed:
            raise RuntimeError("finish complete evidence stream first")
        digest = sha256()
        with self.path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return {
            "path": self.path.name,
            "records": self.records,
            "logical_bytes": self.logical_bytes,
            "logical_sha256": self._hash.hexdigest(),
            "compressed_bytes": self.path.stat().st_size,
            "compressed_sha256": digest.hexdigest(),
            "write_seconds": self.write_seconds,
        }


@contextmanager
def preserved_rng():
    """Evaluation construction and deterministic actor calls cannot advance training RNG."""
    import torch

    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state, cuda_states = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(cuda_states)


def parameter_hash(algorithm):
    digest = sha256()
    for owner, model in (("actor", algorithm.actor), ("critic", algorithm.critic)):
        for key, tensor in sorted(model.state_dict().items()):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(
                (owner + ":" + key + ":" + array.dtype.str + ":" + str(array.shape)).encode()
            )
            digest.update(array.tobytes())
    return digest.hexdigest()


def evaluate_window(env, action, cost, *, steps, writer, label, inspect=None):
    """All declared pre-reset states contribute; no success by early stopping."""
    import torch

    total = np.zeros(env.num_envs, np.float64)
    counts = np.zeros(env.num_envs, np.int64)
    reward_error = 0.0
    for tick in range(1, steps + 1):
        with torch.inference_mode():
            commands = action(env.get_observations())
            writer.write(
                label + ".requested_action",
                {"tick": tick, "action": commands, "input_frame": env.current_frame},
            )
            transition = env.step(
                commands,
                world_keys=env.world_keys,
                expected_frame=env.current_frame,
                producer_stream=torch.cuda.current_stream(env.device),
            )
        # Preserve the actual transition even if its evaluation checks then fail.
        writer.write(label + ".transition", transition)
        check_window_transition(transition, env.world_keys, tick, steps)
        values = native_cost(transition.states, env.world_keys, cost)
        reward = host_array(transition.reward).astype(np.float64)
        if (
            reward.shape != values.shape
            or not np.isfinite(reward).all()
            or not np.allclose(reward, -values, rtol=2e-5, atol=2e-6)
        ):
            raise ValueError("device objective differs from independent native cost")
        reward_error = max(reward_error, float(np.max(np.abs(reward + values))))
        total += values
        counts += 1
        detail = None if inspect is None else inspect(tick, transition)
        writer.write(label + ".independent", {"tick": tick, "cost": values, "inspection": detail})
    if not np.all(counts == steps):
        raise ValueError("incomplete evaluation window")
    return {
        "world_steps": int(counts.sum()),
        "steps": steps,
        "cost": float(np.mean(total / steps)),
        "per_world_cost": (total / steps).tolist(),
        "reward_max_absolute_difference": reward_error,
    }


VISIBLE_CALLS = frozenset(
    {
        "set_controls_from_device",
        "step",
        "snapshot",
        "reset",
        "refresh",
        "to_torch",
        "synchronize",
        "copy_to",
        "copy_from",
        "to_numpy",
        "update",
        "process_env_step",
        "compute_returns",
        "act",
        "_audit",
        "_validate_step",
        "_advance",
    }
)


def _visible_profile(profile):
    rows = []
    for entry in profile.getstats():
        code = entry.code
        if isinstance(code, str):
            selected = any(word in code for word in ("synchronize", "to_numpy", "copy_"))
            identity = code
        else:
            path = code.co_filename.replace("\\", "/")
            selected = code.co_name in VISIBLE_CALLS and any(
                item in path for item in ("phymeo/", "rsl_rl/", "torch/", "tyx/")
            )
            identity = path + ":" + str(code.co_firstlineno) + ":" + code.co_name
        if selected:
            rows.append(
                {
                    "call": identity,
                    "calls": entry.callcount,
                    "recursive_calls": entry.reccallcount,
                    "inclusive_s": entry.totaltime,
                    "self_python_s": entry.inlinetime,
                }
            )
    return sorted(rows, key=lambda row: row["call"])


def plain_counters(value):
    return {
        key: plain_counters(item) if isinstance(item, Mapping) else int(item)
        for key, item in value.items()
    }


# Explicit source-defined lifecycle classes. New counter names are not silently
# promoted into either a cumulative measurement or a physical work estimate.
ENV_COUNTERS = frozenset(("transitions", "physical_steps", "resets",
                         "termination_readback_bytes", "action_finite_scalar_readbacks"))
RUNTIME_COUNTERS = frozenset((
    "completed_steps", "accepted_world_steps", "failed_batches", "device_control_submissions",
    "device_control_refusals", "control_copy_d2d_bytes", "control_audit_downloaded_bytes",
    "control_status_downloaded_bytes", "control_initial_uploaded_bytes", "control_pack_kernels",
))
TASK_COUNTERS = frozenset((
    "task_kernels", "source_synchronizations", "task_status_downloaded_bytes",
    "explicit_output_readback_bytes", "output_d2d_bytes", "published_frames", "failed_refreshes",
))
DEVICE_COUNTERS = frozenset((
    "completed_kernels", "synchronizations", "device_allocations", "static_uploaded_bytes",
    "uploaded_bytes", "downloaded_bytes", "device_control_d2d_bytes", "control_host_uploaded_bytes",
))


def measurement_counter_snapshot(env):
    """Public owner/transaction bindings; no private device generation inference."""
    frame = env.current_frame
    return {
        "counts": plain_counters(env.counters),
        "owners": {"environment": id(env), "runtime": id(env.runtime),
                   "workspace": frame.binding["workspace_owner"]},
        "frame": {"revision": env.runtime.checkpoint_revision, "epoch": frame.epoch,
                  "ticks": list(frame.ticks), "times": list(frame.times)},
        "device_generation": "UNKNOWN: not exposed by the public runtime",
    }


def counter_accounting(before, after, transitions):
    """Keep replaceable device snapshots signed; accumulate only proven scopes."""
    if before["owners"] != after["owners"]:
        raise ValueError("public measurement owner changed")
    for value in (before["counts"], after["counts"]):
        if set(value) != ENV_COUNTERS | {"runtime", "task"}:
            raise ValueError("unclassified environment counter schema")
        if set(value["task"]) != TASK_COUNTERS:
            raise ValueError("unclassified task counter schema")
        runtime_keys = set(value["runtime"])
        required_device = DEVICE_COUNTERS - {"device_control_d2d_bytes", "control_host_uploaded_bytes"}
        if (not (RUNTIME_COUNTERS | required_device) <= runtime_keys
                or runtime_keys - RUNTIME_COUNTERS - DEVICE_COUNTERS):
            raise ValueError("unclassified runtime counter schema")
        for group in ({k: value[k] for k in ENV_COUNTERS}, value["task"], value["runtime"]):
            if any(type(number) is not int or number < 0 for number in group.values()):
                raise ValueError("counter snapshot must contain nonnegative integers")
    a, b = before["counts"], after["counts"]
    monotonic = {
        "environment": counter_delta({k: a[k] for k in ENV_COUNTERS}, {k: b[k] for k in ENV_COUNTERS}),
        "runtime": counter_delta({k: a["runtime"][k] for k in RUNTIME_COUNTERS},
                                 {k: b["runtime"][k] for k in RUNTIME_COUNTERS}),
        "task": counter_delta(a["task"], b["task"]),
    }
    reset_events, accepted = [], 0
    for entry in transitions:
        keys = tuple(entry.world_keys)
        if set(entry.states) != set(keys) or dict(entry.accepted_worlds) != dict.fromkeys(keys, True):
            raise ValueError("accounting requires complete accepted transition worlds")
        reset_keys = []
        for key, output_tick in zip(keys, entry.output_frame.ticks, strict=True):
            accepted_tick = entry.states[key].tick
            if output_tick == 0 and accepted_tick > 0:
                reset_keys.append(key)
            elif output_tick != accepted_tick:
                raise ValueError("output frame differs from accepted state without a reset")
        accepted += len(keys)
        if reset_keys:
            reset_events.append({"sequence": entry.sequence, "world_keys": tuple(reset_keys),
                                 "output_revision": entry.output_frame.revision})
    reset_worlds = sum(len(entry["world_keys"]) for entry in reset_events)
    if (monotonic["environment"]["transitions"] != len(transitions)
            or monotonic["environment"]["physical_steps"] != len(transitions)
            or monotonic["environment"]["resets"] != reset_worlds
            or monotonic["runtime"]["completed_steps"] != len(transitions)
            or monotonic["runtime"]["accepted_world_steps"] != accepted):
        raise ValueError("public receipt events disagree with lifecycle counters")
    device_keys = sorted((set(a["runtime"]) | set(b["runtime"])) & DEVICE_COUNTERS)
    signed = {key: (b["runtime"][key] - a["runtime"][key]
                    if key in a["runtime"] and key in b["runtime"] else None)
              for key in device_keys}
    return {
        "owners_before": before["owners"], "owners_after": after["owners"],
        "transaction_before": before["frame"], "transaction_after": after["frame"],
        "monotonic_delta": monotonic,
        "monotonic_scope": "same public environment/runtime/workspace lifetimes; categories may overlap",
        "device_snapshot": {
            "scope": "current device-owner snapshot difference, NOT cumulative work",
            "signed_difference_not_work": signed,
            "before": {key: a["runtime"].get(key) for key in device_keys},
            "after": {key: b["runtime"].get(key) for key in device_keys},
            "cumulative_device_work": "UNKNOWN", "absolute_device_generation": "UNKNOWN",
            "missing_field": "None means not exposed in that owner snapshot, not zero",
        },
        "receipt_events": {"accepted_world_steps": accepted, "reset_calls": len(reset_events),
                           "reset_worlds": reset_worlds, "resets": tuple(reset_events),
                           "scope": "public post-reset tick publication; no internal work inference"},
    }

class UpdateCaptureError(RuntimeError):
    """A failed update retains accepted transitions; it does not imply rollback."""

    def __init__(self, error, attempts, seconds, *, completed_record=None, before=None, after=None, measurement_phase=None):
        super().__init__("upstream update failed: " + str(error))
        self.attempts = tuple(attempts)
        self.compute_seconds = seconds
        self.original_type = type(error).__name__
        self.completed_record = completed_record
        self.counter_snapshot_before, self.counter_snapshot_after = before, after
        self.measurement_phase = measurement_phase
        for key in (
            "phase",
            "action_accepted",
            "state_accepted",
            "needs_recovery",
            "accepted_worlds",
        ):
            if hasattr(error, key):
                setattr(self, key, getattr(error, key))


@contextmanager
def capture_adapter_steps(adapter):
    """Observe one public call once; retain requests and accepted prefix on failure.

    The session already requires exclusive synchronous adapter ownership. Owned
    action clones stay on the producer CUDA stream; no HOST observation copy or
    physical state mutation is introduced. This observer is removed in finally.
    """
    attempts = []
    sentinel = object()
    previous = vars(adapter).get("step", sentinel)
    original = adapter.step

    def step(actions):
        entry = {"ordinal": len(attempts), "status": "request-copy", "requested_action": None}
        attempts.append(entry)
        try:
            entry["requested_action"] = actions.detach().clone()
            entry["status"] = "calling-public-step"
            result = original(actions)
            entry["transition"] = adapter.last_transition
            entry["status"] = "accepted"
            return result
        except BaseException as error:
            entry["status"] = "failed"
            entry["error_type"], entry["error"] = type(error).__name__, str(error)
            for key in (
                "phase",
                "action_accepted",
                "state_accepted",
                "needs_recovery",
                "accepted_worlds",
                "action_receipt",
                "transition",
            ):
                if hasattr(error, key):
                    entry[key] = getattr(error, key)
            raise

    adapter.step = step
    try:
        yield attempts
    finally:
        if previous is sentinel:
            del adapter.step
        else:
            adapter.step = previous


def measure_update(session, *, observe=False):
    """The failure boundary includes post-update synchronization and accounting."""
    import torch

    env = session.env
    attempts, record, before, after = [], None, None, None
    profile, elapsed, phase = None, 0., "measurement-setup"
    try:
        before = measurement_counter_snapshot(env)
        torch.cuda.synchronize(env.device)
        free_before, total_memory = torch.cuda.mem_get_info(env.device)
        torch.cuda.reset_peak_memory_stats(env.device)
        profile = cProfile.Profile() if observe else None
        started = time.perf_counter()
        try:
            with capture_adapter_steps(session.adapter) as attempts:
                if profile is not None:
                    profile.enable()
                phase = "public-session-update"
                record = session.update()
                phase = "final-CUDA-completion"
                torch.cuda.synchronize(env.device)
        finally:
            if profile is not None:
                profile.disable()
            elapsed = time.perf_counter() - started
        phase = "post-update-measurement"
        if len(attempts) != len(record["transitions"]) or any(
            entry.get("transition") is not record["transitions"][i] for i, entry in enumerate(attempts)
        ):
            raise ValueError("public step observer and session transition history disagree")
        free_after, _ = torch.cuda.mem_get_info(env.device)
        after = measurement_counter_snapshot(env)
        accounting = counter_accounting(before, after, record["transitions"])
        measurement = {
            "compute_and_mandatory_diagnostics_s": elapsed,
            "profiled": observe,
            "observer": None if profile is None else _visible_profile(profile),
            "counters_before": before["counts"], "counters_after": after["counts"],
            "counter_accounting": accounting,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(env.device),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(env.device),
            "cuda_free_before_bytes": free_before, "cuda_free_after_bytes": free_after,
            "cuda_total_bytes": total_memory,
            "cuda_net_used_change_bytes_not_tyx_peak": free_before - free_after,
            "outer_explicit_completion_calls": 2,
            "scope": "public-update-with-owned-request/prefix-observer-plus-final-CUDA-completion",
        }
        published_record = dict(record)
        published_record["step_requests"] = tuple(
            {"ordinal": i, "status": entry["status"], "requested_action": entry["requested_action"],
             "transition_index": i} for i, entry in enumerate(attempts)
        )
        return published_record, measurement
    except BaseException as error:
        raise UpdateCaptureError(error, attempts, elapsed, completed_record=record,
                                 before=before, after=after, measurement_phase=phase) from error

def validate_update(record, env, *, expected_iteration, cost, writer):
    """Save every transition first; independent cost and learning checks follow."""
    transitions = record["transitions"]
    writer.write("update", record)
    if (
        record["iteration"] != expected_iteration
        or record["next_iteration"] != expected_iteration + 1
    ):
        raise ValueError("upstream update sequence changed")
    if not transitions or record["optimization"]["optimizer_step"] != expected_iteration + 1:
        raise ValueError("actual Adam step differs from one-epoch single-minibatch protocol")
    if not record["optimization"]["finite_gradient"] or not all(
        math.isfinite(float(x)) for x in record["loss"].values()
    ):
        raise ValueError("nonfinite optimization")
    ended, timeout, total_cost = 0, 0, 0.0
    for entry in transitions:
        if dict(entry.accepted_worlds) != dict.fromkeys(env.world_keys, True):
            raise ValueError("training physical step was refused")
        values = native_cost(entry.states, env.world_keys, cost)
        reward = host_array(entry.reward).astype(np.float64)
        if not np.allclose(reward, -values, rtol=2e-5, atol=2e-6):
            raise ValueError("training device reward differs from native objective")
        ended += int(np.count_nonzero(host_array(entry.terminated)))
        timeout += int(np.count_nonzero(host_array(entry.time_outs)))
        total_cost += float(np.sum(values))
    if ended:
        raise ValueError("physical task bounds terminated a training/scale world")
    return {
        "accepted_world_steps": len(transitions) * env.num_envs,
        "terminated_worlds": ended,
        "timeout_reset_worlds": timeout,
        "mean_native_cost": total_cost / (len(transitions) * env.num_envs),
        "loss": {key: float(value) for key, value in record["loss"].items()},
        "optimizer_step": int(record["optimization"]["optimizer_step"]),
    }


COVERAGE = {
    "primary": "full public update including complete physics HOST diagnostics",
    "observer": "existing warmup only; inclusive calls overlap and are not additive",
    "formal_updates": "three unprofiled complete updates",
    "unobserved_torch_internal_sync": "UNKNOWN",
    "device_counter_totals_across_reset": "UNKNOWN; signed owner snapshots are not work",
    "exclusive_cpu_physics_validation_seconds": "UNKNOWN",
    "tyx_only_peak_memory": "UNKNOWN; allocation counts and CUDA net change are reported",
    "policy_and_task_path": "CUDA with current audited HOST action/physics boundaries retained",
    "serialization": "full lossless payload encoding/compression/write outside compute timer",
}
