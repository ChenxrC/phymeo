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


class UpdateCaptureError(RuntimeError):
    """A failed update retains accepted transitions; it does not imply rollback."""

    def __init__(self, error, attempts, seconds):
        super().__init__("upstream update failed: " + str(error))
        self.attempts = tuple(attempts)
        self.compute_seconds = seconds
        self.original_type = type(error).__name__
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
    """Measure the complete public update, including accepted-prefix retention."""
    import torch

    env = session.env
    before = plain_counters(env.counters)
    torch.cuda.synchronize(env.device)
    free_before, total_memory = torch.cuda.mem_get_info(env.device)
    torch.cuda.reset_peak_memory_stats(env.device)
    profile = cProfile.Profile() if observe else None
    attempts, failure = [], None
    started = time.perf_counter()
    try:
        with capture_adapter_steps(session.adapter) as attempts:
            if profile is not None:
                profile.enable()
            record = session.update()
            torch.cuda.synchronize(env.device)
    except BaseException as error:
        failure = error
    finally:
        if profile is not None:
            profile.disable()
        elapsed = time.perf_counter() - started
    if failure is not None:
        raise UpdateCaptureError(failure, attempts, elapsed) from failure
    if len(attempts) != len(record["transitions"]) or any(
        entry.get("transition") is not record["transitions"][i] for i, entry in enumerate(attempts)
    ):
        error = ValueError("public step observer and session transition history disagree")
        raise UpdateCaptureError(error, attempts, elapsed) from error
    record = dict(record)
    # Each successful transition has one complete copy, referenced by ordinal.
    record["step_requests"] = tuple(
        {
            "ordinal": i,
            "status": entry["status"],
            "requested_action": entry["requested_action"],
            "transition_index": i,
        }
        for i, entry in enumerate(attempts)
    )
    free_after, _ = torch.cuda.mem_get_info(env.device)
    after = plain_counters(env.counters)
    measurement = {
        "compute_and_mandatory_diagnostics_s": elapsed,
        "profiled": observe,
        "observer": None if profile is None else _visible_profile(profile),
        "counters_before": before,
        "counters_after": after,
        "counter_delta": counter_delta(before, after),
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(env.device),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(env.device),
        "cuda_free_before_bytes": free_before,
        "cuda_free_after_bytes": free_after,
        "cuda_total_bytes": total_memory,
        "cuda_net_used_change_bytes_not_tyx_peak": free_before - free_after,
        "outer_explicit_completion_calls": 2,
        "scope": "public-update-with-owned-request/prefix-observer-plus-final-CUDA-completion",
    }
    return record, measurement


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
    "exclusive_cpu_physics_validation_seconds": "UNKNOWN",
    "tyx_only_peak_memory": "UNKNOWN; allocation counts and CUDA net change are reported",
    "policy_and_task_path": "CUDA with current audited HOST action/physics boundaries retained",
    "serialization": "full lossless payload encoding/compression/write outside compute timer",
}
