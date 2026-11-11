"""Bounded saved-learning diagnostics. Never loads a policy or physical runtime.

The input is N4a's typed JSON evidence, not pickle or an object constructor.
Only NumPy/stdlib arithmetic on saved arrays is used. A successful diagnosis is
not a learning-quality or physics acceptance result.
"""

import base64
import gzip
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_RECORD_BYTES = 64 * 1024 * 1024
MAX_ARRAY_BYTES = 32 * 1024 * 1024
MAX_LOGICAL_BYTES = 4 * 1024**3
MAX_COMPRESSED_BYTES = 1024**3
TRANSITION_TYPE = "phymeo.training.mujoco_tyx_env.MuJoCoTyxTransition"
STATE_TYPE = "phymeo.backends.mujoco_runtime.MuJoCoState"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _float(value):
    result = float(value)
    require(math.isfinite(result), "nonfinite JSON number")
    return result


def strict_json(raw):
    def bad_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)

    try:
        return json.loads(
            raw, object_pairs_hook=_pairs, parse_float=_float, parse_constant=bad_constant
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid bounded JSON") from error


def decode_evidence(value, *, max_array_bytes=MAX_ARRAY_BYTES):
    """Validate typed bytes/hash/shape without importing their declared classes."""
    remaining = [2_000_000]

    def visit(item, depth):
        remaining[0] -= 1
        require(depth <= 96 and remaining[0] >= 0, "typed evidence nesting/node budget")
        if item is None or type(item) in (bool, int, str):
            return item
        if type(item) is float:
            require(math.isfinite(item), "nonfinite untagged value")
            return item
        require(type(item) is dict, "unknown evidence representation")
        keys = set(item)
        if keys == {"array_dtype", "shape", "bytes_base64", "sha256"}:
            try:
                dtype = np.dtype(item["array_dtype"])
            except (TypeError, ValueError) as error:
                raise ValueError("invalid array dtype") from error
            shape = item["shape"]
            require(
                dtype.kind in "biuf" and dtype.itemsize in (1, 2, 4, 8),
                "object/structured/unsupported array dtype",
            )
            require(type(shape) is list and len(shape) <= 8, "array rank")
            size = dtype.itemsize
            for dimension in shape:
                require(
                    type(dimension) is int and 0 <= dimension <= max_array_bytes, "array dimension"
                )
                size *= dimension
                require(size <= max_array_bytes, "array byte budget")
            encoded = item["bytes_base64"]
            require(
                type(encoded) is str and len(encoded) <= 4 * ((size + 2) // 3),
                "array encoding length",
            )
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as error:
                raise ValueError("array base64") from error
            require(
                len(raw) == size and hashlib.sha256(raw).hexdigest() == item["sha256"],
                "array bytes/hash",
            )
            return np.frombuffer(raw, dtype=dtype).reshape(tuple(shape))
        if keys == {"dataclass", "fields"}:
            require(
                type(item["dataclass"]) is str
                and type(item["fields"]) is dict
                and "__type__" not in item["fields"],
                "dataclass data schema",
            )
            return {
                "__type__": item["dataclass"],
                **{k: visit(v, depth + 1) for k, v in item["fields"].items()},
            }
        if keys == {"mapping"}:
            require(type(item["mapping"]) is list, "mapping representation")
            result = {}
            for pair in item["mapping"]:
                require(type(pair) is list and len(pair) == 2, "mapping pair")
                key = visit(pair[0], depth + 1)
                require(type(key) in (int, str, tuple), "mapping key type")
                try:
                    require(key not in result, "duplicate typed mapping key")
                    result[key] = visit(pair[1], depth + 1)
                except TypeError as error:
                    raise ValueError("unhashable mapping key") from error
            return result
        if keys == {"sequence_type", "items"}:
            require(
                item["sequence_type"] in ("list", "tuple") and type(item["items"]) is list,
                "sequence representation",
            )
            result = [visit(x, depth + 1) for x in item["items"]]
            return tuple(result) if item["sequence_type"] == "tuple" else result
        if keys == {"bytes_base64", "sha256"}:
            encoded = item["bytes_base64"]
            require(
                type(encoded) is str and len(encoded) <= 4 * ((max_array_bytes + 2) // 3),
                "bytes budget",
            )
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (TypeError, ValueError) as error:
                raise ValueError("bytes base64") from error
            require(
                len(raw) <= max_array_bytes and hashlib.sha256(raw).hexdigest() == item["sha256"],
                "bytes hash",
            )
            return raw
        if keys == {"nonfinite"}:
            require(item["nonfinite"] in ("nan", "inf", "-inf"), "nonfinite tag")
            # Keep a data tag; metric consumers never convert this into a finite value.
            return item
        raise ValueError("unknown typed evidence fields")

    return visit(value, 0)


class _HashReader(io.RawIOBase):
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.count = 0

    def readable(self):
        return True

    def read(self, size=-1):
        raw = self.stream.read(size)
        self.digest.update(raw)
        self.count += len(raw)
        return raw


def read_records(path, receipt):
    """One bounded streaming pass verifies compressed and logical identities."""
    path = Path(path)
    require(
        receipt.get("path") == "trace.jsonl.gz" and path.name == receipt["path"],
        "evidence basename",
    )
    for name, maximum in (
        ("compressed_bytes", MAX_COMPRESSED_BYTES),
        ("logical_bytes", MAX_LOGICAL_BYTES),
        ("records", 10_000),
    ):
        require(
            type(receipt.get(name)) is int and 0 < receipt[name] <= maximum,
            "evidence resource bound: " + name,
        )
    before = path.stat()
    require(before.st_size == receipt["compressed_bytes"], "compressed size")
    logical, total, count = hashlib.sha256(), 0, 0
    try:
        with path.open("rb") as raw:
            hashed = _HashReader(raw)
            with gzip.GzipFile(fileobj=hashed, mode="rb") as zipped:
                while True:
                    line = zipped.readline(MAX_RECORD_BYTES + 1)
                    if not line:
                        break
                    require(len(line) <= MAX_RECORD_BYTES and line.endswith(b"\n"), "record bound")
                    logical.update(line)
                    total += len(line)
                    require(total <= receipt["logical_bytes"], "logical byte overrun")
                    record = strict_json(line)
                    require(
                        type(record) is dict
                        and set(record) == {"ordinal", "kind", "payload"}
                        and type(record["ordinal"]) is int
                        and record["ordinal"] == count
                        and type(record["kind"]) is str,
                        "record identity/ordinal",
                    )
                    count += 1
                    require(count <= receipt["records"], "record count overrun")
                    yield record["kind"], decode_evidence(record["payload"])
            require(
                hashed.count == before.st_size
                and hashed.digest.hexdigest() == receipt["compressed_sha256"],
                "compressed hash/coverage",
            )
    except (EOFError, OSError, RecursionError) as error:
        raise ValueError("invalid/truncated evidence stream") from error
    after = path.stat()
    require(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        "evidence changed during read",
    )
    require(
        total == receipt["logical_bytes"]
        and count == receipt["records"]
        and logical.hexdigest() == receipt["logical_sha256"],
        "logical evidence binding",
    )


def _array(value, shape=None):
    require(
        isinstance(value, np.ndarray) and value.dtype.kind in "biuf", "saved numeric array required"
    )
    require(shape is None or value.shape == shape, "saved array shape")
    require(np.isfinite(value).all(), "nonfinite consumed metric")
    return value


def _type(value, name):
    require(type(value) is dict and value.get("__type__") == name, "saved object type")
    return value


def _close(a, b, message):
    # Saved FP64 reductions can differ only by summation grouping. This is
    # evidence consistency, not a new physics/learning tolerance.
    require(np.allclose(a, b, rtol=2e-12, atol=2e-14), message)


@dataclass(frozen=True)
class OfflineBudget:
    updates: int = 128
    rollout: int = 32
    evaluation_steps: int = 256
    horizon: int = 256

    def __post_init__(self):
        for name, maximum in (
            ("updates", 128),
            ("rollout", 32),
            ("evaluation_steps", 256),
            ("horizon", 256),
        ):
            value = getattr(self, name)
            require(type(value) is int and 1 <= value <= maximum, "offline budget")


class PhaseAccumulator:
    """All endpoint observations, grouped by declared world/joint and quarter."""

    def __init__(self, protocol, world_keys, *, steps, horizon, offset=0):
        self.keys = tuple(world_keys)
        require(self.keys and len(set(self.keys)) == len(self.keys), "world order")
        self.target = np.asarray(protocol["q_target"], np.float64)
        self.qscale = np.asarray(protocol["q_scale"], np.float64)
        self.vscale = np.asarray(protocol["v_scale"], np.float64)
        self.scale = float(protocol["action_scale"])
        self.h = float(protocol["h"])
        self.width = self.target.size
        require(
            self.target.ndim == 1
            and 0 < self.width <= 35
            and self.qscale.shape == self.target.shape
            and self.vscale.shape == self.target.shape
            and np.isfinite(self.target).all()
            and np.isfinite(self.qscale).all()
            and np.isfinite(self.vscale).all()
            and np.all(self.qscale > 0)
            and np.all(self.vscale > 0)
            and math.isfinite(self.scale)
            and self.scale > 0
            and math.isfinite(self.h)
            and self.h > 0,
            "objective metadata",
        )
        self.steps, self.horizon, self.offset, self.seen = steps, horizon, offset, 0
        shape = (len(self.keys), self.width)
        self.sums = {
            name: np.zeros(shape)
            for name in (
                "q2",
                "v2",
                "qphysical2",
                "vphysical2",
                "raw2",
                "ctrl2",
                "ctrl",
                "near_limit",
                "opposes_q",
                "ctrl_q",
                "ctrl_v",
            )
        }
        self.cost_sum = np.zeros(len(self.keys))
        self.quarter_sum = np.zeros((4, len(self.keys)))
        self.quarter_count = np.zeros(4, np.int64)
        self.max_error = np.zeros(shape)
        self.max_ctrl = np.zeros(shape)
        self.timeout = 0
        self.reward_error = 0.0
        self.support_sum = np.zeros(len(self.keys))
        self.support_four = np.zeros(len(self.keys), np.int64)
        self.support_count = 0
        self.support_peak = 0.0
        self.support_present = False
        self.last_q = None

    def add(self, transition, *, expected_action=None, support=False):
        value = _type(transition, TRANSITION_TYPE)
        n, d = len(self.keys), self.width
        absolute = self.offset + self.seen
        length, episode = absolute % self.horizon + 1, absolute // self.horizon
        require(
            self.seen < self.steps
            and value["sequence"] == absolute + 1
            and tuple(value["world_keys"]) == self.keys
            and set(value["states"]) == set(self.keys)
            and value["accepted_worlds"] == dict.fromkeys(self.keys, True),
            "transition sequence/world identity or refusal",
        )
        terminated = _array(value["terminated"], (n,))
        timeout = _array(value["time_outs"], (n,))
        require(
            not terminated.any() and np.all(timeout == (length == self.horizon)),
            "termination/timeout sequence",
        )
        require(
            np.all(_array(value["episode_length"], (n,)) == length)
            and np.all(_array(value["episode_id"], (n,)) == episode),
            "episode clock/reset identity",
        )
        actions = _array(value["action"], (n, d)).astype(np.float64)
        if expected_action is not None:
            require(np.array_equal(value["action"], expected_action), "requested action binding")
        channels = value["native_channels"]
        require(type(channels) in (tuple, list) and len(channels) == 3, "three native channels")
        controls = _array(channels[0], (n, d)).astype(np.float64)
        qforces = _array(channels[1], (n, d))
        xforces = _array(channels[2])
        require(
            xforces.ndim == 3
            and xforces.shape[0] == n
            and xforces.shape[2] == 6
            and not qforces.any()
            and not xforces.any(),
            "declared zero applied-force channels",
        )
        q, v, force, positive = [], [], [], []
        for row, key in enumerate(self.keys):
            state = _type(value["states"][key], STATE_TYPE)
            require(state["world"] == key and state["tick"] == length, "native state identity")
            q.append(_array(state["q"], (d,)).astype(np.float64))
            v.append(_array(state["v"], (d,)).astype(np.float64))
            require(
                np.array_equal(_array(state["ctrl"], (d,)), channels[0][row])
                and np.array_equal(_array(state["qfrc_applied"], (d,)), qforces[row])
                and np.array_equal(_array(state["xfrc_applied"]), xforces[row]),
                "actual state/native-channel binding",
            )
            receipt = state["last_step"]
            require(
                type(receipt) is dict and receipt.get("backend") == "mujoco-tyx-cuda",
                "actual TYX step evidence",
            )
            if support:
                contacts = [c for c in receipt["contacts"] if set(c["geometry_ids"]) == {0, 1}]
                loaded = [c for c in contacts if c["active"] and c["normal_force"] > 0]
                load = sum(float(c["normal_force"]) for c in loaded)
                require(math.isfinite(load) and len(loaded) <= 4, "support force coverage")
                force.append(load)
                positive.append(len(loaded) == 4)
        q, v = np.asarray(q), np.asarray(v)
        error = q - self.target
        qterm, vterm = (error / self.qscale) ** 2, 0.01 * (v / self.vscale) ** 2
        costs = np.mean(qterm, axis=1) + np.mean(vterm, axis=1)
        reward = _array(value["reward"], (n,)).astype(np.float64)
        require(
            np.allclose(reward, -costs, rtol=2e-5, atol=2e-6),
            "saved reward/native objective mismatch",
        )
        self.reward_error = max(self.reward_error, float(np.max(np.abs(reward + costs))))
        terms = {
            "q2": qterm,
            "v2": vterm,
            "qphysical2": error**2,
            "vphysical2": v**2,
            "raw2": actions**2,
            "ctrl2": controls**2,
            "ctrl": controls,
            "near_limit": np.abs(controls) >= 0.95 * self.scale,
            "opposes_q": controls * error < 0,
            "ctrl_q": controls * error,
            "ctrl_v": controls * v,
        }
        for name, term in terms.items():
            self.sums[name] += term
        self.cost_sum += costs
        quarter = min(3, self.seen * 4 // self.steps)
        self.quarter_sum[quarter] += costs
        self.quarter_count[quarter] += 1
        self.max_error = np.maximum(self.max_error, np.abs(error))
        self.max_ctrl = np.maximum(self.max_ctrl, np.abs(controls))
        self.timeout += int(np.count_nonzero(timeout))
        self.last_q = error.copy()
        if support:
            self.support_present = True
            self.support_peak = max(self.support_peak, max(force))
            if length >= 65:
                self.support_sum += force
                self.support_four += positive
                self.support_count += 1
        self.seen += 1
        return costs

    def finish(self):
        require(self.seen == self.steps, "incomplete diagnostic window")
        mean = {k: v / self.seen for k, v in self.sums.items()}
        result = {
            "steps": self.seen,
            "world_steps": self.seen * len(self.keys),
            "world_keys": self.keys,
            "cost": float(np.mean(self.cost_sum / self.seen)),
            "per_world_cost": (self.cost_sum / self.seen).tolist(),
            "q_cost_per_world_joint": mean["q2"].tolist(),
            "v_cost_per_world_joint": mean["v2"].tolist(),
            "q_error_rms_native": np.sqrt(mean["qphysical2"]).tolist(),
            "velocity_rms_native": np.sqrt(mean["vphysical2"]).tolist(),
            "maximum_absolute_q_error": self.max_error.tolist(),
            "final_q_error_native": self.last_q.tolist(),
            "raw_action_rms_per_world_joint": np.sqrt(mean["raw2"]).tolist(),
            "ctrl_rms_per_world_joint": np.sqrt(mean["ctrl2"]).tolist(),
            "ctrl_mean_per_world_joint": mean["ctrl"].tolist(),
            "ctrl_abs_max_per_world_joint": self.max_ctrl.tolist(),
            "ctrl_at_least_95pct_scale_fraction": mean["near_limit"].tolist(),
            "ctrl_opposes_endpoint_error_fraction": mean["opposes_q"].tolist(),
            "ctrl_times_endpoint_q_error": mean["ctrl_q"].tolist(),
            "ctrl_times_endpoint_velocity": mean["ctrl_v"].tolist(),
            "quarter_world_cost": [
                (self.quarter_sum[i] / count).tolist() if count else None
                for i, count in enumerate(self.quarter_count)
            ],
            "quarter_steps": self.quarter_count.tolist(),
            "timeout_worlds": self.timeout,
            "terminated_worlds": 0,
            "reward_cost_max_absolute_difference": self.reward_error,
        }
        if self.support_present:
            result["actual_step_support"] = {
                "tail_step_samples": self.support_count,
                "peak_force_N": self.support_peak,
                "tail_mean_force_N": (self.support_sum / self.support_count).tolist()
                if self.support_count
                else None,
                "tail_four_positive_fraction": (self.support_four / self.support_count).tolist()
                if self.support_count
                else None,
            }
        return result


def gradient_evidence(optimization, expected_step):
    require(
        optimization.get("finite_gradient") is True
        and optimization.get("optimizer_step") == expected_step,
        "saved gradient/Adam evidence",
    )
    norms = optimization.get("gradient_norms")
    result = {
        "optimizer_step": expected_step,
        "finite_gradient": True,
        "pre_clip_norm": "UNKNOWN",
        "gradient_directions": "UNKNOWN",
        "per_owner_assignment": "UNKNOWN",
        "combined_post_clip_norm": "UNKNOWN",
        "parameter_norms_in_saved_order": "UNKNOWN",
    }
    if norms is not None:
        values = np.asarray(norms, np.float64)
        require(
            values.ndim == 1
            and values.size > 0
            and np.isfinite(values).all()
            and np.all(values >= 0),
            "saved post-clip gradient norms",
        )
        combined = math.hypot(*values)
        require(math.isfinite(combined), "finite combined saved gradient norm")
        result.update(
            combined_post_clip_norm=combined,
            parameter_norms_in_saved_order=values.tolist(),
            nonzero_parameter_gradients=int(np.count_nonzero(values)),
        )
    return result


def _policy_hash(policy):
    digest = hashlib.sha256()
    for owner in ("actor", "critic"):
        require(type(policy[owner]) is dict, "saved model state mapping")
        for name, tensor in sorted(policy[owner].items()):
            a = _array(tensor)
            digest.update((owner + ":" + name + ":" + a.dtype.str + ":" + str(a.shape)).encode())
            digest.update(a.tobytes(order="C"))
    return digest.hexdigest()


def parameter_changes(initial, final):
    result = {}
    for owner in ("actor", "critic"):
        require(set(initial[owner]) == set(final[owner]), "saved model tensor keys")
        before_squared = delta_squared = 0.0
        count = changed = 0
        for name in initial[owner]:
            a, b = _array(initial[owner][name]), _array(final[owner][name])
            require(a.shape == b.shape and a.dtype == b.dtype, "saved model tensor layout")
            x, y = a.astype(np.float64), b.astype(np.float64)
            before_squared += float(np.sum(x * x))
            delta_squared += float(np.sum((y - x) ** 2))
            count += a.size
            changed += int(np.count_nonzero(a != b))
        result[owner] = {
            "saved_state_scalar_count": count,
            "changed_scalars": changed,
            "l2_change": math.sqrt(delta_squared),
            "relative_l2_change": math.sqrt(delta_squared / before_squared)
            if before_squared
            else None,
            "scope": "all saved state tensors, not inferred parameter-only names",
        }
    return result


def _bounded_json(path, maximum):
    path = Path(path)
    require(path.is_file() and 0 < path.stat().st_size <= maximum, "bounded metadata file required")
    raw = path.read_bytes()
    require(len(raw) <= maximum, "metadata changed size")
    return strict_json(raw), hashlib.sha256(raw).hexdigest()


class _Cursor:
    def __init__(self, records):
        self.records = iter(records)

    def take(self, expected):
        try:
            kind, value = next(self.records)
        except StopIteration as error:
            raise ValueError("missing record: " + expected) from error
        require(kind == expected, "record phase/order: expected " + expected + ", got " + kind)
        return value

    def finish(self):
        try:
            next(self.records)
        except StopIteration:
            return
        raise ValueError("unexpected record after completion")


def _check_phase(result, summary):
    require(
        result["world_steps"] == summary["world_steps"] and result["steps"] == summary["steps"],
        "evaluation window summary",
    )
    _close(result["cost"], summary["cost"], "evaluation cost summary")
    _close(result["per_world_cost"], summary["per_world_cost"], "per-world evaluation summary")


def _evaluation(cursor, name, protocol, keys, budget):
    stats = PhaseAccumulator(protocol, keys, steps=budget.evaluation_steps, horizon=budget.horizon)
    for tick in range(1, budget.evaluation_steps + 1):
        request = cursor.take(name + ".requested_action")
        require(request["tick"] == tick, "evaluation request tick")
        action = _array(request["action"], (len(keys), stats.width))
        if name == "zero":
            require(not action.any(), "zero diagnostic has nonzero requested action")
        costs = stats.add(
            cursor.take(name + ".transition"),
            expected_action=action,
            support=protocol.get("name") == "S",
        )
        independent = cursor.take(name + ".independent")
        require(independent["tick"] == tick, "independent cost tick")
        _close(_array(independent["cost"], (len(keys),)), costs, "saved independent cost")
    return stats.finish()


def diagnose_seed(directory, binding, *, expected_task, expected_seed, budget=OfflineBudget()):
    """Read one complete bound seed. Never returns successful partial analysis."""
    directory = Path(directory).resolve()
    summary, digest = _bounded_json(directory / "summary.json", 16 * 1024**2)
    publication, _ = _bounded_json(directory / "publication.json", 16384)
    manifest, manifest_hash = _bounded_json(directory / "manifest.json", 4 * 1024**2)
    require(
        digest == binding["summary_sha256"] == publication.get("summary_sha256"),
        "summary publication/expected binding",
    )
    require(Path(binding["directory"]).resolve() == directory, "seed directory identity")
    require(
        summary.get("task") == expected_task
        and summary.get("seed") == expected_seed
        and summary.get("status") == "COMPLETE_N4A_SEED",
        "seed task/status identity",
    )
    for key in ("task", "seed", "protocol_binding", "ppo_config", "source_files", "dependencies"):
        require(summary.get(key) == manifest.get(key), "manifest/summary mismatch: " + key)
    require(manifest.get("phase") == "learn", "learning manifest required")
    keys = tuple(manifest["world_keys"])
    require(
        keys
        and len(keys) <= 256
        and len(set(keys)) == len(keys)
        and all(type(key) is int for key in keys),
        "declared native world order",
    )
    protocol = summary["protocol_binding"]
    require(
        protocol.get("horizon") == budget.horizon
        and protocol.get("action_squash") == "tanh"
        and protocol.get("decimation") == 1,
        "supported saved task/action protocol",
    )
    require(
        summary["updates"] == budget.updates
        and summary["worlds"] == len(keys)
        and summary["accepted_world_steps"] == budget.updates * budget.rollout * len(keys),
        "complete frozen update budget",
    )
    require(summary["evidence"] == binding["evidence"], "trace bound to selected source summary")
    require(len(summary["update_evidence"]) == budget.updates, "update evidence count")
    records = read_records(directory / "trace.jsonl.gz", summary["evidence"])
    try:
        return _analyze_trace(
            records, summary, manifest, digest, manifest_hash, budget, expected_task, expected_seed
        )
    finally:
        records.close()


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _analyze_trace(
    records, summary, manifest, digest, manifest_hash, budget, expected_task, expected_seed
):
    keys = tuple(manifest["world_keys"])
    protocol = summary["protocol_binding"]
    cursor = _Cursor(records)
    initial = cursor.take("initial_policy")
    require(_policy_hash(initial) == summary["parameter_hash_before"], "initial policy hash")
    phases = {"untrained": _evaluation(cursor, "untrained", protocol, keys, budget)}
    _check_phase(phases["untrained"], summary["baseline"])
    zero = summary["zero_action_diagnostic"]
    if zero is not None:
        require(
            zero["status"] == "COMPLETE_DIAGNOSTIC",
            "this diagnosis requires the bound complete zero window",
        )
        phases["zero"] = _evaluation(cursor, "zero", protocol, keys, budget)
        _check_phase(phases["zero"], zero)
    updates = []
    previous_hash = summary["parameter_hash_before"]
    for index in range(budget.updates):
        record = cursor.take("update")
        require(
            record["iteration"] == index
            and record["next_iteration"] == index + 1
            and len(record["transitions"]) == budget.rollout,
            "update sequence/rollout",
        )
        stats = PhaseAccumulator(
            protocol,
            keys,
            steps=budget.rollout,
            horizon=budget.horizon,
            offset=index * budget.rollout,
        )
        for transition in record["transitions"]:
            stats.add(transition, support=expected_task == "S")
        behavior = stats.finish()
        check = cursor.take("update_checks")
        require(
            _plain(check) == summary["update_evidence"][index],
            "complete saved update check binding",
        )
        require(
            check["iteration"] == index
            and check["optimizer_step"] == index + 1
            and check["parameter_hash_before"] == previous_hash
            and check["parameter_hash_before"] != check["parameter_hash_after"],
            "parameter hash/update chain",
        )
        require(
            check["accepted_world_steps"] == behavior["world_steps"]
            and check["terminated_worlds"] == 0
            and check["timeout_reset_worlds"] == behavior["timeout_worlds"],
            "update complete transition/done coverage",
        )
        _close(check["mean_native_cost"], behavior["cost"], "update native cost summary")
        require(
            record["loss"] == check["loss"]
            and all(math.isfinite(float(v)) for v in record["loss"].values()),
            "saved finite loss binding",
        )
        previous_hash = check["parameter_hash_after"]
        updates.append(
            {
                "iteration": index,
                "behavior": behavior,
                "loss": record["loss"],
                "gradient": gradient_evidence(record["optimization"], index + 1),
                "parameter_hash_before": check["parameter_hash_before"],
                "parameter_hash_after": previous_hash,
            }
        )
    phases["learned"] = _evaluation(cursor, "learned", protocol, keys, budget)
    _check_phase(phases["learned"], summary["learned"])
    final = cursor.take("final_policy_and_adam")
    require(
        final["next_iteration"] == budget.updates
        and _policy_hash(final) == previous_hash == summary["parameter_hash_after"],
        "final learner/update identity",
    )
    optimizer_steps = [
        float(_array(state["step"]).reshape(()))
        for state in final["optimizer"]["state"].values()
        if "step" in state
    ]
    require(
        optimizer_steps and all(step == budget.updates for step in optimizer_steps),
        "final saved Adam step identity",
    )
    completion = cursor.take("completion")
    require(
        completion == {"status": summary["status"], "task": expected_task}, "completion identity"
    )
    cursor.finish()  # Forces both complete stream hashes/byte counts before result publication.
    return {
        "status": "PASS_SAVED_DIAGNOSIS_ONLY",
        "task": expected_task,
        "seed": expected_seed,
        "summary_sha256": digest,
        "manifest_sha256": manifest_hash,
        "evidence": summary["evidence"],
        "protocol_binding": protocol,
        "world_keys": keys,
        "source_files": summary["source_files"],
        "dependencies": summary["dependencies"],
        "ppo_config": summary["ppo_config"],
        "phases": phases,
        "training_updates": updates,
        "saved_model_changes": parameter_changes(initial, final),
        "final_adam_steps": optimizer_steps,
        "unknown": {
            "preclip_gradient_norms": "UNKNOWN",
            "gradient_vectors_and_direction": "UNKNOWN",
            "value_predictions_returns_advantages_and_KL": "UNKNOWN",
            "actor_vs_critic_gradient_assignment": "UNKNOWN",
            "causal_attribution_of_learning_quality": "NOT_ESTABLISHED_BY_SAVED_CORRELATIONS",
        },
        "scope": "saved array arithmetic only; no policy forward, optimizer, simulator or new learning gate",
    }
