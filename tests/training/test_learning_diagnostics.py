"""Independent saved-only diagnostics fixtures; no runtime, CUDA, or learner."""
import base64
import gzip
import hashlib
import json

import numpy as np
import pytest

from phymeo.training.learning_diagnostics import (
    OfflineBudget, PhaseAccumulator, decode_evidence, diagnose_seed,
    gradient_evidence, parameter_changes, read_records, strict_json,
)


def wire(value):
    if isinstance(value, np.ndarray):
        raw = value.tobytes()
        return {"array_dtype": value.dtype.str, "shape": list(value.shape),
                "bytes_base64": base64.b64encode(raw).decode(),
                "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, dict):
        if "__type__" in value:
            return {"dataclass": value["__type__"],
                    "fields": {k: wire(v) for k, v in value.items() if k != "__type__"}}
        return {"mapping": [[wire(k), wire(v)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {"sequence_type": type(value).__name__, "items": [wire(x) for x in value]}
    return value


def array(value, dtype=np.float64):
    return np.asarray(value, dtype=dtype)


def transition(tick, *, action=0.25, end=False, episode=0):
    keys = (5, 41)
    q = array([0.1])
    v = array([0.2])
    ctrl = array([action], np.float32)
    states = {}
    for key in keys:
        states[key] = {
            "__type__": "phymeo.backends.mujoco_runtime.MuJoCoState",
            "world": key, "q": q.copy(), "v": v.copy(), "ctrl": ctrl.copy(),
            "qfrc_applied": array([0.0]), "xfrc_applied": np.zeros((2, 6)),
            "tick": tick, "time": tick * 0.1,
            "last_step": {"__type__": "fixture.Step", "backend": "mujoco-tyx-cuda",
                          "contacts": (), "input_q": q.copy()},
        }
    return {
        "__type__": "phymeo.training.mujoco_tyx_env.MuJoCoTyxTransition",
        "sequence": episode * 2 + tick, "world_keys": keys, "states": states,
        "accepted_worlds": dict.fromkeys(keys, True),
        "action": np.full((2, 1), action, np.float32),
        "native_channels": (np.full((2, 1), action, np.float32),
                            np.zeros((2, 1), np.float32), np.zeros((2, 2, 6), np.float32)),
        "reward": array([-0.0104, -0.0104], np.float32),
        "terminated": array([False, False], bool),
        "time_outs": array([end, end], bool),
        "episode_length": array([tick, tick], np.int64),
        "episode_id": array([episode, episode], np.int64),
    }


def protocol():
    return {"q_target": [0.0], "q_scale": [1.0], "v_scale": [1.0],
            "action_scale": 1.0, "h": 0.1, "horizon": 2,
            "action_squash": "tanh", "decimation": 1}


def write_records(path, records):
    raw = b"".join((json.dumps({"ordinal": i, "kind": k, "payload": wire(v)},
                             separators=(",", ":")) + "\n").encode()
                   for i, (k, v) in enumerate(records))
    compressed = gzip.compress(raw, mtime=0)
    path.write_bytes(compressed)
    return {"path": path.name, "records": len(records), "logical_bytes": len(raw),
            "logical_sha256": hashlib.sha256(raw).hexdigest(),
            "compressed_bytes": len(compressed),
            "compressed_sha256": hashlib.sha256(compressed).hexdigest()}


def test_decoder_is_data_only_and_preserves_typed_keys():
    value = {"__type__": "does.not.exist.Type", "x": {5: array([1.0]), "5": (True,)}}
    result = decode_evidence(wire(value))
    assert result["__type__"] == "does.not.exist.Type"
    assert np.array_equal(result["x"][5], array([1.0]))
    assert result["x"]["5"] == (True,)


@pytest.mark.parametrize("change", ["hash", "shape", "object", "base64", "negative"])
def test_decoder_rejects_corrupt_or_unbounded_array(change):
    item = wire(array([1.0]))
    if change == "hash":
        item["sha256"] = "0" * 64
    elif change == "shape":
        item["shape"] = [2**40]
    elif change == "object":
        item["array_dtype"] = "|O"
    elif change == "negative":
        item["shape"] = [-1]
    else:
        item["bytes_base64"] = "***"
    with pytest.raises(ValueError):
        decode_evidence(item)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_strict_json_refuses_ambiguous_metadata(raw):
    with pytest.raises(ValueError):
        strict_json(raw)


def test_stream_reads_complete_hash_bound_sequence(tmp_path):
    path = tmp_path / "trace.jsonl.gz"
    receipt = write_records(path, [("one", {"a": array([3.0])}), ("two", 7)])
    records = list(read_records(path, receipt))
    assert records[0][0] == "one"
    assert records[1] == ("two", 7)


@pytest.mark.parametrize("defect", ["compressed", "logical", "truncated", "count", "ordinal"])
def test_stream_refuses_corruption_and_partial_prefix(tmp_path, defect):
    path = tmp_path / "trace.jsonl.gz"
    receipt = write_records(path, [("one", 1), ("two", 2)])
    if defect in ("compressed", "logical"):
        receipt[defect + "_sha256"] = "0" * 64
    elif defect == "count":
        receipt["records"] = 3
    elif defect == "truncated":
        path.write_bytes(path.read_bytes()[:-6])
        receipt["compressed_bytes"] = path.stat().st_size
    else:
        raw = b'{"ordinal":1,"kind":"one","payload":1}\n'
        path.write_bytes(gzip.compress(raw, mtime=0))
        receipt.update(compressed_bytes=path.stat().st_size,
                       compressed_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        list(read_records(path, receipt))


def test_phase_decomposes_cost_action_and_endpoint_products():

    stats = PhaseAccumulator(protocol(), (5, 41), steps=2, horizon=2)
    stats.add(transition(1))
    stats.add(transition(2, end=True))
    result = stats.finish()
    assert result["cost"] == pytest.approx(0.0104)
    assert np.allclose(result["q_cost_per_world_joint"], [[0.01], [0.01]])
    assert np.allclose(result["v_cost_per_world_joint"], [[0.0004], [0.0004]])
    assert result["timeout_worlds"] == 2
    assert np.allclose(result["ctrl_rms_per_world_joint"], [[0.25], [0.25]])
    assert np.allclose(result["ctrl_times_endpoint_q_error"], [[0.025], [0.025]])
    assert np.allclose(result["ctrl_times_endpoint_velocity"], [[0.05], [0.05]])


@pytest.mark.parametrize("defect", ["world", "native_ctrl", "reward", "early_done", "missing_step"])
def test_phase_refuses_wrong_world_control_reward_and_incomplete_window(defect):
    stats = PhaseAccumulator(protocol(), (5, 41), steps=2, horizon=2)
    item = transition(1)
    if defect == "world":
        item["states"][5]["world"] = 41
    elif defect == "native_ctrl":
        item["states"][5]["ctrl"] = array([0.9])
    elif defect == "reward":
        item["reward"] = array([0.0, 0.0])
    elif defect == "early_done":
        item["time_outs"] = array([True, True], bool)
    with pytest.raises(ValueError):
        stats.add(item)
        stats.finish()


def test_gradient_norms_are_post_clip_and_missing_is_unknown():
    known = gradient_evidence({"finite_gradient": True, "optimizer_step": 3,
                               "gradient_norms": (0.3, 0.4)}, 3)
    assert known["combined_post_clip_norm"] == pytest.approx(0.5)
    assert known["pre_clip_norm"] == "UNKNOWN"
    missing = gradient_evidence({"finite_gradient": True, "optimizer_step": 3}, 3)
    assert missing["combined_post_clip_norm"] == "UNKNOWN"
    with pytest.raises(ValueError):
        gradient_evidence({"finite_gradient": False, "optimizer_step": 3}, 3)
    with pytest.raises(ValueError):
        gradient_evidence({"finite_gradient": True, "optimizer_step": 2}, 3)


def test_parameter_changes_separate_actor_and_critic():
    initial = {"actor": {"weight": array([1.0, 2.0])}, "critic": {"weight": array([3.0])}}
    final = {"actor": {"weight": array([1.0, 3.0])}, "critic": {"weight": array([3.0])}}
    result = parameter_changes(initial, final)
    assert result["actor"]["l2_change"] == 1.0
    assert result["critic"]["l2_change"] == 0.0
    final["actor"]["weight"] = array([1.0])
    with pytest.raises(ValueError):
        parameter_changes(initial, final)


def seed_fixture(tmp_path, *, missing_gradient=False):
    root = tmp_path / "seed"
    root.mkdir()
    budget = OfflineBudget(updates=2, rollout=2, evaluation_steps=2, horizon=2)
    initial = {"actor": {"weight": array([1.0])}, "critic": {"weight": array([2.0])}}
    records = [("initial_policy", initial)]
    summary_phase = {"world_steps": 4, "steps": 2, "cost": 0.0104,
                     "per_world_cost": [0.0104, 0.0104]}
    for phase in ("untrained", "zero"):
        for tick in (1, 2):
            item = transition(tick, action=0.0 if phase == "zero" else 0.25, end=tick == 2)
            records.extend([(phase + ".requested_action", {"tick": tick, "action": item["action"]}),
                            (phase + ".transition", item),
                            (phase + ".independent", {"tick": tick, "cost": array([0.0104]*2),
                                                     "inspection": None})])
    checks = []
    hashes = ["1"*64, "2"*64, "3"*64]
    for index in range(2):
        optimization = {"finite_gradient": True, "optimizer_step": index + 1}
        if not missing_gradient:
            optimization["gradient_norms"] = (0.3, 0.4)
        loss = {"value": 1.0, "surrogate": 0.0, "entropy": -1.0}
        records.append(("update", {"iteration": index, "next_iteration": index+1,
                                  "transitions": tuple(transition(t, end=t == 2, episode=index)
                                                       for t in (1, 2)),
                                  "loss": loss, "optimization": optimization}))
        check = {"iteration": index, "optimizer_step": index+1, "accepted_world_steps": 4,
                 "mean_native_cost": 0.0104, "terminated_worlds": 0, "timeout_reset_worlds": 2,
                 "loss": loss, "parameter_hash_before": hashes[index],
                 "parameter_hash_after": hashes[index+1]}
        checks.append(check)
        records.append(("update_checks", check))
    for tick in (1, 2):
        item = transition(tick, end=tick == 2)
        records.extend([("learned.requested_action", {"tick": tick, "action": item["action"]}),
                        ("learned.transition", item),
                        ("learned.independent", {"tick": tick, "cost": array([0.0104]*2),
                                                "inspection": None})])
    records.extend([("final_policy_and_adam", {"actor": {"weight": array([1.1])},
                                              "critic": {"weight": array([2.1])},
                                              "optimizer": {"state": {}}, "next_iteration": 2}),
                    ("completion", {"status": "COMPLETE_N4A_SEED", "task": "fixture"})])
    receipt = write_records(root / "trace.jsonl.gz", records)
    common = {"task": "fixture", "seed": 7, "protocol_binding": protocol(),
              "ppo_config": {"algorithm": "fixture"}, "source_files": {"fixture": "a"*64},
              "dependencies": {"numpy": "fixture"}}
    summary = {**common, "status": "COMPLETE_N4A_SEED", "updates": 2, "worlds": 2,
               "accepted_world_steps": 8, "baseline": summary_phase, "learned": summary_phase,

               "zero_action_diagnostic": {**summary_phase, "status": "COMPLETE_DIAGNOSTIC"},
               "update_evidence": checks, "parameter_hash_before": hashes[0],
               "parameter_hash_after": hashes[-1], "evidence": receipt}
    raw = json.dumps(summary, sort_keys=True).encode()
    (root / "summary.json").write_bytes(raw)
    (root / "manifest.json").write_text(json.dumps({**common, "phase": "learn", "world_keys": [5,41]}))
    digest = hashlib.sha256(raw).hexdigest()
    (root / "publication.json").write_text(json.dumps({"summary_sha256": digest}))
    binding = {"directory": str(root), "summary_sha256": digest, "evidence": receipt}
    return root, binding, budget


@pytest.mark.parametrize("missing_gradient", [False, True])
def test_complete_synthetic_seed_is_streamed_and_missing_gradient_not_invented(tmp_path, missing_gradient):
    root, binding, budget = seed_fixture(tmp_path, missing_gradient=missing_gradient)
    result = diagnose_seed(root, binding, expected_task="fixture", expected_seed=7, budget=budget)
    assert result["status"] == "PASS_SAVED_DIAGNOSIS_ONLY"
    assert result["phases"]["learned"]["cost"] == pytest.approx(0.0104)
    assert len(result["training_updates"]) == 2
    value = result["training_updates"][0]["gradient"]["combined_post_clip_norm"]
    assert value == ("UNKNOWN" if missing_gradient else pytest.approx(0.5))


@pytest.mark.parametrize("defect", ["identity", "publication", "manifest", "summary_binding"])
def test_seed_refuses_wrong_identity_and_changed_bindings(tmp_path, defect):
    root, binding, budget = seed_fixture(tmp_path)
    task = "other" if defect == "identity" else "fixture"
    if defect == "publication":
        (root / "publication.json").write_text('{"summary_sha256":"bad"}')
    elif defect == "manifest":
        value = json.loads((root / "manifest.json").read_text())
        value["protocol_binding"]["h"] = 0.2
        (root / "manifest.json").write_text(json.dumps(value))
    elif defect == "summary_binding":
        binding["summary_sha256"] = "0"*64
    with pytest.raises(ValueError):
        diagnose_seed(root, binding, expected_task=task, expected_seed=7, budget=budget)
