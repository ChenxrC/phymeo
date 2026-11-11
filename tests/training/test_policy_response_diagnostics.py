"""Independent N4c stream/grid cases; actual CUDA cases use pinned upstream MLP."""

import base64
import copy
import gzip
import hashlib
import json
import random
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

from phymeo.training import policy_response_diagnostics as response


def protocol(width=2):
    return {
        "name": "fixture",
        "q_target": [0.0] * width,
        "q_scale": [0.2] * width,
        "v_scale": [1.0] * width,
        "q_lower": [-1.0] * width,
        "q_upper": [1.0] * width,
        "v_bound": 8.0,
        "action_scale": 2.0,
        "action_squash": "tanh",
        "decimation": 1,
        "h": 0.002,
        "horizon": 256,
        "held_out_templates": [[[(i - 3) / 100] * width, [0.02] * width] for i in range(8)],
    }


def config():
    network = {
        "class_name": "rsl_rl.models.MLPModel",
        "hidden_dims": [32, 32],
        "activation": "tanh",
        "obs_normalization": False,
    }
    return {
        "actor": {
            **network,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
                "init_std": 0.1,
                "std_type": "log",
                "learn_std": True,
            },
        },
        "critic": network.copy(),
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 32,
    }


def entry(proto):
    values = np.asarray(proto["held_out_templates"], np.float32)
    return {"q": values[:, 0].copy(), "v": values[:, 1].copy()}


def wire(value):
    if isinstance(value, np.ndarray):
        raw = value.tobytes()
        return {
            "array_dtype": value.dtype.str,
            "shape": list(value.shape),
            "bytes_base64": base64.b64encode(raw).decode(),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(value, dict):
        if "__type__" in value:
            return {
                "dataclass": value["__type__"],
                "fields": {k: wire(v) for k, v in value.items() if k != "__type__"},
            }
        return {"mapping": [[wire(k), wire(v)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {"sequence_type": type(value).__name__, "items": [wire(x) for x in value]}
    return value


def policy_hash(model):
    digest = hashlib.sha256()
    for owner in ("actor", "critic"):
        for name, value in sorted(model[owner].items()):
            digest.update(
                (owner + ":" + name + ":" + value.dtype.str + ":" + str(value.shape)).encode()
            )
            digest.update(value.tobytes())
    return digest.hexdigest()


def source_bindings():
    root = Path(response.__file__).resolve().parents[1]
    path = root / "backends" / "mujoco_tyx" / "_task_kernels.py"
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture(tmp_path, *, edit=None):
    proto, keys = protocol(), tuple(5 + 36 * i for i in range(8))
    native = entry(proto)
    initial = {
        "actor": {"weight": np.asarray([1.0], np.float32)},
        "critic": {"weight": np.asarray([2.0], np.float32)},
    }
    final = {
        "actor": {"weight": np.asarray([1.5], np.float32)},
        "critic": {"weight": np.asarray([2.5], np.float32)},
    }
    action = np.zeros((8, 2), np.float32)
    states = {
        key: {
            "__type__": "phymeo.backends.mujoco_runtime.MuJoCoState",
            "world": key,
            "tick": 1,
            "last_step": {
                "__type__": "phymeo.backends._mujoco_tyx_runtime.MuJoCoTyxStepReceipt",
                "world": key,
                "tick": 1,
                "input_time": 0.0,
                "h": 0.002,
                "input_q": native["q"][i].astype(np.float64),
                "input_v": native["v"][i].astype(np.float64),
            },
        }
        for i, key in enumerate(keys)
    }
    transition = {
        "__type__": "phymeo.training.mujoco_tyx_env.MuJoCoTyxTransition",
        "world_keys": keys,
        "states": states,
        "action": action,
    }
    records = [("initial_policy", initial)]
    for phase in ("untrained", "learned"):
        records += [
            (phase + ".requested_action", {"tick": 1, "action": action}),
            (phase + ".transition", transition),
        ]
    records += [
        ("final_policy_and_adam", {**final, "optimizer": {}, "next_iteration": 128}),
        ("completion", {"status": "COMPLETE_N4A_SEED", "task": "fixture"}),
    ]
    if edit is not None:
        edit(records)
    raw = b"".join(
        (json.dumps({"ordinal": i, "kind": k, "payload": wire(v)}) + "\n").encode()
        for i, (k, v) in enumerate(records)
    )
    packed = gzip.compress(raw, mtime=0)
    (tmp_path / "trace.jsonl.gz").write_bytes(packed)
    evidence = {
        "path": "trace.jsonl.gz",
        "records": len(records),
        "logical_bytes": len(raw),
        "logical_sha256": hashlib.sha256(raw).hexdigest(),
        "compressed_bytes": len(packed),
        "compressed_sha256": hashlib.sha256(packed).hexdigest(),
    }
    common = {
        "task": "fixture",
        "seed": 7,
        "protocol_binding": proto,
        "ppo_config": config(),
        "source_files": source_bindings(),
        "dependencies": {"fixture": "synthetic"},
    }
    summary = {
        **common,
        "status": "COMPLETE_N4A_SEED",
        "evidence": evidence,
        "parameter_hash_before": policy_hash(initial),
        "parameter_hash_after": policy_hash(final),
        "updates": 128,
        "worlds": 8,
        "accepted_world_steps": 32768,
    }
    encoded = json.dumps(summary).encode()
    (tmp_path / "summary.json").write_bytes(encoded)
    (tmp_path / "publication.json").write_text(
        json.dumps({"summary_sha256": hashlib.sha256(encoded).hexdigest()})
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({**common, "phase": "learn", "world_keys": keys})
    )
    return {
        "directory": str(tmp_path),
        "summary_sha256": hashlib.sha256(encoded).hexdigest(),
        "evidence": evidence,
    }


@pytest.mark.parametrize("width,points", [(1, 13), (2, 17), (3, 21)])
def test_exact_fixed_grid(width, points):
    proto = protocol(width)
    grid = response.fixed_grid(proto, entry(proto))
    assert grid["q"].shape == grid["v"].shape == (points, width)
    assert grid["labels"][:8] == tuple("held-out-" + str(i) for i in range(8))
    assert grid["labels"][8] == "target"
    assert np.array_equal(grid["q"][:8].view(np.uint32), entry(proto)["q"].view(np.uint32))
    assert np.array_equal(grid["q"][8], np.zeros(width, np.float32))
    for j, (plus, minus) in enumerate(grid["q_pairs"]):
        assert grid["q"][plus, j] == np.float32(0.05)
        assert grid["q"][minus, j] == np.float32(-0.05)
        assert not grid["v"][[plus, minus]].any()


@pytest.mark.parametrize("defect", ["scale", "nonfinite", "template", "entry", "collapsed"])
def test_grid_rejects_changed_or_invalid_inputs(defect):
    proto = protocol()
    native = entry(proto)
    if defect == "scale":
        proto["q_scale"][0] = 0.0
    elif defect == "nonfinite":
        proto["v_scale"][0] = float("nan")
    elif defect == "template":
        proto["held_out_templates"].pop()
    elif defect == "entry":
        native["q"][0, 0] = 0.25
    else:
        proto["q_target"][0] = 1e20
        proto["q_lower"][0], proto["q_upper"][0] = -1e21, 1e21
    with pytest.raises(ValueError):
        response.fixed_grid(proto, native)


def test_known_even_odd_and_cross_joint_response():
    grid = response.fixed_grid(protocol(), entry(protocol()))
    weights = np.asarray([[2.0, -1.0], [3.0, 4.0]])
    raw = grid["q"].astype(float) @ weights + grid["v"].astype(float) * 0.5 + [0.2, -0.3]
    result = response.response_components(raw, raw, grid)
    assert np.allclose(result["raw"]["target"], [0.2, -0.3])
    assert np.allclose(result["raw"]["q"]["secant_native"], weights)
    assert np.allclose(result["raw"]["v"]["secant_native"], np.eye(2) * 0.5)
    assert np.allclose(result["ctrl"]["q"]["even_minus_target"], 0)


def test_complete_bound_policy_pair_extraction(tmp_path):
    binding = fixture(tmp_path)
    pair = response.read_policy_pair(tmp_path, binding, expected_task="fixture", expected_seed=7)
    assert pair["hashes"]["initial"] != pair["hashes"]["final"]
    assert np.array_equal(pair["entry"]["q"], entry(protocol())["q"])
    assert set(pair["reference_actions"]) == {"initial", "final"}
    assert pair["world_keys"] == tuple(5 + 36 * i for i in range(8))


@pytest.mark.parametrize(
    "defect",
    ["missing_initial", "duplicate_initial", "missing_final", "hash", "entry", "first_request"],
)
def test_extractor_rejects_bound_schema_corruption(tmp_path, defect):
    def edit(records):
        if defect == "missing_initial":
            records.pop(0)
        elif defect == "duplicate_initial":
            records.insert(1, records[0])
        elif defect == "missing_final":
            records.pop(-2)
        elif defect == "hash":
            records[0][1]["actor"] = {"weight": np.asarray([9.0], np.float32)}
        elif defect == "entry":
            records[2][1]["states"][5]["last_step"]["input_q"][0] = 0.25
        else:
            records[1][1]["action"] = np.ones((8, 2), np.float32)

    binding = fixture(tmp_path, edit=edit)
    if defect == "hash":
        # The trusted expected model hash must remain the originally authored value.
        value = json.loads((tmp_path / "summary.json").read_text())
        value["parameter_hash_before"] = "0" * 64
        encoded = json.dumps(value).encode()
        (tmp_path / "summary.json").write_bytes(encoded)
        digest = hashlib.sha256(encoded).hexdigest()
        (tmp_path / "publication.json").write_text(json.dumps({"summary_sha256": digest}))
        binding["summary_sha256"] = digest
    with pytest.raises(ValueError):
        response.read_policy_pair(tmp_path, binding, expected_task="fixture", expected_seed=7)


@pytest.mark.parametrize("defect", ["seed", "publication", "truncated", "source"])
def test_extractor_rejects_identity_and_integrity_failures(tmp_path, defect):
    binding = fixture(tmp_path)
    if defect == "publication":
        (tmp_path / "publication.json").write_text('{"summary_sha256":"bad"}')
    elif defect == "truncated":
        path = tmp_path / "trace.jsonl.gz"
        path.write_bytes(path.read_bytes()[:-8])
    elif defect == "source":
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        manifest["source_files"] = {}
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        response.read_policy_pair(
            tmp_path, binding, expected_task="fixture", expected_seed=8 if defect == "seed" else 7
        )


@pytest.mark.parametrize("defect", ["class", "normalizer", "distribution", "groups"])
def test_network_config_never_resolves_arbitrary_classes(defect):
    value = config()
    if defect == "class":
        value["actor"]["class_name"] = "os.system"
    elif defect == "normalizer":
        value["actor"]["obs_normalization"] = True
    elif defect == "distribution":
        value["actor"]["distribution_cfg"]["class_name"] = "elsewhere.Unknown"
    else:
        value["obs_groups"]["actor"] = ["unknown"]
    with pytest.raises(ValueError):
        response.validate_model_config(value)


def test_dependency_mismatch_is_explicit(monkeypatch):
    monkeypatch.setattr(response, "version", lambda name: "different")
    with pytest.raises(ValueError):
        response.check_dependencies({name: "saved" for name in response.DEPENDENCIES})


def cuda_pair():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensordict")
    pytest.importorskip("rsl_rl")
    if not torch.cuda.is_available():
        pytest.skip("dedicated N4c qualification requires real CUDA")
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    proto, cfg = protocol(), config()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(17401)
        obs = TensorDict({"policy": torch.zeros((8, 4))}, batch_size=[8])
        models = {}
        for owner, width in (("actor", 2), ("critic", 1)):
            options = copy.deepcopy(cfg[owner])
            options.pop("class_name")
            model = MLPModel(obs, cfg["obs_groups"], owner, width, **options)
            models[owner] = {
                key: value.detach().cpu().numpy().copy()
                for key, value in model.state_dict().items()
            }
    changed = copy.deepcopy(models)
    last_bias = [name for name in changed["actor"] if name.endswith(".bias")][-1]
    changed["actor"][last_bias] += np.float32(0.125)
    return {
        "task": "fixture",
        "seed": 17401,
        "protocol": proto,
        "ppo_config": cfg,
        "models": {"initial": models, "final": changed},
        "hashes": {"initial": policy_hash(models), "final": policy_hash(changed)},
        "dependencies": {name: version(name) for name in response.DEPENDENCIES},
        "source_files": source_bindings(),
        "entry": entry(proto),
        "reference_actions": {
            "initial": np.zeros((8, 2), np.float32),
            "final": np.zeros((8, 2), np.float32),
        },
        "world_keys": tuple(5 + 36 * i for i in range(8)),
        "summary_sha256": "synthetic-actual-model",
        "manifest_sha256": "synthetic-actual-model",
        "evidence": {"scope": "test-only actual upstream models"},
    }


def random_state():
    import torch

    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
        [value.clone() for value in torch.cuda.get_rng_state_all()],
    )


def assert_random_unchanged(before):
    import torch

    after = random_state()
    assert before[0] == after[0]
    assert before[1][0] == after[1][0]
    assert np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])
    assert len(before[3]) == len(after[3])
    assert all(torch.equal(a, b) for a, b in zip(before[3], after[3], strict=True))


def computation_flags():
    import torch

    return {
        "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "cudnn_conv_fp32_precision": torch.backends.cudnn.conv.fp32_precision,
        "cudnn_rnn_fp32_precision": torch.backends.cudnn.rnn.fp32_precision,
    }


@contextmanager
def caller_computation(precision, cudnn):
    import torch

    previous = computation_flags()
    previous_precision = torch.get_float32_matmul_precision()
    previous_cudnn = torch.backends.cudnn.allow_tf32
    try:
        # Deliberately exercise callers using high/medium. Legacy reads/writes
        # occur only outside the diagnostic's new-API computation context.
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = cudnn
        yield {
            "effective": computation_flags(),
            "legacy_precision": precision,
            "legacy_cudnn": cudnn,
        }
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cudnn.allow_tf32 = previous_cudnn
        torch.backends.cuda.matmul.fp32_precision = previous["cuda_matmul_fp32_precision"]
        torch.backends.cudnn.conv.fp32_precision = previous["cudnn_conv_fp32_precision"]
        torch.backends.cudnn.rnn.fp32_precision = previous["cudnn_rnn_fp32_precision"]


def assert_caller_computation(caller):
    import torch

    assert computation_flags() == caller["effective"]
    assert torch.get_float32_matmul_precision() == caller["legacy_precision"]
    assert torch.backends.cudnn.allow_tf32 == caller["legacy_cudnn"]


def assert_original_computation():
    assert computation_flags() == {
        "cuda_matmul_fp32_precision": "ieee",
        "cudnn_conv_fp32_precision": "ieee",
        "cudnn_rnn_fp32_precision": "ieee",
    }


@pytest.mark.cuda
def test_actual_cuda_feature_and_pinned_models_restore_rng(monkeypatch):
    pair = cuda_pair()
    before = random_state()
    original = response._infer_model
    observed = []

    def infer(*args, **kwargs):
        assert_original_computation()
        observed.append(computation_flags())
        return original(*args, **kwargs)

    with caller_computation("medium", True) as caller:
        monkeypatch.setattr(response, "_infer_model", infer)
        result = response.evaluate_pair(pair)
        assert_caller_computation(caller)
        assert len(observed) == 2
        assert result["computation_flags"] == observed[0] == observed[1]
    assert_random_unchanged(before)
    assert result["status"] == "PASS_FIXED_POLICY_RESPONSE"
    assert result["features"]["kernel_calls"] == 1
    assert result["features"]["physics_steps"] == 0
    assert result["features"]["observation"].shape == (17, 4)
    assert result["features"]["observation"].dtype == np.float32
    assert np.isfinite(result["features"]["observation"]).all()
    # Identity target in this synthetic task has exactly zero feature input.
    assert not result["features"]["observation"][8].any()
    for stage in ("initial", "final"):
        values = result["checkpoints"][stage]
        assert values["loaded_model_hash"] == pair["hashes"][stage]
        assert values["raw_action"].shape == values["mapped_ctrl"].shape == (17, 2)
        assert np.isfinite(values["raw_action"]).all()
        assert np.isfinite(values["mapped_ctrl"]).all()
    assert not np.array_equal(
        result["checkpoints"]["initial"]["raw_action"], result["checkpoints"]["final"]["raw_action"]
    )


@pytest.mark.cuda
@pytest.mark.parametrize("defect", ["missing", "extra", "layout"])
def test_actual_model_state_loading_is_strict_and_rng_safe(defect):
    pair = cuda_pair()
    model = pair["models"]["initial"]["actor"]
    name = next(iter(model))
    if defect == "missing":
        del model[name]
    elif defect == "extra":
        model["obs_normalizer.unexpected"] = np.zeros(1, np.float32)
    else:
        model[name] = np.zeros((19, 23), np.float32)
    pair["hashes"]["initial"] = policy_hash(pair["models"]["initial"])
    before = random_state()
    with pytest.raises(ValueError):
        response.evaluate_pair(pair)
    assert_random_unchanged(before)


@pytest.mark.cuda
@pytest.mark.parametrize("stage", ["feature", "inference"])
def test_actual_failure_restores_rng_closes_runtime_and_allows_retry(monkeypatch, stage):
    pair = cuda_pair()
    import torch
    import tyx as tx

    from phymeo.backends.mujoco_tyx import _task_kernels

    closes = []
    original_close = tx.Runtime.close

    def close(owner):
        closes.append(id(owner))
        return original_close(owner)

    def fail(*args, **kwargs):
        assert_original_computation()
        random.random()
        np.random.random()
        torch.rand(1)
        torch.rand(1, device="cuda:0")
        raise RuntimeError("injected response stage failure")

    before = random_state()
    precision, cudnn = ("high", True) if stage == "feature" else ("medium", False)
    with caller_computation(precision, cudnn) as caller:
        with monkeypatch.context() as patch:
            patch.setattr(tx.Runtime, "close", close)
            if stage == "feature":
                patch.setattr(_task_kernels, "evaluate", fail)
            else:
                patch.setattr(response, "_infer_model", fail)
            with pytest.raises(RuntimeError, match="injected"):
                response.evaluate_pair(pair)
        assert_random_unchanged(before)
        assert_caller_computation(caller)
        assert closes
        assert response.evaluate_pair(pair)["status"] == "PASS_FIXED_POLICY_RESPONSE"
        assert_caller_computation(caller)
        assert_random_unchanged(before)
