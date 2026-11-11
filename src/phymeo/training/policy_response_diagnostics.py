"""Fixed, hash-bound saved MLP response experiment; never advances physics.

Optional CUDA dependencies are imported only at evaluation time. The bounded
N4b stream decoder is reused, but none of its behavioral metrics are rerun.
"""

import copy
import hashlib
import inspect
import math
import random
from contextlib import ExitStack, contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .learning_diagnostics import (
    _array,
    _bounded_json,
    _policy_hash,
    _type,
    read_records,
    require,
)

DEPENDENCIES = ("mujoco", "numpy", "rsl-rl-lib", "tensordict", "torch", "tyx-compute")
FEATURE_SUFFIX = "/src/phymeo/backends/mujoco_tyx/_task_kernels.py"
MLP_CLASS = "rsl_rl.models.MLPModel"
DIST_CLASS = "rsl_rl.modules.distribution.GaussianDistribution"


def validate_model_config(config):
    """Only the actually saved N4a architecture; never resolve arbitrary names."""
    network = {
        "class_name": MLP_CLASS,
        "hidden_dims": [32, 32],
        "activation": "tanh",
        "obs_normalization": False,
    }
    distribution = {"class_name": DIST_CLASS, "init_std": 0.1, "std_type": "log", "learn_std": True}
    require(type(config) is dict, "saved model config")
    require(
        config.get("actor") == {**network, "distribution_cfg": distribution}
        and config.get("critic") == network
        and config.get("obs_groups") == {"actor": ["policy"], "critic": ["policy"]}
        and config.get("num_steps_per_env") == 32,
        "only original N4a MLP/Identity normalizer/config is admitted",
    )


def check_dependencies(saved):
    require(type(saved) is dict and set(saved) == set(DEPENDENCIES), "complete dependency binding")
    current = {name: version(name) for name in DEPENDENCIES}
    require(current == saved, "installed dependencies differ from saved policy environment")
    require(current["rsl-rl-lib"] == "5.5.2", "pinned RSL5.5.2 required")
    return current


def _feature_source(source_files):
    path = Path(__file__).resolve().parents[1] / "backends" / "mujoco_tyx" / "_task_kernels.py"
    matches = [
        (name, digest)
        for name, digest in source_files.items()
        if name.replace("\\", "/").endswith(FEATURE_SUFFIX)
    ]
    require(len(matches) == 1, "one originally bound feature source required")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(digest == matches[0][1], "feature source differs from original evidence")
    return {"path": str(path), "sha256": digest, "original_path": matches[0][0]}


def _protocol_arrays(protocol):
    target = np.asarray(protocol["q_target"], np.float64)
    require(target.ndim == 1 and 1 <= target.size <= 35, "bounded task width")
    width = target.size
    qscale = np.asarray(protocol["q_scale"], np.float64)
    vscale = np.asarray(protocol["v_scale"], np.float64)
    lower = np.asarray(protocol["q_lower"], np.float64)
    upper = np.asarray(protocol["q_upper"], np.float64)
    require(
        all(
            a.shape == (width,) and np.isfinite(a).all()
            for a in (target, qscale, vscale, lower, upper)
        ),
        "task parameter shape/finite",
    )
    require(
        np.all(qscale > 0)
        and np.all(vscale > 0)
        and np.all(lower <= target)
        and np.all(target <= upper),
        "task scale/target bounds",
    )
    require(
        math.isfinite(float(protocol["v_bound"]))
        and protocol["v_bound"] > 0
        and math.isfinite(float(protocol["action_scale"]))
        and protocol["action_scale"] > 0
        and math.isfinite(float(protocol["h"]))
        and protocol["h"] > 0
        and protocol["horizon"] == 256
        and protocol["decimation"] == 1
        and protocol["action_squash"] == "tanh",
        "original task protocol",
    )
    return target, qscale, vscale, lower, upper


def fixed_grid(protocol, entry):
    """Exactly eight original starts, target, and q/v coordinate pairs."""
    target, qs, vs, lower, upper = _protocol_arrays(protocol)
    width = len(target)
    templates = np.asarray(protocol["held_out_templates"], np.float64)
    require(
        templates.shape == (8, 2, width) and np.isfinite(templates).all(),
        "eight held-out templates",
    )
    expected = templates.astype(np.float32)
    require(np.isfinite(expected).all(), "stored held-out template range")
    q_entry, v_entry = _array(entry["q"], (8, width)), _array(entry["v"], (8, width))
    require(q_entry.dtype == v_entry.dtype == np.float32, "actual stored entry dtype")
    require(
        q_entry.tobytes() == expected[:, 0].tobytes()
        and v_entry.tobytes() == expected[:, 1].tobytes(),
        "held-out template/stored entry bits",
    )
    qs_list, vs_list = list(q_entry.astype(np.float64)), list(v_entry.astype(np.float64))
    labels = ["held-out-" + str(i) for i in range(8)]
    qs_list.append(target.copy())
    vs_list.append(np.zeros(width))
    labels.append("target")
    pairs = {"q": [], "v": []}
    for family, scales in (("q", qs), ("v", vs)):
        for axis in range(width):
            indices = []
            for sign, name in ((1, "plus"), (-1, "minus")):
                q, v = target.copy(), np.zeros(width)
                if family == "q":
                    q[axis] += sign * scales[axis] / 4
                else:
                    v[axis] = sign * scales[axis] / 4
                indices.append(len(labels))
                qs_list.append(q)
                vs_list.append(v)
                labels.append(f"{family}{axis}-{name}")
            pairs[family].append(tuple(indices))
    q, v = np.asarray(qs_list, np.float32), np.asarray(vs_list, np.float32)
    require(np.isfinite(q).all() and np.isfinite(v).all(), "finite stored grid")
    require(
        np.all(q >= lower) and np.all(q <= upper) and np.all(np.abs(v) <= protocol["v_bound"]),
        "fixed native grid outside original bounds",
    )
    for family, values in (("q", q), ("v", v)):
        require(
            all(values[a, j] > values[b, j] for j, (a, b) in enumerate(pairs[family])),
            "symmetric pair collapses in stored FP32",
        )
    return {
        "q": q,
        "v": v,
        "labels": tuple(labels),
        "target_index": 8,
        "q_pairs": tuple(pairs["q"]),
        "v_pairs": tuple(pairs["v"]),
    }


def _first_entry(transition, request, protocol, keys):
    value = _type(transition, "phymeo.training.mujoco_tyx_env.MuJoCoTyxTransition")
    width = len(protocol["q_target"])
    require(
        tuple(value["world_keys"]) == keys and set(value["states"]) == set(keys),
        "first-input world order",
    )
    action = _array(request["action"], (8, width))
    require(
        request["tick"] == 1
        and action.dtype == np.float32
        and np.array_equal(action, _array(value["action"], (8, width))),
        "first held-out request binding",
    )
    rows = {"q": [], "v": []}
    for key in keys:
        state = _type(value["states"][key], "phymeo.backends.mujoco_runtime.MuJoCoState")
        receipt = _type(
            state["last_step"], "phymeo.backends._mujoco_tyx_runtime.MuJoCoTyxStepReceipt"
        )
        require(
            state["world"] == receipt["world"] == key
            and state["tick"] == receipt["tick"] == 1
            and receipt["input_time"] == 0.0
            and receipt["h"] == protocol["h"],
            "first receipt identity/clock",
        )
        for kind in ("q", "v"):
            original = _array(receipt["input_" + kind], (width,))
            stored = original.astype(np.float32)
            require(
                np.isfinite(stored).all()
                and np.array_equal(stored.astype(original.dtype), original),
                "first native input is not exact stored FP32",
            )
            rows[kind].append(stored)
    return {kind: np.asarray(values, np.float32) for kind, values in rows.items()}, action.copy()


def read_policy_pair(directory, binding, *, expected_task, expected_seed):
    """Read one bound stream once, retaining only two models and first inputs."""
    directory = Path(directory).resolve()
    summary, digest = _bounded_json(directory / "summary.json", 16 * 1024**2)
    publication, _ = _bounded_json(directory / "publication.json", 16384)
    manifest, manifest_digest = _bounded_json(directory / "manifest.json", 4 * 1024**2)
    require(
        Path(binding["directory"]).resolve() == directory
        and digest == binding["summary_sha256"] == publication.get("summary_sha256"),
        "summary directory/publication binding",
    )
    require(
        summary.get("task") == expected_task
        and summary.get("seed") == expected_seed
        and summary.get("status") == "COMPLETE_N4A_SEED",
        "task/seed/status identity",
    )
    for name in ("task", "seed", "protocol_binding", "ppo_config", "dependencies", "source_files"):
        require(summary.get(name) == manifest.get(name), "manifest/summary mismatch: " + name)
    require(
        manifest.get("phase") == "learn"
        and summary.get("updates") == 128
        and summary.get("worlds") == 8
        and summary.get("accepted_world_steps") == 32768,
        "original complete training budget",
    )
    keys = tuple(manifest["world_keys"])
    require(
        len(keys) == len(set(keys)) == 8 and all(type(k) is int for k in keys),
        "original eight world IDs",
    )
    protocol = summary["protocol_binding"]
    _protocol_arrays(protocol)
    validate_model_config(summary["ppo_config"])
    _feature_source(summary["source_files"])
    evidence = summary["evidence"]
    require(
        evidence == binding["evidence"] and Path(evidence["path"]).name == evidence["path"],
        "complete stream binding/path",
    )
    models, requests, entries, references = {}, {}, {}, {}
    final_index = completion_index = None
    records = read_records(directory / evidence["path"], evidence)
    try:
        for ordinal, (kind, value) in enumerate(records):
            if ordinal == 0:
                require(kind == "initial_policy", "initial model must be first")
            if kind == "initial_policy":
                require(
                    "initial" not in models and set(value) == {"actor", "critic"},
                    "unique complete initial model",
                )
                models["initial"] = value
            elif kind == "final_policy_and_adam":
                require(
                    "final" not in models and value["next_iteration"] == 128,
                    "unique complete final model/update",
                )
                models["final"] = {name: value[name] for name in ("actor", "critic")}
                final_index = ordinal
            elif kind == "completion":
                require(
                    completion_index is None
                    and value == {"status": "COMPLETE_N4A_SEED", "task": expected_task},
                    "completion identity",
                )
                completion_index = ordinal
            else:
                for phase, stage in (("untrained", "initial"), ("learned", "final")):
                    if kind == phase + ".requested_action" and value.get("tick") == 1:
                        require(stage not in requests, "duplicate first request")
                        requests[stage] = value
                    elif (
                        kind == phase + ".transition" and stage in requests and stage not in entries
                    ):
                        entries[stage], references[stage] = _first_entry(
                            value, requests[stage], protocol, keys
                        )
        require(
            final_index == evidence["records"] - 2 and completion_index == evidence["records"] - 1,
            "final model/completion suffix",
        )
    finally:
        records.close()
    require(
        set(models) == set(entries) == {"initial", "final"},
        "both models and held-out starts required",
    )
    hashes = {stage: _policy_hash(model) for stage, model in models.items()}
    require(
        hashes["initial"] == summary["parameter_hash_before"]
        and hashes["final"] == summary["parameter_hash_after"],
        "whole actor/critic model hash",
    )
    for kind in ("q", "v"):
        require(
            entries["initial"][kind].tobytes() == entries["final"][kind].tobytes(),
            "held-out initial/final native input identity",
        )
    fixed_grid(protocol, entries["initial"])
    return {
        "task": expected_task,
        "seed": expected_seed,
        "protocol": protocol,
        "models": models,
        "hashes": hashes,
        "entry": entries["initial"],
        "reference_actions": references,
        "world_keys": keys,
        "ppo_config": summary["ppo_config"],
        "dependencies": summary["dependencies"],
        "source_files": summary["source_files"],
        "summary_sha256": digest,
        "manifest_sha256": manifest_digest,
        "evidence": evidence,
    }


def _computation_flags(torch):
    return {
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


@contextmanager
def _preserved_execution_state(torch):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state, cuda_states = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    flags = _computation_flags(torch)
    try:
        # Exactly the two computation assignments in original tasks.seed_all.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        actual = _computation_flags(torch)
        require(
            not actual["matmul_allow_tf32"] and not actual["cudnn_allow_tf32"],
            "original task computation flags",
        )
        yield actual
    finally:
        # Try every restoration even if one API unexpectedly fails. Restore the
        # precision enum last: a bool alone cannot preserve caller medium/high.
        errors = []
        restorations = (
            (random.setstate, (python_state,)),
            (np.random.set_state, (numpy_state,)),
            (torch.set_rng_state, (cpu_state,)),
            (torch.cuda.set_rng_state_all, (cuda_states,)),
            (setattr, (torch.backends.cuda.matmul, "allow_tf32", flags["matmul_allow_tf32"])),
            (setattr, (torch.backends.cudnn, "allow_tf32", flags["cudnn_allow_tf32"])),
            (torch.set_float32_matmul_precision, (flags["float32_matmul_precision"],)),
        )
        for restore, arguments in restorations:
            try:
                restore(*arguments)
            except BaseException as error:
                errors.append(error)
        try:
            require(_computation_flags(torch) == flags, "caller computation flag restoration")
        except BaseException as error:
            errors.append(error)
        if errors:
            raise BaseExceptionGroup("caller RNG/computation restoration failed", errors)


def _load_network(model, saved, torch, device):
    expected = model.state_dict()
    require(
        type(saved) is dict and set(saved) == set(expected),
        "strict complete model/normalizer tensor keys",
    )
    values = {}
    for name, reference in expected.items():
        value = _array(saved[name], tuple(reference.shape))
        require(value.dtype == reference.detach().cpu().numpy().dtype, "strict model tensor dtype")
        values[name] = torch.from_numpy(value.copy()).to(device=device)
    model.load_state_dict(values, strict=True)
    model.eval()
    return {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}


def _models(pair, torch, device):
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    width = len(pair["protocol"]["q_target"])
    config = pair["ppo_config"]
    example = TensorDict(
        {"policy": torch.zeros((8, 2 * width), dtype=torch.float32, device=device)},
        batch_size=[8],
        device=device,
    )
    actors, bindings = {}, {}
    for stage in ("initial", "final"):
        saved = pair["models"][stage]
        require(_policy_hash(saved) == pair["hashes"][stage], "in-memory model hash changed")
        loaded, networks = {}, {}
        for owner, output_width in (("actor", width), ("critic", 1)):
            options = copy.deepcopy(config[owner])
            options.pop("class_name")
            # The only class path consumed by upstream is the already-whitelisted Gaussian.
            network = MLPModel(example, config["obs_groups"], owner, output_width, **options).to(
                device
            )
            loaded[owner] = _load_network(network, saved[owner], torch, device)
            networks[owner] = network
        require(_policy_hash(loaded) == pair["hashes"][stage], "loaded actor/critic bytes differ")
        actors[stage] = networks["actor"]
        bindings[stage] = {
            owner: {
                name: {
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                }
                for name, value in state.items()
            }
            for owner, state in loaded.items()
        }
    return actors, bindings


def _features_cuda(protocol, grid, device_index):
    import tyx as tx

    from phymeo.backends.mujoco_task_workspace import MuJoCoTyxTaskSpec
    from phymeo.backends.mujoco_tyx import _task_kernels

    target, qs, vs, lower, upper = _protocol_arrays(protocol)
    width, count = len(target), len(grid["labels"])
    spec = MuJoCoTyxTaskSpec(
        tuple(range(width)),
        tuple(range(width)),
        scale=tuple(1 / x for x in qs) + tuple(1 / x for x in vs),
        bias=tuple(-q / x for q, x in zip(target, qs, strict=True)) + (0.0,) * width,
        target=tuple(target) + (0.0,) * width,
        weight=tuple(1 / (width * x * x) for x in qs) + tuple(0.01 / (width * x * x) for x in vs),
        lower=tuple(lower) + (-protocol["v_bound"],) * width,
        upper=tuple(upper) + (protocol["v_bound"],) * width,
    )
    ids = np.asarray([(0, i) for i in range(width)] + [(1, i) for i in range(width)], np.int32)
    parameters = np.column_stack(
        tuple(
            getattr(spec, name) for name in ("scale", "bias", "target", "weight", "lower", "upper")
        )
    )
    with ExitStack() as owners:
        runtime = tx.Runtime(backend="cuda", device=device_index)
        owners.callback(runtime.close)
        stream = tx.CudaStream(device=device_index)
        owners.callback(stream.close)
        with tx.use_runtime(runtime):

            def owned(value):
                array = tx.CudaArray(value, device=device_index)
                owners.callback(array.free)
                return array

            source_ids, source_parameters = owned(ids), owned(parameters)
            q, v = owned(grid["q"]), owned(grid["v"])
            observation = owned(np.zeros((count, 2 * width), np.float32))
            reward = owned(np.zeros(count, np.float32))
            terminated = owned(np.zeros(count, np.int32))
            status = owned(np.zeros(count, np.int32))
            try:
                _task_kernels.evaluate(
                    count,
                    2 * width,
                    source_ids,
                    source_parameters,
                    q,
                    v,
                    observation,
                    reward,
                    terminated,
                    status,
                    mode="cuda",
                    stream=stream,
                )
                stream.synchronize()
                outputs = [
                    array.to_numpy(stream=stream)
                    for array in (observation, reward, terminated, status)
                ]
                stream.synchronize()
            finally:
                # Complete any submitted feature work before buffers/stream/runtime retire.
                stream.synchronize()
    observed, rewards, done, flags = outputs
    require(
        observed.shape == (count, 2 * width)
        and observed.dtype == np.float32
        and np.isfinite(observed).all()
        and np.isfinite(rewards).all()
        and not done.any()
        and not flags.any(),
        "device feature status/output contract",
    )
    return {
        "observation": observed,
        "reward": rewards,
        "terminated": done,
        "status": flags,
        "ids": ids,
        "parameters": parameters,
        "spec_sha256": spec.sha256,
        "kernel_calls": 1,
        "physics_steps": 0,
        "input_origin": "saved native FP32 entry plus fixed counterfactual grid",
        "precision": "actual original TYX feature kernel, not saved observation bytes",
        "transport": "explicit diagnostic HOST feature readback then Torch upload",
    }


def _infer_model(actor, observation):
    return actor(observation, stochastic_output=False)


def response_components(raw_action, mapped_ctrl, grid):
    """Descriptive paired response; no feedback/learning acceptance threshold."""
    width = grid["q"].shape[1]
    count = len(grid["labels"])
    result = {}
    for name, supplied in (("raw", raw_action), ("ctrl", mapped_ctrl)):
        values = _array(supplied, (count, width)).astype(np.float64)
        center = values[grid["target_index"]]
        item = {"target": center.copy()}
        for family in ("q", "v"):
            pairs = grid[family + "_pairs"]
            require(len(pairs) == width, "complete response coordinate pairs")
            plus = np.asarray([values[a] for a, _ in pairs])
            minus = np.asarray([values[b] for _, b in pairs])
            denominator = np.asarray(
                [
                    float(grid[family][a, j]) - float(grid[family][b, j])
                    for j, (a, b) in enumerate(pairs)
                ]
            )
            require(
                np.isfinite(denominator).all() and np.all(denominator > 0),
                "noncollapsed response pair",
            )
            even, odd = (plus + minus) / 2, (plus - minus) / 2
            item[family] = {
                "plus": plus,
                "minus": minus,
                "even": even,
                "odd": odd,
                "even_minus_target": even - center,
                "native_pair_separation": denominator,
                "secant_native": (plus - minus) / denominator[:, None],
            }
        result[name] = item
    result["scope"] = "finite fixed-input secants/correlations, not derivative or stability proof"
    return result


def evaluate_pair(pair, *, device="cuda:0"):
    """Actual fixed-model CUDA inference with no environment or physical runtime."""
    import torch
    from tensordict import TensorDict

    validate_model_config(pair["ppo_config"])
    dependencies = check_dependencies(pair["dependencies"])
    feature_source = _feature_source(pair["source_files"])
    selected = torch.device(device)
    require(
        selected.type == "cuda" and torch.cuda.is_available(), "real CUDA training profile required"
    )
    index = torch.cuda.current_device() if selected.index is None else selected.index
    require(0 <= index < torch.cuda.device_count(), "CUDA device ordinal")
    selected = torch.device("cuda", index)
    grid = fixed_grid(pair["protocol"], pair["entry"])
    with _preserved_execution_state(torch) as computation_flags:
        actors, state_bindings = _models(pair, torch, selected)
        implementations = {}
        sample = actors["initial"]
        for component in (sample, sample.mlp, sample.distribution, sample.obs_normalizer):
            cls = type(component)
            source = Path(inspect.getfile(cls)).resolve()
            implementations[cls.__module__ + "." + cls.__qualname__] = {
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        features = _features_cuda(pair["protocol"], grid, index)
        obs = TensorDict(
            {"policy": torch.from_numpy(features["observation"].copy()).to(selected)},
            batch_size=[len(grid["labels"])],
            device=selected,
        )
        checkpoints = {}
        with torch.inference_mode():
            for stage in ("initial", "final"):
                action = _infer_model(actors[stage], obs)
                require(
                    tuple(action.shape) == (len(grid["labels"]), len(pair["protocol"]["q_target"]))
                    and action.dtype == torch.float32
                    and action.device == selected
                    and bool(torch.isfinite(action).all()),
                    "actual deterministic action layout/finite",
                )
                scale = torch.as_tensor(
                    pair["protocol"]["action_scale"], dtype=torch.float32, device=selected
                )
                mapped = torch.zeros_like(action) + scale * torch.tanh(action)
                torch.cuda.current_stream(selected).synchronize()
                raw, control = (
                    action.detach().cpu().numpy().copy(),
                    mapped.detach().cpu().numpy().copy(),
                )
                require(np.isfinite(control).all(), "mapped ctrl nonfinite")
                reference = _array(pair["reference_actions"][stage], (8, raw.shape[1]))
                checkpoints[stage] = {
                    "loaded_model_hash": pair["hashes"][stage],
                    "loaded_state_bindings": state_bindings[stage],
                    "raw_action": raw,
                    "mapped_ctrl": control,
                    "response": response_components(raw, control, grid),
                    "original_held_out_requested_action": reference.copy(),
                    "held_out_action_difference": raw[:8].astype(np.float64) - reference,
                    "reference_scope": "original deterministic held-out request only; no equality gate; new batch GEMM may round differently",
                }
        return {
            "status": "PASS_FIXED_POLICY_RESPONSE",
            "task": pair["task"],
            "seed": pair["seed"],
            "device": str(selected),
            "dependencies": dependencies,
            "implementation_sources": implementations,
            "computation_flags": computation_flags,
            "source_binding": {
                "summary_sha256": pair["summary_sha256"],
                "manifest_sha256": pair["manifest_sha256"],
                "evidence": pair["evidence"],
                "feature_source": feature_source,
            },
            "protocol_binding": pair["protocol"],
            "ppo_config": pair["ppo_config"],
            "world_keys": pair["world_keys"],
            "grid": grid,
            "features": features,
            "checkpoints": checkpoints,
            "rng": "Python/NumPy/Torch CPU/all CUDA restored in finally",
            "learning_quality": "UNCHANGED_NOT_TESTED_BY_THIS_EXPERIMENT",
            "normalizer": "original Identity; all model state keys strictly checked",
        }
