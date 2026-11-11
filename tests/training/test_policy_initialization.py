"""N4d one-shot initialization cases; no full learning experiment in this suite."""

import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from phymeo.training import mujoco_tyx_evaluation as evaluation
from phymeo.training import policy_initialization as initialization


def models(device="cpu"):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("N4d CUDA qualification requires a real device")
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(14011)
        obs = TensorDict({"policy": torch.zeros((1, 4), dtype=torch.float32, device=device)},
                         batch_size=[1], device=device)
        groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = MLPModel(obs, groups, "actor", 2, hidden_dims=[32, 32], activation="tanh",
                         obs_normalization=False, distribution_cfg={
                             "class_name": "rsl_rl.modules.GaussianDistribution",
                             "init_std": .1, "std_type": "log", "learn_std": True}).to(device)
        critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[32, 32], activation="tanh",
                          obs_normalization=False).to(device)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=.0003)
    return actor, critic, optimizer, obs


def rng():
    import torch
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def same_rng(before):
    import torch
    after = rng()
    assert before[0] == after[0]
    assert before[1][0] == after[1][0]
    assert np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])
    assert len(before[3]) == len(after[3])
    assert all(torch.equal(a, b) for a, b in zip(before[3], after[3], strict=True))


def policy_hash(actor, critic):
    return evaluation.parameter_hash(SimpleNamespace(actor=actor, critic=critic))


def invoke(actor, critic, optimizer, obs, expected=None):
    return initialization.center_initial_mean(
        actor, obs, critic=critic, optimizer=optimizer,
        expected_policy_hash=policy_hash(actor, critic) if expected is None else expected,
    )


def assert_only_bias(actor, critic, before, key):
    for owner, model in (("actor", actor), ("critic", critic)):
        for name, value in model.state_dict().items():
            current = value.detach().cpu().numpy()
            if (owner, name) != ("actor", key):
                assert np.array_equal(current, before[owner][name])


def test_real_cpu_model_only_bias_changes_and_remains_trainable():
    import torch
    actor, critic, optimizer, obs = models()
    actor.train()
    modes = {name: item.training for name, item in actor.named_modules()}
    before_rng = rng()
    result = invoke(actor, critic, optimizer, obs)
    same_rng(before_rng)
    assert result["status"] == "APPLIED_ONE_SHOT_MEAN_CENTERING"
    assert result["optimizer_before"] == result["optimizer_after"]
    assert not optimizer.state
    assert result["model_hash_after"] == policy_hash(actor, critic)
    assert_only_bias(actor, critic, result["model_before"], result["bias_name"])
    bias = dict(actor.named_parameters())[result["bias_name"]]
    expected = result["model_before"]["actor"][result["bias_name"]] - result["mean_before"][0]
    assert np.array_equal(bias.detach().numpy(), expected)
    assert np.array_equal(actor(obs).detach().numpy(), result["mean_after"])
    assert modes == {name: item.training for name, item in actor.named_modules()}
    assert all(parameter.grad is None for parameter in list(actor.parameters()) + list(critic.parameters()))
    old_bias = bias.detach().clone()
    (actor(obs).sum() + critic(obs).sum()).backward()
    optimizer.step()
    assert bias.requires_grad and not torch.equal(old_bias, bias)
    assert np.array_equal(result["model_after"]["actor"][result["bias_name"]], old_bias.numpy())
    assert np.array_equal(result["proposed_bias"], old_bias.numpy())


@pytest.mark.parametrize("fault", ("hash", "dtype", "nonfinite", "batch", "normalizer",
                                  "distribution", "adam", "owner", "gradient", "parameter"))
def test_validation_refuses_before_any_model_publication(fault):
    import torch
    from tensordict import TensorDict
    actor, critic, optimizer, obs = models()
    expected = policy_hash(actor, critic)
    if fault == "hash":
        expected = "0" * 64
    elif fault == "dtype":
        obs = obs.to(dtype=torch.float64)
    elif fault == "nonfinite":
        obs["policy"][0, 0] = float("nan")
    elif fault == "batch":
        obs = TensorDict({"policy": torch.zeros(2, 4)}, batch_size=[2])
    elif fault == "normalizer":
        actor.obs_normalization = True
    elif fault == "distribution":
        actor.distribution = torch.nn.Identity()
    elif fault == "adam":
        next(actor.parameters()).grad = torch.ones_like(next(actor.parameters()))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    elif fault == "owner":
        optimizer = torch.optim.Adam(critic.parameters(), lr=.0003)
    elif fault == "gradient":
        next(actor.parameters()).grad = torch.zeros_like(next(actor.parameters()))
    else:
        with torch.no_grad():
            next(actor.parameters()).flatten()[0] = float("inf")
    before = {owner: {name: value.detach().clone() for name, value in model.state_dict().items()}
              for owner, model in (("actor", actor), ("critic", critic))}
    before_rng = rng()
    with pytest.raises((ValueError, TypeError)):
        invoke(actor, critic, optimizer, obs, expected)
    same_rng(before_rng)
    for owner, model in (("actor", actor), ("critic", critic)):
        assert all(torch.equal(value, before[owner][name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize("stage", ("candidate", "copy", "verify"))
@pytest.mark.parametrize("device", ("cpu", pytest.param("cuda", marks=pytest.mark.cuda)))
def test_injected_failure_restores_all_original_model_state(monkeypatch, stage, device):
    import torch
    actor, critic, optimizer, obs = models(device)
    before_hash, before_rng = policy_hash(actor, critic), rng()
    if stage == "candidate":
        original = initialization._mean
        calls = []
        def fail(*args):
            calls.append(None)
            if len(calls) == 2:
                random.random()
                np.random.random()
                torch.rand(1)
                if device == "cuda":
                    torch.rand(1, device=device)
                raise RuntimeError("injected candidate")
            return original(*args)
        monkeypatch.setattr(initialization, "_mean", fail)
    elif stage == "copy":
        original = initialization._publish_bias
        def fail(*args):
            original(*args)
            raise RuntimeError("injected partial publication")
        monkeypatch.setattr(initialization, "_publish_bias", fail)
    else:
        def fail(*args):
            raise RuntimeError("injected post-copy verification")
        monkeypatch.setattr(initialization, "_verify_published", fail)
    with pytest.raises(initialization.InitializationFailure) as caught:
        invoke(actor, critic, optimizer, obs)
    assert caught.value.rolled_back == (stage != "candidate")
    assert policy_hash(actor, critic) == before_hash
    assert not optimizer.state
    same_rng(before_rng)


def example():
    path = Path(__file__).resolve().parents[2] / "examples/evaluate_mujoco_tyx_initialization_ablation.py"
    spec = importlib.util.spec_from_file_location("initialization_ablation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failed_seed", (None, 14012))
def test_wrapper_uses_only_three_original_runs_and_gate(monkeypatch, tmp_path, failed_seed):
    module = example()
    calls, summaries = [], []
    originals = {seed: {"parameter_hash_before": "0" * 64, "baseline": {"cost": 2.0},
                       "learned": {"cost": 1.8}, "zero_action_diagnostic":
                       {"cost": 1.7} if seed == 14011 else None}
                 for seed in module.SEEDS}
    monkeypatch.setattr(module, "load_originals", lambda root: ({"hash": "bound"}, originals))
    monkeypatch.setattr(module, "make_initializer", lambda *args, **kwargs: object())
    def run(name, seed, output, *, initializer=None):
        assert name == "J" and initializer is not None
        calls.append(seed)
        if seed == failed_seed:
            raise RuntimeError("fixed synthetic seed failure")
        return {"baseline": {"cost": 1.0}, "learned": {"cost": .75}, "initializer": {"schema": "fixture"}}
    def summarize(name, inputs, output):
        summaries.append((name, list(inputs)))
        return {"outcome": {"classification": "PASS_FIXED_LEARNING_GATE"}}
    monkeypatch.setattr(module, "learning_api", lambda: SimpleNamespace(run_learning=run, summarize_task=summarize))
    result = module.run_ablation(tmp_path / "old", tmp_path / "new")
    assert calls == list(module.SEEDS)
    assert len(summaries) == int(failed_seed is None)
    assert len(result["rows"]) == 3
    assert result["original_classification"] == "FIXED_BUDGET_NOT_LEARNED"
    if failed_seed is not None:
        assert result["classification"] == "INCOMPLETE_ABLATION_RETAINED"
    else:
        assert result["classification"] == "PASS_FIXED_LEARNING_GATE"


@pytest.mark.cuda
def test_real_cuda_centering_and_target_features():
    import torch
    actor, critic, optimizer, obs = models("cuda")
    module = example()
    api = module.learning_api()
    task = api.tasks().task("J")
    with api.tasks().make_environment("J") as env:
        revision = env.runtime.checkpoint_revision
        features = module.target_features(api.protocol_binding("J"), env)
        assert env.runtime.checkpoint_revision == revision
        assert features["physics_steps"] == 0 and features["kernel_calls"] == 1
        assert features["spec_sha256"] == env.task_spec.sha256
        assert np.array_equal(features["native_q"], np.asarray([task.q_target], np.float32))
        assert not features["native_v"].any()
        assert features["observation"].shape == (1, 4)
    obs["policy"] = torch.from_numpy(features["observation"].copy()).to("cuda")
    before_rng = rng()
    result = invoke(actor, critic, optimizer, obs)
    assert result["model_hash_after"] == policy_hash(actor, critic)
    assert_only_bias(actor, critic, result["model_before"], result["bias_name"])
    same_rng(before_rng)


@pytest.mark.parametrize("fault", (None, "summary-hash", "manifest", "seed", "truncated"))
def test_original_metadata_binding_is_complete_and_read_only(monkeypatch, tmp_path, fault):
    import hashlib
    import json
    module = example()
    root = tmp_path / "old"
    root.mkdir()
    anchor = {"task": "J", "seeds": list(module.SEEDS), "outcome": {"classification": "FIXED_BUDGET_NOT_LEARNED"},
              "protocol_binding": {"task": "fixed"}, "source_bindings": {}}
    for seed in module.SEEDS:
        directory = root / str(seed)
        directory.mkdir()
        value = {"task": "J", "seed": seed, "status": "COMPLETE_N4A_SEED", "updates": 128,
                 "accepted_world_steps": 32768, "worlds": 8,
                 "protocol_binding": anchor["protocol_binding"], "ppo_config": {}, "source_files": {},
                 "dependencies": {}, "baseline": {"cost": 2., "world_steps": 2048},
                 "learned": {"cost": 1.8, "world_steps": 2048}, "evidence": {"accepted_old_trace": True}}
        raw = json.dumps(value).encode()
        digest = hashlib.sha256(raw).hexdigest()
        (directory / "summary.json").write_bytes(raw)
        (directory / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
        (directory / "publication.json").write_text(json.dumps({"summary_sha256": digest}), encoding="utf-8")
        anchor["source_bindings"][str(seed)] = {
            "directory": str(directory), "summary_sha256": digest, "evidence": value["evidence"]}
    path = root / module.ANCHOR
    path.parent.mkdir(parents=True)
    if fault == "seed":
        anchor["seeds"] = [14011, 14011, 14013]
    raw = json.dumps(anchor).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(module, "ANCHOR_SHA256", hashlib.sha256(raw).hexdigest())
    if fault == "summary-hash":
        (root / "14012/summary.json").write_text("{}", encoding="utf-8")
    elif fault == "manifest":
        (root / "14012/manifest.json").write_text("{}", encoding="utf-8")
    elif fault == "truncated":
        path.write_bytes(raw[:-1])
    if fault is None:
        bindings, rows = module.load_originals(root)
        assert tuple(rows) == module.SEEDS and len(bindings["source_bindings"]) == 3
    else:
        with pytest.raises((ValueError, KeyError)):
            module.load_originals(root)


@pytest.mark.parametrize("mode", ("default", "center", "failure"))
def test_original_learning_hook_order_and_default_output(monkeypatch, tmp_path, mode):
    import gzip
    import hashlib
    import json
    from contextlib import contextmanager

    api = example().learning_api()
    held_calls, hooks, sessions = [], [], []
    env = SimpleNamespace(counters={})
    task = SimpleNamespace(seeds=(14011, 14012, 14013), cost=lambda: None)
    monkeypatch.setattr(api, "tasks", lambda: SimpleNamespace(
        task=lambda name: task, seed_all=lambda seed: None, UPDATES=128, ROLLOUT=32))
    @contextmanager
    def owner(context, name):
        yield env
    monkeypatch.setattr(api, "_observed_environment", owner)

    def new_session(env):
        algorithm = SimpleNamespace(bias=1., iteration=0)
        algorithm.actor = SimpleNamespace(state_dict=lambda: {"bias": np.asarray([algorithm.bias], np.float32)})
        algorithm.critic = SimpleNamespace(state_dict=lambda: {"weight": np.asarray([2.], np.float32)})
        algorithm.optimizer = SimpleNamespace(state_dict=lambda: {"state": {}})
        session = SimpleNamespace(algorithm=algorithm, config={"fixed": True}, next_iteration=0)
        sessions.append(session)
        return session
    monkeypatch.setattr(api, "_session", new_session)
    monkeypatch.setattr(api, "protocol_binding", lambda name: {"name": name})
    monkeypatch.setattr(api, "_manifest", lambda *args: {"source_files": {}, "dependencies": {}})
    def state_hash(algorithm):
        return hashlib.sha256(repr((algorithm.bias, algorithm.iteration)).encode()).hexdigest()
    monkeypatch.setattr(api.evaluation, "parameter_hash", state_hash)
    def held(actor, name, writer, label, **kwargs):
        held_calls.append(label)
        return {"cost": 2. if label == "untrained" else 1., "world_steps": 2048}
    monkeypatch.setattr(api, "_held_out", held)
    def measure(session):
        index = session.next_iteration
        session.next_iteration += 1
        session.algorithm.iteration += 1
        return {"transitions": tuple(range(32)), "iteration": index}, {}
    monkeypatch.setattr(api.evaluation, "measure_update", measure)
    monkeypatch.setattr(api.evaluation, "validate_update", lambda record, *args, **kwargs:
                        {"accepted_world_steps": 256, "optimizer_step": record["iteration"] + 1})
    def initialize(session, actual_env, name, seed):
        hooks.append((session.next_iteration, name, seed))
        assert actual_env is env and session.next_iteration == 0
        if mode == "failure":
            raise RuntimeError("synthetic initializer refusal")
        session.algorithm.bias = 0.
        return {"factor": "fixture"}
    destination = tmp_path / mode
    options = {} if mode == "default" else {"initializer": initialize}
    if mode == "failure":
        with pytest.raises(RuntimeError, match="initializer refusal"):
            api.run_learning("J", 14011, destination, **options)
        assert not held_calls and sessions[0].next_iteration == 0
        failure = json.loads((destination / "failure.json").read_text())
        assert failure["phase"] == "declared-policy-initializer"
        assert not (destination / "summary.json").exists()
    else:
        result = api.run_learning("J", 14011, destination, **options)
        assert held_calls == ["untrained", "zero", "learned"]
        assert sessions[0].next_iteration == 128
        assert ("initializer" in result) == (mode == "center")
        assert hooks == ([] if mode == "default" else [(0, "J", 14011)])
        with gzip.open(destination / "trace.jsonl.gz", "rt") as stream:
            records = [json.loads(line)["kind"] for line in stream]
        assert records[0] == "initial_policy"
        assert records.count("initializer_receipt") == int(mode == "center")
        if mode == "center":
            assert records[1] == "initializer_receipt"
