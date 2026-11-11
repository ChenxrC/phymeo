"""N4d one-shot initialization cases; no full learning experiment in this suite."""

import copy
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
def test_injected_failure_restores_all_original_model_state(monkeypatch, stage):
    import torch
    actor, critic, optimizer, obs = models()
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
