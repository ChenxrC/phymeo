"""HOST algebra/contract tests only; these are not evidence of device learning."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
_path = Path(__file__).resolve().parents[1] / "examples" / "panda_state_ppo.py"
_spec = importlib.util.spec_from_file_location("panda_state_ppo_tested", _path)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)


def test_reward_uses_successor_state_and_actual_motor_effort():
    q, v, goal = torch.zeros(2, 23), torch.zeros(2, 22), torch.zeros(2, 16)
    controls = torch.zeros(3, 2, 16)
    scale = torch.tensor([.2]*14+[.035]*2)
    velocity = torch.tensor([2.]*14+[.2]*2)
    limits = torch.ones(16)*10
    assert torch.equal(module.joint_reward(torch, q, v, goal, controls, scale, velocity, limits), torch.ones(2))
    q[0, 0], v[0, 1], controls[:, 0, 2] = .2, 2., 10.
    result = module.joint_reward(torch, q, v, goal, controls, scale, velocity, limits)
    assert result[0].item() == pytest.approx(1.-1/16-.002/16-.001/16)
    assert result[1].item() == 1.


def test_gae_bootstraps_actual_truncated_successor_without_cross_reset():
    rewards = torch.tensor([[1.], [2.]])
    values = torch.tensor([[.5], [.7]])
    advantage, returns = module.generalized_advantage(torch, rewards, values, torch.tensor([.2]), gamma=1., lam=1.)
    np.testing.assert_allclose(advantage.numpy().ravel(), [2.7, 1.5], atol=1e-6)
    np.testing.assert_allclose(returns.numpy().ravel(), [3.2, 2.2], atol=1e-6)


def test_gaussian_ratio_uses_saved_raw_action_not_resampled_action():
    raw, mean, log_std = torch.tensor([[.2, -.3]]), torch.zeros(1, 2), torch.zeros(1, 2)
    logp = module.log_probability(torch, raw, mean, log_std)
    assert torch.equal((module.log_probability(torch, raw, mean, log_std)-logp).exp(), torch.ones(1))
    changed = module.log_probability(torch, raw, torch.ones_like(mean)*.4, log_std)
    assert not torch.equal(changed, logp)


def test_observation_has_no_clock_or_phase_and_contains_target_memory():
    task = object.__new__(module.DeviceTask)
    task.torch = torch
    task.reach = None
    task.q, task.v = torch.zeros(2, 23), torch.ones(2, 22)
    task.goal, task.target = torch.ones(2, 16), torch.zeros(2, 16)
    task.joint_scale, task.previous_action = torch.ones(16), torch.zeros(2, 16)
    before = task.observe()
    assert before.shape == (2, 93)
    task.time_s, task.phase, task.total_physics_steps = 10000., "insert", 999
    assert torch.equal(before, task.observe())
    task.previous_action[:, 0] = .5
    assert not torch.equal(before, task.observe())


def test_real_ppo_optimizer_uses_rollout_advantages_and_updates_parameters():
    # Synthetic HOST inputs test the optimizer implementation only. The actual
    # CLI independently audits physics successors -> reward -> stored GAE.
    torch.manual_seed(123)
    actor, critic = module.networks(torch, "cpu")
    obs = torch.randn(4, 2, 93)
    with torch.no_grad():
        mean, std = actor(obs.flatten(0, 1))
        raw = mean + std.exp()*torch.randn_like(mean)
        value = critic(obs.flatten(0, 1)).reshape(4, 2)
        logp = module.log_probability(torch, raw, mean, std).reshape(4, 2)
    storage = dict(obs=obs, raw_action=raw.reshape(4, 2, 16), logp=logp, value=value,
                   reward=torch.tensor([[.1, -.2], [.3, .7], [-.4, .8], [.2, .9]]), bootstrap=torch.zeros(2))
    optimizer = torch.optim.Adam(list(actor.parameters())+list(critic.parameters()), lr=.001)
    before = {k: v.clone() for k, v in actor.state_dict().items()}
    metrics, advantage, returns = module.optimize(torch, actor, critic, optimizer, storage,
                                                 SimpleNamespace(epochs=2, minibatch=4))
    assert metrics.shape == (4, 4) and torch.isfinite(metrics).all()
    assert (metrics[:, 3] > 0.).all()
    assert any(not torch.equal(v, before[k]) for k, v in actor.state_dict().items())
    assert advantage.shape == returns.shape == (8,)


def test_reach_has_distinct_observation_and_reward_contract():
    args = module.parser().parse_args(["--out", "unused", "--task", "reach"])
    contract = module.config(args, SimpleNamespace(xml_sha256="a"*64, timestep=.002))
    assert contract["observation_size"] == 98
    assert "actual-reach-features21" in contract["policy_input"]
    assert contract["reward"] != module.REWARD_DESCRIPTION


def test_checkpoint_contract_changes_with_model_and_control_interval():
    args = module.parser().parse_args(["--out", "unused"])
    scene = SimpleNamespace(xml_sha256="a"*64, timestep=.002)
    before = module.config(args, scene)
    args.action_repeat += 1
    assert module.config(args, scene) != before
    args.action_repeat -= 1
    scene.xml_sha256 = "b"*64
    assert module.config(args, scene) != before
    assert "not a grasp" in before["reward"]


def test_cpu_audit_rejects_missing_physical_rollout(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save(dict(schema=module.SCHEMA, rollout_file="missing.npz", rollout_sha256="0"*64), path)
    with pytest.raises(FileNotFoundError):
        module.audit_checkpoint(path, tmp_path / "audit.json")


@pytest.mark.parametrize("eval_seed", (None, 91001, -1))
def test_evaluation_requires_distinct_nonnegative_seed_before_device_import(eval_seed):
    args = module.parser().parse_args(["--out", "unused", "--stage", "evaluate"])
    args.eval_seed = eval_seed
    with pytest.raises(ValueError, match="distinct from the training"):
        module.run(args)


def test_optimizer_contract_does_not_claim_cpu_step_counters_are_gpu():
    args = module.parser().parse_args(["--out", "unused"])
    contract = module.config(args, SimpleNamespace(xml_sha256="a"*64, timestep=.002))
    assert "scalar step counters may be CPU" in contract["adam_execution"]


def test_initial_baseline_is_update_zero_and_matches_training_not_eval_contract(tmp_path):
    contract = dict(model_sha256="a"*64, task="joint-track", seed=91001,
                    physical_reset_sha256="b"*64, observation_size=93)
    before = {"weight": torch.tensor([1., 2.])}
    baseline = dict(schema=module.SCHEMA, contract=contract, next_update=1,
                    rollout_file="update-00000-rollout.npz", frozen_model_sha256="c"*64,
                    goal=torch.ones(2, 16), action_probe={"before_actor": before})
    final = dict(baseline, next_update=9, rollout_file="update-00008-rollout.npz")
    path = tmp_path / "update-00000.pt"
    torch.save(baseline, path)
    state, receipt = module.initial_baseline(torch, path, final)
    assert torch.equal(state["weight"], before["weight"])
    assert receipt["source_update"] == 0 and receipt["checkpoint_sha256"] == module._sha(path)
    # Independent evaluation metadata is deliberately not an input to this
    # source-to-source binding. A later checkpoint may not pose as update zero.
    for changed in (dict(next_update=2), dict(rollout_file="update-00001-rollout.npz"),
                    dict(contract=dict(contract, seed=91002)),
                    dict(contract=dict(contract, model_sha256="d"*64)),
                    dict(contract=dict(contract, task="reach")),
                    dict(frozen_model_sha256="d"*64), dict(goal=torch.zeros(2, 16))):
        torch.save(dict(baseline, **changed), path)
        with pytest.raises(ValueError):
            module.initial_baseline(torch, path, final)


def test_checkpoint_loading_is_weights_only_and_rejects_nonbasic_payload(tmp_path, monkeypatch):
    path = tmp_path / "safe.pt"
    torch.save(dict(schema=module.SCHEMA, tensor=torch.ones(2), optimizer={0: {"step": torch.tensor(1.)}}), path)
    real_load, calls = torch.load, []

    def observed_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", observed_load)
    result = module.load_checkpoint(torch, path)
    assert torch.equal(result["tensor"], torch.ones(2))
    assert calls == [dict(map_location="cpu", weights_only=True)]
    torch.save(dict(schema=module.SCHEMA, nonbasic=Path("not-a-basic-checkpoint-value")), path)
    with pytest.raises(Exception):
        module.load_checkpoint(torch, path)
    with pytest.raises(TypeError, match="basic types and tensors"):
        module.validate_checkpoint_data(torch, {"nonbasic": Path("also-rejected-before-saving")})
