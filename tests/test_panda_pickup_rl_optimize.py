from importlib import import_module
from pathlib import Path

import pytest


def test_pickup_ppo_terminal_credit_and_frozen_prior(monkeypatch):
    torch = pytest.importorskip("torch")
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    policy_type = import_module("_panda_pickup_rl_actor").ResidualPickupActorCritic
    update = import_module("_panda_pickup_rl_optimize").update_policy
    torch.manual_seed(813)
    policy = policy_type.from_student(examples / "checkpoints/panda-pickup-from-reach-s4000/student.npz", "cpu")
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    prior = {key: value.clone() for key, value in policy.named_buffers()}
    obs = torch.cat((policy.prior_mean.expand(2, 2, -1), .1 * torch.randn(2, 2, 22)), -1)
    with torch.no_grad():
        _, raw, logp, _ = policy.sample(obs)
    records = dict(obs=obs, raw=raw, logp=logp, value=torch.tensor([[2., 4.], [5., -1.]]),
                   reward=torch.tensor([[1., 2.], [3., 5.]]))
    report = update(torch, policy, optimizer, records, epochs=2, minibatch=3)
    # Terminal returns equal [3,5]; t=0 uses V(t+1) and the terminal advantages [-2,6].
    returns = [1 + .99 * 5 + .99 * .95 * (-2), 2 + .99 * (-1) + .99 * .95 * 6, 3., 5.]
    assert report["mean_gae_return"] == pytest.approx(sum(returns) / 4, abs=1e-6)
    assert (report["samples"], report["sample_visits"], report["optimizer_steps"]) == (4, 8, 4)
    assert report["actor_max_parameter_change"] > 0 and report["critic_max_parameter_change"] > 0
    assert report["max_grad_norm"] >= report["mean_grad_norm"] > 0
    assert 0 <= report["clip_fraction"] <= 1 and report["approx_kl"] >= 0
    with torch.no_grad():
        final, _, _ = policy.evaluate(obs, raw)
        ratio = (final - logp).exp()
        expected = (ratio - 1 - (final - logp)).clamp_min(0.).mean().item()
    assert report["approx_kl"] == pytest.approx(expected, abs=2e-6)
    for key, value in policy.named_buffers():
        torch.testing.assert_close(value, prior[key], rtol=0, atol=0)

    before = policy.log_std.detach().clone()
    records["reward"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite PPO rollout"):
        update(torch, policy, optimizer, records)
    torch.testing.assert_close(before, policy.log_std, rtol=0, atol=0)
