"""Conditional action density, episode resets and measured pickup milestones."""
from importlib import import_module
from pathlib import Path

import pytest


@pytest.fixture
def actor(monkeypatch):
    torch = pytest.importorskip("torch")
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    module = import_module("_panda_pickup_rl_actor")
    student = examples / "checkpoints/panda-pickup-from-reach-s4000/student.npz"
    policy = module.ResidualPickupActorCritic.from_student(student, "cpu", arm_std=.006, finger_std=.04, arm_rho=.8)
    with torch.no_grad():
        policy.actor[-1].bias.copy_(torch.linspace(-.15, .15, 16))
    obs = policy.observation(policy.prior_mean.expand(3, -1), torch.zeros(3, 22))
    return torch, module, policy, obs, student


def test_conditional_draw_matches_independent_density_and_history_has_no_gradient(actor):
    torch, _, policy, obs, _ = actor
    prev = torch.linspace(.1, .2, 48).reshape(3, 16).requires_grad_()
    mask = torch.tensor([False, True, True])
    mean = policy.actor[-1].bias.detach().expand(3, -1).clone()
    mean[1:, :14] = .2*mean[1:, :14] + .8*prev.detach()[1:, :14]
    sigma = torch.full_like(mean, .006)
    sigma[1:, :14] *= .6
    sigma[:, 14:] = .04
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(97)
        expected_raw = mean + sigma*torch.randn_like(mean)
        torch.manual_seed(97)
        action, raw, old, _ = policy.sample(obs, prev_raw=prev, has_prev=mask)
        torch.testing.assert_close(raw, expected_raw, atol=2e-8, rtol=1e-6)
        prior = policy.prior_action(obs[:, :21]).clamp(-1.+1e-6, 1.-1e-6)
        scale = torch.where(raw >= 0., 1.-prior, 1.+prior)
        torch.testing.assert_close(action, prior+scale*raw.tanh())
        density = torch.distributions.Normal(mean, sigma)
        expected_logp = (density.log_prob(raw) - (scale*(1.-raw.tanh().square())).log()).sum(-1)
        torch.testing.assert_close(old, expected_logp, atol=4e-5, rtol=2e-6)
    # Re-evaluate shuffled transitions, including a first-decision row.
    order = torch.tensor([2, 0, 1])
    logp, entropy, _ = policy.evaluate(obs[order], raw[order], prev_raw=prev[order], has_prev=mask[order])
    torch.testing.assert_close(logp, old[order])
    torch.testing.assert_close(entropy, density.entropy().sum(-1)[order])
    logp.sum().backward()
    assert prev.grad is None and policy.actor[-1].bias.grad is not None
    with pytest.raises(ValueError, match="conditioning"):
        policy.evaluate(obs, raw)


def test_rollout_saves_raw_history_and_deterministic_episode_restarts(actor):
    torch, module, policy, obs, student = actor
    records = {}
    with torch.no_grad():
        _, first = module.rollout_action(policy, obs, "train", records, None)
        module.rollout_action(policy, obs, "train", records, first)
        assert not records["has_prev"][0].any() and records["has_prev"][1].all()
        assert not records["prev_raw"][0].any()
        torch.testing.assert_close(records["prev_raw"][1], records["raw"][0], atol=0., rtol=0.)
        _, initial = module.rollout_action(policy, obs, "candidate", {}, None)
        policy.actor[-1].bias.add_(.25)
        _, following = module.rollout_action(policy, obs, "candidate", {}, initial)
        _, new_episode = module.rollout_action(policy, obs, "candidate", {}, None)
        torch.testing.assert_close(following[:, :14], .8*initial[:, :14]+.2*new_episode[:, :14])
        torch.testing.assert_close(following[:, 14:], new_episode[:, 14:])
    saved = policy.checkpoint()
    rebound = policy.from_checkpoint(saved, student, "cpu")
    assert rebound.exploration == policy.exploration and rebound.arm_rho == .8
    torch.testing.assert_close(rebound.log_std, policy.log_std, atol=0., rtol=0.)
    saved["contract"]["exploration"]["log_std_bounds"][0] = -5.
    with pytest.raises(ValueError, match="contract"):
        policy.from_checkpoint(saved, student, "cpu")


def test_funnel_counts_physical_milestones_separately(actor):
    torch, _, _, _, _ = actor
    pytest.importorskip("mujoco")
    from _panda_pickup_rl_reward import PickupFunnel

    tracker = PickupFunnel(2, "cpu", weight_n=2., dt=.002)
    good = torch.tensor([True, False])
    no = torch.zeros(2, dtype=torch.bool)
    info = dict(grip=good, loading=no, slipped=no, height_m=torch.zeros(2), carried=no, success=no,
                handle_min_normal_n=torch.tensor([1., 0.]), pad_up_n=torch.tensor([.1, 0.]),
                slip_m=torch.tensor([.001, .9]), dwell_intervals=torch.tensor([0, 0]))
    tracker.update(info)
    tracker.update(dict(info, loading=good, carried=good, success=good, height_m=torch.tensor([.03, .08]),
                        slip_m=torch.tensor([.003, 1.]), dwell_intervals=torch.tensor([75, 0])))
    result = tracker.summary()
    assert result["worlds_ever"] == dict(bilateral_handle=1, grip=1, loading=1, slipped=0, lift_25mm=2,
                                         carried=1, success=1)
    lifted, ejected = result["per_world"]
    assert lifted["peak_grip_slip_m"] == pytest.approx(.003)
    assert lifted["peak_loaded_rise_m"] == pytest.approx(.03)
    assert lifted["longest_loaded_dwell_s"] == pytest.approx(.15)
    assert ejected["ever"]["lift_25mm"] and not ejected["ever"]["loading"]
    assert ejected["peak_grip_slip_m"] is None
    assert ejected["peak_loaded_rise_m"] is None


def test_independent_micro_law_preserves_history_free_rollout_records(actor):
    torch, module, _, obs, student = actor
    iid = module.ResidualPickupActorCritic.from_student(student, "cpu", arm_std=.006, finger_std=.04, arm_rho=0.)
    records = {}
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(813)
        _, first = module.rollout_action(iid, obs, "train", records, None)
        torch.manual_seed(813)
        _, second = module.rollout_action(iid, obs, "train", records, first)
    assert set(records) == {"obs", "raw", "value", "logp"}
    torch.testing.assert_close(first, second, rtol=0., atol=0.)
    actual_scale = iid._distribution(obs)[2].exp()
    torch.testing.assert_close(actual_scale[:14], torch.full((14,), .006), rtol=2e-7, atol=0.)
