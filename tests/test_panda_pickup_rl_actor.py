"""Frozen pickup prior, native feature parity, and residual policy density."""
from importlib import import_module
import io
import math
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    torch = pytest.importorskip("torch")
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    module = import_module("_panda_pickup_rl_actor")
    directory = examples / "checkpoints/panda-pickup-from-reach-s4000"
    return torch, module, directory


def test_device_features_match_native_pickup_measurements(modules):
    torch, module, directory = modules
    mj = pytest.importorskip("mujoco")
    native = import_module("panda_pickup_from_reach")
    model = mj.MjModel.from_xml_path(str(native.PUBLISHED / "scene.xml"))
    reach = import_module("_panda_ppo_reach").ReachTask(torch, model, "cpu")
    with np.load(directory / "factor-update.npz", allow_pickle=False) as handoffs:
        initial, targets = handoffs["qpos"], handoffs["target"]
    q = initial.copy()
    q[:, 13] += .25
    q[:, 18] += .013
    init = torch.tensor(initial, dtype=torch.float32)
    batch = torch.tensor(q, dtype=torch.float32)
    z0 = reach.poses(init)[2][0][:, 2]
    features = module.pickup_features(torch, reach.poses(batch), batch, torch.tensor(targets).float(), z0)
    data = mj.MjData(model)
    expected = []
    for reset, state, target in zip(initial, q, targets, strict=True):
        observer = native.PickupFeatures(model, reset)
        data.qpos[:] = state
        mj.mj_kinematics(model, data)
        expected.append(observer.observe(data, target))
    np.testing.assert_allclose(features.numpy(), expected, rtol=2e-5, atol=2e-6)


def test_zero_residual_prior_and_checkpoint(modules):
    torch, module, directory = modules
    path = directory / "student.npz"
    student = import_module("_panda_pickup_student").PickupStudent.load(path)
    policy = module.ResidualPickupActorCritic.from_student(path, "cpu")
    z = student.centers[::7]
    x = student.mean + np.divide(z, student.inverse_scale, out=np.zeros_like(z), where=student.inverse_scale != 0.)
    obs = policy.observation(x, np.zeros((len(x), 22)))
    base = policy.prior_action(obs[:, :21])
    np.testing.assert_allclose(base.detach().numpy(), student.act(obs[:, :21].numpy()), atol=2e-5, rtol=2e-5)
    with torch.no_grad():
        action, raw, _, _ = policy.sample(obs, deterministic=True)
        torch.testing.assert_close(action, base, atol=1.1e-6, rtol=0.)
        assert not torch.any(raw)
        policy.actor[-1].bias.add_(.15)
    out = io.BytesIO()
    torch.save(policy.checkpoint(), out)
    out.seek(0)
    checkpoint = torch.load(out, weights_only=True, map_location="cpu")
    restored = module.ResidualPickupActorCritic.from_checkpoint(checkpoint, path, "cpu")
    for a, b in zip(policy.sample(obs, True), restored.sample(obs, True), strict=True):
        torch.testing.assert_close(a, b, atol=0., rtol=0.)
    checkpoint["contract"]["velocity_scale"][0] = 3.
    with pytest.raises(ValueError, match="contract"):
        module.ResidualPickupActorCritic.from_checkpoint(checkpoint, path, "cpu")


def test_density_matches_motor_transform_and_ppo_updates_only_trainable_parts(modules):
    torch, module, directory = modules
    policy = module.ResidualPickupActorCritic.from_student(directory / "student.npz", "cpu")
    obs = policy.observation(policy.prior_mean.expand(4, -1), torch.zeros(4, 22))
    torch.manual_seed(271136)
    with torch.no_grad():
        action, raw, old, _ = policy.sample(obs)
        assert (action.abs() <= 1.).all() and (raw > 0.).any() and (raw < 0.).any()
        prior = policy.prior_action(obs[:, :21]).clamp(-1.+1e-6, 1.-1e-6)
        scale = torch.where(raw >= 0., 1.-prior, 1.+prior)
        expected = torch.distributions.Normal(torch.zeros_like(raw), policy.log_std.exp()).log_prob(raw)
        expected -= (scale * (1.-raw.tanh().square())).log()
        torch.testing.assert_close(old, expected.sum(-1), atol=2e-5, rtol=2e-6)
    logp, entropy, value = policy.evaluate(obs, raw)
    torch.testing.assert_close(logp, old)
    assert entropy.shape == value.shape == (4,)
    frozen = policy.prior_weights.clone()
    before = policy.actor[-1].weight.detach().clone()
    opt = torch.optim.Adam(policy.parameters(), lr=.001)
    loss = -(logp-old).exp().mean() + (value-1.).square().mean() - .001*entropy.mean()
    loss.backward()
    opt.step()
    assert not torch.equal(before, policy.actor[-1].weight)
    torch.testing.assert_close(frozen, policy.prior_weights, atol=0., rtol=0.)


@pytest.mark.parametrize("angle", [0., 1e-9, math.pi-1e-8, math.pi])
def test_rotation_vector_is_stable_near_identity_and_pi(modules, angle):
    torch, module, _ = modules
    c, s = math.cos(angle), math.sin(angle)
    rotation = torch.tensor([[[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]])
    actual = module._rotvec(torch, rotation)[0]
    torch.testing.assert_close(actual, torch.tensor([0., 0., angle]), atol=3e-7, rtol=1e-6)
