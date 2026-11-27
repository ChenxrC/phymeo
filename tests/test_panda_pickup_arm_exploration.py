"""Arm innovation law and predeclared training-domain selection."""
from importlib import import_module
from pathlib import Path

import pytest


@pytest.fixture
def exploration(monkeypatch):
    pytest.importorskip("mujoco")
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return torch, import_module("panda_pickup_arm_exploration")


def test_correlated_arms_share_innovations_and_fingers_stay_iid(exploration):
    torch, module = exploration
    draws = torch.randn((12, 3, 16), generator=torch.Generator().manual_seed(481))
    original = draws.clone()
    iid = module.latent_noise(draws, module.LAWS["fine_iid"])
    correlated = module.latent_noise(draws, module.LAWS["fine_arm_ar"])
    torch.testing.assert_close(correlated[0], iid[0])
    torch.testing.assert_close(correlated[:, :, 14:], iid[:, :, 14:])
    torch.testing.assert_close(correlated[1:, :, :14], .8*correlated[:-1, :, :14] + .6*iid[1:, :, :14])
    torch.testing.assert_close(draws, original)


def test_selection_excludes_baselines_and_checks_each_domain(exploration):
    _, module = exploration
    groups = [dict(branches={k: dict(counts=dict(loaded_pickup=0, lift=0, bilateral_handle=0))
                            for k in module.LAWS}) for _ in range(2)]
    for group in groups:
        group["branches"]["batch401_thin_fingers_iid"]["counts"]["loaded_pickup"] = 15
    assert module.select_law(groups)["selected"] == "micro_iid"
    assert not module.select_law(groups)["observed_loaded_success"]
    groups[0]["branches"]["micro_iid"]["counts"]["loaded_pickup"] = 4
    for group in groups:
        group["branches"]["fine_arm_ar"]["counts"]["loaded_pickup"] = 2
    assert module.select_law(groups)["selected"] == "fine_arm_ar"
    for group in groups:
        group["branches"]["fine_iid"]["counts"]["loaded_pickup"] = 2
    assert module.select_law(groups)["selected"] == "fine_iid"
