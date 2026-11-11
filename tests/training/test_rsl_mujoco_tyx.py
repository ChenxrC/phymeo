"""Upstream RSL5.5.2 optimization and recurrent timeout protocol."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.cuda, pytest.mark.tyx]
ROOT = Path(__file__).resolve().parents[2]
SEEDS = {"hinge-slide": 11011, "support-slide": 11012}
HORIZONS = (5, 7, 9, 11, 13, 15, 17, 19)


@pytest.fixture
def api():
    spec = importlib.util.spec_from_file_location(
        "n2_ppo_example", ROOT / "examples" / "train_mujoco_tyx_ppo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.require_cuda()
    return module


@pytest.mark.parametrize("case", tuple(SEEDS))
def test_actual_upstream_cuda_ppo_four_updates(case, api, tmp_path):
    summary = api.train_case(case, tmp_path / "ppo.jsonl")
    assert summary["status"] == "PASS_N2_SHORT_PPO"
    assert summary["seed"] == SEEDS[case]
    assert summary["updates"] == 4 and summary["transitions"] == 512
    assert summary["optimizer_steps"] == [1, 2, 3, 4]
    assert summary["all_losses_finite"] and summary["all_gradients_finite"]
    assert summary["parameter_hash_before"] != summary["parameter_hash_after"]
    assert summary["policy_device"] == "cuda:0"
    rows = [json.loads(line) for line in (tmp_path / "ppo.jsonl").read_text().splitlines()]
    steps = [row for row in rows if row["event"] == "transition"]
    updates = [row for row in rows if row["event"] == "update"]
    assert len(steps) == 64 and len(updates) == 4
    assert all(row["optimizer_step_after"] == row["update"] + 1 for row in updates)
    assert all(row["parameter_hash_before"] != row["parameter_hash_after"] for row in updates)
    assert summary["episode_end_counts"][0] >= 1
    assert summary["episode_end_counts"][1] >= 1
    assert summary["runtime_counters"]["accepted_world_steps"] == 512


def test_upstream_timeout_bonus_uses_entry_value_without_final_critic(api):
    from phymeo.training.rsl_mujoco_tyx import RslMuJoCoTyxVecEnv, make_ppo

    torch = api.require_cuda()
    with api.make_environment("hinge-slide", horizons=(1, 7, 9, 11, 13, 15, 17, 19)) as env:
        adapter = RslMuJoCoTyxVecEnv(env)
        algorithm = make_ppo(adapter, steps=2)
        observation = adapter.get_observations()
        with torch.inference_mode():
            action = algorithm.act(observation)
            entry_value = algorithm.transition.values.clone()
            following, reward, done, extras = adapter.step(action)
            calls = []
            handle = algorithm.critic.register_forward_pre_hook(lambda *args: calls.append(True))
            try:
                algorithm.process_env_step(following, reward, done, extras)
            finally:
                handle.remove()
        expected = reward + algorithm.gamma * entry_value[:, 0] * extras["time_outs"]
        assert torch.equal(algorithm.storage.rewards[0, :, 0], expected)
        assert not calls
        assert done[0].item() and extras["time_outs"][0].item()
        assert not extras["time_outs"][1:].any().item()
        assert algorithm.storage.step == 1


def test_gru_partial_done_masks_preserve_other_hidden_and_update(api, tmp_path):
    summary = api.train_recurrent(tmp_path / "gru.jsonl")
    assert summary["status"] == "PASS_N2_GRU_PROTOCOL"
    assert summary["steps"] == 8 and summary["transitions"] == 64
    assert summary["masked_actor_checks"] >= 2
    assert summary["masked_critic_checks"] >= 2
    assert summary["optimizer_steps"] == 1
    assert summary["finite_loss"] and summary["finite_gradient"]
    assert summary["parameter_hash_before"] != summary["parameter_hash_after"]


def test_adapter_refuses_clock_only_randomization(api):
    from phymeo.training.rsl_mujoco_tyx import RslMuJoCoTyxVecEnv

    with api.make_environment("hinge-slide") as env:
        adapter = RslMuJoCoTyxVecEnv(env)
        before = env.runtime.checkpoint().to_bytes()
        with pytest.raises(NotImplementedError, match="init_at_random_ep_len"):
            adapter.episode_length_buf = adapter.episode_length_buf + 1
        assert env.runtime.checkpoint().to_bytes() == before


def test_adapter_packaging_failure_stops_learner_after_real_step(api, monkeypatch):
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxEnvError
    from phymeo.training.rsl_mujoco_tyx import RslMuJoCoTyxVecEnv

    torch = api.require_cuda()
    with api.make_environment("hinge-slide") as env:
        adapter = RslMuJoCoTyxVecEnv(env)
        original = adapter._observations
        def fail(value):
            raise RuntimeError("injected TensorDict packaging failure")
        monkeypatch.setattr(adapter, "_observations", fail)
        with pytest.raises(MuJoCoTyxEnvError) as caught:
            adapter.step(torch.zeros((8, env.num_actions), device=env.device))
        assert caught.value.phase == "adapter-publication"
        assert caught.value.state_accepted and adapter.needs_recovery
        assert env.last_transition.sequence == 1
        assert all(state.tick == 1 for state in env.runtime.snapshot().values())
        with pytest.raises(RuntimeError, match="recovery"):
            adapter.step(torch.zeros((8, env.num_actions), device=env.device))
        monkeypatch.setattr(adapter, "_observations", original)
        adapter.reset()
        assert np.all(env.episode_length == 0)
        assert not adapter.needs_recovery
