"""End-to-end paired observation benchmark; no learning-quality assertion."""

import importlib.util
from pathlib import Path

import pytest


def api():
    path = (
        Path(__file__).resolve().parents[2] / "examples/compare_mujoco_tyx_observation_training.py"
    )
    spec = importlib.util.spec_from_file_location("paired_training_observation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.cuda
@pytest.mark.parametrize("worlds", (8, 128, 256))
@pytest.mark.parametrize("name", ("J", "S", "R"))
def test_complete_paired_ppo(name, worlds, tmp_path):
    result = api().run_pair(name, worlds, tmp_path / "result")
    assert result["status"] == "PASS_PAIRED_PPO_OBSERVATION"
    assert result["physics_and_learner_identical"]
    assert result["measured_pairs"] == 2
    for mode in ("full", "integration"):
        row = result["modes"][mode]
        assert row["measured_world_steps"] == 32 * worlds
        assert len(row["signatures"]) == 3
        assert row["signatures"][0]["learner"] != row["initial_policy_hash"]
        assert row["counter_end"]["physical_steps"] == 48
        assert row["counter_end"]["transitions"] == 48
        assert row["loop_to_trace_close_s"] > row["compute_and_mandatory_diagnostics_s"] > 0
