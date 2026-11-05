"""Public combined use: coupled forces, real affine motion, and atomic history."""

import importlib.util
from pathlib import Path

import pytest


def _example():
    path = Path(__file__).resolve().parents[1] / "examples" / "coupled_tangent_history.py"
    assert path.is_file(), "the coupled material-history workflow is missing"
    spec = importlib.util.spec_from_file_location("coupled_history_workflow", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_station_window_commits_actual_motion_once():
    result = _example().run_window()
    assert result["accepted_motion_steps"] == 192
    assert result["history_steps"] == 193
    assert result["station_count"] == 2
    assert result["final_active_stations"] == 0
    assert result["stale_trial_rejected"]
    assert result["bad_motion_rejected_without_commit"]
    assert result["same_input_retry_identical"]
    assert result["reordered_station_steps"] == 96
    assert result["sliding_steps"] > 0
    assert result["released_spring_energy_j"] > 0
    assert result["maximum_momentum_error_ns"] < 1e-12
    assert result["maximum_motion_error_m"] < 1e-12
    assert result["maximum_energy_balance_error_j"] < 1e-11
    h = 1 / 256
    expected = h * h * 1.5 / (1 + 200 * h * h * 1.5)
    assert result["first_relative_displacement_m"] == pytest.approx(expected, abs=1e-15)
    assert len(result["frames"]) == 192
    assert [result["frames"][i]["external_pair_force_n"] for i in (0, 64, 128)] == [1, 3, -1]
    assert result["motion_domain"] == "affine_translation_given_normal_load"
    assert result["tangent_backend"] == "host"
