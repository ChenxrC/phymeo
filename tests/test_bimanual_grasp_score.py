"""Decision-sample regressions for the offline grasp score."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def score_transport(monkeypatch):
    pytest.importorskip("mujoco")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    scorer = import_module("bimanual_grasp_score")

    def score(phases, airborne, height_mm, yaw_deg):
        count = len(phases)
        ticks = 8 * np.arange(1, count + 1)
        positions = np.zeros((count, 2, 3))
        positions[:, :, 2] = .001 * np.array(height_mm)[:, None]
        rotations = np.tile(np.eye(3), (count, 2, 1, 1))
        for i, angle in enumerate(np.deg2rad(yaw_deg)):
            c, s = np.cos(angle), np.sin(angle)
            rotations[i, :, :2, :2] = [[c, -s], [s, c]]
        up = np.tile([0., 0., 1.], (count, 2, 1))
        return scorer._transport(np.asarray(airborne), phases, ticks, .002 * ticks,
                                 (positions, rotations, up), scorer.SIDES)

    return score


def test_trial_lift_without_airborne_transport_has_no_transport_score(score_transport):
    # The lift is stable; the sole transport command occurs after touchdown.
    result = score_transport(["lift", "lift", "transport"], [True, True, False],
                             [5, 5, 5], [0, 0, 0])
    assert result is None


def test_split_transport_resets_position_rotation_and_duration(score_transport):
    result = score_transport(["transport", "transport", "lower", "transport", "transport", "transport"],
                             [True] * 6, [0, -3, -50, -60, -62, -65], [0, 4, 42, 55, 59, 61])

    assert result["bout_count"] == 2
    assert result["samples"] == 5
    first, second = result["bouts"]
    assert (first["first_output_tick"], first["last_output_tick"]) == (8, 16)
    assert (second["first_output_tick"], second["last_output_tick"]) == (32, 48)
    assert (first["duration_s"], second["duration_s"]) == pytest.approx((.016, .032))
    assert result["duration_s"] == pytest.approx(.048)
    assert result["max_downward_slip_mm"] == pytest.approx(5.)
    assert result["max_relative_rotation_deg"] == pytest.approx(6.)
