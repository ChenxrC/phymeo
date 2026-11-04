"""A rotated uniform payload on the ordinary Panda pick, translate and release path."""

import json

import numpy as np
import pytest

from phymeo.experiments.panda_cube import run_panda_cube

SIZE = np.array([0.046, 0.058, 0.040])
OFFSET = np.array([0.003, -0.001])
YAW = np.pi / 15
MASS = 0.120


@pytest.mark.parametrize("label,mu,carry", [("grip", 0.65, True), ("slip", 0.0, False)])
def test_rotated_offset_box_window(label, mu, carry, tmp_path):
    result = run_panda_cube(
        initial_yaw_rad=YAW, initial_xy_offset_m=OFFSET, box_size_m=SIZE,
        closed_width_m=0.024, grip_above_com_m=0.012, mass_kg=MASS, friction=mu, record=True,
    )
    spec, trace = result["inputs"], result["trace"]
    (tmp_path / f"rotated-{label}.json").write_text(json.dumps(result), encoding="utf8")
    (tmp_path / f"rotated-{label}.inputs.json").write_text(json.dumps(spec), encoding="utf8")
    # At this ordinary home pose the first material axis initially points along -Y.
    half_angle = (YAW - np.pi / 2) / 2
    q0 = np.asarray(spec["body_xyzw"])
    assert abs(q0 @ [0.0, 0.0, np.sin(half_angle), np.cos(half_angle)]) == pytest.approx(1, abs=1e-14)
    assert result["initial_yaw_rad"] == pytest.approx(YAW)
    table = np.asarray(spec["table_center_m"])
    center = np.asarray(spec["body_position_m"])
    initial_pads = np.mean(trace[0]["pad_points_before_m"], axis=0)
    np.testing.assert_allclose(center[:2] - initial_pads[:2], OFFSET, atol=2e-12, rtol=0)
    np.testing.assert_allclose(center[:2], table[:2] - [0.0375, 0] + OFFSET, atol=2e-12, rtol=0)
    assert center[2] - SIZE[2] / 2 == pytest.approx(table[2] + spec["table_full_size_m"][2]/2)
    target = center + [0.075, 0.0, 0.0]
    np.testing.assert_allclose(np.linalg.norm(trace[-1]["cube"] - target), result["final_target_error_m"])
    np.testing.assert_allclose(spec["inertia_diagonal_kg_m2"], [4.964e-5, 3.716e-5, 5.480e-5], atol=1e-18, rtol=0)
    assert spec["full_size_m"] == SIZE.tolist() and "trace" not in spec
    assert result["steps"] == len(trace) == 3600 and trace[-1]["t"] == 2.25
    assert result["peak_bilateral_normal_N"] > 1.0
    assert result["max_joint_limit_violation"] < 1e-4
    assert result["other_robot_cube_axis_clearance_m"] > 0 and result["robot_table_axis_clearance_m"] > 0
    assert result["max_cube_momentum_residual_Ns"] < 1e-12
    assert result["release_pad_force_N"] < 0.001
    assert abs(result["final_table_force_N"] - MASS * 9.81) < 0.025
    # A real horizontal offset and rotated faces can transmit a moment to the body.
    assert min(abs(np.asarray(r["xyzw"]) @ q0) for r in trace) < np.cos(0.005 / 2)
    assert result["success"] is carry and result["carried_during_transfer"] is carry
    if carry:
        assert result["carry_clearance_m"] > 0.035 and result["carry_displacement_m"] > 0.045
        assert result["min_carry_bilateral_normal_N"] > 0 and result["peak_carry_table_N"] < 0.001
        assert result["released_on_target"] and result["final_target_error_m"] < 0.012
    else:
        assert result["carry_clearance_m"] < 0.005 and abs(result["carry_displacement_m"]) < 0.015
        assert result["peak_carry_table_N"] > 0.95 and result["min_carry_bilateral_normal_N"] == 0
        assert not result["released_on_target"] and result["final_target_error_m"] > 0.05


@pytest.mark.parametrize("yaw", [float("nan"), float("inf"), True, [0, 0, 0, 1]])
def test_invalid_yaw_is_rejected_before_force_evaluation(yaw):
    with pytest.raises(ValueError, match="initial_yaw_rad"):
        run_panda_cube(initial_yaw_rad=yaw)


def test_rotated_footprint_overhangs_finite_table_before_advance():
    # At zero yaw 31 + 23 mm fits the 55 mm half-table. At -45 degrees it is 67.77 mm.
    with pytest.raises(ValueError, match="finite tabletop"):
        run_panda_cube(box_size_m=SIZE, initial_yaw_rad=-np.pi / 4, initial_xy_offset_m=(0.0, 0.031))
