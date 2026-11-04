"""Identical Panda approach and transfer trajectories with controlled grip or payload changes."""

import json

import numpy as np
import pytest

from phymeo.experiments.panda_cube import run_panda_cube

SIZE = (0.046, 0.058, 0.040)
OFFSET = (0.001, 0.0)
STRONG, WEAK = 0.024, 0.0448


@pytest.mark.parametrize(
    "label,mass,closed,carry",
    [
        ("polymer-strong", 0.12, STRONG, True),
        ("polymer-weak", 0.12, WEAK, False),
        ("dense-strong", 0.95, STRONG, False),
    ],
)
def test_asymmetric_box_loaded_and_slip_windows(label, mass, closed, carry, tmp_path):
    result = run_panda_cube(
        box_size_m=SIZE,
        mass_kg=mass,
        closed_width_m=closed,
        initial_xy_offset_m=OFFSET,
        friction=0.65,
        record=True,
    )
    (tmp_path / (label + ".json")).write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / (label + ".inputs.json")).write_text(json.dumps(result["inputs"]), encoding="utf-8")
    manifest = result["inputs"]
    np.testing.assert_array_equal(
        manifest["knot_times_s"], [0, 0.32, 0.55, 0.92, 1.35, 1.72, 1.95, 2.25]
    )
    widths = 2 * np.array(manifest["knot_q"])[:, 7:]
    expected_width = np.array([0.076, 0.076, closed, closed, closed, closed, 0.076, 0.076])
    np.testing.assert_allclose(widths, np.tile(expected_width[:, None], (1, 2)), atol=1e-14)
    assert "trace" not in manifest
    x, y, z = SIZE
    expected = mass * np.array([y * y + z * z, z * z + x * x, x * x + y * y]) / 12
    np.testing.assert_allclose(result["cube_inertia_diagonal_kg_m2"], expected, rtol=2e-15)
    np.testing.assert_array_equal(result["cube_full_size_m"], SIZE)
    np.testing.assert_array_equal(result["initial_xy_offset_m"], OFFSET)
    assert result["closed_width_m"] == closed and len(result["trace"]) == result["steps"] == 3600
    assert result["cube_weight_N"] == pytest.approx(mass * 9.81)
    assert result["max_cube_momentum_residual_Ns"] < 1e-12
    assert result["peak_bilateral_normal_N"] > 0.02
    assert result["max_joint_limit_violation"] < 1e-4
    assert result["carried_during_transfer"] is carry
    assert result["success"] is carry
    if carry:
        assert result["carry_clearance_m"] > 0.035 and result["carry_displacement_m"] > 0.045
        assert result["min_carry_bilateral_normal_N"] > 0 and result["peak_carry_table_N"] < 0.001
        assert result["final_target_error_m"] < 0.012 and result["release_pad_force_N"] < 0.001
    else:
        assert result["carry_clearance_m"] < 0.005 and result["carry_displacement_m"] < 0.015
    assert abs(result["final_table_force_N"] - mass * 9.81) < 0.025


@pytest.mark.parametrize(
    "changes",
    [
        {"box_size_m": (0.0, 0.05, 0.05)},
        {"box_size_m": (0.05, np.nan, 0.05)},
        {"box_size_m": (0.05, 0.04)},
        {"closed_width_m": 0.09},
        {"initial_xy_offset_m": (0.02, np.inf)},
    ],
)
def test_invalid_grasp_parameters_fail_before_a_physical_step(changes):
    with pytest.raises(ValueError):
        run_panda_cube(**changes)
