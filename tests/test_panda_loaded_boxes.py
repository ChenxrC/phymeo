"""Identical Panda approach and transfer trajectories with controlled grip or payload changes."""

import json

import numpy as np
import pytest

from phymeo.experiments.panda_cube import run_panda_cube

SIZE = (.046, .058, .040)
OFFSET = (.001, 0.)
STRONG, WEAK = .024, .0448


@pytest.mark.parametrize(
    "label,mass,closed,carry",
    [("polymer-strong", .12, STRONG, True), ("polymer-weak", .12, WEAK, False),
     ("dense-strong", .95, STRONG, False)],
)
def test_asymmetric_box_loaded_and_slip_windows(label, mass, closed, carry, tmp_path):
    result = run_panda_cube(
        box_size_m=SIZE, mass_kg=mass, closed_width_m=closed, initial_xy_offset_m=OFFSET,
        friction=.65, record=True
    )
    (tmp_path / (label + ".json")).write_text(json.dumps(result), encoding="utf-8")
    x, y, z = SIZE
    expected = mass * np.array([y*y + z*z, z*z + x*x, x*x + y*y]) / 12
    np.testing.assert_allclose(result["cube_inertia_diagonal_kg_m2"], expected, rtol=2e-15)
    np.testing.assert_array_equal(result["cube_full_size_m"], SIZE)
    np.testing.assert_array_equal(result["initial_xy_offset_m"], OFFSET)
    assert result["closed_width_m"] == closed and len(result["trace"]) == result["steps"] == 3600
    assert result["cube_weight_N"] == pytest.approx(mass * 9.81)
    assert result["max_cube_momentum_residual_Ns"] < 1e-12
    assert result["peak_bilateral_normal_N"] > .02
    assert result["max_joint_limit_violation"] < 1e-4
    assert result["success"] is carry
    if carry:
        assert result["carry_clearance_m"] > .035 and result["carry_displacement_m"] > .045
        assert result["min_carry_bilateral_normal_N"] > 0 and result["peak_carry_table_N"] < .001
        assert result["final_target_error_m"] < .012 and result["release_pad_force_N"] < .001
    else:
        assert result["carry_clearance_m"] < .005 and result["carry_displacement_m"] < .015
    assert abs(result["final_table_force_N"] - mass * 9.81) < .025


@pytest.mark.parametrize("changes", [
    {"box_size_m": (0., .05, .05)}, {"box_size_m": (.05, np.nan, .05)},
    {"box_size_m": (.05, .04)}, {"closed_width_m": .09}, {"initial_xy_offset_m": (.02, np.inf)},
])
def test_invalid_grasp_parameters_fail_before_a_physical_step(changes):
    with pytest.raises(ValueError):
        run_panda_cube(**changes)
