"""The same physical task must distinguish frictional transport from an empty grasp."""

import json

import numpy as np
import pytest

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.experiments.panda_cube import run_panda_cube
from phymeo.solvers.sampled_box_contact import box_surface_samples


def test_box_surface_load_uses_the_rotated_material_normal_and_point_speed():
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    position = np.array([2.0, 3.0, 4.0])
    samples = position + np.array([[0.0, 0.048, 0.01], [0.0, 0.055, 0.01]])
    material = RobotContactMaterial(1000.0, 0.0, 0.5, 0.01)
    common = np.array([0.3, 0.2, 0.1])
    contact = box_surface_samples(
        samples,
        np.tile(common + [0.0, 0.0, 0.2], (2, 1)),
        position,
        rotation,
        common,
        np.zeros(3),
        [0.05, 0.03, 0.02],
        material,
    )
    np.testing.assert_allclose(contact.forces, [[0, 2, -1], [0, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(contact.depths, [0.002, 0], atol=1e-12)
    np.testing.assert_allclose(contact.forces.sum(axis=0) + contact.body_force, 0, atol=1e-12)
    np.testing.assert_allclose(
        contact.body_torque, np.cross(samples[0] - position, -contact.forces[0])
    )


@pytest.mark.parametrize("friction,should_carry", [(0.65, True), (0.0, False)])
def test_imported_panda_grips_and_moves_a_free_cube(friction, should_carry, tmp_path):
    result = run_panda_cube(friction=friction, record=True)
    (tmp_path / "motion.json").write_text(json.dumps(result), encoding="utf-8")
    assert result["max_cube_momentum_residual_Ns"] < 1e-12
    assert result["success"] == should_carry
    assert result["peak_bilateral_normal_N"] > 1.0
    assert result["max_pad_penetration_m"] < 0.006
    assert result["max_joint_limit_violation"] < 1e-4
    if should_carry:
        assert result["carry_clearance_m"] > 0.035
        assert result["carry_displacement_m"] > 0.045
        assert result["final_target_error_m"] < 0.012
        assert result["min_carry_bilateral_normal_N"] > 0
        assert result["peak_carry_table_N"] < 0.001
        assert result["release_pad_force_N"] < 0.001
        assert abs(result["final_table_force_N"] - result["cube_weight_N"]) < 0.025
    else:
        assert result["carry_clearance_m"] < 0.005
        assert result["carry_displacement_m"] < 0.015
