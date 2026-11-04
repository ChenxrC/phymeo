from types import SimpleNamespace

import numpy as np
import pytest
from structural_impact_balance import generalized_reaction, pair_errors


def test_offset_common_point_checks_both_moments_and_pair_power():
    center = np.array([1.875, 0, 1])
    contact = dict(
        point_world_m=[0.9375, 0, 1],
        link_surface_m=[1, 0, 1],
        sphere_surface_m=[0.875, 0, 1],
        gap_m=-0.125,
        link_com_world_m=[0, 1, 0],
        normal_world=[1, 0, 0],
        normal_load_N=2,
        force_on_payload_N=[2, -0.5, 0.25],
        force_on_link_N=[-2, 0.5, -0.25],
        torque_on_payload_Nm=[0, 0.234375, 0.46875],
        torque_on_link_Nm=[-0.25, -1.765625, -1.53125],
        relative_velocity_m_s=[-0.125, 0.5, -0.25],
        pair_power_W=-0.5625,
    )
    assert max(pair_errors(contact, center, 1, 0.4).values()) == 0
    contact["torque_on_link_Nm"] = [0, 0, 0]
    errors = pair_errors(contact, center, 1, 0.4)
    assert errors["link_moment_Nm"] > 2 and errors["pair_force_N"] == 0


def test_project_reaction_on_only_the_physical_ancestors():
    model = SimpleNamespace(
        dof_count=3, parents=[-1, 0, 1], joint_types=["revolute", "prismatic", "revolute"]
    )
    kin = SimpleNamespace(
        position=np.zeros((3, 3)), axis=np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    )
    p = np.array([0.9375, 0, 1])
    f = np.array([-2, 0.5, -0.25])
    np.testing.assert_array_equal(generalized_reaction(model, kin, 1, p, f), [0.46875, -2, 0])
    np.testing.assert_array_equal(generalized_reaction(model, kin, -1, p, f), [0, 0, 0])
    with pytest.raises(ValueError, match="invalid structure owner"):
        generalized_reaction(model, kin, 3, p, f)
