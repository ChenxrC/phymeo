"""Weighted common-point loads preserve stiffness, power, force and moment."""
from dataclasses import replace

import numpy as np
import pytest

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.geometry.collision_witness import CollisionWitness
from phymeo.geometry.contact_patch_adapter import ContactPatchSample, ContactSite
from phymeo.solvers.contact_patch_loads import sample_patch_loads


def patch(count):
    coordinates = {1: ((0, 0),), 2: ((-0.5, 0), (0.5, 0)),
                   4: ((-0.5, -0.1), (0.5, -0.1), (0.5, 0.1), (-0.5, 0.1))}
    sites = []
    for i, (x, y) in enumerate(coordinates[count]):
        witness = CollisionWitness(-0.01, (0, 0, 1), (x, y, 0.1), (x, y, 0.09),
                                   ((i, 1.0),), ((i + 10, 1.0),))
        sites.append(ContactSite(witness, (x, y, 0.095), 1 / count))
    return ContactPatchSample(tuple(sites), {1: 0, 2: 1, 4: 2}[count], "independent-oracle")


@pytest.mark.parametrize("count", (1, 2, 4))
def test_normal_stiffness_is_per_pair_and_bilateral(count):
    loads = sample_patch_loads(patch(count), RobotContactMaterial(100, 0, 0),
                               np.zeros((count, 3)), np.zeros((count, 3)))
    np.testing.assert_allclose(loads["first_force"].sum(axis=0), (0, 0, -1), atol=1e-12)
    np.testing.assert_array_equal(loads["second_force"], -loads["first_force"])
    origin = np.array((0.02, 0, 0))
    points = np.array([s.load_point for s in patch(count).sites])
    torque = np.cross(points - origin, loads["first_force"]).sum(axis=0)
    np.testing.assert_allclose(torque, (0, -0.02, 0), atol=1e-12)
    total = np.cross(points - origin, loads["first_force"] + loads["second_force"])
    np.testing.assert_array_equal(total, np.zeros((count, 3)))


def test_real_relative_speed_controls_damping_friction_and_power():
    sample = patch(2)
    v1 = np.tile((0.5, 0, 0.25), (2, 1))
    v2 = np.zeros((2, 3))
    loads = sample_patch_loads(sample, RobotContactMaterial(100, 4, 0.25), v1, v2)
    # Closing +z adds 4*.25=1 N to the spring's 1 N; sliding resists +x.
    np.testing.assert_allclose(loads["first_force"].sum(axis=0), (-0.5, 0, -2), atol=1e-12)
    power = np.sum(loads["first_force"] * v1 + loads["second_force"] * v2)
    assert power == pytest.approx(-0.75)
    common = np.tile((3, -2, 1), (2, 1))
    shifted = sample_patch_loads(sample, RobotContactMaterial(100, 4, 0.25), v1 + common, common)
    np.testing.assert_allclose(shifted["first_force"], loads["first_force"], atol=1e-12)


def test_invalid_weights_points_and_velocities_reject_before_loads():
    sample = patch(2)
    with pytest.raises(ValueError):
        replace(sample, sites=(replace(sample.sites[0], weight=0.9), sample.sites[1]))
    with pytest.raises(ValueError):
        replace(sample.sites[0], weight=-1)
    with pytest.raises(ValueError):
        replace(sample.sites[0], load_point=(10, 0, 0))
    with pytest.raises(ValueError):
        sample_patch_loads(sample, RobotContactMaterial(100, 0, 0),
                           np.array([[0, 0, float("nan")], [0, 0, 0]]), np.zeros((2, 3)))
