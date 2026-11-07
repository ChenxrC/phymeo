"""Weighted common-point loads preserve stiffness, power, force and moment."""

from dataclasses import replace

import numpy as np
import pytest

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.geometry.collision_witness import CollisionWitness
from phymeo.geometry.contact_patch_adapter import ContactPatchSample, ContactSite
from phymeo.solvers.contact_patch_loads import sample_patch_loads


def patch(count):
    coordinates = {
        1: ((0, 0),),
        2: ((-0.5, 0), (0.5, 0)),
        4: ((-0.5, -0.1), (0.5, -0.1), (0.5, 0.1), (-0.5, 0.1)),
    }
    sites = []
    for i, (x, y) in enumerate(coordinates[count]):
        witness = CollisionWitness(
            -0.01, (0, 0, 1), (x, y, 0.1), (x, y, 0.09), ((i, 1.0),), ((i + 10, 1.0),)
        )
        sites.append(ContactSite(witness, (x, y, 0.095), 1 / count))
    return ContactPatchSample(tuple(sites), {1: 0, 2: 1, 4: 2}[count], "independent-oracle")


@pytest.mark.parametrize("count", (1, 2, 4))
def test_normal_stiffness_is_per_pair_and_bilateral(count):
    loads = sample_patch_loads(
        patch(count), RobotContactMaterial(100, 0, 0), np.zeros((count, 3)), np.zeros((count, 3))
    )
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
        sample_patch_loads(
            sample,
            RobotContactMaterial(100, 0, 0),
            np.array([[0, 0, float("nan")], [0, 0, 0]]),
            np.zeros((2, 3)),
        )


def test_mutable_gap_input_cannot_change_frozen_contact_or_force():
    original = patch(1).sites[0]
    gap = np.array(-0.01)
    copied = ContactSite(replace(original.witness, gap_m=gap), original.load_point, 1.0)
    sample = ContactPatchSample((copied,), 0, "mutable-input")
    gap[...] = -1.0
    assert sample.sites[0].witness.gap_m == -0.01
    loads = sample_patch_loads(
        sample, RobotContactMaterial(100, 0, 0), np.zeros((1, 3)), np.zeros((1, 3))
    )
    np.testing.assert_allclose(loads["first_force"][0], (0, 0, -1), atol=1e-12)


@pytest.mark.parametrize("shift", (0.0, 1e6))
def test_world_translation_cannot_turn_separated_points_into_compression(shift):
    witness = CollisionWitness(
        -1e-5,
        (0, 0, 1),
        (shift, 0, 0),
        (shift, 0, 1e-5),
        ((1, 1.0),),
        ((2, 1.0),),
    )
    with pytest.raises(ValueError, match="gap"):
        ContactSite(witness, (shift, 0, 0.5e-5), 1.0)


def test_rotating_common_point_power_equals_both_com_wrench_powers():
    sample = patch(4)
    points = np.array([s.load_point for s in sample.sites])
    ca, cb = np.array((0.2, -0.3, 0.4)), np.array((-0.1, 0.2, -0.4))
    va, vb = np.array((0.4, -0.2, 0.3)), np.array((-0.1, 0.1, -0.2))
    wa, wb = np.array((0.7, -0.3, 0.9)), np.array((-0.4, 0.5, -0.6))
    ua = va + np.cross(wa, points - ca)
    ub = vb + np.cross(wb, points - cb)
    loads = sample_patch_loads(sample, RobotContactMaterial(100, 4, 0.25), ua, ub)
    fa, fb = loads["first_force"], loads["second_force"]
    site_power = np.sum(fa * ua + fb * ub)
    wrench_power = (
        fa.sum(axis=0) @ va
        + np.cross(points - ca, fa).sum(axis=0) @ wa
        + fb.sum(axis=0) @ vb
        + np.cross(points - cb, fb).sum(axis=0) @ wb
    )
    assert site_power == pytest.approx(wrench_power, abs=1e-12)
    # Distinct penetrated surface points give a different rotational work.
    wrong_a = va + np.cross(wa, np.array([s.witness.first_point for s in sample.sites]) - ca)
    wrong_b = vb + np.cross(wb, np.array([s.witness.second_point for s in sample.sites]) - cb)
    assert abs(np.sum(fa * wrong_a + fb * wrong_b) - site_power) > 1e-5
