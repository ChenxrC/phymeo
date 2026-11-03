"""Physical OpenUSD stage units, anchored scalar joints and local collider ownership."""

from pathlib import Path

import numpy as np
import pytest

import phymeo as ph


ASSETS = Path(__file__).parent / "reference_newton" / "robot"


def load(path):
    from phymeo.builders.usd import load_usd

    return load_usd(path)


@pytest.mark.parametrize("filename", ["arm.usda", "arm_cm_y.usda"])
def test_usd_units_and_up_axis_give_the_same_physical_arm(filename):
    pytest.importorskip("pxr.Usd")
    asset = load(ASSETS / filename)
    assert isinstance(asset.model, ph.RobotModel)
    assert asset.root_link == "base"
    assert asset.link_names == ("boom", "tool") and asset.joint_names == ("yaw", "slide")
    np.testing.assert_allclose(asset.model.mass, [1, 0.5], atol=2e-7)
    np.testing.assert_allclose(asset.model.inertia_diagonal, [[0.1, 0.1, 0.1]] * 2, atol=2e-7)
    np.testing.assert_allclose(asset.model.com, [[0.25, 0, 0], [0.125, -0.0625, 0.03125]], atol=2e-7)
    np.testing.assert_allclose(asset.lower, [-np.pi / 2, -0.1], atol=2e-7)
    np.testing.assert_allclose(asset.upper, [2 * np.pi / 3, 0.25], atol=2e-7)
    marker, = asset.collisions
    assert (marker.link_index, marker.name, marker.shape.kind) == (1, "marker", "sphere")
    assert marker.shape.radius == pytest.approx(0.03125)
    for q, v, expected, speed in [
        ([np.pi / 2, 0.125], [0.5, -0.125], [0, 0.0625, 0.15625], [-0.15625, -0.125, 0]),
        ([-np.arctan2(4, 3), -0.0625], [-0.75, 0.0625], [0.6375, -0.9125, 0.15625],
         [-0.259375, -0.253125, 0]),
    ]:
        q, v = np.array(q), np.array(v)
        point = np.asarray(marker.local_vertices)
        port = ph.RobotPointPort(asset.model, ph.RobotState(q, v), [1], point)
        np.testing.assert_allclose(port.positions()[0], expected, rtol=0, atol=2e-7)
        np.testing.assert_allclose(port.velocities()[0], speed, rtol=0, atol=2e-7)
        np.testing.assert_allclose(asset.collision_motion(q)[marker.shape.key].start[0], expected, rtol=0, atol=2e-7)
