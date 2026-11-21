from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from experiments.area_pressure.pusht_release_evidence import native_loads, native_release_poses


def sphere_floor():
    return mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="support" type="plane" size="1 1 .01"/>
      <body><joint name="height" type="slide" axis="0 0 1"/>
        <geom name="ball" type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>""")


def test_native_depth_velocity_and_strict_clearance_use_measured_pose():
    model = sphere_floor()
    for speed in [-0.8, 0.35]:
        pose = native_release_poses(model, [0.09], [speed], ((0, 1),))[(0, 1)]
        assert pose.depth == pytest.approx(0.01)
        assert pose.contact and not pose.separated
        assert pose.speed == pytest.approx(speed)
    clear = native_release_poses(model, [0.13], [-8.0], ((0, 1),))[(0, 1)]
    assert clear.depth == 0 and clear.separated and not clear.contact
    assert clear.speed is None


def test_native_force_aggregation_binds_the_order_of_both_real_receivers():
    pair = SimpleNamespace(
        geometry_ids=(1, 0),
        normal_force=0.6,
        force_on_geometry1_world=np.array([0.0, 0.0, 0.6]),
        force_on_geometry2_world=np.array([0.0, 0.0, -0.6]),
    )
    load, a, b = native_loads(((0, 1),), (pair, pair))[(0, 1)]
    assert load == 1.2
    np.testing.assert_array_equal(a, [0, 0, -1.2])
    np.testing.assert_array_equal(b, [0, 0, 1.2])
    with pytest.raises(ValueError):
        native_loads((), (pair,))
