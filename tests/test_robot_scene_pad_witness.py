"""The material surface and a compliant sample's force application point differ."""

import numpy as np
import pytest

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_payload_scene import PadFace
from phymeo.runtime.robot_scene import RobotCommand


@pytest.mark.parametrize("kind", ["sphere", "rotated_box"])
def test_sampled_pad_keeps_true_payload_surface_separate_from_force_point(tmp_path, kind):
    path = tmp_path / "offset_pad.urdf"
    path.write_text(
        '<robot name="offset_pad"><link name="base"/><link name="pad_link">'
        '<inertial><mass value="2"/><inertia ixx=".01" iyy=".01" izz=".01" '
        'ixy="0" ixz="0" iyz="0"/></inertial>'
        '<collision name="pad"><geometry><box size=".02 .04 .06"/></geometry></collision>'
        '</link><joint name="slide" type="prismatic"><parent link="base"/>'
        '<child link="pad_link"/><origin xyz="-.01 0 .3"/><axis xyz="1 0 0"/>'
        '<limit lower="-.1" upper=".1" effort="20" velocity="1"/></joint></robot>',
        encoding="utf-8",
    )
    point, depth = np.array([0.0, 0.006, 0.304]), 0.001
    angle = 0.37
    R = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    normal = np.array([0.6, 0.8, 0.0]) if kind == "sphere" else R[:, 0]
    center = point + 0.019 * normal if kind == "sphere" else point - R @ [-0.019, 0.004, 0.003]
    b = ContactSceneBuilder(gravity=(0, 0, 0))
    b.add_urdf(
        10,
        path,
        robot_state=RobotState(np.zeros(1), np.zeros(1)),
        kp=[0],
        kd=[0],
        pads=[PadFace.named("pad_link", "pad", 0, 1, ((0.006, 0.004),))],
    )
    if kind == "sphere":
        b.add_sphere(20, 0.02, mass=0.1, position=center)
    else:
        b.add_box(
            20,
            (0.02, 0.015, 0.01),
            mass=0.1,
            position=center,
            orientation_xyzw=(0, 0, np.sin(angle / 2), np.cos(angle / 2)),
        )
    b.pair(10, 20, RobotContactMaterial(2000, 0, 0))
    with b.build() as scene:
        reading = scene.step(0.001, actions={10: RobotCommand([0], feedforward=[0])})
        assert len(reading.contacts) == 1
        row = reading.contacts[0]
        assert row["channel"] == "robot-pad"
        robot_key = scene.shape_key(10, "pad_link", "pad")
        payload_key = scene.body_keys[0, 20][0]
        if row["first"] == robot_key:
            pa, pb = row["first_surface_m"], row["second_surface_m"]
            force = row["force_on_second_N"]
            n = row["normal_world"]
        else:
            assert row["first"] == payload_key
            pa, pb = row["second_surface_m"], row["first_surface_m"]
            force = row["force_on_first_N"]
            n = -np.asarray(row["normal_world"])
        np.testing.assert_allclose(force, 2 * normal, atol=2e-13, rtol=0)
        np.testing.assert_allclose(n, normal, atol=1e-14, rtol=0)
        for side in ("first", "second"):
            np.testing.assert_allclose(row[f"application_point_{side}_m"], point, atol=1e-14)
        i = scene.rigid_rows[0, 20]
        expected_torque = np.cross(point - center, 2 * normal)
        np.testing.assert_allclose(
            reading.loads.inputs.rigid.torque[i], expected_torque, atol=1e-14
        )
        np.testing.assert_allclose(pa, point, atol=1e-14)
        # The failed implementation reported p twice even though gap=-1 mm.
        np.testing.assert_allclose(pb, point - depth * normal, atol=1e-13, rtol=0)
        assert float((np.asarray(pb) - pa) @ normal) == pytest.approx(row["gap_m"], abs=1e-13)
        if kind == "sphere":
            assert np.linalg.norm(np.asarray(pb) - center) == pytest.approx(0.02, abs=1e-13)
        else:
            local = R.T @ (np.asarray(pb) - center)
            np.testing.assert_allclose(local, [-0.02, 0.004, 0.003], atol=1e-13)
