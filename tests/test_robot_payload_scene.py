"""A user commands the next physical step and can inspect both participants and the static table."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from phymeo.builders.panda import PANDA_URDF, load_panda
from phymeo.builders.urdf import load_urdf
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.rigid import RigidBodyState
from phymeo.core.robot import RobotState

ARM = Path(__file__).parent / "reference_newton" / "robot" / "arm.urdf"
TOOL = np.array([0.8125, -0.375, 0.15625])
READY = np.pi * np.array([0, -0.25, 0, -0.75, 0, 0.5, 0.25])
LAW = RobotContactMaterial(2000, 0.0, 0.0)


def body(center, orientation=(0, 0, 0, 1)):
    return RigidBodyState(
        np.array([center], float),
        np.array([orientation], float),
        np.zeros((1, 3)),
        np.zeros((1, 3)),
    )


def rail_scene(*, q=(0, 0), v=(0, 0), center=None, dt=0.001):
    from phymeo.runtime.robot_payload_scene import RobotPayloadScene

    center = TOOL + [0, 0.03125 + 0.020 - 0.001, 0] if center is None else center
    return RobotPayloadScene.from_urdf(
        ARM,
        loader=lambda path: load_urdf(path, gravity=(0, 0, 0)),
        robot_state=RobotState(np.array(q, float), np.array(v, float)),
        payload_state=body(center),
        payload_geometry={"kind": "box", "full_size_m": [0.08, 0.04, 0.06]},
        mass_kg=0.125,
        table_center_m=(0.8, -0.35, -0.12),
        table_half_extents_m=(0.55, 0.55, 0.02),
        dt_s=dt,
        kp=(30, 500),
        kd=(3, 8),
        structure_material=LAW,
    )


def test_joint_command_moves_both_owners_and_reports_the_real_input_wrench():
    plus, minus = rail_scene(), rail_scene()
    before = plus.geometry()
    for sign, scene in ((1, plus), (-1, minus)):
        first = scene.step([0, sign * 0.04])
        assert first.step_number == 1 and first.load_time_s == 0 and first.time_s == 0.001
        np.testing.assert_array_equal(first.target_q, [0, sign * 0.04])
        np.testing.assert_array_equal(first.effort_limit, [12, 8])
        assert len(first.structural_contacts) == 1
        row = first.structural_contacts[0]
        assert row["shape"] == "marker"
        np.testing.assert_allclose(row["force_on_payload_N"], [0, 2, 0], atol=1e-12)
        np.testing.assert_allclose(first.contact_effort, [-2 * 0.5625, -2], atol=1e-12)
        np.testing.assert_allclose(
            first.payload_velocity_m_s, [0, 0.001 * 2 / 0.125, 0], atol=1e-13
        )
        assert sign * first.drive_effort[1] == pytest.approx(8)
        assert sign * first.v[1] > 0
        for _ in range(31):
            scene.step([0, sign * 0.04])
    pressed, retreated = plus.observe(), minus.observe()
    assert pressed.q[1] > retreated.q[1] + 0.004
    assert pressed.payload_position_m[1] > retreated.payload_position_m[1] + 0.001
    assert retreated.structural_contacts == ()
    assert before.time_s == 0 and plus.geometry().time_s == pytest.approx(0.032)
    np.testing.assert_allclose(
        before.poses[before.payload_key].start.mean(axis=0), TOOL + [0, 0.05025, 0]
    )
    print(
        "rail",
        pressed.q.tolist(),
        retreated.q.tolist(),
        pressed.payload_position_m.tolist(),
        retreated.payload_position_m.tolist(),
    )


def test_dynamic_limit_rejection_does_not_publish_payload_robot_or_load_time_and_retry_matches():
    settings = dict(q=(0, 0.24965), v=(0, 0.08), center=(0.6, -0.22, 0.3), dt=0.005)
    scene, independent = rail_scene(**settings), rail_scene(**settings)
    before = scene.observe()
    geometry = scene.geometry()
    with pytest.raises(ValueError, match="joint position"):
        scene.step([0, 0.24], effort_limit=[0, 0])
    assert scene.observe() is before and before.load_time_s is None
    np.testing.assert_array_equal(
        scene.geometry().poses[geometry.payload_key].start,
        geometry.poses[geometry.payload_key].start,
    )
    with pytest.raises(ValueError, match="effort"):
        scene.step([0, 0.10], effort_limit=[12, 8.01])
    with pytest.raises(ValueError):
        before.q.setflags(write=True)
    for field in ("q", "v", "payload_position_m", "drive_effort"):
        a = scene.step([0, 0.10]) if field == "q" else scene.observe()
        b = independent.step([0, 0.10]) if field == "q" else independent.observe()
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert a.step_number == 1 and a.time_s == 0.005 and a.load_time_s == 0


def test_geometry_and_tilted_table_use_identical_current_collision_keys_and_surface():
    from phymeo.runtime.robot_payload_scene import RobotPayloadScene

    asset = load_urdf(ARM, gravity=(0, 0, 0))
    angle = np.pi / 6
    normal = np.array([0, -np.sin(angle), np.cos(angle)])
    center, half = np.array([0.55, -0.3, -0.2]), np.array([0.2, 0.15, 0.025])
    ball = center + normal * (half[2] + 0.02 - 0.001)
    scene = RobotPayloadScene(
        asset,
        robot_state=RobotState(np.zeros(2), np.zeros(2)),
        payload_state=body(ball),
        payload_geometry={"kind": "sphere", "radius_m": 0.02},
        mass_kg=0.1,
        table_center_m=center,
        table_half_extents_m=half,
        table_orientation_xyzw=(np.sin(angle / 2), 0, 0, np.cos(angle / 2)),
        dt_s=0.001,
        kp=(0, 0),
        kd=(0, 0),
        payload_table_material=LAW,
    )
    snapshot = scene.geometry()
    shapes = {s.key: s for s in snapshot.model.shapes}
    robot_key = asset.collisions[0].shape.key
    assert (
        set(snapshot.poses) == set(shapes) == {robot_key, snapshot.table_key, snapshot.payload_key}
    )
    assert snapshot.names[robot_key] == "marker"
    np.testing.assert_allclose(snapshot.poses[robot_key].start[0], TOOL)
    assert shapes[snapshot.payload_key].radius == 0.02 and shapes[snapshot.table_key].kind == "hull"
    np.testing.assert_array_equal(snapshot.poses[snapshot.payload_key].start[0], ball)
    np.testing.assert_allclose(
        np.max((snapshot.poses[snapshot.table_key].start - center) @ normal), half[2]
    )
    with pytest.raises(TypeError):
        snapshot.poses[robot_key] = None
    result = scene.step([0, 0])
    np.testing.assert_allclose(result.payload_table_wrench[:3], 2 * normal, atol=2e-13)
    np.testing.assert_allclose(result.fixed_table_wrench[:3], -2 * normal, atol=2e-13)
    np.testing.assert_allclose(result.payload_velocity_m_s, 0.02 * normal, atol=3e-14)
    np.testing.assert_allclose(
        scene.geometry().poses[snapshot.payload_key].start[0], result.payload_position_m
    )


def test_classic_panda_grip_effort_changes_next_step_and_measured_bilateral_contact():
    from phymeo.runtime.robot_payload_scene import PadFace, RobotPayloadScene

    asset = load_panda()
    q = asset.configuration(READY, 0.036)
    positions = [
        asset.authored_link_pose("panda_" + side + "pad", q)[0] for side in ("left", "right")
    ]
    center = np.mean(positions, axis=0)
    cols = {s.name: s for s in asset.collisions}
    pads = [
        PadFace(
            cols[f"panda_{name}finger_pad"].shape.key,
            1,
            sign,
            tuple(product((-0.005, 0.005), repeat=2)),
        )
        for name, sign in (("left", -1), ("right", 1))
    ]
    kwargs = dict(
        robot_state=RobotState(q, np.zeros(9)),
        payload_geometry={"kind": "sphere", "radius_m": 0.02},
        payload_state=body(center),
        mass_kg=0.04096,
        pads=pads,
        table_center_m=center - [0, 0, 0.04],
        table_half_extents_m=(0.09, 0.06, 0.02),
        dt_s=1 / 1600,
        kp=(500,) * 7 + (4500,) * 2,
        kd=(30,) * 7 + (18,) * 2,
    )
    squeezed = RobotPayloadScene.from_urdf(PANDA_URDF, loader=load_panda, **kwargs)
    released = RobotPayloadScene(asset, **kwargs)
    assert set(squeezed.geometry().names.values()) >= set(cols)
    for frame in range(48):
        for scene, width, cap in ((squeezed, 0.024, 7.0), (released, 0.068, 2.0)):
            reading = scene.step(
                asset.configuration(READY, width), effort_limit=(*asset.effort_limit[:7], cap, cap)
            )
            if frame == 0:
                np.testing.assert_allclose(
                    reading.pad_points_m.reshape(2, 4, 3).mean(axis=1), positions, atol=1e-13
                )
                assert np.all(reading.pad_normal_loads_N.reshape(2, 4).sum(axis=1) > 1.0)
                np.testing.assert_allclose(
                    reading.drive_effort[-2:], [-7, -7] if width < 0.036 else [2, 2], atol=1e-10
                )
                assert not any(
                    c["shape"] in {"panda_leftfinger_pad", "panda_rightfinger_pad"}
                    for c in reading.structural_contacts
                )
    a, b = squeezed.observe(), released.observe()
    assert np.min(a.pad_normal_loads_N.reshape(2, 4).sum(axis=1)) > 1
    assert np.all(b.pad_normal_loads_N == 0)
    assert sum(b.q[-2:]) > sum(a.q[-2:]) + 0.002
    print(
        "Panda",
        a.q[-2:].tolist(),
        b.q[-2:].tolist(),
        a.pad_normal_loads_N.reshape(2, 4).sum(axis=1).tolist(),
        b.pad_normal_loads_N.sum(),
    )


def test_invalid_face_outside_actual_box_and_duplicate_channels_fail_closed():
    from phymeo.runtime.robot_payload_scene import PadFace, RobotPayloadScene

    asset = load_panda()
    source = next(c for c in asset.collisions if c.name == "panda_leftfinger_pad")
    config = dict(
        robot_state=RobotState(asset.configuration(READY, 0.04), np.zeros(9)),
        payload_state=body([0.5, 0, 0.4]),
        payload_geometry={"kind": "sphere", "radius_m": 0.02},
        mass_kg=0.1,
        table_center_m=(0.5, 0, 0.05),
        table_half_extents_m=(0.2, 0.2, 0.02),
        dt_s=0.001,
        kp=np.ones(9),
        kd=np.ones(9),
    )
    with pytest.raises(ValueError, match="outside"):
        RobotPayloadScene(asset, pads=[PadFace(source.shape.key, 1, -1, [(0.02, 0)])], **config)
    valid = PadFace(source.shape.key, 1, -1, [(0, 0)])
    with pytest.raises(ValueError, match="duplicate"):
        RobotPayloadScene(asset, pads=[valid, valid], **config)


@pytest.mark.parametrize(
    "yaw,size", [("1.5707963267948966", "0.1 0.08 0.06"), ("0", "0.08 0.1 0.06")]
)
def test_free_box_can_load_a_fixed_root_cuboid_without_loading_a_distant_joint(tmp_path, yaw, size):
    from phymeo.runtime.robot_payload_scene import RobotPayloadScene

    path = tmp_path / "root-stop.urdf"
    source = ARM.read_text().replace(
        '<link name="base">',
        '<link name="base"><collision name="root_stop">'
        f'<origin xyz="0.18 0 0.2" rpy="0 0 {yaw}"/>'
        f'<geometry><box size="{size}"/></geometry></collision>',
    )
    path.write_text(source)
    asset = load_urdf(path, gravity=(0, 0, 0))
    scene = RobotPayloadScene(
        asset,
        robot_state=RobotState(np.zeros(2), np.zeros(2)),
        payload_state=body([0.18, 0.089, 0.2]),
        payload_geometry={"kind": "box", "full_size_m": [0.06, 0.08, 0.04]},
        mass_kg=0.1,
        table_center_m=(0.2, 0, -0.1),
        table_half_extents_m=(1, 1, 0.01),
        dt_s=0.001,
        kp=(0, 0),
        kd=(0, 0),
        structure_material=LAW,
    )
    sample = scene.step([0, 0])
    force = np.sum([r["force_on_payload_N"] for r in sample.structural_contacts], axis=0)
    assert {r["shape"] for r in sample.structural_contacts} == {"root_stop"}
    # The imported decimal quarter-turn leaves two distinct binary Y heights.
    # The exact Y support of the represented original vertices is then an edge.
    # The complementary unrotated asset has four co-supporting face corners.
    root_key = next(c.shape.key for c in asset.collisions if c.name == "root_stop")
    vertices = asset.collision_motion(np.zeros(2))[root_key].start
    supporting = vertices[vertices[:, 1] == vertices[:, 1].max()]
    dimension = np.linalg.matrix_rank(supporting - supporting[0])
    assert dimension == (2 if yaw == "0" else 1)
    assert {r["patch_dimension"] for r in sample.structural_contacts} == {dimension}
    print(
        "root y",
        yaw,
        [float(y).hex() for y in np.unique(vertices[vertices[:, 1] > 0, 1])],
        len(supporting),
    )
    assert all(
        "payload_material_weights" in r and "table_material_weights" not in r
        for r in sample.structural_contacts
    )
    np.testing.assert_allclose(force, [0, 2, 0], atol=3e-12)
    np.testing.assert_allclose(sample.contact_effort, [0, 0])
    np.testing.assert_allclose(sample.payload_velocity_m_s, [0, 0.02, 0], atol=4e-14)
