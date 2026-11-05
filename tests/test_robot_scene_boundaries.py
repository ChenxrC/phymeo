"""Independent input, physical-frame and publication holdouts for robot scenes."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand
from phymeo.solvers.collision_loads import NormalLayer

PUSHER = Path(__file__).parent / "reference_newton" / "robot" / "multiobject_pusher.urdf"
LAW = RobotContactMaterial(2000, 0, 0)
H = 0.001


def declaration(
    *,
    path=PUSHER,
    q=0.0,
    v=0.0,
    center=(0, 0, 0.25),
    axis=(1, 0, 0),
    ids=(20, 30, 40),
    reverse=False,
    box=False,
):
    b = ContactSceneBuilder(gravity=(0, 0, 0))
    b.add_urdf(10, path, robot_state=RobotState(np.array([q]), np.array([v])), kp=[0], kd=[0])
    center, axis = np.asarray(center), np.asarray(axis)
    entries = [
        (ids[0], "sphere", center + (q + 0.124) * axis),
        (ids[1], "box" if box else "sphere", center + (q + 0.248) * axis),
        (ids[2], "table", center - [0, 0, 0.28125]),
    ]
    for key, kind, position in entries[::-1] if reverse else entries:
        if kind == "table":
            b.add_box(key, (0.5, 0.5, 0.03125), mass=None, position=position)
        elif kind == "sphere":
            b.add_sphere(key, 0.0625, mass=1, position=position, velocity=v * axis)
        else:
            b.add_box(key, (0.0625, 0.04, 0.03), mass=1, position=position, velocity=v * axis)
    for a, c in combinations(sorted((10, *ids)), 2):
        b.pair(a, c, LAW)
    return b


def test_external_com_force_torque_and_joint_feedforward_enter_the_same_step():
    with declaration().build() as scene:
        force = scene.inputs()
        body = scene.rigid_rows[0, 30]
        force.rigid.force[body] = [0, 3, 0]
        force.rigid.torque[body] = [0, 0, 0.01]
        out = scene.step(H, actions={10: RobotCommand([0], feedforward=[1])}, external=force)
        np.testing.assert_allclose(out.robots[0, 10].v, [-0.0005], atol=1e-14)
        np.testing.assert_allclose(out.robots[0, 10].drive_effort, [1], atol=1e-13)
        np.testing.assert_allclose(out.rigid.velocity[body], [0.002, 0.003, 0], atol=1e-14)
        # I=(2/5)*1*(1/16)^2=1/640 kg m^2; alpha_z=.01*640=6.4.
        np.testing.assert_allclose(out.rigid.angular_velocity[body], [0, 0, 0.0064], atol=1e-14)
        force.rigid.force[:] = 17
        assert out.external.rigid.force[body, 1] == 3
        np.testing.assert_allclose(out.loads.inputs.rigid.force[body], [2, 0, 0], atol=1e-12)


def test_joint_limit_rejects_every_participant_then_retry_matches_an_independent_session():
    settings = dict(q=0.099995, v=0.01)
    with declaration(**settings).build() as scene, declaration(**settings).build() as direct:
        before, geometry = scene.observe_scene()
        external = scene.inputs()
        row = scene.rigid_rows[0, 30]
        external.rigid.force[row, 1] = 7
        with pytest.raises(ValueError, match="joint position"):
            scene.step(
                H,
                actions={10: RobotCommand([0], effort_limit=[0], feedforward=[0])},
                external=external,
            )
        after, after_geometry = scene.observe_scene()
        assert after.step_number == before.step_number == 0 and after.time_s == 0
        assert scene.last_frame is None and after.load_time_s is None
        np.testing.assert_array_equal(after.rigid.position, before.rigid.position)
        np.testing.assert_array_equal(after.robots[0, 10].q, before.robots[0, 10].q)
        for key, pose in geometry.poses.items():
            np.testing.assert_array_equal(after_geometry.poses[key].start, pose.start)
        action = {10: RobotCommand([0], feedforward=[-20])}
        a, b = (
            scene.step(H, actions=action, external=external),
            direct.step(H, actions=action, external=external),
        )
        assert a.step_number == 1 and a.time_s == H and a.load_time_s == 0
        np.testing.assert_array_equal(a.rigid.position, b.rigid.position)
        np.testing.assert_array_equal(a.rigid.velocity, b.rigid.velocity)
        np.testing.assert_array_equal(a.robots[0, 10].q, b.robots[0, 10].q)
        np.testing.assert_allclose(a.robots[0, 10].v, [-0.001], atol=1e-13)


def test_public_arrays_own_their_data_and_samples_do_not_publish_or_advance():
    with declaration().build() as scene:
        _, initial = scene.observe_scene()
        sample = scene.sample()
        sample.inputs.rigid.force[:] = 123
        assert scene.observe().step_number == 0 and scene.geometry() is initial
        out = scene.step(H, actions={10: RobotCommand([0], feedforward=[0])})
        snapshot = scene.state
        snapshot.robots[0, 10].q[:] = 80
        snapshot.rigid.position[:] = 80
        out.loads.inputs.rigid.force[:] = 41
        out.external.rigid.force[:] = 42
        latest = scene.observe()
        row = scene.rigid_rows[0, 30]
        assert latest.loads.inputs.rigid.force[row, 0] == pytest.approx(2, abs=1e-12)
        np.testing.assert_array_equal(latest.external.rigid.force, 0)
        assert latest.robots[0, 10].q[0] == pytest.approx(-1e-6)
        with pytest.raises(ValueError):
            latest.rigid.position.setflags(write=True)
        with pytest.raises(ValueError):
            latest.robots[0, 10].v.setflags(write=True)
        with pytest.raises(TypeError):
            latest.contacts[0]["power_W"] = 100
        old_time = scene.observe().time_s
        scene.sample()
        scene.geometry()
        assert scene.observe().time_s == old_time


@pytest.mark.parametrize("box", [False, True])
def test_reordered_registration_and_original_local_ids_keep_physical_owners(box):
    with (
        declaration(ids=(1, 2, 3), box=box).build() as a,
        declaration(ids=(1, 2, 3), reverse=True, box=box).build() as b,
    ):
        for scene in (a, b):
            robot_key = scene.shape_key(10, "pusher", "round_tool")
            rigid_keys = {key for addr in ((0, 1), (0, 2), (0, 3)) for key in scene.body_keys[addr]}
            assert robot_key not in rigid_keys
            assert len(scene.geometry().poses) == 4
        action = {10: RobotCommand([0], feedforward=[0])}
        first, second = a.step(H, actions=action), b.step(H, actions=action)
        np.testing.assert_array_equal(first.rigid.position, second.rigid.position)
        np.testing.assert_array_equal(first.robots[0, 10].v, second.robots[0, 10].v)
        assert len(first.loads.contacts) == 2
        np.testing.assert_allclose(
            first.rigid.velocity[a.rigid_rows[0, 2]], [0.002, 0, 0], atol=1e-14
        )
        for pair in first.loads.contacts:
            assert pair.first in a.geometry().poses and pair.second in a.geometry().poses


def test_common_boost_and_oblique_slider_keep_the_hand_derived_three_body_impulse(tmp_path):
    path = tmp_path / "oblique.urdf"
    path.write_text(
        PUSHER.read_text()
        .replace('xyz="0 0 0.25"', 'xyz="3 -2 0.75"')
        .replace('xyz="1 0 0"', 'xyz="0.6 0.8 0"'),
        encoding="utf-8",
    )
    normal = np.array([0.6, 0.8, 0])
    with declaration(path=path, center=(3, -2, 0.75), axis=normal, v=0.02).build() as scene:
        out = scene.step(H, actions={10: RobotCommand([0], feedforward=[0])})
        first, second = (scene.rigid_rows[0, body] for body in (20, 30))
        np.testing.assert_allclose(out.robots[0, 10].v, [0.019], atol=2e-13)
        np.testing.assert_allclose(out.rigid.velocity[first], 0.02 * normal, atol=2e-13)
        np.testing.assert_allclose(out.rigid.velocity[second], 0.022 * normal, atol=2e-13)
        momentum = 2 * out.robots[0, 10].v[0] * normal + out.rigid.velocity[[first, second]].sum(
            axis=0
        )
        np.testing.assert_allclose(momentum, 0.08 * normal, atol=3e-13)
        np.testing.assert_allclose(out.rigid.angular_velocity, 0, atol=5e-13)


@pytest.mark.parametrize("backend", ["cuda", "interpret"])
def test_robot_backend_has_no_implicit_host_fallback(backend):
    with pytest.raises(ValueError, match="HOST"):
        declaration().build(backend=backend)


def test_unsupported_participants_and_material_semantics_are_explicit():
    b = declaration()
    b.add_sphere(99, 0.1, mass=1, position=(1, 1, 1), world=1)
    with pytest.raises(ValueError, match="one world"):
        b.build()
    b = declaration()
    b.add_ellipsoid(99, (0.1, 0.2, 0.3), mass=1, position=(2, 2, 2))
    with pytest.raises(ValueError, match="box/sphere"):
        b.build()
    b = ContactSceneBuilder(gravity=(0, 0, 0))
    b.add_urdf(10, PUSHER, robot_state=RobotState(np.zeros(1), np.zeros(1)), kp=[0], kd=[0])
    b.add_sphere(20, 0.0625, mass=1, position=(0.124, 0, 0.25))
    b.pair(10, 20, NormalLayer(0.01, 2000))
    with pytest.raises(TypeError, match="RobotContactMaterial"):
        b.build()


def test_unknown_controls_and_ambiguous_geometry_preserve_the_accepted_state():
    with declaration().build() as scene:
        with pytest.raises(ValueError, match="actions"):
            scene.step(H, actions={99: RobotCommand([0])})
        with pytest.raises(ValueError, match="rating"):
            scene.step(H, actions={10: RobotCommand([0], effort_limit=[21])})
        assert scene.observe().step_number == 0
    b = ContactSceneBuilder(gravity=(0, 0, 0))
    b.add_urdf(10, PUSHER, robot_state=RobotState(np.zeros(1), np.zeros(1)), kp=[0], kd=[0])
    b.add_sphere(20, 0.0625, mass=1, position=(0, 0, 0.25))
    b.pair(10, 20, LAW)
    with b.build() as scene:
        with pytest.raises(ValueError, match="unique normal"):
            scene.step(H, actions={10: RobotCommand([0])})
        assert scene.observe().step_number == 0 and scene.last_frame is None
