"""A01-A06: several real URDF owners share one accepted physical step.

No load is injected in place of contact. The first oracle uses two independently
loaded 2 kg sliders and one 1 kg sphere, with 1 mm skin overlap at 2000 N/m.
Project execution belongs to the parent serial runner.
"""

from itertools import combinations

import numpy as np
import pytest

from phymeo.builders.mimic_urdf import load_mimic_urdf

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand

RADIUS = 1 / 16
LAW = RobotContactMaterial(2000.0, 0.0, 0.0)
H = 0.001


def _slider(path, *, mass=2.0, center=(0, 0, 0.25), axis=(1, 0, 0)):
    xyz = " ".join(str(float(value)) for value in center)
    direction = " ".join(str(float(value)) for value in axis)
    path.write_text(
        f'''<robot name="independent_slider">
<link name="base"/>
<link name="pusher">
  <inertial><mass value="{mass}"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
  </inertial>
  <collision name="round_tool"><geometry><sphere radius="{RADIUS}"/></geometry></collision>
</link>
<joint name="slide" type="prismatic">
  <parent link="base"/><child link="pusher"/>
  <origin xyz="{xyz}"/><axis xyz="{direction}"/>
  <limit lower="-0.1" upper="0.1" effort="128" velocity="10"/>
</joint>
</robot>
''',
        encoding="utf-8",
    )
    return path


def _symmetric_builder(tmp_path):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    for body, x in ((10, -0.124), (20, 0.124)):
        path = _slider(tmp_path / f"slider-{body}.urdf", center=(x, 0, 0.25))
        builder.add_urdf(
            body, path, robot_state=RobotState(np.zeros(1), np.zeros(1)), kp=[0], kd=[0]
        )
    builder.add_sphere(30, RADIUS, mass=1, position=(0, 0, 0.25))
    for first, second in combinations((10, 20, 30), 2):
        builder.pair(first, second, None if (first, second) == (10, 20) else LAW)
    return builder


def _zero_actions():
    return {body: RobotCommand([0], feedforward=[0]) for body in (10, 20)}


def test_a01_two_robots_share_one_sphere_and_one_actual_drift(tmp_path):
    with _symmetric_builder(tmp_path).build(backend="host") as scene:
        before = scene.observe()
        result = scene.step(H, actions=_zero_actions())
        left, right = (result.robots[0, body] for body in (10, 20))
        row = scene.rigid_rows[0, 30]
        assert before.step_number == 0 and before.time_s == 0
        assert result.step_number == 1 and result.time_s == H and result.load_time_s == 0
        assert set(result.robots) == {(0, 10), (0, 20)}
        np.testing.assert_allclose(left.contact_effort, [-2], atol=2e-12, rtol=0)
        np.testing.assert_allclose(right.contact_effort, [2], atol=2e-12, rtol=0)
        np.testing.assert_allclose(left.v, [-H], atol=2e-14, rtol=0)
        np.testing.assert_allclose(right.v, [H], atol=2e-14, rtol=0)
        np.testing.assert_allclose(left.q, [-H * H], atol=2e-16, rtol=0)
        np.testing.assert_allclose(right.q, [H * H], atol=2e-16, rtol=0)
        np.testing.assert_allclose(result.rigid.velocity[row], 0, atol=2e-14, rtol=0)
        np.testing.assert_allclose(result.rigid.position[row], [0, 0, 0.25], atol=2e-14)
        np.testing.assert_allclose(result.rigid.angular_velocity, 0, atol=2e-14, rtol=0)
        np.testing.assert_allclose(
            2 * (left.v[0] + right.v[0]) + result.rigid.velocity[row, 0],
            0,
            atol=2e-14,
            rtol=0,
        )
        assert len(result.loads.contacts) == 2
        np.testing.assert_allclose(
            [pair.force_n for pair in result.loads.contacts], [2, 2], atol=2e-12, rtol=0
        )
        reading, geometry = scene.observe_scene()
        assert reading.step_number == result.step_number and geometry.time_s == H
        assert len(geometry.poses) == 3
        assert set(geometry.poses) == {shape.key for shape in geometry.model.shapes}
        owners = {
            owner.body
            for record in scene.contact_events.active
            for owner in (record.first_owner, record.second_owner)
        }
        assert owners == {10, 20, 30}



def _asymmetric_builder(tmp_path, *, reverse=False, transformed=False, states=None):
    axis = np.array([0.6, 0, 0.8]) if transformed else np.array([1.0, 0, 0])
    center = np.array([3.0, -2.0, 0.75]) if transformed else np.array([0.0, 0, 0.25])
    b = ContactSceneBuilder(gravity=(0, 0, 0))
    states = {} if states is None else states
    entries = ((10, 2.0, -(2 * RADIUS - 0.0015)), (20, 4.0, 2 * RADIUS - 0.0005))
    for body, mass, distance in entries[::-1] if reverse else entries:
        path = _slider(tmp_path / f"asymmetric-{body}.urdf", mass=mass,
                       center=center + distance * axis, axis=axis)
        q, v = states.get(body, (0.0, 0.0))
        b.add_urdf(body, path, robot_state=RobotState(np.array([q]), np.array([v])),
                   kp=[0], kd=[0])
    b.add_sphere(30, RADIUS, mass=1, position=center)
    for a, c in combinations((10, 20, 30), 2):
        b.pair(a, c, None if (a, c) == (10, 20) else LAW)
    return b, center, axis


@pytest.mark.parametrize("reverse,transformed", [(False, False), (True, False), (True, True)])
def test_a02_unequal_masses_transform_and_registration_preserve_reciprocal_motion(
    tmp_path, reverse, transformed
):
    builder, center, axis = _asymmetric_builder(tmp_path, reverse=reverse, transformed=transformed)
    with builder.build() as scene:
        out = scene.step(H, actions=_zero_actions())
        a, b = out.robots[0, 10], out.robots[0, 20]
        row = scene.rigid_rows[0, 30]
        np.testing.assert_allclose(a.contact_effort, [-3], atol=2e-11, rtol=0)
        np.testing.assert_allclose(b.contact_effort, [1], atol=2e-11, rtol=0)
        np.testing.assert_allclose(a.v, [-1.5 * H], atol=2e-13, rtol=0)
        np.testing.assert_allclose(b.v, [0.25 * H], atol=2e-13, rtol=0)
        np.testing.assert_allclose(out.rigid.velocity[row], 2 * H * axis, atol=2e-13, rtol=0)
        np.testing.assert_allclose(out.rigid.position[row], center + 2 * H * H * axis,
                                   atol=4e-15, rtol=0)
        np.testing.assert_allclose(out.rigid.angular_velocity[row], 0, atol=2e-11, rtol=0)
        np.testing.assert_allclose((2 * a.v[0] + 4 * b.v[0]) * axis + out.rigid.velocity[row],
                                   0, atol=3e-13, rtol=0)
        assert len(scene.contact_events.active) == 2
        for record in scene.contact_events.active:
            owners = {record.first_owner.body, record.second_owner.body}
            assert owners in ({10, 30}, {20, 30})


@pytest.mark.parametrize("outside", [False, True])
def test_a03_free_sphere_table_load_is_sampled_once_and_finite(tmp_path, outside):
    b = ContactSceneBuilder(gravity=(0, 0, 0))
    for body, x in ((10, -3), (20, 3)):
        path = _slider(tmp_path / f"far-{body}.urdf", center=(x, 0, 0.25))
        b.add_urdf(body, path, robot_state=RobotState(np.zeros(1), np.zeros(1)), kp=[0], kd=[0])
    b.add_sphere(30, RADIUS, mass=1, position=(0.4 if outside else 0, 0, RADIUS - 0.001))
    b.add_box(40, (0.25, 0.25, 0.03125), mass=None, position=(0, 0, -0.03125))
    for first, second in combinations((10, 20, 30, 40), 2):
        b.pair(first, second, None if (first, second) == (10, 20) else LAW)
    with b.build() as scene:
        out = scene.step(H, actions=_zero_actions())
        ball, table = (scene.rigid_rows[0, body] for body in (30, 40))
        force = 0 if outside else 2.0
        np.testing.assert_allclose(out.loads.inputs.rigid.force[ball], [0, 0, force],
                                   atol=2e-12, rtol=0)
        np.testing.assert_allclose(out.loads.inputs.rigid.force[table], [0, 0, -force],
                                   atol=2e-12, rtol=0)
        np.testing.assert_allclose(out.rigid.velocity[ball], [0, 0, H * force],
                                   atol=2e-14, rtol=0)
        np.testing.assert_array_equal(out.rigid.velocity[table], 0)
        assert len(out.loads.contacts) == (0 if outside else 1)
        for robot in out.robots.values():
            np.testing.assert_array_equal(robot.q, 0)
            np.testing.assert_array_equal(robot.contact_effort, 0)


def _mimic_asset(path):
    # All three COM axes are independent +X sliders of mass 1 kg. Only the
    # positive follower carries geometry; the motor is on the driver.
    path.write_text(
        f'''<robot name="two_branch_transmission">
<link name="base"/>
<link name="driver"><inertial><mass value="1"/>
<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
</inertial></link>
<link name="follower"><inertial><mass value="1"/>
<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
</inertial><collision name="round_tool"><geometry><sphere radius="{RADIUS}"/></geometry></collision></link>
<link name="counter"><inertial><mass value="1"/>
<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
</inertial></link>
<joint name="drive" type="prismatic"><parent link="base"/><child link="driver"/>
<origin xyz="2 0 0.25"/><axis xyz="1 0 0"/>
<limit lower="-0.1" upper="0.1" effort="128" velocity="10"/></joint>
<joint name="contact_slide" type="prismatic"><parent link="base"/><child link="follower"/>
<origin xyz="0.121 0 0.25"/><axis xyz="1 0 0"/>
<limit lower="-0.1" upper="0.1" effort="128" velocity="10"/>
<mimic joint="drive" multiplier="1" offset="0.003"/></joint>
<joint name="counter_slide" type="prismatic"><parent link="base"/><child link="counter"/>
<origin xyz="3 0 0.25"/><axis xyz="1 0 0"/>
<limit lower="-0.1" upper="0.1" effort="128" velocity="10"/>
<mimic joint="drive" multiplier="-1" offset="-0.002"/></joint>
</robot>
''', encoding="utf-8")
    return path


def test_a04_ordinary_and_multiline_mimic_hold_then_actually_withdraw(tmp_path):
    b = ContactSceneBuilder(gravity=(0, 0, 0))
    b.add_urdf(10, _slider(tmp_path / "ordinary.urdf", center=(-0.124, 0, 0.25)),
               robot_state=RobotState(np.zeros(1), np.zeros(1)), kp=[0], kd=[0])
    b.add_urdf(20, _mimic_asset(tmp_path / "mimic.urdf"), loader=load_mimic_urdf,
               robot_state=RobotState(np.zeros(1), np.zeros(1)), kp=[0], kd=[0])
    b.add_sphere(30, RADIUS, mass=1, position=(0, 0, 0.25))
    for a, c in combinations((10, 20, 30), 2):
        b.pair(a, c, None if (a, c) == (10, 20) else LAW)
    with b.build() as scene:
        mechanism = scene.assets[0, 20]
        np.testing.assert_array_equal(mechanism.matrix, [[1], [1], [-1]])
        np.testing.assert_allclose(mechanism.offset_vector, [0, 0.003, -0.002], atol=1e-15)
        body = scene.rigid_rows[0, 30]
        unloaded = []
        for index in range(40):
            holding = index < 32
            dt = 1 / 8192 if holding else 1 / 256
            actions = {10: RobotCommand([0], feedforward=[2 if holding else -128]),
                       20: RobotCommand([0], feedforward=[-2 if holding else 128])}
            old = scene.observe()
            out = scene.step(dt, actions=actions)
            left, right = out.robots[0, 10], out.robots[0, 20]
            receipt = right.mimic_loads
            assert receipt is not None and len(receipt.row_forces) == 2
            assert receipt.multiplier_force is None
            np.testing.assert_allclose(
                2 * (left.v - old.robots[0, 10].v),
                dt * (left.drive_effort + left.contact_effort), atol=2e-12, rtol=0)
            physical_force = (receipt.actuator_effort_full + receipt.contact_effort_full
                              + receipt.constraint_effort_full)
            np.testing.assert_allclose(right.full_v - old.robots[0, 20].full_v,
                                       dt * physical_force, atol=2e-12, rtol=0)
            np.testing.assert_allclose(right.full_q,
                mechanism.matrix @ right.q + mechanism.offset_vector, atol=2e-14, rtol=0)
            assert abs(receipt.constraint_power_W) < 2e-11
            np.testing.assert_allclose(out.rigid.velocity[body] - old.rigid.velocity[body],
                dt * out.loads.inputs.rigid.force[body], atol=2e-12, rtol=0)
            # Independent diagonal-mass kinetic update; no continuum conservation
            # claim for the semiimplicit spring step or imposed motor work.
            v0 = old.robots[0, 20].full_v
            kinetic_change = 0.5 * (right.full_v @ right.full_v - v0 @ v0)
            discrete_work = dt * (physical_force @ v0) + 0.5 * dt * dt * (physical_force @ physical_force)
            assert abs(kinetic_change - discrete_work) < 2e-11
            if holding:
                np.testing.assert_allclose(left.contact_effort, [-2], atol=2e-10, rtol=0)
                np.testing.assert_allclose(right.contact_effort_full, [0, 2, 0], atol=2e-10, rtol=0)
                np.testing.assert_allclose(receipt.constraint_effort_full, [2, -2, 0], atol=2e-10, rtol=0)
                assert len(out.loads.contacts) == 2
            else:
                unloaded.append(len(out.loads.contacts))
        assert out.step_number == 40
        assert left.q[0] < -0.001 and right.q[0] > 0.001
        assert unloaded[-4:] == [0, 0, 0, 0]
        assert len(scene.contact_events.active) == 0
        assert left.v[0] < 0 and right.v[0] > 0


def _assert_unchanged(scene, old, geometry, events):
    after, actual = scene.observe_scene()
    assert after.step_number == old.step_number and after.time_s == old.time_s
    assert after.load_time_s == old.load_time_s and scene.contact_events is events
    np.testing.assert_array_equal(after.rigid.position, old.rigid.position)
    np.testing.assert_array_equal(after.rigid.velocity, old.rigid.velocity)
    np.testing.assert_array_equal(after.loads.inputs.rigid.force, old.loads.inputs.rigid.force)
    for key in old.robots:
        np.testing.assert_array_equal(after.robots[key].q, old.robots[key].q)
        np.testing.assert_array_equal(after.robots[key].v, old.robots[key].v)
    assert actual.time_s == geometry.time_s
    for key in geometry.poses:
        np.testing.assert_array_equal(actual.poses[key].start, geometry.poses[key].start)


def test_a05_second_robot_candidate_failure_rolls_back_and_retry_matches_fresh(tmp_path):
    states = {20: (0.099995, 0.01)}
    builder, _, _ = _asymmetric_builder(tmp_path, states=states)
    with builder.build() as scene, builder.build() as direct:
        old, geometry = scene.observe_scene()
        events = scene.contact_events
        external = scene.inputs()
        external.rigid.force[scene.rigid_rows[0, 30], 1] = 7
        with pytest.raises(ValueError, match="joint position"):
            scene.step(H, actions=_zero_actions(), external=external)
        _assert_unchanged(scene, old, geometry, events)
        assert scene.last_frame is None
        actions = _zero_actions()
        actions[20] = RobotCommand([0], feedforward=[-40])  # 4 kg: exactly stops +.01 m/s.
        a = scene.step(H, actions=actions, external=external)
        b = direct.step(H, actions=actions, external=external)
        for addr in a.robots:
            np.testing.assert_array_equal(a.robots[addr].q, b.robots[addr].q)
            np.testing.assert_array_equal(a.robots[addr].v, b.robots[addr].v)
        np.testing.assert_array_equal(a.rigid.position, b.rigid.position)
        np.testing.assert_array_equal(a.rigid.velocity, b.rigid.velocity)
        np.testing.assert_allclose(a.robots[0, 20].v, [0], atol=1e-14, rtol=0)


@pytest.mark.parametrize("invalid", ["missing", "unknown", "limit", "rating"])
def test_a05_all_commands_validate_before_any_publication(tmp_path, invalid):
    with _symmetric_builder(tmp_path).build() as scene:
        scene.step(H, actions=_zero_actions())
        old, geometry = scene.observe_scene()
        events = scene.contact_events
        actions = _zero_actions()
        if invalid == "missing":
            del actions[20]
        elif invalid == "unknown":
            actions[99] = RobotCommand([0])
        elif invalid == "limit":
            actions[20] = RobotCommand([0.2])
        else:
            actions[20] = RobotCommand([0], effort_limit=[129])
        with pytest.raises(ValueError):
            scene.step(H, actions=actions)
        _assert_unchanged(scene, old, geometry, events)


def test_a05_event_staging_failure_after_both_candidates_preserves_everything(tmp_path, monkeypatch):
    with _symmetric_builder(tmp_path).build() as scene:
        old, geometry = scene.observe_scene()
        events = scene.contact_events
        original = scene._event_adapter.stage
        def reject(*args, **kwargs):
            raise RuntimeError("event stage sentinel")
        monkeypatch.setattr(scene._event_adapter, "stage", reject)
        with pytest.raises(RuntimeError, match="event stage sentinel"):
            scene.step(H, actions=_zero_actions())
        _assert_unchanged(scene, old, geometry, events)
        monkeypatch.setattr(scene._event_adapter, "stage", original)
        assert scene.step(H, actions=_zero_actions()).step_number == 1


def test_a06_collision_ids_samples_and_owned_observations_do_not_alias(tmp_path):
    builder = _symmetric_builder(tmp_path)
    with builder.build() as scene, builder.build() as other:
        a = scene.shape_key(10, "pusher", "round_tool")
        b = scene.shape_key(20, "pusher", "round_tool")
        assert a != b
        assert set(scene.body_keys[0, 10]) == {a} and set(scene.body_keys[0, 20]) == {b}
        sample = scene.sample()
        assert len(sample.contacts) == 2
        sample.inputs.rigid.force[:] = 123
        assert scene.observe().step_number == 0 and scene.contact_events.step_index == 0
        out = scene.step(H, actions=_zero_actions())
        snapshot = scene.state
        snapshot.robots[0, 10].q[:] = 50
        snapshot.robots[0, 20].v[:] = 50
        snapshot.rigid.position[:] = 50
        out.external.rigid.force[:] = 50
        out.loads.inputs.rigid.force[:] = 50
        actual = scene.observe()
        np.testing.assert_allclose(actual.robots[0, 10].q, [-1e-6], atol=2e-16)
        np.testing.assert_allclose(actual.robots[0, 20].v, [0.001], atol=2e-14)
        np.testing.assert_allclose(actual.loads.inputs.rigid.force, 0, atol=2e-12)
        np.testing.assert_array_equal(actual.external.rigid.force, 0)
        assert other.observe().step_number == 0
        for addr in ((0, 10), (0, 20)):
            with pytest.raises(ValueError):
                actual.robots[addr].q.setflags(write=True)
            with pytest.raises(ValueError):
                scene.assets[addr].model.mass.setflags(write=True)
        assert scene.shape_key(10, "pusher", "round_tool") == a
        assert scene.shape_key(20, "pusher", "round_tool") == b


def test_a06_third_robot_is_independent_without_duplicating_shared_body_forces(tmp_path):
    b = _symmetric_builder(tmp_path)
    path = _slider(tmp_path / "third.urdf", mass=4, center=(0, 3, 0.25))
    b.add_urdf(40, path, robot_state=RobotState(np.zeros(1), np.zeros(1)), kp=[0], kd=[0])
    for body in (10, 20, 30):
        b.pair(body, 40, None if body in (10, 20) else LAW)
    with b.build() as scene:
        actions = _zero_actions()
        actions[40] = RobotCommand([0], feedforward=[4])
        out = scene.step(H, actions=actions)
        assert len(out.robots) == 3 and len(out.loads.contacts) == 2
        np.testing.assert_allclose(out.robots[0, 40].v, [H], atol=2e-14)
        np.testing.assert_allclose(out.rigid.velocity, 0, atol=2e-14)


@pytest.mark.parametrize("backend", ["interpret", "cuda"])
def test_a06_multi_robot_backend_is_not_silently_host(tmp_path, backend):
    with pytest.raises(ValueError, match="HOST"):
        _symmetric_builder(tmp_path).build(backend=backend)


def test_a06_direct_robot_contact_needs_a_future_explicit_capability(tmp_path):
    b = ContactSceneBuilder(gravity=(0, 0, 0))
    for body, x in ((10, 0), (20, 0.124)):
        b.add_urdf(body, _slider(tmp_path / f"direct-{body}.urdf", center=(x, 0, 0.25)),
                   robot_state=RobotState(np.zeros(1), np.zeros(1)), kp=[0], kd=[0])
    b.pair(10, 20, LAW)
    with pytest.raises(ValueError, match="direct robot"):
        b.build()


def test_a06_multirobot_coupled_history_is_explicitly_unavailable(tmp_path):
    with _symmetric_builder(tmp_path).build() as scene:
        with pytest.raises(ValueError, match="multi.robot.*tangent|tangent.*multi.robot"):
            scene.enable_coupled_tangent(None)
        assert scene.observe().step_number == 0
