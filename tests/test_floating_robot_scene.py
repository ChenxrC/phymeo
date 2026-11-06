"""Tests first: one physical floating root and joint tree in the shared HOST scene.

The public first RED uses only existing imports and a valid ordinary URDF before
calling the missing builder entry. No free/dummy body substitutes for the root.
"""

from itertools import combinations

import numpy as np
import pytest

from phymeo.builders.mimic_urdf import load_mimic_urdf
from phymeo.builders.mjcf import load_mjcf
from phymeo.builders.urdf import load_urdf
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.dynamics.floating_robot import FloatingRobotState, floating_kinematics
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand
from phymeo.solvers.floating_tree import FloatingRobotInputs, floating_dynamics_terms

RADIUS = 1 / 16
H = 0.001
FORCE_ATOL = 1e-10
STATE_ATOL = 1e-12
LAW = RobotContactMaterial(2000, 0, 0)


def _xyz(values):
    return " ".join(str(float(value)) for value in values)


def _sphere(name="root_sphere", center=(0, 0, 0)):
    return (
        f'<collision name="{name}"><origin xyz="{_xyz(center)}"/>'
        f'<geometry><sphere radius="{RADIUS}"/></geometry></collision>'
    )


def _inertia(mass, com, diagonal):
    return (
        f'<inertial><origin xyz="{_xyz(com)}"/><mass value="{mass}"/>'
        f'<inertia ixx="{diagonal[0]}" iyy="{diagonal[1]}" izz="{diagonal[2]}" '
        'ixy="0" ixz="0" iyz="0"/></inertial>'
    )


def _asset(
    path, *, root_mass=2, root_com=(0, 0, 0), root_inertia=(0.25, 0.5, 0.75),
    root_collision=None, child=True, child_origin=(0, 0, 0), child_com=(0, 0, 0),
    child_mass=1, axis=(1, 0, 0), child_collision="", lower=-0.1, upper=0.1,
):
    root_collision = _sphere() if root_collision is None else root_collision
    source = (
        '<robot name="floating_mechanism"><link name="base">'
        + _inertia(root_mass, root_com, root_inertia) + root_collision + '</link>'
    )
    if child:
        source += (
            '<link name="tool">' + _inertia(child_mass, child_com, (0.25, 0.25, 0.25))
            + child_collision + '</link><joint name="slide" type="prismatic">'
            '<parent link="base"/><child link="tool"/>'
            f'<origin xyz="{_xyz(child_origin)}"/><axis xyz="{_xyz(axis)}"/>'
            f'<limit lower="{lower}" upper="{upper}" effort="128" velocity="10"/>'
            '</joint>'
        )
    path.write_text(source + '</robot>\n', encoding="utf-8")
    return path


def _state(
    *, q=(0.0,), v=(0.0,), position=(0, 0, 0), xyzw=(0, 0, 0, 1),
    com_velocity=(0, 0, 0), omega=(0, 0, 0),
):
    return FloatingRobotState(
        np.array((*position, *xyzw, *q), dtype=float),
        np.array((*com_velocity, *omega, *v), dtype=float),
    )


def _add(builder, owner, path, state=None, *, kp=0, kd=0, cap=128, **kwargs):
    state = _state() if state is None else state
    n = len(state.v) - 6
    builder.add_floating_urdf(
        owner, path, robot_state=state, kp=[kp] * n, kd=[kd] * n,
        effort_limit=[cap] * n, **kwargs,
    )


def _command(effort=0.0, *, root_only=False, external=None, target=0.0):
    # Keep the real public missing-method RED before importing this NEW type.
    from phymeo.runtime._floating_robot_participant import FloatingRobotCommand

    joints = None if root_only else RobotCommand([target], feedforward=[effort])
    return FloatingRobotCommand(joints, external=external)


def test_a01_real_root_contact_accelerates_full_tree_and_free_body(tmp_path):
    path = _asset(tmp_path / "root-and-slide.urdf")
    old_asset = load_urdf(path, gravity=(0, 0, 0))
    assert old_asset.root_link == "base" and old_asset.joint_names == ("slide",)
    np.testing.assert_array_equal(old_asset.model.mass, [1])
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add(builder, 10, path)
    builder.add_sphere(20, RADIUS, mass=4, position=(0.124, 0, 0))
    builder.pair(10, 20, LAW)
    with builder.build(backend="host") as scene:
        assert scene.rigid_rows == {(0, 20): 0}
        assert scene.state.rigid.body_count == 1
        asset = scene.assets[0, 10]
        np.testing.assert_array_equal(asset.model.root.inverse_mass, [0.5])
        terms = floating_dynamics_terms(asset.model, _state(), FloatingRobotInputs.zeros(1))
        np.testing.assert_array_equal(terms.mass_matrix[np.ix_([0, 6], [0, 6])], [[3, 1], [1, 1]])
        out = scene.step(H, actions={10: _command()})
        row = out.robots[0, 10]
        assert len(row.q) == 8 and len(row.v) == 7 and row.link_wrenches_world.shape == (2, 6)
        np.testing.assert_allclose(row.acceleration_full, [-1, 0, 0, 0, 0, 0, 1], atol=STATE_ATOL)
        np.testing.assert_allclose(row.contact_effort_full, [-2, 0, 0, 0, 0, 0, 0], atol=FORCE_ATOL)
        np.testing.assert_allclose(row.applied_effort_full, row.contact_effort_full, atol=STATE_ATOL)
        np.testing.assert_allclose(row.v, [-H, 0, 0, 0, 0, 0, H], atol=STATE_ATOL)
        np.testing.assert_allclose(row.q, [-H * H, 0, 0, 0, 0, 0, 1, H * H], atol=STATE_ATOL)
        np.testing.assert_array_equal(row.full_q, row.q)
        np.testing.assert_array_equal(row.full_v, row.v)
        np.testing.assert_allclose(row.link_wrenches_world[0], [-2, 0, 0, 0, 0, 0], atol=FORCE_ATOL)
        np.testing.assert_array_equal(row.link_wrenches_world[1], np.zeros(6))
        measured = floating_kinematics(asset.model, FloatingRobotState(row.q, row.v))
        np.testing.assert_allclose(measured.com_position[1], [0, 0, 0], atol=STATE_ATOL)
        np.testing.assert_allclose(measured.com_velocity, [[-H, 0, 0], [0, 0, 0]], atol=STATE_ATOL)
        np.testing.assert_allclose(out.rigid.velocity[0], [H / 2, 0, 0], atol=STATE_ATOL)
        np.testing.assert_allclose(out.rigid.position[0], [0.124 + H * H / 2, 0, 0], atol=STATE_ATOL)
        np.testing.assert_allclose(
            2 * measured.com_velocity[0] + measured.com_velocity[1] + 4 * out.rigid.velocity[0],
            0, atol=STATE_ATOL,
        )
        assert out.step_number == 1 and out.load_time_s == 0 and out.time_s == H
        assert len(out.loads.contacts) == 1 and out.loads.contacts[0].force_n == pytest.approx(2, abs=FORCE_ATOL)
        event = scene.contact_events.active[0]
        root = next(owner for owner in (event.first_owner, event.second_owner) if owner.body == 10)
        assert (root.kind, root.local_frame, root.link_index, root.link_name) == (
            "robot_root", "robot_model_link", 0, "base",
        )


def _external(n, *, force=None, torque=None, effort=None):
    return FloatingRobotInputs(
        np.zeros(n) if effort is None else np.asarray(effort, dtype=float),
        np.zeros((n + 1, 3)) if force is None else np.asarray(force, dtype=float),
        np.zeros((n + 1, 3)) if torque is None else np.asarray(torque, dtype=float),
    )


def _assert_materials(scene, reading):
    events = scene.contact_events
    assert events.sample_time_s == reading.load_time_s
    assert events.commit_time_s == reading.time_s
    for record in events.active:
        for component in record.components:
            for site, addresses in zip(component.load.sites, component.material, strict=True):
                for owner, address, point in (
                    (record.first_owner, addresses.first, site.material.first_point),
                    (record.second_owner, addresses.second, site.material.second_point),
                ):
                    asset = scene.assets.get((owner.world, owner.body))
                    if asset is None or not hasattr(asset, "model") or not hasattr(asset.model, "root"):
                        continue
                    entry = reading.robots[owner.world, owner.body]
                    geometry = floating_kinematics(
                        asset.model, FloatingRobotState(entry.entry_q, entry.entry_v),
                    )
                    assert owner.local_frame == "robot_model_link"
                    assert owner.link_index >= 0
                    actual = geometry.position[owner.link_index] + geometry.rotation[owner.link_index] @ address.local_point_m
                    np.testing.assert_allclose(actual, point, atol=STATE_ATOL, rtol=0)
                    assert sum(weight for _, weight in address.weights) == pytest.approx(1, abs=STATE_ATOL)


@pytest.mark.parametrize("spinning", [False, True])
def test_a02_root_only_com_force_torque_and_eccentric_origin_are_distinct(tmp_path, spinning):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    path = _asset(tmp_path / "root-only.urdf", child=False, root_com=(1 / 8, 0, 0))
    initial = _state(q=(), v=(), omega=(0, 0, 2 if spinning else 0))
    _add(builder, 10, path, initial)
    applied = _external(0) if spinning else _external(0, force=[[2, 0, 0]], torque=[[0, 0, 0.75]])
    with builder.build() as scene:
        assert scene.state.rigid.body_count == 0
        assert scene.assets[0, 10].model.children is None
        before = floating_kinematics(scene.assets[0, 10].model, initial)
        np.testing.assert_allclose(before.origin_velocity[0], [0, -0.25 if spinning else 0, 0], atol=STATE_ATOL)
        out = scene.step(H, actions={10: _command(root_only=True, external=applied)})
        row = out.robots[0, 10]
        speed, angle = (2, 2 * H) if spinning else (H, H * H)
        dx = 0 if spinning else H * H
        expected_origin = [dx + (1 - np.cos(angle)) / 8, -np.sin(angle) / 8, 0]
        np.testing.assert_allclose(row.q[:3], expected_origin, atol=STATE_ATOL)
        np.testing.assert_allclose(row.q[3:7], [0, 0, np.sin(angle / 2), np.cos(angle / 2)], atol=STATE_ATOL)
        np.testing.assert_allclose(row.v, [0 if spinning else H, 0, 0, 0, 0, speed], atol=STATE_ATOL)
        np.testing.assert_allclose(row.applied_effort_full, np.r_[applied.force[0], applied.torque[0]], atol=STATE_ATOL)
        np.testing.assert_allclose(row.acceleration_full, np.zeros(6) if spinning else [1, 0, 0, 0, 0, 1], atol=STATE_ATOL)
        assert row.drive_effort.shape == (0,) and row.contact_effort_full.shape == (6,)
        np.testing.assert_array_equal(row.contact_effort_full, np.zeros(6))
        geometry = floating_kinematics(scene.assets[0, 10].model, FloatingRobotState(row.q, row.v))
        np.testing.assert_allclose(geometry.com_position[0], [1 / 8 + dx, 0, 0], atol=STATE_ATOL)
        key = scene.shape_key(10, "base", "root_sphere")
        np.testing.assert_allclose(scene.geometry().poses[key].start[0], expected_origin, atol=STATE_ATOL)


@pytest.mark.parametrize("drive,cap,expected", [(3, 128, 3), (-3, 128, -3), (3, 2, 2)])
def test_a02_internal_drive_reacts_on_root_and_is_limited_only_once(tmp_path, drive, cap, expected):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add(builder, 10, _asset(tmp_path / "driven.urdf"), cap=cap)
    with builder.build() as scene:
        row = scene.step(H, actions={10: _command(drive)}).robots[0, 10]
        np.testing.assert_allclose(row.drive_effort, [expected], atol=STATE_ATOL)
        np.testing.assert_allclose(row.applied_effort_full, [0, 0, 0, 0, 0, 0, expected], atol=STATE_ATOL)
        np.testing.assert_allclose(row.acceleration_full, [-expected / 2, 0, 0, 0, 0, 0, 1.5 * expected], atol=STATE_ATOL)
        observed = floating_kinematics(scene.assets[0, 10].model, FloatingRobotState(row.q, row.v))
        np.testing.assert_allclose(observed.com_velocity, [[-expected * H / 2, 0, 0], [expected * H, 0, 0]], atol=STATE_ATOL)
        np.testing.assert_allclose(2 * observed.com_velocity[0] + observed.com_velocity[1], 0, atol=STATE_ATOL)
        kinetic = float(observed.com_velocity[0] @ observed.com_velocity[0] + observed.com_velocity[1] @ observed.com_velocity[1] / 2)
        assert kinetic == pytest.approx(0.75 * expected**2 * H**2, abs=STATE_ATOL)


def test_a02_bounded_pd_uses_root_schur_mass_not_fixed_base_child_mass(tmp_path):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add(builder, 10, _asset(tmp_path / "servo.urdf"), kp=100)
    with builder.build() as scene:
        row = scene.step(H, actions={10: _command(target=0.01)}).robots[0, 10]
        # M_eff = 1 - 1/3 = 2/3 kg; u = 100*(.01 - h*h*u/M_eff).
        effort = 20000 / 20003
        np.testing.assert_allclose(row.drive_effort, [effort], atol=FORCE_ATOL)
        np.testing.assert_allclose(row.v, [-H * effort / 2, 0, 0, 0, 0, 0, 1.5 * H * effort], atol=STATE_ATOL)
        assert row.drive_effort[0] == pytest.approx(100 * (0.01 - row.q[7]), abs=FORCE_ATOL)


@pytest.mark.parametrize("loaded", [False, True])
def test_a03_world_gravity_and_declared_com_loads_share_the_full_mass_step(tmp_path, loaded):
    builder = ContactSceneBuilder(gravity=(0, 0, -1))
    _add(builder, 10, _asset(tmp_path / "external.urdf"))
    applied = _external(1, force=[[2, 0, 0], [-1, 0, 0]]) if loaded else _external(1)
    with builder.build() as scene:
        row = scene.step(H, actions={10: _command(3 if loaded else 0, external=applied)}).robots[0, 10]
        expected = [-0.5, 0, -1, 0, 0, 0, 2.5] if loaded else [0, 0, -1, 0, 0, 0, 0]
        np.testing.assert_allclose(row.acceleration_full, expected, atol=STATE_ATOL)
        np.testing.assert_allclose(row.v, H * np.asarray(expected), atol=STATE_ATOL)
        np.testing.assert_array_equal(row.external.force, applied.force)
        np.testing.assert_array_equal(row.external.torque, applied.torque)
        np.testing.assert_array_equal(row.external.effort, [0])
        np.testing.assert_allclose(row.applied_effort_full, [1, 0, 0, 0, 0, 0, 2] if loaded else np.zeros(7), atol=STATE_ATOL)
        obs = floating_kinematics(scene.assets[0, 10].model, FloatingRobotState(row.q, row.v))
        np.testing.assert_allclose(2 * obs.com_velocity[0] + obs.com_velocity[1], H * np.array([1 if loaded else 0, 0, -3]), atol=STATE_ATOL)


def test_a03_normal_damping_reads_real_root_and_rigid_entry_velocities(tmp_path):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add(builder, 10, _asset(tmp_path / "closing.urdf"), _state(com_velocity=(1 / 16, 0, 0)))
    builder.add_sphere(20, RADIUS, mass=4, position=(0.124, 0, 0), velocity=(-1 / 16, 0, 0))
    builder.pair(10, 20, RobotContactMaterial(2000, 8, 0))
    with builder.build() as scene:
        out = scene.step(H, actions={10: _command()})
        row = out.robots[0, 10]
        pair = out.loads.contacts[0]
        assert pair.force_n == pytest.approx(3, abs=FORCE_ATOL)
        assert pair.sites[0].rate_m_per_s == pytest.approx(-1 / 8, abs=STATE_ATOL)
        assert pair.sites[0].power_w == pytest.approx(-3 / 8, abs=STATE_ATOL)
        np.testing.assert_allclose(row.v, [1 / 16 - 1.5 * H, 0, 0, 0, 0, 0, 1.5 * H], atol=STATE_ATOL)
        np.testing.assert_allclose(out.rigid.velocity[0], [-1 / 16 + 0.75 * H, 0, 0], atol=STATE_ATOL)
        _assert_materials(scene, out)


@pytest.mark.parametrize("reverse", [False, True])
def test_a04_rotated_translated_boosted_root_has_true_material_frame(tmp_path, reverse):
    axis = np.array([0.6, 0, 0.8])
    angle = -np.arctan2(0.8, 0.6)
    quat = (0, np.sin(angle / 2), 0, np.cos(angle / 2))
    offset, boost = np.array([3, -2, 0.75]), np.array([0.125, -0.0625, 0.03125])
    initial = _state(position=offset, xyzw=quat, com_velocity=boost)
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    path = _asset(tmp_path / "rotated.urdf")
    for owner in (20, 10) if reverse else (10, 20):
        if owner == 10:
            _add(builder, owner, path, initial)
        else:
            builder.add_sphere(owner, RADIUS, mass=4, position=offset + 0.124 * axis, velocity=boost)
    builder.pair(10, 20, LAW)
    with builder.build() as scene:
        out = scene.step(H, actions={10: _command()})
        row = out.robots[0, 10]
        np.testing.assert_allclose(row.v[:3], boost - H * axis, atol=STATE_ATOL)
        np.testing.assert_allclose(row.q[:3], offset + H * boost - H * H * axis, atol=STATE_ATOL)
        np.testing.assert_allclose(row.q[3:7], quat, atol=STATE_ATOL)
        np.testing.assert_allclose(row.v[6:], [H], atol=STATE_ATOL)
        np.testing.assert_allclose(row.q[7:], [H * H], atol=STATE_ATOL)
        np.testing.assert_allclose(out.rigid.velocity[0], boost + H * axis / 2, atol=STATE_ATOL)
        obs = floating_kinematics(scene.assets[0, 10].model, FloatingRobotState(row.q, row.v))
        np.testing.assert_allclose(obs.com_position[1], offset + H * boost, atol=STATE_ATOL)
        np.testing.assert_allclose(2 * obs.com_velocity[0] + obs.com_velocity[1] + 4 * out.rigid.velocity[0], 7 * boost, atol=STATE_ATOL)
        _assert_materials(scene, out)


def _fixed(builder, owner, path, *, loader=None, q=0, v=0):
    builder.add_urdf(
        owner, path, loader=loader, robot_state=RobotState(np.array([q], dtype=float), np.array([v], dtype=float)),
        kp=[0], kd=[0], effort_limit=[128],
    )


def _mimic(path):
    source = (
        '<robot name="fixed_transmission"><link name="base"/>'
        '<link name="driver">' + _inertia(2, (0, 0, 0), (0.25, 0.25, 0.25)) + '</link>'
        '<link name="follower">' + _inertia(1, (0, 0, 0), (0.25, 0.25, 0.25)) + _sphere("tool_sphere") + '</link>'
        '<joint name="drive" type="prismatic"><parent link="base"/><child link="driver"/>'
        '<origin xyz="0 2 0"/><axis xyz="1 0 0"/><limit lower="-0.1" upper="0.1" effort="128" velocity="10"/></joint>'
        '<joint name="follow" type="prismatic"><parent link="base"/><child link="follower"/>'
        '<origin xyz="0.121 0 0"/><axis xyz="1 0 0"/><limit lower="-0.1" upper="0.1" effort="128" velocity="10"/>'
        '<mimic joint="drive" multiplier="-2" offset="0.003"/></joint></robot>\n'
    )
    path.write_text(source, encoding="utf-8")
    return path


@pytest.mark.parametrize("mimic", [False, True])
def test_a04_mixed_fixed_tree_and_independent_rigid_pair_apply_each_load_once(tmp_path, mimic):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add(builder, 10, _asset(tmp_path / "floating.urdf"))
    if mimic:
        _fixed(builder, 20, _mimic(tmp_path / "fixed.urdf"), loader=load_mimic_urdf)
    else:
        _fixed(builder, 20, _asset(tmp_path / "fixed.urdf", root_collision="", child_mass=2, child_origin=(0.124, 0, 0), child_collision=_sphere("tool_sphere")))
    builder.add_sphere(30, RADIUS, mass=1, position=(3, 0, RADIUS - 0.001))
    builder.add_box(40, (0.25, 0.25, 0.03125), mass=None, position=(3, 0, -0.03125))
    for a, b in combinations((10, 20, 30, 40), 2):
        builder.pair(a, b, LAW if (a, b) in ((10, 20), (30, 40)) else None)
    with builder.build() as scene:
        out = scene.step(H, actions={10: _command(), 20: RobotCommand([0], feedforward=[0])})
        floating, fixed = out.robots[0, 10], out.robots[0, 20]
        np.testing.assert_allclose(floating.v, [-H, 0, 0, 0, 0, 0, H], atol=STATE_ATOL)
        if mimic:
            np.testing.assert_allclose(fixed.contact_effort_full, [0, 2], atol=FORCE_ATOL)
            np.testing.assert_allclose(fixed.full_v, [-2 * H / 3, 4 * H / 3], atol=STATE_ATOL)
            np.testing.assert_allclose(fixed.mimic_loads.constraint_effort_full, [-4 / 3, -2 / 3], atol=FORCE_ATOL)
            assert fixed.mimic_loads.constraint_power_W == pytest.approx(0, abs=STATE_ATOL)
        else:
            np.testing.assert_allclose(fixed.contact_effort_full, [2], atol=FORCE_ATOL)
            np.testing.assert_allclose(fixed.v, [H], atol=STATE_ATOL)
        ball, table = scene.rigid_rows[0, 30], scene.rigid_rows[0, 40]
        np.testing.assert_allclose(out.loads.inputs.rigid.force[ball], [0, 0, 2], atol=FORCE_ATOL)
        np.testing.assert_allclose(out.loads.inputs.rigid.force[table], [0, 0, -2], atol=FORCE_ATOL)
        np.testing.assert_allclose(out.rigid.velocity[ball], [0, 0, 2 * H], atol=STATE_ATOL)
        np.testing.assert_array_equal(out.rigid.velocity[table], np.zeros(3))
        assert len(out.loads.contacts) == 2 and len(scene.contact_events.active) == 2
        assert {frozenset((r.first_owner.body, r.second_owner.body)) for r in scene.contact_events.active} == {frozenset((10, 20)), frozenset((30, 40))}
        _assert_materials(scene, out)


@pytest.mark.parametrize("child_contact,reverse", [(False, False), (True, False), (True, True)])
def test_a04_two_floating_owners_keep_root_and_child_dynamics_and_materials_distinct(tmp_path, child_contact, reverse):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    for owner in (20, 10) if reverse else (10, 20):
        if owner == 10:
            _add(builder, owner, _asset(tmp_path / "left.urdf"))
        elif child_contact:
            _add(builder, owner, _asset(tmp_path / "right.urdf", root_collision="", child_origin=(0.124, 0, 0), child_collision=_sphere("tool_sphere")))
        else:
            _add(builder, owner, _asset(tmp_path / "right.urdf"), _state(position=(0.124, 0, 0)))
    builder.pair(10, 20, LAW)
    with builder.build() as scene:
        assert scene.state.rigid.body_count == 0
        out = scene.step(H, actions={10: _command(), 20: _command()})
        left, right = out.robots[0, 10], out.robots[0, 20]
        expected = [0, 0, 0, 0, 0, 0, 2 * H] if child_contact else [H, 0, 0, 0, 0, 0, -H]
        np.testing.assert_allclose(left.v, [-H, 0, 0, 0, 0, 0, H], atol=STATE_ATOL)
        np.testing.assert_allclose(right.v, expected, atol=STATE_ATOL)
        np.testing.assert_allclose(right.contact_effort_full, [2, 0, 0, 0, 0, 0, 2 if child_contact else 0], atol=FORCE_ATOL)
        right_load = np.zeros((2, 6))
        right_load[int(child_contact), 0] = 2
        np.testing.assert_allclose(right.link_wrenches_world, right_load, atol=FORCE_ATOL)
        momentum = np.zeros(3)
        for owner, row in ((10, left), (20, right)):
            obs = floating_kinematics(scene.assets[0, owner].model, FloatingRobotState(row.q, row.v))
            momentum += 2 * obs.com_velocity[0] + obs.com_velocity[1]
        np.testing.assert_allclose(momentum, 0, atol=STATE_ATOL)
        event = scene.contact_events.active[0]
        side = next(owner for owner in (event.first_owner, event.second_owner) if owner.body == 20)
        assert (side.link_index, side.link_name) == ((1, "tool") if child_contact else (0, "base"))
        assert side.kind == ("robot_link" if child_contact else "robot_root")
        _assert_materials(scene, out)


def test_a05_forty_step_gravity_support_and_true_lift_use_continuous_state(tmp_path):
    builder = ContactSceneBuilder(gravity=(0, 0, -1))
    _add(
        builder, 10, _asset(tmp_path / "vertical.urdf", axis=(0, 0, 1)),
        _state(position=(0, 0, RADIUS - 0.0015)),
    )
    builder.add_box(20, (0.5, 0.5, 0.03125), mass=None, position=(0, 0, -0.03125))
    builder.pair(10, 20, LAW)
    first_events, end_seen, separated = None, False, []
    with builder.build() as scene:
        for step in range(40):
            holding = step < 32
            dt = 1 / 8192 if holding else 1 / 128
            external = _external(1) if holding else _external(1, force=[[0, 0, 16], [0, 0, 8]])
            drive = 1 if holding else 0
            before = scene.observe()
            old = before.robots[0, 10]
            out = scene.step(dt, actions={10: _command(drive, external=external)})
            row = out.robots[0, 10]
            np.testing.assert_array_equal(row.entry_q, old.q)
            np.testing.assert_array_equal(row.entry_v, old.v)
            assert out.load_time_s == before.time_s and out.step_number == step + 1
            normal = sum(pair.force_n for pair in out.loads.contacts)
            root_force = normal + external.force[0, 2] - 2 - drive
            child_force = external.force[1, 2] - 1 + drive
            old_speeds = np.array([old.v[2], old.v[2] + old.v[6]])
            speeds = np.array([row.v[2], row.v[2] + row.v[6]])
            actual_force = np.array([root_force, child_force])
            np.testing.assert_allclose(np.array([2, 1]) * (speeds - old_speeds), dt * actual_force, atol=STATE_ATOL, rtol=0)
            np.testing.assert_allclose(row.q[:3], old.q[:3] + dt * row.v[:3], atol=STATE_ATOL)
            np.testing.assert_allclose(row.q[7:], old.q[7:] + dt * row.v[6:], atol=STATE_ATOL)
            kinetic = float((np.array([2, 1]) * (speeds**2 - old_speeds**2)).sum() / 2)
            work = float(dt * (actual_force @ old_speeds) + dt**2 * ((actual_force**2 / np.array([2, 1])).sum()) / 2)
            assert kinetic == pytest.approx(work, abs=STATE_ATOL)
            np.testing.assert_allclose(out.loads.inputs.rigid.force[0], [0, 0, -normal], atol=FORCE_ATOL)
            np.testing.assert_array_equal(out.rigid.velocity[0], np.zeros(3))
            np.testing.assert_array_equal(row.external.force, external.force)
            if holding:
                assert normal == pytest.approx(3, abs=FORCE_ATOL)
                np.testing.assert_allclose(row.v, np.zeros(7), atol=STATE_ATOL)
                first_events = scene.contact_events if first_events is None else first_events
            else:
                separated.append(len(out.loads.contacts) == 0)
            end_seen |= any(event.kind == "end" for event in scene.contact_events.events)
            _assert_materials(scene, out)
        assert out.time_s == 32 / 8192 + 8 / 128
        assert row.q[2] > RADIUS + 0.001 and row.v[2] > 0
        assert all(separated[-4:]) and end_seen and not scene.contact_events.active
        assert first_events.step_index == 1 and len(first_events.active) == 1
        assert row.q[7] != 0 and row.v[6] != 0  # The unactuated child was not welded/reset.


def _assert_unchanged(scene, old, geometry, events):
    current, actual_geometry = scene.observe_scene()
    assert (current.step_number, current.time_s, current.load_time_s) == (old.step_number, old.time_s, old.load_time_s)
    assert scene.contact_events is events and actual_geometry.time_s == geometry.time_s
    for owner in old.robots:
        for field in ("q", "v", "full_q", "full_v", "drive_effort", "contact_effort_full", "link_wrenches_world"):
            np.testing.assert_array_equal(getattr(current.robots[owner], field), getattr(old.robots[owner], field))
    for field in ("position", "orientation", "velocity", "angular_velocity"):
        np.testing.assert_array_equal(getattr(current.rigid, field), getattr(old.rigid, field))
    np.testing.assert_array_equal(current.loads.inputs.rigid.force, old.loads.inputs.rigid.force)
    np.testing.assert_array_equal(current.loads.inputs.rigid.torque, old.loads.inputs.rigid.torque)
    assert current.contacts == old.contacts and current.loads.contacts == old.loads.contacts
    for key in geometry.poses:
        np.testing.assert_array_equal(actual_geometry.poses[key].start, geometry.poses[key].start)


def test_a06_second_floating_limit_failure_keeps_all_participants_then_retries(tmp_path):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add(builder, 10, _asset(tmp_path / "first.urdf"))
    _add(builder, 20, _asset(tmp_path / "second.urdf"), _state(q=(0.099995,), v=(0.01,), position=(1, 0, 0)))
    builder.add_sphere(30, RADIUS, mass=1, position=(3, 0, 0))
    for a, b in combinations((10, 20, 30), 2):
        builder.pair(a, b, None)
    with builder.build() as scene, builder.build() as fresh:
        old, geometry = scene.observe_scene()
        events = scene.contact_events
        external = scene.inputs()
        external.rigid.force[0, 1] = 7
        actions = {10: _command(external=_external(1, force=[[2, 0, 0], [0, 0, 0]])), 20: _command()}
        with pytest.raises(ValueError, match="joint position"):
            scene.step(H, actions=actions, external=external)
        _assert_unchanged(scene, old, geometry, events)
        assert scene.last_frame is None
        actions[20] = _command(-20)
        actual = scene.step(H, actions=actions, external=external)
        expected = fresh.step(H, actions=actions, external=external)
        for owner in actual.robots:
            np.testing.assert_array_equal(actual.robots[owner].q, expected.robots[owner].q)
            np.testing.assert_array_equal(actual.robots[owner].v, expected.robots[owner].v)
        np.testing.assert_array_equal(actual.rigid.position, expected.rigid.position)
        np.testing.assert_array_equal(actual.rigid.velocity, expected.rigid.velocity)
        assert scene.contact_events == fresh.contact_events


def test_a06_late_event_failure_preserves_dynamic_root_material_and_retry(tmp_path, monkeypatch):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add(builder, 10, _asset(tmp_path / "event.urdf"))
    builder.add_sphere(20, RADIUS, mass=4, position=(0.124, 0, 0))
    builder.pair(10, 20, LAW)
    with builder.build() as scene, builder.build() as fresh:
        scene.step(H, actions={10: _command()})
        fresh.step(H, actions={10: _command()})
        old, geometry = scene.observe_scene()
        events = scene.contact_events
        def fail(*args, **kwargs):
            raise RuntimeError("floating late event failure")
        with monkeypatch.context() as patcher:
            patcher.setattr(scene._event_adapter, "stage", fail)
            with pytest.raises(RuntimeError, match="floating late event failure"):
                scene.step(H, actions={10: _command()})
        _assert_unchanged(scene, old, geometry, events)
        actual = scene.step(H, actions={10: _command()})
        expected = fresh.step(H, actions={10: _command()})
        np.testing.assert_array_equal(actual.robots[0, 10].q, expected.robots[0, 10].q)
        np.testing.assert_array_equal(actual.rigid.position, expected.rigid.position)
        assert scene.contact_events == fresh.contact_events
        assert events.step_index == 1 and events.sample_time_s == 0
        _assert_materials(scene, actual)


@pytest.mark.parametrize("bad", ["effort", "body_count", "joint_target", "wrong_command", "unknown_owner"])
def test_a06_invalid_action_or_external_cannot_publish_a_partial_step(tmp_path, bad):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add(builder, 10, _asset(tmp_path / "invalid-action.urdf"))
    with builder.build() as scene:
        old, geometry = scene.observe_scene()
        events = scene.contact_events
        with pytest.raises((ValueError, TypeError)):
            if bad == "effort":
                command = _command(external=_external(1, effort=[1]))
            elif bad == "body_count":
                command = _command(external=FloatingRobotInputs.zeros(0))
            elif bad == "joint_target":
                command = _command(target=0.2)
            elif bad == "wrong_command":
                command = RobotCommand([0])
            else:
                command = _command()
            scene.step(H, actions={11 if bad == "unknown_owner" else 10: command})
        _assert_unchanged(scene, old, geometry, events)


def test_a06_coincident_root_geometry_failure_is_atomic(tmp_path):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add(builder, 10, _asset(tmp_path / "coincident.urdf"))
    builder.add_sphere(20, RADIUS, mass=4)
    builder.pair(10, 20, LAW)
    with builder.build() as scene:
        old, geometry = scene.observe_scene()
        events = scene.contact_events
        with pytest.raises(ValueError, match="normal|coincident"):
            scene.step(H, actions={10: _command()})
        _assert_unchanged(scene, old, geometry, events)
        assert scene.last_frame is None


@pytest.mark.parametrize("point,axis,force", [((0.124, 0, 0), (1, 0, 0), 2), ((0, 0, 0.374), (0, 0, 1), 2), ((0.124, 0, 0.375), (1, 0, 0), 0)])
def test_a04_floating_sphere_reads_finite_capsule_shaft_end_and_axis_miss(tmp_path, point, axis, force):
    path = tmp_path / "finite-capsule.xml"
    path.write_text(
        '<mujoco model="finite"><compiler angle="radian"/><worldbody><body name="fixed_base"><body name="rod">'
        f'<joint name="slide" type="slide" axis="{_xyz(axis)}" range="-1 1"/>'
        '<inertial pos="0 0 0" mass="2" diaginertia=".25 .25 .25"/>'
        f'<geom name="capsule" type="capsule" fromto="0 0 -.25 0 0 .25" size="{RADIUS}"/>'
        '</body></body></worldbody></mujoco>', encoding="utf-8",
    )
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add(builder, 10, _asset(tmp_path / "ball.urdf", child=False, root_mass=4), _state(q=(), v=(), position=point))
    _fixed(builder, 20, path, loader=load_mjcf)
    builder.pair(10, 20, LAW)
    with builder.build() as scene:
        out = scene.step(H, actions={10: _command(root_only=True), 20: RobotCommand([0], feedforward=[0])})
        np.testing.assert_allclose(out.robots[0, 10].v[:3], np.asarray(axis) * force * H / 4, atol=STATE_ATOL)
        np.testing.assert_allclose(out.robots[0, 10].v[3:6], 0, atol=STATE_ATOL)
        np.testing.assert_allclose(out.robots[0, 20].v, [-force * H / 2], atol=STATE_ATOL)
        assert sum(pair.force_n for pair in out.loads.contacts) == pytest.approx(force, abs=FORCE_ATOL)
        _assert_materials(scene, out)


@pytest.mark.parametrize("gap", [-0.001, 0.01])
def test_a04_floating_sphere_and_rotated_offset_box_keep_actual_face(tmp_path, gap):
    offset = np.array([0.125, -0.25, 0.0625])
    box = '<collision name="offset_box"><origin xyz=".125 -.25 .0625" rpy="0 0 1.5707963267948966"/><geometry><box size=".2 .4 .6"/></geometry></collision>'
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    # Local +X face becomes world +Y, offset relative to its moving link.
    position = offset + np.array([0, 0.1 + RADIUS + gap, 0])
    _add(builder, 10, _asset(tmp_path / "sphere.urdf", child=False, root_mass=4), _state(q=(), v=(), position=position))
    fixed = _asset(tmp_path / "obb.urdf", root_collision="", child_mass=2, axis=(0, 1, 0), child_collision=box)
    _fixed(builder, 20, fixed)
    builder.pair(10, 20, LAW)
    with builder.build() as scene:
        out = scene.step(H, actions={10: _command(root_only=True), 20: RobotCommand([0], feedforward=[0])})
        force = max(0, -2000 * gap)
        np.testing.assert_allclose(out.robots[0, 10].v[:3], [0, force * H / 4, 0], atol=STATE_ATOL)
        np.testing.assert_allclose(out.robots[0, 20].v, [-force * H / 2], atol=STATE_ATOL)
        np.testing.assert_allclose(out.robots[0, 20].link_wrenches_world[0], [0, -force, 0, force * offset[2], 0, -force * offset[0]], atol=FORCE_ATOL)
        _assert_materials(scene, out)


@pytest.mark.parametrize("invalid,match", [("missing", "inertial"), ("zero_mass", "positive"), ("products", "non-diagonal"), ("rotated_inertia", "non-diagonal"), ("fixed", "fixed"), ("mimic", "mimic")])
def test_a06_floating_loader_rejects_unrepresented_root_and_joint_semantics(tmp_path, invalid, match):
    from phymeo.builders.floating_urdf import load_floating_urdf

    path = _asset(tmp_path / "unsupported.urdf")
    source = path.read_text(encoding="utf-8")
    if invalid == "missing":
        source = source.replace(_inertia(2, (0, 0, 0), (0.25, 0.5, 0.75)), "", 1)
    elif invalid == "zero_mass":
        source = source.replace('<mass value="2"/>', '<mass value="0"/>', 1)
    elif invalid == "products":
        source = source.replace('ixy="0"', 'ixy="0.01"', 1)
    elif invalid == "rotated_inertia":
        source = source.replace('<origin xyz="0.0 0.0 0.0"/>', '<origin xyz="0 0 0" rpy="0 0 .3"/>', 1)
    elif invalid == "fixed":
        source = source.replace('type="prismatic"', 'type="fixed"')
    else:
        source = source.replace('</joint>', '<mimic joint="slide"/></joint>')
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_floating_urdf(path)


@pytest.mark.parametrize("restriction", ["pads", "cuda", "worlds", "loader", "state"])
def test_a06_scene_rejects_unimplemented_floating_modes(tmp_path, restriction):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    path = _asset(tmp_path / "scope.urdf")
    with pytest.raises((ValueError, TypeError)):
        if restriction == "pads":
            _add(builder, 10, path, pads=(object(),))
        elif restriction == "loader":
            _add(builder, 10, path, loader=lambda *args, **kwargs: object())
        elif restriction == "state":
            _add(builder, 10, path, _state(q=(), v=()))
        else:
            _add(builder, 10, path)
            if restriction == "worlds":
                _add(builder, 20, path, world=1)
            builder.build(backend="cuda" if restriction == "cuda" else "host")


def test_a06_public_asset_and_observation_snapshots_cannot_mutate_live_mass_or_inputs(tmp_path):
    import phymeo as ph
    from phymeo.builders.floating_urdf import FloatingRobotAsset, load_floating_urdf
    from phymeo.runtime._floating_robot_participant import FloatingRobotCommand

    assert ph.FloatingRobotAsset is FloatingRobotAsset
    assert ph.load_floating_urdf is load_floating_urdf
    assert ph.FloatingRobotCommand is FloatingRobotCommand
    path = _asset(tmp_path / "owned.urdf")
    asset = load_floating_urdf(path, gravity=(0, 0, 0))
    initial = _state()
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add(builder, 10, path, initial, loader=lambda *args, **kwargs: asset)
    # Loader and caller data are independent of both registrations/builds.
    asset.model.root.inverse_mass.setflags(write=True)
    asset.model.root.inverse_mass[0] = 1e-9
    initial.q.setflags(write=True)
    initial.q[0] = 100
    external = _external(1, force=[[2, 0, 0], [0, 0, 0]])
    command = _command(external=external)
    external.force.setflags(write=True)
    external.force[0, 0] = 999
    with builder.build() as scene, builder.build() as fresh:
        for array in (scene.assets[0, 10].model.root.inverse_mass, scene.assets[0, 10].model.children.mass):
            with pytest.raises(ValueError):
                array.setflags(write=True)
        out = scene.step(H, actions={10: command})
        expected = fresh.step(H, actions={10: command})
        np.testing.assert_array_equal(out.robots[0, 10].q, expected.robots[0, 10].q)
        np.testing.assert_allclose(out.robots[0, 10].v, [H, 0, 0, 0, 0, 0, -H], atol=STATE_ATOL)
        np.testing.assert_array_equal(out.robots[0, 10].external.force, [[2, 0, 0], [0, 0, 0]])
        for array in (out.robots[0, 10].q, out.robots[0, 10].external.force, out.robots[0, 10].acceleration_full):
            with pytest.raises(ValueError):
                array.setflags(write=True)
        snapshot = scene.state
        snapshot.robots[0, 10].q.setflags(write=True)
        snapshot.robots[0, 10].q[0] = -100
        assert scene.observe().robots[0, 10].q[0] == pytest.approx(H * H, abs=STATE_ATOL)
