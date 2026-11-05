"""Robot events retain physical owner names and all same-input contact components."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand


PUSHER = Path(__file__).parent / "reference_newton" / "robot" / "multiobject_pusher.urdf"
LAW = RobotContactMaterial(2000.0, 0.0, 0.0)


def three_body_chain():
    make = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    make.add_urdf(
        10, PUSHER, robot_state=RobotState(np.zeros(1), np.zeros(1)),
        kp=[0.0], kd=[0.0],
    )
    make.add_sphere(20, 0.0625, mass=1.0, position=(0.124, 0.0, 0.25))
    make.add_sphere(30, 0.0625, mass=1.0, position=(0.248, 0.0, 0.25))
    make.add_box(40, (0.5, 0.5, 0.03125), mass=None, position=(0.0, 0.0, -0.03125))
    for a, b in combinations((10, 20, 30, 40), 2):
        make.pair(a, b, LAW)
    return make


def test_robot_scene_publishes_both_real_chain_contacts_through_the_common_event_view():
    with three_body_chain().build() as scene:
        command = {10: RobotCommand([0.0], feedforward=[0.0])}
        frame = scene.step(0.001, actions=command)
        np.testing.assert_allclose(frame.robots[0, 10].v, [-0.001], atol=1e-14, rtol=0)
        assert len(frame.loads.contacts) == 2
        np.testing.assert_allclose([p.force_n for p in frame.loads.contacts], [2.0, 2.0], atol=1e-12)
        # The RED is the public property absent on the already working robot route.
        view = scene.contact_events
        from phymeo.runtime.contact_events import SceneContactEvents

        assert isinstance(view, SceneContactEvents)
        assert view.step_index == frame.step_number == 1
        assert view.sample_time_s == frame.load_time_s == 0.0
        assert view.commit_time_s == frame.time_s == 0.001
        assert len(view.active) == 2
        assert [e.kind for e in view.events] == ["begin", "begin"]
        assert sum(len(r.components) for r in view.active) == 2
        physical_pairs = {
            (r.first_owner.body, r.second_owner.body): r for r in view.active
        }
        assert set(physical_pairs) == {(10, 20), (20, 30)}
        link = physical_pairs[10, 20].first_owner
        assert link.kind == "robot_link" and link.link_name == "pusher" and link.link_index == 0
        assert link.local_frame == "robot_model_link"
        for record in view.active:
            (component,) = record.components
            assert component.material_kind == "robot_contact_material"
            assert component.law == LAW
            assert component.load in frame.loads.contacts
            assert record.identity.generation == 1
            assert record.first_owner.world == record.second_owner.world == 0
        scene.sample()
        scene.observe()
        scene.geometry()
        scene.observe_scene()
        assert scene.contact_events == view
        scene.step(0.001, actions=command)
        assert [e.kind for e in scene.contact_events.events] == ["persist", "persist"]
        assert [r.identity for r in scene.contact_events.active] == [r.identity for r in view.active]



def _twisting_pad_scene(tmp_path, *, spin=2.0, transformed=False, reverse=False):
    from phymeo.runtime.robot_payload_scene import PadFace

    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) if transformed else np.eye(3)
    shift = np.array([2.0, -3.0, 1.0]) if transformed else np.zeros(3)
    angle = np.pi / 2 if transformed else 0.0
    origin = shift + R @ [-0.01, 0.0, 0.3]
    path = tmp_path / f"twisting_pad_{transformed}_{reverse}.urdf"
    path.write_text(
        '<robot name="twisting_pad"><link name="base"/><link name="pad_link">'
        '<inertial><mass value="2"/><inertia ixx=".01" iyy=".01" izz=".01" '
        'ixy="0" ixz="0" iyz="0"/></inertial>'
        '<collision name="pad"><geometry><box size=".02 .04 .06"/></geometry></collision>'
        '</link><joint name="slide" type="prismatic"><parent link="base"/>'
        f'<child link="pad_link"/><origin xyz="{origin[0]} {origin[1]} {origin[2]}" '
        f'rpy="0 0 {angle}"/><axis xyz="1 0 0"/>'
        '<limit lower="-.1" upper=".1" effort="20" velocity="1"/></joint></robot>',
        encoding="utf-8",
    )
    offsets = ((0.009, 0.0), (-0.009, 0.0))
    if reverse:
        offsets = offsets[::-1]
    make = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    make.add_urdf(
        10, path, robot_state=RobotState(np.zeros(1), np.zeros(1)), kp=[0.0], kd=[0.0],
        pads=[PadFace.named("pad_link", "pad", 0, 1, offsets)],
    )
    make.add_sphere(
        20, 0.02, mass=0.1, position=shift + R @ [0.015, 0.0, 0.3],
        orientation_xyzw=(0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)),
        angular_velocity=R @ [spin, 0.0, 0.0],
    )
    law = RobotContactMaterial(2000.0, 0.0, 0.5, 0.01)
    make.pair(10, 20, law)
    return make, law, R, shift


@pytest.mark.parametrize("spin", [2.0, -2.0, 0.0])
@pytest.mark.parametrize("transformed", [False, True])
def test_multiple_pad_components_keep_nonparallel_normals_and_torsional_friction(
    tmp_path, spin, transformed
):
    make, law, R, shift = _twisting_pad_scene(tmp_path, spin=spin, transformed=transformed)
    with make.build() as scene:
        initial = scene.contact_events
        h = 1 / 10000
        frame = scene.step(h, actions={10: RobotCommand([0.0], feedforward=[0.0])})
        view = scene.contact_events
        (record,) = view.active
        assert len(frame.loads.contacts) == len(record.components) == 2
        assert tuple(c.load for c in record.components) == frame.loads.contacts
        assert view.events[0].record is record and record.identity.generation == 1
        assert record.first_owner.body == 10 and record.second_owner.body == 20
        assert record.first_owner.link_name == "pad_link"
        assert initial.active == initial.events == ()
        n = 2000.0 * (0.02 - np.hypot(0.015, 0.009))
        tangent_torque = -np.sign(spin) * 2 * 0.009 * 0.5 * n
        center = shift + R @ [0.015, 0.0, 0.3]
        forces, torques, tangents = [], [], []
        for c in record.components:
            assert c.law is law and c.material_kind == "robot_contact_material"
            assert c.load.force_n == pytest.approx(n, abs=5e-12)
            assert c.load.gap_m == pytest.approx(-n / 2000, abs=1e-14)
            assert len(c.load.sites) == len(c.material) == 1
            site, address = c.load.sites[0], c.material[0]
            p = np.asarray(site.application_point_second_m)
            force = np.asarray(site.force_on_second_n)
            forces.append(force)
            torques.append(np.cross(p - center, force))
            tangents.append(force - c.load.force_n * np.asarray(c.load.normal))
            assert address.first.kind == "robot_link" and address.second.kind == "rigid"
            np.testing.assert_allclose(
                address.first.local_point_m,
                [0.01, (R.T @ (p - shift))[1], 0.0], atol=5e-14,
            )
            np.testing.assert_allclose(
                center + R @ address.second.local_point_m, site.material.second_point, atol=5e-14,
            )
            assert np.linalg.norm(address.second.local_point_m) == pytest.approx(0.02, abs=5e-14)
            # The actual sphere surface is not its penetrating pad application point.
            assert np.linalg.norm(np.asarray(site.material.second_point) - p) == pytest.approx(n / 2000)
        assert not np.allclose(record.components[0].load.normal, record.components[1].load.normal)
        np.testing.assert_allclose(np.sum(tangents, axis=0), 0.0, atol=2e-12)
        expected_force = R @ [2 * n * 0.015 / np.hypot(0.015, 0.009), 0.0, 0.0]
        expected_torque = R @ [tangent_torque, 0.0, 0.0]
        np.testing.assert_allclose(np.sum(forces, axis=0), expected_force, atol=2e-12)
        np.testing.assert_allclose(np.sum(torques, axis=0), expected_torque, atol=2e-12)
        row = scene.rigid_rows[0, 20]
        np.testing.assert_allclose(frame.loads.inputs.rigid.force[row], expected_force, atol=2e-12)
        np.testing.assert_allclose(frame.loads.inputs.rigid.torque[row], expected_torque, atol=2e-12)
        assert sum(c.load.power_w for c in record.components) == pytest.approx(tangent_torque * spin, abs=2e-12)
        inertia = (2 / 5) * 0.1 * 0.02**2
        np.testing.assert_allclose(
            frame.rigid.angular_velocity[row], R @ [spin + h * tangent_torque / inertia, 0, 0], atol=2e-12,
        )
        # The two real masses react once, with no averaged-normal replacement.
        np.testing.assert_allclose(
            0.1 * frame.rigid.velocity[row] + 2 * frame.robots[0, 10].v[0] * R[:, 0], 0.0, atol=1e-13,
        )
        frame.loads.inputs.rigid.force[:] = 123.0
        assert scene.contact_events is view
        assert sum(c.load.force_n for c in record.components) == pytest.approx(2 * n)


def test_pad_enumeration_does_not_create_component_identity_or_drop_a_site(tmp_path):
    make, _, _, _ = _twisting_pad_scene(tmp_path)
    reverse, _, _, _ = _twisting_pad_scene(tmp_path, reverse=True)
    with make.build() as a, reverse.build() as b:
        command = {10: RobotCommand([0.0], feedforward=[0.0])}
        ra, rb = a.step(1 / 10000, actions=command), b.step(1 / 10000, actions=command)
        ca, cb = a.contact_events.active[0], b.contact_events.active[0]
        assert ca.identity == cb.identity
        assert ca.components == cb.components[::-1]
        np.testing.assert_allclose(ra.rigid.velocity, rb.rigid.velocity, atol=1e-14)
        np.testing.assert_allclose(ra.rigid.angular_velocity, rb.rigid.angular_velocity, atol=1e-14)


def test_robot_finite_table_reentry_retains_penetration_law_and_sample_times():
    make = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    make.add_urdf(10, PUSHER, robot_state=RobotState(np.zeros(1), np.zeros(1)), kp=[0], kd=[0])
    make.add_box(20, (1 / 4, 1 / 4, 1 / 8), mass=None, position=(3.0, 0.0, 0.0))
    make.add_sphere(30, 1 / 8, mass=1.0, position=(3.0, 0.0, 0.248))
    for pair in combinations((10, 20, 30), 2):
        make.pair(*pair, LAW)
    with make.build() as scene:
        command = {10: RobotCommand([0.0], feedforward=[0.0])}
        h, row = 1 / 64, scene.rigid_rows[0, 30]
        views, frames = [], []
        for force in (252.0, -512.0, 252.0):
            external = scene.inputs()
            external.rigid.force[row, 2] = force
            frames.append(scene.step(h, external, actions=command))
            views.append(scene.contact_events)
        assert [v.events[0].kind for v in views] == ["begin", "end", "begin"]
        assert views[1].active == () and views[1].events[0].reason == "absent"
        assert views[1].events[0].record is None
        assert [v.sample_time_s for v in views] == [0.0, h, 2 * h]
        assert [v.commit_time_s for v in views] == [h, 2 * h, 3 * h]
        assert len(views[2].generations) == 1 and views[2].active[0].identity.generation == 2
        for i in (0, 2):
            record = views[i].active[0]
            assert (record.first_owner.body, record.second_owner.body) == (20, 30)
            c = record.components[0]
            assert c.material_kind == "robot_contact_material" and c.law is LAW
            assert c.load.force_n == pytest.approx(4.0, abs=1e-11)
            assert c.load.gap_m == pytest.approx(-0.002, abs=1e-14)
        np.testing.assert_allclose([f.rigid.velocity[row, 2] for f in frames], [4, -4, 0], atol=1e-12)
        np.testing.assert_allclose([f.rigid.position[row, 2] for f in frames], [0.3105, 0.248, 0.248], atol=1e-13)
        for frame in frames:
            np.testing.assert_array_equal(frame.robots[0, 10].q, [0.0])


def _stage_inputs(scene, frame):
    from phymeo.dynamics.robot_kinematics import forward_kinematics

    return dict(
        geo=forward_kinematics(scene.assets[0, 10].model, frame.robots[0, 10].entry_q),
        rigid=frame.entry_rigid, sample_time_s=frame.load_time_s, commit_time_s=frame.time_s,
    )


def test_pure_robot_stage_preserves_zero_components_reorders_and_bounded_tombstones():
    from dataclasses import replace

    from phymeo.runtime.contact_events import SceneContactEvents

    with three_body_chain().build() as scene:
        frame = scene.step(0.001, actions={10: RobotCommand([0.0], feedforward=[0.0])})
        adapter, initial = scene._event_adapter, SceneContactEvents()
        first, second = frame.loads.contacts
        zero_site = replace(first.sites[0], force_n=0.0, force_on_second_n=(0.0, 0.0, 0.0), power_w=0.0)
        zero = replace(first, force_n=0.0, power_w=0.0, sites=(zero_site,))
        raw = replace(frame.loads, contacts=(zero, first, second))
        kwargs = _stage_inputs(scene, frame)
        before = adapter.stage(initial, raw, **kwargs)
        assert len(before.active) == 2 and len(before.active[0].components) == 2
        assert [c.load.force_n for c in before.active[0].components] == [0.0, first.force_n]
        assert initial.step_index == 0 and initial.active == ()
        kwargs.update(sample_time_s=0.001, commit_time_s=0.002)
        reordered = adapter.stage(before, replace(raw, contacts=raw.contacts[::-1]), **kwargs)
        assert [e.kind for e in reordered.events] == ["persist", "persist"]
        assert not any(r.material_changed or r.support_changed for r in reordered.active)
        assert reordered.active[0].components == before.active[0].components[::-1]
        kwargs.update(sample_time_s=0.002, commit_time_s=0.003)
        ended = adapter.stage(reordered, replace(raw, contacts=(zero,)), **kwargs)
        assert ended.active == () and [e.reason for e in ended.events] == ["zero_force", "absent"]
        assert all(e.record is None for e in ended.events)
        assert len(ended.generations) == 2
        kwargs.update(sample_time_s=0.003, commit_time_s=0.004)
        reentered = adapter.stage(ended, frame.loads, **kwargs)
        assert [r.identity.generation for r in reentered.active] == [2, 2]
        assert len(reentered.generations) == 2 and len(reentered.events) == 2
        assert before.active[0].identity.generation == 1


@pytest.mark.parametrize("bad", ["cross_world", "unknown_shape", "weights", "normal", "time"])
def test_pure_robot_stage_rejects_invalid_source_without_advancing_previous(bad):
    from dataclasses import replace

    from phymeo.constraints.collision_shapes import CollisionKey
    from phymeo.runtime.contact_events import SceneContactEvents

    with three_body_chain().build() as scene:
        frame = scene.step(0.001, actions={10: RobotCommand([0.0], feedforward=[0.0])})
        old, adapter = SceneContactEvents(), scene._event_adapter
        kwargs, row = _stage_inputs(scene, frame), frame.loads.contacts[0]
        if bad == "cross_world":
            row = replace(row, second=CollisionKey(99, row.second.body, row.second.shape))
        elif bad == "unknown_shape":
            row = replace(row, second=CollisionKey(0, 999, 0))
        elif bad == "weights":
            p = replace(row.sites[0].material, first_weights=((999, 1.0),))
            row = replace(row, sites=(replace(row.sites[0], material=p),))
        elif bad == "normal":
            row = replace(row, normal=(float("nan"), 0, 1))
        else:
            kwargs["sample_time_s"] = 0.0001
        with pytest.raises(ValueError):
            adapter.stage(old, replace(frame.loads, contacts=(row,)), **kwargs)
        assert old == SceneContactEvents()
        retry = adapter.stage(old, frame.loads, **_stage_inputs(scene, frame))
        assert retry == scene.contact_events


@pytest.mark.parametrize("failure", ["allocation", "material"])
def test_robot_stage_failure_rolls_back_entire_scene_then_same_input_retries(monkeypatch, failure):
    from dataclasses import replace

    import phymeo.runtime.contact_events as module

    with three_body_chain().build() as scene, three_body_chain().build() as direct:
        command = {10: RobotCommand([0.0], feedforward=[0.0])}
        for target in (scene, direct):
            target.step(0.001, actions=command)
        before, state, geometry = scene.contact_events, scene.state, scene.geometry()
        previous = scene.last_frame
        with monkeypatch.context() as patch:
            if failure == "allocation":
                def fail_allocation(_):
                    raise MemoryError("injected staging allocation")
                patch.setattr(module, "_owned_load", fail_allocation)
                expected, match = MemoryError, "staging allocation"
            else:
                original = scene._contact.evaluate
                def invalid_material(*args, **kwargs):
                    measured = original(*args, **kwargs)
                    row, *rest = measured.loads.contacts
                    site = row.sites[0]
                    material = replace(site.material, first_weights=((999, 1.0),))
                    bad = replace(row, sites=(replace(site, material=material),))
                    return replace(measured, loads=replace(measured.loads, contacts=(bad, *rest)))
                patch.setattr(scene._contact, "evaluate", invalid_material)
                expected, match = ValueError, "material weights"
            with pytest.raises(expected, match=match):
                scene.step(0.001, actions=command)
        assert scene.contact_events is before
        assert scene.state.time_s == state.time_s and scene.geometry().time_s == geometry.time_s
        assert scene.last_frame.step_number == previous.step_number == 1
        np.testing.assert_array_equal(scene.state.rigid.position, state.rigid.position)
        np.testing.assert_array_equal(scene.state.rigid.velocity, state.rigid.velocity)
        np.testing.assert_array_equal(scene.state.robots[0, 10].q, state.robots[0, 10].q)
        np.testing.assert_array_equal(scene.state.robots[0, 10].v, state.robots[0, 10].v)
        result = scene.step(0.001, actions=command)
        reference = direct.step(0.001, actions=command)
        assert scene.contact_events == direct.contact_events
        np.testing.assert_array_equal(result.rigid.position, reference.rigid.position)
        np.testing.assert_array_equal(result.rigid.velocity, reference.rigid.velocity)
        np.testing.assert_array_equal(result.robots[0, 10].q, reference.robots[0, 10].q)
        np.testing.assert_array_equal(result.robots[0, 10].v, reference.robots[0, 10].v)
        assert before.step_index == 1 and scene.contact_events.step_index == 2


def test_folded_collision_records_computational_link_owner_and_real_entry_frame(tmp_path):
    path = tmp_path / "folded_tool.urdf"
    path.write_text(
        '<robot name="folded"><link name="base"/><link name="pusher">'
        '<inertial><mass value="2"/><inertia ixx=".01" iyy=".01" izz=".01" '
        'ixy="0" ixz="0" iyz="0"/></inertial></link><link name="authored_tip">'
        '<collision name="tool"><geometry><sphere radius=".0625"/></geometry></collision>'
        '</link><joint name="slide" type="prismatic"><parent link="base"/>'
        '<child link="pusher"/><origin xyz="0 0 .25"/><axis xyz="1 0 0"/>'
        '<limit lower="-.1" upper=".1" effort="20" velocity="1"/></joint>'
        '<joint name="fixed_tool" type="fixed"><parent link="pusher"/>'
        '<child link="authored_tip"/><origin xyz=".02 0 0"/></joint></robot>',
        encoding="utf-8",
    )
    make = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    make.add_urdf(10, path, robot_state=RobotState(np.zeros(1), np.zeros(1)), kp=[0], kd=[0])
    make.add_sphere(20, 0.0625, mass=1.0, position=(0.144, 0.0, 0.25))
    make.pair(10, 20, LAW)
    with make.build() as scene:
        frame = scene.step(0.001, actions={10: RobotCommand([0.0], feedforward=[0.0])})
        (record,) = scene.contact_events.active
        owner = record.first_owner
        assert owner.body == 10 and owner.link_name == "pusher" and owner.link_index == 0
        assert owner.local_frame == "robot_model_link"
        assert record.identity.first == scene.shape_key(10, "authored_tip", "tool")
        (component,) = record.components
        assert component.load.force_n == pytest.approx(2.0, abs=1e-12)
        np.testing.assert_allclose(component.material[0].first.local_point_m, [0.0825, 0.0, 0.0], atol=1e-13)
        np.testing.assert_allclose(component.material[0].second.local_point_m, [-0.0625, 0.0, 0.0], atol=1e-13)
        assert frame.robots[0, 10].v[0] == pytest.approx(-0.001)


def test_pure_robot_stage_owns_array_backed_source_values():
    from dataclasses import FrozenInstanceError, replace

    from phymeo.runtime.contact_events import SceneContactEvents

    with three_body_chain().build() as scene:
        frame = scene.step(0.001, actions={10: RobotCommand([0.0], feedforward=[0.0])})
        raw = frame.loads.contacts[0]
        site = raw.sites[0]
        normal = np.array(raw.normal)
        point = np.array(site.material.first_point)
        traction = np.array(site.force_on_second_n)
        weights = list(site.material.first_weights)
        material = replace(site.material, first_point=point, first_weights=weights)
        mutable = replace(raw, normal=normal, sites=(replace(site, material=material, force_on_second_n=traction),))
        result = scene._event_adapter.stage(
            SceneContactEvents(), replace(frame.loads, contacts=(mutable,)), **_stage_inputs(scene, frame)
        )
        (component,) = result.active[0].components
        assert component.load == raw
        normal[:] = point[:] = traction[:] = 99.0
        weights[:] = [(999, 1.0)]
        assert component.load == raw
        with pytest.raises(FrozenInstanceError):
            component.material = ()
        with pytest.raises(FrozenInstanceError):
            result.step_index = 99
        assert result.step_index == 1


def test_fixed_root_collision_has_its_actual_owner_and_root_material_frame(tmp_path):
    path = tmp_path / "root_contact.urdf"
    path.write_text(
        '<robot name="root_contact"><link name="base"><collision name="stop">'
        '<origin xyz="0 0 .25"/><geometry><sphere radius=".0625"/></geometry>'
        '</collision></link><link name="unused_arm"><inertial><mass value="2"/>'
        '<inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/>'
        '</inertial><collision name="moving"><geometry><sphere radius=".0625"/></geometry>'
        '</collision></link><joint name="slide" type="prismatic"><parent link="base"/>'
        '<child link="unused_arm"/><origin xyz="-.5 0 .25"/><axis xyz="1 0 0"/>'
        '<limit lower="-.1" upper=".1" effort="20" velocity="1"/></joint></robot>',
        encoding="utf-8",
    )
    make = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    make.add_urdf(10, path, robot_state=RobotState(np.zeros(1), np.zeros(1)), kp=[0], kd=[0])
    make.add_sphere(20, 0.0625, mass=1.0, position=(0.124, 0.0, 0.25))
    make.pair(10, 20, LAW)
    with make.build() as scene:
        frame = scene.step(0.001, actions={10: RobotCommand([0.0], feedforward=[0.0])})
        (record,) = scene.contact_events.active
        owner = record.first_owner
        assert (owner.body, owner.kind, owner.link_index, owner.link_name, owner.local_frame) == (
            10, "robot_root", -1, "base", "robot_root"
        )
        assert record.identity.first == scene.shape_key(10, "base", "stop")
        component = record.components[0]
        np.testing.assert_allclose(component.material[0].first.local_point_m, [0.0625, 0, 0.25], atol=1e-13)
        assert component.load.force_n == pytest.approx(2.0, abs=1e-12)
        np.testing.assert_allclose(frame.robots[0, 10].fixed_root_contact_wrench, [-2, 0, 0, 0, -0.5, 0], atol=1e-12)
        np.testing.assert_array_equal(frame.robots[0, 10].contact_effort, [0.0])
        np.testing.assert_array_equal(frame.robots[0, 10].v, [0.0])
        assert frame.rigid.velocity[0, 0] == pytest.approx(0.002)
