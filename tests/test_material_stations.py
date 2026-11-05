"""Actual scene samples feed explicit stations; this adapter does not solve friction."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from phymeo.dynamics.rotation import quaternion_matrix
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.solvers.collision_loads import NormalLayer
from phymeo.solvers.tangent_history import StationKey, TangentHistory, TangentUpdate

H = 1 / 64


def _table_builder(*, world=7, velocity=(0.0, 0.0, 0.0)):
    make = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    make.add_box(2, (1 / 4, 1 / 4, 1 / 8), mass=None, world=world)
    make.add_sphere(9, 1 / 8, mass=1.0, world=world,
                    position=(0.0, 0.0, 9 / 32), velocity=velocity)
    make.pair(2, 9, NormalLayer(1 / 16, 128.0), world=world)
    return make


def _table_result(world=7):
    with _table_builder(world=world).build() as scene:
        frame = scene.step(H)
        events = scene.contact_events
        assert events.active[0].components[0].load.force_n == 4.0
        return events, frame, dict(scene.rigid_rows)


def _poses(events, state, rows):
    from phymeo.runtime.material_stations import OwnerPose

    rotations = quaternion_matrix(state.orientation)
    owners = {o for r in events.active for o in (r.first_owner, r.second_owner)}
    return {o: OwnerPose(o, state.position[rows[o.world, o.body]],
                         rotations[rows[o.world, o.body]]) for o in owners}


def _registration(record, *, token=41, anchor=(0.0, 0.0, -1 / 8),
                  side="second", axis=(1.0, 0.0, 0.0), first_chart=17, second_chart=None):
    from phymeo.runtime.material_stations import MaterialStationRegistration

    return MaterialStationRegistration(
        StationKey(record.identity.first.world, token),
        record.identity.first, record.identity.second, side, anchor, axis,
        first_chart, second_chart,
    )


def _stored():
    return TangentUpdate((1 / 16, 0.0, 0.0), 1 / 64, 0.0, 0.0, "stick", 8.0)


def test_real_finite_face_load_is_bound_to_explicit_material_station():
    events, frame, rows = _table_result()
    # The first RED follows an actual finite-box/sphere 4 N normal-law step.
    from phymeo.runtime.material_stations import MaterialStationRegistry

    record = events.active[0]
    registration = _registration(record)
    registry = MaterialStationRegistry([registration])
    entry = _poses(events, frame.entry.rigid, rows)
    selected = registry.bind(events, entry, sample_time_s=0.0)
    assert registry.pairs == ((record.identity.first, record.identity.second),)
    assert selected.sample_time_s == 0.0
    (sample,), (binding,) = selected.samples, selected.bindings
    assert sample.key == binding.key == StationKey(7, 41)
    assert sample.normal_force_n == binding.site.force_n == 4.0
    assert sample.first.chart_id == 17 and sample.second.chart_id is None
    assert binding.record == record and binding.component == record.components[0]
    assert binding.site == frame.loads.contacts[0].sites[0]
    assert binding.component_index == binding.site_index == 0
    assert binding.first_owner == record.first_owner
    assert binding.second_owner == record.second_owner
    assert binding.second_material_local_m == (0.0, 0.0, -1 / 8)
    material, application = selected.measure_motion(_poses(events, frame.state.rigid, rows))
    np.testing.assert_allclose(material, [[0.0, 0.0, 1 / 1024]], atol=1e-14)
    np.testing.assert_allclose(application, material, atol=1e-14)
    for array in (material, application):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        registry.pairs = ()


def test_real_exit_reentry_keeps_history_transaction_outside_the_selector():
    from phymeo.runtime.material_stations import MaterialStationRegistry

    book, state, registry = TangentHistory(1), None, None
    generations, releases = [], []
    with _table_builder().build() as scene:
        for index, force in enumerate((252.0, -512.0, 252.0)):
            external = scene.inputs()
            external.rigid.force[scene.rigid_rows[7, 9], 2] = force
            frame = scene.step(H, external)
            events = scene.contact_events
            if registry is None:
                registry = MaterialStationRegistry([_registration(events.active[0])])
                state = book.initial_state()
            frames = _poses(events, frame.entry.rigid, scene.rigid_rows)
            selected = registry.bind(events, frames, sample_time_s=index * H)
            trial = book.prepare(state, selected.samples, sample_time_s=index * H, dt_s=H)
            candidate = book.commit(state, trial, {s.key: _stored() for s in selected.samples})
            assert candidate == book.commit(state, trial, {s.key: _stored() for s in selected.samples})
            state = candidate
            generations.append(tuple(r.identity.generation for r in state.active))
            releases.append(state.dropped_spring_energy_j)
    assert generations == [(1,), (), (2,)]
    assert releases == [0.0, 1 / 64, 0.0]


@pytest.mark.parametrize("chart", [None, 17])
def test_closest_chart_drift_is_separate_from_the_same_entry_material_motion(chart):
    from phymeo.runtime.material_stations import MaterialStationRegistry

    with _table_builder(velocity=(1 / 8, 0.0, 0.0)).build() as scene:
        external = scene.inputs()
        external.rigid.force[scene.rigid_rows[7, 9], 2] = -4.0
        first = scene.step(H, external)
        events = scene.contact_events
        registry = MaterialStationRegistry([_registration(events.active[0], first_chart=chart)])
        selection = registry.bind(events, _poses(events, first.entry.rigid, scene.rigid_rows),
                                  sample_time_s=0.0)
        motion, _ = selection.measure_motion(_poses(events, first.state.rigid, scene.rigid_rows))
        np.testing.assert_allclose(motion, [[1 / 512, 0.0, 0.0]], atol=1e-14)
        book = TangentHistory(1)
        initial = book.initial_state()
        trial = book.prepare(initial, selection.samples, sample_time_s=0.0, dt_s=H)
        state = book.commit(initial, trial, {selection.samples[0].key: _stored()})
        second = scene.step(H, external)
        next_events = scene.contact_events
        next_selection = registry.bind(
            next_events, _poses(next_events, second.entry.rigid, scene.rigid_rows), sample_time_s=H
        )
        old_x = selection.bindings[0].first_material_local_m[0]
        new_x = next_selection.bindings[0].first_material_local_m[0]
        assert new_x - old_x == pytest.approx(1 / 512, abs=1e-14)
        again, _ = selection.measure_motion(_poses(events, first.state.rigid, scene.rigid_rows))
        np.testing.assert_array_equal(again, motion)
        trial = book.prepare(state, next_selection.samples, sample_time_s=H, dt_s=H)
        before, after = selection.samples[0], next_selection.samples[0]
        assert trial.stations[0].continuity == ("reset" if chart is None else "same_chart"), (
            f"first.feature={before.first.feature}->{after.first.feature}; "
            f"second.feature={before.second.feature}->{after.second.feature}; "
            f"ended={tuple(e.reason for e in trial.ended)}; "
            f"first.local={before.first.local_point_m}->{after.first.local_point_m}; "
            f"second.local={before.second.local_point_m}->{after.second.local_point_m}"
        )
        assert trial.dropped_spring_energy_j == (1 / 64 if chart is None else 0.0)
        if chart is not None:
            assert trial.stations[0].prior.elastic_displacement_m == (1 / 16, 0.0, 0.0)
            assert trial.stations[0].prior.plastic_slip_m == 0.0


def test_finite_four_site_patch_uses_actual_site_force_without_fraction_twice():
    from phymeo.runtime.material_stations import MaterialStationRegistry

    make = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    make.add_box(2, (1 / 8,) * 3, mass=None)
    make.add_box(9, (1 / 8,) * 3, mass=1.0, position=(0.0, 0.0, 5 / 16))
    make.pair(2, 9, NormalLayer(1 / 8, 16.0))
    with make.build() as scene:
        frame = scene.step(1 / 512)
        events, rows = scene.contact_events, scene.rigid_rows
    record = events.active[0]
    assert len(record.components[0].load.sites) == 4
    # Tokens belong to four authored cuboid corners, not enumerated output rows.
    declarations = [_registration(record, token=100 + 2 * ix + iy, anchor=(x, y, -1 / 8))
                    for ix, x in enumerate((-1 / 8, 1 / 8))
                    for iy, y in enumerate((-1 / 8, 1 / 8))]
    registry = MaterialStationRegistry(declarations[::-1])
    frames = _poses(events, frame.entry.rigid, rows)
    selected = registry.bind(events, frames, sample_time_s=0.0)
    assert [s.key.token for s in selected.samples] == [100, 101, 102, 103]
    assert [s.normal_force_n for s in selected.samples] == pytest.approx([0.25] * 4)
    assert sum(s.normal_force_n for s in selected.samples) == pytest.approx(1.0)
    for binding in selected.bindings:
        assert binding.site.fraction == pytest.approx(0.25)
        assert binding.record.components[binding.component_index] == binding.component
        assert binding.component.load.sites[binding.site_index] == binding.site
    component = record.components[0]
    reversed_component = replace(component, material=component.material[::-1],
                                 load=replace(component.load, sites=component.load.sites[::-1]))
    reordered = replace(events, active=(replace(record, components=(reversed_component,)),))
    assert registry.bind(reordered, frames, sample_time_s=0.0).samples == selected.samples


@pytest.mark.parametrize("turned", [False, True])
def test_real_urdf_pad_sites_keep_local_anchor_identity_across_component_order(tmp_path, turned):
    from test_robot_contact_events import _twisting_pad_scene
    from phymeo.dynamics.robot_kinematics import forward_kinematics
    from phymeo.runtime.material_stations import MaterialStationRegistry, OwnerPose
    from phymeo.runtime.robot_scene import RobotCommand

    make, _, _, _ = _twisting_pad_scene(tmp_path, spin=0.0, transformed=turned)
    with make.build() as scene:
        frame = scene.step(1 / 10000, actions={10: RobotCommand([0.0], feedforward=[0.0])})
        events = scene.contact_events
        asset = scene.assets[0, 10]
        geo = forward_kinematics(asset.model, frame.robots[0, 10].entry_q)
        end_geo = forward_kinematics(asset.model, frame.robots[0, 10].q)
    record = events.active[0]
    robot, rigid = record.first_owner, record.second_owner
    assert robot.kind == "robot_link" and robot.body == 10 and robot.link_index == 0
    assert record.identity.first.body != robot.body
    frames = {
        robot: OwnerPose(robot, geo.position[0], geo.rotation[0]),
        rigid: OwnerPose(rigid, frame.entry_rigid.position[0],
                         quaternion_matrix(frame.entry_rigid.orientation)[0]),
    }
    declarations = [_registration(record, token=token, anchor=(0.01, y, 0.0), side="first",
                                  axis=(0.0, 0.0, 1.0), first_chart=None, second_chart=11)
                    for token, y in ((101, 0.009), (202, -0.009))]
    registry = MaterialStationRegistry(declarations)
    selected = registry.bind(events, frames, sample_time_s=0.0)
    assert len(selected.samples) == 2
    assert [s.key.token for s in selected.samples] == [101, 202]
    for s, b in zip(selected.samples, selected.bindings, strict=True):
        assert s.normal_force_n == b.site.force_n > 0
        assert b.first_owner.link_index == 0
        assert b.first_material_local_m[1] == pytest.approx(0.009 if s.key.token == 101 else -0.009)
    permuted = replace(events, active=(replace(record, components=record.components[::-1]),))
    repeat = registry.bind(permuted, frames, sample_time_s=0.0)
    assert repeat.samples == selected.samples
    end = {
        robot: OwnerPose(robot, end_geo.position[0], end_geo.rotation[0]),
        rigid: OwnerPose(rigid, frame.rigid.position[0], quaternion_matrix(frame.rigid.orientation)[0]),
    }
    material, application = selected.measure_motion(end)
    np.testing.assert_allclose(material, application, atol=1e-12)
    assert np.linalg.norm(material) > 0


def test_real_rotating_body_keeps_material_and_application_motion_distinct():
    from phymeo.runtime.material_stations import MaterialStationRegistry

    make = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    make.add_sphere(4, 1 / 8, mass=1.0)
    make.add_sphere(9, 1 / 8, mass=1.0, position=(0.0, 0.0, 9 / 32))
    make.pair(4, 9, NormalLayer(1 / 16, 128.0, friction=0.5))
    with make.build() as scene:
        external = scene.inputs()
        external.rigid.torque[scene.rigid_rows[0, 4], 1] = 1 / 8
        frame = scene.step(H, external)
        events, rows = scene.contact_events, scene.rigid_rows
    assert frame.loads.contacts[0].force_n == 4.0
    assert frame.state.rigid.angular_velocity[0, 1] != 0
    declaration = _registration(events.active[0], side="first", anchor=(0.0, 0.0, 1 / 8))
    selected = MaterialStationRegistry([declaration]).bind(
        events, _poses(events, frame.entry.rigid, rows), sample_time_s=0.0
    )
    material, application = selected.measure_motion(_poses(events, frame.state.rigid, rows))
    rotation = quaternion_matrix(frame.state.rigid.orientation)[0]
    np.testing.assert_allclose(application[0] - material[0],
                               -(rotation - np.eye(3)) @ [0.0, 0.0, 1 / 64], atol=1e-14)
    assert np.linalg.norm(application - material) > 1e-6
    binding = selected.bindings[0]
    assert binding.application_point_first_m == binding.application_point_second_m
    assert binding.first_material_local_m != binding.application_point_first_m


@pytest.mark.parametrize("bad", ["missing_anchor", "ambiguous", "duplicate_site", "time", "owner"])
def test_uncovered_ambiguous_or_wrong_sample_rejects_then_exact_retry_is_unchanged(bad):
    from phymeo.runtime.material_stations import MaterialStationRegistry

    events, frame, rows = _table_result()
    record = events.active[0]
    declaration = _registration(record)
    frames = _poses(events, frame.entry.rigid, rows)
    registry = MaterialStationRegistry([declaration])
    expected = registry.bind(events, frames, sample_time_s=0.0)
    source, poses, time, target = events, frames, 0.0, registry
    if bad == "missing_anchor":
        target = MaterialStationRegistry([replace(declaration, anchor_local_m=(1.0, 0, 0))])
    elif bad == "ambiguous":
        target = MaterialStationRegistry([declaration, replace(declaration, key=StationKey(7, 42))])
    elif bad == "duplicate_site":
        source = replace(events, active=(replace(record, components=record.components * 2),))
    elif bad == "time":
        time = H
    else:
        poses = dict(frames)
        poses[record.first_owner] = frames[record.second_owner]
    with pytest.raises(ValueError):
        target.bind(source, poses, sample_time_s=time)
    retry = registry.bind(events, frames, sample_time_s=0.0)
    assert retry.samples == expected.samples and retry.bindings == expected.bindings


def test_owned_source_and_pose_copies_do_not_move_prior_selection():
    from phymeo.runtime.material_stations import MaterialStationRegistry, OwnerPose

    events, frame, rows = _table_result()
    record, component = events.active[0], events.active[0].components[0]
    local = np.asarray(component.material[0].second.local_point_m).copy()
    point = np.asarray(component.load.sites[0].application_point_second_m).copy()
    addresses = (replace(component.material[0], second=replace(component.material[0].second,
                                                             local_point_m=local)),)
    load = replace(component.load, sites=(replace(component.load.sites[0],
                                                  application_point_second_m=point),))
    source = replace(events, active=(replace(record, components=(replace(component,
                             material=addresses, load=load),)),))
    frames = _poses(events, frame.entry.rigid, rows)
    position = np.asarray(frames[record.second_owner].position_m).copy()
    rotation = np.eye(3)
    frames[record.second_owner] = OwnerPose(record.second_owner, position, rotation)
    selected = MaterialStationRegistry([_registration(record)]).bind(source, frames, sample_time_s=0.0)
    local[:] = 99
    point[:] = 88
    position[:] = 77
    rotation[:] = 0
    material, application = selected.measure_motion(_poses(events, frame.entry.rigid, rows))
    np.testing.assert_array_equal(material, np.zeros((1, 3)))
    np.testing.assert_array_equal(application, np.zeros((1, 3)))
    assert selected.bindings[0].component == component
    assert selected.samples[0].second.local_point_m == (0.0, 0.0, -1 / 8)


def test_empty_unselected_pairs_and_same_local_token_in_other_world_are_explicit():
    from phymeo.runtime.contact_events import SceneContactEvents
    from phymeo.runtime.material_stations import MaterialStationRegistry

    a, fa, ra = _table_result(7)
    b, fb, rb = _table_result(42)
    registrations = [_registration(e.active[0]) for e in (a, b)]
    poses = _poses(a, fa.entry.rigid, ra) | _poses(b, fb.entry.rigid, rb)
    combined = replace(a, active=a.active + b.active)
    registry = MaterialStationRegistry(registrations[::-1])
    selected = registry.bind(combined, poses, sample_time_s=0.0)
    assert [s.key for s in selected.samples] == [StationKey(7, 41), StationKey(42, 41)]
    empty = registry.bind(SceneContactEvents(), {}, sample_time_s=0.0)
    assert empty.samples == empty.bindings == ()
    material, application = empty.measure_motion({})
    assert material.shape == application.shape == (0, 3)
    # The undeclared pair is returned to the old friction path, not silently claimed.
    partial = MaterialStationRegistry(registrations[:1]).bind(combined, poses, sample_time_s=0.0)
    assert len(partial.samples) == 1 and partial.samples[0].key.world == 7


@pytest.mark.parametrize("bad", ["duplicate_key", "world", "side", "axis", "tolerance"])
def test_bad_registration_fails_before_any_binding(bad):
    from phymeo.runtime.material_stations import MaterialStationRegistry

    events, _, _ = _table_result()
    registration = _registration(events.active[0])
    with pytest.raises(ValueError):
        if bad == "duplicate_key":
            MaterialStationRegistry([registration, registration])
        else:
            updates = {"world": {"key": StationKey(42, 41)}, "side": {"anchor_side": "either"},
                       "axis": {"tangent_axis_local": (0.0, 0.0, 0.0)},
                       "tolerance": {"match_tolerance_m": float("inf")}}[bad]
            MaterialStationRegistry([replace(registration, **updates)])


def test_bad_pose_unsupported_owner_and_parallel_material_axis_are_explicit():
    from phymeo.runtime.material_stations import MaterialStationRegistry, OwnerPose

    events, frame, rows = _table_result()
    owner = events.active[0].first_owner
    for bad in (np.diag([-1.0, 1.0, 1.0]), np.zeros((3, 3)), np.full((3, 3), np.nan)):
        with pytest.raises(ValueError):
            OwnerPose(owner, (0.0, 0.0, 0.0), bad)
    with pytest.raises(ValueError):
        OwnerPose(replace(owner, kind="particle", local_frame="material_vertices"),
                  (0.0, 0.0, 0.0), np.eye(3))
    declaration = _registration(events.active[0], axis=(0.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        MaterialStationRegistry([declaration]).bind(
            events, _poses(events, frame.entry.rigid, rows), sample_time_s=0.0
        )
