"""The common event view must retain the ordinary scene's existing load episode."""

import numpy as np
import pytest

from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.solvers.collision_loads import NormalLayer


def sphere_pair():
    make = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    make.add_sphere(4, 1 / 8, mass=1.0, world=7)
    make.add_sphere(9, 1 / 8, mass=1.0, position=(0.0, 0.0, 9 / 32), world=7)
    make.pair(4, 9, NormalLayer(1 / 16, 128.0), world=7)
    return make


def test_ordinary_scene_exposes_the_existing_episode_through_the_common_event_view():
    with sphere_pair().build() as scene:
        frame = scene.step(1 / 256)
        assert frame.loads.contacts[0].force_n == 4.0
        np.testing.assert_array_equal(frame.state.rigid.velocity[:, 2], [-1 / 64, 1 / 64])
        old = scene.contact_history
        # The RED is a missing normal public property, after actual 4 N physics.
        view = scene.contact_events
        from phymeo.runtime.contact_events import SceneContactEvents

        assert isinstance(view, SceneContactEvents)
        assert view.step_index == old.step_index == 1
        assert view.sample_time_s == 0.0 and view.commit_time_s == 1 / 256
        assert view.generations == old.generations
        (record,) = view.active
        assert record.identity == old.active[0].identity
        assert record.first_owner.world == record.second_owner.world == 7
        assert (record.first_owner.body, record.second_owner.body) == (4, 9)
        assert record.first_owner.kind == record.second_owner.kind == "rigid"
        assert record.first_owner.local_frame == record.second_owner.local_frame == "rigid_com"
        (component,) = record.components
        assert component.material_kind == "normal_layer"
        assert component.law == NormalLayer(1 / 16, 128.0)
        assert component.load == frame.loads.contacts[0]
        assert component.material == old.active[0].material
        assert view.events[0].kind == "begin" and view.events[0].record == record
        scene.sample()
        scene.geometry()
        scene.observe_scene()
        assert scene.contact_history is old
        assert scene.contact_events == view
        scene.step(1 / 256)
        assert scene.contact_events.events[0].kind == "persist"
        assert scene.contact_events.active[0].identity == record.identity
        assert view.step_index == 1 and view.events[0].kind == "begin"


def test_pure_history_adapter_retains_particle_addresses_and_rejects_a_different_law():
    from phymeo.constraints.robot_contact import RobotContactMaterial
    from phymeo.runtime.contact_events import contact_events_from_history

    make = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    make.add_sphere(4, 1 / 8, mass=1.0, position=(0.0, 0.0, 7 / 32))
    make.add_sheet(
        9,
        [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)],
        [(0, 1, 2)],
        node_masses=1.0,
        edge_compliance=None,
    )
    law = NormalLayer(1 / 8, 64.0)
    make.pair(4, 9, law)
    with make.build() as scene:
        frame = scene.step(1 / 512)
        old = scene.contact_history
        view = contact_events_from_history(
            old, layers=scene._system.layers, bindings=scene._system.bindings
        )
        (record,) = view.active
        (part,) = record.components
        assert part.law is law and part.load == frame.loads.contacts[0]
        assert record.first_owner.kind == "rigid"
        assert record.second_owner.kind == "particle"
        assert record.second_owner.local_frame == "material_vertices"
        assert part.material == old.active[0].material
        assert part.material[0].second.local_point_m is None
        assert dict(part.material[0].second.weights) == {0: 0.25, 1: 0.25, 2: 0.5}
        assert part.load.force_n == pytest.approx(2.0)
        np.testing.assert_allclose(
            frame.loads.inputs.rigid.force.sum(0) + frame.loads.inputs.particles.force.sum(0), 0
        )
        pair = record.identity.first, record.identity.second
        with pytest.raises(TypeError, match="NormalLayer"):
            contact_events_from_history(
                old, layers={pair: RobotContactMaterial()}, bindings=scene._system.bindings
            )
        assert scene.contact_history is old


def test_ordinary_same_ids_in_two_worlds_capacity_retry_and_owned_views():
    from dataclasses import FrozenInstanceError

    from phymeo.core.capabilities import SolverFailure

    make = sphere_pair()
    make.add_sphere(4, 1 / 8, mass=1.0, world=42)
    make.add_sphere(9, 1 / 8, mass=1.0, world=42, position=(0.0, 0.0, 9 / 32))
    make.pair(4, 9, NormalLayer(1 / 16, 128.0), world=42)
    with make.build() as scene, make.build() as direct:
        initial = scene.contact_events
        assert initial.step_index == 0 and initial.sample_time_s == initial.commit_time_s == 0.0
        assert initial.active == initial.events == initial.generations == ()
        scene.sample()
        assert scene.contact_events == initial
        for target in (scene, direct):
            target.step(1 / 256)
        before, state = scene.contact_events, scene.state
        with pytest.raises(SolverFailure, match="capacity"):
            scene.step(1 / 256, capacity=0)
        assert scene.contact_events == before
        np.testing.assert_array_equal(scene.state.rigid.position, state.rigid.position)
        with pytest.raises(FrozenInstanceError):
            before.active[0].components[0].load.force_n = 100.0
        assert [
            (r.first_owner.world, r.first_owner.body, r.second_owner.body) for r in before.active
        ] == [(7, 4, 9), (42, 4, 9)]
        scene.step(1 / 256)
        direct.step(1 / 256)
        assert scene.contact_events == direct.contact_events
        assert before.step_index == 1 and initial.step_index == 0


def _finite_table_reentry(backend, cache):
    make = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    make.add_box(2, (1 / 4, 1 / 4, 1 / 8), mass=None)
    make.add_sphere(9, 1 / 8, mass=1.0, position=(0.0, 0.0, 9 / 32))
    law = NormalLayer(1 / 16, 128.0)
    make.pair(2, 9, law)
    with make.build(backend=backend, cache_root=cache) as scene:
        h = 1 / 64
        views, frames = [], []
        # Prescribed external impulses lift, return, then brake the free ball.
        # There is no pose reset or event-driven force/threshold retuning.
        for force in (252.0, -512.0, 252.0):
            external = scene.inputs()
            external.rigid.force[1, 2] = force
            frames.append(scene.step(h, external))
            views.append(scene.contact_events)
        assert [v.events[0].kind for v in views] == ["begin", "end", "begin"]
        assert [v.sample_time_s for v in views] == [0.0, h, 2 * h]
        assert views[1].active == () and views[1].events[0].record is None
        assert views[1].events[0].reason == "absent"
        assert views[2].active[0].identity.generation == 2
        assert views[0].active[0].identity.generation == 1
        assert len(views[2].generations) == 1
        for i in (0, 2):
            c = views[i].active[0].components[0]
            assert c.law is law and c.material_kind == "normal_layer"
            assert c.load.force_n == pytest.approx(4.0, abs=2e-5)
            assert c.load.gap_m == pytest.approx(1 / 32, abs=2e-7)
        np.testing.assert_allclose(
            [f.state.rigid.velocity[1, 2] for f in frames], [4, -4, 0], atol=2e-6
        )
        np.testing.assert_allclose(
            [f.state.rigid.position[1, 2] for f in frames], [11 / 32, 9 / 32, 9 / 32], atol=2e-7
        )
        if backend != "host":
            assert all(f.rigid_report.backend == "tyx:" + backend for f in frames)


def test_real_finite_table_exit_reentry_retains_normal_layer_semantics():
    _finite_table_reentry("host", None)


@pytest.mark.tyx
def test_device_ordinary_event_view_retains_actual_reentry(tyx_mode, tmp_path):
    _finite_table_reentry(tyx_mode, tmp_path)


def test_ordinary_patch_retains_all_sites_and_pure_torsional_friction():
    make = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    make.add_box(2, (1 / 8,) * 3, mass=None)
    make.add_box(
        9, (1 / 8,) * 3, mass=1.0, position=(0.0, 0.0, 5 / 16), angular_velocity=(0, 0, 0.25)
    )
    law = NormalLayer(1 / 8, 16.0, friction=0.5, slip_speed_m_s=1 / 16)
    make.pair(2, 9, law)
    with make.build() as scene:
        frame = scene.step(1 / 512)
        (component,) = scene.contact_events.active[0].components
        assert component.load == frame.loads.contacts[0]
        assert component.material == scene.contact_history.active[0].material
        assert len(component.load.sites) == len(component.material) == 12
        force = np.sum([s.force_on_second_n for s in component.load.sites], axis=0)
        torque = np.sum(
            [
                np.cross(
                    np.asarray(s.application_point_second_m) - [0, 0, 5 / 16], s.force_on_second_n
                )
                for s in component.load.sites
            ],
            axis=0,
        )
        np.testing.assert_allclose(force, [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(torque, [0, 0, -1 / 48], atol=1e-12)
        np.testing.assert_allclose(frame.loads.inputs.rigid.torque[1], torque, atol=1e-12)
        assert component.load.power_w == pytest.approx(-1 / 192)
        assert 0 < frame.state.rigid.angular_velocity[1, 2] < 0.25
