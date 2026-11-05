"""Committed loaded-pair records retain real forces without consuming read-only samples."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from phymeo.core.capabilities import SolverFailure
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.solvers.collision_loads import NormalLayer


def _two_balls(*, damping=0.0, worlds=(7,)):
    b = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    for world in worlds:
        b.add_sphere(4, 1 / 8, mass=1.0, world=world)
        b.add_sphere(9, 1 / 8, mass=1.0, position=(0.0, 0.0, 9 / 32), world=world)
        b.pair(4, 9, NormalLayer(1 / 16, 128.0, damping), world=world)
    return b


def _keys(identity):
    return (
        (identity.first.world, identity.first.body, identity.first.shape),
        (identity.second.world, identity.second.body, identity.second.shape),
        identity.generation,
    )


def test_public_scene_records_entry_loads_only_when_a_step_commits():
    with _two_balls().build() as scene:
        initial = scene.contact_history
        assert initial.step_index == 0
        assert initial.sample_time_s == initial.commit_time_s == 0.0
        assert initial.active == initial.events == ()
        for _ in range(2):
            sample, geometry = scene.observe_scene()
            assert sample.loads.contacts[0].force_n == pytest.approx(4.0)
            assert geometry.time_s == 0.0
        assert scene.contact_history is initial
        h = 1 / 256
        frame = scene.step(h)
        history = scene.contact_history
        assert history.step_index == 1
        assert history.sample_time_s == 0.0 and history.commit_time_s == h
        (record,) = history.active
        (event,) = history.events
        assert _keys(record.identity) == ((7, 4, 0), (7, 9, 0), 1)
        assert event.kind == "begin" and event.record is record
        assert record.load == frame.loads.contacts[0]
        assert record.load.force_n == pytest.approx(4.0)
        # Two equal masses: the actual impulse is +/-4*h, not a fabricated event force.
        np.testing.assert_allclose(frame.state.rigid.velocity[:, 2], [-1 / 64, 1 / 64])
        np.testing.assert_allclose(frame.state.rigid.velocity.sum(axis=0), 0.0, atol=1e-14)
        assert scene.sample().loads.contacts[0].force_n < record.load.force_n
        assert scene.contact_history is history
        with pytest.raises(FrozenInstanceError):
            history.step_index = 99
        with pytest.raises(FrozenInstanceError):
            record.load.force_n = 99.0
        frame.state.rigid.position[:] = 20.0
        frame.loads.inputs.rigid.force[:] = 30.0
        assert scene.contact_history is history
        assert history.active[0].load.force_n == 4.0
        scene.step(h)
        again = scene.contact_history
        assert again.active[0].identity == record.identity
        assert again.events[0].kind == "persist"
        assert again.sample_time_s == h and again.commit_time_s == 2 * h
        assert history.step_index == 1 and history.events[0].kind == "begin"


def _check_zero_force_reentry(backend, cache):
    h = 1 / 256
    with _two_balls(damping=4.0).build(backend=backend, cache_root=cache) as scene:
        outward = scene.inputs()
        outward.rigid.force[:, 2] = [-60.0, 60.0]
        first = scene.step(h, outward)
        identity = scene.contact_history.active[0].identity
        assert first.loads.contacts[0].force_n == pytest.approx(4.0, abs=3e-6)
        np.testing.assert_allclose(first.state.rigid.velocity[:, 2], [-0.25, 0.25], atol=2e-7)
        zero = scene.sample().loads.contacts
        assert len(zero) == 1 and zero[0].force_n == 0.0
        assert zero[0].gap_m == pytest.approx(17 / 512, abs=2e-7)
        assert scene.contact_history.active[0].identity == identity
        inward = scene.inputs()
        inward.rigid.force[:, 2] = [128.0, -128.0]
        second = scene.step(h, inward)
        ended = scene.contact_history
        assert ended.active == ()
        (event,) = ended.events
        assert event.kind == "end" and event.identity == identity
        assert event.record is None and event.reason == "zero_force"
        assert ended.sample_time_s == h and ended.commit_time_s == 2 * h
        assert second.loads.contacts[0].force_n == 0.0
        np.testing.assert_allclose(second.state.rigid.velocity[:, 2], [0.25, -0.25], atol=2e-7)
        np.testing.assert_allclose(second.state.rigid.position[:, 2], [0.0, 9 / 32], atol=2e-7)
        third = scene.step(h)
        entered = scene.contact_history
        assert entered.events[0].kind == "begin"
        assert entered.active[0].identity.generation == 2
        assert entered.active[0].identity.first == identity.first
        assert entered.active[0].load.force_n == pytest.approx(12.0, abs=3e-5)
        np.testing.assert_allclose(third.state.rigid.velocity.sum(axis=0), 0.0, atol=1e-6)
        assert ended.active == () and ended.events[0].record is None


def test_zero_force_layer_exits_and_reenters_a_new_loaded_generation():
    _check_zero_force_reentry("host", None)


@pytest.mark.tyx
def test_actual_device_preserves_loaded_pair_times_and_generations(tyx_mode, tmp_path):
    _check_zero_force_reentry(tyx_mode, tmp_path)


def test_same_local_pairs_in_different_worlds_and_failed_retry_do_not_share_history():
    b = _two_balls(worlds=(42, 7))
    with b.build() as failed, b.build() as direct:
        initial = failed.contact_history
        for scene in (failed, direct):
            scene.step(1 / 256)
        before = failed.contact_history
        state = failed.state
        frame = failed.last_frame
        with pytest.raises(SolverFailure, match="capacity"):
            failed.step(1 / 256, capacity=0)
        assert failed.contact_history is before
        assert failed.last_frame.path is frame.path
        assert failed.geometry().time_s == 1 / 256
        np.testing.assert_array_equal(failed.state.rigid.position, state.rigid.position)
        failed.sample()
        failed.geometry()
        assert failed.contact_history is before
        failed.step(1 / 256)
        direct.step(1 / 256)
        assert failed.contact_history == direct.contact_history
        assert [_keys(r.identity) for r in failed.contact_history.active] == [
            ((7, 4, 0), (7, 9, 0), 1),
            ((42, 4, 0), (42, 9, 0), 1),
        ]
        assert initial.step_index == 0 and initial.active == ()


def _finite_table_ball(*, rotated=False):
    b = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    transform = rotation if rotated else np.eye(3)
    origin = np.array([2.0, -3.0, 1.0]) if rotated else np.zeros(3)
    q = (0.5,) * 4 if rotated else (0.0, 0.0, 0.0, 1.0)
    b.add_box(2, (1 / 4, 1 / 4, 1 / 8), mass=None, position=origin, orientation_xyzw=q)
    b.add_sphere(
        9,
        1 / 8,
        mass=1.0,
        position=origin + transform @ np.array([3 / 16, 0.0, 9 / 32]),
        velocity=transform @ np.array([1.0, 0.0, 0.0]),
        orientation_xyzw=q,
    )
    b.pair(2, 9, NormalLayer(1 / 16, 128.0))
    return b, transform, origin


def test_finite_support_changes_material_feature_then_ends_without_stale_force():
    b, _, _ = _finite_table_ball()
    with b.build() as scene:
        histories = []
        frames = []
        for _ in range(16):
            frames.append(scene.step(1 / 64))
            histories.append(scene.contact_history)
        active = [h.active[0] for h in histories if h.active]
        assert active[0].load.force_n == pytest.approx(4.0)
        assert all(r.identity == active[0].identity for r in active)
        # Real top-face motion reaches the finite box edge, then leaves the layer.
        assert any(r.material_changed for r in active[1:])
        assert any(r.support_changed for r in active[1:])
        assert any(r.load.normal[0] > 0.0 for r in active)
        assert histories[-1].active == ()
        ended = [e for h in histories for e in h.events if e.kind == "end"]
        assert len(ended) == 1 and ended[0].reason == "absent"
        assert ended[0].record is None
        assert histories[-1].generations == (active[0].identity,)
        assert frames[-1].loads.contacts == ()
        assert frames[-1].state.rigid.position[1, 0] > 0.375
        # Sampled surface coordinate uses the body frame, including at the edge.
        for h, f in zip(histories, frames, strict=True):
            if not h.active:
                continue
            r = h.active[0]
            s = r.load.sites[0]
            local = r.material[0]
            np.testing.assert_allclose(local.first.local_point_m, s.material.first_point)
            np.testing.assert_allclose(
                np.asarray(local.second.local_point_m) + f.entry.rigid.position[1],
                s.material.second_point,
                atol=1e-13,
            )
        assert np.max(np.abs(frames[-1].state.rigid.angular_velocity[1])) < 1e-12


def test_global_rigid_transform_preserves_pair_identity_and_material_addresses():
    b, _, _ = _finite_table_ball()
    rotated, transform, origin = _finite_table_ball(rotated=True)
    with b.build() as scene, rotated.build() as other:
        for _ in range(7):
            a = scene.step(1 / 64)
            c = other.step(1 / 64)
            left, right = scene.contact_history, other.contact_history
            assert [e.kind for e in left.events] == [e.kind for e in right.events]
            assert [r.identity for r in left.active] == [r.identity for r in right.active]
            np.testing.assert_allclose(
                c.state.rigid.position,
                a.state.rigid.position @ transform.T + origin,
                atol=2e-12,
            )
            for x, y in zip(left.active, right.active, strict=True):
                assert x.load.force_n == pytest.approx(y.load.force_n, abs=2e-10)
                np.testing.assert_allclose(y.load.normal, transform @ x.load.normal, atol=2e-12)
                for u, v in zip(x.material, y.material, strict=True):
                    np.testing.assert_allclose(
                        u.first.local_point_m, v.first.local_point_m, atol=2e-12
                    )
                    np.testing.assert_allclose(
                        u.second.local_point_m, v.second.local_point_m, atol=2e-12
                    )
        # Exact material_changed flags are numerical observations, not a new feature identity.
        assert scene.contact_history.active[0].identity.generation == 1


def test_observed_friction_keeps_real_surface_and_shared_application_points_distinct():
    b = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    b.add_sphere(4, 1 / 8, mass=1.0)
    b.add_sphere(9, 1 / 8, mass=1.0, position=(0.0, 0.0, 9 / 32), velocity=(0.5, 0.0, 0.0))
    b.pair(4, 9, NormalLayer(1 / 16, 128.0, friction=0.5))
    with b.build() as scene:
        frame = scene.step(1 / 256)
        record = scene.contact_history.active[0]
        site = record.load.sites[0]
        address = record.material[0]
        np.testing.assert_allclose(site.force_on_second_n, [-2.0, 0.0, 4.0], atol=1e-12)
        assert site.power_w == pytest.approx(-1.0)
        assert site.application_point_first_m == site.application_point_second_m
        assert site.application_point_first_m == (0.0, 0.0, 9 / 64)
        assert address.first.local_point_m == (0.0, 0.0, 1 / 8)
        assert address.second.local_point_m == (0.0, 0.0, -1 / 8)
        np.testing.assert_allclose(frame.state.rigid.velocity.sum(axis=0), [0.5, 0.0, 0.0])
        np.testing.assert_allclose(
            frame.loads.inputs.rigid.torque.sum(axis=0)
            + np.cross(frame.entry.rigid.position, frame.loads.inputs.rigid.force).sum(axis=0),
            0.0,
            atol=1e-12,
        )


def test_material_nodes_on_an_independently_moving_sheet_keep_original_weights():
    b = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    b.add_sphere(4, 1 / 8, mass=1.0, position=(0.0, 0.0, 5 / 32))
    b.add_sheet(
        9,
        [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)],
        [(0, 1, 2)],
        node_masses=1.0,
        edge_compliance=None,
    )
    b.pair(4, 9, NormalLayer(1 / 16, 128.0))
    with b.build() as scene:
        frame = scene.step(1 / 512)
        record = scene.contact_history.active[0]
        material = record.material[0]
        assert material.first.kind == "rigid"
        assert material.second.kind == "particle" and material.second.local_point_m is None
        assert material.second.weights == ((0, 0.25), (1, 0.25), (2, 0.5))
        np.testing.assert_allclose(
            frame.state.particles.velocity[:, 2], [-1 / 512, -1 / 512, -2 / 512]
        )
        np.testing.assert_allclose(
            frame.state.rigid.velocity.sum(axis=0) + frame.state.particles.velocity.sum(axis=0),
            0.0,
            atol=1e-12,
        )
        assert np.ptp(frame.state.particles.position[:, 2]) > 0.0


def test_failed_owned_result_copy_does_not_publish_state_or_contact_lifecycle(monkeypatch):
    from phymeo.solvers.collision_loads import CollisionLoadStep

    with _two_balls().build() as scene, _two_balls().build() as direct:
        before = scene.contact_history
        original = CollisionLoadStep.owned_copy

        def fail_copy(self):
            raise RuntimeError("injected owned-result copy failure")

        with monkeypatch.context() as patch:
            patch.setattr(CollisionLoadStep, "owned_copy", fail_copy)
            with pytest.raises(RuntimeError, match="owned-result"):
                scene.step(1 / 256)
        assert CollisionLoadStep.owned_copy is original
        assert scene.last_frame is None and scene.contact_history is before
        assert scene.geometry().time_s == 0.0
        np.testing.assert_array_equal(scene.state.rigid.velocity, 0.0)
        scene.step(1 / 256)
        direct.step(1 / 256)
        assert scene.contact_history == direct.contact_history


def test_reusable_staging_is_pure_and_rejects_duplicate_or_stale_input():
    from dataclasses import replace
    from types import MappingProxyType

    from phymeo.runtime.contact_history import ContactHistory, advance_contact_history
    from phymeo.solvers.collision_loads import RigidAttachment

    with _two_balls().build() as scene:
        frame = scene.step(1 / 256)
    a, b = frame.loads.contacts[0].first, frame.loads.contacts[0].second
    bindings = MappingProxyType(
        {
            a: RigidAttachment(0, ((0.0, 0.0, 0.0),)),
            b: RigidAttachment(1, ((0.0, 0.0, 0.0),)),
        }
    )
    initial = ContactHistory()
    kwargs = dict(sample_time_s=0.0, commit_time_s=1 / 256, bindings=bindings)
    staged = advance_contact_history(initial, frame, **kwargs)
    assert initial.step_index == 0 and initial.active == initial.generations == ()
    assert staged.active[0].load.force_n == 4.0
    duplicate = replace(frame, loads=replace(frame.loads, contacts=frame.loads.contacts * 2))
    with pytest.raises(ValueError, match="duplicate"):
        advance_contact_history(initial, duplicate, **kwargs)
    with pytest.raises(ValueError, match="times"):
        advance_contact_history(staged, frame, **kwargs)
    assert staged.step_index == 1 and staged.active[0].identity.generation == 1
