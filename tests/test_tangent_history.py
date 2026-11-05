"""Solver-owned tangential state: real material samples, explicit station tokens.

The manually supplied updates below test storage/transport, not a friction solve.
The parent owns the separate coupled-law/physical-state integration test.
"""

from dataclasses import replace

import numpy as np
import pytest

from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.solvers.collision_loads import NormalLayer

H = 1 / 256


def _sphere_record(*, world=7, transformed=False):
    R = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]]) if transformed else np.eye(3)
    shift = np.array([2.0, -3.0, 1.0]) if transformed else np.zeros(3)
    q = (0.5,) * 4 if transformed else (0.0, 0.0, 0.0, 1.0)
    make = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    for body, z in ((4, 0.0), (9, 9 / 32)):
        make.add_sphere(
            body,
            1 / 8,
            mass=1.0,
            world=world,
            position=shift + R @ [0.0, 0.0, z],
            orientation_xyzw=q,
        )
    make.pair(4, 9, NormalLayer(1 / 16, 128.0), world=world)
    with make.build() as scene:
        frame = scene.step(H)
        (record,) = scene.contact_events.active
        assert record.components[0].load.force_n == pytest.approx(4.0)
        np.testing.assert_allclose(frame.state.rigid.velocity.sum(0), 0.0, atol=1e-14)
        return record, R


def _sample(record, token, frame, *, component=0, chart_id=None):
    from phymeo.solvers.tangent_history import MaterialAddress, StationKey, StationSample

    load = record.components[component]
    assert len(load.material) == 1
    material = load.material[0]
    addresses = []
    for side in ("first", "second"):
        address = getattr(material, side)
        addresses.append(
            MaterialAddress(
                owner=getattr(record, side + "_owner"),
                shape=getattr(record.identity, side),
                topology_revision=0,
                feature=tuple(node for node, weight in address.weights if weight != 0),
                weights=address.weights,
                local_point_m=address.local_point_m,
                chart_id=chart_id,
            )
        )
    return StationSample(
        StationKey(record.identity.first.world, token),
        *addresses,
        normal=load.load.normal,
        frame_world=frame,
        normal_force_n=load.load.force_n,
    )


def _update(elastic=(1 / 16, 0.0, 0.0), *, stiffness=8.0):
    from phymeo.solvers.tangent_history import TangentUpdate

    return TangentUpdate(
        elastic_displacement_m=elastic,
        stored_energy_j=0.5 * stiffness * float(np.asarray(elastic) @ elastic),
        plastic_slip_m=1 / 32,
        tangent_work_j=-1 / 128,
        regime="stick",
        stiffness_n_per_m=stiffness,
    )


def test_real_loaded_station_has_transactional_material_identity():
    record, frame = _sphere_record()
    # A real 4 N accepted step precedes the new API's missing-module RED.
    from phymeo.solvers.tangent_history import TangentHistory

    sample = _sample(record, 41, frame)
    book = TangentHistory(4, max_frame_turn_rad=0.3)
    initial = book.initial_state()
    trial = book.prepare(initial, [sample], sample_time_s=0.0, dt_s=H)
    assert initial.step_index == 0 and initial.active == initial.generations == ()
    assert trial.keys == (sample.key,)
    assert trial.stations[0].continuity == "begin"
    np.testing.assert_array_equal(trial.elastic_tangent_m, [[0.0, 0.0]])
    update = _update()
    candidate = book.commit(initial, trial, {sample.key: update})
    assert initial.step_index == 0 and initial.active == ()
    assert candidate.step_index == 1 and candidate.commit_time_s == H
    assert candidate.sample_time_s == 0.0
    assert candidate.active[0].sample.first == sample.first
    assert candidate.active[0].sample.second == sample.second
    assert candidate.active[0].update == update
    assert candidate.active[0].identity.generation == 1
    again = book.commit(initial, trial, {sample.key: update})
    assert again == candidate  # Pure commit has no globally consumed bit.
    with pytest.raises(ValueError, match="stale|base"):
        book.commit(candidate, trial, {sample.key: update})
    next_trial = book.prepare(candidate, [sample], sample_time_s=H, dt_s=H)
    assert next_trial.stations[0].continuity == "persist"
    np.testing.assert_array_equal(next_trial.elastic_tangent_m, [[1 / 16, 0.0]])
    assert next_trial.dropped_spring_energy_j == 0.0


def test_actual_transformed_material_frames_transport_twist_without_fake_slip():
    from phymeo.solvers.tangent_history import TangentHistory

    record, frame = _sphere_record()
    rotated, R = _sphere_record(transformed=True)
    sample, other = _sample(record, 7, frame), _sample(rotated, 7, R)
    for side in ("first", "second"):
        np.testing.assert_allclose(
            getattr(sample, side).local_point_m, getattr(other, side).local_point_m, atol=1e-14
        )
    book = TangentHistory(1, max_frame_turn_rad=3 * np.pi / 4)
    initial = book.initial_state()
    first = book.prepare(initial, [sample], sample_time_s=0.0, dt_s=H)
    state = book.commit(initial, first, {sample.key: _update()})
    # Independent real geometries establish covariance, not a simulated turn trajectory.
    trial = book.prepare(state, [other], sample_time_s=H, dt_s=H)
    np.testing.assert_allclose(trial.elastic_displacement_m[0], R @ [1 / 16, 0, 0], atol=1e-14)
    np.testing.assert_allclose(trial.elastic_tangent_m[0], [1 / 16, 0], atol=1e-14)
    assert trial.dropped_spring_energy_j == 0.0
    assert trial.stations[0].prior.stored_energy_j == state.active[0].update.stored_energy_j
    assert trial.stations[0].prior.plastic_slip_m == 1 / 32
    assert trial.stations[0].prior.tangent_work_j == -1 / 128
    # Normal unchanged, but the material frame twists about it: do not use only n_old -> n_new.
    twist = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    turned = replace(sample, frame_world=twist)
    trial = book.prepare(state, [turned], sample_time_s=H, dt_s=H)
    np.testing.assert_allclose(trial.elastic_displacement_m[0], [0, 1 / 16, 0], atol=1e-14)
    assert trial.stations[0].prior.stored_energy_j == 1 / 64


def test_explicit_keys_reorder_and_world_isolation_do_not_follow_component_slots():
    from phymeo.solvers.tangent_history import TangentHistory

    first, F = _sphere_record(world=7)
    second, G = _sphere_record(world=42)
    a, b = _sample(first, 5, F), _sample(second, 5, G)
    book = TangentHistory(2, max_frame_turn_rad=0.3)
    initial = book.initial_state()
    trial = book.prepare(initial, [b, a], sample_time_s=0.0, dt_s=H)
    state = book.commit(initial, trial, {a.key: _update(), b.key: _update((0, 1 / 8, 0))})
    left = book.prepare(state, [a, b], sample_time_s=H, dt_s=H)
    right = book.prepare(state, [b, a], sample_time_s=H, dt_s=H)
    assert left.keys == right.keys == (a.key, b.key)
    np.testing.assert_array_equal(left.elastic_tangent_m, [[1 / 16, 0], [0, 1 / 8]])
    np.testing.assert_array_equal(right.elastic_tangent_m, left.elastic_tangent_m)
    with pytest.raises(ValueError, match="duplicate"):
        book.prepare(state, [a, a], sample_time_s=H, dt_s=H)
    assert state.step_index == 1


def test_zero_load_exit_reentry_and_capacity_keep_explicit_energy_and_tombstones():
    from phymeo.solvers.tangent_history import StationKey, TangentHistory

    record, F = _sphere_record()
    a = _sample(record, 1, F)
    book = TangentHistory(1, max_frame_turn_rad=0.3)
    initial = book.initial_state()
    trial = book.prepare(initial, [a], sample_time_s=0.0, dt_s=H)
    state = book.commit(initial, trial, {a.key: _update()})
    end = book.prepare(state, [replace(a, normal_force_n=0.0)], sample_time_s=H, dt_s=H)
    assert end.keys == () and end.ended[0].reason == "zero_load"
    assert end.dropped_spring_energy_j == 1 / 64
    empty = book.commit(state, end, {})
    assert empty.active == () and len(empty.generations) == 1
    fresh = book.prepare(empty, [a], sample_time_s=2 * H, dt_s=H)
    assert fresh.stations[0].identity.generation == 2
    np.testing.assert_array_equal(fresh.elastic_tangent_m, [[0.0, 0.0]])
    with pytest.raises(ValueError, match="capacity"):
        book.prepare(empty, [replace(a, key=StationKey(7, 2))], sample_time_s=2 * H, dt_s=H)
    assert empty.generations[0].generation == 1


@pytest.mark.parametrize(
    "change,reason",
    [
        ("face", "feature_changed"),
        ("topology", "topology_changed"),
        ("material", "material_changed"),
        ("owner", "owner_changed"),
    ],
)
def test_unsupported_continuity_resets_and_reports_discarded_spring(change, reason):
    from phymeo.constraints.collision_shapes import CollisionKey
    from phymeo.solvers.tangent_history import TangentHistory

    record, F = _sphere_record()
    a = _sample(record, 1, F)
    book = TangentHistory(1, max_frame_turn_rad=0.3)
    initial = book.initial_state()
    trial = book.prepare(initial, [a], sample_time_s=0.0, dt_s=H)
    state = book.commit(initial, trial, {a.key: _update()})
    if change == "face":
        first = replace(a.first, feature=(99,), weights=((99, 1.0),))
    elif change == "topology":
        first = replace(a.first, topology_revision=1)
    elif change == "material":
        first = replace(a.first, local_point_m=(1 / 64, 0.0, 1 / 8))
    else:
        first = replace(a.first, owner=replace(a.first.owner, body=5), shape=CollisionKey(7, 5, 0))
    altered = replace(a, first=first)
    reset = book.prepare(state, [altered], sample_time_s=H, dt_s=H)
    assert reset.stations[0].continuity == "reset"
    assert reset.stations[0].identity.generation == 2
    assert reset.ended[0].reason == reason and reset.dropped_spring_energy_j == 1 / 64
    np.testing.assert_array_equal(reset.elastic_tangent_m, [[0.0, 0.0]])
    assert state.active[0].identity.generation == 1


def test_real_deformable_weights_require_declared_same_chart_not_just_a_token():
    from phymeo.solvers.tangent_history import TangentHistory

    make = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    make.add_sphere(4, 1 / 8, mass=1.0, position=(0.0, 0.0, 7 / 32))
    make.add_sheet(
        9, [(-1, -1, 0), (1, -1, 0), (0, 1, 0)], [(0, 1, 2)], node_masses=1.0, edge_compliance=None
    )
    make.pair(4, 9, NormalLayer(1 / 8, 64.0))
    with make.build() as scene:
        scene.step(H)
        (record,) = scene.contact_events.active
    F = np.diag([1.0, -1.0, -1.0])
    a = _sample(record, 8, F, chart_id=17)
    assert a.second.weights == ((0, 0.25), (1, 0.25), (2, 0.5))
    assert a.second.owner.kind == "particle" and a.second.local_point_m is None
    book = TangentHistory(1, max_frame_turn_rad=0.3)
    initial = book.initial_state()
    trial = book.prepare(initial, [a], sample_time_s=0.0, dt_s=H)
    state = book.commit(initial, trial, {a.key: _update()})
    # These are valid barycentric points on the same declared finite triangle.
    changed = replace(a, second=replace(a.second, weights=((0, 0.125), (1, 0.375), (2, 0.5))))
    trial = book.prepare(state, [changed], sample_time_s=H, dt_s=H)
    assert trial.stations[0].continuity == "same_chart"
    np.testing.assert_array_equal(trial.elastic_tangent_m, [[1 / 16, 0]])
    assert trial.dropped_spring_energy_j == 0.0
    # No displacement, force, work or slip is inferred from closest-point weight drift.
    assert trial.stations[0].prior.tangent_work_j == -1 / 128
    assert trial.stations[0].prior.plastic_slip_m == 1 / 32
    undeclared = replace(changed, second=replace(changed.second, chart_id=None))
    reset = book.prepare(state, [undeclared], sample_time_s=H, dt_s=H)
    assert reset.stations[0].continuity == "reset" and reset.dropped_spring_energy_j == 1 / 64


@pytest.mark.parametrize("bad", ["reflection", "normal_mismatch", "large_turn", "wrong_world"])
def test_invalid_frame_or_world_rejects_the_trial_without_consuming_state(bad):
    from phymeo.solvers.tangent_history import StationKey, TangentHistory

    record, F = _sphere_record()
    a = _sample(record, 1, F)
    book = TangentHistory(1, max_frame_turn_rad=0.3)
    initial = book.initial_state()
    trial = book.prepare(initial, [a], sample_time_s=0.0, dt_s=H)
    state = book.commit(initial, trial, {a.key: _update()})
    if bad == "reflection":
        changed = replace(a, frame_world=np.diag([-1.0, 1.0, 1.0]))
    elif bad == "normal_mismatch":
        changed = replace(a, normal=(1.0, 0.0, 0.0))
    elif bad == "large_turn":
        changed = replace(a, frame_world=np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    else:
        changed = replace(a, key=StationKey(99, 1))
    with pytest.raises(ValueError):
        book.prepare(state, [changed], sample_time_s=H, dt_s=H)
    retry = book.prepare(state, [a], sample_time_s=H, dt_s=H)
    np.testing.assert_array_equal(retry.elastic_tangent_m, [[1 / 16, 0]])
    assert state.step_index == 1


def test_invalid_candidate_update_is_atomic_owned_and_retryable():
    from phymeo.solvers.tangent_history import TangentHistory

    record, F = _sphere_record()
    a = _sample(record, 1, F)
    book = TangentHistory(1, max_frame_turn_rad=0.3)
    initial = book.initial_state()
    trial = book.prepare(initial, [a], sample_time_s=0.0, dt_s=H)
    update = _update()
    for bad in (
        replace(update, stored_energy_j=1.0),
        replace(update, elastic_displacement_m=(0.0, 0.0, 1 / 16)),
        replace(update, tangent_work_j=float("nan")),
    ):
        with pytest.raises(ValueError):
            book.commit(initial, trial, {a.key: bad})
        assert initial.active == () and initial.step_index == 0
    other = TangentHistory(1, max_frame_turn_rad=0.3)
    with pytest.raises(ValueError, match="owner|base"):
        other.commit(initial, trial, {a.key: update})
    with pytest.raises(ValueError, match="time"):
        book.prepare(initial, [a], sample_time_s=H, dt_s=H)
    with pytest.raises(ValueError, match="updates|keys|station"):
        book.commit(initial, trial, {})
    with pytest.raises(ValueError, match="time"):
        book.commit(initial, replace(trial, commit_time_s=2 * H), {a.key: update})
    result = book.commit(initial, trial, {a.key: update})
    assert result == book.commit(initial, trial, {a.key: update})
    for array in (
        trial.frames_world,
        trial.normal_force_n,
        trial.elastic_displacement_m,
        trial.elastic_tangent_m,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    assert result.active[0].sample.first == a.first


def test_actual_robot_pad_materials_use_declared_distinct_tokens_after_reordering(tmp_path):
    from test_robot_contact_events import _twisting_pad_scene

    from phymeo.runtime.robot_scene import RobotCommand
    from phymeo.solvers.tangent_history import TangentHistory

    make, _, _, _ = _twisting_pad_scene(tmp_path, spin=0.0)
    with make.build() as scene:
        frame = scene.step(1 / 10000, actions={10: RobotCommand([0.0], feedforward=[0.0])})
        (record,) = scene.contact_events.active
    assert len(record.components) == len(frame.loads.contacts) == 2
    samples = []
    # The two authored pad declarations get tokens 101/202 once, before any
    # permutation. No component index or closest point is used to rematch them.
    for row, token, declared_y in ((0, 101, 0.009), (1, 202, -0.009)):
        component = record.components[row]
        assert component.material[0].first.local_point_m[1] == pytest.approx(declared_y)
        n = np.asarray(component.load.normal)
        tangent = np.array([0.0, 0.0, 1.0])
        F = np.column_stack((tangent, np.cross(n, tangent), n))
        samples.append(_sample(record, token, F, component=row))
    book = TangentHistory(2, max_frame_turn_rad=0.3)
    initial = book.initial_state()
    trial = book.prepare(initial, samples, sample_time_s=0.0, dt_s=H)
    updates = {s.key: _update((0, 0, 1 / (64 if i == 0 else 128))) for i, s in enumerate(samples)}
    state = book.commit(initial, trial, updates)
    repeated = book.prepare(state, samples[::-1], sample_time_s=H, dt_s=H)
    assert repeated.keys == tuple(s.key for s in samples)
    np.testing.assert_array_equal(repeated.elastic_tangent_m, [[1 / 64, 0], [1 / 128, 0]])
    for active in state.active:
        assert active.sample.first.owner.kind == "robot_link"
        assert active.sample.first.owner.body == 10
        assert active.sample.first.owner.link_name == "pad_link"
        assert active.sample.first.shape.body != active.sample.first.owner.body
        assert active.sample.second.owner.body == 20
        assert len(active.sample.first.weights) == 4
