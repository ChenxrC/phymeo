"""Frozen HOST staging/acceptance tests; normal geometry is supplied by the caller."""

import importlib
from dataclasses import replace

import numpy as np
import pytest

from phymeo.constraints.collision_shapes import CollisionKey
from phymeo.core.capabilities import SolverFailure
from phymeo.core.rigid import RigidBodyInputs, RigidBodyModel, RigidBodyState
from phymeo.dynamics.rotation import quaternion_matrix
from phymeo.runtime.contact_events import ContactOwner
from phymeo.solvers.rigid_semi_implicit import RigidSemiImplicitSolver
from phymeo.solvers.tangent_history import (
    MaterialAddress,
    StationKey,
    StationSample,
    TangentHistory,
    TangentUpdate,
)


def api():
    return importlib.import_module("phymeo.runtime._coupled_tangent_step")


def sample(token=7, normal=4.0, frame=None):
    frame = np.eye(3) if frame is None else frame
    addresses = tuple(
        MaterialAddress(
            ContactOwner(0, body, "rigid", "rigid_com"), CollisionKey(0, body, 0),
            0, (0,), ((0, 1.0),), (0.0, 0.0, 0.0),
        )
        for body in (1, 2)
    )
    return StationSample(StationKey(0, token), *addresses, frame[:, 2], frame, normal)


def prepare(store, history, samples, W, free, *, kt=None, mu=None, h=.1):
    n = len(samples)
    return api().prepare_tangent_step(
        store, history, samples, np.asarray(W, dtype=float), np.asarray(free, dtype=float),
        sample_time_s=history.commit_time_s, dt_s=h,
        stiffness_n_per_m=np.full(n, 100.0) if kt is None else np.asarray(kt, dtype=float),
        friction=np.full(n, .5) if mu is None else np.asarray(mu, dtype=float),
    )


def finish(store, history, trial, material=None, application=None, *, m=1e-12, j=1e-12):
    material = trial.predicted_displacement_m if material is None else material
    application = material if application is None else application
    return api().finish_tangent_step(
        store, history, trial, np.asarray(material, dtype=float), np.asarray(application, dtype=float),
        motion_tolerance_m=m, energy_tolerance_j=j,
    )


def two_stations():
    store = TangentHistory(4)
    history = store.initial_state()
    samples = (sample(7), sample(19))
    W = np.zeros((6, 6))
    W[np.ix_([0, 3], [0, 3])] = 1.0
    free = np.array([[.3, 0, 0], [.3, 0, 0]])
    return store, history, samples, W, free


def preloaded():
    store = TangentHistory(2)
    initial = store.initial_state()
    s = sample()
    trial = store.prepare(initial, [s], sample_time_s=0.0, dt_s=.1)
    update = TangentUpdate((.02, 0.0, 0.0), .02, 0.0, 0.0, "stick", 100.0)
    return store, store.commit(initial, trial, {s.key: update})


def test_two_station_full_world_response_is_projected_without_losing_cross_blocks():
    store, history, samples, W, free = two_stations()
    trial = prepare(store, history, samples, W, free)
    np.testing.assert_allclose(trial.force_on_second_n, [[-1, 0, 0]]*2, atol=1e-12)
    np.testing.assert_allclose(trial.predicted_displacement_m, [[.01, 0, 0]]*2, atol=1e-12)
    assert trial.history_trial.keys == tuple(s.key for s in samples)
    candidate, receipt = finish(store, history, trial)
    assert receipt.accepted and abs(receipt.energy_balance_j) < 1e-12
    np.testing.assert_allclose(receipt.application_work_j, [-.01, -.01], atol=1e-12)
    assert [row.update.stored_energy_j for row in candidate.active] == pytest.approx([.005, .005])
    assert [row.update.tangent_work_j for row in candidate.active] == pytest.approx([-.01, -.01])
    assert receipt.released_energy_j == 0.0
    assert history.active == () and history.step_index == 0


def test_noncanonical_sample_order_is_rejected_before_matrix_rows_can_be_misbound():
    store, history, samples, W, free = two_stations()
    with pytest.raises(ValueError, match="canonical|order"):
        prepare(store, history, samples[::-1], W, free)
    assert history.active == ()


def test_zero_normal_middle_row_keeps_original_world_force_and_history_addresses():
    store = TangentHistory(3)
    history = store.initial_state()
    samples = (sample(2, 8.0), sample(4, 0.0), sample(9, 8.0))
    trial = prepare(store, history, samples, np.eye(9), [[.2, 0, 0], [.7, 0, 0], [.6, 0, 0]])
    assert trial.history_trial.keys == (samples[0].key, samples[2].key)
    np.testing.assert_allclose(trial.force_on_second_n, [[-1, 0, 0], [0, 0, 0], [-3, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(trial.predicted_displacement_m, [[.01, 0, 0], [.07, 0, 0], [.03, 0, 0]], atol=1e-12)
    state, _ = finish(store, history, trial)
    assert tuple(r.identity.key for r in state.active) == (samples[0].key, samples[2].key)


def test_world_rotation_keeps_force_work_and_normal_motion_components():
    store, history, samples, W, free = two_stations()
    R = np.array([[0., 0, 1], [1, 0, 0], [0, 1, 0]])
    B = np.kron(np.eye(2), R)
    free[:, 2] = .2
    trial = prepare(store, history, tuple(sample(s.key.token, frame=R) for s in samples),
                    B @ W @ B.T, free @ R.T)
    np.testing.assert_allclose(trial.force_on_second_n, [[0, -1, 0]]*2, atol=1e-12)
    np.testing.assert_allclose(trial.predicted_displacement_m, [[.02, .01, 0]]*2, atol=1e-12)
    state, receipt = finish(store, history, trial)
    np.testing.assert_allclose(state.active[0].update.elastic_displacement_m, [0, .01, 0], atol=1e-12)
    np.testing.assert_allclose(receipt.application_work_j, [-.01, -.01], atol=1e-12)


def test_spring_unloading_may_do_positive_work_with_a_balanced_receipt():
    store, history = preloaded()
    trial = prepare(store, history, [sample(normal=1.0)], np.eye(3), [[0, 0, 0]])
    state, receipt = finish(store, history, trial)
    np.testing.assert_allclose(trial.force_on_second_n, [[-.5, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(receipt.application_work_j, [.0025], atol=1e-12)
    assert state.active[0].update.tangent_work_j == pytest.approx(.0025)
    assert abs(receipt.energy_balance_j) < 1e-12


@pytest.mark.parametrize("absent", [False, True])
def test_unload_empty_step_and_reentry_release_the_old_spring_exactly_once(absent):
    store, history = preloaded()
    samples = () if absent else (sample(normal=0.0),)
    n = len(samples)
    trial = prepare(store, history, samples, np.eye(3*n), np.zeros((n, 3)))
    assert trial.result is None and trial.force_on_second_n.shape == (n, 3)
    released, receipt = finish(store, history, trial)
    assert released.active == () and receipt.released_energy_j == pytest.approx(.02)
    assert abs(receipt.energy_balance_j) < 1e-12
    empty = prepare(store, released, (), np.zeros((0, 0)), np.zeros((0, 3)))
    empty_state, empty_receipt = finish(store, released, empty)
    assert empty_state.commit_time_s == pytest.approx(.3)
    assert empty_receipt.released_energy_j == 0.0
    again = prepare(store, empty_state, [sample()], np.eye(3), [[0, 0, 0]])
    next_state, _ = finish(store, empty_state, again)
    np.testing.assert_array_equal(again.force_on_second_n, [[0, 0, 0]])
    assert next_state.active[0].identity.generation == 2
    assert next_state.active[0].update.stored_energy_j == 0.0


def test_zero_friction_loaded_station_releases_energy_without_a_second_history_release():
    store, history = preloaded()
    trial = prepare(store, history, [sample()], np.eye(3), [[.1, 0, 0]], mu=[0])
    state, receipt = finish(store, history, trial)
    assert trial.result.regime == ("frictionless",)
    assert trial.history_trial.dropped_spring_energy_j == 0.0
    assert receipt.released_energy_j == pytest.approx(.02)
    assert state.active[0].update.regime == "frictionless"
    assert state.active[0].update.stored_energy_j == 0.0


@pytest.mark.parametrize("material_error,application_error", [(1e-3, 0), (0, 1e-3), (4e-7, -4e-7)])
def test_material_and_application_motion_each_and_their_difference_must_pass(material_error, application_error):
    store, history, samples, W, free = two_stations()
    trial = prepare(store, history, samples, W, free)
    material, application = trial.predicted_displacement_m.copy(), trial.predicted_displacement_m.copy()
    material[0, 0] += material_error
    application[0, 0] += application_error
    with pytest.raises(SolverFailure, match="motion|displacement"):
        finish(store, history, trial, material, application, m=5e-7, j=1)
    assert history.active == ()
    _, retry = finish(store, history, trial)
    assert retry.accepted


@pytest.mark.parametrize("cancel", [False, True])
def test_energy_gate_rejects_small_motion_error_even_when_total_work_can_cancel(cancel):
    store, history, samples, W, free = two_stations()
    trial = prepare(store, history, samples, W, free)
    actual = trial.predicted_displacement_m.copy()
    actual[:, 0] += [1e-5, -1e-5 if cancel else 1e-5]
    # The declared motion gate passes. Per-station physical work still fails.
    with pytest.raises(SolverFailure, match="energy|work"):
        finish(store, history, trial, actual, actual, m=2e-5, j=1e-8)
    assert history.step_index == 0


def test_actual_rigid_exp_map_is_checked_before_history_can_be_committed():
    store = TangentHistory(1)
    history = store.initial_state()
    h = 1/8
    W = np.array([[2., 1, 0], [1, 2, 0], [0, 0, 3]])
    trial = prepare(store, history, [sample()], W, [[1/8, -1/4, 0]], kt=[64], h=h)
    initial = RigidBodyState(np.zeros((1, 3)), np.array([[0., 0, 0, 1]]),
                             np.array([[1/8, -1/4, 0.]]), np.zeros((1, 3)))
    model = RigidBodyModel(np.ones(1), np.ones((1, 3)), np.zeros(3))
    point = np.array([1., -1, 0])
    force = trial.force_on_second_n[0]
    result = RigidSemiImplicitSolver().step(
        model, initial, RigidBodyInputs(force[None, :], np.cross(point, force)[None, :]), h,
    )
    assert result.report.accepted
    actual = (result.state.position[0] + quaternion_matrix(result.state.orientation)[0] @ point - point)[None, :]
    with pytest.raises(SolverFailure, match="motion|displacement"):
        finish(store, history, trial, actual, actual)
    assert history.active == ()
    np.testing.assert_array_equal(initial.orientation, [[0, 0, 0, 1]])


def test_identical_retry_is_pure_but_later_history_or_another_store_is_rejected():
    store, history, samples, W, free = two_stations()
    trial = prepare(store, history, samples, W, free)
    state, receipt = finish(store, history, trial)
    repeated, again = finish(store, history, trial)
    assert state == repeated and receipt.energy_balance_j == again.energy_balance_j
    with pytest.raises(ValueError, match="stale|base|owner"):
        finish(store, state, trial)
    foreign = TangentHistory(4)
    with pytest.raises(ValueError, match="stale|base|owner"):
        finish(foreign, foreign.initial_state(), trial)


@pytest.mark.parametrize("kind", ["forces", "result", "force_metadata", "result_metadata", "history"])
def test_replaced_or_metadata_mutated_trial_is_not_an_accepted_receipt(kind):
    store, history, samples, W, free = two_stations()
    trial = prepare(store, history, samples, W, free)
    actual = trial.predicted_displacement_m.copy()
    if kind == "forces":
        trial = replace(trial, force_on_second_n=np.zeros((2, 3)))
    elif kind == "result":
        trial = replace(trial, result=replace(trial.result, force_n=np.zeros((2, 2))))
    elif kind == "force_metadata":
        trial.force_on_second_n.shape = (6,)
    elif kind == "result_metadata":
        trial.result.force_n.shape = (4,)
    else:
        trial = replace(trial, history_trial=replace(trial.history_trial, commit_time_s=10.0))
    with pytest.raises(ValueError, match="trial|prepared|binding"):
        finish(store, history, trial, actual, actual)
    assert history.active == ()


def test_entry_arrays_are_copied_and_receipt_arrays_have_owned_read_only_storage():
    store, history, samples, W, free = two_stations()
    stiffness, friction = np.full(2, 100.), np.full(2, .5)
    trial = prepare(store, history, samples, W, free, kt=stiffness, mu=friction)
    W[:] = 99
    free[:] = 99
    stiffness[:] = 99
    friction[:] = 99
    actual = trial.predicted_displacement_m.copy()
    _, receipt = finish(store, history, trial, actual, actual)
    actual[:] = 99
    np.testing.assert_allclose(trial.force_on_second_n, [[-1, 0, 0]]*2, atol=1e-12)
    np.testing.assert_allclose(receipt.application_work_j, [-.01, -.01], atol=1e-12)
    for value in (trial.force_on_second_n, trial.predicted_displacement_m, receipt.application_work_j):
        with pytest.raises(ValueError):
            value.setflags(write=True)


@pytest.mark.parametrize("change", ["asymmetric", "negative_normal_mobility", "zero_stiffness", "negative_mu", "nan_free"])
def test_invalid_world_inputs_are_rejected_before_tangent_projection_can_hide_them(change):
    store = TangentHistory(1)
    history = store.initial_state()
    W, free, kt, mu = np.eye(3), np.array([[.1, 0, 0]]), [100.], [.5]
    if change == "asymmetric":
        W[0, 2] = .1
    elif change == "negative_normal_mobility":
        W[2, 2] = -1
    elif change == "zero_stiffness":
        kt = [0.]
    elif change == "negative_mu":
        mu = [-.5]
    else:
        free[0, 0] = np.nan
    with pytest.raises((ValueError, TypeError)):
        prepare(store, history, [sample()], W, free, kt=kt, mu=mu)
    assert history.active == ()


def test_changing_stiffness_cannot_silently_revalue_an_existing_spring():
    store, history = preloaded()
    with pytest.raises(ValueError, match="stiffness|energy"):
        prepare(store, history, [sample()], np.eye(3), [[0, 0, 0]], kt=[200.])


@pytest.mark.parametrize("m,j", [(0, 1e-12), (float("inf"), 1e-12), (1e-12, 0), (1e-12, float("nan"))])
def test_motion_and_energy_tolerances_are_explicit_finite_positive_bounds(m, j):
    store, history, samples, W, free = two_stations()
    trial = prepare(store, history, samples, W, free)
    with pytest.raises(ValueError):
        finish(store, history, trial, m=m, j=j)