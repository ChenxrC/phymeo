"""Tests-first: explicit force-application coordinates do not redefine material slip.

UNEXECUTED authoring; the parent owns all project execution. Frozen public gates
are 1e-6 m / 1e-8 J. No complete grasp window is repeated here.
"""

import numpy as np
import pytest

from phymeo import ContactSceneBuilder, NormalLayer
from phymeo.constraints.collision_shapes import CollisionKey
from phymeo.runtime.material_stations import (
    MaterialStationRegistration,
    MaterialStationRegistry,
)
from phymeo.solvers.tangent_history import StationKey


MOTION_M = 1e-6
ENERGY_J = 1e-8


def _clamp():
    builder = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_box(10, (0.3, 0.3, 0.1), mass=None, position=(0, 0, -0.2))
    builder.add_box(20, (0.1, 0.1, 0.05), mass=1.0)
    builder.add_box(30, (0.3, 0.3, 0.1), mass=None, position=(0, 0, 0.2))
    layer = NormalLayer(0.06, 200.0, 0.2, 0.5)
    builder.pair(10, 20, layer)
    builder.pair(20, 30, layer)
    builder.pair(10, 30, None)
    return builder


def _clamp_registry():
    # The two real finite square patches each have twelve area quadrature sites.
    # Full face IDs are declared: closest-point sparse weights are not identity.
    corners = ((-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1))
    anchors = []
    for index, left in enumerate(corners):
        right = corners[(index + 1) % 4]
        for wl, wr in ((1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)):
            anchors.append((wl * left[0] + wr * right[0], wl * left[1] + wr * right[1]))
    declarations = []
    for side, (first, second, z) in enumerate(((10, 20, -0.05), (20, 30, 0.05))):
        for index, (x, y) in enumerate(anchors):
            declarations.append(
                MaterialStationRegistration(
                    StationKey(0, side * 12 + index),
                    CollisionKey(0, first, 0),
                    CollisionKey(0, second, 0),
                    "second" if side == 0 else "first",
                    (x, y, z),
                    (1, 0, 0),
                    side * 2,
                    side * 2 + 1,
                    first_feature=(1, 3, 5, 7),
                    second_feature=(0, 2, 4, 6),
                )
            )
    return MaterialStationRegistry(declarations)


def test_a01_public_contact_scene_accepts_application_reference_after_real_normal_sample():
    # RED must reach the new keyword after the unchanged real normal law works.
    with _clamp().build() as scene:
        initial = scene.state
        events = scene.contact_events
        loads = scene.sample().loads
        assert len(loads.contacts) == 2
        assert [load.force_n for load in loads.contacts] == pytest.approx([2.0, 2.0], abs=1e-10)
        assert sum(len(load.sites) for load in loads.contacts) == 24
        row = scene.rigid_rows[0, 20]
        np.testing.assert_allclose(loads.inputs.rigid.force[row], 0, atol=1e-12, rtol=0)
        np.testing.assert_allclose(loads.inputs.rigid.torque[row], 0, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(scene.state.rigid.position, initial.rigid.position)
        assert scene.contact_events == events
        assert scene.tangent_history is None
        assert scene.last_frame is None

        scene.enable_coupled_tangent(
            _clamp_registry(),
            stiffness_n_per_m=200.0,
            motion_reference="application",
        )

        external = scene.inputs()
        external.rigid.force[row, 0] = 0.5
        frame = scene.step(1 / 1024, external)
        receipt = scene.tangent_receipt
        assert receipt.accepted
        assert receipt.motion_reference == "application"
        assert receipt.application_motion_error_m <= MOTION_M
        assert abs(receipt.energy_residual_j) <= ENERGY_J
        assert len(scene.tangent_history.active) == 24
        assert {record.update.motion_reference for record in scene.tangent_history.active} == {
            "application"
        }
        np.testing.assert_allclose(
            frame.state.rigid.velocity[row] - initial.rigid.velocity[row],
            (frame.loads.inputs.rigid.force[row] + external.rigid.force[row]) / 1024,
            atol=2e-13,
            rtol=0,
        )



def _example():
    import sys
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    name = "phymeo_application_reference_example"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "examples" / "general_scene_friction.py"
        spec = spec_from_file_location(name, path)
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _public(value):
    return _example().public_payload(value)


def _seven(scene):
    return _public(
        dict(
            state=scene.state,
            reading=scene.observe(),
            geometry=scene.geometry(),
            history=scene.tangent_history,
            receipt=scene.tangent_receipt,
            events=scene.contact_events,
            last_frame=scene.last_frame,
        )
    )


def _enable(scene, registry, mode="application"):
    scene.enable_coupled_tangent(
        registry,
        stiffness_n_per_m=200.0,
        motion_reference=mode,
    )


_DEFAULT = object()


def _trial(store, history, samples, *, mode=_DEFAULT, free=None, mobility=None, mu=0.5):
    from phymeo.runtime._coupled_tangent_step import prepare_tangent_step

    n = len(samples)
    return prepare_tangent_step(
        store,
        history,
        samples,
        np.zeros((3 * n, 3 * n)) if mobility is None else mobility,
        np.zeros((n, 3)) if free is None else np.asarray(free, dtype=float),
        sample_time_s=history.commit_time_s,
        dt_s=1 / 1024,
        stiffness_n_per_m=np.full(n, 200.0),
        friction=np.full(n, mu),
        **({} if mode is _DEFAULT else {"motion_reference": mode}),
    )


def _finish(store, history, trial, material=None, application=None):
    from phymeo.runtime._coupled_tangent_step import finish_tangent_step

    application = trial.predicted_displacement_m if application is None else application
    material = application if material is None else material
    return finish_tangent_step(
        store,
        history,
        trial,
        material,
        application,
        motion_tolerance_m=MOTION_M,
        energy_tolerance_j=ENERGY_J,
    )


@pytest.fixture(scope="module")
def rotor_entry(tmp_path_factory):
    """Actual original F05 2 N finite-face entry, sampled by one physical step.

    The returned poses are the declared zero-angle entry, not the end geometry.
    Geometry-only tests may transport these owned samples without claiming a
    dynamic scene accepted those manufactured poses.
    """
    from phymeo.runtime.material_stations import OwnerPose

    example = _example()
    with example.make_rotating_tool(tmp_path_factory.mktemp("application-anchor-entry")) as case:
        loads = case.scene.sample()
        assert sum(load.force_n for load in loads.contacts) == pytest.approx(2, abs=1e-9)
        assert sum(len(load.sites) for load in loads.contacts) == 12
        case.scene.step(case.parameters["dt_s"], actions=case.actions)
        events = case.scene.contact_events
        positions = {
            example.LOWER_TOOL: (0.0, 0.0, 1 / 32 - 1 / 1024),
            example.BOX: (0.0, 0.0, -0.05),
        }
        owners = {owner for record in events.active for owner in (record.first_owner, record.second_owner)}
        frames = {owner: OwnerPose(owner, positions[owner.body], np.eye(3)) for owner in owners}
        selected = case.registry.bind(events, frames, sample_time_s=0.0)
        assert len(selected.samples) == 12
        assert sum(sample.normal_force_n for sample in selected.samples) == pytest.approx(
            2, abs=1e-9
        )
        for binding in selected.bindings:
            first = np.asarray(frames[binding.first_owner].position_m) + binding.first_material_local_m
            second = np.asarray(frames[binding.second_owner].position_m) + binding.second_material_local_m
            np.testing.assert_allclose(second - first, (0, 0, 1 / 1024), atol=2e-14, rtol=0)
            np.testing.assert_allclose(
                binding.application_point_first_m,
                binding.application_point_second_m,
                atol=2e-14,
                rtol=0,
            )
        return selected, frames


def test_a02_pure_translation_keeps_default_material_and_application_physics_identical():
    from dataclasses import fields

    saved = []
    for mode in (None, "material", "application"):
        with _clamp().build() as scene:
            if mode is None:
                scene.enable_coupled_tangent(_clamp_registry(), stiffness_n_per_m=200.0)
            else:
                _enable(scene, _clamp_registry(), mode)
            force = scene.inputs()
            force.rigid.force[scene.rigid_rows[0, 20], 0] = 0.5
            for _ in range(3):
                frame = scene.step(1 / 1024, force)
            receipt = scene.tangent_receipt
            assert receipt.motion_reference == ("material" if mode is None else mode)
            saved.append(
                (
                    _public((scene.state, frame, scene.geometry(), scene.contact_events)),
                    scene.tangent_history,
                    receipt,
                )
            )
    for physics, history, receipt in saved[1:]:
        assert physics == saved[0][0]
        original = saved[0][1]
        assert history.step_index == original.step_index == 3
        assert history.commit_time_s == original.commit_time_s
        for old, current in zip(original.active, history.active, strict=True):
            assert _public(old.identity) == _public(current.identity)
            assert _public(old.sample) == _public(current.sample)
            for field in fields(old.update):
                if field.name != "motion_reference":
                    assert _public(getattr(old.update, field.name)) == _public(
                        getattr(current.update, field.name)
                    )
        for field in fields(receipt):
            if field.name != "motion_reference":
                assert _public(getattr(receipt, field.name)) == _public(
                    getattr(saved[0][2], field.name)
                )


def test_b01_penetrated_corotation_separates_surface_motion_without_inventing_anchor_work(
    rotor_entry,
):
    from phymeo.runtime.material_stations import OwnerPose
    from phymeo.solvers.tangent_history import TangentHistory

    selection, entry = rotor_entry
    angle = 1 / 64
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    shift = np.array([1 / 8, -1 / 16, 1 / 32])
    end = {
        owner: OwnerPose(owner, rotation @ np.asarray(pose.position_m) + shift, rotation)
        for owner, pose in entry.items()
    }
    material, application = selection.measure_motion(end)
    # Surface B-A initially equals (0,0,delta), whereas both load points are p.
    # A common rigid transform therefore gives (R-I)*(0,0,delta) versus zero.
    expected = np.tile([s / 1024, 0.0, (c - 1) / 1024], (12, 1))
    np.testing.assert_allclose(material, expected, atol=2e-14, rtol=0)
    np.testing.assert_allclose(application, 0, atol=2e-14, rtol=0)
    assert np.max(np.linalg.norm(material, axis=1)) > MOTION_M

    for mode in ("material", "application"):
        store = TangentHistory(12)
        history = store.initial_state()
        trial = _trial(store, history, selection.samples, mode=mode)
        if mode == "material":
            with pytest.raises(ValueError, match="motion"):
                _finish(store, history, trial, material, application)
            assert history.active == () and history.step_index == 0
            continue
        state, receipt = _finish(store, history, trial, material, application)
        assert receipt.motion_reference == "application"
        assert receipt.material_application_error_m > MOTION_M
        assert receipt.material_motion_error_m > MOTION_M
        assert receipt.application_motion_error_m <= MOTION_M
        np.testing.assert_allclose(trial.force_on_second_n, 0, atol=1e-15, rtol=0)
        np.testing.assert_allclose(receipt.actual_material_displacement_m, expected, atol=2e-14)
        np.testing.assert_array_equal(receipt.actual_application_displacement_m, application)
        assert receipt.actual_work_j == 0.0 and receipt.energy_residual_j == 0.0
        assert all(record.update.stored_energy_j == 0.0 for record in state.active)
        assert all(record.update.plastic_slip_m == 0.0 for record in state.active)


def test_b02_application_work_and_surface_work_remain_separate_observations(rotor_entry):
    from phymeo.solvers.tangent_history import TangentHistory

    selection, _ = rotor_entry
    store = TangentHistory(1)
    history = store.initial_state()
    trial = _trial(
        store,
        history,
        selection.samples[:1],
        mode="application",
        free=[[1 / 16, 0, 0]],
    )
    application = np.array([[1 / 16384, 0, 0]])
    material = application + np.array([[1 / 1024, 0, 0]])
    # One prescribed-motion spring, below mu*N: f=-200*d, E=100*d^2.
    np.testing.assert_allclose(trial.force_on_second_n, -200 * application, atol=2e-14, rtol=0)
    state, receipt = _finish(store, history, trial, material, application)
    expected_app_work = -200 * (1 / 16384) ** 2
    expected_material_work = -200 * (1 / 16384) * (1 / 16384 + 1 / 1024)
    assert receipt.actual_work_j == pytest.approx(expected_app_work, abs=1e-16)
    assert receipt.application_work_j[0] == pytest.approx(expected_app_work, abs=1e-16)
    assert receipt.material_work_j[0] == pytest.approx(expected_material_work, abs=1e-16)
    assert abs(receipt.material_work_j[0] - receipt.application_work_j[0]) > ENERGY_J
    assert abs(receipt.energy_residual_j) <= ENERGY_J
    assert state.active[0].update.tangent_work_j == pytest.approx(expected_app_work, abs=1e-16)
    assert state.active[0].update.motion_reference == "application"


@pytest.mark.parametrize("gate", ["per_station", "total"])
def test_b03_application_mode_keeps_both_original_energy_gates(rotor_entry, gate):
    from phymeo.solvers.tangent_history import TangentHistory

    selection, _ = rotor_entry
    store = TangentHistory(2)
    history = store.initial_state()
    trial = _trial(
        store,
        history,
        selection.samples[:2],
        mode="application",
        free=[[1 / 16, 0, 0], [1 / 16, 0, 0]],
    )
    # |f|=200/16384. Opposite .95 um errors cancel globally but each exceeds
    # 1e-8 J; equal .75 um errors each pass yet their total exceeds 1e-8 J.
    error = np.zeros((2, 3))
    error[:, 0] = (0.95e-6, -0.95e-6) if gate == "per_station" else (0.75e-6, 0.75e-6)
    application = trial.predicted_displacement_m + error
    assert np.max(np.linalg.norm(error, axis=1)) < MOTION_M
    residual = (-200 / 16384) * error[:, 0]
    if gate == "per_station":
        assert np.min(np.abs(residual)) > ENERGY_J
        assert abs(sum(residual)) < ENERGY_J
    else:
        assert np.max(np.abs(residual)) < ENERGY_J
        assert abs(sum(residual)) > ENERGY_J
    with pytest.raises(ValueError, match="energy"):
        _finish(store, history, trial, application, application)
    assert history.active == () and history.step_index == 0


def test_b04_zero_friction_keeps_zero_force_energy_and_work_despite_surface_anchor_difference(
    rotor_entry,
):
    from phymeo.solvers.tangent_history import TangentHistory

    selection, _ = rotor_entry
    store = TangentHistory(1)
    history = store.initial_state()
    trial = _trial(
        store,
        history,
        selection.samples[:1],
        mode="application",
        free=[[0.25, 0, 0]],
        mu=0.0,
    )
    app = trial.predicted_displacement_m
    material = app + np.array([[1 / 1024, 0, 0]])
    state, receipt = _finish(store, history, trial, material, app)
    np.testing.assert_array_equal(trial.force_on_second_n, np.zeros((1, 3)))
    assert trial.result.regime == ("frictionless",)
    assert state.active[0].update.stored_energy_j == 0
    assert state.active[0].update.plastic_slip_m == 0
    assert receipt.actual_work_j == receipt.released_energy_j == receipt.energy_residual_j == 0
    np.testing.assert_array_equal(receipt.application_work_j, [0.0])
    np.testing.assert_array_equal(receipt.material_work_j, [0.0])
    assert receipt.material_application_error_m > MOTION_M
    # Zero friction does not bypass the frozen application-domain check.
    with pytest.raises(ValueError, match="motion"):
        _finish(store, history, trial, material, app + [[2 * MOTION_M, 0, 0]])


def _seed(sample, *, mode="application", strain=1 / 4096):
    from phymeo.solvers.tangent_history import TangentHistory, TangentUpdate

    store = TangentHistory(1)
    initial = store.initial_state()
    prepared = store.prepare(initial, (sample,), sample_time_s=0.0, dt_s=1 / 1024)
    update = TangentUpdate(
        (strain, 0, 0), 100 * strain * strain, 0.0, 0.0, "stick", 200.0, mode
    )
    state = store.commit(initial, prepared, {sample.key: update})
    return store, state


@pytest.mark.parametrize("absent", [False, True])
def test_c01_unload_reentry_releases_once_and_allows_new_generation_to_choose_mode(
    rotor_entry, absent
):
    from dataclasses import replace

    selection, _ = rotor_entry
    sample = selection.samples[0]
    store, history = _seed(sample)
    old_energy = 100 * (1 / 4096) ** 2
    samples = () if absent else (replace(sample, normal_force_n=0.0),)
    trial = _trial(store, history, samples, mode="application")
    ended, receipt = _finish(store, history, trial)
    assert ended.active == ()
    assert receipt.released_energy_j == pytest.approx(old_energy, abs=1e-18)
    again = _trial(store, ended, (), mode="application")
    empty, second = _finish(store, ended, again)
    assert second.released_energy_j == 0.0
    entry = _trial(store, empty, (sample,), mode="material")
    renewed, third = _finish(store, empty, entry)
    assert renewed.active[0].identity.generation == 2
    assert renewed.active[0].update.motion_reference == "material"
    assert renewed.active[0].update.stored_energy_j == 0.0
    assert third.released_energy_j == 0.0
    assert history.active[0].identity.generation == 1
    assert history.active[0].update.stored_energy_j == old_energy


def test_c02_zero_friction_releases_prior_anchor_energy_once(rotor_entry):
    selection, _ = rotor_entry
    sample = selection.samples[0]
    store, history = _seed(sample)
    first = _trial(store, history, (sample,), mode="application", mu=0.0)
    state, receipt = _finish(store, history, first)
    assert receipt.released_energy_j == 100 * (1 / 4096) ** 2
    assert state.active[0].update.regime == "frictionless"
    assert state.active[0].update.stored_energy_j == 0
    second = _trial(store, state, (sample,), mode="application", mu=0.0)
    same, receipt = _finish(store, state, second)
    assert receipt.released_energy_j == 0.0
    assert same.active[0].identity == state.active[0].identity


@pytest.mark.parametrize("mode", [None, True, 0, "", "APPLICATION", "application "])
def test_c03_invalid_mode_is_rejected_without_consuming_history_or_scene(rotor_entry, mode):
    from phymeo.solvers.tangent_history import TangentHistory, TangentUpdate

    sample = rotor_entry[0].samples[0]
    store = TangentHistory(1)
    state = store.initial_state()
    with pytest.raises((TypeError, ValueError)):
        _trial(store, state, (sample,), mode=mode)
    prepared = store.prepare(state, (sample,), sample_time_s=0, dt_s=1 / 1024)
    with pytest.raises((TypeError, ValueError)):
        invalid = TangentUpdate((0, 0, 0), 0, 0, 0, "stick", 200, mode)
        store.commit(state, prepared, {sample.key: invalid})
    assert state.step_index == 0 and state.active == ()
    with _clamp().build() as scene:
        position = scene.state.rigid.position.copy()
        with pytest.raises((TypeError, ValueError)):
            _enable(scene, _clamp_registry(), mode)
        assert scene.tangent_history is None
        assert scene.last_frame is None
        np.testing.assert_array_equal(scene.state.rigid.position, position)


@pytest.mark.parametrize("original", ["material", "application"])
@pytest.mark.parametrize("strain", [0.0, 1 / 4096])
def test_c04_retained_mode_cannot_change_through_prepare_or_direct_commit(
    rotor_entry, original, strain
):
    from dataclasses import replace

    sample = rotor_entry[0].samples[0]
    store, state = _seed(sample, mode=original, strain=strain)
    other = "application" if original == "material" else "material"
    with pytest.raises(ValueError, match="reference|mode"):
        _trial(store, state, (sample,), mode=other)
    prepared = store.prepare(
        state, (sample,), sample_time_s=state.commit_time_s, dt_s=1 / 1024
    )
    changed = replace(state.active[0].update, motion_reference=other)
    with pytest.raises(ValueError, match="reference|mode"):
        store.commit(state, prepared, {sample.key: changed})
    # Both rejected attempts are pure: the same original-mode trial remains usable.
    first = store.commit(state, prepared, {sample.key: state.active[0].update})
    retry = store.commit(state, prepared, {sample.key: state.active[0].update})
    assert _public(first) == _public(retry)
    assert first.active[0].identity.generation == 1
    assert first.active[0].update.motion_reference == original


@pytest.mark.parametrize("change", ["mode", "prediction"])
def test_c05_trial_mode_is_sealed_and_forgery_does_not_consume_the_valid_retry(rotor_entry, change):
    from dataclasses import replace

    from phymeo.solvers.tangent_history import TangentHistory

    store = TangentHistory(1)
    history = store.initial_state()
    trial = _trial(store, history, rotor_entry[0].samples[:1], mode="application")
    forged = (
        replace(trial, motion_reference="material")
        if change == "mode"
        else replace(trial, predicted_displacement_m=trial.predicted_displacement_m + 1e-9)
    )
    with pytest.raises(ValueError, match="bind|prepared|trial"):
        _finish(store, history, forged)
    one, a = _finish(store, history, trial)
    two, b = _finish(store, history, trial)
    assert _public(one) == _public(two)
    assert _public(a) == _public(b)
    assert history.step_index == 0


def test_c06_omitted_keyword_and_legacy_update_constructor_keep_material_mode(rotor_entry):
    from phymeo.solvers.tangent_history import TangentHistory, TangentUpdate

    store = TangentHistory(1)
    state = store.initial_state()
    trial = _trial(store, state, rotor_entry[0].samples[:1])
    candidate, receipt = _finish(store, state, trial)
    assert trial.motion_reference == receipt.motion_reference == "material"
    assert candidate.active[0].update.motion_reference == "material"
    assert TangentUpdate((0, 0, 0), 0, 0, 0, "stick", 200).motion_reference == "material"


def test_d01_actual_single_robot_rotation_keeps_original_mass_force_and_work(tmp_path):
    example = _example()
    with example.make_rotating_tool(tmp_path) as case:
        assert type(case.scene).__name__ == "RobotScene"
        assert sum(load.force_n for load in case.scene.sample().contacts) == pytest.approx(2, abs=1e-9)
        _enable(case.scene, case.registry)
        h = case.parameters["dt_s"]
        anchors = np.asarray(case.parameters["rotor_anchors"])
        angular_stiffness = 200 * float(np.sum(anchors[:, :2] ** 2))
        expected_angle = h * h * 0.01 / (0.02 + h * h * angular_stiffness)
        reading = case.scene.step(h, actions=case.actions)
        robot = reading.robots[case.world, example.LOWER_TOOL]
        assert robot.q[0] == pytest.approx(expected_angle, abs=2e-11)
        assert robot.v[0] == pytest.approx(expected_angle / h, abs=2e-11)
        assert robot.contact_effort_full[0] == pytest.approx(
            -angular_stiffness * expected_angle, abs=1e-9
        )
        c, s = np.cos(robot.q[0]), np.sin(robot.q[0])
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        # The z offsets differ but cannot affect this exact Rz relative motion.
        expected_displacement = anchors - anchors @ rotation.T
        receipt = case.scene.tangent_receipt
        np.testing.assert_allclose(
            receipt.actual_application_displacement_m, expected_displacement, atol=2e-11, rtol=0
        )
        assert receipt.motion_reference == "application" and receipt.actual_work_j < 0
        assert abs(receipt.energy_residual_j) <= ENERGY_J
        assert case.scene.tangent_history.step_index == 1


def test_d02_actual_application_curvature_rejects_and_identical_request_retries_atomically(
    tmp_path, monkeypatch
):
    from phymeo.runtime import _multi_robot_tangent as bridge

    example = _example()
    calls = []
    original = bridge.finish_tangent_step

    def capture(store, history, trial, material, application, **options):
        calls.append((application.copy(), trial.predicted_displacement_m.copy()))
        return original(store, history, trial, material, application, **options)

    monkeypatch.setattr(bridge, "finish_tangent_step", capture)
    with example.make_rotating_tool(tmp_path, tilted=True) as case:
        assert case.scene.observe().robots[case.world, example.LOWER_TOOL].v[0] == 5.0
        assert sum(load.force_n for load in case.scene.sample().contacts) == pytest.approx(2, abs=1e-9)
        _enable(case.scene, case.registry)
        before, history = _seven(case.scene), case.scene.tangent_history
        for attempt in range(2):
            with pytest.raises(ValueError, match="motion"):
                case.scene.step(1 / 128, actions=case.actions)
            assert len(calls) == attempt + 1  # The real candidate reached finish.
            application, predicted = calls[-1]
            assert np.max(np.linalg.norm(application - predicted, axis=1)) > MOTION_M
            assert _seven(case.scene) == before
            assert case.scene.tangent_history is history
        np.testing.assert_array_equal(calls[0][0], calls[1][0])
        np.testing.assert_array_equal(calls[0][1], calls[1][1])


def test_d03_failed_publication_preserves_all_seven_channels_and_valid_retry_matches(tmp_path, monkeypatch):
    from phymeo.runtime import _multi_robot_tangent as bridge

    example = _example()
    with (
        example.make_rotating_tool(tmp_path / "retry") as case,
        example.make_rotating_tool(tmp_path / "reference") as reference,
    ):
        for current in (case, reference):
            _enable(current.scene, current.registry)
            current.scene.step(current.parameters["dt_s"], actions=current.actions)
        before, history = _seven(case.scene), case.scene.tangent_history
        original = bridge.finish_tangent_step

        def fail_after_audit(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected after complete application audit")

        with monkeypatch.context() as patch:
            patch.setattr(bridge, "finish_tangent_step", fail_after_audit)
            with pytest.raises(RuntimeError, match="after complete application audit"):
                case.scene.step(case.parameters["dt_s"], actions=case.actions)
        assert _seven(case.scene) == before
        assert case.scene.tangent_history is history
        case.scene.step(case.parameters["dt_s"], actions=case.actions)
        reference.scene.step(reference.parameters["dt_s"], actions=reference.actions)
        assert _seven(case.scene) == _seven(reference.scene)


def test_d04_scene_mode_is_configured_only_once_before_the_first_step():
    with _clamp().build() as scene:
        registry = _clamp_registry()
        _enable(scene, registry)
        history = scene.tangent_history
        for mode in ("application", "material"):
            with pytest.raises(ValueError, match="once|before|configur"):
                _enable(scene, registry, mode)
            assert scene.tangent_history is history
        scene.step(1 / 1024)
        state = scene.state
        history = scene.tangent_history
        with pytest.raises(ValueError, match="once|before|configur"):
            _enable(scene, registry, "material")
        assert scene.tangent_history is history
        np.testing.assert_array_equal(scene.state.rigid.position, state.rigid.position)


