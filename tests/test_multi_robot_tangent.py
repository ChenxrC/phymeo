"""UNEXECUTED tests-first: real multi-owner tangent forces share one publication."""

import sys
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from phymeo.runtime import _coupled_tangent_step as tangent_step
from phymeo.runtime.material_stations import MaterialStationRegistry
from phymeo.solvers.tangent_history import StationKey


def _example():
    name = "phymeo_general_scene_friction_example"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "examples" / "general_scene_friction.py"
        spec = spec_from_file_location(name, path)
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_a01_public_multi_robot_tangent_accepts_real_bilateral_normals(tmp_path):
    # The old scene must first prove the physical 2 N / 2 N precondition.
    # RED is the existing public enable method's unsupported behavior, never a
    # missing new module or a test-generated substitute for normal contact.
    case_module = _example()
    with case_module.make_two_paddles(tmp_path) as case:
        scene = case.scene
        initial = scene.observe()
        events = scene.contact_events
        loads = scene.sample()
        assert len(loads.contacts) == 2
        (box_key,) = scene.body_keys[case.world, case_module.BOX]
        expected_pairs = {
            tuple(sorted((scene.shape_key(body, "tool", "ball"), box_key)))
            for body in (case_module.LOWER_TOOL, case_module.UPPER_TOOL)
        }
        assert {(load.first, load.second) for load in loads.contacts} == expected_pairs
        for load in loads.contacts:
            assert load.gap_m == pytest.approx(-1 / 1024, abs=2e-13)
            assert load.force_n == pytest.approx(2.0, abs=1e-10)
            assert len(load.sites) == 1
            site = load.sites[0]
            assert site.force_n == pytest.approx(2.0, abs=1e-10)
            assert site.fraction == pytest.approx(1.0, abs=1e-14)
            np.testing.assert_allclose(
                site.application_point_first_m,
                site.application_point_second_m,
                atol=1e-14,
                rtol=0,
            )
        box_row = scene.rigid_rows[case.world, case_module.BOX]
        np.testing.assert_allclose(loads.inputs.rigid.force[box_row], 0, atol=1e-12, rtol=0)
        np.testing.assert_allclose(loads.inputs.rigid.torque[box_row], 0, atol=1e-12, rtol=0)
        assert scene.contact_events is events
        assert scene.observe().step_number == initial.step_number == 0
        assert scene.tangent_history is None

        scene.enable_coupled_tangent(case.registry, stiffness_n_per_m=case_module.KT)

        assert scene.tangent_history.step_index == 0
        assert scene.tangent_history.active == ()
        assert scene.tangent_receipt is None
        assert scene.observe().time_s == 0.0
        assert scene.contact_events is events


def _skew(value):
    x, y, z = value
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], float)


def _spy_mobility(monkeypatch):
    calls = []
    original = tangent_step._validate_world_mobility

    def capture(value):
        calls.append(value.copy())
        return original(value)

    monkeypatch.setattr(tangent_step, "_validate_world_mobility", capture)
    return calls


def _enable(case, registry=None):
    case.scene.enable_coupled_tangent(
        case.registry if registry is None else registry, stiffness_n_per_m=_example().KT
    )


def _snapshot(scene):
    return _example().public_payload(
        dict(
            state=scene.state,
            reading=scene.observe(),
            geometry=scene.geometry(),
            events=scene.contact_events,
            history=scene.tangent_history,
            receipt=scene.tangent_receipt,
            last_frame=scene.last_frame,
        )
    )


def _energy(frame, previous_energy=0.0):
    history, receipt, events = frame["history"], frame["receipt"], frame["events"]
    assert receipt.accepted
    assert receipt.material_motion_error_m <= 1e-6
    assert receipt.application_motion_error_m <= 1e-6
    assert receipt.material_application_error_m <= 1e-6
    assert abs(receipt.energy_residual_j) <= 1e-8
    np.testing.assert_allclose(receipt.per_station_energy_residual_j, 0, atol=1e-8, rtol=0)
    assert history.commit_time_s == events.commit_time_s == frame["reading"].time_s
    assert history.sample_time_s == events.sample_time_s == frame["reading"].load_time_s
    actual_work = 0.0
    for i, record in enumerate(history.active):
        sample, update = record.sample, record.update
        found = []
        for event_record in events.active:
            if (event_record.identity.first, event_record.identity.second) != (
                sample.first.shape,
                sample.second.shape,
            ):
                continue
            for component in event_record.components:
                for site, material in zip(component.load.sites, component.material, strict=True):
                    if np.allclose(
                        material.first.local_point_m, sample.first.local_point_m, atol=1e-11, rtol=0
                    ) and np.allclose(
                        material.second.local_point_m,
                        sample.second.local_point_m,
                        atol=1e-11,
                        rtol=0,
                    ):
                        found.append((site, component.load, component.law))
        assert len(found) == 1, (record.identity, len(found))
        site, load, law = found[0]
        traction = np.asarray(site.force_on_second_n) - site.force_n * np.asarray(load.normal)
        np.testing.assert_allclose(
            traction, -200 * np.asarray(update.elastic_displacement_m), atol=1e-9, rtol=0
        )
        assert np.linalg.norm(traction) <= law.friction * site.force_n + 1e-9
        assert update.stored_energy_j == pytest.approx(
            100 * np.dot(update.elastic_displacement_m, update.elastic_displacement_m), abs=1e-12
        )
        actual_work += float(traction @ receipt.actual_application_displacement_m[i])
    assert actual_work == pytest.approx(receipt.actual_work_j, abs=1e-9)
    stored = sum(record.update.stored_energy_j for record in history.active)
    # Actual work plus spring change must be non-positive up to the published
    # numerical tolerance; the original receipt separately exposes release and
    # its full per-station constitutive balance. It is not total-scene energy.
    assert actual_work + stored - previous_energy <= 1e-8
    return stored


def test_a02_full_cross_owner_mobility_and_first_step_match_independent_mass_oracle(
    tmp_path, monkeypatch
):
    example = _example()
    calls = _spy_mobility(monkeypatch)
    with example.make_two_paddles(tmp_path) as case:
        _enable(case)
        external = case.scene.inputs()
        row = case.scene.rigid_rows[case.world, example.BOX]
        external.rigid.force[row, 0] = 0.5
        entry = case.scene.state
        result = case.scene.step(example.H, actions=case.actions, external=external)
        assert len(calls) == 1
        ell = 0.05 - example.DELTA / 2
        inverse_inertia = np.diag([240.0, 240.0, 150.0])
        levers = (np.array([0, 0, -ell]), np.array([0, 0, ell]))
        expected = np.empty((6, 6))
        for i, ri in enumerate(levers):
            for j, rj in enumerate(levers):
                block = np.eye(3) + _skew(ri) @ inverse_inertia @ _skew(rj).T
                if i == j:
                    block += np.diag([0.5, 0, 0])
                expected[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = block
        np.testing.assert_allclose(calls[0], expected, atol=1e-10, rtol=0)
        f = -200 * example.H**2 * 0.5 / (1 + 2.5 * 200 * example.H**2)
        acceleration = 0.5 + 2 * f
        np.testing.assert_allclose(
            result.rigid.velocity[row], [example.H * acceleration, 0, 0], atol=2e-11, rtol=0
        )
        np.testing.assert_allclose(
            result.rigid.position[row], [example.H**2 * acceleration, 0, 0], atol=2e-11, rtol=0
        )
        np.testing.assert_allclose(result.rigid.angular_velocity[row], 0, atol=2e-11, rtol=0)
        for body, sign in ((example.LOWER_TOOL, -1), (example.UPPER_TOOL, 1)):
            robot = result.robots[case.world, body]
            np.testing.assert_allclose(robot.contact_effort_full, [-f], atol=1e-9, rtol=0)
            np.testing.assert_allclose(robot.v, [-example.H * f / 2], atol=2e-11, rtol=0)
            np.testing.assert_allclose(robot.q, [-(example.H**2) * f / 2], atol=2e-11, rtol=0)
            np.testing.assert_allclose(
                robot.link_wrenches_world[0, :3], [-f, 0, sign * 2], atol=1e-9, rtol=0
            )
            expected_torque = np.cross(
                [0, 0, -sign * (example.RADIUS - example.DELTA / 2)], [-f, 0, sign * 2]
            )
            np.testing.assert_allclose(
                robot.link_wrenches_world[0, 3:], expected_torque, atol=1e-9, rtol=0
            )
            np.testing.assert_array_equal(robot.fixed_root_contact_wrench, np.zeros(6))
        momentum = result.rigid.velocity[row, 0] + sum(2 * r.v[0] for r in result.robots.values())
        assert momentum == pytest.approx(example.H * 0.5, abs=1e-12)
        expected_displacement = np.array([[example.H**2 * (0.5 + 2.5 * f), 0, 0]] * 2)
        np.testing.assert_allclose(
            case.scene.tangent_receipt.actual_material_displacement_m,
            expected_displacement,
            atol=2e-11,
            rtol=0,
        )
        np.testing.assert_allclose(
            case.scene.tangent_receipt.actual_application_displacement_m,
            expected_displacement,
            atol=2e-11,
            rtol=0,
        )
        _energy(example.capture_frame(case, entry, result))


@pytest.fixture(scope="module")
def f01_windows(tmp_path_factory):
    example = _example()
    directory = tmp_path_factory.mktemp("multi-tangent-f01")
    results = {}
    for label, friction, force, reverse, transformed in (
        ("loaded", 0.5, 0.5, False, False),
        ("overload", 0.5, 3.0, False, False),
        ("frictionless", 0.0, 0.5, False, False),
        ("reverse", 0.5, 0.5, True, False),
        ("covariant", 0.5, 0.5, False, True),
    ):
        frames, failure = [], None
        with example.make_two_paddles(
            directory / label, friction=friction, reverse=reverse, transformed=transformed
        ) as case:
            _enable(case)
            try:
                for index in range(128):
                    external = case.scene.inputs()
                    applied = force if label == "overload" or index < 64 else -force
                    row = case.scene.rigid_rows[case.world, example.BOX]
                    external.rigid.force[row] = np.asarray(case.parameters["rotation"]) @ [
                        applied,
                        0,
                        0,
                    ]
                    entry = case.scene.state
                    reading = case.scene.step(example.H, actions=case.actions, external=external)
                    frames.append(example.capture_frame(case, entry, reading))
            except Exception as error:
                failure = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "accepted_steps": len(frames),
                }
                raise
            finally:
                artifact = example.write_trace(
                    directory / f"{label}.json.gz", case=case, frames=frames, failure=failure
                )
                print(
                    "F01 complete raw", label, artifact, "accepted", len(frames), "failure", failure
                )
        results[label] = frames
    return results


def test_a03_loaded_overload_and_zero_friction_windows_keep_real_balance(f01_windows):
    example = _example()
    for label in ("loaded", "overload", "frictionless"):
        frames, energy = f01_windows[label], 0.0
        assert len(frames) == 128
        for i, frame in enumerate(frames, start=1):
            reading, entry = frame["reading"], frame["entry"]
            assert reading.step_number == i
            assert reading.load_time_s == (i - 1) * example.H
            assert len(frame["history"].active) == 2
            assert all(r.identity.generation == 1 for r in frame["history"].active)
            assert sum(load.force_n for load in reading.loads.contacts) == pytest.approx(
                4, abs=1e-9
            )
            expected = example.H * (
                reading.external.rigid.force[0] + reading.loads.inputs.rigid.force[0]
            )
            np.testing.assert_allclose(
                reading.rigid.velocity[0] - entry.rigid.velocity[0], expected, atol=1e-10, rtol=0
            )
            for address, robot in reading.robots.items():
                np.testing.assert_allclose(
                    2 * (robot.v - entry.robots[address].v),
                    example.H * robot.contact_effort,
                    atol=1e-10,
                    rtol=0,
                )
            total = reading.rigid.velocity[0, 0] - entry.rigid.velocity[0, 0]
            total += sum(2 * (r.v[0] - entry.robots[a].v[0]) for a, r in reading.robots.items())
            assert total == pytest.approx(example.H * reading.external.rigid.force[0, 0], abs=1e-10)
            np.testing.assert_allclose(reading.rigid.angular_velocity[0], 0, atol=1e-9, rtol=0)
            energy = _energy(frame, energy)
        assert max(abs(f["reading"].rigid.position[0, 0]) for f in frames) > 1e-6
    assert any(
        r.update.regime == "slide"
        for frame in f01_windows["overload"]
        for r in frame["history"].active
    )
    assert all(
        r.update.stored_energy_j == 0 and r.update.plastic_slip_m == 0
        for frame in f01_windows["frictionless"]
        for r in frame["history"].active
    )
    first_half = f01_windows["frictionless"][63]["reading"].rigid
    assert first_half.position[0, 0] == pytest.approx(0.5 * example.H**2 * 64 * 65 / 2, abs=2e-11)
    assert first_half.velocity[0, 0] == pytest.approx(0.5 * example.H * 64, abs=2e-11)


def test_a04_registration_order_and_world_covariance_preserve_material_motion(f01_windows):
    example = _example()
    for base, reverse, transformed in zip(
        f01_windows["loaded"], f01_windows["reverse"], f01_windows["covariant"], strict=True
    ):
        for address, robot in base["reading"].robots.items():
            for field in ("q", "v", "full_q", "full_v", "contact_effort_full"):
                np.testing.assert_allclose(
                    getattr(robot, field),
                    getattr(reverse["reading"].robots[address], field),
                    atol=1e-10,
                    rtol=0,
                )
                np.testing.assert_allclose(
                    getattr(robot, field),
                    getattr(transformed["reading"].robots[address], field),
                    atol=1e-9,
                    rtol=0,
                )
        for field in ("velocity", "angular_velocity"):
            np.testing.assert_allclose(
                getattr(base["reading"].rigid, field) @ example.CYCLE.T,
                getattr(transformed["reading"].rigid, field),
                atol=1e-9,
                rtol=0,
            )
        np.testing.assert_allclose(
            base["reading"].rigid.position @ example.CYCLE.T + example.SHIFT,
            transformed["reading"].rigid.position,
            atol=1e-9,
            rtol=0,
        )
        for a, b in zip(base["history"].active, transformed["history"].active, strict=True):
            assert a.identity == b.identity
            np.testing.assert_allclose(
                np.asarray(a.update.elastic_displacement_m) @ example.CYCLE.T,
                b.update.elastic_displacement_m,
                atol=1e-9,
                rtol=0,
            )


def test_a05_selected_and_legacy_pairs_preserve_each_original_law(tmp_path):
    example = _example()
    with example.make_two_paddles(
        tmp_path, selected_tools=(example.LOWER_TOOL,), box_velocity_x=1 / 64
    ) as case:
        scene = case.scene
        before = scene.sample()
        upper = scene.shape_key(example.UPPER_TOOL, "tool", "ball")
        old = next(load for load in before.contacts if upper in (load.first, load.second))
        assert old.sites[0].force_on_second_n[0] == pytest.approx(-1, abs=1e-10)
        _enable(case)
        reading = scene.step(example.H, actions=case.actions)
        kept = next(load for load in reading.loads.contacts if upper in (load.first, load.second))
        assert kept == old
        assert len(scene.tangent_history.active) == 1
        assert len(scene.contact_events.active) == 2
        assert scene.tangent_history.active[0].identity.key.token == 101
        assert abs(reading.loads.inputs.rigid.force[0, 0] + 1) > 1e-6


def test_a06_empty_active_set_advances_real_free_state_without_creating_station(tmp_path):
    example = _example()
    with example.make_two_paddles(tmp_path, separated=True) as case:
        assert not case.scene.sample().contacts
        _enable(case)
        external = case.scene.inputs()
        external.rigid.force[0, 0] = 0.5
        reading = case.scene.step(example.H, actions=case.actions, external=external)
        assert reading.rigid.velocity[0, 0] == pytest.approx(0.5 * example.H, abs=2e-11)
        assert case.scene.tangent_history.active == ()
        assert case.scene.tangent_history.step_index == 1
        assert case.scene.tangent_receipt.accepted
        assert case.scene.tangent_receipt.actual_material_displacement_m.shape == (0, 3)


@pytest.mark.parametrize("stage", ["prepare", "finish", "owned", "events"])
def test_a07_rejected_attempt_keeps_every_public_channel_and_retry_is_identical(
    tmp_path, monkeypatch, stage
):
    from phymeo.runtime import multi_robot_scene as public_scene

    example = _example()
    with (
        example.make_two_paddles(tmp_path / stage) as case,
        example.make_two_paddles(tmp_path / (stage + "-reference")) as reference,
    ):
        for current in (case, reference):
            _enable(current)
            external = current.scene.inputs()
            external.rigid.force[0, 0] = 0.5
            current.scene.step(example.H, actions=current.actions, external=external)
        before, history = _snapshot(case.scene), case.scene.tangent_history
        external = case.scene.inputs()
        external.rigid.force[0, 0] = 0.5

        def fail(*args, **kwargs):
            raise ValueError("injected tangent publication failure")

        with monkeypatch.context() as patch:
            if stage == "prepare":
                patch.setattr(tangent_step, "_validate_world_mobility", fail)
            elif stage == "finish":
                patch.setattr(tangent_step, "_maximum_norm", fail)
            elif stage == "owned":
                patch.setattr(public_scene, "_owned", fail)
            else:
                original_owned, original_stage = (
                    public_scene._owned,
                    case.scene._event_adapter.stage,
                )
                owned = []

                def mark(value):
                    result = original_owned(value)
                    owned.append(True)
                    return result

                def reject_after_owned(*args, **kwargs):
                    if owned:
                        fail()
                    return original_stage(*args, **kwargs)

                patch.setattr(public_scene, "_owned", mark)
                patch.setattr(case.scene._event_adapter, "stage", reject_after_owned)
            with pytest.raises(ValueError, match="injected tangent publication failure"):
                case.scene.step(example.H, actions=case.actions, external=external)
        assert case.scene.tangent_history is history
        assert _snapshot(case.scene) == before
        case.scene.step(example.H, actions=case.actions, external=external)
        other = reference.scene.inputs()
        other.rigid.force[0, 0] = 0.5
        reference.scene.step(example.H, actions=reference.actions, external=other)
        assert _snapshot(case.scene) == _snapshot(reference.scene)


def test_a08_world_identity_observation_ownership_and_fixed_capacity(tmp_path):
    example = _example()
    with (
        example.make_two_paddles(tmp_path / "one", world=23) as a,
        example.make_two_paddles(tmp_path / "two", world=71) as b,
    ):
        with pytest.raises(ValueError, match="world|owner|shape|registered|registry|pair"):
            a.scene.enable_coupled_tangent(b.registry, stiffness_n_per_m=200)
        assert a.scene.tangent_history is None
        _enable(a)
        _enable(b)
        external = a.scene.inputs()
        external.rigid.force[0, 0] = 0.5
        reading = a.scene.step(example.H, actions=a.actions, external=external)
        saved, events, history = _snapshot(a.scene), a.scene.contact_events, a.scene.tangent_history
        reading.loads.inputs.rigid.force[:] = 99
        state = a.scene.state
        state.rigid.position[:] = 88
        external.rigid.force[:] = 77
        a.scene.observe_scene()
        a.scene.sample()
        assert _snapshot(a.scene) == saved
        assert a.scene.contact_events is events and a.scene.tangent_history is history
        assert b.scene.tangent_history.step_index == 0
        assert {record.identity.key.world for record in history.active} == {23}
        original = a.registry.registrations[0]
        with pytest.raises(ValueError, match="capacity|128|maximum"):
            MaterialStationRegistry(
                tuple(replace(original, key=StationKey(23, 10000 + i)) for i in range(129))
            )


@pytest.mark.parametrize("drive,friction", [(0.5, 0.5), (4.0, 0.5), (0.5, 0.0)])
def test_b01_real_finite_rod_two_sites_have_normal_weight_once_and_complete_window(
    tmp_path, drive, friction
):
    example = _example()
    frames, failure = [], None
    with example.make_rod(tmp_path, drive=drive, friction=friction) as case:
        loads = case.scene.sample()
        assert sum(load.force_n for load in loads.contacts) == pytest.approx(2, abs=1e-9)
        assert sum(len(load.sites) for load in loads.contacts) == 2
        assert [site.fraction for load in loads.contacts for site in load.sites] == pytest.approx(
            [0.5, 0.5]
        )
        _enable(case)
        energy = 0.0
        try:
            for index in range(64):
                entry = case.scene.state
                reading = case.scene.step(example.H, actions=case.actions)
                frame = example.capture_frame(case, entry, reading)
                frames.append(frame)
                robot = reading.robots[case.world, example.LOWER_TOOL]
                np.testing.assert_allclose(
                    2 * (robot.v - entry.robots[case.world, example.LOWER_TOOL].v),
                    example.H * (robot.drive_effort + robot.contact_effort),
                    atol=1e-10,
                    rtol=0,
                )
                if index == 0:
                    force = (
                        0.0
                        if friction == 0
                        else -400 * example.H**2 * drive / (2 + 400 * example.H**2)
                    )
                    assert robot.contact_effort_full[0] == pytest.approx(force, abs=1e-9)
                    assert robot.q[0] == pytest.approx(
                        example.H**2 * (drive + force) / 2, abs=2e-11
                    )
                energy = _energy(frame, energy)
            if drive == 4:
                assert any(
                    record.update.regime == "slide"
                    for frame in frames
                    for record in frame["history"].active
                )
            if friction == 0:
                assert reading.robots[case.world, example.LOWER_TOOL].q[0] == pytest.approx(
                    drive / 2 * example.H**2 * 64 * 65 / 2, abs=2e-11
                )
        except Exception as error:
            failure = {
                "type": type(error).__name__,
                "message": str(error),
                "accepted_steps": len(frames),
            }
            raise
        finally:
            print(
                "F02 raw",
                example.write_trace(
                    tmp_path / "window.json.gz", case=case, frames=frames, failure=failure
                ),
            )


def test_b02_finite_face_clipping_rejects_unmatched_positive_station_but_separation_is_empty(
    tmp_path,
):
    example = _example()
    with example.make_rod(tmp_path / "clipped", center_y=0.2) as case:
        assert sum(load.force_n for load in case.scene.sample().contacts) > 0
        _enable(case)
        before = _snapshot(case.scene)
        with pytest.raises(ValueError, match="station|anchor|match|registration"):
            case.scene.step(example.H, actions=case.actions)
        assert _snapshot(case.scene) == before
    with example.make_rod(tmp_path / "outside", center_y=0.4) as case:
        assert not case.scene.sample().contacts
        _enable(case)
        case.scene.step(example.H, actions=case.actions)
        assert case.scene.tangent_history.active == ()


def test_b03_missing_station_and_wrong_anchor_are_not_silent_separation(tmp_path):
    example = _example()
    for mode in ("missing", "wrong-anchor"):
        with example.make_rod(tmp_path / mode) as case:
            registrations = case.registry.registrations
            bad = MaterialStationRegistry(
                registrations[:1]
                if mode == "missing"
                else (
                    replace(registrations[0], anchor_local_m=(1 / 64, -1 / 8, -1 / 32)),
                    registrations[1],
                )
            )
            _enable(case, bad)
            before = _snapshot(case.scene)
            with pytest.raises(ValueError, match="station|anchor|match|registration"):
                case.scene.step(example.H, actions=case.actions)
            assert _snapshot(case.scene) == before


def test_c01_affine_fixed_projection_keeps_cross_station_mass_and_full_reaction(
    tmp_path, monkeypatch
):
    example = _example()
    calls = _spy_mobility(monkeypatch)
    with example.make_fixed_mimic_pair(tmp_path) as case:
        assert [load.force_n for load in case.scene.sample().contacts] == pytest.approx(
            [2, 2], abs=1e-9
        )
        _enable(case)
        reading = case.scene.step(example.H, actions=case.actions)
        W = np.array([[4 / 9, 5 / 9], [5 / 9, 7 / 9]])
        full = np.zeros((6, 6))
        full[np.ix_([0, 3], [0, 3])] = W
        np.testing.assert_allclose(calls[0], full, atol=1e-10, rtol=0)
        force = np.linalg.solve(np.eye(2) / 200 + example.H**2 * W, -np.full(2, example.H**2 / 3))
        first = reading.robots[case.world, example.LOWER_TOOL]
        second = reading.robots[case.world, example.UPPER_TOOL]
        a = -(force[0] + 2 * force[1]) / 9
        np.testing.assert_allclose(first.contact_effort_full, -force, atol=1e-9, rtol=0)
        np.testing.assert_allclose(first.contact_effort, [9 * a], atol=1e-9, rtol=0)
        np.testing.assert_allclose(
            first.full_v, example.H * np.array([a, 2 * a]), atol=2e-11, rtol=0
        )
        assert second.v[0] == pytest.approx(example.H * (1 + force.sum()) / 3, abs=2e-11)
        receipt = first.mimic_loads
        np.testing.assert_allclose(
            receipt.constraint_effort_full, np.array([a, 4 * a]) + force, atol=1e-9, rtol=0
        )
        assert np.array([1, 2]) @ receipt.constraint_effort_full == pytest.approx(0, abs=1e-10)
        assert abs(receipt.constraint_power_W) < 1e-10


def test_c02_affine_fixed_window_and_effort_cap_keep_true_full_coordinates(tmp_path):
    example = _example()
    results = []
    for label, reverse, cap in (("base", False, 8), ("reverse", True, 8), ("capped", False, 0.25)):
        frames, failure = [], None
        with example.make_fixed_mimic_pair(
            tmp_path / label, reverse=reverse, partner_cap=cap
        ) as case:
            _enable(case)
            energy = 0.0
            try:
                for step in range(64):
                    entry = case.scene.state
                    reading = case.scene.step(example.H, actions=case.actions)
                    frame = example.capture_frame(case, entry, reading)
                    frames.append(frame)
                    a, b = (
                        reading.robots[case.world, example.LOWER_TOOL],
                        reading.robots[case.world, example.UPPER_TOOL],
                    )
                    assert a.full_q[1] == pytest.approx(2 * a.full_q[0], abs=2e-11)
                    assert a.full_v[1] == pytest.approx(2 * a.full_v[0], abs=2e-11)
                    assert b.drive_effort[0] == pytest.approx(min(1.0, cap), abs=1e-12)
                    assert np.array([1, 2]) @ a.mimic_loads.constraint_effort_full == pytest.approx(
                        0, abs=1e-9
                    )
                    assert abs(a.mimic_loads.constraint_power_W) < 1e-9
                    energy = _energy(frame, energy)
            except Exception as error:
                failure = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "accepted_steps": len(frames),
                }
                raise
            finally:
                print(
                    "F03 raw",
                    example.write_trace(
                        tmp_path / f"{label}.json.gz", case=case, frames=frames, failure=failure
                    ),
                )
            results.append(_snapshot(case.scene))
    assert results[0] == results[1]
    assert results[0] != results[2]


def test_d01_real_normal_withdrawal_reentry_releases_spring_once_and_uses_new_episode(tmp_path):
    example = _example()
    frames, failure = [], None
    with example.make_withdrawal(tmp_path) as case:
        _enable(case)
        energy = 0.0
        try:
            for index, (qz, vz) in enumerate(((-1 / 16, -4), (0, 4), (0, 0))):
                entry = case.scene.state
                reading = case.scene.step(1 / 64, actions=example.withdrawal_actions(index))
                frame = example.capture_frame(case, entry, reading)
                frames.append(frame)
                robot = reading.robots[case.world, example.LOWER_TOOL]
                assert robot.q[1] == pytest.approx(qz, abs=2e-11)
                assert robot.v[1] == pytest.approx(vz, abs=2e-11)
                energy = _energy(frame, energy)
            first, opened, reentry = (frame["history"] for frame in frames)
            assert first.active[0].identity.generation == 1
            assert first.active[0].update.stored_energy_j > 0
            assert opened.active == ()
            assert opened.dropped_spring_energy_j == pytest.approx(
                first.active[0].update.stored_energy_j, abs=1e-12
            )
            assert frames[1]["receipt"].released_energy_j == pytest.approx(
                opened.dropped_spring_energy_j, abs=1e-12
            )
            assert reentry.active[0].identity.generation == 2
            assert reentry.dropped_spring_energy_j == 0
            assert frames[2]["receipt"].released_energy_j == 0
        except Exception as error:
            failure = {
                "type": type(error).__name__,
                "message": str(error),
                "accepted_steps": len(frames),
            }
            raise
        finally:
            print(
                "F06 raw",
                example.write_trace(
                    tmp_path / "window.json.gz", case=case, frames=frames, failure=failure
                ),
            )


def test_e01_ambiguous_material_anchor_and_unknown_shape_are_explicit_configuration_errors(
    tmp_path,
):
    example = _example()
    with example.make_two_paddles(tmp_path / "ambiguous") as case:
        registrations = case.registry.registrations
        duplicate = replace(registrations[0], key=StationKey(case.world, 303))
        _enable(case, MaterialStationRegistry((*registrations, duplicate)))
        before, history = _snapshot(case.scene), case.scene.tangent_history
        with pytest.raises(ValueError, match="ambiguous|station|match|registration"):
            case.scene.step(example.H, actions=case.actions)
        assert _snapshot(case.scene) == before
        assert case.scene.tangent_history is history
    with example.make_two_paddles(tmp_path / "unknown") as case:
        declared = case.registry.registrations[0]
        unknown = replace(declared.second, shape=declared.second.shape + 1000)
        registry = MaterialStationRegistry((replace(declared, second=unknown),))
        before = _snapshot(case.scene)
        with pytest.raises(ValueError, match="shape|registered|registry|pair"):
            _enable(case, registry)
        assert _snapshot(case.scene) == before


def test_e02_full_contact_dimension_failure_is_atomic_and_valid_retry_matches_reference(
    tmp_path, monkeypatch
):
    example = _example()
    with (
        example.make_fixed_mimic_pair(tmp_path / "actual") as case,
        example.make_fixed_mimic_pair(tmp_path / "reference") as reference,
    ):
        for current in (case, reference):
            _enable(current)
            current.scene.step(example.H, actions=current.actions)
        before, history = _snapshot(case.scene), case.scene.tangent_history
        original = case.scene._sample

        def wrong_dimension(terms):
            measured, loads, details = original(terms)
            address = case.world, example.LOWER_TOOL
            assert measured[address].joint_effort_full.shape == (2,)
            malformed = dict(measured)
            malformed[address] = replace(measured[address], joint_effort_full=np.zeros(1))
            return malformed, loads, details

        with monkeypatch.context() as patch:
            patch.setattr(case.scene, "_sample", wrong_dimension)
            with pytest.raises(ValueError, match="dimension|shape|full|mismatch|matmul|contact"):
                case.scene.step(example.H, actions=case.actions)
        assert case.scene.tangent_history is history
        assert _snapshot(case.scene) == before
        case.scene.step(example.H, actions=case.actions)
        reference.scene.step(example.H, actions=reference.actions)
        assert _snapshot(case.scene) == _snapshot(reference.scene)


def test_e03_none_and_explicit_zero_feedforward_keep_original_gravity_semantics(tmp_path):
    example = _example()
    results = []
    for label, compensate in (("compensated", True), ("explicit-zero", False)):
        with example.make_fixed_mimic_pair(tmp_path / label, gravity=(1, 0, 0)) as case:
            assert [load.force_n for load in case.scene.sample().contacts] == pytest.approx(
                [2, 2], abs=1e-9
            )
            _enable(case)
            actions = {
                example.LOWER_TOOL: example.RobotCommand([0], feedforward=None),
                example.UPPER_TOOL: example.RobotCommand(
                    [0], feedforward=None if compensate else [0]
                ),
            }
            reading = case.scene.step(example.H, actions=actions)
            first, second = (
                reading.robots[case.world, example.LOWER_TOOL],
                reading.robots[case.world, example.UPPER_TOOL],
            )
            assert first.drive_effort[0] == pytest.approx(-5, abs=1e-12)
            assert second.drive_effort[0] == pytest.approx(-3 if compensate else 0, abs=1e-12)
            assert first.requested_command.feedforward is None
            if compensate:
                assert second.requested_command.feedforward is None
                np.testing.assert_allclose(first.v, 0, atol=2e-11, rtol=0)
                np.testing.assert_allclose(second.v, 0, atol=2e-11, rtol=0)
            else:
                np.testing.assert_array_equal(second.requested_command.feedforward, [0])
                world = np.array([[4 / 9, 5 / 9], [5 / 9, 7 / 9]])
                force = np.linalg.solve(
                    np.eye(2) / 200 + example.H**2 * world, -np.full(2, example.H**2)
                )
                assert first.v[0] == pytest.approx(
                    -example.H * (force[0] + 2 * force[1]) / 9, abs=2e-11
                )
                assert second.v[0] == pytest.approx(example.H * (1 + force.sum() / 3), abs=2e-11)
            results.append(reading)
    assert results[1].robots[example.WORLD, example.UPPER_TOOL].v[0] > 0
