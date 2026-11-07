"""UNEXECUTED F04/F05: full physical floating owners and finite rotating material motion."""

import sys
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest


ASSETS = Path(__file__).parent / "fixtures" / "urdf"


def _helpers():
    name = "phymeo_scene_tangent_test_helpers"
    if name not in sys.modules:
        path = Path(__file__).with_name("test_multi_robot_tangent.py")
        spec = spec_from_file_location(name, path)
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _independent_floating_mass(mimic):
    """Kinetic energy of three declared bodies, without product dynamics/FK calls."""
    masses = (2.0, 1.0, 2.0)
    centers = (np.zeros(3), np.array([-.25, 0, 0]), np.array([.254, 0, 0]))
    inertias = ((.2, .3, .4), (.0625, .078125, .09375), (.09375, .109375, .125))
    matrix = np.zeros((8, 8))
    for index, (mass, center, inertia) in enumerate(zip(masses, centers, inertias, strict=True)):
        linear = np.zeros((3, 8))
        linear[:, :3] = np.eye(3)
        linear[:, 3:6] = -_helpers()._skew(center)
        if index:
            linear[0, 5+index] = 1.0
        angular = np.zeros((3, 8))
        angular[:, 3:6] = np.eye(3)
        matrix += mass * linear.T @ linear + angular.T @ np.diag(inertia) @ angular
    if mimic:
        projection = np.zeros((8, 7))
        projection[:6, :6] = np.eye(6)
        projection[6:, 6] = (1, -1)
    else:
        projection = np.eye(8)
    return matrix, projection


def _floating_oracle(case, h):
    example = _helpers()._example()
    mimic = case.parameters["mimic"]
    full_mass, projection = _independent_floating_mass(mimic)
    mass = projection.T @ full_mass @ projection
    normal, jacobians = [], []
    for anchor in case.parameters["right_material_anchors"]:
        x, y, z = anchor
        sign = -1 if z < 0 else 1
        point = np.array([.254+x, y, z-sign*example.DELTA/2])
        jac = np.zeros((3, 8))
        jac[:, :3] = np.eye(3)
        jac[:, 3:6] = -_helpers()._skew(point)
        jac[0, 7] = 1.0
        jacobians.append(jac)
        normal.append(np.array([0.0, 0.0, sign]))
    rows = [jac @ projection for jac in jacobians]
    n = len(rows)
    world = np.empty((3*n, 3*n))
    for i, first in enumerate(rows):
        for j, second in enumerate(rows):
            world[3*i:3*i+3, 3*j:3*j+3] = first @ np.linalg.solve(mass, second.T) + np.diag([1/3, 0, 0])
    tangent = np.zeros((3*n, 2*n))
    for i, direction in enumerate(normal):
        tangent[3*i:3*i+3, 2*i] = (1, 0, 0)
        tangent[3*i:3*i+3, 2*i+1] = np.cross(direction, [1, 0, 0])
    drive = np.zeros(mass.shape[0])
    drive[-1] = .1
    free_acceleration = np.linalg.solve(mass, drive)
    free_displacement = np.array([-h*h*jac @ free_acceleration for jac in rows])
    force2 = np.linalg.solve(np.eye(2*n)/200+h*h*(tangent.T @ world @ tangent), -tangent.T @ free_displacement.ravel())
    traction = (tangent @ force2).reshape(n, 3)
    total_force = traction + .5*np.asarray(normal)
    full_contact = -sum((jac.T @ force for jac, force in zip(jacobians, total_force, strict=True)), start=np.zeros(8))
    acceleration = np.linalg.solve(mass, drive + projection.T @ full_contact)
    actuator_full = np.zeros(8)
    actuator_full[6 if mimic else 7] = .1
    constraint = full_mass @ (projection @ acceleration) - actuator_full - full_contact
    return dict(world=world, full_mass=full_mass, projection=projection,
                acceleration=acceleration, full_contact=full_contact, traction=traction,
                plate_acceleration=total_force[:, 0].sum()/3, constraint=constraint)


@pytest.mark.parametrize("mimic", [False, True], ids=["ordinary-floating", "affine-floating"])
def test_a01_floating_full_mass_jacobian_and_bilateral_reactions_match_independent_energy(tmp_path, monkeypatch, mimic):
    helper, example = _helpers(), _helpers()._example()
    calls = helper._spy_mobility(monkeypatch)
    with example.make_floating_clamp(tmp_path, fixture_dir=ASSETS, mimic=mimic) as case:
        sampled = case.scene.sample()
        assert sum(load.force_n for load in sampled.contacts) == pytest.approx(4.0, abs=1e-9)
        assert sum(len(load.sites) for load in sampled.contacts) == 8
        assert [site.force_n for load in sampled.contacts for site in load.sites] == pytest.approx([.5]*8, abs=1e-9)
        helper._enable(case)
        expected = _floating_oracle(case, example.H)
        entry = case.scene.state
        reading = case.scene.step(example.H, actions=case.actions)
        assert len(calls) == 1
        np.testing.assert_allclose(calls[0], expected["world"], atol=1e-9, rtol=0)
        robot = reading.robots[case.world, example.LOWER_TOOL]
        partner = reading.robots[case.world, example.UPPER_TOOL]
        assert robot.contact_effort_full.shape == (8,)
        assert robot.contact_effort.shape == ((7,) if mimic else (8,))
        np.testing.assert_allclose(robot.contact_effort_full, expected["full_contact"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(robot.contact_effort, expected["projection"].T @ expected["full_contact"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(robot.v, example.H*expected["acceleration"], atol=2e-11, rtol=0)
        np.testing.assert_allclose(robot.full_v, example.H*expected["projection"] @ expected["acceleration"], atol=2e-11, rtol=0)
        assert partner.v[0] == pytest.approx(example.H*expected["plate_acceleration"], abs=2e-11)
        np.testing.assert_allclose(robot.link_wrenches_world[:2], 0, atol=1e-9, rtol=0)
        np.testing.assert_allclose(robot.link_wrenches_world.sum(axis=0)[:3] + partner.link_wrenches_world.sum(axis=0)[:3], 0, atol=1e-9, rtol=0)
        if mimic:
            receipt = robot.mimic_loads
            np.testing.assert_allclose(receipt.constraint_effort_full, expected["constraint"], atol=1e-9, rtol=0)
            np.testing.assert_allclose(receipt.constraint_effort_full[:6], 0, atol=1e-9, rtol=0)
            np.testing.assert_allclose(expected["projection"].T @ receipt.constraint_effort_full, 0, atol=1e-9, rtol=0)
        displacement = partner.q[0] - robot.full_q[0] - (robot.full_q[8] - .004)
        np.testing.assert_allclose(case.scene.tangent_receipt.actual_material_displacement_m, [[displacement, 0, 0]]*8, atol=2e-11, rtol=0)
        np.testing.assert_allclose(case.scene.tangent_receipt.actual_application_displacement_m, [[displacement, 0, 0]]*8, atol=2e-11, rtol=0)
        helper._energy(example.capture_frame(case, entry, reading))


@pytest.fixture(scope="module")
def floating_windows(tmp_path_factory):
    helper, example = _helpers(), _helpers()._example()
    directory = tmp_path_factory.mktemp("floating-friction-full")
    results = {}
    for label, mimic, reverse in (("ordinary", False, False), ("mimic", True, False), ("mimic-reverse", True, True)):
        frames, failure = [], None
        with example.make_floating_clamp(directory/label, fixture_dir=ASSETS, mimic=mimic, reverse=reverse) as case:
            helper._enable(case)
            try:
                for _ in range(64):
                    entry = case.scene.state
                    reading = case.scene.step(example.H, actions=case.actions)
                    frames.append(example.capture_frame(case, entry, reading))
            except Exception as error:
                failure = {"type": type(error).__name__, "message": str(error), "accepted_steps": len(frames)}
                raise
            finally:
                print("F04 complete raw", example.write_trace(directory/f"{label}.json.gz", case=case, frames=frames, failure=failure))
        results[label] = frames
    return results


def test_a02_floating_windows_keep_material_owners_root_reaction_and_constraint_power(floating_windows):
    helper, example = _helpers(), _helpers()._example()
    for label, frames in floating_windows.items():
        energy = 0.0
        assert len(frames) == 64
        for index, frame in enumerate(frames, start=1):
            row = frame["reading"].robots[example.WORLD, example.LOWER_TOOL]
            plate = frame["reading"].robots[example.WORLD, example.UPPER_TOOL]
            assert frame["reading"].step_number == index
            assert len(frame["history"].active) == 8
            assert sum(load.force_n for load in frame["reading"].loads.contacts) == pytest.approx(4, abs=1e-8)
            np.testing.assert_allclose(row.full_v[3:6], 0, atol=1e-9, rtol=0)
            np.testing.assert_allclose(row.full_q[3:7], [0, 0, 0, 1], atol=1e-9, rtol=0)
            momentum = 5*row.full_v[0] + row.full_v[6] + 2*row.full_v[7] + 3*plate.v[0]
            assert momentum == pytest.approx(0, abs=1e-9)
            if label.startswith("mimic"):
                assert row.full_q[8] == pytest.approx(-row.full_q[7]+.004, abs=2e-11)
                assert row.full_v[7] == pytest.approx(-row.full_v[6], abs=2e-11)
                np.testing.assert_allclose(row.mimic_loads.constraint_effort_full[:6], 0, atol=1e-9, rtol=0)
                assert abs(row.mimic_loads.constraint_power_W) <= 1e-9
                assert all(abs(item.constraint_power_W) <= 1e-9 for item in row.mimic_loads.row_forces)
            for record in frame["events"].active:
                for owner in (record.first_owner, record.second_owner):
                    if owner.body == example.LOWER_TOOL:
                        assert owner.kind == "robot_link" and owner.link_name == "right_jaw" and owner.link_index == 2
            energy = helper._energy(frame, energy)
        assert abs(frames[-1]["reading"].robots[example.WORLD, example.LOWER_TOOL].q[0]) > 1e-7
    for first, second in zip(floating_windows["mimic"], floating_windows["mimic-reverse"], strict=True):
        assert example.public_payload(first) == example.public_payload(second)


def _grouped(loads, *, reverse):
    groups = {}
    for load in loads.contacts:
        groups.setdefault((load.first, load.second), []).append(load)
    merged = []
    for values in groups.values():
        first = values[0]
        assert all(load.normal == first.normal and abs(load.gap_m-first.gap_m) < 1e-13 for load in values)
        sites = tuple(site for load in values for site in load.sites)
        if reverse:
            sites = sites[::-1]
        merged.append(replace(first, force_n=sum(load.force_n for load in values),
                              potential_j=sum(load.potential_j for load in values),
                              power_w=sum(load.power_w for load in values), sites=sites))
    return replace(loads, contacts=tuple(reversed(merged)) if reverse else tuple(merged))


def test_a03_real_patch_components_and_sites_can_reorder_without_losing_history_or_wrench(tmp_path, monkeypatch):
    helper, example = _helpers(), _helpers()._example()
    with example.make_floating_clamp(tmp_path/"base", fixture_dir=ASSETS) as base, example.make_floating_clamp(tmp_path/"grouped", fixture_dir=ASSETS) as altered:
        for case in (base, altered):
            helper._enable(case)
            case.scene.step(example.H, actions=case.actions)
        previous_energy = sum(record.update.stored_energy_j for record in altered.scene.tangent_history.active)
        original = altered.scene._sample

        def regroup(terms):
            measured, loads, details = original(terms)
            by_owner = {owner: replace(value, loads=_grouped(value.loads, reverse=True)) for owner, value in measured.items()}
            return by_owner, _grouped(loads, reverse=True), tuple(reversed(details))

        monkeypatch.setattr(altered.scene, "_sample", regroup)
        expected = base.scene.step(example.H, actions=base.actions)
        entry = altered.scene.state
        actual = altered.scene.step(example.H, actions=altered.actions)
        for owner, row in expected.robots.items():
            for field in ("q", "v", "full_q", "full_v", "contact_effort", "contact_effort_full", "link_wrenches_world"):
                np.testing.assert_allclose(getattr(row, field), getattr(actual.robots[owner], field), atol=1e-9, rtol=0)
        for a, b in zip(base.scene.tangent_history.active, altered.scene.tangent_history.active, strict=True):
            assert a.identity == b.identity
            assert a.sample.first == b.sample.first and a.sample.second == b.sample.second
            np.testing.assert_allclose(a.update.elastic_displacement_m, b.update.elastic_displacement_m, atol=1e-11, rtol=0)
            assert a.update.stored_energy_j == pytest.approx(b.update.stored_energy_j, abs=1e-12)
            assert a.update.tangent_work_j == pytest.approx(b.update.tangent_work_j, abs=1e-11)
        np.testing.assert_allclose(base.scene.tangent_receipt.actual_material_displacement_m,
                                   altered.scene.tangent_receipt.actual_material_displacement_m, atol=1e-11, rtol=0)
        assert len(actual.loads.contacts) == 2
        assert all(len(load.sites) == 4 for load in actual.loads.contacts)
        assert all(len(record.components) == 1 and len(record.components[0].material) == 4 for record in altered.scene.contact_events.active)
        helper._energy(example.capture_frame(altered, entry, actual), previous_energy)


def test_b01_rotating_point_jacobian_includes_actual_angular_offset(tmp_path, monkeypatch):
    helper, example = _helpers(), _helpers()._example()
    calls = helper._spy_mobility(monkeypatch)
    with example.make_rotating_tool(tmp_path) as case:
        sampled = case.scene.sample()
        assert sum(load.force_n for load in sampled.contacts) == pytest.approx(2, abs=1e-9)
        assert sum(len(load.sites) for load in sampled.contacts) == 12
        helper._enable(case)
        h = case.parameters["dt_s"]
        rows = np.array([-np.cross([0, 0, 1], anchor) for anchor in case.parameters["rotor_anchors"]])
        expected = rows.reshape(-1, 1) @ rows.reshape(1, -1) / .02
        entry = case.scene.state
        reading = case.scene.step(h, actions=case.actions)
        assert len(calls) == 1
        np.testing.assert_allclose(calls[0], expected, atol=1e-10, rtol=0)
        angular_stiffness = 200*float(np.sum(rows*rows))
        increment = h*h*.01/(.02+h*h*angular_stiffness)
        robot = reading.robots[case.world, example.LOWER_TOOL]
        assert robot.q[0] == pytest.approx(increment, abs=2e-11)
        assert robot.v[0] == pytest.approx(increment/h, abs=2e-11)
        assert robot.contact_effort_full[0] == pytest.approx(-angular_stiffness*increment, abs=1e-9)
        assert sum(load.power_w for load in reading.loads.contacts) == pytest.approx(0, abs=1e-12)
        assert case.scene.tangent_receipt.actual_work_j < 0
        angle = robot.q[0]
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        anchors = np.asarray(case.parameters["rotor_anchors"])
        displacement = anchors - anchors @ rotation.T
        np.testing.assert_allclose(case.scene.tangent_receipt.actual_material_displacement_m, displacement, atol=2e-11, rtol=0)
        np.testing.assert_allclose(case.scene.tangent_receipt.actual_application_displacement_m, displacement, atol=2e-11, rtol=0)
        helper._energy(example.capture_frame(case, entry, reading))


@pytest.fixture(scope="module")
def rotation_windows(tmp_path_factory):
    helper, example = _helpers(), _helpers()._example()
    directory = tmp_path_factory.mktemp("rotating-friction-full")
    results = {}
    for label, h, count, common in (("h", 1/4096, 128, False), ("half", 1/8192, 256, False), ("corotate", 1/4096, 8, True)):
        frames, failure = [], None
        with example.make_rotating_tool(directory/label, common=common) as case:
            helper._enable(case)
            try:
                for _ in range(count):
                    entry = case.scene.state
                    reading = case.scene.step(h, actions=case.actions)
                    frames.append(example.capture_frame(case, entry, reading))
            except Exception as error:
                failure = {"type": type(error).__name__, "message": str(error), "accepted_steps": len(frames)}
                raise
            finally:
                print("F05 complete raw", label, "actual dt", h,
                      example.write_trace(directory/f"{label}.json.gz", case=case, frames=frames, failure=failure, actual_dt_s=h))
        results[label] = frames
    return results


def test_b02_rotating_window_and_half_step_have_real_motion_work_and_frozen_engineering_error(rotation_windows):
    helper, example = _helpers(), _helpers()._example()
    impulse, angular_impulse = {}, {}
    for label in ("h", "half"):
        frames = rotation_windows[label]
        h, energy = 1/(4096 if label == "h" else 8192), 0.0
        impulse[label], angular_impulse[label] = np.zeros(3), np.zeros(3)
        for frame in frames:
            reading = frame["reading"]
            robot = reading.robots[example.WORLD, example.LOWER_TOOL]
            entry = frame["entry"].robots[example.WORLD, example.LOWER_TOOL]
            assert .02*(robot.v[0]-entry.v[0]) == pytest.approx(h*(.01+robot.contact_effort_full[0]), abs=1e-10)
            impulse[label] += h*robot.link_wrenches_world[0, :3]
            angular_impulse[label] += h*robot.link_wrenches_world[0, 3:]
            energy = helper._energy(frame, energy)
        assert frames[-1]["reading"].time_s == pytest.approx(1/32, abs=2e-12)
        assert abs(frames[-1]["reading"].robots[example.WORLD, example.LOWER_TOOL].q[0]) > 1e-7
    a = rotation_windows["h"][-1]["reading"].robots[example.WORLD, example.LOWER_TOOL]
    b = rotation_windows["half"][-1]["reading"].robots[example.WORLD, example.LOWER_TOOL]
    assert np.max(abs(a.q-b.q)) <= 1e-5
    assert np.max(abs(a.v-b.v)) <= 1e-4
    ga, gb = rotation_windows["h"][-1]["geometry"], rotation_windows["half"][-1]["geometry"]
    for key in ga.poses:
        assert np.max(abs(ga.poses[key].start-gb.poses[key].start)) <= 1e-5
    assert np.linalg.norm(impulse["h"]-impulse["half"]) <= 1e-5
    assert np.linalg.norm(angular_impulse["h"]-angular_impulse["half"]) <= 1e-5


def test_b03_common_actual_rotation_creates_no_relative_slip_or_spring_work(rotation_windows):
    example = _helpers()._example()
    for index, frame in enumerate(rotation_windows["corotate"], start=1):
        for robot in frame["reading"].robots.values():
            assert robot.q[0] == pytest.approx(.2*index/4096, abs=2e-11)
            assert robot.v[0] == pytest.approx(.2, abs=2e-11)
            np.testing.assert_allclose(robot.contact_effort_full, 0, atol=1e-10, rtol=0)
        assert all(record.update.stored_energy_j < 1e-20 and record.update.plastic_slip_m < 1e-12 for record in frame["history"].active)
        assert abs(frame["receipt"].actual_work_j) < 1e-14
        np.testing.assert_allclose(frame["receipt"].actual_material_displacement_m, 0, atol=1e-12, rtol=0)


def test_b04_legal_fast_tilt_is_rejected_by_real_motion_gate_without_publishing(tmp_path):
    helper, example = _helpers(), _helpers()._example()
    with example.make_rotating_tool(tmp_path, tilted=True) as case:
        assert sum(load.force_n for load in case.scene.sample().contacts) == pytest.approx(2, abs=1e-9)
        assert case.scene.observe().robots[case.world, example.LOWER_TOOL].v[0] == 5
        helper._enable(case)
        before, history = helper._snapshot(case.scene), case.scene.tangent_history
        with pytest.raises(ValueError, match="motion|material|application|station|anchor") as caught:
            case.scene.step(1/128, actions=case.actions)
        assert "velocity outside" not in str(caught.value)
        assert helper._snapshot(case.scene) == before
        assert case.scene.tangent_history is history
        print("F05 genuine motion rejection", example.write_trace(tmp_path/"rejected.json.gz", case=case, frames=[],
            failure={"type": type(caught.value).__name__, "message": str(caught.value), "before": before}))