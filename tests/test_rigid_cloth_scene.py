"""Fixed scene, counterfactual motion, independent audit and reproducible data."""

import importlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


def builder():
    return importlib.import_module("phymeo.builders.rigid_cloth").rigid_cloth_scene


def experiment():
    return importlib.import_module("phymeo.experiments.rigid_cloth")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fixed_scene_has_physical_mass_inertia_topology_and_initial_clearance(dtype):
    scene = builder()(dtype=dtype)
    assert scene.particle_model.particle_count == 49
    assert scene.rigid_model.body_count == 1
    assert scene.initial_state.particles.position.dtype == dtype
    np.testing.assert_array_equal(scene.pinned, [0, 6, 42, 48])
    np.testing.assert_array_equal(scene.particle_model.inverse_mass[scene.pinned], 0)
    assert np.sum(scene.cloth.vertex_mass) == pytest.approx(0.144, rel=1e-6)
    np.testing.assert_allclose(scene.rigid_model.inverse_mass, [20], rtol=1e-7)
    np.testing.assert_allclose(scene.rigid_model.inertia_diagonal, [[0.000648] * 3], rtol=1e-7)
    np.testing.assert_allclose(scene.initial_state.rigid.position, [[0.02, 0.30, 0]], atol=1e-8)
    delta = scene.initial_state.particles.position - scene.initial_state.rigid.position
    assert np.min(np.linalg.norm(delta, axis=1) - scene.spheres.radii[0]) > 0.1


def test_mass_experiment_changes_only_sphere_mass_and_inertia():
    first, second = builder()(sphere_mass=0.025), builder()(sphere_mass=0.1)
    np.testing.assert_array_equal(
        first.initial_state.particles.position, second.initial_state.particles.position
    )
    np.testing.assert_array_equal(first.constraints.rest_length, second.constraints.rest_length)
    np.testing.assert_array_equal(first.spheres.radii, second.spheres.radii)
    np.testing.assert_allclose(first.rigid_model.inverse_mass, 4 * second.rigid_model.inverse_mass)
    np.testing.assert_allclose(
        second.rigid_model.inertia_diagonal, 4 * first.rigid_model.inertia_diagonal
    )


@pytest.mark.parametrize("mass", [0, -1, True, np.nan, np.inf])
def test_scene_rejects_nonphysical_mass(mass):
    with pytest.raises(ValueError):
        builder()(sphere_mass=mass)


@pytest.fixture(scope="module")
def compared_runs():
    api = experiment()
    return api.run_scene(frames=12), api.run_scene(frames=12, reaction=False)


def test_two_way_scene_contacts_and_changes_both_trajectories(compared_runs):
    coupled, prescribed = compared_runs
    comparison = experiment().compare_runs(coupled, prescribed)
    assert comparison["max_sphere_position_difference_m"] >= 1e-3
    assert comparison["max_cloth_position_difference_m"] >= 1e-6
    for run in (coupled, prescribed):
        assert run.metrics["accepted_frames"] == 12
        assert run.metrics["audited_substeps"] == 48
        assert run.metrics["contact_substeps"] > 0
        assert run.metrics["max_vertex_penetration_m"] <= 2e-6
        assert run.metrics["max_pin_displacement_m"] == 0
        assert run.metrics["max_reported_residual"] <= 1e-7
        assert run.metrics["max_rigid_motion_balance_error_kg_mps"] <= 1e-10


def test_one_way_sphere_follows_discrete_free_fall_while_cloth_responds(compared_runs):
    _, run = compared_runs
    h = run.config["effective_substep_dt"]
    n = np.arange(13) * 4
    expected_y = 0.3 + n * h * 2 - 9.81 * h * h * n * (n + 1) / 2
    np.testing.assert_allclose(
        run.arrays["rigid_position"][:, 0, 1], expected_y, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        run.arrays["rigid_velocity"][:, 0, 1], 2 - n * h * 9.81, atol=1e-12, rtol=0
    )
    assert run.config["interaction"] == "one_way_prescribed_sphere"
    assert run.metrics["contact_substeps"] > 0


def test_repeated_runs_have_identical_numeric_arrays_and_digest():
    first, second = experiment().run_scene(frames=2), experiment().run_scene(frames=2)
    assert first.arrays.keys() == second.arrays.keys()
    for name in first.arrays:
        np.testing.assert_array_equal(first.arrays[name], second.arrays[name])
    assert first.metrics["trajectory_sha256"] == second.metrics["trajectory_sha256"]


def test_saved_run_is_numeric_loadable_and_never_overwrites_an_existing_record(tmp_path):
    run = experiment().run_scene(frames=1)
    destination = tmp_path / "record"
    experiment().save_run(run, destination)
    metadata = json.loads((destination / "run.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["config"]["frames"] == 1
    assert metadata["metrics"]["trajectory_sha256"] == run.metrics["trajectory_sha256"]
    assert metadata["provenance"]["source_sha256"]
    with np.load(destination / "trajectory.npz", allow_pickle=False) as data:
        for name, value in run.arrays.items():
            np.testing.assert_array_equal(data[name], value)
    with pytest.raises(FileExistsError):
        experiment().save_run(run, destination)


@pytest.mark.parametrize(
    "kwargs", [{"frames": 0}, {"frames": True}, {"dt": 0}, {"mode": "cuda"}, {"reaction": "no"}]
)
def test_runner_rejects_invalid_or_unimplemented_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        experiment().run_scene(**kwargs)


def test_rejected_frame_raises_instead_of_being_added_to_a_successful_trace():
    with pytest.raises(experiment().SceneRunFailure, match="frame"):
        experiment().run_scene(frames=2, iterations=1)


def test_independent_audit_rejects_pin_drift_even_when_report_claims_success():
    from phymeo import CoupledState, CoupledXPBDSolver

    scene = builder()()
    result = CoupledXPBDSolver(
        scene.particle_model, scene.rigid_model, scene.constraints, scene.spheres, substeps=1
    ).step(scene.initial_state, scene.inputs, 1 / 480)
    assert result.report.accepted
    bad = CoupledState(result.state.particles, result.state.rigid)
    bad.particles.position[0, 0] += 1e-3
    audit = importlib.import_module("phymeo.validation.rigid_cloth").audit_substep
    with pytest.raises(AssertionError, match="pinned"):
        audit(
            scene,
            scene.initial_state,
            bad,
            result.report.substeps[0],
            1 / 480,
            tolerance=1e-7,
            reaction=True,
        )


def test_independent_audit_rejects_nan_report_and_unbalanced_body_motion():
    from phymeo import CoupledState, CoupledXPBDSolver

    scene = builder()()
    result = CoupledXPBDSolver(
        scene.particle_model, scene.rigid_model, scene.constraints, scene.spheres, substeps=1
    ).step(scene.initial_state, scene.inputs, 1 / 480)
    audit = importlib.import_module("phymeo.validation.rigid_cloth").audit_substep
    with pytest.raises(AssertionError, match="residual"):
        audit(
            scene,
            scene.initial_state,
            result.state,
            replace(result.report.substeps[0], contact_residual=np.nan),
            1 / 480,
            tolerance=1e-7,
            reaction=True,
        )
    bad = CoupledState(result.state.particles, result.state.rigid)
    bad.rigid.velocity[0, 0] += 1e-2
    with pytest.raises(AssertionError, match="balance"):
        audit(
            scene,
            scene.initial_state,
            bad,
            result.report.substeps[0],
            1 / 480,
            tolerance=1e-7,
            reaction=True,
        )


def test_cli_smoke_saves_trace_and_reports_true_horizon(tmp_path):
    destination = tmp_path / "cli"
    command = [
        sys.executable,
        "examples/rigid_cloth.py",
        "--frames",
        "1",
        "--output",
        str(destination),
    ]
    result = subprocess.run(
        command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = json.loads((destination / "run.json").read_text(encoding="utf-8"))
    assert metadata["config"]["frames"] == 1
    assert metadata["config"]["duration_seconds"] == pytest.approx(1 / 120)


def test_free_fall_step_halving_has_first_order_position_error():
    evidence = experiment().free_fall_convergence(frames=8, dt=1 / 120, substeps=4)
    assert evidence["coarse_position_error_m"] > 0
    assert evidence["error_ratio"] == pytest.approx(2.0, rel=1e-9)
    assert evidence["fine_velocity_error_mps"] < 1e-12


def test_cli_rejection_has_failure_record_and_no_success_file(tmp_path):
    destination = tmp_path / "failed"
    command = [
        sys.executable,
        "examples/rigid_cloth.py",
        "--frames",
        "2",
        "--iterations",
        "1",
        "--output",
        str(destination),
    ]
    result = subprocess.run(
        command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert not (destination / "run.json").exists()
    assert not (destination / "trajectory.npz").exists()
    record = json.loads((destination / "failure.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert "frame" in record["error"]


@pytest.mark.parametrize("corruption", ["nonfinite_load", "spurious_rotation"])
def test_audit_rejects_invalid_contact_evidence_and_unforced_rotation(corruption):
    from types import SimpleNamespace

    from phymeo import CoupledState, CoupledXPBDSolver
    from phymeo.validation.rigid_cloth import audit_substep

    scene = builder()()
    solver = CoupledXPBDSolver(
        scene.particle_model, scene.rigid_model, scene.constraints, scene.spheres, substeps=1
    )
    result = solver.step(scene.initial_state, scene.inputs, 1 / 480)
    report = result.report.substeps[0]
    state = CoupledState(result.state.particles, result.state.rigid)
    if corruption == "nonfinite_load":
        load = SimpleNamespace(
            particle=report.position_impulses.particle.copy(),
            rigid_linear=report.position_impulses.rigid_linear,
            rigid_angular=report.position_impulses.rigid_angular,
        )
        load.particle[0, 0] = np.nan
        report = replace(report, position_impulses=load)
    else:
        state.rigid.orientation[0] = [0, 0, 1, 0]
    with pytest.raises(AssertionError):
        audit_substep(
            scene, scene.initial_state, state, report, 1 / 480, tolerance=1e-7, reaction=True
        )
