"""External trajectory drivers own no dynamic robot state or hidden index."""

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from phymeo.core.capabilities import SolverFailure
from phymeo.core.scene import SceneState, StepSettings
from phymeo.experiments.articulated import articulated_scene


def api():
    return importlib.import_module("phymeo.experiments.prescribed_articulated")


def setup(substeps=8):
    fixture = articulated_scene("S4")
    trajectory = api().free_robot_trajectory(fixture, substeps=substeps)
    program = api().PrescribedArticulatedProgram(fixture, trajectory)
    return fixture, trajectory, program, program.initial, program.inputs


def test_free_trajectory_matches_independent_discrete_equations_and_is_owned():
    fixture, trajectory, _, _, _ = setup()
    k, h = np.arange(9), fixture.config["h"]
    np.testing.assert_allclose(trajectory.q[:, 0], -0.2 * k * h - h**2 * k * (k + 1), atol=1e-15)
    np.testing.assert_allclose(trajectory.v[:, 0], -0.2 - 2 * k * h, atol=1e-15)
    assert not trajectory.q.flags.writeable and not trajectory.v.flags.writeable
    q = trajectory.q.copy()
    copied = api().PrescribedTrajectory(q, trajectory.v, h)
    q[:] = 99
    np.testing.assert_array_equal(copied.q, trajectory.q)


def test_prescribed_scene_has_only_nodes_and_commits_explicit_consumption_index():
    fixture, trajectory, program, initial, inputs = setup()
    assert {d.name for d in program.scene.domains} == {"cloth", "soft"}
    assert set(initial.domains) == {"cloth", "soft"}
    assert set(initial.history) == {"driver_index"}
    assert program.capabilities.history == "explicit"
    with api().make_prescribed_runtime(
        program, settings=StepSettings(substeps=3, record_substeps=True)
    ) as runtime:
        result = runtime.step(initial, inputs, 3 * trajectory.h)
    assert result.report.accepted
    assert result.state.history["driver_index"] == 3 and initial.history["driver_index"] == 0
    np.testing.assert_array_equal(
        [state.history["driver_index"] for state in result.substep_states], [1, 2, 3]
    )
    for i, report in enumerate(result.report.substeps):
        diagnostic = report.diagnostics["participants"]["driver"]
        np.testing.assert_array_equal(diagnostic["q"], trajectory.q[i + 1])
        np.testing.assert_array_equal(diagnostic["v"], trajectory.v[i + 1])
    for name in fixture.node_names:
        # Native material evaluation can have roundoff-sized rest forces. Compare
        # actual uncontacted native steps, rather than asserting bitwise zero drift.
        from phymeo.core.scene import SceneInputs
        from phymeo.runtime.scene_stepper import SceneStepper

        native = fixture.program.participants[name].program
        with SceneStepper(
            native.scene, native, StepSettings(substeps=3), registry=program.registry
        ) as runtime:
            expected = runtime.step(
                SceneState({name: initial.domains[name]}),
                SceneInputs({name: inputs.domains[name]}),
                3 * trajectory.h,
            )
        np.testing.assert_array_equal(
            result.state.domains[name].position, expected.state.domains[name].position
        )
        np.testing.assert_array_equal(
            result.state.domains[name].velocity, expected.state.domains[name].velocity
        )


@pytest.mark.parametrize("failure", ["prediction", "close", "candidate"])
def test_late_driver_failure_rolls_back_index_and_both_node_domains(failure):
    _, trajectory, program, initial, inputs = setup()
    create, calls = program.create_workspace, [0]

    def wrapped(*args):
        workspace = create(*args)
        calls[0] += 1
        if calls[0] == 3:

            def fail():
                raise SolverFailure("injected driver failure")

            if failure == "prediction":
                workspace.predict = fail
            elif failure == "close":
                close = workspace.close

                def fail_close():
                    close()
                    fail()

                workspace.close = fail_close
            else:
                workspace.result = lambda: SceneState(
                    initial.domains, {"driver_index": np.array(99.0)}
                )
        return workspace

    program.create_workspace = wrapped
    with api().make_prescribed_runtime(
        program, settings=StepSettings(substeps=3, record_substeps=True)
    ) as runtime:
        rejected = runtime.step(initial, inputs, 3 * trajectory.h)
        assert not rejected.report.accepted and rejected.report.completed_substeps == 2
        assert rejected.substep_states == ()
        np.testing.assert_array_equal(rejected.state.history["driver_index"], 0)
        for name, state in initial.domains.items():
            np.testing.assert_array_equal(rejected.state.domains[name].position, state.position)
            np.testing.assert_array_equal(rejected.state.domains[name].velocity, state.velocity)
        retried = runtime.step(initial, inputs, 3 * trajectory.h)
        assert retried.report.accepted and retried.state.history["driver_index"] == 3
    np.testing.assert_array_equal(trajectory.q[0], [0])


@pytest.mark.parametrize("kind", ["shape", "precision", "finite", "short", "h", "not_free"])
def test_invalid_trajectory_is_rejected(kind):
    fixture, trajectory, _, _, _ = setup()
    q, v, h = trajectory.q.copy(), trajectory.v.copy(), trajectory.h
    if kind == "shape":
        q = q.ravel()
    elif kind == "precision":
        q = q.astype(np.float32)
    elif kind == "finite":
        q[2, 0] = np.nan
    elif kind == "short":
        q, v = q[:1], v[:1]
    elif kind == "h":
        h = -h
    else:
        q[4, 0] += 0.01
    with pytest.raises((TypeError, ValueError)):
        api().PrescribedArticulatedProgram(fixture, api().PrescribedTrajectory(q, v, h))


def test_driver_source_change_step_mismatch_and_exhaustion_are_visible():
    _, trajectory, program, initial, inputs = setup(substeps=2)
    with api().make_prescribed_runtime(program) as runtime:
        with pytest.raises(ValueError, match="step|time"):
            runtime.step(initial, inputs, trajectory.h * 4)
        first = runtime.step(initial, inputs, trajectory.h * 2)
        assert first.report.accepted
        exhausted = runtime.step(first.state, inputs, trajectory.h * 2)
        assert not exhausted.report.accepted and "exhaust" in exhausted.report.reason
        assert exhausted.state.history["driver_index"] == 2
        program.trajectory.q.flags.writeable = True
        program.trajectory.q[0] = 99
        with pytest.raises(ValueError, match="changed"):
            runtime.step(initial, inputs, trajectory.h * 2)


def test_boundary_port_reports_zero_response_at_a_shared_world_point():
    _, trajectory, program, initial, inputs = setup()
    from phymeo.core.scene import BackendContext

    workspace = program.create_workspace(
        program.scene, initial, inputs, trajectory.h, StepSettings(), BackendContext()
    )
    workspace.predict()
    driver = workspace.participants["driver"]
    local = np.array([[-0.15, -0.05, 0]])
    port = driver.point_port(np.array([0]), local)
    np.testing.assert_allclose(
        port.positions(), [[-0.15, 0.01 + trajectory.q[1, 0], 0]], atol=1e-15
    )
    load = np.array([[0.0, -1, 0]])
    np.testing.assert_array_equal(port.response(load), np.zeros((1, 3)))
    before_q, before_v = driver.configuration(), driver.velocity()
    driver.apply_position_load(np.array([0]), local, load)
    driver.prepare_velocity()
    driver.apply_velocity_impulses(np.array([0]), local, load)
    np.testing.assert_array_equal(driver.configuration(), before_q)
    np.testing.assert_array_equal(driver.velocity(), before_v)
    workspace.close()


def test_prescribed_trace_records_real_absorbed_reaction_and_loaded_nodes(tmp_path):
    run = api().run_prescribed_scene(scene="S4", frames=100)
    assert run.config["interaction"] == "prescribed_driver"
    assert "robot_q" not in run.arrays and "substep_robot_q" not in run.arrays
    assert "max_robot_balance_error" not in run.metrics
    assert run.metrics["contact_substeps"]["cloth"] >= 5
    assert run.metrics["contact_substeps"]["soft"] >= 5
    np.testing.assert_array_equal(run.arrays["substep_driver_index"], np.arange(1, 201))
    np.testing.assert_array_equal(run.arrays["driver_q"], run.arrays["source_q"][::2])
    np.testing.assert_array_equal(
        run.arrays["driver_contact_impulse"], -run.arrays["driver_supplied_impulse"]
    )
    assert np.max(np.abs(run.arrays["driver_contact_impulse"])) > 1e-6
    destination = tmp_path / "driver"
    api().save_prescribed_run(run, destination)
    loaded = api().load_prescribed_run(destination)
    for key in run.arrays:
        np.testing.assert_array_equal(run.arrays[key], loaded.arrays[key])
    assert loaded.metrics["complete_horizon"] is False


def test_prescribed_cli_names_the_external_driver_experiment(tmp_path):
    script = Path(__file__).resolve().parents[1] / "examples" / "articulated_scene.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--scene",
            "S3",
            "--experiment",
            "prescribed",
            "--frames",
            "2",
            "--output",
            str(tmp_path / "driver"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    run = api().load_prescribed_run(tmp_path / "driver")
    assert run.config["interaction"] == "prescribed_driver"
    assert run.config["dynamic_domains"] == ["soft"]
