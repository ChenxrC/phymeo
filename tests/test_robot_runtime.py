"""Robot physical prediction through the unchanged scene transaction owner."""

import importlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_robot_dynamics import planar_bias, planar_mass
from test_robot_kinematics import robot_model

from phymeo.core.capabilities import SolverFailure
from phymeo.core.robot import RobotInputs, RobotState
from phymeo.core.scene import (
    BackendContext,
    DomainBinding,
    SceneInputs,
    SceneModel,
    SceneState,
    StepSettings,
)
from phymeo.runtime.registry import StateRegistry
from phymeo.runtime.robot_family import robot_family
from phymeo.runtime.scene_stepper import SceneStepper


def api():
    return importlib.import_module("phymeo.solvers.robot")


def builder():
    return importlib.import_module("phymeo.builders.robot").pendulum_chain


def wrapped(state, loads):
    return SceneState({"robot": state}), SceneInputs({"robot": loads})


def test_robot_prediction_is_analytic_semi_implicit_once_per_substep():
    model = robot_model(
        1,
        joint_types=("prismatic",),
        joint_axis=np.array([[0.0, 1, 0]]),
        mass=np.array([2.0]),
        gravity=np.array([0.0, -1, 0]),
    )
    state = RobotState(np.array([0.5]), np.array([0.2]))
    inputs = RobotInputs.zeros(1)
    inputs.effort[:] = 6
    initial, forces = wrapped(state, inputs)
    with api().make_robot_runtime(
        model, settings=StepSettings(substeps=3, record_substeps=True)
    ) as runtime:
        assert type(runtime) is SceneStepper
        result = runtime.step(initial, forces, 0.3)
    assert result.report.accepted and result.report.completed_substeps == 3
    # Constant acceleration 6/2-1=2, h=.1; v=.4,.6,.8, q=.54,.60,.68.
    np.testing.assert_allclose(result.state.domains["robot"].q, [0.68], atol=1e-12)
    np.testing.assert_allclose(result.state.domains["robot"].v, [0.8], atol=1e-12)
    np.testing.assert_allclose(
        [s.domains["robot"].q[0] for s in result.substep_states], [0.54, 0.60, 0.68], atol=1e-12
    )
    np.testing.assert_array_equal(state.q, [0.5])
    np.testing.assert_array_equal(inputs.effort, [6])
    for report in result.report.substeps:
        assert report.position_iterations == report.velocity_iterations == 1
        np.testing.assert_allclose(report.diagnostics["acceleration"], [2], atol=1e-12)
        assert report.diagnostics["equation_residual"] <= 1e-12
    with pytest.raises(ValueError):
        result.report.substeps[0].diagnostics["acceleration"][:] = 99


def test_two_rod_program_matches_independent_equations_and_energies():
    model, state = builder()()
    np.testing.assert_array_equal(state.q, [0.35, -0.2])
    np.testing.assert_array_equal(state.v, [0.1, -0.05])
    initial, forces = wrapped(state, RobotInputs.zeros(2))
    h = 1 / 2000
    expected_v = state.v - h * np.linalg.solve(planar_mass(state.q), planar_bias(state.q, state.v))
    expected_q = state.q + h * expected_v
    with api().make_robot_runtime(model, settings=StepSettings(substeps=1)) as runtime:
        result = runtime.step(initial, forces, h)
    assert result.report.accepted
    after = result.state.domains["robot"]
    np.testing.assert_allclose(after.q, expected_q, atol=1e-14)
    np.testing.assert_allclose(after.v, expected_v, atol=1e-14)
    diagnostic = result.report.substeps[0].diagnostics
    assert diagnostic["kinetic_energy"] == pytest.approx(
        expected_v @ planar_mass(expected_q) @ expected_v / 2, abs=1e-12
    )
    potential = -9.81 * (0.75 * np.cos(expected_q[0]) + 0.25 * np.cos(expected_q.sum()))
    assert diagnostic["potential_energy"] == pytest.approx(potential, abs=1e-12)


def test_robot_program_rejects_a_different_bound_model_and_unsupported_backend():
    model, _ = builder()()
    program = api().RobotProgram(model)
    registry = StateRegistry(defaults=False)
    registry.register(robot_family())
    scene = SceneModel((DomainBinding("robot", "robot", replace(model, gravity=np.zeros(3))),))
    with SceneStepper(scene, program, registry=registry) as runtime:
        state, inputs = wrapped(RobotState(np.zeros(2), np.zeros(2)), RobotInputs.zeros(2))
        with pytest.raises(ValueError, match="bound"):
            runtime.step(state, inputs, 0.01)
    for context in (BackendContext(dtype=np.float32), BackendContext(backend="tyx")):
        with pytest.raises(ValueError, match="unsupported"):
            SceneStepper(program.scene, program, context=context, registry=registry)


def test_robot_model_changes_are_rejected_before_prediction():
    model, state = builder()()
    with api().make_robot_runtime(model) as runtime:
        model.mass.flags.writeable = True
        model.mass[0] = 42
        with pytest.raises(ValueError, match="changed"):
            runtime.step(*wrapped(state, RobotInputs.zeros(2)), 0.01)


@pytest.mark.parametrize("failure", ["predict", "candidate", "diagnostics", "close"])
def test_late_substep_failures_restore_all_robot_fields_and_allow_retry(failure):
    model, state = builder()()
    program = api().RobotProgram(model)
    registry = StateRegistry(defaults=False)
    registry.register(robot_family())
    original, counter, closed = program.create_workspace, [0], []

    def create(*args):
        workspace = original(*args)
        counter[0] += 1
        index = counter[0]
        normal_close = workspace.close

        def close():
            normal_close()
            closed.append(index)
            if index == 3 and failure == "close":
                raise SolverFailure("injected late close")

        workspace.close = close
        if index == 3:
            if failure == "predict":

                def reject():
                    raise SolverFailure("injected late prediction")

                workspace.predict = reject
            elif failure == "candidate":
                workspace.result = lambda: SceneState(
                    {"robot": RobotState(np.zeros(3), np.zeros(3))}
                )
            elif failure == "diagnostics":
                workspace.diagnostics = lambda: {"invalid": np.nan}
        return workspace

    program.create_workspace = create
    initial, inputs = wrapped(state, RobotInputs.zeros(2))
    with SceneStepper(
        program.scene, program, StepSettings(substeps=3, record_substeps=True), registry=registry
    ) as runtime:
        result = runtime.step(initial, inputs, 0.003)
        assert not result.report.accepted and result.report.completed_substeps == 2
        assert result.substep_states == () and closed == [1, 2, 3]
        np.testing.assert_array_equal(result.state.domains["robot"].q, state.q)
        np.testing.assert_array_equal(result.state.domains["robot"].v, state.v)
        retry = runtime.step(initial, inputs, 0.003)
        assert retry.report.accepted
        result.state.domains["robot"].q[:] = 42
        np.testing.assert_array_equal(state.q, [0.35, -0.2])


def test_numerically_failed_robot_prediction_is_rejected_and_owned():
    model = robot_model(1)
    state = RobotState(np.zeros(1), np.array([1e308]))
    with api().make_robot_runtime(model) as runtime:
        result = runtime.step(*wrapped(state, RobotInputs.zeros(1)), 0.001)
    assert not result.report.accepted
    np.testing.assert_array_equal(result.state.domains["robot"].v, [1e308])
    assert not np.shares_memory(state.v, result.state.domains["robot"].v)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_builder_rejects_invalid_dof_count(count):
    with pytest.raises((TypeError, ValueError)):
        builder()(count)


def test_factory_domains_close_and_public_exports():
    import phymeo

    for name in (
        "RobotModel",
        "RobotState",
        "RobotInputs",
        "RobotPointPort",
        "RobotProgram",
        "make_robot_runtime",
        "pendulum_chain",
    ):
        assert name in phymeo.__all__ and getattr(phymeo, name) is not None
    model, state = builder()(1)
    with api().make_robot_runtime(model, domain="arm") as runtime:
        result = runtime.step(
            SceneState({"arm": state}), SceneInputs({"arm": RobotInputs.zeros(1)}), 0.001
        )
        assert result.report.accepted
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.step(SceneState({"arm": state}), SceneInputs({"arm": RobotInputs.zeros(1)}), 0.001)


def test_robot_trace_is_repeatable_numeric_complete_and_refuses_overwrite(tmp_path):
    module = importlib.import_module("phymeo.experiments.robot")
    from phymeo.experiments.rigid_cloth import load_run, save_run

    first, second = module.run_pendulum(steps=4), module.run_pendulum(steps=4)
    assert set(first.arrays) == {"q", "v", "energy", "time"}
    assert first.arrays["q"].shape == (5, 2)
    assert first.arrays["energy"].shape == (5, 2)
    for key in first.arrays:
        np.testing.assert_array_equal(first.arrays[key], second.arrays[key])
    destination = tmp_path / "robot"
    save_run(first, destination)
    loaded = load_run(destination)
    np.testing.assert_array_equal(loaded.arrays["q"], first.arrays["q"])
    assert loaded.config["integrator"] == "semi_implicit_euler"
    with pytest.raises(FileExistsError):
        save_run(second, destination)


def test_robot_cli_saves_source_and_fails_nonzero_for_invalid_run(tmp_path):
    command = [sys.executable, str(Path(__file__).parents[1] / "examples/robot_pendulum.py")]
    target = tmp_path / "trace"
    success = subprocess.run(
        [*command, "--steps", "2", "--output", str(target)], capture_output=True, text=True
    )
    assert success.returncode == 0, success.stderr
    record = json.loads((target / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "complete" and record["provenance"]["source_sha256"]
    for options in (["--output", str(target)], ["--steps", "0", "--output", str(tmp_path / "bad")]):
        failure = subprocess.run([*command, *options], capture_output=True, text=True)
        assert failure.returncode != 0
