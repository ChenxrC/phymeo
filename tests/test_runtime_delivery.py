"""Public and experiment entrypoints exercise the new packaged schedule."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import phymeo
from phymeo.experiments.rigid_cloth import run_scene


def test_public_free_program_runs_without_loading_optional_backend():
    model = phymeo.ParticleModel(np.ones(1), np.array([0.0, -2.0, 0.0]))
    scene = phymeo.SceneModel((phymeo.DomainBinding("dust", "particles", model),))
    state = phymeo.SceneState({"dust": phymeo.ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))})
    inputs = phymeo.SceneInputs({"dust": phymeo.StepInputs.zeros(1)})
    with phymeo.SceneStepper(
        scene, phymeo.FreeSceneProgram(), phymeo.StepSettings(substeps=2)
    ) as runner:
        result = runner.step(state, inputs, 0.2)
    assert result.report.accepted
    np.testing.assert_allclose(result.state.domains["dust"].position, [[0, -0.06, 0]])


def test_new_experiment_records_scheduler_phases_and_unchanged_audit_arrays():
    legacy = run_scene(frames=1)
    staged = run_scene(frames=1, runtime="scene")
    assert staged.config["scheduler"] == "scene"
    assert "scheduler" not in legacy.config
    assert set(staged.arrays) == set(legacy.arrays)
    for name in staged.arrays:
        np.testing.assert_array_equal(staged.arrays[name], legacy.arrays[name])
    assert staged.metrics["trajectory_sha256"] == legacy.metrics["trajectory_sha256"]
    assert staged.metrics["stage_seconds"]["predict"] > 0
    assert staged.metrics["accepted_frames"] == 1 and staged.metrics["audited_substeps"] == 4


@pytest.mark.parametrize(
    "kwargs", [{"runtime": "unknown"}, {"runtime": "scene", "reaction": False}]
)
def test_invalid_scheduler_or_prescribed_motion_is_explicitly_rejected(kwargs):
    with pytest.raises(ValueError):
        run_scene(frames=1, **kwargs)


def test_headless_runtime_example_saves_one_audited_frame(tmp_path):
    destination = tmp_path / "scene"
    result = subprocess.run(
        [
            sys.executable,
            "examples/coupled_runtime.py",
            "--frames",
            "1",
            "--output",
            str(destination),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "saved" in result.stdout
    record = json.loads((destination / "run.json").read_text())
    assert record["config"]["scheduler"] == "scene"
    assert record["metrics"]["audited_substeps"] == 4
