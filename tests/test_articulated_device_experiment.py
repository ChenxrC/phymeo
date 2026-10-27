"""P6 trace identity, independent auditing, source evidence and CLI boundaries."""

import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.tyx


def api():
    name = "phymeo.experiments.articulated_device"
    assert importlib.util.find_spec(name) is not None, "P6 device experiment missing"
    return importlib.import_module(name)


@pytest.mark.parametrize("scene", ["S2", "S3", "S4"])
def test_device_and_reference_keep_frozen_models_with_distinct_precision(scene, tyx_mode):
    from phymeo.core.scene import fingerprint
    from phymeo.experiments.articulated import articulated_scene

    base = articulated_scene(scene)
    fixture = api().articulated_device_scene(scene, tyx_mode)
    assert fingerprint(base.program.scene) == fingerprint(fixture.program.scene)
    assert fixture.settings.tolerance == 2e-6
    assert fixture.config["compute_dtype"] == "float32"
    assert fixture.config["storage_dtype"] == "float64"
    assert fixture.config["backend"] == "tyx:" + tyx_mode
    assert base.settings.tolerance == 1e-8
    run = api().run_articulated_device(scene=scene, mode=tyx_mode, frames=2)
    reference = api().run_articulated_reference(scene=scene, frames=2)
    assert run.metrics["complete_horizon"] is False
    assert reference.config["compute_dtype"] == "float64"
    assert reference.config["mode"] == "reference"
    assert np.all(run.arrays["substep_completed_launches"] > 0)
    np.testing.assert_allclose(
        run.arrays["robot_q"], reference.arrays["robot_q"], atol=5e-4, rtol=0
    )
    api().validate_articulated_device_run(run, expected_mode=tyx_mode, expected_scene=scene)
    with pytest.raises(ValueError, match="200|complete"):
        api().validate_articulated_device_run(run, require_complete=True)


@pytest.mark.parametrize(
    "field,value", [("compute_dtype", "float64"), ("variant", "S3"), ("mode", "reference")]
)
def test_device_config_cannot_silently_change(field, value, tyx_mode):
    run = api().run_articulated_device(scene="S2", mode=tyx_mode, frames=1)
    run.config[field] = value
    with pytest.raises(ValueError):
        api().validate_articulated_device_run(run, expected_mode=tyx_mode, expected_scene="S2")


def test_missing_substep_or_execution_is_not_passing_evidence(tyx_mode):
    run = api().run_articulated_device(scene="S2", mode=tyx_mode, frames=1)
    saved = run.arrays["substep_completed_launches"].copy()
    run.arrays["substep_completed_launches"][:] = 0
    with pytest.raises(ValueError, match="execution|launch"):
        api().validate_articulated_device_run(run)
    run.arrays["substep_completed_launches"][:] = saved
    run.arrays["substep_robot_v"] = run.arrays["substep_robot_v"][:-1]
    with pytest.raises(ValueError, match="schema|shape"):
        api().validate_articulated_device_run(run)


def test_saved_artifact_rehash_cannot_certify_fictitious_contact_load(tyx_mode, tmp_path):
    from phymeo.experiments.rigid_cloth import trajectory_digest
    from phymeo.validation.articulated import EVENT_INTEGER, EVENT_SHAPES

    run = api().run_articulated_device(scene="S2", mode=tyx_mode, frames=1)
    target = tmp_path / "trace"
    api().save_articulated_device_run(run, target)
    loaded = api().load_articulated_device_run(target, expected_scene="S2", expected_mode=tyx_mode)
    assert loaded.metrics["audited_substeps"] == 2
    with pytest.raises(FileExistsError):
        api().save_articulated_device_run(run, target)
    for name, shape in EVENT_SHAPES.items():
        run.arrays["event_" + name] = np.zeros(
            (1, *shape), dtype=np.int64 if name in EVENT_INTEGER else np.float64
        )
    run.arrays["event_h"][0] = 1 / 4000
    run.arrays["event_phase"][0] = 1
    run.arrays["event_world_point"][0] = run.arrays["cloth_position"][0, 0]
    run.arrays["event_node_position_before"][0] = run.arrays["cloth_position"][0, 0]
    run.arrays["event_node_load"][0] = [0.0, 0.1, 0.0]
    run.arrays["event_robot_load"][0] = [0.0, -0.1, 0.0]
    run.arrays["event_robot_generalized_load"][0] = [-0.1]
    run.arrays["event_normal"][0] = [0.0, -1.0, 0.0]
    np.savez_compressed(target / "trajectory.npz", **run.arrays)
    metadata = json.loads((target / "run.json").read_text(encoding="utf-8"))
    metadata["metrics"]["trajectory_sha256"] = trajectory_digest(run.arrays)
    (target / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError), match="independent P6"):
        api().load_articulated_device_run(target, expected_scene="S2", expected_mode=tyx_mode)


def test_device_cli_explicit_mode_and_output_no_overwrite(tyx_mode, tmp_path):
    api()
    script = Path(__file__).resolve().parents[1] / "examples/articulated_device.py"
    command = [
        sys.executable,
        str(script),
        "--scene",
        "S2",
        "--mode",
        tyx_mode,
        "--frames",
        "1",
        "--output",
        str(tmp_path / "cli"),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    api().load_articulated_device_run(tmp_path / "cli", expected_scene="S2", expected_mode=tyx_mode)
    assert subprocess.run(command, capture_output=True, text=True).returncode != 0


def test_reference_is_explicit_and_not_an_automatic_device_fallback(tyx_mode, monkeypatch):
    implementation = api()

    def fail(*args, **kwargs):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(implementation, "make_tyx_articulated_runtime", fail)
    with pytest.raises(RuntimeError, match="unavailable"):
        implementation.run_articulated_device(scene="S2", mode=tyx_mode, frames=1)
    assert (
        implementation.run_articulated_reference(scene="S2", frames=1).metrics["accepted_frames"]
        == 1
    )
