"""Frozen multi-domain experiments must retain actual numeric evidence."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def api():
    return importlib.import_module("phymeo.experiments.articulated")


@pytest.mark.parametrize(
    "name,nodes,spheres",
    [("S2", {"cloth": 9}, 1), ("S3", {"soft": 18}, 1), ("S4", {"cloth": 9, "soft": 18}, 2)],
)
def test_frozen_scene_has_shared_robot_and_exact_parameters(name, nodes, spheres):
    fixture = api().articulated_scene(name)
    assert set(fixture.initial.domains) == {"robot", *nodes}
    assert fixture.settings.substeps == 2 and fixture.settings.tolerance == 1e-8
    assert fixture.settings.position_iterations == fixture.settings.velocity_iterations == 200
    robot = fixture.program.participants["robot"].model
    np.testing.assert_array_equal(robot.mass, [0.5])
    np.testing.assert_array_equal(robot.joint_position, [[0, 0.06, 0]])
    np.testing.assert_array_equal(robot.joint_axis, [[0, 1, 0]])
    np.testing.assert_array_equal(fixture.initial.domains["robot"].v, [-0.2])
    np.testing.assert_array_equal(fixture.inputs.domains["robot"].effort, [-1])
    assert len(fixture.program.attachments.radii) == spheres
    np.testing.assert_array_equal(fixture.program.attachments.radii, np.full(spheres, 0.05))
    for domain, count in nodes.items():
        model = fixture.program.participants[domain].model
        assert model.particle_count == count
        np.testing.assert_array_equal(model.gravity, np.zeros(3))
        contacts = [contact for contact in fixture.program.contacts if contact.domain == domain]
        assert len(contacts) == 1 and len(contacts[0].node_indices) == count
        center = fixture.program.attachments.local_centers[contacts[0].sphere_indices]
        np.testing.assert_array_equal(
            center[:, 0], np.full(count, -0.15 if domain == "cloth" else 0.15)
        )


@pytest.mark.parametrize("name", ["S2", "S3", "S4"])
def test_short_trace_is_repeatable_numeric_and_is_not_full_acceptance(name):
    first, second = (
        api().run_articulated_scene(scene=name, frames=2),
        api().run_articulated_scene(scene=name, frames=2),
    )
    assert first.config["frames"] == 2 and first.metrics["accepted_frames"] == 2
    assert first.metrics["audited_substeps"] == 4
    assert first.metrics["complete_horizon"] is False
    assert first.arrays["robot_q"].shape == (3, 1)
    assert first.arrays["substep_robot_q"].shape == (4, 1)
    # Constant gravity-free acceleration is -2; q_k = -.2*k*h-h^2*k*(k+1).
    k, h = np.arange(1, 5), 1 / 4000
    np.testing.assert_allclose(
        first.arrays["substep_robot_q"][:, 0], -0.2 * k * h - h**2 * k * (k + 1), atol=1e-15
    )
    assert first.arrays.keys() == second.arrays.keys()
    for key, value in first.arrays.items():
        assert value.dtype.kind != "O" and np.isfinite(value).all()
        np.testing.assert_array_equal(value, second.arrays[key])
    api().validate_articulated_run(first)
    with pytest.raises(ValueError, match="complete|200"):
        api().validate_articulated_run(first, require_complete=True)


def test_saved_trace_checks_numeric_hash_configuration_summary_and_source(tmp_path):
    run = api().run_articulated_scene(scene="S4", frames=2)
    target = tmp_path / "run"
    api().save_articulated_run(run, target)
    loaded = api().load_articulated_run(target)
    for key in run.arrays:
        np.testing.assert_array_equal(run.arrays[key], loaded.arrays[key])
    with pytest.raises(FileExistsError):
        api().save_articulated_run(run, target)
    record_path = target / "run.json"
    original = record_path.read_text(encoding="utf-8")
    for section, field, value in (
        ("config", "dt", 1.0),
        ("metrics", "accepted_frames", 1),
        ("provenance", "source_sha256", "invalid"),
    ):
        record = json.loads(original)
        record[section][field] = value
        record_path.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(ValueError):
            api().load_articulated_run(target)
    record_path.write_text(original, encoding="utf-8")
    run.arrays["robot_q"][-1] += 1
    np.savez_compressed(target / "trajectory.npz", **run.arrays)
    with pytest.raises(ValueError, match="digest"):
        api().load_articulated_run(target)


@pytest.mark.parametrize("frames", [0, -1, True, 1.5])
def test_experiment_rejects_invalid_horizon(frames):
    with pytest.raises(ValueError):
        api().run_articulated_scene(frames=frames)


def test_unknown_scene_rejected():
    with pytest.raises(ValueError):
        api().articulated_scene("newton")


def test_cli_saves_and_failed_configuration_exits_nonzero(tmp_path):
    script = Path(__file__).resolve().parents[1] / "examples" / "articulated_scene.py"
    output = tmp_path / "trace"
    command = [
        sys.executable,
        str(script),
        "--scene",
        "S4",
        "--frames",
        "2",
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert api().load_articulated_run(output).metrics["accepted_frames"] == 2
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode != 0


def test_three_domain_trace_records_both_actual_interactions_and_balanced_loads():
    # First collision is around frame 83; this covers actual force exchange without
    # substituting a short smoke run for the separately frozen 200-frame acceptance.
    run = api().run_articulated_scene(scene="S4", frames=100)
    assert run.metrics["contact_substeps"]["cloth"] >= 5
    assert run.metrics["contact_substeps"]["soft"] >= 5
    assert run.metrics["max_robot_balance_error"] <= 1e-10
    assert run.metrics["max_penetration"] <= 2e-7
    assert run.metrics["max_closing_speed"] <= 2e-7
    assert set(run.arrays["event_domain"]) == {0, 1}
    assert len(run.arrays["event_phase"]) > 0
    np.testing.assert_array_equal(run.arrays["event_node_load"], -run.arrays["event_robot_load"])
    api().validate_articulated_run(run)
    # A plausible hash cannot replace physics verification. Tamper an event
    # response and then recompute the digest exactly as an erroneous producer could.
    from phymeo.experiments.rigid_cloth import trajectory_digest

    run.arrays["event_robot_generalized_load"][0, 0] += 0.001
    run.metrics["trajectory_sha256"] = trajectory_digest(run.arrays)
    with pytest.raises(RuntimeError, match="audit"):
        api().validate_articulated_run(run)
