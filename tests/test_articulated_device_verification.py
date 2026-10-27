"""Full-trace comparator must inspect substeps and reject false evidence."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def verifier():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_articulated_device.py"
    spec = importlib.util.spec_from_file_location("verify_device", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def records():
    arrays = {}
    for prefix, count in (("", 3), ("substep_", 4)):
        arrays[prefix + "robot_q"] = np.zeros((count, 1))
        arrays[prefix + "robot_v"] = np.zeros((count, 1))
        for domain in ("cloth", "soft"):
            arrays[prefix + domain + "_position"] = np.zeros((count, 2, 3))
            arrays[prefix + domain + "_velocity"] = np.zeros((count, 2, 3))
    run = SimpleNamespace(
        arrays=arrays,
        config={"node_domains": ["cloth", "soft"]},
        metrics={"contact_substeps": {"cloth": 5, "soft": 5}},
    )
    return run, copy.deepcopy(run), copy.deepcopy(run)


def test_comparator_measures_full_vector_and_all_substeps(records):
    device, repeated, reference = records
    reference.arrays["substep_soft_position"][2, 1] = [0.0003, 0.0004, 0]
    reference.arrays["substep_cloth_velocity"][1, 0] = [0, 0.003, 0.004]
    result = verifier().compare(device, repeated, reference)
    assert result["max_node_position_error_m"] == pytest.approx(0.0005)
    assert result["max_velocity_error_m_s"] == pytest.approx(0.005)
    assert result["exact_repeat_arrays"] == 12


@pytest.mark.parametrize(
    "name,index,value",
    [
        ("substep_robot_q", (1, 0), 0.00051),
        ("substep_robot_v", (1, 0), 0.021),
        ("substep_soft_position", (1, 0), [0.0008, 0.0008, 0]),
        ("substep_cloth_velocity", (1, 0), [0.015, 0.015, 0]),
        ("substep_soft_velocity", (1, 0), [np.nan, 0, 0]),
    ],
)
def test_substep_reference_failures_cannot_hide_behind_matching_frames(records, name, index, value):
    records[2].arrays[name][index] = value
    with pytest.raises(ValueError, match="reference|finite"):
        verifier().compare(*records)


def test_one_bit_repeat_difference_is_rejected(records):
    records[1].arrays["substep_robot_q"][0, 0] = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="non-deterministic"):
        verifier().compare(*records)


def test_same_artifact_root_cannot_count_as_independent_repeat(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        verifier().verify(tmp_path, tmp_path / ".", tmp_path / "reference")


def test_incomplete_saved_trace_cannot_be_full_acceptance(tmp_path):
    from phymeo.experiments.articulated_device import (
        run_articulated_reference,
        save_articulated_device_run,
    )

    save_articulated_device_run(run_articulated_reference(frames=1), tmp_path / "device/S2")
    with pytest.raises(ValueError, match="complete|200|mode"):
        verifier().verify(tmp_path / "device", tmp_path / "repeat", tmp_path / "reference")


def test_cross_source_artifacts_cannot_be_combined_as_one_acceptance(
    tmp_path, records, monkeypatch
):
    module = verifier()
    roots = [tmp_path / name for name in ("device", "repeat", "reference")]
    for index, root in enumerate(roots):
        for scene in ("S2", "S3", "S4"):
            directory = root / scene
            directory.mkdir(parents=True)
            (directory / "run.json").write_text(
                json.dumps({"provenance": {"source_sha256": ("a" if index == 0 else "b") * 64}}),
                encoding="utf-8",
            )

    def load(path, **options):
        assert options["require_complete"] is True
        assert options["expected_scene"] == path.name
        assert options["expected_mode"] == ("reference" if path.parent == roots[2] else "cuda")
        return records[roots.index(path.parent)]

    monkeypatch.setattr(module, "load_articulated_device_run", load)
    with pytest.raises(ValueError, match="source|environment"):
        module.verify(*roots)


def test_cli_rejects_source_drift_during_simulation(monkeypatch, tmp_path):
    from phymeo.experiments.articulated_device import run_articulated_reference

    path = Path(__file__).resolve().parents[1] / "examples/articulated_device.py"
    spec = importlib.util.spec_from_file_location("device_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = run_articulated_reference(frames=1)
    sources = iter([{"source_sha256": "a" * 64}, {"source_sha256": "b" * 64}])
    monkeypatch.setattr(module, "provenance", lambda: next(sources), raising=False)
    monkeypatch.setattr(module, "run_articulated_reference", lambda **kwargs: run)
    monkeypatch.setattr(
        "sys.argv",
        [str(path), "--mode", "reference", "--frames", "1", "--output", str(tmp_path / "out")],
    )
    with pytest.raises(RuntimeError, match="source|environment"):
        module.main()
    assert not (tmp_path / "out").exists()
