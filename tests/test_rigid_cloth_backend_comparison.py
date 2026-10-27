"""Trace integrity and numerical comparison gates independent of optional tyx."""

import importlib
from dataclasses import replace

import numpy as np
import pytest

from phymeo.experiments.rigid_cloth import SceneRun, run_scene, save_run


@pytest.mark.tyx
def test_device_environment_records_the_installed_distribution():
    from importlib.metadata import version

    from phymeo.experiments.backend_comparison import _device_environment

    pytest.importorskip("tyx")
    result = _device_environment("interpret")
    assert result == {"tyx_version": version("tyx-compute"), "mode": "interpret"}


def test_loaded_numeric_trace_is_verified_against_its_record(tmp_path):
    api = importlib.import_module("phymeo.experiments.rigid_cloth")
    original = run_scene(frames=1)
    output = tmp_path / "trace"
    save_run(original, output)
    loaded = api.load_run(output)
    np.testing.assert_array_equal(
        loaded.arrays["particle_position"], original.arrays["particle_position"]
    )
    altered = dict(original.arrays)
    altered["particle_position"] = altered["particle_position"].copy()
    altered["particle_position"][0, 0, 0] += 1
    np.savez_compressed(output / "trajectory.npz", **altered)
    with pytest.raises(ValueError, match="digest"):
        api.load_run(output)


def comparison_records():
    config = dict(
        backend="numpy_colored",
        dtype="float32",
        distance_order="stable_colors",
        projection_arithmetic="fp64",
        interaction="two_way_dynamic",
        sphere_mass_kg=0.05,
    )
    arrays = dict(
        time=np.array([0.0, 1.0]),
        particle_position=np.zeros((2, 1, 3), np.float32),
        rigid_position=np.zeros((2, 1, 3), np.float32),
    )
    reference = SceneRun(config, {"execution_backend": "numpy"}, arrays)
    candidate_arrays = {name: value.copy() for name, value in arrays.items()}
    candidate_arrays["particle_position"][1, 0, 0] = 1e-3
    candidate = SceneRun(
        config | {"backend": "cuda", "projection_arithmetic": "fp32"},
        {"execution_backend": "tyx:cuda"},
        candidate_arrays,
    )
    return reference, candidate


def test_backend_comparison_checks_absolute_position_bound_and_physical_configuration():
    compare = importlib.import_module("phymeo.experiments.backend_comparison").compare_backend_runs
    reference, candidate = comparison_records()
    metrics = compare(reference, candidate)
    assert metrics["max_cloth_position_difference_m"] == pytest.approx(1e-3)
    assert metrics["max_sphere_position_difference_m"] == 0
    with pytest.raises(ValueError, match="configuration"):
        compare(reference, replace(candidate, config=candidate.config | {"sphere_mass_kg": 0.1}))
    candidate.arrays["particle_position"][1, 0, 0] = 3e-3
    with pytest.raises(AssertionError, match="2e-3"):
        compare(reference, candidate)
