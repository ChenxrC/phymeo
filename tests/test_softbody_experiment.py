"""The frozen loading trace must contain actual deforming solid data and provenance."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from phymeo.experiments.rigid_cloth import load_run, save_run


def experiment():
    return importlib.import_module("phymeo.experiments.softbody")


def test_frozen_soft_block_changes_shape_retains_fixed_nodes_and_repeats(tmp_path):
    first = experiment().run_soft_block(steps=4)
    second = experiment().run_soft_block(steps=4)
    assert first.arrays["position"].shape == (5, 18, 3)
    assert first.arrays["velocity"].shape == (5, 18, 3)
    assert first.arrays["support_force"].shape == (4, 18, 3)
    assert first.metrics["compression"] > 0
    assert first.metrics["min_det_f"] > 0.2
    assert first.metrics["min_singular_value"] >= 0.6
    assert first.metrics["max_singular_value"] <= 1.4
    np.testing.assert_allclose(
        first.arrays["external_force"].sum(axis=0), [0, -0.05, 0], atol=1e-15
    )
    fixed = first.arrays["fixed"]
    for step in range(5):
        np.testing.assert_array_equal(
            first.arrays["position"][step, fixed], first.arrays["position"][0, fixed]
        )
        np.testing.assert_array_equal(first.arrays["velocity"][step, fixed], 0.0)
    for key in first.arrays:
        np.testing.assert_array_equal(first.arrays[key], second.arrays[key])
    save_run(first, tmp_path / "soft")
    restored = load_run(tmp_path / "soft")
    np.testing.assert_array_equal(restored.arrays["position"], first.arrays["position"])
    with pytest.raises(FileExistsError):
        save_run(first, tmp_path / "soft")


def test_soft_block_cli_has_nonzero_failure_and_saves_provenance(tmp_path):
    command = [sys.executable, str(Path(__file__).parents[1] / "examples/softbody_block.py")]
    target = tmp_path / "solid"
    result = subprocess.run(
        [*command, "--steps", "3", "--output", str(target)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    record = json.loads((target / "run.json").read_text(encoding="utf-8"))
    assert record["provenance"]["source_sha256"] and record["metrics"]["compression"] > 0
    for options in (
        ["--output", str(target)],
        ["--steps", "0", "--output", str(tmp_path / "invalid")],
    ):
        result = subprocess.run([*command, *options], capture_output=True, text=True)
        assert result.returncode != 0


def test_softbody_public_entrypoints_do_not_require_optional_runtimes():
    import phymeo

    for name in (
        "TetrahedralModel",
        "TetrahedralProgram",
        "TetrahedralPointPort",
        "ClothProgram",
        "make_tetrahedral_runtime",
        "make_cloth_runtime",
        "tetrahedral_block",
    ):
        assert name in phymeo.__all__ and getattr(phymeo, name) is not None


def test_soft_block_failure_never_returns_a_successful_partial_trace(monkeypatch):
    module = experiment()
    normal = module.make_tetrahedral_runtime

    def bad_runtime(*args, **kwargs):
        runtime = normal(*args, **kwargs)
        step = runtime.step
        calls = [0]

        def reject(*step_args):
            calls[0] += 1
            if calls[0] == 3:
                raise RuntimeError("forced material failure")
            return step(*step_args)

        runtime.step = reject
        return runtime

    monkeypatch.setattr(module, "make_tetrahedral_runtime", bad_runtime)
    with pytest.raises(RuntimeError, match="forced material"):
        module.run_soft_block(steps=4)
