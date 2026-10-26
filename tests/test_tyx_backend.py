"""Each selected mode executes actual tyx kernels against independent references."""

import builtins
import importlib
import subprocess
import sys

import numpy as np
import pytest

import phymeo
from phymeo import ParticleModel, ParticleState, SemiImplicitSolver, StepInputs


def implementation():
    result = getattr(phymeo, "TyxSemiImplicitSolver", None)
    assert result is not None, "TyxSemiImplicitSolver is not implemented"
    return result


@pytest.mark.tyx
def test_tyx_free_fall_matches_discrete_solution(tyx_mode, tmp_path):
    model = ParticleModel(np.ones(1, dtype=np.float32), np.array([0.0, -10.0, 0.0]))
    state = ParticleState(
        np.array([[0.0, 10.0, 0.0]], dtype=np.float32), np.zeros((1, 3), dtype=np.float32)
    )
    with implementation()(mode=tyx_mode, cache_root=tmp_path) as solver:
        for _ in range(10):
            result = solver.step(model, state, StepInputs.zeros(1, dtype=np.float32), 0.1)
            assert result.report.accepted
            assert result.report.backend == f"tyx:{tyx_mode}"
            state = result.state
    np.testing.assert_allclose(state.position, [[0.0, 4.5, 0.0]], atol=2e-6)
    np.testing.assert_allclose(state.velocity, [[0.0, -10.0, 0.0]], atol=2e-6)
    assert state.position.dtype == np.float32


@pytest.mark.tyx
def test_tyx_matches_numpy_for_forces_and_fixed_particles_without_aliasing(tyx_mode, tmp_path):
    rng = np.random.default_rng(73)
    count = 17
    inverse_mass = rng.uniform(0.1, 3.0, count).astype(np.float32)
    inverse_mass[::4] = 0.0
    model = ParticleModel(inverse_mass)
    state = ParticleState(
        rng.normal(size=(count, 3)).astype(np.float32),
        rng.normal(size=(count, 3)).astype(np.float32),
    )
    inputs = StepInputs(rng.normal(size=(count, 3)).astype(np.float32))
    before = state.copy()
    reference = SemiImplicitSolver().step(model, state, inputs, 0.025)
    with implementation()(mode=tyx_mode, cache_root=tmp_path) as solver:
        result = solver.step(model, state, inputs, 0.025)
    assert result.report.accepted
    np.testing.assert_allclose(
        result.state.position, reference.state.position, rtol=2e-6, atol=2e-7
    )
    np.testing.assert_allclose(
        result.state.velocity, reference.state.velocity, rtol=2e-6, atol=2e-7
    )
    np.testing.assert_array_equal(state.position, before.position)
    np.testing.assert_array_equal(state.velocity, before.velocity)
    assert not np.shares_memory(result.state.position, state.position)
    assert not np.shares_memory(result.state.velocity, state.velocity)


@pytest.mark.tyx
def test_tyx_rejects_fp64_explicitly(tyx_mode, tmp_path):
    with implementation()(mode=tyx_mode, cache_root=tmp_path) as solver:
        with pytest.raises(ValueError, match="FP32"):
            solver.step(
                ParticleModel(np.ones(1)),
                ParticleState(np.zeros((1, 3)), np.zeros((1, 3))),
                StepInputs.zeros(1),
                0.1,
            )


@pytest.mark.tyx
def test_tyx_empty_arrays_return_without_launch(tyx_mode, tmp_path):
    with implementation()(mode=tyx_mode, cache_root=tmp_path) as solver:
        state = ParticleState(
            np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
        )
        result = solver.step(
            ParticleModel(np.empty(0, dtype=np.float32)),
            state,
            StepInputs.zeros(0, dtype=np.float32),
            0.1,
        )
        assert result.report.accepted
        assert result.state.position.shape == (0, 3)


@pytest.mark.tyx
def test_tyx_nonfinite_output_is_rejected(tyx_mode, tmp_path):
    state = ParticleState(
        np.full((1, 3), 2e38, dtype=np.float32), np.full((1, 3), 2e38, dtype=np.float32)
    )
    with implementation()(mode=tyx_mode, cache_root=tmp_path) as solver:
        result = solver.step(
            ParticleModel(np.ones(1, dtype=np.float32), np.zeros(3)),
            state,
            StepInputs.zeros(1, dtype=np.float32),
            10.0,
        )
    assert not result.report.accepted
    assert result.report.reason
    np.testing.assert_array_equal(result.state.position, state.position)


@pytest.mark.tyx
def test_closed_solver_rejects_step_and_close_is_idempotent(tyx_mode, tmp_path):
    solver = implementation()(mode=tyx_mode, cache_root=tmp_path)
    solver.close()
    solver.close()
    with pytest.raises(RuntimeError, match="closed"):
        solver.step(
            ParticleModel(np.empty(0, dtype=np.float32)),
            ParticleState(np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)),
            StepInputs.zeros(0, dtype=np.float32),
            0.1,
        )


@pytest.mark.parametrize("mode", ["auto", "numpy", "cpu", "hip", None])
def test_unknown_mode_is_rejected_without_fallback(mode):
    cls = implementation()
    with pytest.raises(ValueError, match="mode"):
        cls(mode=mode)


def test_base_import_does_not_load_optional_runtimes():
    implementation()
    code = "import sys, phymeo; assert not {'tyx', 'torch', 'warp'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("name", ["TyxSemiImplicitSolver", "TyxXPBDSolver"])
def test_missing_tyx_dependency_has_actionable_error(monkeypatch, name):
    cls = getattr(phymeo, name)
    real_import = builtins.__import__
    real_import_module = importlib.import_module

    def block_import(name, *args, **kwargs):
        if name == "tyx" or name.startswith("tyx."):
            raise ModuleNotFoundError("No module named 'tyx'", name="tyx")
        return real_import(name, *args, **kwargs)

    def block_import_module(name, *args, **kwargs):
        if name == "tyx" or name.startswith("tyx."):
            raise ModuleNotFoundError("No module named 'tyx'", name="tyx")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_import)
    monkeypatch.setattr(importlib, "import_module", block_import_module)
    with pytest.raises(ImportError, match="requirements/tyx.txt"):
        cls()
