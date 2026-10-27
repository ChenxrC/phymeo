"""Independent arithmetic and ownership tests for the explicit P6 execution boundary."""

import importlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.tyx


def implementation():
    name = "phymeo.backends.tyx_math"
    assert importlib.util.find_spec(name) is not None, "P6 TyxMath is not implemented"
    return importlib.import_module(name)


@pytest.mark.parametrize("value", [1e40, -1e40, 1e-50, -1e-50, np.inf, np.nan, 1e-39])
def test_nonrepresentable_conversion_rejected(value, tyx_mode):
    with pytest.raises((ValueError, FloatingPointError), match="FP32|finite|represent"):
        implementation().as_compute(np.array([value]), "load")


def test_conversion_owned_and_finite(tyx_mode):
    source = np.array([0.0, 1.25, -2.5, 2e-38])
    result = implementation().as_compute(source, "state")
    assert result.dtype == np.float32
    assert not np.shares_memory(source, result)
    np.testing.assert_array_equal(result, source.astype(np.float32))


def test_explicit_modes_and_real_kernel(tyx_mode, tmp_path):
    cls = implementation().TyxMath
    a = np.array([[2.0, -1.0], [3.0, 4.0]])
    b = np.array([[1.0, 5.0], [-2.0, 1.0]])
    before = a.copy()
    with cls(tyx_mode, cache_root=tmp_path) as engine:
        product = engine.matmul(a, b)
        np.testing.assert_allclose(product, [[4.0, 9.0], [-5.0, 19.0]], rtol=0, atol=1e-6)
        value = engine.axpy(a, b, 0.5)
        np.testing.assert_allclose(value, a + 0.5 * b, rtol=0, atol=1e-6)
        assert engine.dot(a.reshape(-1), b.reshape(-1)) == -5
        stats = engine.stats()
        assert stats["mode"] == tyx_mode
        assert stats["launches"] >= 3
        assert stats["compute_dtype"] == "float32"
        if tyx_mode == "cuda":
            assert stats["uploads"] > 0 and stats["downloads"] > 0
        product[:] = 999
        np.testing.assert_array_equal(a, before)
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.matmul(a, b)


@pytest.mark.parametrize("size", [1, 2, 8])
def test_spd_solve_matches_independent_equation(size, tyx_mode, tmp_path, monkeypatch):
    rng = np.random.default_rng(651 + size)
    factor = rng.normal(size=(size, size))
    matrix = factor.T @ factor + np.eye(size)
    expected = np.arange(1, size + 1) / size
    rhs = matrix @ expected
    before = matrix.copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("host solve must not supply device arithmetic")

    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with implementation().TyxMath(tyx_mode, cache_root=tmp_path) as engine:
        answer = engine.solve_spd(matrix, rhs)
    np.testing.assert_allclose(answer, expected, atol=2e-5, rtol=2e-5)
    assert np.max(np.abs(matrix @ answer - rhs)) / max(1.0, np.max(np.abs(rhs))) <= 2e-5
    np.testing.assert_array_equal(matrix, before)


@pytest.mark.parametrize(
    "matrix",
    [np.array([[1.0, 2.0], [2.0, 1.0]]), np.zeros((2, 2)), np.array([[1.0, 0.1], [0.2, 1.0]])],
)
def test_invalid_mass_not_regularized(matrix, tyx_mode, tmp_path):
    from phymeo.core.capabilities import SolverFailure

    with implementation().TyxMath(tyx_mode, cache_root=tmp_path) as engine:
        with pytest.raises((ValueError, SolverFailure), match="positive|SPD|symmetric|mass"):
            engine.solve_spd(matrix, np.ones(2))


@pytest.mark.parametrize("operation", ["axpy", "matmul", "dot"])
def test_shapes_rejected_before_launch(operation, tyx_mode, tmp_path):
    with implementation().TyxMath(tyx_mode, cache_root=tmp_path) as engine:
        arguments = (np.ones((2, 3)), np.ones((4, 2)))
        if operation == "axpy":
            arguments += (1.0,)
        with pytest.raises(ValueError):
            getattr(engine, operation)(*arguments)
        assert engine.stats()["launches"] == 0


@pytest.mark.parametrize("failure", ["allocation", "kernel", "sync", "download"])
def test_partial_allocation_and_execution_failures_release(
    failure, tyx_mode, tmp_path, monkeypatch
):
    engine = implementation().TyxMath("interpret", cache_root=tmp_path)
    # Fault injection isolates resource ownership independently of GPU availability.
    engine._mode = "cuda"
    allocated = []

    class Buffer:
        def __init__(self, value):
            self.value, self.freed = value.copy(), False

        def free(self):
            self.freed = True

        def to_numpy(self):
            if failure == "download":
                raise RuntimeError("injected download")
            return self.value.copy()

    def allocate(value):
        if failure == "allocation" and len(allocated) == 1:
            raise RuntimeError("injected allocation")
        result = Buffer(value)
        allocated.append(result)
        return result

    def sync():
        if failure == "sync":
            raise RuntimeError("injected sync")

    def kernel(*args, **kwargs):
        if failure == "kernel":
            raise RuntimeError("injected kernel")

    from contextlib import nullcontext

    engine._tx = SimpleNamespace(cuda_array=allocate, use_runtime=lambda runtime: nullcontext())
    monkeypatch.setattr(engine._runtime, "sync", sync)
    with pytest.raises(RuntimeError, match="injected"):
        engine.call(
            kernel, np.ones(2, dtype=np.float32), np.zeros(2, dtype=np.float32), outputs=(1,)
        )
    assert allocated and all(buffer.freed for buffer in allocated)
    engine.close()


@pytest.mark.parametrize(
    "values", [np.array([2**32], dtype=np.uint64), np.array([-(2**32)]), np.array([1.5])]
)
def test_topology_conversion_cannot_wrap_or_truncate(values, tyx_mode):
    with pytest.raises(ValueError, match="i32"):
        implementation().as_index(values, "parents")


def test_nonfinite_arithmetic_is_failure_without_mutating_inputs(tyx_mode, tmp_path):
    from phymeo.core.capabilities import SolverFailure

    a = np.array([[2e20]])
    before = a.copy()
    with implementation().TyxMath(tyx_mode, cache_root=tmp_path) as engine:
        with np.errstate(over="ignore", invalid="ignore"):
            with pytest.raises(SolverFailure, match="non-finite"):
                engine.matmul(a, a)
        np.testing.assert_array_equal(a, before)
        np.testing.assert_array_equal(engine.matmul(np.ones((1, 1)), np.ones((1, 1))), [[1.0]])


def test_mass_equation_gate_rejects_finite_wrong_kernel_result(tyx_mode, tmp_path, monkeypatch):
    from phymeo.core.capabilities import SolverFailure

    with implementation().TyxMath(tyx_mode, cache_root=tmp_path) as engine:
        # Corruption at the selected-backend return boundary must not be accepted
        # just because every component and the failed flag is finite.
        monkeypatch.setattr(
            engine, "call", lambda *args, **kwargs: (np.zeros(2, np.float32), np.zeros(1, np.int32))
        )
        with pytest.raises(SolverFailure, match="residual"):
            engine.solve_spd(np.eye(2), np.ones(2))
