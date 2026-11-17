"""Independent I3r equations and bounded CUDA publication/resource contracts.

No production friction evaluator, Newton helper or I3q solver is an oracle.
The HOST resource test uses fake native owners, never a real CUDA release fault.
"""

import json
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import fields, is_dataclass, replace

import numpy as np
import pytest

import experiments.area_pressure.tyx_coupled_friction as gpu
from experiments.area_pressure.coupled_friction_reference import (
    CoupledFrictionProblem,
    CoupledFrictionSample,
)

H = 0.001
PROBLEM_ARRAYS = ("mass_matrix", "damping", "input_velocity", "applied_force")
SAMPLE_ARRAYS = ("point_world", "normal_world", "jacobian_a", "jacobian_b")
SAMPLE_SCALARS = ("stiffness_n_m", "phi_m", "mu", "slip_speed_m_s")
RESULT_ARRAYS = (
    "velocity",
    "free_velocity",
    "normal_force",
    "tangential_force",
    "force_on_a",
    "force_on_b",
    "torque_on_a_world_origin",
    "torque_on_b_world_origin",
    "generalized_force",
    "residual",
    "residual_jacobian",
)


def _sample(point, jacobian, *, stiffness=4000.0, phi=-0.001, mu=0.5):
    jacobian = np.asarray(jacobian, dtype=np.float64)
    return CoupledFrictionSample(
        point,
        (0.0, 0.0, 1.0),
        jacobian,
        np.zeros_like(jacobian),
        stiffness,
        phi,
        mu,
        0.001,
    )


def _problem(mass, damping, velocity, force, samples):
    return CoupledFrictionProblem(mass, damping, velocity, force, H, tuple(samples))


def _bisect(function, low, high):
    """Independent monotone scalar reference on its preregistered bracket."""
    assert function(low) <= 0 <= function(high)
    for _ in range(80):
        mid = (low + high) / 2
        if function(mid) <= 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _frictionless_velocity(problem):
    mass = np.asarray(problem.mass_matrix, dtype=np.float64)
    inverse = np.linalg.inv(mass)
    h = float(problem.h_s)
    free = np.asarray(problem.input_velocity) + h * inverse @ problem.applied_force
    j = np.array([s.normal_world @ (s.jacobian_a - s.jacobian_b) for s in problem.samples])
    stiffness = np.array([s.stiffness_n_m for s in problem.samples])
    phi = np.array([s.phi_m for s in problem.samples])
    block = np.diag(1 / stiffness) + h * h * j @ inverse @ j.T
    rhs = -phi - h * j @ free
    found = []
    for active in ((), (0,), (1,), (0, 1)):
        force = np.zeros(2)
        if len(active) == 1:
            (i,) = active
            force[i] = rhs[i] / block[i, i]
        elif len(active) == 2:
            a, b, c = block[0, 0], block[0, 1], block[1, 1]
            force[:] = (
                (c * rhs[0] - b * rhs[1]) / (a * c - b * b),
                (a * rhs[1] - b * rhs[0]) / (a * c - b * b),
            )
        slack = block @ force - rhs
        if np.all(force >= 0) and all(slack[i] >= -1e-12 for i in range(2) if i not in active):
            if all(abs(slack[i]) <= 1e-12 for i in active):
                found.append(force)
    assert found
    for candidate in found[1:]:
        np.testing.assert_allclose(candidate, found[0], rtol=0, atol=1e-12)
    return free + h * inverse @ j.T @ found[0]


def _cases():
    mapping = np.array(((1.0, 0.0), (0.0, 0.0), (0.0, 1.0)))
    support = _problem(
        0.5 * np.eye(2), (0.0, 0.0), (0.02, 0.0), (0.0, -4.0), (_sample((0, 0, 0), mapping),)
    )
    release = _problem(
        0.5 * np.eye(2),
        (0.0, 0.0),
        (0.02, 0.3),
        (0.0, 0.0),
        (_sample((0, 0, 0), mapping, phi=-0.0001),),
    )
    shared = _problem(
        ((0.5,),), (0.0,), (0.02,), (0.0,), (_sample((0, 0, 0), ((1.0,), (0.0,), (1.0,))),)
    )
    samples = tuple(
        _sample(
            (x, 0.0, -0.01),
            ((1.0, 0.0, -0.01), (0.0, 0.0, 0.0), (0.0, 1.0, -x)),
            stiffness=k,
            phi=phi,
            mu=mu,
        )
        for x, k, phi, mu in ((-0.02, 2000.0, -0.001, 0.4), (0.02, 3000.0, -0.0008, 0.6))
    )
    mass = np.diag((0.5, 0.5, 0.0004))
    damping = np.array((1.0, 2.0, 0.001))
    v0, target = np.array((0.02, -0.01, 0.2)), np.array((0.015, -0.002, 0.3))
    # Hand-manufactured final slips .012 and normal velocities (.004,-.008).
    normal = np.array((1.992, 2.424))
    tangent = -np.array((0.4, 0.6)) * normal * 0.012 / np.sqrt(0.012**2 + 0.001**2)
    generalized = np.array(
        (sum(tangent), sum(normal), -0.01 * sum(tangent) + 0.02 * normal[0] - 0.02 * normal[1])
    )
    force = ((mass + H * np.diag(damping)) @ target - mass @ v0) / H - generalized
    manufactured = _problem(mass, damping, v0, force, samples)
    frictionless = _problem(
        mass,
        (0.0, 0.0, 0.0),
        (0.0, -0.1, 0.2),
        (0.0, 0.0, 0.0),
        tuple(replace(s, mu=0.0) for s in samples),
    )
    empty = replace(support, samples=())
    tangential_root = _bisect(
        lambda v: v + H * 0.5 * 4 / 0.5 * v / np.sqrt(v * v + 0.001**2) - 0.02, 0.0, 0.02
    )
    shared_root = _bisect(
        lambda v: (
            0.5 * (v - 0.02)
            - H * max(0.0, 4000 * (0.001 - H * v)) * (1 - 0.5 * v / np.sqrt(v * v + 0.001**2))
        ),
        0.02,
        0.028,
    )
    return (
        ("support", support, np.array((tangential_root, 0.0))),
        ("release", release, np.array((0.02, 0.3))),
        ("shared-one-dof", shared, np.array((shared_root,))),
        ("two-point-manufactured", manufactured, target),
        ("two-point-frictionless", frictionless, _frictionless_velocity(frictionless)),
        ("empty", empty, np.array((0.02, -0.008))),
    )


def _law(problem, velocity):
    """FP64 evaluation of the equations on the supplied stored input values."""
    v = np.asarray(velocity, dtype=np.float64)
    mass = np.asarray(problem.mass_matrix, dtype=np.float64)
    h = float(problem.h_s)
    response = mass + h * np.diag(np.asarray(problem.damping, dtype=np.float64))
    free = np.linalg.solve(
        response,
        mass @ np.asarray(problem.input_velocity, dtype=np.float64)
        + h * np.asarray(problem.applied_force, dtype=np.float64),
    )
    normals, tangents, forces, moments, velocities = [], [], [], [], []
    generalized = np.zeros(len(v))
    jacobian = response.copy()
    for sample in problem.samples:
        n = np.asarray(sample.normal_world, dtype=np.float64)
        g = np.asarray(sample.jacobian_a, dtype=np.float64) - np.asarray(
            sample.jacobian_b, dtype=np.float64
        )
        relative = g @ v
        vn = float(n @ relative)
        vt = np.cross(n, np.cross(relative, n)) / (n @ n)
        fn = max(0.0, -float(sample.stiffness_n_m) * (float(sample.phi_m) + h * vn))
        speed = np.sqrt(vt @ vt + float(sample.slip_speed_m_s) ** 2)
        mu = float(sample.mu)
        ft = -mu * fn * vt / speed
        f = fn * n + ft
        dnormal = -float(sample.stiffness_n_m) * h * (n @ g) if fn > 0 else np.zeros(len(v))
        tangent_map = (np.eye(3) - np.outer(n, n) / (n @ n)) @ g
        dunit_slip = (np.eye(3) / speed - np.outer(vt, vt) / speed**3) @ tangent_map
        dforce = np.outer(n - mu * vt / speed, dnormal) - mu * fn * dunit_slip
        jacobian -= h * g.T @ dforce
        normals.append(fn)
        tangents.append(ft)
        forces.append(f)
        moments.append(np.cross(sample.point_world, f))
        velocities.append(vt)
        generalized += g.T @ f
    residual = response @ (v - free) - h * generalized
    residual_norm = np.sqrt(max(0.0, float(residual @ np.linalg.solve(response, residual))))
    scale = 1 + np.sqrt(max(0.0, float(free @ response @ free)))
    ns = len(problem.samples)
    return dict(
        free=free,
        response=response,
        normal=np.array(normals),
        tangent=np.array(tangents).reshape(ns, 3),
        force=np.array(forces).reshape(ns, 3),
        moment=np.array(moments).reshape(ns, 3),
        vt=np.array(velocities).reshape(ns, 3),
        generalized=generalized,
        residual=residual,
        residual_norm=residual_norm,
        jacobian=jacobian,
        scale=scale,
    )


def _plain(value):
    if isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=list(value.shape), data=value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(type(value).__name__)


def _save(path, value):
    path.write_text(
        json.dumps(_plain(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def _readonly(array):
    assert not array.flags.writeable
    with pytest.raises(ValueError):
        array.setflags(write=True)


def _packed(result, original):
    assert _plain(result.original_problem) == _plain(original)
    for name in PROBLEM_ARRAYS:
        actual = getattr(result.packed_problem, name)
        assert actual.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(actual, np.asarray(getattr(original, name), dtype=np.float32))
        _readonly(actual)
    assert float(result.packed_problem.h_s) == float(np.float32(original.h_s))
    assert len(result.packed_problem.samples) == len(original.samples)
    for actual, source in zip(result.packed_problem.samples, original.samples, strict=True):
        for name in SAMPLE_ARRAYS:
            value = getattr(actual, name)
            assert value.dtype == np.dtype(np.float32)
            np.testing.assert_array_equal(
                value, np.asarray(getattr(source, name), dtype=np.float32)
            )
            _readonly(value)
        for name in SAMPLE_SCALARS:
            assert float(getattr(actual, name)) == float(np.float32(getattr(source, name)))


def _check(result, problem, reference):
    _packed(result, problem)
    actual = _law(result.packed_problem, result.velocity)
    oracle = _law(problem, reference)
    for name in RESULT_ARRAYS:
        value = getattr(result, name)
        assert isinstance(value, np.ndarray) and value.dtype == np.dtype(np.float32)
        assert np.isfinite(value).all()
        _readonly(value)
    assert result.velocity.shape == np.shape(problem.input_velocity)
    assert result.normal_force.shape == (len(problem.samples),)
    assert result.tangential_force.shape == (len(problem.samples), 3)
    assert np.all(np.abs(result.velocity - reference) <= 5e-5 * (1 + np.abs(reference)))
    for got, expected in (
        (result.normal_force, oracle["normal"]),
        (result.tangential_force, oracle["tangent"]),
        (result.force_on_a, oracle["force"]),
        (result.force_on_b, -oracle["force"]),
    ):
        assert np.all(np.abs(got - expected) <= 0.002 + 0.002 * np.abs(expected))
    for got, expected in (
        (result.torque_on_a_world_origin, oracle["moment"]),
        (result.torque_on_b_world_origin, -oracle["moment"]),
    ):
        assert np.all(np.abs(got - expected) <= 1e-5 + 0.002 * np.abs(expected))
    # Independent actual-input/actual-output qualification, not a HOST polish.
    assert actual["residual_norm"] <= 2e-5 * actual["scale"]
    np.testing.assert_allclose(result.residual_jacobian, actual["jacobian"], rtol=2e-5, atol=2e-7)
    diagnostics = result.diagnostics
    assert diagnostics["device_residual_norm"] <= diagnostics["device_tolerance"]
    assert diagnostics["device_tolerance"] == pytest.approx(1e-5 * actual["scale"], rel=2e-6)
    assert diagnostics["host_tolerance"] == pytest.approx(2e-5 * actual["scale"], rel=1e-12)
    assert diagnostics["host_residual_norm"] == pytest.approx(
        actual["residual_norm"], rel=1e-6, abs=1e-12
    )
    assert bool(diagnostics["host_reference_gate"]) == (
        actual["residual_norm"] <= 1e-10 * actual["scale"]
    )
    assert 0 <= diagnostics["iterations"] <= 24
    assert 0 <= diagnostics["backtracks"] == diagnostics["line_search_trials"] <= 288
    assert diagnostics["pivot_swaps"] >= 0
    assert type(result.trace) is tuple and 2 <= len(result.trace) <= 290
    assert len(result.trace) == diagnostics["line_search_trials"] + 2
    for got, expected in (
        (result.normal_force, actual["normal"]),
        (result.tangential_force, actual["tangent"]),
        (result.generalized_force, actual["generalized"]),
    ):
        assert np.all(np.abs(got - expected) <= 0.002 + 0.002 * np.abs(expected))
    mu = np.array([s.mu for s in result.packed_problem.samples], dtype=np.float64)
    assert np.all(result.normal_force >= 0)
    assert np.all(
        np.linalg.norm(np.asarray(result.tangential_force, dtype=np.float64), axis=1)
        <= mu * result.normal_force + 1e-5
    )
    np.testing.assert_array_equal(result.force_on_b, -result.force_on_a)
    np.testing.assert_array_equal(result.torque_on_b_world_origin, -result.torque_on_a_world_origin)
    with pytest.raises(TypeError):
        diagnostics["iterations"] = -1
    with pytest.raises((AttributeError, TypeError)):
        result.velocity = np.zeros_like(result.velocity)
    return dict(
        actual_input_residual=actual["residual_norm"],
        host_limit=2e-5 * actual["scale"],
        original_input_residual=_law(problem, result.velocity)["residual_norm"],
        jacobian_max_error=float(np.max(np.abs(result.residual_jacobian - actual["jacobian"]))),
        velocity_error=np.abs(result.velocity - reference),
        expected_force=oracle["force"],
        actual_input_force=actual["force"],
        expected_moment=oracle["moment"],
    )


@pytest.mark.cuda
def test_six_single_world_cases_and_reversed_heterogeneous_batch(tmp_path):
    cases = _cases()
    singles, checks = {}, {}
    with gpu.TyxCoupledFrictionWorkspace(capacity=32) as workspace:
        for name, problem, reference in cases:
            before = workspace.stats["launches"]
            results = workspace.solve_many((problem,))
            assert workspace.stats["launches"] == before + 1
            assert type(results) is tuple and len(results) == 1
            _save(
                tmp_path / (name + ".json"),
                dict(original=problem, raw=results[0], reference=reference),
            )
            checks[name] = _check(results[0], problem, reference)
            if name == "two-point-manufactured":
                jacobian = _law(results[0].packed_problem, results[0].velocity)["jacobian"]
                assert np.max(np.abs(jacobian - jacobian.T)) > 1e-6
                assert (
                    np.max(np.abs(results[0].residual_jacobian - results[0].residual_jacobian.T))
                    > 1e-6
                )
            singles[name] = results[0]
        old = _plain(tuple(singles.values()))
        results = workspace.solve_many(tuple(row[1] for row in reversed(cases)))
        _save(tmp_path / "reversed-batch.json", dict(raw=results, stats=workspace.stats))
        assert workspace.last_results is results
        assert len(results) == 6
        for result, (name, problem, reference) in zip(results, reversed(cases), strict=True):
            _check(result, problem, reference)
            for field in RESULT_ARRAYS:
                np.testing.assert_array_equal(getattr(result, field), getattr(singles[name], field))
        assert _plain(tuple(singles.values())) == old
        assert workspace.stats["launches"] == 7
        _save(tmp_path / "independent-checks.json", checks)
    assert workspace.closed


@pytest.mark.cuda
def test_capacity_32_six_dof_eight_sample_split_keeps_original_response(tmp_path):
    _, original, reference = _cases()[4]
    mass = np.eye(6)
    mass[:3, :3] = original.mass_matrix
    samples = []
    for source in original.samples:
        ja, jb = np.zeros((3, 6)), np.zeros((3, 6))
        ja[:, :3], jb[:, :3] = source.jacobian_a, source.jacobian_b
        samples.extend(
            replace(source, jacobian_a=ja, jacobian_b=jb, stiffness_n_m=source.stiffness_n_m / 4)
            for _ in range(4)
        )
    expanded = _problem(
        mass, np.zeros(6), np.r_[original.input_velocity, np.zeros(3)], np.zeros(6), samples
    )
    expected = np.r_[reference, np.zeros(3)]
    with gpu.TyxCoupledFrictionWorkspace(capacity=32) as workspace:
        results = workspace.solve_many((expanded,) * 32)
        _save(
            tmp_path / "capacity32.json",
            dict(raw=results, reference=expected, stats=workspace.stats),
        )
        assert len(results) == 32 and workspace.stats["launches"] == 1
        for result in results:
            _check(result, expanded, expected)
            np.testing.assert_array_equal(result.velocity[3:], np.zeros(3))
        oracle = _law(original, reference)
        combined = np.array([results[0].normal_force[:4].sum(), results[0].normal_force[4:].sum()])
        assert np.all(
            np.abs(combined - oracle["normal"]) <= 0.002 + 0.002 * np.abs(oracle["normal"])
        )


@pytest.mark.cuda
def test_non_diagonal_mass_requires_real_newton_row_pivot(tmp_path):
    problem = _problem(
        ((0.01, 0.1), (0.1, 2.0)),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (_sample((0, 0, 0), ((1.0, 0.0), (0.0, 0.0), (0.0, 1.0)), mu=0.0),),
    )
    force = 4 / 1.004
    reference = np.array((-0.01 * force, 0.001 * force))
    with gpu.TyxCoupledFrictionWorkspace(capacity=1) as workspace:
        (result,) = workspace.solve_many((problem,))
        _save(
            tmp_path / "pivot.json",
            dict(original=problem, raw=result, reference=reference, force=force),
        )
        _check(result, problem, reference)
        assert result.diagnostics["pivot_swaps"] > 0
        assert abs(float(result.normal_force[0]) - force) <= 0.002 + 0.002 * force
        # The first Newton matrix is M+h^2*k*ez*ez^T: the strict swap is
        # required by .1 > .01, independent of a production pivot counter.
        matrix = np.array(((0.01, 0.1), (0.1, 2.0 + H * H * 4000)))
        assert abs(matrix[1, 0]) > abs(matrix[0, 0])
        np.testing.assert_allclose(matrix @ reference, (0.0, 0.004), rtol=0, atol=1e-14)


@pytest.mark.cuda
def test_whole_batch_preflight_numerical_retry_and_late_host_publication(tmp_path, monkeypatch):
    _, support, reference = _cases()[0]
    empty = _cases()[-1][1]
    with gpu.TyxCoupledFrictionWorkspace(capacity=32) as workspace:
        old = workspace.solve_many((empty,))
        previous = _plain(old)
        launches = workspace.stats["launches"]
        broken = replace(support)
        object.__setattr__(broken, "mass_matrix", np.zeros((2, 2)))
        underflow = replace(support, samples=(replace(support.samples[0], stiffness_n_m=1e-50),))
        lost_spd = replace(support, mass_matrix=np.array(((1.0, 1.0), (1.0, 1.0 + 1e-8))))
        for invalid in (
            [support],
            (),
            (support, None),
            (support, broken),
            (support, underflow),
            (support, lost_spd),
            (support,) * 33,
        ):
            with pytest.raises((TypeError, ValueError)):
                workspace.solve_many(invalid)
            assert workspace.stats["launches"] == launches
            assert workspace.last_results is old and _plain(old) == previous
        with pytest.raises(gpu.TyxCoupledFrictionFailure) as failure:
            workspace.solve_many((empty, support), max_iterations=0)
        _save(
            tmp_path / "late-world-budget.json",
            dict(diagnostics=failure.value.diagnostics, stats=workspace.stats),
        )
        assert workspace.stats["launches"] == launches + 1
        assert workspace.last_results is old and _plain(old) == previous and not workspace.faulted
        (retried,) = workspace.solve_many((support,))
        _check(retried, support, reference)
        original_saved = _plain(retried.original_problem)
        object.__setattr__(support, "input_velocity", np.zeros(2))
        assert _plain(retried.original_problem) == original_saved
        publication = workspace.last_results
        assemble = workspace._result
        calls = []

        def late_result(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("late-result-publication")
            return assemble(*args, **kwargs)

        monkeypatch.setattr(workspace, "_result", late_result)
        with pytest.raises(RuntimeError, match="late-result-publication"):
            workspace.solve_many((empty, _cases()[0][1]))
        assert workspace.last_results is publication and not workspace.faulted
        monkeypatch.setattr(workspace, "_result", assemble)
        (result,) = workspace.solve_many((empty,))
        _check(result, empty, _cases()[-1][2])


@pytest.mark.cuda
def test_late_download_keeps_old_results_and_faults_workspace(tmp_path, monkeypatch):
    _, support, _ = _cases()[0]
    workspace = gpu.TyxCoupledFrictionWorkspace(capacity=2)
    try:
        old = workspace.solve_many((support,))
        saved = _plain(old)
        get = workspace._download
        calls = []

        def late_download(name):
            calls.append(name)
            if len(calls) == 2:
                raise RuntimeError("late-device-download")
            return get(name)

        monkeypatch.setattr(workspace, "_download", late_download)
        with pytest.raises(RuntimeError, match="late-device-download"):
            workspace.solve_many((support, support))
        assert len(calls) == 2 and workspace.faulted
        assert workspace.last_results is old and _plain(old) == saved
        count = workspace.stats["launches"]
        with pytest.raises(RuntimeError):
            workspace.solve_many((support,))
        assert workspace.stats["launches"] == count
        _save(
            tmp_path / "download-failure.json",
            dict(old_results=old, stats=workspace.stats, reads=calls),
        )
    finally:
        workspace.close()
    assert workspace.closed
    with pytest.raises(RuntimeError):
        workspace.solve_many((support,))


def test_constructor_bounds_and_unknown_release_use_existing_ledger(monkeypatch):
    import tyx as tx

    from phymeo.backends.mujoco_tyx._resources import DeviceResources

    calls, buffers = [], []

    class Runtime:
        def __init__(self, *args, **kwargs):
            self.device = 0

        def close(self):
            calls.append("runtime-close")

    class Buffer:
        def __init__(self, shape, dtype):
            self.shape, self.host_dtype, self.dtype = shape, np.dtype(dtype), np.dtype(dtype)
            self.device, self.nbytes = 0, int(np.prod(shape)) * self.dtype.itemsize
            self.index = len(buffers)
            buffers.append(self)

        def free(self):
            calls.append(("free", self.index))
            if self.index == 0:
                raise RuntimeError("opaque-release-unknown")

    monkeypatch.setattr(tx, "Runtime", Runtime)
    monkeypatch.setattr(tx, "cuda_empty", lambda shape, dtype, **kwargs: Buffer(shape, dtype))
    monkeypatch.setattr(tx, "use_runtime", lambda runtime: nullcontext())
    for kwargs in ({"capacity": 0}, {"capacity": 33}, {"capacity": True}, {"mode": "interpret"}):
        with pytest.raises(ValueError):
            gpu.TyxCoupledFrictionWorkspace(**kwargs)
    assert not buffers and not calls
    workspace = gpu.TyxCoupledFrictionWorkspace(capacity=2)
    assert isinstance(workspace._resources, DeviceResources) and buffers
    assert gpu.TRACE_CAPACITY == 1 + 24 * 12 + 1 == 290
    assert workspace._buffers["trace"].shape == (2, 290, 15)
    assert workspace._buffers["trace_meta"].shape == (2, 290, 4)
    with pytest.raises(BaseExceptionGroup, match="UNKNOWN_RELEASE"):
        workspace.close()
    frozen_calls = tuple(calls)
    assert "runtime-close" not in calls
    assert workspace._resources.state == "UNKNOWN_RELEASE"
    assert workspace._resources.runtime_pending and not workspace.closed
    with pytest.raises(BaseExceptionGroup, match="UNKNOWN_RELEASE"):
        workspace.close()
    assert tuple(calls) == frozen_calls
    with pytest.raises(RuntimeError):
        workspace.solve_many((_cases()[-1][1],))
