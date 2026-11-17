"""Independent 128-point/two-moving-body CUDA friction qualification.

Manufactured loads are exact rational arithmetic, not outputs from either the
old experiment or this solver. No scene, native contact, or q integration is
claimed. The HOST release fault substitutes fake owners before allocation.
"""

import gzip
import json
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import fields, is_dataclass, replace
from fractions import Fraction as F
from time import perf_counter

import numpy as np
import pytest

import phymeo.backends.tyx_area_friction as gpu
from phymeo.solvers.area_friction import LAW, AreaFrictionProblem, AreaFrictionSample

H = F(1, 1000)
TARGET = (F(23, 4000), F(-1, 500), F(2, 5), F(1, 500), F(1, 1000), F(-1, 10))
MASS = (F(1, 2), F(1, 2), F(1, 5000), F(5, 4), F(5, 4), F(3, 10000))
DAMPING = (F(1), F(2), F(1, 1000), F(2), F(3), F(1, 500))
ROTATION = (
    (F(3, 5), F(-16, 25), F(-12, 25)),
    (F(4, 5), F(12, 25), F(9, 25)),
    (F(0), F(-3, 5), F(4, 5)),
)
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
    "contact_data",
    "response_matrix",
)
PROBLEM_ARRAYS = ("mass_matrix", "damping", "input_velocity", "applied_force")
SAMPLE_ARRAYS = ("point_world", "normal_world", "jacobian_a", "jacobian_b")
SAMPLE_SCALARS = ("stiffness_n_m", "phi_m", "mu", "slip_speed_m_s")


def _rotate(vector, rotation):
    return tuple(sum((a * b for a, b in zip(row, vector, strict=True)), F(0)) for row in rotation)


def _manufactured(count=128, *, rotated=False):
    """Fraction oracle for the two distinct moving planar rigid bodies.

    At v*: vt=3/4000, vs=4/4000, speed=5/4000, so Ft=-3/5 mu Fn.
    The B-side angular column contributes -0.01 Ft+x Fn to generalized load.
    Rotating all world vectors changes neither this six-coordinate load nor M.
    """
    assert count in (12, 128)
    identity = tuple(tuple(F(int(i == j)) for j in range(3)) for i in range(3))
    rotation = ROTATION if rotated else identity
    matrix = np.array(rotation, dtype=np.float64)
    generalized = [F(0)] * 6
    normal, tangent, forces, moments, samples = [], [], [], [], []
    rational_rows = []
    for i in range(count):
        x = F(2 * i - 127, 8000)
        stiffness, phi, mu = F(50 + 10 * (i % 4)), -F(40 + i % 5, 100000), F(1 + i % 3, 10)
        vn = -F(3, 1000) - x / 2
        fn = -stiffness * (phi + H * vn)
        ft = -F(3, 5) * mu * fn
        assert fn > 0
        load = (ft, fn, -ft / 100 - x * fn, -ft, -fn, -ft / 100 + x * fn)
        generalized = [a + b for a, b in zip(generalized, load, strict=True)]
        point = _rotate((x, F(0), F(0)), rotation)
        n = _rotate((F(0), F(0), F(1)), rotation)
        t = _rotate((ft, F(0), F(0)), rotation)
        force = _rotate((ft, F(0), fn), rotation)
        moment = _rotate((F(0), -x * fn, F(0)), rotation)
        ja = np.array(
            ((1, 0, -0.01, 0, 0, 0), (0, 0, 0, 0, 0, 0), (0, 1, -float(x), 0, 0, 0)),
            dtype=np.float64,
        )
        jb = np.array(
            ((0, 0, 0, 1, 0, 0.01), (0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, -float(x))),
            dtype=np.float64,
        )
        samples.append(
            AreaFrictionSample(
                tuple(map(float, point)),
                tuple(map(float, n)),
                matrix @ ja,
                matrix @ jb,
                float(stiffness),
                float(phi),
                float(mu),
                0.001,
            )
        )
        normal.append(float(fn))
        tangent.append(tuple(map(float, t)))
        forces.append(tuple(map(float, force)))
        moments.append(tuple(map(float, moment)))
        rational_rows.append(dict(x=str(x), vn=str(vn), normal=str(fn), tangent=str(ft)))
    external = tuple(
        (m + H * d) * v / H - q
        for m, d, v, q in zip(MASS, DAMPING, TARGET, generalized, strict=True)
    )
    problem = AreaFrictionProblem(
        np.diag(np.array(MASS, dtype=np.float64)),
        tuple(map(float, DAMPING)),
        (0.0,) * 6,
        tuple(map(float, external)),
        float(H),
        tuple(samples),
    )
    oracle = dict(
        velocity=np.array(TARGET, dtype=np.float64),
        normal=np.array(normal),
        tangent=np.array(tangent),
        force=np.array(forces),
        moment=np.array(moments),
        generalized=np.array(generalized, dtype=np.float64),
        rows=rational_rows,
        applied_force_fraction=tuple(map(str, external)),
        rotation=matrix,
    )
    return problem, oracle


def _empty(problem):
    """Remove contacts without changing the registered v0, external f or M/D."""
    empty = replace(problem, samples=())
    velocity = np.array(
        [
            float(H) * f / (float(m) + float(H * d))
            for f, m, d in zip(problem.applied_force, MASS, DAMPING, strict=True)
        ]
    )
    return empty, dict(
        velocity=velocity,
        normal=np.zeros(0),
        tangent=np.zeros((0, 3)),
        force=np.zeros((0, 3)),
        moment=np.zeros((0, 3)),
        generalized=np.zeros(6),
    )


def _equations(problem, velocity):
    """Independent full-J evaluation at actual v, with analytic cross terms."""
    v = np.asarray(velocity, dtype=np.float64)
    mass = np.asarray(problem.mass_matrix, dtype=np.float64)
    h = float(problem.h_s)
    response = mass + h * np.diag(np.asarray(problem.damping, dtype=np.float64))
    free = np.linalg.solve(
        response,
        mass @ np.asarray(problem.input_velocity, dtype=np.float64)
        + h * np.asarray(problem.applied_force, dtype=np.float64),
    )
    total, jacobian = np.zeros(len(v)), response.copy()
    normal, tangent, force, moment, relative_tangent = [], [], [], [], []
    for sample in problem.samples:
        n = np.asarray(sample.normal_world, dtype=np.float64)
        g = np.asarray(sample.jacobian_a, dtype=np.float64) - np.asarray(
            sample.jacobian_b, dtype=np.float64
        )
        relative = g @ v
        vn = float(n @ relative)
        projection = np.eye(3) - np.outer(n, n) / (n @ n)
        vt = projection @ relative
        stiffness, mu = float(sample.stiffness_n_m), float(sample.mu)
        fn = max(0.0, -stiffness * (float(sample.phi_m) + h * vn))
        speed = np.sqrt(vt @ vt + float(sample.slip_speed_m_s) ** 2)
        ft = -mu * fn * vt / speed
        f = fn * n + ft
        dfn = -stiffness * h * (n @ g) if fn > 0 else np.zeros(len(v))
        dslip = (np.eye(3) / speed - np.outer(vt, vt) / speed**3) @ projection @ g
        df = np.outer(n - mu * vt / speed, dfn) - mu * fn * dslip
        total += g.T @ f
        jacobian -= h * g.T @ df
        normal.append(fn)
        tangent.append(ft)
        force.append(f)
        moment.append(np.cross(np.asarray(sample.point_world, dtype=np.float64), f))
        relative_tangent.append(vt)
    residual = response @ (v - free) - h * total
    norm = np.sqrt(max(0.0, float(residual @ np.linalg.solve(response, residual))))
    scale = 1 + np.sqrt(max(0.0, float(free @ response @ free)))
    ns = len(problem.samples)
    return dict(
        free=free,
        response=response,
        residual=residual,
        residual_norm=norm,
        scale=scale,
        jacobian=jacobian,
        generalized=total,
        normal=np.array(normal),
        tangent=np.array(tangent).reshape(ns, 3),
        force=np.array(force).reshape(ns, 3),
        moment=np.array(moment).reshape(ns, 3),
        relative_tangent=np.array(relative_tangent).reshape(ns, 3),
    )


def _plain(value):
    if isinstance(value, np.ndarray):
        # Rejected device payloads may include nonfinite words: preserve them.
        return dict(dtype=value.dtype.str, shape=list(value.shape), hex=value.tobytes().hex())
    if isinstance(value, np.generic):
        return _plain(np.asarray(value))
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, int, bool, float)):
        return value
    raise TypeError(type(value).__name__)


def _save(path, value):
    with gzip.open(path, "xt", encoding="utf-8") as stream:
        json.dump(_plain(value), stream, ensure_ascii=True, allow_nan=False)


def _solve_saved(workspace, problems, path, monkeypatch, **kwargs):
    """Observe the existing downloads, never add another kernel or readback."""
    get, raw, begin = workspace._download, {}, perf_counter()
    evidence = dict(originals=problems, kwargs=kwargs, stats_before=workspace.stats, raw=raw)

    def capture(name):
        value = get(name)
        raw[name] = value.copy()
        return value

    try:
        with monkeypatch.context() as patch:
            patch.setattr(workspace, "_download", capture)
            result = workspace.solve_many(problems, **kwargs)
        evidence["results"] = result
        return result
    except BaseException as error:
        evidence["error"] = dict(type=type(error).__name__, message=str(error))
        if hasattr(error, "diagnostics"):
            evidence["diagnostics"] = error.diagnostics
        raise
    finally:
        evidence.update(stats_after=workspace.stats, elapsed_s=perf_counter() - begin)
        _save(path, evidence)


def _readonly(array):
    assert not array.flags.writeable
    with pytest.raises(ValueError):
        array.setflags(write=True)


def _packed(result, problem):
    assert _plain(result.original_problem) == _plain(problem)
    for name in PROBLEM_ARRAYS:
        value = getattr(result.packed_problem, name)
        expected = np.asarray(getattr(problem, name), dtype=np.float32)
        assert value.dtype == np.dtype(np.float32) and value.tobytes() == expected.tobytes()
        _readonly(value)
    assert float(result.packed_problem.h_s) == float(np.float32(problem.h_s))
    assert len(result.packed_problem.samples) == len(problem.samples)
    for packed, source in zip(result.packed_problem.samples, problem.samples, strict=True):
        for name in SAMPLE_ARRAYS:
            value = getattr(packed, name)
            assert value.dtype == np.dtype(np.float32)
            assert value.tobytes() == np.asarray(getattr(source, name), dtype=np.float32).tobytes()
            _readonly(value)
        for name in SAMPLE_SCALARS:
            assert float(getattr(packed, name)) == float(np.float32(getattr(source, name)))


def _check(result, problem, oracle):
    assert type(result) is gpu.TyxAreaFrictionResult and result.law == LAW
    _packed(result, problem)
    actual = _equations(result.packed_problem, result.velocity)
    ns = len(problem.samples)
    for name in RESULT_ARRAYS:
        value = getattr(result, name)
        assert value.dtype == np.dtype(np.float32) and np.isfinite(value).all()
        _readonly(value)
    assert result.normal_force.shape == (ns,)
    assert result.tangential_force.shape == (ns, 3) and result.contact_data.shape == (ns, 17)
    reference = oracle["velocity"]
    assert np.all(np.abs(result.velocity - reference) <= 5e-5 * (1 + np.abs(reference)))
    for got, expected in (
        (result.normal_force, oracle["normal"]),
        (result.tangential_force, oracle["tangent"]),
        (result.force_on_a, oracle["force"]),
        (result.force_on_b, -oracle["force"]),
        (result.normal_force, actual["normal"]),
        (result.tangential_force, actual["tangent"]),
        (result.generalized_force, actual["generalized"]),
    ):
        assert np.all(np.abs(got - expected) <= 0.002 + 0.002 * np.abs(expected))
    for got, expected in (
        (result.torque_on_a_world_origin, oracle["moment"]),
        (result.torque_on_b_world_origin, -oracle["moment"]),
        (result.torque_on_a_world_origin, actual["moment"]),
    ):
        assert np.all(np.abs(got - expected) <= 1e-5 + 0.002 * np.abs(expected))
    assert actual["residual_norm"] <= 2e-5 * actual["scale"]
    np.testing.assert_allclose(result.residual_jacobian, actual["jacobian"], rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(result.free_velocity, actual["free"], rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(result.response_matrix, actual["response"], rtol=2e-6, atol=1e-9)
    mu = np.array([sample.mu for sample in result.packed_problem.samples])
    assert np.all(result.normal_force >= 0)
    assert np.all(
        np.linalg.norm(result.tangential_force.astype(np.float64), axis=1)
        <= mu * result.normal_force + 1e-5
    )
    np.testing.assert_array_equal(result.force_on_b, -result.force_on_a)
    np.testing.assert_array_equal(result.torque_on_b_world_origin, -result.torque_on_a_world_origin)
    np.testing.assert_array_equal(result.contact_data[:, 0], result.normal_force)
    np.testing.assert_array_equal(result.contact_data[:, 6:9], result.tangential_force)
    np.testing.assert_array_equal(result.contact_data[:, 9:12], result.force_on_a)
    np.testing.assert_array_equal(result.contact_data[:, 12:15], result.torque_on_a_world_origin)
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
    assert (
        type(result.trace) is tuple and len(result.trace) == diagnostics["line_search_trials"] + 2
    )
    assert 2 <= len(result.trace) <= 290
    assert result.trace[0].phase == "initial" and result.trace[-1].phase == "final"
    np.testing.assert_array_equal(result.trace[-1].velocity, result.velocity)
    for row in result.trace:
        _readonly(row.velocity)
        _readonly(row.residual)
    with pytest.raises(TypeError):
        diagnostics["iterations"] = -1
    with pytest.raises((AttributeError, TypeError)):
        result.law = "other"
    return dict(
        velocity_error=np.abs(result.velocity - reference),
        actual_input_residual=actual["residual_norm"],
        actual_input_limit=2e-5 * actual["scale"],
        original_input_residual=_equations(problem, result.velocity)["residual_norm"],
        nonsymmetric_jacobian=float(np.max(np.abs(actual["jacobian"] - actual["jacobian"].T))),
        jacobian_max_error=float(np.max(np.abs(result.residual_jacobian - actual["jacobian"]))),
        analytic=oracle,
        actual_equations=actual,
    )


def test_owned_schema_and_independent_rational_manufacture(tmp_path):
    assert LAW == "final_normal_smooth_v1"
    full, expected = _manufactured()
    holdout, oblique = _manufactured(12, rotated=True)
    assert len(full.samples) == 128 and len({tuple(s.point_world) for s in full.samples}) == 128
    assert len(holdout.samples) == 12
    for problem, oracle in ((full, expected), (holdout, oblique)):
        assert problem.law == LAW and all(s.law == LAW for s in problem.samples)
        equations = _equations(problem, oracle["velocity"])
        np.testing.assert_allclose(equations["normal"], oracle["normal"], rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(equations["tangent"], oracle["tangent"], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(
            equations["generalized"], oracle["generalized"], rtol=1e-12, atol=1e-13
        )
        np.testing.assert_allclose(equations["moment"], oracle["moment"], rtol=1e-12, atol=1e-14)
        assert equations["residual_norm"] < 1e-12
        assert np.max(np.abs(equations["jacobian"] - equations["jacobian"].T)) > 1e-6
        assert all(np.any(s.jacobian_b) for s in problem.samples)
        for name in PROBLEM_ARRAYS:
            _readonly(getattr(problem, name))
        for name in SAMPLE_ARRAYS:
            _readonly(getattr(problem.samples[0], name))
    assert all(
        sum((ROTATION[k][i] * ROTATION[k][j] for k in range(3)), F(0)) == F(int(i == j))
        for i in range(3)
        for j in range(3)
    )
    assert np.all(oblique["tangent"][:, :2] != 0) and np.all(holdout.samples[0].normal_world != 0)
    source_mass, source_force = full.mass_matrix.copy(), full.applied_force.copy()
    owned = replace(full, mass_matrix=source_mass, applied_force=source_force)
    saved = _plain(owned)
    source_mass[:] = 0
    source_force[:] = 123
    assert _plain(owned) == saved
    with pytest.raises((AttributeError, TypeError)):
        owned.law = "none"
    invalid = (
        dict(samples=full.samples + (full.samples[0],)),
        dict(
            mass_matrix=np.eye(7),
            damping=np.zeros(7),
            input_velocity=np.zeros(7),
            applied_force=np.zeros(7),
            samples=(),
        ),
        dict(mass_matrix=-np.eye(6)),
        dict(mass_matrix=np.eye(6, dtype=complex)),
        dict(damping=np.ones(6, dtype=bool)),
        dict(h_s=True),
        dict(h_s=float("nan")),
    )
    for changes in invalid:
        with pytest.raises(ValueError):
            replace(full, **changes)
    for changes in (
        dict(stiffness_n_m=0),
        dict(phi_m=0.001),
        dict(mu=-1),
        dict(slip_speed_m_s=0),
        dict(normal_world=(0, 0, 2)),
        dict(point_world=np.array((0, 0, 0), dtype=object)),
    ):
        with pytest.raises(ValueError):
            replace(full.samples[0], **changes)
    _save(
        tmp_path / "rational-manufacture.json.gz",
        dict(
            full=full,
            full_oracle=expected,
            holdout=holdout,
            holdout_oracle=oblique,
            qualification="independent analytic inputs; no solve",
        ),
    )


@pytest.mark.cuda
def test_full_128_holdout_12_mixed_32_and_reverse_all_raw(tmp_path, monkeypatch):
    full, expected = _manufactured()
    holdout, oblique = _manufactured(12, rotated=True)
    empty, empty_expected = _empty(full)
    checks = {}
    with gpu.TyxAreaFrictionWorkspace(capacity=32, mode="cuda") as workspace:
        assert workspace.law == LAW
        assert workspace.stats["live_bytes"] == workspace.stats["peak_bytes"] == 1770624
        first_stats = workspace.stats
        (one,) = _solve_saved(workspace, (full,), tmp_path / "w1-128.json.gz", monkeypatch)
        assert workspace.stats["uploaded_bytes"] - first_stats["uploaded_bytes"] == 761216
        assert workspace.stats["downloaded_bytes"] - first_stats["downloaded_bytes"] == 1009408
        checks["w1-128"] = _check(one, full, expected)
        (two,) = _solve_saved(
            workspace, (holdout,), tmp_path / "w1-12-oblique.json.gz", monkeypatch
        )
        checks["w1-12"] = _check(two, holdout, oblique)
        old = _plain((one, two))
        variants = ((full, expected, one), (holdout, oblique, two), (empty, empty_expected, None))
        rows = [variants[i % 3] for i in range(32)]
        rows[-1] = variants[0]
        result = _solve_saved(
            workspace, tuple(v[0] for v in rows), tmp_path / "w32-mixed.json.gz", monkeypatch
        )
        assert type(result) is tuple and len(result) == 32 and workspace.last_results is result
        assert len(result[-1].normal_force) == 128
        checks["w32"] = []
        for item, (problem, reference, single) in zip(result, rows, strict=True):
            checks["w32"].append(_check(item, problem, reference))
            if single is not None:
                assert np.all(
                    np.abs(item.velocity - single.velocity) <= 5e-5 * (1 + np.abs(single.velocity))
                )
                for name in ("normal_force", "tangential_force", "force_on_a", "generalized_force"):
                    actual, previous = getattr(item, name), getattr(single, name)
                    assert np.all(np.abs(actual - previous) <= 0.002 + 0.002 * np.abs(previous))
                for name in ("torque_on_a_world_origin", "torque_on_b_world_origin"):
                    actual, previous = getattr(item, name), getattr(single, name)
                    assert np.all(np.abs(actual - previous) <= 1e-5 + 0.002 * np.abs(previous))
        reversed_problem = replace(full, samples=tuple(reversed(full.samples)))
        reversed_oracle = {
            **expected,
            **{name: expected[name][::-1] for name in ("normal", "tangent", "force", "moment")},
        }
        (reverse,) = _solve_saved(
            workspace, (reversed_problem,), tmp_path / "w1-128-reverse.json.gz", monkeypatch
        )
        checks["reverse"] = _check(reverse, reversed_problem, reversed_oracle)
        assert np.all(np.abs(reverse.velocity - one.velocity) <= 5e-5 * (1 + np.abs(one.velocity)))
        for field, tolerance in (("force_on_a", 0.002), ("torque_on_a_world_origin", 1e-5)):
            actual = np.asarray(getattr(reverse, field), dtype=np.float64).sum(axis=0)
            previous = np.asarray(getattr(one, field), dtype=np.float64).sum(axis=0)
            assert np.all(np.abs(actual - previous) <= tolerance + 0.002 * np.abs(previous))
        assert workspace.stats["launches"] == workspace.stats["batch_calls"] == 4
        assert workspace.stats["uploaded_bytes"] == 4 * 761216
        assert workspace.stats["downloaded_bytes"] == 4 * 1009408
        assert _plain((one, two)) == old
        _save(
            tmp_path / "independent-checks.json.gz",
            dict(
                checks=checks,
                stats=workspace.stats,
                world_sample_counts=[len(p.samples) for p, _, _ in rows],
                law=LAW,
            ),
        )
    assert workspace.closed
    with pytest.raises(RuntimeError):
        workspace.solve_many((empty,))


@pytest.mark.cuda
def test_preflight_late_budget_retry_and_download_atomicity(tmp_path, monkeypatch):
    problem, expected = _manufactured()
    empty, empty_expected = _empty(problem)
    with gpu.TyxAreaFrictionWorkspace(capacity=32) as workspace:
        old = _solve_saved(workspace, (empty,), tmp_path / "baseline-empty.json.gz", monkeypatch)
        _check(old[0], empty, empty_expected)
        previous, before = _plain(old), workspace.stats
        bad = replace(problem)
        object.__setattr__(bad, "samples", problem.samples + (problem.samples[0],))
        bad_dtype = replace(problem)
        object.__setattr__(bad_dtype, "mass_matrix", np.eye(6, dtype=complex))
        underflow = replace(problem, samples=(replace(problem.samples[0], stiffness_n_m=1e-50),))
        lost_mass = np.eye(6)
        lost_mass[:2, :2] = ((1, 1), (1, 1 + 1e-8))
        lost_spd = replace(problem, mass_matrix=lost_mass)
        for invalid in (
            [problem],
            (),
            (problem, None),
            (problem, bad),
            (problem, bad_dtype),
            (problem, underflow),
            (problem, lost_spd),
            (problem,) * 33,
        ):
            with pytest.raises((TypeError, ValueError)):
                workspace.solve_many(invalid)
            assert workspace.stats["launches"] == before["launches"]
            assert workspace.stats["uploaded_bytes"] == before["uploaded_bytes"]
            assert workspace.last_results is old and _plain(old) == previous
        with pytest.raises(gpu.TyxAreaFrictionFailure) as failure:
            _solve_saved(
                workspace,
                (empty, problem),
                tmp_path / "late-world-budget0.json.gz",
                monkeypatch,
                max_iterations=0,
            )
        assert failure.value.diagnostics["meta"][0, 0] == 0
        assert failure.value.diagnostics["meta"][1, 0] != 0
        assert workspace.stats["launches"] == before["launches"] + 1
        assert workspace.last_results is old and _plain(old) == previous and not workspace.faulted
        (retry,) = _solve_saved(
            workspace, (problem,), tmp_path / "retried-128.json.gz", monkeypatch
        )
        _check(retry, problem, expected)
        published, saved = workspace.last_results, _plain(workspace.last_results)
        get, calls = workspace._download, []

        def late_download(name):
            calls.append(name)
            if len(calls) == 2:
                raise RuntimeError("i3u-late-download")
            return get(name)

        with monkeypatch.context() as patch:
            patch.setattr(workspace, "_download", late_download)
            with pytest.raises(RuntimeError, match="i3u-late-download"):
                _solve_saved(workspace, (empty, problem), tmp_path / "late-download.json.gz", patch)
        assert len(calls) == 2 and workspace.faulted
        assert workspace.last_results is published and _plain(published) == saved
        launches = workspace.stats["launches"]
        with pytest.raises(RuntimeError):
            workspace.solve_many((empty,))
        assert workspace.stats["launches"] == launches
        _save(
            tmp_path / "fault-publication.json.gz",
            dict(last_results=published, calls=calls, stats=workspace.stats),
        )
    assert workspace.closed


def test_fixed_buffer_budget_and_unknown_release_owned_dependencies(monkeypatch):
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
            gpu.TyxAreaFrictionWorkspace(**kwargs)
    assert not buffers and not calls
    workspace = gpu.TyxAreaFrictionWorkspace(capacity=32)
    assert workspace.law == LAW and workspace.capacity == 32
    assert isinstance(workspace._resources, DeviceResources)
    assert len(buffers) == 14 and sum(b.nbytes for b in buffers) == 1770624
    assert workspace._buffers["samples"].shape == (32, 128, 10)
    assert workspace._buffers["ja"].shape == workspace._buffers["jb"].shape == (32, 128, 3, 6)
    assert workspace._buffers["contact"].shape == (32, 128, 17)
    assert workspace._buffers["vectors"].shape == (32, 8, 6)
    assert gpu.TRACE_CAPACITY == 290
    assert workspace._buffers["trace"].shape == (32, 290, 15)
    assert workspace._buffers["trace_meta"].shape == (32, 290, 4)
    assert workspace._buffers["trace"].nbytes + workspace._buffers["trace_meta"].nbytes == 705280
    assert workspace.stats["live_bytes"] == workspace.stats["peak_bytes"] == 1770624
    with pytest.raises(BaseExceptionGroup, match="UNKNOWN_RELEASE"):
        workspace.close()
    original = tuple(calls)
    assert "runtime-close" not in calls
    assert workspace._resources.state == "UNKNOWN_RELEASE" and workspace._resources.runtime_pending
    assert not workspace.closed
    with pytest.raises(BaseExceptionGroup, match="UNKNOWN_RELEASE"):
        workspace.close()
    assert tuple(calls) == original
    with pytest.raises(RuntimeError):
        workspace.solve_many((_empty(_manufactured()[0])[0],))
