"""Real tyx rigid integration checked against analytic and independent CPU dynamics."""

import numpy as np
import pytest

import phymeo
from phymeo import RigidBodyInputs, RigidBodyModel, RigidBodyState, RigidSemiImplicitSolver, Stepper

pytestmark = pytest.mark.tyx


def implementation():
    cls = getattr(phymeo, "TyxRigidSemiImplicitSolver", None)
    assert cls is not None, "TyxRigidSemiImplicitSolver is not implemented"
    return cls


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("rigid-tyx")


def scene(dtype=np.float32):
    model = RigidBodyModel(np.ones(1, dtype), np.array([[1, 2, 3]], dtype), np.zeros(3))
    state = RigidBodyState(
        np.zeros((1, 3), dtype),
        np.array([[0, 0, 0, 1]], dtype),
        np.zeros((1, 3), dtype),
        np.zeros((1, 3), dtype),
    )
    return model, state, RigidBodyInputs.zeros(1, dtype=dtype)


def test_translation_and_principal_torque_have_analytic_solution(tyx_mode, cache):
    _, state, inputs = scene()
    model = RigidBodyModel(
        np.array([0.5], np.float32), np.array([[1, 2, 3]], np.float32), np.array([0.0, -10.0, 0.0])
    )
    state.velocity[:] = [1, 2, 3]
    inputs.force[:] = [4, 0, 0]
    inputs.torque[:] = [0, 0, 6]
    with implementation()(mode=tyx_mode, cache_root=cache) as solver:
        assert solver.problem_signature() == RigidSemiImplicitSolver().problem_signature()
        assert solver.capabilities.state_kind == "rigid_bodies"
        assert solver.capabilities.dtypes == frozenset(["float32"])
        result = Stepper(solver).step(model, state, inputs, 0.5)
    assert result.report.accepted and result.report.backend == f"tyx:{tyx_mode}"
    np.testing.assert_allclose(result.state.velocity, [[2, -3, 3]], atol=1e-6)
    np.testing.assert_allclose(result.state.position, [[1, -1.5, 1.5]], atol=1e-6)
    np.testing.assert_allclose(result.state.angular_velocity, [[0, 0, 1]], atol=2e-7)
    np.testing.assert_allclose(
        result.state.orientation, [[0, 0, np.sin(0.25), np.cos(0.25)]], atol=2e-7
    )


def test_rotated_inertia_and_world_left_quaternion_composition(tyx_mode, cache):
    model, state, inputs = scene()
    s = np.sqrt(0.5)
    state.orientation[:] = [0, 0, s, s]
    inputs.torque[:] = [2, 0, 0]
    with implementation()(mode=tyx_mode, cache_root=cache) as solver:
        result = solver.step(model, state, inputs, 0.1)
    assert result.report.accepted
    np.testing.assert_allclose(result.state.angular_velocity, [[0.1, 0, 0]], atol=1e-7)
    np.testing.assert_allclose(
        result.state.orientation,
        [[np.sin(0.005) * s, -np.sin(0.005) * s, np.cos(0.005) * s, np.cos(0.005) * s]],
        atol=2e-7,
    )


def test_anisotropic_gyroscopic_acceleration_is_not_omitted(tyx_mode, cache):
    model, state, inputs = scene()
    state.angular_velocity[:] = [1, 2, 3]
    with implementation()(mode=tyx_mode, cache_root=cache) as solver:
        result = solver.step(model, state, inputs, 0.01)
    assert result.report.accepted
    np.testing.assert_allclose(
        result.state.angular_velocity, [[0.94, 2.03, 3 - 0.02 / 3]], atol=3e-7
    )


def test_random_batch_matches_fp64_intermediate_reference_and_preserves_fixed_states(
    tyx_mode, cache
):
    rng, count = np.random.default_rng(70421), 64
    weights = rng.uniform(0.2, 2, count).astype(np.float32)
    weights[[0, 17]] = 0
    axes = rng.uniform(0.5, 2, (count, 3)).astype(np.float32)
    inertia = np.column_stack(
        [axes[:, 0] + axes[:, 1], axes[:, 1] + axes[:, 2], axes[:, 2] + axes[:, 0]]
    )
    inertia[[0, 17]] = 0
    q = rng.normal(size=(count, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    model = RigidBodyModel(weights, inertia)
    state = RigidBodyState(
        rng.uniform(-2, 2, (count, 3)).astype(np.float32),
        q.astype(np.float32),
        rng.uniform(-1, 1, (count, 3)).astype(np.float32),
        rng.uniform(-1, 1, (count, 3)).astype(np.float32),
    )
    inputs = RigidBodyInputs(
        rng.uniform(-2, 2, (count, 3)).astype(np.float32),
        rng.uniform(-1, 1, (count, 3)).astype(np.float32),
    )
    before = state.copy()
    expected = RigidSemiImplicitSolver().step(model, state, inputs, 0.01)
    with implementation()(mode=tyx_mode, cache_root=cache) as solver:
        actual = solver.step(model, state, inputs, 0.01)
    assert actual.report.accepted
    for field in ("position", "velocity", "orientation", "angular_velocity"):
        np.testing.assert_allclose(
            getattr(actual.state, field), getattr(expected.state, field), atol=5e-7, rtol=2e-6
        )
        np.testing.assert_array_equal(getattr(state, field), getattr(before, field))
        assert not np.shares_memory(getattr(actual.state, field), getattr(state, field))
    np.testing.assert_array_equal(actual.state.orientation[[0, 17]], state.orientation[[0, 17]])
    np.testing.assert_array_equal(actual.state.angular_velocity[[0, 17]], 0)


def test_noncontiguous_views_are_canonicalized_before_real_execution(tyx_mode, cache):
    model, state, inputs = scene()
    for field in ("position", "velocity", "orientation", "angular_velocity"):
        value = getattr(state, field)
        backing = np.zeros((1, value.shape[1] * 2), np.float32)
        backing[:, ::2] = value
        setattr(state, field, backing[:, ::2])
    inputs.force = np.zeros((1, 6), np.float32)[:, ::2]
    inputs.torque = np.ones((1, 6), np.float32)[:, ::2]
    expected = RigidSemiImplicitSolver().step(model, state, inputs, 0.1)
    with implementation()(mode=tyx_mode, cache_root=cache) as solver:
        actual = solver.step(model, state, inputs, 0.1)
    assert actual.report.accepted
    np.testing.assert_allclose(actual.state.orientation, expected.state.orientation, atol=3e-7)
    np.testing.assert_allclose(
        actual.state.angular_velocity, expected.state.angular_velocity, atol=3e-7
    )


@pytest.mark.parametrize("failure", ["translation", "gyroscopic", "rotation_vector"])
def test_nonfinite_intermediate_rejects_original_owned_snapshot(tyx_mode, cache, failure):
    model, state, inputs = scene()
    dt = 10.0
    if failure == "translation":
        inputs.force[:] = 2e38
    elif failure == "gyroscopic":
        state.angular_velocity[:] = [1e30, 2e30, 3e30]
    else:
        model = RigidBodyModel(model.inverse_mass, np.ones((1, 3), np.float32), np.zeros(3))
        state.angular_velocity[:] = 2e38
    before = state.copy()
    with implementation()(mode=tyx_mode, cache_root=cache) as solver:
        actual = solver.step(model, state, inputs, dt)
    assert not actual.report.accepted and actual.report.reason
    for field in ("position", "velocity", "orientation", "angular_velocity"):
        np.testing.assert_array_equal(getattr(actual.state, field), getattr(before, field))
        assert not np.shares_memory(getattr(actual.state, field), getattr(state, field))


def test_numerical_fp32_limit_can_fallback_to_same_problem_math(tyx_mode, cache):
    model, state, inputs = scene()
    model = RigidBodyModel(model.inverse_mass, np.full((1, 3), 1e-30, np.float32), np.zeros(3))
    inputs.torque[0, 0] = 1e20
    dt = 1e-20
    expected = RigidSemiImplicitSolver().step(model, state, inputs, dt)
    assert expected.report.accepted
    with implementation()(mode=tyx_mode, cache_root=cache) as solver:
        actual = Stepper(solver, RigidSemiImplicitSolver()).step(model, state, inputs, dt)
    assert actual.report.accepted and actual.report.fallback_used
    assert actual.report.backend == "numpy" and actual.report.primary_failure
    np.testing.assert_array_equal(actual.state.orientation, expected.state.orientation)


def test_fp64_is_rejected_explicitly(tyx_mode, cache):
    with implementation()(mode=tyx_mode, cache_root=cache) as solver:
        with pytest.raises(ValueError, match="FP32"):
            solver.step(*scene(np.float64), 0.1)


def test_empty_and_closed_runtime_contract(tyx_mode, cache):
    model = RigidBodyModel(np.empty(0, np.float32), np.empty((0, 3), np.float32))
    state = RigidBodyState(
        np.empty((0, 3), np.float32),
        np.empty((0, 4), np.float32),
        np.empty((0, 3), np.float32),
        np.empty((0, 3), np.float32),
    )
    inputs = RigidBodyInputs.zeros(0, dtype=np.float32)
    solver = implementation()(mode=tyx_mode, cache_root=cache)
    result = solver.step(model, state, inputs, 0.1)
    assert result.report.accepted and result.state.body_count == 0
    solver.close()
    solver.close()
    with pytest.raises(RuntimeError, match="closed"):
        solver.step(model, state, inputs, 0.1)


def test_backend_errors_propagate_without_silent_fallback(tyx_mode, cache, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("injected compiler failure")

    with implementation()(mode=tyx_mode, cache_root=cache) as solver:
        monkeypatch.setattr(solver, "_kernel", broken)
        with pytest.raises(RuntimeError, match="injected compiler failure"):
            solver.step(*scene(), 0.1)


@pytest.mark.parametrize("source", ["torque", "force"])
def test_underflow_amplified_by_large_timestep_never_reports_false_zero_success(
    tyx_mode, cache, source
):
    model, state, inputs = scene()
    if source == "torque":
        model = RigidBodyModel(model.inverse_mass, np.full((1, 3), 1e30, np.float32), np.zeros(3))
        inputs.torque[0, 0] = 1e-30
    else:
        model = RigidBodyModel(
            np.array([1e-30], np.float32), np.ones((1, 3), np.float32), np.zeros(3)
        )
        inputs.force[0, 0] = 1e-30
    reference_model = RigidBodyModel(
        model.inverse_mass.astype(np.float64),
        model.inertia_diagonal.astype(np.float64),
        model.gravity.astype(np.float64),
    )
    reference_state = RigidBodyState(
        *(
            getattr(state, name).astype(np.float64)
            for name in ("position", "orientation", "velocity", "angular_velocity")
        )
    )
    reference_inputs = RigidBodyInputs(
        inputs.force.astype(np.float64), inputs.torque.astype(np.float64)
    )
    expected = RigidSemiImplicitSolver().step(
        reference_model, reference_state, reference_inputs, float(np.float32(1e30))
    )
    assert expected.report.accepted
    field = "orientation" if source == "torque" else "position"
    assert np.max(np.abs(getattr(expected.state, field) - getattr(state, field))) > 0.1
    with implementation()(mode=tyx_mode, cache_root=cache) as solver:
        actual = solver.step(model, state, inputs, 1e30)
    if actual.report.accepted:
        np.testing.assert_allclose(
            getattr(actual.state, field), getattr(expected.state, field), atol=2e-6, rtol=2e-6
        )
    else:
        assert actual.report.reason
        for name in ("position", "velocity", "orientation", "angular_velocity"):
            np.testing.assert_array_equal(getattr(actual.state, name), getattr(state, name))


def test_many_steps_preserve_rotation_and_match_reference(tyx_mode, cache):
    model, state, inputs = scene()
    state.angular_velocity[:] = [0.5, -0.25, 0.7]
    state.velocity[:] = [0.1, 0.3, -0.2]
    inputs.torque[:] = [0.02, 0.04, -0.01]
    inputs.force[:] = [0, -0.5, 0]
    expected = state.copy()
    with implementation()(mode=tyx_mode, cache_root=cache) as solver:
        for _ in range(100):
            result = solver.step(model, state, inputs, 0.002)
            reference = RigidSemiImplicitSolver().step(model, expected, inputs, 0.002)
            assert result.report.accepted and reference.report.accepted
            state, expected = result.state, reference.state
            for field in ("position", "velocity", "orientation", "angular_velocity"):
                np.testing.assert_allclose(
                    getattr(state, field), getattr(expected, field), atol=2e-5, rtol=2e-5
                )
            np.testing.assert_allclose(np.linalg.norm(state.orientation, axis=1), 1, atol=2e-7)


def test_constructor_import_failure_closes_its_created_runtime(tyx_mode, cache, monkeypatch):
    import builtins

    import tyx as tx

    original_import, original_runtime = builtins.__import__, tx.Runtime
    runtimes = []

    def track_runtime(*args, **kwargs):
        runtime = original_runtime(*args, **kwargs)
        runtimes.append(runtime)
        return runtime

    def fail_kernel(name, *args, **kwargs):
        if name.endswith("_tyx_rigid_kernels"):
            raise RuntimeError("injected rigid kernel import failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(tx, "Runtime", track_runtime)
    monkeypatch.setattr(builtins, "__import__", fail_kernel)
    with pytest.raises(RuntimeError, match="kernel import failure"):
        implementation()(mode=tyx_mode, cache_root=cache)
    assert runtimes and all(runtime.closed for runtime in runtimes)


@pytest.mark.parametrize("field", ["force", "orientation", "dt"])
def test_subnormal_input_precision_domain_is_explicit(tyx_mode, cache, field):
    model, state, inputs = scene()
    h = 0.01
    if field == "force":
        inputs.force[0, 0] = 1e-40
    elif field == "orientation":
        state.orientation[0, 0] = 1e-40
    else:
        h = 1e-40
    with implementation()(mode=tyx_mode, cache_root=cache) as solver:
        result = solver.step(model, state, inputs, h)
    assert not result.report.accepted and result.report.reason
    np.testing.assert_array_equal(result.state.orientation, state.orientation)


@pytest.mark.cuda
@pytest.mark.parametrize("failure", ["allocation", "kernel", "sync", "free"])
def test_cuda_partial_failures_release_all_allocated_arrays(tyx_mode, cache, monkeypatch, failure):
    if tyx_mode != "cuda":
        pytest.skip("requires actual CUDA allocation and release")
    model, state, inputs = scene()
    before = state.copy()
    allocated = []
    with implementation()(mode="cuda", cache_root=cache) as solver:
        original_array = solver._tx.cuda_array

        def allocate(value):
            if failure == "allocation" and len(allocated) == 2:
                raise RuntimeError("injected allocation failure")
            array = original_array(value)
            if failure == "free" and not allocated:
                original_free = type(array).free

                def free(value):
                    original_free(value)
                    if value is allocated[0]:
                        raise RuntimeError("injected free failure")

                monkeypatch.setattr(type(array), "free", free)
            allocated.append(array)
            return array

        monkeypatch.setattr(solver._tx, "cuda_array", allocate)
        if failure == "kernel":

            def broken(*args, **kwargs):
                raise RuntimeError("injected kernel failure")

            monkeypatch.setattr(solver, "_kernel", broken)
        if failure == "sync":
            original_sync = solver._runtime.sync
            count = 0

            def sync():
                nonlocal count
                count += 1
                original_sync()
                if count == 1:
                    raise RuntimeError("injected sync failure")

            monkeypatch.setattr(solver._runtime, "sync", sync)
        with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
            solver.step(model, state, inputs, 0.01)
        assert allocated
        for array in allocated:
            # The public array API exposes this diagnostic but not its error class.
            with pytest.raises(Exception, match="already been freed"):
                array.to_numpy()
    for field in ("position", "velocity", "orientation", "angular_velocity"):
        np.testing.assert_array_equal(getattr(state, field), getattr(before, field))
