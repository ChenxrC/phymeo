"""Real device contact primitives, followed by joint-driver tests as M6 progresses."""

import importlib
from contextlib import ExitStack

import numpy as np
import pytest

pytestmark = pytest.mark.tyx


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("coupled-tyx")


def execute(name, scalars, arrays, mode, cache, *, repeats=1):
    kernel = getattr(importlib.import_module("phymeo.backends._tyx_coupled_kernels"), name)
    import tyx as tx

    with ExitStack() as resources:
        runtime = tx.Runtime(backend="cuda" if mode == "cuda" else "cpu", cache_root=cache)
        resources.callback(runtime.close)
        resources.enter_context(tx.use_runtime(runtime))
        values = []
        for value in arrays:
            if mode == "cuda":
                value = tx.cuda_array(value)
                resources.callback(value.free)
            else:
                value = np.array(value, order="C", copy=True)
            values.append(value)
        resources.callback(runtime.sync)
        with np.errstate(all="ignore"):
            for _ in range(repeats):
                kernel(*scalars, *values, mode=mode)
            runtime.sync()
        return [value.to_numpy() if mode == "cuda" else value.copy() for value in values]


def position_arrays(particle=(0.8, 0, 0), body=(0, 0, 0), wp=1.0, wb=0.5):
    return [
        np.array([particle], np.float32),
        np.array([body], np.float32),
        np.array([wp], np.float32),
        np.array([wb], np.float32),
        np.ones(1, np.float32),
        np.zeros(1, np.float32),
        np.zeros((1, 3), np.float32),
        np.zeros(1, np.int32),
    ]


@pytest.mark.parametrize("moving", ["particle", "rigid"])
def test_normal_correction_preserves_large_origin_tangent_on_device(tyx_mode, cache, moving):
    import phymeo as api

    dtype = np.float32
    gravity = np.zeros(3, dtype)
    pm = api.ParticleModel(np.array([moving == "particle"], dtype), gravity)
    bm = api.RigidBodyModel(np.array([moving == "rigid"], dtype), np.ones((1, 3), dtype), gravity)
    ps = api.ParticleState(np.array([[1e8, 0, 0]], dtype), np.zeros((1, 3), dtype))
    bs = api.RigidBodyState(
        np.array([[1e8, 0, 0]], dtype),
        np.array([[0, 0, 0, 1]], dtype),
        np.zeros((1, 3), dtype),
        np.zeros((1, 3), dtype),
    )
    mobile = ps if moving == "particle" else bs
    mobile.position[0, 1] = 1
    mobile.velocity[:] = [1, -1, 0]
    state = api.CoupledState(ps, bs)
    with api.TyxCoupledXPBDSolver(
        pm,
        bm,
        None,
        api.SphereShapes(np.ones(1, dtype)),
        substeps=1,
        mode=tyx_mode,
        cache_root=cache,
    ) as solver:
        result = solver.step(
            state,
            api.CoupledInputs(
                api.StepInputs.zeros(1, dtype=dtype), api.RigidBodyInputs.zeros(1, dtype=dtype)
            ),
            0.1,
        )
    assert result.report.accepted
    after = result.state.particles if moving == "particle" else result.state.rigid
    np.testing.assert_allclose(after.velocity, [[1, 0, 0]], atol=1e-6, rtol=0)
    np.testing.assert_array_equal(mobile.velocity, [[1, -1, 0]])


def test_device_position_projection_splits_response_by_mass(tyx_mode, cache):
    values = execute("project_contacts", [1, 1], position_arrays(), tyx_mode, cache)
    p, b, _, _, _, multipliers, loads, failed = values
    np.testing.assert_array_equal(failed, 0)
    np.testing.assert_allclose(p, [[14 / 15, 0, 0]], atol=2e-7)
    np.testing.assert_allclose(b, [[-1 / 15, 0, 0]], atol=2e-7)
    np.testing.assert_allclose(multipliers, [2 / 15], atol=2e-7)
    np.testing.assert_allclose(loads, [[2 / 15, 0, 0]], atol=2e-7)
    np.testing.assert_allclose(p + 2 * b, [[0.8, 0, 0]], atol=2e-7)


def test_multiple_spheres_use_stable_particle_major_pairs(tyx_mode, cache):
    data = [
        np.array([[0.8, 0, 0], [10.8, 0, 0]], np.float32),
        np.array([[0, 0, 0], [10, 0, 0]], np.float32),
        np.ones(2, np.float32),
        np.array([0.5, 2], np.float32),
        np.ones(2, np.float32),
        np.zeros(4, np.float32),
        np.zeros((4, 3), np.float32),
        np.zeros(4, np.int32),
    ]
    p, b, _, _, _, multipliers, loads, failed = execute(
        "project_contacts", [2, 2], data, tyx_mode, cache
    )
    np.testing.assert_allclose(p[:, 0], [14 / 15, 10 + 13 / 15], atol=1e-6)
    np.testing.assert_allclose(b[:, 0], [-1 / 15, 10 - 2 / 15], atol=1e-6)
    np.testing.assert_allclose(multipliers, [2 / 15, 0, 0, 1 / 15], atol=1e-6)
    np.testing.assert_array_equal(loads[[1, 2]], 0)
    np.testing.assert_array_equal(failed, 0)


def test_device_position_multiplier_can_release_without_attractive_final_force(tyx_mode, cache):
    values = execute("project_contacts", [1, 1], position_arrays(), tyx_mode, cache)
    values[0][0, 0] += 1
    released = execute("project_contacts", [1, 1], values, tyx_mode, cache)
    np.testing.assert_array_equal(released[5], 0)
    np.testing.assert_array_equal(released[-1], 0)
    assert released[6][0, 0] < 0  # Unwinds the previously accumulated multiplier.
    np.testing.assert_allclose(released[0], [[1.8, 0, 0]], atol=3e-7)
    np.testing.assert_allclose(released[1], 0, atol=3e-7)


@pytest.mark.parametrize("fixed", [False, True])
def test_center_coincidence_and_fixed_pairs_are_deterministic(tyx_mode, cache, fixed):
    initial = position_arrays(particle=(0, 0, 0), wp=0 if fixed else 1, wb=0 if fixed else 0.5)
    a = execute("project_contacts", [1, 1], initial, tyx_mode, cache)
    b = execute("project_contacts", [1, 1], initial, tyx_mode, cache)
    for left, right in zip(a, b, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(a[-1], 0)
    if fixed:
        np.testing.assert_array_equal(a[0], 0)
        np.testing.assert_array_equal(a[1], 0)
        np.testing.assert_array_equal(a[5], 0)
    else:
        np.testing.assert_allclose(a[0] - a[1], [[1, 0, 0]], atol=2e-7)


def test_final_geometry_is_recomputed_after_shared_body_moves(tyx_mode, cache):
    p = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    b = np.array([[0.1, 0, 0]], np.float32)
    data = [
        p,
        b,
        np.ones(1, np.float32),
        np.empty((2, 3), np.float32),
        np.empty(2, np.float32),
        np.zeros(2, np.int32),
    ]
    _, _, _, normals, gaps, failed = execute("contact_geometry", [2, 1], data, tyx_mode, cache)
    delta = p.astype(np.float64) - b.astype(np.float64)
    lengths = np.linalg.norm(delta, axis=1)
    np.testing.assert_allclose(normals, delta / lengths[:, None], atol=2e-7)
    np.testing.assert_allclose(gaps, lengths - 1, atol=2e-7)
    np.testing.assert_array_equal(failed, 0)


def test_shared_body_velocity_sweeps_release_old_impulses_and_conserve_momentum(tyx_mode, cache):
    arrays = [
        np.array([[-1, 0, 0], [-4, 0, 0]], np.float32),
        np.zeros((1, 3), np.float32),
        np.ones(2, np.float32),
        np.ones(1, np.float32),
        np.array([[1, 0, 0], [1, 0, 0]], np.float32),
        np.zeros(2, np.float32),
        np.zeros(2, np.float32),
        np.zeros((2, 3), np.float32),
        np.zeros(2, np.int32),
    ]
    p, b, _, _, _, _, impulses, _, failed = execute(
        "project_contact_velocities", [2, 1, np.float32(1e-6)], arrays, tyx_mode, cache, repeats=5
    )
    np.testing.assert_allclose(p[:, 0], [-1, -2], atol=2e-6)
    np.testing.assert_allclose(b[:, 0], [-2], atol=2e-6)
    np.testing.assert_allclose(impulses, [0, 2], atol=2e-6)
    np.testing.assert_allclose(p.sum(axis=0) + b.sum(axis=0), [-5, 0, 0], atol=2e-6)
    np.testing.assert_array_equal(failed, 0)


def test_device_overflow_is_flagged_instead_of_hidden_by_projection(tyx_mode, cache):
    arrays = position_arrays(particle=(3e38, 0, 0), body=(-3e38, 0, 0))
    values = execute("project_contacts", [1, 1], arrays, tyx_mode, cache)
    assert np.any(values[-1])


@pytest.mark.parametrize("stage", ["position", "velocity", "position_delta", "geometry_delta"])
def test_checked_arithmetic_failure_cannot_be_hidden_by_max_or_clamping(tyx_mode, cache, stage):
    if stage == "velocity":
        arrays = [
            np.array([[-1e-20, 0, 0]], np.float32),
            np.zeros((1, 3), np.float32),
            np.array([1e20], np.float32),
            np.array([1e20], np.float32),
            np.array([[1, 0, 0]], np.float32),
            np.zeros(1, np.float32),
            np.zeros(1, np.float32),
            np.zeros((1, 3), np.float32),
            np.zeros(1, np.int32),
        ]
        values = execute(
            "project_contact_velocities", [1, 1, np.float32(1e-6)], arrays, tyx_mode, cache
        )
    else:
        arrays = position_arrays()
        if stage == "position":
            arrays[0][0] = [0.5e-20, 0, 0]
            arrays[2][:] = arrays[3][:] = 1e20
            arrays[4][:] = 1e-20
        else:
            arrays[0][0] = [np.finfo(np.float32).tiny, 0, 0]
            arrays[1][0] = [
                np.nextafter(np.float32(np.finfo(np.float32).tiny), np.float32(np.inf)),
                0,
                0,
            ]
        if stage == "geometry_delta":
            values = execute(
                "contact_geometry",
                [1, 1],
                [
                    arrays[0],
                    arrays[1],
                    arrays[4],
                    np.empty((1, 3), np.float32),
                    np.empty(1, np.float32),
                    np.zeros(1, np.int32),
                ],
                tyx_mode,
                cache,
            )
        else:
            values = execute("project_contacts", [1, 1], arrays, tyx_mode, cache)
    assert np.any(values[-1])


def test_reconstruction_preserves_free_velocity_and_updates_only_corrected_rows(tyx_mode, cache):
    arrays = [
        np.ones(2, np.float32),
        np.array([[1e20, 0, 0], [0, 0, 0]], np.float32),
        np.array([[1e20, 0, 0], [0.1, 0, 0]], np.float32),
        np.array([[1e20, 0, 0], [0.2, 0, 0]], np.float32),
        np.array([[1, 0, 0], [1, 0, 0]], np.float32),
        np.zeros(2, np.int32),
    ]
    values = execute("reconstruct_velocity", [2, np.float32(0.1)], arrays, tyx_mode, cache)
    np.testing.assert_allclose(values[-2], [[1, 0, 0], [2, 0, 0]], atol=2e-7)
    np.testing.assert_array_equal(values[-1], 0)


def driver():
    import phymeo

    cls = getattr(phymeo, "TyxCoupledXPBDSolver", None)
    assert cls is not None, "joint tyx driver is not implemented"
    return cls


def problem(p_count=1, b_count=1, dtype=np.float32):
    from phymeo import (
        CoupledInputs,
        CoupledState,
        ParticleModel,
        ParticleState,
        RigidBodyModel,
        RigidBodyState,
        SphereShapes,
    )

    pm = ParticleModel(np.ones(p_count, dtype), np.zeros(3))
    bm = RigidBodyModel(np.ones(b_count, dtype), np.ones((b_count, 3), dtype), np.zeros(3))
    state = CoupledState(
        ParticleState(np.full((p_count, 3), 3, dtype), np.zeros((p_count, 3), dtype)),
        RigidBodyState(
            np.zeros((b_count, 3), dtype),
            np.tile(np.array([0, 0, 0, 1], dtype), (b_count, 1)),
            np.zeros((b_count, 3), dtype),
            np.zeros((b_count, 3), dtype),
        ),
    )
    return (
        pm,
        bm,
        SphereShapes(np.ones(b_count, dtype)),
        state,
        CoupledInputs.zeros(p_count, b_count, dtype=dtype),
    )


def fields(state):
    return (
        state.particles.position,
        state.particles.velocity,
        state.rigid.position,
        state.rigid.velocity,
        state.rigid.orientation,
        state.rigid.angular_velocity,
    )


def test_joint_driver_matches_single_pair_mass_and_momentum(tyx_mode, cache):
    from phymeo import RigidBodyModel

    pm, bm, spheres, state, inputs = problem()
    bm = RigidBodyModel(np.array([0.5], np.float32), bm.inertia_diagonal, bm.gravity)
    state.particles.position[:] = [1, 0, 0]
    state.particles.velocity[:] = [-2, 0, 0]
    with driver()(pm, bm, None, spheres, substeps=1, mode=tyx_mode, cache_root=cache) as solver:
        result = solver.step(state, inputs, 0.1)
    assert result.report.accepted and result.report.backend == f"tyx:{tyx_mode}"
    np.testing.assert_allclose(result.state.particles.position, [[14 / 15, 0, 0]], atol=2e-6)
    np.testing.assert_allclose(result.state.rigid.position, [[-1 / 15, 0, 0]], atol=2e-6)
    np.testing.assert_allclose(result.state.particles.velocity, [[-2 / 3, 0, 0]], atol=3e-6)
    np.testing.assert_allclose(result.state.rigid.velocity, [[-2 / 3, 0, 0]], atol=3e-6)
    report = result.report.substeps[0]
    np.testing.assert_allclose(
        report.position_impulses.particle + report.position_impulses.rigid_linear, 0, atol=1e-12
    )
    np.testing.assert_allclose(report.position_impulses.rigid_linear, [[-2 / 1.5, 0, 0]], atol=3e-6)
    for actual, original in zip(fields(result.state), fields(state), strict=True):
        assert not np.shares_memory(actual, original)


def test_joint_multiple_shared_bodies_match_reference_and_repeat_exactly(tyx_mode, cache):
    from phymeo import CoupledXPBDSolver

    pm, bm, spheres, state, inputs = problem(4, 2)
    state.rigid.position[:] = [[0, 0, 0], [4, 0, 0]]
    state.particles.position[:] = [[1.1, 0, 0], [-1.1, 0, 0], [5.1, 0, 0], [2.9, 0, 0]]
    state.particles.velocity[:] = [[-2, 0, 0], [2, 0, 0], [-2, 0, 0], [2, 0, 0]]
    expected = CoupledXPBDSolver(pm, bm, None, spheres, substeps=1, tolerance=1e-6).step(
        state, inputs, 0.1
    )
    with driver()(pm, bm, None, spheres, substeps=1, mode=tyx_mode, cache_root=cache) as solver:
        first, second = solver.step(state, inputs, 0.1), solver.step(state, inputs, 0.1)
    assert first.report.accepted and second.report.accepted and expected.report.accepted
    for actual, repeated, reference in zip(
        fields(first.state), fields(second.state), fields(expected.state), strict=True
    ):
        np.testing.assert_array_equal(actual, repeated)
        np.testing.assert_allclose(actual, reference, atol=3e-5, rtol=0)
    np.testing.assert_allclose(
        first.state.particles.velocity.sum(axis=0) + first.state.rigid.velocity.sum(axis=0),
        0,
        atol=3e-5,
    )


def test_joint_driver_interleaves_distance_and_contact_constraints(tyx_mode, cache):
    from phymeo import DistanceConstraints, ParticleModel

    pm, bm, spheres, state, inputs = problem(2)
    pm = ParticleModel(np.array([0, 1], np.float32), np.zeros(3))
    state.particles.position[:] = [[0, 2, 0], [0, 0.8, 0]]
    edges = DistanceConstraints(
        np.array([[0, 1]]), np.array([1.5], np.float32), np.zeros(1, np.float32)
    )
    with driver()(pm, bm, edges, spheres, substeps=1, mode=tyx_mode, cache_root=cache) as solver:
        result = solver.step(state, inputs, 0.1)
    assert result.report.accepted and result.report.iterations > 1
    np.testing.assert_allclose(result.state.particles.position[0], [0, 2, 0], atol=0)
    np.testing.assert_allclose(result.state.particles.position[1], [0, 0.5, 0], atol=3e-6)
    np.testing.assert_allclose(result.state.rigid.position[0], [0, -0.5, 0], atol=3e-6)


@pytest.mark.parametrize("p_count,b_count", [(0, 0), (0, 2), (2, 0)])
def test_joint_driver_supports_empty_families_and_owned_snapshots(
    tyx_mode, cache, p_count, b_count
):
    pm, bm, spheres, state, inputs = problem(p_count, b_count)
    with driver()(
        pm, bm, None, spheres, substeps=2, record_substeps=True, mode=tyx_mode, cache_root=cache
    ) as solver:
        result = solver.step(state, inputs, 0.1)
    assert result.report.accepted and result.report.completed_substeps == 2
    assert len(result.substep_states) == 2
    for first, second in zip(fields(result.state), fields(result.substep_states[-1]), strict=True):
        assert not np.shares_memory(first, second)


def test_joint_driver_rejects_unconverged_velocity_and_restores_input(tyx_mode, cache):
    pm, bm, spheres, state, inputs = problem(2)
    state.particles.position[:] = [[1.1, 0, 0], [1.4, 0, 0]]
    state.particles.velocity[:] = [[-1, 0, 0], [-4, 0, 0]]
    with driver()(
        pm, bm, None, spheres, substeps=1, iterations=1, mode=tyx_mode, cache_root=cache
    ) as solver:
        result = solver.step(state, inputs, 0.1)
    assert not result.report.accepted and "velocity" in result.report.reason
    for actual, original in zip(fields(result.state), fields(state), strict=True):
        np.testing.assert_array_equal(actual, original)


def test_joint_later_substep_rejection_rolls_back_translation_and_rotation(tyx_mode, cache):
    pm, bm, spheres, state, inputs = problem(2)
    state.particles.position[:] = [[1.2, 0, 0], [-2, 0, 0]]
    state.particles.velocity[:] = [[0, 0, 0], [1.5, 0, 0]]
    state.rigid.angular_velocity[:] = [0, 0, 0.5]
    with driver()(
        pm,
        bm,
        None,
        spheres,
        substeps=2,
        iterations=1,
        record_substeps=True,
        mode=tyx_mode,
        cache_root=cache,
    ) as solver:
        result = solver.step(state, inputs, 1.0)
    assert not result.report.accepted and result.report.completed_substeps == 1
    assert result.substep_states == ()
    for actual, original in zip(fields(result.state), fields(state), strict=True):
        np.testing.assert_array_equal(actual, original)
        assert not np.shares_memory(actual, original)


def test_joint_device_exception_preserves_input_and_subsequent_call_works(
    tyx_mode, cache, monkeypatch
):
    pm, bm, spheres, state, inputs = problem()
    original_state = [array.copy() for array in fields(state)]
    with driver()(pm, bm, None, spheres, substeps=2, mode=tyx_mode, cache_root=cache) as solver:
        real = solver._project_contacts
        calls = 0

        def broken(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected second substep kernel failure")
            return real(*args, **kwargs)

        monkeypatch.setattr(solver, "_project_contacts", broken)
        with pytest.raises(RuntimeError, match="second substep"):
            solver.step(state, inputs, 0.1)
        for actual, original in zip(fields(state), original_state, strict=True):
            np.testing.assert_array_equal(actual, original)
        monkeypatch.setattr(solver, "_project_contacts", real)
        assert solver.step(state, inputs, 0.1).report.accepted


def test_joint_driver_rejects_fp64_and_closed_runtime(tyx_mode, cache):
    pm, bm, spheres, state, inputs = problem(dtype=np.float64)
    with pytest.raises(ValueError, match="FP32"):
        driver()(pm, bm, None, spheres, mode=tyx_mode, cache_root=cache)
    pm, bm, spheres, state, inputs = problem()
    solver = driver()(pm, bm, None, spheres, mode=tyx_mode, cache_root=cache)
    solver.close()
    solver.close()
    with pytest.raises(RuntimeError, match="closed"):
        solver.step(state, inputs, 0.1)


def test_joint_driver_does_not_advertise_a_free_particle_solver_contract(tyx_mode, cache):
    pm, bm, spheres, _, _ = problem()
    with driver()(pm, bm, None, spheres, mode=tyx_mode, cache_root=cache) as solver:
        capabilities = getattr(solver, "capabilities", None)
        assert capabilities is None or capabilities.state_kind == "coupled"


def test_joint_scene_matches_same_color_reference_and_independent_audits(tyx_mode, cache):
    from phymeo.builders.rigid_cloth import rigid_cloth_scene
    from phymeo.solvers.colored_coupled import ColoredCoupledXPBDSolver
    from phymeo.validation.rigid_cloth import audit_substep

    scene = rigid_cloth_scene(np.float32)
    reference = ColoredCoupledXPBDSolver(
        scene.particle_model,
        scene.rigid_model,
        scene.constraints,
        scene.spheres,
        tolerance=1e-6,
        record_substeps=True,
    )
    state, expected = scene.initial_state, scene.initial_state
    contacts = 0
    with driver()(
        scene.particle_model,
        scene.rigid_model,
        scene.constraints,
        scene.spheres,
        tolerance=1e-6,
        record_substeps=True,
        mode=tyx_mode,
        cache_root=cache,
    ) as solver:
        for _ in range(12):
            result = solver.step(state, scene.inputs, 1 / 120)
            cpu = reference.step(expected, scene.inputs, 1 / 120)
            assert result.report.accepted, result.report
            assert cpu.report.accepted, cpu.report
            previous = state
            for sample, report in zip(result.substep_states, result.report.substeps, strict=True):
                metrics = audit_substep(
                    scene,
                    previous,
                    sample,
                    report,
                    float(np.float32(1 / 480)),
                    tolerance=1e-6,
                    reaction=True,
                )
                contacts += metrics["contact"]
                previous = sample
            state, expected = result.state, cpu.state
            np.testing.assert_allclose(
                state.particles.position, expected.particles.position, atol=2e-3, rtol=0
            )
            np.testing.assert_allclose(
                state.rigid.position, expected.rigid.position, atol=2e-3, rtol=0
            )
    assert contacts > 0


def test_scene_runner_records_actual_backend_and_closes_after_callback_failure(
    tyx_mode, cache, monkeypatch
):
    from phymeo.experiments.rigid_cloth import run_scene

    closed = []
    original_close = driver().close

    def close(self):
        original_close(self)
        closed.append(self)

    monkeypatch.setattr(driver(), "close", close)
    run = run_scene(frames=1, mode=tyx_mode, dtype=np.float32, tolerance=1e-6, cache_root=cache)
    assert run.config["backend"] == tyx_mode
    assert run.metrics["execution_backend"] == f"tyx:{tyx_mode}"
    assert len(closed) == 1 and closed[0]._runtime.closed

    def interrupted(*args):
        raise RuntimeError("injected progress callback failure")

    with pytest.raises(RuntimeError, match="progress callback"):
        run_scene(
            frames=1,
            mode=tyx_mode,
            dtype=np.float32,
            tolerance=1e-6,
            cache_root=cache,
            progress=interrupted,
        )
    assert len(closed) == 2 and closed[-1]._runtime.closed


def test_joint_constructor_kernel_failure_closes_both_predictor_runtimes(
    tyx_mode, cache, monkeypatch
):
    import builtins

    import tyx as tx

    pm, bm, spheres, _, _ = problem()
    runtimes = []
    real_runtime, real_import = tx.Runtime, builtins.__import__

    def runtime(*args, **kwargs):
        value = real_runtime(*args, **kwargs)
        runtimes.append(value)
        return value

    def fail(name, *args, **kwargs):
        if name.endswith("_tyx_rigid_kernels"):
            raise RuntimeError("injected rigid import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(tx, "Runtime", runtime)
    monkeypatch.setattr(builtins, "__import__", fail)
    with pytest.raises(RuntimeError, match="rigid import failure"):
        driver()(pm, bm, None, spheres, mode=tyx_mode, cache_root=cache)
    assert len(runtimes) == 2 and all(value.closed for value in runtimes)


@pytest.mark.cuda
@pytest.mark.parametrize("failure", ["allocation", "contact_kernel"])
def test_joint_cuda_failure_releases_prediction_and_projection_buffers(
    tyx_mode, cache, monkeypatch, failure
):
    if tyx_mode != "cuda":
        pytest.skip("requires actual CUDA buffer lifetime checks")
    pm, bm, spheres, state, inputs = problem()
    allocated = []
    with driver()(pm, bm, None, spheres, mode="cuda", cache_root=cache) as solver:
        original = solver._tx.cuda_array

        def allocate(value):
            if failure == "allocation" and len(allocated) == 24:
                raise RuntimeError("injected joint allocation failure")
            device = original(value)
            allocated.append(device)
            return device

        monkeypatch.setattr(solver._tx, "cuda_array", allocate)
        if failure == "contact_kernel":

            def broken(*args, **kwargs):
                raise RuntimeError("injected joint contact failure")

            monkeypatch.setattr(solver, "_project_contacts", broken)
        with pytest.raises(RuntimeError, match="injected joint"):
            solver.step(state, inputs, 0.1)
        assert len(allocated) >= 24
        for device in allocated:
            with pytest.raises(Exception, match="already been freed"):
                device.to_numpy()
    np.testing.assert_array_equal(state.particles.position, 3)
    np.testing.assert_array_equal(state.rigid.position, 0)
