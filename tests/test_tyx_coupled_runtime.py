"""Actual backend kernels and resource lifetime through the new generic schedule."""

import importlib
from contextlib import ExitStack

import numpy as np
import pytest
from test_coupled_runtime import arrays, compare_results, pair

from phymeo.backends.tyx_coupled import TyxCoupledXPBDSolver
from phymeo.builders.rigid_cloth import rigid_cloth_scene
from phymeo.core.capabilities import SolverFailure
from phymeo.core.scene import fingerprint
from phymeo.solvers.coupled_program import NumpyCoupledProgram, make_coupled_runtime
from phymeo.validation.rigid_cloth import audit_substep

pytestmark = pytest.mark.tyx


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("coupled-runtime-tyx")


def module():
    return importlib.import_module("phymeo.backends.tyx_coupled_program")


def test_actual_device_pair_preserves_old_arithmetic_and_avoids_old_schedule(
    tyx_mode, cache, monkeypatch
):
    pm, bm, shapes, state, inputs = pair(np.float32)
    args = pm, bm, None, shapes
    with TyxCoupledXPBDSolver(
        *args, mode=tyx_mode, cache_root=cache, substeps=2, record_substeps=True
    ) as old:
        expected = old.step(state, inputs, 0.1)

    def forbidden(*args, **kwargs):
        raise AssertionError("old device scheduler executed")

    monkeypatch.setattr(TyxCoupledXPBDSolver, "step", forbidden)
    monkeypatch.setattr(TyxCoupledXPBDSolver, "_substep", forbidden)
    with make_coupled_runtime(
        *args, mode=tyx_mode, cache_root=cache, substeps=2, record_substeps=True
    ) as new:
        actual = new.step(state, inputs, 0.1)
        assert new.last_scene_result.report.backend == f"tyx:{tyx_mode}"
        compare_results(actual, expected)
        np.testing.assert_allclose(actual.state.particles.velocity, [[0, -1 / 3, 0]], atol=2e-6)
        assert fingerprint(new.program.problem_signature()) == fingerprint(
            NumpyCoupledProgram(*args).problem_signature()
        )
        engine = new.program.engine
    assert engine._runtime.closed and engine._rigid_solver._runtime.closed


def test_device_twelve_frame_full_scene_matches_old_with_independent_audits(tyx_mode, cache):
    scene = rigid_cloth_scene(np.float32)
    args = scene.particle_model, scene.rigid_model, scene.constraints, scene.spheres
    with (
        TyxCoupledXPBDSolver(*args, mode=tyx_mode, cache_root=cache, record_substeps=True) as old,
        make_coupled_runtime(*args, mode=tyx_mode, cache_root=cache, record_substeps=True) as new,
    ):
        previous = scene.initial_state
        for _ in range(12):
            actual = new.step(previous, scene.inputs, 1 / 120)
            compare_results(actual, old.step(previous, scene.inputs, 1 / 120))
            before = previous
            h = float(np.float32(float(np.float32(1 / 120)) / 4))
            for after, report in zip(actual.substep_states, actual.report.substeps, strict=True):
                audit_substep(scene, before, after, report, h, tolerance=1e-6, reaction=True)
                before = after
            previous = actual.state


@pytest.mark.parametrize("phase", ["position_sweep", "velocity_sweep", "close"])
def test_late_device_failure_rolls_back_and_releases_workspace(tyx_mode, cache, phase):
    pm, bm, shapes, state, inputs = pair(np.float32)
    with make_coupled_runtime(
        pm, bm, None, shapes, mode=tyx_mode, cache_root=cache, substeps=2, record_substeps=True
    ) as new:
        create = new.program.create_workspace
        count, closed = 0, []

        def inject(*args):
            nonlocal count
            count += 1
            workspace = create(*args)
            close = workspace.close

            def cleanup():
                close()
                closed.append(True)
                if count == 2 and phase == "close":
                    raise SolverFailure("device close failure")

            workspace.close = cleanup
            if count == 2 and phase != "close":

                def failure():
                    raise SolverFailure("device phase failure")

                setattr(workspace, phase, failure)
            return workspace

        new.program.create_workspace = inject
        result = new.step(state, inputs, 0.1)
        assert not result.report.accepted and result.substep_states == () and len(closed) == 2
        for a, b in zip(arrays(result.state), arrays(state), strict=True):
            np.testing.assert_array_equal(a, b)
        new.program.create_workspace = create
        assert new.step(state, inputs, 0.1).report.accepted


def test_partial_device_allocation_failure_frees_every_registered_buffer(
    tyx_mode, cache, monkeypatch
):
    implementation = module()
    counts = {"allocated": 0, "freed": 0, "calls": 0}

    class TrackedStack(ExitStack):
        def callback(self, callback, *args, **kwargs):
            if getattr(callback, "__name__", "") == "free":
                counts["allocated"] += 1
                original = callback

                def callback(*a, **kw):
                    original(*a, **kw)
                    counts["freed"] += 1

            return super().callback(callback, *args, **kwargs)

    monkeypatch.setattr(implementation, "ExitStack", TrackedStack)
    allocate = implementation._DeviceWorkspace._buffer

    def broken(self, value):
        counts["calls"] += 1
        if counts["calls"] == 3:
            raise SolverFailure("allocation failure")
        return allocate(self, value)

    monkeypatch.setattr(implementation._DeviceWorkspace, "_buffer", broken)
    pm, bm, shapes, state, inputs = pair(np.float32)
    with make_coupled_runtime(pm, bm, None, shapes, mode=tyx_mode, cache_root=cache) as new:
        result = new.step(state, inputs, 0.1)
        assert not result.report.accepted and result.substep_states == ()
        assert counts["allocated"] == counts["freed"]
        if tyx_mode == "cuda":
            assert counts["allocated"] == 2
        monkeypatch.setattr(implementation._DeviceWorkspace, "_buffer", allocate)
        assert new.step(state, inputs, 0.1).report.accepted


def test_rejected_precision_does_not_construct_device_runtime(tyx_mode, cache):
    pm, bm, shapes, _, _ = pair(np.float64)
    with pytest.raises(ValueError, match="FP32"):
        make_coupled_runtime(pm, bm, None, shapes, mode=tyx_mode, cache_root=cache)


@pytest.mark.parametrize("counts", [(0, 0), (0, 1), (1, 0)])
def test_empty_device_domains(tyx_mode, cache, counts):
    from phymeo import (
        CoupledInputs,
        CoupledState,
        ParticleModel,
        ParticleState,
        RigidBodyModel,
        RigidBodyState,
        SphereShapes,
    )

    n, m = counts
    pm = ParticleModel(np.ones(n, np.float32), np.zeros(3, np.float32))
    bm = RigidBodyModel(
        np.ones(m, np.float32), np.ones((m, 3), np.float32), np.zeros(3, np.float32)
    )
    shapes = SphereShapes(np.ones(m, np.float32))
    state = CoupledState(
        ParticleState(np.zeros((n, 3), np.float32), np.zeros((n, 3), np.float32)),
        RigidBodyState(
            np.zeros((m, 3), np.float32),
            np.tile(np.array([0, 0, 0, 1], np.float32), (m, 1)),
            np.zeros((m, 3), np.float32),
            np.zeros((m, 3), np.float32),
        ),
    )
    with make_coupled_runtime(pm, bm, None, shapes, mode=tyx_mode, cache_root=cache) as new:
        result = new.step(state, CoupledInputs.zeros(n, m, dtype=np.float32), 0.1)
    assert result.report.accepted
    for a, b in zip(arrays(state), arrays(result.state), strict=True):
        np.testing.assert_array_equal(a, b)


def test_actual_kernel_exception_preserves_state_and_frees_cuda_arrays(
    tyx_mode, cache, monkeypatch
):
    pm, bm, shapes, state, inputs = pair(np.float32)
    with make_coupled_runtime(
        pm, bm, None, shapes, mode=tyx_mode, cache_root=cache, substeps=2
    ) as new:
        engine = new.program.engine
        allocated, calls = [], 0
        allocate, project = engine._tx.cuda_array, engine._project_contacts

        def tracked(value):
            result = allocate(value)
            allocated.append(result)
            return result

        def broken(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected kernel execution error")
            return project(*args, **kwargs)

        monkeypatch.setattr(engine._tx, "cuda_array", tracked)
        monkeypatch.setattr(engine, "_project_contacts", broken)
        with pytest.raises(RuntimeError, match="injected kernel"):
            new.step(state, inputs, 0.1)
        if tyx_mode == "cuda":
            assert allocated
            for device in allocated:
                with pytest.raises(Exception, match="already been freed"):
                    device.to_numpy()
        np.testing.assert_array_equal(state.particles.position, [[0, 1, 0]])
        monkeypatch.setattr(engine, "_project_contacts", project)
        assert new.step(state, inputs, 0.1).report.accepted


def test_sync_failure_during_close_still_runs_all_free_callbacks(tyx_mode, cache, monkeypatch):
    implementation = module()
    evidence = {"registered": 0, "freed": 0, "fail": True}

    class Stack(ExitStack):
        def callback(self, callback, *args, **kwargs):
            name = getattr(callback, "__name__", "")
            original = callback
            if name == "free":
                evidence["registered"] += 1

                def callback(*a, **kw):
                    original(*a, **kw)
                    evidence["freed"] += 1
            elif name == "sync":

                def callback(*a, **kw):
                    original(*a, **kw)
                    if evidence["fail"]:
                        raise SolverFailure("injected synchronization failure")

            return super().callback(callback, *args, **kwargs)

    monkeypatch.setattr(implementation, "ExitStack", Stack)
    pm, bm, shapes, state, inputs = pair(np.float32)
    with make_coupled_runtime(pm, bm, None, shapes, mode=tyx_mode, cache_root=cache) as new:
        result = new.step(state, inputs, 0.1)
        assert not result.report.accepted and "synchronization" in result.report.reason
        assert evidence["registered"] == evidence["freed"]
        if tyx_mode == "cuda":
            assert evidence["registered"] > 0
        evidence["fail"] = False
        assert new.step(state, inputs, 0.1).report.accepted
