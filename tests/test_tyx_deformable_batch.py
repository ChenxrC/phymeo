"""D1 persistent native node states: fixed, independently authored cases."""
from dataclasses import replace

import numpy as np
import pytest

from phymeo.backends.tyx_deformable_batch import (
    DeformableNodeWorld, DeformableSurface, TyxDeformableBatch,
)
from phymeo.constraints.distance import DistanceConstraints, project_distances
from phymeo.core.model import ParticleModel
from phymeo.core.scene import StepSettings
from phymeo.core.state import ParticleState, StepInputs

pytestmark = pytest.mark.tyx
H = 1 / 128
SETTINGS = StepSettings(substeps=1, position_iterations=200, tolerance=2e-6)


def world(key=7, square=False, fixed=False, permutation=False, empty=False):
    x = np.array(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)) if square
                 else ((0, 0, 0), (1, 0, 0), (0, 1, 0)), np.float32)
    v = np.zeros_like(x)
    if square:
        v[2:, 0] = 1 / 16
        edges = np.array(((0, 1), (1, 2), (2, 3), (3, 0), (0, 2)), np.int64)
        ids = np.array((101, 113, 127, 139), np.int64)
        faces = np.array(((101, 113, 127), (101, 127, 139)), np.int64)
        facets = np.array((503, 509), np.int64)
    else:
        v[:2, 0] = (-1 / 16, 1 / 16)
        edges = np.array(((0, 1),), np.int64)
        ids = np.array((11, 37, 91), np.int64)
        faces = np.array(((11, 37, 91),), np.int64)
        facets = np.array((501,), np.int64)
    weights = np.ones(len(x), np.float32)
    if fixed:
        weights[0], v[0] = 0, 0
    rest = np.linalg.norm(x[edges[:, 0]].astype(float) - x[edges[:, 1]], axis=1)
    compliance = np.full(len(edges), H * H)
    if empty:
        edges, rest, compliance = np.empty((0, 2), np.int64), np.zeros(0), np.zeros(0)
    if permutation:
        order = np.arange(len(x))[::-1]
        reverse = np.argsort(order)
        x, v, weights, ids, edges = x[order], v[order], weights[order], ids[order], reverse[edges]
        facets, faces = facets[::-1], faces[::-1]
    return DeformableNodeWorld(
        key, ParticleModel(weights, np.zeros(3, np.float32)), ParticleState(x, v),
        DistanceConstraints(edges, rest, compliance), ids,
        (DeformableSurface(23, facets, faces),),
    )


def forces(specs, tick=0):
    result = {}
    for spec in specs:
        value = np.zeros((spec.model.particle_count, 3), np.float32)
        if spec.key == 31:
            value[:, 1] = (1 / 32 if tick < 6 else -1 / 32)
        result[spec.key] = StepInputs(value)
    return result


def arrays_equal(left, right):
    for key in left:
        a, b = left[key], right[key]
        assert (a.world, a.tick, a.time) == (b.world, b.tick, b.time)
        for name in ("position", "velocity", "clock"):
            av, bv = getattr(a, name), getattr(b, name)
            assert av.dtype == bv.dtype and av.shape == bv.shape and av.tobytes() == bv.tobytes()
        assert a.last_step is b.last_step


@pytest.mark.parametrize("square,fixed", [(False, False), (True, False), (True, True)])
def test_real_ordered_deformation(square, fixed, tyx_mode, tmp_path):
    from phymeo.backends.tyx_node_math import TyxNodeMath
    spec = world(square=square, fixed=fixed)
    evidence = []
    with TyxDeformableBatch((spec,), mode=tyx_mode, settings=SETTINGS) as batch:
        original = batch.snapshot()[7].position.copy()
        with TyxNodeMath(tyx_mode, cache_root=tmp_path) as old_math:
            for tick in range(12):
                entry = batch.snapshot()[7]
                packet = forces((spec,), tick)
                before = dict(batch.stats())
                receipt = batch.step(packet, H, expected_revision=batch.revision)
                state = batch.snapshot()[7]
                row = receipt.worlds[7]
                # Exact old device arithmetic, separately invoked, never fed back.
                px, pv = old_math.predict(entry.position, entry.velocity, packet[7].force,
                                         spec.model.inverse_mass, spec.model.gravity, H)
                alpha = old_math.axpy(np.zeros(len(spec.constraints.indices)),
                                     spec.constraints.compliance, 1 / H)
                alpha = old_math.axpy(np.zeros(len(alpha)), alpha, 1 / H)
                x, lam = px, np.zeros(len(alpha))
                for _ in range(row.sweeps):
                    x, lam = old_math.project_distances(x, spec.model.inverse_mass,
                                                      spec.constraints, alpha, lam)
                velocity = old_math.reconstruct(px, pv, x, spec.model.inverse_mass, H)
                np.testing.assert_array_equal(row.predicted_position, px)
                np.testing.assert_array_equal(row.predicted_velocity, pv)
                np.testing.assert_array_equal(state.position, x)
                np.testing.assert_array_equal(state.velocity, velocity)
                np.testing.assert_array_equal(row.multipliers, lam)
                # Independent HOST serial projection at this actual input.
                host_x = px.astype(float).copy()
                host_l = np.zeros(len(alpha))
                for _ in range(row.sweeps):
                    project_distances(host_x, spec.model.inverse_mass, spec.constraints,
                                      alpha.astype(float), host_l)
                np.testing.assert_allclose(state.position, host_x, rtol=2e-5, atol=2e-5)
                np.testing.assert_allclose(row.multipliers, host_l, rtol=2e-5, atol=2e-5)
                np.testing.assert_allclose(
                    state.velocity, row.predicted_velocity +
                    (state.position.astype(float) - row.predicted_position) / H,
                    rtol=2e-5, atol=2e-5,
                )
                assert row.residual <= SETTINGS.tolerance and receipt.accepted
                assert state.tick == tick + 1 and state.time == (tick + 1) * H
                assert batch.stats()["state_uploaded_bytes"] == before["state_uploaded_bytes"]
                assert batch.stats()["allocations"] == before["allocations"]
                assert not state.position.flags.writeable
                with pytest.raises(ValueError):
                    state.position.setflags(write=True)
                if fixed:
                    np.testing.assert_array_equal(state.position[0], original[0])
                    np.testing.assert_array_equal(state.velocity[0], 0)
                if not square and tick == 0:
                    length = np.linalg.norm(state.position[0] - state.position[1])
                    assert length > 1 + H / 48
                    assert length == pytest.approx(1 + H / 24, rel=2e-5, abs=2e-5)
                evidence.append((state.position, state.velocity, row.multipliers, row.sweeps))
        edges = spec.constraints.indices
        old_lengths = np.linalg.norm(original[edges[:, 0]] - original[edges[:, 1]], axis=1)
        new_lengths = np.linalg.norm(state.position[edges[:, 0]] - state.position[edges[:, 1]], axis=1)
        assert np.max(np.abs(new_lengths - old_lengths)) > 1e-6
        np.savez_compressed(tmp_path / "deformation.npz",
                            position=np.asarray([v[0] for v in evidence]),
                            velocity=np.asarray([v[1] for v in evidence]),
                            multiplier=np.asarray([v[2] for v in evidence]),
                            sweeps=np.asarray([v[3] for v in evidence]))


def test_sparse_world_permutation_and_force_isolation(tyx_mode, tmp_path):
    specs = (world(), world(31, square=True, fixed=True))
    permuted = (world(31, square=True, fixed=True, permutation=True),
                world(permutation=True))
    with TyxDeformableBatch(specs, mode=tyx_mode, settings=SETTINGS) as a, \
         TyxDeformableBatch(permuted, mode=tyx_mode, settings=SETTINGS) as b:
        for tick in range(12):
            for owner, declarations in ((a, specs), (b, permuted)):
                assert owner.step(forces(declarations, tick), H,
                                  expected_revision=owner.revision).accepted
            for sa, sb in zip(specs, reversed(permuted), strict=True):
                aa, bb = a.snapshot()[sa.key], b.snapshot()[sb.key]
                np.testing.assert_allclose(aa.position[np.argsort(sa.point_ids)],
                                           bb.position[np.argsort(sb.point_ids)], rtol=2e-5, atol=2e-5)
                np.testing.assert_allclose(aa.velocity[np.argsort(sa.point_ids)],
                                           bb.velocity[np.argsort(sb.point_ids)], rtol=2e-5, atol=2e-5)
        assert a.world_keys == (7, 31) and b.world_keys == (31, 7)
        np.savez_compressed(tmp_path / "two-worlds.npz",
                            first=a.snapshot()[7].position, second=a.snapshot()[31].position)


@pytest.mark.parametrize("trace", [False, True])
def test_zero_edges_and_exact_resource_budget(trace, tyx_mode):
    spec = world(empty=True)
    need = TyxDeformableBatch.requirements((spec,), settings=SETTINGS, trace=trace)
    n, p = spec.model.particle_count, 1
    assert need.device_bytes == 88 * n + 32 * p + 40 + (72 * p if trace else 0)
    assert need.total_bytes == need.device_bytes + need.host_peak_bytes
    with TyxDeformableBatch((spec,), mode=tyx_mode, settings=SETTINGS, trace=trace,
                           memory_limit_bytes=need.total_bytes) as batch:
        result = batch.step(forces((spec,)), H, expected_revision=0)
        assert result.accepted and result.worlds[7].sweeps == 0
        assert result.worlds[7].multipliers.shape == (0,)
        assert batch.stats()["native_buffer_bytes"] == need.device_bytes
    with pytest.raises(MemoryError):
        TyxDeformableBatch((spec,), mode=tyx_mode, settings=SETTINGS, trace=trace,
                          memory_limit_bytes=need.total_bytes - 1)


def test_budget_preflight_before_runtime(monkeypatch):
    import phymeo.backends._tyx_deformable_buffers as buffers
    spec = world()
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("runtime must not be constructed")
    monkeypatch.setattr(buffers, "create_runtime", forbidden)
    need = TyxDeformableBatch.requirements((spec,), settings=SETTINGS)
    with pytest.raises(MemoryError):
        TyxDeformableBatch((spec,), settings=SETTINGS, memory_limit_bytes=need.total_bytes - 1)
    assert calls == []


@pytest.mark.parametrize("bad", ["duplicate-point", "unknown-facet-point", "duplicate-facet",
                                "degenerate", "state-f64", "fixed-velocity", "world-duplicate",
                                "node-cap", "iterations", "mode"])
def test_registration_rejects_before_runtime(bad, monkeypatch):
    import phymeo.backends._tyx_deformable_buffers as buffers
    spec = world()
    calls = []
    monkeypatch.setattr(buffers, "create_runtime", lambda *a, **k: calls.append(True))
    with pytest.raises((ValueError, TypeError)):
        if bad == "duplicate-point":
            spec = replace(spec, point_ids=np.array([11, 11, 91], np.int64))
        elif bad == "unknown-facet-point":
            spec = replace(spec, surfaces=(DeformableSurface(
                23, np.array([501], np.int64), np.array([[11, 37, 100]], np.int64)),))
        elif bad == "duplicate-facet":
            spec = replace(spec, surfaces=(DeformableSurface(
                23, np.array([501, 501], np.int64), np.array([[11,37,91],[11,37,91]], np.int64)),))
        elif bad == "degenerate":
            state = spec.initial_state.copy()
            state.position[2] = state.position[0]
            spec = replace(spec, initial_state=state)
        elif bad == "state-f64":
            spec = replace(spec, initial_state=ParticleState(
                spec.initial_state.position.astype(float), spec.initial_state.velocity.astype(float)))
        elif bad == "fixed-velocity":
            spec = replace(spec, model=ParticleModel(np.array([0,1,1], np.float32),
                                                     np.zeros(3, np.float32)))
        elif bad == "node-cap":
            x = np.zeros((513,3), np.float32)
            spec = DeformableNodeWorld(7, ParticleModel(np.ones(513, np.float32)),
                ParticleState(x,x), DistanceConstraints(np.empty((0,2), np.int64),
                np.zeros(0),np.zeros(0)), np.arange(513,dtype=np.int64), ())
        TyxDeformableBatch((spec,spec) if bad == "world-duplicate" else (spec,),
                          settings=replace(SETTINGS, position_iterations=201)
                          if bad == "iterations" else SETTINGS,
                          mode="cpu_cpp" if bad == "mode" else "interpret")
    assert not calls


def test_trace_is_original_complete_projection(tyx_mode):
    spec = world(square=True)
    with TyxDeformableBatch((spec,), mode=tyx_mode, settings=SETTINGS, trace=True) as traced, \
         TyxDeformableBatch((spec,), mode=tyx_mode, settings=SETTINGS) as plain:
        for _ in range(3):
            a = traced.step(forces((spec,)), H, expected_revision=traced.revision)
            b = plain.step(forces((spec,)), H, expected_revision=plain.revision)
            np.testing.assert_array_equal(traced.snapshot()[7].position, plain.snapshot()[7].position)
            np.testing.assert_array_equal(traced.snapshot()[7].velocity, plain.snapshot()[7].velocity)
            assert len(a.worlds[7].trace) == a.worlds[7].sweeps
            assert all(frame.shape == (5,18) and not frame.flags.writeable for frame in a.worlds[7].trace)
            assert b.worlds[7].trace == ()


def test_no_cpu_projection_fallback(tyx_mode, monkeypatch):
    import phymeo.constraints.distance as distance
    spec = world()
    with TyxDeformableBatch((spec,), mode=tyx_mode, settings=SETTINGS) as batch:
        def forbidden(*args, **kwargs):
            raise AssertionError("CPU projection used to produce device state")
        monkeypatch.setattr(distance, "project_distances", forbidden)
        assert batch.step(forces((spec,)), H, expected_revision=0).accepted
