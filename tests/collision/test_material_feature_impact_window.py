"""Real two-sided deformable impact windows through entry, peak force and unloading."""

import json
from contextlib import ExitStack

import numpy as np
import pytest

import phymeo as ph


def initial(family, dtype):
    gap = 3 / 32
    if family == "VT":
        lower = [[-2, -1, 0], [2, -1, 0], [0, 2, 0]]
        upper = [[0, 0, gap], [0.125, 0, 1 + gap], [0, 0.125, 1 + gap]]
        wa, wb = np.array([1 / 3] * 3), np.array([1.0, 0, 0])
        rotation = np.eye(3)
    else:
        lower = [[-1, 0, 0], [1, 0, 0], [0, 0, -1]]
        upper = [[0, -1, gap], [0, 1, gap], [0, 0, 1 + gap]]
        wa = wb = np.array([0.5, 0.5, 0.0])
        rotation = np.array([[0.0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])
    normal = rotation[:, 2]
    drift = np.array([0.125, -0.0625, 0.0])
    x = (np.array(lower + upper) @ rotation.T + [0.25, -0.25, 0.125]).astype(dtype)
    v = np.tile(drift, (6, 1))
    v[3:] -= 0.5 * normal
    keys = [ph.CollisionKey(19, k, 0) for k in ((3, 7) if family == "VT" else (7, 3))]
    shapes = [
        ph.CollisionShape(k, "triangle", n)
        for k, n in zip(keys, ((9, 13, 27), (52, 73, 28)), strict=True)
    ]
    binds = {k: ph.ParticleAttachment(tuple(range(3 * j, 3 * j + 3))) for j, k in enumerate(keys)}
    return (
        ph.CollisionModel(shapes[::-1]),
        binds,
        ph.ParticleState(x, v.astype(dtype)),
        wa,
        wb,
        normal,
    )


def actual_window(family, mode, tmp_path):
    dtype = np.float64 if mode == "host" else np.float32
    model, bind, state, wa, wb, n = initial(family, dtype)
    h = 1 / 256
    mass = 1 / (wa @ wa + wb @ wb)
    # Uniform nodal masses and symmetric material addresses keep each contacting
    # edge or triangle parallel to itself. The two COM forces give this independent oscillator.
    g, u = 3 / 32, -0.5
    pm = ph.ParticleModel(np.ones(6, dtype=dtype), np.zeros(3, dtype=dtype))
    p0 = np.sum(state.velocity.astype(float), axis=0)
    e0 = np.sum(state.velocity.astype(float) ** 2) / 2
    total_impulse = 0.0
    trajectory = []
    with ExitStack() as run:
        work = run.enter_context(ph.CollisionWorkspace(model, mode, cache_root=tmp_path / "cache"))
        law = (
            None
            if mode == "host"
            else run.enter_context(ph.TyxNormalLayer(mode, cache_root=tmp_path / "cache"))
        )
        solver = (
            ph.SemiImplicitSolver()
            if mode == "host"
            else run.enter_context(ph.TyxSemiImplicitSolver(mode, cache_root=tmp_path / "cache"))
        )
        system = ph.DeformableFeatureLoadSystem(
            model, bind, ph.NormalLayer(1 / 16, 128.0), search_distance_m=3 / 32, layer_operator=law
        )
        for tick in range(113):
            x = state.position.astype(float)
            v = state.velocity.astype(float)
            actual_g = (wb @ x[3:] - wa @ x[:3]) @ n
            actual_u = (wb @ v[3:] - wa @ v[:3]) @ n
            assert actual_g > 0 and abs(actual_g - g) < 2e-5 and abs(actual_u - u) < 5e-4
            sample = system.sample_particles(state, work)
            assert all(c.feature.kind == family for c in sample.contacts)
            actual_n = float(sample.nodal_force_n[3:].sum(axis=0) @ n)
            expected_n = 128 * max(1 / 16 - g, 0.0)
            assert abs(actual_n - expected_n) < 0.005
            np.testing.assert_allclose(
                sample.nodal_force_n[:3].sum(axis=0), -actual_n * n, rtol=0, atol=3e-5
            )
            momentum_error = float(np.max(np.abs(v.sum(axis=0) - p0)))
            energy = float(np.sum(v**2) / 2 + sample.potential_j)
            assert momentum_error < 2e-5 and abs(energy - e0) < 0.006
            trajectory.append(
                dict(
                    tick=tick,
                    time_s=tick * h,
                    gap_m=float(actual_g),
                    relative_velocity_m_s=float(actual_u),
                    force_n=actual_n,
                    source_features=len(sample.contacts),
                    energy_j=energy,
                    momentum_error_ns=momentum_error,
                    position_m=x.tolist(),
                    velocity_m_s=v.tolist(),
                    nodal_force_n=sample.nodal_force_n.tolist(),
                )
            )
            if tick < 112:
                total_impulse += actual_n * h
                new = solver.step(pm, state, sample.step_inputs(), h)
                assert new.report.accepted
                state = new.state
                u += h * expected_n / mass
                g += h * u
        if mode != "host":
            assert law.stats()["completed_launches"] > 0 and work.stats()["completed_launches"] > 0
    active = [r["tick"] for r in trajectory if r["force_n"] > 0.001]
    assert active == list(range(active[0], active[-1] + 1)) and 0 < active[0] < active[-1] < 112
    peak = max(trajectory, key=lambda r: r["force_n"])
    assert active[0] < peak["tick"] < active[-1] and peak["force_n"] > 4
    assert trajectory[-1]["relative_velocity_m_s"] > 0.49 and trajectory[-1]["gap_m"] > 1 / 16
    assert abs(total_impulse - mass * (trajectory[-1]["relative_velocity_m_s"] + 0.5)) < 0.0001
    evidence = dict(
        family=family,
        mode=mode,
        steps=112,
        h=h,
        effective_mass_kg=float(mass),
        first_loaded_step=active[0],
        peak_step=peak["tick"],
        last_loaded_step=active[-1],
        peak_n=peak["force_n"],
        min_gap_m=min(r["gap_m"] for r in trajectory),
        outgoing_m_s=trajectory[-1]["relative_velocity_m_s"],
        total_impulse_ns=total_impulse,
        peak_energy_error_j=max(abs(r["energy_j"] - e0) for r in trajectory),
        max_momentum_error_ns=max(r["momentum_error_ns"] for r in trajectory),
        steps_observed=trajectory,
    )
    (tmp_path / ("material-impact-" + family + ".json")).write_text(
        json.dumps(evidence, allow_nan=False), encoding="utf8"
    )


@pytest.mark.parametrize("family", ["VT", "EE"])
def test_material_feature_impact_and_unloading(family, tmp_path):
    actual_window(family, "host", tmp_path)


@pytest.mark.tyx
@pytest.mark.parametrize("family", ["VT", "EE"])
def test_device_material_velocity_force_and_unloading(family, tyx_mode, tmp_path):
    actual_window(family, tyx_mode, tmp_path)
