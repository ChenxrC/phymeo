"""Geometry-driven finite layer loads and independent RR/RD/DD dynamics windows."""

import numpy as np
import pytest
from load_cases import scene, window
from load_history import save_history

import phymeo as ph
from phymeo.core.capabilities import SolverFailure


def test_stack_equilibrium_and_four_corner_loads():
    system, state, _, _ = scene("RR")
    with ph.CollisionWorkspace(system.model) as work:
        out = system.sample(state, work)
    assert [c.force_n for c in out.contacts] == pytest.approx([2, 1])
    assert [len(c.sites) for c in out.contacts] == [4, 4]
    assert out.inputs.rigid.force[:, 2] == pytest.approx([-2, 1, 1])
    np.testing.assert_allclose(out.inputs.rigid.torque, 0, atol=1e-14)
    assert sum(c.potential_j for c in out.contacts) == pytest.approx(5 / 256)


def test_parallel_cloth_face_distributes_force_and_power_to_material():
    system, state, _, _ = scene("DD")
    state.particles.position[3:] = state.particles.position[:3] + (0, 0, 1 / 32)
    state.particles.velocity[:] = 0
    state.particles.velocity[3:, 2] = -0.5
    with ph.CollisionWorkspace(system.model) as work:
        out = system.sample(state, work)
    (c,) = out.contacts
    assert len(c.sites) == 3
    assert c.force_n == pytest.approx(4.5)
    assert out.inputs.particles.force[:, 2] == pytest.approx([-1.5] * 3 + [1.5] * 3)
    assert c.power_w == pytest.approx(-2.25)
    assert c.potential_j == pytest.approx(1 / 16)
    p = state.particles
    np.testing.assert_allclose(
        np.cross(p.position, out.inputs.particles.force).sum(axis=0), 0, atol=1e-12
    )


def test_capacity_and_interior_crossing_do_not_commit():
    system, state, models, constraints = scene("RD")
    state.rigid.position[0, 2] = 0.16
    loads = ph.CoupledInputs.zeros(3, 1)
    solvers = ph.XPBDSolver(constraints), ph.RigidSemiImplicitSolver()
    with ph.CollisionWorkspace(system.model) as work:
        good = system.step(state, loads, models, solvers, 1 / 64, work)
        original = state.rigid.position.copy()
        with pytest.raises(SolverFailure, match="capacity"):
            system.step(state, loads, models, solvers, 1 / 64, work, capacity=0)
        assert system.last_frame is good
        np.testing.assert_array_equal(state.rigid.position, original)
        retry = system.step(state, loads, models, solvers, 1 / 64, work)
        np.testing.assert_array_equal(retry.state.rigid.position, good.state.rigid.position)
        state.rigid.position[0, 2] = 0.5
        state.rigid.velocity[0, 2] = -64
        with pytest.raises(SolverFailure, match="core"):
            system.step(state, loads, models, solvers, 1 / 64, work)
        assert system.last_frame is retry


@pytest.mark.parametrize("family", ["RR", "RD", "DD"])
def test_dynamic_window(family, tmp_path):
    frames, stats = window(family)
    save_history(tmp_path / (family + ".json"), family, frames, stats)
    check_window(family, frames, 1e-9)


def check_window(family, frames, tol):
    assert len(frames) == (128 if family == "DD" else 42)
    if family == "RR":
        for f in frames[:6]:
            assert [c.force_n for c in f.loads.contacts] == pytest.approx([2, 1], abs=tol)
        for f in frames:
            assert f.state.rigid.position[0].tolist() == [0.0, 0.0, 0.0]
        assert frames[-1].state.rigid.position[2, 2] > 77 / 128 + 0.06
        assert not any(c.second.body == 12 for c in frames[-1].loads.contacts)
    else:
        assert not frames[0].loads.contacts
        active = [f for f in frames if f.loads.contacts]
        assert len(active) >= 4
        assert max(f.loads.contacts[0].force_n for f in active) > 1.0
        assert not frames[-1].loads.contacts
        for f in frames:
            inputs = f.loads.inputs
            total = inputs.particles.force.sum(axis=0) + inputs.rigid.force.sum(axis=0)
            np.testing.assert_allclose(total, 0, atol=tol)
            v = f.state.particles.velocity.sum(axis=0)
            if family == "RD":
                v += 0.5 * f.state.rigid.velocity[0]
            np.testing.assert_allclose(v, [0, 0, -0.25 if family == "RD" else 0], atol=tol)
        if family == "DD":
            # Preserve the original observation at 42 steps and the second approach.
            phases = [bool(f.loads.contacts) for f in frames[:42]]
            starts = [i for i, x in enumerate(phases) if x and (not i or not phases[i - 1])]
            assert len(starts) >= 2
            assert frames[41].loads.contacts and frames[41].loads.contacts[0].power_w < 0.0
            assert all(not f.loads.contacts for f in frames[-8:])
        last = frames[-1].state
        assert np.ptp(last.particles.velocity[:3, 2]) > 0.005
        if family == "RD":
            assert last.rigid.velocity[0, 2] > 0.0
            assert last.particles.velocity[:, 2].mean() < 0.0
        else:
            assert last.particles.velocity[:3, 2].mean() < 0.25
            assert last.particles.velocity[3:, 2].mean() > -0.25


def test_eccentric_ball_uses_com_velocity_moment_and_equal_opposite_work():
    system, state, _, _ = scene("RD")
    sphere, cloth = system.model.shapes
    binds = dict(system.bindings)
    binds[sphere.key] = ph.RigidAttachment(0, ((0.125, 0, 0),))
    system = ph.CollisionLoadSystem(system.model, binds, system.layers)
    state.rigid.position[0, 2] = 0.125 + 0.03125
    state.rigid.angular_velocity[0, 1] = 2.0
    out_velocity = state.rigid.velocity[0, 2] - 0.125 * 2
    with ph.CollisionWorkspace(system.model) as work:
        out = system.sample(state, work)
    (c,) = out.contacts
    assert c.force_n == pytest.approx(4 * (1 - 0.25 * out_velocity))
    assert c.sites[0].rate_m_per_s == pytest.approx(out_velocity)
    assert out.inputs.rigid.torque[0, 1] == pytest.approx(-0.125 * c.force_n)
    total = np.cross(state.particles.position, out.inputs.particles.force).sum(axis=0)
    total += np.cross(state.rigid.position, out.inputs.rigid.force).sum(
        axis=0
    ) + out.inputs.rigid.torque.sum(axis=0)
    np.testing.assert_allclose(total, 0.0, atol=1e-12)
    actual_power = np.sum(out.inputs.rigid.force * state.rigid.velocity) + np.sum(
        out.inputs.rigid.torque * state.rigid.angular_velocity
    )
    assert actual_power == pytest.approx(c.power_w)


def test_invalid_pair_filters_binding_and_layer_fail_explicitly():
    system, _, _, _ = scene("DD")
    a, b = system.model.shapes
    for params in [(0.0, 1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, float("nan"))]:
        with pytest.raises(ValueError):
            ph.NormalLayer(*params)
    with pytest.raises(ValueError, match="filtered"):
        ph.CollisionLoadSystem(
            ph.CollisionModel((a, b), ((a.key, b.key),)), system.bindings, system.layers
        )
    with pytest.raises(ValueError, match="identity"):
        ph.CollisionLoadSystem(
            system.model,
            {a.key: ph.ParticleAttachment((1, 2, 3)), b.key: ph.ParticleAttachment((3, 4, 5))},
            system.layers,
        )
