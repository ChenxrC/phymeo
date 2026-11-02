"""Finite pad forces at both faces, rim and corner, including off-plane sliding work."""

import numpy as np
import pytest

import phymeo as ph


def pad_state(pos, vel=(0.0, 0.0, 0.0), *, swap=False, operator=None):
    body, cloth = (2, 1) if swap else (1, 2)
    vertices = ((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0))
    p = ph.CollisionShape(ph.CollisionKey(0, body, 0), "plate", tuple(range(4)), radius=1 / 16)
    c = ph.CollisionShape(ph.CollisionKey(0, cloth, 0), "particle", (30,))
    model = ph.CollisionModel((p, c))
    binds = {p.key: ph.RigidAttachment(0, vertices), c.key: ph.ParticleAttachment((0,))}
    law = ph.NormalLayer(1 / 16, 128.0, friction=0.5, slip_speed_m_s=0.25)
    system = ph.CollisionLoadSystem(model, binds, {(p.key, c.key): law}, layer_operator=operator)
    state = ph.CoupledState(
        ph.ParticleState(np.array([pos], float), np.array([vel], float)),
        ph.RigidBodyState(
            np.zeros((1, 3)), np.array([[0.0, 0.0, 0.0, 1.0]]), np.zeros((1, 3)), np.zeros((1, 3))
        ),
    )
    return system, state


@pytest.mark.parametrize(
    ("pos", "axis", "magnitude"),
    [
        ((0.0, 0.0, 3 / 32), (0.0, 0.0, 1.0), 4.0),
        ((0.0, 0.0, -3 / 32), (0.0, 0.0, -1.0), 4.0),
        ((0.5 + 3 / 32, 0.0, 0.0), (1.0, 0.0, 0.0), 4.0),
        ((0.5 + 3 / 64, 0.5 + 4 / 64, 0.0), (0.6, 0.8, 0.0), 6.0),
    ],
)
@pytest.mark.parametrize("swap", [False, True])
def test_front_back_rim_and_corner_magnitude(pos, axis, magnitude, swap):
    system, state = pad_state(pos, swap=swap)
    with ph.CollisionWorkspace(system.model) as work:
        loads = system.sample(state, work)
    f = np.array(axis) * magnitude
    np.testing.assert_allclose(loads.inputs.particles.force[0], f, atol=1e-12)
    np.testing.assert_allclose(loads.inputs.rigid.force[0], -f, atol=1e-12)
    np.testing.assert_allclose(loads.inputs.rigid.torque[0], np.cross(pos, -f), atol=1e-12)
    assert loads.contacts[0].force_n == pytest.approx(magnitude)


@pytest.mark.parametrize(("speed", "expected"), [(1.0, -2.0), (0.125, -1.0), (0.0, 0.0)])
def test_sliding_traction_common_point_conserves_moment_and_work(speed, expected):
    x = np.array((0.125, 0.0, 3 / 32))
    system, state = pad_state(x, (speed, 0.0, 0.0))
    with ph.CollisionWorkspace(system.model) as work:
        loads = system.sample(state, work)
    force = np.array((expected, 0.0, 4.0))
    np.testing.assert_allclose(loads.inputs.particles.force[0], force, atol=1e-12)
    np.testing.assert_allclose(loads.inputs.rigid.torque[0], np.cross(x, -force), atol=1e-12)
    assert loads.contacts[0].power_w == pytest.approx(expected * speed)
    (site,) = loads.contacts[0].sites
    np.testing.assert_allclose(site.application_point_first_m, x)
    np.testing.assert_allclose(site.application_point_second_m, x)
    np.testing.assert_allclose(site.force_on_second_n, force)
    assert abs(site.material.first_point[2] - 1 / 16) < 1e-12


def test_spinning_pad_uses_velocity_at_the_cloth_point():
    x = np.array((0.0, 0.0, 3 / 32))
    system, state = pad_state(x)
    state.rigid.angular_velocity[0, 1] = 1.0
    with ph.CollisionWorkspace(system.model) as work:
        loads = system.sample(state, work)
    force = np.array((0.75, 0.0, 4.0))
    np.testing.assert_allclose(loads.inputs.particles.force[0], force)
    expected = -0.75 * 3 / 32
    assert loads.contacts[0].power_w == pytest.approx(expected)
    assert state.rigid.angular_velocity[0] @ loads.inputs.rigid.torque[0] == pytest.approx(expected)


def test_point_far_behind_and_outside_false_planar_footprint_is_clear():
    for x in [(0.0, 0.0, -0.3), (0.65, 0.0, 0.0)]:
        system, state = pad_state(x)
        with ph.CollisionWorkspace(system.model) as work:
            loads = system.sample(state, work)
        assert not loads.contacts
        assert not np.any(loads.inputs.rigid.force)


def test_finite_pad_accepts_complete_triangle_patch_and_catches_crossing():
    system, s = pad_state((0.0, 0.0, 3 / 32))
    plate, point = system.model.shapes
    cloth = ph.CollisionShape(point.key, "triangle", (30, 31, 32))
    binds = {plate.key: system.bindings[plate.key], cloth.key: ph.ParticleAttachment((0, 1, 2))}
    surface = ph.CollisionLoadSystem(ph.CollisionModel((plate, cloth)), binds, system.layers)
    state = ph.CoupledState(
        ph.ParticleState(
            np.array([[-0.25, -0.25, 3 / 32], [0.25, -0.25, 3 / 32], [0.0, 0.25, 3 / 32]]),
            np.zeros((3, 3)),
        ),
        s.rigid,
    )
    models = (
        ph.ParticleModel(np.ones(3), np.zeros(3)),
        ph.RigidBodyModel(np.zeros(1), np.zeros((1, 3)), np.zeros(3)),
    )
    solvers = ph.SemiImplicitSolver(), ph.RigidSemiImplicitSolver()
    with ph.CollisionWorkspace(surface.model) as ws:
        loads = surface.sample(state, ws)
        assert len(loads.contacts[0].sites) == 3
        np.testing.assert_allclose(loads.inputs.particles.force[:, 2], [4 / 3] * 3)
        state.particles.velocity[:, 2] = -32.0
        with pytest.raises(ph.SolverFailure, match="core"):
            surface.step(state, ph.CoupledInputs.zeros(3, 1), models, solvers, 1 / 64, ws)
        assert surface.last_frame is None
