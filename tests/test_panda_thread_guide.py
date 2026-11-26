"""Virtual-work and passivity checks for the already-engaged thread guide."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def guide(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return import_module("_panda_thread_guide")


def _relative_jacobians():
    # A fixture with two opposed measuring handles and a 10 mm linear lever.
    jz, jt = np.zeros((2, 14))
    jz[[0, 7]] = [-.010, .010]
    jt[[1, 8]] = [-1., 1.]
    return jz, jt


@pytest.mark.parametrize("direction", [-1., 1.])
def test_matching_helix_has_only_opposing_turn_drag(guide, direction):
    jz, jt = _relative_jacobians()
    v = np.zeros(14)
    v[8] = direction
    v[7] = guide.LEAD_M_PER_RAD*direction/jz[7]
    tau, m = guide.thread_reaction(direction*guide.PITCH_M_PER_TURN, direction*2*np.pi, jz, jt, v)
    assert m["axial_force_n"] == pytest.approx(0., abs=1e-12)
    assert m["elastic_energy_j"] == pytest.approx(0., abs=1e-16)
    assert m["friction_torque_nm"]*direction < 0.
    np.testing.assert_allclose(tau[:7], -tau[7:], atol=1e-12)
    assert tau@v == pytest.approx(-m["dissipation_w"])


def test_energy_gradient_in_nonlinear_joint_coordinates(guide):
    q = np.linspace(-.2, .6, 14)
    # Nonlinear virtual FK so that differentiation exercises the coordinate
    # pullback, including equal and opposite forces on both moving bodies.
    def relative(x):
        return .006*np.sin(x[7]-x[0]), x[8]-x[1]+.4*np.sin(x[4])

    jz, jt = _relative_jacobians()
    jz[[0, 7]] = .006*np.cos(q[7]-q[0])*np.array([-1., 1.])
    jt[4] = .4*np.cos(q[4])
    tau, _ = guide.thread_reaction(*relative(q), jz, jt, np.zeros(14))
    def energy(x):
        depth, turn = relative(x)
        return .5*guide.STIFFNESS_N_PER_M*(depth-guide.LEAD_M_PER_RAD*turn)**2
    dq = 1e-6
    gradient = [(energy(q+dq*e)-energy(q-dq*e))/(2*dq) for e in np.eye(14)]
    np.testing.assert_allclose(tau, -np.array(gradient), atol=1e-10)


@pytest.mark.parametrize("nv", [1, 14, 20])
def test_mismatch_power_balance_and_reversal(guide, nv):
    rng = np.random.default_rng(321)
    for _ in range(10):
        jz = .02*rng.normal(size=nv)
        jt, v = rng.normal(size=(2, nv))
        depth, turn = .002, -.6
        tau, m = guide.thread_reaction(depth, turn, jz, jt, v)
        reversed_tau, reverse = guide.thread_reaction(-depth, -turn, jz, jt, -v)
        np.testing.assert_allclose(reversed_tau, -tau)
        assert reverse["dissipation_w"] == pytest.approx(m["dissipation_w"])
        assert m["dissipation_w"] > 0.
        assert tau@v == pytest.approx(-m["elastic_energy_rate_w"]-m["dissipation_w"])
        assert m["external_power_w"] == pytest.approx(m["axial_force_n"]*(jz@v)
                                                      +m["total_turn_torque_nm"]*(jt@v))


def test_free_part_wrench_and_board_reaction(guide):
    # In this test basis the board's first two joints induce 0.05 m/rad
    # axial travel and unit spin. The free part has six spatial velocities
    # after the two arms. The unconnected second arm receives no guide load.
    jz, jt = np.zeros((2, 20))
    jz[[0, 16]] = [-.05, 1.]
    jt[[1, 19]] = [-1., 1.]
    v = np.linspace(-.2, .3, 20)
    tau, metrics = guide.thread_reaction(.001, .7, jz, jt, v)
    force, moment = metrics["axial_force_n"], metrics["total_turn_torque_nm"]
    assert tau.shape == v.shape
    np.testing.assert_allclose(tau[[0, 1, 16, 19]], [-.05*force, -moment, force, moment])
    assert np.count_nonzero(tau[7:14]) == 0
    assert tau@v == pytest.approx(-metrics["elastic_energy_rate_w"]-metrics["dissipation_w"])


def test_common_motion_and_invalid_inputs(guide):
    jz, jt = _relative_jacobians()
    tau, m = guide.thread_reaction(.001, 0., jz, jt, np.ones(14))
    assert m["axial_force_n"] == pytest.approx(-12.)
    assert m["external_power_w"] == pytest.approx(0.)
    assert m["dissipation_w"] == pytest.approx(0.)
    np.testing.assert_allclose(tau[:7], -tau[7:])
    with pytest.raises(ValueError):
        guide.thread_reaction(.001, 0., jz[:-1], jt, np.zeros(14))
    with pytest.raises(ValueError):
        guide.thread_reaction(np.nan, 0., jz, jt, np.zeros(14))


@pytest.mark.parametrize("bad", [[], [[0., 1.]], [np.inf], [np.nan]])
def test_rejects_invalid_generalized_space(guide, bad):
    with pytest.raises(ValueError):
        guide.thread_reaction(0., 0., bad, bad, bad)


@pytest.mark.parametrize("free", [-1.527, -.05, -1e-9, 1e-9, .05, 1.527])
def test_implicit_drag_preserves_sign_and_satisfies_impulse(guide, free):
    # Explicit damping has a low-speed gain of 33 for this spindle and step.
    mobility, dt = 33000., .002
    rate, drag = guide.implicit_turn_drag(free, mobility, dt)
    assert 0. < rate/free <= 1.
    assert rate*drag < 0.
    assert abs(drag) <= guide.TURN_DRAG_NM
    assert rate-free-mobility*dt*drag == pytest.approx(0., abs=2e-14*abs(free))
    assert drag == pytest.approx(-guide.TURN_DRAG_NM*np.tanh(rate/guide.DRAG_TRANSITION_RAD_S))


@pytest.mark.parametrize("direction", [-1., 1.])
def test_implicit_drag_saturates_at_speed(guide, direction):
    mobility, dt, free = 33000., .002, direction*100.
    rate, drag = guide.implicit_turn_drag(free, mobility, dt)
    assert drag == pytest.approx(-direction*guide.TURN_DRAG_NM)
    assert rate == pytest.approx(free+dt*mobility*drag)


def test_implicit_drag_at_rest_and_zero_mobility(guide):
    assert guide.implicit_turn_drag(0., 33000., .002) == (0., 0.)
    for mobility, dt in ((0., .002), (33000., 0.)):
        rate, drag = guide.implicit_turn_drag(.01, mobility, dt)
        assert rate == .01
        assert drag == pytest.approx(-guide.TURN_DRAG_NM*np.tanh(.01/guide.DRAG_TRANSITION_RAD_S))
    with pytest.raises(ValueError):
        guide.implicit_turn_drag(.01, -1., .002)
