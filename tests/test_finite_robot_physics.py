"""Independent finite-contact impulse, material, energy and rotor oracles."""

import math

import numpy as np
import pytest
from test_finite_patch_geometry import model

import phymeo as ph


def test_current_configuration_terminal_equation():
    assert hasattr(ph, "FiniteContactTerminalProgram"), "missing finite terminal adapter"
    robot = model(True)
    scene = ph.SceneModel((ph.DomainBinding("robot", "robot", robot),))
    before = ph.SceneState({"robot": ph.RobotState(np.array([0.35]), np.array([0.2]))})
    registry = ph.StateRegistry()
    from phymeo.runtime.robot_family import robot_family

    registry.register(robot_family())
    # Two identical current points aggregate before one mass solve. The
    # independent cross product is Rz(q)*(1,0,0) x (0,.1,0).
    local = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    request = ph.PointVelocityImpulses(
        "robot", np.array([0, 0]), np.array([[0.0, 0.1, 0.0], [0.0, 0.1, 0.0]]), local
    )
    ledger = ph.FiniteContactTerminalLedger(
        scene, before, (request,), ph.StepSettings(tolerance=1e-8), registry=registry
    )
    delta = 0.2 * math.cos(0.35)
    after = ph.SceneState(
        {"robot": ph.RobotState(before.domains["robot"].q.copy(), np.array([0.2 + delta]))}
    )
    assert ledger.require(after)["max_residual"] < 1e-8
    with pytest.raises(ValueError, match="all-prismatic"):
        ph.TerminalImpulseLedger(
            scene, before, (request,), ph.StepSettings(tolerance=1e-8), registry=registry
        )


def test_terminal_wrong_time_mass_units_and_storage_rejected():
    assert hasattr(ph, "FiniteContactTerminalLedger"), "missing finite terminal ledger"
    robot = model(True)
    scene = ph.SceneModel((ph.DomainBinding("robot", "robot", robot),))
    before = ph.SceneState({"robot": ph.RobotState(np.array([0.75]), np.array([0.0]))})
    request = ph.PointVelocityImpulses(
        "robot", np.array([0]), np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])
    )
    settings = ph.StepSettings(tolerance=1e-8)
    ledger = ph.FiniteContactTerminalLedger(scene, before, (request,), settings)
    for q, v in ((0.75, 1.0), (0.75, 0.5 * math.cos(0.75)), (0.75000001, math.cos(0.75))):
        with pytest.raises(ph.SolverFailure):
            ledger.require(ph.SceneState({"robot": ph.RobotState(np.array([q]), np.array([v]))}))
    angular = ph.FiniteContactTerminalLedger(
        scene, before, (request,), settings, reference_angular_speed=0.01
    )
    with pytest.raises(ph.SolverFailure):
        angular.require(
            ph.SceneState(
                {"robot": ph.RobotState(np.array([0.75]), np.array([math.cos(0.75) + 1e-9]))}
            )
        )
    pre = ph.SceneState({"robot": ph.RobotState(np.array([0.75]), np.array([0.0]))})
    device = ph.FiniteContactTerminalLedger(
        scene, pre, (), ph.StepSettings(tolerance=2e-5), ph.BackendContext("tyx:interpret")
    )
    with pytest.raises(ph.SolverFailure, match="FP32"):
        device.require(
            ph.SceneState(
                {"robot": ph.RobotState(np.array([0.75]), np.array([np.nextafter(0.0, 1.0)]))}
            )
        )


def test_finite_material_impact():
    from test_finite_robot_contact import build, step

    case = build(chain=True)
    r = step(case)
    d = r.report.substeps[0].diagnostics["dynamic_robot_finite_contact"]
    np.testing.assert_allclose(d["held_impulses"], [3 / 7, 0.0], atol=1e-8, rtol=0)
    np.testing.assert_allclose(d["terminal_impulses"], [1 / 3, 0.0], atol=1e-8, rtol=0)
    np.testing.assert_allclose(r.state.domains["robot"].q, [-3 / 56], atol=1e-8, rtol=0)
    np.testing.assert_allclose(r.state.domains["robot"].v, [-8 / 21], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        r.state.domains["nodes"].position,
        [[0.25, 0.125 - 3 / 56, 0.0], [0.25, 0.125 + 51 / 56, 0.0]],
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(
        r.state.domains["nodes"].velocity,
        [[1.0, -8 / 21, 0.0], [1.0, -6 / 7, 0.0]],
        atol=1e-8,
        rtol=0,
    )
    assert (
        abs(2 * r.state.domains["robot"].v[0] + sum(r.state.domains["nodes"].velocity[:, 1]) + 2)
        < 1e-8
    )
    initial = 2 + sum(case[1].domains["nodes"].velocity[:, 0] ** 2) / 2
    final = r.state.domains["robot"].v[0] ** 2 + np.sum(r.state.domains["nodes"].velocity ** 2) / 2
    final += (np.linalg.norm(np.diff(r.state.domains["nodes"].position, axis=0)) - 1) ** 2 / (
        2 / 16
    )
    assert final < initial


def rotor_expected(X=1.5):
    alpha = 0.25
    theta = math.atan(0.1)
    c = 10 / math.sqrt(101)
    V = X / 10 + 0.125 * (1 / c - 1)
    ell = X / c + 0.125 / 10
    u = (-(X - 1) + 10 * V) / math.sqrt(101) - ell * theta
    eta = -u / (1 + ell * ell)
    R = np.array(
        [
            [math.cos(alpha), -math.sin(alpha), 0.0],
            [math.sin(alpha), math.cos(alpha), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return theta, V, ell, u, eta, R


def test_rotor_analytic_response_and_energy():
    from test_finite_robot_contact import build, step

    case = build(rotor=True)
    r = step(case, 1.0)
    theta, V, ell, u, eta, R = rotor_expected()
    d = r.report.substeps[0].diagnostics["dynamic_robot_finite_contact"]
    np.testing.assert_allclose(d["held_impulses"], [1 / 16], atol=1e-8, rtol=0)
    np.testing.assert_allclose(d["terminal_impulses"], [eta], atol=1e-8, rtol=0)
    np.testing.assert_allclose(r.state.domains["robot"].q, [0.25 + theta], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        r.state.domains["nodes"].position, [R @ np.array([1.5, 0.125 + V, 0.0])], atol=1e-8, rtol=0
    )
    n = R @ np.array([-1 / math.sqrt(101), 10 / math.sqrt(101), 0.0])
    np.testing.assert_allclose(
        r.state.domains["nodes"].velocity,
        [R @ np.array([0.5, V, 0.0]) + eta * n],
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(r.state.domains["robot"].v, [theta - ell * eta], atol=1e-8, rtol=0)
    before = case[1]

    def K(state):
        return (
            np.sum(state.domains["nodes"].velocity ** 2) + np.sum(state.domains["robot"].v ** 2)
        ) / 2

    expected = (V - theta) / 16 - 1 / 256 - u * u / (2 * (1 + ell * ell))
    assert abs((K(r.state) - K(before)) - expected) < 1e-8 and expected < 0

    def L(state):
        return (
            state.domains["robot"].v[0]
            + np.cross(state.domains["nodes"].position[0], state.domains["nodes"].velocity[0])[2]
        )

    assert abs(L(r.state) - L(before)) < 1e-8
    assert d["ccd"][0].status == "certified"
    assert not np.array_equal(d["held_normal"], d["terminal_normal"])


def test_ccd_safe_energy_gain_rejected():
    from test_finite_robot_contact import build, step

    theta, V, ell, u, eta, R = rotor_expected()
    p = (V - theta) / 2
    gain = p * (V - theta) - p * p - u * u / (2 * (1 + ell * ell))
    assert gain > 2e-4
    r = step(build(rotor=True, gain=True), 1.0, False)
    assert "energy" in r.report.reason


def rotor_window(mode, count):
    from test_finite_robot_contact import build

    case = build(mode, rotor=True, X=1.0)
    scene, state, inputs, program, settings, context = case
    h = 1 / count
    # Independent scalar closed-face oracle: n(q).(x+h*v+h*p*n0)-rho=0,
    # q=q0+h*(omega-lever0*p), solved by deterministic bisection.
    x = state.domains["nodes"].position[0].copy()
    v = state.domains["nodes"].velocity[0].copy()
    angle = float(state.domains["robot"].q[0])
    omega = float(state.domains["robot"].v[0])
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        for step_index in range(count):
            n0 = np.array([-math.sin(angle), math.cos(angle), 0.0])
            lever0 = x[0] * math.cos(angle) + x[1] * math.sin(angle)

            def evaluate(p):
                a = angle + h * (omega - lever0 * p)
                pos = x + h * (v + p * n0)
                n = np.array([-math.sin(a), math.cos(a), 0.0])
                return float(n @ pos - 0.125), a, pos, n

            if evaluate(0.0)[0] < 0:
                lo, hi = 0.0, 1.0
                assert evaluate(hi)[0] > 0
                for _ in range(80):
                    mid = (lo + hi) / 2
                    if evaluate(mid)[0] < 0:
                        lo = mid
                    else:
                        hi = mid
                p = (lo + hi) / 2
            else:
                p = 0.0
            gap, a, pos, n = evaluate(p)
            vm = v + p * n0
            om = omega - lever0 * p
            ell = pos[0] * math.cos(a) + pos[1] * math.sin(a)
            u = float(n @ vm - ell * om)
            eta = max(0.0, -u / (1 + ell * ell)) if abs(gap) < 1e-8 else 0.0
            v = vm + eta * n
            omega = om - ell * eta
            x = pos
            angle = a
            with np.errstate(all="raise"):
                r = runtime.step(state, inputs, h)
            assert r.report.accepted, (count, step_index, r.report.reason)
            tol = 1e-8 if mode == "numpy" else 2e-5
            np.testing.assert_allclose(r.state.domains["nodes"].position, [x], atol=tol, rtol=0)
            np.testing.assert_allclose(r.state.domains["nodes"].velocity, [v], atol=tol, rtol=0)
            np.testing.assert_allclose(r.state.domains["robot"].q, [angle], atol=tol, rtol=0)
            np.testing.assert_allclose(r.state.domains["robot"].v, [omega], atol=tol, rtol=0)
            d = r.report.substeps[0].diagnostics["dynamic_robot_finite_contact"]
            assert all(c.status == "certified" for c in d["ccd"])
            state = r.state
    return state


def test_rotor_explicit_step_refinement():
    for count in (1, 2, 4):
        a = rotor_window("numpy", count)
        b = rotor_window("numpy", count)
        np.testing.assert_array_equal(a.domains["nodes"].position, b.domains["nodes"].position)
        np.testing.assert_array_equal(a.domains["robot"].q, b.domains["robot"].q)
