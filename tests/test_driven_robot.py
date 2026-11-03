"""Analytic stop impulses, actuator latency and rollback on a real fixed-base tree."""

import numpy as np
import pytest

import phymeo as ph


def rail():
    return ph.RobotModel(
        np.array([-1]), ("prismatic",), np.zeros((1, 3)), np.array([[0., 0, 0, 1]]),
        np.array([[1., 0, 0]]), np.array([2.]), np.zeros((1, 3)), np.ones((1, 3)), np.zeros(3),
    )


def program(delay=0, kp=0., kd=0., cap=100., *, limits=None, model=None):
    from phymeo.solvers.driven_robot import DrivenRobotProgram

    return DrivenRobotProgram(rail() if model is None else model, kp=np.array([kp]), kd=np.array([kd]),
                              effort_limit=np.array([cap]), delay_steps=delay, limits=limits)


def initial(p, q=0., v=0.):
    return p.initial_state(ph.RobotState(np.array([q], dtype=float), np.array([v], dtype=float)))


def scalar(row, name):
    return row[name][0]


def stops():
    return ph.JointLimits(np.array([-0.25]), np.array([0.25]), stiffness=2000., damping=5.)


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("toward,expected", [(0., 2.), (0.1, 2.5), (-0.1, 2.)])
def test_mirrored_stops_publish_load_impulse_and_real_motion(side, toward, expected):
    p = program(limits=stops())
    q, v, h = side * 0.251, side * toward, 0.01
    s = initial(p, q, v)
    with p.stepper(ph.StepSettings(substeps=1)) as rt:
        out = rt.step(s, p.inputs(np.array([q])), h)
    assert out.report.accepted
    info = out.report.substeps[0].diagnostics
    effort = -side * expected
    after = out.state.domains["robot"]
    assert scalar(info, "stop_violation") == pytest.approx(0.001)
    assert scalar(info, "stop_effort") == pytest.approx(effort)
    assert scalar(info, "stop_impulse") == pytest.approx(effort * h)
    assert scalar(info, "actuator_effort") == 0
    assert 2 * (after.v[0] - v) == pytest.approx(effort * h)
    assert after.q[0] == pytest.approx(q + h * (v + h * effort / 2))


def test_stop_unloads_on_the_next_accepted_physical_substep():
    p = program(limits=stops())
    with p.stepper(ph.StepSettings(substeps=2, record_substeps=True)) as rt:
        out = rt.step(initial(p, -0.2505, 0.1), p.inputs(np.zeros(1)), 0.02)
    assert out.report.accepted
    assert [scalar(r.diagnostics, "stop_effort") for r in out.report.substeps] == pytest.approx([1, 0])
    assert [s.domains["robot"].q[0] for s in out.substep_states] == pytest.approx([-0.24945, -0.2484])
    assert out.state.domains["robot"].v[0] == pytest.approx(0.105)


def test_unsaturated_implicit_command_and_unclamped_external_load():
    p = program(kp=8., kd=2., cap=3.)
    loads = ph.RobotInputs.zeros(1)
    loads.force[0, 0] = 1
    command = p.inputs(np.array([0.25]), np.array([0.5]), feedforward=np.array([0.5]), loads=loads)
    with p.stepper(ph.StepSettings(substeps=1)) as rt:
        out = rt.step(initial(p), command, 0.125)
    info = out.report.substeps[0].diagnostics
    assert out.report.accepted
    # tau = 3.5 - 3 v1; 2 v1 = h (tau + 1).
    assert scalar(info, "actuator_effort") == pytest.approx(53 / 19)
    assert scalar(info, "actuator_impulse") == pytest.approx(53 / 152)
    assert out.state.domains["robot"].v[0] == pytest.approx(9 / 38)
    assert out.state.domains["robot"].q[0] == pytest.approx(9 / 304)
    assert scalar(info, "applied") == pytest.approx(72 / 19)


def test_stop_and_external_load_do_not_consume_motor_saturation_budget():
    p = program(kp=100., cap=4., limits=stops())
    loads = ph.RobotInputs.zeros(1)
    loads.effort[:] = 8
    with p.stepper(ph.StepSettings(substeps=1)) as rt:
        out = rt.step(initial(p, -0.251), p.inputs(np.ones(1), loads=loads), 0.01)
    info = out.report.substeps[0].diagnostics
    assert [scalar(info, s) for s in ("actuator_effort", "stop_effort", "applied")] == pytest.approx([4, 2, 14])
    assert out.state.domains["robot"].v[0] == pytest.approx(0.07)


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_sample_delay_uses_current_feedback_and_delays_all_command_channels(delay):
    p = program(delay, kp=2., kd=0.75)
    h, q, v = 0.125, 0.1, -0.2
    state = initial(p, q, v)
    preceding = [(q, 0., 0.)] * delay
    commands = [(0.5, 0.25, 2.), (-0.75, -0.5, -3.), (0.25, -1., 1.), (0., 0., 0.)]
    with p.stepper(ph.StepSettings(substeps=1)) as rt:
        for i, cmd in enumerate(commands):
            target, speed, ff = (preceding + commands)[i]
            inputs = p.inputs(np.array([cmd[0]]), np.array([cmd[1]]), feedforward=np.array([cmd[2]]))
            out = rt.step(state, inputs, h)
            assert out.report.accepted
            info = out.report.substeps[0].diagnostics
            expected_v = (2 * v + h * (2 * (target - q) + 0.75 * speed + ff)) / (2 + h * (0.75 + 2 * h))
            assert scalar(info, "actuator_effort") == pytest.approx(2 * (expected_v - v) / h)
            assert [scalar(info, field) for field in ("applied_target", "applied_target_velocity", "applied_feedforward")] == pytest.approx([target, speed, ff])
            v, q = expected_v, q + h * expected_v
            state = out.state
            assert state.domains["robot"].v[0] == pytest.approx(v)
            assert state.domains["robot"].q[0] == pytest.approx(q)


def test_step_in_outer_frame_and_rejected_retry_preserve_queue():
    p = program(2)
    start = initial(p)
    command = p.inputs(np.zeros(1), feedforward=np.array([2.]))
    cfg = ph.StepSettings(substeps=4, record_substeps=True)

    class RejectOnce:
        calls = 0
        def problem_signature(self):
            return p.problem_signature()
        def verify(self, scene, before, inputs, h, candidate):
            self.calls += 1
            return ph.VerificationReport(self.calls != 3, reason="third substep injection")

    with p.stepper(cfg, verifier=RejectOnce()) as rt:
        rejected = rt.step(start, command, 0.25)
        assert not rejected.report.accepted and rejected.report.completed_substeps == 2
        assert rejected.substep_states == ()
        np.testing.assert_array_equal(rejected.state.history["robot/drive_queue"], np.zeros((2, 3, 1)))
        np.testing.assert_array_equal(rejected.state.domains["robot"].v, [0])
        retry = rt.step(rejected.state, command, 0.25)
    assert retry.report.accepted
    assert [scalar(r.diagnostics, "actuator_effort") for r in retry.report.substeps] == [0, 0, 2, 2]
    assert [s.domains["robot"].v[0] for s in retry.substep_states] == [0, 0, 1 / 16, 1 / 8]
    assert retry.state.domains["robot"].q[0] == pytest.approx(3 / 256)
    np.testing.assert_array_equal(start.history["robot/drive_queue"], np.zeros((2, 3, 1)))
    with p.stepper(ph.StepSettings(substeps=1)) as rt:
        state = start
        for _ in range(4):
            state = rt.step(state, command, 1 / 16).state
    for field in ("q", "v"):
        np.testing.assert_array_equal(getattr(state.domains["robot"], field), getattr(retry.state.domains["robot"], field))
    np.testing.assert_array_equal(state.history["robot/drive_queue"], retry.state.history["robot/drive_queue"])
