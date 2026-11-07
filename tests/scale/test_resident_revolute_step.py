"""C3 frozen single-revolute resident dynamics cases; authoring-only, unexecuted.

The analytic oracles do not call the product mass, FK, or projection routines.
Current q/qd units are rad and rad/s. All commands are applied through the public
DrivenRobotProgram/TyxWorldBatch path, never by setting an accepted device state.
"""

import json
from dataclasses import asdict, replace
from math import cos, pi

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends import tyx_world_batch as resident


H = 1 / 64
ANGLE_LIMIT_RAD = 4096.0


def _program(*, q=0.0, v=0.0, delay=0, kp=0.0, kd=0.0, cap=8.0,
             axis=(0, 0, 1), orientation=(0, 0, 0, 1), origin=(0, 0, 0),
             com=(1 / 4, 0, 0), inertia=(1 / 8, 1 / 8, 1 / 8), gravity=(0, 0, 0)):
    model = ph.RobotModel(
        np.array([-1]), ("revolute",), np.array([origin], dtype=float),
        np.array([orientation], dtype=float), np.array([axis], dtype=float),
        np.array([2.0]), np.array([com], dtype=float),
        np.array([inertia], dtype=float), np.array(gravity, dtype=float),
    )
    program = ph.DrivenRobotProgram(
        model, kp=[float(kp)], kd=[float(kd)], effort_limit=[float(cap)], delay_steps=delay,
    )
    initial = program.initial_state(ph.RobotState(np.array([q], dtype=float), np.array([v], dtype=float)))
    return program, initial


def _action(program, *, target=0.0, rate=0.0, ff=0.0, generalized=0.0,
            force=(0, 0, 0), torque=(0, 0, 0)):
    loads = ph.RobotInputs(
        np.array([generalized], dtype=float), np.array([force], dtype=float),
        np.array([torque], dtype=float),
    )
    return program.inputs(np.array([target], dtype=float), np.array([rate], dtype=float), feedforward=np.array([ff], dtype=float), loads=loads)


def _save(path, name, value):
    (path / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


@pytest.mark.tyx
def test_r01_revolute_effort_uses_axis_inertia_actual_accepted_state(tyx_mode, tmp_path):
    """Public RED: a valid one-R program exists before missing RevoluteRotor is used."""
    program, initial = _program()
    rotor = resident.RevoluteRotor(program, initial)
    with resident.TyxWorldBatch([(7, rotor)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: _action(program, ff=1 / 4)})
        assert batch.step(H) == {7: True}
        actual = batch.snapshot()[7]
        # I_axis = I_COM,zz + m*c_x**2 = 1/8 + 2/16 = 1/4 kg m2.
        assert actual.tick == 1
        assert actual.position == (1 / 4096,)
        assert actual.velocity == (1 / 64,)
        assert actual.applied_effort == (1 / 4,)
        summary = batch.summary()[7]
        assert summary["q_rad"] == actual.position[0]
        assert "q_m" not in summary
        assert summary["kinetic_energy_j"] == 1 / 32768
        kinematics = batch.kinematics([7])
        assert kinematics.keys == (7,) and kinematics.widths == (1,)
        np.testing.assert_array_equal(kinematics.values[0], [1 / 4096, 0, 0, 1 / 64, 0, 0])
        _save(tmp_path, f"r01-first-{tyx_mode}", {"checkpoint": asdict(actual), "summary": summary})


@pytest.mark.tyx
@pytest.mark.parametrize("delay", (0, 1))
@pytest.mark.parametrize("sign", (-1, 1))
def test_r01_forty_unreset_steps_keep_real_rotational_momentum_and_work(delay, sign, tyx_mode, tmp_path):
    program, initial = _program(delay=delay)
    with resident.TyxWorldBatch([(7, resident.RevoluteRotor(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: _action(program, ff=sign / 4)})
        trace, prior_v, prior_q = [], 0.0, 0.0
        for n in range(1, 41):
            accepted = batch.step(H)
            actual = batch.snapshot()[7]
            m = n - delay
            q, v = sign * m * (m + 1) / 8192, sign * m / 64
            effort = 0.0 if n <= delay else sign / 4
            dq = actual.position[0] - prior_q
            dv = actual.velocity[0] - prior_v
            delta_ke = (actual.velocity[0] ** 2 - prior_v ** 2) / 8
            balance = effort * dq - delta_ke - dv * dv / 8
            trace.append({"step": n, "acceptance": accepted, "actual": asdict(actual),
                          "oracle_q_rad": q, "oracle_qd_rad_s": v,
                          "discrete_work_residual_j": balance})
            _save(tmp_path, f"r01-window-delay{delay}-sign{sign}-{tyx_mode}", trace)
            assert accepted == {7: True}
            assert actual.position == (q,) and actual.velocity == (v,)
            assert actual.applied_effort == (effort,)
            assert abs((dv / 4) - H * effort) <= 1e-14
            assert abs(balance) <= 1e-14
            prior_q, prior_v = actual.position[0], actual.velocity[0]
        assert actual.tick == 40


@pytest.mark.tyx
@pytest.mark.parametrize("q", (0.0, pi / 2, -pi / 2, 2 * pi + 0.37, -6 * pi - 0.51))
def test_r02_gravity_and_com_force_use_current_accepted_angle(q, tyx_mode, tmp_path):
    program, initial = _program(q=q, gravity=(0, -2, 0))
    with resident.TyxWorldBatch([(7, resident.RevoluteRotor(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: _action(program, ff=1 / 4, force=(0, 2, 0))})
        entry = batch.snapshot()[7]
        # m*g contributes -cos(q); the COM force contributes +cos(q)/2.
        alpha = 1 - 2 * cos(entry.position[0])
        assert batch.step(H) == {7: True}
        actual = batch.snapshot()[7]
        np.testing.assert_allclose(actual.velocity, [H * alpha], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(actual.position, [entry.position[0] + H * H * alpha], rtol=0, atol=4e-6)
        assert actual.applied_effort == (1 / 4,)
        _save(tmp_path, f"r02-q{q:.6f}-{tyx_mode}", {"entry": asdict(entry), "actual": asdict(actual), "oracle_alpha_rad_s2": alpha})


@pytest.mark.tyx
@pytest.mark.parametrize("cycled", (False, True))
def test_r02_nonprincipal_axis_real_inertia_and_world_torque_covariance(cycled, tyx_mode, tmp_path):
    # a=(3/5,0,4/5), c=(0,1/4,0): I_axis=57/200+1/8=41/100.
    orientation = (0.5, 0.5, 0.5, 0.5) if cycled else (0, 0, 0, 1)
    torque = (4 / 5, 3 / 5, 0) if cycled else (3 / 5, 0, 4 / 5)
    program, initial = _program(
        q=0.37, axis=(3 / 5, 0, 4 / 5), com=(0, 1 / 4, 0),
        inertia=(1 / 8, 1 / 4, 3 / 8), orientation=orientation, origin=(0.5, -0.25, 0.75),
    )
    with resident.TyxWorldBatch([(7, resident.RevoluteRotor(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: _action(program, generalized=-59 / 100, torque=torque)})
        entry = batch.snapshot()[7]
        assert batch.step(H) == {7: True}
        actual = batch.snapshot()[7]
        # Axis torque is +1 Nm, generalized external -0.59 Nm -> alpha=1.
        np.testing.assert_allclose(actual.velocity, [H], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(actual.position, [entry.position[0] + H * H], rtol=0, atol=2e-6)
        assert actual.applied_effort == (0.0,)
        assert batch.summary()[7]["kinetic_energy_j"] == pytest.approx(0.5 * 0.41 * H * H, rel=3e-5, abs=1e-8)


@pytest.mark.tyx
@pytest.mark.parametrize("cap", (2.0, 1 / 8))
def test_r03_bounded_pd_and_all_undelayed_load_columns_match_independent_scalar_balance(cap, tyx_mode, tmp_path):
    program, initial = _program(q=1 / 8, v=-1 / 4, kp=8, kd=2, cap=cap)
    command = _action(program, target=1 / 4, rate=1 / 2, ff=1 / 2,
                      generalized=1 / 4, force=(0, 2, 0), torque=(0, 0, 1 / 8))
    h = 1 / 8
    external = 3 / 8 + cos(1 / 8) / 2
    free_v = -1 / 4 + h * external / (1 / 4)
    effort = float(np.clip((2.5 - 3 * free_v) / 2.5, -cap, cap))
    expected_v = free_v + h * effort / (1 / 4)
    with resident.TyxWorldBatch([(7, resident.RevoluteRotor(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: command})
        assert batch.snapshot()[7].control == (1 / 4, 1 / 2, 1 / 2, 1 / 4, 0, 2, 0, 0, 0, 1 / 8)
        assert batch.step(h) == {7: True}
        actual = batch.snapshot()[7]
        np.testing.assert_allclose(actual.applied_effort, [effort], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(actual.velocity, [expected_v], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(actual.position, [1 / 8 + h * expected_v], rtol=0, atol=2e-6)
        assert (abs(effort) < cap) == (cap == 2.0)


@pytest.mark.tyx
def test_r03_only_drive_triplet_is_delayed_current_cartesian_loads_are_not(tyx_mode, tmp_path):
    program, initial = _program(delay=1)
    with resident.TyxWorldBatch([(7, resident.RevoluteRotor(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        command = _action(program, ff=1 / 4, generalized=1 / 8, force=(0, 1 / 4, 0), torque=(0, 0, 1 / 16))
        batch.set_controls({7: command})
        held = batch.snapshot()[7]
        assert len(held.control) == 10
        assert batch.step(H) == {7: True}
        actual = batch.snapshot()[7]
        assert actual.applied_effort == (0.0,)
        assert actual.velocity == (H,) and actual.position == (H * H,)
        assert actual.history == (0.0, 0.0, 1 / 4)
        assert actual.control == held.control
        # Restore must restore all force/torque/control columns, not only the delayed triplet.
        batch.restore({7: held})
        assert batch.snapshot()[7] == held
        assert batch.step(H) == {7: True}
        assert batch.snapshot()[7] == actual


@pytest.mark.tyx
def test_r05_mixed_kind_routing_uses_batch_width_without_changing_old_only_layout(tyx_mode, tmp_path):
    program, initial = _program()
    slider_model = replace(program.model, joint_types=("prismatic",), joint_axis=np.array([[1.0, 0, 0]]))
    slider_program = ph.DrivenRobotProgram(slider_model, kp=[0.0], kd=[0.0], effort_limit=[8.0])
    slider_initial = slider_program.initial_state(ph.RobotState(np.zeros(1), np.zeros(1)))
    old_worlds = [(9, resident.PrismaticSlider(slider_program, slider_initial)),
                  (3, resident.FreeSphere(2, 1 / 16, gravity=(0, 0, 0)))]
    with resident.TyxWorldBatch(old_worlds, tyx_mode, cache_root=tmp_path / "old") as old:
        assert old._commands.shape[1] == 4 and old._props.shape[1] == 9
    worlds = [*old_worlds, (7, resident.RevoluteRotor(program, initial))]
    with resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "mixed") as batch:
        assert batch._commands.shape[1] == 10
        batch.set_controls({3: (2, 0, 0), 9: _action(slider_program, ff=2), 7: _action(program, ff=1 / 4)})
        assert batch.step(H) == {3: True, 7: True, 9: True}
        actual = batch.snapshot()
        assert actual[3].position == (H * H, 0.0, 0.0)
        assert actual[9].position == (H * H,)
        assert actual[7].position == (H * H,)
        summary = batch.summary()
        assert summary[9]["q_m"] == H * H and "q_rad" not in summary[9]
        assert summary[7]["q_rad"] == H * H and "q_m" not in summary[7]
        assert summary[7]["kinetic_energy_j"] == summary[9]["kinetic_energy_j"] / 8


@pytest.mark.tyx
@pytest.mark.parametrize("sign", (-1, 1))
def test_r06_angle_domain_initial_restore_and_candidate_are_one_transaction(sign, tyx_mode, tmp_path):
    program, initial = _program(q=sign * ANGLE_LIMIT_RAD, v=sign, delay=1)
    with resident.TyxWorldBatch([(7, resident.RevoluteRotor(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: _action(program, ff=1 / 4, force=(0, 1 / 4, 0))})
        before = batch.snapshot()[7]
        assert batch.step(H) == {7: False}
        assert batch.snapshot()[7] == before
        outside = sign * float(np.nextafter(np.float32(4096), np.float32(np.inf)))
        with pytest.raises(ValueError, match="angle|domain|4096"):
            batch.restore({7: replace(before, position=(outside,))})
        assert batch.snapshot()[7] == before
        batch.restore({7: replace(before, position=(0.0,), velocity=(0.0,))})
        assert batch.step(H) == {7: True}
        assert batch.snapshot()[7].history == (0.0, 0.0, 1 / 4)
    bad_program, bad_initial = _program(q=outside)
    with pytest.raises(ValueError, match="angle|domain|4096"):
        with resident.TyxWorldBatch([(7, resident.RevoluteRotor(bad_program, bad_initial))], tyx_mode, cache_root=tmp_path / "outside"):
            pass


@pytest.mark.tyx
def test_r06_failure_retry_preserves_control_queue_physics_and_neighbor(tyx_mode, tmp_path, monkeypatch):
    program, initial = _program(delay=1)
    worlds = [(7, resident.RevoluteRotor(program, initial)),
              (9, resident.FreeSphere(2, 1 / 16, gravity=(0, 0, 0)))]
    with resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: _action(program, ff=1 / 4, torque=(0, 0, 1 / 4)), 9: (2, 0, 0)})
        before = batch.snapshot()
        with monkeypatch.context() as patch:
            def fail_status():
                raise RuntimeError("injected before rotor acceptance")
            patch.setattr(batch, "_read_status", fail_status)
            with pytest.raises(RuntimeError, match="rotor acceptance"):
                batch.step(H)
        assert batch.snapshot() == before
        assert batch.step(H) == {7: True, 9: True}
        first = batch.snapshot()
        assert first[7].position == (H * H,) and first[7].history == (0.0, 0.0, 1 / 4)
        batch.restore({7: replace(first[7], tick=2**31 - 1)})
        blocked = batch.snapshot()[7]
        assert batch.step(H) == {7: False, 9: True}
        assert batch.snapshot()[7] == blocked
        assert batch.snapshot()[9].tick == 2
        with pytest.raises(ValueError):
            batch.restore({7: replace(blocked, control=blocked.control[:4])})
        with pytest.raises((ValueError, KeyError)):
            batch.set_controls({7: _action(program, ff=7), 99: (0, 0, 0)})
        assert batch.snapshot()[7] == blocked
        batch.reset([7])
        reset = batch.snapshot()[7]
        assert reset.tick == 0 and reset.history == (0.0, 0.0, 0.0)
        assert reset.control == (0.0,) * 10
        assert batch.snapshot()[9].tick == 2
        _save(tmp_path, f"r06-transaction-{tyx_mode}", {"before": {k: asdict(v) for k, v in before.items()}, "first": {k: asdict(v) for k, v in first.items()}, "reset": asdict(reset)})


@pytest.mark.parametrize("problem", ("prismatic", "two_revolutes", "delay_two", "limits"))
def test_r06_unsupported_topology_fails_before_runtime_allocation(problem, tmp_path):
    program, initial = _program(delay=2 if problem == "delay_two" else 0)
    if problem == "prismatic":
        model = replace(program.model, joint_types=("prismatic",))
        program = ph.DrivenRobotProgram(model, kp=[0.0], kd=[0.0], effort_limit=[8.0])
        initial = program.initial_state(ph.RobotState(np.zeros(1), np.zeros(1)))
    elif problem == "two_revolutes":
        one = program.model
        model = ph.RobotModel(np.array([-1, 0]), ("revolute", "revolute"),
            np.vstack((one.joint_position, [1, 0, 0])), np.tile(one.joint_orientation, (2, 1)),
            np.tile(one.joint_axis, (2, 1)), np.array([2.0, 1.0]), np.tile(one.com, (2, 1)),
            np.tile(one.inertia_diagonal, (2, 1)), np.zeros(3))
        program = ph.DrivenRobotProgram(model, kp=[0.0, 0.0], kd=[0.0, 0.0], effort_limit=[8.0, 8.0])
        initial = program.initial_state(ph.RobotState(np.zeros(2), np.zeros(2)))
    elif problem == "limits":
        limits = ph.JointLimits(np.array([-1.0]), np.array([1.0]), 2.0, 0.0)
        program = ph.DrivenRobotProgram(program.model, kp=[0.0], kd=[0.0], effort_limit=[8.0], limits=limits)
        initial = program.initial_state(ph.RobotState(np.zeros(1), np.zeros(1)))
    with pytest.raises(ValueError, match="revolute|single|one|delay|stop|limit"):
        with resident.TyxWorldBatch([(7, resident.RevoluteRotor(program, initial))], "interpret", cache_root=tmp_path):
            pass



