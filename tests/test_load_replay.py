"""Independent impulse/time bookkeeping and complete replay ownership."""

import importlib

import numpy as np
import pytest
from test_scene_stepper import ScalarProgram, scalar_case

from phymeo import (
    BackendContext,
    DomainBinding,
    FreeSceneProgram,
    ParticleModel,
    ParticleState,
    SceneInputs,
    SceneModel,
    SceneState,
    SolverFailure,
    StepInputs,
    StepSettings,
)


def api():
    return importlib.import_module("phymeo.runtime.load_replay")


def simple(*, force=2.0, dt=0.2, channels=None, context=None):
    model = ParticleModel(np.array([0.5]), np.zeros(3))
    scene = SceneModel((DomainBinding("body", "particles", model),))
    state = SceneState({"body": ParticleState(np.zeros((1, 3)), np.array([[0.2, 0, 0]]))})
    inputs = SceneInputs({"body": StepInputs(np.array([[force, 0, 0]]))})
    replay = api().SceneLoadReplay(
        scene,
        FreeSceneProgram(),
        state,
        inputs,
        dt,
        channels={"body": ("force",)} if channels is None else channels,
        settings=StepSettings(substeps=1),
        context=context,
    )
    return replay, state, inputs


def pulse(x):
    return {"body": {"force": np.array([[x, 0.0, 0.0]])}}


def test_public_api_and_single_integrated_load_without_double_kick():
    import phymeo

    assert phymeo.SceneLoadReplay is api().SceneLoadReplay
    assert phymeo.ReplayTrial is api().ReplayTrial
    replay, state, inputs = simple()
    with replay:
        trial = replay.evaluate(pulse(1.0))
        assert trial.result.report.accepted
        assert trial.duration == 0.2
        np.testing.assert_allclose(
            trial.result.state.domains["body"].velocity, [[0.9, 0, 0]], atol=1e-14
        )
        np.testing.assert_allclose(
            trial.result.state.domains["body"].position, [[0.18, 0, 0]], atol=1e-14
        )
        np.testing.assert_allclose(trial.effective_inputs.domains["body"].force, [[7, 0, 0]])
        np.testing.assert_allclose(trial.integrated_loads["body"]["force"], [[1, 0, 0]])
    np.testing.assert_array_equal(state.domains["body"].position, np.zeros((1, 3)))
    np.testing.assert_array_equal(inputs.domains["body"].force, [[2, 0, 0]])
    with pytest.raises(RuntimeError, match="closed"):
        replay.evaluate({})
    replay.close()


def test_p_zero_p_uses_original_time_state_and_inputs():
    with simple()[0] as replay:
        first = replay.evaluate(pulse(1.0))
        free = replay.evaluate({})
        again = replay.evaluate(pulse(1.0))
        np.testing.assert_allclose(
            free.result.state.domains["body"].position, [[0.08, 0, 0]], atol=1e-14
        )
        np.testing.assert_array_equal(
            again.result.state.domains["body"].position, first.result.state.domains["body"].position
        )
        np.testing.assert_array_equal(replay.snapshot().domains["body"].position, np.zeros((1, 3)))
        np.testing.assert_array_equal(free.integrated_loads["body"]["force"], np.zeros((1, 3)))


def test_external_inputs_and_returned_states_ledgers_are_owned():
    replay, original, inputs = simple()
    original.domains["body"].position[:] = 20
    inputs.domains["body"].force[:] = 200
    request = pulse(1.0)
    with replay:
        trial = replay.evaluate(request)
        request["body"]["force"][:] = 30
        trial.result.state.domains["body"].position[:] = 40
        trial.effective_inputs.domains["body"].force[:] = 500
        trial.integrated_loads["body"]["force"].flags.writeable = True
        trial.integrated_loads["body"]["force"][:] = 90
        replay.snapshot().domains["body"].position[:] = 70
        again = replay.evaluate(pulse(1.0))
        np.testing.assert_allclose(again.result.state.domains["body"].position, [[0.18, 0, 0]])
        np.testing.assert_allclose(again.effective_inputs.domains["body"].force, [[7, 0, 0]])
        np.testing.assert_allclose(again.integrated_loads["body"]["force"], [[1, 0, 0]])


@pytest.mark.parametrize(
    "channels",
    [[], {"other": ("force",)}, {"body": "force"}, {"body": ("q",)}, {"body": ("force", "force")}],
)
def test_channels_must_explicitly_name_registered_input_fields(channels):
    with pytest.raises((ValueError, TypeError)):
        simple(channels=channels)


@pytest.mark.parametrize(
    "loads",
    [
        [],
        {"other": {"force": np.zeros((1, 3))}},
        {"body": {"effort": np.zeros(1)}},
        {"body": np.zeros((1, 3))},
        {"body": {"force": np.zeros((1, 3), np.float32)}},
        {"body": {"force": np.zeros((2, 3))}},
        {"body": {"force": np.full((1, 3), np.nan)}},
        {"body": {"force": np.ones((1, 3), int)}},
    ],
)
def test_invalid_loads_are_rejected_before_trial_and_do_not_poison_retry(loads):
    with simple()[0] as replay:
        with pytest.raises((ValueError, TypeError)):
            replay.evaluate(loads)
        np.testing.assert_allclose(
            replay.evaluate({}).result.state.domains["body"].position, [[0.08, 0, 0]]
        )


@pytest.mark.parametrize("dt", [0, -0.1, np.nan, np.inf, True])
def test_invalid_duration(dt):
    with pytest.raises((ValueError, TypeError)):
        simple(dt=dt)


def test_fp32_storage_is_explicitly_unsupported():
    with pytest.raises((ValueError, TypeError), match="64|precision"):
        simple(context=BackendContext("numpy", np.float32))


def test_small_integral_lost_in_large_background_input_is_refused():
    with simple(force=1e16, dt=1.0)[0] as replay:
        with pytest.raises(SolverFailure, match="represent|integral"):
            replay.evaluate(pulse(1.0))
        np.testing.assert_array_equal(replay.snapshot().domains["body"].position, np.zeros((1, 3)))


@pytest.mark.parametrize("force,dt,amount", [(1e308, 1.0, 1e308), (0.0, 1e300, 1e-300)])
def test_nonrepresentable_rate_arithmetic_is_refused(force, dt, amount):
    with simple(force=force, dt=dt)[0] as replay:
        with pytest.raises(SolverFailure):
            replay.evaluate(pulse(amount))


def scalar_replay(program=None):
    scene, state, inputs, registry = scalar_case()
    program = ScalarProgram() if program is None else program
    replay = api().SceneLoadReplay(
        scene,
        program,
        state,
        inputs,
        0.2,
        channels={"osc": ("tau",)},
        registry=registry,
        settings=StepSettings(
            substeps=2, position_iterations=2, velocity_iterations=2, record_substeps=True
        ),
    )
    return replay, program, state, inputs


def test_registered_extra_state_history_and_multi_substep_integral():
    replay, program, initial, inputs = scalar_replay()
    with replay:
        trial = replay.evaluate({"osc": {"tau": np.array([0.4])}})
        assert trial.result.report.accepted and len(trial.result.substep_states) == 2
        value = trial.result.state.domains["osc"]
        # tau=2+0.4/0.2=4; m=2; two h=.1 kicks, not two copies of the whole .4 impulse.
        np.testing.assert_allclose(value.v, [0.4], atol=1e-14)
        np.testing.assert_allclose(value.q, [1.06], atol=1e-14)
        np.testing.assert_array_equal(value.plastic, [3])
        np.testing.assert_array_equal(trial.result.state.history["count"], [2])
        trial.result.state.history["count"][:] = 90
        trial.result.state.domains["osc"].plastic[:] = 99
        again = replay.evaluate({"osc": {"tau": np.array([0.4])}})
        np.testing.assert_array_equal(again.result.state.history["count"], [2])
        np.testing.assert_array_equal(again.result.state.domains["osc"].plastic, [3])
    assert program.created == program.closed
    np.testing.assert_array_equal(initial.history["count"], [0])
    np.testing.assert_array_equal(inputs.domains["osc"].tau, [2])


@pytest.mark.parametrize(
    "fault",
    ["predict", "position_budget", "velocity_budget", "close", "bad_history", "nan_candidate"],
)
def test_late_failure_restores_all_original_fields_and_retry(fault):
    replay, program, _, _ = scalar_replay(ScalarProgram(fault))
    request = {"osc": {"tau": np.array([0.4])}}
    with replay:
        failed = replay.evaluate(request)
        assert not failed.result.report.accepted
        assert failed.result.substep_states == ()
        np.testing.assert_array_equal(failed.result.state.domains["osc"].q, [1])
        np.testing.assert_array_equal(failed.result.state.domains["osc"].plastic, [3])
        np.testing.assert_array_equal(failed.result.state.history["count"], [0])
        assert program.created == program.closed
        program.fault = None
        recovered = replay.evaluate(request)
        with scalar_replay()[0] as fresh:
            expected = fresh.evaluate(request)
        np.testing.assert_array_equal(
            recovered.result.state.domains["osc"].q, expected.result.state.domains["osc"].q
        )
        np.testing.assert_array_equal(
            recovered.result.state.history["count"], expected.result.state.history["count"]
        )


def test_programming_error_releases_resources_and_can_retry():
    replay, program, _, _ = scalar_replay(ScalarProgram("program_error"))
    with replay:
        with pytest.raises(RuntimeError, match="program bug"):
            replay.evaluate({})
        assert program.created == program.closed
        program.fault = None
        assert replay.evaluate({}).result.report.accepted


def test_changed_binding_cannot_be_replayed():
    replay, program, _, _ = scalar_replay()
    with replay:
        program.signature["law"] = "changed"
        with pytest.raises(ValueError, match="changed"):
            replay.evaluate({})
        assert program.created == 0


@pytest.mark.parametrize("h", [0.01, 0.005])
@pytest.mark.parametrize("mu", [0.6, 0.2])
def test_prescribed_plane_friction_has_no_static_drift_and_analytic_sliding(h, mu):
    from phymeo import solve_friction_disk

    model = ParticleModel(np.ones(1), np.array([0.0, -9.81, 0]))
    scene = SceneModel((DomainBinding("body", "particles", model),))
    current = SceneState({"body": ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))})
    inputs = SceneInputs({"body": StepInputs(np.array([[-20.0, 0, 0]]))})
    total = np.zeros(3)
    n = round(2.0 / h)
    for _ in range(n):
        free = current.domains["body"].velocity[0] + np.array([-20.0, -9.81, 0]) * h
        normal = max(0.0, -free[0])
        friction = solve_friction_disk(np.eye(2), free[1:], normal, mu_static=mu, mu_dynamic=mu)
        impulse = np.array([[normal, *friction.impulse]])
        with api().SceneLoadReplay(
            scene, FreeSceneProgram(), current, inputs, h, channels={"body": ("force",)}
        ) as replay:
            trial = replay.evaluate({"body": {"force": impulse}})
            assert trial.result.report.accepted
            current = trial.result.state
            total += trial.integrated_loads["body"]["force"][0]
    state = current.domains["body"]
    acceleration = max(0.0, 9.81 - mu * 20)
    expected_velocity = -acceleration * n * h
    expected_y = -acceleration * h**2 * n * (n + 1) / 2
    np.testing.assert_allclose(state.velocity, [[0, expected_velocity, 0]], atol=3e-11, rtol=0)
    np.testing.assert_allclose(state.position, [[0, expected_y, 0]], atol=3e-11, rtol=0)
    np.testing.assert_allclose(total + np.array([-20, -9.81, 0]) * 2, state.velocity[0], atol=3e-11)
    if mu == 0.6:
        assert np.linalg.norm(state.position) < 3e-11  # no g*T*h static creep
    else:
        # The remaining sliding displacement error is the stated Euler truncation.
        assert abs(state.position[0, 1] + acceleration * 2**2 / 2) == pytest.approx(
            acceleration * h, abs=3e-11
        )


def mixed_replay(mode="numpy", material=False):
    from test_robot_kinematics import robot_model

    from phymeo import (
        ArticulatedCouplingProgram,
        ClothParticipant,
        DistanceConstraints,
        RobotInputs,
        RobotParticipant,
        RobotSphereAttachments,
        RobotState,
        TyxArticulatedCouplingProgram,
        TyxClothParticipant,
        TyxRobotParticipant,
    )

    robot = robot_model(gravity=np.zeros(3))
    cloth = ParticleModel(np.array([0.0 if material else 1.0, 1.0]), np.zeros(3))
    constraints = (
        DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.array([0.001])) if material else None
    )
    scene = SceneModel(
        (DomainBinding("robot", "robot", robot), DomainBinding("cloth", "particles", cloth))
    )
    state = SceneState(
        {
            "robot": RobotState(np.array([0.2, -0.3]), np.zeros(2)),
            "cloth": ParticleState(np.array([[0.0, 0, 0], [1.0, 0, 0]]), np.zeros((2, 3))),
        }
    )
    inputs = SceneInputs({"robot": RobotInputs.zeros(2), "cloth": StepInputs.zeros(2)})
    attachments = RobotSphereAttachments(np.empty(0, int), np.empty((0, 3)), np.empty(0))
    if mode == "numpy":
        participants = {
            "robot": RobotParticipant(robot),
            "cloth": ClothParticipant(cloth, constraints),
        }
        program = ArticulatedCouplingProgram(scene, participants, attachments, [])
    else:
        participants = {
            "robot": TyxRobotParticipant(robot, mode),
            "cloth": TyxClothParticipant(cloth, constraints, mode),
        }
        program = TyxArticulatedCouplingProgram(scene, participants, attachments, [])
    return api().SceneLoadReplay(
        scene,
        program,
        state,
        inputs,
        0.05,
        channels={"robot": ("effort",), "cloth": ("force",)},
        registry=program.registry,
        context=BackendContext(program.capabilities.backend),
        settings=StepSettings(substeps=1, tolerance=2e-6 if mode != "numpy" else 1e-11),
    )


def test_nonlinear_robot_mass_and_two_domains_replay_from_original_configuration():
    from test_robot_dynamics import planar_mass

    request = {
        "robot": {"effort": np.array([0.02, -0.01])},
        "cloth": {"force": np.array([[0.0, 0, 0], [0.1, 0, 0]])},
    }
    expected_v = np.linalg.solve(planar_mass(np.array([0.2, -0.3])), [0.02, -0.01])
    with mixed_replay() as replay:
        first = replay.evaluate(request)
        free = replay.evaluate({})
        again = replay.evaluate(request)
        assert all(t.result.report.accepted for t in (first, free, again))
        np.testing.assert_allclose(first.result.state.domains["robot"].v, expected_v, atol=1e-14)
        np.testing.assert_allclose(
            first.result.state.domains["robot"].q,
            np.array([0.2, -0.3]) + 0.05 * expected_v,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            first.result.state.domains["cloth"].position, [[0, 0, 0], [1.005, 0, 0]], atol=1e-14
        )
        np.testing.assert_array_equal(free.result.state.domains["robot"].q, [0.2, -0.3])
        np.testing.assert_array_equal(
            again.result.state.domains["robot"].q, first.result.state.domains["robot"].q
        )


def test_actual_xpbd_hidden_multipliers_are_rebuilt_for_each_trial():
    request = {"cloth": {"force": np.array([[0.0, 0, 0], [0.1, 0, 0]])}}
    with mixed_replay(material=True) as replay:
        first = replay.evaluate(request)
        free = replay.evaluate({})
        again = replay.evaluate(request)
        assert all(t.result.report.accepted for t in (first, free, again))
        alpha = 0.001 / 0.05**2
        displacement = 0.005 * alpha / (1 + alpha)
        np.testing.assert_allclose(
            first.result.state.domains["cloth"].position[1], [1 + displacement, 0, 0], atol=1e-14
        )
        np.testing.assert_allclose(
            first.result.state.domains["cloth"].velocity[1], [displacement / 0.05, 0, 0], atol=1e-14
        )
        np.testing.assert_array_equal(
            free.result.state.domains["cloth"].position, [[0, 0, 0], [1, 0, 0]]
        )
        np.testing.assert_array_equal(
            again.result.state.domains["cloth"].position,
            first.result.state.domains["cloth"].position,
        )
        for trial in (first, again):
            multipliers = trial.result.report.substeps[0].diagnostics["participants"]["cloth"][
                "distance_multipliers"
            ]
            np.testing.assert_allclose(multipliers, [-0.005 / (1 + alpha)], atol=1e-14)
