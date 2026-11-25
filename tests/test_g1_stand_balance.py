"""Analytic controls and wrench checks for the G1 standing example."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")


@pytest.fixture
def examples_path(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))


@pytest.fixture
def diagnostics(examples_path):
    return import_module("g1_stand_diagnostics")


@pytest.fixture
def balance(examples_path):
    return import_module("_g1_stand_balance")


@pytest.mark.parametrize("field", ["q", "v", "step_s"])
def test_diagnostics_refuse_different_starting_conditions(diagnostics, field):
    baseline = dict(model_sha256="a", step_s=0.002, initial_state={"q": [0.0], "v": [0.0]})
    candidate = dict(baseline, initial_state=baseline["initial_state"].copy())
    if field == "step_s":
        candidate[field] = 0.004
    else:
        candidate["initial_state"][field] = [0.01]
    with pytest.raises(ValueError, match="different"):
        diagnostics.compare_runs(baseline, candidate)


def test_cop_from_two_wrenches_is_independent_of_body_moment_reference(diagnostics):
    # Forces include shear and free couples, and both COMs are above the plane.
    points = np.array([[0.30, 0.16, 0.50], [-0.10, -0.12, 0.50]])
    forces = np.array([[8.0, -4.0, 60.0], [-3.0, 5.0, 40.0]])
    couples = np.array([[1.2, -0.6, 0.2], [-0.4, 0.2, -0.1]])
    references = np.array([[0.11, 0.13, 0.85], [0.18, -0.20, 0.79]])
    offsets = np.array([[0.08, -0.03, 0.12], [-0.13, 0.09, -0.04]])
    for body_com in (references, references + offsets):
        moments = couples + np.cross(points - body_com, forces)
        values = diagnostics.ground_cop(forces, moments, body_com, ground_z=0.50)
        np.testing.assert_allclose(values["total_force_world_n"], [5, 1, 100], atol=1e-12)
        np.testing.assert_allclose(
            values["total_moment_about_ground_origin_nm"], [5.6, -14.4, -3.24], atol=1e-12
        )
        np.testing.assert_allclose(values["cop_xy_m"], [0.144, 0.056], atol=1e-12)
        np.testing.assert_allclose(
            values["foot_cop_xy_m"], [[0.31, 0.18], [-0.105, -0.13]], atol=1e-12
        )


def test_cop_requires_vertical_load_and_keeps_single_support(diagnostics):
    feet = np.array([[0.3, 0.12, 0.2], [-0.2, -0.14, 0.2]])
    forces = np.array([[0.0, 0.0, 50.0], [0.0, 0.0, 0.0]])
    zero_couples = np.zeros((2, 3))
    supported = diagnostics.ground_cop(forces, zero_couples, feet)
    np.testing.assert_allclose(supported["cop_xy_m"], feet[0, :2], atol=1e-12)
    np.testing.assert_allclose(supported["foot_cop_xy_m"][0], feet[0, :2], atol=1e-12)
    assert supported["foot_cop_xy_m"][1] is None

    # A finite moment alone must not produce an enormous or fabricated CoP.
    forces[:, 2] = (0.4, 0.5)
    unloaded = diagnostics.ground_cop(forces, np.full((2, 3), 0.25), feet, min_fz=1.0)
    assert unloaded["cop_xy_m"] is None
    assert all(point is None for point in unloaded["foot_cop_xy_m"])


def test_forward_onset_rejects_a_single_spike_and_an_unloaded_sample(diagnostics):
    times = np.arange(8) * 0.005
    positions = [0.30, 0.307, 0.30, 0.307, None, 0.307, 0.308, 0.310]
    assert diagnostics.forward_onset(times, positions) == pytest.approx(0.025)
    assert diagnostics.forward_onset(times[:-1], positions[:-1]) is None


@pytest.mark.parametrize(
    ("offset", "forward", "expected_offset", "expected_torque"),
    [
        ((0.02, 0.0), (1, 0), (0.02, 0.0), (0.0, 2.0, 0.0)),
        ((-0.02, 0.0), (1, 0), (-0.02, 0.0), (0.0, -2.0, 0.0)),
        ((0.2, 0.05), (1, 0), (0.09, 0.015), (-1.5, 9.0, 0.0)),
        ((-0.2, -0.05), (1, 0), (-0.04, -0.015), (1.5, -4.0, 0.0)),
        ((-0.05, 0.2), (0, 1), (-0.015, 0.09), (-9.0, -1.5, 0.0)),
    ],
    ids=("forward", "backward", "toe-and-side", "heel-and-side", "yawed-toe-and-side"),
)
def test_ankle_moment_direction_and_footprint_bounds(
    balance, offset, forward, expected_offset, expected_torque
):
    # A positive x CoP offset under 100 N requires +y foot-side actuator torque.
    bounded, moment = balance.ankle_balance_moment(np.array(offset), np.array(forward), 100.0)
    np.testing.assert_allclose(bounded, expected_offset, atol=1e-12)
    np.testing.assert_allclose(moment, expected_torque, atol=1e-12)


def test_capture_uses_height_above_support_and_world_planar_velocity(balance):
    omega, capture, desired = balance.capture_point_target(
        [0.04, -0.03, 0.75], [0.12, -0.06, 0.8], 0.50, [0.02, -0.02], 9.0
    )
    assert omega == pytest.approx(6.0)
    np.testing.assert_allclose(capture, [0.06, -0.04], atol=1e-12)
    np.testing.assert_allclose(desired, [0.08, -0.05], atol=1e-12)


def test_standing_motor_commands_respect_contact_gear_limits_and_other_joints(balance):
    # Reverse the gearing on one pitch axis and interleave the motors. A
    # positive pitch torque then needs opposite control signs on the two feet.
    names = (
        "right_ankle_roll_joint",
        "arm_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_ankle_roll_joint",
    )
    feet = "".join(
        f'<body pos="0 {y} -.38"><joint name="{side}_ankle_pitch_joint" axis="0 1 0"/>'
        '<geom type="sphere" size=".04" mass=".5"/>'
        f'<body><joint name="{side}_ankle_roll_joint" axis="1 0 0"/>'
        f'<geom name="{side}_pad" type="capsule" fromto="-.05 0 -.12 .05 0 -.12" '
        'size=".025" mass="1"/></body></body>'
        for side, y in (("left", 0.1), ("right", -0.1))
    )
    motors = "".join(
        f'<motor joint="{name}" gear="{gear}" ctrllimited="true" ctrlrange="-.5 .5"/>'
        for name, gear in zip(names, (1, 1, 2, -2, 1), strict=True)
    )
    model = mj.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="pelvis" pos="0 0 .525"><freejoint/>'
        '<geom type="box" size=".12 .12 .08" mass="12"/>'
        '<body pos="0 0 .16"><joint name="arm_joint" axis="0 0 1"/>'
        f'<geom size=".04" mass="1"/></body>{feet}</body></worldbody>'
        f"<actuator>{motors}</actuator></mujoco>"
    )
    protocol = SimpleNamespace(
        model=model,
        root=model.body("pelvis").id,
        weight=9.81 * model.body_mass.sum(),
        foot_geoms={model.geom("left_pad").id: 0, model.geom("right_pad").id: 1},
        names=names,
    )
    q = model.qpos0.copy()
    v = np.zeros(model.nv)
    v[0] = 2.0
    correction = balance.AnkleBalance(protocol, q)
    cmd = np.array([0.0, 0.31, 0.0, 0.0, 0.0])
    proposal = dict(ctrl=cmd.copy(), unclamped_ctrl=cmd.copy(), saturated=np.zeros(5, dtype=bool))
    for load in (None, [1.0, -3.0]):
        unchanged, observed = correction.correct(proposal, q, v, load)
        np.testing.assert_array_equal(unchanged["ctrl"], cmd)
        np.testing.assert_array_equal(observed["allocated_vertical_n"], [0.0, 0.0])

    bilateral, observed = correction.correct(proposal, q, v, [500.0, 500.0])
    np.testing.assert_allclose(observed["allocated_vertical_n"], [protocol.weight / 2] * 2)
    np.testing.assert_allclose(bilateral["ctrl"], [0, 0.31, 0.5, -0.5, 0], atol=1e-12)
    np.testing.assert_array_equal(bilateral["saturated"], [False, False, True, True, False])
    np.testing.assert_allclose(observed["applied_joint_delta_nm"], [[1.0, 0], [1.0, 0]])
    np.testing.assert_array_equal(proposal["ctrl"], cmd)
    np.testing.assert_array_equal(proposal["unclamped_ctrl"], cmd)

    single, observed = correction.correct(proposal, q, v, [0.0, 500.0])
    np.testing.assert_allclose(observed["allocated_vertical_n"], [0.0, protocol.weight])
    np.testing.assert_array_equal(single["ctrl"][[1, 2, 4]], cmd[[1, 2, 4]])
    assert single["ctrl"][3] == pytest.approx(-0.5)
