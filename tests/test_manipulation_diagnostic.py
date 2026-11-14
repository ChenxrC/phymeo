"""HOST diagnostic contracts; these tests do not qualify manipulation success."""

import numpy as np
import pytest

from phymeo.control.manipulation_diagnostic import (
    NativeJointPD,
    NativePDDiagnostic,
    solve_native_point_ik,
)
from phymeo.training.manipulation_tasks import manipulation_scenes


@pytest.fixture
def mj():
    return pytest.importorskip("mujoco")


def fixture_model(mj, hinge_motor=None, extra_motor=""):
    hinge_motor = hinge_motor or (
        '<motor name="hinge-motor" joint="hinge" gear="1" ctrllimited="true" '
        'ctrlrange="-20 20" forcelimited="true" forcerange="-20 20"/>'
    )
    # The independent FREE object precedes the robot. Motor order is reversed
    # relative to qpos/DOF order, so policy columns cannot stand in for addresses.
    return mj.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><option gravity="0 0 -9.81"/>'
        '<worldbody><body name="object" pos="0 0 1"><freejoint name="object-free"/>'
        '<geom type="sphere" size=".04" mass=".2"/></body>'
        '<body name="arm" pos=".1 0 .6"><joint name="hinge" type="hinge" '
        'axis="0 1 0" limited="true" range="-1.5 1.5"/>'
        '<geom type="capsule" fromto="0 0 0 .24 0 0" size=".02" mass=".2"/>'
        '<body name="slider" pos=".24 0 0"><joint name="slide" type="slide" '
        'axis="0 0 1" limited="true" range="-.1 .1"/>'
        '<geom type="sphere" size=".025" mass=".1"/>'
        '<body name="tool" pos=".15 0 0"><geom type="sphere" size=".015" mass=".03"/>'
        "</body></body></body></worldbody><actuator>"
        '<motor name="slide-motor" joint="slide" gear="1" ctrllimited="true" '
        'ctrlrange="-20 20" forcelimited="true" forcerange="-20 20"/>'
        + hinge_motor
        + extra_motor
        + "</actuator></mujoco>"
    )


def native_state(mj, model, names, positions, velocities):
    q, v = model.qpos0.copy(), np.zeros(model.nv)
    for name, position, velocity in zip(names, positions, velocities, strict=True):
        joint = model.joint(name).id
        q[model.jnt_qposadr[joint]] = position
        v[model.jnt_dofadr[joint]] = velocity
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:] = q, v
    mj.mj_forward(model, data)
    return q, v, data


def forbidden(*args, **kwargs):
    raise AssertionError("controller must not call forward/inverse/step or publish native state")


def test_pd_uses_original_addresses_and_independent_native_bias(mj, monkeypatch):
    model = fixture_model(mj)
    names = ("slide", "hinge")
    q, v, oracle = native_state(mj, model, names, (0.03, 0.22), (-0.02, 0.1))
    q_before, v_before, q0_before = q.copy(), v.copy(), model.qpos0.copy()
    controller = NativeJointPD(model, names, (250, 25), (12, 3))
    assert controller.qpos_addresses == (8, 7)
    assert controller.dof_addresses == (7, 6)
    assert controller.actuator_ids == (0, 1)
    target, target_v = np.array((0.04, 0.25)), np.array((0.01, -0.02))
    expected_bias = oracle.qfrc_bias[list(controller.dof_addresses)].copy()
    expected = (
        expected_bias
        + np.array((250, 25)) * (target - q[[8, 7]])
        + np.array((12, 3)) * (target_v - v[[7, 6]])
    )
    for name in ("mj_forward", "mj_inverse", "mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mj, name, forbidden)
    ctrl, diagnostic = controller.command(q, v, target, target_v)
    assert isinstance(diagnostic, NativePDDiagnostic)
    np.testing.assert_allclose(diagnostic.bias, expected_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ctrl, np.clip(expected, -19.9, 19.9), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(q, q_before)
    np.testing.assert_array_equal(v, v_before)
    np.testing.assert_array_equal(model.qpos0, q0_before)
    assert not ctrl.flags.writeable
    with pytest.raises(ValueError):
        diagnostic.bias.flags.writeable = True
    first = ctrl.copy()
    controller.command(q, v, (0.02, 0.15))
    np.testing.assert_array_equal(ctrl, first)


@pytest.mark.parametrize("index", (0, 3, 7))
def test_original_three_morphologies_match_native_bias_without_contact_replay(mj, index):
    scene = manipulation_scenes()[index]
    model = mj.MjModel.from_xml_string(scene.mjcf)
    names = scene.robot_joint_names
    controller = NativeJointPD(model, names, 25, 3)
    positions = [model.qpos0[model.jnt_qposadr[model.joint(name).id]] for name in names]
    q, v, oracle = native_state(mj, model, names, positions, [0.0] * len(names))
    ctrl, diagnostic = controller.command(q, v, positions)
    expected = oracle.qfrc_bias[list(controller.dof_addresses)]
    np.testing.assert_allclose(diagnostic.bias, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        ctrl[list(controller.actuator_ids)], np.clip(expected, -19.9, 19.9), rtol=1e-12, atol=1e-12
    )
    assert np.max(np.abs(ctrl)) <= 19.9


def test_pd_clamps_original_force_limits_and_leaves_unselected_motor_zero(mj):
    model = fixture_model(
        mj,
        '<motor name="hinge-motor" joint="hinge" '
        'ctrllimited="true" ctrlrange="-2 3" '
        'forcelimited="true" forcerange="-.5 .75"/>',
    )
    controller = NativeJointPD(model, ("hinge",), 1000, 3)
    ctrl, diagnostic = controller.command(model.qpos0, np.zeros(model.nv), (1.0,))
    assert ctrl[0] == 0.0 and ctrl[1] == 0.75 and diagnostic.saturated[0]
    full = NativeJointPD(fixture_model(mj), ("hinge", "slide"), 1000, 3)
    ctrl, diagnostic = full.command(full.model.qpos0, np.zeros(full.model.nv), (1.0, 0.1))
    assert np.max(np.abs(ctrl)) == 19.9 and np.all(diagnostic.saturated)
    with pytest.raises(ValueError, match="joint limits"):
        controller.command(model.qpos0, np.zeros(model.nv), (2.0,))


@pytest.mark.parametrize(
    "actor",
    (
        '<motor name="hinge-motor" joint="hinge" gear="2"/>',
        '<position name="hinge-motor" joint="hinge" kp="10"/>',
        '<general name="hinge-motor" joint="hinge" dyntype="filter" dynprm=".1"/>',
        '<general name="hinge-motor" joint="hinge" gaintype="fixed" gainprm="2"/>',
    ),
)
def test_pd_rejects_non_motor_transmissions(mj, actor):
    with pytest.raises(ValueError, match="motor|gear"):
        NativeJointPD(fixture_model(mj, actor), ("hinge",), 25, 3)


def test_address_and_gain_rejections_precede_native_commands(mj):
    model = fixture_model(mj)
    for names in (("unknown",), ("hinge", "hinge"), ("object-free",), ()):
        with pytest.raises(ValueError):
            NativeJointPD(model, names, 25, 3)
    duplicate = fixture_model(mj, extra_motor='<motor name="second-hinge" joint="hinge"/>')
    with pytest.raises(ValueError, match="exactly one"):
        NativeJointPD(duplicate, ("hinge",), 25, 3)
    for kp, kd in ((-1, 1), (25, np.nan), ((1, 2), 3), (True, 3)):
        with pytest.raises(ValueError):
            NativeJointPD(model, ("hinge",), kp, kd)
    controller = NativeJointPD(model, ("hinge",), 25, 3)
    q = model.qpos0.copy()
    q[3:7] = 0
    with pytest.raises(ValueError, match="quaternion"):
        controller.command(q, np.zeros(model.nv), (0.0,))


def native_point(mj, model, q, body_name, local):
    data = mj.MjData(model)
    data.qpos[:] = q
    mj.mj_forward(model, data)
    body = model.body(body_name).id
    return data.xpos[body] + data.xmat[body].reshape(3, 3) @ local


def test_ik_reprojects_independently_and_preserves_free_object_and_seed(mj, monkeypatch):
    model = fixture_model(mj)
    seed = model.qpos0.copy()
    known = seed.copy()
    known[[7, 8]] = (0.35, 0.025)
    local = np.array((0.01, 0.02, -0.04))
    target = native_point(mj, model, known, "tool", local)
    seed_before, q0_before = seed.copy(), model.qpos0.copy()
    with monkeypatch.context() as guard:
        for name in ("mj_forward", "mj_inverse", "mj_step", "mj_step1", "mj_step2"):
            guard.setattr(mj, name, forbidden)
        result = solve_native_point_ik(
            model,
            target,
            seed,
            ("slide", "hinge"),
            local_point=local,
            tolerance_m=1e-7,
            max_iterations=100,
        )
    assert result.converged and result.error_m <= 1e-7
    actual = native_point(mj, model, result.configuration, "tool", local)
    np.testing.assert_allclose(actual, target, rtol=0, atol=1e-7)
    np.testing.assert_allclose(result.position, actual, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(result.configuration[:7], seed[:7])
    np.testing.assert_array_equal(result.robot_configuration, result.configuration[[8, 7]])
    np.testing.assert_array_equal(seed, seed_before)
    np.testing.assert_array_equal(model.qpos0, q0_before)
    assert not result.configuration.flags.writeable
    assert -0.1 <= result.configuration[8] <= 0.1 and -1.5 <= result.configuration[7] <= 1.5


def test_ik_unreachable_reports_bounded_residual_without_publishing_state(mj):
    model = fixture_model(mj)
    seed = model.qpos0.copy()
    result = solve_native_point_ik(model, (10, 10, 10), seed, ("hinge", "slide"), max_iterations=5)
    assert not result.converged and result.status in ("stalled", "iteration-limit")
    assert result.iterations <= 5 and result.error_m > 1
    assert np.isfinite(result.configuration).all()
    np.testing.assert_array_equal(result.configuration[:7], seed[:7])
    assert -1.5 <= result.configuration[7] <= 1.5 and -0.1 <= result.configuration[8] <= 0.1
    with pytest.raises(ValueError, match="unknown"):
        solve_native_point_ik(model, (0, 0, 0), seed, ("hinge",), body_name="missing")
    with pytest.raises(ValueError, match="ancestors"):
        solve_native_point_ik(model, (0, 0, 0), seed, ("hinge",), body_name="object")
    with pytest.raises(ValueError, match="bounded"):
        solve_native_point_ik(model, (0, 0, 0), seed, ("hinge",), max_iterations=True)
