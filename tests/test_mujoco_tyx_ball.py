"""BALL1 fixed-root native quaternions: tests authored before implementation.

CPU 3.3.7 is independently compiled; analytical rotor terms use declared inertia.
The retained T1 engineering gates are SI component gates, not exact certificates.
"""

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import numpy as np
import pytest

from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime, MuJoCoTyxUnsupported
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
H = 0.001
TOL = {
    "body_position": (2e-5, 3e-5),
    "body_rotation": (2e-5, 3e-5),
    "body_com": (2e-5, 3e-5),
    "jacobian_com": (2e-4, 5e-5),
    "mass_matrix": (2e-4, 3e-5),
    "bias": (3e-4, 2e-4),
    "passive": (3e-4, 2e-4),
    "applied": (3e-4, 2e-4),
    "smooth_acceleration": (5e-4, 8e-4),
    "integration_acceleration": (5e-4, 8e-4),
}
Q_ATOL, V_ATOL, TIME_ATOL = 5e-5, 3e-4, 2e-6


def body(
    name,
    joint,
    child="",
    *,
    pos=".1 -.03 .17",
    quat=".9238795325112867 0 .3826834323650898 0",
    com=".04 -.03 .02",
):
    return f'''<body name="{name}" pos="{pos}" quat="{quat}">{joint}
      <inertial mass="1.3" pos="{com}" fullinertia=".04 .055 .065 .002 -.001 .003"/>
      <geom type="box" size=".04 .03 .05" contype="0" conaffinity="0"/>{child}</body>'''


def joint(name, kind="ball"):
    scalar = 'axis="1 2 -1" ref=".07" springref="-.03" stiffness=".2"' if kind != "ball" else ""
    return f'<joint name="{name}" type="{kind}" pos=".015 -.02 .01" damping=".12" armature=".04" {scalar}/>'


def xml(name, content, gravity=".2 -.3 -9.81"):
    return f'''<mujoco model="{name}"><compiler angle="radian" fusestatic="false"/>
    <option timestep=".001" gravity="{gravity}" integrator="Euler"/>
    <worldbody>{content}</worldbody></mujoco>'''


ROTOR = """<mujoco model="anisotropic-ball"><compiler angle="radian"/>
<option timestep=".001" gravity="0 0 0" integrator="Euler"/>
<worldbody><body name="rotor"><joint name="ball" type="ball" damping=".12" armature=".04"/>
<inertial mass="2" pos="0 0 0" diaginertia=".04 .06 .08"/>
<geom type="sphere" size=".05" contype="0" conaffinity="0"/>
</body></worldbody></mujoco>"""
ECCENTRIC = xml("eccentric", body("eccentric", joint("ball")))
HINGE_BALL_SLIDE = xml(
    "hinge-ball-slide",
    body(
        "base",
        joint("hinge", "hinge"),
        body("middle", joint("ball"), body("tip", joint("slide", "slide"))),
    ),
)
BALL_FORK = xml(
    "ball-hinge-fork",
    body(
        "base",
        joint("ball"),
        body("left", joint("hinge", "hinge"), pos=".3 .1 .04")
        + body("right", joint("slide", "slide"), pos="-.2 -.1 .08"),
    ),
)
MAXIMUM = xml(
    "two-balls-two-scalars",
    body(
        "base",
        joint("ball0"),
        body(
            "left",
            joint("hinge", "hinge"),
            body("second", joint("ball1"), body("tip", joint("slide", "slide"))),
        ),
    ),
)
CASES = (
    ("rotor", ROTOR),
    ("eccentric", ECCENTRIC),
    ("hinge-ball-slide", HINGE_BALL_SLIDE),
    ("ball-fork", BALL_FORK),
    ("maximum", MAXIMUM),
)


def deps():
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("tyx")
    assert mj.__version__ == "3.3.7"
    return mj


def initial(mj, model, slot=0):
    q = model.qpos0.copy()
    for j in range(model.njnt):
        qa = int(model.jnt_qposadr[j])
        if model.jnt_type[j] == mj.mjtJoint.mjJNT_BALL:
            axis = np.array([1.0 + 0.1 * slot, -2.0, 0.7 + 0.2 * j])
            axis /= np.linalg.norm(axis)
            angle = 0.37 + 0.03 * slot + 0.05 * j
            q[qa : qa + 4] = np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]
        else:
            q[qa] += 0.11 + 0.013 * slot
    v = np.array(
        [(-1 if i % 2 else 1) * (0.2 + 0.07 * i + 0.01 * slot) for i in range(model.nv)], np.float32
    )
    return MuJoCoInitialState(q, v)


def control(model, tick, slot=0):
    sign = -1 if (tick // 8 + slot) % 2 else 1
    force = np.array([sign * 0.03 * (i + 1) for i in range(model.nv)], np.float32)
    wrench = np.zeros((model.nbody, 6), np.float32)
    wrench[-1] = np.array([0.3, -0.2, 0.4, 0.01, -0.03, 0.02]) * (1 + 0.2 * slot)
    if tick >= 16:
        wrench *= 0.25
    return MuJoCoControl(np.empty(0, np.float32), qfrc_applied=force, xfrc_applied=wrench)


def set_native(data, packet):
    data.ctrl[:] = packet.ctrl
    data.qfrc_applied[:] = packet.qfrc_applied
    data.xfrc_applied[:] = packet.xfrc_applied


def native(mj, model, q, v, packet, time):
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = q, v, time
    set_native(data, packet)
    mj.mj_forward(model, data)
    assert data.ncon == data.nefc == 0
    mass = np.empty((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    jac = np.zeros((model.nbody, 6, model.nv))
    applied = data.qfrc_applied.copy()
    for b in range(model.nbody):
        mj.mj_jac(model, data, jac[b, :3], jac[b, 3:], data.xipos[b], b)
        applied += jac[b].T @ packet.xfrc_applied[b]
    rhs = applied + data.qfrc_passive - data.qfrc_bias
    smooth = np.linalg.solve(mass, rhs)
    np.testing.assert_allclose(smooth, data.qacc_smooth, atol=2e-10, rtol=2e-11)
    return dict(
        input_q=np.array(q),
        input_v=np.array(v),
        time=float(time),
        body_position=data.xpos.copy(),
        body_rotation=data.xmat.reshape(model.nbody, 3, 3).copy(),
        body_com=data.xipos.copy(),
        jacobian_com=jac,
        mass_matrix=mass,
        bias=data.qfrc_bias.copy(),
        passive=data.qfrc_passive.copy(),
        applied=applied,
        smooth_acceleration=smooth,
        integration_acceleration=np.linalg.solve(mass + H * np.diag(model.dof_damping), rhs),
    )


def plain(value):
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def save(path, payload):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(plain(payload), allow_nan=False, separators=(",", ":")) + "\n")


def assert_dynamics(row, expected):
    dyn = row.dynamics
    np.testing.assert_array_equal(dyn.input_q, row.input_q)
    np.testing.assert_array_equal(dyn.input_v, row.input_v)
    assert dyn.time == row.input_time
    for name, (rtol, atol) in TOL.items():
        value = (
            row.integration_acceleration
            if name == "integration_acceleration"
            else getattr(dyn, name)
        )
        assert value.shape == expected[name].shape
        np.testing.assert_allclose(value, expected[name], rtol=rtol, atol=atol, err_msg=name)
        with pytest.raises(ValueError):
            value.setflags(write=True)
    np.testing.assert_allclose(row.qacc, expected["smooth_acceleration"], rtol=5e-4, atol=8e-4)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case[0])
def test_ball_native_complete_dynamics_and_continuous_euler(mode, case, tmp_path):
    mj = deps()
    name, source = case
    model = mj.MjModel.from_xml_string(source)
    compiled = compile_mujoco_xml(source, require_asset=False)
    initial_state = initial(mj, model)
    path = tmp_path / "ball-continuous.jsonl"
    save(
        path,
        dict(
            case=name,
            source_xml=source,
            mode=mode,
            initial=initial_state,
            steps=24,
            tolerances=TOL,
            q_atol=Q_ATOL,
            v_atol=V_ATOL,
            time_atol=TIME_ATOL,
        ),
    )
    with MuJoCoTyxRuntime(
        compiled, mode=mode, world_keys=(19,), initial_states={19: initial_state}
    ) as runtime:
        start = runtime.integration_snapshot()[19]
        reference = mj.MjData(model)
        reference.qpos[:], reference.qvel[:], reference.time = start.q, start.v, start.time
        if name == "maximum":
            assert (model.nq, model.nv) == (10, 8)
            # The second ball and final scalar cross the old width-eight q boundary.
            assert int(model.jnt_qposadr[-1]) == 9
        for tick in range(24):
            before = runtime.integration_snapshot()[19]
            packet = control(model, tick)
            expected = native(mj, model, before.q, before.v, packet, before.time)
            set_native(reference, packet)
            mj.mj_step(model, reference)
            runtime.set_controls({19: packet})
            try:
                assert runtime.step(H) == {19: True}
                after = runtime.integration_snapshot()[19]
                row = after.last_step
                save(
                    path,
                    dict(
                        step=tick + 1,
                        before=before,
                        requested=packet,
                        native=expected,
                        continuous_q=reference.qpos.copy(),
                        continuous_v=reference.qvel.copy(),
                        after=after,
                    ),
                )
                np.testing.assert_array_equal(row.input_q, before.q)
                np.testing.assert_array_equal(row.input_v, before.v)
                np.testing.assert_array_equal(row.stored_control.qfrc_applied, packet.qfrc_applied)
                np.testing.assert_array_equal(row.stored_control.xfrc_applied, packet.xfrc_applied)
                assert row.tick == after.tick == tick + 1 and row.contacts == ()
                assert row.input_time == before.time and row.output_time == after.time
                assert_dynamics(row, expected)
                np.testing.assert_allclose(after.q, reference.qpos, rtol=0, atol=Q_ATOL)
                np.testing.assert_allclose(after.v, reference.qvel, rtol=0, atol=V_ATOL)
                assert after.time == pytest.approx(reference.time, abs=TIME_ATOL)
                for j in range(model.njnt):
                    if model.jnt_type[j] == mj.mjtJoint.mjJNT_BALL:
                        qa = int(model.jnt_qposadr[j])
                        assert (
                            abs(np.dot(after.q[qa : qa + 4], after.q[qa : qa + 4]) - 1)
                            <= 8 * np.finfo(np.float32).eps
                        )
            except Exception:
                save(
                    path,
                    dict(
                        failure_step=tick + 1,
                        before=before,
                        requested=packet,
                        native=expected,
                        published=runtime.integration_snapshot(),
                    ),
                )
                raise


@pytest.mark.parametrize("mode", MODES)
def test_ball_anisotropic_rotor_analytic_three_axis_torque(mode, tmp_path):
    mj = deps()
    model = mj.MjModel.from_xml_string(ROTOR)
    with MuJoCoTyxRuntime(
        compile_mujoco_xml(ROTOR, require_asset=False),
        mode=mode,
        initial_states={0: initial(mj, model)},
    ) as runtime:
        before = runtime.integration_snapshot()[0]
        torque = np.array([0.11, -0.07, 0.13], np.float32)
        packet = MuJoCoControl([], qfrc_applied=torque, xfrc_applied=np.zeros((model.nbody, 6)))
        inertia = np.array([0.04, 0.06, 0.08])
        mass = np.diag(inertia + 0.04)
        bias = np.cross(before.v, inertia * before.v)
        passive = -0.12 * before.v
        smooth = np.linalg.solve(mass, torque + passive - bias)
        integration = np.linalg.solve(mass + H * 0.12 * np.eye(3), torque + passive - bias)
        runtime.set_controls({0: packet})
        runtime.step(H)
        after = runtime.integration_snapshot()[0]
        row = after.last_step
        save(
            tmp_path / "analytic-rotor.jsonl",
            dict(
                before=before,
                after=after,
                mass=mass,
                bias=bias,
                passive=passive,
                applied=torque,
                smooth=smooth,
                integration=integration,
            ),
        )
        np.testing.assert_allclose(row.dynamics.mass_matrix, mass, rtol=2e-4, atol=3e-5)
        for actual, expected in (
            (row.dynamics.bias, bias),
            (row.dynamics.passive, passive),
            (row.dynamics.applied, torque),
        ):
            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-4)
        np.testing.assert_allclose(row.qacc, smooth, rtol=5e-4, atol=8e-4)
        np.testing.assert_allclose(row.integration_acceleration, integration, rtol=5e-4, atol=8e-4)
        np.testing.assert_allclose(after.v, before.v + H * integration, rtol=0, atol=V_ATOL)
        assert np.max(np.abs(bias)) > 1e-4


NEGATIVE_XML = (
    ("ball-spring", ROTOR.replace('type="ball"', 'type="ball" stiffness="1"')),
    ("ball-limit", ROTOR.replace('type="ball"', 'type="ball" limited="true" range="0 .5"')),
    ("dry", ROTOR.replace('type="ball"', 'type="ball" frictionloss=".1"')),
    (
        "contact-condim4",
        ROTOR.replace(
            "<worldbody>", '<worldbody><geom type="plane" size="1 1 .1" condim="4"/>'
        ).replace('contype="0" conaffinity="0"', 'contype="1" conaffinity="1"'),
    ),
    (
        "three-balls",
        xml("over-cap", body("a", joint("a"), body("b", joint("b"), body("c", joint("c"))))),
    ),
    (
        "same-body-slide-ball",
        xml("same-body-slide-ball", body("mixed", joint("slide", "slide") + joint("ball"))),
    ),
    (
        "free-and-ball",
        xml("free-and-ball", body("free", "<freejoint/>", body("ball-child", joint("ball")))),
    ),
    (
        "scalar-limit",
        HINGE_BALL_SLIDE.replace(
            'name="hinge" type="hinge"', 'name="hinge" type="hinge" limited="true" range="-1 1"'
        ),
    ),
)


@pytest.mark.parametrize("case", NEGATIVE_XML, ids=lambda case: case[0])
def test_ball_unsupported_domains_reject_before_device(case, monkeypatch):
    deps()
    from phymeo.backends.mujoco_tyx import _tree_device

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported ball model allocated a device")

    monkeypatch.setattr(_tree_device.TyxTreeDevice, "__init__", forbidden)
    with pytest.raises(MuJoCoTyxUnsupported):
        MuJoCoTyxRuntime(compile_mujoco_xml(case[1], require_asset=False), mode="interpret")


@pytest.mark.parametrize("kind", ["zero", "nonunit", "nan"])
def test_ball_bad_initial_quaternion_rejects(kind):
    mj = deps()
    model = mj.MjModel.from_xml_string(ROTOR)
    q = initial(mj, model).q.copy()
    if kind == "zero":
        q[:] = 0
    elif kind == "nonunit":
        q *= 1.01
    else:
        q[2] = np.nan
    with pytest.raises(ValueError):
        MuJoCoTyxRuntime(
            compile_mujoco_xml(ROTOR, require_asset=False),
            mode="interpret",
            initial_states={0: MuJoCoInitialState(q, np.zeros(3))},
        )


def test_ball_static_native_addresses_and_full_qpos_reference():
    mj = deps()
    from phymeo.backends.mujoco_tyx._tree_model import compile_tree_model

    model = mj.MjModel.from_xml_string(MAXIMUM)
    plan = compile_tree_model(mj, model)
    assert (plan.nq, plan.nv, plan.vectors.shape) == (10, 8, (9, 10))
    np.testing.assert_array_equal(plan.vectors[4], model.qpos0.astype(np.float32))
    for joint_id in range(model.njnt):
        qa, da, body_id = map(
            int,
            (model.jnt_qposadr[joint_id], model.jnt_dofadr[joint_id], model.jnt_bodyid[joint_id]),
        )
        is_ball = model.jnt_type[joint_id] == mj.mjtJoint.mjJNT_BALL
        expected_kind = 2 if is_ball else int(model.jnt_type[joint_id] == mj.mjtJoint.mjJNT_HINGE)
        np.testing.assert_array_equal(plan.joint_ids[joint_id], [expected_kind, qa, da, body_id])
        for column in range(da, da + (3 if is_ball else 1)):
            np.testing.assert_array_equal(
                plan.dof_ids[column], [int(expected_kind != 0), -1 if is_ball else qa, joint_id]
            )
    expected_ancestors = np.zeros((model.nbody, 8), np.int32)
    for body_id in range(1, model.nbody):
        cursor = body_id
        while cursor:
            for joint_id in range(
                model.body_jntadr[cursor], model.body_jntadr[cursor] + model.body_jntnum[cursor]
            ):
                start = model.jnt_dofadr[joint_id]
                count = 3 if model.jnt_type[joint_id] == mj.mjtJoint.mjJNT_BALL else 1
                expected_ancestors[body_id, start : start + count] = 1
            cursor = int(model.body_parentid[cursor])
    np.testing.assert_array_equal(plan.ancestors, expected_ancestors)
