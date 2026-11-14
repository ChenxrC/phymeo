"""FREE+BALL Batch B: fixed source, native addresses and independent CPU oracles.

Authored before product implementation. Numerical/physics gates are inherited from
the accepted floating-tree and JOINT motor suites, not fitted to this fixture.
"""

import numpy as np
import pytest
from test_mujoco_tyx_activation import save
from test_mujoco_tyx_ball_joint_motor import dense_native_moment, transmission_oracle
from test_mujoco_tyx_floating import (
    MODES,
    Q_ATOL,
    TIME_ATOL,
    TOL,
    V_ATOL,
    H,
    _assert_dynamics,
    _deps,
    _native,
    _set_control,
    _stored,
)

from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.backends.mujoco_tyx import MuJoCoTyxUnsupported
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

KEYS = (3, 29)
CASES = (
    "passive",
    "oblique",
    "shared",
    "mixed",
    "branches",
    "nested",
    "clamps-disabled",
    "sign",
    "multiple-free",
)
FORCE = dict(rtol=3e-4, atol=2e-4)
SCHEMA = "native-free-ball-joint-motor-transmission-v1"


def body(name, kind="ball", child="", pos=".12 -.04 .23"):
    joint = f'<joint name="{name}" type="{kind}" pos=".02 -.01 .03" damping=".04" armature=".003"/>'
    if kind == "slide":
        joint = joint.replace('type="slide"', 'type="slide" axis=".3 -.4 .5"')
    elif kind == "hinge":
        joint = joint.replace('type="hinge"', 'type="hinge" axis=".2 .7 -.3"')
    return (
        f'<body name="{name}-body" pos="{pos}" quat=".9238795325112867 0 .3826834323650898 0">'
        '<inertial pos=".03 -.02 .04" quat=".9659258262890683 .2588190451025207 0 0" '
        'mass=".7" diaginertia=".035 .045 .055"/>'
        + joint
        + '<geom type="sphere" size=".035" contype="0" conaffinity="0"/>'
        + child
        + "</body>"
    )


def motor(name, target, gear, extra="", gain="1"):
    return (
        f'<general name="{name}" joint="{target}" gear="{gear}" dyntype="none" '
        f'gaintype="fixed" gainprm="{gain}" {extra}/>'
    )


def source(case):
    if case == "multiple-free":
        # Promote the exact former structural rejection, without changing its XML.
        return negative_source("multiple-free")
    child = body("ball")
    actors = "" if case == "passive" else motor("drive", "ball", "1 -2 .5", gain=".8")
    if case == "shared":
        actors += motor("counter", "ball", "-.25 .5 -.125", gain=".6")
        actors += motor("zero", "ball", "0 0 0", gain=".7")
    elif case == "mixed":
        child = body("ball", child=body("hinge", "hinge", body("slide", "slide")))
        actors += motor("hinge-drive", "hinge", "-.5")
        actors += motor("slide-drive", "slide", "-2", 'biastype="affine" biasprm=".01 -.3 -.04"')
    elif case == "branches":
        child = body("ball") + body("other", pos="-.18 .07 .16")
        actors += motor("other-drive", "other", "-.3 .4 .2")
    elif case == "nested":
        child = body(
            "ball", child=body("hinge", "hinge", body("other", child=body("slide", "slide")))
        )
        actors += motor("other-drive", "other", "-.3 .4 .2")
        actors += motor("hinge-drive", "hinge", "-.5") + motor("slide-drive", "slide", ".8")
    elif case == "clamps-disabled":
        actors = motor(
            "drive",
            "ball",
            "1 -2 .5",
            'ctrllimited="true" ctrlrange="-.08 .08" forcelimited="true" forcerange="-.09 .09"',
            gain="2",
        )
        actors += motor(
            "disabled", "ball", ".2 -.3 .1", 'group="1" forcelimited="true" forcerange=".01 .025"'
        )
        actors += motor("zero", "ball", "0 0 0")
    elif case == "capacity":
        child = "".join(body(f"b{i}", pos=f".{i + 1} .04 .2") for i in range(9))
        child += body("hinge", "hinge") + body("slide", "slide")
        actors = "".join(
            motor(f"a{i}", f"b{i % 9}", ".125 -.0625 .03125", gain=".02") for i in range(32)
        )
    option = 'actuatorgroupdisable="1"' if case == "clamps-disabled" else ""
    return f"""<mujoco model="free-ball-{case}">
<compiler angle="radian"/>
<option timestep=".001" integrator="Euler" gravity="0 0 -1.2" {option}/>
<worldbody>
<body name="marker" pos="2 -1 .4"><geom size=".02" contype="0" conaffinity="0"/></body>
<body name="root" pos=".2 -.1 .7" quat=".9238795325112867 .3826834323650898 0 0">
<freejoint name="free"/>
<inertial pos=".04 -.03 .02" quat=".9659258262890683 0 .2588190451025207 0"
 mass="2.6" diaginertia=".13 .19 .23"/>
<geom type="sphere" size=".06" contype="0" conaffinity="0"/>
{child}</body></worldbody><actuator>{actors}</actuator></mujoco>"""


def initial(mj, model, slot=0, sign=False):
    q = model.qpos0.copy()
    for joint, kind in enumerate(model.jnt_type):
        qa = int(model.jnt_qposadr[joint])
        if kind in (mj.mjtJoint.mjJNT_FREE, mj.mjtJoint.mjJNT_BALL):
            adr = qa + (3 if kind == mj.mjtJoint.mjJNT_FREE else 0)
            quat = np.array([1.0, 0.13 + 0.02 * joint, -0.09, 0.07])
            q[adr : adr + 4] = quat / np.linalg.norm(quat)
            if sign and slot:
                q[adr : adr + 4] *= -1
        else:
            q[qa] = 0.03 * (joint + 1)
    if not sign:
        q[:3] += slot * np.array([0.04, -0.02, 0.03])
    velocity = np.linspace(-0.13, 0.17, model.nv) * (1 if sign else 1 + 0.1 * slot)
    return MuJoCoInitialState(q, velocity, time=0.125)


def control(model, tick, slot=0):
    phase = 1 if tick < 8 else (-1 if tick < 16 else 0)
    ctrl = phase * np.array([(0.18 + 0.01 * a) * (-1 if a % 2 else 1) for a in range(model.nu)])
    qforce = np.zeros(model.nv)
    qforce[:6] = phase * np.array([0.3, -0.2, 0.15, 0.08, -0.06, 0.04])
    qforce[6:] = phase * np.linspace(0.025, 0.085, model.nv - 6)
    wrench = np.zeros((model.nbody, 6))
    wrench[-1] = phase * (1 + 0.1 * slot) * np.array([0.35, -0.25, 0.4, 0.025, -0.035, 0.015])
    return MuJoCoControl(ctrl, qfrc_applied=qforce, xfrc_applied=wrench)


def fixture(case, mode):
    mj = _deps()
    text = source(case)
    model = mj.MjModel.from_xml_string(text)
    starts = {key: initial(mj, model, slot, case == "sign") for slot, key in enumerate(KEYS)}
    compiled = compile_mujoco_xml(text, require_asset=False)
    runtime = MuJoCoRuntime(
        compiled, mode=mode, implementation="tyx", world_keys=KEYS, initial_states=starts
    )
    return mj, model, compiled, starts, runtime


def native(mj, model, entry, packet):
    result = _native(mj, model, entry.q, entry.v, packet, entry.time)
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = entry.q, entry.v, entry.time
    _set_control(data, packet)
    mj.mj_forward(model, data)
    result.update(
        dense_moment=dense_native_moment(mj, model, data),
        scalar_force=data.actuator_force.copy(),
        actuator_velocity=data.actuator_velocity.copy(),
    )
    return result


def compare(model, entry, state, expected, analytic):
    row = state.last_step
    _assert_dynamics(row, expected)
    np.testing.assert_array_equal(row.input_q, entry.q)
    np.testing.assert_array_equal(row.input_v, entry.v)
    np.testing.assert_array_equal(row.qfrc_constraint, np.zeros(model.nv))
    assert not row.contacts and state.act.shape == (0,)
    if model.nu:
        tr = row.transmission
        assert tr.schema == SCHEMA
        for field in ("ids", "gear", "row_offsets", "dof_indices", "moment"):
            np.testing.assert_array_equal(getattr(tr, field), analytic[field])
        for field in ("clamped_control", "raw_force", "scalar_force", "generalized_force"):
            np.testing.assert_allclose(getattr(tr, field), analytic[field], **FORCE)
        np.testing.assert_allclose(tr.scalar_force, expected["scalar_force"], **FORCE)
        np.testing.assert_array_equal(tr.generalized_force, row.qfrc_actuator)
        np.testing.assert_array_equal(tr.input_control, row.control.ctrl)
        np.testing.assert_allclose(
            analytic["dense_moment"], expected["dense_moment"], atol=2e-12, rtol=0
        )
        np.testing.assert_allclose(
            analytic["dense_moment"] @ entry.v, expected["actuator_velocity"], **FORCE
        )
        np.testing.assert_allclose(
            tr.generalized_force @ entry.v,
            tr.scalar_force @ (analytic["dense_moment"] @ entry.v),
            **FORCE,
        )


def continuous(case, mode, steps, monkeypatch, tmp_path):
    mj, model, _, starts, runtime = fixture(case, mode)
    payload = dict(
        source=source(case), starts=starts, tolerance=TOL, steps=[], requested_steps=steps
    )
    references = {key: mj.MjData(model) for key in KEYS}
    for key in KEYS:
        state = runtime.integration_snapshot()[key]
        references[key].qpos[:], references[key].qvel[:], references[key].time = (
            state.q,
            state.v,
            state.time,
        )
    try:
        for tick in range(steps):
            entry = runtime.integration_snapshot()
            packets = {key: control(model, tick, slot) for slot, key in enumerate(KEYS)}
            stored = {key: _stored(packet) for key, packet in packets.items()}
            expected = {key: native(mj, model, entry[key], stored[key]) for key in KEYS}
            analytic = {
                key: transmission_oracle(mj, model, entry[key], stored[key]) for key in KEYS
            }
            record = dict(
                tick=tick,
                entry=entry,
                controls=packets,
                stored=stored,
                native=expected,
                analytic=analytic,
            )
            payload["steps"].append(record)
            runtime.set_controls(packets)
            with monkeypatch.context() as patch:

                def forbidden(*args, **kwargs):
                    raise AssertionError("CPU mj_step called by actual TYX step")

                patch.setattr(mj, "mj_step", forbidden)
                runtime.step(H)
            result = runtime.integration_snapshot()
            record["result"] = result
            for key in KEYS:
                compare(model, entry[key], result[key], expected[key], analytic[key])
                _set_control(references[key], stored[key])
                mj.mj_step(model, references[key])
                record.setdefault("continuous_native", {})[key] = dict(
                    q=references[key].qpos.copy(),
                    v=references[key].qvel.copy(),
                    time=references[key].time,
                )
                np.testing.assert_allclose(result[key].q, references[key].qpos, atol=Q_ATOL, rtol=0)
                np.testing.assert_allclose(result[key].v, references[key].qvel, atol=V_ATOL, rtol=0)
                assert abs(result[key].time - references[key].time) <= TIME_ATOL
                assert result[key].tick == tick + 1
        assert runtime.counters["completed_kernels"] == steps * 3
        assert runtime.counters["failed_batches"] == 0
    except BaseException as error:
        payload.update(failure=repr(error), retained=runtime.integration_snapshot())
        raise
    finally:
        payload["counters"] = dict(runtime.counters)
        try:
            runtime.close()
        finally:
            save(tmp_path / "free-ball.json.gz", payload)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES)
def test_free_ball_complete_native_24_steps(case, mode, monkeypatch, tmp_path):
    continuous(case, mode, 24, monkeypatch, tmp_path)


@pytest.mark.parametrize("mode", MODES)
def test_free_ball_native_capacity_35_45_32(mode, monkeypatch, tmp_path):
    continuous("capacity", mode, 2, monkeypatch, tmp_path)


NEGATIVES = {
    # Exact activation XML now has a two-step R3-A positive in both backends.
    "contact": "free-ball-contact",
    "limit": "free-ball-constraint",
    "dry": "free-ball-constraint",
    "spring": "free-ball-spring",
    "ball-affine": "actuator-domain",
    "free-target": "actuator-domain",
    "joint-in-parent": "actuator-domain",
    "actors33": "free-ball-actuator-capacity",
    "nv36": "joint-domain",
    "tiny-y": "fp32-range",
    "tiny-z": "fp32-range",
    "ball-same-body": "ball-same-body",
}


def negative_source(case):
    text = source("oblique")
    if case == "activation":
        return text.replace('dyntype="none"', 'dyntype="filterexact" dynprm=".02"')
    if case == "contact":
        return text.replace("<worldbody>", '<worldbody><geom type="plane" size="2 2 .1"/>').replace(
            'contype="0" conaffinity="0"', 'contype="1" conaffinity="1"'
        )
    if case == "limit":
        return text.replace('type="ball"', 'type="ball" limited="true" range="0 .8"')
    if case == "dry":
        return text.replace('damping=".04"', 'frictionloss=".1" damping=".04"')
    if case == "spring":
        return text.replace('type="ball"', 'type="ball" stiffness=".1"')
    if case == "ball-affine":
        return text.replace('dyntype="none"', 'biastype="affine" biasprm="0 -.1 0" dyntype="none"')
    if case == "free-target":
        return text.replace('joint="ball"', 'joint="free"')
    if case == "joint-in-parent":
        return text.replace('joint="ball"', 'jointinparent="ball"')
    if case == "actors33":
        return text.replace(
            "</actuator>",
            "".join(motor(f"extra{i}", "ball", "1 0 0") for i in range(32)) + "</actuator>",
        )
    if case == "nv36":
        return source("capacity").replace("</worldbody>", body("extra", "hinge") + "</worldbody>")
    if case == "multiple-free":
        return text.replace(
            "</worldbody>",
            '<body name="second-free"><freejoint/><geom size=".03" contype="0" conaffinity="0"/></body></worldbody>',
        )
    if case == "ball-same-body":
        return text.replace(
            '<joint name="ball"', '<joint name="pre-ball" type="slide"/><joint name="ball"'
        )
    if case == "tiny-y":
        return text.replace('gear="1 -2 .5"', 'gear="1 1e-45 .5"')
    if case == "tiny-z":
        return text.replace('gear="1 -2 .5"', 'gear="1 -2 1e-45"')
    raise AssertionError(case)


@pytest.mark.parametrize("case", tuple(name for name in NEGATIVES if name != "dry"))
def test_free_ball_outside_domain_precedes_device(case, monkeypatch):
    mj = _deps()
    text = negative_source(case)
    mj.MjModel.from_xml_string(text)
    compiled = compile_mujoco_xml(text, require_asset=False)
    import phymeo.backends._mujoco_tyx_runtime as implementation

    calls = []
    original = implementation.MuJoCoTyxRuntime._build_device

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("unsupported input allocated a device")

    monkeypatch.setattr(implementation.MuJoCoTyxRuntime, "_build_device", forbidden)
    with pytest.raises((MuJoCoTyxUnsupported, ValueError)) as caught:
        if case == "contact":
            # R3 admits public contacts; the bare floating compiler still needs opt-in.
            from phymeo.backends.mujoco_tyx._floating_model import compile_floating_model

            compile_floating_model(mj, mj.MjModel.from_xml_string(text))
        else:
            MuJoCoRuntime(compiled, implementation="tyx", mode="interpret")
    assert NEGATIVES[case] in str(caught.value)
    assert not calls
    monkeypatch.setattr(implementation.MuJoCoTyxRuntime, "_build_device", original)


@pytest.mark.parametrize("capacity", (None, [35, 45], (True, 10), (35, 46), (9, 10), (8, 10, 1)))
def test_joint_csr_rejects_unregistered_capacity(capacity):
    mj = _deps()
    from phymeo.backends.mujoco_tyx._joint_transmission import compile_joint_transmission

    model = mj.MjModel.from_xml_string(source("oblique"))
    with pytest.raises((MuJoCoTyxUnsupported, ValueError, TypeError), match="capacity"):
        compile_joint_transmission(mj, model, capacity=capacity)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("legacy", ("floating-ball", "fixed-free-ball", "motor-free"))
def test_exact_previous_free_ball_negative_now_has_native_successor(
    mode, legacy, monkeypatch, tmp_path
):
    from test_mujoco_tyx_ball import body as old_body
    from test_mujoco_tyx_ball import joint as old_joint
    from test_mujoco_tyx_ball import xml as old_xml
    from test_mujoco_tyx_ball_joint_motor import negative_source as old_motor
    from test_mujoco_tyx_floating import BALL_XML

    text = {
        "floating-ball": BALL_XML,
        "fixed-free-ball": old_xml(
            "free-and-ball",
            old_body("free", "<freejoint/>", old_body("ball-child", old_joint("ball"))),
        ),
        "motor-free": old_motor("free"),
    }[legacy]
    mj = _deps()
    model = mj.MjModel.from_xml_string(text)
    compiled = compile_mujoco_xml(text, require_asset=False)
    payload = dict(legacy=legacy, source=text, default_entry=True, steps=[])
    with MuJoCoRuntime(compiled, implementation="tyx", mode=mode) as runtime:
        key = runtime.world_keys[0]
        reference = mj.MjData(model)
        entry = runtime.integration_snapshot()[key]
        reference.qpos[:], reference.qvel[:], reference.time = entry.q, entry.v, entry.time
        packet = MuJoCoControl(
            np.zeros(model.nu),
            qfrc_applied=np.zeros(model.nv),
            xfrc_applied=np.zeros((model.nbody, 6)),
        )
        try:
            for tick in range(2):
                entry = runtime.integration_snapshot()[key]
                expected = native(mj, model, entry, packet)
                analytic = transmission_oracle(mj, model, entry, packet)
                record = dict(
                    tick=tick, entry=entry, native=expected, analytic=analytic, control=packet
                )
                payload["steps"].append(record)
                runtime.set_controls({key: packet})
                with monkeypatch.context() as patch:

                    def forbidden(*args, **kwargs):
                        raise AssertionError("native step in TYX")

                    patch.setattr(mj, "mj_step", forbidden)
                    runtime.step(H)
                state = runtime.integration_snapshot()[key]
                record["result"] = state
                compare(model, entry, state, expected, analytic)
                _set_control(reference, packet)
                mj.mj_step(model, reference)
                record["continuous_native"] = dict(
                    q=reference.qpos.copy(), v=reference.qvel.copy(), time=reference.time
                )
                np.testing.assert_allclose(state.q, reference.qpos, atol=Q_ATOL, rtol=0)
                np.testing.assert_allclose(state.v, reference.qvel, atol=V_ATOL, rtol=0)
                assert abs(state.time - reference.time) <= TIME_ATOL
        except BaseException as error:
            payload["failure"] = repr(error)
            raise
        finally:
            save(tmp_path / "legacy-positive.json.gz", payload)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("location", ("root", "child"))
@pytest.mark.parametrize("bad", ("zero", "nonunit"))
def test_free_ball_invalid_quaternion_rejected_before_device(mode, location, bad, monkeypatch):
    mj = _deps()
    text = source("mixed")
    model = mj.MjModel.from_xml_string(text)
    start = initial(mj, model)
    q = start.q.copy()
    qa = 3 if location == "root" else int(model.jnt_qposadr[1])
    q[qa : qa + 4] = (0, 0, 0, 0) if bad == "zero" else (2, 0, 0, 0)
    compiled = compile_mujoco_xml(text, require_asset=False)
    import phymeo.backends._mujoco_tyx_runtime as implementation

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("invalid quaternion allocated a device")

    monkeypatch.setattr(implementation.MuJoCoTyxRuntime, "_build_device", forbidden)
    with pytest.raises(ValueError, match="quaternion"):
        MuJoCoRuntime(
            compiled,
            implementation="tyx",
            mode=mode,
            initial_states={0: MuJoCoInitialState(q, start.v)},
        )
    assert not calls
