"""BALL-M1: predeclared direct JOINT moment mapping and real coupled trajectories."""

import numpy as np
import pytest
from test_mujoco_tyx_activation import save, set_native
from test_mujoco_tyx_ball import (
    ECCENTRIC,
    HINGE_BALL_SLIDE,
    MODES,
    Q_ATOL,
    TIME_ATOL,
    TOL,
    V_ATOL,
    H,
    assert_dynamics,
    body,
    deps,
    joint,
    xml,
)
from test_mujoco_tyx_ball import (
    initial as ball_initial,
)
from test_mujoco_tyx_ball_scalar_actuation import control, native_data, native_fields

from phymeo.backends.mujoco_runtime import MuJoCoInitialState, MuJoCoRuntime
from phymeo.backends.mujoco_tyx import MuJoCoTyxUnsupported
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

KEYS = (3, 29)
FORCE = dict(rtol=3e-4, atol=2e-4)
CASES = (
    "x",
    "y",
    "z",
    "oblique",
    "shared",
    "two-ball",
    "mixed",
    "clamps",
    "disabled-zero",
    "sign",
    "capacity32",
)
TWO_BALL = xml(
    "two-ball-branches",
    body(
        "base",
        joint("hinge", "hinge"),
        body("left", joint("ball0"), pos=".3 .1 .04")
        + body("right", joint("ball1"), pos="-.2 -.1 .08"),
    ),
)


def motor(name, target, gear, extra="", gain="1"):
    return (
        f'<general name="{name}" joint="{target}" gear="{gear}" '
        f'dyntype="none" gaintype="fixed" gainprm="{gain}" {extra}/>'
    )


def source(case):
    base = TWO_BALL if case == "two-ball" else (HINGE_BALL_SLIDE if case == "mixed" else ECCENTRIC)
    gear = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}.get(case, "1 -2 .5")
    actors = motor("ball-drive", "ball", gear)
    if case == "shared":
        actors += motor("counter", "ball", "-.25 .5 -.125", gain=".6")
        actors += motor("cross", "ball", ".2 .1 -.3", gain=".8")
    elif case == "two-ball":
        actors = motor("left-drive", "ball0", "1 -2 .5")
        actors += motor("right-drive", "ball1", "-.3 .2 .4")
        actors += motor("hinge-drive", "hinge", "-.5")
    elif case == "mixed":
        actors += motor("hinge-drive", "hinge", "-.5")
        actors += motor(
            "slide-drive", "slide", "-2", 'biastype="affine" biasprm=".01 -.3 -.04"', gain="1.2"
        )
    elif case == "clamps":
        actors = motor(
            "ball-drive",
            "ball",
            gear,
            'ctrllimited="true" ctrlrange="-.6 .6" forcelimited="true" forcerange="-.7 .7"',
            gain="2",
        )
    elif case == "disabled-zero":
        base = base.replace('integrator="Euler"', 'integrator="Euler" actuatorgroupdisable="1"')
        actors = motor(
            "disabled", "ball", gear, 'group="1" forcelimited="true" forcerange=".1 .25"'
        )
        actors += motor("zero", "ball", "0 0 0", gain=".8")
    elif case == "capacity32":
        actors = "".join(motor(f"actor-{i}", "ball", ".125 0 0", gain=".02") for i in range(32))
    return base.replace("</mujoco>", "<actuator>" + actors + "</actuator></mujoco>")


def fixture(case, mode):
    mj = deps()
    text = source(case)
    model = mj.MjModel.from_xml_string(text)
    starts = {}
    for slot, key in enumerate(KEYS):
        initial = ball_initial(mj, model, 0 if case == "sign" else slot)
        q = initial.q.copy()
        if case == "sign" and slot:
            q[:4] *= -1
        starts[key] = MuJoCoInitialState(q, initial.v, time=0.125)
    compiled = compile_mujoco_xml(text, require_asset=False)
    runtime = MuJoCoRuntime(
        compiled, mode=mode, implementation="tyx", world_keys=KEYS, initial_states=starts
    )
    return mj, model, compiled, starts, runtime


def transmission_oracle(mj, model, state, packet):
    """Independent native-address table and scalar-force law; no product helper."""
    offsets, columns, coefficients, ids = [0], [], [], []
    clamped, raw, scalar = [], [], []
    generalized = np.zeros(model.nv)
    moment = np.zeros((model.nu, model.nv))
    for a in range(model.nu):
        j = int(model.actuator_trnid[a, 0])
        kind = int(model.jnt_type[j])
        qa, dof = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        width = 3 if kind == int(mj.mjtJoint.mjJNT_BALL) else 1
        ids.append((a, j, kind, qa, dof, width))
        u = float(packet.ctrl[a])
        if model.actuator_ctrllimited[a]:
            u = float(np.clip(u, *model.actuator_ctrlrange[a]))
        value = 0.0
        if not model.opt.disableactuator & (1 << int(model.actuator_group[a])):
            value = float(model.actuator_gainprm[a, 0]) * u
            if model.actuator_biastype[a] == mj.mjtBias.mjBIAS_AFFINE:
                assert width == 1
                b = model.actuator_biasprm[a]
                gear = model.actuator_gear[a, 0]
                value += b[0] + b[1] * gear * state.q[qa] + b[2] * gear * state.v[dof]
        force = (
            float(np.clip(value, *model.actuator_forcerange[a]))
            if model.actuator_forcelimited[a]
            else value
        )
        clamped.append(u)
        raw.append(value)
        scalar.append(force)
        for axis in range(width):
            col, coefficient = dof + axis, float(model.actuator_gear[a, axis])
            columns.append(col)
            coefficients.append(coefficient)
            moment[a, col] = coefficient
            generalized[col] += coefficient * force
        offsets.append(len(columns))
    return dict(
        ids=np.asarray(ids, np.int32),
        gear=model.actuator_gear.copy(),
        row_offsets=np.asarray(offsets, np.int32),
        dof_indices=np.asarray(columns, np.int32),
        moment=np.asarray(coefficients, np.float32),
        dense_moment=moment,
        clamped_control=np.asarray(clamped),
        raw_force=np.asarray(raw),
        scalar_force=np.asarray(scalar),
        generalized_force=generalized,
    )


def compare(model, entry, state, native, analytic):
    row = state.last_step
    assert_dynamics(row, native)
    np.testing.assert_array_equal(row.input_q, entry.q)
    np.testing.assert_array_equal(row.input_v, entry.v)
    np.testing.assert_allclose(row.qfrc_actuator, native["qfrc_actuator"], **FORCE)
    np.testing.assert_array_equal(row.qfrc_constraint, np.zeros(model.nv))
    assert not row.contacts and state.act.shape == (0,)
    tr = row.transmission
    assert tr.schema == "native-joint-motor-transmission-v1"
    for key in ("ids", "gear", "row_offsets", "dof_indices", "moment"):
        np.testing.assert_array_equal(getattr(tr, key), analytic[key])
    for key in ("clamped_control", "raw_force", "scalar_force", "generalized_force"):
        np.testing.assert_allclose(getattr(tr, key), analytic[key], **FORCE)
    np.testing.assert_allclose(tr.scalar_force, native["scalar_force"], **FORCE)
    np.testing.assert_array_equal(tr.generalized_force, row.qfrc_actuator)
    np.testing.assert_array_equal(tr.input_control, row.control.ctrl)
    np.testing.assert_allclose(analytic["dense_moment"], native["dense_moment"], rtol=0, atol=2e-12)
    np.testing.assert_allclose(
        analytic["dense_moment"] @ entry.v, native["actuator_velocity"], rtol=3e-4, atol=2e-4
    )
    # Two independent contractions of the same complete physical power.
    np.testing.assert_allclose(
        tr.generalized_force @ entry.v,
        tr.scalar_force @ (analytic["dense_moment"] @ entry.v),
        **FORCE,
    )
    for key in (
        "ids",
        "gear",
        "row_offsets",
        "dof_indices",
        "moment",
        "clamped_control",
        "raw_force",
        "scalar_force",
        "generalized_force",
        "input_control",
    ):
        with pytest.raises(ValueError):
            getattr(tr, key).setflags(write=True)


def dense_native_moment(mj, model, data):
    dense = np.zeros((model.nu, model.nv))
    for a in range(model.nu):
        start, count = int(data.moment_rowadr[a]), int(data.moment_rownnz[a])
        dense[a, data.moment_colind[start : start + count]] = data.actuator_moment[
            start : start + count
        ]
    return dense


def compare_state(state, native):
    np.testing.assert_allclose(state.q, native.qpos, rtol=0, atol=Q_ATOL)
    np.testing.assert_allclose(state.v, native.qvel, rtol=0, atol=V_ATOL)
    assert abs(state.time - native.time) <= TIME_ATOL


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES)
def test_joint_motor_complete_24_steps(mode, case, monkeypatch, tmp_path):
    mj, model, _, starts, runtime = fixture(case, mode)
    history = dict(
        source=source(case),
        starts=starts,
        case=case,
        mode=mode,
        steps=[],
        dynamics_gates=TOL,
        force_gate=FORCE,
        h=H,
        q_atol=Q_ATOL,
        v_atol=V_ATOL,
    )
    try:
        assert runtime.capabilities["domain"] == "fixed-ball-joint-motor-no-contact-v1"
        initial = runtime.integration_snapshot()
        refs = {key: native_data(mj, model, initial[key]) for key in KEYS}
        for tick in range(24):
            entry = runtime.integration_snapshot()
            packets = {
                key: control(model, tick, 0 if case == "sign" else slot)
                for slot, key in enumerate(KEYS)
            }
            expected, same, analytical = {}, {}, {}
            for key in KEYS:
                same[key] = native_data(mj, model, entry[key])
                set_native(same[key], packets[key])
                expected[key] = native_fields(mj, model, same[key])
                expected[key]["dense_moment"] = dense_native_moment(mj, model, same[key])
                analytical[key] = transmission_oracle(mj, model, entry[key], packets[key])
                mj.mj_step(model, same[key])
                set_native(refs[key], packets[key])
                mj.mj_step(model, refs[key])
            runtime.set_controls(packets)
            history["pending"] = dict(
                tick=tick,
                entry=entry,
                control=packets,
                native=expected,
                analytic=analytical,
                same={k: (d.qpos.copy(), d.qvel.copy(), d.time) for k, d in same.items()},
                continuous={k: (d.qpos.copy(), d.qvel.copy(), d.time) for k, d in refs.items()},
            )
            count = runtime.counters["completed_kernels"]

            def forbidden(*args, **kwargs):
                raise AssertionError("actual TYX step called CPU dynamics")

            with monkeypatch.context() as patch:
                patch.setattr(mj, "mj_step", forbidden)
                patch.setattr(mj, "mj_forward", forbidden)
                assert runtime.step(H) == dict.fromkeys(KEYS, True)
                result = runtime.integration_snapshot()
            history["steps"].append(dict(history.pop("pending"), output=result))
            assert runtime.counters["completed_kernels"] == count + 3
            for key in KEYS:
                compare(model, entry[key], result[key], expected[key], analytical[key])
                compare_state(result[key], same[key])
                compare_state(result[key], refs[key])
                assert result[key].tick == tick + 1
            if case == "sign":
                np.testing.assert_allclose(result[3].q[:4], -result[29].q[:4], rtol=0, atol=Q_ATOL)
                np.testing.assert_allclose(result[3].v, result[29].v, rtol=0, atol=V_ATOL)
            if case == "disabled-zero":
                for state in result.values():
                    assert state.last_step.transmission.raw_force[0] == 0
                    np.testing.assert_allclose(
                        state.last_step.transmission.scalar_force[0], 0.1, rtol=0, atol=2e-8
                    )
                    np.testing.assert_array_equal(
                        state.last_step.transmission.moment[3:], np.zeros(3)
                    )
        assert len(history["steps"]) == 24
        assert any(np.any(result[k].v != initial[k].v) for k in KEYS)
    except BaseException as error:
        history["failure"] = repr(error)
        history["retained"] = runtime.integration_snapshot()
        raise
    finally:
        history["counters"] = dict(runtime.counters)
        runtime.close()
        save(tmp_path / "joint-motor.json.gz", history)


def negative_source(case):
    text = source("x")
    if case == "activation":
        return text.replace('dyntype="none"', 'dyntype="filterexact" dynprm=".004"')
    if case == "affine-ball":
        return text.replace('gainprm="1"', 'gainprm="1" biastype="affine" biasprm="0 -.2 -.1"')
    if case == "parent":
        return text.replace('joint="ball"', 'jointinparent="ball"')
    if case == "tail-gear":
        return text.replace('gear="1 0 0"', 'gear="1 0 0 .1 0 0"')
    if case == "tiny-y":
        return text.replace('gear="1 0 0"', 'gear="1 1e-45 0"')
    if case == "tiny-z":
        return text.replace('gear="1 0 0"', 'gear="1 0 1e-45"')
    if case == "contact":
        return text.replace("<worldbody>", '<worldbody><geom type="plane" size="1 1 .1"/>').replace(
            'contype="0" conaffinity="0"', 'contype="1" conaffinity="1"'
        )
    if case == "limit":
        return text.replace(
            'name="ball" type="ball"', 'name="ball" type="ball" limited="true" range="0 .5"'
        )
    if case == "dry":
        return text.replace('damping=".12"', 'damping=".12" frictionloss=".1"')
    if case == "spring":
        return text.replace('name="ball" type="ball"', 'name="ball" type="ball" stiffness=".2"')
    if case == "capacity":
        return source("capacity32").replace(
            "</actuator>", motor("over", "ball", "0 1 0") + "</actuator>"
        )
    if case == "free":
        return text.replace(
            "<worldbody>", '<worldbody><body name="free"><freejoint/><geom size=".1"/>'
        ).replace("</worldbody>", "</body></worldbody>")
    raise AssertionError(case)


NEGATIVES = {
    # Exact former activation source is qualified in test_mujoco_tyx_ball_activation.
    "affine-ball": "ball-actuator-domain",
    "parent": "ball-actuator-domain",
    "tail-gear": "actuator-domain",
    "limit": "ball-constraint-domain",
    "dry": "ball-constraint-domain",
    "spring": "ball-spring",
    "capacity": "ball-actuator-capacity",
    "free": "joint-domain",
    "tiny-y": "fp32-range",
    "tiny-z": "fp32-range",
}


@pytest.mark.parametrize("case", tuple(NEGATIVES))
def test_joint_motor_rejects_outside_domain_before_device(case, monkeypatch):
    deps()
    import phymeo.backends.mujoco_tyx._tree_device as device

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported input allocated a device")

    monkeypatch.setattr(device.TyxTreeDevice, "__init__", forbidden)
    with pytest.raises(MuJoCoTyxUnsupported) as caught:
        MuJoCoRuntime(
            compile_mujoco_xml(negative_source(case), require_asset=False),
            mode="interpret",
            implementation="tyx",
        )
    assert NEGATIVES[case] in {issue.code for issue in caught.value.issues}


def test_complete_immutable_csr_capacity_and_source_identity():
    mj = deps()
    from phymeo.backends.mujoco_tyx._joint_transmission import compile_joint_transmission

    model = mj.MjModel.from_xml_string(source("capacity32"))
    layout = compile_joint_transmission(mj, model)
    assert layout.ids.shape == (32, 6) and layout.gear.shape == (32, 6)
    np.testing.assert_array_equal(layout.row_offsets, np.arange(33, dtype=np.int32) * 3)
    np.testing.assert_array_equal(layout.dof_indices, np.tile(np.arange(3, dtype=np.int32), 32))
    assert layout.moment.shape == (96,)
    frozen = layout.gear.copy()
    model.actuator_gear[0, 0] = 9
    np.testing.assert_array_equal(layout.gear, frozen)
    for key in ("ids", "gear", "row_offsets", "dof_indices", "moment"):
        with pytest.raises(ValueError):
            getattr(layout, key).setflags(write=True)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("legacy", ("ball1-rotor", "ball-a-stateless"))
def test_previous_negative_exact_source_default_entry_now_drives(legacy, mode, tmp_path):
    # Exact source declarations retired by BALL-M1, rather than inferred coverage
    # from the new eccentric fixture with a different gear or initial state.
    from test_mujoco_tyx_ball import ROTOR
    from test_mujoco_tyx_ball_scalar_actuation import source as scalar_source

    mj = deps()
    if legacy == "ball1-rotor":
        text = ROTOR.replace(
            "</mujoco>", '<actuator><motor joint="ball" gear="1 2 3 0 0 0"/></actuator></mujoco>'
        )
    else:
        text = scalar_source("stateless").replace(
            '<motor name="direct" joint="hinge"', '<motor name="direct" joint="ball"'
        )
    model = mj.MjModel.from_xml_string(text)
    history = dict(legacy=legacy, source=text, steps=[])
    with MuJoCoRuntime(
        compile_mujoco_xml(text, require_asset=False), mode=mode, implementation="tyx"
    ) as runtime:
        reference = native_data(mj, model, runtime.integration_snapshot()[0])
        try:
            for tick in range(2):
                entry = runtime.integration_snapshot()[0]
                packet = control(model, tick)
                same = native_data(mj, model, entry)
                set_native(same, packet)
                expected = native_fields(mj, model, same)
                expected["dense_moment"] = dense_native_moment(mj, model, same)
                analytic = transmission_oracle(mj, model, entry, packet)
                mj.mj_step(model, same)
                set_native(reference, packet)
                mj.mj_step(model, reference)
                runtime.set_controls({0: packet})
                runtime.step(H)
                result = runtime.integration_snapshot()[0]
                history["steps"].append(
                    dict(
                        entry=entry,
                        packet=packet,
                        native=expected,
                        analytic=analytic,
                        output=result,
                        continuous_q=reference.qpos.copy(),
                        continuous_v=reference.qvel.copy(),
                        continuous_time=reference.time,
                    )
                )
                compare(model, entry, result, expected, analytic)
                compare_state(result, same)
                compare_state(result, reference)
        except BaseException as error:
            history["failure"] = repr(error)
            history["retained"] = runtime.integration_snapshot()
            raise
        finally:
            history["counters"] = dict(runtime.counters)
            save(tmp_path / "legacy-admission.json.gz", history)
