"""BALL-A scalar actuation on native BALL/R/P trees; tests-first, no new solver.

The frozen BALL1 geometry and gates are reused. Native same-entry mechanics and
an independently evolving CPU trajectory are separate from scalar filter math.
"""

import numpy as np
import pytest
from test_mujoco_tyx_activation import analytic, encode, save, set_native
from test_mujoco_tyx_ball import (
    HINGE_BALL_SLIDE,
    MAXIMUM,
    MODES,
    Q_ATOL,
    TIME_ATOL,
    TOL,
    V_ATOL,
    H,
    assert_dynamics,
    deps,
)
from test_mujoco_tyx_ball import (
    initial as ball_initial,
)

from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

ACT = dict(rtol=3e-5, atol=3e-6)
FORCE = dict(rtol=3e-4, atol=2e-4)
CASES = ("stateless", "filter-late", "filter-early", "disabled", "capacity")
STEPS = 24


def source(case):
    """Fixed declarations: original geometry, four scalar actors or capacity32."""
    base = MAXIMUM if case == "capacity" else HINGE_BALL_SLIDE
    stateful = case != "stateless"
    dtype = "filterexact" if stateful else "none"
    early = "true" if case in ("filter-early", "disabled", "capacity") else "false"
    actlimited = "true" if stateful else "false"
    force_range = ".1 .25" if case == "disabled" else "-.7 .7"
    if case == "capacity":
        actors = "".join(
            f'<general name="capacity-{i}" joint="{"hinge" if i % 2 else "slide"}" '
            f'gear="{-0.125 if i % 2 else 0.125}" dyntype="filterexact" dynprm=".004" '
            f'gainprm=".02" actearly="{early}"/>'
            for i in range(32)
        )
    else:
        actors = f'''<motor name="direct" joint="hinge" gear="-.5"/>
          <general name="filter-slide" joint="slide" gear="-2" dyntype="{dtype}"
           dynprm=".004" gainprm="1.2" biastype="affine" biasprm=".01 -.3 -.04"
           actearly="{early}" ctrllimited="true" ctrlrange="-.4 .4"
           actlimited="{actlimited}" actrange="-.2 .3"
           forcelimited="true" forcerange="-.7 .7"/>
          <general name="filter-hinge" joint="hinge" gear="1.5" group="1"
           dyntype="{dtype}" dynprm=".002" gainprm=".8"
           forcelimited="true" forcerange="{force_range}"/>
          <general name="shared-hinge" joint="hinge" gear="-.25"
           dyntype="{dtype}" dynprm=".008" gainprm=".6"/>'''
    if case == "disabled":
        base = base.replace('integrator="Euler"', 'integrator="Euler" actuatorgroupdisable="1"')
    return base.replace("</mujoco>", f"<actuator>{actors}</actuator></mujoco>")


def fixture(case, mode, keys=(7,)):
    mj = deps()
    text = source(case)
    model = mj.MjModel.from_xml_string(text)
    compiled = compile_mujoco_xml(text, require_asset=False)
    starts = {}
    for slot, key in enumerate(keys):
        original = ball_initial(mj, model, slot)
        starts[key] = MuJoCoInitialState(
            original.q,
            original.v,
            act=np.full(model.na, 0.34 + 0.01 * slot),
            time=0.125,
        )
    runtime = MuJoCoRuntime(
        compiled, mode=mode, implementation="tyx", world_keys=keys, initial_states=starts
    )
    return mj, model, compiled, starts, runtime


def control(model, tick, slot=0):
    # Predeclared 0..7 load, 8..15 reverse, 16..23 unload. No resets or warmup.
    command = 0.8 if tick < 8 else (-0.75 if tick < 16 else 0.0)
    u = np.full(model.nu, command + (0.015 * slot if command else 0.0), np.float32)
    sign = 1 if tick < 8 else -1
    force = np.array([sign * 0.03 * (i + 1) for i in range(model.nv)], np.float32)
    wrench = np.zeros((model.nbody, 6), np.float32)
    wrench[-1] = np.array([0.3, -0.2, 0.4, 0.01, -0.03, 0.02]) * (1 + 0.2 * slot)
    if tick >= 16:
        force.fill(0)
        wrench.fill(0)
    return MuJoCoControl(u, qfrc_applied=force, xfrc_applied=wrench)


def native_data(mj, model, state):
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.act[:], data.time = (state.q, state.v, state.act, state.time)
    return data


def native_fields(mj, model, data):
    mj.mj_forward(model, data)
    assert data.ncon == data.nefc == 0
    mass = np.empty((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    jac = np.zeros((model.nbody, 6, model.nv))
    applied = data.qfrc_applied.copy() + data.qfrc_actuator
    for b in range(model.nbody):
        mj.mj_jac(model, data, jac[b, :3], jac[b, 3:], data.xipos[b], b)
        applied += jac[b].T @ data.xfrc_applied[b]
    rhs = applied + data.qfrc_passive - data.qfrc_bias
    smooth = np.linalg.solve(mass, rhs)
    np.testing.assert_allclose(smooth, data.qacc_smooth, rtol=2e-11, atol=2e-10)
    return dict(
        input_q=data.qpos.copy(),
        input_v=data.qvel.copy(),
        input_act=data.act.copy(),
        time=float(data.time),
        ctrl=data.ctrl.copy(),
        qfrc_applied=data.qfrc_applied.copy(),
        xfrc_applied=data.xfrc_applied.copy(),
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
        scalar_force=data.actuator_force.copy(),
        qfrc_actuator=data.qfrc_actuator.copy(),
        derivative=data.act_dot.copy(),
        actuator_length=data.actuator_length.copy(),
        actuator_velocity=data.actuator_velocity.copy(),
    )


def compare_receipt(model, entry, state, expected, independent):
    row = state.last_step
    assert_dynamics(row, expected)
    np.testing.assert_array_equal(row.input_q, entry.q)
    np.testing.assert_array_equal(row.input_v, entry.v)
    assert row.input_time == entry.time
    np.testing.assert_allclose(row.qfrc_actuator, independent[-1], **FORCE)
    np.testing.assert_allclose(row.qfrc_actuator, expected["qfrc_actuator"], **FORCE)
    np.testing.assert_array_equal(row.qfrc_constraint, np.zeros(model.nv))
    assert row.contacts == ()
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == 1:  # Native BALL owns exactly three tangent columns.
            dof = int(model.jnt_dofadr[j])
            np.testing.assert_array_equal(row.qfrc_actuator[dof : dof + 3], np.zeros(3))
    if not model.na:
        assert type(row).__name__ == "MuJoCoTyxStepReceipt"
        assert not hasattr(row, "actuation")
        return
    a = row.actuation
    assert a.schema == "native-scalar-filterexact-actuation-v1"
    assert a.phase == "tyx-euler-entry-actuation-and-next-activation"
    np.testing.assert_array_equal(a.input_act, entry.act)
    np.testing.assert_array_equal(a.output_act, state.act)
    actuals = (
        a.output_act,
        a.activation_derivative,
        a.clamped_control,
        a.force_input,
        a.raw_force,
        a.scalar_force,
        a.generalized_force,
    )
    for channel, (actual, target) in enumerate(zip(actuals, independent, strict=True)):
        np.testing.assert_allclose(actual, target, **(ACT if channel < 2 else FORCE))
        with pytest.raises(ValueError):
            actual.setflags(write=True)
    np.testing.assert_allclose(a.scalar_force, expected["scalar_force"], **FORCE)
    np.testing.assert_allclose(a.activation_derivative, expected["derivative"], **ACT)
    np.testing.assert_array_equal(a.generalized_force, row.qfrc_actuator)
    integrated = independent[1].copy()
    for actor in range(model.nu):
        j = int(model.actuator_trnid[actor, 0])
        qa, dof, adr = map(
            int, (model.jnt_qposadr[j], model.jnt_dofadr[j], model.actuator_actadr[actor])
        )
        disabled = bool(model.opt.disableactuator & (1 << int(model.actuator_group[actor])))
        np.testing.assert_array_equal(
            a.ids[actor],
            [
                qa,
                dof,
                j,
                adr,
                int(model.actuator_actearly[actor]),
                int(disabled),
                int(model.actuator_ctrllimited[actor]),
                int(model.actuator_actlimited[actor]),
                int(model.actuator_forcelimited[actor]),
            ],
        )
        assert int(model.jnt_type[j]) in (2, 3)
        if adr >= 0 and disabled:
            integrated[adr] = 0
    np.testing.assert_allclose(a.integration_derivative, integrated, **ACT)


def compare_state(state, native):
    np.testing.assert_allclose(state.q, native.qpos, rtol=0, atol=Q_ATOL)
    np.testing.assert_allclose(state.v, native.qvel, rtol=0, atol=V_ATOL)
    np.testing.assert_allclose(state.act, native.act, **ACT)
    assert abs(state.time - native.time) <= TIME_ATOL


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES)
def test_ball_scalar_actuation_complete_24_steps(mode, case, monkeypatch, tmp_path):
    mj, model, _, starts, runtime = fixture(case, mode)
    history = dict(
        source=source(case),
        mode=mode,
        case=case,
        starts=starts,
        steps=[],
        declared_steps=STEPS,
        dynamics_gates=TOL,
        activation_gate=ACT,
        force_gate=FORCE,
        q_atol=Q_ATOL,
        v_atol=V_ATOL,
        time_atol=TIME_ATOL,
    )
    try:
        domain = (
            "fixed-ball-scalar-filterexact-no-contact-v1"
            if model.na
            else "fixed-ball-scalar-actuation-no-contact-v1"
        )
        assert runtime.capabilities["domain"] == domain
        if case == "capacity":
            assert (model.nq, model.nv, model.nu, model.na) == (10, 8, 32, 32)
        else:
            assert (model.nq, model.nv, model.nu) == (6, 5, 4)
        assert any(
            model.jnt_qposadr[j] != model.jnt_dofadr[j]
            for j in range(model.njnt)
            if int(model.jnt_type[j]) in (2, 3)
        )
        beginning = runtime.integration_snapshot()[7]
        reference = native_data(mj, model, beginning)
        for tick in range(STEPS):
            packet = control(model, tick)
            entry = runtime.integration_snapshot()[7]
            same = native_data(mj, model, entry)
            set_native(same, packet)
            expected = native_fields(mj, model, same)
            independent = analytic(model, entry, packet)
            mj.mj_step(model, same)
            set_native(reference, packet)
            mj.mj_step(model, reference)
            runtime.set_controls({7: packet})
            history["pending"] = dict(
                entry=entry,
                control=packet,
                native=expected,
                analytic=independent,
                same_output=(same.qpos.copy(), same.qvel.copy(), same.act.copy(), same.time),
                continuous=(
                    reference.qpos.copy(),
                    reference.qvel.copy(),
                    reference.act.copy(),
                    reference.time,
                ),
            )
            kernels = runtime.counters["completed_kernels"]

            def forbidden(*args, **kwargs):
                raise AssertionError("BALL-A TYX step called native dynamics")

            with monkeypatch.context() as patch:
                patch.setattr(mj, "mj_step", forbidden)
                patch.setattr(mj, "mj_forward", forbidden)
                assert runtime.step(H) == {7: True}
                state = runtime.integration_snapshot()[7]
            history["steps"].append(
                dict(
                    history.pop("pending"),
                    output=state,
                    receipt=state.last_step,
                )
            )
            assert runtime.counters["completed_kernels"] - kernels == (4 if model.na else 3)
            compare_receipt(model, entry, state, expected, independent)
            compare_state(state, same)
            compare_state(state, reference)
            assert state.tick == tick + 1
            assert state.last_step.world == 7 and state.last_step.tick == state.tick
            for name in ("ctrl", "qfrc_applied", "xfrc_applied"):
                for published in (
                    state.held_control,
                    state.last_step.control,
                    state.last_step.stored_control,
                ):
                    np.testing.assert_array_equal(getattr(published, name), getattr(packet, name))
            for j in range(model.njnt):
                if int(model.jnt_type[j]) == 1:
                    qa = int(model.jnt_qposadr[j])
                    quat = state.q[qa : qa + 4]
                    assert abs(float(quat @ quat) - 1) <= 8 * np.finfo(np.float32).eps
            if case == "disabled":
                a = state.last_step.actuation
                assert a.raw_force[2] == 0 and a.integration_derivative[1] == 0
                np.testing.assert_allclose(a.scalar_force[2], 0.1, rtol=0, atol=2e-8)
                assert a.activation_derivative[1] != 0
            if tick in (0, 7, 8, 15, 16, 23):
                full = runtime.snapshot()[7]
                for name in ("q", "v", "act", "ctrl"):
                    np.testing.assert_array_equal(getattr(state, name), getattr(full, name))
                assert (state.time, state.tick) == (full.time, full.tick)
                assert encode(state.last_step) == encode(full.last_step)
        for j in range(model.njnt):
            if int(model.jnt_type[j]) == 1:
                dof = int(model.jnt_dofadr[j])
                assert np.any(state.v[dof : dof + 3] != beginning.v[dof : dof + 3])
        if model.na:
            assert np.any(history["steps"][16]["output"].act != 0)
        assert len(history["steps"]) == STEPS
    except BaseException as error:
        history["failure"] = repr(error)
        history["published_on_failure"] = runtime.integration_snapshot()
        raise
    finally:
        history["counters"] = dict(runtime.counters)
        runtime.close()
        save(tmp_path / "ball-scalar-continuous.json.gz", history)


# This is the exact former BALL1 scalar-motor negative, with its default entry.
LEGACY_SCALAR_SOURCE = HINGE_BALL_SLIDE.replace(
    "</mujoco>", '<actuator><motor joint="hinge"/></actuator></mujoco>'
)


@pytest.mark.parametrize("mode", MODES)
def test_original_ball_scalar_motor_default_entry_is_admitted(mode, tmp_path):
    mj = deps()
    model = mj.MjModel.from_xml_string(LEGACY_SCALAR_SOURCE)
    history = dict(source=LEGACY_SCALAR_SOURCE, mode=mode, steps=[])
    runtime = MuJoCoRuntime(
        compile_mujoco_xml(LEGACY_SCALAR_SOURCE, require_asset=False),
        mode=mode,
        implementation="tyx",
    )
    try:
        reference = native_data(mj, model, runtime.integration_snapshot()[0])
        for tick in range(2):
            entry = runtime.integration_snapshot()[0]
            packet = entry.held_control
            same = native_data(mj, model, entry)
            set_native(same, packet)
            expected = native_fields(mj, model, same)
            independent = analytic(model, entry, packet)
            mj.mj_step(model, same)
            set_native(reference, packet)
            mj.mj_step(model, reference)
            history["pending"] = dict(
                entry=entry, control=packet, native=expected, analytic=independent
            )
            count = runtime.counters["completed_kernels"]
            assert runtime.step(H) == {0: True}
            state = runtime.integration_snapshot()[0]
            history["steps"].append(
                dict(
                    history.pop("pending"),
                    output=state,
                    same_output=(same.qpos.copy(), same.qvel.copy()),
                    continuous=(reference.qpos.copy(), reference.qvel.copy()),
                )
            )
            assert runtime.counters["completed_kernels"] - count == 3
            compare_receipt(model, entry, state, expected, independent)
            compare_state(state, same)
            compare_state(state, reference)
            assert state.tick == tick + 1
    except BaseException as error:
        history["failure"] = repr(error)
        raise
    finally:
        history["counters"] = dict(runtime.counters)
        runtime.close()
        save(tmp_path / "ball-scalar-original-entry.json.gz", history)


def negative_source(feature):
    text = source("filter-early")
    if feature == "ball-target-none":
        return source("stateless").replace(
            '<motor name="direct" joint="hinge"', '<motor name="direct" joint="ball"'
        )
    if feature == "ball-target-filter":
        return text.replace('name="filter-slide" joint="slide"', 'name="filter-slide" joint="ball"')
    if feature == "free-target":
        text = text.replace(
            '<motor name="direct" joint="hinge"', '<motor name="direct" joint="free"'
        )
        return text.replace(
            "<worldbody>",
            '<worldbody><body name="free-owner"><freejoint name="free"/><inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/>',
        ).replace("</worldbody>", "</body></worldbody>")
    if feature == "contact-condim4":
        return text.replace(
            "<worldbody>", '<worldbody><geom type="plane" size="1 1 .1" condim="4"/>'
        ).replace('contype="0" conaffinity="0"', 'contype="1" conaffinity="1"')
    if feature in ("scalar-limit", "scalar-dry"):
        extra = 'limited="true" range="-1 1"' if feature == "scalar-limit" else 'frictionloss=".1"'
        return text.replace('name="hinge" type="hinge"', f'name="hinge" type="hinge" {extra}')
    if feature in ("ball-spring", "ball-limit"):
        extra = 'stiffness=".1"' if feature == "ball-spring" else 'limited="true" range="0 .5"'
        return text.replace('name="ball" type="ball"', f'name="ball" type="ball" {extra}')
    if feature == "same-body":
        return text.replace(
            '<joint name="ball"', '<joint name="extra" type="slide"/><joint name="ball"'
        )
    if feature == "joint-in-parent":
        return text.replace(
            'name="filter-slide" joint="slide"', 'name="filter-slide" jointinparent="slide"'
        )
    if feature in ("filter", "integrator"):
        return text.replace('dyntype="filterexact"', f'dyntype="{feature}"')
    if feature == "capacity":
        extra = '<general joint="slide" dyntype="filterexact" dynprm=".004"/>'
        return source("capacity").replace("</actuator>", extra + "</actuator>")
    raise AssertionError(feature)


NEGATIVES = (
    "ball-target-filter",
    "free-target",
    "contact-condim4",
    "scalar-limit",
    "scalar-dry",
    "ball-spring",
    "ball-limit",
    "same-body",
    "joint-in-parent",
    "filter",
    "integrator",
    "capacity",
)


@pytest.mark.parametrize("feature", NEGATIVES)
def test_ball_scalar_unsupported_domains_reject_before_device(feature, monkeypatch):
    deps()
    from phymeo.backends.mujoco_tyx import MuJoCoTyxUnsupported, _tree_device

    compiled = compile_mujoco_xml(negative_source(feature), require_asset=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported BALL-A model allocated a device")

    monkeypatch.setattr(_tree_device.TyxTreeDevice, "__init__", forbidden)
    with pytest.raises(MuJoCoTyxUnsupported):
        MuJoCoRuntime(compiled, mode="interpret", implementation="tyx")
