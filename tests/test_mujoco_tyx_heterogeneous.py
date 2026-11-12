"""G1 fixed heterogeneous sources and complete existing native oracles; tests first."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import test_mujoco_tyx_activation as scalar

from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml


def load_consumer(name):
    spec = importlib.util.spec_from_file_location(
        "heterogeneous_reference_" + name, Path(__file__).parent / "installed" / name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ball = load_consumer("check_ball_activation.py")
free = load_consumer("check_free_ball.py")
MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
H = 0.001
ORDER = (41, 7, 305, 11, 203, 59)
FAMILY = {41: "scalar", 7: "ball", 305: "free", 11: "scalar", 203: "ball", 59: "free"}
SLOT = {41: 0, 7: 0, 305: 0, 11: 1, 203: 1, 59: 1}


def sources():
    return {
        "scalar": scalar.source("coupled", early=True),
        "ball": ball.source("r4-pyramid"),
        "free": free.source("mixed"),
    }


def declarations(order=ORDER):
    import mujoco as mj

    assert mj.__version__ == "3.3.7"
    xml = sources()
    models = {name: mj.MjModel.from_xml_string(value) for name, value in xml.items()}
    compiled = {name: compile_mujoco_xml(value, require_asset=False) for name, value in xml.items()}
    specs = tuple(MuJoCoTyxModelSpec(name, compiled[name]) for name in xml)
    starts = {}
    for key in order:
        name, slot, model = FAMILY[key], SLOT[key], models[FAMILY[key]]
        if name == "scalar":
            starts[key] = MuJoCoInitialState(
                model.qpos0 + 0.06,
                np.full(model.nv, -0.035),
                act=np.full(model.na, 0.34 + slot * 0.01),
                time=0.125,
            )
        elif name == "ball":
            starts[key] = ball.initial(mj, model, "r4-pyramid", slot)
        else:
            starts[key] = free.initial(mj, model, slot)
    worlds = tuple(MuJoCoTyxWorldSpec(key, FAMILY[key], starts[key]) for key in order)
    return mj, xml, models, specs, worlds


def packets(models, tick, keys=ORDER):
    return {
        key: {"scalar": scalar.control, "ball": ball.control, "free": free.control}[FAMILY[key]](
            models[FAMILY[key]], tick, SLOT[key]
        )
        for key in keys
    }


def stored(packet):
    return MuJoCoControl(
        np.asarray(packet.ctrl, np.float32),
        qfrc_applied=np.asarray(packet.qfrc_applied, np.float32),
        xfrc_applied=np.asarray(packet.xfrc_applied, np.float32),
    )


def native_refs(mj, models, state):
    refs = {}
    for key, value in state.items():
        data = mj.MjData(models[FAMILY[key]])
        data.qpos[:], data.qvel[:], data.act[:], data.time = value.q, value.v, value.act, value.time
        refs[key] = data
    return refs


def scalar_native(mj, model, entry, packet):
    data = native_refs(mj, {"scalar": model}, {41: entry})[41]
    scalar.set_native(data, packet)
    mj.mj_forward(model, data)
    mass = np.empty((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    jac = np.zeros((model.nbody, 6, model.nv))
    for body in range(model.nbody):
        mj.mj_jac(model, data, jac[body, :3], jac[body, 3:], data.xipos[body], body)
    expected = dict(
        mass_matrix=mass,
        jacobian_com=jac,
        bias=data.qfrc_bias.copy(),
        passive=data.qfrc_passive.copy(),
        applied=data.qfrc_smooth.copy() - data.qfrc_passive + data.qfrc_bias,
        smooth_acceleration=data.qacc.copy(),
        scalar_force=data.actuator_force.copy(),
        derivative=data.act_dot.copy(),
        analytic=scalar.analytic(model, entry, packet),
    )
    mj.mj_step(model, data)
    expected["native_successor"] = free.native_state(data)
    return expected


def oracle(mj, models, entry, commands, refs):
    expected = {}
    for key, state in entry.items():
        name, model = FAMILY[key], models[FAMILY[key]]
        packet = stored(commands[key])
        if name == "scalar":
            expected[key] = scalar_native(mj, model, state, packet)
        elif name == "ball":
            expected[key] = ball.native_fields(mj, model, state, packet)
        else:
            expected[key] = free.native_fields(mj, model, state, packet)
        scalar.set_native(refs[key], packet)
        mj.mj_step(model, refs[key])
        expected[key]["continuous"] = free.native_state(refs[key])
    return expected


def compare(key, entry, after, command, expected, model):
    row = after.last_step
    assert row.world == key and row.tick == entry.tick + 1 == after.tick
    ball.exact(row.input_q, entry.q, "input q")
    ball.exact(row.input_v, entry.v, "input v")
    ball.exact(row.control, command, "requested control")
    ball.exact(row.stored_control, stored(command), "actual stored control")
    assert row.input_time == entry.time and row.output_time == after.time and row.h == H
    if FAMILY[key] == "ball":
        ball.compare(row, entry, after, command, expected, model)
        return
    if FAMILY[key] == "free":
        for name, (rtol, atol) in free.TOL.items():
            value = (
                row.integration_acceleration
                if name == "integration_acceleration"
                else getattr(row.dynamics, name)
            )
            assert value.shape == expected[name].shape and not value.flags.writeable
            np.testing.assert_allclose(value, expected[name], rtol=rtol, atol=atol)
        np.testing.assert_allclose(row.qacc, expected["smooth_acceleration"], rtol=2e-3, atol=4e-3)
        np.testing.assert_allclose(row.qfrc_actuator, expected["qfrc_actuator"], **free.FORCE)
        free.check_transmission(row, entry, expected, expected["transmission"])
        qtol, vtol = free.Q_ATOL, free.V_ATOL
    else:
        for name, (rtol, atol) in scalar.DYNAMICS_TOL.items():
            np.testing.assert_allclose(
                getattr(row.dynamics, name), expected[name], rtol=rtol, atol=atol
            )
        np.testing.assert_allclose(
            row.dynamics.jacobian_com, expected["jacobian_com"], rtol=3e-5, atol=4e-5
        )
        actual = row.actuation
        channels = (
            actual.output_act,
            actual.activation_derivative,
            actual.clamped_control,
            actual.force_input,
            actual.raw_force,
            actual.scalar_force,
            actual.generalized_force,
        )
        for index, (value, target) in enumerate(zip(channels, expected["analytic"], strict=True)):
            np.testing.assert_allclose(
                value,
                target,
                rtol=scalar.ACT_RTOL if index < 2 else scalar.FORCE_RTOL,
                atol=scalar.ACT_ATOL if index < 2 else scalar.FORCE_ATOL,
            )
        np.testing.assert_allclose(
            actual.scalar_force,
            expected["scalar_force"],
            rtol=scalar.FORCE_RTOL,
            atol=scalar.FORCE_ATOL,
        )
        np.testing.assert_array_equal(actual.generalized_force, row.qfrc_actuator)
        qtol, vtol = scalar.Q_ATOL, scalar.V_ATOL
    assert not row.contacts and not row.constraints
    np.testing.assert_array_equal(row.qfrc_constraint, np.zeros(model.nv))
    for target in (expected["native_successor"], expected["continuous"]):
        np.testing.assert_allclose(after.q, target["q"], rtol=0, atol=qtol)
        np.testing.assert_allclose(after.v, target["v"], rtol=0, atol=vtol)
        np.testing.assert_allclose(
            after.act, target["act"], rtol=scalar.ACT_RTOL, atol=scalar.ACT_ATOL
        )
        assert abs(after.time - target["time"]) <= scalar.TIME_ATOL


def advance(batch, mj, models, commands, refs, history, label):
    entry = batch.integration_snapshot()
    full_entry = batch.snapshot()
    expected = oracle(mj, models, entry, commands, refs)
    batch.set_controls(commands)
    attempt = dict(
        label=label, entry=entry, full_entry=full_entry, controls=commands, native=expected
    )
    history.append(attempt)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU physics used during TYX heterogeneous step")

    with patch.object(mj, "mj_step", forbidden), patch.object(mj, "mj_forward", forbidden):
        accepted = batch.step(H)
        attempt["raw_return"] = accepted
        output = batch.integration_snapshot()
        attempt["output"] = output
    attempt["full_output"], attempt["receipt"], attempt["counters"] = (
        batch.snapshot(),
        batch.last_step,
        batch.counters,
    )
    assert tuple(output) == tuple(entry) == batch.world_keys
    assert dict(accepted) == dict.fromkeys(batch.world_keys, True)
    assert tuple(batch.last_step.receipts) == batch.world_keys
    for key in batch.world_keys:
        compare(key, entry[key], output[key], commands[key], expected[key], models[FAMILY[key]])
        ball.exact(
            batch.last_step.receipts[key], output[key].last_step, "aggregate complete receipt"
        )
        for field in ("q", "v", "act", "ctrl", "time", "tick", "held_control", "last_step"):
            ball.exact(
                getattr(output[key], field), getattr(attempt["full_output"][key], field), field
            )
    return output


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "order", (ORDER, (59, 203, 11, 305, 7, 41)), ids=("interleaved", "reordered")
)
def test_three_models_force_motion_continuous(mode, order, tmp_path):
    mj, xml, models, specs, worlds = declarations(order)
    history = dict(xml=xml, specs=tuple(s.key for s in specs), worlds=worlds, mode=mode, steps=[])
    batch = MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode)
    try:
        assert batch.capabilities["bucket_count"] == 3
        assert {(r.nq, r.nv, r.nu, r.na) for r in batch.layout.values()} == {
            (2, 2, 4, 3),
            (models["ball"].nq, models["ball"].nv, models["ball"].nu, models["ball"].na),
            (13, 11, 3, 0),
        }
        refs = native_refs(mj, models, batch.integration_snapshot())
        for tick in range(24):
            advance(batch, mj, models, packets(models, tick, order), refs, history["steps"], tick)
        assert batch.counters["completed_steps"] == 24
        assert batch.counters["accepted_world_steps"] == 24 * len(order)
        assert batch.counters["bucket_physics_prepares"] == 72
        assert batch.counters["boundary_clones"] == 0
        assert any(np.any(s["output"][7].last_step.qfrc_constraint != 0) for s in history["steps"])
        assert any(np.any(s["output"][305].last_step.qfrc_actuator != 0) for s in history["steps"])
    except BaseException as error:
        history["failure"] = repr(error)
        raise
    finally:
        try:
            history["counters"] = batch.counters
        finally:
            batch.close()
            scalar.save(tmp_path / "heterogeneous-history.json.gz", history)


@pytest.mark.parametrize("mode", MODES)
def test_single_bucket_is_original_runtime(mode, tmp_path):
    mj, xml, models, specs, worlds = declarations((41, 11))
    chosen = (specs[0],)
    batch = MuJoCoTyxHeterogeneousBatch(chosen, worlds, mode=mode)
    reference = MuJoCoRuntime(
        chosen[0].compilation,
        world_keys=(41, 11),
        initial_states={w.world: w.initial_state for w in worlds},
        mode=mode,
        implementation="tyx",
    )
    history = []
    try:
        for tick in range(4):
            command = packets(models, tick, (41, 11))
            batch.set_controls(command)
            reference.set_controls(command)
            assert batch.step(H) == reference.step(H)
            got, want = batch.integration_snapshot(), reference.integration_snapshot()
            history.append(dict(command=command, got=got, want=want, receipt=batch.last_step))
            ball.exact(got, want, "same original whole-batch physics")
        assert (
            batch.counters["bucket_counters"][0]["completed_kernels"]
            == reference.counters["completed_kernels"]
        )
    finally:
        batch.close()
        reference.close()
        scalar.save(tmp_path / "single-bucket.json.gz", history)


@pytest.mark.parametrize("mode", MODES)
def test_complete_source_and_capacity_bucket_identity(mode, tmp_path):
    mj, xml, models, specs, worlds = declarations((41, 11))
    base = specs[0]
    alias = MuJoCoTyxModelSpec("alias", base.compilation)
    changed = MuJoCoTyxModelSpec(
        "changed",
        compile_mujoco_xml(xml["scalar"].replace('mass="1.1"', 'mass="1.2"'), require_asset=False),
    )
    cap = MuJoCoTyxModelSpec("capacity", base.compilation, contact_capacity=32)
    declared = (
        worlds[0],
        MuJoCoTyxWorldSpec(11, "alias", worlds[1].initial_state),
        MuJoCoTyxWorldSpec(71, "changed", worlds[0].initial_state),
        MuJoCoTyxWorldSpec(91, "capacity", worlds[1].initial_state),
    )
    with MuJoCoTyxHeterogeneousBatch((base, alias, changed, cap), declared, mode=mode) as batch:
        assert batch.capabilities["bucket_count"] == 3
        assert batch.layout[41].bucket == batch.layout[11].bucket
        assert len({batch.layout[key].bucket for key in (41, 71, 91)}) == 3
        controls = {key: scalar.control(models["scalar"], 0, 0) for key in batch.world_keys}
        batch.set_controls(controls)
        batch.step(H)
        scalar.save(
            tmp_path / "identity.json.gz",
            dict(layout=batch.layout, output=batch.integration_snapshot(), receipt=batch.last_step),
        )


@pytest.mark.parametrize(
    "bad",
    (
        "empty",
        "duplicate-model",
        "duplicate-world",
        "unknown-model",
        "unused-model",
        "world-bool",
        "too-many-models",
        "too-many-worlds",
        "different-h",
        "unsupported-last",
    ),
)
def test_declaration_rejected_before_device(bad, monkeypatch):
    mj, xml, models, specs, worlds = declarations((41, 11))
    chosen, declared = (specs[0],), worlds
    if bad == "empty":
        declared = ()
    elif bad == "duplicate-model":
        chosen = (specs[0], specs[0])
    elif bad == "duplicate-world":
        declared = (worlds[0], worlds[0])
    elif bad == "unknown-model":
        declared = (MuJoCoTyxWorldSpec(41, "missing", worlds[0].initial_state),)
    elif bad == "unused-model":
        chosen = specs[:2]
    elif bad == "world-bool":
        with pytest.raises((TypeError, ValueError)):
            MuJoCoTyxWorldSpec(True, "scalar")
        return
    elif bad == "too-many-models":
        chosen = tuple(MuJoCoTyxModelSpec(str(i), specs[0].compilation) for i in range(9))
        declared = tuple(MuJoCoTyxWorldSpec(i, str(i)) for i in range(9))
    elif bad == "too-many-worlds":
        declared = tuple(MuJoCoTyxWorldSpec(i, "scalar") for i in range(17))
    else:
        value = (
            xml["scalar"].replace('timestep="0.001"', 'timestep="0.002"')
            if bad == "different-h"
            else xml["free"].replace('dyntype="none"', 'dyntype="filterexact"')
        )
        if bad == "unsupported-last":
            value = (
                xml["free"]
                .replace("<motor ", '<general dyntype="filterexact" ')
                .replace("</motor>", "</general>")
            )
            # The qualified free source uses general NONE actors; change its explicit dynamics.
            value = value.replace('dyntype="none"', 'dyntype="filterexact"')
        extra = MuJoCoTyxModelSpec("last", compile_mujoco_xml(value, require_asset=False))
        chosen = (*chosen, extra)
        declared = (*declared, MuJoCoTyxWorldSpec(999, "last"))
    from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime

    called = []

    def forbidden(*args, **kwargs):
        called.append(True)
        raise AssertionError("device allocated before complete declaration validation")

    monkeypatch.setattr(MuJoCoTyxRuntime, "_build_device", forbidden)
    with pytest.raises((ValueError, TypeError)):
        MuJoCoTyxHeterogeneousBatch(chosen, declared, mode="interpret")
    assert not called
