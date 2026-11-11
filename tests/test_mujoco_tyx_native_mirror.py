"""Compact mirror equivalence/transactions, authored before product changes.

The native shadow retains the old copy-and-patch representation. It consumes
actual accepted TYX state, not a second solver or a numerical fixture search.
"""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from test_mujoco_tyx_activation import save
from test_mujoco_tyx_activation import source as activation_source
from test_mujoco_tyx_checkpoint import CASES, exact, initial, packet

from phymeo.backends import _mujoco_tyx_runtime as implementation
from phymeo.backends import mujoco_checkpoint as capsule
from phymeo.backends._mujoco_tyx_mirror import NativeMirror, NativeStateLayout
from phymeo.backends.mujoco_runtime import MuJoCoInitialState, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.resident_stepper import PhysicsStepFailure

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
CASES_TO_RUN = ("fixed-tree", "floating-tree", "floating-mixed", "activation")
KEYS = (4, 31)


def fixture(case, mode):
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("tyx")
    assert mj.__version__ == "3.3.7"
    xml = activation_source("coupled") if case == "activation" else CASES[case]
    compiled = compile_mujoco_xml(xml, require_asset=False)
    model = compiled.fresh_native_model()
    starts = {}
    for key in KEYS:
        base = initial(compiled, key)
        starts[key] = MuJoCoInitialState(
            base.q, base.v, time=base.time, act=np.linspace(0.05, 0.15, model.na)
        )
    engine = MuJoCoRuntime(compiled, mode=mode, world_keys=KEYS, initial_states=starts)
    return mj, model, compiled, starts, engine


def native_initial(mj, model, initial_state):
    data = mj.MjData(model)
    data.qpos[:] = initial_state.q
    data.qvel[:] = initial_state.v
    data.act[:] = initial_state.act
    data.time = initial_state.time
    mj.mj_forward(model, data)
    data.qpos[:] = np.asarray(data.qpos, np.float32)
    data.qvel[:] = np.asarray(data.qvel, np.float32)
    data.act[:] = np.asarray(data.act, np.float32)
    high = np.float32(data.time)
    low = np.float32(float(high) - data.time)
    data.time = float(high) - float(low)
    return data


def controls(compiled, tick):
    model = compiled.fresh_native_model()
    result = {}
    from phymeo.backends.mujoco_runtime import MuJoCoControl

    for key in KEYS:
        old = packet(compiled, tick, key)
        result[key] = MuJoCoControl(
            np.full(model.nu, old.ctrl[0]),
            qfrc_applied=old.qfrc_applied,
            xfrc_applied=old.xfrc_applied,
        )
    return result


def compare_native_snapshot(mj, model, native, state):
    endpoint = copy.copy(native)
    mj.mj_forward(model, endpoint)
    for published, native_name in (
        ("q", "qpos"),
        ("v", "qvel"),
        ("act", "act"),
        ("ctrl", "ctrl"),
        ("qfrc_applied", "qfrc_applied"),
        ("xfrc_applied", "xfrc_applied"),
    ):
        assert getattr(state, published).tobytes() == getattr(native, native_name).tobytes()
    for published, native_name in (
        ("body_positions", "xpos"),
        ("body_quaternions", "xquat"),
        ("geometry_positions", "geom_xpos"),
        ("geometry_matrices", "geom_xmat"),
        ("qacc", "qacc"),
        ("qfrc_actuator", "qfrc_actuator"),
        ("qfrc_passive", "qfrc_passive"),
        ("qfrc_constraint", "qfrc_constraint"),
        ("sensordata", "sensordata"),
        ("energy", "energy"),
    ):
        assert getattr(state, published).tobytes() == getattr(endpoint, native_name).tobytes()
    assert state.time == native.time and state.contact_time == native.time
    assert len(state.contacts) == endpoint.ncon
    for actual, expected in zip(state.contacts, endpoint.contact, strict=True):
        for published, native_name in (
            ("distance", "dist"),
            ("position", "pos"),
            ("frame", "frame"),
            ("friction", "friction"),
            ("solref", "solref"),
            ("solimp", "solimp"),
            ("solref_friction", "solreffriction"),
            ("include_margin", "includemargin"),
            ("regularized_mu", "mu"),
            ("cone_hessian", "H"),
        ):
            assert (
                np.asarray(getattr(actual, published)).tobytes()
                == np.asarray(getattr(expected, native_name)).tobytes()
            )
        for published, native_name in (
            ("geometry_ids", "geom"),
            ("flex_ids", "flex"),
            ("element_ids", "elem"),
            ("vertex_ids", "vert"),
        ):
            assert getattr(actual, published) == tuple(
                int(x) for x in getattr(expected, native_name)
            )
        assert actual.dimension == expected.dim and actual.excluded == expected.exclude
        assert actual.constraint_address == expected.efc_address
        wrench = np.empty(6, np.float64)
        mj.mj_contactForce(model, endpoint, actual.index, wrench)
        assert actual.wrench_contact_frame.tobytes() == wrench.tobytes()


@pytest.mark.parametrize("case", CASES_TO_RUN)
def test_public_native_state_roundtrip_preserves_warmstart_and_owned_bytes(case):
    mj = pytest.importorskip("mujoco")
    xml = activation_source("coupled") if case == "activation" else CASES[case]
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    data.qvel[:] = np.linspace(0.025, -0.015, model.nv)
    data.act[:] = np.linspace(0.1, 0.2, model.na)
    data.ctrl[:] = np.linspace(0.03, -0.02, model.nu)
    data.qfrc_applied[:] = np.linspace(0.01, 0.02, model.nv)
    data.xfrc_applied[-1] = [0.01, -0.02, 0.03, 0.004, -0.005, 0.006]
    data.qacc_warmstart[:] = np.linspace(0.3, -0.2, model.nv)
    data.time = 0.123
    mj.mj_forward(model, data)
    layout = NativeStateLayout(mj, model)
    mirror = layout.capture(data)
    before = mirror.cpu_state.tobytes()
    restored = mj.MjData(model)
    layout.materialize(mirror, restored)
    expected = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION))
    mj.mj_getState(model, restored, expected, mj.mjtState.mjSTATE_INTEGRATION)
    assert expected.tobytes() == before
    assert restored.qacc_warmstart.tobytes() == data.qacc_warmstart.tobytes()
    for name in capsule.MIRROR_FIELDS:
        assert getattr(restored, name).tobytes() == getattr(data, name).tobytes()
        assert not getattr(mirror, name).flags.writeable
    duplicate = copy.copy(mirror)
    assert duplicate is not mirror and duplicate.cpu_state.tobytes() == before
    data.qpos[:] += 0.01
    data.qacc_warmstart[:] = 0
    assert mirror.cpu_state.tobytes() == before
    with pytest.raises(ValueError):
        mirror.cpu_state.setflags(write=True)


@pytest.mark.parametrize("fault", ("layout", "nonfinite", "mirror-mismatch", "shape"))
def test_native_mirror_refuses_invalid_complete_state(fault):
    mj = pytest.importorskip("mujoco")
    model = mj.MjModel.from_xml_string(CASES["fixed-tree"])
    layout = NativeStateLayout(mj, model)
    mirror = layout.capture(mj.MjData(model))
    cpu_state = mirror.cpu_state.copy()
    values = {name: getattr(mirror, name).copy() for name in capsule.MIRROR_FIELDS}
    if fault == "nonfinite":
        cpu_state[-1] = np.inf
    elif fault == "mirror-mismatch":
        values["ctrl"][0] += 1
    elif fault == "shape":
        values["qacc"] = np.zeros(model.nv + 1)
    else:
        cpu_state = cpu_state[:-1]
    with pytest.raises(ValueError):
        layout.from_parts(cpu_state, values)


def test_native_observer_rejects_another_model_owner_before_mutation():
    mj = pytest.importorskip("mujoco")
    first = mj.MjModel.from_xml_string(CASES["fixed-tree"])
    second = mj.MjModel.from_xml_string(CASES["fixed-tree"])
    layout = NativeStateLayout(mj, first)
    mirror = layout.capture(mj.MjData(first))
    other = mj.MjData(second)
    other.qvel[:] = 0.125
    before = other.qvel.tobytes()
    with pytest.raises(ValueError, match="model identity"):
        layout.materialize(mirror, other)
    assert other.qvel.tobytes() == before


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES_TO_RUN)
def test_step_copy_elimination_full_snapshot_and_exact_continuation(
    case, mode, monkeypatch, tmp_path
):
    mj, model, compiled, starts, engine = fixture(case, mode)
    saved = []
    try:
        native = {key: native_initial(mj, model, starts[key]) for key in KEYS}
        held = engine.snapshot()
        held_signature = exact(held)
        observer_id = id(engine._observer_data)
        for tick in range(3):
            engine.set_controls(controls(compiled, tick))

            def forbidden(*args, **kwargs):
                raise AssertionError("ordinary TYX step allocated/copied native data")

            with monkeypatch.context() as patch:
                patch.setattr(implementation, "copy", SimpleNamespace(copy=forbidden))
                patch.setattr(mj, "MjData", forbidden)
                assert all(engine.step(engine.timestep).values())
            current = engine.integration_snapshot()
            assert all(type(value) is NativeMirror for value in engine._data.values())
            full = engine.snapshot()
            assert id(engine._observer_data) == observer_id
            for key in KEYS:
                # Independent old representation: copy all original native fields,
                # then patch only the same accepted fields the old step published.
                candidate = copy.copy(native[key])
                state = current[key]
                candidate.qpos[:] = state.q
                candidate.qvel[:] = state.v
                candidate.act[:] = state.act
                candidate.time = state.time
                for name in ("ctrl", "qfrc_applied", "xfrc_applied"):
                    getattr(candidate, name)[:] = getattr(state, name)
                for name in ("qacc", "qfrc_actuator", "qfrc_constraint"):
                    getattr(candidate, name)[:] = getattr(state.last_step, name)
                native[key] = candidate
                compare_native_snapshot(mj, model, candidate, full[key])
                vector = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION))
                mj.mj_getState(model, candidate, vector, mj.mjtState.mjSTATE_INTEGRATION)
                assert vector.tobytes() == engine._data[key].cpu_state.tobytes()
            saved.append(full)
        assert exact(held) == held_signature
        cp = engine.checkpoint()
        decoded = cp._read(cp.binding)
        for key, item in zip(KEYS, decoded["worlds"], strict=True):
            assert item["cpu_state"].tobytes() == engine._data[key].cpu_state.tobytes()
            for name in capsule.MIRROR_FIELDS:
                assert item["mirror"][name].tobytes() == getattr(native[key], name).tobytes()
        before = exact(engine.snapshot())
        assert all(engine.step(engine.timestep).values())
        continued = exact(engine.snapshot())
        engine.restore(cp, expected_revision=engine.checkpoint_revision)
        assert exact(engine.snapshot()) == before
        assert all(engine.step(engine.timestep).values())
        assert exact(engine.snapshot()) == continued
        with engine.clone() as cloned:
            assert cloned._observer_data is None
            assert exact(cloned.snapshot()) == exact(engine.snapshot())
            assert cloned._observer_data is not engine._observer_data
            assert all(cloned.step(cloned.timestep).values())
            assert exact(engine.snapshot()) == continued
        untouched = exact(engine.integration_snapshot()[KEYS[1]])
        engine.reset((KEYS[0],))
        assert exact(engine.integration_snapshot()[KEYS[1]]) == untouched
        assert all(engine.step(engine.timestep).values())
        save(
            tmp_path / "mirror-storage.json.gz",
            {
                "mode": mode,
                "case": case,
                "states": saved,
                "checkpoint": cp.to_bytes().hex(),
                "after_reset": engine.snapshot(),
            },
        )
    finally:
        engine.close()
    assert engine._observer_data is None and not engine._data


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fault", ("finite", "clock", "receipt", "mirror"))
def test_late_world_failure_preserves_all_accepted_records(mode, fault, monkeypatch):
    _, _, compiled, _, engine = fixture("fixed-tree", mode)
    try:
        engine.set_controls(controls(compiled, 0))
        assert all(engine.step(engine.timestep).values())
        before = exact(engine.snapshot())
        checkpoint = engine.checkpoint().to_bytes()
        revision, active = engine.checkpoint_revision, engine._device.active
        owners = dict(engine._data)
        with monkeypatch.context() as patch:
            if fault == "clock":
                download = engine._device.download

                def corrupt():
                    result = download()
                    result["time"][-1, 0] += np.float32(1)
                    return result

                patch.setattr(engine._device, "download", corrupt)
            else:
                target, name = (
                    (engine, "_finite_state")
                    if fault == "finite"
                    else (implementation, "MuJoCoTyxStepReceipt")
                    if fault == "receipt"
                    else (NativeMirror, "changed")
                )
                original = getattr(target, name)
                calls = 0

                def reject(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise MemoryError("authored last-world " + fault)
                    return original(*args, **kwargs)

                patch.setattr(target, name, reject)
            with pytest.raises(PhysicsStepFailure):
                engine.step(engine.timestep)
        assert engine.checkpoint_revision == revision and engine._device.active == active
        assert all(engine._data[key] is owners[key] for key in KEYS)
        assert engine.checkpoint().to_bytes() == checkpoint
        assert exact(engine.snapshot()) == before
        assert all(engine.step(engine.timestep).values())
    finally:
        engine.close()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("operation", ("snapshot", "restore"))
def test_last_observer_failure_is_detached_and_retryable(mode, operation, monkeypatch):
    mj, _, compiled, _, engine = fixture("activation", mode)
    try:
        engine.set_controls(controls(compiled, 0))
        assert all(engine.step(engine.timestep).values())
        cp = engine.checkpoint()
        before = exact(engine.snapshot())
        revision, active = engine.checkpoint_revision, engine._device.active
        original = mj.mj_forward
        calls = 0

        def failure(model, data):
            nonlocal calls
            calls += 1
            original(model, data)
            # Restore also validates two original initial declarations first.
            if calls == (4 if operation == "restore" else 2):
                raise RuntimeError("authored detached last-world observer failure")

        with monkeypatch.context() as patch:
            patch.setattr(mj, "mj_forward", failure)
            with pytest.raises(RuntimeError, match="detached last-world"):
                if operation == "restore":
                    engine.restore(cp, expected_revision=revision)
                else:
                    engine.snapshot()
        assert engine.checkpoint_revision == revision and engine._device.active == active
        assert engine.checkpoint().to_bytes() == cp.to_bytes()
        assert exact(engine.snapshot()) == before
        assert all(engine.step(engine.timestep).values())
    finally:
        engine.close()
