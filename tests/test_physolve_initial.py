"""Complete initialization/reset and world isolation on the frozen native Host."""

from dataclasses import replace

import numpy as np
import pytest
from test_scene_compiler import H, _world

from phymeo import PhySolveRuntime, RobotCommand, RobotState
from phymeo.backends.physolve_initial import INTEGRATION_FIELDS, NativeInitialState
from phymeo.builders.mujoco_scene import MuJoCoWorldProfile, compile_mujoco_world

pytest_plugins = ["test_physolve_runtime"]


def compiled():
    world = replace(
        _world(),
        initial_state=RobotState(np.array([0.125]), np.array([0.25])),
        kp=np.array([2.0]),
        kd=np.array([0.5]),
        initial_control=RobotCommand(np.array([0.375]), target_velocity=np.array([0.125])),
    )
    return compile_mujoco_world(
        world, instance="arm", profile=MuJoCoWorldProfile(H, (0.0, 0.0, 0.0), True)
    )


def test_initial_snapshot_is_owned_and_model_bound():
    record = compiled()
    model, data = record.fresh_native_data()
    initial = NativeInitialState.from_data(record.compilation, data)
    data.qpos[:] = 0
    initial.apply(record.compilation, data)
    np.testing.assert_array_equal(data.qpos, record.initial_qpos)
    with pytest.raises(ValueError):
        dict(initial.fields)["qpos"].flags.writeable = True
    with pytest.raises(ValueError, match="another compiled"):
        replace(initial, model_sha256="wrong").apply(record.compilation, data)
    bad = replace(
        initial,
        fields=tuple((k, np.full(v.shape, 1e100) if k == "ctrl" else v) for k, v in initial.fields),
    )
    with pytest.raises(ValueError, match="float32"):
        bad.apply(record.compilation, data)


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_compiled_initial_controls_reset_and_replay(payload, nworld, monkeypatch):
    import mujoco

    record = compiled()
    qpos0 = record.compilation.fresh_native_model().qpos0.copy()
    assert not np.array_equal(qpos0, record.initial_qpos)
    with PhySolveRuntime(record, payload=payload, nworld=nworld) as runtime:
        initial = runtime.snapshot()
        for name, expected in (
            ("qpos", record.initial_qpos),
            ("qvel", record.initial_qvel),
            ("ctrl", record.initial_ctrl),
        ):
            np.testing.assert_array_equal(
                initial[name], np.broadcast_to(expected.astype(np.float32), initial[name].shape)
            )

        def forbidden(*args, **kwargs):
            raise AssertionError("no CPU forward or step after preparation")

        monkeypatch.setattr(mujoco, "mj_step", forbidden)
        monkeypatch.setattr(mujoco, "mj_forward", forbidden)
        forces = np.arange(1, nworld + 1, dtype=np.float32)[:, None] / nworld
        control = np.full((nworld, 1), 0.5, dtype=np.float32)
        for _ in range(4):
            runtime.step(ctrl=control, qfrc_applied=forces)
        after = runtime.snapshot()
        runtime.forward()
        for name in ("qpos", "qvel", "ctrl", "qfrc_applied", "time"):
            np.testing.assert_array_equal(runtime.snapshot()[name], after[name])
        mask = np.arange(nworld) % 2 == 0
        integration = {
            name: getattr(runtime._banks[runtime._active], name).numpy().copy()
            for name in INTEGRATION_FIELDS
        }
        runtime.reset(mask)
        reset = runtime.snapshot()
        for name in ("qpos", "qvel", "ctrl", "qfrc_applied", "time"):
            np.testing.assert_array_equal(reset[name][mask], initial[name][mask])
            np.testing.assert_array_equal(reset[name][~mask], after[name][~mask])
        for name, before in integration.items():
            actual = getattr(runtime._banks[runtime._active], name).numpy()
            np.testing.assert_array_equal(actual[~mask], before[~mask], err_msg=name)
        runtime.reset()
        for _ in range(4):
            runtime.step(ctrl=control, qfrc_applied=forces)
        for name in ("qpos", "qvel", "ctrl", "qfrc_applied", "time"):
            np.testing.assert_array_equal(runtime.snapshot()[name], after[name])
        np.testing.assert_array_equal(runtime._cpu_model.qpos0, qpos0)


@pytest.mark.cuda
def test_external_force_is_not_actuator_control(payload):
    record = compiled()
    model, data = record.fresh_native_data()
    data.qfrc_applied[:] = 250
    initial = NativeInitialState.from_data(record.compilation, data)
    with PhySolveRuntime(record.compilation, payload=payload, initial_state=initial) as runtime:
        before = runtime.snapshot()
        np.testing.assert_array_equal(before["qfrc_applied"], [[250]])
        runtime.step(ctrl=np.zeros((1, 1)), qfrc_applied=np.zeros((1, 1)))
        runtime.reset()
        for name in ("qpos", "qvel", "ctrl", "qfrc_applied"):
            np.testing.assert_array_equal(runtime.snapshot()[name], before[name])


def test_bad_quaternion_is_rejected_before_mutation():
    import mujoco
    from test_physolve_runtime import FREE

    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    record = compile_mujoco_xml(FREE, require_asset=False)
    model = record.fresh_native_model()
    data = mujoco.MjData(model)
    data.qpos[3:7] = [2, 0, 0, 0]
    initial = NativeInitialState.from_data(record, data)
    target = mujoco.MjData(model)
    qpos = target.qpos.copy()
    with pytest.raises(ValueError, match="unit WXYZ"):
        initial.apply(record, target)
    np.testing.assert_array_equal(target.qpos, qpos)


@pytest.mark.cuda
def test_prepared_reset_nonempty_bool_history_and_packed_fields(monkeypatch):
    """Device copy contract; not a claim of Host dynamics/equality qualification."""
    from contextlib import closing
    from types import SimpleNamespace

    import tyx as tx

    from phymeo.backends._physolve_reset import PreparedReset

    n = 8
    shapes = {
        "time": (n,),
        "qpos": (n, 7),
        "qvel": (n, 6),
        "act": (n, 2),
        "qacc_warmstart": (n, 6),
        "ctrl": (n, 2),
        "qfrc_applied": (n, 6),
        "xfrc_applied": (n, 3, 6),
        "eq_active": (n, 2),
        "mocap_pos": (n, 2, 3),
        "mocap_quat": (n, 2, 4),
    }
    with closing(tx.CudaStream(device=0)) as stream, tx.use_stream(stream):
        rows, expected = [], {}
        for bank in range(2):
            fields = {}
            for name, shape in shapes.items():
                value = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1
                if name == "eq_active":
                    value = value.astype(np.int32) % 2 == 0
                dtype = {
                    "eq_active": tx.bool1,
                    "mocap_pos": tx.vec3f,
                    "mocap_quat": tx.vec4f,
                    "xfrc_applied": tx.vector(6, tx.f32),
                }.get(name, tx.f32)
                expected[name] = value.copy()
                fields[name] = tx.Array(
                    value if bank == 0 else np.zeros_like(value), dtype=dtype, device="cuda:0"
                )
            rows.append(SimpleNamespace(**fields))
        prepared = PreparedReset(rows, n, tx.get_device("cuda:0"))
        prepared.prepare(stream)
        ids = tuple(id(plan) for plans in prepared.plans for plan in plans)

        def forbidden(*args, **kwargs):
            raise AssertionError("reset may not allocate another state clone")

        monkeypatch.setattr(tx, "clone", forbidden)
        mask = np.arange(n) % 2 == 0
        prepared.restore(1, mask, stream)
        for name, wanted in expected.items():
            actual = getattr(rows[1], name).numpy(stream=stream)
            np.testing.assert_array_equal(actual[mask], wanted[mask], err_msg=name)
            np.testing.assert_array_equal(actual[~mask], np.zeros_like(wanted[~mask]), err_msg=name)
        prepared.restore(1, None, stream)
        for name, wanted in expected.items():
            np.testing.assert_array_equal(getattr(rows[1], name).numpy(stream=stream), wanted)
        assert ids == tuple(id(plan) for plans in prepared.plans for plan in plans)
