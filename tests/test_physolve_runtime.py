"""Actual frozen native Host; optional explicit local payload, never a mock solver."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from phymeo import PhySolvePayload, PhySolveRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.builders.mujoco_compiler import compile_mujoco


FREE = '''<mujoco><option timestep=".002" integrator="Euler"/>
<worldbody><body pos="0 0 1"><freejoint/>
<geom type="sphere" size=".1" mass="2"/></body></worldbody></mujoco>'''
SLIDER = '''<mujoco><option timestep=".002" gravity="0 0 0"/>
<worldbody><body><joint type="slide" axis="1 0 0"/>
<geom type="sphere" size=".1" mass="2"/></body></worldbody></mujoco>'''
CONTACT = '''<mujoco><option timestep=".002" iterations="10" ls_iterations="10"/>
<worldbody><geom name="ground" type="plane" size="0 0 1"/>
<body pos="0 0 .08"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
</worldbody></mujoco>'''


@pytest.fixture(scope="module")
def payload():
    path = os.environ.get("PHYMEO_PHYSOLVE_SOURCES")
    if not path:
        pytest.skip("explicit PHYMEO_PHYSOLVE_SOURCES required for native qualification")
    root = Path(path).resolve().parent
    sources = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    frozen = json.loads((root / "evidence/payload-baseline.json").read_text(encoding="utf-8-sig"))
    return PhySolvePayload(sources["native_host"], sources["cubins"],
                          frozen["native_host"]["sha256"],
                          tuple((row["name"], row["sha256"]) for row in frozen["cubins"]))


def test_runtime_requires_compilation_and_explicit_payload():
    with pytest.raises(TypeError, match="compilation|Compilation"):
        PhySolveRuntime(None, payload=None)


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_freefall_forward_reset_and_masked_worlds(payload, nworld, monkeypatch):
    import mujoco
    record = compile_mujoco_xml(FREE, require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=nworld) as runtime:
        initial = runtime.snapshot()
        def forbidden(*args, **kwargs):
            raise AssertionError("CPU simulation/forward after preparation is forbidden")
        monkeypatch.setattr(mujoco, "mj_step", forbidden)
        monkeypatch.setattr(mujoco, "mj_forward", forbidden)
        for _ in range(10):
            runtime.step()
        after = runtime.snapshot()
        np.testing.assert_allclose(after["qpos"][:, 2], 1 - 9.81 * .002**2 * 55,
                                   atol=1e-6, rtol=0)
        np.testing.assert_allclose(after["qvel"][:, 2], -9.81 * .002 * 10,
                                   atol=1e-6, rtol=0)
        runtime.forward()
        forward = runtime.snapshot()
        for field in ("qpos", "qvel", "time"):
            np.testing.assert_array_equal(forward[field], after[field])
        mask = np.arange(nworld) % 2 == 0
        runtime.reset(mask)
        reset = runtime.snapshot()
        for field in ("qpos", "qvel", "time"):
            np.testing.assert_array_equal(reset[field][mask], initial[field][mask])
            np.testing.assert_array_equal(reset[field][~mask], after[field][~mask])
        runtime.reset()
        for _ in range(10):
            runtime.step()
        repeated = runtime.snapshot()
        for field in ("qpos", "qvel", "time"):
            np.testing.assert_array_equal(repeated[field], after[field])
        with pytest.raises(ValueError):
            repeated["qpos"].flags.writeable = True
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.step()


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_joint_force_control_atomicity_failure_recovery(payload, nworld, monkeypatch):
    record = compile_mujoco_xml(SLIDER, require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=nworld) as runtime:
        forces = np.linspace(-2, 2, nworld, dtype=np.float32)[:, None]
        for _ in range(10):
            runtime.step(qfrc_applied=forces)
        after = runtime.snapshot()
        np.testing.assert_allclose(after["qpos"][:, 0], forces[:, 0] / 2 * .002**2 * 55,
                                   atol=1e-6, rtol=0)
        np.testing.assert_array_equal(after["qfrc_applied"], forces)
        with pytest.raises(ValueError, match="float32"):
            runtime.step(qfrc_applied=np.full((nworld, 1), np.inf))
        with pytest.raises(ValueError, match="mask"):
            runtime.reset(np.zeros(nworld, dtype=np.int32))
        for field in after:
            np.testing.assert_array_equal(runtime.snapshot()[field], after[field])
        original = runtime._validate
        def rejected(data):
            original(data)
            raise RuntimeError("injected post-execution publication failure")
        monkeypatch.setattr(runtime, "_validate", rejected)
        with pytest.raises(RuntimeError, match="publication failure"):
            runtime.step()
        for field in after:
            np.testing.assert_array_equal(runtime.snapshot()[field], after[field])
        monkeypatch.setattr(runtime, "_validate", original)
        runtime.reset()
        runtime.step(qfrc_applied=forces)
        np.testing.assert_allclose(runtime.snapshot()["qpos"][:, 0],
                                   forces[:, 0] / 2 * .002**2, atol=1e-6, rtol=0)


@pytest.mark.cuda
def test_contact_native_response_and_capacity_preflight(payload):
    record = compile_mujoco_xml(CONTACT, require_asset=False)
    with pytest.raises(ValueError, match="capacity"):
        PhySolveRuntime(record, payload=payload, njmax=1)
    with PhySolveRuntime(record, payload=payload, nworld=8) as runtime:
        initial = runtime.snapshot()
        assert np.all(initial["qfrc_constraint"][:, 2] > 9.81)
        runtime.step()
        assert np.all(runtime.snapshot()["qpos"][:, 2] > initial["qpos"][:, 2])
        runtime.reset()
        repeated = runtime.snapshot()
        for field in ("qpos", "qvel", "qacc", "qfrc_constraint"):
            np.testing.assert_array_equal(repeated[field], initial[field])


@pytest.mark.cuda
@pytest.mark.parametrize("name", ["panda", "fanuc"])
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_robot_dual_bank_matches_native_single_bank(payload, name, nworld):
    import mujoco
    from test_physolve_models import robot_path
    record = compile_mujoco(robot_path(name), require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=nworld, nconmax=128) as runtime:
        initial = runtime.snapshot()
        # Independent persistent native bank is the reference for history/state
        # migration, using exactly the same frozen CUBINs and frontend model.
        with runtime._scope():
            cpu = runtime._cpu_model
            data = mujoco.MjData(cpu)
            mujoco.mj_forward(cpu, data)
            reference = runtime._frontend.put_data(cpu, data, nworld=nworld,
                                                   nconmax=128, njmax=256)
            handle = runtime._native._create_runtime(runtime._model, reference, "tyx", "tyx")
            try:
                runtime._native._prepare_runtime(handle)
                runtime._native._forward_runtime(handle)
                runtime._complete()
                for field in initial:
                    np.testing.assert_allclose(initial[field], getattr(reference, field).numpy(),
                                               rtol=0, atol=1e-6,
                                               err_msg=f"initial forward: {field}")
                forces = np.broadcast_to(np.linspace(-.05, .05, cpu.nv, dtype=np.float32),
                                         (nworld, cpu.nv)).copy()
                forces *= np.arange(1, nworld + 1, dtype=np.float32)[:, None] / nworld
                for step_index in range(6):
                    reference.qfrc_applied.assign(forces, stream=runtime._stream)
                    runtime._native._step_runtime(handle)
                    runtime.step(qfrc_applied=forces)
                    actual = runtime.snapshot()
                    for field in actual:
                        np.testing.assert_allclose(actual[field], getattr(reference, field).numpy(),
                                                   rtol=0, atol=1e-6,
                                                   err_msg=f"step {step_index + 1}: {field}")
                before = runtime.snapshot()
                assert np.any(before["qvel"] != initial["qvel"])
                np.testing.assert_allclose(before["time"], 6 * cpu.opt.timestep,
                                           atol=1e-6, rtol=0)
                runtime.forward()
                for field in ("qpos", "qvel", "time"):
                    np.testing.assert_array_equal(runtime.snapshot()[field], before[field])
                mask = np.arange(nworld) % 2 == 0
                runtime.reset(mask)
                after = runtime.snapshot()
                for field in ("qpos", "qvel", "time", "qfrc_applied"):
                    np.testing.assert_array_equal(after[field][mask], initial[field][mask])
                    np.testing.assert_array_equal(after[field][~mask], before[field][~mask])
                runtime.reset()
                for _ in range(6):
                    runtime.step(qfrc_applied=forces)
                for field in ("qpos", "qvel", "time"):
                    np.testing.assert_array_equal(runtime.snapshot()[field], before[field])
            finally:
                runtime._complete()
                runtime._native._close_runtime(handle, True)


@pytest.mark.cuda
def test_close_failure_retains_owner_for_retry(payload, monkeypatch):
    runtime = PhySolveRuntime(compile_mujoco_xml(SLIDER, require_asset=False), payload=payload)
    close = runtime._native._close_runtime
    calls = []
    def fail_second(handle, complete):
        calls.append(handle)
        if len(calls) == 2:
            raise RuntimeError("injected close failure")
        return close(handle, complete)
    monkeypatch.setattr(runtime._native, "_close_runtime", fail_second)
    try:
        with pytest.raises(RuntimeError, match="close failure"):
            runtime.close()
        assert runtime._handles[0] is None
        assert runtime._handles[1] is not None
        assert runtime._stream is not None and runtime._banks and runtime._model is not None
        with pytest.raises(RuntimeError, match="closing"):
            runtime.step()
    finally:
        monkeypatch.setattr(runtime._native, "_close_runtime", close)
        runtime.close()
    assert runtime._closed and not runtime._banks


@pytest.mark.cuda
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_panda_finger_box_contacts_match_cpu_geometry(payload, device):
    import mujoco
    from tyx.compat import warp as wp
    from physolve_tyx._src.collision_primitive_core import box_box
    from test_physolve_models import robot_path

    @wp.kernel
    def contacts(p: wp.array[wp.vec3], r: wp.array[wp.mat33], s: wp.array[wp.vec3],
                 positions: wp.array[wp.vec3], distances: wp.array[float]):
        dist, pos, normal = box_box(p[0], r[0], s[0], p[1], r[1], s[1])
        for index in range(8):
            positions[index] = pos[index]
            distances[index] = dist[index]

    model = mujoco.MjModel.from_xml_path(str(robot_path('panda')))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pair = [36, 40]
    expected = np.array([c.pos.copy() for c in data.contact if list(c.geom) == pair])
    assert len(expected) > 0
    p = wp.array(data.geom_xpos[pair].astype(np.float32), dtype=wp.vec3, device=device)
    r = wp.array(data.geom_xmat[pair].reshape(2, 3, 3).astype(np.float32), dtype=wp.mat33, device=device)
    s = wp.array(model.geom_size[pair].astype(np.float32), dtype=wp.vec3, device=device)
    pos, dist = wp.zeros(8, dtype=wp.vec3, device=device), wp.zeros(8, device=device)
    wp.launch(contacts, 1, inputs=[p, r, s, pos, dist], device=device)
    actual = pos.numpy()[dist.numpy() < 0]
    assert len(actual) > 0
    # Contact ordering/duplicate points can differ. Every produced point must
    # still lie at a CPU geometric contact, at the existing absolute tolerance.
    nearest = np.min(np.max(np.abs(actual[:, None] - expected[None]), axis=2), axis=1)
    np.testing.assert_array_less(nearest, 1e-6)


@pytest.mark.cuda
def test_actuator_control_and_two_runtime_isolation(payload):
    xml = SLIDER.replace('<joint type=', '<joint name="axis" type=')
    xml = xml.replace('</mujoco>', '<actuator><motor joint="axis"/></actuator></mujoco>')
    record = compile_mujoco_xml(xml, require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=8) as first, \
            PhySolveRuntime(record, payload=payload, nworld=8) as second:
        initial = second.snapshot()
        assert first._stream is not second._stream
        command = np.full((8, 1), 2., dtype=np.float32)
        first.step(ctrl=command)
        np.testing.assert_allclose(first.snapshot()["qpos"][:, 0], .002**2,
                                   atol=1e-6, rtol=0)
        np.testing.assert_array_equal(first.snapshot()["qfrc_actuator"], command)
        np.testing.assert_array_equal(first.snapshot()["qfrc_applied"], np.zeros((8, 1)))
        for field, value in second.snapshot().items():
            np.testing.assert_array_equal(value, initial[field])
        first.close()
        second.step(ctrl=-command)
        np.testing.assert_allclose(second.snapshot()["qpos"][:, 0], -.002**2,
                                   atol=1e-6, rtol=0)
        second.reset()
        np.testing.assert_array_equal(second.snapshot()["ctrl"], initial["ctrl"])


@pytest.mark.cuda
@pytest.mark.parametrize("name", ["panda", "fanuc"])
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_robot_forward_and_masked_reset_native_state(payload, name, nworld):
    from test_physolve_models import robot_path
    record = compile_mujoco(robot_path(name), require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=nworld, nconmax=128) as runtime:
        initial = runtime.snapshot()
        force = np.full((nworld, record.mj_model.nv), .03, dtype=np.float32)
        for _ in range(3):
            runtime.step(qfrc_applied=force)
        advanced = runtime.snapshot()
        assert np.any(advanced["qvel"] != initial["qvel"])
        runtime.forward()
        for field in ("qpos", "qvel", "time"):
            np.testing.assert_array_equal(runtime.snapshot()[field], advanced[field])
        mask = np.arange(nworld) % 2 == 0
        runtime.reset(mask)
        reset = runtime.snapshot()
        for field in ("qpos", "qvel", "time", "qfrc_applied"):
            np.testing.assert_array_equal(reset[field][mask], initial[field][mask])
            np.testing.assert_array_equal(reset[field][~mask], advanced[field][~mask])
        runtime.reset()
        for field in ("qpos", "qvel", "time", "qfrc_applied"):
            np.testing.assert_array_equal(runtime.snapshot()[field], initial[field])
