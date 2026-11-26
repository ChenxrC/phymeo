"""Physical parity, transport accounting and publication through the training API."""

import os

import numpy as np
import pytest
from test_mujoco_tyx_musa_capsule_box import _source as capsule_box_xml
from test_mujoco_tyx_runtime import SLIDER, compile_source

from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.training.resident_stepper import PhysicsStepFailure


require_musa = pytest.mark.skipif(os.environ.get("PHYMEO_TEST_MUSA") != "1", reason="requires a MUSA GPU")
KEYS = (91, 4)  # Deliberately opposite the sorted world-key order.


def _standard_step(runtime, commands):
    runtime.set_controls({k: MuJoCoControl(row) for k, row in zip(KEYS, commands, strict=True)})
    assert runtime.step(runtime.timestep) == dict.fromkeys(KEYS, True)
    return runtime.snapshot()


def test_default_public_runtime_remains_native_after_resident_refusal():
    pytest.importorskip("mujoco")
    asset = compile_source(SLIDER)
    with MuJoCoRuntime(asset, world_keys=KEYS) as cpu:
        assert cpu.capabilities["mode"] == "cpu"
        with pytest.raises(ValueError, match="mode='musa'"):
            with cpu.musa_resident():
                pytest.fail("CPU owner unexpectedly entered the MUSA context")
        states = _standard_step(cpu, [[1.], [-1.]])
        assert states[91].v.item() > 0. > states[4].v.item()
        assert all(state.tick == 1 and state.last_step is not None for state in states.values())


@pytest.mark.musa
@require_musa
def test_resident_matches_cpu_and_accounts_for_transfers_and_reset(tmp_path):
    import torch
    import torch_musa  # noqa: F401 - register the private-use accelerator

    asset = compile_source(SLIDER)
    starts = {91: MuJoCoInitialState([-.0625], [.015625], time=.125),
              4: MuJoCoInitialState([.03125], [-.03125], time=.5)}
    commands = np.array([[[8.], [-.75]], [[2.], [-1.]], [[-1.], [2.]], [[-.5], [1.]]], dtype=np.float32)
    actions = [torch.tensor(row, device="musa:0") for row in commands]
    with MuJoCoRuntime(asset, world_keys=KEYS, initial_states=starts) as cpu:
        oracle = [_standard_step(cpu, row) for row in commands]
    options = dict(world_keys=KEYS, initial_states=starts, mode="musa", cache_root=tmp_path / "cache")
    owned = MuJoCoRuntime(asset, **options)
    with owned, owned.musa_resident() as bank:
        assert bank.world_keys == KEYS and (bank.nq, bank.nv, bank.nu) == (1, 1, 1)
        views = bank.buffers()
        assert views["qpos"].shape == views["qvel"].shape == (2, 8)
        assert views["time"].shape == (2, 2) and views["ctrl"].shape == (2, 1)
        initial = {name: views[name].to_numpy() for name in ("qpos", "qvel", "time", "ctrl")}
        with pytest.raises(TypeError, match="only.*ctrl"):
            views["qpos"].copy_from_torch(actions[0])
        with pytest.raises(RuntimeError, match="exclusively owned"):
            owned.snapshot()
        with pytest.raises(RuntimeError):
            owned.close()

        # No explicit host observations during the measured interval. Copy
        # endpoints to independently owned MUSA tensors, compare to CPU later.
        before = bank.report()
        history = []
        for action in actions:
            old = views
            old["ctrl"].copy_from_torch(action)
            views = bank.step()
            with pytest.raises(RuntimeError, match="expired"):
                old["qpos"].to_numpy()
            with pytest.raises(RuntimeError, match="expired"):
                old["ctrl"].copy_from_torch(action)
            history.append(tuple(views[name].to_torch() for name in ("qpos", "qvel", "time")))
        after = bank.report()
        count = len(actions)
        status_bytes = count*len(KEYS)*np.dtype(np.int32).itemsize
        observed_bytes = count*sum(initial[name].nbytes for name in ("qpos", "qvel", "time"))
        assert after["accepted_world_steps"] == count*len(KEYS)
        assert after["resident_steps"] == after["episode_steps"] == count
        assert after["completed_kernels"] == after["launched_kernels"] > before["completed_kernels"]
        assert after["uploaded_bytes"] == before["uploaded_bytes"]
        assert after["explicit_readback_bytes"] == before["explicit_readback_bytes"]
        assert after["downloaded_bytes"] - before["downloaded_bytes"] == status_bytes
        assert after["status_downloaded_bytes"] - before["status_downloaded_bytes"] == status_bytes
        assert after["torch_to_tyx_bytes"] - before["torch_to_tyx_bytes"] == commands.nbytes
        assert after["tyx_to_torch_bytes"] - before["tyx_to_torch_bytes"] == observed_bytes
        assert after["device_to_device_bytes"] - before["device_to_device_bytes"] == commands.nbytes + observed_bytes

        old = views
        restored = bank.reset()
        reset = bank.report()
        assert reset["episode_steps"] == 0 and reset["resident_steps"] == count
        assert reset["resident_resets"] == 1
        assert reset["uploaded_bytes"] == after["uploaded_bytes"]
        assert reset["downloaded_bytes"] == after["downloaded_bytes"]
        assert reset["tyx_to_tyx_bytes"] - after["tyx_to_tyx_bytes"] == sum(a.nbytes for a in initial.values())
        with pytest.raises(RuntimeError, match="expired"):
            old["time"].to_numpy()
        for name, value in initial.items():
            np.testing.assert_array_equal(restored[name].to_numpy(), value)
        readback = bank.report()
        assert readback["explicit_readback_bytes"] - reset["explicit_readback_bytes"] == reset["reset_storage_bytes"]
        assert readback["downloaded_bytes"] - reset["downloaded_bytes"] == reset["reset_storage_bytes"]
    with pytest.raises(RuntimeError, match="no longer active"):
        restored["qpos"].to_numpy()
    with pytest.raises(RuntimeError, match="closed"):
        owned.step(owned.timestep)

    for observed, expected in zip(history, oracle, strict=True):
        q, v, clock = (tensor.cpu().numpy() for tensor in observed)
        for i, key in enumerate(KEYS):
            np.testing.assert_allclose(q[i, :1], expected[key].q, rtol=0, atol=3e-5)
            np.testing.assert_allclose(v[i, :1], expected[key].v, rtol=0, atol=3e-5)
            assert float(clock[i, 0]) - float(clock[i, 1]) == pytest.approx(expected[key].time, abs=2e-7)

    # A fresh ordinary MUSA owner of this same compilation still stages host
    # controls and produces the full native-coordinate observation and receipt.
    with MuJoCoRuntime(asset, **options) as ordinary:
        standard = _standard_step(ordinary, commands[0])
        for key in KEYS:
            assert standard[key].last_step.backend == "mujoco-tyx-musa" and standard[key].tick == 1
            np.testing.assert_allclose(standard[key].q, oracle[0][key].q, rtol=0, atol=3e-5)
            np.testing.assert_allclose(standard[key].v, oracle[0][key].v, rtol=0, atol=3e-5)


@pytest.mark.musa
@require_musa
def test_contact_batch_rejects_one_nan_row_without_publishing_and_can_retry(tmp_path):
    import torch
    import torch_musa  # noqa: F401 - register the private-use accelerator

    asset = compile_source(capsule_box_xml(3))
    model = asset.mj_model
    load = model.joint("load")
    starts = {}
    for key in KEYS:
        q, v = model.qpos0.copy(), np.zeros(model.nv)
        if key == 91:
            q[load.qposadr[0]] += .06  # separated, while world 4 starts in physical contact
        v[model.joint("pitch").dofadr[0]], v[load.dofadr[0]+1] = -.05, -.04
        starts[key] = MuJoCoInitialState(q, v)
    commands = np.array([[-.12], [-.16]], dtype=np.float32)
    valid = torch.tensor(commands, device="musa:0")
    bad = torch.tensor([[.6], [float("nan")]], dtype=torch.float32, device="musa:0")
    with MuJoCoRuntime(asset, world_keys=KEYS, initial_states=starts) as cpu:
        first = _standard_step(cpu, commands)
        assert len(first[4].last_step.contacts) == 1 and not first[91].last_step.contacts
        recovered = _standard_step(cpu, commands)
    with MuJoCoRuntime(asset, world_keys=KEYS, initial_states=starts, mode="musa", cache_root=tmp_path / "cache",
                       contact_capacity=2, constraint_capacity=8) as gpu, gpu.musa_resident() as bank:
        views = bank.buffers()
        assert "entry_bias" not in views
        views["ctrl"].copy_from_torch(valid)
        views = bank.step()
        assert views["entry_bias"].shape == (len(KEYS), max(8, model.nv))
        saved = {name: views[name].to_numpy() for name in ("qpos", "qvel", "time")}
        before = bank.report()
        views["ctrl"].copy_from_torch(bad)
        with pytest.raises(PhysicsStepFailure) as caught:
            bank.step()
        assert caught.value.acceptance_known and caught.value.accepted_worlds == dict.fromkeys(KEYS, False)
        after = bank.report()
        assert after["active_bank"] == before["active_bank"]
        assert after["resident_steps"] == before["resident_steps"] == 1
        assert after["failed_batches"] == before["failed_batches"] + 1
        assert after["downloaded_bytes"] - before["downloaded_bytes"] == 4*len(KEYS)
        assert after["uploaded_bytes"] == before["uploaded_bytes"]
        with pytest.raises(RuntimeError, match="expired"):
            views["entry_bias"].to_numpy()
        retained = bank.buffers()
        assert "entry_bias" not in retained
        for name, expected in saved.items():
            np.testing.assert_array_equal(retained[name].to_numpy(), expected)

        retained["ctrl"].copy_from_torch(valid)
        actual = bank.step()
        assert bank.report()["accepted_world_steps"] == 2*len(KEYS)
        q, v = actual["qpos"].to_numpy(), actual["qvel"].to_numpy()
        for i, key in enumerate(KEYS):
            # The same native capsule-box fixture and tolerances as the
            # original MUSA joint/contact regression, including the free box.
            np.testing.assert_allclose(q[i, :model.nq], recovered[key].q, rtol=0, atol=2e-4)
            np.testing.assert_allclose(v[i, :model.nv], recovered[key].v, rtol=0, atol=3e-3)
