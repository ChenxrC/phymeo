"""Opt-in S4000 acceptance through the public MuJoCo solver entry point."""

import json
import os
import sys

import numpy as np
import pytest
from test_mujoco_tyx_runtime import SLIDER, compile_source, source

from phymeo.backends.mujoco_resource_budget import MuJoCoTyxResourceBudget
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime
from phymeo.training.resident_stepper import PhysicsStepFailure

pytestmark = [
    pytest.mark.musa,
    pytest.mark.skipif(
        os.environ.get("PHYMEO_TEST_MUSA") != "1", reason="set PHYMEO_TEST_MUSA=1 on a MUSA GPU"
    ),
]


def test_musa_slider_against_native_at_every_step(monkeypatch, tmp_path):
    import mujoco as mj

    from phymeo.backends.mujoco_tyx import _kernels

    compilation = compile_source(SLIDER)
    model = compilation.mj_model
    controls = {4: 8.0, 91: -0.7}
    references = {key: mj.MjData(model) for key in controls}
    cpu_step = mj.mj_step
    history = []
    budget = MuJoCoTyxResourceBudget(8 * 1024 * 1024)

    def forbidden(*args, **kwargs):
        pytest.fail("MUSA physics called a CPU, Warp, or upstream interpreter execution path")

    monkeypatch.setitem(sys.modules, "mujoco_warp", None)
    monkeypatch.setitem(sys.modules, "warp", None)
    with MuJoCoRuntime(
        compilation,
        world_keys=tuple(controls),
        mode="musa",
        cache_root=tmp_path / "compile",
        resource_budget=budget,
    ) as runtime:
        assert runtime.capabilities["gpu"] and runtime.capabilities["implementation"] == "tyx"
        runtime.set_controls({key: MuJoCoControl([value]) for key, value in controls.items()})
        for tick in range(1, 25):
            for key, control in controls.items():
                references[key].ctrl[:] = control
                cpu_step(model, references[key])
            with monkeypatch.context() as guard:
                guard.setattr(mj, "mj_step", forbidden)
                guard.setattr(mj, "mj_forward", forbidden)
                guard.setattr(type(_kernels.advance), "__call__", forbidden)
                assert runtime.step(model.opt.timestep) == dict.fromkeys(controls, True)
            device = runtime.musa_device_report()
            assert device["active_bank"] == tick % 2
            assert device["completed_kernels"] == device["launched_kernels"] == 2 * tick
            for key, state in runtime.snapshot().items():
                ref = references[key]
                np.testing.assert_allclose(state.q, ref.qpos, rtol=0, atol=3e-5)
                np.testing.assert_allclose(state.v, ref.qvel, rtol=0, atol=3e-5)
                assert state.time == pytest.approx(ref.time, abs=2e-6)
                assert state.tick == tick and state.last_step.backend == "mujoco-tyx-musa"
                assert not state.last_step.contacts
                history.append(
                    dict(
                        tick=tick,
                        world=key,
                        q=state.q.item(),
                        v=state.v.item(),
                        q_ref=ref.qpos.item(),
                        v_ref=ref.qvel.item(),
                    )
                )

        assert references[4].qpos[0] > 0 > references[91].qpos[0]
        assert runtime.counters["completed_kernels"] == 48
        report = runtime.musa_device_report()
        assert report["backend"] == "musa" and report["device_name"] and report["arch"]
        assert report["live_payload_bytes"] == budget.snapshot()["current_bytes"] > 0
        result = dict(device=report, counts=dict(runtime.counters), history=history)
    assert budget.snapshot()["current_bytes"] == budget.snapshot()["reserved_bytes"] == 0
    result["bytes_after_close"] = budget.snapshot()["current_bytes"]
    (tmp_path / "slider.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf8"
    )


def test_failed_batch_keeps_last_published_state_and_recovers(monkeypatch, tmp_path):
    import mujoco as mj

    compilation = compile_source(SLIDER)
    model = compilation.mj_model
    controls = {4: 2.0, 91: -0.5}
    references = {key: mj.MjData(model) for key in controls}
    with MuJoCoRuntime(
        compilation, mode="musa", world_keys=tuple(controls), cache_root=tmp_path / "compile"
    ) as runtime:
        runtime.set_controls({key: MuJoCoControl([value]) for key, value in controls.items()})
        runtime.step(runtime.timestep)
        before = runtime.snapshot()
        report = runtime.musa_device_report()

        def failed_download():
            raise RuntimeError("injected failure reading the candidate bank")

        with monkeypatch.context() as fault:
            fault.setattr(runtime._device, "download", failed_download)
            with pytest.raises(PhysicsStepFailure) as error:
                runtime.step(runtime.timestep)
        assert error.value.accepted_worlds == dict.fromkeys(controls, False)
        after_failure = runtime.musa_device_report()
        assert after_failure["active_bank"] == report["active_bank"]
        assert after_failure["revision"] == report["revision"]
        assert after_failure["completed_kernels"] == report["completed_kernels"] + 2
        for key, state in runtime.snapshot().items():
            assert state.tick == before[key].tick and state.time == before[key].time
            np.testing.assert_array_equal(state.q, before[key].q)
            np.testing.assert_array_equal(state.v, before[key].v)
        assert runtime.step(runtime.timestep) == dict.fromkeys(controls, True)
        for key, state in runtime.snapshot().items():
            ref = references[key]
            ref.ctrl[:] = controls[key]
            mj.mj_step(model, ref, nstep=2)
            assert state.tick == 2
            np.testing.assert_allclose(state.q, ref.qpos, atol=3e-5, rtol=0)
            np.testing.assert_allclose(state.v, ref.qvel, atol=3e-5, rtol=0)


def test_contact_model_is_rejected_before_device_allocation(monkeypatch):
    from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxUnsupported
    from phymeo.backends.mujoco_tyx import _device

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported model reached device allocation")

    monkeypatch.setattr(_device.TyxDevice, "__init__", forbidden)
    with pytest.raises(MuJoCoTyxUnsupported, match="musa-domain"):
        MuJoCoRuntime(compile_source(source()), mode="musa")
