"""BALL1 real multiworld publication and accepted-state lifecycle tests."""

import sys

import numpy as np
import pytest

from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime
from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.resident_stepper import PhysicsStepFailure
from test_mujoco_tyx_ball import H, MAXIMUM, MODES, assert_dynamics, control, deps, initial, native, save


def same(left, right):
    assert set(left) == set(right)
    for key in left:
        for field in ("q", "v", "ctrl", "act"):
            np.testing.assert_array_equal(getattr(left[key], field), getattr(right[key], field))
        assert (left[key].time, left[key].tick) == (right[key].time, right[key].tick)
        for field in ("ctrl", "qfrc_applied", "xfrc_applied"):
            np.testing.assert_array_equal(getattr(left[key].held_control, field), getattr(right[key].held_control, field))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("keys", [(19,), (7, 103), (3, 19, 91, 203)], ids=["w1", "w2", "w4"])
def test_ball_checkpoint_clone_partial_reset_real_successors(mode, keys, tmp_path):
    mj = deps()
    model = mj.MjModel.from_xml_string(MAXIMUM)
    compiled = compile_mujoco_xml(MAXIMUM, require_asset=False)
    starts = {key: initial(mj, model, slot) for slot, key in enumerate(keys)}
    with MuJoCoTyxRuntime(compiled, mode=mode, world_keys=keys, initial_states=starts) as runtime:
        for tick in range(3):
            runtime.set_controls({key: control(model, tick, slot) for slot, key in enumerate(keys)})
            runtime.step(H)
        checkpoint = runtime.checkpoint()
        checkpoint_bytes = checkpoint.to_bytes()
        (tmp_path / "accepted-checkpoint.bin").write_bytes(checkpoint_bytes)
        decoded = MuJoCoTyxCheckpoint.from_bytes(checkpoint_bytes)
        before = runtime.integration_snapshot()
        with runtime.clone() as cloned:
            same(before, cloned.integration_snapshot())
            runtime.step(H)
            successor = runtime.integration_snapshot()
            revision = runtime.checkpoint_revision
            runtime.restore(decoded, expected_revision=revision)
            same(before, runtime.integration_snapshot())
            with pytest.raises(ValueError):
                runtime.restore(decoded, expected_revision=revision)
            runtime.step(H)
            cloned.step(H)
            same(successor, runtime.integration_snapshot())
            same(successor, cloned.integration_snapshot())
            selected = keys[-1]
            runtime.reset((selected,))
            reset = runtime.integration_snapshot()
            assert reset[selected].tick == 0 and reset[selected].last_step is None
            for key in keys[:-1]:
                same({key: reset[key]}, {key: successor[key]})
            with MuJoCoTyxRuntime(compiled, mode=mode, world_keys=(selected,), initial_states={selected: starts[selected]}) as fresh:
                packets = {key: control(model, 4, slot) for slot, key in enumerate(keys)}
                runtime.set_controls(packets)
                cloned.set_controls(packets)
                fresh.set_controls({selected: packets[selected]})
                runtime.step(H)
                cloned.step(H)
                fresh.step(H)
                result, reference = runtime.integration_snapshot(), cloned.integration_snapshot()
                same({selected: result[selected]}, fresh.integration_snapshot())
                for key in keys[:-1]:
                    same({key: result[key]}, {key: reference[key]})
                save(tmp_path / "checkpoint-successors.jsonl", dict(source_xml=MAXIMUM, starts=starts, before=before, successor=successor, reset=reset, result=result, clone=reference, fresh=fresh.integration_snapshot(), counters=runtime.counters))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fault", ["last-world-status", "late-readback"])
def test_ball_late_failure_retains_full_batch_and_retry(mode, fault, monkeypatch, tmp_path):
    mj = deps()
    model = mj.MjModel.from_xml_string(MAXIMUM)
    keys = (7, 29, 113, 211)
    with MuJoCoTyxRuntime(compile_mujoco_xml(MAXIMUM, require_asset=False), mode=mode, world_keys=keys, initial_states={key: initial(mj, model, slot) for slot, key in enumerate(keys)}) as runtime:
        runtime.set_controls({key: control(model, 0, slot) for slot, key in enumerate(keys)})
        runtime.step(H)
        runtime.set_controls({key: control(model, 8, slot) for slot, key in enumerate(keys)})
        with runtime.clone() as reference:
            before = runtime.integration_snapshot()
            original = runtime._device.download
            calls = runtime.counters["completed_kernels"]
            def fail_after_real_readback():
                result = original()
                if fault == "late-readback":
                    raise RuntimeError("BALL1 injected after complete real readback")
                result["status"][-1] = 2
                return result
            monkeypatch.setattr(runtime._device, "download", fail_after_real_readback)
            with pytest.raises(PhysicsStepFailure) as caught:
                runtime.step(H)
            assert caught.value.accepted_worlds == dict.fromkeys(keys, False)
            assert caught.value.acceptance_known and not caught.value.physics_accepted
            assert runtime.counters["completed_kernels"] == calls + 3
            same(before, runtime.integration_snapshot())
            save(tmp_path / "atomic-failure.jsonl", dict(fault=fault, before=before, retained=runtime.integration_snapshot(), mask=caught.value.accepted_worlds, counters=runtime.counters))
            monkeypatch.setattr(runtime._device, "download", original)
            runtime.step(H)
            reference.step(H)
            same(runtime.integration_snapshot(), reference.integration_snapshot())
            save(tmp_path / "atomic-failure.jsonl", dict(retry=runtime.integration_snapshot(), clone=reference.integration_snapshot()))


@pytest.mark.parametrize("mode", MODES)
def test_ball_step_and_integration_observation_have_no_cpu_dynamics(mode, monkeypatch, tmp_path):
    mj = deps()
    model = mj.MjModel.from_xml_string(MAXIMUM)
    with MuJoCoTyxRuntime(compile_mujoco_xml(MAXIMUM, require_asset=False), mode=mode, initial_states={0: initial(mj, model)}) as runtime:
        before = runtime.integration_snapshot()[0]
        packet = control(model, 0)
        expected = native(mj, model, before.q, before.v, packet, before.time)
        runtime.set_controls({0: packet})
        count = runtime.counters["completed_kernels"]
        def forbidden(*args, **kwargs):
            raise AssertionError("BALL1 actual step must not run native/host dynamics")
        with monkeypatch.context() as patch:
            for name in ("mj_step", "mj_forward", "mj_fullM", "mj_rne", "mj_jac"):
                patch.setattr(mj, name, forbidden)
            for name in ("solve", "inv", "cholesky"):
                patch.setattr(np.linalg, name, forbidden)
            patch.setitem(sys.modules, "mujoco_warp", None)
            patch.setitem(sys.modules, "warp", None)
            assert runtime.step(H) == {0: True}
            after = runtime.integration_snapshot()[0]
        assert runtime.counters["completed_kernels"] == count + 3
        assert_dynamics(after.last_step, expected)
        save(tmp_path / "no-fallback.jsonl", dict(before=before, after=after, native=expected, counters=runtime.counters))


@pytest.mark.parametrize("mode", MODES)
def test_ball_bad_last_control_is_rejected_before_any_held_update(mode, tmp_path):
    from phymeo.backends.mujoco_runtime import MuJoCoControl
    mj = deps()
    model = mj.MjModel.from_xml_string(MAXIMUM)
    keys = (7, 113)
    with MuJoCoTyxRuntime(compile_mujoco_xml(MAXIMUM, require_asset=False), mode=mode, world_keys=keys, initial_states={key: initial(mj, model, slot) for slot, key in enumerate(keys)}) as runtime:
        runtime.set_controls({key: control(model, 0, slot) for slot, key in enumerate(keys)})
        runtime.step(H)
        before = runtime.integration_snapshot()
        good = control(model, 8)
        bad_force = np.asarray(control(model, 8, 1).qfrc_applied).copy()
        bad_force[-1] = np.nan
        with pytest.raises(ValueError):
            runtime.set_controls({keys[0]: good, keys[1]: MuJoCoControl([], qfrc_applied=bad_force)})
        same(before, runtime.integration_snapshot())
        save(tmp_path / "invalid-control.jsonl", dict(before=before, retained=runtime.integration_snapshot(), invalid_world=keys[-1]))
