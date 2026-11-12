"""BALL-M1 real publication, lifecycle and failure ownership; no mock dynamics."""
import numpy as np
import pytest
from test_mujoco_tyx_activation import save
from test_mujoco_tyx_ball import MODES, H
from test_mujoco_tyx_ball_joint_motor import KEYS, fixture, control
from test_mujoco_tyx_ball_transactions import same

from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime
from phymeo.training.resident_stepper import PhysicsStepFailure


def assert_complete_equal(left, right):
    same(left, right)
    for key in left:
        a, b = left[key].last_step, right[key].last_step
        if a is None or b is None:
            assert a is b
            continue
        for field in ("ids", "gear", "row_offsets", "dof_indices", "moment",
                      "input_control", "clamped_control", "raw_force", "scalar_force",
                      "generalized_force"):
            np.testing.assert_array_equal(getattr(a.transmission, field), getattr(b.transmission, field))
        for field in ("qfrc_actuator", "qfrc_constraint"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


@pytest.mark.parametrize("mode", MODES)
def test_joint_motor_checkpoint_clone_partial_reset_successor(mode, tmp_path):
    mj, model, compiled, starts, runtime = fixture("mixed", mode)
    payload = dict(starts=starts, events=[])
    try:
        for tick in range(3):
            runtime.set_controls({k:control(model,tick,s) for s,k in enumerate(KEYS)})
            runtime.step(H)
            payload["events"].append(runtime.integration_snapshot())
        before = runtime.integration_snapshot()
        checkpoint = runtime.checkpoint()
        raw = checkpoint.to_bytes()
        (tmp_path/"checkpoint.bin").write_bytes(raw)
        restored = MuJoCoTyxCheckpoint.from_bytes(raw)
        with runtime.clone() as clone:
            assert_complete_equal(before, clone.integration_snapshot())
            runtime.step(H)
            successor = runtime.integration_snapshot()
            revision = runtime.checkpoint_revision
            runtime.restore(restored, expected_revision=revision)
            assert_complete_equal(before, runtime.integration_snapshot())
            with pytest.raises(ValueError):
                runtime.restore(restored, expected_revision=revision)
            runtime.step(H); clone.step(H)
            assert_complete_equal(successor, runtime.integration_snapshot())
            assert_complete_equal(successor, clone.integration_snapshot())
            runtime.reset((KEYS[-1],))
            reset = runtime.integration_snapshot()
            assert reset[KEYS[-1]].tick == 0 and reset[KEYS[-1]].last_step is None
            assert_complete_equal({KEYS[0]:reset[KEYS[0]]}, {KEYS[0]:successor[KEYS[0]]})
            with MuJoCoRuntime(compiled, mode=mode, implementation="tyx", world_keys=(KEYS[-1],),
                              initial_states={KEYS[-1]:starts[KEYS[-1]]}) as fresh:
                packets = {k:control(model,8,s) for s,k in enumerate(KEYS)}
                runtime.set_controls(packets); clone.set_controls(packets)
                fresh.set_controls({KEYS[-1]:packets[KEYS[-1]]})
                runtime.step(H); clone.step(H); fresh.step(H)
                result, other = runtime.integration_snapshot(), clone.integration_snapshot()
                assert_complete_equal({KEYS[0]:result[KEYS[0]]}, {KEYS[0]:other[KEYS[0]]})
                assert_complete_equal({KEYS[-1]:result[KEYS[-1]]}, fresh.integration_snapshot())
                payload.update(before=before, checkpoint=checkpoint, successor=successor,
                               reset=reset, result=result, clone=other, fresh=fresh.integration_snapshot())
    except BaseException as error:
        payload.update(failure=repr(error), retained=runtime.integration_snapshot())
        raise
    finally:
        payload["counters"] = dict(runtime.counters)
        runtime.close()
        save(tmp_path/"lifecycle.json.gz", payload)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fault", ("last-world-status", "late-readback"))
def test_joint_motor_late_failure_retains_old_receipt_then_retries(mode, fault, monkeypatch, tmp_path):
    _, model, _, starts, runtime = fixture("shared", mode)
    payload = dict(starts=starts, fault=fault)
    try:
        runtime.set_controls({k:control(model,0,s) for s,k in enumerate(KEYS)})
        runtime.step(H)
        runtime.set_controls({k:control(model,8,s) for s,k in enumerate(KEYS)})
        with runtime.clone() as clone:
            before = runtime.integration_snapshot()
            original = runtime._device.download
            count = runtime.counters["completed_kernels"]
            def after_real():
                result = original()
                payload["failed_readback"] = result
                if fault == "late-readback":
                    raise RuntimeError("BALL-M1 after actual readback")
                result["status"][-1] = 2
                return result
            monkeypatch.setattr(runtime._device, "download", after_real)
            with pytest.raises(PhysicsStepFailure) as caught:
                runtime.step(H)
            assert caught.value.accepted_worlds == dict.fromkeys(KEYS, False)
            assert caught.value.acceptance_known and not caught.value.physics_accepted
            assert runtime.counters["completed_kernels"] == count+3
            retained = runtime.integration_snapshot()
            assert_complete_equal(before, retained)
            for k in KEYS:
                assert before[k].last_step is retained[k].last_step
            payload.update(before=before, retained=retained,
                           accepted=caught.value.accepted_worlds, error=repr(caught.value))
            monkeypatch.setattr(runtime._device, "download", original)
            runtime.step(H); clone.step(H)
            assert_complete_equal(runtime.integration_snapshot(), clone.integration_snapshot())
            payload.update(retry=runtime.integration_snapshot(), clone=clone.integration_snapshot())
    except BaseException as error:
        payload["failure"] = repr(error)
        raise
    finally:
        payload["counters"] = dict(runtime.counters)
        runtime.close()
        save(tmp_path/"late-failure.json.gz",payload)


@pytest.mark.parametrize("mode", MODES)
def test_joint_motor_bad_last_control_changes_no_world(mode, tmp_path):
    _, model, _, _, runtime = fixture("shared", mode)
    try:
        runtime.set_controls({k:control(model,0,s) for s,k in enumerate(KEYS)})
        runtime.step(H)
        before = runtime.integration_snapshot()
        count = runtime.counters["completed_kernels"]
        with pytest.raises(ValueError):
            runtime.set_controls({KEYS[0]:control(model,8),
                                  KEYS[-1]:MuJoCoControl(np.zeros(model.nu+1))})
        assert_complete_equal(before,runtime.integration_snapshot())
        assert runtime.counters["completed_kernels"] == count
        save(tmp_path/"invalid-control.json.gz",dict(before=before,retained=runtime.integration_snapshot()))
    finally:
        runtime.close()
