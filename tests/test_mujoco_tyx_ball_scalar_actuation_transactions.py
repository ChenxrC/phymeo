"""BALL-A accepted state/control/activation lifecycle on real sparse worlds."""

import sys

import numpy as np
import pytest
from test_mujoco_tyx_ball_scalar_actuation import (
    ACT,
    MODES,
    H,
    analytic,
    compare_receipt,
    compare_state,
    control,
    encode,
    fixture,
    native_data,
    native_fields,
    save,
    set_native,
    source,
)

from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint
from phymeo.backends.mujoco_runtime import MuJoCoRuntime
from phymeo.training.resident_stepper import PhysicsStepFailure


LIFECYCLES = (("stateless", (7, 103)), ("filter-early", (3, 19, 91, 203)))
FAULTS = (
    ("stateless", "late-readback"), ("stateless", "last-world-status"),
    ("filter-early", "late-readback"), ("filter-early", "last-world-status"),
    ("filter-early", "input-act"), ("filter-early", "restore-readback"),
)


def same(left, right):
    # Complete immutable integration publication, including held control/receipt.
    assert encode(left) == encode(right)


def advance_checked(mj, model, runtime, history):
    entry = runtime.integration_snapshot()
    oracles = {}
    for key, state in entry.items():
        packet = state.held_control
        native = native_data(mj, model, state)
        set_native(native, packet)
        expected = native_fields(mj, model, native)
        independent = analytic(model, state, packet)
        mj.mj_step(model, native)
        oracles[key] = (native, expected, independent)
    history["pending"] = dict(entry=entry, native={
        key: dict(fields=fields, analytic=independent,
                  output=(native.qpos.copy(), native.qvel.copy(), native.act.copy(), native.time))
        for key, (native, fields, independent) in oracles.items()
    })
    before = runtime.counters["completed_kernels"]
    assert runtime.step(H) == dict.fromkeys(runtime.world_keys, True)
    result = runtime.integration_snapshot()
    history["events"].append(dict(history.pop("pending"), output=result))
    assert runtime.counters["completed_kernels"] - before == (4 if model.na else 3)
    for key, state in result.items():
        native, expected, independent = oracles[key]
        compare_receipt(model, entry[key], state, expected, independent)
        compare_state(state, native)
    return result


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case,keys", LIFECYCLES, ids=("stateless-w2", "stateful-w4"))
def test_ball_scalar_checkpoint_clone_partial_reset_successors(mode, case, keys, tmp_path):
    mj, model, compiled, starts, runtime = fixture(case, mode, keys)
    history = dict(source=source(case), mode=mode, case=case, starts=starts, events=[])
    try:
        for tick in range(3):
            runtime.set_controls({key: control(model, tick, slot) for slot, key in enumerate(keys)})
            advance_checked(mj, model, runtime, history)
        # Hold a genuinely different command across checkpoint, restore and clone.
        packets = {key: control(model, 8, slot) for slot, key in enumerate(keys)}
        runtime.set_controls(packets)
        checkpoint_bytes = runtime.checkpoint().to_bytes()
        (tmp_path / "ball-scalar-checkpoint.bin").write_bytes(checkpoint_bytes)
        capsule = MuJoCoTyxCheckpoint.from_bytes(checkpoint_bytes)
        assert capsule.to_bytes() == checkpoint_bytes
        expected_schema = f"mujoco-tyx-exact-checkpoint-v{2 if model.na else 1}"
        assert capsule.binding["schema"] == expected_schema
        before = runtime.integration_snapshot()
        history["checkpoint"] = checkpoint_bytes.hex()
        history["checkpoint_state"] = before
        with runtime.clone() as clone:
            same(before, clone.integration_snapshot())
            successor = advance_checked(mj, model, runtime, history)
            revision = runtime.checkpoint_revision
            runtime.restore(capsule, expected_revision=revision)
            same(before, runtime.integration_snapshot())
            restored_bytes = runtime.checkpoint().to_bytes()
            with pytest.raises(ValueError):
                runtime.restore(capsule, expected_revision=revision)
            assert runtime.checkpoint().to_bytes() == restored_bytes
            resumed = advance_checked(mj, model, runtime, history)
            clone_result = advance_checked(mj, model, clone, history)
            same(successor, resumed)
            same(successor, clone_result)
            selected = keys[-1]
            runtime.reset((selected,))
            reset = runtime.integration_snapshot()
            assert reset[selected].tick == 0 and reset[selected].last_step is None
            np.testing.assert_array_equal(reset[selected].act, np.asarray(starts[selected].act, np.float32))
            for key in keys[:-1]:
                same(reset[key], successor[key])
            history["reset"] = reset
            with MuJoCoRuntime(compiled, mode=mode, implementation="tyx", world_keys=(selected,),
                               initial_states={selected: starts[selected]}) as fresh:
                packets = {key: control(model, 16, slot) for slot, key in enumerate(keys)}
                runtime.set_controls(packets)
                clone.set_controls(packets)
                fresh.set_controls({selected: packets[selected]})
                result = advance_checked(mj, model, runtime, history)
                continued = advance_checked(mj, model, clone, history)
                fresh_result = advance_checked(mj, model, fresh, history)
                same(result[selected], fresh_result[selected])
                for key in keys[:-1]:
                    same(result[key], continued[key])
                history["clone_counters"] = dict(clone.counters)
                history["fresh_counters"] = dict(fresh.counters)
        if model.na:
            np.testing.assert_allclose(result[selected].act, fresh_result[selected].act, **ACT)
    except BaseException as error:
        history["failure"] = repr(error)
        history["published_on_failure"] = runtime.integration_snapshot()
        raise
    finally:
        history["counters"] = dict(runtime.counters)
        runtime.close()
        save(tmp_path / "ball-scalar-lifecycle.json.gz", history)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case,fault", FAULTS, ids=lambda value: str(value))
def test_ball_scalar_late_failure_retains_and_real_retry(mode, case, fault, monkeypatch, tmp_path):
    import phymeo.backends.mujoco_checkpoint as checkpoint_module

    mj, model, _, starts, runtime = fixture(case, mode, (7, 91))
    history = dict(source=source(case), mode=mode, case=case, fault=fault, starts=starts, events=[])
    try:
        runtime.set_controls({key: control(model, 0, slot) for slot, key in enumerate(runtime.world_keys)})
        advance_checked(mj, model, runtime, history)
        runtime.set_controls({key: control(model, 8, slot) for slot, key in enumerate(runtime.world_keys)})
        before = runtime.integration_snapshot()
        capsule = runtime.checkpoint()
        checkpoint_bytes = capsule.to_bytes()
        count = runtime.counters["completed_kernels"]
        history["before_failure"] = before
        history["checkpoint"] = checkpoint_bytes.hex()
        with runtime.clone() as reference:
            with monkeypatch.context() as patch:
                if fault == "restore-readback":
                    def bad_restore(*args, **kwargs):
                        raise RuntimeError("BALL-A late restore candidate readback")

                    patch.setattr(checkpoint_module, "_verify_bank", bad_restore)
                    with pytest.raises(RuntimeError, match="BALL-A late restore") as caught:
                        runtime.restore(capsule, expected_revision=runtime.checkpoint_revision)
                else:
                    original = runtime._device.download

                    def bad_download():
                        result = original()
                        if fault == "late-readback":
                            raise RuntimeError("BALL-A after complete real readback")
                        if fault == "last-world-status":
                            result["status"][-1] = 2
                        else:
                            result["input_act"][-1, 0] += .25
                        return result

                    patch.setattr(runtime._device, "download", bad_download)
                    with pytest.raises(PhysicsStepFailure) as caught:
                        runtime.step(H)
                    assert caught.value.acceptance_known and not caught.value.physics_accepted
                    assert caught.value.accepted_worlds == dict.fromkeys(runtime.world_keys, False)
                    assert runtime.counters["completed_kernels"] - count == (4 if model.na else 3)
                same(before, runtime.integration_snapshot())
                assert runtime.checkpoint().to_bytes() == checkpoint_bytes
                history["retained"] = runtime.integration_snapshot()
                history["failure_expected"] = repr(caught.value)
                history["failure_counters"] = dict(runtime.counters)
            result = advance_checked(mj, model, runtime, history)
            uninterrupted = advance_checked(mj, model, reference, history)
            same(result, uninterrupted)
            history["uninterrupted_counters"] = dict(reference.counters)
    except BaseException as error:
        history["unexpected_failure"] = repr(error)
        history["published_on_failure"] = runtime.integration_snapshot()
        raise
    finally:
        history["counters"] = dict(runtime.counters)
        runtime.close()
        save(tmp_path / "ball-scalar-failure-retry.json.gz", history)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", ("stateless", "filter-early"))
def test_ball_scalar_step_and_integration_observation_are_device_only(mode, case, monkeypatch, tmp_path):
    mj, model, _, starts, runtime = fixture(case, mode)
    history = dict(source=source(case), mode=mode, case=case, starts=starts)
    try:
        entry = runtime.integration_snapshot()[7]
        packet = control(model, 0)
        same_input = native_data(mj, model, entry)
        set_native(same_input, packet)
        expected = native_fields(mj, model, same_input)
        independent = analytic(model, entry, packet)
        mj.mj_step(model, same_input)
        runtime.set_controls({7: packet})
        history.update(entry=entry, packet=packet, native=expected, analytic=independent)

        def forbidden(*args, **kwargs):
            raise AssertionError("CPU dynamics entered BALL-A device step")

        count = runtime.counters["completed_kernels"]
        with monkeypatch.context() as patch:
            for name in ("mj_step", "mj_forward", "mj_fullM", "mj_jac", "mj_rne"):
                patch.setattr(mj, name, forbidden)
            for name in ("solve", "inv", "cholesky"):
                patch.setattr(np.linalg, name, forbidden)
            patch.setitem(sys.modules, "warp", None)
            patch.setitem(sys.modules, "mujoco_warp", None)
            assert runtime.step(H) == {7: True}
            state = runtime.integration_snapshot()[7]
        history["output"] = state
        assert runtime.counters["completed_kernels"] - count == (4 if model.na else 3)
        compare_receipt(model, entry, state, expected, independent)
        compare_state(state, same_input)
    except BaseException as error:
        history["failure"] = repr(error)
        raise
    finally:
        history["counters"] = dict(runtime.counters)
        runtime.close()
        save(tmp_path / "ball-scalar-device-only.json.gz", history)
