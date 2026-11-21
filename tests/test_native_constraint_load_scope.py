"""Frozen resident row trials expire before a new step may advance the body."""

import numpy as np
import pytest
from test_damped_coupled_mixed_area_friction import H, _declaration


@pytest.mark.cuda
def test_trial_load_lifetime_and_recovery():
    import mujoco
    import tyx as tx

    from phymeo.backends.mujoco_area_pressure import compile_area_native_rows
    from phymeo.backends.mujoco_tyx._constraint_device import TyxConstraintDevice

    spec, _ = _declaration()
    model = spec.compilation.fresh_native_model()
    plan = compile_area_native_rows(mujoco, model, (("compliant", "rigid"),))
    q, v = np.zeros((1, 8), np.float32), np.zeros((1, 8), np.float32)
    v[0, :2] = (0.12, 0.08)
    time = np.zeros((1, 2), np.float32)
    inputs = (
        np.zeros((1, 1), np.float32),
        np.zeros_like(v),
        np.zeros((1, model.nbody, 6), np.float32),
    )
    device = TyxConstraintDevice(plan, "cuda", None, q, v, time, model.nbody, H, (1, 64))
    invalid = None
    try:
        with device.prepare_native_rows(*inputs) as evaluate:
            with pytest.raises(RuntimeError):
                device.advance(*inputs)
            with pytest.raises(RuntimeError):
                with device.prepare_native_rows(*inputs):
                    pytest.fail("overlapping trial scopes")
            with pytest.raises(ValueError):
                evaluate(np.zeros_like(v))
            evaluate(None)
            baseline = device._get(device.physical_acc)
            bank = device.active
            with tx.use_runtime(device.runtime):
                # A nonfinite candidate must poison the current trial sequence.
                invalid = tx.cuda_array(np.full_like(v, np.nan))
            evaluate(invalid)
            assert device._get(device.status)[0] != 0
            evaluate(None)
            assert device._get(device.status)[0] != 0
            with pytest.raises(RuntimeError, match="rejected"):
                evaluate(None, integrate=True)
            np.testing.assert_array_equal(device._get(device.q[bank]), q)
            np.testing.assert_array_equal(device._get(device.q[1 - bank]), q)
            np.testing.assert_array_equal(device._get(device.time[1 - bank]), time)
            np.testing.assert_array_equal(device._get(device.v[bank]), v)
            np.testing.assert_array_equal(device._get(device.time[bank]), time)
        with device.prepare_native_rows(*inputs) as fresh:
            with pytest.raises(RuntimeError):
                evaluate(None)
            fresh(None)
            assert device._get(device.status)[0] == 0
            np.testing.assert_array_equal(device._get(device.physical_acc), baseline)
        with pytest.raises(RuntimeError):
            fresh(None)
        device.advance(*inputs)
        assert device._get(device.status)[0] == 0
        assert device._get(device.time[1 - bank])[0, 0] == np.float32(H)
    finally:
        if invalid is not None:
            invalid.free()
        device.close()
