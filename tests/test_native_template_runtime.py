"""Actual robot Host qualification; run each CUDA parameter in a fresh process."""

import numpy as np
import pytest
from test_physolve_models import robot_path

from phymeo.builders.native_assembly import NativeAssetInstance, compile_native_template

pytest_plugins = ["test_physolve_runtime"]


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_fanuc_template_step_forward_reset(payload, nworld):
    template = compile_native_template(
        (NativeAssetInstance("fanuc", str(robot_path("fanuc")), drive_mode="scalar_pd"),),
        timestep=0.002,
        gravity=(0, 0, -9.81),
    )
    with template.bind(nworld).create_runtime(payload=payload, nconmax=128, njmax=256) as runtime:
        initial = runtime.snapshot()
        force = np.full((nworld, template.compilation.mj_model.nv), 0.03, dtype=np.float32)
        for _ in range(4):
            runtime.step(qfrc_applied=force)
        after = runtime.snapshot()
        assert np.any(after["qvel"] != initial["qvel"]), "robot failed the existing motion check"
        runtime.forward()
        for field in ("qpos", "qvel", "time"):
            np.testing.assert_array_equal(runtime.snapshot()[field], after[field])
        mask = np.arange(nworld) % 2 == 0
        runtime.reset(mask)
        reset = runtime.snapshot()
        for field in ("qpos", "qvel", "ctrl", "qfrc_applied", "time"):
            np.testing.assert_array_equal(reset[field][mask], initial[field][mask])
            np.testing.assert_array_equal(reset[field][~mask], after[field][~mask])
        runtime.reset()
        for _ in range(4):
            runtime.step(qfrc_applied=force)
        for field in ("qpos", "qvel", "time"):
            np.testing.assert_allclose(runtime.snapshot()[field], after[field], atol=1e-6, rtol=0)
