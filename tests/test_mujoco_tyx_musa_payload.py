"""Contact between two independently actuated arms and a freely moving payload."""

import os
from unittest.mock import patch

import numpy as np
import pytest

from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml


@pytest.mark.musa
@pytest.mark.skipif(os.environ.get("PHYMEO_TEST_MUSA") != "1", reason="requires a MUSA GPU")
def test_two_independent_hinges_transmit_force_to_free_payload(tmp_path):
    mj = pytest.importorskip("mujoco")
    arm = (
        '<body pos="-.074 {y} .36"><joint name="{name}" axis="0 1 0" armature=".02"/>'
        '<geom name="{name}" pos="0 0 -.30" size=".025" mass=".3" '
        'contype="2" conaffinity="1"/></body>'
    )
    xml = (
        '<mujoco><option timestep=".002" solver="Newton" iterations="40" gravity="0 0 0"/>'
        '<default><geom condim="3" friction=".5 .01 .001"/></default><worldbody>'
        + arm.format(y="-.045", name="left")
        + '<body name="payload" pos="0 0 .06"><freejoint name="load"/>'
        '<geom name="load" type="box" size=".05 .09 .05" mass="2" '
        'contype="1" conaffinity="2"/></body>'
        + arm.format(y=".045", name="right")
        + '</worldbody><actuator><motor joint="left"/><motor joint="right"/></actuator></mujoco>'
    )
    asset = compile_mujoco_xml(xml, require_asset=False)
    model = asset.mj_model
    data = mj.MjData(model)
    data.ctrl[:] = [-.6, -.9]
    with MuJoCoRuntime(asset, mode="musa", cache_root=tmp_path, contact_capacity=2,
                       constraint_capacity=8) as runtime:
        assert runtime.capabilities["articulation_root_scope"] == "free-payload-with-fixed-arms"
        runtime.set_controls({0: MuJoCoControl(data.ctrl.copy())})
        before = runtime.musa_device_report()
        for _ in range(8):
            mj.mj_step(model, data)
            with patch.object(mj, "mj_step", side_effect=AssertionError("CPU fallback")):
                runtime.step(model.opt.timestep)
            state = runtime.integration_snapshot()[0]
            np.testing.assert_allclose(state.q, data.qpos, atol=3e-5, rtol=1e-4)
            np.testing.assert_allclose(state.v, data.qvel, atol=3e-4, rtol=1e-3)
            np.testing.assert_allclose(state.last_step.qfrc_constraint, data.qfrc_constraint,
                                       atol=3e-3, rtol=2e-3)
            assert len(state.last_step.contacts) == data.ncon == 2
        after = runtime.musa_device_report()
        launched = after["launched_kernels"] - before["launched_kernels"]
        assert launched == after["completed_kernels"] - before["completed_kernels"] >= 8
        qx = model.jnt_qposadr[model.joint("load").id]
        assert state.q[qx] > 0.0001
        for pad in ("left", "right"):
            assert abs(state.last_step.qfrc_constraint[model.jnt_dofadr[model.joint(pad).id]]) > .01
