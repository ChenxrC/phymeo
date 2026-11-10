"""CPU 3.3.7's post-disable force clamp, including nonzero-only ranges."""

import numpy as np
import pytest

from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml


@pytest.mark.parametrize("mode", ("interpret", pytest.param("cuda", marks=pytest.mark.cuda)))
@pytest.mark.parametrize("force_range,expected", (("2 3", 2.0), ("-3 -2", -2.0), ("-2 3", 0.0)))
def test_disabled_actuator_still_obeys_native_post_force_clamp(mode, force_range, expected):
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("tyx")
    xml = f'''<mujoco><option timestep=".001" gravity="0 0 0" actuatorgroupdisable="1"/>
      <worldbody><body><joint name="j" type="slide" axis="1 0 0"/>
        <inertial pos="0 0 0" mass=".5" diaginertia=".01 .01 .01"/>
      </body></worldbody><actuator><motor joint="j" gear="2" group="1"
        forcelimited="true" forcerange="{force_range}"/>
      </actuator></mujoco>'''
    compiled = compile_mujoco_xml(xml, require_asset=False)
    native = mj.MjData(compiled.mj_model)
    native.ctrl[:] = 99
    with MuJoCoRuntime(compiled, mode=mode) as runtime:
        runtime.set_controls({0: MuJoCoControl([99])})
        for _ in range(3):
            mj.mj_step(compiled.mj_model, native)
            runtime.step(0.001)
        state = runtime.snapshot()[0]
        np.testing.assert_array_equal(native.actuator_force, [expected])
        np.testing.assert_allclose(state.last_step.qfrc_actuator, [2 * expected], rtol=0, atol=1e-7)
        np.testing.assert_allclose(state.q, native.qpos, rtol=0, atol=1e-7)
        np.testing.assert_allclose(state.v, native.qvel, rtol=0, atol=1e-7)
