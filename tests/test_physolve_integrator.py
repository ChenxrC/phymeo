"""Explicit implicitfast qualification; original Euler regressions are unchanged."""

import mujoco
import numpy as np
import pytest
from test_native_exclusions import connection_plan
from test_physolve_models import robot_path

from phymeo import PhySolveRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.builders.native_assembly import NativeAssetInstance, compile_native_template

pytest_plugins = ["test_physolve_runtime"]

# Frozen independent recurrence: M=.01, kp=500, kd=50, dt=.002, no other forces.
DRIVE = '''<mujoco><option timestep=".002" integrator="implicitfast" gravity="0 0 0"/>
<worldbody><body><joint name="axis" type="hinge" axis="0 0 1"/>
<inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>
<geom type="sphere" size=".1" contype="0" conaffinity="0"/></body></worldbody>
<actuator><general joint="axis" gainprm="1" biasprm="0 -500 -50" biastype="affine"/>
</actuator></mujoco>'''


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_implicit_drive_matches_discrete_reference(payload, nworld):
    record = compile_mujoco_xml(DRIVE, require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=nworld) as runtime:
        assert int(runtime._model.opt.integrator) == 3
        command = np.linspace(1, 2, nworld, dtype=np.float32).reshape(nworld, 1)
        q, v = np.zeros((nworld, 1)), np.zeros((nworld, 1))
        for _ in range(100):
            v = (.01 * v + .002 * (command - 500 * q)) / (.01 + .002 * 50)
            q = q + .002 * v
            runtime.step(ctrl=command)
            actual = runtime.snapshot()
            np.testing.assert_allclose(actual["qpos"], q, atol=1e-6, rtol=0)
            np.testing.assert_allclose(actual["qvel"], v, atol=1e-6, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_fanuc_explicit_implicitfast_step_forward_reset(payload, nworld):
    template = compile_native_template(
        (NativeAssetInstance("fanuc", str(robot_path("fanuc")), drive_mode="scalar_pd"),),
        timestep=.002, gravity=(0, 0, -9.81), integrator="implicitfast",
    )
    template = template.with_collision_exclusions(connection_plan(template))
    model = template.compilation.fresh_native_model()
    cpu = mujoco.MjData(model)
    template.initial_state.apply(template.compilation, cpu)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    with template.bind(nworld).create_runtime(payload=payload, nconmax=128) as runtime:
        assert int(runtime._model.opt.integrator) == 3
        initial = runtime.snapshot()
        force = np.full((nworld, model.nv), .03, np.float32)
        for _ in range(250):
            cpu.qfrc_applied[:] = .03
            mujoco.mj_step(model, cpu)
            runtime.step(qfrc_applied=force)
            actual = runtime.snapshot()
            for field in ("qpos", "qvel"):
                np.testing.assert_allclose(actual[field],
                    np.broadcast_to(getattr(cpu, field), actual[field].shape), atol=1e-6, rtol=0)
        assert not np.any(cpu.warning.number)
        np.testing.assert_allclose(actual["time"], .5, atol=1e-6, rtol=0)
        runtime.forward()
        for field in ("qpos", "qvel", "time"):
            np.testing.assert_array_equal(runtime.snapshot()[field], actual[field])
        before = runtime.snapshot()
        mask = np.arange(nworld) % 2 == 0
        runtime.reset(mask)
        reset = runtime.snapshot()
        for field in before:
            np.testing.assert_array_equal(reset[field][~mask], before[field][~mask], err_msg=field)
            np.testing.assert_array_equal(reset[field][mask], initial[field][mask], err_msg=field)
        runtime.reset()
        for _ in range(250):
            runtime.step(qfrc_applied=force)
        repeated = runtime.snapshot()
        for field in ("qpos", "qvel", "time"):
            np.testing.assert_allclose(repeated[field], actual[field], atol=1e-6, rtol=0)
        print("FANUC implicitfast", nworld, "250 steps and replay", repeated["qvel"][0].tolist())


def test_runtime_rejects_unqualified_integrators(payload):
    for integrator in ("RK4", "implicit"):
        record = compile_mujoco_xml(DRIVE.replace('integrator="implicitfast"',
                                   f'integrator="{integrator}"'), require_asset=False)
        with pytest.raises(NotImplementedError, match="Euler or implicitfast"):
            PhySolveRuntime(record, payload=payload)
