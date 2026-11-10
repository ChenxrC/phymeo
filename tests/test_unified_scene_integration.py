"""Consumers of optional compilation and the existing training protocol."""

import numpy as np
import pytest

from phymeo.control.policy_runner import PolicyRunner
from phymeo.training.resident_env import ResidentVectorEnv

SLIDER_XML = """
<mujoco model="compiler_slider">
  <option gravity="0 0 0" timestep="0.001" integrator="Euler"/>
  <default><geom type="sphere" size="0.05" mass="0.5"/></default>
  <worldbody><body name="base"><body name="carriage">
    <joint name="slide" type="slide" axis="1 0 0" limited="true" range="-2 2"/>
    <geom name="carriage_shape"/>
  </body></body></worldbody>
  <actuator><motor name="drive" joint="slide" ctrllimited="true" ctrlrange="-5 5"/></actuator>
</mujoco>
"""


def _compiled_asset_case(backend):
    mujoco = pytest.importorskip("mujoco")
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml
    from phymeo.builders.scene_compiler import ArticulationWorld, SceneDeclaration, compile_scene
    from phymeo.core.robot import RobotState
    from phymeo.runtime.robot_scene import RobotCommand

    compiled = compile_mujoco_xml(SLIDER_XML)
    declaration = SceneDeclaration(
        tuple(
            ArticulationWorld(
                key,
                compiled.asset,
                RobotState(compiled.initial_q, compiled.initial_v),
                kp=np.zeros(1),
                kd=np.zeros(1),
                effort_limit=np.full(1, 5.0),
            )
            for key in (4, 19)
        )
    )
    with compile_scene(declaration, backend=backend) as runtime:
        runtime.set_controls(
            {
                key: RobotCommand(np.zeros(1), feedforward=np.array([force]))
                for key, force in ((4, 1.0), (19, -1.0))
            }
        )
        for _ in range(20):
            assert dict(runtime.step(0.001)) == {4: True, 19: True}
        states = runtime.snapshot()
        for key, force in ((4, 1.0), (19, -1.0)):
            native_model = compiled.fresh_native_model()
            reference = mujoco.MjData(native_model)
            native_model.opt.timestep = 0.001
            reference.ctrl[:] = force
            for _ in range(20):
                mujoco.mj_step(native_model, reference)
            # Same mass, direct force, no contact or passive force. Allow the
            # documented finite-step position convention, not a fitted trace.
            np.testing.assert_allclose(states[key].v, reference.qvel, rtol=0, atol=2e-6)
            np.testing.assert_allclose(states[key].q, reference.qpos, rtol=0, atol=4e-5)
            effective_dt = 0.001 if backend == "host" else float(np.float32(0.001))
            assert states[key].time_s == pytest.approx(20 * effective_dt, abs=1e-15)
            assert states[key].tick == 20
        before = states[19].q.copy()
        runtime.reset((4,))
        np.testing.assert_array_equal(runtime.snapshot()[19].q, before)
        assert runtime.snapshot()[4].tick == 0
    # Recompilation owns fresh state and does not consume the declaration.
    with compile_scene(declaration, backend=backend) as second:
        assert all(state.tick == 0 for state in second.snapshot().values())


def test_compiled_asset_runs_host_from_one_declaration():
    _compiled_asset_case("host")


@pytest.mark.tyx
def test_compiled_asset_runs_device_from_one_declaration(tyx_mode):
    _compiled_asset_case(tyx_mode)


def _native_env_case(mode="cpu", implementation=None):
    pytest.importorskip("mujoco")
    from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    class Task:
        def controls(self, actions, keys):
            return {key: MuJoCoControl(actions[i]) for i, key in enumerate(keys)}

        def observe(self, states, keys):
            return np.array([[states[key].q[0], states[key].v[0]] for key in keys])

        def evaluate(self, before, after, actions, keys):
            return -np.array([after[key].q[0] ** 2 for key in keys]), np.zeros(
                len(keys), dtype=bool
            )

    compiled = compile_mujoco_xml(SLIDER_XML)
    with MuJoCoRuntime(
        compiled, world_keys=(4, 19), mode=mode, implementation=implementation
    ) as runtime:
        env = ResidentVectorEnv(runtime, Task(), (4, 19), action_size=1, dt=0.001, horizon=100)
        calls = []

        def policy(observation, due):
            calls.append(due.copy())
            return np.array([[1.0], [-1.0]])

        runner = PolicyRunner(env, policy, control_decimation=3)
        runner.reset()
        for _ in range(6):
            runner.step()
        states = runtime.snapshot()
        assert states[4].q[0] > 0 > states[19].q[0]
        np.testing.assert_allclose(states[4].v, -states[19].v, atol=1e-12)
        np.testing.assert_allclose([states[4].time, states[19].time], 0.006, rtol=0, atol=1e-8)
        assert len(calls) == 2
        unchanged = states[19].q.copy()
        runner.reset([0])
        states = runtime.snapshot()
        np.testing.assert_array_equal(states[19].q, unchanged)
        assert states[4].time == 0.0
        assert states[19].time == pytest.approx(0.006)
        np.testing.assert_array_equal(runner.snapshot().physical_ticks, [0, 6])


def test_native_runtime_consumed_by_existing_env_and_policy_runner():
    _native_env_case()


@pytest.mark.cuda
def test_explicit_warp_runtime_consumed_by_existing_env_and_policy_runner():
    pytest.importorskip("mujoco_warp")
    _native_env_case("cuda", "warp")


@pytest.mark.tyx
def test_tyx_mujoco_runtime_consumed_by_existing_env_and_policy_runner(tyx_mode):
    _native_env_case(tyx_mode, "tyx")
