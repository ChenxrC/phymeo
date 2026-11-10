"""Native floating coordinates through the existing policy/environment protocol."""

import numpy as np
import pytest

from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.control.policy_runner import PolicyRunner
from phymeo.training.resident_env import ResidentVectorEnv

XML = """<mujoco model="floating-policy"><compiler angle="radian" fusestatic="false"/>
<option timestep=".001" gravity=".2 -.3 -9.81"/>
<worldbody><body name="fixed-marker" pos="2 0 0">
<inertial mass=".4" pos="0 0 0" diaginertia=".02 .03 .04"/></body>
<body name="base" pos=".1 -.2 .8" quat=".9238795325112867 0 .3826834323650898 0">
<freejoint/><inertial pos=".04 -.03 .06" mass="3" fullinertia=".2 .25 .3 .01 -.008 .006"/>
<body name="upper" pos=".15 .03 -.1"><joint name="shoulder" axis="0 1 0" damping=".08" armature=".02"/>
<inertial pos=".2 0 -.1" mass="1.2" diaginertia=".03 .04 .02"/>
<body name="lower" pos=".4 0 -.2"><joint name="elbow" axis="1 0 0" damping=".06" armature=".01"/>
<inertial pos=".15 .02 -.1" mass=".6" diaginertia=".012 .015 .009"/>
</body></body></body></worldbody>
<actuator><motor joint="shoulder"/><motor joint="elbow"/></actuator></mujoco>"""


class FloatingTask:
    def controls(self, actions, keys):
        return {key: MuJoCoControl(actions[index]) for index, key in enumerate(keys)}

    def observe(self, states, keys):
        return np.array([np.r_[states[key].q, states[key].v] for key in keys])

    def evaluate(self, before, after, actions, keys):
        return -np.array([np.sum(after[key].q[7:] ** 2) for key in keys]), np.zeros(
            len(keys), dtype=bool
        )


@pytest.mark.parametrize("mode", ("interpret", pytest.param("cuda", marks=pytest.mark.cuda)))
def test_floating_tree_policy_continuous_native_reference_and_selected_reset(mode):
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("tyx")
    compiled = compile_mujoco_xml(XML, require_asset=False)
    oracle_model = mj.MjModel.from_xml_string(XML)
    initial = {}
    for slot, key in enumerate((4, 31)):
        q = oracle_model.qpos0.copy()
        q[:3] += np.array([0.02, -0.01, 0.03]) * slot
        q[7:] = [0.2 - 0.1 * slot, -0.3 + 0.2 * slot]
        v = np.array([0.1, -0.2, 0.15, 0.3, -0.25, 0.2, 0.4, -0.2]) * (1 + 0.2 * slot)
        initial[key] = MuJoCoInitialState(q, v)
    with MuJoCoRuntime(compiled, world_keys=(4, 31), initial_states=initial, mode=mode) as runtime:
        env = ResidentVectorEnv(
            runtime, FloatingTask(), (4, 31), action_size=2, dt=0.001, horizon=100
        )
        calls = []

        def policy(observation, due):
            calls.append(due.copy())
            # Native nq=9, nv=8: scalar q starts7, scalar v starts6.
            return -1.5 * observation[:, 7:9] - 0.2 * observation[:, 15:17]

        runner = PolicyRunner(env, policy, control_decimation=3)
        runner.reset()
        native = {}
        for key, state in runtime.snapshot().items():
            data = mj.MjData(oracle_model)
            data.qpos[:], data.qvel[:] = state.q, state.v
            native[key] = data
        for step in range(12):
            transition = runner.step()
            states = runtime.snapshot()
            for index, key in enumerate((4, 31)):
                native[key].ctrl[:] = transition.actions[index].astype(np.float32)
                mj.mj_step(oracle_model, native[key])
                state = states[key]
                receipt = state.last_step
                np.testing.assert_array_equal(receipt.control.ctrl, transition.actions[index])
                assert receipt.dynamics.mass_matrix.shape == (8, 8)
                assert receipt.dynamics.input_q.shape == (9,)
                assert not receipt.dynamics.mass_matrix.flags.writeable
                assert state.tick == step + 1
                np.testing.assert_allclose(state.q, native[key].qpos, atol=2e-4, rtol=0)
                np.testing.assert_allclose(state.v, native[key].qvel, atol=2e-3, rtol=0)
        assert len(calls) == 4
        with runtime.clone((31,)) as untouched:
            before = runtime.snapshot()[31]
            runner.reset([0])
            np.testing.assert_array_equal(runtime.snapshot()[31].q, before.q)
            transition = runner.step()
            untouched.set_controls({31: MuJoCoControl(transition.actions[1])})
            untouched.step(0.001)
            actual, expected = runtime.snapshot()[31], untouched.snapshot()[31]
            np.testing.assert_array_equal(actual.q, expected.q)
            np.testing.assert_array_equal(actual.v, expected.v)
            assert actual.tick == expected.tick == 13
        np.testing.assert_array_equal(runner.snapshot().physical_ticks, [1, 13])
        assert not runner.needs_reset
