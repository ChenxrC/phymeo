"""A rotating native tree consumed through the existing environment protocol."""

import numpy as np
import pytest

from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.control.policy_runner import PolicyRunner
from phymeo.training.resident_env import ResidentVectorEnv

XML = """<mujoco model="tree-policy"><compiler angle="radian"/>
<option timestep=".001" gravity="0 0 -9.81"/>
<worldbody><body name="upper"><joint name="shoulder" axis="0 1 0" damping=".08" armature=".02"/>
<inertial pos=".2 0 -.1" mass="1.2" diaginertia=".03 .04 .02"/>
<body name="lower" pos=".4 0 -.2"><joint name="elbow" axis="0 1 0" damping=".06" armature=".01"/>
<inertial pos=".15 .02 -.1" mass=".6" diaginertia=".012 .015 .009"/>
</body></body></worldbody>
<actuator><motor joint="shoulder"/><motor joint="elbow"/></actuator></mujoco>"""


class TreeTask:
    def controls(self, actions, keys):
        return {key: MuJoCoControl(actions[index]) for index, key in enumerate(keys)}

    def observe(self, states, keys):
        return np.array([np.r_[states[key].q, states[key].v] for key in keys])

    def evaluate(self, before, after, actions, keys):
        return -np.array([np.sum(after[key].q ** 2) for key in keys]), np.zeros(
            len(keys), dtype=bool
        )


@pytest.mark.parametrize("mode", ("interpret", pytest.param("cuda", marks=pytest.mark.cuda)))
def test_rotating_tree_policy_holds_controls_and_resets_one_world(mode):
    pytest.importorskip("mujoco")
    pytest.importorskip("tyx")
    compiled = compile_mujoco_xml(XML, require_asset=False)
    initial = {
        4: MuJoCoInitialState([0.2, -0.3], [0.4, -0.2]),
        31: MuJoCoInitialState([-0.15, 0.25], [-0.3, 0.5]),
    }
    with MuJoCoRuntime(compiled, world_keys=(4, 31), initial_states=initial, mode=mode) as runtime:
        env = ResidentVectorEnv(runtime, TreeTask(), (4, 31), action_size=2, dt=0.001, horizon=100)
        calls = []

        def policy(observation, due):
            calls.append(due.copy())
            return -1.5 * observation[:, :2] - 0.2 * observation[:, 2:]

        runner = PolicyRunner(env, policy, control_decimation=3)
        runner.reset()
        first_mass = None
        for step in range(12):
            transition = runner.step()
            states = runtime.snapshot()
            for index, key in enumerate((4, 31)):
                receipt = states[key].last_step
                np.testing.assert_array_equal(receipt.control.ctrl, transition.actions[index])
                np.testing.assert_array_equal(
                    receipt.stored_control.ctrl, transition.actions[index].astype(np.float32)
                )
                assert receipt.backend == f"mujoco-tyx-{mode}"
                assert receipt.dynamics.mass_matrix.shape == (2, 2)
                assert not receipt.dynamics.mass_matrix.flags.writeable
                assert states[key].tick == step + 1
            if step == 0:
                first_mass = states[4].last_step.dynamics.mass_matrix.copy()
        assert len(calls) == 4
        assert not np.array_equal(first_mass, states[4].last_step.dynamics.mass_matrix)
        with runtime.clone((31,)) as control_world:
            before = runtime.snapshot()[31]
            runner.reset([0])
            np.testing.assert_array_equal(runtime.snapshot()[31].q, before.q)
            transition = runner.step()
            control_world.set_controls({31: MuJoCoControl(transition.actions[1])})
            control_world.step(0.001)
            after, expected = runtime.snapshot()[31], control_world.snapshot()[31]
            np.testing.assert_array_equal(after.q, expected.q)
            np.testing.assert_array_equal(after.v, expected.v)
            assert after.tick == expected.tick == 13
            assert runtime.snapshot()[4].tick == 1
        np.testing.assert_array_equal(runner.snapshot().physical_ticks, [1, 13])
        assert not runner.needs_reset
