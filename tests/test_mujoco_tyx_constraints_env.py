"""Mixed friction and soft limits through the public policy/environment protocol."""

import numpy as np
import pytest

from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.control.policy_runner import PolicyRunner
from phymeo.training.resident_env import ResidentVectorEnv

XML = """<mujoco model="normal-policy"><compiler angle="radian" fusestatic="false"/>
<option timestep=".001" gravity="0 0 -9.81" iterations="100" tolerance="1e-10"/>
<default><geom condim="6" friction=".5 .02 .01" solref=".02 1" solimp=".9 .95 .01 .5 2"/></default>
<worldbody><geom name="floor" type="plane" size="2 2 .1"/>
<body name="base" pos="0 0 .19" quat=".9998 0 .02 0"><freejoint/>
<inertial pos=".02 0 .03" mass="3" fullinertia=".2 .25 .3 .01 -.008 .006"/>
<geom name="left-foot" type="sphere" pos="-.12 0 -.1" size=".1"/>
<geom name="right-foot" type="sphere" pos=".12 0 -.1" size=".1"/>
<body name="upper" pos=".1 .02 .2"><joint name="shoulder" axis="0 1 0" damping=".08" armature=".02" limited="true" range="-.02 .02" margin=".03" frictionloss=".03"/>
<inertial pos=".1 0 .05" mass=".6" diaginertia=".03 .04 .02"/>
<body name="lower" pos=".2 0 .1"><joint name="elbow" axis="1 0 0" damping=".06" armature=".01"/>
<inertial pos=".1 .02 .05" mass=".4" diaginertia=".012 .015 .009"/>
</body></body></body></worldbody>
<actuator><motor joint="shoulder"/><motor joint="elbow"/></actuator></mujoco>"""


class MixedTask:
    def controls(self, actions, keys):
        return {key: MuJoCoControl(actions[index]) for index, key in enumerate(keys)}

    def observe(self, states, keys):
        values = []
        for key in keys:
            state = states[key]
            force = (
                0.0
                if state.last_step is None
                else sum(contact.normal_force for contact in state.last_step.contacts)
            )
            tangent = np.zeros(5)
            limit_force = dry_force = 0.0
            if state.last_step is not None:
                for contact in state.last_step.contacts:
                    tangent += contact.wrench_contact_frame[1:]
                for row in state.last_step.constraints:
                    if row.constraint_type == 3:
                        limit_force += row.force * row.jacobian.sum()
                    elif row.constraint_type == 1:
                        dry_force += row.force
            values.append(np.r_[state.q, state.v, force, tangent, limit_force, dry_force])
        return np.asarray(values)

    def evaluate(self, before, after, actions, keys):
        return -np.array([np.sum(after[key].q[7:] ** 2) for key in keys]), np.zeros(len(keys), bool)


@pytest.mark.parametrize("mode", ("interpret", pytest.param("cuda", marks=pytest.mark.cuda)))
def test_mixed_policy_continuous_reference_and_reset(mode, tmp_path):
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("tyx")
    compiled = compile_mujoco_xml(XML, require_asset=False)
    native_model = mj.MjModel.from_xml_string(XML)
    with MuJoCoRuntime(compiled, mode=mode, world_keys=(4, 31)) as runtime:
        env = ResidentVectorEnv(runtime, MixedTask(), (4, 31), action_size=2, dt=0.001, horizon=100)
        calls = []

        def policy(observation, due):
            calls.append(due.copy())
            # Control consumes the last accepted device contact force, not endpoint CPU force.
            return (
                -0.3 * observation[:, 7:9]
                - 0.1 * observation[:, 15:17]
                - 0.001 * observation[:, 17:18]
                - 0.0001 * observation[:, 18:].sum(axis=1, keepdims=True)
            )

        runner = PolicyRunner(env, policy, control_decimation=3)
        runner.reset()
        initial = runtime.snapshot()
        native = {}
        for key, state in initial.items():
            data = mj.MjData(native_model)
            data.qpos[:], data.qvel[:] = state.q, state.v
            native[key] = data
        loaded = False
        unloaded_row = False
        load_history = []
        for step in range(12):
            try:
                transition = runner.step()
            except Exception:
                np.savez_compressed(tmp_path / "rejected-device.npz", **runtime._device.download())
                raise
            for index, (key, state) in enumerate(runtime.snapshot().items()):
                native[key].ctrl[:] = transition.actions[index].astype(np.float32)
                mj.mj_step(native_model, native[key])
                receipt = state.last_step
                native_forces = []
                for contact in range(native[key].ncon):
                    wrench = np.zeros(6)
                    mj.mj_contactForce(native_model, native[key], contact, wrench)
                    native_forces.append(wrench[0])
                actual_forces = [c.normal_force for c in receipt.contacts]
                assert receipt.constraints
                assert all(c.dimension == 6 for c in receipt.contacts)
                assert any(row.constraint_type == 1 for row in receipt.constraints)
                assert any(row.constraint_type == 3 for row in receipt.constraints)
                np.testing.assert_allclose(actual_forces, native_forces, rtol=3e-3, atol=3e-3)
                load_history.append((step, key, actual_forces, native_forces))
                assert receipt.numerical_solver == "pyramidal-box-active-set-v1"
                assert receipt.dynamics.mass_matrix.shape == (8, 8)
                assert receipt.dynamics.input_q.shape == (9,)
                np.testing.assert_array_equal(receipt.control.ctrl, transition.actions[index])
                assert state.tick == step + 1
                np.testing.assert_allclose(state.q, native[key].qpos, atol=2e-4, rtol=0)
                np.testing.assert_allclose(state.v, native[key].qvel, atol=3e-3, rtol=0)
                assert receipt.raw_contact_fields["solver_diagnostics"]["converged"]
                # The original tilted fixture has two detected contacts but
                # native complementarity can unload one. Preserve that row and
                # compare both forces instead of requiring two positive forces.
                loaded |= len(receipt.contacts) == 2 and any(
                    c.normal_force > 0 for c in receipt.contacts
                )
                unloaded_row |= len(receipt.contacts) == 2 and any(
                    c.normal_force == 0 for c in receipt.contacts
                )
        assert loaded, load_history
        assert all(len(row[2]) == 2 for row in load_history)
        assert len(calls) == 4
        with runtime.clone((31,)) as untouched:
            before = runtime.snapshot()[31]
            runner.reset([0])
            reset = runtime.snapshot()
            np.testing.assert_array_equal(reset[4].q, initial[4].q)
            np.testing.assert_array_equal(reset[4].v, initial[4].v)
            np.testing.assert_array_equal(reset[31].q, before.q)
            transition = runner.step()
            untouched.set_controls({31: MuJoCoControl(transition.actions[1])})
            untouched.step(0.001)
            actual, expected = runtime.snapshot()[31], untouched.snapshot()[31]
            np.testing.assert_array_equal(actual.q, expected.q)
            np.testing.assert_array_equal(actual.v, expected.v)
            np.testing.assert_array_equal(
                actual.last_step.qfrc_constraint, expected.last_step.qfrc_constraint
            )
            assert actual.tick == expected.tick == 13
        np.testing.assert_array_equal(runner.snapshot().physical_ticks, [1, 13])
        assert not runner.needs_reset
