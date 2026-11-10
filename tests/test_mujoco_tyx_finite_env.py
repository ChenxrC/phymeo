"""Public feedback environment consumes the finite box manifold and common force."""

import numpy as np
import pytest
from test_mujoco_tyx_finite_contacts import _ordinal_match
from test_mujoco_tyx_normal_env import XML as BASE_XML
from test_mujoco_tyx_normal_env import NormalTask

from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.control.policy_runner import PolicyRunner
from phymeo.training.resident_env import ResidentVectorEnv

XML = BASE_XML.replace(
    '<geom name="floor" type="plane" size="2 2 .1"/>',
    '<geom name="platform" type="box" pos="0 0 -.06" size=".5 .3 .06"/>',
).replace(
    '<geom name="left-foot" type="sphere" pos="-.12 0 -.1" size=".1"/>\n'
    '<geom name="right-foot" type="sphere" pos=".12 0 -.1" size=".1"/>',
    '<geom name="sole" type="box" pos="0 0 -.1" size=".14 .09 .1"/>',
)


@pytest.mark.parametrize("mode", ("interpret", pytest.param("cuda", marks=pytest.mark.cuda)))
def test_finite_box_policy_continuous_reference_and_reset(mode):
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("tyx")
    compiled = compile_mujoco_xml(XML, require_asset=False)
    native_model = mj.MjModel.from_xml_string(XML)
    assert native_model.ngeom == 2
    with MuJoCoRuntime(compiled, mode=mode, world_keys=(4, 31)) as runtime:
        assert runtime.capabilities["normal_candidate_slots"] == 24
        assert runtime.capabilities["normal_geometry_launches"] == 2
        env = ResidentVectorEnv(
            runtime, NormalTask(), (4, 31), action_size=2, dt=0.001, horizon=100
        )
        calls = []

        def policy(observation, due):
            calls.append(due.copy())
            return (
                -0.3 * observation[:, 7:9]
                - 0.1 * observation[:, 15:17]
                - 0.001 * observation[:, -1:]
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
        maximum_contacts = 0
        for step in range(12):
            transition = runner.step()
            for index, (key, state) in enumerate(runtime.snapshot().items()):
                native[key].ctrl[:] = transition.actions[index].astype(np.float32)
                mj.mj_step(native_model, native[key])
                receipt = state.last_step
                assert len(receipt.contacts) == native[key].ncon
                maximum_contacts = max(maximum_contacts, len(receipt.contacts))
                oracle_contacts = []
                for ordinal in range(native[key].ncon):
                    wrench = np.zeros(6)
                    mj.mj_contactForce(native_model, native[key], ordinal, wrench)
                    native_contact = native[key].contact[ordinal]
                    oracle_contacts.append(
                        dict(
                            geometry_ids=tuple(native_contact.geom),
                            feature_index=ordinal,
                            position=native_contact.pos.copy(),
                            frame=native_contact.frame.reshape(3, 3).copy(),
                            force=wrench[0],
                        )
                    )
                for contact, oracle in zip(
                    receipt.contacts, _ordinal_match(receipt.contacts, oracle_contacts), strict=True
                ):
                    np.testing.assert_allclose(
                        contact.normal_force, oracle["force"], rtol=3e-3, atol=3e-3
                    )
                    loaded |= contact.normal_force > 0
                assert receipt.numerical_solver == "normal-dual-active-set-v1"
                np.testing.assert_array_equal(receipt.control.ctrl, transition.actions[index])
                np.testing.assert_allclose(state.q, native[key].qpos, atol=2e-4, rtol=0)
                np.testing.assert_allclose(state.v, native[key].qvel, atol=3e-3, rtol=0)
                assert state.tick == step + 1
                assert receipt.raw_contact_fields["solver_diagnostics"]["converged"]
        assert loaded and maximum_contacts >= 2
        assert len(calls) == 4
        assert runtime.counters["completed_kernels"] == 12 * 6
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
