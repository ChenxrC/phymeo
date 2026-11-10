"""Real optional MuJoCo execution, publication and ownership contracts.

Authored without execution. The native engine is the oracle for integration;
analytic one-DOF and support checks also prevent a state-only facade passing.
"""

import importlib
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from phymeo.backends.mujoco_runtime import (
    MuJoCoControl,
    MuJoCoInitialState,
    MuJoCoRuntime,
)
from phymeo.training.resident_stepper import PhysicsStepFailure


SLIDER_XML = """<mujoco model="native-slider">
  <option timestep="0.002" gravity="0 0 0" integrator="Euler"/>
  <worldbody><body name="slider">
    <joint name="slide" type="slide" axis="1 0 0" armature="0.4" damping="0.3"/>
    <geom name="mass" type="sphere" size="0.1" mass="2"/>
  </body></worldbody>
  <actuator><motor name="drive" joint="slide" gear="3"/></actuator>
</mujoco>"""

FREE_XML = """<mujoco model="free-contact">
  <option timestep="0.001" gravity="0 0 -9.81" integrator="Euler"/>
  <worldbody>
    <geom name="floor" type="box" pos="0 0 -0.05" size="1 1 0.05"
          friction="0.7 0.01 0.001"/>
    <body name="ball" pos="0 0 0.099">
      <freejoint name="root"/>
      <geom name="ball-geom" type="sphere" size="0.1" mass="0.5"
            solref="0.005 1" friction="0.7 0.01 0.001"/>
    </body>
  </worldbody>
</mujoco>"""

TENDON_XML = """<mujoco model="native-tendon">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody><body name="one"><joint name="a" type="slide" axis="1 0 0"/>
    <geom type="sphere" size="0.1" mass="1"/>
    <body name="two" pos="0.3 0 0"><joint name="b" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </body></worldbody>
  <tendon><fixed name="sum"><joint joint="a" coef="1"/>
    <joint joint="b" coef="2"/></fixed></tendon>
  <actuator><motor name="tendon-drive" tendon="sum"/></actuator>
</mujoco>"""


@pytest.fixture
def mj():
    return pytest.importorskip("mujoco", reason="optional CPU MuJoCo dependency")


def compiled(xml):
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    return compile_mujoco_xml(xml, require_asset=False)


def native_contacts(mj, model, data):
    result = []
    for index in range(data.ncon):
        wrench = np.empty(6, dtype=np.float64)
        mj.mj_contactForce(model, data, index, wrench)
        result.append((data.contact[index], wrench))
    return result


def test_module_import_is_lazy_and_missing_dependency_is_explicit(monkeypatch):
    module = importlib.import_module("phymeo.backends.mujoco_runtime")
    monkeypatch.setitem(sys.modules, "mujoco", None)
    with pytest.raises(ImportError, match="optional.*MuJoCo|MuJoCo.*optional"):
        module.MuJoCoRuntime(object())


def test_native_motor_gear_armature_damping_and_sparse_world_isolation(mj):
    record = compiled(SLIDER_XML)
    with MuJoCoRuntime(record, world_keys=(4, 91)) as runtime:
        runtime.set_controls({4: MuJoCoControl([2.0]), 91: {"drive": -1.0}})
        assert runtime.snapshot()[4].time == 0.0
        assert runtime.step(0.002) == {4: True, 91: True}
        states = runtime.snapshot()
        assert states[4].v[0] == pytest.approx(0.002 * 6.0 / (2.4 + 0.002 * 0.3), rel=1e-12)
        assert states[91].v[0] == pytest.approx(-0.002 * 3.0 / (2.4 + 0.002 * 0.3), rel=1e-12)
        assert states[4].q[0] == pytest.approx(0.002 * states[4].v[0], rel=1e-12)
        assert states[4].coordinate_system == "mujoco-native-qpos-qvel-v1"
        assert states[4].contact_law == "mujoco-native-constraint-v1"
        assert states[4].tick == states[91].tick == 1
        assert states[4].last_step.input_time == 0.0
        assert states[4].last_step.output_time == 0.002
        runtime.step(0.002)
        assert runtime.snapshot()[4].v[0] < 2 * states[4].v[0]


def test_native_step_matches_direct_original_model_with_full_controls(mj):
    record = compiled(SLIDER_XML)
    direct = mj.MjData(record.mj_model)
    mj.mj_forward(record.mj_model, direct)
    with MuJoCoRuntime(record) as runtime:
        runtime.set_controls({0: MuJoCoControl([0.7], qfrc_applied=[0.2])})
        for _ in range(12):
            direct.ctrl[:] = [0.7]
            direct.qfrc_applied[:] = [0.2]
            mj.mj_step(record.mj_model, direct)
            runtime.step(0.002)
            state = runtime.snapshot()[0]
            np.testing.assert_array_equal(state.q, direct.qpos)
            np.testing.assert_array_equal(state.v, direct.qvel)
            assert state.time == direct.time


def test_free_body_contact_reads_real_solver_force_and_endpoint_phase(mj):
    record = compiled(FREE_XML)
    direct = mj.MjData(record.mj_model)
    mj.mj_forward(record.mj_model, direct)
    with MuJoCoRuntime(record) as runtime:
        runtime.set_controls({0: MuJoCoControl([])})
        for _ in range(120):
            mj.mj_step(record.mj_model, direct)
            runtime.step(0.001)
        state = runtime.snapshot()[0]
        np.testing.assert_array_equal(state.q, direct.qpos)
        np.testing.assert_array_equal(state.v, direct.qvel)
        assert state.q.shape == (7,) and state.v.shape == (6,)
        assert state.contacts and state.last_step.contacts
        assert state.contact_phase == "endpoint-forward-recomputed"
        assert state.contact_time == state.time
        assert state.last_step.contact_phase == "last-mj-step-constraint-evaluation"
        assert state.last_step.contact_time is None
        native = native_contacts(mj, record.mj_model, direct)
        assert len(native) == len(state.last_step.contacts)
        for saved, (contact, wrench) in zip(state.last_step.contacts, native, strict=True):
            np.testing.assert_array_equal(saved.wrench_contact_frame, wrench)
            np.testing.assert_array_equal(saved.frame, contact.frame.reshape(3, 3))
            assert saved.geometry_ids == tuple(contact.geom)
            assert saved.distance == contact.dist
        normal_force = sum(contact.wrench_contact_frame[0] for contact in state.contacts)
        assert normal_force == pytest.approx(0.5 * 9.81, rel=0.08)
        assert 0.095 < state.q[2] < 0.105
        assert np.linalg.norm(state.v) < 0.02


def test_full_native_tendon_model_runs_without_primitive_lowering(mj):
    record = compiled(TENDON_XML)
    assert record.mj_model.ntendon == 1
    with MuJoCoRuntime(record, world_keys=(8,)) as runtime:
        runtime.set_controls({8: MuJoCoControl([1.0])})
        runtime.step(0.002)
        state = runtime.snapshot()[8]
        assert state.q.shape == state.v.shape == (2,)
        assert np.linalg.norm(state.v) > 0
        assert runtime.capabilities["native_model"] is True
        assert runtime.capabilities["backend"] == "mujoco-cpu"


def test_snapshot_and_packets_are_owned_deeply_immutable(mj):
    command = np.array([2.0])
    packet = MuJoCoControl(command)
    command[:] = 100
    record = compiled(SLIDER_XML)
    with MuJoCoRuntime(record) as runtime:
        runtime.set_controls({0: packet})
        before = runtime.snapshot()[0]
        with pytest.raises(ValueError):
            before.q.setflags(write=True)
        with pytest.raises(ValueError):
            packet.ctrl.setflags(write=True)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            before.tick = 99
        runtime.step(0.002)
        assert before.tick == 0 and before.time == 0.0
        np.testing.assert_array_equal(before.q, [0.0])
        np.testing.assert_array_equal(runtime.snapshot()[0].ctrl, [2.0])


def test_controls_validate_whole_batch_before_replacing_any_packet(mj):
    with MuJoCoRuntime(compiled(SLIDER_XML), world_keys=(3, 11)) as runtime:
        runtime.set_controls({3: MuJoCoControl([1]), 11: MuJoCoControl([2])})
        for invalid in (
            {3: MuJoCoControl([7])},
            {3: MuJoCoControl([7]), 11: MuJoCoControl([1, 2])},
            {3: {"drive": 7}, 11: {"wrong": 4}},
        ):
            with pytest.raises((ValueError, TypeError)):
                runtime.set_controls(invalid)
        np.testing.assert_array_equal(runtime.snapshot()[3].held_control.ctrl, [1])
        np.testing.assert_array_equal(runtime.snapshot()[11].held_control.ctrl, [2])
        with pytest.raises(ValueError):
            MuJoCoControl([float("nan")])


def test_model_options_and_clone_are_owned_and_reset_is_selected(mj):
    record = compiled(SLIDER_XML)
    with MuJoCoRuntime(record, world_keys=(5, 27)) as runtime:
        record.mj_model.opt.timestep = 0.01
        runtime.set_controls({5: MuJoCoControl([1]), 27: MuJoCoControl([2])})
        runtime.step(0.002)
        with runtime.clone() as clone:
            first = runtime.snapshot()
            np.testing.assert_array_equal(first[5].q, clone.snapshot()[5].q)
            assert clone.snapshot()[5].tick == 1
            clone.set_controls({5: MuJoCoControl([-4]), 27: MuJoCoControl([-3])})
            clone.step(0.002)
            np.testing.assert_array_equal(first[5].q, runtime.snapshot()[5].q)
            runtime.reset((5,))
            after = runtime.snapshot()
            assert after[5].time == 0.0 and after[5].tick == 0
            np.testing.assert_array_equal(after[27].q, first[27].q)
            assert after[27].tick == 1
            assert clone.snapshot()[5].tick == 2


def test_explicit_initial_state_restores_native_free_root_without_injection(mj):
    record = compiled(FREE_XML)
    qpos = record.mj_model.qpos0.copy()
    qpos[2] = 0.4
    initial = MuJoCoInitialState(q=qpos, v=[0.1, 0, 0, 0, 0, 0])
    with MuJoCoRuntime(record, world_keys=(17,), initial_states={17: initial}) as runtime:
        runtime.step(0.001)
        assert runtime.snapshot()[17].q[0] > 0
        runtime.reset()
        state = runtime.snapshot()[17]
        np.testing.assert_array_equal(state.q, qpos)
        np.testing.assert_array_equal(state.v, [0.1, 0, 0, 0, 0, 0])
        assert state.time == 0 and state.tick == 0


def test_partial_native_failure_reports_only_published_worlds_and_requires_reset(mj, monkeypatch):
    with MuJoCoRuntime(compiled(SLIDER_XML), world_keys=(7, 19, 23)) as runtime:
        runtime.set_controls({7: MuJoCoControl([1]), 19: MuJoCoControl([-1]), 23: MuJoCoControl([2])})
        original = mj.mj_step

        def fail_second(model, data):
            if data.ctrl[0] < 0:
                raise RuntimeError("injected native failure before candidate publication")
            return original(model, data)

        monkeypatch.setattr(mj, "mj_step", fail_second)
        with pytest.raises(PhysicsStepFailure) as error:
            runtime.step(0.002)
        assert error.value.acceptance_known is True
        assert error.value.accepted_worlds == {7: True, 19: False, 23: False}
        assert isinstance(error.value.__cause__, RuntimeError)
        assert runtime.snapshot()[7].tick == 1
        assert runtime.snapshot()[19].tick == runtime.snapshot()[23].tick == 0
        with pytest.raises(RuntimeError, match="reset"):
            runtime.step(0.002)
        with pytest.raises(ValueError, match="full|all"):
            runtime.reset((19,))
        runtime.reset()
        monkeypatch.setattr(mj, "mj_step", original)
        assert runtime.step(0.002) == {7: True, 19: True, 23: True}


def test_failed_candidate_after_actual_integration_is_not_published(mj, monkeypatch):
    with MuJoCoRuntime(compiled(SLIDER_XML)) as runtime:
        original = mj.mj_step

        def fail_after_native(model, data):
            original(model, data)
            raise RuntimeError("native completed candidate then callback failed")

        monkeypatch.setattr(mj, "mj_step", fail_after_native)
        with pytest.raises(PhysicsStepFailure) as error:
            runtime.step(0.002)
        assert error.value.accepted_worlds == {0: False}
        assert error.value.physics_accepted is False
        assert runtime.snapshot()[0].time == 0.0
        monkeypatch.setattr(mj, "mj_step", original)
        assert runtime.step(0.002) == {0: True}


def test_snapshot_forward_does_not_change_next_solver_trajectory(mj):
    record = compiled(FREE_XML)
    with MuJoCoRuntime(record) as observed, MuJoCoRuntime(record) as untouched:
        for _ in range(5):
            observed.snapshot()
        for _ in range(12):
            observed.step(0.001)
            untouched.step(0.001)
            observed.snapshot()
            observed.snapshot()
        np.testing.assert_array_equal(observed.snapshot()[0].q, untouched.snapshot()[0].q)
        np.testing.assert_array_equal(observed.snapshot()[0].v, untouched.snapshot()[0].v)


def test_timestep_selection_and_close_fail_before_native_advance(mj):
    runtime = MuJoCoRuntime(compiled(SLIDER_XML), world_keys=(2, 100))
    for invalid in (0.001, 0.0, float("nan")):
        with pytest.raises(ValueError):
            runtime.step(invalid)
    with pytest.raises((ValueError, KeyError)):
        runtime.snapshot((99,))
    assert runtime.snapshot()[2].tick == 0
    runtime.close()
    runtime.close()
    for operation in (runtime.snapshot, runtime.reset, lambda: runtime.step(0.002)):
        with pytest.raises(RuntimeError, match="closed"):
            operation()


@pytest.mark.parametrize("keys", [(), (3, 3), (-1,), (True,), (1.5,)])
def test_world_identity_is_explicit_and_unique(mj, keys):
    with pytest.raises((TypeError, ValueError)):
        MuJoCoRuntime(compiled(SLIDER_XML), world_keys=keys)


def test_gpu_mode_never_silently_falls_back_to_cpu(mj):
    with pytest.raises(NotImplementedError, match="Warp|GPU"):
        MuJoCoRuntime(compiled(SLIDER_XML), mode="warp")


def test_runtime_uses_frozen_source_even_if_public_native_model_was_mutated(mj):
    record = compiled(SLIDER_XML)
    record.mj_model.opt.timestep = 0.05
    record.mj_model.body_mass[1] = 500
    with MuJoCoRuntime(record) as runtime:
        runtime.set_controls({0: MuJoCoControl([2])})
        runtime.step(0.002)
        assert runtime.timestep == 0.002
        assert runtime.snapshot()[0].v[0] == pytest.approx(
            0.002 * 6.0 / (2.4 + 0.002 * 0.3), rel=1e-12
        )


def test_reset_prepare_failure_preserves_all_owned_worlds(mj, monkeypatch):
    with MuJoCoRuntime(compiled(SLIDER_XML), world_keys=(3, 77)) as runtime:
        runtime.set_controls({3: MuJoCoControl([1]), 77: MuJoCoControl([2])})
        runtime.step(0.002)
        before = runtime.snapshot()
        original = mj.mj_forward
        calls = []

        def fail_second(model, data):
            calls.append(True)
            original(model, data)
            if len(calls) == 2:
                raise RuntimeError("injected reset preparation failure")

        monkeypatch.setattr(mj, "mj_forward", fail_second)
        with pytest.raises(RuntimeError, match="reset preparation"):
            runtime.reset()
        monkeypatch.setattr(mj, "mj_forward", original)
        for key, state in runtime.snapshot().items():
            np.testing.assert_array_equal(state.q, before[key].q)
            assert state.tick == before[key].tick and state.time == before[key].time
