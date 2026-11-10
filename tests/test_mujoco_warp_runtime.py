"""Independent CUDA adapter cases; original native law with engineering error.

These exercise the new runtime, not the earlier dependency compatibility smoke.
Every accepted step must come from mjwarp.step on cuda, including contact loads.
"""

import sys

import numpy as np
import pytest

from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime
from phymeo.training.resident_stepper import PhysicsStepFailure

pytestmark = pytest.mark.cuda

SLIDER = """<mujoco><option gravity="0 0 0" timestep="0.002"/>
<worldbody><body><joint type="slide" axis="1 0 0" name="slide"
 armature="0.4" damping="0.3"/><geom type="sphere" size="0.1" mass="2"/>
</body></worldbody><actuator><motor name="drive" joint="slide" gear="3"/>
</actuator></mujoco>"""

CONTACT = """<mujoco><option timestep="0.001" gravity="0 0 -9.81"/>
<worldbody><geom type="box" name="floor" pos="0 0 -0.05" size="1 1 0.05"/>
<body name="ball" pos="0 0 0.099"><freejoint/>
<geom name="ball-geom" type="sphere" size="0.1" mass="0.5" solref="0.005 1"/>
</body></worldbody></mujoco>"""


@pytest.fixture
def deps():
    mj = pytest.importorskip("mujoco")
    mw = pytest.importorskip("mujoco_warp")
    wp = pytest.importorskip("warp")
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("requires actual CUDA device")
    return mj, mw, wp


def record(xml):
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    return compile_mujoco_xml(xml, require_asset=False)


def gpu(compilation, **options):
    return MuJoCoRuntime(compilation, mode="cuda", device="cuda:0", **options)


def test_sparse_loaded_worlds_run_actual_cuda_with_native_control_law(deps):
    mj, _, _ = deps
    compiled = record(SLIDER)
    with gpu(compiled, world_keys=(4, 91)) as runtime:
        assert runtime.capabilities["backend"] == "mujoco-warp-cuda"
        assert runtime.capabilities["device"].startswith("cuda:")
        assert runtime.capabilities["gpu"] is True
        assert runtime.capabilities["state_precision"] == "fp32-device-integration"
        runtime.set_controls({4: MuJoCoControl([0.7]), 91: MuJoCoControl([-1.2])})
        refs = {key: mj.MjData(compiled.mj_model) for key in (4, 91)}
        for _ in range(12):
            for key, force in ((4, 0.7), (91, -1.2)):
                refs[key].ctrl[:] = force
                mj.mj_step(compiled.mj_model, refs[key])
            assert runtime.step(0.002) == {4: True, 91: True}
        for key, state in runtime.snapshot().items():
            np.testing.assert_allclose(state.q, refs[key].qpos, rtol=0, atol=2e-7)
            np.testing.assert_allclose(state.v, refs[key].qvel, rtol=0, atol=2e-7)
            assert state.tick == 12
            assert state.last_step.backend == "mujoco-warp-cuda"
            assert state.last_step.contact_phase == "last-mjwarp-step-constraint-evaluation"


def test_cuda_selected_reset_clone_and_owned_model_keep_other_world_exact(deps):
    compiled = record(SLIDER)
    with gpu(compiled, world_keys=(5, 29)) as runtime:
        compiled.mj_model.opt.timestep = 1.0
        runtime.set_controls({5: MuJoCoControl([1]), 29: MuJoCoControl([-2])})
        runtime.step(0.002)
        before = runtime.snapshot()
        with runtime.clone() as copied:
            copied.step(0.002)
            assert copied.snapshot()[5].tick == 2
            assert runtime.snapshot()[5].tick == 1
            runtime.reset((5,))
            after = runtime.snapshot()
            assert after[5].tick == 0 and after[5].time == 0
            assert after[29].tick == 1
            np.testing.assert_array_equal(after[29].q, before[29].q)
            np.testing.assert_array_equal(after[29].v, before[29].v)
            assert after[29].time == before[29].time
            assert copied.snapshot()[5].tick == 2


def test_cuda_free_root_load_keeps_real_gpu_contact_fields_and_wrench(deps):
    with gpu(record(CONTACT), world_keys=(8, 103), contact_capacity=16,
             constraint_capacity=64) as runtime:
        runtime.set_controls({8: MuJoCoControl([]), 103: MuJoCoControl([])})
        for _ in range(120):
            runtime.step(0.001)
        for state in runtime.snapshot().values():
            assert state.contact_law == "mujoco-warp-native-constraint-v0.0.2"
            assert state.contact_phase == "endpoint-cpu-forward-recomputed"
            contacts = state.last_step.contacts
            assert contacts
            force = sum(contact.wrench_contact_frame[0] for contact in contacts)
            assert force == pytest.approx(0.5 * 9.81, rel=0.12)
            assert 0.095 < state.q[2] < 0.105
            assert np.linalg.norm(state.v) < 0.03
            raw = state.last_step.raw_contact_fields
            assert set(raw) >= {"dist", "pos", "frame", "geom", "worldid", "type",
                                "geomcollisionid", "efc_address", "solref", "solimp"}
            assert len(raw["dist"]) == len(contacts)
            assert state.last_step.raw_constraint_fields["force"].size > 0
            with pytest.raises(ValueError):
                raw["dist"].setflags(write=True)


def test_cuda_overflow_rejected_before_clipping_readback_or_publication(deps, monkeypatch):
    _, mw, wp = deps
    with gpu(record(SLIDER), world_keys=(7, 19), contact_capacity=4,
             constraint_capacity=16) as runtime:
        runtime.set_controls({7: MuJoCoControl([1]), 19: MuJoCoControl([-1])})
        before = runtime.snapshot()
        original_step = mw.step
        original_readback = mw.get_data_into
        readbacks = []

        def overflow(model, data):
            original_step(model, data)
            wp.copy(data.nacon, wp.array([data.naconmax + 1], dtype=wp.int32))

        def record_readback(*args, **kwargs):
            readbacks.append(True)
            return original_readback(*args, **kwargs)

        monkeypatch.setattr(mw, "step", overflow)
        monkeypatch.setattr(mw, "get_data_into", record_readback)
        with pytest.raises(PhysicsStepFailure) as error:
            runtime.step(0.002)
        assert error.value.accepted_worlds == {7: False, 19: False}
        assert error.value.acceptance_known is True
        assert not readbacks
        for key, state in runtime.snapshot().items():
            np.testing.assert_array_equal(state.q, before[key].q)
            assert state.time == before[key].time and state.tick == before[key].tick
        monkeypatch.setattr(mw, "step", original_step)
        assert runtime.step(0.002) == {7: True, 19: True}


def test_cuda_late_readback_failure_retains_entire_original_batch(deps, monkeypatch):
    _, mw, _ = deps
    with gpu(record(SLIDER), world_keys=(2, 17)) as runtime:
        original = mw.get_data_into

        def fail_second(result, model, data, world_id=0):
            if world_id == 1:
                raise RuntimeError("injected second-world CUDA readback failure")
            return original(result, model, data, world_id)

        monkeypatch.setattr(mw, "get_data_into", fail_second)
        with pytest.raises(PhysicsStepFailure) as error:
            runtime.step(0.002)
        assert error.value.accepted_worlds == {2: False, 17: False}
        assert all(state.tick == 0 for state in runtime.snapshot().values())
        monkeypatch.setattr(mw, "get_data_into", original)
        assert runtime.step(0.002) == {2: True, 17: True}


def test_missing_cuda_dependency_is_an_error_not_cpu_fallback(monkeypatch):
    pytest.importorskip("mujoco")
    compiled = record(SLIDER)
    monkeypatch.setitem(sys.modules, "mujoco_warp", None)
    with pytest.raises(ImportError, match="MuJoCo-Warp|mujoco_warp"):
        gpu(compiled)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_cuda_capacities_are_explicit_positive_integers(deps, capacity):
    with pytest.raises((TypeError, ValueError)):
        gpu(record(SLIDER), contact_capacity=capacity)
