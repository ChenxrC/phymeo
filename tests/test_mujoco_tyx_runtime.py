"""MuJoCo-law normal rank-one pipeline on genuine TYX operators.

Authoring-only: collection/interpret/CUDA are owned by the parent executor.
The oracle runs original native MuJoCo with the same source/options/control.
"""

import json
import sys

import numpy as np
import pytest

from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime, MuJoCoTyxUnsupported
from phymeo.backends.mujoco_runtime import MuJoCoControl
from phymeo.training.resident_stepper import PhysicsStepFailure

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
SLIDER = """<mujoco model="tyx-slider"><option timestep="0.001" gravity="0 0 0"/>
<worldbody><body name="mass"><joint name="slide" type="slide" axis="0 0 1"
 armature="0.04" damping="0.2"/><geom name="sphere" type="sphere" size="0.1" mass="0.5"/>
</body></worldbody><actuator><motor name="drive" joint="slide" gear="2"
 ctrllimited="true" ctrlrange="-5 5"/></actuator></mujoco>"""


def source(*, free=False, mass=0.5, plane=True, condim=1, angle=False):
    q = '.9238795325112867 0 .3826834323650898 0' if angle else '1 0 0 0'
    center = '.0700035713374682 0 .0700035713374682' if angle else '0 0 .099'
    joint = '<freejoint/>' if free else '<joint name="slide" type="slide" axis="0 0 1"/>'
    floor = f'<geom name="floor" type="plane" size="2 2 .1" quat="{q}" condim="{condim}"/>' if plane else ''
    return f'''<mujoco><option timestep="0.001" gravity="0 0 -9.81"/>
    <worldbody>{floor}<body name="body" pos="{center}">{joint}
    <geom name="ball" type="sphere" size=".1" mass="{mass}" condim="{condim}"
     solref=".006 1" solimp=".85 .95 .002 .4 2"/></body></worldbody></mujoco>'''


def compile_source(xml):
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    return compile_mujoco_xml(xml, require_asset=False)


def deps(mode):
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("tyx")
    return mj


def compare_native(mj, model, native, state, *, atol=3e-5):
    np.testing.assert_allclose(state.q, native.qpos, rtol=0, atol=atol)
    np.testing.assert_allclose(state.v, native.qvel, rtol=0, atol=atol)
    assert state.time == pytest.approx(native.time, abs=2e-6)
    original_force = np.zeros(6)
    for index in range(native.ncon):
        value = np.empty(6)
        mj.mj_contactForce(model, native, index, value)
        original_force += value
    actual_force = sum((row.wrench_contact_frame for row in state.last_step.contacts), np.zeros(6))
    np.testing.assert_allclose(actual_force, original_force, rtol=0.01, atol=0.003)


@pytest.mark.parametrize("mode", MODES)
def test_sparse_slider_native_gear_damping_and_no_contact_condim3(mode):
    mj = deps(mode)
    compilation = compile_source(SLIDER)
    refs = {key: mj.MjData(compilation.mj_model) for key in (4, 91)}
    with MuJoCoTyxRuntime(compilation, mode=mode, world_keys=(4, 91)) as runtime:
        runtime.set_controls({4: MuJoCoControl([8]), 91: MuJoCoControl([-0.7])})
        for _ in range(24):
            for key, control in ((4, 8), (91, -0.7)):
                refs[key].ctrl[:] = control
                mj.mj_step(compilation.mj_model, refs[key])
            assert runtime.step(0.001) == {4: True, 91: True}
        for key, state in runtime.snapshot().items():
            compare_native(mj, compilation.mj_model, refs[key], state)
            assert state.tick == 24
            assert state.last_step.backend == f"mujoco-tyx-{mode}"
            assert runtime.capabilities["implementation"] == "tyx"
            assert runtime.capabilities["domain"] == "prismatic-or-isotropic-free/one-normal-pair-v1"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("free,mass", [(False, 0.05), (False, 4.0), (True, 0.5)])
def test_same_native_soft_law_loaded_hold_unload_and_reentry(mode, free, mass, tmp_path):
    mj = deps(mode)
    compilation = compile_source(source(free=free, mass=mass))
    model = compilation.mj_model
    native = mj.MjData(model)
    history = []
    with MuJoCoTyxRuntime(compilation, mode=mode, world_keys=(17,)) as runtime:
        for tick in range(340):
            upward = 2.2 * mass * 9.81 if 160 <= tick < 200 else 0.0
            applied = np.zeros((model.nbody, 6))
            applied[1, 2] = upward
            packet = MuJoCoControl([], xfrc_applied=applied)
            runtime.set_controls({17: packet})
            native.xfrc_applied[:] = applied
            mj.mj_step(model, native)
            runtime.step(0.001)
            state = runtime.snapshot()[17]
            compare_native(mj, model, native, state, atol=8e-5)
            history.append({"tick": state.tick, "time": state.time, "q": state.q.tolist(),
                            "v": state.v.tolist(), "external_z": upward,
                            "contacts": len(state.last_step.contacts),
                            "normal_force": sum(c.wrench_contact_frame[0] for c in state.last_step.contacts)})
        assert history[150]["normal_force"] == pytest.approx(mass * 9.81, rel=0.06)
        assert any(row["contacts"] == 0 for row in history[180:260])
        assert history[-1]["normal_force"] > 0
        assert state.last_step.contact_time == state.last_step.input_time
        assert state.last_step.contact_phase == "tyx-euler-input-constraint-evaluation"
        assert state.contact_phase == "endpoint-cpu-forward-recomputed"
    (tmp_path / "native-law-trajectory.json").write_text(json.dumps(history), encoding="utf-8")


@pytest.mark.parametrize("mode", MODES)
def test_slanted_plane_and_two_axis_prismatic_tree_use_full_coupled_mass(mode):
    mj = deps(mode)
    xml = source(angle=True).replace(
        '<joint name="slide" type="slide" axis="0 0 1"/>',
        '<joint name="x" type="slide" axis="1 0 0"/>'
    ).replace(
        '<geom name="ball"',
        '<geom name="hidden" type="sphere" size=".03" mass=".2" contype="0" conaffinity="0"/>'
        '<body name="child"><joint name="z" type="slide" axis="0 0 1"/><geom name="ball"'
    ).replace('</body></worldbody>', '</body></body></worldbody>')
    compilation = compile_source(xml)
    native = mj.MjData(compilation.mj_model)
    with MuJoCoTyxRuntime(compilation, mode=mode) as runtime:
        for _ in range(80):
            mj.mj_step(compilation.mj_model, native)
            runtime.step(0.001)
        compare_native(mj, compilation.mj_model, native, runtime.snapshot()[0], atol=1e-4)
        assert runtime.snapshot()[0].last_step.contacts


@pytest.mark.parametrize("mode", MODES)
def test_free_sphere_actual_world_body_torque_and_native_local_quaternion_rates(mode):
    mj = deps(mode)
    compilation = compile_source(source(free=True, plane=False))
    native = mj.MjData(compilation.mj_model)
    applied = np.zeros((compilation.mj_model.nbody, 6))
    applied[1, 3:] = [0.001, -0.0004, 0.0007]
    with MuJoCoTyxRuntime(compilation, mode=mode) as runtime:
        runtime.set_controls({0: MuJoCoControl([], xfrc_applied=applied)})
        for _ in range(40):
            native.xfrc_applied[:] = applied
            mj.mj_step(compilation.mj_model, native)
            runtime.step(0.001)
        compare_native(mj, compilation.mj_model, native, runtime.snapshot()[0], atol=5e-5)
        assert np.linalg.norm(runtime.snapshot()[0].v[3:]) > 0.01


@pytest.mark.parametrize("mode", MODES)
def test_owned_source_clone_controls_and_selected_reset(mode):
    deps(mode)
    compilation = compile_source(SLIDER)
    compilation.mj_model.opt.timestep = 1.0
    with MuJoCoTyxRuntime(compilation, mode=mode, world_keys=(5, 77)) as runtime:
        runtime.set_controls({5: MuJoCoControl([1]), 77: MuJoCoControl([-1])})
        runtime.step(0.001)
        before = runtime.snapshot()
        with runtime.clone() as clone:
            clone.step(0.001)
            runtime.reset((5,))
            assert runtime.snapshot()[5].tick == 0
            assert runtime.snapshot()[77].tick == 1
            np.testing.assert_array_equal(runtime.snapshot()[77].q, before[77].q)
            assert clone.snapshot()[5].tick == 2
            with pytest.raises(ValueError):
                before[5].q.setflags(write=True)


@pytest.mark.parametrize("mode", MODES)
def test_no_warp_or_cpu_step_or_per_step_cpu_forward(mode, monkeypatch):
    mj = deps(mode)
    monkeypatch.setitem(sys.modules, "warp", None)
    monkeypatch.setitem(sys.modules, "mujoco_warp", None)
    with MuJoCoTyxRuntime(compile_source(SLIDER), mode=mode) as runtime:
        forward = mj.mj_forward
        def forbidden(*args, **kwargs):
            raise AssertionError("TYX integration cannot call a CPU/Warp physics path")
        monkeypatch.setattr(mj, "mj_step", forbidden)
        monkeypatch.setattr(mj, "mj_forward", forbidden)
        runtime.step(0.001)
        monkeypatch.setattr(mj, "mj_forward", forward)
        assert runtime.snapshot()[0].tick == 1
        assert runtime.counters["completed_kernels"] >= 1


@pytest.mark.parametrize("mode", MODES)
def test_capacity_failure_preserves_zero_force_site_and_published_state(mode):
    deps(mode)
    xml = source().replace('pos="0 0 .099"', 'pos="0 0 .1"').replace(
        'solref=".006 1"', 'margin=".002" gap=".001" solref=".006 1"'
    )
    with MuJoCoTyxRuntime(compile_source(xml), mode=mode, contact_capacity=0) as runtime:
        before = runtime.snapshot()[0]
        with pytest.raises(PhysicsStepFailure) as error:
            runtime.step(0.001)
        assert error.value.accepted_worlds == {0: False}
        assert runtime.snapshot()[0].tick == 0
        np.testing.assert_array_equal(runtime.snapshot()[0].q, before.q)


@pytest.mark.parametrize("mode", MODES)
def test_late_download_failure_is_not_a_partial_publication(mode, monkeypatch):
    deps(mode)
    with MuJoCoTyxRuntime(compile_source(SLIDER), mode=mode, world_keys=(2, 19)) as runtime:
        original = runtime._device.download
        def failed():
            raise RuntimeError("injected readback failure after TYX candidate work")
        monkeypatch.setattr(runtime._device, "download", failed)
        with pytest.raises(PhysicsStepFailure) as error:
            runtime.step(0.001)
        assert error.value.accepted_worlds == {2: False, 19: False}
        assert all(s.tick == 0 for s in runtime.snapshot().values())
        monkeypatch.setattr(runtime._device, "download", original)
        assert runtime.step(0.001) == {2: True, 19: True}


@pytest.mark.parametrize("change,code", [
    (lambda x: x.replace('type="slide"', 'type="hinge"'), "joint-domain"),
    (lambda x: x.replace('<option ', '<option iterations="0" '), "solver-options"),
    (lambda x: x.replace('</mujoco>', '<equality><joint joint1="slide"/></equality></mujoco>'), "equality"),
    (lambda x: x.replace('</mujoco>', '<tendon><fixed name="t"><joint joint="slide" coef="1"/></fixed></tendon></mujoco>'), "tendon"),
])
def test_unsupported_source_semantics_have_capability_diagnostics(change, code):
    deps("interpret")
    with pytest.raises(MuJoCoTyxUnsupported) as error:
        MuJoCoTyxRuntime(compile_source(change(SLIDER)), mode="interpret")
    assert code in {issue.code for issue in error.value.issues}


def test_possible_friction_contact_is_rejected_but_not_no_contact_condim3():
    deps("interpret")
    with pytest.raises(MuJoCoTyxUnsupported, match="contact-dimension"):
        MuJoCoTyxRuntime(compile_source(source(condim=3)), mode="interpret")
    with MuJoCoTyxRuntime(compile_source(SLIDER), mode="interpret") as runtime:
        assert runtime.capabilities["possible_contact_pairs"] == 0
