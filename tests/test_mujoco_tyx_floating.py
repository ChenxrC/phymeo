"""Independent native free-root cases, authored before T2 product execution."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass

import numpy as np
import pytest

from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime, MuJoCoTyxUnsupported
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.resident_stepper import PhysicsStepFailure

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
H = .001
TOL = {
    "body_position": (3e-5, 4e-5), "body_rotation": (3e-5, 4e-5),
    "body_com": (3e-5, 4e-5), "jacobian_com": (2e-4, 5e-5),
    "mass_matrix": (3e-4, 5e-5), "bias": (4e-4, 4e-4),
    "passive": (4e-4, 4e-4), "applied": (4e-4, 4e-4),
    "smooth_acceleration": (2e-3, 4e-3), "integration_acceleration": (2e-3, 4e-3),
}
Q_ATOL, V_ATOL, TIME_ATOL = 1e-4, 1e-3, 2e-6
ROOT_MASS = 2.6
ROOT_INERTIA = np.diag([.13, .19, .23])


@dataclass(frozen=True)
class FloatingCase:
    name: str
    xml: str
    scalar_count: int
    steps: int
    gyro: bool = False
    marker: bool = False


def _geometry(name):
    return f'<geom name="{name}-shape" type="box" size=".03 .04 .05" contype="0" conaffinity="0"/>'


def _scalar_tree(count):
    def node(index):
        if index >= count:
            return ""
        kind = "slide" if index % 3 == 2 else "hinge"
        axes = ("1 0 0", "0 1 0", "0 0 1")
        ref = .02 * (1 + index % 3)
        pose = ".12 .08 -.04" if index % 2 else ".14 -.06 .05"
        quat = ".9659258262890683 0 0 .2588190451025207" if index % 2 else "1 0 0 0"
        return f'''<body name="link{index}" pos="{pose}" quat="{quat}">
        <joint name="j{index}" type="{kind}" axis="{axes[index % 3]}" pos=".01 -.015 .02"
          ref="{ref}" springref="-.12" stiffness=".6" armature=".025" damping=".08"/>
        <inertial mass="{.6 + .05 * (index % 4)}" pos=".035 -.02 .025"
          fullinertia=".025 .03 .035 .001 -.0006 .0008"/>
        {_geometry('link'+str(index))}{node(2*index+1)}{node(2*index+2)}</body>'''
    return node(0)


def _model(count, *, gyro=False, marker=False):
    root_inertia = 'diaginertia=".13 .19 .23"' if gyro else 'fullinertia=".13 .19 .23 .006 -.004 .005"'
    com = "0 0 0" if gyro else ".07 -.04 .03"
    passive = 'armature=".03" damping=".04"' if count else ''
    fixed = '''<body name="marker" pos="-.4 .2 .1"><inertial mass=".4" pos=".02 0 0"
      diaginertia=".01 .012 .014"/></body>''' if marker else ''
    tool = '''<body name="tool" pos="-.13 .07 .09" quat=".9238795325112867 .3826834323650898 0 0">
      <inertial mass=".45" pos=".03 -.02 .01" fullinertia=".015 .018 .02 .001 -.0004 .0006"/>
      <geom type="box" size=".03 .02 .04" contype="0" conaffinity="0"/></body>''' if count else ''
    motors = ''.join(
        f'<general name="drive{i}" joint="j{i}" gear="{-2 if i % 2 == 0 else .5}" '
        'gaintype="fixed" gainprm="1.3" biastype="affine" biasprm=".07 -.4 -.09" '
        'ctrllimited="true" ctrlrange="-.4 .4" forcelimited="true" forcerange="-.7 .7"/>'
        for i in range(count)
    )
    gravity = "0 0 0" if gyro else ".2 -.3 -9.81"
    return f'''<mujoco model="floating-{count}"><compiler angle="radian" fusestatic="false"/>
    <option timestep=".001" gravity="{gravity}" integrator="Euler"/>
    <worldbody>{fixed}<body name="root" pos=".1 -.08 .7">
    <joint name="free" type="free" {passive}/>
    <inertial mass="2.6" pos="{com}" {root_inertia}/>{_geometry('root')}
    {_scalar_tree(count)}{tool}</body></worldbody><actuator>{motors}</actuator></mujoco>'''


CASES = (
    FloatingCase("gyro0", _model(0, gyro=True), 0, 32, gyro=True),
    FloatingCase("offset0", _model(0), 0, 24),
    FloatingCase("root-child1", _model(1, marker=True), 1, 32, marker=True),
    FloatingCase("mixed7", _model(7, marker=True), 7, 24, marker=True),
    FloatingCase("tree16", _model(16, marker=True), 16, 12, marker=True),
    FloatingCase("tree29", _model(29, marker=True), 29, 12, marker=True),
)


def _deps():
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("tyx")
    assert mj.__version__ == "3.3.7"
    return mj


def _initial(model, slot=0):
    q = model.qpos0.copy()
    q[:3] += [.025 * slot, -.013 * slot, .01 * slot]
    quat = np.array([.9, .17 + .01 * slot, -.23, .11], dtype=np.float64)
    q[3:7] = quat / np.linalg.norm(quat)
    for index in range(model.nq - 7):
        q[7 + index] += (-1 if index % 2 else 1) * (.08 + .002 * index + .003 * slot)
    v = np.array([.18, -.12, .09, .6, -.45, .35] + [
        (-1 if i % 2 else 1) * (.12 + .004 * i) for i in range(model.nv - 6)
    ]) * (1 + .05 * slot)
    return MuJoCoInitialState(q, v)


def _packet(model, case, tick, slot=0):
    if case.gyro:
        return MuJoCoControl([], qfrc_applied=np.zeros(model.nv),
                             xfrc_applied=np.zeros((model.nbody, 6)))
    sign = -1 if (tick // 4 + slot) % 2 else 1
    ctrl = np.array([sign * (-1 if i % 2 else 1) * (.18 + .012 * i) for i in range(model.nu)])
    qforce = np.zeros(model.nv)
    qforce[:6] = sign * np.array([.3, -.2, .15, .08, -.06, .04])
    if model.nv > 6:
        qforce[6:] = sign * np.linspace(.025, .085, model.nv - 6)
    wrench = np.zeros((model.nbody, 6))
    wrench[-1] = np.array([.35, -.25, .4, .025, -.035, .015]) * (1 + .1 * slot)
    return MuJoCoControl(ctrl, qfrc_applied=qforce, xfrc_applied=wrench)


def _stored(packet):
    return MuJoCoControl(np.asarray(packet.ctrl, np.float32),
                         qfrc_applied=np.asarray(packet.qfrc_applied, np.float32),
                         xfrc_applied=np.asarray(packet.xfrc_applied, np.float32))


def _set_control(data, packet):
    data.ctrl[:], data.qfrc_applied[:], data.xfrc_applied[:] = (
        packet.ctrl, packet.qfrc_applied, packet.xfrc_applied
    )


def _native(mj, model, q, v, packet, time):
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = q, v, time
    _set_control(data, packet)
    mj.mj_forward(model, data)
    assert data.ncon == 0 and data.nefc == 0
    mass = np.empty((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    jac = np.zeros((model.nbody, 6, model.nv))
    applied = data.qfrc_actuator.copy() + data.qfrc_applied
    for body in range(model.nbody):
        mj.mj_jac(model, data, jac[body, :3], jac[body, 3:], data.xipos[body], body)
        applied += jac[body].T @ packet.xfrc_applied[body]
    rhs = applied + data.qfrc_passive - data.qfrc_bias
    smooth = np.linalg.solve(mass, rhs)
    np.testing.assert_allclose(smooth, data.qacc_smooth, rtol=2e-10, atol=2e-9)
    return {"body_position": data.xpos.copy(),
            "body_rotation": data.xmat.reshape(model.nbody, 3, 3).copy(),
            "body_com": data.xipos.copy(), "jacobian_com": jac,
            "mass_matrix": mass, "bias": data.qfrc_bias.copy(),
            "passive": data.qfrc_passive.copy(), "applied": applied,
            "smooth_acceleration": smooth,
            "integration_acceleration": np.linalg.solve(mass + H * np.diag(model.dof_damping), rhs),
            "native_qacc": data.qacc_smooth.copy(), "native_actuator": data.qfrc_actuator.copy()}


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _save(path, payload):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_plain(payload), allow_nan=False, separators=(",", ":")) + "\n")


def _assert_dynamics(row, expected):
    dyn = row.dynamics
    assert dyn.phase == "tyx-euler-input-dynamics"
    np.testing.assert_array_equal(dyn.input_q, row.input_q)
    np.testing.assert_array_equal(dyn.input_v, row.input_v)
    assert dyn.time == row.input_time
    for name, (rtol, atol) in TOL.items():
        actual = row.integration_acceleration if name == "integration_acceleration" else getattr(dyn, name)
        assert actual.shape == expected[name].shape
        np.testing.assert_allclose(actual, expected[name], rtol=rtol, atol=atol, err_msg=name)
        with pytest.raises(ValueError):
            actual.setflags(write=True)
    np.testing.assert_allclose(row.qacc, expected["native_qacc"], rtol=2e-3, atol=4e-3)
    np.testing.assert_allclose(row.qfrc_actuator, expected["native_actuator"], rtol=4e-4, atol=4e-4)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda x: x.name)
def test_native_floating_dynamics_and_continuous_euler(mode, case, tmp_path):
    mj = _deps()
    model = mj.MjModel.from_xml_string(case.xml)
    assert (model.nq, model.nv) == (7 + case.scalar_count, 6 + case.scalar_count)
    root = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "root")
    if case.marker:
        assert root > 1 and model.nbody == case.scalar_count + 4
    compiled = compile_mujoco_xml(case.xml, require_asset=False)
    initial = _initial(model)
    path = tmp_path / "floating-dynamics.jsonl"
    _save(path, {"schema": "mujoco-tyx-t2-v1", "case": case.name, "mode": mode,
                 "source_xml": case.xml, "tolerances": TOL, "initial": initial,
                 "steps": case.steps, "native_version": mj.__version__})
    with MuJoCoTyxRuntime(compiled, mode=mode, world_keys=(19,), initial_states={19: initial}) as runtime:
        assert runtime.capabilities["domain"] == "free-root-hinge-slide-no-contact-v1"
        start = runtime.snapshot()[19]
        reference = mj.MjData(model)
        reference.qpos[:], reference.qvel[:] = start.q, start.v
        first_mass = None
        selected = {0, case.steps // 2 - 1, case.steps - 1}
        for tick in range(case.steps):
            before = runtime.snapshot()[19]
            packet = _packet(model, case, tick)
            stored = _stored(packet)
            expected = _native(mj, model, before.q, before.v, stored, before.time)
            _set_control(reference, stored)
            mj.mj_step(model, reference)
            runtime.set_controls({19: packet})
            assert runtime.step(H) == {19: True}
            state = runtime.snapshot()[19]
            row = state.last_step
            detail = {"step": tick + 1, "oracle": expected, "actual": row,
                      "q": state.q, "v": state.v, "cpu_q": reference.qpos.copy(),
                      "cpu_v": reference.qvel.copy()}
            _save(path, detail if tick in selected else {
                "step": tick + 1, "input_q": before.q, "input_v": before.v,
                "q": state.q, "v": state.v, "cpu_q": reference.qpos.copy(),
                "cpu_v": reference.qvel.copy(), "request": packet, "stored": row.stored_control})
            try:
                assert row.input_q.shape == (model.nq,) and row.input_v.shape == (model.nv,)
                np.testing.assert_array_equal(row.input_q, before.q)
                np.testing.assert_array_equal(row.input_v, before.v)
                np.testing.assert_array_equal(row.stored_control.qfrc_applied, stored.qfrc_applied)
                assert row.input_time == before.time and row.output_time == state.time
                assert row.tick == state.tick == tick + 1 and row.contacts == ()
                assert row.backend == "mujoco-tyx-" + mode
                _assert_dynamics(row, expected)
                np.testing.assert_allclose(state.q, reference.qpos, atol=Q_ATOL, rtol=0)
                np.testing.assert_allclose(state.v, reference.qvel, atol=V_ATOL, rtol=0)
                assert state.time == pytest.approx(reference.time, abs=TIME_ATOL)
                assert abs(np.dot(state.q[3:7], state.q[3:7]) - 1) < 2e-6
                if tick == 0:
                    first_mass = expected["mass_matrix"]
                    assert np.all(row.input_v != 0)
                    if case.scalar_count:
                        assert np.max(np.abs(first_mass[:6, 6:])) > 1e-4
                        assert abs(row.stored_control.qfrc_applied[-1]) > .01
                        assert np.linalg.norm(row.dynamics.jacobian_com[-1]) > .1
                if tick == case.steps - 1 and case.scalar_count:
                    assert np.max(np.abs(expected["mass_matrix"] - first_mass)) > 1e-6
            except Exception:
                if tick not in selected:
                    _save(path, {"failure_step": tick + 1, "full": detail})
                raise


def _rotation(quaternion):
    # Independent explicit WXYZ formula, not product or native FK.
    w, x, y, z = np.asarray(quaternion) / np.linalg.norm(quaternion)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


@pytest.mark.parametrize("mode", MODES)
def test_free_gyro_analytic_mass_bias_and_bounded_world_momentum(mode, tmp_path):
    mj = _deps()
    case = CASES[0]
    model = mj.MjModel.from_xml_string(case.xml)
    initial = _initial(model)
    with MuJoCoTyxRuntime(compile_mujoco_xml(case.xml, require_asset=False), mode=mode,
                          initial_states={0: initial}) as runtime:
        start = runtime.snapshot()[0]
        linear = ROOT_MASS * start.v[:3]
        angular = _rotation(start.q[3:7]) @ ROOT_INERTIA @ start.v[3:6]
        energy = .5 * start.v[3:6] @ ROOT_INERTIA @ start.v[3:6]
        mass = np.zeros((6, 6))
        mass[:3, :3], mass[3:, 3:] = ROOT_MASS * np.eye(3), ROOT_INERTIA
        for tick in range(32):
            before = runtime.snapshot()[0]
            omega = before.v[3:6]
            bias = np.r_[np.zeros(3), np.cross(omega, ROOT_INERTIA @ omega)]
            acceleration = np.r_[np.zeros(3), -bias[3:] / np.diag(ROOT_INERTIA)]
            runtime.set_controls({0: _packet(model, case, tick)})
            runtime.step(H)
            state = runtime.snapshot()[0]
            dyn = state.last_step.dynamics
            np.testing.assert_allclose(dyn.mass_matrix, mass, rtol=3e-4, atol=5e-5)
            np.testing.assert_allclose(dyn.bias, bias, rtol=4e-4, atol=4e-4)
            np.testing.assert_allclose(dyn.smooth_acceleration, acceleration, rtol=2e-3, atol=4e-3)
            np.testing.assert_allclose(ROOT_MASS * state.v[:3], linear, rtol=0, atol=2e-5)
            actual_angular = _rotation(state.q[3:7]) @ ROOT_INERTIA @ state.v[3:6]
            actual_energy = .5 * state.v[3:6] @ ROOT_INERTIA @ state.v[3:6]
            assert np.linalg.norm(actual_angular - angular) <= 5e-4 * max(np.linalg.norm(angular), .1)
            assert abs(actual_energy - energy) <= 5e-4 * max(abs(energy), .1)
            _save(tmp_path / "gyro-momentum.jsonl", {"step": tick + 1, "q": state.q, "v": state.v,
                 "linear": linear, "angular": actual_angular, "initial_angular": angular,
                 "energy": actual_energy, "initial_energy": energy,
                 "actual": state.last_step if tick in (0, 15, 31) else None})


@pytest.mark.parametrize("mode", MODES)
def test_offset_root_analytic_jacobian_mass_and_world_wrench_origin(mode, tmp_path):
    mj = _deps()
    case = CASES[1]
    model = mj.MjModel.from_xml_string(case.xml)
    packet = _packet(model, case, 0)
    with MuJoCoTyxRuntime(compile_mujoco_xml(case.xml, require_asset=False), mode=mode,
                          initial_states={0: _initial(model)}) as runtime:
        before = runtime.snapshot()[0]
        r = np.array([.07, -.04, .03])
        cross = np.array([[0., -r[2], r[1]], [r[2], 0., -r[0]], [-r[1], r[0], 0.]])
        rotation = _rotation(before.q[3:7])
        jac = np.zeros((6, 6))
        jac[:3, :3], jac[:3, 3:], jac[3:, 3:] = np.eye(3), -rotation @ cross, rotation
        inertia = np.array([[.13, .006, -.004], [.006, .19, .005], [-.004, .005, .23]])
        world_inertia = rotation @ inertia @ rotation.T
        mass = ROOT_MASS * jac[:3].T @ jac[:3] + jac[3:].T @ world_inertia @ jac[3:]
        stored = _stored(packet)
        applied = stored.qfrc_applied + jac.T @ stored.xfrc_applied[1]
        runtime.set_controls({0: packet})
        runtime.step(H)
        actual = runtime.snapshot()[0].last_step
        np.testing.assert_allclose(actual.dynamics.body_com[1], before.q[:3] + rotation @ r, rtol=3e-5, atol=4e-5)
        np.testing.assert_allclose(actual.dynamics.jacobian_com[1], jac, rtol=2e-4, atol=5e-5)
        np.testing.assert_allclose(actual.dynamics.mass_matrix, mass, rtol=3e-4, atol=5e-5)
        np.testing.assert_allclose(actual.dynamics.applied, applied, rtol=4e-4, atol=4e-4)
        _save(tmp_path / "offset-root-analytic.jsonl", {"source_xml": case.xml, "jacobian": jac,
              "mass": mass, "applied": applied, "actual": actual})


@pytest.mark.parametrize("mode", MODES)
def test_root_and_child_loads_create_real_reciprocal_acceleration(mode, tmp_path):
    mj = _deps()
    case = CASES[2]
    model = mj.MjModel.from_xml_string(case.xml)
    initial = _initial(model)
    results = []
    for field in ("none", "root", "child"):
        qforce = np.zeros(model.nv)
        if field == "root":
            qforce[:6] = [.4, -.3, .2, .1, -.08, .12]
        if field == "child":
            qforce[-1] = .25
        packet = MuJoCoControl(np.zeros(model.nu), qfrc_applied=qforce,
                               xfrc_applied=np.zeros((model.nbody, 6)))
        with MuJoCoTyxRuntime(compile_mujoco_xml(case.xml, require_asset=False), mode=mode,
                              initial_states={0: initial}) as runtime:
            before = runtime.snapshot()[0]
            expected = _native(mj, model, before.q, before.v, _stored(packet), before.time)
            runtime.set_controls({0: packet})
            runtime.step(H)
            row = runtime.snapshot()[0].last_step
            _assert_dynamics(row, expected)
            results.append(row.qacc.copy())
            _save(tmp_path / "root-child-coupling.jsonl", {"load": field, "oracle": expected, "actual": row})
    assert np.max(np.abs(results[1][6:] - results[0][6:])) > 1e-4
    assert np.max(np.abs(results[2][:6] - results[0][:6])) > 1e-4


def _same(left, right):
    assert set(left) == set(right)
    for key in left:
        for name in ("q", "v", "ctrl", "act"):
            np.testing.assert_array_equal(getattr(left[key], name), getattr(right[key], name))
        assert (left[key].tick, left[key].time) == (right[key].tick, right[key].time)
        for name in ("ctrl", "qfrc_applied", "xfrc_applied"):
            np.testing.assert_array_equal(getattr(left[key].held_control, name),
                                          getattr(right[key].held_control, name))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("keys", [(5,), (3, 19, 91, 203)], ids=["w1", "w4-sparse"])
def test_floating_sparse_reset_clone_preserves_next_real_step(mode, keys, tmp_path):
    mj = _deps()
    case = CASES[3]
    model = mj.MjModel.from_xml_string(case.xml)
    compiled = compile_mujoco_xml(case.xml, require_asset=False)
    initial = {key: _initial(model, slot) for slot, key in enumerate(keys)}
    with MuJoCoTyxRuntime(compiled, mode=mode, world_keys=keys, initial_states=initial) as runtime:
        for tick in range(3):
            runtime.set_controls({key: _packet(model, case, tick, slot) for slot, key in enumerate(keys)})
            runtime.step(H)
        with runtime.clone() as cloned:
            before = runtime.snapshot()
            _same(before, cloned.snapshot())
            selected = keys[0]
            runtime.reset((selected,))
            reset = runtime.snapshot()
            assert reset[selected].tick == 0 and reset[selected].last_step is None
            for key in keys[1:]:
                _same({key: reset[key]}, {key: before[key]})
            with MuJoCoTyxRuntime(compiled, mode=mode, world_keys=(selected,),
                                  initial_states={selected: initial[selected]}) as fresh:
                packets = {key: _packet(model, case, 3, slot) for slot, key in enumerate(keys)}
                runtime.set_controls(packets)
                cloned.set_controls(packets)
                fresh.set_controls({selected: packets[selected]})
                runtime.step(H)
                cloned.step(H)
                fresh.step(H)
                actual, expected = runtime.snapshot(), cloned.snapshot()
                _same({selected: actual[selected]}, fresh.snapshot())
                for key in keys[1:]:
                    _same({key: actual[key]}, {key: expected[key]})
                _save(tmp_path / "reset-next-step.jsonl", {"source_xml": case.xml,
                    "before": before, "reset": reset, "actual": actual,
                    "clone": expected, "fresh": fresh.snapshot()})


@pytest.mark.parametrize("mode", MODES)
def test_floating_late_download_failure_rolls_back_full_batch_and_retries(mode, monkeypatch, tmp_path):
    mj = _deps()
    case = CASES[2]
    model = mj.MjModel.from_xml_string(case.xml)
    keys = (7, 29, 113, 211)
    with MuJoCoTyxRuntime(compile_mujoco_xml(case.xml, require_asset=False), mode=mode,
                          world_keys=keys, initial_states={key: _initial(model, slot)
                          for slot, key in enumerate(keys)}) as runtime:
        runtime.set_controls({key: _packet(model, case, 0, slot) for slot, key in enumerate(keys)})
        runtime.step(H)
        runtime.set_controls({key: _packet(model, case, 1, slot) for slot, key in enumerate(keys)})
        with runtime.clone() as reference:
            before = runtime.snapshot()
            count = runtime.counters["completed_kernels"]
            original = runtime._device.download
            def fail_after_real_readback():
                original()
                raise RuntimeError("injected T2 after actual device readback")
            monkeypatch.setattr(runtime._device, "download", fail_after_real_readback)
            with pytest.raises(PhysicsStepFailure) as caught:
                runtime.step(H)
            assert caught.value.accepted_worlds == dict.fromkeys(keys, False)
            assert caught.value.acceptance_known and not caught.value.physics_accepted
            assert runtime.counters["completed_kernels"] > count
            _same(before, runtime.snapshot())
            _save(tmp_path / "atomic-failure.jsonl", {"before": before, "after": runtime.snapshot(),
                "mask": caught.value.accepted_worlds})
            monkeypatch.setattr(runtime._device, "download", original)
            assert runtime.step(H) == dict.fromkeys(keys, True)
            reference.step(H)
            _same(runtime.snapshot(), reference.snapshot())
            _save(tmp_path / "atomic-failure.jsonl", {"retry": runtime.snapshot(), "clone": reference.snapshot()})


@pytest.mark.parametrize("mode", MODES)
def test_floating_step_has_no_cpu_warp_or_host_dynamics_fallback(mode, monkeypatch, tmp_path):
    import importlib
    import sys
    mj = _deps()
    case = CASES[3]
    model = mj.MjModel.from_xml_string(case.xml)
    modules = [(importlib.import_module(name), names) for name, names in (
        ("phymeo.dynamics.robot_dynamics", ("forward_dynamics", "mass_matrix", "bias_forces")),
        ("phymeo.dynamics.robot_kinematics", ("forward_kinematics", "point_jacobians")),
        ("phymeo.solvers.floating_tree", ("step_floating_robot", "floating_dynamics_terms")),
    )]
    with MuJoCoTyxRuntime(compile_mujoco_xml(case.xml, require_asset=False), mode=mode,
                          initial_states={0: _initial(model)}) as runtime:
        before = runtime.snapshot()[0]
        packet = _packet(model, case, 0)
        expected = _native(mj, model, before.q, before.v, _stored(packet), before.time)
        runtime.set_controls({0: packet})
        count = runtime.counters["completed_kernels"]
        def forbidden(*args, **kwargs):
            raise AssertionError("T2 must execute real TYX mechanics")
        with monkeypatch.context() as patch:
            for name in ("mj_step", "mj_forward", "mj_fullM", "mj_rne", "mj_jac"):
                patch.setattr(mj, name, forbidden)
            for name in ("solve", "inv", "cholesky"):
                patch.setattr(np.linalg, name, forbidden)
            for module, names in modules:
                for name in names:
                    patch.setattr(module, name, forbidden)
            patch.setitem(sys.modules, "mujoco_warp", None)
            patch.setitem(sys.modules, "warp", None)
            assert runtime.step(H) == {0: True}
        state = runtime.snapshot()[0]
        assert runtime.counters["completed_kernels"] > count
        assert runtime.capabilities["implementation"] == "tyx"
        assert runtime.capabilities["gpu"] is (mode == "cuda")
        _assert_dynamics(state.last_step, expected)
        _save(tmp_path / "no-fallback.jsonl", {"oracle": expected, "actual": state.last_step,
                                               "counters": runtime.counters})


@pytest.mark.parametrize("mode", MODES)
def test_floating_control_overflow_is_atomic(mode):
    mj = _deps()
    case = CASES[2]
    model = mj.MjModel.from_xml_string(case.xml)
    with MuJoCoTyxRuntime(compile_mujoco_xml(case.xml, require_asset=False), mode=mode,
                          world_keys=(5, 113), initial_states={5: _initial(model), 113: _initial(model, 1)}) as runtime:
        packets = {key: _packet(model, case, 0, slot) for slot, key in enumerate((5, 113))}
        runtime.set_controls(packets)
        before = runtime.snapshot()
        bad = np.zeros(model.nv)
        bad[-1] = np.finfo(np.float64).max
        with pytest.raises(ValueError):
            runtime.set_controls({5: packets[5], 113: MuJoCoControl([.1], qfrc_applied=bad)})
        _same(before, runtime.snapshot())
        runtime.step(H)
        assert all(s.tick == 1 for s in runtime.snapshot().values())


@pytest.mark.parametrize("mode", MODES)
def test_floating_scalar_limit_crossing_rejects_every_world(mode, tmp_path):
    mj = _deps()
    case = CASES[2]
    xml = case.xml.replace('name="j0" type="hinge"',
                           'name="j0" type="hinge" limited="true" range="-.5 .105" margin=".002"')
    model = mj.MjModel.from_xml_string(xml)
    unsafe, safe = _initial(model), _initial(model, 1)
    q, v = unsafe.q.copy(), unsafe.v.copy()
    q[-1], v[-1] = .102, 8.
    q_safe = safe.q.copy()
    q_safe[-1] = .02
    initial = {5: MuJoCoInitialState(q, v), 91: MuJoCoInitialState(q_safe, safe.v)}
    with MuJoCoTyxRuntime(compile_mujoco_xml(xml, require_asset=False), mode=mode,
                          world_keys=(5, 91), initial_states=initial) as runtime:
        before = runtime.snapshot()
        with pytest.raises(PhysicsStepFailure) as caught:
            runtime.step(H)
        assert caught.value.accepted_worlds == {5: False, 91: False}
        _same(before, runtime.snapshot())
        _save(tmp_path / "limit-rejection.jsonl", {"source_xml": xml, "before": before,
                                                  "after": runtime.snapshot(), "mask": caught.value.accepted_worlds})


BALL_XML = _model(0).replace('</body></worldbody>', '''<body name="ball-body" pos=".1 0 0">
<joint name="ball" type="ball"/><inertial mass=".5" pos="0 0 0" diaginertia=".02 .03 .04"/>
</body></body></worldbody>''')


@pytest.mark.parametrize("xml,code", [
    (_model(30, marker=True), "joint-domain"),
    (BALL_XML, "joint-domain"),
    (_model(0).replace('name="free" type="free"', 'name="free" type="free" stiffness="1"'), "free-spring"),
    (_model(7, marker=True).replace('contype="0" conaffinity="0"', 'contype="1" conaffinity="1"'), "contact-domain"),
])
def test_native_valid_outside_floating_domain_has_explicit_diagnostic(xml, code):
    _deps()
    compiled = compile_mujoco_xml(xml, require_asset=False)
    with pytest.raises(MuJoCoTyxUnsupported) as caught:
        MuJoCoTyxRuntime(compiled, mode="interpret")
    assert code in {issue.code for issue in caught.value.issues}
