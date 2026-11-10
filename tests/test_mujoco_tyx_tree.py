"""Independent fixed-root MuJoCo-to-TYX rotational-tree cases (UNEXECUTED).

The constants/cases are declared before product execution. CPU MuJoCo 3.3.7
is the numerical oracle; analytic one-body checks do not reuse product arrays.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime, MuJoCoTyxUnsupported
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.training.resident_stepper import PhysicsStepFailure

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
H = 0.001
# SI entries: hinge coordinates/rates are radians, prismatic are metres.
# M/J columns have the native corresponding mixed SI units; relative terms
# compare like entries only. These are engineering FP32 gates, not FP64 equality.
TOL = {
    "body_position": (2e-5, 3e-5),
    "body_rotation": (2e-5, 3e-5),
    "body_com": (2e-5, 3e-5),
    "jacobian_com": (2e-4, 5e-5),
    "mass_matrix": (2e-4, 3e-5),
    "bias": (3e-4, 2e-4),
    "passive": (3e-4, 2e-4),
    "applied": (3e-4, 2e-4),
    "smooth_acceleration": (5e-4, 8e-4),
    "integration_acceleration": (5e-4, 8e-4),
}
SHORT_Q_ATOL = 5e-5
SHORT_V_ATOL = 3e-4
TIME_ATOL = 2e-6


@dataclass(frozen=True)
class TreeCase:
    name: str
    xml: str
    offsets: tuple
    velocity: tuple
    wrench_body: str
    steps: int = 24


def _joint(name, *, kind="hinge", axis="0 1 0", anchor="0 0 0", ref="0", extra=""):
    return (
        f'<joint name="{name}" type="{kind}" axis="{axis}" pos="{anchor}" '
        f'ref="{ref}" armature=".04" damping=".12" {extra}/>'
    )


def _body(name, joint="", child="", *, pos=".1 .02 .04", quat="1 0 0 0",
          mass="1.1", com=".04 -.03 .02", inertia=".02 .025 .03 .001 -.0005 .0008"):
    return f'''<body name="{name}" pos="{pos}" quat="{quat}">{joint}
      <inertial mass="{mass}" pos="{com}" fullinertia="{inertia}"/>
      <geom name="{name}-visual" type="box" size=".04 .03 .05" contype="0" conaffinity="0"/>
      {child}</body>'''


def _model(name, body, joints, *, gravity=".2 -.3 -9.81", gears=None, extra=""):
    gears = tuple(1 if index % 2 else 2 for index in range(len(joints))) if gears is None else gears
    motors = ''.join(
        f'<motor name="motor-{index}" joint="{joint}" gear="{gear}" '
        'ctrllimited="true" ctrlrange="-.4 .4" forcelimited="true" forcerange="-.35 .35"/>'
        for index, (joint, gear) in enumerate(zip(joints, gears, strict=True))
    )
    return f'''<mujoco model="{name}"><compiler angle="radian" fusestatic="false"/>
      <option timestep=".001" gravity="{gravity}" integrator="Euler"/>
      <worldbody>{body}</worldbody><actuator>{motors}</actuator>{extra}</mujoco>'''


def _chain(count):
    child = ''
    axes = ("0 1 0", "1 0 0", "0 0 1")
    for index in reversed(range(count)):
        joint = _joint(f"j{index}", axis=axes[index % 3], anchor=".01 -.02 .015",
                       ref=str(.025 * (index + 1)))
        child = _body(f"link{index}", joint, child,
                      quat=".9659258262890683 0 0 .2588190451025207")
    return _model(f"chain-{count}", child, tuple(f"j{i}" for i in range(count)))


PENDULUM_XML = '''<mujoco model="eccentric-pendulum"><compiler angle="radian"/>
<option timestep=".001" gravity="0 0 -9.81" integrator="Euler"/>
<worldbody><body name="pendulum" pos=".1 -.08 .25">
<joint name="j0" type="hinge" axis="0 1 0" ref=".125" armature=".11" damping=".17"/>
<inertial mass="2" pos=".4 0 0" diaginertia=".03 .07 .08"/>
<geom name="shape" type="sphere" size=".05" contype="0" conaffinity="0"/>
</body></worldbody><actuator><motor name="motor-0" joint="j0" gear="2"
 ctrllimited="true" ctrlrange="-2 2"/></actuator></mujoco>'''

DOUBLE_XML = _model(
    "double-pendulum",
    _body("upper", _joint("j0", ref=".05"),
          _body("lower", _joint("j1", axis="0 1 0", ref="-.07"),
                pos=".45 0 .03", com=".2 .01 -.03", mass=".8"),
          pos="0 0 .4", com=".22 -.01 .02", mass="1.4"),
    ("j0", "j1"),
)
MIXED_XML = _model(
    "mixed-same-body-rp",
    _body("carriage", _joint("j0", axis="0 0 1", anchor=".02 -.01 .03", ref=".08")
          + _joint("j1", kind="slide", axis="1 0 0", anchor="-.01 .02 0", ref="-.03"),
          _body("tip", _joint("j2", axis="1 0 0", anchor=".01 .01 -.02", ref=".04"), mass=".6")),
    ("j0", "j1", "j2"), gears=(2, -1, .5),
)
FORK_XML = _model(
    "shared-ancestor-fork",
    _body("shoulder", _joint("j0", axis="0 0 1", ref=".03"),
          _body("left", _joint("j1", axis="1 0 0", ref="-.04"), pos=".3 .18 .02", mass=".8")
          + _body("right", _joint("j2", axis="0 1 0", ref=".06"), pos="-.2 -.14 .1", mass="1.5"),
          pos="0 0 .3", mass="2"),
    ("j0", "j1", "j2"),
)
WELDED_XML = _model(
    "welded-tool-load",
    _body("shoulder", _joint("j0", ref=".04"),
          _body("wrist", _joint("j1", axis="0 0 1", anchor=".02 -.01 .01", ref="-.06"),
                _body("tool", pos=".14 -.07 .11", quat=".9238795325112867 .3826834323650898 0 0",
                      mass=".7", com=".04 .02 -.03"), mass=".9"), mass="1.6")
    + _body("fixed-marker", pos="-.2 .1 .4", mass=".4"),
    ("j0", "j1"),
)

CASES = (
    TreeCase("pendulum", PENDULUM_XML, (.295,), (-.6,), "pendulum"),
    TreeCase("double", DOUBLE_XML, (.21, -.28), (.7, -.5), "lower", 48),
    TreeCase("mixed", MIXED_XML, (.18, .035, -.22), (-.4, .12, .55), "tip", 48),
    TreeCase("fork", FORK_XML, (.2, -.17, .25), (.45, -.35, .5), "right"),
    TreeCase("seven", _chain(7), (.11, -.13, .15, -.17, .19, -.21, .23),
             (.2, -.25, .3, -.35, .4, -.45, .5), "link6"),
    TreeCase("welded", WELDED_XML, (.24, -.19), (.65, -.45), "tool", 48),
    TreeCase("eight", _chain(8), (.11, -.13, .15, -.17, .19, -.21, .23, -.25),
             (.2, -.25, .3, -.35, .4, -.45, .5, -.55), "link7"),
)


def _deps():
    mj = pytest.importorskip("mujoco", reason="independent CPU oracle requires MuJoCo")
    pytest.importorskip("tyx", reason="actual TYX runtime is an optional dependency")
    assert mj.__version__ == "3.3.7", "this oracle is fixed to the declared native version"
    return mj


def _compile(case):
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    return compile_mujoco_xml(case.xml, require_asset=False)


def _initial(model, case, slot=0):
    # Each world starts independently from its declared native qpos0. The runtime
    # stores FP32; the independent continuous CPU reference starts at those same
    # representable values and subsequently evolves in its original FP64 model.
    q = np.asarray(model.qpos0) + np.asarray(case.offsets) + slot * .013
    v = np.asarray(case.velocity) * (1 + slot * .09)
    return MuJoCoInitialState(q.astype(np.float32), v.astype(np.float32))


def _stored(packet):
    return MuJoCoControl(
        np.asarray(packet.ctrl, np.float32),
        qfrc_applied=np.asarray(packet.qfrc_applied, np.float32),
        xfrc_applied=np.asarray(packet.xfrc_applied, np.float32),
    )


def _packet(mj, model, case, tick, slot=0):
    sign = -1 if (tick // 8 + slot) % 2 else 1
    ctrl = np.array([sign * (-1 if i % 2 else 1) * (.25 + .055 * i + .01 * slot)
                     for i in range(model.nu)])
    qforce = np.array([sign * .03 * (i + 1) for i in range(model.nv)])
    world_force = np.zeros((model.nbody, 6))
    body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, case.wrench_body)
    assert body > 0
    world_force[body] = np.array([.3, -.2, .4, .01, -.03, .02]) * (1 + slot * .25)
    return MuJoCoControl(ctrl, qfrc_applied=qforce, xfrc_applied=world_force)


def _set_native(data, packet):
    data.ctrl[:] = packet.ctrl
    data.qfrc_applied[:] = packet.qfrc_applied
    data.xfrc_applied[:] = packet.xfrc_applied


def _native_dynamics(mj, model, q, v, packet, time):
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = q, v, time
    _set_native(data, packet)
    mj.mj_forward(model, data)
    assert data.ncon == 0 and data.nefc == 0, "no-contact/interior source premise"
    mass = np.empty((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    jacobian = np.zeros((model.nbody, 6, model.nv))
    applied = data.qfrc_actuator.copy() + data.qfrc_applied
    for body in range(model.nbody):
        mj.mj_jac(model, data, jacobian[body, :3], jacobian[body, 3:], data.xipos[body], body)
        applied += jacobian[body].T @ packet.xfrc_applied[body]
    rhs = applied + data.qfrc_passive - data.qfrc_bias
    smooth = np.linalg.solve(mass, rhs)
    np.testing.assert_allclose(smooth, data.qacc_smooth, atol=2e-10, rtol=2e-11)
    integration = np.linalg.solve(mass + H * np.diag(model.dof_damping), rhs)
    return {
        "phase": "independent-native-input-dynamics",
        "time": float(time),
        "input_q": np.array(q),
        "input_v": np.array(v),
        "body_position": data.xpos.copy(),
        "body_rotation": data.xmat.reshape(model.nbody, 3, 3).copy(),
        "body_com": data.xipos.copy(),
        "jacobian_com": jacobian,
        "mass_matrix": mass,
        "bias": data.qfrc_bias.copy(),
        "passive": data.qfrc_passive.copy(),
        "applied": applied,
        "smooth_acceleration": smooth,
        "integration_acceleration": integration,
    }


def _plain(value):
    from collections.abc import Mapping
    from dataclasses import fields, is_dataclass

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _append(path, payload):
    import json

    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_plain(payload), allow_nan=False, separators=(",", ":")) + "\n")
        stream.flush()


def _assert_dynamics(receipt, expected):
    actual = receipt.dynamics
    assert actual.phase == "tyx-euler-input-dynamics"
    assert actual.time == receipt.input_time
    np.testing.assert_array_equal(actual.input_q, receipt.input_q)
    np.testing.assert_array_equal(actual.input_v, receipt.input_v)
    for name, (rtol, atol) in TOL.items():
        value = receipt.integration_acceleration if name == "integration_acceleration" else getattr(actual, name)
        assert value.shape == expected[name].shape
        np.testing.assert_allclose(value, expected[name], rtol=rtol, atol=atol, err_msg=name)
        with pytest.raises(ValueError):
            value.setflags(write=True)
    np.testing.assert_allclose(receipt.qacc, expected["smooth_acceleration"],
                               rtol=5e-4, atol=8e-4)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_rotational_tree_native_input_dynamics_and_continuous_euler(mode, case, tmp_path):
    mj = _deps()
    compiled = _compile(case)
    # The oracle compiles the original XML independently; no product lowered M/J.
    model = mj.MjModel.from_xml_string(case.xml)
    initial = _initial(model, case)
    reference = mj.MjData(model)
    reference.qpos[:], reference.qvel[:] = initial.q, initial.v
    path = tmp_path / "same-run-dynamics.jsonl"
    _append(path, {"schema": "mujoco-tyx-t1-v1", "case": case.name, "mode": mode,
                   "source_xml": case.xml, "tolerances": TOL, "initial": initial,
                   "steps": case.steps, "native_version": mj.__version__})
    selected = {0, case.steps // 2 - 1, case.steps - 1}
    world = 19
    with MuJoCoTyxRuntime(compiled, mode=mode, world_keys=(world,),
                          initial_states={world: initial}) as runtime:
        first_mass = None
        for tick in range(case.steps):
            before = runtime.snapshot()[world]
            packet = _packet(mj, model, case, tick)
            stored = _stored(packet)
            expected = _native_dynamics(mj, model, before.q, before.v, stored, before.time)
            _set_native(reference, stored)
            mj.mj_step(model, reference)
            runtime.set_controls({world: packet})
            assert runtime.step(H) == {world: True}
            state = runtime.snapshot()[world]
            receipt = state.last_step
            detail = {"step": tick + 1, "oracle": expected, "actual": receipt,
                      "continuous_cpu_q": reference.qpos.copy(), "continuous_cpu_v": reference.qvel.copy(),
                      "published_q": state.q, "published_v": state.v}
            _append(path, detail if tick in selected else {
                "step": tick + 1, "input_q": before.q, "input_v": before.v,
                "q": state.q, "v": state.v, "request": packet,
                "stored_control": receipt.stored_control,
                "cpu_q": reference.qpos.copy(), "cpu_v": reference.qvel.copy()})
            try:
                np.testing.assert_array_equal(receipt.input_q, before.q)
                np.testing.assert_array_equal(receipt.input_v, before.v)
                np.testing.assert_array_equal(receipt.stored_control.ctrl, stored.ctrl)
                assert receipt.input_time == before.time
                assert receipt.output_time == state.time and receipt.h == H
                assert receipt.tick == state.tick == tick + 1
                assert receipt.contacts == ()
                _assert_dynamics(receipt, expected)
                np.testing.assert_allclose(state.q, reference.qpos, rtol=0, atol=SHORT_Q_ATOL)
                np.testing.assert_allclose(state.v, reference.qvel, rtol=0, atol=SHORT_V_ATOL)
                assert state.time == pytest.approx(reference.time, abs=TIME_ATOL)
                assert receipt.backend == f"mujoco-tyx-{mode}"
                if tick == 0:
                    assert np.all(receipt.input_v != 0)
                    first_mass = expected["mass_matrix"]
                    if case.name == "double":
                        gravity = _native_dynamics(mj, model, before.q, np.zeros(model.nv), stored, before.time)
                        assert np.max(np.abs(expected["bias"] - gravity["bias"])) > 1e-4
                if tick == case.steps - 1 and case.name in {"double", "mixed", "seven", "eight"}:
                    assert np.max(np.abs(expected["mass_matrix"] - first_mass)) > 1e-6
            except Exception:
                if tick not in selected:
                    _append(path, {"failure_step": tick + 1, "full": detail})
                raise


@pytest.mark.parametrize("mode", MODES)
def test_eccentric_pendulum_has_independent_analytic_mass_gravity_wrench_and_euler(mode, tmp_path):
    mj = _deps()
    case = CASES[0]
    initial = _initial(mj.MjModel.from_xml_string(case.xml), case)
    wrench = np.array([[0., 0., 0., 0., 0., 0.], [.25, -.125, .5, .03125, .0625, -.015625]])
    packet = MuJoCoControl([.375], qfrc_applied=[.125], xfrc_applied=wrench)
    with MuJoCoTyxRuntime(_compile(case), mode=mode, initial_states={0: initial}) as runtime:
        before = runtime.snapshot()[0]
        runtime.set_controls({0: packet})
        runtime.step(H)
        state = runtime.snapshot()[0]
        row = state.last_step
        q, v = float(before.q[0]), float(before.v[0])
        delta = q - .125
        mass = .07 + .11 + 2 * .4**2
        bias = -2 * 9.81 * .4 * np.cos(delta)
        passive = -.17 * v
        applied = 2 * .375 + .125 - .4 * (.25 * np.sin(delta) + .5 * np.cos(delta)) + .0625
        acceleration = (applied + passive - bias) / mass
        integration = (applied + passive - bias) / (mass + H * .17)
        expected = {"mass": mass, "bias": bias, "passive": passive, "applied": applied,
                    "acceleration": acceleration, "integration": integration,
                    "next_v": v + H * integration, "next_q": q + H * (v + H * integration)}
        _append(tmp_path / "analytic-pendulum.jsonl", {"oracle": expected, "actual": row,
                                                       "published_q": state.q, "published_v": state.v})
        dyn = row.dynamics
        assert dyn.mass_matrix[0, 0] == pytest.approx(mass, rel=2e-4, abs=3e-5)
        for name in ("bias", "passive", "applied"):
            assert getattr(dyn, name)[0] == pytest.approx(expected[name], rel=3e-4, abs=2e-4)
        assert dyn.smooth_acceleration[0] == pytest.approx(acceleration, rel=5e-4, abs=8e-4)
        assert row.integration_acceleration[0] == pytest.approx(integration, rel=5e-4, abs=8e-4)
        assert state.q[0] == pytest.approx(expected["next_q"], abs=SHORT_Q_ATOL)
        assert state.v[0] == pytest.approx(expected["next_v"], abs=SHORT_V_ATOL)


def _assert_same_published(left, right):
    for key in left:
        for name in ("q", "v", "ctrl", "act"):
            np.testing.assert_array_equal(getattr(left[key], name), getattr(right[key], name))
        assert left[key].time == right[key].time
        assert left[key].tick == right[key].tick
        np.testing.assert_array_equal(left[key].held_control.ctrl, right[key].held_control.ctrl)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("keys", [(5,), (3, 19, 91, 203)], ids=["w1", "w4-sparse"])
def test_sparse_controls_clone_and_selected_reset_preserve_next_real_step(mode, keys, tmp_path):
    mj = _deps()
    case = CASES[2]
    compiled = _compile(case)
    model = mj.MjModel.from_xml_string(case.xml)
    initial = {key: _initial(model, case, slot) for slot, key in enumerate(keys)}
    with MuJoCoTyxRuntime(compiled, mode=mode, world_keys=keys, initial_states=initial) as runtime:
        for tick in range(4):
            packets = {key: _packet(mj, model, case, tick, slot) for slot, key in enumerate(keys)}
            runtime.set_controls(packets)
            assert runtime.step(H) == dict.fromkeys(keys, True)
        with runtime.clone() as cloned:
            before = runtime.snapshot()
            _assert_same_published(before, cloned.snapshot())
            selected = keys[0]
            with MuJoCoTyxRuntime(compiled, mode=mode, world_keys=(selected,),
                                  initial_states={selected: initial[selected]}) as fresh:
                runtime.reset((selected,))
                after_reset = runtime.snapshot()
                assert after_reset[selected].tick == 0
                assert after_reset[selected].last_step is None
                for key in keys[1:]:
                    _assert_same_published({key: before[key]}, {key: after_reset[key]})
                packets = {key: _packet(mj, model, case, 4, slot) for slot, key in enumerate(keys)}
                runtime.set_controls(packets)
                cloned.set_controls(packets)
                fresh.set_controls({selected: packets[selected]})
                runtime.step(H)
                cloned.step(H)
                fresh.step(H)
                actual, untouched, restarted = runtime.snapshot(), cloned.snapshot(), fresh.snapshot()
                _assert_same_published({selected: actual[selected]}, restarted)
                for key in keys[1:]:
                    _assert_same_published({key: actual[key]}, {key: untouched[key]})
                _append(tmp_path / "reset-next-step.jsonl", {
                    "source_xml": case.xml, "keys": keys, "initial": initial,
                    "before": before, "reset": after_reset, "actual_next": actual,
                    "clone_next": untouched, "fresh_next": restarted})


@pytest.mark.parametrize("mode", MODES)
def test_late_real_download_failure_is_atomic_and_same_request_retries(mode, monkeypatch, tmp_path):
    mj = _deps()
    case = CASES[5]
    model = mj.MjModel.from_xml_string(case.xml)
    keys = (7, 29, 113, 211)
    with MuJoCoTyxRuntime(_compile(case), mode=mode, world_keys=keys,
                          initial_states={key: _initial(model, case, slot)
                                          for slot, key in enumerate(keys)}) as runtime:
        runtime.set_controls({key: _packet(mj, model, case, 0, slot)
                              for slot, key in enumerate(keys)})
        runtime.step(H)
        packets = {key: _packet(mj, model, case, 1, slot) for slot, key in enumerate(keys)}
        runtime.set_controls(packets)
        with runtime.clone() as reference:
            before = runtime.snapshot()
            kernels_before = runtime.counters["completed_kernels"]
            original = runtime._device.download

            def fail_after_actual_readback():
                original()
                raise RuntimeError("injected T1 late readback failure after actual device work")

            monkeypatch.setattr(runtime._device, "download", fail_after_actual_readback)
            with pytest.raises(PhysicsStepFailure) as caught:
                runtime.step(H)
            assert caught.value.accepted_worlds == dict.fromkeys(keys, False)
            assert caught.value.acceptance_known is True
            assert caught.value.physics_accepted is False
            assert runtime.counters["completed_kernels"] > kernels_before
            after = runtime.snapshot()
            _assert_same_published(before, after)
            _append(tmp_path / "atomic-failure.jsonl", {
                "source_xml": case.xml, "request": packets, "before": before,
                "accepted_worlds": caught.value.accepted_worlds, "after": after})
            monkeypatch.setattr(runtime._device, "download", original)
            assert runtime.step(H) == dict.fromkeys(keys, True)
            assert reference.step(H) == dict.fromkeys(keys, True)
            _assert_same_published(runtime.snapshot(), reference.snapshot())
            _append(tmp_path / "atomic-failure.jsonl", {"retry": runtime.snapshot(),
                                                        "independent_clone": reference.snapshot()})


@pytest.mark.parametrize("mode", MODES)
def test_rotational_step_cannot_call_cpu_warp_or_host_dynamics(mode, monkeypatch, tmp_path):
    import importlib
    import sys

    mj = _deps()
    case = CASES[1]
    model = mj.MjModel.from_xml_string(case.xml)
    initial = _initial(model, case)
    packet = _packet(mj, model, case, 0)
    reference = mj.MjData(model)
    reference.qpos[:], reference.qvel[:] = initial.q, initial.v
    _set_native(reference, _stored(packet))
    mj.mj_step(model, reference)
    host_names = (
        ("phymeo.dynamics.robot_dynamics", ("forward_dynamics", "mass_matrix", "bias_forces")),
        ("phymeo.dynamics.robot_kinematics", ("forward_kinematics", "point_jacobians")),
        ("phymeo.solvers.floating_tree", ("step_floating_robot", "floating_dynamics_terms")),
    )
    hosts = [(importlib.import_module(name), functions) for name, functions in host_names]
    with MuJoCoTyxRuntime(_compile(case), mode=mode, initial_states={0: initial}) as runtime:
        runtime.set_controls({0: packet})
        calls_before = runtime.counters["completed_kernels"]

        def forbidden(*args, **kwargs):
            raise AssertionError("T1 step must use real TYX dynamics, not CPU/Warp/HOST")

        with monkeypatch.context() as guarded:
            guarded.setitem(sys.modules, "warp", None)
            guarded.setitem(sys.modules, "mujoco_warp", None)
            for name in ("mj_step", "mj_forward", "mj_fullM", "mj_rne", "mj_jac"):
                guarded.setattr(mj, name, forbidden)
            for name in ("solve", "inv", "cholesky"):
                guarded.setattr(np.linalg, name, forbidden)
            for module, names in hosts:
                for name in names:
                    guarded.setattr(module, name, forbidden)
            runtime.step(H)
        state = runtime.snapshot()[0]  # Detached CPU observation is explicitly allowed.
        assert runtime.counters["completed_kernels"] > calls_before
        assert runtime.capabilities["implementation"] == "tyx"
        assert runtime.capabilities["gpu"] is (mode == "cuda")
        np.testing.assert_allclose(state.q, reference.qpos, rtol=0, atol=SHORT_Q_ATOL)
        np.testing.assert_allclose(state.v, reference.qvel, rtol=0, atol=SHORT_V_ATOL)
        _append(tmp_path / "producer-guard.jsonl", {"source_xml": case.xml, "actual": state,
                                                   "native_q": reference.qpos, "native_v": reference.qvel,
                                                   "counters": runtime.counters})


@pytest.mark.parametrize("mode", MODES)
def test_limit_crossing_rejects_whole_coupled_batch_without_stop_repair(mode, tmp_path):
    _deps()
    xml = PENDULUM_XML.replace('ref=".125"', 'ref=".125" limited="true" range="-1 .46" margin=".005"')
    limited = TreeCase("strict-limit", xml, (.329,), (8.,), "pendulum")
    keys = (5, 91)
    initial = {5: MuJoCoInitialState([.454], [8.]), 91: MuJoCoInitialState([.2], [.25])}
    with MuJoCoTyxRuntime(_compile(limited), mode=mode, world_keys=keys,
                          initial_states=initial) as runtime:
        before = runtime.snapshot()
        with pytest.raises(PhysicsStepFailure) as caught:
            runtime.step(H)
        assert caught.value.accepted_worlds == dict.fromkeys(keys, False)
        _assert_same_published(before, runtime.snapshot())
        _append(tmp_path / "limit-rejection.jsonl", {"source_xml": xml, "initial": initial,
                                                    "before": before, "after": runtime.snapshot(),
                                                    "accepted_worlds": caught.value.accepted_worlds})


@pytest.mark.parametrize("mode", MODES)
def test_invalid_control_batch_keeps_all_previous_packets(mode):
    mj = _deps()
    case = CASES[3]
    model = mj.MjModel.from_xml_string(case.xml)
    with MuJoCoTyxRuntime(_compile(case), mode=mode, world_keys=(7, 113),
                          initial_states={7: _initial(model, case), 113: _initial(model, case, 1)}) as runtime:
        packets = {7: _packet(mj, model, case, 0), 113: _packet(mj, model, case, 0, 1)}
        runtime.set_controls(packets)
        before = runtime.snapshot()
        with pytest.raises(ValueError):
            runtime.set_controls({7: packets[7], 113: MuJoCoControl([1.])})
        _assert_same_published(before, runtime.snapshot())
        runtime.step(H)
        assert all(state.tick == 1 for state in runtime.snapshot().values())


@pytest.mark.parametrize("xml,code", [
    (PENDULUM_XML.replace('integrator="Euler"', 'integrator="RK4"'), "integrator"),
    (_chain(9), "joint-domain"),
    (PENDULUM_XML.replace('type="hinge"', 'type="ball"').replace('ref=".125"', ''), "joint-domain"),
])
def test_native_valid_unsupported_features_have_construction_diagnostics(xml, code):
    _deps()
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    compilation = compile_mujoco_xml(xml, require_asset=False)
    with pytest.raises(MuJoCoTyxUnsupported) as caught:
        MuJoCoTyxRuntime(compilation, mode="interpret")
    assert code in {issue.code for issue in caught.value.issues}


@pytest.mark.parametrize("mode", MODES)
def test_disabled_motor_group_keeps_native_positive_force_range_clamp(mode, tmp_path):
    mj = _deps()
    xml = _model("disabled-positive-force", _body("body", _joint("j0", ref=".1")),
                 ("j0",), gravity="0 0 0", gears=(2,))
    xml = xml.replace('<option timestep=', '<option actuatorgroupdisable="1" timestep=')
    xml = xml.replace('name="motor-0"', 'name="motor-0" group="1"')
    xml = xml.replace('forcerange="-.35 .35"', 'forcerange="2 3"')
    special = TreeCase("disabled-positive-force", xml, (.1,), (.25,), "body")
    model = mj.MjModel.from_xml_string(xml)
    initial = _initial(model, special)
    packet = MuJoCoControl([.8], qfrc_applied=[0.], xfrc_applied=np.zeros((model.nbody, 6)))
    with MuJoCoTyxRuntime(_compile(special), mode=mode, initial_states={0: initial}) as runtime:
        before = runtime.snapshot()[0]
        expected = _native_dynamics(mj, model, before.q, before.v, _stored(packet), before.time)
        # Native 3.3.7 zeroes the disabled actuator before force-range clamping:
        # max(2,min(3,0)) * gear(2) = 4 Nm, not zero generalized force.
        np.testing.assert_allclose(expected["applied"], [4.], atol=1e-12)
        runtime.set_controls({0: packet})
        runtime.step(H)
        state = runtime.snapshot()[0]
        _append(tmp_path / "disabled-motor-clamp.jsonl", {"source_xml": xml,
                                                          "oracle": expected, "actual": state.last_step})
        _assert_dynamics(state.last_step, expected)
