"""Independent fixed-native finite contacts and continuous load release/reentry.

Authoring-only until the parent serial executor imports this file.
"""
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

import test_mujoco_tyx_normal_contacts as normal
from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime, MuJoCoTyxUnsupported
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.training.resident_stepper import PhysicsStepFailure
from test_mujoco_tyx_floating import _same, _save

MODES = normal.MODES
X_AXIS = ".7071067811865476 0 .7071067811865476 0"
Y_AXIS = ".7071067811865476 .7071067811865476 0 0"
IDENTITY = "1 0 0 0"
PLATFORM = '<geom name="platform" type="box" size=".24 .18 .05"/>'


@dataclass(frozen=True)
class FiniteCase:
    name: str
    xml: str
    steps: int = 1
    velocity: tuple = ()
    expected_contacts: int | None = None


def _inertial(mass=2., *, offset=True):
    values = np.asarray([.08, .10, .12, .002, -.001, .003]) * (mass / 2)
    position = ".03 -.02 .01" if offset else "0 0 0"
    return f'<inertial mass="{mass}" pos="{position}" fullinertia="{" ".join(map(str, values))}"/>'


def _xml(body, static, *, h=.001, gravity="0 0 -9.81", actuator=""):
    source = normal._xml(body, static=static, gravity=gravity, extra=actuator)
    return source.replace('timestep=".001"', f'timestep="{h}"')


def _free(shape, position, static=PLATFORM, *, quat=IDENTITY, gravity="0 0 0"):
    body = f'''<body name="root" pos="{position}" quat="{quat}"><freejoint/>
      {_inertial()}{shape}</body>'''
    return _xml(body, static, gravity=gravity)


CAP = '<geom name="moving" type="capsule" size=".1 .2" quat="{quat}"/>'
BOX = '<geom name="moving" type="box" size=".12 .08 .06"/>'
STATIC_CAP = '<geom name="fixed" type="capsule" size=".1 .2"/>'

GEOMETRY = (
    FiniteCase("cap-cap-parallel", _free(CAP.format(quat=IDENTITY), ".199 0 .04", STATIC_CAP), expected_contacts=2),
    FiniteCase("cap-cap-skew", _free(CAP.format(quat=X_AXIS), "0 .199 0", STATIC_CAP), expected_contacts=1),
    FiniteCase("cap-cap-endpoint", _free(CAP.format(quat=IDENTITY), "0 0 .599", STATIC_CAP), expected_contacts=1),
    FiniteCase("cap-box-side", _free(CAP.format(quat=X_AXIS), "0 0 .149"), expected_contacts=2),
    FiniteCase("cap-box-end", _free(CAP.format(quat=IDENTITY), "0 0 .349"), expected_contacts=1),
    FiniteCase("cap-box-edge", _free(CAP.format(quat=Y_AXIS), ".31 0 .11")),
    FiniteCase("cap-box-corner", _free(CAP.format(quat=IDENTITY), ".30 .24 .30")),
    FiniteCase("box-box-face", _free(BOX, "0 0 .109"), expected_contacts=4),
    FiniteCase("box-box-yaw-face", _free('<geom name="moving" type="box" size=".2 .2 .06"/>',
                                        "0 0 .109", quat=".9238795325112867 0 0 .3826834323650898")),
    FiniteCase("box-box-edge", _free(BOX, ".18 0 .13", quat=".9238795325112867 0 .3826834323650898 0")),
    FiniteCase("box-box-oblique", _free(BOX, ".05 -.03 .12", quat=".96 .18 -.12 .1")),
)


def _branches():
    body = '<body name="root">'
    for index, sign, mass in ((0, -1, 1.), (1, 1, .6)):
        body += f'''<body name="hinge{index}" pos="{sign*.099} 0 .2">
        <joint name="turn{index}" axis="0 1 0"/>
        <inertial mass=".2" pos="0 0 -.03" diaginertia=".01 .012 .014"/>
        <body name="tip{index}" pos="0 0 -.18"><joint name="slide{index}" type="slide" axis="1 0 0"/>
        <inertial mass="{mass}" pos="0 .015 .01" diaginertia=".03 .04 .05"/>
        <geom name="cap{index}" type="capsule" size=".1 .1" pos="0 0 .02"/>
        </body></body>'''
    return _xml(body + "</body>", "", gravity="0 0 0")


DYNAMIC = (
    FiniteCase("cap-box-eccentric", _free(CAP.format(quat=X_AXIS), "0 0 .149",
                                         gravity="0 0 -9.81"), 24,
               (.03, -.02, -.06, .08, -.05, .04)),
    FiniteCase("box-box-eccentric", _free(BOX, ".05 -.03 .12",
                                         quat=".96 .18 -.12 .1", gravity="0 0 -9.81"), 24,
               (.03, -.02, -.06, .08, -.05, .04)),
    FiniteCase("two-moving-branches", _branches(), 32, (.03, .04, -.02, -.05), 2),
)


def _initial(model, case, slot=0):
    q, v = model.qpos0.copy(), np.zeros(model.nv)
    if case.velocity:
        v[:] = case.velocity
    if slot:
        if model.nq > model.nv:
            q[2] += .004 * slot
        else:
            q[:] += .002 * slot
        v *= 1 + .1 * slot
    return MuJoCoInitialState(q, v)


def _packet(model, *, tick=0, driven=False):
    force, wrench = np.zeros(model.nv), np.zeros((model.nbody, 6))
    if driven:
        force[-1] = .01 * (-1 if tick >= 12 else 1)
        wrench[-1] = [.03, -.02, .08, .003, -.002, .004]
    return MuJoCoControl(np.zeros(model.nu), qfrc_applied=force, xfrc_applied=wrench)


def _native(mj, model, q, v, packet, time):
    expected = normal._native(mj, model, q, v, packet, time)
    ordinal = {}
    for contact in expected["contacts"]:
        pair = contact["geometry_ids"]
        contact["feature_index"] = ordinal.get(pair, 0)
        ordinal[pair] = contact["feature_index"] + 1
    rhs = expected["applied"] + expected["passive"] - expected["bias"] + expected["qfrc_constraint"]
    expected["integration_acceleration"] = np.linalg.solve(
        expected["mass_matrix"] + model.opt.timestep * np.diag(model.dof_damping), rhs)
    return expected


def _ordinal_match(actual, expected):
    assert len(actual) == len(expected)
    by_identity = {(c["geometry_ids"], c["feature_index"]): c for c in expected}
    assert len(by_identity) == len(expected)
    found = []
    used = set()
    for contact in actual:
        identity = (tuple(contact.geometry_ids), contact.feature_index)
        assert identity in by_identity and identity not in used
        used.add(identity)
        oracle = by_identity[identity]
        np.testing.assert_allclose(contact.position, oracle["position"], rtol=3e-5, atol=4e-5)
        found.append(oracle)
    assert used == set(by_identity)
    return found


def _assert_receipt(row, expected, model):
    # Only the old test helper's correspondence policy is replaced. Product
    # computation and the native geometry/force oracle are never patched.
    with patch.object(normal, "_match", _ordinal_match):
        normal._assert_receipt(row, expected, model)


def _run(mode, case, tmp_path, *, driven=False):
    mj = normal._deps()
    model = mj.MjModel.from_xml_string(case.xml)
    h = model.opt.timestep
    path = tmp_path / "finite-native.jsonl"
    _save(path, dict(case=case.name, xml=case.xml, mode=mode, native_version=mj.__version__,
                     steps=case.steps, h=h, tolerances=normal.TOL,
                     force_tolerance=normal.FORCE_TOL, q_atol=normal.Q_ATOL, v_atol=normal.V_ATOL))
    selected = {0, case.steps // 2, case.steps - 1}
    receipts = []
    with MuJoCoTyxRuntime(normal._compile(case.xml), mode=mode, world_keys=(19,),
                         initial_states={19: _initial(model, case)}) as runtime:
        start = runtime.snapshot()[19]
        continuous = mj.MjData(model)
        continuous.qpos[:], continuous.qvel[:] = start.q, start.v
        for tick in range(case.steps):
            before = runtime.snapshot()[19]
            packet = normal._stored(_packet(model, tick=tick, driven=driven))
            expected = _native(mj, model, before.q, before.v, packet, before.time)
            normal._set(continuous, packet)
            mj.mj_step(model, continuous)
            runtime.set_controls({19: packet})
            assert runtime.step(h) == {19: True}
            state = runtime.snapshot()[19]
            detail = dict(tick=tick, input=before, control=packet, oracle=expected, actual=state.last_step)
            if tick in selected:
                _save(path, detail)
            try:
                _assert_receipt(state.last_step, expected, model)
                np.testing.assert_allclose(state.q, continuous.qpos, rtol=0, atol=normal.Q_ATOL)
                np.testing.assert_allclose(state.v, continuous.qvel, rtol=0, atol=normal.V_ATOL)
                assert state.time == pytest.approx(continuous.time, abs=2e-6)
                if tick == 0:
                    assert expected["contacts"], "positive fixture must exercise actual contact"
                    if case.expected_contacts is not None:
                        assert len(expected["contacts"]) == case.expected_contacts
            except BaseException:
                if tick not in selected:
                    _save(path, detail)
                raise
            _save(path, dict(tick=tick+1, q=state.q, v=state.v, control=packet,
                            contacts=len(state.last_step.contacts),
                            forces=[c.normal_force for c in state.last_step.contacts]))
            if tick in selected:
                receipts.append(state.last_step)
        assert runtime.capabilities["numerical_solver"] == "normal-dual-active-set-v1"
    return model, receipts


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", GEOMETRY, ids=lambda c: c.name)
def test_new_finite_primitive_manifolds_match_native_complete_ordinals(mode, case, tmp_path):
    _run(mode, case, tmp_path)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", DYNAMIC, ids=lambda c: c.name)
def test_eccentric_rotation_and_moving_branch_reaction(mode, case, tmp_path):
    model, rows = _run(mode, case, tmp_path, driven=True)
    first = rows[0]
    assert any(c.normal_force > .001 for c in first.contacts)
    if case.name == "two-moving-branches":
        assert model.nv == 4
        assert np.all(np.abs(first.qfrc_constraint) > .001)
        for c in first.contacts:
            assert all(body > 0 for body in c.body_ids)
            np.testing.assert_allclose(c.force_on_geometry1_world + c.force_on_geometry2_world,
                                       np.zeros(3), atol=1e-7)
    else:
        assert np.linalg.norm(first.qfrc_constraint[3:6]) > .001


def _support_xml(shape, mass=2., h=.001, *, inactive=False):
    height = ".149" if shape == "capsule" else ".109"
    specification = (f'type="capsule" size=".1 .12" quat="{X_AXIS}"'
                     if shape == "capsule" else 'type="box" size=".12 .08 .06"')
    material = 'margin=".03125" gap=".0234375"' if inactive else ""
    body = f'''<body name="root" pos="0 0 {height}">
      <joint name="vertical" type="slide" axis="0 0 1" armature="0" damping="0"/>
      {_inertial(mass, offset=False)}
      <geom name="moving" {specification} {material}/></body>'''
    return _xml(body, PLATFORM, h=h, actuator='<actuator><motor joint="vertical"/></actuator>')


def _pulse_packet(model, tick, mass):
    h = model.opt.timestep
    force = 3 * mass * 9.81 if round(.096/h) <= tick < round(.160/h) else 0.
    return MuJoCoControl([force], qfrc_applied=np.zeros(model.nv),
                         xfrc_applied=np.zeros((model.nbody, 6)))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", ("capsule", "box"))
@pytest.mark.parametrize("mass", (.25, 2.))
@pytest.mark.parametrize("h", (.001, .002))
def test_finite_support_full_unload_and_reentry_one_continuous_trajectory(
        mode, shape, mass, h, tmp_path):
    mj = normal._deps()
    xml = _support_xml(shape, mass, h)
    model = mj.MjModel.from_xml_string(xml)
    initial = MuJoCoInitialState(model.qpos0.copy(), [-.02])
    steps = round(.512/h)
    selected = {0, round(.096/h)-1, round(.096/h), round(.160/h)-1,
                round(.160/h), steps-1}
    path = tmp_path / "finite-unload-reentry.jsonl"
    _save(path, dict(xml=xml, mode=mode, mass=mass, h=h, steps=steps,
                     pulse=(.096, .160, 3*mass*9.81),
                     q_atol=normal.Q_ATOL, v_atol=normal.V_ATOL,
                     force_tolerance=normal.FORCE_TOL))
    history, separated, reentered = [], False, False
    with MuJoCoTyxRuntime(normal._compile(xml), mode=mode, world_keys=(7,),
                         initial_states={7: initial}) as runtime:
        start = runtime.snapshot()[7]
        continuous = mj.MjData(model)
        continuous.qpos[:], continuous.qvel[:] = start.q, start.v
        for tick in range(steps):
            before = runtime.snapshot()[7]
            packet = normal._stored(_pulse_packet(model, tick, mass))
            oracle = _native(mj, model, before.q, before.v, packet, before.time)
            normal._set(continuous, packet)
            mj.mj_step(model, continuous)
            runtime.set_controls({7: packet})
            runtime.step(h)
            state = runtime.snapshot()[7]
            receipt = state.last_step
            count = len(receipt.contacts)
            force = sum(c.normal_force for c in receipt.contacts)
            new_separation = tick >= round(.096/h) and count == 0 and not separated
            new_reentry = separated and count > 0 and force > .01*mass*9.81 and not reentered
            selected_now = tick in selected or new_separation or new_reentry
            detail = dict(tick=tick, input=before, control=packet, oracle=oracle, actual=receipt)
            if selected_now:
                _save(path, detail)
            try:
                _assert_receipt(receipt, oracle, model)
                np.testing.assert_allclose(state.q, continuous.qpos, rtol=0, atol=normal.Q_ATOL)
                np.testing.assert_allclose(state.v, continuous.qvel, rtol=0, atol=normal.V_ATOL)
                assert state.time == pytest.approx(continuous.time, abs=2e-6)
            except BaseException:
                if not selected_now:
                    _save(path, detail)
                raise
            row = dict(tick=tick+1, input_time=tick*h, time=state.time,
                       q=state.q.copy(), v=state.v.copy(), control=packet,
                       contacts=count, normal_force=force)
            history.append(row)
            _save(path, row)
            separated |= new_separation
            reentered |= new_reentry
        assert separated and reentered
    weight = mass * 9.81
    assert any(r["normal_force"] > .5*weight for r in history if .06 <= r["input_time"] < .096)
    off = [r for r in history if r["input_time"] >= .096 and r["contacts"] == 0]
    assert off and all(r["normal_force"] == 0 for r in off)
    longest, run, previous = 0, 0, None
    for row in off:
        run = run + 1 if previous is not None and row["tick"] == previous + 1 else 1
        longest = max(longest, run)
        previous = row["tick"]
    assert longest*h >= .02
    first_off = off[0]["tick"]
    reentry = next(r for r in history if r["tick"] > first_off and r["contacts"] > 0
                   and r["normal_force"] > .01*weight)
    assert .160 < reentry["input_time"] < .48
    last = history[-round(.02/h):]
    assert .7*weight <= np.mean([r["normal_force"] for r in last]) <= 1.3*weight
    assert abs(history[-1]["v"][0]) < .1
    assert abs(history[-1]["q"][0] - initial.q[0]) < .01


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("index,expected_count", [(0, 2), (3, 2), (7, 4)])
def test_finite_pair_runtime_capacities_exact_and_overflow(mode, index, expected_count, tmp_path):
    mj = normal._deps()
    case = GEOMETRY[index]
    model = mj.MjModel.from_xml_string(case.xml)
    limits_grid = (
        (expected_count, expected_count, True),
        (expected_count-1, expected_count, False),
        (expected_count, expected_count-1, False),
    )
    for contact_cap, constraint_cap, should_pass in limits_grid:
        limits = dict(contact_capacity=contact_cap, constraint_capacity=constraint_cap)
        with MuJoCoTyxRuntime(normal._compile(case.xml), mode=mode, world_keys=(3, 91),
                             **limits) as runtime:
            before = runtime.snapshot()
            if not should_pass:
                with pytest.raises(PhysicsStepFailure) as caught:
                    runtime.step(model.opt.timestep)
                assert caught.value.accepted_worlds == {3: False, 91: False}
                _same(before, runtime.snapshot())
            else:
                runtime.step(model.opt.timestep)
                for key, state in runtime.snapshot().items():
                    packet = _packet(model)
                    expected = _native(mj, model, before[key].q, before[key].v, packet, before[key].time)
                    assert len(expected["contacts"]) == expected_count
                    _assert_receipt(state.last_step, expected, model)
                    _save(tmp_path / "finite-capacity.jsonl",
                          dict(case=case.name, limits=limits, key=key,
                               oracle=expected, actual=state.last_step))


@pytest.mark.parametrize("mode", MODES)
def test_detected_inactive_capsule_contacts_consume_capacity(mode, tmp_path):
    mj = normal._deps()
    static = '<geom name="fixed" type="capsule" size=".125 .125" margin=".03125" gap=".0234375"/>'
    moving = '<geom name="moving" type="capsule" size=".125 .125" margin=".03125" gap=".0234375"/>'
    xml = _free(moving, ".265625 0 0", static)
    model = mj.MjModel.from_xml_string(xml)
    with MuJoCoTyxRuntime(normal._compile(xml), mode=mode, contact_capacity=2,
                         constraint_capacity=0) as runtime:
        before = runtime.snapshot()[0]
        expected = _native(mj, model, before.q, before.v, _packet(model), before.time)
        assert len(expected["contacts"]) == 2 and not any(c["active"] for c in expected["contacts"])
        runtime.step(model.opt.timestep)
        receipt = runtime.snapshot()[0].last_step
        _assert_receipt(receipt, expected, model)
        assert all(c.normal_force == 0 for c in receipt.contacts)
        _save(tmp_path / "inactive-capacity.jsonl", dict(xml=xml, oracle=expected, actual=receipt))
    with MuJoCoTyxRuntime(normal._compile(xml), mode=mode, contact_capacity=1) as runtime:
        before = runtime.snapshot()
        with pytest.raises(PhysicsStepFailure):
            runtime.step(model.opt.timestep)
        _same(before, runtime.snapshot())


def test_two_box_pairs_exceed_conservative_static_layout():
    normal._deps()
    xml = GEOMETRY[7].xml.replace(
        '<geom name="moving" type="box" size=".12 .08 .06"/>',
        '<geom name="moving" type="box" size=".12 .08 .06"/>'
        '<geom name="second" type="box" size=".08 .06 .04" pos=".4 0 0"/>')
    with pytest.raises(MuJoCoTyxUnsupported) as caught:
        MuJoCoTyxRuntime(normal._compile(xml), mode="interpret")
    assert "contact-layout-capacity" in str(caught.value)


def _assert_publication_unchanged(before, after):
    _same(before, after)
    for key in before:
        left, right = before[key].last_step, after[key].last_step
        assert (left is None) == (right is None)
        if left is not None:
            np.testing.assert_array_equal(left.qfrc_constraint, right.qfrc_constraint)
            assert left.input_time == right.input_time
            np.testing.assert_array_equal([c.normal_force for c in left.contacts],
                                          [c.normal_force for c in right.contacts])


@pytest.mark.parametrize("mode", MODES)
def test_finite_w4_selected_reset_clone_and_next_real_step(mode, tmp_path):
    mj = normal._deps()
    case = GEOMETRY[7]
    model = mj.MjModel.from_xml_string(case.xml)
    keys = (3, 19, 91, 203)
    initial = {key: _initial(model, case, slot) for slot, key in enumerate(keys)}
    compiled = normal._compile(case.xml)
    with MuJoCoTyxRuntime(compiled, mode=mode, world_keys=keys, initial_states=initial) as runtime:
        runtime.step(model.opt.timestep)
        before = runtime.snapshot()
        assert len({len(s.last_step.contacts) for s in before.values()}) > 1
        with runtime.clone() as clone:
            runtime.reset((keys[0],))
            for key in keys[1:]:
                _assert_publication_unchanged({key: before[key]}, {key: runtime.snapshot()[key]})
            with MuJoCoTyxRuntime(compiled, mode=mode, world_keys=(keys[0],),
                                 initial_states={keys[0]: initial[keys[0]]}) as fresh:
                runtime.step(model.opt.timestep)
                clone.step(model.opt.timestep)
                fresh.step(model.opt.timestep)
                actual, expected = runtime.snapshot(), clone.snapshot()
                _assert_publication_unchanged({keys[0]: actual[keys[0]]}, fresh.snapshot())
                for key in keys[1:]:
                    _assert_publication_unchanged({key: actual[key]}, {key: expected[key]})
                _save(tmp_path / "finite-reset-next.jsonl",
                      dict(xml=case.xml, before=before, actual=actual, clone=expected,
                           fresh=fresh.snapshot()))


@pytest.mark.parametrize("mode", MODES)
def test_finite_late_download_failure_preserves_force_motion_and_retries(mode, monkeypatch, tmp_path):
    mj = normal._deps()
    case = DYNAMIC[1]
    model = mj.MjModel.from_xml_string(case.xml)
    keys = (7, 29, 113, 211)
    initial = {key: _initial(model, case, slot) for slot, key in enumerate(keys)}
    with MuJoCoTyxRuntime(normal._compile(case.xml), mode=mode, world_keys=keys,
                         initial_states=initial) as runtime:
        runtime.set_controls(dict.fromkeys(keys, _packet(model, driven=True)))
        runtime.step(model.opt.timestep)
        with runtime.clone() as clone:
            before, kernels = runtime.snapshot(), runtime.counters["completed_kernels"]
            download = runtime._device.download
            def fail_after_actual_download():
                download()
                raise RuntimeError("T3b injected after actual finite-contact readback")
            monkeypatch.setattr(runtime._device, "download", fail_after_actual_download)
            with pytest.raises(PhysicsStepFailure) as caught:
                runtime.step(model.opt.timestep)
            assert caught.value.accepted_worlds == dict.fromkeys(keys, False)
            assert caught.value.acceptance_known and not caught.value.physics_accepted
            assert runtime.counters["completed_kernels"] > kernels
            _assert_publication_unchanged(before, runtime.snapshot())
            _save(tmp_path / "finite-late-failure.jsonl",
                  dict(before=before, after=runtime.snapshot(), mask=caught.value.accepted_worlds))
            monkeypatch.setattr(runtime._device, "download", download)
            runtime.step(model.opt.timestep)
            clone.step(model.opt.timestep)
            _assert_publication_unchanged(runtime.snapshot(), clone.snapshot())
            _save(tmp_path / "finite-late-failure.jsonl",
                  dict(retry=runtime.snapshot(), clone=clone.snapshot()))


@pytest.mark.parametrize("mode", MODES)
def test_finite_step_disallows_cpu_warp_host_and_numpy_numerics(mode, monkeypatch, tmp_path):
    import importlib
    import sys
    mj = normal._deps()
    case = DYNAMIC[1]
    model = mj.MjModel.from_xml_string(case.xml)
    modules = [(importlib.import_module(name), functions) for name, functions in (
        ("phymeo.dynamics.robot_dynamics", ("forward_dynamics", "mass_matrix", "bias_forces")),
        ("phymeo.dynamics.robot_kinematics", ("forward_kinematics", "point_jacobians")),
        ("phymeo.solvers.floating_tree", ("step_floating_robot", "floating_dynamics_terms")),
    )]
    with MuJoCoTyxRuntime(normal._compile(case.xml), mode=mode,
                         initial_states={0: _initial(model, case)}) as runtime:
        before = runtime.snapshot()[0]
        packet = normal._stored(_packet(model, driven=True))
        oracle = _native(mj, model, before.q, before.v, packet, before.time)
        runtime.set_controls({0: packet})
        count = runtime.counters["completed_kernels"]
        def forbidden(*args, **kwargs):
            raise AssertionError("finite contacts must execute through genuine TYX operators")
        with monkeypatch.context() as block:
            for name in ("mj_step", "mj_forward", "mj_fullM", "mj_rne", "mj_jac",
                         "mj_collision", "mj_fwdConstraint"):
                block.setattr(mj, name, forbidden)
            for name in ("solve", "inv", "cholesky", "lstsq"):
                block.setattr(np.linalg, name, forbidden)
            for module, functions in modules:
                for name in functions:
                    block.setattr(module, name, forbidden)
            block.setitem(sys.modules, "warp", None)
            block.setitem(sys.modules, "mujoco_warp", None)
            assert runtime.step(model.opt.timestep) == {0: True}
        actual = runtime.snapshot()[0].last_step
        assert runtime.counters["completed_kernels"] > count
        assert runtime.capabilities["gpu"] is (mode == "cuda")
        _assert_receipt(actual, oracle, model)
        _save(tmp_path / "finite-no-fallback.jsonl", dict(xml=case.xml, oracle=oracle, actual=actual))



@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("index,original_position", [(0, ".199 0 .04"), (3, "0 0 .149"), (7, "0 0 .109")])
def test_finite_pair_separation_returns_no_truncated_or_phantom_contacts(
        mode, index, original_position, tmp_path):
    mj = normal._deps()
    case = GEOMETRY[index]
    original = f'<body name="root" pos="{original_position}"'
    assert original in case.xml
    xml = case.xml.replace(original, '<body name="root" pos="0 0 3"')
    model = mj.MjModel.from_xml_string(xml)
    with MuJoCoTyxRuntime(normal._compile(xml), mode=mode, contact_capacity=0,
                         constraint_capacity=0) as runtime:
        before = runtime.snapshot()[0]
        oracle = _native(mj, model, before.q, before.v, _packet(model), before.time)
        assert oracle["contacts"] == []
        runtime.step(model.opt.timestep)
        actual = runtime.snapshot()[0].last_step
        _assert_receipt(actual, oracle, model)
        assert actual.contacts == ()
        _save(tmp_path / "finite-separated.jsonl", dict(xml=xml, oracle=oracle, actual=actual))
