"""Batch publication, admission and compact EFC boundaries for T4.

Pure authored tests; project execution is owned by the parent integration lane.
"""
import importlib
import sys

import numpy as np
import pytest
import test_mujoco_tyx_constraints as mixed
import test_mujoco_tyx_normal_contacts as normal
from test_mujoco_tyx_floating import _same, _save

from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime, MuJoCoTyxUnsupported
from phymeo.backends.mujoco_runtime import MuJoCoControl
from phymeo.training.resident_stepper import PhysicsStepFailure

MODES = normal.MODES
KEYS = (7, 29, 113, 211)


def attempt_scratch(runtime):
    """Read the already executed attempt; never replay a producer or solver."""
    try:
        return runtime._device.download()
    except Exception as error:
        return {"readback_error": str(error)}


def assert_publication(left, right):
    _same(left, right)
    for key in left:
        a, b = left[key].last_step, right[key].last_step
        assert (a is None) == (b is None)
        if a is None:
            continue
        for name in ("input_q", "input_v", "qacc", "integration_acceleration", "qfrc_constraint"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert len(a.constraints) == len(b.constraints) and len(a.contacts) == len(b.contacts)
        for ra, rb in zip(a.constraints, b.constraints, strict=True):
            for field in ("constraint_type", "source_id", "side_or_edge", "geometry_slot", "dof",
                          "position", "margin", "velocity", "regularization", "reference_acceleration",
                          "inverse_regularization", "impedance", "diag_approx", "lower", "upper",
                          "force", "gradient"):
                assert getattr(ra, field) == getattr(rb, field)
            np.testing.assert_array_equal(ra.jacobian, rb.jacobian)
        for ca, cb in zip(a.contacts, b.contacts, strict=True):
            assert ca.geometry_ids == cb.geometry_ids and ca.feature_index == cb.feature_index
            for field in ("position", "frame", "jacobian", "wrench_contact_frame",
                          "wrench_on_body1_world", "wrench_on_body2_world"):
                np.testing.assert_array_equal(getattr(ca, field), getattr(cb, field))
        assert a.raw_contact_fields.keys() == b.raw_contact_fields.keys()
        for field in a.raw_contact_fields:
            va, vb = a.raw_contact_fields[field], b.raw_contact_fields[field]
            if field == "solver_diagnostics":
                assert va == vb
            else:
                np.testing.assert_array_equal(va, vb)


def varied_initial(model):
    result = {}
    for key, q in zip(KEYS, ([.195, 0, 0], [-.195, 0, .02],
                             [0, 0, 0], [0, 0, .05]), strict=True):
        result[key] = mixed.initial(model, np.array(q), np.array([.01, -.02, 0]))
    return result


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dim", [1, 3, 4, 6])
def test_detected_and_expanded_capacities_exact_and_overflow(mode, dim, tmp_path):
    mj = normal._deps()
    xml = mixed.slider_xml(dim=dim, limits=True)
    model = mj.MjModel.from_xml_string(xml)
    oracle = mixed.native(mj, model, model.qpos0, np.zeros(model.nv), mixed.packet(model))
    nc, ne = len(oracle["contacts"]), len(oracle["rows"])
    assert nc == 1 and ne == (1 if dim == 1 else 2*(dim-1))
    with MuJoCoTyxRuntime(normal._compile(xml), mode=mode,
                         contact_capacity=nc, constraint_capacity=ne) as exact:
        before = exact.snapshot()[0]
        oracle = mixed.native(mj, model, before.q, before.v, mixed.packet(model))
        exact.step(model.opt.timestep)
        mixed.assert_receipt(exact.snapshot()[0].last_step, oracle, model)
        _save(tmp_path/"exact-capacity.jsonl", {"dim": dim, "actual": exact.snapshot()[0], "oracle": oracle})
    # All worlds fail atomically even when just one world needs the missing slot.
    initials = {7: mixed.initial(model), 113: mixed.initial(model, np.array([0, 0, .1]))}
    for contact_cap, efc_cap in ((nc-1, ne), (nc, ne-1)):
        with MuJoCoTyxRuntime(normal._compile(xml), mode=mode, world_keys=(7, 113),
                             initial_states=initials,
                             contact_capacity=contact_cap, constraint_capacity=efc_cap) as runtime:
            before = runtime.snapshot()
            with pytest.raises(PhysicsStepFailure) as caught:
                runtime.step(model.opt.timestep)
            assert caught.value.acceptance_known and not caught.value.physics_accepted
            assert caught.value.accepted_worlds == {7: False, 113: False}
            assert_publication(before, runtime.snapshot())
            _save(tmp_path/"capacity-overflow.jsonl", {"caps": [contact_cap, efc_cap],
                                                       "before": before, "after": runtime.snapshot(),
                                                       "mask": caught.value.accepted_worlds,
                                                       "attempt_scratch": attempt_scratch(runtime)})


@pytest.mark.parametrize("mode", MODES)
def test_inactive_detected_contact_keeps_full_six_jacobian_without_efc(mode, tmp_path):
    mj = normal._deps()
    xml = mixed.slider_xml(dim=6).replace(
        '<geom condim="6"', '<geom margin=".02" gap=".015" condim="6"')
    model = mj.MjModel.from_xml_string(xml)
    initial = mixed.initial(model, np.array([0, 0, .011]), np.array([.2, -.1, -.02]))
    with MuJoCoTyxRuntime(normal._compile(xml), mode=mode, initial_states={0: initial},
                         contact_capacity=1, constraint_capacity=0) as runtime:
        before = runtime.snapshot()[0]
        oracle = mixed.native(mj, model, before.q, before.v, mixed.packet(model))
        assert len(oracle["contacts"]) == 1 and not oracle["rows"]
        runtime.step(model.opt.timestep)
        row = runtime.snapshot()[0].last_step
        mixed.assert_receipt(row, oracle, model)
        assert len(row.contacts) == 1 and not row.contacts[0].active
        assert np.linalg.norm(row.contacts[0].jacobian) > 0
        np.testing.assert_array_equal(row.contacts[0].wrench_contact_frame, 0)
        _save(tmp_path/"inactive.jsonl", {"actual": row, "oracle": oracle})


@pytest.mark.parametrize("mode", MODES)
def test_sparse_worlds_reset_clone_preserve_next_force_and_motion(mode, tmp_path):
    mj = normal._deps()
    xml = mixed.slider_xml(dim=6, dry=.2, limits=True)
    model = mj.MjModel.from_xml_string(xml)
    compiled = normal._compile(xml)
    initials = varied_initial(model)
    with MuJoCoTyxRuntime(compiled, mode=mode, world_keys=KEYS, initial_states=initials) as runtime:
        controls = {key: mixed.packet(model, np.array([.03*(i+1), -.01, .02]))
                    for i, key in enumerate(KEYS)}
        runtime.set_controls(controls)
        runtime.step(model.opt.timestep)
        before = runtime.snapshot()
        assert len({len(s.last_step.constraints) for s in before.values()}) >= 3
        for key in KEYS:
            src = initials[key]
            oracle = mixed.native(mj, model, src.q, src.v, controls[key])
            mixed.assert_receipt(before[key].last_step, oracle, model)
        with runtime.clone() as clone:
            runtime.reset((KEYS[0],))
            reset = runtime.snapshot()
            assert reset[KEYS[0]].last_step is None and reset[KEYS[0]].tick == 0
            assert_publication({k: before[k] for k in KEYS[1:]}, {k: reset[k] for k in KEYS[1:]})
            with MuJoCoTyxRuntime(compiled, mode=mode, world_keys=(KEYS[0],),
                                 initial_states={KEYS[0]: initials[KEYS[0]]}) as fresh:
                runtime.set_controls(controls)
                clone.set_controls(controls)
                fresh.set_controls({KEYS[0]: controls[KEYS[0]]})
                runtime.step(model.opt.timestep); clone.step(model.opt.timestep); fresh.step(model.opt.timestep)
                actual, expected = runtime.snapshot(), clone.snapshot()
                assert_publication({KEYS[0]: actual[KEYS[0]]}, fresh.snapshot())
                assert_publication({k: actual[k] for k in KEYS[1:]},
                                   {k: expected[k] for k in KEYS[1:]})
                _save(tmp_path/"reset-force-motion.jsonl", {"before": before, "reset": reset,
                                                           "actual": actual, "clone": expected,
                                                           "fresh": fresh.snapshot()})


@pytest.mark.parametrize("mode", MODES)
def test_late_real_readback_failure_rolls_back_all_mixed_rows_and_retries(mode, monkeypatch, tmp_path):
    mj = normal._deps()
    xml = mixed.slider_xml(dim=6, dry=.2, limits=True)
    model = mj.MjModel.from_xml_string(xml)
    with MuJoCoTyxRuntime(normal._compile(xml), mode=mode, world_keys=KEYS,
                         initial_states=varied_initial(model)) as runtime:
        runtime.step(model.opt.timestep)
        controls = {key: mixed.packet(model, np.array([.03*(i+1), -.01, .02]))
                    for i, key in enumerate(KEYS)}
        runtime.set_controls(controls)
        with runtime.clone() as clone:
            before, launches = runtime.snapshot(), runtime.counters["completed_kernels"]
            download = runtime._device.download
            attempt = {}
            def failed_download():
                attempt.update(download())
                raise RuntimeError("T4 injected after actual mixed constraint readback")
            monkeypatch.setattr(runtime._device, "download", failed_download)
            with pytest.raises(PhysicsStepFailure) as caught:
                runtime.step(model.opt.timestep)
            assert caught.value.accepted_worlds == dict.fromkeys(KEYS, False)
            assert caught.value.acceptance_known and not caught.value.physics_accepted
            assert runtime.counters["completed_kernels"] > launches
            assert_publication(before, runtime.snapshot())
            _save(tmp_path/"late-failure.jsonl", {"before": before, "after": runtime.snapshot(),
                                                 "mask": caught.value.accepted_worlds, "attempt_scratch": attempt})
            monkeypatch.setattr(runtime._device, "download", download)
            runtime.step(model.opt.timestep); clone.step(model.opt.timestep)
            assert_publication(runtime.snapshot(), clone.snapshot())
            _save(tmp_path/"late-failure.jsonl", {"retry": runtime.snapshot(), "clone": clone.snapshot()})


@pytest.mark.parametrize("mode", MODES)
def test_mixed_step_does_not_call_cpu_warp_host_or_numpy_mechanics(mode, monkeypatch, tmp_path):
    mj = normal._deps()
    xml = mixed.mixed_xml()
    model = mj.MjModel.from_xml_string(xml)
    modules = [(importlib.import_module(name), fields) for name, fields in (
        ("phymeo.dynamics.robot_dynamics", ("forward_dynamics", "mass_matrix", "bias_forces")),
        ("phymeo.dynamics.robot_kinematics", ("forward_kinematics", "point_jacobians")),
        ("phymeo.solvers.floating_tree", ("step_floating_robot", "floating_dynamics_terms")))]
    with MuJoCoTyxRuntime(normal._compile(xml), mode=mode,
                         initial_states={0: mixed.initial(model, np.array([.195, -.005]))}) as runtime:
        control = mixed.packet(model, np.array([.03, -.2]))
        before = runtime.snapshot()[0]
        oracle = mixed.native(mj, model, before.q, before.v, control)
        runtime.set_controls({0: control})
        launches = runtime.counters["completed_kernels"]
        def forbidden(*args, **kwargs):
            raise AssertionError("mixed constraint step must use actual TYX operators")
        with monkeypatch.context() as block:
            for name in ("mj_step", "mj_forward", "mj_fullM", "mj_rne", "mj_jac",
                         "mj_collision", "mj_fwdConstraint"):
                block.setattr(mj, name, forbidden)
            for name in ("solve", "inv", "cholesky", "lstsq"):
                block.setattr(np.linalg, name, forbidden)
            for module, fields in modules:
                for name in fields:
                    block.setattr(module, name, forbidden)
            block.setitem(sys.modules, "warp", None)
            block.setitem(sys.modules, "mujoco_warp", None)
            assert runtime.step(model.opt.timestep) == {0: True}
        assert runtime.counters["completed_kernels"] > launches
        assert runtime.capabilities["implementation"] == "tyx"
        assert runtime.capabilities["gpu"] is (mode == "cuda")
        assert runtime.capabilities["numerical_solver"] == "pyramidal-box-active-set-v1"
        mixed.assert_receipt(runtime.snapshot()[0].last_step, oracle, model)
        _save(tmp_path/"no-fallback.jsonl", {"actual": runtime.snapshot()[0], "oracle": oracle})


def many_candidates_xml():
    xml = mixed.slider_xml(dim=3)
    markers = "".join(
        f'<geom name="marker{i}" type="box" pos="{10+i} 0 .099" size=".05 .05 .05"/>'
        for i in range(39))
    # The last finite collider is deliberately active beyond the previous 32 slots.
    markers += '<geom name="last" type="box" pos=".149 0 .099" size=".05 .05 .05"/>'
    markers += '<body name="excluded" pos="0 0 .099">' + mixed._inertial() + '<geom name="ignored" type="sphere" size=".2"/></body>'
    xml = xml.replace(mixed.FLOOR, mixed.FLOOR+markers)
    return xml.replace('</worldbody>', '</worldbody><contact><exclude body1="root" body2="excluded"/></contact>')


@pytest.mark.parametrize("mode", MODES)
def test_dynamic_compaction_preserves_late_candidates_and_native_excludes(mode, tmp_path):
    mj = normal._deps()
    xml = many_candidates_xml()
    model = mj.MjModel.from_xml_string(xml)
    assert model.ngeom > 40 and model.nexclude == 1
    with MuJoCoTyxRuntime(normal._compile(xml), mode=mode, contact_capacity=2,
                         constraint_capacity=8) as runtime:
        before = runtime.snapshot()[0]
        oracle = mixed.native(mj, model, before.q, before.v, mixed.packet(model))
        assert len(oracle["contacts"]) == 2 and len(oracle["rows"]) == 8
        last = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "last")
        ignored = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "ignored")
        assert any(last in c["geometry_ids"] for c in oracle["contacts"])
        assert all(ignored not in c["geometry_ids"] for c in oracle["contacts"])
        runtime.step(model.opt.timestep)
        row = runtime.snapshot()[0].last_step
        mixed.assert_receipt(row, oracle, model)
        assert row.raw_contact_fields["candidate_rows"].shape[0] > 32
        assert row.raw_contact_fields["candidate_jacobian"].shape[0] > 32
        assert max(row.raw_contact_fields["contact_map"][:2, 0]) >= 32
        _save(tmp_path/"compact-late.jsonl", {"xml": xml, "actual": row, "oracle": oracle})


@pytest.mark.parametrize("mode", MODES)
def test_source_iteration_budget_is_real_and_failure_is_atomic(mode, tmp_path):
    mj = normal._deps()
    xml = mixed.slider_xml(dim=1, limits=True).replace(
        '<geom name="moving" type="sphere" size=".1"/>',
        '<geom name="deep" type="sphere" size=".1" pos="-.12 0 -.008"/>'
        '<geom name="shallow" type="sphere" size=".1" pos=".12 0 .0009"/>')
    xml = xml.replace('iterations="128"', 'iterations="1"')
    model = mj.MjModel.from_xml_string(xml)
    oracle = mixed.native(mj, model, model.qpos0, np.zeros(model.nv), mixed.packet(model))
    assert len(oracle["rows"]) == 2
    assert np.all(oracle["linear"] < 0)
    unrestricted = np.linalg.solve(oracle["matrix"], -oracle["linear"])
    assert np.min(unrestricted) < -.01  # first all-free face must bind a row and be re-solved
    _save(tmp_path/"budget.jsonl", {"xml": xml, "unrestricted_face": unrestricted, "oracle": oracle})
    with MuJoCoTyxRuntime(normal._compile(xml), mode=mode, world_keys=(5, 71)) as runtime:
        before = runtime.snapshot()
        with pytest.raises(PhysicsStepFailure) as caught:
            runtime.step(model.opt.timestep)
        assert caught.value.accepted_worlds == {5: False, 71: False}
        assert caught.value.acceptance_known and not caught.value.physics_accepted
        assert_publication(before, runtime.snapshot())
        _save(tmp_path/"budget.jsonl", {"before": before, "after": runtime.snapshot(),
                                      "error": str(caught.value), "attempt_scratch": attempt_scratch(runtime)})


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("bad", [np.nan, np.inf, 1e50], ids=["nan", "infinity", "fp32-overflow"])
def test_invalid_control_has_no_partial_world_publication(mode, bad, tmp_path):
    mj = normal._deps()
    xml = mixed.slider_xml(dim=6, dry=.2, limits=True)
    model = mj.MjModel.from_xml_string(xml)
    with MuJoCoTyxRuntime(normal._compile(xml), mode=mode, world_keys=(7, 113)) as runtime:
        runtime.step(model.opt.timestep)
        before = runtime.snapshot()
        with pytest.raises((ValueError, MuJoCoTyxUnsupported)):
            bad_control = MuJoCoControl(np.zeros(model.nu),
                qfrc_applied=np.array([bad, 0, 0]), xfrc_applied=np.zeros((model.nbody, 6)))
            runtime.set_controls({7: mixed.packet(model, np.array([.1, 0, 0])), 113: bad_control})
        assert_publication(before, runtime.snapshot())
        _save(tmp_path/"invalid-control.jsonl", {"kind": str(bad), "before": before,
                                                 "after": runtime.snapshot()})


@pytest.mark.parametrize("change", ["elliptic", "rk4", "explicit-pair", "equality"])
def test_native_valid_unsupported_source_is_explicit(change):
    mj = normal._deps()
    xml = mixed.slider_xml()
    if change == "elliptic":
        xml = xml.replace('cone="pyramidal"', 'cone="elliptic"')
    elif change == "rk4":
        xml = xml.replace('integrator="Euler"', 'integrator="RK4"')
    elif change == "explicit-pair":
        xml = xml.replace('</mujoco>', '<contact><pair geom1="floor" geom2="moving" condim="3"/></contact></mujoco>')
    else:
        xml = xml.replace('</mujoco>', '<equality><connect body1="root" anchor="0 0 0"/></equality></mujoco>')
    mj.MjModel.from_xml_string(xml)
    with pytest.raises(MuJoCoTyxUnsupported) as caught:
        MuJoCoTyxRuntime(normal._compile(xml), mode="interpret")
    assert caught.value.code and str(caught.value)


@pytest.mark.parametrize("contact_cap,efc_cap", [(129, 512), (128, 513)])
def test_requested_capacity_above_hard_domain_is_not_silently_clamped(contact_cap, efc_cap):
    normal._deps()
    with pytest.raises((ValueError, MuJoCoTyxUnsupported)):
        MuJoCoTyxRuntime(normal._compile(mixed.slider_xml()), mode="interpret",
                         contact_capacity=contact_cap, constraint_capacity=efc_cap)


def test_potential_pair_capacity_cannot_drop_candidates():
    mj = normal._deps()
    static = "".join(f'<geom type="sphere" size=".01" pos="{i} 0 0"/>' for i in range(64))
    shapes = "".join(f'<geom type="sphere" size=".01" pos="{i} 0 0"/>' for i in range(64))
    body = '<body name="root" pos="0 0 3"><joint type="slide"/>' + mixed._inertial() + shapes + '</body>'
    xml = mixed._xml(body, static=static, gravity="0 0 0")
    model = mj.MjModel.from_xml_string(xml)
    assert model.ngeom == 128  # supported geometry count, but 64*64 potential body-pair combinations
    with pytest.raises(MuJoCoTyxUnsupported) as caught:
        MuJoCoTyxRuntime(normal._compile(xml), mode="interpret")
    assert caught.value.code


def test_geometry_matcher_rejects_missing_duplicate_and_ignores_force():
    from types import SimpleNamespace
    base = dict(native_index=0, geometry_ids=(0, 1), position=np.array([0., 0, 0]),
                frame=np.eye(3), normal_force=1.)
    second = dict(base, native_index=1, position=np.array([1., 0, 0]), normal_force=2.)
    a = SimpleNamespace(geometry_ids=(0, 1), position=np.array([0., 0, 0]), frame=np.eye(3), normal_force=900.)
    b = SimpleNamespace(geometry_ids=(0, 1), position=np.array([1., 0, 0]), frame=np.eye(3), normal_force=-5.)
    assert [r["native_index"] for r in mixed.match_contacts([b, a], [base, second])] == [1, 0]
    duplicate = dict(base, native_index=1)
    assert len(mixed.match_contacts([a, a], [base, duplicate])) == 2
    with pytest.raises(AssertionError):
        mixed.match_contacts([a], [base, duplicate])
    with pytest.raises(AssertionError):
        mixed.match_contacts([a, a], [base, second])


@pytest.mark.parametrize("mode", MODES)
def test_sparse_input_order_does_not_cross_contaminate_world_constraint_rows(mode, tmp_path):
    mj = normal._deps()
    xml = mixed.slider_xml(dim=6, dry=.2, limits=True)
    model = mj.MjModel.from_xml_string(xml)
    compiled = normal._compile(xml)
    initials = varied_initial(model)
    with MuJoCoTyxRuntime(compiled, mode=mode, world_keys=KEYS, initial_states=initials) as forward:
        with MuJoCoTyxRuntime(compiled, mode=mode, world_keys=KEYS[::-1],
                             initial_states=dict(reversed(list(initials.items())))) as reverse:
            for tick in range(3):
                controls = {key: mixed.packet(model, np.array([.03*(i+1), -.01*tick, .02]))
                            for i, key in enumerate(KEYS)}
                forward.set_controls(controls)
                reverse.set_controls(dict(reversed(list(controls.items()))))
                forward.step(model.opt.timestep)
                reverse.step(model.opt.timestep)
                assert_publication(forward.snapshot(), reverse.snapshot())
            _save(tmp_path/"world-order.jsonl", {"forward": forward.snapshot(),
                                                  "reverse": reverse.snapshot()})


@pytest.mark.parametrize("mode", MODES)
def test_actual_efc_over_512_rejects_every_world_without_truncation(mode, tmp_path):
    mj = normal._deps()
    static = "".join(f'<geom name="fixed{i}" type="sphere" size=".1" pos=".199 0 0"/>'
                     for i in range(60))
    joints = "".join(f'<joint type="slide" axis="{axis}"/>' for axis in mixed.XYZ)
    body = '<body name="root">' + joints + mixed._inertial() + '<geom name="moving" type="sphere" size=".1"/></body>'
    xml = mixed._xml(body, dim=6, static=static)
    model = mj.MjModel.from_xml_string(xml)
    cpu = mj.MjData(model)
    mj.mj_forward(model, cpu)
    assert cpu.ncon == 60 and cpu.nefc == 600
    assert model.ngeom == 61
    initials = {7: mixed.initial(model), 113: mixed.initial(model, np.array([3., 0, 0]))}
    with MuJoCoTyxRuntime(normal._compile(xml), mode=mode, world_keys=(7, 113),
                         initial_states=initials, contact_capacity=128,
                         constraint_capacity=512) as runtime:
        before = runtime.snapshot()
        with pytest.raises(PhysicsStepFailure) as caught:
            runtime.step(model.opt.timestep)
        assert caught.value.accepted_worlds == {7: False, 113: False}
        assert caught.value.acceptance_known and not caught.value.physics_accepted
        assert_publication(before, runtime.snapshot())
        _save(tmp_path/"actual-efc-overflow.jsonl", {"xml": xml, "native_contacts": int(cpu.ncon),
                                                    "native_efc": int(cpu.nefc), "before": before,
                                                    "after": runtime.snapshot(), "error": str(caught.value), "attempt_scratch": attempt_scratch(runtime)})


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("field", ["input_q", "input_v"])
def test_finite_readback_input_tamper_rejects_full_batch_and_retries(mode, field, monkeypatch, tmp_path):
    mj = normal._deps()
    xml = mixed.slider_xml(dim=6, dry=.2, limits=True)
    model = mj.MjModel.from_xml_string(xml)
    with MuJoCoTyxRuntime(normal._compile(xml), mode=mode, world_keys=KEYS,
                         initial_states=varied_initial(model)) as runtime:
        runtime.step(model.opt.timestep)
        with runtime.clone() as clone:
            before = runtime.snapshot()
            download = runtime._device.download
            observed = {}
            def tampered_download():
                result = dict(download())
                changed = np.array(result[field], copy=True)
                observed["original"] = changed.copy()
                changed[2, 0] += np.float32(.125)
                result[field] = changed
                observed["changed"] = changed.copy()
                observed["attempt_scratch"] = result
                return result
            monkeypatch.setattr(runtime._device, "download", tampered_download)
            with pytest.raises(PhysicsStepFailure) as caught:
                runtime.step(model.opt.timestep)
            assert np.all(np.isfinite(observed["changed"]))
            assert np.count_nonzero(observed["original"] != observed["changed"]) == 1
            assert caught.value.accepted_worlds == dict.fromkeys(KEYS, False)
            assert caught.value.acceptance_known and not caught.value.physics_accepted
            assert_publication(before, runtime.snapshot())
            _save(tmp_path/"input-tamper.jsonl", {"field": field, "readback": observed,
                                                 "before": before, "after": runtime.snapshot()})
            monkeypatch.setattr(runtime._device, "download", download)
            runtime.step(model.opt.timestep)
            clone.step(model.opt.timestep)
            assert_publication(runtime.snapshot(), clone.snapshot())
            _save(tmp_path/"input-tamper.jsonl", {"retry": runtime.snapshot(),
                                                 "clone": clone.snapshot()})
