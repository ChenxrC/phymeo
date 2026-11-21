"""Sampled withdrawal of an already overlapping declared pair, never a CCD pass."""

import importlib
import inspect
from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest

from phymeo.control.manipulation_push_planner import NativePushRoutePlanner


def _release():
    name = "phymeo.control.manipulation_push_release"
    assert importlib.util.find_spec(name) is not None, "missing independent loaded-release audit"
    return importlib.import_module(name)


def _model(*, obstacle=False, second_object=False):
    wall = '<geom name="obstacle" type="sphere" pos=".38 0 1" size=".02"/>' if obstacle else ""
    extra = (
        '<geom name="object-second" type="box" pos="-.1 0 0" size=".01 .05 .05" mass=".1"/>'
        if second_object
        else ""
    )
    return mj.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian" fusestatic="false"/>'
        "<worldbody>" + wall + '<body name="tool" pos="0 0 1">'
        '<joint name="x" type="slide" axis="1 0 0" limited="true" range="-1 1"/>'
        '<joint name="y" type="slide" axis="0 1 0" limited="true" range="-1 1"/>'
        '<geom name="pusher" type="sphere" size=".02" mass="1"/>'
        '</body><body name="object" pos=".5 0 1"><freejoint/>'
        '<geom name="object-shape" type="box" size=".05 .05 .05" mass=".1"/>'
        + extra
        + "</body></worldbody></mujoco>"
    )


def _knots(model, positions):
    q = np.repeat(model.qpos0[None, :], len(positions), axis=0)
    q[:, 0] = positions
    return tuple(q)


def _binary(model):
    raw = np.empty(mj.mj_sizeModel(model), np.uint8)
    mj.mj_saveModel(model, None, raw)
    return raw.tobytes()


def _audit(
    model, positions=(0.435, 0.410), *, names=("object-shape",), times=(0.0, 0.25), **kwargs
):
    return _release().audit_native_push_release(
        model,
        times,
        _knots(model, positions),
        ("x", "y"),
        pusher_geom_name="pusher",
        object_geom_names=names,
        **kwargs,
    )


def _proposal():
    return SimpleNamespace(
        status="PROPOSAL_ONLY",
        reference_usable=True,
        predicted_improvement=1.0,
        index=0,
        input_binding="a" * 64,
        precontact_center_world=(0.4, 0, 1),
        contact_center_world=(0.43, 0, 1),
        push_center_world=(0.44, 0, 1),
        distance_m=0.01,
        mobility_radius_m=0.08,
        object_com_world=(0.5, 0, 1),
        direction_world=(1.0, 0.0, 0.0),
    )


def _planner(model, **kwargs):
    return NativePushRoutePlanner(
        model,
        ("x", "y"),
        object_geom_names=("object-shape",),
        pusher_geom_name="pusher",
        **kwargs,
    )


def test_loaded_withdrawal_requires_explicit_opt_in_and_keeps_ordinary_failure():
    parameters = inspect.signature(NativePushRoutePlanner).parameters
    assert "allow_loaded_withdrawal" in parameters, "missing opt-in loaded withdrawal route"
    assert parameters["allow_loaded_withdrawal"].default is False
    model = _model()
    q = _knots(model, (0.435,))[0]
    for options in ({}, {"allow_loaded_withdrawal": False}):
        result = _planner(model, **options).plan(q, (_proposal(),), previous_direction=(1, 0, 0))
        assert result.status == "withdraw_rejected" and not result.nodes
        original = result.attempts[0]["paths"][0][1]
        assert original.status == "sampled_collision" and not original.sampled_clear
        assert original.samples[0].contacts[0].allowed_depth_m == 0.001


def test_independent_release_is_per_pair_monotone_and_preserves_all_original_records(monkeypatch):
    model = _model()
    knots = _knots(model, (0.435, 0.410))
    original_q = tuple(q.tobytes() for q in knots)
    original_model = _binary(model)

    def forbidden(*args, **kwargs):
        raise AssertionError("release audit may query geometry only, never advance physics")

    for name in ("mj_forward", "mj_inverse", "mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mj, name, forbidden)
    audit = _release().audit_native_push_release(
        model,
        (0, 0.25),
        knots,
        ("x", "y"),
        pusher_geom_name="pusher",
        object_geom_names=("object-shape",),
    )
    assert audit.status == "sampled_release_ready" and audit.sampled_release
    assert audit.reason == "initial_pairs_only_monotone_and_separated"
    assert audit.original_audit.status == "sampled_collision"
    assert not audit.original_audit.sampled_clear
    assert len(audit.samples) == audit.original_audit.sample_count == 251
    (pair,) = audit.eligible_pairs
    assert set(pair.geometry_names) == {"pusher", "object-shape"}
    assert pair.initial_depth_m == pytest.approx(0.005, abs=1e-12)
    expected = np.maximum(
        0.0, np.array([s.q[0] for s in audit.original_audit.samples]) + 0.02 - 0.45
    )
    actual = np.array([s.pair_depths_m[0] for s in audit.samples])
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    assert np.all(np.diff(actual) <= 0)
    assert audit.samples[-1].pair_distances_m[0] > audit.separation_guard_m
    assert audit.samples[-1].pair_distances_m[0] == pytest.approx(0.02, abs=1e-12)
    assert all(s.outward_step_projection_m[0] >= 0 for s in audit.samples)
    assert sum(s.outward_step_projection_m[0] for s in audit.samples) > 0.024
    assert tuple(q.tobytes() for q in knots) == original_q
    assert _binary(model) == original_model
    for sample in audit.original_audit.samples:
        assert sample.q[2:].tobytes() == knots[0][2:].tobytes()
    with pytest.raises(ValueError):
        audit.original_audit.samples[0].q[0] = 0


@pytest.mark.parametrize(
    "positions,times",
    (
        ((0.435, 0.460), (0.0, 0.25)),  # Further into the same object.
        ((0.435, 0.435), (0.0, 0.25)),  # No actual outward motion.
        ((0.435, 0.430), (0.0, 0.25)),  # Mere touching is not strict release.
        ((0.435, 0.433, 0.434, 0.410), (0.0, 0.004, 0.008, 0.012)),  # Relief then renewed depth.
        (
            (0.435, 0.410, 0.434, 0.410),
            (0.0, 0.004, 0.008, 0.012),
        ),  # Reentry cannot inherit a waiver.
    ),
)
def test_initial_pair_cannot_deepen_stay_touch_or_reenter(positions, times):
    audit = _audit(_model(), positions, times=times)
    assert not audit.sampled_release
    assert audit.status == "release_rejected"
    assert audit.original_audit.status == "sampled_collision"


def test_new_undeclared_collision_cannot_inherit_the_old_pair_exception():
    audit = _audit(_model(obstacle=True))
    assert not audit.sampled_release
    assert len(audit.eligible_pairs) == 1
    assert "obstacle" not in audit.eligible_pairs[0].geometry_names
    assert any(
        "obstacle" in c.geometry_names and not c.sampled_clear
        for s in audit.original_audit.samples
        for c in s.contacts
    )


def test_new_declared_object_pair_cannot_inherit_another_pairs_exception():
    audit = _audit(_model(second_object=True), names=("object-shape", "object-second"))
    assert not audit.sampled_release
    assert len(audit.eligible_pairs) == 1
    assert set(audit.eligible_pairs[0].geometry_names) == {"pusher", "object-shape"}
    assert any(
        "object-second" in c.geometry_names and not c.sampled_clear
        for s in audit.original_audit.samples
        for c in s.contacts
    )


def test_release_is_not_a_new_exception_for_an_initially_ordinary_pair():
    audit = _audit(_model(), (0.4305, 0.410))
    assert not audit.sampled_release and not audit.eligible_pairs
    assert audit.reason == "no_initial_declared_violation"
    assert audit.original_audit.sampled_clear


def test_opted_in_planner_returns_withdrawal_only_then_ordinary_replanning():
    model = _model()
    q = _knots(model, (0.435,))[0]
    before = q.tobytes()
    planner = _planner(model, allow_loaded_withdrawal=True, max_ik_per_segment=1)
    result = planner.plan(q, (_proposal(),), previous_direction=(1, 0, 0))
    assert result.status == "sampled_release_ready" and result.proposal is None
    assert len(result.nodes) == 2 and all(node[1] == "withdraw" for node in result.nodes)
    assert result.nodes[0][0] == 0 and result.nodes[-1][0] == 0.25
    assert result.ik_calls == 1 and result.path_calls == 2
    assert result.attempts[0]["paths"][0][1].status == "sampled_collision"
    assert result.attempts[0]["release_audit"].sampled_release
    assert q.tobytes() == before
    released = q.copy()
    released[:2] = result.nodes[-1][2]
    subsequent = _planner(model).plan(released, (_proposal(),))
    assert subsequent.status == "sampled_reference_ready"
    assert all(node[1] != "withdraw" for node in subsequent.nodes)
    assert all(audit.sampled_clear for _, audit in subsequent.attempts[0]["paths"])


def test_release_query_consumes_the_existing_path_budget():
    model = _model()
    q = _knots(model, (0.435,))[0]
    result = _planner(model, allow_loaded_withdrawal=True, max_paths=1).plan(
        q, (_proposal(),), previous_direction=(1, 0, 0)
    )
    assert result.status == "path_budget" and result.path_calls == 1 and not result.nodes


@pytest.mark.parametrize("flag", (0, 1, None, "yes"))
def test_opt_in_is_an_explicit_boolean(flag):
    assert "allow_loaded_withdrawal" in inspect.signature(NativePushRoutePlanner).parameters
    with pytest.raises(ValueError):
        _planner(_model(), allow_loaded_withdrawal=flag)


def test_invalid_pair_membership_and_grid_bounds_fail_closed():
    model = _model()
    with pytest.raises(ValueError):
        _audit(model, names=("object-shape", "object-shape"))
    with pytest.raises(ValueError):
        _audit(model, max_samples=2)


@pytest.mark.parametrize("declared, depth", ((True, 0.0005), (False, 0.00005)))
def test_new_shallow_physical_contact_is_rejected_even_when_ordinary_allowance_is_clear(
    declared, depth
):
    model = _model(second_object=True)
    second = model.geom("object-second").id
    # Its right face lies at .390 + depth. At x=.410 the .02-radius
    # sphere therefore enters by exactly depth, after starting .025 m farther
    # right. The original .005 m overlap with object-shape still releases.
    model.geom_pos[second, 0] = -0.12 + depth
    names = ("object-shape", "object-second") if declared else ("object-shape",)
    audit = _audit(model, names=names)
    original = audit.original_audit
    assert not any(
        "object-second" in contact.geometry_names for contact in original.samples[0].contacts
    )
    shallow = [
        contact
        for sample in original.samples
        for contact in sample.contacts
        if "object-second" in contact.geometry_names
    ]
    assert shallow and all(contact.sampled_clear for contact in shallow)
    assert all(contact.allowed_depth_m == (0.001 if declared else None) for contact in shallow)
    assert max(-contact.distance_m for contact in shallow) == pytest.approx(depth, abs=1e-12)
    assert all(-contact.distance_m <= (0.001 if declared else 0.0001) for contact in shallow)
    assert len(audit.eligible_pairs) == 1
    assert set(audit.eligible_pairs[0].geometry_names) == {"pusher", "object-shape"}
    assert audit.status == "release_rejected" and not audit.sampled_release
    assert audit.reason == "new_or_other_pair_collision"


@pytest.mark.parametrize("declared, depth", ((True, 0.0005), (False, 0.00005)))
def test_initial_shallow_contact_keeps_its_ordinary_gate_during_loaded_release(declared, depth):
    model = _model(second_object=True)
    second = model.geom("object-second").id
    # This face starts just inside the pusher's RIGHT edge, so the same leftward
    # release leaves both initial pairs instead of creating a new contact.
    model.geom_pos[second, 0] = -0.035 - depth
    names = ("object-shape", "object-second") if declared else ("object-shape",)
    audit = _audit(model, names=names)
    initial = [
        contact
        for contact in audit.original_audit.samples[0].contacts
        if "object-second" in contact.geometry_names
    ]
    assert initial and all(contact.sampled_clear for contact in initial)
    assert max(-contact.distance_m for contact in initial) == pytest.approx(depth, abs=1e-12)
    assert all(
        contact.sampled_clear
        for sample in audit.original_audit.samples
        for contact in sample.contacts
        if "object-second" in contact.geometry_names
    )
    assert len(audit.eligible_pairs) == 1
    assert audit.sampled_release and audit.status == "sampled_release_ready"
    assert audit.original_audit.status == "sampled_collision"
