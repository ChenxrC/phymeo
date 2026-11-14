"""Independent native and hand-vector sampled-path checks; no CCD claim."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from phymeo.control import manipulation_path
from phymeo.control.manipulation_path import audit_native_joint_path


@pytest.fixture
def mj():
    return pytest.importorskip("mujoco")


def model_fixture(mj, *, wall_y="0", wall_mask="1", extra=""):
    return mj.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian" fusestatic="false"/>'
        f'<worldbody><geom name="wall" type="sphere" pos="0 {wall_y} 1" size=".1" '
        f'contype="{wall_mask}" conaffinity="{wall_mask}"/>'
        '<body name="parked" pos="0 2 1" quat=".9238795325 0 0 .3826834324">'
        '<freejoint name="free"/><geom name="parked-shape" type="sphere" size=".05" mass=".1"/>'
        '</body><body name="mover" pos="-.3 0 1">'
        '<joint name="slide" type="slide" axis="1 0 0" limited="true" range="0 .6"/>'
        '<geom name="pad" type="sphere" size=".1" mass=".1"/></body>'
        '<body name="other" pos="0 3 1"><joint name="hinge" type="hinge" '
        'axis="0 0 1" limited="true" range="-1 1"/>'
        '<geom name="other-shape" size=".05" mass=".1"/></body>'
        + extra
        + "</worldbody></mujoco>"
    )


def knots(model, start=0.0, end=0.6):
    result = np.repeat(model.qpos0[None, :], 2, axis=0)
    result[:, 0] = -0.0
    address = int(model.jnt_qposadr[model.joint("slide").id])
    result[:, address] = (start, end)
    return result


def binary(mj, model):
    buffer = np.empty(mj.mj_sizeModel(model), dtype=np.uint8)
    mj.mj_saveModel(model, None, buffer)
    return buffer.tobytes()


def forbidden(*args, **kwargs):
    raise AssertionError("path query must not forward/inverse-solve or integrate physics")


def test_interior_collision_is_found_with_all_samples_and_independent_native_contacts(mj, monkeypatch):
    model = model_fixture(mj)
    q = knots(model)
    source, original_model = q.tobytes(), binary(mj, model)
    expected_q = (0.0, 0.09375, 0.3, 0.50625, 0.6)
    oracle = []
    for value in expected_q:
        data = mj.MjData(model)
        data.qpos[:] = q[0]
        data.qpos[7] = value
        mj.mj_forward(model, data)
        oracle.append(
            [
                (tuple(map(int, item.geom)), float(item.dist), item.pos.copy())
                for item in data.contact[: data.ncon]
            ]
        )
    original_collision, calls = mj.mj_collision, []

    def collision(*args):
        calls.append(1)
        return original_collision(*args)

    with monkeypatch.context() as guard:
        for name in ("mj_forward", "mj_inverse", "mj_step", "mj_step1", "mj_step2"):
            guard.setattr(mj, name, forbidden)
        guard.setattr(mj, "mj_collision", collision)
        result = audit_native_joint_path(model, (0.0, 0.004), q, ("slide",))
    assert result.status == "sampled_collision" and not result.sampled_clear
    assert result.sample_count == len(calls) == len(result.samples) == 5
    assert result.samples[0].sampled_clear and result.samples[-1].sampled_clear
    assert not result.samples[2].sampled_clear
    assert [sample.time_s for sample in result.samples] == pytest.approx((0, .001, .002, .003, .004))
    assert [sample.q[7] for sample in result.samples] == pytest.approx(expected_q)
    for sample, expected in zip(result.samples, oracle, strict=True):
        assert sample.q[:7].tobytes() == q[0, :7].tobytes()
        assert sample.q[8:].tobytes() == q[0, 8:].tobytes()
        assert len(sample.contacts) == len(expected)
        for contact, (ids, distance, position) in zip(sample.contacts, expected, strict=True):
            assert contact.geometry_ids == ids
            assert set(contact.geometry_names) == {"wall", "pad"}
            assert contact.distance_m == pytest.approx(distance)
            np.testing.assert_allclose(contact.position_world, position, atol=1e-12, rtol=0)
    assert result.worst_disallowed_depth_m == pytest.approx(0.2 - 1e-4)
    assert q.tobytes() == source and binary(mj, model) == original_model
    coarse = audit_native_joint_path(model, (0.0, 0.004), q, ("slide",), max_time_step=0.004)
    assert coarse.sample_count == 2 and coarse.status == "sampled_clear"


def test_segment_endpoints_are_exact_and_not_duplicated(mj):
    model = model_fixture(mj, wall_y="1")
    q = knots(model)
    middle = q[0].copy()
    middle[7] = 0.3
    result = audit_native_joint_path(model, (0, .002, .005), (q[0], middle, q[1]), ("slide",))
    assert result.sampled_clear and result.sample_count == 6
    assert [s.time_s for s in result.samples].count(.002) == 1
    assert result.samples[0].q.tobytes() == q[0].tobytes()
    assert result.samples[2].q.tobytes() == middle.tobytes()
    assert result.samples[-1].q.tobytes() == q[1].tobytes()


def test_allowed_pair_is_still_recorded_and_excess_remains_unsafe(mj):
    model = model_fixture(mj)
    q = knots(model, .11, .11)
    safe = audit_native_joint_path(
        model, (0, .002), q, ("slide",), allowed_contacts=(("pad", "wall", .02),)
    )
    assert safe.sampled_clear and safe.worst_disallowed_depth_m == 0
    assert all(len(sample.contacts) == 1 for sample in safe.samples)
    assert all(sample.contacts[0].allowed_depth_m == .02 for sample in safe.samples)
    assert safe.samples[0].contacts[0].distance_m == pytest.approx(-.01)
    unsafe = audit_native_joint_path(
        model, (0, .002), q, ("slide",), allowed_contacts=(("wall", "pad", .005),)
    )
    assert not unsafe.sampled_clear
    assert unsafe.worst_disallowed_depth_m == pytest.approx(.005)
    assert unsafe.input_binding != safe.input_binding


def test_native_contact_masks_and_model_binding_use_original_recompiled_model(mj):
    first = model_fixture(mj)
    disabled = model_fixture(mj, wall_mask="0")
    moved = model_fixture(mj, wall_y="1")
    original = audit_native_joint_path(first, (0, .002), knots(first), ("slide",))
    for model in (disabled, moved):
        receipt = audit_native_joint_path(model, (0, .002), knots(model), ("slide",))
        assert receipt.sampled_clear and all(not sample.contacts for sample in receipt.samples)
        assert receipt.input_binding != original.input_binding


def test_joint_order_maps_native_addresses_and_all_inputs_are_bound(mj):
    model = model_fixture(mj, wall_y="1")
    q = knots(model)
    first = audit_native_joint_path(model, (0, .002), q, ("hinge", "slide"))
    second = audit_native_joint_path(model, (0, .002), q, ("slide", "hinge"))
    assert first.input_binding != second.input_binding
    for left, right in zip(first.samples, second.samples, strict=True):
        np.testing.assert_array_equal(left.q, right.q)
    changed = audit_native_joint_path(model, (0, .002), q, ("hinge", "slide"), max_samples=100)
    assert changed.input_binding != first.input_binding
    allowed = audit_native_joint_path(
        model, (0, .002), q, ("slide",), allowed_contacts=(("wall", "pad", .05),)
    )
    reversed_pair = audit_native_joint_path(
        model, (0, .002), q, ("slide",), allowed_contacts=(("pad", "wall", .05),)
    )
    assert allowed.input_binding == reversed_pair.input_binding


@pytest.mark.parametrize(
    "options",
    [
        {"times": (0, 0)},
        {"times": (.001, .002)},
        {"times": (0, 4.001)},
        {"times": (0, np.nan)},
        {"times": (0,)},
        {"times": tuple(range(33))},
        {"joint_names": ("missing",)},
        {"joint_names": ("slide", "slide")},
        {"joint_names": ("free",)},
        {"max_time_step": 0},
        {"max_time_step": True},
        {"max_time_step": 1e-300},
        {"max_samples": 1},
        {"max_samples": 2},
        {"max_samples": 4098},
        {"max_samples": True},
        {"penetration_tolerance_m": -1e-4},
        {"penetration_tolerance_m": .000101},
        {"allowed_contacts": (("wall", "pad", .051),)},
        {"allowed_contacts": (("wall", "pad", -.001),)},
        {"allowed_contacts": (("wall", "unknown", .01),)},
        {"allowed_contacts": (("wall", "wall", .01),)},
        {"allowed_contacts": (("wall", "pad", .01), ("pad", "wall", .02))},
    ],
)
def test_invalid_requests_fail_before_native_path_queries(mj, monkeypatch, options):
    model = model_fixture(mj)
    request = dict(times=(0, .004), q_knots=knots(model), joint_names=("slide",))
    request.update(options)
    monkeypatch.setattr(mj, "mj_kinematics", forbidden)
    monkeypatch.setattr(mj, "mj_collision", forbidden)
    with pytest.raises(ValueError):
        audit_native_joint_path(model, **request)


@pytest.mark.parametrize("change", ["limit", "nan", "free", "signed-zero", "quaternion", "shape"])
def test_knot_limits_all_frozen_coordinates_and_quaternions_are_prevalidated(mj, monkeypatch, change):
    model = model_fixture(mj)
    q = knots(model)
    if change == "limit":
        q[1, 7] = .601
    elif change == "nan":
        q[1, 7] = np.nan
    elif change == "free":
        q[1, 1] += .001
    elif change == "signed-zero":
        q[1, 0] = 0.0
    elif change == "quaternion":
        q[:, 3:7] = 0
    else:
        q = q[:, :-1]
    before = q.tobytes()
    monkeypatch.setattr(mj, "mj_kinematics", forbidden)
    monkeypatch.setattr(mj, "mj_collision", forbidden)
    with pytest.raises(ValueError):
        audit_native_joint_path(model, (0, .004), q, ("slide",))
    assert q.tobytes() == before


def test_model_dimension_and_binary_caps_precede_queries(mj, monkeypatch):
    extras = "".join(
        f'<geom name="extra{i}" type="sphere" pos="{i} 10 1" size=".001"/>'
        for i in range(129)
    )
    too_many = model_fixture(mj, extra=extras)
    with pytest.raises(ValueError, match="dimensions"):
        audit_native_joint_path(too_many, (0, .002), knots(too_many), ("slide",))
    ordinary = model_fixture(mj)
    monkeypatch.setattr(mj, "mj_sizeModel", lambda model: 8 * 1024 * 1024 + 1)
    monkeypatch.setattr(mj, "mj_kinematics", forbidden)
    with pytest.raises(ValueError, match="8 MiB"):
        audit_native_joint_path(ordinary, (0, .002), knots(ordinary), ("slide",))


def test_outputs_are_owned_immutable_records(mj):
    model = model_fixture(mj)
    q = knots(model, .11, .11)
    result = audit_native_joint_path(model, (0, .002), q, ("slide",))
    sample, contact = result.samples[0], result.samples[0].contacts[0]
    q[:] = 0
    assert sample.q[7] == .11
    with pytest.raises(FrozenInstanceError):
        result.status = "sampled_clear"
    with pytest.raises(ValueError):
        sample.q[7] = .2
    with pytest.raises(ValueError):
        sample.q.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        contact.distance_m = 0
    with pytest.raises(TypeError):
        contact.position_world[0] = 0


def test_total_contact_record_cap_includes_allowed_contacts_across_all_samples(mj, monkeypatch):
    model = model_fixture(mj)
    q = knots(model, .11, .11)
    before = q.tobytes()
    # Exercise the cumulative budget boundary without collecting 65537 contacts.
    assert manipulation_path.MAX_CONTACT_RECORDS == 65536
    monkeypatch.setattr(manipulation_path, "MAX_CONTACT_RECORDS", 1)
    with pytest.raises(ValueError, match="total contact record budget"):
        audit_native_joint_path(
            model, (0, .002), q, ("slide",), allowed_contacts=(("wall", "pad", .02),)
        )
    assert q.tobytes() == before
