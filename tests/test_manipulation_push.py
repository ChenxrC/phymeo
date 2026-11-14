"""Independent native and hand-geometry oracles; no push-success claims."""

from dataclasses import FrozenInstanceError
from math import cos, pi, sin, sqrt

import numpy as np
import pytest

from phymeo.control.manipulation_push import enumerate_native_planar_push_contacts


@pytest.fixture
def mj():
    return pytest.importorskip("mujoco")


T_BOXES = (
    ("stem", 0, -.025, .020, .045, .08),
    ("bar", 0, .038, .060, .018, .08),
)


def model_fixture(
    mj, boxes=T_BOXES, *, object_pos=(.5, 0, .016), yaw=0, tilt=0,
    second_yaw=0, second_z=0, pusher_kind="capsule", pusher_z=.06,
    pusher_tilt=0, extra_object="", child_object="",
):
    shapes = "".join(
        f'<geom name="{name}" type="box" size="{hx} {hy} .015" '
        f'pos="{x} {y} {second_z if index else 0}" mass="{mass}" '
        f'euler="0 0 {second_yaw if index else 0}"/>'
        for index, (name, x, y, hx, hy, mass) in enumerate(boxes)
    )
    pusher_size = ".018 .035" if pusher_kind == "capsule" else ".018"
    position = " ".join(map(str, object_pos))
    return mj.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian" fusestatic="false"/>'
        '<option gravity="0 0 0"/>'
        '<worldbody>'
        f'<body name="object" pos="{position}" euler="0 {tilt} {yaw}">'
        '<freejoint name="object-free"/>'
        + shapes + extra_object + child_object
        + '</body>'
        f'<body name="pusher-body" pos="-.2 -.2 {pusher_z}">'
        '<joint name="push-x" type="slide" axis="1 0 0"/>'
        '<joint name="push-y" type="slide" axis="0 1 0"/>'
        '<joint name="push-z" type="slide" axis="0 0 1"/>'
        f'<geom name="pusher" type="{pusher_kind}" size="{pusher_size}" '
        f'euler="{pusher_tilt} 0 0" mass=".05"/>'
        '</body></worldbody></mujoco>'
    )


def query(model, **kwargs):
    options = dict(object_geom_names=("stem", "bar"), pusher_geom_name="pusher")
    options.update(kwargs)
    return enumerate_native_planar_push_contacts(model, model.qpos0.copy(), **options)


def binary(mj, model):
    data = np.empty(mj.mj_sizeModel(model), dtype=np.uint8)
    mj.mj_saveModel(model, None, data)
    return data.tobytes()


def native(mj, model, q):
    data = mj.MjData(model)
    data.qpos[:] = q
    mj.mj_forward(model, data)
    return data


def native_at_pusher(mj, model, center):
    q = model.qpos0.copy()
    original = native(mj, model, q).geom_xpos[model.geom("pusher").id]
    for axis, name in enumerate(("push-x", "push-y", "push-z")):
        address = model.jnt_qposadr[model.joint(name).id]
        q[address] += center[axis] - original[axis]
    return native(mj, model, q)


def object_contacts(model, data):
    pusher = model.geom("pusher").id
    body = model.body("object").id
    return tuple(
        contact
        for contact in data.contact
        if (int(contact.geom1) == pusher and model.geom_bodyid[contact.geom2] == body)
        or (int(contact.geom2) == pusher and model.geom_bodyid[contact.geom1] == body)
    )


def left_candidate(result, owner, fraction=.5, angle=0):
    return next(
        candidate for candidate in result.candidates
        if candidate.owner_geometry_ids == (owner,)
        and candidate.outward_normal_world == (-1., 0., 0.)
        and candidate.edge_fraction == fraction
        and candidate.approach_angle_rad == angle
    )


def forbidden(*args, **kwargs):
    raise AssertionError("candidate generation must use only native kinematics")


def perimeter(result):
    return sum(
        np.linalg.norm(np.subtract(edge.end_world, edge.start_world))
        for edge in result.boundary
    )


@pytest.mark.parametrize(
    ("boxes", "edge_count", "length"),
    (
        ((("only", 0, 0, .02, .04, .1),), 4, .24),
        (T_BOXES, 8, .492),
        ((("a", 0, 0, .02, .04, .1), ("b", .04, -.02, .02, .02, .1)), 7, .32),
        ((("a", 0, 0, .02, .04, .1), ("b", 0, 0, .02, .04, .2)), 4, .24),
        ((("a", 0, 0, .02, .04, .1), ("b", .02, 0, .02, .04, .1)), 8, .28),
        ((("a", 0, 0, .04, .05, .1), ("b", 0, 0, .01, .02, .1)), 4, .36),
    ),
)
def test_full_union_exterior_and_owner_provenance(mj, boxes, edge_count, length):
    model = model_fixture(mj, boxes)
    result = query(model, object_geom_names=tuple(box[0] for box in boxes), edge_fractions=(.5,))
    assert len(result.boundary) == edge_count
    assert perimeter(result) == pytest.approx(length, abs=1e-12)
    assert len(result.candidates) == edge_count
    assert len({(e.start_world, e.end_world, e.outward_normal_world) for e in result.boundary}) == edge_count
    assert all(e.owner_geometry_ids for e in result.boundary)
    if len(boxes) == 2 and boxes[0][1:5] == boxes[1][1:5]:
        owners = tuple(model.geom(box[0]).id for box in boxes)
        assert all(edge.owner_geometry_ids == owners for edge in result.boundary)
    if boxes == T_BOXES:
        # The shared stem/bar seam is absent; exposed underside wings remain.
        underside = [e for e in result.boundary if e.outward_normal_world == (0., -1., 0.)]
        assert len(underside) == 3
        assert not any(
            e.start_world[1] == pytest.approx(.02)
            and e.start_world[0] < .5 < e.end_world[0]
            for e in underside
        )


def test_native_com_and_normal_moment_are_not_area_centroid_or_body_origin(mj):
    model = model_fixture(mj)
    result = query(model, edge_fractions=(.5,), approach_angles=(0., .3))
    data = native(mj, model, model.qpos0)
    np.testing.assert_allclose(result.object_com_world, data.xipos[model.body("object").id], atol=1e-14, rtol=0)
    np.testing.assert_allclose(result.object_com_world, (.5, .0065, .016), atol=1e-14, rtol=0)
    area_com_y = (-.025 * (.04 * .09) + .038 * (.12 * .036)) / (.04 * .09 + .12 * .036)
    assert abs(area_com_y - result.object_com_world[1]) > .002
    stem = left_candidate(result, model.geom("stem").id)
    bar = left_candidate(result, model.geom("bar").id)
    assert stem.usable and bar.usable
    assert stem.normal_wrench_com_world[5] == pytest.approx(.0315, abs=1e-14)
    assert bar.normal_wrench_com_world[5] == pytest.approx(-.0315, abs=1e-14)
    assert stem.normal_wrench_com_world[4] == pytest.approx(.012, abs=1e-14)
    angled = left_candidate(result, model.geom("stem").id, angle=.3)
    assert angled.normal_wrench_com_world == stem.normal_wrench_com_world
    assert angled.approach_direction_world != stem.approach_direction_world
    for candidate in result.candidates:
        point, center, pre = map(np.asarray, (
            candidate.point_world, candidate.contact_center_world, candidate.precontact_center_world,
        ))
        direction, normal = map(np.asarray, (
            candidate.approach_direction_world, candidate.outward_normal_world,
        ))
        assert normal @ direction < 0
        np.testing.assert_allclose((center - point)[:2], .018 * normal[:2], atol=1e-14, rtol=0)
        np.testing.assert_allclose(center - pre, .01 * direction, atol=1e-14, rtol=0)


def test_concave_corner_radius_blocks_com_line_but_keeps_all_attempts(mj):
    model = model_fixture(mj)
    result = query(model, edge_fractions=(.5, .85))
    stem = model.geom("stem").id
    clear = left_candidate(result, stem)
    blocked = left_candidate(result, stem, fraction=.85)
    assert clear.usable and clear.reason == "clear"
    assert blocked.point_world[1] == pytest.approx(.0065, abs=1e-14)
    assert not blocked.usable and blocked.reason == "contact-occluded"
    bar_index = next(i for i, rect in enumerate(result.rectangles) if rect.geometry_name == "bar")
    assert blocked.contact_distances_m[bar_index] == pytest.approx(.0135, abs=1e-14)
    assert len(result.candidates) == 16
    assert [candidate.index for candidate in result.candidates] == list(range(16))


def test_segment_interior_detects_corner_obstruction_with_both_endpoints_clear(mj):
    model = model_fixture(mj)
    result = query(model, edge_fractions=(.5,), approach_angles=(-pi / 3,), stand_off_m=.12)
    blocked = left_candidate(result, model.geom("stem").id, angle=-pi / 3)
    assert not blocked.usable and blocked.reason == "entry-occluded"
    bar_index = next(i for i, rect in enumerate(result.rectangles) if rect.geometry_name == "bar")
    # Entire-segment minimum is the lower-left bar corner's perpendicular distance.
    hand_distance = abs(.022 * sqrt(3) / 2 - .045 / 2)
    assert blocked.entry_distances_m[bar_index] == pytest.approx(hand_distance, abs=1e-14)
    assert blocked.contact_distances_m[bar_index] > .018
    for center in (blocked.precontact_center_world, blocked.contact_center_world):
        assert not any(
            contact.dist < -1e-9 for contact in object_contacts(model, native_at_pusher(mj, model, center))
        )
    middle = np.array((.5 - .038 - .045 / sqrt(3), .02, .06))
    contacts = object_contacts(model, native_at_pusher(mj, model, middle))
    assert any(contact.dist < -.005 for contact in contacts)


@pytest.mark.parametrize("kind", ("capsule", "sphere"))
def test_native_fixed_geometry_oracle_and_no_physics_calls(mj, kind, monkeypatch):
    model = model_fixture(mj, pusher_kind=kind, pusher_z=.016 if kind == "sphere" else .06)
    q = model.qpos0.copy()
    q.setflags(write=False)
    before_q, before_model = q.tobytes(), binary(mj, model)
    with monkeypatch.context() as guard:
        for name in ("mj_forward", "mj_inverse", "mj_step", "mj_step1", "mj_step2", "mj_collision"):
            guard.setattr(mj, name, forbidden)
        result = enumerate_native_planar_push_contacts(
            model, q, object_geom_names=("stem", "bar"), pusher_geom_name="pusher", edge_fractions=(.5,),
        )
    assert q.tobytes() == before_q and binary(mj, model) == before_model
    clear = left_candidate(result, model.geom("stem").id)
    assert clear.usable
    assert not object_contacts(model, native_at_pusher(mj, model, clear.precontact_center_world))
    # Query a separately declared 10 um inward offset to avoid a tangent/no-contact ambiguity.
    inward = np.asarray(clear.contact_center_world) - 1e-5 * np.asarray(clear.outward_normal_world)
    contacts = object_contacts(model, native_at_pusher(mj, model, inward))
    assert contacts
    assert min(contact.dist for contact in contacts) == pytest.approx(-1e-5, abs=2e-9)
    assert binary(mj, model) == before_model


def test_world_translation_yaw_and_canonical_name_order(mj):
    original = model_fixture(mj)
    transformed = model_fixture(mj, object_pos=(.3, -.4, .016), yaw=.7)
    first, second = query(original), query(transformed)
    reversed_names = query(original, object_geom_names=("bar", "stem"))
    assert first.input_binding == reversed_names.input_binding
    assert first.candidates == reversed_names.candidates
    assert first.input_binding != second.input_binding
    rotation = np.array(((cos(.7), -sin(.7), 0), (sin(.7), cos(.7), 0), (0, 0, 1)))
    old, new = np.array((.5, 0, .016)), np.array((.3, -.4, .016))
    np.testing.assert_allclose(second.object_com_world, rotation @ (np.asarray(first.object_com_world) - old) + new, atol=1e-14, rtol=0)
    for a, b in zip(first.candidates, second.candidates, strict=True):
        assert (a.edge_index, a.owner_geometry_ids, a.reason) == (b.edge_index, b.owner_geometry_ids, b.reason)
        for name in ("point_world", "contact_center_world", "precontact_center_world"):
            np.testing.assert_allclose(getattr(b, name), rotation @ (np.asarray(getattr(a, name)) - old) + new, atol=1e-14, rtol=0)
        for name in ("outward_normal_world", "approach_direction_world"):
            np.testing.assert_allclose(getattr(b, name), rotation @ getattr(a, name), atol=1e-14, rtol=0)
        for offset in (0, 3):
            np.testing.assert_allclose(b.normal_wrench_com_world[offset:offset + 3], rotation @ a.normal_wrench_com_world[offset:offset + 3], atol=1e-14, rtol=0)
        np.testing.assert_allclose(b.entry_distances_m, a.entry_distances_m, atol=1e-14, rtol=0)


@pytest.mark.parametrize(("world", "stem_width", "bar_width"), ((5, .018, .065), (6, .022, .075), (7, .014, .055)))
def test_original_three_t_shapes_keep_clear_and_shadowed_candidates(mj, world, stem_width, bar_width):
    from phymeo.training.manipulation_tasks import manipulation_scenes

    scene = manipulation_scenes()[world]
    model = mj.MjModel.from_xml_string(scene.mjcf)
    before = binary(mj, model)
    result = query(model, object_geom_names=("object-0", "object-1"), edge_fractions=(.5, .85))
    np.testing.assert_allclose(model.geom_size[model.geom("object-0").id], (stem_width, .045, .015), rtol=0, atol=1e-14)
    np.testing.assert_allclose(model.geom_size[model.geom("object-1").id], (bar_width, .018, .015), rtol=0, atol=1e-14)
    assert result.pusher_radius_m == .018
    assert len(result.boundary) == 8 and len(result.candidates) == 16
    np.testing.assert_allclose(result.object_com_world, (.52, .0065, .016), rtol=0, atol=1e-14)
    owner = model.geom("object-0").id
    assert left_candidate(result, owner).usable
    assert left_candidate(result, owner, fraction=.85).reason == "contact-occluded"
    assert binary(mj, model) == before
    # Geometric queries do not assert reachability, yaw response, or the original Goal.


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (dict(pusher_z=.2), "no Z volume overlap"),
        (dict(pusher_z=.07), "cap-only"),
        (dict(pusher_tilt=.1), "not vertical"),
        (dict(tilt=.1), "not planar"),
        (dict(second_yaw=.1), "not co-aligned"),
        (dict(second_z=.01), "Z interval"),
        (dict(extra_object='<geom name="extra" size=".01" mass=".01"/>'), "complete native body"),
        (dict(child_object='<body name="child"><geom name="child-shape" size=".01" mass=".01"/></body>'), "descendant"),
    ),
)
def test_unsupported_geometry_is_explicit(mj, kwargs, message):
    with pytest.raises(ValueError, match=message):
        query(model_fixture(mj, **kwargs))


@pytest.mark.parametrize(
    "kwargs",
    (
        dict(edge_fractions=(0.,)), dict(edge_fractions=(1.,)), dict(edge_fractions=(.5, .5)),
        dict(edge_fractions=(.1,) * 9), dict(edge_fractions=(float("nan"),)),
        dict(edge_fractions=(True,)), dict(edge_fractions=iter((.5,))),
        dict(approach_angles=(pi / 2,)), dict(approach_angles=(-pi / 2,)),
        dict(approach_angles=(0.,) * 6), dict(approach_angles=(float("inf"),)),
        dict(stand_off_m=0), dict(stand_off_m=-.1), dict(stand_off_m=True),
        dict(stand_off_m=float("inf")), dict(max_candidates=True), dict(max_candidates=0),
        dict(max_candidates=129), dict(object_geom_names=("stem",)),
        dict(object_geom_names=("stem", "stem")), dict(object_geom_names=("missing", "bar")),
        dict(pusher_geom_name="missing"), dict(pusher_geom_name="stem"),
    ),
)
def test_bad_declarations_reject_without_modifying_model(mj, kwargs):
    model = model_fixture(mj)
    before = binary(mj, model)
    with pytest.raises(ValueError):
        query(model, **kwargs)
    assert binary(mj, model) == before


@pytest.mark.parametrize("bad_q", ("short", "nan", "quaternion", "bool"))
def test_invalid_configuration_rejects(mj, bad_q):
    model = model_fixture(mj)
    q = model.qpos0.copy()
    if bad_q == "short":
        q = q[:-1]
    elif bad_q == "nan":
        q[0] = np.nan
    elif bad_q == "quaternion":
        q[3:7] = 0
    else:
        q = q.astype(np.bool_)
    before = q.tobytes()
    with pytest.raises(ValueError):
        enumerate_native_planar_push_contacts(model, q, object_geom_names=("stem", "bar"), pusher_geom_name="pusher")
    assert q.tobytes() == before


def test_candidate_budget_rejects_full_product_before_distance_queries(mj, monkeypatch):
    import phymeo.control.manipulation_push as module

    model = model_fixture(mj)
    monkeypatch.setattr(module, "_point_rectangle", forbidden)
    with pytest.raises(ValueError, match="candidate budget"):
        query(model, edge_fractions=tuple(i / 9 for i in range(1, 9)), approach_angles=(-.2, 0, .2))
    with pytest.raises(ValueError, match="candidate budget"):
        query(model, max_candidates=23)


def test_exact_candidate_cap_keeps_declared_order_and_every_attempt(mj):
    model = model_fixture(mj)
    fractions, angles = tuple(i / 9 for i in range(1, 9)), (-.2, .2)
    result = query(model, edge_fractions=fractions, approach_angles=angles)
    assert len(result.candidates) == 128
    assert [(c.edge_index, c.edge_fraction, c.approach_angle_rad) for c in result.candidates] == [
        (edge.index, fraction, angle)
        for edge in result.boundary for fraction in fractions for angle in angles
    ]
    assert any(c.usable for c in result.candidates)
    assert any(not c.usable for c in result.candidates)


def test_model_byte_budget_is_checked_before_scratch(mj, monkeypatch):
    model = model_fixture(mj)
    monkeypatch.setattr(mj, "mj_sizeModel", lambda model: 8 * 1024 * 1024 + 1)
    monkeypatch.setattr(mj, "MjData", forbidden)
    with pytest.raises(ValueError, match="8 MiB"):
        query(model)


def test_immutable_result_and_complete_binding(mj):
    model = model_fixture(mj)
    first = query(model)
    assert first.query_tolerance_m == 1e-9
    assert len(first.input_binding) == 64
    for array in (first.q, first.plane_axes_world):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        first.candidates[0].reason = "clear"
    with pytest.raises(TypeError):
        first.candidates[0].point_world[0] = 0
    for kwargs in (
        dict(stand_off_m=.02), dict(edge_fractions=(.5,)), dict(approach_angles=(.1,)),
        dict(max_candidates=127),
    ):
        assert query(model, **kwargs).input_binding != first.input_binding
    changed_mass = model_fixture(mj, (T_BOXES[0], (*T_BOXES[1][:-1], .16)))
    assert query(changed_mass).input_binding != first.input_binding
    q = model.qpos0.copy()
    q[model.jnt_qposadr[model.joint("push-x").id]] += .01
    moved = enumerate_native_planar_push_contacts(model, q, object_geom_names=("stem", "bar"), pusher_geom_name="pusher")
    assert moved.input_binding != first.input_binding
    assert moved.candidates == first.candidates
