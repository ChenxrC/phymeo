"""Hand-declared geometry and rational proposal oracles; no native producer or simulation."""

import builtins
from dataclasses import FrozenInstanceError, replace
from math import pi

import numpy as np
import pytest

from phymeo.control.manipulation_push import (
    NativePlanarPushContactCandidate,
    NativePlanarPushContacts,
    PlanarPushBoundaryEdge,
    PlanarPushRectangle,
)
from phymeo.control.manipulation_push_objective import (
    NativePlanarPushProposal,
    propose_planar_pushes,
)

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
QUARTER = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
HALF = ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))


def readonly(values):
    array = np.asarray(values, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def query_fixture(origin=(0.0, 0.0, 0.0)):
    """A 60x80 mm rectangle has corner radius 50 mm about its declared COM."""

    def point(x, y):
        return (origin[0] + x, origin[1] + y, origin[2])

    edges = (
        PlanarPushBoundaryEdge(0, point(-0.03, -0.04), point(-0.03, 0.04), (-1.0, 0.0, 0.0), (1,)),
        PlanarPushBoundaryEdge(1, point(-0.03, -0.04), point(0.03, -0.04), (0.0, -1.0, 0.0), (1,)),
        PlanarPushBoundaryEdge(2, point(0.03, -0.04), point(0.03, 0.04), (1.0, 0.0, 0.0), (1,)),
        PlanarPushBoundaryEdge(3, point(-0.03, 0.04), point(0.03, 0.04), (0.0, 1.0, 0.0), (1,)),
    )
    candidates = (
        NativePlanarPushContactCandidate(
            0,
            0,
            0.75,
            0.0,
            (1,),
            point(-0.03, 0.02),
            (-1.0, 0.0, 0.0),
            point(-0.04, 0.02),
            point(-0.052, 0.02),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0, 0.0, -0.02),
            (0.01,),
            (0.022,),
            True,
            "clear",
        ),
        NativePlanarPushContactCandidate(
            1,
            1,
            5 / 6,
            0.0,
            (1,),
            point(0.02, -0.04),
            (0.0, -1.0, 0.0),
            point(0.02, -0.05),
            point(0.02, -0.062),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0, 0.0, 0.0, 0.02),
            (0.01,),
            (0.022,),
            True,
            "clear",
        ),
    )
    return NativePlanarPushContacts(
        "a" * 64,
        readonly((*origin, 1.0, 0.0, 0.0, 0.0)),
        1,
        origin,
        2,
        0.01,
        (0.0, 0.0, 0.0),
        readonly(IDENTITY),
        (
            PlanarPushRectangle(
                1,
                "box",
                (origin[0] - 0.03, origin[1] - 0.04),
                (origin[0] + 0.03, origin[1] + 0.04),
                (origin[2] - 0.01, origin[2] + 0.01),
            ),
        ),
        edges,
        candidates,
        1e-9,
    )


def propose(query=None, **kwargs):
    options = dict(
        reference_position=(0.0, 0.0, 0.0),
        reference_rotation=IDENTITY,
        current_position=(0.0, 0.0, 0.0),
        current_rotation=IDENTITY,
        target_xy=(0.01, 0.0),
        target_yaw=0.1,
        pusher_height_m=0.06,
        position_tolerance_m=0.01,
        yaw_tolerance_rad=0.02,
        push_distances_m=(0.01,),
        stand_off_m=0.012,
    )
    return propose_planar_pushes(query_fixture() if query is None else query, **(options | kwargs))


def test_material_point_and_normal_transport_use_both_full_body_rotations():
    query = query_fixture((1.0, 2.0, 3.0))
    query = replace(
        query,
        object_com_world=(1.0, 2.01, 3.0),
        candidates=(
            replace(query.candidates[0], normal_wrench_com_world=(1.0, 0.0, 0.0, 0.0, 0.0, -0.01)),
            query.candidates[1],
        ),
    )
    records = propose(
        query,
        reference_position=(1.0, 2.0, 3.0),
        reference_rotation=QUARTER,
        current_position=(4.0, 5.0, 6.0),
        current_rotation=HALF,
        target_xy=(4.0, 5.01),
        target_yaw=pi - 0.1,
        pusher_height_m=0.2,
    )
    first = next(record for record in records if record.index == 0)
    assert isinstance(first, NativePlanarPushProposal) and first.status == "PROPOSAL_ONLY"
    assert first.reference_usable and (first.distance_ordinal, first.distance_m) == (0, 0.01)
    # Offset COM gives r=sqrt(.03**2+.05**2), Mz=-.01, yaw delta=-1/35.
    assert first.mobility_radius_m == pytest.approx(0.058309518948453, abs=1e-14)
    np.testing.assert_allclose(first.point_world, (3.98, 4.97, 6.0), rtol=0, atol=1e-14)
    np.testing.assert_allclose(first.outward_normal_world, (0.0, -1.0, 0.0), rtol=0, atol=1e-14)
    np.testing.assert_allclose(first.object_com_world, (3.99, 5.0, 6.0), rtol=0, atol=1e-14)
    np.testing.assert_allclose(first.contact_center_world, (3.98, 4.96, 0.2), rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        first.precontact_center_world, (3.98, 4.948, 0.2), rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(first.push_center_world, (3.98, 4.97, 0.2), rtol=0, atol=1e-14)
    assert first.direction_world == (0.0, 1.0, 0.0)
    assert first.nominal_yaw_moment == pytest.approx(-0.01, abs=1e-14)
    np.testing.assert_allclose(first.predicted_translation_xy, (0.0, 17 / 1750), rtol=0, atol=1e-14)
    assert first.predicted_yaw_change == pytest.approx(-1 / 35, abs=1e-14)
    np.testing.assert_allclose(
        first.predicted_position_xy, (4.0, 5 + 17 / 1750), rtol=0, atol=1e-14
    )
    assert first.predicted_yaw == pytest.approx(pi - 1 / 35, abs=1e-14)


def test_pose_cost_reverses_the_translation_first_wrong_yaw_choice():
    records = propose()
    assert tuple(record.index for record in records) == (1, 0)
    right, wrong = records
    # A translation-first lexicographic key prefers candidate 0: alignments 1 vs 0.
    assert wrong.direction_world == (1.0, 0.0, 0.0) and right.direction_world == (0.0, 1.0, 0.0)
    assert wrong.nominal_yaw_moment == -0.02 and right.nominal_yaw_moment == 0.02
    # r=.05, |Mz|=.02: translation=.01/(1+.16)=1/116, yaw=+/-2/29.
    assert right.cost_before == wrong.cost_before == pytest.approx(26.0, abs=1e-12)
    assert wrong.cost_after == pytest.approx(60041 / 841, abs=1e-11)
    assert right.cost_after == pytest.approx(3491 / 841, abs=1e-11)
    assert wrong.predicted_improvement == pytest.approx(26 - 60041 / 841, abs=1e-11)
    assert right.predicted_improvement == pytest.approx(26 - 3491 / 841, abs=1e-11)
    assert wrong.predicted_improvement < 0 < right.predicted_improvement


def test_yaw_wrap_crosses_pi_using_the_short_target_error():
    records = propose(
        current_rotation=HALF,
        target_xy=(0.0, -1 / 232),
        target_yaw=-pi + 1 / 29,
        push_distances_m=(0.005,),
    )
    record = next(item for item in records if item.index == 1)
    assert record.predicted_yaw_change == pytest.approx(1 / 29, abs=1e-14)
    assert record.predicted_yaw == pytest.approx(-pi + 1 / 29, abs=1e-14)
    assert record.cost_before == pytest.approx(10625 / 3364, abs=1e-11)
    assert record.cost_after < 1e-24


def test_current_pose_and_target_translate_and_rotate_covariantly():
    original = propose()
    shifted = propose(
        current_position=(2.0, -3.0, 0.0),
        current_rotation=QUARTER,
        target_xy=(2.0, -2.99),
        target_yaw=pi / 2 + 0.1,
    )
    assert tuple(record.index for record in shifted) == tuple(record.index for record in original)
    a, b = (
        next(record for record in original if record.index == 0),
        next(record for record in shifted if record.index == 0),
    )
    np.testing.assert_allclose(b.point_world, (1.98, -3.03, 0.0), rtol=0, atol=1e-14)
    np.testing.assert_allclose(b.contact_center_world, (1.98, -3.04, 0.06), rtol=0, atol=1e-14)
    assert b.direction_world == (0.0, 1.0, 0.0) and b.nominal_yaw_moment == pytest.approx(-0.02)
    assert b.cost_before == pytest.approx(a.cost_before, abs=1e-10)
    assert b.cost_after == pytest.approx(a.cost_after, abs=1e-10)
    assert b.predicted_improvement == pytest.approx(a.predicted_improvement, abs=1e-10)
    assert b.input_binding != a.input_binding


def test_small_tilt_preserves_full_material_geometry_but_only_proposes_a_planar_push():
    # Rational rotation: cos=399/401, sin=40/401; tilt=2*atan(1/20)<.1 rad.
    tilted = ((1.0, 0.0, 0.0), (0.0, 399 / 401, -40 / 401), (0.0, 40 / 401, 399 / 401))
    records = propose(current_rotation=tilted)
    bottom = next(record for record in records if record.index == 1)
    np.testing.assert_allclose(
        bottom.point_world, (0.02, -0.04 * 399 / 401, -0.04 * 40 / 401), rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        bottom.outward_normal_world, (0.0, -399 / 401, -40 / 401), rtol=0, atol=1e-15
    )
    assert bottom.direction_world == (0.0, 1.0, 0.0) and bottom.nominal_yaw_moment == 0.02
    assert (
        bottom.contact_center_world[2]
        == bottom.precontact_center_world[2]
        == bottom.push_center_world[2]
        == 0.06
    )
    assert all(record.status == "PROPOSAL_ONLY" for record in records)
    assert all(
        not hasattr(record, "sampled_clear") and not hasattr(record, "usable") for record in records
    )
    with pytest.raises(ValueError, match="tilt"):
        propose(
            current_rotation=(
                (1.0, 0.0, 0.0),
                (0.0, 99 / 101, -20 / 101),
                (0.0, 20 / 101, 99 / 101),
            )
        )


def test_rotation_roundoff_admission_keeps_the_original_matrix_without_repair():
    raw = np.diag((1.0000005, 1.0, 1.0))
    record = next(item for item in propose(current_rotation=raw) if item.index == 0)
    assert record.point_world[0] == pytest.approx(-0.030000015, abs=1e-16)
    assert record.input_binding != propose()[0].input_binding
    for rotation in (
        np.eye(2),
        np.diag((1.0, 1.0, -1.0)),
        np.diag((1.00001, 1.0, 1.0)),
        np.array(((1.0, 0.00001, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        np.full((3, 3), np.nan),
        np.full((3, 3), np.inf),
        np.eye(3, dtype=bool),
    ):
        for key in ("reference_rotation", "current_rotation"):
            with pytest.raises(ValueError):
                propose(**{key: rotation})


def test_bad_vectors_scalar_parameters_and_explicit_length_budgets_reject():
    cases = [
        {"reference_position": (0.0, 0.0)},
        {"current_position": (0.0, np.nan, 0.0)},
        {"current_position": (True, False, True)},
        {"target_xy": (0.0, 0.0, 0.0)},
        {"target_xy": (np.inf, 0.0)},
        {"target_yaw": np.nan},
        {"pusher_height_m": np.inf},
        {"push_distances_m": ()},
        {"push_distances_m": (0.01, 0.01)},
        {"push_distances_m": (0.01,) * 4},
        {"push_distances_m": (0.0,)},
        {"push_distances_m": (-0.01,)},
        {"push_distances_m": (0.030001,)},
        {"push_distances_m": (np.nan,)},
        {"push_distances_m": (True,)},
        {"push_distances_m": iter((0.01,))},
        {"push_distances_m": np.array([[0.01]])},
    ]
    for field in ("position_tolerance_m", "yaw_tolerance_rad", "stand_off_m"):
        cases.extend({field: value} for value in (0.0, -1.0, np.inf, np.nan, True))
    for case in cases:
        with pytest.raises(ValueError):
            propose(**case)
    with pytest.raises(ValueError):
        propose(target_xy=(1e308, 1e308), position_tolerance_m=1e-300)
    assert len(propose(push_distances_m=[0.005, 0.01, 0.03])) == 6
    assert len(propose(push_distances_m=np.array((0.005, 0.01, 0.03)))) == 6


def test_query_storage_identity_and_nonzero_mobility_or_projection_are_required():
    query = query_fixture()
    for bad in (
        None,
        replace(query, input_binding="invalid"),
        replace(query, q=query.q.copy()),
        replace(query, candidates=(query.candidates[0], query.candidates[0])),
        replace(query, object_body_id=128),
    ):
        with pytest.raises(ValueError):
            propose_planar_pushes(
                bad,
                reference_position=(0.0, 0.0, 0.0),
                reference_rotation=IDENTITY,
                current_position=(0.0, 0.0, 0.0),
                current_rotation=IDENTITY,
                target_xy=(0.01, 0.0),
                target_yaw=0.0,
                pusher_height_m=0.06,
                position_tolerance_m=0.01,
                yaw_tolerance_rad=0.02,
            )
    zero_boundary = tuple(
        replace(edge, start_world=(0.0, 0.0, 0.0), end_world=(0.0, 0.0, 0.0))
        for edge in query.boundary
    )
    with pytest.raises(ValueError, match="radius"):
        propose(replace(query, boundary=zero_boundary))
    for normal in ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)):
        with pytest.raises(ValueError, match="normal"):
            propose(
                replace(
                    query, candidates=(replace(query.candidates[0], outward_normal_world=normal),)
                )
            )


def test_all_384_proposals_are_retained_and_unusable_is_only_a_reference_flag():
    query = query_fixture()
    candidates = tuple(replace(query.candidates[0], index=index) for index in range(128))
    candidates = (*candidates[:-1], replace(candidates[-1], usable=False, reason="entry-occluded"))
    records = propose(replace(query, candidates=candidates), push_distances_m=(0.005, 0.01, 0.03))
    assert len(records) == 384
    assert {(record.index, record.distance_ordinal) for record in records} == {
        (index, ordinal) for index in range(128) for ordinal in range(3)
    }
    assert all(record.reference_usable for record in records[:-3])
    assert all(
        not record.reference_usable and record.status == "PROPOSAL_ONLY" for record in records[-3:]
    )
    with pytest.raises(ValueError):
        propose(
            replace(
                query,
                candidates=tuple(replace(query.candidates[0], index=index) for index in range(129)),
            )
        )
    assert propose(replace(query, candidates=())) == ()


def test_exact_score_ties_use_original_index_then_explicit_distance_ordinal():
    query = query_fixture()
    centered = replace(
        query.candidates[0],
        edge_fraction=0.5,
        point_world=(-0.03, 0.0, 0.0),
        contact_center_world=(-0.04, 0.0, 0.0),
        precontact_center_world=(-0.052, 0.0, 0.0),
        normal_wrench_com_world=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    query = replace(query, candidates=(replace(centered, index=1), centered))
    records = propose(
        query,
        target_xy=(2 / 128, 0.0),
        target_yaw=0.0,
        position_tolerance_m=1 / 64,
        push_distances_m=(3 / 128, 1 / 128),
    )
    assert tuple((record.index, record.distance_ordinal) for record in records) == (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    )
    assert all(
        record.cost_before == 1.0
        and record.cost_after == 0.25
        and record.predicted_improvement == 0.75
        for record in records
    )
    assert tuple(record.sort_key for record in records) == (
        (0, -0.75, 0, 0),
        (0, -0.75, 0, 1),
        (0, -0.75, 1, 0),
        (0, -0.75, 1, 1),
    )


@pytest.mark.parametrize("offset", (0.0, 1e-9))
def test_target_at_or_near_current_pose_does_not_invent_a_useful_nonzero_push(offset):
    records = propose(
        target_xy=(offset, 0.0), target_yaw=offset, push_distances_m=(0.005, 0.01, 0.02)
    )
    assert all(
        record.cost_before < 1e-12 and record.predicted_improvement < 0 for record in records
    )


def test_owned_immutable_deterministic_binding_and_no_native_or_simulation_import(monkeypatch):
    query = query_fixture()
    position, rotation, target = np.zeros(3), np.eye(3), np.array((0.01, 0.0))
    before = (
        query.q.tobytes(),
        query.plane_axes_world.tobytes(),
        position.tobytes(),
        rotation.tobytes(),
        target.tobytes(),
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "mujoco" or name == "tyx" or name.startswith("tyx."):
            raise AssertionError("a geometric proposal must not query native physics or simulation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    first = propose(query, current_position=position, current_rotation=rotation, target_xy=target)
    assert first == propose(
        query, current_position=position, current_rotation=rotation, target_xy=target
    )
    assert before == (
        query.q.tobytes(),
        query.plane_axes_world.tobytes(),
        position.tobytes(),
        rotation.tobytes(),
        target.tobytes(),
    )
    record = first[0]
    assert len(record.input_binding) == 64 and all(
        value in "0123456789abcdef" for value in record.input_binding
    )
    with pytest.raises(FrozenInstanceError):
        record.cost_before = 0
    with pytest.raises(TypeError):
        record.point_world[0] = 0
    position[:] = 99
    rotation[:] = 0
    target[:] = -99
    assert record.cost_before == pytest.approx(26.0)
    for values in (
        {"reference_position": (0.001, 0.0, 0.0)},
        {"reference_rotation": QUARTER},
        {"current_position": (0.001, 0.0, 0.0)},
        {"current_rotation": QUARTER},
        {"target_xy": (0.011, 0.0)},
        {"target_yaw": 0.11},
        {"pusher_height_m": 0.061},
        {"position_tolerance_m": 0.011},
        {"yaw_tolerance_rad": 0.021},
        {"push_distances_m": (0.02,)},
        {"stand_off_m": 0.013},
    ):
        assert propose(query, **values)[0].input_binding != record.input_binding
    changed = replace(
        query,
        candidates=(
            replace(query.candidates[0], point_world=(-0.03, 0.021, 0.0)),
            query.candidates[1],
        ),
    )
    assert (
        changed.input_binding == query.input_binding
    )  # A claimed binding alone is not authentication.
    assert propose(changed)[0].input_binding != record.input_binding
