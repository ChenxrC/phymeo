"""Hand-written ranking oracles; no producer, native model, or FK calls."""

import builtins
from dataclasses import FrozenInstanceError, replace
from math import cos, pi, sin, sqrt

import numpy as np
import pytest

from phymeo.control.manipulation_push import (
    NativePlanarPushContactCandidate,
    NativePlanarPushContacts,
    PlanarPushBoundaryEdge,
    PlanarPushRectangle,
)
from phymeo.control.manipulation_push_rank import rank_planar_push_candidates


SPECS = (
    ((1., 0.), -.4, True),
    ((1., 0.), .3, True),
    ((1., 0.), 0., True),
    ((0., 1.), .8, True),
    ((-1., 0.), .7, True),
    ((1., 0.), .9, False),
    ((1., 0.), .3, True),
)


def readonly(values):
    source = np.asarray(values, dtype=np.float64)
    return np.frombuffer(source.tobytes(), dtype=np.float64).reshape(source.shape)


def query_fixture(specs=SPECS):
    """Four hand-declared square faces, with point moments about the origin."""
    edges = (
        PlanarPushBoundaryEdge(0, (-1., -1., 0.), (-1., 1., 0.), (-1., 0., 0.), (1,)),
        PlanarPushBoundaryEdge(1, (1., -1., 0.), (1., 1., 0.), (1., 0., 0.), (1,)),
        PlanarPushBoundaryEdge(2, (-1., -1., 0.), (1., -1., 0.), (0., -1., 0.), (1,)),
        PlanarPushBoundaryEdge(3, (-1., 1., 0.), (1., 1., 0.), (0., 1., 0.), (1,)),
    )
    edge_ids = {(1., 0.): 0, (-1., 0.): 1, (0., 1.): 2, (0., -1.): 3}
    candidates = []
    for index, (force, moment, usable) in enumerate(specs):
        fx, fy = force
        point = (-fx + moment * fy, -fy - moment * fx, 0.)
        normal = (-fx, -fy, 0.)
        center = tuple(point[axis] + .1 * normal[axis] for axis in range(3))
        precontact = (center[0] - .1 * fx, center[1] - .1 * fy, center[2])
        edge = edge_ids[force]
        fraction = (point[1 if edge < 2 else 0] + 1) / 2
        candidates.append(
            NativePlanarPushContactCandidate(
                index, edge, fraction, 0., (1,), point, normal, center, precontact,
                (fx, fy, 0.), (fx, fy, 0., 0., 0., moment), (.1,), (.1,),
                usable, "clear" if usable else "entry-occluded",
            )
        )
    return NativePlanarPushContacts(
        "a" * 64, readonly((0, 0, 0, 1, 0, 0, 0)), 1, (0., 0., 0.), 2, .1,
        (0., 0., 0.), readonly(np.eye(3)),
        (PlanarPushRectangle(1, "shape", (-1., -1.), (1., 1.), (-.1, .1)),),
        edges, tuple(candidates), 1e-9,
    )


def indices(records):
    return tuple(record.index for record in records)


@pytest.mark.parametrize(
    ("yaw", "expected"),
    (
        (.2, (1, 6, 2, 0, 3, 4, 5)),
        (-.2, (0, 2, 1, 6, 3, 4, 5)),
        (0., (2, 1, 6, 0, 3, 4, 5)),
    ),
)
def test_translation_then_signed_yaw_then_index_with_blocked_retained(yaw, expected):
    query = query_fixture()
    records = rank_planar_push_candidates(query, (3., 0.), yaw)
    assert indices(records) == expected
    assert len(records) == len(query.candidates)
    by_index = {record.index: record for record in records}
    for index, (force, moment, usable) in enumerate(SPECS):
        record = by_index[index]
        assert record.translation_alignment == force[0]
        assert record.nominal_yaw_moment == moment
        assert record.usable is usable
        moment_key = abs(moment) if yaw == 0 else -moment if yaw > 0 else moment
        assert record.sort_key == (0 if usable else 1, -force[0], moment_key, index)
    assert by_index[5].translation_alignment == 1
    assert by_index[5].nominal_yaw_moment == .9
    assert records[-1].index == 5


@pytest.mark.parametrize(
    ("translation", "yaw", "expected"),
    (
        ((0., 0.), .2, (3, 4, 1, 6, 2, 0, 5)),
        ((0., 0., -0.), -.2, (0, 2, 1, 6, 4, 3, 5)),
        ((0., 0.), 0., (2, 1, 6, 0, 4, 3, 5)),
    ),
)
def test_zero_translation_keeps_yaw_or_minimum_absolute_moment(translation, yaw, expected):
    records = rank_planar_push_candidates(query_fixture(), translation, yaw)
    assert indices(records) == expected
    assert all(record.translation_alignment == 0 for record in records)


def test_translation_normalization_is_scale_invariant_and_does_not_project_z():
    query = query_fixture()
    expected = rank_planar_push_candidates(query, (1., 1.), .1)
    for direction in ((2., 2., 0.), (1e308, 1e308), (1e-310, 1e-310)):
        result = rank_planar_push_candidates(query, direction, .1)
        assert indices(result) == indices(expected)
        for a, b in zip(result, expected, strict=True):
            assert a.translation_alignment == pytest.approx(b.translation_alignment, abs=1e-15)
    assert next(record for record in expected if record.index == 0).translation_alignment == pytest.approx(1 / sqrt(2))
    with pytest.raises(ValueError, match="planar"):
        rank_planar_push_candidates(query, (1., 1., 1e-30), .1)


def transformed(query, yaw, shift):
    rotation = np.array(((cos(yaw), -sin(yaw), 0), (sin(yaw), cos(yaw), 0), (0, 0, 1)))
    shift = np.asarray(shift)

    def point(value):
        return tuple(map(float, rotation @ value + shift))

    def vector(value):
        return tuple(map(float, rotation @ value))

    edges = tuple(
        replace(edge, start_world=point(edge.start_world), end_world=point(edge.end_world),
                outward_normal_world=vector(edge.outward_normal_world))
        for edge in query.boundary
    )
    candidates = tuple(
        replace(
            candidate, point_world=point(candidate.point_world),
            contact_center_world=point(candidate.contact_center_world),
            precontact_center_world=point(candidate.precontact_center_world),
            outward_normal_world=vector(candidate.outward_normal_world),
            approach_direction_world=vector(candidate.approach_direction_world),
            normal_wrench_com_world=(
                *vector(candidate.normal_wrench_com_world[:3]),
                *vector(candidate.normal_wrench_com_world[3:]),
            ),
        )
        for candidate in query.candidates
    )
    return replace(
        query, input_binding="b" * 64,
        q=readonly((*shift, cos(yaw / 2), 0., 0., sin(yaw / 2))),
        object_com_world=point(query.object_com_world),
        plane_origin_world=point(query.plane_origin_world),
        plane_axes_world=readonly(rotation @ query.plane_axes_world),
        boundary=edges, candidates=candidates,
    )


@pytest.mark.parametrize(("yaw", "shift"), ((0., (12., -7., 0.)), (.7, (0., 0., 0.)), (-pi / 3, (2., 5., 0.))))
def test_world_translation_and_planar_rotation_covariance(yaw, shift):
    query = query_fixture()
    first = rank_planar_push_candidates(query, (2., 1.), -.4)
    moved = transformed(query, yaw, shift)
    direction = (2 * cos(yaw) - sin(yaw), 2 * sin(yaw) + cos(yaw))
    second = rank_planar_push_candidates(moved, direction, -.4)
    assert indices(first) == indices(second)
    for a, b in zip(first, second, strict=True):
        assert a.nominal_yaw_moment == b.nominal_yaw_moment
        assert a.translation_alignment == pytest.approx(b.translation_alignment, abs=1e-15)


def test_angle_of_approach_is_not_substituted_for_nominal_force():
    query = query_fixture()
    candidate = replace(query.candidates[0], approach_angle_rad=.6,
                        approach_direction_world=(cos(.6), sin(.6), 0.))
    changed = replace(query, candidates=(candidate, *query.candidates[1:]))
    assert rank_planar_push_candidates(changed, (0., 1.), .2) == rank_planar_push_candidates(query, (0., 1.), .2)


def test_ties_use_original_indices_even_when_candidate_tuple_is_permuted():
    query = query_fixture((((1., 0.), 0., True),) * 5)
    reverse = replace(query, candidates=tuple(reversed(query.candidates)))
    assert indices(rank_planar_push_candidates(reverse, (0., 0.), 0.)) == (0, 1, 2, 3, 4)
    assert indices(rank_planar_push_candidates(reverse, (1., 0.), .3)) == (0, 1, 2, 3, 4)


def test_empty_and_maximum_candidate_counts_remain_bounded():
    assert rank_planar_push_candidates(query_fixture(()), (0., 0.), 0.) == ()
    query = query_fixture((((1., 0.), 0., True),) * 128)
    assert indices(rank_planar_push_candidates(query, (1., 0.), 0.)) == tuple(range(128))
    with pytest.raises(ValueError, match="bounded"):
        rank_planar_push_candidates(query_fixture((((1., 0.), 0., True),) * 129), (1., 0.), 0.)


@pytest.mark.parametrize(
    "translation",
    (
        (), (1,), (1, 2, 0, 0), (float("nan"), 0), (0, float("inf")),
        (True, 0), ("1", 0), (1 + 0j, 0), (1, 0, 1), "10", iter((1, 0)),
        np.ones((1, 2)), np.array((True, False)),
    ),
)
def test_bad_translation_rejects(translation):
    with pytest.raises(ValueError):
        rank_planar_push_candidates(query_fixture(), translation, 0.)


@pytest.mark.parametrize("yaw", (True, np.bool_(True), "0", float("nan"), float("inf"), (0,), 1j))
def test_bad_yaw_rejects(yaw):
    with pytest.raises(ValueError):
        rank_planar_push_candidates(query_fixture(), (1., 0.), yaw)


@pytest.mark.parametrize(
    "change",
    (
        dict(index=1), dict(index=7), dict(index=True), dict(edge_index=4),
        dict(owner_geometry_ids=(2,)), dict(owner_geometry_ids=[1]),
        dict(point_world=(0., 0.)), dict(outward_normal_world=(0., float("inf"), 0.)),
        dict(precontact_center_world=[0., 0., 0.]), dict(approach_direction_world=(True, 0., 0.)),
        dict(normal_wrench_com_world=(1., 0., 0., 0., 0.)),
        dict(normal_wrench_com_world=(1., 0., 0., 0., 0., float("nan"))),
        dict(entry_distances_m=(float("inf"),)), dict(contact_distances_m=(-.1,)),
        dict(edge_fraction=0.), dict(approach_angle_rad=pi / 2),
        dict(usable=1), dict(reason=""), dict(reason="entry-occluded"),
    ),
)
def test_malformed_candidate_rejects_without_modifying_input(change):
    query = query_fixture()
    broken = replace(query, candidates=(replace(query.candidates[0], **change), *query.candidates[1:]))
    original_q = query.q.tobytes()
    with pytest.raises(ValueError):
        rank_planar_push_candidates(broken, (1., 0.), .2)
    assert query.q.tobytes() == original_q
    assert query.candidates[0].index == 0 and query.candidates[0].reason == "clear"


def bad_query_cases(query):
    writable_base = np.array(query.q)
    readonly_view = writable_base.view()
    readonly_view.setflags(write=False)
    return (
        replace(query, input_binding="a" * 63),
        replace(query, input_binding="G" * 64),
        replace(query, q=np.array(query.q)),
        replace(query, q=readonly_view),
        replace(query, q=readonly((float("nan"),))),
        replace(query, q=readonly(np.zeros((1, 7)))),
        replace(query, plane_axes_world=readonly(np.eye(2))),
        replace(query, plane_axes_world=np.eye(3)),
        replace(query, object_com_world=[0., 0., 0.]),
        replace(query, object_body_id=True),
        replace(query, pusher_radius_m=0.),
        replace(query, query_tolerance_m=1e-6),
        replace(query, rectangles=list(query.rectangles)),
        replace(query, rectangles=(replace(query.rectangles[0], upper_xy=(-2., 1.)),)),
        replace(query, boundary=(query.boundary[0], query.boundary[0])),
        replace(query, candidates=list(query.candidates)),
    )


def test_invalid_public_query_schema_and_mutable_storage_reject():
    query = query_fixture()
    for broken in bad_query_cases(query):
        with pytest.raises(ValueError):
            rank_planar_push_candidates(broken, (1., 0.), 0.)
    with pytest.raises(ValueError):
        rank_planar_push_candidates({}, (1., 0.), 0.)


def test_rankings_are_owned_immutable_scalars_and_do_not_import_native(monkeypatch):
    query = query_fixture()
    old_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "mujoco" or name.startswith("mujoco."):
            raise AssertionError("ranking must not import or execute native physics")
        return old_import(name, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "__import__", guarded_import)
        result = rank_planar_push_candidates(query, (1., 0.), .2)
    assert type(result) is tuple
    assert all(type(record.translation_alignment) is float for record in result)
    assert all(type(record.nominal_yaw_moment) is float for record in result)
    assert all(type(record.sort_key) is tuple for record in result)
    with pytest.raises(FrozenInstanceError):
        result[0].index = 100
    with pytest.raises(TypeError):
        result[0].sort_key[1] = 0
    with pytest.raises(TypeError):
        result[0] = result[1]
    assert query.q.tobytes() == readonly((0, 0, 0, 1, 0, 0, 0)).tobytes()
