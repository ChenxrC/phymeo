"""Tests first: two independent noncommuting rigid arcs through public detect."""

import json
from dataclasses import asdict, replace
from fractions import Fraction as F
from math import cos, sin, sqrt

import numpy as np
import pytest
from dual_turn_mesh_round_cases import (
    A, B01, B02, B03, B04, B05, B06, B07, B08, CASES,
    PI_HIGH, PI_LOW, height_interval, mesh_distance_reference, public_case,
    root_interval, shaft_span_reference, static_paths, stored_pose, transform,
)
from rotating_capsule_mesh_cases import closed_oriented_reference

import phymeo as ph


ADDRESS_ATOL_M = 2e-9
REFERENCE_DISTANCE_ATOL_M = 2e-12


def detect(model, paths):
    with ph.CollisionWorkspace(model) as work:
        result = ph.detect(model, paths, ph.CollisionPolicy(), work)
    assert result.status == "ok"
    return result.pairs[0]


def check_materials(row, pose, mesh, round_shape, paths, time):
    mesh_points = dict(zip(mesh.vertex_ids, stored_pose(paths[mesh.key], time), strict=True))
    round_points = dict(zip(round_shape.vertex_ids, stored_pose(paths[round_shape.key], time), strict=True))
    spans = []
    assert pose.supports
    for support in (*pose.supports, *pose.boundary):
        assert support.face in mesh.triangles
        witness = support.witness
        mp, rp, mw, rw, offset = (
            (witness.first_point, witness.second_point, witness.first_weights, witness.second_weights, witness.second_offset)
            if row.first == mesh.key else
            (witness.second_point, witness.first_point, witness.second_weights, witness.first_weights, witness.first_offset)
        )
        assert set(i for i, _ in mw) <= set(support.face)
        assert set(i for i, _ in rw) <= set(round_shape.vertex_ids)
        for weights in (mw, rw):
            assert all(0 <= weight <= 1 for _, weight in weights)
            assert sum(weight for _, weight in weights) == pytest.approx(1, abs=1e-13)
        np.testing.assert_allclose(sum(w*mesh_points[i] for i, w in mw), mp, rtol=0, atol=ADDRESS_ATOL_M)
        anchor = sum(w*round_points[i] for i, w in rw)
        np.testing.assert_allclose(anchor + offset, rp, rtol=0, atol=ADDRESS_ATOL_M)
        assert np.linalg.norm(offset) == pytest.approx(round_shape.radius, abs=ADDRESS_ATOL_M)
        mesh_offset = witness.first_offset if row.first == mesh.key else witness.second_offset
        assert mesh_offset == (0.0, 0.0, 0.0)
        if round_shape.kind == "capsule":
            p, q = (round_points[i] for i in round_shape.vertex_ids)
            axis = q-p
            u = float(np.clip((np.asarray(rp)-p) @ axis / (axis @ axis), 0, 1))
            assert np.linalg.norm(np.asarray(rp)-(p+u*axis)) == pytest.approx(round_shape.radius, abs=ADDRESS_ATOL_M)
        if witness.normal is not None:
            assert np.linalg.norm(witness.normal) == pytest.approx(1, abs=ADDRESS_ATOL_M)
            assert np.dot(np.subtract(witness.second_point, witness.first_point), witness.normal) == pytest.approx(witness.gap_m, abs=ADDRESS_ATOL_M)
        if witness.capsule_axis_span is not None:
            assert round_shape.kind == "capsule"
            low, high = witness.capsule_axis_span
            assert 0 <= low.axis_parameter <= high.axis_parameter <= 1
            for endpoint in (low, high):
                u = endpoint.axis_parameter
                p, q = (round_points[i] for i in round_shape.vertex_ids)
                np.testing.assert_allclose(endpoint.axis_point, (1-u)*p+u*q, rtol=0, atol=ADDRESS_ATOL_M)
                assert set(i for i, _ in endpoint.face_weights) <= set(support.face)
                np.testing.assert_allclose(sum(w*mesh_points[i] for i, w in endpoint.face_weights), endpoint.face_point, rtol=0, atol=ADDRESS_ATOL_M)
            spans.append((low.axis_parameter, high.axis_parameter))
    return sorted(spans)


def check_entry(row, reference, mesh, round_shape, paths):
    assert row.candidate and row.status == "hit", row
    trace = row.mesh_round
    event = trace.first_contact
    low, high = map(F, event.interval)
    assert low <= reference[1] and reference[0] <= high
    assert high-low <= F(1, 1 << 27)
    assert low <= F(event.witness_time) <= high
    assert F(event.witness_time) >= reference[0]
    assert row.sweep.first_t == event.interval
    assert row.sweep.witness_time == event.witness_time and row.sweep.witness == event.pose.witness
    assert event.pose.relation != "separated"
    if reference != (F(0), F(0)):
        squared = mesh_distance_reference(mesh, round_shape, paths, event.witness_time)[0][0]
        assert float(squared) <= (round_shape.radius + REFERENCE_DISTANCE_ATOL_M)**2
        assert event.pose.axis_gap_m <= 0
    check_materials(row, event.pose, mesh, round_shape, paths, event.witness_time)
    for pose, time in ((trace.start, 0), (trace.end, 1)):
        check_materials(row, pose, mesh, round_shape, paths, time)
    return event


def test_b01_public_two_axes_after_independent_static_overlap(tmp_path):
    """Effective RED: old public static contact succeeds before both-turn route fails."""
    model, paths, mesh, capsule = public_case(B01)
    assert np.linalg.norm(np.cross(paths[mesh.key].turn.axis, paths[capsule.key].turn.axis)) == 1
    start, end = (stored_pose(paths[capsule.key], time) for time in (0, 1))
    assert np.linalg.norm(np.cross(start[1]-start[0], end[1]-end[0])) > 0.9
    assert paths[mesh.key].turn.angle == -2*paths[capsule.key].turn.angle
    reference = root_interval(B01, paths[capsule.key].turn.angle)
    assert F(1, 3) < reference[0] < reference[1] < F(3, 8)
    for time in (0, 1):
        assert mesh_distance_reference(mesh, capsule, paths, time)[0][0] > capsule.radius**2
    sample_time = 3/8
    sample = static_paths(paths, sample_time)
    static = detect(model, sample)
    assert static.status == "overlap" and static.mesh_round.start.relation == "boundary_overlap"
    height = height_interval(F(3, 8), paths[capsule.key].turn.angle)
    assert height[0] > F(3, 4)
    gap = F(3, 4)-(height[0]+height[1])/2
    assert static.mesh_round.start.axis_gap_m == pytest.approx(float(gap), abs=REFERENCE_DISTANCE_ATOL_M)
    check_materials(static, static.mesh_round.start, mesh, capsule, sample, 0)
    row = detect(model, paths)
    (tmp_path / "b01-dual-turn-public.json").write_text(json.dumps(
        dict(case=asdict(B01), independent_root=[str(x) for x in reference], static=asdict(static), actual=asdict(row))
    ), encoding="utf8")
    event = check_entry(row, reference, mesh, capsule, paths)
    assert row.mesh_round.start.relation == row.mesh_round.end.relation == "separated"
    assert event.pose.relation == "boundary_overlap"
    t = event.witness_time
    normal = (0, -sin(2*A*t), -cos(2*A*t))
    np.testing.assert_allclose(event.pose.witness.normal, normal, rtol=0, atol=ADDRESS_ATOL_M)
    assert all(all(point.axis_parameter == pytest.approx(1, abs=1e-10) for point in support.witness.capsule_axis_span) for support in event.pose.supports)


def test_independent_eight_inputs_topology_time_and_clearance_qualifications():
    assert len(CASES) == 8 and len({case.name for case in CASES}) == 8
    for case in CASES:
        assert closed_oriented_reference(case.vertices, case.triangles) > 0
        assert np.linalg.norm(np.cross(case.mesh_axis, case.round_axis)) == 1
    assert closed_oriented_reference(B06.vertices, B06.triangles) == 5
    assert 2*F(A) < PI_LOW < PI_HIGH
    low, high = root_interval(B03)
    assert F(5, 16) <= low < high <= F(1, 3) and high-low <= F(1, 1 << 42)
    assert height_interval(low, drift=F(1, 64))[1] < F(3, 4)
    assert height_interval(high, drift=F(1, 64))[0] > F(3, 4)
    assert 7/8-4/(3*sqrt(3)) > 0
    # End balls stay outside the circumspheres for the complete B04 path.
    assert sqrt(10)-sqrt(93)/4-1/8 > 1/2


@pytest.mark.parametrize("case", [B02, B03])
def test_orbiting_sphere_and_independent_pivot_travel_have_predeclared_first(case, tmp_path):
    model, paths, mesh, round_shape = public_case(case)
    reference = root_interval(case, paths[round_shape.key].turn.angle)
    row = detect(model, paths)
    (tmp_path / f"{case.name}.json").write_text(json.dumps(dict(
        case=asdict(case), independent_root=[str(x) for x in reference], actual=asdict(row)
    )), encoding="utf8")
    event = check_entry(row, reference, mesh, round_shape, paths)
    assert row.mesh_round.start.relation == row.mesh_round.end.relation == "separated"
    if case is B02:
        for support in event.pose.supports:
            weights = support.witness.second_weights
            assert weights == ((41, 1.0),)
            assert support.witness.capsule_axis_span is None
    else:
        assert paths[mesh.key].turn.pivot_end != paths[round_shape.key].turn.pivot_end


def test_b04_full_shaft_span_first_while_both_end_balls_stay_clear(tmp_path):
    model, paths, mesh, capsule = public_case(B04)
    reference = root_interval(B04)
    assert sqrt(10)-sqrt(93)/4-capsule.radius > 1/2
    row = detect(model, paths)
    (tmp_path / "b04-shaft.json").write_text(json.dumps(asdict(row)), encoding="utf8")
    event = check_entry(row, reference, mesh, capsule, paths)
    assert row.mesh_round.start.relation == row.mesh_round.end.relation == "separated"
    spans = check_materials(row, event.pose, mesh, capsule, paths, event.witness_time)
    assert spans and all(0 < low <= high < 1 for low, high in spans)
    first, last = spans[0]
    for low, high in spans[1:]:
        assert low <= last + 1e-10
        last = max(last, high)
    np.testing.assert_allclose((first, last), shaft_span_reference(event.witness_time), rtol=0, atol=ADDRESS_ATOL_M)


@pytest.mark.parametrize("case,clearance", [(B05, 7/8-4/(3*sqrt(3))), (B06, 5/8)])
def test_complete_clear_path_uses_true_height_or_notch_not_sampled_exclusion(case, clearance):
    # These are analytic full-path bounds: the cubic's maximum, or the U's
    # unit-radius excluded ball and the complete radius-1/4 finite spine.
    assert clearance > 0
    model, paths, mesh, capsule = public_case(case)
    row = detect(model, paths)
    assert row.candidate and row.status == "separated", row
    assert row.sweep.first_t is None and row.mesh_round.first_contact is None
    for pose, time in ((row.mesh_round.start, 0), (row.mesh_round.end, 1)):
        assert pose.relation == "separated" and pose.axis_gap_m >= clearance-REFERENCE_DISTANCE_ATOL_M
        check_materials(row, pose, mesh, capsule, paths, time)
        squared = mesh_distance_reference(mesh, capsule, paths, time)[0][0]
        assert sqrt(float(squared))-capsule.radius >= clearance-REFERENCE_DISTANCE_ATOL_M


def test_initial_containment_is_first_zero_not_a_surface_clear_false_negative():
    model, paths, mesh, capsule = public_case(B07)
    row = detect(model, paths)
    event = check_entry(row, root_interval(B07), mesh, capsule, paths)
    assert event.interval == (0, 0) and event.witness_time == 0
    assert row.mesh_round.start.relation == row.mesh_round.end.relation == "mesh_contains_round"
    for pose in (row.mesh_round.start, row.mesh_round.end):
        assert pose.boundary_gap_m >= 11/8-REFERENCE_DISTANCE_ATOL_M
        assert pose.axis_location == "inside"


def test_isolated_exact_grazing_cannot_be_reported_clear():
    model, paths, mesh, capsule = public_case(B08)
    row = detect(model, paths)
    assert row.candidate and row.status in ("hit", "unresolved"), row
    if row.status == "hit":
        check_entry(row, root_interval(B08), mesh, capsule, paths)
    else:
        assert row.mesh_round is None and row.sweep is None and row.witness is None


@pytest.mark.parametrize("kind", ["mesh_angle", "round_angle", "samples", "invalid_boundary"])
def test_angle_sample_budget_and_invalid_boundary_remain_explicit(kind, monkeypatch):
    model, paths, mesh, capsule = public_case(B01)
    if kind in ("mesh_angle", "round_angle"):
        key = mesh.key if kind == "mesh_angle" else capsule.key
        paths[key] = ph.CollisionMotion(paths[key].start, turn=replace(paths[key].turn, angle=33.0))
    elif kind == "samples":
        from phymeo.geometry import _dual_turn_mesh_round as implementation

        monkeypatch.setattr(implementation, "_MAX_SAMPLES", 1)
    else:
        broken = replace(mesh, triangles=mesh.triangles[:-1])
        model = ph.CollisionModel((broken, capsule))
    row = detect(model, paths)
    assert row.candidate
    if kind == "invalid_boundary":
        assert row.status == "invalid_geometry" and row.reason == "mesh_open_or_misoriented_edge"
    else:
        assert row.status == "unresolved" and "capacity" in row.reason
    assert row.mesh_round is None and row.sweep is None and row.witness is None


@pytest.mark.parametrize("reverse,reorder,matrix,offset", [
    (True, False, np.eye(3), (0, 0, 0)),
    (False, True, np.diag((-1, 1, 1)), (9/4, -3/4, 3/8)),
    (True, True, np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0))), (9/4, -3/4, 3/8)),
])
def test_shape_order_storage_mirror_and_rigid_frame_preserve_true_material(reverse, reorder, matrix, offset):
    case = transform(B01, matrix, offset)
    model, paths, mesh, capsule = public_case(case, reverse=reverse, reorder=reorder)
    row = detect(model, paths)
    event = check_entry(row, root_interval(case), mesh, capsule, paths)
    t = event.witness_time
    normal = matrix @ np.array((0, -sin(2*A*t), -cos(2*A*t)))
    if reverse:
        normal = -normal
    np.testing.assert_allclose(event.pose.witness.normal, normal, rtol=0, atol=ADDRESS_ATOL_M)


def test_common_rigid_change_of_frame_preserves_moving_pivot_first():
    matrix = np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0)))
    case = transform(B03, matrix, (1/4, -1/2, 3/8))
    model, paths, mesh, capsule = public_case(case, reverse=True, reorder=True)
    check_entry(detect(model, paths), root_interval(case), mesh, capsule, paths)


def test_multiple_worlds_capacity_abort_and_retry_preserve_prior_publication():
    first, pa, ma, ca = public_case(B01, world=3)
    second, pb, mb, cb = public_case(B05, world=8)
    model, paths = ph.CollisionModel(first.shapes + second.shapes), pa | pb
    with ph.CollisionWorkspace(model) as work:
        result = ph.detect(model, paths, ph.CollisionPolicy(), work)
        rows = {(row.first, row.second): row for row in result.pairs}
        assert len(result.pairs) == 6 and sum(row.reason == "different_world" for row in result.pairs) == 4
        assert rows[ma.key, ca.key].status == "hit" and rows[mb.key, cb.key].status == "separated"
        old = asdict(result)
        rejected = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0), work)
        assert rejected.status == "capacity_exceeded" and not rejected.pairs and not rejected.bounds
        assert work.last_result is result and asdict(result) == old
        with pytest.raises(RuntimeError, match="abort frame"):
            with work.transaction():
                ph.detect(model, paths, ph.CollisionPolicy(), work)
                raise RuntimeError("abort frame")
        assert work.last_result is result and asdict(result) == old
        assert ph.detect(model, paths, ph.CollisionPolicy(), work) == result
    model, paths, _, _ = public_case(B01, round_world=1)
    row = detect(model, paths)
    assert not row.candidate and row.status == "filtered" and row.reason == "different_world"
