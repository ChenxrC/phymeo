"""B01--B06 public detection contracts, with independent geometric references."""

import json
from dataclasses import asdict
from math import sqrt

import numpy as np
import pytest
from rotating_capsule_mesh_cases import B01, b01_face_gap, b01_first, public_case, stored_pose

import phymeo as ph


def detect(model, paths):
    with ph.CollisionWorkspace(model) as workspace:
        result = ph.detect(model, paths, ph.CollisionPolicy(), workspace)
    assert result.status == "ok"
    return result.pairs[0]


def check_materials(row, pose, mesh, capsule, paths, time):
    vertices = dict(zip(mesh.vertex_ids, stored_pose(paths[mesh.key], time), strict=True))
    axis = stored_pose(paths[capsule.key], time)
    spans = []
    assert pose.supports
    for support in (*pose.supports, *pose.boundary):
        witness = support.witness
        assert support.face in mesh.triangles
        mp, cp, mw, cw, radial = (
            (witness.first_point, witness.second_point, witness.first_weights,
             witness.second_weights, witness.second_offset)
            if row.first == mesh.key else
            (witness.second_point, witness.first_point, witness.second_weights,
             witness.first_weights, witness.first_offset)
        )
        assert set(i for i, weight in mw if weight) <= set(support.face)
        assert set(i for i, _ in cw) <= set(capsule.vertex_ids)
        assert sum(weight for _, weight in mw) == pytest.approx(1.0, abs=1e-13)
        assert sum(weight for _, weight in cw) == pytest.approx(1.0, abs=1e-13)
        np.testing.assert_allclose(sum(weight * vertices[i] for i, weight in mw), mp, rtol=0, atol=2e-9)
        anchor = sum(weight * axis[capsule.vertex_ids.index(i)] for i, weight in cw)
        np.testing.assert_allclose(anchor + radial, cp, rtol=0, atol=2e-9)
        assert np.linalg.norm(radial) == pytest.approx(capsule.radius, abs=2e-9)
        if witness.normal is not None:
            assert np.linalg.norm(witness.normal) == pytest.approx(1.0, abs=2e-9)
            assert np.dot(np.subtract(witness.second_point, witness.first_point), witness.normal) == pytest.approx(witness.gap_m, abs=2e-9)
        if witness.capsule_axis_span is not None:
            low, high = witness.capsule_axis_span
            assert 0 <= low.axis_parameter <= high.axis_parameter <= 1
            for item in (low, high):
                u = item.axis_parameter
                np.testing.assert_allclose(item.axis_point, (1-u)*axis[0]+u*axis[1], rtol=0, atol=2e-9)
                np.testing.assert_allclose(sum(weight * vertices[i] for i, weight in item.face_weights), item.face_point, rtol=0, atol=2e-9)
            spans.append((low.axis_parameter, high.axis_parameter))
    return sorted(spans)


def test_b01_public_orbit_shaft_first_after_real_static_positive(tmp_path):
    """Primary RED: old static native mesh/capsule works, full arc is unsupported."""
    model, paths, mesh, capsule = public_case(B01)
    angle = paths[capsule.key].turn.angle
    root = b01_first(angle)
    assert 1 < sqrt(35) / 4 < 2
    assert 0 < root < 15 / 32
    # Exact z separation proves both endpoint spheres stay clear for the full arc.
    assert 1 - 1 / 4 - capsule.radius == 5 / 8
    middle = 15 / 32
    endpoints = stored_pose(paths[capsule.key], middle)
    assert 1 < endpoints[0, 0] < 2
    static_paths = paths | {capsule.key: ph.CollisionMotion(endpoints)}
    static = detect(model, static_paths)
    assert static.status == "overlap" and static.mesh_round.start.relation == "boundary_overlap"
    assert static.mesh_round.start.axis_gap_m == pytest.approx(b01_face_gap(angle, middle), abs=2e-12)
    check_materials(static, static.mesh_round.start, mesh, capsule, static_paths, 0)

    row = detect(model, paths)
    (tmp_path / "b01-public-first.json").write_text(
        json.dumps(dict(case=asdict(B01), root=root, static=asdict(static), actual=asdict(row))), encoding="utf8"
    )
    assert row.candidate and row.status == "hit", row
    trace = row.mesh_round
    assert trace.start.relation == trace.end.relation == "separated"
    event = trace.first_contact
    lo, hi = event.interval
    assert lo <= root <= event.witness_time <= hi and hi - lo <= 2**-27
    assert row.sweep.first_t == event.interval and row.sweep.witness_time == event.witness_time
    assert event.pose.relation == "boundary_overlap" and event.pose.axis_gap_m <= 0
    spans = check_materials(row, event.pose, mesh, capsule, paths, event.witness_time)
    assert spans
    first, last = spans[0]
    for low, high in spans[1:]:
        assert low <= last + 1e-10
        last = max(last, high)
    np.testing.assert_allclose((first, last), (3/8, 5/8), rtol=0, atol=1e-10)
    for pose, time in ((trace.start, 0), (trace.end, 1)):
        check_materials(row, pose, mesh, capsule, paths, time)


def test_frozen_closed_meshes_and_independent_finite_feature_oracle():
    from rotating_capsule_mesh_cases import (
        B03, B04, closed_oriented_reference, segment_triangle_reference,
    )

    assert closed_oriented_reference(B01.vertices, B01.triangles) == 1/8
    assert closed_oriented_reference(B03.vertices, B03.triangles) == 5
    assert closed_oriented_reference(B04.vertices, B04.triangles) == 1/2
    tri = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
    # Full segment crosses a finite face, although both endpoints are outside.
    assert segment_triangle_reference(((1/4, 1/4, -1), (1/4, 1/4, 1)), tri)[0] == 0
    # Its infinite supporting line would cross, but its finite segment is clear.
    assert segment_triangle_reference(((1/4, 1/4, 1), (1/4, 1/4, 2)), tri)[0] == 1
    assert segment_triangle_reference(((2, 0, 1), (2, 0, 2)), tri)[0] == 2
    assert segment_triangle_reference(((0, 0, 1), (0, 0, 1)), tri)[0] == 1


def _check_first(row, expected, model, paths, mesh, capsule):
    from rotating_capsule_mesh_cases import mesh_distance_reference

    assert row.candidate and row.status == "hit", row
    trace = row.mesh_round
    event = trace.first_contact
    lo, hi = event.interval
    # Numerical finite-feature reference for a transformed B04 uses a declared
    # 2e-12 normalized-time allowance; the product interval is never widened.
    if isinstance(expected, tuple):
        assert lo <= expected[1] + 2e-12 and expected[0] - 2e-12 <= hi
    else:
        assert lo <= expected <= hi
    assert lo <= event.witness_time <= hi and hi-lo <= 2**-27
    assert event.pose.relation != "separated"
    if expected != 0:
        assert event.pose.axis_gap_m <= 0
        squared = mesh_distance_reference(mesh, capsule, paths, event.witness_time)[0][0]
        assert float(squared) <= (capsule.radius + 2e-12)**2
    assert row.sweep.first_t == event.interval and row.sweep.witness == event.pose.witness
    check_materials(row, event.pose, mesh, capsule, paths, event.witness_time)
    check_materials(row, trace.start, mesh, capsule, paths, 0)
    check_materials(row, trace.end, mesh, capsule, paths, 1)
    return event


@pytest.mark.parametrize("name", ["B02", "B03", "B06_FINITE"])
def test_b02_b03_full_path_clearance_preserves_concavity_and_finite_ends(name, tmp_path):
    from math import atan
    import rotating_capsule_mesh_cases as cases

    case = getattr(cases, name)
    model, paths, mesh, capsule = public_case(case)
    target = {"B02": (17-sqrt(257))/8, "B03": 5/8, "B06_FINITE": 1/4}[name]
    if name == "B02":
        t = atan(16)/abs(paths[capsule.key].turn.angle)
        value = cases.mesh_distance_reference(mesh, capsule, paths, t)[0][0]
        assert sqrt(float(value))-capsule.radius == pytest.approx(target, abs=2e-12)
    for t in (0, 1/4, 1/2, 3/4, 1):
        value = cases.mesh_distance_reference(mesh, capsule, paths, t)[0][0]
        assert sqrt(float(value))-capsule.radius >= target-2e-12
    row = detect(model, paths)
    (tmp_path / f"{name}.json").write_text(json.dumps(asdict(row)), encoding="utf8")
    assert row.candidate and row.status == "separated", row
    assert row.sweep.first_t is None and row.mesh_round.first_contact is None
    for pose, time in ((row.mesh_round.start, 0), (row.mesh_round.end, 1)):
        assert pose.relation == "separated" and pose.axis_gap_m >= target-2e-12
        check_materials(row, pose, mesh, capsule, paths, time)


@pytest.mark.parametrize("oblique", [False, True])
def test_b04_true_axis_change_and_independent_finite_segment_reference(oblique, tmp_path):
    from math import asin
    from rotating_capsule_mesh_cases import B04, b04_oblique_moving, first_endcap_reference, mesh_distance_reference

    case = b04_oblique_moving() if oblique else B04
    model, paths, mesh, capsule = public_case(case)
    start, end = (stored_pose(paths[capsule.key], t) for t in (0, 1))
    assert np.linalg.norm(np.cross(start[1]-start[0], end[1]-end[0])) > 0.1
    reference = first_endcap_reference(mesh, capsule, paths)
    if not oblique:
        analytic = asin(7/8)/paths[capsule.key].turn.angle
        assert reference[0] <= analytic <= reference[1]
    for t in (0, 1):
        assert mesh_distance_reference(mesh, capsule, paths, t)[0][0] > capsule.radius**2
    row = detect(model, paths)
    (tmp_path / f"b04-{oblique}.json").write_text(
        json.dumps(dict(case=asdict(case), independent_first=reference, actual=asdict(row))), encoding="utf8"
    )
    event = _check_first(row, reference, model, paths, mesh, capsule)
    assert row.mesh_round.start.relation == row.mesh_round.end.relation == "separated"
    for support in event.pose.supports:
        span = support.witness.capsule_axis_span
        assert span is not None and all(p.axis_parameter == pytest.approx(1.0, abs=2e-9) for p in span)
    assert 0 < row.mesh_round.axis_samples <= 512
    assert row.mesh_round.axis_speed_upper_m > 0
    assert 0 <= row.mesh_round.observed_axis_error_upper_m <= 1e-6


@pytest.mark.parametrize("name,relation", [
    ("B05_INSIDE", "mesh_contains_round"), ("B05_OPEN", "touching"), ("B05_CLOSE", "touching"),
])
def test_b05_initial_occupancy_is_first_zero_with_distinct_open_close_endpoints(name, relation):
    import rotating_capsule_mesh_cases as cases

    model, paths, mesh, capsule = public_case(getattr(cases, name))
    row = detect(model, paths)
    event = _check_first(row, 0, model, paths, mesh, capsule)
    assert event.interval == (0.0, 0.0) and event.witness_time == 0
    assert row.mesh_round.start.relation == relation
    if name == "B05_OPEN":
        assert row.mesh_round.end.relation == "separated"
    elif name == "B05_CLOSE":
        assert row.mesh_round.end.relation == "boundary_overlap"
    else:
        assert row.mesh_round.end.relation == "mesh_contains_round"


def test_b05_isolated_tangency_never_becomes_a_false_clear():
    from math import atan
    from rotating_capsule_mesh_cases import B05_TANGENT

    model, paths, mesh, capsule = public_case(B05_TANGENT)
    row = detect(model, paths)
    assert row.candidate and row.status in ("unresolved", "hit"), row
    if row.status == "unresolved":
        assert row.sweep is None and row.mesh_round is None
    else:
        _check_first(row, atan(3/4)/abs(paths[capsule.key].turn.angle), model, paths, mesh, capsule)


@pytest.mark.parametrize("kind", ["angle", "samples", "closed_boundary", "two_turns"])
def test_b05_capacity_invalid_mesh_and_unsupported_motion_are_explicit(kind, monkeypatch):
    from dataclasses import replace
    from math import pi

    model, paths, mesh, capsule = public_case(B01)
    if kind == "angle":
        paths[capsule.key] = ph.CollisionMotion(B01.endpoints, turn=replace(paths[capsule.key].turn, angle=-33.0))
    elif kind == "samples":
        from phymeo.geometry import _rotating_capsule_mesh as implementation

        monkeypatch.setattr(implementation, "_MAX_SAMPLES", 1)
    elif kind == "closed_boundary":
        broken = replace(mesh, triangles=mesh.triangles[:-1])
        model = ph.CollisionModel((broken, capsule))
    else:
        paths[mesh.key] = ph.CollisionMotion(paths[mesh.key].start, turn=ph.RigidTurn((0, 0, 0), (0, 0, 0), (1, 0, 0), pi/4))
    row = detect(model, paths)
    status, reason = {
        "angle": ("unresolved", "capsule_mesh_arc_angle_capacity"),
        "samples": ("unresolved", "capsule_mesh_arc_axis_capacity"),
        "closed_boundary": ("invalid_geometry", "mesh_open_or_misoriented_edge"),
        "two_turns": ("unsupported", "mesh_rounding_or_rotation"),
    }[kind]
    assert row.candidate and row.status == status and row.reason == reason
    assert row.sweep is None and row.mesh_round is None


@pytest.mark.parametrize("reverse,reorder,placement", [
    (True, False, "base"), (False, True, "mirror"), (True, True, "rigid"),
])
def test_b06_shape_order_material_storage_and_world_covariance(reverse, reorder, placement):
    from rotating_capsule_mesh_cases import transform

    matrix = np.diag((-1, 1, 1)) if placement == "mirror" else np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0)))
    case = B01 if placement == "base" else transform(B01, matrix, (2.25, -0.75, 0.375))
    model, paths, mesh, capsule = public_case(case, reverse=reverse, reorder=reorder)
    row = detect(model, paths)
    event = _check_first(row, b01_first(paths[capsule.key].turn.angle), model, paths, mesh, capsule)
    assert event.pose.witness.normal is not None
    expected = np.array((0.0, 1.0, 0.0))
    if placement != "base":
        expected = matrix @ expected
    if reverse:
        expected = -expected
    np.testing.assert_allclose(event.pose.witness.normal, expected, rtol=0, atol=2e-9)


def test_b06_world_keys_capacity_transaction_and_retry():
    from rotating_capsule_mesh_cases import B02

    first, pa, ma, ca = public_case(B01, world=4)
    second, pb, mb, cb = public_case(B02, world=9)
    model, paths = ph.CollisionModel(first.shapes + second.shapes), pa | pb
    with ph.CollisionWorkspace(model) as work:
        result = ph.detect(model, paths, ph.CollisionPolicy(), work)
        rows = {(row.first, row.second): row for row in result.pairs}
        assert len(result.pairs) == 6 and sum(row.reason == "different_world" for row in result.pairs) == 4
        assert rows[ma.key, ca.key].status == "hit" and rows[mb.key, cb.key].status == "separated"
        limited = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0), work)
        assert limited.status == "capacity_exceeded" and work.last_result is result
        assert not limited.pairs and not limited.bounds
        with pytest.raises(RuntimeError, match="caller abort"):
            with work.transaction():
                ph.detect(model, paths, ph.CollisionPolicy(), work)
                raise RuntimeError("caller abort")
        assert work.last_result is result
        assert ph.detect(model, paths, ph.CollisionPolicy(), work) == result
    separated_model, separated_paths, _, _ = public_case(B01, capsule_world=1)
    row = detect(separated_model, separated_paths)
    assert row.status == "filtered" and row.reason == "different_world" and not row.candidate
