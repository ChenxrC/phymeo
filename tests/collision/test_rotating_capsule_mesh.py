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
