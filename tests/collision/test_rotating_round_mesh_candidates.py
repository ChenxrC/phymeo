"""Rotating mesh-round face selection, with existing independent path oracles."""

from dataclasses import asdict

import pytest
from dual_turn_mesh_round_cases import B01, B02, B03, B04, public_case, root_interval, static_paths
from mesh_round_candidate_cases import record_facets
from test_dual_turn_mesh_round import check_entry, check_materials

import phymeo as ph


def assert_face_calls(calls, mesh, driver):
    assert calls, "rotating mesh-round branch never queried per-face candidates"
    for call in calls:
        binding = call["key"][1]
        assert binding.key == mesh.key
        assert binding.faces == mesh.triangles and binding.vertex_ids == mesh.vertex_ids
        assert call["driver"] is driver
        assert len(call["rows"]) == 1 and call["rows"][0]
        assert call["rows"][0] == tuple(sorted(set(call["rows"][0])))
        assert all(0 <= index < len(mesh.triangles) for index in call["rows"][0])


def test_turning_pair_queries_real_face_candidates_after_independent_hit(monkeypatch):
    model, paths, mesh, capsule = public_case(B01)
    policy = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model) as full:
        static = ph.detect(model, static_paths(paths, 3 / 8), policy, full).pairs[0]
        assert static.status == "overlap" and static.mesh_round.start.axis_gap_m < 0
        expected = ph.detect(model, paths, policy, full)
    reference = root_interval(B01)
    check_entry(expected.pairs[0], reference, mesh, capsule, paths)
    assert expected.pairs[0].mesh_round.start.relation == "separated"
    assert expected.pairs[0].mesh_round.end.relation == "separated"
    with ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        calls = record_facets(work, monkeypatch)
        actual = ph.detect(model, paths, policy, work)
        assert asdict(actual) == asdict(expected)
        check_entry(actual.pairs[0], reference, mesh, capsule, paths)
        assert_face_calls(calls, mesh, None)


@pytest.mark.parametrize("case", (B02, B03, B04), ids=lambda case: case.name)
def test_dual_turn_sphere_pivot_and_shaft_use_original_material_faces(case, monkeypatch):
    model, paths, mesh, rounded = public_case(case)
    policy = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model) as full:
        expected = ph.detect(model, paths, policy, full)
    with ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        calls = record_facets(work, monkeypatch)
        actual = ph.detect(model, paths, policy, work)
        assert asdict(actual) == asdict(expected)
        event = check_entry(actual.pairs[0], root_interval(case), mesh, rounded, paths)
        if case is B04:
            spans = check_materials(actual.pairs[0], event.pose, mesh, rounded, paths, event.witness_time)
            assert spans and all(0 < low <= high < 1 for low, high in spans)
        assert_face_calls(calls, mesh, None)
