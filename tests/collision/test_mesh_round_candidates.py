"""B3 public mechanism RED and complete-result controls for finite face selection."""

from fractions import Fraction as F

import pytest
from mesh_round_candidate_cases import FACE, build, check_materials, face_distances, five_fields

import phymeo as ph


def test_public_bvh_uses_mesh_round_faces_after_correct_static_witness(monkeypatch):
    """Removing the round-face midphase must fail after the real physics is checked."""
    model, paths, mesh, round_shape = build(FACE)
    assert min(face_distances(mesh, round_shape, paths)) == F(9, 1024)
    with ph.CollisionWorkspace(model) as baseline:
        expected = ph.detect(model, paths, ph.CollisionPolicy(), baseline)
    (prior,) = expected.pairs
    assert prior.status == "overlap" and prior.mesh_round.end.relation == "boundary_overlap"
    assert prior.mesh_round.end.axis_gap_m == -1 / 32
    check_materials(prior, prior.mesh_round.end, mesh, round_shape, paths)
    calls = []
    with ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        original = work._trees.facets

        def observed(pending, key, leaves, queries, driver):
            result = original(pending, key, leaves, queries, driver)
            calls.append((len(leaves), tuple(tuple(row) for row in result), driver))
            return result

        monkeypatch.setattr(work._trees, "facets", observed)
        actual = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert actual == expected
        assert five_fields(actual.pairs[0]) == five_fields(prior)
        check_materials(actual.pairs[0], actual.pairs[0].mesh_round.end, mesh, round_shape, paths)
        assert calls, "public mesh_midphase='bvh' still never dispatches mesh-round face candidates"
        assert all(driver is None for _, _, driver in calls)
        assert any(0 < len(row) < total for total, rows, _ in calls for row in rows)
        assert work.stats()["mesh_bvh_refits"] == len(calls)
