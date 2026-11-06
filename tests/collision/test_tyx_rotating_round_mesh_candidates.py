"""Actual TYX traversal for rotating samples; exact distance and CCD remain HOST."""

from dataclasses import asdict

import pytest
from dual_turn_mesh_round_cases import B03, public_case, root_interval
from mesh_round_candidate_cases import record_facets
from test_dual_turn_mesh_round import check_entry
from test_rotating_round_mesh_candidates import assert_face_calls

import phymeo as ph
from phymeo.geometry._collision_bvh import BoundsTree

pytestmark = pytest.mark.tyx


def test_device_turning_pair_really_traverses_faces(tyx_mode, tmp_path, monkeypatch):
    model, paths, mesh, rounded = public_case(B03)
    policy = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model) as host:
        expected = ph.detect(model, paths, policy, host)
    check_entry(expected.pairs[0], root_interval(B03), mesh, rounded, paths)

    def forbidden(*args, **kwargs):
        raise AssertionError("TYX candidate query used HOST BVH traversal")

    monkeypatch.setattr(BoundsTree, "host_hits", forbidden)
    with ph.CollisionWorkspace(model, tyx_mode, mesh_midphase="bvh", cache_root=tmp_path / "cache") as work:
        calls = record_facets(work, monkeypatch)
        actual = ph.detect(model, paths, policy, work)
        assert actual.pairs == expected.pairs and actual.bounds == expected.bounds
        assert actual.backend == tyx_mode and actual.status == "ok"
        assert_face_calls(calls, mesh, work._engine)
        stats = work.stats()
        assert stats["mesh_bvh_refits"] == len(calls)
        assert stats["spatial_completed_launches"] == 2 * len(calls)
        if tyx_mode == "cuda":
            assert stats["spatial_uploads"] > 0 and stats["spatial_downloads"] > 0
            assert stats["spatial_synchronizations"] > 0
        assert asdict(actual.pairs[0]) == asdict(expected.pairs[0])
