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


def test_device_mixed_rotations_prune_and_isolate_repeated_local_ids(tyx_mode, tmp_path, monkeypatch):
    import json
    from dual_turn_mesh_round_cases import B02, B06, B07
    from test_rotating_round_mesh_candidates import assert_trace_materials, single_case

    registrations = [single_case("capsule", world=503),
                     single_case("mesh_sphere", dense=True, world=91)]
    registrations.extend(public_case(case, world=world, reverse=True, reorder=True)
                         for case, world in ((B02, 17), (B06, 88), (B07, 302)))
    shapes, paths, owners = [], {}, {}
    for local, motion, mesh, rounded in registrations:
        shapes.extend(local.shapes)
        paths.update(motion)
        owners[mesh.key.world] = mesh, rounded
    model = ph.CollisionModel(tuple(reversed(shapes)))
    policy = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model) as host:
        host_result = ph.detect(model, paths, policy, host)
    assert host_result.required_candidates == 5

    def forbidden(*args, **kwargs):
        raise AssertionError("actual device mesh candidates used HOST tree traversal")

    monkeypatch.setattr(BoundsTree, "host_hits", forbidden)
    with (
        ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "all") as full,
        ph.CollisionWorkspace(model, tyx_mode, mesh_midphase="bvh", cache_root=tmp_path / "selected") as work,
    ):
        expected = ph.detect(model, paths, policy, full)
        calls = record_facets(work, monkeypatch)
        actual = ph.detect(model, paths, policy, work)
        assert asdict(actual) == asdict(expected)
        assert actual.pairs == host_result.pairs and actual.bounds == host_result.bounds
        assert sum(row.reason == "different_world" for row in actual.pairs) == 40
        by_world = {row.first.world: row for row in actual.pairs if row.first.world == row.second.world}
        assert {world: row.status for world, row in by_world.items()} == {
            503: "hit", 91: "hit", 17: "hit", 88: "separated", 302: "hit",
        }
        for world, (mesh, rounded) in owners.items():
            local_calls = [call for call in calls if call["key"][1].key == mesh.key]
            assert_face_calls(local_calls, mesh, work._engine)
            assert_trace_materials(by_world[world], mesh, rounded, paths)
            assert by_world[world].mesh_round.end.witness.source == "host-closed-mesh-round"
        assert any(0 < len(call["rows"][0]) < len(call["leaves"]) == 108 for call in calls)
        stats = work.stats()
        assert stats["mesh_bvh_builds"] == len(owners)
        assert stats["mesh_bvh_refits"] == len(calls)
        assert stats["spatial_completed_launches"] == 2 * len(calls)
        if tyx_mode == "cuda":
            assert stats["spatial_uploaded_bytes"] > 0 and stats["spatial_downloaded_bytes"] > 0
            assert stats["spatial_synchronizations"] >= len(calls)
        before, count, layouts = asdict(actual), len(calls), work._trees.layouts.copy()
        rejected = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0), work)
        assert rejected.status == "capacity_exceeded" and rejected.required_candidates == 5
        assert work.last_result is actual and len(calls) == count and work._trees.layouts == layouts
        assert asdict(ph.detect(model, paths, policy, work)) == before
        assert asdict(actual) == before
        (tmp_path / "rotating-face-worlds.json").write_text(json.dumps(dict(
            backend=tyx_mode, expected=asdict(expected), actual=before, stats=stats,
            final_stats=work.stats(),
            calls=[dict(key=asdict(call["key"][1].key), faces=len(call["leaves"]), selected=call["rows"])
                   for call in calls[:count]],
        ), allow_nan=False), encoding="utf8")


@pytest.mark.parametrize("fault", ("exception", "bad_shape", "bad_bit"))
def test_device_rotating_face_failure_closes_without_partial_publication(tyx_mode, tmp_path, monkeypatch, fault):
    from dual_turn_mesh_round_cases import B07

    model, paths, _, _ = public_case(B07)
    policy = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model, tyx_mode, mesh_midphase="bvh", cache_root=tmp_path / "cache") as work:
        previous = ph.detect(model, paths, policy, work)
        assert previous.pairs[0].sweep.first_t == (0, 0)
        before, layouts = asdict(previous), work._trees.layouts.copy()
        original = work._engine.tree_hits
        calls = []

        def damaged(key, tree, leaves, queries, params, shape):
            assert not shape
            calls.append(key)
            if fault == "exception":
                raise RuntimeError("injected rotating face device failure")
            raw = original(key, tree, leaves, queries, params, shape)
            if fault == "bad_shape":
                return raw[:, :-1]
            result = raw.copy()
            result[0, 0] |= 1 << len(leaves)
            return result

        monkeypatch.setattr(work._engine, "tree_hits", damaged)
        with pytest.raises(RuntimeError, match="face"):
            ph.detect(model, paths, policy, work)
        assert calls and work.last_result is previous
        assert asdict(previous) == before and work._trees.layouts == layouts
        with pytest.raises(RuntimeError, match="closed"):
            ph.detect(model, paths, policy, work)
    with ph.CollisionWorkspace(model, tyx_mode, mesh_midphase="bvh", cache_root=tmp_path / "rebuild") as fresh:
        restored = ph.detect(model, paths, policy, fresh)
        assert asdict(restored) == before and fresh.stats()["spatial_completed_launches"] > 0
