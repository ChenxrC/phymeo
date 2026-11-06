"""Real TYX face traversal and public readback-failure transaction boundaries."""

import json
from dataclasses import asdict

import numpy as np
import pytest
from mesh_round_candidate_cases import (
    CLEAR_SWEEP, CROSSING, FACE, INSIDE, NOTCH, SHAFT, THROUGH,
    build, check_materials, record_facets, required_faces,
)

import phymeo as ph
from phymeo.core.capabilities import SolverFailure
from phymeo.geometry._collision_bvh import BoundsTree


def test_public_face_readback_failure_is_atomic_and_closes_workspace(monkeypatch):
    """A backend validation error must not become a newly published narrow unknown."""
    model, paths, _, _ = build(FACE)
    with ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        previous = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert previous.pairs[0].mesh_round.end.axis_gap_m == -1 / 32
        assert work.stats()["mesh_bvh_refits"] > 0
        before = asdict(previous)
        layouts = work._trees.layouts.copy()
        called = []

        def broken_readback(*args):
            called.append(args)
            raise SolverFailure("invalid BVH output")

        monkeypatch.setattr(work._trees, "facets", broken_readback)
        with pytest.raises(RuntimeError, match="face"):
            ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert called and work.last_result is previous
        assert asdict(previous) == before and work._trees.layouts == layouts
        with pytest.raises(RuntimeError, match="closed"):
            ph.detect(model, paths, ph.CollisionPolicy(), work)


@pytest.mark.tyx
def test_device_faces_match_full_output_prune_and_isolate_world_updates(tyx_mode, tmp_path, monkeypatch):
    shapes, paths, owners = [], {}, []
    for index, case in enumerate((SHAFT, INSIDE, CROSSING, THROUGH, NOTCH, CLEAR_SWEEP)):
        local, motion, mesh, rounded = build(
            case, dense=index == 0, world=17 + index, reverse=bool(index % 2),
            reorder=bool(index % 2), variant="rigid" if index == 4 else "identity",
        )
        shapes.extend(local.shapes)
        paths.update(motion)
        owners.append((case, mesh, rounded))
    model = ph.CollisionModel(tuple(reversed(shapes)))
    policy = ph.CollisionPolicy(margin=1 / 16)
    with ph.CollisionWorkspace(model) as host:
        host_result = ph.detect(model, paths, policy, host)
    assert host_result.required_candidates == 6

    def forbidden(*args):
        raise AssertionError("selected TYX face query used HOST BVH traversal")

    monkeypatch.setattr(BoundsTree, "host_hits", forbidden)
    with (
        ph.CollisionWorkspace(model, tyx_mode, broad_phase="bvh", cache_root=tmp_path / "full") as full,
        ph.CollisionWorkspace(model, tyx_mode, broad_phase="bvh", mesh_midphase="bvh",
                              cache_root=tmp_path / "selected") as work,
    ):
        expected = ph.detect(model, paths, policy, full)
        calls = record_facets(work, monkeypatch)
        actual = ph.detect(model, paths, policy, work)
        assert asdict(actual) == asdict(expected)
        assert actual.backend == tyx_mode and actual.status == "ok"
        assert actual.pairs == host_result.pairs and actual.bounds == host_result.bounds
        assert sum(row.reason == "different_world" for row in actual.pairs) == 60
        rows = [row for row in actual.pairs if row.first.world == row.second.world]
        assert [row.status for row in rows] == ["overlap", "overlap", "overlap", "hit", "separated", "separated"]
        for row, (case, mesh, rounded) in zip(rows, owners, strict=True):
            trace = row.mesh_round
            check_materials(row, trace.start, mesh, rounded, paths)
            check_materials(row, trace.end, mesh, rounded, paths, 1)
            if case.first_t is not None:
                low, high = row.sweep.first_t
                assert low <= case.first_t <= high and high - low <= 2**-38
                event = trace.first_contact
                check_materials(row, event.pose, mesh, rounded, paths, event.witness_time)
            else:
                assert trace.first_contact is None
            if not paths[mesh.key].moving and not paths[rounded.key].moving:
                relevant = [call for call in calls if call["key"][1].key == mesh.key]
                assert len(relevant) == 1
                assert set(required_faces(mesh, rounded, paths)) <= set(relevant[0]["rows"][0])
            assert trace.end.witness.source == "host-closed-mesh-round"
        assert calls and all(call["driver"] is work._engine for call in calls)
        assert any(0 < len(call["rows"][0]) < len(call["leaves"]) == 108 for call in calls)
        assert any(not call["rows"][0] for call in calls)
        for call in calls:
            assert call["leaves"].dtype == call["queries"].dtype == np.float32
            assert call["queries"].shape == (1, 6)
        initial_stats = work.stats()
        assert initial_stats["mesh_bvh_builds"] == len(owners)
        assert initial_stats["mesh_bvh_refits"] == len(calls)
        assert initial_stats["spatial_completed_launches"] == 2 * (
            initial_stats["mesh_bvh_refits"] + initial_stats["shape_bvh_refits"])
        assert initial_stats["mesh_bvh_node_visits"] > 0
        if tyx_mode == "cuda":
            assert initial_stats["spatial_uploads"] > 0 and initial_stats["spatial_downloads"] > 0
            assert initial_stats["spatial_synchronizations"] > 0
            assert initial_stats["spatial_uploaded_bytes"] > 0 and initial_stats["spatial_downloaded_bytes"] > 0
        before, layouts, count = asdict(actual), work._trees.layouts.copy(), len(calls)
        rejected = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0, margin=1 / 16), work)
        assert rejected.status == "capacity_exceeded" and rejected.required_candidates == 6
        assert work.last_result is actual and len(calls) == count
        assert work._trees.layouts == layouts and asdict(actual) == before
        assert asdict(ph.detect(model, paths, policy, work)) == before

        # Same registered IDs and revisions, but different actual stored geometry
        # in one world. Cached tree layout must be refitted, not used as occupancy.
        _, mesh, _ = owners[0]
        changed = {**paths, mesh.key: ph.CollisionMotion(paths[mesh.key].start - (1 / 64, 0, 0))}
        changed_expected = ph.detect(model, changed, policy, full)
        changed_actual = ph.detect(model, changed, policy, work)
        assert asdict(changed_actual) == asdict(changed_expected)
        changed_rows = [row for row in changed_actual.pairs if row.first.world == row.second.world]
        assert changed_rows[0].mesh_round.end.axis_gap_m == -1 / 64
        assert changed_rows[1:] == rows[1:]
        assert work.stats()["mesh_bvh_builds"] == len(owners)
        first_calls = [call for call in calls if call["key"][1].key == mesh.key]
        assert first_calls[0]["key"] == first_calls[-1]["key"]
        assert not np.array_equal(first_calls[0]["leaves"], first_calls[-1]["leaves"])
        assert asdict(actual) == before
        (tmp_path / "mesh-round-device-candidates.json").write_text(json.dumps(dict(
            backend=tyx_mode, expected=asdict(expected), actual=before,
            changed_expected=asdict(changed_expected), changed_actual=asdict(changed_actual),
            initial_stats=initial_stats, final_stats=work.stats(),
            face_calls=[dict(world=call["key"][1].key.world, leaves=len(call["leaves"]),
                             selected=call["rows"]) for call in calls],
        ), allow_nan=False), encoding="utf8")


@pytest.mark.tyx
@pytest.mark.parametrize("fault", ("exception", "bad_shape", "bad_bit"))
def test_device_face_failures_preserve_publication_and_fresh_retry(tyx_mode, tmp_path, monkeypatch, fault):
    model, paths, _, _ = build(FACE)
    policy = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model, tyx_mode, mesh_midphase="bvh", cache_root=tmp_path / "cache") as work:
        previous = ph.detect(model, paths, policy, work)
        before, layouts = asdict(previous), work._trees.layouts.copy()
        assert work.stats()["spatial_completed_launches"] == 2
        original = work._engine.tree_hits
        called = []

        def damaged(key, tree, leaves, queries, params, shape):
            assert not shape, "this failure must occur at the per-mesh query"
            called.append(key)
            if fault == "exception":
                raise RuntimeError("injected device face failure")
            raw = original(key, tree, leaves, queries, params, shape)
            if fault == "bad_shape":
                return raw[:, :-1]
            raw = raw.copy()
            raw[0, 0] |= 1 << len(leaves)
            return raw

        monkeypatch.setattr(work._engine, "tree_hits", damaged)
        with pytest.raises(RuntimeError, match="face"):
            ph.detect(model, paths, policy, work)
        assert called and work.last_result is previous
        assert work._trees.layouts == layouts and asdict(previous) == before
        with pytest.raises(RuntimeError, match="closed"):
            ph.detect(model, paths, policy, work)
    # A failed device workspace stays closed. Rebuild the public owner and retry
    # the unchanged input; never reset a poisoned engine or reuse partial output.
    with ph.CollisionWorkspace(model, tyx_mode, mesh_midphase="bvh", cache_root=tmp_path / "retry") as fresh:
        restored = ph.detect(model, paths, policy, fresh)
        assert asdict(restored) == before
        assert fresh.stats()["spatial_completed_launches"] == 2
