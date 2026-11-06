"""Real TYX swept broad phase followed by HOST-certified rotating-capsule material."""

import json
from dataclasses import asdict

import pytest
from rotating_capsule_mesh_cases import B01, B02, b04_oblique_moving, public_case
from test_rotating_capsule_mesh import check_materials

import phymeo as ph
from phymeo.geometry import collision_pipeline

pytestmark = pytest.mark.tyx


def test_device_capsule_mesh_multiple_worlds_match_full_host_and_restore(tyx_mode, tmp_path, monkeypatch):
    shapes, paths, owners = [], {}, []
    for world, case in enumerate((B01, B02, b04_oblique_moving())):
        model, local, mesh, capsule = public_case(case, world=world+11, reverse=bool(world % 2))
        shapes.extend(model.shapes)
        paths.update(local)
        owners.append((mesh, capsule))
    model = ph.CollisionModel(tuple(reversed(shapes)))
    with ph.CollisionWorkspace(model, broad_phase="bvh") as host:
        expected = ph.detect(model, paths, ph.CollisionPolicy(), host)

    def forbidden(*args, **kwargs):
        raise AssertionError("selected device silently used HOST broad phase")

    monkeypatch.setattr(collision_pipeline, "_host_query", forbidden)
    with ph.CollisionWorkspace(model, tyx_mode, broad_phase="bvh", cache_root=tmp_path / "cache") as work:
        actual = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert actual.backend == tyx_mode and actual.status == "ok"
        assert actual.pairs == expected.pairs and actual.bounds == expected.bounds
        rows = [row for row in actual.pairs if row.first.world == row.second.world]
        assert [row.status for row in rows] == ["hit", "separated", "hit"]
        for row, (mesh, capsule) in zip(rows, owners, strict=True):
            trace = row.mesh_round
            check_materials(row, trace.start, mesh, capsule, paths, 0)
            check_materials(row, trace.end, mesh, capsule, paths, 1)
            if trace.first_contact is not None:
                event = trace.first_contact
                check_materials(row, event.pose, mesh, capsule, paths, event.witness_time)
                assert event.pose.witness.source == "host-closed-mesh-round"
        stats = work.stats()
        assert stats["completed_launches"] > 0
        if tyx_mode == "cuda":
            assert stats["downloads"] > 0 and stats["synchronizations"] > 0
        exceeded = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0), work)
        assert exceeded.status == "capacity_exceeded" and work.last_result is actual
        restored = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert restored == actual
        assert work.stats()["completed_launches"] > stats["completed_launches"]
        (tmp_path / "rotating-capsule-mesh-device.json").write_text(json.dumps(
            dict(backend=tyx_mode, expected=asdict(expected), actual=asdict(actual), stats=work.stats())
        ), encoding="utf8")

        def failure(*args, **kwargs):
            raise RuntimeError("injected device classification failure")

        monkeypatch.setattr(work._engine, "query", failure)
        with pytest.raises(RuntimeError, match="injected device"):
            ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert work.last_result is restored
        with pytest.raises(RuntimeError, match="closed"):
            ph.detect(model, paths, ph.CollisionPolicy(), work)
