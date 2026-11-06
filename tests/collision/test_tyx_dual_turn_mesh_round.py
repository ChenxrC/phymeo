"""Actual TYX swept broad phase with HOST dual-turn geometry; no GPU CCD claim."""

import json
from dataclasses import asdict

import pytest
from dual_turn_mesh_round_cases import B01, B02, B03, B06, public_case, root_interval
from test_dual_turn_mesh_round import check_entry, check_materials

import phymeo as ph
from phymeo.geometry import collision_pipeline

pytestmark = pytest.mark.tyx


def test_dual_turn_device_worlds_match_all_host_fields_and_recover(tyx_mode, tmp_path, monkeypatch):
    shapes, paths, owners = [], {}, []
    for index, case in enumerate((B01, B02, B03, B06)):
        model, local, mesh, round_shape = public_case(
            case, world=17 + index, reverse=bool(index % 2)
        )
        shapes.extend(model.shapes)
        paths.update(local)
        owners.append((case, mesh, round_shape))
    model = ph.CollisionModel(tuple(reversed(shapes)))
    with ph.CollisionWorkspace(model, broad_phase="bvh") as host:
        expected = ph.detect(model, paths, ph.CollisionPolicy(), host)
    assert expected.status == "ok"

    def forbidden(*args, **kwargs):
        raise AssertionError("selected TYX call used HOST broad phase")

    monkeypatch.setattr(collision_pipeline, "_host_query", forbidden)
    with ph.CollisionWorkspace(
        model, tyx_mode, broad_phase="bvh", cache_root=tmp_path / "cache"
    ) as work:
        actual = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert actual.status == "ok" and actual.backend == tyx_mode
        assert actual.pairs == expected.pairs and actual.bounds == expected.bounds
        rows = [row for row in actual.pairs if row.first.world == row.second.world]
        assert [row.status for row in rows] == ["hit", "hit", "hit", "separated"]
        assert sum(row.reason == "different_world" for row in actual.pairs) == 24
        for row, (case, mesh, round_shape) in zip(rows, owners, strict=True):
            if case is not B06:
                event = check_entry(row, root_interval(case), mesh, round_shape, paths)
                assert event.pose.witness.source == "host-closed-mesh-round"
            else:
                assert row.sweep.first_t is None
                check_materials(row, row.mesh_round.start, mesh, round_shape, paths, 0)
                check_materials(row, row.mesh_round.end, mesh, round_shape, paths, 1)
        stats = work.stats()
        assert stats["completed_launches"] > 0
        if tyx_mode == "cuda":
            assert stats["downloads"] > 0 and stats["synchronizations"] > 0
        previous = asdict(actual)
        rejected = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0), work)
        assert rejected.status == "capacity_exceeded" and work.last_result is actual
        assert asdict(actual) == previous
        recovered = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert (
            recovered == actual and work.stats()["completed_launches"] > stats["completed_launches"]
        )
        (tmp_path / "dual-turn-device.json").write_text(
            json.dumps(
                dict(
                    backend=tyx_mode,
                    expected=asdict(expected),
                    actual=asdict(actual),
                    stats=work.stats(),
                )
            ),
            encoding="utf8",
        )

        def failed_query(*args, **kwargs):
            raise RuntimeError("dual-turn injected device failure")

        monkeypatch.setattr(work._engine, "query", failed_query)
        with pytest.raises(RuntimeError, match="injected device failure"):
            ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert work.last_result is recovered and asdict(actual) == previous
        with pytest.raises(RuntimeError, match="closed"):
            ph.detect(model, paths, ph.CollisionPolicy(), work)
