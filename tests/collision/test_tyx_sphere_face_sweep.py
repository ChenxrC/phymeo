"""Pinned tyx feature proposals and host-checked face results on actual backends."""

import json
from dataclasses import asdict

import pytest
from sphere_face_cases import NAMES, authored, deforming, scene

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_device_catalog_and_deforming_face(tyx_mode, tmp_path, monkeypatch):
    shapes = []
    paths = {}
    labels = []
    for world, name in enumerate((*NAMES, "deforming")):
        sub, p, _, _ = deforming(world) if name == "deforming" else authored(name, world=world)
        shapes.extend(sub.shapes)
        paths.update(p)
        labels.append(name)
    model = ph.CollisionModel(tuple(reversed(shapes)))
    policy = ph.CollisionPolicy(margin=3.0)
    with ph.CollisionWorkspace(model) as host:
        baseline = ph.detect(model, paths, policy, host)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        out = ph.detect(model, paths, policy, work)
        assert out.pairs == baseline.pairs and out.bounds == baseline.bounds
        assert len(out.pairs) == 66 and out.required_candidates == 6
        rows = [p for p in out.pairs if p.candidate]
        assert [p.status for p in rows] == ["hit", "separated", "hit", "hit", "hit", "hit"]
        stats = work.stats()
        assert stats["surface_features"] == 42 and stats["surface_completed_launches"] >= 1
        assert stats["surface_proposed_hits"] >= 5
        small = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0, margin=3), work)
        assert small.status == "capacity_exceeded" and small.required_candidates == 6
        assert (
            work.last_result is out
            and work.stats()["surface_completed_launches"] == stats["surface_completed_launches"]
        )
        assert ph.detect(model, paths, policy, work) == out
        assert work.stats()["allocations"] == stats["allocations"]
        (tmp_path / "observation.json").write_text(
            json.dumps(
                {"name": "catalog", "labels": labels, "result": asdict(out), "stats": work.stats()},
                allow_nan=False,
            )
        )

        def injected(_features):
            raise RuntimeError("injected surface proposal failure")

        monkeypatch.setattr(work._engine, "entry_times", injected)
        with pytest.raises(RuntimeError, match="injected"):
            ph.detect(model, paths, policy, work)
        assert work.last_result == out
        with pytest.raises(RuntimeError, match="closed"):
            ph.detect(model, paths, policy, work)


def test_device_unsampled_vertex_graze(tyx_mode, tmp_path):
    model, paths, _, _ = scene(
        [[-1, 0, 0], [1, 0, 0], [0, 2, 0]],
        None,
        [-19 / 16, -1 / 4, 0.5],
        [-19 / 16, -1 / 4, -1],
        radius=5 / 16,
    )
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        (row,) = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
        assert row.status == "hit" and row.sweep.first_t[0] <= 1 / 3 <= row.sweep.first_t[1]
        assert dict(row.sweep.witness.first_weights) == {7: 1.0, 15: 0.0, 31: 0.0}
        stats = work.stats()
        assert stats["surface_proposed_hits"] == 0 and stats["surface_features"] == 7
        (tmp_path / "observation.json").write_text(
            json.dumps({"name": "vertex", "pair": asdict(row), "stats": stats}, allow_nan=False)
        )
