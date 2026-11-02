"""Actual tyx filtering and triangle/convex SAT over concave closed meshes."""

import json
from dataclasses import asdict

import pytest
from closed_mesh_cases import NAMES, authored, containing

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_device_rr05_full_matrix_and_containment(tyx_mode, tmp_path, monkeypatch):
    scenes = [authored(name, reverse=bool(i % 2), world=i)[:2] for i, name in enumerate(NAMES)]
    scenes.append(containing(world=len(scenes)))
    model = ph.CollisionModel([s for m, _ in scenes for s in m.shapes])
    paths = {k: p for _, ps in scenes for k, p in ps.items()}
    with ph.CollisionWorkspace(model) as host:
        ref = ph.detect(model, paths, ph.CollisionPolicy(), host)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        out = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert out.pairs == ref.pairs and out.bounds == ref.bounds
        rows = [p for p in out.pairs if p.first.world == p.second.world]
        assert [p.status for p in rows] == ["hit", "hit", "separated", "hit", "hit", "overlap"]
        stats = work.stats()
        assert stats["hull_completed_launches"] > 0 and stats["hull_axis_terms"] > 0
        if tyx_mode == "cuda":
            assert stats["hull_downloads"] and stats["hull_synchronizations"]
        with monkeypatch.context() as m:

            def fault(*args):
                raise RuntimeError("mesh projection fault")

            m.setattr(work._engine, "hull_intervals", fault)
            with pytest.raises(RuntimeError, match="mesh projection fault"):
                ph.detect(model, paths, ph.CollisionPolicy(), work)
            assert work.last_result is out
        (tmp_path / "rr05.json").write_text(
            json.dumps({"rows": [asdict(p) for p in rows], "stats": stats})
        )
