"""One mixed-device submission exercises RD05 material witnesses and area."""

import json
from dataclasses import asdict

import pytest
from hull_surface_cases import NAMES, authored, changing_face, wide_face

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_device_hull_deforming_face_matrix(tyx_mode, tmp_path, monkeypatch):
    scenes = [authored(name, reverse=bool(i % 2), world=i)[:2] for i, name in enumerate(NAMES)]
    scenes.append(wide_face(world=len(scenes)))
    scenes.append(changing_face(world=len(scenes)))
    model = ph.CollisionModel([s for m, _ in scenes for s in m.shapes])
    paths = {k: p for _, ps in scenes for k, p in ps.items()}
    with ph.CollisionWorkspace(model) as host:
        ref = ph.detect(model, paths, ph.CollisionPolicy(), host)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as device:
        out = ph.detect(model, paths, ph.CollisionPolicy(), device)
        assert out.bounds == ref.bounds and out.pairs == ref.pairs
        rows = [p for p in out.pairs if p.first.world == p.second.world]
        assert [p.status for p in rows] == [
            "hit",
            "separated",
            "separated",
            "hit",
            "hit",
            "hit",
            "hit",
        ]
        stats = device.stats()
        assert stats["triangle_completed_launches"] and stats["triangle_proposed_hits"]
        assert stats["triangle_edge_edge_features"]
        if tyx_mode == "cuda":
            assert stats["sweep_synchronizations"] and stats["sweep_downloads"]

        def fault(_):
            raise RuntimeError("surface proposal fault")

        with monkeypatch.context() as m:
            m.setattr(device._engine, "entry_times", fault)
            with pytest.raises(RuntimeError, match="surface proposal fault"):
                ph.detect(model, paths, ph.CollisionPolicy(), device)
            assert device.last_result is out
        (tmp_path / "rd05.json").write_text(
            json.dumps({"rows": [asdict(p) for p in rows], "stats": stats})
        )
