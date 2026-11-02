"""tyx proposals and same-query material patches for all registered plate cases."""

import json
from dataclasses import asdict

import pytest
from rounded_plate_cases import NAMES, authored, edge_case

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_device_plate_matrix(tyx_mode, tmp_path):
    scenes = [authored(name, reverse=bool(i % 2), world=i)[:2] for i, name in enumerate(NAMES)]
    scenes.extend([edge_case(len(scenes)), edge_case(len(scenes) + 1, parallel=True)])
    model = ph.CollisionModel([s for m, _ in scenes for s in m.shapes])
    paths = {k: p for _, ps in scenes for k, p in ps.items()}
    policy = ph.CollisionPolicy(margin=0.5)
    with ph.CollisionWorkspace(model) as host:
        ref = ph.detect(model, paths, policy, host)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as device:
        out = ph.detect(model, paths, policy, device)
        assert out.pairs == ref.pairs and out.bounds == ref.bounds
        rows = [p for p in out.pairs if p.family == "RD" and p.candidate]
        assert [p.status for p in rows] == ["separated"] * 4 + [
            "hit",
            "separated",
            "hit",
            "overlap",
            "hit",
            "touching",
        ]
        stats = device.stats()
        assert stats["surface_completed_launches"] and stats["surface_proposed_hits"]
        assert stats["capsule_shaft_edge_features"] > 0
        if tyx_mode == "cuda":
            assert stats["sweep_synchronizations"] and stats["sweep_downloads"]
        (tmp_path / "rd03.json").write_text(
            json.dumps({"rows": [asdict(p) for p in rows], "stats": stats})
        )
