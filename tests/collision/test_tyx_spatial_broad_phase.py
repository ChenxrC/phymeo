"""Actual tyx sweep traversal, pair bits, capacity and publication."""

import json

import pytest
from m1_cases import mixed_scene
from test_spatial_broad_phase import balls

import phymeo as ph

pytestmark = pytest.mark.tyx


@pytest.mark.parametrize("scene", ["sparse", "dense", "mixed"])
def test_native_sweep_matches_exhaustive(tyx_mode, tmp_path, scene):
    if scene == "mixed":
        model, paths, _, _ = mixed_scene()
    else:
        model, paths = balls(65 if scene == "sparse" else 34, scene == "dense")
    rule = ph.CollisionPolicy(margin=0.125)
    with ph.CollisionWorkspace(model) as oracle:
        expected = ph.detect(model, paths, rule, oracle)
    with ph.CollisionWorkspace(
        model, tyx_mode, broad_phase="sap", cache_root=tmp_path / "cache"
    ) as work:
        out = ph.detect(model, paths, rule, work)
        assert (out.bounds, out.pairs) == (expected.bounds, expected.pairs)
        stats = work.stats()
        assert stats["spatial_completed_launches"] == 1
        if scene == "sparse":
            assert stats["sap_window_pairs"] == stats["sap_box_tests"] == 0
        if tyx_mode == "cuda":
            assert stats["spatial_uploads"] and stats["spatial_downloads"]
        (tmp_path / "spatial.json").write_text(json.dumps(stats, allow_nan=False))
