"""Inspect the actual TYX terrain face selector and the exact roots consuming its output."""

import json
from fractions import Fraction

import numpy as np
import pytest
from test_heightfield_sphere import scene

import phymeo as ph


pytestmark = pytest.mark.tyx


def test_device_selects_terrain_faces_before_exact_ridge_roots(tyx_mode, tmp_path):
    grid = ph.Heightfield([[0,1,0]]*3, (1,1), origin=(-1,-1,0))
    start, end = [-0.75,0.125,1.25], [0.75,0.125,1.25]
    model, paths, _ = scene(grid,start,5/16,end=end)
    with ph.CollisionWorkspace(model,tyx_mode,cache_root=tmp_path/"cache") as work:
        (row,) = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
        assert row.status == "hit" and row.sweep.first_t == (3/8, 3/8)
        assert row.heightfield.last_contact.interval == (5/8, 5/8)
        assert row.sweep.witness.source == "host-heightfield"
        stats = work.stats()
        assert stats["heightfield_completed_launches"] == 1
        assert stats["heightfield_cells_tested"] == 4 and stats["heightfield_faces_kept"] > 0
        if tyx_mode == "cuda":
            assert stats["heightfield_downloads"] == stats["heightfield_synchronizations"] == 1
        (tmp_path/"terrain-ridge.json").write_text(json.dumps({
            "backend": row.sweep.witness.source, "first": row.sweep.first_t,
            "last": row.heightfield.last_contact.interval, "stats": stats
        },allow_nan=False),encoding="utf-8")
