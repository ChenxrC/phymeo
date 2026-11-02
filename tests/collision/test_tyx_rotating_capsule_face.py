"""Actual device proposals and full public witnesses along the original angular arc."""

import json
from dataclasses import asdict

import pytest
from rotating_face_cases import switching_segment, turning_face

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_device_angular_face_motion_and_history(tyx_mode, tmp_path):
    scenes = [
        turning_face(world=0),
        turning_face(clear=True, world=1),
        turning_face(0.53, 0.55, swap=True, world=2, change_frame=True),
    ]
    segment_model, segment_paths = switching_segment(0, 0.5, world=3)
    scenes.append((segment_model, segment_paths, None, None))
    model = ph.CollisionModel([s for m, _, _, _ in scenes for s in m.shapes])
    paths = {k: p for _, ps, _, _ in scenes for k, p in ps.items()}
    with ph.CollisionWorkspace(model) as ref:
        expected = ph.detect(model, paths, ph.CollisionPolicy(), ref)
    with ph.CollisionWorkspace(
        model, tyx_mode, broad_phase="bvh", cache_root=tmp_path / "cache"
    ) as work:
        out = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert (out.bounds, out.pairs) == (expected.bounds, expected.pairs)
        hits = [p for p in out.pairs if p.first.world == p.second.world]
        assert [p.status for p in hits] == ["hit", "separated", "hit", "hit"]
        assert tuple(x.axis_parameter for x in hits[-1].witness.capsule_axis_span) == (0.0, 1.0)
        stats = work.stats()
        assert stats["surface_features"] > 0 and stats["surface_proposed_hits"] > 0
        assert stats["spatial_completed_launches"] == 2
        if tyx_mode == "cuda":
            assert stats["sweep_downloads"] and stats["spatial_downloads"]
        (tmp_path / "angular-face.json").write_text(
            json.dumps(dict(rows=[asdict(h) for h in hits], stats=stats), allow_nan=False)
        )
