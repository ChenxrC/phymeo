"""Run true node refit and traversal on the selected tyx backend."""

import json

import pytest
from spatial_index_cases import mesh_scene, sheet

import phymeo as ph

pytestmark = pytest.mark.tyx


@pytest.mark.parametrize("surface", [False, True])
def test_spatial_device_tree_and_mesh_parity(tyx_mode, tmp_path, monkeypatch, surface):
    model, frames = mesh_scene(False) if surface else sheet(3)
    with ph.CollisionWorkspace(model) as brute:
        refs = [ph.detect(model, p, ph.CollisionPolicy(), brute) for p in frames]
    from phymeo.geometry._collision_bvh import BoundsTree

    def forbidden(*args):
        raise AssertionError("native path used host traversal")

    monkeypatch.setattr(BoundsTree, "host_hits", forbidden)
    with ph.CollisionWorkspace(
        model, tyx_mode, broad_phase="bvh", mesh_midphase="bvh", cache_root=tmp_path / "cache"
    ) as work:
        for p, ref in zip(frames, refs, strict=True):
            out = ph.detect(model, p, ph.CollisionPolicy(), work)
            assert (out.pairs, out.bounds) == (ref.pairs, ref.bounds)
        s = work.stats()
        assert s["shape_bvh_builds"] == 1 and s["shape_bvh_refits"] == 3
        assert s["spatial_completed_launches"] == (12 if surface else 6)
        if surface:
            assert (
                s["mesh_bvh_refits"] == 3
                and s["mesh_facet_pairs_tested"] < s["mesh_facet_pairs_total"]
            )
        if tyx_mode == "cuda":
            assert s["spatial_downloads"] and s["spatial_uploads"]
        (tmp_path / "bvh.json").write_text(json.dumps(s, allow_nan=False))


def test_device_traversal_failure_preserves_last_publication(tyx_mode, tmp_path, monkeypatch):
    model, frames = mesh_scene(False)
    with ph.CollisionWorkspace(
        model, tyx_mode, broad_phase="sap", mesh_midphase="bvh", cache_root=tmp_path / "cache"
    ) as work:
        first = ph.detect(model, frames[0], ph.CollisionPolicy(), work)

        def fault(*args):
            raise RuntimeError("injected mesh traversal failure")

        monkeypatch.setattr(work._engine, "tree_hits", fault)
        with pytest.raises(RuntimeError, match="mesh traversal failure"):
            ph.detect(model, frames[1], ph.CollisionPolicy(), work)
        assert work.last_result is first
        with pytest.raises(RuntimeError, match="closed"):
            ph.detect(model, frames[0], ph.CollisionPolicy(), work)
