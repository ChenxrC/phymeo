"""Integration of independently developed rigid pairs and optional USD link geometry."""

from pathlib import Path

import numpy as np
import pytest

import phymeo as ph


def _mixed_scene():
    cylinder, sphere = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    first, second = ph.CollisionKey(1, 1, 0), ph.CollisionKey(1, 2, 0)
    solid = ph.CurvedSolid("cylinder", (1 / 2, 1 / 4))
    nodes = (4, 8, 9, 14)
    faces = ((4, 9, 8), (4, 8, 14), (4, 14, 9), (8, 9, 14))
    model = ph.CollisionModel(
        (
            ph.CollisionShape(second, "mesh", nodes, triangles=faces),
            ph.CollisionShape(sphere, "sphere", (4,), 3 / 16),
            ph.CollisionShape(first, "mesh", nodes, triangles=faces),
            ph.CollisionShape(cylinder, "cylinder", (4,), curved=solid),
        )
    )
    tetrahedron = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    paths = {
        cylinder: ph.CollisionMotion([[0, 0, 0]]),
        sphere: ph.CollisionMotion([[19 / 32, 0, 3 / 8]]),
        first: ph.CollisionMotion(tetrahedron),
        second: ph.CollisionMotion(tetrahedron + (1, 0, 0)),
    }
    return model, paths


def _check_mixed(backend, tmp_path):
    model, paths = _mixed_scene()
    with ph.CollisionWorkspace(
        model, backend, broad_phase="sap", cache_root=tmp_path / "cache"
    ) as work:
        out = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert out.backend == backend and out.required_candidates == 2
        assert len(out.pairs) == 6
        assert sum(p.reason == "different_world" for p in out.pairs) == 4
        cylinder, mesh = (p for p in out.pairs if p.candidate)
        assert cylinder.status == "overlap" and cylinder.mesh_mesh is None
        assert cylinder.curved.owner == cylinder.first
        assert cylinder.witness.gap_m == pytest.approx(-1 / 32)
        np.testing.assert_allclose(cylinder.witness.normal, (3 / 5, 0, 4 / 5))
        assert mesh.status == "touching" and mesh.curved is None
        assert mesh.mesh_mesh.end.boundary and not mesh.mesh_mesh.end.deepest
        for support in mesh.mesh_mesh.end.boundary:
            assert support.witness.first_point == pytest.approx((1, 0, 0))
            assert support.witness.second_point == pytest.approx((1, 0, 0))
        if backend == "cuda":
            assert work.stats()["spatial_completed_launches"] > 0
            assert work.stats()["spatial_downloads"] > 0


def test_local_shape_ids_in_coincident_worlds(tmp_path):
    _check_mixed("host", tmp_path)


@pytest.mark.tyx
def test_curve_and_dual_mesh_after_device_broad_phase(tyx_mode, tmp_path):
    _check_mixed(tyx_mode, tmp_path)


def test_usd_tool_motion_against_finite_cylinder():
    pytest.importorskip("pxr.Usd")
    path = Path(__file__).resolve().parents[1] / "reference_newton" / "robot" / "arm.usda"
    asset = ph.load_usd(path)
    (tool,) = asset.collisions
    key = ph.CollisionKey(0, 93, 0)
    barrel = ph.CollisionShape(
        key, "cylinder", (71,), curved=ph.CurvedSolid("cylinder", (1 / 4, 1 / 4))
    )
    model = ph.CollisionModel(
        (*asset.collision_model.shapes, barrel), asset.collision_model.excluded
    )
    fixed = {key: ph.CollisionMotion([[17 / 64, 1 / 16, 5 / 32]])}
    with ph.CollisionWorkspace(model) as work:
        clear = ph.detect(
            model, asset.collision_motion(np.zeros(2)) | fixed, ph.CollisionPolicy(), work
        )
        assert clear.pairs[0].status == "separated"
        poses = asset.collision_motion(np.array([np.pi / 2, 1 / 8])) | fixed
        (contact,) = ph.detect(model, poses, ph.CollisionPolicy(), work).pairs
    assert (contact.first, contact.second) == (tool.shape.key, key)
    assert contact.status == "overlap" and contact.curved.owner == key
    assert contact.witness.gap_m == pytest.approx(-1 / 64, abs=2e-6)
    np.testing.assert_allclose(contact.witness.normal, (1, 0, 0), atol=2e-6)
    np.testing.assert_allclose(contact.witness.second_point, (1 / 64, 1 / 16, 5 / 32), atol=2e-6)
