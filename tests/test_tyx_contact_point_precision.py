"""Unique original support witnesses and precision at the force lever arm."""

import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from phymeo.backends.mujoco_tyx._mesh_catalog import MeshQueryCatalog
from phymeo.backends.mujoco_tyx._mesh_query import MeshQueryWorkspace

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/m16_unique_contact_points.json").read_text()
)


def catalog(case, swap=False):
    points = [np.asarray(v, np.float32) for v in case["points"]]
    return MeshQueryCatalog(
        np.asarray(case["geom_ids"], np.int32),
        np.asarray(case["body_ids"], np.int32),
        np.array([0, len(points[0]), sum(map(len, points))], np.int32),
        np.concatenate(points),
        np.array([[1, 0] if swap else [0, 1]], np.int32),
        np.zeros(2, np.float32),
        FIXTURE["source_ngeom"],
    )


@pytest.mark.tyx
@pytest.mark.parametrize("index", range(8))
@pytest.mark.parametrize("swap", [False, True])
def test_unique_original_feature_point_and_material(tyx_mode, tmp_path, index, swap):
    case = FIXTURE["cases"][index]
    cat = catalog(case, swap)
    pos, rot = np.asarray(case["positions"]), np.asarray(case["rotations"])
    with MeshQueryWorkspace(cat, capacity=2, mode=tyx_mode, cache_root=tmp_path) as q:
        result = q.query(pos[None], rot[None])
        raw = result.read()
        assert raw["witness_receipt"][0, 0] == 2
        expected_features = case["expected_features"][::-1] if swap else case["expected_features"]
        np.testing.assert_array_equal(raw["witness_receipt"][0, 1:], expected_features)
        surfaces = raw["surface"][0].astype(float) + raw["surface_low"][0].astype(float)
        np.testing.assert_allclose(surfaces.mean(0), case["expected_point"], rtol=0, atol=2e-9)
        for side, g in enumerate([1, 0] if swap else [0, 1]):
            ids = raw["material_ids"][0, side]
            used = ids >= 0
            weights = raw["barycentric"][0, side, used].astype(float)
            assert np.all(weights >= 0) and abs(weights.sum() - 1) < 2e-7
            assert np.all((ids[used] >= cat.offsets[g]) & (ids[used] < cat.offsets[g + 1]))
            local = cat.points[ids[used]].astype(float)
            point = local[0] + (weights[1:, None] * (local[1:] - local[0])).sum(0)
            expected = rot[g] @ point + pos[g]
            np.testing.assert_allclose(surfaces[side], expected, rtol=0, atol=5e-10)
        with result.device_view() as view:
            assert view["surface_low"] is q._buffers["surface_low"]
            with pytest.raises(RuntimeError, match="borrowed"):
                q.close()


@pytest.mark.tyx
@pytest.mark.parametrize("index", [0, 4])
def test_unique_point_rigid_transform_and_buffer_reuse(tyx_mode, tmp_path, index):
    case = FIXTURE["cases"][index]
    cat = catalog(case)
    pos, rot = np.asarray(case["positions"]), np.asarray(case["rotations"])
    # A proper signed permutation is exactly representable, preserving both
    # the original support topology and the input rotation conditioning.
    Q = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    shift = np.array([0.125, -0.25, 0.0625])
    with MeshQueryWorkspace(cat, capacity=2, mode=tyx_mode, cache_root=tmp_path) as q:
        first = q.query(pos[None], rot[None])
        actual = q.query((pos @ Q.T + shift)[None], (Q @ rot)[None]).read()
        with pytest.raises(RuntimeError, match="expired"):
            first.read()
        assert actual["witness_receipt"][0, 0] == 2
        point = (actual["surface"][0].astype(float) + actual["surface_low"][0]).mean(0)
        np.testing.assert_allclose(point, Q @ case["expected_point"] + shift, rtol=0, atol=2e-9)


@pytest.mark.tyx
def test_nonunique_face_face_retains_certificate_without_false_unique_claim(tyx_mode, tmp_path):
    p = (np.array(list(product((-1, 1), repeat=3))) * 0.1).astype(np.float32)
    case = dict(geom_ids=[0, 1], body_ids=[0, 1], points=[p, p])
    cat = catalog(case)
    pos = np.array([[[0, 0, 0], [0.195, 0, 0]]], float)
    rot = np.broadcast_to(np.eye(3), (1, 2, 3, 3)).copy()
    with MeshQueryWorkspace(cat, capacity=2, mode=tyx_mode, cache_root=tmp_path) as q:
        result = q.query(pos, rot)
        raw = result.read()
        assert raw["witness_receipt"][0].tolist() == [1, 4, 4]
        assert raw["meta"][0, 9] == 7
        with result.device_view():
            pass
        pos[0, 1, 0] = 0.201
        raw = q.query(pos, rot).read()
        assert raw["witness_receipt"].shape[0] == 0  # no stale low lanes exported


@pytest.mark.tyx
@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.001])
def test_invalid_point_low_cannot_enter_force_response(tyx_mode, tmp_path, bad):
    case = FIXTURE["cases"][0]
    with MeshQueryWorkspace(catalog(case), capacity=2, mode=tyx_mode, cache_root=tmp_path) as q:
        result = q.query(np.asarray(case["positions"])[None], np.asarray(case["rotations"])[None])
        low = np.zeros((2, 2, 3), np.float32)
        low[0, 0, 0] = bad
        if tyx_mode == "cuda":
            q._buffers["surface_low"].copy_from(low)
        else:
            q._buffers["surface_low"][...] = low
        with pytest.raises(ValueError, match="invalid geometry"):
            with result.device_view():
                pass
