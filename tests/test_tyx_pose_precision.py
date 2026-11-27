"""Original nanometre contact activation regressions and split-pose API."""

import json
from pathlib import Path

import numpy as np
import pytest

from phymeo.backends.mujoco_tyx._mesh_catalog import MeshQueryCatalog
from phymeo.backends.mujoco_tyx._mesh_query import MeshQueryWorkspace

FIXTURE = json.loads((Path(__file__).parent / "fixtures/m16_contact_activation.json").read_text())
FEATURES = json.loads((Path(__file__).parent / "fixtures/m16_feature_selection.json").read_text())


def inputs(index):
    case = FIXTURE["cases"][index]
    a, b = (np.asarray(p, np.float32) for p in case["points"])
    cat = MeshQueryCatalog(
        np.asarray(case["geom_ids"], np.int32),
        np.asarray(case["body_ids"], np.int32),
        np.array([0, len(a), len(a) + len(b)], np.int32),
        np.concatenate((a, b)),
        np.array([[0, 1]], np.int32),
        np.zeros(2, np.float32),
        FIXTURE["source_ngeom"],
    )
    return case, cat, np.asarray(case["positions"])[None], np.asarray(case["rotations"])[None]


@pytest.mark.tyx
@pytest.mark.parametrize("index", range(len(FEATURES["cases"])))
def test_retained_edge_triangle_selects_active_axis(tyx_mode, tmp_path, index):
    case = FEATURES["cases"][index]
    a, b = (np.asarray(p, np.float32) for p in case["points"])
    cat = MeshQueryCatalog(
        np.asarray(case["geom_ids"], np.int32),
        np.asarray(case["body_ids"], np.int32),
        np.array([0, len(a), len(a) + len(b)], np.int32),
        np.concatenate((a, b)),
        np.array([[0, 1]], np.int32),
        np.zeros(2, np.float32),
        FEATURES["source_ngeom"],
    )
    with MeshQueryWorkspace(cat, capacity=2, mode=tyx_mode, cache_root=tmp_path) as q:
        raw = q.query(
            np.asarray(case["positions"])[None], np.asarray(case["rotations"])[None]
        ).read()
        assert raw["meta"][0, 0] == 1
        assert raw["feature_corrected"][0] == 2
        # Near-coplanar supports need the axis low lane until after projection.
        assert raw["gap"][0, 0] == pytest.approx(case["signed_sat"], abs=5e-12)
        np.testing.assert_allclose(raw["normal"][0], case["sat_normal"], rtol=0, atol=2e-7)


@pytest.mark.tyx
@pytest.mark.parametrize("index", range(22))
def test_original_contact_activation(tyx_mode, tmp_path, index):
    case, cat, pos, rot = inputs(index)
    with MeshQueryWorkspace(cat, capacity=2, mode=tyx_mode, cache_root=tmp_path) as q:
        result = q.query(pos, rot)
        raw = result.read()
        assert result.counts == (1,)
        assert raw["meta"][0, 0] == 1
        assert bool(raw["gap"][0, 0] < 0) == case["contact"]
        assert raw["gap"][0, 0] == pytest.approx(case["signed_sat"], abs=2e-9)
        with result.device_view() as view:
            assert view["positions_low"] is q._buffers["positions_low"]
            with pytest.raises(RuntimeError, match="borrowed"):
                q.query(pos, rot)


@pytest.mark.tyx
def test_split_device_input_does_not_upload_or_inherit_stale_low(tyx_mode, tmp_path):
    import tyx as tx

    _, cat, pos, rot = inputs(0)
    values = [pos.astype(np.float32), rot.astype(np.float32)]
    values += [
        (pos - values[0].astype(float)).astype(np.float32),
        (rot - values[1].astype(float)).astype(np.float32),
    ]
    with (
        MeshQueryWorkspace(cat, capacity=2, mode=tyx_mode, cache_root=tmp_path) as q,
        q._runtime.activate(),
    ):
        expected = q.query(pos, rot).read()
        uploaded = q.stats["uploaded_bytes"]
        device = [tx.cuda.array(v) if tyx_mode == "cuda" else v for v in values]
        try:
            actual = q.query_device(
                device[0], device[1], positions_low=device[2], rotations_low=device[3]
            ).read()
            assert q.stats["uploaded_bytes"] == uploaded
            for key in ("gap", "normal", "surface", "material_ids", "barycentric"):
                np.testing.assert_array_equal(actual[key], expected[key])
            # An FP32-only producer must get zero lows, not the preceding call's residuals.
            plain = q.query_device(device[0], device[1]).read()
            explicit_plain = q.query(values[0], values[1]).read()
            np.testing.assert_array_equal(plain["gap"], explicit_plain["gap"])
            with pytest.raises(ValueError, match="together"):
                q.query_device(device[0], device[1], positions_low=device[2])
        finally:
            for v in device:
                if hasattr(v, "free"):
                    v.free()


@pytest.mark.tyx
@pytest.mark.parametrize("bad", ["nan", "overlapping", "shape", "dtype"])
def test_bad_low_lanes_expire_previous_results(tyx_mode, tmp_path, bad):
    import tyx as tx

    _, cat, pos, rot = inputs(0)
    values = [
        pos.astype(np.float32),
        rot.astype(np.float32),
        np.zeros_like(pos, np.float32),
        np.zeros_like(rot, np.float32),
    ]
    if bad == "nan":
        values[2][0, 0, 0] = np.nan
    elif bad == "overlapping":
        values[3][0, 0, 0, 0] = 1e-2
    elif bad == "shape":
        values[2] = values[2][:, :1]
    else:
        values[2] = values[2].astype(np.int32)
    with (
        MeshQueryWorkspace(cat, capacity=2, mode=tyx_mode, cache_root=tmp_path) as q,
        q._runtime.activate(),
    ):
        old = q.query(pos, rot)
        device = [tx.cuda.array(v) if tyx_mode == "cuda" else v for v in values]
        try:
            with pytest.raises(ValueError):
                q.query_device(
                    device[0], device[1], positions_low=device[2], rotations_low=device[3]
                )
            with pytest.raises(RuntimeError, match="expired"):
                old.read()
        finally:
            for v in device:
                if hasattr(v, "free"):
                    v.free()
