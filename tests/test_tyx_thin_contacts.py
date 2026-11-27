"""Original M16 counterexamples; fixed geometric accuracy, no training claims."""

import json
from pathlib import Path

import numpy as np
import pytest

from phymeo.backends.mujoco_tyx._mesh_catalog import MeshQueryCatalog
from phymeo.backends.mujoco_tyx._mesh_query import MeshQueryWorkspace

CASES = json.loads((Path(__file__).parent / "fixtures/m16_thin_contacts.json").read_text())["cases"]


@pytest.mark.tyx
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"w{c['world']}-{c['source_pair']}")
def test_saved_thin_contact_converges_with_owned_witnesses(case, tyx_mode, tmp_path):
    offsets = np.asarray(case["offsets"], np.int32)
    points = np.asarray(case["points"], np.float32)
    cat = MeshQueryCatalog(
        np.arange(2, dtype=np.int32),
        np.array([1, 2], np.int32),
        offsets,
        points,
        np.array([[0, 1]], np.int32),
        np.asarray(case["margins"], np.float32),
        2,
    )
    positions, rotations = np.asarray(case["positions"]), np.asarray(case["rotations"])
    with MeshQueryWorkspace(cat, capacity=4, tile=4, mode=tyx_mode, cache_root=tmp_path) as query:
        result = query.query(positions[None], rotations[None])
        data = result.read()
        with result.device_view() as view:
            assert view["gap"] is query._buffers["gap"]
            with pytest.raises(RuntimeError, match="borrowed"):
                query.query(positions[None], rotations[None])
            with pytest.raises(RuntimeError, match="borrowed"):
                query.close()
        query.query(positions[None], rotations[None])
        with pytest.raises(RuntimeError, match="expired"):
            with result.device_view():
                pass
    assert len(data["gap"]) == 1
    assert data["meta"][0, 0] == 1
    gap, low, high = data["gap"][0, :3].astype(float)
    assert abs(gap - case["reference_gap"]) <= 5e-7
    assert high - low <= 5e-7
    assert abs(np.linalg.norm(data["normal"][0]) - 1) < 1e-5
    witness = []
    for side in range(2):
        ids = data["material_ids"][0, side]
        weights = data["barycentric"][0, side]
        used = ids >= 0
        assert np.all(weights[used] >= 0)
        assert abs(float(weights[used].sum()) - 1) < 1e-5
        assert np.all((ids[used] >= offsets[side]) & (ids[used] < offsets[side + 1]))
        source = points[ids[used]].astype(float) @ rotations[side].T + positions[side]
        point = (source * weights[used, None]).sum(0)
        np.testing.assert_allclose(point, data["surface"][0, side], atol=2e-7, rtol=0)
        witness.append(point)
    np.testing.assert_allclose(witness[1] - witness[0], gap * data["normal"][0], atol=5e-7, rtol=0)


@pytest.mark.tyx
def test_unresolved_geometry_rejected_before_device_consumer(tyx_mode, tmp_path):
    case = CASES[0]
    cat = MeshQueryCatalog(
        np.arange(2, dtype=np.int32),
        np.array([1, 2], np.int32),
        np.asarray(case["offsets"], np.int32),
        np.asarray(case["points"], np.float32),
        np.array([[0, 1]], np.int32),
        np.asarray(case["margins"], np.float32),
        2,
    )
    positions = np.asarray(case["positions"])[None]
    rotations = np.asarray(case["rotations"])[None]
    with MeshQueryWorkspace(cat, capacity=4, tile=4, mode=tyx_mode, cache_root=tmp_path) as query:
        result = query.query(positions, rotations)
        meta = query._download("meta")
        meta[0, 0] = 2
        if tyx_mode == "cuda":
            query._buffers["meta"].copy_from(meta)
        else:
            query._buffers["meta"][...] = meta
        with pytest.raises(ValueError, match="uncertain or invalid"):
            with result.device_view():
                pass
        result = query.query(positions, rotations)
        with result.device_view():
            pass
