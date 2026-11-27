"""All original large-face witnesses rejected by independent .003s checking."""

import json
from pathlib import Path

import numpy as np
import pytest

from phymeo.backends.mujoco_tyx._mesh_catalog import MeshQueryCatalog
from phymeo.backends.mujoco_tyx._mesh_query import MeshQueryWorkspace
from phymeo.validation.convex_contact import certify_contact

CASES = json.loads(
    (Path(__file__).parent / "fixtures/m16_face_projection_precision.json").read_text()
)


@pytest.mark.tyx
@pytest.mark.parametrize("transform", [False, True])
def test_all161_original_face_projections(tyx_mode, tmp_path, transform):
    points = []
    offsets = [0]
    positions = []
    rotations = []
    pairs = []
    Q = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]) if transform else np.eye(3)
    shift = np.array([0.125, -0.25, 0.0625]) if transform else np.zeros(3)
    for i, c in enumerate(CASES):
        for p in c["points"]:
            points.extend(p)
            offsets.append(len(points))
        positions.extend(np.array(c["positions"]) @ Q.T + shift)
        rotations.extend(Q @ np.array(c["rotations"]))
        pairs.append([2 * i, 2 * i + 1])
    n = len(positions)
    cat = MeshQueryCatalog(
        np.arange(n, dtype=np.int32),
        np.arange(n, dtype=np.int32),
        np.array(offsets, np.int32),
        np.array(points, np.float32),
        np.array(pairs, np.int32),
        np.zeros(n, np.float32),
        n,
    )
    pos = np.array(positions)
    rot = np.array(rotations)
    with MeshQueryWorkspace(cat, capacity=256, tile=256, mode=tyx_mode, cache_root=tmp_path) as q:
        raw = q.query(pos[None], rot[None]).read()
        assert len(raw["gap"]) == 161
        for i, pair in enumerate(raw["geom_ids"]):
            cert = certify_contact(
                cat.points,
                cat.offsets,
                pos,
                rot,
                pair,
                raw["material_ids"][i],
                raw["barycentric"][i],
                raw["surface"][i].astype(float) + raw["surface_low"][i],
                raw["normal"][i],
                float(raw["gap"][i, 0]),
                raw["witness_receipt"][i],
            )
            assert cert.unique and cert.kind == "unique_vertex_face"
            expected = Q @ CASES[int(pair[0]) // 2]["independent_point"] + shift
            np.testing.assert_allclose(cert.point, expected, rtol=0, atol=2e-12)
