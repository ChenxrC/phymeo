"""Saved moving-M16 finite-hull progress regression; no integration."""

import json
from pathlib import Path

import numpy as np
import pytest

from phymeo.backends.mujoco_tyx._mesh_catalog import MeshQueryCatalog
from phymeo.backends.mujoco_tyx._mesh_query import MeshQueryWorkspace


@pytest.mark.tyx
@pytest.mark.parametrize("variant", ["original", "reverse", "permute", "rotate", "translate"])
def test_saved_finite_hull_progress(tyx_mode, tmp_path, variant):
    case = json.loads((Path(__file__).parent / "fixtures/m16_gjk_progress.json").read_text())
    points = [np.array(p, np.float32) for p in case["points"]]
    positions = np.array(case["positions"], float)
    rotations = np.array(case["rotations"], float)
    expected_axis = np.array(case["axis"], float)
    if variant == "reverse":
        points.reverse()
        positions = positions[::-1].copy()
        rotations = rotations[::-1].copy()
        expected_axis = -expected_axis
    elif variant == "permute":
        points = [p[::-1].copy() for p in points]
    elif variant == "rotate":
        R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
        positions = positions @ R.T
        rotations = R @ rotations
        expected_axis = R @ expected_axis
    elif variant == "translate":
        positions += np.array([2.5, -1.25, 0.75])
    cat = MeshQueryCatalog(
        np.array([0, 1], np.int32),
        np.array([1, 2], np.int32),
        np.array([0, len(points[0]), sum(map(len, points))], np.int32),
        np.concatenate(points),
        np.array([[0, 1]], np.int32),
        np.zeros(2, np.float32),
        2,
    )
    with MeshQueryWorkspace(
        cat, worlds=1, capacity=4, tile=1, mode=tyx_mode, cache_root=tmp_path
    ) as work:
        result = work.query(positions[None], rotations[None])
        raw = result.read()
        assert len(raw["meta"]) == 1
        assert raw["meta"][0, 0] == 1, raw["meta"].tolist()
        assert raw["meta"][0, 4] < 64
        assert raw["meta"][0, 9] == 7
        assert abs(float(raw["gap"][0, 0]) - case["distance"]) < 1e-7
        np.testing.assert_allclose(raw["normal"][0], expected_axis, atol=2e-4, rtol=0)
        assert raw["gap"][0, 1] <= raw["gap"][0, 0] <= raw["gap"][0, 2]
        assert raw["gap"][0, 2] - raw["gap"][0, 1] < 5e-6
        surface = raw["surface"].astype(float) + raw["surface_low"]
        np.testing.assert_allclose(
            surface[0, 1] - surface[0, 0], raw["gap"][0, 0] * raw["normal"][0], atol=5e-6, rtol=0
        )
        with result.device_view():
            pass
