"""Two specialized continuous paths in one public collision call and different worlds."""

from math import acos, asin, pi, sqrt
from pathlib import Path

import numpy as np
import pytest
from test_mesh_mesh_contact import box
from test_rotating_mesh_capsule_solid import CASES, scene

import phymeo as ph


@pytest.mark.parametrize("backend", ["host", pytest.param("device", marks=pytest.mark.tyx)])
def test_two_hinge_sphere_and_rotating_mesh_shaft_share_only_the_dispatcher(
    backend, tmp_path, request
):
    if backend == "device":
        backend = request.getfixturevalue("tyx_mode")
    robot = ph.load_mjcf(Path(__file__).parents[1] / "reference_newton/robot/two_hinge_ellipse.xml")
    robot_paths = robot.collision_motion(np.zeros(2), np.array([pi, -2 * pi]))
    tool, target = (
        next(c.shape for c in robot.collisions if c.name == name) for name in ("tool", "north")
    )
    _, mesh_paths, mesh, shaft, _ = scene(
        box(*CASES["bar_bounds_m"]),
        [(0, 1.5, z) for z in CASES["axis_ends_z_m"]],
        CASES["capsule_radius_m"],
        pi,
        worlds=(1, 1),
    )
    model = ph.CollisionModel([tool, target, shaft, mesh])
    paths = {tool.key: robot_paths[tool.key], target.key: robot_paths[target.key], **mesh_paths}
    with ph.CollisionWorkspace(model, backend, cache_root=tmp_path) as workspace:
        result = ph.detect(model, paths, ph.CollisionPolicy(), workspace)
    assert result.status == "ok" and len(result.pairs) == 6
    expected = {tool.key.world: asin((sqrt(322) - 2) / 16) / pi, mesh.key.world: acos(1 / 6) / pi}
    for pair in result.pairs:
        if pair.first.world != pair.second.world:
            assert pair.status == "filtered" and pair.reason == "different_world"
            assert not pair.candidate
        else:
            assert pair.candidate and pair.status == "hit" and pair.sweep
            lo, hi = pair.sweep.first_t
            assert lo <= expected[pair.first.world] <= hi
            assert pair.sweep.start_gap_m > 0 and pair.sweep.end_gap_m > 0
