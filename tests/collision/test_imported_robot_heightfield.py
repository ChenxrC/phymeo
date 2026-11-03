"""Imported link geometry uses the same material addresses as public terrain queries."""

from pathlib import Path

import numpy as np
import pytest

import phymeo as ph

ASSETS = Path(__file__).resolve().parents[1] / "reference_newton" / "robot"


@pytest.mark.parametrize(
    "loader, filename", [(ph.load_urdf, "arm.urdf"), (ph.load_mjcf, "arm.mjcf.xml")]
)
def test_tool_sphere_crosses_finite_terrain_footprint(loader, filename):
    asset = loader(ASSETS / filename)
    (tool,) = asset.collisions
    land_key = ph.CollisionKey(0, 91, 7)
    grid = ph.Heightfield([[0, 0, 0], [0, 0, 0]], (0.5, 0.5), origin=(-3 / 8, -1 / 4, 9 / 64))
    land = ph.CollisionShape(land_key, "heightfield", (2, 7, 12, 17, 22, 27), heightfield=grid)
    model = ph.CollisionModel((*asset.collision_model.shapes, land), asset.collision_model.excluded)
    pose = {land_key: ph.CollisionMotion(grid.world_vertices())}
    with ph.CollisionWorkspace(model) as work:
        clear = ph.detect(
            model, asset.collision_motion(np.zeros(2)) | pose, ph.CollisionPolicy(), work
        )
        assert clear.pairs[0].status == "separated"
        paths = asset.collision_motion(np.array([np.pi / 2, 1 / 8])) | pose
        (pair,) = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
    assert (pair.first, pair.second) == (tool.shape.key, land_key)
    assert pair.status == "overlap"
    assert pair.witness.gap_m == pytest.approx(-1 / 64)
    np.testing.assert_allclose(pair.witness.normal, (0, 0, -1), rtol=0, atol=1e-12)
    np.testing.assert_allclose(pair.witness.first_point, (0, 1 / 16, 1 / 8), rtol=0, atol=1e-12)
    np.testing.assert_allclose(pair.witness.second_point, (0, 1 / 16, 9 / 64), rtol=0, atol=1e-12)
    nodes = dict(zip(land.vertex_ids, grid.world_vertices(), strict=True))
    face = np.array([nodes[index] for index in pair.heightfield.vertex_ids])
    np.testing.assert_allclose(
        pair.heightfield.barycentric @ face, pair.witness.second_point, atol=1e-12
    )
