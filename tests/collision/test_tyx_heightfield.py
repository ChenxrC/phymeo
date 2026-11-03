"""Physical heightfield witnesses after the selected tyx broad phase."""

import numpy as np
import pytest

import phymeo as ph


@pytest.mark.tyx
@pytest.mark.parametrize("broad_phase", ["all_pairs", "sap", "bvh"])
def test_heightfield_uses_device_bounds_and_host_witness(tyx_mode, tmp_path, broad_phase):
    grid = ph.Heightfield([[0, 1, 0]] * 3, (1, 1), origin=(-1, -1, 0))
    land, ball = ph.CollisionKey(0, 0, 0), ph.CollisionKey(0, 1, 0)
    model = ph.CollisionModel(
        (
            ph.CollisionShape(land, "heightfield", tuple(range(9)), heightfield=grid),
            ph.CollisionShape(ball, "sphere", (42,), 5 / 16),
        )
    )
    with ph.CollisionWorkspace(
        model, tyx_mode, broad_phase=broad_phase, cache_root=tmp_path / "cache"
    ) as workspace:
        paths = {
            land: ph.CollisionMotion(grid.world_vertices()),
            ball: ph.CollisionMotion([[0, 1 / 8, 5 / 4]]),
        }
        report = ph.detect(model, paths, ph.CollisionPolicy(), workspace)
        assert report.backend == tyx_mode and report.required_candidates == 1
        (pair,) = report.pairs
        assert pair.status == "overlap" and pair.witness.source == "host-heightfield"
        assert pair.witness.gap_m == pytest.approx(-1 / 16)
        assert pair.witness.normal == pytest.approx((0, 0, 1))
        assert pair.witness.first_point == pytest.approx((0, 1 / 8, 1))
        coordinates = grid.world_vertices()[list(pair.heightfield.vertex_ids)]
        np.testing.assert_allclose(
            pair.heightfield.barycentric @ coordinates, pair.witness.first_point
        )
        paths[ball] = ph.CollisionMotion([[0, 1 / 8, 2]])
        clear = ph.detect(model, paths, ph.CollisionPolicy(), workspace)
        assert clear.required_candidates == 0 and clear.pairs[0].status == "separated"
        paths[ball] = ph.CollisionMotion([[-0.75, 1 / 8, 5 / 4]], [[0.75, 1 / 8, 5 / 4]])
        swept = ph.detect(model, paths, ph.CollisionPolicy(), workspace)
        (crossing,) = swept.pairs
        assert crossing.status == "hit" and swept.required_candidates == 1
        assert crossing.sweep.first_t == (3 / 8, 3 / 8)
        assert crossing.heightfield.last_contact.interval == (5 / 8, 5 / 8)
        assert crossing.heightfield.first_contact.witness.source == "host-heightfield"
        stats = workspace.stats()
        assert stats["completed_launches"] > 0
        if tyx_mode == "cuda":
            assert stats["downloads"] > 0
