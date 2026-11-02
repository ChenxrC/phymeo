"""Device geometry observations survive only the accepted outer transaction."""

import pytest
from load_cases import scene

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_device_geometry_rolls_back_both_public_query_and_physical_frame(tyx_mode, tmp_path):
    system, state, models, constraints = scene("RD")
    solvers = ph.XPBDSolver(constraints), ph.RigidSemiImplicitSolver()
    loads = ph.CoupledInputs.zeros(3, 1)
    with ph.CollisionWorkspace(
        system.model, tyx_mode, broad_phase="bvh", cache_root=tmp_path / "cache"
    ) as ws:
        good = system.step(state, loads, models, solvers, 1 / 64, ws)
        prior = ws.stats()["completed_launches"]
        state.rigid.position[0, 2] = 0.5
        state.rigid.velocity[0, 2] = -64.0
        with pytest.raises(ph.SolverFailure, match="core"):
            system.step(state, loads, models, solvers, 1 / 64, ws)
        assert ws.last_result is good.path and system.last_frame is good
        assert ws.stats()["completed_launches"] > prior
