"""A real tyx contact force enters a finite-inertia articulated substep."""

import numpy as np
import pytest
from test_cloth_link_contact import make_scene

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_native_shaft_force_reaches_cloth_and_articulated_coordinates(tyx_mode, tmp_path):
    with ph.TyxNormalLayer(tyx_mode, cache_root=tmp_path / "cache") as device:
        program, state, inputs = make_scene(revolute=True, operator=device)
        with program.stepper(
            ph.StepSettings(substeps=1, position_iterations=1, velocity_iterations=1)
        ) as stepper:
            result = stepper.step(state, inputs, 1 / 256)
        assert result.report.accepted
        row = result.report.substeps[0].diagnostics["robot_contact"]
        assert row["cloth_link_efforts"]["arm"][0] == pytest.approx(-1.0, rel=3e-3)
        assert result.state.domains["arm"].v[0] < 0.0
        assert np.all(result.state.domains["cloth"].velocity[:, 2] < 0.0)
        assert device.stats()["sites"] > 0 and device.stats()["completed_launches"] > 0
