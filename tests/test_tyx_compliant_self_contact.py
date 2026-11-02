"""The commanded backend runs both self-contact geometry and force on a real material step."""

import pytest
from test_compliant_self_contact import setup_sheet

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_native_self_layer_enters_the_compliant_elastic_sheet(tyx_mode, tmp_path):
    with ph.TyxNormalLayer(tyx_mode, cache_root=tmp_path / "cache") as op:
        program, state, inputs = setup_sheet(native=op, mode=tyx_mode)
        with program.stepper(
            ph.StepSettings(substeps=1, position_iterations=1, velocity_iterations=1)
        ) as rt:
            out = rt.step(state, inputs, 1 / 256)
        assert out.report.accepted, out.report.reason
        obs = out.report.substeps[0].diagnostics["self_contact"]
        assert obs["geometry_backend"] == tyx_mode and obs["force_backend"] == tyx_mode
        assert obs["contacts"][0]["force_n"] == pytest.approx(2.0, rel=0.01)
        assert op.stats()["completed_launches"] > 0
        assert out.state.domains["sheet"].velocity[0, 2] < 0
