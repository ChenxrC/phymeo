"""A real tyx force and particle solver act on two remote faces of one connected sheet."""

import pytest
from test_connected_cloth_loads import folded_strip, verify_folded_strip

pytestmark = pytest.mark.tyx


def test_native_connected_sheet_load_and_material_state(tyx_mode, tmp_path):
    frame, stats = folded_strip(tyx_mode, tmp_path / "cache")
    verify_folded_strip(frame, 3e-5)
    assert frame.path.backend == tyx_mode and frame.particle_report.accepted
    assert stats["completed_launches"] > 0 and stats["layer"]["completed_launches"] > 0
