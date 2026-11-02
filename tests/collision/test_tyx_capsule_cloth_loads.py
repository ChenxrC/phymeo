"""The selected tyx mode produces detection, load and both dynamic responses."""

import numpy as np
import pytest
from capsule_load_cases import actual_step

pytestmark = pytest.mark.tyx


def test_capsule_shaft_drives_both_native_participants(tyx_mode, tmp_path):
    frame, stats = actual_step(tyx_mode, tmp_path / "cache")
    assert stats["completed_launches"] > 0
    assert frame.path.backend == tyx_mode
    np.testing.assert_allclose(
        frame.state.rigid.velocity, [[0.0, 0.0, 4 / 64]], rtol=2e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        frame.state.particles.velocity[:, 2], [-(2 / 3) / 64] * 3, rtol=2e-6, atol=1e-7
    )
    assert frame.particle_report.accepted and frame.rigid_report.accepted
    assert len(frame.loads.contacts[0].sites) == 2
