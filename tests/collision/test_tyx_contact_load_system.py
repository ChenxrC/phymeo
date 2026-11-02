"""Same independent physical windows using actual tyx collision and state solvers."""

import pytest
from load_cases import window
from load_history import save_history
from test_contact_load_system import check_window

pytestmark = pytest.mark.tyx


@pytest.mark.parametrize("family", ["RR", "RD", "DD"])
def test_device_load_window(family, tyx_mode, tmp_path):
    frames, stats = window(family, tyx_mode, tmp_path / "cache")
    check_window(family, frames, 2e-4)
    for frame in frames:
        assert frame.path.backend == tyx_mode or frame.path.backend == "tyx:" + tyx_mode
        for report in (frame.particle_report, frame.rigid_report):
            if report is not None:
                assert report.accepted and not report.fallback_used
                assert report.backend == "tyx:" + tyx_mode
    assert stats["completed_launches"] >= 2 * len(frames)
    layer = stats["normal_layer"]
    assert layer["sites"] == sum(len(c.sites) for f in frames for c in f.loads.contacts)
    assert layer["completed_launches"] > 0
    if tyx_mode == "cuda":
        assert layer["uploads"] == layer["downloads"] == layer["completed_launches"]
    save_history(tmp_path / (family + ".json"), family, frames, stats)


def test_device_normal_law_crosses_a_page(tyx_mode, tmp_path):
    import numpy as np

    from phymeo.backends.tyx_normal_layer import TyxNormalLayer

    # Columns: n, relative velocity, pair width, gap, k, site fraction, damping.
    rows = np.tile([0.0, 1.0, 0.0, 0.0, -0.5, 0.0, 0.0625, 0.03125, 128.0, 0.25, 0.25], (131, 1))
    rows[1, 4] = 8.0
    rows[-1, 7] = 0.0625
    with TyxNormalLayer(tyx_mode, cache_root=tmp_path / "cache") as engine:
        out = engine.evaluate(rows)
        assert out[:, 0] == pytest.approx(rows[:, 4])
        assert out[0, 1:].tolist() == [1.125, 0.0, 1.125, 0.0]
        assert out[1, 1:].tolist() == [0.0, 0.0, 0.0, 0.0]
        assert out[-1, 1:].tolist() == [0.0, 0.0, 0.0, 0.0]
        assert engine.stats()["sites"] == 131
        assert engine.stats()["completed_launches"] == 2
    with pytest.raises(RuntimeError, match="closed"):
        engine.evaluate(rows[:1])
