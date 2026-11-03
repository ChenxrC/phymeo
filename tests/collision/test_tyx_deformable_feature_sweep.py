"""Real swept TYX face bounds with independently verified HOST material-time roots."""

import json

import pytest
from test_deformable_feature_sweep import check_event, ee_mesh, lookup, sweep, vt_mesh

import phymeo as ph

pytestmark = pytest.mark.tyx


@pytest.mark.parametrize("shape", ["vt", "ee"])
def test_device_midstep_positive_margin_entry_is_consumed(tyx_mode, tmp_path, shape):
    if shape == "vt":
        m, p, k = vt_mesh()
        search, force, expected = 1 / 8, 1 / 16, 3 / 16
    else:
        m, p, k = ee_mesh()
        search, force, expected = 3 / 16, 3 / 32, 3 / 8
    with ph.CollisionWorkspace(m, tyx_mode, cache_root=tmp_path / "cache") as work:
        before = work.stats()
        frame = sweep(m, p, work, search=search, force=force)
        stats = work.stats()
        assert work.last_result is None
    assert frame.status == "ok" and frame.backend == tyx_mode and frame.narrow_backend == "host"
    assert stats["completed_launches"] > before["completed_launches"]
    if tyx_mode == "cuda":
        assert stats["downloads"] > before["downloads"]
    event = lookup(frame, k).contact.first_entry
    check_event(event, expected, m, p, force, strict=True)
    (tmp_path / "device-entry.json").write_text(
        json.dumps(
            dict(
                shape=shape,
                backend=tyx_mode,
                interval=event.interval,
                sample_time=event.sample_time,
                sample_distance_m=event.sample.distance_m,
                root_nodes=frame.root_nodes,
                geometry={
                    k: v
                    for k, v in stats.items()
                    if k
                    in (
                        "completed_launches",
                        "uploads",
                        "downloads",
                        "uploaded_bytes",
                        "downloaded_bytes",
                        "synchronizations",
                    )
                },
            ),
            allow_nan=False,
        ),
        encoding="utf-8",
    )
