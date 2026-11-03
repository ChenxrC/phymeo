"""Only face broad phase uses TYX; the full material primitive set is constructed on HOST."""

import json

import pytest
from test_deformable_features import CAT, key, query, readbacks, register

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_real_broad_phase_supplies_faces_for_every_native_fold_feature(tyx_mode, tmp_path):
    shapes, paths = register(CAT["DD-REMOTE-FOLD"])
    model = ph.CollisionModel(shapes)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        before = work.stats()
        result = query(model, paths, work)
        after = work.stats()
        assert work.last_result is None
    assert result.status == "ok" and result.backend == tyx_mode and result.narrow_backend == "host"
    assert after["completed_launches"] > before["completed_launches"]
    if tyx_mode == "cuda":
        assert (
            after["downloads"] > before["downloads"]
            and after["downloaded_bytes"] > before["downloaded_bytes"]
        )
    rows = {key(r): r for r in result.contacts}
    readbacks(rows, "cpu")
    readbacks(rows, "cuda:0")
    assert len(rows) == 32 and result.broad_face_pairs > 0
    (tmp_path / "deformable-features-device.json").write_text(
        json.dumps(
            dict(
                backend=result.backend,
                narrow=result.narrow_backend,
                broad=result.broad_face_pairs,
                feature_tests=result.feature_tests,
                retained=len(rows),
                stats={
                    k: v
                    for k, v in after.items()
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


def test_device_rounding_preserves_threshold_neighbors_and_fault_is_explicit(
    tyx_mode, tmp_path, monkeypatch
):
    import numpy as np

    shapes, paths = register(CAT["DD-VT-AT-QUERY"])
    model = ph.CollisionModel(shapes)
    focus = shapes[1].key
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        previous = ph.detect(model, paths, ph.CollisionPolicy(), work)
        for value in (np.nextafter(3 / 32, 0.0), 3 / 32, np.nextafter(3 / 32, 1.0)):
            pts = paths[focus].start.copy()
            pts[0, 2] = value
            obs = query(model, {**paths, focus: ph.CollisionMotion(pts)}, work)
            assert (("VT", (53,), (11, 23, 37)) in {key(x) for x in obs.contacts}) == (
                value < 3 / 32
            )
            assert work.last_result is previous

        def fail(*args, **kwargs):
            raise RuntimeError("injected feature broad phase failure")

        monkeypatch.setattr(work._engine, "query", fail)
        with pytest.raises(RuntimeError, match="injected"):
            query(model, paths, work)
        assert work.last_result is previous
        with pytest.raises(RuntimeError, match="closed"):
            query(model, paths, work)
