"""Actual TYX point sampling consumed by a mixed material/rigid-field query."""

import numpy as np
import pytest
from test_sdf_material import analytic_triangle, grid, setup, shape

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_tyx_batch_reads_the_field_for_verified_material_witnesses(tyx_mode, tmp_path, monkeypatch):
    g, tri, p = analytic_triangle()
    ball = shape("rounded_particle", (73,), body=2, radius=0.125)
    remote = shape("particle", (73,), world=1)
    m, f = setup(g, materials=(tri, ball, remote))
    points = {tri.key: p, ball.key: [[0, 0, 0]], remote.key: [[0, 0, 0]]}
    host = ph.sample_sdf_contacts(m, f, points)
    monkeypatch.setattr(
        type(g),
        "sample_local",
        lambda *args: pytest.fail("used HOST interpolation with TYX sampler"),
    )
    with ph.TyxSdfSampler(g, mode=tyx_mode, capacity=3, cache_root=tmp_path / "cache") as sampler:
        result = ph.sample_sdf_contacts(m, f, points, sampler=sampler)
        stats = sampler.stats()
        with pytest.raises(ValueError, match="grid"):
            ph.sample_sdf_contacts(
                m,
                setup(grid(lambda x, y, z: x), materials=(tri, ball, remote))[1],
                points,
                sampler=sampler,
            )
    assert result.backend == tyx_mode and result.samples_evaluated == stats["samples"] > 3
    assert stats["completed_launches"] == (result.samples_evaluated + 2) // 3
    if tyx_mode == "cuda":
        assert stats["grid_uploaded_bytes"] == 64 and stats["grid_uploads"] == 1
        assert stats["query_uploaded_bytes"] == 16 * result.samples_evaluated
        assert stats["downloaded_bytes"] == 20 * result.samples_evaluated
    assert [q.field_status for q in result.results] == ["contact", "clear", "filtered"]
    for cpu, device in zip(host.results, result.results, strict=True):
        assert cpu.field_gap_bounds_m == device.field_gap_bounds_m
        for a, b in zip(cpu.samples, device.samples, strict=True):
            assert a.weights == b.weights and a.cell_xyz == b.cell_xyz
            assert b.field_distance_m == pytest.approx(a.field_distance_m, abs=2e-6)
            np.testing.assert_allclose(a.gradient_world, b.gradient_world, rtol=0, atol=3e-6)
    import json

    (tmp_path / "sdf-material-device.json").write_text(
        json.dumps(
            dict(
                backend=result.backend,
                samples=result.samples_evaluated,
                rows=[
                    dict(
                        status=row.field_status,
                        bounds=row.field_gap_bounds_m,
                        points=len(row.samples),
                    )
                    for row in result.results
                ],
                stats=stats,
            ),
            allow_nan=False,
        ),
        encoding="utf-8",
    )
