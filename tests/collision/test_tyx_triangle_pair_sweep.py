"""Actual tyx vertex/face and edge/edge time proposals with independent public checks."""

import json
from dataclasses import asdict

import pytest
from triangle_pair_cases import VARIANTS, authored, graze, piercing

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_device_triangle_catalog_and_edge_graze(tyx_mode, tmp_path, monkeypatch):
    shapes = []
    paths = {}
    names = []
    for case, variants in VARIANTS.items():
        for name in variants:
            sub, p = authored(case, name, len(names))
            names.append(case + "." + name)
            shapes.extend(sub.shapes)
            paths.update(p)
    for name, fn in (("graze", graze), ("initial_pierce", piercing)):
        sub, p = fn(len(names))
        names.append(name)
        shapes.extend(sub.shapes)
        paths.update(p)
    model = ph.CollisionModel(tuple(reversed(shapes)))
    policy = ph.CollisionPolicy(margin=3.0)
    with ph.CollisionWorkspace(model) as host:
        reference = ph.detect(model, paths, policy, host)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        out = ph.detect(model, paths, policy, work)
        assert out.bounds == reference.bounds and out.pairs == reference.pairs
        assert len(out.pairs) == 231 and out.required_candidates == 11
        rows = [r for r in out.pairs if r.candidate]
        assert [r.status for r in rows] == [
            "separated" if name.endswith("clear") else "hit" for name in names
        ]
        stats = work.stats()
        assert stats["triangle_completed_launches"] >= 1 and stats["triangle_proposed_hits"] >= 1
        assert (
            stats["triangle_vertex_face_features"] == 66
            and stats["triangle_edge_edge_features"] >= 1
        )
        small = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0, margin=3), work)
        assert (
            small.status == "capacity_exceeded"
            and small.required_candidates == 11
            and work.last_result is out
        )
        assert work.stats()["triangle_completed_launches"] == stats["triangle_completed_launches"]
        assert (
            ph.detect(model, paths, policy, work) == out
            and work.stats()["allocations"] == stats["allocations"]
        )
        (tmp_path / "observation.json").write_text(
            json.dumps(
                {
                    "names": names,
                    "rows": [asdict(r) for r in rows],
                    "pair_count": len(out.pairs),
                    "stats": work.stats(),
                },
                allow_nan=False,
            )
        )

        def fault(_features):
            raise RuntimeError("injected triangle entry failure")

        monkeypatch.setattr(work._engine, "entry_times", fault)
        with pytest.raises(RuntimeError, match="injected"):
            ph.detect(model, paths, policy, work)
        assert work.last_result == out
        with pytest.raises(RuntimeError, match="closed"):
            ph.detect(model, paths, policy, work)


def test_device_off_grid_edge_seed(tyx_mode, tmp_path):
    model, paths = authored("DD02", "holdout")
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        (row,) = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
        assert row.status == "hit" and row.sweep.first_t[0] <= 0.6 <= row.sweep.first_t[1]
        stats = work.stats()
        assert stats["triangle_edge_edge_proposed_hits"] >= 1
        (tmp_path / "observation.json").write_text(
            json.dumps({"pair": asdict(row), "stats": stats}, allow_nan=False)
        )


@pytest.mark.parametrize("name", ["moving_frame", "coplanar_clear"])
def test_device_triangle_pair_frame_and_separation(name, tyx_mode, tmp_path):
    import numpy as np
    from triangle_pair_cases import scene, transit

    if name == "moving_frame":
        base, p = transit()
        shapes = []
        paths = {}
        rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        offset = np.array([3, -4, 1])
        drift = np.array([0.5, -0.25, 0.75])
        order = [2, 0, 1]
        for shape in base.shapes:
            path = p[shape.key]
            start = path.start[order] @ rotation.T + offset
            end = (
                (path.start if path.end is None else path.end)[order] @ rotation.T + offset + drift
            )
            shapes.append(
                ph.CollisionShape(shape.key, "triangle", tuple(shape.vertex_ids[i] for i in order))
            )
            paths[shape.key] = ph.CollisionMotion(start, end)
        model = ph.CollisionModel(tuple(reversed(shapes)))
    else:
        a = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        b = np.array([[0.625, 0.625, 0], [1.125, 0.625, 0], [0.625, 1.125, 0]])
        model, paths = scene(a, None, b, b + [-1 / 16, -1 / 16, 0])

    with (
        ph.CollisionWorkspace(model) as host,
        ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as device,
    ):
        reference = ph.detect(model, paths, ph.CollisionPolicy(), host)
        out = ph.detect(model, paths, ph.CollisionPolicy(), device)
        assert out.pairs == reference.pairs and out.bounds == reference.bounds
        (row,) = out.pairs
        assert row.candidate
        if name == "moving_frame":
            assert row.status == "hit" and row.sweep.first_t == (0.25, 0.25)
            assert row.sweep.start_gap_m == 0.5 and row.witness.gap_m == 0.25
            (point,) = row.sweep.witness.intersection_points
            assert point.point == (3.125, -4.0625, 1.1875)
            assert dict(point.first_weights) == {11: 0.5, 7: 0.5, 3: 0.0}
            assert dict(point.second_weights) == {31: 0.5, 41: 0.5, 53: 0.0}
        else:
            assert row.status == "separated" and row.sweep.first_t is None
            assert row.sweep.start_gap_m == pytest.approx(0.25 / np.sqrt(2))
            assert row.witness.gap_m == pytest.approx(0.125 / np.sqrt(2))
            assert not row.witness.intersection_points
        stats = device.stats()
        assert stats["triangle_completed_launches"] > 0
        (tmp_path / "observation.json").write_text(
            json.dumps({"name": name, "pair": asdict(row), "stats": stats}, allow_nan=False)
        )
