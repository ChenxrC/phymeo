"""Required tyx projections, interval proposals, material patches and publication."""

import json
from dataclasses import asdict

import pytest
from convex_hull_cases import (
    NAMES,
    authored,
    corners,
    isolated,
    long_axis_table,
    over_axis_budget,
    passing,
    scene,
    sliding_frames,
    tetrahedra,
)

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_device_hull_contacts_intervals_and_transactions(tyx_mode, tmp_path, monkeypatch):
    scenes = [(name, authored(name, i, reverse=bool(i % 2))) for i, name in enumerate(NAMES)]
    for name, make in (("passing", passing), ("isolated", isolated), ("tetrahedra", tetrahedra)):
        scenes.append((name, make(len(scenes))))
    shapes, paths = [], {}
    for _, (m, p) in scenes:
        shapes.extend(m.shapes)
        paths.update(p)
    model = ph.CollisionModel(tuple(reversed(shapes)))
    policy = ph.CollisionPolicy(margin=0.03)
    with ph.CollisionWorkspace(model) as cpu:
        ref = ph.detect(model, paths, policy, cpu)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        out = ph.detect(model, paths, policy, work)
        assert (out.pairs, out.bounds) == (ref.pairs, ref.bounds)
        rows = [r for r in out.pairs if r.first.world == r.second.world]
        assert [r.status for r in rows] == [
            "hit",
            "hit",
            "separated",
            "hit",
            "hit",
            "hit",
            "hit",
            "hit",
        ]
        stats = work.stats()
        assert stats["hull_axis_terms"] == sum(r.convex.axis_count for r in rows)
        assert (
            stats["hull_completed_launches"] >= len(scenes) and stats["hull_nonempty_intervals"] > 0
        )
        exceeded = ph.detect(
            model, paths, ph.CollisionPolicy(candidate_capacity=0, margin=0.03), work
        )
        assert (
            exceeded.status == "capacity_exceeded"
            and exceeded.required_candidates == out.required_candidates
        )
        assert (
            work.last_result is out
            and work.stats()["hull_completed_launches"] == stats["hull_completed_launches"]
        )
        retry = ph.detect(model, paths, policy, work)
        assert retry == out and work.stats()["allocations"] == stats["allocations"]
        (tmp_path / "observation.json").write_text(
            json.dumps(
                dict(
                    rows={name: asdict(row) for (name, _), row in zip(scenes, rows, strict=True)},
                    pairs=len(out.pairs),
                    candidates=out.required_candidates,
                    stats=work.stats(),
                ),
                allow_nan=False,
            )
        )

        def fail(*_):
            raise RuntimeError("injected convex narrow failure")

        monkeypatch.setattr(work._engine, "hull_intervals", fail)
        with pytest.raises(RuntimeError, match="injected"):
            ph.detect(model, paths, policy, work)
        assert work.last_result is retry
        with pytest.raises(RuntimeError, match="closed"):
            ph.detect(model, paths, policy, work)


def test_device_exact_gap_can_reject_all_rounded_overlap_hints(tyx_mode, tmp_path):
    a = corners({"center": [0, 0, 0], "half": [0.5, 0.5, 0.5]})
    model, paths = scene([(a, None, range(8)), (a + [0, 0, 1 + 2**-30], None, range(8))])
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        (row,) = ph.detect(model, paths, ph.CollisionPolicy(margin=0.125), work).pairs
        assert row.candidate and row.status == "separated" and not row.convex.start.patches
        assert row.witness.gap_m == 2**-30
        stats = work.stats()
        assert stats["hull_nonempty_intervals"] == stats["hull_axis_terms"] == 3
        (tmp_path / "observation.json").write_text(
            json.dumps(dict(row=asdict(row), stats=stats), allow_nan=False)
        )


def test_device_hull_axis_pages_cover_general_point_clouds(tyx_mode, tmp_path):
    model, paths = long_axis_table()
    with ph.CollisionWorkspace(model) as cpu:
        ref = ph.detect(model, paths, ph.CollisionPolicy(), cpu)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        out = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert out.pairs == ref.pairs and out.bounds == ref.bounds
        (row,) = out.pairs
        assert row.status == "overlap" and row.convex.axis_count > 256
        stats = work.stats()
        assert stats["hull_axis_terms"] == row.convex.axis_count
        assert stats["hull_completed_launches"] == (row.convex.axis_count + 255) // 256
        if tyx_mode == "cuda":
            assert (
                stats["hull_synchronizations"]
                == stats["hull_downloads"]
                == stats["hull_completed_launches"]
            )
        (tmp_path / "observation.json").write_text(
            json.dumps(dict(row=asdict(row), stats=stats), allow_nan=False)
        )


def test_device_history_and_axis_limit(tyx_mode, tmp_path):
    rows, stats = sliding_frames(tyx_mode, tmp_path / "history-cache")
    model, paths = over_axis_budget()
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "large-cache") as work:
        (limit,) = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
        assert (
            limit.candidate
            and limit.status == "unresolved"
            and limit.reason == "convex_axis_capacity"
        )
        assert work.stats()["hull_completed_launches"] == 0
    (tmp_path / "history.json").write_text(
        json.dumps(
            dict(rows=list(map(asdict, rows)), limit=asdict(limit), stats=stats), allow_nan=False
        )
    )
