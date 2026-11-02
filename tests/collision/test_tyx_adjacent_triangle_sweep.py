"""Required tyx backends on connected and independently moving incident features."""

import json
from dataclasses import asdict

import numpy as np
import pytest
from adjacent_triangle_cases import NAMES, authored, hinge_path, row_for, vertex_path

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_device_incident_catalog_and_full_paths(tyx_mode, tmp_path, monkeypatch):
    shapes = []
    motions = {}
    keys = []
    labels = []
    for world, name in enumerate(NAMES):
        sub, paths, target = authored(name, world)
        shapes.extend(sub.shapes)
        motions.update(paths)
        keys.append(target)
        labels.append(name)
    for name, make in (
        ("moving_hinge_clear", lambda w: hinge_path(False, w)),
        ("moving_hinge_hit", lambda w: hinge_path(True, w)),
        ("vertex_birth", vertex_path),
    ):
        sub, paths = make(len(labels))
        shapes.extend(sub.shapes)
        motions.update(paths)
        keys.append(tuple(s.key for s in sub.shapes))
        labels.append(name)
    model = ph.CollisionModel(shapes)
    with ph.CollisionWorkspace(model) as host:
        reference = ph.detect(model, motions, ph.CollisionPolicy(), host)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        out = ph.detect(model, motions, ph.CollisionPolicy(), work)
        assert reference.pairs == out.pairs and reference.bounds == out.bounds
        rows = [row_for(out, k) for k in keys]
        assert [r.status for r in rows] == [
            "touching",
            "adjacent",
            "adjacent",
            "touching",
            "touching",
            "adjacent",
            "hit",
            "hit",
        ]
        assert all(r.candidate for r in rows)
        stats = work.stats()
        assert (
            stats["triangle_completed_launches"] > 0
            and stats["triangle_edge_edge_proposed_hits"] > 0
        )
        exceeded = ph.detect(model, motions, ph.CollisionPolicy(candidate_capacity=0), work)
        assert (
            exceeded.status == "capacity_exceeded"
            and exceeded.required_candidates == out.required_candidates
        )
        assert (
            work.last_result is out
            and work.stats()["triangle_completed_launches"] == stats["triangle_completed_launches"]
        )
        assert ph.detect(model, motions, ph.CollisionPolicy(), work) == out
        assert work.stats()["allocations"] == stats["allocations"]
        (tmp_path / "observation.json").write_text(
            json.dumps(
                {
                    "labels": labels,
                    "rows": [asdict(r) for r in rows],
                    "pair_count": len(out.pairs),
                    "candidate_count": out.required_candidates,
                    "stats": work.stats(),
                },
                allow_nan=False,
            )
        )

        def fault(_features):
            raise RuntimeError("injected adjacency failure")

        monkeypatch.setattr(work._engine, "entry_times", fault)
        with pytest.raises(RuntimeError, match="injected"):
            ph.detect(model, motions, ph.CollisionPolicy(), work)
        assert work.last_result == out
        with pytest.raises(RuntimeError, match="closed"):
            ph.detect(model, motions, ph.CollisionPolicy(), work)


def test_device_reordered_material_in_moving_frame(tyx_mode, tmp_path):
    base, paths = vertex_path()
    r = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    shift = np.array([2, -3, 1])
    drift = np.array([0.5, 0.25, -0.5])
    shapes = []
    transformed = {}
    for shape in base.shapes:
        order = [2, 0, 1]
        p = paths[shape.key]
        shapes.append(
            ph.CollisionShape(shape.key, "triangle", tuple(shape.vertex_ids[i] for i in order))
        )
        transformed[shape.key] = ph.CollisionMotion(
            p.start[order] @ r.T + shift, p.end[order] @ r.T + shift + drift
        )
    model = ph.CollisionModel(tuple(reversed(shapes)))
    with (
        ph.CollisionWorkspace(model) as host,
        ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work,
    ):
        ref = ph.detect(model, transformed, ph.CollisionPolicy(), host)
        out = ph.detect(model, transformed, ph.CollisionPolicy(), work)
        assert out.pairs == ref.pairs and out.bounds == ref.bounds
        (row,) = out.pairs
        assert row.sweep.first_t == (0.5, 0.5)
        w = row.sweep.witness
        np.testing.assert_allclose(w.first_point, [2.25, -2.375, 0.75], atol=1e-12)
        assert dict(w.first_weights) == {0: 0.875, 1: 0.125, 2: 0}
        assert dict(w.second_weights) == {0: 0.5, 3: 0.25, 4: 0.25}
        assert work.stats()["triangle_edge_edge_proposed_hits"] >= 1
        (tmp_path / "observation.json").write_text(
            json.dumps({"row": asdict(row), "stats": work.stats()}, allow_nan=False)
        )
