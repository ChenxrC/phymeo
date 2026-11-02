"""Required tyx surface kernels on finite capsules and moving triangle features."""

import json
from dataclasses import asdict

import pytest
from capsule_face_cases import (
    NAMES,
    authored,
    barrel_edge,
    deforming_edge,
    edge_graze,
    parallel_turn,
    shaft_graze,
    tangent_span,
)

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_device_capsule_face_catalog_and_transactions(tyx_mode, tmp_path, monkeypatch):
    scenes = [(name, authored(name, w, swap=bool(w % 2))) for w, name in enumerate(NAMES)]
    for name, make in (
        ("barrel_edge", barrel_edge),
        ("deforming_edge", deforming_edge),
        ("tangent_span", tangent_span),
        ("shaft_graze", shaft_graze),
        ("shaft_miss", lambda w: shaft_graze(w, miss=True)),
        ("parallel_clear", parallel_turn),
        ("parallel_entry", lambda w: parallel_turn(w, enters=True)),
        ("edge_graze", edge_graze),
        ("edge_near_miss", lambda w: edge_graze(w, miss=True)),
    ):
        scenes.append((name, make(len(scenes))))
    shapes, paths = [], {}
    for _, (m, p, _, _) in scenes:
        shapes.extend(m.shapes)
        paths.update(p)
    model = ph.CollisionModel(tuple(reversed(shapes)))
    policy = ph.CollisionPolicy(margin=8.0)
    with ph.CollisionWorkspace(model) as cpu:
        ref = ph.detect(model, paths, policy, cpu)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        out = ph.detect(model, paths, policy, work)
        assert (out.pairs, out.bounds) == (ref.pairs, ref.bounds)
        targets = [q for q in out.pairs if q.first.world == q.second.world]
        assert [r.status for r in targets] == [
            "overlap",
            "separated",
            "touching",
            "overlap",
            "overlap",
            "hit",
            "hit",
            "hit",
            "hit",
            "separated",
            "separated",
            "hit",
            "hit",
            "separated",
        ]
        stats = work.stats()
        assert (
            stats["surface_completed_launches"] > 0
            and stats["capsule_shaft_edge_proposed_hits"] > 0
        )
        assert (
            stats["capsule_shaft_vertex_features"] > 0
            and stats["capsule_shaft_vertex_proposed_hits"] > 0
        )
        exceeded = ph.detect(
            model, paths, ph.CollisionPolicy(candidate_capacity=0, margin=8.0), work
        )
        assert (
            exceeded.status == "capacity_exceeded"
            and exceeded.required_candidates == out.required_candidates
        )
        assert (
            work.last_result is out
            and stats["sweep_completed_launches"] == work.stats()["sweep_completed_launches"]
        )
        retry = ph.detect(model, paths, policy, work)
        assert retry == out and work.last_result is retry
        assert work.stats()["allocations"] == stats["allocations"]
        labels = [name for name, _ in scenes]
        (tmp_path / "observation.json").write_text(
            json.dumps(
                dict(
                    rows=dict(zip(labels, map(asdict, targets), strict=True)),
                    pairs=len(out.pairs),
                    candidates=out.required_candidates,
                    stats=work.stats(),
                ),
                allow_nan=False,
            )
        )

        def fail(_features):
            raise RuntimeError("injected capsule surface failure")

        monkeypatch.setattr(work._engine, "entry_times", fail)
        with pytest.raises(RuntimeError, match="injected"):
            ph.detect(model, paths, policy, work)
        assert work.last_result is retry
        with pytest.raises(RuntimeError, match="closed"):
            ph.detect(model, paths, policy, work)


def test_device_isolated_graze_between_samples(tyx_mode, tmp_path):
    model, paths, _, _ = shaft_graze()
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        (row,) = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
        assert row.status == "hit" and row.sweep.first_t[0] <= 1 / 3 <= row.sweep.first_t[1]
        assert work.stats()["surface_proposed_hits"] == 0
        (tmp_path / "observation.json").write_text(
            json.dumps(dict(row=asdict(row), stats=work.stats()), allow_nan=False)
        )


def test_device_unsampled_edge_root_and_near_miss(tyx_mode, tmp_path):
    shapes, paths = [], {}
    for i in range(2):
        m, p, _, _ = edge_graze(i, miss=bool(i))
        shapes.extend(m.shapes)
        paths.update(p)
    model = ph.CollisionModel(shapes)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        out = ph.detect(model, paths, ph.CollisionPolicy(margin=0.125), work)
        rows = [r for r in out.pairs if r.candidate]
        assert [r.status for r in rows] == ["hit", "separated"]
        assert rows[0].sweep.first_t[0] <= 1 / 3 <= rows[0].sweep.first_t[1]
        stats = work.stats()
        assert stats["capsule_shaft_edge_features"] > 0 and stats["surface_proposed_hits"] == 0
        (tmp_path / "observation.json").write_text(
            json.dumps(dict(rows=list(map(asdict, rows)), stats=stats), allow_nan=False)
        )
