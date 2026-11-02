"""Required interpret/CUDA broad phase and feature kernels on distinct filled bodies."""

import json
from dataclasses import asdict

import pytest
from tetrahedral_solid_cases import (
    NAMES,
    across_seam,
    authored,
    crossed_cells,
    crossing_internal_face,
    edge_birth,
    invalid_material_mesh,
)

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_device_closed_body_catalog_and_transactions(tyx_mode, tmp_path, monkeypatch):
    shapes, paths = [], {}
    scenes = [(name, authored(name, w)) for w, name in enumerate(NAMES)]
    for name, make in (
        ("across_seam", across_seam),
        ("crossed_cells", crossed_cells),
        ("edge_birth", edge_birth),
        ("internal_crossing", crossing_internal_face),
    ):
        scenes.append((name, make(len(scenes))))
    scenes.append(("coincident_material_cells", invalid_material_mesh(len(scenes))))
    scenes.append(("crossing_material_cells", invalid_material_mesh(len(scenes), crossing=True)))
    for _, (model, motion) in scenes:
        shapes.extend(model.shapes)
        paths.update(motion)
    model = ph.CollisionModel(tuple(reversed(shapes)))
    with ph.CollisionWorkspace(model) as cpu:
        ref = ph.detect(model, paths, ph.CollisionPolicy(), cpu)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        out = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert (out.pairs, out.bounds, out.solid_pairs) == (ref.pairs, ref.bounds, ref.solid_pairs)
        assert len(out.solid_pairs) == len(scenes)
        assert [b.start.relation if b.start else b.status for b in out.solid_pairs] == [
            "first_contains_second",
            "separated",
            "invalid_geometry",
            "second_contains_first",
            "separated",
            "separated",
            "first_contains_second",
            "overlap",
            "separated",
            "first_contains_second",
            "invalid_geometry",
            "unresolved",
        ]
        stats = work.stats()
        assert (
            stats["triangle_completed_launches"] > 0
            and stats["triangle_edge_edge_proposed_hits"] > 0
        )
        too_small = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0), work)
        assert (
            too_small.status == "capacity_exceeded"
            and too_small.required_candidates == out.required_candidates
        )
        assert not too_small.solid_pairs and work.last_result is out
        assert work.stats()["sweep_completed_launches"] == stats["sweep_completed_launches"]
        assert ph.detect(model, paths, ph.CollisionPolicy(), work) == out
        assert work.stats()["allocations"] == stats["allocations"]
        labels = [name for name, _ in scenes]
        (tmp_path / "observation.json").write_text(
            json.dumps(
                dict(
                    bodies={labels[b.world]: asdict(b) for b in out.solid_pairs},
                    cells={
                        name: [
                            asdict(row)
                            for row in out.pairs
                            if row.first.world == w == row.second.world
                            and row.first.body != row.second.body
                        ]
                        for w, name in enumerate(labels)
                    },
                    pairs=len(out.pairs),
                    candidates=out.required_candidates,
                    stats=work.stats(),
                ),
                allow_nan=False,
            )
        )

        def fail(_features):
            raise RuntimeError("injected closed cell failure")

        monkeypatch.setattr(work._engine, "entry_times", fail)
        with pytest.raises(RuntimeError, match="injected"):
            ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert work.last_result == out
        with pytest.raises(RuntimeError, match="closed"):
            ph.detect(model, paths, ph.CollisionPolicy(), work)
