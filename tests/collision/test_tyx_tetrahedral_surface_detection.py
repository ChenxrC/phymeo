"""Device execution through the same public sphere/cloth/volume detector."""

import json
from dataclasses import asdict

import pytest
from tetrahedral_surface_cases import CASES, partners, scene

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_device_surface_catalog_material_rows_and_transactions(tyx_mode, tmp_path, monkeypatch):
    shapes, paths, targets = [], {}, []
    for family, names in CASES.items():
        for name in names:
            model, motions, key = scene(family, name, len(targets), reverse=(len(targets) % 2 == 1))
            targets.append((family, name, key))
            shapes.extend(model.shapes)
            paths.update(motions)
    model = ph.CollisionModel(shapes)
    with ph.CollisionWorkspace(model) as cpu:
        reference = ph.detect(model, paths, ph.CollisionPolicy(), cpu)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        out = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert out.pairs == reference.pairs and out.bounds == reference.bounds
        stats = work.stats()
        assert stats["surface_completed_launches"] > 0 and stats["surface_proposed_hits"] > 0
        assert stats["triangle_completed_launches"] > 0 and stats["triangle_proposed_hits"] > 0
        assert stats["triangle_edge_edge_features"] > 0
        small = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0), work)
        assert (
            small.status == "capacity_exceeded"
            and small.required_candidates == out.required_candidates
        )
        assert (
            work.last_result is out
            and work.stats()["sweep_completed_launches"] == stats["sweep_completed_launches"]
        )
        assert ph.detect(model, paths, ph.CollisionPolicy(), work) == out
        assert work.stats()["allocations"] == stats["allocations"]
        rows = {
            f"{fam}/{name}": [asdict(row) for row in partners(out, key)]
            for fam, name, key in targets
        }
        (tmp_path / "observation.json").write_text(
            json.dumps(
                dict(
                    rows=rows,
                    stats=work.stats(),
                    required=out.required_candidates,
                    pairs=len(out.pairs),
                ),
                allow_nan=False,
            )
        )

        def fail(_features):
            raise RuntimeError("injected facet dispatch failure")

        monkeypatch.setattr(work._engine, "entry_times", fail)
        with pytest.raises(RuntimeError, match="injected facet"):
            ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert work.last_result == out
        with pytest.raises(RuntimeError, match="closed"):
            ph.detect(model, paths, ph.CollisionPolicy(), work)


def test_device_positive_static_gap_below_binary64(tyx_mode, tmp_path):
    tet, sphere = ph.CollisionKey(0, 0, 0), ph.CollisionKey(0, 1, 0)
    model = ph.CollisionModel(
        (
            ph.CollisionShape(tet, "tetrahedron", (0, 1, 2, 3)),
            ph.CollisionShape(sphere, "sphere", (0,), 1.0),
        )
    )
    paths = {
        tet: ph.CollisionMotion([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        sphere: ph.CollisionMotion([[-1.0, -(2.0**-540), 0.0]]),
    }
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        (row,) = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
        assert row.witness.gap_m == 0.0 and row.status == "surface_clear"
        (tmp_path / "observation.json").write_text(json.dumps(asdict(row), allow_nan=False))
