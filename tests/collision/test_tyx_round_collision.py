"""Required tyx backends exercise actual narrow-phase proposals and verified CCD."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
from round_cases import RR01, RR02, catalog_scene, crossing_capsules

import phymeo as ph

pytestmark = pytest.mark.tyx


@pytest.mark.parametrize(
    "name", ["base", "axis_holdout", "sphere_pair", "crossing_capsules", "far_plane"]
)
def test_device_round_sweep(name, tyx_mode, tmp_path):
    if name == "crossing_capsules":
        model, paths = crossing_capsules()
    else:
        model, paths, _, _ = catalog_scene("RR01", "base" if name == "far_plane" else name)
        if name == "far_plane":
            paths[min(paths)] = ph.CollisionMotion([[1e100, 0, 0]])
    policy = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model) as host:
        truth = ph.detect(model, paths, policy, host)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        out = ph.detect(model, paths, policy, work)
        (row,) = out.pairs
        assert row.status == "hit" and out.pairs == truth.pairs and out.bounds == truth.bounds
        stats = work.stats()
        assert stats["sweep_completed_launches"] >= 1 and stats["sweep_features"] >= 1
        assert (
            stats["sweep_proposed_hits"] >= 1
            and stats["sweep_completed_launches"] == stats["sweep_launches"]
        )
        failed = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0), work)
        assert failed.status == "capacity_exceeded" and work.last_result is out
        assert work.stats()["sweep_launches"] == stats["sweep_launches"]
        again = ph.detect(model, paths, policy, work)
        assert again == out and work.stats()["sweep_allocations"] == stats["sweep_allocations"]
        if tyx_mode == "cuda":
            assert stats["sweep_uploads"] >= 1 and stats["sweep_downloads"] >= 1
        (tmp_path / "observation.json").write_text(
            json.dumps(
                {"name": name, "result": asdict(out), "stats": work.stats()}, allow_nan=False
            )
        )


def test_device_grazing_and_one_ulp_miss(tyx_mode, tmp_path):
    a, b = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    model = ph.CollisionModel(
        [
            ph.CollisionShape(a, "capsule", (2, 4), 0.125),
            ph.CollisionShape(b, "sphere", (1,), 0.125),
        ]
    )
    observations = []
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        for x, expected in ((1.25, "hit"), (np.nextafter(1.25, np.inf), "separated")):
            paths = {
                a: ph.CollisionMotion([[0, 0, 0], [1, 0, 0]]),
                b: ph.CollisionMotion([[x, -1, 0]], [[x, 1, 0]]),
            }
            (row,) = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
            assert row.candidate and row.status == expected
            assert row.sweep.first_t == ((0.5, 0.5) if expected == "hit" else None)
            observations.append(asdict(row))
        assert work.stats()["sweep_completed_launches"] == 2
        (tmp_path / "observation.json").write_text(
            json.dumps(
                {"name": "device_graze", "rows": observations, "stats": work.stats()},
                allow_nan=False,
            )
        )


def test_device_pages_cover_every_translating_capsule_pair(tyx_mode, tmp_path):
    keys = [ph.CollisionKey(0, i, 0) for i in range(19)]
    model = ph.CollisionModel([ph.CollisionShape(k, "capsule", (1, 2), 0.125) for k in keys])
    paths = {
        k: ph.CollisionMotion(
            [[-1, 0, i / 1024], [1, 0, i / 1024]], [[-0.5, 0, i / 1024], [1.5, 0, i / 1024]]
        )
        for i, k in enumerate(keys)
    }
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        out = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert len(out.pairs) == out.required_candidates == 171
        assert all(row.status == "hit" and row.sweep.first_t == (0.0, 0.0) for row in out.pairs)
        stats = work.stats()
        assert stats["sweep_features"] > 512 and stats["sweep_completed_launches"] >= 2
        assert stats["sweep_allocations"] == 2
        (tmp_path / "observation.json").write_text(
            json.dumps({"name": "pages", "required": 171, "stats": stats}, allow_nan=False)
        )


def test_registered_rr_matrix_in_one_device_query(tyx_mode, tmp_path, monkeypatch):
    shapes, paths, excluded, labels = [], {}, [], {}
    for world, (case, name) in enumerate([("RR01", n) for n in RR01] + [("RR02", n) for n in RR02]):
        model, motion, variant, _ = catalog_scene(case, name)
        new = {s.key: replace(s.key, world=world) for s in model.shapes}
        for s in model.shapes:
            k = new[s.key]
            shapes.append(replace(s, key=k))
            paths[k] = motion[s.key]
        excluded.extend((new[a], new[b]) for a, b in model.excluded)
        labels[world] = (case, name, variant)
    model = ph.CollisionModel(tuple(reversed(shapes)), excluded)
    assert len(model.shapes) == 22
    policy = ph.CollisionPolicy(margin=0.25)
    with ph.CollisionWorkspace(model) as host:
        reference = ph.detect(model, paths, policy, host)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        result = ph.detect(model, paths, policy, work)
        assert result.pairs == reference.pairs and result.bounds == reference.bounds
        assert len(result.pairs) == 231 and result.required_candidates == 10
        records = []
        for pair in result.pairs:
            if pair.first.world != pair.second.world:
                assert pair.status == "filtered" and pair.reason == "different_world"
                continue
            case, name, variant = labels[pair.first.world]
            expected = variant["expected"]
            if name == "filtered":
                assert pair.status == "filtered" and pair.reason == "excluded"
            elif case == "RR01":
                if name == "clear":
                    assert pair.status == "separated" and pair.sweep.first_t is None
                else:
                    lo, hi = pair.sweep.first_t
                    t = expected["first_t"]
                    assert lo - 1e-14 <= t <= hi + 1e-14 and hi - lo <= 1e-9
            else:
                gap = expected["gap_m"]
                assert pair.witness.gap_m == pytest.approx(gap, abs=1e-12)
                assert pair.status == (
                    "overlap" if gap < 0 else "separated" if gap > 0 else "touching"
                )
            records.append({"case": case, "name": name, "pair": asdict(pair)})
        assert len(records) == 11
        stats = work.stats()
        assert stats["sweep_completed_launches"] >= 1
        (tmp_path / "observation.json").write_text(
            json.dumps(
                {
                    "name": "catalog_matrix",
                    "records": records,
                    "pairs": len(result.pairs),
                    "stats": stats,
                },
                allow_nan=False,
            )
        )

        def injected(_features):
            raise RuntimeError("injected narrow-phase failure")

        monkeypatch.setattr(work._engine, "entry_times", injected)
        with pytest.raises(RuntimeError, match="injected"):
            ph.detect(model, paths, policy, work)
        assert work.last_result is result
        with pytest.raises(RuntimeError, match="closed"):
            ph.detect(model, paths, policy, work)
