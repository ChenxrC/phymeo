"""The pinned device evaluates the circular chart features on each supported backend."""

import json
from dataclasses import asdict
from math import pi

import pytest
from rotating_round_cases import authored, isolated, scene

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_device_rotating_shaft_caps_and_singleton(tyx_mode, tmp_path, monkeypatch):
    inputs = [authored(name, world=i)[:2] for i, name in enumerate(("base", "clear", "holdout"))]
    inputs.extend((isolated(True, 3), isolated(True, 4, miss=True)))
    for i, high in enumerate((0.25, 0.265625)):
        inputs.append(
            scene(
                [[0.5, 0, 0], [1, 0, 0]],
                [0, 0.75, high],
                pi,
                ball_travel=[0, 0, -0.25 if i == 0 else -0.125],
                world=5 + i,
                radii=(0.0625, 0.0625),
            )
        )
    model = ph.CollisionModel([s for m, _ in inputs for s in m.shapes])
    paths = {k: p for _, ps in inputs for k, p in ps.items()}
    policy = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model) as cpu:
        truth = ph.detect(model, paths, policy, cpu)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as device:
        out = ph.detect(model, paths, policy, device)
        assert out.pairs == truth.pairs and out.bounds == truth.bounds
        active = [p for p in out.pairs if p.first.world == p.second.world]
        assert [p.status for p in active] == [
            "hit",
            "separated",
            "hit",
            "hit",
            "separated",
            "hit",
            "separated",
        ]
        counts = device.stats()
        assert counts["arc_chart_features"] >= 15 and counts["arc_chart_proposed_hits"] > 0
        assert counts["arc_chart_completed_launches"] > 0 and counts["sweep_completed_launches"] > 0
        if tyx_mode == "cuda":
            assert counts["sweep_synchronizations"] > 0 and counts["sweep_downloads"] > 0
        with monkeypatch.context() as m:
            m.setattr(
                device._engine,
                "entry_times",
                lambda _: (_ for _ in ()).throw(RuntimeError("arc proposal fault")),
            )
            with pytest.raises(RuntimeError, match="arc proposal fault"):
                ph.detect(model, paths, policy, device)
            assert device.last_result == out
        (tmp_path / "rr04.json").write_text(
            json.dumps({"pairs": [asdict(p) for p in active], "stats": counts})
        )
