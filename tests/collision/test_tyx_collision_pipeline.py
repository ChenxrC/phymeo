"""Run the M1 candidate kernels on the explicitly requested real tyx backend."""

import json
from dataclasses import asdict
from time import perf_counter_ns

import numpy as np
import pytest
from m1_cases import mixed_scene, saved_inputs, verify_mixed

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_device_candidates_reuse_storage_and_recover_from_capacity(tyx_mode, tmp_path, monkeypatch):
    model, paths, keys, expected = mixed_scene(reverse=True)
    options = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model) as host:
        baseline = ph.detect(model, paths, options, host)
    import phymeo.geometry.collision_pipeline as pipeline

    def forbidden(*args):
        raise AssertionError("device path fell back to host broad phase")

    monkeypatch.setattr(pipeline, "_host_query", forbidden)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        out = ph.detect(model, paths, options, work)
        verify_mixed(out, keys, expected)
        assert out.pairs == baseline.pairs and out.bounds == baseline.bounds
        s1 = work.stats()
        assert s1["backend"] == tyx_mode and s1["completed_launches"] == 2
        failed = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0), work)
        assert (
            failed.status == "capacity_exceeded"
            and failed.required_candidates == out.required_candidates
        )
        assert work.last_result is out
        distant = dict(paths)
        distant[keys[1]] = ph.CollisionMotion([[30, 0, 0]])
        updated = ph.detect(model, distant, options, work)
        assert not next(
            p for p in updated.pairs if (p.first, p.second) == (keys[0], keys[1])
        ).candidate
        start = perf_counter_ns()
        again = ph.detect(model, paths, options, work)
        warm_query_ns = perf_counter_ns() - start
        assert again == out
        s2 = work.stats()
        assert s2["completed_launches"] == 8 and s1["allocations"] == s2["allocations"]
        if tyx_mode == "cuda":
            assert s1["static_uploads"] > 0 and s1["static_uploads"] == s2["static_uploads"]
            assert s2["dynamic_uploads"] == 4 and s2["downloads"] == 8
            assert s2["synchronizations"] == 4 and s2["uploaded_bytes"] > 0
        (tmp_path / "device-result.json").write_text(
            json.dumps(
                dict(
                    result=asdict(out),
                    inputs=saved_inputs(model, paths, options),
                    capacity=asdict(failed),
                    stats=s2,
                    warm_query_ns=warm_query_ns,
                ),
                allow_nan=False,
            )
        )
        before = work.last_result

        def fault(*args):
            raise RuntimeError("injected device failure")

        monkeypatch.setattr(work._engine, "query", fault)
        with pytest.raises(RuntimeError, match="injected"):
            ph.detect(model, paths, options, work)
        assert work.last_result is before
        with pytest.raises(RuntimeError, match="closed"):
            ph.detect(model, paths, options, work)


def test_native_outward_rounding_at_single_precision_boundary(tyx_mode, tmp_path):
    a, b = ph.CollisionKey(0, 0, 0), ph.CollisionKey(0, 1, 0)
    model = ph.CollisionModel(
        [ph.CollisionShape(a, "sphere", (1,), 0.125), ph.CollisionShape(b, "sphere", (1,), 0.125)]
    )
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        result = ph.detect(
            model,
            {
                a: ph.CollisionMotion([[2**24 + 1, 0, 0]]),
                b: ph.CollisionMotion([[2**24 + 1.25, 0, 0]]),
            },
            ph.CollisionPolicy(),
            work,
        )
        (row,) = result.pairs
        assert row.status == "touching" and row.candidate
        assert result.bounds[0].lower[0] <= 2**24 + 0.875
        np.testing.assert_array_less(2**24 + 1.124, result.bounds[0].upper[0])
