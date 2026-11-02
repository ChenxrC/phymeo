"""Explicit M3 comparison. Invoke directly with pytest; excluded from normal discovery."""

import json
from statistics import median
from time import perf_counter_ns

import pytest
from spatial_index_cases import mesh_scene, sheet
from test_spatial_broad_phase import balls

import phymeo as ph


def compare(backend, tmp_path):
    runs = {}
    for name in ("sparse65", "dense34", "sheet18", "sheet72", "mesh28", "mesh112"):
        if name in ("sparse65", "dense34"):
            n = 65 if name == "sparse65" else 34
            model, p = balls(n, name == "dense34")
            frames = [p]
        elif name.startswith("sheet"):
            model, frames = sheet(3 if name == "sheet18" else 6)
        else:
            model, frames = mesh_scene(name == "mesh112")
        modes = (("all_pairs", "all_facets"), ("sap", "all_facets"), ("bvh", "all_facets"))
        if name.startswith("mesh"):
            # Hold shape broad phase constant to isolate per-facet pruning.
            modes = (("all_pairs", "all_facets"), ("all_pairs", "bvh"))
        oracle = {}
        records = {}
        for broad, mid in modes:
            started = perf_counter_ns()
            with ph.CollisionWorkspace(
                model, backend, broad_phase=broad, mesh_midphase=mid, cache_root=tmp_path / "cache"
            ) as work:
                built_ns = perf_counter_ns() - started
                policy = ph.CollisionPolicy()
                cold = perf_counter_ns()
                initial = ph.detect(model, frames[0], policy, work)
                cold_ns = perf_counter_ns() - cold
                if 0 in oracle:
                    assert initial == oracle[0]
                else:
                    oracle[0] = initial
                samples = []
                for repeat in range(3):
                    for i, p in enumerate(frames):
                        before = work.stats()
                        tick = perf_counter_ns()
                        out = ph.detect(model, p, policy, work)
                        wall = perf_counter_ns() - tick
                        if i not in oracle:
                            oracle[i] = out
                        assert out == oracle[i]
                        after = work.stats()
                        delta = {k: after[k] - v for k, v in before.items() if isinstance(v, int)}
                        samples.append(
                            dict(
                                repeat=repeat,
                                frame=i,
                                wall_ns=wall,
                                candidates=out.required_candidates,
                                work=delta,
                            )
                        )
                cycle = [sum(s["wall_ns"] for s in samples if s["repeat"] == i) for i in range(3)]
                records[broad + "/" + mid] = dict(
                    shapes=len(model.shapes),
                    pair_slots=len(initial.pairs),
                    vertices=sum(len(s.vertex_ids) for s in model.shapes),
                    frames=len(frames),
                    construction_ns=built_ns,
                    cold_ns=cold_ns,
                    median_warm_cycle_ns=int(median(cycle)),
                    samples=samples,
                )
        runs[name] = records
    (tmp_path / "spatial-cost.json").write_text(
        json.dumps(dict(backend=backend, workloads=runs), allow_nan=False)
    )


def test_spatial_workloads_host(tmp_path):
    compare("host", tmp_path)


@pytest.mark.tyx
def test_spatial_workloads_device(tyx_mode, tmp_path):
    compare(tyx_mode, tmp_path)
