"""Compare identical G4 and G1 outputs across explicit pages and bounded workspace residency."""

import argparse
import json
import sys
from contextlib import ExitStack
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import phymeo as ph
from phymeo.geometry.paged_collision import CollisionPageLimits, PagedWorldCollisionWorkspace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "scale"))
from cases import g1_model, inspect_phymeo  # noqa: E402
from g4_geometry_cases import WORLD_IDS, check_geometry, copies, digest, frozen  # noqa: E402


def scene(name):
    if name in ("sparse65", "sheet72"):
        m, frames = frozen(name)
        group, worldframes = copies(m, frames, WORLD_IDS)
        return group, worldframes, None
    keys = tuple(1000 + 7 * i for i in range(128))
    group = []
    frame = {}
    for key in keys:
        model, _, paths = g1_model((key,))
        group.append((key, model))
        frame[key] = paths
    moved = keys[76]
    k = ph.CollisionKey(moved, 1, 0)
    neg = dict(frame)
    neg[moved] = {**frame[moved], k: ph.CollisionMotion([[5 / 16, 0, 0]])}
    return group, [frame, neg], moved


def measure_case(name, backend, cache, samples):
    groups, frames, moved = scene(name)
    models = dict(groups)
    rules = dict.fromkeys(models, ph.CollisionPolicy())
    result = {}
    reference = {}

    def native():
        if backend != "cuda":
            return {}
        from tyx import runtime_stats

        s = runtime_stats()
        return {
            key: getattr(s, key)
            for key in (
                "cuda_kernel_launches",
                "cuda_h2d_bytes",
                "cuda_d2h_bytes",
                "cuda_d2d_bytes",
            )
        }

    for mode in ("manual", "resident2", "stream1"):
        with ExitStack() as stack:
            begin = perf_counter_ns()
            planner = stack.enter_context(
                PagedWorldCollisionWorkspace(
                    groups,
                    backend,
                    limits=CollisionPageLimits(resident_pages=1 if mode == "stream1" else 2),
                    cache_root=cache,
                )
            )
            if mode == "manual":
                engines = [
                    (
                        p.world_keys,
                        stack.enter_context(
                            ph.WorldCollisionWorkspace(
                                [(k, models[k]) for k in p.world_keys], backend, cache_root=cache
                            )
                        ),
                    )
                    for p in planner.layout
                ]

                def query(frame):
                    rows = {}
                    for keys, work in engines:
                        rows.update(
                            work.detect({k: frame[k] for k in keys}, {k: rules[k] for k in keys})
                        )
                    return rows
            else:

                def query(frame):
                    return planner.detect(frame, rules)

            built = perf_counter_ns() - begin

            def take(frame):
                low = native()
                start = perf_counter_ns()
                found = query(frame)
                wall = perf_counter_ns() - start
                high = native()
                return found, dict(wall_ns=wall, native={k: high[k] - low[k] for k in high})

            verified = {}
            counts = {}

            def check(index, observed):
                assert tuple(observed) == tuple(models)
                if index in verified:
                    assert observed == verified[index]
                    return
                hashes = {}
                hits = 0
                for world, out in observed.items():
                    hashes[world] = digest(out, world)
                    if name == "g1-128":
                        observation = inspect_phymeo(
                            models[world], (world,), out, moved if index else None
                        )
                        hits += len(observation["contacts"])
                    else:
                        audit = check_geometry(
                            name,
                            models[world],
                            frames[index][world],
                            out,
                            0 if name == "sparse65" else 4,
                        )
                        hits += audit["hits"]
                if mode == "manual":
                    reference[index] = hashes
                assert hashes == reference[index]
                verified[index] = observed
                counts[index] = dict(
                    worlds=len(observed),
                    pair_rows=sum(len(v.pairs) for v in observed.values()),
                    candidates=sum(v.required_candidates for v in observed.values()),
                    hits=hits,
                    world_sha256=hashes,
                )

            first, cold = take(frames[0])
            check(0, first)
            records = []
            for repeat in range(samples):
                for idx, frame in enumerate(frames):
                    rows, cost = take(frame)
                    check(idx, rows)
                    records.append(dict(repeat=repeat, frame=idx, **cost))
            cycles = [
                sum(s["wall_ns"] for s in records if s["repeat"] == i) for i in range(samples)
            ]
            stats = (
                {"manual_pages": [w.stats() for _, w in engines]}
                if mode == "manual"
                else planner.stats()
            )
            result[mode] = dict(
                registration_ns=built,
                cold=cold,
                frames=counts,
                samples=records,
                warm_cycle_ns=cycles,
                median_warm_cycle_ns=median(cycles),
                layout=[asdict(p) for p in planner.layout],
                stats=stats,
            )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("host", "cuda"), required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=3)
    ns = p.parse_args()
    if ns.samples < 1:
        p.error("samples must be positive")
    report = dict(
        schema="phymeo-paged-world-cost/1",
        backend=ns.backend,
        source=ph.__file__,
        runtime=dict(
            python=sys.version.split()[0],
            numpy=version("numpy"),
            tyx=version("tyx-compute") if ns.backend == "cuda" else None,
        ),
        physical_steps=0,
        cases={},
    )
    if ns.backend == "cuda":
        from benchmark_worlds import gpu_memory

        report["whole_device_before"] = gpu_memory()
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    for name in ("sparse65", "sheet72", "g1-128"):
        report["cases"][name] = measure_case(name, ns.backend, ns.cache_root, ns.samples)
        ns.out.write_text(json.dumps(report, allow_nan=False, indent=2), encoding="utf8")
        print(
            name,
            [round(v["median_warm_cycle_ns"] / 1e6, 3) for v in report["cases"][name].values()],
            flush=True,
        )


if __name__ == "__main__":
    main()
