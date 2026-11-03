"""G4 cold/warm queries on the current world-partition entry using original M3 workloads."""

import argparse
import json
import sys
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import phymeo as ph
from phymeo.constraints.collision_shapes import MAX_SHAPES, MAX_VERTICES

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "scale"))
from g4_geometry_cases import NAMES, WORLD_IDS, check_geometry, copies, digest, frozen  # noqa: E402

ARCHIVED_CANDIDATES = {
    "sparse65": [0],
    "dense34": [561],
    "sheet18": [2, 2, 2],
    "sheet72": [4, 4, 4],
    "mesh28": [1, 1, 1],
    "mesh112": [1, 1, 1],
}
STAGES = (
    "engine_setup_ns",
    "envelope_ns",
    "broad_ns",
    "midphase_ns",
    "narrow_ns",
    "completed_launches",
    "synchronizations",
    "allocations",
    "static_uploads",
    "uploaded_bytes",
    "downloaded_bytes",
    "hull_axis_terms",
    "mesh_facet_pairs_total",
    "mesh_facet_pairs_tested",
)
NATIVE = ("cuda_kernel_launches", "cuda_h2d_bytes", "cuda_d2h_bytes", "cuda_d2d_bytes")


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf8")
    tmp.replace(path)


def run(name, count, model, frames, args, ref):
    shape_count = len(model.shapes) * count
    vertices = sum(len(s.vertex_ids) for s in model.shapes) * count
    if shape_count > MAX_SHAPES or vertices > MAX_VERTICES:
        return dict(
            worlds=count,
            status="preflight_limit",
            shapes=shape_count,
            vertices=vertices,
            max_shapes=MAX_SHAPES,
            max_vertices=MAX_VERTICES,
        )
    raw_ids = WORLD_IDS[:count]
    t = perf_counter_ns()
    models, motion = copies(model, frames, raw_ids)
    copy_ns = perf_counter_ns() - t
    model_by_world = dict(models)
    rules = dict.fromkeys(raw_ids, ph.CollisionPolicy())
    if args.backend == "cuda":
        from tyx import runtime_stats

    def native():
        if args.backend == "host":
            return {}
        totals = runtime_stats()
        return {k: getattr(totals, k) for k in NATIVE}

    start = perf_counter_ns()
    workspace = ph.WorldCollisionWorkspace(models, args.backend, cache_root=args.cache_root)
    construction_ns = perf_counter_ns() - start
    with workspace as work:

        def query(paths, policies):
            before = work.stats()
            src = native()
            t0 = perf_counter_ns()
            rows = work.detect(paths, policies)
            duration = perf_counter_ns() - t0
            dest = native()
            after = work.stats()
            return rows, dict(
                wall_ns=duration,
                stages={k: after.get(k, 0) - before.get(k, 0) for k in STAGES},
                native={k: dest[k] - src[k] for k in src},
            )

        checked = {}
        summaries = {}

        def verify(frame, rows):
            assert tuple(rows) == raw_ids
            for world, out in rows.items():
                assert out.status == "ok" and work.last_result(world) is out
                if world in checked.get(frame, {}):
                    assert out == checked[frame][world]
                    continue
                current = digest(out, world)
                if frame not in ref:
                    assert count == 1
                    ref[frame] = current
                assert current == ref[frame], "fully normalized public rows differ from W1"
                if world == raw_ids[0]:
                    audit = check_geometry(
                        name,
                        model_by_world[world],
                        motion[frame][world],
                        out,
                        ARCHIVED_CANDIDATES[name][frame],
                    )
                    summaries[frame] = dict(audit, canonical_sha256=current)
                checked.setdefault(frame, {})[world] = out

        cold, cold_cost = query(motion[0], rules)
        verify(0, cold)
        samples = []
        for repeat in range(args.samples):
            for frame, paths in enumerate(motion):
                rows, cost = query(paths, rules)
                verify(frame, rows)
                samples.append(dict(repeat=repeat, frame=frame, **cost))
        cycle = []
        for n in range(args.samples):
            sequence = [s for s in samples if s["repeat"] == n]
            cycle.append(
                dict(
                    wall_ns=sum(x["wall_ns"] for x in sequence),
                    stages={k: sum(x["stages"][k] for x in sequence) for k in STAGES},
                    native={
                        k: sum(x["native"][k] for x in sequence) for k in sequence[0]["native"]
                    },
                )
            )
        stats = work.stats()
        cap_record = None
        # Check the actual local-capacity transaction once per repeated asset. Leave costly
        # W4 frames to the timing/identity check after the W2 refusal and retry have passed.
        if count == 2:
            held, neighbor = raw_ids
            cap = ARCHIVED_CANDIDATES[name][0]
            capacity = dict(rules)
            capacity[held] = ph.CollisionPolicy(candidate_capacity=max(0, cap - 1))
            old = work.last_result(held)
            selected = {held: 0, neighbor: 1 % len(frames)}
            mixed = {key: motion[index][key] for key, index in selected.items()}
            tried, cc = query(mixed, capacity)
            if cap:
                assert (
                    tried[held].status == "capacity_exceeded"
                    and tried[held].required_candidates == cap
                )
                assert not tried[held].pairs and work.last_result(held) is old
            else:
                assert tried[held].status == "ok" and tried[held].required_candidates == 0
            assert (
                tried[neighbor] == checked[selected[neighbor]][neighbor]
                and work.last_result(neighbor) is tried[neighbor]
            )
            accepted, rc = query(mixed, rules)
            for key, index in selected.items():
                assert (
                    accepted[key] == checked[index][key] and work.last_result(key) is accepted[key]
                )
            cap_record = dict(
                held=held,
                neighbor=neighbor,
                requested=max(0, cap - 1),
                required=cap,
                status=tried[held].status,
                preserves_prior=bool(cap),
                frame_by_world=selected,
                neighbor_equals_w1=True,
                retry_equals_w1=True,
                rejection_cost=cc,
                retry_cost=rc,
            )
        return dict(
            worlds=count,
            status="ok",
            world_keys=list(raw_ids),
            shapes=shape_count,
            vertices=vertices,
            logical_shape_pairs=count * len(model.shapes) * (len(model.shapes) - 1) // 2,
            frame_count=len(frames),
            facets_per_world=(
                sum(len(s.triangles) for s in model.shapes)
                if name.startswith("mesh")
                else sum(s.kind == "triangle" for s in model.shapes)
            ),
            copies_ns=copy_ns,
            construction_ns=construction_ns,
            cold=cold_cost,
            frames=summaries,
            warm_samples=samples,
            warm_cycles=cycle,
            median_warm_cycle_ns=median(x["wall_ns"] for x in cycle),
            last_stats=stats,
            capacity=cap_record,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("host", "cuda"), required=True)
    parser.add_argument("--cases", nargs="+", choices=NAMES, default=list(NAMES))
    parser.add_argument("--worlds", nargs="+", type=int, choices=(1, 2, 4), default=[1, 2, 4])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--legacy-cost", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.worlds[0] != 1 or len(set(args.worlds)) != len(args.worlds):
        parser.error("choose at least one warm sample and start distinct world counts with W1")
    report = dict(
        schema="phymeo-g4-world-geometry/1",
        backend=args.backend,
        samples=args.samples,
        source=ph.__file__,
        cache_root=str(args.cache_root),
        precision="original FP64 geometry and material witnesses; outward FP32 broad bounds",
        physical_substeps=0,
        archive=None,
        workloads={},
    )
    report["runtime"] = dict(
        python=sys.version.split()[0],
        phymeo=version("phymeo"),
        numpy=version("numpy"),
        tyx=version("tyx-compute") if args.backend == "cuda" else None,
    )
    if args.backend == "cuda":
        from benchmark_worlds import gpu_memory

        report["whole_device_before"] = gpu_memory()
        report["process_peak_device_bytes"] = None
    if args.legacy_cost:
        data = args.legacy_cost.read_bytes()
        earlier = json.loads(data)
        assert earlier["backend"] == args.backend
        report["archive"] = dict(
            source=str(args.legacy_cost), sha256=sha256(data).hexdigest(), workloads={}
        )
        for name in args.cases:
            archived = earlier["workloads"][name]["all_pairs/all_facets"]
            candidates = [row["candidates"] for row in archived["samples"] if row["repeat"] == 0]
            assert candidates == ARCHIVED_CANDIDATES[name]
            report["archive"]["workloads"][name] = dict(
                shapes=archived["shapes"],
                frames=archived["frames"],
                candidates=candidates,
                cold_ns=archived["cold_ns"],
                median_warm_cycle_ns=archived["median_warm_cycle_ns"],
            )
    for name in args.cases:
        model, frames = frozen(name)
        reference = {}
        records = report["workloads"][name] = []
        for count in args.worlds:
            rec = run(name, count, model, frames, args, reference)
            records.append(rec)
            save(args.out, report)
            print(
                f"{name} W{count} {rec['status']} warm cycle ns {rec.get('median_warm_cycle_ns')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
