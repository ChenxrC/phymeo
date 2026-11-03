"""End-to-end contact-buffer comparison with a full detection projected to identical records."""

import argparse
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import phymeo as ph
from phymeo.geometry.compact_contacts import CompactWorldContactWorkspace, compact_contacts

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "scale"))
from compact_contact_cases import NAMES, check_spheres, sphere_case  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("host", "cuda"), required=True)
    p.add_argument("--method", choices=("full", "compact"), required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.warmup < 0 or min(a.samples, a.window) < 1:
        p.error("invalid sample or window count")
    if a.backend == "cuda":
        from tyx import runtime_stats

    def native():
        if a.backend != "cuda":
            return {}
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

    report = dict(
        schema="phymeo-compact-contact-cost/1",
        method=a.method,
        backend=a.backend,
        source=ph.__file__,
        runtime=dict(
            python=sys.version.split()[0],
            numpy=version("numpy"),
            tyx=version("tyx-compute") if a.backend == "cuda" else None,
        ),
        physical_substeps=0,
        warmup=a.warmup,
        samples=a.samples,
        window=a.window,
        cases={},
    )
    if a.backend == "cuda":
        from benchmark_worlds import gpu_memory

        report["device_before"] = gpu_memory()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    key = 37
    policy = {key: ph.CollisionPolicy()}
    for name in NAMES:
        model, paths = sphere_case(name, key)
        motion = {key: paths}
        start = perf_counter_ns()
        kind = CompactWorldContactWorkspace if a.method == "compact" else ph.WorldCollisionWorkspace
        with kind([(key, model)], a.backend, cache_root=a.cache_root) as work:
            setup_ns = perf_counter_ns() - start
            if a.method == "full":

                def run():
                    return compact_contacts(work.detect(motion, policy)[key])
            else:

                def run():
                    return work.detect(motion, policy)[key]

            def measure(n):
                pre = work.stats()
                low = native()
                tick = perf_counter_ns()
                for _ in range(n):
                    buf = run()
                elapsed = perf_counter_ns() - tick
                high = native()
                post = work.stats()
                names = (
                    "completed_launches",
                    "synchronizations",
                    "uploaded_bytes",
                    "downloaded_bytes",
                    "compact_selected_slots",
                    "compact_validation_slots",
                    "compact_skipped_slots",
                    "envelope_ns",
                    "broad_ns",
                    "narrow_ns",
                )
                return buf, dict(
                    wall_ns=elapsed,
                    native={k: high[k] - low[k] for k in high},
                    stats={k: post[k] - pre.get(k, 0) for k in names if k in post},
                )

            cold, first = measure(1)
            digest = check_spheres(model, paths, cold)
            for _ in range(a.warmup):
                for _ in range(a.window):
                    assert run() == cold
            records = []
            for _ in range(a.samples):
                now, sample = measure(a.window)
                assert now == cold
                records.append(sample)
            wall = sorted(s["wall_ns"] for s in records)
            report["cases"][name] = dict(
                shapes=len(model.shapes),
                pair_slots=len(model.shapes) * (len(model.shapes) - 1) // 2,
                setup_ns=setup_ns,
                cold=first,
                first_contact=None
                if not cold.contacts
                else [
                    cold.contacts[0].first.body,
                    cold.contacts[0].second.body,
                    cold.contacts[0].witness.gap_m,
                ],
                candidates=cold.required_candidates,
                contacts=cold.required_contacts,
                diagnostics=cold.required_diagnostics,
                output_sha256=digest,
                measurements=records,
                warm_p50_ns=median(wall),
                warm_p95_ns=wall[math.ceil(0.95 * len(wall)) - 1],
            )
            a.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
            print(f"{name} {a.method} {median(wall) / a.window / 1e6:.4f} ms/query", flush=True)


if __name__ == "__main__":
    main()
