"""Cold and paired warm CUDA runs for the same loaded cloth states at W16/W32."""

import argparse
import gzip
import json
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np
from benchmark_multiface_phase_batch import (
    STEPS,
    H,
    certify,
    make_world,
    native,
    summary,
    whole_window,
)

import phymeo as ph


def make_scene(i, density, operator):
    count = 8 if density == "dense" else (2, 8, 2, 8)[i % 4]
    w = make_world(i, count, operator)
    if density == "mixed" and i % 4 == 3:
        c = w.initial.domains["cloth"]
        state = ph.SceneState(
            {**w.initial.domains, "cloth": ph.ParticleState(c.position - [0, 0, 0.5], c.velocity)}
        )
        return replace(w, initial=state), False
    return w, True


def cache_contents(path):
    files = [f for f in path.rglob("*") if f.is_file()]
    return dict(files=len(files), bytes=sum(f.stat().st_size for f in files))


class Lane:
    def __init__(self, kind, count, density, cache, resources, tx):
        self.kind, self.tx = kind, tx
        self.settings = ph.StepSettings(
            substeps=1, position_iterations=1, velocity_iterations=1, record_substeps=True
        )
        self.layer = (
            resources.enter_context(ph.TyxNormalLayer("cuda", cache_root=cache))
            if kind == "ordinary"
            else None
        )
        models = [make_scene(i, density, self.layer) for i in range(count)]
        self.worlds, self.loaded = zip(*models, strict=True)
        self.keys = tuple(37 + 3 * i for i in reversed(range(count)))
        self.inputs = [
            {k: w.controls[t] for k, w in zip(self.keys, self.worlds, strict=True)}
            for t in range(STEPS)
        ]
        if kind == "ordinary":
            self.steppers = [
                resources.enter_context(w.program.stepper(self.settings)) for w in self.worlds
            ]
            self.op = self.layer
        else:
            self.op = resources.enter_context(
                ph.TyxMultifaceContactBatch(
                    [
                        (k, ph.ClothLinkWorld(w.program, w.initial))
                        for k, w in zip(self.keys, self.worlds, strict=True)
                    ],
                    "cuda",
                    settings=self.settings,
                    cache_root=cache,
                )
            )

    def counters(self):
        rt = self.tx.runtime_stats()
        counts = {k: v for k, v in self.op.stats().items() if k != "backend"}
        for key in ("cuda_kernel_launches", "cuda_h2d_bytes", "cuda_d2h_bytes", "cuda_d2d_bytes"):
            counts[key] = int(getattr(rt, key))
        return counts

    def run(self, *, phase_audit=False):
        if self.kind == "grouped":
            self.op.reset()
        before = self.counters()
        start = perf_counter_ns()
        phases = []
        if self.kind == "grouped":
            frames = {k: [] for k in self.keys}
            for controls in self.inputs:
                for k, out in self.op.step(H, controls).items():
                    frames[k].append(out)
                if phase_audit:
                    phases.extend(self.op.last_phases)
            results = list(frames.values())
        else:
            results = whole_window(self.worlds, self.steppers)
        stop = perf_counter_ns()
        after = self.counters()
        diff = {k: v - before[k] for k, v in after.items()}
        assert all(f.report.accepted for group in results for f in group)
        assert (
            diff["completed_launches"]
            == diff["uploads"]
            == diff["downloads"]
            == diff["cuda_kernel_launches"]
        )
        assert diff["sites"] > 0 and diff["cuda_d2d_bytes"] == 0
        assert diff["uploaded_bytes"] == diff["uploads"] * 6656
        assert diff["downloaded_bytes"] == diff["downloads"] * 2560
        raw_clocks = [
            [[dict(sub.timings) for sub in frame.report.substeps] for frame in one]
            for one in results
        ]
        return (
            results,
            dict(
                wall_ns=stop - start, counts=diff, hashes=summary(results), raw_clocks_s=raw_clocks
            ),
            phases,
        )

    def validate(self, results):
        audit = []
        for k, w, loaded, frames in zip(self.keys, self.worlds, self.loaded, results, strict=True):
            if loaded:
                audit.append(
                    dict(key=k, faces=len(w.faces), loaded=True, material=certify(w, frames))
                )
            else:
                d = [v.report.substeps[0].diagnostics["robot_contact"] for v in frames]
                assert all(
                    not a["cloth_links"] and not np.any(a["cloth_link_efforts"]["link"]) for a in d
                )
                assert not np.array_equal(
                    w.initial.domains["cloth"].position, frames[-1].state.domains["cloth"].position
                )
                audit.append(
                    dict(
                        key=k,
                        faces=len(w.faces),
                        loaded=False,
                        final_cloth_z_m=frames[-1].state.domains["cloth"].position[:, 2].tolist(),
                    )
                )
        return audit


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--worlds", type=int, choices=(16, 32), required=True)
    cli.add_argument("--density", choices=("dense", "mixed"), required=True)
    cli.add_argument("--first", choices=("ordinary", "grouped"), required=True)
    cli.add_argument("--pairs", type=int, default=5)
    cli.add_argument("--cache-root", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    a = cli.parse_args()
    if a.out.exists() or a.pairs < 2 or cache_contents(a.cache_root)["files"]:
        cli.error("use fresh output/cache locations and at least two pairs")
    import tyx as tx

    order = (a.first, "ordinary" if a.first == "grouped" else "grouped")
    lanes, cold, costs, physical = {}, {}, {}, {}
    with ExitStack() as scope:
        for label in order:
            cache = a.cache_root / label
            cache.mkdir(parents=True, exist_ok=False)
            start = perf_counter_ns()
            lanes[label] = lane = Lane(label, a.worlds, a.density, cache, scope, tx)
            setup = perf_counter_ns() - start
            frames, cold[label], _ = lane.run()
            costs[label] = dict(
                setup_ns=setup,
                setup_plus_first_window_ns=setup + cold[label]["wall_ns"],
                cache_after_first=cache_contents(cache),
            )
            # Preserve both complete cold frames separately. Checks/encoding stay outside timed windows.
            detail = a.out.with_name(label + "-frames.json.gz")
            with gzip.open(detail, "xt", encoding="utf8", compresslevel=1) as file:
                json.dump(native(frames), file, allow_nan=False, separators=(",", ":"))
            physical[label] = detail.name
        assert cold["ordinary"]["hashes"] == cold["grouped"]["hashes"]
        expected = cold["ordinary"]["hashes"]
        frames, warm, phases = lanes["grouped"].run(phase_audit=True)
        assert warm["hashes"] == expected and len(phases) == 12
        proof = lanes["grouped"].validate(frames)
        layout = [
            dict(
                substep=index // 3,
                phase=phase.phase,
                sites=phase.sites,
                world_offsets=phase.world_offsets,
            )
            for index, phase in enumerate(phases)
        ]
        required = {
            "ordinary": sum(
                (hi - lo + 127) // 128 for p in phases for _, lo, hi in p.world_offsets
            ),
            "grouped": sum((p.sites + 127) // 128 for p in phases),
        }
        assert (
            sum(row["sites"] for row in layout)
            == cold["ordinary"]["counts"]["sites"]
            == cold["grouped"]["counts"]["sites"]
        )
        _, warm_o, _ = lanes["ordinary"].run()
        assert warm_o["hashes"] == expected
        pairs = []
        for i in range(a.pairs):
            seq = order if i % 2 == 0 else order[::-1]
            pair = {"order": seq}
            for label in seq:
                _, s, _ = lanes[label].run()
                assert (
                    s["hashes"] == expected and s["counts"]["completed_launches"] == required[label]
                )
                assert s["counts"]["uploaded_bytes"] == s["counts"]["cuda_h2d_bytes"]
                assert s["counts"]["downloaded_bytes"] == s["counts"]["cuda_d2h_bytes"]
                pair[label] = s
            pairs.append(pair)
    p50 = {label: median(row[label]["wall_ns"] for row in pairs) / 1e6 for label in order}
    speed = {
        label: a.worlds * STEPS * len(pairs) * 1e9 / sum(row[label]["wall_ns"] for row in pairs)
        for label in order
    }
    receipt = dict(
        comparison="Phymeo internal ordinary vs grouped same geometry CUDA",
        worlds=a.worlds,
        density=a.density,
        steps=STEPS,
        dt_s=H,
        first=a.first,
        setup=costs,
        cold=cold,
        qualification_layout=layout,
        full_cold_frames=physical,
        per_world_proof=proof,
        pairs=pairs,
        p50_ms=p50,
        verified_world_steps_s=speed,
        accepted_per_window=a.worlds * STEPS,
        registered_faces=sum(w["faces"] for w in proof),
        loaded_faces_each_midpoint=sum(w["faces"] for w in proof if w["loaded"]),
        pages_per_window=required,
    )
    with a.out.open("x", encoding="utf8") as file:
        json.dump(receipt, file, allow_nan=False)
    print(
        json.dumps(
            {
                "p50_ms": p50,
                "registered_faces": receipt["registered_faces"],
                "loaded_faces": receipt["loaded_faces_each_midpoint"],
                "pages": required,
                "sites": cold["grouped"]["counts"]["sites"],
            }
        )
    )


if __name__ == "__main__":
    main()
