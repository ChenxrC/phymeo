"""Same observed COM/gap/impulse frame for independent, finite rigid spheres on horizontal planes."""

import argparse
import json
import sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "scale"))
from loaded_sphere_engines import Newton, Phymeo  # noqa: E402
from sphere_plane_cases import MASS, PLANE, RADIUS, WINDOW, H, application, cases, metrics  # noqa: E402


def observe(a, case):
    source = cases(case)
    worlds = tuple(sorted(source, key=lambda w: w.key))
    timed = a.phase == "measure"
    setup = perf_counter_ns()
    engine = (Phymeo if a.engine == "phymeo" else Newton)(
        source, a.device, a.cache_root, force_audit=not timed
    )
    setup = perf_counter_ns() - setup
    try:
        order = [engine.keys.index(w.key) for w in worlds]
        initial = application(None, engine.read()[order], worlds)
        expected = np.array([[0, 0, w.z, w.vx, 0, w.vz] for w in worlds], np.float32)
        np.testing.assert_array_equal(initial[:, :6], expected)
        trajectory = [initial[None]]
        previous = initial
        native = []
        tick = 0
        records = []
        vx = np.array([w.vx for w in worlds])

        def window(n):
            nonlocal tick, previous
            before = engine.counts()
            start = perf_counter_ns()
            frames = np.empty((n, len(worlds), 8), np.float32)
            for i in range(n):
                tick += 1
                frames[i] = application(previous, engine.advance()[order], worlds)
                previous = frames[i]
                if timed:
                    if (
                        np.max(abs(previous[:, 2] - RADIUS - PLANE)) > 2e-5
                        or np.max(abs(previous[:, 5])) > 2e-4
                        or np.max(abs(previous[:, 7] / H - 16)) > 0.16
                        or np.max(abs(previous[:, 0] - tick * H * vx)) > 2e-6
                        or np.max(abs(previous[:, 3] - vx)) > 2e-5
                    ):
                        raise AssertionError(f"loaded primary gate failed at {tick}")
                else:
                    native.append(engine.details())
            wall = perf_counter_ns() - start
            after = engine.counts()
            trajectory.append(frames)
            return dict(
                tick=tick,
                wall_ns=wall,
                steps=n,
                state_sha256=sha256(frames.tobytes()).hexdigest(),
                counts={k: v - before[k] for k, v in after.items()},
            )

        if timed:
            cold = window(1)
            warmup = [window(WINDOW) for _ in range(3)]
            records = [window(WINDOW) for _ in range(10)]
        else:
            cold = None
            warmup = []
            records = [window(16 if case == "isolated" else 32)]
        raw = np.concatenate(trajectory)
        name = case + ".npy"
        with a.out.with_name(name).open("xb") as f:
            np.save(f, raw, allow_pickle=False)
        physical = metrics(raw, worlds)
        obs = dict(
            scene=case,
            engine=engine.meta,
            setup_ns=setup,
            rows=[asdict(w) for w in worlds],
            dt_s=H,
            mass_kg=MASS,
            radius_m=RADIUS,
            plane_height_m=PLANE,
            trajectory=name,
            trajectory_shape=raw.shape,
            cold=cold,
            warmup=warmup,
            samples=records,
            final_tick=tick,
            physics=physical,
            all_functional_gates=all(m["passed"] for m in physical),
        )
        if timed:
            for sample in records:
                c = sample["counts"]
                assert c["solver_calls"] == c["explicit_sync"] == WINDOW
                if a.engine == "phymeo":
                    assert (
                        c["accepted_world_steps"] == len(worlds) * WINDOW
                        and c["rejected_world_steps"] == 0
                    )
                    assert c["native_cuda_kernels"] == c["device_submissions"] == 2 * WINDOW
                    assert c["native_h2d_b"] == 0 and c["native_d2h_b"] == len(worlds) * WINDOW * 56
                else:
                    assert c["collision_calls"] == WINDOW and c["public_state_reads"] == 2 * WINDOW
                    assert c["state_read_b"] == len(worlds) * WINDOW * 52
            # One extra observable step after the published trajectory records the real
            # native GPU activity without enabling tracing in the wall-clock samples.
            if a.engine == "newton":
                engine.wp.timing_begin()
                try:
                    extra = application(previous, engine.advance()[order], worlds)
                finally:
                    events = engine.wp.timing_end()
                assert np.all(extra[:, 7] == MASS * 8 * H)
                kernels = [str(e.name) for e in events if e.filter == engine.wp.TIMING_KERNEL]
                assert any("solve_body_contact_positions" in name for name in kernels)
                obs["extra_native_step"] = dict(
                    kernel_names=kernels,
                    activities=[(str(e.name), float(e.elapsed)) for e in events],
                )
            walls = [s["wall_ns"] for s in records]
            obs.update(
                p50_ms=median(walls) / 1e6,
                p95_ms=max(walls) / 1e6,
                verified_world_steps=len(worlds) * WINDOW * 10,
                world_steps_s=len(worlds) * WINDOW * 10e9 / sum(walls),
            )
        else:
            obs["native_per_step"] = native
        return obs
    finally:
        engine.close()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--engine", choices=("phymeo", "newton"), required=True)
    cli.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    cli.add_argument("--phase", choices=("qualify", "measure"), required=True)
    cli.add_argument("--eligibility", type=Path)
    cli.add_argument("--cache-root", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    a = cli.parse_args()
    if a.out.exists():
        cli.error("use a fresh receipt path")
    if a.phase == "measure" and (
        a.device != "cuda:0"
        or a.eligibility is None
        or not json.loads(a.eligibility.read_text())["w32_same_output_eligible"]
    ):
        cli.error(
            "formal paired CUDA timings require the predeclared W32 exact-trajectory eligibility"
        )
    a.cache_root.mkdir(exist_ok=True, parents=True)
    a.out.parent.mkdir(exist_ok=True, parents=True)
    prior = any(f.is_file() for f in a.cache_root.rglob("*"))
    result = dict(
        engine=a.engine,
        device=a.device,
        phase=a.phase,
        cache_prepopulated=prior,
        program=sys.executable,
        scope="Sorted X/Y/Z, COM velocity, physical gap and impulse inferred from actual state",
        runs={},
    )
    for name in ("support",) if a.phase == "measure" else ("mixed", "isolated", "support"):
        row = observe(a, name)
        result["runs"][name] = row
        a.out.write_text(json.dumps(result, allow_nan=False, indent=2), encoding="utf8")
        print(name, row["all_functional_gates"], row.get("p50_ms"), row["final_tick"], flush=True)


if __name__ == "__main__":
    main()
