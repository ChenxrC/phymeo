"""Serial material trace off/on observations on identical nonlinear snapshots."""

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np

from phymeo import (
    BackendContext,
    ClothParticipant,
    DistanceConstraints,
    ParticleModel,
    ParticleState,
    StepInputs,
    StepSettings,
    TyxClothParticipant,
)
from phymeo.experiments.rigid_cloth import provenance


def make_case(mode, trace):
    model = ParticleModel(np.ones(3), np.zeros(3))
    state = ParticleState(
        np.array([[0.0, 0, 0], [1.2, 0.2, 0], [1.5, 1.4, 0]]),
        np.array([[0.2, 0.8, 0], [-0.3, 0.1, 0], [0.1, -0.5, 0]]),
    )
    constraints = DistanceConstraints(
        np.array([[0, 1], [1, 2]]), np.ones(2), np.array([0.001, 0.002])
    )
    factory = (
        ClothParticipant(model, constraints, record_material_path=trace)
        if mode == "numpy"
        else TyxClothParticipant(model, constraints, mode, record_material_path=trace)
    )
    settings = StepSettings(substeps=1, tolerance=1e-8 if mode == "numpy" else 2e-5)
    return factory, state, settings


def operation(case, trace, expected=None):
    factory, state, settings = case
    ws = factory.create_workspace(
        "cloth",
        state,
        StepInputs.zeros(3),
        0.1,
        settings,
        BackendContext(factory.capabilities.backend),
    )
    try:
        ws.predict()
        for count in range(1, 51):
            ws.position_sweep()
            if ws.position_metrics().max_residual <= settings.tolerance:
                break
        else:
            raise RuntimeError("material iteration budget exhausted")
        ws.prepare_velocity()
        final, diag = ws.result(), ws.diagnostics()
        result = (
            final.position.copy(),
            final.velocity.copy(),
            diag["distance_multipliers"].copy(),
            count,
        )
        if expected is not None:
            for a, b in zip(result, expected, strict=True):
                np.testing.assert_array_equal(a, b)
        if trace:
            path = diag["material_path"]
            if path["schema"] != "xpbd-distance-path-v1" or len(path["frames"]) != count:
                raise RuntimeError("missing actual material frames")
            if any(
                f.shape != (2, 18) or not np.isfinite(f).all() or f.flags.writeable
                for f in path["frames"]
            ):
                raise RuntimeError("invalid material frames")
        elif "material_path" in diag:
            raise RuntimeError("default path unexpectedly traced")
        execution = diag.get("execution", {})
        if factory.capabilities.backend != "numpy" and execution.get("completed_launches", 0) <= 0:
            raise RuntimeError("missing actual backend execution")
        return result, dict(
            sweeps=count, trace_bytes=count * 2 * 18 * 8 if trace else 0, **execution
        )
    finally:
        ws.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["numpy", "interpret", "cuda"], required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1 or args.output.exists():
        parser.error("positive samples/repeats and new output required")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record = dict(
        mode=args.mode,
        source=provenance(),
        script_sha256=script_hash,
        status="running",
        scope="same three-node/two-edge snapshot; native workspace, solve, diagnostics, bitwise parity checks and close timed; factory excluded; off then on, not randomized",
        not_measured=[
            "continuous time trajectory",
            "coupled robot contact",
            "Warp",
            "cold process/cache",
            "peak memory",
            "kernel attribution",
            "C++ endurance",
        ],
        configurations=[],
    )
    phase = "setup"
    try:
        expected = None
        for trace in (False, True):
            case = make_case(args.mode, trace)
            config = dict(
                trace=trace,
                batches=[],
                calls=[],
                accepted_measured_calls=0,
                successful_calls_including_warmups=0,
            )
            record["configurations"].append(config)
            for warm in range(3):
                phase = f"trace={trace}/warm={warm}"
                start = perf_counter()
                result, stats = operation(case, trace, expected)
                elapsed = perf_counter() - start
                if expected is None:
                    expected = result
                config["calls"].append(stats)
                config["successful_calls_including_warmups"] += 1
                if warm == 0:
                    config["first_call_seconds"] = elapsed
            for repeat in range(args.repeats):
                phase = f"trace={trace}/batch={repeat}"
                samples = []
                batch = dict(seconds=samples)
                config["batches"].append(batch)
                for _ in range(args.samples):
                    start = perf_counter()
                    _, stats = operation(case, trace, expected)
                    samples.append(perf_counter() - start)
                    config["calls"].append(stats)
                    config["accepted_measured_calls"] += 1
                    config["successful_calls_including_warmups"] += 1
                batch.update(
                    median=float(np.median(samples)), p95=float(np.percentile(samples, 95))
                )
        record["status"] = "complete"
    except Exception as error:
        record.update(
            status="failed",
            failure_phase=phase,
            failed_calls=1,
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
    finally:
        if (
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_hash
            or provenance()["source_sha256"] != record["source"]["source_sha256"]
        ):
            record["status"] = "source_changed"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(record["status"], args.output)
    return 0 if record["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
