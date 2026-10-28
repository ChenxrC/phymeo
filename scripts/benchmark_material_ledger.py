"""Serial native distance solve plus independent material ledger measurements."""

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
from benchmark_material_path import make_case

from phymeo import BackendContext, MaterialPathLedger, StepInputs
from phymeo.experiments.rigid_cloth import provenance


def operation(case, expected=None):
    factory, initial, settings = case
    inputs = StepInputs.zeros(3)
    context = BackendContext(factory.capabilities.backend)
    native_start = perf_counter()
    ws = factory.create_workspace("cloth", initial, inputs, 0.1, settings, context)
    try:
        ws.predict()
        for count in range(1, settings.position_iterations + 1):
            ws.position_sweep()
            if ws.position_metrics().max_residual <= settings.tolerance:
                break
        else:
            raise RuntimeError("native material did not converge")
        ws.prepare_velocity()
        state, diag = ws.result(), ws.diagnostics()
        native_seconds = perf_counter() - native_start
        before = (
            state.position.copy(),
            state.velocity.copy(),
            diag["distance_multipliers"].copy(),
            count,
        )
        start = perf_counter()
        book = MaterialPathLedger(
            factory.model, factory.program.constraints, initial, inputs, 0.1, settings, context
        )
        audit = book.require(state, inputs, np.zeros((3, 3)), diag, sweeps=count)
        audit_seconds = perf_counter() - start
        np.testing.assert_array_equal(state.position, before[0])
        np.testing.assert_array_equal(state.velocity, before[1])
        if expected is not None:
            for a, b in zip(before, expected, strict=True):
                np.testing.assert_array_equal(a, b)
        execution = diag.get("execution", {})
        if context.backend != "numpy" and execution.get("completed_launches", 0) <= 0:
            raise RuntimeError("missing actual execution")
        return before, dict(
            sweeps=count,
            native_seconds=native_seconds,
            audit_seconds=audit_seconds,
            max_residual=audit["max_residual"],
            energy_change_minus_work=audit["energy"]["change_minus_work"],
            **execution,
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
        parser.error("positive counts and a new output are required")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record = dict(
        mode=args.mode,
        source=provenance(),
        script_sha256=script_hash,
        status="running",
        scope="identical 3-node 2-edge snapshot; fresh native workspace solve, owned diagnostics, fresh ledger construction and audit, unchanged-state/repetition checks and close; factory excluded",
        batches=[],
        calls=[],
        accepted_measured_calls=0,
        successful_calls_including_warmups=0,
        not_measured=[
            "continuous trajectory",
            "contact outer",
            "Warp",
            "C++ endurance",
            "cold cache",
            "peak memory",
        ],
    )
    phase = "construction"
    try:
        case = make_case(args.mode, True)
        expected = None
        for warm in range(3):
            phase = f"warm{warm}"
            start = perf_counter()
            actual, stats = operation(case, expected)
            elapsed = perf_counter() - start
            if expected is None:
                expected = actual
            record["calls"].append(stats)
            record["successful_calls_including_warmups"] += 1
            if warm == 0:
                record["first_call_seconds"] = elapsed
        for repeat in range(args.repeats):
            phase = f"batch{repeat}"
            samples = []
            batch = dict(seconds=samples)
            record["batches"].append(batch)
            for _ in range(args.samples):
                start = perf_counter()
                _, stats = operation(case, expected)
                samples.append(perf_counter() - start)
                record["calls"].append(stats)
                record["accepted_measured_calls"] += 1
                record["successful_calls_including_warmups"] += 1
            batch.update(median=float(np.median(samples)), p95=float(np.percentile(samples, 95)))
        record["status"] = "complete"
    except Exception as error:
        record.update(
            status="failed", failure_phase=phase, error_type=type(error).__name__, failed_calls=1
        )
        traceback.print_exc()
    finally:
        if (
            provenance()["source_sha256"] != record["source"]["source_sha256"]
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_hash
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
