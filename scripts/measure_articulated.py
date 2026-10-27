"""Serial P5 step measurements from a validated, fixed active-contact frame."""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from phymeo.experiments.articulated import (
    _state_at,
    articulated_scene,
    load_articulated_run,
    validate_articulated_run,
)
from phymeo.experiments.rigid_cloth import SceneRunFailure, provenance
from phymeo.solvers.articulated_coupling import make_articulated_runtime
from phymeo.validation.articulated import audit_slider_substep, compact_event


def stats(values):
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}


def _check_result(fixture, before, result, run, frame):
    if not result.report.accepted:
        raise SceneRunFailure(result.report.reason)
    if len(result.substep_states) != 2 or len(result.report.substeps) != 2:
        raise SceneRunFailure("measurement requires both actual substep states and reports")
    records = []
    for offset, (after, report) in enumerate(
        zip(result.substep_states, result.report.substeps, strict=True)
    ):
        index = frame * 2 + offset
        events = [
            compact_event(event, fixture.node_names.index(event["domain"]), index)
            for event in report.diagnostics["contact_events"]
        ]
        robot = report.diagnostics["participants"]["robot"]
        quality, contacts = audit_slider_substep(
            fixture,
            before,
            after,
            events,
            robot["predicted_q"],
            robot["predicted_v"],
            h=fixture.config["h"],
        )
        for name, native in after.domains.items():
            for field, value in vars(native).items():
                if not np.array_equal(value, run.arrays[f"substep_{name}_{field}"][index]):
                    raise SceneRunFailure(f"measured {name}.{field} differs from saved substep")
        records.append(
            {
                "position_iterations": report.position_iterations,
                "velocity_iterations": report.velocity_iterations,
                "max_residual": report.max_residual,
                "quality": quality,
                "contacts": contacts,
                "event_count": len(events),
                "timings_ms": {key: seconds * 1000 for key, seconds in report.timings.items()},
            }
        )
        before = after
    for name, native in result.state.domains.items():
        for field, value in vars(native).items():
            if not np.array_equal(value, run.arrays[f"{name}_{field}"][frame + 1]):
                raise SceneRunFailure(f"committed {name}.{field} differs from saved frame")
    return records


def measure(run, *, frame=100, warmups=2, repeats=3, calls=100, record=None, progress=None):
    for name, value, minimum in (
        ("frame", frame, 0),
        ("warmups", warmups, 2),
        ("repeats", repeats, 1),
        ("calls", calls, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    validate_articulated_run(run)
    if frame >= run.config["frames"]:
        raise ValueError("frame must have a successor in the saved trace")
    record = {} if record is None else record
    record.update(
        status="running",
        scene=run.config["variant"],
        frame=frame,
        config=run.config,
        source_trajectory_sha256=run.metrics["trajectory_sha256"],
        acceptance_protocol=(
            run.config["frames"] == 200 and frame == 100 and repeats >= 3 and calls >= 100
        ),
        warmup_calls=0,
        accepted_measured_steps=0,
        failed_calls=0,
        samples=[],
        warmup_samples=[],
        repeats=[],
    )
    start = perf_counter()
    try:
        fixture = articulated_scene(run.config["variant"])
        state = fixture.initial if frame == 0 else _state_at(fixture, run.arrays, frame * 2 - 1)
        runtime = make_articulated_runtime(fixture.program, settings=fixture.settings)
    except Exception as error:
        record.update(status="failed", failure={"phase": "construction", "error": str(error)})
        raise
    record["construction_ms"] = (perf_counter() - start) * 1000
    record["workload"] = {
        "robot_dof": len(state.domains["robot"].q),
        "node_counts": {name: len(state.domains[name].position) for name in fixture.node_names},
        "declared_pairs": sum(len(contact.node_indices) for contact in fixture.program.contacts),
        "cloth_edges": len(fixture.topology.get("cloth_edges", [])),
        "tetrahedra": len(fixture.topology.get("soft_tetrahedra", [])),
    }

    def call():
        begin = perf_counter()
        value = runtime.step(state, fixture.inputs, fixture.config["dt"])
        duration = (perf_counter() - begin) * 1000
        begin = perf_counter()
        substeps = _check_result(fixture, state, value, run, frame)
        for name, native in state.domains.items():
            for field, array in vars(native).items():
                if not np.array_equal(array, run.arrays[f"{name}_{field}"][frame]):
                    raise SceneRunFailure("benchmark initial state was mutated")
        return {
            "step_ms": duration,
            "audit_ms": (perf_counter() - begin) * 1000,
            "substeps": substeps,
        }

    phase, index = "first", 0
    try:
        record["first_sample"] = call()
        phase = "warmup"
        for index in range(warmups):
            record["warmup_samples"].append(call())
            record["warmup_calls"] += 1
        phase = "measured"
        for repeat in range(repeats):
            samples = []
            for index in range(calls):
                sample = call()
                sample.update(repeat=repeat + 1, call=index + 1)
                record["samples"].append(sample)
                record["accepted_measured_steps"] += 1
                samples.append(sample["step_ms"])
            record["repeats"].append({"repeat": repeat + 1, "calls": calls, **stats(samples)})
            if progress is not None:
                progress(repeat + 1, record["repeats"][-1])
        phases = {}
        for sample in record["samples"]:
            for step in sample["substeps"]:
                for key, value in step["timings_ms"].items():
                    phases.setdefault(key, []).append(value)
        record["phase_timings_per_substep"] = {key: stats(values) for key, values in phases.items()}
        record.update(status="complete", exact_saved_state=True)
    except Exception as error:
        record.update(
            status="failed",
            failed_calls=record["failed_calls"] + 1,
            failure={"phase": phase, "call": index + 1, "error": str(error)},
        )
        raise
    finally:
        try:
            runtime.close()
        except Exception as error:
            record.update(status="failed", close_error=str(error))
            record.setdefault("failure", {"phase": "close", "error": str(error)})
            raise
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    source = provenance()
    report = {
        "status": "running",
        "provenance": source,
        "measurement_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "protocol": {
            "serial": True,
            "frame": 100,
            "warmups": 2,
            "repeats": 3,
            "calls_per_repeat": 100,
            "timing": "full runtime.step including production checks/diagnostics, independent audit after timer",
            "state": "restore validated frame100 for every call, compare both substeps and frame101 exactly",
        },
        "limitations": [
            "fixed active-contact state microbenchmark, not continuous trajectory throughput",
            "first call after imports and saved-trace validation, not independent-process cold start",
            "no peak memory, allocation counts, per-function profiling or controlled OS/BLAS scheduling",
            "no tyx/Warp comparison or performance optimization",
        ],
        "trace_sources": {},
        "workloads": [],
    }
    try:
        for scene in ("S2", "S3", "S4"):
            directory = args.input / scene
            stored = json.loads((directory / "run.json").read_text(encoding="utf-8"))
            if any(
                stored["provenance"].get(key) != source.get(key)
                for key in ("source_sha256", "python", "numpy", "platform")
            ):
                raise ValueError("measurement and trace package/numerical environments must match")
            report["trace_sources"][scene] = stored["provenance"]
            start = perf_counter()
            run = load_articulated_run(directory, require_complete=True)
            validation_ms = (perf_counter() - start) * 1000
            workload = {"trace_validation_ms": validation_ms}
            report["workloads"].append(workload)
            measure(
                run,
                record=workload,
                progress=lambda repeat, values, scene=scene: print(
                    json.dumps({"scene": scene, **values}), flush=True
                ),
            )
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
