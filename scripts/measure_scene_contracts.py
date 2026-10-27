"""Measure P1 contract ownership/validation without claiming physical-step throughput."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from phymeo import ParticleModel, ParticleState, StepInputs
from phymeo.core.scene import (
    BackendContext,
    DomainBinding,
    HistoryField,
    SceneInputs,
    SceneModel,
    SceneState,
)
from phymeo.experiments.rigid_cloth import provenance
from phymeo.runtime.registry import StateRegistry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    model = ParticleModel(np.ones(49), np.array([0.0, -9.81, 0.0]))
    scene = SceneModel((DomainBinding("cloth", "particles", model),))
    state = SceneState(
        {"cloth": ParticleState(np.zeros((49, 3)), np.zeros((49, 3)))},
        {"warm": np.zeros(49)},
    )
    inputs = SceneInputs({"cloth": StepInputs.zeros(49)})
    registry, context = StateRegistry(), BackendContext()
    schema = (HistoryField("warm", (49,), np.float64),)
    for _ in range(2):
        for _ in range(100):
            registry.snapshot(scene, state, inputs, context, schema)
    rows = []
    for repetition in range(3):
        samples = []
        for _ in range(100):
            start = time.perf_counter_ns()
            owned, _ = registry.snapshot(scene, state, inputs, context, schema)
            samples.append((time.perf_counter_ns() - start) / 1e6)
        np.testing.assert_array_equal(owned.history["warm"], state.history["warm"])
        assert not np.shares_memory(
            owned.domains["cloth"].position, state.domains["cloth"].position
        )
        rows.append(
            {
                "repeat": repetition + 1,
                "samples": 100,
                "median_ms": float(np.median(samples)),
                "p95_ms": float(np.percentile(samples, 95)),
            }
        )
    report = {
        "stage": "P1-T1",
        "scope": "contract snapshot only; no physical step or GPU work",
        "provenance": provenance(),
        "workload": {
            "particles": 49,
            "history_values": 49,
            "dtype": "float64",
            "warmup_batches": 2,
            "samples_per_batch": 100,
            "repeats": 3,
        },
        "results": rows,
        "limitations": [
            "timer overhead not subtracted",
            "peak memory and phase attribution unmeasured",
            "recorded source hash identifies measured code; no speedup claim",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
