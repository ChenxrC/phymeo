"""Serial fixed-configuration thickness-map observations, not simulation throughput."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from benchmark_surface_foundation import observe

from phymeo import (
    BackendContext,
    ParticleModel,
    ParticlePointPort,
    ParticleState,
    StepInputs,
    StepSettings,
    TriangleOffsetPointPort,
    TyxClothParticipant,
)
from phymeo.experiments.rigid_cloth import provenance


def measure(mode, count, samples, repeats):
    model = ParticleModel(np.array([1.0, 2, 3]), np.zeros(3))
    state = ParticleState(np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]]), np.zeros((3, 3)))
    ws = None
    try:
        if mode == "numpy":
            base = ParticlePointPort(model, state, np.arange(3))
        else:
            ws = TyxClothParticipant(model, None, mode).create_workspace(
                "cloth",
                state,
                StepInputs.zeros(3),
                0.01,
                StepSettings(tolerance=2e-6),
                BackendContext("tyx:" + mode, np.float64),
            )
            ws.predict()
            base = ws.point_port(np.arange(3))
        triangles = np.tile([0, 1, 2], (count, 1))
        weights = np.tile([0.2, 0.3, 0.5], (count, 1))
        offsets = np.full(count, 0.2)
        impulses = np.tile([1.0 / count, 0, 0], (count, 1))

        def construct():
            result = TriangleOffsetPointPort(base, triangles, weights, offsets)
            np.testing.assert_allclose(
                result.positions(), np.tile([0.3, 0.5, 0.2], (count, 1)), atol=1e-14
            )

        port = TriangleOffsetPointPort(base, triangles, weights, offsets)

        def response():
            np.testing.assert_allclose(
                port.response(impulses),
                np.tile([1.09, 0.04, -0.08], (count, 1)),
                atol=2e-5,
                rtol=2e-5,
            )

        def candidate():
            result = port.apply_impulses(impulses)
            np.testing.assert_allclose(
                result.velocity, [[0.2, 0, 0.2], [0.6, 0, -0.4], [1.5, 0, 0]], atol=2e-5
            )
            np.testing.assert_allclose(port.candidate_audit(result).energy_change, 0.545, atol=2e-5)

        before = dict(ws.diagnostics()["execution"]) if ws else None
        records = {
            name: observe(operation, samples, repeats)
            for name, operation in (
                ("construct_host_map", construct),
                ("response_and_assert", response),
                ("candidate_audit_and_assert", candidate),
            )
        }
        after = dict(ws.diagnostics()["execution"]) if ws else None
        if ws and after["completed_launches"] <= before["completed_launches"]:
            raise RuntimeError("no actual backend execution")
        return {
            "attachments": count,
            "base_nodes": 3,
            "records": records,
            "execution_before": before,
            "execution_after": after,
        }
    finally:
        if ws:
            ws.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["numpy", "interpret", "cpu_cpp", "cuda"], required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1:
        parser.error("samples/repeats must be positive")
    root = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__),
        root / "scripts/benchmark_surface_foundation.py",
        root / "src/phymeo/coupling/offset_port.py",
    ]
    hashes = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    data = {
        "scope": "fixed snapshot; host FP64 geometry/maps, actual base backend response; assertions included",
        "mode": args.mode,
        "source": provenance(),
        "source_hashes": hashes,
        "not_measured": [
            "full step",
            "dynamic contact",
            "Warp",
            "cold process or cold cache",
            "memory",
            "profiler attribution",
        ],
        "cases": [measure(args.mode, count, args.samples, args.repeats) for count in [1, 16]],
    }
    for name, expected in hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("source changed during measurement")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
