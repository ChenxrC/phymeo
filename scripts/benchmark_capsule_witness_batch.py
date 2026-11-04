"""Pair the frozen and current material-witness implementations in the same CUDA worlds."""

import argparse
import importlib.util
import json
import sys
from contextlib import ExitStack
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from unittest.mock import patch

import numpy as np
from benchmark_multiface_phase_batch import STEPS, H, certify, make_world, summary
from multiface_load_cases import native

import phymeo as ph
from phymeo.geometry import _capsule_triangle_sweep, _rotating_capsule_face, _static_capsule_face


def archived(path):
    name = "phymeo.geometry._frozen_capsule_witness_benchmark"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.segment_witness


def population(layer, holdout):
    if not holdout:
        return [make_world(i, 8, layer) for i in range(4)]
    # Reuse local face/node IDs across worlds. Tilt two connected strips by
    # different affine maps and put a third sheet wholly outside the force layer.
    worlds = []
    for faces, ax, ay, z in (
        (2, 1 / 64, -1 / 32, 0),
        (8, -1 / 32, 1 / 16, 0),
        (8, 1 / 64, 1 / 32, -0.375),
    ):
        w = make_world(0, faces, layer)
        original = w.initial.domains["cloth"]
        x = original.position.copy()
        x[:, 2] += ax * x[:, 0] + ay * x[:, 1] + z
        state = ph.SceneState(
            {**w.initial.domains, "cloth": ph.ParticleState(x, original.velocity)}
        )
        worlds.append(replace(w, initial=state))
    return worlds


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--baseline-module", type=Path, required=True)
    cli.add_argument("--profile-reference", type=Path, required=True)
    cli.add_argument("--first", choices=("baseline", "candidate"), required=True)
    cli.add_argument("--pairs", type=int, default=5)
    cli.add_argument("--cache-root", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    a = cli.parse_args()
    if a.out.exists() or a.pairs < 2:
        cli.error("fresh receipt and at least two pairs required")
    old = archived(a.baseline_module)
    current = _capsule_triangle_sweep.segment_witness
    source = {
        label: sha256(path.read_bytes()).hexdigest()
        for label, path in (
            ("baseline", a.baseline_module),
            ("candidate", Path(_capsule_triangle_sweep.__file__)),
        )
    }
    assert source["baseline"] != source["candidate"]
    previous = json.loads(a.profile_reference.read_text())["reference_hashes"]
    settings = ph.StepSettings(
        substeps=1, position_iterations=1, velocity_iterations=1, record_substeps=True
    )
    result = {"source_sha256": source}
    with ExitStack() as resources:
        for holdout in (False, True):
            worlds = population(None, holdout)
            keys = (19, 4, 7) if holdout else (0, 1, 2, 3)
            batch = resources.enter_context(
                ph.TyxMultifaceContactBatch(
                    [
                        (k, ph.ClothLinkWorld(w.program, w.initial))
                        for k, w in zip(keys, worlds, strict=True)
                    ],
                    "cuda",
                    settings=settings,
                    cache_root=a.cache_root,
                )
            )
            controls = [
                {k: w.controls[t] for k, w in zip(keys, worlds, strict=True)} for t in range(STEPS)
            ]

            def run(label):
                batch.reset()
                before = batch.stats()
                # Only the changed function is swapped. Both selections take the same
                # uninstrumented public step and the same live device/runtime buffers.
                with ExitStack() as bindings:
                    for module in (
                        _capsule_triangle_sweep,
                        _rotating_capsule_face,
                        _static_capsule_face,
                    ):
                        bindings.enter_context(
                            patch.object(
                                module, "segment_witness", old if label == "baseline" else current
                            )
                        )
                    frames = {k: [] for k in keys}
                    start = perf_counter_ns()
                    for control in controls:
                        for key, frame in batch.step(H, control).items():
                            frames[key].append(frame)
                    stop = perf_counter_ns()
                after = batch.stats()
                values = list(frames.values())
                assert all(f.report.accepted for world in values for f in world)
                counters = {k: v - before[k] for k, v in after.items() if k != "backend"}
                return values, dict(
                    wall_ns=stop - start,
                    counters=counters,
                    hashes=summary(values),
                    clocks_s=native(
                        [[f.report.substeps[0].timings for f in group] for group in values]
                    ),
                )

            frozen, expected = run("baseline")
            qualified, warm = run("candidate")
            assert warm["hashes"] == expected["hashes"] and warm["counters"] == expected["counters"]
            if not holdout:
                assert expected["hashes"] == previous
                assert expected["counters"] == dict(
                    completed_launches=12,
                    sites=416,
                    uploads=12,
                    downloads=12,
                    uploaded_bytes=79872,
                    downloaded_bytes=30720,
                )
                proofs = [certify(w, f) for w, f in zip(worlds, qualified, strict=True)]
            else:
                proofs = [certify(w, f) for w, f in zip(worlds[:2], qualified[:2], strict=True)]
                assert all(
                    not f.report.substeps[0].diagnostics["robot_contact"]["cloth_links"]
                    for f in qualified[-1]
                )
                assert not np.array_equal(
                    qualified[-1][-1].state.domains["cloth"].position,
                    worlds[-1].initial.domains["cloth"].position,
                )
            pairs = []
            order = (
                ("baseline", "candidate") if a.first == "baseline" else ("candidate", "baseline")
            )
            for i in range(a.pairs):
                seq = order if i % 2 == 0 else order[::-1]
                pair = {"order": seq}
                for label in seq:
                    _, obs = run(label)
                    assert (
                        obs["hashes"] == expected["hashes"]
                        and obs["counters"] == expected["counters"]
                    )
                    pair[label] = obs
                pairs.append(pair)
            times = {label: median(p[label]["wall_ns"] for p in pairs) / 1e6 for label in order}
            result["holdout" if holdout else "w4x8"] = dict(
                p50_ms=times,
                pairs=pairs,
                expected=expected,
                baseline_frames=native(frozen),
                candidate_frames=native(qualified),
                proof=native(proofs),
            )
    with a.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, allow_nan=False)
    print(json.dumps({k: v["p50_ms"] for k, v in result.items() if "p50_ms" in v}))


if __name__ == "__main__":
    main()
