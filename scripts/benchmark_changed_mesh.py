"""G4 worst case for mesh reuse: authored material geometry moves before every query."""

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np

import phymeo as ph

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "scale"))
from g4_geometry_cases import WORLD_IDS, check_geometry, copies, digest, frozen  # noqa: E402


def translated(paths, z):
    shift = np.array([0.0, 0.0, z])
    return {
        k: ph.CollisionMotion(
            p.start + shift,
            None if p.end is None else p.end + shift,
            None
            if p.turn is None
            else ph.RigidTurn(
                np.array(p.turn.pivot_start) + shift,
                np.array(p.turn.pivot_end) + shift,
                p.turn.axis,
                p.turn.angle,
            ),
        )
        for k, p in paths.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("host", "cuda"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--expected", type=Path, help="before-change receipt on these exact translated frames"
    )
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    source, frames = frozen("mesh112")
    shifts = (0.0, 1 / 32, -1 / 32)
    workload = [translated(frames[-1], 3 / 32)] + [
        translated(f, z) for f, z in zip(frames, shifts, strict=True)
    ]
    w = WORLD_IDS[0]
    models, packed = copies(source, workload, (w,))
    rule = {w: ph.CollisionPolicy()}
    prior = None if a.expected is None else json.loads(a.expected.read_text(encoding="utf8"))
    if a.backend == "cuda":
        from tyx import runtime_stats
    with ph.WorldCollisionWorkspace(models, a.backend, cache_root=a.cache_root) as work:

        def measure(motion):
            before = work.stats()
            native = None if a.backend == "host" else runtime_stats()
            t = perf_counter_ns()
            out = work.detect(motion, rule)[w]
            wall = perf_counter_ns() - t
            after = work.stats()
            actual = None if native is None else runtime_stats()
            counts = {
                k: after[k] - before.get(k, 0)
                for k in (
                    "validated_mesh_attempts",
                    "validated_mesh_builds",
                    "validated_mesh_hits",
                    "narrow_ns",
                    "broad_ns",
                    "envelope_ns",
                )
                if k in after
            }
            return out, dict(
                wall_ns=wall,
                delta=counts,
                tyx_launches=None
                if native is None
                else actual.cuda_kernel_launches - native.cuda_kernel_launches,
            )

        _, cold = measure(packed[0])
        signatures, records = {}, []
        for repeat in range(3):
            for index, paths in enumerate(packed[1:]):
                out, sample = measure(paths)
                sig = digest(out, w)
                if index not in signatures:
                    check_geometry("mesh112", models[0][1], paths[w], out, 1)
                    signatures[index] = sig
                assert signatures[index] == sig
                if prior is not None:
                    assert sig == prior["frames"][str(index)]
                    assert sample["delta"]["validated_mesh_attempts"] == 1
                    assert sample["delta"]["validated_mesh_builds"] == 1
                    assert sample["delta"]["validated_mesh_hits"] == 0
                records.append(dict(repeat=repeat, frame=index, **sample))
        cycles = [sum(r["wall_ns"] for r in records if r["repeat"] == rep) for rep in range(3)]
        report = dict(
            schema="phymeo-g4-changed-mesh/1",
            backend=a.backend,
            shifts_z_m=shifts,
            cold=cold,
            frames=signatures,
            samples=records,
            cycles_ns=cycles,
            median_cycle_ns=median(cycles),
            stats=work.stats(),
            verified_against=None if a.expected is None else str(a.expected),
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(
        f"mesh112 W1 all input poses change, three warm cycles median {median(cycles) * 1e-9:.6f} s"
    )


if __name__ == "__main__":
    main()
