"""Profile real, evolving multiface cloth on the public single-world program and FP32 CUDA normal layer."""

import argparse
import json
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from unittest.mock import patch

from profile_loaded_worlds import ExclusiveTimes

import phymeo as ph
from phymeo.backends.tyx_world_contact import ClothLinkWorld, TyxWorldContactBatch
from phymeo.solvers import cloth_link_contact
from phymeo.solvers.cloth_link_contact import ClothLinkContacts
from phymeo.solvers.compliant_cloth_contact import CompliantClothContactProgram

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "scale"))
from multiface_load_cases import STEPS, H, certify, make_world, native, physical_hash, whole_window  # noqa: E402


def counts(operator, alloc):
    return {**{k: v for k, v in operator.stats().items() if k != "backend"}, **alloc}


def instrument(stack, operator, timer):
    boundaries = [
        (ClothLinkContacts, "_witness", "closest_feature"),
        (ClothLinkContacts, "evaluate", "material_load_and_scatter"),
        (cloth_link_contact, "capsule_material_sites", "material_address"),
        (CompliantClothContactProgram, "_check_link_motion", "path_check"),
        (CompliantClothContactProgram, "_link_geometry", "link_fk"),
        (CompliantClothContactProgram, "accelerations", "other_acceleration"),
        (ph.TyxNormalLayer, "evaluate", "law_host_wrapper"),
        (type(operator._in), "copy_from", "h2d_api"),
        (type(operator._out), "to_numpy", "d2h_api"),
        (operator, "_kernel", "kernel_submission_api"),
        (operator._runtime, "sync", "existing_sync_api"),
    ]
    for owner, method, group in boundaries:
        stack.enter_context(patch.object(owner, method, timer.wrap(getattr(owner, method), group)))


def measure(worlds, steppers, operator, alloc, *, traced=False, isolated=False):
    before = counts(operator, alloc)
    timer = ExclusiveTimes()
    with ExitStack() as hooks:
        if traced:
            instrument(hooks, operator, timer)
        start = perf_counter_ns()
        output = whole_window(worlds, steppers, isolated=isolated)
        elapsed = perf_counter_ns() - start
    after = counts(operator, alloc)
    counters = {k: after[k] - before[k] for k in before}
    if traced:
        timer.ns["other_host_and_publication"] = elapsed - sum(timer.ns.values())
        assert min(timer.ns.values()) >= 0 and sum(timer.ns.values()) == elapsed
        assert (
            timer.calls["kernel_submission_api"]
            == timer.calls["existing_sync_api"]
            == counters["completed_launches"]
        )
        assert (
            timer.calls["h2d_api"] == counters["uploads"]
            and timer.calls["d2h_api"] == counters["downloads"]
        )
    # Wall clocks are intentionally different. Preserve every timer separately;
    # typed hashes cover the entire physical publication, including diagnostics.
    hashes = [[physical_hash(frame) for frame in frames] for frames in output]
    clocks = [
        [[dict(sub.timings) for sub in frame.report.substeps] for frame in frames]
        for frames in output
    ]
    return output, dict(
        wall_ns=elapsed,
        counters=counters,
        phase_ns=dict(timer.ns),
        calls=dict(timer.calls),
        frame_hashes=hashes,
        raw_substep_clocks_s=clocks,
    )


def batch_boundary(world):
    try:
        with TyxWorldContactBatch([(13, ClothLinkWorld(world.program, world.initial))], "cuda"):
            pass
    except ValueError as error:
        if "one three-node face" not in str(error):
            raise
        return dict(
            status="rejected", reason=str(error), faces=len(world.faces), nodes=len(world.reference)
        )
    raise AssertionError(
        "dedicated cross-world batch now accepts this case; revisit benchmark routing"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, choices=(1, 4, 8, 16), required=True)
    parser.add_argument("--faces", type=int, choices=(2, 8), required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--first", choices=("plain", "profile"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    if a.pairs < 1 or a.out.exists():
        parser.error("positive pair count and a fresh output path required")
    import tyx as tx

    alloc = Counter(workspace_cuda_arrays=0, workspace_cuda_allocated_bytes=0)
    cuda_empty = tx.cuda_empty

    def allocate(*args, **kwargs):
        result = cuda_empty(*args, **kwargs)
        alloc["workspace_cuda_arrays"] += 1
        alloc["workspace_cuda_allocated_bytes"] += result.nbytes
        return result

    settings = ph.StepSettings(
        substeps=1, position_iterations=1, velocity_iterations=1, record_substeps=True
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(tx, "cuda_empty", allocate))
        setup_start = perf_counter_ns()
        operator = stack.enter_context(ph.TyxNormalLayer("cuda", cache_root=a.cache_root))
        solo = [make_world(i, a.faces, operator) for i in range(a.worlds)]
        mixed = [make_world(i, a.faces, operator) for i in range(a.worlds)]
        refused = batch_boundary(mixed[0])
        solo_steppers = [stack.enter_context(w.program.stepper(settings)) for w in solo]
        steppers = [stack.enter_context(w.program.stepper(settings)) for w in mixed]
        setup_ns = perf_counter_ns() - setup_start
        setup_alloc = dict(alloc)
        assert setup_alloc["workspace_cuda_arrays"] == 2
        reference, expected = measure(solo, solo_steppers, operator, alloc, isolated=True)
        proofs = [certify(w, frames) for w, frames in zip(solo, reference, strict=True)]

        def check(sample):
            assert sample["frame_hashes"] == expected["frame_hashes"]
            assert sample["counters"] == expected["counters"]
            assert (
                sample["counters"]["workspace_cuda_arrays"] == 0
                and sample["counters"]["completed_launches"] > 0
            )

        _, warm = measure(mixed, steppers, operator, alloc)
        check(warm)
        pairs = []
        order = ("plain", "profile") if a.first == "plain" else ("profile", "plain")
        for k in range(a.pairs):
            visits = order if k % 2 == 0 else order[::-1]
            pair = {"order": visits}
            for name in visits:
                _, sample = measure(mixed, steppers, operator, alloc, traced=name == "profile")
                check(sample)
                pair[name] = sample
            pairs.append(pair)
    summary = {}
    for name in ("plain", "profile"):
        times = sorted(p[name]["wall_ns"] for p in pairs)
        summary[name] = dict(
            p50_ms=median(times) / 1e6,
            max_ms=max(times) / 1e6,
            accepted_world_substeps=a.worlds * STEPS * len(times),
            world_substeps_s=a.worlds * STEPS * len(times) * 1e9 / sum(times),
        )
    phase_ns = Counter()
    for pair in pairs:
        phase_ns.update(pair["profile"]["phase_ns"])
    total = sum(phase_ns.values())
    summary["phases"] = {
        k: dict(mean_ms_per_window=v / len(pairs) / 1e6, share=v / total)
        for k, v in phase_ns.items()
    }
    actual_faces = [f for proof in proofs for mid in proof["midpoints"] for f in mid["faces"]]
    summary["midpoint_work"] = dict(
        positive_face_samples=len(actual_faces),
        material_sites=sum(f["site_count"] for f in actual_faces),
        smallest_face_load_n=min(f["normal_force_n"] for f in actual_faces),
        largest_face_load_n=max(f["normal_force_n"] for f in actual_faces),
    )
    receipt = dict(
        worlds=a.worlds,
        triangles_per_world=a.faces,
        h_s=H,
        steps_per_window=STEPS,
        reference_area_m2_per_world=float(mixed[0].areas.sum()),
        comparison="independent world-major reference, then step-major interleave; physical worlds run on HOST",
        timing_hash_contract="only SceneSubstepReport.timings scalar values are set to zero; original keys and all raw clocks are retained",
        actual_device="cuda",
        device_timing="exclusive host API wall clocks; no CUDA event or pure device execution time",
        batch_boundary=refused,
        setup_ns=setup_ns,
        normal_layer_workspace_setup_allocation=setup_alloc,
        reference_window=expected,
        warmup=warm,
        paired_windows=pairs,
        summary=summary,
        faces_and_motion=proofs,
        independent_frames=native(reference),
        inputs=[
            dict(
                world=i,
                initial=native(w.initial),
                controls=native(w.controls),
                face_nodes=w.faces,
                reference_vertices_m=w.reference.tolist(),
                area_m2=w.areas.tolist(),
            )
            for i, w in enumerate(mixed)
        ],
    )
    with a.out.open("x", encoding="utf8") as out:
        json.dump(receipt, out, allow_nan=False, separators=(",", ":"))
    print(
        "W",
        a.worlds,
        "faces/world",
        a.faces,
        "summary",
        json.dumps(summary),
        "actual CUDA counts/window",
        expected["counters"],
    )


if __name__ == "__main__":
    main()
