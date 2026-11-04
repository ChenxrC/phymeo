"""Attribute elapsed host time in unchanged TYX contact-loaded world steps."""

import argparse
import json
import sys
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from functools import wraps
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from unittest.mock import patch

from phymeo.backends import _world_contact_sites
from phymeo.backends._tyx_world_contact_engine import WorldContactEngine
from phymeo.backends._world_contact_sites import WorldMaterial
from phymeo.backends.tyx_world_batch import TyxWorldBatch
from phymeo.backends.tyx_world_contact import TyxWorldContactBatch
from phymeo.solvers.cloth_link_contact import ClothLinkContacts
from phymeo.solvers.compliant_cloth_contact import CompliantClothContactProgram

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "scale"))
from mixed_physical_cases import WINDOW, MixedRun, Population, population, signature  # noqa: E402

# These are wall times at real Python call boundaries. Device submission with
# its existing sync stays together; no extra device synchronization is added.
BOUNDARIES = (
    (WorldMaterial, "geometry", "pose_geometry"),
    (CompliantClothContactProgram, "_link_geometry", "link_geometry"),
    (ClothLinkContacts, "_witness", "closest_feature"),
    (_world_contact_sites, "capsule_material_sites", "material_address"),
    (WorldMaterial, "site_rows", "material_address"),
    (WorldMaterial, "elasticity", "material_elasticity"),
    (CompliantClothContactProgram, "_check_link_motion", "path_check"),
    (WorldMaterial, "audit", "acceptance"),
    (WorldMaterial, "sample", "publication"),
    (WorldMaterial, "point", "publication"),
    (TyxWorldContactBatch, "_publish_points", "publication"),
    (WorldContactEngine, "evaluate", "device_submit_and_wait"),
    (WorldContactEngine, "upload", "h2d_api"),
    (WorldContactEngine, "read", "d2h_api"),
    (TyxWorldBatch, "_upload", "h2d_api"),
    (TyxWorldBatch, "_read", "d2h_api"),
    (TyxWorldBatch, "_launch", "device_submit_and_wait"),
    (TyxWorldBatch, "_sync", "device_submit_and_wait"),
    (MixedRun, "snapshot", "publication"),
)


class ExclusiveTimes:
    def __init__(self):
        self.ns = Counter()
        self.calls = Counter()
        self.children = []
        self.active = []

    def wrap(self, function, group):
        @wraps(function)
        def measured(*args, **kwargs):
            name = group
            if group == "closest_feature":
                name = (
                    "path_closest_feature"
                    if "path_check" in self.active
                    else "pose_closest_feature"
                )
            self.active.append(name)
            self.children.append(0)
            start = perf_counter_ns()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = perf_counter_ns() - start
                self.active.pop()
                self.ns[name] += elapsed - self.children.pop()
                self.calls[name] += 1
                if self.children:
                    self.children[-1] += elapsed

        return measured

    def attach(self, stack):
        for owner, method, group in BOUNDARIES:
            original = getattr(owner, method)
            stack.enter_context(patch.object(owner, method, self.wrap(original, group)))


def make_case(size):
    if size != 1:
        return population(size)
    full = population(4)
    key, world = next((k, w) for k, w in full.worlds if full.roles[k] == "contact")
    return Population(
        [(key, world)], {key: "contact"}, [({}, {key: loads[key]}) for _, loads in full.inputs]
    )


def measure(runner, traced):
    runner.reset()
    before = runner.counts()
    timer = ExclusiveTimes()
    with ExitStack() as patches:
        if traced:
            timer.attach(patches)
        start = perf_counter_ns()
        frames = [runner.advance(i) for i in range(WINDOW)]
        elapsed = perf_counter_ns() - start
    after = runner.counts()
    counts = {k: v - before[k] for k, v in after.items()}
    if not all(all(dict(frame.flags).values()) for frame in frames):
        raise RuntimeError("a measured world rejected its physical step")
    if traced:
        timer.ns["unattributed_wall"] = elapsed - sum(timer.ns.values())
        if (
            timer.children
            or timer.active
            or min(timer.ns.values()) < 0
            or not all(
                timer.calls[key]
                for key in (
                    "pose_closest_feature",
                    "path_check",
                    "acceptance",
                    "device_submit_and_wait",
                    "d2h_api",
                )
            )
        ):
            raise RuntimeError("incomplete or overlapping instrumentation")
    return frames, dict(
        wall_ns=elapsed,
        exclusive_ns=dict(timer.ns),
        calls=dict(timer.calls),
        device_counts=counts,
        sha256=signature(frames),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, choices=(1, 4, 16, 32), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--reference", type=Path, help="retained size-matched CUDA benchmark JSON (W4/16/32)"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5)
    args = parser.parse_args()
    if args.out.exists() or args.pairs < 3 or ((args.worlds == 1) != (args.reference is None)):
        parser.error(
            "require fresh output, three paired windows or more, and size-matched reference for W4/16/32"
        )
    saved = json.loads(args.reference.read_text(encoding="utf8")) if args.reference else None
    if saved and saved["worlds"] != args.worlds:
        parser.error("reference world count differs")
    case = make_case(args.worlds)
    rows = []
    ref = None
    frame_file = args.out.with_suffix(".frames.json")
    if frame_file.exists() or frame_file == args.out:
        parser.error("choose a fresh report and frame sidecar")
    with ExitStack() as stack:
        grouped = MixedRun(case, True, "cuda", args.cache_root, stack)
        if saved:
            expected = saved["window_sha256"]
        else:
            independent = MixedRun(case, False, "cuda", args.cache_root, stack)
            single_frames, ref = measure(independent, False)
            expected = ref["sha256"]
            if abs(single_frames[0].material[0][1].audit.sample.normal_force_n - 2.0) > 1e-6:
                raise AssertionError("the one-world baseline must carry the physical load")
        warm, _ = measure(grouped, False)
        if signature(warm) != expected:
            raise AssertionError("warmup did not reproduce the full physical reference")
        payload = json.dumps(
            [asdict(frame) for frame in warm],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf8")
        if sha256(payload).hexdigest() != expected:
            raise AssertionError("saved frame encoding differs from the physical reference")
        frame_file.parent.mkdir(parents=True, exist_ok=True)
        with frame_file.open("xb") as output:
            output.write(payload)
        for i in range(args.pairs):
            row = {}
            for profiled in (bool(i % 2), not bool(i % 2)):
                frames, sample = measure(grouped, profiled)
                if sample["sha256"] != expected or frames != warm:
                    raise AssertionError("profile changed accepted output or history")
                if saved and sample["device_counts"] != saved["measurements"][0]["scene"]["counts"]:
                    raise AssertionError("steady device work differs from the retained benchmark")
                row["profiled" if profiled else "plain"] = sample
            if row["profiled"]["device_counts"] != row["plain"]["device_counts"]:
                raise AssertionError("sampling altered device work")
            rows.append(row)
    total = sum(row["profiled"]["wall_ns"] for row in rows)
    phases = {
        group: sum(row["profiled"]["exclusive_ns"][group] for row in rows) / total
        for group in rows[0]["profiled"]["exclusive_ns"]
    }
    result = dict(
        worlds=args.worlds,
        backend="cuda",
        steps_per_window=WINDOW,
        sha256=expected,
        reference=str(args.reference) if saved else "independent W1 loaded stepper",
        plain_p50_ms=median(r["plain"]["wall_ns"] for r in rows) / 1e6,
        profiled_p50_ms=median(r["profiled"]["wall_ns"] for r in rows) / 1e6,
        paired_clock_ratio=median(r["profiled"]["wall_ns"] / r["plain"]["wall_ns"] for r in rows),
        phase_fractions=phases,
        pairs=rows,
        frame_file=frame_file.name,
        single_world_reference=ref,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(
        f"W{args.worlds} {result['plain_p50_ms']:.2f} ms/plain, {result['profiled_p50_ms']:.2f} ms/instrumented; {phases}"
    )


if __name__ == "__main__":
    main()
