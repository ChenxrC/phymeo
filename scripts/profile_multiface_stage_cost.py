"""Attribute one shared wall clock across suspended world stacks and the active TYX caller."""

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from threading import get_ident
from time import perf_counter_ns
from unittest.mock import patch

from benchmark_multiface_phase_batch import STEPS, H, certify, make_world, summary, whole_window

import phymeo as ph
from phymeo.backends import tyx_multiface_contact as group
from phymeo.geometry import _static_capsule_face as static
from phymeo.runtime import scene_stepper
from phymeo.runtime.registry import StateRegistry
from phymeo.solvers.cloth_link_contact import ClothLinkContacts


class WallTimeline:
    """Each wall interval is assigned once; simultaneous parked stacks are separate diagnostics."""

    def __init__(self):
        self.main = get_ident()
        self.records = []

    def attach(self, stack, owner, name, label):
        original = getattr(owner, name)

        @wraps(original)
        def observe(*args, **kwargs):
            category = label
            if label == "material_witness":
                _, lo, hi = args[5]
                category = "material_singleton" if lo == hi else "material_span"
            tid, begin = get_ident(), perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                self.records.append((begin, perf_counter_ns(), tid, category))

        stack.enter_context(patch.object(owner, name, observe))

    def instrument(self, stack, layer):
        for owner, name, label in (
            (group._WorldCall, "run", "world_stack"),
            (group._WorldCall, "submit", "parked"),
            (group._WorldCall, "next_event", "await_world"),
            (ThreadPoolExecutor, "submit", "schedule_world"),
            (group.TyxMultifaceContactBatch, "_evaluate", "assemble_pages"),
            (ph.TyxNormalLayer, "evaluate", "law_wrapper"),
            (type(layer._in), "copy_from", "h2d"),
            (type(layer._out), "to_numpy", "d2h"),
            (layer, "_kernel", "kernel_submit"),
            (layer._runtime, "sync", "sync"),
            (scene_stepper.SceneStepper, "step", "world_mechanics"),
            (StateRegistry, "snapshot", "state_and_inputs_snapshot"),
            (StateRegistry, "clone_state", "state_clone"),
            (StateRegistry, "validate_candidate", "candidate_validation"),
            (scene_stepper, "own_diagnostics", "diagnostic_publication"),
            (ClothLinkContacts, "evaluate", "material_assembly_and_scatter"),
            (ClothLinkContacts, "_witness", "pose_query_other"),
            (static, "_plane_minimum", "test_plane_minimum"),
            (static, "_boundary_minimum", "test_boundary_minimum"),
            (static, "segment_witness", "material_witness"),
            (ph.CompliantClothContactProgram, "_check_link_motion", "path_validation"),
        ):
            self.attach(stack, owner, name, label)

    def partition(self, begin, end):
        # Records are nested on each OS thread. For an interval that contains a
        # parked world, discard that world's entire paused stack from wall ownership.
        edges = [(begin, 0, 0, "", 0), (end, 0, 0, "", 0)]
        calls = Counter()
        spans = []
        for i, (a, b, tid, label) in enumerate(self.records, 1):
            a, b = max(begin, a), min(end, b)
            if a >= b:
                continue
            calls[label] += 1
            spans.append((a - begin, b - begin, tid, label))
            edges.extend(((a, 1, tid, label, i), (b, -1, tid, label, i)))
        active, totals, blocked = defaultdict(dict), Counter(), Counter()
        previous = begin
        for at, sign, thread, label, token in sorted(edges):
            if previous < at:
                dt = at - previous
                work = []
                controller = active[self.main]
                for tid, stack in active.items():
                    if tid == self.main or not stack:
                        continue
                    if "parked" in stack.values():
                        blocked["parked_world_stack_sum"] += dt
                    else:
                        work.append(next(reversed(stack.values())))
                lead = next(reversed(controller.values()), "scheduler_or_other_caller")
                if lead == "await_world":
                    blocked["caller_await_inclusive"] += dt
                if len(work) > 1 or (
                    work
                    and lead not in ("await_world", "schedule_world", "scheduler_or_other_caller")
                ):
                    owner = "handoff_overlap"
                elif work:
                    owner = work[0]
                else:
                    owner = "handoff_idle" if lead == "await_world" else lead
                totals[owner] += dt
            if sign == 1:
                active[thread][token] = label
            elif sign == -1:
                del active[thread][token]
            previous = at
        assert sum(totals.values()) == end - begin and all(not x for x in active.values())
        return dict(
            exclusive_ns=dict(totals),
            inclusive_wait_ns=dict(blocked),
            calls=dict(calls),
            main_thread=self.main,
            spans=spans,
        )


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--cache-root", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--pairs", type=int, default=3)
    args = cli.parse_args()
    if args.out.exists() or args.pairs < 1:
        cli.error("fresh output and positive pair count required")
    settings = ph.StepSettings(
        substeps=1, position_iterations=1, velocity_iterations=1, record_substeps=True
    )
    with ExitStack() as resources:
        solo = resources.enter_context(ph.TyxNormalLayer("cuda", cache_root=args.cache_root))
        worlds = [make_world(i, 8, solo) for i in range(4)]
        steppers = [resources.enter_context(w.program.stepper(settings)) for w in worlds]
        ordinary = whole_window(worlds, steppers, isolated=True)
        assert solo.stats()["completed_launches"] == 48 and solo.stats()["sites"] == 416
        expected = summary(ordinary)
        for w, frames in zip(worlds, ordinary, strict=True):
            certify(w, frames)
        batch = resources.enter_context(
            ph.TyxMultifaceContactBatch(
                [(i, ph.ClothLinkWorld(w.program, w.initial)) for i, w in enumerate(worlds)],
                "cuda",
                settings=settings,
                cache_root=args.cache_root,
            )
        )
        controls = [{i: w.controls[t] for i, w in enumerate(worlds)} for t in range(STEPS)]

        def measure(traced):
            batch.reset()
            counts = batch.stats()
            timer = WallTimeline()
            with ExitStack() as instruments:
                if traced:
                    timer.instrument(instruments, batch._layer)
                frames = [[] for _ in worlds]
                start = perf_counter_ns()
                for control in controls:
                    for key, frame in batch.step(H, control).items():
                        frames[key].append(frame)
                stop = perf_counter_ns()
            assert summary(frames) == expected
            after = batch.stats()
            diff = {k: v - counts[k] for k, v in after.items() if k != "backend"}
            assert (
                diff["completed_launches"],
                diff["sites"],
                diff["uploaded_bytes"],
                diff["downloaded_bytes"],
            ) == (12, 416, 79872, 30720)
            record = dict(wall_ns=stop - start, counts=diff)
            if traced:
                record.update(timer.partition(start, stop))
                c = record["calls"]
                assert c["h2d"] == c["d2h"] == c["kernel_submit"] == c["sync"] == 12
                assert c["pose_query_other"] == c["material_singleton"] + c["material_span"] == 384
            return record

        measure(False)
        pairs = []
        for i in range(args.pairs):
            order = ("plain", "trace") if i % 2 == 0 else ("trace", "plain")
            pairs.append({mode: measure(mode == "trace") for mode in order})
    output = dict(reference_hashes=expected, pairs=pairs)
    with args.out.open("x", encoding="utf-8") as file:
        json.dump(output, file, allow_nan=False)
    tally = Counter()
    for row in pairs:
        tally.update(row["trace"]["exclusive_ns"])
    print(json.dumps({k: round(v / len(pairs) / 1e6, 3) for k, v in tally.most_common()}))


if __name__ == "__main__":
    main()
