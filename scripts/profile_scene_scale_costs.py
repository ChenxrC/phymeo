"""One W1 diagnostic: real nested caller intervals, never a formal performance score."""

import argparse
import importlib.util
import json
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from threading import get_ident
from time import perf_counter_ns

import phymeo as ph

_TOOL = Path(__file__).with_name("benchmark_scene_scale_costs.py")
_SPEC = importlib.util.spec_from_file_location("_c1_diagnostic_driver", _TOOL)
driver = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = driver
_SPEC.loader.exec_module(driver)


class CallTimeline:
    """Trace this calling thread without changing any product or SDK function."""

    def __init__(self):
        self.records = []
        self._open = {}
        self._labels = {}
        self._serial = 0

    def _bind(self, function, label):
        code = getattr(function, "__func__", function).__code__
        if code in self._labels and self._labels[code] != label:
            raise RuntimeError("two diagnostic labels share the same function body")
        self._labels[code] = label

    def _configure(self, scene):
        import tyx as tx
        from tyx.cuda_array import CudaArray
        from tyx.kernel import KernelFunction

        from phymeo.backends.tyx_normal_layer import TyxNormalLayer
        from phymeo.backends.tyx_rigid import TyxRigidSemiImplicitSolver
        from phymeo.geometry import collision_pipeline
        from phymeo.solvers.collision_loads import CollisionLoadStep, CollisionLoadSystem

        self._labels.clear()
        for function, label in (
            (collision_pipeline._envelopes, "geometry_envelopes"),
            (collision_pipeline._detect_pairs, "geometry_narrow"),
            (collision_pipeline.CollisionWorkspace._classify_world_envelopes, "broad_wrapper"),
            (CollisionLoadSystem._sample, "material_assembly"),
            (CollisionLoadSystem._paths, "state_to_geometry"),
            (CollisionLoadSystem._checked_path, "path_validation"),
            (CollisionLoadStep.owned_copy, "published_frame_copy"),
            (type(scene).geometry, "geometry_snapshot"),
            (TyxNormalLayer.evaluate, "response_law_wrapper"),
            (TyxRigidSemiImplicitSolver.step, "rigid_step_wrapper"),
            (CudaArray.__init__, "allocation_and_initialization"),
            (CudaArray.empty, "allocation"),
            (CudaArray.free, "deallocation"),
            (CudaArray.copy_from, "h2d_api"),
            (CudaArray.to_numpy, "d2h_api"),
            (KernelFunction.__call__, "kernel_call_api"),
            (tx.Runtime.sync, "synchronization_api"),
        ):
            self._bind(function, label)
        try:
            from phymeo.runtime.contact_history import advance_contact_history
        except ModuleNotFoundError as error:
            if error.name != "phymeo.runtime.contact_history":
                raise
        else:
            self._bind(advance_contact_history, "contact_history_staging")

    def _observe(self, frame, event, arg):
        if event == "call" and frame.f_code in self._labels:
            self._serial += 1
            self._open[id(frame)] = (
                perf_counter_ns(),
                self._serial,
                self._labels[frame.f_code],
                len(self._open),
            )
        elif event == "return":
            item = self._open.pop(id(frame), None)
            if item is not None:
                start, token, label, depth = item
                self.records.append((start, perf_counter_ns(), token, label, depth))

    @contextmanager
    def active(self, scene):
        if sys.getprofile() is not None:
            raise RuntimeError("do not combine the diagnostic with another Python profiler")
        self._configure(scene)
        self.records.clear()
        self._open.clear()
        self._serial = 0
        self.thread = get_ident()
        sys.setprofile(self._observe)
        try:
            yield
        finally:
            sys.setprofile(None)
        if self._open:
            raise AssertionError("a diagnostic function did not return")

    def partition(self, begin, end):
        """Partition measured caller wall time; inclusive children are never added again."""
        if end < begin:
            raise ValueError("negative diagnostic interval")
        edges = [(begin, 0, 0), (end, 0, 0)]
        records, calls, inclusive = {}, Counter(), Counter()
        spans = []
        for a, b, token, label, depth in self.records:
            a, b = max(begin, a), min(end, b)
            if a >= b:
                continue
            if token in records:
                raise AssertionError("duplicate interval identity")
            records[token] = (label, depth)
            edges.extend(((a, 1, token), (b, -1, token)))
            calls[label] += 1
            inclusive[label] += b - a
            spans.append(dict(begin_ns=a - begin, end_ns=b - begin, label=label, depth=depth))
        active, exclusive, previous = {}, Counter(), begin
        for at, event, token in sorted(edges):
            if at > previous:
                deepest = max(active, key=lambda k: (active[k][1], k)) if active else None
                label = active[deepest][0] if deepest is not None else "other_public_caller"
                exclusive[label] += at - previous
            if event == 1:
                active[token] = records[token]
            elif event == -1:
                del active[token]
            previous = at
        if active or sum(exclusive.values()) != end - begin:
            raise AssertionError("exclusive caller time does not cover the public boundary")
        return dict(
            exclusive_ns=dict(exclusive),
            inclusive_ns=dict(inclusive),
            calls=dict(calls),
            spans=spans,
            caller_wall_ns=end - begin,
            pure_gpu_device_ns=None,
            scope="current caller thread only; nested API wall times include runtime overhead",
        )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--case", choices=("free", "loaded", "birth"), default="loaded")
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--expected-head", required=True)
    args = cli.parse_args()
    if args.out.exists():
        cli.error("use a fresh diagnostic directory")
    args.out.mkdir(parents=True)
    source = driver.provenance(args.expected_head)
    cache, make = args.out / "cache", driver.builder(ph, args.case, 1)
    report = dict(
        source=source,
        case=args.case,
        worlds=1,
        diagnostic_only=True,
        windows=[],
        scope="not a formal score; Python profile overhead is present; no pure GPU time",
    )
    try:
        for phase, timeline in (("untraced_prepare", None), ("diagnostic", CallTimeline())):
            report["windows"].append(
                driver.run_window(
                    make,
                    args.case,
                    1,
                    phase,
                    "cuda",
                    cache,
                    args.out / f"{phase}.json.gz",
                    profile=timeline,
                )
            )
        report["source_after"] = driver.provenance(args.expected_head)
        report["success"] = True
    except BaseException as error:
        report["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        with (args.out / "profile.json").open("x", encoding="utf8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(success=True, diagnostic_only=True, case=args.case, worlds=1)))


if __name__ == "__main__":
    main()
