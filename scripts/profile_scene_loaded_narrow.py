"""C3: bounded, interleaved public-scene comparison and separate narrow diagnostics.

A retains the full dispatcher in THIS product revision; B uses its filtered completion.
This is not an old-version binary comparison and never changes installed product files.
"""

import argparse
import copy
import gzip
import hashlib
import importlib.util
import inspect
import json
import math
import sys
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from threading import Lock, get_ident
from time import perf_counter_ns

import phymeo as ph
from phymeo.geometry import collision_pipeline

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_c3_original_profile", ROOT / "scripts" / "profile_scene_scale_costs.py"
)
profile = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = profile
_SPEC.loader.exec_module(profile)
driver = profile.driver
H = driver.H
MATRIX = (("loaded", 1), ("birth", 1), ("loaded", 32), ("birth", 32))
FRAMES = tuple(range(1, 17))
VARIANTS = ("full_dispatch", "filtered_dispatch")
PUBLIC_CLOCK_FIELDS = ()
COUNTER_CLOCK_FIELDS = (
    "geometry.envelope_ns",
    "geometry.broad_ns",
    "geometry.midphase_ns",
    "geometry.narrow_ns",
)
EXTRA_TOOL_PATHS = {
    "diagnostic": Path(__file__).resolve(),
    "protocol": ROOT / "tests" / "scale" / "test_loaded_narrow_protocol.py",
}
MAX_SPANS_PER_FRAME = 100_000
MAX_TRACE_TO_UNTRACED_RATIO = 20.0
_FILTERED_DISPATCH = collision_pipeline._complete_pairs
_DISPATCH_LOCK = Lock()


def pair_order(frame):
    if type(frame) is not int or frame not in FRAMES:
        raise ValueError("observation is outside the declared frame indices")
    return VARIANTS if frame % 2 else VARIANTS[::-1]


@contextmanager
def dispatch_variant(variant):
    """A single calling-thread diagnostic; restore the exact entry callable on all exits."""
    if variant not in VARIANTS:
        raise ValueError("unknown dispatch variant")
    if not _DISPATCH_LOCK.acquire(blocking=False):
        raise RuntimeError("a diagnostic dispatch context is already active")
    original = collision_pipeline._complete_pairs
    try:
        if original is not _FILTERED_DISPATCH:
            raise RuntimeError("another dispatch replacement is active")
        collision_pipeline._complete_pairs = (
            collision_pipeline._detect_pairs if variant == VARIANTS[0] else original
        )
        yield
    finally:
        collision_pipeline._complete_pairs = original
        _DISPATCH_LOCK.release()


def assert_full_equal(left, right, path="$"):
    """No public field is removed, including unknown names or physical/sample times."""
    if type(left) is not type(right):
        raise AssertionError(f"{path}: type changed")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"{path}: fields changed")
        for key in left:
            assert_full_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        if len(left) != len(right):
            raise AssertionError(f"{path}: length changed")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_full_equal(a, b, f"{path}[{index}]")
    elif left != right:
        raise AssertionError(f"{path}: value changed ({str(left)[:80]} -> {str(right)[:80]})")


def _zero_clocks(value, *, nested):
    result = copy.deepcopy(value)
    for key in COUNTER_CLOCK_FIELDS:
        group, name = key.split(".")
        mapping = result.get(group, {}) if nested else result
        actual_key = name if nested else key
        if actual_key not in mapping:
            raise AssertionError(f"missing declared counter clock: {key}")
        clock = mapping[actual_key]
        if isinstance(clock, bool) or not isinstance(clock, (int, float)):
            raise AssertionError(f"invalid declared counter clock: {key}")
        if not math.isfinite(clock) or clock < 0:
            raise AssertionError(f"invalid declared counter clock: {key}")
        mapping[actual_key] = 0
    return result


def assert_counters_equal(left, right):
    assert_full_equal(_zero_clocks(left, nested=False), _zero_clocks(right, nested=False), "counters")


def canonical_payload(scene, frame, geometry):
    return driver.public_value(
        dict(
            frame=frame,
            geometry=geometry,
            contact_history=scene.contact_history,
            contact_events=scene.contact_events,
        )
    )


def provenance(expected_head=None):
    result = driver.provenance(expected_head)
    result["extra_tools"] = {
        name: dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for name, path in EXTRA_TOOL_PATHS.items()
    }
    return result


def verify_provenance(before, after):
    driver.verify_provenance(before, after)
    if set(before.get("extra_tools", {})) != {"diagnostic", "protocol"}:
        raise AssertionError("diagnostic tool changed: missing source binding")
    if before["extra_tools"] != after.get("extra_tools"):
        raise AssertionError("diagnostic tool changed during the measurement")


class NarrowTimeline(profile.CallTimeline):
    """Current-thread function spans plus two source-bound projection line regions.

    Construction remains a function span. The line regions distinguish actual face/edge
    search from the following distance bounds, resolution gate and material publication.
    Those latter operations are deliberately a combined region, not pure verification.
    """

    def _configure(self, scene):
        from phymeo.geometry import _convex_hull as hull
        from phymeo.geometry import _convex_support as support
        from phymeo.geometry import _convex_support_path as path
        from phymeo.runtime.contact_history import advance_contact_history
        from phymeo.solvers.collision_loads import CollisionLoadStep, CollisionLoadSystem

        self._labels.clear()
        for function, label in (
            (collision_pipeline._envelopes, "geometry_envelopes"),
            (_FILTERED_DISPATCH, "filtered_completion"),
            (collision_pipeline._detect_pairs, "geometry_narrow"),
            (collision_pipeline._pair, "pair_record"),
            (collision_pipeline.CollisionWorkspace._classify_world_envelopes, "broad_wrapper"),
            (hull.HullShape.__init__, "hull_reconstruction"),
            (hull.HullShape._boundary, "hull_boundary"),
            (hull.MaterialChart.__init__, "material_chart_build"),
            (hull.MaterialChart.weights, "material_chart_weights"),
            (support.static_convex_support, "static_support_dispatch"),
            (support._exterior_sphere_hull, "sphere_hull_projection"),
            (support._closest, "generic_simplex_closest"),
            (path._AffineSupport.__init__, "path_support_build"),
            (path._AffineSupport.pose, "path_sample_pose"),
            (path._AffineSupport.upper, "path_support_upper"),
            (path._Window.sample, "path_sample_dispatch"),
            (path._Window.clear_on, "path_interval_verify"),
            (path._Window.run, "path_window_driver"),
            (path._end_trace, "path_trace_finish"),
            (CollisionLoadSystem._sample, "material_assembly"),
            (CollisionLoadSystem._paths, "state_to_geometry"),
            (CollisionLoadSystem._checked_path, "path_validation"),
            (CollisionLoadStep.owned_copy, "published_frame_copy"),
            (type(scene).geometry, "geometry_snapshot"),
            (advance_contact_history, "contact_history_staging"),
        ):
            self._bind(function, label)
        if scene.stats()["backend"] != "host":
            import tyx as tx
            from tyx.cuda_array import CudaArray
            from tyx.kernel import KernelFunction

            from phymeo.backends.tyx_normal_layer import TyxNormalLayer
            from phymeo.backends.tyx_rigid import TyxRigidSemiImplicitSolver

            for function, label in (
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
        function = support._exterior_sphere_hull
        source, first = inspect.getsourcelines(function)
        names = {
            "projection_feature_search": "center = _vector(sp.start[0])",
            "projection_distance_material_finish": "delta = center - projected",
        }
        anchors = {}
        for label, text in names.items():
            matches = [first + index for index, line in enumerate(source) if line.strip() == text]
            if len(matches) != 1:
                raise RuntimeError("projection source anchor is not unique: " + label)
            anchors[label] = matches[0]
        if anchors["projection_feature_search"] >= anchors["projection_distance_material_finish"]:
            raise RuntimeError("projection source regions changed order")
        source_path = Path(inspect.getsourcefile(function)).resolve()
        self.source_binding = dict(
            path=str(source_path),
            sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            function=function.__qualname__,
            anchors=anchors,
            region_scope="search / combined distance bounds, resolution check and material finish",
        )
        self._projection_code = function.__code__
        self._line_labels = {line: label for label, line in anchors.items()}

    def _observe(self, frame, event, arg):
        if get_ident() != self.thread:
            raise RuntimeError("diagnostic spans crossed the caller thread")
        if len(self.records) >= MAX_SPANS_PER_FRAME:
            raise RuntimeError("diagnostic span capacity exceeded")
        super()._observe(frame, event, arg)

    def _close_region(self, frame, now):
        item = self._regions.pop(id(frame), None)
        if item is not None:
            start, token, label, depth = item
            self.records.append((start, now, token, label, depth))

    def _line(self, frame, event, arg):
        if frame.f_code is not self._projection_code:
            return None
        if event == "line" and frame.f_lineno in self._line_labels:
            now = perf_counter_ns()
            self._close_region(frame, now)
            self._serial += 1
            parent = self._open.get(id(frame))
            if parent is None:
                raise RuntimeError("projection region has no enclosing function span")
            self._regions[id(frame)] = (
                now, self._serial, self._line_labels[frame.f_lineno], parent[3] + 1
            )
        elif event == "return":
            self._close_region(frame, perf_counter_ns())
        return self._line

    @contextmanager
    def active(self, scene):
        if sys.getprofile() is not None or sys.gettrace() is not None:
            raise RuntimeError("another Python profiler or trace is active")
        self._configure(scene)
        self.records.clear()
        self._open.clear()
        self._regions = {}
        self._serial = 0
        self.thread = get_ident()
        sys.setprofile(self._observe)
        sys.settrace(self._line)
        try:
            yield
        finally:
            sys.settrace(None)
            sys.setprofile(None)
        if self._open or self._regions:
            raise AssertionError("a diagnostic region did not return")


def _save(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _physical_row(frame):
    return dict(
        actual_contacts=len(frame.loads.contacts),
        actual_sites=sum(len(c.sites) for c in frame.loads.contacts),
        actual_required_candidates=frame.path.required_candidates,
        published_pairs=len(frame.path.pairs),
        different_world_rows=sum(p.first.world != p.second.world for p in frame.path.pairs),
        force_worlds=[c.first.world for c in frame.loads.contacts if c.force_n > 3e-4],
    )


def _compare_observations(left, right):
    for key in (
        "actual_contacts", "actual_sites", "actual_required_candidates", "published_pairs",
        "different_world_rows", "force_worlds", "frame_logical_sha256", "physical_errors",
    ):
        assert_full_equal(left[key], right[key], key)
    assert_counters_equal(left["counters"], right["counters"])
    assert_full_equal(left["native_counters"], right["native_counters"], "native_counters")
    for key in ("statistics_before", "statistics_after"):
        assert_full_equal(
            _zero_clocks(left[key], nested=True), _zero_clocks(right[key], nested=True), key
        )


def _check_trace_counts(left, right):
    a = left["diagnostic_intervals"]["calls"]
    b = right["diagnostic_intervals"]["calls"]
    if a.get("geometry_narrow", 0) != 2 or b.get("geometry_narrow", 0) != 2:
        raise AssertionError("the declared physical step did not use entry and path narrow queries")
    expected = 2 * left["different_world_rows"]
    actual = a.get("pair_record", 0) - b.get("pair_record", 0)
    if actual != expected:
        raise AssertionError("retained dispatch difference does not match all cross-world rows")
    # With W1 the unchanged _complete_pairs wrapper uses the same full dispatcher;
    # with W32 it also restores all immutable rows in their original global slots.
    if a.get("filtered_completion", 0) != 0 or b.get("filtered_completion", 0) != 2:
        raise AssertionError("the two dispatch controls were not actually exercised")
    equal = (
        "hull_reconstruction", "hull_boundary", "material_chart_build", "material_chart_weights",
        "sphere_hull_projection", "projection_feature_search", "projection_distance_material_finish",
        "generic_simplex_closest", "path_support_build", "path_sample_pose", "path_support_upper",
        "path_sample_dispatch", "path_interval_verify", "path_window_driver", "path_trace_finish",
    )
    for label in equal:
        if a.get(label, 0) != b.get(label, 0):
            raise AssertionError("geometry/proof call count changed: " + label)
    return dict(
        extra_full_dispatch_pair_records=actual,
        expected_cross_world_records=expected,
        filtered_completion_calls={VARIANTS[0]: 0, VARIANTS[1]: 2},
        equal_geometry_call_labels=list(equal),
    )


def run_pair(case, worlds, phase, mode, cache, destination):
    if (case, worlds) not in MATRIX or phase not in ("untraced", "traced"):
        raise ValueError("only the frozen four cases and two declared windows are supported")
    if mode not in ("host", "interpret", "cuda"):
        raise ValueError("unsupported backend")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(
        case=case, worlds=worlds, phase=phase, backend=mode, dt_s=H, frames=list(FRAMES),
        excluded_public_fields=list(PUBLIC_CLOCK_FIELDS),
        counter_clock_fields=list(COUNTER_CLOCK_FIELDS), rows=[], outputs={}, builds={},
        scope="retained full dispatch vs current filtered dispatch in one product revision",
        timing_scope="scene.step plus geometry; observations/oracle/serialization are outside",
        resource_policy="two independent scenes; same code-cache directory; new state/buffers",
        diagnostic_only=phase == "traced",
    )
    writers, scenes = {}, {}
    try:
        with ExitStack() as stack:
            for variant in VARIANTS:
                native_before = driver._raw_counts(mode)
                begin = perf_counter_ns()
                make = driver.builder(ph, case, worlds)
                declared = perf_counter_ns()
                scene = stack.enter_context(make.build(backend=mode, cache_root=cache))
                built = perf_counter_ns()
                scenes[variant] = scene
                report["builds"][variant] = dict(
                    declaration_ns=declared - begin, build_ns=built - declared,
                    native_counters=driver.delta_counts(driver._raw_counts(mode), native_before),
                )
                writers[variant] = stack.enter_context(driver.WindowWriter(
                    destination / f"{variant}.json.gz",
                    dict(case=case, worlds=worlds, phase=phase, variant=variant, dt_s=H,
                         frames=FRAMES, initial=scene.state, rigid_rows=scene.rigid_rows,
                         input_rows=driver.seeds(case, worlds)),
                ))
            assert_full_equal(
                driver.public_value(scenes[VARIANTS[0]].state),
                driver.public_value(scenes[VARIANTS[1]].state), "initial_state"
            )
            spans = None
            if phase == "traced":
                spans = stack.enter_context(gzip.open(
                    destination / "timelines.jsonl.gz", "xt", encoding="utf8"
                ))
            for number in FRAMES:
                row = dict(frame=number, entry_s=(number - 1) * H, end_s=number * H,
                           order=list(pair_order(number)), observations={})
                report["rows"].append(row)
                values = {}
                for variant in pair_order(number):
                    scene = scenes[variant]
                    before, native = scene.stats(), driver._raw_counts(mode)
                    timeline = NarrowTimeline() if phase == "traced" else None
                    with dispatch_variant(variant):
                        context = timeline.active(scene) if timeline is not None else nullcontext()
                        with context:
                            begin = perf_counter_ns()
                            frame = scene.step(H)
                            stepped = perf_counter_ns()
                            geometry = scene.geometry()
                            end = perf_counter_ns()
                    after, native_after = scene.stats(), driver._raw_counts(mode)
                    if geometry.time_s != number * H:
                        raise AssertionError("accepted physical time differs from the declared frame")
                    certify_begin = perf_counter_ns()
                    errors = driver.certify_frame(frame, scene.rigid_rows, case, worlds, number - 1)
                    certify_end = perf_counter_ns()
                    value = canonical_payload(scene, frame, geometry)
                    values[variant] = value
                    digest = writers[variant].append(value)
                    observation = dict(
                        step_ns=stepped - begin, geometry_snapshot_ns=end - stepped,
                        public_to_end_ns=end - begin, oracle_ns=certify_end - certify_begin,
                        archive_and_observation_ns=perf_counter_ns() - certify_end,
                        frame_logical_sha256=digest, physical_errors=errors,
                        statistics_before=before, statistics_after=after,
                        counters=driver.delta_counts(after, before),
                        native_before=native, native_after=native_after,
                        native_counters=driver.delta_counts(native_after, native),
                        **_physical_row(frame),
                    )
                    row["observations"][variant] = observation
                    if mode == "cuda" and (
                        observation["native_counters"].get("cuda_kernel_launches", 0) <= 0
                        or observation["counters"].get("geometry.completed_launches", 0) <= 0
                        or frame.rigid_report.backend != "tyx:cuda"
                    ):
                        raise AssertionError("the CUDA call did not execute the real selected backend")
                    if timeline is not None:
                        intervals = timeline.partition(begin, end)
                        raw_spans = intervals.pop("spans")
                        spans.write(json.dumps(dict(
                            frame=number, variant=variant, spans=raw_spans,
                            source=timeline.source_binding,
                        ), separators=(",", ":"), allow_nan=False) + "\n")
                        observation["diagnostic_intervals"] = intervals
                        observation["projection_source"] = timeline.source_binding
                assert_full_equal(values[VARIANTS[0]], values[VARIANTS[1]])
                left, right = (row["observations"][variant] for variant in VARIANTS)
                _compare_observations(left, right)
                if phase == "traced":
                    row["dispatch_call_check"] = _check_trace_counts(left, right)
                row["all_fields_equal"] = True
            for writer in writers.values():
                writer.close()
            # Only the new small windows are decoded here, once, in the same session.
            for variant, writer in writers.items():
                count = 0
                for count, value in enumerate(driver.read_window(writer.path), start=1):
                    expected = report["rows"][count - 1]["observations"][variant]
                    if driver.logical_digest(value) != expected["frame_logical_sha256"]:
                        raise AssertionError("decoded frame differs from its same-call observation")
                if count != len(FRAMES):
                    raise AssertionError("missing declared frames in the new lossless window")
                report["outputs"][variant] = dict(
                    **writer.stats(), path=str(writer.path.resolve()), decode_verified_frames=count
                )
            report["statistics"] = {
                variant: dict(
                    all_16=driver._distribution([
                        row["observations"][variant]["public_to_end_ns"] for row in report["rows"]
                    ]),
                    first_call_ns=report["rows"][0]["observations"][variant]["public_to_end_ns"],
                    subsequent_15=driver._distribution([
                        row["observations"][variant]["public_to_end_ns"] for row in report["rows"][1:]
                    ]),
                ) for variant in VARIANTS
            }
            report["success"] = True
    except BaseException as error:
        report["failure"] = dict(type=type(error).__name__, message=str(error))
        # The scoped dispatcher, tracing hooks and all scene resources have exited.
        report["dispatch_restored"] = collision_pipeline._complete_pairs is _FILTERED_DISPATCH
        raise
    finally:
        _save(destination / "pair.json", report)
    return report


def compare_windows(untraced, traced):
    for key in ("case", "worlds", "backend", "dt_s", "frames"):
        assert_full_equal(untraced[key], traced[key], "windows." + key)
    for a, b in zip(untraced["rows"], traced["rows"], strict=True):
        for variant in VARIANTS:
            _compare_observations(a["observations"][variant], b["observations"][variant])
    result = {}
    for variant in VARIANTS:
        assert_full_equal(
            untraced["outputs"][variant]["logical_sha256"],
            traced["outputs"][variant]["logical_sha256"], "whole_window_payload"
        )
        a = untraced["statistics"][variant]["all_16"]["total_ns"]
        b = traced["statistics"][variant]["all_16"]["total_ns"]
        ratio = b / a
        result[variant] = dict(
            trace_to_untraced_wall_ratio=ratio,
            attribution_usable=ratio <= MAX_TRACE_TO_UNTRACED_RATIO,
            overhead_limit=MAX_TRACE_TO_UNTRACED_RATIO,
            interpretation="traced spans are diagnostic; never substitute for untraced timings",
        )
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=("host", "interpret", "cuda"), default="cuda")
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--expected-head", required=True)
    args = cli.parse_args()
    if args.out.exists():
        cli.error("use a fresh result directory")
    args.out.mkdir(parents=True)
    source = provenance(args.expected_head)
    report = dict(
        schema="phymeo-c3-loaded-narrow-v1", source=source, matrix=[list(p) for p in MATRIX],
        frame_indices=list(FRAMES), backend=args.backend, windows=[], comparisons=[],
        excluded_public_fields=list(PUBLIC_CLOCK_FIELDS),
        counter_clock_fields=list(COUNTER_CLOCK_FIELDS),
        mechanism="A retained full dispatch; B current filtered dispatch; same source",
        pure_gpu_device_ns=None, max_spans_per_frame=MAX_SPANS_PER_FRAME,
        scope="one bounded session, no claim about another revision or Newton performance",
    )
    try:
        for case, worlds in MATRIX:
            paired = {}
            for phase in ("untraced", "traced"):
                result = run_pair(
                    case, worlds, phase, args.backend, args.out / "cache",
                    args.out / f"{case}-w{worlds}-{phase}",
                )
                paired[phase] = result
                report["windows"].append(result)
                print(json.dumps(dict(case=case, worlds=worlds, phase=phase, success=True)), flush=True)
            report["comparisons"].append(dict(
                case=case, worlds=worlds, same_complete_outputs=True,
                perturbation=compare_windows(paired["untraced"], paired["traced"]),
            ))
        report["source_after"] = provenance(args.expected_head)
        verify_provenance(source, report["source_after"])
        report["success"] = True
    except BaseException as error:
        report["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        report["dispatch_restored"] = collision_pipeline._complete_pairs is _FILTERED_DISPATCH
        report["trace_restored"] = sys.gettrace() is None and sys.getprofile() is None
        _save(args.out / "manifest.json", report)
    print(json.dumps(dict(success=True, pairs=len(MATRIX), public_frames=4 * 16 * 2 * 2)))


if __name__ == "__main__":
    main()
