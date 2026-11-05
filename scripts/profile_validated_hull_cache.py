"""C4 bounded same-session public loaded-scene comparison, with complete outputs."""

import argparse
import hashlib
import importlib.util
import json
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter_ns

import phymeo as ph
from phymeo.geometry import _convex_hull
from phymeo.geometry._validated_hull_cache import ValidatedHullCache

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_c4_original_scale_driver", ROOT / "scripts" / "benchmark_scene_scale_costs.py"
)
driver = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = driver
_SPEC.loader.exec_module(driver)

VARIANTS = ("uncached", "cached")
FRAMES = (1, 2, 3)
FORMAL_FRAMES = (2, 3)
CLOCK_FIELDS = (
    "geometry.envelope_ns",
    "geometry.broad_ns",
    "geometry.midphase_ns",
    "geometry.narrow_ns",
)
CACHE_FIELDS = (
    "geometry.validated_hull_attempts",
    "geometry.validated_hull_builds",
    "geometry.validated_hull_hits",
)
_GET = ValidatedHullCache.get
_LOCK = Lock()
_ACTIVE = None


def _tracked_get(self, shape, path, builder=None):
    if _ACTIVE is None:
        raise RuntimeError("missing active cache comparison")
    _ACTIVE["get_calls"] += 1
    if _ACTIVE["variant"] == "cached":
        return _GET(self, shape, path, builder)
    # Retain the original construction, with the same live constructor passed
    # by the product hook. Do not weaken a geometry check or change the dispatcher.
    return (_convex_hull.HullShape if builder is None else builder)(shape, path)


@contextmanager
def cache_variant(variant):
    """Serial diagnostic context only; both variants incur the same callback counter."""
    global _ACTIVE
    if variant not in VARIANTS:
        raise ValueError("unknown cache variant")
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("a cache diagnostic is already active")
    original = ValidatedHullCache.get
    try:
        if original is not _GET:
            raise RuntimeError("another cache replacement is active")
        _ACTIVE = dict(variant=variant, get_calls=0)
        ValidatedHullCache.get = _tracked_get
        yield _ACTIVE
    finally:
        ValidatedHullCache.get = original
        _ACTIVE = None
        _LOCK.release()


def assert_payload_equal(left, right):
    # JSON bytes distinguish signed zero; the shared codec encodes all array
    # bytes, dataclass fields, Fractions, tuple/list identities and material IDs.
    if driver.logical_bytes(left) != driver.logical_bytes(right):
        raise AssertionError("complete public payload differs")


def compare_counters(left, right, *, worlds, number):
    a, b = dict(left), dict(right)
    if a.keys() != b.keys():
        raise AssertionError("public statistics keys differ")
    for key in CLOCK_FIELDS:
        for values in (a, b):
            if key not in values or values[key] < 0:
                raise AssertionError("missing or invalid declared clock: " + key)
            values[key] = 0
    expected = (worlds, worlds, 2 * worlds) if number == 1 else (0, 0, 3 * worlds)
    for key, count in zip(CACHE_FIELDS, expected, strict=True):
        if a.get(key) != 0 or b.get(key) != count:
            raise AssertionError("unexpected cache mechanism count: " + key)
        a[key] = b[key] = 0
    if a != b:
        raise AssertionError("non-clock non-cache statistics differ")
    return dict(
        uncached={k: left[k] for k in CACHE_FIELDS}, cached={k: right[k] for k in CACHE_FIELDS}
    )


def provenance(expected_head):
    result = driver.provenance(expected_head)
    result["c4_tools"] = {
        label: dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for label, path in (
            ("profile", Path(__file__)),
            ("tests", ROOT / "tests" / "collision" / "test_validated_hull_cache.py"),
            ("cache", ROOT / "src" / "phymeo" / "geometry" / "_validated_hull_cache.py"),
        )
    }
    return result


def verify_provenance(before, after):
    driver.verify_provenance(before, after)
    if set(before.get("c4_tools", {})) != {"profile", "tests", "cache"} or before[
        "c4_tools"
    ] != after.get("c4_tools"):
        raise AssertionError("C4 bound source changed")


def run_pair(mode, worlds, destination):
    if mode not in ("host", "interpret", "cuda") or worlds not in (1, 32):
        raise ValueError("only HOST/interpret/CUDA loaded W1/W32 are declared")
    destination = Path(destination)
    report = dict(
        backend=mode,
        worlds=worlds,
        case="loaded",
        dt_s=driver.H,
        frames=list(FRAMES),
        warm_frame=1,
        formal_frames=list(FORMAL_FRAMES),
        public_exclusions=[],
        counter_clocks=list(CLOCK_FIELDS),
        differing_mechanism_counters=list(CACHE_FIELDS),
        rows=[],
        builds={},
        outputs={},
        timing_scope="scene.step through geometry readback; oracle/history/events/encoding outside",
        physical_scope="original frozen 4 N supported slide; original C1 scalar oracle",
        pure_gpu_device_ns=None,
    )
    start_window = perf_counter_ns()
    writers = {}
    try:
        with ExitStack() as stack:
            scenes = {}
            for variant in VARIANTS:
                begin = perf_counter_ns()
                make = driver.builder(ph, "loaded", worlds)
                declaration = perf_counter_ns()
                scene = stack.enter_context(
                    make.build(backend=mode, cache_root=destination / "cache")
                )
                report["builds"][variant] = dict(
                    declaration_ns=declaration - begin, build_ns=perf_counter_ns() - declaration
                )
                scenes[variant] = scene
                writers[variant] = stack.enter_context(
                    driver.WindowWriter(
                        destination / f"{variant}.json.gz",
                        dict(
                            variant=variant,
                            case="loaded",
                            worlds=worlds,
                            frames=FRAMES,
                            initial=scene.state,
                            rigid_rows=scene.rigid_rows,
                            inputs=driver.seeds("loaded", worlds),
                        ),
                    )
                )
            assert_payload_equal(
                driver.public_value(scenes["uncached"].state),
                driver.public_value(scenes["cached"].state),
            )
            for number in FRAMES:
                # The two formal observations have opposite ordering; the first
                # physical step is retained as the explicitly separate warm call.
                order = VARIANTS if number % 2 else VARIANTS[::-1]
                row = dict(
                    frame=number, formal=number in FORMAL_FRAMES, order=list(order), observations={}
                )
                report["rows"].append(row)
                payloads = {}
                for variant in order:
                    scene = scenes[variant]
                    before, native = scene.stats(), driver._raw_counts(mode)
                    with cache_variant(variant) as mechanism:
                        begin = perf_counter_ns()
                        frame = scene.step(driver.H)
                        stepped = perf_counter_ns()
                        geometry = scene.geometry()
                        end = perf_counter_ns()
                    after, native_after = scene.stats(), driver._raw_counts(mode)
                    if geometry.time_s != number * driver.H:
                        raise AssertionError("accepted time differs from fixed frame")
                    errors = driver.certify_frame(
                        frame, scene.rigid_rows, "loaded", worlds, number - 1
                    )
                    value = driver.public_value(
                        dict(
                            frame=frame,
                            geometry=geometry,
                            contact_history=scene.contact_history,
                            contact_events=scene.contact_events,
                        )
                    )
                    payloads[variant] = value
                    row["observations"][variant] = dict(
                        step_ns=stepped - begin,
                        geometry_ns=end - stepped,
                        public_to_end_ns=end - begin,
                        counters=driver.delta_counts(after, before),
                        native_counters=driver.delta_counts(native_after, native),
                        statistics_before=before,
                        statistics_after=after,
                        native_before=native,
                        native_after=native_after,
                        get_calls=mechanism["get_calls"],
                        errors=errors,
                        actual_contacts=len(frame.loads.contacts),
                        actual_sites=sum(len(c.sites) for c in frame.loads.contacts),
                        published_pairs=len(frame.path.pairs),
                        required_candidates=frame.path.required_candidates,
                        logical_sha256=writers[variant].append(value),
                    )
                    observation = row["observations"][variant]
                    if mechanism["get_calls"] != 3 * worlds:
                        raise AssertionError(
                            "actual geometry call count differs from frozen workload"
                        )
                    if mode == "cuda" and (
                        observation["native_counters"].get("cuda_kernel_launches", 0) <= 0
                        or observation["counters"].get("geometry.completed_launches", 0) <= 0
                        or frame.rigid_report.backend != "tyx:cuda"
                    ):
                        raise AssertionError("real selected CUDA execution was not observed")
                assert_payload_equal(payloads["uncached"], payloads["cached"])
                left, right = (row["observations"][v] for v in VARIANTS)
                row["cache_mechanism"] = compare_counters(
                    left["counters"], right["counters"], worlds=worlds, number=number
                )
                if left["native_counters"] != right["native_counters"]:
                    raise AssertionError("native kernels, bytes or allocations differ")
                row["all_public_fields_equal"] = True
            for variant, writer in writers.items():
                writer.close()
                count = 0
                for count, value in enumerate(driver.read_window(writer.path), start=1):
                    if (
                        driver.logical_digest(value)
                        != report["rows"][count - 1]["observations"][variant]["logical_sha256"]
                    ):
                        raise AssertionError("same-session lossless output differs")
                if count != len(FRAMES):
                    raise AssertionError("missing declared frame")
                report["outputs"][variant] = dict(**writer.stats(), decoded_frames=count)
            report["statistics"] = {
                v: dict(
                    warm_ns=report["rows"][0]["observations"][v]["public_to_end_ns"],
                    two_formal=driver._distribution(
                        [
                            row["observations"][v]["public_to_end_ns"]
                            for row in report["rows"]
                            if row["formal"]
                        ]
                    ),
                )
                for v in VARIANTS
            }
            report["success"] = True
    except BaseException as error:
        report["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        report["whole_pair_ns"] = perf_counter_ns() - start_window
        report["host_budget_s"] = 20.0 if mode == "host" else None
        report["host_budget_exceeded"] = mode == "host" and report["whole_pair_ns"] > 20_000_000_000
        report["getter_restored"] = ValidatedHullCache.get is _GET
        (destination / "pair.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
    return report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=("host", "interpret", "cuda"), required=True)
    cli.add_argument("--worlds", type=int, choices=(1, 32), required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--expected-head", required=True)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(
        schema="phymeo-c4-exact-hull-cache-v1",
        scope="same-source internal comparison, not Newton or an old binary",
    )
    try:
        report["source"] = provenance(args.expected_head)
        report["pair"] = run_pair(args.backend, args.worlds, args.out)
        report["source_after"] = provenance(args.expected_head)
        verify_provenance(report["source"], report["source_after"])
        report["success"] = True
    except BaseException as error:
        report["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        report["getter_restored"] = ValidatedHullCache.get is _GET
        (args.out / "manifest.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
    print(
        json.dumps(dict(success=True, worlds=args.worlds, backend=args.backend, public_frames=6)),
        flush=True,
    )


if __name__ == "__main__":
    main()
