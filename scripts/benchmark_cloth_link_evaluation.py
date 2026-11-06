"""Uninstrumented full-output A/B of opt-in, per-evaluation cloth/link input reuse.

Only the already measured W1F8 and W32F8 scenes are used. This is not a new GPU
algorithm or a repeat of the five-configuration material-transfer cost survey.
"""

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median
from time import time_ns

ROOT = Path(__file__).resolve().parents[1]
ORDER = {1: ("scalar", "reuse"), 2: ("reuse", "scalar"), 3: ("scalar", "reuse")}
NEW_SOURCE = (
    "scripts/benchmark_cloth_link_evaluation.py",
    "src/phymeo/solvers/_cloth_link_evaluation.py",
    "tests/test_cloth_link_evaluation.py",
    "docs/benchmarks/p65-cloth-link-evaluation.md",
)


def tools():
    name = "_material_phase_transfer_for_local_reuse"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/profile_material_phase_transfer.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module.SOURCE_TARGETS = tuple(dict.fromkeys((*module.SOURCE_TARGETS, *NEW_SOURCE)))
    return sys.modules[name]


def build_worlds(world_count, reuse):
    """Change only the public optimization switch, keeping the accepted frozen data."""
    import phymeo as ph

    base = tools()._cases().build_worlds(world_count, 8, "uniform")
    result = {}
    for key, world in base.items():
        old = world.program
        program = ph.CompliantClothContactProgram(
            old.scene, old.cloth_domain, old.material, old.surfaces, world.reference,
            old.patches, material_damping_s=old.damping,
            max_penetration_m=old.max_penetration_m, robot_contact=old.robot_contact,
            max_joint_violation=old.max_joint_violation,
            cloth_link_material=old.cloth_links.law,
            cloth_link_operator=old.cloth_links.operator,
            cloth_link_local_reuse=reuse,
        )
        result[key] = replace(world, program=program)
    return result


def statistics(windows, worlds):
    values = [window["whole_ns"] for window in windows]
    if len(values) != 3 or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("exactly three complete positive formal wall samples required")
    ordered = sorted(values)
    return dict(whole_ns=values, p50_ms=median(values) / 1e6,
                p95_ms=ordered[math.ceil(.95 * len(ordered)) - 1] / 1e6,
                world_substeps_per_s=worlds * 4 * len(values) * 1e9 / sum(values))


def run(args):
    p = tools()
    source = p.bind_source(ROOT, args.expected_head)
    config = p.ProfileConfig(args.worlds, 8, args.mode, warm_windows=1, formal_windows=3,
                             steps=4, max_seconds=180, max_raw_bytes=32 * 1024 * 1024)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=False)
    start = time_ns()
    budget = p._Budget(config)
    completed, full = [], []
    failure, raw = None, None
    result = dict(status="running", source=source, config=asdict(config),
                  implementation=args.implementation, pair=args.pair,
                  expected_order=list(ORDER[args.pair]), run_start_unix_ns=start,
                  command=list(sys.argv), cache_root=str(args.cache_root),
                  raw_frames_path=str(args.raw_path), default_changed=False,
                  scope="HOST local input reuse with unchanged TYX law; complete public outputs")

    def preserve(label, index, window):
        payload = dict(phase=label, index=index, window=window)
        budget.reserve(payload)
        raw.write(payload)
        summary = dict(phase=label, index=index, **p._compact_window(window))
        completed.append(summary)
        print(f"{args.implementation} W{args.worlds} {label} {index}: {raw.json_bytes} JSON bytes", flush=True)
        return summary

    with args.out.open("x", encoding="utf8", newline="\n") as output:
        try:
            with p.RawArchive(args.raw_path, max_bytes=config.max_raw_bytes) as raw:
                try:
                    import phymeo as ph
    
                    p._check_loaded_sources(ROOT)
                    worlds = build_worlds(config.worlds, args.implementation == "reuse")
                    manifest = p._input_manifest(worlds, config)
                    manifest_sha = p._digest(manifest)
                    preserve("input-manifest", 0, dict(input_manifest=manifest, input_manifest_sha256=manifest_sha))
                    settings = ph.StepSettings(substeps=1, position_iterations=1,
                                               velocity_iterations=1, record_substeps=True)
                    with ph.TyxMultifaceContactBatch(
                        [(key, ph.ClothLinkWorld(world.program, world.initial)) for key, world in worlds.items()],
                        args.mode, settings=settings, cache_root=args.cache_root,
                    ) as batch:
                        for label, count in (("cold", 1), ("warm", 1), ("formal", 3)):
                            for index in range(count):
                                window = p._measure_window(batch, worlds, config, instrumented=False,
                                                           cold=label == "cold", budget=budget)
                                full.append(preserve(label, index, window))
                        actual_device = p._device_signature(batch._layer, args.mode)
                    if p._digest(p._input_manifest(worlds, config)) != manifest_sha:
                        raise ValueError("physical inputs changed during the uninstrumented run")
                    expected = full[0]["complete_output_sha256"]
                    if any(window["complete_output_sha256"] != expected for window in full):
                        raise ValueError("reset windows have different complete physical outputs")
                    p.verify_source(source)
                    budget.time()
                    result.update(
                        status="complete", input_manifest_sha256=manifest_sha,
                        complete_output_sha256=expected, actual_device=actual_device,
                        source_verified_after=True, windows=full,
                        formal=statistics([w for w in full if w["phase"] == "formal"], args.worlds),
                        loaded_modules=p._check_loaded_sources(ROOT),
                        canonical_normalization="only report.substeps[*].timings values, keys retained",
                        timing_scope="uninstrumented four-step public task; reset, audits, JSON and IO outside",
                    )
                except BaseException as error:
                    failed = getattr(error, "profile_window", None)
                    if failed is not None:
                        try:
                            preserve("failed-attempt", 0, failed)
                        except BaseException as archival_error:
                            error.add_note(f"failed-attempt archive unavailable: {archival_error}")
                    raise
        except BaseException as error:
            failure = error
            window = getattr(error, "profile_window", None)
            # Complete failure data was attempted before closing the bounded gzip.
            result.update(status="failed", error_type=type(error).__name__, error=str(error),
                          completed_windows=completed, error_notes=list(getattr(error, '__notes__', ())),
                          failed_attempt_summary=None if window is None else p._compact_window(window))
            try:
                p.verify_source(source)
                result["source_verified_after"] = True
            except BaseException as verification:
                result["source_verified_after"] = False
                result["source_verification_error"] = str(verification)
        finally:
            result["run_end_unix_ns"] = time_ns()
            if args.raw_path.is_file():
                result["raw_frames_sha256"] = p._sha_file(args.raw_path)
                result["raw_compressed_bytes"] = args.raw_path.stat().st_size
            if raw is not None:
                result["raw_json_bytes"], result["raw_complete_records"] = raw.json_bytes, raw.records
            result["storage_cap"] = dict(raw_json_bytes=config.max_raw_bytes,
                                          summary_json_bytes=p.MAX_SUMMARY_BYTES)
            encoded = p._encoded(result)
            if len(encoded) > p.MAX_SUMMARY_BYTES:
                failure = p.ResourceLimit("summary cap exceeded; complete raw sidecar retained")
                result = dict(status="failed", error=str(failure), source_head=source["head"],
                              raw_frames_path=str(args.raw_path),
                              raw_frames_sha256=p._sha_file(args.raw_path))
                encoded = p._encoded(result)
            output.write(encoded.decode("utf8") + "\n")
    if failure is not None:
        raise failure
    print(json.dumps(dict(output=str(args.out), implementation=args.implementation,
                          formal=result["formal"], device=result["actual_device"]), allow_nan=False))


def compare_profiles(profiles, *, expected_head):
    """Read saved public-output/counter receipts; never infer gains from different tasks."""
    groups, by_world = {}, {}
    intervals = []
    for profile in profiles:
        config = profile["config"]
        if (profile["status"] != "complete" or profile["source"]["head"] != expected_head
                or profile.get("source_verified_after") is not True
                or (config["faces"], config["steps"], config["warm_windows"], config["formal_windows"])
                != (8, 4, 1, 3)):
            raise ValueError("incomplete, changed-source or mismatched protocol receipt")
        world, pair, implementation = config["worlds"], profile["pair"], profile["implementation"]
        if world not in (1, 32) or pair not in ORDER or implementation not in ORDER[pair]:
            raise ValueError("unexpected comparison lane")
        signature = (config, profile["input_manifest_sha256"], profile["complete_output_sha256"],
                     profile["actual_device"], profile["source"]["files"])
        if world in by_world and by_world[world] != signature:
            raise ValueError("independent repeats differ in source, physical task, output or device")
        by_world[world] = signature
        intervals.append((profile["run_start_unix_ns"], profile["run_end_unix_ns"]))
        key = world, pair
        lane = groups.setdefault(key, {})
        if implementation in lane:
            raise ValueError("duplicate comparison lane")
        lane[implementation] = profile
    if set(groups) != {(world, pair) for world in (1, 32) for pair in ORDER}:
        raise ValueError("both worlds and all three independent pairs are required")
    ordered_intervals = sorted(intervals)
    if any(a >= b for a, b in ordered_intervals) or any(
            first[1] > last[0] for first, last in zip(ordered_intervals, ordered_intervals[1:])):
        raise ValueError("formal processes overlap or have invalid timing intervals")
    result = []
    for (world, pair), lanes in sorted(groups.items()):
        if set(lanes) != {"scalar", "reuse"}:
            raise ValueError("missing paired implementation")
        scalar, reuse = lanes["scalar"], lanes["reuse"]
        first, last = (lanes[name] for name in ORDER[pair])
        if not (first["run_start_unix_ns"] < first["run_end_unix_ns"]
                <= last["run_start_unix_ns"] < last["run_end_unix_ns"]):
            raise ValueError("AB/BA process order or exclusive execution is not evidenced")
        for field in ("input_manifest_sha256", "complete_output_sha256", "actual_device"):
            if scalar[field] != reuse[field]:
                raise ValueError(f"paired {field} differs")
        if (scalar["config"]["mode"] != "cuda" or reuse["config"]["mode"] != "cuda"
                or scalar["actual_device"]["real_cuda"] is not True):
            raise ValueError("formal comparison requires actual CUDA on both lanes")
        if len(scalar["windows"]) != 5 or len(reuse["windows"]) != 5:
            raise ValueError("missing complete windows")
        for a, b in zip(scalar["windows"], reuse["windows"], strict=True):
            for field in ("phase", "index", "complete_output_sha256", "counter_delta", "phase_sites"):
                if a[field] != b[field]:
                    raise ValueError(f"paired full-output/counter field {field} differs")
            if (a["counter_audit"] != "passed" or b["counter_audit"] != "passed"
                    or a["accepted_world_steps"] != 4 * world or b["accepted_world_steps"] != 4 * world
                    or a["instrumented"] is not False or b["instrumented"] is not False):
                raise ValueError("incomplete or instrumented timing lane")
        old = statistics([w for w in scalar["windows"] if w["phase"] == "formal"], world)
        new = statistics([w for w in reuse["windows"] if w["phase"] == "formal"], world)
        result.append(dict(worlds=world, pair=pair, order=list(ORDER[pair]),
                           scalar=old, reuse=new, p50_ratio=old["p50_ms"] / new["p50_ms"],
                           p95_ratio=old["p95_ms"] / new["p95_ms"],
                           complete_output_sha256=scalar["complete_output_sha256"],
                           actual_device=scalar["actual_device"]))
    return dict(status="complete", source_head=expected_head, pairs=result,
                default_changed=False, scope="same full physical application, local-reuse switch only")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--worlds", type=int, choices=(1, 32), required=True)
    execute.add_argument("--implementation", choices=("scalar", "reuse"), required=True)
    execute.add_argument("--mode", choices=("cuda", "interpret"), default="cuda")
    execute.add_argument("--pair", type=int, choices=(1, 2, 3), required=True)
    execute.add_argument("--cache-root", type=Path)
    summary = commands.add_parser("summarize")
    summary.add_argument("--profiles", nargs=12, type=Path, required=True)
    for sub in (execute, summary):
        sub.add_argument("--expected-head", required=True)
        sub.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.expected_head) != 40 or any(ch not in "0123456789abcdef" for ch in args.expected_head):
        raise ValueError("full lowercase source HEAD required")
    if not args.out.is_absolute():
        raise ValueError("absolute fresh output required")
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.command == "run":
        args.cache_root = args.cache_root or args.out.with_suffix(".cache")
        args.raw_path = args.out.with_suffix(".frames.jsonl.gz")
        paths = (args.out, args.cache_root, args.raw_path)
        if any(not path.is_absolute() for path in paths) or len(set(paths)) != 3:
            raise ValueError("distinct absolute output/cache/sidecar paths required")
        if any(path.exists() for path in paths):
            raise FileExistsError("fresh lane paths required")
    elif any(not path.is_absolute() or not path.is_file() for path in args.profiles):
        raise ValueError("existing absolute profile paths required")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "run":
        run(args)
    else:
        p = tools()
        profiles = [json.loads(path.read_text(encoding="utf8")) for path in args.profiles]
        result = compare_profiles(profiles, expected_head=args.expected_head)
        result["profiles"] = [dict(path=str(path), sha256=p._sha_file(path)) for path in args.profiles]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf8") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
        print(json.dumps(dict(output=str(args.out), pairs=len(result["pairs"])), allow_nan=False))


if __name__ == "__main__":
    main()
