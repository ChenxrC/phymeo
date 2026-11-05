"""Compare the frozen nine C1 runs without dropping public fields or work counters."""

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import math
import sys
from contextlib import closing
from itertools import zip_longest
from pathlib import Path

_TOOL = Path(__file__).with_name("benchmark_scene_scale_costs.py")
_SPEC = importlib.util.spec_from_file_location("_c2_original_c1_codec", _TOOL)
driver = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = driver
_SPEC.loader.exec_module(driver)

ROW_TIMINGS = ("step_ns", "geometry_snapshot_ns", "public_to_end_ns", "oracle_ns", "archive_ns")
COUNTER_TIMINGS = (
    "geometry.envelope_ns",
    "geometry.broad_ns",
    "geometry.midphase_ns",
    "geometry.narrow_ns",
)
DISTRIBUTION_TIMINGS = ("total_ns", "p50_ns", "p95_ns", "minimum_ns", "maximum_ns")
PRODUCT_CHANGE = "src/phymeo/geometry/collision_pipeline.py"
TOOL_SUFFIXES = {
    "benchmark": "scripts/benchmark_scene_scale_costs.py",
    "profile": "scripts/profile_scene_scale_costs.py",
    "cases": "tests/scale/scene_scale_cases.py",
}


def compare_values(before, after, path="$"):
    """Strict structural/type/value comparison, including unknown or newly added fields."""
    if type(before) is not type(after):
        raise AssertionError(f"{path}: type changed")
    if isinstance(before, dict):
        if before.keys() != after.keys():
            raise AssertionError(
                f"{path}: fields changed; missing={sorted(before.keys() - after.keys())}, "
                f"added={sorted(after.keys() - before.keys())}"
            )
        for key in before:
            compare_values(before[key], after[key], f"{path}.{key}")
    elif isinstance(before, list):
        if len(before) != len(after):
            raise AssertionError(f"{path}: length changed")
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            compare_values(left, right, f"{path}[{index}]")
    elif before != after:
        raise AssertionError(f"{path}: value changed ({str(before)[:100]} -> {str(after)[:100]})")


def _zero_duration(mapping, key, path):
    if key not in mapping:
        raise AssertionError(f"{path}.{key}: missing wall-clock field")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(f"{path}.{key}: nonnumeric wall clock")
    if not math.isfinite(value) or value < 0:
        raise AssertionError(f"{path}.{key}: invalid wall clock")
    mapping[key] = 0


def normalized_step(row):
    """Retain every key; normalize exactly five row clocks and four geometry clocks."""
    view = copy.deepcopy(row)
    for key in ROW_TIMINGS:
        _zero_duration(view, key, "row")
    for key in COUNTER_TIMINGS:
        _zero_duration(view["counters"], key, "row.counters")
    return view


def compare_step_rows(before, after):
    compare_values(normalized_step(before), normalized_step(after), "step")


def _checked_manifest(path):
    data = json.loads(path.read_text(encoding="utf8"))
    if (
        data.get("schema") != driver.SCHEMA
        or data.get("success") is not True
        or data.get("backend") != "cuda"
        or data.get("steps_per_window") != driver.STEPS
        or data.get("matrix") != [list(pair) for pair in driver.MATRIX]
        or data.get("h_s") != driver.H
    ):
        raise AssertionError(f"{path}: not a complete frozen C1 CUDA run")
    if [window["phase"] for window in data["windows"]] != list(driver.WINDOWS):
        raise AssertionError(f"{path}: cold/warm windows changed")
    driver.verify_provenance(data["source"], data["source_after"])
    for window in data["windows"]:
        rows = window["rows"]
        if window["successful_steps"] != driver.STEPS or len(rows) != driver.STEPS:
            raise AssertionError(f"{path}: missing physical steps")
        for step, row in enumerate(rows, 1):
            if (
                row["step"] != step
                or row["entry_s"] != (step - 1) * driver.H
                or row["end_s"] != step * driver.H
                or row["public_to_end_ns"] != row["step_ns"] + row["geometry_snapshot_ns"]
            ):
                raise AssertionError(f"{path}: physical/timing sequence changed at step {step}")
            normalized_step(row)
        for key, sample in (
            ("public_statistics", "public_to_end_ns"),
            ("step_statistics", "step_ns"),
        ):
            compare_values(
                window[key], driver._distribution([row[sample] for row in rows]), str(path) + key
            )
    return data


def _check_sources(before, after):
    """Allow the declared C2 product file, but no changed physics/input/measurement tool."""
    for side in (before, after):
        for stage in ("source", "source_after"):
            receipt = side[stage]
            if receipt["product_dirty"] is not False:
                raise AssertionError("dirty measured product")
            if set(receipt["tools"]) != set(TOOL_SUFFIXES):
                raise AssertionError("missing original tool binding")
            for name, suffix in TOOL_SUFFIXES.items():
                item = receipt["tools"][name]
                if not item["path"].replace("\\", "/").endswith("/" + suffix):
                    raise AssertionError("unexpected original tool path: " + name)
            if receipt["driver_sha256"] != receipt["tools"]["benchmark"]["sha256"]:
                raise AssertionError("inconsistent original driver binding")
    for stage in ("source", "source_after"):
        left, right = before[stage], after[stage]
        if left.keys() != right.keys():
            raise AssertionError("source receipt fields changed")
        for name in TOOL_SUFFIXES:
            if left["tools"][name]["sha256"] != right["tools"][name]["sha256"]:
                raise AssertionError("original measurement tool changed: " + name)
    left = {
        key.replace("\\", "/"): value
        for key, value in before["source_after"]["imported_files"].items()
    }
    right = {
        key.replace("\\", "/"): value
        for key, value in after["source_after"]["imported_files"].items()
    }
    if left.keys() != right.keys():
        raise AssertionError("imported product set changed")
    changed = [key for key in left if left[key] != right[key]]
    if set(changed) - {PRODUCT_CHANGE}:
        raise AssertionError("out-of-scope product changes: " + repr(changed))
    return changed


def _manifest_view(data):
    view = copy.deepcopy(data)
    # These execution/source receipts are validated separately, never treated as physics.
    for key in ("source", "source_after"):
        view[key] = "validated separately; full receipts retained in comparison"
    for key in ("started_utc", "ended_utc"):
        if not isinstance(view[key], str):
            raise AssertionError("invalid UTC wall clock")
        view[key] = "wall clock"
    _zero_duration(view, "declaration_ns", "manifest")
    for window in view["windows"]:
        _zero_duration(window, "build_ns", "window")
        window["rows"] = [normalized_step(row) for row in window["rows"]]
        for name in ("public_statistics", "step_statistics"):
            for key in DISTRIBUTION_TIMINGS:
                _zero_duration(window[name], key, name)
    return view


def _archive_header(path, window):
    """Read input metadata only; accepted full-payload hashes are reused, not recomputed."""
    output = window["output"]
    if output["frames"] != driver.STEPS or path.stat().st_size != output["gzip_bytes"]:
        raise AssertionError(f"{path}: archive size/frame binding changed")
    with gzip.open(path, "rt", encoding="utf8") as stream:
        header = json.loads(next(stream))
    if header.get("schema") != driver.SCHEMA:
        raise AssertionError(f"{path}: wrong archive schema")
    return header


def _locate_payload_difference(first, second, old, new, target):
    """Only a new digest mismatch justifies decoding an accepted old payload again."""
    with (
        closing(driver.read_window(first)) as left_stream,
        closing(driver.read_window(second)) as right_stream,
    ):
        missing = object()
        for index, (left, right) in enumerate(
            zip_longest(left_stream, right_stream, fillvalue=missing)
        ):
            if left is missing or right is missing:
                raise AssertionError("complete public frame count changed")
            if index != target:
                continue
            for value, window, side in ((left, old, "before"), (right, new, "after")):
                if driver.logical_digest(value) != window["rows"][index]["frame_logical_sha256"]:
                    raise AssertionError(f"{side}: archive/manifest frame binding changed")
            compare_values(left, right, f"{old['phase']}.frame[{index}]")
            raise AssertionError("different complete payload digests despite equal decoded values")
    raise AssertionError("missing mismatching public frame")


def compare_case(before_directory, after_directory):
    before_directory, after_directory = Path(before_directory), Path(after_directory)
    before = _checked_manifest(before_directory / "manifest.json")
    after = _checked_manifest(after_directory / "manifest.json")
    changed = _check_sources(before, after)
    windows = []
    for old, new in zip(before["windows"], after["windows"], strict=True):
        phase = old["phase"]
        if (
            old["output"]["file"] != phase + ".json.gz"
            or new["output"]["file"] != phase + ".json.gz"
        ):
            raise AssertionError("unexpected archive path")
        first = before_directory / old["output"]["file"]
        second = after_directory / new["output"]["file"]
        compare_values(_archive_header(first, old), _archive_header(second, new), "archive.header")
        mismatching = next(
            (
                index
                for index, (left, right) in enumerate(zip(old["rows"], new["rows"], strict=True))
                if left["frame_logical_sha256"] != right["frame_logical_sha256"]
            ),
            None,
        )
        if mismatching is not None:
            _locate_payload_difference(first, second, old, new, mismatching)
        if old["output"]["logical_sha256"] != new["output"]["logical_sha256"]:
            raise AssertionError("whole logical digest changed despite equal per-frame digests")
        windows.append(
            dict(
                phase=phase,
                compared_frames=driver.STEPS,
                full_public_fields_exact=True,
                excluded_public_fields=[],
                extra_full_frames_decoded=0,
                evidence="accepted same-run full decode; equal per-frame and whole logical SHA",
                before_public=old["public_statistics"],
                after_public=new["public_statistics"],
                before_build_ns=old["build_ns"],
                after_build_ns=new["build_ns"],
            )
        )
    compare_values(_manifest_view(before), _manifest_view(after), "manifest")
    return dict(
        case=before["case"],
        worlds=before["worlds"],
        success=True,
        changed_product_files=changed,
        before_source=before["source"],
        before_source_after=before["source_after"],
        after_source=after["source"],
        after_source_after=after["source_after"],
        windows=windows,
    )


def compare_matrix(before, after):
    before, after = Path(before), Path(after)
    rows = []
    source_heads = [set(), set()]
    for case, worlds in driver.MATRIX:
        directory = f"matrix-{case}-w{worlds}"
        row = compare_case(before / directory, after / directory)
        if (row["case"], row["worlds"]) != (case, worlds):
            raise AssertionError("matrix directory differs from measured input")
        source_heads[0].add(row["before_source"]["source_head"])
        source_heads[1].add(row["after_source"]["source_head"])
        rows.append(row)
    if any(len(heads) != 1 for heads in source_heads):
        raise AssertionError("mixed product heads in the fixed matrix")
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--before", required=True, type=Path)
    cli.add_argument("--after", required=True, type=Path)
    cli.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    if args.out.exists():
        cli.error("use a fresh comparison output file")
    script = Path(__file__).resolve()
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    report = dict(
        schema="phymeo-c2-full-frame-comparison-v1",
        comparator=dict(path=str(script), sha256=digest),
        before=str(args.before.resolve()),
        after=str(args.after.resolve()),
        excluded_public_fields=[],
        separate_wall_clocks=dict(
            row=list(ROW_TIMINGS),
            counters=list(COUNTER_TIMINGS),
            manifest=["declaration_ns", "started_utc", "ended_utc"],
            window=["build_ns", "public_statistics", "step_statistics"],
            distribution_values=list(DISTRIBUTION_TIMINGS),
        ),
        success=False,
    )
    try:
        report["matrix"] = compare_matrix(args.before, args.after)
        if hashlib.sha256(script.read_bytes()).hexdigest() != digest:
            raise AssertionError("comparison script changed while reading the receipts")
        report["success"] = True
    except BaseException as error:
        report["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        with args.out.open("x", encoding="utf8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(success=True, cases=len(report["matrix"]), frames=9 * 2 * driver.STEPS)))


if __name__ == "__main__":
    main()
