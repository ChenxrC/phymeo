"""Fixed C1 public-scene costs; full outputs, independent physics and reversible receipts."""

import argparse
import base64
import copy
import gzip
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

import numpy as np

import phymeo as ph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "scale"))
from scene_scale_cases import (  # noqa: E402
    STEPS,
    H,
    builder,
    certify_frame,
    matrix,
    seeds,
)

MATRIX = matrix()
WINDOWS = ("cold", "warm")
SCHEMA = "phymeo-c1-scale-cost-v1"


def public_value(value):
    """Encode ALL dataclass fields; do not silently omit an unknown field or type."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return dict(
            __array__=base64.b64encode(array.tobytes()).decode("ascii"),
            dtype=array.dtype.str,
            shape=list(array.shape),
        )
    if isinstance(value, np.generic):
        return public_value(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": type(value).__module__ + "." + type(value).__qualname__,
            **{field.name: public_value(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, Fraction):
        return {"__fraction__": [value.numerator, value.denominator]}
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: public_value(item) for key, item in value.items()}
        return {"__mapping__": [[public_value(k), public_value(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"__tuple__": [public_value(item) for item in value]}
    if isinstance(value, list):
        return [public_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"__float__": "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unrecorded public type: {type(value).__module__}.{type(value).__qualname__}")


def logical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def logical_digest(value):
    return hashlib.sha256(logical_bytes(value)).hexdigest()


def _difference(old, new):
    if old == new:
        return None
    if isinstance(old, dict) and isinstance(new, dict):
        changes = []
        for key, value in new.items():
            delta = {"set": value} if key not in old else _difference(old[key], value)
            if delta is not None:
                changes.append([key, delta])
        return {"dict": changes, "delete": sorted(set(old) - set(new))}
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        changed = []
        for index, (a, b) in enumerate(zip(old, new, strict=True)):
            delta = _difference(a, b)
            if delta is not None:
                changed.append([index, delta])
        return {"list": changed}
    return {"set": new}


def _apply(old, delta):
    if delta is None:
        return old
    if "set" in delta:
        return copy.deepcopy(delta["set"])
    if "dict" in delta:
        for key in delta["delete"]:
            del old[key]
        for key, change in delta["dict"]:
            old[key] = _apply(old.get(key), change)
        return old
    if "list" in delta:
        for index, change in delta["list"]:
            old[index] = _apply(old[index], change)
        return old
    raise ValueError("unknown receipt patch")


class WindowWriter:
    """Stream complete logical frames; unchanged model/pair records are references, not omissions."""

    def __init__(self, path, metadata):
        self.path = Path(path)
        self.stream = gzip.open(self.path, "xt", encoding="utf8")
        self.previous = None
        self.count = self.logical_bytes = self.stored_json_bytes = 0
        self.digest = hashlib.sha256()
        self._line(dict(schema=SCHEMA, metadata=public_value(metadata)))

    def _line(self, value):
        encoded = logical_bytes(value)
        self.stream.write(encoded.decode("utf8") + "\n")
        self.stored_json_bytes += len(encoded) + 1

    def append(self, value):
        encoded = logical_bytes(value)
        check = hashlib.sha256(encoded).hexdigest()
        self.digest.update(encoded)
        delta = {"set": value} if self.previous is None else _difference(self.previous, value)
        self._line(dict(frame=self.count, logical_sha256=check, delta=delta))
        self.previous = value
        self.count += 1
        self.logical_bytes += len(encoded)
        return check

    def close(self):
        if not self.stream.closed:
            self._line(dict(frames=self.count, window_logical_sha256=self.digest.hexdigest()))
            self.stream.close()

    def stats(self):
        return dict(
            file=self.path.name,
            frames=self.count,
            logical_sha256=self.digest.hexdigest(),
            logical_uncompressed_bytes=self.logical_bytes,
            delta_json_bytes=self.stored_json_bytes,
            gzip_bytes=self.path.stat().st_size,
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read_window(path):
    """Reconstruct and independently verify every field/digest, yielding owned logical records."""
    with gzip.open(path, "rt", encoding="utf8") as stream:
        header = json.loads(next(stream))
        if header.get("schema") != SCHEMA:
            raise ValueError("unexpected C1 receipt schema")
        current, count, digest = None, 0, hashlib.sha256()
        complete = False
        for text in stream:
            row = json.loads(text)
            if "frames" in row:
                if row["frames"] != count or row["window_logical_sha256"] != digest.hexdigest():
                    raise AssertionError("window manifest digest mismatch")
                complete = True
                if stream.read():
                    raise AssertionError("extra data after receipt manifest")
                break
            if row["frame"] != count:
                raise AssertionError("receipt frame order changed")
            current = _apply(current, row["delta"])
            encoded = logical_bytes(current)
            if hashlib.sha256(encoded).hexdigest() != row["logical_sha256"]:
                raise AssertionError("complete public frame digest mismatch")
            digest.update(encoded)
            count += 1
            yield copy.deepcopy(current)
        if not complete:
            raise AssertionError("incomplete streaming receipt")


def delta_counts(after, before, prefix=""):
    result = {}
    for key, value in after.items():
        label = prefix + key
        previous = before.get(key)
        if isinstance(value, Mapping) and isinstance(previous, Mapping):
            result.update(delta_counts(value, previous, label + "."))
        elif (
            isinstance(value, (float, int))
            and not isinstance(value, bool)
            and isinstance(previous, (float, int))
            and not isinstance(previous, bool)
        ):
            result[label] = value - previous
    return result


def provenance(expected_head=None):
    source = Path(ph.__file__).resolve()
    root = source.parents[2]
    command = ["git", "-C", str(root)]
    head = subprocess.check_output(command + ["rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        command + ["status", "--porcelain", "--", "src/phymeo"], text=True
    ).strip()
    if expected_head is not None and head != expected_head:
        raise AssertionError("actual Phymeo source differs from --expected-head")
    if dirty:
        raise AssertionError("commit shared product changes before measuring")
    files = {}
    for name, module in tuple(sys.modules.items()):
        if (name == "phymeo" or name.startswith("phymeo.")) and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(root / "src"):
                raise AssertionError("mixed source or installed Phymeo modules")
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(
        source_head=head,
        root=str(root),
        module=str(source),
        product_dirty=False,
        imported_files=files,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


def _distribution(values):
    return dict(
        count=len(values),
        total_ns=sum(values),
        p50_ns=float(np.median(values)),
        p95_ns=float(np.percentile(values, 95)),
        minimum_ns=min(values),
        maximum_ns=max(values),
    )


def _raw_counts(mode):
    if mode != "cuda":
        return {}
    import tyx as tx

    return asdict(tx.runtime_stats())


def run_window(make, case, count, phase, mode, cache, destination, *, profile=None):
    before_build = _raw_counts(mode)
    build_start = perf_counter_ns()
    scene = make.build(backend=mode, cache_root=cache)
    build_ns = perf_counter_ns() - build_start
    rows, maximum = [], {}
    count_before = _raw_counts(mode)
    last_value = None
    try:
        meta = dict(
            case=case,
            worlds=count,
            phase=phase,
            dt_s=H,
            steps=STEPS,
            initial=scene.state,
            rigid_rows=scene.rigid_rows,
            input_rows=seeds(case, count),
            output_contract="entire step and geometry; published contact_history if available",
        )
        writer = WindowWriter(destination, meta)
        try:
            for step in range(STEPS):
                before, native = scene.stats(), _raw_counts(mode)
                if profile is None:
                    start = perf_counter_ns()
                    frame = scene.step(H)
                    stepped = perf_counter_ns()
                    geometry = scene.geometry()
                    stop = perf_counter_ns()
                else:
                    # Profiling only touches the diagnostic call, never a formal score.
                    with profile.active(scene):
                        start = perf_counter_ns()
                        frame = scene.step(H)
                        stepped = perf_counter_ns()
                        geometry = scene.geometry()
                        stop = perf_counter_ns()
                statistics, raw = scene.stats(), _raw_counts(mode)
                history = getattr(scene, "contact_history", None)
                expected_time = (step + 1) * H
                if geometry.time_s != expected_time:
                    raise AssertionError("accepted physical clock differs from the fixed dt")
                oracle_start = perf_counter_ns()
                errors = certify_frame(frame, scene.rigid_rows, case, count, step)
                for key, value in errors.items():
                    maximum[key] = max(maximum.get(key, 0), value)
                oracle_ns = perf_counter_ns() - oracle_start
                encoding_start = perf_counter_ns()
                last_value = public_value(
                    dict(frame=frame, geometry=geometry, contact_history=history)
                )
                digest = writer.append(last_value)
                row = dict(
                    step=step + 1,
                    entry_s=step * H,
                    end_s=expected_time,
                    step_ns=stepped - start,
                    geometry_snapshot_ns=stop - stepped,
                    public_to_end_ns=stop - start,
                    oracle_ns=oracle_ns,
                    archive_ns=perf_counter_ns() - encoding_start,
                    actual_contacts=len(frame.loads.contacts),
                    actual_sites=sum(len(c.sites) for c in frame.loads.contacts),
                    actual_required_candidates=frame.path.required_candidates,
                    published_pairs=len(frame.path.pairs),
                    different_world_rows=sum(
                        p.first.world != p.second.world for p in frame.path.pairs
                    ),
                    force_worlds=[c.first.world for c in frame.loads.contacts if c.force_n > 3e-4],
                    counters=delta_counts(statistics, before),
                    native_counters=delta_counts(raw, native),
                    frame_logical_sha256=digest,
                )
                if profile is not None:
                    row["diagnostic_intervals"] = profile.partition(start, stop)
                rows.append(row)
        finally:
            writer.close()
        # Re-read streaming data now, outside all public timing intervals.
        verified = 0
        for value in read_window(destination):
            verified += 1
            if verified == STEPS and value != last_value:
                raise AssertionError("accepted endpoint differs after lossless decoding")
        if verified != STEPS:
            raise AssertionError("missing public physical steps")
        return dict(
            phase=phase,
            build_ns=build_ns,
            successful_steps=len(rows),
            rows=rows,
            native_build_counters=delta_counts(count_before, before_build),
            output=writer.stats(),
            max_physical_errors=maximum,
            public_statistics=_distribution([r["public_to_end_ns"] for r in rows]),
            step_statistics=_distribution([r["step_ns"] for r in rows]),
            native_window_counters=delta_counts(_raw_counts(mode), count_before),
            resource_policy="one new scene; code cache may be warm, its buffers are not reused",
        )
    except BaseException as error:
        failure = dict(
            phase=phase,
            build_ns=build_ns,
            successful_steps=len(rows),
            rows=rows,
            type=type(error).__name__,
            message=str(error),
            state=public_value(scene.state),
            last_frame=public_value(scene.last_frame),
        )
        try:
            failure["time_s"] = scene.geometry().time_s
        except BaseException as observation_error:
            failure["clock_observation_error"] = str(observation_error)
        with Path(destination).with_suffix(".failure.json").open("x", encoding="utf8") as stream:
            json.dump(failure, stream, allow_nan=False)
        raise
    finally:
        scene.close()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--case", choices=("free", "loaded", "birth"), required=True)
    cli.add_argument("--worlds", type=int, choices=(1, 32, 128), required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--expected-head", required=True)
    args = cli.parse_args()
    if args.out.exists():
        cli.error("use a fresh output directory")
    args.out.mkdir(parents=True)
    cache = args.out / "cache"
    before = provenance(args.expected_head)
    make_start = perf_counter_ns()
    make = builder(ph, args.case, args.worlds)
    declaration_ns = perf_counter_ns() - make_start
    report = dict(
        schema=SCHEMA,
        source=before,
        case=args.case,
        worlds=args.worlds,
        started_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version,
        tyx=version("tyx-compute"),
        numpy=version("numpy"),
        backend="cuda",
        matrix=[list(pair) for pair in MATRIX],
        h_s=H,
        steps_per_window=STEPS,
        declaration_ns=declaration_ns,
        windows=[],
        success=False,
        scope="internal public ContactScene costs; no Newton speed comparison",
    )
    try:
        for phase in WINDOWS:
            result = run_window(
                make, args.case, args.worlds, phase, "cuda", cache, args.out / f"{phase}.json.gz"
            )
            report["windows"].append(result)
        report["source_after"] = provenance(args.expected_head)
        if report["source_after"]["source_head"] != before["source_head"]:
            raise AssertionError("product changed during the measurement")
        for name, digest in before["imported_files"].items():
            if report["source_after"]["imported_files"].get(name) != digest:
                raise AssertionError("loaded product file changed during measurement")
        report["success"] = True
    except BaseException as error:
        report["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        with (args.out / "manifest.json").open("x", encoding="utf8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "case": args.case,
                "worlds": args.worlds,
                "success": True,
                "warm_public": report["windows"][1]["public_statistics"],
                "warm_bytes": report["windows"][1]["output"]["gzip_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
