"""Run the Newton comparison catalogs with one engine and one explicit device per process."""

import argparse
import contextlib
import importlib.util
import json
import math
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LANES = ("rr", "rd", "robot", "dd")
ENGINES = ("newton", "phymeo")
DEVICES = ("cpu", "cuda:0")


def load_cases(lane):
    data = json.loads((ROOT / lane / "cases.json").read_text(encoding="utf-8"))
    if data.get("schema") != "newton-reference-v1" or data.get("lane") != lane:
        raise ValueError(f"Invalid {lane} catalog header")
    cases = data["cases"]
    ids = set()
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Empty {lane} catalog")
    for row in cases:
        key = row["id"]
        if not isinstance(key, str) or not key.startswith(lane.upper() + "-") or key in ids:
            raise ValueError(f"Invalid or duplicate {lane} case ID {key!r}")
        ids.add(key)
        if (
            row["difficulty"] not in ("easy", "medium", "hard")
            or not row["family"]
            or not row["oracle"]
        ):
            raise ValueError(f"Missing case definition for {key}")
        for name in ("inputs", "expected", "tolerances"):
            if not isinstance(row[name], dict) or not row[name]:
                raise ValueError(f"{key} needs explicit {name}")
        for value in row["tolerances"].values():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid tolerance in {key}")
        for engine in ENGINES:
            plan = row[engine]
            if plan["status"] not in ("run", "gap", "not_comparable") or not plan["reason"]:
                raise ValueError(f"Invalid {engine} expectation for {key}")
            devices = plan["devices"]
            if (
                not isinstance(devices, list)
                or len(set(devices)) != len(devices)
                or set(devices) - set(DEVICES)
            ):
                raise ValueError(f"Invalid device set in {key}")
            if bool(devices) != (plan["status"] == "run"):
                raise ValueError(f"Execution devices must match the declared status in {key}")
    return cases


def load_runner(lane):
    path = ROOT / lane / "baseline.py"
    spec = importlib.util.spec_from_file_location(
        "newton_reference_" + lane, path, submodule_search_locations=[str(path.parent)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.run_case


def evaluate(lane, cases, engine, device):
    run_case = None
    records = []
    for case in cases:
        policy = case[engine]
        record = {"id": case["id"], "lane": lane, "reason": policy["reason"]}
        if policy["status"] != "run":
            record["status"] = policy["status"]
        elif device not in policy["devices"]:
            record["status"] = "device_not_covered"
        else:
            try:
                if run_case is None:
                    run_case = load_runner(lane)
                observed = run_case(case, engine=engine, device=device)
                if not isinstance(observed, dict) or not observed:
                    raise ValueError("The runner returned no measurements")
                json.dumps(observed, allow_nan=False)
                record.update(status="pass", observed=observed)
            except Exception as exc:
                record.update(status="fail", error=f"{type(exc).__name__}: {exc}")
                evidence = getattr(exc, "observed", None)
                if isinstance(evidence, dict):
                    try:
                        json.dumps(evidence, allow_nan=False)
                    except (TypeError, ValueError) as invalid:
                        record["evidence_error"] = str(invalid)
                    else:
                        record["observed"] = evidence
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("all", *LANES), default="all")
    parser.add_argument("--case", help="Select one catalog ID")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--engine", choices=ENGINES)
    parser.add_argument("--device", choices=DEVICES, default="cpu")
    parser.add_argument("--out", type=Path, help="Create a new JSON report")
    args = parser.parse_args()
    if args.out is not None and args.out.exists():
        parser.error("Choose a new output path")
    lanes = LANES if args.lane == "all" else (args.lane,)
    selections = {
        lane: [row for row in load_cases(lane) if not args.case or row["id"] == args.case]
        for lane in lanes
    }
    if not any(selections.values()):
        parser.error("No matching cases")
    if args.list:
        print(
            json.dumps(
                {lane: [row["id"] for row in cases] for lane, cases in selections.items()}, indent=2
            )
        )
        return 0
    if not __debug__:
        parser.error("Python optimization would disable case assertions")
    runtime = {"python": sys.version.split()[0]}
    device_scope = contextlib.nullcontext()
    if args.engine == "newton":
        import warp as wp

        runtime.update(newton=version("newton"), warp=version("warp-lang"))
        if runtime["newton"] != "1.7.0":
            raise RuntimeError("This catalog targets Newton 1.7.0")
        wp.config.kernel_cache_dir = str(Path(sys.prefix) / "warp-kernels")
        wp.init()
        selected = wp.get_device(args.device)
        if str(selected) != args.device:
            raise RuntimeError(f"Unexpected Warp device {selected}")
        runtime["device_name"] = selected.name
        device_scope = wp.ScopedDevice(selected)
    else:
        import phymeo

        origin = Path(phymeo.__file__).resolve()
        if not origin.is_relative_to(ROOT.parents[1] / "src"):
            raise RuntimeError("Set PYTHONPATH to the src directory of this checkout")
        runtime["package_path"] = str(origin)
        if args.device == "cuda:0":
            runtime["tyx"] = version("tyx-compute")
    with device_scope:
        records = [
            r
            for lane, cases in selections.items()
            for r in evaluate(lane, cases, args.engine, args.device)
        ]
    counts = dict(Counter(r["status"] for r in records))
    report = dict(
        schema="newton-reference-results-v1",
        engine=args.engine,
        device=args.device,
        runtime=runtime,
        counts=counts,
        records=records,
    )
    if args.out is not None:
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
    for record in records:
        if record["status"] == "fail":
            print(record["id"], record["error"], file=sys.stderr)
    print(
        json.dumps(dict(engine=args.engine, device=args.device, counts=counts), ensure_ascii=False)
    )
    return int("fail" in counts)


if __name__ == "__main__":
    raise SystemExit(main())
