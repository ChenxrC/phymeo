"""Probe the FP64 compute gap using public tyx API; no phymeo solver is involved."""

import argparse
import importlib.metadata
import json
import platform
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tyx as tx

# A public descriptor is a diagnostic probe, not a claim of supported builtin f64.
F64 = tx.ScalarType("f64", "float64")


@tx.kernel
def double_values(n: tx.i32, source: tx.ndarray(F64, 1), output: tx.ndarray(F64, 1)):
    for i in range(n):
        output[i] = source[i] + source[i]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/tyx-fp64-probe-cache"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    distribution = importlib.metadata.distribution("tyx-compute")
    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "tyx_version": distribution.version,
        "tyx_source": json.loads(distribution.read_text("direct_url.json")),
        "builtin_f64_present": hasattr(tx, "f64"),
        "results": [],
        "scope": "unsupported custom ScalarType probe; builtin absence is a capability request, not a broken support promise; Warp unmeasured",
    }
    source = np.array([16777217.0], np.float64)
    for mode in ("interpret", "cpu_cpp", "cuda"):
        row = {"mode": mode, "input": source.tolist(), "expected": (source * 2).tolist()}
        try:
            with ExitStack() as resources:
                runtime = tx.Runtime(
                    backend="cuda" if mode == "cuda" else "cpu", cache_root=args.cache_root
                )
                resources.callback(runtime.close)
                resources.enter_context(tx.use_runtime(runtime))
                if mode == "cuda":
                    a, b = tx.cuda_array(source), None
                    resources.callback(a.free)
                    row["fp64_storage_roundtrip"] = bool(np.array_equal(a.to_numpy(), source))
                    b = tx.cuda_array(np.zeros_like(source))
                    resources.callback(b.free)
                    resources.callback(runtime.sync)
                else:
                    a, b = source.copy(), np.zeros_like(source)
                double_values(len(source), a, b, mode=mode)
                runtime.sync()
                result = b.to_numpy() if mode == "cuda" else b
                row.update(
                    status="passed" if np.array_equal(result, source * 2) else "wrong_value",
                    actual=result.tolist(),
                )
        except Exception as error:
            row.update(status="rejected", error_type=type(error).__name__, error=str(error))
        record["results"].append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
