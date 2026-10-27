"""Public tyx syntax probe; no phymeo imports or performance claims."""

import argparse
import importlib.metadata
import json
import platform
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tyx as tx


@tx.kernel
def chained(source: tx.ndarray(tx.f32, 1), output: tx.ndarray(tx.i32, 1)):
    for i in tx.range(5):
        if 0.0 < source[i] < 1.0:
            output[i] = 1


@tx.kernel
def expanded(source: tx.ndarray(tx.f32, 1), output: tx.ndarray(tx.i32, 1)):
    for i in tx.range(5):
        if 0.0 < source[i] and source[i] < 1.0:
            output[i] = 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/tyx-chain-probe-cache"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    dist = importlib.metadata.distribution("tyx-compute")
    record = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        numpy=np.__version__,
        tyx_version=dist.version,
        tyx_source=json.loads(dist.read_text("direct_url.json")),
        scope="syntax capability request, not a promised support regression; Warp unmeasured",
        results=[],
    )
    values = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], np.float32)
    for mode in ("interpret", "cpu_cpp", "cuda"):
        for name, kernel in (("chained", chained), ("expanded", expanded)):
            row = dict(mode=mode, syntax=name, input=values.tolist(), expected=[0, 0, 1, 0, 0])
            try:
                with ExitStack() as stack:
                    runtime = tx.Runtime(
                        backend="cuda" if mode == "cuda" else "cpu", cache_root=args.cache_root
                    )
                    stack.callback(runtime.close)
                    stack.enter_context(tx.use_runtime(runtime))
                    source, output = values.copy(), np.zeros(5, np.int32)
                    if mode == "cuda":
                        source = tx.cuda_array(source)
                        stack.callback(source.free)
                        output = tx.cuda_array(output)
                        stack.callback(output.free)
                    kernel(source, output, mode=mode)
                    runtime.sync()
                    actual = output.to_numpy() if mode == "cuda" else output
                    row.update(
                        status="passed" if actual.tolist() == row["expected"] else "wrong_value",
                        actual=actual.tolist(),
                    )
            except Exception as error:
                row.update(status="rejected", error_type=type(error).__name__, error=str(error))
            record["results"].append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
