"""Public tyx cosine probe; no phymeo import, one bounded kernel per case."""

import argparse
import importlib.metadata
import json
import platform
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tyx as tx


@tx.kernel
def cosine(source: tx.ndarray(tx.f32, 1), output: tx.ndarray(tx.f32, 1)):
    for i in tx.range(3):
        output[i] = tx.cos(source[i])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output path required")
    dist = importlib.metadata.distribution("tyx-compute")
    result = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        numpy=np.__version__,
        tyx=dist.version,
        source=json.loads(dist.read_text("direct_url.json")),
        scope="functional errstate sensitivity; no performance or Warp comparison",
        cases=[],
    )
    for mode in ("interpret", "cuda"):
        for policy in ("raise", "ignore"):
            row = dict(
                mode=mode,
                numpy_errstate=policy,
                input=[0.0, 1e-25, -1e-25],
                expected=[1.0, 1.0, 1.0],
            )
            try:
                with ExitStack() as stack:
                    runtime = tx.Runtime(
                        backend="cuda" if mode == "cuda" else "cpu",
                        cache_root=Path("artifacts/tyx-cos-cache"),
                    )
                    stack.callback(runtime.close)
                    stack.enter_context(tx.use_runtime(runtime))
                    source = np.array(row["input"], np.float32)
                    output = np.zeros(3, np.float32)
                    if mode == "cuda":
                        source = tx.cuda_array(source)
                        stack.callback(source.free)
                        output = tx.cuda_array(output)
                        stack.callback(output.free)
                    with np.errstate(all=policy):
                        cosine(source, output, mode=mode)
                        runtime.sync()
                    actual = output.to_numpy() if mode == "cuda" else output
                    row.update(
                        status="passed" if actual.tolist() == row["expected"] else "wrong_value",
                        actual=actual.tolist(),
                    )
            except Exception as error:
                row.update(status="rejected", error_type=type(error).__name__, error=str(error))
            result["cases"].append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
