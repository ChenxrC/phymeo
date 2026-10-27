import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tyx as tx


@tx.kernel
def probe(x: tx.ndarray(tx.f32, 1), y: tx.ndarray(tx.f32, 1)):
    for i in tx.range(4):
        y[i] = x[i] * 2.0


rows = []
for label, root in [("default", None), ("workspace", Path("artifacts/p6-cache-probe"))]:
    for index in range(2):
        runtime = tx.Runtime(backend="cpu", cache_root=root)
        try:
            with tx.use_runtime(runtime), tx.capture_jit_profiles() as capture:
                for call in range(2):
                    x = np.array([1, 2, 3, 4], np.float32)
                    y = np.zeros(4, np.float32)
                    probe(x, y, mode="cpu_cpp")
                    runtime.sync()
                    assert y.tolist() == [2, 4, 6, 8]
            rows.append(
                dict(
                    label=label,
                    runtime=index,
                    cache_root=str(runtime.cache_root),
                    profiles=[asdict(p) for p in capture.profiles],
                )
            )
        finally:
            runtime.close()
result = dict(
    scope="public API cache diagnostics; concurrent workload, not performance baseline; no Warp comparison",
    results=rows,
)
Path("artifacts/tmp/p6-cache-probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
for row in rows:
    print(
        row["label"],
        row["runtime"],
        [(p["cache_status"], p["cache_reason_codes"]) for p in row["profiles"]],
    )
