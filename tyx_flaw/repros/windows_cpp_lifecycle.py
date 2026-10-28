"""Public tyx Runtime lifecycle probe; Windows module reads, no Phymeo import."""

import argparse
import ctypes
import gc
import importlib.metadata
import json
import os
import platform
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import tyx as tx


@tx.kernel
def twice(source: tx.ndarray(tx.f32, 1), output: tx.ndarray(tx.f32, 1)):
    for i in tx.range(4):
        output[i] = 2.0 * source[i]


def loaded_tyx_libraries():
    """Read this process's loaded DLLs; do not load/unload target libraries."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    process = kernel.GetCurrentProcess()
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.EnumProcessModules.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    psapi.EnumProcessModules.restype = ctypes.c_int
    psapi.GetModuleFileNameExW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
    ]
    psapi.GetModuleFileNameExW.restype = ctypes.c_uint32
    modules = (ctypes.c_void_p * 16384)()
    needed = ctypes.c_uint32()
    if not psapi.EnumProcessModules(process, modules, ctypes.sizeof(modules), ctypes.byref(needed)):
        raise ctypes.WinError(ctypes.get_last_error())
    if needed.value > ctypes.sizeof(modules):
        raise RuntimeError("module enumeration buffer too small")
    names = []
    for module in modules[: needed.value // ctypes.sizeof(ctypes.c_void_p)]:
        name = ctypes.create_unicode_buffer(32768)
        if not psapi.GetModuleFileNameExW(process, module, name, len(name)):
            raise ctypes.WinError(ctypes.get_last_error())
        if "tyx-cpu-cpp-load-" in name.value:
            names.append(name.value)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifetime", choices=["new", "shared"], required=True)
    parser.add_argument("--cycles", type=int, default=64)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "nt" or args.cycles < 1 or args.output.exists():
        parser.error("Windows, positive cycles, and a new output path are required")
    args.cache_root.mkdir(parents=True, exist_ok=True)
    dist = importlib.metadata.distribution("tyx-compute")
    record = dict(
        scope="public API lifecycle diagnostic, no Phymeo import; not a speed benchmark; Warp unmeasured",
        python=platform.python_version(),
        platform=platform.platform(),
        numpy=np.__version__,
        tyx_version=dist.version,
        tyx_source=json.loads(dist.read_text("direct_url.json")),
        lifetime=args.lifetime,
        requested_cycles=args.cycles,
        cache_root=str(args.cache_root.resolve()),
        temp_root=os.environ.get("TEMP"),
        before_count=len(loaded_tyx_libraries()),
        checkpoints=[],
        successful_calls=0,
        closed_runtimes=0,
        seconds=[],
        cache_statuses={},
        status="running",
        sample_profiles=[],
    )
    runtime = None
    statuses = Counter()
    try:
        if args.lifetime == "shared":
            runtime = tx.Runtime(backend="cpu", cache_root=args.cache_root)
        for index in range(args.cycles):
            start = perf_counter()
            if args.lifetime == "new":
                runtime = tx.Runtime(backend="cpu", cache_root=args.cache_root)
            try:
                with tx.use_runtime(runtime), tx.capture_jit_profiles() as capture:
                    source = np.array([1, 2, 3, 4], np.float32)
                    output = np.zeros(4, np.float32)
                    twice(source, output, mode="cpu_cpp")
                    runtime.sync()
                    np.testing.assert_array_equal(output, [2, 4, 6, 8])
                for profile in capture.profiles:
                    statuses[profile.cache_status] += 1
                if index in (0, args.cycles - 1):
                    record["sample_profiles"].extend(asdict(p) for p in capture.profiles)
                record["successful_calls"] += 1
            finally:
                if args.lifetime == "new":
                    runtime.close()
                    assert runtime.closed
                    record["closed_runtimes"] += 1
                    runtime = None
            record["seconds"].append(perf_counter() - start)
            if (index + 1) % 16 == 0:
                gc.collect()
                row = dict(cycle=index + 1, loaded_count=len(loaded_tyx_libraries()))
                record["checkpoints"].append(row)
                print(json.dumps(row), flush=True)
        record["status"] = "all_calls_correct"
    except Exception as error:
        record.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
    finally:
        if runtime is not None:
            runtime.close()
            assert runtime.closed
            record["closed_runtimes"] += 1
        gc.collect()
        remaining = loaded_tyx_libraries()
        record["after_close_gc_count"] = len(remaining)
        record["remaining_path_sample"] = remaining[:3]
        record["cache_statuses"] = dict(statuses)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                k: record[k]
                for k in ("status", "successful_calls", "closed_runtimes", "after_close_gc_count")
            }
        )
    )
    return 0 if record["status"] == "all_calls_correct" else 1


if __name__ == "__main__":
    raise SystemExit(main())
