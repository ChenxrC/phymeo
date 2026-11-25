"""Lower existing Tyx scalar/ndarray kernels to MCC and launch them on a MUSA stream.

The shared Tyx GPU emitter owns the device algorithm and element addressing.
MCC builds the resulting translation unit and the host entry uses the MUSA
runtime API on the caller's stream. The owning runtime keeps array allocations
live through completion and reports asynchronous errors at synchronization.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import operator
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


class MusaCompileError(RuntimeError):
    """Source or toolchain cannot implement the requested kernel."""


class MusaLaunchError(RuntimeError):
    """Device runtime rejected a kernel launch."""


_LAUNCHER = """
extern "C" int phymeo_musa_launch(unsigned int grid, unsigned int block,
                                  void* stream, void** arguments) {
    return static_cast<int>(musaLaunchKernel(
        reinterpret_cast<const void*>(&SYMBOL), dim3(grid), dim3(block),
        arguments, 0, reinterpret_cast<musaStream_t>(stream)));
}
extern "C" const char* phymeo_musa_error(int code) {
    return musaGetErrorString(static_cast<musaError_t>(code));
}
"""


def _emit(kernel):
    from tyx.backends.cuda_source import optimize_cuda_ir
    from tyx.backends.hip_source import analyze_hip_schedule, generate_hip
    from tyx.backends.kernel_abi import kernel_abi_slots
    from tyx.types import ArrayType, ScalarType

    module = kernel.lower()
    if tuple(module.args) != tuple(kernel.signature):
        raise MusaCompileError("flattened struct parameters require a separate by-value ABI")
    for name, typ in module.args.items():
        scalar = typ.dtype if isinstance(typ, ArrayType) else typ
        if not isinstance(scalar, ScalarType) or scalar.name not in ("i32", "u32", "f32", "f64"):
            raise MusaCompileError(f"unsupported scalar or packed MUSA parameter {name}: {typ}")
    if kernel.gpu_schedule.mode != "manual":
        raise MusaCompileError("MUSA auto scheduling needs measured target capabilities")
    schedule = kernel.gpu_schedule.resolve()
    options = schedule.codegen_options
    if options.fast_math or getattr(options, "launch_bounds", None) is not None:
        raise MusaCompileError("MUSA adapter requires the precise scalar schedule")
    optimized = optimize_cuda_ir(module)
    report = analyze_hip_schedule(optimized, options)
    if (
        not report.eligible
        or getattr(report, "cooperative_tiles", False)
        or getattr(module, "uses_block_lanes", False)
    ):
        raise MusaCompileError(f"unsupported MUSA kernel schedule: {report.reason_codes}")

    # Tyx's HIP emitter already removes NVRTC-only typedefs, substitutes portable
    # reductions, and preserves explicitly rounded operations. Retarget only its
    # toolchain header. Unsupported AMD math/assembly must never be approximated.
    source, symbol = generate_hip(optimized, codegen_options=options)
    hip_headers = "#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n"
    if not source.startswith(hip_headers):
        raise MusaCompileError("unrecognized Tyx portable-GPU prelude")
    body = source[len(hip_headers) :]
    if re.search(r"__ocml_|\b(?:hip[A-Z]\w*|__half|__nv_bfloat16|__asm__|asm)\b", body):
        raise MusaCompileError(f"{symbol} uses an unqualified vendor-specific device operation")
    prelude = "#include <musa_runtime.h>\n#include <stdint.h>\n#include <math.h>\n"
    return (
        prelude + body + _LAUNCHER.replace("SYMBOL", symbol),
        symbol,
        optimized.module,
        kernel_abi_slots(optimized.module.args.items()),
        schedule.block_dim,
        report,
        options,
    )


def generate_musa_source(kernel):
    """Return (MCC translation unit, device symbol) without requiring MUSA hardware."""
    source, symbol, *_ = _emit(kernel)
    return source, symbol


def _native_toolchain(mcc):
    home = Path(os.environ.get("MUSA_HOME", "/usr/local/musa"))
    candidate = mcc or os.environ.get("MCC") or shutil.which("mcc") or home / "bin" / "mcc"
    found = shutil.which(os.fspath(candidate))
    if found is None:
        raise MusaCompileError(f"cannot find MCC executable {candidate}")
    compiler = Path(found).resolve()
    libdirs = tuple(
        dict.fromkeys(
            path
            for prefix in (home, compiler.parent.parent)
            for path in (prefix / "lib", prefix / "lib64")
            if path.is_dir()
        )
    )
    return compiler, libdirs


class MusaKernel:
    """A concrete Tyx kernel. All arrays must be resident on the caller's MUSA device."""

    def __init__(self, kernel, *, cache_root=None, arch="mp_22", mcc=None):
        if sys.platform != "linux":
            raise MusaCompileError("the MCC shared-library launcher currently requires Linux")
        if not re.fullmatch(r"mp_[0-9]+", arch):
            raise ValueError(f"invalid MCC architecture: {arch}")
        source, self.symbol, self.module, self.slots, self.block_dim, self.report, options = _emit(
            kernel
        )
        self.source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        self._strided = any(getattr(slot, "stride_axis", None) is not None for slot in self.slots)
        if self.report.root_execution not in ("parallel_root", "serial_root"):
            raise MusaCompileError(f"unsupported root execution: {self.report.root_execution}")
        compiler, libdirs = _native_toolchain(mcc)
        self.arch = arch
        version = subprocess.run(
            [str(compiler), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        flags = ["-x", "musa", f"--offload-arch={arch}", "-std=c++17", "-O2", "-fPIC", "-shared"]
        if not getattr(options, "fuse_fp", True):
            flags.append("-ffp-contract=off")
        ldflags = [f"-L{path}" for path in libdirs] + ["-Wl,--no-as-needed", "-lmusart", "-lmusa"]
        identity = json.dumps([self.source_sha256, str(compiler), version, flags, ldflags])
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        base = Path(cache_root) if cache_root is not None else Path.home() / ".cache" / "phymeo"
        target = base.resolve() / "tyx-musa" / key
        target.mkdir(parents=True, exist_ok=True)
        self.artifact = target / "kernel.so"
        self.source_path = target / "kernel.mu"
        self.cache_hit = self.artifact.is_file()
        if not self.cache_hit:
            with tempfile.TemporaryDirectory(dir=target, prefix="build-") as work:
                src = Path(work) / "kernel.mu"
                binary = Path(work) / "kernel.so"
                src.write_text(source, encoding="utf-8")
                command = [str(compiler), *flags, str(src), *ldflags, "-o", str(binary)]
                build = subprocess.run(command, capture_output=True, text=True)
                if build.returncode:
                    # Keep exact untranslated diagnostics and input for reproducibility.
                    os.replace(src, self.source_path)
                    raise MusaCompileError(
                        f"MCC failed for {self.symbol} (exit {build.returncode}, "
                        f"source {self.source_path}):\n{build.stdout}{build.stderr}"
                    )
                os.replace(src, self.source_path)
                os.replace(binary, self.artifact)
        self._library = ctypes.CDLL(str(self.artifact))
        self._launch = self._library.phymeo_musa_launch
        self._launch.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._launch.restype = ctypes.c_int
        self._error = self._library.phymeo_musa_error
        self._error.argtypes = [ctypes.c_int]
        self._error.restype = ctypes.c_char_p

    def launch(self, *args, stream_ptr: int, device: int) -> bool:
        from tyx.backends.cuda_runtime import _infer_launch_size
        from tyx.types import ArrayType

        argmap = dict(zip(self.module.args, args, strict=True))
        arrays = {
            name: _array_argument(argmap[name], typ, device, self._strided)
            for name, typ in self.module.args.items()
            if isinstance(typ, ArrayType)
        }
        values = []
        for slot in self.slots:
            name = slot.source_name
            if slot.shape_axis is not None:
                values.append(ctypes.c_int32(arrays[name][1][slot.shape_axis]))
            elif getattr(slot, "stride_axis", None) is not None:
                values.append(ctypes.c_int32(arrays[name][2][slot.stride_axis]))
            elif name in arrays:
                values.append(ctypes.c_void_p(arrays[name][0]))
            else:
                values.append(_scalar_argument(argmap[name], slot.typ))
        # The portable-GPU emitter maps serial roots to lane 0 and emits its own
        # grid-stride loop for parallel roots. Use Tyx's IR bound/shape evaluator.
        if self.report.root_execution == "serial_root":
            grid = 1
        else:
            extent = _infer_launch_size(self.module, argmap)
            if extent < 0 or extent > 0x7FFFFFFF:
                raise ValueError(f"logical kernel extent outside i32: {extent}")
            if extent == 0:
                return False
            grid = (extent + self.block_dim - 1) // self.block_dim
        pointers = (ctypes.c_void_p * len(values))(*(ctypes.addressof(v) for v in values))
        status = self._launch(grid, self.block_dim, stream_ptr, pointers)
        if status:
            reason = self._error(status)
            message = reason.decode(errors="replace") if reason else f"error {status}"
            raise MusaLaunchError(f"{self.symbol} launch on musa:{device} failed: {message}")
        return True


def _array_argument(value, typ, device, strided):
    dtype = np.dtype(typ.dtype.numpy_dtype)
    value_device = value.device_index
    if value_device != device:
        raise ValueError(f"MUSA array device {value_device} differs from launch device {device}")
    if value.dtype != dtype:
        raise TypeError(f"expected MUSA array of {dtype}, got {value.dtype}")
    shape = tuple(operator.index(n) for n in value.shape)
    if len(shape) != typ.ndim or any(n < 0 or n > 0x7FFFFFFF for n in shape):
        raise ValueError(f"shape {shape} is incompatible with rank {typ.ndim} and i32 metadata")
    strides = value.stride()
    if len(strides) != len(shape) or any(s < 0 or s > 0x7FFFFFFF for s in strides):
        raise ValueError(f"invalid MUSA element strides: {strides}")
    occupied = all(shape)
    if occupied:
        if sum((n - 1) * s for n, s in zip(shape, strides)) > 0x7FFFFFFF:
            raise ValueError("MUSA array element offsets overflow Tyx i32 addressing")
        if not strided:
            pitch = 1
            for n, stride in zip(reversed(shape), reversed(strides)):
                if n > 1 and stride != pitch:
                    raise ValueError("this Tyx ABI requires contiguous MUSA arrays")
                pitch *= n
    pointer = operator.index(value.data_ptr())
    if pointer < 0 or (occupied and pointer == 0) or pointer % dtype.itemsize:
        raise ValueError("invalid MUSA device pointer")
    return pointer, shape, strides


def _scalar_argument(value, typ):
    name = typ.name
    if name in ("f32", "f64"):
        return (ctypes.c_float if name == "f32" else ctypes.c_double)(value)
    integer = operator.index(value)
    low, high = (-(2**31), 2**31 - 1) if name == "i32" else (0, 2**32 - 1)
    if not low <= integer <= high:
        raise OverflowError(f"{integer} does not fit {name}")
    return (ctypes.c_int32 if name == "i32" else ctypes.c_uint32)(integer)
