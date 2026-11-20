"""Explicit frozen PhySolve native payload; no Newton or Warp loader."""

import struct
import sys
from dataclasses import dataclass
from hashlib import file_digest
from importlib import metadata, util
from pathlib import Path
from threading import Lock

CUBINS = (
    "mw_phycoreCudaBackEnd.cubin", "hostAssitRuntimeCudaBackEnd.cubin",
    *(f"solverMathDx_b{size}.cubin" for size in (32, 64, 96, 128)),
)
_LOCK = Lock()
_LOADED = None
# Exact paired profiles; do not admit the Cartesian product of both versions.
_FRONTEND_TYX_PAIRS = frozenset({
    ("0.0.0a7", "0.1.0a0"),
    ("0.0.0a8", "0.2.0a1"),
    ("0.0.0a9", "0.2.0a1"),
    ("0.0.0a10", "0.2.0a1"),
    ("0.0.0a11", "0.2.0a1"),
    ("0.0.0a12", "0.2.0a1"),
    ("0.0.0a13", "0.2.0a1"),
    ("0.0.0a14", "0.2.0a1"),
    ("0.0.0a15", "0.2.0a1"),
    ("0.0.0a16", "0.2.0a1"),
    ("0.0.0a17", "0.2.0a1"),
    ("0.0.0a18", "0.2.0a1"),
    ("0.0.0a19", "0.2.0a1"),
    ("0.0.0a20", "0.2.0a1"),
    ("0.0.0a21", "0.2.0a1"),
})


@dataclass(frozen=True)
class PhySolvePayload:
    """Paths and independently recorded hashes; caller owns provenance/notice records.

    The six CUBINs must be siblings. This experimental input is not a replacement
    for the future installed SDK payload manifest or its distribution checks.
    """

    native_path: str
    cubin_directory: str
    native_sha256: str
    cubin_sha256: tuple[tuple[str, str], ...]

    def validate(self, *, primitive=False):
        if sys.platform != "win32" or sys.version_info[:3] != (3, 13, 13):
            raise RuntimeError("PhySolve requires Windows CPython 3.13.13")
        if struct.calcsize("P") != 8 or not sys._is_gil_enabled():
            raise RuntimeError("PhySolve requires standard x64 CPython")
        if not primitive and metadata.version("mujoco") != "3.8.0":
            raise RuntimeError("PhySolve requires mujoco==3.8.0")
        pair = (metadata.version("physolve-tyx-frontend"), metadata.version("tyx-compute"))
        if pair not in _FRONTEND_TYX_PAIRS:
            raise RuntimeError(
                f"PhySolve requires an exact frontend/TYX pair; got {pair!r}, "
                f"supported: {sorted(_FRONTEND_TYX_PAIRS)!r}"
            )
        hashes = dict(self.cubin_sha256)
        if len(self.cubin_sha256) != 6 or set(hashes) != set(CUBINS):
            raise ValueError("exactly six frozen CUBIN hashes are required")
        native = Path(self.native_path).resolve(strict=True)
        directory = Path(self.cubin_directory).resolve(strict=True)
        files = [(native, self.native_sha256)] + [(directory / n, hashes[n]) for n in CUBINS]
        for path, expected in files:
            with path.open("rb") as source:
                actual = file_digest(source, "sha256").hexdigest()
            if actual != expected.lower():
                raise RuntimeError(f"PhySolve frozen payload mismatch: {path.name}")
        return native, directory

    def load(self, *, primitive=False):
        global _LOADED
        native, directory = self.validate(primitive=True) if primitive else self.validate()
        identity = (str(native), str(directory), self.native_sha256.lower(),
                    tuple(sorted((n, h.lower()) for n, h in self.cubin_sha256)))
        with _LOCK:
            if _LOADED is not None:
                if _LOADED[0] != identity:
                    raise RuntimeError("a different PhySolve payload is already loaded")
                return _LOADED[1]
            name = "phymeo.backends.mw_phycore_native_host"
            if name in sys.modules:
                raise RuntimeError("PhySolve native module name is occupied")
            spec = util.spec_from_file_location(name, native)
            module = util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if module.P2PE_RUNTIME_ABI != 1 or module.P2PE_CUDA_ARRAY_INTERFACE_ABI != 1:
                raise RuntimeError("PhySolve native ABI mismatch")
            module._configure(str(directory / CUBINS[0]))
            sys.modules[name] = module
            _LOADED = identity, module
            return module
