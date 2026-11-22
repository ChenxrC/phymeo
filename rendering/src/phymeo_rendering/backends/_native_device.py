"""Private ctypes boundary to the independently built Vulkan C ABI."""

import ctypes as ct
import hashlib
import math
import sys
from pathlib import Path

from . import BackendUnavailableError

_F4 = ct.c_float * 4
_U4 = ct.c_uint32 * 4


class Vertex(ct.Structure):
    _fields_ = [("position", _F4), ("normal_metallic", _F4), ("color_roughness", _F4), ("ids", _U4)]


class Parameters(ct.Structure):
    _fields_ = [
        ("view", ct.c_float * 16),
        ("intrinsics", _F4),
        ("clipping", _F4),
        ("camera", _F4),
        ("light_directions", _F4 * 4),
        ("light_colors", _F4 * 4),
        ("counts", _U4),
        ("options", _F4),
    ]


class _Info(ct.Structure):
    _fields_ = [
        (name, ct.c_uint32)
        for name in (
            "abi_version",
            "device_index",
            "api_version",
            "driver_version",
            "vendor_id",
            "device_id",
            "device_type",
            "max_width",
            "max_height",
            "max_vertices",
        )
    ] + [
        ("device_uuid", ct.c_uint8 * 16),
        ("device_name", ct.c_char * 256),
        ("validation_errors", ct.c_uint64),
        ("validation_warnings", ct.c_uint64),
        ("captures", ct.c_uint64),
    ]


class _Counts(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in ("live_contexts", "live_frames", "live_resources")]


def validate_parameters(value):
    """Check the actual FP32 ABI values, shared by admission and submission."""
    blocks = (
        value.view,
        value.intrinsics,
        value.clipping,
        value.camera,
        *(row for row in value.light_directions),
        *(row for row in value.light_colors),
        value.options,
    )
    if not all(math.isfinite(x) for block in blocks for x in block):
        raise ValueError("native uniform FP32 conversion is nonfinite")
    if not (
        value.intrinsics[0] > 0
        and value.intrinsics[1] > 0
        and value.clipping[2] > 0
        and value.clipping[3] > value.clipping[2]
        and value.options[0] >= 0
        and value.options[1] > 0
    ):
        raise ValueError("native projection/exposure is not usable after FP32 conversion")


def default_library():
    name = "phymeo_native.dll" if sys.platform == "win32" else "libphymeo_native.so"
    return Path(__file__).with_name("_native") / name


def require_library(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise BackendUnavailableError(f"native Vulkan library is not built or installed: {path}")
    return path


class NativeDevice:
    def __init__(self, library_path, shader_directory, device_index, validation):
        self._handle = None
        self._last_info = {}
        self.library_path = require_library(library_path)
        self.shader_directory = Path(shader_directory).resolve()
        if sys.byteorder != "little":
            raise BackendUnavailableError("native C ABI requires a little-endian host")
        if (ct.sizeof(Vertex), ct.sizeof(Parameters), ct.sizeof(_Info)) != (64, 272, 336):
            raise BackendUnavailableError("native C ABI structure layout mismatch")
        shaders = []
        self.shader_sha256 = {}
        for stage in ("vert", "frag"):
            path = self.shader_directory / f"native.{stage}.spv"
            try:
                data = path.read_bytes()
            except OSError as error:
                raise BackendUnavailableError(f"native SPIR-V file unavailable: {path}") from error
            if len(data) < 20 or len(data) % 4 or data[:4] != b"\x03\x02\x23\x07":
                raise BackendUnavailableError(f"invalid SPIR-V file: {path}")
            shaders.append((ct.c_uint32 * (len(data) // 4)).from_buffer_copy(data))
            self.shader_sha256[stage] = hashlib.sha256(data).hexdigest()
        self.library_sha256 = hashlib.sha256(self.library_path.read_bytes()).hexdigest()
        try:
            self._lib = ct.CDLL(str(self.library_path))
            self._bind()
        except (OSError, AttributeError) as error:
            raise BackendUnavailableError(
                f"native Vulkan library could not load: {error}"
            ) from error
        if self._lib.pr_abi_version() != 2:
            raise BackendUnavailableError("unsupported native Vulkan C ABI version")
        self.validation = validation
        self._handle = self._lib.pr_create(
            device_index,
            int(validation),
            shaders[0],
            ct.sizeof(shaders[0]),
            shaders[1],
            ct.sizeof(shaders[1]),
        )
        if not self._handle:
            raise BackendUnavailableError(self._error())
        try:
            self.info()
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "native initialization and cleanup failed", [primary, cleanup]
                )
            raise

    def _bind(self):
        lib = self._lib
        lib.pr_abi_version.argtypes = []
        lib.pr_abi_version.restype = ct.c_uint32
        lib.pr_error.argtypes = []
        lib.pr_error.restype = ct.c_char_p
        lib.pr_create.argtypes = [
            ct.c_uint32,
            ct.c_uint32,
            ct.POINTER(ct.c_uint32),
            ct.c_size_t,
            ct.POINTER(ct.c_uint32),
            ct.c_size_t,
        ]
        lib.pr_create.restype = ct.c_void_p
        lib.pr_info.argtypes = [ct.c_void_p, ct.POINTER(_Info)]
        lib.pr_info.restype = ct.c_int
        lib.pr_render.argtypes = [
            ct.c_void_p,
            ct.POINTER(Vertex),
            ct.c_uint32,
            ct.POINTER(Parameters),
            ct.c_uint32,
            ct.c_uint32,
            ct.POINTER(ct.c_uint8),
            ct.c_size_t,
        ]
        lib.pr_render.restype = ct.c_int
        lib.pr_destroy.argtypes = [ct.c_void_p, ct.POINTER(_Info)]
        lib.pr_destroy.restype = ct.c_int
        lib.pr_counts.argtypes = [ct.POINTER(_Counts)]
        lib.pr_counts.restype = ct.c_int

    def _error(self):
        text = self._lib.pr_error()
        return text.decode("utf-8", errors="replace") if text else "native Vulkan operation failed"

    def _remember(self, value):
        self._last_info = {
            name: int(getattr(value, name))
            for name, _ in _Info._fields_
            if name not in ("device_name", "device_uuid")
        }
        self._last_info["device_name"] = value.device_name.decode("utf-8", errors="replace")
        self._last_info["device_uuid"] = bytes(value.device_uuid).hex()

    def info(self):
        if self._handle is not None:
            value = _Info()
            if self._lib.pr_info(self._handle, ct.byref(value)):
                raise RuntimeError(self._error())
            self._remember(value)
        counts = _Counts()
        if self._lib.pr_counts(ct.byref(counts)):
            raise RuntimeError(self._error())
        return dict(
            self._last_info,
            closed=self._handle is None,
            validation_enabled=self.validation,
            **{name: int(getattr(counts, name)) for name, _ in _Counts._fields_},
        )

    def render(self, vertices, parameters, width, height):
        if self._handle is None:
            raise RuntimeError("native device is closed")
        limits = self._last_info or self.info()
        if (
            type(width) is not int
            or type(height) is not int
            or not 0 < width <= min(limits["max_width"], 2**32 - 1)
            or not 0 < height <= min(limits["max_height"], 2**32 - 1)
            or width * height > 4096 * 4096
        ):
            raise ValueError("native capture extent or pixel budget exceeded")
        if len(vertices) % 3 or len(vertices) > limits["max_vertices"]:
            raise ValueError("native triangle budget exceeded")
        validate_parameters(parameters)
        if parameters.clipping[0] != width or parameters.clipping[1] != height:
            raise ValueError("native clipping extent mismatch")
        for vertex in vertices:
            if not all(
                math.isfinite(x)
                for block in (vertex.position, vertex.normal_metallic, vertex.color_roughness)
                for x in block
            ):
                raise ValueError("native vertex FP32 conversion is nonfinite")
            length = math.hypot(*vertex.normal_metallic[:3])
            if not 0.999 <= length <= 1.001:
                raise ValueError("native vertex normal must be unit length in FP32")
        output = (ct.c_uint8 * (width * height * 13))()
        if self._lib.pr_render(
            self._handle,
            vertices,
            len(vertices),
            ct.byref(parameters),
            width,
            height,
            output,
            ct.sizeof(output),
        ):
            raise RuntimeError(self._error())
        return bytes(output)

    def close(self):
        if self._handle is None:
            return
        # Destruction cannot depend on a successful diagnostic query. The C ABI
        # returns diagnostics after real teardown, even when it reports errors.
        handle, self._handle = self._handle, None
        final = _Info()
        result = self._lib.pr_destroy(handle, ct.byref(final))
        self._remember(final)
        if result:
            raise RuntimeError(self._error())
