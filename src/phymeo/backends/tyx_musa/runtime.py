"""Owned MUSA allocations and one ordered kernel stream per runtime.

The kernel compiler supplies a host launch stub. All copies use libmusart
and all kernels receive the same explicit stream. No device reset or global
device synchronization is needed when independent solver worlds are closed.
"""

import ctypes as ct
import math
import operator
import os
import sys
from contextlib import contextmanager
from ctypes.util import find_library
from pathlib import Path
from threading import RLock

import numpy as np


class MusaError(RuntimeError):
    """A MUSA operation or runtime dependency failed."""


def _load(library):
    if sys.platform != "linux":
        raise MusaError("The Phymeo MUSA runtime requires Linux and the Moore Threads MUSA driver")
    if library is None:
        root = Path(os.environ.get("MUSA_HOME") or "/usr/local/musa")
        candidates = (
            find_library("musart"),
            "libmusart.so",
            root / "lib" / "libmusart.so",
            root / "lib64" / "libmusart.so",
        )
    else:
        candidates = (os.fspath(library),)
    errors = []
    for name in dict.fromkeys(candidates):
        if name:
            try:
                return ct.CDLL(os.fspath(name))
            except OSError as exc:
                errors.append(str(exc))
    raise MusaError("Could not load the MUSA runtime (libmusart): " + " | ".join(errors))


def _bind(lib, name, result, *args):
    try:
        fn = getattr(lib, name)
    except AttributeError as exc:
        raise MusaError(f"Missing {name} in {lib._name}") from exc
    fn.argtypes, fn.restype = list(args), result
    return fn


def _device_identity():
    """Ask the driver about the actual current libmusart context, independent of compiler target."""
    try:
        driver = ct.CDLL("libmusa.so.1")
    except OSError as exc:
        raise MusaError(f"Could not load the MUSA driver (libmusa.so.1): {exc}") from exc
    integer = ct.c_int
    int_out = ct.POINTER(integer)
    get_device = _bind(driver, "muCtxGetDevice", integer, int_out)
    get_name = _bind(driver, "muDeviceGetName", integer, ct.POINTER(ct.c_char), integer, integer)
    capability = _bind(driver, "muDeviceComputeCapability", integer, int_out, int_out, integer)
    device, major, minor = integer(), integer(), integer()
    name = ct.create_string_buffer(256)
    for fn, args in (
        (get_device, (ct.byref(device),)),
        (get_name, (name, len(name), device)),
        (capability, (ct.byref(major), ct.byref(minor), device)),
    ):
        status = fn(*args)
        if status:
            raise MusaError(
                f"{fn.__name__} failed to identify the active MUSA device (status {status})"
            )
    if not name.value or major.value <= 0 or minor.value < 0:
        raise MusaError("The MUSA driver returned an invalid device name or compute capability")
    return dict(
        device_name=name.value.decode("utf-8", "replace"), arch=f"mp_{major.value}{minor.value}"
    )


def _layout(shape, dtype):
    if isinstance(shape, (int, np.integer)):
        shape = (shape,)
    dimensions = []
    for axis in shape:
        if isinstance(axis, (bool, np.bool_)):
            raise TypeError("MUSA array dimensions must be integers")
        size = operator.index(axis)
        if not 0 <= size <= sys.maxsize:
            raise ValueError("Invalid MUSA array dimension")
        dimensions.append(size)
    shape = tuple(dimensions)
    dtype = np.dtype(dtype)
    if not dtype.isnative or dtype.kind not in "biufc" or dtype.subdtype:
        raise TypeError("MUSA device arrays require a native-endian numeric dtype")
    nbytes = math.prod(shape) * dtype.itemsize
    if nbytes > sys.maxsize:
        raise OverflowError("MUSA allocation exceeds host address range")
    return shape, dtype, nbytes


class MusaArray:
    """A contiguous device allocation. Its creating runtime controls its lifetime."""

    def __init__(self, runtime, shape, dtype, nbytes, pointer):
        self._runtime = runtime
        self._shape, self._dtype, self._nbytes = shape, dtype, nbytes
        self._pointer = pointer
        self._closed = False

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def nbytes(self):
        return self._nbytes

    @property
    def device_index(self):
        return self._runtime.device_index

    @property
    def strides(self):
        step, result = self.dtype.itemsize, []
        for size in reversed(self.shape):
            result.append(step)
            step *= size
        return tuple(reversed(result))

    @property
    def closed(self):
        return self._closed

    @property
    def ptr(self):
        with self._runtime._lock:
            self._runtime._require_array(self)
            return self._pointer

    def data_ptr(self):
        return self.ptr

    def stride(self):
        return tuple(s // self.dtype.itemsize for s in self.strides)

    def copy_from(self, value):
        """Synchronously upload a matching NumPy array, including strided host views."""
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != self.dtype
            or value.shape != self.shape
        ):
            raise ValueError("MUSA upload requires a NumPy array of the same shape and dtype")
        self._runtime._copy(self, np.require(value, requirements=["C"]), upload=True)

    def to_numpy(self):
        """Return a new CPU-owned array after the stream and device copy complete."""
        result = np.empty(self.shape, self.dtype)
        self._runtime._copy(self, result, upload=False)
        return result

    def close(self):
        self._runtime._free(self)

    free = close

    def __enter__(self):
        self.data_ptr()
        return self

    def __exit__(self, *args):
        self.close()


# After an ambiguous native release, retain the opaque handle and its loaded
# modules. Repeating that release or unloading running code could corrupt another world.
_QUARANTINED = {}


class MusaRuntime:
    """One Linux MUSA device/stream owner. Kernels launch asynchronously."""

    backend = "musa"

    def __init__(self, device_index=0, *, library=None):
        if type(device_index) is not int or not 0 <= device_index <= 2**31 - 1:
            raise ValueError("device_index must be a nonnegative C int")
        self._device_index = device_index
        self._lock = RLock()
        self._closed = False
        self._failure = self._release_error = None
        self._buffers = set()
        self._in_flight = {}
        self._pending_host_copy = None
        self._stream = ct.c_void_p()
        self._counts = dict(
            device_allocations=0,
            allocated_bytes=0,
            freed_bytes=0,
            uploaded_bytes=0,
            downloaded_bytes=0,
            launched_kernels=0,
            completed_kernels=0,
            stream_synchronizations=0,
            upload_synchronizations=0,
        )
        self._lib = _load(library)
        lib = self._lib
        integer, ptr = ct.c_int, ct.c_void_p
        int_out, ptr_out = ct.POINTER(integer), ct.POINTER(ptr)
        _bind(lib, "musaGetErrorString", ct.c_char_p, integer)
        _bind(lib, "musaGetDeviceCount", integer, int_out)
        _bind(lib, "musaGetDevice", integer, int_out)
        _bind(lib, "musaSetDevice", integer, integer)
        _bind(lib, "musaStreamCreateWithFlags", integer, ptr_out, ct.c_uint)
        _bind(lib, "musaStreamDestroy", integer, ptr)
        _bind(lib, "musaStreamSynchronize", integer, ptr)
        _bind(lib, "musaMalloc", integer, ptr_out, ct.c_size_t)
        _bind(lib, "musaFree", integer, ptr)
        _bind(lib, "musaMemcpy", integer, ptr, ptr, ct.c_size_t, integer)
        self.device_count = self._integer(lib.musaGetDeviceCount)
        if device_index >= self.device_count:
            raise MusaError(
                f"MUSA device {device_index} unavailable ({self.device_count} visible devices)"
            )
        self._info = dict(
            backend=self.backend,
            library=os.fsdecode(lib._name),
            device_index=device_index,
            device_count=self.device_count,
        )
        try:
            with self._selected():
                for name, field in (
                    ("musaRuntimeGetVersion", "runtime_version"),
                    ("musaDriverGetVersion", "driver_version"),
                ):
                    if hasattr(lib, name):
                        self._info[field] = self._integer(_bind(lib, name, integer, int_out))
                if hasattr(lib, "musaDeviceGetPCIBusId"):
                    bus = ct.create_string_buffer(32)
                    get_bus = _bind(
                        lib,
                        "musaDeviceGetPCIBusId",
                        integer,
                        ct.POINTER(ct.c_char),
                        integer,
                        integer,
                    )
                    self._call(get_bus, bus, len(bus), device_index)
                    self._info["pci_bus_id"] = bus.value.decode("ascii")
                self._call(
                    lib.musaStreamCreateWithFlags, ct.byref(self._stream), 1
                )  # musaStreamNonBlocking
                if not self._stream.value:
                    raise MusaError("musaStreamCreateWithFlags returned a null stream")
                self._info.update(_device_identity())
        except BaseException:
            if self._stream.value:
                self.close()
            raise

    @property
    def device_index(self):
        return self._device_index

    @property
    def closed(self):
        return self._closed

    @property
    def stream_ptr(self):
        with self._lock:
            self._require_open()
            return self._stream.value

    def _call(self, fn, *args):
        status = fn(*args)
        if status:
            detail = self._lib.musaGetErrorString(status) or b"unknown MUSA status"
            raise MusaError(
                f"{fn.__name__} failed on device {self.device_index} "
                f"({status}: {detail.decode('utf-8', 'replace')})"
            )

    def _integer(self, function):
        result = ct.c_int()
        self._call(function, ct.byref(result))
        return result.value

    @contextmanager
    def _selected(self):
        previous = self._integer(self._lib.musaGetDevice)
        if previous == self.device_index:
            yield
        else:
            self._call(self._lib.musaSetDevice, self.device_index)
            try:
                yield
            finally:
                self._call(self._lib.musaSetDevice, previous)

    def _require_open(self):
        if self._closed or self._release_error:
            raise MusaError("MUSA runtime is closed or has an unresolved release")
        if self._failure:
            raise MusaError(f"MUSA runtime failed previously: {self._failure}")

    @contextmanager
    def _operation(self):
        with self._lock:
            self._require_open()
            with self._selected():
                yield

    def _require_array(self, value):
        self._require_open()
        if value not in self._buffers or value._closed:
            raise MusaError("MUSA array is closed or belongs to a different runtime")

    def empty(self, shape, dtype):
        shape, dtype, nbytes = _layout(shape, dtype)
        with self._operation():
            pointer = ct.c_void_p()
            buffer = MusaArray(self, shape, dtype, nbytes, 0)
            self._buffers.add(buffer)
            try:
                if nbytes:
                    self._call(self._lib.musaMalloc, ct.byref(pointer), nbytes)
                    if not pointer.value:
                        raise MusaError("musaMalloc returned a null address for a nonempty array")
                buffer._pointer = pointer.value or 0
            except BaseException:
                self._buffers.remove(buffer)
                raise
            self._counts["device_allocations"] += int(bool(nbytes))
            self._counts["allocated_bytes"] += nbytes
            return buffer

    def from_numpy(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError("from_numpy requires a NumPy array")
        with self._lock:
            buffer = self.empty(value.shape, value.dtype)
            try:
                buffer.copy_from(value)
            except BaseException:
                buffer.close()
                raise
            return buffer

    def _sync(self):
        self._call(self._lib.musaStreamSynchronize, self._stream)
        self._counts["stream_synchronizations"] += 1
        self._counts["completed_kernels"] = self._counts["launched_kernels"]
        self._in_flight.clear()

    def _finish_host_copy(self):
        if self._pending_host_copy is not None:
            self._call(self._lib.musaStreamSynchronize, None)
            self._pending_host_copy = None

    def _drain_for_release(self):
        self._finish_host_copy()
        if self._in_flight:
            self._sync()

    def synchronize(self):
        with self._operation():
            try:
                self._sync()
            except MusaError as exc:
                self._failure = str(exc)
                raise

    sync = synchronize

    def _copy(self, buffer, host, *, upload):
        with self._operation():
            self._require_array(buffer)
            if not buffer.nbytes:
                return
            try:
                if self._in_flight:
                    self._sync()
                # NumPy owns pageable host memory. Use the synchronous host
                # API; a pageable H2D upload may return after internal staging.
                # Finish that copy on stream zero before our nonblocking stream
                # can consume it. D2H has completed when musaMemcpy returns.
                self._pending_host_copy = host
                if upload:
                    self._call(
                        self._lib.musaMemcpy, buffer._pointer, host.ctypes.data, buffer.nbytes, 1
                    )
                    self._finish_host_copy()
                    self._counts["upload_synchronizations"] += 1
                    self._counts["uploaded_bytes"] += buffer.nbytes
                else:
                    self._call(
                        self._lib.musaMemcpy, host.ctypes.data, buffer._pointer, buffer.nbytes, 2
                    )
                    self._pending_host_copy = None
                    self._counts["downloaded_bytes"] += buffer.nbytes
            except MusaError as exc:
                self._failure = str(exc)
                raise

    def launch(self, compiled, *args):
        """Enqueue a MusaKernel compiled from Tyx on this runtime's device and stream."""
        with self._operation():
            for arg in args:
                if isinstance(arg, MusaArray):
                    self._require_array(arg)
                elif not isinstance(arg, (int, float, bool, np.number, np.bool_)):
                    raise TypeError("MUSA kernels accept owned device arrays and numeric scalars")
            launch = compiled.launch
            key = id(compiled)
            already_pending = key in self._in_flight
            self._in_flight[key] = compiled
            try:
                submitted = launch(*args, stream_ptr=self._stream.value, device=self.device_index)
            except BaseException as exc:
                self._failure = str(exc) or type(exc).__name__
                raise
            if submitted is False:  # parallel Tyx root with zero logical extent
                if not already_pending:
                    del self._in_flight[key]
                return
            self._counts["launched_kernels"] += 1

    def _quarantine(self, exc):
        self._release_error = str(exc) or type(exc).__name__
        _QUARANTINED[id(self)] = self

    def _release(self, buffer):
        if buffer._pointer:
            self._call(self._lib.musaFree, buffer._pointer)
        self._buffers.remove(buffer)
        buffer._pointer, buffer._closed = 0, True
        self._counts["freed_bytes"] += buffer.nbytes

    def _free(self, buffer):
        with self._lock:
            if buffer._closed:
                return
            if self._release_error:
                raise MusaError(f"Unresolved MUSA cleanup: {self._release_error}")
            if buffer not in self._buffers:
                raise MusaError("MUSA allocation is not owned by this runtime")
            try:
                with self._selected():
                    self._drain_for_release()
                    self._release(buffer)
            except BaseException as exc:
                self._quarantine(exc)
                raise

    def report(self):
        with self._lock:
            return dict(
                self._info,
                **self._counts,
                closed=self.closed,
                failure=self._failure,
                release_error=self._release_error,
                live_buffers=len(self._buffers),
                live_payload_bytes=self._counts["allocated_bytes"] - self._counts["freed_bytes"],
            )

    def close(self):
        with self._lock:
            if self._release_error:
                raise MusaError(f"Unresolved MUSA cleanup: {self._release_error}")
            if self._closed:
                return
            try:
                with self._selected():
                    self._drain_for_release()
                    for buffer in tuple(self._buffers):
                        self._release(buffer)
                    self._call(self._lib.musaStreamDestroy, self._stream)
                    self._stream = ct.c_void_p()
                    self._closed = True
            except BaseException as exc:
                self._quarantine(exc)
                raise

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *args):
        self.close()
