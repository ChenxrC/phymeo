"""Win32 CUDA ownership transfers for the resident Vulkan transform buffer."""
from contextlib import contextmanager
import ctypes as ct
from threading import RLock
import sys

from . import BackendUnavailableError
from ._native_device import Parameters


class _Export(ct.Structure):
    _fields_ = [(key, ct.c_uint64) for key in
                ("memory", "writable", "ready", "allocation_bytes", "buffer_bytes")]


def bind(lib):
    if ct.sizeof(_Export) != 40:
        raise BackendUnavailableError("resident shared export structure layout mismatch")
    version = lib.pr_resident_shared_abi_version
    version.argtypes, version.restype = [], ct.c_uint32
    if version() != 1:
        raise BackendUnavailableError("unsupported native shared-matrix ABI")
    ptr, u32 = ct.c_void_p, ct.c_uint32
    functions = (
        ("pr_resident_create_shared", ptr, [*lib.pr_create.argtypes, ct.POINTER(ct.c_uint8)]),
        ("pr_resident_shared_export", ct.c_int, [ptr, ct.POINTER(_Export)]),
        ("pr_resident_shared_drop_exports", ct.c_int, [ptr]),
        ("pr_resident_shared_begin", ct.c_int, [ptr]),
        ("pr_resident_shared_end", ct.c_int, [ptr]),
        ("pr_resident_render_shared_planes", ct.c_int,
         [ptr, ct.POINTER(Parameters), u32, u32, u32, ct.POINTER(ct.c_uint8), ct.c_size_t]),
    )
    for name, result, arguments in functions:
        f = getattr(lib, name)
        f.argtypes, f.restype = arguments, result


def _cu(operation, *args):
    status, *values = operation(*args)
    if int(status):
        raise RuntimeError(f"{operation.__name__} returned {status}")
    return values[0] if values else None


def cuda_identity(ordinal):
    if sys.platform != "win32":
        raise BackendUnavailableError("resident CUDA matrix sharing requires Win32 NT handles")
    try:
        from cuda.bindings import driver
    except ImportError as error:
        raise BackendUnavailableError("resident shared matrices require cuda-bindings") from error
    _cu(driver.cuInit, 0)
    device = _cu(driver.cuDeviceGet, ordinal)
    uuid = bytes(_cu(driver.cuDeviceGetUuid, device).bytes)
    if len(uuid) != 16:
        raise BackendUnavailableError("CUDA returned an invalid physical device UUID")
    return driver, device, uuid


class SharedMatrices:
    """One imported device-local SSBO, one active publication, multiple raster views.

    publish() expects trusted finite world and inverse transforms as contiguous
    column-major float32 records [N,32] or [N,2,4,4]. The explicit CUDA stream
    must already be ordered after the producer's last write. It must keep the
    source contents unchanged through publication. Loads and draws use the same
    instance order as the resident geometry upload.
    """
    def __init__(self, owner):
        self._owner = owner
        self._lock = RLock()
        self._driver, self._device, self._uuid = owner._shared_identity
        self._ordinal = owner._shared_cuda_device
        self._size = owner._instance_count * 128
        self._imports = []
        self._context = None
        self._closed = False
        self._broken = False
        self._raw_export = None
        self._opened = False
        self._need_end = False
        self._active = None
        self._source = self._stream = None
        self._copies = self._publications = self._captures = 0

    @contextmanager
    def _current(self):
        d = self._driver
        _cu(d.cuCtxPushCurrent, self._context)
        try:
            yield
        finally:
            _cu(d.cuCtxPopCurrent)

    def _call(self, name, *args):
        owner = self._owner
        result = getattr(owner._lib, name)(owner._handle, *args)
        if result == -2:
            raise TimeoutError(owner._error())
        if result:
            raise RuntimeError(owner._error())

    def open(self):
        with self._lock:
            if self._closed or self._broken:
                raise RuntimeError("resident CUDA exchange is closed or failed")
            if self._opened:
                return self
            if self._context is None:
                self._context = _cu(self._driver.cuDevicePrimaryCtxRetain, self._device)
            if self._raw_export is None:
                raw = _Export()
                self._call("pr_resident_shared_export", ct.byref(raw))
                self._raw_export = raw
            try:
                self._import(self._raw_export)
            except BaseException as error:
                # The native slot exports once. Keep the borrowed NT handles
                # for a retry; unwind all successful CUDA imports immediately.
                try:
                    with self._current():
                        self._unmap()
                except BaseException as cleanup:
                    self._broken = True
                    raise BaseExceptionGroup("resident import and CUDA rollback failed", [error, cleanup])
                raise
            self._opened = True
            return self

    def _unmap(self):
        while self._imports:
            fn, handle = self._imports[-1]
            _cu(fn, handle)
            self._imports.pop()

    def _import(self, raw):
        d = self._driver
        if (not all((raw.memory,raw.ready,raw.writable)) or
                raw.buffer_bytes != self._size or raw.allocation_bytes < self._size):
            raise RuntimeError("resident matrix export has invalid size or handles")
        with self._current():
            desc = d.CUDA_EXTERNAL_MEMORY_HANDLE_DESC()
            desc.type = d.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
            desc.handle.win32.handle = raw.memory
            desc.size = raw.allocation_bytes
            desc.flags = 1  # CUDA_EXTERNAL_MEMORY_DEDICATED
            memory = _cu(d.cuImportExternalMemory, desc)
            self._imports.append((d.cuDestroyExternalMemory, memory))
            mapping = d.CUDA_EXTERNAL_MEMORY_BUFFER_DESC()
            mapping.size = raw.buffer_bytes
            self._address = _cu(d.cuExternalMemoryGetMappedBuffer, memory, mapping)
            self._imports.append((d.cuMemFree, self._address))
            semaphores = []
            for handle in (raw.writable, raw.ready):
                desc = d.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC()
                desc.type = d.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32
                desc.handle.win32.handle = handle
                semaphore = _cu(d.cuImportExternalSemaphore, desc)
                self._imports.append((d.cuDestroyExternalSemaphore, semaphore))
                semaphores.append(semaphore)
            self._writable, self._ready = semaphores
        # CUDA imports retain their references; the exporting API owns and
        # closes the original three NT handles, even after a partial import.
        self._call("pr_resident_shared_drop_exports")
        self._raw_export = None

    def _validate_source(self, packed, producer_stream):
        import torch
        n = self._owner._instance_count
        device = torch.device("cuda", self._ordinal)
        if (not isinstance(packed, torch.Tensor) or packed.device != device or
                packed.dtype != torch.float32 or not packed.is_contiguous() or
                tuple(packed.shape) not in ((n,32),(n,2,4,4))):
            raise ValueError("expected contiguous float32 CUDA world/inverse records on the matched device")
        if (not isinstance(producer_stream, torch.cuda.Stream) or producer_stream.device != device or
                producer_stream == torch.cuda.default_stream(device)):
            raise ValueError("shared matrices require an explicit non-default stream on the matched CUDA device")

    def _finish(self):
        if self._need_end:
            self._active = None
            self._call("pr_resident_shared_end")
            self._need_end = False
            # The native fence covers the prior CUDA signal and every draw.
            self._source = self._stream = None

    @contextmanager
    def publish(self, packed, *, producer_stream):
        with self._lock:
            if not self._opened or self._closed or self._broken or self._active is not None or self._need_end:
                raise RuntimeError("shared matrix exchange is closed, failed or already publishing")
            self._validate_source(packed, producer_stream)
            d = self._driver
            stream = d.CUstream(producer_stream.cuda_stream)
            self._source, self._stream = packed, producer_stream
            try:
                with self._current():
                    _cu(d.cuWaitExternalSemaphoresAsync, [self._writable],
                        [d.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS()], 1, stream)
                    _cu(d.cuMemcpyDtoDAsync, self._address, packed.data_ptr(), self._size, stream)
                    self._copies += 1
                    _cu(d.cuSignalExternalSemaphoresAsync, [self._ready],
                        [d.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS()], 1, stream)
                self._call("pr_resident_shared_begin")
            except BaseException as error:
                self._broken = True
                if isinstance(error, TimeoutError):
                    # The acquire was submitted. A timed-out release is kept
                    # pending for close() to retry without destroying objects.
                    self._need_end = True
                    try:
                        self._finish()
                    except BaseException as cleanup:
                        raise BaseExceptionGroup("shared matrix acquire and release failed", [error, cleanup])
                raise
            self._active = token = object()
            self._need_end = True
            self._publications += 1
        try:
            yield token
        finally:
            with self._lock:
                if self._active is token:
                    self._finish()

    @contextmanager
    def capturing(self, token):
        with self._lock:
            if self._closed or self._active is None or self._active is not token:
                raise ValueError("shared raster requires this device's active matrix publication")
            yield
            self._captures += 1

    def diagnostics(self):
        with self._lock:
            return dict(cuda_ordinal=self._ordinal, device_uuid=self._uuid.hex(),
                        matrix_record_bytes=128, slot_bytes=self._size, publications=self._publications,
                        captures=self._captures, matrix_d2d_copies=self._copies,
                        matrix_d2d_bytes=self._copies*self._size, closed=self._closed,
                        imported_cuda_resources=len(self._imports), exports_open=self._raw_export is not None)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._broken = True
            self._finish()
            d = self._driver
            if self._context is not None:
                with self._current():
                    if self._stream is not None:
                        # An exception interrupted publication before Vulkan's
                        # acquire fence. Retain the source/imports until drained.
                        _cu(d.cuStreamSynchronize, d.CUstream(self._stream.cuda_stream))
                        self._source = self._stream = None
                    self._unmap()
            if self._raw_export is not None:
                self._call("pr_resident_shared_drop_exports")
                self._raw_export = None
            if self._context is not None:
                _cu(d.cuDevicePrimaryCtxRelease, self._device)
                self._context = None
            self._closed = True
