"""Vulkan attachment output to independently owned Torch CUDA tensor views."""

import ctypes as ct
from dataclasses import dataclass

from ..outputs import Channel
from . import BackendUnavailableError
from ._native_camera_batch import ResidentBatchView, batch_views
from ._native_shared_matrices import _Export, _cu, _require_version, import_external_slot


class _OutputInfo(ct.Structure):
    _fields_ = [('abi_version', ct.c_uint32), ('phase', ct.c_uint32)] + [
        (name, ct.c_uint64) for name in ('capacity_bytes', 'batches', 'views', 'image_bytes')]


def bind_device_output(lib):
    _require_version(lib, 'pr_resident_device_output_abi_version', 'shared device image output')
    ptr = ct.c_void_p
    functions = {
        'pr_resident_device_output_export': [ptr, ct.c_uint64, ct.POINTER(_Export)],
        'pr_resident_device_output_drop_exports': [ptr],
        'pr_resident_render_shared_batch_device': [ptr, ct.POINTER(ResidentBatchView), ct.c_uint32, ct.c_size_t],
        'pr_resident_device_output_return': [ptr],
        'pr_resident_device_output_retire': [ptr],
        'pr_resident_device_output_info': [ptr, ct.POINTER(_OutputInfo)],
    }
    for name, args in functions.items():
        f = getattr(lib, name, None)
        if f is None:
            raise BackendUnavailableError(f'native shared device image output ABI is missing {name}')
        f.argtypes, f.restype = args, ct.c_int


def _plane_views(torch, slab, views, spans):
    result = []
    for view, (start, end, pixels, channels, mask) in zip(views, spans, strict=True):
        offset = start
        planes = {}
        for bit, (channel, stride) in enumerate(zip(Channel, (4, 4, 4, 1), strict=True)):
            if not mask & (1 << bit):
                continue
            segment = slab.narrow(0, offset, pixels * stride)
            if channel is Channel.RGB:
                tensor = segment.view(view.height, view.width, 4)[..., :3]
            else:
                if channel is Channel.FORWARD_DEPTH:
                    segment = segment.view(torch.float32)
                elif channel is Channel.INSTANCE_ID:
                    segment = segment.view(torch.uint32)
                tensor = segment.view(view.height, view.width)
            planes[channel] = tensor
            offset += pixels * stride
        if offset != end:
            raise ValueError('device attachment layout differs from the native batch extent')
        result.append({ch: planes[ch] for ch in channels})
    return tuple(result)


@dataclass
class _Handoff:
    stream: object
    event: object
    slab: object
    released: bool | None = None
    ready_wait_queued: bool = False
    writable_signal_queued: bool = False
    recorded: bool = False


class DeviceImageOutput:
    """Single growable export slot. Each returned batch holds a separate Torch allocation."""

    def __init__(self, owner):
        bind_device_output(owner._lib)
        self.owner = owner
        self.exchange = owner._shared_exchange
        self.driver = owner._shared_identity[0]
        self._imports = []
        self._export = self._handoff = None
        self._capacity = 0
        self._mapped = False
        self._copies = self._bytes = self._waits = 0
        self._created = self._destroyed = self._generations = 0
        self._native = {}

    def _call(self, symbol, *args):
        self.exchange._call(symbol, *args)

    def _native_info(self):
        value = _OutputInfo()
        self._call('pr_resident_device_output_info', ct.byref(value))
        if value.abi_version != 1:
            raise RuntimeError('unsupported native device output diagnostics ABI')
        self._native = {name: int(getattr(value, name)) for name, _ in value._fields_}
        return self._native

    def diagnostics(self):
        if self.owner._handle is not None:
            self._native_info()
        return dict(self._native, attachment_d2d_copies=self._copies, attachment_d2d_bytes=self._bytes,
                    materialization_d2d_copies=0, materialization_d2d_bytes=0,
                    image_h2d_bytes=0, image_d2h_bytes=0, return_event_waits=self._waits,
                    imports_created=self._created, imports_destroyed=self._destroyed,
                    imports_live=len(self._imports), slot_generations=self._generations,
                    exports_open=self._export is not None, return_pending=self._handoff is not None)

    def _drop_exports(self):
        if self._export is not None:
            self._call('pr_resident_device_output_drop_exports')
            self._export = None

    def _retire(self):
        # Vulkan may own a signaled writable semaphore after the last return.
        # It can retire that allocation after every CUDA reference has gone.
        self._mapped = False
        if self._imports:
            with self.exchange._current():
                while self._imports:
                    destroy, handle = self._imports[-1]
                    _cu(destroy, handle)
                    self._imports.pop()
                    self._destroyed += 1
        self._drop_exports()
        if self._capacity:
            self._call('pr_resident_device_output_retire')
            self._capacity = 0

    def _reserve(self, size):
        if self._mapped and self._capacity >= size:
            return
        self._retire()
        exported = _Export()
        self._call('pr_resident_device_output_export', size, ct.byref(exported))
        self._export = exported
        self._capacity = size
        self._generations += 1
        try:
            if (not all((exported.memory, exported.writable, exported.ready)) or
                    exported.buffer_bytes != size or exported.allocation_bytes < size):
                raise RuntimeError('device output export has invalid capacity or handles')
            with self.exchange._current():
                try:
                    self._address, self._writable, self._ready = import_external_slot(
                        self.driver, exported, self._imports)
                finally:
                    self._created += len(self._imports)
            self._drop_exports()
        except BaseException as error:
            try:
                self._retire()
            except BaseException as cleanup:
                raise BaseExceptionGroup('device image import and rollback failed', [error, cleanup])
            raise
        self._mapped = True

    def _wait_ready(self, handoff, stream):
        if not handoff.ready_wait_queued:
            d = self.driver
            _cu(d.cuWaitExternalSemaphoresAsync, [self._ready], [d.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS()], 1, stream)
            handoff.ready_wait_queued = True

    def _signal(self, handoff, stream):
        if not handoff.writable_signal_queued:
            d = self.driver
            _cu(d.cuSignalExternalSemaphoresAsync, [self._writable],
                [d.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS()], 1, stream)
            handoff.writable_signal_queued = True
        if not handoff.recorded:
            handoff.event.record(handoff.stream)
            handoff.recorded = True

    def _finish(self):
        handoff = self._handoff
        if handoff is None:
            return
        if handoff.released is None:
            # A native failure after submission can still release the buffer.
            # Waiting for ready after a pre-submit error would deadlock.
            phase = self._native_info()['phase']
            if phase == 3:
                raise TimeoutError("Vulkan attachment fence still pending; CUDA image handoff retained")
            handoff.released = phase == 2
        if handoff.released:
            if not handoff.recorded:
                with self.exchange._current():
                    stream = self.driver.CUstream(handoff.stream.cuda_stream)
                    self._wait_ready(handoff, stream)
                    self._signal(handoff, stream)
            # This event covers the signal, without waiting for later consumer work.
            handoff.event.synchronize()
            self._waits += 1
            self._call('pr_resident_device_output_return')
        self._handoff = None

    def capture(self, lease, rows, consumer_stream):
        import torch

        target = torch.device('cuda', self.owner._shared_cuda_device)
        if (not isinstance(consumer_stream, torch.cuda.Stream) or consumer_stream.device != target
                or consumer_stream == torch.cuda.default_stream(target)):
            raise ValueError('device image output requires an explicit non-default stream on the matched CUDA device')
        with self.exchange.capturing(lease, camera_count=len(rows)):
            views, spans, size = batch_views(self.owner, rows, device_output=True)
            with torch.cuda.device(target), torch.cuda.stream(consumer_stream):
                self._finish()
                self._reserve(size)
                slab = torch.empty(size, dtype=torch.uint8, device=target)
                result = _plane_views(torch, slab, views, spans)
                handoff = _Handoff(consumer_stream, torch.cuda.Event(), slab)
                self._handoff = handoff
                try:
                    self._call('pr_resident_render_shared_batch_device', views, len(views), size)
                    handoff.released = True
                    self.owner._has_captured = True
                    with self.exchange._current():
                        stream = self.driver.CUstream(consumer_stream.cuda_stream)
                        self._wait_ready(handoff, stream)
                        _cu(self.driver.cuMemcpyDtoDAsync, slab.data_ptr(), self._address, size, stream)
                        self._copies += 1
                        self._bytes += size
                        self._signal(handoff, stream)
                except TimeoutError:
                    # Keep the export and its imports until a later close can
                    # establish that Vulkan actually signaled the ready event.
                    raise
                except BaseException as error:
                    try:
                        self._finish()
                    except BaseException as cleanup:
                        raise BaseExceptionGroup('device image capture and ownership return failed', [error, cleanup])
                    raise
                return result

    def close(self):
        self._finish()
        self._retire()
