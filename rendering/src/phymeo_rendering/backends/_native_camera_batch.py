"""Optional Vulkan ABI for recording several compact camera views in one submission."""

import ctypes as ct

from ..outputs import Channel
from . import BackendUnavailableError
from ._native_device import Parameters
from ._native_pixels import _image_planes


MAX_VIEWS = 16
MAX_TOTAL_PIXELS = 4096 * 4096
FIXED_CAMERA = 0xffffffff


class ResidentBatchView(ct.Structure):
    _fields_ = [('params', Parameters), ('width', ct.c_uint32), ('height', ct.c_uint32),
                ('channel_mask', ct.c_uint32), ('camera_index', ct.c_uint32)]


def bind_camera_batch(lib):
    version = getattr(lib, 'pr_resident_shared_batch_abi_version', None)
    draw = getattr(lib, 'pr_resident_render_shared_batch', None)
    if version is None or draw is None:
        raise BackendUnavailableError('native shared camera batch ABI is unavailable; rebuild the resident library')
    version.argtypes = []
    version.restype = ct.c_uint32
    actual = version()
    if actual != 1:
        raise BackendUnavailableError(f'unsupported native shared camera batch ABI {actual}')
    draw.argtypes = [ct.c_void_p, ct.POINTER(ResidentBatchView), ct.c_uint32, ct.POINTER(ct.c_uint8), ct.c_size_t]
    draw.restype = ct.c_int
    return draw


def batch_views(device, rows, *, device_output=False):
    if not 1 <= len(rows) <= MAX_VIEWS:
        raise ValueError(f'camera batch requires 1..{MAX_VIEWS} views')
    views = (ResidentBatchView * len(rows))()
    spans = []
    total_bytes = total_pixels = 0
    for i, (params, width, height, channels, slot) in enumerate(rows):
        device._validate_capture_view(params, width, height)
        if not channels or any(not isinstance(ch, Channel) for ch in channels) or len(set(channels)) != len(channels):
            raise ValueError('batch views require unique native channels')
        mask = sum(1 << bit for bit, ch in enumerate(Channel) if ch in channels)
        device._require_compact_mask(params, mask, require_host_export=not device_output)
        if slot is not None and (type(slot) is not int or not 0 <= slot < device._shared_camera_capacity):
            raise ValueError('invalid shared camera index')
        pixels = width * height
        total_pixels += pixels
        if total_pixels > MAX_TOTAL_PIXELS:
            raise ValueError('camera batch total pixel budget exceeded')
        size = pixels * sum(b for bit, b in enumerate((4, 4, 4, 1)) if mask & (1 << bit))
        if device_output:
            total_bytes = (total_bytes + 3) & ~3  # VkBufferImageCopy::bufferOffset
        views[i] = ResidentBatchView(params, width, height, mask, FIXED_CAMERA if slot is None else slot)
        spans.append((total_bytes, total_bytes + size, pixels, channels, mask))
        total_bytes += size
    return views, spans, total_bytes


def render_shared_batch(device, lease, rows):
    if device._shared_exchange is None:
        raise RuntimeError('shared camera batch requires an open CUDA matrix exchange')
    with device._shared_exchange.capturing(lease, camera_count=len(rows)):
        views, spans, total_bytes = batch_views(device, rows)
        if device._shared_batch_draw is None:
            device._shared_batch_draw = bind_camera_batch(device._lib)
        output = device._reserve_output(total_bytes)
        status = device._shared_batch_draw(device._handle, views, len(rows), output, total_bytes)
        if status:
            error = device._error()
            if status == -2:
                raise TimeoutError(error)
            raise RuntimeError(error)
        device._has_captured = True
        device._shared_batch_calls += 1
        device._shared_batch_views += len(rows)
        device._shared_batch_bytes += total_bytes
        data = memoryview(output).cast('B')
        return tuple(_image_planes(data[start:end], pixels, channels, mask)
                     for start, end, pixels, channels, mask in spans)
