"""Optional same-exposure GPU guide and first-batch GPU spatial display contract."""
import ctypes as ct

from . import BackendUnavailableError


class SpatialInfo(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        'spatial_abi_version', 'supported', 'shader_compatible', 'enabled',
        'last_capture_filtered', 'kernel_radius', 'guide_code_limit', 'reserved',
    )] + [(name, ct.c_uint64) for name in (
        'filtered_rgb_captures', 'rgb_captures', 'guide_readbacks', 'guide_readback_bytes',
    )]


def configure_spatial(device, mode):
    if mode == 'none':
        return None
    if ct.sizeof(SpatialInfo) != 64:
        raise BackendUnavailableError('spatial info ABI layout mismatch')
    try:
        query = device._lib.pr_resident_spatial_info
        read = device._lib.pr_resident_read_spatial_guide
    except AttributeError as error:
        raise BackendUnavailableError('native spatial ABI unavailable; no fallback') from error
    query.argtypes = [ct.c_void_p, ct.POINTER(SpatialInfo)]
    query.restype = ct.c_int
    read.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
    read.restype = ct.c_int
    device._spatial_query = query
    device._spatial_read = read
    info = query_spatial(device)
    if info['enabled'] != 1 or info['shader_compatible'] != 1:
        raise BackendUnavailableError('guided spatial shader/setup unavailable; no fallback')
    return dict(
        mode='guided', scope='first-rgb-batch-only', last_capture_filtered=False,
        filtered_captures=0, kernel=[1, 4, 6, 4, 1], depth_relative_limit=.003,
        guide_code_limit=2, guide_source='same-exposure-gpu-direct-pass',
        guide_space='display-srgb-direct', raw_mean='unmodified scene-linear GPU mean',
        arithmetic='FP32 weighted sum in y-major order; CPU reference uses FP64',
        cost='one additional RGBA8 target; first-batch extra GPU direct/environment raster pass with its shadow queries; 25 taps in first-batch compute; later batches do not draw the guide; debug guide readback only on explicit call',
    )


def query_spatial(device):
    query = getattr(device, '_spatial_query', None)
    if query is not None and device._handle:
        value = SpatialInfo()
        if query(device._handle, ct.byref(value)):
            raise RuntimeError(device._error())
        device._spatial_info = {
            name: getattr(value, name) for name, _ in SpatialInfo._fields_ if name != 'reserved'
        }
    value = getattr(device, '_spatial_info', None)
    return dict(value) if value is not None else None
