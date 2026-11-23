"""Optional same-exposure GPU guide and bounded spatial display contracts."""
import ctypes as ct
import math

from . import BackendUnavailableError


class SpatialInfo(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        'spatial_abi_version', 'supported', 'shader_compatible', 'enabled',
        'last_capture_filtered', 'kernel_radius', 'guide_code_limit', 'reserved',
    )] + [(name, ct.c_uint64) for name in (
        'filtered_rgb_captures', 'rgb_captures', 'guide_readbacks', 'guide_readback_bytes',
    )]


class SettlingInfo(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        'settling_abi_version', 'enabled', 'last_batch', 'last_capture_filtered',
    )] + [('last_filtered_fraction', ct.c_float)] + [(name, ct.c_uint32) for name in (
        'last_filtered_primary_samples', 'guide_valid', 'reserved',
    )] + [(name, ct.c_uint64) for name in (
        'guide_draws', 'filtered_rgb_captures', 'filtered_primary_samples', 'filter_dispatches',
    )]


def settling_fraction(batch):
    """Registered FP64 schedule; device diagnostics record the actual FP32 push."""
    if type(batch) is not int:
        raise TypeError('settling batch must be a positive integer')
    if batch < 1:
        raise ValueError('settling batch must be positive')
    return 1.-math.sqrt((batch-1)/15.) if batch < 16 else 0.


def query_settling(device):
    query = getattr(device, '_settling_query', None)
    if query is not None and device._handle:
        value = SettlingInfo()
        if query(device._handle, ct.byref(value)):
            raise RuntimeError(device._error())
        device._settling_info = {
            name: getattr(value, name) for name, _ in SettlingInfo._fields_ if name != 'reserved'
        }
    value = getattr(device, '_settling_info', None)
    return dict(value) if value is not None else None


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
    declaration = dict(
        mode=mode, scope='first-rgb-batch-only', last_capture_filtered=False,
        filtered_captures=0, kernel=[1, 4, 6, 4, 1], depth_relative_limit=.003,
        guide_code_limit=2, guide_source='same-exposure-gpu-direct-pass',
        guide_space='display-srgb-direct', raw_mean='unmodified scene-linear GPU mean',
        arithmetic='FP32 weighted sum in y-major order; CPU reference uses FP64',
        cost='one additional RGBA8 target; first-batch extra GPU direct/environment raster pass with its shadow queries; 25 taps in first-batch compute; later batches do not draw the guide; debug guide readback only on explicit call',
    )
    if mode == 'guided_settling':
        if ct.sizeof(SettlingInfo) != 64:
            raise BackendUnavailableError('settling info ABI layout mismatch')
        try:
            query = device._lib.pr_resident_settling_info
        except AttributeError as error:
            raise BackendUnavailableError('native settling ABI unavailable; no fallback') from error
        query.argtypes = [ct.c_void_p, ct.POINTER(SettlingInfo)]
        query.restype = ct.c_int
        device._settling_query = query
        info = query_settling(device)
        if info['enabled'] != 1:
            raise BackendUnavailableError('requested settling mode is not enabled')
        declaration.update(
            scope='settling-first-16-rgb-batches', last_batch=0,
            last_filtered_fraction=0., scheduled_filtered_fraction=0.,
            last_filtered_primary_samples=0, filtered_primary_samples=0,
            schedule='w=1-sqrt((n-1)/15), one-based n; n>=16: w=0',
            cost='one retained guide draw per new RGB state; original mean dispatch then barrier and a display-only filter dispatch for n<16; no extra guide draw on static or sensor-only target rebuild',
            sample_count_scope='filtered_primary_samples counts samples represented in accepted filtered exposures, including their reused earlier batches; not new independent samples',
        )
    return declaration


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
