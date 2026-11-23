"""Optional resident 4-sample color extension; the default never probes its ABI."""
import ctypes as ct
import hashlib
from . import BackendUnavailableError


class ResidentMsaaInfo(ct.Structure):
    _fields_=[(name,ct.c_uint32) for name in (
        'msaa_abi_version','requested_samples','color_samples','sensor_samples',
        'color_format','depth_format','resolve_format','output_format',
        'color_sample_counts','depth_sample_counts','max_width','max_height',
        'shader_compatible','standard_sample_locations','reserved0','reserved1')]+[
        (name,ct.c_uint64) for name in ('color_passes','center_passes','resolve_passes','target_rebuilds')]+[
        ('reserved64',ct.c_uint64*4)]


def msaa_info(value):
    if value.msaa_abi_version!=1:
        raise RuntimeError('unsupported resident MSAA info ABI')
    return {name:int(getattr(value,name)) for name,_ in ResidentMsaaInfo._fields_
            if not name.startswith('reserved')}


def configure_msaa(device,samples):
    if type(samples) is not int:
        raise TypeError('msaa_samples must be a strict integer')
    if samples not in (1,4):
        raise ValueError('msaa_samples must be 1 or 4')
    result=dict(requested_samples=samples,actual_color_samples=samples,sensor_samples=1,
        resolve_space='none (single sample)' if samples==1 else 'display-linear then IEC sRGB once',
        sensor_sampling='deterministic pixel center')
    if samples==1:
        return result
    if ct.sizeof(ResidentMsaaInfo)!=128:
        raise BackendUnavailableError('resident MSAA info layout mismatch')
    try:
        version=device._lib.pr_resident_msaa_abi_version
        query=device._lib.pr_resident_msaa_info
        setter=device._lib.pr_resident_set_msaa
    except AttributeError as error:
        raise BackendUnavailableError('native resident MSAA ABI unavailable; no 1x fallback') from error
    version.argtypes=[]; version.restype=ct.c_uint32
    query.argtypes=[ct.c_void_p,ct.POINTER(ResidentMsaaInfo)]; query.restype=ct.c_int
    setter.argtypes=[ct.c_void_p,ct.c_uint32,ct.POINTER(ct.c_uint32),ct.c_size_t,
                     ct.POINTER(ct.c_uint32),ct.c_size_t]; setter.restype=ct.c_int
    if version()!=1:
        raise BackendUnavailableError('unsupported native resident MSAA ABI')
    capabilities=ResidentMsaaInfo()
    if query(device._handle,ct.byref(capabilities)):
        raise BackendUnavailableError(device._error())
    if (capabilities.msaa_abi_version!=1 or not capabilities.shader_compatible or
        not capabilities.color_sample_counts&4 or not capabilities.depth_sample_counts&4 or
        not capabilities.max_width or not capabilities.max_height):
        raise BackendUnavailableError('native MSAA format/sample count or shader support unavailable')
    shaders=[]; hashes={}
    for stage in ('vert','frag'):
        path=device.shader_directory/f'resident_msaa_resolve.{stage}.spv'
        try:
            with path.open('rb') as stream: raw=stream.read(4*1024*1024+1)
        except OSError as error:
            raise BackendUnavailableError(f'native MSAA resolve shader unavailable: {path}') from error
        if len(raw)<20 or len(raw)>4*1024*1024 or len(raw)%4 or raw[:4]!=b'\x03\x02\x23\x07':
            raise BackendUnavailableError(f'invalid native MSAA resolve SPIR-V: {path}')
        shaders.append((ct.c_uint32*(len(raw)//4)).from_buffer_copy(raw))
        hashes['msaa_resolve_'+stage]=hashlib.sha256(raw).hexdigest()
    if setter(device._handle,samples,shaders[0],ct.sizeof(shaders[0]),shaders[1],ct.sizeof(shaders[1])):
        raise BackendUnavailableError(device._error())
    actual=ResidentMsaaInfo()
    if query(device._handle,ct.byref(actual)):
        raise BackendUnavailableError(device._error())
    if actual.requested_samples!=4 or actual.color_samples!=4 or actual.sensor_samples!=1:
        raise BackendUnavailableError('native MSAA did not retain the requested sample contract')
    device._msaa_info_function=query
    device._msaa_info=msaa_info(actual)
    device.shader_sha256.update(hashes)
    return result
