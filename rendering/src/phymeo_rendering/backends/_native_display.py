"""Optional native resident display extension; default clip never probes new ABI."""
import ctypes as ct
from . import BackendUnavailableError

SOURCE='https://github.com/KhronosGroup/ToneMapping/blob/b5a2eed5ddf6c2227090449399de9c7affb9e4c9/PBR_Neutral/pbrNeutral.glsl'


def configure_display(device, transform):
    if transform == 'clip':
        return {'display_transform':'clip', 'display_transform_abi':None}
    if transform != 'pbr-neutral':
        raise ValueError('display_transform must be clip or pbr-neutral')
    try:
        version=device._lib.pr_resident_display_transform_abi_version
        compatible=device._lib.pr_resident_display_transform_shader_compatible
        setter=device._lib.pr_resident_set_display_transform
    except AttributeError as error:
        raise BackendUnavailableError('native resident display-transform ABI unavailable; no clip fallback') from error
    version.argtypes=[]; version.restype=ct.c_uint32
    compatible.argtypes=[ct.c_void_p]; compatible.restype=ct.c_int
    setter.argtypes=[ct.c_void_p,ct.c_uint32]; setter.restype=ct.c_int
    if version()!=1:
        raise BackendUnavailableError('native resident display-transform ABI version unsupported')
    if compatible(device._handle)!=1:
        raise BackendUnavailableError('native resident fragment lacks display-transform ABI marker')
    if setter(device._handle,1)!=0:
        raise BackendUnavailableError(device._error())
    return {'display_transform':'pbr-neutral', 'display_transform_abi':1,
            'display_transform_source':SOURCE}
