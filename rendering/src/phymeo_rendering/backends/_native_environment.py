"""Native environment ABI packing; radiance preparation remains shared and explicit."""
import ctypes as ct
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
import struct
import sys


class ResidentEnvironment(ct.Structure):
    _fields_=[('side',ct.c_uint32),('mip_levels',ct.c_uint32),('reserved',ct.c_uint32*2),
              ('diffuse_sh',(ct.c_float*4)*9)]


class ResidentEnvironmentInfo(ct.Structure):
    _fields_=[(name,ct.c_uint32) for name in (
        'environment_abi_version','supported','shader_compatible','max_side','max_mip_levels',
        'side','mip_levels','loaded','format','cube_layers','sh_coefficients','axis_transform',
        'diffuse_basis')]+[('reserved',ct.c_uint32*3)]+[(name,ct.c_uint64) for name in (
        'max_rgba_bytes','source_rgba_bytes','image_upload_bytes','parameter_upload_bytes',
        'upload_calls','rgb_captures_with_environment','sky_draws','reserved64')]


@dataclass(frozen=True)
class EnvironmentUpload:
    config: ResidentEnvironment
    rgba: object
    diagnostics: dict


def pack_environment(data):
    """Bind one immutable preprocessed level-major six-face payload, without filtering."""
    from ..environment import NativeEnvironmentData
    if not isinstance(data,NativeEnvironmentData):
        raise TypeError('environment upload requires NativeEnvironmentData')
    if sys.byteorder!='little': raise ValueError('native environment ABI requires little endian')
    side=data.side; levels=data.mip_levels
    if type(side) is not int or not 2<=side<=1024 or side&(side-1):
        raise ValueError('native environment side must be power of two in [2,1024]')
    if type(levels) is not int or levels!=side.bit_length():
        raise ValueError('native environment requires the complete mip chain through side 1')
    expected=sum((side>>level)**2*6*16 for level in range(levels))
    if not isinstance(data.rgba_f32le,bytes) or len(data.rgba_f32le)!=expected or expected>128*1024*1024:
        raise ValueError('native environment RGBA byte length or budget is invalid')
    if not isinstance(data.diffuse_sh_f32le,bytes) or len(data.diffuse_sh_f32le)!=144:
        raise ValueError('native environment requires exactly nine float32 SH vec4 coefficients')
    for red,green,blue,alpha in struct.iter_unpack('<4f',data.rgba_f32le):
        if not all(math.isfinite(value) and 0<=value<=100 for value in (red,green,blue)) or alpha!=1.:
            raise ValueError('native environment RGB must be finite [0,100] and alpha exactly 1')
    config=ResidentEnvironment(side,levels)
    for index,row in enumerate(struct.iter_unpack('<4f',data.diffuse_sh_f32le)):
        if not all(math.isfinite(value) for value in row) or row[3]!=0.:
            raise ValueError('native environment SH must be finite signed coefficients with zero w padding')
        config.diffuse_sh[index][:]=row
    rgba=(ct.c_float*(expected//4)).from_buffer_copy(data.rgba_f32le)
    diagnostics=deepcopy(data.diagnostics)
    diagnostics.update(rgba_sha256=hashlib.sha256(data.rgba_f32le).hexdigest(),
        diffuse_sh_sha256=hashlib.sha256(data.diffuse_sh_f32le).hexdigest(),
        side=side,mip_levels=levels,rgba_bytes=expected,diffuse_sh_bytes=144,
        native_axis_transform='world xyz -> sampler xzy; SH basis remains world xyz',
        native_face_order=['+X','-X','+Y','-Y','+Z','-Z'],
        native_layout='level-major, then six faces, top-row-first RGBA32F little endian',
        native_roughness='perceptual roughness r -> mip LOD r*(levels-1); no second prefilter',
        native_diffuse='shared normalized real SH9 coefficients already include Lambert convolution / pi',
        native_environment_precedence='map replaces uniform scalar environment; rectangular panels and scene lights remain')
    return EnvironmentUpload(config,rgba,diagnostics)