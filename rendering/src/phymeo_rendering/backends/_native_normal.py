"""Explicit normal-map upload: linear data and a bounded derivative tangent frame.

Primary UV only; glTF TANGENT remains an explicitly unsupported import attribute.
For each mapped triangle the FP32 UV determinant must be well conditioned relative
 to its two products. Coordinates are bounded to +/-65536. Shader derivatives
preserve mirrored handedness, with geometric-normal fallback for singular screen
projections or zero sampled vectors. Strength is fixed at one.
"""
import ctypes as ct
from dataclasses import dataclass
import hashlib
import math

from ..assets import TexturedMaterial
from . import BackendUnavailableError, UnsupportedRequestError
from ._native_texture import ResidentTexture, ResidentUV


class ResidentNormalInfo(ct.Structure):
    _fields_=[(name,ct.c_uint32) for name in (
        'normal_abi_version','max_textures','max_width','max_height','max_layers',
        'texture_count','array_width','array_height','array_layers','uv_count',
        'bound_instances','loaded','sampler_filter','mip_levels','format','shader_compatible')]+[
        (name,ct.c_uint64) for name in ('max_array_bytes','source_rgba_bytes','padded_rgba_bytes',
        'metadata_upload_bytes','upload_calls','image_upload_bytes','rgb_captures_with_normals','reserved64')]


@dataclass(frozen=True)
class NormalUpload:
    records: object
    rgba: object
    uv: object
    slots: object
    diagnostics: dict


def normal_info(value):
    if value.normal_abi_version!=1: raise RuntimeError('unsupported resident normal-map info ABI')
    result={name:int(getattr(value,name)) for name,_ in ResidentNormalInfo._fields_ if name!='reserved64'}
    result['upload_scope']='normal image and metadata writes, including sentinel; excluded from geometry/color uploads'
    result['capture_scope']='successful RGB captures with bound normal maps; not visible pixel count'
    return result


def pack_normal_maps(scene,decoded,info=None):
    materials={m.id:m for m in scene.materials}
    bound=[i for i in scene.instances if isinstance(materials[i.material_id],TexturedMaterial) and materials[i.material_id].normal_map is not None]
    if not bound: return None
    mapped_meshes={i.mesh_id for i in bound}
    uv=(ResidentUV*sum(len(decoded[m.id].positions) for m in scene.meshes))()
    offset=0
    for asset in scene.meshes:
        mesh=decoded[asset.id]
        if asset.id in mapped_meshes:
            if mesh.texcoords is None: raise UnsupportedRequestError('normal map requires explicit UV texcoords')
            values=[]
            for j,pair in enumerate(mesh.texcoords):
                value=ResidentUV(*pair)
                if not all(math.isfinite(x) and abs(x)<=65536 for x in (value.u,value.v)):
                    raise UnsupportedRequestError('normal-map UV FP32 coordinates must be finite and within +/-65536')
                uv[offset+j]=value; values.append((float(value.u),float(value.v)))
            for ia,ib,ic in mesh.triangles:
                a,b,c=values[ia],values[ib],values[ic]
                u=(b[0]-a[0],b[1]-a[1]); v=(c[0]-a[0],c[1]-a[1])
                first,second=u[0]*v[1],u[1]*v[0]
                if not abs(first-second)>3.814697265625e-6*(abs(first)+abs(second)):
                    raise UnsupportedRequestError('normal-map UV triangle is singular or ill-conditioned after FP32 conversion')
        offset+=len(mesh.positions)
    caps=(info or {}).get('resident_normal_info') or {}
    sources=[]; indices={}
    for item in bound:
        source=materials[item.material_id].normal_map
        if source not in indices: indices[source]=len(sources); sources.append(source)
    if len(sources)>min(64,caps.get('max_textures',64),caps.get('max_layers',64)):
        raise UnsupportedRequestError('native normal texture count exceeds capacity')
    from ..texture_data import load_texture
    images=[]; width=height=total=0
    budget=min(128*1024*1024,caps.get('max_array_bytes',128*1024*1024))
    for source in sources:
        image=load_texture(source)
        if image.source_sha256!=source.sha256: raise ValueError('normal decoded source SHA mismatch')
        if image.width>min(2048,caps.get('max_width',2048)) or image.height>min(2048,caps.get('max_height',2048)):
            raise UnsupportedRequestError('native normal texture dimension exceeds capacity')
        width=max(width,image.width); height=max(height,image.height)
        if (width+2)*(height+2)*len(sources)*4>budget:
            raise UnsupportedRequestError('native padded normal array exceeds capacity')
        images.append(image); total+=image.width*image.height*4
    records=(ResidentTexture*len(sources))(); rgba=(ct.c_uint8*total)(); cursor=0; evidence=[]
    for slot,(source,image) in enumerate(zip(sources,images,strict=True)):
        records[slot]=ResidentTexture(image.width,image.height,cursor,0)
        packed=bytearray(b'\xff')*(image.width*image.height*4)
        for channel in range(3): packed[channel::4]=image.rgb8[channel::3]
        ct.memmove(ct.addressof(rgba)+cursor,bytes(packed),len(packed)); cursor+=len(packed)
        evidence.append(dict(uri=source.uri,source_sha256=source.sha256,rgb8_sha256=hashlib.sha256(image.rgb8).hexdigest(),width=image.width,height=image.height))
    slots=(ct.c_uint32*len(scene.instances))()
    for slot,item in enumerate(scene.instances):
        source=getattr(materials[item.material_id],'normal_map',None)
        slots[slot]=indices[source] if source is not None else 2**32-1
    diagnostics=dict(textures=evidence,strength=1.,encoding='linear RGB8 UNORM, normalize(2*c-1), glTF +Y',
        tangent_frame='primary UV derivatives; Gram-Schmidt tangent; mirrored handedness from dP/dv; no explicit TANGENT support',
        degenerate_policy='reject singular/ill-conditioned source UV; singular screen derivatives or zero sampled vector fall back to geometric shading normal',
        sampler='bilinear repeat, mip0, top-row-first; one wrap guard on each side',
        source_rgba_bytes=total,padded_array_bytes=(width+2)*(height+2)*len(sources)*4,
        uv_fp32_sha256=hashlib.sha256(bytes(uv)).hexdigest(),instance_slots=list(slots),
        geometry_and_sensor_policy='shading normal only; unchanged geometry, original shadow bias normal, pixel-center sensors')
    return NormalUpload(records,rgba,uv,slots,diagnostics)


def configure_normal_maps(device,upload):
    if upload is None: return None
    if not isinstance(upload,NormalUpload): raise TypeError('normal-map upload required')
    if ct.sizeof(ResidentNormalInfo)!=128: raise BackendUnavailableError('normal-map ABI layout mismatch')
    try:
        version=device._lib.pr_resident_normal_maps_abi_version
        query=device._lib.pr_resident_normal_info
        setter=device._lib.pr_resident_set_normal_maps
    except AttributeError as error:
        raise BackendUnavailableError('native normal-map ABI unavailable; no fallback') from error
    if not device._handle or not device._uploaded or device._has_captured or getattr(device,'_normal_maps_set',False):
        raise RuntimeError('normal maps require an uploaded, uncaptured session and one setter call')
    version.argtypes=[]; version.restype=ct.c_uint32
    query.argtypes=[ct.c_void_p,ct.POINTER(ResidentNormalInfo)]; query.restype=ct.c_int
    setter.argtypes=[ct.c_void_p,ct.POINTER(ResidentTexture),ct.c_uint32,ct.POINTER(ct.c_uint8),ct.c_size_t,
        ct.POINTER(ResidentUV),ct.c_uint32,ct.POINTER(ct.c_uint32),ct.c_uint32]; setter.restype=ct.c_int
    if version()!=1: raise BackendUnavailableError('unsupported normal-map ABI version')
    value=ResidentNormalInfo()
    if query(device._handle,ct.byref(value)): raise BackendUnavailableError(device._error())
    if value.normal_abi_version!=1 or not value.shader_compatible or not value.max_textures:
        raise BackendUnavailableError('native normal-map shader or format unavailable')
    if setter(device._handle,upload.records,len(upload.records),upload.rgba,len(upload.rgba),upload.uv,len(upload.uv),upload.slots,len(upload.slots)):
        raise BackendUnavailableError(device._error())
    device._normal_maps_set=True; device._normal_info_function=query
    if query(device._handle,ct.byref(value)): raise BackendUnavailableError(device._error())
    if value.loaded!=1 or value.upload_calls!=1: raise BackendUnavailableError('normal-map upload was not retained')
    device._normal_info=normal_info(value)
    return upload.diagnostics
