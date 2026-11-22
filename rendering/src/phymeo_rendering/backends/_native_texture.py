"""Bounded resident texture packing; shared decoding owns image interpretation."""
import ctypes as ct
from dataclasses import dataclass
import hashlib
import math

from ..assets import TexturedMaterial
from . import UnsupportedRequestError


class ResidentTexture(ct.Structure):
    _fields_=[(name,ct.c_uint32) for name in ('width','height','first_byte','reserved')]


class ResidentUV(ct.Structure):
    _fields_=[('u',ct.c_float),('v',ct.c_float)]


class ResidentTextureInfo(ct.Structure):
    _fields_=[(name,ct.c_uint32) for name in (
        'texture_abi_version','max_textures','max_width','max_height','max_layers',
        'texture_count','array_width','array_height','array_layers','uv_count',
        'bound_instances','loaded','sampler_filter','mip_levels','format','reserved')] + [
        (name,ct.c_uint64) for name in ('max_array_bytes','source_rgba_bytes',
        'padded_rgba_bytes','metadata_upload_bytes','upload_calls','image_upload_bytes',
        'rgb_captures_with_textures','reserved64')]


@dataclass(frozen=True)
class TextureUpload:
    records: object
    rgba: object
    uv: object
    slots: object
    diagnostics: dict


def pack_textures(scene, decoded, info=None):
    """Preserve mesh-local UV ordering, source rows and linear material factors.

    Only bound textures are decoded. Native repeats with bilinear filtering and
    one mip; a one-texel wrap guard on *each* side pads the array layers. The C
    implementation uploads those guards, so this function sends original RGBA8
    only. Plain vertices retain zero UV and instances use UINT32_MAX.
    """
    materials={material.id:material for material in scene.materials}
    bound=[item for item in scene.instances if isinstance(materials[item.material_id],TexturedMaterial)]
    if not bound:
        return None
    for item in bound:
        if decoded[item.mesh_id].texcoords is None:
            raise UnsupportedRequestError(f'textured instance {item.id} requires explicit mesh UV texcoords')
    capabilities=(info or {}).get('resident_texture_info') or {}
    max_count=min(64,capabilities.get('max_textures',64),capabilities.get('max_layers',64))
    max_width=min(2048,capabilities.get('max_width',2048))
    max_height=min(2048,capabilities.get('max_height',2048))
    max_bytes=min(128*1024*1024,capabilities.get('max_array_bytes',128*1024*1024))
    textures=[]; by_key={}
    for item in bound:
        source=materials[item.material_id].base_color_map
        key=(source.uri,source.sha256)
        if key not in by_key:
            by_key[key]=len(textures); textures.append(source)
    if len(textures)>max_count:
        raise UnsupportedRequestError(f'native texture count exceeds {max_count}')
    # This validates FP32 before file decode and before any native allocation.
    uv=(ResidentUV*sum(len(decoded[m.id].positions) for m in scene.meshes))()
    offset=0
    for mesh_asset in scene.meshes:
        mesh=decoded[mesh_asset.id]
        values=mesh.texcoords
        if values is not None:
            if len(values)!=len(mesh.positions): raise ValueError('texture UV count must equal vertex count')
            for index,value in enumerate(values):
                uv[offset+index]=ResidentUV(*value)
                if not math.isfinite(uv[offset+index].u) or not math.isfinite(uv[offset+index].v):
                    raise ValueError('native texture UV FP32 conversion is nonfinite')
        offset+=len(mesh.positions)
    from ..texture_data import load_texture
    images=[]; width=height=0; raw_size=0
    for source in textures:
        image=load_texture(source)
        if image.source_sha256!=source.sha256:
            raise ValueError('decoded texture source SHA does not match declaration')
        if image.width>max_width or image.height>max_height:
            raise UnsupportedRequestError(f'native texture dimension exceeds {max_width}x{max_height}')
        width=max(width,image.width); height=max(height,image.height)
        if (width+2)*(height+2)*len(textures)*4>max_bytes:
            raise UnsupportedRequestError(f'native padded texture array exceeds {max_bytes} bytes')
        images.append(image); raw_size+=image.width*image.height*4
    records=(ResidentTexture*len(images))(); rgba=(ct.c_uint8*raw_size)(); cursor=0
    texture_rows=[]
    for index,(source,image) in enumerate(zip(textures,images,strict=True)):
        records[index]=ResidentTexture(image.width,image.height,cursor,0)
        pixel_count=image.width*image.height
        # Preserve encoded RGB exactly; opacity is explicit, never discarded.
        packed=bytearray(pixel_count*4)
        packed[0::4]=image.rgb8[0::3]; packed[1::4]=image.rgb8[1::3]
        packed[2::4]=image.rgb8[2::3]; packed[3::4]=b'\xff'*pixel_count
        ct.memmove(ct.addressof(rgba)+cursor,bytes(packed),len(packed))
        texture_rows.append(dict(uri=source.uri,source_sha256=source.sha256,
            width=image.width,height=image.height,first_byte=cursor,
            rgb8_sha256=hashlib.sha256(image.rgb8).hexdigest()))
        cursor+=len(packed)
    slots=(ct.c_uint32*len(scene.instances))()
    for index,item in enumerate(scene.instances):
        material=materials[item.material_id]
        source=material.base_color_map if isinstance(material,TexturedMaterial) else None
        slots[index]=by_key[(source.uri,source.sha256)] if source is not None else 2**32-1
    diagnostics=dict(textures=texture_rows,source_rgba_bytes=raw_size,
        padded_array_bytes=(width+2)*(height+2)*len(images)*4,
        array_width=width+2,array_height=height+2,array_layers=len(images),
        uv_count=len(uv),instance_slots=list(slots),
        uv_fp32_sha256=hashlib.sha256(bytes(uv)).hexdigest(),
        image_encoding='original top-row-first sRGB8 RGB plus opaque alpha; no flip or re-encoding',
        material_factor='linear base_color multiplied after sRGB texel decode in native shader',
        sampler='bilinear repeat, mip level 0, one wrap guard texel on each side; no resizing',
        admission='native limits and device-reported lower limits; no alternate renderer')
    return TextureUpload(records,rgba,uv,slots,diagnostics)