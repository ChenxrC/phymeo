"""Explicit-GPU resident texture oracle; no scene simulation or image editing.

PYTHONPATH=rendering/src python -B rendering/examples/native_resident_texture_smoke.py \
  --library BUILD/phymeo_native.dll --device-index 1 --output NEW_DIRECTORY

Checks hardware sRGB decoding before bilinear filtering, linear RGB factors,
repeat at negative/boundary UV, separate array layers of different original
sizes, perspective UV interpolation, source pointer lifetime and teardown.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import gc
import hashlib
import json
import math
from pathlib import Path
import struct
from phymeo_rendering.backends._native_resident import (
    ResidentDevice, ResidentVertex, ResidentMesh, ResidentInstance,
)
from phymeo_rendering.backends._native_device import Parameters

class Texture(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ("width", "height", "first_byte", "reserved")]
class UV(ct.Structure):
    _fields_ = [("u", ct.c_float), ("v", ct.c_float)]
class TextureInfo(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "texture_abi_version", "max_textures", "max_width", "max_height", "max_layers",
        "texture_count", "array_width", "array_height", "array_layers", "uv_count",
        "bound_instances", "loaded", "sampler_filter", "mip_levels", "format", "reserved")] + [
        (name, ct.c_uint64) for name in ("max_array_bytes", "source_rgba_bytes", "padded_rgba_bytes",
        "metadata_upload_bytes", "upload_calls", "image_upload_bytes", "rgb_captures_with_textures", "reserved64")]

FACTOR = (0.6, 0.8, 0.4)
FIRST = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255))
SECOND = ((48, 144, 224),)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def decode(x):
    x = x / 255.0
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


def encode(x):
    x = min(1.0, max(0.0, x))
    return round(255 * (12.92*x if x <= 0.0031308 else 1.055*x**(1/2.4)-0.055))


def sampled(image, width, height, uv):
    # Independent repeat bilinear oracle: decode EACH source texel first.
    x, y = (uv[0] % 1)*width-0.5, (uv[1] % 1)*height-0.5
    ix, iy = math.floor(x), math.floor(y)
    ax, ay = x-ix, y-iy
    return tuple(sum(decode(image[((iy+dy) % height)*width+(ix+dx) % width][k]) * wx * wy
                     for dx, wx in ((0, 1-ax), (1, ax)) for dy, wy in ((0, 1-ay), (1, ay)))
                 for k in range(3))


def lit_color(texel, x, tilt):
    normal = (-tilt, 0.0, 1.0)
    length = math.hypot(*normal)
    n = tuple(v / length for v in normal)
    length = math.hypot(x, 1.0)
    v = (-x/length, 0.0, 1/length)
    h0 = (v[0], 0.0, v[2]+1)
    length = math.hypot(*h0)
    h = tuple(value/length for value in h0)
    nv, nl = sum(a*b for a,b in zip(n,v)), n[2]
    vh = sum(a*b for a,b in zip(v,h))
    fresnel = 0.04 + 0.96*(1-vh)**5
    # roughness=1: GGX D=1/pi; exact Smith masking below.
    gv, gl = 2*nv/(nv+1), 2*nl/(nl+1)
    specular = fresnel*gv*gl/(4*math.pi*nv*nl)
    return tuple(encode(((1-fresnel)*FACTOR[k]*texel[k]/math.pi+specular)*nl) for k in range(3))


def params():
    p = Parameters()
    p.view[:] = (1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,-1,1)
    p.intrinsics[:] = (2,2,1.5,0.5)
    p.clipping[:] = (3,1,0.1,5)
    p.camera[:] = (0,0,1,1)
    p.light_directions[0][:] = (0,0,-1,0)
    p.light_colors[0][:] = (1,1,1,0)
    p.counts[:] = (1,4,1,2)
    p.options[:] = (1,1e-5,0,0)
    return p


def geometry(tilt=0.0):
    vertices = (ResidentVertex*4)()
    normal = (-tilt/math.hypot(tilt,1),0,1/math.hypot(tilt,1),0)
    for target,(x,y) in zip(vertices,((-0.24,-0.24),(0.24,-0.24),(0.24,0.24),(-0.24,0.24))):
        target.position[:] = (x,y,tilt*x,1)
        target.normal[:] = normal
    indices = (ct.c_uint32*6)(0,1,2,0,2,3)
    meshes = (ResidentMesh*1)()
    meshes[0].first_vertex, meshes[0].vertex_count = 0,4
    meshes[0].first_index, meshes[0].index_count = 0,6
    instances = (ResidentInstance*2)()
    for i, instance in enumerate(instances):
        instance.mesh_index, instance.instance_id = 0,21+i
        instance.color_roughness[:] = (*FACTOR,1)
        instance.material[:] = (0,0,0,0)
    matrices = (ct.c_float*32)(
        1,0,0,0, 0,1,0,0, 0,0,1,0, -0.5,0,0,1,
        1,0,0,0, 0,1,0,0, 0,0,1,0, 0.5,0,0,1)
    return (vertices,indices,meshes,instances), matrices


def upload_data(uv, *, varying=False, plain_second=False):
    records = (Texture*2)(Texture(2,2,0,0),Texture(1,1,16,0))
    rgba = (ct.c_uint8*20)(*(value for pixel in (*FIRST,*SECOND) for value in (*pixel,255)))
    coords = ((0,1),(1,1),(1,0),(0,0)) if varying else (uv,)*4
    uvs = (UV*4)(*(UV(*pair) for pair in coords))
    slots = (ct.c_uint32*2)(0,0xffffffff if plain_second else 1)
    return records,rgba,uvs,slots


def bind(device):
    require(getattr(device._lib,"pr_resident_texture_abi_version",None) is not None,"resident texture ABI missing")
    abi = ct.CFUNCTYPE(ct.c_uint32)(("pr_resident_texture_abi_version",device._lib))
    require(abi()==1,"texture ABI version")
    setter = ct.CFUNCTYPE(ct.c_int,ct.c_void_p,ct.POINTER(Texture),ct.c_uint32,
        ct.POINTER(ct.c_uint8),ct.c_size_t,ct.POINTER(UV),ct.c_uint32,
        ct.POINTER(ct.c_uint32),ct.c_uint32)(("pr_resident_set_textures",device._lib))
    query = ct.CFUNCTYPE(ct.c_int,ct.c_void_p,ct.POINTER(TextureInfo))(("pr_resident_texture_info",device._lib))
    require(tuple(map(ct.sizeof,(Texture,UV,TextureInfo)))==(16,8,128),"texture structure sizes")
    return setter,query


def call(setter,device,data,*,uv_count=4,instance_count=2):
    records,rgba,uvs,slots=data
    return setter(device._handle,records,len(records),rgba,len(rgba),uvs,uv_count,slots,instance_count)


def untextured_sensors(library, index, tilt):
    """Same raster geometry, without any material texture, for byte-level sensors.

    The deliberately subpixel tilted triangles have fixed-function raster
    interpolation quantization. They are not an exact ray/plane depth oracle.
    The independent color/UV calculation still detects affine UV interpolation.
    """
    device = ResidentDevice(library, library.parent, index, True)
    try:
        buffers, matrices = geometry(tilt)
        device.upload(*buffers)
        raw = device.render(matrices, params(), 3, 1)
        setter, _ = bind(device)
        require(call(setter, device, upload_data((0.5, 0.5))) != 0,
                "texture upload after first capture accepted")
        require(device.info()["resident_poisoned"] == 0, "late upload rejection poisoned session")
        return raw[12:]
    finally:
        device.close()
        require(device.info()["live_resources"] == 0, "untextured control leaked resources")

def run_case(library,index,uv,*,tilt=0.0,varying=False,plain_second=False,negative=False):
    reference_sensors = untextured_sensors(library, index, tilt) if tilt else None
    device = ResidentDevice(library,library.parent,index,True)
    record={"uv":list(uv),"tilt":tilt,"varying":varying,"plain_second":plain_second}
    try:
        buffers,matrices=geometry(tilt)
        device.upload(*buffers)
        setter,query=bind(device)
        if negative:
            for invalid in ("zero_width","too_wide","offset","alpha","slot","nan_uv","short_uv","short_slots"):
                data=upload_data(uv)
                if invalid=="zero_width": data[0][0].width=0
                if invalid=="too_wide": data[0][0].width=2049
                if invalid=="offset": data[0][1].first_byte=12
                if invalid=="alpha": data[1][3]=0
                if invalid=="slot": data[3][1]=2
                if invalid=="nan_uv": data[2][0].u=float("nan")
                code=call(setter,device,data,uv_count=3 if invalid=="short_uv" else 4,
                          instance_count=1 if invalid=="short_slots" else 2)
                require(code!=0,f"invalid texture accepted: {invalid}")
                require(device.info()["resident_poisoned"]==0,"input rejection poisoned session")
            record["negative_controls"]=8
        data=upload_data(uv,varying=varying,plain_second=plain_second)
        require(call(setter,device,data)==0,device._error())
        require(call(setter,device,data)!=0,"second texture upload accepted")
        for array in data: ct.memset(ct.addressof(array),0,ct.sizeof(array))
        del data
        gc.collect()
        raw=device.render(matrices,params(),3,1)
        raw_again=device.render(matrices,params(),3,1)
        require(raw==raw_again,"static texture capture changed")
        colors=tuple(tuple(raw[4*i:4*i+3]) for i in (0,2))
        expected_uv=(0.5,0.5) if varying else uv
        expected=(lit_color(sampled(FIRST,2,2,expected_uv),-0.5,tilt),
                  lit_color((1,1,1) if plain_second else sampled(SECOND,1,1,expected_uv),0.5,tilt))
        require(all(abs(a-b)<=2 for got,want in zip(colors,expected) for a,b in zip(got,want)),
                f"texture oracle mismatch: {colors} vs {expected}")
        require(struct.unpack_from("<3I",raw,24)==(21,0,22),"texture changed instance IDs")
        depths=struct.unpack_from("<3f",raw,12)
        if reference_sensors is None:
            require(abs(depths[0]-1)<2e-6 and math.isinf(depths[1]) and abs(depths[2]-1)<2e-6,
                    f"texture changed constant-plane depth: {depths}")
        else:
            require(raw[12:] == reference_sensors, "texture changed tilted raster sensor bytes")
            record["untextured_sensor_sha256"] = hashlib.sha256(reference_sensors).hexdigest()
            record["tilted_depth_m"] = (depths[0], depths[2])
            record["sensor_control"] = "same subpixel raster geometry; no texture; exact bytes, not ideal ray depth"
        require(raw[36:39]==bytes((1,0,1)),"texture changed visibility")
        info=TextureInfo()
        require(query(device._handle,ct.byref(info))==0,device._error())
        require(info.texture_count==2 and info.array_layers==2 and info.array_width==4 and info.array_height==4,
                "texture atlas extent/source count")
        require(info.upload_calls==1 and info.mip_levels==1 and info.sampler_filter==1,"texture residency/filter")
        record.update(actual_rgb=colors,expected_rgb=expected,output_sha256=hashlib.sha256(raw).hexdigest(),
                      texture={name:getattr(info,name) for name,_ in info._fields_},
                      caller_arrays_zeroed_before_first_draw=True)
    finally:
        device.close()
        closed=device.info()
        require(all(closed[key]==0 for key in ("live_contexts","live_frames","live_resources","validation_errors")),
                "texture resources survived close")
        require(closed["resident_telemetry"]["device_local_bytes_current"]==0 and
                closed["resident_telemetry"]["host_visible_bytes_current"]==0,"texture allocation leak")
        record["closed"]=closed
    return record


def main():
    global ResidentDevice
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--device-index",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--ray-query",action="store_true",help="Use the explicit hardware shadow creation path")
    args=parser.parse_args()
    if args.ray_query:
        from native_resident_rayquery_smoke import RayDevice
        ResidentDevice=RayDevice
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    library=args.library.resolve()
    report={"schema":"phymeo-native-resident-texture-smoke-v1","status":"running","cases":[],
            "library_sha256":hashlib.sha256(library.read_bytes()).hexdigest()}
    try:
        for i,uv in enumerate(((0.25,0.25),(0.75,0.25),(0.25,0.75),(0.75,0.75),(0.5,0.5),
                               (-0.25,0.25),(1.25,0.25),(0,0.25),(0.25,-0.25),(0.99999,0.25))):
            report["cases"].append(run_case(library,args.device_index,uv,negative=i==0))
        report["cases"].append(run_case(library,args.device_index,(0.5,0.5),tilt=0.5,varying=True))
        report["cases"].append(run_case(library,args.device_index,(0.25,0.25),plain_second=True))
        report["status"]="passed"
    except BaseException as error:
        report.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output/"receipt.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"status":report["status"],"receipt":str(output/"receipt.json")}))

if __name__=="__main__":
    main()
