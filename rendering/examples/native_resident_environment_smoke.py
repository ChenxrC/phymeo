"""Independent explicit-GPU directional-environment ABI and sky/PBR oracle.

PYTHONPATH=rendering/src python -B rendering/examples/native_resident_environment_smoke.py \
  --library BUILD/phymeo_native.dll --device-index 1 --output NEW_DIRECTORY

Synthetic cube signals test mapping and mip addressing; they are not claimed
as a physically convolved photograph. Constant-map SH/specular is consistent.
No physical simulation, generated image, or host pixel substitution is used.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import struct
from native_resident_texture_smoke import geometry, params, encode, require, FACTOR
from phymeo_rendering.backends._native_resident import ResidentDevice, ResidentVertex, ResidentMesh, ResidentInstance
from phymeo_rendering.backends._native_device import Parameters

Y00=0.28209479177387814
class Environment(ct.Structure):
    _fields_=[("side",ct.c_uint32),("mip_levels",ct.c_uint32),("reserved",ct.c_uint32*2),
              ("diffuse_sh",(ct.c_float*4)*9)]
class EnvironmentInfo(ct.Structure):
    _fields_=[(name,ct.c_uint32) for name in (
        "environment_abi_version","supported","shader_compatible","max_side","max_mip_levels",
        "side","mip_levels","loaded","format","cube_layers","sh_coefficients","axis_transform","diffuse_basis")]+[
        ("reserved",ct.c_uint32*3)]+[(name,ct.c_uint64) for name in (
        "max_rgba_bytes","source_rgba_bytes","image_upload_bytes","parameter_upload_bytes",
        "upload_calls","rgb_captures_with_environment","sky_draws","reserved64")]


def unit(v):
    length=math.hypot(*v)
    return tuple(x/length for x in v)


def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def sh_basis(n):
    x,y,z=n
    return (Y00,0.4886025119029199*y,0.4886025119029199*z,0.4886025119029199*x,
            1.0925484305920792*x*y,1.0925484305920792*y*z,0.31539156525252005*(3*z*z-1),
            1.0925484305920792*x*z,0.5462742152960396*(x*x-y*y))


def data(kind="signal",side=16):
    levels=side.bit_length()
    config=Environment(side,levels)
    source=[]; mips=[]
    constant=(0.18,0.24,0.30)
    for level in range(levels):
        dim=side>>level; faces=[]
        for face in range(6):
            pixels=[]
            for y in range(dim):
                for x in range(dim):
                    if kind=="constant": color=constant
                    elif kind=="sh": color=(0,0,0)
                    elif level==0: color=(0.06+0.1*face+0.14*(x+0.5)/dim,
                                           0.05+0.09*face+0.12*(y+0.5)/dim,0.08+0.07*face)
                    else: color=(0.12+0.08*level,0.22+0.04*level,0.38-0.03*level)
                    pixels.append(color); source.extend((*color,1))
            faces.append(pixels)
        mips.append(faces)
    if kind=="constant": config.diffuse_sh[0][:]=(*(x/Y00 for x in constant),0)
    if kind=="sh":
        config.diffuse_sh[0][:]=(0.3/Y00,0.3/Y00,0.3/Y00,0)
        config.diffuse_sh[1][:]=(0,0.1,0,0)
        config.diffuse_sh[2][:]=(0.2,0,0,0)
        config.diffuse_sh[3][:]=(0,0,0.15,0)
    return config,(ct.c_float*len(source))(*source),mips


def cube_sample(mips,side,world,level):
    # Independent Vulkan cube face selection. The public environment convention
    # maps world XYZ to cube XZY, while diffuse SH remains in world XYZ.
    x,y,z=world[0],world[2],world[1]
    axis=max(range(3),key=lambda i:abs((x,y,z)[i]))
    if axis==0: face,sc,tc,ma=(0,-z,-y,x) if x>=0 else (1,z,-y,-x)
    elif axis==1: face,sc,tc,ma=(2,x,z,y) if y>=0 else (3,x,-z,-y)
    else: face,sc,tc,ma=(4,x,-y,z) if z>=0 else (5,-x,-y,-z)
    dim=side>>level
    px,py=(sc/ma+1)*dim/2-0.5,(tc/ma+1)*dim/2-0.5
    ix,iy=math.floor(px),math.floor(py); ax,ay=px-ix,py-iy
    # Test directions are deliberately away from cross-face filtering seams.
    return tuple(sum(mips[level][face][min(dim-1,max(0,iy+dy))*dim+min(dim-1,max(0,ix+dx))][k]*wx*wy
                     for dx,wx in ((0,1-ax),(1,ax)) for dy,wy in ((0,1-ay),(1,ay))) for k in range(3))


def camera(look,width=9,height=5):
    up=(0,1,0) if abs(look[2])>0.9 else (0,0,1)
    right=unit(cross(look,up)); back=tuple(-x for x in look); up=cross(back,right)
    columns=(right,up,back)
    p=Parameters()
    p.view[:]=tuple(columns[row][col] if row<3 and col<3 else float(row==col)
                     for col in range(4) for row in range(4))
    p.intrinsics[:]=(9,7,3.8,2.2); p.clipping[:]=(width,height,0.1,5)
    p.camera[:]=(0,0,0,1); p.counts[:]=(0,0,1,0); p.options[:]=(1,1e-5,0,0)
    return p,columns


def bind(device,library):
    require(getattr(device._lib,"pr_resident_environment_abi_version",None) is not None,"resident environment ABI missing")
    abi=ct.CFUNCTYPE(ct.c_uint32)(("pr_resident_environment_abi_version",device._lib))
    require(abi()==1,"environment ABI version")
    setter=ct.CFUNCTYPE(ct.c_int,ct.c_void_p,ct.POINTER(Environment),ct.POINTER(ct.c_float),ct.c_size_t,
        ct.POINTER(ct.c_uint32),ct.c_size_t,ct.POINTER(ct.c_uint32),ct.c_size_t)(("pr_resident_set_environment",device._lib))
    query=ct.CFUNCTYPE(ct.c_int,ct.c_void_p,ct.POINTER(EnvironmentInfo))(("pr_resident_environment_info",device._lib))
    require(ct.sizeof(Environment)==160 and ct.sizeof(EnvironmentInfo)==128,"environment structure sizes")
    shaders=[]
    for name in ("resident_sky.vert.spv","resident_sky.frag.spv"):
        raw=(library.parent/name).read_bytes()
        shaders.append((ct.c_uint32*(len(raw)//4)).from_buffer_copy(raw))
    return setter,query,shaders


def upload(setter,device,config,source,shaders,byte_count=None):
    return setter(device._handle,ct.byref(config),source,ct.sizeof(source) if byte_count is None else byte_count,
                  shaders[0],ct.sizeof(shaders[0]),shaders[1],ct.sizeof(shaders[1]))


def close_clean(device):
    device.close(); info=device.info()
    require(all(info[k]==0 for k in ("live_contexts","live_frames","live_resources","validation_errors")),"environment resource leak")
    require(info["resident_telemetry"]["device_local_bytes_current"]==0 and
            info["resident_telemetry"]["host_visible_bytes_current"]==0,"environment memory leak")
    return info


def sky_case(library,index):
    device=ResidentDevice(library,library.parent,index,True); result={"views":[]}
    try:
        device.upload((ResidentVertex*0)(),(ct.c_uint32*0)(),(ResidentMesh*0)(),(ResidentInstance*0)())
        setter,query,shaders=bind(device,library)
        for invalid in ("side","levels","reserved","alpha","negative","high","sh_nan","bytes","shader"):
            config,source,_=data()
            if invalid=="side": config.side=15
            if invalid=="levels": config.mip_levels-=1
            if invalid=="reserved": config.reserved[0]=1
            if invalid=="alpha": source[3]=0
            if invalid=="negative": source[0]=-1
            if invalid=="high": source[0]=101
            if invalid=="sh_nan": config.diffuse_sh[0][0]=float("nan")
            wrong=[shaders[1],shaders[1]] if invalid=="shader" else shaders
            code=upload(setter,device,config,source,wrong,ct.sizeof(source)-4 if invalid=="bytes" else None)
            require(code!=0,f"invalid environment accepted: {invalid}")
            require(device.info()["resident_poisoned"]==0,"environment input rejection poisoned session")
        config,source,mips=data()
        require(upload(setter,device,config,source,shaders)==0,device._error())
        require(upload(setter,device,config,source,shaders)!=0,"environment reupload accepted")
        ct.memset(ct.addressof(source),0,ct.sizeof(source))
        for look in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            p,columns=camera(look)
            raw=device.render((ct.c_float*0)(),p,9,5)
            pixels=45
            require(raw[4*pixels:8*pixels]==struct.pack("<f",float("inf"))*pixels,"sky changed miss depth")
            require(not any(raw[8*pixels:]),"sky changed ID or validity")
            spots=[]
            for x,y in ((4,2),(1,1),(7,3)):
                local=((x+0.5-p.intrinsics[2])/p.intrinsics[0],-(y+0.5-p.intrinsics[3])/p.intrinsics[1],-1)
                world=unit(tuple(sum(columns[k][j]*local[k] for k in range(3)) for j in range(3)))
                expected=tuple(encode(v) for v in cube_sample(mips,config.side,world,0))
                actual=tuple(raw[4*(y*9+x):4*(y*9+x)+3])
                require(all(abs(a-b)<=2 for a,b in zip(actual,expected)),f"sky projection mismatch {look} {x,y}: {actual} != {expected}")
                spots.append({"pixel":[x,y],"actual":actual,"expected":expected})
            result["views"].append({"look":look,"spots":spots,"output_sha256":hashlib.sha256(raw).hexdigest()})
        info=EnvironmentInfo(); require(query(device._handle,ct.byref(info))==0,device._error())
        require(info.loaded==1 and info.upload_calls==1 and info.sky_draws==6,"environment residency/sky count")
        result["environment"]={name:list(getattr(info,name)) if name=="reserved" else getattr(info,name) for name,_ in info._fields_}
        result["negative_controls"]=9
    finally: result["closed"]=close_clean(device)
    return result


def dfg(x,metal,roughness):
    v=unit((-x,0,1)); alpha=max(roughness*roughness,0.045*0.045); a2=alpha*alpha
    ve=unit((alpha*v[0],alpha*v[1],v[2])); xy=ve[0]*ve[0]+ve[1]*ve[1]
    tangent=(-ve[1]/math.sqrt(xy),ve[0]/math.sqrt(xy),0) if xy>1e-12 else (1,0,0)
    bitangent=cross(ve,tangent); result=[0.0]*3
    f0=FACTOR if metal else (0.04,)*3
    for i in range(8):
        reverse=int(f"{i:032b}"[::-1],2)
        r=math.sqrt((i+0.5)/8); angle=2*math.pi*(0.0625+reverse/2**32)
        px,py=r*math.cos(angle),r*math.sin(angle)
        blend=0.5*(1+ve[2]); py=(1-blend)*math.sqrt(max(0,1-px*px))+blend*py
        z=math.sqrt(max(0,1-px*px-py*py))
        h0=tuple(tangent[k]*px+bitangent[k]*py+ve[k]*z for k in range(3))
        h=unit((alpha*h0[0],alpha*h0[1],max(0,h0[2])))
        vh=max(0,min(1,dot(v,h))); light=tuple(2*vh*h[k]-v[k] for k in range(3))
        nl=light[2]
        if nl<=0: continue
        g1=2*nl/(nl+math.sqrt(a2+(1-a2)*nl*nl))
        for k in range(3): result[k]+=(f0[k]+(1-f0[k])*(1-vh)**5)*g1/8
    return result


def pbr_case(library,index,kind,metal,roughness):
    device=ResidentDevice(library,library.parent,index,True); result={"kind":kind,"metal":metal,"roughness":roughness}
    try:
        buffers,matrices=geometry()
        for instance in buffers[3]: instance.color_roughness[3]=roughness; instance.material[0]=metal
        device.upload(*buffers); setter,query,shaders=bind(device,library)
        config,source,mips=data(kind)
        require(upload(setter,device,config,source,shaders)==0,device._error())
        p=params(); p.counts[0]=0
        raw=device.render(matrices,p,3,1)
        require(struct.unpack_from("<3I",raw,24)==(21,0,22) and raw[36:39]==bytes((1,0,1)),"environment changed object visibility")
        depths=struct.unpack_from("<3f",raw,12)
        require(abs(depths[0]-1)<2e-6 and math.isinf(depths[1]) and abs(depths[2]-1)<2e-6,"environment changed object depth")
        result["pixels"]=[]
        for pixel,x in ((0,-0.5),(2,0.5)):
            diffuse=tuple(max(0,sum(config.diffuse_sh[i][k]*sh_basis((0,0,1))[i] for i in range(9))) for k in range(3))
            view=unit((-x,0,1)); reflection=(-view[0],-view[1],view[2])
            level=0 if roughness==0 else config.mip_levels-1
            environment=cube_sample(mips,config.side,reflection,level)
            weight=dfg(x,metal,roughness)
            expected=tuple(encode(diffuse[k]*(1-metal)*FACTOR[k]*0.96+environment[k]*weight[k]) for k in range(3))
            actual=tuple(raw[4*pixel:4*pixel+3])
            require(all(abs(a-b)<=2 for a,b in zip(actual,expected)),f"environment PBR mismatch {kind,metal,roughness}: {actual}!={expected}")
            result["pixels"].append({"actual":actual,"expected":expected})
        result["output_sha256"]=hashlib.sha256(raw).hexdigest()
    finally: result["closed"]=close_clean(device)
    return result


def sensor_planes_case(library,index):
    """Same geometry/parameters: environment may change only the RGB plane."""
    buffers,matrices=geometry()
    p=params(); p.counts[0]=0
    plain=ResidentDevice(library,library.parent,index,True)
    try:
        plain.upload(*buffers)
        baseline=plain.render(matrices,p,3,1)
    finally: close_clean(plain)
    mapped=ResidentDevice(library,library.parent,index,True)
    result={"masks":[]}
    try:
        mapped.upload(*buffers)
        setter,query,shaders=bind(mapped,library)
        config,source,_=data()
        require(upload(setter,mapped,config,source,shaders)==0,mapped._error())
        full=mapped.render(matrices,p,3,1)
        require(full[12:]==baseline[12:],"environment changed a same-geometry sensor byte")
        planes=(full[:12],full[12:24],full[24:36],full[36:39])
        for mask in range(1,16):
            p.counts[2]=int(bool(mask&1))
            compact=mapped.render_planes(matrices,p,3,1,mask)
            expected=b"".join(plane for bit,plane in enumerate(planes) if mask&(1<<bit))
            require(compact==expected,f"environment requested-plane mismatch: {mask}")
            result["masks"].append({"mask":mask,"bytes":len(compact),"sha256":hashlib.sha256(compact).hexdigest()})
        info=EnvironmentInfo(); require(query(mapped._handle,ct.byref(info))==0,mapped._error())
        require(info.sky_draws==9 and info.rgb_captures_with_environment==9,"sky ran for an unrequested RGB plane")
        require(info.image_upload_bytes==24+ct.sizeof(source) and info.parameter_upload_bytes==320,"environment upload counters")
        result["sensor_sha256"]=hashlib.sha256(full[12:]).hexdigest()
        result["sky_draws"]=info.sky_draws
    finally: result["closed"]=close_clean(mapped)
    return result

def rejected_lifecycle(library,index,legacy_shaders=None):
    result=[]
    for mode in (("after_capture",library.parent),("old_main_spv",legacy_shaders)):
        label,directory=mode
        if directory is None: continue
        device=ResidentDevice(library,directory,index,True)
        try:
            device.upload((ResidentVertex*0)(),(ct.c_uint32*0)(),(ResidentMesh*0)(),(ResidentInstance*0)())
            setter,query,shaders=bind(device,library)
            if label=="after_capture":
                p,_=camera((1,0,0)); device.render((ct.c_float*0)(),p,9,5)
            else:
                info=EnvironmentInfo(); require(query(device._handle,ct.byref(info))==0,device._error())
                require(info.shader_compatible==0,"old main SPV was declared environment compatible")
            config,source,_=data()
            require(upload(setter,device,config,source,shaders)!=0,f"invalid environment lifecycle accepted: {label}")
            require(device.info()["resident_poisoned"]==0,"lifecycle input failure poisoned context")
            result.append({"case":label,"error":device._error()})
        finally: close_clean(device)
    return result

def main():
    global ResidentDevice
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path,required=True); parser.add_argument("--device-index",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--legacy-shaders",type=Path,help="Old resident SPIR-V directory for explicit incompatibility control")
    parser.add_argument("--ray-query",action="store_true",help="Use the explicit hardware shadow creation path")
    args=parser.parse_args()
    if args.ray_query:
        from native_resident_rayquery_smoke import RayDevice
        ResidentDevice=RayDevice
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    library=args.library.resolve(); report={"schema":"phymeo-native-environment-smoke-v1","status":"running","pbr":[],
        "library_sha256":hashlib.sha256(library.read_bytes()).hexdigest()}
    report["shadow_mode"]="ray_query" if args.ray_query else "software"
    try:
        report["sky"] = sky_case(library,args.device_index)
        report["sensor_planes"] = sensor_planes_case(library,args.device_index)
        report["rejected_lifecycle"]=rejected_lifecycle(library,args.device_index,args.legacy_shaders)
        for kind,metal,roughness in (("constant",0,1),("constant",1,0),("signal",1,0),("signal",1,1),("sh",0,1)):
            report["pbr"].append(pbr_case(library,args.device_index,kind,metal,roughness))
        report["status"]="passed"
    except BaseException as error:
        report.update(status="failed",error=f"{type(error).__name__}: {error}"); raise
    finally: (output/"receipt.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"status":report["status"],"receipt":str(output/"receipt.json")}))

if __name__=="__main__": main()
