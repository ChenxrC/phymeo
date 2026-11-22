"""Explicit Vulkan opaque RGB + linear roughness-alpha regression.

PYTHONPATH=rendering/src python -B rendering/examples/native_resident_surface_maps_smoke.py \
  --library BUILD/phymeo_native.dll --device-index 1 --output NEW_DIR \
  [--ray-query] [--legacy-shaders OLD_BUILD]

RGB stays sRGB and alpha is a LINEAR material multiplier, never transparency.
Analytic direct GGX, split-sum IBL, repeat/bilinear and sensor bytes are checked.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import struct
from native_resident_texture_smoke import (
    Texture,UV,TextureInfo,FIRST,SECOND,FACTOR,sampled,encode,geometry,params,call,require,
)
from native_resident_environment_smoke import data as environment_data,bind as bind_environment,upload as upload_environment,cube_sample,dfg
from native_resident_smoke import _close_clean
from phymeo_rendering.backends._native_resident import ResidentDevice

ALPHA=(0,64,128,255)
BASE_ROUGHNESS=0.8


def bind(device):
    setter_type=ct.CFUNCTYPE(ct.c_int,ct.c_void_p,ct.POINTER(Texture),ct.c_uint32,
        ct.POINTER(ct.c_uint8),ct.c_size_t,ct.POINTER(UV),ct.c_uint32,ct.POINTER(ct.c_uint32),ct.c_uint32)
    setter=setter_type(('pr_resident_set_surface_maps',device._lib))
    legacy=setter_type(('pr_resident_set_textures',device._lib))
    compatible=ct.CFUNCTYPE(ct.c_int,ct.c_void_p)(('pr_resident_surface_maps_shader_compatible',device._lib))
    return setter,legacy,compatible


def upload_data(uv,varying=False,alpha=None,white=False):
    records=(Texture*2)(Texture(2,2,0,1),Texture(1,1,16,0))
    colors=((255,255,255),)*5 if white else (*FIRST,*SECOND)
    alphas=ALPHA if alpha is None else (alpha,)*4
    pixels=[component for color,a in zip(colors,(*alphas,255)) for component in (*color,a)]
    rgba=(ct.c_uint8*20)(*pixels)
    coordinates=((0,1),(1,1),(1,0),(0,0)) if varying else (uv,)*4
    coords=(UV*4)(*(UV(*value) for value in coordinates))
    return records,rgba,coords,(ct.c_uint32*2)(0,1)


def sampled_alpha(uv):
    x,y=(uv[0]%1)*2-0.5,(uv[1]%1)*2-0.5
    ix,iy=math.floor(x),math.floor(y); ax,ay=x-ix,y-iy
    return sum(ALPHA[((iy+dy)%2)*2+(ix+dx)%2]/255*wx*wy
        for dx,wx in ((0,1-ax),(1,ax)) for dy,wy in ((0,1-ay),(1,ay)))


def direct_color(texel,x,tilt,roughness):
    n0=(-tilt,0,1); n=tuple(v/math.hypot(*n0) for v in n0)
    v=(-x/math.hypot(x,1),0,1/math.hypot(x,1))
    h0=(v[0],0,v[2]+1); h=tuple(value/math.hypot(*h0) for value in h0)
    nv=sum(a*b for a,b in zip(n,v)); nl=n[2]
    nh=max(0,sum(a*b for a,b in zip(n,h))); vh=max(0,sum(a*b for a,b in zip(v,h)))
    alpha=max(roughness*roughness,0.045*0.045); a2=alpha*alpha
    d=a2/(math.pi*(nh*nh*(a2-1)+1)**2)
    gv=2*nv/(nv+math.sqrt(a2+(1-a2)*nv*nv))
    gl=2*nl/(nl+math.sqrt(a2+(1-a2)*nl*nl))
    f=0.04+0.96*(1-vh)**5
    return tuple(encode(((1-f)*FACTOR[k]*texel[k]/math.pi+d*gv*gl*f/(4*nv*nl))*nl) for k in range(3))


def control_sensors(kind,library,index,tilt):
    device=kind(library,library.parent,index,True)
    try:
        buffers,matrices=geometry(tilt); device.upload(*buffers)
        return device.render(matrices,params(),3,1)[12:]
    finally: _close_clean(device)


def direct_case(kind,library,index,uv,tilt=0,varying=False,negative=False):
    sensors=control_sensors(kind,library,index,tilt)
    device=kind(library,library.parent,index,True)
    result={'uv':uv,'tilt':tilt,'varying':varying}
    try:
        buffers,matrices=geometry(tilt)
        for instance in buffers[3]: instance.color_roughness[3]=BASE_ROUGHNESS
        device.upload(*buffers)
        setter,legacy,compatible=bind(device)
        require(compatible(device._handle)==1,'surface map shader marker missing')
        if negative:
            for failure in ('legacy_flag','legacy_alpha','flag2','flag0_alpha'):
                packed=upload_data(uv)
                if failure=='legacy_flag':
                    for pixel in range(4): packed[1][4*pixel+3]=255
                if failure=='legacy_alpha': packed[0][0].reserved=0
                if failure=='flag2': packed[0][0].reserved=2
                if failure=='flag0_alpha': packed[0][0].reserved=0
                function=legacy if failure.startswith('legacy') else setter
                require(call(function,device,packed)!=0,f'invalid alpha protocol accepted: {failure}')
                require(device.info()['resident_poisoned']==0,'alpha input failure poisoned device')
            result['negative_controls']=4
        packed=upload_data(uv,varying)
        require(call(setter,device,packed)==0,device._error())
        require(call(setter,device,packed)!=0,'surface map reupload accepted')
        for value in packed: ct.memset(ct.addressof(value),0,ct.sizeof(value))
        raw=device.render(matrices,params(),3,1)
        require(raw==device.render(matrices,params(),3,1),'surface map residency changed between captures')
        require(raw[12:]==sensors,'roughness map changed opacity or another sensor byte')
        actual=[tuple(raw[4*p:4*p+3]) for p in (0,2)]
        sample_uv=(0.5,0.5) if varying else uv
        r=BASE_ROUGHNESS*sampled_alpha(sample_uv)
        expected=[direct_color(sampled(FIRST,2,2,sample_uv),-0.5,tilt,r),
                  direct_color(sampled(SECOND,1,1,sample_uv),0.5,tilt,BASE_ROUGHNESS)]
        require(all(abs(a-b)<=2 for got,want in zip(actual,expected) for a,b in zip(got,want)),
            f'linear roughness/GGX mismatch: {actual} != {expected}')
        result.update(actual=actual,expected=expected,first_perceptual_roughness=r,
            output_sha256=hashlib.sha256(raw).hexdigest(),caller_data_zeroed=True)
    finally: result['closed']=_close_clean(device)
    return result


def ibl_case(kind,library,index,alpha):
    device=kind(library,library.parent,index,True); result={'alpha_u8':alpha}
    try:
        buffers,matrices=geometry()
        for instance in buffers[3]: instance.color_roughness[3]=BASE_ROUGHNESS; instance.material[0]=1
        device.upload(*buffers)
        setter,_,_=bind(device)
        require(call(setter,device,upload_data((0.5,0.5),alpha=alpha,white=True))==0,device._error())
        env_set,_,shaders=bind_environment(device,library)
        config,source,mips=environment_data('signal')
        require(upload_environment(env_set,device,config,source,shaders)==0,device._error())
        p=params(); p.counts[0]=0
        raw=device.render(matrices,p,3,1)
        require(struct.unpack_from('<3I',raw,24)==(21,0,22) and raw[36:39]==bytes((1,0,1)),'IBL roughness changed sensors')
        rows=[]
        for pixel,x,a in ((0,-0.5,alpha/255),(2,0.5,1)):
            roughness=BASE_ROUGHNESS*a; lod=roughness*(config.mip_levels-1)
            lo=math.floor(lod); hi=min(config.mip_levels-1,lo+1); blend=lod-lo
            length=math.hypot(x,1); reflection=(x/length,0,1/length)
            c0=cube_sample(mips,config.side,reflection,lo); c1=cube_sample(mips,config.side,reflection,hi)
            environment=tuple((1-blend)*a+blend*b for a,b in zip(c0,c1))
            response=dfg(x,1,roughness)
            expected=tuple(encode(a*b) for a,b in zip(environment,response))
            actual=tuple(raw[pixel*4:pixel*4+3])
            require(all(abs(a-b)<=2 for a,b in zip(actual,expected)),f'IBL did not use mapped perceptual roughness: {actual} != {expected}')
            rows.append({'actual':actual,'expected':expected,'perceptual_roughness':roughness,'lod':lod})
        result['pixels']=rows
    finally: result['closed']=_close_clean(device)
    return result


def actual_lights_case(kind,library,index):
    from native_resident_rayquery_smoke import Studio
    outputs=[]; report={}
    mapped_r=ct.c_float(ct.c_float(BASE_ROUGHNESS).value*ct.c_float(128/255).value).value
    for mapped in (False,True):
        device=kind(library,library.parent,index,True)
        try:
            buffers,matrices=geometry()
            buffers[3][0].color_roughness[3]=BASE_ROUGHNESS if mapped else mapped_r
            buffers[3][1].color_roughness[3]=BASE_ROUGHNESS
            device.upload(*buffers)
            if mapped:
                setter,_,_=bind(device)
                require(call(setter,device,upload_data((0.5,0.5),alpha=128,white=True))==0,device._error())
            studio=Studio(); studio.environment[:]=(0.125,0.125,0.125,1)
            for panel,position,extent,color in zip(studio.panels,((-0.5,-0.3,1.2),(0.5,0.4,0.8)),
                ((0.3,0.2),(0.2,0.15)),((3,2,1),(0.8,1.2,2))):
                panel.position[:]=(*position,0); panel.half_u[:]=(extent[0],0,0,0)
                panel.half_v[:]=(0,-extent[1],0,0); panel.radiance[:]=(*color,0)
            set_studio=ct.CFUNCTYPE(ct.c_int,ct.c_void_p,ct.POINTER(Studio))(('pr_resident_set_studio',device._lib))
            require(set_studio(device._handle,ct.byref(studio))==0,device._error())
            p=params(); p.counts[0]=0
            outputs.append(device.render(matrices,p,3,1))
        finally: report['mapped_closed' if mapped else 'constant_closed']=_close_clean(device)
    require(outputs[0]==outputs[1],'mapped roughness differs from equal constant material under actual rectangle lights')
    report.update(mapped_perceptual_roughness=mapped_r,rectangle_lights=2,
        identical_rgba_and_sensors=True,sha256=hashlib.sha256(outputs[1]).hexdigest())
    return report

def old_shader(kind,library,index,directory):
    if directory is None: return None
    device=kind(library,directory,index,True)
    try:
        buffers,_=geometry(); device.upload(*buffers)
        setter,_,compatible=bind(device)
        require(compatible(device._handle)==0,'old SPV falsely declared surface-map support')
        before=device.info()['resident_texture_info']['upload_calls']
        require(call(setter,device,upload_data((0.5,0.5)))!=0,'old SPV accepted roughness maps')
        after=device.info()
        require(after['resident_poisoned']==0 and after['resident_texture_info']['upload_calls']==before,
            'old shader rejection mutated texture resources')
        return {'rejected':True,'reason':device._error()}
    finally: _close_clean(device)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True); parser.add_argument('--device-index',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--ray-query',action='store_true')
    parser.add_argument('--legacy-shaders',type=Path)
    args=parser.parse_args(); library=args.library.resolve(); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    report={'schema':'phymeo-native-surface-maps-smoke-v1','status':'running','direct':[], 'ibl':[],
        'shadow_mode':'ray_query' if args.ray_query else 'software','library_sha256':hashlib.sha256(library.read_bytes()).hexdigest()}
    try:
        lib=ct.CDLL(str(library))
        require(getattr(lib,'pr_resident_surface_maps_abi_version',None) is not None,'resident surface-maps ABI missing')
        version=ct.CFUNCTYPE(ct.c_uint32)(('pr_resident_surface_maps_abi_version',lib))
        require(version()==1,'surface-maps ABI version')
        kind=ResidentDevice
        if args.ray_query:
            from native_resident_rayquery_smoke import RayDevice
            kind=RayDevice
        report['old_shader']=old_shader(kind,library,args.device_index,args.legacy_shaders)
        for i,uv in enumerate(((0.25,0.25),(0.75,0.25),(0.25,0.75),(0.75,0.75),(0.5,0.5),(-0.25,0.25),(0,0.25))):
            report['direct'].append(direct_case(kind,library,args.device_index,uv,negative=i==0))
        report['direct'].append(direct_case(kind,library,args.device_index,(0.5,0.5),tilt=0.5,varying=True))
        for alpha in (0,64,128,255): report['ibl'].append(ibl_case(kind,library,args.device_index,alpha))
        report['actual_lights']=actual_lights_case(kind,library,args.device_index)
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}'); raise
    finally: (output/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'status':report['status'],'receipt':str(output/'receipt.json')}))

if __name__=='__main__': main()
