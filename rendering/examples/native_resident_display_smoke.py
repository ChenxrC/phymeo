"""Independent controlled-radiance native display calibration (no simulation).

--library BUILD/phymeo_native.dll --device-index 1 --output NEW_DIRECTORY
[--ray-query] [--legacy-shaders PRE_DISPLAY_BUILD]

Decimal equations follow KhronosGroup/ToneMapping b5a2eed5ddf6c2227090449399de9c7affb9e4c9.
The oracle never consumes clipped RGB. Constant cubemap/clear radiance and an
independently evaluated directional GGX plane supply the linear input. GPU
FP32 and UNORM rounding permit at most two RGB8 codes; sensor bytes are exact.
"""
from __future__ import annotations
import argparse
import ctypes as ct
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
from native_resident_texture_smoke import geometry, params, require, FACTOR
from native_resident_environment_smoke import Environment, bind as bind_environment, upload as upload_environment
from native_resident_rayquery_smoke import Studio
from native_resident_smoke import _close_clean
from phymeo_rendering.backends._native_resident import ResidentDevice

COLORS=((0.,)*3,(1e-10,)*3,(.0001,)*3,(.02,)*3,(.07999999,)*3,(.08,)*3,
        (.08000001,)*3,(.18,)*3,(.5,)*3,(.7999999,)*3,(.8,)*3,(.8000001,)*3,
        (1.,)*3,(2.,)*3,(4.,)*3,(8.,)*3,(8.,.4,.1),(.1,4.,1.5),(1.5,.1,4.))


def reference(rgb, exposure=1., neutral=True):
    with localcontext() as context:
        context.prec=70
        values=[max(Decimal(0),Decimal.from_float(float(v))*Decimal.from_float(float(exposure))) for v in rgb]
        if neutral:
            minimum=min(values); fresnel=Decimal('.04')
            offset=minimum-minimum*minimum/(4*fresnel) if minimum<=2*fresnel else fresnel
            values=[value-offset for value in values]; peak=max(values); start=Decimal('.76')
            if peak>start:
                shoulder=1-(1-start)**2/(peak+1-2*start)
                weight=1/(Decimal('.15')*(peak-shoulder)+1)
                values=[value*shoulder/peak*weight+shoulder*(1-weight) for value in values]
        output=[]
        for value in values:
            v=min(1.,max(0.,float(value)))
            srgb=12.92*v if v<=.0031308 else 1.055*v**(1/2.4)-.055
            output.append(min(255,int(math.floor(srgb*255+.5))))
        return tuple(output)


def bind(device):
    compatible=ct.CFUNCTYPE(ct.c_int,ct.c_void_p)(('pr_resident_display_transform_shader_compatible',device._lib))
    setter=ct.CFUNCTYPE(ct.c_int,ct.c_void_p,ct.c_uint32)(('pr_resident_set_display_transform',device._lib))
    return compatible,setter


def constant_cube(device, library, rgb, shaders_directory=None):
    config=Environment(2,2)
    values=(ct.c_float*(6*(4+1)*4))(*((*rgb,1.)*(6*(4+1))))
    setter,_,shaders=bind_environment(device,(shaders_directory or library.parent)/library.name)
    return upload_environment(setter,device,config,values,shaders)


def direct_radiance(x, light):
    # Independent roughness=1, metallic=0 planar GGX, unit incident +Z.
    nv=1/math.hypot(x,1); vx=-x*nv
    vh=math.hypot(vx,nv+1)/2
    fresnel=.04+.96*(1-vh)**5
    gv=2*nv/(nv+1)
    specular=fresnel*gv/(4*math.pi*nv)
    return tuple(((1-fresnel)*float(ct.c_float(FACTOR[k]).value)/math.pi+specular)*light[k] for k in range(3))


def capture(kind, library, index, rgb, exposure, source, mode):
    device=kind(library,library.parent,index,True); item={}
    try:
        buffers,matrices=geometry(); device.upload(*buffers)
        compatible,setter=bind(device)
        require(compatible(device._handle)==1,'new main fragment missing display marker')
        if mode is not None: require(setter(device._handle,mode)==0,device._error())
        p=params(); p.options[0]=exposure
        rgb=tuple(ct.c_float(v).value for v in rgb)
        if source=='sky':
            require(constant_cube(device,library,rgb)==0,device._error()); p.counts[0]=0
            pixel=1; linear=rgb
        elif source=='clear':
            studio=Studio(); studio.environment[:]=(*rgb,1)
            set_studio=ct.CFUNCTYPE(ct.c_int,ct.c_void_p,ct.POINTER(Studio))(('pr_resident_set_studio',device._lib))
            require(set_studio(device._handle,ct.byref(studio))==0,device._error()); p.counts[0]=0
            pixel=1; linear=rgb
        else:
            p.light_colors[0][:]=(*rgb,0); pixel=0; linear=direct_radiance(-.5,rgb)
        raw=device.render(matrices,p,3,1)
        actual=tuple(raw[4*pixel:4*pixel+3]); expected=reference(linear,float(p.options[0]),mode==1)
        require(all(abs(a-b)<=2 for a,b in zip(actual,expected)),f'{source} display {actual} != {expected}; radiance {linear}')
        require(setter(device._handle,1)!=0,'display change after capture accepted')
        item.update(source=source,declared_rgb=rgb,exposure=float(p.options[0]),linear_oracle=linear,
            actual=actual,expected=expected,output_sha256=hashlib.sha256(raw).hexdigest())
    finally: item['closed']=_close_clean(device)
    return item,raw


def negatives(kind, library, index, old):
    report={}
    target=kind(library,library.parent,index,True)
    try:
        buffers,_=geometry(); target.upload(*buffers); compatible,setter=bind(target)
        require(setter(target._handle,2)!=0,'unknown display mode accepted')
        require(setter(target._handle,1)==0,target._error())
        if old:
            require(constant_cube(target,library,(.2,.3,.4),old)!=0,'old sky accepted after neutral')
            require(target.info()['resident_environment_info']['loaded']==0,'rejected sky modified environment')
        require(target.info()['resident_poisoned']==0,'input rejection poisoned context')
        report['invalid_mode_and_old_sky_after_neutral']=True
    finally: _close_clean(target)
    if old:
        target=kind(library,old,index,True)
        try:
            buffers,_=geometry(); target.upload(*buffers); compatible,setter=bind(target)
            require(compatible(target._handle)==0,'old main falsely compatible')
            require(setter(target._handle,1)!=0,'old main accepted neutral')
            require(setter(target._handle,0)==0,'old main clip no longer supported')
            report['old_main_rejected']=True
        finally: _close_clean(target)
        target=kind(library,library.parent,index,True)
        try:
            buffers,_=geometry(); target.upload(*buffers); _,setter=bind(target)
            require(constant_cube(target,library,(.2,.3,.4),old)==0,target._error())
            require(setter(target._handle,1)!=0,'neutral accepted already loaded old sky')
            report['old_sky_before_neutral_rejected']=True
        finally: _close_clean(target)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True); parser.add_argument('--device-index',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--ray-query',action='store_true')
    parser.add_argument('--legacy-shaders',type=Path)
    args=parser.parse_args(); library=args.library.resolve(); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    protocol={'schema':'phymeo-native-neutral-protocol-v1','colors':COLORS,'exposures':[1.,3.5,0.],
        'sources':['clear','sky','directional-ggx'],'rgb8_max_abs_error':2,'sensor_relation':'byte-identical',
        'source':'https://github.com/KhronosGroup/ToneMapping/tree/b5a2eed5ddf6c2227090449399de9c7affb9e4c9/PBR_Neutral',
        'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    report={'schema':'phymeo-native-neutral-smoke-v1','status':'running','captures':[],
        'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'shadow_mode':'ray_query' if args.ray_query else 'software'}
    try:
        lib=ct.CDLL(str(library))
        require(getattr(lib,'pr_resident_display_transform_abi_version',None) is not None,'resident display-transform ABI missing')
        version=ct.CFUNCTYPE(ct.c_uint32)(('pr_resident_display_transform_abi_version',lib))
        require(version()==1,'display ABI version mismatch')
        kind=ResidentDevice
        if args.ray_query:
            from native_resident_rayquery_smoke import RayDevice
            kind=RayDevice
        for source in protocol['sources']:
            for rgb in COLORS:
                modes=[]
                for mode in (None,0,1):
                    item,raw=capture(kind,library,args.device_index,rgb,1.,source,mode)
                    item['mode']=mode; report['captures'].append(item); modes.append(raw)
                require(modes[0]==modes[1],'default vs explicit clip differs')
                require(modes[0][12:]==modes[2][12:],'display transform changed depth/ID/VALID')
            for exposure in (3.5,0.):
                item,_=capture(kind,library,args.device_index,(.23,.04,.37),exposure,source,1)
                report['captures'].append(item)
        report['negative_controls']=negatives(kind,library,args.device_index,args.legacy_shaders)
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}'); raise
    finally:
        (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'output':str(output)}))


if __name__=='__main__': main()
