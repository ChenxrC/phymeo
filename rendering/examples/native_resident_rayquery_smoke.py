"""Explicit GPU hardware ray-query shadow regression; never a fallback render.

PYTHONPATH=rendering/src python -B rendering/examples/native_resident_rayquery_smoke.py \
  --library BUILD/phymeo_native.dll --device-index 1 --output NEW_DIR [--full-panda]

Only visibility changes between paths. Synthetic tests use analytic receiver
rays and finite emitter segments, while sensor attachments must match exactly.
Panda RGB differences are reported rather than disguised as bitwise RT parity.
The separate full 300-frame benchmark must use the unchanged public recipe.
"""
from __future__ import annotations
import argparse
import ctypes as ct
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import struct
from phymeo_rendering.backends._native_resident import ResidentDevice,ResidentVertex,ResidentMesh,ResidentInstance,pack_scene
from phymeo_rendering.backends._native_device import Parameters
from native_resident_smoke import ROOT,_close_clean,require

class RayInfo(ct.Structure):
    _fields_=[(name,ct.c_uint32) for name in (
        'ray_query_abi_version','enabled','blas_count','tlas_instance_count','primitive_count','loaded','poisoned','reserved',
        'max_meshes','max_instances','max_triangles','build_flags','update_flags','trace_flags')]+[
        ('reserved2',ct.c_uint32*2)]+[(name,ct.c_uint64) for name in (
        'max_acceleration_bytes','acceleration_buffer_bytes','scratch_buffer_bytes','blas_builds','tlas_builds',
        'tlas_updates','instance_upload_bytes','shadow_captures')]

class RayDevice(ResidentDevice):
    _shader_prefix='resident_rayquery'
    def _bind(self):
        super()._bind()
        self._ray_create=ct.CFUNCTYPE(ct.c_void_p,ct.c_uint32,ct.c_uint32,ct.POINTER(ct.c_uint32),ct.c_size_t,
            ct.POINTER(ct.c_uint32),ct.c_size_t)(('pr_resident_create_ray_query',self._lib))
    def _create(self,*args): return self._ray_create(*args)

class Panel(ct.Structure):
    _fields_=[(name,ct.c_float*4) for name in ('position','half_u','half_v','radiance')]
class Studio(ct.Structure):
    _fields_=[('environment',ct.c_float*4),('panels',Panel*2)]


def query(device):
    result=RayInfo()
    fn=ct.CFUNCTYPE(ct.c_int,ct.c_void_p,ct.POINTER(RayInfo))(('pr_resident_ray_query_info',device._lib))
    require(ct.sizeof(RayInfo)==128 and fn(device._handle,ct.byref(result))==0,device._error())
    return {name:list(getattr(result,name)) if name=='reserved2' else getattr(result,name) for name,_ in result._fields_}


def matrices(x,z,angle=0):
    c,s=math.cos(angle),math.sin(angle)
    return (ct.c_float*32)(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
        c,0,-s,0,0,1,0,0,s,0,c,0,x,0,z,1)


def geometry():
    vertices=(ResidentVertex*8)()
    for mesh,half in enumerate((1.0,0.10)):
        for i,(x,y) in enumerate(((-half,-half),(half,-half),(half,half),(-half,half))):
            vertices[4*mesh+i].position[:]=(x,y,0,1)
            vertices[4*mesh+i].normal[:]=(0,0,1,0)
    indices=(ct.c_uint32*12)(0,1,2,0,2,3,0,1,2,0,2,3)
    meshes=(ResidentMesh*2)(ResidentMesh(0,4,0,6),ResidentMesh(4,4,6,6))
    instances=(ResidentInstance*2)()
    for i in range(2):
        instances[i].mesh_index=i; instances[i].instance_id=11+i
        instances[i].color_roughness[:]=(0.6,0.6,0.6,0.7)
    return vertices,indices,meshes,instances


def parameters(direction=(1,0,1),camera_x=0):
    p=Parameters()
    p.view[:]=(1,0,0,0,0,1,0,0,0,0,1,0,-camera_x,0,-2,1)
    p.intrinsics[:]=(64,64,32.5,32.5); p.clipping[:]=(65,65,0.01,5)
    p.camera[:]=(camera_x,0,2,1)
    length=math.hypot(*direction)
    p.light_directions[0][:]=tuple(-v/length for v in direction)+(0,)
    p.light_colors[0][:]=(1,1,1,0); p.counts[:]=(1,4,1,2); p.options[:]=(1,1e-5,0,0)
    return p


def set_panel(device):
    value=Studio(); value.environment[:]=(0,0,0,1)
    panel=value.panels[0]
    panel.position[:]=(0.5,0,0.5,0)
    panel.half_u[:]=(0.02,0,0,0); panel.half_v[:]=(0,-0.02,0,0)
    panel.radiance[:]=(20,20,20,0)
    fn=ct.CFUNCTYPE(ct.c_int,ct.c_void_p,ct.POINTER(Studio))(('pr_resident_set_studio',device._lib))
    require(fn(device._handle,ct.byref(value))==0,device._error())


def pixel(raw):
    n=65*65; i=32*65+32
    require(struct.unpack_from('<I',raw,8*n+4*i)[0]==11,'receiver ID changed')
    require(abs(struct.unpack_from('<f',raw,4*n+4*i)[0]-2)<2e-6,'receiver depth changed')
    require(raw[12*n+i]==1,'receiver validity changed')
    return list(raw[4*i:4*i+3])


def paired_case(library,index,area=False):
    cases=[('front',0.5,0.5,0),('rotated',0.5,0.5,math.pi/4),('reverse_face',0.5,0.5,math.pi),
           ('moved',1.5,0.5,0),('behind',-0.5,-0.5,0),('returned',0.5,0.5,0)]
    if area: cases=[('before_emitter',0.25,0.25,0),('behind_emitter',1,1,0),('clear',1.5,0.5,0)]
    output={}; raws={}
    for mode,kind in (('software',ResidentDevice),('ray_query',RayDevice)):
        device=kind(library,library.parent,index,True); rows=[]; mode_raw=[]
        try:
            buffers=geometry(); device.upload(*buffers)
            for buffer in buffers: ct.memset(ct.addressof(buffer),0,ct.sizeof(buffer))
            if area: set_panel(device)
            for name,x,z,angle in cases:
                p=parameters()
                if area: p.counts[0]=0
                raw=device.render(matrices(x,z,angle),p,65,65)
                rows.append({'case':name,'receiver_rgb':pixel(raw),'sha256':hashlib.sha256(raw).hexdigest()})
                mode_raw.append(raw)
            # Same actual transform, distinct camera: two captures, no frame-key
            # cache shortcut or transformation of saved pixels is permitted.
            p=parameters(camera_x=0.025)
            if area: p.counts[0]=0
            raw=device.render(matrices(cases[-1][1],cases[-1][2],cases[-1][3]),p,65,65)
            rows.append({'case':'second_camera_same_transform','receiver_rgb':pixel(raw),'sha256':hashlib.sha256(raw).hexdigest()})
            mode_raw.append(raw)
            if mode=='ray_query':
                info=query(device)
                require(info['blas_count']==2 and info['primitive_count']==4 and info['blas_builds']==2,'real BLAS mesh/triangle count')
                require(info['tlas_builds']==1 and info['tlas_updates']==len(rows)-1,'TLAS update omitted or stale')
                require(info['instance_upload_bytes']==64*2*len(rows),'AS matrix upload accounting')
                require(info['shadow_captures']==len(rows),'ray-query successful RGB count')
                invalid=matrices(0.5,0.5); invalid[12]=float('nan')
                try: device.render(invalid,p,65,65)
                except (ValueError,RuntimeError): pass
                else: raise AssertionError('nonfinite transform was accepted')
                require(query(device)==info,'invalid input changed acceleration diagnostics')
                output['ray_info']=info
                p.counts[2]=0
                sensor_only=device.render(matrices(cases[-1][1],cases[-1][2],cases[-1][3]),p,65,65)
                require(sensor_only[4*65*65:]==mode_raw[-1][4*65*65:],'sensor-only AS update changed raster sensors')
                final=query(device)
                require(final['tlas_updates']==info['tlas_updates']+1 and final['shadow_captures']==info['shadow_captures'],
                    'sensor-only capture was miscounted or retained an old TLAS')
                output['after_sensor_only']=final
            output[mode]=rows; raws[mode]=mode_raw
        finally: output[mode+'_closed']=_close_clean(device)
    for a,b in zip(raws['software'],raws['ray_query']):
        require(a[4*65*65:]==b[4*65*65:],'hardware shadows changed primary sensor bytes')
    for a,b in zip(output['software'],output['ray_query']):
        require(a['receiver_rgb']==b['receiver_rgb'],'analytic receiver visibility differs between traversal paths')
    brightness=[sum(row['receiver_rgb']) for row in output['ray_query']]
    if area:
        require(brightness[0]==0 and brightness[1]>0 and brightness[1]==brightness[2],
            'finite emitter segment incorrectly includes an occluder behind the light')
    else:
        require(brightness[0]==brightness[1]==brightness[2]==brightness[5]==0 and brightness[3]>0 and brightness[4]>0,
            'front/rotated/moved/behind ray visibility failed')
    return output


def empty_case(library,index):
    device=RayDevice(library,library.parent,index,True)
    try:
        device.upload((ResidentVertex*0)(),(ct.c_uint32*0)(),(ResidentMesh*0)(),(ResidentInstance*0)())
        p=parameters(); p.counts[:]=(0,0,1,0)
        raw=device.render((ct.c_float*0)(),p,65,65); n=65*65
        require(raw[4*n:8*n]==struct.pack('<f',float('inf'))*n and not any(raw[8*n:]),'empty AS changed sensors')
        info=query(device)
        require(info['blas_builds']==0 and info['tlas_builds']==1 and info['tlas_instance_count']==0,'empty TLAS accounting')
        return {'ray_info':info}
    finally: _close_clean(device)


def panda_case(library,index,directory):
    from _panda_scene import build_scene,cameras
    from phymeo_rendering.adapters.panda_replay import PandaReplaySource
    from phymeo_rendering.scene import VisualScene
    from phymeo_rendering.mesh_data import load_mesh
    from phymeo_rendering.backends.native import _parameters,_instance_matrices
    from phymeo_rendering.pipeline import CaptureRequest,RenderPath
    from phymeo_rendering.outputs import Channel
    fixtures=ROOT/'rendering/tests/fixtures/panda_replay'
    initial=PandaReplaySource(VisualScene('input',(),(),()),fixtures/'panda.urdf',fixtures/'trace.json')
    source_geometry=initial.provenance(next(initial.frames()))['geometry']
    directory.mkdir()
    scene=build_scene(ROOT/'rendering/assets/panda',directory,source_geometry['cube_full_size_m'],source_geometry['table_full_size_m'])
    source=PandaReplaySource(scene,fixtures/'panda.urdf',fixtures/'trace.json')
    decoded={mesh.id:load_mesh(mesh) for mesh in scene.meshes}
    instanced=sum(len(decoded[instance.mesh_id].triangles) for instance in scene.instances)
    unique=sum(len(mesh.triangles) for mesh in decoded.values())
    require(instanced==134960,'full Panda geometry was reduced')
    outputs={}; report={'instanced_triangles':instanced,'unique_triangles':unique}
    for mode,kind in (('software',ResidentDevice),('ray_query',RayDevice)):
        device=kind(library,library.parent,index,True); rows=[]; raws=[]
        try:
            device.upload(*pack_scene(scene,decoded))
            for frame in source.frames():
                for original in cameras(frame)[:2]:
                    camera=replace(original,width=320,height=240,fx_px=original.fx_px*320/original.width,
                        fy_px=original.fy_px*240/original.height,cx_px=original.cx_px*320/original.width,
                        cy_px=original.cy_px*240/original.height)
                    request=CaptureRequest(camera,RenderPath.RASTER,tuple(Channel))
                    p=_parameters(scene,request,instanced,1.0); p.counts[3]=len(scene.instances)
                    raw=device.render(_instance_matrices(scene,frame),p,320,240); raws.append(raw)
                    rows.append({'time_s':frame.time_s,'camera':camera.id,'sha256':hashlib.sha256(raw).hexdigest()})
            if mode=='ray_query':
                info=query(device); require(info['primitive_count']==unique and info['blas_count']==len(scene.meshes),'Panda BLAS omitted geometry')
                require(info['tlas_builds']==1 and info['tlas_updates']==len(rows)-1,'Panda/camera matrices were stale')
                report['ray_info']=info
            outputs[mode]=raws; report[mode]=rows
        finally: report[mode+'_closed']=_close_clean(device)
    report['rgb_comparison']=[]
    for a,b in zip(outputs['software'],outputs['ray_query']):
        n=320*240; require(a[4*n:]==b[4*n:],'Panda hardware visibility changed raster sensors')
        delta=[abs(x-y) for x,y in zip(a[:4*n],b[:4*n])]
        report['rgb_comparison'].append({'max_u8':max(delta),'different_components':sum(x!=0 for x in delta)})
    return report


def rejected_creation(library,index,unsupported_index=None):
    class WrongShader(RayDevice): _shader_prefix='resident'
    result=[]
    for label,device_type,selected in (('software_spv',WrongShader,index),('unsupported_device',RayDevice,unsupported_index)):
        if selected is None: continue
        try: owner=device_type(library,library.parent,selected,True)
        except Exception as error:
            reason=str(error)
            require('ray-query' in reason or 'ray_query' in reason,'creation rejected for an unrelated reason')
            result.append({'case':label,'device_index':selected,'error':reason})
        else:
            _close_clean(owner)
            raise AssertionError(f'hardware request silently accepted invalid configuration: {label}')
        lib=ct.CDLL(str(library)); counts=(ct.c_uint64*3)()
        counter=ct.CFUNCTYPE(ct.c_int,ct.POINTER(ct.c_uint64))(('pr_counts',lib))
        require(counter(counts)==0 and not any(counts),'partial hardware-context creation leaked resources')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True); parser.add_argument('--device-index',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--full-panda',action='store_true')
    parser.add_argument('--unsupported-device-index',type=int,help='Explicit negative device from a saved feature report')
    args=parser.parse_args(); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    library=args.library.resolve(); report={'schema':'phymeo-native-rayquery-smoke-v1','status':'running',
        'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest()}
    try:
        lib=ct.CDLL(str(library))
        require(getattr(lib,'pr_resident_ray_query_abi_version',None) is not None,'resident ray-query ABI missing')
        version=ct.CFUNCTYPE(ct.c_uint32)(('pr_resident_ray_query_abi_version',lib))
        require(version()==1,'ray-query ABI version')
        report['rejected_creation']=rejected_creation(library,args.device_index,args.unsupported_device_index)
        report['directional']=paired_case(library,args.device_index)
        report['finite_emitter']=paired_case(library,args.device_index,True)
        report['empty']=empty_case(library,args.device_index)
        if args.full_panda: report['panda']=panda_case(library,args.device_index,output/'geometry')
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}'); raise
    finally: (output/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'status':report['status'],'receipt':str(output/'receipt.json')}))

if __name__=='__main__': main()
