"""Real closed-room/odd/thin capture controls, no fixture-specific shading."""
import argparse
from dataclasses import asdict,replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

from runtime import EdgeBackend,activate


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(data):return sha256(data).hexdigest()


def run(args):
    root,manifest=activate(args.prepared)
    # Explicit public fixture helpers, only after private runtime activation.
    before=list(sys.path)
    try:
        sys.path.insert(0,str(args.examples_root.resolve()))
        import closed_room_bounce_calibration as room
        from _panda_scene import write_mesh
    finally:sys.path[:]=before
    import numpy as np
    from PIL import Image
    from phymeo_rendering.assets import Material
    from phymeo_rendering.scene import Instance,NodePose,Pose
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest,RenderPath
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='phymeo-edge-rescue-smoke-v1',status='started',cases=[],
        source_manifest_sha256=digest((root/'source-manifest.json').read_bytes()),
        full_library_sha256=digest(args.full_library.read_bytes()),
        fixture_sha256=digest(Path(room.__file__).read_bytes()),contract=manifest['contract']['edge_rescue'])
    save(out/'report.json',report)
    try:
        scene,frame,camera,studio=room.build_public(out/'geometry')
        odd=replace(camera,width=97,height=95,cx_px=48.5,cy_px=47.5)
        strip=write_mesh(out/'geometry','edge-thin',((-.01,.1,.2),(.01,.1,.2),(.01,.1,1.8),(-.01,.1,1.8)),
                         ((0.,-1.,0.),)*4,((0,1,2),(0,2,3)))
        thin=replace(scene,key='edge-thin',meshes=scene.meshes+(strip,),
            materials=scene.materials+(Material('edge-thin',(.03,.95,.04),0.,1.),),
            instances=scene.instances+(Instance(99,strip.id,'edge-thin','edge-thin'),))
        thin_frame=replace(frame,key=replace(frame.key,scene_key=thin.key),nodes=frame.nodes+(NodePose('edge-thin',Pose()),))
        cases=(('odd',scene,frame,odd,studio),('thin',thin,thin_frame,replace(odd,frame_key=thin_frame.key),studio),
               ('zero-light',scene,frame,camera,replace(studio,panel_strength=0.,environment_radiance=0.)))
        for name,sc,fr,cam,st in cases:
            path=out/name;path.mkdir();row=dict(name=name,frame=asdict(fr),camera=asdict(cam),outputs={});report['cases'].append(row)
            stored={};linears={}
            for mode in ('full-rate','edge'):
                settings=dict(device_index=args.device_index,studio=st,exposure=1.,display_transform='pbr-neutral',
                              reflection_mode='environment',validation=args.validation)
                if mode=='edge':owner=EdgeBackend(root,seed=0,**settings).open_scene(sc)
                else:owner=NativeBackend(library_path=args.full_library,shader_directory=args.full_shaders or args.full_library.parent,
                    resident=True,shadow_mode='ray_query',msaa_samples=1,indirect_mode='diffuse_bounce',indirect_seed=0,
                    indirect_bounces=1,progressive=True,spatial_filter='none',area_light_grid=2,**settings).open_scene(sc)
                receipt={};row['outputs'][mode]=receipt
                try:
                    req=CaptureRequest(cam,RenderPath.RASTER,tuple(Channel))
                    start=perf_counter();shot=owner.capture(fr,req);elapsed=(perf_counter()-start)*1000
                    if shot.frame_key!=fr.key or shot.camera!=cam or shot.time_s!=fr.time_s:raise ValueError('wrong actual exposure identity')
                    planes={p.channel:p.data for p in shot.planes};stored[mode]=planes
                    Image.frombytes('RGB',(cam.width,cam.height),planes[Channel.RGB]).save(path/(mode+'.png'))
                    packet=owner.read_reconstructed_linear() if mode=='edge' else owner.read_linear_mean()
                    raw=np.frombuffer(packet['data'],'<f4').reshape(cam.height,cam.width,4)
                    if not np.isfinite(raw).all() or np.any(raw[...,:3]<0):raise ValueError('nonfinite actual output')
                    linears[mode]=raw.copy();(path/(mode+'.rgba32f')).write_bytes(packet['data'])
                    receipt.update(capture_ms=elapsed,diagnostics=owner.diagnostics(),plane_sha256={c.value:digest(v) for c,v in planes.items()})
                    if mode=='edge':
                        rescue=owner.read_rescue_incoming();data=np.frombuffer(rescue['data'],'<f4').reshape(cam.height,cam.width,4)
                        valid=(data[...,3].astype(np.uint32)&(1<<20))!=0
                        count=int(valid.sum());actual=receipt['diagnostics']['edge_rescue']
                        if count!=actual['selected_pixels']:raise ValueError('actual rescue attachment/count mismatch')
                        ids=np.frombuffer(planes[Channel.INSTANCE_ID],'<u4').reshape(cam.height,cam.width)
                        receipt['rescued_by_id']={str(i):int(((ids==i)&valid).sum()) for i in np.unique(ids) if i}
                        if name=='thin' and receipt['rescued_by_id'].get('99',0)==0:raise ValueError('thin no-anchor control received no genuine rescue')
                        (path/'rescue.rgba32f').write_bytes(rescue['data'])
                        changed=replace(cam,width=99,height=93,cx_px=49.5,cy_px=46.5)
                        owner.capture(fr,CaptureRequest(changed,RenderPath.RASTER,(Channel.INSTANCE_ID,)))
                        try:owner.read_rescue_incoming()
                        except RuntimeError:pass
                        else:raise ValueError('stale rescue after sensor-only capture')
                        repeated=owner.capture(fr,req)
                        if {p.channel:p.data for p in repeated.planes}!=planes:raise ValueError('real resize/reset changed same-state result')
                        receipt['repeat_after_resize_exact']=True
                finally:
                    active_error=sys.exc_info()[0] is not None
                    try:
                        owner.close();receipt['closed']=owner.diagnostics();device=receipt['closed']['device']
                        if not device.get('closed') or any(device.get(k,1) for k in ('live_contexts','live_frames','live_resources','validation_errors')):
                            raise ValueError('real close/resource/validation failure')
                    except BaseException as error:
                        receipt['close_error']=str(error)
                        if not active_error:raise
                    finally:save(path/'receipt.json',row)
            if any(stored['edge'][c]!=stored['full-rate'][c] for c in Channel if c is not Channel.RGB):
                raise ValueError('rescue changed actual full-rate sensor bytes')
            if name=='zero-light' and any(np.any(v[...,:3]) for v in linears.values()):raise ValueError('zero light emitted energy')
            row['full_sensor_bytes_identical']=True
            row['scene_linear_difference_rmse']=float(np.sqrt(np.mean((linears['edge'][...,:3].astype(float)-linears['full-rate'][...,:3])**2)))
            save(path/'receipt.json',row);save(out/'report.json',report)
        report['status']='complete'
    except BaseException as error:
        report['status']='partial_failure';report['error']=dict(type=type(error).__name__,message=str(error));raise
    finally:save(out/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--examples-root',type=Path,required=True)
    p.add_argument('--full-library',type=Path,required=True);p.add_argument('--full-shaders',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device-index',type=int,default=1)
    p.add_argument('--validation',action='store_true');run(p.parse_args())

if __name__=='__main__':main()