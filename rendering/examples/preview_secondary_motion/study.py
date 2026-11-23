"""Offscreen secondary motion: distinguish retained incoming from noisy samples.

Private rigid4 consumer only. No renderer import before activate; no algorithm
change. Every threshold/geometry/ROI is registered before the first capture.
"""
import argparse
from dataclasses import asdict,replace
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter
import traceback
import zlib

WIDTH=HEIGHT=96
FOCAL=160.
ROI=(32,32,64,64)
OPEN=8
FRAMES=16
SENSORS=('FORWARD_DEPTH','INSTANCE_ID','VALID')


def protocol():
    return dict(schema='phymeo-preview-secondary-motion-v1',width=WIDTH,height=HEIGHT,focal_px=FOCAL,
        eye=[0.,0.,1.],roi=list(ROI),opening_index=OPEN,frames=FRAMES,
        motion_seeds=[0,1],reference_seeds=list(range(100,132)),reference_groups=[list(range(100,116)),list(range(116,132))],
        first_open_frames=[8,9,10,11],reference='32 independent base-seed current-estimator incoming means per static phase; not unbiased PT',
        signal_min=.01,signal_se_multiplier=5.,stale_fraction_min=.10,stale_se_multiplier=3.,
        stale_rule='positive old-phase color projection in history minus paired current, >=10% phase change, >3 seed-mean SE, both seeds positive; history itself must retain positive old-phase projection',
        uncertainty='seed is replication unit, never spatially correlated ROI pixels; two motion seeds have limited power',
        geometry='receiver and camera fixed; only opaque black offscreen shutter translates y=0 to y=2 at frame8',
        primary_guard='independent full-frame rectangle-ray oracle plus actual identical ID/depth/VALID; receiver ID1 everywhere; no IDs2/3',
        lighting='fixed directional travel (+1,0,-1)/sqrt2, illuminance (3,3,3), no environment or panels',
        shutter_motion_is_authored=True,physics_steps=0,history_age_cap=4,exposure=1.,display_transform='clip',
        expected_captures=136,geometry_closed=geometry(False),geometry_open=geometry(True))


def geometry(opened):
    return [dict(id=1,name='receiver',origin=(-.35,-.35,0.),u=(.7,0.,0.),v=(0.,.7,0.),normal=(0.,0.,1.),color=(.7,.7,.7)),
        dict(id=2,name='red-wall',origin=(.5,.5,.02),u=(0.,-1.,0.),v=(0.,0.,.88),normal=(-1.,0.,0.),color=(.85,.01,.01)),
        dict(id=3,name='shutter',origin=(.4,.6+(2. if opened else 0.),.01),u=(0.,-1.2,0.),v=(0.,0.,.94),normal=(-1.,0.,0.),color=(0.,0.,0.))]


def dot(a,b):return sum(x*y for x,y in zip(a,b))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))


def primary_witness(opened):
    counts={1:0,2:0,3:0};depths=[];eye=(0.,0.,1.)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            direction=((x+.5-WIDTH/2)/FOCAL,-(y+.5-HEIGHT/2)/FOCAL,-1.)
            hits=[]
            for face in geometry(opened):
                denominator=dot(direction,face['normal'])
                if abs(denominator)<1e-14:continue
                t=dot(sub(face['origin'],eye),face['normal'])/denominator
                if not .01<t<5.:continue
                p=tuple(eye[i]+t*direction[i] for i in range(3));d=sub(p,face['origin'])
                a=dot(d,face['u'])/dot(face['u'],face['u']);b=dot(d,face['v'])/dot(face['v'],face['v'])
                if 0<=a<=1 and 0<=b<=1:hits.append((t,face['id']))
            if not hits:raise ValueError('registered primary ray misses authored receiver')
            depth,identity=min(hits);counts[identity]+=1;depths.append(depth)
    return dict(receiver_pixels=counts[1],secondary_primary_pixels=counts[2]+counts[3],min_depth=min(depths),max_depth=max(depths))


def mean_rgb(values):return [statistics.mean(v[k] for v in values) for k in range(3)]
def se(values):return statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else 0.
def red(v):return v[0]-.5*(v[1]+v[2])


def score(closed,opened,history,current):
    for values,n in ((closed,32),(opened,32),(history,2),(current,2)):
        if len(values)!=n or any(len(v)!=3 or any(not math.isfinite(c) or c<0 for c in v) for v in values):
            raise ValueError('complete finite nonnegative registered seed RGB means required')
    before,after=mean_rgb(closed),mean_rgb(opened)
    delta=sub(before,after);norm2=dot(delta,delta)
    signal=red(after)-red(before);signal_se=math.hypot(se([red(v) for v in closed]),se([red(v) for v in opened]))
    valid=signal>=.01 and signal>5.*signal_se and norm2>0
    result=dict(outcome='insufficient_secondary_signal',measured_stale_indirect=False,signal=signal,signal_se=signal_se,
        reference_closed=before,reference_open=after,reference_group_difference={name:sub(mean_rgb(v[:16]),mean_rgb(v[16:])) for name,v in (('closed',closed),('open',opened))},
        history_mean=mean_rgb(history),current_mean=mean_rgb(current),sufficient_signal=valid,
        incremental_old_fraction=None,incremental_old_fraction_se=None)
    if not valid:return result
    old_h=[dot(sub(v,after),delta)/norm2 for v in history]
    old_c=[dot(sub(v,after),delta)/norm2 for v in current]
    difference=[a-b for a,b in zip(old_h,old_c)]
    ref_uncertainty=se([dot(sub(v,after),delta)/norm2 for v in opened])
    estimate=statistics.mean(difference);uncertainty=se(difference)
    detected=(estimate>=.10 and estimate>3.*uncertainty and min(difference)>0 and statistics.mean(old_h)>max(.10,3.*ref_uncertainty))
    result.update(outcome='measured_stale_indirect' if detected else 'no_registered_stale_signal',measured_stale_indirect=detected,
        incremental_old_fraction=estimate,incremental_old_fraction_se=uncertainty,per_seed_old_fraction_history=old_h,
        per_seed_old_fraction_current=old_c,per_seed_incremental_old_fraction=difference,reference_projection_se=ref_uncertainty,
        history_seed_mean_rmse=math.sqrt(statistics.mean((v[k]-after[k])**2 for v in history for k in range(3))),
        current_seed_mean_rmse=math.sqrt(statistics.mean((v[k]-after[k])**2 for v in current for k in range(3))))
    return result


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def build_scene(root):
    from _panda_scene import write_mesh
    from phymeo_rendering.assets import Material
    from phymeo_rendering.scene import VisualScene,Instance,DirectionalLight,SceneFrame,FrameKey,Pose,NodePose
    from phymeo_rendering.cameras import Camera
    root.mkdir(parents=True,exist_ok=True)
    faces=geometry(False);meshes=[]
    for face in faces:
        verts=[tuple(face['origin'][i]+a*face['u'][i]+b*face['v'][i] for i in range(3)) for a,b in ((0,0),(1,0),(1,1),(0,1))]
        meshes.append(write_mesh(root,face['name'],verts,(face['normal'],)*4,((0,1,2),(0,2,3))))
    scene=VisualScene('offscreen-secondary-motion',tuple(meshes),tuple(Material(f['name'],f['color'],0.,1.) for f in faces),
        tuple(Instance(f['id'],f['name'],f['name'],f['name']) for f in faces),
        (DirectionalLight((math.sqrt(.5),0.,-math.sqrt(.5)),(3.,3.,3.)),))
    frames=[];cameras=[]
    for index in range(FRAMES):
        key=FrameKey(scene.key,'synthetic-secondary',0,index,index,'authored')
        nodes=tuple(NodePose(f['name'],Pose((0.,2.,0.)) if f['name']=='shutter' and index>=OPEN else Pose()) for f in faces)
        frame=SceneFrame(key,index/30.,nodes);frames.append(frame)
        cameras.append(Camera('fixed-receiver',key,Pose((0.,0.,1.)),WIDTH,HEIGHT,FOCAL,FOCAL,WIDTH/2,HEIGHT/2,.01,5.))
    return scene,frames,cameras


def run(args):
    sys.path.insert(0,str(args.examples.resolve()))
    from preview_compact_rigid_history.runtime import activate,RigidHistoryBackend
    from preview_compact_rigid_history.state import bind_presentations,frame_digest
    prepared,manifest=activate(args.prepared)
    if manifest['contract'].get('rigid_history',{}).get('schema')!='phymeo-preview-compact-rigid-history-v1' or 'local_diffuse' in manifest['contract']:
        raise ValueError('registered full-distance rigid history package required; no finite-range substitute')
    for relative,expected in manifest['generated_sha256'].items():
        path=(prepared/relative).resolve()
        if not path.is_relative_to(prepared) or digest(path)!=expected:raise ValueError('prepared source changed')
    import numpy as np
    from PIL import Image
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest,RenderPath
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='phymeo-preview-secondary-motion-result-v1',status='running',protocol=protocol(),
        prepared=str(prepared),manifest_sha256=digest(prepared/'source-manifest.json'),dll_sha256=digest(prepared/'build/phymeo_native.dll'),
        shader_sha256={p.name:digest(p) for p in sorted((prepared/'build').glob('*.spv'))},author_script_sha256=digest(__file__),records=[],cleanup=[],scores={})
    save(out/'registration.json',report)
    global_sensors=None;global_primary=None
    dt=np.dtype([('incoming','<f4',(4,)),('geometry','<u4',(4,)),('lighting','<f4',(4,))])
    x0,y0,x1,y1=ROI
    def pack(key,name,data):
        path=out/(key+'.'+name+'.zlib');path.write_bytes(zlib.compress(data,6))
        return dict(file=path.name,sha256=sha256(data).hexdigest(),compressed_sha256=digest(path),bytes=len(data))
    def opened_owner(seed,reuse):
        return RigidHistoryBackend(prepared,seed=seed,reuse_history=reuse,device_index=args.device_index,
            exposure=1.,display_transform='clip',reflection_mode='environment',validation=True).open_scene(scene)
    def close(owner,label):
        owner.close();diag=owner.diagnostics();device=diag['device'];t=device.get('resident_telemetry') or {}
        good=diag['closed'] and all(device.get(k)==0 for k in ('live_contexts','live_frames','live_resources','validation_errors'))
        good=good and t.get('final_after_destroy') is True and t.get('device_local_bytes_current')==0 and t.get('host_visible_bytes_current')==0 and t.get('allocation_count')==t.get('free_count')
        report['cleanup'].append(dict(owner=label,passed=bool(good),diagnostics=diag))
        if not good:raise ValueError('real teardown failed')
    def capture(owner,index,binding,key):
        nonlocal global_sensors,global_primary
        request=CaptureRequest(cameras[index],RenderPath.RASTER,tuple(Channel))
        started=perf_counter();shot=owner.capture(frames[index],request,presentation=binding);wall=1000*(perf_counter()-started)
        if shot.frame_key!=frames[index].key or shot.camera!=cameras[index] or shot.time_s!=frames[index].time_s:raise ValueError('publication identity changed')
        planes={p.channel.name:p.data for p in shot.planes}
        sensors={k:planes[k] for k in SENSORS}
        if global_sensors is None:global_sensors=sensors
        if sensors!=global_sensors:raise ValueError('main-ray geometry changed; secondary-only claim invalid')
        ids=np.frombuffer(planes['INSTANCE_ID'],'<u4');valid=np.frombuffer(planes['VALID'],'u1');depth=np.frombuffer(planes['FORWARD_DEPTH'],'<f4')
        if not (np.all(ids==1) and np.all(valid==1) and np.all(depth==1.)):raise ValueError('actual primary oracle differs or offscreen object is primary-visible')
        current=owner.read_current_incoming();history=owner.read_history_incoming();total=owner.read_reconstructed_linear()
        if current['format']!='RGBA32F_LE' or history['format']!='F32x4_U32x4_F32x4_LE':raise ValueError('actual incoming format differs')
        a=np.frombuffer(current['data'],'<f4').reshape(HEIGHT,WIDTH,4);b=np.frombuffer(history['data'],dt).reshape(HEIGHT,WIDTH)
        primary=b['lighting'].tobytes()
        if global_primary is None:global_primary=primary
        if primary!=global_primary:raise ValueError('current direct/specular/rho changed; not isolated incoming')
        ages=b['incoming'][...,3];support=a[...,3]
        if not np.isfinite(a).all() or not np.isfinite(b['incoming']).all() or np.any(a[...,:3]<0) or np.any(b['incoming'][...,:3]<0):raise ValueError('invalid actual incoming')
        if np.any(ages<0) or np.any(ages>4) or np.any(ages!=np.floor(ages)):raise ValueError('history age outside declared cap')
        if not np.all(support[y0:y1,x0:x1]==1):raise ValueError('fixed ROI lacks current traced support')
        roi_a=a[y0:y1,x0:x1,:3];roi_b=b['incoming'][y0:y1,x0:x1,:3]
        row=dict(key=key,frame_index=index,opened=index>=OPEN,presentation=binding,wall_ms=wall,
            current_mean=roi_a.mean(axis=(0,1),dtype=np.float64).tolist(),history_mean=roi_b.mean(axis=(0,1),dtype=np.float64).tolist(),
            age_histogram={str(i):int(np.count_nonzero(ages[y0:y1,x0:x1]==i)) for i in range(5)},
            history_current_rmse=float(np.sqrt(np.mean((roi_b.astype(float)-roi_a.astype(float))**2))),
            planes={k:pack(key,k,v) for k,v in planes.items()},current=pack(key,'current',current['data']),history=pack(key,'history',history['data']),
            total=pack(key,'total',total['data']),primary_sha256=sha256(primary).hexdigest(),diagnostics=owner.diagnostics())
        Image.frombytes('RGB',(WIDTH,HEIGHT),planes['RGB']).save(out/(key+'.png'))
        report['records'].append(row);save(out/'report.json',report)
        return row
    try:
        witness=primary_witness(False)
        if witness!=primary_witness(True) or witness['secondary_primary_pixels']!=0:raise ValueError('registered analytic primary independence failed')
        scene,frames,cameras=build_scene(out/'geometry')
        inputs=dict(scene=asdict(scene),frames=[asdict(f) for f in frames],cameras=[asdict(c) for c in cameras],primary_witness=witness)
        save(out/'inputs.json',inputs);source=digest(out/'inputs.json');bindings=bind_presentations(frames,source)
        references={phase:[] for phase in ('closed','open')}
        for phase,index in (('closed',0),('open',OPEN)):
            one=bind_presentations([frames[index]],source,start_index=index)[0]
            for seed in protocol()['reference_seeds']:
                owner=opened_owner(seed,False);label=f'reference-{phase}-s{seed}'
                try:references[phase].append(capture(owner,index,one,label)['current_mean'])
                finally:close(owner,label)
        current_records={};history_records={}
        for seed in protocol()['motion_seeds']:
            for reuse,dest,label in ((False,current_records,'current'),(True,history_records,'history')):
                owner=opened_owner(seed,reuse)
                try:
                    for index in range(FRAMES):dest[seed,index]=capture(owner,index,bindings[index],f'{label}-s{seed}-f{index:02d}')
                finally:close(owner,f'{label}-s{seed}')
            for index in range(OPEN,OPEN+4):
                c=current_records[seed,index];h=history_records[seed,index]
                if c['current']['sha256']!=h['current']['sha256']:raise ValueError('paired current estimator differs')
                actual_seed=c['diagnostics']['history']['effective_seed']
                if actual_seed!=h['diagnostics']['history']['effective_seed']:raise ValueError('paired effective seed differs')
                owner=opened_owner(actual_seed,True);label=f'cold-s{seed}-f{index:02d}'
                try:
                    cold=capture(owner,index,bind_presentations([frames[index]],source,start_index=index)[0],label)
                    if cold['current']['sha256']!=c['current']['sha256'] or cold['total']['sha256']!=c['total']['sha256'] or cold['planes']['RGB']['sha256']!=c['planes']['RGB']['sha256']:
                        raise ValueError('same-phase effective-seed cold control differs')
                    if cold['age_histogram']['1']!=(x1-x0)*(y1-y0):raise ValueError('cold phase retains history')
                finally:close(owner,label)
        for index in range(FRAMES):
            for seed in protocol()['motion_seeds']:
                c,h=current_records[seed,index],history_records[seed,index]
                if c['current']['sha256']!=h['current']['sha256'] or c['diagnostics']['history']['effective_seed']!=h['diagnostics']['history']['effective_seed']:raise ValueError('motion source/current mismatch')
                if index==0 and (c['total']['sha256']!=h['total']['sha256'] or c['planes']['RGB']['sha256']!=h['planes']['RGB']['sha256']):raise ValueError('fresh initial owner differs')
        for index in range(OPEN,FRAMES):
            report['scores'][str(index-OPEN+1)]=score(references['closed'],references['open'],
                [history_records[s,index]['history_mean'] for s in protocol()['motion_seeds']],
                [current_records[s,index]['current_mean'] for s in protocol()['motion_seeds']])
        first=[report['scores'][str(i)] for i in range(1,5)]
        report.update(status='complete',sufficient_signal=all(x['sufficient_signal'] for x in first),
            measured_stale_first_four=any(x['measured_stale_indirect'] for x in first),
            scope='bounded fixed offscreen change; no claim that weak/no detected signal proves absence in other scenes',
            captures=len(report['records']),input_sha256=source,cleanup_passed=all(x['passed'] for x in report['cleanup']))
        if report['captures']!=protocol()['expected_captures']:raise ValueError('registered capture count differs')
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    finally:save(out/'summary.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','examples','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device-index',type=int,default=1)
    result=run(parser.parse_args())
    return 0 if result['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())