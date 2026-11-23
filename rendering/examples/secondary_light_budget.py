"""Bounded public Vulkan A/B: unchanged primary grid, explicit terminal budget.

This experiment does not certify Panda's 25% speed or pixel/edge quality gates.
It reuses the frozen six-wall room and registers its own thin-occluder controls.
Room radiance comparisons use four paired seeds and real GPU mean. The thin
indirect diagnostic has one seed, separately labelled without a quality gate.
No material, light, camera, exposure, sampling or threshold is fitted to results.
"""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from time import perf_counter
import traceback

SENSORS=('FORWARD_DEPTH','INSTANCE_ID','VALID')
CASES=('direct','room','thin','thin-indirect','mirror','zero','single')


def protocol():
    return dict(schema='phymeo-secondary-light-budget-registration-v1',budgets=[[8,8],[8,2]],
        seeds=[0,1,2,3],batches=16,rounds=2,bounces=[1,2],primary_samples_per_batch=16,
        extent=[96,96],relative_room_energy_limit=.05,reference_signal_min_sem_multiple=2.,
        room_rois={'bright':[14,44,34,64],'shadow':[72,44,84,64],'red':[22,71,32,79]},
        thin_width_m=.02,thin_offsets_m=[-.12,0.,.12],ggx_proposals_per_panel=8,
        thin_indirect=dict(seeds=[0],bounces=2,batches=16,phases=[-1,0,1],
            quality_acceptance_claimed=False,
            scope='paired single-seed thin-occluder diagnostic; 16 progressive batches are not independent base seeds'),
        environment=0.,exposure=1.,msaa_samples=1,
        primary_and_guide='8x8 midpoint; same original direct guide SPIR-V',
        secondary='first diffuse hit, terminal diffuse hit and near-mirror hit: 2x2 vs 8x8',
        singles='one directional light and zero panel strength; public studio has no individual-panel toggle',
        statistics='four paired base seeds; repeat round is a timing/replay control, not independent evidence',
        roi_policy='registered before measurements; dark/reference-within-two-SEM is insufficient, never silently passed',
        timing='per-capture host wall and actual device GPU interval; no claim of separately timed primary/secondary kernels',
        panda_quality_proven_by_this_trial=False,external_pt_reference_recomputed=False)


def energy_comparison(reference,actual):
    if len(reference)!=4 or len(actual)!=4 or any(not math.isfinite(x) or x<0 for x in (*reference,*actual)):
        raise ValueError('four matched finite nonnegative seed energies required')
    mean=statistics.mean(reference);other=statistics.mean(actual)
    changes=[b-a for a,b in zip(reference,actual)]
    sem=statistics.stdev(reference)/2;paired_sem=statistics.stdev(changes)/2
    relative=(other-mean)/mean if mean>0 else None
    sufficient=mean>0 and mean>2*sem
    return dict(reference_seeds=list(reference),actual_seeds=list(actual),reference_mean=mean,actual_mean=other,
        absolute_change=other-mean,relative_change=relative,reference_sample_sd=statistics.stdev(reference),
        actual_sample_sd=statistics.stdev(actual),reference_sem=sem,paired_difference_sem=paired_sem,
        status=('insufficient' if not sufficient else 'passed' if abs(relative)<=.05 else 'failed'),
        limit=.05,uncertainty='sample SD/SEM across four fixed base seeds; no pixel-IID claim')


def identical_primary(left,right):
    return all(left.get(name)==right.get(name) for name in ('RGB',*SENSORS))


def wrong_primary_detected(left,right):
    """Grid sensitivity requires changed RGB with unchanged geometry channels."""
    return (all(name in left and name in right for name in ('RGB',*SENSORS))
            and all(left[name]==right[name] for name in SENSORS)
            and left['RGB']!=right['RGB'])


def build_case(root,kind,*,phase=0):
    if kind not in CASES:raise ValueError('unregistered case')
    import closed_room_bounce_calibration as room
    from _panda_scene import write_mesh
    from phymeo_rendering.scene import DirectionalLight,NodePose,Pose
    scene,frame,camera,studio=room.build_public(Path(root))
    if kind in ('thin','thin-indirect'):
        if phase not in (-1,0,1):raise ValueError('registered thin phase is -1,0,1')
        face=dict(room.geometry()[-1]);face['u']=(.02,0.,0.)
        mesh=write_mesh(Path(root),'thin-baffle',room.vertices(face),(face['normal'],)*4,((0,1,2),(0,2,3)))
        scene=replace(scene,meshes=scene.meshes[:-1]+(mesh,),
            instances=scene.instances[:-1]+(replace(scene.instances[-1],mesh_id=mesh.id),))
        frame=replace(frame,nodes=frame.nodes[:-1]+(NodePose('baffle',Pose((phase*.12,0.,0.))),))
    if kind=='mirror':
        scene=replace(scene,materials=tuple(replace(m,metallic=1.,roughness=.04) if m.id=='back' else m for m in scene.materials))
    if kind in ('zero','single'):
        studio=replace(studio,panel_strength=0.,environment_radiance=0.)
    if kind=='single':
        # An open, actually illuminated receiver: an external directional light
        # would correctly be blocked by a closed room's front wall.
        scene=replace(scene,meshes=(scene.meshes[4],),materials=(scene.materials[4],),instances=(scene.instances[4],),
            lights=(DirectionalLight((0.,1.,0.),(1.,1.,1.)),))
        frame=replace(frame,nodes=(frame.nodes[4],))
    return scene,frame,camera,studio


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def run(library,output,*,shaders=None,device_index=1,validation=True,cases=CASES,baseline_library=None,baseline_shaders=None):
    import numpy as np
    from PIL import Image
    import phymeo_rendering
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest,RenderPath
    from near_reflection_chart import closed
    import closed_room_bounce_calibration as room
    cases=tuple(cases)
    if not cases or len(set(cases))!=len(cases) or any(x not in CASES for x in cases):raise ValueError('unique registered cases required')
    library=Path(library).resolve();shaders=Path(shaders or library.parent).resolve()
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='phymeo-secondary-light-budget-trial-v1',status='running',registration=protocol(),
        physics_steps=0,source=str(Path(phymeo_rendering.__file__).resolve()),script_sha256=digest(__file__),
        library=dict(path=str(library),sha256=digest(library)),shader_directory=str(shaders),
        cases=list(cases),captures=[],cleanup=[],pairs=[],scenes={},functional_passed=False,
        all_cleanup_passed=False,panda_acceptance_claimed=False,room_quality=None,thin_indirect_diagnostics=[],
        wrong_primary_negative_control_detected=None,baseline_artifact_equivalence=None)
    dump(output/'registration.json',report)
    saved={};memo={};started=perf_counter();luma=np.asarray((.2126,.7152,.0722))
    def save_bytes(label,suffix,data):
        name=label+'.'+suffix;(output/name).write_bytes(data)
        return dict(file=name,sha256=sha256(data).hexdigest(),bytes=len(data))
    def shot(kind,phase,depth,seed,round_index,variant):
        case_key=f'{kind}-{phase}'
        if case_key not in memo:
            memo[case_key]=build_case(output/'geometry'/case_key,kind,phase=phase)
            scene,frame,camera,studio=memo[case_key]
            report['scenes'][case_key]=dict(scene=asdict(scene),frame=asdict(frame),camera=asdict(camera),studio=asdict(studio))
        scene,frame,camera,studio=memo[case_key]
        primary=2 if variant=='wrong-primary' else 8
        secondary=2 if variant=='split' else None if variant in ('inherit','baseline') else primary
        progressive=depth>0;batch_count=16 if progressive else 1
        use_guide=progressive and seed==0 and round_index==0
        selected_library=Path(baseline_library).resolve() if variant=='baseline' else library
        selected_shaders=Path(baseline_shaders or selected_library.parent).resolve() if variant=='baseline' else shaders
        label=f'{case_key}-d{depth}-s{seed}-r{round_index}-{variant}'
        owner=None;record=dict(label=label,case=kind,phase=phase,bounces=depth,seed=seed,round=round_index,
            variant=variant,primary_grid=primary,secondary_grid=primary if secondary is None else secondary,batches=batch_count)
        report['captures'].append(record)
        try:
            options=dict(library_path=selected_library,shader_directory=selected_shaders,device_index=device_index,
                validation=validation,resident=True,shadow_mode='ray_query',area_light_grid=primary,studio=studio,
                indirect_mode='diffuse_bounce' if progressive else 'none',indirect_seed=seed,
                progressive=progressive,indirect_bounces=depth if progressive else 1,
                spatial_filter='guided' if use_guide else 'none',reflection_mode='single_bounce' if kind=='mirror' else 'environment')
            # Baseline/default deliberately omit the new argument entirely.
            if secondary is not None:options['secondary_area_light_grid']=secondary
            owner=NativeBackend(**options).open_scene(scene)
            request=CaptureRequest(camera,RenderPath.RASTER,tuple(Channel));times=[];previous_gpu=0;guide=None
            first=None;last=None;raw=None
            for batch in range(1,batch_count+1):
                start=perf_counter();capture=owner.capture(frame,request);wall=perf_counter()-start
                if capture.frame_key!=frame.key or capture.camera!=camera or capture.time_s!=frame.time_s:
                    raise ValueError('capture source/camera/time mismatch')
                planes={p.channel.name:p.data for p in capture.planes}
                diagnostic=owner.diagnostics();profile=diagnostic['device'].get('resident_profile')
                cumulative=profile['gpu_render_ns'] if profile and profile.get('timestamp_samples',0)>0 else None
                times.append(dict(batch=batch,wall_ms=wall*1000,gpu_render_ms=(cumulative-previous_gpu)/1e6 if cumulative is not None else None))
                if cumulative is not None:previous_gpu=cumulative
                if batch==1:
                    first=planes
                    if use_guide:
                        guide=owner.read_spatial_guide()['data'];record['guide']=save_bytes(label,'guide.rgba8',guide)
                elif any(first[n]!=planes[n] for n in SENSORS):raise ValueError('static capture changed geometric channels')
                if progressive and batch in (1,batch_count):
                    payload=owner.read_linear_mean();room.validate_mean(payload,camera.width,camera.height,batch)
                    raw=payload['data'];rgba=np.frombuffer(raw,'<f4').reshape(camera.height,camera.width,4)
                    if not np.isfinite(rgba).all() or np.any(rgba[...,:3]<0):raise ValueError('invalid scene-linear mean')
                    record[f'linear_batch{batch}']=save_bytes(label,f'batch{batch}.linear-rgba-f32.raw',raw)
                last=planes
            record['planes']={name:save_bytes(label,name+'.raw',data) for name,data in last.items()}
            png=output/(label+'.png');Image.frombytes('RGB',(camera.width,camera.height),last['RGB']).save(png)
            record.update(png=dict(file=png.name,sha256=digest(png)),timing=times,diagnostics=diagnostic)
            rgb=np.frombuffer(raw,'<f4').reshape(camera.height,camera.width,4)[...,:3].astype(float) if progressive else None
            if rgb is not None:
                record['roi_luminance']={name:float(np.mean(rgb[y0:y1,x0:x1]@luma)) for name,(x0,y0,x1,y1) in room.ROIS.items()}
            saved[(kind,phase,depth,seed,round_index,variant)]=dict(planes=last,first=first,raw=raw,guide=guide,record=record)
            return saved[(kind,phase,depth,seed,round_index,variant)]
        except BaseException as error:
            record['error']=repr(error);raise
        finally:
            if owner is not None:
                try:owner.close()
                finally:
                    diagnostic=owner.diagnostics();report['cleanup'].append(dict(label=label,passed=closed(diagnostic),diagnostics=diagnostic))
            dump(output/'summary.json',report)
    def compare_pair(a,b,primary=False):
        sensors=all(a['planes'][n]==b['planes'][n] for n in SENSORS)
        guide=a['guide']==b['guide'] if a['guide'] is not None or b['guide'] is not None else None
        diff=np.abs(np.frombuffer(a['planes']['RGB'],np.uint8).astype(int)-np.frombuffer(b['planes']['RGB'],np.uint8).astype(int))
        pair=dict(left=a['record']['label'],right=b['record']['label'],sensor_bytes_equal=sensors,
            guide_bytes_equal=guide,primary_isolation=primary,rgb_bytes_equal=a['planes']['RGB']==b['planes']['RGB'],
            display_absolute_code_p99=float(np.quantile(diff,.99)),display_absolute_code_max=int(diff.max()),
            scope='whole-frame display differences only; not Panda or registered half-shadow edge acceptance')
        report['pairs'].append(pair)
        return sensors and guide is not False and (not primary or identical_primary(a['planes'],b['planes']))
    try:
        functional=True
        for kind in cases:
            if kind=='room':continue
            for phase in ((-1,0,1) if kind in ('thin','thin-indirect') else (0,)):
                depth=2 if kind=='thin-indirect' else 0
                full=shot(kind,phase,depth,0,0,'full');split=shot(kind,phase,depth,0,0,'split')
                functional &= compare_pair(full,split,primary=kind not in ('mirror','thin-indirect'))
                if kind=='thin-indirect':
                    a=np.frombuffer(full['raw'],'<f4').reshape(-1,4)[:,:3].astype(float)
                    b=np.frombuffer(split['raw'],'<f4').reshape(-1,4)[:,:3].astype(float)
                    delta=b-a
                    rois={}
                    for name in room.ROIS:
                        av=full['record']['roi_luminance'][name];bv=split['record']['roi_luminance'][name]
                        rois[name]=dict(full=av,split=bv,signed_difference=bv-av,
                            relative_difference=(bv-av)/av if av>0 else None)
                    report['thin_indirect_diagnostics'].append(dict(phase=phase,seed=0,bounces=depth,batches=16,
                        full=full['record']['label'],split=split['record']['label'],
                        scene_linear_rgb_rmse=float(np.sqrt(np.mean(delta*delta))),
                        scene_linear_rgb_signed_mean=float(np.mean(delta)),roi_luminance=rois,
                        quality_status='diagnostic-only-single-seed',quality_acceptance_claimed=False,
                        uncertainty='no independent-seed uncertainty estimated; not included in room energy gates'))
                if kind=='zero':functional &= not any(full['planes']['RGB']) and not any(split['planes']['RGB'])
                if kind=='single':functional &= any(full['planes']['RGB'])
                if kind=='mirror':
                    for value in (full,split):
                        info=value['record']['diagnostics']['device'].get('resident_reflection_info') or {}
                        functional &= info.get('enabled')==1 and info.get('receiver_count',0)>0
                if kind=='direct':
                    inherit=shot(kind,phase,0,0,0,'inherit');functional &= compare_pair(full,inherit,primary=True)
                    wrong=shot(kind,phase,0,0,0,'wrong-primary')
                    report['wrong_primary_negative_control']=dict(full=full['record']['label'],wrong=wrong['record']['label'],
                        sensor_bytes_equal=all(full['planes'][n]==wrong['planes'][n] for n in SENSORS),
                        rgb_bytes_differ=full['planes']['RGB']!=wrong['planes']['RGB'])
                    report['wrong_primary_negative_control_detected']=wrong_primary_detected(full['planes'],wrong['planes'])
                    functional &= report['wrong_primary_negative_control_detected']
                    if baseline_library:
                        original=shot(kind,phase,0,0,0,'baseline')
                        report['baseline_artifact_equivalence']=compare_pair(full,original,primary=True)
                        functional &= report['baseline_artifact_equivalence']
        if 'room' in cases:
            for round_index in (0,1):
                order=('full','split') if round_index==0 else ('split','full')
                for depth in (1,2):
                    for seed in (0,1,2,3):
                        pair={variant:shot('room',0,depth,seed,round_index,variant) for variant in order}
                        functional &= compare_pair(pair['full'],pair['split'])
                        if round_index:
                            for variant in order:
                                prior=saved[('room',0,depth,seed,0,variant)]
                                # Last n16 is unfiltered in both rounds. Round1 has no
                                # guide draw, so it is not an independent shader estimate.
                                functional &= prior['raw']==pair[variant]['raw'] and identical_primary(prior['planes'],pair[variant]['planes'])
            inherited=shot('room',0,1,0,0,'inherit');full=saved[('room',0,1,0,0,'full')]
            functional &= compare_pair(full,inherited,primary=True) and full['raw']==inherited['raw']
            if baseline_library:
                original=shot('room',0,1,0,0,'baseline')
                equivalent=compare_pair(full,original,primary=True) and full['raw']==original['raw']
                report['baseline_artifact_equivalence']=equivalent and report['baseline_artifact_equivalence'] is not False
                functional &= equivalent
            quality={};timing={}
            for depth in (1,2):
                quality[str(depth)]={name:energy_comparison(
                    [saved[('room',0,depth,s,0,'full')]['record']['roi_luminance'][name] for s in (0,1,2,3)],
                    [saved[('room',0,depth,s,0,'split')]['record']['roi_luminance'][name] for s in (0,1,2,3)]) for name in room.ROIS}
                timing[str(depth)]={}
                for round_index in (0,1):
                    costs={}
                    for variant in ('full','split'):
                        samples=[t['gpu_render_ms'] for seed in (0,1,2,3) for t in saved[('room',0,depth,seed,round_index,variant)]['record']['timing'][1:]]
                        costs[variant]=statistics.mean(samples) if all(t is not None for t in samples) else None
                    timing[str(depth)][str(round_index)]=dict(mean_gpu_render_ms=costs,
                        split_over_full=costs['split']/costs['full'] if costs['full'] and costs['split'] is not None else None,
                        scope='batches2..16; actual GPU render interval includes traversal and lighting; not isolated kernels')
            report.update(room_quality=quality,room_timing=timing,
                room_quality_passed=all(x['status']=='passed' for values in quality.values() for x in values.values()),
                external_reference='not rerun or replaced; these are matched native order comparisons, not validation against PT')
        report.update(status='complete',functional_passed=bool(functional),all_cleanup_passed=all(x['passed'] for x in report['cleanup']))
        report['functional_passed'] &= report['all_cleanup_passed']
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    finally:
        report['elapsed_wall_s']=perf_counter()-started
        dump(output/'summary.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True);parser.add_argument('--shaders',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device-index',type=int,default=1)
    parser.add_argument('--no-validation',action='store_true');parser.add_argument('--cases',default=','.join(CASES))
    parser.add_argument('--baseline-library',type=Path);parser.add_argument('--baseline-shaders',type=Path)
    args=parser.parse_args();result=run(args.library,args.output,shaders=args.shaders,device_index=args.device_index,
        validation=not args.no_validation,cases=tuple(args.cases.split(',')),baseline_library=args.baseline_library,baseline_shaders=args.baseline_shaders)
    print(json.dumps(dict(status=result['status'],functional_passed=result['functional_passed'],room_quality_passed=result.get('room_quality_passed'),output=str(args.output))))
    return 0 if result['status']=='complete' and result['functional_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
