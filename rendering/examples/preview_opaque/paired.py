"""Paired public-session timing/identity trial for a prepared opaque overlay.

No adaptive scene/lighting changes. A/B/B/A uses the same DLL, sample count,
materials, exposure, seeds and frames. A timing result never substitutes for
image/geometry equivalence. Invoke endpoint_probe.py before this trial.
"""
import argparse
from hashlib import sha256
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from overlay import digest,write_json


def run(library,overlay,output,device_index=1,secondary=None):
    import numpy as np
    from PIL import Image
    import phymeo_rendering
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest,RenderPath
    from secondary_light_budget import build_case,SENSORS
    from near_reflection_chart import closed
    overlay=Path(overlay).resolve();manifest=json.loads((overlay/'manifest.json').read_text())
    if manifest['status']!='ready':raise ValueError('overlay is not compiled/checked')
    baseline=Path(manifest['baseline']);optimized=overlay/'shaders'
    for directory,key in ((baseline,'baseline_files'),(optimized,'output_files')):
        for name,expected in manifest[key].items():
            if digest(directory/name)!=expected:raise ValueError('shader digest changed: '+name)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='phymeo-preview-opaque-paired-v1',status='running',library=str(Path(library).resolve()),
        library_sha256=digest(library),import_source=str(Path(phymeo_rendering.__file__).resolve()),
        overlay_manifest_sha256=digest(overlay/'manifest.json'),script_sha256=digest(__file__),
        registration=dict(order=['original','opaque','opaque','original'],seeds=[0,1],primary_grid=8,
            secondary_grid=8 if secondary is None else secondary,progressive_batches=4,primary_samples_per_batch=16,
            scenes=['direct','thin-indirect:-1','thin-indirect:1','room','mirror'],
            image_gate='all RGB, raw mean, direct-guide and three geometric planes byte equal',
            timing='steady batches2..4 GPU render timestamp; direct/mirror single captures reported separately',
            tie_scope='no new guarantee for equal-distance coplanar different-material hits',public_default_changed=False),
        records=[],comparisons=[],cleanup=[],performance=None,scenes={})
    write_json(output/'registration.json',report)
    saved={};timings={'original':[],'opaque':[]};owner=None
    try:
        for case,phase,depth in (('direct',0,0),('thin-indirect',-1,2),('thin-indirect',1,2),('room',0,2),('mirror',0,0)):
            scene,frame,camera,studio=build_case(output/'geometry'/f'{case}-{phase}',case,phase=phase)
            report['scenes'][f'{case}-{phase}']=dict(scene=asdict(scene),frame=asdict(frame),camera=asdict(camera),studio=asdict(studio))
            fragment='resident_rayquery'+('_reflection' if case=='mirror' else '')+('_indirect_progressive' if depth else '')+'_area8'+(f'_secondary{secondary}' if secondary is not None else '')+'.frag.spv'
            if fragment not in {x['file'] for x in manifest['compiled']}:
                raise ValueError('requested main fragment was not actually patched: '+fragment)
            for seed in (0,1):
                reference=None
                for order,variant in enumerate(('original','opaque','opaque','original')):
                    label=f'{case}-{phase}-s{seed}-{order}-{variant}'
                    options=dict(library_path=library,shader_directory=baseline if variant=='original' else optimized,
                        device_index=device_index,validation=True,resident=True,shadow_mode='ray_query',area_light_grid=8,
                        studio=studio,indirect_mode='diffuse_bounce' if depth else 'none',indirect_seed=seed,
                        progressive=bool(depth),indirect_bounces=depth or 1,spatial_filter='guided' if depth else 'none',
                        reflection_mode='single_bounce' if case=='mirror' else 'environment')
                    if secondary is not None:options['secondary_area_light_grid']=secondary
                    owner=NativeBackend(**options).open_scene(scene)
                    record=dict(label=label,variant=variant,case=case,phase=phase,seed=seed,order=order,timing=[])
                    report['records'].append(record);last_gpu=0;guide=None;raw=None
                    try:
                        for batch in range(1,(4 if depth else 1)+1):
                            started=perf_counter();capture=owner.capture(frame,CaptureRequest(camera,RenderPath.RASTER,tuple(Channel)))
                            wall=perf_counter()-started;planes={p.channel.name:p.data for p in capture.planes}
                            diag=owner.diagnostics()
                            actual_dir=baseline if variant=='original' else optimized
                            if diag['library_sha256']!=report['library_sha256'] or diag['shader_sha256']['frag']!=digest(actual_dir/fragment):
                                raise ValueError('actual public session DLL/fragment identity mismatch')
                            profile=diag['device'].get('resident_profile') or {}
                            current=profile.get('gpu_render_ns') if profile.get('timestamp_samples',0)>0 else None
                            gpu=(current-last_gpu)/1e6 if current is not None else None
                            record['timing'].append(dict(batch=batch,wall_ms=wall*1000,gpu_render_ms=gpu))
                            if current is not None:last_gpu=current
                            if depth and batch==1:guide=owner.read_spatial_guide()['data']
                            if depth and batch>1 and gpu is not None:timings[variant].append(gpu)
                        if depth:
                            mean=owner.read_linear_mean();raw=mean['data']
                            if mean['batches']!=4 or mean['space']!='scene-linear-before-exposure':raise ValueError('wrong linear readback')
                        payload=dict(planes=planes,raw=raw,guide=guide)
                        files={}
                        for name,data in {**planes,'linear':raw,'guide':guide}.items():
                            if data is None:continue
                            filename=label+'.'+name+'.raw';(output/filename).write_bytes(data)
                            files[name]=dict(file=filename,sha256=sha256(data).hexdigest(),bytes=len(data))
                        Image.frombytes('RGB',(camera.width,camera.height),planes['RGB']).save(output/(label+'.png'))
                        record.update(files=files,diagnostics=diag)
                        if reference is None:reference=payload
                        else:
                            delta=np.frombuffer(planes['RGB'],dtype=np.uint8).astype(int)-np.frombuffer(reference['planes']['RGB'],dtype=np.uint8).astype(int)
                            report['comparisons'].append(dict(label=label,rgb_equal=planes['RGB']==reference['planes']['RGB'],
                                sensor_equal=all(planes[n]==reference['planes'][n] for n in SENSORS),
                                raw_equal=raw==reference['raw'],guide_equal=guide==reference['guide'],
                                differing_rgb_codes=int(np.count_nonzero(delta)),max_rgb_error=int(np.max(np.abs(delta)))))
                    finally:
                        owner.close();diag=owner.diagnostics();owner=None
                        report['cleanup'].append(dict(label=label,passed=closed(diag),diagnostics=diag))
                    write_json(output/'summary.json',report)
        means={k:statistics.mean(v) if v else None for k,v in timings.items()}
        report.update(status='complete',equivalent=all(all(c[k] for k in ('rgb_equal','sensor_equal','raw_equal','guide_equal')) for c in report['comparisons']),
            cleanup_passed=all(c['passed'] for c in report['cleanup']),performance=dict(mean_gpu_render_ms=means,
                opaque_over_original=means['opaque']/means['original'] if means['original'] and means['opaque'] else None,
                sample_counts={k:len(v) for k,v in timings.items()},scope='measured steady render GPU interval; not isolated traversal'))
    except BaseException as error:report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    finally:write_json(output/'summary.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--overlay',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device-index',type=int,default=1);p.add_argument('--secondary',type=int,choices=(2,4))
    a=p.parse_args();r=run(a.library,a.overlay,a.output,a.device_index,a.secondary)
    print(json.dumps({k:r.get(k) for k in ('status','equivalent','cleanup_passed','performance')}))
    return 0 if r['status']=='complete' and r['equivalent'] and r['cleanup_passed'] else 1
if __name__=='__main__':raise SystemExit(main())
