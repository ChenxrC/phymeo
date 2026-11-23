"""Compose opaque traversal over a sealed C prepared package, without changing C.

All project/compile/device actions happen only through an explicit CLI call.
The original DLL/runtime and C policy are retained byte for byte in a NEW package.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from time import perf_counter
import traceback
from overlay import digest,write_json,patch_shadow,patch_closest,marker_map


def read_json(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('duplicate JSON key: '+key)
            result[key]=value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=unique)

def canonical_key(value):
    if not isinstance(value,str) or not value or ':' in value or '\0' in value:
        raise ValueError('invalid manifest relative path')
    key=value.replace('\\','/')
    if any(part in ('','.', '..') or part.endswith((' ','.')) for part in key.split('/')):
        raise ValueError('manifest path escapes or aliases prepared package')
    return key


def normalized_hashes(mapping):
    result={};seen=set()
    for raw,expected in mapping.items():
        key=canonical_key(raw)
        if key.casefold() in seen:raise ValueError('duplicate manifest path alias: '+raw)
        if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('invalid manifest SHA256: '+raw)
        seen.add(key.casefold());result[key]=expected
    return result


def checked_file(root,key,expected):
    path=(root/canonical_key(key)).resolve()
    if not path.is_relative_to(root) or path==root:raise ValueError('manifest path escapes prepared package')
    if not path.is_file() or digest(path)!=expected:raise ValueError('prepared generated source changed: '+key)
    return path


def validate_prepared(directory):
    root=Path(directory).resolve();manifest=read_json(root/'source-manifest.json')
    if manifest['contract']['schema']!='phymeo-preview-sparse-v1':raise ValueError('wrong C prepared schema')
    manifest['generated_sha256']=normalized_hashes(manifest['generated_sha256'])
    for relative,expected in manifest['generated_sha256'].items():checked_file(root,relative,expected)
    return root,manifest


def recovered_hashes(original,broken,patches):
    original=normalized_hashes(original)
    required={'source/_shaders/resident_rayquery.glsl','source/_shaders/resident_reflection.glsl'}
    if set(patches)!=required:raise ValueError('recovery requires exactly the two proven shader patches')
    desired=dict(original)
    for key,change in patches.items():
        if original.get(key)!=change['before']:raise ValueError('patch before SHA is not original')
        desired[key]=change['after']
    desired=normalized_hashes(desired)
    groups={}
    for raw,value in broken.items():groups.setdefault(canonical_key(raw),[]).append((raw,value))
    if set(groups)!=set(desired):raise ValueError('unexpected generated files in recovery')
    for key,entries in groups.items():
        if key in required:
            # This recovery is intentionally specific: exact old Windows key and
            # new POSIX key, with original-before and compiled-after authority.
            if dict(entries)!={key.replace('/','\\'):original[key],key:desired[key]}:
                raise ValueError('not the proven Windows two-shader manifest defect')
        elif len(entries)!=1 or entries[0][1]!=desired[key]:
            raise ValueError('unrelated duplicate or changed generated file')
    return desired

def variant_specs(policy):
    tile=policy['tile'];paths=policy['paths_per_anchor'];grid=policy['area_light_grid'];neighbors=policy['neighbor_candidates']
    revision=policy.get('reconstruction_revision',1)
    if any(type(x) is not int for x in (tile,paths,grid,neighbors,revision)) or tile not in (1,2,4) or paths not in (2,4,16) or grid not in (2,4,8) or neighbors not in (4,25) or revision not in (1,2):
        raise ValueError('unknown C sampling policy')
    common=[f'PR_PREVIEW_TILE={tile}',f'PR_PREVIEW_PATHS={paths}',f'PR_STUDIO_AREA_GRID={grid}',
        f'PR_PREVIEW_NEIGHBORS={neighbors}',f'PR_PREVIEW_RECONSTRUCTION_REVISION={revision}',
        'PR_RESIDENT_RAY_QUERY=1','PR_RESIDENT_INDIRECT=1','PR_RESIDENT_PROGRESSIVE=1']
    specs=[];suffix='' if grid==2 else f'_area{grid}'
    for normal in (False,True):
        prefix='resident_rayquery'+('_normal' if normal else '')
        flags=common+(['PR_RESIDENT_NORMAL_MAPS=1'] if normal else [])
        specs.append((prefix+suffix+'.frag.spv',flags+['PR_PREVIEW_SPARSE_GUIDE=1']))
        for reflection in (False,True):
            name=prefix+('_reflection' if reflection else '')+'_indirect_progressive'+suffix+'.frag.spv'
            specs.append((name,flags+['PR_PREVIEW_SPARSE_MAIN=1']+(['PR_RESIDENT_REFLECTION=1'] if reflection else [])))
    return specs


def compose(prepared,output,compiler,expected_manifest_sha256):
    original,manifest=validate_prepared(prepared);manifest_path=original/'source-manifest.json'
    original_sha=digest(manifest_path)
    if original_sha!=expected_manifest_sha256:raise ValueError('explicit original source manifest SHA differs')
    if 'opaque_overlay' in manifest:raise ValueError('input already carries an opaque overlay')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    record=dict(schema='phymeo-preview-sparse-opaque-v1',status='preparing',original_prepared=str(original),
        original_source_manifest_sha256=original_sha,original_policy=manifest['contract'],
        builder_sha256=digest(__file__),query_helper_sha256=digest(Path(__file__).with_name('overlay.py')),
        compiler=str(Path(compiler).resolve()),compiler_sha256=digest(compiler),patched_sources={},fragments=[],
        contract='opaque triangle acceleration only; preserve C sampling, reconstruction, output and private ABI; entire-query endpoint fallback',
        performance='unmeasured; no inherited opaque/full-rate speedup claim')
    write_json(output/'opaque-overlay.json',record)
    try:
        # Copy only already SHA-verified files in the original source manifest.
        for relative in manifest['generated_sha256']:
            target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original/relative,target)
        shutil.copyfile(manifest_path,output/'original-source-manifest.json')
        for filename,patch in (('resident_rayquery.glsl',patch_shadow),('resident_reflection.glsl',patch_closest)):
            relative='source/_shaders/'+filename;path=output/relative;before=digest(path)
            path.write_text(patch(path.read_text(encoding='utf-8')),encoding='utf-8',newline='\n')
            after=digest(path);record['patched_sources'][relative]=dict(before=before,after=after)
            manifest['generated_sha256'][relative]=after
        build=output/'build';build.mkdir()
        original_build=original/'build';library=original_build/'phymeo_native.dll'
        if not library.is_file():raise ValueError('original validated C DLL missing')
        shutil.copyfile(library,build/library.name)
        original_spv={}
        for path in sorted(original_build.glob('*.spv')):
            original_spv[path.name]=digest(path);shutil.copyfile(path,build/path.name)
        if not original_spv:raise ValueError('no original C SPIR-V')
        for name,flags in variant_specs(manifest['contract']):
            old=original_build/name
            if not old.is_file():raise ValueError('missing matching C fragment: '+name)
            original_markers=marker_map(old.read_bytes())
            if original_markers.get('83')!=dict(name=manifest['profile_marker'],value=[1]):raise ValueError('original C SPIR-V profile mismatch')
            if '74' in original_markers:raise ValueError('original C preview marker collision')
            cmd=[str(compiler),'-V','--target-env','vulkan1.2','--glsl-version','460']+['-D'+flag for flag in flags]
            cmd += [str(output/'source/_shaders/resident.frag'),'-o',str(build/name)]
            result=subprocess.run(cmd,text=True,capture_output=True)
            (output/(name+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
            if result.returncode:raise RuntimeError('C+opaque shader compile failed: '+name)
            markers=marker_map((build/name).read_bytes())
            if any(markers.get(k)!=v for k,v in original_markers.items()):raise ValueError('original C marker changed: '+name)
            if markers.get('74')!=dict(name='PR_PREVIEW_OPAQUE_ABI',value=[1]):raise ValueError('actual opaque marker absent')
            record['fragments'].append(dict(file=name,command=cmd,original_sha256=digest(old),sha256=digest(build/name),markers=markers))
        record.update(status='ready',dll_sha256=digest(build/'phymeo_native.dll'),original_spv_sha256=original_spv,
            spv_sha256={p.name:digest(p) for p in sorted(build.glob('*.spv'))})
        if record['dll_sha256']!=digest(library):raise ValueError('copied original DLL changed')
        patched={x['file'] for x in record['fragments']}
        if any(record['spv_sha256'][name]!=expected for name,expected in original_spv.items() if name not in patched):
            raise ValueError('unrelated C shader changed')
        manifest['opaque_overlay']=dict(schema=record['schema'],original_source_manifest_sha256=original_sha,
            actual_marker=dict(spec_id=74,name='PR_PREVIEW_OPAQUE_ABI',value=1),
            patched_sources=record['patched_sources'],builder_sha256=record['builder_sha256'],
            contract=record['contract'],private_transport_unchanged=True)
        manifest['state']='C source plus explicit opaque shader overlay; compiled, not device validated'
        write_json(output/'source-manifest.json',manifest)
        record['source_manifest_sha256']=digest(output/'source-manifest.json')
    except BaseException as error:record.update(status='failed',error=repr(error));raise
    finally:write_json(output/'opaque-overlay.json',record)
    return record


def verify_composed(directory):
    root,manifest=validate_prepared(directory)
    proof=json.loads((root/'opaque-overlay.json').read_text())
    if proof['status']!='ready' or proof['source_manifest_sha256']!=digest(root/'source-manifest.json'):
        raise ValueError('C+opaque preparation not complete or changed')
    if digest(root/'build/phymeo_native.dll')!=proof['dll_sha256']:raise ValueError('C+opaque DLL changed')
    for name,expected in proof['spv_sha256'].items():
        if digest(root/'build'/name)!=expected:raise ValueError('C+opaque SPIR-V changed: '+name)
    return root,manifest,proof


def recover(failed,original,output,expected_manifest_sha256):
    """Copy verified compiled bytes to a new package; NEVER compile or render."""
    original,base=validate_prepared(original);failed=Path(failed).resolve()
    original_sha=digest(original/'source-manifest.json')
    if original_sha!=expected_manifest_sha256:raise ValueError('explicit original manifest SHA differs')
    old=read_json(failed/'source-manifest.json');proof=read_json(failed/'opaque-overlay.json')
    if proof['status']!='ready' or proof['source_manifest_sha256']!=digest(failed/'source-manifest.json'):
        raise ValueError('failed package does not have a completed, hash-bound compile')
    if proof['original_source_manifest_sha256']!=original_sha or proof['original_policy']!=base['contract']:
        raise ValueError('compiled package belongs to a different C original')
    if old['contract']!=base['contract']:raise ValueError('C contract changed')
    desired=recovered_hashes(base['generated_sha256'],old['generated_sha256'],proof['patched_sources'])
    overlay=old['opaque_overlay']
    if overlay['patched_sources']!=proof['patched_sources'] or overlay['original_source_manifest_sha256']!=original_sha:
        raise ValueError('overlay source provenance differs')
    # All non-overlay manifest metadata must still be the original C producer's.
    for key in set(base)|set(old):
        if key not in ('generated_sha256','opaque_overlay','state') and base.get(key)!=old.get(key):
            raise ValueError('C manifest metadata changed: '+key)
    copies={key:checked_file(failed,key,value) for key,value in desired.items()}
    dll=checked_file(failed,'build/phymeo_native.dll',proof['dll_sha256'])
    checked_file(original,'build/phymeo_native.dll',proof['dll_sha256'])
    copies['build/phymeo_native.dll']=dll
    original_spv=normalized_hashes(proof['original_spv_sha256']);current_spv=normalized_hashes(proof['spv_sha256'])
    if any('/' in name or not name.endswith('.spv') for name in set(original_spv)|set(current_spv)):
        raise ValueError('invalid SPIR-V filename')
    if set(original_spv)!=set(current_spv):raise ValueError('SPIR-V inventory changed')
    expected_fragments={name for name,_ in variant_specs(base['contract'])}
    fragments={item['file']:item for item in proof['fragments']}
    if set(fragments)!=expected_fragments or len(proof['fragments'])!=6:raise ValueError('wrong compiled C fragment set')
    for name,expected in current_spv.items():
        previous=checked_file(original,'build/'+name,original_spv[name])
        current=checked_file(failed,'build/'+name,expected);copies['build/'+name]=current
        if name not in expected_fragments:
            if expected!=original_spv[name]:raise ValueError('unrelated C shader changed')
            continue
        item=fragments[name];old_markers=marker_map(previous.read_bytes());new_markers=marker_map(current.read_bytes())
        if item['original_sha256']!=original_spv[name] or item['sha256']!=expected or item['markers']!=new_markers:
            raise ValueError('compiled fragment receipt changed')
        if old_markers.get('83')!=dict(name=base['profile_marker'],value=[1]) or '74' in old_markers:
            raise ValueError('original C shader profile mismatch')
        if any(new_markers.get(k)!=v for k,v in old_markers.items()) or new_markers.get('74')!=dict(name='PR_PREVIEW_OPAQUE_ABI',value=[1]):
            raise ValueError('C or opaque marker changed')
    # Retain the exact evidence that established these bytes, including compiler logs.
    copies['original-source-manifest.json']=original/'source-manifest.json'
    copies['recovery-input-source-manifest.json']=failed/'source-manifest.json'
    copies['recovery-input-opaque-overlay.json']=failed/'opaque-overlay.json'
    for name in expected_fragments:
        log=failed/(name+'.log')
        if not log.is_file():raise ValueError('original compile log missing')
        copies[name+'.log']=log
    output=Path(output).resolve()
    if output==failed or output.is_relative_to(failed) or output==original or output.is_relative_to(original):
        raise ValueError('recovery output must be separate from both source packages')
    output.mkdir(parents=True,exist_ok=False)
    audit=dict(schema='phymeo-preview-sparse-opaque-recovery-v1',status='copying',
        failed_prepared=str(failed),original_prepared=str(original),original_source_manifest_sha256=original_sha,
        failed_manifest_sha256=digest(failed/'source-manifest.json'),failed_overlay_sha256=digest(failed/'opaque-overlay.json'),
        builder_sha256=digest(__file__),project_execution=False,compiled_bytes_changed=False,copied=[])
    try:
        for key,source in copies.items():
            target=output/key;target.parent.mkdir(parents=True,exist_ok=True);expected=digest(source)
            shutil.copyfile(source,target)
            if digest(target)!=expected:raise ValueError('recovery copy differs: '+key)
            audit['copied'].append(dict(file=key,source=str(source),sha256=expected,verified=True))
        old['generated_sha256']=desired
        write_json(output/'source-manifest.json',old)
        proof['source_manifest_sha256']=digest(output/'source-manifest.json')
        proof['recovery']=dict(failed_prepared=str(failed),failed_manifest_sha256=audit['failed_manifest_sha256'],
            failed_overlay_sha256=audit['failed_overlay_sha256'],reason='canonicalize proven Windows two-shader alias defect; compiled bytes unchanged')
        write_json(output/'opaque-overlay.json',proof)
        verify_composed(output)
        audit.update(status='ready',source_manifest_sha256=proof['source_manifest_sha256'])
    except BaseException as error:audit.update(status='failed',error=repr(error));raise
    finally:write_json(output/'recovery.json',audit)
    return audit

def compare_reports(reports):
    if len(reports)!=4 or any(x['status']!='complete' for x in reports):raise ValueError('four complete independent process reports required')
    if any(x['contract']!=reports[0]['contract'] for x in reports):raise ValueError('C estimator/sampling policy changed')
    fields=('key','input_sha256','geometry_sha256','rgb_sha256','all_rgb_sha256','sensors','linear_sha256','guide_sha256','actual')
    for name in ('material_binding','studio','trace_sha256'):
        if any(r.get(name)!=reports[0].get(name) for r in reports):raise ValueError('matched C scene changed: '+name)
    base=reports[0]['records'];differences=[]
    for round_index,report in enumerate(reports):
        if len(report['records'])!=len(base):raise ValueError('missing paired exposure')
        for expected,actual in zip(base,report['records']):
            changed=[key for key in fields if expected[key]!=actual[key]]
            if actual['rgb_sha256']!=actual['all_rgb_sha256']:changed.append('rgb-vs-all')
            if changed:differences.append(dict(round=round_index,key=actual['key'],fields=changed))
    costs=[dict(rgb_wall_ms=statistics.mean(x['rgb_wall_ms'] for x in r['records']),
        gpu_render_ms=statistics.mean(x['gpu_render_ms'] for x in r['records']) if all(x['gpu_render_ms'] is not None for x in r['records']) else None) for r in reports]
    return dict(equivalent=not differences and all(r['cleanup_passed'] for r in reports),differences=differences,costs_abba=costs,
        opaque_over_original_wall=[costs[1]['rgb_wall_ms']/costs[0]['rgb_wall_ms'],costs[2]['rgb_wall_ms']/costs[3]['rgb_wall_ms']],
        scope='new same-C-contract one-hop synchronous preview captures; no previous full-rate speedup imported')


def worker(prepared,output,c_examples,scene_examples,device_index=1):
    # A fresh process activates exactly one C runtime BEFORE any renderer import.
    root,manifest=validate_prepared(prepared)
    if 'opaque_overlay' in manifest:verify_composed(root)
    sys.path.insert(0,str(Path(c_examples).resolve()));sys.path.insert(0,str(Path(scene_examples).resolve()))
    from preview_sparse.runtime import activate
    activate(root)
    from preview_sparse.runtime import SparseBackend
    import panda_render_performance as bench
    from dataclasses import replace
    from types import SimpleNamespace
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest,RenderPath
    import phymeo_rendering
    from PIL import Image
    import zlib
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='phymeo-preview-sparse-opaque-capture-v1',status='running',prepared=str(root),
        source_manifest_sha256=digest(root/'source-manifest.json'),contract=manifest['contract'],
        implementation=dict(wrapper_sha256=digest(__file__),runtime_source=str(Path(phymeo_rendering.__file__).resolve())),
        registration=dict(poses=[59,60,149],seeds=[0,1],warmup_per_seed=2,size=[1280,720],
            requests='timed RGB followed by separate same-input all-channel capture; sparse exposures independent, no history',
            scope='same C policy; no full-image 16-spp claim; no inherited 30 FPS or prior speedup'),records=[],cleanup=[],cleanup_passed=False)
    write_json(output/'registration.json',report)
    def store(label,name,data,compressed=False):
        path=output/(label+'.'+name+('.zlib' if compressed else '.raw'))
        path.write_bytes(zlib.compress(data,6) if compressed else data)
        return dict(file=path.name,sha256=digest(path),raw_sha256=sha256(data).hexdigest(),raw_bytes=len(data),compression='zlib' if compressed else None)
    try:
        scenes,frames,studio,trace=bench.bind_scene(output/'geometry');scene=scenes[1]
        report['scene']=asdict(scene);report['studio']=asdict(studio);report['trace_sha256']=digest(trace)
        report['material_binding']=sha256(json.dumps(dict(materials=[asdict(x) for x in scene.materials],
            lights=[asdict(x) for x in scene.lights],instances=[asdict(x) for x in scene.instances]),sort_keys=True).encode()).hexdigest()
        items={}
        for index in (0,59,60,149):
            frame=replace(frames[index],key=replace(frames[index].key,scene_key=scene.key))
            camera=bench.film.exposure_camera(SimpleNamespace(camera='detail'),frame,index,len(frames),1280,720)
            items[index]=(frame,camera)
        for seed in (0,1):
            backend=SparseBackend(root,device_index=device_index,studio=studio,seed=seed,exposure=8.,
                display_transform='pbr-neutral',reflection_mode='single_bounce',validation=True)
            owner=backend.open_scene(scene)
            try:
                warm_frame,warm_camera=items[0]
                for _ in range(2):owner.capture(warm_frame,CaptureRequest(warm_camera,RenderPath.RASTER,(Channel.RGB,)))
                for index in (59,60,149):
                    frame,camera=items[index];label=f's{seed}-f{index}';request=CaptureRequest(camera,RenderPath.RASTER,(Channel.RGB,))
                    before=owner.diagnostics()['device'].get('resident_profile') or {}
                    start=perf_counter();shot=owner.capture(frame,request);rgb=shot.planes[0].data;elapsed=(perf_counter()-start)*1000
                    if shot.frame_key!=frame.key or shot.time_s!=frame.time_s or shot.camera!=camera:raise ValueError('stale C RGB output')
                    diagnostic=owner.diagnostics();after=diagnostic['device'].get('resident_profile') or {}
                    gpu=(after['gpu_render_ns']-before['gpu_render_ns'])/1e6 if after.get('timestamp_samples',0)>before.get('timestamp_samples',0) else None
                    expected_fragment='resident_rayquery_normal_reflection_indirect_progressive'+('' if manifest['contract']['area_light_grid']==2 else '_area'+str(manifest['contract']['area_light_grid']))+'.frag.spv'
                    if diagnostic['library_sha256']!=digest(root/'build/phymeo_native.dll') or diagnostic['shader_sha256']['frag']!=digest(root/'build'/expected_fragment):
                        raise ValueError('actual C DLL/fragment digest differs')
                    linear=owner.read_reconstructed_linear()['data'];guide=owner.read_incoming_guide()['data']
                    all_shot=owner.capture(frame,CaptureRequest(camera,RenderPath.RASTER,tuple(Channel)))
                    if all_shot.frame_key!=frame.key or all_shot.time_s!=frame.time_s or all_shot.camera!=camera:raise ValueError('stale C full output')
                    planes={p.channel.name:p.data for p in all_shot.planes}
                    all_linear=owner.read_reconstructed_linear()['data'];all_guide=owner.read_incoming_guide()['data']
                    if all_linear!=linear or all_guide!=guide:raise ValueError('C RGB/all reconstructed output differs')
                    files={name:store(label,name,data) for name,data in planes.items()}
                    files['linear']=store(label,'linear',linear,True);files['guide']=store(label,'guide',guide,True)
                    Image.frombytes('RGB',(1280,720),rgb).save(output/(label+'.png'))
                    report['records'].append(dict(key=label,input_sha256=sha256(json.dumps(dict(frame=asdict(frame),camera=asdict(camera)),sort_keys=True).encode()).hexdigest(),
                        frame=asdict(frame),camera=asdict(camera),geometry_sha256=diagnostic['decoded_geometry_sha256'],
                        rgb_sha256=sha256(rgb).hexdigest(),all_rgb_sha256=sha256(planes['RGB']).hexdigest(),
                        sensors={k:sha256(planes[k]).hexdigest() for k in ('FORWARD_DEPTH','INSTANCE_ID','VALID')},
                        linear_sha256=sha256(linear).hexdigest(),guide_sha256=sha256(guide).hexdigest(),actual=owner.diagnostics()['actual'],
                        rgb_wall_ms=elapsed,gpu_render_ms=gpu,diagnostics=diagnostic,files=files))
                    write_json(output/'summary.json',report)
            finally:
                owner.close();diag=owner.diagnostics();device=diag['device'];telemetry=device.get('resident_telemetry') or {}
                passed=(diag['closed'] and all(device.get(k)==0 for k in ('validation_errors','live_contexts','live_frames','live_resources'))
                    and telemetry.get('final_after_destroy') is True and telemetry.get('device_local_bytes_current')==0 and telemetry.get('host_visible_bytes_current')==0
                    and telemetry.get('allocation_count')==telemetry.get('free_count'))
                report['cleanup'].append(dict(seed=seed,passed=passed,diagnostics=diag))
        report.update(status='complete',cleanup_passed=all(x['passed'] for x in report['cleanup']))
    except BaseException as error:report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    finally:write_json(output/'summary.json',report)
    return report


def pair(original,opaque,output,c_examples,scene_examples,device_index=1):
    old,old_manifest=validate_prepared(original);new,new_manifest,proof=verify_composed(opaque)
    if proof['original_source_manifest_sha256']!=digest(old/'source-manifest.json') or old_manifest['contract']!=new_manifest['contract']:
        raise ValueError('opaque package is not derived from this exact C original')
    if digest(old/'build/phymeo_native.dll')!=proof['dll_sha256']:raise ValueError('C baseline DLL changed')
    for name,expected in proof['original_spv_sha256'].items():
        if digest(old/'build'/name)!=expected:raise ValueError('C baseline shader changed')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    receipt=dict(schema='phymeo-preview-sparse-opaque-pair-v1',status='running',order=['original','opaque','opaque','original'],runs=[])
    write_json(output/'registration.json',receipt);reports=[]
    try:
        for index,prepared in enumerate((old,new,new,old)):
            target=output/f'run-{index}';command=[sys.executable,'-B',str(Path(__file__).resolve()),'capture','--prepared',str(prepared),
                '--output',str(target),'--c-examples',str(c_examples),'--scene-examples',str(scene_examples),'--device-index',str(device_index)]
            result=subprocess.run(command,text=True,capture_output=True)
            (output/f'run-{index}.log').write_text(result.stdout+result.stderr,encoding='utf-8')
            receipt['runs'].append(dict(command=command,returncode=result.returncode,summary=str(target/'summary.json')))
            if result.returncode:raise RuntimeError('isolated C capture failed; prefix retained')
            reports.append(json.loads((target/'summary.json').read_text()))
        receipt.update(status='complete',comparison=compare_reports(reports))
    except BaseException as error:receipt.update(status='failed',error=repr(error))
    finally:write_json(output/'summary.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    build=sub.add_parser('prepare');build.add_argument('--prepared',type=Path,required=True);build.add_argument('--output',type=Path,required=True)
    build.add_argument('--compiler',type=Path,required=True);build.add_argument('--expected-manifest-sha256',required=True)
    r=sub.add_parser('recover');r.add_argument('--failed',type=Path,required=True);r.add_argument('--original',type=Path,required=True)
    r.add_argument('--output',type=Path,required=True);r.add_argument('--expected-manifest-sha256',required=True)
    for name in ('capture','pair'):
        p=sub.add_parser(name);p.add_argument('--output',type=Path,required=True);p.add_argument('--c-examples',type=Path,required=True)
        p.add_argument('--scene-examples',type=Path,required=True);p.add_argument('--device-index',type=int,default=1)
        if name=='capture':p.add_argument('--prepared',type=Path,required=True)
        else:p.add_argument('--original',type=Path,required=True);p.add_argument('--opaque',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='prepare':result=compose(a.prepared,a.output,a.compiler,a.expected_manifest_sha256);ok=result['status']=='ready'
    elif a.command=='recover':result=recover(a.failed,a.original,a.output,a.expected_manifest_sha256);ok=result['status']=='ready'
    elif a.command=='capture':result=worker(a.prepared,a.output,a.c_examples,a.scene_examples,a.device_index);ok=result['status']=='complete' and result['cleanup_passed']
    else:result=pair(a.original,a.opaque,a.output,a.c_examples,a.scene_examples,a.device_index);ok=result['status']=='complete' and result['comparison']['equivalent']
    print(json.dumps(dict(status=result['status'],passed=ok,output=str(a.output))))
    return 0 if ok else 1
if __name__=='__main__':raise SystemExit(main())
