"""R4 real archived heterogeneous worlds through public native sessions.

Example (repository root):
 python rendering/examples/native_multiworld.py --source <cuda-01> --output <NEW_DIR> \
   --library <phymeo_native.dll> --shaders <SPV_DIR> --device-index 1 --validation

Default: resident, 1/2/8 worlds, two 320x240 cameras, 300 completed batches each.
--legacy explicitly selects the bounded original ABI2 path; there is no fallback.
Independent references are captured first, one world/session at a time. 'Batch'
means all requested cameras are CPU-consumable, not a claim of GPU concurrency.
"""
import argparse
import hashlib
import json
import math
import platform
import struct
import time
import traceback
import zlib
from dataclasses import asdict
from pathlib import Path

from phymeo_rendering.outputs import Capture, Channel, ImagePlane
from phymeo_rendering.pipeline import CaptureRequest, RenderPath

CHANNELS=(Channel.RGB,Channel.FORWARD_DEPTH,Channel.INSTANCE_ID,Channel.VALID)


def source_slot(batch, batches, records, *, reset_at):
    if (any(type(x) is not int for x in (batch,batches,records)) or
        batches<1 or records<1 or not 0<=batch<batches):
        raise ValueError('invalid source schedule')
    if reset_at is not None and (type(reset_at) is not int or not 0<reset_at<batches):
        raise ValueError('reset must be strictly inside the batch window')
    epoch=int(reset_at is not None and batch>=reset_at)
    start=reset_at if epoch else 0
    count=(batches-reset_at) if epoch else (reset_at or batches)
    ordinal=0 if count==1 else ((batch-start)*(records-1))//(count-1)
    return ordinal,epoch


def _identity(capture):
    return dict(frame_key=asdict(capture.frame_key),time_s=capture.time_s,
                camera=asdict(capture.camera),channels=[p.channel.value for p in capture.planes])


def _plain(value):
    return json.loads(json.dumps(value,allow_nan=False))


def _digest(raw): return hashlib.sha256(raw).hexdigest()


def compare_capture(reference, candidate):
    if _identity(reference)!=_identity(candidate):
        raise ValueError('capture identity/time/camera/channels mismatch')
    a={p.channel:p.data for p in reference.planes}; b={p.channel:p.data for p in candidate.planes}
    if set(a)!=set(CHANNELS) or set(b)!=set(CHANNELS):
        raise ValueError('R4 requires all four actual channels')
    ids_equal=a[Channel.INSTANCE_ID]==b[Channel.INSTANCE_ID]
    valid_equal=a[Channel.VALID]==b[Channel.VALID]
    hits=0; difference=0.
    for (ad,), (bd,), av, bv in zip(struct.iter_unpack('<f',a[Channel.FORWARD_DEPTH]),
            struct.iter_unpack('<f',b[Channel.FORWARD_DEPTH]),a[Channel.VALID],b[Channel.VALID],strict=True):
        if av and bv:
            hits+=1; difference=max(difference,abs(ad-bd))
    return dict(passed=ids_equal and valid_equal and hits>0 and difference<=.002,
                id_exact=ids_equal,valid_exact=valid_equal,compared_hit_pixels=hits,
                total_pixels=reference.camera.width*reference.camera.height,
                max_depth_difference_m=difference,rgb_exact=a[Channel.RGB]==b[Channel.RGB])


def save_reference(path, capture):
    raw=b''.join(p.data for p in capture.planes); compressed=zlib.compress(raw,1)
    Path(path).write_bytes(compressed)
    return dict(identity=_plain(_identity(capture)),lengths=[len(p.data) for p in capture.planes],
                compressed_sha256=_digest(compressed),raw_sha256=_digest(raw))


def compare_reference(path, metadata, candidate):
    if metadata['identity']!=_plain(_identity(candidate)):
        raise ValueError('saved reference identity mismatch')
    compressed=Path(path).read_bytes()
    if _digest(compressed)!=metadata['compressed_sha256']:
        raise ValueError('reference compressed digest mismatch')
    raw=zlib.decompress(compressed)
    if _digest(raw)!=metadata['raw_sha256'] or len(raw)!=sum(metadata['lengths']):
        raise ValueError('reference raw digest/length mismatch')
    offset=0; planes=[]
    for plane,length in zip(candidate.planes,metadata['lengths'],strict=True):
        planes.append(ImagePlane(plane.channel,raw[offset:offset+length])); offset+=length
    reference=Capture(candidate.frame_key,candidate.time_s,candidate.camera,tuple(planes))
    return compare_capture(reference,candidate)


def timing_report(seconds, *, worlds, phase_wall_seconds=None):
    if not seconds or any(not math.isfinite(x) or x<=0 for x in seconds):
        raise ValueError('timing requires positive completed batch durations')
    ordered=sorted(seconds)
    wall=sum(seconds) if phase_wall_seconds is None else phase_wall_seconds
    if not math.isfinite(wall) or wall<sum(seconds):
        raise ValueError('phase wall time cannot be shorter than its captures')
    return dict(completed_batches=len(seconds),worlds=worlds,cameras_per_world=2,
        p50_ms=1000*ordered[math.ceil(.5*len(ordered))-1],
        p95_ms=1000*ordered[math.ceil(.95*len(ordered))-1],
        camera_frames_per_second=len(seconds)*worlds*2/wall,
        capture_only_camera_frames_per_second=len(seconds)*worlds*2/sum(seconds),
        phase_wall_seconds=wall,
        throughput_scope=('sum of capture call windows' if phase_wall_seconds is None else
                          'post-warmup whole phase including comparison, journal and encoding'),
        capture_wall_seconds=sum(seconds),batch_seconds=seconds,
        timing_source='time.perf_counter around frame/camera preparation, supports and all synchronous public captures; CPU bytes returned',
        submission='serial public capture calls; one persistent session per world',
        peak_device_memory_bytes=None,peak_device_memory_reason='backend exposes resource counts, not peak allocated VRAM telemetry',
        gpu_wait_seconds=None,gpu_wait_reason='GPU wait is not measured separately; see transfer counters for host fence waits and optional Vulkan GPU execution timestamps',
        gpu_parallelism_measured=False)


def transfer_report(diagnostics):
    data=diagnostics['device']; captures=data.get('resident_captures',data.get('captures',0))
    counters=('resident_geometry_upload_bytes','resident_transform_upload_bytes','resident_parameter_upload_bytes')
    telemetry=data.get('resident_telemetry'); profile=data.get('resident_profile')
    known=sum(data[key] for key in counters) if all(key in data for key in counters) else None
    tlas=profile['tlas_upload_bytes'] if profile is not None else None
    texture=data.get('resident_texture_info')
    texture_expected=data.get('resident_texture_abi_version') is not None
    texture_image=texture['image_upload_bytes'] if texture is not None else (None if texture_expected else 0)
    texture_metadata=texture['metadata_upload_bytes'] if texture is not None else (None if texture_expected else 0)
    # These GPU buffer layouts are the resident ABI1 shader layouts. The native
    # byte counter remains authoritative; report any decomposition disagreement.
    sizes={'resident_vertex_count':('vertex_buffer_bytes',32),
           'resident_index_count':('index_buffer_bytes',4),
           'resident_mesh_count':('mesh_metadata_bytes',32),
           'resident_instance_count':('instance_material_bytes',48),
           'resident_bvh_nodes':('mesh_blas_bytes',48)}
    static={'counter_bytes':data.get('resident_geometry_upload_bytes')}
    if all(key in data for key in sizes) and 'resident_upload_calls' in data:
        for key,(name,size) in sizes.items():
            # Even an empty scene uploads one zero sentinel for each buffer.
            static[name]=max(1,data[key])*size*data['resident_upload_calls']
        static['sum_matches_counter']=sum(static[name] for name,_ in sizes.values())==static['counter_bytes']
    uniforms=data.get('resident_parameter_upload_bytes')
    camera_bytes=captures*272 if uniforms is not None else None
    setup_bytes=uniforms-camera_bytes if uniforms is not None else None
    components=dict(static_geometry=static,
        texture_image_upload_bytes=texture_image,texture_metadata_upload_bytes=texture_metadata,
        texture_scope='actual image+UV/slot metadata writes including default white sentinel; absent extension has no texture buffers',
        gpu_instance_forward_and_inverse_bytes=data.get('resident_transform_upload_bytes'),
        gpu_instance_transform_bytes_per_instance_per_capture=128,
        host_transform_input_bytes_not_added=data.get('resident_host_transform_input_bytes'),
        host_transform_explanation='64 B input matrix per instance is expanded to the counted 128 B forward+inverse GPU record; not an additional transfer',
        uniforms=dict(counter_bytes=uniforms,camera_ubo_bytes=camera_bytes,
            studio_setup_bytes_in_counter=setup_bytes,
            studio_setup_explanation='new native code uploads one 144 B studio UBO before first capture, including all-zero default; already included, never add twice'),
        tlas_upload_bytes=tlas,tlas_source='optional native profile counter' if profile else 'unavailable; not assumed zero')
    return dict(captures=captures,includes_warmup=True,
        session_allocation_telemetry=telemetry,capture_profile=profile,
        fence_wait_wall_ns=telemetry['fence_wait_wall_ns'] if telemetry else None,
        fence_wait_calls=telemetry['fence_wait_calls'] if telemetry else None,
        fence_wait_scope='native host fence wait wall time, upload+capture, not GPU kernel time',
        host_to_device_bytes=known+tlas+texture_image+texture_metadata if all(x is not None for x in (known,tlas,texture_image,texture_metadata)) else None,
        known_upload_subtotal_bytes=known,upload_components=components,
        host_to_device_source='native static copies plus mapped transform/UBO/TLAS writes; input bytes, not hardware PCIe traffic; unavailable if optional TLAS counter absent',
        device_to_host_bytes=telemetry['readback_bytes'] if telemetry else captures*320*240*13,
        device_to_host_source=('native completed readback memcpy bytes' if telemetry else
            'derived from successful four-channel R4 captures and packed RGBA/depth/ID/VALID ABI (13 B/pixel); native counter unavailable'),
        consumer_image_bytes=captures*320*240*12,
        synchronous_capture_completions=captures,
        wait_source='native synchronous fence completion per capture; separate profile excludes initial static upload',
        native_counters={k:v for k,v in data.items() if k.startswith('resident_')})

def _capture_pair(source,world,session,batch,batches):
    ordinal,epoch=source_slot(batch,batches,source.record_count,reset_at=batches//2 if world==0 else None)
    frame=source.frame(world,ordinal,epoch)
    result=[]
    for camera in source.cameras(world,frame):
        request=CaptureRequest(camera,RenderPath.RASTER,CHANNELS)
        if not session.supports(request): raise ValueError(f'unsupported R4 view {world}/{camera.id}')
        result.append(session.capture(frame,request))
    return ordinal,epoch,result


def _luid(diagnostics):
    return (diagnostics['device'].get('resident_telemetry') or {}).get('device_luid_hex')


def _overview(path, source, actual):
    from PIL import Image, ImageDraw
    columns=min(2,len(actual)); rows=math.ceil(len(actual)/columns)
    canvas=Image.new('RGB',(columns*640,rows*260),(18,20,23)); draw=ImageDraw.Draw(canvas)
    for world,(ordinal,epoch,captures) in enumerate(actual):
        x=(world%columns)*640; y=(world//columns)*260
        for camera,capture in enumerate(captures):
            rgb=next(p.data for p in capture.planes if p.channel is Channel.RGB)
            canvas.paste(Image.frombytes('RGB',(320,240),rgb),(x+320*camera,y+20))
        draw.text((x+3,y+3),f'world {world} {source.manifest["worlds"][world]["key"]} | source {ordinal} epoch {epoch} | front / reverse',fill='white')
    canvas.save(path)


def run(source_directory,output,backend,*,batches=300,warmup=3,scales=(1,2,8),memory_period_s=.25):
    from _r4_archive import ArchiveWorlds
    from _r4_resources import WindowsProcessMemory
    if type(batches) is not int or batches<2 or type(warmup) is not int or warmup<0:
        raise ValueError('at least two batches and nonnegative warmup required')
    if not scales or any(n not in (1,2,8) for n in scales) or len(set(scales))!=len(scales):
        raise ValueError('scales must be a unique subset of 1,2,8')
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='phymeo-rendering-native-r4-v1',status='running',requested_batches=batches,
        requested_scales=list(scales),warmup_batches_per_session=warmup,resident=backend.resident,
        resolution=[320,240],channels=[c.value for c in CHANNELS],
        runtime=dict(python=platform.python_version(),platform=platform.platform()),
        reference={},scales={},physics_steps=0,encoding='zlib level1 raw reference planes; no video encoding',
        measurement_gaps=[],
        scope='read-only archived heterogeneous geom scenes; replay reset, not physics reset')
    journal_path=output/'batches.jsonl'; sessions=[]; source=None; references={}
    journal_hash=hashlib.sha256(); completed_views=0; encoding_seconds=0.
    sampler=WindowsProcessMemory(memory_period_s); sampler.start()
    observed_luids=set(); run_started=time.perf_counter()
    def save():
        report['journal']=dict(file=journal_path.name,sha256=journal_hash.hexdigest(),completed_views=completed_views)
        (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    try:
        source=ArchiveWorlds(source_directory,output/'assets')
        report['source']=source.provenance()
        report['source_unique_records_per_world']={str(w):len({source_slot(i,batches,source.record_count,
            reset_at=batches//2 if w==0 else None)[0] for i in range(batches)}) for w in range(max(scales))}
        # Save exact source schedule before any capture, including repeated exposures.
        report['schedule']={str(w):[dict(source_index=s,epoch=e) for s,e in
            (source_slot(i,batches,source.record_count,reset_at=batches//2 if w==0 else None)
             for i in range(batches))] for w in range(max(scales))}
        with journal_path.open('xb') as journal:
            def emit(row):
                raw=(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode()
                journal.write(raw); journal.flush(); journal_hash.update(raw)
            # Independent sessions process one complete world timeline before the next.
            for world in range(max(scales)):
                report['active_phase']=dict(kind='independent',world=world)
                phase_started=time.perf_counter()
                session=backend.open_scene(source.scenes[world]); sessions.append(session)
                directory=output/'references'/str(world); directory.mkdir(parents=True)
                for _ in range(warmup): _capture_pair(source,world,session,0,batches)
                seconds=[]; measurement_started=time.perf_counter()
                for index in range(batches):
                    report['attempted_batch']=index
                    started=time.perf_counter()
                    ordinal,epoch,captures=_capture_pair(source,world,session,index,batches)
                    seconds.append(time.perf_counter()-started)
                    views=[]
                    for camera,capture in enumerate(captures):
                        path=directory/f'{index:04d}-{camera}.zlib'
                        start=time.perf_counter(); metadata=save_reference(path,capture)
                        encoding_seconds+=time.perf_counter()-start
                        references[world,index,camera]=(path,metadata)
                        views.append(dict(camera=capture.camera.id,reference=metadata,file=str(path.relative_to(output))))
                    completed_views+=2
                    emit(dict(phase='independent',world=world,batch=index,source_index=ordinal,epoch=epoch,
                              duration_s=seconds[-1],views=views))
                measurement_finished=time.perf_counter()
                session.close()
                diagnostics=session.diagnostics(); luid=_luid(diagnostics)
                if luid: observed_luids.add(luid)
                memory=sampler.report(luid,phase_started,time.perf_counter())
                timing=timing_report(seconds,worlds=1,phase_wall_seconds=measurement_finished-measurement_started); timing['peak_device_memory_bytes']=memory['peak_device_memory_bytes']
                timing['peak_device_memory_reason']='see process_memory sampling source/coverage'
                report['reference'][str(world)]=dict(timing=timing,process_memory=memory,
                    diagnostics=diagnostics,transfers=transfer_report(diagnostics))
                sessions.remove(session); save()
            for scale in scales:
                report['active_phase']=dict(kind='batch',worlds=scale)
                phase_started=time.perf_counter()
                owned=[]
                for w in range(scale):
                    session=backend.open_scene(source.scenes[w])
                    owned.append(session); sessions.append(session)
                for _ in range(warmup):
                    for w,session in enumerate(owned): _capture_pair(source,w,session,0,batches)
                seconds=[]; failures=0; maximum=0.; comparison_pixels=0
                measurement_started=time.perf_counter()
                for index in range(batches):
                    report['attempted_batch']=index
                    started=time.perf_counter()
                    actual=[_capture_pair(source,w,session,index,batches) for w,session in enumerate(owned)]
                    seconds.append(time.perf_counter()-started)
                    # Comparison and archival I/O are outside the completed-batch timer.
                    views=[]
                    for w,(ordinal,epoch,captures) in enumerate(actual):
                        for camera,capture in enumerate(captures):
                            path,metadata=references[w,index,camera]
                            comparison=compare_reference(path,metadata,capture)
                            failures+=int(not comparison['passed'])
                            maximum=max(maximum,comparison['max_depth_difference_m'])
                            comparison_pixels+=comparison['compared_hit_pixels']
                            if not comparison['passed'] or not comparison['rgb_exact']:
                                difference_dir=output/'differences'; difference_dir.mkdir(exist_ok=True)
                                difference_path=difference_dir/f'{scale}-{index:04d}-{w}-{camera}.zlib'
                                comparison['actual_capture']=dict(file=str(difference_path.relative_to(output)),
                                    metadata=save_reference(difference_path,capture))
                            views.append(dict(world=w,source_index=ordinal,epoch=epoch,
                                identity=_plain(_identity(capture)),comparison=comparison,
                                plane_sha256={p.channel.value:_digest(p.data) for p in capture.planes}))
                    completed_views+=2*scale
                    emit(dict(phase='batch',worlds=scale,batch=index,duration_s=seconds[-1],views=views))
                    report['progress']=dict(worlds=scale,completed_batches=index+1,failed_views=failures)
                    if index in {0,batches//2,batches-1}:
                        image_path=output/f'{scale}-worlds-batch-{index:04d}.png'
                        start=time.perf_counter(); _overview(image_path,source,actual)
                        encoding_seconds+=time.perf_counter()-start
                        report.setdefault('overview_images',[]).append(dict(file=image_path.name,sha256=_digest(image_path.read_bytes()),worlds=scale,batch=index))
                    if (index+1)%25==0: save()
                measurement_finished=time.perf_counter()
                final=[]
                for session in owned:
                    session.close(); final.append(session.diagnostics()); sessions.remove(session)
                luid=_luid(final[0])
                if luid: observed_luids.add(luid)
                memory=sampler.report(luid,phase_started,time.perf_counter())
                timing=timing_report(seconds,worlds=scale,phase_wall_seconds=measurement_finished-measurement_started); timing['peak_device_memory_bytes']=memory['peak_device_memory_bytes']
                timing['peak_device_memory_reason']='see process_memory sampling source/coverage'
                report['scales'][str(scale)]=dict(timing=timing,process_memory=memory,
                    failed_views=failures,max_depth_difference_m=maximum,compared_hit_pixels=comparison_pixels,
                    diagnostics=final,transfers=[transfer_report(d) for d in final],
                    reset_world=0,reset_at_batch=batches//2,comparison='exact independent per-world same-source schedule')
                save()
            source.assert_unchanged()
            report['status']='passed' if not any(v['failed_views'] for v in report['scales'].values()) else 'failed_comparison'
            report['full_schedule_completed']=batches>=300 and set(scales)=={1,2,8}
            report['r4_acceptance_complete']=False
            report['acceptance']='per-scale actual correctness/timing/memory evidence; final R4 signoff and high-quality single-world movie remain separate'
    except BaseException as error:
        report['status']='failed_prefix'; report['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        cleanup=[]; partial_diagnostics=[]
        for session in reversed(sessions):
            try: session.close()
            except BaseException as error: cleanup.append(repr(error))
            try: partial_diagnostics.append(session.diagnostics())
            except BaseException as error: cleanup.append('diagnostics: '+repr(error))
        if partial_diagnostics: report['partial_session_diagnostics']=partial_diagnostics
        if cleanup: report['cleanup_errors']=cleanup; report['status']='failed_cleanup'
        if source is not None:
            try: source.assert_unchanged()
            except BaseException as error: report['source_error']=repr(error); report['status']='failed_source'
        sampler.stop()
        memory_path=output/'gpu-process-memory.json'
        memory_path.write_text(json.dumps(sampler.samples,separators=(',',':'),allow_nan=False),encoding='utf-8')
        report['process_memory_samples']=dict(file=memory_path.name,sha256=_digest(memory_path.read_bytes()),
            sampling=sampler.report(next(iter(observed_luids)) if len(observed_luids)==1 else None,run_started,time.perf_counter()))
        for scale,value in report['scales'].items():
            if value['process_memory']['peak_device_memory_bytes'] is None:
                report['measurement_gaps'].append(f'{scale} worlds: no adapter-bound PID memory peak')
            if any(t['host_to_device_bytes'] is None for t in value['transfers']):
                report['measurement_gaps'].append(f'{scale} worlds: complete native input-byte counters unavailable')
            if any(t['fence_wait_wall_ns'] is None for t in value['transfers']):
                report['measurement_gaps'].append(f'{scale} worlds: optional native fence telemetry unavailable')
        report['encoding_seconds']=encoding_seconds
        report['physics_steps']=0 if source is None else source.physics_steps
        report.pop('active_phase',None) if report['status']=='passed' else None
        save()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--library',required=True,type=Path)
    parser.add_argument('--shaders',required=True,type=Path)
    parser.add_argument('--device-index',type=int,default=0)
    parser.add_argument('--validation',action='store_true')
    parser.add_argument('--legacy',action='store_true')
    parser.add_argument('--batches',type=int,default=300)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--scales',type=int,nargs='+',default=[1,2,8])
    parser.add_argument('--memory-period',type=float,default=.25)
    args=parser.parse_args()
    from phymeo_rendering.backends.native import NativeBackend
    backend=NativeBackend(library_path=args.library,shader_directory=args.shaders,
        device_index=args.device_index,validation=args.validation,resident=not args.legacy)
    report=run(args.source,args.output,backend,batches=args.batches,warmup=args.warmup,scales=tuple(args.scales),memory_period_s=args.memory_period)
    print(json.dumps(dict(status=report['status'],summary=str(args.output/'summary.json'),
                         progress=report.get('progress'),error=report.get('error')),indent=2))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__': raise SystemExit(main())
