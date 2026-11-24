"""R4 replay with the original cameras and four transport paths to the same CUDA consumer."""

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch

from _device_sensor_workload import PinnedUploader
from _panda_scene import pose_matrix
from _r4_archive import ArchiveWorlds
from _scale_camera_sensors import SceneSensorConsumer
from galaxea_r1.verify_device_station import authored_records
from native_multiworld import CHANNELS, source_slot
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.cameras import Camera
from phymeo_rendering.device_images import DeviceImageBatch, publish_device_images
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import Pose


ROUTES = ('cpu_single', 'shared_single_cpu', 'shared_batch_cpu', 'shared_device')
ORIGINAL = Path(__file__).resolve().parents[1] / 'results/r4/gpu-cameras.zip'


def archived_evidence(path):
    with ZipFile(path) as archive:
        original = json.loads(archive.read('summary.json'))
        cameras, hashes = {}, {}
        with archive.open('batches.jsonl') as journal:
            for line in journal:
                row = json.loads(line)
                if row['phase'] != 'batch' or row['worlds'] != 8:
                    continue
                for view in row['views']:
                    w, slot = view['world'], view['source_index']
                    camera = view['identity']['camera']
                    cameras.setdefault((w, camera['id']), camera)
                    key = w, slot, camera['id']
                    prior = hashes.setdefault(key, view['plane_sha256'])
                    if prior != view['plane_sha256']:
                        raise ValueError(f'inconsistent archived pixels for repeated source {key}')
    if original['status'] != 'passed' or len(cameras) != 16 or original['source']['record_count'] != 221:
        raise ValueError('expected the full archived R4 report, cameras and per-plane SHA256 journal')
    return original, cameras, hashes


def restore_ground_patches(source, original, directory):
    """R4 sized its finite support planes from trajectory bounds. Replay the recorded extents."""
    from _panda_scene import write_mesh

    adjusted = []
    for w, (old, rebuilt) in enumerate(zip(original['source']['assets'],source.assets,strict=True)):
        for before, after in zip(old,rebuilt,strict=True):
            if before['source_geom'] != after['source_geom'] or before['kind'] != after['kind']:
                raise ValueError('rebuilt primitive ordering or types differ from archived R4')
            if before['kind'] != 'plane' or before['mesh_sha256'] == after['mesh_sha256']:
                continue
            h = before['finite_plane_half_extent_m']
            folder = directory / f'world-{source.manifest["worlds"][w]["world"]}'
            name = f'geom-{after["source_geom"]}'
            write_mesh(folder,name,((-h,-h,0),(h,-h,0),(h,h,0),(-h,h,0)),
                       ((0,0,1),)*4,((0,1,2),(0,2,3)))
            digest = sha256((folder/(name+'.gltf')).read_bytes()).hexdigest()
            if digest != before['mesh_sha256']:
                raise ValueError('rebuilding the recorded finite plane failed its original glTF digest')
            adjusted.append(dict(world=w, original_half_extent_m=h, refitted_half_extent_m=after['finite_plane_half_extent_m']))
            after.update(mesh_sha256=digest, finite_plane_half_extent_m=h)
    return adjusted


@dataclass(frozen=True)
class ScheduledWorld:
    slot: int
    frame: object
    views: tuple


@dataclass(frozen=True)
class ArchiveBinding:
    world_key: int


@dataclass(frozen=True)
class ArchiveImageSource:
    """Metadata for the existing waitable image publisher, without a live physics workspace."""
    exposure: object
    binding: ArchiveBinding
    frame: tuple


def prepare(source, cameras, batches, count, device, producer):
    schedules, banks, total_bytes, camera_differences = [], [], 0, {}
    for w in range(count):
        slots = [source_slot(i, batches, source.record_count, reset_at=batches//2 if w == 0 else None)
                 for i in range(batches)]
        scheduled, states, cache = [], {}, {}
        for slot, epoch in slots:
            if (slot, epoch) not in cache:
                frame = source.frame(w, slot, epoch)
                views = []
                for fitted in source.cameras(w, frame):
                    camera = cameras[w, fitted.id]
                    p = camera['world_pose']
                    pose = Pose(tuple(p['position_m']), tuple(p['orientation_xyzw']))
                    lens = {k: camera[k] for k in ('width','height','fx_px','fy_px','cx_px','cy_px','near_m','far_m')}
                    old = Camera(fitted.id, frame.key, pose, **lens)
                    delta = np.abs(pose_matrix(fitted.world_pose) - pose_matrix(old.world_pose))
                    camera_differences[f'{w}/{old.id}'] = float(delta.max())
                    if any(getattr(fitted, k) != x for k, x in lens.items()):
                        raise ValueError('source camera intrinsics differ from R4')
                    views.append(CaptureRequest(old, RenderPath.RASTER, CHANNELS))
                cache[slot, epoch] = ScheduledWorld(slot, frame, tuple(views))
                if slot not in states:
                    states[slot] = authored_records(source.scenes[w], frame)
            scheduled.append(cache[slot,epoch])
        ordered = tuple(sorted(states))
        host = np.stack([states[k] for k in ordered])
        with torch.cuda.stream(producer):
            gpu = torch.from_numpy(host).to(device)
        total_bytes += host.nbytes
        banks.append({k: gpu[i] for i,k in enumerate(ordered)})
        schedules.append(scheduled)
    producer.synchronize()
    return tuple(zip(*schedules, strict=True)), banks, total_bytes, camera_differences


def open_sessions(stack, args, source, count, route):
    opts = dict(resident=True, device_index=args.device_index, validation=args.validation,
                shadow_mode='software', msaa_samples=1, display_transform='clip', exposure=1.,
                library_path=args.library, shader_directory=args.shaders or args.library.parent)
    if route != 'cpu_single':
        opts['shared_cuda_device'] = args.device
    backend = NativeBackend(**opts)
    sessions = []
    for scene in source.scenes[:count]:
        session = backend.open_scene(scene)
        sessions.append(session)
        stack.callback(session.close)
    return sessions


def capture(route, sessions, batch, banks, producer, delivery):
    results = []
    for w, (session, step) in enumerate(zip(sessions, batch, strict=True)):
        if route == 'cpu_single':
            results.extend(session.capture(step.frame, req) for req in step.views)
            continue
        delegate = session._delegate
        exchange = delegate.device.open_shared_matrices(cuda_device=producer.device.index)
        with exchange.publish(banks[w][step.slot], producer_stream=producer) as lease:
            slots = (None, None)  # Both registered R4 lenses are fixed in world coordinates.
            if route == 'shared_device':
                planes = delegate.capture_shared_device_batch(step.views, slots, lease, delivery)
                active = ArchiveImageSource(step.frame, ArchiveBinding(w), batch)
                results.append(publish_device_images(active, step.views, planes, session._instance_ids, delivery))
            else:
                if route == 'shared_single_cpu':
                    shots = tuple(delegate.capture(step.frame, req, shared_matrices=lease) for req in step.views)
                else:
                    shots = delegate.capture_shared_batch(step.frame, step.views, shared_matrices=lease,
                                                          shared_camera_indices=slots)
                results.extend(session._validate_publication(shot, step.frame.key, step.frame.time_s, req)
                               for shot, req in zip(shots, step.views, strict=True))
    return DeviceImageBatch.combine(results) if route == 'shared_device' else tuple(results)


def counters(sessions):
    totals = Counter()
    for s in sessions:
        d = s.diagnostics()['device']
        for k in ('resident_captures', 'resident_host_transform_input_bytes', 'resident_transform_upload_bytes'):
            totals[k] += d[k]
        for k in ('submissions', 'fence_wait_calls', 'fence_wait_wall_ns', 'readback_bytes'):
            totals[k] += d['resident_telemetry'][k]
        for k in ('matrix_d2d_bytes', 'matrix_d2d_copies', 'publications'):
            totals[k] += (d.get('resident_shared_matrices') or {}).get(k, 0)
        for k in ('attachment_d2d_bytes', 'attachment_d2d_copies', 'return_event_waits', 'image_d2h_bytes'):
            totals[k] += (d.get('resident_device_output') or {}).get(k, 0)
    return totals


def closed(sessions):
    ds = [s.diagnostics()['device'] for s in sessions]
    for d in ds:
        if (any(d[k] for k in ('live_resources', 'live_frames', 'live_contexts', 'validation_errors'))
                or (d.get('resident_device_output') or {}).get('imports_live', 0)
                or (d.get('resident_shared_matrices') or {}).get('imported_cuda_resources', 0)):
            raise ValueError('native resources, imports, or Vulkan errors remained after scale run')
    return dict(device_uuid=ds[0]['device_uuid'], name=ds[0]['device_name'],
                validation_warnings=sum(d['validation_warnings'] for d in ds),
                final_live_handles=0, validation_errors=0,
                peak_context_device_local_bytes=[d['resident_telemetry']['device_local_bytes_peak'] for d in ds])


def index_phase(args, source, schedule, count, route, banks, streams, sensor):
    producer, delivery, consumer = streams
    result_checks, elapsed = [], []
    uploader = PinnedUploader(consumer.device)
    with ExitStack() as stack:
        sessions = open_sessions(stack,args,source,count,route)
        for w, s in enumerate(sessions):
            if not all(s.supports(r) for r in schedule[0][w].views):
                raise ValueError(f'unsupported R4 raster camera in world {w}')
        def run(batch):
            raw = capture(route, sessions, batch, banks, producer, delivery)
            with torch.cuda.stream(consumer):
                inputs = ([c.planes for c in raw.wait(consumer)] if route == 'shared_device' else uploader.upload(raw))
                features = sensor(inputs)
                done.record(consumer)
            done.synchronize()
            return features

        done = torch.cuda.Event()
        for _ in range(args.warmup):
            run(schedule[0][:count])
        initial = counters(sessions)
        h2d_start = uploader.h2d_bytes
        for frame in schedule:
            started = perf_counter()
            features = run(frame[:count])
            elapsed.append((perf_counter()-started)*1000.)
            # Only reduced sensor statistics cross to host for verification, outside timing.
            stats = features.cpu().numpy()
            if np.any(stats[:,-1]) or np.any(stats[:,3] <= 0):
                raise ValueError(f'{route}: bad scene IDs/depth/visibility or no common foreground')
            result_checks.append(sha256(stats.tobytes()).hexdigest())
        final = counters(sessions)
    measured = {k:final[k]-initial[k] for k in final}
    expected = len(schedule)*count*2
    if measured['resident_captures'] != expected:
        raise ValueError(f'{route}: expected {expected} actual camera draws')
    rgba_bytes = expected*320*240*13
    is_device = route == 'shared_device'
    if (measured['readback_bytes'] != (0 if is_device else rgba_bytes)
            or measured['attachment_d2d_bytes'] != (rgba_bytes if is_device else 0)
            or uploader.h2d_bytes-h2d_start != (0 if is_device else expected*320*240*12)):
        raise ValueError(f'{route}: missing or unexpected full image transfers')
    report = dict(actual_camera_draws=expected, batch_ms=elapsed,
                  p50_ms=float(np.percentile(elapsed,50,method='inverted_cdf')),
                  p95_ms=float(np.percentile(elapsed,95,method='inverted_cdf')),
                  camera_frames_per_second=1000.*expected/sum(elapsed), native=measured,
                  public_cpu_to_gpu_image_bytes=uploader.h2d_bytes-h2d_start,
                  device_validation_d2h_bytes=len(schedule)*count if is_device else 0,
                  post_timer_feature_d2h_bytes=len(schedule)*count*2*8*8, cleanup=closed(sessions))
    return report, result_checks


def plane_hashes(shot):
    return {p.channel.value: sha256(p.data).hexdigest() for p in shot.planes}


def resource_probe(args, source, schedule, count, route, banks, streams, sensor):
    producer, delivery, consumer = streams
    selected = (0, len(schedule)//2, len(schedule)-1)
    uploader = PinnedUploader(consumer.device)
    begin, done = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    profile_fields = ('captures', 'timestamp_samples', 'gpu_render_ns', 'gpu_copy_ns', 'gpu_total_ns')
    memory_fields = ('device_local_bytes_peak','host_visible_bytes_peak',
                     'device_local_bytes_current','host_visible_bytes_current')
    with ExitStack() as stack:
        opened = perf_counter()
        sessions = open_sessions(stack,args,source,count,route)
        open_ms = (perf_counter()-opened)*1000.

        def measure(index):
            start = perf_counter()
            raw = capture(route,sessions,schedule[index][:count],banks,producer,delivery)
            with torch.cuda.stream(consumer):
                planes = ([c.planes for c in raw.wait(consumer)] if route == 'shared_device' else uploader.upload(raw))
                begin.record(consumer)
                result = sensor(planes)
                done.record(consumer)
            done.synchronize()
            wall = (perf_counter()-start)*1000.
            return result, dict(batch_index=index, capture_through_consumer_wall_ms=wall,
                                cuda_consumer_event_ms=begin.elapsed_time(done))

        def profile():
            devices = [s.diagnostics()['device'] for s in sessions]
            profiles = [d['resident_profile'] for d in devices]
            if any(p is None for p in profiles):
                raise ValueError('resident GPU profiling export missing during R4 resource probe')
            return devices, {k:sum(p[k] for p in profiles) for k in profile_fields}

        for _ in range(3):
            measure(0)
        initial = counters(sessions)
        _, last_profile = profile()
        first_profile = dict(last_profile)
        start_h2d = uploader.h2d_bytes
        samples, feature_bytes = [], 0
        for index in selected:
            features, sample = measure(index)
            cpu = features.cpu().numpy()
            if np.any(cpu[:,-1]) or np.any(cpu[:,3] <= 0):
                raise ValueError(f'{route}: sensor validation failed at R4 resource sample {index}')
            feature_bytes += cpu.nbytes
            devices, current = profile()
            sample['vulkan_profile_delta'] = {k:current[k]-last_profile[k] for k in profile_fields}
            sample['consumer_feature_sha256'] = sha256(cpu.tobytes()).hexdigest()
            last_profile = current
            samples.append(sample)
        final = counters(sessions)
        transfers = {k:final[k]-initial[k] for k in initial}
        profiles = {k:last_profile[k]-first_profile[k] for k in profile_fields}
        if profiles['captures'] != 3*count*2 or transfers['resident_captures'] != profiles['captures']:
            raise ValueError('resource probe did not render both real views of every sampled world')
        support = [bool(d['resident_profile']['timestamp_supported']) for d in devices]
        if all(support) and not profiles['timestamp_samples']:
            raise ValueError('Vulkan timestamp support reported without completed samples')
        memory = [{k:d['resident_telemetry'][k] for k in memory_fields} for d in devices]
    cpu_image = uploader.h2d_bytes-start_h2d
    draws = 3*count*2
    if (transfers['readback_bytes'] != (0 if route=='shared_device' else draws*320*240*13)
            or cpu_image != (0 if route=='shared_device' else draws*320*240*12)
            or transfers['attachment_d2d_bytes'] != (draws*320*240*13 if route=='shared_device' else 0)):
        raise ValueError('resource probe image transfer counts do not cover the selected exposures')
    return dict(worlds=count, route=route, warmup_batches=3, sampled_batches=3, samples=samples,
                open_all_sessions_wall_ms=open_ms, explicit_vulkan_memory_per_world=memory,
                sum_per_context_memory_bytes={k:sum(m[k] for m in memory) for k in memory_fields},
                vulkan_timestamp_supported_per_world=support, vulkan_profile_total=profiles,
                native_counter_delta=transfers, host_public_image_h2d_bytes=cpu_image,
                device_sensor_status_d2h_bytes=(3*count if route=='shared_device' else 0),
                after_timer_consumer_feature_d2h_bytes=feature_bytes, cleanup=closed(sessions))


def independent(args, source, schedule, count):
    refs = {}
    for w in range(count):
        session = NativeBackend(resident=True, shadow_mode='software', exposure=1., msaa_samples=1,
                                device_index=args.device_index, validation=args.validation, library_path=args.library,
                                shader_directory=args.shaders or args.library.parent).open_scene(source.scenes[w])
        try:
            for i,batch in enumerate(schedule):
                for req in batch[w].views:
                    shot = session.capture(batch[w].frame, req)
                    refs[w,i,req.camera.id] = plane_hashes(shot)
        finally:
            session.close()
        closed([session])
    return refs


def compare_images(ref, shot):
    if (shot.camera, shot.frame_key, shot.time_s) != (ref.camera, ref.frame_key, ref.time_s):
        raise ValueError('view metadata differs from the registered archived exposure')
    if tuple(p.channel for p in shot.planes) != CHANNELS:
        raise ValueError('R4 image channel order or count differs')
    a, b = ({p.channel: p.data for p in c.planes} for c in (ref, shot))
    eq = {ch.value: a[ch] == b[ch] for ch in CHANNELS}
    av = np.frombuffer(a[Channel.VALID], np.uint8)
    bv = np.frombuffer(b[Channel.VALID], np.uint8)
    hits = (av != 0) & (bv != 0)
    n = int(hits.sum())
    if not n:
        raise ValueError('R4 view contains no common valid depth samples')
    depth = Channel.FORWARD_DEPTH
    dz = (0. if eq[depth.value] else
          float(np.abs(np.frombuffer(a[depth], '<f4')[hits] - np.frombuffer(b[depth], '<f4')[hits]).max()))
    delta = None
    changed = maximum = 0
    if not eq[Channel.RGB.value]:
        ar, br = (np.frombuffer(buff[Channel.RGB], np.uint8).reshape(ref.camera.height, ref.camera.width, 3)
                  for buff in (a,b))
        delta = np.abs(ar.astype(np.int16)-br).astype(np.uint8)
        changed, maximum = int(np.any(delta, axis=2).sum()), int(delta.max())
    return dict(exact=eq, common_hits=n, depth_max_error_m=dz,
                different_rgb_pixels=changed, maximum_rgb_delta=maximum), delta


def quality(args, source, schedule, scales, banks, streams, history, report):
    from PIL import Image

    start = perf_counter()
    refs = independent(args, source, schedule, max(scales))
    report['independent_reference_camera_draws'] = len(refs)
    report['independent_reference_seconds'] = perf_counter()-start
    journal = args.output/'quality.jsonl'
    results = {}
    with journal.open('x', encoding='utf-8') as out:
        for count in scales:
            stats = {r: Counter() for r in ROUTES}
            sensor = SceneSensorConsumer(source.scenes, tuple(range(count)), streams[2].device)
            timing = perf_counter()
            with ExitStack() as stack:
                sessions = {r: open_sessions(stack,args,source,count,r) for r in ROUTES}
                for i, group in enumerate(schedule):
                    batch = group[:count]
                    actual = {r: capture(r, sessions[r], batch, banks, *streams[:2]) for r in ROUTES}
                    with torch.cuda.stream(streams[2]):
                        features = sensor([shot.planes for shot in actual['shared_device'].wait(streams[2])]).cpu().numpy()
                    oracle = sensor.numpy_oracle(actual['cpu_single'])
                    stats['shared_device']['oracle_mismatched_views'] += int(np.any(features != oracle, axis=1).sum())
                    # Explicit raw audit D2H only in this separate phase.
                    actual['shared_device'] = actual['shared_device'].readback_async(streams[2]).result()
                    for j, base in enumerate(actual['cpu_single']):
                        w, ordinal = j//2, batch[j//2].slot
                        fingerprint = plane_hashes(base)
                        history_hashes = history.get((w,ordinal,base.camera.id))
                        if fingerprint != refs[w,i,base.camera.id]:
                            raise ValueError(f'host scale {count}, batch {i}, world {w} differs from independent draw')
                        row = dict(scale=count,batch=i,world=w,source_index=ordinal,epoch=base.frame_key.epoch,
                                   tick=base.frame_key.tick,view=base.camera.id,host_sha256=fingerprint,
                                   archived_matches=({k: fingerprint[k]==v for k,v in history_hashes.items()}
                                                     if history_hashes is not None else None),routes={})
                        for route in ROUTES:
                            s = stats[route]
                            shot = actual[route][j]
                            diff, image = compare_images(base, shot)
                            row['routes'][route] = diff
                            s['views'] += 1
                            s['rgb_pixels'] += base.camera.width*base.camera.height
                            s['common_hit_pixels'] += diff['common_hits']
                            s['rgb_changed_pixels'] += diff['different_rgb_pixels']
                            s['max_rgb_delta'] = max(s['max_rgb_delta'],diff['maximum_rgb_delta'])
                            s['max_depth_error_m'] = max(s['max_depth_error_m'],diff['depth_max_error_m'])
                            for channel, equal in diff['exact'].items():
                                s[channel+'_mismatched_views'] += int(not equal)
                            if image is not None and s[Channel.RGB.value+'_mismatched_views'] <= 2:
                                name = f'difference-{count}-{i}-{w}-{base.camera.id}-{route}.png'
                                Image.fromarray(np.minimum(255, image.astype(np.uint16)*8).astype(np.uint8)).save(
                                    args.output/name)
                                diff['difference_x8'] = name
                        hoststats = stats['cpu_single']
                        hoststats['historical_comparable_views'] += int(history_hashes is not None)
                        if history_hashes is not None:
                            hoststats['historical_all_four_exact_views'] += int(all(row['archived_matches'].values()))
                            for key, good in row['archived_matches'].items():
                                hoststats['historical_'+key+'_mismatched_views'] += int(not good)
                        # The two delivery modes must expose identical bytes for one shared batch ABI.
                        twin = actual['shared_batch_cpu'][j]
                        direct = actual['shared_device'][j]
                        twins_equal = twin.planes == direct.planes
                        row['shared_batch_delivery_bytes_equal'] = twins_equal
                        stats['shared_device']['delivery_mismatched_views'] += int(not twins_equal)
                        out.write(json.dumps(row, separators=(',',':'), allow_nan=False)+'\n')
            results[str(count)] = dict(
                routes=stats, cleanup={r:closed(sessions[r]) for r in ROUTES},
                explicit_image_audit_d2h_bytes=len(schedule)*count*2*320*240*12,
                wall_seconds=perf_counter()-timing)
            print(f'quality complete: {count} world(s), {len(schedule)} batches', flush=True)
    report['quality'] = results
    report['quality_journal'] = dict(file=journal.name,sha256=sha256(journal.read_bytes()).hexdigest())


def run(args):
    if args.batches < 2 or args.warmup < 1 or not args.scales or len(set(args.scales)) != len(args.scales):
        raise ValueError('use at least two batches, one warmup and a distinct nonempty subset of scales')
    if not set(args.scales) <= {1,2,8}:
        raise ValueError('R4 scales are 1, 2, 8')
    if args.phase == 'resources' and (args.batches != 300 or args.warmup != 3):
        raise ValueError('resources selects 0, 150, 299 of the original R4 timeline after three warmup batches')
    original, cameras, history = archived_evidence(args.archive_zip)
    path = args.source or Path(original['source']['source_directory'])
    if not (path/'manifest.json').is_file() or not (path/'states.jsonl.gz').is_file():
        raise FileNotFoundError(f'R4 model/state bundle is missing at {path}; supply an original or regenerated --source')
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('use an empty output directory')
    source = ArchiveWorlds(path, args.output/'meshes')
    original_source = original['source']
    restored = restore_ground_patches(source,original,args.output/'meshes')
    source_info = source.provenance()
    same = {k: original_source[k] == source_info[k] for k in ('manifest_sha256','state_sha256','record_count')}
    models_match = ([r['source_sha256'] for r in source_info['worlds']] ==
                    [r['source_sha256'] for r in original_source['worlds']])
    same_meshes = ([[r['mesh_sha256'] for r in world] for world in source.assets] ==
                   [[r['mesh_sha256'] for r in world] for world in original_source['assets']])
    full_source = all(same.values()) and models_match and same_meshes
    report = dict(source_class='original R4 archive' if full_source else 'regenerated multiworld trajectory; compare original R4 visual digests separately',
                  same_as_old_source=same, all_source_models_byte_exact=models_match, all_tessellated_meshes_byte_exact=same_meshes,
                  restored_finite_ground_patches=restored,
                  original_source_directory=original_source['source_directory'], original_state_sha256=original_source['state_sha256'],
                  original_manifest_sha256=original_source['manifest_sha256'], source=source_info,
                  historical_archive_sha256=sha256(args.archive_zip.read_bytes()).hexdigest(),
                  historical_available='per-plane checksums and exposure identities; zip omits raw images, model XMLs and states',
                  runtime=dict(pixel_size=[320,240],channels=[c.value for c in CHANNELS],shadows='software',
                               samples=1,display='clip',exposure=1.,geometry='ArchiveWorlds original R4 tessellation and materials'),
                  physics_steps=source.physics_steps, reset_world=0, reset_batch=args.batches//2,preloaded_images=0,
                  consumer_fields=SceneSensorConsumer.fields, measurement='host wall until identical CUDA sensor reductions finish on the consumer stream',
                  warmup_batches=args.warmup, timed_batches=args.batches, scales=args.scales,
                  full_schedule=args.batches==300 and args.warmup==3 and set(args.scales)=={1,2,8})
    with torch.cuda.device(args.device):
        device = torch.device('cuda',args.device)
        streams = tuple(torch.cuda.Stream(device=device) for _ in range(3))
        start = perf_counter()
        schedule,banks,upload,differences = prepare(source,cameras,args.batches,max(args.scales),device,streams[0])
        report.update(registered_vs_refitted_camera_matrix_max_error=differences, preparations_seconds=perf_counter()-start,
                      archived_visual_records_preload_h2d_bytes=upload,
                      preload_scope='one Torch CUDA world/inverse bank per world for unique source records, outside timed windows; native pose-buffer D2D transfers remain timed')
        report['world_schedule'] = {str(w):[(batch[w].slot,batch[w].frame.key.epoch,batch[w].frame.key.tick)
                                             for batch in schedule] for w in range(max(args.scales))}
        def save():
            (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        if args.phase == 'resources':
            report.update(timed_batches=0, full_schedule=False, resource_batches=3,
                          resource_sample_indices=[0,150,299], schedule_length=len(schedule),
                          resource_scope='Three representative batches per route/scale, following three warmups; individual samples only.',
                          memory_scope='Sum of per-context explicit Vulkan allocation peaks across setup, warmup and samples. '
                                       'Context peaks may occur at different instants. CUDA allocations and driver overhead are separate. '
                                       'Device-local and host-visible heaps may overlap.',
                          timestamp_scope='Vulkan timestamps sum raster and attachment-copy work over rendered views. '
                                          'CUDA event interval encloses consumer reductions after input upload or output readiness.',
                          resources={})
            for scale in args.scales:
                sensor = SceneSensorConsumer(source.scenes,tuple(range(scale)),device)
                report['resources'][str(scale)] = {}
                reference = None
                for route in ROUTES:
                    row = resource_probe(args,source,schedule,scale,route,banks,streams,sensor)
                    fingerprint = [s['consumer_feature_sha256'] for s in row['samples']]
                    if reference is not None and fingerprint != reference:
                        raise ValueError(f'{route}: representative consumer outputs differ at scale {scale}')
                    reference = fingerprint
                    report['resources'][str(scale)][route] = row
                    save()
                print(f'resource probes complete: {scale} world(s), four routes, three samples per route',flush=True)
        save()
        if args.phase in ('quality','both'):
            quality(args,source,schedule,args.scales,banks,streams,history,report)
            save()
        if args.phase in ('timing','both'):
            orders = ('forward','reverse') if args.order=='both' else (args.order,)
            report['timing'] = {}
            checks = {}
            for order in orders:
                report['timing'][order] = {}
                for scale in args.scales if order=='forward' else reversed(args.scales):
                    sensor = SceneSensorConsumer(source.scenes, tuple(range(scale)), device)
                    result = {}
                    for route in ROUTES if order=='forward' else reversed(ROUTES):
                        details, digests = index_phase(args,source,schedule,scale,route,banks,streams,sensor)
                        ref = checks.setdefault(scale, digests)
                        details['feature_mismatched_batches'] = sum(a!=b for a,b in zip(ref,digests,strict=True))
                        details['per_batch_feature_sha256'] = digests
                        result[route] = details
                        report['timing'][order][str(scale)] = result
                        save()
                        print(f'{order} {scale} {route}: p50 {details["p50_ms"]:.2f}, p95 {details["p95_ms"]:.2f} ms',flush=True)
    source.assert_unchanged()
    report['completed'] = True
    report['cross_path_features_exact'] = all(not r['feature_mismatched_batches']
        for order in report.get('timing',{}).values() for scale in order.values() for r in scale.values()) if 'timing' in report else None
    report['cross_path_all_planes_exact'] = all(not v
        for scale in report.get('quality',{}).values() for stats in scale['routes'].values() for k,v in stats.items()
        if k.endswith('_mismatched_views') and not k.startswith('historical_')) if 'quality' in report else None
    save()
    if report['cross_path_features_exact'] is False or report['cross_path_all_planes_exact'] is False:
        raise RuntimeError('R4 cross-path difference recorded in report.json')


def compare_reports(paths, output):
    """Join independently launched validation/forward/reverse runs on exact source and workload."""
    modes, digests, quality_rows = {}, {}, {}
    common, origin, schedules, adapter = None, [], {}, None
    for path in paths:
        path = path/'report.json' if path.is_dir() else path
        raw = path.read_bytes()
        data = json.loads(raw)
        if not data.get('completed'):
            raise ValueError(f'incomplete R4 run: {path}')
        contract = [data['source']['state_sha256'], data['historical_archive_sha256'], data['source']['assets'],
                    data['runtime'], data['timed_batches'], data['warmup_batches'], data['consumer_fields']]
        if common is not None and contract != common:
            raise ValueError(f'different source, cameras, mesh layout or batch schedule: {path}')
        common = contract
        for w, schedule in data['world_schedule'].items():
            if schedules.setdefault(w,schedule) != schedule:
                raise ValueError(f'different replay sequence for world {w}')
        origin.append(dict(report=str(path.resolve()), sha256=sha256(raw).hexdigest()))
        for scale, q in data.get('quality', {}).items():
            if scale in quality_rows:
                raise ValueError(f'duplicate quality evidence for {scale} worlds')
            quality_rows[scale] = dict(host_vs_original=q['routes']['cpu_single'],
                                       cross_path_all_planes_exact=data['cross_path_all_planes_exact'])
        for order, by_scale in data.get('timing', {}).items():
            compact = modes.setdefault(order,{})
            for scale, routes in by_scale.items():
                if scale in compact:
                    raise ValueError(f'duplicate {order}/{scale} timing evidence')
                compact[scale] = {}
                for r in ROUTES:
                    info = routes[r]
                    uid = info['cleanup']['device_uuid']
                    if adapter is not None and uid != adapter:
                        raise ValueError('timing reports use different physical Vulkan/CUDA devices')
                    adapter = uid
                    reference = digests.setdefault(scale, info['per_batch_feature_sha256'])
                    if info['per_batch_feature_sha256'] != reference:
                        raise ValueError(f'{order}/{scale}/{r}: CUDA consumer outputs differ across runs')
                    compact[scale][r] = {k:info[k] for k in ('p50_ms','p95_ms','camera_frames_per_second')}
                for info in compact[scale].values():
                    info['p95_speedup_vs_cpu_single'] = routes['cpu_single']['p95_ms']/info['p95_ms']
    if set(modes) != {'forward','reverse'} or set(modes['forward']) != set(modes['reverse']):
        raise ValueError('supply one matched forward and one reverse timing run')
    output.mkdir(parents=True,exist_ok=True)
    result = dict(reports=origin, state_sha256=common[0], device_uuid=adapter, per_batch_sensor_features_exact=True,
                  timing=modes, quality=quality_rows)
    with (output/'comparison.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,help='original or regenerated eight-world manipulation state directory')
    cli.add_argument('--archive-zip',type=Path,default=ORIGINAL)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--library',type=Path)
    cli.add_argument('--compare-reports',nargs='+',type=Path,help='join completed report directories; no rendering')
    cli.add_argument('--shaders',type=Path)
    cli.add_argument('--device-index',type=int,default=0)
    cli.add_argument('--device',type=int,default=0)
    cli.add_argument('--validation',action='store_true')
    cli.add_argument('--batches',type=int,default=300)
    cli.add_argument('--warmup',type=int,default=3)
    cli.add_argument('--scales',nargs='+',type=int,default=[1,2,8])
    cli.add_argument('--phase',choices=('quality','timing','both','resources'),default='both')
    cli.add_argument('--order',choices=('forward','reverse','both'),default='both')
    args = cli.parse_args()
    if args.compare_reports:
        compare_reports(args.compare_reports,args.output)
    elif args.library is None:
        cli.error('--library is required for rendering')
    else:
        run(args)


if __name__ == '__main__':
    main()
