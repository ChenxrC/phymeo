"""Time the public preparation and native RGB capture paths on the thirty-asset room."""
import argparse
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from _home_collection import (COLLECTION_CHAPTERS, build_collection_room, collection_camera,
                              collection_lighting, collection_photo_checks, collection_placements)
from _home_showcase_assets import load_home_assets
from _home_showcase_room import ROOM_EXPOSURE
from film_home_showcase import assemble_sources, FPS, SIZE
from phymeo_rendering import StudioRenderer
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, SceneFrame


def seconds_since(start):
    return (perf_counter_ns() - start) / 1e9


def summary(values):
    if not values:
        return None
    p50, p95 = np.percentile(values, [50, 95], method='linear')
    return dict(n=len(values), p50_ms=float(p50), p95_ms=float(p95))


def cache_size(folder):
    sizes = [p.stat().st_size for p in folder.rglob('*') if p.is_file()]
    return dict(files=len(sizes), bytes=sum(sizes))


def resource_identity(prepared):
    report = prepared.report
    geometry = report['geometry']
    identity = {
        'scene': prepared.scene.key,
        'geometry': [(m.id, geometry['meshes'][m.id]['output_sha256']) for m in prepared.scene.meshes],
        'materials': [],
    }
    for m in prepared.scene.materials:
        base, rough, normal = (getattr(m, name, None) for name in ('base_color_map', 'roughness_map', 'normal_map'))
        identity['materials'].append([m.id, m.base_color, m.roughness, m.metallic, base and base.sha256,
                                      rough and (rough.channel, rough.image.sha256), normal and normal.sha256])
    digest = sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(scene_key=prepared.scene.key, asset_content_sha256=digest,
                meshes=len(prepared.scene.meshes), materials=len(prepared.scene.materials),
                instances=len(prepared.scene.instances), geometry=geometry['after'],
                texture_arrays={kind: report['textures'][kind]['after'] for kind in ('color','normal')})


def capture_metrics(session, frame, camera):
    request = CaptureRequest(camera, RenderPath.RASTER, (Channel.RGB,))
    before = session.diagnostics()['device'].get('resident_profile')
    start = perf_counter_ns()
    shot = session.capture(frame, request)
    elapsed = (perf_counter_ns() - start) / 1e6
    after = session.diagnostics()['device'].get('resident_profile')
    if shot.frame_key != frame.key or shot.camera != camera or shot.time_s != frame.time_s:
        raise RuntimeError('capture was not bound to the requested prepared exposure')
    plane, = shot.planes
    if plane.channel != Channel.RGB or len(plane.data) != camera.width*camera.height*3:
        raise RuntimeError('incomplete native RGB publication')
    row = dict(tour_frame=frame.key.tick, rgb_sha256=sha256(plane.data).hexdigest(), host_capture_ms=elapsed)
    row['gpu_render_ms'] = row['gpu_copy_ms'] = row['gpu_total_ms'] = None
    row['timestamp_samples'] = 0
    if before is not None and after is not None:
        if after['captures'] - before['captures'] != 1:
            raise RuntimeError('GPU profile snapshot spans more or fewer than one capture')
        if before['timestamp_supported'] and after['timestamp_supported']:
            samples = after['timestamp_samples'] - before['timestamp_samples']
            if samples == 1:
                for metric in ('gpu_render', 'gpu_copy', 'gpu_total'):
                    change = after[metric+'_ns'] - before[metric+'_ns']
                    if change < 0 or (metric == 'gpu_total' and change == 0):
                        raise RuntimeError('invalid GPU timestamp delta')
                    row[metric+'_ms'] = change / 1e6
                row['timestamp_samples'] = samples
            elif samples != 0:
                raise RuntimeError('ambiguous GPU timestamp count for a single frame')
    return row


def session_pass(renderer, prepared, views, warmup_tours, measured_tours):
    bound = [prepared.bind_view(frame, camera) for frame, camera in views]
    start = perf_counter_ns()
    session = renderer.open_scene(prepared)
    open_seconds = seconds_since(start)
    with closing(session):
        warm_hashes = {}
        for _ in range(warmup_tours):
            for frame, camera in bound:
                row = capture_metrics(session, frame, camera)
                warm_hashes[row['tour_frame']] = row['rgb_sha256']
        rows = []
        for cycle in range(measured_tours):
            for frame, camera in bound:
                row = capture_metrics(session, frame, camera)
                if row['rgb_sha256'] != warm_hashes[row['tour_frame']]:
                    raise RuntimeError('same static exposure changed across measured tours')
                rows.append(dict(tour=cycle, **row))
        diag = session.diagnostics()
        device = diag['device']
        ray = device['resident_ray_query_info']
        total = (warmup_tours + measured_tours)*len(bound)
        if (device['validation_errors'] or not ray['enabled'] or ray['shadow_captures'] != total
                or diag['msaa']['actual_color_samples'] != 4 or diag['area_light_samples_per_panel'] != 16
                or not device['resident_environment_info']['loaded']):
            raise RuntimeError('resident capture configuration or validation differs from the requested benchmark')
        profile = device.get('resident_profile') or {}
        if profile and profile['captures'] != total:
            raise RuntimeError('profile capture total does not match the declared tours')
        available = sum(r['timestamp_samples'] for r in rows)
        measures = {}
        for key in ('host_capture_ms', 'gpu_render_ms', 'gpu_copy_ms', 'gpu_total_ms'):
            measures[key] = summary([r[key] for r in rows if r[key] is not None])
        return dict(open_scene_s=open_seconds, samples=rows, metrics=measures,
                    per_view={f.key.tick: {key: summary([r[key] for r in rows if r['tour_frame']==f.key.tick
                                                         and r[key] is not None]) for key in measures}
                              for f, _ in bound},
                    timestamp_status='available' if available == len(rows) else 'partial' if available else 'unavailable',
                    gpu_timestamp_samples=available, timestamp_period_ns=profile.get('timestamp_period_ns'),
                    timestamp_valid_bits=profile.get('timestamp_valid_bits'), warmup_exposures=warm_hashes,
                    device={k:device.get(k) for k in ('device_name','device_uuid','driver_version','validation_errors')},
                    msaa_samples=diag['msaa']['actual_color_samples'], panel_samples=diag['area_light_samples_per_panel'],
                    ray_queries_enabled=bool(ray['enabled']), shadow_captures=ray['shadow_captures'])


def run(args):
    if args.warmup_tours < 1 or args.measured_tours < 2:
        raise ValueError('use one or more warmup tours and at least two measured tours')
    output, cache = args.output.resolve(), args.asset_cache.resolve()
    output.mkdir(parents=True, exist_ok=False)
    derived = output/'derived-cache'
    derived.mkdir()
    root = args.assets.resolve(strict=True)
    print('Constructing one source manifest for all measurements', flush=True)
    start = perf_counter_ns()
    objects = load_home_assets(root, cache/'objects', count=30)
    room = build_collection_room(cache/'room')
    scene, poses, _, _ = assemble_sources(objects, room, collection_placements(objects), root, 'home-collection')
    assembly = seconds_since(start)
    views = []
    for tick in collection_photo_checks():
        frame = SceneFrame(FrameKey(scene.key, 'whole-room', 0, tick, tick, 'authored'), tick/FPS, poses)
        views.append((frame, collection_camera(frame.key, frame.time_s, *SIZE)))
    if len(views) != 13 or len(objects) != 30:
        raise RuntimeError('the reference scene or camera schedule changed')
    library = args.library.resolve(strict=True)
    renderer = StudioRenderer(library_path=library, shader_directory=library.parent, device_index=args.device_index,
                              validation=True, studio=collection_lighting(), exposure=ROOM_EXPOSURE, panel_samples=16)
    report = dict(schema='phymeo-rendering/scene-preparation-benchmark-v1',
                  dimensions=SIZE, targets='native RGB8 pinhole; no CPU backplate, supersampling, or encoding',
                  sources={a.key:a.details['source_sha256'] for a in objects}, source_assembly_s=assembly,
                  exposure=ROOM_EXPOSURE, protected_meshes=['home-oak'], view_schedule=COLLECTION_CHAPTERS,
                  library_sha256=sha256(library.read_bytes()).hexdigest(), device_index=args.device_index,
                  derived_cache_reset='new, exclusive, initially empty directory; OS and driver caches are uncontrolled',
                  warmup_tours=args.warmup_tours, measured_tours=args.measured_tours,
                  timing='host monotonic wall clock; GPU cumulative Vulkan timestamp differences for one capture; '
                         'RGB hashes and GPU diagnostic reads outside host capture timer; linear percentiles',
                  passes={})
    cases = []
    for label in ('cold','warm'):
        initial = cache_size(derived)
        if label == 'cold' and initial['files']:
            raise RuntimeError('cold preparation cache is not empty')
        start = perf_counter_ns()
        selected = renderer.prepare_scene(scene, usage='film', cameras=views, preserve_meshes=('home-oak',),
                                          cache_dir=derived)
        elapsed = seconds_since(start)
        if selected.scene.instances != scene.instances:
            raise RuntimeError('preparation changed scene instance, pose or material bindings')
        report['passes'][label] = dict(prepare_scene_s=elapsed, cache_before=initial, cache_after=cache_size(derived),
                                       resources=resource_identity(selected))
        cases.append(selected)
        print(f'{label}: prepare_scene={elapsed:.3f} seconds', flush=True)
    if (cases[0].scene != cases[1].scene or
            report['passes']['cold']['resources'] != report['passes']['warm']['resources']):
        raise RuntimeError('cold and warm cache produced different scene or asset content')
    for label, prepared in zip(('cold','warm'), cases, strict=True):
        print(f'{label}: opening session and measuring native tours', flush=True)
        report['passes'][label].update(session_pass(renderer, prepared, views, args.warmup_tours, args.measured_tours))
    cold, warm = (report['passes'][k] for k in ('cold','warm'))
    if (cold['warmup_exposures'] != warm['warmup_exposures'] or cold['device'] != warm['device'] or
            len(set(cold['warmup_exposures'].values())) < 2):
        raise RuntimeError('cross-session image parity, camera diversity or device check failed')
    report['all_timed_rgb_sha256_identical_per_exposure'] = True
    (output/'benchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                                         encoding='utf-8')
    for case in ('cold','warm'):
        row = report['passes'][case]
        print(case, row['open_scene_s'], row['metrics'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('library','assets','asset-cache','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--device-index', type=int, default=1)
    parser.add_argument('--warmup-tours', type=int, default=1)
    parser.add_argument('--measured-tours', type=int, default=3)
    run(parser.parse_args())
