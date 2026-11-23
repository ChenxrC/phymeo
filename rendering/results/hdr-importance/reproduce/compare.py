"""One-pose public GPU diagnostic; saved-output evaluation never recaptures.

Capture examples (same DLL, separately compiled shader directories):
  python compare.py --library DLL --shader-directory EXP_A --output RUN_A --device-index 1 --preset high --grids 2 4 8 --batches 1 4 16 64
  python compare.py --library DLL --shader-directory EXP_B --output RUN_B --device-index 1 --preset high --grids 2 4 8 --batches 1 16 --compare-to RUN_A
Offline reevaluation: python compare.py --output RUN_B --evaluate-only --compare-to RUN_A
Requires the selected rendering/src on PYTHONPATH, NumPy and Pillow. No physics.
"""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter
from urllib.parse import urlsplit
from urllib.request import url2pathname

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[4]
EXAMPLES = REPO / 'rendering/examples'
SIZE = (1280, 720)
SHELL_IDS = (150, 151, 154)
FINGER_IDS = (156, 158)
CHECKPOINTS = (1, 4, 16, 64)
LUMA = np.array((.2126, .7152, .0722), dtype=np.float64)


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(value):
    return sha256(value).hexdigest()


def file_hash(path):
    return digest(Path(path).read_bytes())


def save(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def json_hash(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def uri_path(uri):
    parts = urlsplit(uri)
    require(parts.scheme == 'file' and parts.netloc in ('', 'localhost'), 'only local bound assets are supported')
    return Path(url2pathname(parts.path))


def normalize_meshes(scene):
    result = asdict(scene)
    for mesh in result['meshes']:
        mesh['uri'] = 'sha256:' + file_hash(uri_path(mesh['uri']))
    return result


def geometry_binding(scene, frame, camera):
    stable = normalize_meshes(scene)
    state, view = asdict(frame), asdict(camera)
    state['key']['scene_key'] = '<scene variant>'
    view['frame_key']['scene_key'] = '<scene variant>'
    return {'meshes': stable['meshes'], 'instances': stable['instances'], 'frame': state, 'camera': view}


def lighting_variant(scene, studio, mode):
    if mode == 'no-hdr':
        studio = replace(studio, environment_map=None)
    elif mode == 'no-directional':
        scene = replace(scene, lights=())
    elif mode == 'no-panels':
        studio = replace(studio, panel_strength=0.)
    return scene, studio


def roi_masks(ids, depth, valid):
    valid = (valid == 1) & np.isfinite(depth) & (depth > 0) & (ids > 0)
    interior = np.zeros(ids.shape, dtype=bool)
    center = ids[1:-1, 1:-1]
    core = valid[1:-1, 1:-1].copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            core &= valid[1+dy:ids.shape[0]-1+dy, 1+dx:ids.shape[1]-1+dx]
            core &= ids[1+dy:ids.shape[0]-1+dy, 1+dx:ids.shape[1]-1+dx] == center
    interior[1:-1, 1:-1] = core
    return {'shell': interior & np.isin(ids, SHELL_IDS),
            'fingers': interior & np.isin(ids, FINGER_IDS)}


def distribution(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    if not len(values):
        return {'count': 0}
    require(np.isfinite(values).all(), 'nonfinite diagnostic values')
    ordered = np.sort(values)
    quantiles = {str(p): float(ordered[max(0, math.ceil(p / 100 * len(ordered))-1)]) for p in (1, 5, 50, 95, 99, 100)}
    return {'count': int(len(values)), 'mean': float(values.mean()), 'rms': float(np.sqrt(np.mean(values*values))),
            'standard_deviation': float(values.std()), 'quantile_nearest_rank': quantiles}


def measure(values, masks):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (SIZE[1], SIZE[0], 3), 'RGB extent differs')
    luma = values @ LUMA
    hp = np.zeros(luma.shape, dtype=np.float64)
    hp[1:-1, 1:-1] = luma[1:-1, 1:-1] - .25 * (luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:])
    result = {}
    for name, mask in masks.items():
        pixels = values[mask]
        result[name] = {'pixels': int(mask.sum()), 'evidence': 'available' if mask.sum() >= 128 else 'insufficient_evidence',
                        'mean_rgb': pixels.mean(axis=0).tolist() if len(pixels) else None,
                        'luminance': distribution(luma[mask]), 'max_channel': distribution(values.max(axis=2)[mask]),
                        'same_id_four_neighbor_highpass': distribution(hp[mask]),
                        'absolute_same_id_four_neighbor_highpass': distribution(np.abs(hp[mask]))}
    return result


def display_linear(rgb):
    code = rgb.astype(np.float64) / 255.
    return np.where(code <= .04045, code / 12.92, ((code + .055) / 1.055)**2.4)


def checked_bytes(root, item):
    path = (root / item['file']).resolve()
    require(path.is_relative_to(root.resolve()), 'saved artifact escaped its run directory')
    data = path.read_bytes()
    require(digest(data) == item['sha256'], 'saved artifact hash differs: ' + item['file'])
    return data


def save_bytes(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), 'artifact already exists: ' + str(path))
    path.write_bytes(data)
    return {'file': name, 'sha256': digest(data), 'bytes': len(data)}


def png_record(root, name, rgb):
    path = root / name
    require(not path.exists(), 'PNG already exists: ' + str(path))
    Image.fromarray(rgb).save(path)
    return {'file': name, 'sha256': file_hash(path), 'decoded_rgb_sha256': digest(rgb.tobytes())}


def capture_case(root, key, scene, frame, request, studio, settings, args, shared):
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.outputs import Channel

    directory = root / key
    directory.mkdir()
    chosen = tuple(args.batches) if args.preset == 'high' else (1,)
    count = max(chosen)
    owner = {'status': 'opening', 'settings': settings, 'selected_batches': list(chosen),
             'actual_capture_target': count, 'capture_ms': [], 'captures': [], 'checkpoints': {}}
    session = None
    error = None
    try:
        backend = NativeBackend(library_path=args.library, shader_directory=args.shader_directory,
                                device_index=args.device_index, resident=True, studio=studio, exposure=8.,
                                display_transform='pbr-neutral', validation=False, **settings)
        start = perf_counter()
        session = backend.open_scene(scene)
        owner['open_ms'] = 1000*(perf_counter()-start)
        initial = session.diagnostics()['device']['resident_profile'].copy()
        previous = initial
        for n in range(1, count+1):
            start = perf_counter()
            shot = session.capture(frame, request)
            capture_ms = 1000*(perf_counter()-start)
            require(shot.frame_key == frame.key and shot.time_s == frame.time_s and shot.camera == request.camera,
                    'capture returned a different frame, time or camera')
            planes = {plane.channel: plane.data for plane in shot.planes}
            require(set(planes) == set(Channel), 'expected all four actual public planes')
            diag = session.diagnostics()
            profile = diag['device']['resident_profile']
            require(profile['captures'] - previous['captures'] == 1 and profile['timestamp_samples'] - previous['timestamp_samples'] == 1,
                    'capture/timestamp increment differs')
            timing = {field: (profile[field]-previous[field])/1e6 for field in ('gpu_render_ns', 'gpu_copy_ns', 'gpu_total_ns')}
            previous = profile.copy()
            geometry_hashes = {channel.value: digest(data) for channel, data in planes.items() if channel != Channel.RGB}
            entry = {'capture_index': n, 'accumulated_batches': n if args.preset == 'high' else None,
                     'capture_ms': capture_ms, 'gpu_ms': timing, 'sensor_sha256': geometry_hashes,
                     'public_rgb_sha256': digest(planes[Channel.RGB])}
            owner['capture_ms'].append(capture_ms)
            owner['captures'].append(entry)
            require(diag['area_light_grid'] == settings['area_light_grid'] and diag['shadow_mode'] == 'ray_query', 'wrong grid/shadow path')
            require(diag['reflection']['mode'] == settings['reflection_mode'] and diag['indirect']['mode'] == settings['indirect_mode'], 'wrong feature path')
            require(diag['msaa']['actual_color_samples'] == 1 and diag['spatial_filter']['mode'] == settings['spatial_filter'], 'wrong sampling/filter path')
            require(bool(diag['normal_maps']) == (args.preset == 'high'), 'normal-map selection differs')
            if args.preset == 'high':
                require(diag['progressive']['batches'] == n and diag['indirect']['bounces'] == 2, 'history advanced or reset unexpectedly')
            actual_shaders = set(diag['shader_sha256'].values())
            require(actual_shaders.issubset(set(shared['shader_hashes'].values())), 'backend used a shader outside the declared directory')
            if shared.get('sensor_sha256') is None:
                shared['sensor_sha256'] = geometry_hashes
                ids = np.frombuffer(planes[Channel.INSTANCE_ID], '<u4').reshape(SIZE[::-1])
                depth = np.frombuffer(planes[Channel.FORWARD_DEPTH], '<f4').reshape(SIZE[::-1])
                valid = np.frombuffer(planes[Channel.VALID], 'u1').reshape(SIZE[::-1])
                shared['masks'] = roi_masks(ids, depth, valid)
                for name, mask in shared['masks'].items():
                    Image.fromarray(mask.astype(np.uint8)*255).save(root / f'roi-{name}.png')
                np.savez_compressed(root / 'geometry.npz', instance_id=ids, depth=depth, valid=valid)
                shared['geometry'] = {'file': 'geometry.npz', 'sha256': file_hash(root/'geometry.npz')}
                shared['roi_counts'] = {name: int(mask.sum()) for name, mask in shared['masks'].items()}
                shared['per_id_counts'] = {str(identity): int((ids == identity).sum()) for identity in SHELL_IDS + FINGER_IDS}
            require(shared['sensor_sha256'] == geometry_hashes, 'full sensor bytes differ across same-pose cases/batches')
            if n in chosen:
                rgb = np.frombuffer(planes[Channel.RGB], np.uint8).reshape(SIZE[1], SIZE[0], 3)
                checkpoint = {'batch': n, 'image': png_record(root, f'{key}/b{n:04d}.png', rgb),
                              'diagnostics': diag, 'display_linear_metrics': measure(display_linear(rgb), shared['masks'])}
                owner['checkpoints'][str(n)] = checkpoint
                if args.preset == 'high':
                    read_start = perf_counter()
                    raw = session.read_linear_mean()
                    checkpoint['raw_readback_ms'] = 1000*(perf_counter()-read_start)
                    require(raw['format'] == 'RGBA32F_LE' and raw['space'] == 'scene-linear-before-exposure' and
                            (raw['width'], raw['height']) == SIZE and raw['batches'] == n and raw['primary_samples_per_pixel'] == 16*n,
                            'raw GPU mean metadata differs')
                    require(isinstance(raw['data'], bytes) and len(raw['data']) == SIZE[0]*SIZE[1]*16, 'raw GPU mean ownership/length differs')
                    checkpoint['linear'] = dict(save_bytes(root, f'{key}/b{n:04d}.rgba32f', raw['data']),
                                                format=raw['format'], space=raw['space'], shape=[SIZE[1], SIZE[0], 4],
                                                batches=n, primary_samples_per_pixel=16*n)
                    values = np.frombuffer(raw['data'], '<f4').reshape(SIZE[1], SIZE[0], 4)
                    require(np.isfinite(values).all(), 'raw mean contains nonfinite values; original bytes preserved')
                    checkpoint['scene_linear_metrics'] = measure(values[..., :3], shared['masks'])
                    after_read = session.diagnostics()
                    require(after_read['progressive']['batches'] == n and after_read['device']['resident_profile']['captures'] == profile['captures'], 'raw readback advanced the renderer')
                else:
                    checkpoint['linear'] = {'status': 'unavailable', 'reason': 'original direct-only baseline has no public raw-mean attachment; no inverse-display surrogate'}
                save(directory/'receipt.json', owner)
            print(f'{key} capture {n}/{count}: {capture_ms:.3f} ms' + (' [saved]' if n in chosen else ''), flush=True)
        owner['before_close'] = session.diagnostics()
        owner['status'] = 'complete'
    except Exception as exc:
        error = exc
        owner['status'] = 'failed'
        owner['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        if session is not None:
            try:
                owner.setdefault('before_close', session.diagnostics())
                session.close()
                owner['after_close'] = session.diagnostics()
                device = owner['after_close']['device']
                require(device['closed'] and all(device[key] == 0 for key in ('live_contexts','live_frames','live_resources','validation_errors')), 'native close/live-resource check failed')
                memory = device['resident_telemetry']
                require(memory['final_after_destroy'] is True and memory['device_local_bytes_current'] == 0 and
                        memory['host_visible_bytes_current'] == 0 and memory['allocation_count'] == memory['free_count'],
                        'native final allocation/free evidence is incomplete or unbalanced')
            except Exception as close_error:
                owner['cleanup_error'] = {'type': type(close_error).__name__, 'message': str(close_error)}
                owner['status'] = 'failed'
                if error is None:
                    error = close_error
        save(directory/'receipt.json', owner)
    if error is not None:
        raise error
    owner['timing_summary'] = {'count': count, 'first_ms': owner['capture_ms'][0],
                              'mean_ms': math.fsum(owner['capture_ms'])/count,
                              'total_capture_ms': math.fsum(owner['capture_ms']),
                              'raw_readback_ms': sum(item.get('raw_readback_ms',0.) for item in owner['checkpoints'].values())}
    save(directory/'receipt.json', owner)
    return owner


def run(args):
    sys.path.insert(0, str(EXAMPLES))
    import panda_render_performance as bench
    import phymeo_rendering
    from phymeo_rendering.backends import native

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    args.library = args.library.resolve()
    args.shader_directory = (args.shader_directory or args.library.parent).resolve()
    require(args.library.is_file(), 'DLL missing')
    shader_hashes = {path.name: file_hash(path) for path in sorted(args.shader_directory.glob('*.spv'))}
    require(shader_hashes, 'no compiled SPIR-V files in requested shader directory')
    report = {'schema': 'phymeo-composite-diagnosis-v1', 'status': 'preparing', 'cases': {}, 'physics_steps': 0}
    started = perf_counter()
    try:
        scenes, frames, studio_original, trace = bench.bind_scene(root/'geometry-input')
        scene = scenes[1 if args.preset == 'high' else 0]
        scene, studio = lighting_variant(scene, studio_original, args.lighting)
        frame, request = bench.make_inputs(scene, frames)[bench.HERO]
        require((request.camera.width, request.camera.height) == SIZE and bench.HERO == 60, 'fixed source camera/pose differs')
        by_id = {item.id: item for item in scene.instances}
        for identity in SHELL_IDS + FINGER_IDS:
            require(identity in by_id, 'registered original Panda white ROI ID missing')
        materials = {item.id: item for item in scene.materials}
        require(all(min(materials[by_id[identity].material_id].base_color) >= .9 for identity in SHELL_IDS+FINGER_IDS),
                'registered source white-material binding differs before exposure')
        settings = dict(shadow_mode='ray_query', area_light_grid=2, msaa_samples=1, reflection_mode='environment',
                        indirect_mode='none', indirect_seed=args.seed, indirect_bounces=1, progressive=False, spatial_filter='none')
        if args.preset == 'high':
            settings.update(reflection_mode='single_bounce', indirect_mode='diffuse_bounce', indirect_bounces=2,
                            progressive=True, spatial_filter='guided_settling')
        package_root = Path(phymeo_rendering.__file__).resolve().parent
        geometry = geometry_binding(scene, frame, request.camera)
        protocol = {'schema': 'phymeo-composite-diagnosis-protocol-v1', 'preset': args.preset, 'lighting': args.lighting,
                    'grids': args.grids, 'seed': args.seed, 'selected_batches': args.batches if args.preset == 'high' else [1],
                    'requested_batches': args.batches, 'hero_pose': 60, 'extent': list(SIZE), 'exposure': 8., 'display_transform': 'pbr-neutral',
                    'warmup_captures': 0, 'mode_order': list(args.grids), 'scene': asdict(scene), 'frame': asdict(frame), 'camera': asdict(request.camera),
                    'studio': asdict(studio), 'studio_original': asdict(studio_original), 'settings_before_grid_override': settings,
                    'geometry_binding': geometry, 'geometry_binding_sha256': json_hash(geometry),
                    'materials_sha256': json_hash(asdict(scene)['materials']),
                    'source_trace_sha256': file_hash(trace), 'library': {'file': str(args.library), 'sha256': file_hash(args.library)},
                    'shader_directory': str(args.shader_directory), 'shader_sha256': shader_hashes,
                    'script_sha256': file_hash(__file__), 'benchmark_helper_sha256': file_hash(bench.__file__),
                    'runtime': {'python':sys.version, 'executable':sys.executable, 'platform':platform.platform(), 'numpy':np.__version__,
                                'native_module':str(Path(native.__file__).resolve()), 'package_root':str(package_root)},
                    'package_python_sha256': {path.relative_to(package_root).as_posix():file_hash(path) for path in sorted(package_root.rglob('*.py'))},
                    'device_index':args.device_index,
                    'roi': {'rectangle_xyxy':[0,0,1280,720], 'shell_ids':list(SHELL_IDS), 'finger_ids':list(FINGER_IDS),
                            'rule':'fixed original white-material IDs; VALID and finite positive depth; one-pixel 3x3 same-ID interior; no intensity selection',
                            'minimum_pixels':128, 'registered_instances':[asdict(by_id[identity]) for identity in SHELL_IDS+FINGER_IDS]},
                    'metrics': {'quantiles':'nearest rank 1/5/50/95/99/100 percent',
                                'highpass':'luminance(center) minus mean of four same-ID axial neighbors; spatial structure/noise diagnostic, not pure stochastic variance',
                                'display':'inverse IEC sRGB of actual public RGB8, does not undo Neutral/exposure',
                                'raw':'actual public GPU mean in scene-linear-before-exposure; unavailable for original direct-only baseline'},
                    'timing':'public capture call to return; no warmup; first separately retained; raw readback timed separately; open/upload, PNG and metrics outside capture interval',
                    'lighting_note':'no-hdr removes only environment_map; original environment_radiance=0.125 remains and becomes active; no-directional clears only scene lights; no-panels zeros only panel_strength',
                    'scope':'fixed optical input; one owner per grid; one loop through maximum requested batch; no replay or resampling in evaluation'}
        save(root/'protocol.json', protocol)
        report.update(protocol_sha256=file_hash(root/'protocol.json'), preparation_ms=1000*(perf_counter()-started), status='capturing')
        shared = {'shader_hashes':shader_hashes}
        save(root/'summary.json',report)
        for grid in args.grids:
            key = f'grid{grid}'
            selected = dict(settings, area_light_grid=grid)
            report['cases'][key] = capture_case(root,key,scene,frame,request,studio,selected,args,shared)
            report.update(sensor_sha256=shared['sensor_sha256'], geometry=shared['geometry'], roi_counts=shared['roi_counts'], per_id_counts=shared['per_id_counts'])
            save(root/'summary.json',report)
        report['status'] = 'complete'
        report['wall_seconds_including_preparation_and_saves'] = perf_counter()-started
        report['actual_captures'] = sum(len(case['captures']) for case in report['cases'].values())
        report['limits'] = ['same-pose diagnosis, not a moving-render throughput claim', 'one deterministic seed; no independent-reference error or convergence claim',
                            'same-ID highpass contains real lighting structure; not noise alone', 'all requested checkpoints retained; no quality-driven selection',
                            'no unexposed floating output for direct-only baseline', 'context allocation receipts are not total driver/process VRAM']
        save(root/'summary.json',report)
    except Exception as exc:
        report.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)},wall_seconds_including_preparation_and_saves=perf_counter()-started)
        save(root/'summary.json',report)
        raise
    if args.compare_to:
        evaluate(root,args.compare_to)
    return report


def load_run(root):
    root = Path(root).resolve()
    report = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    require(report['status'] == 'complete', 'offline comparison requires a complete saved run')
    require(file_hash(root/'protocol.json') == report['protocol_sha256'], 'protocol hash differs')
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    geometry = checked_bytes(root,report['geometry'])
    from io import BytesIO
    with np.load(BytesIO(geometry),allow_pickle=False) as saved:
        masks = roi_masks(saved['instance_id'],saved['depth'],saved['valid'])
    return root,report,protocol,masks


def read_rgb(root, item):
    from io import BytesIO
    data = checked_bytes(root,item)
    with Image.open(BytesIO(data)) as picture:
        require(picture.mode == 'RGB' and picture.size == SIZE, 'saved RGB image format changed')
        rgb = np.asarray(picture)
    require(digest(rgb.tobytes()) == item['decoded_rgb_sha256'], 'decoded saved RGB differs')
    return rgb


def difference(current, previous, masks):
    delta = current.astype(np.float64)-previous.astype(np.float64)
    result = {}
    for name,mask in masks.items():
        values = delta[mask]
        result[name] = {'pixels':int(mask.sum()), 'rmse_rgb':float(np.sqrt(np.mean(values*values))) if len(values) else None,
                        'mean_signed_rgb':values.mean(axis=0).tolist() if len(values) else None,
                        'luminance_difference':distribution((delta@LUMA)[mask]),
                        'max_absolute_channel':float(np.max(np.abs(values))) if len(values) else None}
    return result


def evaluate(output, reference):
    root,current,p,masks = load_run(output)
    old_root,old,q,old_masks = load_run(reference)
    require(root != old_root,'comparison requires two different result directories')
    require(p['geometry_binding_sha256'] == q['geometry_binding_sha256'],'comparison geometry/pose/camera differs')
    require(current['sensor_sha256'] == old['sensor_sha256'],'comparison full geometry planes differ')
    require(all(np.array_equal(masks[name],old_masks[name]) for name in masks),'comparison ROI differs')
    if p['preset'] == q['preset']:
        require(p['materials_sha256'] == q['materials_sha256'],'same-preset material inputs differ')
    require(p['seed'] == q['seed'],'paired experiment seeds differ')
    rows = []
    for key in sorted(set(current['cases']) & set(old['cases'])):
        first,second = current['cases'][key]['checkpoints'],old['cases'][key]['checkpoints']
        for n in sorted(set(first)&set(second),key=int):
            a,b = first[n],second[n]
            ca,cb = read_rgb(root,a['image']),read_rgb(old_root,b['image'])
            row = {'grid':key,'batch':int(n),'rgb8_identical':bool(np.array_equal(ca,cb)),
                   'display_linear':difference(display_linear(ca),display_linear(cb),masks),
                   'linear':{'status':'unavailable in one or both runs'}}
            if 'file' in a['linear'] and 'file' in b['linear']:
                ab,bb = checked_bytes(root,a['linear']),checked_bytes(old_root,b['linear'])
                va = np.frombuffer(ab,'<f4').reshape(SIZE[1],SIZE[0],4)
                vb = np.frombuffer(bb,'<f4').reshape(SIZE[1],SIZE[0],4)
                row['linear'] = {'raw_bytes_identical':ab==bb,'space':'scene-linear-before-exposure',
                                 'difference':difference(va[...,:3],vb[...,:3],masks)}
            rows.append(row)
    require(rows,'no matching saved grid/checkpoint pair')
    result = {'schema':'phymeo-composite-saved-comparison-v1','status':'complete','renderer_invocations':0,
              'current':{'path':str(root),'summary_sha256':file_hash(root/'summary.json'),'protocol_sha256':current['protocol_sha256']},
              'reference':{'path':str(old_root),'summary_sha256':file_hash(old_root/'summary.json'),'protocol_sha256':old['protocol_sha256']},
              'settings':{'current':{key:p[key] for key in ('preset','lighting','library','shader_directory')},
                          'reference':{key:q[key] for key in ('preset','lighting','library','shader_directory')}},
              'difference_direction':'current minus reference; reference is a paired experiment, not radiometric truth',
              'full_sensors_identical':True,'rows':rows}
    target = root / ('evaluation-'+file_hash(old_root/'protocol.json')[:16]+'.json')
    save(target,result)
    print(str(target),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path)
    parser.add_argument('--shader-directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device-index',type=int,default=1)
    parser.add_argument('--preset',choices=('high','baseline'),default='high')
    parser.add_argument('--lighting',choices=('original','no-hdr','no-directional','no-panels'),default='original')
    parser.add_argument('--grids',type=int,nargs='+',default=[2,4,8])
    parser.add_argument('--batches',type=int,nargs='+',default=list(CHECKPOINTS))
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--compare-to',type=Path)
    parser.add_argument('--evaluate-only',action='store_true')
    args = parser.parse_args()
    require(len(args.grids)==len(set(args.grids)) and all(grid in (2,4,8) for grid in args.grids),'choose unique grids from 2,4,8')
    require(len(args.batches)==len(set(args.batches)) and all(n in CHECKPOINTS for n in args.batches),'choose unique checkpoints from 1,4,16,64')
    require(0<=args.seed<=2**32-1 and args.device_index>=0,'seed/device index invalid')
    args.batches = sorted(args.batches)
    if args.evaluate_only:
        require(args.compare_to is not None,'--evaluate-only requires --compare-to')
        evaluate(args.output,args.compare_to)
    else:
        require(args.library is not None,'capture requires --library')
        run(args)


if __name__ == '__main__':
    main()
