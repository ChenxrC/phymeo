"""Paired real Panda performance, never a replay of saved timings.

Use the intended rendering/src on PYTHONPATH. Requires NumPy and Pillow.
  python perf.py --old-library OLD_DLL --new-library NEW_DLL --output NEW_DIR
Optional --old-shader-directory/--new-shader-directory permit separately built
shader sets. No scene, camera, seed, exposure or timing-budget tuning arguments.
All 12 owners execute serially: 6 warm + 8 measured exposures, twice per case.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'rendering/examples'))
import numpy as np
from PIL import Image
import panda_render_performance as bench
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.outputs import Channel

WARMUP = tuple(range(27, 33))
MEASURE = tuple(range(57, 65))  # Fixed original trajectory subset, including hero 60.
GRIDS = (2, 4, 8)
SIZE = (1280, 720)
SETTINGS = dict(resident=True, shadow_mode='ray_query', msaa_samples=1,
                reflection_mode='single_bounce', indirect_mode='diffuse_bounce',
                indirect_seed=0, indirect_bounces=2, progressive=True,
                spatial_filter='guided_settling', exposure=8.,
                display_transform='pbr-neutral', validation=False)
GPU_KEYS = ('gpu_render_ns', 'gpu_copy_ns', 'gpu_total_ns')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(data):
    return sha256(data).hexdigest()


def file_hash(path):
    return digest(Path(path).read_bytes())


def json_hash(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def manifest(library, shaders):
    library, shaders = library.resolve(), shaders.resolve()
    require(library.is_file() and shaders.is_dir(), 'missing library or shader directory')
    files = sorted(shaders.glob('*.spv'))
    require(bool(files), 'shader directory contains no SPIR-V')
    return dict(library=str(library), library_sha256=file_hash(library),
                shader_directory=str(shaders), shaders={p.name:file_hash(p) for p in files})


def imported_sources():
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name.startswith('phymeo_rendering') or name in ('panda_render_performance', 'panda_realtime_film'):
            value = getattr(module, '__file__', None)
            if value and Path(value).is_file():
                path = Path(value).resolve()
                result[name] = dict(file=str(path), sha256=file_hash(path))
    return result


def stats(values):
    require(bool(values) and all(math.isfinite(x) and x > 0 for x in values), 'invalid timing')
    ordered = sorted(values)
    mean = sum(values) / len(values)
    hits = sum(x <= 1000/30 for x in values)
    return dict(count=len(values), mean_ms=mean,
                p50_ms=ordered[math.ceil(.50*len(values))-1],
                p95_ms=ordered[math.ceil(.95*len(values))-1],
                fps_from_mean=1000/mean, budget_ms=1000/30,
                within_budget_count=hits, within_budget_fraction=hits/len(values),
                percentile_method='nearest rank; merge all measured rounds first')


def subtract_profile(before, after):
    keys = ('captures', 'timestamp_samples', *GPU_KEYS,
            'host_prepare_ns', 'host_upload_ns', 'host_record_ns',
            'host_submit_ns', 'host_wait_ns', 'host_readback_ns')
    delta = {key:after[key]-before[key] for key in keys}
    require(all(x >= 0 for x in delta.values()), 'device counters moved backwards')
    return delta


def check_capture(diag, build, grid):
    require(diag['library_sha256'] == build['library_sha256'], 'loaded DLL hash differs')
    hashes = set(build['shaders'].values())
    require(diag['shader_sha256'] and all(x in hashes for x in diag['shader_sha256'].values()),
            'used shader is absent from the selected frozen shader directory')
    require(diag['area_light_grid'] == grid and diag['shadow_mode'] == 'ray_query', 'wrong light/shadow mode')
    require(diag['msaa']['actual_color_samples'] == 1 and diag['msaa']['sensor_samples'] == 1, 'wrong sample count')
    require(diag['reflection']['mode'] == 'single_bounce', 'reflection not selected')
    require(diag['indirect']['mode'] == 'diffuse_bounce' and diag['indirect']['seed'] == 0
            and diag['indirect']['bounces'] == 2, 'wrong indirect selection')
    require(diag['progressive']['batches'] == 1 and diag['progressive']['primary_samples_per_pixel'] == 16,
            'moving pose reused an old progressive history')
    require(diag['spatial_filter']['mode'] == 'guided_settling'
            and diag['spatial_filter']['last_capture_filtered'], 'first-batch settling display not active')
    require(bool(diag['normal_maps']) and diag['exposure'] == 8.
            and diag['display_transform'] == 'pbr-neutral', 'wrong material/display selection')
    require(diag['physics_steps'] == 0, 'benchmark must not advance physics')


def check_close(device):
    require(device['closed'], 'owner not closed')
    for key in ('validation_errors', 'validation_warnings', 'live_contexts', 'live_frames', 'live_resources'):
        require(device[key] == 0, 'nonzero close field: ' + key)
    memory = device['resident_telemetry']
    require(memory['final_after_destroy'] and memory['device_local_bytes_current'] == 0
            and memory['host_visible_bytes_current'] == 0
            and memory['allocation_count'] == memory['free_count'], 'unreleased explicit native allocation')


def run_owner(root, row, scene, inputs, studio, build, device_index, canonical):
    record = dict(case=row['case'], implementation=row['implementation'], grid=row['grid'],
                  round=row['round'], order_index=row['order_index'], started_utc=now(),
                  status='running', warmup=[], measurements=[], errors=[])
    owner = None
    original_error = None
    started = perf_counter()
    try:
        # The complete file set is bound before and after each owner, outside capture timing.
        require(manifest(Path(build['library']), Path(build['shader_directory'])) == build,
                'build changed after protocol registration')
        backend = NativeBackend(library_path=Path(build['library']),
                                shader_directory=Path(build['shader_directory']),
                                device_index=device_index, studio=studio,
                                area_light_grid=row['grid'], **SETTINGS)
        opening = perf_counter()
        owner = backend.open_scene(scene)
        record['open_ms'] = (perf_counter()-opening)*1000
        for index in WARMUP:
            shot, elapsed = bench.timed(owner, inputs[index])
            sensors = bench.sensor_digest(shot)
            if index in canonical['warmup_sensors']:
                require(sensors == canonical['warmup_sensors'][index], 'warmup sensors differ')
            else:
                canonical['warmup_sensors'][index] = sensors
            record['warmup'].append(dict(pose_index=index, capture_ms=elapsed, sensor_sha256=sensors))
        initial = owner.diagnostics()
        check_capture(initial, build, row['grid'])
        record['initial_diagnostics'] = initial
        previous = initial['device']['resident_profile']
        for index in MEASURE:
            shot, elapsed = bench.timed(owner, inputs[index])
            current = owner.diagnostics()
            check_capture(current, build, row['grid'])
            profile = current['device']['resident_profile']
            delta = subtract_profile(previous, profile)
            require(delta['captures'] == 1 and delta['timestamp_samples'] == 1, 'one exposure/timestamp expected')
            previous = profile
            planes = {p.channel:p.data for p in shot.planes}
            sensors = {c.value:planes[c] for c in Channel if c != Channel.RGB}
            if index in canonical['sensors']:
                if sensors != canonical['sensors'][index]:
                    failure_arrays = {'actual_' + k:np.frombuffer(v, dtype=np.uint8) for k,v in sensors.items()}
                    failure_arrays.update({'expected_' + k:np.frombuffer(v, dtype=np.uint8)
                                           for k,v in canonical['sensors'][index].items()})
                    failed_path = root / ('sensor-mismatch-round%d-%s-p%d.npz' % (row['round'],row['case'],index))
                    np.savez_compressed(failed_path, **failure_arrays)
                    record['sensor_mismatch'] = dict(file=failed_path.name, sha256=file_hash(failed_path), pose_index=index)
                    raise RuntimeError('full-frame depth/ID/VALID bytes differ at pose ' + str(index))
            else:
                canonical['sensors'][index] = sensors
            rgb_hash = digest(planes[Channel.RGB])
            repeat_key = (row['case'], index)
            if repeat_key in canonical['rgb']:
                require(canonical['rgb'][repeat_key] == rgb_hash, 'same implementation/pose changed between rounds')
            else:
                canonical['rgb'][repeat_key] = rgb_hash
            record['measurements'].append(dict(pose_index=index, capture_ms=elapsed,
                gpu_ms={key:delta[key]/1e6 for key in GPU_KEYS}, profile_delta=delta,
                rgb_sha256=rgb_hash, sensor_sha256={k:digest(v) for k,v in sensors.items()},
                progressive=current['progressive'], spatial_filter=current['spatial_filter'],
                passes=current['last_passes']))
            if index == bench.HERO:
                ids = np.frombuffer(planes[Channel.INSTANCE_ID], dtype='<u4')
                pixels = {str(k):int((ids == k).sum()) for k in (1, 2)}
                require(min(pixels.values()) >= 16, 'eligible mirror payload/tabletop are not visible')
                record['hero_pixels'] = pixels
                record['hero_rgb_sha256'] = rgb_hash
                if row['round'] == 1:
                    path = root / 'images' / (row['case'] + '.png')
                    Image.frombytes('RGB', SIZE, planes[Channel.RGB]).save(path)
                    record['hero_image'] = dict(file=path.relative_to(root).as_posix(),
                                               sha256=file_hash(path), decoded_rgb_sha256=rgb_hash)
        diag = owner.diagnostics()
        record['final_live_diagnostics'] = diag
        identity = dict(device=diag['device']['device_name'], uuid=diag['device']['device_uuid'],
                        triangles=diag['triangles'], decoded_geometry_sha256=diag['decoded_geometry_sha256'],
                        environment={key:diag['environment'][key] for key in
                            ('source_sha256','strength','yaw_degrees','rgba_sha256','diffuse_sh_sha256','side','mip_levels')},
                        lights=diag['last_request']['effective_directional_lights'])
        if canonical['identity'] is None:
            canonical['identity'] = identity
        require(identity == canonical['identity'], 'device/geometry/environment/light inputs differ')
        total = subtract_profile(initial['device']['resident_profile'], diag['device']['resident_profile'])
        require(total['captures'] == len(MEASURE) and total['timestamp_samples'] == len(MEASURE), 'measured counter total differs')
        record['measured_profile_delta'] = total
        record['capture_ms'] = [m['capture_ms'] for m in record['measurements']]
        record['statistics'] = stats(record['capture_ms'])
        record['gpu_mean_ms'] = {key:total[key]/len(MEASURE)/1e6 for key in GPU_KEYS}
        require(manifest(Path(build['library']), Path(build['shader_directory'])) == build, 'build changed during owner lifetime')
        record['status'] = 'passed'
    except BaseException as exc:
        original_error = exc
        record['status'] = 'failed'
        record['errors'].append(dict(type=type(exc).__name__, message=str(exc)))
    finally:
        if owner is not None:
            try:
                owner.close()
                record['closed_device'] = owner.diagnostics()['device']
                check_close(record['closed_device'])
            except BaseException as exc:
                record['status'] = 'failed'
                record['errors'].append(dict(stage='close', type=type(exc).__name__, message=str(exc)))
                if original_error is None:
                    original_error = exc
        record['owner_total_wall_ms'] = (perf_counter()-started)*1000
        record['ended_utc'] = now()
        name = 'round-%d-%s.json' % (row['round'], row['case'])
        save(root / 'receipts' / name, record)
    if original_error is not None:
        raise original_error
    return record


def summarize(records):
    cases = {}
    for grid in GRIDS:
        for implementation in ('old', 'new'):
            key = '%s-grid%d' % (implementation, grid)
            rounds = [r for r in records if r['case'] == key]
            require(len(rounds) == 2 and all(r['status'] == 'passed' for r in rounds), 'incomplete paired rounds')
            values = [v for r in rounds for v in r['capture_ms']]
            cases[key] = dict(statistics=stats(values), round_means_ms=[r['statistics']['mean_ms'] for r in rounds],
                gpu_mean_ms={k:sum(m['gpu_ms'][k] for r in rounds for m in r['measurements'])/len(values) for k in GPU_KEYS},
                image='images/' + key + '.png')
    paired = {}
    for grid in GRIDS:
        old, new = cases['old-grid%d'%grid], cases['new-grid%d'%grid]
        paired[str(grid)] = dict(new_over_old_mean= new['statistics']['mean_ms']/old['statistics']['mean_ms'],
            new_minus_old_mean_ms=new['statistics']['mean_ms']-old['statistics']['mean_ms'],
            new_over_old_p95=new['statistics']['p95_ms']/old['statistics']['p95_ms'],
            per_round=[dict(round=r, pose_index=i,
                new_minus_old_capture_ms=next(x for x in records if x['round']==r and x['case']=='new-grid%d'%grid)['capture_ms'][j]
                -next(x for x in records if x['round']==r and x['case']=='old-grid%d'%grid)['capture_ms'][j])
                for r in (1,2) for j,i in enumerate(MEASURE)])
    return dict(cases=cases, paired=paired, interpretation='descriptive paired timings; no speed threshold or universal FPS claim')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--old-library', required=True, type=Path)
    ap.add_argument('--new-library', required=True, type=Path)
    ap.add_argument('--old-shader-directory', type=Path)
    ap.add_argument('--new-shader-directory', type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--device-index', type=int, default=1)
    args = ap.parse_args()
    builds = {key:manifest(lib, shaders or lib.parent) for key,lib,shaders in (
        ('old',args.old_library,args.old_shader_directory), ('new',args.new_library,args.new_shader_directory))}
    require(SIZE == bench.SIZE and WARMUP == bench.WARMUP and bench.HERO in MEASURE
            and set(MEASURE).issubset(bench.MEASURE), 'registered original input range changed')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'images').mkdir()
    (output/'receipts').mkdir()
    schedule = [(kind,grid) for grid in GRIDS for kind in ('old','new')]
    order = [dict(round=r, order_index=j, implementation=kind, grid=grid, case='%s-grid%d'%(kind,grid))
             for r,items in ((1,schedule),(2,list(reversed(schedule)))) for j,(kind,grid) in enumerate(items)]
    report = dict(schema='phymeo-panda-paired-hdr-performance-v1', status='preparing',
        started_utc=now(), python=sys.executable, python_version=sys.version, platform=platform.platform(),
        driver=dict(file=str(Path(__file__).resolve()), sha256=file_hash(__file__)), builds=builds,
        device_index=args.device_index, width=SIZE[0], height=SIZE[1], settings=SETTINGS,
        warmup_indices=WARMUP, measured_indices=MEASURE, hero_pose=bench.HERO, order=order,
        planned_captures=168, planned_measured_captures=96, planned_warmup_captures=72,
        channels=[c.value for c in Channel], records=[],
        timing_scope='wall capture request to owned four-channel CPU return, including guide/filter and public validation; excludes open/upload, warmup, diagnostics/hashes/PNG. GPU counters are inclusive measured-capture timestamp deltas. Owners serial; second round exactly reverses first.',
        motion_scope='original recorded poses 27..32 warmup then fixed measured subset 57..64; no physics replay and no claim that the skipped trajectory interval was executed',
        memory_scope='explicit VkDeviceMemory telemetry only, not driver/process total VRAM',
        comparison_scope='same source model/assets/pose/camera/seed/settings, except selected old/new DLL+SPV and area-light grid; old/new RGB need not match')
    save(output/'benchmark.json', report)
    canonical = dict(warmup_sensors={}, sensors={}, rgb={}, identity=None)
    try:
        preparation = perf_counter()
        scenes, frames, studio, trace = bench.bind_scene(output/'geometry')
        scene = scenes[1]
        inputs = bench.make_inputs(scene, frames)
        bindings = [dict(pose_index=i, frame=asdict(inputs[i][0]), camera=asdict(inputs[i][1].camera),
                         channels=[c.value for c in inputs[i][1].channels]) for i in WARMUP+MEASURE]
        declaration = dict(scene=asdict(scene), studio=asdict(studio), inputs=bindings,
                           source_trace=dict(file=str(trace), sha256=file_hash(trace)))
        save(output/'input-binding.json', declaration)
        report.update(preparation_ms=(perf_counter()-preparation)*1000,
            input_binding=dict(file='input-binding.json', sha256=file_hash(output/'input-binding.json'),
                               content_sha256=json_hash(declaration)), imported_sources_before=imported_sources(), status='running')
        save(output/'benchmark.json', report)
        for row in order:
            record = run_owner(output, row, scene, inputs, studio, builds[row['implementation']], args.device_index, canonical)
            report['records'].append(record)
            report['identity'] = canonical['identity']
            save(output/'benchmark.json', report)
            print('round %d %s: %.3f ms wall; %.3f ms GPU' %
                  (row['round'], row['case'], record['statistics']['mean_ms'], record['gpu_mean_ms']['gpu_total_ns']), flush=True)
        # Preserve one compact sensor reference per measured pose, not twelve copies.
        # All other sensor comparisons above used the complete immutable byte arrays.
        sensor_arrays = {'p%d_%s'%(i,k):np.frombuffer(v, dtype=np.uint8)
                         for i,planes in canonical['sensors'].items() for k,v in planes.items()}
        np.savez_compressed(output/'sensor-reference.npz', **sensor_arrays)
        report['sensor_reference'] = dict(file='sensor-reference.npz', sha256=file_hash(output/'sensor-reference.npz'),
            layout='uint8 byte views, per channel; public depth little-endian f32, instance ID little-endian u32, VALID u8; 1280x720 top-left order')
        report.update(status='passed', summary=summarize(report['records']), ended_utc=now(),
            actual_measured_captures=sum(len(r['measurements']) for r in report['records']),
            actual_warmup_captures=sum(len(r['warmup']) for r in report['records']),
            imported_sources_after=imported_sources(), full_measured_sensor_bytes_equal=True,
            within_implementation_round_rgb_sha_equal=True)
        save(output/'performance.json', report['summary'])
        save(output/'benchmark.json', report)
    except BaseException as exc:
        report.update(status='failed', ended_utc=now(), error=dict(type=type(exc).__name__,message=str(exc)),
            failure_evidence='receipts retain the first failed owner and its cleanup; accepted previous owner records remain unchanged')
        save(output/'benchmark.json', report)
        raise


if __name__ == '__main__':
    main()
