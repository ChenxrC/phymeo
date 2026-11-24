"""Measure one live physics-to-camera route per process, with the packed-pose suffix timed separately."""
import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from _device_sensor_workload import PinnedUploader
from _heterogeneous_camera_worlds import H, WORLD_KEYS, open_worlds
from _r4_resources import WindowsProcessMemory
from _scale_camera_sensors import SceneSensorConsumer, capture_worlds
from verify_heterogeneous_camera_scale import open_scale_sessions
from verify_shared_camera_batch import _counts, _statistics
from verify_shared_device_images import telemetry


def numbers(mapping):
    return {k: v for k, v in mapping.items() if type(v) in (int, float)}


def counters(world, sessions):
    per_session = [{**_counts(s), **telemetry(s)} for s in sessions.values()]
    return dict(render={k: sum(d[k] for d in per_session) for k in per_session[0]},
                physics=numbers(world.batch.counters), workspace=numbers(world.workspace.counters),
                packing={key: getattr(world.batch_binding, key) for key in
                         ('packs', 'status_readback_bytes', 'camera_metadata_readback_bytes', 'packed_output_bytes')})


def differences(before, after):
    return {group: {key: value - before[group].get(key, 0) for key, value in values.items()}
            for group, values in after.items()}


def sample(world, sessions, args, delivery, consume, uploader, consumer, events):
    # Controls are latched on the solver. Reset, diagnostic sampling, feature hashing
    # and asset construction are all outside this interval.
    t0 = perf_counter()
    world.step()
    t1 = perf_counter()
    packed = world.pack(world.snapshot())
    t2 = perf_counter()
    delivered = capture_worlds(args.route, sessions, world.workspace, packed, world.keys, world.requests, delivery)
    with torch.cuda.stream(consume):
        if args.route == 'direct_cuda':
            planes = [shot.planes for shot in delivered.wait(consume)]
        else:
            planes = uploader.upload(delivered)
        begin, done = events
        begin.record(consume)
        result = consumer(planes)
        done.record(consume)
    done.synchronize()
    end = perf_counter()
    return result, packed.snapshot.frame, {
        'physics_to_consumer_ms': 1000*(end-t0), 'packed_to_consumer_ms': 1000*(end-t2),
        'step_call_ms': 1000*(t1-t0), 'snapshot_and_pack_ms': 1000*(t2-t1),
        'consumer_event_ms': begin.elapsed_time(done)}


def measure(args):
    if args.repeats < 2 or args.warmup < 0:
        raise ValueError('need at least two measured batches and a nonnegative warmup')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'meshes').mkdir()
    keys = WORLD_KEYS[:args.worlds]
    device = torch.device('cuda', args.device)
    samples, observations = [], []
    sampler = WindowsProcessMemory()
    with ExitStack() as stack:
        stack.enter_context(torch.cuda.device(device))
        world = stack.enter_context(open_worlds(args.output / 'meshes', keys, device=args.device))
        sessions = stack.enter_context(open_scale_sessions(
            world, library=args.library, shader_directory=args.shader_directory, device=args.device,
            device_index=args.device_index, validation=args.validation))
        graphics = sessions[keys[0]].diagnostics()['device']
        luid = graphics['resident_telemetry'].get('device_luid_hex')
        sampler.start()
        stack.callback(sampler.stop)
        delivery, consume = (torch.cuda.Stream(device=device) for _ in range(2))
        consumer = SceneSensorConsumer(world.scenes, keys, device)
        uploader = PinnedUploader(device)
        events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
        for _ in range(args.warmup):
            sample(world, sessions, args, delivery, consume, uploader, consumer, events)
        before = counters(world, sessions)
        base_upload = uploader.h2d_bytes
        torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        for index in range(args.repeats):
            if index == args.repeats // 2:
                world.reset(keys=(keys[0],))
            result, frame, timings = sample(world, sessions, args, delivery, consume, uploader, consumer, events)
            samples.append(timings)
            # This small verification download is accounted separately and never part of end-to-end latency.
            observed = result.cpu().numpy()
            if observed.shape != (len(keys)*2, 8) or np.any(observed[:, -1]):
                raise RuntimeError(f'invalid GPU sensor outputs in batch {index}')
            observations.append(dict(ticks=tuple(map(int, frame.ticks)), endpoints_sha256=sha256(observed.tobytes()).hexdigest()))
        ended = perf_counter()
        after = counters(world, sessions)
        delta = differences(before, after)
        n = len(keys) * args.repeats
        pixels = n * 2 * 320 * 240
        host = args.route == 'cpu_upload'
        upload = uploader.h2d_bytes - base_upload
        if (delta['physics']['accepted_world_steps'] != n or delta['packing']['packs'] != args.repeats
                or delta['render']['readback_bytes'] != (13*pixels if host else 0)
                or upload != (12*pixels if host else 0)
                or delta['render']['attachment_d2d_bytes'] != (0 if host else 13*pixels)
                or delta['render']['device_image_status_d2h_bytes'] != (0 if host else n)):
            raise ValueError(f'physical step or image transport accounting mismatch: {delta}, H2D={upload}')
        stats = {name: _statistics([row[name] for row in samples]) for name in samples[0]}
        for name in ('physics_to_consumer_ms', 'packed_to_consumer_ms'):
            stats[name]['camera_images_per_s'] = 1000*len(keys)*2 / np.mean([row[name] for row in samples])
        memory = dict(process=sampler.report(luid, started, ended),
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                      vulkan_device_local_peak_sum_upper_bytes=after['render']['device_local_bytes_peak'],
                      vulkan_host_visible_peak_sum_upper_bytes=after['render']['host_visible_bytes_peak'],
                      solver_manifest_payload_bytes=world.batch.resource_snapshot()['manifest_payload_bytes'])
    cleanup = {k: sessions[keys[0]].diagnostics()['device'][k] for k in
               ('validation_errors', 'live_resources', 'live_contexts', 'live_frames')}
    if any(cleanup.values()):
        raise RuntimeError(f'unclean camera shutdown: {cleanup}')
    report = dict(schema='live-heterogeneous-camera-scale-v1', route=args.route, validation=args.validation,
                  device=graphics['device_name'], device_uuid=graphics['device_uuid'], width=320, height=240,
                  mode='RQ4', keys=keys, h=H, warmup_steps=args.warmup, timed_steps=args.repeats,
                  physical_models=[row['physical_sha256'] for row in world.manifest['worlds']],
                  reset_before_batch=args.repeats//2, reset_world=keys[0],
                  library_sha256=sha256(args.library.read_bytes()).hexdigest(), counters=delta,
                  host_image_upload_bytes=upload, feature_verification_d2h_bytes=observed.nbytes*args.repeats,
                  stats=stats, memory=memory, cleanup=cleanup, observations=observations, timings=samples)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(args.route, len(keys), args.repeats, stats['physics_to_consumer_ms'], flush=True)


def compare(paths, output):
    """Compare four fresh processes in CPU, direct, direct, CPU order."""
    runs = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    if [r['route'] for r in runs] != ['cpu_upload', 'direct_cuda', 'direct_cuda', 'cpu_upload']:
        raise ValueError('expected forward CPU/direct and reverse direct/CPU measurements')
    fields = ('schema', 'device_uuid', 'width', 'height', 'mode', 'keys', 'physical_models', 'h', 'warmup_steps',
              'timed_steps', 'reset_before_batch', 'reset_world', 'library_sha256', 'observations')
    for field in fields:
        if any(run[field] != runs[0][field] for run in runs[1:]):
            raise ValueError(f'live measurement inputs or completed sensor values differ: {field}')
    observations = runs[0]['observations']
    if len(observations) != runs[0]['timed_steps']:
        raise ValueError('incomplete recorded exposure schedule')
    for i, row in enumerate(observations):
        expected = [i + runs[0]['warmup_steps'] + 1] * len(runs[0]['keys'])
        if i >= runs[0]['reset_before_batch']:
            expected[0] = i - runs[0]['reset_before_batch'] + 1
        if row['ticks'] != expected:
            raise ValueError(f'wrong accepted ticks at batch {i}')
    gates = {}
    for phase in ('physics_to_consumer_ms', 'packed_to_consumer_ms'):
        p95 = [run['stats'][phase]['p95_ms'] for run in runs]
        gains = [p95[0] - p95[1], p95[3] - p95[2]]
        drift = abs(p95[0] - p95[3])
        gates[phase] = dict(p95_gain_ms=gains, baseline_drift_ms=drift,
                            both_gains_exceed_baseline_drift=min(gains) > drift)
    report = dict(schema='live-camera-route-comparison-v1', routes=[r['route'] for r in runs],
                  world_keys=runs[0]['keys'], device=runs[0]['device'], repeated_batches=len(observations),
                  matched_sensor_and_clock_batches=len(observations), gates=gates,
                  measurements=[dict(route=r['route'], stats=r['stats'], memory=r['memory'],
                                     host_image_upload_bytes=r['host_image_upload_bytes'],
                                     counters=r['counters'], source_report=p.name,
                                     source_sha256=sha256(p.read_bytes()).hexdigest()) for r, p in zip(runs, paths)])
    with output.open('x', encoding='utf-8') as target:
        json.dump(report, target, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path)
    parser.add_argument('--shader-directory', type=Path)
    parser.add_argument('--route', choices=('cpu_upload', 'direct_cuda'))
    parser.add_argument('--compare', type=Path, nargs=4, metavar=('CPU1', 'GPU1', 'GPU2', 'CPU2'))
    parser.add_argument('--worlds', type=int, choices=(1, 2, 8), default=8)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=300)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.compare:
        compare(args.compare, args.output)
    elif args.library and args.shader_directory and args.route:
        measure(args)
    else:
        parser.error('measurement requires --library, --shader-directory and --route')


if __name__ == '__main__':
    main()
