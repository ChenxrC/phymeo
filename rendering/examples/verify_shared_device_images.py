"""Compare CPU camera batch uploads and direct Vulkan/CUDA images through the same GPU consumer."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch

from _device_sensor_workload import FEATURES, PinnedUploader, audit_planes, numpy_features, sensor_features
from verify_live_pose_raster import H, KEYS, SIZE, _scene
from verify_mounted_camera_quality import COMMANDS, FIXED, LENS, MODES, STAGES, XML, _must_reject, _request
from verify_shared_camera_batch import _sheet, _statistics
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.cameras import Camera
from phymeo_rendering.device_pose_packing import DeviceCameraMount, DeviceVisualBatchBinding, DeviceVisualBinding
from phymeo_rendering.outputs import Channel
from phymeo_rendering.scene import Pose


ROUTES = ('cpu_upload', 'direct_cuda')
ARCHIVE = Path(__file__).resolve().parents[1] / 'results/device-interop/shared-camera-batch/validation.json'
COUNTERS = ('attachment_d2d_copies', 'attachment_d2d_bytes', 'materialization_d2d_copies',
            'materialization_d2d_bytes', 'image_d2h_bytes', 'image_h2d_bytes', 'return_event_waits')


def telemetry(session):
    report = session.diagnostics()
    device = report['device']
    native = device['resident_telemetry']
    img = device.get('resident_device_output') or {}
    poses = device.get('resident_shared_matrices') or {}
    return {**{c: native[c] for c in ('readback_bytes', 'submissions', 'fence_wait_calls', 'fence_wait_wall_ns')},
            **{c: img.get(c, 0) for c in COUNTERS},
            **{'output_' + c: img.get(c, 0) for c in ('batches', 'views', 'image_bytes')},
            **{'pose_' + c: poses.get(c, 0) for c in ('publications', 'captures', 'matrix_d2d_bytes')},
            'device_image_status_d2h_bytes': report.get('device_image_status_d2h_bytes', 0),
            'cpu_native_batch_output_bytes': (device.get('resident_shared_batch') or {}).get('output_bytes', 0),
            'host_transform_input_bytes': device['resident_host_transform_input_bytes']}


def requests_for(session, exposure, mixed):
    if not mixed:
        cameras = (Camera('fixed', exposure.key, FIXED, **LENS),
                   session.mounted_camera(exposure, 'gimbal-cam', **LENS))
        return tuple(_request(c) for c in cameras)
    def lens(w, h):
        return dict(LENS, width=w, height=h, fx_px=LENS['fx_px']*w/SIZE[0], fy_px=LENS['fy_px']*h/SIZE[1],
                    cx_px=LENS['cx_px']*w/SIZE[0], cy_px=LENS['cy_px']*h/SIZE[1])
    mounted = session.mounted_camera(exposure, 'gimbal-cam', **lens(157, 101))
    fixed = Camera('fixed', exposure.key, FIXED, **lens(237, 165))
    return (replace(_request(mounted), channels=(Channel.VALID, Channel.INSTANCE_ID)),
            replace(_request(fixed), channels=(Channel.FORWARD_DEPTH, Channel.RGB)))


def deliver(route, session, ws, packed, delivery, *, keys=KEYS, mixed=False, guard=None):
    output = []
    for key in keys:
        with session.publish_packed_device_poses(packed, ws, key) as exposure:
            requests = requests_for(session, exposure, mixed)
            if route == 'cpu_upload':
                output.extend(session.capture_device_poses_batch(exposure, requests))
            else:
                if guard is not None and key == KEYS[0]:
                    before = telemetry(session)
                    forged = replace(requests[-1], camera=replace(requests[-1].camera, world_pose=Pose()))
                    guard['final_request_rejected'] = _must_reject(lambda: session.capture_device_poses_device_batch(
                        exposure, requests[:-1] + (forged,), consumer_stream=delivery), 'invalid final camera')
                    if before != telemetry(session):
                        raise RuntimeError('an invalid final request partially delivered a device batch')
                out = session.capture_device_poses_device_batch(exposure, requests, consumer_stream=delivery)
                output.append(out)
    if route == 'cpu_upload':
        return tuple(output)
    if len(output) == 1:
        return output[0]
    return type(output[0]).combine(tuple(output))


def transact(route, session, ws, packed, delivery, consumer, uploader, events):
    begin, work_begin, done = events
    start = perf_counter_ns()
    begin.record(consumer)
    raw = deliver(route, session, ws, packed, delivery)
    with torch.cuda.stream(consumer):
        if route == 'cpu_upload':
            products = uploader.upload(raw)
        else:
            products = [shot.planes for shot in raw.wait(consumer)]
        work_begin.record(consumer)
        result = sensor_features(products)
        done.record(consumer)
    done.synchronize()
    wall_ms = (perf_counter_ns()-start)/1e6
    # The first event includes potential host scheduling gaps. The second interval brackets the common kernels.
    return result, (wall_ms, begin.elapsed_time(done), work_begin.elapsed_time(done))


def _diff(a, b):
    return {k: b[k]-x for k, x in a.items()}


def benchmark(args, sessions, ws, packed, delivery, consumer, uploader, expected):
    samples = {m: {r: [] for r in ROUTES} for m in sessions}
    events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
    last = {}
    modes = list(sessions)[::(-1 if args.reverse_order else 1)]
    h2d = {m: {r: 0 for r in ROUTES} for m in modes}
    for run in range(-args.warmup, args.repeats):
        if run == 0:
            before = {m: {r: telemetry(s[r]) for r in ROUTES} for m, s in sessions.items()}
        offset = run % len(modes)
        for m in modes[offset:]+modes[:offset]:
            reverse = (run + modes.index(m) + args.reverse_order) % 2
            for route in ROUTES[::(-1 if reverse else 1)]:
                at = uploader.h2d_bytes
                result, elapsed = transact(route, sessions[m][route], ws, packed, delivery, consumer, uploader, events)
                if run >= 0:
                    samples[m][route].append(elapsed)
                    h2d[m][route] += uploader.h2d_bytes-at
                    last[m, route] = result
    data = {}
    for m, routes in samples.items():
        data[m] = {}
        for route, rows in routes.items():
            # Only the final tiny feature tensor is downloaded, outside the ended measurements.
            last_cpu = last[m, route].cpu().numpy()
            np.testing.assert_array_equal(last_cpu, expected[m])
            d = _diff(before[m][route], telemetry(sessions[m][route]))
            pixels = args.repeats * 4 * SIZE[0] * SIZE[1]
            direct = route == 'direct_cuda'
            if (d['readback_bytes'] != (0 if direct else 13*pixels)
                    or d['cpu_native_batch_output_bytes'] != (0 if direct else 13*pixels)
                    or h2d[m][route] != (0 if direct else 12*pixels)
                    or d['attachment_d2d_bytes'] != (13*pixels if direct else 0)
                    or d['attachment_d2d_copies'] != (2*args.repeats if direct else 0)
                    or d['device_image_status_d2h_bytes'] != (2*args.repeats if direct else 0)
                    or d['output_batches'] != (2*args.repeats if direct else 0)
                    or d['output_views'] != (4*args.repeats if direct else 0)
                    or d['output_image_bytes'] != (13*pixels if direct else 0)
                    or d['materialization_d2d_copies'] or d['materialization_d2d_bytes']
                    or d['image_h2d_bytes'] or d['image_d2h_bytes'] or d['host_transform_input_bytes']
                    or d['pose_publications'] != 2*args.repeats):
                raise ValueError(f'{m}/{route}: unexpected image or pose transfer: {d}, upload={h2d[m][route]}')
            a = np.asarray(rows)
            data[m][route] = dict(wall=_statistics(a[:, 0]), wall_samples_ms=a[:, 0].tolist(),
                                  stream_elapsed=_statistics(a[:, 1]), consumer_kernels=_statistics(a[:, 2]),
                                  counters=d, baseline_public_image_h2d_bytes=h2d[m][route],
                                  final_consumer_values=last_cpu.tolist(),
                                  final_consumer_verification_d2h_bytes_outside_timing=last_cpu.nbytes)
        diff = np.array(data[m]['cpu_upload']['wall_samples_ms']) - data[m]['direct_cuda']['wall_samples_ms']
        data[m]['paired_cpu_minus_direct_median_ms'] = float(np.median(diff))
        data[m]['faster_direct_rounds'] = int((diff > 0).sum())
    return dict(modes=data, warmup_rounds=args.warmup, timed_rounds=args.repeats, starting_mode_order=modes,
                reverse_order=args.reverse_order,
                scope='one prepacked accepted endpoint; two world publications and four captures; common GPU '
                      'consumer on a distinct nondefault stream; wall clock ends at its completion event; '
                      'host baseline stages immutable CPU planes through reusable pinned buffers',
                stream_elapsed_scope='event window starts before rendering and may include idle or host enqueue gaps',
                consumer_kernels_scope='same event-bracketed RGB/depth/ID/valid reductions after upload or ready wait',
                archive_hardware_validation_and_hashing_in_timing=False)


def _slow_reuse(session, packed, ws, out, expected, delivery):
    """Read old tensors on a delayed second stream while reusing the Vulkan output slot."""
    slow = torch.cuda.Stream(device=delivery.device)
    done, entered = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(slow):
        old = out.wait(slow)
        entered.record(slow)
        # Bounded artificial consumer delay, used only in the ownership test.
        torch.cuda._sleep(40_000_000)
        result = sensor_features([s.planes for s in old])
        done.record(slow)
    pending = not done.query()
    # This writes a different pair of image extents/masks through the same session.
    newer = deliver('direct_cuda', session, ws, packed, delivery, keys=(KEYS[0],), mixed=True)
    newer.synchronize()
    pending_after = not done.query()
    done.synchronize()
    verified = result.cpu().numpy()
    np.testing.assert_array_equal(verified, expected)
    if not pending:
        raise RuntimeError('slow consumer already completed before publishing the next output')
    return dict(pending_before_reuse=pending, pending_after_reuse=pending_after,
                delayed_consumer_stream_ms=entered.elapsed_time(done), old_consumer_matches=True,
                verification_feature_d2h_bytes=verified.nbytes)


def _mixed_proof(mode_sessions, packed, ws, delivery, consumer):
    key = (KEYS[0],)
    cpu = deliver('cpu_upload', mode_sessions['cpu_upload'], ws, packed, delivery, keys=key, mixed=True)
    s = mode_sessions['direct_cuda']
    d0 = telemetry(s)
    out = deliver('direct_cuda', s, ws, packed, delivery, keys=key, mixed=True)
    out.synchronize()
    with torch.cuda.stream(consumer):
        shots = out.wait(consumer)
        hashes, audited = audit_planes(shots, cpu, consumer.device)
    d = _diff(d0, telemetry(s))
    compact = 157*101*5 + 237*165*8
    aligned = ((157*101*5 + 3)//4)*4 + 237*165*8
    if d['readback_bytes'] or d['attachment_d2d_bytes'] != aligned:
        raise ValueError(f'odd-size mixed views have wrong transport size: {d}')
    # Verify the public opt-in host handoff for a previously submitted device batch too.
    readback = out.readback_async(consumer).result()
    if readback != cpu:
        raise ValueError('explicit asynchronous CPU readback differs from the normal CPU capture')
    readback_bytes = sum(len(p.data) for shot in readback for p in shot.planes)
    return dict(native_compact_image_bytes=compact, raw_bridge_storage_bytes=aligned, raw_oracle_d2h_bytes=audited,
                explicit_async_readback_public_bytes=readback_bytes,
                views=[dict(camera=s.camera.id, width=s.camera.width, height=s.camera.height,
                            channels=[c.value for c in s.planes], sha256=h) for s, h in zip(shots, hashes, strict=True)])


def retained_audit(held, consumer):
    report = {}
    for mode, (out, cpu) in held.items():
        out.synchronize()
        with torch.cuda.stream(consumer):
            hashes, size = audit_planes(out.wait(consumer), cpu, consumer.device)
        report[mode] = dict(initial_sha256=hashes, raw_oracle_d2h_bytes=size)
    return report


def run(args):
    from phymeo.backends.mujoco_heterogeneous import (
        MuJoCoTyxHeterogeneousBatch, MuJoCoTyxModelSpec, MuJoCoTyxWorldSpec)
    from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    modes = [s.strip().upper() for s in args.modes.split(',')]
    if not set(modes) <= MODES.keys() or len(set(modes)) != len(modes) or min(args.warmup, args.repeats-1) < 1:
        raise ValueError('select distinct SW1/SW4/RQ1/RQ4 modes, at least one warmup and two timed rounds')
    src = args.quality_report.read_bytes()
    archive = json.loads(src)
    prior = {(r['mode'], r['stage'], r['world'], r['view']): r for r in archive['captures']}
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('choose an empty output directory')
    meshes = args.output / 'meshes'
    meshes.mkdir()
    spec = MuJoCoTyxModelSpec('gimbal', compile_mujoco_xml(XML, require_asset=False))
    model = spec.compilation.fresh_native_model()
    scene, nodes = _scene(model, meshes)
    scene = replace(scene, key='native-mounted-camera-' + sha256(XML.encode()).hexdigest()[:16])
    if archive['scene'] != scene.key or archive['exposure'] != 1:
        raise ValueError('archived exposures use a different scene or exposure')
    camera_id = int(model.camera('gimbal-cam').id)
    w, x, y, z = map(float, model.cam_quat[camera_id])
    mount = DeviceCameraMount('gimbal-cam', int(model.cam_bodyid[camera_id]),
                              Pose(tuple(model.cam_pos[camera_id]), (x, y, z, w)))
    worlds = tuple(MuJoCoTyxWorldSpec(k, spec.key) for k in KEYS)
    device = torch.device('cuda', args.device)
    sessions, held, lifetime, mixed, guards, images, raw_rows, expected = {}, {}, {}, {}, {}, {}, [], {}
    main_mode = next(m for m in ('RQ4', 'SW4', 'RQ1', 'SW1') if m in modes)
    raw_audit_bytes = completed_feature_bytes = 0
    frames = []
    with (torch.cuda.device(device), MuJoCoTyxHeterogeneousBatch((spec,), worlds, mode='cuda') as physics,
          physics.prepare_device_workspace(include_bodies=True) as ws, ExitStack() as stack):
        if ws.binding['device'] != args.device:
            raise ValueError('TYX and CUDA image delivery are using different devices')
        producer, delivery, consumer = (torch.cuda.Stream(device=device) for _ in range(3))
        if len({producer.cuda_stream, delivery.cuda_stream, consumer.cuda_stream, 0}) != 4:
            raise ValueError('expected separate nondefault producer, delivery and consumer streams')
        uploader = PinnedUploader(device)
        binding = DeviceVisualBatchBinding(tuple(DeviceVisualBinding(scene, ws, k, nodes, camera_mounts=(mount,))
                                                for k in KEYS))
        for mode in modes:
            sessions[mode] = {}
            for route in ROUTES:
                shadow, samples = MODES[mode]
                session = NativeBackend(library_path=args.library, shader_directory=args.shader_directory or args.library.parent,
                                        device_index=args.device_index, validation=args.validation, resident=True,
                                        display_transform='clip', exposure=1., shadow_mode=shadow, msaa_samples=samples,
                                        shared_cuda_device=args.device, shared_camera_capacity=1).open_scene(scene)
                stack.callback(session.close)
                sessions[mode][route] = session
                if session.diagnostics()['msaa']['actual_color_samples'] != samples:
                    raise ValueError(f'{mode}: color sampling differs from the requested mode')
        physics.set_controls({k: MuJoCoControl(cmd) for k, cmd in zip(KEYS, COMMANDS, strict=True)})
        previous = None
        for idx, stage in enumerate(STAGES):
            if stage in ('step40', 'step80'):
                for _ in range(40):
                    if physics.step(H) != dict.fromkeys(KEYS, True):
                        raise RuntimeError('nominal physical step rejected')
            if stage == 'reset41':
                physics.reset_states({KEYS[0]: MuJoCoInitialState((-.032, -.09), (0., 0.), time=0.)},
                                     expected_revision=physics.checkpoint_revision)
            if previous is not None:
                def stale():
                    with sessions[main_mode]['direct_cuda'].publish_packed_device_poses(previous, ws, KEYS[0]):
                        pass
                guards['stale_before_'+stage] = _must_reject(stale, 'old physics snapshot')
            frame = ws.refresh(expected_revision=physics.checkpoint_revision)
            with torch.cuda.stream(producer):
                snap = ws.body_pose_snapshot(frame)
            if frame.ticks != ((0, 80) if stage == 'reset41' else (40*idx, 40*idx)):
                raise ValueError('unexpected accepted physical ticks')
            packed = binding.pack(snap, producer_stream=producer)
            previous = packed
            times = {k: ws.accepted_frame_time(frame, k) for k in KEYS}
            frames.append(dict(stage=stage, ticks=frame.ticks, revision=frame.revision, time_s=times,
                               world_boundaries=frame.world_boundaries))
            for mode, pair in sessions.items():
                cpu = deliver('cpu_upload', pair['cpu_upload'], ws, packed, delivery)
                guard = guards.setdefault(mode, {}) if stage == 'step40' else None
                out = deliver('direct_cuda', pair['direct_cuda'], ws, packed, delivery, guard=guard)
                out.synchronize()
                with torch.cuda.stream(consumer):
                    shots = out.wait(consumer)
                    hashes, audited = audit_planes(shots, cpu, device)
                    direct_state = sensor_features([s.planes for s in shots]).cpu().numpy()
                    uploaded_state = sensor_features(uploader.upload(cpu)).cpu().numpy()
                raw_audit_bytes += audited
                completed_feature_bytes += direct_state.nbytes+uploaded_state.nbytes
                oracle = np.array([numpy_features(s) for s in cpu], np.int64)
                np.testing.assert_array_equal(direct_state, oracle)
                np.testing.assert_array_equal(uploaded_state, oracle)
                if not (oracle[:, -1] == 0).all() or not (oracle[:, 5] > 0).all() or not (oracle[:, 3]+oracle[:, 4] == SIZE[0]*SIZE[1]).all():
                    raise ValueError('invalid visible, moving or +infinity background sensor population')
                for i, (s, digest) in enumerate(zip(shots, hashes, strict=True)):
                    k, view = KEYS[i//2], ('fixed', 'gimbal-cam')[i%2]
                    old = prior[mode, stage, k, view]
                    if (s.camera.id != view or asdict(s.frame_key) != old['frame_key'] or
                            s.time_s != times[k] or s.time_s != old['time_s']):
                        raise ValueError('combined device batch does not match world/camera order or accepted exposure')
                    matches = digest == old['plane_sha256']
                    if args.require_archive_identity and not matches:
                        raise ValueError(f'{mode}/{stage}/{k}/{view}: differs from archived shared camera planes')
                    ids = next(p.data for p in cpu[i].planes if p.channel == Channel.INSTANCE_ID)
                    ys, xs = np.nonzero(np.frombuffer(ids, '<u4').reshape(SIZE[1], SIZE[0]) == 2)
                    if len(xs) < 20:
                        raise ValueError('moving carriage is absent or too small in the camera')
                    raw_rows.append(dict(mode=mode, stage=stage, world=k, view=view, time_s=s.time_s,
                                         frame_key=asdict(s.frame_key), plane_sha256=digest, archive_exact=matches,
                                         block_centroid_xy=[float(xs.mean()), float(ys.mean())],
                                         gpu_consumer=direct_state[i].tolist()))
                    if mode == main_mode:
                        images[stage, k, view] = next(p.data for p in cpu[i].planes if p.channel == Channel.RGB)
                if stage == 'initial':
                    held[mode] = (out, cpu)
                if stage == 'step40':
                    mixed[mode] = _mixed_proof(pair, packed, ws, delivery, consumer)
                    raw_audit_bytes += mixed[mode]['raw_oracle_d2h_bytes'] + mixed[mode]['explicit_async_readback_public_bytes']
                    guard['duplicate_world_group'] = _must_reject(lambda: type(out).combine((out, out)), 'duplicated worlds')
                    guard['different_frame_group'] = _must_reject(lambda: type(out).combine((held[mode][0], out)),
                                                                 'world images from two accepted frames')
                if stage == 'step80':
                    expected[mode] = oracle
                    if mode == main_mode:
                        lifetime['slow_consumer'] = _slow_reuse(pair['direct_cuda'], packed, ws, out, oracle, delivery)
            if stage in ('step80', 'reset41'):
                record = retained_audit(held, consumer)
                raw_audit_bytes += sum(v['raw_oracle_d2h_bytes'] for v in record.values())
                lifetime['initial_group_after_'+stage] = record
            if stage == 'step80':
                timing = benchmark(args, sessions, ws, packed, delivery, consumer, uploader, expected)
        if (physics.counters['completed_steps'], physics.counters['accepted_world_steps']) != (80, 160):
            raise ValueError('the capture workload advanced or lost a physical endpoint')
        totals = {m: {r: telemetry(s) for r, s in pair.items()} for m, pair in sessions.items()}
    record = retained_audit(held, consumer)
    raw_audit_bytes += sum(v['raw_oracle_d2h_bytes'] for v in record.values())
    lifetime['initial_group_after_native_close'] = record
    final = sessions[modes[0]]['cpu_upload'].diagnostics()['device']
    cleanup = {c: final[c] for c in ('live_resources', 'live_contexts', 'live_frames', 'validation_errors')}
    if any(cleanup.values()):
        raise RuntimeError(f'Vulkan teardown or validation failed: {cleanup}')
    output_close = {}
    for mode, pair in sessions.items():
        d = pair['direct_cuda'].diagnostics()['device']['resident_device_output']
        if d['imports_live'] or d['exports_open'] or d['return_pending'] or d['phase'] != 0:
            raise RuntimeError(f'{mode}: outstanding image bridge resources after close')
        output_close[mode] = d
    by_key = {(r['mode'], r['stage'], r['world'], r['view']): r for r in raw_rows}
    for m in modes:
        for index, k in enumerate(KEYS):
            initial, end = (by_key[m, st, k, 'fixed']['block_centroid_xy'][1] for st in ('initial', 'step80'))
            if abs(end-initial) < 3 or (end-initial)*COMMANDS[index][0] >= 0:
                raise ValueError(f'{m}/{k}: the carriage did not follow its commanded image direction')
            for view in ('fixed', 'gimbal-cam'):
                at80, reset = (by_key[m, st, k, view]['plane_sha256'] for st in ('step80', 'reset41'))
                if (at80 == reset) != (k == KEYS[1]):
                    raise ValueError('device imagery after reset is not isolated to the selected world')
    _sheet(args.output / 'device-images.png', images, main_mode)
    report = dict(device=final['device_name'], library_sha256=sha256(args.library.read_bytes()).hexdigest(),
                  validation=args.validation, cleanup=cleanup, output_after_close=output_close, scene=scene.key,
                  size=SIZE, world_keys=KEYS, controls=dict(zip(KEYS, COMMANDS, strict=True)),
                  frames=frames, captures=raw_rows, accepted_world_steps=160, exposure=1., source_archive=str(args.quality_report),
                  source_archive_sha256=sha256(src).hexdigest(), archive_identity_required=args.require_archive_identity,
                  consumer_features=FEATURES, consumer_moving_ids=(2, 3), consumer_valid_ids=(1, 2, 3),
                  timing=timing, cumulative_native=totals, lifetime=lifetime, mixed_extent_mask=mixed,
                  guards=guards, verification_public_plane_bytes_downloaded=raw_audit_bytes,
                  raw_oracle_scope='sum of explicitly downloaded public plane payloads during quality, '
                                   'lifetime and mixed-channel checks; excluded from timed image transfers',
                  quality_features_cpu_oracle_bytes_downloaded=completed_feature_bytes, sheet_mode=main_mode)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Verified {len(raw_rows)} paired CPU/CUDA camera exposures: {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--shader-directory', type=Path)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--modes', default=','.join(MODES))
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=300)
    parser.add_argument('--reverse-order', action='store_true')
    parser.add_argument('--quality-report', type=Path, default=ARCHIVE)
    parser.add_argument('--require-archive-identity', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
