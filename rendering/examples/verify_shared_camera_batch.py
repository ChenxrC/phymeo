"""Compare per-camera and batched Vulkan acquisition on identical live two-world poses."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from PIL import Image, ImageDraw

from verify_live_pose_raster import H, KEYS, SIZE, _planes, _scene
from verify_mounted_camera_quality import COMMANDS, FIXED, LENS, MODES, STAGES, XML, _must_reject, _request
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.cameras import Camera
from phymeo_rendering.device_pose_packing import DeviceCameraMount, DeviceVisualBatchBinding, DeviceVisualBinding
from phymeo_rendering.outputs import Channel
from phymeo_rendering.scene import Pose


ARCHIVE = Path(__file__).resolve().parents[1] / 'results/device-interop/mounted-quality/validation.json'
ROUTES = ('single', 'batch')
PHASES = ('all_world_pack', 'publish_acquire', 'camera_requests', 'draw_readback', 'release')


def _counts(session):
    diagnostics = session.diagnostics()
    d = diagnostics['device']
    telemetry, shared = d['resident_telemetry'], d.get('resident_shared_matrices') or {}
    profile, rq = d.get('resident_profile') or {}, d.get('resident_ray_query_info') or {}
    return {
        **{k: telemetry[k] for k in ('submissions', 'fence_wait_calls', 'fence_wait_wall_ns', 'readback_bytes')},
        **{k: telemetry[k] for k in ('device_local_bytes_current', 'device_local_bytes_peak',
                                     'host_visible_bytes_current', 'host_visible_bytes_peak')},
        'mapped_readback_capacity': d['resident_readback']['capacity_bytes'],
        **{k: shared.get(k, 0) for k in ('publications', 'captures', 'matrix_d2d_copies', 'matrix_d2d_bytes')},
        **{k: rq.get(k, 0) for k in ('tlas_builds', 'tlas_updates', 'instance_upload_bytes')},
        'camera_uniform_d2d_bytes': d['resident_shared_camera_d2d_bytes'],
        'host_transform_input_bytes': d['resident_host_transform_input_bytes'],
        'batch_publications': diagnostics.get('published_device_pose_batches', 0),
        **{'native_batch_' + key: (d.get('resident_shared_batch') or {}).get(key, 0)
           for key in ('calls', 'views', 'output_bytes')},
        **{'profile_' + k: profile[k] for k in ('captures', 'timestamp_samples', 'gpu_total_ns', 'gpu_render_ns',
           'gpu_copy_ns', 'host_prepare_ns', 'host_upload_ns', 'host_record_ns', 'host_submit_ns',
           'host_wait_ns', 'host_readback_ns') if k in profile},
    }


def _packing_counts(bindings, batch_binding):
    fields = ('packs', 'status_readback_bytes', 'camera_metadata_readback_bytes', 'packed_output_bytes')
    single = {field: sum(getattr(binding, field) for binding in bindings.values()) for field in fields}
    return {'single': single, 'batch': {field: getattr(batch_binding, field) for field in (*fields, 'readback_calls')}}


def _delta(before, after):
    return {k: after[k] - value for k, value in before.items()}


def _world_pair(route, session, manager, binding, ws, snapshot, phases, guard):
    start = perf_counter_ns()
    with manager as exposure:
        acquired = perf_counter_ns()
        cameras = (Camera('fixed', exposure.key, FIXED, **LENS),
                   session.mounted_camera(exposure, 'gimbal-cam', **LENS))
        requests = tuple(_request(camera) for camera in cameras)
        ready = perf_counter_ns()
        if guard is not None:
            before = _counts(session)
        if guard is not None and route == 'batch':
            # A legal first camera followed by invalid mount metadata must leave the entire submission untouched.
            bad = replace(cameras[1], world_pose=Pose())
            reason = _must_reject(lambda: session.capture_device_poses_batch(exposure, (requests[0], _request(bad))),
                                  'a forged final camera in the batch')
            after = _counts(session)
            if before != after:
                raise RuntimeError(f'partial publication after invalid final request: {_delta(before, after)}')
            guard.update(reason=reason, native_work_delta=_delta(before, after))
        if route == 'single':
            shots = tuple(session.capture_device_poses(exposure, request) for request in requests)
        else:
            shots = session.capture_device_poses_batch(exposure, requests)
        drawn = perf_counter_ns()
        if guard is not None:
            delta = _delta(before, _counts(session))
            draws = 2 if route == 'single' else 1
            if (delta['submissions'] != draws or delta['fence_wait_calls'] != draws
                    or delta['captures'] != 2 or delta['batch_publications'] != int(route == 'batch')):
                raise ValueError(f'wrong draw-only native or public counter delta: {delta}')
            guard['successful_same_exposure_draw_delta'] = delta
    released = perf_counter_ns()
    phases[1:] += (acquired-start, ready-acquired, drawn-ready, released-drawn)
    if not isinstance(shots, tuple) or len(shots) != len(cameras):
        raise RuntimeError('camera batch did not return one ordered capture per request')
    # This check is common to both paths. Consumers receive finished CPU-owned planes.
    if exposure.key != binding.frame_key(snapshot) or exposure.time_s != ws.accepted_frame_time(
            snapshot.frame, binding.world_key):
        raise RuntimeError('published exposure differs from the accepted world and clock')
    return tuple(_planes(shot, exposure.key, camera, exposure.time_s)
                 for shot, camera in zip(shots, cameras, strict=True)), cameras


def _draw(route, session, bindings, batch_binding, ws, snapshot, stream, guard=None):
    start = perf_counter_ns()
    phases = np.zeros(len(PHASES), dtype=np.int64)
    packed = None
    if route == 'batch':
        t0 = perf_counter_ns()
        packed = batch_binding.pack(snapshot, producer_stream=stream)
        phases[0] = perf_counter_ns()-t0
    worlds = {}
    for k in KEYS:
        if route == 'single':
            manager = session.publish_device_poses(bindings[k], ws, snapshot, producer_stream=stream)
        else:
            manager = session.publish_packed_device_poses(packed, ws, k)
        worlds[k] = _world_pair(route, session, manager, bindings[k], ws, snapshot, phases,
                                guard if k == KEYS[0] else None)
    return worlds, packed, (perf_counter_ns()-start)/1e6, phases/1e6


def _statistics(values):
    return dict(median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)))


def _bench(args, sessions, bindings, batch_binding, ws, snapshot, stream):
    rows = {m: {r: [] for r in ROUTES} for m in sessions}
    phases = {m: {r: [] for r in ROUTES} for m in sessions}
    base_modes = list(sessions)
    if args.reverse_order:
        base_modes.reverse()
    for run in range(-args.warmup, args.repeats):
        if run == 0:
            # Setup, warmup, resize and the quality/failure probes are excluded from telemetry deltas.
            before = {m: {r: _counts(p[r]) for r in ROUTES} for m, p in sessions.items()}
            pack_start = _packing_counts(bindings, batch_binding)
        offset = run % len(base_modes)
        for m in base_modes[offset:] + base_modes[:offset]:
            swap = (run + base_modes.index(m) + int(args.reverse_order)) % 2
            for route in ROUTES[::(-1 if swap else 1)]:
                _, _, elapsed, parts = _draw(route, sessions[m][route], bindings, batch_binding, ws, snapshot, stream)
                if run >= 0:
                    rows[m][route].append(elapsed)
                    phases[m][route].append(parts)
    results = {}
    for mode, pair in sessions.items():
        results[mode] = {}
        for route in ROUTES:
            delta = _delta(before[mode][route], _counts(pair[route]))
            if (delta['publications'] != 2*args.repeats or delta['captures'] != 4*args.repeats
                    or delta['matrix_d2d_copies'] != 2*args.repeats or delta['host_transform_input_bytes'] != 0
                    or delta['camera_uniform_d2d_bytes'] != 2*args.repeats*76):
                raise ValueError(f'wrong measured transfer/capture count in {mode}/{route}: {delta}')
            expected_batches = 2*args.repeats if route == 'batch' else 0
            # The native transfer retains alpha (4+4+4+1); public RGB strips it (3+4+4+1).
            if (delta['native_batch_calls'] != expected_batches or delta['batch_publications'] != expected_batches
                    or delta['native_batch_views'] != 2*expected_batches or
                    delta['native_batch_output_bytes'] != 2*expected_batches*SIZE[0]*SIZE[1]*13):
                raise ValueError(f'incomplete atomic camera products in {mode}/{route}: {delta}')
            if mode.startswith('RQ') and (delta['tlas_updates'] != 2*args.repeats
                                           or delta['tlas_builds'] or delta['instance_upload_bytes']):
                raise ValueError(f'incorrect TLAS reuse in {mode}/{route}: {delta}')
            part = np.asarray(phases[mode][route])
            results[mode][route] = dict(**_statistics(rows[mode][route]), samples_ms=rows[mode][route],
                                        public_output_bytes_per_round=4*SIZE[0]*SIZE[1]*12,
                                        phases={p: _statistics(part[:, i]) for i, p in enumerate(PHASES)},
                                        native_delta=delta,
                                        profile_mean_ms_per_round={
                                            k.removeprefix('profile_').removesuffix('_ns'): v / (1e6*args.repeats)
                                            for k, v in delta.items() if k.startswith('profile_') and k.endswith('_ns')
                                            and (not k.startswith('profile_gpu_') or delta.get('profile_timestamp_samples'))})
        single, batched = (results[mode][r]['native_delta'] for r in ROUTES)
        for c in ('submissions', 'fence_wait_calls'):
            if single[c] - batched[c] != args.repeats * len(KEYS):
                raise ValueError(f'{mode}: expected one fewer actual {c} per world: single={single}, batch={batched}')
        if single['readback_bytes'] != batched['readback_bytes']:
            raise ValueError(f'{mode}: changed readback size between equivalent sensor products')
        paired = np.asarray(rows[mode]['single'])-np.asarray(rows[mode]['batch'])
        results[mode]['paired_single_minus_batch_ms'] = dict(median_ms=float(np.median(paired)),
                                                            faster_batch_rounds=int((paired > 0).sum()))
    pack_end = _packing_counts(bindings, batch_binding)
    counts = {r: _delta(pack_start[r], pack_end[r]) for r in ROUTES}
    n = args.repeats*len(sessions)
    if (counts['single']['packs'] != 2*n or counts['batch']['packs'] != n
            or counts['batch']['readback_calls'] != n or counts['batch']['status_readback_bytes'] != n
            or counts['single']['status_readback_bytes'] != 2*n or
            any(counts[r]['camera_metadata_readback_bytes'] != 96*n for r in ROUTES)
            or counts['single']['packed_output_bytes'] != counts['batch']['packed_output_bytes']):
        raise ValueError(f'wrong pose pack or readback count: {counts}')
    return dict(modes=results, packing_delta=counts, warmup_per_route=args.warmup, repeats_per_route=args.repeats,
                initial_mode_order=base_modes, reverse_order=args.reverse_order,
                ordering='rotate modes each round and alternate route order per mode and round',
                scope='one already accepted snapshot to four CPU RGB/ID/depth/valid captures, two worlds, '
                      'two cameras per world; includes pose preparation/publication, metadata, draw, readback '
                      'and release; excludes physics, source snapshot, evidence hashing and disk IO',
                phase_scope='single pose packing is included in publish_acquire; batch has an explicit '
                            'all_world_pack, then two acquisitions; phase medians need not sum to wall median',
                profile_scope='native capture host phase counters and per-view Vulkan timestamp intervals; '
                              'mean per four-camera round, excludes matrix publication and TLAS acquisition')


def _sheet(path, images, mode):
    w, h = SIZE
    board = Image.new('RGB', (4*w + 30, 2*(h+26)+6), '#19282d')
    draw = ImageDraw.Draw(board)
    for row, k in enumerate(KEYS):
        for col, (stage, view) in enumerate((('initial', 'fixed'), ('step80', 'fixed'),
                                             ('step80', 'gimbal-cam'), ('reset41', 'fixed'))):
            x, y = 6 + col*(w+6), 6+row*(h+26)
            draw.text((x, y+2), f'{mode} | world {k} | {view} | {stage}', fill='white')
            board.paste(Image.frombytes('RGB', SIZE, images[stage, k, view]), (x, y+20))
    board.save(path)


def _mixed_views(session, packed, ws):
    """Read unlike channel masks and odd-sized targets in reversed camera order from one exposure."""
    with session.publish_packed_device_poses(packed, ws, KEYS[0]) as exposure:
        def lens(w, h):
            return dict(LENS, width=w, height=h, fx_px=LENS['fx_px']*w/SIZE[0], fy_px=LENS['fy_px']*h/SIZE[1],
                        cx_px=LENS['cx_px']*w/SIZE[0], cy_px=LENS['cy_px']*h/SIZE[1])
        sensor = session.mounted_camera(exposure, 'gimbal-cam', **lens(157, 101))
        color = Camera('fixed', exposure.key, FIXED, **lens(237, 165))
        requests = (replace(_request(sensor), channels=(Channel.VALID, Channel.INSTANCE_ID)),
                    replace(_request(color), channels=(Channel.FORWARD_DEPTH, Channel.RGB)))
        baseline = tuple(session.capture_device_poses(exposure, r) for r in requests)
        before = _counts(session)
        batched = session.capture_device_poses_batch(exposure, requests)
        delta = _delta(before, _counts(session))
        # Native output has an odd byte boundary between the sensor-only and RGB views.
        native_bytes = sensor.width*sensor.height*5 + color.width*color.height*8
        if (delta['submissions'] != 1 or delta['fence_wait_calls'] != 1 or
                delta['native_batch_output_bytes'] != native_bytes or delta['readback_bytes'] != native_bytes):
            raise ValueError(f'wrong compact mixed-view readback or submit count: {delta}')
        rows = []
        for r, first, second in zip(requests, baseline, batched, strict=True):
            a, b = (_planes(shot, exposure.key, r.camera, exposure.time_s) for shot in (first, second))
            if tuple(a) != r.channels or tuple(b) != r.channels or a != b:
                raise ValueError('mixed extent, view order or channel mask changed the public CPU planes')
            rows.append(dict(camera=r.camera.id, width=r.camera.width, height=r.camera.height,
                             channels=[c.value for c in r.channels], public_bytes=sum(map(len, b.values())),
                             plane_sha256={c.value: sha256(data).hexdigest() for c, data in b.items()}))
        return dict(world=KEYS[0], frame_key=asdict(exposure.key), views=rows,
                    native_aggregate_bytes=native_bytes, draw_only_native_delta=delta)


def _verify_views(mode, stage, single, batched, expected_archive, sensor_rows, images, sheet_mode, block):
    rows = []
    for k in KEYS:
        splanes, scameras = single[k]
        bplanes, bcameras = batched[k]
        if scameras != bcameras:
            raise ValueError(f'{mode}/{stage}/{k}: camera metadata differs between two GPU routes')
        for a, b, camera in zip(splanes, bplanes, bcameras, strict=True):
            view, key = camera.id, camera.frame_key
            if a.keys() != set(Channel) or b.keys() != set(Channel):
                raise ValueError('one or more CPU sensor products missing')
            shas = {}
            for ch in Channel:
                if type(a[ch]) is not bytes or type(b[ch]) is not bytes or a[ch] != b[ch]:
                    raise ValueError(f'{mode}/{stage}/{k}/{view}: {ch.value} changed between routes')
                shas[ch.value] = sha256(b[ch]).hexdigest()
            sensors = tuple(shas[ch.value] for ch in Channel if ch != Channel.RGB)
            if sensor_rows.setdefault((stage, k, view), sensors) != sensors:
                raise ValueError('center sample sensors changed across shadow or MSAA configurations')
            ids = np.frombuffer(b[Channel.INSTANCE_ID], '<u4').reshape(SIZE[1], SIZE[0])
            y, x = np.nonzero(ids == block)
            if len(x) < 20:
                raise ValueError('commanded carriage is absent or too small')
            if mode == sheet_mode:
                images[stage, k, view] = b[Channel.RGB]
            old = expected_archive[mode, stage, k, view]
            if asdict(key) != old['frame_key']:
                raise ValueError('comparison against a different archived physical endpoint')
            rows.append(dict(mode=mode, stage=stage, world=k, view=view, frame_key=asdict(key),
                             exact_single_vs_batch=True, plane_sha256=shas,
                             matches_archive={ch: digest == old['plane_sha256'][ch] for ch, digest in shas.items()},
                             block_centroid_xy=[float(x.mean()), float(y.mean())]))
    return rows


def run(args):
    import torch
    from phymeo.backends.mujoco_heterogeneous import (
        MuJoCoTyxHeterogeneousBatch, MuJoCoTyxModelSpec, MuJoCoTyxWorldSpec)
    from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    modes = [mode.strip().upper() for mode in args.modes.split(',')]
    if not set(modes) <= MODES.keys() or len(modes) != len(set(modes)):
        raise ValueError('specify distinct SW1, SW4, RQ1 or RQ4 modes')
    if args.repeats < 2 or args.warmup < 1:
        raise ValueError('at least one warmup and two timed rounds required')
    archive_bytes = args.quality_report.read_bytes()
    archive = json.loads(archive_bytes)
    prior = {(r['mode'], r['stage'], r['world'], r['view']): r for r in archive['captures']}
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('choose a new output directory')
    meshes = args.output / 'meshes'
    meshes.mkdir()
    spec = MuJoCoTyxModelSpec('gimbal', compile_mujoco_xml(XML, require_asset=False))
    model = spec.compilation.fresh_native_model()
    scene, nodes = _scene(model, meshes)
    scene = replace(scene, key='native-mounted-camera-' + sha256(XML.encode()).hexdigest()[:16])
    if (archive['scene'], archive['native_mjcf'], archive['width'], archive['height'], archive['exposure']) != (
            scene.key, XML, *SIZE, 1.):
        raise ValueError('archived camera evidence is for a different scene, resolution or exposure')
    block = int(model.geom('block').id)+1
    camera_id = int(model.camera('gimbal-cam').id)
    w, x, y, z = map(float, model.cam_quat[camera_id])
    mount = DeviceCameraMount('gimbal-cam', int(model.cam_bodyid[camera_id]),
                              Pose(tuple(model.cam_pos[camera_id]), (x, y, z, w)))
    worlds = tuple(MuJoCoTyxWorldSpec(k, spec.key) for k in KEYS)
    sessions, images, sensor_rows = {}, {}, {}
    frames, captures, guards, mixed = [], [], {}, {}
    sheet_mode = next(m for m in ('RQ4', 'SW4', 'RQ1', 'SW1') if m in modes)
    with (torch.cuda.device(args.device), MuJoCoTyxHeterogeneousBatch((spec,), worlds, mode='cuda') as physics,
          physics.prepare_device_workspace(include_bodies=True) as ws, ExitStack() as stack):
        if ws.binding['device'] != args.device:
            raise ValueError('CUDA physics and visual binding target different adapters')
        stream = torch.cuda.Stream(device=args.device)
        bindings = {k: DeviceVisualBinding(scene, ws, k, nodes, camera_mounts=(mount,)) for k in KEYS}
        group = DeviceVisualBatchBinding(tuple(bindings[k] for k in KEYS))
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
                    raise ValueError(f'{mode}: wrong realized color sample count')
        physics.set_controls({k: MuJoCoControl(command) for k, command in zip(KEYS, COMMANDS, strict=True)})
        previous = None
        for idx, stage in enumerate(STAGES):
            if stage in ('step40', 'step80'):
                for _ in range(40):
                    if physics.step(H) != dict.fromkeys(KEYS, True):
                        raise RuntimeError('nominal TYX physical step rejected')
            if stage == 'reset41':
                physics.reset_states({KEYS[0]: MuJoCoInitialState((-.032, -.09), (0., 0.), time=0.)},
                                     expected_revision=physics.checkpoint_revision)
            if previous is not None:
                def submit_old_world():
                    with sessions[modes[0]]['batch'].publish_packed_device_poses(previous, ws, KEYS[0]):
                        pass
                guards[f'old_pack_before_{stage}'] = _must_reject(submit_old_world, 'stale multi-world pose pack')
            frame = ws.refresh(expected_revision=physics.checkpoint_revision)
            with torch.cuda.stream(stream):
                snapshot = ws.body_pose_snapshot(frame)
            if frame.ticks != ((0, 80) if stage == 'reset41' else (40*idx, 40*idx)) or snapshot.frame is not frame:
                raise RuntimeError('unexpected accepted native world ticks')
            times = {k: ws.accepted_frame_time(frame, k) for k in KEYS}
            frames.append(dict(stage=stage, ticks=frame.ticks, revision=frame.revision, time_s=times,
                               world_boundaries=frame.world_boundaries))
            for mode in modes:
                views = {}
                for route in ROUTES:
                    proof = {} if stage == 'step40' else None
                    views[route], packed, _, _ = _draw(route, sessions[mode][route], bindings, group, ws, snapshot,
                                                      stream, proof)
                    if route == 'batch':
                        previous = packed
                    if proof is not None:
                        guards[f'{mode}/{route}'] = proof
                result = _verify_views(mode, stage, views['single'], views['batch'], prior, sensor_rows, images,
                                       sheet_mode, block)
                for row in result:
                    row['time_s'] = times[row['world']]
                    if row['time_s'] != prior[mode, stage, row['world'], row['view']]['time_s']:
                        raise ValueError('archive clock differs from the physical exposure')
                captures.extend(result)
                if stage == 'step40':
                    mixed[mode] = _mixed_views(sessions[mode]['batch'], previous, ws)
            if stage == 'step80':
                timing = _bench(args, sessions, bindings, group, ws, snapshot, stream)
        if physics.counters['accepted_world_steps'] != 160 or physics.counters['completed_steps'] != 80:
            raise ValueError('reset or capture advanced the physical simulation')
        audit = {m: {r: _counts(s) for r, s in pair.items()} for m, pair in sessions.items()}
    final = sessions[modes[0]]['single'].diagnostics()['device']
    cleanup = {f: final[f] for f in ('live_resources', 'live_contexts', 'live_frames', 'validation_errors')}
    if any(cleanup.values()):
        raise RuntimeError(f'Vulkan teardown or validation failed: {cleanup}')
    for pair in sessions.values():
        for s in pair.values():
            closed = s.diagnostics()['device']['resident_shared_matrices']
            if not closed['closed'] or closed['exports_open'] or closed['imported_cuda_resources']:
                raise RuntimeError('shared matrix imports did not close')

    records = {(r['mode'], r['stage'], r['world'], r['view']): r for r in captures}
    comparison = {}
    for mode in modes:
        counts = {c.value: sum(row['matches_archive'][c.value] for row in captures if row['mode'] == mode)
                  for c in Channel}
        for i, k in enumerate(KEYS):
            start, end = (records[mode, stage, k, 'fixed'] for stage in ('initial', 'step80'))
            dy = end['block_centroid_xy'][1] - start['block_centroid_xy'][1]
            if abs(dy) < 3 or dy*COMMANDS[i][0] >= 0:
                raise ValueError(f'{mode}/{k}: missing commanded vertical image displacement')
            for view in ('fixed', 'gimbal-cam'):
                a, b = (records[mode, stage, k, view]['plane_sha256'] for stage in ('step80', 'reset41'))
                if (a == b) != (i == 1):
                    raise ValueError(f'{mode}: selected-world reset failed isolation for {k}/{view}')
        expected = len(STAGES)*len(KEYS)*2
        matches = all(v == expected for v in counts.values())
        if args.require_archive_identity and not matches:
            raise ValueError(f'{mode}: matching SHA counts for the pinned reference: {counts} / {expected}')
        contour_match = all(records[mode, 'step80', KEYS[0], 'fixed']['matches_archive'].values())
        comparison[mode] = dict(view_pairs=expected, all_live_channels_byte_exact=True, archive_sha_matches=counts,
                                archived_contour_linear_mae=(archive['comparison'][mode]['contour_display_linear_mae']
                                                             if contour_match else None),
                                single_median_ms=timing['modes'][mode]['single']['median_ms'],
                                batch_median_ms=timing['modes'][mode]['batch']['median_ms'])
    _sheet(args.output / 'camera-batch.png', images, sheet_mode)
    report = dict(device=final['device_name'], library_sha256=sha256(args.library.read_bytes()).hexdigest(),
                  scene=scene.key, size=SIZE, exposure=1., display='clip + IEC sRGB', validation=args.validation,
                  world_keys=KEYS, frames=frames, controls=dict(zip(KEYS, COMMANDS, strict=True)),
                  accepted_world_steps=160, comparison=comparison, captures=captures, publication_guards=guards,
                  mixed_views_outside_benchmark=mixed,
                  archive_report=str(args.quality_report), archive_report_sha256=sha256(archive_bytes).hexdigest(),
                  archive_identity_required=args.require_archive_identity,
                  archive_planes='previous per-world shared CUDA poses, single-camera Vulkan calls',
                  native_bytes_per_pixel=13, public_bytes_per_pixel=12,
                  archival_contour_scope='previous 4x-width/height reference on fixed world 41 at step80; reported '
                                         'only when all four new planes match that archived exposure byte-for-byte',
                  timings=timing, cumulative_native=audit, cleanup=cleanup, sheet_mode=sheet_mode)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Verified {len(captures)} paired views: {args.output}')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--library', type=Path, required=True)
    cli.add_argument('--shader-directory', type=Path)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--modes', default=','.join(MODES))
    cli.add_argument('--device-index', type=int, default=0)
    cli.add_argument('--device', type=int, default=0)
    cli.add_argument('--validation', action='store_true')
    cli.add_argument('--warmup', type=int, default=4)
    cli.add_argument('--repeats', type=int, default=24)
    cli.add_argument('--reverse-order', action='store_true')
    cli.add_argument('--quality-report', type=Path, default=ARCHIVE)
    cli.add_argument('--require-archive-identity', action='store_true')
    run(cli.parse_args())


if __name__ == '__main__':
    main()
