"""Compare resident host and CUDA-published multi-camera rendering on accepted TYX poses."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

import numpy as np

from _mounted_camera_evidence import contour_error, reference, write_board, write_motion_board
from _panda_scene import look_at, pose_matrix
from _r4_archive import _pose
from verify_live_pose_raster import H, KEYS, SIZE, VIEWS, _compare, _planes, _scene, _shadow_change
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.cameras import Camera
from phymeo_rendering.device_pose_packing import DeviceCameraMount, DeviceVisualBinding
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import NodePose, Pose, SceneFrame, compose_pose


MODES = {'SW1': ('software', 1), 'SW4': ('software', 4),
         'RQ1': ('ray_query', 1), 'RQ4': ('ray_query', 4)}
COMMANDS = ((4., .65), (-4., -.65))
STAGES = ('initial', 'step40', 'step80', 'reset41')
LENS = dict(width=SIZE[0], height=SIZE[1], fx_px=340., fy_px=340.,
            cx_px=160., cy_px=120., near_m=.025, far_m=4.)
FIXED = look_at(VIEWS['front'], (0., 0., -.025), (0., 0., 1.))
XML = """<mujoco model="camera-slide-and-gimbal">
<option timestep=".001" gravity="0 0 0"/>
<default><geom type="box" contype="0" conaffinity="0"/></default>
<worldbody>
  <geom name="receiver" size=".42 .33 .012" pos="0 0 -.17" rgba=".46 .53 .53 1"/>
  <body name="carriage" pos="0 0 .015" euler="0 0 90">
    <inertial pos="0 0 0" mass=".5" diaginertia=".001 .001 .001"/>
    <joint name="slide" type="slide" axis="0 0 1" armature=".04" damping=".2"/>
    <geom name="block" size=".073 .045 .034" pos=".012 -.014 .016"
          euler="0 0 18" rgba=".83 .29 .07 1"/>
    <body name="gimbal" pos="0 0 .045">
      <inertial pos="0 0 0" mass=".07" diaginertia=".0003 .0003 .0003"/>
      <joint name="pitch" type="hinge" axis="0 1 0" armature=".012" damping=".05"/>
      <geom name="cap" size=".033 .052 .015" pos="-.014 -.005 .020"
            euler="12 0 -17" rgba=".13 .37 .61 1"/>
      <camera name="gimbal-cam" mode="fixed" pos=".28 -.28 .21" xyaxes="1 1 0 -.375 .375 1"/>
    </body>
  </body>
</worldbody>
<actuator>
  <motor name="drive" joint="slide" gear="2" ctrllimited="true" ctrlrange="-5 5"/>
  <motor name="tilt" joint="pitch" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
</actuator></mujoco>"""


def _request(camera):
    return CaptureRequest(camera, RenderPath.RASTER, tuple(Channel))


def _host_frame(snapshot, rows, binding, time_s, nodes):
    return SceneFrame(binding.frame_key(snapshot), time_s, tuple(
        NodePose(name, _pose(rows[i, :, 3], rows[i, :, :3].astype(float))) for name, i in nodes.items()))


def _host_capture(session, frame, mount_pose):
    result = {}
    for view, pose in (('fixed', FIXED), ('gimbal-cam', mount_pose)):
        camera = Camera(view, frame.key, pose, **LENS)
        result[view] = _planes(session.capture(frame, _request(camera)), frame.key, camera, frame.time_s)
    return result


def _must_reject(fn, reason):
    try:
        fn()
    except ValueError as exc:
        return str(exc)
    raise AssertionError(f'accepted {reason}')


def _shared_capture(session, binding, ws, snapshot, stream, safety=None, *, phase_times=None):
    result = {}
    start = perf_counter_ns()
    with session.publish_device_poses(binding, ws, snapshot, producer_stream=stream) as exposure:
        acquired = perf_counter_ns()
        if exposure.key != binding.frame_key(snapshot) or exposure.time_s != ws.accepted_frame_time(
                snapshot.frame, binding.world_key):
            raise ValueError('published exposure has an incorrect frame or time')
        fixed = Camera('fixed', exposure.key, FIXED, **LENS)
        mounted = session.mounted_camera(exposure, 'gimbal-cam', **LENS)
        cameras_ready = perf_counter_ns()
        if safety is not None:
            forged = replace(mounted, world_pose=Pose())
            safety['forged_mount'] = _must_reject(
                lambda: session.capture_device_poses(exposure, _request(forged)), 'forged camera')
        for camera in (fixed, mounted):
            shot = session.capture_device_poses(exposure, _request(camera))
            result[camera.id] = _planes(shot, exposure.key, camera, exposure.time_s)
        drawn = perf_counter_ns()
    released = perf_counter_ns()
    if phase_times is not None:
        phase_times.append((acquired-start, cameras_ready-acquired, drawn-cameras_ready, released-drawn))
    if safety is not None:
        safety['expired_exposure'] = _must_reject(
            lambda: session.capture_device_poses(exposure, _request(mounted)), 'expired exposure')
    return result, mounted


def _rotation_angle(a, b):
    qa = np.asarray(a.orientation_xyzw)
    qb = np.asarray(b.orientation_xyzw)
    return float(2 * np.arccos(np.clip(abs(qa @ qb) / (np.linalg.norm(qa) * np.linalg.norm(qb)), 0., 1.)))


def _audit_reference(mj, model, state, rows, time_s, camera_id):
    mj.mj_forward(model, state)
    actual = rows[:, :, :3]
    rerr = float(np.abs(actual - state.xmat.reshape(-1, 3, 3)).max())
    perr = float(np.abs(rows[:, :, 3] - state.xpos).max())
    if rerr > 4e-5 or perr > 4e-5 or abs(state.time - time_s) > 2e-6:
        raise ValueError(f'native world reference mismatch at {time_s}: {rerr=}, {perr=}, t={state.time}')
    pose = _pose(state.cam_xpos[camera_id], state.cam_xmat[camera_id])
    return pose, dict(body_position_max_abs_m=perr, body_rotation_max_abs=rerr, native_time=float(state.time),
                      native_camera_world_matrix=pose_matrix(pose).tolist(),
                      body_pose_sha256=sha256(rows.tobytes()).hexdigest())


def _reject_candidate(batch):
    """Veto a real GPU-produced trial at the documented pre-commit test seam."""
    from phymeo.training.resident_stepper import PhysicsStepFailure

    revision, receipt = batch.checkpoint_revision, batch.last_step
    prepares = batch.counters['bucket_physics_prepares']
    def veto(operation, pending):
        if operation != 'step' or pending.revision_before != revision:
            raise AssertionError('unexpected trial at the veto point')
        raise RuntimeError('proof veto after the physical candidate was prepared')
    with patch.object(batch, '_before_commit', side_effect=veto):
        try:
            batch.step(H)
        except PhysicsStepFailure as exc:
            if (dict(exc.accepted_worlds) != dict.fromkeys(KEYS, False)
                    or str(exc.__cause__) != 'proof veto after the physical candidate was prepared'):
                raise
        else:
            raise AssertionError('vetoed trial advanced physics')
    attempts = batch.counters['bucket_physics_prepares'] - prepares
    if batch.checkpoint_revision != revision or batch.last_step is not receipt or attempts <= 0:
        raise RuntimeError('failed trial changed the accepted endpoint')
    return dict(revision=revision, bucket_prepares=attempts, accepted_worlds=dict.fromkeys(KEYS, False),
                injection='explicit pre-commit veto after real GPU step preparation')


def _fail_and_recover_observation(ws, batch, snapshot, bindings, pairs, stream):
    """Complete actual body observations, then fault the pending bank before its publication."""
    import torch

    revision = batch.checkpoint_revision
    observer = ws._body_observer
    refresh_bucket = observer.refresh_bucket
    def fail_after_observing(*args, **kwargs):
        refresh_bucket(*args, **kwargs)
        raise RuntimeError('proof veto after CUDA body observation')
    counts = ws.counters
    with patch.object(observer, 'refresh_bucket', side_effect=fail_after_observing):
        try:
            ws.refresh(expected_revision=revision)
        except RuntimeError as exc:
            if str(exc) != 'proof veto after CUDA body observation':
                raise
        else:
            raise AssertionError('failed observation published a new bank')
    if ws.counters['failed_refreshes'] != counts['failed_refreshes']+1 or batch.checkpoint_revision != revision:
        raise RuntimeError('failed observation changed the physics revision or lacked a failure receipt')
    denied = {}
    for mode, (_, session) in pairs.items():
        def publish():
            with session.publish_device_poses(bindings[KEYS[0]], ws, snapshot, producer_stream=stream):
                pass
        denied[mode] = _must_reject(publish, 'new exposure after a failed observation refresh')
        if 'successful observation refresh' not in denied[mode]:
            raise ValueError(f'wrong failed-observation rejection: {denied[mode]}')
    frame = ws.refresh(expected_revision=revision)
    with torch.cuda.stream(stream):
        recovered = ws.body_pose_snapshot(frame)
        same = bool(torch.equal(snapshot.transforms, recovered.transforms))
    if (frame.revision != revision or frame.ticks != snapshot.frame.ticks or
            frame.world_boundaries != snapshot.frame.world_boundaries or not same):
        raise RuntimeError('observer retry changed accepted frames or poses')
    return frame, recovered, dict(revision=revision, failed_snapshot_epoch=snapshot.frame.epoch,
                                 recovered_snapshot_epoch=frame.epoch, publication_rejection=denied,
                                 exact_recovered_gpu_poses=same,
                                 injection='controlled fault after actual CUDA body observer, before bank publication')


def _benchmark(args, pairs, bindings, frame, ws, snapshot, stream, nodes, mount):
    """Wall-clock per complete two-world, four-camera synchronous capture transaction."""
    samples = {mode: {'host': [], 'shared': []} for mode in pairs}
    shared_phases = {mode: [] for mode in pairs}
    after_copies = 0
    order_rows = []
    for iteration in range(args.warmup + args.repeats):
        order = list(pairs)
        start = iteration % len(order)
        order = order[start:] + order[:start]
        for mode in order:
            modeslot = list(pairs).index(mode)
            routes = ('shared', 'host') if (iteration + modeslot) % 2 else ('host', 'shared')
            for route in routes:
                t0 = perf_counter_ns()
                if route == 'host':
                    rows = snapshot.transforms.cpu().numpy()
                    for i, k in enumerate(KEYS):
                        lo, hi = snapshot.body_offsets[i:i+2]
                        host_frame = _host_frame(snapshot, rows[lo:hi], bindings[k],
                                                 ws.accepted_frame_time(frame, k), nodes)
                        parent_pose = next(p.world_pose for p in host_frame.nodes if nodes[p.node_id] == mount.body_id)
                        _host_capture(pairs[mode][0], host_frame, compose_pose(parent_pose, mount.local_pose))
                else:
                    phase_times = []
                    for k in KEYS:
                        _shared_capture(pairs[mode][1], bindings[k], ws, snapshot, stream, phase_times=phase_times)
                elapsed = (perf_counter_ns() - t0) / 1e6
                if route == 'host':
                    after_copies += rows.nbytes
                if iteration >= args.warmup:
                    samples[mode][route].append(elapsed)
                    if route == 'shared':
                        shared_phases[mode].append(np.sum(phase_times, axis=0) / 1e6)
            if iteration >= args.warmup:
                order_rows.append([iteration - args.warmup, mode, *routes])
    times = {mode: {route: dict(samples_ms=values, median_ms=float(np.median(values)),
                                p95_ms=float(np.percentile(values, 95))) for route, values in results.items()}
             for mode, results in samples.items()}
    phase_names = ('pack_publish_acquire', 'resolve_camera_metadata', 'draw_and_return_cpu_planes', 'release')
    phase_medians = {mode: dict(zip(phase_names, np.median(vals, axis=0).tolist(), strict=True))
                     for mode, vals in shared_phases.items()}
    return dict(scope='wall ms for two worlds x (fixed + mounted), four synchronous RGB/ID/depth/valid captures; '
                      'includes host D2H/body and camera composition+uploads, or shared pack/publication+camera metadata; '
                      'both include returned CPU image planes; physics, source snapshot, oracle and IO excluded',
                warmup_rounds_excluded=args.warmup, timed_rounds=args.repeats, order=order_rows, modes=times,
                shared_phase_median_ms=phase_medians,
                host_body_readback_bytes_including_warmups=after_copies)


def _counters(mode, host, shared, pubs):
    h, s = host.diagnostics(), shared.diagnostics()
    hd, sd = h['device'], s['device']
    ext = sd['resident_shared_matrices']
    if (s['msaa']['actual_color_samples'] != MODES[mode][1] or h['msaa'] != s['msaa']
            or ext['publications'] != pubs or ext['captures'] != pubs*2 or
            ext['matrix_d2d_copies'] != pubs or ext['matrix_d2d_bytes'] != pubs*ext['slot_bytes']
            or sd['resident_shared_camera_capacity'] != 1 or sd['resident_shared_camera_d2d_bytes'] != 76*pubs or
            sd['resident_host_transform_input_bytes'] != 0 or hd['resident_host_transform_input_bytes'] <= 0):
        raise ValueError(f'wrong sample count, transform transport or publication reuse for {mode}')
    rq = sd.get('resident_ray_query_info')
    if mode.startswith('RQ') and (not rq or not rq['enabled'] or rq['tlas_builds'] != 1
            or rq['tlas_updates'] != pubs-1 or rq['instance_upload_bytes'] != 0 or rq['shadow_captures'] != pubs*2):
        raise ValueError(f'{mode} did not reuse the GPU TLAS across both cameras: {rq}')
    return dict(device=sd['device_name'], msaa=s['msaa'], shared=ext,
                shared_host_transform_input_bytes=sd['resident_host_transform_input_bytes'],
                host_transform_input_bytes=hd['resident_host_transform_input_bytes'],
                shared_camera={k: v for k, v in sd.items() if k.startswith('resident_shared_camera')},
                ray_query=dict(host=hd.get('resident_ray_query_info'), shared=rq),
                shader_sha256=s.get('shader_sha256'), validation_errors=sd['validation_errors'])


def run(args):
    import mujoco as mj
    import torch
    from phymeo.backends.mujoco_heterogeneous import (
        MuJoCoTyxHeterogeneousBatch, MuJoCoTyxModelSpec, MuJoCoTyxWorldSpec)
    from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    modes = [s.strip().upper() for s in args.modes.split(',')]
    if not set(modes) <= MODES.keys() or len(set(modes)) != len(modes):
        raise ValueError('provide a comma-separated list of distinct SW1, SW4, RQ1 or RQ4 modes')
    if args.warmup < 1 or args.repeats < 2:
        raise ValueError('at least one warmup and two timed rounds required')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('choose an empty directory for this capture')
    meshes = args.output / 'meshes'
    meshes.mkdir()
    spec = MuJoCoTyxModelSpec('gimbal', compile_mujoco_xml(XML, require_asset=False))
    model = spec.compilation.fresh_native_model()
    scene, nodes = _scene(model, meshes)
    scene = replace(scene, key='native-mounted-camera-' + sha256(XML.encode()).hexdigest()[:16])
    receiver, block, cap = (int(model.geom(name).id)+1 for name in ('receiver', 'block', 'cap'))
    camera_id = int(model.camera('gimbal-cam').id)
    w, x, y, z = map(float, model.cam_quat[camera_id])
    mount = DeviceCameraMount('gimbal-cam', int(model.cam_bodyid[camera_id]),
                              Pose(tuple(model.cam_pos[camera_id]), (x, y, z, w)))
    common = dict(library_path=args.library, shader_directory=args.shader_directory or args.library.parent,
                  device_index=args.device_index, validation=args.validation, resident=True,
                  display_transform='clip', exposure=1.)
    worlds = tuple(MuJoCoTyxWorldSpec(k, spec.key) for k in KEYS)
    native = {k: mj.MjData(model) for k in KEYS}
    rows_report, exposures, guards = [], [], {mode: {} for mode in modes}
    images, seen_sensors, camera_paths, refs, counters, pairs = {}, {}, {}, {}, {}, {}
    audit_bytes = returned_bytes = 0
    with (torch.cuda.device(args.device),
          MuJoCoTyxHeterogeneousBatch((spec,), worlds, mode='cuda') as batch,
          batch.prepare_device_workspace(include_bodies=True) as ws, ExitStack() as stack):
        if ws.binding['device'] != args.device:
            raise ValueError('CUDA workspace and raster transport device differ')
        stream = torch.cuda.Stream(device=args.device)
        bindings = {k: DeviceVisualBinding(scene, ws, k, nodes, camera_mounts=(mount,)) for k in KEYS}
        for mode in modes:
            shadow, samples = MODES[mode]
            options = dict(**common, shadow_mode=shadow, msaa_samples=samples)
            host = NativeBackend(**options).open_scene(scene)
            stack.callback(host.close)
            shared = NativeBackend(**options, shared_cuda_device=args.device, shared_camera_capacity=1).open_scene(scene)
            stack.callback(shared.close)
            pairs[mode] = host, shared
        batch.set_controls({k: MuJoCoControl(command) for k, command in zip(KEYS, COMMANDS, strict=True)})
        previous = None
        for stage_idx, stage in enumerate(STAGES):
            if stage_idx in (1, 2):
                for _ in range(40):
                    if batch.step(H) != dict.fromkeys(KEYS, True):
                        raise ValueError('physics failed on a nominal step')
                    for k, command in zip(KEYS, COMMANDS, strict=True):
                        native[k].ctrl[:] = command
                        mj.mj_step(model, native[k])
            elif stage == 'reset41':
                k = KEYS[0]
                q = (-.032, -.09)
                batch.reset_states({k: MuJoCoInitialState(q, (0., 0.), time=0.)},
                                   expected_revision=batch.checkpoint_revision)
                native[k] = mj.MjData(model)
                native[k].qpos[:] = q
                native[k].time = 0.
            if previous is not None:
                # Check immediately after the physics transition, before a new observation supersedes it.
                def publish_old():
                    with pairs[modes[0]][1].publish_device_poses(bindings[KEYS[0]], ws, previous,
                                                                 producer_stream=stream):
                        pass
                reason = _must_reject(publish_old, 'stale snapshot')
                if 'stale heterogeneous frame' not in reason:
                    raise ValueError(f'wrong stale-endpoint guard: {reason}')
                guards[modes[0]][f'stale_snapshot_before_{stage}'] = reason
            frame = ws.refresh(expected_revision=batch.checkpoint_revision)
            expected_ticks = (0, 80) if stage == 'reset41' else (40*stage_idx, 40*stage_idx)
            if (frame.ticks != expected_ticks or frame.revision != batch.checkpoint_revision):
                raise ValueError('wrong physical tick or revision')
            with torch.cuda.stream(stream):
                snapshot = ws.body_pose_snapshot(frame)
            if snapshot.world_keys != KEYS or snapshot.frame is not frame:
                raise ValueError('snapshot does not identify the accepted physics worlds')
            if stage == 'step40':
                clocks = tuple(ws.accepted_frame_time(frame, k) for k in KEYS)
                veto = _reject_candidate(batch)
                if tuple(ws.accepted_frame_time(frame, k) for k in KEYS) != clocks:
                    raise ValueError('rejection changed the accepted clock')
                frame, snapshot, observation_retry = _fail_and_recover_observation(
                    ws, batch, snapshot, bindings, pairs, stream)
            previous = snapshot
            poses = snapshot.transforms.cpu().numpy()
            audit_bytes += poses.nbytes
            for i, k in enumerate(KEYS):
                lo, hi = snapshot.body_offsets[i:i+2]
                time_s = ws.accepted_frame_time(frame, k)
                native_cam, truth = _audit_reference(mj, model, native[k], poses[lo:hi], time_s, camera_id)
                host_frame = _host_frame(snapshot, poses[lo:hi], bindings[k], time_s, nodes)
                key = host_frame.key
                if (key.world_id != str(k) or key.tick != expected_ticks[i] or key.phase != 'accepted_endpoint'
                        or key.epoch != int(stage == 'reset41' and i == 0) or key.revision != frame.revision):
                    raise ValueError('camera frame does not identify the exact committed world endpoint')
                rows_report.append(dict(stage=stage, frame_key=asdict(key), time_s=time_s, **truth))
                if stage == 'step80' and i == 0:
                    fixed = Camera('fixed', key, FIXED, **LENS)
                    for family in dict.fromkeys(m[:2] for m in modes):
                        # A separate single-sample session keeps the target resize out of both timed routes.
                        oracle = NativeBackend(**common, shadow_mode=MODES[family+'1'][0],
                                               msaa_samples=1).open_scene(scene)
                        try:
                            refs[family] = reference(oracle, host_frame, fixed, args.supersample, (block, cap))
                        finally:
                            oracle.close()
                for mode, (host, shared) in pairs.items():
                    # Host baseline resolves its camera independently via native MuJoCo cam_xpos/cam_xmat.
                    sh, actual_camera = _shared_capture(shared, bindings[k], ws, snapshot, stream,
                                                        guards[mode] if stage_idx == i == 0 else None)
                    old = _host_capture(host, host_frame, native_cam)
                    pos_error = float(np.linalg.norm(np.subtract(actual_camera.world_pose.position_m,
                                                                 native_cam.position_m)))
                    rot_error = _rotation_angle(actual_camera.world_pose, native_cam)
                    if pos_error > 5e-5 or rot_error > 2e-4:
                        raise RuntimeError(f'wrong world camera at {mode}/{stage}/{k}: {pos_error=}, {rot_error=}')
                    camera_paths[mode, stage, k] = actual_camera.world_pose
                    for view in sh:
                        metrics, planes = _compare(old[view], sh[view], block)
                        hashes = metrics['plane_sha256']
                        sensor = tuple(hashes[c.value] for c in (Channel.INSTANCE_ID, Channel.VALID,
                                                                 Channel.FORWARD_DEPTH))
                        source = stage, k, view
                        if seen_sensors.setdefault(source, sensor) != sensor:
                            raise ValueError(f'center sensors changed across raster/shadow modes: {source}')
                        returned_bytes += sum(map(len, old[view].values())) + sum(map(len, sh[view].values()))
                        images[mode, stage, k, view] = planes
                        exposures.append(dict(mode=mode, stage=stage, world=k, view=view, time_s=time_s,
                                              frame_key=asdict(key), mount_translation_error_m=pos_error,
                                              mount_rotation_error_rad=rot_error, **metrics))
            if stage == 'step80':
                timing = _benchmark(args, pairs, bindings, frame, ws, snapshot, stream, nodes, mount)
        pubs = len(KEYS) * (len(STAGES) + args.warmup + args.repeats)
        for mode, sessions in pairs.items():
            counters[mode] = _counters(mode, *sessions, pubs)
        captures = batch.counters
        if (captures['completed_steps'], captures['accepted_world_steps'], captures['failed_batches']) != (80, 160, 1):
            raise ValueError('physics counters disagree with accepted steps and controlled rejection')
        workspace = dict(ws.counters)
        transfer = dict(verification_body_pose_readback_bytes=audit_bytes,
                        verification_image_readback_bytes_host_and_shared=returned_bytes,
                        timing_host_body_readback_bytes=timing['host_body_readback_bytes_including_warmups'],
                        binding_static_h2d_bytes=sum(b.layout_upload_bytes for b in bindings.values()),
                        pack_status_d2h_bytes=sum(b.status_readback_bytes for b in bindings.values()),
                        public_camera_metadata_d2h_bytes=sum(b.camera_metadata_readback_bytes for b in bindings.values()),
                        visual_and_camera_pack_bytes=sum(b.packed_output_bytes for b in bindings.values()))
        if (sum(b.packs for b in bindings.values()) != pubs * len(modes) or
                transfer['public_camera_metadata_d2h_bytes'] != pubs*len(modes)*48):
            raise ValueError('unexpected camera metadata downloads or redundant pose packs')

    last = pairs[modes[0]][0].diagnostics()['device']
    cleanup = {k: last[k] for k in ('live_resources', 'live_contexts', 'live_frames', 'validation_errors')}
    if any(cleanup.values()):
        raise ValueError(f'native cleanup or validation failed: {cleanup}')
    for mode, (_, shared) in pairs.items():
        closed = shared.diagnostics()['device']['resident_shared_matrices']
        if not closed['closed'] or closed['imported_cuda_resources'] or closed['exports_open']:
            raise ValueError(f'{mode}: shared CUDA resources remain open')
    shadow_motion, contours, panels = {}, {}, {}
    for mode in modes:
        for i, k in enumerate(KEYS):
            a, b = (images[mode, stage, k, 'fixed'] for stage in ('initial', 'step80'))
            proof = _shadow_change(a, b, receiver)
            if not (proof['brightened_18'] or proof['darkened_18']):
                raise ValueError(f'{mode}/{k} did not show moving occlusion on the fixed receiver')
            start, end = (camera_paths[mode, stage, k] for stage in ('initial', 'step80'))
            angle = _rotation_angle(start, end)
            distance = float(np.linalg.norm(np.subtract(start.position_m, end.position_m)))
            if angle < .01 or distance < .002:
                raise ValueError('mounted camera must rotate and translate in both worlds')
            shadow_motion[f'{mode}/{k}'] = dict(**proof, mount_motion_m=distance, mount_rotation_rad=angle)
            for view in ('fixed', 'gimbal-cam'):
                before, reset = (images[mode, stage, k, view] for stage in ('step80', 'reset41'))
                same = all(np.array_equal(u, v) for u, v in zip(before, reset, strict=True))
                if same != (i == 1):
                    raise ValueError(f'{mode}: incorrect world isolation for {k}/{view} after reset')
        fixed, mounted = (images[mode, 'step80', KEYS[0], view][0] for view in ('fixed', 'gimbal-cam'))
        if np.array_equal(fixed, mounted):
            raise ValueError('fixed camera and mounted camera returned identical images')
        panels[mode] = fixed, mounted
        contours[mode] = contour_error(fixed, refs[mode[:2]])
    for family in refs:
        one, four = family+'1', family+'4'
        if one in contours and four in contours:
            ratio = contours[four]['linear_mae'] / max(1e-12, contours[one]['linear_mae'])
            contours[four]['linear_mae_ratio_to_matching_1x'] = ratio
            if ratio >= 1.:
                raise ValueError(f'{four} did not reduce matched-reference contour MAE ({ratio=})')
    crop = write_board(args.output / 'mounted-camera-quality.png', panels, refs, 'step 80', KEYS[0])
    witness_mode = next(m for m in ('RQ4', 'SW4', 'RQ1', 'SW1') if m in modes)
    write_motion_board(args.output / 'moving-shadow.png', witness_mode, images, shadow_motion, KEYS)
    comparison = {mode: dict(verified_view_pairs=sum(r['mode'] == mode for r in exposures),
                             native_color_samples=counters[mode]['msaa']['actual_color_samples'],
                             contour_display_linear_mae=contours[mode]['linear_mae'],
                             rgb_max_codes_vs_host=max(r['rgb_max_codes'] for r in exposures if r['mode'] == mode),
                             **{route + '_' + stat: timing['modes'][mode][route][stat]
                                for route in ('host', 'shared') for stat in ('median_ms', 'p95_ms')})
                  for mode in modes}
    report = dict(scene=scene.key, native_mjcf=XML, controls=dict(zip(KEYS, COMMANDS, strict=True)),
                  device=last['device_name'], device_index=args.device_index, cuda_ordinal=args.device,
                  library=str(args.library.resolve()), validation_enabled=args.validation, cleanup=cleanup,
                  width=SIZE[0], height=SIZE[1], exposure=1., display='clip + IEC sRGB',
                  native_reference='MuJoCo CPU integration and independent cam_xpos / cam_xmat, fixed camera mode',
                  modes=modes, comparison=comparison, frames=rows_report, captures=exposures,
                  moving_shadow_and_camera=shadow_motion, shadow_sheet_mode=witness_mode,
                  physical_veto=veto, observation_retry=observation_retry, rejected_publications=guards,
                  accepted_steps=80, accepted_world_steps=160,
                  world_reset_scope=KEYS[0], center_sensor_hashes_identical_across_modes=True,
                  contour_reference=dict(scale=args.supersample, world=KEYS[0], tick=80, camera='fixed',
                      domain='single-sample native host RGB8 decoded to display-linear then area averaged; '
                             'mask from mixed supersample instance IDs including moving geometry',
                      crop_xyxy=crop, errors=contours), timings=timing, native_counters=counters,
                  workspace=workspace, measured_bytes=transfer)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Verified {len(exposures)} shared/host view pairs in {len(modes)} mode(s): {args.output}')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--library', required=True, type=Path)
    cli.add_argument('--shader-directory', type=Path)
    cli.add_argument('--device-index', type=int, default=0)
    cli.add_argument('--device', type=int, default=0)
    cli.add_argument('--validation', action='store_true')
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--modes', default=','.join(MODES))
    cli.add_argument('--warmup', type=int, default=2)
    cli.add_argument('--repeats', type=int, default=6)
    cli.add_argument('--supersample', type=int, choices=(2, 4), default=4)
    run(cli.parse_args())


if __name__ == '__main__':
    main()
