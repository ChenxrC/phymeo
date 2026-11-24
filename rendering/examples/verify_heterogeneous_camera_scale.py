"""Audit eight live manipulation models and sixteen camera views against independently run worlds."""
import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from _device_sensor_workload import PinnedUploader, audit_planes
from _heterogeneous_camera_worlds import FIXED_POSE, WORLD_KEYS, open_worlds
from _panda_scene import pose_matrix
from _scale_camera_sensors import SceneSensorConsumer, capture_worlds
from verify_shared_device_images import telemetry
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.outputs import Channel


SIZE = (320, 240)
STAGES = ('initial', 'middle', 'end', 'reset_first')
LIMITS = dict(position_m=4e-5, rotation_coefficient=4e-5, integration_coordinate_abs=4e-5,
              velocity_abs=4e-5, forward_depth_m=2e-5, max_rgb8_codes=1)


@contextmanager
def open_scale_sessions(world, *, library, shader_directory=None, device_index=0, device=0,
                        validation=False, shadow_mode='ray_query', msaa_samples=4):
    """One resident shared-image session per true world scene, shared by both delivery routes."""
    with ExitStack() as stack:
        sessions = {}
        for key in world.keys:
            mounts = world.bindings[key].camera_mounts
            session = NativeBackend(library_path=library, shader_directory=shader_directory or Path(library).parent,
                                    device_index=device_index, validation=validation, shared_cuda_device=device,
                                    shared_camera_capacity=len(mounts), resident=True, shadow_mode=shadow_mode,
                                    msaa_samples=msaa_samples, exposure=1., display_transform='clip').open_scene(world.scenes[key])
            stack.callback(session.close)
            sessions[key] = session
            if session.diagnostics()['msaa']['actual_color_samples'] != msaa_samples:
                raise RuntimeError('realized color sample count differs from requested samples')
        yield sessions


@dataclass
class State:
    poses: np.ndarray
    q: np.ndarray
    v: np.ndarray
    ctrl: np.ndarray
    epoch: int
    tick: int
    time_s: float
    revision: int


def _source(world, snapshot):
    transforms = snapshot.transforms.detach().cpu().numpy()
    native = world.batch.integration_snapshot()
    states = {}
    if tuple(snapshot.world_keys) != tuple(world.keys):
        raise ValueError('snapshot reordered the declared physics worlds')
    for i, k in enumerate(world.keys):
        start, stop = snapshot.body_offsets[i:i+2]
        frame = world.bindings[k].frame_key(snapshot)
        time_s = world.workspace.accepted_frame_time(snapshot.frame, k)
        data = native[k]
        if ((stop-start) != world.models[k].nbody or data.tick != frame.tick or data.time != time_s
                or frame.revision != world.batch.checkpoint_revision or frame.world_id != str(k)
                or frame.phase != 'accepted_endpoint' or frame.scene_key != world.scenes[k].key):
            raise ValueError('camera source disagrees with committed integration state')
        states[k] = State(transforms[start:stop], data.q, data.v, data.ctrl, frame.epoch, frame.tick,
                          time_s, frame.revision)
    return states


def _maxdiff(a, b):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite or incompatible numeric reference')
    return float(np.max(np.abs(a.astype(float)-b.astype(float)), initial=0.))


def _physics_error(a, b):
    if (a.epoch, a.tick, a.time_s) != (b.epoch, b.tick, b.time_s) or not np.array_equal(a.ctrl, b.ctrl):
        raise ValueError('different simulation time, epoch, tick or motor control from independent physics')
    err = dict(position_m=_maxdiff(a.poses[:, :, 3], b.poses[:, :, 3]),
               rotation_coefficient=_maxdiff(a.poses[:, :, :3], b.poses[:, :, :3]),
               integration_coordinate_abs=_maxdiff(a.q, b.q), velocity_abs=_maxdiff(a.v, b.v))
    if any(value > LIMITS[k] for k, value in err.items()):
        raise ValueError(f'accepted physics diverged from independent world: {err}')
    return dict(**err, batch_revision=a.revision, independent_revision=b.revision)


def _state_receipt(state):
    return dict(epoch=state.epoch, tick=state.tick, time_s=state.time_s, revision=state.revision,
                q=state.q.tolist(), v=state.v.tolist(), ctrl=state.ctrl.tolist(),
                body_count=len(state.poses), body_pose_sha256=sha256(state.poses.tobytes()).hexdigest())


def _camera_truth(shots, state, binding):
    by_id = {m.id: m for m in binding.camera_mounts}
    if len(shots) != 2 or len({s.camera.id for s in shots}) != 2:
        raise ValueError('each world requires two distinct real camera views')
    mounted = [s for s in shots if s.camera.id in by_id]
    if len(mounted) != 1:
        raise ValueError('expected one body-mounted and one fixed camera')
    for s in shots:
        key = s.frame_key
        if ((key.scene_key, key.world_id, key.epoch, key.tick, key.revision, key.phase, s.time_s) !=
                (binding.scene.key, str(binding.world_key), state.epoch, state.tick, state.revision,
                 'accepted_endpoint', state.time_s) or s.camera.frame_key != key or
                (s.camera.width, s.camera.height) != SIZE):
            raise ValueError('captured exposure has a different camera extent or physical identity')
        if s.camera.id not in by_id and s.camera.world_pose != FIXED_POSE:
            raise ValueError('world-fixed camera moved away from its declared calibration')
    camera = mounted[0].camera
    local = by_id[camera.id]
    parent = np.eye(4)
    parent[:3] = state.poses[local.body_id]
    expected = parent @ pose_matrix(local.local_pose)
    actual = pose_matrix(camera.world_pose)
    pos = _maxdiff(actual[:3, 3], expected[:3, 3])
    rot = _maxdiff(actual[:3, :3], expected[:3, :3])
    if pos > LIMITS['position_m'] or rot > LIMITS['rotation_coefficient']:
        raise ValueError(f'published mounted camera disagrees with native body and local mount: {pos=}, {rot=}')
    return dict(mount_id=camera.id, native_body_id=local.body_id, position_max_abs_m=pos, rotation_max_abs=rot)


def _independent_image(batch, solo):
    a, b = batch.camera, solo.camera
    lens = ('id', 'width', 'height', 'fx_px', 'fy_px', 'cx_px', 'cy_px', 'near_m', 'far_m')
    if (any(getattr(a, n) != getattr(b, n) for n in lens) or batch.time_s != solo.time_s
            or batch.frame_key.scene_key != solo.frame_key.scene_key):
        raise ValueError('independent camera does not match lens, scene or time')
    ma, mb = pose_matrix(a.world_pose), pose_matrix(b.world_pose)
    cerr = _maxdiff(ma[:3, :3], mb[:3, :3])
    perr = _maxdiff(ma[:3, 3], mb[:3, 3])
    left, right = ({p.channel: p.data for p in c.planes} for c in (batch, solo))
    if set(left) != set(Channel) or set(right) != set(Channel):
        raise ValueError('full RGB/forward depth/ID/valid required')
    for ch in (Channel.INSTANCE_ID, Channel.VALID):
        if left[ch] != right[ch]:
            dtype = '<u4' if ch == Channel.INSTANCE_ID else 'u1'
            count = np.count_nonzero(np.frombuffer(left[ch], dtype) != np.frombuffer(right[ch], dtype))
            raise ValueError(f'{a.id}: {int(count)} inconsistent independent {ch.value} pixels')
    ids = np.frombuffer(left[Channel.INSTANCE_ID], '<u4')
    valid = np.frombuffer(left[Channel.VALID], np.uint8)
    hit = valid == 1
    if not hit.any() or (valid > 1).any() or not np.array_equal(hit, ids != 0):
        raise ValueError('invalid center-sample mask or empty view')
    za, zb = (np.frombuffer(t[Channel.FORWARD_DEPTH], '<f4') for t in (left, right))
    for z in (za, zb):
        if not np.isposinf(z[~hit]).all() or not (z[hit] > 0).all():
            raise ValueError('invalid hit depth or background representation')
    depth = _maxdiff(za[hit], zb[hit])
    ra, rb = (np.frombuffer(t[Channel.RGB], np.uint8).reshape(-1, 3) for t in (left, right))
    diff = np.abs(ra.astype(np.int16)-rb)
    largest = int(diff.max())
    if (cerr > LIMITS['rotation_coefficient'] or perr > LIMITS['position_m'] or
            depth > LIMITS['forward_depth_m'] or largest > LIMITS['max_rgb8_codes']):
        raise ValueError(f'{a.id}: independent image/camera exceeds tolerance: {cerr=}, {perr=}, {depth=}, RGB={largest}')
    return dict(exact_instance_id=True, exact_visibility=True, hit_pixels=int(hit.sum()),
                positive_inf_background=int((~hit).sum()), forward_depth_max_abs_m=depth, rgb_max_codes=largest,
                rgb_mae_codes=float(diff.mean()), rgb_changed_pixels=int(np.any(diff != 0, axis=1).sum()),
                camera_position_max_abs_m=perr, camera_rotation_max_abs=cerr)


def _hashes(shot):
    return {p.channel.value: sha256(p.data).hexdigest() for p in shot.planes}


def _stage(world, phase, middle, end, first):
    if phase == 'middle':
        world.step(middle)
    elif phase == 'end':
        world.step(end-middle)
    elif phase == 'reset_first' and first in world.keys:
        world.reset((first,))
    snapshot = world.snapshot()
    return _source(world, snapshot), world.pack(snapshot)


def _board(path, keys, gathered, names):
    # All tiles show the same accepted endpoint; no interpolation or independent exposures.
    tile, gap, caption = (240, 180), 5, 21
    columns = min(4, len(keys))
    rows = 2*((len(keys)+columns-1)//columns)
    board = Image.new('RGB', (columns*(tile[0]+gap)+gap, rows*(tile[1]+caption+gap)+gap), '#14232a')
    draw = ImageDraw.Draw(board)
    for n, key in enumerate(keys):
        for v, shot in enumerate(gathered['end', key][1]):
            row = (n//columns)*2 + v
            x, y = gap + (n%columns)*(tile[0]+gap), gap + row*(tile[1]+caption+gap)
            rgb = next(p.data for p in shot.planes if p.channel == Channel.RGB)
            image = Image.frombytes('RGB', SIZE, rgb).resize(tile, Image.Resampling.LANCZOS)
            draw.text((x, y+2), f'{key}  {names[key]}  |  {shot.camera.id}', fill='white')
            board.paste(image, (x, y+caption))
    board.save(path)


def _native_fk(model, state):
    """Independent MuJoCo endpoint forward evaluation at the accepted generalized coordinates."""
    import mujoco as mj

    data = mj.MjData(model)
    data.qpos[:] = state.q
    data.qvel[:] = state.v
    mj.mj_forward(model, data)
    p = _maxdiff(state.poses[:, :, 3], data.xpos)
    r = _maxdiff(state.poses[:, :, :3], data.xmat.reshape(-1, 3, 3))
    if p > LIMITS['position_m'] or r > LIMITS['rotation_coefficient']:
        raise ValueError(f'GPU accepted body transforms differ from native endpoint FK: {p=}, {r=}')
    return dict(position_m=p, rotation_coefficient=r)


def _reset_receipt(gathered, keys):
    outcomes = []
    for i, key in enumerate(keys):
        first, a = gathered['initial', key]
        before, b = gathered['end', key]
        reset, c = gathered['reset_first', key]
        if reset.revision <= before.revision or reset.epoch != before.epoch + (i == 0):
            raise ValueError(f'wrong world boundary handling for {key}')
        expected_state, shots = (first, a) if i == 0 else (before, b)
        for name in ('q', 'v', 'poses'):
            if not np.array_equal(getattr(expected_state, name), getattr(reset, name)):
                raise ValueError(f'{key}/{name}: reset changed the wrong physical state')
        if (reset.tick, reset.time_s) != (expected_state.tick, expected_state.time_s):
            raise ValueError('reset changed the wrong world clock')
        if not all(left.planes == right.planes and left.camera.world_pose == right.camera.world_pose
                   for left, right in zip(shots, c, strict=True)):
            raise ValueError('published images or camera matrices did not isolate the selected reset')
        changed = [left.camera.id for left, right in zip(a, b, strict=True) if left.planes != right.planes]
        if np.array_equal(first.q, before.q) or not changed:
            raise ValueError(f'no physical or visible motion through the sampled trajectory in {key}')
        outcomes.append(dict(world=key, expected_image_source='initial' if i == 0 else 'end',
                             pixels_and_pose_exact=True, changed_cameras_during_motion=changed,
                             tick_before=before.tick, tick_after=reset.tick,
                             epoch_before=before.epoch, epoch_after=reset.epoch))
    return outcomes


def run(args):
    if not 0 < args.middle_step < args.end_step:
        raise ValueError('middle and end must be distinct positive physical steps')
    keys = WORLD_KEYS if args.keys is None else tuple(int(item) for item in args.keys.split(','))
    if not keys or len(set(keys)) != len(keys) or any(key not in WORLD_KEYS for key in keys):
        raise ValueError('provide distinct declared world keys in capture order')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('choose an empty output directory')
    assets = args.asset_dir or args.output/'meshes'
    settings = dict(library=args.library, shader_directory=args.shader_directory, device=args.device,
                    device_index=args.device_index, validation=args.validation, shadow_mode=args.shadow_mode,
                    msaa_samples=args.samples)
    device = torch.device('cuda', args.device)
    gathered, exposure_rows, independent, state_rows, totals = {}, [], [], [], {}
    raw_d2h_bytes = audit_pose_bytes = feature_d2h_bytes = 0
    with open_worlds(assets, keys, device=args.device) as full, open_scale_sessions(full, **settings) as sessions:
        manifest = full.manifest
        names = {row['world_key']: row['model_key'] for row in manifest['worlds']}
        if manifest['physical_configuration_count'] != len(keys) or manifest['native_bucket_count'] != len(keys):
            raise ValueError('the selected worlds did not retain distinct physical models and buckets')
        delivery, compute = (torch.cuda.Stream(device=device) for _ in range(2))
        sensor = SceneSensorConsumer(full.scenes, keys, device)
        uploader = PinnedUploader(device)
        for stage in STAGES:
            states, packed = _stage(full, stage, args.middle_step, args.end_step, keys[0])
            audit_pose_bytes += sum(state.poses.nbytes for state in states.values())
            args_for_capture = (sessions, full.workspace, packed, keys, full.requests, delivery)
            cpu = capture_worlds('cpu_upload', *args_for_capture)
            group = capture_worlds('direct_cuda', *args_for_capture)
            with torch.cuda.stream(compute):
                shots = group.wait(compute)
                hashes, payload = audit_planes(shots, cpu, device)
                cuda_result = sensor([s.planes for s in shots]).cpu().numpy()
                uploaded_result = sensor(uploader.upload(cpu)).cpu().numpy()
            raw_d2h_bytes += payload
            feature_d2h_bytes += cuda_result.nbytes + uploaded_result.nbytes
            oracle = sensor.numpy_oracle(cpu)
            np.testing.assert_array_equal(cuda_result, oracle)
            np.testing.assert_array_equal(uploaded_result, oracle)
            if (oracle.shape != (2*len(keys), len(sensor.fields)) or np.any(oracle[:, -1]) or
                    np.any(oracle[:, 3] <= 0) or np.any(oracle[:, 3] + oracle[:, 4] != SIZE[0]*SIZE[1])):
                raise ValueError('the downstream GPU consumer did not account for every center sample')
            for i, key in enumerate(keys):
                start = 2*i
                state = states[key]
                stage_tick = dict(initial=0, middle=args.middle_step, end=args.end_step,
                                  reset_first=0 if key == keys[0] else args.end_step)[stage]
                if state.tick != stage_tick:
                    raise ValueError(f'{stage}/{key}: wrong committed physical tick {state.tick}')
                pair = cpu[start:start+2]
                mount = _camera_truth(pair, state, full.bindings[key])
                native = _native_fk(full.models[key], state)
                gathered[stage, key] = state, pair
                for j, shot in enumerate(pair, start):
                    exposure_rows.append(dict(stage=stage, world=key, camera=asdict(shot.camera), time_s=shot.time_s,
                                              exact_cpu_device_channels=True, plane_sha256=hashes[j],
                                              consumer_values=cuda_result[j].tolist(), native_body_fk=native,
                                              mount_reference=mount))
        counts = full.batch.counters
        if counts['completed_steps'] != args.end_step or counts['accepted_world_steps'] != args.end_step*len(keys):
            raise ValueError('batched physical step count disagrees with sampled trajectory')
        if full.batch_binding.packs != len(STAGES) or full.batch_binding.readback_calls != len(STAGES):
            raise ValueError('more than one multiworld pack was used per accepted endpoint')
        for key, session in sessions.items():
            d = telemetry(session)
            diag = session.diagnostics()
            ray = diag['device'].get('resident_ray_query_info')
            n = len(STAGES)
            if (d['readback_bytes'] != n*2*SIZE[0]*SIZE[1]*13 or d['output_image_bytes'] != d['readback_bytes']
                    or d['output_batches'] != n or d['output_views'] != 2*n or d['image_d2h_bytes']
                    or d['image_h2d_bytes'] or d['pose_publications'] != 2*n or d['host_transform_input_bytes']):
                raise ValueError(f'{key}: route counters disagree with two cameras per publication: {d}')
            if args.shadow_mode == 'ray_query' and (not ray or not ray['enabled'] or ray['tlas_builds'] != 1
                    or ray['tlas_updates'] != 2*n-1 or ray['shadow_captures'] != 4*n or ray['instance_upload_bytes']):
                raise ValueError(f'{key}: hardware ray queries did not use one GPU TLAS per publication: {ray}')
            totals[key] = dict(**d, msaa=diag['msaa'], ray_query=ray)
    reset = _reset_receipt(gathered, keys)

    singleton_steps = {}
    for key in keys:
        # A new TYX batch, workspace, visual binding and native session initialize and evolve this world alone.
        with open_worlds(assets, (key,), device=args.device) as solo, open_scale_sessions(solo, **settings) as single_session:
            original = next(r for r in manifest['worlds'] if r['world_key'] == key)
            local = solo.manifest['worlds'][0]
            for field in ('model_key', 'physical_sha256', 'source_sha256', 'scene_key', 'geometry'):
                if original[field] != local[field]:
                    raise ValueError(f'{key}: independent fixture differs in {field}')
            for stage in STAGES:
                states, packed = _stage(solo, stage, args.middle_step, args.end_step, keys[0])
                state = states[key]
                audit_pose_bytes += state.poses.nbytes
                reference = capture_worlds('cpu_upload', single_session, solo.workspace, packed, (key,), solo.requests,
                                           solo.stream)
                _camera_truth(reference, state, solo.bindings[key])
                collective, pair = gathered[stage, key]
                physics = _physics_error(collective, state)
                state_rows.append(dict(stage=stage, world=key, batched=_state_receipt(collective),
                                       independent=_state_receipt(state), errors=physics))
                for shot, single in zip(pair, reference, strict=True):
                    independent.append(dict(stage=stage, world=key, camera=shot.camera.id, time_s=single.time_s,
                                            frame_key=asdict(shot.frame_key), solo_frame_key=asdict(single.frame_key),
                                            solo_plane_sha256=_hashes(single),
                                            **_independent_image(shot, single)))
            steps = solo.batch.counters
            singleton_steps[key] = int(steps['accepted_world_steps'])
            if singleton_steps[key] != args.end_step or steps['completed_steps'] != args.end_step:
                raise ValueError(f'{key}: independent run did not complete the same number of physics steps')
    final = single_session[keys[-1]].diagnostics()['device']
    cleanup = {k: final[k] for k in ('live_resources', 'live_contexts', 'live_frames', 'validation_errors')}
    if any(cleanup.values()):
        raise RuntimeError(f'rendering validation or cleanup failed: {cleanup}')
    for key, session in sessions.items():
        d = session.diagnostics()['device']
        output, poses = d['resident_device_output'], d['resident_shared_matrices']
        if output['imports_live'] or output['exports_open'] or output['return_pending'] or output['phase']:
            raise RuntimeError(f'{key}: native image interop resources survived session close')
        if not poses['closed'] or poses['exports_open'] or poses['imported_cuda_resources']:
            raise RuntimeError(f'{key}: pose exchange remained open')
    _board(args.output/'heterogeneous-cameras.png', keys, gathered, names)
    summary = dict(worlds=len(keys), views_per_endpoint=2*len(keys), endpoints=len(STAGES),
                   independent_view_comparisons=len(independent), exact_cpu_gpu_plane_pairs=len(exposure_rows)*len(Channel),
                   max_rgb8_codes=max(r['rgb_max_codes'] for r in independent),
                   max_rgb_mae_codes=max(r['rgb_mae_codes'] for r in independent),
                   max_depth_error_m=max(r['forward_depth_max_abs_m'] for r in independent),
                   max_independent_camera_position_error_m=max(r['camera_position_max_abs_m'] for r in independent),
                   max_independent_body_position_error_m=max(r['errors']['position_m'] for r in state_rows),
                   exact_independent_instance_id_and_valid=True)
    report = dict(summary=summary, limits=LIMITS, device=final['device_name'], manifest=manifest, size=SIZE,
                  exposure=1., display_transform='clip', validation=args.validation, cleanup=cleanup,
                  library_sha256=sha256(args.library.read_bytes()).hexdigest(), shadow_mode=args.shadow_mode,
                  msaa_samples=args.samples, stages=STAGES, selected_reset_world=keys[0], reset=reset,
                  full_world_steps=args.end_step*len(keys), independent_world_steps=singleton_steps,
                  sensor_features=SceneSensorConsumer.fields, measured_native_counters=totals,
                  audit_body_pose_payload_bytes=audit_pose_bytes, audit_raw_device_image_payload_bytes=raw_d2h_bytes,
                  audit_consumer_feature_bytes=feature_d2h_bytes, cpu_route_public_upload_bytes=uploader.h2d_bytes,
                  exposures=exposure_rows, physics_comparison=state_rows, independent_runs=independent,
                  comparison_scope='independent fresh TYX simulation, native renderer and matched camera per world; '
                                   'per-world epoch, tick and physical time are compared. Global revisions are reported '
                                   'separately when the multiworld batch resets a different world.')
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Verified {len(independent)} independent camera comparisons, {len(keys)} worlds: {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--shader-directory', type=Path)
    parser.add_argument('--asset-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--shadow-mode', choices=('ray_query', 'software'), default='ray_query')
    parser.add_argument('--samples', type=int, choices=(1, 4), default=4)
    parser.add_argument('--middle-step', type=int, default=16)
    parser.add_argument('--end-step', type=int, default=32)
    parser.add_argument('--keys', help='optional comma-separated world subset; default is all eight')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
