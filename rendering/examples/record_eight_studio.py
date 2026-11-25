"""Film eight concurrent simulation timelines in a common studio and eight close views."""
import argparse
from contextlib import ExitStack, closing
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image

from _eight_studio_motion import load_motion, record_motion
from _eight_studio_movies import StudioMovieWriter, probe_movies
from _eight_studio_set import build_eight_studio, overview_lighting
from _heterogeneous_camera_worlds import FIXED_POSE, WORLD_KEYS, _registered_visual
from _manipulation_photo_studio import STUDIO_DISPLAY, STUDIO_EXPOSURE, studio_lighting, style_manipulation_studio
from _r4_archive import _pose
from _panda_scene import look_at
from _studio_task_markers import add_task_visuals, target_ids
from phymeo.training.manipulation_tasks import manipulation_scenes
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, NodePose, SceneFrame


OVERVIEW_SIZE = (1440, 810)
CAMERA_SIZE = (640, 480)
ROUND_PART_POSE = look_at((1.32,-.22,.31), (.23,.025,.09), (0.,0.,1.))


def station_camera(frame, key, end_on, size=CAMERA_SIZE):
    """Use the same optics in the interactive viewer and the studio films."""
    w, h = size
    pose = ROUND_PART_POSE if key in end_on else FIXED_POSE
    return Camera('station', frame.key, pose, w, h, .96*w, .96*w, w/2, h/2, .012, 4.)


def renderer(args, lighting):
    return NativeBackend(library_path=args.library, shader_directory=args.library.parent,
                         device_index=args.device_index, validation=True, resident=True,
                         studio=lighting, shadow_mode='ray_query', area_light_grid=args.area_grid,
                         msaa_samples=4, exposure=STUDIO_EXPOSURE, display_transform=STUDIO_DISPLAY)


def build_scenes(motion, directory):
    scenes, nodes, names, workpieces = {}, {}, {}, {}
    tasks = {row['world']: row.get('task') for row in motion.report['worlds']}
    for key, authored in zip(WORLD_KEYS, manipulation_scenes(), strict=True):
        model = motion.models[key]
        root = directory / str(key)
        scene, bindings, _, _ = _registered_visual(authored, model, root / 'original')
        if tasks[key] and 'studio_mjcf_sha256' in tasks[key]:
            scene = replace(scene, key=scene.key+':mjcf:'+tasks[key]['studio_mjcf_sha256'])
        scene, static = style_manipulation_studio(scene, model, root / 'studio')
        if tasks[key] is not None:
            scene = add_task_visuals(scene, model, authored, tasks[key], root / 'task')
        scenes[key], nodes[key], names[key] = scene, {**bindings, **static}, authored.key
        workpieces[key] = [i.id for i in scene.instances if i.mesh_id.startswith('object-')]
    return scenes, nodes, names, workpieces


def local_frame(scene, nodes, rows, key, tick, time):
    frame_key = FrameKey(scene.key, str(key), 0, tick, tick, 'accepted_endpoint')
    poses = tuple(NodePose(name, _pose(rows[i, :, 3], rows[i, :, :3])) for name, i in nodes.items())
    return SceneFrame(frame_key, time, poses)


def capture(session, frame, camera, inspect):
    channels = (Channel.RGB, Channel.INSTANCE_ID) if inspect else (Channel.RGB,)
    shot = session.capture(frame, CaptureRequest(camera, RenderPath.RASTER, channels))
    if (shot.camera, shot.frame_key, shot.time_s) != (camera, frame.key, frame.time_s):
        raise RuntimeError('captured a different exposure')
    planes = {p.channel: p.data for p in shot.planes}
    size = camera.height, camera.width
    rgb = np.frombuffer(planes[Channel.RGB], np.uint8).reshape(*size, 3)
    ids = np.frombuffer(planes[Channel.INSTANCE_ID], '<u4').reshape(size) if inspect else None
    return rgb, ids


def visibility(large_ids, station_ids, set3d, workpieces, scenes):
    result = {}
    for key, original in workpieces.items():
        mapped = [set3d.instance_ids[key][i] for i in original]
        count = int(np.isin(large_ids, mapped).sum())
        close = int(np.isin(station_ids[key], original).sum())
        if min(count, close) == 0:
            raise RuntimeError(f'workpiece invisible at station {key}: overview={count}, local={close}')
        result[key] = dict(overview_pixels=count, independent_camera_pixels=close)
        ink = target_ids(scenes[key])
        if ink:
            wide = int(np.isin(large_ids, [set3d.instance_ids[key][i] for i in ink]).sum())
            local = int(np.isin(station_ids[key], ink).sum())
            if min(wide, local) == 0:
                raise RuntimeError(f'target not visible at station {key}')
            result[key].update(overview_target_pixels=wide, independent_target_pixels=local)
    return result


def lift_checkpoint(motion):
    """Choose a filmed moment of bilateral contact with daylight below the red ball."""
    row = next(row for row in motion.report['worlds'] if row['model_key'] == 'grasp-sphere')
    samples = row.get('interaction', {}).get('samples', ())
    held = [s for s in samples if s[4] >= .03 and min(s[5:7]) > .02 and s[7] < .02]
    if not held:
        return None
    tick = max(held, key=lambda s: s[4])[0]
    return int(np.abs(motion.ticks - tick).argmin())


def record(args, motion):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    scenes, nodes, names, objects = build_scenes(motion, args.motion / 'meshes')
    end_on = {row['world'] for row in motion.report['worlds'] if row.get('task', {}).get('worktop_contact')}
    combined = build_eight_studio(scenes, nodes, motion.models, args.motion / 'meshes' / 'overview', end_on=end_on)
    total = len(motion.ticks)
    checks = {0, total//4, total//2, 3*total//4, total-1}
    lift = lift_checkpoint(motion)
    if lift is not None:
        checks.add(lift)
    indices = sorted(checks) if args.preview else range(total)
    visible, diagnostics = {}, {}
    with ExitStack() as stack:
        hero = stack.enter_context(closing(renderer(args, overview_lighting()).open_scene(combined.scene)))
        source = renderer(args, studio_lighting())
        closeups = {key: stack.enter_context(closing(source.open_scene(scene))) for key, scene in scenes.items()}
        movies = None if args.preview else stack.enter_context(StudioMovieWriter(
            output, WORLD_KEYS, names, overview_size=OVERVIEW_SIZE, camera_size=CAMERA_SIZE, fps=args.fps,
            contact_tasks=all(row.get('task') is not None for row in motion.report['worlds'])))
        for i in indices:
            tick, time = int(motion.ticks[i]), float(motion.times[i])
            frames = {k: local_frame(scenes[k], nodes[k], motion.rows[k][i], k, tick, time) for k in WORLD_KEYS}
            full = combined.frame(frames)
            wide, wide_ids = capture(hero, full, combined.camera(full.key, *OVERVIEW_SIZE), i in checks)
            tiles, tile_ids = {}, {}
            for key, frame in frames.items():
                cam = station_camera(frame, key, end_on)
                tiles[key], tile_ids[key] = capture(closeups[key], frame, cam, i in checks)
            if i in checks:
                visible[i] = visibility(wide_ids, tile_ids, combined, objects, scenes)
                if i == lift:
                    Image.fromarray(tiles[WORLD_KEYS[0]]).save(output / 'grasp-lift.jpg', quality=95)
                if i in (0, total-1) and motion.report['format'] == 'studio-motion-v2':
                    push_key = next(k for k in WORLD_KEYS if target_ids(scenes[k]))
                    stage = 'start' if i == 0 else 'finish'
                    Image.fromarray(tiles[push_key]).save(output / f'push-{stage}.jpg', quality=95)
                if args.preview or i == total//2:
                    Image.fromarray(wide).save(output / f'overview-{i:04d}.jpg', quality=93)
                if args.preview:
                    Image.fromarray(tiles[WORLD_KEYS[0]]).save(output / f'station-{i:04d}.jpg', quality=92)
                    pk = next((k for k in WORLD_KEYS if target_ids(scenes[k])), None)
                    if pk is not None:
                        Image.fromarray(tiles[pk]).save(output / f'push-{i:04d}.jpg', quality=92)
            if movies is not None:
                movies.write(wide, tiles)
            if i % 90 == 0:
                print(f'Rendered exposure {i+1}/{total}, physical clock {time:.3f} s', flush=True)
        for key, session in [('overview', hero), *closeups.items()]:
            diag = session.diagnostics()
            dev = diag['device']
            ray = dev['resident_ray_query_info']
            if (dev['validation_errors'] or not ray['enabled'] or not dev['resident_environment_info']['loaded']
                    or diag['msaa']['actual_color_samples'] != 4 or ray['shadow_captures'] != len(indices)):
                raise RuntimeError(f'Incomplete Vulkan studio render on {key}')
            diagnostics[key] = dict(gpu=dev['device_name'], ray_shadow_frames=ray['shadow_captures'],
                                    msaa=4, samples_per_panel=diag['area_light_samples_per_panel'],
                                    triangles=diag['triangles'], validation_errors=dev['validation_errors'])
    report = dict(physics=motion.report, motion_archive_sha256=sha256((args.motion/'motion.npz').read_bytes()).hexdigest(),
                  offsets=combined.offsets, bay_orientation_xyzw={k:p.orientation_xyzw for k,p in combined.placements.items()},
                  overview_size=OVERVIEW_SIZE, closeup_size=CAMERA_SIZE,
                  visible_workpiece_samples=visible, vulkan=diagnostics,
                  lift_checkpoint_s=None if lift is None else float(motion.times[lift]))
    if not args.preview:
        report['encoded'] = probe_movies(output, WORLD_KEYS, names, fps=args.fps, frames=total)
        for file, stream in report['encoded'].items():
            if stream['max_changed_fraction_at_5_levels'] <= 0:
                raise RuntimeError(f'no visible motion in {file}')
            native_size = OVERVIEW_SIZE if file == 'overview.mp4' else CAMERA_SIZE if file.startswith('cameras/') else None
            if native_size is not None and tuple(stream['size']) != native_size:
                raise RuntimeError(f'unexpected encoded size in {file}')
    (output/'recording.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--device-index', type=int, default=0)
    p.add_argument('--area-grid', type=int, choices=(2, 4, 8), default=8)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--seconds', type=int, default=30)
    p.add_argument('--preview', action='store_true', help='render synchronised verification frames without encoding')
    args = p.parse_args()
    if (args.motion/'motion.json').exists():
        motion = load_motion(args.motion)
    else:
        motion = record_motion(args.motion, WORLD_KEYS, args.fps, args.seconds)
    if len(motion.times) != args.fps*args.seconds or int(motion.ticks[-1]) != round((len(motion.times)-1)*1000/args.fps):
        raise ValueError('motion archive does not match the requested video clock')
    record(args, motion)


if __name__ == '__main__':
    main()
