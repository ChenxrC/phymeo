"""View or record the first steps of the official Unitree RL Gym G1 walking policy."""
import argparse
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter, sleep
import webbrowser

from _unitree_walk_motion import UnitreeWalk, fetch_assets
from _unitree_walk_visual import build_unitree_visual
from phymeo_rendering import LiveSceneViewer, StudioRenderer
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath


ROOT = Path(__file__).resolve().parents[2]
VIEWS = {'follow': '前侧跟拍', 'side': '侧面跟拍', 'front': '正面跟拍', 'fixed': '固定场边'}
SIZE = (1440, 810)
FPS = 30


def camera(visual, frame, view):
    return visual.camera(frame, view=view, id=view, width=SIZE[0], height=SIZE[1])


def snapshot(walk, visual):
    return visual.frame(walk.kinematics(), tick=walk.tick)


def placement_error(walk, visual):
    error = visual.audit_transforms(walk.kinematics())
    if error['max_origin_error_m'] > 1e-6 or error['max_basis_error'] > 1e-6:
        raise ValueError(f'Visual geometry differs from the physics frame: {error}')
    return error


def record(args, walk, visual, renderer, prepared):
    from PIL import Image

    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('Recording requires ffmpeg on PATH')
    output = args.record
    output.mkdir(parents=True, exist_ok=True)
    count = round(args.duration * FPS)
    dt = float(walk.model.opt.timestep)
    movie = 'g1-walk.mp4'
    placement = {'initial': placement_error(walk, visual)}
    with TemporaryDirectory(dir=output) as directory, closing(renderer.open_scene(prepared)) as session:
        staging = Path(directory)
        cmd = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo',
               '-pixel_format', 'rgb24', '-video_size', f'{SIZE[0]}x{SIZE[1]}', '-framerate', str(FPS),
               '-i', 'pipe:0', '-an', '-vf', 'scale=out_color_matrix=bt709:out_range=tv',
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'veryfast',
               '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'iec61966-2-1',
               '-movflags', '+faststart', str(staging/movie)]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
            try:
                for i in range(count):
                    target = round(i / (FPS * dt))
                    if target > walk.tick:
                        walk.step(target-walk.tick)
                    frame = snapshot(walk, visual)
                    frame, view = prepared.bind_view(frame, camera(visual, frame, args.view))
                    shot = session.capture(frame, CaptureRequest(view, RenderPath.RASTER, (Channel.RGB,)))
                    if (shot.frame_key, shot.camera, shot.time_s) != (frame.key, view, frame.time_s):
                        raise RuntimeError('recording received a mismatched exposure')
                    pixels = shot.planes[0].data
                    if i in (0, FPS):
                        name = 'first-frame.png' if i == 0 else 'one-second.png'
                        Image.frombytes('RGB', SIZE, pixels).save(staging/name)
                    encoder.stdin.write(pixels)
            finally:
                encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('ffmpeg could not finish the G1 video')
        final_tick = round(args.duration/dt)
        if final_tick > walk.tick:
            walk.step(final_tick-walk.tick)
        if args.duration >= 6:
            walk.assert_walking()
        placement['end'] = placement_error(walk, visual)
        result = dict(motion=walk.report(), visual=visual.report, frames=count, fps=FPS,
                      video_seconds=count/FPS, size=SIZE, view=args.view,
                      geometry_adapted=prepared.report['adapted'], lighting='StudioRenderer default',
                      device=prepared.report['device'], render_profile=renderer.probe_device(), placement=placement)
        (staging/'run.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                                        encoding='utf-8')
        for file in staging.iterdir():
            file.replace(output/file.name)
    print(output/movie, flush=True)


def observe(args, walk, visual, renderer, prepared):
    def bind_camera(frame, view):
        source_key = replace(frame.key, scene_key=visual.scene.key)
        source_camera = camera(visual, replace(frame, key=source_key), view)
        return replace(source_camera, frame_key=frame.key)

    callbacks = {v: (lambda f, v=v: bind_camera(f, v)) for v in VIEWS}
    with LiveSceneViewer(prepared.scene, cameras=callbacks, camera_labels=VIEWS, renderer_factory=lambda: renderer,
                         title='Unitree G1 · 行走', fps=15, port=args.port) as monitor:
        print(monitor.url, flush=True)
        start = perf_counter()
        display_tick = 0
        while walk.time_s < args.duration:
            frame = snapshot(walk, visual)
            if frame.key.scene_key != prepared.scene.key:
                frame = replace(frame, key=replace(frame.key, scene_key=prepared.scene.key))
            monitor.publish(frame)
            if display_tick == 0 and not args.no_browser:
                webbrowser.open(monitor.url)
            display_tick += 1
            target = min(round(args.duration/walk.model.opt.timestep),
                         round(display_tick/(FPS*walk.model.opt.timestep)))
            if target <= walk.tick:
                break
            sleep(max(0., start+target*walk.model.opt.timestep-perf_counter()))
            walk.step(target-walk.tick)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=ROOT/'artifacts/unitree-walk')
    parser.add_argument('--download', action='store_true', help='fetch the pinned official model and walking policy')
    parser.add_argument('--library', type=Path, required=True, help='matching resident Vulkan library and shaders')
    parser.add_argument('--device-index', type=int, help='pin a Vulkan physical device (default: automatic)')
    parser.add_argument('--profile', choices=('auto', 'strict', 'compatible'), default='auto',
                        help='Vulkan studio profile used for both viewing and recording')
    parser.add_argument('--view', choices=VIEWS, default='follow', help='recording camera')
    parser.add_argument('--record', type=Path, help='write MP4, original stills and run.json instead of serving')
    parser.add_argument('--duration', type=float, help='simulated seconds; default 6 for recording, 30 for live')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if args.duration is None:
        args.duration = 6 if args.record else 30
    if not 1/FPS <= args.duration <= 3600:
        parser.error('duration must be between one frame and one hour')
    library = args.library.resolve(strict=True)
    renderer = StudioRenderer.for_device(library_path=library, shader_directory=library.parent,
                                         device_index=args.device_index, profile=args.profile)
    print(json.dumps(renderer.probe_device(), ensure_ascii=False), flush=True)
    if args.download:
        fetch_assets(args.assets)
    walk = UnitreeWalk(args.assets)
    visual = build_unitree_visual(walk.model, args.assets/'visual-cache', robot='g1')
    first = snapshot(walk, visual)
    prepared = renderer.prepare_scene(visual.scene, usage='film' if args.record else 'preview',
                                      cameras=[(first, camera(visual, first, v)) for v in VIEWS],
                                      cache_dir=args.assets/'prepared')
    if args.record:
        record(args, walk, visual, renderer, prepared)
    else:
        observe(args, walk, visual, renderer, prepared)


if __name__ == '__main__':
    main()
