"""Film ten local GLB assets together in a daylight dining room with native Vulkan.

Pass --assets <numbered-asset-pack> --cache <derived-mesh-directory>
--output <movie-directory> --library <native-dll> and the Vulkan --device-index.
Add --preview for stills and per-object coverage. The film uses the lowest
numbered GLB variant of each category and leaves the asset pack read-only.
"""
import argparse
from contextlib import closing, contextmanager, nullcontext
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from _home_showcase_assets import load_home_assets
from _home_showcase_room import HOME_PLACEMENTS, ROOM_EXPOSURE, ROOM_NODE, build_room, room_camera, room_lighting
from _panda_movie_encode import _verify
from phymeo_rendering.outputs import Channel
from phymeo_rendering import StudioRenderer
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, NodePose, Pose, SceneFrame, VisualScene


FPS = 30
FRAMES = 30 * FPS
CHECKS = (0, 150, 345, 459, 540, 651, 750, FRAMES - 1)
SIZE = (1440, 810)


def assemble(root, cache):
    assets = load_home_assets(root, cache / "objects")
    room = build_room(cache / "room")
    meshes, materials, instances = list(room.meshes), list(room.materials), list(room.instances)
    poses = [NodePose(ROOM_NODE, Pose())]
    asset_ids, sources = {}, {}
    next_id = max(item.id for item in instances) + 1
    if {a.key for a in assets} != set(HOME_PLACEMENTS):
        raise ValueError("the dining room layout requires the ten selected asset categories")
    for asset in assets:
        meshes.extend(asset.meshes)
        materials.extend(asset.materials)
        ids = []
        for part in asset.instances:
            instances.append(replace(part, id=next_id))
            ids.append(next_id)
            next_id += 1
        poses.append(NodePose(asset.node_key, HOME_PLACEMENTS[asset.key]))
        asset_ids[asset.key] = ids
        sources[asset.key] = dict(source=asset.source.relative_to(root).as_posix(),
                                  bounds_m=asset.bounds, position_m=HOME_PLACEMENTS[asset.key].position_m,
                                  import_details=asset.details)
    scene = VisualScene("home-breakfast-ten", tuple(meshes), tuple(materials), tuple(instances), room.lights)
    return scene, tuple(poses), asset_ids, sources


@contextmanager
def encoder(path, width, height):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg with libx264 and ffprobe are required")
    command = [ffmpeg, "-nostdin", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s:v", f"{width}x{height}", "-r", str(FPS), "-i", "pipe:0", "-an",
               "-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p,setsar=1,"
                      "setparams=range=limited:color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709",
               "-c:v", "libx264", "-crf", "17", "-preset", "fast", "-pix_fmt", "yuv420p",
               "-x264-params", "colorprim=bt709:transfer=iec61966-2-1:colormatrix=bt709",
               "-color_range", "tv", "-colorspace", "bt709", "-color_trc", "iec61966-2-1",
               "-color_primaries", "bt709", "-movflags", "+faststart", str(path)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        try:
            yield process.stdin
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("encoding failed for " + str(path))


def run(args):
    output, cache = args.output.resolve(), args.cache.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    final = output / "daylight-at-home.mp4"
    if not args.preview and final.exists():
        raise FileExistsError(final)
    print("Importing the ten source assets and room", flush=True)
    scene, poses, asset_ids, sources = assemble(args.assets.resolve(), cache)
    library = args.library.resolve()
    renderer = StudioRenderer(library_path=library, shader_directory=library.parent,
                              device_index=args.device_index, validation=True,
                              studio=room_lighting(), exposure=ROOM_EXPOSURE, panel_samples=64)
    width, height = renderer.size
    if (width, height) != SIZE:
        raise ValueError("this film is composed for 1440 x 810")
    report = dict(sources=sources, resolution=SIZE, panel_samples=64,
                  exposure=ROOM_EXPOSURE, visibility={})
    with TemporaryDirectory(dir=output) as staging:
        staged = Path(staging) / final.name
        # First finish every Vulkan capture and check the scene before publishing the MP4.
        with closing(renderer.open_scene(scene)) as session:
            with nullcontext() if args.preview else encoder(staged, width, height) as pipe:
                for i in CHECKS if args.preview else range(FRAMES):
                    key = FrameKey(scene.key, "dining-room", 0, i, i, "authored")
                    frame = SceneFrame(key, i/FPS, poses)
                    camera = room_camera(key, frame.time_s, width, height)
                    inspect = i in CHECKS
                    channels = (Channel.RGB, Channel.INSTANCE_ID) if inspect else (Channel.RGB,)
                    shot = session.capture(frame, CaptureRequest(camera, RenderPath.RASTER, channels))
                    rgb = shot.planes[0].data
                    if pipe is not None:
                        pipe.write(rgb)
                    if inspect:
                        labels = np.frombuffer(shot.planes[1].data, dtype="<u4")
                        report["visibility"][str(i)] = {
                            name: int(np.count_nonzero(np.isin(labels, ids))) for name, ids in asset_ids.items()}
                        Image.frombytes("RGB", SIZE, rgb).save(output / f"frame-{i:04d}.jpg", quality=94)
                        print(f"{i + 1}/{FRAMES} rendered", flush=True)
                info = session.diagnostics()
                dev = info["device"]
                ray = dev["resident_ray_query_info"]
                renders = len(CHECKS) if args.preview else FRAMES
                if (dev["validation_errors"] or not ray["enabled"] or ray["shadow_captures"] != renders
                        or info["msaa"]["actual_color_samples"] != 4
                        or info["area_light_samples_per_panel"] != 64
                        or not dev["resident_environment_info"]["loaded"]):
                    raise RuntimeError("the requested Vulkan lighting or image quality was not active")
                report["vulkan"] = dict(device=dev["device_name"], shadow_captures=ray["shadow_captures"],
                                        triangles=info["triangles"], msaa=4, validation_errors=dev["validation_errors"])
        if not args.preview:
            if any(report["visibility"]["0"][name] == 0 for name in sources):
                raise RuntimeError("the establishing shot does not show all ten objects")
            if any(max(counts[name] for counts in report["visibility"].values()) < 64 for name in sources):
                raise RuntimeError("one or more source objects were not visible in the checked exposures")
            report["video"] = _verify(staged, shutil.which("ffmpeg"), shutil.which("ffprobe"),
                                      FPS, FRAMES, width, height)
            staged.rename(final)
    manifest = output / ("preview.json" if args.preview else "film.json")
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("assets", "cache", "output", "library"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--preview", action="store_true")
    run(parser.parse_args())
