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
from _home_film_quality import capture_film_frame, capture_lens_reference, focus_distance
from _home_showcase_room import (
    HOME_PLACEMENTS, ROOM_EXPOSURE, ROOM_NODE, build_room, room_camera, room_focus_target, room_lighting,
    with_home_backplate,
)
from _panda_movie_encode import _verify
from phymeo_rendering.outputs import Channel
from phymeo_rendering import StudioRenderer
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, NodePose, Pose, SceneFrame, VisualScene


FPS = 30
FRAMES = 30 * FPS
CHECKS = (0, 150, 345, 459, 540, 651, 750, FRAMES - 1)
SIZE = (1440, 810)


def compare_edge(session, frame, camera, native_rgb, supersampled_rgb, crop):
    """Compare two pinhole images with a finer render of the same optical rays."""
    x, y, w, h = crop
    window = replace(camera, width=w, height=h, cx_px=camera.cx_px-x, cy_px=camera.cy_px-y)
    raw, _, _ = capture_film_frame(session, frame, window, focus_distance_m=1.,
                                   aperture_radius_m=0., scale=6)
    reference = np.frombuffer(raw, np.uint8).reshape(h, w, 3).astype(np.float32)
    luma = reference @ np.array([.2126, .7152, .0722], np.float32)
    gy, gx = np.gradient(luma)
    contrast = np.hypot(gx, gy)
    # Exclude the render-target boundary from the derivative comparison.
    edge = contrast >= max(4., float(np.percentile(contrast[2:-2,2:-2], 90)))
    edge[:2] = edge[-2:] = False
    edge[:,:2] = edge[:,-2:] = False
    result = dict(crop_xywh=list(crop), reference_scale=6, edge_pixels=int(edge.sum()))
    if not edge.any():
        raise ValueError("reference crop contains no measurable edges")
    for name, data in (("native",native_rgb),("supersampled",supersampled_rgb)):
        image = np.frombuffer(data, np.uint8).reshape(camera.height,camera.width,3)
        error = np.abs(image[y:y+h,x:x+w].astype(np.float32)-reference)
        result[name + "_edge_mae_rgb8"] = float(error[edge].mean())
    return result


def compare_background_focus(session, frame, camera, focus, aperture, sharp, final, crop):
    x, y, w, h = crop
    window = replace(camera, width=w, height=h, cx_px=camera.cx_px-x, cy_px=camera.cy_px-y)
    raw, info = capture_lens_reference(session, frame, window, focus_distance_m=focus,
                                       aperture_radius_m=aperture, samples=24, scale=2)
    reference = np.frombuffer(raw, np.uint8).reshape(h,w,3).astype(np.float32)
    info["crop_xywh"] = list(crop)
    for name, data in (("pinhole",sharp),("depth_layers",final)):
        image = np.frombuffer(data, np.uint8).reshape(camera.height,camera.width,3)
        info[name + "_mae_rgb8"] = float(np.abs(image[y:y+h,x:x+w]-reference).mean())
    return info


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
    scene = VisualScene("home-breakfast-ten:" + room.key, tuple(meshes), tuple(materials), tuple(instances), room.lights)
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
               "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p",
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
    final = output / ("motion-check.mp4" if args.motion_check else "daylight-at-home-quality.mp4")
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
    indices = sorted(set(CHECKS) | {346,347}) if args.preview else (
        range(330,360) if args.motion_check else range(FRAMES))
    checks = {0,345,346,347}
    report = dict(sources=sources, resolution=SIZE, panel_samples=64, supersample_scale=2,
                  exposure=ROOM_EXPOSURE, visibility={}, quality_samples={},
                  pinhole_edge_checks={}, background_lens_checks={})
    comparison_captures = 0
    with TemporaryDirectory(dir=output) as staging:
        staged = Path(staging) / final.name
        # First finish every Vulkan capture and check the scene before publishing the MP4.
        with closing(renderer.open_scene(scene)) as native:
            session = with_home_backplate(native, scene)
            with nullcontext() if args.preview else encoder(staged, width, height) as pipe:
                for i in indices:
                    key = FrameKey(scene.key, "dining-room", 0, i, i, "authored")
                    frame = SceneFrame(key, i/FPS, poses)
                    camera = room_camera(key, frame.time_s, width, height)
                    inspect = i in CHECKS or (args.preview and i in checks)
                    focus = focus_distance(camera, room_focus_target(frame.time_s))
                    # Maintain a similar background separation as the camera dollies
                    # from the wide room view to the much closer tableware.
                    aperture = min(.038, max(.007, .014*focus))
                    rgb, labels, quality = capture_film_frame(
                        session, frame, camera, focus_distance_m=focus, aperture_radius_m=aperture,
                        scale=2, inspect=inspect)
                    if pipe is not None:
                        pipe.write(rgb)
                    if inspect:
                        labels = np.frombuffer(labels, dtype="<u4") if isinstance(labels, bytes) else labels
                        report["visibility"][str(i)] = {
                            name: int(np.count_nonzero(np.isin(labels, ids))) for name, ids in asset_ids.items()}
                        report["quality_samples"][str(i)] = quality
                        Image.frombytes("RGB", SIZE, rgb).save(output / f"frame-{i:04d}.png")
                    if args.preview and i in checks:
                        # Same camera, lighting and assets: isolate spatial AA before adding depth of field.
                        old = session.capture(frame, CaptureRequest(camera, RenderPath.RASTER, (Channel.RGB,)))
                        sharp, _, _ = capture_film_frame(
                            session, frame, camera, focus_distance_m=focus, aperture_radius_m=0., scale=2)
                        for prefix, data in (("native", old.planes[0].data), ("supersampled", sharp)):
                            directory = output / prefix
                            directory.mkdir(exist_ok=True)
                            Image.frombytes("RGB", SIZE, data).save(directory / f"frame-{i:04d}.png")
                        comparison_captures += 2
                        crop = (160,482,150,88) if i == 0 else (379,430,155,98)
                        report["pinhole_edge_checks"][str(i)] = compare_edge(
                            session, frame, camera, old.planes[0].data, sharp, crop)
                        comparison_captures += 1
                        if i in (0,345):
                            background = (445,179,145,87) if i == 0 else (364,0,155,98)
                            optics = compare_background_focus(session, frame, camera, focus, aperture,
                                                              sharp, rgb, background)
                            report["background_lens_checks"][str(i)] = optics
                            comparison_captures += optics["lens_samples"]
                    if inspect or i % 60 == 0:
                        print(f"Filmed exposure {i+1}/{FRAMES}", flush=True)
                info = session.diagnostics()
                dev = info["device"]
                ray = dev["resident_ray_query_info"]
                renders = len(indices) + comparison_captures
                if (dev["validation_errors"] or not ray["enabled"] or ray["shadow_captures"] != renders
                        or info["msaa"]["actual_color_samples"] != 4
                        or info["area_light_samples_per_panel"] != 64
                        or not dev["resident_environment_info"]["loaded"]):
                    raise RuntimeError("the requested Vulkan lighting or image quality was not active")
                report["vulkan"] = dict(device=dev["device_name"], shadow_captures=ray["shadow_captures"],
                                        triangles=info["triangles"], msaa=4, validation_errors=dev["validation_errors"])
                report["window_photograph"] = info["exterior_backplate"]
        if not args.preview:
            if not args.motion_check and any(report["visibility"]["0"][name] == 0 for name in sources):
                raise RuntimeError("the establishing shot does not show all ten objects")
            if not args.motion_check and any(
                    max(counts[name] for counts in report["visibility"].values()) < 64 for name in sources):
                raise RuntimeError("one or more source objects were not visible in the checked exposures")
            report["video"] = _verify(staged, shutil.which("ffmpeg"), shutil.which("ffprobe"),
                                      FPS, len(indices), width, height)
            staged.rename(final)
    manifest = output / ("preview.json" if args.preview else "motion-check.json" if args.motion_check else "film.json")
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("assets", "cache", "output", "library"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--motion-check", action="store_true", help="encode a one-second moving crop-check sequence")
    run(parser.parse_args())
