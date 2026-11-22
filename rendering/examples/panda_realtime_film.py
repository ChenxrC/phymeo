"""Photograph a sealed Panda episode through public sessions, on its physical clock.

Native timing ends when the requested plane is available to the consumer. PNG
compression, integrity hashes, video encoding and the initial upload are
reported outside the interval.
"""
import argparse
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter

from _panda_scene import build_scene, cameras
from phymeo_rendering.adapters.panda_replay import PandaReplaySource
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.backends.sapien import SapienBackend, SapienConfig, SapienStudio
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import VisualScene
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def fixed(frame, width, height):
    cam = cameras(frame)[0]
    return replace(cam, width=width, height=height, fx_px=cam.fx_px * height / cam.height,
                   fy_px=cam.fy_px * height / cam.height, cx_px=width / 2, cy_px=height / 2)


def load_episode(geo, excerpt, *, workshop=False, photographic=False, local_light_scale=1.):
    geo.mkdir(parents=True)
    original = ROOT / "tests/fixtures/panda_replay/panda.urdf"
    src = PandaReplaySource(VisualScene("geometry-input", (), (), ()), original, excerpt)
    geom = src.provenance(next(src.frames()))["geometry"]
    scene = build_scene(ROOT / "assets/panda", geo, geom["cube_full_size_m"], geom["table_full_size_m"])
    if workshop:
        from _panda_lookdev import lookdev_scene

        scene = lookdev_scene(scene, geo)
    elif photographic:
        from _panda_textures import textured_workshop

        scene = textured_workshop(scene, geo)
    if local_light_scale != 1.:
        scene = replace(scene, key=f"{scene.key}:lights-{local_light_scale.hex()}",
                        lights=tuple(replace(light, illuminance=tuple(
                            local_light_scale * value for value in light.illuminance))
                            for light in scene.lights))
    src = PandaReplaySource(scene, original, excerpt)
    frames = tuple(src.frames())
    if len(frames) != 300 or frames[-1].time_s != 10:
        raise ValueError("expected the frozen ten-second 300-state recording")
    if len({f.key for f in frames}) != len(frames):
        raise ValueError("duplicate source poses")
    return scene, frames


def capture(args):
    args.output.mkdir(parents=True, exist_ok=False)
    # The relative source peaks at 1072. Represent all daylight scene lights
    # in 1/8 units inside the integrator, and restore that scale at the camera.
    unit_scale = .125 if args.daylight else 1.
    scene, frames = load_episode(args.output / "geometry", args.excerpt, workshop=args.workshop,
                                 photographic=args.photographic, local_light_scale=unit_scale)
    is_native = args.mode == "native"
    width, height = (1280, 720) if is_native else (960, 540)
    path = RenderPath.RASTER if is_native else RenderPath.PATH_TRACE
    studio = SapienStudio(center_m=(0.18, 0, 0.42), scale_m=0.85)
    if args.daylight:
        from phymeo_rendering.environment import RadianceEnvironment

        source = ROOT/"examples/materials/university_workshop_1k.hdr"
        environment = RadianceEnvironment(source.as_uri(),
            "a125ada38fc602025326f29b68df09531bb16f899f1973b03d9ac5ac98142548",
            strength=unit_scale*args.light_strength, yaw_degrees=args.light_yaw, face_size=args.cube_size)
        studio = replace(studio, environment_map=environment, panel_strength=unit_scale)
    extra = {"studio": studio} if args.native_studio or args.daylight else {}
    if args.shadow_mode != "software":
        if not is_native:
            raise ValueError("shadow traversal selection requires the native raster backend")
        extra["shadow_mode"] = args.shadow_mode
    backend = (NativeBackend(library_path=args.library, resident=True, device_index=args.device_index,
                             validation=args.validation, exposure=1./unit_scale, **extra) if is_native else
               SapienBackend(SapienConfig(path=path, device=args.device, samples_per_pixel=args.samples,
                                         studio=studio, exposure=1./unit_scale)))
    started = perf_counter()
    sess = backend.open_scene(scene)
    open_ms = 1000 * (perf_counter() - started)
    def exposure(index, frame):
        if args.camera == "fixed":
            return fixed(frame, width, height)
        from panda_movies import movie_camera

        cam = movie_camera(frame, "materials-" + args.camera, index / (len(frames)-1))
        scale = height / cam.height
        return replace(cam, width=width, height=height, fx_px=cam.fx_px*scale, fy_px=cam.fy_px*scale,
                       cx_px=width/2, cy_px=height/2)

    indices = range(0,len(frames),args.stride) if args.frame is None else (args.frame,)
    items = [(frames[i], CaptureRequest(exposure(i,frames[i]), path, (Channel.RGB,))) for i in indices]
    hashes, times, consumer_times = [], [], []
    warm = 10 if is_native and len(items)>=10 else 0
    movies = args.output / "frames"
    movies.mkdir()
    try:
        for frame, req in items[:warm]:
            sess.capture(frame, req)
        # Keep display bytes until after timing so file and codec work cannot
        # inject stalls between timed frames. Bound memory in the offline PT path.
        buffered = []
        for i, (frame, req) in enumerate(items):
            start = perf_counter()
            shot = sess.capture(frame, req)
            rgb = shot.planes[0].data
            times.append((perf_counter() - start) * 1000)
            if shot.frame_key != frame.key or shot.time_s != frame.time_s:
                raise RuntimeError("camera result did not bind the requested physical state")
            if is_native:
                buffered.append(rgb)
            else:
                start = perf_counter()
                hashes.append(sha256(rgb).hexdigest())
                Image.frombytes("RGB", (width, height), rgb).save(movies / f"{i:04d}.png", compress_level=2)
                consumer_times.append((perf_counter() - start) * 1000)
            if (i + 1) % 60 == 0:
                print(f"{args.mode} {i + 1}/{len(items)}", flush=True)
    finally:
        sess.close()
    for i, rgb in enumerate(buffered):
        start = perf_counter()
        hashes.append(sha256(rgb).hexdigest())
        Image.frombytes("RGB", (width, height), rgb).save(movies / f"{i:04d}.png", compress_level=2)
        consumer_times.append((perf_counter() - start) * 1000)
    diag = sess.diagnostics()
    if diag.get("physics_steps") != 0:
        raise RuntimeError("rendering advanced the simulation")
    if is_native:
        d = diag["device"]
        if d["resident_upload_calls"] != 1 or d["resident_captures"] != len(items) + warm:
            raise RuntimeError("unexpected resident uploads or captures")
    elif diag["lifecycle"]["completed_captures"] != len(items):
        raise RuntimeError("incomplete reference exposures")
    ranked = sorted(times)
    receipt = dict(mode=args.mode, backend=diag, camera_setup=args.camera, workshop=args.workshop,
                   shadow_traversal=args.shadow_mode if is_native else None,
                   scene_radiance_unit_scale=unit_scale, display_exposure=1./unit_scale,
                   panorama_relative_strength=args.light_strength if args.daylight else None,
                   photographic_worktop=args.photographic,
                   lighting="workshop daylight + area lights" if args.daylight else
                            "studio" if args.native_studio or not is_native else "directional",
                   source_archive=frames[0].key.world_id,
                   source_excerpt_sha256=sha256(args.excerpt.read_bytes()).hexdigest(),
                   frames=len(items), size=[width,height],
                   fps=30 if args.stride == 1 else str(Fraction(30,args.stride)),
                   recorded_window_s=[0.,10.], simulated_range_s=[items[0][0].time_s,items[-1][0].time_s],
                   source_indices=[f.key.revision for f,_ in items], source_times_s=[f.time_s for f,_ in items],
                   open_ms=open_ms, warmup=warm, capture_ms=times, consumer_work_ms=consumer_times,
                   p50_ms=ranked[math.ceil(len(ranked)*.50)-1], p95_ms=ranked[math.ceil(len(ranked)*.95)-1],
                   max_ms=max(times), over_33_3=sum(t>33.3 for t in times), frame_sha256=hashes)
    (args.output / "capture.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({k:receipt[k] for k in ("mode","frames","p50_ms","p95_ms","max_ms","over_33_3")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("native","studio"))
    parser.add_argument("--excerpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shadow-mode", choices=("software","ray_query"),default="software")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--native-studio", action="store_true",
                        help="request the resident rectangular lights and uniform environment")
    surface = parser.add_mutually_exclusive_group()
    surface.add_argument("--workshop", action="store_true",
                        help="add the oak, metal and rubber visual finishes to the recorded scene")
    surface.add_argument("--photographic", action="store_true",
                         help="map scanned wood onto the original surface and use steel and rubber finishes")
    parser.add_argument("--camera", choices=("fixed","wide","detail"), default="fixed")
    parser.add_argument("--daylight", action="store_true",
                        help="use the pinned workshop panorama for distant radiance and reflections")
    parser.add_argument("--light-strength", type=float, default=.5)
    parser.add_argument("--light-yaw", type=float, default=25.)
    parser.add_argument("--cube-size", type=int, choices=(16,32,64,128,256),default=128)
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--stride", type=int, choices=range(1,301), default=1, metavar="1..300",
                        help="capture every Nth original pose for comparisons")
    select.add_argument("--frame",type=int,choices=range(300),metavar="0..299",
                        help="photograph only one source exposure for composition review")
    capture(parser.parse_args())
