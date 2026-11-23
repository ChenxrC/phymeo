"""A camera-only optical motion fixture for one-bounce nearby reflections.

The same real SceneFrame drives direct and reflected geometry. A red cube moves
out of the direct image while remaining in the steel sphere's reflected view.
Analytic ideal-sphere rays supply conservative, preregistered interior masks;
this is not a physics rollout or a recursive/rough-reflection demonstration.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np

import material_orbit as orbit
import normal_map_film as media
from _panda_scene import box, write_mesh
from panda_msaa_compare import (_hash, _implementation_files, _run, _timing,
                                summarize_codec_errors, verify_probe)
from phymeo_rendering.assets import Material
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import (DirectionalLight, FrameKey, Instance, NodePose,
                                    Pose, SceneFrame, VisualScene)

ROOT = Path(__file__).resolve().parent
CROP = (448, 168, 832, 552)
MODES = ("environment", "single_bounce")
SENSORS = media.SENSORS
compare_sensors = media.compare_sensors
verify_close = media.verify_close
_json = media._json
_image = media._image


def frozen_profile():
    return dict(frames=90, fps=30, size=[1280, 720], modes=list(MODES), warmup=5,
        channels=[c.value for c in Channel], msaa_samples=1,
        sphere_radius_m=.30, sphere_roughness=.04, sphere_metallic=1.,
        cube_size_m=.28, cube_start_m=[.45, -.75, .05], cube_end_m=[1.35, -.75, .05],
        camera_m=[0., -2., 0.], camera_looks="+Y; image right +X; image up +Z",
        focal_px=1000., crop_xyxy=list(CROP), phase_frames=[15, 75],
        display_transform="pbr-neutral", exposure=8., shadow_mode="ray_query", area_light_grid=2,
        environment_strength=.0625, environment_yaw_degrees=25., cube_size=256,
        studio_panel_strength=.125, directional_travel=[1/math.sqrt(3), 1/math.sqrt(3), -1/math.sqrt(3)],
        directional_illuminance=[.25, .25, .25], validation=False,
        gain_expression="(R-max(G,B))_single_bounce - (R-max(G,B))_environment",
        red_gain_mean_min=10., red_gain_fraction_at_least_10_min=.5,
        minimum_reference_pixels=16, miss_max_rgb8=1, background_max_rgb8=0,
        primary_depth_error_max_m=.002, hit_box_inset_m=.035, miss_box_expansion_m=.06,
        minimum_view_cosine=.55,
        timing="public synchronous RGB+forward-depth+ID+VALID to owned CPU bytes; excludes PNG/NPZ/hash/codec",
        source="authored optical motion fixture; no physics; same frames in both modes",
        limitation="one ideal mirror bounce; no recursive reflections or rough-reflection blur")


def cube_position(index):
    if type(index) is not int or not 0 <= index < 90:
        raise ValueError("frame index must be 0..89")
    phase = index/89
    eased = phase*phase*(3-2*phase)
    return (.45+.90*eased, -.75, .05)


def build_fixture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sphere = write_mesh(directory, "mirror-sphere", *orbit.sphere_mesh(radius=.30, rings=64, segments=128))
    cube = box(directory, "moving-red-cube", (.28, .28, .28))
    scene = VisualScene("near-reflection-optical-fixture-v1", (sphere, cube),
        (Material("mirror-steel", (.56, .58, .60), 1., .04),
         Material("moving-red-block", (1., .005, .005), 0., 1.)),
        (Instance(1, sphere.id, "mirror-steel", "steel-ball"),
         Instance(2, cube.id, "moving-red-block", "red-cube")),
        (DirectionalLight((1/math.sqrt(3), 1/math.sqrt(3), -1/math.sqrt(3)), (.25, .25, .25)),))
    sequence = []
    camera_pose = Pose((0., -2., 0.), (math.sqrt(.5), 0., 0., math.sqrt(.5)))
    for i in range(90):
        frame = SceneFrame(FrameKey(scene.key, "optical-fixture", 0, 0, i, "authored"), i/30,
            (NodePose("steel-ball", Pose()), NodePose("red-cube", Pose(cube_position(i)))))
        camera = Camera("fixed-optical-camera", frame.key, camera_pose,
                        1280, 720, 1000., 1000., 640., 360., .01, 20.)
        sequence.append((frame, camera))
    return scene, tuple(sequence)


def project_cube(frame, camera):
    """Independent projection in the registered axis-aligned camera basis."""
    x, y, z = frame.nodes[1].world_pose.position_m
    values = [(640+1000*(x+dx)/(y+dy+2), 360-1000*(z+dz)/(y+dy+2))
              for dx in (-.14, .14) for dy in (-.14, .14) for dz in (-.14, .14)]
    low = np.min(values, axis=0); high = np.max(values, axis=0)
    return dict(min_x_px=float(low[0]), min_y_px=float(low[1]),
        max_x_px=float(high[0]), max_y_px=float(high[1]),
        fully_in_view=bool(low[0]>=0 and low[1]>=0 and high[0]<1280 and high[1]<720),
        fully_outside_right=bool(low[0]>=1280))


def sphere_ray(pixel_x, pixel_y):
    """Ideal-sphere intersection and specular direction, independent of renderer."""
    x, y = np.broadcast_arrays(np.asarray(pixel_x, float), np.asarray(pixel_y, float))
    direction = np.stack(((x-640)/1000, np.ones_like(x), -(y-360)/1000), axis=-1)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    origin = np.array((0., -2., 0.))
    dot = np.sum(origin*direction, axis=-1)
    discriminant = dot*dot-(4.-.30*.30)
    distance = -dot-np.sqrt(np.maximum(discriminant, 0.))
    point = origin+distance[..., None]*direction
    point = np.where((discriminant >= 0)[..., None], point, np.nan)
    normal = point/.30
    reflected = direction-2*np.sum(direction*normal, axis=-1, keepdims=True)*normal
    return point, normal, reflected, point[..., 1]+2


def ray_box(origin, direction, centre, half_size):
    origin, direction = np.broadcast_arrays(np.asarray(origin, float), np.asarray(direction, float))
    near = np.zeros(origin.shape[:-1]); far = np.full_like(near, np.inf)
    valid = np.isfinite(origin).all(axis=-1) & np.isfinite(direction).all(axis=-1)
    for axis in range(3):
        lo, hi = centre[axis]-half_size, centre[axis]+half_size
        o, d = origin[..., axis], direction[..., axis]
        parallel = d == 0
        a = np.divide(lo-o, d, out=np.full_like(o, -np.inf), where=~parallel)
        b = np.divide(hi-o, d, out=np.full_like(o, np.inf), where=~parallel)
        valid &= ~parallel | ((o >= lo) & (o <= hi))
        near = np.maximum(near, np.minimum(a, b))
        far = np.minimum(far, np.maximum(a, b))
    return valid & (far >= near) & (far > 0)


def reference_masks(frame, camera):
    expected_pose = Pose((0., -2., 0.), (math.sqrt(.5), 0., 0., math.sqrt(.5)))
    if (camera.world_pose != expected_pose or (camera.width, camera.height, camera.fx_px, camera.fy_px,
        camera.cx_px, camera.cy_px) != (1280, 720, 1000., 1000., 640., 360.)
        or frame.nodes[0].world_pose != Pose() or frame.nodes[1].world_pose != Pose(cube_position(frame.key.tick))
        or camera.frame_key != frame.key or frame.time_s != frame.key.tick/30):
        raise ValueError("reference requires the registered camera and exact current optical frame")
    x, y = np.meshgrid(np.arange(448, 832)+.5, np.arange(168, 552)+.5)
    point, normal, reflected, depth = sphere_ray(x, y)
    view = np.array((0., -2., 0.))-point
    view /= np.linalg.norm(view, axis=-1, keepdims=True)
    interior = np.isfinite(depth) & (np.sum(view*normal, axis=-1) > .55)
    centre = frame.nodes[1].world_pose.position_m
    hit = interior & ray_box(point, reflected, centre, .14-.035)
    miss = interior & ~ray_box(point, reflected, centre, .14+.06)
    if int(hit.sum()) < 16 or int(miss.sum()) < 5000:
        raise ValueError("predeclared conservative optical masks have insufficient support")
    return dict(hit=hit, miss=miss, depth=depth)


def red_gain(off, on, mask):
    if int(mask.sum()) < 16:
        raise ValueError("red gain requires at least 16 preregistered interior pixels")
    a, b = off[mask].astype(np.int16), on[mask].astype(np.int16)
    delta = (b[:, 0]-np.maximum(b[:, 1], b[:, 2]))-(a[:, 0]-np.maximum(a[:, 1], a[:, 2]))
    mean = float(delta.mean()); fraction = float(np.mean(delta >= 10))
    return dict(expression=frozen_profile()["gain_expression"], pixels=int(mask.sum()),
        mean_delta_red_excess=mean, fraction_delta_at_least_10=fraction,
        passed=bool(mean >= 10 and fraction >= .5))


def measure_frame(off, on, ids, depth, reference, *, direct_phase):
    region = np.s_[168:552, 448:832]
    hit, miss = reference["hit"], reference["miss"]
    if np.any(ids[region][hit | miss] != 1):
        raise ValueError("analytic sphere masks do not match actual receiver ID 1")
    depth_error = float(np.max(np.abs(depth[region][hit | miss]-reference["depth"][hit | miss])))
    if not math.isfinite(depth_error) or depth_error > .002:
        raise ValueError("actual primary depth differs from analytic sphere beyond 2 mm")
    direct_count = int(np.count_nonzero(ids == 2))
    if direct_phase == "offscreen" and direct_count != 0:
        raise ValueError("offscreen red object still has direct ID 2 pixels")
    if direct_phase == "in-frame" and direct_count < 1000:
        raise ValueError("in-frame red object has insufficient actual direct ID 2 pixels")
    difference = np.abs(off.astype(np.int16)-on.astype(np.int16))
    background = ids == 0
    if not np.any(background) or int(difference[background].max()) != 0:
        raise ValueError("unchanged actual background RGB differs")
    gain = red_gain(off[region], on[region], hit)
    miss_max = int(difference[region][miss].max())
    return dict(red=gain, miss_pixels=int(miss.sum()), miss_max_rgb8=miss_max,
        primary_depth_error_max_m=depth_error, direct_red_id2_pixels=direct_count,
        background_exact=True, direct_phase=direct_phase,
        optical_passed=bool(gain["passed"] and miss_max <= 1))


def timing(values):
    if len(values) != 90:
        raise ValueError("require ninety measured four-channel captures")
    return _timing(values)


def verify_movie(probe, *, split=False):
    return verify_probe(probe, width=2562 if split else 1280, height=768, count=90, fps=30)




def _capture(root, mode, scene, sequence, backend):
    from PIL import Image
    directory = root/mode
    (directory/"frames").mkdir(parents=True)
    (directory/"sensors").mkdir()
    report = dict(mode=mode, status="opening", captures=[], capture_ms=[])
    session = None
    try:
        before = perf_counter()
        session = backend.open_scene(scene)
        report["open_ms"] = 1000*(perf_counter()-before)
        for _ in range(5):
            f, c = sequence[0]
            session.capture(f, CaptureRequest(c, RenderPath.RASTER, tuple(Channel)))
        with (directory/"frames.jsonl").open("w", encoding="utf-8", newline="\n") as journal:
            for index, (frame, camera) in enumerate(sequence):
                before = perf_counter()
                capture = session.capture(frame, CaptureRequest(camera, RenderPath.RASTER, tuple(Channel)))
                elapsed = 1000*(perf_counter()-before)
                if (capture.frame_key != frame.key or capture.time_s != frame.time_s
                    or capture.camera != camera or tuple(p.channel for p in capture.planes) != tuple(Channel)):
                    raise ValueError("published frame/camera/clock/channel differs from request")
                planes = {p.channel: p.data for p in capture.planes}
                image = directory/f"frames/{index:04d}.png"
                Image.frombytes("RGB", (1280, 720), planes[Channel.RGB]).save(image)
                sensor = directory/f"sensors/{index:04d}.npz"
                np.savez_compressed(sensor, **{ch.name: np.frombuffer(planes[ch], np.uint8) for ch in SENSORS})
                row = dict(index=index, source_time_s=frame.time_s, frame_key=asdict(frame.key),
                    capture_wall_ms=elapsed, file=f"frames/{index:04d}.png",
                    png_sha256=_hash(image), rgb_sha256=sha256(planes[Channel.RGB]).hexdigest(),
                    sensor_file=f"sensors/{index:04d}.npz", sensor_sha256=_hash(sensor),
                    plane_sha256={ch.value: sha256(data).hexdigest() for ch, data in planes.items()})
                report["captures"].append(row); report["capture_ms"].append(elapsed)
                journal.write(json.dumps(row, allow_nan=False)+"\n"); journal.flush()
        report.update(status="complete", timing=timing(report["capture_ms"]),
            backend_before_close=session.diagnostics(), journal_sha256=_hash(directory/"frames.jsonl"))
    except BaseException as error:
        report.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        original = sys.exc_info()[0] is not None
        close_error = None
        if session is not None:
            try:
                session.close()
                report["backend"] = session.diagnostics()
                report["close"] = verify_close(report["backend"])
            except Exception as error:
                close_error = error
                report["close_error"] = dict(type=type(error).__name__, message=str(error))
        _json(directory/"capture.json", report)
        if close_error is not None and not original:
            raise close_error
    return report


def _sensor(root, mode, index, receipt):
    row = receipt["captures"][index]
    path = root/mode/row["sensor_file"]
    if _hash(path) != row["sensor_sha256"]:
        raise ValueError("published sensor archive hash differs")
    with np.load(path, allow_pickle=False) as saved:
        values = {ch: saved[ch.name].tobytes() for ch in SENSORS}
    if any(sha256(raw).hexdigest() != row["plane_sha256"][ch.value] for ch, raw in values.items()):
        raise ValueError("published sensor bytes differ")
    return values


def verify_reflection_binding(backend, *, mode):
    import struct
    fp32 = lambda value: struct.unpack("<f", struct.pack("<f", value))[0]
    reflection = backend.get("reflection", {})
    retained = backend.get("device", {}).get("resident_reflection_info")
    if reflection.get("mode") != mode:
        raise ValueError("actual reflection mode differs")
    if mode == "environment":
        if reflection.get("bounces") != 0 or reflection.get("receiver_instances") != [] or retained is not None:
            raise ValueError("default environment path unexpectedly enabled nearby reflection")
        return True
    expected = [dict(instance_id=1, material_id="mirror-steel", metallic_fp32=1.,
                     authored_roughness_fp32=fp32(.04))]
    if (mode != "single_bounce" or reflection.get("bounces") != 1
        or reflection.get("receiver_instances") != expected or not isinstance(retained, dict)):
        raise ValueError("actual reflection receiver identity or one-bounce evidence differs")
    for field, value in (("reflection_abi_version",1), ("supported",1), ("shader_compatible",1),
        ("enabled",1), ("max_bounces",1), ("receiver_count",1), ("set_calls",1),
        ("captures",95), ("rgb_captures",95), ("metallic_min",fp32(.99)), ("roughness_max",fp32(.10))):
        if retained.get(field) != value:
            raise ValueError(f"actual reflection retained evidence differs: {field}")
    return True



def compare(root, protocol, receipts, sequence):
    from PIL import Image, ImageDraw
    # The native per-mode admission and runtime receiver evidence is checked here.
    for mode in MODES:
        report = receipts[mode]
        if report["status"] != "complete" or len(report["captures"]) != 90:
            raise ValueError("incomplete actual acquisition")
        if report["timing"] != timing(report["capture_ms"]):
            raise ValueError("reported public timing differs from actual samples")
        backend, device = report["backend"], report["backend"]["device"]
        verify_close(backend)
        for field, value in (("library_sha256", protocol["library"]["sha256"]),
            ("display_transform", "pbr-neutral"), ("exposure", 8.), ("shadow_mode", "ray_query"), ("area_light_grid", 2)):
            if backend.get(field) != value:
                raise ValueError(f"actual backend binding differs: {field}")
        base = receipts["environment"]["backend"]
        if any(backend.get(field) != base.get(field) for field in
               ("scene_key", "decoded_geometry_sha256", "studio", "environment", "texture_assets")):
            raise ValueError("actual geometry/material/light changed between modes")
        if not device.get("device_uuid") or device["device_uuid"] != base["device"]["device_uuid"]:
            raise ValueError("actual physical device changed")
        if (device.get("resident_captures") != 95 or device.get("resident_upload_calls") != 1
            or backend.get("msaa", {}).get("actual_color_samples") != 1):
            raise ValueError("actual captures/upload/MSAA differs")
        verify_reflection_binding(backend, mode=mode)
    rows = []
    for index, (frame, camera) in enumerate(sequence):
        pair = [_sensor(root, mode, index, receipts[mode]) for mode in MODES]
        sensor = compare_sensors(*pair)
        ids = np.frombuffer(pair[0][Channel.INSTANCE_ID], "<u4").reshape(720, 1280)
        depth = np.frombuffer(pair[0][Channel.FORWARD_DEPTH], "<f4").reshape(720, 1280)
        rgb = [np.asarray(_image(root, mode, index, receipts)) for mode in MODES]
        reference = reference_masks(frame, camera)
        registered = protocol["references"][index]
        if (registered["hit_sha256"] != sha256(reference["hit"].tobytes()).hexdigest()
            or registered["miss_sha256"] != sha256(reference["miss"].tobytes()).hexdigest()):
            raise ValueError("pre-exposure analytic masks changed")
        phase = ("offscreen" if registered["projection"]["fully_outside_right"]
                 else "in-frame" if registered["projection"]["fully_in_view"] else "transition")
        measured = measure_frame(*rgb, ids, depth, reference, direct_phase=phase)
        row = dict(index=index, time_s=frame.time_s, sensors=sensor, **measured)
        rows.append(row)
        if index in (15, 75):
            label = "in-frame" if index == 15 else "offscreen"
            pair_image = Image.new("RGB", (770, 432), (22, 24, 29))
            draw = ImageDraw.Draw(pair_image)
            for mode, x in (("environment", 0), ("single_bounce", 386)):
                crop = _image(root, mode, index, receipts).crop(CROP)
                crop.save(root/f"{label}-{mode.replace('_','-')}.png")
                pair_image.paste(crop, (x, 48))
                draw.text((x+12, 17), "ENVIRONMENT" if mode == "environment" else "ONE NEARBY BOUNCE", fill="white")
            pair_image.save(root/f"{label}-pair.png")
    result = dict(input_match=True, geometry_sensor_bytes_identical=True, frames=rows,
        optical_gates_passed=all(r["optical_passed"] for r in rows),
        interpretation="ID 1 remains the primary mirror sphere even when its RGB reflects ID 2; no secondary depth/ID publication")
    _json(root/"comparison.json", result)
    return result


def run(output, library, *, device_index=1, shader_directory=None, videos=True):
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.environment import RadianceEnvironment
    from phymeo_rendering.studio import StudioLighting
    output, library = Path(output).resolve(), Path(library).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="phymeo-near-reflection-film-v1", status="preparing", physics_steps=0)
    try:
        scene, sequence = build_fixture(output/"geometry")
        references = []
        (output/"reference").mkdir()
        for index, (frame, camera) in enumerate(sequence):
            masks = reference_masks(frame, camera)
            projection = project_cube(frame, camera)
            references.append(dict(index=index, projection=projection,
                hit_pixels=int(masks["hit"].sum()), miss_pixels=int(masks["miss"].sum()),
                hit_sha256=sha256(masks["hit"].tobytes()).hexdigest(),
                miss_sha256=sha256(masks["miss"].tobytes()).hexdigest()))
            if index in (15, 75):
                np.savez_compressed(output/f"reference/{index:04d}.npz", **masks)
        if not references[15]["projection"]["fully_in_view"] or not references[75]["projection"]["fully_outside_right"]:
            raise ValueError("registered direct-view phase failed before GPU")
        source = ROOT/"materials/university_workshop_1k.hdr"
        if _hash(source) != orbit.SOURCES[source.name]:
            raise ValueError("registered HDR source changed")
        studio = StudioLighting(center_m=(0.,0.,0.), scale_m=.85, panel_strength=.125,
            environment_map=RadianceEnvironment(source.as_uri(), orbit.SOURCES[source.name],
                                               strength=.0625, yaw_degrees=25., face_size=256))
        shaders = library.parent if shader_directory is None else Path(shader_directory).resolve()
        protocol = dict(schema="phymeo-near-reflection-film-protocol-v1", profile=frozen_profile(),
            scene=asdict(scene), sequence=[dict(frame=asdict(f), camera=asdict(c)) for f,c in sequence],
            references=references, studio=asdict(studio), source_hdr=dict(file=str(source), sha256=_hash(source)),
            source_notice=(ROOT/"materials/NOTICE.md").read_text(encoding="utf-8"),
            script_sha256=_hash(__file__), helper_sha256={p.name:_hash(p) for p in
                (Path(orbit.__file__), Path(media.__file__), ROOT/"_panda_scene.py", ROOT/"panda_msaa_compare.py", ROOT/"_panda_movie_encode.py")},
            implementation=_implementation_files(), device_index=device_index,
            mesh_sha256={m.id:orbit.load_mesh(m).source_sha256 for m in scene.meshes},
            library=dict(file=str(library), sha256=_hash(library)),
            shaders={p.name:_hash(p) for p in sorted(shaders.glob("*.spv"))},
            identities={"1": "镜面钢球 / mirror steel sphere", "2": "真实移动红块 / moving red block"},
            only_changed="reflection_mode; all frames, cameras, meshes, material factors and illumination shared")
        _json(output/"protocol.json", protocol)
        receipts = {}
        for mode in MODES:
            backend = NativeBackend(library_path=library, shader_directory=shaders, device_index=device_index,
                resident=True, validation=False, studio=studio, exposure=8., display_transform="pbr-neutral",
                shadow_mode="ray_query", area_light_grid=2, msaa_samples=1, reflection_mode=mode)
            receipts[mode] = _capture(output, mode, scene, sequence, backend)
        if (protocol["implementation"] != _implementation_files() or protocol["script_sha256"] != _hash(__file__)
            or protocol["library"]["sha256"] != _hash(library)
            or protocol["shaders"] != {p.name:_hash(p) for p in sorted(shaders.glob("*.spv"))}
            or protocol["mesh_sha256"] != {m.id:orbit.load_mesh(m).source_sha256 for m in scene.meshes}
            or any(_hash(ROOT/name) != digest for name,digest in protocol["helper_sha256"].items())):
            raise ValueError("executing source, geometry or shader binding changed")
        comparison = compare(output, protocol, receipts, sequence)
        report.update(status="complete" if comparison["optical_gates_passed"] else "optical_gate_failed",
            protocol_sha256=_hash(output/"protocol.json"), comparison_sha256=_hash(output/"comparison.json"),
            comparison=comparison,
            modes={m:dict(timing=receipts[m]["timing"], close=receipts[m]["close"], backend=receipts[m]["backend"])
                   for m in MODES},
            receipts={m:dict(file=f"{m}/capture.json", sha256=_hash(output/m/"capture.json")) for m in MODES})
    except BaseException as error:
        report.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        _json(output/"summary.json", report)
    if videos:
        encode(output)
    return report




def _compose(root, mode, index, receipts, protocol, failed):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (2562 if mode == "split" else 1280, 768), (22,24,29))
    draw = ImageDraw.Draw(image)
    projection = protocol["references"][index]["projection"]
    phase = ("RED BLOCK OUTSIDE DIRECT VIEW" if projection["fully_outside_right"] else
             "RED BLOCK IN DIRECT VIEW" if projection["fully_in_view"] else "RED BLOCK LEAVING DIRECT VIEW")
    for variant, x in ((("environment",0),("single_bounce",1282)) if mode == "split" else ((mode,0),)):
        image.paste(_image(root, variant, index, receipts), (x,48))
        label = "ENVIRONMENT ONLY" if variant == "environment" else "ONE NEARBY BOUNCE"
        prefix = "OPTICAL GATE FAILED | " if failed else ""
        draw.text((x+16,15), f"{prefix}{label} | {phase} | {index/30:.2f} s | OPTICAL FIXTURE", fill="white")
    return image


def _encode_one(root, mode, target, receipts, protocol, failed, ffmpeg, ffprobe):
    from _panda_movie_encode import _faststart
    width, height = (2562 if mode == "split" else 1280), 768
    filters = ("setsar=1,scale=in_range=full:out_range=limited:out_color_matrix=bt709:"
        "flags=bicubic+accurate_rnd+full_chroma_int,format=yuv420p,"
        "setparams=range=limited:color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
        "-framerate", "30", "-i", "pipe:0", "-an", "-vf", filters, "-c:v", "libx264",
        "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-fps_mode", "cfr",
        "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709",
        "-color_trc", "iec61966-2-1", "-movflags", "+faststart", str(target)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    hashes = []
    with target.with_suffix(".encode.log").open("wb") as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=errors, creationflags=flags)
        try:
            for index in range(90):
                raw = _compose(root,mode,index,receipts,protocol,failed).tobytes()
                hashes.append(sha256(raw).hexdigest()); process.stdin.write(raw)
        finally:
            process.stdin.close()
            code = process.wait(timeout=600)
        if code:
            raise RuntimeError("encoding failed; retained original inputs and log")
    probe = json.loads(_run([ffprobe, "-v", "error", "-count_frames", "-show_streams",
        "-show_format", "-show_frames", "-show_entries",
        "stream=codec_type,codec_name,codec_tag_string,pix_fmt,width,height,avg_frame_rate,r_frame_rate,"
        "time_base,duration_ts,nb_frames,nb_read_frames,color_range,color_space,color_transfer,"
        "color_primaries,sample_aspect_ratio:frame=best_effort_timestamp_time:format=duration", "-of", "json", target]))
    _json(target.with_suffix(".probe.json"), probe)
    report = verify_movie(probe, split=mode=="split")
    rows = []
    with target.with_suffix(".decode.log").open("wb") as errors:
        process = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-xerror", "-err_detect", "explode", "-i", str(target), "-map", "0:v:0", "-an",
            "-vf", "scale=in_range=limited:out_range=full:in_color_matrix=bt709,format=rgb24",
            "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"], stdout=subprocess.PIPE,
            stdin=subprocess.DEVNULL, stderr=errors, creationflags=flags)
        try:
            size = width*height*3
            for index in range(90):
                parts, length = [], 0
                while length < size:
                    part = process.stdout.read(size-length)
                    if not part:
                        raise ValueError("truncated actual decoded frame")
                    parts.append(part); length += len(part)
                expected = _compose(root,mode,index,receipts,protocol,failed).tobytes()
                if sha256(expected).hexdigest() != hashes[index]:
                    raise ValueError("reconstructed encoding input changed")
                delta = np.abs(np.frombuffer(b"".join(parts),np.uint8).astype(np.int16)
                               - np.frombuffer(expected,np.uint8).astype(np.int16))
                rows.append(dict(index=index, mae_rgb8=float(delta.mean()), max_rgb8=int(delta.max())))
            if process.stdout.read(1):
                raise ValueError("extra decoded frames")
        finally:
            process.stdout.close()
            code = process.wait(timeout=600)
        if code:
            raise RuntimeError("full movie decode failed")
    report.update(file=target.name,sha256=_hash(target),faststart=_faststart(target),
        complete_decode=True,encode_input_sha256=hashes,codec_error=summarize_codec_errors(rows),command=command)
    return report


def encode(root, *, media_output=None):
    root = Path(root).resolve()
    protocol = json.loads((root/"protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    if (summary.get("status") not in ("complete","optical_gate_failed")
        or protocol["profile"] != frozen_profile()
        or summary["protocol_sha256"] != _hash(root/"protocol.json")
        or summary["comparison_sha256"] != _hash(root/"comparison.json")
        or summary["comparison"].get("geometry_sensor_bytes_identical") is not True
        or summary["comparison"].get("input_match") is not True):
        raise ValueError("require complete source/media/geometry bindings; no bypass for physical mismatches")
    failed = summary["status"] != "complete"
    receipts = {}
    for mode in MODES:
        if _hash(root/mode/"capture.json") != summary["receipts"][mode]["sha256"]:
            raise ValueError("original capture receipt changed")
        receipts[mode] = json.loads((root/mode/"capture.json").read_text(encoding="utf-8"))
    target_dir = root if media_output is None else Path(media_output).resolve()
    if media_output is not None:
        target_dir.mkdir(parents=True,exist_ok=False)
    tools = {key:shutil.which(key) for key in ("ffmpeg","ffprobe")}
    if not all(tools.values()):
        raise RuntimeError("ffmpeg/ffprobe required on PATH")
    report = dict(schema="phymeo-near-reflection-film-codec-v1", movies={}, optical_gate_failed=failed,
        source_directory=str(root), media_directory=str(target_dir),
        protocol_sha256=_hash(root/"protocol.json"), summary_sha256=_hash(root/"summary.json"),
        conversion="full sRGB8 to limited BT.709 YUV420; IEC sRGB transfer retained; no image adjustment",
        labels="embedded font in external 48px band; original picture pixels not scaled",
        tools={key:_run([value,"-version"]).splitlines()[0] for key,value in tools.items()})
    try:
        for mode in (*MODES,"split"):
            target = target_dir/f"near-reflection-{mode.replace('_','-')}.mp4"
            if target.exists():
                raise FileExistsError("preserve original movie; use --media-output NEW")
            report["movies"][mode] = _encode_one(root,mode,target,receipts,protocol,failed,**tools)
            _json(target_dir/"encoding.json",report)
    except BaseException as error:
        report["error"] = dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        _json(target_dir/"encoding.json",report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--library",type=Path)
    parser.add_argument("--shader-directory",type=Path)
    parser.add_argument("--device-index",type=int,default=1)
    parser.add_argument("--no-video",action="store_true")
    parser.add_argument("--encode-only",action="store_true")
    parser.add_argument("--media-output",type=Path)
    args = parser.parse_args()
    if args.media_output and not args.encode_only:
        parser.error("--media-output requires --encode-only")
    if args.encode_only:
        encode(args.output,media_output=args.media_output)
    else:
        if args.library is None:
            parser.error("--library is required for capture")
        run(args.output,args.library,device_index=args.device_index,
            shader_directory=args.shader_directory,videos=not args.no_video)


if __name__ == "__main__":
    main()
