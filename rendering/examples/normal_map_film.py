"""Original material-orbit frames 60..179, with only two normal maps toggled.

Authored analytic height derivatives, not a scan, measured anisotropic BRDF or
geometry displacement. Mirror steel and scanned wood remain unchanged controls.
"""
import argparse
from dataclasses import asdict, replace
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
from _panda_scene import write_mesh
from panda_msaa_compare import (_hash, _implementation_files, _run,
                                summarize_codec_errors, verify_probe)
from phymeo_rendering.assets import TextureMap, TexturedMaterial
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath

ROOT = Path(__file__).resolve().parent
KINDS = {"rough-steel": "steel", "matte-rubber": "rubber"}
SPECIMENS = {"steel": 2, "rubber": 3}
SENSORS = (Channel.FORWARD_DEPTH, Channel.INSTANCE_ID, Channel.VALID)
SCOUTS = (0, 60, 119)


def frozen_profile():
    return dict(variants=["off", "on"], frames=120, fps=30, size=[1280, 720],
        source_start=60, source_stop_exclusive=180, source_total_frames=240,
        warmup=5, sensor_scout_indices=list(SCOUTS), msaa_samples=1,
        display_transform="pbr-neutral", exposure=8., cube_size=256,
        environment_strength=.0625, environment_yaw_degrees=25.,
        panel_strength=.125, shadow_mode="ray_query", area_light_grid=2,
        validation=False, texture_side=1024, normal_strength=1.,
        crop_source_frame=120, crop_size=[192, 192],
        effect_gate=dict(min_changed_fraction=.05, channel_delta_at_least=2),
        timing="public synchronous RGB capture to owned bytes; excludes PNG/hash/codec and separate sensor scouts",
        percentile="nearest rank ceil(n*p), excludes five warmups")


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False)+"\n", encoding="utf-8")


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def height_and_slopes(kind, u, v):
    """Height [m], dh/dx, dh/dy on a reference rectangular UV chart.

    x=2*pi*.11*u, y=pi*.11*v. Its metre metric is exact at the equator,
    not a globe displacement. +U is increasing azimuth, +V north to south.
    """
    u, v = np.asarray(u), np.asarray(v)
    width, height = 2*math.pi*.11, math.pi*.11
    if kind == "steel":
        phase = 2*math.pi*64*u
        amplitude = .22*width/(2*math.pi*64)
        value = amplitude*(np.sin(phase)+.16*np.sin(2*phase+.7))
        dx = .22*(np.cos(phase)+.32*np.cos(2*phase+.7))
        return value+np.zeros_like(v), dx+np.zeros_like(v), np.zeros_like(u+v)
    if kind == "rubber":
        a, b = 2*math.pi*48*u, 2*math.pi*24*v
        amplitude = .35*width/(2*math.pi*48)
        return (amplitude*np.sin(a)*np.sin(b), .35*np.cos(a)*np.sin(b),
                amplitude*2*math.pi*24/height*np.sin(a)*np.cos(b))
    raise ValueError("unknown registered normal material")


def tangent_normal(kind, u, v):
    _, dx, dy = height_and_slopes(kind, u, v)
    value = np.stack((-dx, -dy, np.ones_like(dx)), axis=-1)
    return value/np.linalg.norm(value, axis=-1, keepdims=True)


def make_textures(directory, *, side=1024):
    from PIL import Image
    if type(side) is not int or side not in (256, 512, 1024):
        raise ValueError("bounded normal texture side must be 256, 512 or 1024")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    u, v = np.meshgrid((np.arange(side)+.5)/side, (np.arange(side)+.5)/side)
    result = {}
    for kind in ("steel", "rubber"):
        rgb = np.floor((tangent_normal(kind, u, v)+1)*127.5+.5).astype(np.uint8)
        path = directory/f"{kind}-normal.png"
        Image.fromarray(rgb).save(path)
        result[kind] = dict(file=str(path.resolve()), sha256=_hash(path),
            rgb8_sha256=sha256(rgb.tobytes()).hexdigest(), size=[side, side],
            source="authored analytic height derivatives; not a scan",
            transfer="linear RGB8 normal data, no sRGB decode",
            encoding="floor((unit_normal+1)*127.5+0.5); no special 128 bias cancellation",
            sampling="texel centres, top-row-first; bilinear repeat, mip 0",
            direction="+T: increasing azimuth U; +B: north-to-south V; +Z: outward",
            derivation="normalize(-dh/dx,-dh/dy,1); height_and_slopes in hash-bound script",
            reference_chart_m=[2*math.pi*.11, math.pi*.11],
            physical_limit="chart metric at equator; no displacement or measured BRDF claim",
            periods_uv=[64, 0] if kind == "steel" else [48, 24],
            slope_parameter=.22 if kind == "steel" else .35,
            generator_sha256=_hash(__file__))
    _json(directory/"provenance.json", result)
    return result


def uv_sphere(radius=.11, rings=48, segments=96):
    """Duplicate seams and poles so every face has finite UV differentials."""
    p, n, uv, triangles = [], [], [], []
    for ring in range(1, rings):
        theta = math.pi*ring/rings
        for j in range(segments+1):
            phi = 2*math.pi*(j % segments)/segments
            normal = (math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta))
            n.append(normal); p.append(tuple(radius*x for x in normal))
            uv.append((j/segments, ring/rings))
    for j in range(segments):
        top = len(p)
        p.append((0., 0., radius)); n.append((0., 0., 1.)); uv.append(((j+.5)/segments, 0.))
        triangles.append((top, j, j+1))
        for ring in range(rings-2):
            a = ring*(segments+1)+j
            b = a+segments+1
            triangles.extend(((a, b, a+1), (a+1, b, b+1)))
        bottom = len(p)
        p.append((0., 0., -radius)); n.append((0., 0., -1.)); uv.append(((j+.5)/segments, 1.))
        a = (rings-2)*(segments+1)+j
        triangles.append((a, bottom, a+1))
    return tuple(p), tuple(n), tuple(triangles), tuple(uv)


def build_pair(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scene, nodes = orbit.build_scene(directory/"geometry")
    p, n, t, uv = uv_sphere()
    sphere = write_mesh(directory/"geometry", "normal-study-sphere", p, n, t, texcoords=uv)
    old_sphere = scene.instances[0].mesh_id
    meshes = tuple(sphere if m.id == old_sphere else m for m in scene.meshes)
    instances = tuple(replace(i, mesh_id=sphere.id) if i.mesh_id == old_sphere else i for i in scene.instances)
    off = replace(scene, key="authored-normal-map-off-v1", meshes=meshes, instances=instances)
    textures = make_textures(directory/"textures")
    materials = []
    for m in off.materials:
        if m.id in KINDS:
            item = textures[KINDS[m.id]]
            materials.append(TexturedMaterial(m.id, m.base_color, m.metallic, m.roughness,
                normal_map=TextureMap(Path(item["file"]).as_uri(), item["sha256"])))
        else:
            materials.append(m)
    return off, replace(off, key="authored-normal-map-on-v1", materials=tuple(materials)), nodes, textures


def exposures(scene, nodes):
    return orbit.exposures(scene, nodes, fps=30, width=1280, height=720)[60:180]


def _material(m):
    return dict(id=m.id, base_color=m.base_color, metallic=m.metallic, roughness=m.roughness,
        base_color_map=getattr(m, "base_color_map", None), roughness_map=getattr(m, "roughness_map", None))


def bind_pair(off, on, left, right):
    if off.key == on.key or off.meshes != on.meshes or off.instances != on.instances:
        raise ValueError("paired geometry/instance identity changed")
    if off.lights != on.lights:
        raise ValueError("paired light changed")
    if len(off.materials) != len(on.materials):
        raise ValueError("material count changed")
    changes = {}
    for a, b in zip(off.materials, on.materials, strict=True):
        if _material(a) != _material(b) or getattr(a, "normal_map", None) is not None:
            raise ValueError("paired base material changed")
        normal = getattr(b, "normal_map", None)
        if (a.id in KINDS) != isinstance(normal, TextureMap):
            raise ValueError("normal material scope differs from registration")
        if normal is not None:
            changes[a.id] = asdict(normal)
    if len(left) != 120 or len(right) != 120:
        raise ValueError("paired frame count changed")
    rows = []
    for i, ((fa, ca), (fb, cb)) in enumerate(zip(left, right, strict=True)):
        a, b = asdict(fa), asdict(fb)
        if (fa.key.scene_key, fb.key.scene_key) != (off.key, on.key):
            raise ValueError("frame scene binding changed")
        a["key"]["scene_key"] = b["key"]["scene_key"] = "paired-scene"
        if a != b or fa.key.tick != 60+i or fa.time_s != (60+i)/30:
            raise ValueError("paired frame/motion/time changed")
        a, b = asdict(ca), asdict(cb)
        if ca.frame_key != fa.key or cb.frame_key != fb.key:
            raise ValueError("camera frame binding changed")
        a["frame_key"]["scene_key"] = b["frame_key"]["scene_key"] = "paired-scene"
        if a != b:
            raise ValueError("paired camera changed")
        rows.append(dict(source_tick=60+i, off_frame=asdict(fa), on_frame=asdict(fb),
                         off_camera=asdict(ca), on_camera=asdict(cb)))
    return dict(matched=True, permitted_changes=["scene key", "two declared normal maps"],
                normal_maps=changes, rows=rows, input_sha256=_digest(rows),
                off_scene=asdict(off), on_scene=asdict(on))


def crop_rectangles(camera):
    result = {}
    for kind, identity in SPECIMENS.items():
        x, y, _ = orbit.project(camera, orbit.CENTRES[identity])
        left, top = round(x)-96, round(y)-96
        if not (0 <= left and left+192 <= camera.width and 0 <= top and top+192 <= camera.height):
            raise ValueError("registered crop is outside image")
        result[kind] = [left, top, left+192, top+192]
    return result


def compare_sensors(a, b):
    if any(ch not in a or ch not in b or a[ch] != b[ch] for ch in SENSORS):
        raise ValueError("paired centre sensor bytes changed or missing")
    return dict(identical=True, sha256={ch.value: sha256(a[ch]).hexdigest() for ch in SENSORS})


def timing(values):
    if len(values) != 120 or any(isinstance(v, bool) or not isinstance(v, (float, int))
                                 or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("require 120 finite nonnegative actual capture durations")
    ordered = sorted(values)
    return dict(p50_ms=ordered[59], p95_ms=ordered[113], mean_ms=math.fsum(values)/120,
                max_ms=max(values), worst_frame=values.index(max(values)),
                over_33_3=sum(v > 33.3 for v in values))


def verify_close(value):
    d = value.get("device", {})
    t = d.get("resident_telemetry", {})
    if (d.get("closed") is not True or any(d.get(k) != 0 for k in
        ("validation_errors", "live_contexts", "live_frames", "live_resources"))
        or t.get("final_after_destroy") is not True
        or any(t.get(k) != 0 for k in ("device_local_bytes_current", "host_visible_bytes_current"))
        or not isinstance(t.get("allocation_count"), int)
        or t.get("allocation_count") != t.get("free_count")):
        raise ValueError("native close allocation/resource evidence incomplete or not balanced")
    return dict(closed=True, telemetry=t)


def verify_movie(probe, *, split=False):
    return verify_probe(probe, width=2562 if split else 1280, height=768, count=120, fps=30)




def _capture_mode(root, mode, scene, sequence, backend):
    from PIL import Image
    directory = root/mode
    directory.mkdir()
    (directory/"frames").mkdir()
    report = dict(mode=mode, status="opening", captures=[], sensor_scouts=[], capture_ms=[])
    session = None
    try:
        before = perf_counter()
        session = backend.open_scene(scene)
        report["open_ms"] = 1000*(perf_counter()-before)
        for _ in range(5):
            f, c = sequence[0]
            session.capture(f, CaptureRequest(c, RenderPath.RASTER, (Channel.RGB,)))
        with (directory/"frames.jsonl").open("w", encoding="utf-8", newline="\n") as journal:
            for index, (frame, camera) in enumerate(sequence):
                before = perf_counter()
                captured = session.capture(frame, CaptureRequest(camera, RenderPath.RASTER, (Channel.RGB,)))
                elapsed = 1000*(perf_counter()-before)
                if (captured.frame_key != frame.key or captured.camera != camera
                    or captured.time_s != frame.time_s or tuple(p.channel for p in captured.planes) != (Channel.RGB,)):
                    raise ValueError("published capture changed requested source/camera/channel")
                raw = captured.planes[0].data
                path = directory/f"frames/{index:04d}.png"
                Image.frombytes("RGB", (1280, 720), raw).save(path)
                row = dict(index=index, source_tick=frame.key.tick, source_time_s=frame.time_s,
                    movie_pts_s=index/30, capture_wall_ms=elapsed, file=f"frames/{index:04d}.png",
                    png_sha256=_hash(path), rgb_sha256=sha256(raw).hexdigest())
                report["captures"].append(row); report["capture_ms"].append(elapsed)
                journal.write(json.dumps(row, allow_nan=False)+"\n"); journal.flush()
        for index in SCOUTS:
            frame, camera = sequence[index]
            captured = session.capture(frame, CaptureRequest(camera, RenderPath.RASTER, tuple(Channel)))
            if captured.frame_key != frame.key or captured.time_s != frame.time_s or captured.camera != camera:
                raise ValueError("sensor scout changed source/camera")
            metrics, planes = orbit._observe(captured)
            if sha256(planes[Channel.RGB]).hexdigest() != report["captures"][index]["rgb_sha256"]:
                raise ValueError("same-state scout RGB differs from timed RGB capture")
            path = directory/f"sensor-{index:04d}.npz"
            np.savez_compressed(path, **{ch.name: np.frombuffer(data, np.uint8) for ch, data in planes.items()})
            report["sensor_scouts"].append(dict(index=index, file=path.name, sha256=_hash(path),
                plane_sha256={ch.value: sha256(data).hexdigest() for ch, data in planes.items()},
                specimens=metrics))
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


def _image(root, mode, index, receipts):
    from PIL import Image
    row = receipts[mode]["captures"][index]
    path = root/mode/row["file"]
    if _hash(path) != row["png_sha256"]:
        raise ValueError("source PNG file hash changed")
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (1280, 720):
            raise ValueError("source PNG layout changed")
        value = image.copy()
    if sha256(value.tobytes()).hexdigest() != row["rgb_sha256"]:
        raise ValueError("source PNG no longer matches public RGB bytes")
    return value


def verify_normal_binding(backend, textures, *, mode):
    """Match actual retained GPU binding to the two registered source maps."""
    maps = backend.get("normal_maps")
    retained = backend.get("device", {}).get("resident_normal_info") or {}
    if mode == "off":
        if maps is not None or retained.get("loaded", 0) != 0:
            raise ValueError("normal maps unexpectedly active in off control")
        return True
    if mode != "on" or not isinstance(maps, dict):
        raise ValueError("actual normal map diagnostics missing")
    if (maps.get("strength") != 1.
        or maps.get("instance_slots") != [2**32-1, 0, 1, 2**32-1, 2**32-1]):
        raise ValueError("actual normal strength or specimen slots differ")
    actual = maps.get("textures", [])
    if len(actual) != 2:
        raise ValueError("actual normal texture count differs")
    for item, kind in zip(actual, ("steel", "rubber"), strict=True):
        expected = textures[kind]
        if (item.get("source_sha256") != expected["sha256"]
            or item.get("rgb8_sha256") != expected["rgb8_sha256"]
            or [item.get("width"), item.get("height")] != expected["size"]):
            raise ValueError("actual normal source bytes/layout differ")
    for field, expected in (("loaded", 1), ("texture_count", 2), ("bound_instances", 2),
        ("upload_calls", 1), ("rgb_captures_with_normals", 128), ("normal_abi_version", 1), ("mip_levels", 1)):
        if retained.get(field) != expected:
            raise ValueError(f"retained normal GPU evidence differs: {field}")
    return True



def _compare(root, protocol, receipts):
    from PIL import Image, ImageDraw
    for mode in ("off", "on"):
        report = receipts[mode]
        if report["status"] != "complete" or len(report["captures"]) != 120:
            raise ValueError("incomplete captured mode")
        if report["timing"] != timing(report["capture_ms"]):
            raise ValueError("claimed timing differs from actual samples")
        backend, device = report["backend"], report["backend"]["device"]
        verify_close(backend)
        verify_normal_binding(backend, protocol["textures"], mode=mode)
        for field, expected in (("library_sha256", protocol["library"]["sha256"]),
            ("display_transform", "pbr-neutral"), ("exposure", 8.),
            ("shadow_mode", "ray_query"), ("area_light_grid", 2)):
            if backend.get(field) != expected:
                raise ValueError(f"actual backend differs: {field}")
        if any(backend.get(field) != receipts["off"]["backend"].get(field)
               for field in ("decoded_geometry_sha256", "studio", "environment")):
            raise ValueError("actual geometry or lighting changed across modes")
        if not device.get("device_uuid") or device["device_uuid"] != receipts["off"]["backend"]["device"]["device_uuid"]:
            raise ValueError("paired device changed")
        if (device.get("resident_captures") != 128 or device.get("resident_upload_calls") != 1
            or backend.get("msaa", {}).get("actual_color_samples") != 1):
            raise ValueError("actual capture/upload/MSAA count differs")
    evidence, effect_failures = [], []
    for index in SCOUTS:
        sensors = []
        for mode in ("off", "on"):
            item = next(r for r in receipts[mode]["sensor_scouts"] if r["index"] == index)
            path = root/mode/item["file"]
            if _hash(path) != item["sha256"]:
                raise ValueError("sensor file hash changed")
            with np.load(path, allow_pickle=False) as saved:
                sensors.append({ch: saved[ch.name].tobytes() for ch in Channel})
        proof = compare_sensors(*sensors)
        ids = np.frombuffer(sensors[0][Channel.INSTANCE_ID], "<u4").reshape(720, 1280)
        rgb = [np.asarray(_image(root, m, index, receipts)) for m in ("off", "on")]
        delta = np.abs(rgb[0].astype(np.int16)-rgb[1].astype(np.int16)).max(axis=2)
        controls = np.isin(ids, (1, 4))
        if not np.any(controls) or np.any(delta[controls]):
            raise ValueError("unchanged mirror/wood control pixels changed")
        effects = {}
        for kind, identity in SPECIMENS.items():
            mask = ids == identity
            if int(mask.sum()) < 200:
                raise ValueError("specimen has insufficient actual visible pixels")
            fraction = float(np.mean(delta[mask] >= 2))
            effects[kind] = dict(visible_pixels=int(mask.sum()), changed_fraction_at_least_2=fraction,
                p95_max_channel_delta=float(np.percentile(delta[mask], 95)),
                mean_absolute_channel_delta=float(np.abs(rgb[0][mask].astype(float)-rgb[1][mask]).mean()),
                off_p95_max_rgb8=float(np.percentile(rgb[0][mask].max(axis=1), 95)),
                on_p95_max_rgb8=float(np.percentile(rgb[1][mask].max(axis=1), 95)))
            if fraction < .05:
                effect_failures.append(dict(kind=kind, index=index, changed_fraction=fraction))
        evidence.append(dict(index=index, sensors=proof, controls_exact=True, effects=effects))
    for kind, rectangle in protocol["crop_rectangles"].items():
        strip = Image.new("RGB", (386, 240), (22, 24, 29))
        draw = ImageDraw.Draw(strip)
        for mode, x in (("off", 0), ("on", 194)):
            crop = _image(root, mode, 60, receipts).crop(rectangle)
            crop.save(root/f"{kind}-{mode}.png")
            strip.paste(crop, (x, 48))
            draw.text((x+8, 17), f"{kind.upper()} / {mode.upper()}", fill="white")
        strip.save(root/f"{kind}-pair.png")
    result = dict(input_match=True, sensor_scouts=evidence, crop_source_frame=120,
        modes={mode: dict(timing=receipts[mode]["timing"], close=receipts[mode]["close"],
                         device=receipts[mode]["backend"]["device"]) for mode in ("off", "on")},
        scope="same-input normal-map appearance and centre-sensor invariance; not a scanned-material or BRDF accuracy claim")
    result["visible_effect_failures"] = effect_failures
    _json(root/"comparison.json", result)
    if effect_failures:
        raise ValueError(f"registered visible effect failed; see retained comparison.json: {effect_failures}")
    return result


def run(output, library, *, device_index=1, shader_directory=None, videos=True):
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.environment import RadianceEnvironment
    from phymeo_rendering.studio import StudioLighting
    output, library = Path(output).resolve(), Path(library).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="phymeo-normal-map-film-v1", status="preparing", physics_steps=0)
    try:
        off, on, nodes, textures = build_pair(output)
        left, right = exposures(off, nodes), exposures(on, nodes)
        studio = StudioLighting(center_m=(0., 0., .13), scale_m=.85, panel_strength=.125,
            environment_map=RadianceEnvironment((ROOT/"materials/university_workshop_1k.hdr").as_uri(),
                orbit.SOURCES["university_workshop_1k.hdr"], strength=.0625, yaw_degrees=25., face_size=256))
        shaders = library.parent if shader_directory is None else Path(shader_directory).resolve()
        protocol = dict(schema="phymeo-normal-map-film-protocol-v1", profile=frozen_profile(),
            inputs=bind_pair(off, on, left, right), textures=textures, studio=asdict(studio),
            crop_rectangles=crop_rectangles(left[60][1]),
            mesh_sha256={m.id: orbit.load_mesh(m).source_sha256 for m in off.meshes},
            source_assets=orbit.verify_sources(), source_notice=(ROOT/"materials/NOTICE.md").read_text(encoding="utf-8"),
            script_sha256=_hash(__file__), orbit_sha256=_hash(orbit.__file__),
            implementation=_implementation_files(), device_index=device_index,
            library=dict(file=str(library), sha256=_hash(library)),
            shaders={p.name: _hash(p) for p in sorted(shaders.glob("*.spv"))},
            invariants="geometry, UV, cameras, motion, base/metal/roughness, light, exposure, samples fixed; scene keys and normal maps differ")
        _json(output/"protocol.json", protocol)
        receipts = {}
        for mode, scene, sequence in (("off", off, left), ("on", on, right)):
            backend = NativeBackend(library_path=library, shader_directory=shaders, device_index=device_index,
                resident=True, validation=False, studio=studio, exposure=8.,
                display_transform="pbr-neutral", shadow_mode="ray_query", area_light_grid=2, msaa_samples=1)
            receipts[mode] = _capture_mode(output, mode, scene, sequence, backend)
        if (protocol["implementation"] != _implementation_files() or protocol["script_sha256"] != _hash(__file__)
            or protocol["orbit_sha256"] != _hash(orbit.__file__)
            or protocol["mesh_sha256"] != {m.id: orbit.load_mesh(m).source_sha256 for m in off.meshes}
            or protocol["library"]["sha256"] != _hash(library)
            or protocol["shaders"] != {p.name: _hash(p) for p in sorted(shaders.glob("*.spv"))}):
            raise ValueError("executing implementation changed during acquisition")
        report.update(status="complete", protocol_sha256=_hash(output/"protocol.json"),
            comparison=_compare(output, protocol, receipts),
            receipts={m: dict(file=f"{m}/capture.json", sha256=_hash(output/m/"capture.json")) for m in receipts})
    except BaseException as error:
        report.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        _json(output/"summary.json", report)
    if videos:
        encode(output)
    return report




def _compose(root, mode, index, receipts):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (2562 if mode == "split" else 1280, 768), (22, 24, 29))
    draw = ImageDraw.Draw(image)
    for variant, x in ((("off", 0), ("on", 1282)) if mode == "split" else ((mode, 0),)):
        image.paste(_image(root, variant, index, receipts), (x, 48))
        draw.text((x+18, 16), f"NORMAL MAP {variant.upper()} | SAME FRAME {60+index:03d} | AUTHORED DETAIL", fill="white")
    return image


def _encode_one(root, mode, target, receipts, ffmpeg, ffprobe):
    from _panda_movie_encode import _faststart
    width, height = (2562 if mode == "split" else 1280), 768
    filters = ("setsar=1,scale=in_range=full:out_range=limited:out_color_matrix=bt709:"
        "flags=bicubic+accurate_rnd+full_chroma_int,format=yuv420p,"
        "setparams=range=limited:color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
        "-framerate", "30", "-i", "pipe:0", "-an", "-vf", filters,
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr", "-color_range", "tv", "-colorspace", "bt709",
        "-color_primaries", "bt709", "-color_trc", "iec61966-2-1", "-movflags", "+faststart", str(target)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    hashes = []
    with target.with_suffix(".encode.log").open("wb") as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=errors, creationflags=flags)
        try:
            for i in range(120):
                raw = _compose(root, mode, i, receipts).tobytes()
                hashes.append(sha256(raw).hexdigest())
                process.stdin.write(raw)
        finally:
            process.stdin.close()
            code = process.wait(timeout=600)
        if code:
            raise RuntimeError("encoding failed; original inputs and encode log retained")
    probe = json.loads(_run([ffprobe, "-v", "error", "-count_frames", "-show_streams",
        "-show_format", "-show_frames", "-show_entries",
        "stream=codec_type,codec_name,codec_tag_string,pix_fmt,width,height,avg_frame_rate,r_frame_rate,"
        "time_base,duration_ts,nb_frames,nb_read_frames,color_range,color_space,color_transfer,"
        "color_primaries,sample_aspect_ratio:frame=best_effort_timestamp_time:format=duration", "-of", "json", target]))
    _json(target.with_suffix(".probe.json"), probe)
    report = verify_movie(probe, split=mode == "split")
    rows = []
    with target.with_suffix(".decode.log").open("wb") as errors:
        process = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-xerror", "-err_detect", "explode", "-i", str(target), "-map", "0:v:0", "-an",
            "-vf", "scale=in_range=limited:out_range=full:in_color_matrix=bt709,format=rgb24",
            "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"], stdout=subprocess.PIPE,
            stdin=subprocess.DEVNULL, stderr=errors, creationflags=flags)
        try:
            count = width*height*3
            for i in range(120):
                pieces, have = [], 0
                while have < count:
                    part = process.stdout.read(count-have)
                    if not part:
                        raise ValueError("truncated actual movie decode")
                    pieces.append(part); have += len(part)
                original = _compose(root, mode, i, receipts).tobytes()
                if sha256(original).hexdigest() != hashes[i]:
                    raise ValueError("encoding input changed")
                diff = np.abs(np.frombuffer(b"".join(pieces), np.uint8).astype(np.int16)
                              - np.frombuffer(original, np.uint8).astype(np.int16))
                rows.append(dict(index=i, mae_rgb8=float(diff.mean()), max_rgb8=int(diff.max())))
            if process.stdout.read(1):
                raise ValueError("extra decoded frames")
        finally:
            process.stdout.close()
            code = process.wait(timeout=600)
        if code:
            raise RuntimeError("actual movie decode failed")
    report.update(file=target.name, sha256=_hash(target), faststart=_faststart(target),
        complete_decode=True, encode_input_sha256=hashes, codec_error=summarize_codec_errors(rows),
        command=command)
    return report


def encode(root, *, media_output=None):
    root = Path(root).resolve()
    protocol = json.loads((root/"protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    if (summary.get("status") != "complete" or protocol["profile"] != frozen_profile()
            or summary["protocol_sha256"] != _hash(root/"protocol.json")):
        raise ValueError("require a complete, bound original acquisition")
    receipts = {}
    for mode in ("off", "on"):
        if _hash(root/mode/"capture.json") != summary["receipts"][mode]["sha256"]:
            raise ValueError("captured receipt changed")
        receipts[mode] = json.loads((root/mode/"capture.json").read_text(encoding="utf-8"))
    media = root if media_output is None else Path(media_output).resolve()
    if media_output is not None:
        media.mkdir(parents=True, exist_ok=False)
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    if not all(tools.values()):
        raise RuntimeError("ffmpeg and ffprobe must be on PATH")
    result = dict(schema="phymeo-normal-map-film-codec-v1", movies={},
        source_directory=str(root), media_directory=str(media),
        protocol_sha256=_hash(root/"protocol.json"), summary_sha256=_hash(root/"summary.json"),
        conversion="full sRGB8 to limited BT.709 YUV420; IEC sRGB transfer retained; no exposure or tone adjustment",
        labels="embedded Pillow font in external 48px band, original pixels not resized",
        tools={name: _run([value, "-version"]).splitlines()[0] for name, value in tools.items()})
    try:
        for mode in ("off", "on", "split"):
            target = media/f"normal-map-{mode}.mp4"
            if target.exists():
                raise FileExistsError("preserve existing movie; use --media-output NEW")
            result["movies"][mode] = _encode_one(root, mode, target, receipts, **tools)
            _json(media/"encoding.json", result)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        _json(media/"encoding.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--shader-directory", type=Path)
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--media-output", type=Path)
    args = parser.parse_args()
    if args.media_output and not args.encode_only:
        parser.error("--media-output requires --encode-only")
    if args.encode_only:
        encode(args.output, media_output=args.media_output)
    else:
        if args.library is None:
            parser.error("--library is required for actual capture")
        run(args.output, args.library, shader_directory=args.shader_directory,
            device_index=args.device_index, videos=not args.no_video)


if __name__ == "__main__":
    main()
