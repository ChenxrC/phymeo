"""One frozen 1x/4x Panda pair; public readback timing precedes all codecs.

Run capture once, then use --encode-only to produce alternate presentation
without rendering again. PNGs are authoritative; codec errors are reported as
lossy presentation errors, never as an AA quality oracle.
"""
import argparse
from fractions import Fraction
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import panda_realtime_film as film
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath

TRACE_SHA256 = "58073ab38acab8f5057061fa487117aa0284f81b097a4f5bb0473f890515f118"
BAND = 48
DIVIDER = 2


def frozen_profile():
    return dict(samples=[1, 4], frames=300, fps=30, size=[1280, 720], warmup=10,
        camera="sequence", camera_cuts_s=[.5, 2.3], display_transform="pbr-neutral",
        photographic=True, surface_roughness=True, cube_size=256, daylight=True,
        light_strength=.5, light_yaw=25., exposure=8., shadow_mode="ray_query",
        area_light_grid=2, validation=False, crop_frame=60, crop_size=[320, 240],
        timing="public synchronous session.capture through owned RGB bytes; no PNG/hash/codec",
        percentile="nearest rank ceil(n*p), no warmup samples", fast_limit_p95_ms=33.3)


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _hash(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _timing(values):
    if not values or any(isinstance(x, bool) or not isinstance(x, (int, float))
                         or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("capture timing must be finite nonnegative measured milliseconds")
    ordered = sorted(values)
    return dict(p50_ms=ordered[math.ceil(len(values) * .5) - 1],
                p95_ms=ordered[math.ceil(len(values) * .95) - 1],
                mean_ms=math.fsum(values) / len(values), max_ms=max(values),
                worst_frame=values.index(max(values)), over_33_3=sum(x > 33.3 for x in values))


def compare_receipts(protocol, one, four):
    """Audit actual paired receipts against the pre-exposure input registration."""
    expected = frozen_profile()
    if protocol.get("schema") != "phymeo-panda-msaa-protocol-v1" or protocol.get("profile") != expected:
        raise ValueError("protocol profile differs from the frozen experiment")
    binding = protocol["input_binding"]
    if len(binding["frames"]) != 300:
        raise ValueError("protocol input count")
    result = dict(schema="phymeo-panda-msaa-comparison-v1", input_match=True, modes={})
    for samples, receipt in ((1, one), (4, four)):
        if receipt.get("profile") != expected:
            raise ValueError("capture profile does not match protocol")
        if receipt.get("input_binding") != binding:
            raise ValueError("capture input binding differs from pre-exposure protocol")
        if any(receipt.get(key) != value for key, value in
               (("msaa_samples", samples), ("frames", 300), ("size", [1280, 720]), ("fps", 30))):
            raise ValueError("capture sample/count/size/fps mismatch")
        if receipt.get("warmup") != 10 or len(receipt["capture_ms"]) != 300:
            raise ValueError("capture timing requires ten warmups and 300 measured frames")
        if len(receipt.get("frame_sha256", ())) != 300:
            raise ValueError("incomplete captured frame hashes")
        times = _timing(receipt["capture_ms"])
        for name in ("p50_ms", "p95_ms", "max_ms"):
            if receipt.get(name) != times[name]:
                raise ValueError(f"capture timing claimed {name} differs from individual samples")
        backend, device = receipt["backend"], receipt["backend"]["device"]
        if backend.get("library_sha256") != protocol["library_sha256"]:
            raise ValueError("capture uses a foreign native DLL")
        msaa = backend.get("msaa", {})
        resolve = "display-linear then IEC sRGB once" if samples == 4 else "none (single sample)"
        if any(msaa.get(key) != value for key, value in
               (("requested_samples", samples), ("actual_color_samples", samples),
                ("sensor_samples", 1), ("sensor_sampling", "deterministic pixel center"),
                ("resolve_space", resolve))):
            raise ValueError("native MSAA actual samples/resolve/sensor contract mismatch")
        if device.get("closed") is not True or any(device.get(name) != 0 for name in
                ("live_contexts", "live_frames", "live_resources", "validation_errors")):
            raise ValueError("native session not cleanly closed")
        if device.get("resident_upload_calls") != 1 or device.get("resident_captures") != 310:
            raise ValueError("native upload/capture count mismatch")
        result["modes"][str(samples)] = dict(times, samples=300, warmup=10,
            capture_ms=receipt["capture_ms"], consumer_work_ms=receipt["consumer_work_ms"],
            backend_before_close=receipt.get("backend_before_close"), backend=backend,
            implementation=receipt.get("implementation"))
    result["fast_profile_passed"] = result["modes"]["4"]["p95_ms"] <= 33.3
    result["performance_scope"] = "this DLL/device and frozen Neutral+scanned-roughness 720p profile only"
    result["aa_quality_scope"] = "Panda visual evidence only; analytic AA/sensor acceptance is the separate calibration"
    return result


def summarize_codec_errors(rows):
    if not rows or [r["index"] for r in rows] != list(range(len(rows))):
        raise ValueError("codec rows require contiguous index starting at zero")
    if any(not math.isfinite(r["mae_rgb8"]) or not 0 <= r["mae_rgb8"] <= 255
           or not 0 <= r["max_rgb8"] <= 255 for r in rows):
        raise ValueError("invalid RGB8 codec error")
    return dict(frames=len(rows), mean_mae_rgb8=math.fsum(r["mae_rgb8"] for r in rows) / len(rows),
        worst_mae_rgb8=max(r["mae_rgb8"] for r in rows),
        worst_mae_frame=max(rows, key=lambda r: r["mae_rgb8"])["index"],
        max_rgb8=max(r["max_rgb8"] for r in rows),
        worst_max_frame=max(rows, key=lambda r: r["max_rgb8"])["index"], per_frame=rows)


def verify_probe(probe, *, width, height, count, fps=30):
    streams = probe.get("streams", [])
    if len(streams) != 1 or streams[0].get("codec_type") != "video":
        raise ValueError("require exactly one video stream and no audio")
    stream = streams[0]
    expected = dict(codec_name="h264", codec_tag_string="avc1", pix_fmt="yuv420p", width=width,
        height=height, color_range="tv", color_space="bt709", color_transfer="iec61966-2-1",
        color_primaries="bt709", sample_aspect_ratio="1:1")
    if any(stream.get(k) != v for k, v in expected.items()):
        raise ValueError("video size/codec/color contract mismatch")
    if any(int(stream.get(k, -1)) != count for k in ("nb_frames", "nb_read_frames")):
        raise ValueError("incomplete decoded frame count")
    if any(Fraction(stream[k]) != fps for k in ("avg_frame_rate", "r_frame_rate")):
        raise ValueError("video frame rate mismatch")
    duration = Fraction(count, fps)
    if int(stream["duration_ts"]) * Fraction(stream["time_base"]) != duration:
        raise ValueError("video duration mismatch")
    if abs(float(probe["format"]["duration"]) - float(duration)) > .000002:
        raise ValueError("container duration mismatch")
    frames = probe.get("frames", [])
    if len(frames) != count or any(abs(Fraction(f["best_effort_timestamp_time"]) - Fraction(i, fps))
                                > Fraction(1, 1_000_000) for i, f in enumerate(frames)):
        raise ValueError("noncontiguous video presentation timestamps")
    return dict(contiguous_timestamps=True, frames=count, duration_s=float(duration), stream=stream)


def _args(output, excerpt, library, device_index, samples):
    p = frozen_profile()
    return SimpleNamespace(mode="native", output=output, excerpt=excerpt, library=library,
        device_index=device_index, device="cuda:0", validation=p["validation"], samples=256,
        msaa_samples=samples, reference_seed=None, native_studio=False, workshop=False,
        photographic=True, surface_roughness=True, display_transform=p["display_transform"],
        camera="sequence", resolution="720p", tiles=1, daylight=True, light_strength=.5,
        light_yaw=25., cube_size=256, shadow_mode="ray_query", area_light_grid=2,
        stride=1, frame=None, comparison_profile=p)


def _crop_rectangle(frame, camera):
    # Geometry-only pre-registration, before either image exists. The two
    # finger joint origins choose a centre; no threshold or image content is used.
    nodes = {node.node_id: node.world_pose.position_m for node in frame.nodes}
    points = [nodes[name] for name in ("panda_leftfinger", "panda_rightfinger")]
    point = tuple((a + b) / 2 for a, b in zip(*points))
    x, y, z, w = camera.world_pose.orientation_xyzw
    rotation = ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))
    delta = tuple(a-b for a,b in zip(point, camera.world_pose.position_m))
    local = tuple(sum(rotation[row][col]*delta[row] for row in range(3)) for col in range(3))
    if -local[2] <= camera.near_m:
        raise ValueError("registered finger crop is behind the camera")
    u = camera.cx_px + camera.fx_px * local[0] / -local[2]
    v = camera.cy_px - camera.fy_px * local[1] / -local[2]
    width, height = frozen_profile()["crop_size"]
    left, top = round(u-width/2), round(v-height/2)
    if left < 0 or top < 0 or left+width > camera.width or top+height > camera.height:
        raise ValueError("registered finger crop does not fit the original image")
    return [left, top, left+width, top+height]


def _implementation_files():
    import phymeo_rendering
    package = Path(phymeo_rendering.__file__).resolve().parent
    return dict(directory=str(package), files={str(p.relative_to(package)): _hash(p)
        for p in sorted(package.rglob("*")) if p.is_file() and p.suffix in
        (".py", ".cpp", ".h", ".inc", ".vert", ".frag", ".glsl")})


def run_pair(output, excerpt, library, *, device_index=1, videos="both"):
    output, excerpt, library = Path(output).resolve(), Path(excerpt).resolve(), Path(library).resolve()
    if _hash(excerpt) != TRACE_SHA256:
        raise ValueError("require the original frozen 300-state Panda excerpt")
    library_sha = _hash(library)
    output.mkdir(parents=True, exist_ok=False)
    args = _args(output / "1x", excerpt, library, device_index, 1)
    scene, frames = film.load_episode(output / "geometry", excerpt, photographic=True,
                                     local_light_scale=.125, surface_roughness=True)
    cameras = [film.exposure_camera(args, frame, i, len(frames), 1280, 720)
               for i, frame in enumerate(frames)]
    items = tuple((frame, CaptureRequest(camera, RenderPath.RASTER, (Channel.RGB,)))
                  for frame, camera in zip(frames, cameras))
    protocol = dict(schema="phymeo-panda-msaa-protocol-v1", profile=frozen_profile(),
        library=str(library), library_sha256=library_sha, device_index=device_index,
        shader_sha256={p.name: _hash(p) for p in sorted(library.parent.glob("*.spv"))},
        implementation=_implementation_files(),
        input_binding=film.input_binding(scene, items, excerpt),
        crop_rectangle=_crop_rectangle(frames[60], cameras[60]),
        crop_source="frame60; projected mean of left/right finger joint origins; pixels 1:1",
        driver_sha256=_hash(Path(__file__)), film_sha256=_hash(Path(film.__file__)))
    _json(output / "protocol.json", protocol)
    receipts = []
    try:
        for samples in (1, 4):
            receipts.append(film.capture(_args(output / f"{samples}x", excerpt, library,
                device_index, samples), prepared=(scene, frames)))
        if _hash(library) != library_sha or _implementation_files() != protocol["implementation"]:
            raise ValueError("implementation changed during paired capture")
        result = compare_receipts(protocol, *receipts)
        result["protocol_sha256"] = _hash(output / "protocol.json")
        result["capture_receipt_sha256"] = {str(n): _hash(output/f"{n}x/capture.json") for n in (1,4)}
        result["images"] = analyze_images(output, protocol, receipts)
        _json(output / "comparison.json", result)
    except Exception as exc:
        _json(output / "failure.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}",
            completed_modes=len(receipts), protocol_sha256=_hash(output / "protocol.json")))
        raise
    if videos != "none":
        encode_pair(output, videos=videos)
    return result


def _load_image(root, samples, index, receipt):
    from PIL import Image
    path = root / f"{samples}x/frames/{index:04d}.png"
    with Image.open(path) as source:
        if source.mode != "RGB" or source.size != (1280, 720):
            raise ValueError("source PNG layout changed")
        image = source.copy()
    if sha256(image.tobytes()).hexdigest() != receipt["frame_sha256"][index]:
        raise ValueError("source PNG differs from the original public capture")
    return image


def analyze_images(root, protocol, receipts):
    import numpy as np
    from PIL import Image, ImageDraw
    rows = []
    for i in range(300):
        a, b = [_load_image(root, n, i, receipt) for n, receipt in zip((1,4), receipts)]
        difference = np.abs(np.asarray(a).astype(np.int16)-np.asarray(b).astype(np.int16))
        rows.append(dict(index=i, mae_rgb8=float(difference.mean()), max_rgb8=int(difference.max()),
                         changed_pixels=int(np.any(difference != 0, axis=2).sum())))
        if i == 60:
            crops = [image.crop(protocol["crop_rectangle"]) for image in (a,b)]
            for n, crop in zip((1,4), crops):
                crop.save(root / f"finger-1to1-{n}x.png")
            strip = Image.new("RGB", (642, 288), (22,24,29))
            strip.paste(crops[0], (0,48)); strip.paste(crops[1], (322,48))
            draw = ImageDraw.Draw(strip)
            draw.text((12,15), "1X - SAME FRAME 60", fill="white")
            draw.text((334,15), "4X - SAME FRAME 60", fill="white")
            strip.save(root / "finger-1to1-pair.png")
    return dict(scope="1x/4x image difference only, not an AA error oracle",
                difference=summarize_codec_errors(rows), crop_rectangle=protocol["crop_rectangle"],
                crop_scale=1, crop_frame=60)


def _compose(root, mode, index, receipts):
    from PIL import Image, ImageDraw
    size = (2562, 768) if mode == "split" else (1280, 768)
    image = Image.new("RGB", size, (22,24,29))
    draw = ImageDraw.Draw(image)
    for samples, x in (((1,0),(4,1282)) if mode == "split" else ((int(mode),0),)):
        source = _load_image(root, samples, index, receipts[samples])
        image.paste(source, (x,48))
        draw.text((x+20,17), f"{samples}X MSAA  |  SAME POSE / CAMERA  |  FRAME {index:03d}", fill="white")
    return image


def _run(command):
    result = subprocess.run([str(v) for v in command], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True,
        encoding="utf-8", errors="replace", timeout=600,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        raise RuntimeError(result.stderr[-12000:])
    return result.stdout


def _encode_one(root, mode, target, receipts, ffmpeg, ffprobe):
    import numpy as np
    from _panda_movie_encode import _faststart
    width, height = (2562,768) if mode == "split" else (1280,768)
    filters = ("scale=in_range=full:out_range=limited:out_color_matrix=bt709:"
               "flags=bicubic+accurate_rnd+full_chroma_int,format=yuv420p")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
        "-framerate", "30", "-i", "pipe:0", "-an", "-vf", filters,
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr", "-color_range", "tv", "-colorspace", "bt709",
        "-color_primaries", "bt709", "-color_trc", "iec61966-2-1", "-movflags", "+faststart", str(target)]
    log = target.with_suffix(".encode.log")
    hashes = []
    with log.open("wb") as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=errors, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            for i in range(300):
                data = _compose(root, mode, i, receipts).tobytes()
                hashes.append(sha256(data).hexdigest())
                process.stdin.write(data)
        finally:
            process.stdin.close()
            code = process.wait(timeout=600)
        if code:
            raise RuntimeError(f"encoding failed; see {log}")
    probe = json.loads(_run([ffprobe, "-v", "error", "-count_frames", "-show_streams",
        "-show_format", "-show_frames", "-show_entries",
        "stream=codec_type,codec_name,codec_tag_string,pix_fmt,width,height,avg_frame_rate,r_frame_rate,"
        "time_base,duration_ts,nb_frames,nb_read_frames,color_range,color_space,color_transfer,"
        "color_primaries,sample_aspect_ratio:frame=best_effort_timestamp_time:format=duration", "-of", "json", target]))
    report = verify_probe(probe, width=width, height=height, count=300)
    rows = []
    log = target.with_suffix(".decode.log")
    # Decode all actual encoded frames, comparing the exact RGB encode inputs.
    # No raw video is written to disk, and no second GPU capture is requested.
    with log.open("wb") as errors:
        process = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-xerror", "-err_detect", "explode", "-i", str(target), "-map", "0:v:0", "-an",
            "-vf", "scale=in_range=limited:out_range=full:in_color_matrix=bt709,format=rgb24",
            "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"], stdout=subprocess.PIPE,
            stdin=subprocess.DEVNULL, stderr=errors,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            count = width*height*3
            for i in range(300):
                pieces, have = [], 0
                while have < count:
                    part = process.stdout.read(count-have)
                    if not part:
                        raise ValueError("truncated decoded frame")
                    pieces.append(part); have += len(part)
                decoded = b"".join(pieces)
                expected = _compose(root, mode, i, receipts).tobytes()
                if sha256(expected).hexdigest() != hashes[i]:
                    raise ValueError("encode input reconstruction changed")
                diff = np.abs(np.frombuffer(decoded,np.uint8).astype(np.int16)
                              - np.frombuffer(expected,np.uint8).astype(np.int16))
                rows.append(dict(index=i, mae_rgb8=float(diff.mean()), max_rgb8=int(diff.max())))
            if process.stdout.read(1):
                raise ValueError("extra decoded frames")
        finally:
            process.stdout.close()
            code = process.wait(timeout=600)
        if code:
            raise RuntimeError(f"decoding failed; see {log}")
    report.update(file=target.name, sha256=_hash(target), faststart=_faststart(target),
        encode_input_sha256=hashes, codec_error=summarize_codec_errors(rows), complete_decode=True)
    return report


def encode_pair(root, *, videos="both"):
    root = Path(root).resolve()
    if videos not in ("none", "toggle", "split", "both"):
        raise ValueError("unknown video selection")
    protocol = json.loads((root / "protocol.json").read_text())
    receipts = {n: json.loads((root/f"{n}x/capture.json").read_text()) for n in (1,4)}
    compare_receipts(protocol, receipts[1], receipts[4])
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    if not all(tools.values()):
        raise RuntimeError("ffmpeg and ffprobe are required for video output")
    modes = [] if videos == "none" else ["1", "4"] if videos == "toggle" else ["split"]
    if videos == "both": modes = ["1", "4", "split"]
    report = dict(schema="phymeo-panda-msaa-codec-v1", movies={},
        input_protocol_sha256=_hash(root/"protocol.json"), source_color="sRGB RGB8 unchanged",
        conversion="BT.709 matrix, full RGB to limited YUV420, sRGB transfer retained; no display transform",
        labels="Pillow embedded default font, external 48px band; no source-image labels",
        tools={name: _run([exe,"-version"]).splitlines()[0] for name,exe in tools.items()})
    for mode in modes:
        target = root / ("panda-msaa-split.mp4" if mode == "split" else f"panda-msaa-{mode}x.mp4")
        if target.exists():
            raise FileExistsError(f"preserve existing movie: {target}")
        report["movies"][mode] = _encode_one(root, mode, target, receipts, **tools)
        _json(root / "encoding.json", report)
    if videos in ("toggle", "both"):
        (root/"compare.html").write_text('''<!doctype html><meta charset="utf-8"><title>Panda 1x / 4x MSAA</title>
<style>body{background:#17191e;color:#eee;font:18px sans-serif;max-width:1280px;margin:24px auto}video{width:100%}button{font:inherit;padding:8px}</style>
<h1>Same frame: 1x / 4x MSAA</h1><p>Only coverage sampling changes. Look at the fingers during 0.5–2.3 s. PNG crops are 1:1; H.264 is lossy.</p>
<button id="swap">Show 4x</button><span id="label"> 1x</span>
<video id="a" controls preload="auto" src="panda-msaa-1x.mp4"></video>
<video id="b" controls preload="auto" src="panda-msaa-4x.mp4" hidden></video>
<script>let current=document.getElementById('a'),other=document.getElementById('b');
document.getElementById('swap').onclick=()=>{const playing=!current.paused,t=current.currentTime;current.pause();
other.currentTime=t;current.hidden=true;other.hidden=false;[current,other]=[other,current];
document.getElementById('label').textContent=current.id==='a'?' 1x':' 4x';
document.getElementById('swap').textContent=current.id==='a'?'Show 4x':'Show 1x';if(playing)current.play();};</script>''', encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--excerpt", type=Path, default=film.ROOT/"tests/fixtures/panda_film/trace.json")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--videos", choices=("none","toggle","split","both"), default="both")
    parser.add_argument("--encode-only", action="store_true")
    args = parser.parse_args()
    if args.encode_only:
        encode_pair(args.output, videos=args.videos)
    elif args.library is None:
        parser.error("--library is required for capture")
    else:
        run_pair(args.output, args.excerpt, args.library, device_index=args.device_index, videos=args.videos)