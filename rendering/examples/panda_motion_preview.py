"""Public, synchronous Panda motion preview with named, inspectable presets.

Examples (rendering/src must be on PYTHONPATH):
  python rendering/examples/panda_motion_preview.py capture --library DLL --output NEW --quick
  python rendering/examples/panda_motion_preview.py capture --library DLL --output NEW --presets preview720 quality720 direct720 fast360
  python rendering/examples/panda_motion_preview.py encode NEW --media-output NEW_MEDIA

No simulation is run. The sealed 300-state episode is replayed unchanged.
Capture completion includes CPU-owned requested planes; this is not a GPU-only
present loop. Diagnostic six-pose movies are montages, not continuous motion.
"""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

SCHEMA = "phymeo-panda-motion-preview-v1"
NAMES = ("preview720", "quality720", "direct720", "fast540", "fast360", "raw720")
WARMUP = 10
QUICK_WARMUP = 2
STOP_BATCHES = (1, 2, 4, 8, 16)
FPS = 30


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def preset(name):
    require(name in NAMES, "unknown named preview preset")
    settings = dict(resident=True, shadow_mode="ray_query", area_light_grid=2,
                    msaa_samples=1, reflection_mode="single_bounce",
                    indirect_mode="diffuse_bounce", indirect_seed=0,
                    indirect_bounces=1, progressive=True, spatial_filter="guided_settling")
    extent = [1280, 720]
    labels = dict(preview720="720p 单跳运动预览", quality720="720p 两跳检查",
                  direct720="720p 直接灯＋HDR IBL＋近场镜面，无漫反射追踪",
                  fast540="540p 单跳低分辨率预览", fast360="360p 单跳低分辨率预览",
                  raw720="720p 单跳未滤波对照")
    if name == "quality720":
        settings["indirect_bounces"] = 2
    elif name == "direct720":
        settings.update(indirect_mode="none", progressive=False, spatial_filter="none")
    elif name == "raw720":
        settings["spatial_filter"] = "none"
    elif name == "fast540":
        extent = [960, 540]
    elif name == "fast360":
        extent = [640, 360]
    return dict(name=name, label=labels[name], extent=extent, settings=settings,
                exposure=8., display_transform="pbr-neutral", normal_map=True,
                eligible_for_720p_claim=extent == [1280, 720],
                resolution_scope="native pixel grid; no upscaling or image recoloring")


def pose_indices(quick=False):
    return (0, 59, 60, 149, 239, 299) if quick else tuple(range(300))


def timing_summary(times, *, full):
    require(times and all(isinstance(t, (int, float)) and not isinstance(t, bool)
                          and math.isfinite(t) and t > 0 for t in times), "invalid capture times")
    require(not full or len(times) == 300, "full acceptance requires 300 completed captures")
    ranked = sorted(times)
    n = len(times)
    mean = sum(times) / n
    p95 = ranked[math.ceil(.95 * n) - 1]
    hits = sum(t <= 33.3 for t in times)
    return dict(count=n, mean_ms=mean, p50_ms=ranked[math.ceil(.5 * n) - 1],
                p95_ms=p95, max_ms=ranked[-1], completed_fps_from_mean=1000 / mean,
                within_33_3_ms=hits, within_33_3_fraction=hits/n,
                slow_frame_ordinals=[i for i, t in enumerate(times) if t > 33.3],
                latency_gate_passed=p95 <= 33.3 if full else None,
                scope="300 completed synchronous captures" if full else
                      "six-pose diagnostic; not 300-frame acceptance")


def verify_capture(shot, frame, request):
    require(shot.frame_key == frame.key and shot.time_s == frame.time_s
            and shot.camera == request.camera, "capture frame/time/camera binding differs")
    require(tuple(p.channel for p in shot.planes) == request.channels,
            "capture channel order differs")


def profile_delta(before, after, count):
    require(after["captures"] - before["captures"] == count and
            after["timestamp_samples"] - before["timestamp_samples"] == count,
            "native profile capture/timestamp count mismatch")
    keys = [k for k in before if k.endswith("_ns") and k != "timestamp_period_ns"]
    delta = {k: after[k] - before[k] for k in keys}
    require(all(v >= 0 for v in delta.values()), "native profile counters went backwards")
    return dict(total_ns=delta, mean_ms={k: v / count / 1e6 for k, v in delta.items()},
                captures=count, scope="GPU time overlaps CPU wait; do not add them")


def verify_backend(diag, declaration):
    s = declaration["settings"]
    require(diag["shadow_mode"] == s["shadow_mode"] and
            diag["area_light_grid"] == s["area_light_grid"] and
            diag["msaa"]["actual_color_samples"] == 1 and
            diag["reflection"]["mode"] == s["reflection_mode"] and
            bool(diag["normal_maps"]), "backend geometry/light/material selection differs")
    require(diag["indirect"]["mode"] == s["indirect_mode"] and
            diag["spatial_filter"]["mode"] == s["spatial_filter"], "backend indirect/filter selection differs")
    if s["progressive"]:
        h = diag["progressive"]
        require(h["enabled"] and diag["indirect"]["bounces"] == s["indirect_bounces"]
                and diag["indirect"]["samples_per_pixel"] == 16
                and h["primary_samples_per_pixel"] == 16 * h["batches"],
                "actual bounce/sample/history declaration differs")
    else:
        require(not diag.get("progressive") and diag["indirect"]["samples_per_pixel"] == 0,
                "direct-only control unexpectedly accumulated indirect light")


def verify_close(device):
    telemetry = device["resident_telemetry"]
    require(device["closed"] and all(device[k] == 0 for k in
            ("live_contexts", "live_frames", "live_resources", "validation_errors"))
            and telemetry["final_after_destroy"]
            and all(telemetry[k] == 0 for k in ("device_local_bytes_current", "host_visible_bytes_current")),
            "native resource/validation close check failed")


def compare_runs(runs):
    seen = set()
    for r in runs:
        key = (r["preset"], r["channels"])
        require(key not in seen, "duplicate preset/channel run")
        seen.add(key)
        require(r["status"] == "complete", "cannot compare an incomplete run")
    rgb_pairs = 0
    sensor_pairs = 0
    identity = ("pose_index", "frame_key", "time_s", "state_sha256", "camera_sha256")
    for i, a in enumerate(runs):
        for b in runs[i+1:]:
            same_preset = a["preset"] == b["preset"]
            same_extent = a["extent"] == b["extent"]
            if not same_extent:
                continue
            require(len(a["frames"]) == len(b["frames"]), "matched runs have different frame counts")
            for x, y in zip(a["frames"], b["frames"], strict=True):
                require(all(x[k] == y[k] for k in identity), "matched input state/camera differs")
                if same_preset:
                    require(x["rgb_sha256"] == y["rgb_sha256"], "RGB differs between RGB/all requests")
                if a["channels"] == b["channels"] == "all":
                    require(x["sensors"] == y["sensors"], "same-extent geometry channels differ")
            rgb_pairs += int(same_preset)
            sensor_pairs += int(a["channels"] == b["channels"] == "all")
    return dict(rgb_vs_all_exact=True if rgb_pairs else None, rgb_pair_count=rgb_pairs,
                same_extent_sensors_exact=True if sensor_pairs else None,
                sensor_pair_count=sensor_pairs,
                limitations="different resolutions have different sensor grids; no cross-resolution byte equality or quality claim")


def make_inputs(scene, frames, declaration, camera):
    import panda_realtime_film as film
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath
    require(len(frames) == 300, "expected full sealed episode")
    result = {}
    for i, source in enumerate(frames):
        f = replace(source, key=replace(source.key, scene_key=scene.key))
        c = film.exposure_camera(SimpleNamespace(camera=camera), f, i, len(frames),
                                 *declaration["extent"])
        result[i] = (f, CaptureRequest(c, RenderPath.RASTER, (Channel.RGB,)))
    return result


def compact_history(diag):
    return {key: diag.get(key) for key in ("progressive", "spatial_filter", "indirect",
                                          "environment_sampling", "last_passes")}


def save_pngs(root, extent, buffered, records):
    from PIL import Image
    started = perf_counter()
    for ordinal, rgb in buffered:
        path = root / "frames" / f"{ordinal:04d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.frombytes("RGB", tuple(extent), rgb).save(path, compress_level=2)
        records[ordinal].update(png=f"frames/{ordinal:04d}.png", png_sha256=digest(path))
    return (perf_counter() - started) * 1000


def stopped_samples(session, item, count, folder):
    from phymeo_rendering.outputs import Channel
    from PIL import Image
    frame, request = item
    request = replace(request, channels=(Channel.RGB,))
    session.reset_accumulation()
    result = dict(scope="same owner and final motion pose; explicit history reset, no state injection",
                  pose_key=asdict(frame.key), time_s=frame.time_s, samples=[])
    folder.mkdir()
    for n in range(1, count + 1):
        start = perf_counter()
        shot = session.capture(frame, request)
        wall = (perf_counter() - start) * 1000
        verify_capture(shot, frame, request)
        diag = session.diagnostics()
        require(diag["progressive"]["batches"] == n, "stopped-pose accumulation did not progress")
        row = dict(batch=n, capture_ms=wall, history=compact_history(diag),
                   rgb_sha256=sha256(shot.planes[0].data).hexdigest())
        if n in STOP_BATCHES:
            path = folder / f"batch-{n:04d}.png"
            Image.frombytes("RGB", (request.camera.width, request.camera.height),
                            shot.planes[0].data).save(path)
            row.update(png=path.name, png_sha256=digest(path))
        result["samples"].append(row)
    return result


def capture_run(args, scene, studio, inputs, declaration, channels, folder, *, save_all):
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.outputs import Channel
    indices = pose_indices(args.quick)
    warm = QUICK_WARMUP if args.quick else WARMUP
    requested = (Channel.RGB,) if channels == "rgb" else tuple(Channel)
    folder.mkdir(parents=True)
    report = dict(preset=declaration["name"], channels=channels, extent=declaration["extent"],
                  status="opening", frames=[], warmup_ms=[], declaration=declaration,
                  order_index=args.order_index, failure=None)
    buffered = []
    owner = None
    failure = None
    try:
        start = perf_counter()
        owner = NativeBackend(library_path=args.library, shader_directory=args.shader_directory,
                              device_index=args.device_index, studio=studio,
                              exposure=declaration["exposure"],
                              display_transform=declaration["display_transform"],
                              validation=args.validation, **declaration["settings"]).open_scene(scene)
        report["open_ms"] = (perf_counter() - start) * 1000
        for i in range(warm):
            frame, req = inputs[i]
            req = replace(req, channels=requested)
            start = perf_counter()
            shot = owner.capture(frame, req)
            report["warmup_ms"].append((perf_counter() - start) * 1000)
            verify_capture(shot, frame, req)
        if declaration["settings"]["progressive"]:
            owner.reset_accumulation()
        before = owner.diagnostics()["device"]["resident_profile"].copy()
        last_profile = before
        loop_start = perf_counter()
        for ordinal, i in enumerate(indices):
            frame, req = inputs[i]
            req = replace(req, channels=requested)
            start = perf_counter()
            shot = owner.capture(frame, req)
            elapsed = (perf_counter() - start) * 1000
            verify_capture(shot, frame, req)
            diag = owner.diagnostics()
            verify_backend(diag, declaration)
            planes = {p.channel: p.data for p in shot.planes}
            profile = diag["device"]["resident_profile"]
            row = dict(pose_index=i, frame_key=asdict(frame.key), time_s=frame.time_s,
                       state_sha256=sha256(canonical(asdict(frame))).hexdigest(),
                       camera_sha256=sha256(canonical(asdict(req.camera))).hexdigest(),
                       capture_ms=elapsed, rgb_sha256=sha256(planes[Channel.RGB]).hexdigest(),
                       sensors={c.value: sha256(data).hexdigest() for c, data in planes.items()
                                if c != Channel.RGB},
                       public_bytes=sum(len(v) for v in planes.values()),
                       native_profile=profile_delta(last_profile, profile, 1),
                       history=compact_history(diag))
            report["frames"].append(row)
            last_profile = profile.copy()
            if save_all or ordinal in (0, len(indices) // 2, len(indices) - 1):
                buffered.append((ordinal, planes[Channel.RGB]))
        report["consumer_loop_ms"] = (perf_counter() - loop_start) * 1000
        report["consumer_loop_scope"] = "capture + binding/hashes/diagnostics; excludes PNG/encoding"
        report["backend"] = owner.diagnostics()
        report["profile"] = profile_delta(before, last_profile, len(indices))
        report["timing"] = timing_summary([r["capture_ms"] for r in report["frames"]], full=not args.quick)
        report["cold_first_capture_ms"] = report["warmup_ms"][0]
        report["resident_preview_scope"] = "synchronous CPU-owned planes; no GPU presentation or async queue"
        expected = warm + len(indices)
        dev = report["backend"]["device"]
        require(dev["resident_upload_calls"] == 1 and dev["resident_captures"] == expected,
                "owner upload/capture counters disagree")
        # Stopped-pose work is deliberately after the measured motion and separately counted.
        if args.settle_batches and channels == "rgb" and declaration["settings"]["progressive"]:
            report["stopped"] = stopped_samples(owner, inputs[indices[-1]], args.settle_batches,
                                                  folder / "stopped")
            expected += args.settle_batches
        report["final_backend"] = owner.diagnostics()
        require(report["final_backend"]["device"]["resident_captures"] == expected,
                "final owner capture count differs")
        report["status"] = "complete"
    except BaseException as error:
        failure = error
        report.update(status="failed", failure=dict(type=type(error).__name__, message=str(error)))
    finally:
        if owner is not None:
            try:
                owner.close()
                closed = owner.diagnostics()["device"]
                report["closed"] = closed
                verify_close(closed)
            except BaseException as error:
                report["close_error"] = dict(type=type(error).__name__, message=str(error))
                if failure is None:
                    failure = error
                    report["status"] = "failed"
        report["png_save_ms"] = save_pngs(folder, declaration["extent"], buffered, report["frames"])
        save(folder / "receipt.json", report)
    if failure is not None:
        raise failure
    return report


def capture(args):
    import panda_render_performance as bench
    import phymeo_rendering
    from phymeo_rendering.backends import native
    args.library = args.library.resolve(strict=True)
    args.shader_directory = (args.shader_directory or args.library.parent).resolve(strict=True)
    args.output = args.output.resolve()
    require(len(set(args.presets)) == len(args.presets) and len(set(args.channels)) == len(args.channels),
            "duplicate requested preset or channel mode")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, status="preparing", quick=args.quick,
                  source=dict(script=str(Path(__file__).resolve()), sha256=digest(__file__),
                              package=str(Path(phymeo_rendering.__file__).resolve()),
                              native_python=str(Path(native.__file__).resolve()),
                              native_python_sha256=digest(native.__file__),
                              fixture_builder=str(Path(bench.__file__).resolve()),
                              fixture_builder_sha256=digest(bench.__file__),
                              library=str(args.library), library_sha256=digest(args.library),
                              shaders={p.name: digest(p) for p in sorted(args.shader_directory.glob("*.spv"))}),
                  platform=dict(system=platform.platform(), python=sys.version, device_index=args.device_index),
                  camera=args.camera, validation=args.validation, nominal_playback_fps=FPS,
                  trajectory_scope="six registered diagnostic samples" if args.quick else "all 300 source states in order",
                  quality_acceptance="not evaluated: no independent high-sample/reference ROI gate in this runner",
                  realtime_claim=False, runs=[], requested_presets=args.presets, requested_channels=args.channels)
    save(args.output / "summary.json", report)
    try:
        with TemporaryDirectory(prefix="phymeo-motion-preview-") as temp:
            scenes, frames, studio, trace = bench.bind_scene(Path(temp) / "geometry")
            scene = scenes[1]
            declarations = [preset(name) for name in args.presets]
            all_inputs = {p["name"]: make_inputs(scene, frames, p, args.camera) for p in declarations}
            binding = dict(scene=asdict(scene), studio=asdict(studio),
                           source_trace=dict(file=str(trace), sha256=digest(trace)),
                           declarations=declarations, selected_indices=pose_indices(args.quick),
                           warmup=QUICK_WARMUP if args.quick else WARMUP,
                           actual_full_inputs={name: [dict(pose_index=i, frame=asdict(f),
                               camera=asdict(req.camera)) for i, (f, req) in rows.items()]
                               for name, rows in all_inputs.items()},
                           scene_uri_scope="generated geometry lived through all captures; content hashes are in backend receipts",
                           order=[dict(preset=p["name"], channels=c) for p in declarations for c in args.channels])
            save(args.output / "inputs.json", binding)
            report["inputs_sha256"] = digest(args.output / "inputs.json")
            report["source_trace_sha256"] = digest(trace)
            report["status"] = "capturing"
            for p in declarations:
                for channel in args.channels:
                    args.order_index = len(report["runs"])
                    folder = args.output / p["name"] / channel
                    row = capture_run(args, scene, studio, all_inputs[p["name"]], p, channel, folder,
                                      save_all=channel == "rgb" or "rgb" not in args.channels)
                    row["receipt"] = str((folder / "receipt.json").relative_to(args.output))
                    row["receipt_sha256"] = digest(folder / "receipt.json")
                    report["runs"].append(row)
                    save(args.output / "summary.json", report)
                    print(json.dumps(dict(preset=p["name"], channels=channel, timing=row["timing"]),
                                     ensure_ascii=False), flush=True)
            report["comparisons"] = compare_runs(report["runs"])
            rgb = {r["preset"]: r for r in report["runs"] if r["channels"] == "rgb"}
            if "preview720" in rgb:
                baseline = rgb["preview720"]["timing"]
                report["speed_comparison"] = {name: dict(
                    mean_speedup=baseline["mean_ms"]/row["timing"]["mean_ms"],
                    p95_speedup=baseline["p95_ms"]/row["timing"]["p95_ms"],
                    actually_faster_on_both=row["timing"]["mean_ms"] < baseline["mean_ms"] and
                                           row["timing"]["p95_ms"] < baseline["p95_ms"],
                    resolution=row["extent"]) for name, row in rgb.items() if name != "preview720"}
            report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", failure=dict(type=type(error).__name__, message=str(error)))
        save(args.output / "summary.json", report)
        raise
    save(args.output / "summary.json", report)
    return report


def _run(command):
    return subprocess.run([str(x) for x in command], check=True, capture_output=True, text=True).stdout


def encode(args):
    from PIL import Image
    from panda_msaa_compare import verify_probe
    from _panda_movie_encode import _faststart
    root = args.result.resolve(strict=True)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8-sig"))
    require(summary["schema"] == SCHEMA and summary["status"] == "complete", "capture is not complete")
    require(digest(root / "inputs.json") == summary["inputs_sha256"], "input binding changed")
    destination = (args.media_output or root / "media").resolve()
    destination.mkdir(parents=True, exist_ok=False)
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    require(all(tools.values()), "ffmpeg and ffprobe are required; capture PNGs remain reusable")
    media = dict(schema="phymeo-panda-motion-preview-media-v1",
                 summary_sha256=digest(root / "summary.json"), movies=[])
    try:
        for name in summary["requested_presets"]:
            rows = [r for r in summary["runs"] if r["preset"] == name]
            row = next((r for r in rows if r["channels"] == "rgb"), rows[0])
            receipt = root / row["receipt"]
            require(digest(receipt) == row["receipt_sha256"], "receipt changed")
            frames = row["frames"]
            count = len(frames)
            width, height = row["extent"]
            for f in frames:
                path = receipt.parent / f["png"]
                require(path.resolve().is_relative_to(root) and digest(path) == f["png_sha256"], "PNG identity differs")
                with Image.open(path) as image:
                    require(image.mode == "RGB" and image.size == (width, height) and
                            sha256(image.tobytes()).hexdigest() == f["rgb_sha256"],
                            "PNG pixels differ from capture")
            suffix = "-diagnostic" if summary["quick"] else ""
            target = destination / f"panda-{name}{suffix}.mp4"
            filters = ("setsar=1,scale=in_range=full:out_range=limited:out_color_matrix=bt709:"
                       "flags=bicubic+accurate_rnd+full_chroma_int,format=yuv420p,"
                       "setparams=range=limited:color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709")
            cmd = [tools["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                   "-framerate", str(FPS), "-start_number", "0", "-i",
                   str(receipt.parent / "frames/%04d.png"), "-frames:v", str(count),
                   "-an", "-vf", filters, "-c:v", "libx264", "-preset", "medium",
                   "-crf", "18", "-pix_fmt", "yuv420p", "-fps_mode", "cfr",
                   "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709",
                   "-color_trc", "iec61966-2-1", "-movflags", "+faststart", str(target)]
            save(target.with_suffix(".command.json"), cmd)
            _run(cmd)
            probe = json.loads(_run([tools["ffprobe"], "-v", "error", "-count_frames",
                                    "-show_streams", "-show_format", "-show_frames", "-of", "json", target]))
            save(target.with_suffix(".probe.json"), probe)
            verified = verify_probe(probe, width=width, height=height, count=count, fps=FPS)
            _run([tools["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin", "-xerror",
                  "-i", target, "-map", "0:v:0", "-f", "null", "-"])
            require(_faststart(target), "MP4 moov is not before media data")
            media["movies"].append(dict(file=target.name, sha256=digest(target),
                                        verification=verified, full_decode=True,
                                        actual_extent=[width, height], source_png_sha256=[f["png_sha256"] for f in frames],
                                        playback_scope=summary["trajectory_scope"],
                                        timing_scope="nominal playback cadence; not measured rendering throughput"))
            save(destination / "media.json", media)
    except BaseException as error:
        media["failure"] = dict(type=type(error).__name__, message=str(error))
        save(destination / "media.json", media)
        raise
    return media


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--library", type=Path, required=True)
    cap.add_argument("--shader-directory", type=Path)
    cap.add_argument("--device-index", type=int, default=1)
    cap.add_argument("--output", type=Path, required=True)
    cap.add_argument("--quick", action="store_true")
    cap.add_argument("--presets", nargs="+", choices=NAMES, default=["preview720"])
    cap.add_argument("--channels", nargs="+", choices=("rgb", "all"), default=["rgb", "all"])
    cap.add_argument("--camera", choices=("fixed", "wide", "detail", "sequence"), default="detail")
    cap.add_argument("--settle-batches", type=int, choices=(0, 16), default=16)
    cap.add_argument("--validation", action="store_true")
    enc = sub.add_parser("encode")
    enc.add_argument("result", type=Path)
    enc.add_argument("--media-output", type=Path,
                     help="new destination; use a new directory after a media failure, without recapturing")
    args = parser.parse_args()
    capture(args) if args.command == "capture" else encode(args)


if __name__ == "__main__":
    main()
