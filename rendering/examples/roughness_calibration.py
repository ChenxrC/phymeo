"""Controlled R2 roughness comparison using the unmodified official SAPIEN PT.

Only perceptual roughness changes. Images remain untouched; profile averaging
is a declared measurement, not an image denoiser. This demonstrates a renderer
parameter's appearance, not calibration against a measured real material.
"""

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
from time import perf_counter
import zlib
from dataclasses import asdict
from datetime import datetime, timezone


_ROUGHNESS = (0.18, 0.52)


def make_protocol():
    return {
        "schema": "phymeo-r2-roughness-protocol-v1",
        "roughness": list(_ROUGHNESS),
        "geometry": {"shape": "smooth-normal UV sphere", "radius_m": 0.25,
                     "latitude_segments": 64, "longitude_segments": 128,
                     "vertices": 8066, "triangles": 16128, "center_m": [0., 0., 0.]},
        "material": {"linear_base_color": [0.7, 0.7, 0.7], "metallic": 1.0},
        "camera": {"width": 640, "height": 480, "position_m": [0., 0., 1.],
                   "orientation_xyzw": [0., 0., 0., 1.], "fx_px": 500., "fy_px": 500.,
                   "cx_px": 320., "cy_px": 240., "near_m": 0.1, "far_m": 5.},
        "lighting": {"kind": "one official directional light", "direction_of_travel": [-0.6, 0., -0.8],
                     "linear_rgb_illuminance": [0.012]*3, "ambient": [0., 0., 0.],
                     "environment": None, "area_emitters": 0,
                     "reason": "narrow source isolates BRDF width from a large emitter's angular width"},
        "sampling": {"samples_per_pixel": 1024, "max_bounces": 4, "denoiser": "none",
                     "mode": "official-fixed-sequence-fresh-camera", "seed": None,
                     "arbitrary_seed_control": False, "independent_random_trials_claimed": False},
        "display": "exposure=1; clamp linear [0,1]; standard sRGB once; no image edits",
        "metric_domain": "public sRGB8 / 255; equal RGB-channel mean",
        "profile": {"rows": [238, 242], "columns": [192, 448],
                    "aggregation": "mean of four original rows and three channels at each column",
                    "geometry_reason": "camera and directional light lie in XZ; highlight is on y=240",
                    "background_boxes": [[20, 220, 100, 260], [540, 220, 620, 260]],
                    "half_level": "background + (profile peak - background)/2",
                    "crossings": "linear interpolation at both boundaries of the unique above-half component",
                    "saturation_code": 254, "minimum_peak_contrast_codes": 12,
                    "maximum_background_code": 1, "off_band_peak_tolerance_codes": 1},
        "expected_relation": {"rough_052_fwhm_exceeds_018_by_px": 1.0},
        "invalid_reasons": ["saturated", "low_contrast", "multiple_peaks", "unbounded",
                            "off_profile", "nonzero_background"],
        "limitations": ["8-bit display-domain FWHM; not a linear radiance BRDF fit",
                        "same official sampling sequence; no independent-noise confidence interval",
                        "black background leaves unlit metal dark; no auto exposure or fill adjustment"],
    }


def write_sphere(directory):
    from phymeo_rendering.assets import Mesh
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rings, sectors, radius = 64, 128, 0.25
    normals = [(0., 0., 1.)]
    for ring in range(1, rings):
        phi = math.pi*ring/rings
        for sector in range(sectors):
            theta = 2*math.pi*sector/sectors
            normals.append((math.sin(phi)*math.cos(theta), math.sin(phi)*math.sin(theta), math.cos(phi)))
    normals.append((0., 0., -1.))
    positions = [tuple(radius*v for v in n) for n in normals]
    triangles = []
    for sector in range(sectors):
        triangles.append((0, 1+sector, 1+(sector+1) % sectors))
    for ring in range(rings-2):
        upper, lower = 1+ring*sectors, 1+(ring+1)*sectors
        for sector in range(sectors):
            following = (sector+1) % sectors
            a, b, c, d = upper+sector, upper+following, lower+sector, lower+following
            triangles.extend(((a, c, b), (b, c, d)))
    south, last = len(positions)-1, 1+(rings-2)*sectors
    for sector in range(sectors):
        triangles.append((south, last+(sector+1) % sectors, last+sector))
    position_bytes = struct.pack(f"<{3*len(positions)}f", *(v for p in positions for v in p))
    normal_bytes = struct.pack(f"<{3*len(normals)}f", *(v for n in normals for v in n))
    index_bytes = struct.pack(f"<{3*len(triangles)}H", *(v for t in triangles for v in t))
    raw = position_bytes+normal_bytes+index_bytes
    stored = tuple(struct.iter_unpack("<3f", position_bytes))
    document = {
        "asset": {"version": "2.0", "generator": "Phymeo R2 controlled sphere"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "mode": 4}]}],
        "buffers": [{"byteLength": len(raw), "uri": "data:application/octet-stream;base64,"
                     +base64.b64encode(raw).decode("ascii")}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes)},
                        {"buffer": 0, "byteOffset": len(position_bytes), "byteLength": len(normal_bytes)},
                        {"buffer": 0, "byteOffset": len(position_bytes)+len(normal_bytes), "byteLength": len(index_bytes)}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "type": "VEC3", "count": len(positions),
                       "min": [min(p[i] for p in stored) for i in range(3)],
                       "max": [max(p[i] for p in stored) for i in range(3)]},
                      {"bufferView": 1, "componentType": 5126, "type": "VEC3", "count": len(normals)},
                      {"bufferView": 2, "componentType": 5123, "type": "SCALAR", "count": 3*len(triangles)}],
    }
    path = directory / "controlled-sphere.gltf"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, separators=(",", ":"), allow_nan=False)
    return Mesh("controlled-sphere", path.resolve().as_uri())


def make_scene(mesh, roughness):
    from phymeo_rendering.assets import Material
    from phymeo_rendering.scene import DirectionalLight, Instance, VisualScene
    if roughness not in _ROUGHNESS:
        raise ValueError("only the two preregistered roughness values are admitted")
    return VisualScene(f"r2-roughness-{roughness:.2f}", (mesh,),
                       (Material("sphere-metal", (0.7,)*3, 1., roughness),),
                       (Instance(1, mesh.id, "sphere-metal", "sphere"),),
                       (DirectionalLight((-0.6, 0., -0.8), (0.012,)*3),))


def measure_profile(x, values, background):
    x, values = list(x), list(values)
    if (len(x) != len(values) or len(x) < 3 or not math.isfinite(background)
            or not 0 <= background <= 1 or any(not math.isfinite(v) for v in x+values)
            or any(not 0 <= v <= 1 for v in values)
            or any(b <= a for a, b in zip(x, x[1:]))):
        raise ValueError("profile needs increasing finite coordinates and normalized finite intensities")
    peak = max(values)
    half = background+(peak-background)/2
    result = {"status": "invalid", "reasons": [], "background": background, "peak": peak,
              "contrast": peak-background, "half_level": half,
              "profile_x_px": x, "profile_mean_srgb": values}
    if peak >= 254/255:
        result["reasons"].append("saturated")
    if peak-background < 12/255:
        result["reasons"].append("low_contrast")
    spans, begin = [], None
    for index, value in enumerate(values):
        if value >= half and begin is None:
            begin = index
        if begin is not None and (value < half or index == len(values)-1):
            end = index-1 if value < half else index
            spans.append((begin, end))
            begin = None
    if len(spans) != 1:
        result["reasons"].append("multiple_peaks")
    elif spans[0][0] == 0 or spans[0][1] == len(values)-1:
        result["reasons"].append("unbounded")
    if result["reasons"]:
        return result
    left, right = spans[0]
    def crossing(a, b):
        return x[a]+(half-values[a])*(x[b]-x[a])/(values[b]-values[a])
    left_x, right_x = crossing(left-1, left), crossing(right, right+1)
    peak_positions = [position for position, value in zip(x, values) if value == peak]
    peak_x = sum(peak_positions)/len(peak_positions)
    result.update(status="valid", left_crossing_px=left_x, right_crossing_px=right_x,
                  fwhm_px=right_x-left_x, peak_x_px=peak_x,
                  left_half_width_px=peak_x-left_x, right_half_width_px=right_x-peak_x)
    return result


def measure_image(rgb, protocol):
    if not isinstance(rgb, bytes) or len(rgb) != 640*480*3:
        raise ValueError("expected public 640x480 packed RGB8 bytes")
    profile = protocol["profile"]
    top, bottom = profile["rows"]
    left, right = profile["columns"]
    def intensity(u, v):
        offset = (v*640+u)*3
        return sum(rgb[offset:offset+3])/(3*255)
    background = [intensity(u, v) for x0, y0, x1, y1 in profile["background_boxes"]
                  for v in range(y0, y1) for u in range(x0, x1)]
    samples = [sum(intensity(u, v) for v in range(top, bottom))/(bottom-top) for u in range(left, right)]
    result = measure_profile([u+0.5 for u in range(left, right)], samples, sum(background)/len(background))
    # Analytic sphere projection; this mask does not depend on brightness or
    # renderer-generated segmentation, and it includes the complete highlight.
    radius_px = 500*0.25/math.sqrt(1.-0.25**2)
    sphere_max, sphere_samples, saturated = 0., 0, 0
    for v in range(480):
        for u in range(640):
            if (u+0.5-320)**2+(v+0.5-240)**2 < radius_px**2:
                sphere_max = max(sphere_max, intensity(u, v))
                offset = (v*640+u)*3
                saturated += max(rgb[offset:offset+3]) >= profile["saturation_code"]
                sphere_samples += 1
    band_max = max(intensity(u, v) for v in range(top, bottom) for u in range(left, right))
    if saturated and "saturated" not in result["reasons"]:
        result["reasons"].append("saturated")
    if sphere_max-band_max > profile["off_band_peak_tolerance_codes"]/255:
        result["reasons"].append("off_profile")
    if max(background) > profile["maximum_background_code"]/255:
        result["reasons"].append("nonzero_background")
    result.update(profile_rows=list(range(top, bottom)), profile_samples=len(samples),
                  background_pixels=len(background), background_max=max(background),
                  analytic_sphere_pixels=sphere_samples, sphere_peak=sphere_max,
                  profile_band_peak=band_max, saturated_sphere_pixels=saturated,
                  sphere_projection_radius_px=radius_px)
    if result["reasons"]:
        result["status"] = "invalid"
        for key in ("fwhm_px", "left_crossing_px", "right_crossing_px", "peak_x_px",
                    "left_half_width_px", "right_half_width_px"):
            result.pop(key, None)
    return result


def _png(path, rgb):
    def chunk(kind, data):
        return struct.pack(">I", len(data))+kind+data+struct.pack(">I", zlib.crc32(kind+data))
    raw = b"".join(b"\0"+rgb[row*640*3:(row+1)*640*3] for row in range(480))
    path.write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR", struct.pack(">IIBBBBB", 640, 480, 8, 2, 0, 0, 0))
                     +chunk(b"IDAT", zlib.compress(raw))+chunk(b"IEND", b""))


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def run_experiment(output_dir, *, device="cuda:0"):
    from phymeo_rendering.backends.sapien import SapienBackend, SapienConfig
    from phymeo_rendering.cameras import Camera
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath
    from phymeo_rendering.scene import FrameKey, NodePose, Pose, SceneFrame
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    mesh = write_sphere(output / "assets")
    protocol = make_protocol()
    protocol.update(declared_utc=datetime.now(timezone.utc).isoformat(), device=device,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    mesh_sha256=hashlib.sha256((output/"assets"/"controlled-sphere.gltf").read_bytes()).hexdigest())
    scenes = [make_scene(mesh, roughness) for roughness in _ROUGHNESS]
    protocol["scenes"] = [asdict(scene) for scene in scenes]
    # This file exists before construction of any SAPIEN backend or render scene.
    _json(output/"protocol.json", protocol)
    report = {"schema": "phymeo-r2-roughness-result-v1", "status": "running", "captures": [],
              "protocol_sha256": hashlib.sha256((output/"protocol.json").read_bytes()).hexdigest(),
              "scope": "unmodified official PT; two controlled materials; no arbitrary seed or denoising"}
    try:
        for roughness, scene in zip(_ROUGHNESS, scenes, strict=True):
            key = FrameKey(scene.key, "controlled-sphere", 0, 0, 0, "authored")
            frame = SceneFrame(key, 0., (NodePose("sphere", Pose()),))
            camera = Camera("fixed", key, Pose((0., 0., 1.)), 640, 480, 500., 500., 320., 240., 0.1, 5.)
            backend = SapienBackend(SapienConfig(path=RenderPath.PATH_TRACE, device=device,
                                                samples_per_pixel=1024, max_bounces=4, denoiser="none"))
            session = backend.open_scene(scene)
            item = {"roughness": roughness, "frame": asdict(frame), "camera": asdict(camera)}
            try:
                started = perf_counter()
                capture = session.capture(frame, CaptureRequest(camera, RenderPath.PATH_TRACE, (Channel.RGB,)))
                item["capture_wall_s"] = perf_counter()-started
                rgb = capture.planes[0].data
                stem = f"roughness-{round(roughness*100):03d}"
                (output/f"{stem}.rgb").write_bytes(rgb)
                _png(output/f"{stem}.png", rgb)
                item.update(file=f"{stem}.png", raw_file=f"{stem}.rgb",
                            rgb_sha256=hashlib.sha256(rgb).hexdigest(), measurement=measure_image(rgb, protocol))
                item["diagnostics"] = session.diagnostics()
                info = item["diagnostics"]
                if (info["samples_per_pixel"] != 1024 or info["max_bounces"] != 4
                        or info["denoiser"] != "none" or info["seed"] is not None
                        or info["sampling_pattern"]["mode"] != protocol["sampling"]["mode"]):
                    raise RuntimeError("actual sampling technique differs from the prior declaration")
            finally:
                session.close()
                item["closed_diagnostics"] = session.diagnostics()
            report["captures"].append(item)
            _json(output/"summary.json", report)
        narrow, rough = (item["measurement"] for item in report["captures"])
        if narrow["status"] == rough["status"] == "valid":
            difference = rough["fwhm_px"]-narrow["fwhm_px"]
            report["comparison"] = {"width_difference_px": difference,
                "width_ratio_052_over_018": rough["fwhm_px"]/narrow["fwhm_px"],
                "peak_ratio_052_over_018": rough["peak"]/narrow["peak"],
                "rougher_is_wider": difference > protocol["expected_relation"]["rough_052_fwhm_exceeds_018_by_px"]}
            report["status"] = "passed" if report["comparison"]["rougher_is_wider"] else "not_demonstrated"
        else:
            report["status"] = "invalid_measurement"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _json(output/"summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    result = run_experiment(args.output, device=args.device)
    print(json.dumps({"status": result["status"], "summary": str(args.output/"summary.json")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
