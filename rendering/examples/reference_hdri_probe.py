"""Public-camera directional HDRI calibration, with an independent analytic oracle.

The fixture encoder and world-ray math below do not call environment decoding,
cubemap conversion, render projection, or pixel encoding product helpers. Run
only under an explicitly assigned reference-device slot. Output must be new.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
from time import perf_counter
import zlib


_DIRECTIONS = ((1., 0., 0.), (-1., 0., 0.), (0., 1., 0.),
               (0., -1., 0.), (0., 0., 1.), (0., 0., -1.))
_NAMES = ("px", "nx", "py", "ny", "pz", "nz")
_PIXELS = tuple((u, v) for v in (18, 36, 54) for u in (24, 48, 72))
_SENSOR_SCOPE = "independent existing raster session; PT sensors remain unsupported"


def _unit(vector):
    norm = math.hypot(*vector)
    if len(vector) != 3 or not math.isfinite(norm) or norm == 0:
        raise ValueError("expected a finite nonzero xyz direction")
    return tuple(component/norm for component in vector)


def expected_radiance(direction, *, yaw_degrees=0.):
    """Known world field; inverse rotation samples the unrotated panorama."""
    x, y, z = _unit(direction)
    angle = math.radians(yaw_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return (.20+.15*(cosine*x+sine*y),
            .30+.20*(-sine*x+cosine*y), .40+.25*z)


def calibration_hdr():
    """Own flat RGBE fixture at panorama pixel centres, with no external assets."""
    body = bytearray()
    for v in range(256):
        latitude = math.pi/2-math.pi*(v+.5)/256
        for u in range(512):
            longitude = 2*math.pi*((u+.5)/512-.5)
            direction = (math.cos(latitude)*math.cos(longitude),
                         math.cos(latitude)*math.sin(longitude), math.sin(latitude))
            linear = expected_radiance(direction)
            _, exponent = math.frexp(max(linear))
            scale = math.ldexp(1., 8-exponent)
            body.extend((*[int(channel*scale) for channel in linear], exponent+128))
    return b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 256 +X 512\n"+bytes(body)


def protocol():
    return {
        "schema": "phymeo-reference-hdri-protocol-v1",
        "source_sha256": hashlib.sha256(calibration_hdr()).hexdigest(),
        "source_size": [512, 256], "face_size": 128,
        "world_convention": "+Z up; panorama centre +X; longitude increases toward +Y",
        "radiance_formula": "(0.20+0.15*dx,0.30+0.20*dy,0.40+0.25*dz)",
        "linear_radiance_bounds": [.05, .65], "strength": 1.,
        "yaw_degrees": [0., 90.], "view_directions": [list(d) for d in _DIRECTIONS],
        "probe_pixels": [list(p) for p in _PIXELS],
        "background_camera": {"width": 96, "height": 72, "fx_px": 65., "fy_px": 65.,
                              "cx_px": 48., "cy_px": 36., "near_m": .1, "far_m": 5.},
        "rgb8_max_absolute_error": 3,
        "display_oracle": "standard sRGB OETF at exposure 1, half-up UNORM8; no postprocessing",
        "sampling": {"mode": "official", "diagnostic_mode": "official-fixed-sequence-fresh-camera",
                     "samples_per_pixel": 64, "max_bounces": 4, "denoiser": "none", "seed": None},
        "reflection": {"radius_m": .25, "linear_base_color": [.95]*3, "metallic": 1.,
                       "roughness": .01, "width": 160, "height": 120, "focal_px": 160.,
                       "camera_position_m": [0., 0., 1.], "camera_forward": [0., 0., -1.],
                       "roi": [45, 45, 70, 75],
                       "expected_red_mean_delta_90_minus_0_min_codes": 8.,
                       "scope": "same faceted smooth-normal sphere; environmental reflection response"},
        "raster_sensors": {"scope": _SENSOR_SCOPE, "required_relation": "byte-identical",
                           "roi_required_id": 1},
        "limitations": ["authored smooth low-dynamic-range RGBE direction test, not a photographic HDR benchmark",
                        "quantization and cubemap interpolation included in the predeclared 3-code gate",
                        "same installed official sampler, no arbitrary user seed or cross-device guarantee",
                        "PT remains RGB-only; sensor check observes a separate unchanged raster session"],
    }


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _quaternion(right, up, back):
    # Independent scalar rotation-matrix conversion, columns are camera axes.
    matrix = tuple(tuple(column[row] for column in (right, up, back)) for row in range(3))
    trace = sum(matrix[i][i] for i in range(3))
    if trace > 0:
        scale = 2*math.sqrt(1+trace)
        result = ((matrix[2][1]-matrix[1][2])/scale,
                  (matrix[0][2]-matrix[2][0])/scale,
                  (matrix[1][0]-matrix[0][1])/scale, scale/4)
    else:
        i = max(range(3), key=lambda k: matrix[k][k])
        j, k = (i+1) % 3, (i+2) % 3
        scale = 2*math.sqrt(1+matrix[i][i]-matrix[j][j]-matrix[k][k])
        values = [0., 0., 0., (matrix[k][j]-matrix[j][k])/scale]
        values[i] = scale/4
        values[j] = (matrix[j][i]+matrix[i][j])/scale
        values[k] = (matrix[k][i]+matrix[i][k])/scale
        result = tuple(values)
    norm = math.hypot(*result)
    return tuple(value/norm for value in result)


def view_camera(direction, *, key=None):
    from phymeo_rendering.cameras import Camera
    from phymeo_rendering.scene import FrameKey, Pose
    forward = _unit(direction)
    nominal_up = (0., 1., 0.) if abs(forward[2]) > .99 else (0., 0., 1.)
    right = _unit(_cross(forward, nominal_up))
    up = _unit(_cross(right, forward))
    back = tuple(-value for value in forward)
    if key is None:
        key = FrameKey("hdri-calibration", "directions", 0, 0, 0, "authored")
    camera = Camera("world-direction", key, Pose((0., 0., 0.), _quaternion(right, up, back)),
                    96, 72, 65., 65., 48., 36., .1, 5.)
    return camera, {"right": right, "up": up, "forward": forward}


def world_ray(camera, basis, u, v):
    x, y = (u+.5-camera.cx_px)/camera.fx_px, (camera.cy_px-v-.5)/camera.fy_px
    return _unit(tuple(f+x*r+y*t for f, r, t in
                       zip(basis["forward"], basis["right"], basis["up"], strict=True)))


def _srgb8(linear):
    display = 12.92*linear if linear <= .0031308 else 1.055*linear**(1/2.4)-.055
    return int(math.floor(255*max(0., min(1., display))+.5))


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def _png(path, rgb, width, height):
    def chunk(kind, data):
        return struct.pack(">I", len(data))+kind+data+struct.pack(">I", zlib.crc32(kind+data))
    raw = b"".join(b"\0"+rgb[row*width*3:(row+1)*width*3] for row in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     +chunk(b"sRGB", b"\0")+chunk(b"IDAT", zlib.compress(raw))+chunk(b"IEND", b""))


def _save_capture(output, stem, capture):
    from phymeo_rendering.outputs import Channel
    record = {"frame_key": asdict(capture.frame_key), "time_s": capture.time_s,
              "camera": asdict(capture.camera), "planes": []}
    for plane in capture.planes:
        filename = f"{stem}.{plane.channel.name.lower()}.raw"
        (output/filename).write_bytes(plane.data)
        record["planes"].append({"channel": plane.channel.value, "file": filename,
                                 "bytes": len(plane.data), "sha256": hashlib.sha256(plane.data).hexdigest()})
        if plane.channel is Channel.RGB:
            record["image"] = f"{stem}.png"
            _png(output/record["image"], plane.data, capture.camera.width, capture.camera.height)
    return record


def _check_sampling(info, declaration):
    actual = info["sampling_pattern"]
    if (actual["mode"] != declaration["diagnostic_mode"]
            or info["samples_per_pixel"] != declaration["samples_per_pixel"]
            or info["max_bounces"] != declaration["max_bounces"]
            or info["denoiser"] != declaration["denoiser"] or info["seed"] is not None):
        raise RuntimeError("actual reference sampling differs from the predeclared protocol")


def _check_closed(session):
    info = session.diagnostics()
    lifecycle = info["lifecycle"]
    if (not lifecycle["closed"] or lifecycle["attached_entities_after_clear"] != 0
            or lifecycle["studio_temporary_files_exist"]):
        raise RuntimeError("reference resources remain attached after scene close")
    return info


def run_probe(output_dir, *, device="cuda:0"):
    from phymeo_rendering.assets import Material
    from phymeo_rendering.backends.sapien import SapienBackend, SapienConfig, SapienStudio
    from phymeo_rendering.cameras import Camera
    from phymeo_rendering.environment import RadianceEnvironment
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath
    from phymeo_rendering.scene import FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = calibration_hdr()
    (output/"calibration.hdr").write_bytes(source)
    helper_path = Path(__file__).with_name("roughness_calibration.py")
    spec = importlib.util.spec_from_file_location("controlled_roughness_fixture", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    sphere = helper.write_sphere(output/"assets")
    sphere_scene = VisualScene("hdri-mirror", (sphere,),
                              (Material("mirror", (.95,)*3, 1., .01),),
                              (Instance(1, sphere.id, "mirror", "sphere"),))
    sphere_key = FrameKey(sphere_scene.key, "mirror", 0, 0, 0, "authored")
    sphere_frame = SceneFrame(sphere_key, 0., (NodePose("sphere", Pose()),))
    sphere_camera = Camera("mirror", sphere_key, Pose((0., 0., 1.)),
                           160, 120, 160., 160., 80., 60., .1, 5.)
    empty = VisualScene("hdri-calibration", (), (), ())
    declaration = protocol()
    declaration.update(device=device, declared_utc=datetime.now(timezone.utc).isoformat(),
                       probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       sphere_helper_sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest(),
                       sphere_mesh_sha256=hashlib.sha256((output/"assets"/"controlled-sphere.gltf").read_bytes()).hexdigest(),
                       scenes=[asdict(empty), asdict(sphere_scene)])
    # Persist intended parameters and assets before constructing any device.
    _json(output/"protocol.json", declaration)
    report = {"schema": "phymeo-reference-hdri-result-v1", "status": "running",
              "protocol_sha256": hashlib.sha256((output/"protocol.json").read_bytes()).hexdigest(),
              "calibration": {"captures": [], "exposures": 0, "compared_pixels": 0, "max_channel_error": 0},
              "reflection": {"captures": [], "changed": False},
              "raster_sensors": {"scope": _SENSOR_SCOPE, "unchanged": False},
              "closed_sessions": []}
    raster = None
    try:
        raster = SapienBackend(SapienConfig(path=RenderPath.RASTER, device=device)).open_scene(sphere_scene)
        sensor_request = CaptureRequest(sphere_camera, RenderPath.RASTER,
                                        (Channel.FORWARD_DEPTH, Channel.INSTANCE_ID, Channel.VALID))
        before = raster.capture(sphere_frame, sensor_request)
        report["raster_sensors"]["before"] = _save_capture(output, "sensors-before", before)
        sensor_data = {plane.channel: plane.data for plane in before.planes}
        x0, y0, x1, y1 = declaration["reflection"]["roi"]
        for v in range(y0, y1):
            for u in range(x0, x1):
                index = v*sphere_camera.width+u
                if (sensor_data[Channel.VALID][index] != 1
                        or struct.unpack_from("<I", sensor_data[Channel.INSTANCE_ID], 4*index)[0] != 1):
                    raise RuntimeError("predeclared mirror ROI is not entirely on sphere ID 1")
        reflection_images = []
        for yaw in declaration["yaw_degrees"]:
            environment = RadianceEnvironment((output/"calibration.hdr").as_uri(),
                                               declaration["source_sha256"], strength=1.,
                                               yaw_degrees=yaw, face_size=declaration["face_size"])
            config = SapienConfig(path=RenderPath.PATH_TRACE, device=device, samples_per_pixel=64,
                                  max_bounces=4, denoiser="none", sampling_mode="official",
                                  studio=SapienStudio(panel_strength=0., environment_map=environment))
            session = SapienBackend(config).open_scene(empty)
            try:
                for index, (name, direction) in enumerate(zip(_NAMES, _DIRECTIONS, strict=True)):
                    key = FrameKey(empty.key, "directions", 0, index, index, "authored")
                    frame = SceneFrame(key, float(index), ())
                    camera, basis = view_camera(direction, key=key)
                    start = perf_counter()
                    capture = session.capture(frame, CaptureRequest(camera, RenderPath.PATH_TRACE, (Channel.RGB,)))
                    elapsed = perf_counter()-start
                    item = _save_capture(output, f"axis-{name}-yaw-{int(yaw):03d}", capture)
                    item.update(yaw_degrees=yaw, direction_world=list(direction), capture_wall_s=elapsed,
                                diagnostics=session.diagnostics(), probes=[])
                    _check_sampling(item["diagnostics"], declaration["sampling"])
                    recorded = item["diagnostics"]["studio"]["environment"]
                    if (recorded["source_sha256"] != environment.sha256 or recorded["strength"] != 1.
                            or recorded["yaw_degrees"] != yaw or recorded["face_count"] != 6):
                        raise RuntimeError("actual environment is not bound to this source/yaw")
                    rgb = capture.planes[0].data
                    for u, v in _PIXELS:
                        ray = world_ray(camera, basis, u, v)
                        expected = tuple(_srgb8(value) for value in expected_radiance(ray, yaw_degrees=yaw))
                        offset = (v*camera.width+u)*3
                        actual = tuple(rgb[offset:offset+3])
                        error = max(abs(a-b) for a, b in zip(actual, expected, strict=True))
                        item["probes"].append({"u": u, "v": v, "ray_world": list(ray),
                                               "expected_rgb8": expected, "actual_rgb8": actual,
                                               "max_channel_error": error})
                        report["calibration"]["compared_pixels"] += 1
                        report["calibration"]["max_channel_error"] = max(report["calibration"]["max_channel_error"], error)
                    report["calibration"]["captures"].append(item)
                    report["calibration"]["exposures"] += 1
                    _json(output/"summary.json", report)
            finally:
                session.close()
                report["closed_sessions"].append(_check_closed(session))
            session = SapienBackend(config).open_scene(sphere_scene)
            try:
                capture = session.capture(sphere_frame, CaptureRequest(sphere_camera, RenderPath.PATH_TRACE, (Channel.RGB,)))
                item = _save_capture(output, f"mirror-yaw-{int(yaw):03d}", capture)
                item.update(yaw_degrees=yaw, diagnostics=session.diagnostics())
                _check_sampling(item["diagnostics"], declaration["sampling"])
                image = capture.planes[0].data
                roi_red = [image[(v*sphere_camera.width+u)*3] for v in range(y0, y1) for u in range(x0, x1)]
                item.update(roi_red_mean=sum(roi_red)/len(roi_red), roi_pixels=len(roi_red))
                report["reflection"]["captures"].append(item)
                reflection_images.append(image)
            finally:
                session.close()
                report["closed_sessions"].append(_check_closed(session))
        after = raster.capture(sphere_frame, sensor_request)
        report["raster_sensors"].update(after=_save_capture(output, "sensors-after", after),
                                        diagnostics=raster.diagnostics(),
                                        unchanged=before == after)
        base, rotated = report["reflection"]["captures"]
        difference = rotated["roi_red_mean"]-base["roi_red_mean"]
        report["reflection"].update(red_mean_delta_90_minus_0=difference,
                                     changed=(reflection_images[0] != reflection_images[1]
                                              and difference >= declaration["reflection"]["expected_red_mean_delta_90_minus_0_min_codes"]))
        passed = (report["calibration"]["exposures"] == 12
                  and report["calibration"]["compared_pixels"] == 108
                  and report["calibration"]["max_channel_error"] <= declaration["rgb8_max_absolute_error"]
                  and report["reflection"]["changed"] and report["raster_sensors"]["unchanged"])
        report["status"] = "passed" if passed else "not_demonstrated"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if raster is not None:
                raster.close()
                report["closed_sessions"].append(_check_closed(raster))
        except Exception as cleanup_error:
            report.update(status="failed", cleanup_error=f"{type(cleanup_error).__name__}: {cleanup_error}")
            raise
        finally:
            _json(output/"summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    report = run_probe(args.output, device=args.device)
    print(json.dumps({"status": report["status"], "summary": str(args.output.resolve()/"summary.json")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
