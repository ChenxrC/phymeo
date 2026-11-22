"""Controlled public SAPIEN display comparison; native rendering is not involved.

A read-only tap saves the real HDR camera buffer supplied to the display encoder.
It returns the original encoder's output unchanged. Independent scalar checks in
test_reference_neutral audit these saved buffers against the public RGB bytes.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def run_calibration(output_dir, *, device="cuda:0"):
    import numpy as np
    from phymeo_rendering.assets import Material
    from phymeo_rendering.backends import _sapien_session
    from phymeo_rendering.backends.sapien import SapienBackend, SapienConfig, SapienStudio
    from phymeo_rendering.cameras import Camera
    from phymeo_rendering.environment import RadianceEnvironment
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath
    from phymeo_rendering.scene import DirectionalLight, FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).parent
    hdri = _module("neutral_hdr_fixture", here/"reference_hdri_probe.py")
    sphere_helper = _module("neutral_sphere_fixture", here/"roughness_calibration.py")
    raw = hdri.calibration_hdr()
    environment_file = output/"calibration.hdr"
    environment_file.write_bytes(raw)
    environment_sha = hashlib.sha256(raw).hexdigest()
    sphere = sphere_helper.write_sphere(output/"assets")
    material = Material("white-shell", (.8,)*3, 0., .35)
    scene = VisualScene("neutral-display", (sphere,), (material,),
                        (Instance(1, sphere.id, material.id, "sphere"),),
                        (DirectionalLight((0., 0., -1.), (1.,)*3),))
    key = FrameKey(scene.key, "display-calibration", 0, 0, 0, "authored")
    frame = SceneFrame(key, 0., (NodePose("sphere", Pose()),))
    camera = Camera("fixed", key, Pose((0., 0., 1.)), 160, 120, 160., 160., 80., 60., .1, 5.)
    declaration = {
        "schema": "phymeo-reference-neutral-protocol-v1", "declared_utc": datetime.now(timezone.utc).isoformat(),
        "exposure": 8., "display_transforms": ["clip", "pbr-neutral"], "device": device,
        "scene": asdict(scene), "frame": asdict(frame), "camera": asdict(camera),
        "pt_environment": {"source_sha256": environment_sha, "strength": 1., "yaw_degrees": 0., "face_size": 128},
        "sampling": {"samples_per_pixel": 64, "max_bounces": 4, "mode": "official", "denoiser": "none"},
        "foreground_roi": [60, 40, 100, 80], "minimum_recovered_foreground_pixels": 64,
        "recovered_definition": "all clip RGB channels 255; all neutral RGB channels below 255, in fixed sphere-interior ROI",
        "raw_readback_relation": "exactly identical for the two display choices at each render path",
        "raster_sensor_relation": "depth/ID/VALID byte-identical; PT remains RGB-only",
        "display_source": "https://github.com/KhronosGroup/ToneMapping/blob/b5a2eed5ddf6c2227090449399de9c7affb9e4c9/PBR_Neutral/pbrNeutral.glsl",
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": {name: hashlib.sha256((here/name).read_bytes()).hexdigest()
                          for name in ("reference_hdri_probe.py", "roughness_calibration.py")},
        "scope": "fixed authored shell/highlight demonstration; no exposure fitting, native support or arbitrary HDR preservation claim",
    }
    _save(output/"protocol.json", declaration)
    report = {"schema": "phymeo-reference-neutral-result-v1", "status": "running", "captures": [],
              "protocol_sha256": hashlib.sha256((output/"protocol.json").read_bytes()).hexdigest(),
              "raster_sensors_unchanged": False, "raw_readbacks_unchanged": {},
              "pt_foreground_recovered_pixels": 0, "closed_sessions": []}
    raw_by_path, images, sensors = {}, {}, []
    original_encoder = _sapien_session.encode_linear_image
    try:
        for path in (RenderPath.RASTER, RenderPath.PATH_TRACE):
            label = "raster" if path is RenderPath.RASTER else "pt"
            for transform in declaration["display_transforms"]:
                studio = None
                if path is RenderPath.PATH_TRACE:
                    studio = SapienStudio(panel_strength=0., environment_map=RadianceEnvironment(
                        environment_file.as_uri(), environment_sha))
                config = SapienConfig(path=path, device=device, samples_per_pixel=64,
                                      max_bounces=4, denoiser="none", studio=studio,
                                      exposure=8., display_transform=transform)
                session = SapienBackend(config).open_scene(scene)
                item = {"path": label, "display_transform": transform, "exposure": config.exposure}
                taps = []

                def observe_readback(values, **options):
                    if values.dtype != np.float32:
                        raise ValueError("expected the real SAPIEN binary32 HDR camera readback")
                    taps.append(np.array(values, dtype="<f4", copy=True))
                    return original_encoder(values, **options)

                try:
                    channels = tuple(Channel) if path is RenderPath.RASTER else (Channel.RGB,)
                    with patch.object(_sapien_session, "encode_linear_image", side_effect=observe_readback):
                        captured = session.capture(frame, CaptureRequest(camera, path, channels))
                    if len(taps) != 1:
                        raise ValueError("each public exposure must apply exactly one display conversion")
                    planes = {plane.channel: plane.data for plane in captured.planes}
                    stem = label+"-"+transform
                    item.update(linear_file=stem+".linear-f32.raw", rgb_file=stem+".rgb8.raw",
                                image_file=stem+".png", shape=list(taps[0].shape),
                                diagnostics=session.diagnostics())
                    (output/item["linear_file"]).write_bytes(taps[0].tobytes())
                    (output/item["rgb_file"]).write_bytes(planes[Channel.RGB])
                    hdri._png(output/item["image_file"], planes[Channel.RGB], camera.width, camera.height)
                    item["linear_sha256"] = hashlib.sha256(taps[0].tobytes()).hexdigest()
                    item["rgb_sha256"] = hashlib.sha256(planes[Channel.RGB]).hexdigest()
                    raw_by_path.setdefault(label, []).append(item["linear_sha256"])
                    images[(label, transform)] = np.frombuffer(planes[Channel.RGB], dtype="u1").reshape(120, 160, 3)
                    if path is RenderPath.RASTER:
                        sensor_hashes = {}
                        for channel in (Channel.FORWARD_DEPTH, Channel.INSTANCE_ID, Channel.VALID):
                            filename = stem+"."+channel.name.lower()+".raw"
                            (output/filename).write_bytes(planes[channel])
                            sensor_hashes[channel.value] = hashlib.sha256(planes[channel]).hexdigest()
                        item["sensor_sha256"] = sensor_hashes
                        sensors.append(sensor_hashes)
                        ids = np.frombuffer(planes[Channel.INSTANCE_ID], dtype="<u4").reshape(120, 160)
                        if not np.all(ids[40:80, 60:100] == 1):
                            raise ValueError("fixed foreground ROI is not fully covered by shell ID 1")
                    report["captures"].append(item)
                    _save(output/"summary.json", report)
                finally:
                    session.close()
                    closed = session.diagnostics()
                    report["closed_sessions"].append(closed)
                    if (not closed["lifecycle"]["closed"]
                            or closed["lifecycle"]["attached_entities_after_clear"] != 0
                            or closed["lifecycle"]["studio_temporary_files_exist"]):
                        raise RuntimeError("reference display calibration did not release scene resources")
        report["raster_sensors_unchanged"] = sensors[0] == sensors[1]
        report["raw_readbacks_unchanged"] = {label: values[0] == values[1] for label, values in raw_by_path.items()}
        clip, neutral = (images[("pt", transform)][40:80, 60:100] for transform in declaration["display_transforms"])
        recovered = np.all(clip == 255, axis=-1) & np.all(neutral < 255, axis=-1)
        report["pt_foreground_recovered_pixels"] = int(recovered.sum())
        report["foreground_rgb8"] = {
            "clip_mean": clip.mean(axis=(0, 1)).tolist(), "neutral_mean": neutral.mean(axis=(0, 1)).tolist(),
            "clip_range": [int(clip.min()), int(clip.max())], "neutral_range": [int(neutral.min()), int(neutral.max())],
        }
        passed = (report["raster_sensors_unchanged"] and all(report["raw_readbacks_unchanged"].values())
                  and report["pt_foreground_recovered_pixels"] >= declaration["minimum_recovered_foreground_pixels"])
        report["status"] = "passed" if passed else "not_demonstrated"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _save(output/"summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    report = run_calibration(args.output, device=args.device)
    print(json.dumps({"status": report["status"], "summary": str(args.output.resolve()/"summary.json")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
