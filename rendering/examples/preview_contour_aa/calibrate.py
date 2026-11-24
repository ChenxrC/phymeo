"""Measure actual contour coverage against analytic pixel areas on the compact GPU owner."""
import argparse
from contextlib import closing
import ctypes as ct
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image

import msaa_calibration as calibration
from preview_compact_rigid_history.runtime import RigidHistoryBackend, activate
from preview_compact_rigid_history.state import bind_presentations, frame_digest


def counts(owner):
    function = owner.backend.lib.pr_preview_contour_counts
    function.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint32), ct.c_size_t]
    function.restype = ct.c_int
    result = (ct.c_uint32 * 4)()
    device = owner.underlying._delegate.device
    if function(device._handle, result, ct.sizeof(result)):
        raise RuntimeError(device._error())
    return dict(zip(("candidate_pixels", "resolved_pixels", "unsupported_pixels", "visibility_rays"), result))


def coverage_errors(aa, center, solid, analytical):
    area = np.asarray(analytical)
    mask = (area > 1.e-12) & (area < 1.-1.e-12)
    if np.any(solid[mask] <= 0):
        raise ValueError("calibration requires a bright foreground")
    aa_error = float(np.abs(aa[mask] / solid[mask] - area[mask]).mean())
    center_error = float(np.abs(center[mask] - area[mask]).mean())
    return {"edge_pixels": int(mask.sum()), "mae_center": center_error, "mae_resolve": aa_error,
            "fractional_reduction": 1 - aa_error / center_error}


def measure(prepared, output, device_index):
    root, manifest = activate(prepared)
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath

    output.mkdir(parents=True, exist_ok=False)
    mesh_dir = output / "geometry"
    mesh_dir.mkdir()
    scene, models = calibration._fixture(mesh_dir)
    cases = [("solid", ("solid",), 0.), ("edge", ("edge",), 0.)]
    work = [calibration._exposure(scene, models, case, index) for index, case in enumerate(cases)]
    bindings = bind_presentations([row[0] for row in work], frame_digest(asdict(scene)))
    report = {"schema": "phymeo-compact-contour-calibration-v1", "captures": {},
              "dll_sha256": sha256((root / "build/phymeo_native.dll").read_bytes()).hexdigest(),
              "history_shader_sha256": sha256((root / "build/preview_history.comp.spv").read_bytes()).hexdigest(),
              "contract": manifest["contract"]}
    settings = dict(device_index=device_index, exposure=1., display_transform="clip",
                    reflection_mode="environment", validation=True, reuse_history=False)
    with closing(RigidHistoryBackend(root, **settings).open_scene(scene)) as owner:
        for (name, active, _), (frame, camera, layers, analytical), binding in zip(cases, work, bindings):
            request = CaptureRequest(camera, RenderPath.RASTER, tuple(Channel))
            shot = owner.capture(frame, request, presentation=binding)
            planes = {plane.channel: plane.data for plane in shot.planes}
            check = calibration._sensor_check(planes, layers, camera.width, camera.height)
            if not check["passed"]:
                raise ValueError(f"center sensor disagrees with the analytic fixture: {check}")
            telemetry = counts(owner)
            hdr = owner.read_reconstructed_linear()
            linear = np.frombuffer(hdr["data"], "<f4").reshape(camera.height, camera.width, 4)[..., :3].mean(axis=2)
            Image.frombytes("RGB", (camera.width, camera.height), planes[Channel.RGB]).save(output / (name + ".png"))
            report["captures"][name] = {"sensor_check": check, "contour": telemetry,
                                        "sensor_sha256": {ch.value: sha256(value).hexdigest()
                                                          for ch, value in planes.items() if ch != Channel.RGB}}
            if name == "solid":
                if telemetry["candidate_pixels"] or telemetry["visibility_rays"]:
                    raise ValueError("uniform surface triggered contour rays")
                solid = linear.copy()
            else:
                ids = np.frombuffer(planes[Channel.INSTANCE_ID], "<u4").reshape(linear.shape)
                result = coverage_errors(linear, ids == models[active[0]]["id"], solid, analytical)
                report["slanted_edge"] = result
                if not telemetry["resolved_pixels"] or not telemetry["visibility_rays"]:
                    raise ValueError("no mixed-coverage pixels published")
    report["device"] = owner.diagnostics()["device"]
    report["passed"] = calibration._cleanup(owner.diagnostics()) and report["slanted_edge"]["fractional_reduction"] >= .25
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise ValueError("analytic contour or cleanup gate failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=1)
    args = parser.parse_args()
    measure(args.prepared, args.output, args.device_index)
