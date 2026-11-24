"""Measure sixteen subpixel rod poses and vacated background on the sealed motion preset."""
import argparse
from contextlib import closing
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

import msaa_calibration as calibration
from preview_compact_rigid_history.runtime import activate
from preview_compact_rigid_history.state import bind_presentations, frame_digest
from preview_surface_quality.runtime import SurfaceQualityBackend
from .calibrate import counts


def capture_sequence(root, scene, work, bindings, rod_id, device_index, reuse_history):
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath

    records, linear, center, rgb = [], [], [], []
    settings = dict(device_index=device_index, exposure=1., display_transform="clip",
                    reflection_mode="environment", validation=True, reuse_history=reuse_history)
    with closing(SurfaceQualityBackend(root, **settings).open_scene(scene)) as owner:
        for (frame, camera, layers, _), binding in zip(work, bindings):
            request = CaptureRequest(camera, RenderPath.RASTER, tuple(Channel))
            shot = owner.capture(frame, request, presentation=binding)
            planes = {plane.channel: plane.data for plane in shot.planes}
            shape = (camera.height, camera.width)
            pixels = np.frombuffer(owner.read_reconstructed_linear()["data"], "<f4").reshape(*shape, 4)
            linear.append(pixels[..., :3])
            rgb.append(np.frombuffer(planes[Channel.RGB], "u1").reshape(*shape, 3))
            center.append(np.frombuffer(planes[Channel.INSTANCE_ID], "<u4").reshape(shape) == rod_id)
            sensors = calibration._sensor_check(planes, layers, camera.width, camera.height)
            history = owner.diagnostics()["history"]["actual"]
            records.append(dict(sensor_check=sensors, contour=counts(owner),
                                center_sensors_sha256={ch.value: sha256(data).hexdigest()
                                                       for ch, data in planes.items() if ch != Channel.RGB},
                                history_accepted=history["accepted_pixels"], history_age_sum=history["age_sum"],
                                history_sequence=history["sequence"]))
    return dict(records=records, linear=np.stack(linear), center=np.stack(center),
                rgb=np.stack(rgb), cleanup=calibration._cleanup(owner.diagnostics()))


def score(sequence, reference):
    # Case order is full white, empty background, 16 moving rod poses, empty background.
    bg, fg = sequence["linear"][1].mean(axis=-1), sequence["linear"][0].mean(axis=-1)
    if np.any(fg <= bg):
        raise ValueError("foreground and background exposures need ordered finite contrast across the frame")
    signal = (sequence["linear"][2:-1].mean(axis=-1) - bg) / (fg - bg)
    center = sequence["center"][2:-1].astype(float)
    mask = np.any((reference > 1e-12) & (reference < 1 - 1e-12), axis=0)
    if not np.any(mask) or not np.isfinite(signal).all():
        raise ValueError("motion region and actual normalized radiance must be finite and nonempty")
    ideal_delta = np.diff(reference, axis=0)[:, mask]
    error_aa = np.abs(np.diff(signal, axis=0)[:, mask] - ideal_delta).mean(axis=1)
    error_center = np.abs(np.diff(center, axis=0)[:, mask] - ideal_delta).mean(axis=1)

    # These positions had nonzero analytic rod area in the previous pose and
    # now have zero area. A center-only ID change would incorrectly include
    # antialiased pixels where a fraction of the rod remains visible.
    vacated = (reference[:-1] > 1e-12) & (reference[1:] <= 1e-12)
    if not vacated.any():
        raise ValueError("no vacated pixels in the registered rod motion")
    hdr_difference = np.abs(sequence["linear"][3:-1] - sequence["linear"][1]).max(axis=-1)
    rgb_difference = np.any(sequence["rgb"][3:-1] != sequence["rgb"][1], axis=-1)
    endpoints_equal = (np.array_equal(sequence["linear"][1], sequence["linear"][-1])
                       and np.array_equal(sequence["rgb"][1], sequence["rgb"][-1])
                       and sequence["records"][1]["center_sensors_sha256"] == sequence["records"][-1]["center_sensors_sha256"])
    return dict(region_pixels=int(mask.sum()), transitions=len(error_aa),
                center_transition_mae=float(error_center.mean()), contour_transition_mae=float(error_aa.mean()),
                fractional_reduction=float(1 - error_aa.mean() / error_center.mean()),
                center_per_transition=error_center.tolist(), contour_per_transition=error_aa.tolist(),
                vacated_observations=int(vacated.sum()), vacated_rgb_mismatches=int(rgb_difference[vacated].sum()),
                vacated_linear_max_abs=float(hdr_difference[vacated].max()),
                empty_endpoints_exact=bool(endpoints_equal))


def measure(prepared, output, device_index):
    root, _ = activate(prepared)
    output.mkdir(parents=True, exist_ok=False)
    geometry = output / "geometry"
    geometry.mkdir()
    scene, models = calibration._fixture(geometry)
    cases = [("solid", ("solid",), 0.), ("before", (), 0.)]
    cases += [(f"rod-{i:02d}", ("rod",), i / 16) for i in range(16)]
    cases.append(("after", (), 0.))
    work = [calibration._exposure(scene, models, case, index) for index, case in enumerate(cases)]
    bindings = bind_presentations([row[0] for row in work], frame_digest(asdict(scene)))
    reference = np.asarray([row[3] for row in work[2:-1]])
    modes = {name: capture_sequence(root, scene, work, bindings, models["rod"]["id"], device_index, reuse)
             for name, reuse in (("current", False), ("rigid_history", True))}
    current, history = modes.values()
    scores = {name: score(run, reference) for name, run in modes.items()}
    histories = dict(rgb_changed_observations=int(np.any(current["rgb"] != history["rgb"], axis=-1).sum()),
                     max_linear_abs=float(np.abs(current["linear"] - history["linear"]).max()),
                     sensor_hashes_equal=all(a["center_sensors_sha256"] == b["center_sensors_sha256"]
                                             for a, b in zip(current["records"], history["records"])))
    report = dict(schema="phymeo-compact-contour-motion-v1", source_sha256=frame_digest(asdict(scene)),
                  history_shader_sha256=sha256((root / "build/preview_history.comp.spv").read_bytes()).hexdigest(),
                  dll_sha256=sha256((root / "build/phymeo_native.dll").read_bytes()).hexdigest(),
                  phases_px=[i/16 for i in range(16)], scope="0.625-pixel-wide rod against empty unlit background",
                  metric="absolute error of adjacent normalized linear brightness changes against polygon/pixel intersection area changes",
                  modes={name: dict(score=scores[name], captures=run["records"], cleanup=run["cleanup"])
                         for name, run in modes.items()}, history_comparison=histories)
    np.savez_compressed(output / "motion.npz", reference=reference, center=current["center"][2:-1],
                        current_hdr=current["linear"], history_hdr=history["linear"],
                        current_rgb=current["rgb"], history_rgb=history["rgb"])
    def actual_queries(run):
        rows = run["records"]
        return (all(rows[i]["contour"]["visibility_rays"] == 0 for i in (0, 1, -1))
                and all(row["contour"]["visibility_rays"] > 0 and row["contour"]["resolved_pixels"] > 0
                        for row in rows[2:-1]))

    report["passed_controls"] = bool(histories["sensor_hashes_equal"]
        and all(actual_queries(run) and run["cleanup"] and all(row["sensor_check"]["passed"] for row in run["records"])
                for run in modes.values())
        and all(s["empty_endpoints_exact"] and not s["vacated_rgb_mismatches"] and s["vacated_linear_max_abs"] == 0.
                for s in scores.values()))
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not report["passed_controls"]:
        raise ValueError("motion control failed; see the recorded raw planes and scores")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--prepared", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--device-index", type=int, default=1)
    args = cli.parse_args()
    measure(args.prepared, args.output, args.device_index)
