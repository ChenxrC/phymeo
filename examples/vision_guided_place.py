"""Change the destination of a loaded gripper using only labelled depth at two exposures.

The command is expressed as extra travel beyond the measured displacement.
The payload pose from the simulator is read only after the run to score the estimate.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from observe_robot_payload import look_at
from robot_payload_pick_place import make_session, run_session

import phymeo as ph


def visible_span_x(camera, snapshot, extrinsics):
    samples, counts = [], []
    for eye, xyzw in extrinsics:
        image = camera.capture(
            snapshot.model,
            snapshot.poses,
            world=snapshot.payload_key.world,
            position_m=eye,
            orientation_xyzw=xyzw,
        )
        mask = image.shape_index == image.keys.index(snapshot.payload_key)
        counts.append(int(mask.sum()))
        if np.any(mask):
            points = camera.unproject(image.ray_depth_m, position_m=eye, orientation_xyzw=xyzw)
            samples.append(points[mask, 0])
    if not samples:
        raise ValueError("payload has no visible surface in either calibrated camera")
    x = np.concatenate(samples)
    return {
        "time_s": snapshot.time_s,
        "xmin_m": float(x.min()),
        "xmax_m": float(x.max()),
        "center_of_visible_span_m": float((x.min() + x.max()) / 2),
        "pixels_by_view": counts,
    }


def run(kind, remaining_m):
    scene, info = make_session(kind)
    camera = ph.DepthCamera(192, 144, 0.68)
    origin = np.asarray(info["table_center"])
    target = origin + [0, 0, 0.075]
    views = []
    for dx in (-0.28, 0.28):
        eye = origin + [dx, -0.48, 0.28]
        views.append((eye, look_at(eye, target)))
    exposures = {"rest": visible_span_x(camera, scene.geometry(), views)}

    def on_command(_reading):
        current = visible_span_x(camera, scene.geometry(), views)
        exposures["in_flight"] = current
        dx = current["center_of_visible_span_m"] - exposures["rest"]["center_of_visible_span_m"]
        requested = dx + remaining_m
        print(f"visible displacement {dx:.9f} m; requested destination {requested:.9f} m", flush=True)
        return requested

    result = run_session(scene, info, on_command)
    measured = result["selected_destination_m"] - remaining_m
    actual = (
        result["events"]["moving_before_change"]["payload_position_m"][0]
        - result["events"]["initial"]["payload_position_m"][0]
    )
    exposures["placed"] = visible_span_x(camera, scene.geometry(), views)
    return dict(
        shape=kind,
        extra_travel_m=remaining_m,
        exposures=exposures,
        measured_displacement_m=measured,
        actual_displacement_m=actual,
        calibration_error_m=measured - actual,
        command_m=result["selected_destination_m"],
        full_run=result,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=("box", "sphere"), default="box")
    parser.add_argument("--extra-travel-m", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("choose a new result path")
    result = run(args.shape, args.extra_travel_m)
    with args.out.open("x", encoding="utf8") as stream:
        json.dump(result, stream, allow_nan=False)
    print(json.dumps({key: v for key, v in result.items() if key != "full_run"}, allow_nan=False))
    run_result = result["full_run"]
    print(
        "steps",
        run_result["accepted_steps"],
        "error_m",
        run_result["target_error_m"],
        "carried",
        run_result["carried"],
        "placed",
        run_result["placed"],
    )
    if not run_result["qualified"] or abs(result["calibration_error_m"]) > 0.005:
        raise SystemExit("did not meet manipulation and 5 mm visual-displacement gates")
