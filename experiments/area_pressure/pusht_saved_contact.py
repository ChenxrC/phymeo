"""Compare native and elastic area contact at identical accepted RR PushT input poses."""

import argparse
import gzip
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from experiments.area_pressure.capsule_box_reference import axis_box_clearance, integrate
from phymeo.backends.mujoco_area_pressure import area_collision_ownership
from phymeo.training.manipulation_tasks import manipulation_scenes

GRADIENT_PA_M = 1e6
CLEAR_STEPS = 20


class PushGeometry:
    def __init__(self, model, scene):
        self.pusher = model.geom("pusher").id
        self.objects = tuple(model.geom(name).id for name in scene.object_geom_names)
        self.support = model.geom(scene.support_geom_name).id
        self.tool = model.geom_bodyid[self.pusher]
        self.body = model.geom_bodyid[self.objects[0]]
        if model.geom_type[self.pusher] != mj.mjtGeom.mjGEOM_CAPSULE:
            raise ValueError("expected a capsule tool")
        boxes = []
        for i in self.objects:
            if (
                model.geom_type[i] != mj.mjtGeom.mjGEOM_BOX
                or model.geom_bodyid[i] != self.body
                or not np.array_equal(model.geom_quat[i], [1, 0, 0, 0])
            ):
                raise ValueError("expected co-aligned local boxes on one body")
            boxes.append(
                [model.geom_pos[i] - model.geom_size[i], model.geom_pos[i] + model.geom_size[i]]
            )
        self.boxes = np.asarray(boxes)
        self.com = model.body_ipos[self.body].copy()
        self.radius, half_length = model.geom_size[self.pusher, :2]
        mat = np.zeros(9)
        mj.mju_quat2Mat(mat, model.geom_quat[self.pusher])
        axis = mat.reshape(3, 3)[:, 2] * half_length
        self.local_ends = model.geom_pos[self.pusher] + np.array([-axis, axis])

    def frame(self, record):
        entry = record["entry"]
        pos = np.asarray(entry["entry_body_position"][0])
        rot = np.asarray(entry["entry_body_rotation"][0])
        obj_rotation = rot[self.body]
        capsule = (
            self.local_ends @ rot[self.tool].T + (pos[self.tool] - pos[self.body])
        ) @ obj_rotation
        count = entry["entry_meta"][0][3]
        q = np.asarray(entry["entry_q"][0])
        wrench = np.zeros(3)
        load = support = 0.0
        detected = 0
        owners = []
        for ids, data, world in zip(
            entry["entry_contact_ids"][0][:count],
            entry["entry_contact_data"][0][:count],
            entry["entry_contact_wrench_world"][0][:count],
            strict=True,
        ):
            _, active, first, second = ids[:4]
            if (first in self.objects) == (second in self.objects):
                continue
            on_first = first in self.objects
            other = second if on_first else first
            if other == self.pusher:
                detected += 1
                if active:
                    load += data[13]
                    wrench += (-1 if on_first else 1) * np.array(world[:3])
                    owners.append(int(first if on_first else second))
            elif other == self.support and active:
                support += data[13]
        return dict(
            tick=int(entry["entry_meta"][0][1]),
            q=q.tolist(),
            ends=capsule.tolist(),
            rotation=obj_rotation.tolist(),
            position=pos[self.body].tolist(),
            tool_normal_n=load,
            tool_force_world_n=wrench.tolist(),
            support_n=support,
            detected=detected,
            object_geom_ids=owners,
            clearance_m=axis_box_clearance(capsule, self.boxes, self.radius),
        )


def compare(directory):
    manifest = json.loads((directory / "manifest.json").read_text("utf-8-sig"))
    (world,) = manifest["worlds"]
    scene = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
    if (
        manifest["mode"] != "cuda"
        or manifest["status"] != "COMPLETE"
        or world["key"] != scene.key
        or world["source_sha256"] != scene.source_sha256
        or manifest["h"] != scene.timestep_s
    ):
        raise ValueError("expected completed original RR CUDA trajectory")
    model = mj.MjModel.from_xml_string(scene.mjcf)
    geometry = PushGeometry(model, scene)
    ownership = area_collision_ownership(
        mj, model, ((model.body(geometry.tool).name, model.body(geometry.body).name),)
    )

    def named(pairs):
        return [(model.geom(i).name, model.geom(j).name) for i, j in pairs]

    key = str(world["world"])
    samples = {}
    first_segment = None
    clear_count = 0
    impulse = np.zeros(3)
    with gzip.open(directory / "trace.jsonl.gz", "rt", encoding="utf-8") as stream:
        for tick, line in enumerate(stream):
            row = json.loads(line)
            if row["step"] != tick + 1 or row["accepted"] != {key: True}:
                raise ValueError("noncontiguous or unaccepted physical step")
            frame = geometry.frame(row)
            if frame["tick"] != tick:
                raise ValueError("wrong contact/pose phase")
            segment = row["reference_feedback"][key]["segment"]
            samples.setdefault("initial", frame)
            if frame["tool_normal_n"] > 0 and first_segment is None:
                first_segment = segment
                samples["first_load"] = frame
            if first_segment is not None and first_segment == segment:
                impulse += np.array(frame["tool_force_world_n"]) * scene.timestep_s
                if (
                    "peak" not in samples
                    or frame["tool_normal_n"] > samples["peak"]["tool_normal_n"]
                ):
                    samples["peak"] = frame
                samples["last_push_entry"] = frame
            if first_segment is not None and segment > first_segment:
                clear = frame["detected"] == 0 and frame["clearance_m"] > 0.001
                clear_count = clear_count + 1 if clear else 0
                if clear_count == CLEAR_STEPS:
                    samples["withdrawn"] = frame
                    break
    if set(samples) != {"initial", "first_load", "peak", "last_push_entry", "withdrawn"}:
        raise ValueError("missing load, peak or verified withdrawal")
    evaluate_poses(geometry, samples)
    displacement = (
        np.array(samples["last_push_entry"]["position"]) - samples["first_load"]["position"]
    )
    return dict(
        scene=scene.key,
        source_sha256=scene.source_sha256,
        saved_trace_sha256=manifest["trace_sha256"],
        first_segment=first_segment,
        h=scene.timestep_s,
        gradient_pa_m=GRADIENT_PA_M,
        radius_m=geometry.radius,
        replaced_candidates=named(ownership.replaced_geometry_pairs[0]),
        retained_native_candidates=named(ownership.native_geometry_pairs),
        geometry_names=list(scene.object_geom_names),
        geometry_ids=list(map(int, geometry.objects)),
        peak_fit=peak_fit_check(samples),
        displacement_while_pushing_m=displacement.tolist(),
        tool_impulse_world_ns=impulse.tolist(),
        clear_steps=CLEAR_STEPS,
        bound_rows=[
            "area_m2",
            "pressure_load_n",
            "Fx_n",
            "Fy_n",
            "Fz_n",
            "Mx_nm",
            "My_nm",
            "Mz_nm",
        ],
        samples=samples,
    )


def evaluate_poses(geometry, samples):
    for frame in samples.values():
        frame["area_reference"] = {
            str(depth): integrate(
                frame["ends"],
                geometry.boxes,
                geometry.radius,
                GRADIENT_PA_M,
                geometry.com,
                frame["rotation"],
                depth=depth,
            )
            for depth in (7, 9)
        }


def compare_saved_poses():
    """Recompute the five shipped pose comparisons without requiring the full motion archive."""
    source = Path(__file__).with_name("pusht_reference_poses.json")
    report = json.loads(source.read_text("utf-8"))
    scene = next(s for s in manipulation_scenes() if s.key == report["scene"])
    if (
        report["schema"] != "phymeo-pusht-saved-poses-v1"
        or report["source_sha256"] != scene.source_sha256
    ):
        raise ValueError("saved poses require the original scene and schema")
    geometry = PushGeometry(mj.MjModel.from_xml_string(scene.mjcf), scene)
    evaluate_poses(geometry, report["samples"])
    report["gradient_pa_m"] = GRADIENT_PA_M
    report["peak_fit"] = peak_fit_check(report["samples"])
    return report


def peak_fit_check(samples):
    """Hold the measured poses fixed. Fit only the native peak normal load, then transfer."""
    peak = samples["peak"]
    low, high = peak["area_reference"]["9"]["bounds"][1]
    if low <= 0:
        raise ValueError("peak load calibration requires resolved positive pressure area")
    scaling = (peak["tool_normal_n"] / high, peak["tool_normal_n"] / low)
    holdout = {}
    for name in ("first_load", "last_push_entry"):
        low, high = samples[name]["area_reference"]["9"]["bounds"][1]
        prediction = (low * scaling[0], high * scaling[1])
        actual = samples[name]["tool_normal_n"]
        holdout[name] = dict(
            predicted_normal_n=prediction,
            observed_normal_n=actual,
            compatible=prediction[0] <= actual <= prediction[1],
        )
    return dict(gradient_pa_m=[factor * GRADIENT_PA_M for factor in scaling], holdout=holdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory",
        type=Path,
        nargs="?",
        help="Full saved trajectory; omit to use the five shipped input poses",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.trajectory) if args.trajectory is not None else compare_saved_poses()
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2, allow_nan=False)
    for name, sample in report["samples"].items():
        print(
            name,
            sample["tick"],
            sample["tool_normal_n"],
            sample["area_reference"]["9"]["bounds"][:2],
        )


if __name__ == "__main__":
    main()
