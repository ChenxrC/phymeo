"""Measure saved public entry wrenches without replaying physical motion.

Single-world diagnostic traces with feedback_source are supported. Entry k is
read from trace k+1 and checked against trace k. The last entry has no following
feedback source and is explicitly reported as unmeasured, never synthesized.
"""

import argparse
import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import mujoco as mj
import numpy as np

from phymeo.control.manipulation_wrench import measure_object_contact_wrench


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def measure_run(
    run_directory, output_directory, object_name="object", group_names=("left-pad", "right-pad")
):
    root, output = Path(run_directory).resolve(), Path(output_directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    require(
        manifest["status"] == "COMPLETE" and len(manifest["worlds"]) == 1,
        "requires a complete single-world diagnostic trace",
    )
    info = manifest["worlds"][0]
    model_path = (root / info["source_file"]).resolve()
    require(model_path.is_relative_to(root), "model path escapes run")
    trace_path = root / "trace.jsonl.gz"
    require(sha(model_path) == info["source_sha256"], "model binding differs")
    require(sha(trace_path) == manifest["trace_sha256"], "trace binding differs")
    model = mj.MjModel.from_xml_path(str(model_path))
    body = int(model.body(object_name).id)
    groups = tuple(int(model.geom(name).id) for name in group_names)
    require(len(groups) <= 8 and len(set(groups)) == len(groups), "invalid groups")
    output.mkdir(parents=True, exist_ok=False)
    maximum_force = maximum_moment = maximum_load = 0.0
    count = last_step = 0
    previous = None
    output_file = output / "wrenches.jsonl.gz"
    with (
        gzip.open(trace_path, "rt", encoding="utf-8") as source,
        gzip.open(output_file, "xt", encoding="utf-8") as destination,
    ):
        for line in source:
            current = json.loads(line)
            require(current["step"] == last_step + 1, "noncontiguous trace")
            last_step = current["step"]
            if previous is None:
                require(
                    current["feedback_source"]["entry_meta"][0][0] == 0,
                    "first feedback source must be invalid",
                )
                previous = current
                continue
            saved = current["feedback_source"]
            names = (
                "entry_meta",
                "entry_clock",
                "entry_contact_ids",
                "entry_contact_data",
                "entry_body_mask",
                "entry_body_position",
            )
            arrays = {
                name: np.asarray(
                    saved[name],
                    dtype=np.int32
                    if name in ("entry_meta", "entry_contact_ids", "entry_body_mask")
                    else np.float32,
                )
                for name in names
            }
            require(arrays["entry_meta"].shape == (1, 6), "requires W1 entry")
            require(
                np.array_equal(
                    arrays["entry_contact_ids"], np.asarray(previous["contact_ids"], dtype=np.int32)
                ),
                "contact identities are not from preceding trace",
            )
            require(
                np.array_equal(
                    arrays["entry_contact_data"],
                    np.asarray(previous["contact_data"], dtype=np.float32),
                ),
                "contact payload is not from preceding trace",
            )
            frame = previous["success_frames"][str(info["world"])]
            origin = np.asarray(frame["object_position"], dtype=np.float64)
            quat = np.asarray(frame["object_wxyz"], dtype=np.float64)
            require(
                origin.shape == (3,)
                and quat.shape == (4,)
                and np.isfinite(origin).all()
                and np.isfinite(quat).all()
                and abs(np.linalg.norm(quat) - 1) < 2e-6,
                "invalid entry object pose",
            )
            require(
                np.array_equal(origin, arrays["entry_body_position"][0, body]),
                "object pose is not from the same entry",
            )
            rotation = np.zeros(9)
            mj.mju_quat2Mat(rotation, quat / np.linalg.norm(quat))
            com = origin + rotation.reshape(3, 3) @ model.body_ipos[body]
            measured = measure_object_contact_wrench(
                arrays, 0, object_body=body, object_com_world=com, group_geom_ids=groups
            )
            require(
                measured is not None
                and (measured.input_tick, measured.output_tick)
                == (previous["step"] - 1, previous["step"]),
                "entry tick mismatch",
            )
            require(
                frame["phase"] == "accepted-entry"
                and frame["world"] == info["world"]
                and frame["tick"] == measured.input_tick
                and abs(frame["time_s"] - measured.time_s) < 2e-6,
                "success frame phase mismatch",
            )
            for contact in measured.contacts:
                index = contact.index
                ids = previous["contact_ids"][0][index]
                side = 0 if ids[4] == body else 1
                sign = -1 if side == 0 else 1
                published = sign * np.asarray(previous["contact_world_wrench"][0][index])
                body_wrench = np.asarray(previous["contact_body_wrench"][0][index][side])
                shifted = body_wrench[3:] + np.cross(origin - com, body_wrench[:3])
                force_error = max(
                    np.max(np.abs(published[:3] - contact.force_world)),
                    np.max(np.abs(body_wrench[:3] - contact.force_world)),
                )
                moment_error = max(
                    np.max(np.abs(published[3:] - contact.couple_world)),
                    np.max(np.abs(shifted - contact.moment_com_world)),
                )
                maximum_force = max(maximum_force, float(force_error))
                maximum_moment = max(maximum_moment, float(moment_error))
            if group_names == ("left-pad", "right-pad"):
                maximum_load = max(
                    maximum_load,
                    float(
                        np.max(
                            np.abs(
                                np.asarray(frame["finger_normal_forces_n"])
                                - measured.group_normal_loads
                            )
                        )
                    ),
                )
            require(
                max(maximum_force, maximum_moment, maximum_load) < 2e-6,
                "published FP32 wrench conversion differs by >= 2e-6",
            )
            destination.write(
                json.dumps(
                    dict(step=previous["step"], measurement=asdict(measured)),
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
            previous = current
    require(
        last_step == manifest["steps"] and count == last_step - 1 and count > 0,
        "incomplete input trace",
    )
    result = dict(
        schema="saved-public-object-wrench-v1",
        status="PASS",
        world=info["world"],
        source_trace_sha256=manifest["trace_sha256"],
        source_model_sha256=info["source_sha256"],
        object_body=body,
        object_com_local=model.body_ipos[body].tolist(),
        group_geom_ids=groups,
        measured_entries=count,
        first_entry_step=1,
        last_entry_step=last_step - 1,
        unmeasured_entry_steps=[last_step],
        output_sha256=sha(output_file),
        max_published_force_difference_n=maximum_force,
        max_published_moment_difference_nm=maximum_moment,
        max_group_load_difference_n=maximum_load,
        qualification="saved public entry consumer; no forward/step or new physics; force measurement is not grasp stability",
    )
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    parser.add_argument("output_directory")
    parser.add_argument("--object", default="object")
    parser.add_argument("--groups", nargs="*", default=["left-pad", "right-pad"])
    args = parser.parse_args()
    print(
        json.dumps(
            measure_run(args.run_directory, args.output_directory, args.object, tuple(args.groups))
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
