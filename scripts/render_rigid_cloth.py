"""Render two verified numeric experiments as a self-contained offline replay."""

import argparse
import json
from pathlib import Path

import numpy as np

from phymeo.experiments.rigid_cloth import load_run


def _record(directory, interaction):
    run = load_run(directory)
    config, arrays, metrics = run.config, run.arrays, run.metrics
    frames = config["frames"]
    count = config["rows"] * config["columns"]
    substeps = frames * config["substeps"]
    if config["scene"] != "pinned_cloth_dynamic_sphere_v1" or config["interaction"] != interaction:
        raise ValueError("replay requires the fixed two-way and prescribed one-way experiments")
    shapes = {
        "particle_position": (frames + 1, count, 3),
        "rigid_position": (frames + 1, 1, 3),
        "rigid_velocity": (frames + 1, 1, 3),
        "time": (frames + 1,),
    }
    if any(arrays[name].shape != shape for name, shape in shapes.items()):
        raise ValueError("trajectory shapes disagree with the recorded scene")
    expected_time = np.arange(frames + 1) * config["substeps"] * config["effective_substep_dt"]
    if not np.array_equal(arrays["time"], expected_time):
        raise ValueError("trajectory time disagrees with the recorded step duration")
    if config["duration_seconds"] != float(expected_time[-1]):
        raise ValueError("recorded horizon disagrees with trajectory time")
    for name, width in (("triangles", 3), ("edges", 2)):
        value = arrays[name]
        if (
            value.ndim != 2
            or value.shape[1] != width
            or value.dtype.kind not in "iu"
            or np.any(value < 0)
            or np.any(value >= count)
        ):
            raise ValueError("invalid cloth topology")
    pins = arrays["pinned"]
    if pins.ndim != 1 or pins.dtype.kind not in "iu" or np.any(pins < 0) or np.any(pins >= count):
        raise ValueError("invalid pin indices")
    if pins.tolist() != config["pinned"]:
        raise ValueError("recorded pin indices disagree")
    if metrics["accepted_frames"] != frames or metrics["audited_substeps"] != substeps:
        raise ValueError("accepted counts disagree with trajectory")
    for name, value in arrays.items():
        if name.startswith("substep_") and value.shape != (substeps,):
            raise ValueError("substep evidence shape disagrees with trajectory")
    if metrics["contact_substeps"] != int(np.count_nonzero(arrays["substep_contact"])):
        raise ValueError("contact metric disagrees with saved substep evidence")
    for metric, field in (
        ("max_vertex_penetration_m", "vertex_penetration_m"),
        ("max_pin_displacement_m", "pin_displacement_m"),
        ("max_reported_residual", "reported_residual"),
    ):
        if metrics[metric] != float(np.max(arrays["substep_" + field])):
            raise ValueError("summary metric disagrees with saved substep evidence")
    metadata = json.loads((Path(directory) / "run.json").read_text(encoding="utf-8"))
    fields = (*shapes, "triangles", "edges", "pinned", "substep_contact")
    return {
        "config": config,
        "metrics": metrics,
        "source": metadata["provenance"],
        "arrays": {name: arrays[name].tolist() for name in fields},
    }


def render(two_way, one_way, output):
    records = {
        "schema": "phymeo.rigid_cloth.replay.v1",
        "two_way": _record(two_way, "two_way_dynamic"),
        "one_way": _record(one_way, "one_way_prescribed_sphere"),
    }
    configs = [records[name]["config"] for name in ("two_way", "one_way")]
    if {k: v for k, v in configs[0].items() if k != "interaction"} != {
        k: v for k, v in configs[1].items() if k != "interaction"
    }:
        raise ValueError("replay experiments must share all controls except interaction")
    if records["two_way"]["arrays"]["time"] != records["one_way"]["arrays"]["time"]:
        raise ValueError("replay experiments must share the same timeline")
    # Metadata is data, never markup, including provenance from external traces.
    encoded = json.dumps(records, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("templates") / "rigid_cloth.html"
    content = template.read_text(encoding="utf-8").replace("__SCENE_DATA__", encoded)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(content)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-way", type=Path, required=True)
    parser.add_argument("--one-way", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.two_way, args.one_way, args.output))


if __name__ == "__main__":
    main()
