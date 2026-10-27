"""Render actual P5 robot, cloth and solid traces into one offline HTML viewer."""

import argparse
import json
from pathlib import Path

import numpy as np

from phymeo.experiments.articulated import load_articulated_run
from phymeo.experiments.prescribed_articulated import load_prescribed_run


def surface_triangles(cells):
    faces = {}
    for a, b, c, d in cells:
        for face in ((a, c, b), (a, b, d), (a, d, c), (b, c, d)):
            key = tuple(sorted(face))
            faces.setdefault(key, []).append(face)
    if any(len(values) > 2 for values in faces.values()):
        raise ValueError("non-manifold tetrahedral surface")
    return np.array([values[0] for values in faces.values() if len(values) == 1], dtype=np.int64)


def record(directory, prescribed, expected_scene):
    loader = load_prescribed_run if prescribed else load_articulated_run
    run = loader(directory)
    expected_mode = "prescribed_driver" if prescribed else "two_way"
    if run.config["variant"] != expected_scene or run.config["interaction"] != expected_mode:
        raise ValueError(
            f"{expected_scene}: scene variant or interaction does not match the replay slot"
        )
    source = json.loads((Path(directory) / "run.json").read_text(encoding="utf-8"))["provenance"]
    prefix = "driver" if prescribed else "robot"
    fields = [
        "time",
        f"{prefix}_q",
        f"{prefix}_v",
        "attachment_links",
        "attachment_centers",
        "attachment_radii",
    ]
    for domain in run.config["node_domains"]:
        fields.extend(
            (
                domain + "_position",
                domain + "_velocity",
                domain + "_fixed",
                f"substep_{domain}_contact",
            )
        )
        fields.extend(
            ("cloth_triangles", "cloth_edges") if domain == "cloth" else ("soft_tetrahedra",)
        )
    result = {
        "config": run.config,
        "metrics": run.metrics,
        "source": source,
        "arrays": {name: run.arrays[name].tolist() for name in fields},
    }
    if "soft" in run.config["node_domains"]:
        result["soft_surface"] = surface_triangles(run.arrays["soft_tetrahedra"]).tolist()
    return result


def render(root, output):
    root = Path(root)
    scenes, all_records = {}, []
    for name in ("S2", "S3", "S4"):
        groups = {"two_way": record(root / name, False, name)}
        if name != "S4":
            groups["prescribed"] = record(root / (name + "-driver"), True, name)
        scenes[name] = groups
        all_records.extend(groups.values())
    environment = {
        tuple(item["source"].get(key) for key in ("source_sha256", "python", "numpy", "platform"))
        for item in all_records
    }
    if len(environment) != 1:
        raise ValueError("replay traces must share package source and numerical environment")
    timeline = all_records[0]["arrays"]["time"]
    if any(item["arrays"]["time"] != timeline for item in all_records):
        raise ValueError("replay traces must share the same timeline")
    data = {
        "schema": "phymeo.articulated.replay.v1",
        "scenes": scenes,
        "complete_horizon": all(item["metrics"]["complete_horizon"] for item in all_records),
    }
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    template = Path(__file__).with_name("templates") / "articulated.html"
    content = template.read_text(encoding="utf-8").replace("__SCENE_DATA__", encoded)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(content)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.input, args.output))


if __name__ == "__main__":
    main()
