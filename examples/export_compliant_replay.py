"""Export a self-contained, offline Three.js replay of a Phymeo compliant-cloth trace."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from phymeo.constraints.robot_contact_patch import RobotContactPatch
from phymeo.core.robot import RobotModel
from phymeo.dynamics.robot_kinematics import forward_kinematics
from phymeo.experiments.compliant_cloth import build_impact
from phymeo.experiments.compliant_handoff import build_handoff

DEMOS = Path(__file__).resolve().parents[1] / "docs/demos"


def _bind_recorded_models(trace):
    models = {name: RobotModel(**data) for name, data in trace["robot_models"].items()}
    parents = {name: model.parents.tolist() for name, model in models.items()}
    if "robot_parents" in trace and trace["robot_parents"] != parents:
        raise ValueError("recorded robot parents differ from the model")
    trace["robot_parents"] = parents
    attachments = {}
    for row in trace["patches"]:
        patch = RobotContactPatch(
            row["name"], row["link_index"], np.asarray(row["local_vertices_m"])
        )
        robot = row["robot"]
        if patch.key in attachments or (
            robot is not None
            and (robot not in models or patch.link_index >= models[robot].dof_count)
        ):
            raise ValueError("invalid or duplicate recorded patch binding")
        attachments[patch.key] = (robot, patch)
    for frame in trace["frames"]:
        if set(frame["robots"]) != set(models):
            raise ValueError("recorded robot bindings differ from the model set")
        poses = {}
        for name, model in models.items():
            robot = frame["robots"][name]
            pose = forward_kinematics(model, np.asarray(robot["q"], dtype=float))
            if "links_m" in robot and not np.array_equal(robot["links_m"], pose.position):
                raise ValueError("recorded robot links differ from forward kinematics")
            robot["links_m"] = pose.position.tolist()
            poses[name] = pose
        if set(frame["patch_vertices_m"]) != set(attachments):
            raise ValueError("recorded patch membership differs from model attachments")
        for key, (robot, patch) in attachments.items():
            vertices = patch.local_vertices
            if robot is not None:
                pose = poses[robot]
                link = patch.link_index
                vertices = vertices @ pose.rotation[link].T + pose.position[link]
            if not np.array_equal(frame["patch_vertices_m"][key], vertices):
                raise ValueError("recorded patch differs from its model attachment")


def add_link_geometry(trace):
    if "robot_models" in trace:
        _bind_recorded_models(trace)
        return
    if trace.get("robot_parents") and all(
        all("links_m" in r for r in f["robots"].values()) for f in trace["frames"]
    ):
        return
    name = trace["case"]
    if name in ("impact", "support"):
        program = build_impact(support=name == "support").program
    elif name in ("handoff", "no-grip"):
        program = build_handoff(grip_friction=0.9 if name == "handoff" else 0.0).program
    else:
        return
    trace["robot_parents"] = {n: r.model.parents.tolist() for n, r in program.robots.items()}
    for frame in trace["frames"]:
        for n, r in program.robots.items():
            pose = forward_kinematics(r.model, np.asarray(frame["robots"][n]["q"], dtype=float))
            for patch in program.patches:
                if patch.robot_domain != n:
                    continue
                p = patch.geometry
                world = (
                    p.local_vertices @ pose.rotation[p.link_index].T + pose.position[p.link_index]
                )
                if not np.array_equal(world, np.asarray(frame["patch_vertices_m"][p.key])):
                    raise ValueError("saved patch geometry differs from forward kinematics")
            frame["robots"][n]["links_m"] = pose.position.tolist()
        for patch in program.patches:
            key = patch.geometry.key
            contact = frame.get("contacts", {}).get(key)
            if contact is None or "normal" in contact:
                continue
            # These captured cases use only prismatic joints: orientation is constant over the substep.
            if patch.robot_domain is not None and any(
                j != "prismatic" for j in program.robots[patch.robot_domain].model.joint_types
            ):
                continue
            vertices = np.asarray(frame["patch_vertices_m"][key])
            axis = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            axis = patch.normal_sign * axis / np.linalg.norm(axis)
            if not np.allclose(
                np.asarray(contact["force_n"]) @ axis,
                contact["normal_force_n"],
                rtol=1e-11,
                atol=1e-11,
            ):
                raise ValueError("saved normal magnitude disagrees with recorded force")
            contact["normal"] = axis.tolist()


def export(source, output):
    data = source.read_bytes()
    trace = json.loads(data)
    if trace.get("schema") != "phymeo-compliant-case-v1" or not trace.get("frames"):
        raise ValueError("expected a compliant-cloth trajectory")
    add_link_geometry(trace)
    # The full source file retains every substep. The viewer reads the recorded frame data.
    trace.pop("samples", None)
    cap = dict(source_sha256=hashlib.sha256(data).hexdigest(), trajectory=trace)
    packed = json.dumps(cap, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    html = (DEMOS / "compliant-cloth.html").read_text(encoding="utf8")
    lib = (DEMOS / "vendor/three-r160.min.js").read_text(encoding="utf8")
    license_text = (DEMOS / "vendor/three-r160.LICENSE").read_text(encoding="utf8")
    html = html.replace(
        '<script src="./vendor/three-r160.min.js"></script>', "<script>\n" + lib + "\n</script>"
    )
    html = html.replace(
        '<script id="capture" type="application/json">null</script>',
        '<script id="capture" type="application/json">' + packed + "</script>",
    )
    html = html.replace("</html>", "<!-- Three.js r160 license\n" + license_text + "\n-->\n</html>")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf8", newline="\n") as f:
        f.write(html)
    return cap["source_sha256"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    print(export(args.source, args.output))
