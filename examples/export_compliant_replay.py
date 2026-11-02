"""Export a self-contained, offline Three.js replay of a Phymeo compliant-cloth trace."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from phymeo.constraints.robot_contact import (
    CollisionPlane,
    JointLimits,
    LinkCapsule,
    RobotContactMaterial,
)
from phymeo.constraints.robot_contact_patch import RobotContactPatch
from phymeo.core.robot import RobotModel
from phymeo.dynamics.robot_contact import RobotContactSystem
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
    bodies = None
    if "robot_contact" in trace:
        config = trace["robot_contact"]
        bodies = RobotContactSystem(
            models,
            limits={n: JointLimits(**r) for n, r in config["limits"].items()},
            capsules=tuple(LinkCapsule(**r) for r in config["capsules"]),
            planes=tuple(CollisionPlane(**r) for r in config["planes"]),
            material=RobotContactMaterial(**config["material"]),
            excluded_pairs=config["excluded_pairs"],
        )
    attachments = {}
    rounded = set()
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
        if row.get("radius_m") is not None:
            radius = row["radius_m"]
            if (
                type(radius) not in (int, float)
                or not np.isfinite(radius)
                or radius <= 0
                or row["normal_sign"] not in (-1, 1)
            ):
                raise ValueError("invalid recorded finite pad radius or orientation")
            rounded.add(patch.key)
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
        if bodies is not None:
            recorded = frame.get("body_capsules_m", {})
            if set(recorded) != {c.key for c in bodies.capsules}:
                raise ValueError("recorded capsule membership differs")
            for c in bodies.capsules:
                pose = poses[c.domain]
                expected = [
                    pose.position[c.link_index] + pose.rotation[c.link_index] @ p
                    for p in (c.start, c.end)
                ]
                if not np.array_equal(recorded[c.key], expected):
                    raise ValueError("recorded capsule differs from its model attachment")
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
            if key in rounded and "contacts" in frame:
                contact = frame["contacts"].get(key, {})
                normal = np.asarray(contact.get("site_normals"), dtype=float)
                force = np.asarray(contact.get("force_n"), dtype=float)
                magnitude = np.asarray(contact.get("normal_force_n"), dtype=float)
                points = np.asarray(contact.get("points_m"), dtype=float)
                if (
                    normal.ndim != 2
                    or normal.shape[1] != 3
                    or force.shape != normal.shape
                    or points.shape != normal.shape
                    or magnitude.shape != (len(normal),)
                    or not np.isfinite(points).all()
                    or not np.isfinite(force).all()
                    or np.any(magnitude < 0)
                    or not np.allclose(np.linalg.norm(normal, axis=1), 1.0, atol=2e-10, rtol=0.0)
                    or not np.allclose(
                        np.sum(force * normal, axis=1), magnitude, atol=2e-10, rtol=2e-10
                    )
                ):
                    raise ValueError("invalid recorded contact directions or decomposed force")


def _check_structural_records(trace):
    if "cloth_link_material" not in trace:
        return
    shapes = {c["key"]: c for c in trace["robot_contact"]["capsules"]}
    owners = trace["owners"]
    faces = {}
    for i, ids in enumerate(trace["faces"]):
        group = [x.rsplit("/", 1) for x in owners[3 * i : 3 * i + 3]]
        if (
            len(group) != 3
            or len({name for name, _ in group}) != 1
            or [int(n) for _, n in group] != ids
        ):
            raise ValueError("invalid structural material face binding")
        faces[group[0][0]] = ids
    tolerance = 2e-6 if trace.get("cloth_link_force_backend") in ("interpret", "cuda") else 2e-10
    for frame in trace["frames"]:
        body = frame.get("robot_contact", {})
        rows = body.get("cloth_links")
        if not isinstance(rows, list) or len({(r["face"], r["link"]) for r in rows}) != len(rows):
            raise ValueError("invalid structural contact set")
        for row in rows:
            link = shapes.get(row["link"])
            ids = faces.get(row["face"])
            if link is None or link["domain"] != row["robot"] or ids != row["face_nodes"]:
                raise ValueError("unknown structural face or link")
            pts = np.asarray(row["face_positions_m"], dtype=float)
            ends = np.asarray(row["axis_ends_m"], dtype=float)
            normal = np.asarray(row["normal"], dtype=float)
            if (
                pts.shape != (3, 3)
                or ends.shape != (2, 3)
                or normal.shape != (3,)
                or not np.isfinite(pts).all()
                or not np.isfinite(ends).all()
                or not np.isclose(normal @ normal, 1.0, atol=2e-10, rtol=0.0)
                or len(row["sites"]) not in (1, 2)
            ):
                raise ValueError("invalid structural contact geometry")
            total_force = total_power = 0.0
            for site in row["sites"]:
                weights = dict(site["face_weights"])
                t = site["axis_parameter"]
                if (
                    len(site["face_weights"]) != 3
                    or set(weights) != set(ids)
                    or min(weights.values()) < -1e-13
                    or not np.isclose(sum(weights.values()), 1.0, atol=1e-12, rtol=0.0)
                    or not 0 <= t <= 1
                ):
                    raise ValueError("invalid structural material weights")
                point = np.asarray(site["point_m"])
                arm = ends[0] + t * (ends[1] - ends[0])
                force = np.asarray(site["force_on_cloth_n"])
                opposite = np.asarray(site["force_on_link_n"])
                velocity = np.asarray(site["relative_velocity_m_s"])
                fn = site["normal_force_n"]
                distance = row["gap_m"] + link["radius_m"] + row["half_thickness_m"]
                if (
                    point.shape != (3,)
                    or force.shape != (3,)
                    or opposite.shape != (3,)
                    or velocity.shape != (3,)
                    or distance <= 0
                    or fn < 0
                    or not np.array_equal(force, -opposite)
                    or not np.allclose(
                        point, np.array([weights[i] for i in ids]) @ pts, atol=2e-11, rtol=2e-11
                    )
                    or not np.allclose(point - arm, distance * normal, atol=2e-11, rtol=2e-11)
                    or not np.isclose(force @ normal, fn, atol=tolerance, rtol=tolerance)
                    or not np.isclose(force @ velocity, site["power_w"], atol=2e-11, rtol=2e-11)
                ):
                    raise ValueError("invalid structural force, address or work")
                total_force += fn
                total_power += site["power_w"]
            if not np.isclose(total_force, row["normal_force_n"]) or not np.isclose(
                total_power, row["power_w"]
            ):
                raise ValueError("structural load sum differs from its sites")


def add_link_geometry(trace):
    if "robot_models" in trace:
        _bind_recorded_models(trace)
        _check_structural_records(trace)
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


def add_backing_windows(trace):
    """Summarize actual scalar normal impulses in each displayed frame interval."""
    fronts = [p["name"] for p in trace.get("patches", ()) if "backing_clearance_m" in p]
    by_step = trace.get("backing_samples")
    frames = trace["frames"]
    if (not fronts and by_step is None) or (len(frames) == 1 and not by_step):
        return
    dt, steps = trace["dt_s"], trace["substeps"]
    if not fronts or type(steps) is not int or steps < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("invalid backing sample interval")
    h = dt / steps
    names = [p["name"] for p in trace["patches"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate backing patch name")
    expected = steps * (len(frames) - 1)
    samples = trace.get("samples", ())
    if by_step is None or len(by_step) != expected or len(samples) != expected * len(names):
        raise ValueError("incomplete backing sample sequence")

    for i, frame in enumerate(frames[1:]):
        start = frames[i]["t"]
        if not np.isclose(frame["t"] - start, dt, rtol=0, atol=1e-12):
            raise ValueError("invalid backing frame interval")
        accumulated = dict.fromkeys(fronts, 0.0)
        active = 0
        for j in range(steps):
            index = i * steps + j
            time = start + j * h
            result = by_step[index]
            rows = samples[index * len(names) : (index + 1) * len(names)]
            if {r["patch"] for r in rows} != set(names) or not np.isclose(
                result["t"], time, rtol=0, atol=1e-12
            ):
                raise ValueError("incorrect backing sample identity")
            total = 0.0
            for row in rows:
                if not np.isfinite(row["t"]) or abs(row["t"] - time) > 1e-12:
                    raise ValueError("incorrect backing sample time")
                key = row["patch"]
                if key not in accumulated:
                    continue
                impulse = np.asarray(row.get("backing_impulse_ns"), dtype=float)
                if impulse.ndim != 1 or not np.isfinite(impulse).all() or np.any(impulse < 0):
                    raise ValueError("invalid backing impulse")
                value = float(impulse.sum())
                accumulated[key] += value
                total += value
                if j == steps - 1:
                    contact = frame["contacts"][key]
                    force = np.asarray(contact["normal_force_n"], dtype=float)
                    if (
                        not np.array_equal(impulse, contact.get("backing_impulse_ns"))
                        or force.shape != impulse.shape
                        or not np.isfinite(force).all()
                        or np.any(impulse / h > force + 2e-10 * np.maximum(1.0, abs(force)))
                        or not np.isclose(frame["force_sample_s"], time, rtol=0, atol=1e-12)
                    ):
                        raise ValueError("frame backing impulse differs from its source substep")
            if not np.isclose(total, result["total_impulse_ns"], atol=0, rtol=1e-12):
                raise ValueError("backing patch sum differs from the velocity projection")
            active += int(total > 0)
        frame["backing_window"] = dict(
            start_s=start,
            end_s=frame["t"],
            substep_s=h,
            active_substeps=active,
            impulse_ns_by_patch=accumulated,
        )


def export(source, output):
    with source.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        stream.seek(0)
        zipped = stream.read(2) == b"\x1f\x8b"
        stream.seek(0)
        if zipped:
            with gzip.GzipFile(fileobj=stream) as decompressed:
                trace = json.load(decompressed)
        else:
            trace = json.load(stream)
    if trace.get("schema") != "phymeo-compliant-case-v1" or not trace.get("frames"):
        raise ValueError("expected a compliant-cloth trajectory")
    add_link_geometry(trace)
    add_backing_windows(trace)
    # The full source file retains every substep. The viewer reads the recorded frame data.
    trace.pop("samples", None)
    trace.pop("robot_samples", None)
    cap = dict(source_sha256=digest.hexdigest(), trajectory=trace)
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
