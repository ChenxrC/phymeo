"""Compare bounded, sampled two-push references at an audited RR PushT boundary."""

import argparse
import gzip
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .pusht_pose_replan import _digest
from .pusht_public_scene import build_scene_spec, build_side_asset


def require(value, message):
    if not value:
        raise ValueError("two-push comparison: " + message)


def shortlisted(proposals, admissions, center, goal, base, radius, tolerance):
    """Filter by material, target direction and a necessary precontact reach bound."""
    desired = np.asarray(goal, dtype=float) - center
    distance = float(np.linalg.norm(desired))
    require(distance > 0 and radius > 0 and tolerance >= 0, "nondegenerate planar screen")
    desired /= distance
    admission = {(a["index"], a["distance_ordinal"]): a for a in admissions}
    proposal_keys = {(p["index"], p["distance_ordinal"]) for p in proposals}
    require(
        len(admission) == len(admissions) == len(proposals) and proposal_keys == set(admission),
        "complete material admission",
    )
    selected = []
    rows = []
    for proposal in proposals:
        key = proposal["index"], proposal["distance_ordinal"]
        source = admission[key]
        require(source["input_binding"] == proposal["input_binding"], "material proposal identity")
        excess = max(
            0.0,
            float(np.linalg.norm(np.array(proposal["precontact_center_world"][:2]) - base))
            - radius,
        )
        alignment = float(np.dot(desired, proposal["direction_world"][:2]))
        qualifies = bool(
            proposal["reference_usable"]
            and source["eligible"]
            and alignment >= np.sqrt(0.5)
            and excess <= tolerance
        )
        if qualifies and key[0] not in selected:
            selected.append(key[0])
        rows.append(
            dict(
                index=key[0],
                distance_ordinal=key[1],
                input_binding=proposal["input_binding"],
                alignment=alignment,
                precontact_excess_m=excess,
                material=source["eligible"],
                reference_usable=proposal["reference_usable"],
                shortlist=qualifies,
                original_selection=proposal["selection_reason"],
                position=proposal["point_world"],
                precontact=proposal["precontact_center_world"],
            )
        )
    return dict(
        indices=selected,
        rows=rows,
        base_xy_m=base.tolist(),
        radius_m=radius,
        tolerance_m=tolerance,
        target_direction=desired.tolist(),
    )


def planar_screen(driver, q, reference):
    """Read the original two-hinge kinematic chain, including a fixed tool offset."""
    import mujoco as mj

    from phymeo.control.manipulation_diagnostic import solve_native_point_ik

    model = driver.model
    data = mj.MjData(model)
    data.qpos[:] = q
    mj.mj_kinematics(model, data)
    joints = []
    body = int(model.geom_bodyid[driver.pusher])
    while body:
        count = int(model.body_jntnum[body])
        if count:
            require(count == 1, "one joint per moving ancestor")
            joints.append(int(model.body_jntadr[body]))
        body = int(model.body_parentid[body])
    joints.reverse()
    require(
        len(joints) == 2 and joints == [model.joint(n).id for n in driver.scene.robot_joint_names],
        "compiled planar RR chain",
    )
    for joint in joints:
        require(
            model.jnt_type[joint] == mj.mjtJoint.mjJNT_HINGE
            and np.allclose(abs(data.xaxis[joint]), (0, 0, 1), rtol=0, atol=1e-12),
            "world-vertical revolute joints",
        )
    points = [data.xanchor[j, :2] for j in joints] + [data.geom_xpos[driver.pusher, :2]]
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    tol = inspect.signature(solve_native_point_ik).parameters["tolerance_m"].default
    (material,) = [a for a in reference["route"]["attempts"] if a["kind"] == "material_admission"]
    result = shortlisted(
        reference["proposals"],
        material["proposals"],
        data.xpos[driver.object_body, :2],
        driver.target.position[:2],
        points[0],
        float(sum(lengths)),
        tol,
    )
    result.update(link_lengths_m=lengths.tolist(), starting_pusher_xy_m=points[-1].tolist())
    return result


def build_driver(original, candidate_path, candidate_sha, candidate_world, output):
    import mujoco as mj

    from scripts.run_manipulation_push import SOURCE_SHA
    from scripts.run_manipulation_push_feedback import PushFeedbackDriver

    require(_digest(candidate_path) == candidate_sha == SOURCE_SHA, "original candidate archive")
    with Path(candidate_path).open(encoding="utf-8") as stream:
        rows = json.load(stream)["rows"]
    (saved,) = [row for row in rows if row["summary"]["world"] == candidate_world]
    model = mj.MjModel.from_xml_string(original.mjcf)
    return PushFeedbackDriver(original, model, output, saved)


def run(report_path, audit_path, output_dir):
    from phymeo.training.manipulation_tasks import manipulation_scenes

    from .pusht_lookahead_candidates import freeze_material_faces
    from .pusht_lookahead_pairs import search_pairs
    from .pusht_lookahead_source import load_lookahead_source

    sources = load_lookahead_source(report_path, audit_path)
    entry, baseline = sources["entry"], sources["baseline"]
    with gzip.open(baseline["route_path"], "rt", encoding="utf-8") as stream:
        reference = json.load(stream)
    require(
        _digest(baseline["route_path"]) == baseline["route_sha256"], "bound first-stage reference"
    )
    (material,) = [x for x in reference["route"]["attempts"] if x["kind"] == "material_admission"]
    q = np.asarray(entry["state"]["q"], dtype=np.dtype(material["q_dtype"]))
    require(q.tolist() == reference["q"] == entry["state"]["q"], "lossless original planning q")

    (original,) = [s for s in manipulation_scenes() if s.key == "pusht-planar"]
    require(original.source_sha256 == entry["source_model_sha256"], "original RR scene")
    candidate = entry["driver_candidate"]
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=False)
    with TemporaryDirectory(dir=output_dir) as temp:
        driver = build_driver(
            original,
            candidate["path"],
            candidate["sha256"],
            candidate["world"],
            Path(temp) / "planner",
        )
        require(
            driver.query.input_binding == candidate["query_input_binding"], "original query binding"
        )
        goal = entry["goal"]
        require(
            tuple(driver.target.position[:2]) == tuple(goal["target_xy"])
            and driver.target.yaw == goal["target_yaw"]
            and driver.goal.position_tolerance_m == goal["position_tolerance_m"]
            and driver.goal.yaw_tolerance_rad == goal["yaw_tolerance_rad"],
            "original goal and tolerances",
        )
        counts = entry["planner_counts_before"]
        driver.planner.ik_calls = counts["ik_calls"]
        driver.planner.path_calls = counts["path_calls"]
        asset = build_side_asset(driver.model, side_coverage="all-lateral")
        model_key = build_scene_spec(original.mjcf, side_coverage="all-lateral").key

        def material_at(configuration):
            return freeze_material_faces(
                driver,
                configuration,
                asset,
                model_key=model_key,
                asset_sha256=entry["scene_asset_sha256"],
            )

        require(material_at(q).binding == material["binding"], "actual planning material geometry")
        screen = planar_screen(driver, q, reference)
        require(screen["indices"], "at least one first-push candidate for this comparison")
        search = search_pairs(
            driver,
            q,
            material_at,
            remaining_steps=sources["budget"]["remaining_steps"],
            first_indices=tuple(screen["indices"]),
            baseline_prediction=baseline["prediction"],
        )
    with (output_dir / "source.json").open("x", encoding="utf-8") as stream:
        json.dump(sources, stream, allow_nan=False)
    with (output_dir / "screen.json").open("x", encoding="utf-8") as stream:
        json.dump(screen, stream, allow_nan=False)
    with gzip.open(output_dir / "search.json.gz", "xt", encoding="utf-8") as stream:
        json.dump(search, stream, separators=(",", ":"), allow_nan=False)
    return dict(
        tick=entry["tick"],
        baseline=baseline["selected_index"],
        first_indices=screen["indices"],
        saved_search=str(output_dir / "search.json.gz"),
        physical_steps=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.report, args.audit, args.output_dir)))


if __name__ == "__main__":
    main()
