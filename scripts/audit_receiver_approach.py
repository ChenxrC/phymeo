"""Audit actual first-layer geometry and the spread of forecasts in a saved receiver replay."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from phymeo.constraints.collision_shapes import CollisionKey, CollisionMotion, CollisionShape
from phymeo.core.robot import RobotModel
from phymeo.dynamics.robot_kinematics import forward_kinematics, point_jacobians
from phymeo.geometry._capsule_triangle_sweep import capsule_face_query


def first_pair_motion(trace, frame, samples, poses, model):
    """A same-timestamp geometric attribution after the solve, in metres."""
    width = trace["cloth_link_material"]["width_m"]
    for index, sample in enumerate(samples):
        active = next((r for r in sample["cloth_links"] if r["robot"] == "B"), None)
        if active is None:
            continue
        name = active["link"]
        geometry = next(c for c in trace["robot_contact"]["capsules"] if c["key"] == name)
        radius = geometry["radius_m"]
        capsule = CollisionShape(CollisionKey(0, 1, 0), "capsule", (0, 1), radius)
        face = CollisionShape(CollisionKey(0, 0, 0), "triangle", (0, 1, 2))

        def margin(ends, nodes):
            witness = capsule_face_query(
                capsule, face, CollisionMotion(ends), CollisionMotion(nodes)
            ).sample(0)[0]
            return float(witness.gap_m - active["half_thickness_m"] - width)

        ids = active["face_nodes"]
        x0 = np.asarray(frame["cloth_position_m"])[ids]
        v0 = np.asarray(frame["cloth_velocity_m_s"])[ids]
        x1 = np.asarray(active["face_positions_m"])
        ends0 = np.asarray(frame["body_capsules_m"][name])
        ends1 = np.asarray(active["axis_ends_m"])
        tau = sample["t"] - frame["t"]
        actual = margin(ends1, x1)
        np.testing.assert_allclose(actual, active["gap_m"] - width, rtol=0, atol=2e-12)
        at_load = frame["robots"]["B"] if index == 0 else poses[index - 1]
        q, qv = np.asarray(at_load["q"]), np.asarray(at_load["v"])
        link = geometry["link_index"]
        pose = forward_kinematics(model, q)
        normal = np.asarray(active["normal"])
        normal_rates = []
        for site in active["sites"]:
            local = pose.rotation[link].T @ (np.asarray(site["point_m"]) - pose.position[link])
            jac, _ = point_jacobians(model, q, [link], local[None, :])
            link_rate = float(normal @ (jac[0] @ qv))
            relative = float(normal @ np.asarray(site["relative_velocity_m_s"]))
            np.testing.assert_allclose(relative, site["normal_speed_m_s"], atol=2e-12, rtol=0)
            normal_rates.append(
                dict(
                    face_weights=site["face_weights"],
                    axis_parameter=site["axis_parameter"],
                    normal=normal.tolist(),
                    link_normal_m_s=link_rate,
                    material_normal_m_s=link_rate + relative,
                    relative_m_s=relative,
                    normal_force_n=site["normal_force_n"],
                )
            )
        return dict(
            sample_s=sample["t"],
            pair=[name, active["face"]],
            face_nodes=ids,
            normal_rates=normal_rates,
            entry_m=margin(ends0, x0),
            entry_capsule_measured_face_m=margin(ends0, x1),
            measured_capsule_entry_face_m=margin(ends1, x0),
            measured_m=actual,
            max_actual_material_node_motion_m=float(np.linalg.norm(x1 - x0, axis=1).max()),
            max_actual_capsule_end_motion_m=float(np.linalg.norm(ends1 - ends0, axis=1).max()),
            max_material_forecast_error_at_first_sample_m=float(
                np.linalg.norm(x1 - x0 - tau * v0, axis=1).max()
            ),
        )
    return None


def pad_window(samples, count):
    """Summarize the two measured pad normals on each hand at every original force time."""
    width = len(samples) // count
    active = bilateral = both = None
    min_a = np.inf
    max_b = 0.0
    max_b_sides = np.zeros(2)
    for i in range(0, len(samples), width):
        step = samples[i : i + width]
        force = {r["patch"]: r["normal_n"] for r in step}
        assert len(force) == width and len({r["t"] for r in step}) == 1
        a = min(force["A_left"], force["A_right"])
        b = np.array([force["B_left"], force["B_right"]])
        min_a = min(min_a, a)
        max_b = max(max_b, float(min(b)))
        max_b_sides = np.maximum(max_b_sides, b)
        if active is None and max(b) > 0:
            active = step[0]["t"]
        if bilateral is None and min(b) > 0.20:
            bilateral = step[0]["t"]
        if both is None and min(a, min(b)) > 0.20:
            both = step[0]["t"]
    return dict(
        min_A_weaker_pad_n=float(min_a),
        peak_B_weaker_pad_n=max_b,
        peak_B_left_right_n=max_b_sides.tolist(),
        first_B_pad_force_s=active,
        first_B_bilateral_above_0p2_s=bilateral,
        first_both_arms_bilateral_above_0p2_s=both,
    )


def audit(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        trace = json.load(stream)
    stats = trace["metrics"]
    count = stats["substeps"]
    rows = []
    model = RobotModel(**trace["robot_models"]["B"])
    assert len(trace["robot_samples"]) == len(stats["rows"]) * count
    assert len(trace["receiver_substep_motion"]) == len(trace["robot_samples"])
    assert len(trace["self_samples"]) == len(trace["robot_samples"])
    patch_count = len(trace["patches"])
    assert len(trace["samples"]) == patch_count * len(trace["robot_samples"])
    for i, (frame, last, stored) in enumerate(
        zip(trace["frames"][:-1], trace["frames"][1:], stats["rows"], strict=True)
    ):
        samples = trace["robot_samples"][i * count : (i + 1) * count]
        poses = trace["receiver_substep_motion"][i * count : (i + 1) * count]
        pads = trace["samples"][i * count * patch_count : (i + 1) * count * patch_count]
        fc = stored["prediction"] or {}
        tips = np.asarray(
            [c["first_tool_m"] for c in fc.get("candidates", ()) if c["first_tool_m"] is not None]
        )
        spread = (
            float(np.linalg.norm(tips[:, None, :] - tips[None, :, :], axis=2).max())
            if len(tips)
            else None
        )
        x0 = np.asarray(frame["cloth_position_m"])
        v0 = np.asarray(frame["cloth_velocity_m_s"])
        x1 = np.asarray(last["cloth_position_m"])
        result = stored["physical_path"]
        old = stored["original_physical_path"]
        rows.append(
            dict(
                entry_s=frame["t"],
                end_s=last["t"],
                selected=fc.get("selected"),
                candidates=len(fc.get("candidates", ())),
                admissible_forecast=fc.get("admissible"),
                candidates_with_forecast_stop=sum(
                    c["reason"] is None for c in fc.get("candidates", ())
                ),
                max_first_predicted_tool_spread_m=spread,
                first_measured_control_error_m=stored["first_pose_forecast_error_m"],
                true_tool_displacement_m=float(
                    np.linalg.norm(np.asarray(stored["actual_tool_m"]) - stored["entry_tool_m"])
                ),
                tool_deviation_from_original_m=float(
                    np.linalg.norm(np.asarray(stored["actual_tool_m"]) - stored["original_tool_m"])
                ),
                max_material_forecast_error_m=float(
                    np.linalg.norm(x1 - x0 - stats["dt_s"] * v0, axis=1).max()
                ),
                min_sampled_gap_m=result["minimum_sampled_gap_m"],
                old_min_sampled_gap_m=None if old is None else old["minimum_sampled_gap_m"],
                first_blocked=result["first_blocked_drift"],
                peak_speed_m_s=result["peak_sampled_tool_speed_m_s"],
                first_pair_geometry=first_pair_motion(trace, frame, samples, poses, model),
                structure=stored["structure"],
                pads=pad_window(pads, count),
                max_accepted_penetration_m=max(r["penetration_m"] for r in pads),
                max_measured_joint_violation=max(s["joint_violation"] for s in samples),
                shared_support=stored["shared_support"],
            )
        )
    return dict(
        path=path.name,
        status=stats["status"],
        original_replay_max_abs=stats["old_drive_replay_max_abs"],
        speed_budget_m_s=stats["motion_budget"]["peak_speed"],
        physical_substeps=count * len(rows),
        rows=rows,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("traces", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    results = [audit(path) for path in args.traces]
    with args.out.open("x", encoding="utf8") as out:
        json.dump(results, out, indent=2, allow_nan=False)
    for r in results:
        print(
            json.dumps(
                dict(
                    path=r["path"],
                    substeps=r["physical_substeps"],
                    first_contact=next(
                        (
                            row["first_pair_geometry"]
                            for row in r["rows"]
                            if row["first_pair_geometry"] is not None
                        ),
                        None,
                    ),
                )
            )
        )
