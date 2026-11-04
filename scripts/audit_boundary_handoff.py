"""Audit the stored accepted prefix of a soft handoff using original node IDs and material forces."""

import argparse
import gzip
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from audit_planar_handoff import finite_axis_distance


def read(path):
    with gzip.open(path, "rt", encoding="utf8") as f:
        return json.load(f)


def audit(observed, fixed):
    a = read(observed)
    ref = read(fixed)
    assert a["origin"]["receiver_control"] == "boundary-presentation"
    assert {k: v for k, v in a["origin"].items() if k != "receiver_control"} == ref["origin"]
    assert a["frames"][0]["robots"] == ref["frames"][0]["robots"]
    assert a["frames"][0]["cloth_position_m"] == ref["frames"][0]["cloth_position_m"]
    assert a["frames"][0]["cloth_velocity_m_s"] == ref["frames"][0]["cloth_velocity_m_s"]
    assert a["faces"] == ref["faces"] and a["patches"] == ref["patches"]
    count = Counter(tuple(sorted(edge)) for face in a["faces"] for edge in combinations(face, 2))
    boundary = {edge for edge, v in count.items() if v == 1}
    capsules = {c["key"] for c in a["robot_contact"]["capsules"] if c["domain"] == "B"}
    half_widths = [v["half_thickness_m"] for v in a["cloth_self_material"]["faces"]]
    axial_margin = max(half_widths) + next(
        v["width_m"] for v in a["patches"] if v["name"] == "B_left"
    )
    address = None
    history = []
    seen = set()
    for previous, f in zip(a["frames"][:-1], a["frames"][1:], strict=True):
        row = f["receiver_observation"] or {}
        decision = row.get("boundary_selection")
        if decision is None:
            continue
        assert decision["boundary_edges"] == len(boundary)
        for trial in decision["trials"]:
            assert tuple(trial["edge"]) in boundary and 0 <= trial["fraction"] <= 1
            if trial["obstruction"] is not None:
                link, face = trial["obstruction"]
                assert link == "receiver_ik" or (
                    link in capsules and face in {f"face_{i}" for i in range(len(a["faces"]))}
                )
        if decision["target"] is None:
            continue
        ids = tuple(decision["target"]["nodes"])
        w = decision["target"]["fraction"]
        assert ids in boundary and decision["trials"][-1]["obstruction"] is None
        if address is None:
            address = (ids, w)
        assert address == (ids, w)
        x = np.asarray(previous["cloth_position_m"])
        point = (1 - w) * x[ids[0]] + w * x[ids[1]]
        if "material_point_m" in row:
            np.testing.assert_allclose(row["material_point_m"], point, atol=4e-14, rtol=0)
        q = [np.mean(previous["patch_vertices_m"]["B_" + s], axis=0)[1] for s in ("left", "right")]
        applied_point = x[ids[0]] + w * (x[ids[1]] - x[ids[0]])
        exact_slack = min(applied_point[1] - q[0], q[1] - applied_point[1]) - axial_margin
        if decision["decision_time_s"] not in seen:
            seen.add(decision["decision_time_s"])
            assert abs(previous["t"] - decision["decision_time_s"]) < 1e-10
            assert exact_slack >= 0, (
                "chosen material point is outside the registered axial clearance"
            )
        history.append(
            dict(
                t=previous["t"],
                address_y_m=point[1],
                measured_axial_slack_m=exact_slack,
                original_edge_y_m=x[[13, 20], 1].tolist(),
                applied_action=row["action"],
                reason=row.get("reason"),
                arm_front_y_m=q,
            )
        )
    steps = a["substeps"]
    dt = a["dt_s"]
    h = dt / steps
    names = [v["name"] for v in a["patches"]]
    W = 0.048 * 9.81
    joints = 0.0
    max_error = 0.0
    weaker = {n: 0.0 for n in ("A", "B")}
    bilateral = {n: 0 for n in weaker}
    shared = 0
    pad_peaks = dict(left=0.0, right=0.0)
    link_impulses = {}
    link_normals = {}
    B_pad_impulses = {side: np.zeros(3) for side in ("left", "right")}
    for k, (body, sheet) in enumerate(zip(a["robot_samples"], a["self_samples"], strict=True)):
        assert sheet["registered_pairs"] == 276
        np.testing.assert_allclose(np.sum(sheet["nodal_force_n"], axis=0), 0, atol=1e-10)
        patches = {s["patch"]: s for s in a["samples"][k * len(names) : (k + 1) * len(names)]}
        assert set(patches) == set(names)
        if k % steps == 0:
            imp = np.array([0.0, 0.0, -W * dt])
        for s in patches.values():
            imp += h * np.array(s["force_n"])
        for side in B_pad_impulses:
            pad_peaks[side] = max(pad_peaks[side], patches["B_" + side]["normal_n"])
            B_pad_impulses[side] += h * np.array(patches["B_" + side]["force_n"])
        joint_forces = {}
        for s in body["cloth_links"]:
            force = sum((np.asarray(site["force_on_cloth_n"]) for site in s["sites"]), np.zeros(3))
            imp += h * force
            if s["robot"] == "B":
                link = s["link"]
                link_impulses[link] = link_impulses.get(link, np.zeros(3)) + h * force
                joint_forces[link] = joint_forces.get(link, 0.0) + s["normal_force_n"]
        for name, f in joint_forces.items():
            link_normals[name] = max(link_normals.get(name, 0), f)
        joints = max(body["joint_violation"], joints)
        low = {n: min(patches[n + "_" + s]["normal_n"] for s in ("left", "right")) for n in weaker}
        up = {n: sum(patches[n + "_" + s]["force_n"][2] for s in ("left", "right")) for n in weaker}
        for n, v in low.items():
            weaker[n] = max(weaker[n], v)
            bilateral[n] += int(v > 0.20)
        shared += int(
            min(low.values()) > 0.20
            and min(up.values()) > 0.08 * W
            and patches["table"]["force_n"][2] < 0.02 * W
        )
        if (k + 1) % steps == 0:
            i = k // steps
            first, last = (
                np.asarray(fr["cloth_velocity_m_s"]).mean(0)
                for fr in (a["frames"][i], a["frames"][i + 1])
            )
            max_error = max(max_error, float(np.max(np.abs(0.048 * (last - first) - imp))))
    assert len(a["robot_samples"]) == (len(a["frames"]) - 1) * steps and max_error < 1e-10
    last = a["frames"][-1]
    other = next(f for f in ref["frames"] if abs(f["t"] - last["t"]) < 1e-10)

    def finite_structure(frame):
        nodes = np.asarray(frame["cloth_position_m"])
        cap_table = {
            item["key"]: item for item in a["robot_contact"]["capsules"] if item["domain"] == "B"
        }
        values = []
        for name, cap in cap_table.items():
            axis = np.asarray(frame["body_capsules_m"][name])
            for i, face in enumerate(a["faces"]):
                margin = cap["radius_m"] + half_widths[i] + a["cloth_link_material"]["width_m"]
                values.append((finite_axis_distance(axis, nodes[face]) - margin, name, i))
        current = min(values)
        original = next(row for row in values if row[1:] == ("B_wrist", 2))
        return dict(
            t=frame["t"],
            closest_layer_clearance_m=current[0],
            closest_link=current[1],
            closest_face=current[2],
            original_B_wrist_face2_layer_clearance_m=original[0],
        )

    summary = dict(
        action_counts=dict(Counter(item["applied_action"] for item in history)),
        actual_end_surface=finite_structure(last),
        prior_fixed_end_surface=finite_structure(other),
        actual_B_single_pad_normal_peaks_n=pad_peaks,
        start_s=a["frames"][0]["t"],
        end_s=last["t"],
        substeps=len(a["self_samples"]),
        entry_hash=a["origin"]["entry_state"],
        boundary_edges=len(boundary),
        address=address,
        first_and_last_tracked_samples=[] if not history else [history[0], history[-1]],
        tracked_input_frames=len(history),
        true_bilateral_count=bilateral,
        true_weaker_peak_n=weaker,
        shared_substeps=shared,
        joint_violation=joints,
        actual_B_link_peak_n=link_normals,
        actual_B_link_impulse_ns={k: v.tolist() for k, v in link_impulses.items()},
        actual_B_pad_impulse_ns={k: v.tolist() for k, v in B_pad_impulses.items()},
        max_sheet_momentum_residual_ns=max_error,
        peak_matched_end_node_displacement_m=float(
            np.linalg.norm(
                np.array(last["cloth_position_m"]) - other["cloth_position_m"], axis=1
            ).max()
        ),
        raw_decisions=a["metrics"]["boundary_decisions"],
        control_gate=a["metrics"]["shared_gate"],
        stop=a["metrics"]["stop"],
    )
    return summary


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("observed", type=Path)
    cli.add_argument("fixed", type=Path)
    cli.add_argument("output", type=Path)
    a = cli.parse_args()
    out = audit(a.observed, a.fixed)
    with a.output.open("x", encoding="utf8") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in out.items() if k != "raw_decisions"}))
