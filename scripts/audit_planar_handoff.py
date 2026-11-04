"""Independent material, motor, geometric and force audit on original and paired soft handoff traces."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from phymeo.geometry.surface_features import closest_point_triangle, closest_segment_segment


def read(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        return json.load(stream)


def finite_axis_distance(axis, tri):
    """Distance of a segment to a closed triangle using both caps, all edges and a planar piercing."""
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    height = normal @ (axis[0] - tri[0])
    rate = normal @ (axis[1] - axis[0])
    if rate and 0 <= -height / rate <= 1:
        p = axis[0] - (height / rate) * (axis[1] - axis[0])
        if all(normal @ np.cross(tri[(i + 1) % 3] - tri[i], p - tri[i]) >= 0 for i in range(3)):
            return 0.0
    return min(
        *(closest_point_triangle(p, tri).distance for p in axis),
        *(closest_segment_segment(axis, tri[[i, (i + 1) % 3]]).distance for i in range(3)),
    )


def ranges(t, h):
    result = []
    for value in t:
        if not result or value - result[-1][1] > 1.5 * h:
            result.append([value, value])
        else:
            result[-1][1] = value
    return result


def audit(original, reference, observed, expected, output):
    src = read(original)
    new = read(observed)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == new["origin"]["sha256"]
    dt = new["dt_s"]
    steps = new["substeps"]
    h = dt / steps
    old = {round(f["t"] / dt): f for f in src["frames"]}
    for name in reference:
        old.update((round(f["t"] / dt), f) for f in read(name)["frames"])
    recorded = new["frames"]
    a0 = recorded[0]
    idx = round(a0["t"] / dt)
    for field in ("cloth_position_m", "cloth_velocity_m_s", "robots"):
        assert a0[field] == old[idx][field], field
    mass = src["fabric"]["mass_kg"]
    weight = mass * 9.81
    total_steps = steps * (len(recorded) - 1)
    assert new["dt_s"] == src["dt_s"] and new["substeps"] == src["substeps"]
    for name in (
        "faces",
        "patches",
        "robot_models",
        "robot_contact",
        "cloth_self_material",
        "cloth_link_material",
    ):
        assert new[name] == src[name], name
    assert len(new["self_samples"]) == len(new["robot_samples"]) == total_steps
    patch_count = len(new["patches"])
    assert len(new["samples"]) == patch_count * total_steps
    error = 0.0
    moment_self = 0.0
    peak_pen = 0.0
    joint = 0.0
    wrist = np.zeros(3)
    impulse = np.zeros(3)
    times = {k: [] for k in ("A", "B", "both", "shared")}
    peak_b_pad = 0.0
    peak_b_bilateral = 0.0
    peak_b_link = 0.0
    for k, (sheet, body) in enumerate(zip(new["self_samples"], new["robot_samples"], strict=True)):
        if k % steps == 0:
            impulse = np.array([0.0, 0.0, -weight * dt])
        time = recorded[0]["t"] + k * h
        assert (
            abs(time - sheet["t"]) < 1e-10
            and abs(time - body["t"]) < 1e-10
            and sheet["registered_pairs"] == 276
        )
        joint = max(joint, body["joint_violation"])
        sf = np.sum(sheet["nodal_force_n"], axis=0)
        moment_self = max(moment_self, float(np.max(abs(sf))))
        group = {p["patch"]: p for p in new["samples"][k * patch_count : (k + 1) * patch_count]}
        assert all(abs(p["t"] - time) < 1e-10 for p in group.values())
        for p in group.values():
            impulse += h * np.array(p["force_n"])
            peak_pen = max(peak_pen, p["penetration_m"])
        peak_b_link = max(
            peak_b_link, sum(c["normal_force_n"] for c in body["cloth_links"] if c["robot"] == "B")
        )
        for c in body["cloth_links"]:
            for s in c["sites"]:
                load = h * np.array(s["force_on_cloth_n"])
                impulse += load
                if c["link"] == "B_wrist":
                    wrist += load
        norm = {
            n: min(group[n + "_" + side]["normal_n"] for side in ("left", "right"))
            for n in ("A", "B")
        }
        up = {
            n: sum(group[n + "_" + side]["force_n"][2] for side in ("left", "right"))
            for n in ("A", "B")
        }
        peak_b_pad = max(
            peak_b_pad, sum(group["B_" + side]["normal_n"] for side in ("left", "right"))
        )
        peak_b_bilateral = max(peak_b_bilateral, norm["B"])
        for n in ("A", "B"):
            if norm[n] > 0.20:
                times[n].append(time)
        if min(norm.values()) > 0.20:
            times["both"].append(time)
            if min(up.values()) > 0.08 * weight and group["table"]["force_n"][2] < 0.02 * weight:
                times["shared"].append(time)
        if (k + 1) % steps == 0:
            i = k // steps
            p0 = np.asarray(recorded[i]["cloth_velocity_m_s"]).mean(0) * mass
            p1 = np.asarray(recorded[i + 1]["cloth_velocity_m_s"]).mean(0) * mass
            error = max(error, float(np.max(abs((p1 - p0) - impulse))))
            for n in ("A", "B"):
                limits = new["origin"]["force_limits"][n]
                assert np.all(np.abs(recorded[i + 1]["actuator_effort"][n]) <= limits)
    assert error < 1e-11 and moment_self < 1e-10
    original_block = next(
        p
        for p in expected["probes"]
        if p["recorded_fraction"] == 0 and p["recorded_endpoint_blocker"] is not None
    )
    wanted_time = original_block["input_s"]
    which = round(wanted_time / dt)
    link, face = original_block["recorded_endpoint_blocker"]
    ids = new["faces"][int(face.split("_")[-1])]
    cap = next(c for c in new["robot_contact"]["capsules"] if c["key"] == link)
    thickness = src["fabric"]["half_thickness_m"]
    margin = cap["radius_m"] + thickness + new["cloth_link_material"]["width_m"]
    receiver = np.arange(
        src["fabric"]["grid"][0] - 1, np.prod(src["fabric"]["grid"]), src["fabric"]["grid"][0]
    )
    receiver = receiver[int(np.ceil((len(receiver) - 1) / 2)) :]
    pad = next(p for p in src["patches"] if p["name"] == "B_left")
    all_caps = {c["key"]: c for c in new["robot_contact"]["capsules"] if c["domain"] == "B"}
    limits = src["robot_contact"]["limits"]["B"]
    model = src["robot_models"]["B"]
    assert model["joint_types"] == ["revolute", "revolute", "prismatic", "prismatic"]
    axes = np.asarray(model["joint_axis"])
    np.testing.assert_array_equal(np.abs(axes), np.tile([0, 1, 0], (4, 1)))
    np.testing.assert_array_equal(model["joint_orientation"], np.tile([0, 0, 0, 1], (4, 1)))

    def aperture_fields(f):
        x = np.asarray(f["cloth_position_m"])
        q = f["robots"]["B"]["q"]
        centers = [
            float(np.mean(f["patch_vertices_m"]["B_" + s], axis=0)[1]) for s in ("left", "right")
        ]
        reserve = thickness + pad["width_m"]
        limits_y = [
            c + axes[j, 1] * (v[j] - q[j])
            for j, c in enumerate(centers, 2)
            for v in (limits["lower"], limits["upper"])
        ]
        actual = [min(centers) + reserve, max(centers) - reserve]
        widest = [float(min(limits_y) + reserve), float(max(limits_y) - reserve)]
        edge = [float(x[receiver, 1].min()), float(x[receiver, 1].max())]

        def overlap(band):
            return bool(max(band[0], edge[0]) <= min(band[1], edge[1]))

        return dict(
            available_axial_y_m=actual,
            widest_legal_y_m=widest,
            receiver_edge_y_m=edge,
            material_overlaps_axial_band=overlap(actual),
            material_overlaps_widest_legal_band=overlap(widest),
        )

    def geometry(f):
        x = np.asarray(f["cloth_position_m"])
        ends = np.asarray(f["body_capsules_m"][link])
        triangle = x[ids]
        gap = finite_axis_distance(ends, triangle) - margin
        mirror = np.array([-1.0, 1.0, 1.0])
        assert abs(finite_axis_distance(ends * mirror, triangle * mirror) - margin - gap) < 1e-11
        empty = finite_axis_distance(ends, triangle + [0, 1.0, 0]) - margin
        assert empty > 0.5
        closest = min(
            (
                finite_axis_distance(np.asarray(f["body_capsules_m"][name]), x[face_ids])
                - c["radius_m"]
                - thickness
                - new["cloth_link_material"]["width_m"],
                name,
                i,
            )
            for name, c in all_caps.items()
            for i, face_ids in enumerate(new["faces"])
        )
        return dict(
            t=f["t"],
            link=link,
            face=face,
            clearance_m=gap,
            mirrored_clearance_m=gap,
            empty_clearance_m=empty,
            **aperture_fields(f),
            closest_actual_structure=list(closest),
            tools_xz_m=f["actual_tool_xz_m"],
        )

    old_geometry = geometry(old[which])
    assert (
        abs(old_geometry["clearance_m"] - original_block["measured_start_clearance_to_layer_m"])
        < 1e-10
    )
    lookup = {round(f["t"] / dt): f for f in recorded}
    comparison = geometry(lookup[which]) if which in lookup else None
    last = recorded[-1]
    same = old[round(last["t"] / dt)]
    delta = {
        n: (np.asarray(last["actual_tool_xz_m"][n]) - same["actual_tool_xz_m"][n]).tolist()
        for n in ("A", "B")
    }
    result = dict(
        tilt_deg=src["initial_tilt_deg"],
        start_s=a0["t"],
        end_s=last["t"],
        substeps=total_steps,
        actual_substep_B_peak_total_pad_n=peak_b_pad,
        actual_substep_B_peak_weaker_pad_n=peak_b_bilateral,
        actual_substep_B_peak_structural_normal_n=peak_b_link,
        original_obstruction=old_geometry,
        new_geometry_at_that_same_time=comparison,
        final_new_minus_original_tool_xz_m=delta,
        peak_cloth_position_difference_at_matched_end_m=float(
            np.max(
                np.linalg.norm(
                    np.array(last["cloth_position_m"]) - same["cloth_position_m"], axis=1
                )
            )
        ),
        true_new_bilateral_intervals={n: ranges(ts, h) for n, ts in times.items()},
        peak_pad_penetration_m=peak_pen,
        max_joint_violation=joint,
        max_cloth_momentum_error_ns=error,
        max_sum_internal_self_n=moment_self,
        actual_B_wrist_impulse_ns=wrist.tolist(),
        physical_stop=new["metrics"]["stop"],
        new_predicted_blockers=new["metrics"]["blocked"],
        entry_fingerprint=new["origin"]["entry_state"],
        source_sha256=new["origin"]["sha256"],
        impossible_axial_at_measured_aperture_s=ranges(
            [f["t"] for f in recorded if not aperture_fields(f)["material_overlaps_axial_band"]], dt
        ),
        impossible_axial_even_with_legal_finger_stroke_s=ranges(
            [
                f["t"]
                for f in recorded
                if not aperture_fields(f)["material_overlaps_widest_legal_band"]
            ],
            dt,
        ),
        original_same_times_impossible_even_with_legal_finger_stroke_s=ranges(
            [
                f["t"]
                for f in recorded
                if not aperture_fields(old[round(f["t"] / dt)])[
                    "material_overlaps_widest_legal_band"
                ]
            ],
            dt,
        ),
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--window", type=Path, required=True)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--original-options", type=Path, required=True)
    p.add_argument("--variant", choices=("vertical", "tilted"), required=True)
    p.add_argument("--observed", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    options = next(
        row
        for row in json.loads(a.original_options.read_text(encoding="utf8"))
        if row["variant"] == a.variant
    )
    audit(a.original, (a.window, a.capture), a.observed, options, a.output)


if __name__ == "__main__":
    main()
