"""Audit paired-history HOST evidence from files; never import the solver or advance its state."""

import argparse
import hashlib
import json
from fractions import Fraction as Q
from math import comb
from pathlib import Path

import numpy as np
from audit_p65_joint_readback import arr, demand, exact_signal, map_from_coordinates, norm_inf, read

PLAN = Path(__file__).resolve().parents[1] / "docs/benchmarks/p65-paired-load-history-protocol.json"
PLAN_SHA = "a6379ca2a7b5094f73df58880223c0a5d835a602d54c6cf2d5811d75da110a24"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def independent_moments(macros, h):
    # Expand the degree-seven Bernstein basis to monomials, then integrate 1 and t.
    j = first = t = Q(0)
    for macro in macros:
        dt = Q(macro["dt"])
        words = [Q(float.fromhex(w)) for w in macro["words"]]
        demand(
            dt > 0 and t == Q(macro["t"]) and len(words) == 8 and min(words) >= 0,
            "gap or negative curve",
        )
        power = [
            sum(
                Q((-1) ** (k - i) * comb(7, i) * comb(7 - i, k - i)) * words[i]
                for i in range(k + 1)
            )
            for k in range(8)
        ]
        local_j = dt * sum(c / (k + 1) for k, c in enumerate(power))
        first += t * local_j + dt**2 * sum(c / (k + 2) for k, c in enumerate(power))
        j += local_j
        t += dt
    demand(t == h and 0 < first < h * j, "pressure time coverage")
    return j, h * j - first


def vertex_witnesses(x0, x1, q1, face, rho, t):
    x = np.array(
        [
            [Q(float(a)) + t * (Q(float(b)) - Q(float(a))) for a, b in zip(o, n, strict=True)]
            for o, n in zip(x0, x1, strict=True)
        ],
        object,
    )
    a, b, c = x[face]
    u = b - a
    v = c - a
    cx, cy, cz = u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]
    demand(
        cz > 0 and max(abs(k) for k in x[face, :2].flat) + rho < 2,
        "unsupported finite-patch witness",
    )
    # 1 - 1/sqrt(1+s) <= s/2. Both lift projections remain inside the square.
    top_correction = rho * (cx * cx + cy * cy) / (2 * cz * cz)
    middle = [v[2] - t * Q(float(q1)) for v in x[face]]
    demand(min(middle) + rho > 0, "upper lift below platform")
    lo = min(middle)
    hi = lo + top_correction
    return [float(lo), float(hi)]


def inspect(root, case, plan, base, source):
    packet_file = root / f"{case}.packet.json"
    data = packet_file.read_bytes()
    p = json.loads(data)
    index = read(root / f"{case}.packet-index.json")
    spec = plan["cases"][case]
    demand(
        p["case"] == case and p["protocol_sha256"] == PLAN_SHA and index["digest"] == sha(data),
        "packet binding",
    )
    h = Q(base["h_s"])
    rho = Q(base["cloth"]["half_thickness_m"])
    m = float(Q(base["cloth"]["m_kg"]))
    M = 4 * m
    expected_source = source[::-1] if spec["reverse"] else source
    for got, old in zip(p["macros"], expected_source, strict=True):
        t, dt = Q(old["t"]), Q(old["dt"])
        demand(
            Q(got["t"]) == (h - t - dt if spec["reverse"] else t)
            and Q(got["dt"]) == dt
            and got["words"] == (old["words"][::-1] if spec["reverse"] else old["words"]),
            "pressure curve substitution",
        )
    j, b = independent_moments(p["macros"], h)
    j_origin, b_origin = exact_signal(base)
    demand(
        j == j_origin == Q(index["j"])
        and b == Q(index["b"]) == (h * j - b_origin if spec["reverse"] else b_origin),
        "independent time moment",
    )
    faces = np.array(base["cloth"]["faces"])
    owners = p["owners"]
    shares = [Q(v) for v in spec["shares"]]
    demand(
        len(owners) == 6
        and sum(shares) == 1
        and p["axis_world"] == [0, 0, 1]
        and Q(p["duration"]) == h,
        "owner coverage",
    )
    for i, (owner, weight) in enumerate(zip(owners, shares, strict=True)):
        demand(
            (
                owner["surface"],
                owner["triangle"],
                owner["corner"],
                Q(owner["offset"]),
                Q(owner["share"]),
            )
            == (("U", "D")[i // 3], faces[i // 3].tolist(), i % 3, -rho, weight),
            "owner/weight binding",
        )
    ev = read(root / f"{case}-evidence.json.gz")
    pre = read(root / f"{case}-selected.json.gz")
    final = read(root / f"{case}-published.json.gz")
    status = read(root / f"{case}-status.json")
    demand(
        ev["packet_digest"] == index["digest"]
        and Q(ev["j"]) == j
        and Q(ev["b"]) == b
        and Q(ev["left"]) == b / h
        and Q(ev["right"]) == j - b / h,
        "delivery moments",
    )
    for trial in (pre, final):
        demand(
            trial["duration"] == float(h)
            and trial["result"]["report"]["accepted"]
            and len(trial["result"]["report"]["substeps"]) == 1,
            "inner replay",
        )
    step = final["result"]["report"]["substeps"][0]
    diag = step["diagnostics"]
    terminal = diag["terminal_impulse"]
    demand(
        terminal["applications"] == 1
        and terminal["ledger"]["max_residual"] <= 1e-8
        and terminal["pre_impulse_state"] == pre["result"]["state"]
        and terminal["pre_impulse_diagnostics"]
        == pre["result"]["report"]["substeps"][0]["diagnostics"],
        "selected binding or terminal count",
    )
    demand(
        final["effective_inputs"] == pre["effective_inputs"]
        and final["integrated_loads"] == pre["integrated_loads"] == ev["held_request"]
        and ev["selected_pre_state_bound"]
        and ev["selected_pre_diagnostics_bound"],
        "different held inputs",
    )
    initial = ev["original_state"]["domains"]
    state = final["result"]["state"]["domains"]
    selected = pre["result"]["state"]["domains"]
    demand(
        final["result"]["state"] == final["result"]["substep_states"][0]
        and all(state[d][f] == selected[d][f] for d, f in (("robot", "q"), ("cloth", "position"))),
        "configuration changed at terminal",
    )
    x0 = arr(initial["cloth"]["position"])
    v0 = arr(initial["cloth"]["velocity"])
    x1 = arr(state["cloth"]["position"])
    v1 = arr(state["cloth"]["velocity"])
    vp = arr(selected["cloth"]["velocity"])
    q1 = float(arr(state["robot"]["q"])[0])
    vr = float(arr(state["robot"]["v"])[0])
    vpr = float(arr(selected["robot"]["v"])[0])
    demand(
        np.array_equal(x0, base["cloth"]["nodes"])
        and not v0.any()
        and not arr(initial["robot"]["q"]).any()
        and not arr(initial["robot"]["v"]).any(),
        "different initial fixture",
    )
    maps = [ev["initial_map"], ev["endpoint_map"]]
    mom = []
    right_action = None
    for sample, x, v, records, total in zip(
        maps, (x0, x1), (v0, vp), ("left_world", "right_world"), (b / h, j - b / h), strict=True
    ):
        world = arr(ev[records])
        expected = np.zeros((6, 3))
        expected[:, 2] = [float(total * s) for s in shares]
        demand(np.array_equal(world, expected), "represented world shares")
        map_from_coordinates(sample, x, v, faces, float(rho), world)
        local = arr(sample["robot_local"])
        rp = arr(sample["robot_points"])
        robot_q = 0.0 if records == "left_world" else q1
        demand(
            norm_inf(rp - local - [0, 0, robot_q]) < 1e-12
            and max(sample["world_error"]) < base["bounds"]["geometry_proposal_m"]
            and all(sample["inside"]),
            "spatial action point binding",
        )
        demand(
            norm_inf(
                np.einsum("rjk,rj->k", arr(sample["robot_j"]), world) + arr(sample["robot_effort"])
            )
            < 1e-12,
            "robot point response",
        )
        mom.append(np.sum(np.cross(rp, -world), axis=0))
        if records == "right_world":
            right_action = world
    eff = final["effective_inputs"]["domains"]
    original = ev["original_inputs"]["domains"]
    loads = final["integrated_loads"]
    demand(
        all(not arr(v).any() for d in original.values() for v in d.values())
        and not arr(eff["robot"]["effort"]).any(),
        "original external inputs",
    )
    for domain, values in loads.items():
        for key, delivered in values.items():
            demand(
                np.array_equal(
                    float(h) * (arr(eff[domain][key]) - arr(original[domain][key])), arr(delivered)
                ),
                "represented load readback",
            )
    demand(
        norm_inf(arr(loads["robot"]["force"])[0] + arr(ev["left_world"]).sum(axis=0)) < 1e-12
        and norm_inf(arr(loads["robot"]["torque"])[0] - mom[0]) < 1e-12,
        "link force/couple lost",
    )
    cloth_req, rob_req = terminal["requests"]
    demand(
        cloth_req["domain"] == "cloth"
        and np.array_equal(arr(cloth_req["indices"]), np.arange(4))
        and np.array_equal(arr(cloth_req["impulses"]), arr(maps[1]["node"]))
        and rob_req["domain"] == "robot"
        and np.array_equal(arr(rob_req["indices"]), np.zeros(6))
        and np.array_equal(arr(rob_req["impulses"]), -right_action)
        and np.array_equal(arr(rob_req["local_points"]), arr(maps[1]["robot_local"])),
        "terminal point request",
    )
    demand(
        terminal["requests"] == terminal["ledger"]["requests"]
        and terminal["ledger"]["configuration_preserved"]
        and terminal["ledger"]["history_preserved"],
        "terminal certificate request",
    )
    gravity = np.array([0, 0, float(Q(base["cloth"]["gravity_z"]))])
    cd = diag["participants"]["cloth"]
    rd = diag["participants"]["robot"]
    demand(
        cd["material_path"]["input_fingerprint"]
        == ev["effective_cloth_input"]
        != ev["original_cloth_input"]
        and arr(cd["bending_multipliers"])[0] != 0,
        "material or input mismatch",
    )
    pred = arr(cd["material_path"]["predicted_velocity"])
    demand(
        norm_inf(m * (pred - v0 - float(h) * gravity) - arr(loads["cloth"]["force"])) < 1e-12
        and norm_inf(m * (v1 - vp) - arr(cloth_req["impulses"])) < 1e-12,
        "measured particle phase delivery",
    )
    cloth_impulse = np.sum(m * (v1 - v0 - float(h) * gravity), axis=0)
    robot_impulse = M * vr
    demand(
        abs(robot_impulse + float(j)) < 1e-12
        and abs(M * (vr - vpr) + float(j - b / h)) < 1e-12
        and norm_inf(cloth_impulse - [0, 0, float(j)]) < 1e-12,
        "two-sided momentum",
    )
    demand(
        abs(q1 + float(b) / M) <= base["bounds"]["reference_position_error_m"]
        and abs(vr + float(j) / M) < 1e-12
        and abs(float(arr(rd["applied"])[0]) * float(h) + float(b / h)) < 1e-12,
        "temporal discriminator",
    )
    spatial = mom[0] + mom[1]
    cloth_m = np.sum(np.cross(x1, m * v1) - np.cross(x0, m * (v0 + float(h) * gravity)), axis=0)
    angular_error = float(np.linalg.norm(cloth_m + spatial, ord=1))
    demand(angular_error <= base["bounds"]["joint_angular_nms"], "cloth angular balance")
    hard = []
    for face, sw, manifold in zip(faces, ev["sweeps"], maps[1]["manifolds"], strict=True):
        witness = Q(sw["witness_time"])
        last = vertex_witnesses(x0, x1, q1, face, rho, Q(1))
        mid = vertex_witnesses(x0, x1, q1, face, rho, witness)
        lower, upper = map(Q, manifold["gap"])
        demand(
            sw["status"] == "violation"
            and sw["reason"] == "complete SAT penetration witness"
            and 0 < witness <= 1
            and mid[1] < 0
            and last[1] < 0
            and lower <= upper < 0,
            "incomplete hard violation",
        )
        slope = min(Q(float(x1[k, 2])) - Q(q1) for k in face)
        demand(slope < 0, "expected immediate entry into hard core")
        hard.append(
            dict(
                sweep=sw["status"],
                witness_t_fraction=str(witness),
                witness_lower_vertex_gap_m=mid,
                endpoint_sat_gap_m=[float(lower), float(upper)],
                endpoint_lower_vertex_gap_m=last,
                initial_path_gap_derivative_per_unit_t_m=float(slope),
            )
        )
    mean_gap = float(np.mean(x1[:, 2]) - q1)
    pred_mean = float(h) ** 2 * gravity[2] + float(b) * (1 / (4 * m) + 1 / M)
    demand(abs(mean_gap - pred_mean) < 1e-12, "summed material drift inconsistent")
    workg = float(np.sum(m * gravity * (x1 - x0)))
    local = float(np.sum(arr(eff["cloth"]["force"]) * (x1 - x0)))
    demand(
        abs(cd["physical_audit"]["original_input_work"] - local) < 1e-15
        and status["successor_published"] is False
        and not status["joint_admitted"],
        "original work or publication eligibility",
    )
    return dict(
        case=case,
        packet_sha256=index["digest"],
        force_curve_sha256=p["macro_source_sha256"],
        macros=len(p["macros"]),
        j_ns=float(j),
        b_ns2=float(b),
        held_ns=float(b / h),
        terminal_ns=float(j - b / h),
        source_start_force_n=float.fromhex(p["macros"][0]["words"][0]),
        q_m=q1,
        reference_q_m=-float(b) / M,
        q_error_m=abs(q1 + float(b) / M),
        v_m_s=vr,
        terminal_applications=terminal["applications"],
        pre_state_and_diagnostics_match=True,
        cloth_impulse_ns=cloth_impulse.tolist(),
        platform_impulse_ns=robot_impulse,
        pair_z_residual_ns=float(cloth_impulse[2] + robot_impulse),
        platform_input_held_couple_nms=mom[0].tolist(),
        platform_terminal_point_couple_nms=mom[1].tolist(),
        platform_delivered_spatial_action_nms=spatial.tolist(),
        cloth_measured_angular_impulse_nms=cloth_m.tolist(),
        paired_angular_residual_l1_nms=angular_error,
        independent_guide_wrench_observed=False,
        raw_original_non_gravity_external_work_j=0.0,
        gravity_work_j=workg,
        cloth_local_effective_input_work_j=local,
        initial_layer_storage_j=0.001,
        eligible_full_step_layer_storage_dissipation=None,
        layer_domain="initially on the hard boundary; lower-vertex gap derivative negative at entry",
        hard=hard,
        mean_middle_gap_m=mean_gap,
        summed_current_update_gap_m=pred_mean,
        continuous_gravity_difference_m=-(float(h) ** 2) * gravity[2] / 2,
        inner_mechanical_accepted=True,
        joint_admitted=False,
        successor_published=False,
        source_bindings=[
            p[k]
            for k in ("scene_id", "original_inputs", "state_and_inputs", "material", "geometry")
        ],
        evidence_sha256={
            s: sha((root / f"{case}-{s}").read_bytes())
            for s in ("selected.json.gz", "published.json.gz", "evidence.json.gz", "status.json")
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    demand(sha(PLAN.read_bytes().replace(b"\r\n", b"\n")) == PLAN_SHA, "protocol")
    plan = read(PLAN)
    fixture = PLAN.with_name(plan["fixture"])
    base = read(fixture)
    demand(
        sha(fixture.read_bytes().replace(b"\r\n", b"\n")) == plan["fixture_sha256_lf"], "fixture"
    )
    sr = PLAN.with_name(plan["source"])
    demand(
        sha(sr.read_bytes().replace(b"\r\n", b"\n")) == plan["source_sha256_lf"], "source report"
    )
    origin = read(sr)
    f = PLAN.parents[2] / origin["raw_directory"] / "step-0" / "forces.jsonl"
    source = [json.loads(s) for s in f.read_bytes().splitlines()]
    demand(
        sha(f.read_bytes()) == origin["steps"][0]["hashes"]["forces.jsonl"] and len(source) == 63,
        "original 504 words",
    )
    rows = [inspect(root, case, plan, base, source) for case in ("forward", "reverse", "shifted")]
    a, b, c = rows
    demand(
        all(
            r["force_curve_sha256"] == sha(f.read_bytes())
            and r["source_bindings"] == a["source_bindings"]
            for r in rows
        ),
        "different scene or source",
    )
    split = abs(a["q_m"] - b["q_m"])
    shift = (
        c["platform_delivered_spatial_action_nms"][1]
        - a["platform_delivered_spatial_action_nms"][1]
    )
    demand(
        a["j_ns"] == b["j_ns"] == c["j_ns"]
        and abs(a["v_m_s"] - b["v_m_s"]) < 1e-12
        and split > 2 * base["bounds"]["reference_position_error_m"]
        and abs(a["q_m"] - c["q_m"]) <= base["bounds"]["reference_position_error_m"]
        and abs(a["v_m_s"] - c["v_m_s"]) < 1e-12
        and shift > a["j_ns"] / 16,
        "temporal or spatial contrast lost",
    )
    neg = read(root / "packet-controls.json")
    demand(
        {n["fault"] for n in neg} == {"time", "owner", "original", "digest"} and len(neg) == 4,
        "packet negative controls",
    )
    result = dict(
        schema="p65-n2-paired-load-history-result-v1",
        scope="HOST prescribed history with four-node B2 and 2 kg prismatic platform",
        protocol_sha256_lf=PLAN_SHA,
        raw_directory=root.relative_to(PLAN.parents[2]).as_posix(),
        cases=rows,
        packet_negative_controls=neg,
        timing_transport_pass=True,
        forward_vs_reverse_displacement_m=split,
        shifted_vs_forward_delivered_y_couple_nms=shift,
        measured_guide_torque=False,
        hard_sweeps_violated=6,
        joint_admitted=False,
        published_successors=0,
        decision="This representation distinguishes timing and spatial actions on the genuine participants. Six exact hard-sweep violations reject the three candidates. The prescribed scalar pressures are not a coupled four-node solution.",
    )
    with (root / "audited-result.json").open("x", encoding="utf8", newline="\n") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    for row in rows:
        print(
            row["case"],
            row["q_m"],
            row["hard"][0]["endpoint_sat_gap_m"][1],
            row["platform_delivered_spatial_action_nms"][1],
        )
    print("Independent discriminator complete; three joint candidates rejected.")


if __name__ == "__main__":
    main()
