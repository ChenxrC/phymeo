"""Independently audit saved joint readback evidence without importing the simulator or replaying a step."""

import argparse
import gzip
import hashlib
import json
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

PLAN = Path(__file__).resolve().parents[1] / "docs/benchmarks/p65-b2-joint-readback-protocol.json"
PLAN_SHA = "1b70ba85d8255f6e390626952ac84c466083fb755f8033f83d3e54c0fb7ab1b4"


def demand(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return json.loads(data)


def arr(record):
    data = np.asarray(record["values"], dtype=record["dtype"])
    demand(
        list(data.shape) == record["shape"] and np.isfinite(data).all(), "array shape or finiteness"
    )
    return data


def norm_inf(values):
    return float(np.max(np.abs(values)))


def exact_signal(plan):
    conf = plan["load_source"]
    report_path = PLAN.with_name(conf["report"])
    normalized = report_path.read_bytes().replace(b"\r\n", b"\n")
    demand(hashlib.sha256(normalized).hexdigest() == conf["sha256_lf"], "source report SHA")
    report = json.loads(normalized)
    packet = PLAN.parents[2] / report["raw_directory"] / "step-0" / "end.state.json"
    demand(
        hashlib.sha256(packet.read_bytes()).hexdigest()
        == conf["end_sha256"]
        == report["steps"][0]["hashes"]["end.state.json"],
        "calibration packet SHA",
    )
    end = read(packet)
    y, g = (Q(float.fromhex(end[k][0])) + Q(end[k][1]) for k in ("y", "gap"))
    m = Q(plan["cloth"]["m_kg"])
    h = Q(plan["h_s"])
    demand(Q(end["t"]) == h, "source time")
    return (m / 4) * h - 2 * m * (y - 1) / 1000, (m / 4) * h * h / 2 + 2 * m * g


def map_from_coordinates(record, x, v, faces, rho, world):
    ids = arr(record["triangles"])
    weights = arr(record["weights"])
    demand(
        np.array_equal(ids, np.repeat(faces, 3, axis=0))
        and np.array_equal(weights, np.tile(np.eye(3), (2, 1))),
        "material address identity",
    )
    demand(
        np.array_equal(arr(record["offsets"]), np.full(6, -rho)) and all(record["inside"]),
        "signed offset or finite patch support",
    )
    jac = arr(record["jacobians"])
    points = arr(record["points"])
    rates = arr(record["point_velocity"])
    reference = np.zeros((6, 3, 3, 3))
    bary = weights[:, :, None, None] * np.eye(3)
    nodes = np.zeros((4, 3))
    point_error = velocity_error = 0.0
    for owner, (face, weight) in enumerate(zip(ids, weights, strict=True)):
        tri = x[face]
        e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
        normal = np.cross(e1, e2)
        length = np.linalg.norm(normal)
        normal /= length
        expected_point = weight @ tri - rho * normal
        point_error = max(point_error, norm_inf(expected_point - points[owner]))
        for node in range(3):
            for axis in range(3):
                variation = np.zeros((3, 3))
                variation[node, axis] = 1
                dc = np.cross(variation[1] - variation[0], e2) + np.cross(
                    e1, variation[2] - variation[0]
                )
                reference[owner, node, :, axis] = (
                    weight[node] * np.eye(3)[:, axis]
                    - rho * (dc - normal * np.dot(normal, dc)) / length
                )
        local = np.einsum("ijk,j->ik", reference[owner], world[owner])
        np.add.at(nodes, face, local)
        velocity_error = max(
            velocity_error,
            norm_inf(np.einsum("ijk,ik->j", reference[owner], v[face]) - rates[owner]),
        )
    d = norm_inf(jac - reference)
    node_error = norm_inf(nodes - arr(record["node"]))
    couple = norm_inf(np.sum(np.cross(x, nodes), axis=0) - np.sum(np.cross(points, world), axis=0))
    power = abs(float(np.sum(nodes * v) - np.sum(world * rates)))
    demand(
        max(d, point_error, velocity_error, node_error, couple, power) < 1e-12,
        "full Jacobian, virtual work or wrench",
    )
    demand(norm_inf(jac - bary) > rho / 2, "barycentric-only negative")
    return dict(
        entries=int(reference.size),
        max_component_error=d,
        point_error_m=point_error,
        rate_error_m_s=velocity_error,
        bary_only_discrepancy=norm_inf(jac - bary),
        normal_only_discrepancy=norm_inf(jac[:, :, 2, :] - bary[:, :, 2, :]),
        mapped_node_error_ns=node_error,
        moment_residual_nms=couple,
        power_residual_j=power,
        public_ledger=record["mapping"],
    )


def chord_certificate(x0, x1, q0, q1, faces, patch, rho):
    demand(q0 == 0 and np.all(x0[:, 2] == 0), "affine certificate needs the original flat contact")
    corners = {(Q(float(u)), Q(float(v)), Q(float(z))) for u, v, z in patch}
    demand(
        corners == {(Q(i), Q(j), -rho) for i in (-2, 2) for j in (-2, 2)},
        "patch is not the prescribed plane",
    )

    def as_q(x):
        return np.array([[Q(float(val)) for val in row] for row in x], object)

    a, b = as_q(x0), as_q(x1)
    vertical = min(row[2] - Q(float(q1)) for row in b)

    def det(left, right):
        return left[0] * right[1] - left[1] * right[0]

    coefficients = []
    for face in faces:
        e, f = a[face[1]] - a[face[0]], a[face[2]] - a[face[0]]
        u, v = b[face[1]] - b[face[0]], b[face[2]] - b[face[0]]
        coefficients.append([det(e, f), det(e, f) + (det(u - e, f) + det(e, v - f)) / 2, det(u, v)])
    margin = 2 - rho - max(abs(value) for array in (a, b) for value in array[:, :2].flat)
    demand(
        vertical > 0 and all(k > 0 for coeff in coefficients for k in coeff) and margin > 0,
        "exact chord proof outside domain",
    )
    return dict(
        start_gap_m=0,
        min_z_minus_q_displacement_m=float(vertical),
        exact_min_z_minus_q=str(vertical),
        projected_area_bernstein=[[str(k) for k in coeff] for coeff in coefficients],
        finite_patch_coordinate_margin_m=float(margin),
        status="certified_for_reported_affine_middle_path",
        proof="For either lift the gap over the patch plane is t*(z_i(1)-q(1)) + rho*(1 +/- n_z(t)). Positive projected-area Bernstein coefficients define a unit normal for the full segment. Each prism is a convex hull of the lifted vertices. The closed-path minimum is zero at entry.",
    )


def audit(root):
    demand(
        hashlib.sha256(PLAN.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == PLAN_SHA,
        "protocol SHA",
    )
    plan = read(PLAN)
    auto = read(root / "automatic.json")
    time = read(root / "time-channel.json")
    i = read(root / "inlet.json.gz")
    t = read(root / "trial.json.gz")
    geo = read(root / "geometry.json.gz")
    binding = read(root / "bindings.json")
    demand(i["protocol_sha256_lf"] == auto["protocol_sha256_lf"] == PLAN_SHA, "protocol binding")
    demand(
        auto["admitted"] is False
        and auto["exception"] == "TypeError"
        and auto["reason"] == "materials require FP64 DistanceConstraints or None",
        "automatic constructor disposition",
    )
    demand(
        time["rejected"]
        and time["before_step"]
        and time["reason"] == "integrated load field is not a declared channel",
        "temporal input disposition",
    )
    demand(
        binding["caller_unchanged"] and i["scene"]["_identity"] == binding["scene_identity"],
        "original scene binding",
    )
    hq = Q(plan["h_s"])
    mq = Q(plan["cloth"]["m_kg"])
    h = float(hq)
    m = float(mq)
    M = 4 * m
    j, b = exact_signal(plan)
    world = arr(i["world"])
    jr = 6 * Q(float(world[0, 2]))
    demand(
        Q(i["signal_j"]) == j and Q(i["signal_b"]) == b and 0 < b < hq * j,
        "original source moments",
    )
    demand(
        np.array_equal(world, np.tile([0, 0, float(j / 6)], (6, 1))),
        "six prescribed world impulses",
    )
    report = t["result"]["report"]
    demand(
        report["accepted"]
        and report["backend"] == "numpy"
        and len(report["substeps"]) == len(t["result"]["substep_states"]) == 1
        and t["duration"] == h,
        "one HOST step",
    )
    step = report["substeps"][0]
    diag = step["diagnostics"]["participants"]
    path = diag["cloth"]["material_path"]
    demand(
        step["accepted"]
        and step["dt"] == h
        and path["input_fingerprint"] == binding["effective_cloth"] != binding["original_cloth"]
        and binding["readback_residual"] <= plan["bounds"]["host_tolerance"],
        "bound accepted material replay",
    )
    start = i["state"]["domains"]
    final = t["result"]["state"]["domains"]
    demand(final == t["result"]["substep_states"][0]["domains"], "recorded endpoint differs")
    x0, v0 = (arr(start["cloth"][k]) for k in ("position", "velocity"))
    x1, v1 = (arr(final["cloth"][k]) for k in ("position", "velocity"))
    q0 = float(arr(start["robot"]["q"])[0])
    q1 = float(arr(final["robot"]["q"])[0])
    vr = float(arr(final["robot"]["v"])[0])
    faces = np.asarray(plan["cloth"]["faces"])
    rho = Q(plan["cloth"]["half_thickness_m"])
    demand(
        np.array_equal(x0, plan["cloth"]["nodes"])
        and not v0.any()
        and q0 == 0
        and not arr(start["robot"]["v"]).any(),
        "initial data",
    )
    startmap = i["mapping"]
    endmap = geo["endpoint"]
    required = {tuple(Q(int(a == k)) for a in range(3)) + (-rho,) for k in range(3)}
    for entry, end in zip(startmap["manifolds"], endmap["manifolds"], strict=True):
        demand(
            entry["status"] == "contact"
            and all(Q(x) == 0 for x in entry["gap"])
            and {tuple(map(Q, row)) for row in entry["material"]} == required,
            "inlet manifold",
        )
        demand(end["status"] == "open" and Q(end["gap"][0]) > 0, "endpoint hard gap")
    maps = [
        map_from_coordinates(sample, x, v, faces, float(rho), world)
        for sample, x, v in ((startmap, x0, v0), (endmap, x1, v1))
    ]
    demand(
        maps[0]["normal_only_discrepancy"] == 0 < maps[1]["normal_only_discrepancy"],
        "normal map did not deform",
    )
    sweep = [
        dict(
            status=s["status"],
            reason=s["reason"],
            checked_intervals=s["checked_intervals"],
            input_fingerprint=s["input_fingerprint"],
        )
        for s in geo["sweeps"]
    ]
    demand(
        len(sweep) == 2 and all(s["status"] in ("unresolved", "certified") for s in sweep),
        "public sweep reports penetration or invalid state",
    )
    chord = chord_certificate(x0, x1, q0, q1, faces, arr(i["patch"]["local_vertices"]), rho)
    force = arr(t["integrated_loads"]["cloth"]["force"])
    effort = arr(t["integrated_loads"]["robot"]["effort"])
    demand(
        set(t["integrated_loads"]) == {"cloth", "robot"}
        and set(t["integrated_loads"]["cloth"]) == {"force"}
        and set(t["integrated_loads"]["robot"]) == {"effort"},
        "registered fields",
    )
    demand(
        i["request"] == t["integrated_loads"]
        and np.array_equal(force, arr(startmap["node"]))
        and np.array_equal(effort, arr(startmap["robot_effort"])),
        "represented request",
    )
    expected = np.zeros_like(force)
    expected[:, 2] = [float(jr / 3), float(jr / 3), float(jr / 6), float(jr / 6)]
    demand(
        np.array_equal(force, expected) and effort[0] == -float(jr),
        "prescribed direction or mass-share",
    )
    original = i["inputs"]["domains"]
    effective = t["effective_inputs"]["domains"]
    F0 = arr(original["cloth"]["force"])
    Feff = arr(effective["cloth"]["force"])
    demand(
        not F0.any()
        and not any(arr(v).any() for v in original["robot"].values())
        and np.array_equal(h * (Feff - F0), force)
        and np.array_equal(h * arr(effective["robot"]["effort"]), effort),
        "original and effective loads",
    )
    g = np.array([0, 0, float(Q(plan["cloth"]["gravity_z"]))])
    pred = arr(path["predicted_velocity"])
    predicted = m * (pred - v0 - h * g)
    cloth_momentum = np.sum(m * (v1 - v0 - h * g), axis=0)
    robot_momentum = M * vr
    pred_error = norm_inf(predicted - force)
    cloth_error = norm_inf(cloth_momentum - force.sum(axis=0))
    demand(
        max(pred_error, cloth_error, abs(robot_momentum - effort[0])) < 1e-12,
        "actual vs requested delivery",
    )
    demand(
        np.array_equal(arr(diag["robot"]["predicted_q"]), [q1])
        and np.array_equal(arr(diag["robot"]["predicted_v"]), [vr])
        and np.array_equal(arr(diag["robot"]["applied"]), effort / h),
        "platform published state or effort",
    )
    robot_held = -h * float(jr) / M
    robot_signal = -float(b) / M
    robot_reverse = -float(hq * j - b) / M
    demand(
        abs(q1 - robot_held) < 1e-15
        and abs(q1 - robot_signal) > plan["bounds"]["reference_position_error_m"]
        and abs(robot_reverse - robot_signal) > plan["bounds"]["reference_position_error_m"],
        "time-moment discriminator",
    )
    requested_moment = np.sum(np.cross(x0, force), axis=0)
    measured_moment = np.sum(np.cross(x1, m * v1) - np.cross(x0, m * (v0 + h * g)), axis=0)
    footprint = arr(startmap["robot_points"])
    partner = np.sum(np.cross(footprint, -world), axis=0)
    demand(
        norm_inf(requested_moment + partner) < 1e-12
        and norm_inf(requested_moment - measured_moment) < plan["bounds"]["joint_angular_nms"],
        "pair or cloth angular balance",
    )
    rob_j = arr(startmap["robot_j"])
    delta = np.cross(footprint[1] - footprint[0], -world[0])
    demand(
        np.array_equal(rob_j[0], rob_j[1]) and norm_inf(delta) > 1e-5,
        "robot wrench nonidentifiability counterexample",
    )
    dx = x1 - x0
    work0 = float(np.sum(F0 * dx))
    workg = float(np.sum(m * g * dx))
    workeff = float(np.sum(Feff * dx))
    work_robot = float(effort[0] / h * (q1 - q0))
    phys = diag["cloth"]["physical_audit"]
    demand(
        work0 == 0
        and abs(workeff - phys["original_input_work"]) < 1e-15
        and abs(workeff) > plan["bounds"]["joint_energy_j"]
        and abs(phys["gravity_after"] - phys["gravity_before"] + workg) < 1e-15,
        "local versus external work",
    )
    sweeps = path["sweeps"]
    bend = arr(sweeps[-1]["blocks"][1]["rows"])
    demand(
        all(
            [r["kind"] for r in s["blocks"]] == ["distance", "bending"]
            and arr(s["blocks"][0]["rows"]).shape == (3, 18)
            and arr(s["blocks"][1]["rows"]).shape == (1, 45)
            for s in sweeps
        )
        and bend[0, 37] != 0
        and bend[0, 41] == arr(diag["cloth"]["bending_multipliers"])[0] != 0
        and len(sweeps) == step["position_iterations"],
        "nonzero bending row trace",
    )
    return dict(
        schema="p65-n2-sc1-joint-readback-result-v1",
        scope="single HOST prescribed signal, real composite participants",
        raw_directory=root.relative_to(PLAN.parents[2]).as_posix(),
        protocol_sha256_lf=PLAN_SHA,
        archive_files={
            f: hashlib.sha256((root / f).read_bytes()).hexdigest()
            for f in (
                "automatic.json",
                "time-channel.json",
                "inlet.json.gz",
                "trial.json.gz",
                "bindings.json",
                "geometry.json.gz",
            )
        },
        automatic=auto,
        unregistered_time_channel=time,
        bindings=binding,
        transport_accepted=True,
        joint_gate_open=False,
        substep=dict(
            h_s=h,
            material_sweeps=len(sweeps),
            distance_rows=3 * len(sweeps),
            bend_rows=len(sweeps),
            angle_before_rad=float(bend[0, 36]),
            angle_after_rad=float(bend[0, 37]),
            bend_multiplier=float(bend[0, 41]),
            residuals=step["position"]["residuals"],
        ),
        maps=maps,
        geometry=dict(
            endpoint_gaps_m=[float(Q(e["gap"][0])) for e in endmap["manifolds"]],
            public_sweeps=sweep,
            chord=chord,
        ),
        delivery=dict(
            source_j_ns=float(j),
            source_b_ns2=float(b),
            six_owner_impulse_ns=world[0, 2],
            node_requests_ns=force.tolist(),
            predictor_error_ns=pred_error,
            final_cloth_ns=cloth_momentum.tolist(),
            final_cloth_error_ns=cloth_error,
            robot_momentum_ns=robot_momentum,
            pair_z_residual_ns=float(cloth_momentum[2] + robot_momentum),
            measured_robot_v_m_s=vr,
            measured_robot_q_m=q1,
            held_prediction_q_m=robot_held,
            signal_q_m=robot_signal,
            reversed_signal_q_m=robot_reverse,
            displacement_error_m=abs(q1 - robot_signal),
            position_limit_m=plan["bounds"]["reference_position_error_m"],
        ),
        moments=dict(
            cloth_requested_nms=requested_moment.tolist(),
            cloth_measured_nms=measured_moment.tolist(),
            platform_requested_nms=partner.tolist(),
            shift_point_moment_delta_nms=delta.tolist(),
            platform_contact_world_moment_observed=False,
        ),
        work=dict(
            original_applied_work_j=work0,
            gravity_work_j=workg,
            local_cloth_report_input_work_j=phys["original_input_work"],
            effective_cloth_work_j=workeff,
            effective_platform_work_j=work_robot,
            joint_layer_potential_and_dissipation_audited=False,
        ),
        material_ledger_fields=sorted(diag["cloth"]["material_audit"]),
        joint_evidence_missing=[
            "automatic B2 admission",
            "force time moment and motion binding",
            "measured spatial partner wrench",
            "original-input layer energy balance",
            "generic swept hard clearance",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    result = audit(root)
    with (root / "audited-result.json").open("x", encoding="utf8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("Independent transport checks passed; automatic joint gate stays closed.")
    print(json.dumps(result["delivery"]))


if __name__ == "__main__":
    main()
