"""Independent file-only checks of original gravity, full material addresses and the conditional RK proposal."""

import argparse
import hashlib
import json
from fractions import Fraction as Q
from math import atan2, inf, nextafter
from pathlib import Path

import numpy as np
from audit_p65_joint_readback import (
    arr,
    chord_certificate,
    demand,
    map_from_coordinates,
    norm_inf,
    read,
)
from audit_p65_paired_load_history import independent_moments, vertex_witnesses

PLAN = Path(__file__).resolve().parents[1] / "docs/benchmarks/p65-gravity-six-owner-protocol.json"
PLAN_SHA = "def848e6494eab9148ac924fc410bea400f9468a7446b1b697776a851beaa690"
K = Q(500)
WIDTH = Q(1, 500)
DAMP = Q(1000)
AREA = Q(1, 6)


def file_sha(path, normalize=False):
    data = path.read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if normalize else data).hexdigest()


def qr(x):
    return Q(float(x))


def display(bounds):
    return [nextafter(float(bounds[0]), -inf), nextafter(float(bounds[1]), inf)]


def deviation(value, bounds):
    return max(bounds[0] - value, value - bounds[1], Q(0))


def total_bounds(boxes):
    return tuple(sum((box[k] for box in boxes), Q(0)) for k in (0, 1))


def tilt_error(x0, x1, face, rho, h):
    p = np.array([[qr(a) for a in row] for row in x0[face]], object)
    y = np.array([[qr(a) for a in row] for row in x1[face]], object)
    u, v = p[1] - p[0], p[2] - p[0]
    du, dv = y[1] - y[0] - u, y[2] - y[0] - v

    def cross(a, b):
        return np.array(
            [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
            object,
        )

    c0 = cross(u, v)
    c1 = cross(du, v) + cross(u, dv)
    c2 = cross(du, dv)
    coeff = np.array([c0, c0 + c1 / 2, c0 + c1 + c2], object)
    low = min(coeff[:, 2])
    demand(
        low > 0 and max(abs(k) for a in (p, y) for k in a[:, :2].flat) + rho < 2,
        "finite support or orientation",
    )
    maxima = [max(abs(k) for k in coeff[:, a]) for a in range(2)]
    deriv = [max(abs(c1[i]), abs(c1[i] + 2 * c2[i])) for i in range(3)]
    r = [a / low for a in maxima]
    dr = [(deriv[i] + r[i] * deriv[2]) / low for i in range(2)]
    return rho * sum(a * a for a in r) / 2, rho * sum(a * b for a, b in zip(r, dr, strict=True)) / h


def owner_on_chord(dz, tilt, rate_err, h, j, b):
    # g(t)=t*dz+rho*(1-n_z), |(rho*n_z)'/h| <= rate_err, 0<=t<=1.
    lo, hi = dz / h - rate_err, dz / h + rate_err
    demand(lo > 0 and dz + tilt < WIDTH, "prescribed chord outside monotone compressed domain")
    xl, xu = WIDTH - dz - tilt, WIDTH - dz
    mean_low, mean_high = WIDTH - dz / 2 - tilt, WIDTH - dz / 2
    weight_low, weight_high = WIDTH / 2 - dz / 6 - tilt / 2, WIDTH / 2 - dz / 6
    rate_low, rate_high = max(0, 1 - DAMP * hi), max(0, 1 - DAMP * lo)
    law_j = (h * AREA * K * mean_low * rate_low, h * AREA * K * mean_high * rate_high)
    law_b = (h * h * AREA * K * weight_low * rate_low, h * h * AREA * K * weight_high * rate_high)
    loss = (
        h * AREA * K * mean_low * lo * min(DAMP * lo, 1),
        h * AREA * K * mean_high * hi * min(DAMP * hi, 1),
    )
    energy = (AREA * K * xl * xl / 2, AREA * K * xu * xu / 2)
    row = dict(
        gap_m=display((dz, dz + tilt)),
        chord_rate_m_s=display((lo, hi)),
        delivered_j_ns=float(j),
        delivered_b_ns2=float(b),
        required_law_j_ns=display(law_j),
        required_law_b_ns2=display(law_b),
        law_j_error_lower_ns=float(deviation(j, law_j)),
        law_b_error_lower_ns2=float(deviation(b, law_b)),
        storage_j=display(energy),
        dissipation_j=display(loss),
    )
    return row, energy, loss, (j * lo, j * hi)


def local_law(points, vel, q, u, patch_z):
    # The public port's current speed is distinct from the path's affine derivative.
    gap = points[:, 2] - q - patch_z
    rate = vel[:, 2] - u
    force = [
        None if g < 0 else float(AREA * K) * max(float(WIDTH) - g, 0) * max(1 - float(DAMP) * v, 0)
        for g, v in zip(gap, rate, strict=True)
    ]
    return gap, rate, force


def extract(case, root, base, previous, reference=None):
    selected = read(root / f"{case}-selected.json.gz")
    final = read(root / f"{case}-final.json.gz")
    e = read(root / f"{case}-evidence.json.gz")
    h = Q(base["h_s"])
    m = Q(base["cloth"]["m_kg"])
    M = 4 * m
    rho = Q(base["cloth"]["half_thickness_m"])
    faces = np.array(base["cloth"]["faces"])
    js = list(map(Q, e["source"]["j"]))
    bs = list(map(Q, e["source"]["b"]))
    if reference is None:
        prior = next(z for z in previous["cases"] if z["case"] == case)
        pp = PLAN.parents[2] / previous["raw_directory"] / f"{case}.packet.json"
        demand(
            file_sha(pp) == e["source"]["packet_sha256"] == prior["packet_sha256"],
            "different pressure history",
        )
        packet = read(pp)
        j, b = independent_moments(packet["macros"], h)
        shares = [Q(o["share"]) for o in packet["owners"]]
        demand(
            js == [j * s for s in shares]
            and bs == [b * s for s in shares]
            and prior["source_bindings"][:3]
            == [e["scene"], packet["original_inputs"], e["state_and_inputs"]],
            "prescribed original input",
        )
    else:
        demand(
            js == reference["js"]
            and bs == reference["bs"]
            and e["source"]["reference_sha256"] == file_sha(root / "coupled-reference.json.gz"),
            "coupled force quadrature binding",
        )
    for trial in (selected, final):
        demand(
            trial["duration"] == float(h)
            and trial["result"]["report"]["accepted"]
            and len(trial["result"]["report"]["substeps"]) == 1,
            "mechanical replay",
        )
    diag = final["result"]["report"]["substeps"][0]["diagnostics"]
    term = diag["terminal_impulse"]
    cd = diag["participants"]["cloth"]
    demand(
        term["applications"] == 1
        and term["pre_impulse_state"] == selected["result"]["state"]
        and term["pre_impulse_diagnostics"]
        == selected["result"]["report"]["substeps"][0]["diagnostics"]
        and term["ledger"]["configuration_preserved"]
        and term["ledger"]["history_preserved"]
        and term["ledger"]["max_residual"] <= 1e-8,
        "terminal replay prebinding",
    )
    demand(
        final["result"]["substep_states"][0] == final["result"]["state"]
        and final["effective_inputs"] == selected["effective_inputs"]
        and final["integrated_loads"] == selected["integrated_loads"] == e["held"],
        "effective input or endpoint changed",
    )
    initial = e["initial"]["domains"]
    state = final["result"]["state"]["domains"]
    pre = selected["result"]["state"]["domains"]
    x0, v0 = arr(initial["cloth"]["position"]), arr(initial["cloth"]["velocity"])
    x1, v1 = arr(state["cloth"]["position"]), arr(state["cloth"]["velocity"])
    vpre = arr(pre["cloth"]["velocity"])
    q = float(arr(state["robot"]["q"])[0])
    u = float(arr(state["robot"]["v"])[0])
    up = float(arr(pre["robot"]["v"])[0])
    demand(
        np.array_equal(x0, base["cloth"]["nodes"])
        and not v0.any()
        and not arr(initial["robot"]["q"]).any()
        and not arr(initial["robot"]["v"]).any()
        and state["cloth"]["position"] == pre["cloth"]["position"]
        and state["robot"]["q"] == pre["robot"]["q"],
        "fixture or terminal configuration",
    )
    gravity = arr(e["model_gravity"])
    gz = Q(base["cloth"]["gravity_z"])
    G = np.tile([0.0, 0.0, float(m * gz * h)], (4, 1))
    demand(
        np.array_equal(gravity, [0, 0, float(gz)])
        and np.array_equal(arr(e["node_mass"]), np.full(4, float(m)))
        and np.array_equal(arr(e["gravity_body"]), G)
        and np.array_equal(arr(e["gravity_left_adjust"]), -G / 2)
        and np.array_equal(arr(e["gravity_right"]), G / 2),
        "gravity source or split changed",
    )
    orig = e["inputs"]["domains"]
    eff = final["effective_inputs"]["domains"]
    loads = final["integrated_loads"]
    L = arr(e["left_world"])
    R = arr(e["right_world"])
    demand(
        all(not arr(v).any() for d in orig.values() for v in d.values())
        and not arr(eff["robot"]["effort"]).any(),
        "original nongravity loads",
    )
    for d, fields in loads.items():
        for name, value in fields.items():
            demand(
                np.array_equal(arr(value), float(h) * (arr(eff[d][name]) - arr(orig[d][name]))),
                "unbound delivered field",
            )
    demand(
        np.array_equal(L[:, 2], [float(b / h) for b in bs])
        and np.array_equal(R[:, 2], [float(j - b / h) for j, b in zip(js, bs, strict=True)])
        and not L[:, :2].any()
        and not R[:, :2].any(),
        "force moments lost",
    )
    start, end = e["start"], e["end"]
    map_from_coordinates(start, x0, v0, faces, float(rho), L)
    map_from_coordinates(end, x1, vpre, faces, float(rho), R)
    ct, rt = term["requests"]
    nL = arr(start["node"])
    nR = arr(end["node"])
    demand(
        np.array_equal(arr(loads["cloth"]["force"]), nL - G / 2)
        and np.array_equal(arr(ct["impulses"]), nR + G / 2)
        and np.array_equal(arr(cd["terminal_requests"]), arr(ct["impulses"]))
        and term["ledger"]["requests"] == term["requests"],
        "internal and original body allocations conflated",
    )
    demand(
        np.array_equal(arr(rt["impulses"]), -R)
        and np.array_equal(arr(rt["local_points"]), arr(end["robot_local"]))
        and norm_inf(arr(loads["robot"]["force"])[0] + L.sum(axis=0)) < 1e-12,
        "platform delivery",
    )
    momentL = np.cross(arr(start["robot_points"]), -L).sum(axis=0)
    momentR = np.cross(arr(end["robot_points"]), -R).sum(axis=0)
    demand(norm_inf(arr(loads["robot"]["torque"])[0] - momentL) < 1e-12, "platform spatial load")
    pred = arr(cd["material_path"]["predicted_velocity"])
    demand(
        cd["material_path"]["input_fingerprint"] == e["effective_cloth"] != e["original_cloth"]
        and norm_inf(float(m) * (pred - v0) - (nL + G / 2)) < 1e-12
        and norm_inf(float(m) * (v1 - vpre) - (nR + G / 2)) < 1e-12
        and arr(cd["bending_multipliers"])[0] != 0,
        "predictor/terminal body and contact impulse",
    )
    com_z = sum(qr(z) for z in x1[:, 2]) / 4
    mean = com_z - qr(q)
    predicted = gz * h * h / 2 + sum(bs) * (1 / (4 * m) + 1 / M)
    mean_v = np.mean(v1[:, 2]) - u
    expected_v = float(gz * h + sum(js) * (1 / (4 * m) + 1 / M))
    demand(
        abs(mean - predicted) <= Q("1e-8")
        and abs(q + float(sum(bs) / M)) <= 1e-8
        and abs(u + float(sum(js) / M)) < 1e-12
        and abs(M * (u - up) + sum(j - b / h for j, b in zip(js, bs, strict=True))) < 1e-12
        and abs(mean_v - expected_v) < 1e-12,
        "COM or platform time moment",
    )
    measured = np.sum(np.cross(x1, float(m) * v1) - np.cross(x0, float(m) * v0), axis=0)
    # Gravity is actually delivered at two temporal action points for angular balance.
    grav_m = np.cross(x0, G / 2).sum(axis=0) + np.cross(x1, G / 2).sum(axis=0)
    angular = float(np.linalg.norm(measured - grav_m + momentL + momentR, ord=1))
    demand(angular <= base["bounds"]["joint_angular_nms"], "paired original angular balance")
    patch = arr(e["patch"]["local_vertices"])
    gstart, rate0, f0 = local_law(
        arr(start["points"]), arr(start["point_velocity"]), 0.0, 0.0, patch[0, 2]
    )
    gend, rate1, f1 = local_law(arr(e["post_points"]), arr(e["post_rates"]), q, u, patch[0, 2])
    demand(
        np.array_equal(arr(e["post_points"]), arr(end["points"]))
        and norm_inf(
            arr(e["post_rates"])
            - np.einsum("rijk,rik->rj", arr(end["jacobians"]), v1[arr(end["triangles"])])
        )
        < 1e-12,
        "actual final material-point motion",
    )
    demand(
        not gstart.any() and not rate0.any() and norm_inf(np.array(f0) - 1 / 6) < 1e-15,
        "preload at original state",
    )
    hard_positive = min(qr(z) - qr(q) for z in x1[:, 2]) > 0
    hard = []
    laws = []
    energies = []
    losses = []
    works = []
    chord = chord_certificate(x0, x1, 0, q, faces, patch, rho) if hard_positive else None
    for f, (face, sw, mf) in enumerate(zip(faces, e["sweeps"], end["manifolds"], strict=True)):
        tilt, rate_err = tilt_error(x0, x1, face, rho, h)
        dz = [qr(x1[k, 2]) - qr(q) for k in face]
        sat = [float(Q(v)) for v in mf["gap"]]
        info = dict(
            surface=("U", "D")[f],
            status=sw["status"],
            reason=sw["reason"],
            witness_t=sw["witness_time"],
            endpoint_sat_gap_m=sat,
            tilt_gap_bound_m=float(tilt),
            tilt_rate_bound_m_s=float(rate_err),
        )
        if min(dz) > 0:
            demand(
                sat[0] > 0 and sw["status"] in ("certified", "unresolved"), "observed separation"
            )
        elif min(dz) < 0:
            demand(sat[1] < 0 and sw["status"] == "violation", "missing finite penetration witness")
            wit = vertex_witnesses(x0, x1, q, face, rho, Q(sw["witness_time"]))
            demand(wit[1] < 0, "independent witness")
            info["witness_vertex_gap_m"] = wit
        hard.append(info)
        for k, dz_i in enumerate(dz):
            i = 3 * f + k
            row = dict(
                face=("U", "D")[f],
                corner=k,
                node=int(face[k]),
                port_gap_m=float(gend[i]),
                port_gap_rate_m_s=float(rate1[i]),
                source_start_n=float(arr(e["source"]["source_start"])[i]),
                source_end_n=float(arr(e["source"]["source_end"])[i]),
                required_at_inlet_n=f0[i],
                required_at_published_end_n=f1[i],
            )
            if hard_positive:
                measures, E, D, W = owner_on_chord(dz_i, tilt, rate_err, h, js[i], bs[i])
                row.update(measures)
                energies.append(E)
                losses.append(D)
                works.append(W)
            laws.append(row)
    phys = cd["physical_audit"]
    gravity_work = sum(
        (m * qr(gravity[k]) * (qr(x1[i, k]) - qr(x0[i, k])) for i in range(4) for k in range(3)),
        Q(0),
    )
    feff_work = float(np.sum(arr(eff["cloth"]["force"]) * (x1 - x0)))
    demand(
        abs(phys["original_input_work"] - feff_work) < 1e-15,
        "local work confused with external work",
    )
    budget = base["bounds"]["joint_energy_j"]
    balance = None
    storage0 = 6 * AREA * K * WIDTH**2 / 2
    if hard_positive:
        E = total_bounds(energies)
        D = total_bounds(losses)
        W = total_bounds(works)
        kinetic = m * sum(qr(v) ** 2 for v in v1.flat) / 2 + M * qr(u) ** 2 / 2
        e0, e1 = phys["elastic_before"], phys["elastic_after"]
        defect = (
            kinetic + qr(e1[0]) - qr(e0[1]) + E[0] - storage0 + D[0] - gravity_work,
            kinetic + qr(e1[1]) - qr(e0[0]) + E[1] - storage0 + D[1] - gravity_work,
        )
        balance = dict(
            final_layer_storage_j=display(E),
            layer_dissipation_j=display(D),
            prescribed_chord_work_j=display(W),
            remaining_joint_defect_j=display(defect),
            positive_energy_upper_within_budget=defect[1] <= qr(budget),
        )
    result = dict(
        case=case,
        prescribed=reference is None,
        center_pass=True,
        mean_gap_m=float(mean),
        expected_mean_gap_m=float(predicted),
        mean_error_m=float(abs(mean - predicted)),
        platform_q_m=q,
        platform_v_m_s=u,
        j_ns=float(sum(js)),
        b_ns2=float(sum(bs)),
        actual_gap_rate_reference="terminal public offset point velocity minus platform velocity",
        independent_hard_chord=hard_positive,
        hard=hard,
        chord_certificate=chord,
        owners=laws,
        max_law_j_error_lower_ns=max((r.get("law_j_error_lower_ns", 0) for r in laws), default=0)
        if hard_positive
        else None,
        own_gravity_impulse_z_ns=float(m * gz * h),
        own_gravity_time_moment_z_ns2=float(m * gz * h * h / 2),
        original_non_gravity_work_j=0.0,
        original_gravity_work_j=float(gravity_work),
        local_effective_work_j=feff_work,
        initial_layer_storage_j=float(storage0),
        affine_path_energy=balance,
        paired_angular_l1_nms=angular,
        physical_successor=False,
        measured_guide_reaction=False,
    )
    if reference is None:
        result["change_in_mean_vs_saved_m"] = float(mean) - prior["mean_middle_gap_m"]
        demand(
            abs(result["change_in_mean_vs_saved_m"] + float(gz * h * h / 2)) < 1e-12,
            "matched-history gravity discrimination",
        )
    else:
        result["max_endpoint_gap_vs_proposal_m"] = norm_inf(gend - reference["end_gaps"])
        result["max_endpoint_rate_vs_proposal_m_s"] = norm_inf(rate1 - reference["end_rates"])
    return result


def unit_rate(raw, draw):
    length = np.linalg.norm(raw)
    n = raw / length
    return n, (draw - n * np.dot(n, draw)) / length


def normal_rate(x, v, i):
    tri = x[i]
    vel = v[i]
    a, b = tri[1] - tri[0], tri[2] - tri[0]
    da, db = vel[1] - vel[0], vel[2] - vel[0]
    return unit_rate(np.cross(a, b), np.cross(da, b) + np.cross(a, db))


def stage_law(record, base):
    y = arr(record["state"])
    dy = arr(record["dy"])
    x = y[:12].reshape(4, 3)
    v = y[12:24].reshape(4, 3)
    q, u = y[-2:]
    faces = np.array(base["cloth"]["faces"])
    m = float(Q(base["cloth"]["m_kg"]))
    rho = float(Q(base["cloth"]["half_thickness_m"]))
    frames = [normal_rate(x, v, i) for i in faces]
    gap = np.concatenate(
        [x[i, 2] - q + rho * (1 - n[2]) for i, (n, dn) in zip(faces, frames, strict=True)]
    )
    rate = np.concatenate(
        [v[i, 2] - u - rho * dn[2] for i, (n, dn) in zip(faces, frames, strict=True)]
    )
    demand(
        min(gap) >= 0
        and norm_inf(gap - arr(record["gap"])) < 1e-15
        and norm_inf(rate - arr(record["rate"])) < 1e-15,
        "reference stage outside layer",
    )
    force = (
        float(AREA * K)
        * np.maximum(0.0, float(WIDTH) - gap)
        * np.maximum(0.0, 1 - float(DAMP) * rate)
    )
    demand(norm_inf(force - arr(record["force"])) < 1e-12, "pressure law at six moving points")
    # Differentiate atan2(sin,cos) directionally and differentiate each oriented offset point.
    nl, nr = frames[0][0], frames[1][0]
    edge = x[1] - x[0]
    axis = edge / np.linalg.norm(edge)
    cosine = np.dot(nl, nr)
    sine = np.dot(axis, np.cross(nl, nr))
    theta = atan2(sine, cosine)
    rest = float(Q(base["cloth"]["rest_angle_rad"]))
    elastic = np.zeros_like(x)
    contact = np.zeros_like(x)
    for a, b in base["cloth"]["distance_edges"]:
        e = x[b] - x[a]
        length = np.linalg.norm(e)
        f = (length - 1) * e / length
        elastic[a] += f
        elastic[b] -= f
    for k in range(12):
        direction = np.zeros_like(x)
        direction.flat[k] = 1
        _, da = unit_rate(edge, direction[1] - direction[0])
        dnl = normal_rate(x, direction, faces[0])[1]
        dnr = normal_rate(x, direction, faces[1])[1]
        ds = np.dot(da, np.cross(nl, nr)) + np.dot(axis, np.cross(dnl, nr) + np.cross(nl, dnr))
        dc = np.dot(dnl, nr) + np.dot(nl, dnr)
        dtheta = (cosine * ds - sine * dc) / (cosine * cosine + sine * sine)
        elastic.flat[k] -= (theta - rest) * dtheta
        dpoint = np.r_[direction[faces[0], 2] - rho * dnl[2], direction[faces[1], 2] - rho * dnr[2]]
        contact.flat[k] = np.dot(force, dpoint)
    demand(
        norm_inf(elastic - arr(record["material_force"])) < 1e-12
        and norm_inf(contact - arr(record["contact_force"])) < 1e-12,
        "independent force-map derivative",
    )
    expected = np.r_[
        v.ravel(),
        ((elastic + contact) / m + [0.0, 0.0, float(Q(base["cloth"]["gravity_z"]))]).ravel(),
        u,
        -force.sum() / (4 * m),
    ]
    demand(norm_inf(dy - expected) < 1e-12, "coupled motion equation")
    return arr(record["force"]), gap, rate


def audit_source(root, base):
    data = read(root / "coupled-reference.json.gz")
    h = Q(base["h_s"])
    dt = h / 20
    stages = data["stages"]
    demand(
        len(stages) == 80 and data["owner_force_evaluations"] == 486 <= 512,
        "reference force charge",
    )
    x = np.array(base["cloth"]["nodes"], float)
    y = np.r_[x.ravel(), np.zeros(14)]
    js = [Q(0)] * 6
    bs = [Q(0)] * 6
    sample_min = inf
    for n in range(20):
        deriv = []
        for s, w in zip((Q(0), Q(1, 2), Q(1, 2), Q(1)), (1, 2, 2, 1), strict=True):
            rec = stages[4 * n + len(deriv)]
            at = (n + s) * dt
            expected = y if not deriv else y + float(dt * s) * deriv[-1]
            demand(
                Q(rec["time"]) == at
                and rec["weight"] == w
                and np.array_equal(arr(rec["state"]), expected),
                "RK stage not from previous state",
            )
            f, g, _ = stage_law(rec, base)
            sample_min = min(sample_min, float(min(g)))
            for i, v in enumerate(f):
                j = dt * w * qr(v) / 6
                js[i] += j
                bs[i] += (h - at) * j
            deriv.append(arr(rec["dy"]))
        a, b, c, d = deriv
        y = y + float(dt) / 6 * (a + 2 * b + 2 * c + d)
    demand(
        np.array_equal(arr(data["endpoint"]["state"]), y)
        and list(map(Q, data["j"])) == js
        and list(map(Q, data["b"])) == bs,
        "reference endpoint or quadrature",
    )
    force, gap, rate = stage_law(data["endpoint"], base)
    return dict(
        js=js,
        bs=bs,
        end_gaps=gap,
        end_rates=rate,
        end_force=force,
        sample_min=sample_min,
        evaluations=data["owner_force_evaluations"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("audit-controls", "audit-coupled"), required=True)
    args = parser.parse_args()
    root = args.root
    spec = read(PLAN)
    fixture = PLAN.with_name(spec["original_fixture"])
    previous = PLAN.with_name(spec["previous_result"])
    demand(
        file_sha(PLAN, True) == PLAN_SHA
        and file_sha(previous, True) == spec["previous_result_sha256_lf"]
        and file_sha(fixture, True) == spec["fixture_sha256_lf"],
        "source protocol binding",
    )
    base = read(fixture)
    prior = read(previous)
    reference = None
    if args.phase == "audit-controls":
        rows = [extract(case, root, base, prior) for case in spec["cases"]]
        threshold = read(PLAN.with_name("p65-contact-layer-intrastep-protocol.json"))["gates"][
            "law_impulse_vs_delivered_ns"
        ]
        transfer = all(r["center_pass"] for r in rows) and any(
            (not r["independent_hard_chord"]) or r["max_law_j_error_lower_ns"] > threshold
            for r in rows
        )
        result = dict(
            scope="gravity-timed controlled transport", rows=rows, transfer_to_coupled=transfer
        )
        name = "audit-controls.json"
    else:
        control = read(root / "audit-controls.json")
        demand(control["transfer_to_coupled"], "no transition evidence")
        reference = audit_source(root, base)
        coupled = extract("coupled", root, base, prior, reference)
        e = read(root / "coupled-evidence.json.gz")
        demand(
            norm_inf(arr(e["source"]["source_end"]) - reference["end_force"]) < 1e-12
            and norm_inf(arr(e["source"]["source_start"]) - np.full(6, 1 / 6)) < 1e-12,
            "coupled initial or endpoint load",
        )
        rows = control["rows"] + [coupled]
        result = dict(
            schema="p65-n2-gravity-six-owner-result-v1",
            raw_directory=root.relative_to(PLAN.parents[2]).as_posix(),
            protocol_sha256_lf=PLAN_SHA,
            controls_sha256=file_sha(root / "audit-controls.json"),
            controls=control["rows"],
            coupled=coupled,
            coupled_reference=dict(
                force_evaluations=reference["evaluations"],
                stage_minimum_gap_m=reference["sample_min"],
                final_gaps_m=reference["end_gaps"].tolist(),
                final_rates_m_s=reference["end_rates"].tolist(),
                final_forces_n=reference["end_force"].tolist(),
                force_moments="six simultaneous nonlinear stage quadratures, no prescribed scalar curve",
            ),
            raw_sha256={
                f.name: file_sha(f) for f in root.iterdir() if f.is_file() and f.suffix == ".gz"
            },
            joint_accepted=False,
            successors_committed=0,
        )
        name = "audited-result.json"
    with (root / name).open("x", encoding="utf8", newline="\n") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    for row in rows:
        print(
            row["case"],
            "mean",
            row["mean_gap_m"],
            "chord safe",
            row["independent_hard_chord"],
            "law impulse error lower",
            row["max_law_j_error_lower_ns"],
        )


if __name__ == "__main__":
    main()
