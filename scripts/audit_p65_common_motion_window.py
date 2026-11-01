"""Check the assembled SC1 window from the saved curves and accepted public checkpoints."""

import argparse
from fractions import Fraction as Q
from hashlib import sha256
from math import inf, nextafter
from pathlib import Path

import numpy as np
from audit_p65_common_motion import bernstein_moments, file_sha, need
from check_p65_common_motion import (
    ZERO,
    add,
    bounds,
    cross_poly,
    derive,
    integral,
    one_integral,
    power,
    scale,
    sub,
    times,
)
from p65_common_motion import DEGREE, GRID, unpack
from probe_p65_common_motion import ClothPlatform, PathProgram, joint_state, saved, verified_config
from probe_p65_joint_readback import fixture, save
from run_p65_common_motion_window import PLAN, PLAN_SHA, metadata, tag

from phymeo.core.scene import fingerprint
from phymeo.core.validation import validate_dt


def upper(number):
    """Allow one upward float ULP when reusing a previously published upper bound."""
    return Q(nextafter(float(number), inf))


def hi(value):
    f = float(value)
    return nextafter(f, inf) if Q(f) < value else f


def lo(value):
    f = float(value)
    return nextafter(f, -inf) if Q(f) > value else f


def interval(pair):
    return tuple(Q(float(v)) for v in pair)


def show(pair):
    return [lo(pair[0]), hi(pair[1])]


def floats(a):
    return np.asarray(a, dtype=float).tolist()


def moment_pair(words, h, weights):
    coeff = np.asarray(words, dtype=object)
    return np.stack(
        [
            np.tensordot(np.array([h * w[0] / GRID for w in weights], object), coeff, axes=1),
            np.tensordot(np.array([h * h * w[1] / GRID for w in weights], object), coeff, axes=1),
        ],
        axis=-1,
    )


def shifted(pair, tail):
    out = pair.copy()
    out[..., 1] += tail * out[..., 0]
    return out


def chain(reports, h, T):
    need(len(reports) == 8 and 8 * h == T, "missing original step")
    seen = set()
    for i, r in enumerate(reports):
        need(
            r["accepted"]
            and r["certificate"]["accepted"]
            and all(r["certificate"]["checks"].values()),
            "uncertified step",
        )
        need(
            Q(r["start_s"]) == i * h
            and Q(r["end_s"]) == (i + 1) * h
            and r["tag"] == (["first", "next"][i] if i < 2 else tag(i)),
            "missing or aliased step",
        )
        need(r["output_sha256"] not in seen, "aliased outgoing checkpoint")
        seen.add(r["output_sha256"])
        if i:
            need(
                r["inlet_sha256"] == reports[i - 1]["output_sha256"]
                and r["true_inlet_velocity"] == reports[i - 1]["public_velocity"],
                "broken checkpoint or velocity seam",
            )


def ranges(curve, q0, v0, h, model):
    """Bound total support, layer compression and nonzero bending over the curved path."""
    ext = list(map(Q, map(float, model.external)))
    mass = list(map(Q, map(float, model.mass)))
    contact = np.sum(curve["rows"].astype(object), axis=1)
    net = curve["material"].astype(object) + contact
    positions = []
    velocities = []
    for i in range(13):
        f = add(power(net[:, i]), (ext[i],))
        v = integral(scale(f, h / mass[i]), v0[i])
        positions.append(integral(scale(v, h), q0[i]))
        velocities.append(v)
    world = power(np.sum(curve["pressure"].astype(object), axis=1))
    represented = bounds(world)
    faces = model.faces
    rho = Q(float(model.rho))
    width = Q(1, 500)
    damping = Q(1000)
    ak = Q(500, 6)
    xx = [positions[i : i + 3] for i in range(0, 12, 3)]
    normals = []
    for f in faces:
        edges = [[sub(a, b) for a, b in zip(xx[node], xx[f[0]], strict=True)] for node in f[1:]]
        normal = cross_poly(*edges)
        normals.append(([bounds(p) for p in normal], [bounds(derive(p)) for p in normal]))
    gaps = [sub(positions[3 * k + 2], positions[-1]) for k in faces.ravel()]
    rates = [scale(derive(b), 1 / h) for b in gaps]
    nominal = ZERO
    for b, u in zip(gaps, rates, strict=True):
        nominal = add(nominal, scale(times(sub((width,), b), sub((1,), scale(u, damping))), ak))
    target = bounds(nominal)
    min_law = None
    max_law = None
    max_gap = Q(0)
    max_angle = Q(0)
    for n in range(len(represented)):
        angle_bound = Q(0)
        correction = Q(0)
        for face, (cn, dn) in enumerate(normals):
            normal = [row[n] for row in cn]
            gradient = [row[n] for row in dn]
            z = normal[2].low
            need(z > 0, "unsupported winding in window")
            slopes = [a.magnitude / z for a in normal[:2]]
            deriv = [
                (a.magnitude + s * gradient[2].magnitude) / z
                for a, s in zip(gradient[:2], slopes, strict=True)
            ]
            angle_bound += sum(slopes)
            tilt = rho * sum(s * s for s in slopes) / 2
            rate_err = rho * sum(s * d for s, d in zip(slopes, deriv, strict=True)) / h
            for corner in range(3):
                j = 3 * face + corner
                b = bounds(gaps[j])[n]
                u = bounds(rates[j])[n]
                max_gap = max(max_gap, b.high + tilt)
                correction += ak * (
                    tilt * (1 + damping * u.magnitude)
                    + (width + b.magnitude + tilt) * damping * rate_err
                )
        lower, upper_law = target[n].low - correction, target[n].high + correction
        min_law = lower if min_law is None else min(min_law, lower)
        max_law = upper_law if max_law is None else max(max_law, upper_law)
        max_angle = max(max_angle, angle_bound)
    C_margin = abs(model.rest) - max_angle
    need(
        min_law > 0 and max_gap < width and C_margin > 0,
        "lost positive support, active layer, or hinge strain",
    )
    contact_work = h * sum(
        one_integral(times(power(contact[:, i]), velocities[i])) for i in range(13)
    )
    return dict(
        represented_force=(min(r.low for r in represented), max(r.high for r in represented)),
        ideal_force=(min_law, max_law),
        end_force=sum(world),
        max_gap_upper=max_gap,
        compression_lower=width - max_gap,
        bending_strain_lower=C_margin,
        contact_work=contact_work,
    )


def noncancelling_energy(highs, limit):
    value = sum((max(Q(0), v) for v in highs), Q(0))
    need(value <= limit, "noncancelling window energy limit")
    return value


def require_external(claim, expected):
    need(claim == expected, "internal work cannot be external work")


def require_platform(total, change_v, change_q, M, velocity0, T):
    need(
        abs(total[0] + M * change_v) <= Q("1e-10")
        and abs(total[1] / M + change_q - T * velocity0) <= Q("1e-8"),
        "wrong global platform impulse or time origin",
    )


def failure(label, function):
    try:
        function()
    except ValueError as exc:
        return dict(control=label, rejected=True, reason=str(exc))
    raise ValueError("unrejected negative: " + label)


def audit(root):
    spec, prefix, h, T = metadata()
    config, _ = verified_config()
    parts = fixture(config)
    model = ClothPlatform(parts)
    try:
        return assemble(root, spec, prefix, h, T, parts, model)
    finally:
        parts[3].close()


def assemble(root, spec, prefix, h, T, parts, model):
    old = PLAN.parents[2] / prefix["raw_directory"]
    names = ["first", "next"] + [tag(i) for i in spec["step_indices"]]
    reports = list(prefix["steps"]) + [
        saved(root / f"{tag(i)}-audit.json") for i in spec["step_indices"]
    ]
    chain(reports, h, T)
    weights = [bernstein_moments(k) for k in range(DEGREE + 1)]
    ext = np.array([Q(float(x)) for x in model.external], object)
    mass = np.array([Q(float(x)) for x in model.mass], object)
    small = min(mass)
    world = np.full((6, 2), Q(0), object)
    raw_b = np.full(6, Q(0), object)
    mapped = np.full((6, 13, 2), Q(0), object)
    material = np.full((13, 2), Q(0), object)
    external = np.full((13, 2), Q(0), object)
    error_j = np.full(6, Q(0), object)
    error_b = np.full(6, Q(0), object)
    node_x = node_v = Q(0)
    angular = old_angular = Q(0)
    linear_abs = Q(0)
    works = []
    diss = []
    signed = []
    stats = []
    rows = []
    prior = None
    for i, (label, r) in enumerate(zip(names, reports, strict=True)):
        folder = old if i < 2 else root
        curve_path = folder / f"{label}-attempt.json.gz"
        # This read supplies force coefficients for the new window moments, not another per-step verification.
        need(file_sha(curve_path) == r["attempt_sha256"], "curve not bound to the accepted frame")
        attempt = saved(curve_path)
        curve = attempt["curve"]
        cert = r["certificate"]
        need(
            tuple(curve["owners"]) == model.owners
            and curve["grid"] == GRID
            and np.array_equal(curve["external"], model.external)
            and np.array_equal(curve["mass"], model.mass),
            "changed world addresses, external loads or masses",
        )
        incoming = attempt["inlet"]["history"]["motion"]
        need(
            sha256(incoming.tobytes()).hexdigest() == r["inlet_sha256"],
            "actual entry bytes do not match accepted receipt",
        )
        q0, v0, t0 = unpack(incoming, model.signature)
        need(
            t0 == i * h
            and fingerprint(
                (
                    PathProgram(model).problem_signature(),
                    joint_state(q0, v0, incoming),
                    parts[2],
                    validate_dt(float(h), np.float64),
                )
            )
            == curve["binding"],
            "original input/state binding",
        )
        need(prior is None or prior == incoming.tobytes(), "full checkpoint bytes changed")
        if i == 0:
            initial_q, initial_v = np.array(q0, object), np.array(v0, object)
            need(
                not any(v0)
                and np.array_equal(
                    np.asarray(q0, float),
                    np.r_[parts[1].domains["cloth"].position.ravel(), parts[1].domains["robot"].q],
                ),
                "incorrect original rest state",
            )
        output = folder / f"{label}-output.bin"
        prior = output.read_bytes()
        need(file_sha(output) == r["output_sha256"], "output changed")
        final_q, final_v, t1 = unpack(np.frombuffer(prior, dtype="<i4"), model.signature)
        need(
            t1 == (i + 1) * h
            and r["solver_owner_evaluations"] == 486
            and r["proof_owner_enclosures"] == 384
            and r["checkpoint_wire_bytes"] == 1024,
            "step resource or timeline",
        )
        tail = T - t1
        P = moment_pair(curve["pressure"], h, weights)
        world += shifted(P, tail)
        raw_b += P[:, 1]
        mapped += shifted(moment_pair(curve["rows"], h, weights), tail)
        material += shifted(moment_pair(curve["material"], h, weights), tail)
        external += shifted(np.stack((ext * h, ext * h * h / 2), axis=-1), tail)
        ej = np.array(list(map(upper, cert["j_defect_upper_ns"])), object)
        eb = np.array(list(map(upper, cert["b_defect_upper_per_mass_m"])), object)
        error_j += ej
        error_b += eb + tail * ej / small
        node_x += upper(cert["node_force_position_defect_upper_m"]) + tail * upper(
            cert["node_force_velocity_defect_upper_m_s"]
        )
        node_v += upper(cert["node_force_velocity_defect_upper_m_s"])
        angular += upper(cert["paired_angular_upper_nms"])
        old_angular += upper(cert["initial_convention_angular_upper_nms"])
        linear_abs += sum(upper(abs(v)) for v in cert["net_momentum_residual_ns"])
        works.append(ext @ (np.array(final_q, object) - np.array(q0, object)))
        diss.append(interval(cert["dissipation_j"]))
        signed.append(interval(cert["signed_energy_defect_j"]))
        if i:
            prev = reports[i - 1]["certificate"]
            need(
                prev["material_energy_j"][1] == cert["material_energy_j"][0]
                and prev["layer_energy_j"][1] == cert["layer_energy_j"][0],
                "discontinuous stored energy",
            )
        span = ranges(curve, q0, v0, h, model)
        stats.append(span)
        rows.append(
            dict(
                index=i,
                origin="reused accepted prefix"
                if i < 2
                else "fresh public HOST frame and separate disk audit",
                start_s=float(t0),
                end_s=float(t1),
                inlet_sha256=r["inlet_sha256"],
                output_sha256=r["output_sha256"],
                attempt_sha256=r["attempt_sha256"],
                result_sha256=r["result_sha256"],
                checkpoint_content_bytes=r["checkpoint_content_bytes"],
                owner_force_evaluations=r["solver_owner_evaluations"],
                owner_proof_enclosures=r["proof_owner_enclosures"],
                total_contact_n=show(span["represented_force"]),
                required_total_contact_n=show(span["ideal_force"]),
                end_force_n=float(span["end_force"]),
                end_gaps_m=cert["endpoint_gaps_m"],
                active_bend_strain_lower_rad=lo(span["bending_strain_lower"]),
                min_compression_lower_m=lo(span["compression_lower"]),
                signed_energy_defect_j=cert["signed_energy_defect_j"],
                paired_angular_upper_nms=hi(upper(cert["paired_angular_upper_nms"])),
                max_law_j_error_ns=hi(max(ej)),
            )
        )
    q1 = np.array(final_q, object)
    v1 = np.array(final_v, object)
    all_forces = material + mapped.sum(axis=0) + external
    need(
        np.array_equal(all_forces[:, 0], mass * (v1 - initial_v))
        and np.array_equal(all_forces[:, 1], mass * (q1 - initial_q - T * initial_v)),
        "global all-coordinate force-to-motion identities",
    )
    need(
        np.array_equal(external[:, 0], ext * T) and np.array_equal(external[:, 1], ext * T * T / 2),
        "wrong original gravity history",
    )
    platform = world.sum(axis=0)
    require_platform(
        platform, v1[-1] - initial_v[-1], q1[-1] - initial_q[-1], mass[-1], initial_v[-1], T
    )
    need(all(s[0] > 0 and 0 < s[1] < T * s[0] for s in world), "nonpositive owner window delivery")
    need(
        max(error_j) <= Q("1e-10")
        and max(error_b) <= Q("1e-10")
        and node_v <= Q("2e-6")
        and node_x <= Q("1e-8"),
        "window force or time-moment residual limit",
    )
    need(
        max(angular, old_angular) <= Q("1e-8") and linear_abs <= Q("1e-10"),
        "window angular or momentum limit",
    )
    initial = reports[0]["certificate"]
    final = reports[-1]["certificate"]
    budget = Q("1e-8")
    positive = noncancelling_energy([r[1] for r in signed], budget)
    DeltaK = sum(m * (b * b - a * a) / 2 for m, a, b in zip(mass, initial_v, v1, strict=True))
    W0 = ext @ (q1 - initial_q)
    require_external(sum(works), W0)
    mat0, mat1 = interval(initial["material_energy_j"][0]), interval(final["material_energy_j"][1])
    layer0, layer1 = interval(initial["layer_energy_j"][0]), interval(final["layer_energy_j"][1])
    D = tuple(sum(r[k] for r in diss) for k in (0, 1))
    step_sum = tuple(sum(r[k] for r in signed) for k in (0, 1))
    direct = tuple(
        DeltaK + mat1[k] - mat0[1 - k] + layer1[k] - layer0[1 - k] + D[k] - W0 for k in (0, 1)
    )
    need(
        D[0] > 0
        and direct[1] <= budget
        and max(step_sum[0], direct[0]) <= min(step_sum[1], direct[1]),
        "global energy enclosure",
    )
    contact_work = sum(s["contact_work"] for s in stats)
    controls = [
        failure("missing interval", lambda: chain(reports[:-1], h, T)),
        failure(
            "aliased interior interval",
            lambda: chain(reports[:3] + [reports[2]] + reports[4:], h, T),
        ),
        failure(
            "drop backward time-origin shifts",
            lambda: require_platform(
                [platform[0], sum(raw_b)],
                v1[-1] - initial_v[-1],
                q1[-1] - initial_q[-1],
                mass[-1],
                initial_v[-1],
                T,
            ),
        ),
        failure(
            "signed energy cancellation in synthetic ledger",
            lambda: noncancelling_energy((2 * budget, -2 * budget), budget),
        ),
        failure(
            "credit actual internal contact work as external",
            lambda: require_external(W0 + contact_work, W0),
        ),
    ]
    force_range = (
        min(s["represented_force"][0] for s in stats),
        max(s["represented_force"][1] for s in stats),
    )
    law_range = (min(s["ideal_force"][0] for s in stats), max(s["ideal_force"][1] for s in stats))
    balancing = -Q(float(model.cloth.gravity[2])) / (1 / sum(mass[0:12:3]) + 1 / mass[-1])
    need(balancing > 0 and force_range[0] > 0 and law_range[0] > 0, "sustained compressive support")
    return dict(
        schema="p65-common-motion-sc1-window-results-v1",
        accepted=True,
        raw_directory=root.relative_to(PLAN.parents[2]).as_posix(),
        protocol_sha256_lf=PLAN_SHA,
        prefix_result_sha256_lf=spec["seed_result_sha256_lf"],
        duration_s=float(T),
        h_s=float(h),
        reused_steps=2,
        new_steps=6,
        checkpoint_pairs=7,
        initial_sha256=reports[0]["inlet_sha256"],
        final_sha256=reports[-1]["output_sha256"],
        steps=rows,
        window_negative_controls=controls,
        support=dict(
            hard_gap_minimum_m=0.0,
            only_at_original_inlet=all(
                s["minimum_factor_m"] > 0 for r in reports for s in r["certificate"]["hard_witness"]
            ),
            total_represented_force_n=show(force_range),
            true_layer_law_total_force_n=show(law_range),
            terminal_force_n=float(stats[-1]["end_force"]),
            equivalent_balancing_load_n=float(balancing),
            represented_load_fraction_lower=lo(force_range[0] / balancing),
            minimum_active_bend_strain_rad=lo(min(s["bending_strain_lower"] for s in stats)),
            minimum_layer_compression_m=lo(min(s["compression_lower"] for s in stats)),
            final_hard_gaps_m=final["endpoint_gaps_m"],
        ),
        loads=dict(
            owners=list(model.owners),
            owner_impulse_ns=floats(world[:, 0]),
            owner_backward_moment_ns2=floats(world[:, 1]),
            total_normal_impulse_ns=float(platform[0]),
            total_normal_backward_moment_ns2=float(platform[1]),
            final_platform_q_m=float(q1[-1]),
            final_platform_v_m_s=float(v1[-1]),
            coordinate_order="a.xyz,b.xyz,c.xyz,d.xyz,platform.q",
            coordinate_columns=["impulse_N_s", "backward_moment_N_s2"],
            original_external=floats(external),
            material=floats(material),
            per_owner_mapped=floats(mapped),
            all_coordinate_identities_exact=True,
            maximum_law_impulse_defect_upper_ns=hi(max(error_j)),
            maximum_law_backward_defect_upper_per_mass_m=hi(max(error_b)),
            net_coordinate_position_defect_upper_m=hi(node_x),
            net_coordinate_velocity_defect_upper_m_s=hi(node_v),
        ),
        energy=dict(
            kinetic_change_j=float(DeltaK),
            material_initial_j=show(mat0),
            material_final_j=show(mat1),
            layer_initial_j=show(layer0),
            layer_final_j=show(layer1),
            dissipation_j=show(D),
            original_gravity_work_j=float(W0),
            original_nongravity_work_j=0.0,
            internal_mapped_contact_work_diagnostic_j=float(contact_work),
            direct_signed_defect_j=show(direct),
            signed_step_defect_sum_j=show(step_sum),
            noncancelling_positive_defect_upper_j=hi(positive),
            budget_j=float(budget),
        ),
        angular=dict(
            noncancelling_paired_upper_nms=hi(angular),
            noncancelling_initial_convention_upper_nms=hi(old_angular),
            budget_nms=1e-8,
            noncancelling_linear_upper_ns=hi(linear_abs),
            independent_guide_wrench_observed=False,
        ),
        resources=dict(
            owner_evaluations_each=486,
            owner_evaluations_total=8 * 486,
            per_step_limit=512,
            proof_owner_enclosures_total=8 * 384,
            geometric_leaves_each=64,
            checkpoint_bytes=1024,
            max_checkpoint_content_bytes=max(r["checkpoint_content_bytes"] for r in reports),
        ),
        limit="One HOST mass, material, mesh, timestep and SC1 compressive window. No SC2, native, general robot or guide-wrench result.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    result = audit(root)
    save(root / "window-audit.json", result)
    print(
        "Eight steps, seven disk seams, final endpoint",
        result["duration_s"],
        result["final_sha256"],
    )
    print(
        "Sustained support",
        result["support"]["total_represented_force_n"],
        "positive energy upper",
        result["energy"]["noncancelling_positive_defect_upper_j"],
        "angular upper",
        result["angular"]["noncancelling_initial_convention_upper_nms"],
    )


if __name__ == "__main__":
    main()
