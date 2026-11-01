"""Transfer and regression controls for the SC2 common-motion peak experiment."""

import argparse
import json
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
from check_p65_common_motion import add, bounds, integral, one_integral, power, scale, sub, times
from check_p65_common_motion_transfer import PLAN as SPRING_PLAN
from check_p65_common_motion_transfer import PLAN_SHA, SpringPair, error, trig
from p65_common_motion import endpoint, need
from p65_peak_tangent import Jet, peak_fit
from probe_p65_joint_readback import save
from verify_p65_contact_layer_screen import sha

BASE = Path(__file__).resolve().parents[1] / "docs/benchmarks"
PLAN = BASE / "p65-sc2-peak-levels-protocol.json"


def red(root):
    rows = []
    for spec, key in (("current-tangent", "candidate"), ("slope", "certificate")):
        rec = json.loads((BASE / f"p65-sc2-peak-{spec}-results.json").read_bytes())
        data = rec[key]
        limits = data.get(
            "local_absolute_J_error_lower_ns", data.get("local_absolute_impulse_defect_lower_ns")
        )
        need(
            not data["gates"]["owner_impulse_ns"] and rec["readback"]["same_inlet"],
            "baseline binding",
        )
        rows.append(
            dict(
                method=spec,
                minimum=min(limits),
                maximum=max(limits),
                denied=sum(x > 1e-10 for x in limits),
            )
        )
    save(root / "original-falsifier.json", rows)
    need(
        all(x["maximum"] <= 1e-10 for x in rows),
        "original noncancelling owner impulse counterexample",
    )


def spring_jet(model, q, v):
    frame = model.forces(q, v)
    direction = np.array([1.0, -1.0])
    pair = np.outer(direction, direction)
    return Jet(
        frame,
        -model.k * pair,
        np.array([-model.kc * direction]),
        np.zeros((1, 2)),
        np.array([-model.kc * pair]),
        np.zeros((1, 2, 2)),
        np.zeros((1, 2, 2)),
        np.array([direction @ q]),
    )


def transfer(root):
    need(sha(SPRING_PLAN, normalize=True) == PLAN_SHA, "spring control definition")
    case = json.loads(SPRING_PLAN.read_bytes())
    m = SpringPair(case)
    q = list(map(Q, case["initial_positions_m"]))
    v = list(map(Q, case["initial_velocities_m_s"]))
    h = Q(case["h_s"])

    def run(budget, axes):
        ledger = []
        claim, terminal = peak_fit(
            m,
            q,
            v,
            h,
            PLAN_SHA,
            budget,
            ledger,
            multilevel=True,
            provider=spring_jet,
            paired_axes=axes,
        )
        x, u = endpoint(claim, q, v)
        terminal(np.array(x, float), np.array(u, float))
        return claim, ledger, x, u

    claim, ledger, x, u = run(37, ((0, 1), (1, -1)))
    need(
        len(ledger) == 37
        and all(r["returned"] for r in ledger)
        and claim["force_calls"] == 36
        and [len(claim[g]) for g in ("coarse", "bridge", "iterations")] == [2, 1, 1]
        and claim["pressure"].shape == (21, 1),
        "multilevel schedule",
    )
    need(
        np.array_equal(claim["rows"][:, 0, 0], claim["pressure"][:, 0])
        and np.array_equal(claim["rows"][:, 0, 1], -claim["pressure"][:, 0]),
        "separately rounded shared load",
    )
    m1, m2 = map(Q, case["masses_kg"])
    km, kc = Q(case["material_k_n_m"]), Q(case["layer_k_n_m"])
    width, rest = Q(case["layer_width_m"]), Q(case["material_rest_m"])
    g0, w = q[0] - q[1], v[0] - v[1]
    eq = (km * rest + kc * width) / (km + kc)
    w2 = (km + kc) * (1 / m1 + 1 / m2)
    C, S = trig(w2 * h * h, 0), trig(w2 * h * h, 1)
    gap = eq + (g0 - eq) * C + w * h * S
    speed = -(g0 - eq) * w2 * h * S + w * C
    pol = power(claim["pressure"][:, 0])
    jc = kc * ((width - eq) * h - (g0 - eq) * h * S - w * (1 - C) / w2)
    bc = kc * ((width - eq) * h * h / 2 - (g0 - eq) * (1 - C) / w2 - w * h * (1 - S) / w2)
    je = error(h * one_integral(pol), jc)
    be = error(h * h * one_integral(times((1, -1), pol)), bc) / min(m1, m2)
    pos = error(x[0] - x[1], gap)
    vel = error(u[0] - u[1], speed)
    grav = Q(case["gravity_m_s2"])
    need(
        m1 * x[0] + m2 * x[1]
        == m1 * q[0] + m2 * q[1] + h * (m1 * v[0] + m2 * v[1]) + (m1 + m2) * grav * h * h / 2
        and m1 * u[0] + m2 * u[1] == m1 * v[0] + m2 * v[1] + (m1 + m2) * grav * h,
        "ballistic center",
    )
    paths = []
    for i, mi in enumerate((m1, m2)):
        f = add((mi * grav,), add(power(claim["material"][:, i]), power(claim["rows"][:, 0, i])))
        velocity = integral(scale(f, h / mi), v[i])
        paths.append(integral(scale(velocity, h), q[i]))
        need(sum(velocity) == u[i] and sum(paths[-1]) == x[i], "two independent integrals")
    separation = sub(*paths)
    pb = add(pol, scale(sub(separation, (width,)), kc))
    mb = add(power(claim["material"][:, 0]), scale(sub(separation, (rest,)), km))
    law = h * max(b.magnitude for b in bounds(pb))
    material = h * h * max(b.magnitude for b in bounds(mb)) / (2 * min(m1, m2))
    need(
        all(b.low > 0 and b.high < width for b in bounds(separation))
        and all(b.low > 0 for b in bounds(pol))
        and max(je, be, law) < Q(1, 10**10)
        and max(pos, material) < Q(1, 10**8)
        and vel < Q(2, 10**6),
        "analytic force or motion mismatch",
    )
    rejected = {}
    for name, budget, axes, marker in (
        ("wrong_map", 37, ((0, -1),), "constant map"),
        ("missing_budget", 35, ((1, -1),), "budget"),
    ):
        try:
            run(budget, axes)
        except ValueError as exc:
            need(marker in str(exc), "unrelated negative")
            rejected[name] = str(exc)
        else:
            raise ValueError("negative control accepted")
    data = dict(
        accepted=True,
        queries=len(ledger),
        all_moments_exact=True,
        position_error_m=float(pos),
        speed_error_m_s=float(vel),
        pressure_J_error_ns=float(je),
        pressure_B_error_per_mass_m=float(be),
        whole_law_J_defect_ns=float(law),
        whole_material_position_defect_m=float(material),
        proper_pair_words=True,
        negatives=rejected,
    )
    save(root / "levels-control.json", data)
    print(data)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--phase", choices=("transfer", "red"), required=True)
    args = cli.parse_args()
    (red if args.phase == "red" else transfer)(args.root)
