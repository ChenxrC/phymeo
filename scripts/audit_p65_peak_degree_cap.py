"""Check the original pressure degree cap on a retained motion, without a dynamics solve."""

import argparse
from fractions import Fraction as Q
from math import comb, lcm
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from audit_p65_peak_representation import PolynomialProducts
from check_p65_common_motion import add, derive, outward, scale, sub, times
from check_p65_sc2_causal_entries import Curve
from check_p65_sc2_common_entry import AREA_K, WIDTH
from p65_common_motion import DEGREE, GRID, need, unpack
from probe_p65_common_motion import saved
from probe_p65_joint_readback import save
from run_p65_sc2_step_tail import filehash
from verify_p65_contact_layer_screen import sha

from phymeo.validation.bending_bounds import Interval as I

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "docs/benchmarks"
PLAN = BASE / "p65-sc2-peak-degree-cap-protocol.json"
CAP = Q(1, 10**10)


def cheb_coefficients(p):
    """For s=(1+cos(theta))/2, expand s**k=cos(theta/2)**(2k) exactly."""
    degree = len(p) - 1
    need(0 <= degree <= 320, "fixed diagnostic maximum degree")
    common = lcm(*(v.denominator for v in p))
    need(common.bit_length() < 32768, "basis conversion denominator")
    num = [0] * len(p)
    for k, val in enumerate(p):
        factor = int(common * val) << (2 * (degree - k))
        choose = comb(2 * k, k)
        for n in range(k + 1):
            num[n] += factor * choose * (1 if n == 0 else 2)
            choose = choose * (k - n) // (k + n + 1)
    den = common << (2 * degree)
    need(max(abs(x).bit_length() for x in num) < 65536, "basis conversion numerator")
    a = [Q(x, den) for x in num]
    need(sum(a) == sum(p) and sum(v * (-1) ** i for i, v in enumerate(a)) == p[0], "endpoints")
    return a


def legendre_witness(p, order):
    # Integral of s**k * P_order(2s-1) by Rodrigues and integration by parts.
    weight = Q(1, (2 * order + 1) * comb(2 * order, order))
    total = Q(0)
    for k in range(order, len(p)):
        total += p[k] * weight
        weight *= Q((k + 1) ** 2, (k + 1 - order) * (k + order + 2))
    return total


def cheb_moments(a):
    def mean(n):
        return Q(0) if n % 2 else Q(1, 1 - n * n)

    impulse = sum(c * mean(n) for n, c in enumerate(a))
    backward = sum(
        c * (mean(n) / 2 - (mean(n + 1) + mean(abs(n - 1))) / 4) for n, c in enumerate(a)
    )
    return impulse, backward


def controls():
    prev, p = (Q(0),), (Q(1),)
    for n in range(23):
        need(cheb_coefficients(p) == [Q(0)] * n + [Q(1)], "Chebyshev mode")
        integral, backward = cheb_moments(cheb_coefficients(p))
        need(integral == sum(v / (k + 1) for k, v in enumerate(p)), "uniform zeroth moment")
        need(
            backward == sum(v / ((k + 1) * (k + 2)) for k, v in enumerate(p)),
            "uniform first moment",
        )
        for k in (21, 22):
            exact = sum(
                c * Q(comb(j, k), (j + k + 1) * comb(j + k, k)) for j, c in enumerate(p) if j >= k
            )
            need(legendre_witness(p, k) == exact and (n >= k or exact == 0), "orthogonal witness")
        if n == 0:
            nxt = (Q(-1), Q(2))
        else:
            nxt = sub(times((Q(-2), Q(4)), p), prev)
        prev, p = p, nxt


def fixed_path():
    spec = saved(PLAN)
    record_path = BASE / spec["source_result"]
    need(sha(record_path, normalize=True) == spec["source_sha256_lf"], "reference record")
    record = saved(record_path)
    prior_dir = REPO / record["evidence_root"]
    raw = {r["name"]: r["sha256"] for r in record["raw"]}
    for name in ("proposal.json.gz", "representation-readback.json"):
        need(filehash(prior_dir / name) == raw[name], "fixed path evidence changed")
    proposal = saved(prior_dir / "proposal.json.gz")
    reference = saved(prior_dir / "representation-readback.json")
    need(
        reference["rejected_proposal_sha"] == raw["proposal.json.gz"],
        "different rejected trajectory",
    )
    parent = saved(BASE / "p65-sc2-causal-entry-results.json")
    d = parent["d"]
    packet_path = REPO / parent["evidence_root"] / "d.bin"
    need(
        d["accepted"]
        and filehash(packet_path) == d["endpoint"] == proposal["input_sha"] == reference["input"],
        "unbound inlet",
    )
    packet = np.frombuffer(packet_path.read_bytes(), dtype="<i4")
    q, v, start = unpack(packet, d["model"])
    step = saved(BASE / "p65-sc2-common-motion-entry-protocol.json")
    fixture_path = BASE / step["fixture"]
    need(sha(fixture_path, normalize=True) == step["fixture_sha256_lf"], "fixture binding")
    fixture = saved(fixture_path)
    cloth = fixture["cloth"]
    mass = Q(cloth["m_kg"])
    need(fixture["platform"]["mass"] == "4m", "unexpected platform")
    model = SimpleNamespace(
        owners=tuple(d["owners"]),
        mass=np.array([float(mass)] * 12 + [float(4 * mass)]),
        external=np.zeros(13),
    )
    h = Q(step["h"]) - start
    need(
        proposal["endpoint"] == record["candidate"]["candidate_endpoint"]
        and Q(record["readback"]["returned_time_s"]) == start
        and not record["readback"]["accepted"],
        "source result is not the retained rejection",
    )
    curve = Curve(proposal["chart"], model, q, v, h, "active-newton", 13)
    return curve, cloth, d, reference, raw["proposal.json.gz"]


def face_shift(curve, vertices, rho, products):
    xyz = [curve.x[3 * j : 3 * j + 3] for j in vertices]
    e, u = [[sub(y, x) for x, y in zip(xyz[0], z, strict=True)] for z in xyz[1:]]
    c = products.cross(e, u)
    ref = c[2][0]
    need(ref > 0, "reference normal")
    w = sub(scale(sum_polys(products.multiply(z, z) for z in c), 1 / ref**2), (1,))
    inv = add(add((1,), scale(w, -Q(1, 2))), scale(products.multiply(w, w), Q(3, 8)))
    return scale(sub((1,), products.multiply(scale(c[2], 1 / ref), inv)), rho)


def sum_polys(items):
    result = (Q(0),)
    for p in items:
        result = add(result, p)
    return result


def bounds_for_owner(p, h, ref, prefix):
    a = cheb_coefficients(p)
    need(len(p) - 1 == ref["degree"], "algebraic degree changed")
    j, b = cheb_moments(a)
    for moment, key in ((h * j, "pressure_J_n_s"), (h * h * b, "pressure_B_n_s2")):
        lo, hi = map(Q, ref[key])
        need(lo <= moment <= hi, "analytic moments disagree with the archived enclosures")
    words = [round(c * GRID) for c in a[: DEGREE + 1]]
    need(len(words) == DEGREE + 1 and max(map(abs, words)) < 1 << 63, "coefficient grid")
    quant = h * sum(abs(c - Q(word, GRID)) for c, word in zip(a, words))
    tail = h * sum(abs(c) for c in a[DEGREE + 1 :])
    analytic = Q(ref["fixed_path_J_error_upper_ns"])
    upper = quant + tail + analytic
    lower = max(Q(0), *(h * abs(legendre_witness(p, n)) - analytic for n in (21, 22)))
    need(lower <= upper, "inconsistent fixed-path bounds")

    def up(z):
        return outward(I.point(z))[1]

    return dict(
        owner=ref["owner"],
        degree=len(p) - 1,
        quantized_force_words=words,
        discarded_norm_ns=up(tail),
        coefficient_rounding_ns=up(quant),
        analytic_remainder_ns=float(analytic),
        local_impulse_upper_ns=up(upper),
        inherited_impulse_upper_ns=float(prefix),
        cumulative_impulse_upper_ns=up(prefix + upper),
        necessary_local_lower_ns=max(0.0, outward(I.point(lower))[0]),
        within_original_budget=prefix + upper <= CAP,
        impossible_on_this_path=lower > CAP,
    )


def run(root):
    controls()
    curve, cloth, d, reference, digest = fixed_path()
    baseline = reference["algebraic"]["owners"]
    engine = PolynomialProducts()
    rows = []
    for face, vertices in enumerate(cloth["faces"]):
        shift = face_shift(curve, vertices, Q(cloth["half_thickness_m"]), engine)
        rate = scale(derive(shift), 1 / curve.h)
        for corner, node in enumerate(vertices):
            index = 3 * face + corner
            need(d["owners"][index] == baseline[index]["owner"], "owner identity")
            b = sub(curve.x[3 * node + 2], curve.x[-1])
            v = sub(curve.v[3 * node + 2], curve.v[-1])
            p = scale(
                engine.multiply(sub(sub((WIDTH,), b), shift), sub((1,), scale(add(v, rate), 1000))),
                AREA_K,
            )
            rows.append(
                bounds_for_owner(
                    p, curve.h, baseline[index], Q(d["cumulative_J_error_upper_ns"][index])
                )
            )
    decision = (
        "sufficient_on_fixed_motion"
        if all(r["within_original_budget"] for r in rows)
        else "infeasible_on_fixed_motion"
        if any(r["impossible_on_this_path"] for r in rows)
        else "unresolved"
    )
    result = dict(
        schema="p65-sc2-peak-degree-cap-results-v1",
        protocol=PLAN.name,
        projection_source_sha256=digest,
        inlet=d["endpoint"],
        representation="shifted Chebyshev, stored signed words / 2^60",
        original_cap_per_owner_ns=float(CAP),
        shape_controls=23,
        fixed_orders_for_dual_witness=[21, 22],
        decision=decision,
        owners=rows,
        polynomial_term_products=engine.work,
        dynamics_or_force_calls=0,
        issued_state=False,
    )
    save(root / "degree-cap.json", result)
    print(decision, [r["cumulative_impulse_upper_ns"] for r in rows])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True, type=Path)
    run(cli.parse_args().root)
