"""Separate nodal convergence from between-node force representation on one stored path."""

import argparse
from fractions import Fraction as Q
from hashlib import sha256
from math import lcm
from pathlib import Path

from check_p65_common_motion import add, bounds, derive, outward, poly, scale, sub, times
from check_p65_sc2_causal_entries import Curve
from check_p65_sc2_common_entry import AREA_K, WIDTH, evaluate, geometry
from check_p65_sc2_event_cell import moment, options, packet_hash
from p65_common_motion import need, unpack
from probe_p65_common_motion import saved
from probe_p65_joint_readback import save
from probe_p65_peak_newton import source_data
from probe_p65_sc2_common_entry import original

from phymeo.validation.bending_bounds import Interval as I


def points(curve, model, s):
    x = [I.point(evaluate(p, s)) for p in curve.x]
    v = [I.point(evaluate(p, s)) for p in curve.v]
    ports = geometry(x, v, model)
    need(
        all(g.high < WIDTH and rate.high < Q(1, 1000) for g, rate, _ in ports),
        "unresolved active sample",
    )
    return [
        evaluate(p, s) - AREA_K * (WIDTH - g) * (1 - 1000 * z)
        for p, (g, z, _) in zip(curve.ps, ports, strict=True)
    ]


def node_control(curve, model, nodes, cert):
    card = []
    for i, s in enumerate(nodes):
        p = (Q(1),)
        for j, t in enumerate(nodes):
            if i != j:
                p = times(p, (-t / (s - t), 1 / (s - t)))
        card.append(bounds(p))
    lebesgue = max(sum(row[k].magnitude for row in card) for k in range(64))
    samples = [points(curve, model, s) for s in nodes]
    mid = [points(curve, model, (a + b) / 2) for a, b in zip(nodes, nodes[1:])]
    out = []
    for i, owner in enumerate(model.owners):
        eps = max(row[i].magnitude for row in samples)
        lower = max(Q(0), Q(cert["local_absolute_J_error_lower_ns"][i]) - curve.h * lebesgue * eps)
        out.append(
            dict(
                owner=owner,
                node_defect_upper_n=outward(I.point(eps))[1],
                midpoint_defect_nsamples=len(mid),
                midpoint_defects_n=[outward(row[i]) for row in mid],
                same_degree_exact_nodal_absolute_J_lower_ns=max(0.0, outward(I.point(lower))[0]),
            )
        )
    return dict(
        cardinal_lebesgue_upper=outward(I.point(lebesgue))[1],
        nodes=len(nodes),
        midpoints=len(mid),
        owners=out,
    )


class PolynomialProducts:
    def __init__(self):
        self.work = 0

    def multiply(self, a, b):
        self.work += len(a) * len(b)
        need(self.work <= 300000 and len(a) + len(b) - 2 <= 320, "fixed-path polynomial work limit")
        da = lcm(*(x.denominator for x in a))
        db = lcm(*(x.denominator for x in b))
        ia = [int(x * da) for x in a]
        ib = [int(x * db) for x in b]
        zz = [0] * (len(a) + len(b) - 1)
        for i, x in enumerate(ia):
            if x:
                for j, y in enumerate(ib):
                    zz[i + j] += x * y
        den = da * db
        need(
            max(den.bit_length(), max(abs(z).bit_length() for z in zz)) <= 32768,
            "diagnostic coefficient bit limit",
        )
        return poly([Q(z, den) for z in zz])

    def cross(self, a, b):
        return [
            sub(self.multiply(a[i], b[j]), self.multiply(a[j], b[i]))
            for i, j in ((1, 2), (2, 0), (0, 1))
        ]


def algebraic_control(curve, model, cert):
    rho = Q(float(model.rho))
    h = curve.h
    engine = PolynomialProducts()
    data = []
    platform_j = platform_b = Q(0)
    # The path is fixed throughout. No source fitting, contact solve or state update.
    for face_no, ids in enumerate(model.faces):
        xyz = [curve.x[3 * n : 3 * n + 3] for n in ids]
        e = [sub(b, a) for a, b in zip(xyz[0], xyz[1], strict=True)]
        u = [sub(b, a) for a, b in zip(xyz[0], xyz[2], strict=True)]
        c = engine.cross(e, u)
        ref = c[2][0]
        need(ref > 0, "normalized chart reference")
        z = scale(c[2], 1 / ref)
        squared = (Q(0),)
        for p in c:
            squared = add(squared, engine.multiply(p, p))
        w = sub(scale(squared, 1 / ref**2), (1,))
        inverse = add(add((1,), scale(w, -Q(1, 2))), scale(engine.multiply(w, w), Q(3, 8)))
        shift = scale(sub((1,), engine.multiply(z, inverse)), rho)
        sr = scale(derive(shift), 1 / h)
        wbox, wpbox = bounds(w), bounds(scale(derive(w), 1 / h))
        zbox, zpbox = bounds(z), bounds(scale(derive(z), 1 / h))
        need(max(i.magnitude for i in wbox) < Q(1, 2), "binomial remainder chart")
        for corner, n in enumerate(ids):
            a = 3 * face_no + corner
            b = sub(curve.x[3 * n + 2], curve.x[-1])
            speed = sub(curve.v[3 * n + 2], curve.v[-1])
            comp = sub(sub((WIDTH,), b), shift)
            fac = sub((1,), scale(add(speed, sr), 1000))
            pressure = scale(engine.multiply(comp, fac), AREA_K)
            # Direct Bernstein ranges of compression/factor are costly at this degree.
            # Raw face and state enclosures below prove branch and strict sign independently.
            je = be = Q(0)
            worst_w = Q(0)
            remaining_closing = True
            initial_g = options([p[0] for p in curve.x], [p[0] for p in curve.v], model)[a][0]
            for k in range(64):
                lo, hi = Q(k, 64), Q(k + 1, 64)
                x = [bounds(p)[k] for p in curve.x]
                v = [bounds(p)[k] for p in curve.v]
                g, rate, _ = geometry(x, v, model)[a]
                mag = wbox[k].magnitude
                worst_w = max(worst_w, mag)
                # For |w|<=1/2, max |(1+w)^(-1/2)'''| < 45/2.
                rem = Q(15, 4) * mag**3
                drem = Q(45, 4) * mag**2 * wpbox[k].magnitude
                es = rho * zbox[k].magnitude * rem
                er = rho * (zpbox[k].magnitude * rem + zbox[k].magnitude * drem)
                remaining_closing = remaining_closing and rate.high < 0
                lower = (
                    max(WIDTH - g.high, WIDTH - initial_g.high)
                    if remaining_closing
                    else WIDTH - g.high
                )
                need(lower > es and 1 - 1000 * rate.high > 1000 * er, "unproved shadow force sign")
                err = AREA_K * (
                    es * (1 - 1000 * rate).magnitude + ((WIDTH - g).magnitude + es) * 1000 * er
                )
                je += h * (hi - lo) * err
                be += h * h * (1 - lo) * (hi - lo) * err
            pJ = h * moment(pressure, 0, 1)
            weighted = sub(pressure, (Q(0), *pressure))
            pB = h**2 * moment(weighted, 0, 1)
            diff_j = pJ - Q(cert["local_J_ns"][a])
            diff_b = pB - Q(cert["local_B_ns2"][a])
            platform_j -= diff_j
            platform_b -= diff_b
            data.append(
                dict(
                    owner=model.owners[a],
                    degree=len(pressure) - 1,
                    sup_radical_argument=outward(I.point(worst_w))[1],
                    fixed_path_J_error_upper_ns=outward(I.point(je))[1],
                    fixed_path_B_error_over_min_mass_m=outward(
                        I.point(be / Q(float(min(model.mass))))
                    )[1],
                    pressure_J_n_s=outward(I.point(pJ)),
                    pressure_B_n_s2=outward(I.point(pB)),
                    difference_from_proposal_J_ns=outward(I.point(diff_j)),
                    difference_from_proposal_B_ns2=outward(I.point(diff_b)),
                )
            )
    return dict(
        owners=data,
        scalar_term_products=engine.work,
        platform_velocity_change_if_only_these_forces_replaced=outward(
            I.point(platform_j / Q(float(model.mass[-1])))
        ),
        platform_position_change_if_only_these_forces_replaced=outward(
            I.point(platform_b / Q(float(model.mass[-1])))
        ),
    )


def run(root):
    model, _, _, bind = original()
    try:
        src = source_data(model, bind)
        cp = src["input"]
        q, v, t = unpack(cp, model.signature)
        inp = saved(root / "proposal.json.gz")
        cert = saved(root / "newton-attempt.json.gz")["certificate"]
        rb = saved(root / "readback.json")
        need(
            inp["input_sha"] == cert["inlet"] == packet_hash(cp)
            and not cert["accepted"]
            and rb["returned"] == packet_hash(cp)
            and Q(rb["returned_time_s"]) == t,
            "unbound fixed-path comparison",
        )
        curve = Curve(inp["chart"], model, q, v, src["h"], "active-newton", 13)
        nodes = list(map(lambda s: Q(float(s)), inp["chart"]["nodes"]))
        interp = node_control(curve, model, nodes, cert)
        control = algebraic_control(curve, model, cert)
        for i, row in enumerate(control["owners"]):
            row["prefix_plus_fixed_path_J_bound_ns"] = outward(
                I.point(src["totals"]["j"][i] + Q(row["fixed_path_J_error_upper_ns"]))
            )[1]
        result = dict(
            schema="p65-sc2-peak-representation-readback-v1",
            input=packet_hash(cp),
            rejected_proposal_sha=sha256((root / "proposal.json.gz").read_bytes()).hexdigest(),
            solver_force_evaluations=0,
            published_new_state=False,
            nodal=interp,
            algebraic=control,
        )
        save(root / "representation-readback.json", result)
        for row, other in zip(interp["owners"], control["owners"], strict=True):
            print(
                row["owner"],
                "exact-node lower",
                row["same_degree_exact_nodal_absolute_J_lower_ns"],
                "algebraic upper",
                other["fixed_path_J_error_upper_ns"],
            )
    finally:
        model.parts[3].close()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True, type=Path)
    run(cli.parse_args().root)
