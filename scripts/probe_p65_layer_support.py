"""Publish a single registered SC-L1 support step from the previous step's disk packet."""

import argparse
import json
import math
from fractions import Fraction as Q
from math import comb
from pathlib import Path

from probe_p65_layer_entry import append, fields, half_controls, line, read
from probe_p65_layer_joint_publication import decode, packet, ratio
from verify_p65_contact_layer_screen import INPUT, INPUT_SHA, fit, sha

PLAN = INPUT.with_name("p65-contact-layer-support-protocol.json")
PLAN_SHA = "7e9a5854d41d26bfc7bb7adfe86caf64d09cded47122bf92db9d3e897d7bbf96"


def support_pressure(g, y, dt, mass, force, layer, work):
    """Taylor force on the loaded motion. A constant force has one nonzero power coefficient."""
    x = [layer.delta - g]
    rate = [y]
    power = []
    for k in range(8):
        power.append(layer.k * sum(x[j] * rate[k - j] for j in range(k + 1)))
        work["convolution_terms"] += k + 1
        work["force_charges"] += 1
        if k < 7:
            x.append(dt * (rate[k] - int(k == 0)) / (layer.d * (k + 1)))
            drive = force if k == 0 else 0
            rate.append(dt * layer.d * (drive - power[k]) / (mass * (k + 1)))
    coeff = []
    for i in range(8):
        value = sum(power[k] * Q(comb(i, k), comb(7, k)) for k in range(i + 1))
        work["conversion_terms"] += i + 1
        if value < 0:
            raise ArithmeticError("negative predicted traction")
        coeff.append(float(value).hex())
    return coeff


def advance(folder, seed, step, h, M, F, layer, rate_max, cap):
    folder.mkdir(parents=True, exist_ok=False)
    t0, g0, y0 = decode(seed)
    if t0 != step * h or not 0 <= g0 < layer.delta:
        raise ValueError("unsupported support inlet")
    rate = max(
        float(layer.k * layer.d * layer.delta / M),
        math.sqrt(float(layer.k * (1 + layer.d * rate_max) / M)),
    )
    n = max(1, math.ceil(8 * float(h) * rate))
    dt = h / n
    work = fields("loaded-series7")
    states, forces, ledger = (
        folder / file for file in ("states.jsonl", "forces.jsonl", "pieces.jsonl")
    )
    with states.open("xb") as motion, forces.open("xb") as macros, ledger.open("xb") as pieces:
        offset = append(motion, seed)
        for i in range(0, n, 2):
            parts = min(n - i, 2)
            t, g, y = decode(read(states, offset))
            if t != t0 + i * dt or work["force_charges"] + 8 > cap:
                raise ValueError("macro time or resource limit")
            words = support_pressure(g, y, parts * dt, M, F, layer, work)
            addr = append(
                macros, line(dict(t=str(t), parts=parts, dt=str(parts * dt), words=words))
            )
            for side in range(parts):
                t, g, y = decode(read(states, offset))
                if t != t0 + (i + side) * dt:
                    raise ValueError("discontinuous disk state")
                p = [Q(float.fromhex(w)) for w in json.loads(read(forces, addr))["words"]]
                if parts == 2:
                    p = half_controls(p, side, work)
                order = len(p)
                j = dt * sum(p) / order
                b = dt**2 * sum((order - k) * v for k, v in enumerate(p)) / (order * (order + 1))
                f_j, f_b = F * dt, F * dt**2 / 2
                g1 = g - dt * (y - 1) / layer.d + (b - f_b) / M
                y1 = y + layer.d * (f_j - j) / M
                offset = append(motion, packet(t + dt, g1, y1))
                append(
                    pieces,
                    line(
                        dict(
                            macro=i // 2,
                            side=side,
                            j=ratio(j),
                            b=ratio(b),
                            external_n=ratio(F),
                            external_j=ratio(f_j),
                            external_b=ratio(f_b),
                            external_work=ratio(F * (g - g1)),
                        )
                    ),
                )
        final = read(states, offset)
    t1, g1, y1 = decode(final)
    if t1 != (step + 1) * h:
        raise ValueError("unfinished interval")
    (folder / "end.state.json").write_bytes(final)
    return dict(
        work=work,
        parts=n,
        force_macros=(n + 1) // 2,
        initial_elastic_j=ratio(layer.k * (layer.delta - g0) ** 2 / 2),
        final_elastic_j=ratio(layer.k * (layer.delta - g1) ** 2 / 2),
        initial_kinetic_j=ratio(M * ((y0 - 1) / layer.d) ** 2 / 2),
        final_kinetic_j=ratio(M * ((y1 - 1) / layer.d) ** 2 / 2),
        hashes={
            f: sha(folder / f)
            for f in ("states.jsonl", "forces.jsonl", "pieces.jsonl", "end.state.json")
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    if sha(PLAN, normalize=True) != PLAN_SHA or sha(INPUT, normalize=True) != INPUT_SHA:
        raise ValueError("registered support inputs changed")
    plan = json.loads(PLAN.read_bytes())
    fixture = json.loads(INPUT.read_bytes())
    c = fixture["controls"]
    chosen = plan["case"]
    m, h, T = map(Q, (chosen["m_kg"], chosen["h_s"], chosen["T_s"]))
    if (
        str(m) not in c["trace_masses_kg"]
        or str(h) not in c["report_h_s"]
        or T != Q(c["window_s"])
        or T / h != chosen["steps"]
        or not 0 <= args.step < chosen["steps"]
        or Q(c["area_m2"]) != 1
        or c["sc1"]["compression_initial"] != "delta"
        or c["sc1"]["compressive_load_n"] != "m/4"
        or c["sc1"]["reduced_mass_kg"] != "2m"
    ):
        raise ValueError("unexpected original support control")
    rows = [*fixture["fit"]["static"], fixture["fit"]["compression"]]
    layer = fit(rows, [Q(r["pressure_pa"]) for r in rows])
    wire = json.loads(INPUT.with_name(plan["wire_from"]).read_bytes())["wire_limits"]
    if args.step == 0:
        origin = "support initial g=v=0"
        seed = packet(
            Q(0), Q(chosen["initial_gap"]), 1 + layer.d * Q(chosen["initial_closing_speed"])
        )
    else:
        origin = f"step-{args.step - 1}/end.state.json"
        seed = (args.root / origin).read_bytes()
    folder = args.root / f"step-{args.step}"
    result = advance(
        folder,
        seed,
        args.step,
        h,
        2 * m,
        m / 4,
        layer,
        Q(fixture["rate_range_m_s"][1]),
        wire["force_evaluations_per_outer_step"],
    )
    result.update(
        step=args.step,
        origin=origin,
        protocol_sha256_lf=PLAN_SHA,
        source_sha256_lf=sha(Path(__file__), normalize=True),
    )
    (folder / "index.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(
        f"support step {args.step}, {result['parts']} pieces, {result['work']['force_charges']} force coefficients"
    )


if __name__ == "__main__":
    main()
