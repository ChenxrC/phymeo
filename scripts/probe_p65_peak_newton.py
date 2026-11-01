"""A bounded Newton peak experiment and independent accepted/rejected readback."""

import argparse
import json
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
from check_p65_common_motion import add, derive, outward, scale, times
from check_p65_peak_tangent import PLAN, baseline
from check_p65_sc2_causal_entries import Curve, H, finish
from check_p65_sc2_common_entry import difference, evaluate
from check_p65_sc2_event_cell import audit_chart, blank, packet_hash, shift_and_arm
from p65_common_motion import checkpoint, endpoint, need, unpack
from p65_peak_tangent import peak_fit
from probe_p65_common_motion import joint_state, saved
from probe_p65_joint_readback import save
from probe_p65_sc2_causal_entries import checked_publish, transaction
from probe_p65_sc2_common_entry import original
from run_p65_sc2_step_tail import filehash, successor
from trace_p65_sc2_common_window import poly_range
from verify_p65_contact_layer_screen import sha

from phymeo.core.scene import fingerprint
from phymeo.validation.bending_bounds import Interval as I
from phymeo.validation.bending_bounds import cross, dot, sqrt_interval


def source_data(model, bind):
    old, record = baseline()
    link = next(row for row in record["raw"] if row["name"] == "d-link.json")
    need(filehash(old / "d-link.json") == link["sha256"], "uncertified d link")
    source = successor(old, model, bind, "peak")
    need(
        source["previous_sha"]
        == next(r["sha256"] for r in record["raw"] if r["name"] == "d-accepted.json")
        and packet_hash(source["input"]) == record["d"]["endpoint"]
        and source["prefix_charges"] == 288,
        "accepted d checkpoint mismatch",
    )
    source["binding"] = fingerprint(
        (sha(PLAN, normalize=True), model.signature, bind, source["input"], source["previous_sha"])
    )
    source["step"] = source["h"]
    return source


def fields_at(words, s):
    from check_p65_sc2_event_cell import word_polys

    return [evaluate(word_polys(words[:, i]), s) for i in range(words.shape[1])]


def propose(root, model, source, *, lifting=None):
    gate_path = source.get("derivative_gate", root / "derivative.json")
    gate = saved(gate_path)
    need(
        gate["accepted"] and gate["fixed_states"] == 12 and gate["coordinate_columns"] == 26,
        "derivatives not verified",
    )
    q, v, t = unpack(source["input"], model.signature)
    h = source["h"]
    ledger = []
    try:
        claim, terminal = peak_fit(
            model,
            q,
            v,
            h,
            source["binding"],
            512 - source["prefix_charges"],
            ledger,
            lifting=lifting,
        )
        x, u = endpoint(claim, q, v)
        out = checkpoint(x, u, t + h, model.signature)
        oracle = terminal(np.asarray(x, float), np.asarray(u, float))
        pressure = fields_at(claim["pressure"], Q(1))
        need(
            len(ledger) == 37 and claim["force_calls"] == 36 and all(r["returned"] for r in ledger),
            "unmetered force call",
        )
    finally:
        save(root / "provider-ledger.json.gz", ledger)
    data = dict(
        binding=source["binding"],
        input_sha=packet_hash(source["input"]),
        previous_certificate=source["previous_sha"],
        derivative_gate_sha=filehash(gate_path),
        chart=claim,
        charges=len(ledger) * 6,
        search_cells=0,
        endpoint=packet_hash(out),
        terminal_pressure_n=oracle.value.pressures.copy(),
        terminal_proposed_pressure_n=list(map(float, pressure)),
        ledger_sha256=filehash(root / "provider-ledger.json.gz"),
    )
    save(root / "proposal.json.gz", data)
    print(
        "coarse and fine residuals",
        [r["residual"] for r in claim["coarse"]],
        [r["residual"] for r in claim["iterations"]],
    )
    print(
        "new endpoint check",
        max(abs(float(p) - s) for p, s in zip(pressure, oracle.value.pressures, strict=True)),
        "charges",
        data["charges"],
    )


def midpoint(iv):
    return Q(round((iv.low + iv.high) / 2 * (1 << 96)), 1 << 96)


def shift_taylor(curve, ids, lo, hi, x, v, rho):
    # Outward second derivative of rho*(1-nz) in physical time, on this one closed leaf.
    xx = [x[3 * n : 3 * n + 3] for n in ids]
    vv = [v[3 * n : 3 * n + 3] for n in ids]
    aa = [
        [poly_range(scale(derive(curve.v[3 * n + j]), 1 / curve.h), lo, hi) for j in range(3)]
        for n in ids
    ]
    e, u = difference(xx[1], xx[0]), difference(xx[2], xx[0])
    de, du = difference(vv[1], vv[0]), difference(vv[2], vv[0])
    dde, ddu = difference(aa[1], aa[0]), difference(aa[2], aa[0])
    c = cross(e, u)
    cp = [a + b for a, b in zip(cross(de, u), cross(e, du), strict=True)]
    cpp = [
        a + 2 * b + d for a, b, d in zip(cross(dde, u), cross(de, du), cross(e, ddu), strict=True)
    ]
    need(c[2].low > Q(1, 2), "offset Taylor chart")
    r = [c[j] / c[2] for j in range(2)]
    rp = [(cp[j] - r[j] * cp[2]) / c[2] for j in range(2)]
    rpp = [(cpp[j] - r[j] * cpp[2] - 2 * rp[j] * cp[2]) / c[2] for j in range(2)]
    ss = 1 + sum(z.square() for z in r)
    rt = sqrt_interval(ss, 96)
    second = (
        rho
        * (
            (sum(z.square() for z in rp) + dot(r, rpp)) / (ss * rt)
            - 3 * dot(r, rp).square() / (ss.square() * rt)
        )
    ).rounded(96)
    x0 = [I.point(evaluate(z, lo)) for z in curve.x]
    v0 = [I.point(evaluate(z, lo)) for z in curve.v]
    sh, ds, _ = shift_and_arm(x0, v0, ids, rho)
    a, s0, u0 = midpoint(second), midpoint(sh), midpoint(ds)
    span = (hi - lo) * curve.h
    tau = (-lo * curve.h, curve.h)
    sp = add(add((s0,), scale(tau, u0)), scale(times(tau, tau), a / 2))
    up = add((u0,), scale(tau, a))
    spread = (sh - s0).magnitude + span * (ds - u0).magnitude + span**2 * (second - a).magnitude / 2
    drift = (ds - u0).magnitude + span * (second - a).magnitude
    return sp, up, spread, drift


def verify(model, source, data):
    q, v, t = unpack(source["input"], model.signature)
    h = source["h"]
    claim = data["chart"]
    need(
        data["binding"] == claim["binding"] == source["binding"]
        and data["input_sha"] == packet_hash(source["input"])
        and data["previous_certificate"] == source["previous_sha"]
        and t + h == H
        and all(source["mask"]),
        "wrong inlet or duration",
    )
    need(
        len(claim["coarse"]) == 3
        and len(claim["iterations"]) == 2
        and len(claim["nodes"]) == 14
        and all(r["acc"].shape == (4, 13) for r in claim["coarse"])
        and all(r["acc"].shape == (14, 13) for r in claim["iterations"])
        and data["charges"] == 222
        and all(
            r["linear_error"] < 1e-10 and r["projected_error"] < 1e-10
            for r in [*claim["coarse"], *claim["iterations"]]
        ),
        "incomplete nonlinear sweep, failed linear solve, or owner budget",
    )
    proposed_x, proposed_v = endpoint(claim, q, v)
    bound_packet = checkpoint(proposed_x, proposed_v, t + h, model.signature)
    need(packet_hash(bound_packet) == data["endpoint"], "wrong endpoint before interval check")
    variant = source.get("slope_modes", False)
    if variant:
        need(
            all(
                r["slope"]["coefficients"].shape == (21, 97)
                and r["slope"]["feedback_error"] < 1e-10
                for r in claim["iterations"]
            ),
            "incomplete coupled slope history",
        )
    curve = Curve(
        claim, model, q, v, h, "slope-newton" if variant else "active-newton", 20 if variant else 13
    )
    acc = blank()
    acc.update(law_lower_abs=[Q(0)] * 6, law_signed=[I.point(0) for _ in range(6)])
    audit_chart(
        curve,
        model,
        source["mask"],
        Q(1),
        Q(0),
        acc,
        closing=False,
        leaves=512,
        shift_taylor=shift_taylor,
    )
    out, rec = finish(source, model, data, curve, [], source["mask"], Q(1), acc, [], 1)
    rec.update(
        previous_certificate_sha=source["previous_sha"],
        preceding_peak_upper_n=source["prefix_peak"],
        local_absolute_J_error_lower_ns=[
            max(0.0, outward(I.point(z))[0]) for z in acc["law_lower_abs"]
        ],
        local_signed_J_defects_ns=[outward(z) for z in acc["law_signed"]],
        endpoint_force_sample_error_n=max(
            abs(a - b)
            for a, b in zip(
                data["terminal_pressure_n"], data["terminal_proposed_pressure_n"], strict=True
            )
        ),
    )
    rec["peak"]["dominates_prefix"] = rec["peak"]["force_n"][0] > source["prefix_peak"]
    rec["peak"]["interior"] = (
        float(t) < rec["peak"]["time_s"][0] <= rec["peak"]["time_s"][1] < float(H)
    )
    rec["peak"]["accepted"] = (
        rec["accepted"] and rec["peak"]["dominates_prefix"] and rec["peak"]["interior"]
    )
    return out, rec


def publish(root, model, source, *, phase="newton"):
    data = saved(root / "proposal.json.gz")
    ledger = saved(root / "provider-ledger.json.gz")
    need(
        filehash(root / "provider-ledger.json.gz") == data["ledger_sha256"]
        and len(ledger) == 37
        and all(
            z["call"] == i + 1 and z["addresses"] == 6 and z["returned"]
            for i, z in enumerate(ledger)
        )
        and filehash(source.get("derivative_gate", root / "derivative.json"))
        == data["derivative_gate_sha"],
        "missing charged evaluation or derivative gate",
    )
    claim = data["chart"]
    cursor = 1
    for sweep in [*claim["coarse"], *claim["iterations"]]:
        for i, (x, u, p) in enumerate(zip(sweep["q"], sweep["v"], sweep["pressures"], strict=True)):
            observed = ledger[0] if i == 0 else ledger[cursor]
            need(
                np.array_equal(observed["q"], x)
                and np.array_equal(observed["v"], u)
                and np.array_equal(observed["pressures"], p),
                "changed stage or hidden evaluation",
            )
            if i:
                cursor += 1
    q, v, _ = unpack(source["input"], model.signature)
    x1, v1 = endpoint(claim, q, v)
    need(
        cursor == len(ledger) - 1
        and np.array_equal(ledger[-1]["q"], np.asarray(x1, float))
        and np.array_equal(ledger[-1]["v"], np.asarray(v1, float))
        and np.array_equal(ledger[-1]["pressures"], data["terminal_pressure_n"]),
        "wrong endpoint observation",
    )
    _, rec, out = checked_publish(root, phase, model, source, data, verify)
    save(root / "accepted.json", rec)
    with (root / "peak.bin").open("xb") as fp:
        fp.write(out.tobytes())
    print(json.dumps(rec, indent=2))


def readback(root, model, source, *, phase="newton"):
    tx = saved(root / f"{phase}-transaction.json.gz")
    info = saved(root / f"{phase}-attempt.json.gz")
    rec = info["certificate"]
    q, v, t = unpack(source["input"], model.signature)
    need(
        rec["inlet"] == packet_hash(source["input"])
        and rec["accepted"] == tx["report"]["accepted"],
        "wrong outcome",
    )
    history = tx["state"]["history"]["motion"]
    x, u, stop = unpack(history, model.signature)
    if rec["accepted"]:
        need(
            stop == H
            and packet_hash(history) == rec["endpoint"] == filehash(root / "peak.bin")
            and all(rec["gates"].values())
            and rec["peak"]["accepted"]
            and len(tx["substep_states"]) == 1,
            "unbound accepted peak",
        )
    else:
        need(
            x == q
            and u == v
            and stop == t
            and not tx["substep_states"]
            and not (root / "peak.bin").exists(),
            "partial state escaped rejection",
        )
    data = saved(root / "proposal.json.gz")
    spec = baseline()[1]
    # The earlier rejected endpoint is a new negative control; never use it to initialize a solve.
    fake = deepcopy(data)
    fake["endpoint"] = spec["peak_rejection"]["rejected_candidate"]
    fail, _ = transaction(model, source, fake, verify, joint_state(q, v, source["input"]))
    need(
        not fail.report.accepted and "endpoint" in str(fail.report.reason), "accepted old endpoint"
    )
    save(
        root / "readback.json",
        dict(
            accepted=rec["accepted"],
            same_inlet=True,
            returned=packet_hash(history),
            returned_time_s=str(stop),
            peak=rec["peak"],
            cumulative_upper=rec["upper"],
            charges=rec["charges"],
            earlier_attempt_charges=spec["peak_rejection"]["charges"],
            stale_endpoint=str(fail.report.reason),
        ),
    )
    print("readback", rec["accepted"], float(stop), rec["upper"])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True, type=Path)
    cli.add_argument("--phase", required=True, choices=("propose", "verify", "readback"))
    args = cli.parse_args()
    model, _, _, bind = original()
    try:
        src = source_data(model, bind)
        {"propose": propose, "verify": publish, "readback": readback}[args.phase](
            args.root, model, src
        )
    finally:
        model.parts[3].close()
