"""One fresh bounded SC2 peak candidate using the already evaluated material and contact slopes."""

import argparse
import json
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
from check_p65_common_motion import add, derive, outward
from check_p65_peak_slope import BASE, PLAN
from check_p65_sc2_causal_entries import Curve
from check_p65_sc2_common_entry import evaluate
from check_p65_sc2_event_cell import packet_hash
from p65_common_motion import need, unpack
from p65_peak_slope import SlopeModes
from probe_p65_common_motion import saved
from probe_p65_joint_readback import save
from probe_p65_peak_newton import propose, publish, readback, source_data
from probe_p65_sc2_common_entry import original
from run_p65_sc2_step_tail import filehash
from trace_p65_sc2_common_window import poly_range
from verify_p65_contact_layer_screen import sha

from phymeo.core.scene import fingerprint
from phymeo.validation.bending_bounds import Interval as I


def bound_source(root, model, bind):
    spec = json.loads(PLAN.read_bytes())
    parent = BASE / spec["previous"]
    need(sha(parent, normalize=True) == spec["previous_sha256_lf"], "different saved failure")
    fail = json.loads(parent.read_bytes())
    p = BASE.parents[1] / fail["evidence_root"] / "derivative.json"
    need(
        filehash(p) == next(r["sha256"] for r in fail["raw"] if r["name"] == p.name)
        and saved(p)["accepted"],
        "unbound derivative gate",
    )
    control = saved(root / "basis-check.json")
    need(
        control["accepted"] and control["polynomial_controls"] == 21,
        "unverified slope representation",
    )
    src = source_data(model, bind)
    src.update(
        binding=fingerprint((sha(PLAN, normalize=True), src["binding"], "peak slope v1")),
        slope_modes=True,
        derivative_gate=p,
        earlier_result=fail,
    )
    return src


def peak_ranges(curve, model, cert):
    # Independent bounds on both maxima. The certified relative error holds at every positive time.
    poly = (Q(0),)
    for p in curve.ps:
        poly = add(poly, p)
    dp = derive(poly)
    eta = Q(max(cert["local_relative"]))
    need(0 <= eta < 1, "unusable relative bound")
    t = Q(cert["start_s"])
    h = curve.h
    ranges = []
    best = Q(0)
    created = 0

    def insert(a, b, depth):
        nonlocal best, created
        vals = [evaluate(poly, s) for s in (a, (a + b) / 2, b)]
        best = max(best, max(vals))
        d = poly_range(dp, a, b)
        upper = max(vals[0], vals[-1]) if d.low >= 0 or d.high <= 0 else poly_range(poly, a, b).high
        ranges.append((a, b, depth, upper))
        created += 1

    insert(Q(0), Q(1), 0)
    limit = min(1023, 4096 - cert["cells"]["total"])
    need(limit >= 1, "no peak cells available")
    while created + 2 <= limit:
        choices = [
            (r[1] - r[0], r[3], i)
            for i, r in enumerate(ranges)
            if r[3] / (1 - eta) >= best / (1 + eta) and r[2] < 22
        ]
        if not choices:
            break
        _, _, index = max(choices)
        a, b, d, _ = ranges.pop(index)
        mid = (a + b) / 2
        insert(a, mid, d + 1)
        insert(mid, b, d + 1)
    upper = max(r[3] for r in ranges)
    preceding = Q(cert["preceding_peak_upper_n"])
    need(best / (1 + eta) > preceding, "earlier law maximum cannot be excluded")
    need(
        best > preceding * (1 + Q(cert["upper"]["relative_positive"])),
        "earlier represented maximum cannot be excluded",
    )

    def support(matches):
        return outward(I(t + h * min(x[0] for x in matches), t + h * max(x[1] for x in matches)))

    return dict(
        represented_force_n=outward(I(best, upper)),
        represented_time_s=support([z for z in ranges if z[3] >= best]),
        law_force_n=outward(I(best / (1 + eta), upper / (1 - eta))),
        law_time_s=support([z for z in ranges if z[3] / (1 - eta) >= best / (1 + eta)]),
        new_time_cells=created,
        total_including_existing=cert["cells"]["total"] + created,
        full_first_original_step=True,
    )


def window_readback(root, model, src, *, phase="slope", mode="slope-newton"):
    readback(root, model, src, phase=phase)
    status = saved(root / "readback.json")
    if not status["accepted"]:
        print(phase, "candidate rejected; exact d checkpoint retained")
        return
    proposal = saved(root / "proposal.json.gz")
    cert = saved(root / "accepted.json")
    q, v, t = unpack(src["input"], model.signature)
    packet = np.frombuffer((root / "peak.bin").read_bytes(), dtype="<i4")
    x, u, T = unpack(packet, model.signature)
    need(
        packet_hash(packet) == status["returned"] == cert["endpoint"] and T == Q(1, 64),
        "returned checkpoint",
    )
    curve = Curve(proposal["chart"], model, q, v, src["h"], mode, 20)
    j, b = linked_moments(model, T, cert, d_cert=src.get("paired_d"))
    mass = Q(float(model.mass[-1]))
    need(
        mass * u[-1] == -sum(j) and mass * x[-1] == -sum(b), "global platform force or time moment"
    )
    result = dict(
        exact_owner_J_ns=list(map(str, j)),
        exact_owner_B_ns2=list(map(str, b)),
        total_J_ns=float(sum(j)),
        total_B_ns2=float(sum(b)),
        platform_speed_m_s=float(u[-1]),
        platform_q_m=float(x[-1]),
        all_address_moments_reproduce_platform=True,
        peak=peak_ranges(curve, model, cert),
        physical_peak_accepted=True,
    )
    save(root / "whole-step.json", result)
    print(json.dumps(result["peak"], indent=2))


def linked_moments(model, T, cert=None, *, d_cert=None):
    parent = json.loads((BASE / "p65-sc2-causal-entry-results.json").read_bytes())
    entry = json.loads((BASE / "p65-sc2-common-motion-entry-results.json").read_bytes())["entry"]
    d = parent["d"] if d_cert is None else d_cert
    need(
        parent["cluster"]["inlet"] == entry["checkpoint_sha256"]
        and d["inlet"] == parent["cluster"]["endpoint"]
        and (cert is None or cert["inlet"] == d["endpoint"]),
        "discontinuous prefix",
    )
    j = [Q(0)] * 6
    b = j.copy()
    k = model.owners.index(entry["first"])
    j[k] = Q(entry["J_ns"])
    b[k] = Q(entry["B_ns2"]) + (T - Q(entry["bridge_end_s"])) * j[k]
    segments = [parent["cluster"], d]
    if cert is not None:
        segments.append(cert)
    need(Q(segments[-1]["stop_s"]) == T, "unbound moment origin")
    for segment in segments:
        for i in range(6):
            sj = Q(segment["local_J_ns"][i])
            j[i] += sj
            b[i] += Q(segment["local_B_ns2"][i]) + (T - Q(segment["stop_s"])) * sj
    return j, b


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True, type=Path)
    cli.add_argument("--phase", choices=("propose", "verify", "readback"), required=True)
    args = cli.parse_args()
    model, _, _, bind = original()
    try:
        src = bound_source(args.root, model, bind)
        if args.phase == "propose":
            propose(args.root, model, src, lifting=SlopeModes())
        elif args.phase == "verify":
            publish(args.root, model, src, phase="slope")
        else:
            window_readback(args.root, model, src)
    finally:
        model.parts[3].close()
