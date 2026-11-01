"""Restart accepted SC2 events and attempt the remaining branches of the original step."""

import argparse
import json
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path

import numpy as np
from check_p65_sc2_causal_entries import (
    Curve,
    H,
    Ramp,
    cluster_audit,
    finish,
    represented_end,
    require_order,
)
from check_p65_sc2_causal_regression import BASE, PLAN
from check_p65_sc2_event_cell import audit_chart, blank, options, packet_hash
from p65_common_motion import checkpoint, endpoint, fit, need, unpack
from probe_p65_common_motion import saved
from probe_p65_joint_readback import save
from probe_p65_sc2_causal_entries import Meter, binding, checked_publish, isolate_round, make_ramp
from probe_p65_sc2_common_entry import original
from probe_p65_sc2_event_cell import InitialBranch

from phymeo.validation.bending_bounds import Interval as I


def filehash(path):
    return sha256(path.read_bytes()).hexdigest()


def successor(root, model, bind, phase):
    previous = {"d": "cluster", "peak": "d"}[phase]
    link = json.loads((root / f"{previous}-link.json").read_bytes())
    certpath = root / f"{previous}-accepted.json"
    packetpath = root / f"{previous}.bin"
    txpath = root / f"{previous}-transaction.json.gz"
    need(
        link["phase"] == previous
        and link["certificate"] == filehash(certpath)
        and link["checkpoint"] == filehash(packetpath)
        and link["transaction"] == filehash(txpath),
        "unbound restart evidence",
    )
    old = saved(certpath)
    tx = saved(txpath)
    packet = np.frombuffer(packetpath.read_bytes(), dtype="<i4").copy()
    q, v, t = unpack(packet, model.signature)
    need(
        old["accepted"]
        and all(old["gates"].values())
        and old["endpoint"] == packet_hash(packet)
        and old["owners"] == list(model.owners)
        and Q(old["stop_s"]) == t
        and tx["report"]["accepted"]
        and np.array_equal(tx["state"]["history"]["motion"], packet),
        "restart was not actually published",
    )
    bounds = old["upper"]
    totals = dict(
        j=list(map(Q, old["cumulative_J_error_upper_ns"])),
        b=list(map(Q, old["cumulative_B_error_over_mass_m"])),
        pos=Q(bounds["node_position_m"]),
        vel=Q(bounds["node_velocity_m_s"]),
        energy=Q(bounds["positive_energy_j"]),
        angular=Q(bounds["paired_angular_nms"]),
        linear=Q(bounds["linear_ns"]),
        rel=Q(bounds["relative_positive"]),
    )
    ports = options(q, v, model)
    mask = [g.high < Q(1, 500) for g, _, _ in ports]
    need(
        all(on or g.low > Q(1, 500) for on, (g, _, _) in zip(mask, ports, strict=True)),
        "ambiguous inlet",
    )
    if phase == "d":
        waiting = [
            (g.high - Q(1, 500)) / (-s.high)
            for on, (g, s, _) in zip(mask, ports, strict=True)
            if not on and s.high < 0
        ]
        need(len(waiting) == 1 and mask.count(False) == 1, "next single birth chart")
        grid = H / 1024
        h = min(H - t, grid * (Q(3, 2) * min(waiting) / grid).__ceil__())
    else:
        need(all(mask), "peak requires all six active at the inlet")
        h = H - t
    need(h > 0 and Q(float(h)) == h, "unbound duration")
    if previous == "cluster":
        parent = json.loads((BASE / json.loads(PLAN.read_bytes())["initial"]).read_bytes())["entry"]
        live = Q(parent["bridge_end_s"]) - Q(parent["event_seconds"][1])
        prefix = float(
            (18 * Q(parent["J_ns"]) / live + 60 * Q(parent["B_ns2"]) / live**2)
            / (1 - Q(parent["incident_relative_error_upper"]))
        )
    else:
        prefix = old["preceding_peak_upper_n"]
    return dict(
        input=packet,
        h=h,
        mask=mask,
        binding=binding(model, bind, packet, phase),
        previous_sha=link["certificate"],
        previous_phase=previous,
        totals=totals,
        prefix_charges=old["charges"]["total"],
        prefix_cells=old["cells"]["total"],
        prefix_peak=max(prefix, old["peak"]["force_n"][1]),
    )


def proposal(root, model, source, phase):
    q, v, t = unpack(source["input"], model.signature)
    h = source["h"]
    mask = source["mask"]
    degree = 8 if phase == "d" else 11
    meter = Meter(model, source)
    _, g, s = meter.maps(np.asarray(q, float), np.asarray(v, float))
    need(
        [bool(z < 0.002 and r < 0.001) for z, r in zip(g, s, strict=True)] == mask,
        "observed branch",
    )
    provider = InitialBranch(model, mask) if phase == "d" else model
    claim = fit(
        provider,
        q,
        v,
        h,
        source["binding"],
        charge_limit=meter.limit - meter.count,
        degree=degree,
        sweeps=3,
        paired_axes=source.get("paired_axes", ()),
    )
    meter.count += claim["owner_charges"]
    curve = Curve(claim, model, q, v, h, "chebyshev", degree)
    data = []
    work = 0
    end = Q(1)
    if phase == "d":
        _, hit, work = isolate_round(curve, model, [], {i for i, z in enumerate(mask) if not z})
        need(hit is not None, "no birth on inlet-selected horizon")
        end = Q((hit[1] * 4096).__ceil__() + 1, 4096)
        need(end <= 1, "join outside search horizon")
        wire = make_ramp(model, curve, [], hit, meter, end)
        data = [wire]
        ramps = [Ramp(wire, model, h, end)]
        x, u = represented_end(curve, model, q, v, ramps, end)
    else:
        x, u = endpoint(claim, q, v)
    need(source["prefix_charges"] + meter.count <= 512, "original owner cap")
    out = checkpoint(x, u, t + h * end, model.signature)
    save(
        root / f"{phase}-proposal.json.gz",
        dict(
            phase=phase,
            binding=source["binding"],
            input_sha=packet_hash(source["input"]),
            previous_certificate=source["previous_sha"],
            h=str(h),
            end=str(end),
            active=mask,
            chart=claim,
            ramps=data,
            charges=meter.count,
            search_cells=work,
            endpoint=packet_hash(out),
        ),
    )
    print("new address calls", meter.count, "endpoint time", float(t + h * end))


def verify(model, source, data):
    q, v, t = unpack(source["input"], model.signature)
    h = source["h"]
    end = Q(data["end"])
    d = data["phase"] == "d"
    need(
        data["phase"] in ("d", "peak")
        and (source["previous_phase"] == "cluster") == d
        and data["binding"] == data["chart"]["binding"] == source["binding"]
        and data["input_sha"] == packet_hash(source["input"])
        and data["previous_certificate"] == source["previous_sha"]
        and data["active"] == source["mask"]
        and Q(data["h"]) == h
        and h * end == source["step"],
        "different restart",
    )
    degree = 8 if d else 11
    for coordinate, sign in source.get("paired_axes", ()):
        need(
            np.array_equal(
                data["chart"]["rows"][:, :, coordinate], sign * data["chart"]["pressure"]
            ),
            "different scalar and mapped pressure words",
        )
    curve = Curve(data["chart"], model, q, v, h, "chebyshev", degree)
    ramps = [Ramp(z, model, h, end) for z in data["ramps"]]
    need(
        len(ramps) == int(d)
        and data["charges"] == 6 * (2 + degree * 3 + int(d))
        and data["search_cells"] == (65 if d else 0),
        "tail execution ledger",
    )
    acc = blank()
    acc.update(law_lower_abs=[Q(0)] * 6, law_signed=[I.point(0) for _ in range(6)])
    if d:
        need(end == Q((ramps[0].hi * 4096).__ceil__() + 1, 4096), "wrong join")
        require_order(curve, model, source["mask"], ramps)
        events = cluster_audit(curve, model, source["mask"], ramps, end, acc, leaves=256)
    else:
        need(end == 1 and t + h == H and all(source["mask"]), "original step end or active set")
        audit_chart(curve, model, source["mask"], end, Q(0), acc, closing=False, leaves=512)
        events = []
    out, rec = finish(
        source, model, data, curve, ramps, source["mask"], end, acc, events, 2 * len(ramps) + 1
    )
    rec.update(
        previous_certificate_sha=source["previous_sha"],
        preceding_peak_upper_n=source["prefix_peak"],
        same_published_restart=True,
        local_smooth_absolute_J_error_lower_ns=[
            max(0.0, float(z) * (1 - 2**-52)) for z in acc["law_lower_abs"]
        ],
        local_smooth_signed_J_error_ns=[[float(z.low), float(z.high)] for z in acc["law_signed"]],
    )
    if not d:
        rec["peak"]["covers_first_original_step"] = (
            rec["peak"]["force_n"][0] > source["prefix_peak"]
        )
        rec["peak"]["interior"] = (
            float(t) < rec["peak"]["time_s"][0] <= rec["peak"]["time_s"][1] < float(H)
        )
    return out, rec


def publish(root, model, source, phase):
    path = root / f"{phase}-proposal.json.gz"
    data = saved(path)
    source["step"] = source["h"] * Q(data["end"])
    need(Q(float(source["step"])) == source["step"], "public step representation")
    _, cert, packet = checked_publish(root, phase, model, source, data, verify)
    save(root / f"{phase}-accepted.json", cert)
    with (root / f"{phase}.bin").open("xb") as fp:
        fp.write(packet.tobytes())
    save(
        root / f"{phase}-link.json",
        dict(
            phase=phase,
            certificate=filehash(root / f"{phase}-accepted.json"),
            checkpoint=packet_hash(packet),
            transaction=filehash(root / f"{phase}-transaction.json.gz"),
        ),
    )
    print(json.dumps(cert, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True, type=Path)
    cli.add_argument(
        "--phase", choices=("d-propose", "d-verify", "peak-propose", "peak-verify"), required=True
    )
    args = cli.parse_args()
    phase, task = args.phase.split("-")
    model, _, _, bind = original()
    try:
        src = successor(args.root, model, bind, phase)
        (proposal if task == "propose" else publish)(args.root, model, src, phase)
    finally:
        model.parts[3].close()
