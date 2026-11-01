"""Independent readback of old witness times, event ordering and actual published restarts."""

import argparse
import json
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
from check_p65_common_motion import outward
from check_p65_sc2_causal_entries import (
    Curve,
    Ramp,
    director_gap,
    represented_end,
    state_at,
    verify_cluster,
)
from check_p65_sc2_causal_regression import BASE, PLAN
from check_p65_sc2_common_entry import AREA_K, WIDTH, evaluate, geometry
from check_p65_sc2_event_cell import packet_hash
from p65_common_motion import checkpoint, need, unpack
from probe_p65_common_motion import joint_state, saved
from probe_p65_joint_readback import save
from probe_p65_sc2_causal_entries import cluster_input, transaction
from probe_p65_sc2_common_entry import original
from verify_p65_contact_layer_screen import sha

from phymeo.validation.bending_bounds import Interval as I


def peak_witness(root, model):
    # Evaluate the original pressure law on two saved states, without requesting any new force stages.
    proposal = saved(root / "peak-proposal.json.gz")
    attempted = saved(root / "peak-attempt.json.gz")["certificate"]
    tx = saved(root / "peak-transaction.json.gz")
    previous = saved(root / "d-accepted.json")
    packet = np.frombuffer((root / "d.bin").read_bytes(), dtype="<i4")
    q, v, t = unpack(packet, model.signature)
    need(
        not attempted["accepted"]
        and not tx["report"]["accepted"]
        and not tx["substep_states"]
        and np.array_equal(tx["state"]["history"]["motion"], packet)
        and attempted["inlet"] == previous["endpoint"] == packet_hash(packet),
        "peak rejection did not roll back",
    )
    old = proposal["chart"]
    h = Q(proposal["h"])
    curve = Curve(old, model, q, v, h, "chebyshev", 11)
    histories = old["iterations"]
    residuals = []
    for row in histories:
        measured = float(np.max(np.abs(row["evaluated"] - row["acc"])))
        need(measured == row["residual"], "wrong saved sweep residual")
        residuals.append(measured)
    last = histories[-1]
    interp = oracle = position_gap = velocity_gap = Q(0)
    worst = None

    def peak_law(port):
        gap, rate, _ = port
        need(
            gap.high < WIDTH and rate.high < Q(1, 1000), "collocation sample outside active branch"
        )
        return AREA_K * (WIDTH - gap) * (1 - 1000 * rate)

    for k, node in enumerate(old["nodes"]):
        s = Q(float(node))
        x = [I.point(evaluate(p, s)) for p in curve.x]
        u = [I.point(evaluate(p, s)) for p in curve.v]
        trial_x = [I.point(Q(float(z))) for z in last["q"][k]]
        trial_u = [I.point(Q(float(z))) for z in last["v"][k]]
        position_gap = max(
            position_gap, *(abs(a.low - b.low) for a, b in zip(x, trial_x, strict=True))
        )
        velocity_gap = max(
            velocity_gap, *(abs(a.low - b.low) for a, b in zip(u, trial_u, strict=True))
        )
        pubs = geometry(x, u, model)
        trials = geometry(trial_x, trial_u, model)
        for i, (pub, trial) in enumerate(zip(pubs, trials, strict=True)):
            fp = evaluate(curve.ps[i], s)
            fs = Q(float(last["pressures"][k, i]))
            interp = max(interp, abs(fp - fs))
            oracle = max(oracle, (peak_law(trial) - fs).magnitude)
            miss = peak_law(pub) - fp
            if worst is None or miss.low > worst[0]:
                worst = (
                    miss.low,
                    dict(
                        time_s=float(t + s * h),
                        node=k,
                        owner=model.owners[i],
                        published_n=float(fp),
                        law_on_published_n=outward(peak_law(pub)),
                        law_on_trial_n=outward(peak_law(trial)),
                        shortage_n=outward(miss),
                        trial_rate_m_s=outward(trial[1]),
                        published_rate_m_s=outward(pub[1]),
                    ),
                )
    stats = dict(
        sweep_max_acceleration_defects_m_s2=residuals,
        sample_interpolation_error_upper_n=outward(I.point(interp))[1],
        sampled_law_on_own_trial_error_upper_n=outward(I.point(oracle))[1],
        last_trial_to_published_position_max_m=outward(I.point(position_gap))[1],
        last_trial_to_published_velocity_max_m_s=outward(I.point(velocity_gap))[1],
        largest_sample_shortage=worst[1],
        global_force_law_J_error_lower_ns=attempted["local_smooth_absolute_J_error_lower_ns"],
        local_signed_J_defect_ns=attempted["local_smooth_signed_J_error_ns"],
        joint_signed_energy_j=attempted["local_energy_j"],
        rolled_back_to=packet_hash(packet),
        original_h_charges=attempted["charges"],
        candidate_peak=attempted["peak"],
        candidate_peak_accepted=False,
    )
    save(root / "peak-forensics.json", stats)
    print(json.dumps(stats, indent=2))


def remoment(data, h, bracket):
    data["lo"], data["hi"] = bracket
    d = h * (1 - (Q(data["lo"]) + Q(data["hi"])) / 2)
    j = Q(round(Q(data["slope"]) * d * d / 2 * (1 << 112)), 1 << 112)
    data["J"] = str(j)
    data["B"] = str(j * d / 3)


def run(root):
    spec = json.loads(PLAN.read_bytes())
    oldfile = BASE / spec["failure_control"]
    need(
        sha(oldfile, normalize=True) == spec["failure_sha256_lf"], "foreign old witness collection"
    )
    bad = json.loads(oldfile.read_bytes())["first_unverified_successor"]
    model, q0, v0, bind = original()
    try:
        src = cluster_input(model, q0, v0, bind)
        src["step"] = src["h"]
        proposal = saved(root / "cluster-proposal.json.gz")
        accepted = saved(root / "cluster-accepted.json")
        packet = src["input"]
        q, v, t = unpack(packet, model.signature)
        h = src["h"]
        need(
            accepted["accepted"]
            and accepted["inlet"] == bad["source_cp_sha256"] == packet_hash(packet),
            "wrong accepted inlet",
        )
        curve = Curve(proposal["chart"], model, q, v, h, "affine", 1)
        ramps = [Ramp(d, model, h) for d in proposal["ramps"]]
        evidence = []
        for i, r in enumerate(ramps):
            old = next(d for d in bad["owners"] if d["owner"] == model.owners[r.idx])["witness"]
            s = (Q(old["at_s"]) - t) / h
            x, _ = state_at(curve, ramps, s)
            gap = director_gap(x, model, r.idx)
            need(
                0 < s < r.lo and gap.low > Q(1, 500),
                "old sample is not outside on the new trajectory",
            )
            lo = hi = Q(0)
            for prev in ramps[:i]:
                lo += prev.moments_at(r.lo)[0].low
                hi += prev.moments_at(r.hi)[0].high
            need(not i or lo > 0, "earlier owner impulse has not acted at the next birth")
            evidence.append(
                dict(
                    owner=model.owners[r.idx],
                    witness_s=old["at_s"],
                    old_p_n=old["pressure_n"],
                    corrected_p_n=0,
                    gap_m=outward(gap),
                    earlier_new_owner_impulse_ns=outward(I(lo, hi)) if i else [0, 0],
                )
            )
        rejections = {}
        for case in ("merge with consistent moments", "bypass earliest true root"):
            fake = deepcopy(proposal)
            rr = fake["ramps"]
            if case.startswith("merge"):
                remoment(rr[1], h, [rr[0]["lo"], rr[0]["hi"]])
                expect = "overlapping, reordered or duplicate births"
            else:
                chosen = rr[1]
                trial = next(it for it in fake["rounds"][0] if it["index"] == chosen["index"])
                remoment(chosen, h, trial["bracket"])
                fake["ramps"] = [chosen]
                fake["rounds"] = [fake["rounds"][0], []]
                fake["search_cells"] = sum(
                    1 + (64 if p["bracket"] else 0) for p in fake["rounds"][0]
                )
                fake["charges"] = 24
                expect = "not the earliest event"
            forged_ramps = [Ramp(z, model, h) for z in fake["ramps"]]
            x, u = represented_end(curve, model, q, v, forged_ramps, Q(1))
            fake["endpoint"] = packet_hash(checkpoint(x, u, t + h, model.signature))
            result, _ = transaction(model, src, fake, verify_cluster, joint_state(q, v, packet))
            need(
                not result.report.accepted and expect in str(result.report.reason),
                "unexpected forged transaction",
            )
            rejections[case] = str(result.report.reason)
        report = dict(
            origin=packet_hash(packet),
            four_samples=evidence,
            negative_transactions=rejections,
            address_force_evaluations=0,
        )
        save(root / "ordering-readback.json", report)
        print(json.dumps(report, indent=2))
        peak_witness(root, model)
    finally:
        model.parts[3].close()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True, type=Path)
    run(cli.parse_args().root)
