"""Publish one multilevel common-motion peak candidate from the certified SC2 inlet."""

import argparse
import json
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

from check_p65_peak_levels import BASE, PLAN
from check_p65_sc2_event_cell import packet_hash
from p65_common_motion import GRID, checkpoint, endpoint, need, unpack
from probe_p65_common_motion import joint_state, saved
from probe_p65_joint_readback import save
from probe_p65_peak_newton import propose, publish, verify
from probe_p65_peak_slope import linked_moments, window_readback
from probe_p65_sc2_causal_entries import transaction
from probe_p65_sc2_common_entry import original
from run_p65_sc2_step_tail import filehash, successor
from run_p65_sc2_step_tail import proposal as propose_d
from run_p65_sc2_step_tail import publish as publish_d
from run_p65_sc2_step_tail import verify as check_d
from verify_p65_contact_layer_screen import sha

from phymeo.core.scene import fingerprint


def incoming_d_source(model, bind):
    parent = saved(BASE / "p65-sc2-causal-entry-results.json")
    old = BASE.parents[1] / parent["evidence_root"]
    src = successor(old, model, bind, "d")
    need(
        packet_hash(src["input"]) == parent["cluster"]["endpoint"] and src["prefix_charges"] == 126,
        "different four-birth prefix",
    )
    src["paired_axes"] = ((len(model.mass) - 1, -1),)
    return old, src


def paired_d_phase(root, model, bind, phase):
    old, src = incoming_d_source(model, bind)
    if phase == "d-negative":
        claim = saved(old / "d-proposal.json.gz")
        q, v, _ = unpack(src["input"], model.signature)
        src["step"] = src["h"] * Q(claim["end"])
        res, attempt = transaction(model, src, claim, check_d, joint_state(q, v, src["input"]))
        need(
            not res.report.accepted
            and not res.substep_states
            and packet_hash(res.state.history["motion"]) == packet_hash(src["input"])
            and attempt.get("reason") == "different scalar and mapped pressure words",
            "old d not rejected by new gate",
        )
        save(root / "old-d-negative-attempt.json.gz", attempt)
        save(root / "old-d-negative-transaction.json.gz", res)
        print(attempt["reason"])
        return
    src["binding"] = fingerprint(
        (src["binding"], sha(PLAN, normalize=True), "shared platform pressure")
    )
    if phase == "d-propose":
        propose_d(root, model, src, "d")
    else:
        publish_d(root, model, src, "d")


def bound_source(root, model, bind):
    spec = json.loads(PLAN.read_bytes())
    need(spec["schedule"] == [[3, 2], [9, 1], [20, 1]], "schedule mismatch")
    parent = BASE / "p65-sc2-peak-current-tangent-results.json"
    need(
        sha(parent, normalize=True)
        == "b1b76adb08ece99b8db8437b42c3f4af474c4700372621607d9729784b1dbe1b",
        "derivative parent binding",
    )
    proven = json.loads(parent.read_bytes())
    gate = BASE.parents[1] / proven["evidence_root"] / "derivative.json"
    need(
        filehash(gate) == next(r["sha256"] for r in proven["raw"] if r["name"] == gate.name)
        and saved(gate)["accepted"],
        "derivative gate",
    )
    cpath = root / "levels-control.json"
    chk = saved(cpath)
    need(
        chk["accepted"]
        and chk["queries"] == 37
        and chk["proper_pair_words"]
        and set(chk["negatives"]) == {"wrong_map", "missing_budget"},
        "uncertified level transfer",
    )
    src = successor(root, model, bind, "peak")
    d = saved(root / "d-accepted.json")
    old, previous = incoming_d_source(model, bind)
    need(
        d["accepted"]
        and d["inlet"] == packet_hash(previous["input"])
        and d["previous_certificate_sha"] == previous["previous_sha"]
        and src["prefix_charges"] == 288
        and d["endpoint"] != saved(old / "d-link.json")["checkpoint"],
        "unchanged or unbound paired-d repair",
    )
    src.update(step=src["h"], paired_d=d)
    # Translation of the prismatic patch subtracts q_R from each world gap.
    src.update(
        multilevel=True,
        paired_axes=((len(model.mass) - 1, -1),),
        derivative_gate=gate,
        binding=fingerprint(
            (
                sha(PLAN, normalize=True),
                src["binding"],
                src["previous_sha"],
                filehash(cpath),
                "levels/1",
            )
        ),
    )
    return src


def prefix_control(root, model, src):
    q, v, t = unpack(src["input"], model.signature)
    j, b = linked_moments(model, t, d_cert=src["paired_d"])
    mass = Q(float(model.mass[-1]))
    ej = mass * v[-1] + sum(j)
    eb = mass * q[-1] + sum(b)
    save(
        root / "inlet-moments-paired.json",
        dict(
            input=packet_hash(src["input"]),
            time=str(t),
            impulses=list(map(str, j)),
            backward=list(map(str, b)),
            momentum_residual=str(ej),
            position_moment_residual=str(eb),
            exact=ej == eb == 0,
        ),
    )
    need(ej == eb == 0, "incoming published moments do not match the platform")


def negatives(root, model, src):
    rec = saved(root / "accepted.json")
    need(
        rec["accepted"] and rec["endpoint"] == filehash(root / "peak.bin"),
        "negative requires genuine acceptance",
    )
    raw = saved(root / "proposal.json.gz")
    q, v, t = unpack(src["input"], model.signature)
    results = {}
    for name in ("unpaired_platform", "interior_quadratic"):
        data = deepcopy(raw)
        claim = data["chart"]
        if name == "unpaired_platform":
            claim["rows"][0, 0, -1] += 1
        else:
            # P2(2s-1)=(1+3*T2(2s-1))/4. Both uniform time moments vanish.
            node = int(model.faces[0, 0])
            amp = GRID // (1 << 22)
            for index, coeff in ((0, amp), (2, 3 * amp)):
                claim["pressure"][index, 0] += coeff
                claim["rows"][index, 0, 3 * node + 2] += coeff
                claim["rows"][index, 0, -1] -= coeff
        x, u = endpoint(claim, q, v)
        data["endpoint"] = packet_hash(checkpoint(x, u, t + src["h"], model.signature))
        same = data["endpoint"] == rec["endpoint"]
        need(same == (name == "interior_quadratic"), "counterexample endpoint")
        out, attempt = transaction(model, src, data, verify, joint_state(q, v, src["input"]))
        before = packet_hash(out.state.history["motion"])
        save(root / f"{name}-attempt.json.gz", attempt)
        save(root / f"{name}-transaction.json.gz", out)
        need(
            not out.report.accepted and before == rec["inlet"] and not out.substep_states,
            "failed rollback",
        )
        item = dict(same_endpoint_as_accepted=same, returned=before, reason=attempt.get("reason"))
        if same:
            cert = attempt["certificate"]
            lower = cert["local_absolute_J_error_lower_ns"][0]
            need(
                not cert["gates"]["owner_impulse_ns"] and lower > 1e-10,
                "lost between-node counterexample",
            )
            item.update(
                local_lower_ns=lower,
                changed_gates=[k for k, passed in cert["gates"].items() if not passed],
            )
        else:
            need(
                "changed signed platform pressure" in str(attempt.get("reason")),
                "unrelated platform rejection",
            )
        results[name] = item
    save(root / "negative-results.json", results)
    print(results)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True, type=Path)
    cli.add_argument(
        "--phase",
        choices=(
            "d-negative",
            "d-propose",
            "d-verify",
            "propose",
            "verify",
            "readback",
            "negative",
        ),
        required=True,
    )
    args = cli.parse_args()
    model, _, _, bind = original()
    try:
        if args.phase.startswith("d-"):
            paired_d_phase(args.root, model, bind, args.phase)
            raise SystemExit(0)
        src = bound_source(args.root, model, bind)
        if args.phase == "propose":
            prefix_control(args.root, model, src)
            propose(args.root, model, src)
        elif args.phase == "verify":
            publish(args.root, model, src, phase="levels")
        elif args.phase == "readback":
            window_readback(args.root, model, src, phase="levels", mode="levels-newton")
        else:
            negatives(args.root, model, src)
    finally:
        model.parts[3].close()
