"""Falsify an endpoint-preserving display fix using only the saved SC2 event cell."""

import argparse
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path

import numpy as np
from check_p65_common_motion import integral, outward, scale, times
from check_p65_sc2_common_entry import WIDTH, difference, evaluate, geometry, state_polynomials
from p65_common_motion import need, unpack
from probe_p65_common_motion import saved
from probe_p65_joint_readback import save
from probe_p65_sc2_common_entry import original
from trace_p65_sc2_common_window import H, cubic_force_curves, poly_range, step_moments

from phymeo.validation.bending_bounds import Interval as I
from phymeo.validation.bending_bounds import cross, dot, sqrt_interval


def direct_gap(polys, s, owner, model):
    # Unsquared director evaluation from the oriented area, independent of the root proposer.
    x = [evaluate(p, s) for p in polys]
    ids = model.faces[owner // 3]
    pts = [x[3 * int(n) : 3 * int(n) + 3] for n in ids]
    c = cross(difference(pts[1], pts[0]), difference(pts[2], pts[0]))
    need(c[2] > 0, "unsigned director ambiguity")
    length = sqrt_interval(I.point(dot(c, c)), 160)
    rho = Q(float(model.rho))
    return I.point(pts[owner % 3][2] - x[12] + rho) - I.point(rho * c[2]) / length


def check_first(root, model, q, v, bind):
    packet = saved(root / "proposal.json.gz")
    cert = saved(root / "entry.json")
    need(
        sha256((root / "proposal.json.gz").read_bytes()).hexdigest() == cert["proposal_sha256"]
        and bind == packet["binding"] == cert["input_binding"],
        "foreign entry",
    )
    _, pos, _ = state_polynomials(packet["free"], q, v, model, H)
    brackets = [tuple(map(Q, r)) for r in packet["roots"]]
    need(
        len(brackets) == len(model.owners) and cert["latest_closing_rate_upper"] < 0,
        "owner completeness or monotonicity",
    )
    for i, (lo, hi) in enumerate(brackets):
        need(
            hi - lo <= Q(1, 2**32)
            and direct_gap(pos, lo, i, model).low > WIDTH
            and direct_gap(pos, hi, i, model).high < WIDTH,
            "unsquared director does not enclose the event",
        )
    first = packet["first"]
    need(
        model.owners[first] == cert["first"]
        and all(brackets[first][1] < lo for i, (lo, _) in enumerate(brackets) if i != first),
        "unsquared earliest event",
    )


def run(root):
    model, q0, v0, bind = original()
    try:
        check_first(root, model, q0, v0, bind)
        window = saved(root / "window.json")
        need(
            window["entry_sha"] == sha256((root / "entry.json").read_bytes()).hexdigest()
            and window["input_binding"] == bind,
            "foreign window",
        )
        tf, witness = window["first_spurious_or_negative"]
        tprobe = Q(tf)
        i = model.owners.index(witness["owner"])
        k = int(tprobe // H)
        path = root / f"sc2-step-{k}.json.gz"
        need(
            sha256(path.read_bytes()).hexdigest() == window["eight_stage_files_sha256"][k],
            "wrong trace",
        )
        trace = saved(path)
        packet = (root / ("bridge.bin" if k == 0 else f"sc2-step-{k - 1}.bin")).read_bytes()
        need(sha256(packet).hexdigest() == trace["in_sha"], "wrong checkpoint")
        q, v, start = unpack(np.frombuffer(packet, dtype="<i4"), model.signature)
        dt = Q(trace["dt"])
        cell = int((tprobe - start) // dt)
        for raw in trace["cells"][:cell]:
            q, v = step_moments(raw, model, q, v, dt)
        start += cell * dt
        raw = trace["cells"][cell]
        need(
            all(Q(float(f["rows"][i, -1])) == -Q(float(f["pressure"][i])) for f in raw),
            "paired robot row is not the negative of the same owner pressure",
        )
        fields, pressures = cubic_force_curves(raw, model)
        vel = [
            integral(scale(f, dt / Q(float(m))), u)
            for f, m, u in zip(fields, model.mass, v, strict=True)
        ]
        pos = [integral(scale(z, dt), x) for z, x in zip(vel, q, strict=True)]
        whole = geometry(
            [poly_range(p, Q(0), Q(1)) for p in pos],
            [poly_range(p, Q(0), Q(1)) for p in vel],
            model,
        )
        need(
            whole[i][1].high < 0
            and direct_gap(pos, Q(0), i, model).low > WIDTH
            and direct_gap(pos, Q(1), i, model).high < WIDTH,
            "witness cell does not contain one strict entry",
        )
        left, right = Q(0), Q(1)
        for _ in range(32):
            middle = (left + right) / 2
            gap = direct_gap(pos, middle, i, model)
            if gap.low > WIDTH:
                left = middle
            else:
                need(gap.high < WIDTH, "unresolved root sign")
                right = middle
        pr = pressures[i]
        # On [0,root) the true pressure is exactly zero. Bound the signed moments of
        # the published field that would be removed, with the same cell-end time origin.
        roots = I(left, right)
        j = dt * evaluate(integral(pr), roots)
        back = dt**2 * evaluate(integral(times((1, -1), pr)), roots)
        need(
            not j.low <= 0 <= j.high and not back.low <= 0 <= back.high,
            "two temporal moments unresolved",
        )
        m = Q(float(model.mass[-1]))
        result = dict(
            owner=model.owners[i],
            step=k,
            cell_zero_based=cell,
            cell_start_s=str(start),
            cell_end_s=str(start + dt),
            earlier_first_unsquared_root_pairs=6,
            early_sample_s=tf,
            early_sample_n=witness["p_n"],
            empty_force_until_s=outward(I(start + dt * left, start + dt * right)),
            resolved_signed_early_J_ns=outward(j),
            resolved_signed_early_B_ns2=outward(back),
            delta_platform_v_if_only_removed_m_s=outward(j / m),
            delta_platform_q_if_only_removed_m=outward(back / m),
            unchanged_motion_cannot_publish_clipped_force=True,
            root_depth=32,
            trace_sha256=window["eight_stage_files_sha256"][k],
            model=model.signature,
        )
        save(root / "event-moment-counterexample.json", result)
        print(result)
    finally:
        model.parts[3].close()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, required=True)
    run(cli.parse_args().root)
