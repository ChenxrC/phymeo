"""Inspect the earliest uncertified SC2 successor from saved bytes, without advancing dynamics."""

import argparse
import json
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path

import numpy as np
from audit_p65_sc2_event_moments import direct_gap
from check_p65_common_motion import integral, outward, scale
from check_p65_sc2_common_entry import WIDTH, evaluate, geometry
from check_p65_sc2_event_regression import BASE, PLAN
from p65_common_motion import need, unpack
from probe_p65_common_motion import saved
from probe_p65_joint_readback import save
from probe_p65_sc2_common_entry import original
from trace_p65_sc2_common_window import cubic_force_curves, poly_range

from phymeo.validation.bending_bounds import Interval as I


def inspect(root):
    cfg = json.loads(PLAN.read_bytes())
    previous = json.loads((BASE / cfg["source"]).read_bytes())
    store = BASE.parents[1] / previous["raw_directory"]
    p = store / "sc2-step-0.json.gz"
    need(
        sha256(p.read_bytes()).hexdigest() == previous["window"]["eight_stage_files_sha256"][0],
        "foreign first successor",
    )
    frames = saved(p)
    model, _, _, binding = original()
    try:
        cp = (store / "bridge.bin").read_bytes()
        need(
            sha256(cp).hexdigest() == frames["in_sha"] == previous["entry"]["checkpoint_sha256"]
            and frames["binding"] == binding
            and frames["owners"] == list(model.owners),
            "foreign accepted boundary",
        )
        q, v, t = unpack(np.frombuffer(cp, dtype="<i4"), model.signature)
        dt = Q(frames["dt"])
        f, ps = cubic_force_curves(frames["cells"][0], model)
        vel = [
            integral(scale(z, dt / Q(float(m))), u)
            for z, m, u in zip(f, model.mass, v, strict=True)
        ]
        pos = [integral(scale(z, dt), x) for z, x in zip(vel, q, strict=True)]
        box = geometry(
            [poly_range(z, Q(0), Q(1)) for z in pos],
            [poly_range(z, Q(0), Q(1)) for z in vel],
            model,
        )
        rows = []
        for i, owner in enumerate(model.owners):
            initial = direct_gap(pos, 0, i, model)
            if initial.high < WIDTH:
                rows.append(dict(owner=owner, branch="already inside"))
                continue
            need(initial.low > WIDTH and box[i][1].high < 0, "outside or closing branch unresolved")
            last = direct_gap(pos, 1, i, model)
            low, high = Q(0), Q(1)
            if last.high < WIDTH:
                for _ in range(32):
                    s = (low + high) / 2
                    g = direct_gap(pos, s, i, model)
                    if g.low > WIDTH:
                        low = s
                    else:
                        need(g.high < WIDTH, "ambiguous side")
                        high = s
                outside_until = low
                event = outward(I(t + dt * low, t + dt * high))
            else:
                need(last.low > WIDTH, "unresolved terminal side")
                outside_until = Q(1)
                event = None
            coeff = ps[i]
            witness = None
            if any(coeff):
                k = next(j for j, c in enumerate(coeff) if c)
                tail = sum(abs(z) for z in coeff[k + 1 :])
                w = min(outside_until / 2, abs(coeff[k]) / (2 * tail) if tail else Q(1, 2))
                need(
                    w > 0 and direct_gap(pos, w, i, model).low > WIDTH,
                    "sample not strictly outside",
                )
                force = evaluate(coeff, w)
                need(
                    force and abs(force) >= abs(coeff[k]) * w**k / 2, "unproved forbidden pressure"
                )
                witness = dict(
                    at_s=float(t + dt * w),
                    pressure_n=float(force),
                    interval_open_s=outward(I(t, t + dt * w)),
                    first_nonzero_power=k,
                )
            rows.append(
                dict(
                    owner=owner,
                    branch="outside at inlet",
                    next_entry_s=event,
                    identically_zero_outside=witness is None,
                    witness=witness,
                )
            )
        result = dict(
            source_cp_sha256=frames["in_sha"],
            source_trace_sha256=previous["window"]["eight_stage_files_sha256"][0],
            cell_index=0,
            start_s=str(t),
            end_s=str(t + dt),
            model=model.signature,
            owner_force_evaluations=0,
            owners=rows,
            forbidden_count=sum(r.get("witness") is not None for r in rows),
        )
        save(root / "first-successor.json", result)
        print(json.dumps(result, indent=2))
    finally:
        model.parts[3].close()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True, type=Path)
    inspect(cli.parse_args().root)
