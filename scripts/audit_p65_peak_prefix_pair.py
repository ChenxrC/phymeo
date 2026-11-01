"""Locate the first scalar versus mapped platform moment mismatch on retained SC2 cells."""

import argparse
import json
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
from check_p65_common_motion import add, times
from check_p65_sc2_causal_entries import Curve
from check_p65_sc2_event_cell import moment
from p65_common_motion import need, unpack
from probe_p65_common_motion import saved
from probe_p65_joint_readback import save
from probe_p65_sc2_common_entry import original
from run_p65_sc2_step_tail import filehash

BASE = Path(__file__).resolve().parents[1] / "docs/benchmarks"


def audit(root):
    parent = saved(BASE / "p65-sc2-causal-entry-results.json")
    entry = saved(BASE / "p65-sc2-common-motion-entry-results.json")
    evidence = BASE.parents[1] / parent["evidence_root"]
    bound = {r["name"]: r["sha256"] for r in parent["raw"]}
    ep = BASE.parents[1] / entry["raw_directory"] / "bridge.bin"
    need(filehash(ep) == entry["entry"]["checkpoint_sha256"], "bridge binding")
    model, _, _, _ = original()
    try:
        mass = Q(float(model.mass[-1]))
        q, v, t = unpack(np.frombuffer(ep.read_bytes(), dtype="<i4"), model.signature)
        ej = mass * v[-1] + Q(entry["entry"]["J_ns"])
        eb = mass * q[-1] + Q(entry["entry"]["B_ns2"])
        rows = [dict(phase="bridge", packet=filehash(ep), momentum=str(ej), position=str(eb))]
        for phase in ("cluster", "d"):
            files = [evidence / (phase + "-proposal.json.gz"), evidence / (phase + ".bin")]
            need(
                all(filehash(f) == bound[f.name] for f in files), "retained packet/proposal binding"
            )
            data = saved(files[0])
            out = np.frombuffer(files[1].read_bytes(), dtype="<i4")
            x, u, T = unpack(out, model.signature)
            h = Q(data["chart"]["h"])
            end = Q(data.get("end", "1"))
            need(t + h * end == T, "recorded segment duration")
            mode, degree = ("affine", 1) if phase == "cluster" else ("chebyshev", 8)
            curve = Curve(data["chart"], model, q, v, h, mode, degree)
            cert = parent[phase]
            sj = list(map(Q, cert["local_J_ns"]))
            sb = list(map(Q, cert["local_B_ns2"]))
            calc_j = []
            calc_b = []
            aliases = []
            for i in range(6):
                ramps = [z for z in data["ramps"] if z["index"] == i]
                need(all(Q(z["map"][-1]) == -1 for z in ramps), "different ramp map")
                calc_j.append(h * moment(curve.ps[i], 0, end) + sum(Q(z["J"]) for z in ramps))
                calc_b.append(
                    h * h * moment(times((end, -1), curve.ps[i]), 0, end)
                    + sum(Q(z["B"]) for z in ramps)
                )
                aliases.append(
                    (data["chart"]["rows"][:, i, -1] + data["chart"]["pressure"][:, i]).tolist()
                )
            dv = mass * (u[-1] - v[-1]) + sum(sj)
            dx = mass * (x[-1] - q[-1] - (T - t) * v[-1]) + sum(sb)
            combined = curve.mat[-1]
            for i in range(6):
                combined = add(combined, add(curve.ps[i], curve.rows[i][-1]))
            map_j = h * moment(combined, 0, end)
            map_b = h * h * moment(times((end, -1), combined), 0, end)
            need(
                dv == map_j + sum(sj) - sum(calc_j) and dx == map_b + sum(sb) - sum(calc_b),
                "unexplained local mismatch",
            )
            rows.append(
                dict(
                    phase=phase,
                    packet=cert["endpoint"],
                    old_time=str(t),
                    time=str(T),
                    local_platform_momentum=str(dv),
                    local_platform_position=str(dx),
                    integrated_map_momentum=str(map_j),
                    integrated_map_position=str(map_b),
                    owner_scalar_J_minus_record=list(map(str, (i - j for i, j in zip(calc_j, sj)))),
                    owner_scalar_B_minus_record=list(map(str, (i - j for i, j in zip(calc_b, sb)))),
                    pressure_plus_mapped_words=aliases,
                )
            )
            eb += dx + (T - t) * ej
            ej += dv
            q, v, t = x, u, T
        failed = saved(root / "inlet-moments.json")
        need(
            Q(failed["momentum_residual"]) == ej and Q(failed["position_moment_residual"]) == eb,
            "original failure not reproduced",
        )
        save(
            root / "prefix-forensics.json",
            dict(segments=rows, total_momentum=str(ej), total_position=str(eb), dynamics_calls=0),
        )
        print(json.dumps(rows, indent=2))
    finally:
        model.parts[3].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    audit(parser.parse_args().root)
