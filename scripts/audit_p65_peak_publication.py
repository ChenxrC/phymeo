"""Read the four connected SC2 cells as one original step, using their actual force words."""

import argparse
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
from p65_common_motion import GRID, need, unpack
from probe_p65_common_motion import saved
from probe_p65_joint_readback import save
from probe_p65_sc2_common_entry import original
from run_p65_sc2_step_tail import filehash

BASE = Path(__file__).resolve().parents[1] / "docs/benchmarks"


def time_weights(count, end, h, affine):
    if affine:
        need(count == 2, "affine data")
        a = (end - end**2 / 2, end**2 / 2)
        b = (end**2 / 2 - end**3 / 6, end**3 / 6)
    else:
        z = 2 * end - 1
        values = [Q(1), z]
        for _ in range(count + 1):
            values.append(2 * z * values[-1] - values[-2])

        def delta(i):
            return values[i] - (-1) ** i

        a = [end, end * (end - 1)]
        a.extend(
            delta(j + 1) / (4 * (j + 1)) - delta(j - 1) / (4 * (j - 1)) for j in range(2, count + 1)
        )
        b = [(end - Q(1, 2)) * a[j] - (a[j + 1] + a[abs(j - 1)]) / 4 for j in range(count)]
        a = a[:count]
    return [h * x / GRID for x in a], [h * h * x / GRID for x in b]


def public_state(tx, packet, q, v):
    need(
        tx["report"]["accepted"] and len(tx["substep_states"]) == 1,
        "actual substep did not publish",
    )
    for state in (tx["state"], tx["substep_states"][0]):
        need(
            np.array_equal(packet, state["history"]["motion"])
            and set(state["domains"]) == {"cloth", "robot"},
            "state packet or domains",
        )
        c, r = state["domains"]["cloth"], state["domains"]["robot"]
        for actual, expected in (
            (c["position"], np.asarray(q[:12], float).reshape(4, 3)),
            (c["velocity"], np.asarray(v[:12], float).reshape(4, 3)),
            (r["q"], np.asarray(q[12:], float)),
            (r["v"], np.asarray(v[12:], float)),
        ):
            need(
                actual.dtype == expected.dtype
                and actual.shape == expected.shape
                and actual.tobytes() == expected.tobytes(),
                "public state differs from its rounded decoded packet",
            )


def check(root, model, q0, v0):
    parent = saved(BASE / "p65-sc2-causal-entry-results.json")
    first = saved(BASE / "p65-sc2-common-motion-entry-results.json")
    ep = BASE.parents[1] / first["raw_directory"]
    old = BASE.parents[1] / parent["evidence_root"]
    prior = saved(ep / "proposal.json.gz")
    d = saved(root / "d-accepted.json")
    peak = saved(root / "accepted.json")
    need(
        parent["cluster"]["inlet"] == first["entry"]["checkpoint_sha256"]
        and d["inlet"] == parent["cluster"]["endpoint"]
        and peak["inlet"] == d["endpoint"],
        "broken packet chain",
    )
    c = saved(old / "cluster-proposal.json.gz")
    dp = saved(root / "d-proposal.json.gz")
    p = saved(root / "proposal.json.gz")
    orig_files = {row["name"]: row["sha256"] for row in first["raw"]}
    cluster_files = {row["name"]: row["sha256"] for row in parent["raw"]}
    need(
        filehash(ep / "proposal.json.gz") == orig_files["proposal.json.gz"]
        and filehash(old / "cluster-proposal.json.gz") == cluster_files["cluster-proposal.json.gz"],
        "different prefix data",
    )
    segments = [
        (
            "c",
            prior["free"],
            Q(prior["end"]),
            [dict(index=prior["first"], **prior["ramp"])],
            ep / "bridge.bin",
            first["entry"]["checkpoint_sha256"],
            None,
        ),
        (
            "cluster",
            c["chart"],
            Q(1),
            c["ramps"],
            old / "cluster.bin",
            parent["cluster"]["endpoint"],
            old / "cluster-transaction.json.gz",
        ),
        (
            "d",
            dp["chart"],
            Q(dp["end"]),
            dp["ramps"],
            root / "d.bin",
            d["endpoint"],
            root / "d-transaction.json.gz",
        ),
        (
            "peak",
            p["chart"],
            Q(1),
            [],
            root / "peak.bin",
            peak["endpoint"],
            root / "levels-transaction.json.gz",
        ),
    ]
    m = list(map(lambda s: Q(float(s)), model.mass))
    previous = Q(0)
    # Rows 0..5 are mapped owner forces, row 6 is the material field; external is exactly zero.
    j = np.full((7, 13), Q(0))
    b = j.copy()
    pj = np.full(6, Q(0))
    pb = pj.copy()
    checks = []
    for label, chart, end, ramps, path, signature, tx in segments:
        need(
            filehash(path) == signature
            and chart["grid"] == GRID
            and np.array_equal(model.mass, chart["mass"])
            and not np.any(chart["external"]),
            "input words",
        )
        packet = np.frombuffer(path.read_bytes(), dtype="<i4")
        q, v, T = unpack(packet, model.signature)
        h = Q(chart["h"])
        dt = h * end
        need(previous + dt == T and tuple(chart["owners"]) == model.owners, "noncontiguous motion")
        aj, ab = time_weights(len(chart["material"]), end, h, label == "cluster")
        forces = np.concatenate((chart["rows"], chart["material"][:, None, :]), axis=1)
        fj, fb = [np.tensordot(weights, forces.astype(object), axes=1) for weights in (aj, ab)]
        sj, sb = [
            np.tensordot(weights, chart["pressure"].astype(object), axes=1) for weights in (aj, ab)
        ]
        for ramp in ramps:
            k = ramp["index"]
            row = np.array(list(map(Q, ramp["map"])), object)
            jj, bb = Q(ramp["J"]), Q(ramp["B"])
            fj[k] += jj * row
            fb[k] += bb * row
            sj[k] += jj
            sb[k] += bb
        need(
            all(fj[i, -1] == -sj[i] and fb[i, -1] == -sb[i] for i in range(6)),
            "unpaired local platform moments",
        )
        b += dt * j + fb
        j += fj
        pb += dt * pj + sb
        pj += sj
        sumj = j.sum(axis=0)
        sumb = b.sum(axis=0)
        need(
            all(
                v0[i] + sumj[i] / m[i] == v[i] and q0[i] + T * v0[i] + sumb[i] / m[i] == q[i]
                for i in range(13)
            ),
            "coordinate state not determined by original entry and all force moments",
        )
        if tx is not None:
            public_state(saved(tx), packet, q, v)
        checks.append(
            dict(
                segment=label,
                stop_s=str(T),
                packet=signature,
                coordinates=13,
                public_state_and_substep_checked=tx is not None,
            )
        )
        previous = T
    aggregate = saved(root / "whole-step.json")
    need(
        previous == Q(1, 64)
        and list(map(Q, aggregate["exact_owner_J_ns"])) == pj.tolist()
        and list(map(Q, aggregate["exact_owner_B_ns2"])) == pb.tolist(),
        "independent full-window scalar moments",
    )
    evidence = dict(
        accepted=True,
        ordered_segments=checks,
        coordinate_count=13,
        material_and_six_mapped_fields=True,
        source_momentum_ns=[[str(x) for x in row] for row in j],
        source_backward_kg_m=[[str(x) for x in row] for row in b],
        scalar_owner_J_ns=list(map(str, pj)),
        scalar_owner_B_ns2=list(map(str, pb)),
        exact_original_entry_velocity=True,
        all_published_domain_arrays_byte_equal=True,
        new_force_queries=0,
    )
    save(root / "published-motion.json", evidence)
    print(
        "Four continuous cells; all 13 exact moment balances; all three public states and substeps match their packets."
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True, type=Path)
    model, q, v, _ = original()
    try:
        check(cli.parse_args().root, model, q, v)
    finally:
        model.parts[3].close()
