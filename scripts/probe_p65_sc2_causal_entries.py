"""Advance the earliest neighboring SC2 births from the last accepted physical checkpoint."""

import argparse
import json
from copy import deepcopy
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path

import numpy as np
from check_p65_sc2_causal_entries import (
    DEPTH,
    Curve,
    H,
    Ramp,
    arc_horizon,
    director_gap,
    first_totals,
    represented_end,
    state_at,
    verify_cluster,
)
from check_p65_sc2_causal_regression import BASE, PLAN
from check_p65_sc2_event_cell import packet_hash
from p65_common_motion import GRID, checkpoint, need, quantized, unpack
from probe_p65_common_motion import PathProgram, joint_state, saved
from probe_p65_joint_readback import save
from probe_p65_sc2_common_entry import original
from probe_p65_sc2_event_cell import CellWorkspace
from verify_p65_contact_layer_screen import sha

import phymeo as ph
from phymeo.core.capabilities import SolverFailure
from phymeo.core.scene import fingerprint
from phymeo.runtime.scene_stepper import SceneStepper


def binding(model, original_bind, packet, tag):
    return fingerprint((sha(PLAN, normalize=True), original_bind, model.signature, packet, tag))


def cluster_input(model, q0, v0, original_bind):
    spec = json.loads(PLAN.read_bytes())
    p = BASE / spec["initial"]
    need(sha(p, normalize=True) == spec["initial_sha256_lf"], "changed accepted entry record")
    old = json.loads(p.read_bytes())
    entry = old["entry"]
    cp = (BASE.parents[1] / old["raw_directory"] / "bridge.bin").read_bytes()
    need(
        sha256(cp).hexdigest() == entry["checkpoint_sha256"]
        and original_bind == entry["input_binding"],
        "unverified origin packet",
    )
    packet = np.frombuffer(cp, dtype="<i4").copy()
    q, v, t = unpack(packet, model.signature)
    h, mask = arc_horizon(model, q, v)
    need(t == Q(entry["bridge_end_s"]) and t + h <= H, "unaligned inlet time")
    return dict(
        input=packet,
        binding=binding(model, original_bind, packet, "cluster"),
        h=h,
        mask=mask,
        prefix_charges=entry["owner_charges"],
        prefix_cells=entry["time_cells"] + 1,
        totals=first_totals(entry, model, packet, q0, v0),
        origin_sha=entry["checkpoint_sha256"],
    )


class Meter:
    def __init__(self, model, source):
        self.model = model
        self.limit = 512 - source["prefix_charges"]
        self.count = 0

    def take(self):
        need(self.count + 6 <= self.limit, "original owner-evaluation limit")
        self.count += 6

    def maps(self, q, v):
        self.take()
        return self.model.maps(q, v)

    def forces(self, q, v):
        self.take()
        return self.model.forces(q, v)


def affine_background(model, q, v, h, mask, meter, tag):
    q0 = np.asarray(q, float)
    v0 = np.asarray(v, float)
    frames = []
    x, u = q0, v0
    for _ in range(2):
        true = meter.forces(x, u)
        rows = np.where(np.asarray(mask)[:, None], true.rows, 0.0)
        p = np.where(mask, true.pressures, 0.0)
        frames.append((true.material, rows, p))
        acc = (true.material + rows.sum(axis=0) + model.external) / model.mass
        x = q0 + float(h) * v0 + float(h) ** 2 * acc / 2
        u = v0 + float(h) * acc
    mat, rows, ps = (quantized(np.stack(f)) for f in zip(*frames, strict=True))
    return dict(
        binding=tag,
        h=float(h),
        grid=GRID,
        force_calls=2,
        owner_charges=12,
        owners=list(model.owners),
        mass=model.mass,
        external=model.external,
        material=mat,
        rows=rows,
        pressure=ps,
    )


def isolate_round(curve, model, ramps, remaining):
    # Every trial uses the common motion with all preceding births already present.
    cache = {}

    def at(s):
        if s not in cache:
            cache[s] = state_at(curve, ramps, s)[0]
        return cache[s]

    ring = []
    hits = []
    work = 0
    for idx in sorted(remaining):
        work += 1
        g = director_gap(at(Q(1)), model, idx)
        if g.low > Q(1, 500):
            ring.append(dict(index=idx, bracket=None))
            continue
        need(g.high < Q(1, 500), "ambiguous search endpoint")
        lo, hi = Q(0), Q(1)
        for _ in range(DEPTH):
            mid = (lo + hi) / 2
            gap = director_gap(at(mid), model, idx)
            work += 2
            if gap.low > Q(1, 500):
                lo = mid
            else:
                need(gap.high < Q(1, 500), "ambiguous close roots")
                hi = mid
        hits.append((lo, hi, idx))
        ring.append(dict(index=idx, bracket=[str(lo), str(hi)]))
    hits.sort()
    if hits:
        need(len(hits) == 1 or hits[0][1] < hits[1][0], "earliest intervals overlap")
    return ring, (hits[0] if hits else None), work


def make_ramp(model, curve, old, hit, meter, end):
    lo, hi, idx = hit
    mid = (lo + hi) / 2
    x, v = state_at(curve, old, mid)

    def approx(boxes):
        return np.array([float((z.low + z.high) / 2) for z in boxes])

    maps, _, rates = meter.maps(approx(x), approx(v))
    speed = -rates[idx]
    need(speed > 0, "not an approaching event")
    row = [Q(round(z * (1 << 48)), 1 << 48) for z in maps[idx]]
    slope = Q(round((500 / 6) * speed * (1 + 1000 * speed) * (1 << 32)), 1 << 32)
    dm = curve.h * (end - mid)
    j = Q(round(slope * dm**2 / 2 * (1 << 112)), 1 << 112)
    return dict(
        owner=model.owners[idx],
        index=idx,
        lo=str(lo),
        hi=str(hi),
        before=0,
        map=list(map(str, row)),
        slope=str(slope),
        J=str(j),
        B=str(j * dm / 3),
    )


def propose_cluster(root, model, source):
    h = source["h"]
    q, v, t = unpack(source["input"], model.signature)
    meter = Meter(model, source)
    _, gap, rate = meter.maps(np.asarray(q, float), np.asarray(v, float))
    mask = [bool(g < 0.002 and z < 0.001) for g, z in zip(gap, rate, strict=True)]
    need(mask == source["mask"], "inlet branch disagrees with geometry")
    claim = affine_background(model, q, v, h, mask, meter, source["binding"])
    curve = Curve(claim, model, q, v, h, "affine", 1)
    remaining = set(i for i, on in enumerate(mask) if not on)
    ramps = []
    data = []
    rings = []
    count = 0
    while True:
        ring, hit, used = isolate_round(curve, model, ramps, remaining)
        rings.append(ring)
        count += used
        need(count + source["prefix_cells"] <= 4096, "original created cell limit")
        if hit is None:
            break
        wire = make_ramp(model, curve, ramps, hit, meter, Q(1))
        ramps.append(Ramp(wire, model, h))
        data.append(wire)
        remaining.remove(hit[2])
    x, u = represented_end(curve, model, q, v, ramps, Q(1))
    out = checkpoint(x, u, t + h, model.signature)
    save(
        root / "cluster-proposal.json.gz",
        dict(
            binding=source["binding"],
            input_sha=packet_hash(source["input"]),
            h=str(h),
            active=mask,
            chart=claim,
            ramps=data,
            rounds=rings,
            charges=meter.count,
            search_cells=count,
            endpoint=packet_hash(out),
        ),
    )
    print(
        "ordered proposal",
        [r["owner"] for r in data],
        "new owner calls",
        meter.count,
        "cells",
        count,
    )


class CausalProgram(PathProgram):
    def __init__(self, model, source, data, check):
        super().__init__(model)
        self.source = source
        self.data = data
        self.check = check

    def problem_signature(self):
        return dict(
            model=self.model.signature, original=self.source["binding"], law="SC2 causal birth v1"
        )

    def create_workspace(self, scene, state, inputs, h, settings, context):
        need(
            scene.identity == self.model.parts[0].identity
            and context.backend == "numpy"
            and fingerprint(inputs) == fingerprint(self.model.parts[2])
            and Q(h) == self.source["step"],
            "scene or original inputs",
        )
        return CausalWorkspace(self, state, inputs, h)


class CausalWorkspace(CellWorkspace):
    def position_sweep(self):
        if self.receipt is not None:
            return
        try:
            out, rec = self.program.check(self.model, self.program.source, self.program.data)
            self.program.attempt["certificate"] = rec
            need(
                rec["accepted"],
                "failed physical gates: " + ", ".join(k for k, v in rec["gates"].items() if not v),
            )
            q, v, _ = unpack(out, self.model.signature)
            self.out = joint_state(q, v, out)
            self.receipt = rec
        except (ValueError, ArithmeticError) as exc:
            self.program.attempt["reason"] = str(exc)
            raise SolverFailure(str(exc)) from exc


def transaction(model, source, data, check, state):
    p = CausalProgram(model, source, data, check)
    settings = ph.StepSettings(
        substeps=1,
        position_iterations=1,
        velocity_iterations=1,
        tolerance=1e-8,
        record_substeps=True,
    )
    before = fingerprint(state)
    with SceneStepper(
        model.parts[0], p, settings, ph.BackendContext("numpy"), model.parts[3].registry
    ) as rt:
        result = rt.step(state, model.parts[2], float(source["step"]))
    need(fingerprint(state) == before, "incoming state mutated")
    if not result.report.accepted:
        need(
            fingerprint(result.state) == before and not result.substep_states,
            "failed transaction leaked state",
        )
    return result, p.attempt


def checked_publish(root, phase, model, source, data, checker):
    q, v, _ = unpack(source["input"], model.signature)
    state = joint_state(q, v, source["input"])
    result, info = transaction(model, source, data, checker, state)
    save(root / f"{phase}-attempt.json.gz", info)
    save(root / f"{phase}-transaction.json.gz", result)
    if not result.report.accepted:
        print(info)
        raise ValueError("public step rejected: " + str(result.report.reason))
    rec = info["certificate"]
    need(
        packet_hash(result.state.history["motion"]) == rec["endpoint"],
        "different published endpoint",
    )
    return state, rec, result.state.history["motion"]


def check_cluster(root, model, source):
    data = saved(root / "cluster-proposal.json.gz")
    source["step"] = source["h"]
    state, rec, out = checked_publish(root, "cluster", model, source, data, verify_cluster)
    old = source["input"]
    q, v, t = unpack(old, model.signature)
    checks = {}
    for k in (
        "lost-inlet",
        "missing-owner",
        "merged",
        "exchanged",
        "early",
        "impulse",
        "backward",
        "end",
        "omitted-event",
    ):
        fake = deepcopy(data)
        incoming = state
        ramps = fake["ramps"]
        if k == "lost-inlet":
            u = list(v)
            u[2] = Q(0)
            need(u != v, "moving inlet")
            incoming = joint_state(q, u, old)
        elif k == "missing-owner":
            fake["chart"]["owners"] = fake["chart"]["owners"][:-1]
        elif k == "merged":
            ramps[1]["lo"], ramps[1]["hi"] = ramps[0]["lo"], ramps[0]["hi"]
        elif k == "exchanged":
            ramps[:2] = ramps[1::-1]
        elif k == "early":
            ramps[0]["before"] = 1
        elif k in ("impulse", "backward"):
            key = "J" if k == "impulse" else "B"
            ramps[0][key] = str(2 * Q(ramps[0][key]))
        elif k == "end":
            fake["endpoint"] = packet_hash(old)
        else:
            ramps.pop()
            fake["charges"] -= 6
            fake["rounds"].pop()
        failure, _ = transaction(model, source, fake, verify_cluster, incoming)
        need(not failure.report.accepted, "accepted forged " + k)
        checks[k] = str(failure.report.reason)
    rec["rejected_transactions"] = checks
    with (root / "cluster.bin").open("xb") as fp:
        fp.write(out.tobytes())
    save(root / "cluster-accepted.json", rec)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True, type=Path)
    cli.add_argument("--phase", required=True, choices=("cluster-propose", "cluster-verify"))
    args = cli.parse_args()
    model, q, v, bind = original()
    try:
        src = cluster_input(model, q, v, bind)
        (propose_cluster if args.phase == "cluster-propose" else check_cluster)(
            args.root, model, src
        )
    finally:
        model.parts[3].close()
