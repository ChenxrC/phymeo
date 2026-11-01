"""One event-aligned HOST replacement for the first saved SC2 forbidden-load cell."""

import argparse
import json
from copy import deepcopy
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path

import numpy as np
from audit_p65_sc2_event_moments import direct_gap
from check_p65_sc2_common_entry import WIDTH, squared_guard
from check_p65_sc2_event_cell import Chart, packet_hash, verify
from check_p65_sc2_event_regression import BASE, PLAN
from p65_common_motion import Forces, checkpoint, endpoint, fit, need, unpack
from probe_p65_common_motion import (
    ClothPlatform,
    PathProgram,
    Workspace,
    coordinates,
    joint_state,
    saved,
)
from probe_p65_joint_readback import save
from probe_p65_sc2_common_entry import original
from trace_p65_sc2_common_window import step_moments
from verify_p65_contact_layer_screen import sha

import phymeo as ph
from phymeo.core.capabilities import SolverFailure
from phymeo.core.scene import StageMetrics, fingerprint
from phymeo.runtime.scene_stepper import SceneStepper


def inlet(model, bind):
    protocol = json.loads(PLAN.read_bytes())
    source = BASE / protocol["source"]
    need(sha(source, normalize=True) == protocol["source_sha256_lf"], "changed reference")
    old = json.loads(source.read_bytes())
    case = old["counterexample"]
    root = BASE.parents[1] / old["raw_directory"]
    path = root / f"sc2-step-{case['step']}.json.gz"
    need(sha256(path.read_bytes()).hexdigest() == case["trace_sha256"], "foreign trace")
    ref = saved(path)
    initial = (
        root / ("bridge.bin" if ref["k"] == 0 else f"sc2-step-{ref['k'] - 1}.bin")
    ).read_bytes()
    need(
        sha256(initial).hexdigest() == ref["in_sha"]
        and ref["binding"] == bind
        and ref["owners"] == list(model.owners),
        "reference inlet",
    )
    q, v, t = unpack(np.frombuffer(initial, dtype="<i4"), model.signature)
    h = Q(ref["dt"])
    index = case["cell_zero_based"]
    for cell in ref["cells"][:index]:
        q, v = step_moments(cell, model, q, v, h)
        t += h
    packet = checkpoint(q, v, t, model.signature)
    need(
        np.array_equal(np.asarray(q, float), ref["cells"][index][0]["q"])
        and np.array_equal(np.asarray(v, float), ref["cells"][index][0]["v"]),
        "inherited cell state",
    )
    signature = fingerprint((sha(PLAN, normalize=True), bind, model.signature, packet, str(h)))
    inherited = old["entry"]["owner_charges"] + index * 24
    geo = old["entry"]["time_cells"] + 1 + index * (16 + 1)
    return dict(
        input=packet,
        h=h,
        binding=signature,
        prefix_charges=inherited,
        prefix_geometry=geo,
        witness_time=Q(case["early_sample_s"]),
        reference_cell=ref["cells"][index],
    )


class InitialBranch(ClothPlatform):
    def __init__(self, model, active):
        super().__init__(model.parts)
        self.active = np.array(active, bool)

    def forces(self, q, v):
        f = super().forces(q, v)
        return Forces(
            f.material,
            np.where(self.active[:, None], f.rows, 0.0),
            np.where(self.active, f.pressures, 0.0),
        )


def find_first(chart, model, mask):
    # Only the smooth extension of the inlet branch defines the incoming event.
    # The checker independently proves monotonicity and validates every other guard.
    brackets = []
    for i, enabled in enumerate(mask):
        if enabled:
            continue
        start = direct_gap(chart.x, Q(0), i, model)
        end = direct_gap(chart.x, Q(1), i, model)
        need(start.low > WIDTH, "unresolved outside inlet")
        if end.low > WIDTH:
            continue
        need(end.high < WIDTH, "unresolved extension endpoint")
        left, right = Q(0), Q(1)
        face = model.faces[i // 3]
        for _ in range(32):
            s = (left + right) / 2
            sign = squared_guard(chart.x, s, face, int(face[i % 3]), Q(float(model.rho)))
            need(sign != 0, "exact dyadic equality unsupported by this packet")
            if sign < 0:
                left = s
            else:
                right = s
        brackets.append((left, right, i))
    need(len(brackets) == 1, "single-birth chart unavailable or ambiguous")
    return brackets[0]


def propose(root, model, source):
    h = source["h"]
    q, v, t = unpack(source["input"], model.signature)
    _, gap, rate = model.maps(np.asarray(q, float), np.asarray(v, float))
    mask = [bool(g < 0.002 and u < 0.001) for g, u in zip(gap, rate, strict=True)]
    left = fit(InitialBranch(model, mask), q, v, h, source["binding"], degree=4, sweeps=3)
    curve = Chart(left, model, q, v, h)
    lo, hi, idx = find_first(curve, model, mask)
    b = Q((hi * 4096).__ceil__() + 1, 4096)
    mid = (lo + hi) / 2
    x, u = curve.at(mid)
    mat, _, rates = model.maps(np.asarray(x, float), np.asarray(u, float))
    row = [Q(round(z * (1 << 48)), 1 << 48) for z in mat[idx]]
    speed = -rates[idx]
    slope = Q(round((500 / 6) * speed * (1 + 1000 * speed) * (1 << 32)), 1 << 32)
    dm = (b - mid) * h
    j = Q(round(slope * dm**2 / 2 * (1 << 112)), 1 << 112)
    back = j * dm / 3
    x, u = curve.at(b)
    m = list(map(lambda z: Q(float(z)), model.mass))
    x = [z + r * back / mi for z, r, mi in zip(x, row, m, strict=True)]
    u = [z + r * j / mi for z, r, mi in zip(u, row, m, strict=True)]
    join = checkpoint(x, u, t + h * b, model.signature)
    duration = h * (1 - b)
    need(Q(float(duration)) == duration, "right interval cannot bind binary duration")
    right = fit(model, x, u, duration, packet_hash(join), degree=4, sweeps=3)
    x1, v1 = endpoint(right, x, u)
    out = checkpoint(x1, v1, t + h, model.signature)
    ramp = dict(
        owner=idx,
        lo=str(lo),
        hi=str(hi),
        join=str(b),
        map=list(map(str, row)),
        slope=str(slope),
        J=str(j),
        B=str(back),
    )
    save(
        root / "proposal.json.gz",
        dict(
            binding=source["binding"],
            input_sha=packet_hash(source["input"]),
            active=mask,
            left=left,
            right=right,
            ramp=ramp,
            join_sha=packet_hash(join),
            end_sha=packet_hash(out),
        ),
    )
    print(
        "isolated proposed owner",
        model.owners[idx],
        "charged",
        left["owner_charges"] + right["owner_charges"] + 12,
    )


class CellProgram(PathProgram):
    def __init__(self, model, source, proposal):
        super().__init__(model)
        self.source = source
        self.proposal = proposal

    def problem_signature(self):
        return dict(
            model=self.model.signature, input=self.source["binding"], method="SC2 event cell v1"
        )

    def create_workspace(self, scene, state, inputs, h, settings, context):
        need(
            scene.identity == self.model.parts[0].identity
            and context.backend == "numpy"
            and fingerprint(inputs) == fingerprint(self.model.parts[2])
            and Q(h) == self.source["h"],
            "original scene or inputs",
        )
        return CellWorkspace(self, state, inputs, h)


class CellWorkspace(Workspace):
    def predict(self):
        try:
            source = self.program.source
            q, v, _ = unpack(self.packet, self.model.signature)
            x, u = coordinates(self.state)
            need(
                np.array_equal(self.packet, source["input"])
                and np.array_equal(x, np.asarray(q, float))
                and np.array_equal(u, np.asarray(v, float)),
                "wrong physical inlet",
            )
        except (ArithmeticError, ValueError) as exc:
            raise SolverFailure(str(exc)) from exc

    def position_sweep(self):
        if self.receipt is not None:
            return
        try:
            source = self.program.source
            out, rec = verify(self.model, source["binding"], source, self.program.proposal)
            self.program.attempt["certificate"] = rec
            need(
                rec["accepted"],
                "cell gate failed: " + ", ".join(k for k, v in rec["gates"].items() if not v),
            )
            q, v, _ = unpack(out, self.model.signature)
            self.out = joint_state(q, v, out)
            self.receipt = rec
        except (ArithmeticError, ValueError) as exc:
            self.program.attempt["reason"] = str(exc)
            raise SolverFailure(str(exc)) from exc

    def position_metrics(self):
        return StageMetrics({"force_position": self.receipt["upper"]["node_position_m"]})

    def velocity_metrics(self):
        # The physical velocity defect has its own 2e-6 gate in the certificate.
        return StageMetrics(
            {"two_moment_readback": 0.0 if self.receipt["gates"]["exact_two_moments"] else 1.0}
        )


def transaction(model, source, data, state):
    program = CellProgram(model, source, data)
    settings = ph.StepSettings(
        substeps=1,
        position_iterations=1,
        velocity_iterations=1,
        tolerance=1e-8,
        record_substeps=True,
    )
    before = fingerprint(state)
    with SceneStepper(
        model.parts[0], program, settings, ph.BackendContext("numpy"), model.parts[3].registry
    ) as runtime:
        result = runtime.step(state, model.parts[2], float(source["h"]))
    need(fingerprint(state) == before, "caller state changed")
    if not result.report.accepted:
        need(
            fingerprint(result.state) == before and not result.substep_states,
            "rejected state leaked",
        )
    return result, program.attempt


def validate(root, model, source):
    data = saved(root / "proposal.json.gz")
    q, v, t = unpack(source["input"], model.signature)
    state = joint_state(q, v, source["input"])
    old_x, old_u = step_moments(source["reference_cell"], model, q, v, source["h"])
    old_hash = packet_hash(checkpoint(old_x, old_u, t + source["h"], model.signature))
    result, attempt = transaction(model, source, data, state)
    save(root / "attempt.json.gz", attempt)
    save(root / "transaction.json.gz", result)
    if not result.report.accepted:
        print(attempt)
        raise ValueError("public transaction rejected: " + str(result.report.reason))
    cert = attempt["certificate"]
    need(
        packet_hash(result.state.history["motion"]) == cert["new_endpoint"] != old_hash,
        "unmodified failed output",
    )
    _, v1, _ = unpack(result.state.history["motion"], model.signature)
    x1, _, _ = unpack(result.state.history["motion"], model.signature)
    cert.update(
        old_endpoint=old_hash,
        platform_velocity_change_vs_old_m_s=float(v1[-1] - old_u[-1]),
        platform_position_change_vs_old_m=float(x1[-1] - old_x[-1]),
    )
    negatives = {}
    for kind in (
        "lost-inlet",
        "early-force",
        "omitted-owner",
        "shifted-root",
        "forged-J",
        "old-endpoint",
    ):
        fake = deepcopy(data)
        inp = state
        if kind == "lost-inlet":
            changed = list(v)
            changed[-1] = Q(0)
            need(changed != v, "negative fixture needs moving platform")
            inp = joint_state(q, changed, source["input"])
        elif kind == "early-force":
            fake["left"]["pressure"][0, int(fake["ramp"]["owner"])] += 1
        elif kind == "omitted-owner":
            fake["right"]["owners"] = fake["right"]["owners"][:-1]
        elif kind == "shifted-root":
            fake["ramp"]["lo"] = str(Q(fake["ramp"]["lo"]) + Q(2, 2**32))
            fake["ramp"]["hi"] = str(Q(fake["ramp"]["hi"]) + Q(2, 2**32))
        elif kind == "forged-J":
            fake["ramp"]["J"] = str(2 * Q(fake["ramp"]["J"]))
        else:
            fake["end_sha"] = old_hash
        failed, _ = transaction(model, source, fake, inp)
        need(not failed.report.accepted, "accepted forgery: " + kind)
        negatives[kind] = str(failed.report.reason)
    cert["rejected_transactions"] = negatives
    save(root / "accepted.json", cert)
    with (root / "cell.bin").open("xb") as fp:
        fp.write(result.state.history["motion"].tobytes())
    print(json.dumps(cert, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", required=True, choices=("propose", "verify"))
    args = parser.parse_args()
    model, _, _, bind = original()
    try:
        source = inlet(model, bind)
        (propose if args.phase == "propose" else validate)(args.root, model, source)
    finally:
        model.parts[3].close()
