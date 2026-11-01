"""N2 metrology: drive real HOST participants from one disk-bound, prescribed pressure history."""

import argparse
import copy
import hashlib
import json
import traceback
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
from probe_p65_joint_readback import PLAN as FIXTURE
from probe_p65_joint_readback import encode, fixture, observe, save
from verify_p65_contact_layer_screen import sha

import phymeo as ph
from phymeo.core.scene import fingerprint
from phymeo.dynamics.robot_kinematics import forward_kinematics
from phymeo.runtime.terminal_impulse import PointVelocityImpulses, TerminalImpulseProgram

PLAN = FIXTURE.with_name("p65-paired-load-history-protocol.json")
PLAN_SHA = "a6379ca2a7b5094f73df58880223c0a5d835a602d54c6cf2d5811d75da110a24"


def wire(data):
    return (
        json.dumps(encode(data), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def make_packet(case, plan, parts):
    scene, state, inputs, _, mat, _, _, surfaces, patch = parts
    spec = plan["cases"][case]
    prior = FIXTURE.with_name(plan["source"])
    if sha(prior, normalize=True) != plan["source_sha256_lf"]:
        raise ValueError("different source report")
    report = json.loads(prior.read_bytes())
    folder = PLAN.parents[2] / report["raw_directory"] / "step-0"
    file = folder / "forces.jsonl"
    source_hash = report["steps"][0]["hashes"]["forces.jsonl"]
    if sha(file) != source_hash:
        raise ValueError("different force curve bytes")
    raw = [json.loads(line) for line in file.read_bytes().splitlines()]
    h = Q(plan["duration_s"])
    macros = []
    for phase in reversed(raw) if spec["reverse"] else raw:
        dt = Q(phase["dt"])
        t = Q(phase["t"])
        words = phase["words"]
        macros.append(
            dict(
                t=str(h - t - dt if spec["reverse"] else t),
                dt=str(dt),
                words=words[::-1] if spec["reverse"] else words,
            )
        )
    owners = [
        dict(
            surface=s.key,
            triangle=list(s.node_indices),
            corner=k,
            offset=str(-Q(s.half_thickness_m)),
            share=spec["shares"][3 * f + k],
        )
        for f, s in enumerate(surfaces)
        for k in range(3)
    ]
    return dict(
        schema="paired-world-load-history-v1",
        case=case,
        protocol_sha256=PLAN_SHA,
        scene_id=scene.identity,
        state_and_inputs=fingerprint((state, inputs)),
        original_inputs=fingerprint(inputs),
        material=fingerprint(mat),
        geometry=fingerprint((surfaces, patch)),
        owners=owners,
        duration=str(h),
        axis_world=[0, 0, 1],
        macro_source_sha256=source_hash,
        macros=macros,
    )


def checked_moments(packet):
    h = Q(packet["duration"])
    j = b = t = Q(0)
    if len(packet["macros"]) != 63 or sum(Q(o["share"]) for o in packet["owners"]) != 1:
        raise ValueError("wrong coverage")
    for macro in packet["macros"]:
        dt = Q(macro["dt"])
        coeff = [Q(float.fromhex(word)) for word in macro["words"]]
        if len(coeff) != 8 or min(coeff) < 0 or dt <= 0 or Q(macro["t"]) != t:
            raise ValueError("invalid nonnegative degree-seven partition")
        mass = dt * sum(coeff) / 8
        local = dt**2 * sum((8 - k) * v for k, v in enumerate(coeff)) / 72
        j += mass
        b += (h - t - dt) * mass + local
        t += dt
    if t != h or not 0 < b < h * j:
        raise ValueError("invalid force time moments")
    return j, b


def verify_packet(data, claim, expected):
    if digest(data) != claim:
        raise ValueError("packet digest mismatch")
    if data != wire(expected):
        raise ValueError("packet differs from bound pressure, owner or original input")
    return checked_moments(json.loads(data))


def prepare(root, plan, parts):
    negative = []
    for case in plan["cases"]:
        packet = make_packet(case, plan, parts)
        data = wire(packet)
        j, b = verify_packet(data, digest(data), packet)
        with (root / f"{case}.packet.json").open("xb") as file:
            file.write(data)
        save(root / f"{case}.packet-index.json", dict(digest=digest(data), j=j, b=b))
        if case == "forward":
            for fault in ("time", "owner", "original", "digest"):
                forged = copy.deepcopy(packet)
                if fault == "time":
                    forged["macros"][0]["dt"] = "1/8000"
                if fault == "owner":
                    forged["owners"][0]["triangle"] = [1, 0, 3]
                if fault == "original":
                    forged["original_inputs"] = "0" * 64
                changed = wire(forged)
                try:
                    verify_packet(
                        changed, "0" * 64 if fault == "digest" else digest(changed), packet
                    )
                except ValueError as exc:
                    negative.append(dict(fault=fault, reason=str(exc)))
                else:
                    raise AssertionError("forgery accepted")
    save(root / "packet-controls.json", negative)
    print("Three bound packets and four rejected corruptions written.")


def world_loads(owners, impulse):
    world = np.zeros((len(owners), 3))
    world[:, 2] = [float(Q(owner["share"]) * impulse) for owner in owners]
    return world


def replay_case(root, case, plan, parts):
    scene, state, inputs, base, mat, robot, cloth, surfaces, patch = parts
    expected = make_packet(case, plan, parts)
    data = (root / f"{case}.packet.json").read_bytes()
    index = json.loads((root / f"{case}.packet-index.json").read_bytes())
    j, b = verify_packet(data, index["digest"], expected)
    packet = json.loads(data)
    h = Q(packet["duration"])
    left = b / h
    right = j - left
    left_world = world_loads(packet["owners"], left)
    right_world = world_loads(packet["owners"], right)
    start = observe(robot, cloth, state, surfaces, patch, left_world)
    com = forward_kinematics(robot, state.domains["robot"].q).com_position[0]
    partner = -left_world
    held = {
        "cloth": {"force": start["node"]},
        "robot": {
            "force": np.sum(partner, axis=0, keepdims=True),
            "torque": np.sum(np.cross(start["robot_points"] - com, partner), axis=0, keepdims=True),
        },
    }
    settings = ph.StepSettings(
        substeps=1,
        position_iterations=512,
        velocity_iterations=512,
        tolerance=1e-8,
        record_substeps=True,
    )
    context = ph.BackendContext("numpy")

    def evaluate(program):
        with ph.SceneLoadReplay(
            scene,
            program,
            state,
            inputs,
            float(h),
            channels={"cloth": ("force",), "robot": ("force", "torque")},
            settings=settings,
            context=context,
            registry=base.registry,
        ) as replay:
            return replay.evaluate(held)

    selected = evaluate(base)
    save(root / f"{case}-selected.json.gz", selected)
    if not selected.result.report.accepted:
        raise ValueError("selected material trial rejected: " + str(selected.result.report.reason))
    endpoint = selected.result.state
    end = observe(robot, cloth, endpoint, surfaces, patch, right_world)
    requests = (
        PointVelocityImpulses("cloth", np.arange(4), end["node"]),
        PointVelocityImpulses("robot", np.zeros(6, dtype=int), -right_world, end["robot_local"]),
    )
    terminal = TerminalImpulseProgram(base, requests, registry=base.registry)
    try:
        final = evaluate(terminal)
    finally:
        terminal.close()
    save(root / f"{case}-published.json.gz", final)
    if not final.result.report.accepted:
        raise ValueError("terminal trial rejected: " + str(final.result.report.reason))
    sub = final.result.report.substeps[0]
    terminal_record = sub.diagnostics["terminal_impulse"]
    pre_ok = wire(terminal_record["pre_impulse_state"]) == wire(selected.result.state)
    diag_ok = wire(terminal_record["pre_impulse_diagnostics"]) == wire(
        selected.result.report.substeps[0].diagnostics
    )
    if (
        not pre_ok
        or not diag_ok
        or wire(selected.effective_inputs) != wire(final.effective_inputs)
        or fingerprint((state, inputs)) != packet["state_and_inputs"]
    ):
        raise ValueError("pre-terminal, represented input or original binding mismatch")
    out = final.result.state
    sweeps = []
    for surface in surfaces:
        ids = list(surface.node_indices)
        sweeps.append(
            ph.triangle_prism_patch_sweep(
                robot,
                state.domains["robot"].q,
                out.domains["robot"].q,
                patch,
                state.domains["cloth"].position[ids],
                out.domains["cloth"].position[ids],
                half_thickness_m=surface.half_thickness_m,
                penetration_budget_m=0.0,
                max_intervals=4096,
                max_depth=32,
            )
        )
    save(
        root / f"{case}-evidence.json.gz",
        dict(
            packet_digest=index["digest"],
            original_state=state,
            original_inputs=inputs,
            material=mat,
            patch=patch,
            initial_com=com,
            initial_map=start,
            endpoint_map=end,
            left_world=left_world,
            right_world=right_world,
            j=j,
            b=b,
            left=left,
            right=right,
            held_request=held,
            selected_pre_state_bound=pre_ok,
            selected_pre_diagnostics_bound=diag_ok,
            original_cloth_input=fingerprint(inputs.domains["cloth"]),
            effective_cloth_input=fingerprint(final.effective_inputs.domains["cloth"]),
            sweeps=sweeps,
        ),
    )
    hard = all(s.status == "certified" for s in sweeps)
    save(
        root / f"{case}-status.json",
        dict(
            mechanical_replay_accepted=True,
            hard_sweep_certified=hard,
            sweep_status=[s.status for s in sweeps],
            measured_transverse_guide_reaction=False,
            joint_admitted=False,
            successor_published=False,
            diagnostic_trials=2,
            applied_terminal_batches=int(terminal_record["applications"]),
        ),
    )
    print(
        case,
        "robot q/v",
        out.domains["robot"].q,
        out.domains["robot"].v,
        "hard sweeps",
        [s.status for s in sweeps],
    )
    return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", required=True, choices=("prepare", "forward", "reverse", "shifted")
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if (
        sha(PLAN, normalize=True) != PLAN_SHA
        or Path(ph.__file__).resolve().parent != (PLAN.parents[2] / "src/phymeo").resolve()
    ):
        raise ValueError("protocol or active module mismatch")
    plan = json.loads(PLAN.read_bytes())
    if sha(FIXTURE, normalize=True) != plan["fixture_sha256_lf"]:
        raise ValueError("different four-node fixture")
    parts = fixture(json.loads(FIXTURE.read_bytes()))
    try:
        if args.phase == "prepare":
            prepare(args.root, plan, parts)
            return 0
        return replay_case(args.root, args.phase, plan, parts)
    except Exception as exc:
        save(
            args.root / f"{args.phase}-failure.json",
            dict(error=type(exc).__name__, message=str(exc), trace=traceback.format_exc()),
        )
        raise
    finally:
        parts[3].close()


if __name__ == "__main__":
    raise SystemExit(main())
