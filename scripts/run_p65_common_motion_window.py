"""Resume the already certified two-step SC1 prefix through the original public HOST program."""

import argparse
import json
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
from audit_p65_common_motion import file_sha, need, readback
from p65_common_motion import PACKET_BYTES, unpack
from probe_p65_common_motion import ClothPlatform, PathProgram, joint_state, saved, verified_config
from probe_p65_joint_readback import fixture, save
from verify_p65_contact_layer_screen import sha

import phymeo as ph
from phymeo.core.scene import fingerprint
from phymeo.runtime.scene_stepper import SceneStepper

PLAN = (
    Path(__file__).resolve().parents[1] / "docs/benchmarks/p65-common-motion-window-protocol.json"
)
PLAN_SHA = "f62162e31b352360c6c75d44994ff175632bfc13a1b236a741953559b4ebc65e"


def metadata():
    spec = json.loads(PLAN.read_bytes())
    seed = PLAN.with_name(spec["seed_result"])
    need(
        sha(PLAN, normalize=True) == PLAN_SHA
        and sha(seed, normalize=True) == spec["seed_result_sha256_lf"]
        and sha(PLAN.with_name(spec["active_protocol"]), normalize=True)
        == spec["active_sha256_lf"],
        "common-motion window metadata changed",
    )
    prefix = json.loads(seed.read_bytes())
    first, second = prefix["steps"]
    h = Q(spec["h_s"])
    T = Q(spec["T_s"])
    need(
        Q(second["end_s"]) == Q(spec["seed_at_s"]) == 2 * h
        and T == 8 * h
        and spec["step_indices"] == list(range(2, 8)),
        "original time grid",
    )
    need(
        first["accepted"]
        and second["accepted"]
        and first["output_sha256"] == second["inlet_sha256"]
        and second["output_sha256"] == spec["seed_checkpoint_sha256"],
        "different accepted prefix",
    )
    return spec, prefix, h, T


def tag(i):
    return f"step-{i:02d}"


def predecessor(root, i, prefix, h):
    if i == 2:
        data = prefix["steps"][1]
        path = PLAN.parents[2] / prefix["raw_directory"] / "next-output.bin"
    else:
        label = tag(i - 1)
        data = saved(root / f"{label}-audit.json")
        path = root / f"{label}-output.bin"
        need(
            data["window_protocol_sha256"] == PLAN_SHA
            and data["tag"] == label
            and data["step_index"] == i - 1,
            "uncertified predecessor index",
        )
    need(
        data["accepted"] and Q(data["end_s"]) == i * h and file_sha(path) == data["output_sha256"],
        "unaccepted or changed predecessor checkpoint",
    )
    return path, data


def advance(root, i, parts, model, path, data, h):
    packet = np.frombuffer(path.read_bytes(), dtype="<i4").copy()
    q, v, t = unpack(packet, model.signature)
    need(
        t == i * h and any(v) and len(packet.tobytes()) == PACKET_BYTES,
        "wrong exact incoming time, velocity or packet",
    )
    scene, _, inputs, base = parts[:4]
    state = joint_state(q, v, packet)
    entry = fingerprint((state, inputs))
    label = tag(i)
    save(
        root / f"{label}-input.json",
        dict(
            step_index=i,
            window_protocol_sha256=PLAN_SHA,
            inlet_sha256=data["output_sha256"],
            predecessor_source=path.relative_to(PLAN.parents[2]).as_posix(),
            clock=t,
            model_signature=model.signature,
            registered_entry=entry,
        ),
    )
    program = PathProgram(model)
    with SceneStepper(
        scene,
        program,
        ph.StepSettings(
            substeps=1,
            position_iterations=1,
            velocity_iterations=1,
            tolerance=1e-8,
            record_substeps=True,
        ),
        ph.BackendContext("numpy"),
        base.registry,
    ) as runtime:
        result = runtime.step(state, inputs, float(h))
    save(root / f"{label}-attempt.json.gz", program.attempt)
    save(root / f"{label}-result.json.gz", result)
    need(fingerprint((state, inputs)) == entry, "caller state/input mutated")
    if not result.report.accepted:
        need(
            fingerprint(result.state) == fingerprint(state) and not result.substep_states,
            "rejected real frame did not roll back",
        )
        save(
            root / f"{label}-rejected.json",
            dict(reason=result.report.reason, rolled_back=True, last_accepted_s=float(t)),
        )
        print(label, "REJECTED", result.report.reason)
        return 2
    output = result.state.history["motion"]
    _, _, time = unpack(output, model.signature)
    need(time == (i + 1) * h, "wrong outgoing clock")
    with (root / f"{label}-output.bin").open("xb") as file:
        file.write(output.tobytes())
    cert = program.attempt["certificate"]
    print(
        label,
        "committed",
        (i + 1) * h,
        "end gap",
        min(z[0] for z in cert["endpoint_gaps_m"]),
        "max law impulse error",
        max(cert["j_defect_upper_ns"]),
    )
    return 0


def audit_step(root, i, model, path, previous, h):
    label = tag(i)
    r = readback(root, label, model, predecessor_checkpoint=path)
    record = saved(root / f"{label}-input.json")
    need(
        record["inlet_sha256"] == previous["output_sha256"] == r["inlet_sha256"]
        and Q(record["clock"]) == i * h
        and record["window_protocol_sha256"] == PLAN_SHA
        and Q(r["start_s"]) == i * h
        and Q(r["end_s"]) == (i + 1) * h,
        "audit predecessor identity",
    )
    need(
        r["true_inlet_velocity"] == previous["public_velocity"]
        and record["model_signature"] == model.signature,
        "velocity or model reset",
    )
    r.update(
        step_index=i,
        window_protocol_sha256=PLAN_SHA,
        predecessor_source=record["predecessor_source"],
    )
    save(root / f"{label}-audit.json", r)
    print(label, "disk audit accepted", r["output_sha256"])
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=tuple(f"{kind}-{i}" for kind in ("step", "audit") for i in range(2, 8)),
        required=True,
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    _, prefix, h, _ = metadata()
    kind, number = args.phase.split("-")
    i = int(number)
    config, _ = verified_config()
    parts = fixture(config)
    try:
        model = ClothPlatform(parts)
        path, previous = predecessor(args.root, i, prefix, h)
        if kind == "step":
            return advance(args.root, i, parts, model, path, previous, h)
        return audit_step(args.root, i, model, path, previous, h)
    finally:
        parts[3].close()


if __name__ == "__main__":
    raise SystemExit(main())
