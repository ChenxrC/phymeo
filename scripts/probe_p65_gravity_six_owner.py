"""N2 gravity time weighting and one conditional six-owner HOST pressure proposal."""

import argparse
import json
import traceback
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
from probe_p65_joint_readback import PLAN as FIXTURE
from probe_p65_joint_readback import fixture, observe, save
from probe_p65_paired_load_history import checked_moments, digest, wire
from verify_p65_contact_layer_screen import sha

import phymeo as ph
from phymeo.constraints.bending import bending_angle_gradient
from phymeo.core.scene import fingerprint
from phymeo.dynamics.robot_kinematics import forward_kinematics
from phymeo.runtime.terminal_impulse import PointVelocityImpulses, TerminalImpulseProgram

PLAN = FIXTURE.with_name("p65-gravity-six-owner-protocol.json")
PLAN_SHA = "def848e6494eab9148ac924fc410bea400f9468a7446b1b697776a851beaa690"


def legacy(case, plan, parts):
    archive = FIXTURE.with_name(plan["previous_result"])
    if sha(archive, normalize=True) != plan["previous_result_sha256_lf"]:
        raise ValueError("previous readback result changed")
    old = json.loads(archive.read_bytes())
    row = next(r for r in old["cases"] if r["case"] == case)
    path = PLAN.parents[2] / old["raw_directory"] / f"{case}.packet.json"
    data = path.read_bytes()
    p = json.loads(data)
    scene, state, inputs, _, mat, _, _, surfaces, patch = parts
    if (
        digest(data) != row["packet_sha256"]
        or p["protocol_sha256"] != plan["previous_protocol_sha256_lf"]
        or p["case"] != case
        or p["scene_id"] != scene.identity
        or p["state_and_inputs"] != fingerprint((state, inputs))
        or p["material"] != fingerprint(mat)
        or p["geometry"] != fingerprint((surfaces, patch))
    ):
        raise ValueError("prescribed history binding")
    j, b = checked_moments(p)
    return dict(
        kind="prescribed",
        packet_sha256=row["packet_sha256"],
        source_start=np.array(
            [float(Q(o["share"])) * float.fromhex(p["macros"][0]["words"][0]) for o in p["owners"]]
        ),
        source_end=np.array(
            [
                float(Q(o["share"])) * float.fromhex(p["macros"][-1]["words"][-1])
                for o in p["owners"]
            ]
        ),
        j=[j * Q(o["share"]) for o in p["owners"]],
        b=[b * Q(o["share"]) for o in p["owners"]],
    )


def offset(cloth, ps, surfaces):
    ids = np.repeat([s.node_indices for s in surfaces], 3, axis=0)
    return ph.TriangleOffsetPointPort(
        ph.ParticlePointPort(cloth, ps, np.arange(4), domain="cloth"),
        ids,
        np.tile(np.eye(3), (2, 1)),
        np.full(6, -surfaces[0].half_thickness_m),
    )


def local_force(state, parts):
    _, _, _, _, mat, robot, cloth, surfaces, patch = parts
    x = state[:12].reshape(4, 3)
    v = state[12:24].reshape(4, 3)
    q, u = state[-2:]
    particle = ph.ParticleState(x, v)
    port = offset(cloth, particle, surfaces)
    points = port.positions()
    gap = points[:, 2] - q - patch.local_vertices[0, 2]
    rate = port.velocities()[:, 2] - u
    if np.any(gap < 0) or np.max(abs(points[:, :2])) >= 2:
        raise ValueError("coupled stage outside original material or finite patch")
    # All six reference areas are fixed at 1/6, independently of shared nodes.
    pressure = 500 / 6 * np.maximum(0.002 - gap, 0) * np.maximum(1 - 1000 * rate, 0)
    mat_force = np.zeros_like(x)
    for (a, b), rest, alpha in zip(
        mat.distance.indices, mat.distance.rest_length, mat.distance.compliance, strict=True
    ):
        edge = x[b] - x[a]
        length = np.linalg.norm(edge)
        f = ((length - rest) / alpha) * edge / length
        mat_force[a] += f
        mat_force[b] -= f
    for ids, rest, alpha in zip(
        mat.bending.indices, mat.bending.rest_angle, mat.bending.compliance, strict=True
    ):
        theta, G = bending_angle_gradient(x[ids])
        if abs(theta - rest) >= np.pi:
            raise ValueError("bending branch crossed")
        np.add.at(mat_force, ids, -((theta - rest) / alpha) * G)
    ids = np.repeat([s.node_indices for s in surfaces], 3, axis=0)
    contact = np.zeros_like(x)
    np.add.at(
        contact,
        ids.reshape(-1),
        (port.jacobians[:, :, 2, :] * pressure[:, None, None]).reshape(-1, 3),
    )
    accel = cloth.gravity + (mat_force + contact) * cloth.inverse_mass[:, None]
    dy = np.r_[v.ravel(), accel.ravel(), u, -pressure.sum() / robot.mass[0]]
    return dy, dict(
        state=state.copy(),
        gap=gap,
        rate=rate,
        force=pressure,
        dy=dy,
        material_force=mat_force,
        contact_force=contact,
    )


def coupled_history(root, h, parts):
    check = json.loads((root / "audit-controls.json").read_bytes())
    if not check["transfer_to_coupled"]:
        raise ValueError("coupled transition condition not met")
    original = parts[1]
    x = original.domains["cloth"]
    r = original.domains["robot"]
    y = np.r_[x.position.ravel(), x.velocity.ravel(), r.q, r.v]
    cells = 20
    dt = h / cells
    stages = []
    js = [Q(0)] * 6
    bs = [Q(0)] * 6
    for n in range(cells):
        derivatives = []
        for offset_t, weight in zip((Q(0), Q(1, 2), Q(1, 2), Q(1)), (1, 2, 2, 1), strict=True):
            trial = y if not derivatives else y + float(dt * offset_t) * derivatives[-1]
            try:
                dy, record = local_force(trial, parts)
            except Exception:
                save(
                    root / "coupled-incomplete.json.gz",
                    dict(stages=stages, cell=n, attempted_state=trial),
                )
                raise
            derivatives.append(dy)
            time = (n + offset_t) * dt
            record.update(time=time, weight=weight)
            stages.append(record)
            for i, f in enumerate(record["force"]):
                j = dt * weight * Q(float(f)) / 6
                js[i] += j
                bs[i] += (h - time) * j
        a, b, c, d = derivatives
        y = y + float(dt) / 6 * (a + 2 * b + 2 * c + d)
    _, out = local_force(y, parts)
    save(
        root / "coupled-reference.json.gz",
        dict(
            stages=stages,
            endpoint=out,
            n_cells=cells,
            owner_force_evaluations=6 * (len(stages) + 1),
            j=js,
            b=bs,
            state_and_inputs=fingerprint((parts[1], parts[2])),
        ),
    )
    return dict(
        kind="coupled quadrature proposal",
        reference_sha256=sha(root / "coupled-reference.json.gz"),
        source_start=stages[0]["force"],
        source_end=out["force"],
        j=js,
        b=bs,
    )


def replay(root, case, spec, parts, h):
    scene, state, inputs, base, mat, robot, cloth, surfaces, patch = parts
    source = coupled_history(root, h, parts) if case == "coupled" else legacy(case, spec, parts)
    left = np.zeros((6, 3))
    right = np.zeros((6, 3))
    for i, (j, b) in enumerate(zip(source["j"], source["b"], strict=True)):
        if j < 0 or not 0 <= b <= h * j:
            raise ValueError("negative temporal shape weights")
        left[i, 2] = float(b / h)
        right[i, 2] = float(j - b / h)
    start = observe(robot, cloth, state, surfaces, patch, left)
    full = np.array([cloth.gravity * float(h) / w for w in cloth.inverse_mass])
    gravity_left_adjust = -full / 2
    gravity_right = full / 2
    com = forward_kinematics(robot, state.domains["robot"].q).com_position[0]
    held = {
        "cloth": {"force": start["node"] + gravity_left_adjust},
        "robot": {
            "force": -left.sum(axis=0, keepdims=True),
            "torque": np.cross(start["robot_points"] - com, -left).sum(axis=0, keepdims=True),
        },
    }
    settings = ph.StepSettings(
        substeps=1,
        position_iterations=512,
        velocity_iterations=512,
        tolerance=1e-8,
        record_substeps=True,
    )
    binding = fingerprint((state, inputs))

    def evaluate(program):
        with ph.SceneLoadReplay(
            scene,
            program,
            state,
            inputs,
            float(h),
            channels={"cloth": ("force",), "robot": ("force", "torque")},
            settings=settings,
            context=ph.BackendContext("numpy"),
            registry=base.registry,
        ) as runtime:
            return runtime.evaluate(held)

    selected = evaluate(base)
    save(root / f"{case}-selected.json.gz", selected)
    if not selected.result.report.accepted:
        raise ValueError("selected material step rejected: " + str(selected.result.report.reason))
    end = observe(robot, cloth, selected.result.state, surfaces, patch, right)
    requests = (
        PointVelocityImpulses("cloth", np.arange(4), end["node"] + gravity_right),
        PointVelocityImpulses("robot", np.zeros(6, dtype=int), -right, end["robot_local"]),
    )
    wrapper = TerminalImpulseProgram(base, requests, registry=base.registry)
    try:
        final = evaluate(wrapper)
    finally:
        wrapper.close()
    save(root / f"{case}-final.json.gz", final)
    if not final.result.report.accepted:
        raise ValueError("terminal material step rejected: " + str(final.result.report.reason))
    out = final.result.state
    postport = offset(cloth, out.domains["cloth"], surfaces)
    evidence = final.result.report.substeps[0].diagnostics["terminal_impulse"]
    if (
        wire(evidence["pre_impulse_state"]) != wire(selected.result.state)
        or wire(evidence["pre_impulse_diagnostics"])
        != wire(selected.result.report.substeps[0].diagnostics)
        or wire(selected.effective_inputs) != wire(final.effective_inputs)
        or binding != fingerprint((state, inputs))
    ):
        raise ValueError("different original input or replay selection")
    sweeps = []
    for s in surfaces:
        ids = list(s.node_indices)
        sweeps.append(
            ph.triangle_prism_patch_sweep(
                robot,
                state.domains["robot"].q,
                out.domains["robot"].q,
                patch,
                state.domains["cloth"].position[ids],
                out.domains["cloth"].position[ids],
                half_thickness_m=s.half_thickness_m,
                penetration_budget_m=0.0,
                max_intervals=4096,
                max_depth=32,
            )
        )
    save(
        root / f"{case}-evidence.json.gz",
        dict(
            initial=state,
            inputs=inputs,
            material=mat,
            scene=scene.identity,
            state_and_inputs=binding,
            source=source,
            model_gravity=cloth.gravity,
            node_mass=1 / cloth.inverse_mass,
            patch=patch,
            platform_com=com,
            effective_cloth=fingerprint(final.effective_inputs.domains["cloth"]),
            original_cloth=fingerprint(inputs.domains["cloth"]),
            gravity_body=full,
            gravity_left_adjust=gravity_left_adjust,
            gravity_right=gravity_right,
            left_world=left,
            right_world=right,
            held=held,
            start=start,
            end=end,
            post_points=postport.positions(),
            post_rates=postport.velocities(),
            sweeps=sweeps,
        ),
    )
    print(
        case,
        "mechanical step returned; public sweeps",
        [s.status for s in sweeps],
        "end gaps",
        [float(s["gap"][1]) for s in end["manifolds"]],
    )
    return 2 if any(s.status != "certified" for s in sweeps) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("forward", "reverse", "shifted", "coupled"), required=True
    )
    args = parser.parse_args()
    spec = json.loads(PLAN.read_bytes())
    if (
        sha(PLAN, normalize=True) != PLAN_SHA
        or sha(FIXTURE, normalize=True) != spec["fixture_sha256_lf"]
        or sha(PLAN.with_name(spec["previous_protocol"]), normalize=True)
        != spec["previous_protocol_sha256_lf"]
        or Path(ph.__file__).resolve().parent != (PLAN.parents[2] / "src/phymeo").resolve()
    ):
        raise ValueError("different protocol or project")
    f = json.loads(FIXTURE.read_bytes())
    parts = fixture(f)
    try:
        return replay(args.root, args.phase, spec, parts, Q(f["h_s"]))
    except Exception as exc:
        save(
            args.root / f"{args.phase}-failure.json",
            dict(reason=str(exc), trace=traceback.format_exc()),
        )
        raise
    finally:
        parts[3].close()


if __name__ == "__main__":
    raise SystemExit(main())
