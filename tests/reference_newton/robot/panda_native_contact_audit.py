"""Audit the installed Newton 1.7 Featherstone contact kernel in independent output buffers."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton._src.solvers.semi_implicit.kernels_contact import eval_body_contact
from panda_loaded_box import ContactRows, make_model

from phymeo.builders.panda import load_panda
from phymeo.core.robot import RobotState
from phymeo.dynamics.robot_dynamics import robot_dynamics_terms
from phymeo.dynamics.rotation import quaternion_matrix
from phymeo.experiments.panda_cube import _command_at, _servo_effort

CONTACT_FIELDS = (
    "rigid_contact_point0",
    "rigid_contact_point1",
    "rigid_contact_surface_velocity",
    "rigid_contact_normal",
    "rigid_contact_shape0",
    "rigid_contact_shape1",
    "rigid_contact_margin0",
    "rigid_contact_margin1",
    "rigid_contact_stiffness",
    "rigid_contact_damping",
    "rigid_contact_friction",
)


def kernel_wrench(model, state, solver, stream, box_body, mu, row=None, smoothing=None):
    # The solver has already advanced its separate output state. body_qd_fk is its actual
    # pre-step COM twist used in eval_body_contact, while state is still that step's input.
    selected = []
    for field in CONTACT_FIELDS:
        buf = getattr(stream, field)
        if row is not None and buf is not None and buf.shape[0]:
            buf = wp.array(buf.numpy()[row : row + 1], dtype=buf.dtype, device=model.device)
        selected.append(buf)
    count = (
        stream.rigid_contact_count if row is None else wp.array([1], dtype=int, device=model.device)
    )
    output = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
    wp.launch(
        eval_body_contact,
        dim=stream.rigid_contact_max if row is None else 1,
        inputs=[
            state.body_q,
            solver.body_qd_fk,
            model.body_com,
            model.shape_material_ke,
            model.shape_material_kd,
            model.shape_material_kf,
            model.shape_material_ka,
            mu,
            model.shape_body,
            count,
            *selected,
            False,
            solver.friction_smoothing if smoothing is None else smoothing,
        ],
        outputs=[output],
        device=model.device,
    )
    return output.numpy()[box_body].astype(float)


def row_material(stream, model, index, a, b):
    values = {}
    for key in ("ke", "kd", "kf", "mu"):
        props = getattr(model, "shape_material_" + key).numpy()
        values[key] = 0.5 * float(props[a] + props[b])
    for key, field in (
        ("ke", "rigid_contact_stiffness"),
        ("kd", "rigid_contact_damping"),
        ("mu", "rigid_contact_friction"),
    ):
        buf = getattr(stream, field)
        value = float(buf.numpy()[index]) if buf is not None and buf.shape[0] else 0.0
        values[key + "_override"] = value
        if value > 0:
            values[key] = value if key != "mu" else values[key] * value
    return values


def audit_step(step, model, before, after, solver, surface, payload, box, pads, mu_zero, dt):
    stream = surface.stream
    n = int(stream.rigid_contact_count.numpy()[0])
    all_force = kernel_wrench(model, before, solver, stream, payload, model.shape_material_mu)
    normal_only = kernel_wrench(model, before, solver, stream, payload, mu_zero)
    default_huber = (
        inspect.signature(type(solver).__init__).parameters["friction_smoothing"].default
    )
    default_force = kernel_wrench(
        model, before, solver, stream, payload, model.shape_material_mu, smoothing=default_huber
    )
    mass = float(model.body_mass.numpy()[payload])
    body0, body1 = before.body_q.numpy().astype(float), after.body_q.numpy().astype(float)
    spin0, spin1 = solver.body_qd_fk.numpy().astype(float), after.body_qd.numpy().astype(float)
    rotation0, rotation1 = quaternion_matrix(body0[:, 3:]), quaternion_matrix(body1[:, 3:])
    com0 = body0[:, :3] + np.einsum("nij,nj->ni", rotation0, model.body_com.numpy())
    dynamic = mass * ((spin1[payload, :3] - spin0[payload, :3]) / dt - [0.0, 0.0, -9.81])
    inertia = model.body_inertia.numpy()[payload]
    angular_change = (
        rotation1[payload] @ inertia @ rotation1[payload].T @ spin1[payload, 3:]
        - rotation0[payload] @ inertia @ rotation0[payload].T @ spin0[payload, 3:]
    )
    rows = []
    lookup = model.shape_body.numpy()
    shape_a, shape_b = stream.rigid_contact_shape0.numpy(), stream.rigid_contact_shape1.numpy()
    normals = stream.rigid_contact_normal.numpy()
    margins = stream.rigid_contact_margin0.numpy(), stream.rigid_contact_margin1.numpy()
    tangential_velocity = stream.rigid_contact_surface_velocity
    slots = (
        np.zeros((n, 3))
        if tangential_velocity is None or not tangential_velocity.shape[0]
        else tangential_velocity.numpy()[:n]
    )
    gaps = surface.distance.numpy()[:n]
    p0, p1 = [buf.numpy()[:n] for buf in surface.points]
    for i in range(n):
        a, b = int(shape_a[i]), int(shape_b[i])
        normal = normals[i].astype(float)
        opposite = -normal
        va = []
        for shape, p, margin, sign in ((a, p0[i], margins[0][i], -1), (b, p1[i], margins[1][i], 1)):
            owner = lookup[shape]
            actual_p = p.astype(float) + sign * margin * opposite if owner >= 0 else p
            va.append(
                np.zeros(3)
                if owner < 0
                else spin0[owner, :3] + np.cross(spin0[owner, 3:], actual_p - com0[owner])
            )
        tangent = va[0] - va[1] - (slots[i] - opposite * np.dot(opposite, slots[i]))
        tangent -= np.dot(tangent, opposite) * opposite
        speed = float(np.linalg.norm(tangent))
        delta = solver.friction_smoothing
        huber = 0.5 * speed**2 if speed <= delta else delta * (speed - 0.5 * delta)
        wrench = kernel_wrench(model, before, solver, stream, payload, model.shape_material_mu, i)
        zero_mu = kernel_wrench(model, before, solver, stream, payload, mu_zero, i)
        fn = float(np.linalg.norm(zero_mu[:3]))
        ft = float(np.linalg.norm(wrench[:3] - zero_mu[:3]))
        law = row_material(stream, model, i, a, b)
        cap, dashpot = law["mu"] * fn, law["kf"] * huber
        active = gaps[i] < 0 and speed > 0
        branch = ("viscous" if dashpot <= cap else "huber_limited") if active else "inactive"
        predicted_tangent = speed / huber * min(dashpot, cap) if active and huber else 0.0
        rows.append(
            dict(
                shapes=[a, b],
                gap_m=float(gaps[i]),
                normal=normal.tolist(),
                vt_m_s=speed,
                huber=huber,
                material=law,
                viscous_argument_N=dashpot,
                muN=cap,
                branch=branch,
                normal_N=fn,
                tangent_N=ft,
                predicted_tangent_N=predicted_tangent,
                wrench_on_payload=wrench.tolist(),
                normal_only=zero_mu.tolist(),
            )
        )
    row_total = np.sum([r["wrench_on_payload"] for r in rows], axis=0) if rows else np.zeros(6)
    return dict(
        t=(step + 1) * dt,
        contacts=rows,
        aggregate_kernel_wrench=all_force.tolist(),
        default_huber_on_same_state=default_force.tolist(),
        default_friction_smoothing=default_huber,
        zero_mu_kernel_wrench=normal_only.tolist(),
        net_force_from_momentum_N=dynamic.tolist(),
        force_error_N=float(np.linalg.norm(dynamic - all_force[:3])),
        kernel_row_sum_error=float(np.max(np.abs(row_total - all_force))),
        angular_impulse_error_Nms=float(np.linalg.norm(angular_change - dt * all_force[3:])),
        public_vs_contact_fk_twist=float(np.max(np.abs(before.body_qd.numpy() - spin0))),
        any_finger_row=any(r["gap_m"] < 0 and any(s in pads for s in r["shapes"]) for r in rows),
    )


def run(urdf, inputs):
    spec = json.loads(inputs.read_text(encoding="utf8"))
    assert spec["schema"] == "panda-box-control-v1" and "trace" not in spec
    assert hashlib.sha256(urdf.read_bytes()).hexdigest() == spec["urdf_sha256"]
    asset = load_panda(urdf)
    model, state, iq, iv, pads, box, payload, table, pipe = make_model(urdf, spec, "cpu")
    surface = ContactRows(model, pipe, [*pads, box, table], spec)
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0, friction_smoothing=0.04)
    other, control = model.state(), model.control()
    forces = np.zeros_like(control.joint_f.numpy())
    zero_mu = wp.zeros_like(model.shape_material_mu)
    dt = spec["dt_s"]
    results = []
    for step in range(min(spec["steps"], 1000)):
        q, v = state.joint_q.numpy()[iq].astype(float), state.joint_qd.numpy()[iv].astype(float)
        arm = RobotState(q, v)
        _, _, _, mass, bias = robot_dynamics_terms(asset.model, arm)
        target, rates = _command_at(
            np.array(spec["knot_times_s"]), np.array(spec["knot_q"]), step * dt
        )
        forces[iv] = _servo_effort(arm, target, rates, mass, bias, np.array(asset.effort_limit))
        body = state.body_q.numpy().astype(float)
        _, counts = surface.read(
            state, body, quaternion_matrix(body[:, 3:]), state.body_qd.numpy().astype(float)
        )
        state.clear_forces()
        control.joint_f.assign(forces)
        solver.step(state, other, control, surface.stream, dt)
        if step < 16 or any(counts[:2]):
            results.append(
                audit_step(
                    step, model, state, other, solver, surface, payload, box, pads, zero_mu, dt
                )
            )
        state, other = other, state
        if any(counts[:2]):
            break
    return dict(
        source="actual installed Newton 1.7 eval_body_contact, detached buffers",
        input_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
        friction_smoothing=solver.friction_smoothing,
        steps=step + 1,
        records=results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.urdf, args.inputs)
    with args.out.open("x", encoding="utf8") as dest:
        json.dump(result, dest, allow_nan=False)
    rows = result["records"]
    assert (
        len(rows) == 17
        and rows[-1]["any_finger_row"]
        and not any(r["any_finger_row"] for r in rows[:16])
    )
    force_error = max(r["force_error_N"] for r in rows)
    row_error = max(r["kernel_row_sum_error"] for r in rows)
    tangent_error = max(
        abs(c["tangent_N"] - c["predicted_tangent_N"]) for r in rows for c in r["contacts"]
    )
    print(
        json.dumps(
            dict(
                steps=result["steps"],
                max_force_error_N=force_error,
                max_row_error=row_error,
                max_tangent_error_N=tangent_error,
                first_finger=rows[-1],
            ),
            allow_nan=False,
        )
    )
    assert force_error < 0.015 and row_error < 0.002 and tangent_error < 0.02
