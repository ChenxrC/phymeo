"""Independently advance Newton's Panda and free payload from an initial/control-only manifest."""

import argparse
import hashlib
import json
from importlib.metadata import version
from itertools import product
from pathlib import Path

import newton
import numpy as np
import warp as wp

from phymeo.builders.panda import load_panda
from phymeo.core.robot import RobotState
from phymeo.dynamics.robot_dynamics import robot_dynamics_terms
from phymeo.dynamics.rotation import quaternion_matrix
from phymeo.experiments.panda_cube import _command_at, _PassiveGeometry, _servo_effort


def named(labels, name):
    found = [i for i, s in enumerate(labels) if s.rsplit("/", 1)[-1] == name]
    if len(found) != 1:
        raise ValueError(f"expected one {name}: {found}")
    return found[0]


def make_model(urdf, spec, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=tuple(spec["gravity_m_s2"]))
    # Inertias are supplied by the URDF. Massless welded collision carriers stay massless.
    builder.default_shape_cfg.density = 0
    builder.rigid_gap = 0
    builder.add_urdf(
        str(urdf),
        floating=False,
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
        joint_damping_mode="passive",
        up_axis=newton.Axis.Z,
    )
    joints = [named(builder.joint_label, name) for name in spec["joint_names"]]
    if builder.joint_mimic_joint[joints[-1]] != joints[-2]:
        raise ValueError("Panda URDF lacks the expected finger mimic")
    # Match the independently moving physical slides and externally applied servo used by Phymeo.
    builder.set_joint_mimic(joints[-1], None)
    qidx = np.array([builder.joint_q_start[j] for j in joints])
    vidx = np.array([builder.joint_qd_start[j] for j in joints])
    for i in vidx:
        for field in (
            "joint_armature",
            "joint_damping",
            "joint_target_ke",
            "joint_target_kd",
            "joint_limit_ke",
            "joint_limit_kd",
        ):
            getattr(builder, field)[i] = 0.0
    pads = []
    scales = np.array([tuple(s) for s in builder.shape_scale])
    for side in ("left", "right"):
        finger = named(builder.body_label, f"panda_{side}finger")
        ids = np.flatnonzero(
            (np.array(builder.shape_body) == finger)
            & (np.array(builder.shape_type) == int(newton.GeoType.BOX))
            & np.all(np.isclose(scales, [0.00875, 0.0076, 0.00925], rtol=0, atol=1e-7), axis=1)
        )
        (pad,) = ids
        pads.append(int(pad))
    center, quat = spec["body_position_m"], spec["body_xyzw"]
    moments = np.array(spec["inertia_diagonal_kg_m2"])
    payload = builder.add_body(
        xform=wp.transform(wp.vec3(*center), wp.quat(*quat)),
        mass=spec["mass_kg"],
        inertia=wp.mat33(*np.diag(moments).ravel()),
        lock_inertia=True,
        label="payload",
    )
    free = named(builder.joint_label, "payload_free_joint")
    half = np.array(spec["full_size_m"]) / 2
    zero = newton.ModelBuilder.ShapeConfig(density=0, ke=0, kd=0, kf=0, mu=0, gap=0, margin=0)
    box = builder.add_shape_box(
        payload, hx=half[0], hy=half[1], hz=half[2], cfg=zero, label="payload_box"
    )
    half_table = np.array(spec["table_full_size_m"]) / 2
    table = builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(*spec["table_center_m"]), wp.quat_identity()),
        hx=half_table[0],
        hy=half_table[1],
        hz=half_table[2],
        cfg=zero,
        label="finite_table",
    )
    for shape, material in [(p, spec["pad_material"]) for p in pads] + [
        (table, spec["table_material"])
    ]:
        for field, value in zip(("ke", "kd", "mu", "kf"), (*material[:3], 1000.0), strict=True):
            # Featherstone combines the two materials by arithmetic mean. Payload values are zero.
            getattr(builder, "shape_material_" + field)[shape] = 2 * value
    model = builder.finalize(device=device)
    state = model.state()
    q, v = state.joint_q.numpy(), state.joint_qd.numpy()
    q[qidx], v[vidx] = spec["initial_q"], spec["initial_v"]
    iq, iv = int(model.joint_q_start.numpy()[free]), int(model.joint_qd_start.numpy()[free])
    q[iq : iq + 7] = [*center, *quat]
    v[iv : iv + 6] = [*spec["initial_linear_velocity_m_s"], *spec["initial_omega_rad_s"]]
    state.joint_q.assign(q)
    state.joint_qd.assign(v)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    initial_body = state.body_q.numpy()[payload]
    np.testing.assert_allclose(initial_body[:3], center, atol=2e-6)
    assert abs(float(initial_body[3:] @ np.array(quat))) > 1 - 1e-6
    np.testing.assert_allclose(state.joint_qd.numpy()[vidx], spec["initial_v"], atol=1e-7)
    assert abs(float(model.body_mass.numpy()[payload]) - spec["mass_kg"]) < 2e-7
    np.testing.assert_allclose(model.body_inertia.numpy()[payload], np.diag(moments), atol=2e-9)
    # No other shape contributes load in the sampled Phymeo scene. Audit its passive geometry separately.
    pairs = wp.array(
        [(min(p, box), max(p, box)) for p in (*pads, table)], dtype=wp.vec2i, device=device
    )
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="explicit",
        shape_pairs_filtered=pairs,
        reduce_contacts=False,
        rigid_contact_max=64,
    )
    return model, state, qidx, vidx, pads, box, payload, table, pipeline


class ContactRows:
    """Audit native finite-box manifold rows. Featherstone has no CONTACT_F observable in 1.7."""

    def __init__(self, model, pipeline, ids, spec):
        self.model, self.pipeline = model, pipeline
        self.stream = pipeline.contacts()
        n = self.stream.rigid_contact_max
        self.distance = wp.empty(n, dtype=float, device=model.device)
        self.points = [wp.empty(n, dtype=wp.vec3, device=model.device) for _ in range(2)]
        self.owner = model.shape_body.numpy()
        self.com = model.body_com.numpy()
        self.pad0, self.pad1, self.box, self.table = ids
        self.laws = (spec["pad_material"], spec["pad_material"], spec["table_material"])

    def read(self, state, body, rotations, twist):
        self.pipeline.collide(state, self.stream)
        stream = self.stream
        n = int(stream.rigid_contact_count.numpy()[0])
        if not 0 <= n < stream.rigid_contact_max:
            raise RuntimeError(f"native contact capacity: {n}")
        loads, counts = np.zeros(3), np.zeros(3, dtype=int)
        if not n:
            return loads, counts
        newton.eval_rigid_contact_kinematics(
            self.model,
            state,
            stream,
            out_distance=self.distance,
            out_point0_world=self.points[0],
            out_point1_world=self.points[1],
        )
        centers = body[:, :3] + np.einsum("nij,nj->ni", rotations, self.com)
        a, b = stream.rigid_contact_shape0.numpy()[:n], stream.rigid_contact_shape1.numpy()[:n]
        normals, gaps = stream.rigid_contact_normal.numpy()[:n], self.distance.numpy()[:n]
        p0, p1 = [buf.numpy()[:n] for buf in self.points]
        for i, (a_id, b_id) in enumerate(zip(a, b, strict=True)):
            if gaps[i] >= 0:
                continue
            if self.box not in (a_id, b_id):
                raise RuntimeError("unexpected native shape pair")
            other = b_id if a_id == self.box else a_id
            k = (self.pad0, self.pad1, self.table).index(other)
            velocities = []
            for shape, point in ((a_id, p0[i]), (b_id, p1[i])):
                owner = self.owner[shape]
                velocities.append(
                    np.zeros(3)
                    if owner < 0
                    else twist[owner, :3] + np.cross(twist[owner, 3:], point - centers[owner])
                )
            normal_speed = (velocities[1] - velocities[0]) @ normals[i]
            ke, kd = self.laws[k][:2]
            loads[k] += -ke * gaps[i] - kd * min(normal_speed, 0.0)
            counts[k] += 1
        return loads, counts


def run(urdf, inputs, device, max_steps=None):
    spec = json.loads(inputs.read_text(encoding="utf8"))
    assert spec["schema"] == "panda-box-control-v1" and "trace" not in spec
    digest = hashlib.sha256(urdf.read_bytes()).hexdigest()
    assert digest == spec["urdf_sha256"] and version("newton") == "1.7.0"
    asset = load_panda(urdf)
    model, state, iq, iv, pads, box, payload, table, pipeline = make_model(urdf, spec, device)
    joints = [named(model.joint_label, name) for name in spec["joint_names"]]
    arm_owners = model.joint_child.numpy()[joints]
    arm_masses = model.body_mass.numpy()[arm_owners]
    np.testing.assert_allclose(arm_masses, asset.model.mass, rtol=2e-6, atol=1e-7)
    next_state, control = model.state(), model.control()
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0, friction_smoothing=0.04)
    assert str(state.body_q.device) == str(state.joint_q.device) == device
    surface = ContactRows(model, pipeline, [*pads, box, table], spec)
    passive = _PassiveGeometry(asset)
    dt = spec["dt_s"]
    step_count = spec["steps"] if max_steps is None else min(spec["steps"], max_steps)
    times, knots = np.array(spec["knot_times_s"]), np.array(spec["knot_q"])
    cn, half = np.array(spec["table_center_m"]), np.array(spec["table_full_size_m"]) / 2
    ext = np.array(spec["full_size_m"]) / 2
    vertices = np.array(list(product(*[(-h, h) for h in ext])))
    origin = np.array(spec["body_position_m"])
    force_buffer = np.zeros_like(control.joint_f.numpy())
    effort_limit = np.array(asset.effort_limit)
    lower, upper = np.array(asset.lower), np.array(asset.upper)
    rows = []
    cube_margin = table_margin = np.inf
    max_joint_violation = 0.0
    for step in range(step_count):
        t = step * dt
        q = state.joint_q.numpy()[iq].astype(float)
        v = state.joint_qd.numpy()[iv].astype(float)
        body = state.body_q.numpy().astype(float)
        twist = state.body_qd.numpy().astype(float)
        rotations = quaternion_matrix(body[:, 3:])
        motion = RobotState(q, v)
        geometry, _, _, mass, bias = robot_dynamics_terms(asset.model, motion)
        cmd, cmd_v = _command_at(times, knots, t)
        drive = _servo_effort(motion, cmd, cmd_v, mass, bias, effort_limit)
        gap, table_gap, _, _ = passive.clearances(
            geometry, body[payload, :3], rotations[payload], ext, cn, half
        )
        cube_margin, table_margin = min(cube_margin, gap), min(table_margin, table_gap)
        loads, counts = surface.read(state, body, rotations, twist)
        state.clear_forces()
        force_buffer[iv] = drive
        control.joint_f.assign(force_buffer)
        solver.step(state, next_state, control, surface.stream, dt)
        state, next_state = next_state, state
        now = state.body_q.numpy()[payload].astype(float)
        speed = state.body_qd.numpy()[payload].astype(float)
        after_q = state.joint_q.numpy()[iq].astype(float)
        if not np.isfinite(np.r_[now, speed, after_q, loads]).all():
            raise RuntimeError(f"non-finite native state at physical step {step + 1}")
        max_joint_violation = max(
            max_joint_violation, float(np.max(lower - after_q)), float(np.max(after_q - upper))
        )
        rotation = quaternion_matrix(now[None, 3:])[0]
        clearance = now[2] + (vertices @ rotation.T)[:, 2].min() - (cn[2] + half[2])
        net_force = spec["mass_kg"] * ((speed[:3] - twist[payload, :3]) / dt - spec["gravity_m_s2"])
        rows.append(
            dict(
                t=(step + 1) * dt,
                q=after_q.tolist(),
                drive=drive.tolist(),
                box=now.tolist(),
                box_twist=speed.tolist(),
                contact_rows=counts.tolist(),
                reconstructed_normal_N=loads.tolist(),
                net_non_gravity_N=net_force.tolist(),
                clearance_m=float(clearance),
            )
        )
    carry = next((row for row in rows if abs(row["t"] - 1.35) < dt / 2), None)
    during = [row for row in rows if 0.95 <= row["t"] <= 1.35]
    release = [row for row in rows if row["t"] > times[-1] - 0.05]
    return dict(
        engine="newton-1.7.0",
        solver="SolverFeatherstone",
        device=device,
        steps=len(rows),
        inputs_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
        urdf_sha256=digest,
        control="same waypoints and capped computed torque evaluated on native pre-step q/v",
        contacts="Newton finite box manifolds, zero margins, explicit pad0/pad1/table vs payload",
        normal_channel="reconstructed from native penetrating rows and Featherstone spring/damper law",
        observables=sorted(str(v) for v in solver.SUPPORTED_OBSERVABLES),
        shape_ids=dict(left_pad=pads[0], right_pad=pads[1], payload=box, table=table),
        native_arm_masses_kg=arm_masses.astype(float).tolist(),
        carry_clearance_m=None if carry is None else carry["clearance_m"],
        carry_displacement_m=None if carry is None else carry["box"][0] - origin[0],
        min_carry_bilateral_normal_N=min(
            (min(r["reconstructed_normal_N"][:2]) for r in during), default=None
        ),
        peak_carry_table_N=max((r["reconstructed_normal_N"][2] for r in during), default=None),
        peak_bilateral_normal_N=max(min(r["reconstructed_normal_N"][:2]) for r in rows),
        final_target_error_m=float(
            np.linalg.norm(np.array(rows[-1]["box"][:3]) - origin - [spec["travel_m"], 0, 0])
        ),
        final_table_normal_N=float(np.mean([r["reconstructed_normal_N"][2] for r in release]))
        if release
        else None,
        final_net_non_gravity_z_N=float(np.mean([r["net_non_gravity_N"][2] for r in release]))
        if release
        else None,
        release_pad_normal_N=max(
            (sum(r["reconstructed_normal_N"][:2]) for r in release), default=None
        ),
        other_robot_cube_axis_clearance_m=cube_margin,
        robot_table_axis_clearance_m=table_margin,
        max_joint_limit_violation=max_joint_violation,
        trace=rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--steps", type=int, help="short explicit construction/step diagnostic")
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    with options.out.open("x", encoding="utf8") as out:
        result = run(options.urdf, options.inputs, options.device, options.steps)
        json.dump(result, out, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}, allow_nan=False))
