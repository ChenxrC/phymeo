"""Pinned Newton 1.7 D01/D03 double-dynamic contact observations.

Default channel: native CollisionPipeline + Featherstone contact force, with no
external contact wrench. Optional matched channel independently evaluates the
same frictionless sphere law at Newton's own entry state, supplies body_f, and
passes contacts=None to Featherstone. These are separate physical runs; neither
channel imports Phymeo or copies its trajectory. CPU/CUDA are explicit.

This authoring checkpoint is UNEXECUTED. The caller owns execution/provenance.
"""

import argparse
import gzip
import hashlib
import json
import sys
import traceback
from importlib.metadata import version
from math import cos, sin
from pathlib import Path

import numpy as np

H, STEPS = 1 / 1024, 40
RADIUS, STIFFNESS = 1 / 8, 64.0
ATOL, RTOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-4
MAX_RECORD_BYTES = 128 * 1024 * 1024
CONTACT_FIELDS = (
    "rigid_contact_point_id",
    "rigid_contact_shape0",
    "rigid_contact_shape1",
    "rigid_contact_point0",
    "rigid_contact_point1",
    "rigid_contact_offset0",
    "rigid_contact_offset1",
    "rigid_contact_normal",
    "rigid_contact_margin0",
    "rigid_contact_margin1",
    "rigid_contact_surface_velocity",
    "rigid_contact_stiffness",
    "rigid_contact_damping",
    "rigid_contact_friction",
    "rigid_contact_tids",
    "rigid_contact_force",
)
MODEL_FIELDS = (
    "body_mass",
    "body_inv_mass",
    "body_com",
    "body_inertia",
    "body_inv_inertia",
    "body_flags",
    "body_world",
    "shape_type",
    "shape_body",
    "shape_world",
    "shape_transform",
    "shape_scale",
    "shape_margin",
    "shape_gap",
    "shape_collision_group",
    "shape_collision_filter_pairs",
    "shape_contact_pairs",
    "shape_material_ke",
    "shape_material_kd",
    "shape_material_kf",
    "shape_material_ka",
    "shape_material_mu",
    "joint_type",
    "joint_parent",
    "joint_child",
    "joint_world",
    "joint_axis",
    "joint_X_p",
    "joint_X_c",
    "joint_q_start",
    "joint_qd_start",
    "joint_dof_dim",
    "joint_armature",
    "joint_target_ke",
    "joint_target_kd",
    "joint_damping",
    "joint_friction",
    "joint_limit_lower",
    "joint_limit_upper",
    "joint_limit_ke",
    "joint_limit_kd",
    "joint_articulation",
    "articulation_start",
    "articulation_end",
    "gravity",
)
SOURCES = (
    "_src/sim/builder.py",
    "_src/sim/state.py",
    "_src/sim/articulation.py",
    "_src/sim/collide.py",
    "_src/sim/contacts.py",
    "_src/geometry/broad_phase_common.py",
    "_src/solvers/semi_implicit/kernels_contact.py",
    "_src/solvers/featherstone/solver_featherstone.py",
    "_src/solvers/featherstone/kernels.py",
)


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def save_record(path, record):
    encoded = json.dumps(json_value(record), allow_nan=False, separators=(",", ":")).encode()
    if len(encoded) > MAX_RECORD_BYTES:
        raise RuntimeError("predeclared 128 MiB reference record limit exceeded")
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wb") as stream:
        stream.write(encoded)
    temporary.replace(path)


def gate(row, name, actual, expected, *, atol=ATOL, rtol=RTOL):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    ok = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    passed = bool(ok and np.allclose(a, b, atol=atol, rtol=rtol))
    row.setdefault("checks", {})[name] = dict(
        passed=passed,
        atol=atol,
        rtol=rtol,
        max_abs_error=float(np.max(np.abs(a - b))) if ok and a.size else None,
        actual_shape=a.shape,
        expected_shape=b.shape,
    )
    return passed


def rz(angle):
    c, s = cos(float(angle)), sin(float(angle))
    return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1.0)))


def rotation(xyzw):
    # Do not normalize or repair native state. Newton stores unit XYZW.
    x, y, z, w = np.asarray(xyzw, float)
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def declaration(case):
    return dict(
        case=case,
        mass=(2.0, 1.0),
        pivots=((0, 0, 0), (7 / 32, 0, 0)),
        inertia=((1 / 8,) * 3,) * 2 if case == "D01" else ((1 / 4,) * 3, (1 / 2,) * 3),
        joint_kind="prismatic" if case == "D01" else "revolute",
        axis=(1, 0, 0) if case == "D01" else (0, 0, 1),
        local_sphere_center=(0, 0, 0) if case == "D01" else (0, 1 / 4, 0),
        radius_m=RADIUS,
        stiffness_n_per_m=STIFFNESS,
        damping=0.0,
        friction=0.0,
        gravity=(0, 0, 0),
        initial_q=(0.0, 0.0),
        initial_v=(0.0, 0.0),
        h_s=H,
        steps=STEPS,
        armature=0.0,
        motor=(0.0, 0.0),
        angular_damping=0.0,
        shape_margin_m=0.0,
        detection_gap_m=0.0,
        articulation_boundary="Two native root articulations in one explicit world; block-diagonal physical forest, contact couples their loads.",
    )


def analytic(case, q, v):
    """Declared two-body kinematics and scalar mass solve; no native diagnostic input."""
    data = declaration(case)
    q, v = np.asarray(q, float), np.asarray(v, float)
    origins = np.asarray(data["pivots"], float).copy()
    matrices = np.repeat(np.eye(3)[None], 2, axis=0)
    twists = np.zeros((2, 6))
    if case == "D01":
        origins[:, 0] += q
        twists[:, 0] = v
    else:
        matrices = np.array([rz(a) for a in q])
        twists[:, 5] = v
    centers = origins + np.einsum("bij,j->bi", matrices, data["local_sphere_center"])
    delta = centers[1] - centers[0]
    distance = float(np.linalg.norm(delta))
    if distance <= 0:
        raise ValueError("frozen analytic sphere normal is undefined at coincident centers")
    normal, gap = delta / distance, distance - 2 * RADIUS
    magnitude = STIFFNESS * max(0.0, -gap)
    forces = np.array((-magnitude * normal, magnitude * normal))
    points = np.array((centers[0] + RADIUS * normal, centers[1] - RADIUS * normal))
    torque = np.cross(points - origins, forces)
    wrench = np.column_stack((forces, torque))
    inertia = np.asarray(data["inertia"])[:, 2]
    acceleration = forces[:, 0] / data["mass"] if case == "D01" else torque[:, 2] / inertia
    metric = np.asarray(data["mass"]) if case == "D01" else inertia
    return dict(
        origins=origins,
        rotations=matrices,
        centers=centers,
        twists=twists,
        normal=normal,
        gap=gap,
        points=points,
        magnitude=magnitude,
        wrench=wrench,
        acceleration=acceleration,
        kinetic_energy_j=float(np.dot(metric, v * v) / 2),
    )


def actual_sphere_law(model, state, rows, shapes):
    """Matched channel only: evaluate at this Newton state's actual body poses."""
    pose = state.body_q.numpy().astype(float)
    local = model.shape_transform.numpy().astype(float)
    com_local = model.body_com.numpy().astype(float)
    centers, com = [], []
    for body, shape in zip(rows, shapes, strict=True):
        R = rotation(pose[body, 3:])
        centers.append(pose[body, :3] + R @ local[shape, :3])
        com.append(pose[body, :3] + R @ com_local[body])
    centers, com = np.array(centers), np.array(com)
    delta = centers[1] - centers[0]
    distance = float(np.linalg.norm(delta))
    if distance <= 0:
        raise ValueError("matched-law undefined normal")
    n = delta / distance
    magnitude = STIFFNESS * max(0.0, 2 * RADIUS - distance)
    force = np.array((-magnitude * n, magnitude * n))
    point = (centers[0] + centers[1]) / 2
    applied = np.zeros((model.body_count, 6), np.float32)
    applied[rows, :3] = force
    applied[rows, 3:] = np.cross(point - com, force)
    return applied, dict(centers=centers, com=com, normal=n, load_point=point, magnitude=magnitude)


def make_model(newton, wp, case, device):
    data = declaration(case)
    builder = newton.ModelBuilder(gravity=(0, 0, 0), up_axis=newton.Axis.Z)
    builder.begin_world(label="C8_" + case)
    cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=STIFFNESS,
        kd=0.0,
        kf=0.0,
        ka=0.0,
        mu=0.0,
        mu_torsional=0.0,
        mu_rolling=0.0,
        margin=0.0,
        gap=0.0,
    )
    bodies, joints, shapes = [], [], []
    for i in range(2):
        pivot = data["pivots"][i]
        body = builder.add_link(
            xform=wp.transform(pivot, wp.quat_identity()),
            mass=data["mass"][i],
            com=(0, 0, 0),
            inertia=np.diag(data["inertia"][i]),
            lock_inertia=True,
            label=f"{case}_body_{i}",
        )
        method = builder.add_joint_prismatic if case == "D01" else builder.add_joint_revolute
        joint = method(
            -1,
            body,
            parent_xform=wp.transform(pivot, wp.quat_identity()),
            child_xform=wp.transform(),
            axis=data["axis"],
            target_pos=0.0,
            target_vel=0.0,
            target_ke=0.0,
            target_kd=0.0,
            damping=0.0,
            limit_lower=-4096.0,
            limit_upper=4096.0,
            limit_ke=0.0,
            limit_kd=0.0,
            armature=0.0,
            effort_limit=8.0,
            velocity_limit=4096.0,
            friction=0.0,
            collision_filter_parent=False,
            label=f"{case}_joint_{i}",
        )
        builder.add_articulation([joint], label=f"{case}_root_{i}")
        shape = builder.add_shape_sphere(
            body,
            radius=RADIUS,
            xform=wp.transform(data["local_sphere_center"], wp.quat_identity()),
            cfg=cfg,
            label=f"{case}_sphere_{i}",
        )
        bodies.append(body)
        joints.append(joint)
        shapes.append(shape)
    builder.end_world()
    model = builder.finalize(device=device)
    return model, np.asarray(bodies, int), np.asarray(joints, int), np.asarray(shapes, int)


def native_array(owner, name):
    value = getattr(owner, name, None)
    if value is None:
        return None
    return value.numpy().copy() if hasattr(value, "numpy") else value


def state_record(state):
    return {
        name: native_array(state, name)
        for name in (
            "joint_q",
            "joint_qd",
            "body_q",
            "body_qd",
            "body_f",
            "body_parent_f",
        )
    }


def contact_record(contacts):
    count = int(contacts.rigid_contact_count.numpy()[0])
    if not 0 <= count <= contacts.rigid_contact_max:
        raise RuntimeError("native contact capacity exceeded")
    raw = {}
    for name in CONTACT_FIELDS:
        value = native_array(contacts, name)
        raw[name] = value if value is None else value[:count]
    return dict(
        count=count,
        capacity=contacts.rigid_contact_max,
        rows=raw,
        rigid_contact_force_semantics="Legacy buffer is unused by Featherstone; it is NOT the force observation.",
    )


def reconstruct_rows(model, entry, contacts, actual_twist):
    """Independent reading of native row geometry, never applied back into native run."""
    raw, count = contacts["rows"], contacts["count"]
    owner = model.shape_body.numpy()
    local_com = model.body_com.numpy().astype(float)
    pose = entry["body_q"].astype(float)
    total = np.zeros((model.body_count, 6))
    observations = []
    for i in range(count):
        shape_a, shape_b = (int(raw["rigid_contact_shape" + str(j)][i]) for j in (0, 1))
        body_a, body_b = int(owner[shape_a]), int(owner[shape_b])
        if body_a < 0 or body_b < 0 or body_a == body_b:
            raise ValueError("frozen reference requires two distinct real dynamic owners")
        normal = raw["rigid_contact_normal"][i].astype(float)
        points, com, velocities, supports = [], [], [], []
        for side, body in enumerate((body_a, body_b)):
            R = rotation(pose[body, 3:])
            support = pose[body, :3] + R @ raw["rigid_contact_point" + str(side)][i]
            margin = float(raw["rigid_contact_margin" + str(side)][i])
            point = support + (1 if side == 0 else -1) * margin * normal
            reference = pose[body, :3] + R @ local_com[body]
            velocities.append(
                actual_twist[body, :3] + np.cross(actual_twist[body, 3:], point - reference)
            )
            points.append(point)
            supports.append(support)
            com.append(reference)
        points, com = np.array(points), np.array(com)
        gap = float(normal @ (points[1] - points[0]))
        material = {}
        for name in ("ke", "kd", "kf", "ka", "mu"):
            values = getattr(model, "shape_material_" + name).numpy()
            material[name] = 0.5 * float(values[shape_a] + values[shape_b])
        for field, parameter in (
            ("rigid_contact_stiffness", "ke"),
            ("rigid_contact_damping", "kd"),
            ("rigid_contact_friction", "mu"),
        ):
            values = raw[field]
            override = float(values[i]) if values is not None and len(values) else 0.0
            material[parameter + "_override"] = override
            if override > 0:
                material[parameter] = (
                    material[parameter] * override if parameter == "mu" else override
                )
        if material["kd"] != 0 or material["mu"] != 0 or material["ka"] != 0:
            raise ValueError("native material drifted outside frozen frictionless undamped law")
        magnitude = material["ke"] * max(0.0, -gap)
        forces = np.array((-magnitude * normal, magnitude * normal))
        wrenches = np.column_stack((forces, np.cross(points - com, forces)))
        total[[body_a, body_b]] += wrenches
        observations.append(
            dict(
                shape_ids=(shape_a, shape_b),
                body_ids=(body_a, body_b),
                normal_a_to_b=normal,
                support_points_world_m=supports,
                effective_surface_points_world_m=points,
                com_world_m=com,
                point_velocity_world_m_s=velocities,
                signed_gap_m=gap,
                material=material,
                reconstructed_body_wrenches_world_com=wrenches,
            )
        )
    return total, observations


def run_case(newton, wp, case, channel, device, output):
    path = output / f"{case}-{channel}.json.gz"
    record = dict(
        case=case,
        channel=channel,
        status="running",
        declaration=declaration(case),
        trace=[],
        execution_failure=None,
        first_failed_gate=None,
        units=dict(
            q="D01 m; D03 rad", v="D01 m/s; D03 rad/s", wrench="world N then N*m about each COM"
        ),
        force_observation="Pinned solver.body_f_ext after actual step: original external + native contact, before RNEA; original external is zero in native channel.",
        native_force_buffer="Contacts.rigid_contact_force is unused; Featherstone supports BODY_PARENT_F, not CONTACT_F observables.",
    )
    try:
        model, bodies, joints, shapes = make_model(newton, wp, case, device)
        metadata = {name: native_array(model, name) for name in MODEL_FIELDS}
        metadata.update(
            body_label=model.body_label,
            joint_label=model.joint_label,
            shape_label=model.shape_label,
            body_count=model.body_count,
            joint_count=model.joint_count,
            shape_count=model.shape_count,
            articulation_count=model.articulation_count,
            world_count=model.world_count,
        )
        record.update(
            model=metadata,
            body_rows=bodies,
            joint_rows=joints,
            shape_rows=shapes,
            actual_device=str(model.device),
        )
        assert str(model.device) == device
        assert model.body_count == model.joint_count == model.shape_count == 2
        assert model.world_count == 1 and model.articulation_count == 2
        np.testing.assert_array_equal(model.body_world.numpy(), (0, 0))
        np.testing.assert_array_equal(model.shape_world.numpy(), (0, 0))
        np.testing.assert_array_equal(
            model.shape_type.numpy(), np.full(2, int(newton.GeoType.SPHERE))
        )
        np.testing.assert_array_equal(model.body_mass.numpy(), (2.0, 1.0))
        np.testing.assert_allclose(
            model.body_inertia.numpy(),
            np.array([np.diag(i) for i in declaration(case)["inertia"]]),
            atol=0,
            rtol=0,
        )
        qcols = model.joint_q_start.numpy()[joints].astype(int)
        vcols = model.joint_qd_start.numpy()[joints].astype(int)
        state, candidate, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        control.joint_f.assign(np.zeros(model.joint_dof_count, np.float32))
        solver = newton.solvers.SolverFeatherstone(
            model, angular_damping=0.0, update_mass_matrix_interval=1
        )
        pipeline = newton.CollisionPipeline(
            model, broad_phase="nxn", rigid_contact_max=8, soft_contact_max=0, deterministic=True
        )
        contacts = pipeline.contacts()
        record["initial_state"] = state_record(state)
        record["pipeline"] = dict(
            broad_phase="nxn", contact_matching="disabled", rigid_contact_max=8
        )
        initial_truth = analytic(case, np.zeros(2), np.zeros(2))
        record["frozen_initial_oracle"] = initial_truth
        np.testing.assert_allclose(
            initial_truth["acceleration"], (-1, 2) if case == "D01" else (2, -1), atol=0, rtol=0
        )
        np.testing.assert_allclose(initial_truth["magnitude"], 2.0, atol=0, rtol=0)
        expected_q, expected_v = np.zeros(2), np.zeros(2)
        save_record(path, record)
        for step in range(1, STEPS + 1):
            state.clear_forces()
            applied = np.zeros((model.body_count, 6), np.float32)
            matched_geometry = None
            if channel == "matched":
                applied, matched_geometry = actual_sphere_law(model, state, bodies, shapes)
                state.body_f.assign(applied)
            entry = state_record(state)
            q, v = entry["joint_q"][qcols].astype(float), entry["joint_qd"][vcols].astype(float)
            truth = analytic(case, q, v)
            independent = analytic(case, expected_q, expected_v)
            expected_v = expected_v + H * independent["acceleration"]
            expected_q = expected_q + H * expected_v
            pipeline.collide(state, contacts)
            raw = contact_record(contacts)
            row = dict(
                step=step,
                entry_time_s=(step - 1) * H,
                commit_time_s=step * H,
                entry=entry,
                native_contacts=raw,
                contacts_applied=channel == "native",
                original_external_wrench=applied,
                matched_geometry=matched_geometry,
                independent_entry=truth,
                independent_window_q=expected_q.copy(),
                independent_window_v=expected_v.copy(),
                checks_complete=False,
            )
            record["trace"].append(row)
            solver.step(state, candidate, control, contacts if channel == "native" else None, H)
            actual = state_record(candidate)
            actual_force = solver.body_f_ext.numpy().copy()
            actual_twist = solver.body_qd_fk.numpy().copy()
            raw_force, rows = reconstruct_rows(model, entry, raw, actual_twist)
            q_next, v_next = actual["joint_q"][qcols], actual["joint_qd"][vcols]
            after_truth = analytic(case, q_next, v_next)
            row.update(
                actual=actual,
                native_entry_twist=actual_twist,
                native_applied_body_wrench=actual_force,
                reconstructed_rows=rows,
                reconstructed_native_contact_wrench=raw_force,
                native_joint_qdd=native_array(solver, "joint_qdd"),
                native_mass_matrix=native_array(solver, "H"),
            )
            gate(row, "one_real_dynamic_pair", raw["count"], 1, atol=0, rtol=0)
            gate(row, "entry_native_FK_position", entry["body_q"][bodies, :3], truth["origins"])
            gate(
                row,
                "entry_native_FK_rotation",
                [rotation(p[3:]) for p in entry["body_q"][bodies]],
                truth["rotations"],
            )
            gate(row, "actual_precontact_COM_twist", actual_twist[bodies], truth["twists"])
            gate(
                row,
                "native_force_reconstruction",
                actual_force,
                raw_force if channel == "native" else applied,
            )
            gate(row, "same_entry_physical_wrench", actual_force[bodies], truth["wrench"])
            gate(row, "same_entry_velocity", v_next, v + H * truth["acceleration"])
            gate(row, "semi_implicit_position", q_next, q + H * v_next)
            gate(row, "independent_40step_position", q_next, expected_q, atol=WINDOW_ATOL)
            gate(row, "independent_40step_velocity", v_next, expected_v, atol=WINDOW_ATOL)
            gate(
                row,
                "accepted_native_FK_position",
                actual["body_q"][bodies, :3],
                after_truth["origins"],
            )
            gate(
                row,
                "accepted_native_FK_rotation",
                [rotation(p[3:]) for p in actual["body_q"][bodies]],
                after_truth["rotations"],
            )
            gate(row, "accepted_COM_twist", actual["body_qd"][bodies], after_truth["twists"])
            gate(row, "internal_force_sum", actual_force[bodies, :3].sum(0), np.zeros(3))
            total_moment = (
                actual_force[bodies, 3:] + np.cross(truth["origins"], actual_force[bodies, :3])
            ).sum(0)
            gate(row, "internal_world_moment_sum", total_moment, np.zeros(3))
            if case == "D01":
                gate(row, "free_axis_total_momentum", 2 * v_next[0] + v_next[1], 0.0)
            native_work = 0.0
            for observed in rows:
                ia, ib = observed["body_ids"]
                a_index, b_index = (int(np.flatnonzero(bodies == b)[0]) for b in (ia, ib))
                gate(
                    row,
                    f"row_{ia}_{ib}_normal",
                    observed["normal_a_to_b"],
                    truth["normal"] if a_index == 0 else -truth["normal"],
                )
                gate(row, f"row_{ia}_{ib}_gap", observed["signed_gap_m"], truth["gap"])
                gate(
                    row,
                    f"row_{ia}_{ib}_surface_points",
                    observed["effective_surface_points_world_m"],
                    truth["points"][[a_index, b_index]],
                )
                for side, body in enumerate((ia, ib)):
                    point = observed["effective_surface_points_world_m"][side]
                    before_pose, after_pose = entry["body_q"][body], actual["body_q"][body]
                    local = np.linalg.solve(rotation(before_pose[3:]), point - before_pose[:3])
                    displacement = after_pose[:3] + rotation(after_pose[3:]) @ local - point
                    native_work += float(
                        observed["reconstructed_body_wrenches_world_com"][side, :3] @ displacement
                    )
            row["energy"] = dict(
                entry_kinetic_j=truth["kinetic_energy_j"],
                accepted_kinetic_j=after_truth["kinetic_energy_j"],
                native_row_finite_material_work_j=native_work,
                entry_power_times_h_j=float(np.sum(actual_force[bodies] * actual_twist[bodies]))
                * H,
                note="Finite material work, entry power*h and kinetic increment are separate observations, not an asserted conservation identity.",
            )
            row["checks_complete"] = True
            for name, check in row["checks"].items():
                if not check["passed"] and record["first_failed_gate"] is None:
                    record["first_failed_gate"] = dict(step=step, name=name, **check)
            save_record(path, record)
            state, candidate = candidate, state
        record["status"] = "passed" if record["first_failed_gate"] is None else "failed"
    except Exception as error:
        record["status"] = "failed"
        record["execution_failure"] = dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
        )
    save_record(path, record)
    return dict(
        case=case,
        channel=channel,
        status=record["status"],
        path=str(path),
        completed_steps=sum(row.get("checks_complete", False) for row in record["trace"]),
        first_failed_gate=record["first_failed_gate"],
        execution_failure=record["execution_failure"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=("D01", "D03"))
    parser.add_argument("--channel", choices=("native", "matched"), default="native")
    args = parser.parse_args()
    import newton
    import warp as wp

    versions = {name: version(name) for name in ("newton", "warp-lang", "numpy")}
    if versions["newton"] != "1.7.0" or versions["warp-lang"] != "1.17.0":
        raise ValueError(f"required pinned Newton1.7/Warp1.17: {versions}")
    if args.device == "cuda:0" and not wp.is_cuda_available():
        raise RuntimeError("requested CUDA is unavailable; no CPU fallback")
    if any(name == "phymeo" or name.startswith("phymeo.") for name in sys.modules):
        raise RuntimeError("reference must not import Phymeo")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(newton.__file__).resolve().parent
    sources = {}
    for name in SOURCES:
        path = root / name
        content = path.read_bytes()
        destination = args.output_dir / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        sources[name] = dict(
            path=str(path), snapshot=str(destination), sha256=hashlib.sha256(content).hexdigest()
        )
    driver = Path(__file__).resolve()
    content = driver.read_bytes()
    (args.output_dir / "source_snapshot" / driver.name).write_bytes(content)
    sources["driver"] = dict(path=str(driver), sha256=hashlib.sha256(content).hexdigest())
    summary = dict(
        versions=versions,
        source_snapshot=sources,
        requested_device=args.device,
        channel=args.channel,
        results=[],
        status="running",
        raw_exit=None,
        gates=dict(atol=ATOL, rtol=RTOL, window_atol=WINDOW_ATOL, h_s=H, steps=STEPS),
        limits="Native single sphere-pair contact with two dynamic articulations; no manifold-weight equivalence, friction, CCD, C8 shared-ancestor or affine-reference claim.",
    )
    with wp.ScopedDevice(args.device):
        for case in args.case or ("D01", "D03"):
            summary["results"].append(
                run_case(newton, wp, case, args.channel, args.device, args.output_dir)
            )
            (args.output_dir / "summary.json").write_text(
                json.dumps(json_value(summary), indent=2, allow_nan=False), encoding="utf-8"
            )
    summary["source_after"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES
    }
    summary["sources_unchanged"] = (
        all(summary["source_after"][name] == sources[name]["sha256"] for name in SOURCES)
        and hashlib.sha256(driver.read_bytes()).hexdigest() == sources["driver"]["sha256"]
    )
    summary["status"] = (
        "passed"
        if summary["sources_unchanged"]
        and all(row["status"] == "passed" for row in summary["results"])
        else "failed"
    )
    summary["raw_exit"] = int(summary["status"] != "passed")
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_value(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(json_value(summary), allow_nan=False))
    return summary["raw_exit"]


if __name__ == "__main__":
    raise SystemExit(main())
