"""Newton 1.7 native FREE-tree contacts: frozen F01/F03, without matched forces.

This standalone reference owns its native states and advances both cases for
40 steps. Numerical checks never feed forces or states back into Newton.
Authoring status: UNEXECUTED. Execution belongs to the parent serial runner.
"""

import argparse
import gzip
import hashlib
import json
import math
import platform
import sys
import traceback
from importlib.metadata import version
from pathlib import Path

import numpy as np

H, STEPS = 1 / 1024, 40
RADIUS, STIFFNESS = 1 / 8, 64.0
ATOL, RTOL, MASS_ATOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-5, 2e-4
MAX_RECORD_BYTES = 128 * 1024 * 1024
CASES = ("F01", "F03")
SOURCES = (
    "_src/sim/builder.py",
    "_src/sim/model.py",
    "_src/sim/state.py",
    "_src/sim/articulation.py",
    "_src/sim/collide.py",
    "_src/sim/contacts.py",
    "_src/geometry/broad_phase_common.py",
    "_src/geometry/collision_primitive.py",
    "_src/geometry/contact_data.py",
    "_src/geometry/narrow_phase.py",
    "_src/solvers/semi_implicit/kernels_contact.py",
    "_src/solvers/featherstone/solver_featherstone.py",
    "_src/solvers/featherstone/kernels.py",
)
MODEL_FIELDS = (
    "body_mass", "body_inv_mass", "body_com", "body_inertia",
    "body_inv_inertia", "body_flags", "body_world",
    "joint_type", "joint_parent", "joint_child", "joint_X_p", "joint_X_c",
    "joint_axis", "joint_dof_dim", "joint_q_start", "joint_qd_start",
    "joint_armature", "joint_damping", "joint_friction", "joint_target_ke",
    "joint_target_kd", "joint_limit_ke", "joint_limit_kd", "joint_limit_lower",
    "joint_limit_upper", "joint_enabled", "joint_world", "joint_articulation",
    "articulation_start", "articulation_end", "gravity",
    "shape_type", "shape_body", "shape_world", "shape_transform", "shape_scale",
    "shape_flags", "shape_margin", "shape_gap", "shape_collision_group",
    "shape_collision_filter_pairs", "shape_contact_pairs", "shape_material_ke",
    "shape_material_kd", "shape_material_kf", "shape_material_ka",
    "shape_material_mu", "shape_material_mu_rolling", "shape_material_mu_torsional",
)
CONTACT_FIELDS = (
    "rigid_contact_point_id", "rigid_contact_shape0", "rigid_contact_shape1",
    "rigid_contact_point0", "rigid_contact_point1", "rigid_contact_offset0",
    "rigid_contact_offset1", "rigid_contact_normal", "rigid_contact_margin0",
    "rigid_contact_margin1", "rigid_contact_surface_velocity",
    "rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction",
    "rigid_contact_tids", "rigid_contact_force",
)
AUX_FIELDS = (
    "joint_qdd", "joint_tau", "joint_qd_internal_in", "joint_qd_internal_out",
    "joint_f_internal", "joint_S_s", "body_qd_fk", "body_q_com",
    "body_solve_origin", "body_I_s", "body_v_s", "body_a_s", "body_f_s",
    "body_f_ext", "body_ft_s", "H",
)
STATE_FIELDS = ("joint_q", "joint_qd", "body_q", "body_qd", "body_f", "body_parent_f")
CONTROL_FIELDS = ("joint_f", "joint_target_q", "joint_target_qd", "joint_act")
MATERIAL = dict(
    density=0.0, ke=STIFFNESS, kd=0.0, kf=0.0, ka=0.0, mu=0.0,
    mu_rolling=0.0, mu_torsional=0.0, margin=0.0, gap=0.0,
    collision_group=1, collision_filter_parent=True, has_shape_collision=True,
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
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def save_record(path, record):
    data = json.dumps(json_value(record), allow_nan=False, separators=(",", ":")).encode()
    if len(data) > MAX_RECORD_BYTES:
        raise RuntimeError("predeclared 128 MiB per-case uncompressed limit exceeded")
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wb") as stream:
        stream.write(data)
    temporary.replace(path)
    return len(data)


def check(record, name, actual, expected, *, atol=ATOL, rtol=RTOL):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    valid = a.shape == b.shape and bool(np.isfinite(a).all() and np.isfinite(b).all())
    error = np.abs(a - b) if valid else np.array([np.inf])
    passed = valid and bool(np.all(error <= atol + rtol * np.abs(b)))
    record.setdefault("checks", {})[name] = dict(
        passed=passed, max_abs_error=float(error.max(initial=0)),
        atol=atol, rtol=rtol, actual_shape=a.shape, expected_shape=b.shape,
    )
    return passed


def rotation(q):
    x, y, z, w = np.asarray(q, float)
    return np.array((
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    ))


def skew(v):
    x, y, z = v
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def declaration(case):
    if case not in CASES:
        raise ValueError("only the two preregistered native cases are allowed")
    f01 = case == "F01"
    return dict(
        case=case, h_s=H, steps=STEPS, gravity_m_s2=(0.0, 0.0, 0.0),
        root_origin_m=(0.0, 0.0, 3 / 32) if f01 else (0.0, 0.0, 0.0),
        root_xyzw=(0.0, 0.0, 0.0, 1.0),
        free_parent_anchor_xyz=(0.0, 0.0, 0.0),
        free_parent_anchor_xyzw=(0.0, 0.0, 0.0, 1.0),
        mass_kg=np.array((2.0,) if f01 else (2.0, 2.0, 1.0)),
        inertia_diagonal_kg_m2=np.array(
            ((0.5, 0.5, 0.5),) if f01
            else ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), (0.25, 0.25, 0.25))
        ),
        local_com_m=np.zeros((1 if f01 else 3, 3)),
        joint_offsets_m=np.zeros((0, 3)) if f01 else np.array(
            ((-7 / 64, 0.0, 0.0), (7 / 64, 0.0, 0.0))
        ),
        joint_axis=(1.0, 0.0, 0.0), joint_initial_q=np.zeros(0 if f01 else 2),
        initial_velocity=np.zeros(6 if f01 else 8),
        sphere_radius_m=RADIUS,
        static_box=None if not f01 else dict(
            owner=-1, center_m=(0.0, 0.0, -1 / 8), half_extents_m=(1.0, 1.0, 1 / 8)
        ),
        material=MATERIAL.copy(), angular_damping=0.0,
        all_external_wrenches_n_nm=np.zeros((1 if f01 else 3, 6)),
        all_joint_efforts=np.zeros(6 if f01 else 8),
        free_twist="public parent-anchor basis, linear COM first; angular second",
        body_twist="world linear COM first; world angular second",
        channel="native_contact_only_no_matched_wrench",
        root_collision_shape=f01, direct_parent_filter=True, free_to_world_filter=False,
    )


def oracle(case, coordinates, rates):
    """Independent translational recurrence and full J/M from literal body data.

    These fixtures retain identity orientation and zero angular speed. This is
    not an oracle for general FREE rotation or for native quaternion integration.
    """
    data = declaration(case)
    q, v = np.asarray(coordinates, float), np.asarray(rates, float)
    b, d = (1, 6) if case == "F01" else (3, 8)
    origins = np.zeros((b, 3))
    acceleration, velocity = np.zeros(d), np.zeros(d)
    wrench = np.zeros((b, 6))
    if case == "F01":
        origins[0, 2] = q[0]
        gap = float(q[0] - RADIUS)
        magnitude = STIFFNESS * max(0.0, -gap)
        wrench[0, 2], acceleration[2], velocity[2] = magnitude, magnitude / 2, v[0]
        coordinate_acceleration = np.array((magnitude / 2,))
        surfaces = np.array(((0.0, 0.0, q[0] - RADIUS), (0.0, 0.0, 0.0)))
        role_forces = np.array(((0.0, 0.0, magnitude), (0.0, 0.0, -magnitude)))
        role_normal = np.array((0.0, 0.0, -1.0))
    else:
        origins[1:] = data["joint_offsets_m"]
        origins[1:, 0] += q
        gap = float(-1 / 32 + q[1] - q[0])
        magnitude = STIFFNESS * max(0.0, -gap)
        wrench[1, 0], wrench[2, 0] = -magnitude, magnitude
        acceleration[6:], velocity[6:] = (-magnitude / 2, magnitude), v
        coordinate_acceleration = np.array((-magnitude / 2, magnitude))
        surfaces = origins[1:].copy()
        surfaces[:, 0] += (RADIUS, -RADIUS)
        role_forces = np.array(((-magnitude, 0.0, 0.0), (magnitude, 0.0, 0.0)))
        role_normal = np.array((1.0, 0.0, 0.0))
    jacobian = np.zeros((b, 6, d))
    for i in range(b):
        jacobian[i, :3, :3] = np.eye(3)
        jacobian[i, 3:, 3:6] = np.eye(3)
        jacobian[i, :3, 3:6] = -skew(origins[i] - origins[0])
        if i:
            jacobian[i, 0, 5 + i] = 1.0
    mass = np.zeros((d, d))
    for i in range(b):
        linear, angular = jacobian[i, :3], jacobian[i, 3:]
        mass += data["mass_kg"][i] * linear.T @ linear
        mass += angular.T @ np.diag(data["inertia_diagonal_kg_m2"][i]) @ angular
    generalized = np.einsum("bkd,bk->d", jacobian, wrench)
    twists = np.einsum("bkd,d->bk", jacobian, velocity)
    momentum = np.sum(data["mass_kg"][:, None] * twists[:, :3], axis=0)
    return dict(
        coordinates=q.copy(), rates=v.copy(), origins_m=origins,
        xyzw=np.tile((0.0, 0.0, 0.0, 1.0), (b, 1)),
        gap_m=gap, normal_force_n=magnitude, role_normal_a_to_b=role_normal,
        role_surfaces_m=surfaces, role_forces_n=role_forces,
        body_wrench_world_com=wrench, generalized_force=generalized,
        acceleration=acceleration, coordinate_acceleration=coordinate_acceleration,
        full_velocity=velocity, body_twists=twists, full_jacobian=jacobian, full_mass=mass,
        linear_momentum_kg_m_s=momentum, kinetic_energy_j=float(velocity @ mass @ velocity / 2),
        potential_energy_j=float(STIFFNESS * max(0.0, -gap) ** 2 / 2),
    )


def make_model(newton, wp, case, device):
    data = declaration(case)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    builder.begin_world(label="C10_native_" + case)
    cfg = newton.ModelBuilder.ShapeConfig(**MATERIAL)
    root = builder.add_link(
        xform=wp.transform(data["root_origin_m"], wp.quat_identity()),
        mass=2.0, com=(0.0, 0.0, 0.0), inertia=np.eye(3) / 2,
        lock_inertia=True, label="root",
    )
    free = builder.add_joint_free(
        root, parent_xform=wp.transform(), child_xform=wp.transform(),
        collision_filter_parent=False, label="free",
    )
    bodies, joints, shapes = [root], [free], []
    if case == "F03":
        for i in range(2):
            offset = tuple(data["joint_offsets_m"][i])
            child = builder.add_link(
                xform=wp.transform(offset, wp.quat_identity()), mass=float(data["mass_kg"][i + 1]),
                com=(0.0, 0.0, 0.0), inertia=np.eye(3) / 4,
                lock_inertia=True, label=("left", "right")[i],
            )
            joint = builder.add_joint_prismatic(
                root, child, parent_xform=wp.transform(offset, wp.quat_identity()),
                child_xform=wp.transform(), axis=(1.0, 0.0, 0.0),
                target_pos=0.0, target_vel=0.0, target_ke=0.0, target_kd=0.0,
                damping=0.0, armature=0.0, friction=0.0,
                limit_lower=-4096.0, limit_upper=4096.0, limit_ke=0.0, limit_kd=0.0,
                collision_filter_parent=True, label=("left_px", "right_px")[i],
            )
            bodies.append(child)
            joints.append(joint)
            shapes.append(builder.add_shape_sphere(
                child, radius=RADIUS, xform=wp.transform(), cfg=cfg,
                label=("left_sphere", "right_sphere")[i],
            ))
    builder.add_articulation(joints, label="one_free_articulation")
    if case == "F01":
        shapes.append(builder.add_shape_sphere(
            root, radius=RADIUS, xform=wp.transform(), cfg=cfg, label="root_sphere",
        ))
        shapes.append(builder.add_shape_box(
            -1, xform=wp.transform((0.0, 0.0, -1 / 8), wp.quat_identity()),
            hx=1.0, hy=1.0, hz=1 / 8, cfg=cfg, label="finite_support",
        ))
    builder.end_world()
    model = builder.finalize(device=device, requires_grad=False)
    state = model.state()
    state.joint_q.assign(model.joint_q.numpy())
    state.joint_qd.assign(np.zeros(model.joint_dof_count, np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state, dict(bodies=bodies, joints=joints, shapes=shapes)


def native_array(owner, name):
    if name == "shape_collision_filter_pairs":
        return owner.shape_collision_filter_pairs_array().copy()
    value = getattr(owner, name, None)
    return value.numpy().copy() if hasattr(value, "numpy") else value


def metadata(model):
    result = {name: native_array(model, name) for name in MODEL_FIELDS}
    result.update(
        body_label=list(model.body_label), joint_label=list(model.joint_label),
        shape_label=list(model.shape_label), body_count=model.body_count,
        joint_count=model.joint_count, joint_dof_count=model.joint_dof_count,
        joint_coord_count=model.joint_coord_count, shape_count=model.shape_count,
        articulation_count=model.articulation_count, world_count=model.world_count,
        actual_device=str(model.device),
    )
    return result


def observe(newton, model, state):
    raw = {name: native_array(state, name) for name in STATE_FIELDS}
    b, d = model.body_count, model.joint_dof_count
    public_j = newton.eval_jacobian(model, state)
    public_m = newton.eval_mass_matrix(model, state, J=public_j)
    native_j = public_j.numpy()[0].astype(float).reshape(b, 6, d)
    native_m = public_m.numpy()[0].astype(float)
    # Both frozen FREE parent anchors are identity. Keep the basis explicit.
    world_to_native_velocity = np.eye(d)
    full_velocity = np.r_[raw["body_qd"][0], raw["joint_qd"][6:]].astype(float)
    masses = model.body_mass.numpy().astype(float)
    inertia = model.body_inertia.numpy().astype(float)
    local_com = model.body_com.numpy().astype(float)
    centers, linear, angular, energy = [], np.zeros(3), np.zeros(3), 0.0
    for i in range(b):
        pose, twist = raw["body_q"][i].astype(float), raw["body_qd"][i].astype(float)
        R = rotation(pose[3:])
        center = pose[:3] + R @ local_com[i]
        momentum = masses[i] * twist[:3]
        spin = R @ inertia[i] @ R.T @ twist[3:]
        centers.append(center)
        linear += momentum
        angular += np.cross(center, momentum) + spin
        energy += (twist[:3] @ momentum + twist[3:] @ spin) / 2
    return dict(
        raw=raw, raw_dtypes={k: str(v.dtype) for k, v in raw.items() if isinstance(v, np.ndarray)},
        actual_device=str(state.body_q.device), centers_world_m=np.array(centers),
        public_native_jacobian=native_j, public_native_mass=native_m,
        world_to_native_velocity=world_to_native_velocity,
        jacobian_world_com=native_j @ world_to_native_velocity,
        mass_world_com=world_to_native_velocity.T @ native_m @ world_to_native_velocity,
        full_velocity_world_com=full_velocity, linear_momentum_kg_m_s=linear,
        angular_momentum_world_origin_kg_m2_s=angular, kinetic_energy_j=float(energy),
    )


def coordinates(case, observation):
    raw = observation["raw"]
    if case == "F01":
        return (
            np.array((raw["body_q"][0, 2],), float),
            np.array((raw["body_qd"][0, 2],), float),
        )
    return raw["joint_q"][7:].astype(float), raw["joint_qd"][6:].astype(float)


def contact_record(contacts):
    count = int(contacts.rigid_contact_count.numpy()[0])
    capacity = int(contacts.rigid_contact_max)
    rows = {}
    for name in CONTACT_FIELDS:
        value = native_array(contacts, name)
        rows[name] = value if value is None else value[: min(max(count, 0), capacity)]
    return dict(
        count=count, capacity=capacity, valid_capacity=0 <= count <= capacity, rows=rows,
        force_buffer_semantics="rigid_contact_force is unused by Featherstone, not a measured wrench",
        offset_semantics="raw friction-anchor offsets retained; not added again to normal-law surfaces",
    )


def inspect_solver(solver, entry):
    raw = {name: native_array(solver, name) for name in AUX_FIELDS}
    source = raw["body_f_ext"].astype(float)
    force = -source[:, :3]
    centers = raw["body_q_com"][:, :3].astype(float)
    solve_origin = raw["body_solve_origin"].astype(float)
    torque = -source[:, 3:] - np.cross(centers - solve_origin, force)
    acceleration = raw["joint_qdd"].astype(float).copy()
    # Native FREE internal acceleration is about the parent anchor. Frozen
    # anchors are identity; expose the public COM acceleration explicitly.
    omega = entry["full_velocity_world_com"][3:6]
    linear = entry["full_velocity_world_com"][:3]
    acceleration[:3] += np.cross(acceleration[3:6], entry["centers_world_m"][0])
    acceleration[:3] += np.cross(omega, linear)
    return dict(
        raw=raw, raw_dtypes={k: str(v.dtype) for k, v in raw.items() if isinstance(v, np.ndarray)},
        decoded_com_world_wrench=np.column_stack((force, torque)),
        decoded_public_world_com_acceleration=acceleration,
        decoder="F=-raw[:3]; tauCOM=-raw[3:]-(body_q_com.xyz-body_solve_origin) cross F",
        timing="entry mechanics scratch, read immediately after this real native step",
        body_parent_f_semantics="recursive net joint wrench, not a contact-only sensor",
    )


def reconstruct_rows(meta, entry, raw_contacts, entry_twist):
    """Read native contacts independently; never use this output as an input."""
    total = np.zeros((meta["body_count"], 6))
    static, rows = {}, []
    poses, coms = entry["raw"]["body_q"].astype(float), entry["centers_world_m"]
    raw = raw_contacts["rows"]
    for i in range(raw_contacts["count"]):
        shapes = tuple(int(raw["rigid_contact_shape" + str(side)][i]) for side in (0, 1))
        if any(s < 0 or s >= meta["shape_count"] for s in shapes):
            raise ValueError("invalid native shape id in completed contact row")
        owners = tuple(int(meta["shape_body"][s]) for s in shapes)
        normal = raw["rigid_contact_normal"][i].astype(float)
        supports, points, references, velocities, margins = [], [], [], [], []
        for side, body in enumerate(owners):
            point = raw["rigid_contact_point" + str(side)][i].astype(float)
            margin = float(raw["rigid_contact_margin" + str(side)][i])
            if body >= 0:
                support = poses[body, :3] + rotation(poses[body, 3:]) @ point
                point = support + (1 if side == 0 else -1) * margin * normal
                reference = coms[body]
                speed = entry_twist[body, :3] + np.cross(entry_twist[body, 3:], point - reference)
            else:
                # Static points are already world-space in this pin. The
                # frozen static box has zero effective radius and margin.
                support = point.copy()
                reference = meta["shape_transform"][shapes[side], :3].astype(float)
                speed = np.zeros(3)
            supports.append(support)
            points.append(point)
            references.append(reference)
            velocities.append(speed)
            margins.append(margin)
        points, references = np.asarray(points), np.asarray(references)
        gap = float(normal @ (points[1] - points[0]))
        material = {
            name: float(sum(meta["shape_material_" + name][s] for s in shapes) / 2)
            for name in ("ke", "kd", "kf", "ka", "mu")
        }
        for field, parameter in (
            ("rigid_contact_stiffness", "ke"), ("rigid_contact_damping", "kd"),
            ("rigid_contact_friction", "mu"),
        ):
            values = raw[field]
            override = float(values[i]) if values is not None else 0.0
            material[parameter + "_override"] = override
            if override > 0:
                material[parameter] = (
                    material[parameter] * override if parameter == "mu" else override
                )
        magnitude = material["ke"] * max(0.0, -gap)
        forces = np.asarray((-magnitude * normal, magnitude * normal))
        wrenches = np.column_stack((forces, np.cross(points - references, forces)))
        for side, body in enumerate(owners):
            if body >= 0:
                total[body] += wrenches[side]
            else:
                key = shapes[side]
                reaction = static.setdefault(key, dict(
                    reference_point_world_m=references[side].copy(),
                    force_n=np.zeros(3), torque_nm=np.zeros(3),
                    provenance="row-derived, not a static body force buffer",
                ))
                reaction["force_n"] += wrenches[side, :3]
                reaction["torque_nm"] += wrenches[side, 3:]
        rows.append(dict(
            native_row=i, shape_ids=shapes, owner_body_ids=owners,
            normal_a_to_b=normal, raw_support_points_world_m=np.asarray(supports),
            effective_surface_points_world_m=points, effective_margin_m=np.asarray(margins),
            reference_points_world_m=references, point_velocities_world_m_s=np.asarray(velocities),
            signed_gap_m=gap, mixed_material=material, normal_force_n=magnitude,
            force_on_sides_n=forces, wrench_on_sides_about_reference=wrenches,
        ))
    return dict(dynamic_body_wrench_world_com=total, static_reactions=static, rows=rows)


def check_model(record, newton, meta, roles, case):
    data = declaration(case)
    b, d = (1, 6) if case == "F01" else (3, 8)
    check(record, "one_complete_FREE_articulation", meta["articulation_count"], 1, atol=0, rtol=0)
    check(record, "body_and_generalized_counts", (meta["body_count"], meta["joint_dof_count"]), (b, d), atol=0, rtol=0)
    check(record, "authored_mass", meta["body_mass"], data["mass_kg"], atol=0, rtol=0)
    check(record, "authored_COM", meta["body_com"], data["local_com_m"], atol=0, rtol=0)
    check(record, "authored_inertia", meta["body_inertia"],
          [np.diag(row) for row in data["inertia_diagonal_kg_m2"]], atol=0, rtol=0)
    check(record, "FREE_root_actual_type", meta["joint_type"][0], int(newton.JointType.FREE), atol=0, rtol=0)
    expected_types = [int(newton.GeoType.SPHERE), int(
        newton.GeoType.BOX if case == "F01" else newton.GeoType.SPHERE
    )]
    check(record, "real_shape_types", meta["shape_type"][roles["shapes"]], expected_types, atol=0, rtol=0)
    check(record, "shape_owners", meta["shape_body"][roles["shapes"]],
          (0, -1) if case == "F01" else (1, 2), atol=0, rtol=0)
    check(record, "authored_margin", meta["shape_margin"], (0.0, 0.0), atol=0, rtol=0)
    check(record, "authored_gap", meta["shape_gap"], (0.0, 0.0), atol=0, rtol=0)
    check(record, "shape_collision_groups", meta["shape_collision_group"], (1, 1), atol=0, rtol=0)
    for parameter in ("ke", "kd", "kf", "ka", "mu", "mu_rolling", "mu_torsional"):
        check(record, "material_" + parameter, meta["shape_material_" + parameter],
              (MATERIAL[parameter], MATERIAL[parameter]), atol=0, rtol=0)
    filters = {
        tuple(sorted(map(int, pair))) for pair in np.asarray(meta["shape_collision_filter_pairs"]).reshape(-1, 2)
    }
    check(record, "frozen_pair_is_not_filtered", tuple(sorted(roles["shapes"])) not in filters, True, atol=0, rtol=0)
    if case == "F03":
        check(record, "siblings_share_the_real_root", meta["joint_parent"], (-1, 0, 0), atol=0, rtol=0)
        check(record, "two_Px_actual_type", meta["joint_type"][1:],
              (int(newton.JointType.PRISMATIC),) * 2, atol=0, rtol=0)


def check_observation(row, prefix, actual, expected, *, window=False):
    tol = WINDOW_ATOL if window else ATOL
    check(row, prefix + "_body_origins", actual["raw"]["body_q"][:, :3], expected["origins_m"], atol=tol)
    check(row, prefix + "_body_rotations", actual["raw"]["body_q"][:, 3:], expected["xyzw"], atol=tol)
    check(row, prefix + "_body_COM_twists", actual["raw"]["body_qd"], expected["body_twists"], atol=tol)
    check(row, prefix + "_full_velocity", actual["full_velocity_world_com"], expected["full_velocity"], atol=tol)
    check(row, prefix + "_linear_momentum", actual["linear_momentum_kg_m_s"],
          expected["linear_momentum_kg_m_s"], atol=tol)
    check(row, prefix + "_angular_momentum", actual["angular_momentum_world_origin_kg_m2_s"], np.zeros(3), atol=tol)
    check(row, prefix + "_kinetic_energy", actual["kinetic_energy_j"], expected["kinetic_energy_j"], atol=tol)
    if not window:
        check(row, prefix + "_full_J", actual["jacobian_world_com"], expected["full_jacobian"])
        check(row, prefix + "_full_M", actual["mass_world_com"], expected["full_mass"], atol=MASS_ATOL)
        check(row, prefix + "_public_FREE_basis", actual["raw"]["joint_qd"], actual["full_velocity_world_com"])


def failed_checks(record):
    failed = []
    for label, block in [("initial", record), *(
        (f"step_{row['step']:02d}", row) for row in record["trace"]
    )]:
        failed.extend(
            label + "/" + name for name, result in block.get("checks", {}).items()
            if not result["passed"]
        )
    return failed


def run_case(newton, wp, case, device, output, provenance):
    path = output / (case + "-native.json.gz")
    record = dict(
        case=case, channel="native", status="running", declared_steps=STEPS, h_s=H,
        declaration=declaration(case), source_binding=provenance,
        native_steps_completed=0, complete_steps=0, trace=[], execution_error=None,
        first_failed_check=None, failed_checks=[],
        semantics=dict(
            contact_time="entry state; collide once then native Featherstone step once",
            zero_matched_force=True,
            raw_wrench="post-RNEA negative wrench about saved actual solve origin",
            force_buffer="rigid_contact_force is saved but is not the force observation",
            static_reaction="F01 row-derived force/moment; no native static body wrench buffer",
            orientation="native normalized Euler; only zero-angular-motion equivalence is tested",
        ),
    )
    save_record(path, record)
    try:
        model, state, roles = make_model(newton, wp, case, device)
        meta = metadata(model)
        record.update(model=meta, role_ids=roles)
        check_model(record, newton, meta, roles, case)
        candidate, control = model.state(), model.control()
        control.joint_f.assign(np.zeros(model.joint_dof_count, np.float32))
        solver = newton.solvers.SolverFeatherstone(
            model, angular_damping=0.0, update_mass_matrix_interval=1,
        )
        pipeline = newton.CollisionPipeline(
            model, broad_phase="nxn", rigid_contact_max=8, soft_contact_max=0,
            deterministic=True, contact_matching="disabled",
        )
        contacts = pipeline.contacts()
        record["pipeline"] = dict(
            broad_phase="nxn", rigid_contact_max=8, soft_contact_max=0,
            deterministic=True, contact_matching="disabled",
        )
        record["solver"] = dict(
            name=type(solver).__name__, angular_damping=0.0, update_mass_matrix_interval=1,
            requires_grad=False,
        )
        initial_q = np.array((3 / 32,), float) if case == "F01" else np.zeros(2)
        independent_q, independent_v = initial_q.copy(), np.zeros_like(initial_q)
        first = oracle(case, independent_q, independent_v)
        record["literal_first_step"] = dict(
            gap_m=-1 / 32, normal_force_n=2.0,
            generalized_force=(0.0, 0.0, 2.0, 0.0, 0.0, 0.0) if case == "F01"
            else (0.0,) * 6 + (-2.0, 2.0),
            acceleration=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0) if case == "F01"
            else (0.0,) * 6 + (-1.0, 2.0),
            root_x_joint_mass_block=None if case == "F01"
            else ((5.0, 2.0, 1.0), (2.0, 2.0, 0.0), (1.0, 0.0, 1.0)),
        )
        literal = record["literal_first_step"]
        check(record, "literal_gap", first["gap_m"], literal["gap_m"], atol=0, rtol=0)
        check(record, "literal_N", first["normal_force_n"], 2.0, atol=0, rtol=0)
        check(record, "literal_generalized_force", first["generalized_force"], literal["generalized_force"], atol=0, rtol=0)
        check(record, "literal_acceleration", first["acceleration"], literal["acceleration"], atol=0, rtol=0)
        if case == "F03":
            check(record, "literal_cross_mass_block", first["full_mass"][np.ix_((0, 6, 7), (0, 6, 7))],
                  literal["root_x_joint_mass_block"], atol=0, rtol=0)
        record["initial_state"] = observe(newton, model, state)
        check_observation(record, "initial", record["initial_state"], first)
        for step in range(1, STEPS + 1):
            state.clear_forces()
            entry = observe(newton, model, state)
            actual_q, actual_v = coordinates(case, entry)
            same_entry = oracle(case, actual_q, actual_v)
            independent_entry = oracle(case, independent_q, independent_v)
            independent_v = independent_v + H * independent_entry["coordinate_acceleration"]
            independent_q = independent_q + H * independent_v
            independent_exit = oracle(case, independent_q, independent_v)
            original_control = {name: native_array(control, name) for name in CONTROL_FIELDS}
            row = dict(
                step=step, entry_time_s=(step - 1) * H, exit_time_s=step * H,
                entry=entry, original_control=original_control,
                original_external_body_wrench=entry["raw"]["body_f"].copy(),
                matched_contact_wrench_supplied=False,
                independent_same_entry=same_entry, independent_window_entry=independent_entry,
                independent_window_exit=independent_exit,
                native_step_submitted=False, native_step_completed=False, checks_complete=False,
            )
            record["trace"].append(row)
            check(row, "zero_original_external_wrench", row["original_external_body_wrench"],
                  np.zeros((model.body_count, 6)), atol=0, rtol=0)
            check(row, "zero_original_joint_effort", original_control["joint_f"],
                  np.zeros(model.joint_dof_count), atol=0, rtol=0)
            pipeline.collide(state, contacts)
            raw_contacts = contact_record(contacts)
            row["native_contacts"] = raw_contacts
            save_record(path, record)
            if not raw_contacts["valid_capacity"]:
                raise RuntimeError("actual native contact count exceeds frozen row capacity")
            solver.step(state, candidate, control, contacts, H)
            row["native_step_submitted"] = True
            # Read entry-mechanics scratch before any subsequent step. CUDA
            # submission alone is not evidence that a physical step completed.
            scratch = inspect_solver(solver, entry)
            after = observe(newton, model, candidate)
            row.update(exit=after, native_step_scratch=scratch)
            row["native_step_completed"] = True
            record["native_steps_completed"] = step
            if not all(np.isfinite(after["raw"][name]).all() for name in (
                "joint_q", "joint_qd", "body_q", "body_qd"
            )):
                raise FloatingPointError("native completed state is nonfinite; retained raw prefix")
            rebuilt = reconstruct_rows(meta, entry, raw_contacts, scratch["raw"]["body_qd_fk"])
            row["native_row_reconstruction"] = rebuilt
            q_next, v_next = coordinates(case, after)
            after_same_entry = oracle(case, q_next, v_next)
            actual_force = scratch["decoded_com_world_wrench"]
            actual_generalized = np.einsum("bkd,bk->d", entry["jacobian_world_com"], actual_force)
            row["decoded_contact_generalized_force"] = actual_generalized
            check(row, "one_actual_contact_row", raw_contacts["count"], 1, atol=0, rtol=0)
            check_observation(row, "entry", entry, same_entry)
            check_observation(row, "exit_same_state", after, after_same_entry)
            check_observation(row, "independent_40step", after, independent_exit, window=True)
            check(row, "scratch_real_entry_twists", scratch["raw"]["body_qd_fk"], entry["raw"]["body_qd"])
            check(row, "scratch_real_entry_COM", scratch["raw"]["body_q_com"][:, :3], entry["centers_world_m"])
            check(row, "actual_FREE_solve_origin", scratch["raw"]["body_solve_origin"],
                  np.tile(entry["centers_world_m"][0], (model.body_count, 1)))
            check(row, "decoded_vs_native_rows", actual_force, rebuilt["dynamic_body_wrench_world_com"])
            check(row, "same_entry_literal_wrench", actual_force, same_entry["body_wrench_world_com"])
            check(row, "same_entry_full_generalized_force", actual_generalized, same_entry["generalized_force"])
            check(row, "same_entry_COM_acceleration", scratch["decoded_public_world_com_acceleration"],
                  same_entry["acceleration"])
            check(row, "native_full_mass_equation", entry["mass_world_com"] @ scratch["decoded_public_world_com_acceleration"],
                  same_entry["generalized_force"], atol=MASS_ATOL)
            check(row, "same_entry_velocity_increment", after["full_velocity_world_com"],
                  entry["full_velocity_world_com"] + H * same_entry["acceleration"])
            check(row, "semiimplicit_coordinates", q_next, actual_q + H * v_next)
            check(row, "semiimplicit_root_origin", after["raw"]["body_q"][0, :3],
                  entry["raw"]["body_q"][0, :3] + H * after["raw"]["body_qd"][0, :3])
            for native_row in rebuilt["rows"]:
                prefix = "contact_" + str(native_row["native_row"])
                shape_ids = native_row["shape_ids"]
                check(row, prefix + "_exact_shape_pair", sorted(shape_ids), sorted(roles["shapes"]), atol=0, rtol=0)
                if set(shape_ids) != set(roles["shapes"]):
                    continue
                order = [roles["shapes"].index(s) for s in shape_ids]
                sign = 1.0 if order == [0, 1] else -1.0
                check(row, prefix + "_normal", native_row["normal_a_to_b"], sign * same_entry["role_normal_a_to_b"])
                check(row, prefix + "_physical_surfaces", native_row["effective_surface_points_world_m"],
                      same_entry["role_surfaces_m"][order])
                check(row, prefix + "_physical_gap", native_row["signed_gap_m"], same_entry["gap_m"])
                check(row, prefix + "_force_on_real_owners", native_row["force_on_sides_n"],
                      same_entry["role_forces_n"][order])
                check(row, prefix + "_no_COM_torque", native_row["wrench_on_sides_about_reference"][:, 3:], np.zeros((2, 3)))
                for parameter in ("ke", "kd", "kf", "ka", "mu"):
                    check(row, prefix + "_effective_" + parameter,
                          native_row["mixed_material"][parameter], MATERIAL[parameter], atol=0, rtol=0)
            static_force = np.zeros(3)
            static_moment = np.zeros(3)
            for reaction in rebuilt["static_reactions"].values():
                static_force += reaction["force_n"]
                static_moment += reaction["torque_nm"] + np.cross(
                    reaction["reference_point_world_m"], reaction["force_n"]
                )
            net_force = np.sum(actual_force[:, :3], axis=0) + static_force
            net_moment = np.sum(actual_force[:, 3:] + np.cross(
                entry["centers_world_m"], actual_force[:, :3]
            ), axis=0) + static_moment
            row["net_contact_force_n"] = net_force
            row["net_contact_world_origin_moment_nm"] = net_moment
            check(row, "two_sided_force_balance", net_force, np.zeros(3))
            check(row, "two_sided_world_moment_balance", net_moment, np.zeros(3))
            check(row, "native_linear_impulse_balance",
                  after["linear_momentum_kg_m_s"] - entry["linear_momentum_kg_m_s"],
                  H * np.sum(actual_force[:, :3], axis=0))
            velocity_before = entry["full_velocity_world_com"]
            velocity_after = after["full_velocity_world_com"]
            dv = velocity_after - velocity_before
            discrete_delta_ke = H * actual_generalized @ velocity_after - dv @ entry["mass_world_com"] @ dv / 2
            actual_delta_ke = after["kinetic_energy_j"] - entry["kinetic_energy_j"]
            row["energy"] = dict(
                native_kinetic_delta_j=float(actual_delta_ke),
                force_postvelocity_work_j=float(H * actual_generalized @ velocity_after),
                discrete_kinetic_delta_expected_j=float(discrete_delta_ke),
                same_entry_spring_energy_j=same_entry["potential_energy_j"],
                exit_spring_energy_j=after_same_entry["potential_energy_j"],
                total_energy_change_j=float(actual_delta_ke + after_same_entry["potential_energy_j"] - same_entry["potential_energy_j"]),
                semantics="translational discrete kinetic identity; spring-plus-kinetic need not be exactly conserved",
            )
            check(row, "discrete_kinetic_balance", actual_delta_ke, discrete_delta_ke)
            if step == 1:
                check(row, "literal_first_native_acceleration", scratch["decoded_public_world_com_acceleration"], literal["acceleration"])
                check(row, "literal_first_native_generalized", actual_generalized, literal["generalized_force"])
            row["checks_complete"] = True
            record["complete_steps"] = step
            record["failed_checks"] = failed_checks(record)
            if record["first_failed_check"] is None and record["failed_checks"]:
                record["first_failed_check"] = record["failed_checks"][0]
            state, candidate = candidate, state
            save_record(path, record)
    except Exception as error:
        record["execution_error"] = dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
        )
    record["failed_checks"] = failed_checks(record)
    if record["first_failed_check"] is None and record["failed_checks"]:
        record["first_failed_check"] = record["failed_checks"][0]
    record["status"] = "passed" if (
        record["complete_steps"] == STEPS and record["native_steps_completed"] == STEPS
        and not record["failed_checks"] and record["execution_error"] is None
    ) else "failed"
    uncompressed_bytes = save_record(path, record)
    compressed = path.read_bytes()
    return dict(
        case=case, status=record["status"], declared_steps=STEPS,
        complete_steps=record["complete_steps"], native_steps_completed=record["native_steps_completed"],
        first_failed_check=record["first_failed_check"], failed_checks=record["failed_checks"],
        execution_error=record["execution_error"], output=str(path),
        uncompressed_bytes=uncompressed_bytes, compressed_bytes=len(compressed), compressed_sha256=hashlib.sha256(compressed).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute():
        raise ValueError("--output must be an absolute new directory")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = "cuda:0" if args.device == "cuda" else args.device
    script = Path(__file__).resolve()
    script_bytes = script.read_bytes()
    document = dict(
        status="running", device=device, h_s=H, declared_cases=CASES, steps_per_case=STEPS,
        cases=[], execution_error=None, checks={},
        gates=dict(same_entry_atol=ATOL, rtol=RTOL, mass_atol=MASS_ATOL, window_atol=WINDOW_ATOL),
        limits=dict(per_case_uncompressed_bytes=MAX_RECORD_BYTES, contact_capacity=8),
        script=dict(path=str(script), sha256=hashlib.sha256(script_bytes).hexdigest()),
        invocation=list(sys.argv), no_phy_or_tyx_or_repository_helper=True,
    )
    summary = output / "summary.json"
    try:
        import newton
        import warp as wp

        if version("newton") != "1.7.0" or version("warp-lang") != "1.17.0":
            raise RuntimeError("requires the installed Newton 1.7.0 / Warp 1.17.0 pin")
        document["versions"] = dict(
            newton=version("newton"), warp=version("warp-lang"), numpy=np.__version__,
            python=sys.version, platform=platform.platform(),
        )
        root = Path(newton.__file__).resolve().parent
        document["module_paths"] = dict(newton=str(Path(newton.__file__).resolve()), warp=str(Path(wp.__file__).resolve()))
        sources = {}
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        (snapshot / script.name).write_bytes(script_bytes)
        for name in SOURCES:
            source = root / name
            data = source.read_bytes()
            copy = snapshot / "newton" / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(data)
            sources[name] = dict(
                path=str(source), snapshot=str(copy), size_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        document["sources_before"] = sources
        provenance = dict(
            script_sha256=document["script"]["sha256"],
            newton_sources={name: row["sha256"] for name, row in sources.items()},
            versions=document["versions"], device=device,
        )
        wp.config.kernel_cache_dir = str(output / "warp-cache")
        wp.init()
        with wp.ScopedDevice(device):
            for case in CASES:
                try:
                    document["cases"].append(run_case(newton, wp, case, device, output, provenance))
                except Exception as error:
                    # An observer/IO failure is not physical rollback. Still try
                    # the other independently declared case and fail this run.
                    document["cases"].append(dict(
                        case=case, status="failed", native_steps_completed=None, complete_steps=None,
                        execution_error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()),
                        failure_semantics="runner or persistence failure; inspect the preserved case prefix",
                    ))
        document["sources_after"] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES
        }
        check(document, "all_pinned_sources_unchanged",
              all(document["sources_after"][name] == row["sha256"] for name, row in sources.items()),
              True, atol=0, rtol=0)
        check(document, "driver_unchanged", hashlib.sha256(script.read_bytes()).hexdigest() == document["script"]["sha256"],
              True, atol=0, rtol=0)
        forbidden = sorted(name for name in sys.modules if name.split(".")[0] in ("phymeo", "tyx"))
        document["forbidden_loaded_modules"] = forbidden
        check(document, "no_product_or_TYX_import", len(forbidden), 0, atol=0, rtol=0)
        document["status"] = "passed" if (
            len(document["cases"]) == len(CASES)
            and all(row["status"] == "passed" for row in document["cases"])
            and all(item["passed"] for item in document["checks"].values())
        ) else "failed"
    except Exception as error:
        document["status"] = "failed"
        document["execution_error"] = dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
        )
    summary.write_text(json.dumps(json_value(document), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(json_value(dict(
        status=document["status"], device=device, h_s=H, cases=document["cases"],
        execution_error=document["execution_error"], summary=str(summary),
    )), allow_nan=False))
    return 0 if document["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
