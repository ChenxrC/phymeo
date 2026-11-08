"""Pinned Newton native FREE + affine mimic short references (UNEXECUTED).

No Phymeo/TYX imports, matched contact force, state replay or solver fallback.
The adjacent independent NumPy time-jet/full-KKT reference is loaded explicitly.
Only the parent's serial runner may execute this authoring delivery.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import platform
import sys
import traceback
from importlib.metadata import version
from pathlib import Path

import numpy as np

H = 1 / 1024
ATOL, RTOL, WINDOW_ATOL, ACCEL_ATOL = 5e-6, 5e-5, 2e-4, 3e-4
RAW_LIMIT, GZIP_LIMIT = 256 * 1024 * 1024, 64 * 1024 * 1024
CASES = (
    ("A01-motor", "A01", None, 1),
    ("A03-fan", "A03", "fan", 4),
    ("A05-native-contact", "A05", None, 4),
)
REFERENCE_NAME = "resident_floating_affine_reference.py"
BASE_REFERENCE_NAME = "resident_floating_reference.py"
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
    "body_mass",
    "body_inv_mass",
    "body_com",
    "body_inertia",
    "body_inv_inertia",
    "body_flags",
    "body_world",
    "joint_type",
    "joint_parent",
    "joint_child",
    "joint_X_p",
    "joint_X_c",
    "joint_axis",
    "joint_dof_dim",
    "joint_q_start",
    "joint_qd_start",
    "joint_armature",
    "joint_damping",
    "joint_friction",
    "joint_target_ke",
    "joint_target_kd",
    "joint_limit_ke",
    "joint_limit_kd",
    "joint_limit_lower",
    "joint_limit_upper",
    "joint_enabled",
    "joint_world",
    "joint_articulation",
    "articulation_start",
    "articulation_end",
    "gravity",
    "shape_type",
    "shape_body",
    "shape_world",
    "shape_transform",
    "shape_scale",
    "shape_flags",
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
    "shape_material_mu_rolling",
    "shape_material_mu_torsional",
)
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
AUX_FIELDS = (
    "joint_qdd",
    "joint_tau",
    "joint_qd_internal_in",
    "joint_qd_internal_out",
    "joint_f_internal",
    "joint_S_s",
    "body_qd_fk",
    "body_q_com",
    "body_solve_origin",
    "body_I_s",
    "body_v_s",
    "body_a_s",
    "body_f_s",
    "body_f_ext",
    "body_ft_s",
    "H",
)
STATE_FIELDS = ("joint_q", "joint_qd", "body_q", "body_qd", "body_f", "body_parent_f")
CONTROL_FIELDS = ("joint_f", "joint_target_q", "joint_target_qd", "joint_act")
SOURCES += ("_src/sim/joint_mimic.py", "_src/sim/control.py", "_src/sim/enums.py")
MODEL_FIELDS += (
    "joint_mimic_joint",
    "joint_mimic_coeffs",
    "joint_effort_limit",
    "joint_velocity_limit",
    "joint_target_mode",
    "joint_target_q_start",
)
AUX_FIELDS += (
    "J",
    "P",
    "L",
    "M",
    "H_mimic",
    "joint_tau_mimic",
    "articulation_J_start",
    "articulation_J_rows",
    "articulation_J_cols",
    "articulation_H_start",
    "articulation_H_rows",
    "articulation_dof_start",
    "articulation_M_start",
    "articulation_M_rows",
    "joint_armature_effective",
    "joint_armature_zero",
)
MATERIAL = dict(
    density=0.0,
    ke=64.0,
    kd=0.0,
    kf=0.0,
    ka=0.0,
    mu=0.0,
    mu_rolling=0.0,
    mu_torsional=0.0,
    margin=0.0,
    gap=0.0,
    collision_group=1,
    collision_filter_parent=True,
    has_shape_collision=True,
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


def rotation(q):
    x, y, z, w = np.asarray(q, float)
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def skew(v):
    x, y, z = v
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def native_array(owner, name):
    if name == "shape_collision_filter_pairs":
        return owner.shape_collision_filter_pairs_array().copy()
    value = getattr(owner, name, None)
    return value.numpy().copy() if hasattr(value, "numpy") else value


def metadata(model):
    result = {name: native_array(model, name) for name in MODEL_FIELDS}
    result.update(
        body_label=list(model.body_label),
        joint_label=list(model.joint_label),
        shape_label=list(model.shape_label),
        body_count=model.body_count,
        joint_count=model.joint_count,
        joint_dof_count=model.joint_dof_count,
        joint_coord_count=model.joint_coord_count,
        shape_count=model.shape_count,
        articulation_count=model.articulation_count,
        world_count=model.world_count,
        actual_device=str(model.device),
    )
    return result


def save_record(path, record):
    """Bounded complete JSON/gzip; an IO failure never means physical rollback."""
    temporary = path.with_name(path.name + ".tmp")
    size = 0
    encoder = json.JSONEncoder(allow_nan=False, separators=(",", ":"))
    with gzip.open(temporary, "wb") as stream:
        for part in encoder.iterencode(json_value(record)):
            chunk = part.encode("utf-8")
            size += len(chunk)
            if size > RAW_LIMIT:
                raise RuntimeError("C11 native record exceeds frozen 256 MiB decoded limit")
            stream.write(chunk)
    if temporary.stat().st_size > GZIP_LIMIT:
        raise RuntimeError("C11 native record exceeds frozen 64 MiB compressed limit")
    temporary.replace(path)
    return size


def check(record, name, actual, expected, *, atol=ATOL, rtol=RTOL):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    valid = a.shape == b.shape and bool(np.isfinite(a).all() and np.isfinite(b).all())
    error = np.abs(a - b) if valid else np.array([np.inf])
    passed = valid and bool(np.all(error <= atol + rtol * np.abs(b)))
    record.setdefault("checks", {})[name] = dict(
        passed=passed,
        actual=a.copy(),
        expected=b.copy(),
        atol=atol,
        rtol=rtol,
        max_abs_error=float(error.max(initial=0)),
        actual_shape=a.shape,
        expected_shape=b.shape,
    )
    return passed


def load_reference(path):
    spec = importlib.util.spec_from_file_location("c11_native_independent_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def declared_limits(data):
    if data["name"] == "A05":
        return [(-1.0, 1.0, 1.0, 8.0)] * 2
    limits = [(-0.032, 0.032, 0.2, 1.5), (-0.012, 0.020, 0.08, 0.125)]
    if data["name"] == "A03":
        limits.append((-0.003, 0.007, 0.03, 0.125))
    return limits


def make_model(newton, wp, reference, data, device):
    """Literal full bodies + one FREE articulation; native mimic, no flattening."""
    n = len(data["children"]["mass"])
    full_q, full_v = reference.expand(data, data["q"], data["v"])
    builder = newton.ModelBuilder(gravity=tuple(data["gravity"]), up_axis=newton.Axis.Z)
    builder.begin_world(label="C11_" + data["name"])
    root = builder.add_link(
        xform=wp.transform(tuple(full_q[:3]), tuple(full_q[3:7])),
        mass=float(data["root_mass"]),
        com=tuple(data["root_com"]),
        inertia=np.diag(data["root_inertia"]),
        lock_inertia=True,
        label="root",
    )
    free = builder.add_joint_free(
        root,
        parent_xform=wp.transform(),
        child_xform=wp.transform(),
        collision_filter_parent=False,
        label="free",
    )
    bodies, joints, shapes, source_shapes = [root], [free], [], []
    labels = ("driver", "follower", "gauge_axis")
    cfg = newton.ModelBuilder.ShapeConfig(**MATERIAL)
    for i in range(n):
        offset, axis = data["children"]["offsets"][i], data["children"]["axes"][i]
        pose = full_q[:3] + offset + axis * full_q[7 + i]
        body = builder.add_link(
            xform=wp.transform(tuple(pose), wp.quat_identity()),
            mass=float(data["children"]["mass"][i]),
            com=tuple(data["children"]["com"][i]),
            inertia=np.diag(data["children"]["inertia"][i]),
            lock_inertia=True,
            label=labels[i],
        )
        lower, upper, speed, effort = declared_limits(data)[i]
        joint = builder.add_joint_prismatic(
            root,
            body,
            axis=tuple(axis),
            parent_xform=wp.transform(tuple(offset), wp.quat_identity()),
            child_xform=wp.transform(),
            target_pos=0.0,
            target_vel=0.0,
            target_ke=0.0,
            target_kd=0.0,
            damping=0.0,
            armature=0.0,
            friction=0.0,
            limit_lower=lower,
            limit_upper=upper,
            limit_ke=0.0,
            limit_kd=0.0,
            effort_limit=effort,
            velocity_limit=speed,
            collision_filter_parent=True,
            label=labels[i],
        )
        bodies.append(body)
        joints.append(joint)
        if data["name"] == "A05":
            shape = builder.add_shape_sphere(
                body, radius=1 / 8, xform=wp.transform(), cfg=cfg, label=labels[i] + "_sphere"
            )
            shapes.append(shape)
            source_shapes.append(
                dict(
                    role=labels[i],
                    full_group=i + 1,
                    native_shape_id=shape,
                    kind="sphere",
                    local_center=(0.0, 0.0, 0.0),
                    radius_m=1 / 8,
                )
            )
    builder.add_articulation(joints, label="one_FREE_full_mimic_articulation")
    relations = []
    for follower in range(1, n):
        offset, gain = float(data["offset"][follower]), float(data["P"][follower, 0])
        builder.set_joint_mimic(joints[follower + 1], joints[1], coeffs=(offset, gain))
        relations.append(
            dict(
                leader=joints[1],
                follower=joints[follower + 1],
                offset=offset,
                multiplier=gain,
                authored_row=follower - 1,
            )
        )
    builder.end_world()
    model = builder.finalize(device=device, requires_grad=False)
    state = model.state()
    state.joint_q.assign(np.asarray(full_q, np.float32))
    state.joint_qd.assign(np.asarray(full_v, np.float32))
    # Initialization only. During all real steps the native solver enforces mimic.
    newton.eval_mimic(model, state)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return (
        model,
        state,
        dict(
            bodies=bodies,
            joints=joints,
            shapes=shapes,
            source_shapes=source_shapes,
            relations=relations,
            initialization_full_q=full_q,
            initialization_full_v=full_v,
        ),
    )


def observe(newton, model, state, meta):
    raw = {name: native_array(state, name) for name in STATE_FIELDS}
    require_values(
        raw, ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"), "public native state"
    )
    b, d = model.body_count, model.joint_dof_count
    public_j = newton.eval_jacobian(model, state)
    public_m = newton.eval_mass_matrix(model, state, J=public_j)
    native_j = public_j.numpy()[0].astype(float).reshape(b, 6, d)
    native_m = public_m.numpy()[0].astype(float)
    poses, twists = raw["body_q"].astype(float), raw["body_qd"].astype(float)
    centers = np.array(
        [p[:3] + rotation(p[3:]) @ c for p, c in zip(poses, meta["body_com"], strict=True)]
    )
    parent_R, parent_p = rotation(meta["joint_X_p"][0, 3:]), meta["joint_X_p"][0, :3].astype(float)
    public_map = np.eye(d)
    public_map[:3, :3] = public_map[3:6, 3:6] = parent_R.T
    internal_map = public_map.copy()
    internal_map[:3, 3:6] = parent_R.T @ skew(centers[0] - parent_p)
    full_v = np.r_[twists[0], raw["joint_qd"][6:]].astype(float)
    full_q = np.r_[poses[0], raw["joint_q"][7:]].astype(float)
    linear, angular, energy = np.zeros(3), np.zeros(3), 0.0
    for i in range(b):
        R = rotation(poses[i, 3:])
        momentum = float(meta["body_mass"][i]) * twists[i, :3]
        spin = R @ meta["body_inertia"][i].astype(float) @ R.T @ twists[i, 3:]
        linear += momentum
        angular += np.cross(centers[i], momentum) + spin
        energy += (twists[i, :3] @ momentum + twists[i, 3:] @ spin) / 2
    return dict(
        raw=raw,
        actual_device=str(state.body_q.device),
        raw_dtypes={k: str(v.dtype) for k, v in raw.items() if isinstance(v, np.ndarray)},
        centers_world_m=centers,
        parent_anchor_position_m=parent_p,
        parent_anchor_rotation=parent_R,
        public_native_jacobian=native_j,
        public_native_mass=native_m,
        world_com_to_public_joint_velocity=public_map,
        world_com_to_internal_anchor_velocity=internal_map,
        jacobian_world_com=native_j @ public_map,
        mass_world_com=public_map.T @ native_m @ public_map,
        full_position=full_q,
        full_velocity=full_v,
        linear_momentum_kg_m_s=linear,
        angular_momentum_world_origin_kg_m2_s=angular,
        kinetic_energy_j=float(energy),
    )


def contact_record(contacts):
    if contacts is None:
        return dict(
            count=0,
            capacity=0,
            valid_capacity=True,
            rows={name: None for name in CONTACT_FIELDS},
            pipeline_active=False,
        )
    count = int(contacts.rigid_contact_count.numpy()[0])
    capacity = int(contacts.rigid_contact_max)
    rows = {}
    for name in CONTACT_FIELDS:
        value = native_array(contacts, name)
        rows[name] = value if value is None else value[: min(max(count, 0), capacity)]
    return dict(
        count=count,
        capacity=capacity,
        valid_capacity=0 <= count <= capacity,
        rows=rows,
        pipeline_active=True,
        force_buffer_semantics="unused by Featherstone; not measured contact force",
        offset_semantics="friction anchors saved raw, not reapplied to normal surfaces",
    )


def inspect_solver(solver, entry):
    raw = {name: native_array(solver, name) for name in AUX_FIELDS}
    require_values(raw, AUX_FIELDS, "mimic solver scratch")
    negative = raw["body_f_ext"].astype(float)
    force = -negative[:, :3]
    centers = raw["body_q_com"][:, :3].astype(float)
    origins = raw["body_solve_origin"].astype(float)
    torque = -negative[:, 3:] - np.cross(centers - origins, force)
    transform = entry["world_com_to_internal_anchor_velocity"]
    acceleration = np.linalg.solve(transform, raw["joint_qdd"].astype(float))
    # d(v_COM)=d(v_anchor)+alpha cross r+omega cross v_COM for the
    # fixed world parent-anchor basis. The final term vanishes in these cases,
    # but is retained and the complete angular vectors are independently checked.
    acceleration[:3] += np.cross(entry["full_velocity"][3:6], entry["full_velocity"][:3])
    b, d = len(centers), len(acceleration)
    internal_j = raw["J"].astype(float).reshape(b, 6, d)
    world_j = np.empty_like(internal_j)
    for body in range(b):
        shift = np.eye(6)
        shift[:3, 3:] = -skew(centers[body] - origins[body])
        world_j[body] = shift @ internal_j[body] @ transform
    return dict(
        raw=raw,
        raw_dtypes={k: str(v.dtype) for k, v in raw.items() if isinstance(v, np.ndarray)},
        decoded_com_world_wrench=np.column_stack((force, torque)),
        decoded_public_world_com_acceleration=acceleration,
        decoded_com_world_jacobian=world_j,
        decoder="F=-rawF; tauCOM=-rawTau-(body_q_com.xyz-body_solve_origin) cross F",
        timing="same entry mechanics scratch, read immediately after native step",
        body_parent_f_semantics="recursive joint wrench, never original-row mimic lambda",
    )


def reconstruct_contacts(meta, entry, contacts, entry_twist):
    """Decode actual penalty rows for observation only; never feed the solver."""
    total = np.zeros((meta["body_count"], 6))
    rows = []
    if not contacts["valid_capacity"]:
        raise ValueError("native contact count exceeds its declared capacity")
    raw = contacts["rows"]
    for i in range(contacts["count"]):
        shapes = tuple(int(raw["rigid_contact_shape" + str(side)][i]) for side in (0, 1))
        if any(s < 0 or s >= meta["shape_count"] for s in shapes):
            raise ValueError("native contact shape outside the complete model")
        owners = tuple(int(meta["shape_body"][s]) for s in shapes)
        if any(body < 0 or body >= meta["body_count"] for body in owners):
            raise ValueError("this declared native case has two dynamic owners only")
        normal = raw["rigid_contact_normal"][i].astype(float)
        points, supports, velocities, references = [], [], [], []
        for side, body in enumerate(owners):
            pose = entry["raw"]["body_q"][body].astype(float)
            support = pose[:3] + rotation(pose[3:]) @ raw["rigid_contact_point" + str(side)][i]
            margin = float(raw["rigid_contact_margin" + str(side)][i])
            point = support + (1 if side == 0 else -1) * margin * normal
            reference = entry["centers_world_m"][body]
            supports.append(support)
            points.append(point)
            references.append(reference)
            velocities.append(
                entry_twist[body, :3] + np.cross(entry_twist[body, 3:], point - reference)
            )
        points, references = np.asarray(points), np.asarray(references)
        gap = float(normal @ (points[1] - points[0]))
        law = {
            name: float(sum(meta["shape_material_" + name][s] for s in shapes) / 2)
            for name in ("ke", "kd", "kf", "ka", "mu")
        }
        for field, name in (
            ("rigid_contact_stiffness", "ke"),
            ("rigid_contact_damping", "kd"),
            ("rigid_contact_friction", "mu"),
        ):
            values = raw[field]
            override = float(values[i]) if values is not None else 0.0
            law[name + "_override"] = override
            if override > 0:
                law[name] = law[name] * override if name == "mu" else override
        if any(law[name] != 0 for name in ("kd", "kf", "ka", "mu")):
            raise ValueError("native row changed the frozen frictionless undamped material")
        magnitude = law["ke"] * max(0.0, -gap)
        forces = np.asarray((-magnitude * normal, magnitude * normal))
        wrenches = np.column_stack((forces, np.cross(points - references, forces)))
        for side, body in enumerate(owners):
            total[body] += wrenches[side]
        rows.append(
            dict(
                native_row=i,
                shape_ids=shapes,
                owner_body_ids=owners,
                normal_a_to_b=normal,
                raw_support_points_world_m=np.asarray(supports),
                effective_surface_points_world_m=points,
                reference_points_world_m=references,
                point_velocities_world_m_s=np.asarray(velocities),
                signed_gap_m=gap,
                mixed_material=law,
                normal_force_n=magnitude,
                force_on_sides_n=forces,
                wrench_on_sides_about_com=wrenches,
            )
        )
    return dict(
        dynamic_body_wrench_world_com=total,
        rows=rows,
        attribution="reconstructed native law; no matched-force input",
    )


def coordinates(data, observation):
    drivers = np.asarray(data["drivers"], int)
    return (
        np.r_[observation["full_position"][:7], observation["full_position"][7 + drivers]],
        np.r_[observation["full_velocity"][:6], observation["full_velocity"][6 + drivers]],
    )


def check_model(record, newton, meta, roles, data, device):
    n = len(data["children"]["mass"])
    b, d = n + 1, n + 6
    check(record, "actual_device_exact", meta["actual_device"] == device, True, atol=0, rtol=0)
    check(
        record,
        "counts_B_joints_D_Q_articulations_worlds",
        [
            meta[k]
            for k in (
                "body_count",
                "joint_count",
                "joint_dof_count",
                "joint_coord_count",
                "articulation_count",
                "world_count",
            )
        ],
        [b, b, d, n + 7, 1, 1],
        atol=0,
        rtol=0,
    )
    check(
        record,
        "full_body_mass",
        meta["body_mass"],
        np.r_[data["root_mass"], data["children"]["mass"]],
    )
    check(
        record,
        "full_body_com",
        meta["body_com"],
        np.vstack((data["root_com"], data["children"]["com"])),
    )
    check(
        record,
        "full_body_inertia",
        meta["body_inertia"],
        np.array(
            [np.diag(v) for v in np.vstack((data["root_inertia"], data["children"]["inertia"]))]
        ),
    )
    check(
        record,
        "full_types",
        meta["joint_type"],
        [int(newton.JointType.FREE)] + [int(newton.JointType.PRISMATIC)] * n,
        atol=0,
        rtol=0,
    )
    check(
        record, "full_parent", meta["joint_parent"], [-1] + [roles["bodies"][0]] * n, atol=0, rtol=0
    )
    check(record, "full_child", meta["joint_child"], roles["bodies"], atol=0, rtol=0)
    check(
        record, "one_articulation_all_rows", meta["joint_articulation"], np.zeros(b), atol=0, rtol=0
    )
    check(
        record, "full_q_starts", meta["joint_q_start"], [0] + list(range(7, 8 + n)), atol=0, rtol=0
    )
    check(
        record,
        "full_qd_starts",
        meta["joint_qd_start"],
        [0] + list(range(6, 7 + n)),
        atol=0,
        rtol=0,
    )
    check(
        record,
        "full_parent_offsets",
        meta["joint_X_p"][:, :3],
        np.vstack((np.zeros(3), data["children"]["offsets"])),
    )
    check(
        record, "identity_parent_rotations", meta["joint_X_p"][:, 3:], np.tile([0, 0, 0, 1], (b, 1))
    )
    check(
        record, "identity_child_anchors", meta["joint_X_c"], np.tile([0, 0, 0, 0, 0, 0, 1], (b, 1))
    )
    check(record, "child_axes", meta["joint_axis"][6:], data["children"]["axes"])
    check(
        record,
        "mimic_leaders",
        meta["joint_mimic_joint"],
        [-1, -1] + [roles["joints"][1]] * (n - 1),
        atol=0,
        rtol=0,
    )
    coefficients = np.tile([0.0, 1.0], (b, 1))
    coefficients[2:, 0], coefficients[2:, 1] = data["offset"][1:], data["P"][1:, 0]
    check(record, "mimic_coefficients", meta["joint_mimic_coeffs"], coefficients)
    for name in (
        "joint_armature",
        "joint_damping",
        "joint_friction",
        "joint_target_ke",
        "joint_target_kd",
        "joint_limit_ke",
        "joint_limit_kd",
    ):
        check(record, "zero_" + name, meta[name], np.zeros_like(meta[name]), atol=0, rtol=0)
    limits = np.array(declared_limits(data))
    for i, name in enumerate(
        ("joint_limit_lower", "joint_limit_upper", "joint_velocity_limit", "joint_effort_limit")
    ):
        check(record, "declared_" + name, meta[name][6:], limits[:, i])
    # Explicit begin_world adds one trailing global gravity row in this pin.
    check(record, "gravity", meta["gravity"], np.tile(data["gravity"], (2, 1)))
    check(record, "body_world_identity", meta["body_world"], np.zeros(b), atol=0, rtol=0)
    check(record, "joint_world_identity", meta["joint_world"], np.zeros(b), atol=0, rtol=0)
    expected_count = 2 if data["name"] == "A05" else 0
    check(record, "actual_shape_count", meta["shape_count"], expected_count, atol=0, rtol=0)
    if expected_count:
        check(
            record,
            "actual_sphere_types",
            meta["shape_type"],
            [int(newton.GeoType.SPHERE)] * 2,
            atol=0,
            rtol=0,
        )
        check(
            record, "actual_shape_owners", meta["shape_body"], roles["bodies"][1:], atol=0, rtol=0
        )
        check(record, "original_radii", meta["shape_scale"][:, 0], [1 / 8] * 2, atol=0, rtol=0)
        check(
            record,
            "sphere_local_pose",
            meta["shape_transform"],
            np.tile([0, 0, 0, 0, 0, 0, 1], (2, 1)),
            atol=0,
            rtol=0,
        )
        for field in ("ke", "kd", "kf", "ka", "mu", "mu_rolling", "mu_torsional"):
            check(
                record,
                "material_" + field,
                meta["shape_material_" + field],
                np.full(2, MATERIAL[field]),
                atol=0,
                rtol=0,
            )
        check(record, "shape_margin", meta["shape_margin"], np.zeros(2), atol=0, rtol=0)
        check(record, "shape_gap", meta["shape_gap"], np.zeros(2), atol=0, rtol=0)
        filtered = {tuple(sorted(map(int, pair))) for pair in meta["shape_collision_filter_pairs"]}
        check(
            record,
            "siblings_not_filtered",
            tuple(sorted(roles["shapes"])) not in filtered,
            True,
            atol=0,
            rtol=0,
        )
        collides = [
            (int(flags) & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
            for flags in meta["shape_flags"]
        ]
        check(record, "shape_collision_enabled", collides, [True, True], atol=0, rtol=0)


def check_observation(record, prefix, observation, reference, data):
    terms = reference.mechanics(data, observation["full_position"], observation["full_velocity"])
    native_pose, native_twist = observation["raw"]["body_q"], observation["raw"]["body_qd"]
    expected_j = np.concatenate((terms["linear_jacobian"], terms["angular_jacobian"]), axis=1)
    for name, actual, expected in (
        ("full_M", observation["mass_world_com"], terms["mass"]),
        ("full_J", observation["jacobian_world_com"], expected_j),
        ("body_positions", native_pose[:, :3], terms["position"]),
        ("body_rotations", np.array([rotation(q) for q in native_pose[:, 3:]]), terms["rotation"]),
        ("body_COM", observation["centers_world_m"], terms["com_position"]),
        ("body_linear_velocity", native_twist[:, :3], terms["com_velocity"]),
        ("body_angular_velocity", native_twist[:, 3:], terms["angular_velocity"]),
        ("linear_momentum", observation["linear_momentum_kg_m_s"], terms["momentum"]),
        (
            "angular_momentum",
            observation["angular_momentum_world_origin_kg_m2_s"],
            terms["angular_momentum"],
        ),
        ("kinetic_energy", observation["kinetic_energy_j"], terms["energy"]),
        (
            "public_configuration_identity_anchors",
            observation["raw"]["joint_q"],
            observation["full_position"],
        ),
        (
            "public_velocity_basis",
            observation["raw"]["joint_qd"],
            observation["world_com_to_public_joint_velocity"] @ observation["full_velocity"],
        ),
    ):
        check(record, prefix + "_" + name, actual, expected)
    q, v = coordinates(data, observation)
    fullq, fullv = reference.expand(data, q, v)
    check(record, prefix + "_full_q_graph", observation["full_position"], fullq)
    check(record, prefix + "_full_v_graph", observation["full_velocity"], fullv)
    check(
        record,
        prefix + "_original_rows_position",
        data["C"][:, 6:] @ observation["full_position"][7:],
        data["C"][:, 6:] @ data["offset"],
    )
    check(
        record,
        prefix + "_original_rows_velocity",
        data["C"] @ observation["full_velocity"],
        np.zeros(len(data["C"])),
    )
    check(
        record,
        prefix + "_all_angular_components_zero",
        native_twist[:, 3:],
        np.zeros_like(native_twist[:, 3:]),
    )
    check(record, prefix + "_root_quaternion", observation["full_position"][3:7], data["q"][3:7])
    return terms


def literal_first(data):
    """Hand-derived X/root/P blocks, independent of native and reference code."""
    if data["name"] == "A01":
        return dict(
            indices=(0, 6, 7),
            mass=[[5, 1, 2], [1, 1, 0], [2, 0, 2]],
            reduced_mass=[[5, -1], [-1, 3]],
            acceleration=[1 / 14, 5 / 14, -5 / 14],
            multipliers=[-4 / 7],
            reduced_load=[0, 1],
        )
    if data["name"] == "A03":
        return dict(
            indices=(0, 6, 7, 8),
            mass=[[11 / 2, 1, 2, 1 / 2], [1, 1, 0, 0], [2, 0, 2, 0], [1 / 2, 0, 0, 1 / 2]],
            reduced_mass=[[11 / 2, -3 / 4], [-3 / 4, 25 / 8]],
            acceleration=[20 / 133, 58 / 133, -58 / 133, 29 / 133],
            multipliers=[-209 / 1064, -42 / 133],
            reduced_load=[1 / 2, 5 / 4],
        )
    return dict(
        indices=(0, 6, 7),
        mass=[[5, 2, 1], [2, 2, 0], [1, 0, 1]],
        reduced_mass=[[5, 1], [1, 3]],
        acceleration=[2 / 7, -10 / 7, 10 / 7],
        multipliers=[-2 / 7],
        reduced_load=[0, -4],
        gap=-1 / 32,
        normal_force=2.0,
    )


def check_first(record, literal, truth, actual_acceleration, actual_lambda):
    indices = np.asarray(literal["indices"])
    check(
        record, "literal_full_mass_block", truth["mass"][np.ix_(indices, indices)], literal["mass"]
    )
    check(
        record,
        "literal_reduced_mass_block",
        truth["reduced_mass"][np.ix_([0, 6], [0, 6])],
        literal["reduced_mass"],
    )
    check(
        record,
        "literal_independent_acceleration",
        truth["full_acceleration"][indices],
        literal["acceleration"],
    )
    check(record, "literal_independent_lambda", truth["row_multipliers"], literal["multipliers"])
    check(record, "literal_reduced_load", truth["reduced_applied"][[0, 6]], literal["reduced_load"])
    check(
        record,
        "literal_actual_acceleration",
        actual_acceleration[indices],
        literal["acceleration"],
        atol=ACCEL_ATOL,
    )
    check(record, "literal_actual_lambda", actual_lambda, literal["multipliers"], atol=ACCEL_ATOL)
    if "gap" in literal:
        check(record, "literal_gap", truth["contact_sites"][0]["gap"], literal["gap"])
        check(
            record, "literal_force", truth["contact_sites"][0]["force_n"], literal["normal_force"]
        )


def check_contacts(record, data, roles, meta, entry, rows, truth):
    expected_count = 1 if data["name"] == "A05" else 0
    check(record, "native_contact_count", len(rows["rows"]), expected_count, atol=0, rtol=0)
    check(
        record,
        "native_contact_body_wrench",
        rows["dynamic_body_wrench_world_com"],
        truth["independent_wrench"],
    )
    if expected_count and len(rows["rows"]) == 1:
        row, site = rows["rows"][0], truth["contact_sites"][0]
        expected_owners = tuple(roles["bodies"][i] for i in (1, 2))
        check(
            record,
            "native_pair_owners",
            sorted(row["owner_body_ids"]),
            sorted(expected_owners),
            atol=0,
            rtol=0,
        )
        check(
            record,
            "native_shape_identity",
            sorted(row["shape_ids"]),
            sorted(roles["shapes"]),
            atol=0,
            rtol=0,
        )
        orientation = 1 if row["owner_body_ids"] == expected_owners else -1
        check(record, "native_normal", row["normal_a_to_b"], -orientation * site["normal"])
        check(record, "native_gap", row["signed_gap_m"], site["gap"])
        check(record, "native_force_magnitude", row["normal_force_n"], site["force_n"])
        for side, body in enumerate(row["owner_body_ids"]):
            wanted = site["first_surface"] if body == expected_owners[0] else site["second_surface"]
            check(
                record,
                "native_surface_" + str(side),
                row["effective_surface_points_world_m"][side],
                wanted,
            )
            shape = row["shape_ids"][side]
            center = entry["raw"]["body_q"][body, :3].astype(float)
            # This frozen source shape has identity local pose and authored radius.
            radius = np.linalg.norm(row["effective_surface_points_world_m"][side] - center)
            check(
                record,
                "native_skin_radius_" + str(side),
                radius,
                float(meta["shape_scale"][shape, 0]),
            )


def check_step(row, reference, data, meta, roles, independent_q, independent_v):
    entry, end, scratch = row["entry"], row["exit"], row["solver_scratch"]
    raw, terms = scratch["raw"], row["independent_same_entry"]
    entry_terms = check_observation(row, "entry", entry, reference, data)
    check_observation(row, "exit", end, reference, data)
    check(row, "native_scratch_entry_twists", raw["body_qd_fk"], entry["raw"]["body_qd"])
    check(row, "native_scratch_entry_COM", raw["body_q_com"][:, :3], entry["centers_world_m"])
    check(
        row,
        "native_solve_origin_root_COM",
        raw["body_solve_origin"],
        np.tile(entry["centers_world_m"][0], (meta["body_count"], 1)),
    )
    check(
        row,
        "native_internal_velocity_basis",
        raw["joint_qd_internal_in"],
        entry["world_com_to_internal_anchor_velocity"] @ entry["full_velocity"],
    )
    check(
        row,
        "native_output_internal_velocity_basis",
        raw["joint_qd_internal_out"],
        end["world_com_to_internal_anchor_velocity"] @ end["full_velocity"],
    )
    decoded = reconstruct_contacts(
        meta, entry, row["native_contacts"], raw["body_qd_fk"].astype(float)
    )
    row["contact_observation"] = decoded
    check_contacts(row, data, roles, meta, entry, decoded, terms)
    body_wrench = scratch["decoded_com_world_wrench"]
    check(
        row,
        "actual_post_RNEA_wrench",
        body_wrench,
        row["original_external_body_wrench"] + decoded["dynamic_body_wrench_world_com"],
    )
    J = np.concatenate((entry_terms["linear_jacobian"], entry_terms["angular_jacobian"]), axis=1)
    g = np.einsum("bij,bi->j", J, body_wrench) + data["E"] @ data["drive"][:, 2]
    check(
        row,
        "full_load_native_vs_independent_J",
        np.einsum("bij,bi->j", entry["jacobian_world_com"], body_wrench)
        + data["E"] @ data["drive"][:, 2],
        g,
    )
    check(row, "full_load_native_vs_declared", g, terms["full_applied"])
    A = entry["world_com_to_internal_anchor_velocity"]
    inverse_A = np.linalg.inv(A)
    d = len(entry["full_velocity"])
    internal_M = inverse_A.T @ entry_terms["mass"] @ inverse_A
    acceleration_correction = np.zeros(d)
    acceleration_correction[:3] = np.cross(entry["full_velocity"][3:6], entry["full_velocity"][:3])
    internal_rhs = inverse_A.T @ (
        g - entry_terms["bias"] - entry_terms["mass"] @ acceleration_correction
    )
    check(row, "internal_full_M_basis", raw["H"].reshape(d, d), internal_M)
    check(row, "internal_full_J_basis", scratch["decoded_com_world_jacobian"], J)
    check(row, "internal_full_tau_basis", raw["joint_tau"], internal_rhs)
    active = np.r_[np.arange(6), 6 + np.asarray(data["drivers"], int)]
    followers = np.setdiff1d(np.arange(d), active)
    reduced_H = np.zeros((d, d))
    reduced_tau = np.zeros(d)
    reduced_H[np.ix_(active, active)] = data["S"].T @ internal_M @ data["S"]
    reduced_H[followers, followers] = 1.0
    reduced_tau[active] = data["S"].T @ internal_rhs
    check(row, "native_mimic_full_padded_H", raw["H_mimic"].reshape(d, d), reduced_H)
    check(row, "native_mimic_full_padded_tau", raw["joint_tau_mimic"], reduced_tau)
    row["internal_basis_expected"] = dict(
        full_mass=internal_M,
        full_rhs=internal_rhs,
        padded_reduced_mass=reduced_H,
        padded_reduced_rhs=reduced_tau,
        active_indices=active,
        inactive_follower_indices=followers,
        acceleration_correction=acceleration_correction,
    )
    delta_a = (end["full_velocity"] - entry["full_velocity"]) / H
    check(row, "actual_delta_v_acceleration", delta_a, terms["full_acceleration"], atol=ACCEL_ATOL)
    check(
        row,
        "decoded_internal_acceleration",
        scratch["decoded_public_world_com_acceleration"],
        terms["full_acceleration"],
        atol=ACCEL_ATOL,
    )
    reaction = entry_terms["mass"] @ delta_a + entry_terms["bias"] - g
    C = data["C"]
    multipliers = np.linalg.solve(C @ C.T, C @ reaction)
    row_power = multipliers * (C @ entry["full_velocity"])
    row["full_balance_observation"] = dict(
        acceleration_from_actual_delta_v=delta_a,
        full_generalized_applied=g,
        full_reaction=reaction,
        row_multipliers=multipliers,
        row_powers_w=row_power,
        constraint_power_w=float(row_power.sum()),
        root_constraint_residual=reaction[:6],
        source="independent full M/b/J and decoded native load; actual delta-v; original C",
    )
    check(row, "actual_full_reaction", reaction, terms["constraint_effort_full"], atol=ACCEL_ATOL)
    check(row, "actual_original_row_lambda", multipliers, terms["row_multipliers"], atol=ACCEL_ATOL)
    check(row, "actual_original_row_representation", reaction, C.T @ multipliers, atol=ACCEL_ATOL)
    check(row, "actual_root_six_balance", reaction[:6], np.zeros(6), atol=ACCEL_ATOL)
    check(
        row,
        "actual_reduced_balance",
        data["S"].T @ reaction,
        np.zeros(data["S"].shape[1]),
        atol=ACCEL_ATOL,
    )
    check(row, "actual_original_row_acceleration", C @ delta_a, np.zeros(len(C)), atol=ACCEL_ATOL)
    check(row, "actual_original_row_power", row_power, terms["row_powers_w"], atol=ACCEL_ATOL)
    fullq, fullv = reference.expand(data, independent_q, independent_v)
    check(row, "independent_window_full_q", end["full_position"], fullq, atol=WINDOW_ATOL)
    check(row, "independent_window_full_v", end["full_velocity"], fullv, atol=WINDOW_ATOL)
    final_terms = reference.mechanics(data, fullq, fullv)
    check(row, "independent_window_kinetic_energy", end["kinetic_energy_j"], final_terms["energy"])
    delta_p = end["linear_momentum_kg_m_s"] - entry["linear_momentum_kg_m_s"]
    delta_L = (
        end["angular_momentum_world_origin_kg_m2_s"]
        - entry["angular_momentum_world_origin_kg_m2_s"]
    )
    force_total = body_wrench[:, :3].sum(axis=0)
    torque_total = (
        body_wrench[:, 3:] + np.cross(entry["centers_world_m"], body_wrench[:, :3])
    ).sum(axis=0)
    check(row, "actual_linear_momentum_balance", delta_p, H * force_total)
    check(row, "actual_angular_momentum_balance", delta_L, H * torque_total)
    velocity_increment = end["full_velocity"] - entry["full_velocity"]
    actual_work = H * (g @ end["full_velocity"])
    implicit_loss = 0.5 * velocity_increment @ entry_terms["mass"] @ velocity_increment
    delta_energy = end["kinetic_energy_j"] - entry["kinetic_energy_j"]
    row["energy_observation"] = dict(
        actual_delta_kinetic_j=float(delta_energy),
        endpoint_force_work_j=float(actual_work),
        implicit_loss_j=float(implicit_loss),
        balance_j=float(delta_energy - actual_work + implicit_loss),
        scope="constant collinear translational mass; native discrete endpoint work, not a generic angular energy law",
    )
    check(row, "actual_discrete_kinetic_balance", delta_energy - actual_work + implicit_loss, 0.0)
    if row["step"] == 1:
        row["literal_first_step"] = literal_first(data)
        check_first(row, row["literal_first_step"], terms, delta_a, multipliers)
    return final_terms


def require_values(values, names, scope):
    missing = [name for name in names if values.get(name) is None]
    if missing:
        raise ValueError(scope + " missing required pinned fields: " + ", ".join(missing))


def failed_checks(record):
    failures = [
        "model/" + name for name, item in record.get("checks", {}).items() if not item["passed"]
    ]
    for row in record["trace"]:
        failures.extend(
            "step" + str(row["step"]) + "/" + name
            for name, item in row.get("checks", {}).items()
            if not item["passed"]
        )
    return failures


def run_case(newton, wp, reference, declaration, device, output, provenance):
    label, name, variant, steps = declaration
    path = output / (label + "-native.json.gz")
    record = dict(
        case=label,
        channel="native",
        status="running",
        h_s=H,
        declared_steps=steps,
        native_steps_completed=0,
        complete_steps=0,
        trace=[],
        checks={},
        execution_error=None,
        source_binding=provenance,
        semantics=dict(
            contact="entry collision then one real Featherstone step",
            zero_matched_contact_force=True,
            state_updates="native only after initialization",
            row_lambda="independent full balance reconstructed from actual velocity increment",
            public_velocity="FREE parent-anchor axes COM-linear first; body_qd COM/world",
            scratch="same-entry internal quantities; body_f_ext post-RNEA negative solve-origin wrench",
            angular_scope="three collinear zero-angular cases, not general rotating integrator parity",
            failure="IO/observation failure never means physical rollback",
        ),
    )
    save_record(path, record)
    try:
        data = reference.case(name, variant=variant)
        record["declaration"] = data
        check(record, "frozen_h", data["h"], H, atol=0, rtol=0)
        check(
            record,
            "root_inertia_authored_axes",
            data["root_inertia_frame"],
            [0, 0, 0, 1],
            atol=0,
            rtol=0,
        )
        check(
            record,
            "zero_independent_PD",
            np.r_[data["kp"], data["kd"]],
            np.zeros(2),
            atol=0,
            rtol=0,
        )
        check(
            record,
            "zero_external_joint_effort",
            data["effort"],
            np.zeros(len(data["effort"])),
            atol=0,
            rtol=0,
        )
        model, state, roles = make_model(newton, wp, reference, data, device)
        meta = metadata(model)
        record.update(model=meta, role_ids=roles)
        required_model = [field for field in MODEL_FIELDS if not field.startswith("shape_")]
        require_values(meta, required_model, "full model")
        if name == "A05":
            require_values(
                meta,
                [
                    field
                    for field in MODEL_FIELDS
                    if field.startswith("shape_") and field != "shape_contact_pairs"
                ],
                "contact model",
            )
        check_model(record, newton, meta, roles, data, device)
        candidate, control = model.state(), model.control()
        original_effort = np.zeros(model.joint_dof_count, np.float32)
        original_effort[6 + np.asarray(data["drivers"], int)] = data["drive"][:, 2]
        control.joint_f.assign(original_effort)
        solver = newton.solvers.SolverFeatherstone(
            model,
            angular_damping=0.0,
            update_mass_matrix_interval=1,
            use_tile_gemm=False,
            fuse_cholesky=True,
        )
        record["solver"] = dict(
            name=type(solver).__name__,
            angular_damping=0.0,
            update_mass_matrix_interval=1,
            use_tile_gemm=False,
            fuse_cholesky=True,
            friction_smoothing=float(solver.friction_smoothing),
            requires_grad=False,
        )
        pipeline = None
        if name == "A05":
            pipeline = newton.CollisionPipeline(
                model,
                broad_phase="nxn",
                rigid_contact_max=8,
                soft_contact_max=0,
                deterministic=True,
                contact_matching="disabled",
            )
        contacts = None if pipeline is None else pipeline.contacts()
        record["pipeline"] = dict(
            active=pipeline is not None,
            broad_phase="nxn" if pipeline else None,
            rigid_contact_max=8 if pipeline else 0,
            soft_contact_max=0,
            deterministic=True,
            contact_matching="disabled",
        )
        independent_q, independent_v = data["q"].copy(), data["v"].copy()
        record["initial_state"] = observe(newton, model, state, meta)
        check_observation(record, "initial", record["initial_state"], reference, data)
        check(
            record,
            "initial_full_q",
            record["initial_state"]["full_position"],
            roles["initialization_full_q"],
        )
        check(
            record,
            "initial_full_v",
            record["initial_state"]["full_velocity"],
            roles["initialization_full_v"],
        )
        original_wrench = np.asarray(np.column_stack((data["force"], data["torque"])), np.float32)
        for step in range(1, steps + 1):
            state.clear_forces()
            state.body_f.assign(original_wrench)
            entry = observe(newton, model, state, meta)
            q, v = coordinates(data, entry)
            _, _, same_entry = reference.advance(data, q, v, data["drive"], h=H)
            independent_entry_q, independent_entry_v = independent_q.copy(), independent_v.copy()
            independent_q, independent_v, window_terms = reference.advance(
                data, independent_q, independent_v, data["drive"], h=H
            )
            original_control = {field: native_array(control, field) for field in CONTROL_FIELDS}
            row = dict(
                step=step,
                entry_time_s=(step - 1) * H,
                exit_time_s=step * H,
                entry=entry,
                original_control=original_control,
                original_external_body_wrench=entry["raw"]["body_f"].copy(),
                matched_contact_wrench_supplied=False,
                independent_same_entry=same_entry,
                independent_window=dict(
                    entry_q=independent_entry_q,
                    entry_v=independent_entry_v,
                    exit_q=independent_q.copy(),
                    exit_v=independent_v.copy(),
                    entry_terms=window_terms,
                ),
                native_step_submitted=False,
                native_step_completed=False,
                checks_complete=False,
            )
            record["trace"].append(row)
            require_values(
                original_control, ("joint_f", "joint_target_q", "joint_target_qd"), "control"
            )
            check(
                row,
                "only_authored_motor",
                original_control["joint_f"],
                original_effort,
                atol=0,
                rtol=0,
            )
            check(
                row, "no_root_motor", original_control["joint_f"][:6], np.zeros(6), atol=0, rtol=0
            )
            check(
                row,
                "only_authored_external_wrench",
                row["original_external_body_wrench"],
                original_wrench,
                atol=0,
                rtol=0,
            )
            if pipeline is not None:
                pipeline.collide(state, contacts)
            row["native_contacts"] = contact_record(contacts)
            check(
                row,
                "native_contact_capacity",
                row["native_contacts"]["valid_capacity"],
                True,
                atol=0,
                rtol=0,
            )
            if name == "A05":
                require_values(
                    row["native_contacts"]["rows"],
                    (
                        "rigid_contact_shape0",
                        "rigid_contact_shape1",
                        "rigid_contact_point0",
                        "rigid_contact_point1",
                        "rigid_contact_normal",
                        "rigid_contact_margin0",
                        "rigid_contact_margin1",
                    ),
                    "native contact geometry",
                )
            # Persist the actual entry before the native call, then the actual
            # exit before independent checks. A later error preserves both.
            save_record(path, record)
            row["native_step_submitted"] = True
            result = solver.step(state, candidate, control, contacts, H)
            row["native_return"] = (
                result
                if result is None or isinstance(result, (bool, int, float, str))
                else repr(result)
            )
            row["native_step_completed"] = True
            record["native_steps_completed"] += 1
            row["poststep_input_raw"] = {
                field: native_array(state, field) for field in STATE_FIELDS
            }
            row["poststep_output_raw"] = {
                field: native_array(candidate, field) for field in STATE_FIELDS
            }
            row["solver_scratch"] = inspect_solver(solver, entry)
            row["exit"] = observe(newton, model, candidate, meta)
            save_record(path, record)
            check_step(row, reference, data, meta, roles, independent_q, independent_v)
            row["checks_complete"] = True
            record["complete_steps"] += 1
            state, candidate = candidate, state
            save_record(path, record)
        record["final_state"] = observe(newton, model, state, meta)
        check(
            record,
            "completed_native_steps",
            record["native_steps_completed"],
            steps,
            atol=0,
            rtol=0,
        )
        check(record, "completed_observer_steps", record["complete_steps"], steps, atol=0, rtol=0)
        check(
            record,
            "step_device_exact",
            all(
                row["entry"]["actual_device"] == device and row["exit"]["actual_device"] == device
                for row in record["trace"]
            ),
            True,
            atol=0,
            rtol=0,
        )
    except Exception as error:
        record["execution_error"] = dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
        )
    record["failed_checks"] = failed_checks(record)
    record["first_failed_check"] = next(iter(record["failed_checks"]), None)
    record["status"] = (
        "passed"
        if (
            record["execution_error"] is None
            and not record["failed_checks"]
            and record["native_steps_completed"] == steps
            and record["complete_steps"] == steps
        )
        else "failed"
    )
    decoded_bytes = save_record(path, record)
    return dict(
        case=label,
        status=record["status"],
        h_s=H,
        declared_steps=steps,
        native_steps_completed=record["native_steps_completed"],
        complete_steps=record["complete_steps"],
        failed_checks=record["failed_checks"],
        execution_error=record["execution_error"],
        path=str(path),
        decoded_bytes=decoded_bytes,
        compressed_bytes=path.stat().st_size,
        gzip_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def source_snapshot(output, source_paths):
    records = {}
    for name, original in source_paths.items():
        data = original.read_bytes()
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records[name] = dict(
            path=str(original),
            snapshot=str(target),
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda:0"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.output.is_absolute():
        raise ValueError("--output must be an absolute fresh directory")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    reference_path = script.with_name(REFERENCE_NAME)
    base_path = script.with_name(BASE_REFERENCE_NAME)
    document = dict(
        status="running",
        channel="native",
        device=args.device,
        h_s=H,
        declared_cases=CASES,
        declared_total_steps=9,
        cases=[],
        checks={},
        execution_error=None,
        invocation=list(sys.argv),
        actual_cwd=str(Path.cwd()),
        parent_contract="881c28c / resident-floating-affine plan section 8",
        gates=dict(
            same_entry_atol=ATOL,
            rtol=RTOL,
            window_atol=WINDOW_ATOL,
            acceleration_reaction_atol=ACCEL_ATOL,
        ),
        limits=dict(per_case_decoded_bytes=RAW_LIMIT, per_case_compressed_bytes=GZIP_LIMIT),
        dependency_semantics="adjacent explicit NumPy/time-jet reference; no Phymeo/TYX/testfixture import",
    )
    try:
        import newton
        import warp as wp

        if version("newton") != "1.7.0" or version("warp-lang") != "1.17.0":
            raise RuntimeError("requires installed Newton 1.7.0 / Warp 1.17.0 exactly")
        document["versions"] = dict(
            newton=version("newton"),
            warp=version("warp-lang"),
            numpy=np.__version__,
            python=sys.version,
            platform=platform.platform(),
        )
        root = Path(newton.__file__).resolve().parent
        document["module_paths"] = dict(
            newton=str(Path(newton.__file__).resolve()),
            warp=str(Path(wp.__file__).resolve()),
            numpy=str(Path(np.__file__).resolve()),
        )
        paths = {"newton/" + name: root / name for name in SOURCES}
        paths.update(
            {
                "driver/" + script.name: script,
                "independent/" + REFERENCE_NAME: reference_path,
                "independent/" + BASE_REFERENCE_NAME: base_path,
            }
        )
        document["sources_before"] = source_snapshot(output, paths)
        provenance = dict(
            versions=document["versions"],
            device=args.device,
            sources={name: row["sha256"] for name, row in document["sources_before"].items()},
            reference_path=str(reference_path),
            base_reference_path=str(base_path),
        )
        reference = load_reference(reference_path)
        wp.config.kernel_cache_dir = str(output / "warp-cache")
        wp.init()
        with wp.ScopedDevice(args.device):
            for case in CASES:
                try:
                    document["cases"].append(
                        run_case(newton, wp, reference, case, args.device, output, provenance)
                    )
                except Exception as error:
                    document["cases"].append(
                        dict(
                            case=case[0],
                            status="failed",
                            declared_steps=case[3],
                            native_steps_completed=None,
                            complete_steps=None,
                            execution_error=dict(
                                type=type(error).__name__,
                                message=str(error),
                                traceback=traceback.format_exc(),
                            ),
                            failure_semantics="runner/persistence failure; inspect saved prefix, not a physical rollback",
                        )
                    )
        document["sources_after"] = {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
        }
        check(
            document,
            "all_sources_unchanged",
            all(
                document["sources_after"][name] == value["sha256"]
                for name, value in document["sources_before"].items()
            ),
            True,
            atol=0,
            rtol=0,
        )
        forbidden = sorted(name for name in sys.modules if name.split(".")[0] in ("phymeo", "tyx"))
        document["forbidden_loaded_modules"] = forbidden
        check(document, "no_Phymeo_TYX_import", len(forbidden), 0, atol=0, rtol=0)
        check(
            document,
            "exact_declared_case_order",
            [row["case"] for row in document["cases"]] == [case[0] for case in CASES],
            True,
            atol=0,
            rtol=0,
        )
        total = sum(row["native_steps_completed"] or 0 for row in document["cases"])
        check(document, "all_nine_native_steps_completed", total, 9, atol=0, rtol=0)
        document["status"] = (
            "passed"
            if (
                len(document["cases"]) == len(CASES)
                and all(row["status"] == "passed" for row in document["cases"])
                and all(row["passed"] for row in document["checks"].values())
            )
            else "failed"
        )
    except Exception as error:
        document["status"] = "failed"
        document["execution_error"] = dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
        )
    summary = output / "summary.json"
    summary.write_text(
        json.dumps(json_value(document), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            json_value(
                dict(
                    status=document["status"],
                    device=args.device,
                    cases=document["cases"],
                    execution_error=document["execution_error"],
                    summary=str(summary),
                )
            ),
            allow_nan=False,
        )
    )
    return 0 if document["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
