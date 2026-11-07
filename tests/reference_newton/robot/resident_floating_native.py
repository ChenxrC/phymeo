"""Pinned Newton 1.7 free-root trajectories; no Phymeo/TYX imports.

All seven declared cases run 40 native Featherstone steps at h=1/1024. Public
FREE qd is COM-referenced in the parent-anchor basis; the internal solve speed
and post-RNEA force buffer have different meanings and are retained separately.
Authoring only: no result in this source is an executed acceptance claim.
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
ATOL, RTOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-4
MASS_ATOL = 2e-5
MAX_BYTES = 256 * 1024 * 1024
SOURCES = (
    "_src/sim/builder.py",
    "_src/sim/model.py",
    "_src/sim/state.py",
    "_src/sim/articulation.py",
    "_src/solvers/featherstone/kernels.py",
    "_src/solvers/featherstone/solver_featherstone.py",
)
MODEL_ARRAYS = (
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
    "shape_transform",
    "shape_scale",
    "shape_flags",
)
AUX_ARRAYS = (
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
)


def encode(value):
    if isinstance(value, np.ndarray):
        return encode(value.tolist())
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def rot(q):
    x, y, z, w = np.asarray(q, float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def qmul(a, b):
    return np.r_[a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]), a[3] * b[3] - a[:3] @ b[:3]]


def normalized_euler(q, omega, h):
    candidate = q + h / 2 * qmul(np.r_[omega, 0.0], q)
    return candidate / np.linalg.norm(candidate)


def exponential(q, omega, h):
    angle = h * np.linalg.norm(omega)
    delta = (
        np.r_[omega * (math.sin(angle / 2) / np.linalg.norm(omega)), math.cos(angle / 2)]
        if angle
        else np.array([0.0, 0, 0, 1])
    )
    return qmul(delta, q)


def axis_rotation(axis, q):
    cross = skew(axis)
    return np.eye(3) + math.sin(q) * cross + (1 - math.cos(q)) * (cross @ cross)


def check(record, name, actual, expected, *, atol=ATOL, rtol=RTOL):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    finite = a.shape == b.shape and bool(np.isfinite(a).all() and np.isfinite(b).all())
    error = abs(a - b) if finite else np.array([np.inf])
    ok = finite and bool(np.all(error <= atol + rtol * abs(b)))
    record.setdefault("checks", {})[name] = dict(
        passed=ok,
        max_abs_error=float(error.max(initial=0)),
        atol=atol,
        rtol=rtol,
        actual_shape=a.shape,
        expected_shape=b.shape,
    )
    return ok


def cases():
    base = dict(
        root_mass=2.0,
        root_com=np.zeros(3),
        root_inertia=np.eye(3) * 0.5,
        origin=np.zeros(3),
        quaternion=np.array([0.0, 0, 0, 1]),
        velocity=np.zeros(7),
        gravity=np.zeros(3),
        kind="prismatic",
        child_mass=1.0,
        child_com=np.zeros(3),
        child_inertia=np.eye(3) * 0.25,
        axis=np.array([1.0, 0, 0]),
        joint_offset=np.zeros(3),
        joint_q=0.0,
        motor=0.0,
        effort=0.0,
        force=np.zeros((2, 3)),
        torque=np.zeros((2, 3)),
        anchor_xyz=np.zeros(3),
        anchor_xyzw=np.array([0.0, 0, 0, 1]),
        origin_force=False,
    )
    rows = []
    for name in (
        "P01_freefall",
        "P02_origin_force",
        "P02_principal_torque",
        "P03_internal",
        "P03_child_force",
        "P04_internal",
        "P04_eccentric",
    ):
        row = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in base.items()}
        row["id"] = name
        if name == "P01_freefall":
            row.update(
                joint_q=1 / 8,
                gravity=np.array([0.0, 0, -2]),
                literal_first_acceleration=np.array([0.0, 0, -2, 0, 0, 0, 0]),
            )
        elif name.startswith("P02"):
            row.update(
                kind=None, velocity=np.zeros(6), force=np.zeros((1, 3)), torque=np.zeros((1, 3))
            )
            if name == "P02_origin_force":
                row.update(
                    root_com=np.array([1 / 8, 0, 0]),
                    root_inertia=np.diag([1 / 4, 3 / 8, 1 / 2]),
                    origin_force=True,
                    literal_first_acceleration=np.array([0.0, 1, 0, 0, 0, -0.5]),
                )
                row["force"][0] = (0, 2, 0)
            else:
                row.update(
                    root_inertia=np.array([[3 / 8, -1 / 8, 0], [-1 / 8, 3 / 8, 0], [0, 0, 1 / 2]]),
                    literal_first_acceleration=np.array([0.0, 0, 0, 3, 1, 0]),
                )
                row["torque"][0] = (1, 0, 0)
        elif name == "P03_internal":
            row.update(motor=3.0, literal_first_acceleration=np.array([-1.5, 0, 0, 0, 0, 0, 4.5]))
        elif name == "P03_child_force":
            row["force"][1] = (3, 0, 0)
            row["literal_first_acceleration"] = np.array([0.0, 0, 0, 0, 0, 0, 3])
        elif name == "P04_internal":
            row.update(
                kind="revolute",
                axis=np.array([0.0, 0, 1]),
                motor=1.0,
                literal_first_acceleration=np.array([0.0, 0, 0, 0, 0, -2, 6]),
            )
        else:
            principal = rot([0.0, 0, 3 / 5, 4 / 5])
            row.update(
                kind="revolute",
                axis=np.array([0.0, 0, 1]),
                root_com=np.array([1 / 16, -1 / 32, 1 / 64]),
                root_inertia=principal @ np.diag([1 / 2, 5 / 8, 3 / 4]) @ principal.T,
                quaternion=np.full(4, 0.5),
                child_com=np.array([1 / 8, 0, 0]),
                joint_q=-1 / 16,
                velocity=np.array([1 / 16, -1 / 32, 1 / 64, 1 / 32, -1 / 64, 1 / 128, -1 / 64]),
                gravity=np.array([0.0, 0, -1 / 4]),
                motor=1 / 8,
                effort=-1 / 32,
                anchor_xyz=np.array([1 / 2, -1 / 4, 3 / 4]),
                anchor_xyzw=np.full(4, 0.5),
            )
            row["force"][1] = (1 / 8, -1 / 16, 1 / 32)
            row["torque"][0] = (-1 / 32, 1 / 64, 0)
        rows.append(row)
    return rows


def build(newton, wp, case, device):
    builder = newton.ModelBuilder(gravity=tuple(case["gravity"]))
    root = builder.add_link(
        xform=wp.transform(case["origin"], wp.quat(*case["quaternion"])),
        mass=case["root_mass"],
        com=case["root_com"],
        inertia=case["root_inertia"],
        lock_inertia=True,
        label="root",
    )
    free = builder.add_joint_free(
        root,
        parent_xform=wp.transform(case["anchor_xyz"], wp.quat(*case["anchor_xyzw"])),
        label="free",
    )
    joints = [free]
    if case["kind"]:
        child = builder.add_link(
            mass=case["child_mass"],
            com=case["child_com"],
            inertia=case["child_inertia"],
            lock_inertia=True,
            label="child",
        )
        method = (
            builder.add_joint_prismatic
            if case["kind"] == "prismatic"
            else builder.add_joint_revolute
        )
        joints.append(
            method(
                root,
                child,
                axis=case["axis"],
                parent_xform=wp.transform(case["joint_offset"], wp.quat_identity()),
                armature=0.0,
                damping=0.0,
                target_ke=0.0,
                target_kd=0.0,
                limit_ke=0.0,
                limit_kd=0.0,
                label="scalar",
            )
        )
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    state = model.state()
    q, v = model.joint_q.numpy(), model.joint_qd.numpy()
    anchor_rotation = rot(case["anchor_xyzw"])
    v[:3], v[3:6] = (
        anchor_rotation.T @ case["velocity"][:3],
        anchor_rotation.T @ case["velocity"][3:6],
    )
    if case["kind"]:
        q[7], v[6] = case["joint_q"], case["velocity"][6]
    state.joint_q.assign(q)
    state.joint_qd.assign(v)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state


def metadata(model):
    out = {name: getattr(model, name).numpy() for name in MODEL_ARRAYS}
    out.update(
        body_label=list(model.body_label),
        joint_label=list(model.joint_label),
        body_count=model.body_count,
        joint_count=model.joint_count,
        joint_dof_count=model.joint_dof_count,
        joint_coord_count=model.joint_coord_count,
        articulation_count=model.articulation_count,
        device=str(model.device),
        shape_collision_filter_pairs=model.shape_collision_filter_pairs_array(),
    )
    return out


def observe(newton, model, state, case):
    raw = {
        name: getattr(state, name).numpy()
        for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f")
    }
    d, b = len(raw["joint_qd"]), model.body_count
    native_j = newton.eval_jacobian(model, state).numpy()[0].astype(float).reshape(b, 6, d)
    native_m = newton.eval_mass_matrix(model, state).numpy()[0].astype(float)
    world_to_native = np.eye(d)
    world_to_native[:3, :3] = rot(case["anchor_xyzw"]).T
    world_to_native[3:6, 3:6] = rot(case["anchor_xyzw"]).T
    q = np.r_[raw["body_q"][0], raw["joint_q"][7:]].astype(float)
    velocity = np.r_[raw["body_qd"][0], raw["joint_qd"][6:]].astype(float)
    centers, momentum, angular = [], np.zeros(3), np.zeros(3)
    energy = 0.0
    masses = (case["root_mass"], case["child_mass"]) if b == 2 else (case["root_mass"],)
    coms = (case["root_com"], case["child_com"])
    inertias = (case["root_inertia"], case["child_inertia"])
    for i in range(b):
        pose, twist = raw["body_q"][i].astype(float), raw["body_qd"][i].astype(float)
        orientation = rot(pose[3:])
        center = pose[:3] + orientation @ coms[i]
        inertia = orientation @ inertias[i] @ orientation.T
        p, spin = masses[i] * twist[:3], inertia @ twist[3:]
        centers.append(center)
        momentum += p
        angular += np.cross(center, p) + spin
        energy += (twist[:3] @ p + twist[3:] @ spin) / 2
    return dict(
        raw=raw,
        raw_dtypes={k: str(v.dtype) for k, v in raw.items()},
        actual_device=str(state.body_q.device),
        position=q,
        velocity=velocity,
        centers=np.asarray(centers),
        linear_momentum=momentum,
        angular_momentum_world_origin=angular,
        kinetic_energy_j=float(energy),
        public_native_jacobian=native_j,
        public_native_mass=native_m,
        world_to_native_velocity=world_to_native,
        jacobian_world_com=native_j @ world_to_native,
        mass_world_com=world_to_native.T @ native_m @ world_to_native,
    )


def analytical(case, q, v):
    """Closed one-child geometry and Newton-Euler terms, no engine arrays or FD."""
    b, d = (2, 7) if case["kind"] else (1, 6)
    orientation = rot(q[3:7])
    root_arm = orientation @ case["root_com"]
    root_center = q[:3] + root_arm
    points, rotations = [q[:3]], [orientation]
    centers = [root_center]
    axis = orientation @ case["axis"]
    pivot = q[:3] + orientation @ case["joint_offset"]
    if b == 2:
        child_rotation = (
            orientation @ axis_rotation(case["axis"], q[7])
            if case["kind"] == "revolute"
            else orientation
        )
        point = pivot + (axis * q[7] if case["kind"] == "prismatic" else 0)
        points.append(point)
        rotations.append(child_rotation)
        centers.append(point + child_rotation @ case["child_com"])
    linear, angular = np.zeros((b, 3, d)), np.zeros((b, 3, d))
    for i in range(b):
        linear[i, :, :3] = np.eye(3)
        linear[i, :, 3:6] = -skew(centers[i] - root_center)
        angular[i, :, 3:6] = np.eye(3)
    if b == 2:
        if case["kind"] == "prismatic":
            linear[1, :, 6] = axis
        else:
            linear[1, :, 6] = np.cross(axis, centers[1] - pivot)
            angular[1, :, 6] = axis
    force, torque = case["force"].copy(), case["torque"].copy()
    if case["origin_force"]:
        torque[0] = np.cross(-root_arm, force[0])
    acceleration_bias, alpha_bias = [np.zeros(3)], [np.zeros(3)]
    omega = v[3:6]
    if b == 2:
        child_alpha = np.cross(omega, axis) * v[6] if case["kind"] == "revolute" else np.zeros(3)
        arm = points[1] - points[0]
        origin_acc = -np.cross(omega, np.cross(omega, root_arm)) + np.cross(
            omega, np.cross(omega, arm)
        )
        if case["kind"] == "prismatic":
            origin_acc += 2 * np.cross(omega, axis * v[6])
        child_omega = angular[1] @ v
        child_arm = centers[1] - points[1]
        acceleration_bias.append(
            origin_acc
            + np.cross(child_alpha, child_arm)
            + np.cross(child_omega, np.cross(child_omega, child_arm))
        )
        alpha_bias.append(child_alpha)
    mass, bias, external = np.zeros((d, d)), np.zeros(d), np.zeros(d)
    for i, (m, inertia_body) in enumerate(
        zip(
            (case["root_mass"], case["child_mass"]),
            (case["root_inertia"], case["child_inertia"]),
            strict=True,
        )
    ):
        if i >= b:
            break
        inertia = rotations[i] @ inertia_body @ rotations[i].T
        w = angular[i] @ v
        mass += m * linear[i].T @ linear[i] + angular[i].T @ inertia @ angular[i]
        bias += linear[i].T @ (m * (acceleration_bias[i] - case["gravity"])) + angular[i].T @ (
            inertia @ alpha_bias[i] + np.cross(w, inertia @ w)
        )
        external += linear[i].T @ force[i] + angular[i].T @ torque[i]
    if b == 2:
        external[6] += case["effort"]
    rhs = external.copy()
    if b == 2:
        rhs[6] += case["motor"]
    acceleration = np.linalg.solve(mass, rhs - bias)
    return dict(
        origin=np.asarray(points),
        rotation=np.asarray(rotations),
        center=np.asarray(centers),
        linear=linear,
        angular=angular,
        mass=mass,
        bias=bias,
        external=external,
        acceleration=acceleration,
        force=force,
        torque=torque,
    )


def inspect_native_step(solver, case, entry, independent):
    raw = {name: getattr(solver, name).numpy() for name in AUX_ARRAYS}
    force = -raw["body_f_ext"][:, :3].astype(float)
    arm = raw["body_q_com"][:, :3].astype(float) - raw["body_solve_origin"].astype(float)
    torque = -raw["body_f_ext"][:, 3:].astype(float) - np.cross(arm, force)
    accel = raw["joint_qdd"].astype(float).copy()
    orient = rot(case["anchor_xyzw"])
    omega, velocity = entry["velocity"][3:6], entry["velocity"][:3]
    alpha = orient @ accel[3:6]
    relative = entry["centers"][0] - case["anchor_xyz"]
    accel[:3] = orient @ accel[:3] + np.cross(alpha, relative) + np.cross(omega, velocity)
    accel[3:6] = alpha
    return dict(
        raw=raw,
        raw_dtypes={k: str(v.dtype) for k, v in raw.items()},
        decoded_com_world_wrench=np.column_stack((force, torque)),
        decoded_world_com_acceleration=accel,
        original_com_world_wrench=np.column_stack((independent["force"], independent["torque"])),
        decoder="F=-raw[:3]; tauCOM=-raw[3:]-(body_q_com.xyz-body_solve_origin) cross F",
    )


def run_case(newton, wp, case, device, record):
    from newton.solvers import SolverFeatherstone

    model, state = build(newton, wp, case, device)
    candidate, control = model.state(), model.control()
    solver = SolverFeatherstone(model, angular_damping=0.0)
    record.update(declaration=case, metadata=metadata(model), rows=[], checks={})
    b = model.body_count
    check(
        record,
        "original_mass",
        record["metadata"]["body_mass"],
        [case["root_mass"], case["child_mass"]][:b],
        atol=0,
        rtol=0,
    )
    check(
        record,
        "original_com",
        record["metadata"]["body_com"],
        [case["root_com"], case["child_com"]][:b],
    )
    check(
        record,
        "original_inertia",
        record["metadata"]["body_inertia"],
        [case["root_inertia"], case["child_inertia"]][:b],
    )
    for name in (
        "joint_armature",
        "joint_damping",
        "joint_target_ke",
        "joint_target_kd",
        "joint_limit_ke",
        "joint_limit_kd",
    ):
        check(
            record,
            name + "_zero",
            record["metadata"][name],
            np.zeros_like(record["metadata"][name]),
            atol=0,
            rtol=0,
        )
    first = observe(newton, model, state, case)
    record["initial"] = first
    check(
        record,
        "initial_world_q",
        first["position"],
        np.r_[case["origin"], case["quaternion"], [case["joint_q"]] if b == 2 else []],
    )
    check(record, "initial_world_v", first["velocity"], case["velocity"])
    motor = np.zeros(model.joint_dof_count, np.float32)
    if b == 2:
        motor[6] = case["motor"] + case["effort"]
    control.joint_f.assign(motor)
    expected_momentum = first["linear_momentum"].copy()
    for index in range(STEPS):
        entry = observe(newton, model, state, case)
        truth = analytical(case, entry["position"], entry["velocity"])
        row = dict(
            index=index,
            entry_time_s=index * H,
            exit_time_s=(index + 1) * H,
            entry=entry,
            independent_entry=truth,
            joint_f_input=motor.copy(),
            checks={},
        )
        record["rows"].append(row)
        check(row, "full_mass_world_com", entry["mass_world_com"], truth["mass"], atol=MASS_ATOL)
        check(row, "full_linear_J", entry["jacobian_world_com"][:, :3], truth["linear"])
        check(row, "full_angular_J", entry["jacobian_world_com"][:, 3:], truth["angular"])
        check(
            row, "body_pose_from_declared_geometry", entry["raw"]["body_q"][:, :3], truth["origin"]
        )
        check(
            row,
            "body_twists_from_public_J",
            entry["raw"]["body_qd"],
            entry["jacobian_world_com"] @ entry["velocity"],
        )
        check(
            row,
            "public_parent_anchor_qd",
            entry["raw"]["joint_qd"],
            entry["world_to_native_velocity"] @ entry["velocity"],
        )
        check(
            row,
            "kinetic_mass_identity",
            entry["kinetic_energy_j"],
            entry["velocity"] @ entry["mass_world_com"] @ entry["velocity"] / 2,
        )
        if index == 0 and "literal_first_acceleration" in case:
            check(
                row,
                "literal_independent_first_acceleration",
                truth["acceleration"],
                case["literal_first_acceleration"],
                atol=1e-12,
                rtol=0,
            )
        state.clear_forces()
        state.body_f.assign(np.column_stack((truth["force"], truth["torque"])).astype(np.float32))
        row["state_body_f_before_step"] = state.body_f.numpy()
        solver.step(state, candidate, control, None, H)
        row["solver"] = inspect_native_step(solver, case, entry, truth)
        state, candidate = candidate, state
        row["accepted"] = observe(newton, model, state, case)
        after = row["accepted"]
        check(
            row,
            "decoded_force_is_original",
            row["solver"]["decoded_com_world_wrench"],
            row["solver"]["original_com_world_wrench"],
        )
        check(
            row,
            "decoded_acceleration",
            row["solver"]["decoded_world_com_acceleration"],
            truth["acceleration"],
        )
        if index == 0 and "literal_first_acceleration" in case:
            check(
                row,
                "literal_native_first_acceleration",
                row["solver"]["decoded_world_com_acceleration"],
                case["literal_first_acceleration"],
            )
        expected_v = entry["velocity"] + H * truth["acceleration"]
        check(row, "actual_velocity_step", after["velocity"], expected_v)
        check(
            row,
            "root_COM_drift",
            after["centers"][0],
            entry["centers"][0] + H * after["velocity"][:3],
        )
        native_quat = normalized_euler(entry["position"][3:7], after["velocity"][3:6], H)
        if native_quat @ after["position"][3:7] < 0:
            native_quat = -native_quat
        check(row, "native_normalized_Euler_quaternion", after["position"][3:7], native_quat)
        row["expmap_alternative_not_native_oracle"] = exponential(
            entry["position"][3:7], after["velocity"][3:6], H
        )
        if b == 2:
            check(
                row,
                "child_scalar_drift",
                after["position"][7],
                entry["position"][7] + H * after["velocity"][6],
            )
        expected_momentum += H * (
            truth["force"].sum(0) + sum(record["metadata"]["body_mass"]) * case["gravity"]
        )
        check(
            row,
            "cumulative_linear_impulse",
            after["linear_momentum"],
            expected_momentum,
            atol=WINDOW_ATOL,
        )
        if case["id"] in ("P01_freefall", "P03_internal", "P03_child_force", "P04_internal"):
            k = index + 1
            acceleration = case["literal_first_acceleration"]
            check(
                row,
                "closed_window_velocity",
                after["velocity"],
                case["velocity"] + k * H * acceleration,
                atol=WINDOW_ATOL,
            )
            check(
                row,
                "closed_window_root_COM",
                after["centers"][0],
                first["centers"][0]
                + k * H * case["velocity"][:3]
                + H * H * k * (k + 1) / 2 * acceleration[:3],
                atol=WINDOW_ATOL,
            )
            if b == 2:
                check(
                    row,
                    "closed_window_child_q",
                    after["position"][7],
                    case["joint_q"]
                    + k * H * case["velocity"][6]
                    + H * H * k * (k + 1) / 2 * acceleration[6],
                    atol=WINDOW_ATOL,
                )
        row["completed"] = True
    record["complete_steps"] = len(record["rows"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute():
        raise ValueError("--output must be absolute")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = "cuda:0" if args.device == "cuda" else args.device
    document = dict(
        status="running",
        device=device,
        h_s=H,
        declared_steps=STEPS,
        cases=[],
        errors=[],
        gates=dict(atol=ATOL, rtol=RTOL, mass_atol=MASS_ATOL, window_atol=WINDOW_ATOL),
        script=dict(
            path=str(Path(__file__).resolve()),
            sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
        semantics="No contacts or Phymeo state injection. Native Featherstone body COM/world observations, basis-transformed public M/J; normalized Euler orientation is separate from Phymeo exp-map.",
    )
    try:
        import newton
        import warp as wp

        if version("newton") != "1.7.0" or version("warp-lang") != "1.17.0":
            raise RuntimeError(
                "requires the existing pinned Newton 1.7.0 / Warp 1.17.0 environment"
            )
        document["versions"] = dict(
            newton=version("newton"),
            warp=version("warp-lang"),
            numpy=np.__version__,
            python=sys.version,
            platform=platform.platform(),
        )
        root = Path(newton.__file__).resolve().parent
        document["sources"] = {
            name: dict(
                path=str(root / name),
                sha256=hashlib.sha256((root / name).read_bytes()).hexdigest(),
                text=(root / name).read_text(encoding="utf-8"),
            )
            for name in SOURCES
        }
        wp.config.kernel_cache_dir = str(output / "warp-cache")
        wp.init()
        with wp.ScopedDevice(device):
            for case in cases():
                record = dict(id=case["id"], complete_steps=0)
                document["cases"].append(record)
                try:
                    run_case(newton, wp, case, device, record)
                except Exception as error:
                    record["execution_error"] = dict(
                        type=type(error).__name__,
                        message=str(error),
                        traceback=traceback.format_exc(),
                    )
        complete = all(
            row.get("complete_steps") == STEPS and "execution_error" not in row
            for row in document["cases"]
        )
        checks = all(
            check_row["passed"]
            for record in document["cases"]
            for block in [record, *record.get("rows", ())]
            for check_row in block.get("checks", {}).values()
        )
        document["status"] = (
            "passed" if complete and checks and len(document["cases"]) == 7 else "failed"
        )
    except Exception as error:
        document["status"] = "failed"
        document["errors"].append(
            dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        )
    encoded = json.dumps(encode(document), allow_nan=False, separators=(",", ":")).encode()
    if len(encoded) > MAX_BYTES:
        raise RuntimeError("complete native record exceeds predeclared 256 MiB bound")
    path = output / "native-floating.trace.json.gz"
    with gzip.open(path.with_suffix(".gz.tmp"), "wb") as stream:
        stream.write(encoded)
    path.with_suffix(".gz.tmp").replace(path)
    summary = dict(
        status=document["status"],
        raw_exit=int(document["status"] != "passed"),
        device=device,
        cases=[
            dict(
                id=row["id"],
                completed=sum(bool(item.get("completed")) for item in row.get("rows", ())),
                failed_checks=[
                    key for key, value in row.get("checks", {}).items() if not value["passed"]
                ],
                failed_steps=[
                    item["index"]
                    for item in row.get("rows", ())
                    if not all(c["passed"] for c in item.get("checks", {}).values())
                ],
                execution_error=row.get("execution_error"),
            )
            for row in document["cases"]
        ],
        trace=dict(
            path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            uncompressed_bytes=len(encoded),
        ),
        errors=document["errors"],
        versions=document.get("versions"),
    )
    (output / "summary.json").write_text(
        json.dumps(encode(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(encode(summary), allow_nan=False))
    return summary["raw_exit"]


if __name__ == "__main__":
    raise SystemExit(main())
