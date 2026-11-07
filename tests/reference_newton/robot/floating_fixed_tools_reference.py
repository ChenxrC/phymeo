"""Pinned Newton 1.7.0: uncollapsed free-root fixed tools, F01/F02 only.

No Phymeo import. Raw native arrays are saved before fixed analytical checks.
Run separately on cpu and cuda:0. Failed checks still write JSON and return 1.
"""

import argparse
import hashlib
import json
import math
import platform
import sys
import traceback
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np

H = 1 / 1024
# Absolute predeclared engineering gates for native FP32 arrays, never fitted.
GATES = {
    "position_m": 5e-6,
    "rotation": 5e-6,
    "mass_kg": 2e-6,
    "inertia_kg_m2": 5e-6,
    "velocity": 3e-6,
    "jacobian": 5e-6,
    "mass_matrix": 8e-6,
    "energy_j": 5e-6,
    "power_w": 5e-6,
    "first_acceleration": 2e-5,
    "first_velocity": 2e-8,
    "first_drift_m": 3e-8,
    "first_quaternion": 2e-8,
}
CYCLE = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
POSES = (
    ((0.0, 1 / 8), (0, 0, 0), (0, 0, 0, 1), np.eye(3)),
    ((0.37, -1 / 16), (1 / 2, -1 / 4, 3 / 4), (0.5, 0.5, 0.5, 0.5), CYCLE),
    ((-0.51, 3 / 16), (-1 / 4, 1 / 2, 0), (0, 0, 0, 1), np.eye(3)),
)
# Authored physical inputs, not values read back from a loaded model.
PARTS = (
    ("base", 2.0, (0, 0, 0), (1 / 5, 3 / 10, 2 / 5), 0),
    ("mount", 1 / 2, (0, 0, 0), (1 / 50, 3 / 100, 1 / 25), 0),
    ("arm", 1.0, (1 / 4, 0, 0), (1 / 20, 3 / 40, 1 / 10), 1),
    ("carriage", 1 / 2, (0, 0, 0), (1 / 40, 1 / 40, 1 / 40), 2),
    ("tool", 1 / 4, (0, 0, 0), (1 / 100, 1 / 50, 3 / 100), 2),
)
PHYSICAL_RATES = np.array([0.1, -0.2, 0.3, -0.1, 0.2, 0.1, 0.4, -0.3])
ZERO_PARAMETERS = (
    "joint_armature",
    "joint_damping",
    "joint_friction",
    "joint_target_ke",
    "joint_target_kd",
    "joint_limit_ke",
    "joint_limit_kd",
)
MODEL_ARRAYS = (
    "body_mass",
    "body_inv_mass",
    "body_com",
    "body_inertia",
    "body_inv_inertia",
    "body_flags",
    "body_world",
    "shape_type",
    "shape_body",
    "shape_transform",
    "shape_scale",
    "shape_flags",
    "shape_margin",
    "shape_gap",
    "joint_type",
    "joint_parent",
    "joint_child",
    "joint_X_p",
    "joint_X_c",
    "joint_axis",
    "joint_dof_dim",
    "joint_q_start",
    "joint_qd_start",
    "joint_limit_lower",
    "joint_limit_upper",
    "joint_effort_limit",
    "joint_velocity_limit",
    "joint_mimic_joint",
    "joint_mimic_coeffs",
    "joint_enabled",
    "joint_articulation",
    "joint_world",
    "joint_ancestor",
    "joint_target_mode",
    "joint_target_q",
    "joint_target_qd",
    "joint_act",
    "articulation_start",
    "articulation_end",
    "gravity",
    *ZERO_PARAMETERS,
)
NATIVE_SOURCES = (
    "_src/utils/import_urdf.py",
    "_src/sim/builder.py",
    "_src/sim/model.py",
    "_src/sim/state.py",
    "_src/sim/articulation.py",
    "_src/solvers/featherstone/solver_featherstone.py",
    "_src/solvers/featherstone/kernels.py",
)


def rotation(xyzw):
    x, y, z, w = np.asarray(xyzw, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def skew(point):
    x, y, z = point
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def close(row, name, actual, expected, tolerance):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    same_shape = a.shape == b.shape
    finite = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
    error = float(np.max(np.abs(a - b), initial=0.0)) if same_shape and finite else None
    row["checks"][name] = {
        "passed": bool(error is not None and error <= tolerance),
        "max_abs_error": error,
        "absolute_limit": tolerance,
        "actual_shape": a.shape,
        "expected_shape": b.shape,
    }


def condition(row, name, passed, detail=None):
    row["checks"][name] = {"passed": bool(passed), "detail": detail}


def require(passed, message):
    if not passed:
        raise ValueError(message)


def snapshot(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "utf8": data.decode("utf-8-sig"),
    }


def read_array(owner, name):
    value = getattr(owner, name)
    return None if value is None else value.numpy().copy()


def build(newton, path, device, row, *, probe):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    builder.default_shape_cfg.density = 0.0
    builder.add_urdf(
        str(path),
        floating=True,
        collapse_fixed_joints=False,
        collapse_massless_fixed_root=False,
        ignore_inertial_definitions=False,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
        joint_damping_mode="passive",
        up_axis=newton.Axis.Z,
    )
    row["import_options"] = {
        "floating": True,
        "collapse_fixed_joints": False,
        "collapse_massless_fixed_root": False,
        "ignore_inertial_definitions": False,
        "enable_self_collisions": False,
        "parse_visuals_as_colliders": False,
        "shape_density_kg_m3": 0.0,
        "up_axis": "Z",
        "gravity_m_s2": [0, 0, 0],
        "contacts_passed_to_step": None,
        "angular_damping": 0.0,
    }
    row["parser_parameters_before_explicit_zero"] = {
        name: list(getattr(builder, name)) for name in ZERO_PARAMETERS
    }
    # Isolate external forcing; preserve real fixed joints and limit metadata.
    for name in ZERO_PARAMETERS:
        values = getattr(builder, name)
        for index in range(len(values)):
            values[index] = 0.0
    model = builder.finalize(device=device)
    meta = {name: read_array(model, name) for name in MODEL_ARRAYS}
    for name in ("body_label", "joint_label", "shape_label"):
        meta[name] = list(getattr(model, name))
    meta.update(
        device=str(model.device),
        body_count=model.body_count,
        joint_count=model.joint_count,
        joint_coord_count=model.joint_coord_count,
        joint_dof_count=model.joint_dof_count,
        shape_count=model.shape_count,
        articulation_count=model.articulation_count,
    )
    row["native_model"] = meta
    names = [str(label).rsplit("/", 1)[-1] for label in model.body_label]
    joint_names = [str(label).rsplit("/", 1)[-1] for label in model.joint_label]
    require(len(set(names)) == len(names), "Ambiguous native body labels")
    require(len(set(joint_names)) == len(joint_names), "Ambiguous native joint labels")
    bodies = dict(zip(names, range(len(names))))
    joints = dict(zip(joint_names, range(len(joint_names))))
    expected_bodies = (
        {"base", "mount", "arm", "carriage", "tool", "witness"} if probe else {"base", "tool"}
    )
    require(
        set(bodies) == expected_bodies, "Native body set differs: fixed parts must not be collapsed"
    )
    types = meta["joint_type"]
    free = np.flatnonzero(types == int(newton.JointType.FREE))
    require(len(free) == 1, "Exactly one explicit native FREE joint required")
    root_joint = int(free[0])
    require(int(meta["joint_parent"][root_joint]) == -1, "FREE root must be connected to world")
    require(
        int(meta["joint_child"][root_joint]) == bodies["base"], "FREE root must own authored base"
    )
    expected_joints = (
        {
            "mount_fixed": ("base", "mount", newton.JointType.FIXED),
            "hinge": ("mount", "arm", newton.JointType.REVOLUTE),
            "extension": ("arm", "carriage", newton.JointType.PRISMATIC),
            "tool_fixed": ("carriage", "tool", newton.JointType.FIXED),
            "witness_fixed": ("tool", "witness", newton.JointType.FIXED),
        }
        if probe
        else {"tool_mount": ("base", "tool", newton.JointType.FIXED)}
    )
    require(model.joint_count == len(expected_joints) + 1, "Unexpected native joint count")
    for name, (parent, child, kind) in expected_joints.items():
        index = joints[name]
        require(int(types[index]) == int(kind), f"Native joint type changed: {name}")
        require(
            int(meta["joint_parent"][index]) == bodies[parent]
            and int(meta["joint_child"][index]) == bodies[child],
            f"Native joint ownership changed: {name}",
        )
    condition(row, "free_root_and_all_authored_fixed_joints_retained", True)
    condition(row, "requested_device", str(model.device) == device, str(model.device))
    close(
        row,
        "coordinate_velocity_dimensions",
        [model.joint_coord_count, model.joint_dof_count],
        [9, 8] if probe else [7, 6],
        0,
    )
    root_v = int(meta["joint_qd_start"][root_joint])
    columns = list(range(root_v, root_v + 6))
    if probe:
        columns.extend(int(meta["joint_qd_start"][joints[name]]) for name in ("hinge", "extension"))
    require(model.articulation_count == 1, "A single uncollapsed articulation is required")
    return model, meta, bodies, joints, root_joint, columns


def state_record(newton, model, meta, state):
    raw = {
        name: read_array(state, name)
        for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f")
    }
    poses, twists, force = raw["body_q"], raw["body_qd"], raw["body_f"]
    entries, total_p, total_l, kinetic = [], np.zeros(3), np.zeros(3), 0.0
    for index, pose in enumerate(poses):
        orient = rotation(pose[3:])
        center = pose[:3] + orient @ meta["body_com"][index]
        inertia = orient @ meta["body_inertia"][index] @ orient.T
        velocity, omega = twists[index, :3], twists[index, 3:]
        mass = float(meta["body_mass"][index])
        momentum = mass * velocity
        spin = inertia @ omega
        total_p += momentum
        total_l += np.cross(center, momentum) + spin
        kinetic += float(velocity @ momentum + omega @ spin) / 2
        entries.append(
            {
                "body_id": index,
                "native_label": meta["body_label"][index],
                "mass_kg": mass,
                "com_body_m": meta["body_com"][index],
                "inertia_body_about_com_kg_m2": meta["body_inertia"][index],
                "origin_world_m": pose[:3],
                "rotation_world_from_body": orient,
                "com_world_m": center,
                "inertia_world_about_com_kg_m2": inertia,
                "velocity_com_world_m_s": velocity,
                "omega_world_rad_s": omega,
                "velocity_origin_world_m_s": velocity + np.cross(omega, pose[:3] - center),
                "velocity_com_body_m_s": orient.T @ velocity,
                "omega_body_rad_s": orient.T @ omega,
                "force_com_world_n": force[index, :3],
                "torque_com_world_nm": force[index, 3:],
                "force_com_body_n": orient.T @ force[index, :3],
                "torque_com_body_nm": orient.T @ force[index, 3:],
                "torque_about_world_origin_nm": force[index, 3:]
                + np.cross(center, force[index, :3]),
            }
        )
    shapes = []
    for index, owner in enumerate(meta["shape_body"]):
        require(owner >= 0, "Unexpected world-owned shape")
        part = entries[int(owner)]
        local = meta["shape_transform"][index]
        orient = part["rotation_world_from_body"] @ rotation(local[3:])
        center = part["origin_world_m"] + part["rotation_world_from_body"] @ local[:3]
        kind = int(meta["shape_type"][index])
        scale = meta["shape_scale"][index].astype(float)
        if kind == int(newton.GeoType.BOX):
            points = (
                np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * scale
            )
            kind_name = "BOX"
        elif kind == int(newton.GeoType.SPHERE):
            points = np.vstack((np.eye(3), -np.eye(3))) * scale[0]
            kind_name = "SPHERE"
        else:
            raise ValueError(f"Unexpected fixture geometry type: {kind}")
        shapes.append(
            {
                "shape_id": index,
                "native_label": meta["shape_label"][index],
                "body_id": int(owner),
                "body_label": meta["body_label"][int(owner)],
                "native_geo_type": kind,
                "native_geo_type_name": kind_name,
                "local_pose_xyz_xyzw": local,
                "native_scale_m": scale,
                "origin_world_m": center,
                "rotation_world_from_shape": orient,
                "sample_point_ids": list(range(len(points))),
                "sample_points_shape_m": points,
                "sample_points_world_m": center + points @ orient.T,
                "point_semantics": "8 box vertices or 6 sphere axis surface samples; not contact rows",
            }
        )
    return {
        "raw": raw,
        "raw_dtypes": {name: str(value.dtype) for name, value in raw.items()},
        "device": str(state.body_q.device),
        "bodies": entries,
        "shapes": shapes,
        "linear_momentum_world_kg_m_s": total_p,
        "angular_momentum_about_world_origin_kg_m2_s": total_l,
        "kinetic_energy_j": kinetic,
    }


def aggregate(parts):
    mass = sum(p["mass_kg"] for p in parts)
    center = sum(p["mass_kg"] * p["com_world_m"] for p in parts) / mass
    inertia, velocity = np.zeros((3, 3)), np.zeros(3)
    for part in parts:
        arm = part["com_world_m"] - center
        inertia += part["inertia_world_about_com_kg_m2"] + part["mass_kg"] * (
            (arm @ arm) * np.eye(3) - np.outer(arm, arm)
        )
        velocity += part["mass_kg"] * part["velocity_com_world_m_s"] / mass
    return {
        "mass_kg": mass,
        "com_world_m": center,
        "inertia_world_about_com_kg_m2": inertia,
        "velocity_com_world_m_s": velocity,
    }


def audit_shapes(newton, row, observed, frames, bodies, *, probe):
    specs = (
        ("base", "base_marker", "sphere", 1 / 32 if probe else 1 / 16),
        (
            "tool",
            "tip_box" if probe else "tool_box",
            "box",
            (1 / 8, 1 / 4 if probe else 1 / 8, 1 / 8),
        ),
    )
    close(row, "actual_collision_shape_count", len(observed["shapes"]), 2, 0)
    for name, authored_name, kind, dimension in specs:
        found = [shape for shape in observed["shapes"] if shape["body_id"] == bodies[name]]
        require(len(found) == 1, f"Expected exactly one actual collision shape on {name}")
        shape = found[0]
        prefix = f"shape.{name}.{authored_name}"
        shape["authored_collision_name"] = authored_name
        shape["authored_name_mapping"] = (
            "fixture link ownership and unique geometry; importer label is separately retained"
        )
        close(
            row,
            prefix + ".type",
            shape["native_geo_type"],
            int(newton.GeoType.SPHERE if kind == "sphere" else newton.GeoType.BOX),
            0,
        )
        origin, orient = frames[name]
        close(row, prefix + ".world_origin_m", shape["origin_world_m"], origin, GATES["position_m"])
        close(
            row,
            prefix + ".world_orientation",
            shape["rotation_world_from_shape"],
            orient,
            GATES["rotation"],
        )
        close(
            row,
            prefix + ".local_pose",
            shape["local_pose_xyz_xyzw"],
            [0, 0, 0, 0, 0, 0, 1],
            GATES["rotation"],
        )
        if kind == "sphere":
            close(
                row,
                prefix + ".radius_m",
                shape["native_scale_m"][0],
                dimension,
                GATES["position_m"],
            )
            points = np.vstack((np.eye(3), -np.eye(3))) * dimension
        else:
            half = np.asarray(dimension) / 2
            close(
                row, prefix + ".half_extents_m", shape["native_scale_m"], half, GATES["position_m"]
            )
            points = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
        close(
            row,
            prefix + ".material_samples_world_m",
            shape["sample_points_world_m"],
            origin + points @ orient.T,
            GATES["position_m"],
        )


def run_f01(newton, path, device, row):
    model, meta, bodies, _, root_joint, columns = build(newton, path, device, row, probe=False)
    state, candidate, control = model.state(), model.state(), model.control()
    state.clear_forces()
    candidate.clear_forces()
    state.joint_qd.zero_()
    control.joint_f.zero_()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    force = np.zeros((model.body_count, 6), dtype=np.float32)
    force[bodies["tool"], :3] = (0, 3, 0)
    # TOOL COM load, shifted once by the tree. Adding the aggregate-COM moment
    # (0,0,1/2) here would double count the lever arm.
    state.body_f.assign(force)
    before = state_record(newton, model, meta, state)
    row.update(
        dt_s=H,
        before=before,
        applied_body_f_world_com=force.copy(),
        control_joint_f=control.joint_f.numpy().copy(),
    )
    combined = aggregate(before["bodies"])
    row["before_aggregate"] = combined
    close(row, "total_mass_kg", combined["mass_kg"], 3, GATES["mass_kg"])
    close(row, "aggregate_com_m", combined["com_world_m"], [1 / 12, 0, 0], GATES["position_m"])
    close(
        row,
        "aggregate_inertia_kg_m2",
        combined["inertia_world_about_com_kg_m2"],
        np.diag([3 / 10, 53 / 120, 13 / 24]),
        GATES["inertia_kg_m2"],
    )
    frames = {"base": (np.zeros(3), np.eye(3)), "tool": (np.array([1 / 4, 0, 0]), np.eye(3))}
    audit_shapes(newton, row, before, frames, bodies, probe=False)
    for name, mass, diagonal in (("base", 2, (1 / 5, 3 / 10, 2 / 5)), ("tool", 1, (1 / 10,) * 3)):
        part = before["bodies"][bodies[name]]
        close(row, f"part.{name}.mass_kg", part["mass_kg"], mass, GATES["mass_kg"])
        close(row, f"part.{name}.com_body_m", part["com_body_m"], np.zeros(3), GATES["position_m"])
        close(
            row,
            f"part.{name}.inertia_body",
            part["inertia_body_about_com_kg_m2"],
            np.diag(diagonal),
            GATES["inertia_kg_m2"],
        )
    row["independent_oracle"] = {
        "mass_kg": 3,
        "com_world_m": [1 / 12, 0, 0],
        "inertia_diagonal_kg_m2": [3 / 10, 53 / 120, 13 / 24],
        "force_at_tool_com_n": [0, 3, 0],
        "torque_about_aggregate_com_nm": [0, 0, 1 / 2],
        "aggregate_acceleration_m_s2": [0, 1, 0],
        "angular_acceleration_rad_s2": [0, 0, 12 / 13],
        "base_com_acceleration_m_s2": [0, 12 / 13, 0],
        "tool_com_acceleration_m_s2": [0, 15 / 13, 0],
    }
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    solver.step(state, candidate, control, None, H)
    # Read the actual native end state; no host-integrated pose replaces it.
    after = state_record(newton, model, meta, candidate)
    row["after"] = after
    row["input_after_step"] = state_record(newton, model, meta, state)
    final = aggregate(after["bodies"])
    row["after_aggregate"] = final
    close(
        row,
        "aggregate_acceleration_from_actual_velocity",
        (final["velocity_com_world_m_s"] - combined["velocity_com_world_m_s"]) / H,
        [0, 1, 0],
        GATES["first_acceleration"],
    )
    for name, ay in (("base", 12 / 13), ("tool", 15 / 13)):
        part = after["bodies"][bodies[name]]
        close(
            row,
            f"{name}.actual_velocity_m_s",
            part["velocity_com_world_m_s"],
            [0, ay * H, 0],
            GATES["first_velocity"],
        )
        close(
            row,
            f"{name}.actual_angular_acceleration_rad_s2",
            part["omega_world_rad_s"] / H,
            [0, 0, 12 / 13],
            GATES["first_acceleration"],
        )
    base = after["bodies"][bodies["base"]]
    close(
        row,
        "base_origin_semimplicit_drift_m",
        base["origin_world_m"],
        [0, (12 / 13) * H * H, 0],
        GATES["first_drift_m"],
    )
    close(
        row,
        "aggregate_com_first_drift_m",
        final["com_world_m"],
        [1 / 12, H * H, 0],
        GATES["first_drift_m"],
    )
    # Native FREE uses normalized Euler, not an exponential-map quaternion.
    theta = 2 * math.atan((12 / 13) * H * H / 2)
    base_pose = after["raw"]["body_q"][bodies["base"]]
    close(
        row,
        "native_normalized_euler_xyzw",
        base_pose[3:],
        [0, 0, math.sin(theta / 2), math.cos(theta / 2)],
        GATES["first_quaternion"],
    )
    close(
        row,
        "linear_impulse_ns",
        after["linear_momentum_world_kg_m_s"] - before["linear_momentum_world_kg_m_s"],
        [0, 3 * H, 0],
        6e-8,
    )
    close(
        row,
        "angular_impulse_about_world_origin_nms",
        after["angular_momentum_about_world_origin_kg_m2_s"]
        - before["angular_momentum_about_world_origin_kg_m2_s"],
        [0, 0, 3 * H / 4],
        6e-8,
    )
    for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"):
        close(
            row,
            "input_not_mutated." + name,
            row["input_after_step"]["raw"][name],
            before["raw"][name],
            0,
        )
    row["native_free_velocity_columns"] = columns
    row["native_free_coordinate_start"] = int(meta["joint_q_start"][root_joint])
    row["scope"] = (
        "One actual Featherstone step; initial-rest finite-difference acceleration; no contact pipeline or sustained window"
    )


def probe_oracle(joints, position, orient):
    theta, extension = joints
    position = np.asarray(position, dtype=float)
    arm_r, tool_r = orient @ rz(theta), orient @ rz(theta) @ rz(math.pi / 2)
    pivot = position + orient @ [1 / 4, 0, 0]
    carriage = pivot + arm_r @ [1 / 2 + extension, 0, 0]
    tool = carriage + arm_r @ [1 / 8, 0, 0]
    frames = {
        "base": (position, orient),
        "mount": (pivot, orient),
        "arm": (pivot, arm_r),
        "carriage": (carriage, arm_r),
        "tool": (tool, tool_r),
        "witness": (tool + tool_r @ [1 / 32, 0, 0], tool_r),
    }
    group_com = position + orient @ [1 / 20, 0, 0]

    def jacobian(point, group):
        linear, angular = np.zeros((3, 8)), np.zeros((3, 8))
        linear[:, :3], linear[:, 3:6], angular[:, 3:6] = (
            np.eye(3),
            -skew(point - group_com),
            np.eye(3),
        )
        if group > 0:
            linear[:, 6], angular[:, 6] = np.cross(orient[:, 2], point - pivot), orient[:, 2]
        if group == 2:
            linear[:, 7] = arm_r[:, 0]
        return np.vstack((linear, angular))

    parts, mass = {}, np.zeros((8, 8))
    for name, weight, local_com, diagonal, group in PARTS:
        origin, part_r = frames[name]
        center = origin + part_r @ local_com
        inertia = part_r @ np.diag(diagonal) @ part_r.T
        jac = jacobian(center, group)
        mass += weight * jac[:3].T @ jac[:3] + jac[3:].T @ inertia @ jac[3:]
        parts[name] = {
            "mass_kg": weight,
            "com_body_m": np.array(local_com),
            "inertia_body_about_com_kg_m2": np.diag(diagonal),
            "com_world_m": center,
            "inertia_world_about_com_kg_m2": inertia,
            "jacobian_physical_basis": jac,
            "twist_world_com": jac @ PHYSICAL_RATES,
        }
    # Native root COM is base COM; the comparison's root COM includes mount.
    # v_base = v_group + omega x (C_base - C_group), so this is a fixed basis map.
    basis = np.eye(8)
    basis[:3, 3:6] = -skew(position - group_com)
    return frames, parts, mass, basis, jacobian


def run_f02(newton, path, device, row, case):
    joints_q, position, xyzw, orient = case
    model, meta, bodies, joints, root_joint, columns = build(newton, path, device, row, probe=True)
    frames, parts, expected_mass, basis, point_j = probe_oracle(joints_q, position, orient)
    state = model.state()
    state.clear_forces()
    q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
    root_q = int(meta["joint_q_start"][root_joint])
    q[root_q : root_q + 7] = (*position, *xyzw)
    for name, value in zip(("hinge", "extension"), joints_q, strict=True):
        q[int(meta["joint_q_start"][joints[name]])] = value
    qd[:] = 0
    qd[columns] = basis @ PHYSICAL_RATES
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    observed = state_record(newton, model, meta, state)
    row.update(
        inputs={
            "q_R_q_P": joints_q,
            "root_origin_m": position,
            "root_xyzw": xyzw,
            "physical_v_root_group_com_omega_joint_rates": PHYSICAL_RATES,
            "native_from_physical_rate_matrix": basis,
            "native_rate_columns": columns,
        },
        native=observed,
        independent_frames={
            name: {"origin_world_m": p, "rotation": r} for name, (p, r) in frames.items()
        },
        independent_components=parts,
        independent_mass_matrix=expected_mass,
    )
    jacobian = newton.eval_jacobian(model, state)
    native_j = jacobian.numpy().astype(float)
    native_mass = newton.eval_mass_matrix(model, state, J=jacobian).numpy().astype(float)
    row["native_jacobian_raw"] = native_j
    row["native_mass_matrix_raw"] = native_mass
    mapped_mass = basis.T @ native_mass[0][np.ix_(columns, columns)] @ basis
    row["native_mass_matrix_physical_basis"] = mapped_mass
    close(row, "full_mass_matrix_physical_basis", mapped_mass, expected_mass, GATES["mass_matrix"])
    close(
        row,
        "kinetic_energy_five_original_components_j",
        observed["kinetic_energy_j"],
        PHYSICAL_RATES @ expected_mass @ PHYSICAL_RATES / 2,
        GATES["energy_j"],
    )
    mapped_j = {}
    start = int(meta["articulation_start"][0])
    for name, (p, r) in frames.items():
        body = bodies[name]
        actual = observed["bodies"][body]
        close(row, name + ".origin_m", actual["origin_world_m"], p, GATES["position_m"])
        close(row, name + ".rotation", actual["rotation_world_from_body"], r, GATES["rotation"])
        (joint_id,) = np.flatnonzero(meta["joint_child"] == body)
        native_slice = native_j[0, 6 * (int(joint_id) - start) : 6 * (int(joint_id) - start + 1)]
        mapped_j[name] = native_slice[:, columns] @ basis
        if name in parts:
            expected = parts[name]
            for field, gate in (
                ("mass_kg", "mass_kg"),
                ("com_body_m", "position_m"),
                ("com_world_m", "position_m"),
                ("inertia_body_about_com_kg_m2", "inertia_kg_m2"),
                ("inertia_world_about_com_kg_m2", "inertia_kg_m2"),
            ):
                close(row, name + "." + field, actual[field], expected[field], GATES[gate])
            close(
                row,
                name + ".com_twist",
                observed["raw"]["body_qd"][body],
                expected["twist_world_com"],
                GATES["velocity"],
            )
            close(
                row,
                name + ".full_jacobian",
                mapped_j[name],
                expected["jacobian_physical_basis"],
                GATES["jacobian"],
            )
        else:
            close(row, "witness.mass_kg", actual["mass_kg"], 0, 0)
            close(
                row,
                "witness.zero_inertia",
                actual["inertia_body_about_com_kg_m2"],
                np.zeros((3, 3)),
                0,
            )
            close(
                row,
                "witness.material_twist",
                observed["raw"]["body_qd"][body],
                point_j(p, 2) @ PHYSICAL_RATES,
                GATES["velocity"],
            )
    row["native_body_jacobians_physical_basis"] = mapped_j
    audit_shapes(newton, row, observed, frames, bodies, probe=True)
    for name, limits, effort, speed, axis in (
        ("hinge", (-1.5, 1.5), 16, 4, (0, 0, 1)),
        ("extension", (-1 / 8, 1 / 4), 32, 2, (1, 0, 0)),
    ):
        index = int(meta["joint_qd_start"][joints[name]])
        close(
            row,
            name + ".limits",
            [meta["joint_limit_lower"][index], meta["joint_limit_upper"][index]],
            limits,
            GATES["position_m"],
        )
        close(
            row,
            name + ".effort_velocity_limits",
            [meta["joint_effort_limit"][index], meta["joint_velocity_limit"][index]],
            [effort, speed],
            GATES["mass_kg"],
        )
        close(row, name + ".local_axis", meta["joint_axis"][index], axis, GATES["rotation"])
    local_point = np.array([1 / 32, -1 / 64, 0.0])
    tool = observed["bodies"][bodies["tool"]]
    point = tool["origin_world_m"] + tool["rotation_world_from_body"] @ local_point
    offset = point - tool["com_world_m"]
    measured_j = mapped_j["tool"][:3] - skew(offset) @ mapped_j["tool"][3:]
    measured_v = tool["velocity_com_world_m_s"] + np.cross(tool["omega_world_rad_s"], offset)
    expected_point = frames["tool"][0] + frames["tool"][1] @ local_point
    expected_j = point_j(expected_point, 2)[:3]
    force = orient @ [-1.0, 2.0, 0.5]
    row["tool_material_readout"] = {
        "local_point_m": local_point,
        "point_world_m": point,
        "velocity_world_m_s": measured_v,
        "jacobian_physical_basis": measured_j,
        "virtual_force_world_n": force,
        "virtual_power_w": float(force @ measured_v),
        "force_is_applied": False,
    }
    close(row, "tool_material_point_m", point, expected_point, GATES["position_m"])
    close(row, "tool_material_jacobian", measured_j, expected_j, GATES["jacobian"])
    close(
        row,
        "tool_material_velocity_m_s",
        measured_v,
        expected_j @ PHYSICAL_RATES,
        GATES["velocity"],
    )
    close(
        row,
        "tool_virtual_power_w",
        force @ measured_v,
        force @ expected_j @ PHYSICAL_RATES,
        GATES["power_w"],
    )
    row["scope"] = (
        "Actual native FK/J/M at one frozen state; virtual work only; no F02 dynamics or F03 contact window"
    )


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--composite-urdf", type=Path, required=True)
    parser.add_argument("--probe-urdf", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="New JSON path; never overwrites prior evidence"
    )
    args = parser.parse_args()
    require(not args.out.exists(), "Output exists; use a new path to preserve previous evidence")
    report = {
        "status": "RUNNING",
        "device_requested": args.device,
        "absolute_gates": GATES,
        "versions": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "source_snapshot": {
            "runner": snapshot(__file__),
            "composite_urdf": snapshot(args.composite_urdf),
            "probe_urdf": snapshot(args.probe_urdf),
        },
        "semantics": {
            "pose": "authored body origin xyz [m], XYZW quaternion",
            "body_qd": "world (v_at_each_body_COM [m/s], omega [rad/s])",
            "body_f": "world (force [N], torque_about_each_body_COM [Nm])",
            "free_qd": "base-body COM velocity and angular velocity in world-aligned parent frame; not aggregate-COM velocity",
            "fixed_policy": "floating=True; both fixed-collapse options False; fixed bodies retained",
            "readback": "raw native arrays plus algebraic body/world transforms; geometry samples are not contact observations",
        },
        "records": [],
        "execution_error": None,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            import newton
            import warp as wp

            report["versions"].update(
                newton=version("newton"), warp=version("warp-lang"), numpy=version("numpy")
            )
            require(
                version("newton") == "1.7.0" and version("warp-lang") == "1.17.0",
                "Pinned Newton 1.7.0 / Warp 1.17.0 required",
            )
            native_root = Path(newton.__file__).resolve().parent
            report["source_snapshot"]["native"] = {
                name: snapshot(native_root / name) for name in NATIVE_SOURCES
            }
            wp.config.kernel_cache_dir = str(Path(sys.prefix) / "warp-kernels")
            wp.init()
            device = wp.get_device(args.device)
            report["device_actual"] = {
                "alias": str(device),
                "name": device.name,
                "is_cuda": device.is_cuda,
            }
            require(
                (args.device == "cuda:0") == device.is_cuda, "Requested device was not selected"
            )
            with wp.ScopedDevice(args.device):
                cases = (
                    ("F01", None),
                    *((f"F02-{index}", pose) for index, pose in enumerate(POSES)),
                )
                for case_id, case in cases:
                    row = {"id": case_id, "checks": {}, "execution_error": None}
                    report["records"].append(row)
                    try:
                        if case is None:
                            run_f01(newton, args.composite_urdf.resolve(), args.device, row)
                        else:
                            run_f02(newton, args.probe_urdf.resolve(), args.device, row, case)
                    except Exception:
                        row["execution_error"] = traceback.format_exc()
                    row["accepted"] = (
                        row["execution_error"] is None
                        and bool(row["checks"])
                        and all(check["passed"] for check in row["checks"].values())
                    )
        except Exception:
            report["execution_error"] = traceback.format_exc()
        report["warnings"] = [str(item.message) for item in caught]
    success = (
        report["execution_error"] is None
        and len(report["records"]) == 4
        and all(row["accepted"] for row in report["records"])
    )
    report["status"] = "PASSED" if success else "FAILED"
    report["accepted"] = success
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(json_value(report), stream, allow_nan=False, indent=2)
    summary = {
        "status": report["status"],
        "device": args.device,
        "output": str(args.out.resolve()),
        "execution_error": report["execution_error"],
        "cases": [
            {
                "id": row["id"],
                "accepted": row["accepted"],
                "execution_error": row["execution_error"],
                "failed_checks": [
                    name for name, check in row["checks"].items() if not check["passed"]
                ],
            }
            for row in report["records"]
        ],
    }
    print(json.dumps(json_value(summary), allow_nan=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
