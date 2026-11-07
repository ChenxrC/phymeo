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
        root_collision_shape=f01, direct_parent_filter=True,
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
        collision_filter_parent=True, label="free",
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
