"""Frozen ordinary R/P tree oracles and pinned Newton 1.7 reference windows.

This file never imports Phymeo. NumPy helpers are independent authored-model
oracles; Newton/Warp are imported only by the CLI. No reference has been run by
the author. All poses, wrenches, controls, and native rows are saved before gates.
"""

import argparse
import hashlib
import importlib.metadata
import json
from math import atan2, cos, pi, sin
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


H = 1 / 1024
RTOL = 5e-5
ATOL = 5e-6
WINDOW_ATOL = 2e-4


def quaternion_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float)
    return np.array([[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)]])


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def rotation(axis, angle):
    k = skew(axis)
    return np.eye(3) + sin(angle) * k + (1 - cos(angle)) * (k @ k)


def _model(parents, kinds, offsets, axes, mass, com, inertia):
    n = len(parents)
    return dict(parents=np.array(parents, dtype=int), kinds=tuple(kinds),
                offsets=np.array(offsets, dtype=float),
                orientations=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
                axes=np.array(axes, dtype=float), mass=np.array(mass, dtype=float),
                com=np.array(com, dtype=float), inertia=np.array(inertia, dtype=float),
                gravity=np.zeros(3), names=tuple(f"link{i}" for i in range(n)),
                joint_names=tuple(f"joint{i}" for i in range(n)))


def model_data(name):
    if name in ("T01", "T03"):
        model = _model([-1, 0], ["revolute"]*2, [[0, 0, 0], [1, 0, 0]],
                       [[0, 0, 1]]*2, [1, 1], [[0.5, 0, 0]]*2,
                       [[1/24, 1/24, 1/12]]*2)
        if name == "T03":
            model["parents"] = np.array([-1, 0, 1])
            model["kinds"] = ("revolute", "revolute", "fixed")
            model["offsets"] = np.vstack((model["offsets"], [0.75, 0, 0]))
            model["orientations"] = np.vstack((model["orientations"], [0, 0, sin(pi/4), cos(pi/4)]))
            model["axes"] = np.vstack((model["axes"], [0, 0, 1]))
            model["mass"] = np.array([1, 1, 0.5])
            model["com"] = np.vstack((model["com"], [0.25, 0, 0]))
            model["inertia"] = np.vstack((model["inertia"], [1/64, 1/32, 5/128]))
            model["names"] = ("link0", "link1", "tool")
            model["joint_names"] = ("joint0", "joint1", "tool_fixed")
        return model
    if name == "T02":
        return _model([-1, 0, 0], ["revolute", "prismatic", "prismatic"],
                      [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                      [[0, 0, 1], [0, 1, 0], [1, 0, 0]], [2, 1, 1],
                      np.zeros((3, 3)), [[1/8, 1/8, 1/4], [1/16, 1/16, 1/8], [1/16, 1/16, 1/8]])
    if name == "T04":
        model = _model([-1, 0], ["revolute", "prismatic"],
                       [[0, 0, 0], [0.5, 0, 0]], [[3/5, 0, 4/5], [0, 1, 0]],
                       [2, 1], [[0, 0.25, 0], [0, 0.125, 0]],
                       [[1/8, 1/4, 3/8], [1/16, 1/8, 3/16]])
        model["gravity"] = np.array([0, 0, -2.0])
        return model
    if name in ("N9", "N16", "N17"):
        n = int(name[1:])
        return _model(np.arange(n)-1,
                      tuple("revolute" if i % 2 == 0 else "prismatic" for i in range(n)),
                      [[1/8, 1/16, 0]]*n,
                      [((0, 0, 1), (1, 0, 0), (0, 1, 0))[i % 3] for i in range(n)],
                      [1+(i % 3)/2 for i in range(n)], [[1/16, 1/32, 0]]*n,
                      [[1/64, 1/32, 3/64]]*n)
    raise ValueError(f"unknown frozen case: {name}")


def coordinates(model):
    return tuple(i for i, kind in enumerate(model["kinds"]) if kind != "fixed")


def case_input(name):
    model = model_data(name)
    n = len(coordinates(model))
    q, v, effort = np.zeros(n), np.zeros(n), np.zeros(n)
    force, torque = np.zeros((len(model["mass"]), 3)), np.zeros((len(model["mass"]), 3))
    if name in ("T01", "T03"):
        effort[:] = [1, 1/6]
    elif name == "T02":
        effort[:] = [0, 1, 0]
    elif name == "T04":
        q[:], v[:], effort[:] = [0.37, -1/16], [1/4, -1/8], [1/8, -1/4]
        force[:] = [[0, -1, 0], [1, 2, 3]]
        torque[:] = [[1/8, 0, 0], [1/4, -1/2, 3/4]]
    else:
        q[:] = [(i % 3 - 1)/(8 if kind == "revolute" else 64)
                for i, kind in enumerate(model["kinds"])]
        effort[:] = [(-1)**i/16 for i in range(n)]
    return model, q, v, effort, force, torque


def transformed(model, *, cycled=False, translation=(0, 0, 0)):
    out = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in model.items()}
    r = quaternion_matrix([0.5]*4) if cycled else np.eye(3)
    for i, parent in enumerate(out["parents"]):
        if parent < 0:
            out["offsets"][i] = r @ out["offsets"][i] + np.asarray(translation)
            # Frozen models have identity root orientation before this global transform.
            if not np.array_equal(out["orientations"][i], [0, 0, 0, 1]):
                raise ValueError("global transform helper expects an authored identity root")
            if cycled:
                out["orientations"][i] = [0.5]*4
    out["gravity"] = r @ out["gravity"]
    return out, r


def _vee(m):
    return np.array([m[2, 1]-m[1, 2], m[0, 2]-m[2, 0], m[1, 0]-m[0, 1]]) / 2


def oracle(model, q, v):
    """Homogeneous-transform derivatives, not a simulator's FK/RNEA routine.

    First partial derivatives yield all point/angular Jacobians. Time derivatives
    at qdd=0 yield actual COM bias acceleration and alpha; no finite differences.
    Includes authored FIXED components as independent physical inertia terms.
    """
    q, v = np.asarray(q, dtype=float), np.asarray(v, dtype=float)
    active = coordinates(model)
    n, nb = len(active), len(model["mass"])
    column = {body: j for j, body in enumerate(active)}
    if q.shape != (n,) or v.shape != (n,):
        raise ValueError("oracle coordinates do not match authored active joints")
    transforms, derivatives, first, second = [], [], [], []
    jv, jw = np.zeros((nb, 3, n)), np.zeros((nb, 3, n))
    com_world, omega, alpha, acc = (np.zeros((nb, 3)) for _ in range(4))
    for body, kind in enumerate(model["kinds"]):
        static = np.eye(4)
        static[:3, :3] = quaternion_matrix(model["orientations"][body])
        static[:3, 3] = model["offsets"][body]
        motion, dm, ddm = np.eye(4), np.zeros((4, 4)), np.zeros((4, 4))
        speed = 0.0
        if body in column:
            j = column[body]
            axis = model["axes"][body]
            speed = v[j]
            if kind == "revolute":
                k = skew(axis)
                motion[:3, :3] = rotation(axis, q[j])
                dm[:3, :3] = k @ motion[:3, :3]
                ddm[:3, :3] = k @ k @ motion[:3, :3]
            elif kind == "prismatic":
                motion[:3, 3], dm[:3, 3] = axis*q[j], axis
            else:
                raise ValueError("unsupported oracle joint")
        local, dl, ddl = static @ motion, static @ dm, static @ ddm
        parent = int(model["parents"][body])
        p = np.eye(4) if parent < 0 else transforms[parent]
        dp = np.zeros((n, 4, 4)) if parent < 0 else derivatives[parent]
        pt = np.zeros((4, 4)) if parent < 0 else first[parent]
        ptt = np.zeros((4, 4)) if parent < 0 else second[parent]
        t = p @ local
        partial = dp @ local
        if body in column:
            partial[column[body]] += p @ dl
        td = pt @ local + speed * (p @ dl)
        tdd = ptt @ local + 2*speed*(pt @ dl) + speed*speed*(p @ ddl)
        transforms.append(t)
        derivatives.append(partial)
        first.append(td)
        second.append(tdd)
        point = np.r_[model["com"][body], 1.0]
        com_world[body] = (t @ point)[:3]
        jv[body] = (partial @ point)[:, :3].T
        r = t[:3, :3]
        for j in range(n):
            jw[body, :, j] = _vee(partial[j, :3, :3] @ r.T)
        omega[body] = _vee(td[:3, :3] @ r.T)
        alpha[body] = _vee(tdd[:3, :3] @ r.T + td[:3, :3] @ td[:3, :3].T)
        acc[body] = (tdd @ point)[:3]
    transforms = np.asarray(transforms)
    mass, bias = np.zeros((n, n)), np.zeros(n)
    for body in range(nb):
        r = transforms[body, :3, :3]
        inertia = r @ np.diag(model["inertia"][body]) @ r.T
        mass += model["mass"][body] * (jv[body].T @ jv[body]) + jw[body].T @ inertia @ jw[body]
        force = model["mass"][body] * (acc[body] - model["gravity"])
        moment = inertia @ alpha[body] + np.cross(omega[body], inertia @ omega[body])
        bias += jv[body].T @ force + jw[body].T @ moment
    return dict(position=transforms[:, :3, 3], rotation=transforms[:, :3, :3],
                com_position=com_world, linear_jacobian=jv, angular_jacobian=jw,
                mass=mass, bias=bias, com_velocity=np.einsum("bij,j->bi", jv, v),
                angular_velocity=omega, bias_com_acceleration=acc, bias_angular_acceleration=alpha)


def external(terms, effort, force, torque):
    return np.asarray(effort, dtype=float) + np.einsum("bij,bi->j", terms["linear_jacobian"], force) + np.einsum("bij,bi->j", terms["angular_jacobian"], torque)


def reference_step(model, q, v, effort, force, torque, h):
    terms = oracle(model, q, v)
    load = external(terms, effort, force, torque)
    acceleration = np.linalg.solve(terms["mass"], load - terms["bias"])
    next_v = np.asarray(v) + h * acceleration
    return np.asarray(q) + h*next_v, next_v, terms, load, acceleration


def rr_terms(q, v):
    c, s = cos(q[1]), sin(q[1])
    return (np.array([[5/3+c, 1/3+c/2], [1/3+c/2, 1/3]]),
            np.array([-s*(2*v[0]*v[1]+v[1]*v[1])/2, s*v[0]*v[0]/2]))


def branch_terms(q, v):
    _, s, t = q
    vr, vs, vt = v
    return (np.array([[2.5+s*s+t*t, 1, -1], [1, 1, 0], [-1, 0, 1]]),
            np.array([2*s*vr*vs+2*t*vr*vt, -s*vr*vr, -t*vr*vr]))


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def _numbers(values):
    return " ".join(format(float(v), ".17g") for v in values)


def urdf_text(model):
    root = ET.Element("robot", name="resident_tree_frozen_reference")
    ET.SubElement(root, "link", name="base")
    for i, name in enumerate(model["names"]):
        link = ET.SubElement(root, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=_numbers(model["com"][i]))
        ET.SubElement(inertial, "mass", value=format(float(model["mass"][i]), ".17g"))
        ix, iy, iz = model["inertia"][i]
        ET.SubElement(inertial, "inertia", ixx=str(ix), iyy=str(iy), izz=str(iz),
                      ixy="0", ixz="0", iyz="0")
        col = ET.SubElement(link, "collision", name="tool_box" if name == "tool" else "material_ball")
        ET.SubElement(col, "origin", xyz="0.25 -0.125 0", rpy="0 0 0")
        geometry = ET.SubElement(col, "geometry")
        if name == "tool":
            ET.SubElement(geometry, "box", size="0.5 0.25 0.125")
        else:
            ET.SubElement(geometry, "sphere", radius="0.0625")
        joint = ET.SubElement(root, "joint", name=model["joint_names"][i], type=model["kinds"][i])
        parent = model["parents"][i]
        ET.SubElement(joint, "parent", link="base" if parent < 0 else model["names"][parent])
        ET.SubElement(joint, "child", link=name)
        r = quaternion_matrix(model["orientations"][i])
        pitch = atan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0]))
        rpy = [atan2(r[2, 1], r[2, 2]), pitch, atan2(r[1, 0], r[0, 0])]
        ET.SubElement(joint, "origin", xyz=_numbers(model["offsets"][i]), rpy=_numbers(rpy))
        if model["kinds"][i] != "fixed":
            ET.SubElement(joint, "axis", xyz=_numbers(model["axes"][i]))
            ET.SubElement(joint, "limit", lower="-4096", upper="4096", effort="8", velocity="4")
    return ET.tostring(root, encoding="unicode")

ZERO_PARAMETERS = ("joint_armature", "joint_damping", "joint_friction", "joint_target_ke",
                   "joint_target_kd", "joint_limit_ke", "joint_limit_kd")
MODEL_ARRAYS = ("body_mass", "body_com", "body_inertia", "body_flags", "body_world",
                "joint_type", "joint_parent", "joint_child", "joint_q_start", "joint_qd_start",
                "joint_X_p", "joint_X_c", "joint_axis", "joint_dof_dim", "joint_limit_lower",
                "joint_limit_upper", "joint_effort_limit", "joint_velocity_limit", "joint_articulation",
                "articulation_start", "articulation_end", "gravity", "shape_type", "shape_body",
                "shape_transform", "shape_scale", "shape_margin", *ZERO_PARAMETERS)


def native_array(owner, name):
    value = getattr(owner, name, None)
    return None if value is None else value.numpy().copy()


def _gate(record, name, actual, expected, *, atol=ATOL, rtol=RTOL):
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    passed = a.shape == e.shape and bool(np.isfinite(a).all()) and bool(np.allclose(a, e, atol=atol, rtol=rtol))
    record.setdefault("checks", []).append(dict(name=name, passed=passed, atol=atol, rtol=rtol,
        max_abs_error=None if a.shape != e.shape else float(np.max(np.abs(a-e), initial=0))))
    return passed


def _condition(record, name, passed):
    record.setdefault("checks", []).append(dict(name=name, passed=bool(passed)))
    if not passed:
        raise ValueError(name)


def _build_native(newton, model_data_, path, device, record):
    path.write_text(urdf_text(model_data_), encoding="utf-8")
    builder = newton.ModelBuilder(gravity=tuple(model_data_["gravity"]), up_axis=newton.Axis.Z)
    builder.default_shape_cfg.density = 0.0
    builder.add_urdf(str(path), floating=False, collapse_fixed_joints=False,
                     collapse_massless_fixed_root=False, ignore_inertial_definitions=False,
                     enable_self_collisions=False, parse_visuals_as_colliders=False,
                     joint_damping_mode="passive", up_axis=newton.Axis.Z)
    record["parser_parameters_before_zero"] = {name: list(getattr(builder, name)) for name in ZERO_PARAMETERS}
    for name in ZERO_PARAMETERS:
        values = getattr(builder, name)
        for j in range(len(values)):
            values[j] = 0.0
    native = builder.finalize(device=device)
    meta = {name: native_array(native, name) for name in MODEL_ARRAYS}
    meta.update(body_label=list(native.body_label), joint_label=list(native.joint_label),
                shape_label=list(native.shape_label), device=str(native.device),
                body_count=native.body_count, joint_count=native.joint_count,
                joint_coord_count=native.joint_coord_count, joint_dof_count=native.joint_dof_count,
                articulation_count=native.articulation_count)
    record["native_model"] = meta
    record["urdf"] = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           text=path.read_text(encoding="utf-8"))
    body_names = [str(x).rsplit("/", 1)[-1] for x in native.body_label]
    joint_names = [str(x).rsplit("/", 1)[-1] for x in native.joint_label]
    _condition(record, "requested_real_device", str(native.device) == device)
    _condition(record, "unique_native_labels", len(set(body_names)) == len(body_names) and len(set(joint_names)) == len(joint_names))
    bodies, joints = dict(zip(body_names, range(len(body_names)))), dict(zip(joint_names, range(len(joint_names))))
    _condition(record, "all_authored_bodies_retained", set(bodies) == {"base", *model_data_["names"]})
    active = coordinates(model_data_)
    _condition(record, "ordinary_scalar_dimensions", native.joint_coord_count == native.joint_dof_count == len(active))
    _condition(record, "one_articulation", native.articulation_count == 1)
    types = meta["joint_type"]
    _condition(record, "no_free_root", not np.any(types == int(newton.JointType.FREE)))
    qcols, vcols, body_rows = [], [], []
    for i, name in enumerate(model_data_["names"]):
        row, joint = bodies[name], joints[model_data_["joint_names"][i]]
        body_rows.append(row)
        kind = model_data_["kinds"][i]
        expected_type = {"fixed": newton.JointType.FIXED, "revolute": newton.JointType.REVOLUTE,
                         "prismatic": newton.JointType.PRISMATIC}[kind]
        _condition(record, f"{name}_joint_type", int(types[joint]) == int(expected_type))
        parent = int(model_data_["parents"][i])
        _condition(record, f"{name}_ownership", int(meta["joint_child"][joint]) == row and
                   int(meta["joint_parent"][joint]) == bodies["base" if parent < 0 else model_data_["names"][parent]])
        _gate(record, f"{name}_mass", meta["body_mass"][row], model_data_["mass"][i])
        _gate(record, f"{name}_com", meta["body_com"][row], model_data_["com"][i])
        _gate(record, f"{name}_inertia", meta["body_inertia"][row], np.diag(model_data_["inertia"][i]))
        if kind != "fixed":
            qcols.append(int(meta["joint_q_start"][joint]))
            vcols.append(int(meta["joint_qd_start"][joint]))
    return native, body_rows, qcols, vcols


def _state_record(native, state, rows, qcols, vcols):
    raw = {name: native_array(state, name) for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f")}
    raw["q"] = raw["joint_q"][qcols]
    raw["v"] = raw["joint_qd"][vcols]
    poses, twists = raw["body_q"][rows], raw["body_qd"][rows]
    rotation_ = np.asarray([quaternion_matrix(p[3:]) for p in poses])
    raw["authored_position_m"] = poses[:, :3]
    raw["authored_rotation"] = rotation_
    raw["com_world_m"] = poses[:, :3] + np.einsum("bij,bj->bi", rotation_, native.body_com.numpy()[rows])
    raw["com_velocity_world_m_s"] = twists[:, :3]
    raw["angular_velocity_world_rad_s"] = twists[:, 3:]
    return raw


def _seed(newton, native, q, v, qcols, vcols):
    state = native.state()
    jq, jv = native_array(state, "joint_q"), native_array(state, "joint_qd")
    jq[qcols], jv[vcols] = q, v
    state.joint_q.assign(jq)
    state.joint_qd.assign(jv)
    newton.eval_fk(native, state.joint_q, state.joint_qd, state)
    return state


def run_case(newton, name, device, directory, *, cycled=False):
    authored, q0, v0, effort, force, torque = case_input(name)
    if cycled:
        authored, transform = transformed(authored, cycled=True, translation=(0.5, -0.25, 0.75))
        force, torque = force @ transform.T, torque @ transform.T
    record = dict(case=name, device=device, cycled=cycled, input=dict(model=authored, q=q0, v=v0,
        effort=effort, force_world_com_n=force, torque_world_com_nm=torque, h_s=H, steps=40),
        conventions=dict(body_qd="linear COM/world, then angular/world",
            body_f="force/world at COM, then torque/world at COM", contacts="None",
            controls="constant authored external generalized effort; native PD disabled",
            integrator="Featherstone symplectic Euler; update_mass_matrix_interval=1",
            fixed_tools="retained native fixed joints; no collapse", geometry="current FK only, not CCD"),
        trace=[], static_poses=[], execution_failure=None)
    output = directory / f"resident-tree-{name}-{'cycled' if cycled else 'base'}-{device.replace(':', '-')}.json"
    def save():
        output.write_text(json.dumps(json_value(record), indent=2, allow_nan=False), encoding="utf-8")
    try:
        native, rows, qcols, vcols = _build_native(newton, authored, directory / f"{name}-{'cycled' if cycled else 'base'}.urdf", device, record)
        poses = [q0.copy()]
        if len(q0) == 2:
            poses += [np.array([0.37, -0.51]), np.array([-0.51, 0.37])]
        elif name == "T02":
            poses += [np.array([0.37, -1/16, 3/16]), np.array([-0.51, 3/16, -1/16])]
        for q in poses:
            state = _seed(newton, native, q, v0, qcols, vcols)
            actual = _state_record(native, state, rows, qcols, vcols)
            truth = oracle(authored, actual["q"], actual["v"])
            item = dict(actual=actual, oracle=truth)
            record["static_poses"].append(item)
            for observed, expected in (("authored_position_m", "position"), ("authored_rotation", "rotation"),
                                       ("com_world_m", "com_position"), ("com_velocity_world_m_s", "com_velocity"),
                                       ("angular_velocity_world_rad_s", "angular_velocity")):
                _gate(item, observed, actual[observed], truth[expected])
            save()
        state = _seed(newton, native, q0, v0, qcols, vcols)
        next_state = native.state()
        control = native.control()
        joint_force = np.zeros(native.joint_dof_count, dtype=np.float32)
        joint_force[vcols] = effort
        control.joint_f.assign(joint_force)
        solver = newton.solvers.SolverFeatherstone(native, angular_damping=0.0, update_mass_matrix_interval=1)
        reference_q, reference_v = np.asarray(q0, np.float32).astype(float), np.asarray(v0, np.float32).astype(float)
        for step in range(1, 41):
            state.clear_forces()
            applied = np.zeros((native.body_count, 6), dtype=np.float32)
            applied[rows, :3], applied[rows, 3:] = force, torque
            state.body_f.assign(applied)
            entry = _state_record(native, state, rows, qcols, vcols)
            terms = oracle(authored, entry["q"], entry["v"])
            load = external(terms, effort, force, torque)
            acceleration = np.linalg.solve(terms["mass"], load-terms["bias"])
            reference_q, reference_v, _, _, _ = reference_step(authored, reference_q, reference_v, effort, force, torque, H)
            solver.step(state, next_state, control, None, H)
            actual = _state_record(native, next_state, rows, qcols, vcols)
            item = dict(step=step, entry_time_s=(step-1)*H, commit_time_s=step*H,
                        entry=entry, actual=actual, applied_body_f_world_com=applied,
                        control_joint_f=joint_force, oracle_entry=terms, oracle_load=load,
                        oracle_acceleration=acceleration, oracle_window_q=reference_q.copy(),
                        oracle_window_v=reference_v.copy(), native_private_H=native_array(solver, "H"),
                        native_private_qdd=native_array(solver, "joint_qdd"))
            record["trace"].append(item)
            save()
            _gate(item, "velocity_same_entry", actual["v"], entry["v"] + H*acceleration)
            _gate(item, "semi_implicit_q", actual["q"], entry["q"] + H*actual["v"])
            _gate(item, "independent_window_q", actual["q"], reference_q, atol=WINDOW_ATOL)
            _gate(item, "independent_window_v", actual["v"], reference_v, atol=WINDOW_ATOL)
            final = oracle(authored, actual["q"], actual["v"])
            _gate(item, "actual_end_fk", actual["authored_position_m"], final["position"])
            _gate(item, "actual_end_com_velocity", actual["com_velocity_world_m_s"], final["com_velocity"])
            _gate(item, "actual_end_omega", actual["angular_velocity_world_rad_s"], final["angular_velocity"])
            item["actual_kinetic_energy_j"] = sum(
                0.5*authored["mass"][i]*np.dot(actual["com_velocity_world_m_s"][i], actual["com_velocity_world_m_s"][i]) +
                0.5*np.dot(actual["angular_velocity_world_rad_s"][i], actual["authored_rotation"][i] @ np.diag(authored["inertia"][i]) @ actual["authored_rotation"][i].T @ actual["angular_velocity_world_rad_s"][i])
                for i in range(len(rows)))
            save()
            state, next_state = next_state, state
    except Exception as error:
        record["execution_failure"] = dict(type=type(error).__name__, message=str(error))
    record["passed"] = record["execution_failure"] is None and len(record["trace"]) == 40 and all(
        check["passed"] for part in [record, *record["static_poses"], *record["trace"]]
        for check in part.get("checks", ()))
    save()
    return dict(case=name, path=str(output.resolve()), passed=record["passed"],
                steps=len(record["trace"]), execution_failure=record["execution_failure"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=("T01", "T02", "T03", "T04", "N9", "N16"))
    parser.add_argument("--cycled", action="store_true")
    args = parser.parse_args()
    import newton
    import warp as wp
    args.output_dir.mkdir(parents=True, exist_ok=True)
    versions = {name: importlib.metadata.version(name) for name in ("newton", "warp-lang", "numpy")}
    if versions["newton"] != "1.7.0" or versions["warp-lang"] != "1.17.0":
        raise ValueError(f"pinned Newton/Warp versions required: {versions}")
    if args.device == "cuda:0" and not wp.is_cuda_available():
        raise RuntimeError("requested CUDA device is unavailable; no CPU fallback")
    source_root = Path(newton.__file__).resolve().parent
    source_names = ("_src/sim/state.py", "_src/sim/articulation.py", "_src/sim/control.py",
                    "_src/solvers/featherstone/solver_featherstone.py", "_src/solvers/featherstone/kernels.py",
                    "_src/utils/import_urdf.py")
    sources = {}
    for name in source_names:
        source = source_root / name
        snapshot = args.output_dir / "source_snapshot" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_bytes()
        snapshot.write_bytes(content)
        sources[name] = dict(path=str(source), snapshot_path=str(snapshot.resolve()),
                             sha256=hashlib.sha256(content).hexdigest())
    driver = Path(__file__).resolve()
    driver_copy = args.output_dir / "source_snapshot" / driver.name
    driver_copy.write_bytes(driver.read_bytes())
    sources["driver"] = dict(path=str(driver), snapshot_path=str(driver_copy.resolve()),
                              sha256=hashlib.sha256(driver.read_bytes()).hexdigest())
    report = dict(versions=versions, sources=sources, requested_device=args.device,
                  cycled=args.cycled, results=[])
    for name in args.case or ("T01", "T02", "T03", "T04", "N9", "N16"):
        result = run_case(newton, name, args.device, args.output_dir, cycled=args.cycled)
        report["results"].append(result)
        (args.output_dir / "summary.json").write_text(json.dumps(json_value(report), indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(result), flush=True)
    return 0 if all(row["passed"] for row in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
