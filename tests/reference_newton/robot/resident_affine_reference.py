"""C5 frozen affine mechanisms and independent Newton 1.7 reference.

Authoring only. No Phymeo import: full mechanics come from derivatives of
homogeneous transforms and an independent full-coordinate KKT solve.
"""

import argparse
import copy
import gzip
import hashlib
import importlib.metadata
import json
import xml.etree.ElementTree as ET
from math import atan2, cos, pi, sin
from pathlib import Path

import numpy as np

H = 1 / 1024
STEPS = 40
ATOL = 5e-6
RTOL = 5e-5
WINDOW_ATOL = 2e-4
REACTION_ATOL = 3e-4

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


def coordinates(model):
    return tuple(i for i, kind in enumerate(model["kinds"]) if kind != "fixed")


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


def native_array(owner, name):
    value = getattr(owner, name, None)
    return None if value is None else value.numpy().copy()


def _gate(record, name, actual, expected, *, atol=ATOL, rtol=RTOL):
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    passed = a.shape == e.shape and bool(np.isfinite(a).all()) and bool(np.allclose(a, e, atol=atol, rtol=rtol))
    record.setdefault("checks", []).append(dict(name=name, passed=passed, atol=atol, rtol=rtol,
        max_abs_error=None if a.shape != e.shape else float(np.max(np.abs(a-e), initial=0))))
    return passed


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


def case(name):
    """Authored physical data; P/c are explicit independent test inputs."""
    if name in ("A01", "A01F", "A02", "A03", "A03_flat") or name.startswith("N"):
        n = int(name[1:]) if name.startswith("N") else 2 if name.startswith("A01") else 3
        k = 3 if n == 9 else 4 if n in (16, 17) else 1
        masses = [2, 1] if n == 2 else [2, 1, 4] if name == "A02" else [2, 1, 1] if n == 3 else [1+i/8 for i in range(n)]
        offsets = [[0, 0, 0], [0, .5, 0]] if n == 2 else [[0, 0, 0], [0, .5, 0], [0, -.5, 0]] if n == 3 else [[0, i/8, 0] for i in range(n)]
        m = _model([-1]*n, ["prismatic"]*n, offsets, [[1, 0, 0]]*n,
                   masses, np.zeros((n, 3)), [[1/8]*3]*n)
        P, c = np.zeros((n, k)), np.zeros(n)
        P[:k] = np.eye(k)
        if n == 2:
            P[1, 0], c[1] = 2, 1/8
            rows = ((0, 1, 2, 1/8),)
        elif n == 3:
            P[:, 0] = [1, 2, -.5] if name == "A02" else [1, 2, -1]
            c[:] = [0, 1/8, -1/16] if name == "A02" else [0, 1/8, 1/8]
            rows = ((0, 1, 2, 1/8), (0, 2, -.5, -1/16)) if name == "A02" else ((0, 1, 2, 1/8), (1, 2, -.5, 3/16)) if name == "A03" else ((0, 1, 2, 1/8), (0, 2, -1, 1/8))
        else:
            rows = tuple((i % k, i, (-1)**i*(1+(i % 3)/2), i/64) for i in range(k, n))
            for driver, follower, a, b in rows:
                P[follower, driver], c[follower] = a, b
        independent = tuple(range(k))
        s = np.array([(j+1)/64 for j in range(k)]) if n > 3 else np.zeros(k)
        u = np.array([(j-1)/32 for j in range(k)]) if n > 3 else np.zeros(k)
        motor = np.array([1+j/8 for j in range(k)]) if n > 3 else np.array([0 if name == "A01F" else 3 if n == 2 else 7.0])
    elif name in ("A04", "A05T"):
        m = _model([-1, 0], ["revolute"]*2, [[0, 0, 0], [1, 0, 0]],
                   [[0, 0, 1]]*2, [1, 1], [[.5, 0, 0]]*2,
                   [[1/24, 1/24, 1/12]]*2)
        if name == "A05T":
            m["parents"] = np.array([-1, 0, 1])
            m["kinds"] = ("revolute", "revolute", "fixed")
            for field, extra in (("offsets", [.75, 0, 0]), ("orientations", [0, 0, sin(pi/4), cos(pi/4)]),
                                 ("axes", [0, 0, 1]), ("com", [.25, 0, 0]), ("inertia", [1/64, 1/32, 5/128])):
                m[field] = np.vstack((m[field], extra))
            m["mass"] = np.array([1, 1, .5])
            m["names"], m["joint_names"] = ("link0", "link1", "tool"), ("joint0", "joint1", "tool_fixed")
        n, k, independent = 2, 1, (0,)
        offset = pi/2 if name == "A04" else 0.0
        rows, P, c = ((0, 1, -2, offset),), np.array([[1.0], [-2.0]]), np.array([0.0, offset])
        s, u = np.zeros(1), np.array([.5 if name == "A04" else 0])
        motor = np.array([1 if name == "A04" else 295/384])
    elif name in ("A05", "A05G"):
        m = _model([-1, 0, 0], ["revolute", "prismatic", "prismatic"],
                   [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                   [2, 1, 1], np.zeros((3, 3)), [[1/8, 1/8, 1/4], [1/16, 1/16, 1/8], [1/16, 1/16, 1/8]])
        n, k, independent = 3, 2, (0, 1)
        rows, P, c = ((1, 2, 2, 1/8),), np.array([[1., 0], [0, 1], [0, 2]]), np.array([0., 0., 1/8])
        s, u, motor = np.zeros(2), np.zeros(2), np.zeros(2)
        if name == "A05G":
            m["gravity"] = np.array([0., -2., 0.])
    else:
        raise ValueError(name)
    C = np.zeros((len(rows), n))
    for i, (driver, follower, a, _) in enumerate(rows):
        C[i, driver], C[i, follower] = -a, 1
    out = dict(name=name, model=m, P=P, c=c, C=C, rows=rows, independent=independent,
               s=s, u=u, motor=motor, effort=np.zeros(n), force=np.zeros((len(m["mass"]), 3)),
               torque=np.zeros((len(m["mass"]), 3)), lower=np.full(n, -8.), upper=np.full(n, 8.),
               speed=np.full(n, 16.), rating=np.full(n, 16.))
    if name == "A01F":
        out["force"][1, 0] = 3
    if name.startswith("A05") and name != "A05T":
        out["effort"][1] = 1
    if name == "A05G":
        out["force"][2, 0] = 3
    return out


def transformed(data, placement):
    data = copy.deepcopy(data)
    if placement == "base":
        return data
    Q = np.array([[0., 0, 1], [1, 0, 0], [0, 1, 0]])
    translation = np.array([4096., -2048, 1024]) if placement == "large" else np.array([.5, -.25, .75])
    m = data["model"]
    for i, p in enumerate(m["parents"]):
        if p < 0:
            m["offsets"][i] = Q @ m["offsets"][i] + translation
            m["orientations"][i] = [.5, .5, .5, .5]
    m["gravity"] = Q @ m["gravity"]
    data["force"], data["torque"] = data["force"] @ Q.T, data["torque"] @ Q.T
    return data


def terms(data, s, u):
    P, c, m = data["P"], data["c"], data["model"]
    q, v = P @ s + c, P @ u
    full = oracle(m, q, v)
    e = external(full, data["effort"], data["force"], data["torque"])
    return dict(full=full, q=q, v=v, mass=P.T @ full["mass"] @ P,
                bias=P.T @ full["bias"], external=P.T @ e, external_full=e)


def kkt(data, s, u, motor=None):
    t = terms(data, s, u)
    motor = data["motor"] if motor is None else np.asarray(motor, float)
    n, C = len(data["c"]), data["C"]
    drive = np.zeros(n)
    drive[list(data["independent"])] = motor
    M, b = t["full"]["mass"], t["full"]["bias"]
    A = np.block([[M, -C.T], [C, np.zeros((len(C), len(C)))]])
    solution = np.linalg.solve(A, np.r_[drive + t["external_full"] - b, np.zeros(len(C))])
    acceleration, multipliers = solution[:n], solution[n:]
    reduced = acceleration[list(data["independent"])]
    return dict(**t, acceleration=reduced, full_acceleration=acceleration,
                motor_full=drive, multipliers=multipliers, reaction=C.T @ multipliers)


def advance(data, s, u, motor=None, h=H):
    t = kkt(data, s, u, motor)
    next_u = np.asarray(u) + h*t["acceleration"]
    return np.asarray(s) + h*next_u, next_u, t


def urdf(data, *, reorder=False, flatten=False):
    m = data["model"]
    active = coordinates(m)
    index = {body: j for j, body in enumerate(active)}
    rows = data["rows"]
    if flatten:
        rows = tuple((data["independent"][int(np.flatnonzero(data["P"][i])[0])], i,
                      float(data["P"][i, np.flatnonzero(data["P"][i])[0]]), float(data["c"][i]))
                     for i in range(len(data["c"])) if i not in data["independent"])
    followers = {f: (d, a, b) for d, f, a, b in rows}
    root = ET.Element("robot", name="resident_affine_frozen")
    ET.SubElement(root, "link", name="base")
    for i, name in enumerate(m["names"]):
        link = ET.SubElement(root, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=_numbers(m["com"][i]))
        ET.SubElement(inertial, "mass", value=str(m["mass"][i]))
        x, y, z = m["inertia"][i]
        ET.SubElement(inertial, "inertia", ixx=str(x), iyy=str(y), izz=str(z), ixy="0", ixz="0", iyz="0")
        col = ET.SubElement(link, "collision", name="material")
        ET.SubElement(col, "origin", xyz="0.25 -0.125 0", rpy="0 0 0.7853981633974483")
        geometry = ET.SubElement(col, "geometry")
        if name == "tool":
            ET.SubElement(geometry, "box", size="0.5 0.25 0.125")
        else:
            ET.SubElement(geometry, "sphere", radius="0.0625")
        joint = ET.SubElement(root, "joint", name=m["joint_names"][i], type=m["kinds"][i])
        parent = int(m["parents"][i])
        ET.SubElement(joint, "parent", link="base" if parent < 0 else m["names"][parent])
        ET.SubElement(joint, "child", link=name)
        R = quaternion_matrix(m["orientations"][i])
        rpy = [atan2(R[2, 1], R[2, 2]), atan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0])), atan2(R[1, 0], R[0, 0])]
        ET.SubElement(joint, "origin", xyz=_numbers(m["offsets"][i]), rpy=_numbers(rpy))
        if i in index:
            j = index[i]
            ET.SubElement(joint, "axis", xyz=_numbers(m["axes"][i]))
            ET.SubElement(joint, "limit", lower=str(data["lower"][j]), upper=str(data["upper"][j]),
                          velocity=str(data["speed"][j]), effort=str(data["rating"][j]))
            if j in followers:
                d, a, b = followers[j]
                ET.SubElement(joint, "mimic", joint=m["joint_names"][active[d]], multiplier=str(a), offset=str(b))
    if reorder:
        root[:] = list(reversed(list(root)))
    return ET.tostring(root, encoding="unicode")


ZERO = ("joint_armature", "joint_damping", "joint_friction", "joint_target_ke",
        "joint_target_kd", "joint_limit_ke", "joint_limit_kd")
META = ("body_mass", "body_com", "body_inertia", "joint_type", "joint_parent", "joint_child",
        "joint_q_start", "joint_qd_start", "joint_X_p", "joint_X_c", "joint_axis", "joint_dof_dim",
        "joint_articulation", "articulation_start", "articulation_end", "joint_mimic_joint",
        "joint_mimic_coeffs", "joint_limit_lower", "joint_limit_upper", "joint_effort_limit",
        "joint_velocity_limit", "shape_type", "shape_body", "shape_transform", "shape_scale", "gravity", *ZERO)


def native_builder(newton, data, path, *, flatten=False):
    path.write_text(urdf(data, flatten=flatten), encoding="utf-8")
    b = newton.ModelBuilder(gravity=tuple(data["model"]["gravity"]), up_axis=newton.Axis.Z)
    b.default_shape_cfg.density = 0.0
    b.add_urdf(str(path), floating=False, collapse_fixed_joints=False,
               collapse_massless_fixed_root=False, ignore_inertial_definitions=False,
               enable_self_collisions=False, parse_visuals_as_colliders=False,
               joint_damping_mode="passive", up_axis=newton.Axis.Z)
    return b


def native_case(newton, name, device, directory, *, flatten=False):
    data = case(name)
    label = name + ("-flattened" if flatten else "")
    report = dict(case=label, requested_device=device, input=data, h_s=H, steps=STEPS,
                  conventions=dict(body_qd="world COM linear first, world angular last",
                    body_f="world force at COM first, world torque at COM last",
                    motor="independent joint_f; full nonmotor effort also uses joint_f",
                    joints="fixed joints preserved, mimic metadata asserted",
                    contacts=None, integrator="Featherstone semiimplicit; recompute mass every step",
                    chain="original chain rejected separately; flattened graph explicitly labeled"),
                  checks=[], trace=[], execution_failure=None)
    output = directory / (label + ".json.gz")
    def save():
        with gzip.open(output, "wt", encoding="utf-8") as stream:
            json.dump(json_value(report), stream, indent=2, allow_nan=False)
    try:
        path = directory / (label + ".urdf")
        builder = native_builder(newton, data, path, flatten=flatten)
        report["urdf"] = dict(text=path.read_text(encoding="utf-8"), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report["parameters_before_zero"] = {key: list(getattr(builder, key)) for key in ZERO}
        for key in ZERO:
            for i in range(len(getattr(builder, key))):
                getattr(builder, key)[i] = 0.0
        model = builder.finalize(device=device)
        meta = {key: native_array(model, key) for key in META}
        meta.update(body_label=list(model.body_label), joint_label=list(model.joint_label), shape_label=list(model.shape_label), device=str(model.device))
        report["native_model"] = meta
        assert str(model.device) == device
        bodies = [str(x).rsplit("/", 1)[-1] for x in model.body_label]
        joints = [str(x).rsplit("/", 1)[-1] for x in model.joint_label]
        rows = [bodies.index(x) for x in data["model"]["names"]]
        active = coordinates(data["model"])
        joint_rows = [joints.index(data["model"]["joint_names"][i]) for i in active]
        qcols = [int(meta["joint_q_start"][j]) for j in joint_rows]
        vcols = [int(meta["joint_qd_start"][j]) for j in joint_rows]
        for i, row in enumerate(rows):
            _gate(report, f"body{i}_mass", meta["body_mass"][row], data["model"]["mass"][i])
            _gate(report, f"body{i}_com", meta["body_com"][row], data["model"]["com"][i])
            _gate(report, f"body{i}_inertia", meta["body_inertia"][row], np.diag(data["model"]["inertia"][i]))
        for f in range(len(data["c"])):
            j = joint_rows[f]
            expected_reference = -1
            expected_coeff = [0, 1]
            if f not in data["independent"]:
                if flatten:
                    col = int(np.flatnonzero(data["P"][f])[0])
                    d, a, b = data["independent"][col], data["P"][f, col], data["c"][f]
                else:
                    d, _, a, b = next(row for row in data["rows"] if row[1] == f)
                expected_reference, expected_coeff = joint_rows[d], [b, a]
            assert int(meta["joint_mimic_joint"][j]) == expected_reference
            _gate(report, f"joint{f}_mimic_coefficients", meta["joint_mimic_coeffs"][j], expected_coeff)
        q0, v0 = data["P"] @ data["s"] + data["c"], data["P"] @ data["u"]
        state = _seed(newton, model, q0, v0, qcols, vcols)
        after, control = model.state(), model.control()
        applied_joint = np.zeros(model.joint_dof_count, np.float32)
        motor_full = np.zeros(len(q0))
        motor_full[list(data["independent"])] = data["motor"]
        applied_joint[vcols] = data["effort"] + motor_full
        control.joint_f.assign(applied_joint)
        solver = newton.solvers.SolverFeatherstone(model, angular_damping=0., update_mass_matrix_interval=1)
        s, u = data["s"].copy(), data["u"].copy()
        for step in range(1, STEPS+1):
            state.clear_forces()
            body_f = np.zeros((model.body_count, 6), np.float32)
            body_f[rows, :3], body_f[rows, 3:] = data["force"], data["torque"]
            state.body_f.assign(body_f)
            before = _state_record(model, state, rows, qcols, vcols)
            entry_s, entry_u = before["q"][list(data["independent"])], before["v"][list(data["independent"])]
            expected = kkt(data, entry_s, entry_u)
            assert model.articulation_count == 1
            art_joint = int(meta["articulation_start"][0])
            mass_columns = np.asarray(vcols)-int(meta["joint_qd_start"][art_joint])
            native_mass_all = newton.eval_mass_matrix(model, state).numpy().copy()
            native_mass = native_mass_all[0][np.ix_(mass_columns, mass_columns)]
            s, u, _ = advance(data, s, u)
            solver.step(state, after, control, None, H)
            actual = _state_record(model, after, rows, qcols, vcols)
            observed_acceleration = (actual["v"]-before["v"])/H
            balance = expected["full"]["mass"] @ observed_acceleration + expected["full"]["bias"] - motor_full - expected["external_full"]
            item = dict(step=step, entry_time_s=(step-1)*H, commit_time_s=step*H,
                        entry=before, actual=actual, applied_body_f=body_f, applied_joint_f=applied_joint,
                        native_full_mass=native_mass, native_raw_mass=native_mass_all,
                        oracle=expected, oracle_window_s=s.copy(), oracle_window_u=u.copy(),
                        derived_full_reaction=balance, derived_reaction_reference="KKT at actual input; not native sensor",
                        native_H=native_array(solver, "H"), native_H_mimic=native_array(solver, "H_mimic"))
            report["trace"].append(item)
            save()
            _gate(item, "full_native_mass", native_mass, expected["full"]["mass"])
            _gate(item, "same_entry_velocity", actual["v"], before["v"]+H*expected["full_acceleration"])
            _gate(item, "same_entry_position", actual["q"], before["q"]+H*actual["v"])
            _gate(item, "full_affine_position", actual["q"], data["P"] @ actual["q"][list(data["independent"])] + data["c"])
            _gate(item, "full_affine_velocity", actual["v"], data["P"] @ actual["v"][list(data["independent"])])
            _gate(item, "window_position", actual["q"], data["P"] @ s+data["c"], atol=WINDOW_ATOL)
            _gate(item, "window_velocity", actual["v"], data["P"] @ u, atol=WINDOW_ATOL)
            _gate(item, "derived_reaction", balance, expected["reaction"], atol=REACTION_ATOL)
            geom = oracle(data["model"], actual["q"], actual["v"])
            for observed, key in (("authored_position_m", "position"), ("authored_rotation", "rotation"),
                                  ("com_world_m", "com_position"), ("com_velocity_world_m_s", "com_velocity"),
                                  ("angular_velocity_world_rad_s", "angular_velocity")):
                _gate(item, observed, actual[observed], geom[key])
            item["kinetic_energy_j"] = float(.5*actual["v"] @ geom["mass"] @ actual["v"])
            save()
            state, after = after, state
    except Exception as error:
        report["execution_failure"] = dict(type=type(error).__name__, message=str(error))
    report["passed"] = report["execution_failure"] is None and len(report["trace"]) == STEPS and all(
        check["passed"] for record in (report, *report["trace"]) for check in record.get("checks", ()))
    save()
    return dict(case=label, passed=report["passed"], steps=len(report["trace"]), execution_failure=report["execution_failure"], path=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import newton
    import warp as wp
    versions = {key: importlib.metadata.version(key) for key in ("newton", "warp-lang", "numpy")}
    if versions["newton"] != "1.7.0" or versions["warp-lang"] != "1.17.0":
        raise ValueError(f"pinned versions required: {versions}")
    if args.device == "cuda:0" and not wp.is_cuda_available():
        raise RuntimeError("CUDA unavailable, no fallback")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_root, sources = Path(newton.__file__).resolve().parent, {}
    for relative in ("_src/sim/state.py", "_src/sim/articulation.py", "_src/sim/joint_mimic.py", "_src/sim/builder.py",
                     "_src/utils/import_urdf.py", "_src/solvers/featherstone/kernels.py", "_src/solvers/featherstone/solver_featherstone.py"):
        source = source_root / relative
        target = args.output_dir / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        sources[relative] = dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    script = Path(__file__).resolve()
    (args.output_dir / "source_snapshot" / script.name).write_bytes(script.read_bytes())
    sources["driver"] = dict(path=str(script), sha256=hashlib.sha256(script.read_bytes()).hexdigest())
    try:
        native_builder(newton, case("A03"), args.output_dir / "original-chain.urdf")
    except ValueError as error:
        chain = dict(accepted=False, error=str(error), expected_rejection="mimic" in str(error).lower())
    else:
        chain = dict(accepted=True, expected_rejection=False)
    original_chain_path = args.output_dir / "original-chain.urdf"
    chain["source"] = dict(path=str(original_chain_path), text=original_chain_path.read_text(encoding="utf-8"),
                           sha256=hashlib.sha256(original_chain_path.read_bytes()).hexdigest(), input=case("A03"))
    report = dict(versions=versions, device=str(wp.get_device(args.device)), sources=sources, original_chain=chain, results=[])
    for name in ("A01", "A02", "A04", "A05G", "A03"):
        result = native_case(newton, name, args.device, args.output_dir, flatten=name == "A03")
        report["results"].append(result)
        (args.output_dir / "summary.json").write_text(json.dumps(json_value(report), indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(result), flush=True)
    return 0 if chain["expected_rejection"] and all(row["passed"] for row in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())