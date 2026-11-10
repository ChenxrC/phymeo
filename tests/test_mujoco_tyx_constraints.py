"""Independent native mixed constraints, physical wrenches and continuous paths.

Preregistered in mujoco-tyx-t4-cases.md. Authoring does not execute this module.
"""

import itertools

import numpy as np
import pytest
import test_mujoco_tyx_normal_contacts as normal
from test_mujoco_tyx_floating import _save

from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState

MODES = normal.MODES
FORCE_TOL = normal.FORCE_TOL
ROW_TOL = dict(rtol=5e-4, atol=4e-4)
FLOOR = '<geom name="floor" type="plane" size="2 2 .1"/>'
PLATFORM = '<geom name="floor" type="box" size=".24 .18 .05"/>'
XYZ = ("1 0 0", "0 1 0", "0 0 1")
BASE_FRICTION = ".6 .02 .003"


def _xml(
    body,
    *,
    dim=3,
    h=0.001,
    static=FLOOR,
    gravity="0 0 -9.81",
    friction=BASE_FRICTION,
    option="",
    extra="",
    joint="",
):
    return f'''<mujoco model="t4-mixed"><compiler angle="radian" autolimits="true" fusestatic="false"/>
      <option timestep="{h}" gravity="{gravity}" integrator="Euler" solver="Newton"
        iterations="128" tolerance="1e-10" jacobian="dense" cone="pyramidal" {option}/>
      <default><geom condim="{dim}" friction="{friction}" solref=".02 1" solimp=".8 .9 .002 .5 2"/>
        <joint armature=".01" damping=".04" {joint}/></default>
      <worldbody>{static}{body}</worldbody>{extra}</mujoco>'''


def _inertial(mass=1.0, offset=False):
    pos = ".03 -.02 .01" if offset else "0 0 0"
    values = [mass * x for x in (0.04, 0.05, 0.06, 0.001, -0.0005, 0.0015)]
    return f'<inertial pos="{pos}" mass="{mass}" fullinertia="{" ".join(map(str, values))}"/>'


def slider_xml(
    *,
    dim=3,
    mass=1.0,
    h=0.001,
    shape="sphere",
    finite=False,
    friction=BASE_FRICTION,
    dry=0.0,
    limits=False,
    extra="",
    option="",
):
    geom = {
        "sphere": 'type="sphere" size=".1"',
        "capsule": 'type="capsule" size=".1 .12" quat=".7071067811865476 0 .7071067811865476 0"',
        "box": 'type="box" size=".12 .08 .06"',
    }[shape]
    height = (0.099 if shape != "box" else 0.059) + (0.05 if finite else 0)
    joints = "".join(
        f'<joint name="j{i}" type="slide" axis="{axis}" armature="0" damping="0" '
        f'frictionloss="{dry}"'
        + (' limited="true" range="-.2 .2" margin=".01"' if limits and i == 0 else "")
        + "/>"
        for i, axis in enumerate(XYZ)
    )
    body = f'<body name="root" pos="0 0 {height}">{joints}{_inertial(mass)}<geom name="moving" {geom}/></body>'
    return _xml(
        body,
        dim=dim,
        h=h,
        static=PLATFORM if finite else FLOOR,
        friction=friction,
        extra=extra,
        option=option,
    )


def free_xml(*, dim=6, mass=1.0, h=0.001, geometry="sphere", oblique=False, two=False, offset=True):
    if two:
        shapes = '<geom name="a" type="sphere" size=".1" pos="-.14 0 0"/><geom name="b" type="sphere" size=".1" pos=".14 0 0"/>'
    elif geometry == "capsule":
        shapes = '<geom name="moving" type="capsule" size=".1 .12" quat=".7071067811865476 0 .7071067811865476 0"/>'
    elif geometry == "box":
        shapes = '<geom name="moving" type="box" size=".12 .08 .06"/>'
    else:
        shapes = '<geom name="moving" type="sphere" size=".1"/>'
    height = 0.059 if geometry == "box" else 0.099
    quat = ".96 .18 -.12 .1" if oblique else "1 0 0 0"
    body = f'<body name="root" pos="0 0 {height}" quat="{quat}"><freejoint/>{_inertial(mass, offset)}{shapes}</body>'
    return _xml(body, dim=dim, h=h)


def joint_xml(kind="hinge", *, dry=0.0, damping=0.04, overlap=False, root_free=False):
    interval = "-.005 .005" if overlap else "-.2 .2"
    outer = '<body name="root"><freejoint/>' + _inertial(2) if root_free else ""
    body = (
        outer + f'<body name="link" pos=".1 0 .3"><joint name="j" type="{kind}" axis="0 1 0" '
        f'limited="true" range="{interval}" margin=".01" frictionloss="{dry}" damping="{damping}"/>'
        + _inertial()
        + "</body>"
        + ("</body>" if root_free else "")
    )
    return _xml(body, static="", gravity="0 0 0")


def mixed_xml(*, dim=6, mass=1.0, h=0.001):
    body = f"""<body name="base" pos="0 0 .26">
      <joint name="turn" axis="0 1 0" limited="true" range="-.2 .2" margin=".01" frictionloss=".2"/>
      {_inertial(0.2 * mass)}
      <body name="tip" pos="0 0 -.16"><joint name="slide" type="slide" axis="0 0 1"
        limited="true" range="-.2 .2" margin=".01" frictionloss=".2"/>
        {_inertial(mass, True)}<geom name="tipshape" type="sphere" size=".101"/></body></body>"""
    return _xml(body, dim=dim, h=h)


def branch_xml(dim=6):
    body = '<body name="ancestor">'
    for i, (x, mass) in enumerate(((-0.099, 1.0), (0.099, 0.6))):
        body += f'''<body name="hinge{i}" pos="{x} 0 .2"><joint name="r{i}" axis="0 1 0"/>
          {_inertial(0.2)}<body name="tip{i}" pos="0 0 -.18">
          <joint name="p{i}" type="slide" axis="1 0 0"/>{_inertial(mass, True)}
          <geom name="shape{i}" type="capsule" size=".1 .1" pos="0 0 .02"/>
          </body></body>'''
    return _xml(body + "</body>", dim=dim, static="", gravity="0 0 0")


def packet(model, qforce=None, wrench=None):
    return normal._stored(
        MuJoCoControl(
            np.zeros(model.nu),
            qfrc_applied=np.zeros(model.nv) if qforce is None else qforce,
            xfrc_applied=np.zeros((model.nbody, 6)) if wrench is None else wrench,
        )
    )


def tiny_box_qp(matrix, linear, lower, upper):
    """Enumerate complete small box faces, independently of product active sets."""
    n = len(linear)
    assert n <= 6
    labels = [(0, 1, 2) if np.isfinite(upper[i]) else (0, 1) for i in range(n)]
    valid = []
    for flags in itertools.product(*labels):
        flags = np.array(flags)
        free = np.flatnonzero(flags == 1)
        fixed = np.flatnonzero(flags != 1)
        candidate = lower.copy()
        candidate[flags == 2] = upper[flags == 2]
        if len(free):
            rhs = -linear[free] - matrix[np.ix_(free, fixed)] @ candidate[fixed]
            candidate[free] = np.linalg.solve(matrix[np.ix_(free, free)], rhs)
        gradient = matrix @ candidate + linear
        if np.any(candidate < lower - 1e-8) or np.any(candidate > upper + 1e-8):
            continue
        if np.any(gradient[flags == 0] < -1e-7) or np.any(gradient[flags == 2] > 1e-7):
            continue
        if np.max(np.abs(gradient[free]), initial=0) > 1e-7:
            continue
        valid.append(candidate)
    assert valid, "independent strictly convex box problem has no feasible KKT face"
    for other in valid[1:]:
        np.testing.assert_allclose(other, valid[0], rtol=2e-6, atol=2e-6)
    return valid[0]


def native(mj, model, q, v, control, time=0.0, *, tiny=False):
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = q, v, time
    normal._set(data, control)
    mj.mj_forward(model, data)
    mass = np.empty((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    jc = np.zeros((model.nbody, 6, model.nv))
    applied = data.qfrc_actuator.copy() + data.qfrc_applied
    for body in range(model.nbody):
        mj.mj_jac(model, data, jc[body, :3], jc[body, 3:], data.xipos[body], body)
        applied += jc[body].T @ control.xfrc_applied[body]
    rhs = applied + data.qfrc_passive - data.qfrc_bias
    smooth = np.linalg.solve(mass, rhs)
    contacts = []
    for index in range(data.ncon):
        c = data.contact[index]
        geoms = tuple(map(int, c.geom))
        bodies = tuple(int(model.geom_bodyid[g]) for g in geoms)
        point_j = np.zeros((6, model.nv))
        frame = c.frame.reshape(3, 3).copy()
        for sign, body in ((-1, bodies[0]), (1, bodies[1])):
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mj.mj_jac(model, data, jp, jr, c.pos, body)
            point_j[:3] += sign * frame @ jp
            point_j[3:] += sign * frame @ jr
        wrench = np.zeros(6)
        mj.mj_contactForce(model, data, index, wrench)
        dim = int(c.dim)
        contacts.append(
            dict(
                native_index=index,
                geometry_ids=geoms,
                body_ids=bodies,
                dimension=dim,
                distance=float(c.dist),
                position=c.pos.copy(),
                frame=frame,
                include_margin=float(c.includemargin),
                friction=c.friction.copy(),
                solref=c.solref.copy(),
                solimp=c.solimp.copy(),
                active=c.efc_address >= 0,
                efc_address=int(c.efc_address),
                jacobian=point_j,
                wrench_contact_frame=wrench,
                normal_force=float(wrench[0]),
            )
        )
    n = data.nefc
    j = data.efc_J.reshape(n, model.nv).copy()
    types, ids = data.efc_type.copy(), data.efc_id.copy()
    rows = []
    for i in range(n):
        kind, source = int(types[i]), int(ids[i])
        if kind == 1:
            side, dof, contact = -1, source, -1
        elif kind == 3:
            dof = int(model.jnt_dofadr[source])
            side, contact = (-1 if j[i, dof] > 0 else 1), -1
        else:
            assert kind in (5, 6)
            dof, contact = -1, source
            side = i - contacts[source]["efc_address"]
        lower = -float(data.efc_frictionloss[i]) if kind == 1 else 0.0
        upper = float(data.efc_frictionloss[i]) if kind == 1 else None
        rows.append(
            dict(
                constraint_type=kind,
                source_id=source,
                side_or_edge=side,
                dof=dof,
                contact=contact,
                jacobian=j[i],
                position=float(data.efc_pos[i]),
                margin=float(data.efc_margin[i]),
                velocity=float(data.efc_vel[i]),
                reference_acceleration=float(data.efc_aref[i]),
                regularization=float(data.efc_R[i]),
                inverse_regularization=float(data.efc_D[i]),
                impedance=float(data.efc_KBIP[i, 2]),
                diag_approx=float(data.efc_diagApprox[i]),
                lower=lower,
                upper=upper,
                force=float(data.efc_force[i]),
            )
        )
    regularization = np.array([r["regularization"] for r in rows])
    aref = np.array([r["reference_acceleration"] for r in rows])
    force = np.array([r["force"] for r in rows])
    matrix = j @ np.linalg.solve(mass, j.T) + np.diag(regularization)
    linear = j @ smooth - aref
    lower = np.array([r["lower"] for r in rows])
    upper = np.array([np.inf if r["upper"] is None else r["upper"] for r in rows])
    if tiny and n <= 6:
        np.testing.assert_allclose(
            force, tiny_box_qp(matrix, linear, lower, upper), rtol=2e-6, atol=2e-6
        )
    np.testing.assert_allclose(j.T @ force, data.qfrc_constraint, rtol=1e-9, atol=1e-8)
    return dict(
        body_position=data.xpos.copy(),
        body_rotation=data.xmat.reshape(model.nbody, 3, 3).copy(),
        body_com=data.xipos.copy(),
        jacobian_com=jc,
        mass_matrix=mass,
        bias=data.qfrc_bias.copy(),
        passive=data.qfrc_passive.copy(),
        applied=applied,
        smooth_acceleration=smooth,
        qacc=data.qacc.copy(),
        integration_acceleration=np.linalg.solve(
            mass + model.opt.timestep * np.diag(model.dof_damping), rhs + data.qfrc_constraint
        ),
        qfrc_constraint=data.qfrc_constraint.copy(),
        qfrc_actuator=data.qfrc_actuator.copy(),
        contacts=contacts,
        rows=rows,
        matrix=matrix,
        linear=linear,
        lower=lower,
        upper_bound=[r["upper"] for r in rows],
        force=force,
    )


def match_contacts(actual, expected):
    assert len(actual) == len(expected)
    candidates = [
        [
            j
            for j, oracle in enumerate(expected)
            if tuple(c.geometry_ids) == oracle["geometry_ids"]
            and np.allclose(c.position, oracle["position"], rtol=3e-5, atol=4e-5)
            and np.allclose(c.frame[0], oracle["frame"][0], rtol=0, atol=1e-4)
        ]
        for c in actual
    ]
    matched = {}

    def augment(i, seen):
        for j in candidates[i]:
            if j not in seen:
                seen.add(j)
                if j not in matched or augment(matched[j], seen):
                    matched[j] = i
                    return True
        return False

    for i in range(len(actual)):
        assert augment(i, set()), "complete native geometry bijection missing"
    result = [None] * len(actual)
    for j, i in matched.items():
        result[i] = expected[j]
    assert len({r["native_index"] for r in result}) == len(expected)
    return result


def assert_receipt(row, expected, model):
    """Compare all rows after geometry-only contact bijection, never force matching."""
    dyn = row.dynamics
    np.testing.assert_array_equal(dyn.input_q, row.input_q)
    np.testing.assert_array_equal(dyn.input_v, row.input_v)
    assert dyn.time == row.input_time
    for name, (rtol, atol) in normal.TOL.items():
        value = getattr(dyn, name)
        assert value.shape == expected[name].shape
        np.testing.assert_allclose(value, expected[name], rtol=rtol, atol=atol, err_msg=name)
        with pytest.raises(ValueError):
            value.setflags(write=True)
    for name in ("qacc", "integration_acceleration"):
        np.testing.assert_allclose(getattr(row, name), expected[name], **normal.ACC_TOL)
    np.testing.assert_allclose(row.qfrc_constraint, expected["qfrc_constraint"], **FORCE_TOL)
    assert isinstance(row.constraints, tuple)
    contacts = match_contacts(row.contacts, expected["contacts"])
    n, nc, nv = len(row.constraints), len(contacts), model.nv
    assert n == len(expected["rows"])
    raw = row.raw_contact_fields
    ids, values, jac = (np.asarray(raw[key]) for key in ("efc_ids", "efc_data", "efc_J"))
    counts = np.asarray(raw["counts"])
    np.testing.assert_array_equal(counts, [nc, sum(c.active for c in row.contacts), n])
    cmap = np.asarray(raw["contact_map"])
    cj, cw = np.asarray(raw["contact_J"]), np.asarray(raw["contact_wrench"])
    assert ids.shape[1] == 5 and values.shape[1] == 14 and jac.shape[1] >= nv
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(jac))
    assert set(raw) >= {"candidate_rows", "candidate_jacobian", "working_set", "solver_diagnostics"}
    mapping = {i: c["native_index"] for i, c in enumerate(contacts)}
    expected_rows = {
        (r["constraint_type"], r["source_id"], r["side_or_edge"]): r for r in expected["rows"]
    }
    assert len(expected_rows) == n
    actual_keys, generalized = [], np.zeros(nv)
    force, lower, upper, aref, reg = [], [], [], [], []
    scalar_fields = (
        "position",
        "margin",
        "velocity",
        "reference_acceleration",
        "regularization",
        "inverse_regularization",
        "impedance",
        "diag_approx",
    )
    for i, r in enumerate(row.constraints):
        assert r.schema == "mujoco-tyx-constraint-row-v1"
        source = mapping[r.source_id] if r.constraint_type in (5, 6) else r.source_id
        key = (r.constraint_type, source, r.side_or_edge)
        actual_keys.append(key)
        oracle = expected_rows[key]
        assert r.dof == oracle["dof"]
        np.testing.assert_array_equal(
            ids[i], [r.constraint_type, r.source_id, r.side_or_edge, r.geometry_slot, r.dof]
        )
        if r.constraint_type in (5, 6):
            assert r.geometry_slot == cmap[r.source_id, 0]
        else:
            assert r.geometry_slot == -1
        for field in scalar_fields:
            np.testing.assert_allclose(getattr(r, field), oracle[field], **ROW_TOL, err_msg=field)
        np.testing.assert_allclose(r.jacobian, oracle["jacobian"], rtol=2e-4, atol=5e-5)
        np.testing.assert_array_equal(r.jacobian, jac[i, :nv])
        np.testing.assert_allclose(r.force, oracle["force"], **FORCE_TOL)
        np.testing.assert_allclose(r.lower, oracle["lower"], rtol=0, atol=2e-7)
        assert (r.upper is None) == (oracle["upper"] is None)
        if r.upper is not None:
            np.testing.assert_allclose(r.upper, oracle["upper"], rtol=0, atol=2e-7)
        assert values[i, 11] == 1
        np.testing.assert_array_equal(
            values[i, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]],
            [
                r.position,
                r.margin,
                r.velocity,
                r.reference_acceleration,
                r.regularization,
                r.inverse_regularization,
                r.impedance,
                r.lower,
                0 if r.upper is None else r.upper,
                r.force,
                r.gradient,
                int(r.upper is not None),
                r.diag_approx,
            ],
        )
        force.append(r.force)
        lower.append(r.lower)
        upper.append(np.inf if r.upper is None else r.upper)
        aref.append(r.reference_acceleration)
        reg.append(r.regularization)
        generalized += r.jacobian * r.force
        with pytest.raises((AttributeError, TypeError)):
            r.force = 0
    assert len(set(actual_keys)) == n and set(actual_keys) == set(expected_rows)
    np.testing.assert_allclose(generalized, row.qfrc_constraint, rtol=5e-5, atol=5e-5)
    contact_generalized = np.zeros(nv)
    for i, (c, oracle) in enumerate(zip(row.contacts, contacts, strict=True)):
        assert c.schema == "mujoco-tyx-pyramidal-contact-v1"
        assert c.dimension == oracle["dimension"] and c.active == oracle["active"]
        assert tuple(c.body_ids) == oracle["body_ids"]
        np.testing.assert_allclose(c.distance, oracle["distance"], rtol=3e-5, atol=3e-5)
        for field in ("position", "frame", "friction", "solref", "solimp", "include_margin"):
            np.testing.assert_allclose(getattr(c, field), oracle[field], rtol=3e-5, atol=4e-5)
        np.testing.assert_allclose(c.jacobian, oracle["jacobian"], rtol=2e-4, atol=5e-5)
        np.testing.assert_array_equal(c.jacobian, cj[i, :, :nv])
        np.testing.assert_array_equal(c.wrench_contact_frame, cw[i])
        np.testing.assert_allclose(
            c.wrench_contact_frame, oracle["wrench_contact_frame"], **FORCE_TOL
        )
        assert c.normal_force == c.wrench_contact_frame[0]
        np.testing.assert_allclose(
            c.normal_velocity, c.jacobian[0] @ row.input_v, rtol=2e-5, atol=2e-5
        )
        first, count = map(int, cmap[i, 1:])
        expected_count = (1 if c.dimension == 1 else 2 * (c.dimension - 1)) if c.active else 0
        assert count == expected_count and (first >= 0 if c.active else first == -1)
        decoded = np.zeros(6)
        if c.active:
            edges = np.array([r.force for r in row.constraints[first : first + count]])
            assert all(r.source_id == i for r in row.constraints[first : first + count])
            decoded[0] = edges.sum()
            for k in range(c.dimension - 1):
                decoded[k + 1] = c.friction[k] * (edges[2 * k] - edges[2 * k + 1])
        np.testing.assert_allclose(c.wrench_contact_frame, decoded, rtol=3e-5, atol=3e-5)
        if c.dimension > 1:
            assert (
                np.sum(np.abs(decoded[1 : c.dimension]) / c.friction[: c.dimension - 1])
                <= decoded[0] + 5e-5
            )
        world_force = c.frame.T @ decoded[:3]
        world_couple = c.frame.T @ decoded[3:]
        expected_body_wrenches = []
        for sign, body in ((-1, c.body_ids[0]), (1, c.body_ids[1])):
            lever = c.position - dyn.body_com[body]
            expected_body_wrenches.append(
                np.r_[sign * world_force, sign * (world_couple + np.cross(lever, world_force))]
            )
        for actual, wanted in zip(
            (c.wrench_on_body1_world, c.wrench_on_body2_world), expected_body_wrenches, strict=True
        ):
            np.testing.assert_allclose(actual, wanted, rtol=4e-5, atol=4e-5)
        np.testing.assert_allclose(c.force_on_geometry1_world, -world_force, rtol=4e-5, atol=4e-5)
        np.testing.assert_allclose(c.force_on_geometry2_world, world_force, rtol=4e-5, atol=4e-5)
        # Both sides' actual COM Jacobians include angular reaction from spin/rolling.
        from_bodies = sum(
            dyn.jacobian_com[b].T @ w
            for b, w in zip(c.body_ids, expected_body_wrenches, strict=True)
        )
        np.testing.assert_allclose(from_bodies, c.jacobian.T @ decoded, rtol=4e-5, atol=4e-5)
        contact_generalized += from_bodies
    noncontact = sum(
        (r.jacobian * r.force for r in row.constraints if r.constraint_type not in (5, 6)),
        np.zeros(nv),
    )
    np.testing.assert_allclose(
        contact_generalized + noncontact, row.qfrc_constraint, rtol=5e-5, atol=5e-5
    )
    j = jac[:n, :nv].astype(float)
    f, lo, hi = np.array(force), np.array(lower), np.array(upper)
    hessian = j @ np.linalg.solve(dyn.mass_matrix.astype(float), j.T) + np.diag(reg)
    linear = j @ dyn.smooth_acceleration - np.array(aref)
    native_order = [
        next(
            i
            for i, r in enumerate(expected["rows"])
            if (r["constraint_type"], r["source_id"], r["side_or_edge"]) == key
        )
        for key in actual_keys
    ]
    np.testing.assert_allclose(
        hessian, expected["matrix"][np.ix_(native_order, native_order)], rtol=7e-4, atol=5e-4
    )
    gradient = hessian @ f + linear
    feasibility = max(np.max(lo - f, initial=0), np.max(f - hi, initial=0))
    assert feasibility <= 2e-7
    working = np.asarray(raw["working_set"])[:n]
    assert set(map(int, working)) <= {0, 1, 2}
    violation = np.where(
        working == 0,
        np.maximum(0, -gradient),
        np.where(working == 2, np.maximum(0, gradient), np.abs(gradient)),
    )
    for i in range(n):
        if working[i] == 0:
            assert abs(f[i] - lo[i]) <= 2e-7
        if working[i] == 2:
            assert np.isfinite(hi[i]) and abs(f[i] - hi[i]) <= 2e-7
    independent_scale = max(1.0, np.max(np.abs(linear) + np.abs(hessian) @ np.abs(f), initial=0))
    assert np.max(violation, initial=0) / independent_scale <= 5e-5
    for i, r in enumerate(row.constraints):
        np.testing.assert_allclose(
            r.gradient, gradient[i], rtol=3e-3, atol=5e-5 * independent_scale
        )
    physical_rhs = dyn.applied + dyn.passive - dyn.bias + row.qfrc_constraint
    mass_error = dyn.mass_matrix.astype(float) @ row.qacc - physical_rhs
    mass_scale = max(
        1.0, np.max(np.abs(physical_rhs) + np.abs(dyn.mass_matrix) @ np.abs(row.qacc), initial=0)
    )
    mass_residual = np.max(np.abs(mass_error), initial=0) / mass_scale
    assert mass_residual <= 5e-5
    diag = raw["solver_diagnostics"]
    assert diag["converged"] == 1 and diag["actual_efc"] == n
    assert (
        (1 if np.max(np.abs(f), initial=0) > 0 else 0)
        <= diag["face_solves"]
        <= diag["budget"]
        <= model.opt.iterations
    )
    assert diag["bound_violation"] <= 2e-7
    assert diag["normalized_kkt"] <= float(np.float32(max(float(model.opt.tolerance), 2e-5)))
    assert diag["normalized_mass_residual"] <= 5e-5
    # No stale tail rows or stale compact identities survive an unload/reset attempt.
    np.testing.assert_array_equal(values[n:], 0)
    np.testing.assert_array_equal(jac[n:], 0)
    np.testing.assert_array_equal(cw[nc:], 0)
    np.testing.assert_array_equal(cj[nc:], 0)
    np.testing.assert_array_equal(cmap[nc:, :2], -1)
    np.testing.assert_array_equal(cmap[nc:, 2], 0)
    return dict(
        matrix=hessian,
        linear=linear,
        force=f,
        kkt=float(np.max(violation, initial=0) / independent_scale),
        normalized_mass_residual=float(mass_residual),
    )


def initial(model, q=None, v=None):
    return MuJoCoInitialState(
        model.qpos0.copy() if q is None else q, np.zeros(model.nv) if v is None else v
    )


def run_path(
    xml,
    mode,
    path,
    *,
    q=None,
    v=None,
    steps=1,
    controls=None,
    tiny=True,
    long=False,
    wrench_body=None,
):
    mj = normal._deps()
    model = mj.MjModel.from_xml_string(xml)
    h = float(model.opt.timestep)
    target_body = (
        None if wrench_body is None else mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, wrench_body)
    )
    assert target_body is None or target_body > 0
    states, selected = [], set()
    with MuJoCoTyxRuntime(
        normal._compile(xml), mode=mode, initial_states={0: initial(model, q, v)}
    ) as runtime:
        start = runtime.snapshot()[0]
        cpu = mj.MjData(model)
        cpu.qpos[:], cpu.qvel[:] = start.q, start.v
        mj.mj_forward(model, cpu)
        _save(path, {"source_xml": xml, "initial": start, "h": h, "steps": steps})
        previous = None
        previous_signature = None
        for tick in range(steps):
            control = packet(model) if controls is None else controls(model, tick * h)
            before = runtime.snapshot()[0]
            expected = native(
                mj,
                model,
                before.q,
                before.v,
                control,
                before.time,
                tiny=tiny and (not long or tick in (0, steps - 1)),
            )
            runtime.set_controls({0: control})
            try:
                runtime.step(h)
                after = runtime.snapshot()[0]
                row = after.last_step
                assert abs(after.time - (tick + 1) * h) <= 2e-6
                assert abs(cpu.time - tick * h) <= 2e-6
                for field in ("ctrl", "qfrc_applied", "xfrc_applied"):
                    np.testing.assert_array_equal(
                        getattr(row.stored_control, field), getattr(control, field)
                    )
                independent = assert_receipt(row, expected, model)
                normal._set(cpu, control)
                mj.mj_step(model, cpu)
                np.testing.assert_allclose(after.q, cpu.qpos, rtol=0, atol=2e-3 if long else 2e-4)
                np.testing.assert_allclose(after.v, cpu.qvel, rtol=0, atol=2e-2 if long else 3e-3)
            except Exception:
                try:
                    scratch = runtime._device.download()
                except Exception as readback_error:
                    scratch = {"readback_error": str(readback_error)}
                _save(
                    path,
                    {
                        "failure_tick": tick,
                        "input": before,
                        "control": control,
                        "oracle": expected,
                        "published": runtime.snapshot()[0],
                        "attempt_scratch": scratch,
                    },
                )
                raise
            present = len(row.contacts)
            signature = tuple(
                (r.constraint_type, r.source_id, r.side_or_edge) for r in row.constraints
            )
            mark = (
                signature != previous_signature
                or tick in (0, steps - 1)
                or not long
                or present != previous
                or abs(tick * h - 0.096) < h / 2
                or abs(tick * h - 0.160) < h / 2
            )
            actual_bodies = np.zeros((model.nbody, 6))
            native_bodies = np.zeros((model.nbody, 6))
            continuous_bodies = np.zeros((model.nbody, 6))
            for c in row.contacts:
                actual_bodies[c.body_ids[0]] += c.wrench_on_body1_world
                actual_bodies[c.body_ids[1]] += c.wrench_on_body2_world
            for c in expected["contacts"]:
                fworld, tworld = (
                    c["frame"].T @ c["wrench_contact_frame"][:3],
                    c["frame"].T @ c["wrench_contact_frame"][3:],
                )
                for sign, body in ((-1, c["body_ids"][0]), (1, c["body_ids"][1])):
                    native_bodies[body] += (
                        sign
                        * np.r_[
                            fworld,
                            tworld + np.cross(c["position"] - expected["body_com"][body], fworld),
                        ]
                    )
            for i in range(cpu.ncon):
                buf = np.zeros(6)
                mj.mj_contactForce(model, cpu, i, buf)
                c = cpu.contact[i]
                frame = c.frame.reshape(3, 3)
                fw, tw = frame.T @ buf[:3], frame.T @ buf[3:]
                for sign, geom in ((-1, c.geom[0]), (1, c.geom[1])):
                    body = int(model.geom_bodyid[geom])
                    continuous_bodies[body] += (
                        sign * np.r_[fw, tw + np.cross(c.pos - cpu.xipos[body], fw)]
                    )
            actual_wrench, native_wrench, continuous_wrench = [
                a[1:].sum(axis=0) if target_body is None else a[target_body].copy()
                for a in (actual_bodies, native_bodies, continuous_bodies)
            ]
            light = dict(
                tick=tick,
                time=after.time,
                q=after.q,
                v=after.v,
                control=control,
                q_native=cpu.qpos.copy(),
                v_native=cpu.qvel.copy(),
                count=present,
                native_count=int(cpu.ncon),
                wrench=actual_wrench,
                same_input_native_wrench=native_wrench,
                continuous_wrench=continuous_wrench,
                wrench_body=target_body,
                body_wrenches=actual_bodies,
                native_body_wrenches=native_bodies,
                continuous_body_wrenches=continuous_bodies,
                efc=len(row.constraints),
                constraint_types=sorted({r.constraint_type for r in row.constraints}),
                constraint_keys=signature,
                limit_count=sum(r.constraint_type == 3 for r in row.constraints),
                continuous_limit_count=int(np.count_nonzero(np.asarray(cpu.efc_type) == 3)),
                kkt=independent["kkt"],
            )
            states.append(light)
            payload = dict(light)
            if mark:
                payload.update(actual=row, oracle=expected, independent=independent)
                selected.add(tick)
            _save(path, payload)
            previous = present
            previous_signature = signature
        return model, states, runtime.snapshot()[0], selected


@pytest.mark.parametrize(
    "matrix,linear,lower,upper,wanted",
    [
        ([[2, 1], [1, 2]], [-3, -1], [0, 0], [np.inf, np.inf], [1.5, 0]),
        ([[2]], [-3], [-0.5], [0.5], [0.5]),
        (
            [[2, 1, 1], [1, 2, 1], [1, 1, 2]],
            [-4, 0, 0],
            [0, -0.25, -0.25],
            [np.inf, 0.25, 0.25],
            [2.25, -0.25, -0.25],
        ),
        ([[2]], [-1], [0], [np.inf], [0.5]),
    ],
    ids=["normal-bind", "dry-upper", "simultaneous-two-bounds", "zero-jacobian"],
)
def test_independent_tiny_oracle_known_faces(matrix, linear, lower, upper, wanted):
    answer = tiny_box_qp(*(np.asarray(a, float) for a in (matrix, linear, lower, upper)))
    np.testing.assert_allclose(answer, wanted, rtol=0, atol=1e-12)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("speed", [-0.4, 0.4])
def test_true_frictionless_condim1_in_t4(mode, speed, tmp_path):
    _, _, after, _ = run_path(
        slider_xml(dim=1, limits=True, friction="0 0 0"),
        mode,
        tmp_path / "frictionless.jsonl",
        v=np.array([speed, 0, 0]),
        steps=16,
    )
    assert after.last_step.contacts
    for c in after.last_step.contacts:
        np.testing.assert_array_equal(c.wrench_contact_frame[1:], 0)
        assert c.dimension == 1


@pytest.mark.parametrize("mode", MODES)
def test_native_minmu_authored_zero_is_not_condim1(mode, tmp_path):
    _, _, after, _ = run_path(
        slider_xml(dim=3, friction="0 0 0"),
        mode,
        tmp_path / "native-minmu.jsonl",
        v=np.array([0.4, -0.2, 0]),
        steps=4,
        tiny=False,
    )
    assert after.last_step.contacts
    for c in after.last_step.contacts:
        assert c.dimension == 3 and np.all(np.asarray(c.friction) > 0)
        assert len(after.last_step.constraints) == 4


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("axis,speed", [(0, -0.4), (0, 0.4), (1, -0.4), (1, 0.4)])
def test_both_signed_sliding_directions(mode, axis, speed, tmp_path):
    import json

    velocity = np.zeros(3)
    velocity[axis] = speed
    _, _, after, _ = run_path(slider_xml(), mode, tmp_path / "sliding.jsonl", v=velocity, steps=24)
    # Initial penetration can release before the last step. Inspect the complete
    # already-saved physical path; do not rerun a solver or require forced contact.
    records = [
        json.loads(line)
        for line in (tmp_path / "sliding.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    receipts = [record for record in records if "actual" in record]
    assert [record["tick"] for record in receipts] == list(range(24))
    powers = []
    nonzero_friction = False
    for record in receipts:
        row = record["actual"]
        velocity = np.asarray(row["input_v"])
        power = 0.0
        for contact in row["contacts"]:
            tangent = np.asarray(contact["wrench_contact_frame"])[1:3]
            tangent_velocity = np.asarray(contact["jacobian"])[1:3] @ velocity
            power += float(tangent @ tangent_velocity)
            nonzero_friction |= bool(np.any(tangent != 0))
        powers.append(power)
    work = sum(
        record["actual"]["h"] * power for record, power in zip(receipts, powers, strict=True)
    )
    _save(
        tmp_path / "sliding.jsonl",
        {"tangential_power": powers, "friction_work": work, "nonzero_friction": nonzero_friction},
    )
    assert nonzero_friction and work < 0
    assert abs(after.v[axis]) < abs(speed)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ratio", [-1.2, -0.25, 0.25, 1.2])
def test_applied_load_static_and_sliding_sides(mode, ratio, tmp_path):
    def controls(model, time):
        return packet(model, np.array([ratio * 9.81, 0, 0]))

    _, states, after, _ = run_path(
        slider_xml(), mode, tmp_path / "load-regimes.jsonl", steps=64, controls=controls
    )
    # Soft normal/pyramidal coupling can lift the body temporarily, as native
    # does too. Require actual support/friction in the unchanged complete path.
    assert states[0]["count"] == 1
    assert all(s["count"] in (0, 1) for s in states)
    assert any(s["wrench"][2] > 0 for s in states)
    assert any(abs(s["wrench"][0]) > 0 for s in states)
    # A soft friction law is not relabelled as an exact zero-velocity constraint.
    if abs(ratio) > 1:
        assert np.sign(after.v[0]) == np.sign(ratio) and abs(after.v[0]) > 0.01


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fraction", [0.9, 1.0, 1.1])
def test_independently_derived_static_face_threshold(mode, fraction, tmp_path):
    mj = normal._deps()
    xml = slider_xml()
    model = mj.MjModel.from_xml_string(xml)
    q, v = model.qpos0.copy(), np.zeros(model.nv)
    zero = native(mj, model, q, v, packet(model), tiny=True)
    unit = native(mj, model, q, v, packet(model, np.array([1.0, 0, 0])), tiny=True)
    assert len(zero["rows"]) == 4
    np.testing.assert_array_equal(zero["matrix"], unit["matrix"])
    f0 = np.linalg.solve(zero["matrix"], -zero["linear"])
    df = np.linalg.solve(zero["matrix"], -(unit["linear"] - zero["linear"]))
    assert np.all(f0 > 0)
    thresholds = [-f0[i] / df[i] for i in range(4) if df[i] < -1e-12]
    assert thresholds
    critical = min(thresholds)
    assert critical > 0
    # Critical forcing is computed from source-only native rows before TYX construction.
    _save(
        tmp_path / "threshold.jsonl",
        {"critical": critical, "fraction": fraction, "zero": zero, "unit": unit},
    )

    def controls(model, time):
        return packet(model, np.array([fraction * critical, 0, 0]))

    run_path(xml, mode, tmp_path / "threshold.jsonl", steps=1, controls=controls)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "dim,axis,speed",
    [(4, 2, -2.0), (4, 2, 2.0), (6, 0, -2.0), (6, 0, 2.0), (6, 1, -2.0), (6, 1, 2.0)],
)
def test_torsional_and_rolling_couples(mode, dim, axis, speed, tmp_path):
    import json

    velocity = np.zeros(6)
    velocity[3 + axis] = speed
    run_path(
        free_xml(dim=dim, offset=False),
        mode,
        tmp_path / "couples.jsonl",
        v=velocity,
        steps=12,
        tiny=dim == 4,
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "couples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    receipts = [record for record in records if "actual" in record]
    assert [record["tick"] for record in receipts] == list(range(12))
    powers = []
    nonzero_body_torque = False
    native_couple_max = np.zeros(dim - 3)
    actual_couple_max = np.zeros(dim - 3)
    for record in receipts:
        row = record["actual"]
        power = 0.0
        for contact, oracle in zip(row["contacts"], record["oracle"]["contacts"], strict=True):
            assert tuple(contact["geometry_ids"]) == tuple(oracle["geometry_ids"])
            wrench = np.asarray(contact["wrench_contact_frame"])
            point_velocity = np.asarray(contact["jacobian"]) @ np.asarray(row["input_v"])
            # Point velocity already includes the lever; add only the pure
            # contact couple's angular power, never COM lever torque again.
            power += float(wrench[1:] @ point_velocity[1:])
            native_couple = np.asarray(oracle["wrench_contact_frame"])[3:dim]
            native_couple_max = np.maximum(native_couple_max, np.abs(native_couple))
            actual_couple_max = np.maximum(actual_couple_max, np.abs(wrench[3:dim]))
            for body, key in zip(
                contact["body_ids"], ("wrench_on_body1_world", "wrench_on_body2_world"), strict=True
            ):
                if body > 0:
                    nonzero_body_torque |= bool(np.linalg.norm(np.asarray(contact[key])[3:]) > 1e-5)
        powers.append(power)
    work = sum(
        record["actual"]["h"] * power for record, power in zip(receipts, powers, strict=True)
    )
    _save(
        tmp_path / "couples.jsonl",
        {
            "friction_power": powers,
            "friction_work": work,
            "native_couple_max": native_couple_max,
            "actual_couple_max": actual_couple_max,
            "nonzero_body_torque": nonzero_body_torque,
        },
    )
    assert work < 0 and nonzero_body_torque
    assert np.all(actual_couple_max[native_couple_max > 1e-5] > 1e-5)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("mass", [0.25, 2.0])
@pytest.mark.parametrize("h", [0.001, 0.002])
def test_oblique_offset_inertia_complete_six_dimensional_response(mode, mass, h, tmp_path):
    velocity = np.array([0.3, -0.15, -0.04, 1.0, -0.8, 0.6])

    def controls(model, time):
        w = np.zeros((model.nbody, 6))
        w[-1] = [0.07, -0.04, 0.02, 0.01, -0.02, 0.03]
        return packet(model, wrench=w)

    run_path(
        free_xml(dim=6, mass=mass, h=h, geometry="box", oblique=True),
        mode,
        tmp_path / "oblique.jsonl",
        v=velocity,
        steps=24,
        controls=controls,
        tiny=False,
    )


@pytest.mark.parametrize("mode", MODES)
def test_two_feet_shared_mass_is_not_independent_contacts(mode, tmp_path):
    mj = normal._deps()
    xml = free_xml(dim=3, two=True, offset=False)
    model = mj.MjModel.from_xml_string(xml)
    w = np.zeros((model.nbody, 6))
    w[-1, 4] = 0.03
    control = packet(model, wrench=w)
    expected = native(mj, model, model.qpos0, np.zeros(6), control)
    assert len(expected["contacts"]) == 2 and len(expected["rows"]) == 8
    separated = np.zeros(8)
    for c in expected["contacts"]:
        indices = np.arange(c["efc_address"], c["efc_address"] + 4)
        separated[indices] = tiny_box_qp(
            expected["matrix"][np.ix_(indices, indices)],
            expected["linear"][indices],
            np.zeros(4),
            np.full(4, np.inf),
        )
    assert np.max(np.abs(separated - expected["force"])) > 1e-3
    _save(
        tmp_path / "coupling.jsonl",
        {"independent_contact_forces": separated, "shared_cpu": expected},
    )
    run_path(
        xml,
        mode,
        tmp_path / "coupling.jsonl",
        steps=24,
        controls=lambda model, time: packet(model, wrench=w),
        tiny=False,
    )


@pytest.mark.parametrize("mode", MODES)
def test_same_ancestor_two_active_bodies_receive_complete_reaction(mode, tmp_path):
    model, _, after, _ = run_path(
        branch_xml(),
        mode,
        tmp_path / "two-sided.jsonl",
        v=np.array([0.1, 0.15, -0.08, -0.12]),
        steps=16,
        tiny=False,
    )
    assert after.last_step.contacts
    assert any(all(body > 0 for body in c.body_ids) for c in after.last_step.contacts)
    force = after.last_step.qfrc_constraint
    assert np.linalg.norm(force[:2]) > 1e-4 and np.linalg.norm(force[2:]) > 1e-4
    assert model.nv == 4


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", ["capsule", "box"])
def test_finite_surface_full_manifold_friction(mode, shape, tmp_path):
    xml = free_xml(dim=6, geometry=shape, oblique=True)
    xml = xml.replace(FLOOR, PLATFORM)
    old = 'pos="0 0 .099"' if shape == "capsule" else 'pos="0 0 .059"'
    new = 'pos="0 0 .149"' if shape == "capsule" else 'pos="0 0 .109"'
    xml = xml.replace(old, new)
    run_path(
        xml,
        mode,
        tmp_path / "finite-manifold.jsonl",
        v=np.array([0.12, -0.08, -0.03, 0.2, -0.1, 0.15]),
        steps=32,
        tiny=False,
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["hinge", "slide"])
@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("position", [0.195, 0.205])
@pytest.mark.parametrize("velocity", [-0.3, 0.3])
def test_joint_limit_inside_margin_and_outside_bounds(
    mode, kind, side, position, velocity, tmp_path
):
    _, _, after, _ = run_path(
        joint_xml(kind),
        mode,
        tmp_path / "joint-limit.jsonl",
        q=np.array([side * position]),
        v=np.array([velocity]),
        steps=1,
    )
    rows = [r for r in after.last_step.constraints if r.constraint_type == 3]
    assert len(rows) == 1 and rows[0].side_or_edge == side
    # Actual published positions are never projected to the admissible range.
    if position > 0.2:
        assert abs(after.q[0]) > 0.2


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["hinge", "slide"])
def test_joint_limit_interior_has_no_constraint(mode, kind, tmp_path):
    _, _, after, _ = run_path(joint_xml(kind), mode, tmp_path / "limit-inactive.jsonl", steps=1)
    assert not after.last_step.constraints


@pytest.mark.parametrize("mode", MODES)
def test_both_joint_limit_sides_survive_overlapping_margins(mode, tmp_path):
    _, _, after, _ = run_path(
        joint_xml(overlap=True), mode, tmp_path / "both-limits.jsonl", steps=1
    )
    rows = after.last_step.constraints
    assert len(rows) == 2 and {r.side_or_edge for r in rows} == {-1, 1}
    assert rows[0].source_id == rows[1].source_id
    np.testing.assert_allclose(rows[0].jacobian, -rows[1].jacobian, rtol=0, atol=0)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["hinge", "slide"])
@pytest.mark.parametrize(
    "load,velocity", [(-0.1, 0), (0.1, 0), (-0.3, 0), (0.3, 0), (0, -0.4), (0, 0.4)]
)
def test_dry_friction_bounded_force_with_viscous_damping(mode, kind, load, velocity, tmp_path):
    _, _, after, _ = run_path(
        joint_xml(kind, dry=0.2),
        mode,
        tmp_path / "dry.jsonl",
        v=np.array([velocity]),
        steps=8,
        controls=lambda model, time: packet(model, np.array([load])),
    )
    rows = [r for r in after.last_step.constraints if r.constraint_type == 1]
    assert len(rows) == 1 and rows[0].lower == np.float32(-0.2) and rows[0].upper == np.float32(0.2)
    if velocity:
        assert rows[0].force * velocity < 0
    assert abs(rows[0].force) <= float(np.float32(0.2))


@pytest.mark.parametrize("mode", MODES)
def test_free_root_all_six_dof_dry_rows(mode, tmp_path):
    body = (
        '<body name="free" quat=".96 .18 -.12 .1"><joint type="free" frictionloss=".2"/>'
        + _inertial()
        + "</body>"
    )
    xml = _xml(body, static="", gravity="0 0 0")
    _, _, after, _ = run_path(
        xml,
        mode,
        tmp_path / "free-dry.jsonl",
        v=np.array([0.3, -0.2, 0.1, 0.4, -0.2, 0.1]),
        steps=8,
    )
    rows = [r for r in after.last_step.constraints if r.constraint_type == 1]
    assert {r.dof for r in rows} == set(range(6))
    assert all(r.source_id == r.dof and r.side_or_edge == -1 for r in rows)


def _impulse_check(states, h):
    actual = h * np.sum([s["wrench"] for s in states], axis=0)
    continuous = h * np.sum([s["continuous_wrench"] for s in states], axis=0)
    np.testing.assert_array_less(np.abs(actual - continuous), 0.02 * np.abs(continuous) + 2e-3)
    return actual, continuous


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("mass", [0.25, 2.0])
@pytest.mark.parametrize("h", [0.001, 0.002])
def test_one_second_mixed_contact_limit_and_dry_continuation(mode, mass, h, tmp_path):
    def controls(model, time):
        load = np.array([0.03, -0.1 * mass * 9.81])
        if 0.096 <= time < 0.160:
            load = np.array([-0.03, 2 * mass * 9.81])
        return packet(model, load)

    _, states, after, selected = run_path(
        mixed_xml(mass=mass, h=h),
        mode,
        tmp_path / "mixed-window.jsonl",
        q=np.array([0.195, -0.005]),
        steps=round(1 / h),
        controls=controls,
        tiny=False,
        long=True,
        wrench_body="tip",
    )
    assert len(states) == round(1 / h) and after.tick == len(states)
    assert states[0]["count"] > 0 and states[0]["efc"] > 10
    assert np.max(np.abs(after.q - np.array([0.195, -0.005]))) > 1e-4
    assert states[0]["limit_count"] > 0 and states[0]["continuous_limit_count"] > 0
    leave_limit = next(i for i, s in enumerate(states) if s["limit_count"] == 0)
    cpu_leave_limit = next(i for i, s in enumerate(states) if s["continuous_limit_count"] == 0)
    leave, reenter = _first_separation_reentry(states, "count")
    cpu_leave, cpu_reenter = _first_separation_reentry(states, "native_count")
    assert abs(leave_limit - cpu_leave_limit) * h <= max(2 * h, 0.005)
    assert abs(leave - cpu_leave) * h <= max(2 * h, 0.005)
    assert abs(reenter - cpu_reenter) * h <= max(2 * h, 0.005)
    actual, expected = _impulse_check(states, h)
    _save(
        tmp_path / "mixed-window.jsonl",
        {
            "selected_ticks": sorted(selected),
            "leave_limit": leave_limit,
            "cpu_leave_limit": cpu_leave_limit,
            "leave": leave,
            "reenter": reenter,
            "cpu_leave": cpu_leave,
            "cpu_reenter": cpu_reenter,
            "actual_impulse": actual,
            "continuous_native_impulse": expected,
        },
    )


def _first_separation_reentry(states, count_key):
    first = next(i for i, s in enumerate(states) if s[count_key] == 0)
    again = next(i for i in range(first + 1, len(states)) if states[i][count_key] > 0)
    return first, again


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", ["capsule", "box"])
@pytest.mark.parametrize("mass", [0.25, 2.0])
@pytest.mark.parametrize("h", [0.001, 0.002])
def test_one_second_finite_unload_and_reentry_without_reset(mode, shape, mass, h, tmp_path):
    def controls(model, time):
        load = np.zeros(model.nv)
        if 0.096 <= time < 0.160:
            load[2] = 3 * mass * 9.81
        return packet(model, load)

    _, states, after, selected = run_path(
        slider_xml(dim=6, shape=shape, mass=mass, h=h, finite=True),
        mode,
        tmp_path / "unload-reenter.jsonl",
        v=np.array([0.15, 0, -0.02]),
        steps=round(1 / h),
        controls=controls,
        tiny=False,
        long=True,
        wrench_body="root",
    )
    assert len(states) == round(1 / h) and after.tick == len(states)
    assert states[0]["count"] > 0
    leave, reenter = _first_separation_reentry(states, "count")
    cpu_leave, cpu_reenter = _first_separation_reentry(states, "native_count")
    assert (reenter - leave) * h >= 0.02 and states[reenter]["time"] < 0.7
    assert abs(leave - cpu_leave) * h <= max(2 * h, 0.005)
    assert abs(reenter - cpu_reenter) * h <= max(2 * h, 0.005)
    last = states[-round(0.05 / h) :]
    mean_force = np.mean([s["wrench"][2] for s in last])
    assert 0.7 * mass * 9.81 < mean_force < 1.3 * mass * 9.81
    assert abs(after.v[2]) < 0.1 and abs(after.q[2]) < 0.01
    actual, expected = _impulse_check(states, h)
    _save(
        tmp_path / "unload-reenter.jsonl",
        {
            "selected_ticks": sorted(selected),
            "leave_tick": leave,
            "reenter_tick": reenter,
            "cpu_leave_tick": cpu_leave,
            "cpu_reenter_tick": cpu_reenter,
            "actual_impulse": actual,
            "continuous_native_impulse": expected,
            "final_mean_support_force": mean_force,
        },
    )


@pytest.mark.parametrize("mode", MODES)
def test_oblique_plane_frame_full_friction_wrench(mode, tmp_path):
    xml = free_xml(dim=6, oblique=True)
    xml = xml.replace(
        FLOOR,
        '<geom name="floor" type="plane" size="2 2 .1" quat=".9659258262890683 0 .25881904510252074 0"/>',
    )
    xml = xml.replace('pos="0 0 .099"', 'pos=".0495 0 .08573651497465943"')
    _, _, after, _ = run_path(
        xml,
        mode,
        tmp_path / "oblique-frame.jsonl",
        v=np.array([-0.15, -0.12, -0.18, 1.0, -0.8, 0.6]),
        steps=16,
        tiny=False,
    )
    assert after.last_step.contacts
    assert abs(after.last_step.contacts[0].frame[0, 0]) > 0.4
    assert any(np.linalg.norm(c.wrench_contact_frame[3:]) > 1e-6 for c in after.last_step.contacts)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dim", [3, 6])
def test_same_input_mixed_classes_share_one_physical_solve(mode, dim, tmp_path):
    def controls(model, time):
        return packet(model, np.array([0.03, -0.981]))

    _, states, _, _ = run_path(
        mixed_xml(dim=dim),
        mode,
        tmp_path / "mixed-short.jsonl",
        q=np.array([0.195, -0.005]),
        steps=8,
        controls=controls,
        tiny=False,
    )
    assert set(states[0]["constraint_types"]) >= {1, 3, 6}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "solref,solimp",
    [
        (".02 1", ".8 .9 .002 .5 2"),
        ("-1000 -30", ".8 .9 .002 .5 2"),
        (".02 1", ".2 .8 0 .5 2"),
        (".02 1", ".9 .2 .002 .5 2"),
        (".0001 1", ".8 .9 .002 .5 2"),
    ],
    ids=["default", "direct-kb", "zero-width", "decreasing-impedance", "refsafe"],
)
def test_new_mixed_row_law_branches_match_native(mode, solref, solimp, tmp_path):
    xml = slider_xml(dim=3).replace('solref=".02 1"', f'solref="{solref}"')
    xml = xml.replace('solimp=".8 .9 .002 .5 2"', f'solimp="{solimp}"')
    run_path(xml, mode, tmp_path / "law-branches.jsonl", v=np.array([0.1, -0.03, -0.01]), steps=1)
