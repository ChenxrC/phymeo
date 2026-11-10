"""Independent CPU MuJoCo oracles for coupled TYX normal contacts.

Authored after preregistration; project execution belongs to the parent.
"""

import itertools
import sys
from dataclasses import dataclass

import numpy as np
import pytest
from test_mujoco_tyx_floating import _same, _save

from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime, MuJoCoTyxUnsupported
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.resident_stepper import PhysicsStepFailure

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
H = 0.001
TOL = {
    "body_position": (3e-5, 4e-5),
    "body_rotation": (3e-5, 4e-5),
    "body_com": (3e-5, 4e-5),
    "jacobian_com": (2e-4, 5e-5),
    "mass_matrix": (3e-4, 5e-5),
    "bias": (4e-4, 4e-4),
    "passive": (4e-4, 4e-4),
    "applied": (4e-4, 4e-4),
    "smooth_acceleration": (2e-3, 4e-3),
}
FORCE_TOL = dict(rtol=3e-3, atol=3e-3)
ACC_TOL = dict(rtol=3e-3, atol=1e-2)
Q_ATOL, V_ATOL = 2e-4, 3e-3


def _deps():
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("tyx")
    assert mj.__version__ == "3.3.7"
    return mj


def _xml(bodies, *, gravity="0 0 -9.81", static="", extra=""):
    return f'''<mujoco model="t3-normal"><compiler angle="radian" fusestatic="false"/>
    <option timestep=".001" gravity="{gravity}" integrator="Euler" solver="Newton"
      iterations="128" tolerance="1e-10" jacobian="dense"/>
    <default><geom condim="1" solref=".02 1" solimp=".8 .9 .002 .5 2"/>
      <joint armature=".01" damping=".04"/></default>
    <worldbody>{static}{bodies}</worldbody>{extra}</mujoco>'''


FLOOR = '<geom name="floor" type="plane" size="2 2 .1"/>'
ROOT_INERTIA = '<inertial mass="2" pos=".03 -.02 .01" fullinertia=".08 .1 .12 .002 -.001 .003"/>'


def _free(shape, position, *, static=FLOOR, gravity="0 0 -9.81", quat="1 0 0 0"):
    body = f'''<body name="root" pos="{position}" quat="{quat}"><freejoint/>
      {ROOT_INERTIA}{shape}</body>'''
    return _xml(body, gravity=gravity, static=static)


def _slider_pair(count=2, *, z=".099", inactive=False):
    material = 'margin=".015625" gap=".0078125"' if inactive else ""
    geoms = "".join(
        f'<geom name="ball{i}" type="sphere" size=".1" pos="{0.3 * i - 0.15} 0 0" {material}/>'
        for i in range(count)
    )
    body = f"""<body name="root" pos="0 0 {z}">
    <joint name="vertical" type="slide" axis="0 0 1" armature="0" damping="0"/>
    <inertial mass="1" pos="0 0 0" diaginertia=".04 .05 .06"/>{geoms}</body>"""
    return _xml(body, static=FLOOR)


def _two_feet():
    return _xml(
        f"""<body name="root" pos="0 0 .4"><freejoint/>{ROOT_INERTIA}
      <body name="foot0" pos="-.18 0 -.301"><joint name="ankle" axis="0 1 0"/>
      <inertial mass=".4" pos=".02 0 .01" diaginertia=".01 .012 .014"/>
      <geom name="toe0" type="sphere" size=".1" pos=".04 0 0"/></body>
      <body name="foot1" pos=".18 0 -.301"><joint name="leg" type="slide" axis="0 0 1"/>
      <inertial mass=".5" pos="-.01 0 .01" diaginertia=".012 .014 .016"/>
      <geom name="toe1" type="sphere" size=".1"/></body></body>""",
        static=FLOOR,
    )


def _siblings():
    return _xml(
        """<body name="root">
    <body name="left" pos="-.099 0 .2"><joint name="a" type="slide" axis="1 0 0"/>
      <inertial mass="1" pos="0 .02 0" diaginertia=".03 .04 .05"/>
      <geom name="left-shape" type="sphere" size=".1"/></body>
    <body name="right" pos=".099 0 .2"><joint name="b" type="slide" axis="1 0 0"/>
      <inertial mass="1.4" pos="0 -.01 0" diaginertia=".04 .05 .06"/>
      <geom name="right-shape" type="sphere" size=".1"/></body></body>""",
        gravity="0 0 0",
    )


def _tree29():
    def node(i):
        if i >= 29:
            return ""
        kind = "slide" if i % 3 == 2 else "hinge"
        geom = (
            '<geom name="last-foot" type="sphere" size=".1" pos=".06 0 -.061"/>' if i == 14 else ""
        )
        return f'''<body name="link{i}" pos=".03 .02 -.06">
        <joint name="j{i}" type="{kind}" axis="{"0 0 1" if i == 14 else "0 1 0"}"/>
        <inertial mass=".3" pos=".015 -.01 .02" diaginertia=".01 .012 .014"/>
        {geom}{node(2 * i + 1)}{node(2 * i + 2)}</body>'''

    return _xml(
        f"""<body name="marker" pos="2 0 0"><inertial mass=".2" pos="0 0 0"
      diaginertia=".01 .01 .01"/></body>
      <body name="root" pos="0 0 .4"><freejoint/>{ROOT_INERTIA}{node(0)}
      <body name="tool" pos=".1 0 .1"><inertial mass=".2" pos="0 0 .02"
      diaginertia=".01 .012 .014"/></body></body>""",
        static=FLOOR,
    )


@dataclass(frozen=True)
class Case:
    name: str
    xml: str
    steps: int
    velocity: tuple = ()
    initial_contacts: int | None = None


BOX = '<geom name="obstacle" type="box" size=".2 .2 .05"/>'
CASES = (
    Case("redundant-slider", _slider_pair(), 32, (-0.03,), 2),
    Case(
        "eccentric-box",
        _free('<geom name="shape" type="box" size=".18 .12 .08"/>', "0 0 .079"),
        48,
        (0.02, -0.01, -0.06, 0.04, -0.03, 0.02),
        4,
    ),
    Case("two-feet", _two_feet(), 48, (0.01, 0, -0.06, 0.04, -0.03, 0.02, 0.015, -0.02), 2),
    Case("sibling-moving", _siblings(), 24, (0.06, -0.04), 1),
    Case(
        "finite-withdraw",
        _free('<geom name="shape" type="sphere" size=".1"/>', ".19 0 .149", static=BOX),
        128,
        (1.0, 0, 0, 0, 0, 0),
        1,
    ),
    Case("tree29-last-contact", _tree29(), 8, (0, 0, -0.04, 0.02, -0.01, 0.015) + (0.0,) * 29, 1),
)


def _geometry_cases():
    sphere = '<geom name="shape" type="sphere" size=".1"/>'
    capsule = '<geom name="shape" type="capsule" size=".1 .2" quat="{quat}"/>'
    box = '<geom name="shape" type="box" size=".18 .12 .08"/>'
    out = [
        Case("plane-sphere", _free(sphere, "0 0 .099"), 1, initial_contacts=1),
        Case(
            "plane-capsule-two",
            _free(capsule.format(quat=".7071067811865476 0 .7071067811865476 0"), "0 0 .099"),
            1,
            initial_contacts=2,
        ),
        Case(
            "plane-capsule-one",
            _free(capsule.format(quat="1 0 0 0"), "0 0 .299"),
            1,
            initial_contacts=1,
        ),
        Case("plane-box-four", _free(box, "0 0 .079"), 1, initial_contacts=4),
        Case(
            "plane-box-tilted",
            _free(box, "0 0 .11", quat=".9659258262890683 0 .2588190451025207 0"),
            1,
            initial_contacts=2,
        ),
    ]
    obstacles = (
        ("sphere-sphere", '<geom name="obstacle" type="sphere" size=".1"/>', ".199 0 0"),
        ("sphere-capsule-side", '<geom name="obstacle" type="capsule" size=".1 .2"/>', ".199 0 0"),
        ("sphere-capsule-end", '<geom name="obstacle" type="capsule" size=".1 .2"/>', "0 0 .399"),
        ("sphere-box-face", BOX, "0 0 .149"),
        ("sphere-box-edge", BOX, ".26 0 .12"),
        ("sphere-box-corner", BOX, ".25 .25 .11"),
        ("sphere-box-inside", BOX, ".18 0 .01"),
    )
    out.extend(
        Case(name, _free(sphere, pos, static=obstacle, gravity="0 0 0"), 1, initial_contacts=1)
        for name, obstacle, pos in obstacles
    )
    return tuple(out)


GEOMETRY_CASES = _geometry_cases()


def _initial(model, case, slot=0):
    q = model.qpos0.copy()
    v = np.zeros(model.nv)
    if case.velocity:
        v[:] = case.velocity
    if slot:
        if model.nq == model.nv:
            q[:] += 0.002 * slot
        else:
            q[2] += 0.002 * slot
        v *= 1 + 0.1 * slot
    return MuJoCoInitialState(q, v)


def _packet(model, case, tick=0, slot=0):
    qforce = np.zeros(model.nv)
    wrench = np.zeros((model.nbody, 6))
    if case.name in ("eccentric-box", "two-feet", "tree29-last-contact"):
        qforce[-1] = 0.015 * (-1 if tick >= case.steps // 2 else 1)
        wrench[-1] = [0.05, -0.03, 0.1 + 0.02 * slot, 0.003, -0.002, 0.004]
    return MuJoCoControl(np.zeros(model.nu), qfrc_applied=qforce, xfrc_applied=wrench)


def _stored(packet):
    return MuJoCoControl(
        np.asarray(packet.ctrl, np.float32),
        qfrc_applied=np.asarray(packet.qfrc_applied, np.float32),
        xfrc_applied=np.asarray(packet.xfrc_applied, np.float32),
    )


def _set(data, packet):
    data.ctrl[:], data.qfrc_applied[:], data.xfrc_applied[:] = (
        packet.ctrl,
        packet.qfrc_applied,
        packet.xfrc_applied,
    )


def _compile(xml):
    return compile_mujoco_xml(xml, require_asset=False)


def _small_qp(matrix, linear):
    """Exhaustive test-only active sets, independent of native/product solvers."""
    count = len(linear)
    assert count <= 4
    candidates = []
    for flags in itertools.product((False, True), repeat=count):
        active = np.flatnonzero(flags)
        force = np.zeros(count)
        if len(active):
            force[active] = np.linalg.solve(matrix[np.ix_(active, active)], -linear[active])
        gradient = matrix @ force + linear
        if np.min(force, initial=0) >= -1e-8 and np.min(gradient, initial=0) >= -1e-7:
            if np.max(np.abs(force * gradient), initial=0) <= 1e-6:
                candidates.append(force)
    assert candidates, "independent convex QP has no valid enumerated active set"
    for force in candidates[1:]:
        np.testing.assert_allclose(force, candidates[0], rtol=1e-8, atol=1e-8)
    return candidates[0]


def _native(mj, model, q, v, packet, time):
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = q, v, time
    _set(data, packet)
    mj.mj_forward(model, data)
    mass = np.empty((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    jaccom = np.zeros((model.nbody, 6, model.nv))
    applied = data.qfrc_actuator.copy() + data.qfrc_applied
    for body in range(model.nbody):
        mj.mj_jac(model, data, jaccom[body, :3], jaccom[body, 3:], data.xipos[body], body)
        applied += jaccom[body].T @ packet.xfrc_applied[body]
    rhs = applied + data.qfrc_passive - data.qfrc_bias
    smooth = np.linalg.solve(mass, rhs)
    contacts = []
    for index in range(data.ncon):
        c = data.contact[index]
        geom = tuple(map(int, c.geom))
        bodies = tuple(int(model.geom_bodyid[g]) for g in geom)
        jacpoints = []
        for body in bodies:
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mj.mj_jac(model, data, jp, jr, c.pos, body)
            jacpoints.append(jp)
        normal = c.frame.reshape(3, 3)[0].copy()
        j = normal @ (jacpoints[1] - jacpoints[0])
        adr = int(c.efc_address)
        force = np.zeros(6)
        mj.mj_contactForce(model, data, index, force)
        item = dict(
            geometry_ids=geom,
            body_ids=bodies,
            distance=float(c.dist),
            position=c.pos.copy(),
            frame=c.frame.reshape(3, 3).copy(),
            include_margin=float(c.includemargin),
            friction=c.friction.copy(),
            solref=c.solref.copy(),
            solimp=c.solimp.copy(),
            active=adr >= 0,
            jacobian=j,
            normal_velocity=float(j @ v),
            normal_force=float(force[0]),
            wrench_contact_frame=force,
            efc_address=adr,
        )
        if adr >= 0:
            item.update(
                regularization=float(data.efc_R[adr]),
                reference_acceleration=float(data.efc_aref[adr]),
            )
            native_j = data.efc_J.reshape(data.nefc, model.nv)[adr]
            np.testing.assert_allclose(j, native_j, rtol=1e-11, atol=1e-11)
            invweight = sum(model.body_invweight0[body, 0] for body in bodies)
            item["impedance"] = invweight / (invweight + item["regularization"])
        feature = 0
        if model.geom_type[geom[0]] == mj.mjtGeom.mjGEOM_PLANE:
            other = geom[1]
            rotation = data.geom_xmat[other].reshape(3, 3)
            center = data.geom_xpos[other]
            if model.geom_type[other] == mj.mjtGeom.mjGEOM_CAPSULE:
                endpoint = c.pos + normal * (model.geom_size[other, 0] + 0.5 * c.dist)
                coordinate = float((endpoint - center) @ rotation[:, 2])
                feature = 0 if coordinate > 0 else 1
            elif model.geom_type[other] == mj.mjtGeom.mjGEOM_BOX:
                corner = rotation.T @ (c.pos + 0.5 * c.dist * normal - center)
                feature = sum((1 << axis) for axis in range(3) if corner[axis] > 0)
        item["feature_index"] = feature

        contacts.append(item)
    active = [c for c in contacts if c["active"]]
    j = np.array([c["jacobian"] for c in active]).reshape(len(active), model.nv)
    reg = np.array([c["regularization"] for c in active])
    dual = j @ np.linalg.solve(mass, j.T) + np.diag(reg)
    linear = j @ smooth - np.array([c["reference_acceleration"] for c in active])
    forces = np.array([c["normal_force"] for c in active])
    if len(active) <= 4:
        np.testing.assert_allclose(forces, _small_qp(dual, linear), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(j.T @ forces, data.qfrc_constraint, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(smooth, data.qacc_smooth, rtol=1e-10, atol=1e-9)
    return dict(
        body_position=data.xpos.copy(),
        body_rotation=data.xmat.reshape(model.nbody, 3, 3).copy(),
        body_com=data.xipos.copy(),
        jacobian_com=jaccom,
        mass_matrix=mass,
        bias=data.qfrc_bias.copy(),
        passive=data.qfrc_passive.copy(),
        applied=applied,
        smooth_acceleration=smooth,
        qacc=data.qacc.copy(),
        qfrc_constraint=data.qfrc_constraint.copy(),
        integration_acceleration=np.linalg.solve(
            mass + H * np.diag(model.dof_damping), rhs + data.qfrc_constraint
        ),
        contacts=contacts,
        dual_matrix=dual,
        linear=linear,
        active_force=forces,
    )


def _match(actual, expected):
    """Complete pair-local bijection; deterministic positions, no dropped rows."""
    assert len(actual) == len(expected)
    order = []
    unused = set(range(len(expected)))
    for contact in actual:
        candidates = [
            i
            for i in unused
            if tuple(contact.geometry_ids) == expected[i]["geometry_ids"]
            and np.max(np.abs(contact.position - expected[i]["position"])) <= 4e-5
        ]
        assert len(candidates) == 1, ("ambiguous or missing native contact", candidates)
        found = candidates[0]
        unused.remove(found)
        order.append(expected[found])
    assert not unused
    return order


def _assert_receipt(row, expected, model):
    dyn = row.dynamics
    np.testing.assert_array_equal(dyn.input_q, row.input_q)
    np.testing.assert_array_equal(dyn.input_v, row.input_v)
    assert dyn.time == row.input_time
    for field, (rtol, atol) in TOL.items():
        value = getattr(dyn, field)
        assert value.shape == expected[field].shape
        np.testing.assert_allclose(value, expected[field], rtol=rtol, atol=atol, err_msg=field)
        with pytest.raises(ValueError):
            value.setflags(write=True)
    for field in ("qacc", "integration_acceleration"):
        np.testing.assert_allclose(getattr(row, field), expected[field], **ACC_TOL, err_msg=field)
    np.testing.assert_allclose(row.qfrc_constraint, expected["qfrc_constraint"], **FORCE_TOL)
    native = _match(row.contacts, expected["contacts"])
    generalized = np.zeros(model.nv)
    actual_active, expected_active, raw_indices = [], [], []
    raw = row.raw_contact_fields
    device = np.asarray(raw["device_rows"])
    jac = np.asarray(raw["normal_jacobian"])
    slots = np.asarray(raw["slot_ids"])
    assert device.ndim == 2 and device.shape[1] == 32
    assert jac.shape == (len(device), model.nv) and slots.shape == (len(device), 4)
    assert np.count_nonzero(device[:, 0]) == len(row.contacts)
    for contact, oracle in zip(row.contacts, native, strict=True):
        assert contact.schema == "mujoco-tyx-primitive-normal-v1"
        assert tuple(contact.body_ids) == oracle["body_ids"]
        assert contact.dimension == 1 and contact.active == oracle["active"]
        assert contact.feature_index == oracle["feature_index"]
        normal = np.asarray(contact.frame)[0]
        np.testing.assert_allclose(normal, oracle["frame"][0], rtol=0, atol=1e-4)
        np.testing.assert_allclose(contact.frame @ contact.frame.T, np.eye(3), rtol=0, atol=2e-4)
        np.testing.assert_allclose(contact.position, oracle["position"], rtol=3e-5, atol=4e-5)
        assert contact.distance == pytest.approx(oracle["distance"], rel=3e-5, abs=4e-5)
        for name in ("include_margin", "friction", "solref", "solimp"):
            np.testing.assert_allclose(getattr(contact, name), oracle[name], rtol=3e-6, atol=3e-7)
        np.testing.assert_allclose(contact.jacobian, oracle["jacobian"], rtol=2e-4, atol=5e-5)
        assert contact.normal_velocity == pytest.approx(
            oracle["normal_velocity"], rel=5e-4, abs=4e-4
        )
        assert contact.normal_force >= 0
        np.testing.assert_allclose(contact.normal_force, oracle["normal_force"], **FORCE_TOL)
        np.testing.assert_allclose(
            contact.wrench_contact_frame, [contact.normal_force, 0, 0, 0, 0, 0], rtol=0, atol=1e-7
        )
        force = normal * contact.normal_force
        np.testing.assert_allclose(contact.force_on_geometry1_world, -force, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(contact.force_on_geometry2_world, force, rtol=1e-6, atol=1e-6)
        for side, sign in ((1, -1), (2, 1)):
            body = contact.body_ids[side - 1]
            wrench = np.r_[
                sign * force, np.cross(contact.position - dyn.body_com[body], sign * force)
            ]
            np.testing.assert_allclose(
                getattr(contact, f"wrench_on_body{side}_world"), wrench, rtol=2e-5, atol=2e-5
            )
        generalized += np.asarray(contact.jacobian) * contact.normal_force
        locations = np.flatnonzero(
            (slots[:, 1] == contact.feature_index)
            & (slots[:, 2] == contact.geometry_ids[0])
            & (slots[:, 3] == contact.geometry_ids[1])
        )
        assert len(locations) == 1
        k = int(locations[0])
        raw_indices.append(k)
        assert device[k, 0] == 1 and bool(device[k, 1]) == contact.active
        np.testing.assert_array_equal(jac[k], contact.jacobian)
        np.testing.assert_allclose(
            device[k, [2, 3, 4, 6, 7, 8, 21]],
            [
                contact.distance,
                contact.normal_force,
                contact.regularization,
                contact.reference_acceleration,
                contact.normal_velocity,
                contact.impedance,
                contact.include_margin,
            ],
            rtol=0,
            atol=0,
        )
        np.testing.assert_array_equal(device[k, 9:12], contact.position)
        np.testing.assert_array_equal(device[k, 12:21].reshape(3, 3), contact.frame)
        np.testing.assert_array_equal(device[k, 27:32], np.zeros(5))
        if contact.active:
            for name in ("regularization", "reference_acceleration", "impedance"):
                np.testing.assert_allclose(
                    getattr(contact, name), oracle[name], rtol=5e-4, atol=4e-4
                )
            actual_active.append(contact)
            expected_active.append(oracle)
        else:
            assert contact.normal_force == 0
    assert set(raw_indices) == set(np.flatnonzero(device[:, 0]))
    np.testing.assert_allclose(generalized, row.qfrc_constraint, rtol=3e-5, atol=3e-5)
    if actual_active:
        j = np.array([c["jacobian"] for c in expected_active])
        r = np.array([c["regularization"] for c in expected_active])
        matrix = j @ np.linalg.solve(expected["mass_matrix"], j.T) + np.diag(r)
        linear = j @ expected["smooth_acceleration"] - np.array(
            [c["reference_acceleration"] for c in expected_active]
        )
        force = np.array([c.normal_force for c in actual_active])
        gradient = matrix @ force + linear
        violation = np.where(force > 0, np.abs(gradient), np.maximum(0, -gradient))
        scale = max(1.0, float(np.max(np.abs(linear) + np.abs(matrix) @ np.abs(force))))
        assert np.max(violation) / scale <= 5e-5
        active_slots = [k for k in raw_indices if device[k, 1]]
        np.testing.assert_allclose(
            np.asarray(raw["dual_matrix"])[np.ix_(active_slots, active_slots)],
            matrix,
            rtol=7e-4,
            atol=5e-4,
        )
    diagnostics = raw["solver_diagnostics"]
    assert diagnostics["converged"]
    assert diagnostics["iterations"] <= diagnostics["budget"] == model.opt.iterations
    assert diagnostics["present_count"] == len(row.contacts)
    assert diagnostics["active_count"] == len(actual_active)
    assert diagnostics["min_force"] >= 0
    assert diagnostics["normalized_kkt"] <= float(np.float32(max(model.opt.tolerance, 2e-5)))
    assert row.contact_time == row.input_time
    assert row.contact_phase == "tyx-euler-input-constraint-evaluation"


def _run_case(mode, case, tmp_path):
    mj = _deps()
    model = mj.MjModel.from_xml_string(case.xml)
    packet_path = tmp_path / "normal-trajectory.jsonl"
    _save(
        packet_path,
        dict(
            schema="mujoco-tyx-t3-cases-v1",
            case=case.name,
            source_xml=case.xml,
            mode=mode,
            tolerances=TOL,
            force_tolerance=FORCE_TOL,
            q_atol=Q_ATOL,
            v_atol=V_ATOL,
            native_version=mj.__version__,
            steps=case.steps,
        ),
    )
    counts, receipts = [], []
    with MuJoCoTyxRuntime(
        _compile(case.xml), mode=mode, world_keys=(19,), initial_states={19: _initial(model, case)}
    ) as runtime:
        assert runtime.capabilities["domain"] == "tree-primitive-coupled-normal-v1"
        assert runtime.capabilities["numerical_solver"] == "normal-dual-active-set-v1"
        start = runtime.snapshot()[19]
        continuous = mj.MjData(model)
        continuous.qpos[:], continuous.qvel[:] = start.q, start.v
        selected = {0, case.steps // 2, case.steps - 1}
        for tick in range(case.steps):
            before = runtime.snapshot()[19]
            packet = _packet(model, case, tick)
            stored = _stored(packet)
            oracle = _native(mj, model, before.q, before.v, stored, before.time)
            _set(continuous, stored)
            mj.mj_step(model, continuous)
            runtime.set_controls({19: packet})
            assert runtime.step(H) == {19: True}
            state = runtime.snapshot()[19]
            row = state.last_step
            detail = dict(step=tick + 1, input=before, control=stored, oracle=oracle, actual=row)
            if tick in selected:
                _save(packet_path, detail)
            try:
                _assert_receipt(row, oracle, model)
                np.testing.assert_allclose(state.q, continuous.qpos, rtol=0, atol=Q_ATOL)
                np.testing.assert_allclose(state.v, continuous.qvel, rtol=0, atol=V_ATOL)
                assert state.time == pytest.approx(continuous.time, abs=2e-6)
                if tick == 0 and case.initial_contacts is not None:
                    assert len(row.contacts) == case.initial_contacts
            except BaseException:
                if tick not in selected:
                    _save(packet_path, detail)
                raise
            _save(
                packet_path,
                dict(
                    step=tick + 1,
                    q=state.q,
                    v=state.v,
                    control=stored,
                    contacts=len(row.contacts),
                    normal_force=sum(c.normal_force for c in row.contacts),
                ),
            )
            counts.append(len(row.contacts))
            if tick in selected:
                receipts.append(row)
        assert runtime.counters["completed_kernels"] >= case.steps
    return model, counts, receipts


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", GEOMETRY_CASES, ids=lambda c: c.name)
def test_six_native_primitive_families(mode, case, tmp_path):
    _run_case(mode, case, tmp_path)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_coupled_contacts_and_continuous_native_trajectory(mode, case, tmp_path):
    model, counts, receipts = _run_case(mode, case, tmp_path)
    if case.name == "finite-withdraw":
        assert counts[0] == 1 and counts[-1] == 0
    if case.name == "tree29-last-contact":
        assert (model.nq, model.nv) == (36, 35)
        assert model.joint("j14").dofadr[0] == 34
        last = receipts[0]
        assert any(abs(c.jacobian[-1]) > 0.01 for c in last.contacts)
        assert np.max(np.abs(last.dynamics.mass_matrix[-1, :-1])) > 1e-4
    if case.name == "sibling-moving":
        c = receipts[0].contacts[0]
        assert all(body > 0 for body in c.body_ids)
        assert c.force_on_geometry1_world[0] < 0 < c.force_on_geometry2_world[0]


@pytest.mark.parametrize("mode", MODES)
def test_two_normals_discriminate_independent_single_contact_solves(mode, tmp_path):
    mj = _deps()
    case = CASES[0]
    model = mj.MjModel.from_xml_string(case.xml)
    with MuJoCoTyxRuntime(_compile(case.xml), mode=mode) as runtime:
        before = runtime.snapshot()[0]
        packet = _packet(model, case)
        oracle = _native(mj, model, before.q, before.v, packet, before.time)
        assert len(oracle["contacts"]) == 2
        r = oracle["contacts"][0]["regularization"]
        np.testing.assert_allclose(oracle["dual_matrix"], [[1 + r, 1], [1, 1 + r]], atol=1e-10)
        beta = -oracle["linear"][0]
        assert beta > 0
        exact = beta / (2 + r)
        independent = beta / (1 + r)
        assert abs(independent - exact) > 0.05 * exact
        runtime.set_controls({0: packet})
        runtime.step(H)
        row = runtime.snapshot()[0].last_step
        _assert_receipt(row, oracle, model)
        np.testing.assert_allclose(
            [c.normal_force for c in row.contacts], [exact, exact], **FORCE_TOL
        )
        _save(
            tmp_path / "coupled-counterexample.jsonl",
            dict(
                source_xml=case.xml,
                beta=beta,
                regularization=r,
                coupled=exact,
                wrong_independent=independent,
                oracle=oracle,
                actual=row,
            ),
        )


def _many_contacts(kind, count):
    shapes = []
    for i in range(count):
        if kind == "box":
            spec = 'type="box" size=".08 .06 .1"'
        elif kind == "capsule":
            spec = 'type="capsule" size=".1 .12" quat=".7071067811865476 0 .7071067811865476 0"'
        else:
            spec = 'type="sphere" size=".1"'
        shapes.append(f'<geom name="shape{i}" {spec} pos="{0.4 * i} 0 0"/>')
    return _xml(
        f"""<body name="root" pos="0 0 .099">
      <joint name="vertical" type="slide" axis="0 0 1"/>
      <inertial mass="2" pos="0 0 0" diaginertia=".1 .12 .14"/>
      {"".join(shapes)}</body>""",
        static=FLOOR,
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "kind,count,contacts", [("sphere", 16, 16), ("box", 4, 16), ("capsule", 16, 32)]
)
def test_exact_layout_and_runtime_contact_capacity(mode, kind, count, contacts, tmp_path):
    mj = _deps()
    xml = _many_contacts(kind, count)
    model = mj.MjModel.from_xml_string(xml)
    with MuJoCoTyxRuntime(
        _compile(xml), mode=mode, contact_capacity=contacts, constraint_capacity=contacts
    ) as runtime:
        before = runtime.snapshot()[0]
        packet = _packet(model, Case(kind, xml, 1))
        oracle = _native(mj, model, before.q, before.v, packet, before.time)
        assert len(oracle["contacts"]) == contacts
        runtime.set_controls({0: packet})
        runtime.step(H)
        actual = runtime.snapshot()[0].last_step
        _assert_receipt(actual, oracle, model)
        assert len(actual.contacts) == contacts
        _save(
            tmp_path / "capacity-exact.jsonl",
            dict(xml=xml, contacts=contacts, oracle=oracle, actual=actual),
        )
    for field in ("contact_capacity", "constraint_capacity"):
        limits = dict(contact_capacity=contacts, constraint_capacity=contacts)
        limits[field] -= 1
        with MuJoCoTyxRuntime(_compile(xml), mode=mode, world_keys=(3, 19), **limits) as runtime:
            before = runtime.snapshot()
            with pytest.raises(PhysicsStepFailure) as caught:
                runtime.step(H)
            assert caught.value.accepted_worlds == {3: False, 19: False}
            _same(before, runtime.snapshot())
            assert all(s.last_step is None for s in runtime.snapshot().values())


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "height,active", [(".140625", False), (".1328125", False), (".12890625", True)]
)
def test_detection_margin_and_include_gap_boundaries(mode, height, active, tmp_path):
    mj = _deps()
    xml = _free(
        '<geom name="shape" type="sphere" size=".125" margin=".015625" gap=".0078125"/>',
        f"0 0 {height}",
        gravity="0 0 0",
    )
    case = Case("exact-boundary", xml, 1, initial_contacts=1)
    model = mj.MjModel.from_xml_string(xml)
    with MuJoCoTyxRuntime(
        _compile(xml), mode=mode, contact_capacity=1, constraint_capacity=int(active)
    ) as runtime:
        before = runtime.snapshot()[0]
        packet = _packet(model, case)
        oracle = _native(mj, model, before.q, before.v, packet, before.time)
        runtime.set_controls({0: packet})
        runtime.step(H)
        row = runtime.snapshot()[0].last_step
        _assert_receipt(row, oracle, model)
        assert len(row.contacts) == 1 and row.contacts[0].active is active
        if not active:
            assert row.contacts[0].normal_force == 0
        _save(tmp_path / "exact-boundary.jsonl", dict(xml=xml, oracle=oracle, actual=row))
    with MuJoCoTyxRuntime(_compile(xml), mode=mode, contact_capacity=0) as runtime:
        before = runtime.snapshot()
        with pytest.raises(PhysicsStepFailure):
            runtime.step(H)
        _same(before, runtime.snapshot())


@pytest.mark.parametrize("kind,count", [("sphere", 17), ("box", 5)])
def test_pair_and_static_feature_slot_overflow_rejected(kind, count):
    _deps()
    with pytest.raises(MuJoCoTyxUnsupported):
        MuJoCoTyxRuntime(_compile(_many_contacts(kind, count)), mode="interpret")


@pytest.mark.parametrize("count,accepted", [(128, True), (129, False)])
def test_geometry_capacity_keeps_all_masked_native_ids(count, accepted):
    _deps()
    xml = CASES[0].xml
    # Existing floor and two balls: three geoms. Masked world markers are real IDs.
    markers = "".join(
        f'<geom name="marker{i}" type="sphere" size=".01" '
        f'pos="{10 + i} 0 0" contype="0" conaffinity="0"/>'
        for i in range(count - 3)
    )
    xml = xml.replace("<worldbody>", "<worldbody>" + markers)
    compiled = _compile(xml)
    assert compiled.mj_model.ngeom == count
    if not accepted:
        with pytest.raises(MuJoCoTyxUnsupported):
            MuJoCoTyxRuntime(compiled, mode="interpret")
    else:
        with MuJoCoTyxRuntime(compiled, mode="interpret") as runtime:
            runtime.step(H)
            actual = runtime.snapshot()[0].last_step
            assert len(actual.contacts) == 2
            assert min(g for c in actual.contacts for g in c.geometry_ids) == count - 3


@pytest.mark.parametrize("mode", MODES)
def test_native_or_mask_and_no_contact_route(mode, tmp_path):
    mj = _deps()
    shape = '<geom name="shape" type="sphere" size=".1" contype="0" conaffinity="1"/>'
    floor = FLOOR.replace('size="2 2 .1"', 'size="2 2 .1" contype="1" conaffinity="0"')
    positive = Case("mask-or", _free(shape, "0 0 .099", static=floor), 1, initial_contacts=1)
    _run_case(mode, positive, tmp_path)
    blocked = positive.xml.replace('conaffinity="1"', 'conaffinity="0"').replace(
        'name="shape" type="sphere"', 'name="shape" type="sphere" condim="3"'
    )
    model = mj.MjModel.from_xml_string(blocked)
    with MuJoCoTyxRuntime(_compile(blocked), mode=mode) as runtime:
        assert runtime.capabilities["domain"] == "free-root-hinge-slide-no-contact-v1"
        before = runtime.snapshot()[0]
        packet = _packet(model, positive)
        native = mj.MjData(model)
        native.qpos[:], native.qvel[:] = before.q, before.v
        _set(native, packet)
        mj.mj_step(model, native)
        runtime.set_controls({0: packet})
        runtime.step(H)
        actual = runtime.snapshot()[0]
        assert not actual.last_step.contacts
        np.testing.assert_allclose(actual.q, native.qpos, rtol=0, atol=Q_ATOL)
        np.testing.assert_allclose(actual.v, native.qvel, rtol=0, atol=V_ATOL)


@pytest.mark.parametrize("mode", MODES)
def test_same_body_welded_and_parent_geom_exclusions(mode, tmp_path):
    # Three overlapping round geoms in one weld group and its child must not
    # become extra contacts; floor remains the genuine dynamic interaction.
    shapes = """<geom name="shape" type="sphere" size=".1"/>
      <geom name="same-body" type="sphere" size=".1" pos=".02 0 0"/>
      <body name="weld"><inertial mass=".2" pos="0 0 0" diaginertia=".01 .01 .01"/>
        <geom name="welded-shape" type="sphere" size=".1" pos="-.02 0 0"/></body>
      <body name="child" pos="0 0 .02"><joint name="child-slide" type="slide" axis="0 0 1"/>
        <inertial mass=".3" pos="0 0 0" diaginertia=".01 .012 .014"/>
        <geom name="child-shape" type="sphere" size=".1"/></body>"""
    case = Case("native-parent-filter", _free(shapes, "0 0 .099"), 1, initial_contacts=3)
    _, _, rows = _run_case(mode, case, tmp_path)
    assert all(0 in c.body_ids for c in rows[0].contacts)


@pytest.mark.parametrize("mode", MODES)
def test_sparse_batch_reset_clone_then_real_contact_step(mode, tmp_path):
    mj = _deps()
    case = CASES[1]
    model = mj.MjModel.from_xml_string(case.xml)
    keys = (3, 19, 91, 203)
    states = {key: _initial(model, case, slot) for slot, key in enumerate(keys)}
    compiled = _compile(case.xml)
    with MuJoCoTyxRuntime(compiled, mode=mode, world_keys=keys, initial_states=states) as runtime:
        for tick in range(3):
            runtime.set_controls(
                {key: _packet(model, case, tick, slot) for slot, key in enumerate(keys)}
            )
            runtime.step(H)
        with runtime.clone() as cloned:
            before = runtime.snapshot()
            assert len({len(s.last_step.contacts) for s in before.values()}) > 1
            runtime.reset((keys[0],))
            for key in keys[1:]:
                _same({key: before[key]}, {key: runtime.snapshot()[key]})
            with MuJoCoTyxRuntime(
                compiled,
                mode=mode,
                world_keys=(keys[0],),
                initial_states={keys[0]: states[keys[0]]},
            ) as fresh:
                packets = {key: _packet(model, case, 3, slot) for slot, key in enumerate(keys)}
                runtime.set_controls(packets)
                cloned.set_controls(packets)
                fresh.set_controls({keys[0]: packets[keys[0]]})
                runtime.step(H)
                cloned.step(H)
                fresh.step(H)
                actual, expected = runtime.snapshot(), cloned.snapshot()
                _same({keys[0]: actual[keys[0]]}, fresh.snapshot())
                for key in keys[1:]:
                    _same({key: actual[key]}, {key: expected[key]})
                _save(
                    tmp_path / "batch-reset-next.jsonl",
                    dict(before=before, actual=actual, clone=expected, fresh=fresh.snapshot()),
                )


@pytest.mark.parametrize("mode", MODES)
def test_late_readback_failure_all_world_rollback_and_retry(mode, monkeypatch, tmp_path):
    mj = _deps()
    case = CASES[2]
    model = mj.MjModel.from_xml_string(case.xml)
    keys = (7, 29, 113, 211)
    initial = {key: _initial(model, case, slot) for slot, key in enumerate(keys)}
    with MuJoCoTyxRuntime(
        _compile(case.xml), mode=mode, world_keys=keys, initial_states=initial
    ) as runtime:
        runtime.set_controls({key: _packet(model, case, 0, slot) for slot, key in enumerate(keys)})
        runtime.step(H)
        with runtime.clone() as clone:
            before, count = runtime.snapshot(), runtime.counters["completed_kernels"]
            original = runtime._device.download

            def fail_after_download():
                original()
                raise RuntimeError("T3 injected after actual device readback")

            monkeypatch.setattr(runtime._device, "download", fail_after_download)
            with pytest.raises(PhysicsStepFailure) as caught:
                runtime.step(H)
            assert caught.value.accepted_worlds == dict.fromkeys(keys, False)
            assert caught.value.acceptance_known and not caught.value.physics_accepted
            assert runtime.counters["completed_kernels"] > count
            _same(before, runtime.snapshot())
            _save(
                tmp_path / "late-failure.jsonl",
                dict(before=before, after=runtime.snapshot(), mask=caught.value.accepted_worlds),
            )
            monkeypatch.setattr(runtime._device, "download", original)
            runtime.step(H)
            clone.step(H)
            _same(runtime.snapshot(), clone.snapshot())
            _save(
                tmp_path / "late-failure.jsonl",
                dict(retry=runtime.snapshot(), clone=clone.snapshot()),
            )


@pytest.mark.parametrize("mode", MODES)
def test_contact_step_has_no_cpu_warp_host_or_numpy_solver(mode, monkeypatch, tmp_path):
    import importlib

    mj = _deps()
    case = CASES[2]
    model = mj.MjModel.from_xml_string(case.xml)
    modules = [
        (importlib.import_module(name), functions)
        for name, functions in (
            ("phymeo.dynamics.robot_dynamics", ("forward_dynamics", "mass_matrix", "bias_forces")),
            ("phymeo.dynamics.robot_kinematics", ("forward_kinematics", "point_jacobians")),
            ("phymeo.solvers.floating_tree", ("step_floating_robot", "floating_dynamics_terms")),
        )
    ]
    with MuJoCoTyxRuntime(
        _compile(case.xml), mode=mode, initial_states={0: _initial(model, case)}
    ) as runtime:
        before = runtime.snapshot()[0]
        packet = _packet(model, case)
        oracle = _native(mj, model, before.q, before.v, _stored(packet), before.time)
        runtime.set_controls({0: packet})
        count = runtime.counters["completed_kernels"]

        def forbidden(*args, **kwargs):
            raise AssertionError("T3 step must perform genuine TYX contact dynamics")

        with monkeypatch.context() as patch:
            for name in (
                "mj_step",
                "mj_forward",
                "mj_fullM",
                "mj_rne",
                "mj_jac",
                "mj_collision",
                "mj_fwdConstraint",
            ):
                patch.setattr(mj, name, forbidden)
            for name in ("solve", "inv", "cholesky", "lstsq"):
                patch.setattr(np.linalg, name, forbidden)
            for module, functions in modules:
                for name in functions:
                    patch.setattr(module, name, forbidden)
            patch.setitem(sys.modules, "warp", None)
            patch.setitem(sys.modules, "mujoco_warp", None)
            assert runtime.step(H) == {0: True}
        actual = runtime.snapshot()[0].last_step
        assert runtime.counters["completed_kernels"] > count
        assert runtime.capabilities["gpu"] is (mode == "cuda")
        _assert_receipt(actual, oracle, model)
        _save(tmp_path / "no-fallback.jsonl", dict(oracle=oracle, actual=actual))


@pytest.mark.parametrize("mode", MODES)
def test_normal_control_overflow_is_atomic(mode):
    mj = _deps()
    case = CASES[2]
    model = mj.MjModel.from_xml_string(case.xml)
    with MuJoCoTyxRuntime(_compile(case.xml), mode=mode, world_keys=(5, 113)) as runtime:
        packet = _packet(model, case)
        runtime.set_controls({5: packet, 113: packet})
        before = runtime.snapshot()
        bad = np.zeros(model.nv)
        bad[-1] = np.finfo(np.float64).max
        with pytest.raises(ValueError):
            runtime.set_controls({5: packet, 113: MuJoCoControl([], qfrc_applied=bad)})
        _same(before, runtime.snapshot())


@pytest.mark.parametrize("mode", MODES)
def test_joint_solve_budget_exhaustion_rejects_whole_batch(mode):
    _deps()
    # Two equal loaded normals require two active faces from zero. One solve
    # cannot produce their joint optimum and must not publish a partial answer.
    xml = CASES[0].xml.replace('iterations="128"', 'iterations="1"')
    with MuJoCoTyxRuntime(_compile(xml), mode=mode, world_keys=(5, 113)) as runtime:
        before = runtime.snapshot()
        with pytest.raises(PhysicsStepFailure) as caught:
            runtime.step(H)
        assert caught.value.accepted_worlds == {5: False, 113: False}
        _same(before, runtime.snapshot())
        assert runtime.counters["completed_kernels"] == 5


@pytest.mark.parametrize(
    "mutation",
    [
        "explicit-pair",
        "exclude",
        "condim3",
        "unsupported-shape",
        "nondefault-flags",
        "iterations-zero",
        "iterations-large",
        "negative-margin",
    ],
)
def test_unsupported_contact_semantics_are_rejected_before_execution(mutation):
    _deps()
    xml = CASES[0].xml
    if mutation == "explicit-pair":
        xml = xml.replace(
            "</mujoco>", '<contact><pair geom1="floor" geom2="ball0"/></contact></mujoco>'
        )
    elif mutation == "exclude":
        xml = xml.replace(
            "</mujoco>", '<contact><exclude body1="world" body2="root"/></contact></mujoco>'
        )
    elif mutation == "condim3":
        xml = xml.replace('condim="1"', 'condim="3"')
    elif mutation == "unsupported-shape":
        xml = xml.replace('type="sphere" size=".1"', 'type="ellipsoid" size=".1 .12 .13"')
    elif mutation == "nondefault-flags":
        xml = xml.replace(
            'jacobian="dense"/>', 'jacobian="dense"><flag filterparent="disable"/></option>'
        )
    elif mutation == "negative-margin":
        xml = GEOMETRY_CASES[8].xml.replace("<geom name=", '<geom margin="-.2" name=')
    elif mutation == "iterations-zero":
        xml = xml.replace('iterations="128"', 'iterations="0"')
    else:
        xml = xml.replace('iterations="128"', 'iterations="513"')
    with pytest.raises(MuJoCoTyxUnsupported):
        MuJoCoTyxRuntime(_compile(xml), mode="interpret")
