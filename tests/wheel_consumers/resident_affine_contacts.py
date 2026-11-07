"""Standalone C7 wheel consumer; authoring only, UNEXECUTED.

Only public Phymeo, NumPy and stdlib are used. Fixed-root ordinary/affine trees,
finite static convex contact, mu=0: HOST narrow/law plus actual TYX response.
Forty steps, tick-19 replay, local capacity refusal/retry; not a benchmark,
Newton equivalence test, full C7 suite, floating, friction or CCD claim.
"""

import argparse
import gzip
import hashlib
import json
import sys
import traceback
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from importlib.metadata import distribution
from itertools import product
from math import cos, sin
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname
from zipfile import ZipFile

import numpy as np

H, STEPS, REPLAY_FROM = 1 / 1024, 40, 19
ATOL, RTOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-4
ORDER = (1207, 6, 91, 503)
NAMES = {503: "ordinary", 91: "A03", 1207: "A05", 6: "free"}


def json_value(value):
    if is_dataclass(value):
        return {f.name: json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def close(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def box(lo, hi):
    return np.array(list(product(*zip(lo, hi, strict=True))), dtype=float)


def rz(angle):
    c, s = cos(angle), sin(angle)
    return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1.0)))


def declaration(name, *, flat=False):
    """Frozen physical inputs; contact replaces the selected original applied load."""
    if name == "A03":
        return dict(
            name=name,
            mass=(2, 1, 1),
            inertia=((1 / 8,) * 3,) * 3,
            parents=(-1, -1, -1),
            kinds=("prismatic",) * 3,
            offsets=((0, 0, 0), (0, 0.5, 0), (0, -0.5, 0)),
            axes=((1, 0, 0),) * 3,
            rows=((0, 1, 2, 0.125), (0, 2, -1, 0.125))
            if flat
            else ((0, 1, 2, 0.125), (1, 2, -0.5, 0.1875)),
            P=((1,), (2,), (-1,)),
            c=(0, 0.125, 0.125),
            C=((-2, 1, 0), (1, 0, 1)) if flat else ((-2, 1, 0), (0, 0.5, 1)),
            motor=(7,),
            effort=(0, 0, 0),
            initial_center=(0.125, -0.5, 0),
        )
    assert name == "A05"
    return dict(
        name=name,
        mass=(2, 1, 1),
        inertia=((0.125, 0.125, 0.25), (0.0625, 0.0625, 0.125), (0.0625, 0.0625, 0.125)),
        parents=(-1, 0, 0),
        kinds=("revolute", "prismatic", "prismatic"),
        offsets=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        axes=((0, 0, 1), (0, 1, 0), (1, 0, 0)),
        rows=((1, 2, 2, 0.125),),
        P=((1, 0), (0, 1), (0, 2)),
        c=(0, 0, 0.125),
        C=((0, -2, 1),),
        motor=(0, 0),
        effort=(0, 1, 0),
        initial_center=(0.125, 1, 0),
    )


def xml_source(data):
    root = ET.Element("robot", name="c7_" + data["name"])
    ET.SubElement(root, "link", name="base")
    for i in range(3):
        link = ET.SubElement(root, "link", name=f"link{i}")
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=str(data["mass"][i]))
        ix, iy, iz = data["inertia"][i]
        ET.SubElement(
            inertial, "inertia", ixx=str(ix), iyy=str(iy), izz=str(iz), ixy="0", ixz="0", iyz="0"
        )
        joint = ET.SubElement(root, "joint", name=f"joint{i}", type=data["kinds"][i])
        parent = data["parents"][i]
        ET.SubElement(joint, "parent", link="base" if parent < 0 else f"link{parent}")
        ET.SubElement(joint, "child", link=f"link{i}")
        ET.SubElement(joint, "origin", xyz=" ".join(map(str, data["offsets"][i])), rpy="0 0 0")
        ET.SubElement(joint, "axis", xyz=" ".join(map(str, data["axes"][i])))
        ET.SubElement(joint, "limit", lower="-8", upper="8", velocity="16", effort="16")
        for driver, follower, gain, offset in data["rows"]:
            if follower == i:
                ET.SubElement(
                    joint, "mimic", joint=f"joint{driver}", multiplier=str(gain), offset=str(offset)
                )
    return ET.tostring(root, encoding="unicode")


def make_affine(ph, directory, data):
    suffix = "-flat" if data["name"] == "A03" and data["rows"][-1][0] == 0 else ""
    path = directory / (data["name"] + suffix + ".urdf")
    path.write_text(xml_source(data), encoding="utf-8")
    asset = ph.load_mimic_urdf(path, gravity=(0, 0, 0))
    assert asset.tree.joint_names == ("joint0", "joint1", "joint2")
    k = len(data["motor"])
    np.testing.assert_array_equal(asset.matrix, data["P"])
    np.testing.assert_array_equal(asset.offset_vector, data["c"])
    world = ph.AffineRobotTree(
        asset,
        ph.RobotState(np.zeros(k), np.zeros(k)),
        kp=np.zeros(k),
        kd=np.zeros(k),
        effort_limit=np.full(k, 16.0),
    )
    command = world.inputs(
        np.zeros(k),
        np.zeros(k),
        feedforward=np.asarray(data["motor"]),
        loads=ph.RobotInputs(np.asarray(data["effort"], float), np.zeros((3, 3)), np.zeros((3, 3))),
    )
    return asset, world, command, path


def make_ordinary(ph):
    model = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.array([[0, 0, 3 / 32]]),
        np.array([[0, 0, 0, 1.0]]),
        np.array([[0, 0, 1.0]]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.full((1, 3), 1 / 8),
        np.zeros(3),
    )
    program = ph.DrivenRobotProgram(
        model, kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.array([8.0])
    )
    world = ph.FixedRobotTree(
        program, program.initial_state(ph.RobotState(np.zeros(1), np.zeros(1)))
    )
    command = program.inputs(
        np.zeros(1), np.zeros(1), feedforward=np.zeros(1), loads=ph.RobotInputs.zeros(1)
    )
    return world, command


def make_geometry(ph, batch, assets, declarations):
    shapes, bindings, laws, sources = [], [], {}, {}
    for world in (503, 91, 1207):
        if world == 503:
            group, local = 0, box([-1 / 8] * 3, [1 / 8] * 3)
            static = box([-1 / 2, -1 / 2, -1 / 4], [1 / 2, 1 / 2, 0])
            kind, ids, radius = "hull", tuple(range(101, 109)), 0
        else:
            group = assets[world].tree.sites["link2"].link_index
            assert group == 2
            local, kind, ids, radius = np.zeros((1, 3)), "sphere", (71,), 1 / 8
            cx, cy, _ = declarations[world]["initial_center"]
            face = cx - 1 / 8 + 3 / 64
            static = box((face - 1 / 4, cy - 1 / 4, -1 / 4), (face, cy + 1 / 4, 1 / 4))
        first, second = ph.CollisionKey(world, 100 + group, 11), ph.CollisionKey(world, 900, 1)
        a, b = (
            ph.CollisionShape(first, kind, ids, radius),
            ph.CollisionShape(second, "hull", tuple(range(201, 209))),
        )
        cls = ph.ResidentTreeBinding if world == 503 else ph.ResidentAffineTreeBinding
        bindings.extend(
            (
                cls(first, world, group, local, topology=batch.snapshot([world])[world].topology),
                ph.ResidentStaticBinding(second, ph.CollisionMotion(static)),
            )
        )
        shapes.extend((a, b))
        laws[(first, second)] = ph.RobotContactMaterial(64, 0, 0)
        sources[world] = {first: (a, group, local), second: (b, None, static)}
    return ph.CollisionModel(tuple(shapes)), tuple(bindings), laws, sources


def oracle(name, s, u):
    """Independent energy derivatives and closed-form constrained solve.

    No product M/J, diagnostic acceleration or contact force enters this function.
    """
    s, u = np.asarray(s, float), np.asarray(u, float)
    if name == "ordinary":
        normal = 64 * max(0.0, 1 / 32 - s[0])
        return dict(a=np.array([normal / 2]), normal=normal)
    d = declaration(name)
    P, c, C = np.array(d["P"], float), np.array(d["c"]), np.array(d["C"], float)
    q, v = P @ s + c, P @ u
    jv, jw = np.zeros((3, 3, 3)), np.zeros((3, 3, 3))
    if name == "A03":
        center = np.column_stack((q, [0, 0.5, -0.5], np.zeros(3)))
        rotation = np.repeat(np.eye(3)[None], 3, axis=0)
        for i in range(3):
            jv[i, 0, i] = 1
        normal = 64 * max(0.0, 3 / 64 + s[0])
        M, b = np.diag((2.0, 1.0, 1.0)), np.zeros(3)
        e = np.array((0, 0, normal))
        a = np.array([(7 - normal) / 7])
        full_a = P @ a
        reaction = M @ full_a - np.array((7, 0, 0)) - e
        multipliers = np.array((reaction[1] - reaction[2] / 2, reaction[2]))
    else:
        theta, slide = s
        omega, speed = u
        R = rz(theta)
        r1, r2 = R @ [1, slide, 0], R @ [q[2], 1, 0]
        center = np.array((np.zeros(3), r1, r2))
        rotation = np.repeat(R[None], 3, axis=0)
        jv[1, :, 0], jv[2, :, 0] = np.cross([0, 0, 1], r1), np.cross([0, 0, 1], r2)
        jv[1, :, 1], jv[2, :, 2] = R[:, 1], R[:, 0]
        jw[:, 2, 0] = 1
        face = 3 / 64
        assert r2[0] > face and abs(r2[1] - 1) < 1 / 4 and abs(r2[2]) < 1 / 4
        normal = 64 * max(0.0, 1 / 8 + face - r2[0])
        A = 2.5 + slide * slide + q[2] * q[2]
        M = np.array(((A, 1, -1), (1, 1, 0), (-1, 0, 1)))
        b = np.array(
            (
                2 * omega * (slide * speed + q[2] * 2 * speed),
                -slide * omega * omega,
                -q[2] * omega * omega,
            )
        )
        e = np.array((-r2[1] * normal, 1, cos(theta) * normal))
        rhs = P.T @ (e - b)
        det = 5 * A - 1
        a = np.array(((5 * rhs[0] + rhs[1]) / det, (rhs[0] + A * rhs[1]) / det))
        full_a = P @ a
        multiplier = -a[0] + 2 * a[1] - q[2] * omega * omega - cos(theta) * normal
        multipliers = np.array([multiplier])
        reaction = np.array((0, -2 * multiplier, multiplier))
    return dict(
        q=q,
        v=v,
        P=P,
        c=c,
        C=C,
        center=center,
        rotation=rotation,
        jv=jv,
        jw=jw,
        M=M,
        b=b,
        e=e,
        Mr=P.T @ M @ P,
        br=P.T @ b,
        er=P.T @ e,
        a=a,
        full_a=full_a,
        reaction=reaction,
        multipliers=multipliers,
        normal=normal,
        input_external=np.asarray(d["effort"], float),
        contact_generalized=e - np.asarray(d["effort"], float),
    )


def audit_sites(receipt, diagnostic, sources):
    full_velocity = np.asarray(
        getattr(receipt.entry_state, "full_velocity", receipt.entry_state.velocity), dtype=float
    )
    wrench = np.zeros_like(receipt.law_wrench_fp64, dtype=float)
    static, total_work, total_material_work = {}, 0.0, 0.0
    for item in receipt.sites:
        witness = item.geometry.witness
        point = np.asarray(item.geometry.load_point, float)
        np.testing.assert_array_equal(item.first_force_n, -item.second_force_n)
        work = material_work = 0.0
        for side, key in zip(("first", "second"), item.pair, strict=True):
            shape, group, local = sources[key]
            weights = getattr(witness, side + "_weights")
            by_id = dict(zip(shape.vertex_ids, local, strict=True))
            material_local = sum(weight * by_id[node] for node, weight in weights)
            skin = getattr(witness, side + "_offset")
            surface = np.asarray(getattr(witness, side + "_point"), float)
            if group is None:
                rebuilt = material_local
                reference, velocity = local.mean(axis=0), np.zeros(3)
                displacement = application = np.zeros(3)
            else:
                entry, end = receipt.entry_geometry, receipt.accepted_geometry
                p0, p1 = entry.link_position_m[group], end.link_position_m[group]
                r0, r1 = entry.link_rotation[group], end.link_rotation[group]
                rebuilt = p0 + r0 @ material_local
                reference = entry.com_position_m[group]
                omega = diagnostic.angular_jacobian[group] @ full_velocity
                velocity = diagnostic.linear_jacobian[group] @ full_velocity + np.cross(
                    omega, point - reference
                )
                displacement = p1 + r1 @ np.linalg.solve(r0, surface - p0) - surface
                application = p1 + r1 @ np.linalg.solve(r0, point - p0) - point
            rebuilt = rebuilt + (np.zeros(3) if skin is None else skin)
            close(rebuilt, surface, atol=1e-10, rtol=0)
            close(getattr(item, side + "_reference_point_m"), reference, atol=1e-10, rtol=0)
            close(getattr(item, side + "_velocity_m_s"), velocity)
            close(
                getattr(item, side + "_material_displacement_m"), displacement, atol=1e-10, rtol=0
            )
            close(
                getattr(item, side + "_application_displacement_m"), application, atol=1e-10, rtol=0
            )
            force = np.asarray(getattr(item, side + "_force_n"))
            work += float(force @ application)
            material_work += float(force @ displacement)
            contribution = np.r_[force, np.cross(point - reference, force)]
            if group is None:
                static.setdefault(key, np.zeros(6))
                static[key] += contribution
            else:
                wrench[group] += contribution
        close(item.application_work_j, work, atol=1e-12, rtol=1e-10)
        close(item.material_work_j, material_work, atol=1e-12, rtol=1e-10)
        total_work += work
        total_material_work += material_work
    close(receipt.law_wrench_fp64, wrench)
    np.testing.assert_array_equal(receipt.uploaded_wrench_fp32, np.asarray(wrench, np.float32))
    projected = np.einsum("bij,bi->j", diagnostic.linear_jacobian, wrench[:, :3]) + np.einsum(
        "bij,bi->j", diagnostic.angular_jacobian, wrench[:, 3:]
    )
    close(receipt.contact_generalized, projected)
    for key, reaction in receipt.static_reactions.items():
        close(reaction.reference_point_m, sources[key][2].mean(axis=0), atol=1e-12, rtol=0)
        close(reaction.force_n, static.get(key, np.zeros(6))[:3])
        close(reaction.torque_nm, static.get(key, np.zeros(6))[3:])
    return dict(application_work_j=total_work, material_work_j=total_material_work)


def audit_world(world, entry, receipt, state, diagnostic, sources):
    name = NAMES[world]
    truth = oracle(name, entry.position, entry.velocity)
    close(state.velocity, np.asarray(entry.velocity) + H * truth["a"])
    close(state.position, np.asarray(entry.position) + H * np.asarray(state.velocity))
    assert receipt.entry_state == entry and receipt.accepted_state == state
    assert (receipt.entry_tick, receipt.exit_tick, receipt.dt_s) == (entry.tick, entry.tick + 1, H)
    assert state.control == entry.control
    if name == "ordinary":
        close(receipt.contact_generalized, [truth["normal"]])
        close(receipt.law_wrench_fp64, [[0, 0, truth["normal"], 0, 0, 0]])
    else:
        meta = receipt.diagnostics
        assert meta["coordinate_kind"] == "affine"
        assert meta["full_dof_count"] == 3 and meta["independent_dof_count"] == len(entry.position)
        assert meta["graph_token"]
        close(meta["projection_matrix"], truth["P"])
        close(meta["offset_vector"], truth["c"])
        close(meta["constraint_rows"], truth["C"])
        assert json_value(meta["affine_diagnostic"]) == json_value(diagnostic)
        close(entry.full_position, truth["q"])
        close(entry.full_velocity, truth["v"])
        close(state.full_position, truth["P"] @ np.asarray(state.position) + truth["c"], atol=2e-5)
        close(state.full_velocity, truth["P"] @ np.asarray(state.velocity), atol=2e-5)
        close(truth["C"] @ np.asarray(state.full_velocity), np.zeros(len(truth["C"])), atol=2e-5)
        close(receipt.entry_geometry.com_position_m, truth["center"])
        close(receipt.entry_geometry.link_rotation, truth["rotation"])
        for field, expected in (
            ("mass", "M"),
            ("bias", "b"),
            ("external", "e"),
            ("reduced_mass", "Mr"),
            ("reduced_bias", "br"),
            ("reduced_external", "er"),
            ("acceleration", "a"),
            ("full_acceleration", "full_a"),
            ("constraint_effort_full", "reaction"),
            ("row_multipliers", "multipliers"),
            ("linear_jacobian", "jv"),
            ("angular_jacobian", "jw"),
        ):
            close(getattr(diagnostic, field), truth[expected])
        close(diagnostic.row_powers_w, np.zeros(len(truth["C"])), atol=2e-5)
        close(diagnostic.constraint_power_w, 0, atol=2e-5)
        close(receipt.input_external, truth["input_external"])
        close(receipt.contact_generalized, truth["contact_generalized"])
        full_power = float(receipt.contact_generalized @ np.asarray(entry.full_velocity))
        reduced_power = float(
            (truth["P"].T @ receipt.contact_generalized) @ np.asarray(entry.velocity)
        )
        close(full_power, reduced_power, atol=2e-5)
    work = audit_sites(receipt, diagnostic, sources)
    meta = receipt.diagnostics
    if name == "ordinary":
        entry_ke, accepted_ke = entry.velocity[0] ** 2, state.velocity[0] ** 2
    else:
        next_truth = oracle(name, state.position, state.velocity)
        entry_ke = float(truth["v"] @ truth["M"] @ truth["v"] / 2)
        accepted_ke = float(next_truth["v"] @ next_truth["M"] @ next_truth["v"] / 2)
    close(meta["entry_kinetic_energy_j"], entry_ke)
    close(meta["accepted_kinetic_energy_j"], accepted_ke)
    close(meta["kinetic_energy_change_j"], accepted_ke - entry_ke)
    close(meta["application_work_j"], work["application_work_j"], atol=1e-12, rtol=1e-10)
    close(meta["material_work_j"], work["material_work_j"], atol=1e-12, rtol=1e-10)
    return dict(independent=truth, actual_work=work)


def run_window(ph, mode, output, record):
    ordinary, ordinary_control = make_ordinary(ph)
    assets = {}
    worlds = {503: ordinary, 6: ph.FreeSphere(2, 1 / 8, position=(0, 0, 2), gravity=(0, 0, -8))}
    commands, declarations = {503: ordinary_control, 6: (0, 0, 0)}, {}
    for key in (91, 1207):
        data = declaration(NAMES[key])
        assets[key], worlds[key], commands[key], source = make_affine(ph, output, data)
        declarations[key] = data
        record.setdefault("assets", []).append(
            dict(
                path=str(source),
                content=source.read_text(encoding="utf-8"),
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
        )
    record.update(
        declarations=declarations,
        inputs=commands,
        world_order=ORDER,
        trace=[],
        replay=[],
        checks={},
    )
    initial_chain, initial_rp = oracle("A03", [0], [0]), oracle("A05", [0, 0], [0, 0])
    close(initial_chain["a"], [4 / 7], atol=2e-13, rtol=0)
    close(initial_chain["multipliers"], [41 / 14, -25 / 7], atol=2e-13, rtol=0)
    close(initial_rp["a"], [-512 / 741, 935 / 741], atol=2e-13, rtol=0)
    close(initial_rp["multipliers"], [53 / 247], atol=2e-13, rtol=0)
    record["independent_initial_checks"] = dict(chain=initial_chain, coupled_rp=initial_rp)
    expected = {
        503: (np.zeros(1), np.zeros(1)),
        91: (np.zeros(1), np.zeros(1)),
        1207: (np.zeros(2), np.zeros(2)),
    }
    batch = ph.TyxWorldBatch(
        [(key, worlds[key]) for key in ORDER], mode, cache_root=output / "cache"
    )
    plan = None
    try:
        batch.set_controls(commands)
        model, bindings, laws, sources = make_geometry(ph, batch, assets, declarations)
        record.update(collision_model=model, bindings=bindings, laws=laws, initial=batch.snapshot())
        plan = batch.prepare_contacts(model, bindings, laws)
        for index in range(STEPS):
            entry = batch.snapshot()
            row = dict(entry=entry, checks_complete=False)
            record["trace"].append(row)
            predicted = {}
            for key, (q, v) in expected.items():
                t = oracle(NAMES[key], q, v)
                next_v = v + H * t["a"]
                predicted[key] = (q + H * next_v, next_v)
            row["acceptance"] = batch.step_contacts(plan, H)
            row.update(
                state=batch.snapshot(),
                geometry=batch.tree_kinematics(),
                diagnostic=batch.tree_diagnostics(),
                receipts=dict(plan.last_accepted),
                expected=predicted,
                stats=batch.stats(),
            )
            assert row["acceptance"] == dict.fromkeys(ORDER, True)
            assert set(row["receipts"]) == {503, 91, 1207}
            audits = {}
            for key in ORDER:
                state = row["state"][key]
                assert state.tick == index + 1
                if key == 6:
                    n = index + 1
                    close(state.position, [0, 0, 2 - 4 * H * H * n * (n + 1)], atol=WINDOW_ATOL)
                    close(state.velocity, [0, 0, -8 * H * n], atol=WINDOW_ATOL)
                else:
                    audits[key] = audit_world(
                        key,
                        entry[key],
                        row["receipts"][key],
                        state,
                        row["diagnostic"][key],
                        sources[key],
                    )
                    close(state.position, predicted[key][0], atol=WINDOW_ATOL)
                    close(state.velocity, predicted[key][1], atol=WINDOW_ATOL)
            row.update(audits=audits, checks_complete=True)
            expected = predicted
            if index + 1 == REPLAY_FROM:
                record["checkpoint_19"] = batch.snapshot()
        record["checks"]["continuous_40"] = True
        # Equal P/c do not authorize exchanging the original constraint graph.
        _, flat_world, _, source = make_affine(ph, output, declaration("A03", flat=True))
        record["graph_holdout_asset"] = dict(
            path=str(source), content=source.read_text(encoding="utf-8")
        )
        with ph.TyxWorldBatch(
            [(91, flat_world)], mode, cache_root=output / "flat-cache"
        ) as foreign:
            before = foreign.snapshot()
            try:
                foreign.restore({91: record["checkpoint_19"][91]})
            except (ValueError, TypeError) as error:
                record["graph_rejection"] = dict(type=type(error).__name__, message=str(error))
            else:
                raise AssertionError("flattened graph accepted a chain checkpoint")
            assert foreign.snapshot() == before
        record["checks"]["graph_identity_rejected"] = True
        epoch = batch.stats()["state_epoch"]
        batch.restore(record["checkpoint_19"])
        assert (
            batch.snapshot() == record["checkpoint_19"]
            and batch.stats()["state_epoch"] == epoch + 1
        )
        for index in range(REPLAY_FROM, STEPS):
            row = dict(checks_complete=False)
            record["replay"].append(row)
            row["acceptance"] = batch.step_contacts(plan, H)
            row.update(
                state=batch.snapshot(),
                geometry=batch.tree_kinematics(),
                diagnostic=batch.tree_diagnostics(),
                receipts=dict(plan.last_accepted),
                stats=batch.stats(),
            )
            assert row["acceptance"] == dict.fromkeys(ORDER, True)
            prior = record["trace"][index]
            assert row["state"] == prior["state"]
            assert json_value(row["geometry"]) == json_value(prior["geometry"])
            assert json_value(row["receipts"]) == json_value(prior["receipts"])
            row["checks_complete"] = True
        record["checks"]["checkpoint_19_replay_21"] = True
        before, old_receipts, energy = (
            batch.snapshot(),
            json_value(plan.last_accepted),
            batch.summary(),
        )
        with batch.prepare_contacts(
            model, bindings, laws, site_capacity={503: 128, 91: 0, 1207: 128}
        ) as limited:
            acceptance = batch.step_contacts(limited, H)
            record["capacity"] = dict(
                before=before,
                acceptance=acceptance,
                after=batch.snapshot(),
                statuses=limited.last_status,
                receipts=limited.last_accepted,
                stats=batch.stats(),
            )
            assert acceptance == {503: True, 91: False, 1207: True, 6: True}
            assert batch.snapshot()[91] == before[91]
            assert batch.summary()[91] == energy[91]
            assert limited.last_status[91] == "SITE_CAPACITY_EXCEEDED"
            assert batch.stats()["affine_tree_status_codes"][91] == "SITE_CAPACITY_EXCEEDED"
            assert json_value(plan.last_accepted) == old_receipts
        assert batch.step_contacts(plan, H) == dict.fromkeys(ORDER, True)
        assert plan.last_accepted[91].entry_state == before[91]
        record["capacity_retry"] = dict(
            state=batch.snapshot(), receipts=plan.last_accepted, stats=batch.stats()
        )
        record["checks"]["local_capacity_and_retry"] = True
        retained = plan.last_accepted[1207]
        for array in (
            retained.contact_generalized,
            retained.law_wrench_fp64,
            retained.diagnostics["projection_matrix"],
            retained.diagnostics["affine_diagnostic"].reduced_mass,
            retained.accepted_geometry.link_rotation,
        ):
            assert not np.asarray(array).flags.writeable
            try:
                np.asarray(array).setflags(write=True)
            except ValueError:
                pass
            else:
                raise AssertionError("owned public contact data became writeable")
        state, epoch = batch.snapshot(), batch.stats()["state_epoch"]
        batch.tree_kinematics()
        batch.tree_diagnostics()
        assert batch.snapshot() == state and batch.stats()["state_epoch"] == epoch
        record["checks"]["owned_readonly_observations"] = True
        record["batch_stats_before_close"] = batch.stats()
        record["plan_stats_before_close"] = plan.stats()
        assert (
            batch.stats()["submitted_kernels"] > 0 and plan.stats()["contact_kernel_launches"] > 0
        )
    except Exception as error:
        record["window_error"] = dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        try:
            record["retained_at_exit"] = batch.snapshot()
        finally:
            batch.close()
            batch.close()
            record["batch_stats_after_close"] = batch.stats()
            if plan is not None:
                record["plan_stats_after_close"] = plan.stats()
    assert batch.stats()["device_owned_bytes"] == 0
    assert plan.stats()["closed"] and plan.stats()["device_owned_bytes"] == 0
    try:
        batch.step_contacts(plan, H)
    except RuntimeError as error:
        record["closed_rejection"] = dict(type=type(error).__name__, message=str(error))
    else:
        raise AssertionError("closed batch accepted a contact step")
    record["checks"]["close_released_owned_buffers"] = True


def archive_identity(installed, expected):
    wheel = expected.resolve(strict=True)
    if wheel.suffix.lower() != ".whl":
        raise ValueError("--expected-wheel must name the actual installed wheel archive")
    with wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    direct_text = installed.read_text("direct_url.json")
    if direct_text is None:
        raise AssertionError("install the local wheel directly, retaining direct_url.json")
    direct = json.loads(direct_text)
    origin = urlsplit(direct["url"])
    assert origin.scheme == "file" and origin.netloc in ("", "localhost")
    assert Path(url2pathname(origin.path)).resolve(strict=True) == wheel.resolve(strict=True)
    archive = direct["archive_info"]
    recorded = archive.get("hashes", {}).get("sha256")
    if recorded is None and archive.get("hash"):
        algorithm, _, recorded = archive["hash"].partition("=")
        assert algorithm == "sha256"
    if recorded is not None:
        assert recorded.lower() == digest
    # The direct-URL specification permits an empty archive_info (e.g. uv).
    # Bind the exact local archive and compare the entire installed product,
    # rather than depending on optional installer hash metadata.
    site = Path(installed.locate_file("")).resolve(strict=True)
    with ZipFile(wheel) as archive:
        members = {
            name
            for name in archive.namelist()
            if name.startswith("phymeo/") and not name.endswith("/")
        }
        actual = {
            path.relative_to(site).as_posix()
            for path in (site / "phymeo").rglob("*")
            if path.is_file()
        }
        assert actual == members, "installed product membership differs from wheel"
        for name in sorted(members):
            path = (site / name).resolve(strict=True)
            assert path.is_relative_to(site)
            assert path.read_bytes() == archive.read(name), f"installed content differs: {name}"
        for name in archive.namelist():
            if ".dist-info/" in name and name.rsplit("/", 1)[-1] in {
                "METADATA",
                "WHEEL",
                "top_level.txt",
            }:
                assert (site / name).read_bytes() == archive.read(name), name
    return dict(
        path=str(wheel),
        sha256=digest,
        installed_from=direct["url"],
        version=installed.version,
        installer_recorded_sha256=recorded,
        installed_product_files_verified=len(members),
    )


def module_provenance(site, package_root, wheel):
    rows = []
    with ZipFile(wheel) as archive:
        for name, module in sorted(tuple(sys.modules.items())):
            if name != "phymeo" and not name.startswith("phymeo."):
                continue
            file = getattr(module, "__file__", None)
            assert file is not None, f"no installed-wheel origin for {name}"
            path = Path(file).resolve(strict=True)
            assert path.is_relative_to(package_root), f"non-wheel module {name}: {path}"
            spec = getattr(module, "__spec__", None)
            assert spec is not None and spec.origin is not None
            assert Path(spec.origin).resolve(strict=True) == path
            for search in getattr(module, "__path__", ()):
                assert Path(search).resolve(strict=True).is_relative_to(package_root)
            member = path.relative_to(site).as_posix()
            content = path.read_bytes()
            assert archive.read(member) == content, f"installed bytes differ from wheel: {member}"
            rows.append(
                dict(
                    module=name,
                    path=str(path),
                    origin=spec.origin,
                    wheel_member=member,
                    sha256=hashlib.sha256(content).hexdigest(),
                )
            )
    assert rows
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wheel-site", type=Path, required=True)
    parser.add_argument("--expected-package-root", type=Path, required=True)
    parser.add_argument("--expected-wheel", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    args = parser.parse_args()
    assert __debug__ and sys.flags.isolated and sys.flags.dont_write_bytecode, "use python -I -B"
    for path in (
        args.output_dir,
        args.wheel_site,
        args.expected_package_root,
        args.expected_wheel,
        args.checkout,
    ):
        assert path.is_absolute(), "all external paths must be absolute"
    site, root, checkout = (
        x.resolve(strict=True) for x in (args.wheel_site, args.expected_package_root, args.checkout)
    )
    script, output = Path(__file__).resolve(strict=True), args.output_dir.resolve()
    assert root == (site / "phymeo").resolve(strict=True)
    assert Path.cwd().resolve() == script.parent
    assert (
        not script.is_relative_to(checkout)
        and not site.is_relative_to(checkout)
        and not output.is_relative_to(checkout)
    )
    assert "phymeo" not in sys.modules
    output.mkdir(parents=True, exist_ok=False)
    record = dict(
        status="running",
        mode=args.mode,
        errors=[],
        script=dict(path=str(script), sha256=hashlib.sha256(script.read_bytes()).hexdigest()),
        expected_package_root=str(root),
        gates=dict(
            h_s=H,
            steps=STEPS,
            replay_steps=STEPS - REPLAY_FROM,
            rtol=RTOL,
            state_atol=ATOL,
            window_atol=WINDOW_ATOL,
            constraint_atol=2e-5,
            mass_residual=2e-5,
            bounded_pd_residual=5e-5,
            material_record_atol=1e-10,
        ),
        units=dict(
            q="R rad/P m; scalar xyz m",
            v="R rad/s/P m/s; scalar m/s",
            force="N",
            torque="N*m",
            energy="J",
            time="s",
        ),
        limits="Installed public functionality smoke; HOST finite narrow/law plus TYX fixed-root response. No benchmark, floating, moving/moving, friction, CCD, native Newton equivalence or replacement for repository C7 tests.",
    )
    sys.path.insert(0, str(site))
    try:
        import phymeo as ph

        assert Path(ph.__file__).resolve(strict=True).parent == root
        installed = distribution("phymeo")
        assert Path(installed.locate_file("")).resolve(strict=True) == site
        record["wheel"] = archive_identity(installed, args.expected_wheel)
        record["modules_before"] = module_provenance(site, root, args.expected_wheel)
        record["versions"] = dict(
            phymeo=installed.version,
            numpy=np.__version__,
            tyx=distribution("tyx-compute").version,
            python=sys.version,
        )
        run_window(ph, args.mode, output, record)
        record["status"] = "passed"
    except Exception as error:
        record["status"] = "failed"
        record["errors"].append(
            dict(
                stage="wheel_import_or_physical_window",
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            )
        )
    try:
        if "phymeo" in sys.modules:
            record["modules_after"] = module_provenance(site, root, args.expected_wheel)
    except Exception as error:
        record["status"] = "failed"
        record["errors"].append(
            dict(
                stage="final_module_provenance",
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            )
        )
    trace = output / "resident-affine-contacts.trace.json.gz"
    with gzip.open(trace, "xt", encoding="utf-8") as stream:
        json.dump(json_value(record), stream, allow_nan=False, separators=(",", ":"))
    summary = dict(
        status=record["status"],
        raw_exit=0 if record["status"] == "passed" else 1,
        mode=args.mode,
        full_steps=sum(bool(row.get("checks_complete")) for row in record.get("trace", ())),
        full_replay_steps=sum(bool(row.get("checks_complete")) for row in record.get("replay", ())),
        checks=record.get("checks", {}),
        wheel=record.get("wheel"),
        verified_module_count=len(record.get("modules_after", ())),
        batch_stats=record.get("batch_stats_after_close"),
        trace=str(trace),
        errors=record["errors"],
        limits=record["limits"],
    )
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(json_value(summary), stream, allow_nan=False, indent=2)
    print(json.dumps(json_value(summary), allow_nan=False))
    return summary["raw_exit"]


if __name__ == "__main__":
    raise SystemExit(main())
