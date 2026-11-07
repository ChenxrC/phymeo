"""Source-independent public wheel consumer for frozen C8 double-dynamic contact.

Run a copied file with python -I -B. Public APIs only: ordinary D01, affine D04,
legacy static box support, and an unbound FreeSphere share one batch. The fixed
window is 40 steps; restore tick 19 and replay all 21 remaining steps; then one
local capacity refusal and one retry. No checkout/test/private advance imports.
Author checkpoint UNEXECUTED; no parameters adapt to observed output.
"""

import argparse
import gzip
import hashlib
import json
import sys
import traceback
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from importlib.metadata import distribution
from itertools import product
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname
from zipfile import ZipFile

import numpy as np

H, STEPS, REPLAY_FROM = 1 / 1024, 40, 19
ATOL, RTOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-4
MAX_RECORD_BYTES = 256 * 1024 * 1024
ORDER = (1207, 6, 91, 503)
TREE_WORLDS = (503, 91, 1207)
NAMES = {503: "D01", 91: "D04", 1207: "legacy_static", 6: "free"}

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


def owned_arrays(value):
    if isinstance(value, np.ndarray):
        assert not value.flags.writeable
    elif is_dataclass(value):
        for field in fields(value):
            owned_arrays(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            owned_arrays(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            owned_arrays(item)


def make_ordinary(ph, *, static=False):
    n = 1 if static else 2
    offsets = ((0, 0, 3 / 32),) if static else ((0, 0, 0), (7 / 32, 0, 0))
    axis = (0, 0, 1) if static else (1, 0, 0)
    model = ph.RobotModel(
        np.full(n, -1, dtype=int), ("prismatic",) * n,
        np.asarray(offsets, float), np.tile((0, 0, 0, 1.0), (n, 1)),
        np.tile(axis, (n, 1)).astype(float), np.asarray((2.0,) if static else (2.0, 1.0)),
        np.zeros((n, 3)), np.full((n, 3), 1 / 8), np.zeros(3),
    )
    program = ph.DrivenRobotProgram(model, kp=np.zeros(n), kd=np.zeros(n),
                                    effort_limit=np.full(n, 8.0), delay_steps=0)
    initial = program.initial_state(ph.RobotState(np.zeros(n), np.zeros(n)))
    world = ph.FixedRobotTree(program, initial, control_max_iterations=512)
    command = program.inputs(np.zeros(n), np.zeros(n), feedforward=np.zeros(n),
                             loads=ph.RobotInputs.zeros(n))
    return world, command, dict(
        parents=(-1,) * n, kinds=("prismatic",) * n, offsets=offsets, axis=axis,
        mass=(2.0,) if static else (2.0, 1.0), inertia=((1 / 8,) * 3,) * n,
        com=((0, 0, 0),) * n, kp=(0,) * n, kd=(0,) * n, effort_cap=(8,) * n,
        delay_steps=0, control_max_iterations=512, gravity=(0, 0, 0),
    )


def affine_xml():
    root = ET.Element("robot", name="c8_D04_declared")
    ET.SubElement(root, "link", name="base")
    for i, mass in enumerate((2.0, 1.0)):
        link = ET.SubElement(root, "link", name=f"link{i}")
        inertia = ET.SubElement(link, "inertial")
        ET.SubElement(inertia, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(inertia, "mass", value=str(mass))
        ET.SubElement(inertia, "inertia", ixx="0.125", iyy="0.125", izz="0.125",
                      ixy="0", ixz="0", iyz="0")
        joint = ET.SubElement(root, "joint", name=f"joint{i}", type="prismatic")
        ET.SubElement(joint, "parent", link="base")
        ET.SubElement(joint, "child", link=f"link{i}")
        ET.SubElement(joint, "origin", xyz="0 0 0" if i == 0 else "0.09375 0 0", rpy="0 0 0")
        ET.SubElement(joint, "axis", xyz="1 0 0")
        ET.SubElement(joint, "limit", lower="-8", upper="8", velocity="16", effort="8")
        if i == 1:
            ET.SubElement(joint, "mimic", joint="joint0", multiplier="-0.5", offset="0.125")
    return ET.tostring(root, encoding="unicode")


def make_affine(ph, output):
    path = output / "D04.urdf"
    text = affine_xml()
    path.write_text(text, encoding="utf-8")
    asset = ph.load_mimic_urdf(path, gravity=(0, 0, 0))
    assert asset.tree.joint_names == ("joint0", "joint1")
    np.testing.assert_array_equal(asset.matrix, [[1], [-0.5]])
    np.testing.assert_array_equal(asset.offset_vector, [0, 1 / 8])
    world = ph.AffineRobotTree(
        asset, ph.RobotState(np.zeros(1), np.zeros(1)), kp=np.zeros(1), kd=np.zeros(1),
        effort_limit=np.array([8.0]), delay_steps=0, control_max_iterations=512,
    )
    command = world.inputs(np.zeros(1), np.zeros(1), feedforward=np.zeros(1),
                           loads=ph.RobotInputs.zeros(2))
    return world, command, dict(path=str(path), content=text, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                P=((1,), (-0.5,)), c=(0, 1 / 8), C=((0.5, 1),),
                                full_mass=(2, 1), full_inertia=((1 / 8,) * 3,) * 2)


def geometry(ph, checkpoint):
    shapes, bindings, laws, sources = [], [], {}, {}
    for world in TREE_WORLDS:
        first, second = ph.CollisionKey(world, 100, 11), ph.CollisionKey(world, 101 if world != 1207 else 900, 7)
        if world != 1207:
            left = ph.CollisionShape(first, "sphere", (107,), radius=1 / 8)
            right = ph.CollisionShape(second, "sphere", (211,), radius=1 / 8)
            points_a = points_b = np.zeros((1, 3))
            groups = (0, 1)
        else:
            left = ph.CollisionShape(first, "hull", tuple(range(101, 109)))
            right = ph.CollisionShape(second, "hull", tuple(range(201, 209)))
            points_a = box([-1 / 8] * 3, [1 / 8] * 3)
            points_b = box([-1 / 2, -1 / 2, -1 / 4], [1 / 2, 1 / 2, 0])
            groups = (0, None)
        cls = ph.ResidentAffineTreeBinding if world == 91 else ph.ResidentTreeBinding
        bindings.append(cls(first, world, groups[0], points_a, topology=checkpoint[world].topology))
        bindings.append(
            ph.ResidentStaticBinding(second, ph.CollisionMotion(points_b)) if groups[1] is None
            else cls(second, world, groups[1], points_b, topology=checkpoint[world].topology)
        )
        shapes.extend((left, right))
        laws[(first, second)] = ph.RobotContactMaterial(64, 0, 0)
        sources[world] = {first: (left, groups[0], points_a), second: (right, groups[1], points_b)}
    # Reverse registration independently of physical body/group and canonical force signs.
    return ph.CollisionModel(tuple(reversed(shapes))), tuple(reversed(bindings)), laws, sources


def oracle(world, q, v):
    """Independent diagonal full mechanics plus the declared affine map, never diagnostics."""
    q, v = np.asarray(q, float), np.asarray(v, float)
    if world == 6:
        return dict(a=np.array([0.0, 0.0, -8.0]), kinetic_energy_j=float(v @ v))
    if world == 1207:
        full_q, full_v, P, c = q, v, np.eye(1), np.zeros(1)
        positions = np.array([[0, 0, 3 / 32 + q[0]]])
        normal = 64 * max(0.0, 1 / 32 - q[0])
        external = np.array([normal])
        mass = np.array([[2.0]])
        linear = np.zeros((1, 3, 1))
        linear[0, 2, 0] = 1
        wrench = np.array([[0, 0, normal, 0, 0, 0]])
        axis = 2
    else:
        P = np.array([[1.0], [-0.5]]) if world == 91 else np.eye(2)
        c = np.array([0, 1 / 8]) if world == 91 else np.zeros(2)
        full_q, full_v = P @ q + c, P @ v
        origin = 3 / 32 if world == 91 else 7 / 32
        positions = np.array([[full_q[0], 0, 0], [origin + full_q[1], 0, 0]])
        distance = positions[1, 0] - positions[0, 0]
        assert distance > 0, "frozen branch must retain its declared normal orientation"
        normal = 64 * max(0.0, 1 / 4 - distance)
        external = np.array([-normal, normal])
        mass = np.diag([2.0, 1.0])
        linear = np.zeros((2, 3, 2))
        linear[0, 0, 0] = linear[1, 0, 1] = 1
        wrench = np.array([[-normal, 0, 0, 0, 0, 0], [normal, 0, 0, 0, 0, 0]])
        axis = 0
    reduced_mass, reduced_external = P.T @ mass @ P, P.T @ external
    if world == 91:
        acceleration = np.array([-2 * normal / 3])
        reaction = np.array([-normal / 3, -2 * normal / 3])
        multipliers = np.array([-2 * normal / 3])
    else:
        acceleration = external / np.diag(mass)
        reaction, multipliers = np.zeros(len(full_q)), np.zeros(0)
    n = len(full_q)
    return dict(
        a=acceleration, full_a=P @ acceleration, q=full_q, v=full_v, P=P, c=c,
        position=positions, rotation=np.repeat(np.eye(3)[None], n, axis=0),
        mass=mass, reduced_mass=reduced_mass, external=external, reduced_external=reduced_external,
        bias=np.zeros(n), linear=linear, angular=np.zeros((n, 3, n)),
        wrench=wrench, normal=normal, reaction=reaction, multipliers=multipliers, axis=axis,
        kinetic_energy_j=float(full_v @ mass @ full_v / 2),
    )


def broad_record(result):
    return {name: getattr(result, name) for name in (
        "topology_generation", "state_epoch", "world_keys", "ticks", "shape_keys",
        "shape_revisions", "bounds", "registered_pairs", "candidates",
    )}


def capture(batch, plan):
    entry_broad = None if plan.last_result is None else broad_record(plan.last_result)
    current = plan.query()
    return dict(
        state=batch.snapshot(), geometry=batch.tree_kinematics(), diagnostics=batch.tree_diagnostics(),
        energy=batch.summary(), receipts=dict(plan.last_accepted), statuses=dict(plan.last_status),
        entry_broad_phase=entry_broad, current_broad_phase=broad_record(current),
        batch_stats=batch.stats(), plan_stats=plan.stats(),
    )


def check_geometry(row, sources, laws):
    result = row["current_broad_phase"]
    assert set(result["registered_pairs"]) == set(laws)
    assert set(result["candidates"]) == set(laws), "all three frozen pairs remain overlapping"
    for world, tick in zip(result["world_keys"], result["ticks"], strict=True):
        assert tick == row["state"][world].tick
    for world in TREE_WORLDS:
        state, kin = row["state"][world], row["geometry"][world]
        truth = oracle(world, state.position, state.velocity)
        close(kin.position, truth["q"])
        close(kin.velocity, truth["v"])
        close(kin.link_position_m, truth["position"])
        close(kin.com_position_m, truth["position"])
        close(kin.link_rotation, truth["rotation"])
        close(row["energy"][world]["kinetic_energy_j"], truth["kinetic_energy_j"])
        if world == 91:
            close(kin.independent_position, state.position)
            close(kin.independent_velocity, state.velocity)
            assert kin.joint_names == ("joint0",) and kin.full_joint_names == ("joint0", "joint1")
        for key, (shape, group, local) in sources[world].items():
            points = local if group is None else truth["position"][group] + local
            low, high = points.min(0) - shape.radius, points.max(0) + shape.radius
            slot = result["shape_keys"].index(key)
            bounds = result["bounds"][slot]
            assert np.isfinite(bounds).all()
            assert np.all(bounds[:3] <= low) and np.all(bounds[3:] >= high)
            assert result["shape_revisions"][slot] == 0
    owned_arrays(row["geometry"])

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


def audit_world(world, entry, row, source):
    state, receipt, diagnostic = row["state"][world], row["receipts"][world], row["diagnostics"][world]
    truth = oracle(world, entry.position, entry.velocity)
    assert receipt.entry_state == entry and receipt.accepted_state == state
    assert (receipt.entry_tick, receipt.exit_tick, receipt.dt_s) == (entry.tick, entry.tick + 1, H)
    assert state.control == entry.control and state.history == entry.history == ()
    n, k = len(truth["q"]), len(entry.position)
    if world == 91:
        np.testing.assert_array_equal(state.control[0], np.zeros((k, 3), np.float32))
        np.testing.assert_array_equal(state.control[1], np.zeros((n, 7), np.float32))
        close(state.full_position, truth["P"] @ np.asarray(state.position) + truth["c"])
        close(state.full_velocity, truth["P"] @ np.asarray(state.velocity))
        close(state.full_applied_effort, np.zeros(n))
        close(state.full_generalized_applied, truth["external"])
        close(state.generalized_applied, truth["reduced_external"])
        assert receipt.diagnostics["coordinate_kind"] == "affine"
        close(receipt.diagnostics["projection_matrix"], truth["P"])
        close(receipt.diagnostics["constraint_rows"], [[0.5, 1]])
        close(receipt.diagnostics["offset_vector"], truth["c"])
        close(diagnostic.reduced_mass, [[9 / 4]])
        close(diagnostic.reduced_external, truth["reduced_external"])
        close(diagnostic.reduced_bias, np.zeros(k))
        close(diagnostic.full_acceleration, truth["full_a"])
        close(diagnostic.constraint_effort_full, truth["reaction"])
        close(diagnostic.row_multipliers, truth["multipliers"])
        close(diagnostic.row_powers_w, [0], atol=2e-5)
        close(diagnostic.constraint_power_w, 0, atol=2e-5)
        close(np.array([[0.5, 1]]) @ np.asarray(state.full_velocity), [0], atol=2e-5)
        assert json_value(receipt.diagnostics["affine_diagnostic"]) == json_value(diagnostic)
    else:
        np.testing.assert_array_equal(state.control, np.zeros((n, 10), np.float32))
        close(state.generalized_applied, truth["external"])
    close(state.applied_effort, np.zeros(k))
    close(state.velocity, np.asarray(entry.velocity) + H * truth["a"])
    close(state.position, np.asarray(entry.position) + H * np.asarray(state.velocity))
    for name, expected in (
        ("mass", truth["mass"]), ("bias", truth["bias"]), ("external", truth["external"]),
        ("linear_jacobian", truth["linear"]), ("angular_jacobian", truth["angular"]),
        ("drive", np.zeros(k)), ("acceleration", truth["a"]),
    ):
        close(getattr(diagnostic, name), expected)
    assert diagnostic.equation_residual <= 2e-5 and diagnostic.control_residual <= 5e-5
    close(receipt.input_external, np.zeros(n))
    close(receipt.selected_drive, np.zeros((k, 3)))
    close(receipt.actual_drive, np.zeros(k))
    close(receipt.contact_generalized, truth["external"])
    close(receipt.law_wrench_fp64, truth["wrench"])
    close(sum(float(np.linalg.norm(site.first_force_n)) for site in receipt.sites), truth["normal"])
    close(sum(site.geometry.weight for site in receipt.sites), 1.0)
    assert receipt.diagnostics["tree_dynamic_pairs"] is True
    if world != 1207:
        assert not receipt.static_reactions
        assert len(receipt.sites) == 1
        close(receipt.law_wrench_fp64[:, :3].sum(0), np.zeros(3))
        close((receipt.law_wrench_fp64[:, 3:] + np.cross(truth["position"], receipt.law_wrench_fp64[:, :3])).sum(0), np.zeros(3))
    else:
        assert len(receipt.static_reactions) == 1
        reaction = next(iter(receipt.static_reactions.values()))
        close(reaction.force_n, [0, 0, -truth["normal"]])
    if world == 503:
        close(2 * state.velocity[0] + state.velocity[1], 0.0)
    work = audit_sites(receipt, diagnostic, source)
    next_truth = oracle(world, state.position, state.velocity)
    meta = receipt.diagnostics
    close(meta["entry_kinetic_energy_j"], truth["kinetic_energy_j"])
    close(meta["accepted_kinetic_energy_j"], next_truth["kinetic_energy_j"])
    close(meta["kinetic_energy_change_j"], next_truth["kinetic_energy_j"] - truth["kinetic_energy_j"])
    close(meta["application_work_j"], work["application_work_j"], atol=1e-12, rtol=1e-10)
    close(meta["material_work_j"], work["material_work_j"], atol=1e-12, rtol=1e-10)
    owned_arrays(receipt)
    return dict(independent_entry=truth, finite_work=work)


def predictions(expected):
    result = {}
    for world, (q, v) in expected.items():
        a = oracle(world, q, v)["a"]
        next_v = np.asarray(v) + H * a
        result[world] = (np.asarray(q) + H * next_v, next_v)
    return result


def audit_step(entry, row, sources, laws, predicted, acceptance):
    assert row["acceptance"] == acceptance
    row["audits"] = {}
    for world in ORDER:
        state = row["state"][world]
        if not acceptance[world]:
            assert state == entry[world]
            continue
        assert state.tick == entry[world].tick + 1
        close(state.position, predicted[world][0], atol=WINDOW_ATOL)
        close(state.velocity, predicted[world][1], atol=WINDOW_ATOL)
        if world in TREE_WORLDS:
            row["audits"][world] = audit_world(world, entry[world], row, sources[world])
        else:
            close(state.velocity, np.asarray(entry[world].velocity) + [0, 0, -8 * H])
            close(state.position, np.asarray(entry[world].position) + H * np.asarray(state.velocity))
            assert state.control == entry[world].control
    check_geometry(row, sources, laws)
    row["checks_complete"] = True


def snapshot_fingerprint(snapshot):
    return json.dumps(json_value(snapshot), sort_keys=True, allow_nan=False, separators=(",", ":"))


def run_window(ph, mode, output, record):
    ordinary, command, ordinary_source = make_ordinary(ph)
    static, static_command, static_source = make_ordinary(ph, static=True)
    affine, affine_command, affine_source = make_affine(ph, output)
    worlds = {503: ordinary, 91: affine, 1207: static,
              6: ph.FreeSphere(2, 1 / 8, position=(0, 0, 2), gravity=(0, 0, -8))}
    commands = {503: command, 91: affine_command, 1207: static_command, 6: (0, 0, 0)}
    record.update(inputs=commands, declarations={503: ordinary_source, 91: affine_source,
                  1207: static_source, 6: dict(mass=2, radius=1 / 8, position=(0, 0, 2), velocity=(0, 0, 0), gravity=(0, 0, -8))},
                  world_order=ORDER, trace=[], replay=[], checks={})
    expected = {503: (np.zeros(2), np.zeros(2)), 91: (np.zeros(1), np.zeros(1)),
                1207: (np.zeros(1), np.zeros(1)), 6: (np.array([0.0, 0.0, 2.0]), np.zeros(3))}
    close(oracle(503, *expected[503])["a"], [-1, 2], atol=0, rtol=0)
    close(oracle(91, *expected[91])["a"], [-4 / 3], atol=0, rtol=0)
    close(oracle(91, *expected[91])["multipliers"], [-4 / 3], atol=0, rtol=0)
    batch = ph.TyxWorldBatch([(key, worlds[key]) for key in ORDER], mode, cache_root=output / "cache")
    plan = None
    try:
        batch.set_controls(commands)
        initial = batch.snapshot()
        model, bindings, laws, sources = geometry(ph, initial)
        record.update(initial=initial, collision_model=model, bindings=bindings, materials=laws, original_sources=sources)
        plan = batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True)
        record["initial_observation"] = capture(batch, plan)
        check_geometry(record["initial_observation"], sources, laws)
        retained, retained_bytes = None, None
        for index in range(STEPS):
            entry = batch.snapshot()
            predicted = predictions(expected)
            row = dict(entry=entry, independent_predicted=predicted, checks_complete=False)
            record["trace"].append(row)
            row["acceptance"] = batch.step_contacts(plan, H)
            row.update(capture(batch, plan))
            audit_step(entry, row, sources, laws, predicted, dict.fromkeys(ORDER, True))
            expected = predicted
            if retained is None:
                retained = dict(plan.last_accepted)
                retained_bytes = snapshot_fingerprint(retained)
            assert snapshot_fingerprint(retained) == retained_bytes
            if index + 1 == REPLAY_FROM:
                record["checkpoint_19"] = batch.snapshot()
        record["checks"]["continuous_40_with_independent_physics"] = True
        record["checks"]["retained_owned_receipts_unchanged"] = True
        before_restore, before_epoch = batch.snapshot(), batch.stats()["state_epoch"]
        before_receipts = snapshot_fingerprint(plan.last_accepted)
        invalid = dict(record["checkpoint_19"])
        invalid[91] = replace(invalid[91], position=(4097.0,))
        record["invalid_restore_input"] = invalid
        try:
            batch.restore(invalid)
        except (ValueError, TypeError) as error:
            record["invalid_restore_rejection"] = dict(type=type(error).__name__, message=str(error))
        else:
            raise AssertionError("mixed restore accepted an out-of-domain independent state")
        assert batch.snapshot() == before_restore and batch.stats()["state_epoch"] == before_epoch
        assert snapshot_fingerprint(plan.last_accepted) == before_receipts
        record["checks"]["invalid_mixed_restore_atomic"] = True
        batch.restore(record["checkpoint_19"])
        assert batch.snapshot() == record["checkpoint_19"]
        assert batch.stats()["state_epoch"] == before_epoch + 1
        record["restored_observation"] = capture(batch, plan)
        for index in range(REPLAY_FROM, STEPS):
            row = dict(checks_complete=False)
            record["replay"].append(row)
            row["acceptance"] = batch.step_contacts(plan, H)
            row.update(capture(batch, plan))
            prior = record["trace"][index]
            assert row["acceptance"] == dict.fromkeys(ORDER, True)
            assert row["state"] == prior["state"]
            for key in ("geometry", "diagnostics", "energy", "receipts"):
                assert json_value(row[key]) == json_value(prior[key]), f"replay differs: {key}"
            for key in ("world_keys", "ticks", "shape_keys", "shape_revisions", "bounds", "registered_pairs", "candidates"):
                assert json_value(row["current_broad_phase"][key]) == json_value(prior["current_broad_phase"][key])
            # Epoch, launch/readback counters are not rewindable state and are recorded separately.
            row["checks_complete"] = True
        record["checks"]["tick19_full21_replay"] = True
        before = batch.snapshot()
        old_receipts = snapshot_fingerprint(plan.last_accepted)
        energy = batch.summary()
        predicted = predictions(expected)
        with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True,
                                    site_capacity={503: 128, 91: 0, 1207: 128}) as limited:
            capacity = dict(entry=before, checks_complete=False)
            record["capacity"] = capacity
            capacity["acceptance"] = batch.step_contacts(limited, H)
            capacity.update(capture(batch, limited))
            wanted = {503: True, 91: False, 1207: True, 6: True}
            audit_step(before, capacity, sources, laws, predicted, wanted)
            assert capacity["state"][91] == before[91] and capacity["energy"][91] == energy[91]
            assert limited.last_status[91] == "SITE_CAPACITY_EXCEEDED"
            assert batch.stats()["affine_tree_status_codes"][91] == "SITE_CAPACITY_EXCEEDED"
            assert 91 not in limited.last_accepted
            assert snapshot_fingerprint(plan.last_accepted) == old_receipts
            capacity["checks_complete"] = True
        record["capacity_plan_closed_stats"] = limited.stats()
        assert limited.stats()["closed"] and limited.stats()["device_owned_bytes"] == 0
        expected = {key: expected[key] if key == 91 else predicted[key] for key in ORDER}
        entry = batch.snapshot()
        predicted = predictions(expected)
        retry = dict(entry=entry, independent_predicted=predicted, checks_complete=False)
        record["capacity_retry"] = retry
        retry["acceptance"] = batch.step_contacts(plan, H)
        retry.update(capture(batch, plan))
        audit_step(entry, retry, sources, laws, predicted, dict.fromkeys(ORDER, True))
        assert plan.last_accepted[91].entry_state == before[91]
        record["checks"]["local_capacity_retention_and_retry"] = True
        assert snapshot_fingerprint(retained) == retained_bytes
        # Stored first-step observations remain byte-serializable and immutable after replay/rejection.
        owned_arrays(retained)
        for array in (retained[91].contact_generalized, retained[91].diagnostics["projection_matrix"],
                      retained[503].accepted_geometry.link_position_m):
            try:
                array.setflags(write=True)
            except ValueError:
                pass
            else:
                raise AssertionError("owned accepted data can be made writeable")
        record["checks"]["public_observations_owned"] = True
    except Exception as error:
        record["window_error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            record["retained_at_exit"] = batch.snapshot()
        except Exception as error:
            record["retained_read_error"] = dict(type=type(error).__name__, message=str(error))
        finally:
            record["batch_stats_before_close"] = batch.stats()
            if plan is not None:
                record["plan_stats_before_close"] = plan.stats()
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
        raise AssertionError("closed batch advanced")
    record["checks"]["closed_resources_released"] = True

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


def save_record(output, record):
    encoded = json.dumps(json_value(record), allow_nan=False, separators=(",", ":")).encode()
    if len(encoded) > MAX_RECORD_BYTES:
        raise RuntimeError("predeclared 256 MiB complete output limit exceeded; no partial success")
    path = output / "resident-dynamic-contacts.trace.json.gz"
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wb") as stream:
        stream.write(encoded)
    temporary.replace(path)
    return dict(path=str(path), uncompressed_bytes=len(encoded), gzip_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wheel-site", type=Path, required=True)
    parser.add_argument("--expected-package-root", type=Path, required=True)
    parser.add_argument("--expected-wheel", type=Path, required=True)
    parser.add_argument("--forbidden-checkout", type=Path)
    args = parser.parse_args()
    assert __debug__ and sys.flags.isolated and sys.flags.dont_write_bytecode, "use python -I -B"
    for path in (args.output_dir, args.wheel_site, args.expected_package_root, args.expected_wheel):
        assert path.is_absolute(), "external paths must be absolute"
    site, root = (path.resolve(strict=True) for path in (args.wheel_site, args.expected_package_root))
    script, output = Path(__file__).resolve(strict=True), args.output_dir.resolve()
    assert root == (site / "phymeo").resolve(strict=True)
    assert Path.cwd().resolve() == script.parent
    if args.forbidden_checkout is not None:
        checkout = args.forbidden_checkout.resolve()
        assert not any(path.is_relative_to(checkout) for path in (script, site, output))
    assert "phymeo" not in sys.modules
    output.mkdir(parents=True, exist_ok=False)
    record = dict(
        status="running", mode=args.mode, errors=[],
        script=dict(path=str(script), sha256=hashlib.sha256(script.read_bytes()).hexdigest()),
        expected_package_root=str(root), checks={},
        gates=dict(h_s=H, continuous_steps=40, replay_from=19, replay_steps=21,
                   rtol=RTOL, state_atol=ATOL, window_atol=WINDOW_ATOL,
                   mass_residual=2e-5, bounded_pd_residual=5e-5, constraint_atol=2e-5,
                   material_record_atol=1e-10, rotation_atol=5e-5),
        units=dict(q="all declared joints P: m; free xyz m", v="m/s", force="N", torque="N*m", energy="J", time="s"),
        limits="Public installed-wheel smoke, not the C8 whole matrix or benchmark. Same-world complete fixed-root ordinary/affine trees; HOST finite narrow/law plus resident device response. No cross-world force, floating, friction, CCD or native Newton equivalence claim.",
    )
    sys.path.insert(0, str(site))
    try:
        import phymeo as ph

        assert Path(ph.__file__).resolve(strict=True).parent == root
        installed = distribution("phymeo")
        assert Path(installed.locate_file("")).resolve(strict=True) == site
        record["wheel"] = archive_identity(installed, args.expected_wheel)
        record["modules_before"] = module_provenance(site, root, args.expected_wheel)
        record["versions"] = dict(phymeo=installed.version, tyx=distribution("tyx-compute").version,
                                  numpy=np.__version__, python=sys.version)
        run_window(ph, args.mode, output, record)
        record["status"] = "passed"
    except Exception as error:
        record["status"] = "failed"
        record["errors"].append(dict(stage="import_or_physical_window", type=type(error).__name__,
                                     message=str(error), traceback=traceback.format_exc()))
    try:
        if "phymeo" in sys.modules:
            record["modules_after"] = module_provenance(site, root, args.expected_wheel)
    except Exception as error:
        record["status"] = "failed"
        record["errors"].append(dict(stage="module_provenance", type=type(error).__name__,
                                     message=str(error), traceback=traceback.format_exc()))
    # An IO error here is not a physics rollback: all completed/retained steps above remain facts.
    trace = save_record(output, record)
    summary = dict(
        status=record["status"], raw_exit=int(record["status"] != "passed"), mode=args.mode,
        complete_forward_steps=sum(bool(row.get("checks_complete")) for row in record.get("trace", ())),
        complete_replay_steps=sum(bool(row.get("checks_complete")) for row in record.get("replay", ())),
        capacity_complete=bool(record.get("capacity", {}).get("checks_complete")),
        retry_complete=bool(record.get("capacity_retry", {}).get("checks_complete")),
        checks=record["checks"], wheel=record.get("wheel"), trace=trace,
        verified_module_count=len(record.get("modules_after", ())),
        batch_stats=record.get("batch_stats_after_close"), errors=record["errors"], limits=record["limits"],
    )
    (output / "summary.json").write_text(json.dumps(json_value(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(json_value(summary), allow_nan=False))
    return summary["raw_exit"]


if __name__ == "__main__":
    raise SystemExit(main())
