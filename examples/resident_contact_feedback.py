"""Standalone C6 wheel consumer: two contact trees and two unbound controls.

Run a copied file outside the checkout with an installed wheel, for example:
  python -I -B resident_contact_feedback.py --tyx-mode cuda --output result.json

Only public phymeo APIs, NumPy and the standard library are used. R01's sliding
box and R03's eccentric sphere use their original constants and h=1/1024. The
main window is 40 continuous physical steps. A complete tick-20 checkpoint is
then restored and step 21 is replayed once: 41 batch calls, 164 world steps.
No target, material, geometry or threshold is adjusted after a result is read.

The output JSON is a small manifest; its companion .trace.json.gz contains the
complete input, all accepted states, FK, diagnostics, contacts, work and counters.
Contact sampling is discrete-entry HOST narrow phase with resident device tree
mechanics. This is a public wheel smoke, not a benchmark, CCD or grasp claim.
"""

import argparse
import gzip
import json
import platform
import sys
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from importlib.metadata import distribution
from itertools import product
from math import cos, sin
from pathlib import Path

import numpy as np

H = 1 / 1024
STEPS = 40
RESTORE_TICK = 20
WORLD_ORDER = (409, 17, 23, 7)
TREE_WORLDS = (409, 17, 7)
CONTACT_WORLDS = (17, 7)
ATOL = 5e-6
RTOL = 5e-5
WINDOW_ATOL = 2e-4
CONSTRUCTION_ATOL = 1e-10


def json_value(value):
    if is_dataclass(value):
        return {item.name: json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def owned_arrays(value):
    """Read ownership flags without attempting to mutate a live result."""
    if isinstance(value, np.ndarray):
        require(not value.flags.writeable, "a receipt array is writable")
    elif is_dataclass(value):
        for item in fields(value):
            owned_arrays(getattr(value, item.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            owned_arrays(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            owned_arrays(item)


def module_provenance(ph, expected_root):
    """Check installed origins, without reading product source or old evidence."""
    installed = distribution("phymeo")
    package_root = Path(installed.locate_file("phymeo")).resolve(strict=True)
    require(package_root.is_dir(), "the installed distribution has no phymeo directory")
    require(Path(ph.__file__).resolve().parent == package_root, "phymeo is not from this install")
    if expected_root is not None:
        require(package_root == expected_root.resolve(strict=True), "unexpected installed root")
    modules = []
    for name, module in sorted(tuple(sys.modules.items())):
        if name != "phymeo" and not name.startswith("phymeo."):
            continue
        origin = getattr(module, "__file__", None)
        require(origin is not None, f"no installed file origin: {name}")
        path = Path(origin).resolve(strict=True)
        require(path.is_relative_to(package_root), f"non-installed module: {name}: {path}")
        modules.append(dict(name=name, path=str(path)))
    require(bool(modules), "no phymeo modules were observed")
    return dict(
        package_root=str(package_root), phymeo_version=installed.version,
        tyx_distribution="tyx-compute", tyx_version=distribution("tyx-compute").version,
        numpy_version=np.__version__, python=sys.version, platform=platform.platform(),
        modules=modules,
        boundary="Origin check only; wheel archive/member parity belongs to the installer harness.",
    )


def box(low, high):
    return np.asarray(list(product(*zip(low, high, strict=True))), dtype=float)


def rz(angle):
    c, s = cos(angle), sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def tree_world(ph, *, rotary=False, feedforward=0.0):
    model = ph.RobotModel(
        np.array([-1]), ("revolute" if rotary else "prismatic",),
        np.array([[0., 0., 0. if rotary else 3 / 32]]),
        np.array([[0., 0., 0., 1.]]), np.array([[0., 0., 1.]]),
        np.array([1. if rotary else 2.]), np.zeros((1, 3)),
        np.array([[1 / 8, 1 / 8, 1 / 4 if rotary else 1 / 8]]), np.zeros(3),
    )
    program = ph.DrivenRobotProgram(
        model, kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.full(1, 8.), delay_steps=0,
    )
    initial = program.initial_state(ph.RobotState(np.zeros(1), np.zeros(1)))
    command = program.inputs(
        np.zeros(1), np.zeros(1), feedforward=np.array([feedforward]),
        loads=ph.RobotInputs(np.zeros(1), np.zeros((1, 3)), np.zeros((1, 3))),
    )
    return program, ph.FixedRobotTree(program, initial, control_max_iterations=512), command


def contact_geometry(ph, checkpoint):
    sources, shapes, bindings, materials = {}, [], [], {}
    law = ph.RobotContactMaterial(stiffness=64, damping=0, friction=0)
    for world in CONTACT_WORLDS:
        moving, fixed = ph.CollisionKey(world, 100, 11), ph.CollisionKey(world, 900, 1)
        if world == 17:
            local = box([-1 / 8] * 3, [1 / 8] * 3)
            support = box([-1 / 2, -1 / 2, -1 / 4], [1 / 2, 1 / 2, 0])
            shape = ph.CollisionShape(moving, "hull", tuple(range(101, 109)))
        else:
            local = np.array([[1 / 2, 3 / 32, 0]], dtype=float)
            support = box([-1, -1 / 4, -1 / 2], [1, 0, 1 / 2])
            shape = ph.CollisionShape(moving, "sphere", (101,), radius=1 / 8)
        static = ph.CollisionShape(fixed, "hull", tuple(range(201, 209)))
        shapes.extend((shape, static))
        bindings.extend((
            ph.ResidentTreeBinding(moving, world, 0, local, topology=checkpoint[world].topology),
            ph.ResidentStaticBinding(fixed, ph.CollisionMotion(support)),
        ))
        materials[(moving, fixed)] = law
        sources[moving] = dict(shape=shape, group=0, original_points_m=local)
        sources[fixed] = dict(shape=static, group=None, original_points_m=support)
    return ph.CollisionModel(tuple(shapes)), tuple(bindings), materials, sources


def advance_oracle(previous):
    """FP64 scalar recurrences from the declared mechanics, never product state."""
    result = {}
    for world, (q, v) in previous.items():
        normal, torque = 0., 0.
        if world == 17:
            normal = 64 * max(0., 1 / 32 - q)
            acceleration = normal / 2
        elif world == 7:
            cx, cy, _ = rz(q) @ np.array([1 / 2, 3 / 32, 0])
            normal = 64 * max(0., 1 / 8 - cy)
            torque = cx * normal
            acceleration = torque / (1 / 4)
        else:
            acceleration = .5 if world == 409 else 1.
        next_v = v + H * acceleration
        result[world] = dict(
            entry_q=q, entry_v=v, q=q + H * next_v, v=next_v,
            acceleration=acceleration, normal_n=normal, torque_nm=torque,
        )
    return result


def broad_packet(plan):
    result = plan.last_result
    if result is None:
        return None
    return {name: getattr(result, name) for name in (
        "topology_generation", "state_epoch", "world_keys", "ticks", "shape_keys",
        "shape_revisions", "bounds", "registered_pairs", "candidates",
    )}


def capture(batch, plan):
    # The caller is single-threaded. Each public read is independently locked;
    # these calls deliberately do not claim a new combined observation API.
    return dict(
        state=batch.snapshot(), geometry=batch.tree_kinematics(),
        diagnostics=batch.tree_diagnostics(), energy=batch.summary(),
        receipts=dict(plan.last_accepted), status=dict(plan.last_status),
        entry_broad_phase=broad_packet(plan), batch_stats=batch.stats(), plan_stats=plan.stats(),
    )


def audit_receipt(receipt, sources):
    """Audit saved material addresses, common force points and actual endpoint work."""
    require(receipt.exit_tick == receipt.entry_tick + 1, "receipt did not advance once")
    require(receipt.entry_geometry.tick == receipt.entry_tick, "wrong entry FK tick")
    require(receipt.accepted_geometry.tick == receipt.exit_tick, "wrong accepted FK tick")
    np.testing.assert_array_equal(receipt.uploaded_wrench_fp32,
                                  np.asarray(receipt.law_wrench_fp64, np.float32))
    owned_arrays(receipt)
    frames_by_time = (receipt.diagnostics["entry_frames"], receipt.diagnostics["accepted_frames"])
    expected_keys = {key for key in sources if key.world == receipt.world}
    for frames, geometry in zip(frames_by_time,
                                (receipt.entry_geometry, receipt.accepted_geometry), strict=True):
        require(set(frames) == expected_keys, "missing or foreign source frame")
        for key, frame in frames.items():
            source = sources[key]
            points, group = source["original_points_m"], source["group"]
            require(frame.shape == source["shape"], "changed original collision shape")
            require(frame.group_index == group, "changed owner group")
            expected = points if group is None else (
                points @ np.asarray(geometry.link_rotation[group], float).T
                + geometry.link_position_m[group]
            )
            close(frame.world_points_m, expected, atol=CONSTRUCTION_ATOL, rtol=0)
    total = np.zeros(6)
    reactions, application_work, material_work = {}, 0., 0.
    for site in receipt.sites:
        geometry, pair = site.geometry, site.pair
        witness, load = geometry.witness, np.asarray(geometry.load_point, float)
        require(pair == tuple(sorted(pair)), "noncanonical contact pair")
        require(pair[0].world == pair[1].world == receipt.world, "cross-world site")
        normal = np.asarray(witness.normal, float)
        close(np.linalg.norm(normal), 1., atol=1e-12, rtol=0)
        close(np.asarray(site.first_force_n) + site.second_force_n, 0., atol=1e-12, rtol=0)
        close(np.dot(np.asarray(witness.second_point) - witness.first_point, normal),
              witness.gap_m, atol=CONSTRUCTION_ATOL, rtol=0)
        magnitude = 64 * max(0., -witness.gap_m) * geometry.weight
        close(site.first_force_n, -normal * magnitude, atol=1e-10, rtol=1e-10)
        site_application, site_material = 0., 0.
        for side, key in zip(("first", "second"), pair, strict=True):
            source = sources[key]
            shape, original = source["shape"], source["original_points_m"]
            point = np.asarray(getattr(witness, side + "_point"), float)
            frame, end = frames_by_time[0][key], frames_by_time[1][key]
            weights = getattr(witness, side + "_weights")
            by_id = dict(zip(shape.vertex_ids, original, strict=True))
            require(bool(weights) and all(i in by_id for i, _ in weights), "lost material IDs")
            close(sum(weight for _, weight in weights), 1., atol=1e-12, rtol=0)
            local = sum((weight * by_id[i] for i, weight in weights), np.zeros(3))
            offset = getattr(witness, side + "_offset")
            if source["group"] is None:
                rebuilt, dm, da = local, np.zeros(3), np.zeros(3)
                velocity = np.zeros(3)
            else:
                p0, p1 = np.asarray(frame.position_m, float), np.asarray(end.position_m, float)
                r0, r1 = np.asarray(frame.rotation, float), np.asarray(end.rotation, float)
                rebuilt = p0 + r0 @ local
                dm = p1 + r1 @ np.linalg.solve(r0, point - p0) - point
                da = p1 + r1 @ np.linalg.solve(r0, load - p0) - load
                velocity = np.asarray(frame.linear_velocity_m_s) + np.cross(
                    frame.angular_velocity_rad_s, load - frame.reference_point_m)
            if offset is not None:
                rebuilt = rebuilt + offset
            close(rebuilt, point, atol=CONSTRUCTION_ATOL, rtol=0)
            close(getattr(site, side + "_material_displacement_m"), dm,
                  atol=CONSTRUCTION_ATOL, rtol=0)
            close(getattr(site, side + "_application_displacement_m"), da,
                  atol=CONSTRUCTION_ATOL, rtol=0)
            close(getattr(site, side + "_velocity_m_s"), velocity)
            close(getattr(site, side + "_reference_point_m"), frame.reference_point_m,
                  atol=CONSTRUCTION_ATOL, rtol=0)
            force = np.asarray(getattr(site, side + "_force_n"), float)
            wrench = np.r_[force, np.cross(load - frame.reference_point_m, force)]
            if source["group"] is None:
                reactions.setdefault(key, np.zeros(6))
                reactions[key] += wrench
            else:
                total += wrench
            site_application += float(force @ da)
            site_material += float(force @ dm)
        close(site.application_work_j, site_application, atol=1e-12, rtol=1e-10)
        close(site.material_work_j, site_material, atol=1e-12, rtol=1e-10)
        application_work += site_application
        material_work += site_material
    close(receipt.law_wrench_fp64, total[None, :])
    require(set(receipt.static_reactions) == set(reactions), "static reaction coverage changed")
    for key, wrench in reactions.items():
        reaction = receipt.static_reactions[key]
        close(reaction.reference_point_m, sources[key]["original_points_m"].mean(axis=0),
              atol=CONSTRUCTION_ATOL, rtol=0)
        close(reaction.force_n, wrench[:3])
        close(reaction.torque_nm, wrench[3:])
    close(receipt.diagnostics["application_work_j"], application_work, atol=1e-12, rtol=1e-10)
    close(receipt.diagnostics["material_work_j"], material_work, atol=1e-12, rtol=1e-10)
    return dict(application_work_j=application_work, material_work_j=material_work)


def audit_step(row, expected, held, sources, step, entry):
    require(row["accepted"] == dict.fromkeys(WORLD_ORDER, True), "a world did not accept")
    require(set(row["receipts"]) == set(CONTACT_WORLDS), "wrong receipt owners")
    require(set(row["geometry"]) == set(TREE_WORLDS), "missing unbound or contact tree FK")
    audits = {}
    for world in WORLD_ORDER:
        state, oracle = row["state"][world], expected[world]
        require(state.tick == step, "wrong accepted tick")
        require(state.control == held[world].control, "held command changed")
        require(state.history == held[world].history == (), "delay-zero history changed")
        if world == 23:
            close(state.position, [2 + oracle["q"], 0, 0], atol=WINDOW_ATOL)
            close(state.velocity, [oracle["v"], 0, 0], atol=WINDOW_ATOL)
            close(state.applied_effort, [2, 0, 0])
        else:
            close(state.position, [oracle["q"]], atol=WINDOW_ATOL)
            close(state.velocity, [oracle["v"]], atol=WINDOW_ATOL)
            geometry = row["geometry"][world]
            require(geometry.tick == step, "FK tick is not current")
            np.testing.assert_array_equal(geometry.position, state.position)
            np.testing.assert_array_equal(geometry.velocity, state.velocity)
            position = [0, 0, 0] if world == 7 else [0, 0, 3 / 32 + oracle["q"]]
            rotation = rz(oracle["q"]) if world == 7 else np.eye(3)
            close(geometry.link_position_m, [position], atol=WINDOW_ATOL)
            close(geometry.com_position_m, [position], atol=WINDOW_ATOL)
            close(geometry.link_rotation, rotation[None, :, :], atol=WINDOW_ATOL)
            expected_drive = 1. if world == 409 else 0.
            close(state.applied_effort, [expected_drive])
            if world == 409:
                close(state.generalized_applied, [1.])
            diagnostic = row["diagnostics"][world]
            require(diagnostic.entry_tick == step - 1 and diagnostic.exit_tick == step,
                    "dynamics diagnostic belongs to another step")
            close(diagnostic.acceleration, [oracle["acceleration"]], atol=WINDOW_ATOL)
        inertia_factor = 1 / 8 if world == 7 else 1.
        close(row["energy"][world]["kinetic_energy_j"], inertia_factor * oracle["v"] ** 2,
              atol=WINDOW_ATOL)
        if world not in CONTACT_WORLDS:
            continue
        receipt = row["receipts"][world]
        require(receipt.entry_state == entry[world], "contact was not sampled at this entry state")
        require(receipt.dt_s == H, "receipt duration changed")
        require(receipt.accepted_state == state, "receipt did not publish the accepted state")
        require(row["status"][world] == "ACCEPTED", "wrong contact status")
        normal, torque = oracle["normal_n"], oracle["torque_nm"]
        force = [0, normal, 0] if world == 7 else [0, 0, normal]
        wrench = [*force, 0, 0, torque]
        close(receipt.law_wrench_fp64, [wrench], atol=WINDOW_ATOL)
        close(receipt.contact_generalized, [torque if world == 7 else normal], atol=WINDOW_ATOL)
        close(receipt.input_external, [0.])
        close(receipt.selected_drive, [[0., 0., 0.]])
        fixed_key = next(key for key in receipt.static_reactions if key.body == 900)
        reaction = receipt.static_reactions[fixed_key]
        close(reaction.force_n, -np.asarray(force), atol=WINDOW_ATOL)
        close(reaction.torque_nm, [0, 0, -torque], atol=WINDOW_ATOL)
        audits[world] = audit_receipt(receipt, sources)
    if step == 1:
        require(row["state"][17].position == (H * H,), "R01 lost its exact first position")
        require(row["state"][17].velocity == (H,), "R01 lost its exact first velocity")
        close(row["receipts"][7].law_wrench_fp64, [[0, 2, 0, 0, 0, 1]])
    return audits


def run(ph, mode, cache_root, report):
    p, translation, pc = tree_world(ph)
    r, rotation, rc = tree_world(ph, rotary=True)
    e, empty, ec = tree_world(ph, feedforward=1.)
    scalar = ph.FreeSphere(2., 1 / 16, position=(2., 0., 0.), gravity=(0., 0., 0.))
    worlds = [(409, empty), (17, translation), (23, scalar), (7, rotation)]
    commands = {409: ec, 17: pc, 23: np.array([2., 0., 0.]), 7: rc}
    report["input"] = dict(
        models={17: p.model, 7: r.model, 409: e.model}, scalar=scalar,
        commands=commands, h_s=H, main_steps=STEPS, restore_tick=RESTORE_TICK,
        replay_steps=1, world_order=WORLD_ORDER, effort_cap=8., kp=0., kd=0., delay_steps=0,
        control_max_iterations=512, candidate_capacity=4096, site_capacity=128,
        sampling="discrete_entry", units=dict(q17="m", q7="rad", q409="m", sphere="m"),
        physical_tolerances=dict(atol=ATOL, rtol=RTOL, window_atol=WINDOW_ATOL),
        record_construction_tolerance_m=CONSTRUCTION_ATOL,
    )
    batch, plan = None, None
    old_receipts, old_receipt_value, checkpoint, reference_row = None, None, None, None
    previous = {world: (0., 0.) for world in WORLD_ORDER}
    try:
        with ph.TyxWorldBatch(worlds, mode, cache_root=cache_root) as batch:
            report["backend"] = dict(
                requested_mode=mode, observed_mode=batch.stats()["mode"],
                mechanics="TYX resident full-tree FP32", narrow_phase="HOST finite discrete_entry",
            )
            require(report["backend"]["observed_mode"] == mode, "backend mode changed")
            batch.set_controls(commands)
            held = batch.snapshot()
            actual_before = held
            report["initial_state"] = json_value(held)
            model, bindings, materials, sources = contact_geometry(ph, held)
            report["input"].update(collision_model=model, bindings=bindings,
                                   materials=materials, original_materials=sources)
            for world, ff in ((17, 0.), (7, 0.), (409, 1.)):
                np.testing.assert_array_equal(held[world].control,
                                              np.array([[0., 0., ff, 0., 0., 0., 0., 0., 0., 0.]], np.float32))
            np.testing.assert_array_equal(held[23].control, [2., 0., 0.])
            with batch.prepare_contacts(
                model, bindings, materials, candidate_capacity=4096,
                site_capacity=128, sampling="discrete_entry",
            ) as plan:
                report["initial_geometry"] = json_value(batch.tree_kinematics())
                report["initial_stats"] = dict(batch=batch.stats(), plan=plan.stats())
                for index in range(STEPS):
                    expected = advance_oracle(previous)
                    row = dict(step=index + 1, expected=expected, attempted_h_s=H)
                    report["trace"].append(row)
                    report["batch_calls_attempted"] += 1
                    row["accepted"] = batch.step_contacts(plan, H)
                    report["main_steps_completed"] = index + 1
                    row.update(capture(batch, plan))
                    # Save full actual outputs before any physical assertion can fail.
                    report["trace"][-1] = json_value(row)
                    audits = audit_step(row, expected, held, sources, index + 1, actual_before)
                    actual_before = row["state"]
                    report["trace"][-1].update(audits=json_value(audits), checks_passed=True)
                    if index == 0:
                        old_receipts = plan.last_accepted
                        old_receipt_value = json_value(old_receipts)
                    require(json_value(old_receipts) == old_receipt_value,
                            "previously returned receipt changed after another step")
                    if index + 1 == RESTORE_TICK:
                        checkpoint = row["state"]
                    if index + 1 == RESTORE_TICK + 1:
                        reference_row = json_value(row)
                    previous = {world: (expected[world]["q"], expected[world]["v"])
                                for world in WORLD_ORDER}
                report["checks"]["main_window"] = True
                require(checkpoint is not None and reference_row is not None, "missing replay inputs")
                before_restore = batch.stats()
                report["restore"] = dict(checkpoint=json_value(checkpoint), stats_before=before_restore)
                batch.restore(checkpoint)
                restored = batch.snapshot()
                report["restore"].update(restored=json_value(restored), stats_after=batch.stats())
                require(restored == checkpoint, "restore lost complete state, held input or history")
                require(batch.stats()["state_epoch"] == before_restore["state_epoch"] + 1,
                        "restore did not advance the accepted-state epoch")
                require(json_value(old_receipts) == old_receipt_value, "restore mutated an old receipt")
                report["batch_calls_attempted"] += 1
                replay = dict(step=RESTORE_TICK + 1, accepted=batch.step_contacts(plan, H))
                report["replay_steps_completed"] = 1
                replay.update(capture(batch, plan))
                report["replay"] = json_value(replay)
                # Same backend, complete saved inputs, one deterministic replay. Counters
                # and epochs are intentionally new; physical outputs must be identical.
                for name in ("state", "geometry", "diagnostics", "energy", "receipts", "status"):
                    require(report["replay"][name] == reference_row[name], f"replay changed {name}")
                require(replay["accepted"] == dict.fromkeys(WORLD_ORDER, True), "replay rejected")
                require(json_value(old_receipts) == old_receipt_value, "replay mutated an old receipt")
                owned_arrays(old_receipts)
                report["owned_first_receipts"] = old_receipt_value
                report["checks"]["restore_replay"] = True
                report["checks"]["owned_receipt"] = True
                report["final_stats"] = dict(batch=batch.stats(), plan=plan.stats())
    finally:
        # Resource statistics are observations, not a claim that IO/physical failures
        # rolled back an already accepted step. The main records retain that distinction.
        if batch is not None:
            report["closed_batch_stats"] = batch.stats()
        if plan is not None:
            report["closed_plan_stats"] = plan.stats()
    require(report["closed_batch_stats"]["completed_steps"] == STEPS + 1, "extra or missing batch step")
    require(report["closed_batch_stats"]["accepted_world_steps"] == (STEPS + 1) * len(WORLD_ORDER),
            "extra or missing accepted world step")
    require(report["closed_batch_stats"]["rejected_world_steps"] == 0, "unexpected rejection")
    require(report["closed_batch_stats"]["device_owned_bytes"] == 0, "batch resources still owned")
    require(report["closed_plan_stats"]["closed"], "contact plan was not closed")
    require(report["closed_plan_stats"]["device_owned_bytes"] == 0, "contact scratch still owned")
    require(json_value(old_receipts) == old_receipt_value, "close mutated an old receipt")
    report["checks"]["resource_close"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tyx-mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON manifest path")
    parser.add_argument("--expected-package-root", type=Path,
                        help="optional exact installed phymeo directory supplied by the wheel harness")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = output.with_name(output.stem + ".trace.json.gz")
    report = dict(
        schema="phymeo-c6-wheel-contact-feedback-v1", mode=args.tyx_mode,
        producer_path=str(Path(__file__).resolve()),
        scope="ordinary resident trees; HOST finite frictionless entry contact; no CCD or friction",
        checks=dict(installed_origins=False, main_window=False, restore_replay=False,
                    owned_receipt=False, resource_close=False),
        batch_calls_attempted=0, main_steps_completed=0, replay_steps_completed=0,
        trace=[], execution_failure=None,
        limits="No throughput claim, no warm-readback gate, no fault injection or full C6 matrix rerun.",
    )
    ph = None
    try:
        import phymeo as ph

        report["provenance_before"] = module_provenance(ph, args.expected_package_root)
        run(ph, args.tyx_mode, output.parent / (output.stem + "-cache"), report)
    except Exception as error:
        report["execution_failure"] = dict(type=type(error).__name__, message=str(error),
                                            traceback=traceback.format_exc())
    finally:
        if ph is not None:
            try:
                report["provenance_after"] = module_provenance(ph, args.expected_package_root)
                report["checks"]["installed_origins"] = True
            except Exception as error:
                report["provenance_failure"] = dict(type=type(error).__name__, message=str(error))
        report["passed"] = report["execution_failure"] is None and all(report["checks"].values())
        report["physical_batch_calls_completed"] = (
            report["main_steps_completed"] + report["replay_steps_completed"])
        report["accepted_world_steps_expected_if_passed"] = (STEPS + 1) * len(WORLD_ORDER)
        with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
            json.dump(json_value(report), stream, allow_nan=False)
        manifest = {name: json_value(report.get(name)) for name in (
            "schema", "mode", "backend", "scope", "checks", "passed", "execution_failure",
            "provenance_failure", "batch_calls_attempted", "main_steps_completed", "replay_steps_completed",
            "physical_batch_calls_completed", "accepted_world_steps_expected_if_passed",
            "final_stats", "closed_batch_stats", "closed_plan_stats", "limits",
        )}
        manifest["trace_path"] = str(trace_path)
        manifest["installed_module_count"] = len(report.get("provenance_after", {}).get("modules", []))
        output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(manifest, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
