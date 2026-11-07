"""C9 installed-wheel public consumer: W4, 40 steps, full 19->21 replay.

Self-contained P03 recoil/P02 eccentric origin-force, ordinary P and FreeSphere.
Finite S/C/H bounds use original points and independent poses. No repository
imports, private advance calls, contact-response claim or parameter adaptation.
Run a copied file with python -I -B; all paths must name the new wheel install.
"""

import argparse
import gzip
import hashlib
import json
import math
import sys
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from importlib.metadata import distribution
from itertools import combinations, product
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname
from zipfile import ZipFile

import numpy as np

H, STEPS, REPLAY_FROM = 1 / 1024, 40, 19
ATOL, RTOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-4
MAX_BYTES = 256 * 1024 * 1024
ORDER = (1207, 6, 91, 503)
FLOATING = (503, 91)
TREE_WORLDS = (503, 91, 1207)


def encode(value):
    if is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return encode(value.tolist())
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def fingerprint(value):
    return json.dumps(encode(value), sort_keys=True, allow_nan=False, separators=(",", ":"))


def close(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


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


def exp_drift(q, omega):
    magnitude = np.linalg.norm(omega)
    delta = (
        np.r_[omega * (math.sin(H * magnitude / 2) / magnitude), math.cos(H * magnitude / 2)]
        if magnitude
        else np.array([0.0, 0, 0, 1])
    )
    result = qmul(delta, q)
    result /= np.linalg.norm(result)
    return -result if result @ q < 0 else result


def readonly(value):
    if isinstance(value, np.ndarray):
        assert not value.flags.writeable
        try:
            value.setflags(write=True)
        except ValueError:
            pass
        else:
            raise AssertionError("public owned array can be made writable")
    elif is_dataclass(value):
        for f in fields(value):
            readonly(getattr(value, f.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            readonly(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            readonly(item)


def declarations(ph):
    child = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0, 0, 1]]),
        np.array([[1.0, 0, 0]]),
        np.array([1.0]),
        np.zeros((1, 3)),
        np.full((1, 3), 0.25),
        np.zeros(3),
    )
    root = ph.RigidBodyModel(np.array([0.5]), np.full((1, 3), 0.5), np.zeros(3))
    recoil = ph.FloatingRobotTree(
        ph.FloatingRobotModel(root, np.zeros(3), child),
        ph.FloatingRobotState(np.array([0.0, 0, 0, 0, 0, 0, 1, 0]), np.zeros(7)),
        kp=np.zeros(1),
        kd=np.zeros(1),
        effort_limit=np.array([8.0]),
    )
    eccentric_root = ph.RigidBodyModel(
        np.array([0.5]), np.array([[1 / 4, 3 / 8, 1 / 2]]), np.zeros(3)
    )
    eccentric = ph.FloatingRobotTree(
        ph.FloatingRobotModel(eccentric_root, np.array([1 / 8, 0, 0])),
        ph.FloatingRobotState(np.array([0.0, 0, 0, 0, 0, 0, 1]), np.zeros(6)),
        kp=np.zeros(0),
        kd=np.zeros(0),
        effort_limit=np.zeros(0),
    )
    ordinary_model = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.array([[1.0, 0, 0]]),
        np.array([[0.0, 0, 0, 1]]),
        np.array([[0.0, 1, 0]]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.full((1, 3), 1 / 8),
        np.zeros(3),
    )
    ordinary_program = ph.DrivenRobotProgram(
        ordinary_model, kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.array([8.0])
    )
    ordinary = ph.FixedRobotTree(
        ordinary_program,
        ordinary_program.initial_state(ph.RobotState(np.array([1 / 8]), np.array([1 / 16]))),
    )
    ordinary_command = ordinary_program.inputs(
        np.zeros(1),
        np.zeros(1),
        feedforward=np.zeros(1),
        loads=ph.RobotInputs(np.zeros(1), np.array([[0.0, 2, 0]]), np.zeros((1, 3))),
    )
    worlds = {
        503: recoil,
        91: eccentric,
        1207: ordinary,
        6: ph.FreeSphere(
            2, 1 / 16, position=(0, 0, 2), velocity=(1 / 16, 0, 0), gravity=(0, 0, -2)
        ),
    }
    original = {
        503: dict(
            model=recoil.model, state=recoil.state, kp=[0], kd=[0], cap=[8], delay=0, motor=[3]
        ),
        91: dict(
            model=eccentric.model,
            state=eccentric.state,
            kp=[],
            kd=[],
            cap=[],
            delay=0,
            origin_force_world_n=[0, 2, 0],
            rule="tauCOM=(-R*localCOM) cross F at this accepted entry",
        ),
        1207: dict(
            model=ordinary_model, initial_q=[1 / 8], initial_v=[1 / 16], command=ordinary_command
        ),
        6: dict(
            mass_kg=2,
            radius_m=1 / 16,
            position_m=[0, 0, 2],
            velocity_m_s=[1 / 16, 0, 0],
            gravity_m_s2=[0, 0, -2],
        ),
    }
    return worlds, ordinary_command, original


def commands(ph, worlds, ordinary, snapshot):
    torque = np.cross(-rot(snapshot[91].position[3:7]) @ np.array([1 / 8, 0, 0]), [0, 2, 0])
    external = ph.FloatingRobotInputs(np.zeros(0), np.array([[0.0, 2, 0]]), torque[None])
    return {
        503: worlds[503].inputs(np.array([[0.0, 0, 3.0]])),
        91: worlds[91].inputs(np.zeros((0, 3)), external),
        1207: ordinary,
        6: (0.0, 0.0, 0.0),
    }


def bounds_model(ph, checkpoints):
    shapes, bindings, original, excluded = [], [], {}, []
    for world in ORDER:
        dynamic = []
        if world in FLOATING:
            for shape_id, kind, local, radius, group, ids in (
                (11, "sphere", np.array([[0.0, 1 / 8, 0]]), 1 / 32, 0, (101,)),
                (
                    12,
                    "capsule",
                    np.array([[0.0, 0, -1 / 8], [0, 0, 1 / 8]]),
                    1 / 64,
                    int(world == 503),
                    (207, 211),
                ),
                (
                    13,
                    "hull",
                    np.array(list(product((-1 / 16, 1 / 16), (-1 / 32, 1 / 32), (-1 / 64, 1 / 64))))
                    + [1 / 4, 0, 0],
                    0.0,
                    0,
                    (301, 302, 303, 304, 305, 306, 307, 308),
                ),
            ):
                key = ph.CollisionKey(world, 100 + group, shape_id)
                shape = ph.CollisionShape(key, kind, ids, radius=radius, revision=7)
                shapes.append(shape)
                bindings.append(
                    ph.ResidentFloatingTreeBinding(
                        key, world, group, local, topology=checkpoints[world].topology
                    )
                )
                original[key] = (shape, group, local)
                dynamic.append(key)
        else:
            key = ph.CollisionKey(world, 100, 11)
            shape = ph.CollisionShape(key, "sphere", (101,), radius=1 / 16, revision=7)
            shapes.append(shape)
            if world == 6:
                bindings.append(ph.ResidentSphereBinding(key, world))
            else:
                bindings.append(
                    ph.ResidentTreeBinding(
                        key, world, 0, np.zeros((1, 3)), topology=checkpoints[world].topology
                    )
                )
            original[key] = (shape, 0, np.zeros((1, 3)))
            dynamic.append(key)
        # The default static-only domain requires every dynamic pair to be
        # explicitly excluded, including two shapes of the same physical body.
        for a, b in combinations(dynamic, 2):
            excluded.append((a, b))
        center = (
            np.array([0.0, 0, 2])
            if world == 6
            else np.array([1.0, 1 / 8, 0])
            if world == 1207
            else np.zeros(3)
        )
        for body, point, radius in ((900, center, 1 / 4), (901, center + [2.0, 0, 0], 1 / 16)):
            key = ph.CollisionKey(world, body, 41)
            shape = ph.CollisionShape(key, "sphere", (901,), radius=radius)
            shapes.append(shape)
            bindings.append(ph.ResidentStaticBinding(key, ph.CollisionMotion(point[None])))
            original[key] = (shape, None, point[None])
    return (
        ph.CollisionModel(tuple(reversed(shapes)), tuple(excluded)),
        tuple(reversed(bindings)),
        original,
    )


def broad(result):
    return {
        name: getattr(result, name)
        for name in (
            "topology_generation",
            "state_epoch",
            "world_keys",
            "ticks",
            "shape_keys",
            "shape_revisions",
            "bounds",
            "registered_pairs",
            "candidates",
        )
    }


def capture(batch, plan):
    return dict(
        state=batch.snapshot(),
        geometry=batch.tree_kinematics(),
        diagnostics=batch.tree_diagnostics(),
        energy=batch.summary(),
        broad=broad(plan.query()),
        batch_stats=batch.stats(),
        plan_stats=plan.stats(),
    )


def mechanics(world, q, v):
    q, v = np.asarray(q, float), np.asarray(v, float)
    if world == 503:
        points = np.array([q[:3], q[:3] + [q[7], 0, 0]])
        linear, angular = np.zeros((2, 3, 7)), np.zeros((2, 3, 7))
        for i in range(2):
            linear[i, :, :3] = np.eye(3)
            linear[i, :, 3:6] = -skew(points[i] - points[0])
            angular[i, :, 3:6] = np.eye(3)
        linear[1, :, 6] = (1, 0, 0)
        mass = (
            2 * linear[0].T @ linear[0]
            + 0.5 * angular[0].T @ angular[0]
            + linear[1].T @ linear[1]
            + 0.25 * angular[1].T @ angular[1]
        )
        external, acceleration = np.zeros(7), np.array([-1.5, 0, 0, 0, 0, 0, 4.5])
        return dict(
            mass=mass,
            bias=np.zeros(7),
            external=external,
            drive=np.array([3.0]),
            acceleration=acceleration,
            linear=linear,
            angular=angular,
            points=points,
            centers=points,
            rotations=np.tile(np.eye(3), (2, 1, 1)),
            com_velocity=linear @ v,
            omega=angular @ v,
            kinetic_energy_j=float(v @ mass @ v / 2),
        )
    if world == 91:
        r = rot(q[3:7])
        center = q[:3] + r @ np.array([1 / 8, 0, 0])
        inertia = r @ np.diag([1 / 4, 3 / 8, 1 / 2]) @ r.T
        mass = np.zeros((6, 6))
        mass[:3, :3] = 2 * np.eye(3)
        mass[3:, 3:] = inertia
        torque = np.cross(-r @ np.array([1 / 8, 0, 0]), [0, 2, 0])
        external = np.r_[0.0, 2, 0, torque]
        bias = np.r_[np.zeros(3), np.cross(v[3:], inertia @ v[3:])]
        acceleration = np.r_[0.0, 1, 0, np.linalg.solve(inertia, torque - bias[3:])]
        return dict(
            mass=mass,
            bias=bias,
            external=external,
            drive=np.zeros(0),
            acceleration=acceleration,
            linear=np.eye(6)[None, :3],
            angular=np.eye(6)[None, 3:],
            points=q[None, :3],
            centers=center[None],
            rotations=r[None],
            com_velocity=v[None, :3],
            omega=v[None, 3:],
            kinetic_energy_j=float(v @ mass @ v / 2),
        )
    if world == 1207:
        return dict(
            acceleration=np.array([1.0]),
            points=np.array([[1.0, q[0], 0]]),
            rotations=np.eye(3)[None],
            kinetic_energy_j=float(v[0] * v[0]),
        )
    return dict(acceleration=np.array([0.0, 0, -2.0]), kinetic_energy_j=float(v @ v))


def prediction(world, q, v):
    q, v = np.asarray(q, float), np.asarray(v, float)
    truth = mechanics(world, q, v)
    vp = v + H * truth["acceleration"]
    if world in FLOATING:
        qp = q.copy()
        qp[3:7] = exp_drift(q[3:7], vp[3:6])
        local_com = np.zeros(3) if world == 503 else np.array([1 / 8, 0, 0])
        qp[:3] = q[:3] + H * vp[:3] - (rot(qp[3:7]) - rot(q[3:7])) @ local_com
        qp[7:] += H * vp[6:]
    else:
        qp = q + H * vp
    return qp, vp


def geometry_audit(row, sources, model):
    result = row["broad"]
    assert set(result["world_keys"]) == set(ORDER)
    assert tuple(result["shape_keys"]) == tuple(shape.key for shape in model.shapes)
    registered = set(result["registered_pairs"])
    assert all(a.world == b.world and a.body != b.body for a, b in registered)
    assert not set(model.excluded) & registered
    wanted = {}
    for key, (shape, group, local) in sources.items():
        state = row["state"][key.world]
        if group is None:
            points = local
        elif key.world == 6:
            points = np.asarray(state.position)[None] + local
        else:
            truth = mechanics(key.world, state.position, state.velocity)
            points = truth["points"][group] + local @ truth["rotations"][group].T
        lo, hi = points.min(0) - shape.radius, points.max(0) + shape.radius
        at = result["shape_keys"].index(key)
        actual = result["bounds"][at]
        assert np.isfinite(actual).all() and np.all(actual[:3] <= lo) and np.all(actual[3:] >= hi)
        assert result["shape_revisions"][at] == shape.revision
        wanted[key] = (lo, hi)
    true_pairs = {
        pair
        for pair in registered
        if np.all(wanted[pair[0]][0] <= wanted[pair[1]][1])
        and np.all(wanted[pair[1]][0] <= wanted[pair[0]][1])
    }
    assert true_pairs <= set(result["candidates"]), "missing a true original-material AABB pair"
    assert not any(a.body == 901 or b.body == 901 for a, b in result["candidates"]), (
        "far static sphere must stay a miss"
    )
    assert true_pairs, "independent finite candidates disappeared"
    for world, tick in zip(result["world_keys"], result["ticks"], strict=True):
        assert tick == row["state"][world].tick
    row["independent_material_bounds"] = wanted
    row["independent_true_pairs"] = sorted(true_pairs)


def step_audit(entry, row, expected, acceptance, sources, model):
    assert row["acceptance"] == acceptance
    for world in ORDER:
        state = row["state"][world]
        if not acceptance[world]:
            assert state == entry[world]
            assert row["diagnostics"].get(world) is None
            continue
        assert state.tick == entry[world].tick + 1
        close(state.position, expected[world][0], atol=WINDOW_ATOL)
        close(state.velocity, expected[world][1], atol=WINDOW_ATOL)
        assert state.control == entry[world].control and state.history == entry[world].history
        truth = mechanics(world, entry[world].position, entry[world].velocity)
        end = mechanics(world, state.position, state.velocity)
        close(row["energy"][world]["kinetic_energy_j"], end["kinetic_energy_j"])
        if world in FLOATING:
            diag, kin = row["diagnostics"][world], row["geometry"][world]
            assert (diag.entry_tick, diag.exit_tick, diag.dt_s) == (
                entry[world].tick,
                state.tick,
                H,
            )
            close(diag.entry_position, entry[world].position)
            close(diag.entry_velocity, entry[world].velocity)
            for field, desired in (
                ("mass", truth["mass"]),
                ("bias", truth["bias"]),
                ("external", truth["external"]),
                ("drive", truth["drive"]),
                ("acceleration", truth["acceleration"]),
                ("linear_jacobian", truth["linear"]),
                ("angular_jacobian", truth["angular"]),
            ):
                close(getattr(diag, field), desired, atol=2e-5 if field == "mass" else ATOL)
            assert diag.control_residual <= 5e-5 and diag.equation_residual <= 2e-5
            load = truth["external"].copy()
            load[6:] += truth["drive"]
            close(state.generalized_applied, load)
            close(state.applied_effort, truth["drive"])
            close(state.local_origin_m, state.position[:3], atol=0, rtol=0)
            close(kin.position, state.position)
            close(kin.velocity, state.velocity)
            close(kin.link_position_m, end["points"])
            close(kin.com_position_m, end["centers"])
            close(kin.link_rotation, end["rotations"])
            close(kin.linear_velocity_m_s, end["com_velocity"])
            close(kin.angular_velocity_rad_s, end["omega"])
            readonly(kin)
            readonly(diag)
        elif world == 1207:
            close(row["geometry"][world].link_position_m, end["points"])
            np.testing.assert_array_equal(state.control, [[0, 0, 0, 0, 0, 2, 0, 0, 0, 0]])
            close(row["diagnostics"][world].acceleration, [1.0])
            close(state.generalized_applied, [2.0])
        else:
            close(state.applied_effort, [0.0, 0, 0])
    geometry_audit(row, sources, model)
    readonly(row["broad"]["bounds"])
    row["checks_complete"] = True


def rejection(call, name):
    try:
        call()
    except (ValueError, TypeError, RuntimeError) as error:
        return dict(name=name, type=type(error).__name__, message=str(error))
    raise AssertionError(name + " unexpectedly succeeded")


def run(ph, mode, output, record):
    worlds, ordinary, declared = declarations(ph)
    batch = ph.TyxWorldBatch(
        [(world, worlds[world]) for world in ORDER], mode, cache_root=output / "cache"
    )
    plan = None
    record.update(declarations=declared, trace=[], replay=[], faults=[], checks={})
    try:
        initial = batch.snapshot()
        batch.set_controls(commands(ph, worlds, ordinary, initial))
        initial = batch.snapshot()
        model, bindings, sources = bounds_model(ph, initial)
        record.update(initial=initial, model=model, bindings=bindings, original_sources=sources)
        plan = batch.prepare_collision(model, bindings)
        record["initial_observation"] = capture(batch, plan)
        geometry_audit(record["initial_observation"], sources, model)
        root_key, static_key = ph.CollisionKey(91, 100, 11), ph.CollisionKey(91, 900, 41)
        pair_model = ph.CollisionModel(
            tuple(s for s in model.shapes if s.key in (root_key, static_key))
        )
        pair_bindings = tuple(b for b in bindings if b.key in (root_key, static_key))
        pair_law = {(root_key, static_key): ph.RobotContactMaterial(2000, 0, 0)}
        record["floating_contacts_rejected"] = rejection(
            lambda: batch.prepare_contacts(pair_model, pair_bindings, pair_law),
            "floating contact binding",
        )
        assert (
            "fixed-root" in record["floating_contacts_rejected"]["message"]
            or "floating" in record["floating_contacts_rejected"]["message"]
        )
        assert batch.snapshot() == initial
        expected = {
            w: (np.asarray(s.position, float), np.asarray(s.velocity, float))
            for w, s in initial.items()
        }
        retained = record["initial_observation"]
        retained_bytes = fingerprint(retained)
        for index in range(STEPS):
            before_control = batch.snapshot()
            packet = commands(ph, worlds, ordinary, before_control)
            batch.set_controls(packet)
            entry = batch.snapshot()
            expected = {w: prediction(w, *expected[w]) for w in ORDER}
            row = dict(
                input_packet=packet,
                entry=entry,
                independent_predicted=expected,
                checks_complete=False,
            )
            record["trace"].append(row)
            row["acceptance"] = batch.step(H)
            row.update(capture(batch, plan))
            step_audit(entry, row, expected, dict.fromkeys(ORDER, True), sources, model)
            assert fingerprint(retained) == retained_bytes
            if index + 1 == REPLAY_FROM:
                record["checkpoint_19"] = batch.snapshot()
        record["checks"]["W4_continuous_40_independent_physics"] = True
        before = batch.snapshot()
        old_result = plan.query()
        old_epoch = batch.stats()["state_epoch"]
        batch.restore(record["checkpoint_19"])
        assert batch.snapshot() == record["checkpoint_19"]
        assert batch.stats()["state_epoch"] == old_epoch + 1
        record["stale_query_rejected"] = rejection(
            lambda: plan.require_current(old_result), "pre-restore epoch"
        )
        record["restored"] = capture(batch, plan)
        for index in range(REPLAY_FROM, STEPS):
            packet = commands(ph, worlds, ordinary, batch.snapshot())
            batch.set_controls(packet)
            row = dict(input_packet=packet, entry=batch.snapshot(), checks_complete=False)
            record["replay"].append(row)
            row["acceptance"] = batch.step(H)
            row.update(capture(batch, plan))
            prior = record["trace"][index]
            assert row["acceptance"] == dict.fromkeys(ORDER, True)
            for key in ("input_packet", "entry", "state", "geometry", "diagnostics", "energy"):
                assert fingerprint(row[key]) == fingerprint(prior[key]), (
                    "full replay differs: " + key
                )
            for key in (
                "world_keys",
                "ticks",
                "shape_keys",
                "shape_revisions",
                "bounds",
                "registered_pairs",
                "candidates",
            ):
                assert fingerprint(row["broad"][key]) == fingerprint(prior["broad"][key]), (
                    "geometry replay differs: " + key
                )
            # Epoch/counters are monotonic observations, not checkpoint state.
            geometry_audit(row, sources, model)
            row["checks_complete"] = True
        assert batch.snapshot() == before
        record["checks"]["full_19_to_21_replay"] = True
        baseline = batch.snapshot()
        for bank in (0, 1):
            batch.restore(baseline)
            if batch.stats()["current_buffer"] != bank:
                batch.restore(baseline)
            entry = batch.snapshot()
            frame = capture(batch, plan)
            epoch = batch.stats()["state_epoch"]
            fault = dict(target_bank=bank, entry=frame, checks_complete=False)
            record["faults"].append(fault)
            wrong = dict(entry)
            wrong[503] = replace(
                wrong[503],
                local_origin_m=(4097.0, 0, 0),
                position=(4097.0, *wrong[503].position[1:]),
            )
            fault["invalid_restore_input"] = wrong
            fault["invalid_restore"] = rejection(
                lambda: batch.restore(wrong), "out-of-domain root checkpoint"
            )
            # A valid packet from the other root-only model has the wrong n here.
            bad_commands = {6: (1.0, 0, 0), 503: worlds[91].inputs(np.zeros((0, 3)))}
            fault["invalid_control_input"] = bad_commands
            fault["invalid_control"] = rejection(
                lambda: batch.set_controls(bad_commands), "mixed wrong-size control"
            )
            assert batch.snapshot() == entry and batch.stats()["state_epoch"] == epoch
            assert batch.stats()["current_buffer"] == bank
            assert fingerprint(batch.tree_kinematics()) == fingerprint(frame["geometry"])
            boundary = dict(entry)
            v = np.asarray(boundary[503].velocity).copy()
            v[:3] = (1.0, 0, 0)
            boundary[503] = replace(
                boundary[503],
                local_origin_m=(4096.0, 0, 0),
                position=(4096.0, 0.0, 0.0, *boundary[503].position[3:]),
                velocity=tuple(v),
            )
            batch.restore(boundary)
            if batch.stats()["current_buffer"] != bank:
                batch.restore(boundary)
            batch.set_controls(commands(ph, worlds, ordinary, batch.snapshot()))
            boundary = batch.snapshot()
            fault["domain_entry"] = capture(batch, plan)
            fault["domain_acceptance"] = batch.step(H)
            fault["domain_exit"] = capture(batch, plan)
            assert fault["domain_acceptance"] == {503: False, 91: True, 1207: True, 6: True}
            assert fault["domain_exit"]["state"][503] == boundary[503]
            assert fault["domain_exit"]["diagnostics"][503] is None
            for key in ("geometry", "energy"):
                assert fingerprint(fault["domain_exit"][key][503]) == fingerprint(
                    fault["domain_entry"][key][503]
                )
            assert batch.stats()["floating_tree_status_codes"][503] == "STATE_DOMAIN"
            for world in (91, 1207, 6):
                want_q, want_v = prediction(
                    world, boundary[world].position, boundary[world].velocity
                )
                close(fault["domain_exit"]["state"][world].position, want_q, atol=WINDOW_ATOL)
                close(fault["domain_exit"]["state"][world].velocity, want_v, atol=WINDOW_ATOL)
                assert fault["domain_exit"]["state"][world].tick == boundary[world].tick + 1
            retry = dict(batch.snapshot())
            v = np.asarray(retry[503].velocity).copy()
            v[:3] = (-1.0, 0, 0)
            retry[503] = replace(retry[503], velocity=tuple(v))
            batch.restore(retry)
            batch.set_controls(commands(ph, worlds, ordinary, batch.snapshot()))
            retry = batch.snapshot()
            fault["retry_entry"] = retry
            fault["retry_acceptance"] = batch.step(H)
            fault["retry_exit"] = capture(batch, plan)
            assert fault["retry_acceptance"] == dict.fromkeys(ORDER, True)
            assert fault["retry_exit"]["state"][503].local_origin_m[0] < 4096.0
            for world in ORDER:
                want_q, want_v = prediction(world, retry[world].position, retry[world].velocity)
                close(fault["retry_exit"]["state"][world].position, want_q, atol=WINDOW_ATOL)
                close(fault["retry_exit"]["state"][world].velocity, want_v, atol=WINDOW_ATOL)
                assert fault["retry_exit"]["state"][world].tick == retry[world].tick + 1
            assert fingerprint(retained) == retained_bytes
            fault["checks_complete"] = True
        record["checks"]["both_banks_invalid_and_retained_domain_retry"] = True
        record["checks"]["owned_old_observations_unchanged"] = True
        readonly(retained["geometry"])
        readonly(retained["broad"]["bounds"])
        record["checks"]["floating_contact_remains_unsupported"] = True
    finally:
        try:
            record["retained_at_exit"] = batch.snapshot()
        except Exception as error:
            record["retained_read_error"] = dict(type=type(error).__name__, message=str(error))
        record["before_close_stats"] = batch.stats()
        if plan is not None:
            record["before_close_plan_stats"] = plan.stats()
        batch.close()
        batch.close()
        record["after_close_stats"] = batch.stats()
        record["after_close_plan_stats"] = None if plan is None else plan.stats()
    assert batch.stats()["device_owned_bytes"] == 0
    assert plan.stats()["closed"] and plan.stats()["device_owned_bytes"] == 0
    record["closed_rejection"] = rejection(lambda: batch.step(H), "closed batch")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel-site", type=Path, required=True)
    parser.add_argument("--expected-package-root", type=Path, required=True)
    parser.add_argument("--expected-wheel", type=Path, required=True)
    parser.add_argument("--forbidden-checkout", type=Path)
    args = parser.parse_args()
    assert __debug__ and sys.flags.isolated and sys.flags.dont_write_bytecode, "use python -I -B"
    for path in (args.output, args.wheel_site, args.expected_package_root, args.expected_wheel):
        if not path.is_absolute():
            raise ValueError("all installation and output paths must be absolute")
    site, package = (
        args.wheel_site.resolve(strict=True),
        args.expected_package_root.resolve(strict=True),
    )
    script, output = Path(__file__).resolve(strict=True), args.output.resolve()
    assert package == (site / "phymeo").resolve(strict=True)
    assert Path.cwd().resolve() == script.parent
    if args.forbidden_checkout is not None:
        checkout = args.forbidden_checkout.resolve()
        assert not any(p.is_relative_to(checkout) for p in (script, site, output))
    assert "phymeo" not in sys.modules
    output.mkdir(parents=True, exist_ok=False)
    record = dict(
        status="running",
        mode=args.mode,
        world_order=ORDER,
        errors=[],
        checks={},
        script=dict(path=str(script), sha256=hashlib.sha256(script.read_bytes()).hexdigest()),
        expected_package_root=str(package),
        gates=dict(
            h_s=H,
            steps=40,
            replay_from=19,
            replay_steps=21,
            transaction_banks=(0, 1),
            state_atol=ATOL,
            state_rtol=RTOL,
            window_atol=WINDOW_ATOL,
            mass_atol=2e-5,
            equation_residual=2e-5,
            control_residual=5e-5,
        ),
        units=dict(
            configuration="root xyz m, body-to-world XYZW, child rad/m",
            velocity="root body COM world m/s, world rad/s, child rad/s or m/s",
            force="N",
            torque="N*m about actual COM",
            time="s",
            energy="J",
        ),
        limits="Installed-wheel consumer, not a performance benchmark or full C9 matrix. Floating ordinary root/tree current-pose bounds only; floating contact, affine floating and CCD remain unsupported.",
    )
    sys.path.insert(0, str(site))
    try:
        import phymeo as ph

        assert Path(ph.__file__).resolve(strict=True).parent == package
        installed = distribution("phymeo")
        assert Path(installed.locate_file("")).resolve(strict=True) == site
        record["wheel"] = archive_identity(installed, args.expected_wheel)
        record["modules_before"] = module_provenance(site, package, args.expected_wheel)
        record["versions"] = dict(
            phymeo=installed.version,
            tyx=distribution("tyx-compute").version,
            numpy=np.__version__,
            python=sys.version,
        )
        run(ph, args.mode, output, record)
        record["status"] = "passed"
    except Exception as error:
        record["status"] = "failed"
        record["errors"].append(
            dict(
                stage="import_or_window",
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            )
        )
    try:
        if "phymeo" in sys.modules:
            record["modules_after"] = module_provenance(site, package, args.expected_wheel)
    except Exception as error:
        record["status"] = "failed"
        record["errors"].append(
            dict(
                stage="module_provenance",
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            )
        )
    encoded = json.dumps(encode(record), allow_nan=False, separators=(",", ":")).encode()
    if len(encoded) > MAX_BYTES:
        raise RuntimeError("complete record exceeds predeclared 256 MiB limit; no partial success")
    path = output / "resident-floating-wheel.trace.json.gz"
    temporary = path.with_name(path.name + ".tmp")
    # IO failure cannot roll back completed physical steps; it propagates as IO failure.
    with gzip.open(temporary, "wb") as stream:
        stream.write(encoded)
    temporary.replace(path)
    summary = dict(
        status=record["status"],
        raw_exit=int(record["status"] != "passed"),
        mode=args.mode,
        complete_forward_steps=sum(bool(r.get("checks_complete")) for r in record.get("trace", ())),
        complete_replay_steps=sum(bool(r.get("checks_complete")) for r in record.get("replay", ())),
        complete_fault_banks=sum(bool(r.get("checks_complete")) for r in record.get("faults", ())),
        checks=record["checks"],
        wheel=record.get("wheel"),
        verified_module_count=len(record.get("modules_after", ())),
        trace=dict(
            path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            uncompressed_bytes=len(encoded),
        ),
        batch_stats=record.get("after_close_stats"),
        plan_stats=record.get("after_close_plan_stats"),
        errors=record["errors"],
        limits=record["limits"],
    )
    (output / "summary.json").write_text(
        json.dumps(encode(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(encode(summary), allow_nan=False))
    return summary["raw_exit"]


if __name__ == "__main__":
    raise SystemExit(main())
