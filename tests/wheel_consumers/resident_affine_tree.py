"""Standalone C5 wheel smoke, not a benchmark or a replacement for A01--A08.

Copy this file outside the checkout; run with python -I -B against a separately
installed local wheel. Only public ``phymeo`` symbols, NumPy and the standard
library are used. No repository fixture, private backend or Newton is imported.

Five mixed worlds: original A03 chain, A04 coupled 2R, N9/K3 fan-out, the C4
ordinary 2R and FreeSphere controls. Forty continuous steps use h=1/1024.
A same-tick restore and a tick-19 checkpoint replay preserve the original held
commands; the last 21 steps must reproduce saved complete accepted states.
Geometry is current-pose broad phase only, with no contact force or CCD claim.
"""

import argparse
import copy
import gzip
import hashlib
import json
import sys
import traceback
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from importlib.metadata import distribution
from itertools import combinations, product
from math import cos, pi, sin
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile

import numpy as np

H = 1 / 1024
STEPS = 40
REPLAY_FROM = 19
ATOL = 5e-6
RTOL = 5e-5
WINDOW_ATOL = 2e-4
CONSTRAINT_ATOL = 2e-5
WORLD_ORDER = (117, 7, 53, 23, 91)
AFFINE_KEYS = (53, 91, 117)
TREE_KEYS = (53, 91, 117, 23)


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
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def close(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def rotation_z(angle):
    c, s = cos(angle), sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def quaternion_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def box(low, high):
    return np.array(list(product(*zip(low, high, strict=True))), dtype=float)


def declaration(name):
    """Independent frozen physical constants, not copied from loaded products."""
    if name in ("A04", "ordinary-2R"):
        n = 2
        model = dict(
            parents=(-1, 0),
            kinds=("revolute",) * 2,
            offsets=np.array([[0.0, 0, 0], [1.0, 0, 0]]),
            axes=np.array([[0.0, 0, 1]] * 2),
            mass=np.ones(2),
            com=np.array([[0.5, 0, 0]] * 2),
            inertia=np.array([[1 / 24, 1 / 24, 1 / 12]] * 2),
        )
        if name == "A04":
            P, c = np.array([[1.0], [-2.0]]), np.array([0.0, pi / 2])
            rows, independent = ((0, 1, -2.0, pi / 2),), (0,)
            s, u, motor = np.array([0.0]), np.array([0.5]), np.array([1.0])
        else:
            P, c, rows, independent = np.eye(2), np.zeros(2), (), (0, 1)
            s, u, motor = np.array([0.37, -0.51]), np.array([0.5, -0.25]), np.zeros(2)
    else:
        assert name in ("A03", "A03-flat", "N9")
        n, k = (9, 3) if name == "N9" else (3, 1)
        mass = [1 + i / 8 for i in range(n)] if n == 9 else [2, 1, 1]
        offsets = (
            [[0, i / 8, 0] for i in range(n)] if n == 9 else [[0, 0, 0], [0, 0.5, 0], [0, -0.5, 0]]
        )
        model = dict(
            parents=(-1,) * n,
            kinds=("prismatic",) * n,
            offsets=np.array(offsets, dtype=float),
            axes=np.array([[1.0, 0, 0]] * n),
            mass=np.array(mass, dtype=float),
            com=np.zeros((n, 3)),
            inertia=np.array([[1 / 8] * 3] * n),
        )
        P, c = np.zeros((n, k)), np.zeros(n)
        P[:k] = np.eye(k)
        independent = tuple(range(k))
        if n == 3:
            P[:, 0], c[:] = [1, 2, -1], [0, 1 / 8, 1 / 8]
            rows = (
                ((0, 1, 2.0, 1 / 8), (1, 2, -0.5, 3 / 16))
                if name == "A03"
                else ((0, 1, 2.0, 1 / 8), (0, 2, -1.0, 1 / 8))
            )
            s, u, motor = np.zeros(1), np.zeros(1), np.array([7.0])
        else:
            rows = tuple((i % k, i, (-1) ** i * (1 + (i % 3) / 2), i / 64) for i in range(k, n))
            for driver, follower, multiplier, offset in rows:
                P[follower, driver], c[follower] = multiplier, offset
            s = np.array([(j + 1) / 64 for j in range(k)], dtype=float)
            u = np.array([(j - 1) / 32 for j in range(k)], dtype=float)
            motor = np.array([1 + j / 8 for j in range(k)], dtype=float)
    C = np.zeros((len(rows), n))
    for i, (driver, follower, multiplier, _) in enumerate(rows):
        C[i, driver], C[i, follower] = -multiplier, 1
    model.update(
        gravity=np.zeros(3),
        names=tuple(f"link{i}" for i in range(n)),
        joint_names=tuple(f"joint{i}" for i in range(n)),
    )
    effort = np.array([1.0, 1 / 6]) if name == "ordinary-2R" else np.zeros(n)
    return dict(
        name=name,
        model=model,
        P=P,
        c=c,
        C=C,
        rows=rows,
        independent=independent,
        s=s,
        u=u,
        motor=motor,
        effort=effort,
        force=np.zeros((n, 3)),
        torque=np.zeros((n, 3)),
        lower=None if name == "ordinary-2R" else np.full(n, -8.0),
        upper=None if name == "ordinary-2R" else np.full(n, 8.0),
        speed=None if name == "ordinary-2R" else np.full(n, 16.0),
        rating=None if name == "ordinary-2R" else np.full(n, 16.0),
        kp=np.zeros(len(s)),
        kd=np.zeros(len(s)),
        cap=np.full(len(s), 8.0 if name == "ordinary-2R" else 16.0),
        delay_steps=0,
        control_max_iterations=512,
    )


def numbers(values):
    return " ".join(format(float(value), ".17g") for value in values)


def urdf_text(data):
    model = data["model"]
    followers = {f: (d, a, b) for d, f, a, b in data["rows"]}
    root = ET.Element("robot", name="resident_affine_frozen")
    ET.SubElement(root, "link", name="base")
    for i, name in enumerate(model["names"]):
        link = ET.SubElement(root, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=numbers(model["com"][i]))
        ET.SubElement(inertial, "mass", value=str(float(model["mass"][i])))
        x, y, z = model["inertia"][i]
        ET.SubElement(
            inertial, "inertia", ixx=str(x), iyy=str(y), izz=str(z), ixy="0", ixz="0", iyz="0"
        )
        collider = ET.SubElement(link, "collision", name="material")
        ET.SubElement(collider, "origin", xyz="0.25 -0.125 0", rpy="0 0 0.7853981633974483")
        ET.SubElement(ET.SubElement(collider, "geometry"), "sphere", radius="0.0625")
        joint = ET.SubElement(root, "joint", name=model["joint_names"][i], type=model["kinds"][i])
        parent = model["parents"][i]
        ET.SubElement(joint, "parent", link="base" if parent < 0 else model["names"][parent])
        ET.SubElement(joint, "child", link=name)
        ET.SubElement(joint, "origin", xyz=numbers(model["offsets"][i]), rpy="0 0 0")
        ET.SubElement(joint, "axis", xyz=numbers(model["axes"][i]))
        ET.SubElement(
            joint,
            "limit",
            lower=str(data["lower"][i]),
            upper=str(data["upper"][i]),
            velocity=str(data["speed"][i]),
            effort=str(data["rating"][i]),
        )
        if i in followers:
            driver, multiplier, offset = followers[i]
            ET.SubElement(
                joint,
                "mimic",
                joint=model["joint_names"][driver],
                multiplier=str(multiplier),
                offset=str(offset),
            )
    return ET.tostring(root, encoding="unicode")


def physical_terms(data, q, v):
    """Closed planar 2R / independent parallel-slide mechanics and all COM J.

    These are explicit fixture equations. No Phymeo FK/dynamics or finite
    differences enter this oracle; world-space material transforms are separate.
    """
    model = data["model"]
    q, v = np.asarray(q, dtype=float), np.asarray(v, dtype=float)
    n = len(q)
    linear, angular = np.zeros((n, 3, n)), np.zeros((n, 3, n))
    if model["kinds"][0] == "prismatic":
        positions = model["offsets"] + q[:, None] * np.array([1.0, 0, 0])
        rotations = np.tile(np.eye(3), (n, 1, 1))
        com = positions.copy()
        for i in range(n):
            linear[i, 0, i] = 1
        mass, bias = np.diag(model["mass"]), np.zeros(n)
    else:
        assert n == 2
        r1 = np.array([cos(q[0]), sin(q[0]), 0.0])
        r2 = np.array([cos(q[0] + q[1]), sin(q[0] + q[1]), 0.0])
        t1, t2 = np.cross([0.0, 0, 1], r1), np.cross([0.0, 0, 1], r2)
        positions = np.array([np.zeros(3), r1])
        rotations = np.array([rotation_z(q[0]), rotation_z(q[0] + q[1])])
        com = np.array([r1 / 2, r1 + r2 / 2])
        linear[0, :, 0], linear[1, :, 0], linear[1, :, 1] = t1 / 2, t1 + t2 / 2, t2 / 2
        angular[0, 2, 0], angular[1, 2, :] = 1.0, [1.0, 1.0]
        c, s = cos(q[1]), sin(q[1])
        mass = np.array([[5 / 3 + c, 1 / 3 + c / 2], [1 / 3 + c / 2, 1 / 3]])
        bias = np.array([-s * (2 * v[0] * v[1] + v[1] ** 2) / 2, s * v[0] ** 2 / 2])
    assembled = np.zeros((n, n))
    body_energy = []
    for i in range(n):
        inertia = rotations[i] @ np.diag(model["inertia"][i]) @ rotations[i].T
        assembled += (
            model["mass"][i] * (linear[i].T @ linear[i]) + angular[i].T @ inertia @ angular[i]
        )
        vc, omega = linear[i] @ v, angular[i] @ v
        body_energy.append(float((model["mass"][i] * (vc @ vc) + omega @ inertia @ omega) / 2))
    close(assembled, mass, atol=1e-12, rtol=1e-12)
    external = data["effort"] + np.einsum("bij,bi->j", linear, data["force"])
    external += np.einsum("bij,bi->j", angular, data["torque"])
    return dict(
        position=positions,
        rotation=rotations,
        com=com,
        linear=linear,
        angular=angular,
        mass=mass,
        bias=bias,
        external=external,
        body_kinetic_energy_j=body_energy,
    )


def dynamics(data, s, u):
    P, C = data["P"], data["C"]
    q, v = P @ s + data["c"], P @ u
    full = physical_terms(data, q, v)
    n = len(q)
    drive = np.zeros(n)
    drive[list(data["independent"])] = data["motor"]
    if len(C):
        kkt = np.block([[full["mass"], -C.T], [C, np.zeros((len(C), len(C)))]])
        solved = np.linalg.solve(
            kkt, np.r_[drive + full["external"] - full["bias"], np.zeros(len(C))]
        )
        acceleration, multipliers = solved[:n], solved[n:]
    else:
        acceleration = np.linalg.solve(full["mass"], drive + full["external"] - full["bias"])
        multipliers = np.zeros(0)
    reduced = acceleration[list(data["independent"])]
    return dict(
        full=full,
        q=q,
        v=v,
        mass=P.T @ full["mass"] @ P,
        bias=P.T @ full["bias"],
        external=P.T @ full["external"],
        motor_full=drive,
        acceleration=reduced,
        full_acceleration=acceleration,
        multipliers=multipliers,
        reaction=C.T @ multipliers,
    )


def loaded_affine(ph, data, output):
    text = urdf_text(data)
    path = output / (data["name"] + ".urdf")
    path.write_text(text, encoding="utf-8")
    asset = ph.load_mimic_urdf(path, gravity=data["model"]["gravity"])
    assert asset.tree.joint_names == data["model"]["joint_names"]
    assert asset.joint_names == tuple(data["model"]["joint_names"][i] for i in data["independent"])
    close(asset.matrix, data["P"], atol=1e-13, rtol=0)
    close(asset.offset_vector, data["c"], atol=1e-13, rtol=0)
    assert tuple(
        (row.driver, row.follower, row.multiplier, row.offset) for row in asset.relations
    ) == tuple(
        (data["model"]["joint_names"][d], data["model"]["joint_names"][f], a, b)
        for d, f, a, b in data["rows"]
    )
    world = ph.AffineRobotTree(
        asset,
        ph.RobotState(data["s"], data["u"]),
        kp=data["kp"],
        kd=data["kd"],
        effort_limit=data["cap"],
        delay_steps=0,
        control_max_iterations=512,
    )
    loads = ph.RobotInputs(data["effort"], data["force"], data["torque"])
    action = world.inputs(
        data["s"], np.zeros(len(data["s"])), feedforward=data["motor"], loads=loads
    )
    return (
        asset,
        world,
        action,
        dict(path=str(path), text=text, sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
    )


def make_worlds(ph, output, record):
    inputs = {
        53: declaration("A03"),
        91: declaration("A04"),
        117: declaration("N9"),
        23: declaration("ordinary-2R"),
    }
    inputs[7] = dict(
        name="FreeSphere",
        mass_kg=2.0,
        radius_m=1 / 16,
        s=np.array([0.25, -0.125, 0.5]),
        u=np.array([0.125, 0.0, -0.0625]),
        force_n=np.array([0.25, 0.0, 0.0]),
        gravity=np.zeros(3),
    )
    assets, specs, actions, urdfs = {}, {}, {}, {}
    # Bind all declared inputs before the first loader/device operation.
    record.update(input=inputs, controls={}, source_urdfs=urdfs, world_order=WORLD_ORDER)
    for world in AFFINE_KEYS:
        data = inputs[world]
        record["source_urdfs"][world] = dict(text=urdf_text(data))
        asset, spec, action, source = loaded_affine(ph, data, output)
        assets[world], specs[world], actions[world], urdfs[world] = asset, spec, action, source
        record["controls"][world] = action
    data = inputs[23]
    m = data["model"]
    model = ph.RobotModel(
        np.array(m["parents"]),
        m["kinds"],
        m["offsets"],
        np.tile([0.0, 0, 0, 1], (2, 1)),
        m["axes"],
        m["mass"],
        m["com"],
        m["inertia"],
        m["gravity"],
    )
    drive = ph.DrivenRobotProgram(
        model, kp=data["kp"], kd=data["kd"], effort_limit=data["cap"], delay_steps=0
    )
    initial = drive.initial_state(ph.RobotState(data["s"], data["u"]))
    specs[23] = ph.FixedRobotTree(drive, initial)
    actions[23] = drive.inputs(
        np.zeros(2),
        np.zeros(2),
        feedforward=np.zeros(2),
        loads=ph.RobotInputs(data["effort"], data["force"], data["torque"]),
    )
    sphere = inputs[7]
    specs[7] = ph.FreeSphere(
        sphere["mass_kg"],
        sphere["radius_m"],
        position=sphere["s"],
        velocity=sphere["u"],
        gravity=sphere["gravity"],
    )
    actions[7] = sphere["force_n"]
    record["controls"][7] = actions[7]
    record["ordinary_control"] = dict(
        target=np.zeros(2),
        rate=np.zeros(2),
        feedforward=np.zeros(2),
        effort=data["effort"],
        force=data["force"],
        torque=data["torque"],
    )
    return [(key, specs[key]) for key in WORLD_ORDER], actions, inputs, assets


def material_geometry(ph, inputs, assets, checkpoint):
    shapes, bindings, materials, excluded = [], [], {}, []
    for world in WORLD_ORDER:
        dynamic = []
        if world in AFFINE_KEYS:
            data, asset = inputs[world], assets[world]
            for collider in asset.tree.collisions:
                body = data["model"]["names"].index(collider.link_name)
                group = asset.tree.sites[collider.link_name].link_index
                assert group == body and collider.name == "material"
                assert collider.shape.kind == "sphere" and collider.shape.vertex_ids == (0,)
                assert collider.shape.radius == 1 / 16
                close(collider.local_vertices, [[0.25, -0.125, 0]], atol=1e-13, rtol=0)
                key = ph.CollisionKey(world, 100 + group, body + 1)
                shapes.append(replace(collider.shape, key=key))
                bindings.append(
                    ph.ResidentAffineTreeBinding.from_asset(
                        asset, collider.shape.key, key=key, world=world
                    )
                )
                materials[key] = dict(
                    dynamic=True,
                    body=body,
                    group=group,
                    kind="sphere",
                    local_points_m=np.array([[0.25, -0.125, 0.0]]),
                    radius_m=1 / 16,
                    vertex_ids=(0,),
                    revision=collider.shape.revision,
                    source_key=collider.shape.key,
                    source_shape=collider.shape,
                    source_link=collider.link_name,
                    registration="from_asset",
                )
                dynamic.append(key)
            declarations = (
                (1, 12, "capsule", np.array([[-0.5, 0, 0], [0.5, 0, 0]]), 1 / 32, (201, 202)),
                (
                    1,
                    13,
                    "hull",
                    box([-0.25, -0.125, -0.0625], [0.25, 0.125, 0.0625]) @ rotation_z(pi / 4).T,
                    0.0,
                    tuple(range(301, 309)),
                ),
                (
                    0,
                    14,
                    "hull",
                    np.array(
                        [
                            [-0.25, -0.125, -0.0625],
                            [-0.25, 0.125, -0.0625],
                            [0.25, -0.125, -0.0625],
                            [0.25, 0.125, -0.0625],
                            [0, 0, 0.25],
                        ]
                    ),
                    0.0,
                    (401, 402, 403, 404, 405),
                ),
            )
        elif world == 23:
            declarations = (
                (0, 11, "sphere", np.array([[0.25, -0.125, 0]]), 1 / 16, (101,)),
                (1, 12, "capsule", np.array([[-0.5, 0, 0], [0.5, 0, 0]]), 1 / 32, (201, 202)),
            )
        else:
            declarations = ((None, 11, "sphere", np.zeros((1, 3)), 1 / 16, (101,)),)
        for body, shape_id, kind, local, radius, ids in declarations:
            group = body
            group_local = local
            if world in AFFINE_KEYS:
                site = assets[world].tree.sites[inputs[world]["model"]["names"][body]]
                group = site.link_index
                group_local = (
                    np.asarray(site.local_position_m)
                    + local @ quaternion_matrix(site.local_orientation_xyzw).T
                )
            key = ph.CollisionKey(world, 100 + (0 if group is None else group), shape_id)
            shapes.append(ph.CollisionShape(key, kind, ids, radius, revision=7))
            if world == 7:
                binding = ph.ResidentSphereBinding(key, world)
            else:
                binding_type = (
                    ph.ResidentAffineTreeBinding if world in AFFINE_KEYS else ph.ResidentTreeBinding
                )
                binding = binding_type(
                    key, world, group, group_local, topology=checkpoint[world].topology
                )
            bindings.append(binding)
            materials[key] = dict(
                dynamic=True,
                body=body,
                group=group,
                kind=kind,
                local_points_m=local.copy(),
                group_local_points_m=np.array(group_local, copy=True),
                radius_m=radius,
                vertex_ids=ids,
                revision=7,
                registration="explicit_local_material",
            )
            dynamic.append(key)
        excluded.extend(combinations(dynamic, 2))
        for body, low, high in ((900, [-4.0] * 3, [4.0] * 3), (901, [20.0] * 3, [21.0] * 3)):
            key = ph.CollisionKey(world, body, 1)
            points = box(low, high)
            shapes.append(ph.CollisionShape(key, "hull", tuple(range(8)), revision=3))
            bindings.append(ph.ResidentStaticBinding(key, ph.CollisionMotion(points)))
            materials[key] = dict(
                dynamic=False,
                kind="hull",
                local_points_m=points,
                radius_m=0.0,
                vertex_ids=tuple(range(8)),
                revision=3,
            )
    # Same body/shape IDs appear in different worlds; registration order is
    # intentionally unrelated to world slots, group rows or sorted keys.
    return (
        ph.CollisionModel(tuple(reversed(shapes)), tuple(excluded)),
        tuple(reversed(bindings)),
        materials,
    )


def collision_record(result):
    return {
        name: json_value(getattr(result, name))
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


def check_geometry(result, inputs, states, materials):
    poses = {}
    for world in TREE_KEYS:
        state = states[world]
        q = state.full_position if world in AFFINE_KEYS else state.position
        v = state.full_velocity if world in AFFINE_KEYS else state.velocity
        poses[world] = physical_terms(inputs[world], q, v)
    assert result.shape_keys == tuple(sorted(materials))
    assert result.shape_revisions == tuple(materials[key]["revision"] for key in result.shape_keys)
    truth, vertices = {}, {}
    for index, key in enumerate(result.shape_keys):
        material = materials[key]
        points = material["local_points_m"]
        if material["dynamic"]:
            if key.world == 7:
                points = np.asarray([states[key.world].position])
            else:
                body, pose = material["body"], poses[key.world]
                points = pose["position"][body] + points @ pose["rotation"][body].T
        radius = material["radius_m"]
        low, high = points.min(0) - radius, points.max(0) + radius
        bounds = result.bounds[index]
        assert np.isfinite(bounds).all()
        # Original local material and independently evaluated actual stored N
        # coordinates define truth. Device FK/bounds never define this oracle.
        assert np.all(bounds[:3] <= low) and np.all(bounds[3:] >= high), str(key)
        truth[key], vertices[key] = (low, high), points
    registered, expected = [], []
    for first, second in combinations(result.shape_keys, 2):
        if (
            first.world != second.world
            or materials[first]["dynamic"] == materials[second]["dynamic"]
        ):
            continue
        registered.append((first, second))
        a, b = truth[first], truth[second]
        if np.all(a[0] <= b[1]) and np.all(b[0] <= a[1]):
            expected.append((first, second))
    assert result.registered_pairs == tuple(registered)
    assert result.candidates == tuple(expected)
    assert len(expected) == 26  # 6 chain + 5 affine-2R + 12 N9 + 2 ordinary + 1 free
    assert result.world_keys == tuple(sorted(inputs))
    assert result.ticks.tolist() == [states[world].tick for world in result.world_keys]
    return {
        str(key): dict(low_m=low, high_m=high, vertices_world_m=vertices[key])
        for key, (low, high) in truth.items()
    }


def check_pose(data, asset, state, frame):
    affine = asset is not None
    q = np.asarray(state.full_position if affine else state.position)
    v = np.asarray(state.full_velocity if affine else state.velocity)
    truth = physical_terms(data, q, v)
    assert frame.tick == state.tick and frame.joint_types == data["model"]["kinds"]
    close(frame.position, q)
    close(frame.velocity, v)
    close(frame.com_position_m, truth["com"])
    if affine:
        assert frame.joint_names == asset.joint_names
        assert frame.full_joint_names == data["model"]["joint_names"]
        close(frame.independent_position, state.position)
        close(frame.independent_velocity, state.velocity)
        close(q, data["P"] @ state.position + data["c"], atol=CONSTRAINT_ATOL)
        close(v, data["P"] @ state.velocity, atol=CONSTRAINT_ATOL)
        close(data["C"] @ q, [row[3] for row in data["rows"]], atol=CONSTRAINT_ATOL)
        close(data["C"] @ v, np.zeros(len(data["rows"])), atol=CONSTRAINT_ATOL)
        assert np.asarray(state.control[0]).shape == (len(state.position), 3)
        assert np.asarray(state.control[1]).shape == (len(q), 7)
        expected_drive = np.column_stack((data["s"], np.zeros(len(data["s"])), data["motor"]))
        expected_load = np.column_stack((data["effort"], data["force"], data["torque"]))
        np.testing.assert_array_equal(state.control[0], expected_drive.astype(np.float32))
        np.testing.assert_array_equal(state.control[1], expected_load.astype(np.float32))
        for body, name in enumerate(data["model"]["names"]):
            site = asset.tree.sites[name]
            group = site.link_index
            R = frame.link_rotation[group]
            close(frame.link_position_m[group] + R @ site.local_position_m, truth["position"][body])
            close(R @ quaternion_matrix(site.local_orientation_xyzw), truth["rotation"][body])
    else:
        assert np.asarray(state.control).shape == (len(q), 10)
        # Original ordinary_control: target/rate/FF, then effort/world COM F/T.
        expected_control = np.column_stack(
            (np.zeros((len(q), 3)), data["effort"], data["force"], data["torque"])
        ).astype(np.float32)
        np.testing.assert_array_equal(state.control, expected_control)
        close(frame.link_position_m, truth["position"])
        close(frame.link_rotation, truth["rotation"])
    assert state.history == ()  # original delay-0 windows, not a new delay test
    return truth


def advance_expected(inputs, expected):
    rows = {}
    for world, data in inputs.items():
        s, u = expected[world]
        if world == 7:
            acceleration = data["force_n"] / data["mass_kg"]
            truth = None
        else:
            truth = dynamics(data, s, u)
            acceleration = truth["acceleration"]
        next_u = u + H * acceleration
        expected[world] = (s + H * next_u, next_u)
        rows[world] = dict(
            entry_position=s.copy(),
            entry_velocity=u.copy(),
            acceleration=acceleration,
            independent_terms=truth,
            exit_position=expected[world][0].copy(),
            exit_velocity=next_u.copy(),
        )
    return rows


def check_step(inputs, assets, before, after, frames, diagnostics, summaries, expected, tick):
    independent = {}
    for world, data in inputs.items():
        state = after[world]
        assert state.tick == tick and before[world].tick == tick - 1
        close(state.position, expected[world][0], atol=WINDOW_ATOL)
        close(state.velocity, expected[world][1], atol=WINDOW_ATOL)
        if world == 7:
            close(state.applied_effort, data["force_n"])
            velocity = np.asarray(state.velocity)
            close(summaries[world]["kinetic_energy_j"], data["mass_kg"] * (velocity @ velocity) / 2)
            continue
        s, u = np.asarray(before[world].position), np.asarray(before[world].velocity)
        truth = dynamics(data, s, u)
        independent[world] = truth
        diagnostic = diagnostics[world]
        assert diagnostic is not None
        assert (diagnostic.entry_tick, diagnostic.exit_tick, diagnostic.dt_s) == (tick - 1, tick, H)
        close(state.applied_effort, data["motor"])
        close(state.generalized_applied, data["motor"] + truth["external"])
        close(state.velocity, u + H * truth["acceleration"])
        close(state.position, s + H * np.asarray(state.velocity))
        common = dict(
            entry_position=s,
            entry_velocity=u,
            mass=truth["full"]["mass"],
            bias=truth["full"]["bias"],
            external=truth["full"]["external"],
            drive=data["motor"],
            acceleration=truth["acceleration"],
            linear_jacobian=truth["full"]["linear"],
            angular_jacobian=truth["full"]["angular"],
        )
        if world in AFFINE_KEYS:
            common.update(
                full_entry_position=truth["q"],
                full_entry_velocity=truth["v"],
                reduced_mass=truth["mass"],
                reduced_bias=truth["bias"],
                reduced_external=truth["external"],
                reduced_linear_jacobian=truth["full"]["linear"] @ data["P"],
                reduced_angular_jacobian=truth["full"]["angular"] @ data["P"],
                full_acceleration=truth["full_acceleration"],
                constraint_effort_full=truth["reaction"],
                row_multipliers=truth["multipliers"],
            )
            close(state.full_applied_effort, truth["motor_full"])
            close(state.full_generalized_applied, truth["motor_full"] + truth["full"]["external"])
            close(diagnostic.row_powers_w, np.zeros(len(data["rows"])), atol=CONSTRAINT_ATOL)
            close(diagnostic.constraint_power_w, 0.0, atol=CONSTRAINT_ATOL)
        for name, oracle in common.items():
            value = getattr(diagnostic, name)
            close(value, oracle)
            assert not value.flags.writeable
        assert diagnostic.equation_residual <= 2e-5
        assert np.isfinite(diagnostic.control_residual)
        # Original clipped-effort residual, not normalized by a possibly huge cap.
        expected_drive = np.clip(data["motor"], -data["cap"], data["cap"])
        actual_drive = np.asarray(state.applied_effort)
        assert np.all(
            np.abs(actual_drive - expected_drive)
            <= 5e-5 * np.maximum(1.0, np.maximum(abs(actual_drive), abs(expected_drive)))
        )
        pose = check_pose(data, assets.get(world), state, frames[world])
        assert set(summaries[world]) == {"kinetic_energy_j"}
        close(summaries[world]["kinetic_energy_j"], sum(pose["body_kinetic_energy_j"]))
        if tick == 1 and world == 53:
            close(diagnostic.reduced_mass, [[7.0]])
            close(diagnostic.acceleration, [1.0])
            close(diagnostic.row_multipliers, [2.5, -1.0])
            close(diagnostic.constraint_effort_full, [-5.0, 2.0, -1.0])
        if tick == 1 and world == 91:
            close(diagnostic.reduced_mass, [[5 / 3]])
            close(diagnostic.reduced_bias, [-0.25])
            close(diagnostic.acceleration, [0.75])
            close(diagnostic.row_multipliers, [-0.125])
    return independent


def rejection(function):
    try:
        function()
    except (ValueError, TypeError) as error:
        return dict(type=type(error).__name__, message=str(error))
    raise AssertionError("invalid public graph/checkpoint/result was not rejected")


def wrong_graph_checkpoint(ph, mode, output, record):
    data = declaration("A03-flat")
    asset, world, action, source = loaded_affine(ph, data, output)
    with ph.TyxWorldBatch([(53, world)], mode, cache_root=output / "cache") as alternate:
        alternate.set_controls({53: action})
        checkpoint = alternate.snapshot()[53]
        record["wrong_graph_fixture"] = dict(
            input=data,
            source_urdf=source,
            checkpoint=checkpoint,
            initial_kinematics=alternate.tree_kinematics([53]),
            stats_before_close=alternate.stats(),
        )
    record["wrong_graph_fixture"]["stats_after_close"] = alternate.stats()
    assert alternate.stats()["device_owned_bytes"] == 0
    return asset, action, checkpoint


def run_window(ph, mode, output, record):
    worlds, actions, inputs, assets = make_worlds(ph, output, record)
    flat_asset, flat_action, flat_checkpoint = wrong_graph_checkpoint(ph, mode, output, record)
    # Equal full physical mapping is intentional; original row identity differs.
    np.testing.assert_array_equal(flat_asset.matrix, assets[53].matrix)
    np.testing.assert_array_equal(flat_asset.offset_vector, assets[53].offset_vector)
    record.update(h_s=H, main_steps=STEPS, replay_from_tick=REPLAY_FROM, trace=[], replay=[])
    expected = {world: (data["s"].copy(), data["u"].copy()) for world, data in inputs.items()}
    batch = ph.TyxWorldBatch(worlds, mode, cache_root=output / "cache")
    try:
        with batch:
            batch.set_controls(actions)
            initial = batch.snapshot()
            initial_frames = batch.tree_kinematics(TREE_KEYS)
            record.update(
                initial=initial,
                initial_kinematics=initial_frames,
                initial_scalar_kinematics=batch.kinematics([7]),
                initial_diagnostics=batch.tree_diagnostics(TREE_KEYS),
                initial_stats=batch.stats(),
            )
            assert record["initial_diagnostics"] == {world: None for world in TREE_KEYS}
            for world in TREE_KEYS:
                close(initial[world].position, inputs[world]["s"])
                close(initial[world].velocity, inputs[world]["u"])
                check_pose(inputs[world], assets.get(world), initial[world], initial_frames[world])
            model, bindings, materials = material_geometry(ph, inputs, assets, initial)
            record.update(
                collision_model=model,
                original_material=materials,
                collision_semantics="finite original skin; same-world dynamic/static; explicit dynamic/self exclusions",
            )
            with batch.prepare_collision(model, bindings) as plan:
                beginning = plan.query()
                record["initial_geometry"] = collision_record(beginning)
                record["initial_oracle_material"] = check_geometry(
                    beginning, inputs, initial, materials
                )
                previous, saved = initial, {}
                for tick in range(1, STEPS + 1):
                    frame = dict(
                        tick=tick,
                        entry_time_s=(tick - 1) * H,
                        exit_time_s=tick * H,
                        entry=previous,
                        checks_complete=False,
                    )
                    record["trace"].append(frame)
                    frame["accepted"] = batch.step(H)
                    actual = batch.snapshot()
                    frame["actual"] = actual
                    frames, diagnostics = (
                        batch.tree_kinematics(TREE_KEYS),
                        batch.tree_diagnostics(TREE_KEYS),
                    )
                    summaries = batch.summary()
                    frame.update(
                        kinematics=frames,
                        diagnostics=diagnostics,
                        summary=summaries,
                        scalar_kinematics=batch.kinematics([7]),
                    )
                    result = plan.query()
                    frame.update(
                        geometry=collision_record(result),
                        independent_window=advance_expected(inputs, expected),
                        batch_stats=batch.stats(),
                        plan_stats=plan.stats(),
                    )
                    assert tuple(frame["accepted"]) == WORLD_ORDER and all(
                        frame["accepted"].values()
                    )
                    frame["actual_entry_oracle"] = check_step(
                        inputs,
                        assets,
                        previous,
                        actual,
                        frames,
                        diagnostics,
                        summaries,
                        expected,
                        tick,
                    )
                    frame["original_material_world"] = check_geometry(
                        result, inputs, actual, materials
                    )
                    frame["checks_complete"] = True
                    saved[tick] = (actual, result, frames, copy.deepcopy(expected))
                    previous = actual
                record["window_final"] = previous
                assert {frame["batch_stats"]["current_buffer"] for frame in record["trace"]} == {
                    0,
                    1,
                }
                # Only original graph identity changes: all physical state,
                # held commands and accepted ticks remain exactly the same.
                wrong = replace(previous[53], topology=flat_checkpoint.topology)
                assert wrong.topology != previous[53].topology
                before_stats = batch.stats()
                rejection_rows = dict(
                    checkpoint=rejection(lambda: batch.restore({53: wrong})),
                    control=rejection(lambda: batch.set_controls({53: flat_action})),
                )
                source = next(c for c in flat_asset.tree.collisions if c.link_name == "link0")
                key = next(
                    k
                    for k, m in materials.items()
                    if k.world == 53 and m.get("source_link") == "link0"
                )
                bad_binding = ph.ResidentAffineTreeBinding.from_asset(
                    flat_asset, source.shape.key, key=key, world=53
                )
                bad_bindings = tuple(bad_binding if item.key == key else item for item in bindings)

                def bad_prepare():
                    with batch.prepare_collision(model, bad_bindings):
                        pass

                rejection_rows["material_source"] = rejection(bad_prepare)
                record["wrong_graph_rejections"] = dict(
                    actual_wrong_checkpoint=wrong,
                    errors=rejection_rows,
                    stats_before=before_stats,
                    stats_after=batch.stats(),
                )
                assert batch.snapshot() == previous
                assert batch.stats()["current_buffer"] == before_stats["current_buffer"]
                assert batch.stats()["state_epoch"] == before_stats["state_epoch"]
                plan.require_current(result)
                # Valid retry after graph rejection, at exactly the same tick.
                batch.restore(previous)
                assert batch.snapshot() == previous
                same_tick = plan.query()
                record["same_tick_restore"] = dict(
                    old=collision_record(result),
                    current=collision_record(same_tick),
                    stale_rejection=rejection(lambda: plan.require_current(result)),
                    stats=batch.stats(),
                    diagnostic=batch.tree_diagnostics(TREE_KEYS),
                )
                assert same_tick.state_epoch == result.state_epoch + 1
                np.testing.assert_array_equal(same_tick.ticks, result.ticks)
                np.testing.assert_array_equal(same_tick.bounds, result.bounds)
                assert same_tick.candidates == result.candidates
                assert record["same_tick_restore"]["diagnostic"] == {
                    world: None for world in TREE_KEYS
                }
                checkpoint, old_geometry, _, expected = saved[REPLAY_FROM]
                expected = copy.deepcopy(expected)
                batch.restore(checkpoint)
                assert batch.snapshot() == checkpoint
                replay_geometry = plan.query()
                record["replay_checkpoint"] = dict(
                    checkpoint=checkpoint,
                    restored=batch.snapshot(),
                    geometry=collision_record(replay_geometry),
                    stale_rejection=rejection(lambda: plan.require_current(old_geometry)),
                    stats=batch.stats(),
                )
                np.testing.assert_array_equal(replay_geometry.bounds, old_geometry.bounds)
                assert replay_geometry.candidates == old_geometry.candidates
                previous = checkpoint
                # Do not resend controls. Complete checkpoint held input/history
                # is the only control source during these 21 real replay steps.
                for tick in range(REPLAY_FROM + 1, STEPS + 1):
                    row = dict(
                        tick=tick,
                        entry_time_s=(tick - 1) * H,
                        exit_time_s=tick * H,
                        entry=previous,
                        checks_complete=False,
                    )
                    record["replay"].append(row)
                    row["accepted"] = batch.step(H)
                    actual = batch.snapshot()
                    row["actual"] = actual
                    frames, diagnostics = (
                        batch.tree_kinematics(TREE_KEYS),
                        batch.tree_diagnostics(TREE_KEYS),
                    )
                    summaries, result = batch.summary(), plan.query()
                    row.update(
                        kinematics=frames,
                        diagnostics=diagnostics,
                        summary=summaries,
                        scalar_kinematics=batch.kinematics([7]),
                        geometry=collision_record(result),
                        independent_window=advance_expected(inputs, expected),
                        batch_stats=batch.stats(),
                        plan_stats=plan.stats(),
                    )
                    assert all(row["accepted"].values()) and tuple(row["accepted"]) == WORLD_ORDER
                    assert actual == saved[tick][0]
                    assert json_value(frames) == json_value(saved[tick][2])
                    np.testing.assert_array_equal(result.bounds, saved[tick][1].bounds)
                    assert result.candidates == saved[tick][1].candidates
                    row["actual_entry_oracle"] = check_step(
                        inputs,
                        assets,
                        previous,
                        actual,
                        frames,
                        diagnostics,
                        summaries,
                        expected,
                        tick,
                    )
                    row["original_material_world"] = check_geometry(
                        result, inputs, actual, materials
                    )
                    row["checks_complete"] = True
                    previous = actual
                record["checks"] = dict(
                    full_40_step_mixed_window=True,
                    full_and_reduced_coordinates=True,
                    independent_kkt_and_original_row_reactions=True,
                    all_original_material_bounds=True,
                    exact_same_world_candidates=True,
                    graph_checkpoint_control_material_rejected=True,
                    invalid_graph_no_publish=True,
                    same_tick_epoch_and_owned_results=True,
                    complete_checkpoint_21_step_exact_replay=True,
                )
                stats = batch.stats()
                assert stats["completed_steps"] == STEPS + (STEPS - REPLAY_FROM)
                assert stats["accepted_world_steps"] == len(WORLD_ORDER) * (
                    STEPS + STEPS - REPLAY_FROM
                )
                assert stats["rejected_world_steps"] == 0
                record["batch_stats_before_close"], record["plan_stats_before_close"] = (
                    stats,
                    plan.stats(),
                )
                if mode == "cuda":
                    assert (
                        stats["tree_state_downloaded_bytes"] > 0
                        and stats["affine_state_downloaded_bytes"] > 0
                    )
                    assert stats["submitted_kernels"] > 0 and stats["device_allocations"] > 0
    finally:
        record["batch_stats_after_close"] = batch.stats()
    assert (
        batch.stats()["device_owned_bytes"] == 0 and batch.stats()["collision_plan_live_count"] == 0
    )
    record["checks"]["resources_closed"] = True


def archive_identity(installed, path):
    wheel = path.resolve(strict=True)
    if wheel.suffix.lower() != ".whl":
        raise ValueError("--expected-wheel must name the installed wheel archive")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    direct_text = installed.read_text("direct_url.json")
    assert direct_text is not None, "install the local wheel directly, preserving direct_url.json"
    direct = json.loads(direct_text)
    parsed = urlsplit(direct["url"])
    assert parsed.scheme == "file" and parsed.path.lower().endswith(".whl")
    archive = direct["archive_info"]
    recorded = archive.get("hashes", {}).get("sha256")
    if recorded is None:
        algorithm, _, recorded = archive.get("hash", "").partition("=")
        assert algorithm == "sha256"
    assert recorded.lower() == digest
    return dict(
        path=str(wheel), sha256=digest, installed_from=direct["url"], version=installed.version
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
    site = args.wheel_site.resolve(strict=True)
    package_root = args.expected_package_root.resolve(strict=True)
    checkout = args.checkout.resolve(strict=True)
    script = Path(__file__).resolve(strict=True)
    output = args.output_dir.resolve()
    assert __debug__ and sys.flags.isolated and sys.flags.dont_write_bytecode, "use python -I -B"
    assert package_root == (site / "phymeo").resolve(strict=True)
    assert Path.cwd().resolve() == script.parent
    assert not script.is_relative_to(checkout) and not site.is_relative_to(checkout)
    assert not output.is_relative_to(checkout)
    assert "phymeo" not in sys.modules
    output.mkdir(parents=True, exist_ok=False)
    record = dict(
        status="running",
        mode=args.mode,
        errors=[],
        script=dict(path=str(script), sha256=hashlib.sha256(script.read_bytes()).hexdigest()),
        expected_package_root=str(package_root),
        units=dict(
            position="R rad / P m; free xyz m",
            velocity="R rad/s / P m/s; free xyz m/s",
            effort="R N*m / P N; free force N",
            geometry="m",
            energy="J",
            time="s",
        ),
        limits="Installation smoke, not benchmark/full C5 matrix. Fixed-root affine and ordinary dynamics; current-pose broad phase only. No contact response, CCD, floating, loop, device-loss injection or new delay qualification. Source parity is checked by the parent outside this consumer.",
        gates=dict(
            h_s=H,
            main_steps=STEPS,
            replay_steps=STEPS - REPLAY_FROM,
            rtol=RTOL,
            state_atol=ATOL,
            window_atol=WINDOW_ATOL,
            constraint_atol=CONSTRAINT_ATOL,
            equation_relative=2e-5,
            clipped_effort_relative=5e-5,
            material_containment="strict at actual stored N coordinates",
            candidate_domain="same-world dynamic/static; wide separated hit/miss controls",
        ),
    )
    sys.path.insert(0, str(site))
    try:
        import phymeo as ph

        assert Path(ph.__file__).resolve(strict=True).parent == package_root
        installed = distribution("phymeo")
        assert Path(installed.locate_file("")).resolve(strict=True) == site
        record["wheel"] = archive_identity(installed, args.expected_wheel)
        record["modules_before"] = module_provenance(site, package_root, args.expected_wheel)
        names = (
            "TyxWorldBatch",
            "AffineRobotTree",
            "FixedRobotTree",
            "FreeSphere",
            "RobotModel",
            "RobotState",
            "RobotInputs",
            "DrivenRobotProgram",
            "load_mimic_urdf",
            "ResidentAffineTreeBinding",
            "ResidentTreeBinding",
            "ResidentSphereBinding",
            "ResidentStaticBinding",
            "CollisionKey",
            "CollisionShape",
            "CollisionMotion",
            "CollisionModel",
        )
        assert all(hasattr(ph, name) for name in names), "public resident inputs are missing"
        record["public_api"] = names
        record["versions"] = dict(
            numpy=np.__version__,
            tyx=distribution("tyx-compute").version,
            python=sys.version,
            phymeo=installed.version,
        )
        run_window(ph, args.mode, output, record)
        record["status"] = "passed"
    except Exception as error:
        record["status"] = "failed"
        record["errors"].append(
            dict(
                stage="wheel_import_or_window",
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            )
        )
    try:
        if "phymeo" in sys.modules:
            record["modules_after"] = module_provenance(site, package_root, args.expected_wheel)
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
    trace = output / "resident-affine-wheel.trace.json.gz"
    with gzip.open(trace, "xt", encoding="utf-8") as stream:
        json.dump(json_value(record), stream, allow_nan=False, separators=(",", ":"))
    summary = dict(
        status=record["status"],
        raw_exit=0 if record["status"] == "passed" else 1,
        mode=args.mode,
        attempted_window_steps=len(record.get("trace", ())),
        full_window_steps=sum(bool(row.get("checks_complete")) for row in record.get("trace", ())),
        attempted_replay_steps=len(record.get("replay", ())),
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
