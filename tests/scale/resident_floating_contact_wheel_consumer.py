"""C10 installed-wheel public contact consumer, AUTHORING_UNEXECUTED.

Copy this single file outside every checkout and use python -I -B. The fixed
W4 and sibling-tree windows use only public Phymeo APIs. Independent COM/J/M,
finite sphere contacts and semi-implicit reference states are self-contained.
HOST narrow phase plus real TYX response is not full-device narrow phase/CCD.
"""

import argparse
import gzip
import hashlib
import json
import math
import sys
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from importlib.metadata import distribution
from itertools import product
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname
from zipfile import ZipFile

import numpy as np

H, STEPS, REPLAY_FROM = 1 / 1024, 40, 19
ATOL, RTOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-4
MAX_BYTES = 256 * 1024 * 1024
ORDER = (503, 91, 1207, 6)
BROAD_PHYSICAL = ("world_keys", "ticks", "shape_keys", "shape_revisions", "bounds", "registered_pairs", "candidates")


def encode(value):
    if is_dataclass(value):
        return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return encode(value.tolist())
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def fingerprint(value):
    return json.dumps(encode(value), sort_keys=True, allow_nan=False, separators=(",", ":"))


def close(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


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
        for field in fields(value):
            readonly(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            readonly(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            readonly(item)


def rejected(call, label):
    try:
        call()
    except (ValueError, TypeError, RuntimeError) as error:
        return dict(label=label, type=type(error).__name__, message=str(error))
    raise AssertionError(label + " unexpectedly succeeded")


def rot(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(((1 - 2 * (y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)),
                     (2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)),
                     (2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y))))


def skew(v):
    x, y, z = v
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


def drift_quaternion(q, omega):
    magnitude = float(np.linalg.norm(omega))
    delta = (np.r_[omega * (math.sin(H*magnitude/2) / magnitude), math.cos(H*magnitude/2)]
             if magnitude else np.array((0., 0., 0., 1.)))
    result = np.r_[delta[3]*q[:3] + q[3]*delta[:3] + np.cross(delta[:3], q[:3]),
                   delta[3]*q[3] - delta[:3] @ q[:3]]
    result /= np.linalg.norm(result)
    return -result if result @ q < 0 else result


def floating_input(name):
    if name == "F01":
        masses, offsets, motor = (), (), ()
        origin = (0., 0., 3/32)
    elif name == "F02-motor":
        masses, offsets, motor = (1.,), (0.,), (3.,)
        origin = (0., 0., 0.)
    elif name == "F03":
        masses, offsets, motor = (2., 1.), (-7/64, 7/64), (0., 0.)
        origin = (0., 0., 0.)
    else:
        raise ValueError("undeclared floating case")
    n = len(masses)
    return dict(name=name, root_mass_kg=2., root_inertia_kg_m2=np.full(3, .5),
                root_com_m=np.zeros(3), mass_kg=np.r_[2., masses],
                inertia_kg_m2=np.r_[.5, np.full(n, .25)],
                parents=np.full(n, -1, dtype=int), kinds=("prismatic",)*n,
                offsets_m=np.column_stack((offsets, np.zeros(n), np.zeros(n))),
                axes=np.tile([1., 0., 0.], (n, 1)),
                joint_orientations_xyzw=np.tile([0., 0., 0., 1.], (n, 1)),
                child_com_m=np.zeros((n, 3)), gravity_m_s2=np.zeros(3),
                q=np.r_[origin, [0., 0., 0., 1.], np.zeros(n)], v=np.zeros(n+6),
                kp=np.zeros(n), kd=np.zeros(n), effort_limit_n=np.full(n, 64.),
                drive=np.column_stack((np.zeros(n), np.zeros(n), motor)),
                external_effort=np.zeros(n), external_force_n=np.zeros((n+1, 3)),
                external_torque_nm=np.zeros((n+1, 3)), delay_steps=0,
                radius_m=1/8, stiffness_n_m=64., damping_n_s_m=0., friction=0.)


def floating_declaration(ph, data):
    n = len(data["parents"])
    child = None if n == 0 else ph.RobotModel(
        data["parents"], data["kinds"], data["offsets_m"],
        data["joint_orientations_xyzw"], data["axes"], data["mass_kg"][1:],
        data["child_com_m"], np.full((n, 3), .25), data["gravity_m_s2"],
    )
    root = ph.RigidBodyModel(np.array([.5]), np.full((1, 3), .5), np.zeros(3))
    tree = ph.FloatingRobotTree(
        ph.FloatingRobotModel(root, data["root_com_m"], child),
        ph.FloatingRobotState(data["q"], data["v"]), kp=data["kp"], kd=data["kd"],
        effort_limit=data["effort_limit_n"], delay_steps=0,
    )
    external = ph.FloatingRobotInputs(data["external_effort"], data["external_force_n"], data["external_torque_nm"])
    return tree, tree.inputs(data["drive"], external)


def declarations(ph, sibling=False):
    data = {503: floating_input("F03" if sibling else "F01")}
    if not sibling:
        data[91] = floating_input("F02-motor")
    worlds, controls = {}, {}
    for world, original in data.items():
        worlds[world], controls[world] = floating_declaration(ph, original)
    if sibling:
        return worlds, controls, data
    model = ph.RobotModel(np.array([-1]), ("prismatic",), np.array([[1., 0., 0.]]),
                         np.array([[0., 0., 0., 1.]]), np.array([[0., 1., 0.]]),
                         np.array([2.]), np.zeros((1, 3)), np.full((1, 3), 1/8), np.zeros(3))
    program = ph.DrivenRobotProgram(model, kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.array([8.]))
    worlds[1207] = ph.FixedRobotTree(program, program.initial_state(ph.RobotState(np.array([1/8]), np.array([1/16]))))
    controls[1207] = program.inputs(np.zeros(1), np.zeros(1), feedforward=np.zeros(1),
                                   loads=ph.RobotInputs(np.zeros(1), np.array([[0., 2., 0.]]), np.zeros((1, 3))))
    worlds[6] = ph.FreeSphere(2, 1/16, position=(0, 0, 2), velocity=(1/16, 0, 0), gravity=(0, 0, -2))
    controls[6] = (0., 0., 0.)
    data[1207] = dict(name="ordinary-P", model=model, q=np.array([1/8]), v=np.array([1/16]),
                      mass_kg=2., offset_m=(1., 0., 0.), axis=(0., 1., 0.),
                      kp=(0.,), kd=(0.,), effort_limit_n=(8.,), delay_steps=0,
                      command=controls[1207], force_n=(0., 2., 0.))
    data[6] = dict(name="FreeSphere", mass_kg=2., radius_m=1/16, q=np.array([0., 0., 2.]),
                   v=np.array([1/16, 0., 0.]), gravity_m_s2=(0., 0., -2.), control=controls[6])
    assert tuple(worlds) == ORDER
    return worlds, controls, data


def box(low, high):
    return np.array(list(product(*zip(low, high, strict=True))), dtype=float)


def contact_declarations(ph, batch, data):
    shapes, bindings, sources, laws = [], [], {}, {}
    snapshots = batch.snapshot()
    for world, original in data.items():
        if original["name"] in ("ordinary-P", "FreeSphere"):
            continue
        if original["name"] == "F03":
            groups, wall = (1, 2), None
        elif original["name"] == "F01":
            groups, wall = (0,), box((-1, -1, -1/4), (1, 1, 0))
        else:
            groups, wall = (1,), box((-1/2, -1, -1), (-3/32, 1, 1))
        keys = []
        for ordinal, group in enumerate(groups):
            key = ph.CollisionKey(world, 101 + group, 7 + ordinal)
            shape = ph.CollisionShape(key, "sphere", (101 + 3*ordinal,), 1/8, revision=7)
            local = np.zeros((1, 3))
            shapes.append(shape)
            bindings.append(ph.ResidentFloatingTreeBinding(key, world, group, local, topology=snapshots[world].topology))
            sources[key] = (shape, group, local)
            keys.append(key)
        if wall is not None:
            key = ph.CollisionKey(world, 900, 2)
            shape = ph.CollisionShape(key, "hull", tuple(501 + 3*i for i in range(8)), revision=3)
            shapes.append(shape)
            bindings.append(ph.ResidentStaticBinding(key, ph.CollisionMotion(wall)))
            sources[key] = (shape, None, wall)
            keys.append(key)
        laws[tuple(sorted(keys))] = ph.RobotContactMaterial(64, 0, 0)
    return ph.CollisionModel(tuple(shapes)), tuple(bindings), sources, laws


def mechanics(data, q, v):
    """Independent full root+Px-forest inertia sum; no product M/J/FK calls."""
    q, v = np.asarray(q, float), np.asarray(v, float)
    n, masses = len(data["parents"]), data["mass_kg"]
    b, d = n + 1, n + 6
    rotation = rot(q[3:7])
    relative = np.zeros((b, 3))
    for i in range(n):
        relative[i+1] = rotation @ (data["offsets_m"][i] + np.array([q[7+i], 0., 0.]))
    points = q[:3] + relative
    linear, angular = np.zeros((b, 3, d)), np.zeros((b, 3, d))
    omega, axis = v[3:6], rotation[:, 0]
    bias_acceleration = np.zeros((b, 3))
    for body in range(b):
        linear[body, :, :3] = np.eye(3)
        linear[body, :, 3:6] = -skew(relative[body])
        angular[body, :, 3:6] = np.eye(3)
        if body:
            linear[body, :, 5+body] = axis
            bias_acceleration[body] = np.cross(omega, np.cross(omega, relative[body])) + 2*np.cross(omega, axis*v[5+body])
    mass = sum(masses[i] * linear[i].T @ linear[i] + data["inertia_kg_m2"][i] * angular[i].T @ angular[i] for i in range(b))
    bias = sum(linear[i].T @ (masses[i] * bias_acceleration[i]) for i in range(b))
    com_velocity, angular_velocity = linear @ v, angular @ v
    energy = sum(.5*masses[i]*float(com_velocity[i] @ com_velocity[i]) +
                 .5*data["inertia_kg_m2"][i]*float(angular_velocity[i] @ angular_velocity[i]) for i in range(b))
    momentum = sum(masses[i]*com_velocity[i] for i in range(b))
    angular_momentum = sum(np.cross(points[i], masses[i]*com_velocity[i]) +
                           data["inertia_kg_m2"][i]*angular_velocity[i] for i in range(b))
    return dict(mass=mass, bias=bias, linear_jacobian=linear, angular_jacobian=angular,
                position=points, rotation=np.tile(rotation, (b, 1, 1)), com_position=points.copy(),
                com_velocity=com_velocity, angular_velocity=angular_velocity,
                energy=energy, momentum=momentum, angular_momentum=angular_momentum)


def reference(data, q, v):
    """Original finite geometric law, then complete full-M response and drift."""
    q, v = np.asarray(q, float), np.asarray(v, float)
    truth = mechanics(data, q, v)
    wrench = np.zeros((len(data["mass_kg"]), 6))
    if data["name"] == "F03":
        first, second = truth["position"][1:3]
        delta = second-first
        distance = float(np.linalg.norm(delta))
        assert distance > 0
        normal = delta/distance
        gap = distance-1/4
        force = max(0., -64*gap)
        common = (first+second)/2
        for body, value in ((1, -force*normal), (2, force*normal)):
            wrench[body] = np.r_[value, np.cross(common-truth["com_position"][body], value)]
        surfaces = (first+normal/8, second-normal/8)
        first_force = -force*normal
    else:
        body, normal, wall_point = ((0, np.array([0., 0., 1.]), np.zeros(3)) if data["name"] == "F01"
                                    else (1, np.array([1., 0., 0.]), np.array([-3/32, 0., 0.])))
        center = truth["position"][body]
        distance = float(normal @ (center-wall_point))
        foot = center-distance*normal
        assert np.all(np.abs(foot[[0, 1] if body == 0 else [1, 2]]) < 1)
        assert distance > 0, "frozen finite-face domain crossed the solid interior"
        gap = distance-1/8
        force = max(0., -64*gap)
        surfaces = (center-normal/8, foot)
        common = (surfaces[0]+surfaces[1])/2
        first_force = force*normal
        wrench[body] = np.r_[first_force, np.cross(common-truth["com_position"][body], first_force)]
    generalized = (np.einsum("bij,bi->j", truth["linear_jacobian"], wrench[:, :3]) +
                   np.einsum("bij,bi->j", truth["angular_jacobian"], wrench[:, 3:]))
    drive = data["drive"][:, 2]
    applied = generalized.copy()
    applied[6:] += drive
    acceleration = np.linalg.solve(truth["mass"], applied-truth["bias"])
    vp, qp = v+H*acceleration, q.copy()
    qp[:3] += H*vp[:3]
    qp[3:7] = drift_quaternion(q[3:7], vp[3:6])
    qp[7:] += H*vp[6:]
    truth.update(wrench=wrench, contact_generalized=generalized, input_external=np.zeros_like(v),
                 drive=drive.copy(), acceleration=acceleration, gap_m=gap, normal_force_n=force,
                 first_force_n=first_force, common_point_m=common, surface_points_m=surfaces)
    return qp, vp, truth


def initial_reference(data):
    return {world: (item["q"].copy(), item["v"].copy()) for world, item in data.items()}


def independent_advance(data, q, v):
    if data["name"] == "ordinary-P":
        vp = np.asarray(v) + H
        return np.asarray(q)+H*vp, vp, dict(acceleration=np.array([1.]))
    if data["name"] == "FreeSphere":
        vp = np.asarray(v) + H*np.array([0., 0., -2.])
        return np.asarray(q)+H*vp, vp, dict(acceleration=np.array([0., 0., -2.]))
    return reference(data, q, v)


def first_anchors(data):
    anchors = {}
    for world, item in data.items():
        if item["name"] in ("ordinary-P", "FreeSphere"):
            continue
        q, v, truth = reference(item, item["q"], item["v"])
        if item["name"] == "F01":
            close(truth["mass"], np.diag([2., 2, 2, .5, .5, .5]), atol=1e-12, rtol=0)
            close(truth["wrench"], [[0, 0, 2, 0, 0, 0]], atol=1e-12, rtol=0)
            close(truth["acceleration"], [0, 0, 1, 0, 0, 0], atol=1e-12, rtol=0)
        elif item["name"] == "F02-motor":
            close(truth["mass"][np.ix_([0, 6], [0, 6])], [[3, 1], [1, 1]], atol=1e-12, rtol=0)
            close(truth["wrench"], [[0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]], atol=1e-12, rtol=0)
            close(truth["acceleration"][[0, 6]], [-1.5, 6.5], atol=1e-12, rtol=0)
        else:
            close(truth["mass"][np.ix_([0, 6, 7], [0, 6, 7])], [[5, 2, 1], [2, 2, 0], [1, 0, 1]], atol=1e-12, rtol=0)
            close(truth["wrench"][:, :3], [[0, 0, 0], [-2, 0, 0], [2, 0, 0]], atol=1e-12, rtol=0)
            close(truth["acceleration"][[0, 6, 7]], [0, -1, 2], atol=1e-12, rtol=0)
        anchors[world] = dict(truth=truth, q=q, v=v)
    return anchors


def broad(result):
    return {name: getattr(result, name) for name in ("topology_generation", "state_epoch", *BROAD_PHYSICAL)}


def capture(batch, plan):
    return dict(state=batch.snapshot(), geometry=batch.tree_kinematics(), diagnostics=batch.tree_diagnostics(),
                energy=batch.summary(), receipts=dict(plan.last_accepted), status=dict(plan.last_status),
                broad=broad(plan.query()), batch_stats=batch.stats(), plan_stats=plan.stats())


def bounds_audit(row, sources, model):
    result = row["broad"]
    assert set(result["world_keys"]) == {key.world for key in sources}
    assert tuple(result["shape_keys"]) == tuple(shape.key for shape in model.shapes)
    wanted = {}
    for key, (shape, group, local) in sources.items():
        if group is None:
            points = local
        else:
            geometry = row["geometry"][key.world]
            points = geometry.link_position_m[group] + local @ geometry.link_rotation[group].T
        low, high = points.min(axis=0)-shape.radius, points.max(axis=0)+shape.radius
        index = result["shape_keys"].index(key)
        actual = result["bounds"][index]
        assert np.isfinite(actual).all() and np.all(actual[:3] <= low) and np.all(actual[3:] >= high)
        assert result["shape_revisions"][index] == shape.revision
        wanted[key] = (low, high)
    registered = set(result["registered_pairs"])
    assert all(a.world == b.world and a.body != b.body for a, b in registered)
    assert set(result["candidates"]) <= registered
    true_pairs = {pair for pair in registered if np.all(wanted[pair[0]][0] <= wanted[pair[1]][1])
                  and np.all(wanted[pair[1]][0] <= wanted[pair[0]][1])}
    assert true_pairs <= set(result["candidates"])
    assert true_pairs, "the declared loaded finite pairs disappeared"
    for world, tick in zip(result["world_keys"], result["ticks"], strict=True):
        assert tick == row["state"][world].tick
    readonly(result["bounds"])
    return dict(material_bounds=wanted, true_pairs=tuple(sorted(true_pairs)))


def material_audit(receipt, sources, truth):
    wrench, reactions = np.zeros_like(receipt.law_wrench_fp64), {}
    total_application = total_material = 0.
    weights, detail = {}, []
    for site in receipt.sites:
        assert site.pair == tuple(sorted(site.pair)) and all(k.world == receipt.world for k in site.pair)
        geometry, pair = site.geometry, site.pair
        witness, common = geometry.witness, np.asarray(geometry.load_point, float)
        assert geometry.weight > 0
        assert (site.law.stiffness, site.law.damping, site.law.friction, site.law.slip_speed_m_s) == (64., 0., 0., .01)
        weights[pair] = weights.get(pair, 0.) + geometry.weight
        close(witness.gap_m, truth["gap_m"])
        close(common, truth["common_point_m"])
        close(site.first_force_n, geometry.weight*truth["first_force_n"])
        close(witness.first_point, truth["surface_points_m"][0])
        close(witness.second_point, truth["surface_points_m"][1])
        if truth["normal_force_n"] > 0:
            close(witness.normal, -truth["first_force_n"]/truth["normal_force_n"])
        np.testing.assert_array_equal(site.first_force_n, -site.second_force_n)
        # Material law and normalized geometric weight are used once.
        close(np.linalg.norm(site.first_force_n), max(0., -64*witness.gap_m)*geometry.weight)
        application_work = material_work = 0.
        sides = {}
        for side, key in zip(("first", "second"), pair, strict=True):
            shape, group, local = sources[key]
            declared = dict(zip(shape.vertex_ids, local, strict=True))
            address = getattr(witness, side+"_weights")
            assert address and all(node in declared for node, _ in address)
            close(sum(weight for _, weight in address), 1., atol=1e-12, rtol=0)
            material_local = sum(weight*declared[node] for node, weight in address)
            skin = getattr(witness, side+"_offset")
            surface = np.asarray(getattr(witness, side+"_point"), float)
            if group is None:
                rebuilt, reference_point = material_local, local.mean(axis=0)
                velocity, material_delta, application_delta = np.zeros(3), np.zeros(3), np.zeros(3)
            else:
                entry, end = receipt.entry_geometry, receipt.accepted_geometry
                p0, p1 = entry.link_position_m[group], end.link_position_m[group]
                r0, r1 = entry.link_rotation[group], end.link_rotation[group]
                rebuilt = p0+r0 @ material_local
                reference_point = entry.com_position_m[group]
                velocity = truth["com_velocity"][group] + np.cross(truth["angular_velocity"][group], common-reference_point)
                # Transport the same entry anchors with the actual stored R/p;
                # 1e-10 checks record construction, not dynamics or COM motion.
                material_delta = p1+r1 @ np.linalg.solve(r0, surface-p0)-surface
                application_delta = p1+r1 @ np.linalg.solve(r0, common-p0)-common
            rebuilt = rebuilt + (np.zeros(3) if skin is None else skin)
            close(rebuilt, surface, atol=1e-10, rtol=0)
            close(getattr(site, side+"_reference_point_m"), reference_point, atol=1e-10, rtol=0)
            close(getattr(site, side+"_velocity_m_s"), velocity)
            close(getattr(site, side+"_material_displacement_m"), material_delta, atol=1e-10, rtol=0)
            close(getattr(site, side+"_application_displacement_m"), application_delta, atol=1e-10, rtol=0)
            force = np.asarray(getattr(site, side+"_force_n"))
            contribution = np.r_[force, np.cross(common-reference_point, force)]
            if group is None:
                reactions.setdefault(key, np.zeros(6))
                reactions[key] += contribution
            else:
                wrench[group] += contribution
            application_work += float(force @ application_delta)
            material_work += float(force @ material_delta)
            sides[side] = dict(key=key, group=group, material_local_m=material_local,
                               material_delta_m=material_delta, application_delta_m=application_delta,
                               velocity_m_s=velocity, com_wrench=contribution)
        close(site.application_work_j, application_work, atol=1e-12, rtol=1e-10)
        close(site.material_work_j, material_work, atol=1e-12, rtol=1e-10)
        total_application += application_work
        total_material += material_work
        detail.append(dict(pair=pair, sides=sides, application_work_j=application_work, material_work_j=material_work))
    assert len(receipt.sites) == 1 and len(weights) == 1, "frozen sphere/finite-face or sphere pair has one site"
    close(tuple(weights.values()), [1.], atol=1e-12, rtol=0)
    close(receipt.law_wrench_fp64, wrench)
    np.testing.assert_array_equal(receipt.uploaded_wrench_fp32, wrench.astype(np.float32))
    assert set(receipt.static_reactions) == set(reactions)
    for key, expected in reactions.items():
        actual = receipt.static_reactions[key]
        close(actual.reference_point_m, sources[key][2].mean(axis=0), atol=1e-12, rtol=0)
        close(actual.force_n, expected[:3])
        close(actual.torque_nm, expected[3:])
    return dict(application_work_j=total_application, material_work_j=total_material, details=detail)


def floating_audit(data, entry, row, world, sources):
    after, receipt = row["state"][world], row["receipts"][world]
    n, b, d = len(data["parents"]), len(data["parents"])+1, len(data["parents"])+6
    assert receipt.world == world and receipt.entry_state == entry and receipt.accepted_state == after
    assert (receipt.entry_tick, receipt.exit_tick, receipt.dt_s) == (entry.tick, entry.tick+1, H)
    assert receipt.selected_drive.shape == (n, 3) and receipt.actual_drive.shape == (n,)
    assert receipt.law_wrench_fp64.shape == receipt.uploaded_wrench_fp32.shape == (b, 6)
    assert receipt.input_external.shape == receipt.contact_generalized.shape == (d,)
    q, v, truth = reference(data, entry.position, entry.velocity)
    close(after.position, q)
    close(after.velocity, v)
    close(receipt.selected_drive, data["drive"], atol=0, rtol=0)
    close(receipt.actual_drive, truth["drive"])
    close(receipt.input_external, np.zeros(d))
    close(receipt.law_wrench_fp64, truth["wrench"])
    close(receipt.contact_generalized, truth["contact_generalized"])
    diag = receipt.diagnostics
    assert diag["sampling"] == "discrete_entry" and diag["coordinate_kind"] == "floating"
    assert tuple(diag[k] for k in ("joint_count", "body_count", "velocity_count", "position_count")) == (n, b, d, n+7)
    for field in ("mass", "bias", "linear_jacobian", "angular_jacobian", "acceleration"):
        close(diag[field], truth[field], atol=2e-5 if field == "mass" else ATOL)
    close(diag["total_external"], truth["contact_generalized"])
    nested = diag["floating_diagnostic"]
    assert fingerprint(nested) == fingerprint(row["diagnostics"][world])
    assert (nested.entry_tick, nested.exit_tick, nested.dt_s) == (entry.tick, after.tick, H)
    close(nested.entry_position, entry.position)
    close(nested.entry_velocity, entry.velocity)
    for field in ("mass", "bias", "linear_jacobian", "angular_jacobian", "acceleration"):
        close(getattr(nested, field), diag[field], atol=0, rtol=0)
    close(nested.external, diag["total_external"], atol=0, rtol=0)
    close(nested.drive, receipt.actual_drive, atol=0, rtol=0)
    assert diag["equation_residual"] == nested.equation_residual <= 2e-5
    assert nested.control_residual == diag["control_residual"] <= 5e-5
    applied = truth["contact_generalized"].copy()
    applied[6:] += truth["drive"]
    close(truth["mass"] @ diag["acceleration"]+truth["bias"], applied, atol=2e-5)
    close(after.generalized_applied, applied)
    close(after.applied_effort, truth["drive"])
    close(np.asarray(after.local_origin_m) + data["q"][:3], after.position[:3], atol=0, rtol=0)
    observed = mechanics(data, after.position, after.velocity)
    kin = row["geometry"][world]
    assert fingerprint(kin) == fingerprint(receipt.accepted_geometry)
    assert (kin.joint_count, kin.body_count, kin.velocity_count, kin.configuration_count) == (n, b, d, n+7)
    close(kin.position, after.position)
    close(kin.velocity, after.velocity)
    for public, independent in (("link_position_m", "position"), ("com_position_m", "com_position"),
                                ("link_rotation", "rotation"), ("linear_velocity_m_s", "com_velocity"),
                                ("angular_velocity_rad_s", "angular_velocity")):
        close(getattr(kin, public), observed[independent])
    works = material_audit(receipt, sources, truth)
    for field in ("application_work_j", "material_work_j"):
        close(diag[field], works[field], atol=1e-12, rtol=1e-10)
    power = sum(float(site.first_force_n @ site.first_velocity_m_s + site.second_force_n @ site.second_velocity_m_s)
                for site in receipt.sites)
    close(diag["entry_power_times_h_j"], H*power, atol=1e-12, rtol=1e-10)
    close(diag["entry_kinetic_energy_j"], truth["energy"])
    close(diag["accepted_kinetic_energy_j"], observed["energy"])
    close(diag["kinetic_energy_change_j"], observed["energy"]-truth["energy"])
    close(row["energy"][world]["kinetic_energy_j"], observed["energy"])
    if data["name"] == "F03":
        # Compare a COM-velocity residual, with the original FP32 velocity gate.
        close(observed["momentum"]/sum(data["mass_kg"]), np.zeros(3), atol=ATOL, rtol=0)
    readonly(receipt)
    readonly(kin)
    readonly(nested)
    return dict(same_entry=truth, accepted=observed, material_and_work=works)


def audit_step(data, entry, row, expected, acceptance, sources, model):
    assert row["acceptance"] == acceptance and set(row["state"]) == set(data)
    assert row["batch_stats"]["state_epoch"] == row["entry_stats"]["state_epoch"]+1
    assert row["batch_stats"]["current_buffer"] == 1-row["entry_stats"]["current_buffer"]
    audits = row.setdefault("audit", {})
    for world, original in data.items():
        before, after = entry[world], row["state"][world]
        if not acceptance[world]:
            assert after == before and row["status"][world] == "SITE_CAPACITY_EXCEEDED"
            assert fingerprint(row["geometry"][world]) == fingerprint(row["entry_geometry"][world])
            assert fingerprint(row["energy"][world]) == fingerprint(row["entry_energy"][world])
            # A rejected world keeps the previously published receipt, if any.
            assert fingerprint(row["receipts"].get(world)) == fingerprint(row["entry_receipts"].get(world))
            continue
        assert after.tick == before.tick+1
        assert after.control == before.control and after.history == before.history
        close(after.position, expected[world][0], atol=WINDOW_ATOL)
        close(after.velocity, expected[world][1], atol=WINDOW_ATOL)
        if original["name"] not in ("ordinary-P", "FreeSphere"):
            audits[world] = floating_audit(original, before, row, world, sources)
        elif world == 1207:
            same_q, same_v, truth = independent_advance(original, before.position, before.velocity)
            close(after.position, same_q)
            close(after.velocity, same_v)
            close(row["diagnostics"][world].acceleration, [1.])
            close(after.generalized_applied, [2.])
            close(row["geometry"][world].link_position_m, [[1., after.position[0], 0.]])
            close(row["geometry"][world].link_rotation, np.eye(3)[None])
            close(row["energy"][world]["kinetic_energy_j"], after.velocity[0]**2)
            audits[world] = truth
        else:
            same_q, same_v, truth = independent_advance(original, before.position, before.velocity)
            close(after.position, same_q)
            close(after.velocity, same_v)
            close(after.applied_effort, [0., 0., 0.])
            close(row["energy"][world]["kinetic_energy_j"], np.asarray(after.velocity) @ after.velocity)
            audits[world] = truth
    row["bounds_audit"] = bounds_audit(row, sources, model)
    row["checks_complete"] = True


def one_step(batch, plan, data, controls, expected, rows, sources, model, *, acceptance=None, label="forward"):
    acceptance = dict.fromkeys(data, True) if acceptance is None else acceptance
    entry = batch.snapshot()
    row = dict(label=label, input=controls, entry=entry, entry_broad=broad(plan.query()),
               entry_geometry=batch.tree_kinematics(), entry_energy=batch.summary(),
               entry_receipts=dict(plan.last_accepted), entry_status=dict(plan.last_status), entry_stats=batch.stats())
    rows.append(row)
    # Append original request and entry before the only physical call. A failure
    # retains its real prefix; there is no fallback advance or state overwrite.
    row["acceptance"] = batch.step_contacts(plan, H)
    row.update(capture(batch, plan))
    for world, original in data.items():
        if acceptance[world]:
            q, v, truth = independent_advance(original, *expected[world])
            expected[world] = (q, v)
            row.setdefault("independent_window", {})[world] = dict(q=q.copy(), v=v.copy(), truth=truth)
    audit_step(data, entry, row, expected, acceptance, sources, model)
    return row


def replay_equal(actual, original):
    # Store all fields in both rows. Only explicit nonphysical counters/epochs
    # are compared separately; no record or material field is normalized away.
    for field in ("input", "entry", "entry_geometry", "entry_energy", "acceptance", "state", "geometry", "diagnostics", "energy", "receipts", "status"):
        assert fingerprint(actual[field]) == fingerprint(original[field]), field
    for field in ("entry_broad", "broad"):
        for part in BROAD_PHYSICAL:
            assert fingerprint(actual[field][part]) == fingerprint(original[field][part]), field+"."+part
        assert actual[field]["state_epoch"] > original[field]["state_epoch"]
        assert actual[field]["topology_generation"] == original[field]["topology_generation"]
    for field in ("control_uploaded_bytes", "restore_uploaded_bytes", "snapshot_downloaded_bytes", "submitted_kernels", "synchronizations", "completed_steps", "accepted_world_steps", "rejected_world_steps"):
        assert actual["batch_stats"][field] >= original["batch_stats"][field]
    actual["complete_replay_equal"] = True


def run_window(ph, mode, output, root_record, *, sibling=False):
    worlds, controls, data = declarations(ph, sibling=sibling)
    label = "f03" if sibling else "w4"
    section = dict(inputs=data, controls=controls, world_order=tuple(worlds), h_s=H,
                   steps=STEPS, trace=[], replay=[], capacity=[], alignments=[], checks={})
    root_record[label] = section
    expected = initial_reference(data)
    batch = plan = None
    retained = retained_bytes = None
    try:
        section["first_literal_anchors"] = first_anchors(data)
        batch = ph.TyxWorldBatch(list(worlds.items()), mode, cache_root=output/(label+"-cache"))
        batch.set_controls(controls)
        model, bindings, sources, laws = contact_declarations(ph, batch, data)
        section.update(collision_model=model, bindings=bindings, source_shapes=sources, laws=laws,
                       initial_state=batch.snapshot(), initial_geometry=batch.tree_kinematics())
        assert tuple(batch.snapshot()) == tuple(worlds)
        before = batch.snapshot()
        before_stats = batch.stats()
        section["unsupported_sampling"] = rejected(
            lambda: batch.prepare_contacts(model, bindings, laws, sampling="swept", tree_dynamic_pairs=sibling),
            "unsupported sampling with a complete legitimate law mapping",
        )
        assert any(word in section["unsupported_sampling"]["message"].lower() for word in ("sampling", "ccd"))
        assert batch.snapshot() == before
        assert batch.stats()["state_epoch"] == before_stats["state_epoch"]
        assert batch.stats()["device_owned_bytes"] == before_stats["device_owned_bytes"]
        plan = batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=sibling)
        section["initial_broad"] = broad(plan.query())
        section["initial_stats"] = batch.stats()
        saved = saved_reference = None
        for index in range(STEPS):
            row = one_step(batch, plan, data, controls, expected, section["trace"], sources, model)
            if index == 0:
                retained = dict(receipts=row["receipts"], state=row["state"], geometry=row["geometry"], diagnostics=row["diagnostics"])
                retained_bytes = fingerprint(retained)
                readonly(retained)
            if not sibling and index+1 == REPLAY_FROM:
                saved = batch.snapshot()
                saved_reference = {w: (q.copy(), v.copy()) for w, (q, v) in expected.items()}
                section["restore_checkpoint"] = saved
        if not sibling:
            assert saved is not None and saved_reference is not None
            restore_before = batch.stats()
            section["before_restore"] = capture(batch, plan)
            batch.restore(saved)
            section["after_restore"] = dict(state=batch.snapshot(), geometry=batch.tree_kinematics(), energy=batch.summary(), stats=batch.stats())
            assert batch.snapshot() == saved
            assert batch.stats()["state_epoch"] == restore_before["state_epoch"]+1
            expected = saved_reference
            for index in range(REPLAY_FROM, STEPS):
                row = one_step(batch, plan, data, controls, expected, section["replay"], sources, model, label="restore-replay")
                replay_equal(row, section["trace"][index])
            # True cross-owner boundary, not forged private state. The other
            # instance has the same public world/type but no relation to plan.
            foreign = ph.TyxWorldBatch([(503, worlds[503])], mode, cache_root=output/"foreign-cache")
            try:
                foreign.set_controls({503: controls[503]})
                foreign_before = foreign.snapshot()
                main_before = batch.snapshot()
                section["foreign_owner"] = rejected(lambda: foreign.step_contacts(plan, H), "foreign batch contact plan")
                assert any(word in section["foreign_owner"]["message"].lower() for word in ("owner", "batch"))
                assert foreign.snapshot() == foreign_before and batch.snapshot() == main_before
            finally:
                foreign.close()
                foreign.close()
                section["foreign_closed_stats"] = foreign.stats()
                assert foreign.stats()["device_owned_bytes"] == 0
            for bank, limited_world in ((0, 503), (1, 91)):
                if batch.stats()["current_buffer"] != bank:
                    one_step(batch, plan, data, controls, expected, section["alignments"], sources, model,
                             label="real accepted alignment to bank "+str(bank))
                assert batch.stats()["current_buffer"] == bank
                capacities = {503: 128, 91: 128}
                capacities[limited_world] = 0
                transaction = dict(bank=bank, limited_world=limited_world, site_capacity=capacities, limited_rows=[], retry_rows=[])
                section["capacity"].append(transaction)
                transaction["entry_full_plan_receipts"] = dict(plan.last_accepted)
                previous_receipts = fingerprint(plan.last_accepted)
                limited = batch.prepare_contacts(model, bindings, laws, site_capacity=capacities)
                try:
                    acceptance = dict.fromkeys(worlds, True)
                    acceptance[limited_world] = False
                    rejected_row = one_step(batch, limited, data, controls, expected, transaction["limited_rows"],
                                            sources, model, acceptance=acceptance, label="capacity-zero")
                    assert rejected_row["entry_stats"]["current_buffer"] == bank
                    assert fingerprint(plan.last_accepted) == previous_receipts
                    transaction["limited_status"] = dict(limited.last_status)
                    transaction["full_plan_receipts_after_rejection"] = dict(plan.last_accepted)
                finally:
                    limited.close()
                    limited.close()
                    transaction["closed_plan_stats"] = limited.stats()
                    assert limited.stats()["device_owned_bytes"] == 0
                    assert limited.stats()["contact_device_owned_bytes"] == 0
                retry = one_step(batch, plan, data, controls, expected, transaction["retry_rows"], sources, model,
                                 label="unchanged full-capacity plan retry")
                assert retry["receipts"][limited_world].entry_state == rejected_row["entry"][limited_world]
                transaction["checks_complete"] = True
        assert retained is not None and fingerprint(retained) == retained_bytes
        readonly(retained)
        section["checks"].update(old_owned_unchanged=True, fixed_forward_steps=len(section["trace"]) == 40,
                                  full_replay_equal=sibling or len(section["replay"]) == 21,
                                  capacity_banks=sibling or [r["bank"] for r in section["capacity"]] == [0, 1])
        section["retained_initial_owned"] = retained
        section["final"] = capture(batch, plan)
        section["independent_final_state"] = expected
        assert batch.stats()["floating_world_count"] == (1 if sibling else 2)
        assert batch.stats()["submitted_kernels"] > 0
        assert batch.stats()["mode"] == mode and batch.stats()["floating_dynamics_kernel_launches"] > 0
        if mode == "cuda":
            assert batch.stats()["device_allocations"] > 0
        assert plan.stats()["floating_bounds_kernel_launches"] > 0
        if not sibling:
            assert len(section["alignments"]) <= 2
            for key in ORDER:
                wanted = STEPS + len(section["alignments"]) + 4 - int(key in (503, 91))
                assert section["final"]["state"][key].tick == wanted
    except Exception as error:
        section["failure"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        if batch is not None:
            try:
                section["retained_on_failure"] = dict(state=batch.snapshot(), geometry=batch.tree_kinematics(),
                                                        diagnostics=batch.tree_diagnostics(), energy=batch.summary(), stats=batch.stats())
                if plan is not None:
                    section["retained_on_failure"].update(receipts=dict(plan.last_accepted), status=dict(plan.last_status), plan_stats=plan.stats())
            except Exception as retained_error:
                section["retained_read_error"] = dict(type=type(retained_error).__name__, message=str(retained_error))
        raise
    finally:
        # Independent finally levels ensure the real batch closes even if plan
        # cleanup itself raises; cleanup failures are not rewritten as physics.
        try:
            if plan is not None:
                plan.close()
                plan.close()
                section["closed_plan_stats"] = plan.stats()
                assert plan.stats()["device_owned_bytes"] == 0
                assert plan.stats()["contact_device_owned_bytes"] == 0
        finally:
            if batch is not None:
                batch.close()
                batch.close()
                section["closed_batch_stats"] = batch.stats()
                assert batch.stats()["device_owned_bytes"] == 0
                assert batch.stats()["collision_plan_live_count"] == 0
        if plan is not None and batch is not None:
            section["read_after_close"] = rejected(lambda: batch.step_contacts(plan, H), "step after close")
    section["status"] = "passed"


def archive_identity(installed, expected):
    wheel = expected.resolve(strict=True)
    if wheel.suffix.lower() != ".whl":
        raise ValueError("--expected-wheel must name the actual wheel archive")
    with wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    direct_text = installed.read_text("direct_url.json")
    if direct_text is None:
        raise AssertionError("local wheel installation must retain direct_url.json")
    direct = json.loads(direct_text)
    origin = urlsplit(direct["url"])
    assert origin.scheme == "file" and origin.netloc in ("", "localhost")
    assert Path(url2pathname(origin.path)).resolve(strict=True) == wheel
    archive_info = direct["archive_info"]
    recorded = archive_info.get("hashes", {}).get("sha256")
    if recorded is None and archive_info.get("hash"):
        algorithm, _, recorded = archive_info["hash"].partition("=")
        assert algorithm == "sha256"
    if recorded is not None:
        assert recorded.lower() == digest
    # Installer hashes are optional under direct-URL metadata. Exact local
    # archive binding plus every installed member's bytes is mandatory here.
    site = Path(installed.locate_file("")).resolve(strict=True)
    product_members, metadata_members = [], []
    with ZipFile(wheel) as archive:
        members = {name for name in archive.namelist() if name.startswith("phymeo/") and not name.endswith("/")}
        actual = {path.relative_to(site).as_posix() for path in (site/"phymeo").rglob("*") if path.is_file()}
        assert members and actual == members, "installed product membership differs from actual wheel"
        for name in sorted(members):
            path = (site/name).resolve(strict=True)
            assert path.is_relative_to(site/"phymeo")
            content = path.read_bytes()
            assert content == archive.read(name), "installed content differs: "+name
            product_members.append(dict(name=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
        for name in archive.namelist():
            if ".dist-info/" in name and name.rsplit("/", 1)[-1] in ("METADATA", "WHEEL", "top_level.txt"):
                content = (site/name).read_bytes()
                assert content == archive.read(name), name
                metadata_members.append(dict(name=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
    return dict(path=str(wheel), bytes=wheel.stat().st_size, sha256=digest, direct_url=direct,
                installed_from=direct["url"], version=installed.version, installer_recorded_sha256=recorded,
                installed_product_files_verified=len(product_members), members=product_members, metadata_members=metadata_members)


def module_provenance(site, package, wheel):
    rows = []
    with ZipFile(wheel) as archive:
        for name, module in sorted(tuple(sys.modules.items())):
            if name != "phymeo" and not name.startswith("phymeo."):
                continue
            file = getattr(module, "__file__", None)
            assert file is not None, "no installed origin for "+name
            path = Path(file).resolve(strict=True)
            assert path.is_relative_to(package), "non-wheel module: "+name
            spec = getattr(module, "__spec__", None)
            assert spec is not None and spec.origin is not None
            assert Path(spec.origin).resolve(strict=True) == path
            search_paths = tuple(str(Path(p).resolve(strict=True)) for p in getattr(module, "__path__", ()))
            assert all(Path(p).is_relative_to(package) for p in search_paths)
            member = path.relative_to(site).as_posix()
            content = path.read_bytes()
            assert archive.read(member) == content, "module differs from actual wheel: "+member
            rows.append(dict(module=name, path=str(path), origin=spec.origin, package_paths=search_paths,
                             wheel_member=member, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
    assert rows
    return rows


def outside_checkout(path):
    return not any((parent/".git").exists() for parent in (path, *path.parents))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    for name in ("output", "wheel-site", "expected-package-root", "expected-wheel", "forbidden-checkout"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    assert __debug__ and sys.flags.isolated and sys.flags.dont_write_bytecode, "use python -I -B"
    for path in (args.output, args.wheel_site, args.expected_package_root, args.expected_wheel, args.forbidden_checkout):
        if not path.is_absolute():
            raise ValueError("all declared paths must be absolute")
    site, package = args.wheel_site.resolve(strict=True), args.expected_package_root.resolve(strict=True)
    script, output = Path(__file__).resolve(strict=True), args.output.resolve()
    forbidden = args.forbidden_checkout.resolve(strict=True)
    assert package == (site/"phymeo").resolve(strict=True)
    assert Path.cwd().resolve() == script.parent
    assert all(not path.is_relative_to(forbidden) for path in (script, site, output))
    assert outside_checkout(script.parent) and outside_checkout(site) and outside_checkout(output)
    assert "phymeo" not in sys.modules
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status="running", mode=args.mode, world_order=ORDER, errors=[],
                  script=dict(path=str(script), sha256=hashlib.sha256(script.read_bytes()).hexdigest()),
                  expected_package_root=str(package), forbidden_checkout=str(forbidden),
                  gates=dict(h_s=H, forward_steps=40, replay_from=19, replay_steps=21, f03_steps=40,
                             transaction_banks=(0, 1), maximum_alignment_steps=2, payload_bytes_cap=MAX_BYTES,
                             state_atol=ATOL, state_rtol=RTOL, window_atol=WINDOW_ATOL,
                             mass_atol=2e-5, equation_residual=2e-5, control_residual=5e-5,
                             material_record_construction_atol_m=1e-10),
                  units=dict(configuration="root xyz m, body-to-world XYZW, Px joint m",
                             velocity="root COM world m/s, world rad/s, Px joint m/s",
                             force="N", torque="N*m about actual COM", energy="J", time="s"),
                  limits="Fixed C10 installed-wheel subset: ordinary floating, finite HOST mu=0 entry contact, TYX full-tree response. Not Newton, a performance result, all-device narrow phase, CCD, affine floating or persistent friction.")
    sys.path.insert(0, str(site))
    installed = None
    try:
        import phymeo as ph

        assert Path(ph.__file__).resolve(strict=True).parent == package
        installed = distribution("phymeo")
        assert Path(installed.locate_file("")).resolve(strict=True) == site
        record["wheel_before"] = archive_identity(installed, args.expected_wheel)
        record["modules_before"] = module_provenance(site, package, args.expected_wheel)
        record["versions"] = dict(phymeo=installed.version, tyx=distribution("tyx-compute").version,
                                   numpy=np.__version__, python=sys.version)
        # Fixed order; a real failure is preserved, never replaced with a rerun.
        run_window(ph, args.mode, output, record)
        run_window(ph, args.mode, output, record, sibling=True)
        record["status"] = "passed"
    except Exception as error:
        record["status"] = "failed"
        record["errors"].append(dict(stage="import_or_fixed_windows", type=type(error).__name__,
                                     message=str(error), traceback=traceback.format_exc()))
    try:
        if installed is not None:
            record["wheel_after"] = archive_identity(installed, args.expected_wheel)
            assert record["wheel_after"] == record["wheel_before"]
        if "phymeo" in sys.modules:
            record["modules_after"] = module_provenance(site, package, args.expected_wheel)
    except Exception as error:
        record["status"] = "failed"
        record["errors"].append(dict(stage="final_provenance", type=type(error).__name__,
                                     message=str(error), traceback=traceback.format_exc()))
    payload = json.dumps(encode(record), allow_nan=False, separators=(",", ":")).encode()
    path = output/"resident-floating-contact-wheel.trace.json.gz"
    trace = dict(path=str(path), uncompressed_bytes=len(payload), complete=False)
    if len(payload) > MAX_BYTES:
        record["status"] = "failed"
        record["errors"].append(dict(stage="resource_stop", type="PayloadCapacityExceeded",
                                     message="complete payload exceeds frozen 256 MiB; no success or truncated trace published"))
    else:
        # IO failure is not a physical rollback. Preserve its own type and any
        # partial temporary file; never remove earlier evidence.
        temporary = path.with_name(path.name+".tmp")
        try:
            with gzip.open(temporary, "wb") as stream:
                stream.write(payload)
            temporary.replace(path)
            trace.update(complete=True, compressed_bytes=path.stat().st_size,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         uncompressed_sha256=hashlib.sha256(payload).hexdigest())
        except Exception as error:
            record["status"] = "failed"
            record["errors"].append(dict(stage="archive_io", type=type(error).__name__,
                                         message=str(error), traceback=traceback.format_exc()))
    summary = dict(status=record["status"], raw_exit=int(record["status"] != "passed"), mode=args.mode,
                   wheel=record.get("wheel_before"), verified_module_count=len(record.get("modules_after", ())),
                   trace=trace, windows={}, errors=record["errors"], gates=record["gates"], limits=record["limits"])
    for label in ("w4", "f03"):
        section = record.get(label, {})
        summary["windows"][label] = dict(
            status=section.get("status", "not_completed"),
            completed_forward=sum(bool(row.get("checks_complete")) for row in section.get("trace", ())),
            completed_replay=sum(bool(row.get("complete_replay_equal")) for row in section.get("replay", ())),
            completed_capacity_banks=sum(bool(row.get("checks_complete")) for row in section.get("capacity", ())),
            real_alignment_steps=sum(bool(row.get("checks_complete")) for row in section.get("alignments", ())),
            checks=section.get("checks", {}), closed_batch_stats=section.get("closed_batch_stats"),
            closed_plan_stats=section.get("closed_plan_stats"), failure=section.get("failure"),
        )
    (output/"summary.json").write_text(json.dumps(encode(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(encode(summary), allow_nan=False))
    return summary["raw_exit"]


if __name__ == "__main__":
    raise SystemExit(main())
