"""C8 same-tree double-dynamic contact, tests-first / UNEXECUTED.

The parent owns every project execution. Full physical inputs and independent
oracles remain separate from the actual device geometry and contact receipts.
"""

import copy
import gzip
import importlib.util
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_resident_affine_contacts import _audit_sites

import phymeo as ph
from phymeo.geometry.contact_patch_adapter import contact_patch_sample
from phymeo.solvers.contact_patch_loads import sample_patch_loads

H = 1 / 1024
WORLD = 503
ATOL, RTOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-4


def _json(value):
    if is_dataclass(value):
        return {f.name: _json(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def _save(directory, name, payload):
    with gzip.open(directory / (name + ".json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(_json(payload), stream, allow_nan=False, separators=(",", ":"))


def _close(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def _d01(*, velocity=(0, 0), delay=0, kp=0, kd=0, cap=8):
    model = ph.RobotModel(
        np.array([-1, -1]),
        ("prismatic", "prismatic"),
        np.array([[0, 0, 0], [7 / 32, 0, 0]]),
        np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        np.tile([1.0, 0.0, 0.0], (2, 1)),
        np.array([2.0, 1.0]),
        np.zeros((2, 3)),
        np.full((2, 3), 1 / 8),
        np.zeros(3),
    )
    program = ph.DrivenRobotProgram(
        model,
        kp=np.full(2, kp, dtype=float),
        kd=np.full(2, kd, dtype=float),
        effort_limit=np.full(2, cap, dtype=float),
        delay_steps=delay,
    )
    state = program.initial_state(ph.RobotState(np.zeros(2), np.array(velocity, float)))
    world = ph.FixedRobotTree(program, state, control_max_iterations=512)
    command = program.inputs(
        np.zeros(2), np.zeros(2), feedforward=np.zeros(2), loads=ph.RobotInputs.zeros(2)
    )
    return program, world, command


def _two_spheres(batch, *, world=WORLD, groups=(0, 1), local=None, reverse=False):
    local = (np.zeros((1, 3)), np.zeros((1, 3))) if local is None else local
    keys = (ph.CollisionKey(world, 101, 7), ph.CollisionKey(world, 309, 2))
    shapes = tuple(
        ph.CollisionShape(k, "sphere", (11 + 19 * i,), radius=1 / 8) for i, k in enumerate(keys)
    )
    topology = batch.snapshot([world])[world].topology
    bindings = tuple(
        ph.ResidentTreeBinding(k, world, g, p, topology=topology)
        for k, g, p in zip(keys, groups, local, strict=True)
    )
    if reverse:
        shapes, bindings = tuple(reversed(shapes)), tuple(reversed(bindings))
    return ph.CollisionModel(shapes), bindings, keys


@pytest.mark.tyx
def test_d01_public_double_dynamic_tree_follows_two_sided_two_newton_patch(tyx_mode, tmp_path):
    """The valid RED must reach the new opt-in, after old HOST/device premises."""
    program, world, command = _d01()
    law = ph.RobotContactMaterial(64, 0, 0)
    record = dict(
        h_s=H,
        mode=tyx_mode,
        physical_model=program.model,
        initial=world.state,
        command=command,
        law=law,
        independent=dict(
            mass=[[2, 0], [0, 1]], gap=-1 / 32, force=[[-2, 0, 0], [2, 0, 0]], acceleration=[-1, 2]
        ),
    )
    with ph.TyxWorldBatch([(WORLD, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: command})
        before, geometry = batch.snapshot()[WORLD], batch.tree_kinematics()[WORLD]
        model, bindings, pair = _two_spheres(batch)
        points = (np.array([[0.0, 0.0, 0.0]]), np.array([[7 / 32, 0.0, 0.0]]))
        assert tuple(program.model.parents) == (-1, -1)
        assert before.position == before.velocity == (0.0, 0.0)
        _close(geometry.link_position_m, [[0, 0, 0], [7 / 32, 0, 0]], atol=1e-12, rtol=0)
        patch = contact_patch_sample(model.shapes[0], model.shapes[1], *points)
        assert patch is not None and len(patch.sites) == 1
        assert patch.sites[0].witness.gap_m == -1 / 32
        assert sum(s.weight for s in patch.sites) == 1
        sampled = sample_patch_loads(patch, law, np.zeros((1, 3)), np.zeros((1, 3)))
        _close(sampled["first_force"], [[-2, 0, 0]], atol=1e-12, rtol=0)
        _close(sampled["second_force"], [[2, 0, 0]], atol=1e-12, rtol=0)
        record.update(
            before=before,
            geometry=geometry,
            collision_model=model,
            bindings=bindings,
            host_patch=patch,
            host_loads=sampled,
        )
        try:
            # No new HOST bridge is called before this public capability boundary.
            with batch.prepare_contacts(
                model, bindings, {pair: law}, tree_dynamic_pairs=True
            ) as plan:
                assert batch.step_contacts(plan, H) == {WORLD: True}
                receipt = plan.last_accepted[WORLD]
                record.update(receipt=receipt)
                after = batch.snapshot()[WORLD]
                assert after.tick == before.tick + 1
                _close(after.velocity, np.array([-1.0, 2.0]) * H)
                _close(after.position, np.array([-1.0, 2.0]) * H * H)
                _close(receipt.contact_generalized, [-2, 2])
                _close(receipt.law_wrench_fp64, [[-2, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]])
                _close(receipt.diagnostics["mass"], np.diag([2.0, 1.0]), atol=2e-5)
                _close(receipt.diagnostics["acceleration"], [-1, 2])
                assert receipt.diagnostics["tree_dynamic_pairs"] is True
                assert not receipt.static_reactions
                assert after.control == before.control
                _close(2 * after.velocity[0] + after.velocity[1], 0, atol=1e-12, rtol=0)
                assert batch.stats()["submitted_kernels"] > 0
        finally:
            record.update(retained=batch.snapshot(), stats=batch.stats())
            _save(tmp_path, "d01-public-double-dynamic", record)


# Independent full-component reference is accepted C5 test infrastructure. It is
# loaded only when a case runs; no product FK, M, J or solve constructs an oracle.


@lru_cache(maxsize=1)
def _reference():
    path = (
        Path(__file__).resolve().parents[1] / "reference_newton/robot/resident_affine_reference.py"
    )
    spec = importlib.util.spec_from_file_location("c8_independent_components", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case(name, *, shape="sphere", placement="base", velocity=None):
    n = 3 if name in ("D02", "D05", "D05_flat") else 2
    parents, kinds = [-1] * n, ["prismatic"] * n
    offsets, axes = np.zeros((n, 3)), np.tile([1.0, 0.0, 0.0], (n, 1))
    offsets[-1, 0] = 7 / 32
    mass = [2.0, 1.0, 1.0] if n == 3 else [2.0, 1.0]
    inertia = np.full((n, 3), 1 / 8)
    groups, local = (0, n - 1), (np.zeros(3), np.zeros(3))
    P, c, rows = np.eye(n), np.zeros(n), ()
    if name == "D02":
        parents, kinds, mass = [-1, 0, 0], ["revolute", "prismatic", "prismatic"], [1.0, 2.0, 1.0]
        offsets[:] = [[0, 0, 0], [0, 1 / 4, 0], [7 / 32, 1 / 4, 0]]
        axes[:] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        inertia[0] = 1 / 4
        groups = (1, 2)
    elif name == "D03":
        kinds, axes = ["revolute"] * 2, np.tile([0.0, 0.0, 1.0], (2, 1))
        inertia[:] = [[1 / 4] * 3, [1 / 2] * 3]
        local = (np.array([0.0, 1 / 4, 0.0]), np.array([0.0, 1 / 4, 0.0]))
    elif name in ("D04", "D04Z"):
        gain = -1 / 2 if name == "D04" else 1
        P, c, rows = np.array([[1.0], [gain]]), np.array([0.0, 1 / 8]), ((0, 1, gain, 1 / 8),)
        offsets[1, 0] = 3 / 32
    elif name in ("D05", "D05_flat"):
        P, c = np.array([[1.0], [2.0], [-1.0]]), np.array([0.0, 1 / 8, 1 / 8])
        rows = (
            ((0, 1, 2.0, 1 / 8), (1, 2, -1 / 2, 3 / 16))
            if name == "D05"
            else ((0, 1, 2.0, 1 / 8), (0, 2, -1.0, 1 / 8))
        )
        offsets[1], offsets[2] = [0, 1, 0], [3 / 32, 0, 0]
    elif name != "D01":
        raise ValueError(name)
    flat_gap = -1 / 32
    if shape in ("cap_end", "cap_miss", "CC", "HH"):
        assert name == "D01"
        axes[:] = [0, 0, 1]
        separation = {
            "cap_end": 1 / 4 + 7 / 32,
            "cap_miss": 1 / 4 + 9 / 32,
            "CC": 0.09,
            "HH": 0.19,
        }[shape]
        offsets[1] = [0, 0, separation]
        flat_gap = 1 / 32 if shape == "cap_miss" else -0.01 if shape in ("CC", "HH") else -1 / 32
    model = _reference()._model(parents, kinds, offsets, axes, mass, np.zeros((n, 3)), inertia)
    rotation, shift = np.eye(3), np.zeros(3)
    if placement != "base":
        if placement == "z":
            angle = 0.37
            rotation = _reference().rotation(np.array([0.0, 0.0, 1.0]), angle)
            quat = [0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]
        else:
            assert placement == "cycle"
            rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            quat, shift = [0.5, 0.5, 0.5, 0.5], np.array([0.5, -0.25, 0.75])
        for i, parent in enumerate(parents):
            if parent < 0:
                model["offsets"][i] = rotation @ model["offsets"][i] + shift
                model["orientations"][i] = quat
    k = P.shape[1]
    C = np.zeros((len(rows), n))
    for i, (driver, follower, gain, _) in enumerate(rows):
        C[i, driver], C[i, follower] = -gain, 1
    return dict(
        name=name,
        model=model,
        P=P,
        c=c,
        C=C,
        rows=rows,
        independent=tuple(range(k)),
        s=np.zeros(k),
        u=np.zeros(k) if velocity is None else np.array(velocity, float),
        motor=np.zeros(k),
        effort=np.zeros(n),
        force=np.zeros((n, 3)),
        torque=np.zeros((n, 3)),
        lower=np.full(n, -4.0),
        upper=np.full(n, 4.0),
        speed=np.full(n, 64.0),
        rating=np.full(n, 8.0),
        groups=groups,
        sphere_local=local,
        shape=shape,
        flat_gap=flat_gap,
        rotation=rotation,
        shift=shift,
        placement=placement,
    )


def _build(data, directory, *, reorder=False, delay=0, kp=0.0, kd=0.0, cap=8.0):
    m, k = data["model"], data["P"].shape[1]
    n = len(m["mass"])
    if data["rows"]:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "frozen.urdf"
        path.write_text(_reference().urdf(data, reorder=reorder), encoding="utf-8")
        asset = ph.load_mimic_urdf(path, gravity=m["gravity"])
        order = [m["joint_names"].index(x) for x in asset.tree.joint_names]
        independent_names = [m["joint_names"][i] for i in data["independent"]]
        reduced_order = [independent_names.index(x) for x in asset.joint_names]
        world = ph.AffineRobotTree(
            asset,
            ph.RobotState(data["s"][reduced_order], data["u"][reduced_order]),
            kp=np.full(k, kp, float),
            kd=np.full(k, kd, float),
            effort_limit=np.full(k, cap, float),
            delay_steps=delay,
        )
        program = None
    else:
        asset, order, reduced_order = None, list(range(n)), list(range(n))
        model = ph.RobotModel(
            m["parents"],
            m["kinds"],
            m["offsets"],
            m["orientations"],
            m["axes"],
            m["mass"],
            m["com"],
            m["inertia"],
            m["gravity"],
        )
        program = ph.DrivenRobotProgram(
            model,
            kp=np.full(n, kp, float),
            kd=np.full(n, kd, float),
            effort_limit=np.full(n, cap, float),
            delay_steps=delay,
        )
        world = ph.FixedRobotTree(
            program,
            program.initial_state(ph.RobotState(data["s"], data["u"])),
            control_max_iterations=512,
        )
    item = dict(
        data=data,
        asset=asset,
        program=program,
        world=world,
        order=order,
        reduced_order=reduced_order,
    )
    item["control"] = _command(item)
    return item


def _command(item, *, target=None, rate=None, motor=None):
    d, order, reduced = item["data"], item["order"], item["reduced_order"]
    k = len(reduced)
    target = np.zeros(k) if target is None else np.asarray(target, float)
    rate = np.zeros(k) if rate is None else np.asarray(rate, float)
    motor = d["motor"] if motor is None else np.asarray(motor, float)
    loads = ph.RobotInputs(d["effort"][order], d["force"][order], d["torque"][order])
    owner = item["world"] if item["asset"] is not None else item["program"]
    return owner.inputs(target[reduced], rate[reduced], feedforward=motor[reduced], loads=loads)


def _box(lo, hi):
    return np.array(list(product(*zip(lo, hi, strict=True))), float)


def _geometry(batch, item, *, world=WORLD, reverse=False, two_pairs=False):
    data, order = item["data"], item["order"]
    a, b = data["groups"]
    shape = data["shape"]
    kinds, radius = ["sphere"] * 2, [1 / 8] * 2
    local = [np.asarray(p).reshape(1, 3) for p in data["sphere_local"]]
    if shape == "hull":
        kinds, radius = ["hull"] * 2, [0.0] * 2
        local = [_box([-1 / 8] * 3, [1 / 8] * 3)] * 2
    elif shape.startswith("cap_"):
        kinds[0], local[0] = "capsule", np.array([[0, 0, -1 / 4], [0, 0, 1 / 4]], float)
    elif shape == "CC":
        kinds, radius = ["capsule"] * 2, [0.05] * 2
        local = [np.array([[-0.5, 0, 0], [0.5, 0, 0]], float)] * 2
    elif shape == "HH":
        kinds, radius = ["hull"] * 2, [0.0] * 2
        ring = ((-0.5, -0.1), (0.5, -0.1), (0.6, 0), (0.3, 0.1), (-0.5, 0.1))
        local = [np.array([(x, y, z) for x, y in ring for z in (-0.1, 0.1)], float)] * 2
    topology = batch.snapshot([world])[world].topology
    binding_type = (
        ph.ResidentAffineTreeBinding if item["asset"] is not None else ph.ResidentTreeBinding
    )
    sources, shapes, bindings, laws = {}, [], [], {}
    for station, dz in enumerate((-1 / 8, 1 / 8) if two_pairs else (0.0,)):
        keys = (
            ph.CollisionKey(world, 101 + 208 * a, 7 + station),
            ph.CollisionKey(world, 101 + 208 * b, 2 + station),
        )
        for side, (original, key_) in enumerate(zip((a, b), keys, strict=True)):
            points = local[side] + [0, 0, dz]
            ids = tuple(1000 * original + 100 * station + 11 + 3 * j for j in range(len(points)))
            registered = ph.CollisionShape(key_, kinds[side], ids, radius[side], revision=7)
            group = order.index(original)
            shapes.append(registered)
            bindings.append(binding_type(key_, world, group, points, topology=topology))
            sources[key_] = (registered, group, points)
        laws[tuple(sorted(keys))] = ph.RobotContactMaterial(64, 0, 0)
    excluded = tuple(
        (x.key, y.key)
        for x, y in combinations(shapes, 2)
        if x.key.body != y.key.body and tuple(sorted((x.key, y.key))) not in laws
    )
    if reverse:
        shapes.reverse()
        bindings.reverse()
    return ph.CollisionModel(tuple(shapes), excluded=excluded), tuple(bindings), sources, laws


def _state(item, checkpoint):
    inverse = np.argsort(item["reduced_order"])
    return np.asarray(checkpoint.position)[inverse], np.asarray(checkpoint.velocity)[inverse]


def _truth(data, s, u, *, damping=0.0, motor=None, multiplicity=1):
    """Exact sphere distance or frozen translating finite-feature gap + C5 M/b/J."""
    q, v = data["P"] @ s + data["c"], data["P"] @ u
    full = _reference().oracle(data["model"], q, v)
    a, b = data["groups"]
    if data["shape"] == "sphere":
        points = [
            full["position"][g] + full["rotation"][g] @ local
            for g, local in zip((a, b), data["sphere_local"], strict=True)
        ]
        delta = points[1] - points[0]
        length = np.linalg.norm(delta)
        assert length > 0
        normal, gap = delta / length, length - 1 / 4
        common = (points[0] + points[1]) / 2
    else:
        normal = data["rotation"] @ data["model"]["axes"][a]
        # Both branches translate along this normal; finite cap/line/face remains
        # the declared feature. No product nearest point enters this oracle.
        gap = data["flat_gap"] + (q[b] - data["c"][b]) - (q[a] - data["c"][a])
        common = (full["com_position"][a] + full["com_position"][b]) / 2
        if data["shape"] == "HH":
            # Shoelace on the declared five-vertex section: area=1/5,
            # first moments=(1/1500,-1/1500); area centroid=(1/300,-1/300).
            common += data["rotation"] @ np.array([1 / 300, -1 / 300, 0.0])
    velocities = [
        full["com_velocity"][g]
        + np.cross(full["angular_velocity"][g], common - full["com_position"][g])
        for g in (a, b)
    ]
    speed = float(normal @ (velocities[1] - velocities[0]))
    normal_force = max(0.0, -64 * gap - damping * speed) if gap < 0 else 0.0
    applied = copy.deepcopy(data)
    contact = np.zeros(len(q))
    force, torque = np.zeros((len(q), 3)), np.zeros((len(q), 3))
    for group, sign in ((a, -1), (b, 1)):
        force[group] = sign * multiplicity * normal_force * normal
        torque[group] = np.cross(common - full["com_position"][group], force[group])
        contact += (
            full["linear_jacobian"][group].T @ force[group]
            + full["angular_jacobian"][group].T @ torque[group]
        )
    applied["force"] += force
    applied["torque"] += torque
    truth = _reference().kkt(applied, s, u, motor)
    truth.update(
        contact=contact,
        force=force,
        torque=torque,
        gap=gap,
        normal_force=normal_force,
        common=common,
    )
    return truth


def _audit(
    batch,
    item,
    entry,
    receipt,
    sources,
    *,
    world=WORLD,
    damping=0.0,
    motor=None,
    multiplicity=1,
    h=H,
):
    data, order, reduced = item["data"], item["order"], item["reduced_order"]
    s, u = _state(item, entry)
    truth = _truth(data, s, u, damping=damping, motor=motor, multiplicity=multiplicity)
    actual_s, actual_u = _state(item, receipt.accepted_state)
    _close(actual_u, u + h * truth["acceleration"])
    _close(actual_s, s + h * (u + h * truth["acceleration"]))
    diag = receipt.diagnostics
    full = truth["full"]
    _close(diag["mass"], full["mass"][np.ix_(order, order)], atol=2e-5)
    _close(diag["bias"], full["bias"][order])
    _close(diag["linear_jacobian"], full["linear_jacobian"][order][:, :, order])
    _close(diag["angular_jacobian"], full["angular_jacobian"][order][:, :, order])
    _close(receipt.contact_generalized, truth["contact"][order])
    _close(receipt.law_wrench_fp64[:, :3], truth["force"][order])
    _close(receipt.law_wrench_fp64[:, 3:], truth["torque"][order])
    if item["asset"] is not None:
        affine = diag["affine_diagnostic"]
        names = data["model"]["joint_names"]
        declared = [(names[d], names[f], a, b) for d, f, a, b in data["rows"]]
        rows = [
            declared.index((r.driver, r.follower, r.multiplier, r.offset))
            for r in item["asset"].relations
        ]
        _close(affine.full_acceleration, truth["full_acceleration"][order])
        _close(affine.constraint_effort_full, truth["reaction"][order])
        _close(affine.row_multipliers, truth["multipliers"][rows])
        _close(
            affine.row_powers_w,
            truth["multipliers"][rows] * (data["C"] @ truth["v"])[rows],
            atol=5e-5,
        )
        assert diag["graph_token"] == entry.topology[1]
        _close(diag["projection_matrix"], data["P"][order][:, reduced], atol=2e-5)
        _close(affine.acceleration, truth["acceleration"][reduced])
    else:
        _close(diag["acceleration"], truth["full_acceleration"][order])
    assert diag["tree_dynamic_pairs"] is True and not receipt.static_reactions
    assert receipt.entry_state == entry and receipt.accepted_state == batch.snapshot([world])[world]
    assert receipt.accepted_state.tick == entry.tick + 1
    assert receipt.accepted_state.control == entry.control
    actual = SimpleNamespace(
        linear_jacobian=diag["linear_jacobian"], angular_jacobian=diag["angular_jacobian"]
    )
    observed = _audit_sites(receipt, actual, sources)
    # Common-point equal/opposite forces have zero total world wrench even when
    # the fixed base can react and the body COM moments are not zero separately.
    total_force, total_torque = np.zeros(3), np.zeros(3)
    for site in receipt.sites:
        for side in ("first", "second"):
            f = getattr(site, side + "_force_n")
            total_force += f
            total_torque += np.cross(site.geometry.load_point, f)
    _close(total_force, np.zeros(3), atol=1e-12, rtol=0)
    _close(total_torque, np.zeros(3), atol=1e-12, rtol=0)
    old_v, new_v = truth["v"], data["P"] @ actual_u
    end_terms = _reference().oracle(data["model"], data["P"] @ actual_s + data["c"], new_v)
    _close(diag["entry_kinetic_energy_j"], 0.5 * old_v @ full["mass"] @ old_v)
    _close(diag["accepted_kinetic_energy_j"], 0.5 * new_v @ end_terms["mass"] @ new_v)
    power = sum(
        float(
            site.first_force_n @ site.first_velocity_m_s
            + site.second_force_n @ site.second_velocity_m_s
        )
        for site in receipt.sites
    )
    _close(diag["entry_power_times_h_j"], h * power, atol=1e-12, rtol=0)
    _close(diag["application_work_j"], observed["application_work_j"], atol=1e-12, rtol=1e-10)
    return dict(independent=truth, actual_material_observation=observed)


EXACT = {
    "D01": ([-1, 2], [], [-2, 2]),
    "D02": ([-8 / 9, -11 / 9, 7 / 36], [], [0, -2, 0]),
    "D03": ([2, -1], [], [0.5, -0.5]),
    "D04": ([-4 / 3], [-4 / 3], [-2, 2]),
    "D04Z": ([0], [-2], [-2, 2]),
    "D05": ([-4 / 7], [-3 / 7, -10 / 7], [-2, 0, 2]),
    "D05_flat": ([-4 / 7], [-8 / 7, -10 / 7], [-2, 0, 2]),
}


def _window(item, directory, mode, *, steps=40, reverse=False, damping=0.0, multiplicity=1):
    directory.mkdir(parents=True, exist_ok=True)
    expected_s, expected_u = item["data"]["s"].copy(), item["data"]["u"].copy()
    rows, failure = [], None
    with ph.TyxWorldBatch([(WORLD, item["world"])], mode, cache_root=directory) as batch:
        batch.set_controls({WORLD: item["control"]})
        model, bindings, sources, laws = _geometry(
            batch, item, reverse=reverse, two_pairs=multiplicity == 2
        )
        laws = {pair: ph.RobotContactMaterial(64, damping, 0) for pair in laws}
        with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as plan:
            try:
                for _ in range(steps):
                    entry = batch.snapshot()[WORLD]
                    expected = _truth(
                        item["data"],
                        expected_s,
                        expected_u,
                        damping=damping,
                        multiplicity=multiplicity,
                    )
                    expected_u = expected_u + H * expected["acceleration"]
                    expected_s = expected_s + H * expected_u
                    assert batch.step_contacts(plan, H) == {WORLD: True}
                    receipt = plan.last_accepted[WORLD]
                    row = dict(
                        receipt=receipt,
                        actual_coordinates=_state(item, receipt.accepted_state),
                        expected_s=expected_s.copy(),
                        expected_u=expected_u.copy(),
                        expected=expected,
                        audit=_audit(
                            batch,
                            item,
                            entry,
                            receipt,
                            sources,
                            damping=damping,
                            multiplicity=multiplicity,
                        ),
                        stats=batch.stats(),
                    )
                    rows.append(row)
                    actual_s, actual_u = _state(item, receipt.accepted_state)
                    _close(actual_s, expected_s, atol=WINDOW_ATOL)
                    _close(actual_u, expected_u, atol=WINDOW_ATOL)
            except Exception as error:
                failure = dict(type=type(error).__name__, message=str(error))
                raise
            finally:
                _save(
                    directory,
                    "dynamic-window",
                    dict(
                        input=item["data"],
                        h_s=H,
                        steps=steps,
                        mode=mode,
                        control=item["control"],
                        collision_model=model,
                        bindings=bindings,
                        laws=laws,
                        rows=rows,
                        failure=failure,
                        retained=batch.snapshot(),
                        geometry=batch.tree_kinematics(),
                        last_accepted=plan.last_accepted,
                        status=plan.last_status,
                        stats=batch.stats(),
                        plan_stats=plan.stats(),
                    ),
                )
    return rows


@pytest.mark.tyx
@pytest.mark.parametrize("name", tuple(EXACT))
def test_d01_d05_independent_full_coupling_and_original_reaction(name, tyx_mode, tmp_path):
    data = _case(name)
    truth = _truth(data, data["s"], data["u"])
    a, multipliers, contact = EXACT[name]
    _close(truth["acceleration"], a, atol=1e-12, rtol=1e-12)
    _close(truth["multipliers"], multipliers, atol=1e-12, rtol=1e-12)
    _close(truth["contact"], contact, atol=1e-12, rtol=1e-12)
    if name == "D02":
        _close(
            truth["full"]["mass"],
            [[753 / 1024, -0.5, 7 / 32], [-0.5, 2, 0], [7 / 32, 0, 1]],
            atol=1e-12,
            rtol=0,
        )
    item = _build(data, tmp_path / "asset")
    _window(item, tmp_path, tyx_mode)


@pytest.mark.tyx
def test_d04z_positive_full_contact_zero_reduced_load_in_motion(tyx_mode, tmp_path):
    data = _case("D04Z", velocity=[1 / 8])
    rows = _window(_build(data, tmp_path / "asset"), tmp_path, tyx_mode)
    for row in rows:
        receipt = row["receipt"]
        _close(receipt.contact_generalized, [-2, 2])
        _close(receipt.diagnostics["affine_diagnostic"].row_multipliers, [-2])
        _close(receipt.accepted_state.velocity, [1 / 8])
        assert receipt.sites


@pytest.mark.tyx
@pytest.mark.parametrize(
    "name,placement", (("D02", "z"), ("D02", "cycle"), ("D03", "cycle"), ("D04", "cycle"))
)
def test_d02_d04_rigid_covariance_and_reversed_registration(name, placement, tyx_mode, tmp_path):
    base = _window(
        _build(_case(name), tmp_path / "base_asset"), tmp_path / "base", tyx_mode, steps=3
    )
    moved = _window(
        _build(_case(name, placement=placement), tmp_path / "moved_asset", reorder=True),
        tmp_path / "moved",
        tyx_mode,
        steps=3,
        reverse=True,
    )
    for a, b in zip(base, moved, strict=True):
        _close(a["expected_s"], b["expected_s"], atol=1e-12)
        _close(a["actual_coordinates"][0], b["actual_coordinates"][0])
        _close(a["actual_coordinates"][1], b["actual_coordinates"][1])


@pytest.mark.tyx
@pytest.mark.parametrize("shape", ("hull", "cap_side", "cap_end", "cap_miss", "CC", "HH"))
def test_d06_finite_axis_end_line_and_nonbox_face_load_once(shape, tyx_mode, tmp_path):
    data = _case("D01", shape=shape)
    magnitude = 0 if shape == "cap_miss" else 16 / 25 if shape in ("CC", "HH") else 2
    _close(_truth(data, data["s"], data["u"])["normal_force"], magnitude, atol=1e-12, rtol=0)
    rows = _window(_build(data, tmp_path / "asset"), tmp_path, tyx_mode, steps=2)
    receipt = rows[0]["receipt"]
    _close(receipt.contact_generalized, [-magnitude, magnitude])
    if magnitude:
        assert abs(sum(s.geometry.weight for s in receipt.sites) - 1) <= 1e-12
        if shape in ("hull", "HH", "CC"):
            assert len(receipt.sites) > 1
    else:
        assert not receipt.sites


@pytest.mark.tyx
@pytest.mark.parametrize(
    "velocity,magnitude", (((1 / 8, 1 / 8), 2), ((1 / 8, -1 / 8), 3), ((-1 / 8, 1 / 8), 1))
)
def test_d07_damping_reads_both_entry_velocities(velocity, magnitude, tyx_mode, tmp_path):
    data = _case("D01", velocity=velocity)
    _close(
        _truth(data, data["s"], data["u"], damping=4)["normal_force"], magnitude, atol=1e-12, rtol=0
    )
    row = _window(_build(data, tmp_path / "asset"), tmp_path, tyx_mode, steps=2, damping=4)[0]
    _close(row["receipt"].contact_generalized, [-magnitude, magnitude])


@pytest.mark.tyx
def test_d07_two_actual_pairs_cancel_body_moments_without_duplicate_force(tyx_mode, tmp_path):
    data = _case("D01")
    row = _window(_build(data, tmp_path / "asset"), tmp_path, tyx_mode, steps=2, multiplicity=2)[0]
    _close(row["receipt"].contact_generalized, [-4, 4])
    _close(row["receipt"].law_wrench_fp64[:, 3:], np.zeros((2, 3)), atol=1e-12)
    assert len(row["receipt"].sites) == 2


@pytest.mark.tyx
@pytest.mark.parametrize("name,delay", tuple(product(("D01", "D04"), (0, 1))))
def test_d07_pd_delay_full_external_force_and_low_motor_cap(name, delay, tyx_mode, tmp_path):
    data = _case(name)
    item = _build(data, tmp_path / "asset", delay=delay, kp=16.0, kd=2.0, cap=1 / 4)
    k = data["P"].shape[1]
    item["control"] = _command(item, target=np.full(k, 1 / 16), motor=np.ones(k))
    rows = []
    with ph.TyxWorldBatch([(WORLD, item["world"])], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: item["control"]})
        model, bindings, sources, laws = _geometry(batch, item)
        with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as plan:
            try:
                for step in range(2):
                    before = batch.snapshot()[WORLD]
                    s, u = _state(item, before)
                    base = _truth(data, s, u, motor=np.zeros(k))
                    # These frozen mechanisms have diagonal reduced M; this
                    # scalar implicit-PD equation is independent of the solver.
                    mass = np.diag(base["mass"])
                    _close(base["mass"], np.diag(mass), atol=1e-12, rtol=0)
                    target = np.zeros(k) if delay and step == 0 else np.full(k, 1 / 16)
                    ff = np.zeros(k) if delay and step == 0 else np.ones(k)
                    beta = H * H * 16 + H * 2
                    unconstrained = (
                        ff
                        + 16 * (target - s - H * u)
                        - 2 * u
                        - beta * (base["external"] - base["bias"]) / mass
                    ) / (1 + beta / mass)
                    motor = np.clip(unconstrained, -1 / 4, 1 / 4)
                    assert batch.step_contacts(plan, H) == {WORLD: True}
                    receipt = plan.last_accepted[WORLD]
                    rows.append(
                        dict(
                            receipt=receipt,
                            selected_target=target,
                            selected_ff=ff,
                            independent_motor=motor,
                            audit=_audit(batch, item, before, receipt, sources, motor=motor),
                        )
                    )
                    _close(receipt.actual_drive, motor[item["reduced_order"]], atol=5e-5)
                    assert np.max(np.abs(receipt.law_wrench_fp64[:, :3])) > 1 / 4
                    assert receipt.accepted_state.control == before.control
                    assert bool(receipt.accepted_state.history) == bool(delay)
            finally:
                _save(
                    tmp_path,
                    "dynamic-pd",
                    dict(
                        input=data,
                        kp=16,
                        kd=2,
                        cap=1 / 4,
                        delay=delay,
                        command=item["control"],
                        rows=rows,
                        retained=batch.snapshot(),
                        stats=batch.stats(),
                    ),
                )


@pytest.mark.tyx
def test_d05_chain_flatten_same_motion_original_rows_and_wrong_graph_restore(tyx_mode, tmp_path):
    chain = _build(_case("D05"), tmp_path / "chain_asset")
    flat = _build(_case("D05_flat"), tmp_path / "flat_asset")
    left = _window(chain, tmp_path / "chain", tyx_mode, steps=2)
    right = _window(flat, tmp_path / "flat", tyx_mode, steps=2)
    for a, b in zip(left, right, strict=True):
        _close(
            a["receipt"].accepted_state.position,
            b["receipt"].accepted_state.position,
            atol=1e-12,
            rtol=0,
        )
        _close(
            a["receipt"].accepted_state.full_velocity,
            b["receipt"].accepted_state.full_velocity,
            atol=1e-12,
            rtol=0,
        )
        assert a["receipt"].entry_state.topology != b["receipt"].entry_state.topology
        assert not np.array_equal(
            a["receipt"].diagnostics["affine_diagnostic"].row_multipliers,
            b["receipt"].diagnostics["affine_diagnostic"].row_multipliers,
        )
    with ph.TyxWorldBatch(
        [(WORLD, flat["world"])], tyx_mode, cache_root=tmp_path / "restore"
    ) as batch:
        before = batch.snapshot()
        with pytest.raises(ValueError, match="graph|topology|model"):
            batch.restore({WORLD: left[-1]["receipt"].accepted_state})
        assert batch.snapshot() == before
