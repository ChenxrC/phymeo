"""ROB1c: independent articulated/free contact, not an implicit inter-tree solve.

R02 literal first step: M01=-9/32, k*depth=2 N, arm/box COM
moments +(3/16)ez and -(1/8)ez for mu=0. R03 uses held controls
(outward 48 steps, inward 80), never prescribed positions or resets.
All oracles use transform derivatives and small exhaustive PD active faces;
no production FK/contact/integrator supplies the expected trajectory.
"""

import importlib.util
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pytest
from test_resident_multi_participant import close, save

import phymeo as ph

H = 1 / 1024
WORLDS = (503, 91, 1207, 6)
FAMILIES = ("fixed", "affine", "floating", "floating_affine")
_REFS = {}


def reference(name):
    if name not in _REFS:
        path = (
            Path(__file__).parents[1]
            / "reference_newton"
            / "robot"
            / ("resident_" + name + "_reference.py")
        )
        spec = importlib.util.spec_from_file_location("rob1c_" + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _REFS[name] = module
    return _REFS[name]


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def quat_z(angle):
    return np.array([0.0, 0, np.sin(angle / 2), np.cos(angle / 2)])


def model(parents, kinds, offsets, axes, masses, centers):
    return dict(
        parents=np.asarray(parents, int),
        kinds=tuple(kinds),
        offsets=np.asarray(offsets, float).reshape(-1, 3),
        orientations=np.tile([0.0, 0, 0, 1], (len(parents), 1)),
        axes=np.asarray(axes, float).reshape(-1, 3),
        mass=np.asarray(masses, float),
        com=np.asarray(centers, float).reshape(-1, 3),
        inertia=np.full((len(parents), 3), 1 / 8),
        gravity=np.zeros(3),
    )


def robot_model(m):
    return ph.RobotModel(
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


def spec(family, m, q, v, *, projection=None, offset=None, root_mass=4.0):
    floating = family.startswith("floating")
    n = len(m["mass"])
    P = np.eye(n) if projection is None else np.asarray(projection, float)
    k = P.shape[1]
    S = np.zeros((6 + n, 6 + k)) if floating else P.copy()
    if floating:
        S[:6, :6], S[6:, 6:] = np.eye(6), P
    bodies = n + int(floating)
    return dict(
        family=family,
        model=m,
        q=np.asarray(q, float),
        v=np.asarray(v, float),
        P=P,
        c=np.zeros(n) if offset is None else np.asarray(offset, float),
        S=S,
        root_mass=root_mass,
        root_inertia=np.full(3, 1 / 8),
        root_com=np.zeros(3),
        root_inertia_frame=np.array([0.0, 0, 0, 1]),
        kp=np.full(k, 4.0),
        kd=np.full(k, 0.5),
        caps=np.full(k, 64.0),
        drive=np.zeros((k, 3)),
        force=np.zeros((bodies, 3)),
        torque=np.zeros((bodies, 3)),
        effort=np.zeros(n),
        shapes=[],
    )


def free_box(position, *, angle=0.0, eccentric=True, mass=1.0):
    m = model([], [], [], [], [], [])
    result = spec(
        "floating",
        m,
        np.r_[position, quat_z(angle)],
        [-1 / 64, 1 / 32, 0, 0, 0, 1 / 128],
        root_mass=mass,
    )
    result["shapes"] = [
        dict(
            body=0,
            kind="hull",
            center=np.array([0, 1 / 16, 0]) if eccentric else np.zeros(3),
            half=np.array([1 / 8, 3 / 16, 1 / 8]),
        )
    ]
    return result


def r02(mu=0.0):
    m = model(
        [-1, 0],
        ("revolute", "prismatic"),
        [[0, 0, 0], [0, 1 / 4, 0]],
        [[0, 0, 1], [1, 0, 0]],
        [2, 1],
        [[1 / 8, 0, 0], [1 / 16, 1 / 32, 0]],
    )
    arm = spec("fixed", m, [1 / 16, 1 / 32], [1 / 32, -1 / 64])
    arm["drive"] = np.array(
        [[1 / 16 + 1 / 64, 1 / 64, 1 / 32], [1 / 32 + 1 / 64, -1 / 128, 1 / 16]]
    )
    arm["shapes"] = [dict(body=1, kind="sphere", center=np.array([1 / 8, 1 / 8, 0]), radius=1 / 8)]
    R = rz(1 / 16)
    sphere = R @ np.array([1 / 32 + 1 / 8, 3 / 8, 0])
    box_center = sphere + R @ np.array([7 / 32, 0, 0])
    box = free_box(box_center - R @ np.array([0, 1 / 16, 0]), angle=1 / 16)
    box["force"][0] = [0, 1 / 32, 0]
    return dict(
        name="R02",
        specs={7: arm, 29: box},
        pairs=((7, 0, 29, 0),),
        stiffness=64.0,
        friction=mu,
        slip_speed=1 / 16,
        steps=40,
        switch=None,
    )


def r03(asymmetric=False):
    specs = {}
    # Hinge rotates around the push axis; its real inertia and nonzero speed
    # remain, while sphere centers stay on that axis. No joint is fixed/welded.
    for side, key in ((-1, 7), (1, 17)):
        x = side * 31 / 128
        y = (1 / 32 if side < 0 else -1 / 64) if asymmetric else 0.0
        m = model(
            [-1, 0],
            ("revolute", "prismatic"),
            [[x, y, 0], [0, 0, 0]],
            [[1, 0, 0], [1, 0, 0]],
            [2, 1],
            [[0, 0, 0], [0, 0, 0]],
        )
        arm = spec("fixed", m, [1 / 32, 0], [side / 64, 0])
        arm["kp"] = np.array([1.0, 0.0])
        arm["kd"] = np.array([0.5, 1024.0])
        arm["caps"] = np.array([8.0, 4096.0])
        arm["drive"] = np.array(
            [
                [1 / 32, side / 64, side / 128],
                [0, side * (3 / 4 if asymmetric and side > 0 else 1), 0],
            ]
        )
        arm["shapes"] = [dict(body=1, kind="sphere", center=np.zeros(3), radius=1 / 8)]
        arm["side"] = side
        specs[key] = arm
    tool = free_box([0, 0, 0], eccentric=False, mass=16.0)
    tool["q"][3:] = [0, 0, 0, 1]
    tool["v"][:] = 0
    tool["root_inertia"][:] = 1
    specs[29] = tool
    # Initial depth1/128. Outward travel~48/1024>depth by >5x;
    # subsequent inward travel80/1024 exceeds outward travel by1/32.
    return dict(
        name="R03",
        specs=specs,
        pairs=((7, 0, 29, 0), (17, 0, 29, 0)),
        stiffness=64.0,
        friction=0.0,
        slip_speed=1 / 16,
        steps=128,
        switch=48,
        asymmetric=asymmetric,
    )


def r04(family):
    affine = family in ("affine", "floating_affine")
    floating = family.startswith("floating")
    m = model(
        [-1, -1],
        ("prismatic", "prismatic"),
        [[-7 / 32 - 1 / 32, -1 / 16, 0], [-7 / 32 + 1 / 32, 1 / 16, 0]],
        [[1, 0, 0], [1, 0, 0]],
        [1, 1],
        [[0, 0, 0], [0, 0, 0]],
    )
    P = np.array([[1.0], [-1.0]]) if affine else np.eye(2)
    scalar_q = np.array([1 / 32]) if affine else np.array([1 / 32, -1 / 32])
    scalar_v = np.array([1 / 64]) if affine else np.array([1 / 64, -1 / 64])
    q = np.r_[np.zeros(3), [0, 0, 0, 1], scalar_q] if floating else scalar_q
    v = np.r_[np.zeros(6), scalar_v] if floating else scalar_v
    arm = spec(family, m, q, v, projection=P)
    arm["drive"] = np.column_stack((scalar_q + 1 / 128, scalar_v, np.full(len(scalar_q), 1 / 16)))
    arm["shapes"] = [
        dict(body=i + int(floating), kind="sphere", center=np.zeros(3), radius=1 / 8)
        for i in range(2)
    ]
    box = free_box([0, 0, 0], eccentric=False)
    box["v"][:] = 0
    return dict(
        name="R04",
        specs={7: arm, 29: box},
        pairs=((7, 0, 29, 0), (7, 1, 29, 0)),
        stiffness=64.0,
        friction=0.0,
        slip_speed=1 / 16,
        steps=16,
        switch=None,
    )


def urdf(s):
    m = s["model"]
    root = f'<link name="base"><inertial><mass value="{s["root_mass"]}"/><inertia ixx="0.125" iyy="0.125" izz="0.125" ixy="0" ixz="0" iyz="0"/></inertial></link>'
    links, joints = [], []
    for i in range(2):
        links.append(
            f'<link name="link{i}"><inertial><mass value="1"/><inertia ixx="0.125" iyy="0.125" izz="0.125" ixy="0" ixz="0" iyz="0"/></inertial></link>'
        )
        xyz = " ".join(str(float(x)) for x in m["offsets"][i])
        mimic = '<mimic joint="joint0" multiplier="-1" offset="0"/>' if i else ""
        parent = "base" if m["parents"][i] < 0 else "link" + str(m["parents"][i])
        joints.append(
            f'<joint name="joint{i}" type="prismatic"><parent link="{parent}"/><child link="link{i}"/><origin xyz="{xyz}"/><axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="64" velocity="10"/>{mimic}</joint>'
        )
    return '<robot name="rob1c_graph">' + root + "".join(links + joints) + "</robot>"


def declaration(s, directory):
    family, m = s["family"], s["model"]
    floating = family.startswith("floating")
    if family in ("affine", "floating_affine"):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "graph.urdf"
        path.write_text(urdf(s), encoding="utf-8")
        loader = ph.load_floating_mimic_urdf if floating else ph.load_mimic_urdf
        asset = loader(path, gravity=(0, 0, 0))
        assert tuple(asset.tree.joint_names) == ("joint0", "joint1")
        cls = ph.FloatingAffineRobotTree if floating else ph.AffineRobotTree
        state = ph.FloatingRobotState(s["q"], s["v"]) if floating else ph.RobotState(s["q"], s["v"])
        tree = cls(asset, state, kp=s["kp"], kd=s["kd"], effort_limit=s["caps"])
        owner = tree
    elif floating:
        physical = ph.FloatingRobotModel(
            ph.RigidBodyModel(np.array([1 / s["root_mass"]]), s["root_inertia"][None], np.zeros(3)),
            s["root_com"],
            None if not len(m["mass"]) else robot_model(m),
        )
        tree = ph.FloatingRobotTree(
            physical,
            ph.FloatingRobotState(s["q"], s["v"]),
            kp=s["kp"],
            kd=s["kd"],
            effort_limit=s["caps"],
        )
        owner = tree
    else:
        owner = ph.DrivenRobotProgram(
            robot_model(m), kp=s["kp"], kd=s["kd"], effort_limit=s["caps"]
        )
        tree = ph.FixedRobotTree(
            owner, owner.initial_state(ph.RobotState(s["q"], s["v"])), control_max_iterations=512
        )
    return tree, owner


def command(s, owner, drive=None):
    d = s["drive"] if drive is None else drive
    if s["family"].startswith("floating"):
        return owner.inputs(d, ph.FloatingRobotInputs(s["effort"], s["force"], s["torque"]))
    loads = ph.RobotInputs(s["effort"], s["force"], s["torque"])
    return owner.inputs(d[:, 0], d[:, 1], feedforward=d[:, 2], loads=loads)


def build(data, directory, *, worlds=WORLDS[:1], reverse=False):
    participants, bindings, shapes, controls, laws, owners, keys = [], [], [], {}, {}, {}, {}
    for world in worlds:
        for pid, s in data["specs"].items():
            tree, controller = declaration(s, directory / f"w{world}-p{pid}")
            key = ph.ResidentParticipantKey(world, pid)
            b = len(s["model"]["mass"]) + int(s["family"].startswith("floating"))
            p = ph.ResidentParticipant(key, tree, tuple(100 * pid + j for j in range(b)))
            participants.append(p)
            controls[key], owners[key] = command(s, controller), controller
            binding_class = dict(
                fixed=ph.ResidentTreeBinding,
                affine=ph.ResidentAffineTreeBinding,
                floating=ph.ResidentFloatingTreeBinding,
                floating_affine=ph.ResidentFloatingAffineTreeBinding,
            )[s["family"]]
            for si, shape in enumerate(s["shapes"]):
                group = shape["body"]
                c = ph.CollisionKey(world, p.body_ids[group], 10 + si)
                local = (
                    shape["center"][None]
                    if shape["kind"] == "sphere"
                    else (
                        np.array(list(product((-1, 1), repeat=3))) * shape["half"] + shape["center"]
                    )
                )
                registered = ph.CollisionShape(
                    c,
                    shape["kind"],
                    tuple(range(31, 31 + len(local))),
                    shape.get("radius", 0.0),
                    revision=1,
                )
                kwargs = dict(topology=p.compiled.topology)
                if p.graph_token is not None:
                    kwargs["graph_token"] = p.graph_token
                binding = binding_class(c, world, group, local, **kwargs)
                shapes.append(registered)
                bindings.append(ph.ResidentParticipantBinding(key, binding))
                keys[world, pid, si] = c
        for a, ai, b, bi in data["pairs"]:
            pair = tuple(sorted((keys[world, a, ai], keys[world, b, bi])))
            laws[pair] = ph.RobotContactMaterial(
                data["stiffness"], 0, data["friction"], data["slip_speed"]
            )
    excluded = tuple(
        (a.key, b.key)
        for a, b in combinations(shapes, 2)
        if a.key.world == b.key.world
        and a.key.body != b.key.body
        and tuple(sorted((a.key, b.key))) not in laws
    )
    if reverse:
        participants.reverse()
        bindings.reverse()
        shapes.reverse()
    return dict(
        participants=tuple(participants),
        controls=controls,
        controllers=owners,
        model=ph.CollisionModel(tuple(shapes), excluded),
        bindings=tuple(bindings),
        laws=laws,
        keys=keys,
        worlds=tuple(sorted(worlds)),
        data=data,
    )


def prepare(scene, built, **overrides):
    kwargs = dict(
        candidate_capacity=len(built["laws"]), site_capacity=128, narrow_phase="host_patch"
    )
    kwargs["intra_participant_pairs"] = built["data"].get("intra", False)
    kwargs.update(overrides)
    return scene.prepare_contacts(built["model"], built["bindings"], built["laws"], **kwargs)


def expand(s, q, v):
    if s["family"].startswith("floating"):
        return np.r_[q[:7], s["P"] @ q[7:] + s["c"]], s["S"] @ v
    return s["P"] @ q + s["c"], s["S"] @ v


def mechanics(s, q, v):
    full_q, full_v = expand(s, np.asarray(q), np.asarray(v))
    if s["family"].startswith("floating"):
        data = dict(
            children=s["model"],
            root_mass=s["root_mass"],
            root_com=s["root_com"],
            root_inertia=s["root_inertia"],
            root_inertia_frame=s["root_inertia_frame"],
            gravity=s["model"]["gravity"],
        )
        result = reference("floating").mechanics(data, full_q, full_v)
    else:
        result = reference("affine").oracle(s["model"], full_q, full_v)
    return dict(result, full_q=full_q, full_v=full_v)


def shape_pose(s, terms, shape):
    b = shape["body"]
    R = terms["rotation"][b]
    return terms["position"][b] + R @ shape["center"], R


def independent_contacts(data, states):
    terms = {pid: mechanics(s, *states[pid]) for pid, s in data["specs"].items()}
    wrenches = {pid: np.zeros((len(t["com_position"]), 6)) for pid, t in terms.items()}
    sites = []
    for a, ai, b, bi in data["pairs"]:
        sa, sb = data["specs"][a]["shapes"][ai], data["specs"][b]["shapes"][bi]
        assert sa["kind"] == "sphere" and sb["kind"] in ("sphere", "hull")
        center, _ = shape_pose(data["specs"][a], terms[a], sa)
        box, R = shape_pose(data["specs"][b], terms[b], sb)
        if sb["kind"] == "sphere":
            delta = box - center
            distance = np.linalg.norm(delta)
            assert distance > 0
            normal = delta / distance
            gap = distance - sa["radius"] - sb["radius"]
            point = (center + sa["radius"] * normal + box - sb["radius"] * normal) / 2
        else:
            local = R.T @ (center - box)
            closest = box + R @ np.clip(local, -sb["half"], sb["half"])
            delta = closest - center
            distance = np.linalg.norm(delta)
            assert distance > 0, "frozen oracle requires sphere center outside finite box"
            normal = delta / distance
            gap = distance - sa["radius"]
            point = (center + sa["radius"] * normal + closest) / 2
        velocities = []
        for pid, shape in ((a, sa), (b, sb)):
            t, body = terms[pid], shape["body"]
            velocities.append(
                t["com_velocity"][body]
                + np.cross(t["angular_velocity"][body], point - t["com_position"][body])
            )
        relative = velocities[0] - velocities[1]
        tangent = relative - (relative @ normal) * normal
        normal_load = data["stiffness"] * max(0.0, -gap)
        force = -normal_load * normal - data["friction"] * normal_load * tangent / max(
            data["slip_speed"], np.linalg.norm(tangent)
        )
        for pid, shape, f in ((a, sa, force), (b, sb, -force)):
            body = shape["body"]
            wrenches[pid][body, :3] += f
            wrenches[pid][body, 3:] += np.cross(point - terms[pid]["com_position"][body], f)
        sites.append(
            dict(
                ident=(a, ai, b, bi),
                gap=gap,
                normal=normal,
                point=point,
                first_force=force,
                tangent=tangent,
                first_velocity=velocities[0],
                second_velocity=velocities[1],
                normal_load=normal_load,
            )
        )
    return terms, wrenches, sites


def full_load(s, t, wrench):
    external = np.einsum("bij,bi->j", t["linear_jacobian"], s["force"])
    external += np.einsum("bij,bi->j", t["angular_jacobian"], s["torque"])
    start = 6 if s["family"].startswith("floating") else 0
    external[start:] += s["effort"]
    contact = np.einsum("bij,bi->j", t["linear_jacobian"], wrench[:, :3])
    contact += np.einsum("bij,bi->j", t["angular_jacobian"], wrench[:, 3:])
    return external, contact


def bounded_motor(s, M, free_rhs, q, v, drive):
    k = len(s["kp"])
    if not k:
        return np.zeros(0)
    start = 6 if s["family"].startswith("floating") else 0
    ids = np.arange(start, start + k)
    scalar_q = q[7:] if start else q
    beta = H * H * s["kp"] + H * s["kd"]
    free_a = np.linalg.solve(M, free_rhs)
    mobility = np.linalg.inv(M)[np.ix_(ids, ids)]
    rhs = drive[:, 2] + s["kp"] * (drive[:, 0] - scalar_q - H * v[ids])
    rhs += s["kd"] * (drive[:, 1] - v[ids]) - beta * free_a[ids]
    fixed, moving = np.flatnonzero(beta == 0), np.flatnonzero(beta > 0)
    tau = np.clip(rhs, -s["caps"], s["caps"])
    if not len(moving):
        return tau
    A = mobility[np.ix_(moving, moving)] + np.diag(1 / beta[moving])
    b = rhs[moving] / beta[moving] - mobility[np.ix_(moving, fixed)] @ tau[fixed]
    for signs in product((-1, 0, 1), repeat=len(moving)):
        x = np.array(signs, float) * s["caps"][moving]
        f, bound = np.flatnonzero(np.array(signs) == 0), np.flatnonzero(np.array(signs) != 0)
        if len(f):
            x[f] = np.linalg.solve(A[np.ix_(f, f)], b[f] - A[np.ix_(f, bound)] @ x[bound])
        if np.any(abs(x) > s["caps"][moving] + 1e-10):
            continue
        g = A @ x - b
        if any(
            (sign == 0 and abs(value) > 1e-9)
            or (sign < 0 and value < -1e-9)
            or (sign > 0 and value > 1e-9)
            for sign, value in zip(signs, g, strict=True)
        ):
            continue
        tau[moving] = x
        return tau
    raise AssertionError("independent PD oracle found no bounded active face")


def advance_one(s, q, v, t, wrench, drive):
    external, contact = full_load(s, t, wrench)
    S = s["S"]
    M, b = S.T @ t["mass"] @ S, S.T @ t["bias"]
    reduced_load = S.T @ (external + contact)
    motor = bounded_motor(s, M, reduced_load - b, q, v, drive)
    start = 6 if s["family"].startswith("floating") else 0
    actuator = np.zeros(len(t["bias"]))
    # Independent motor is attached to authored independent physical joints.
    actuator[start : start + len(motor)] = motor
    acceleration = np.linalg.solve(M, S.T @ (external + contact + actuator) - b)
    next_v = v + H * acceleration
    if start:
        next_q = reference("floating").drift(dict(root_com=s["root_com"]), q, next_v, H)
    else:
        next_q = q + H * next_v
    full_acceleration = S @ acceleration
    reaction = t["mass"] @ full_acceleration + t["bias"] - external - contact - actuator
    return (next_q, next_v), dict(
        external=external,
        contact=contact,
        motor=motor,
        acceleration=acceleration,
        full_acceleration=full_acceleration,
        reaction=reaction,
        reduced_mass=M,
    )


def drives(data, tick):
    result = {pid: s["drive"].copy() for pid, s in data["specs"].items()}
    if data["switch"] is not None and tick >= data["switch"]:
        for pid, s in data["specs"].items():
            if "side" in s:
                result[pid][1, 1] *= -1
    return result


def normalized_pd_residual(spec, entry, accepted, drive, effort):
    """Independent public PD law; raw diagnostic has units of effort."""
    floating = spec["family"].startswith("floating")
    q0 = np.asarray(entry.position)[7 if floating else 0 :]
    q1 = np.asarray(accepted.position)[7 if floating else 0 :]
    v1 = np.asarray(accepted.velocity)[6 if floating else 0 :]
    kp, kd, cap = (np.asarray(spec[k], np.float32).astype(float) for k in ("kp", "kd", "caps"))
    held = np.asarray(drive, np.float32).astype(float)
    tau = np.asarray(effort, float)
    assert q0.shape == q1.shape == v1.shape == kp.shape == tau.shape
    assert np.isfinite(tau).all() and np.all(abs(tau) <= cap)
    target, rate, ff = held.T
    before_rounding = ff + kp * (target - q0 - H * v1) + kd * (rate - v1)
    published = ff + kp * (target - q1) + kd * (rate - v1)
    first, second = (np.clip(x, -cap, cap) for x in (before_rounding, published))
    residual = np.maximum(abs(tau - first), abs(tau - second))
    scale = np.maximum(1.0, np.maximum(abs(tau), np.maximum(abs(first), abs(second))))
    return float(np.max(residual / scale, initial=0.0))


def audit_receipt(data, built, world, before, after, receipt, current_drives):
    actual_states = {
        key.participant: (np.asarray(p.position), np.asarray(p.velocity))
        for key, p in before.participants.items()
    }
    terms, wrench, sites = independent_contacts(data, actual_states)
    expected = {}
    assert receipt.entry_tick == before.tick and receipt.exit_tick == after.tick == before.tick + 1
    assert tuple(receipt.participants) == tuple(before.participants) == tuple(after.participants)
    assert not receipt.static_reactions
    for pid, s in data["specs"].items():
        key = ph.ResidentParticipantKey(world, pid)
        item = receipt.participants[key]
        q, v = actual_states[pid]
        (nq, nv), truth = advance_one(s, q, v, terms[pid], wrench[pid], current_drives[pid])
        expected[pid] = truth
        assert item.entry_state == before.participants[key]
        assert item.accepted_state == after.participants[key]
        assert item.contact_generalized.shape == (s["S"].shape[0],)
        close(item.entry_mass, terms[pid]["mass"], atol=2e-5)
        close(item.entry_bias, terms[pid]["bias"], atol=2e-5)
        close(item.entry_linear, terms[pid]["linear_jacobian"])
        close(item.entry_angular, terms[pid]["angular_jacobian"])
        close(item.entry_body_twist[:, :3], terms[pid]["com_velocity"])
        close(item.entry_body_twist[:, 3:], terms[pid]["angular_velocity"])
        accepted_terms = mechanics(
            s, np.asarray(item.accepted_state.position), np.asarray(item.accepted_state.velocity)
        )
        close(item.accepted_body_twist[:, :3], accepted_terms["com_velocity"])
        close(item.accepted_body_twist[:, 3:], accepted_terms["angular_velocity"])
        close(item.input_external, truth["external"])
        close(item.law_wrench_fp64, wrench[pid])
        close(item.uploaded_wrench_fp32, wrench[pid])
        close(item.contact_generalized, truth["contact"])
        close(after.participants[key].position, nq)
        close(after.participants[key].velocity, nv)
        d = item.diagnostic
        close(d.external, truth["external"] + truth["contact"])
        close(d.acceleration, truth["acceleration"], atol=2e-5)
        close(d.drive, truth["motor"], atol=5e-5)
        assert d.equation_residual <= 2e-5 and np.isfinite(d.control_residual)
        truth["control_residual_normalized"] = normalized_pd_residual(
            s, item.entry_state, item.accepted_state, current_drives[pid], d.drive
        )
        assert truth["control_residual_normalized"] <= 5e-5
        if s["family"] in ("affine", "floating_affine"):
            close(d.full_acceleration, truth["full_acceleration"], atol=2e-5)
            close(d.constraint_effort_full, truth["reaction"], atol=2e-5)
            close(s["S"].T @ d.constraint_effort_full, np.zeros(s["S"].shape[1]), atol=2e-5)
        for value in (
            item.entry_mass,
            item.law_wrench_fp64,
            item.uploaded_wrench_fp32,
            item.contact_generalized,
        ):
            assert not value.flags.writeable
    assert len(receipt.sites) == len(sites)
    by_pair = {site.pair: site for site in receipt.sites}
    for truth in sites:
        a, ai, b, bi = truth["ident"]
        first, second = built["keys"][world, a, ai], built["keys"][world, b, bi]
        actual = by_pair[tuple(sorted((first, second)))]
        sign = 1 if first < second else -1
        close(actual.first_force_n, sign * truth["first_force"])
        close(actual.second_force_n, -sign * truth["first_force"])
        close(actual.geometry.load_point, truth["point"])
        assert abs(actual.geometry.witness.gap_m - truth["gap"]) < 5e-6
    return dict(terms=terms, wrenches=wrench, sites=sites, dynamics=expected)


def run_window(data, built, mode, directory, *, steps=None):
    states = {pid: (s["q"].copy(), s["v"].copy()) for pid, s in data["specs"].items()}
    rows = []
    count = data["steps"] if steps is None else steps
    with ph.ResidentSceneBatch(built["participants"], mode=mode, cache_root=directory) as scene:
        scene.set_controls(built["controls"])
        with prepare(scene, built) as plan:
            initial_stats = scene.stats()
            failure = None
            before, selected = scene.snapshot(), drives(data, 0)
            try:
                for tick in range(count):
                    selected = drives(data, tick)
                    scene.set_controls(
                        {
                            key: command(
                                data["specs"][key.participant], owner, selected[key.participant]
                            )
                            for key, owner in built["controllers"].items()
                        }
                    )
                    before = scene.snapshot()
                    entry_terms, entry_wrench, entry_sites = independent_contacts(data, states)
                    next_states = {
                        pid: advance_one(
                            s, *states[pid], entry_terms[pid], entry_wrench[pid], selected[pid]
                        )[0]
                        for pid, s in data["specs"].items()
                    }
                    assert scene.step_contacts(H, plan) == dict.fromkeys(built["worlds"], True)
                    after = scene.snapshot()
                    audited = {}
                    for world in built["worlds"]:
                        audited[world] = audit_receipt(
                            data,
                            built,
                            world,
                            before[world],
                            after[world],
                            plan.last_accepted[world],
                            selected,
                        )
                        for pid in data["specs"]:
                            key = ph.ResidentParticipantKey(world, pid)
                            close(
                                after[world].participants[key].position,
                                next_states[pid][0],
                                atol=2e-4,
                            )
                            close(
                                after[world].participants[key].velocity,
                                next_states[pid][1],
                                atol=2e-4,
                            )
                    rows.append(
                        dict(
                            entry=before,
                            accepted=after,
                            receipts=plan.last_accepted,
                            status=plan.last_status,
                            controls=selected,
                            oracle_same_entry=audited,
                            oracle_continuous=dict(
                                entry=states, accepted=next_states, sites=entry_sites
                            ),
                            scene_stats=scene.stats(),
                            plan_stats=plan.stats(),
                        )
                    )
                    states = next_states
                final_stats = scene.stats()
                used = set(s["family"] for s in data["specs"].values())
                labels = dict(
                    fixed="tree",
                    floating="floating_tree",
                    affine="affine",
                    floating_affine="floating_affine",
                )
                for family in used:
                    name = labels[family]
                    for field in (
                        "entry_mechanics_stages",
                        "candidate_solve_stages",
                        "candidate_finalize_stages",
                    ):
                        assert (
                            final_stats["storage_stage_counts"][name][field]
                            - initial_stats["storage_stage_counts"][name][field]
                            == count
                        )
            except BaseException as error:
                failure = dict(type=type(error).__name__, message=str(error))
                raise
            finally:
                save(
                    directory,
                    "robot-window",
                    dict(
                        inputs=data,
                        world_order=built["worlds"],
                        initial_stats=initial_stats,
                        final_stats=scene.stats(),
                        rows=rows,
                        failure=failure,
                        retained=scene.snapshot(),
                        status=plan.last_status,
                        last_entry=before,
                        last_controls=selected,
                        last_receipts=plan.last_accepted,
                    ),
                )
    assert scene.stats()["device_owned_bytes"] == 0
    return rows


@pytest.mark.tyx
@pytest.mark.parametrize("friction", (0.0, 0.25))
def test_r02_arm_pushes_eccentric_free_box_40_steps(friction, tyx_mode, tmp_path):
    data = r02(friction)
    initial = {p: (s["q"], s["v"]) for p, s in data["specs"].items()}
    terms, wrench, sites = independent_contacts(data, initial)
    close(terms[7]["mass"][0, 1], -9 / 32, atol=1e-12, rtol=0)
    close(sites[0]["normal_load"], 2.0, atol=1e-12, rtol=0)
    if friction == 0:
        close(wrench[7][1, 3:], [0, 0, 3 / 16], atol=1e-12, rtol=0)
        close(wrench[29][0, 3:], [0, 0, -1 / 8], atol=1e-12, rtol=0)
    else:
        tangent_force = sites[0]["first_force"] + sites[0]["normal_load"] * sites[0]["normal"]
        assert tangent_force @ sites[0]["tangent"] < 0
        assert np.linalg.norm(tangent_force) > 1e-5
    run_window(data, build(data, tmp_path), tyx_mode, tmp_path)


@pytest.mark.tyx
@pytest.mark.parametrize("asymmetric", (False, True))
def test_r03_two_arms_free_tool_separate_and_reenter(asymmetric, tyx_mode, tmp_path):
    data = r03(asymmetric)
    rows = run_window(data, build(data, tmp_path), tyx_mode, tmp_path)
    for index in range(2):
        gaps = [row["oracle_continuous"]["sites"][index]["gap"] for row in rows]
        assert gaps[0] < 0
        assert max(gaps[32:48]) > 1 / 64
        assert min(gaps[96:]) < -1 / 1024
        live = [row["receipts"][WORLDS[0]].sites[index] for row in rows]
        assert any(np.linalg.norm(site.first_force_n) == 0 for site in live[32:48])
        assert any(np.linalg.norm(site.first_force_n) > 1 / 32 for site in live[96:])
    if asymmetric:
        assert any(
            abs(
                row["accepted"][WORLDS[0]]
                .participants[ph.ResidentParticipantKey(WORLDS[0], 29)]
                .velocity[5]
            )
            > 1e-6
            for row in rows
        )
    else:
        for row in rows:
            close(
                row["accepted"][WORLDS[0]]
                .participants[ph.ResidentParticipantKey(WORLDS[0], 29)]
                .position[:3],
                [0, 0, 0],
                atol=2e-4,
            )


@pytest.mark.tyx
@pytest.mark.parametrize("family", FAMILIES)
def test_r04_full_physical_projection_each_family(family, tyx_mode, tmp_path):
    data = r04(family)
    terms, wrench, _ = independent_contacts(
        data, {p: (s["q"], s["v"]) for p, s in data["specs"].items()}
    )
    s = data["specs"][7]
    _, contact = full_load(s, terms[7], wrench[7])
    assert np.linalg.norm(contact) > 1
    if family in ("affine", "floating_affine"):
        close((s["S"].T @ contact)[-1], 0.0, atol=1e-12, rtol=0)
    run_window(data, build(data, tmp_path), tyx_mode, tmp_path)


@pytest.mark.tyx
@pytest.mark.parametrize("count", (2, 4))
def test_r05_sparse_coincident_worlds_reverse_registration(count, tyx_mode, tmp_path):
    data = r02(0.25)
    singles = []
    for world in WORLDS[:count]:
        directory = tmp_path / f"single-{world}"
        singles.append(
            run_window(data, build(data, directory, worlds=(world,)), tyx_mode, directory, steps=3)
        )
    batch = run_window(
        data,
        build(data, tmp_path / "batch", worlds=WORLDS[:count], reverse=True),
        tyx_mode,
        tmp_path / "batch",
        steps=3,
    )
    for wi, world in enumerate(WORLDS[:count]):
        for tick in range(3):
            assert batch[tick]["accepted"][world] == singles[wi][tick]["accepted"][world]
            for key, item in batch[tick]["receipts"][world].participants.items():
                other = singles[wi][tick]["receipts"][world].participants[key]
                close(item.contact_generalized, other.contact_generalized, atol=0, rtol=0)


@pytest.mark.tyx
@pytest.mark.parametrize("family", ("affine", "floating_affine"))
def test_explicit_intra_affine_siblings_use_full_physical_parent_table(family, tyx_mode, tmp_path):
    data = r04(family)
    data["pairs"] += ((7, 0, 7, 1),)
    data["intra"] = True
    states = {pid: (s["q"], s["v"]) for pid, s in data["specs"].items()}
    _, _, sites = independent_contacts(data, states)
    close(sites[-1]["first_force"], [0, -8, 0], atol=1e-12, rtol=0)
    rows = run_window(data, build(data, tmp_path), tyx_mode, tmp_path, steps=3)
    assert len(rows[0]["receipts"][WORLDS[0]].sites) == 3


@pytest.mark.tyx
@pytest.mark.parametrize("family", ("affine", "floating_affine"))
def test_explicit_intra_affine_direct_parent_still_requires_exclusion(family, tyx_mode, tmp_path):
    data = r04(family)
    data["specs"][7]["model"]["parents"][1] = 0
    data["pairs"] += ((7, 0, 7, 1),)
    data["intra"] = True
    built = build(data, tmp_path)
    with ph.ResidentSceneBatch(built["participants"], mode=tyx_mode, cache_root=tmp_path) as scene:
        before = scene.snapshot()
        with pytest.raises(ValueError, match="direct parent-child"):
            prepare(scene, built)
        assert scene.snapshot() == before
        save(
            tmp_path,
            "intra-direct-parent-rejection",
            dict(inputs=data, before=before, after=scene.snapshot(), stats=scene.stats()),
        )
