"""Source-external ROB1d installed consumer; no test or private-runtime imports.

Copy this file outside every checkout. Supply the new installation/dependency
paths through the caller's environment, then run with -B and the three CLI
arguments documented in the accompanying report. Nothing here modifies sys.path.
The checks below were authored before the two public scenarios. No producer,
CPU integrator, numerical optimizer, or Newton backend is used as an oracle.
"""

import argparse
import base64
import gzip
import hashlib
import json
import struct
import sys
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from itertools import combinations, product
from pathlib import Path

import numpy as np

H = 1 / 1024
FAMILIES = ("fixed", "affine", "floating", "floating_affine")
STORAGES = dict(
    fixed="tree", affine="affine", floating="floating_tree", floating_affine="floating_affine"
)
REQUIRED_MODULES = (
    "phymeo.backends.resident_scene",
    "phymeo.backends._resident_participant_layout",
    "phymeo.backends._resident_participant_contacts",
    "phymeo.backends._resident_batch_transaction",
    "phymeo.backends._tyx_participant_transaction",
    "phymeo.backends.tyx_world_batch",
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def encode(value):
    """Full typed payload; process owner tokens are labelled, never serialized as state."""
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject, "object arrays are not evidence values")
        return dict(
            dtype=value.dtype.str,
            shape=list(value.shape),
            bytes_b64=base64.b64encode(value.tobytes(order="C")).decode(),
        )
    if isinstance(value, np.generic):
        return dict(dtype=value.dtype.str, scalar_b64=base64.b64encode(value.tobytes()).decode())
    if type(value) is object:
        return dict(process_owner_token=id(value))
    if is_dataclass(value):
        return dict(
            type=type(value).__module__ + "." + type(value).__name__,
            fields={f.name: encode(getattr(value, f.name)) for f in fields(value)},
        )
    if isinstance(value, Mapping):
        return dict(mapping_entries=[[encode(k), encode(v)] for k, v in value.items()])
    if isinstance(value, tuple):
        return dict(tuple=[encode(v) for v in value])
    if isinstance(value, list):
        return [encode(v) for v in value]
    if type(value) is float:
        return dict(float64_bits=struct.pack(">d", value).hex())
    if isinstance(value, bytes):
        return dict(bytes_b64=base64.b64encode(value).decode())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError("unsupported complete evidence value: " + type(value).__name__)


def exact(a, b, label):
    require(encode(a) == encode(b), label)


def close(a, b, *, atol=5e-6, rtol=5e-5):
    a, b = np.asarray(a, float), np.asarray(b, float)
    require(np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite comparison")
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def checkout_ancestor(path):
    return next((p for p in (path, *path.parents) if (p / ".git").exists()), None)


def venv_dependency(path):
    for p in (path, *path.parents):
        if p.name.lower() == "site-packages":
            return any((q / "pyvenv.cfg").is_file() for q in p.parents)
    return False


def reject_source_entry(path, install):
    if checkout_ancestor(path) is not None:
        require(
            path.is_relative_to(install) or venv_dependency(path),
            "source checkout entry in sys.path: " + str(path),
        )


def provenance(install, *, final=False):
    rows = {}
    for name, module in sorted(sys.modules.items()):
        if name == "phymeo" or name.startswith("phymeo."):
            source = getattr(module, "__file__", None)
            if source is not None:
                path = Path(source).resolve()
                require(path.is_relative_to(install), "Phymeo source leakage: " + str(path))
                rows[name] = str(path)
            for item in getattr(module, "__path__", ()):
                require(
                    Path(item).resolve().is_relative_to(install),
                    "Phymeo package path escapes installation",
                )
    if final:
        require(set(REQUIRED_MODULES).issubset(rows), "new installed modules were not exercised")
    return rows


def preflight(install, output):
    require(install.is_dir(), "expected installation root must exist")
    require(checkout_ancestor(Path.cwd().resolve()) is None, "cwd is inside a checkout")
    require(checkout_ancestor(Path(__file__).resolve()) is None, "copy consumer outside source")
    require(checkout_ancestor(output) is None, "output must be outside source")
    require(
        not any(n == "phymeo" or n.startswith("phymeo.") for n in sys.modules),
        "Phymeo was imported before the provenance guard",
    )
    rows = []
    for entry in sys.path:
        path = Path(entry or Path.cwd()).resolve()
        reject_source_entry(path, install)
        rows.append(str(path))
    # Pure path-guard negative; no checkout is imported or inserted in sys.path.
    fake = output / "guard-only-checkout"
    (fake / "src").mkdir(parents=True)
    (fake / ".git").write_text("guard fixture; not a Git repository\n", encoding="utf-8")
    rejected = None
    try:
        reject_source_entry(fake / "src", install)
    except AssertionError as error:
        rejected = str(error)
    require(rejected is not None, "checkout/src negative did not reject")
    return dict(
        cwd=str(Path.cwd().resolve()),
        script=str(Path(__file__).resolve()),
        sys_path=rows,
        source_path_negative=rejected,
        dependency_rule="Existing virtualenv site-packages may be checkout descendants.",
    )


def normalized_pd(spec, item):
    floating = spec["family"].startswith("floating")
    q0 = np.asarray(item.entry_state.position)[7 if floating else 0 :]
    q1 = np.asarray(item.accepted_state.position)[7 if floating else 0 :]
    v1 = np.asarray(item.accepted_state.velocity)[6 if floating else 0 :]
    kp, kd, cap = (np.asarray(spec[k], np.float32).astype(float) for k in ("kp", "kd", "caps"))
    held = np.asarray(spec["drive"], np.float32).astype(float)
    tau = np.asarray(item.diagnostic.drive)
    require(q0.shape == q1.shape == v1.shape == kp.shape == tau.shape, "PD dimensions")
    require(np.isfinite(tau).all() and np.all(abs(tau) <= cap), "actuator cap or finiteness")
    target, rate, ff = held.T
    first = np.clip(ff + kp * (target - q0 - H * v1) + kd * (rate - v1), -cap, cap)
    second = np.clip(ff + kp * (target - q1) + kd * (rate - v1), -cap, cap)
    residual = np.maximum(abs(tau - first), abs(tau - second))
    scale = np.maximum(1.0, np.maximum(abs(tau), np.maximum(abs(first), abs(second))))
    result = float(np.max(residual / scale, initial=0.0))
    require(result <= 5e-5, "normalized PD publication residual")
    return result


def check_receipt(built, world, before, after, receipt):
    require(
        receipt.entry_tick == before.tick and receipt.exit_tick == after.tick == before.tick + 1,
        "continuous common-entry publication",
    )
    require(
        tuple(receipt.participants) == tuple(before.participants) == tuple(after.participants),
        "complete participant membership",
    )
    require(not receipt.static_reactions, "no dynamic participant may become static")
    require(len(receipt.sites) == len(built["data"]["pairs"]), "all authored contact sites")
    wrench = {
        key: np.zeros_like(item.law_wrench_fp64) for key, item in receipt.participants.items()
    }
    site_checks = []
    for site in receipt.sites:
        require(site.pair[0].world == site.pair[1].world == world, "cross-world contact")
        close(site.first_force_n + site.second_force_n, np.zeros(3))
        point = np.asarray(site.geometry.load_point)
        n = np.asarray(site.geometry.witness.normal)
        gap = float(site.geometry.witness.gap_m)
        material = built["laws"][site.pair]
        relative = site.first_velocity_m_s - site.second_velocity_m_s
        tangent = relative - np.dot(relative, n) * n
        normal_load = material.stiffness * max(0.0, -gap) * site.geometry.weight
        expected = -normal_load * n - material.friction * normal_load * tangent / max(
            material.slip_speed_m_s, float(np.linalg.norm(tangent))
        )
        close(site.first_force_n, expected)
        require(normal_load > 0, "loaded installed smoke lost its real contact")
        for collision, force, com in (
            (site.pair[0], site.first_force_n, site.first_reference_point_m),
            (site.pair[1], site.second_force_n, site.second_reference_point_m),
        ):
            key, body = built["shape_owners"][collision]
            item = receipt.participants[key]
            close(com, item.entry_geometry.com_position_m[body])
            moment = np.cross(point - com, force)
            wrench[key][body, :3] += force
            wrench[key][body, 3:] += moment
        site_checks.append(
            dict(pair=site.pair, normal_load=normal_load, force_expected=expected, tangent=tangent)
        )
    checks = {}
    for key, item in receipt.participants.items():
        exact(item.entry_state, before.participants[key], "receipt entry binding")
        exact(item.accepted_state, after.participants[key], "receipt successor binding")
        spec = built["data"]["specs"][key.participant]
        n = len(spec["model"]["mass"])
        d = n + (6 if spec["family"].startswith("floating") else 0)
        require(
            item.entry_mass.shape == (d, d)
            and item.entry_bias.shape == item.contact_generalized.shape == (d,),
            "full physical dimensions, including dependent joints",
        )
        close(item.law_wrench_fp64, wrench[key])
        exact(
            item.uploaded_wrench_fp32,
            np.asarray(wrench[key], np.float32),
            "complete body wrench FP32 upload",
        )
        jtw = np.einsum("bij,bi->j", item.entry_linear, item.uploaded_wrench_fp32[:, :3])
        jtw += np.einsum("bij,bi->j", item.entry_angular, item.uploaded_wrench_fp32[:, 3:])
        close(item.contact_generalized, jtw, atol=2e-5)
        diag = item.diagnostic
        require(
            diag.equation_residual <= 2e-5 and np.isfinite(diag.control_residual),
            "actual mechanical/actuator diagnostics",
        )
        checks[key] = dict(
            wrench=wrench[key], full_contact_projection=jtw, normalized_pd=normalized_pd(spec, item)
        )
        if spec["family"] in ("affine", "floating_affine"):
            close(
                spec["S"].T @ diag.constraint_effort_full, np.zeros(spec["S"].shape[1]), atol=2e-5
            )
        for value in (
            item.entry_mass,
            item.entry_linear,
            item.entry_angular,
            item.law_wrench_fp64,
            item.uploaded_wrench_fp32,
        ):
            require(not value.flags.writeable, "published receipt is mutable")
    total_force = sum((w[:, :3].sum(axis=0) for w in wrench.values()), np.zeros(3))
    total_moment = np.zeros(3)
    for key, value in wrench.items():
        centers = receipt.participants[key].entry_geometry.com_position_m
        total_moment += (value[:, 3:] + np.cross(centers, value[:, :3])).sum(axis=0)
    close(total_force, np.zeros(3))
    close(total_moment, np.zeros(3))
    return dict(
        sites=site_checks,
        participants=checks,
        total_force=total_force,
        total_moment_about_world_origin=total_moment,
    )


def check_stages(before, after, families):
    used = {STORAGES[f] for f in families}
    for storage, old in before["storage_stage_counts"].items():
        for name in (
            "entry_mechanics_stages",
            "candidate_solve_stages",
            "candidate_finalize_stages",
        ):
            require(
                after["storage_stage_counts"][storage][name] - old[name] == int(storage in used),
                "storage advanced other than once per step",
            )


def mechanical_model(parents, kinds, offsets, axes, masses, centers):
    count = len(parents)
    return dict(
        parents=np.asarray(parents, int),
        kinds=tuple(kinds),
        offsets=np.asarray(offsets, float).reshape(-1, 3),
        orientations=np.tile([0.0, 0, 0, 1], (count, 1)),
        axes=np.asarray(axes, float).reshape(-1, 3),
        mass=np.asarray(masses, float),
        com=np.asarray(centers, float).reshape(-1, 3),
        inertia=np.full((count, 3), 1 / 8),
        gravity=np.zeros(3),
    )


def participant_spec(family, model, q, v, *, root_mass=4.0):
    floating = family.startswith("floating")
    n = len(model["mass"])
    affine = family in ("affine", "floating_affine")
    P = np.array([[1.0], [-1.0]]) if affine else np.eye(n)
    k = P.shape[1]
    S = np.zeros((6 + n, 6 + k)) if floating else P.copy()
    if floating:
        S[:6, :6], S[6:, 6:] = np.eye(6), P
    count = n + int(floating)
    return dict(
        family=family,
        model=model,
        q=np.asarray(q, float),
        v=np.asarray(v, float),
        root_mass=root_mass,
        root_com=np.zeros(3),
        root_inertia=np.full(3, 1 / 8),
        P=P,
        S=S,
        kp=np.full(k, 4.0),
        kd=np.full(k, 0.5),
        caps=np.full(k, 64.0),
        drive=np.zeros((k, 3)),
        effort=np.zeros(n),
        force=np.zeros((count, 3)),
        torque=np.zeros((count, 3)),
        shapes=[],
    )


def box_spec(position, *, angle=0.0, mass=1.0, eccentric=True):
    value = participant_spec(
        "floating",
        mechanical_model([], [], [], [], [], []),
        np.r_[position, [0, 0, np.sin(angle / 2), np.cos(angle / 2)]],
        [-1 / 64, 1 / 32, 0, 0, 0, 1 / 128],
        root_mass=mass,
    )
    value["shapes"] = [
        dict(
            body=0,
            kind="hull",
            half=np.array([1 / 8, 3 / 16, 1 / 8]),
            center=np.array([0, 1 / 16, 0]) if eccentric else np.zeros(3),
        )
    ]
    return value


def r02_declaration():
    model = mechanical_model(
        [-1, 0],
        ("revolute", "prismatic"),
        [[0, 0, 0], [0, 1 / 4, 0]],
        [[0, 0, 1], [1, 0, 0]],
        [2, 1],
        [[1 / 8, 0, 0], [1 / 16, 1 / 32, 0]],
    )
    arm = participant_spec("fixed", model, [1 / 16, 1 / 32], [1 / 32, -1 / 64])
    arm["drive"] = np.array(
        [[1 / 16 + 1 / 64, 1 / 64, 1 / 32], [1 / 32 + 1 / 64, -1 / 128, 1 / 16]]
    )
    arm["shapes"] = [dict(body=1, kind="sphere", center=np.array([1 / 8, 1 / 8, 0]), radius=1 / 8)]
    c, s = np.cos(1 / 16), np.sin(1 / 16)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    origin = R @ (np.array([1 / 32 + 1 / 8, 3 / 8, 0]) + [7 / 32, -1 / 16, 0])
    box = box_spec(origin, angle=1 / 16)
    box["force"][0] = [0, 1 / 32, 0]
    return dict(
        name="R02-installed",
        specs={7: arm, 29: box},
        pairs=((7, 0, 29, 0),),
        friction=0.0,
        rotation=R,
        steps=5,
    )


def mixed_declaration():
    specs, pairs = {}, []
    for pid, family in zip((7, 17, 27, 37), FAMILIES, strict=True):
        model = mechanical_model(
            [-1, -1],
            ("prismatic", "prismatic"),
            [[-7 / 32 - 1 / 32, -1 / 16, 0], [-7 / 32 + 1 / 32, 1 / 16, 0]],
            [[1, 0, 0], [1, 0, 0]],
            [1, 1],
            [[0, 0, 0], [0, 0, 0]],
        )
        affine, floating = family in ("affine", "floating_affine"), family.startswith("floating")
        q = np.array([1 / 32]) if affine else np.array([1 / 32, -1 / 32])
        v = np.array([1 / 64]) if affine else np.array([1 / 64, -1 / 64])
        value = participant_spec(
            family,
            model,
            np.r_[np.zeros(3), [0, 0, 0, 1], q] if floating else q,
            np.r_[np.zeros(6), v] if floating else v,
        )
        value["drive"] = np.column_stack((q + 1 / 128, v, np.full(len(q), 1 / 16)))
        value["shapes"] = [
            dict(body=i + int(floating), kind="sphere", center=np.zeros(3), radius=1 / 8)
            for i in range(2)
        ]
        specs[pid] = value
        pairs.extend(((pid, 0, 59, 0), (pid, 1, 59, 0)))
    specs[59] = box_spec([0, 0, 0], mass=4.0, eccentric=False)
    specs[59]["v"][:] = 0
    return dict(name="R04-R06-installed", specs=specs, pairs=tuple(pairs), friction=0.0)


def write_graph(spec, path):
    """Literal two-prismatic mimic asset, built through public loaders at runtime."""
    root = (
        '<link name="base"><inertial><mass value="4"/>'
        '<inertia ixx="0.125" iyy="0.125" izz="0.125" ixy="0" ixz="0" iyz="0"/>'
        "</inertial></link>"
    )
    links, joints = [], []
    for index in range(2):
        links.append(
            f'<link name="link{index}"><inertial><mass value="1"/>'
            '<inertia ixx="0.125" iyy="0.125" izz="0.125" ixy="0" ixz="0" iyz="0"/>'
            "</inertial></link>"
        )
        xyz = " ".join(str(float(x)) for x in spec["model"]["offsets"][index])
        mimic = '<mimic joint="joint0" multiplier="-1" offset="0"/>' if index else ""
        joints.append(
            f'<joint name="joint{index}" type="prismatic"><parent link="base"/>'
            f'<child link="link{index}"/><origin xyz="{xyz}"/><axis xyz="1 0 0"/>'
            '<limit lower="-2" upper="2" effort="64" velocity="10"/>' + mimic + "</joint>"
        )
    text = '<robot name="rob1d_graph">' + root + "".join(links + joints) + "</robot>"
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_text(text, encoding="utf-8")
    return dict(path=str(path), text=text, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def robot_model(ph, model):
    return ph.RobotModel(
        *(
            model[k]
            for k in (
                "parents",
                "kinds",
                "offsets",
                "orientations",
                "axes",
                "mass",
                "com",
                "inertia",
                "gravity",
            )
        )
    )


def build(ph, data, directory, worlds):
    participants, shapes, bindings, controls, laws = [], [], [], {}, {}
    shape_keys, shape_owners, assets = {}, {}, []
    for world in worlds:
        for pid, spec in data["specs"].items():
            family, model = spec["family"], spec["model"]
            floating = family.startswith("floating")
            if family in ("affine", "floating_affine"):
                path = directory / f"w{world}-p{pid}" / "graph.urdf"
                assets.append(write_graph(spec, path))
                loader = ph.load_floating_mimic_urdf if floating else ph.load_mimic_urdf
                asset = loader(path, gravity=(0, 0, 0))
                require(
                    tuple(asset.tree.joint_names) == ("joint0", "joint1"), "asset joint identity"
                )
                cls = ph.FloatingAffineRobotTree if floating else ph.AffineRobotTree
                state = (
                    ph.FloatingRobotState(spec["q"], spec["v"])
                    if floating
                    else ph.RobotState(spec["q"], spec["v"])
                )
                tree = cls(asset, state, kp=spec["kp"], kd=spec["kd"], effort_limit=spec["caps"])
                controller = tree
            elif floating:
                full = ph.FloatingRobotModel(
                    ph.RigidBodyModel(
                        np.array([1 / spec["root_mass"]]), spec["root_inertia"][None], np.zeros(3)
                    ),
                    spec["root_com"],
                    robot_model(ph, model) if len(model["mass"]) else None,
                )
                tree = ph.FloatingRobotTree(
                    full,
                    ph.FloatingRobotState(spec["q"], spec["v"]),
                    kp=spec["kp"],
                    kd=spec["kd"],
                    effort_limit=spec["caps"],
                )
                controller = tree
            else:
                controller = ph.DrivenRobotProgram(
                    robot_model(ph, model), kp=spec["kp"], kd=spec["kd"], effort_limit=spec["caps"]
                )
                tree = ph.FixedRobotTree(
                    controller,
                    controller.initial_state(ph.RobotState(spec["q"], spec["v"])),
                    control_max_iterations=512,
                )
            key = ph.ResidentParticipantKey(world, pid)
            body_count = len(model["mass"]) + int(floating)
            participant = ph.ResidentParticipant(
                key, tree, tuple(100 * pid + i for i in range(body_count))
            )
            participants.append(participant)
            drive = spec["drive"]
            if floating:
                control = controller.inputs(
                    drive, ph.FloatingRobotInputs(spec["effort"], spec["force"], spec["torque"])
                )
            else:
                control = controller.inputs(
                    drive[:, 0],
                    drive[:, 1],
                    feedforward=drive[:, 2],
                    loads=ph.RobotInputs(spec["effort"], spec["force"], spec["torque"]),
                )
            controls[key] = control
            cls = dict(
                fixed=ph.ResidentTreeBinding,
                affine=ph.ResidentAffineTreeBinding,
                floating=ph.ResidentFloatingTreeBinding,
                floating_affine=ph.ResidentFloatingAffineTreeBinding,
            )[family]
            for ordinal, shape in enumerate(spec["shapes"]):
                body = shape["body"]
                collision = ph.CollisionKey(world, participant.body_ids[body], 10 + ordinal)
                local = (
                    shape["center"][None]
                    if shape["kind"] == "sphere"
                    else np.array(list(product((-1, 1), repeat=3))) * shape["half"]
                    + shape["center"]
                )
                shapes.append(
                    ph.CollisionShape(
                        collision,
                        shape["kind"],
                        tuple(range(31, 31 + len(local))),
                        shape.get("radius", 0.0),
                        revision=1,
                    )
                )
                kwargs = dict(topology=participant.compiled.topology)
                if family == "floating_affine":
                    kwargs["graph_token"] = participant.graph_token
                bindings.append(
                    ph.ResidentParticipantBinding(key, cls(collision, world, body, local, **kwargs))
                )
                shape_keys[world, pid, ordinal] = collision
                shape_owners[collision] = key, body
        for left, li, right, ri in data["pairs"]:
            pair = tuple(sorted((shape_keys[world, left, li], shape_keys[world, right, ri])))
            laws[pair] = ph.RobotContactMaterial(64.0, 0.0, data["friction"], 1 / 16)
    excluded = tuple(
        (a.key, b.key)
        for a, b in combinations(shapes, 2)
        if a.key.world == b.key.world
        and a.key.body != b.key.body
        and tuple(sorted((a.key, b.key))) not in laws
    )
    return dict(
        data=data,
        participants=tuple(reversed(participants)),
        controls=controls,
        model=ph.CollisionModel(tuple(reversed(shapes)), excluded),
        bindings=tuple(reversed(bindings)),
        laws=laws,
        shape_owners=shape_owners,
        worlds=tuple(sorted(worlds)),
        assets=assets,
    )


class Journal:
    def __init__(self, path):
        self.path = path
        self.stream = gzip.open(path, "wb")
        self.digest = hashlib.sha256()
        self.logical_bytes = 0
        self.events = 0

    def emit(self, event, **values):
        raw = (
            json.dumps(
                dict(event=event, payload=encode(values)), allow_nan=False, separators=(",", ":")
            )
            + "\n"
        ).encode()
        self.stream.write(raw)
        self.stream.flush()
        self.digest.update(raw)
        self.logical_bytes += len(raw)
        self.events += 1

    def close(self):
        self.stream.close()
        compressed = hashlib.sha256()
        with self.path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                compressed.update(block)
        return dict(
            path=str(self.path),
            events=self.events,
            logical_bytes=self.logical_bytes,
            logical_sha256=self.digest.hexdigest(),
            compressed_bytes=self.path.stat().st_size,
            compressed_sha256=compressed.hexdigest(),
        )


def prepare(scene, built):
    return scene.prepare_contacts(
        built["model"],
        built["bindings"],
        built["laws"],
        candidate_capacity=len(built["laws"]),
        site_capacity=128,
        sampling="discrete_entry",
        narrow_phase="host_patch",
    )


def construction_evidence(built):
    return dict(
        declarations=built["data"],
        controls=built["controls"],
        collision_model=built["model"],
        bindings=built["bindings"],
        laws=built["laws"],
        assets=built["assets"],
        worlds=built["worlds"],
        full_body_counts={
            w: sum(len(p.body_ids) for p in built["participants"] if p.key.world == w)
            for w in built["worlds"]
        },
    )


def capture(scene, plan):
    return dict(
        states=scene.snapshot(),
        receipts=plan.last_accepted,
        status=plan.last_status,
        epoch=scene.state_epoch,
        bank=scene.stats()["current_buffer"],
        stats=scene.stats(),
    )


def retained(scene, plan, before):
    exact(scene.snapshot(), before["states"], "negative mutated physical checkpoint")
    require(
        plan.last_accepted is before["receipts"] and plan.last_status is before["status"],
        "negative mutated retained publication",
    )
    require(
        scene.state_epoch == before["epoch"] and scene.stats()["current_buffer"] == before["bank"],
        "negative committed a bank",
    )


def step(scene, plan, built, journal, label):
    before, stats = scene.snapshot(), scene.stats()
    journal.emit(
        "attempt",
        case=built["data"]["name"],
        label=label,
        h=H,
        controls=built["controls"],
        entry=before,
        stats=stats,
    )
    accepted = scene.step_contacts(H, plan)
    after = scene.snapshot()
    receipts = plan.last_accepted
    # Save the actual result before any checker can fail.
    journal.emit(
        "step",
        case=built["data"]["name"],
        label=label,
        accepted=accepted,
        entry=before,
        output=after,
        receipts=receipts,
        status=plan.last_status,
        scene_stats=scene.stats(),
        plan_stats=plan.stats(),
    )
    require(dict(accepted) == dict.fromkeys(built["worlds"], True), "physical world rejection")
    checks = {w: check_receipt(built, w, before[w], after[w], receipts[w]) for w in built["worlds"]}
    check_stages(stats, scene.stats(), {s["family"] for s in built["data"]["specs"].values()})
    journal.emit("checks", case=built["data"]["name"], label=label, checks=checks)
    return after, receipts


def physical_successor(receipts):
    # Owner, global state_epoch and bank belong to a transaction, not its physical
    # successor. Preserve the complete receipts in the journal; compare exactly
    # the full participant records/sites/frames of the restored same-owner step.
    return {
        w: dict(
            entry_tick=r.entry_tick,
            exit_tick=r.exit_tick,
            dt_s=r.dt_s,
            participants=r.participants,
            sites=r.sites,
            static_reactions=r.static_reactions,
            entry_frames=r.entry_frames,
            accepted_frames=r.accepted_frames,
            geometry_fingerprint=r.geometry_fingerprint,
            candidate_pairs=r.candidate_pairs,
        )
        for w, r in receipts.items()
    }


def close_scene(scene, journal, label):
    scene.close()
    stats = scene.stats()
    journal.emit("closed", label=label, stats=stats)
    require(stats["device_owned_bytes"] == 0, "scene retained device resources after close")
    return stats


def run_arm(ph, mode, directory, journal):
    directory.mkdir(parents=True, exist_ok=False)
    built = build(ph, r02_declaration(), directory / "assets", (503,))
    journal.emit("declaration", **construction_evidence(built))
    scene = ph.ResidentSceneBatch(built["participants"], mode=mode, cache_root=directory / "cache")
    try:
        scene.set_controls(built["controls"])
        with prepare(scene, built) as plan:
            initial = scene.snapshot()
            start_stats = scene.stats()
            require(
                any(np.any(p.velocity) for p in initial[503].participants.values()),
                "R02 requires incoming motion",
            )
            for tick in range(5):
                after, receipts = step(scene, plan, built, journal, "arm-" + str(tick))
                if tick == 0:
                    arm = receipts[503].participants[ph.ResidentParticipantKey(503, 7)]
                    box = receipts[503].participants[ph.ResidentParticipantKey(503, 29)]
                    close(arm.entry_mass[0, 1], -9 / 32)
                    close(arm.law_wrench_fp64[1, :3], built["data"]["rotation"] @ [-2.0, 0, 0])
                    close(arm.law_wrench_fp64[1, 3:], [0, 0, 3 / 16])
                    close(box.law_wrench_fp64[0, 3:], [0, 0, -1 / 8])
            require(after[503].tick == 5, "R02 measured window")
            require(
                any(
                    not np.array_equal(p.position, initial[503].participants[k].position)
                    for k, p in after[503].participants.items()
                ),
                "R02 did not move",
            )
            require(
                scene.stats()["accepted_world_steps"] - start_stats["accepted_world_steps"] == 5,
                "R02 accepted-world counter",
            )
            require(
                scene.stats()["rejected_world_steps"] == start_stats["rejected_world_steps"],
                "R02 unexpectedly rejected a world",
            )
            journal.emit(
                "case_pass",
                name=built["data"]["name"],
                physical_steps=5,
                final=after,
                receipts=receipts,
            )
    finally:
        closed = close_scene(scene, journal, "R02")
    return closed


def run_mixed(ph, mode, directory, journal):
    directory.mkdir(parents=True, exist_ok=False)
    built = build(ph, mixed_declaration(), directory / "assets", (503, 91))
    journal.emit("declaration", **construction_evidence(built))
    require(
        all(
            sum(len(p.body_ids) for p in built["participants"] if p.key.world == w) == 11
            for w in built["worlds"]
        ),
        "declared four-family physical body count",
    )
    scene = ph.ResidentSceneBatch(built["participants"], mode=mode, cache_root=directory / "cache")
    try:
        scene.set_controls(built["controls"])
        with prepare(scene, built) as plan:
            initial = scene.snapshot()
            start_stats = scene.stats()
            for tick in range(3):
                step(scene, plan, built, journal, "mixed-pre-" + str(tick))
            checkpoint = scene.snapshot()
            require(
                all(c.tick == 3 for c in checkpoint.values()), "checkpoint is not a real prefix"
            )
            journal.emit("checkpoint", checkpoint=checkpoint, state_epoch=scene.state_epoch)
            successors = []
            for tick in range(2):
                state, receipts = step(scene, plan, built, journal, "mixed-successor-" + str(tick))
                successors.append((state, physical_successor(receipts)))
            pre_restore = capture(scene, plan)
            journal.emit("restore_attempt", checkpoint=checkpoint, before=pre_restore)
            scene.restore(checkpoint, expected_epoch=scene.state_epoch)
            exact(scene.snapshot(), checkpoint, "restore did not recover actual held checkpoint")
            journal.emit("restored", after=capture(scene, plan))
            for tick in range(2):
                state, receipts = step(scene, plan, built, journal, "mixed-replayed-" + str(tick))
                exact(state, successors[tick][0], "restored successor state differs")
                exact(
                    physical_successor(receipts),
                    successors[tick][1],
                    "restored successor contact/force/motion receipt differs",
                )
            # Independent worlds occupy the same coordinates; compare physical
            # fields without conflating their deliberately distinct world identity.
            left, right = scene.snapshot().values()
            for lk, lp in left.participants.items():
                rk = ph.ResidentParticipantKey(right.world, lk.participant)
                rp = right.participants[rk]
                for field in ("position", "velocity", "applied_effort"):
                    exact(
                        getattr(lp, field), getattr(rp, field), "coincident worlds are not isolated"
                    )
            reset_world, keep_world = built["worlds"]
            before_reset = capture(scene, plan)
            scene.reset((reset_world,))
            after_reset = scene.snapshot()
            require(after_reset[reset_world].tick == 0, "selected world did not reset")
            exact(
                after_reset[keep_world],
                before_reset["states"][keep_world],
                "partial reset changed another world",
            )
            for key, point in after_reset[reset_world].participants.items():
                old = initial[reset_world].participants[key]
                exact(point.position, old.position, "reset position")
                exact(point.velocity, old.velocity, "reset velocity")
            require(
                plan.last_accepted[keep_world] is before_reset["receipts"][keep_world],
                "partial reset dropped the unaffected receipt",
            )
            journal.emit(
                "partial_reset",
                selected=reset_world,
                before=before_reset,
                after=capture(scene, plan),
            )
            scene.set_controls(
                {k: c for k, c in built["controls"].items() if k.world == reset_world}
            )
            state, _ = step(scene, plan, built, journal, "mixed-after-partial-reset")
            require(
                state[reset_world].tick == 1 and state[keep_world].tick == 6,
                "world clocks did not continue independently",
            )

            # New public scene has its own owner. No private token replacement,
            # bank access or fault injection is needed to reject a foreign capsule.
            foreign_built = build(ph, mixed_declaration(), directory / "foreign-assets", (503, 91))
            foreign = ph.ResidentSceneBatch(
                foreign_built["participants"], mode=mode, cache_root=directory / "foreign-cache"
            )
            try:
                foreign.set_controls(foreign_built["controls"])
                with prepare(foreign, foreign_built) as foreign_plan:
                    before = capture(foreign, foreign_plan)
                    error_text = None
                    try:
                        foreign.restore(checkpoint, expected_epoch=foreign.state_epoch)
                    except (ValueError, RuntimeError) as error:
                        error_text = type(error).__name__ + ": " + str(error)
                    require(error_text is not None, "foreign owner checkpoint was accepted")
                    retained(foreign, foreign_plan, before)
                    journal.emit(
                        "foreign_owner_rejected",
                        inputs=construction_evidence(foreign_built),
                        attempted=checkpoint,
                        error=error_text,
                        before=before,
                        after=capture(foreign, foreign_plan),
                    )
            finally:
                close_scene(foreign, journal, "foreign-owner")
            require(
                scene.stats()["accepted_world_steps"] - start_stats["accepted_world_steps"] == 16,
                "mixed accepted-world counter",
            )
            require(
                scene.stats()["rejected_world_steps"] == start_stats["rejected_world_steps"],
                "mixed unexpectedly rejected a world",
            )
            journal.emit(
                "case_pass",
                name=built["data"]["name"],
                physical_steps=8,
                final=scene.snapshot(),
                receipts=plan.last_accepted,
                retained_checkpoint=checkpoint,
            )
    finally:
        closed = close_scene(scene, journal, "four-family")
    return closed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--expected-install-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.output.is_absolute() and args.expected_install_root.is_absolute(),
        "installation and output paths must be absolute",
    )
    output, install = args.output.resolve(), args.expected_install_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output / "resident-participants.jsonl.gz")
    summary = dict(
        status="RUNNING",
        mode=args.mode,
        expected_install_root=str(install),
        consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=("R02-installed", "R04-R06-installed"),
        scope="HOST finite patch + actual TYX four-family mechanics; installed integration",
        excluded="R03 precision failures; CCD/device BVH/cross-tree implicit KKT",
    )
    try:
        summary["preflight"] = preflight(install, output)
        import phymeo as ph

        summary["initial_modules"] = provenance(install)
        journal.emit("start", summary=summary)
        arm_stats = run_arm(ph, args.mode, output / "arm", journal)
        mixed_stats = run_mixed(ph, args.mode, output / "mixed", journal)
        summary["final_modules"] = provenance(install, final=True)
        summary["status"] = "PASS"
        summary["main_scene_physical_calls"] = 13
        summary["accepted_logical_world_steps"] = (
            arm_stats["accepted_world_steps"] + mixed_stats["accepted_world_steps"]
        )
        require(summary["accepted_logical_world_steps"] == 21, "actual total world steps")
        summary["closed_scene_stats"] = dict(arm=arm_stats, mixed=mixed_stats)
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["exception"] = dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
        )
        journal.emit("failure", summary=summary)
        raise
    finally:
        journal.emit("summary", summary=summary)
        summary["trace"] = journal.close()
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
