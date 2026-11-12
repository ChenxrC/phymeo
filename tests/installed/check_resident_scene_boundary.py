"""Source-external scene boundary consumer; two sparse worlds, two real arms.

Uses only installed product APIs. Original materials, candidates, accepted source binding
and raw TYX packets are saved; physics advances the accepted bank. HOST geometry is
used only by the independent result oracle, never as device source. No source test /
example import occurs. Caller prepares the installed wheel and dependency path.
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
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import numpy as np

SCHEMA = "phymeo-installed-scene-boundary-v1"
DISTANCE_GATE = 1e-4
WITNESS_GATE = 2e-4
NORMAL_GATE = 2e-3


def require(condition, message):
    if not condition:
        raise AssertionError(message)


_TOKENS = {}


def encode(value):
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject, "object array in evidence")
        return dict(
            dtype=value.dtype.str,
            shape=list(value.shape),
            bytes_b64=base64.b64encode(value.tobytes()).decode(),
        )
    if isinstance(value, np.generic):
        return dict(dtype=value.dtype.str, scalar_b64=base64.b64encode(value.tobytes()).decode())
    if isinstance(value, bytes):
        return dict(bytes_b64=base64.b64encode(value).decode())
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
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if type(value) is object:
        _TOKENS.setdefault(id(value), len(_TOKENS))
        return dict(opaque_token=_TOKENS[id(value)])
    raise TypeError("unhandled evidence value " + type(value).__name__)


class Journal:
    def __init__(self, path):
        self.path = path
        self.stream = gzip.open(path, "xb")
        self.sha = hashlib.sha256()
        self.logical_bytes = self.events = 0

    def emit(self, event, **payload):
        record = dict(schema=SCHEMA, ordinal=self.events, event=event, payload=encode(payload))
        line = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
        self.stream.write(line)
        self.stream.flush()
        self.sha.update(line)
        self.logical_bytes += len(line)
        self.events += 1

    def close(self):
        self.stream.close()
        return dict(
            path=str(self.path),
            events=self.events,
            logical_bytes=self.logical_bytes,
            logical_sha256=self.sha.hexdigest(),
            compressed_bytes=self.path.stat().st_size,
            compressed_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
        )


def provenance(installed_root, *, preflight=False):
    def source_entry(path):
        return ((path / "pyproject.toml").is_file() and (path / "src" / "phymeo").is_dir()) or (
            path.name.lower() == "src"
            and (path / "phymeo").is_dir()
            and (path.parent / "pyproject.toml").is_file()
        )

    require(
        (installed_root / "phymeo" / "__init__.py").is_file(), "installed package root required"
    )
    paths = [Path(entry or Path.cwd()).resolve() for entry in sys.path]
    require(not any(source_entry(path) for path in paths), "checkout source entry in sys.path")
    if preflight:
        require(
            not any(name == "phymeo" or name.startswith("phymeo.") for name in sys.modules),
            "Phymeo imported before preflight",
        )
        for origin in (Path.cwd().resolve(), Path(__file__).resolve().parent, installed_root):
            require(
                not any(source_entry(parent) for parent in (origin, *origin.parents)),
                "consumer/cwd/install must be outside checkouts",
            )
    import phymeo

    require(
        Path(phymeo.__file__).resolve().is_relative_to(installed_root), "installed phymeo origin"
    )
    loaded = {}
    for name, module in sorted(sys.modules.items()):
        if not (name == "phymeo" or name.startswith("phymeo.")):
            continue
        filename = getattr(module, "__file__", None)
        if filename is not None:
            actual = Path(filename).resolve(strict=True)
            require(actual.is_relative_to(installed_root), "noninstalled product module " + name)
            loaded[name] = dict(
                path=str(actual), sha256=hashlib.sha256(actual.read_bytes()).hexdigest()
            )
        for directory in getattr(module, "__path__", ()):
            require(
                Path(directory).resolve(strict=True).is_relative_to(installed_root),
                "noninstalled package search path " + name,
            )
    return dict(
        installed_root=str(installed_root),
        cwd=str(Path.cwd().resolve()),
        consumer=str(Path(__file__).resolve()),
        sys_path=[str(path) for path in paths],
        loaded=loaded,
        numpy=dict(version=np.__version__, path=str(Path(np.__file__).resolve())),
    )


H = 1 / 1024
WORLDS = (17, 83)
MARGIN = 0.5
RADIUS = 1 / 8
VERTICES = np.array(((0, 0, 0), (0, 2, 0), (0, 0, 2), (-1, 0, 0)), np.float64)
FACES = ((0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2))
ROUND = np.array(((-0.25, 0.5, 0.5),), np.float64)


def rotation(q):
    x, y, z, w = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def asset_file(output, anchor, *, floating):
    output.mkdir()
    obj = "".join("v " + " ".join(str(float(x)) for x in row) + "\n" for row in VERTICES)
    obj += "".join("f " + " ".join(str(i + 1) for i in row) + "\n" for row in FACES)
    (output / "surface.obj").write_text(obj, encoding="utf-8")
    positions = [
        np.array((x, 0, 0)) if floating else anchor + np.array((x, 0, 0)) for x in (-0.25, 0.25)
    ]
    left, right = [" ".join(str(float(x)) for x in row) for row in positions]
    xml = f'''<robot name="scene_boundary_two_arms">
<link name="base"><inertial><mass value="2"/><inertia ixx=".2" iyy=".3" izz=".4" ixy="0" ixz="0" iyz="0"/></inertial></link>
<link name="left_jaw"><inertial><mass value="1"/><inertia ixx=".0625" iyy=".078125" izz=".09375" ixy="0" ixz="0" iyz="0"/></inertial><collision name="surface"><geometry><mesh filename="surface.obj"/></geometry></collision></link>
<joint name="driver" type="prismatic"><parent link="base"/><child link="left_jaw"/><origin xyz="{left}"/><axis xyz="1 0 0"/><limit lower="-.1" upper=".1" effort="1.5" velocity=".2"/></joint>
<link name="right_jaw"><inertial><mass value="2"/><inertia ixx=".09375" iyy=".109375" izz=".125" ixy="0" ixz="0" iyz="0"/></inertial><collision name="round"><origin xyz="-.25 .5 .5"/><geometry><sphere radius=".125"/></geometry></collision></link>
<joint name="follower" type="prismatic"><parent link="base"/><child link="right_jaw"/><origin xyz="{right}"/><axis xyz="1 0 0"/><limit lower="-.1" upper=".1" effort=".125" velocity=".08"/><mimic joint="driver" multiplier="-1" offset=".004"/></joint></robot>'''
    path = output / "source.urdf"
    path.write_text(xml, encoding="utf-8")
    return path, dict(xml=xml, obj=obj, anchor=anchor, floating=floating)


def make_scene(output):
    import phymeo as ph

    participants = []
    shapes = []
    bindings = []
    sources = {}
    originals = {}
    for i, world in enumerate(WORLDS):
        base = np.array((1 + 2 * i, -0.75, 0.125), np.float64)
        for pid, floating in ((7, False), (29, True)):
            key = ph.ResidentParticipantKey(world, pid)
            anchor = base + np.array((0.125 if floating else 0, 0, 0))
            path, description = asset_file(output / f"{world}-{pid}", anchor, floating=floating)
            if floating:
                asset = ph.load_floating_mimic_urdf(path, gravity=np.zeros(3))
                q = np.r_[anchor, (0, 0, 0, 1), 1 / 512]
                v = np.r_[(1 / 128, 0, 0, 0, 0, 0), 1 / 1024]
                tree = ph.FloatingAffineRobotTree(
                    asset,
                    ph.FloatingRobotState(q, v),
                    kp=np.zeros(1),
                    kd=np.zeros(1),
                    effort_limit=np.ones(1),
                    delay_steps=0,
                )
                full = asset.tree
                body_ids = (2900, 2901, 2902)
            else:
                asset = ph.load_mimic_urdf(path, gravity=np.zeros(3)).tree
                q = np.array((1 / 512, 0.004 - 1 / 512))
                v = np.array((1 / 1024, -1 / 1024))
                program = ph.DrivenRobotProgram(
                    asset.model,
                    kp=np.zeros(2),
                    kd=np.zeros(2),
                    effort_limit=np.ones(2),
                    delay_steps=0,
                )
                tree = ph.FixedRobotTree(program, program.initial_state(ph.RobotState(q, v)))
                full = asset
                body_ids = (700, 701)
            owner = ph.ResidentParticipant(key, tree, body_ids)
            participants.append(owner)
            description.update(
                q=q, v=v, original_model=full.model, original_colliders=full.collisions
            )
            sources[key] = description
            name = "round" if floating else "surface"
            original = next(c for c in full.collisions if c.name == name)
            require(
                original.shape.kind == ("sphere" if floating else "mesh"), "original asset kind"
            )
            local = ROUND if floating else VERTICES
            np.testing.assert_array_equal(original.local_vertices, local)
            shape_key = ph.CollisionKey(world, 2902 if floating else 700, 0)
            shape = replace(original.shape, key=shape_key)
            cls = ph.ResidentFloatingAffineTreeBinding if floating else ph.ResidentTreeBinding
            bound = cls.from_asset(asset, original.shape.key, key=shape_key, world=world)
            shapes.append(shape)
            bindings.append(ph.ResidentParticipantBinding(key, bound))
            originals[shape_key] = dict(
                owner=key, which=name, local=local, source_shape=original.shape
            )
            if floating:
                axis = np.vstack((ROUND, ROUND + (0, 0.25, 0)))
                shape_key = ph.CollisionKey(world, 2902, 1)
                shape = ph.CollisionShape(shape_key, "capsule", (0, 1), radius=RADIUS, revision=3)
                bound = ph.ResidentFloatingAffineTreeBinding(
                    shape_key,
                    world,
                    2,
                    axis,
                    topology=owner.compiled.physical.topology,
                    graph_token=owner.graph_token,
                )
                shapes.append(shape)
                bindings.append(ph.ResidentParticipantBinding(key, bound))
                originals[shape_key] = dict(owner=key, which="round", local=axis, source_shape=None)
    return (
        tuple(reversed(participants)),
        ph.CollisionModel(tuple(reversed(shapes))),
        tuple(reversed(bindings)),
        sources,
        originals,
    )


def actual_material(states, sources, originals):
    material = {}
    for key, original in originals.items():
        owner = original["owner"]
        declaration = sources[owner]
        point = states[key.world].participants[owner]
        if declaration["floating"]:
            q = np.asarray(point.full_position)
            frame = rotation(q[3:7])
            origin = q[:3]
            offset = np.array((0.25 + q[8], 0, 0))
        else:
            q = np.asarray(point.position)
            frame = np.eye(3)
            origin = declaration["anchor"]
            offset = np.array((-0.25 + q[0], 0, 0))
        material[key] = origin + (original["local"] + offset) @ frame.T
    return material


def point_triangle(point, tri):
    a, b, c = tri
    u = b - a
    v = c - a
    e = point - a
    uu, uv, vv = float(u @ u), float(u @ v), float(v @ v)
    det = uu * vv - uv * uv
    require(det > 0, "original finite nondegenerate facet")
    beta = (float(e @ u) * vv - float(e @ v) * uv) / det
    gamma = (float(e @ v) * uu - float(e @ u) * uv) / det
    candidates = []
    if beta >= 0 and gamma >= 0 and beta + gamma <= 1:
        candidates.append(a + beta * u + gamma * v)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        edge = tri[j] - tri[i]
        t = np.clip(float((point - tri[i]) @ edge) / float(edge @ edge), 0, 1)
        candidates.append(tri[i] + t * edge)
    return min(candidates, key=lambda p: float((p - point) @ (p - point)))


def segment_triangle(axis, tri):
    """Complete endpoint/edge/interior-crossing finite candidates; no product call."""
    a, b = axis
    u = b - a
    aa = float(u @ u)
    pairs = []
    for p in axis:
        pairs.append((p, point_triangle(p, tri)))
    for i, j in ((0, 1), (1, 2), (2, 0)):
        c, d = tri[i], tri[j]
        v = d - c
        bb = float(v @ v)
        uv = float(u @ v)
        w = a - c
        for p in (c, d):
            t = 0.0 if aa == 0 else float(np.clip(float((p - a) @ u) / aa, 0, 1))
            pairs.append((a + t * u, p))
        determinant = aa * bb - uv * uv
        if determinant > 0:
            uw, vw = float(u @ w), float(v @ w)
            t = (uv * vw - bb * uw) / determinant
            r = (aa * vw - uv * uw) / determinant
            if 0 <= t <= 1 and 0 <= r <= 1:
                pairs.append((a + t * u, c + r * v))
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    denominator = float(n @ u)
    if denominator != 0:
        t = float(n @ (tri[0] - a)) / denominator
        if 0 <= t <= 1:
            p = a + t * u
            q = point_triangle(p, tri)
            if np.linalg.norm(q - p) <= 1e-12:
                pairs.append((p, q))
    p, q = min(pairs, key=lambda x: float((x[1] - x[0]) @ (x[1] - x[0])))
    return float(np.linalg.norm(q - p)), p, q


def verify(scene, plan, packet, sources, originals):
    states = scene.snapshot()
    material = actual_material(states, sources, originals)
    require(packet.source_kind == "resident-scene-rigid-device-v1", "logical scene accepted source")
    require(packet.world_keys == WORLDS and packet.margin == MARGIN, "logical worlds and margin")
    require(plan.require_current(packet) is packet, "current public evidence")
    np.testing.assert_array_equal(packet.ticks, [states[w].tick for w in WORLDS])
    require(
        packet.source["participant_keys"] == scene.participant_keys, "all participant ownership"
    )
    require(
        packet.source["scene_registry_fingerprint"] == scene.layout.fingerprint,
        "full static registry",
    )
    require(len(packet.source["participant_addresses"]) == 4, "two participants per logical world")
    require(
        packet.source["stamps"]["native_poses"].shape == (6, 12), "actual native FK pose evidence"
    )
    require(packet.source["anchor_shifts_exact"].shape == (6, 3), "original anchor shifts")
    definitions = {item["key"]: item for item in packet.source["bindings"]}
    for key, original in originals.items():
        definition = definitions[key]
        require(
            definition["participant"] == original["owner"], "original logical participant owner"
        )
        if original["source_shape"] is not None:
            proof = definition["provenance"]
            require(
                definition["admission"] == "asset"
                and proof["source_shape"] == original["source_shape"],
                "original public asset identity",
            )
            np.testing.assert_array_equal(proof["local_points_m"], original["local"])
            require(
                proof["physical_topology"] == definition["topology"], "complete physical topology"
            )
    for key, points in material.items():
        origin = packet.source["world_anchors"][key.world]
        np.testing.assert_allclose(
            packet.source["gathered_points"][key], points - origin, atol=2e-4 * np.sqrt(8), rtol=0
        )
    require(len(packet.groups) == 2 and len(packet.pairs) == 6, "complete logical pair table")
    require(all(p.first.world == p.second.world for p in packet.pairs), "no cross-world candidates")
    checks = []
    total = 0
    for group in packet.groups:
        require(group.facet_vertices == FACES, "original four material facets")
        verts = material[group.surface]
        expected = []
        for si, key in enumerate(group.round_keys):
            raw = material[key]
            axis = raw if len(raw) == 2 else np.repeat(raw, 2, axis=0)
            lo = axis.min(axis=0) - RADIUS - MARGIN
            hi = axis.max(axis=0) + RADIUS + MARGIN
            for fi, face in enumerate(FACES):
                tri = verts[list(face)]
                if np.all(tri.min(axis=0) <= hi) and np.all(lo <= tri.max(axis=0)):
                    expected.append((si, fi))
        require(
            tuple(map(tuple, group.candidates.tolist())) == tuple(expected),
            "complete independent AABB candidates",
        )
        require(
            group.bvh_raw.shape == (2, 4) and np.all(group.bvh_raw[:, -1] == 7),
            "complete K+3 BVH traversal",
        )
        require(
            all(int(group.bvh_raw[si, 0]) == 15 for si in range(2)), "all original tetra facets"
        )
        narrow = group.packet
        require(narrow.source_binding == packet.source_binding, "one accepted input binding")
        np.testing.assert_array_equal(narrow.packed["owners"], group.candidates)
        require(len(narrow.rows) == len(expected), "no partial narrow publication")
        for row, (si, fi) in zip(narrow.rows, expected, strict=True):
            require(
                row.status == "RESOLVED" and (row.segment_id, row.facet_id) == (si, fi),
                "finite identity/status",
            )
            require(row.vertex_ids == FACES[fi], "original facet vertex identity")
            tri = verts[list(FACES[fi])]
            raw = material[group.round_keys[si]]
            axis = raw if len(raw) == 2 else np.repeat(raw, 2, axis=0)
            length = max(
                *(np.linalg.norm(tri[j] - tri[i]) for i, j in ((0, 1), (1, 2), (2, 0))),
                np.linalg.norm(axis[1] - axis[0]),
                RADIUS,
            )
            distance, p, q = segment_triangle(axis, tri)
            require(abs(row.distance - distance) <= 1e-4 * length, "independent finite distance")
            require(abs(row.gap - distance + RADIUS) <= 1e-4 * length, "radius gap law")
            bary = np.asarray(row.barycentric)
            t = row.axis_parameter
            require(
                min(bary) >= 0 and abs(sum(bary) - 1) <= 8 * 2**-23 and 0 <= t <= 1,
                "finite material parameters",
            )
            ap = axis[0] + t * (axis[1] - axis[0])
            tp = tri[0] + bary @ (tri - tri[0])
            np.testing.assert_allclose(
                np.asarray(row.axis_point_relative_m) + narrow.origin_m,
                ap,
                atol=2e-4 * length,
                rtol=0,
            )
            np.testing.assert_allclose(
                np.asarray(row.triangle_point_relative_m) + narrow.origin_m,
                tp,
                atol=2e-4 * length,
                rtol=0,
            )
            require(abs(np.linalg.norm(tp - ap) - distance) <= 2e-4 * length, "common witnesses")
            if distance > 32 * 2**-23 * length:
                require(
                    row.normal is not None
                    and np.linalg.norm(np.asarray(row.normal) - (q - p) / distance) <= 2e-3,
                    "independent normal",
                )
            else:
                require(row.normal is None, "undefined zero-distance normal")
            checks.append(
                dict(
                    surface=group.surface,
                    round=group.round_keys[si],
                    facet=fi,
                    distance=distance,
                    p=p,
                    q=q,
                    local_length=length,
                )
            )
        total += len(expected)
    require(total == 16 and packet.required_candidates == 16, "two full eight-row groups")
    require(
        plan.stats()["live_bytes"] == plan.stats()["persistent_bytes"],
        "temporary resources released",
    )
    return checks


def run(mode, journal, installed_root, output):
    import phymeo as ph

    scene = plan = None
    primary = None
    attempts = []
    steps = queries = stale = 0
    try:
        participants, model, bindings, sources, originals = make_scene(output)
        journal.emit(
            "fixture",
            h=H,
            margin=MARGIN,
            sources=sources,
            original_materials=originals,
            model=model,
            bindings=bindings,
            participants=tuple(
                dict(key=p.key, body_ids=p.body_ids, family=p.family, compiled=p.compiled)
                for p in participants
            ),
        )
        scene = ph.ResidentSceneBatch(participants, mode=mode, cache_root=output / "cache")
        plan = scene.prepare_boundary_query(
            model, bindings, limits=ph.BoundaryDistanceLimits(), memory_limit_bytes=32 * 1024 * 1024
        )
        last = None
        for index in range(3):
            before = scene.snapshot()
            attempt = dict(index=index, h=H, before=before)
            attempts.append(attempt)
            journal.emit("physical-attempt", **attempt)
            result = scene.step(H)
            attempt["result"] = result
            steps += sum(bool(v) for v in result.values())
            journal.emit("physical-return", index=index, result=result, accepted_worldsteps=steps)
            after = scene.snapshot()
            attempt["after"] = after
            journal.emit("physical-after", index=index, after=after)
            require(all(result.values()), "real physics accepted")
            for world in WORLDS:
                for key, point in after[world].participants.items():
                    require(
                        np.any(np.asarray(point.velocity) != 0),
                        "both participants retain real velocity",
                    )
                    require(
                        not np.array_equal(
                            point.position, before[world].participants[key].position
                        ),
                        "both participants actually move",
                    )
            if last is not None:
                try:
                    plan.require_current(last)
                except ValueError:
                    stale += 1
                else:
                    raise AssertionError("previous physical epoch remained current")
            journal.emit("query-attempt", index=index, before=after, stats=plan.stats())
            try:
                packet = plan.query(margin=MARGIN)
            finally:
                journal.emit(
                    "query-outcome",
                    index=index,
                    last_attempt=plan.last_attempt,
                    last_result=plan.last_result,
                    stats=plan.stats(),
                )
            queries += 1
            journal.emit("query-return", index=index, packet=packet)
            checks = verify(scene, plan, packet, sources, originals)
            require(encode(scene.snapshot()) == encode(after), "query changed accepted physics")
            journal.emit("oracle", index=index, checks=checks)
            last = packet
        before = scene.snapshot()
        scene.reset((17,))
        reset = scene.snapshot()
        require(
            encode(reset[83]) == encode(before[83]) and reset[17].tick == 0,
            "selected reset isolation",
        )
        try:
            plan.require_current(last)
        except ValueError:
            stale += 1
        else:
            raise AssertionError("reset kept old evidence current")
        scene.restore({17: before[17]}, expected_epoch=scene.state_epoch)
        journal.emit("reset-restore", before=before, reset=reset, restored=scene.snapshot())
        require(
            encode(scene.snapshot()) == encode(before), "public restore restored selected world"
        )
        # Source transition is checked by a fresh query; no fabricated motion or extra window.
        journal.emit("post-restore-query-attempt", before=scene.snapshot(), stats=plan.stats())
        try:
            packet = plan.query(margin=MARGIN)
        finally:
            journal.emit(
                "post-restore-query-outcome",
                last_attempt=plan.last_attempt,
                last_result=plan.last_result,
                stats=plan.stats(),
            )
        queries += 1
        journal.emit("post-restore-query", packet=packet, last_attempt=plan.last_attempt)
        checks = verify(scene, plan, packet, sources, originals)
        journal.emit("post-restore-oracle", checks=checks)
        return dict(
            status="PASS",
            logical_worldsteps=steps,
            participant_steps=2 * steps,
            queries=queries,
            rows=16 * queries,
            stale_rejections=stale,
            provenance=provenance(installed_root),
        )
    except BaseException as error:
        primary = error
        journal.emit(
            "failure",
            type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
            attempts=attempts,
            last_attempt=None if plan is None else plan.last_attempt,
        )
        raise
    finally:
        errors = []
        for label, owner in (("plan", plan), ("scene", scene)):
            if owner is not None:
                try:
                    owner.close()
                    journal.emit("closed", owner=label)
                except BaseException as error:
                    errors.append(error)
                    journal.emit(
                        "close-failure", owner=label, type=type(error).__name__, message=str(error)
                    )
        if errors and primary is None:
            raise BaseExceptionGroup("scene consumer cleanup", errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--installed-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.installed_root)
    output = Path(args.output)
    require(
        root.is_absolute() and output.is_absolute(), "absolute install and fresh output required"
    )
    root = root.resolve(strict=True)
    require(not output.exists(), "output must be fresh")
    output.mkdir(parents=True)
    journal = Journal(output / "complete.jsonl.gz")
    result = dict(status="FAILED", mode=args.mode)
    try:
        journal.emit("environment", mode=args.mode, provenance=provenance(root, preflight=True))
        result.update(run(args.mode, journal, root, output))
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result["evidence"] = journal.close()
        (output / "summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
