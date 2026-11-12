"""Source-external public compiled scene: affine fixed tool and free object.

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
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

SCHEMA = "phymeo-installed-participant-boundary-v1"
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


def asset_files(output, anchor):
    output.mkdir()
    obj = "".join("v " + " ".join(str(float(x)) for x in row) + "\n" for row in VERTICES)
    obj += "".join("f " + " ".join(str(i + 1) for i in row) + "\n" for row in FACES)
    (output / "surface.obj").write_text(obj, encoding="utf-8")
    left, right = [
        " ".join(str(float(x)) for x in anchor + (offset, 0, 0)) for offset in (-0.25, 0.25)
    ]
    tool_xml = f'''<robot name="public_affine_tool">
<link name="base"><inertial><mass value="2"/><inertia ixx=".2" iyy=".3" izz=".4" ixy="0" ixz="0" iyz="0"/></inertial></link>
<link name="left_jaw"><inertial><mass value="1"/><inertia ixx=".0625" iyy=".078125" izz=".09375" ixy="0" ixz="0" iyz="0"/></inertial></link>
<joint name="driver" type="prismatic"><parent link="base"/><child link="left_jaw"/><origin xyz="{left}"/><axis xyz="1 0 0"/><limit lower="-.1" upper=".1" effort="1.5" velocity=".2"/></joint>
<link name="tool"><collision name="surface"><geometry><mesh filename="surface.obj"/></geometry></collision></link>
<joint name="tool_fixed" type="fixed"><parent link="left_jaw"/><child link="tool"/><origin xyz="0 1.75 0" rpy="1.5707963267948966 0 0"/></joint>
<link name="right_jaw"><inertial><mass value="2"/><inertia ixx=".09375" iyy=".109375" izz=".125" ixy="0" ixz="0" iyz="0"/></inertial><collision name="round"><origin xyz="-.25 .5 .5"/><geometry><sphere radius=".125"/></geometry></collision></link>
<joint name="follower" type="prismatic"><parent link="base"/><child link="right_jaw"/><origin xyz="{right}"/><axis xyz="1 0 0"/><limit lower="-.1" upper=".1" effort=".125" velocity=".08"/><mimic joint="driver" multiplier="-1" offset=".004"/></joint></robot>'''
    free_xml = """<robot name="public_free_object"><link name="root">
<inertial><mass value="2"/><inertia ixx=".125" iyy=".125" izz=".125" ixy="0" ixz="0" iyz="0"/></inertial>
<collision name="free_round"><geometry><sphere radius=".125"/></geometry></collision>
</link></robot>"""
    tool_path, free_path = output / "tool.urdf", output / "free.urdf"
    tool_path.write_text(tool_xml, encoding="utf-8")
    free_path.write_text(free_xml, encoding="utf-8")
    return tool_path, free_path, dict(tool_xml=tool_xml, free_xml=free_xml, obj=obj)


TOOL_VERTICES = np.array(((0, 1.75, 0), (0, 1.75, 2), (0, -0.25, 0), (-1, 1.75, 0)))
TOOL_POSITION = np.array((0., 1.75, 0.))
TOOL_ROTATION = np.array(((1., 0., 0.), (0., 0., -1.), (0., 1., 0.)))
EFFORT = 1 / 64
FORCE = np.array((1 / 64, 0.0, 0.0))


def declaration(output):
    import phymeo as ph

    rows, sources, originals = [], {}, {}
    for index, world in enumerate(WORLDS):
        anchor = np.array((1 + 2 * index, -0.75, 0.125))
        tool_path, free_path, authored = asset_files(output / str(world), anchor)
        tool = ph.load_mimic_urdf(tool_path, gravity=np.zeros(3))
        free = ph.load_floating_urdf(free_path, gravity=np.zeros(3))
        tool_key = ph.ResidentParticipantKey(world, 7)
        free_key = ph.ResidentParticipantKey(world, 97)
        mapping = {}
        for collider in tool.tree.collisions:
            which = "driver" if collider.name == "surface" else "follower"
            target = ph.CollisionKey(world, 700 + int(which == "follower"), 0)
            mapping[collider.shape.key] = target
            # FoldedURDFAsset retains authored-link vertices. The device binding
            # separately applies the fixed site's pose to obtain group-local material.
            authored_local = VERTICES if which == "driver" else ROUND
            fixed_position = TOOL_POSITION if which == "driver" else np.zeros(3)
            fixed_rotation = TOOL_ROTATION if which == "driver" else np.eye(3)
            require(collider.link_name == ("tool" if which == "driver" else "right_jaw"),
                    "original authored link")
            np.testing.assert_allclose(collider.local_vertices, authored_local, atol=1e-14, rtol=0)
            np.testing.assert_allclose(collider.origin_xyz,
                                       np.zeros(3) if which == "driver" else ROUND[0],
                                       atol=1e-14, rtol=0)
            np.testing.assert_allclose(rotation(collider.origin_xyzw), np.eye(3),
                                       atol=1e-14, rtol=0)
            site = tool.tree.sites[collider.link_name]
            require(site.link_index == int(which == "follower"), "original moving-group owner")
            np.testing.assert_allclose(site.local_position_m, fixed_position, atol=1e-14, rtol=0)
            np.testing.assert_allclose(rotation(site.local_orientation_xyzw), fixed_rotation,
                                       atol=1e-14, rtol=0)
            # Oracle uses authored literals, never the product's transformed points.
            local = authored_local @ fixed_rotation.T + fixed_position
            np.testing.assert_allclose(local, TOOL_VERTICES if which == "driver" else ROUND,
                                       atol=1e-14, rtol=0)
            originals[target] = dict(owner=tool_key, which=which, local=local,
                                     authored_local=authored_local,
                                     fixed_position=fixed_position, fixed_rotation=fixed_rotation,
                                     source_shape=collider.shape)
        q, v = np.array((1 / 512,)), np.array((1 / 1024,))
        command = ph.RobotCommand(q, np.zeros(1), feedforward=np.array((EFFORT,)))
        rows.append(
            ph.ParticipantArticulation(
                tool_key,
                tool,
                ph.RobotState(q, v),
                kp=np.zeros(1),
                kd=np.zeros(1),
                effort_limit=np.array((1 / 16,)),
                body_ids=(700, 701),
                collider_keys=mapping,
                initial_control=command,
            )
        )
        sources[tool_key] = dict(kind="affine-tool", anchor=anchor, asset=tool, authored=authored)
        source = free.collisions[0]
        target = ph.CollisionKey(world, 9700, 0)
        position = anchor + (0.125, 0.5, 0.5)
        command = ph.FloatingRobotCommand(
            None, ph.FloatingRobotInputs(np.zeros(0), FORCE[None], np.zeros((1, 3)))
        )
        rows.append(
            ph.ParticipantArticulation(
                free_key,
                free,
                ph.FloatingRobotState(
                    np.r_[position, (0.0, 0.0, 0.0, 1.0)],
                    np.array((-1 / 64, 0.0, 0.0, 0.0, 0.0, 0.0)),
                ),
                kp=np.zeros(0),
                kd=np.zeros(0),
                effort_limit=np.zeros(0),
                body_ids=(9700,),
                collider_keys={source.shape.key: target},
                initial_control=command,
            )
        )
        sources[free_key] = dict(kind="free-object", anchor=position, asset=free, authored=authored)
        originals[target] = dict(
            owner=free_key, which="root", local=np.zeros((1, 3)), source_shape=source.shape
        )
    return (
        ph.ParticipantSceneDeclaration(tuple(reversed(rows)), contact_response="none"),
        sources,
        originals,
    )


def actual_material(states, sources, originals):
    material = {}
    for key, item in originals.items():
        state = states[key.world].participants[item["owner"]]
        source = sources[item["owner"]]
        if source["kind"] == "free-object":
            q = np.asarray(state.position)
            material[key] = q[:3] + item["local"] @ rotation(q[3:7]).T
        else:
            q = np.asarray(state.full_position)
            i = int(item["which"] == "follower")
            offset = (-0.25, 0.25)[i] + q[i]
            material[key] = source["anchor"] + item["local"] + (offset, 0, 0)
    return material


def verify_motion(before, after, receipts, sources):
    checks = []
    for world in WORLDS:
        require(after[world].tick == before[world].tick + 1, "one real accepted step")
        receipt = receipts[world]
        require(
            not receipt.sites and not receipt.candidate_pairs, "none mode creates no contact law"
        )
        for key, entry in before[world].participants.items():
            end = after[world].participants[key]
            q, v = np.asarray(entry.position), np.asarray(entry.velocity)
            if sources[key]["kind"] == "free-object":
                expected_v = v.copy()
                expected_v[:3] += H * FORCE / 2
                expected_q = q.copy()
                expected_q[:3] += H * expected_v[:3]
            else:
                expected_v = v + H * EFFORT / 3
                expected_q = q + H * expected_v
            np.testing.assert_allclose(end.velocity, expected_v, atol=1e-5, rtol=0)
            np.testing.assert_allclose(end.position, expected_q, atol=1e-5, rtol=0)
            require(
                np.any(np.asarray(end.position) != q), "actual motion, not static source replay"
            )
            actual = receipt.participants[key]
            np.testing.assert_array_equal(
                actual.law_wrench_fp64, np.zeros_like(actual.law_wrench_fp64)
            )
            np.testing.assert_array_equal(
                actual.uploaded_wrench_fp32, np.zeros_like(actual.uploaded_wrench_fp32)
            )
            checks.append(
                dict(
                    participant=key,
                    expected_position=expected_q,
                    expected_velocity=expected_v,
                    actual_receipt=actual,
                )
            )
    return checks


def verify_query(scene, plan, packet, sources, originals):
    states = scene.snapshot()
    material = actual_material(states, sources, originals)
    require(scene.last_boundary_result is packet, "compiled current plan's publication")
    require(scene.require_boundary_result(packet) is packet, "compiled current binding")
    require(plan.require_current(packet) is packet, "returned plan current binding")
    require(packet.source_kind == "resident-scene-rigid-device-v1", "accepted scene source")
    require(packet.world_keys == WORLDS and packet.state_epoch == scene.state_epoch, "world/epoch")
    np.testing.assert_array_equal(packet.ticks, [states[w].tick for w in WORLDS])
    require(packet.source["participant_keys"] == scene.participant_keys, "complete participants")
    require(
        packet.source["scene_registry_fingerprint"] == scene.layout.fingerprint, "full registry"
    )
    require(len(packet.source["participant_addresses"]) == 4, "two original participants per world")
    require(packet.source["stamps"]["native_poses"].shape == (6, 12), "actual native FK receipt")
    require(packet.source["anchor_shifts_exact"].shape == (6, 3), "full per-shape frame shifts")
    definitions = {row["key"]: row for row in packet.source["bindings"]}
    for key, original in originals.items():
        row = definitions[key]
        require(
            row["participant"] == original["owner"] and row["admission"] == "asset",
            "original declared installation",
        )
        proof = row["provenance"]
        require(proof["source_shape"] == original["source_shape"], "original shape identity")
        np.testing.assert_allclose(proof["local_points_m"], original["local"], atol=1e-14, rtol=0)
        require(proof["physical_topology"] == row["topology"], "full original topology")
        length = max(
            1.0, float(np.max(np.linalg.norm(original["local"] - original["local"][0], axis=1)))
        )
        np.testing.assert_allclose(
            packet.source["gathered_points"][key],
            material[key] - packet.source["world_anchors"][key.world],
            atol=2e-4 * length,
            rtol=0,
        )
    require(len(packet.groups) == 2 and len(packet.pairs) == 6, "complete logical geometry pairs")
    require(all(p.first.world == p.second.world for p in packet.pairs), "no world cross talk")
    checks, total = [], 0
    for group in packet.groups:
        original = originals[group.surface]["source_shape"]
        ids = original.vertex_ids
        faces = tuple(tuple(ids.index(v) for v in face) for face in original.triangles)
        require(group.facet_vertices == original.triangles, "complete original facet IDs")
        vertices = material[group.surface]
        candidates = []
        for si, key in enumerate(group.round_keys):
            axis = np.repeat(material[key], 2, axis=0)
            lo, hi = axis.min(axis=0) - RADIUS - MARGIN, axis.max(axis=0) + RADIUS + MARGIN
            for fi, face in enumerate(faces):
                tri = vertices[list(face)]
                if np.all(tri.min(axis=0) <= hi) and np.all(lo <= tri.max(axis=0)):
                    candidates.append((si, fi))
        require(
            tuple(map(tuple, group.candidates.tolist())) == tuple(candidates),
            "complete independent finite AABB coverage",
        )
        require(group.bvh_raw.shape == (len(group.round_keys), 4), "K+3 shape")
        require(np.all(group.bvh_raw[:, -1] == 7), "full four-facet traversal")
        bits = tuple(
            (si, fi)
            for si in range(len(group.round_keys))
            for fi in range(4)
            if int(group.bvh_raw[si, 0]) & (1 << fi)
        )
        require(bits == tuple(candidates), "raw candidate membership")
        narrow = group.packet
        require(narrow.source_binding == packet.source_binding, "same accepted source")
        require(len(narrow.rows) == len(candidates), "complete narrow rows")
        require(narrow.raw["candidates"].shape == (len(candidates), 18, 11), "raw candidates")
        require(narrow.raw["summary"].shape == (len(candidates), 16), "raw summary")
        for row, (si, fi) in zip(narrow.rows, candidates, strict=True):
            require(
                row.status == "RESOLVED" and (row.segment_id, row.facet_id) == (si, fi),
                "resolved finite material row",
            )
            require(row.vertex_ids == group.facet_vertices[fi], "full facet material address")
            tri = vertices[list(faces[fi])]
            axis = np.repeat(material[group.round_keys[si]], 2, axis=0)
            length = max(
                *(np.linalg.norm(tri[j] - tri[i]) for i, j in ((0, 1), (1, 2), (2, 0))),
                np.linalg.norm(axis[1] - axis[0]),
                RADIUS,
            )
            distance, p, q = segment_triangle(axis, tri)
            require(abs(row.distance - distance) <= DISTANCE_GATE * length, "finite distance")
            require(abs(row.gap - distance + RADIUS) <= DISTANCE_GATE * length, "radius gap")
            bary = np.asarray(row.barycentric)
            require(min(bary) >= 0 and abs(sum(bary) - 1) <= 8 * 2**-23, "material barycentrics")
            require(0 <= row.axis_parameter <= 1, "finite axis")
            ap = axis[0] + row.axis_parameter * (axis[1] - axis[0])
            tp = tri[0] + bary @ (tri - tri[0])
            np.testing.assert_allclose(
                np.asarray(row.axis_point_relative_m) + narrow.origin_m,
                ap,
                atol=WITNESS_GATE * length,
                rtol=0,
            )
            np.testing.assert_allclose(
                np.asarray(row.triangle_point_relative_m) + narrow.origin_m,
                tp,
                atol=WITNESS_GATE * length,
                rtol=0,
            )
            require(
                abs(np.linalg.norm(tp - ap) - distance) <= WITNESS_GATE * length, "both witnesses"
            )
            if distance > 32 * 2**-23 * length:
                require(
                    row.normal is not None
                    and np.linalg.norm(np.asarray(row.normal) - (q - p) / distance) <= NORMAL_GATE,
                    "normal direction",
                )
            else:
                require(row.normal is None, "no invented zero-distance normal")
            checks.append(
                dict(
                    surface=group.surface,
                    round=group.round_keys[si],
                    facet=fi,
                    distance=distance,
                    axis=p,
                    triangle=q,
                    local_length=length,
                )
            )
        total += len(candidates)
    require(total > 0 and packet.required_candidates == total, "complete nonempty aggregate")
    require(plan.stats()["live_bytes"] == plan.stats()["persistent_bytes"], "scratch released")
    return checks


def run(mode, journal, installed_root, output):
    import phymeo as ph

    declaration_value, sources, originals = declaration(output)
    journal.emit("declaration", declaration=declaration_value, sources=sources, originals=originals)
    scene = plan = None
    primary = None
    steps = queries = rows = stale = 0
    attempts = []

    def take_query(label):
        nonlocal queries, rows
        before = scene.snapshot()
        journal.emit("query-attempt", label=label, before=before, stats=plan.stats())
        try:
            packet = plan.query(margin=MARGIN)
        finally:
            journal.emit(
                "query-outcome",
                label=label,
                last_attempt=plan.last_attempt,
                last_result=plan.last_result,
                stats=plan.stats(),
            )
        queries += 1
        journal.emit("query-return", label=label, packet=packet)
        checks = verify_query(scene, plan, packet, sources, originals)
        rows += len(checks)
        require(encode(scene.snapshot()) == encode(before), "query cannot advance physics")
        journal.emit("query-oracle", label=label, checks=checks)
        return packet

    def take_step(label):
        nonlocal steps
        before = scene.snapshot()
        attempt = dict(label=label, before=before, h=H)
        attempts.append(attempt)
        journal.emit("physical-attempt", **attempt)
        result = scene.step(H)
        attempt["result"] = result
        steps += sum(bool(v) for v in result.values())
        journal.emit(
            "physical-return",
            label=label,
            result=result,
            logical_worldsteps=steps,
            receipts=scene.last_accepted,
            status=scene.last_status,
        )
        after = scene.snapshot()
        attempt["after"] = after
        journal.emit("physical-after", label=label, after=after)
        require(all(result.values()), "actual physical step accepted")
        checks = verify_motion(before, after, scene.last_accepted, sources)
        journal.emit("physical-oracle", label=label, checks=checks)
        return after

    def reject_stale(packet, label):
        nonlocal stale
        try:
            scene.require_boundary_result(packet)
        except ValueError as error:
            stale += 1
            journal.emit("stale-rejection", label=label, reason=str(error))
        else:
            raise AssertionError("stale compiled evidence accepted")

    try:
        scene = ph.compile_participant_scene(
            declaration_value, mode=mode, cache_root=output / "cache"
        )
        require(scene.declaration.contact_response == "none", "explicit no-response mode")
        require(scene.last_boundary_result is None, "no hidden query")
        journal.emit(
            "compiled",
            identity=scene.identity,
            source_map=scene.source_map,
            initial=scene.snapshot(),
            stats=scene.stats(),
            provenance=provenance(installed_root),
        )
        plan = scene.prepare_boundary_query(
            limits=ph.BoundaryDistanceLimits(), memory_limit_bytes=64 * 1024 * 1024
        )
        last = None
        for index in range(3):
            take_step(str(index))
            if last is not None:
                reject_stale(last, "next-step")
            last = take_query(str(index))
        before = scene.snapshot()
        journal.emit("reset-attempt", worlds=(17,), before=before)
        scene.reset((17,))
        reset = scene.snapshot()
        journal.emit("reset-return", after=reset)
        require(reset[17].tick == 0 and encode(reset[83]) == encode(before[83]), "selected reset")
        reject_stale(last, "selected-reset")
        journal.emit("restore-attempt", selected={17: before[17]}, expected_epoch=scene.state_epoch)
        scene.restore({17: before[17]}, expected_epoch=scene.state_epoch)
        restored = scene.snapshot()
        journal.emit("restore-return", after=restored)
        require(encode(restored) == encode(before), "public selected restore")
        take_step("post-restore")
        last = take_query("post-restore")
        physics = scene.snapshot()
        retained = plan
        small = scene.prepare_boundary_query(
            limits=ph.BoundaryDistanceLimits(candidates=0), memory_limit_bytes=64 * 1024 * 1024
        )
        journal.emit(
            "capacity-attempt", before=physics, limits=ph.BoundaryDistanceLimits(candidates=0)
        )
        try:
            small.query(margin=MARGIN)
        except ph.BoundaryDistanceCapacityError as error:
            journal.emit("capacity-rejection", type=type(error).__name__, message=str(error))
        else:
            raise AssertionError("zero candidate capacity was silently truncated")
        finally:
            journal.emit(
                "capacity-outcome",
                last_attempt=small.last_attempt,
                last_result=small.last_result,
                stats=small.stats(),
            )
        require(small.last_result is None, "failed query published no packet")
        require(
            encode(scene.snapshot()) == encode(physics), "failed independent query rolled physics"
        )
        require(
            retained.last_result is last and retained.require_current(last) is last,
            "previous plan still owns its complete packet",
        )
        plan = scene.prepare_boundary_query(
            limits=ph.BoundaryDistanceLimits(), memory_limit_bytes=64 * 1024 * 1024
        )
        take_query("retry")
        journal.emit(
            "final",
            snapshot=scene.snapshot(),
            scene_stats=scene.stats(),
            current_stats=plan.stats(),
        )
        return dict(
            status="PASS",
            mode=mode,
            logical_worldsteps=steps,
            participant_steps=2 * steps,
            queries=queries,
            rows=rows,
            stale_rejections=stale,
            capacity_rejections=1,
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
        if scene is not None:
            try:
                scene.close()
                journal.emit(
                    "closed",
                    scene_stats=scene.stats(),
                    current_plan=None if plan is None else plan.stats(),
                )
            except BaseException as error:
                journal.emit("close-failure", type=type(error).__name__, message=str(error))
                if primary is None:
                    raise


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
