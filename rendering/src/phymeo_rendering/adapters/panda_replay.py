"""Read-only replay of the named nine-coordinate Panda archive format.

The URDF supplies authored link frames, never principal-inertia frames or visual
mesh offsets. Both measured finger coordinates are authoritative: the URDF mimic
is a command convention and does not overwrite either archived physical joint.
No simulator, numerical package, graphics runtime, or physics step is imported.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from ..scene import FrameKey, NodePose, Pose, SceneFrame, VisualScene, compose_pose, validate_scene_frame

MODEL_SHA256 = "2b5bd00abb4ba68e91d532d2d778da64d1c942a9b22aa7acd58dd12a713971fd"
JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8)) + (
    "panda_finger_joint1", "panda_finger_joint2",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _json(text):
    return json.loads(text, object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _vector(values, size, label):
    _require(isinstance(values, (list, tuple)) and len(values) == size, f"invalid {label} shape")
    _require(all(type(x) in (int, float) and math.isfinite(x) for x in values), f"nonfinite {label}")
    return tuple(values)


def _rpy(text):
    r, p, y = (float(x)/2 for x in text.split())
    cr, cp, cy = math.cos(r), math.cos(p), math.cos(y)
    sr, sp, sy = math.sin(r), math.sin(p), math.sin(y)
    return (sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
            cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy)


def _chain(raw):
    root = ET.fromstring(raw)
    links = tuple(link.attrib["name"] for link in root.findall("link"))
    _require(root.tag == "robot" and root.get("name") == "panda", "foreign robot model")
    joints = []
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        xyz = "0 0 0" if origin is None else origin.get("xyz", "0 0 0")
        rpy = "0 0 0" if origin is None else origin.get("rpy", "0 0 0")
        axis = joint.find("axis")
        joints.append((joint.attrib["name"], joint.attrib["type"],
                       joint.find("parent").attrib["link"], joint.find("child").attrib["link"],
                       Pose(tuple(float(x) for x in xyz.split()), _rpy(rpy)),
                       tuple(float(x) for x in ("1 0 0" if axis is None else axis.attrib["xyz"]).split())))
    movable = tuple(j[0] for j in joints if j[1] != "fixed")
    _require(movable == JOINT_NAMES, "URDF joint order differs from named Panda archive")
    return links, tuple(joints)


def _fk(links, joints, q):
    coordinates = dict(zip(JOINT_NAMES, _vector(q, 9, "q"), strict=True))
    poses = {"panda_link0": Pose()}
    remaining = list(joints)
    while remaining:
        next_joints = []
        for name, kind, parent, child, origin, axis in remaining:
            if parent not in poses:
                next_joints.append((name, kind, parent, child, origin, axis))
                continue
            local = origin
            if kind != "fixed":
                value = coordinates[name]
                norm = math.hypot(*axis)
                _require(norm > 0, "zero joint axis")
                unit = tuple(x/norm for x in axis)
                if kind == "revolute":
                    movement = Pose(orientation_xyzw=tuple(x*math.sin(value/2) for x in unit)
                                    + (math.cos(value/2),))
                elif kind == "prismatic":
                    movement = Pose(tuple(x*value for x in unit))
                else:
                    raise ValueError(f"unsupported joint type {kind!r}")
                local = compose_pose(origin, movement)
            _require(child not in poses, "duplicate URDF child")
            poses[child] = compose_pose(poses[parent], local)
        _require(len(next_joints) < len(remaining), "disconnected or cyclic URDF")
        remaining = next_joints
    _require(set(poses) == set(links), "URDF root or link membership mismatch")
    return tuple(NodePose(name, poses[name]) for name in links)


class PandaReplaySource:
    """Bind a static visual manifest to a hash-checked finite archive excerpt.

    ``excerpt_path`` has a sibling ``manifest.json`` containing original archive,
    model, excerpt and raw-record SHA256s. The caller owns the trusted manifest;
    a checksum is integrity evidence, not an authenticated signature. A complete
    archive is not required at runtime. ``provenance(frame)`` accepts only a full
    frame emitted by this source and returns an independent JSON-ready copy.

    FrameKey world/epoch/revision are an explicit *replay* namespace. Revision is
    source record index, tick is source step index + 1, and time is copied from
    the record (also checked against recorded dt). Live physics identities were
    not saved and are reported as None. Only q/cube/xyzw/t are endpoint values;
    recorded forces and pad witnesses belong to the step's input geometry.
    """

    def __init__(self, scene: VisualScene, urdf_path, excerpt_path):
        if not isinstance(scene, VisualScene):
            raise TypeError("scene must be VisualScene")
        excerpt_path, urdf_path = Path(excerpt_path), Path(urdf_path)
        raw = excerpt_path.read_bytes()
        manifest = _json(excerpt_path.with_name("manifest.json").read_text(encoding="utf-8"))
        _require(manifest["schema"] == "phymeo-panda-archive-manifest-v1", "unknown manifest schema")
        _require(_digest(raw) == manifest["excerpt_sha256"], "excerpt SHA256 mismatch")
        model = urdf_path.read_bytes()
        _require(_digest(model) == manifest["model_sha256"] == MODEL_SHA256, "URDF model SHA256 mismatch")
        text = raw.decode("utf-8")
        data = _json(text)
        _require(data["schema"] == "phymeo-panda-archive-excerpt-v1", "unknown excerpt schema")
        inputs = data["inputs"]
        _require(inputs["schema"] == "panda-box-control-v1", "unknown archive input schema")
        _require(tuple(inputs["joint_names"]) == JOINT_NAMES, "archive joint order mismatch")
        _require(inputs["urdf_sha256"] == MODEL_SHA256, "archive model SHA256 mismatch")
        archive_sha = manifest["archive_sha256"]
        _require(isinstance(archive_sha, str) and re.fullmatch("[0-9a-f]{64}", archive_sha), "invalid archive digest")
        _require(manifest["live_physics_identity"] is None, "archive cannot claim live physics identity")
        _require(manifest["base_pose"] == dict(position_m=[0, 0, 0], orientation_xyzw=[0, 0, 0, 1]),
                 "archive model requires its original fixed base")
        dt = inputs["dt_s"]
        _require(type(dt) in (int, float) and math.isfinite(dt) and dt > 0, "invalid archive dt")
        count = manifest["archive_record_count"]
        _require(type(count) is int and count == inputs["steps"] and count > 0, "archive count mismatch")
        records = data["records"]
        _require(isinstance(records, list) and records, "excerpt needs records")
        _require(len(records) == len(manifest["records"]), "record manifest length mismatch")
        # raw_decode identifies the original record token boundary without
        # reserialization changing numeric lexemes, whitespace, or signed zero.
        decoder = json.JSONDecoder(object_pairs_hook=_pairs)
        raw_records = []
        for match in re.finditer(r'"record"\s*:\s*', text):
            _, end = decoder.raw_decode(text, match.end())
            raw_records.append(text[match.end():end].encode("utf-8"))
        _require(len(raw_records) == len(records), "raw record token membership mismatch")
        self._links, self._joints = _chain(model)
        self._scene = deepcopy(scene)
        self._records = {}
        frames = []
        last = -1
        for item, witness, raw_record in zip(records, manifest["records"], raw_records, strict=True):
            index, record = item["source_index"], item["record"]
            _require(type(index) is int and last < index < count, "invalid or unordered record index")
            _require(index == witness["source_index"], "record source index mismatch")
            _require(_digest(raw_record) == witness["raw_record_sha256"], "raw record SHA256 mismatch")
            _vector(record["q"], 9, "q")
            _vector(record["q_before"], 9, "q_before")
            _vector(record["v"], 9, "v")
            _require(type(record["t"]) in (int, float) and record["t"] == (index+1)*dt,
                     "record time differs from archive input dt/index")
            cube = Pose(_vector(record["cube"], 3, "cube"), _vector(record["xyzw"], 4, "cube rotation"))
            table = Pose(_vector(inputs["table_center_m"], 3, "table center"))
            nodes = _fk(self._links, self._joints, record["q"]) + (NodePose("cube", cube), NodePose("table", table))
            key = FrameKey(scene.key, f"archive:{archive_sha}", 0, index, index+1, "accepted_endpoint")
            frame = SceneFrame(key, record["t"], nodes)
            validate_scene_frame(scene, frame)
            frames.append(frame)
            self._records[index] = deepcopy(record)
            last = index
        for name in ("full_size_m", "table_full_size_m"):
            _require(all(x > 0 for x in _vector(inputs[name], 3, name)), "invalid shape dimensions")
        self._frames = tuple(frames)
        self._by_key = {frame.key: frame for frame in frames}
        self._manifest = deepcopy(manifest)
        self._record_digests = {row["source_index"]: row["raw_record_sha256"] for row in manifest["records"]}
        self._geometry = dict(cube_full_size_m=inputs["full_size_m"],
                              table_full_size_m=inputs["table_full_size_m"],
                              table_center_m=inputs["table_center_m"],
                              base_pose=manifest["base_pose"])

    def scene(self):
        return self._scene

    def frames(self):
        return iter(self._frames)

    def link_poses(self, record_index, *, before=False):
        """Authored URDF link poses only; before=True is an input-witness query."""
        _require(type(record_index) is int and record_index in self._records, "foreign record index")
        if type(before) is not bool:
            raise TypeError("before must be bool")
        return _fk(self._links, self._joints, self._records[record_index]["q_before" if before else "q"])

    def provenance(self, frame):
        if not isinstance(frame, SceneFrame):
            raise TypeError("provenance expects a complete SceneFrame")
        _require(self._by_key.get(frame.key) == frame, "foreign or changed replay frame")
        return deepcopy(dict(
            identity_scope="archive-replay", live_physics_identity=None,
            archive_sha256=self._manifest["archive_sha256"],
            excerpt_sha256=self._manifest["excerpt_sha256"], model_sha256=MODEL_SHA256,
            record_index=frame.key.revision, record=self._records[frame.key.revision],
            raw_record_sha256=self._record_digests[frame.key.revision],
            joint_names=list(JOINT_NAMES), geometry=self._geometry,
            state_phase=self._manifest["state_phase"],
            replay_identity=dict(world_id=frame.key.world_id, epoch=frame.key.epoch,
                                 revision=frame.key.revision, tick=frame.key.tick),
        ))
