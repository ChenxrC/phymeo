"""Source-external M1 resident device boundary consumer; two worlds, three real steps.

Uses only installed product APIs. Original materials, candidates, accepted source binding
and raw TYX packets are saved; physics advances the accepted bank; no HOST search or source test /
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

SCHEMA = "phymeo-installed-resident-boundary-v1"
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
WORLDS = (7, 4096)
IDS = (11, 37, 91)
TRIANGLE = np.array(((0, 0, 0), (0, 2, 0), (0, 0, 2)), np.float64)
SPHERE = np.array(((-0.25, 0.5, 0.5),), np.float64)
CAPSULE = np.array(((-0.25, 0.5, 0.5), (0, 0.5, 0.5)), np.float64)
RADIUS = 1 / 8
MARGIN = 0.2


def exact(left, right, message):
    require(encode(left) == encode(right), message)


def immutable(value):
    if isinstance(value, np.ndarray):
        require(not value.flags.writeable, "mutable published array")
        try:
            value.setflags(write=True)
        except ValueError:
            pass
        else:
            raise AssertionError("published array permits writeable reenable")
    elif is_dataclass(value):
        for field in fields(value):
            immutable(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            immutable(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            immutable(item)


def make_scene(output):
    import phymeo as ph

    # Exact original F0 URDF data are embedded; no source test/fixture import.
    xml = """<?xml version="1.0"?>
<robot name="floating_mimic_jaws">
<link name="base"><inertial><mass value="2"/>
<inertia ixx="0.2" iyy="0.3" izz="0.4" ixy="0" ixz="0" iyz="0"/></inertial>
<collision name="root_marker"><geometry><sphere radius="0.03125"/></geometry></collision></link>
<link name="left_jaw"><inertial><mass value="1"/>
<inertia ixx="0.0625" iyy="0.078125" izz="0.09375" ixy="0" ixz="0" iyz="0"/></inertial>
<collision name="left_pad"><geometry><box size="0.0625 0.125 0.125"/></geometry></collision></link>
<joint name="driver" type="prismatic"><parent link="base"/><child link="left_jaw"/>
<origin xyz="-0.25 0 0"/><axis xyz="1 0 0"/>
<limit lower="-0.032" upper="0.032" effort="1.5" velocity="0.2"/></joint>
<link name="right_jaw"><inertial><mass value="2"/>
<inertia ixx="0.09375" iyy="0.109375" izz="0.125" ixy="0" ixz="0" iyz="0"/></inertial>
<collision name="right_pad"><geometry><box size="0.0625 0.125 0.125"/></geometry></collision></link>
<joint name="follower" type="prismatic"><parent link="base"/><child link="right_jaw"/>
<origin xyz="0.25 0 0"/><axis xyz="1 0 0"/>
<limit lower="-0.012" upper="0.020" effort="0.125" velocity="0.08"/>
<mimic joint="driver" multiplier="-1" offset="0.004"/></joint></robot>
"""
    asset_path = output / "input-f0.urdf"
    asset_path.write_text(xml, encoding="utf-8")
    asset = ph.load_mimic_urdf(asset_path, gravity=np.zeros(3))
    require(asset.tree.joint_names == ("driver", "follower"), "original F0 joint identities")
    trees, initial = {}, {}
    for i, world in enumerate(WORLDS):
        q = np.array(((i + 1) / 512, 0.004 - (i + 1) / 512), np.float64)
        v = np.array(((i + 1) / 1024, -(i + 1) / 1024), np.float64)
        program = ph.DrivenRobotProgram(asset.tree.model, kp=np.zeros(2), kd=np.zeros(2),
                                       effort_limit=np.ones(2), delay_steps=0)
        trees[world] = ph.FixedRobotTree(program, program.initial_state(ph.RobotState(q, v)))
        initial[world] = dict(q=q, v=v)
    return trees, initial, dict(path=str(asset_path), sha256=hashlib.sha256(asset_path.read_bytes()).hexdigest(), xml=xml)


def register(batch):
    import phymeo as ph

    shapes, bindings = [], []
    states = batch.snapshot()
    for world in WORLDS:
        surface = ph.CollisionShape(ph.CollisionKey(world, 101, 0), "triangle", IDS, revision=3)
        shapes.append(surface)
        bindings.append(ph.ResidentTreeBinding(surface.key, world, 0, TRIANGLE,
                                               topology=states[world].topology))
        for ordinal, kind, points, ids in (
            (0, "sphere", SPHERE, (11,)), (1, "capsule", CAPSULE, (11, 37))
        ):
            shape = ph.CollisionShape(ph.CollisionKey(world, 202, ordinal), kind, ids,
                                      radius=RADIUS, revision=5)
            shapes.append(shape)
            bindings.append(ph.ResidentTreeBinding(shape.key, world, 1, points,
                                                   topology=states[world].topology))
    return ph.CollisionModel(tuple(shapes)), tuple(bindings)


def verify(packet, states, initial, tick):
    import phymeo as ph

    require(packet.schema == "phymeo-resident-boundary-distances-v1", "M1 aggregate schema")
    require(packet.source_kind == "resident-rigid-device-v1", "actual resident source kind")
    require(packet.semantics == "threshold-limited-boundary-distance", "finite boundary semantics")
    require(packet.world_keys == WORLDS and packet.margin == MARGIN, "world/margin binding")
    np.testing.assert_array_equal(packet.ticks, (tick, tick))
    require(len(packet.groups) == 2 and packet.required_candidates == 4, "complete two-world membership")
    require(len(packet.pairs) == 6, "all within-world original pairs retained")
    require(sum(row.reason == "same_rigid_body" for row in packet.pairs) == 2,
            "sphere/capsule same follower filtered")
    require(sum(row.status == "boundary_candidates" for row in packet.pairs) == 4,
            "four eligible finite pairs")
    require(len(packet.source_binding) == 64, "complete source binding")
    immutable(packet)
    checks = {}
    length = np.sqrt(8.0)  # Original triangle diagonal, not world translation/distance.
    for group in packet.groups:
        world = group.surface.world
        state = states[world]
        q, v = np.asarray(state.position), np.asarray(state.velocity)
        np.testing.assert_allclose(q, initial[world]["q"] + tick * H * initial[world]["v"],
                                   atol=3e-6, rtol=0)
        np.testing.assert_allclose(v, initial[world]["v"], atol=3e-6, rtol=0)
        require(state.tick == tick, "physical accepted tick")
        require(np.any(v != 0), "real nonzero motion retained")
        expected_distance = float(0.25 + q[1] - q[0])
        expected_axis = np.array((q[1], 0.5, 0.5))
        expected_triangle = np.array((-0.25 + q[0], 0.5, 0.5))
        anchor = np.asarray(packet.source["world_anchors"][world])
        np.testing.assert_array_equal(group.candidates, np.array(((0, 0), (1, 0)), np.int64))
        require(group.facet_vertices == (IDS,), "original facet IDs")
        require(group.round_keys == (ph.CollisionKey(world, 202, 0), ph.CollisionKey(world, 202, 1)),
                "round original owners")
        require(group.bvh_raw.shape == (2, 4) and group.bvh_raw.dtype == np.int32,
                "full K+3 raw traversal")
        np.testing.assert_array_equal(group.bvh_raw[:, 0], (1, 1))
        require(np.all(group.bvh_raw[:, 1:3] >= 1), "node/leaf visits retained")
        np.testing.assert_array_equal(group.bvh_raw[:, 3], (1, 1))
        narrow = group.packet
        require(narrow.schema == "phymeo-resident-triangle-distance-v1", "resident packed schema")
        require(narrow.source_binding == packet.source_binding, "shared same-query source binding")
        np.testing.assert_array_equal(narrow.origin_m, anchor)
        require(len(narrow.rows) == 2, "no candidate loss")
        for name, shape, dtype in (("candidates", (2, 18, 11), np.float32),
                ("valid", (2, 18), np.int32), ("summary", (2, 16), np.float32),
                ("meta", (2, 7), np.int32)):
            require(narrow.raw[name].shape == shape and narrow.raw[name].dtype == dtype,
                    "full original narrow raw " + name)
        np.testing.assert_array_equal(narrow.packed["owners"], group.candidates)
        for row in narrow.rows:
            require(row.status == "RESOLVED" and row.facet_id == 0 and row.vertex_ids == IDS,
                    "complete finite resolved ownership")
            require(abs(row.distance - expected_distance) <= DISTANCE_GATE * length, "distance oracle")
            require(abs(row.gap - (expected_distance-RADIUS)) <= DISTANCE_GATE * length, "radius gap")
            require(0 <= row.axis_parameter <= 1, "finite axis address")
            if row.segment_id == 1:
                require(abs(row.axis_parameter) <= WITNESS_GATE, "nearest capsule endpoint")
            np.testing.assert_allclose(row.barycentric, (0.5, 0.25, 0.25), atol=WITNESS_GATE, rtol=0)
            np.testing.assert_allclose(np.asarray(row.axis_point_relative_m) + anchor, expected_axis,
                                       atol=WITNESS_GATE * length, rtol=0)
            np.testing.assert_allclose(np.asarray(row.triangle_point_relative_m) + anchor, expected_triangle,
                                       atol=WITNESS_GATE * length, rtol=0)
            np.testing.assert_allclose(row.normal, (-1, 0, 0), atol=NORMAL_GATE, rtol=0)
        for ordinal, local in ((0, SPHERE), (1, CAPSULE)):
            key = ph.CollisionKey(world, 202, ordinal)
            expected = local + np.array((0.25 + q[1], 0, 0)) - anchor
            np.testing.assert_allclose(packet.source["gathered_points"][key], expected,
                                       atol=WITNESS_GATE * length, rtol=0)
        expected = TRIANGLE + np.array((-0.25 + q[0], 0, 0)) - anchor
        np.testing.assert_allclose(packet.source["gathered_points"][group.surface], expected,
                                   atol=WITNESS_GATE * length, rtol=0)
        checks[world] = dict(distance=expected_distance, axis=expected_axis,
                             triangle=expected_triangle, length=length)
    require(packet.counters["live_bytes"] == packet.counters["persistent_bytes"], "temporary cleanup")
    require(packet.counters["launches"] == packet.counters["completed_launches"], "completed submissions")
    if packet.backend == "cuda":
        require(packet.counters["downloads"] > 0 and packet.counters["downloaded_bytes"] > 0,
                "real transfers counted")
    return checks

def run(mode, journal, installed_root, output):
    import phymeo as ph

    owners = []
    attempts = []
    primary = None
    try:
        trees, initial, asset = make_scene(output)
        journal.emit("fixture", asset=asset, initial=initial, worlds=WORLDS,
                     triangle=TRIANGLE, sphere=SPHERE, capsule=CAPSULE,
                     radius=RADIUS, margin=MARGIN, h=H)
        batch = ph.TyxWorldBatch(list(trees.items())[::-1], mode, cache_root=output / "cache")
        owners.append(("batch", batch))
        model, bindings = register(batch)
        journal.emit("registration", model=model, bindings=bindings, initial=batch.snapshot(),
                     batch_stats=batch.stats())
        plan = batch.prepare_boundary_query(model, bindings,
            limits=ph.BoundaryDistanceLimits(), memory_limit_bytes=32 * 1024 * 1024)
        owners.append(("plan", plan))
        last = None
        for tick in (1, 2, 3):
            before = batch.snapshot()
            step_result = batch.step(H)
            states = batch.snapshot()
            journal.emit("physical-step", tick=tick, h=H, before=before,
                         result=step_result, after=states, batch_stats=batch.stats())
            require(all(step_result.values()), "real accepted physical step")
            if last is not None:
                rejection = dict(tick=tick, previous_sequence=last.sequence)
                try:
                    plan.require_current(last)
                except ValueError as error:
                    rejection.update(status="rejected", error_type=type(error).__name__, message=str(error))
                else:
                    rejection.update(status="unexpected-success")
                journal.emit("stale-query-binding", **rejection)
                require(rejection["status"] == "rejected", "physical source change invalidates old packet")
            attempt = dict(tick=tick, before=states, before_stats=plan.stats(), margin=MARGIN)
            attempts.append(attempt)
            try:
                result = plan.query(margin=MARGIN)
                attempt["result"] = result
            finally:
                attempt.update(after=batch.snapshot(), last_attempt=plan.last_attempt,
                               last_result=plan.last_result, after_stats=plan.stats())
                # Full returned result/raw/source and any failed prefix precede every assertion.
                journal.emit("query-outcome", **attempt)
            exact(states, attempt["after"], "query cannot mutate accepted physical bank")
            require(plan.require_current(result) is result, "new current source accepted")
            checks = verify(result, states, initial, tick)
            journal.emit("query-checks", tick=tick, checks=checks)
            last = result
        modules = provenance(installed_root)
        journal.emit("completed", accepted_world_steps=6, complete_queries=3,
                     full_rows=12, modules=modules, plan_stats=plan.stats(), batch_stats=batch.stats())
        return dict(accepted_world_steps=6, complete_queries=3, rows=12,
                    stale_rejections=2, final_ticks=last.ticks, modules=modules)
    except BaseException as error:
        primary = error
        try:
            journal.emit("attempt-failure", error_type=type(error).__name__, message=str(error),
                         complete_attempt_prefix=attempts,
                         owners=tuple(dict(label=label, stats=owner.stats(),
                            last=getattr(owner, "last_result", None),
                            attempt=getattr(owner, "last_attempt", None)) for label, owner in owners))
        except BaseException as log_error:
            primary = BaseExceptionGroup("original attempt and evidence-write errors", [error, log_error])
        raise primary
    finally:
        failures = []
        for label, owner in reversed(owners):
            outcome = dict(label=label, status="closed")
            try:
                owner.close()
            except BaseException as error:
                failures.append(error)
                outcome.update(status="close-failed", error_type=type(error).__name__, message=str(error))
            try:
                journal.emit("owner-close", **outcome)
            except BaseException as error:
                failures.append(error)
        if failures:
            raise BaseExceptionGroup("M1 consumer/cleanup errors (original retained)",
                                     ([primary] if primary is not None else []) + failures)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--installed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.output.is_absolute() and args.installed_root.is_absolute(), "absolute paths required"
    )
    root = args.installed_root.resolve(strict=True)
    output = args.output.resolve()
    require(not any((parent / "pyproject.toml").is_file() and
                    (parent / "src" / "phymeo").is_dir()
                    for parent in (output, *output.parents)), "output must be outside checkouts")
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output / "resident-boundary.jsonl.gz")
    summary = dict(
        status="RUNNING",
        mode=args.mode,
        schema=SCHEMA,
        consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="installed accepted rigid physical bank to device BVH and finite triangle distance",
        excluded="solid inside/outside, global nearest outside band, CCD, load response, deformable source bank",
        gates=dict(
            distance_over_local_L=DISTANCE_GATE,
            witness_over_local_L=WITNESS_GATE,
            normal=NORMAL_GATE,
        ),
    )
    primary = None
    try:
        summary["initial_provenance"] = provenance(root, preflight=True)
        result = run(args.mode, journal, root, output)
        summary["result"] = encode(result)
        summary["status"] = "PASS"
    except BaseException as error:
        primary = error
        summary["status"] = "FAIL"
        summary["exception"] = dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
        )
        try:
            journal.emit("failure", summary=summary)
        except BaseException as log_error:
            primary = BaseExceptionGroup(
                "original consumer and failure-write errors", [error, log_error]
            )
    final_errors = []
    try:
        summary["final_provenance"] = provenance(root)
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["provenance_error"] = traceback.format_exc()
        final_errors.append(error)
    try:
        journal.emit("summary", summary=summary)
    except BaseException as error:
        final_errors.append(error)
    try:
        summary["trace"] = journal.close()
    except BaseException as error:
        final_errors.append(error)
    try:
        if final_errors:
            summary["status"] = "FAIL"
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(json.dumps(summary, allow_nan=False))
    except BaseException as error:
        final_errors.append(error)
    if primary is not None or final_errors:
        errors = ([primary] if primary is not None else []) + final_errors
        if len(errors) == 1:
            raise errors[0]
        raise BaseExceptionGroup(
            "installed consumer and finalization errors (original retained)", errors
        )


if __name__ == "__main__":
    main()
