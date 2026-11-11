"""Source-external COL3 public workspace consumer; four bounded aggregates.

Uses only installed product APIs. Original meshes, candidates, source binding
and raw TYX packets are saved; no HOST nearest search, physics, or source test /
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

SCHEMA = "phymeo-installed-triangle-dispatch-v1"
DISTANCE_GATE = 1e-4
WITNESS_GATE = 2e-4
NORMAL_GATE = 2e-3


def require(condition, message):
    if not condition:
        raise AssertionError(message)


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


def exact(left, right, message):
    require(encode(left) == encode(right), message)


def scene():
    from phymeo.constraints.collision_shapes import (
        CollisionKey,
        CollisionModel,
        CollisionMotion,
        CollisionShape,
    )

    vertices = np.array(
        ((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0), (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2)),
        np.float64,
    )
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7)]
    for i in range(4):
        j = (i + 1) % 4
        faces.extend(((i, j, j + 4), (i, j + 4, i + 4)))
    ids = tuple(11 + 13 * i for i in range(8))
    shapes = []
    motion = {}
    metadata = {}
    for world, origin, kind in (
        (7, (0.0, 0.0, 0.0), "sphere"),
        (4096, (2**20, -(2**20), 4.0), "capsule"),
    ):
        origin = np.array(origin, np.float64)
        surface = CollisionShape(
            CollisionKey(world, 101, 0),
            "mesh",
            ids,
            revision=3,
            triangles=tuple(tuple(ids[i] for i in face) for face in faces),
        )
        round_shape = CollisionShape(
            CollisionKey(world, 202, 0),
            kind,
            (11,) if kind == "sphere" else (11, 24),
            radius=0.125,
            revision=5,
        )
        axis = np.array(
            ((1, 1, 0.25),) if kind == "sphere" else ((1, 1, 0.25), (1, 1, 0.5)), np.float64
        )
        shapes.extend((surface, round_shape))
        motion[surface.key] = CollisionMotion(vertices + origin)
        motion[round_shape.key] = CollisionMotion(axis + origin)
        metadata[surface.key] = dict(
            surface=surface, round=round_shape, origin=origin, bottom=((11, 37, 24), (11, 50, 37))
        )
    return CollisionModel(tuple(shapes)), motion, metadata


def verify(packet, metadata, height):
    require(packet.schema == "phymeo-boundary-distances-v1", "aggregate schema")
    require(packet.semantics == "threshold-limited-boundary-distance", "boundary-only semantics")
    require(packet.backend in ("interpret", "cuda"), "actual selected TYX backend")
    require(packet.margin == 0.2 and len(packet.source_binding) == 64, "full input/margin binding")
    require(packet.required_candidates == 4 and len(packet.groups) == 2, "all two-world candidates")
    require(len(packet.pairs) == 6, "full original shape pair table")
    require(
        sum(p.status == "filtered" and p.reason == "different_world" for p in packet.pairs) == 4,
        "cross-world filters retained",
    )
    require(sum(p.status == "boundary_candidates" for p in packet.pairs) == 2, "two eligible pairs")
    checks = []
    for group in packet.groups:
        info = metadata[group.surface]
        require(group.round_keys == (info["round"].key,), "original round owner map")
        require(
            group.round_vertex_ids == (info["round"].vertex_ids,), "original round material IDs"
        )
        require(group.facet_vertices == info["surface"].triangles, "original complete facet table")
        # Independent original registered material triples, not generated closest points.
        expected_facets = tuple(
            i for i, face in enumerate(info["surface"].triangles) if face in info["bottom"]
        )
        require(
            len(expected_facets) == 2 and len(group.facet_vertices) == 12,
            "shared and distant fixture facets",
        )
        expected = np.array([(0, facet) for facet in expected_facets], np.int64)
        np.testing.assert_array_equal(group.candidates, expected)
        require(
            group.bvh_raw.shape == (1, 3) and group.bvh_raw.dtype == np.int32,
            "complete one-word BVH return",
        )
        expected_bits = sum(1 << facet for facet in expected_facets)
        require(
            int(group.bvh_raw[0, 0]) == expected_bits, "actual BVH prunes all ten remote facets"
        )
        require(
            int(group.bvh_raw[0, -2]) > 0 and int(group.bvh_raw[0, -1]) >= 2, "actual BVH counters"
        )
        for array in (group.candidates, group.bvh_raw, group.leaf_bounds, group.query_bounds):
            require(
                not array.flags.writeable and np.all(np.isfinite(array)),
                "immutable finite BVH/source packet",
            )
        native = group.packet
        require(
            native.snapshot_id == packet.source_binding and native.revision == 3,
            "narrow source binding",
        )
        require(len(native.rows) == 2, "both shared original facets retained")
        np.testing.assert_array_equal(native.inputs["candidates"], expected)
        np.testing.assert_array_equal(native.inputs["vertex_ids"], info["surface"].vertex_ids)
        expected_distance = 0.25 - height
        length = np.sqrt(8.0)
        for row in native.rows:
            require(
                row.status == "RESOLVED" and row.segment_id == 0,
                "resolved finite row and mapped segment",
            )
            require(
                row.facet_id in expected_facets
                and row.vertex_ids == group.facet_vertices[row.facet_id],
                "original facet and vertex identity",
            )
            require(
                0 <= row.axis_parameter <= 1 and min(row.barycentric) >= 0,
                "finite material parameters",
            )
            require(abs(sum(row.barycentric) - 1) <= 8 * 2**-24, "complete barycentric weight")
            require(
                abs(row.distance - expected_distance) <= DISTANCE_GATE * length,
                "independent plane distance",
            )
            require(
                abs(row.gap - (expected_distance - 0.125)) <= DISTANCE_GATE * length,
                "independent radius gap",
            )
            frame = np.asarray(native.inputs["origin_m"]) - info["origin"]
            axis = np.asarray(row.axis_point_relative_m) + frame
            triangle = np.asarray(row.triangle_point_relative_m) + frame
            np.testing.assert_allclose(axis, (1, 1, 0.25), atol=WITNESS_GATE * length, rtol=0)
            np.testing.assert_allclose(triangle, (1, 1, height), atol=WITNESS_GATE * length, rtol=0)
            require(row.normal is not None, "nonzero distance has a direction")
            require(
                np.linalg.norm(np.asarray(row.normal) - (0, 0, -1)) <= NORMAL_GATE,
                "independent downward direction",
            )
            if info["round"].kind == "capsule":
                require(
                    abs(row.axis_parameter) <= WITNESS_GATE,
                    "finite capsule endpoint, not infinite line",
                )
            require(
                row.selected_candidate in row.tied_candidates,
                "original finite candidate tie retained",
            )
        require(
            set(native.raw) == {"candidates", "valid", "summary", "meta"},
            "complete same-call narrow raw fields",
        )
        require(
            native.raw["candidates"].shape == (2, 18, 11) and native.raw["valid"].shape == (2, 18),
            "full finite candidate list",
        )
        for collection in (native.raw, native.uploaded, native.conversion_deltas):
            for array in collection.values():
                require(
                    not array.flags.writeable and np.all(np.isfinite(array)),
                    "complete immutable source/raw conversion",
                )
        checks.append(
            dict(
                surface=group.surface,
                round=info["round"].key,
                expected_facets=expected_facets,
                expected_distance=expected_distance,
                expected_normal=(0.0, 0.0, -1.0),
                actual_bvh_counters=group.bvh_raw[0, -2:],
                narrow_counters=group.narrow_counters,
            )
        )
    return checks


def run(mode, journal, root):
    from phymeo import BoundaryDistanceLimits, BoundaryDistances
    from phymeo.constraints.collision_shapes import CollisionMotion
    from phymeo.geometry.collision_pipeline import CollisionWorkspace

    model, motion, metadata = scene()
    changed = dict(motion)
    for surface in metadata:
        coordinates = motion[surface].start.copy()
        # Deform the original bottom four vertices; keep the top four fixed.
        coordinates[:4, 2] += 1 / 16
        changed[surface] = CollisionMotion(coordinates)
    owners = []
    attempts = []
    primary = None
    try:
        ws = CollisionWorkspace(model, backend=mode)
        owners.append(("main", ws))
        journal.emit(
            "initial",
            model=model,
            motion=motion,
            changed_motion=changed,
            mode=mode,
            metadata=metadata,
        )
        first = ws.boundary_distances(
            motion, margin=0.2, limits=BoundaryDistanceLimits(groups=2, candidates=4)
        )
        require(type(first) is BoundaryDistances, "public typed boundary result")
        attempts.append(first)
        journal.emit("aggregate", label="initial", result=first, stats=ws.stats())
        checks = [verify(first, metadata, 0.0)]
        first_copy = encode(first)
        ws.require_boundary_distances(first, motion, margin=0.2)
        before = ws.stats()
        try:
            ws.require_boundary_distances(first, changed, margin=0.2)
        except ValueError as error:
            require("binding" in str(error), "wrong-input rejection is source-bound")
            journal.emit("wrong-input-rejected", error=str(error), source=changed)
        else:
            raise AssertionError("old aggregate accepted for changed source")
        exact(ws.stats(), before, "binding check performs no device query")
        other = CollisionWorkspace(model, backend=mode)
        owners.append(("independent-owner", other))
        foreign = other.boundary_distances(motion, margin=0.2)
        attempts.append(foreign)
        journal.emit("aggregate", label="other-owner", result=foreign, stats=other.stats())
        checks.append(verify(foreign, metadata, 0.0))
        try:
            other.require_boundary_distances(first, motion, margin=0.2)
        except ValueError as error:
            require("owner" in str(error), "genuine different-workspace owner guard")
            journal.emit(
                "wrong-owner-rejected",
                error=str(error),
                accepted_owner=foreign.owner,
                wrong_owner=first.owner,
            )
        else:
            raise AssertionError("foreign aggregate accepted")
        try:
            with ws.transaction():
                staged = ws.boundary_distances(changed, margin=0.2)
                attempts.append(staged)
                journal.emit(
                    "aggregate", label="staged-before-abort", result=staged, stats=ws.stats()
                )
                checks.append(verify(staged, metadata, 1 / 16))
                require(ws.last_boundary_distances is first, "no early public aggregate swap")
                try:
                    ws.require_boundary_distances(staged, changed, margin=0.2)
                except ValueError as error:
                    require("current" in str(error), "uncommitted packet not current")
                else:
                    raise AssertionError("staged packet reported current")
                raise RuntimeError("authored outer transaction abort")
        except RuntimeError as error:
            if str(error) != "authored outer transaction abort":
                raise
            journal.emit(
                "outer-abort",
                error=str(error),
                retained=ws.last_boundary_distances,
                stats=ws.stats(),
            )
        require(ws.last_boundary_distances is first, "outer abort retains prior aggregate")
        require(encode(first) == first_copy, "prior complete evidence bytes unchanged")
        ws.require_boundary_distances(first, motion, margin=0.2)
        final = ws.boundary_distances(changed, margin=0.2)
        attempts.append(final)
        journal.emit("aggregate", label="normal-successor", result=final, stats=ws.stats())
        checks.append(verify(final, metadata, 1 / 16))
        require(
            final is ws.last_boundary_distances and final.sequence == first.sequence + 1,
            "aborted candidate did not consume the public sequence",
        )
        require(
            final.source_binding != first.source_binding,
            "actual changed vertices change complete source binding",
        )
        ws.require_boundary_distances(final, changed, margin=0.2)
        try:
            ws.require_boundary_distances(first, motion, margin=0.2)
        except ValueError as error:
            require("current" in str(error), "superseded public aggregate guard")
            journal.emit(
                "stale-rejected",
                error=str(error),
                old_binding=first.source_binding,
                new_binding=final.source_binding,
            )
        else:
            raise AssertionError("superseded aggregate accepted")
        require(
            ws.last_result is None and other.last_result is None,
            "old solid detection publication unchanged",
        )
        require(
            ws.stats()["mesh_bvh_refits"] == 6 and other.stats()["mesh_bvh_refits"] == 2,
            "exact same-call BVH work including aborted prefix",
        )
        require(
            final.groups[-1].narrow_counters["queries"] == 6
            and foreign.groups[-1].narrow_counters["queries"] == 2,
            "exact eight real COL2 group queries",
        )
        require(len(attempts) == 4, "four bounded aggregate attempts only")
        modules = provenance(root)
        for name in (
            "phymeo.geometry._triangle_distance_dispatch",
            "phymeo.backends._device_triangle_distance",
            "phymeo.backends._tyx_triangle_distance",
        ):
            require(name in modules["loaded"], "installed new module loaded: " + name)
        journal.emit(
            "verified",
            checks=checks,
            main_stats=ws.stats(),
            other_stats=other.stats(),
            modules=modules,
        )
        return dict(
            aggregate_attempts=4,
            public_main_publications=2,
            narrow_group_queries=8,
            finite_rows=16,
            worlds=(7, 4096),
            pruned_facets_per_group=10,
            outer_abort_retained=True,
            wrong_source_owner_current_rejected=True,
            modules=modules,
        )
    except BaseException as error:
        primary = error
        try:
            journal.emit(
                "attempt-failure",
                error_type=type(error).__name__,
                message=str(error),
                complete_returned_prefix=attempts,
                retained=tuple(
                    dict(label=label, last=owner.last_boundary_distances, stats=owner.stats())
                    for label, owner in owners
                ),
            )
        except BaseException as log_error:
            primary = BaseExceptionGroup(
                "original attempt and evidence-write errors", [error, log_error]
            )
        raise primary
    finally:
        failures = []
        for label, owner in reversed(owners):
            outcome = dict(label=label, status="closed")
            try:
                owner.close()
            except BaseException as error:
                failures.append(error)
                outcome.update(
                    status="close-failed", error_type=type(error).__name__, message=str(error)
                )
            try:
                journal.emit("owner-close", **outcome)
            except BaseException as error:
                failures.append(error)
        if failures:
            errors = ([primary] if primary is not None else []) + failures
            raise BaseExceptionGroup("boundary consumer/cleanup errors (original retained)", errors)


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
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output / "triangle-dispatch.jsonl.gz")
    summary = dict(
        status="RUNNING",
        mode=args.mode,
        schema=SCHEMA,
        consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="installed current HOST-snapshot boundary proximity through public CollisionWorkspace",
        excluded="solid inside/outside, global nearest outside band, CCD, load response, resident vertex source",
        gates=dict(
            distance_over_local_L=DISTANCE_GATE,
            witness_over_local_L=WITNESS_GATE,
            normal=NORMAL_GATE,
        ),
    )
    primary = None
    try:
        summary["initial_provenance"] = provenance(root, preflight=True)
        result = run(args.mode, journal, root)
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
