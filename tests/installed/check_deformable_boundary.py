"""Installed D2 consumer: real changing node banks, finite static round queries."""

import argparse
import base64
import gzip
import hashlib
import json
import sys
import traceback
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

H, MARGIN = 1 / 128, 1 / 4
WORLDS = (7, 31)


def require(value, message):
    if not value:
        raise AssertionError(message)


def encode(value):
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject, "numeric evidence only")
        return dict(
            dtype=value.dtype.str,
            shape=value.shape,
            bytes_b64=base64.b64encode(value.tobytes()).decode(),
        )
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return dict(
            type=type(value).__module__ + "." + type(value).__name__,
            fields={f.name: encode(getattr(value, f.name)) for f in fields(value)},
        )
    if isinstance(value, Mapping):
        return dict(mapping_entries=[[encode(k), encode(v)] for k, v in value.items()])
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if type(value) is object:
        return dict(opaque_owner_identity=True)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError("unhandled evidence type " + type(value).__name__)


class Journal:
    def __init__(self, path):
        self.path, self.stream = path, gzip.open(path, "xb")
        self.digest, self.bytes, self.records = hashlib.sha256(), 0, 0

    def emit(self, event, **payload):
        raw = (
            json.dumps(
                dict(event=event, payload=encode(payload)), allow_nan=False, separators=(",", ":")
            )
            + "\n"
        ).encode()
        self.stream.write(raw)
        self.digest.update(raw)
        self.bytes += len(raw)
        self.records += 1

    def close(self):
        self.stream.close()
        return dict(
            logical_sha256=self.digest.hexdigest(),
            logical_bytes=self.bytes,
            records=self.records,
            compressed_bytes=self.path.stat().st_size,
        )


def provenance(root, *, preflight=False):
    def checkout(path):
        return ((path / "pyproject.toml").is_file() and (path / "src/phymeo").is_dir()) or (
            path.name.lower() == "src"
            and (path / "phymeo").is_dir()
            and (path.parent / "pyproject.toml").is_file()
        )

    require(
        not any(checkout(Path(p or Path.cwd()).resolve()) for p in sys.path),
        "checkout source entry in sys.path",
    )
    if preflight:
        require(
            not any(n == "phymeo" or n.startswith("phymeo.") for n in sys.modules),
            "product imported before installed preflight",
        )
        for origin in (Path.cwd().resolve(), Path(__file__).resolve().parent, root):
            require(
                not any(checkout(p) for p in (origin, *origin.parents)),
                "consumer, cwd and install must be outside checkouts",
            )
    import phymeo

    require(Path(phymeo.__file__).resolve().is_relative_to(root), "wrong installed package")
    result = {}
    for name, module in sorted(sys.modules.items()):
        if name == "phymeo" or name.startswith("phymeo."):
            location = getattr(module, "__file__", None)
            if location is not None:
                path = Path(location).resolve()
                require(path.is_relative_to(root), "mixed checkout module " + name)
                result[name] = dict(
                    path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                )
            for directory in getattr(module, "__path__", ()):
                require(
                    Path(directory).resolve().is_relative_to(root), "mixed package path " + name
                )
    return result


def declarations():
    from phymeo.backends.tyx_deformable_batch import DeformableNodeWorld, DeformableSurface
    from phymeo.constraints.distance import DistanceConstraints
    from phymeo.core.model import ParticleModel
    from phymeo.core.state import ParticleState

    specs = []
    for world in WORLDS:
        square = world == 31
        x = np.array(
            ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
            if square
            else ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
            np.float32,
        )
        v, weights = np.zeros_like(x), np.ones(len(x), np.float32)
        if square:
            v[2:, 0], weights[0] = 1 / 16, 0
            edges = np.array(((0, 1), (1, 2), (2, 3), (3, 0), (0, 2)), np.int64)
            ids, facets = np.array((101, 113, 127, 139)), np.array((503, 509))
            faces = np.array(((101, 113, 127), (101, 127, 139)))
        else:
            v[:2, 0] = (-1 / 16, 1 / 16)
            edges = np.array(((0, 1),), np.int64)
            ids, facets, faces = np.array((11, 37, 91)), np.array((501,)), np.array(((11, 37, 91),))
        rest = np.linalg.norm(x[edges[:, 0]].astype(float) - x[edges[:, 1]], axis=1)
        specs.append(
            DeformableNodeWorld(
                world,
                ParticleModel(weights, np.zeros(3, np.float32)),
                ParticleState(x, v),
                DistanceConstraints(edges, rest, np.full(len(edges), H * H)),
                ids,
                (DeformableSurface(23, facets, faces),),
            )
        )
    return tuple(specs)


def point_segment(point, a, b):
    delta = b - a
    denominator = float(delta @ delta)
    t = 0.0 if denominator == 0 else float(np.clip((point - a) @ delta / denominator, 0, 1))
    return a + t * delta


def point_triangle(point, tri):
    # Independent convex-face minimization: interior stationary point plus edges.
    a, b, c = tri
    matrix = np.column_stack((b - a, c - a))
    uv = np.linalg.solve(matrix.T @ matrix, matrix.T @ (point - a))
    choices = [point_segment(point, tri[i], tri[(i + 1) % 3]) for i in range(3)]
    if uv[0] >= 0 and uv[1] >= 0 and uv.sum() <= 1:
        choices.append(a + matrix @ uv)
    return min(choices, key=lambda q: float((q - point) @ (q - point)))


def distance_reference(tri, axis):
    # Independent finite segment/triangle distance. No product packed geometry is input.
    candidates = [(p, point_triangle(p, tri)) for p in axis]
    a, b = axis
    u = b - a
    for i in range(3):
        c, d = tri[i], tri[(i + 1) % 3]
        v, w = d - c, a - c
        candidates.extend(((point_segment(c, a, b), c), (point_segment(d, a, b), d)))
        aa, bb, cc = float(u @ u), float(u @ v), float(v @ v)
        det = aa * cc - bb * bb
        if det > 0:
            du, dv = float(u @ w), float(v @ w)
            s, t = (bb * dv - cc * du) / det, (aa * dv - bb * du) / det
            if 0 <= s <= 1 and 0 <= t <= 1:
                candidates.append((a + s * u, c + t * v))
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    denominator = float(normal @ u)
    if denominator != 0:
        s = float(normal @ (tri[0] - a)) / denominator
        if 0 <= s <= 1:
            point = a + s * u
            witness = point_triangle(point, tri)
            candidates.append((point, witness))
    p, q = min(candidates, key=lambda pair: float((pair[0] - pair[1]) @ (pair[0] - pair[1])))
    return float(np.linalg.norm(q - p)), p, q


def run(mode, journal, progress):
    import phymeo as ph
    from phymeo.backends.tyx_deformable_batch import TyxDeformableBatch
    from phymeo.core.scene import StepSettings
    from phymeo.core.state import StepInputs

    specs = declarations()
    settings = StepSettings(substeps=1, position_iterations=200, tolerance=2e-6)
    shapes, bindings, rounds = [], [], {}
    for spec in specs:
        key = ph.CollisionKey(spec.key, 101, 0)
        shapes.append(ph.DeformableBoundarySurface(key, 23))
        for ordinal, (kind, points) in enumerate(
            (
                ("sphere", ((0.25, 0.25, 0.125),)),
                ("capsule", ((0.375, 0.375, 0.25), (0.625, 0.625, 0.25))),
                ("sphere", ((8.0, 8.0, 8.0),)),
            )
        ):
            key = ph.CollisionKey(spec.key, 202, ordinal)
            shapes.append(ph.CollisionShape(key, kind, tuple(range(len(points))), radius=0.125))
            rounds[key] = np.asarray(points, np.float64)
            bindings.append(ph.ResidentStaticBinding(key, ph.CollisionMotion(rounds[key])))
    model = ph.DeformableBoundaryModel(tuple(shapes))
    by_world = {s.key: s for s in specs}
    peak_edge, peak_bounds = {w: 0.0 for w in WORLDS}, {w: 0.0 for w in WORLDS}
    original_bounds = {}
    forbidden = []

    def no_cpu_projection(*args, **kwargs):
        forbidden.append(True)
        raise AssertionError("HOST projection called by D2 consumer")

    primary, cleanup = None, []
    row_states = {"RESOLVED": 0, "CONDITIONING": 0}
    stack = ExitStack()
    try:
        stack.enter_context(
            patch("phymeo.constraints.distance.project_distances", no_cpu_projection)
        )
        owner = stack.enter_context(TyxDeformableBatch(specs, mode=mode, settings=settings))
        plan = owner.prepare_boundary_query(model, tuple(bindings))
        stack.callback(plan.close)
        journal.emit(
            "initial",
            declarations=specs,
            model=model,
            bindings=bindings,
            state=owner.snapshot(),
            checkpoint=owner.checkpoint(),
        )

        def query():
            before, revision = owner.snapshot(), owner.revision
            journal.emit("query-request", before=before, revision=revision, margin=MARGIN)
            try:
                result = plan.query(margin=MARGIN)
                progress["queries_returned"] += 1
                journal.emit(
                    "query-return", result=result, attempt=plan.last_attempt, stats=plan.stats()
                )
                require(
                    owner.revision == revision and owner.snapshot() is before,
                    "query changed physics",
                )
                require(result.source_kind == "deformable-node-device-v1", "true producer source")
                require(
                    result.world_keys == WORLDS and result.state_epoch == revision,
                    "complete world and accepted revision binding",
                )
                require(
                    result.source["producer_binding_sha256"] == owner.binding_sha256,
                    "original producer binding",
                )
                np.testing.assert_array_equal(result.ticks, [before[w].tick for w in WORLDS])
                require(
                    tuple(g.surface for g in result.groups)
                    == tuple(ph.CollisionKey(w, 101, 0) for w in WORLDS),
                    "all declared surfaces",
                )
                total = 0
                for group in result.groups:
                    require(
                        group.round_keys
                        == tuple(ph.CollisionKey(group.surface.world, 202, i) for i in range(3)),
                        "all original round shapes",
                    )
                    require(
                        group.packet.source_binding == result.source_binding,
                        "same query source binding",
                    )
                    spec, state = by_world[group.surface.world], before[group.surface.world]
                    surface = spec.surfaces[0]
                    ids = tuple(map(int, spec.point_ids))
                    faces = tuple(
                        tuple(ids.index(int(v)) for v in face)
                        for face in surface.oriented_point_ids
                    )
                    current = state.position.astype(np.float64)
                    expected = []
                    for si, key in enumerate(group.round_keys):
                        axis = rounds[key]
                        axis = np.repeat(axis, 2, axis=0) if len(axis) == 1 else axis
                        lo, hi = (
                            axis.min(axis=0) - 0.125 - MARGIN,
                            axis.max(axis=0) + 0.125 + MARGIN,
                        )
                        for fi, face in enumerate(faces):
                            tri = current[list(face)]
                            if np.all(tri.min(axis=0) <= hi) and np.all(lo <= tri.max(axis=0)):
                                expected.append((si, fi))
                    require(
                        tuple(map(tuple, group.candidates)) == tuple(expected),
                        "complete independent AABB set",
                    )
                    np.testing.assert_array_equal(
                        result.source["surfaces"][group.surface]["facet_ids"], surface.facet_ids
                    )
                    independent = []
                    for row, (si, fi) in zip(group.packet.rows, expected, strict=True):
                        tri, axis = current[list(faces[fi])], rounds[group.round_keys[si]]
                        axis = np.repeat(axis, 2, axis=0) if len(axis) == 1 else axis
                        distance, p, q = distance_reference(tri, axis)
                        independent.append(
                            dict(
                                segment=si,
                                facet_ordinal=fi,
                                original_facet=int(surface.facet_ids[fi]),
                                triangle=tri,
                                axis=axis,
                                distance=distance,
                                witness=(p, q),
                                row=row,
                            )
                        )
                    journal.emit(
                        "independent-before-check", surface=group.surface, values=independent
                    )
                    for row_index, item in enumerate(independent):
                        row, tri, axis, distance = (
                            item["row"],
                            item["triangle"],
                            item["axis"],
                            item["distance"],
                        )
                        scale = max(
                            0.125,
                            np.linalg.norm(axis[1] - axis[0]),
                            *(np.linalg.norm(tri[i] - tri[j]) for i, j in ((0, 1), (1, 2), (2, 0))),
                        )
                        # This original finite kernel conservatively rejects oblique
                        # parallel edge-axis pairs. Preserve that status on the
                        # declared shared-edge patch; it is not a contact normal.
                        patch_capsule = (
                            len(faces) == 2 and len(rounds[group.round_keys[item["segment"]]]) == 2
                        )
                        allowed = ("RESOLVED", "CONDITIONING") if patch_capsule else ("RESOLVED",)
                        require(row.status in allowed, "declared finite query status domain")
                        if patch_capsule and state.tick == 0:
                            require(row.status == "CONDITIONING", "initial oblique parallel state")
                        require(row.selected_candidate >= 0, "finite diagnostic candidate")
                        row_states[row.status] += 1
                        require(
                            (row.segment_id, row.facet_id)
                            == (item["segment"], item["facet_ordinal"]),
                            "row material identity",
                        )
                        require(0 <= row.axis_parameter <= 1, "finite axis parameter")
                        require(
                            abs(row.distance - distance) <= 1e-4 * scale
                            and abs(row.gap - (distance - 0.125)) <= 1e-4 * scale,
                            "independent gap",
                        )
                        require(
                            tuple(row.vertex_ids)
                            == tuple(surface.oriented_point_ids[item["facet_ordinal"]]),
                            "original oriented IDs",
                        )
                        bary = np.asarray(row.barycentric)
                        require(
                            np.min(bary) >= 0
                            and abs(bary.sum() - 1) <= 8 * np.finfo(np.float32).eps,
                            "barycentric identity",
                        )
                        ap = axis[0] + row.axis_parameter * (axis[1] - axis[0])
                        tp = tri[0] + bary @ (tri - tri[0])
                        np.testing.assert_allclose(
                            row.axis_point_relative_m + group.packet.origin_m,
                            ap,
                            rtol=0,
                            atol=2e-4 * scale,
                        )
                        np.testing.assert_allclose(
                            row.triangle_point_relative_m + group.packet.origin_m,
                            tp,
                            rtol=0,
                            atol=2e-4 * scale,
                        )
                        require(
                            abs(np.linalg.norm(tp - ap) - distance) <= 2e-4 * scale,
                            "common witnesses",
                        )
                        if row.status == "RESOLVED":
                            np.testing.assert_allclose(
                                row.normal, (tp - ap) / distance, rtol=0, atol=2e-3
                            )
                        else:
                            require(
                                row.normal is None, "unresolved row cannot supply a force normal"
                            )
                            require(
                                group.packet.raw["meta"][row_index, 3] == 0, "raw normal undefined"
                            )
                            np.testing.assert_array_equal(
                                group.packet.raw["summary"][row_index, 12:15], 0
                            )
                    total += len(expected)
                    w = group.surface.world
                    if w not in original_bounds:
                        original_bounds[w] = group.leaf_bounds.copy()
                    peak_bounds[w] = max(
                        peak_bounds[w],
                        float(np.max(np.abs(group.leaf_bounds - original_bounds[w]))),
                    )
                require(total == result.required_candidates, "complete candidate count")
                plan.require_current(result)
                progress["queries_checked"] += 1
                return result
            except BaseException:
                journal.emit(
                    "query-failure-prefix",
                    attempt=plan.last_attempt,
                    last=plan.last_result,
                    stats=plan.stats(),
                )
                raise

        def advance(tick):
            inputs = {}
            for spec in specs:
                force = np.zeros_like(spec.initial_state.position)
                if spec.key == 31:
                    force[:, 1] = 1 / 32 if tick < 6 else -1 / 32
                inputs[spec.key] = StepInputs(force)
            journal.emit("step-request", inputs=inputs, h=H, expected_revision=owner.revision)
            progress["steps_attempted"] += 1
            receipt = owner.step(inputs, H, expected_revision=owner.revision)
            progress["steps_returned"] += 1
            journal.emit("step-return", receipt=receipt)
            require(receipt.accepted, "producer accepted step")
            after = owner.snapshot()
            journal.emit("accepted-state", state=after)
            for spec in specs:
                edges = spec.constraints.indices
                initial = spec.initial_state.position
                current = after[spec.key].position
                delta = np.max(
                    np.abs(
                        np.linalg.norm(current[edges[:, 1]] - current[edges[:, 0]], axis=1)
                        - np.linalg.norm(initial[edges[:, 1]] - initial[edges[:, 0]], axis=1)
                    )
                )
                peak_edge[spec.key] = max(peak_edge[spec.key], float(delta))
                require(
                    np.all(after[spec.key].position[:, 2] == 0), "no spurious out-of-plane motion"
                )
            np.testing.assert_array_equal(after[31].position[0], specs[1].initial_state.position[0])
            query()
            return after

        query()
        for tick in range(12):
            advance(tick)
        require(all(v > 1e-6 for v in peak_edge.values()), "both accepted surfaces really deformed")
        require(all(v > 1e-6 for v in peak_bounds.values()), "both real bounds changed")
        capsule = owner.checkpoint()
        a = advance(12)
        old = plan.last_result
        journal.emit("restore-request", checkpoint=capsule, expected_revision=owner.revision)
        owner.restore(capsule, expected_revision=owner.revision)
        journal.emit("restore-return", state=owner.snapshot())
        try:
            plan.require_current(old)
        except ValueError as error:
            journal.emit("expected-stale", error=str(error))
        else:
            raise AssertionError("old query survived restore")
        query()
        b = advance(12)
        for world in WORLDS:
            for field in ("position", "velocity", "clock"):
                require(
                    getattr(a[world], field).tobytes() == getattr(b[world], field).tobytes(),
                    "restore successor bytes",
                )
        before_reset, old = owner.snapshot(), plan.last_result
        other = before_reset[31]
        journal.emit(
            "reset-request", worlds=(7,), expected_revision=owner.revision, before=before_reset
        )
        owner.reset((7,), expected_revision=owner.revision)
        journal.emit("partial-reset", state=owner.snapshot())
        require(owner.snapshot()[31] is other, "unselected accepted state retained")
        selected = owner.snapshot()[7]
        np.testing.assert_array_equal(selected.position, specs[0].initial_state.position)
        np.testing.assert_array_equal(selected.velocity, specs[0].initial_state.velocity)
        np.testing.assert_array_equal(selected.clock, np.zeros(2, np.float32))
        require(selected.tick == 0, "selected world reset to initial tick")
        try:
            plan.require_current(old)
        except ValueError as error:
            journal.emit("expected-reset-stale", error=str(error))
        else:
            raise AssertionError("old query survived reset")
        query()
        advance(13)
        advance(14)
        journal.emit(
            "final",
            state=owner.snapshot(),
            checkpoint=owner.checkpoint(),
            query=plan.last_result,
            query_stats=plan.stats(),
            producer_stats=owner.stats(),
            peak_edge_change=peak_edge,
            peak_bounds_change=peak_bounds,
        )
    except BaseException as error:
        primary = error
    finally:
        try:
            stack.close()
        except BaseException as error:
            cleanup.append(error)
    if primary is not None:
        if cleanup:
            raise BaseExceptionGroup("D2 consumer and cleanup failed", [primary, *cleanup])
        raise primary
    if cleanup:
        raise BaseExceptionGroup("D2 cleanup failed", cleanup)
    require(
        progress
        == dict(steps_attempted=16, steps_returned=16, queries_returned=19, queries_checked=19),
        "fixed complete counts",
    )
    require(not forbidden, "no HOST projection")
    return dict(
        steps=16,
        worldsteps=32,
        queries=19,
        cpu_projection_calls=0,
        row_states=row_states,
        scope="D2 deformable query with conservative parallel CONDITIONING; no contact response or CCD",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--expected-install-root", "--installed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute(), "absolute fresh output required")
    args.output.mkdir(parents=True, exist_ok=False)
    require(args.expected_install_root.is_absolute(), "absolute installed root required")
    root = args.expected_install_root.resolve(strict=True)
    journal = Journal(args.output / "deformable-boundary.jsonl.gz")
    progress = dict(steps_attempted=0, steps_returned=0, queries_returned=0, queries_checked=0)
    summary = dict(
        status="RUNNING",
        mode=args.mode,
        consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    errors = []

    def remember(stage, error):
        errors.append(error)
        summary["status"] = "FAIL"
        summary.setdefault("errors", []).append(
            dict(
                stage=stage,
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            )
        )

    try:
        summary["initial_provenance"] = provenance(root, preflight=True)
        summary.update(run(args.mode, journal, progress))
        summary["status"] = "PASS"
    except BaseException as error:
        remember("run", error)
    finally:
        summary["progress"] = progress
        for stage, callback in (
            ("final-provenance", lambda: summary.update(final_provenance=provenance(root))),
            ("journal-final", lambda: journal.emit("summary", summary=summary)),
            ("journal-close", lambda: summary.update(trace=journal.close())),
        ):
            try:
                callback()
            except BaseException as error:
                remember(stage, error)
        try:
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
            )
        except BaseException as error:
            remember("summary-write", error)
    print(json.dumps(summary, allow_nan=False))
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("D2 consumer/finalization failures", errors)


if __name__ == "__main__":
    main()
