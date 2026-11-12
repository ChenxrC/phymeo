"""D2 tests first: accepted deforming vertices, original finite-distance oracle."""

import base64
import gzip
import importlib.util
import json
from collections.abc import Mapping
from contextlib import closing, contextmanager
from dataclasses import fields, is_dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends._deformable_boundary_source import (
    DeformableBoundaryModel,
    DeformableBoundarySurface,
)
from phymeo.backends.tyx_deformable_batch import (
    DeformableNodeWorld,
    DeformableSurface,
    TyxDeformableBatch,
)
from phymeo.constraints.distance import DistanceConstraints
from phymeo.core.model import ParticleModel
from phymeo.core.scene import StepSettings
from phymeo.core.state import ParticleState, StepInputs

pytestmark = pytest.mark.tyx
H = 1 / 128
SETTINGS = StepSettings(substeps=1, position_iterations=200, tolerance=2e-6)
MARGIN = 1 / 4


def world(key=7, square=False, fixed=False, reverse=False):
    # Exact D1 declarations; changing device state is produced by original kernels.
    x = np.array(
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
        if square
        else ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        np.float32,
    )
    v = np.zeros_like(x)
    if square:
        v[2:, 0] = 1 / 16
        edges = np.array(((0, 1), (1, 2), (2, 3), (3, 0), (0, 2)), np.int64)
        ids = np.array((101, 113, 127, 139), np.int64)
        faces = np.array(((101, 113, 127), (101, 127, 139)), np.int64)
        facets = np.array((503, 509), np.int64)
    else:
        v[:2, 0] = (-1 / 16, 1 / 16)
        edges = np.array(((0, 1),), np.int64)
        ids = np.array((11, 37, 91), np.int64)
        faces = np.array(((11, 37, 91),), np.int64)
        facets = np.array((501,), np.int64)
    weights = np.ones(len(x), np.float32)
    if fixed:
        weights[0], v[0] = 0, 0
    rest = np.linalg.norm(x[edges[:, 0]].astype(float) - x[edges[:, 1]], axis=1)
    if reverse:
        order = np.arange(len(x))[::-1]
        inverse = np.argsort(order)
        x, v, weights, ids, edges = x[order], v[order], weights[order], ids[order], inverse[edges]
        faces, facets = faces[::-1], facets[::-1]
    return DeformableNodeWorld(
        key,
        ParticleModel(weights, np.zeros(3, np.float32)),
        ParticleState(x, v),
        DistanceConstraints(edges, rest, np.full(len(edges), H * H)),
        ids,
        (DeformableSurface(23, facets, faces),),
    )


def forces(specs, tick=0):
    out = {}
    for spec in specs:
        value = np.zeros((spec.model.particle_count, 3), np.float32)
        if spec.key == 31:
            value[:, 1] = 1 / 32 if tick < 6 else -1 / 32
        out[spec.key] = StepInputs(value)
    return out


def scene(specs, *, reverse=False, masked=False, excluded=False):
    shapes, bindings, rounds = [], [], {}
    exclusions = []
    for spec in specs:
        surface = ph.CollisionKey(spec.key, 101, 0)
        shapes.append(DeformableBoundarySurface(surface, 23, mask=0 if masked else 2**31 - 1))
        for ordinal, (kind, points) in enumerate(
            (
                ("sphere", ((0.25, 0.25, 0.125),)),
                ("capsule", ((0.375, 0.375, 0.25), (0.625, 0.625, 0.25))),
                ("sphere", ((8.0, 8.0, 8.0),)),
            )
        ):
            key = ph.CollisionKey(spec.key, 202, ordinal)
            shapes.append(
                ph.CollisionShape(key, kind, tuple(range(11, 11 + len(points))), radius=0.125)
            )
            bindings.append(ph.ResidentStaticBinding(key, ph.CollisionMotion(np.asarray(points))))
            rounds[key] = np.asarray(points, np.float64)
            if excluded and ordinal == 1:
                exclusions.append((surface, key))
    if reverse:
        shapes.reverse()
        bindings.reverse()
    return DeformableBoundaryModel(tuple(shapes), tuple(exclusions)), tuple(bindings), rounds


def encode(value):
    if isinstance(value, np.ndarray):
        return dict(
            dtype=value.dtype.str,
            shape=value.shape,
            bytes_b64=base64.b64encode(value.tobytes()).decode(),
        )
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return dict(
            type=type(value).__name__,
            fields={f.name: encode(getattr(value, f.name)) for f in fields(value)},
        )
    if isinstance(value, Mapping):
        return dict(mapping_entries=[[encode(k), encode(v)] for k, v in value.items()])
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if type(value) is object:
        return dict(opaque_owner_identity=True)
    return value


@contextmanager
def saved(path, specs, model, bindings):
    payload = dict(
        schema="D2-complete-query-evidence-v1",
        declarations=specs,
        model=model,
        bindings=bindings,
        events=[],
        complete=False,
    )
    try:
        yield payload["events"]
        payload["complete"] = True
    except BaseException as error:
        payload["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        with gzip.open(path / "evidence.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(encode(payload), stream, allow_nan=False, sort_keys=True)


@lru_cache(maxsize=1)
def reference():
    path = Path(__file__).parents[1] / "reference_newton/rr/resident_boundary_source_reference.py"
    spec = importlib.util.spec_from_file_location("d2_independent_distance_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def declared_membership(model):
    """Expected groups and complete pair coverage from declarations, never output."""
    groups = {}
    pairs = []
    excluded = set(model.excluded)
    for i, a in enumerate(model.shapes):
        for b in model.shapes[i + 1 :]:
            if a.key.world != b.key.world:
                continue
            reason = (
                "same_rigid_body"
                if a.key.body == b.key.body
                else "mask"
                if not (a.group & b.mask and b.group & a.mask)
                else "excluded"
                if (a.key, b.key) in excluded
                else None
            )
            if reason is not None:
                pairs.append((a.key, b.key, "filtered", reason, None, None))
                continue
            a_surface = type(a) is DeformableBoundarySurface
            b_surface = type(b) is DeformableBoundarySurface
            if a_surface == b_surface:
                pairs.append((a.key, b.key, "not_applicable", "same_family", None, None))
                continue
            surface, segment = (a, b) if a_surface else (b, a)
            groups.setdefault(surface.key, []).append(segment.key)
            pairs.append((a.key, b.key, None, "finite_boundary", surface.key, segment.key))
    return {key: tuple(sorted(groups[key])) for key in sorted(groups)}, tuple(pairs)


def assert_d2_row(
    row, triangle, segment, radius, packet, index, *, allow_conditioning, initial, evidence
):
    """Unresolved is not promoted; every retained witness still gets an oracle check."""
    ref = reference().distance_reference()
    tri, axis = np.asarray(triangle, np.float64), reference().axis(segment)
    length = ref.local_scale(tri, axis, radius)
    ideal = ref.oracle(tri, axis)
    distance = float(ideal["distance2"]) ** 0.5
    evidence.append(
        dict(
            triangle=tri,
            axis=axis,
            radius=radius,
            length=length,
            status=row.status,
            distance_squared_exact=str(ideal["distance2"]),
            independent_witnesses_exact=repr(ideal["witnesses"]),
            distance=distance,
            allow_conditioning=allow_conditioning,
            initial=initial,
        )
    )
    if allow_conditioning:
        assert row.status in ("RESOLVED", "CONDITIONING")
        if initial:
            assert row.status == "CONDITIONING"
    else:
        assert row.status == "RESOLVED"
    raw, summary = packet.raw, packet.raw["summary"][index]
    meta, valid, candidates = raw["meta"][index], raw["valid"][index], raw["candidates"][index]
    for value in raw.values():
        assert np.isfinite(value).all()
    assert int(meta[0]) == {"RESOLVED": 1, "CONDITIONING": 2}[row.status]
    assert (int(meta[5]), int(meta[6])) == (row.segment_id, row.facet_id)
    assert int(meta[1]) == row.selected_candidate and 0 <= row.selected_candidate < 18
    assert np.isin(valid, (0, 1)).all() and valid[row.selected_candidate] == 1
    assert row.selected_candidate in row.tied_candidates
    assert all(valid[c] == 1 for c in row.tied_candidates)
    assert np.all(candidates[valid == 0] == 0)
    pair_origin = np.asarray(packet.origin_m) + packet.pair_origins_relative_m[index]
    # Check every emitted candidate in the independently supplied source frame.
    for candidate in candidates[valid == 1]:
        s, b0, b1, b2, d2 = map(float, candidate[:5])
        bary = np.array((b0, b1, b2))
        assert 0 <= s <= 1 and np.min(bary) >= 0
        assert abs(float(bary.sum()) - 1) <= 8 * reference().U and d2 >= 0
        ap = axis[0] + s * (axis[1] - axis[0])
        tp = tri[0] + bary @ (tri - tri[0])
        np.testing.assert_allclose(candidate[5:8] + pair_origin, ap, rtol=0, atol=2e-4 * length)
        np.testing.assert_allclose(candidate[8:11] + pair_origin, tp, rtol=0, atol=2e-4 * length)
        delta = candidate[8:11].astype(float) - candidate[5:8]
        assert abs(d2 - float(delta @ delta)) <= 32 * reference().U * max(length * length, d2)
    assert candidates[row.selected_candidate, 4] == np.min(candidates[valid == 1, 4])
    assert row.distance == float(summary[0]) and row.gap == float(summary[1])
    assert abs(row.distance - distance) <= 1e-4 * length
    assert abs(row.gap - (distance - radius)) <= 1e-4 * length
    bary = np.asarray(row.barycentric)
    assert np.min(bary) >= 0 and abs(float(bary.sum()) - 1) <= 8 * reference().U
    assert 0 <= row.axis_parameter <= 1
    ap = axis[0] + row.axis_parameter * (axis[1] - axis[0])
    tp = tri[0] + bary @ (tri - tri[0])
    np.testing.assert_allclose(
        np.asarray(row.axis_point_relative_m) + packet.origin_m, ap, rtol=0, atol=2e-4 * length
    )
    np.testing.assert_allclose(
        np.asarray(row.triangle_point_relative_m) + packet.origin_m, tp, rtol=0, atol=2e-4 * length
    )
    assert abs(float(np.linalg.norm(tp - ap)) - distance) <= 2e-4 * length
    if row.status == "CONDITIONING" or distance <= 32 * reference().U * length:
        assert row.normal is None and int(meta[3]) == 0
        np.testing.assert_array_equal(summary[12:15], np.zeros(3, np.float32))
    else:
        assert int(meta[3]) == 1 and row.normal is not None
        direction = (
            np.asarray(ideal["witnesses"][0][4], float)
            - np.asarray(ideal["witnesses"][0][3], float)
        ) / distance
        assert np.linalg.norm(np.asarray(row.normal) - direction) <= 2e-3


def check(batch, plan, specs, model, rounds, events):
    before = batch.snapshot()
    event = dict(before=before, revision=batch.revision, margin=MARGIN)
    events.append(event)
    try:
        result = plan.query(margin=MARGIN)
        event["result"] = result  # Retain the returned complete result before assertions.
        assert result.source_kind == "deformable-node-device-v1"
        assert result.state_epoch == batch.revision
        np.testing.assert_array_equal(result.ticks, [before[w].tick for w in batch.world_keys])
        assert result.source["producer_binding_sha256"] == batch.binding_sha256
        by_world = {s.key: s for s in specs}
        shape_map = {s.key: s for s in model.shapes}
        expected_groups, expected_pairs = declared_membership(model)
        event["expected_groups"] = expected_groups
        event["expected_pairs"] = expected_pairs
        event["independent_rows"] = []
        assert tuple(group.surface for group in result.groups) == tuple(expected_groups)
        assert tuple((pair.first, pair.second) for pair in result.pairs) == tuple(
            (a, b) for a, b, *_ in expected_pairs
        )
        wanted_groups = {}
        total = 0
        for group in result.groups:
            spec = by_world[group.surface.world]
            surface = next(
                s for s in spec.surfaces if s.key == shape_map[group.surface].surface_key
            )
            ids = tuple(map(int, spec.point_ids))
            current = before[spec.key].position.astype(np.float64)
            faces = tuple(
                tuple(ids.index(int(v)) for v in face) for face in surface.oriented_point_ids
            )
            expected_rounds = expected_groups[group.surface]
            assert group.round_keys == expected_rounds
            assert group.round_vertex_ids == tuple(
                shape_map[key].vertex_ids for key in expected_rounds
            )
            axes = [reference().axis(rounds[key]) for key in expected_rounds]
            radii = [shape_map[key].radius for key in expected_rounds]
            wanted = reference().aabb_candidates(current, faces, axes, radii, MARGIN)
            wanted_groups[group.surface] = wanted
            assert tuple(map(tuple, group.candidates.tolist())) == wanted
            if len(surface.facet_ids) == 2 and before[spec.key].tick == 0:
                for segment, key in enumerate(expected_rounds):
                    if shape_map[key].kind == "capsule":
                        assert tuple(f for s, f in wanted if s == segment) == (0, 1)
            declaration = result.source["surfaces"][group.surface]
            np.testing.assert_array_equal(declaration["facet_ids"], surface.facet_ids)
            np.testing.assert_array_equal(
                declaration["oriented_point_ids"], surface.oriented_point_ids
            )
            assert tuple(map(tuple, group.facet_vertices)) == tuple(
                map(tuple, surface.oriented_point_ids)
            )
            for i, (row, (segment, facet)) in enumerate(
                zip(group.packet.rows, wanted, strict=True)
            ):
                assert (row.segment_id, row.facet_id) == (segment, facet)
                assert tuple(row.vertex_ids) == tuple(surface.oriented_point_ids[facet])
                # Packet facet IDs are local ordinals; the source maps them to actual IDs.
                assert int(declaration["facet_ids"][row.facet_id]) == int(surface.facet_ids[facet])
                allow_conditioning = (
                    len(surface.facet_ids) == 2
                    and shape_map[expected_rounds[segment]].kind == "capsule"
                )
                assert_d2_row(
                    row,
                    current[list(faces[facet])],
                    axes[segment],
                    radii[segment],
                    group.packet,
                    i,
                    allow_conditioning=allow_conditioning,
                    initial=before[spec.key].tick == 0,
                    evidence=event["independent_rows"],
                )
                assert group.packet.raw["candidates"].shape[1:] == (18, 11)
                assert group.packet.packed["owners"][i].tolist() == [segment, facet]
            total += len(wanted)
        for pair, (a, b, status, reason, surface_key, round_key) in zip(
            result.pairs, expected_pairs, strict=True
        ):
            assert (pair.first, pair.second, pair.reason) == (a, b, reason)
            if status is not None:
                assert pair.status == status and pair.group is None and pair.segment_id is None
                assert pair.facet_ids == pair.row_indices == ()
            else:
                group_index = tuple(expected_groups).index(surface_key)
                segment = expected_groups[surface_key].index(round_key)
                indices = tuple(
                    i for i, (s, _) in enumerate(wanted_groups[surface_key]) if s == segment
                )
                facets = tuple(wanted_groups[surface_key][i][1] for i in indices)
                assert pair.group == group_index and pair.segment_id == segment
                assert pair.row_indices == indices and pair.facet_ids == facets
                assert pair.status == (
                    "boundary_candidates" if indices else "no_boundary_candidate"
                )
        assert result.required_candidates == total
        plan.require_current(result)
        assert plan.stats()["live_bytes"] == plan.stats()["persistent_bytes"]
        assert plan.stats()["native_owned_limit"] == plan.stats()["memory_limit_bytes"]
        assert result.source["material_delta_host_bytes"] == 24 * len(
            result.source["stamps"]["material_reconstruction_delta"]
        )
        return result
    finally:
        event.update(after=batch.snapshot(), attempt=plan.last_attempt, stats=plan.stats())
        assert batch.revision == event["revision"]
        for key, state in before.items():
            assert event["after"][key] is state


@pytest.mark.parametrize("square,fixed", [(False, False), (True, False), (True, True)])
def test_twelve_real_deformations(square, fixed, tyx_mode, tmp_path):
    specs = (world(square=square, fixed=fixed),)
    model, bindings, rounds = scene(specs)
    with TyxDeformableBatch(specs, mode=tyx_mode, settings=SETTINGS) as batch:
        with closing(batch.prepare_boundary_query(model, bindings)) as plan:
            with saved(tmp_path, specs, model, bindings) as events:
                initial = batch.snapshot()[7].position.copy()
                first = check(batch, plan, specs, model, rounds, events)
                initial_bounds = first.groups[0].leaf_bounds.copy()
                deformation, bounds_change = [], []
                for tick in range(12):
                    request = forces(specs, tick)
                    receipt = batch.step(request, H, expected_revision=batch.revision)
                    events.append(dict(request=request, receipt=receipt))
                    assert receipt.accepted
                    result = check(batch, plan, specs, model, rounds, events)
                    current = batch.snapshot()[7].position
                    edges = specs[0].constraints.indices
                    deformation.append(
                        float(
                            np.max(
                                np.abs(
                                    np.linalg.norm(
                                        current[edges[:, 1]] - current[edges[:, 0]], axis=1
                                    )
                                    - np.linalg.norm(
                                        initial[edges[:, 1]] - initial[edges[:, 0]], axis=1
                                    )
                                )
                            )
                        )
                    )
                    bounds_change.append(
                        float(np.max(np.abs(result.groups[0].leaf_bounds - initial_bounds)))
                    )
                    if fixed:
                        np.testing.assert_array_equal(current[0], initial[0])
                assert max(deformation) > 1e-6
                assert max(bounds_change) > 1e-6


@pytest.mark.parametrize("reverse", [False, True])
def test_two_worlds_identity_restore_and_successor(reverse, tyx_mode, tmp_path):
    specs = (world(7, reverse=reverse), world(31, square=True, fixed=True, reverse=reverse))
    if reverse:
        specs = specs[::-1]
    model, bindings, rounds = scene(specs, reverse=reverse)
    with TyxDeformableBatch(specs, mode=tyx_mode, settings=SETTINGS) as batch:
        with closing(batch.prepare_boundary_query(model, bindings)) as plan:
            with saved(tmp_path, specs, model, bindings) as events:
                capsule = batch.checkpoint()
                first = check(batch, plan, specs, model, rounds, events)
                for tick in range(12):
                    receipt = batch.step(forces(specs, tick), H, expected_revision=batch.revision)
                    events.append(dict(receipt=receipt))
                    check(batch, plan, specs, model, rounds, events)
                with pytest.raises(ValueError):
                    plan.require_current(first)
                batch.restore(capsule, expected_revision=batch.revision)
                restored = check(batch, plan, specs, model, rounds, events)
                for old, new in zip(first.groups, restored.groups, strict=True):
                    np.testing.assert_array_equal(
                        old.packet.raw["summary"], new.packet.raw["summary"]
                    )
                receipt = batch.step(forces(specs), H, expected_revision=batch.revision)
                events.append(dict(receipt=receipt))
                check(batch, plan, specs, model, rounds, events)
                other = batch.snapshot()[31]
                batch.reset((7,), expected_revision=batch.revision)
                assert batch.snapshot()[31] is other
                check(batch, plan, specs, model, rounds, events)
                receipt = batch.step(forces(specs, 1), H, expected_revision=batch.revision)
                events.append(dict(receipt=receipt))
                check(batch, plan, specs, model, rounds, events)


@pytest.mark.parametrize("masked,excluded", [(True, False), (False, True)])
def test_filters_preserve_complete_pair_membership(masked, excluded, tyx_mode, tmp_path):
    specs = (world(square=True),)
    model, bindings, rounds = scene(specs, masked=masked, excluded=excluded)
    with TyxDeformableBatch(specs, mode=tyx_mode, settings=SETTINGS) as batch:
        with closing(batch.prepare_boundary_query(model, bindings)) as plan:
            with saved(tmp_path, specs, model, bindings) as events:
                value = check(batch, plan, specs, model, rounds, events)
                if masked:
                    assert value.required_candidates == 0 and not value.groups
                else:
                    assert all(key.shape != 1 for group in value.groups for key in group.round_keys)


def test_empty_all_world_ticks_and_no_allocations(tyx_mode, tmp_path):
    specs = (world(), world(31, square=True))
    with TyxDeformableBatch(specs, mode=tyx_mode, settings=SETTINGS) as batch:
        receipt = batch.step(forces(specs), H, expected_revision=batch.revision)
        model = DeformableBoundaryModel(())
        with closing(batch.prepare_boundary_query(model)) as plan:
            with saved(tmp_path, specs, model, ()) as events:
                events.append(dict(receipt=receipt))
                value = check(batch, plan, specs, model, {}, events)
                assert value.world_keys == batch.world_keys and tuple(value.ticks) == (1, 1)
                assert plan.stats()["allocations"] == 0
                assert plan.stats()["host_downloaded_bytes"] == 8
