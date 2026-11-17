"""Exact boundary-certificate attacks and complete-step mixed-path comparison."""

import json
from contextlib import ExitStack
from fractions import Fraction as F
from time import perf_counter

import numpy as np
import pytest
from test_mujoco_tyx_area_pressure_heterogeneous_scene import H, _declarations
from test_mujoco_tyx_area_pressure_lifecycle import _same

from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.backends.tyx_area_pressure_intersection import (
    TyxAreaPressureIntersector,
    _refine_polygon,
    _Unresolved,
)
from phymeo.geometry import _area_pressure_clipping as exact


def _certificate():
    face = exact.triangle(
        tuple(tuple(F(v) for v in row) for row in ((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    )
    coefficients = (
        (F(0), F(0), F(0)),
        (F(3, 4), F(-1, 4), F(3, 4)),
        (F(3, 4), F(3, 4), F(-1, 4)),
        (F(-1, 4), F(3, 4), F(3, 4)),
    )
    constraints = ((F(1), F(0), F(0)), (F(0), F(1), F(0)), (F(0), F(0), F(1)), *coefficients)
    vertices = (
        (F(1, 4), F(0)),
        (F(3, 4), F(0)),
        (F(3, 4), F(1, 4)),
        (F(1, 4), F(3, 4)),
        (F(0), F(3, 4)),
        (F(0), F(1, 4)),
    )
    active = np.array(
        [[int(exact.dot(c, (1 - u - v, u, v)) == 0) for c in constraints] for u, v in vertices],
        dtype=np.int32,
    )
    return face, coefficients, vertices, active


def test_exact_full_boundary_and_coplanar_constraint_cannot_certify_subset():
    face, coefficients, vertices, active = _certificate()
    polygon = _refine_polygon(coefficients, face, active)
    assert set(polygon) == {(u, v, F(0)) for u, v in vertices}
    # Every proposed corner is an actual feasible vertex. The common plane is
    # identically zero and must NOT be accepted as a supporting edge.
    with pytest.raises(_Unresolved, match="edge_has_no_support_line"):
        _refine_polygon(coefficients, face, active[[0, 2, 4]])


@pytest.mark.parametrize("order", [(0, 2, 4, 1, 3, 5), (5, 4, 3, 2, 1, 0), (0, 1, 1, 3, 4, 5)])
def test_exact_boundary_refuses_star_reverse_and_repeated_vertices(order):
    face, coefficients, _, active = _certificate()
    with pytest.raises(_Unresolved):
        _refine_polygon(coefficients, face, active[list(order)])


def test_exact_boundary_refuses_false_tight_constraint():
    face, coefficients, _, active = _certificate()
    active[0, 0] = 1
    with pytest.raises(_Unresolved):
        _refine_polygon(coefficients, face, active)


@pytest.mark.parametrize("mode", ("interpret", pytest.param("cuda", marks=pytest.mark.cuda)))
def test_heterogeneous_mixed_geometry_complete_steps_and_cost(mode, tmp_path):
    models, worlds, assets = _declarations()
    with ExitStack() as stack:
        intersector = stack.enter_context(TyxAreaPressureIntersector(mode=mode, capacity=8))
        batches, scenes, workspaces = {}, {}, {}
        for name in ("host", "mixed"):
            batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(models, worlds, mode=mode))
            batches[name] = batch
            scenes[name] = MuJoCoTyxHeterogeneousAreaPressureScene(
                batch, assets, intersector=intersector if name == "mixed" else None
            )
            workspaces[name] = stack.enter_context(
                batch.prepare_device_workspace(include_area_pressure=True)
            )
        timings, outputs = {"host": [], "mixed": []}, {}
        routes = []
        # Alternate order to avoid consistently favoring either warm cache.
        for step in range(12):
            order = ("host", "mixed") if step % 2 == 0 else ("mixed", "host")
            for name in order:
                batch, scene, workspace = batches[name], scenes[name], workspaces[name]
                start = perf_counter()
                scene.step(H, expected_revision=batch.checkpoint_revision)
                frame = workspace.refresh(expected_revision=batch.checkpoint_revision)
                outputs[name] = workspace.readback(frame)
                timings[name].append(perf_counter() - start)
            _same(batches["mixed"].integration_snapshot(), batches["host"].integration_snapshot())
            _same(scenes["mixed"].last_geometry, scenes["host"].last_geometry)
            _same(outputs["mixed"], outputs["host"])
            routes.append(intersector.last_batch_routes)
        stats = dict(intersector.stats)
        assert stats["attempts"] == 48 and stats["batch_calls"] == 12
        assert 0 < stats["launches"] <= 24
        assert stats["device_polygons"] > 0
        assert stats["candidate_pairs"] == 192
        assert stats["culled_pairs"] > 0
        assert stats["device_polygons"] + stats["fallback_pairs"] + stats["culled_pairs"] == 192
        assert intersector.last_routes
        # Geometry is stateless across new physical owners after selective reset.
        for name in ("host", "mixed"):
            scenes[name].reset((41, 7))
            assert not scenes[name].last_geometry
            scenes[name].step(H, expected_revision=batches[name].checkpoint_revision)
        _same(batches["mixed"].integration_snapshot(), batches["host"].integration_snapshot())
        _same(scenes["mixed"].last_geometry, scenes["host"].last_geometry)
        payload = dict(
            schema="area-batched-candidate-cost-v1",
            mode=mode,
            steps=12,
            h=H,
            worlds=5,
            buckets=3,
            complete_step_seconds=timings,
            first_step_seconds={k: v[0] for k, v in timings.items()},
            warm_total_seconds={k: sum(v[1:]) for k, v in timings.items()},
            intersector_after_12_steps=stats,
            last_routes_by_step=routes,
            note="Includes FK, validation, topology, transfers, exact refinement/fallback, physics and unified readback; no speed threshold.",
        )
        (tmp_path / "complete-step-cost.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


@pytest.mark.parametrize("mode", ("interpret", pytest.param("cuda", marks=pytest.mark.cuda)))
def test_single_scene_clone_shares_caller_owned_intersector(mode):
    from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime
    from phymeo.backends.mujoco_area_pressure_scene import MuJoCoTyxAreaPressureScene

    models, worlds, assets = _declarations()
    spec = next(m for m in models if m.key == "soft")
    world = next(w for w in worlds if w.world == 41)
    with ExitStack() as stack:
        query = stack.enter_context(TyxAreaPressureIntersector(mode=mode, capacity=8))
        runtime = stack.enter_context(
            MuJoCoTyxRuntime(
                spec.compilation,
                mode=mode,
                world_keys=(41,),
                initial_states={41: world.initial_state},
                contact_capacity=4,
                constraint_capacity=32,
                area_pressure_pairs=spec.area_pressure_pairs,
            )
        )
        scene = MuJoCoTyxAreaPressureScene(runtime, assets["soft"], intersector=query)
        scene.step(H, expected_revision=runtime.checkpoint_revision)
        clone = scene.clone()
        stack.enter_context(clone.runtime)
        assert clone._intersector is query
        for current in (scene, clone):
            current.step(H, expected_revision=current.runtime.checkpoint_revision)
        _same(scene.last_geometry, clone.last_geometry)
        _same(runtime.integration_snapshot(), clone.runtime.integration_snapshot())
        assert query.stats["attempts"] == 3
        query.close()
        with pytest.raises(ValueError, match="open"):
            MuJoCoTyxAreaPressureScene(runtime, assets["soft"], intersector=query)


@pytest.mark.parametrize("mode", ("interpret", pytest.param("cuda", marks=pytest.mark.cuda)))
def test_last_world_query_failure_preserves_batch_and_geometry(mode, monkeypatch):
    from test_mujoco_tyx_area_pressure_heterogeneous_scene import _make, _refresh, _retained

    with ExitStack() as stack:
        query = stack.enter_context(TyxAreaPressureIntersector(mode=mode, capacity=8))
        batch, _, workspace, _, _, assets = _make(stack, mode)
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets, intersector=query)
        scene.step(H, expected_revision=batch.checkpoint_revision)
        frame, arrays = _refresh(batch, workspace)
        state, geometry = batch.integration_snapshot(), scene.last_geometry
        capsule, revision = batch.checkpoint().to_bytes(), batch.checkpoint_revision
        original, seen = query._publish_query, []

        def fail_last(*args):
            output = original(*args)
            seen.append(output.surface.stamp.world_id)
            if len(seen) == 4:
                raise RuntimeError("last world query publication failure")
            return output

        with monkeypatch.context() as patch:
            patch.setattr(query, "_publish_query", fail_last)
            with pytest.raises(RuntimeError, match="last world query"):
                scene.step(H, expected_revision=revision)
        assert len(seen) == 4 and set(seen) == {"41", "11", "305", "91"}
        _retained(batch, scene, workspace, state, geometry, frame, arrays, capsule, revision)
        scene.step(H, expected_revision=revision)
        assert all(s.tick == state[k].tick + 1 for k, s in batch.integration_snapshot().items())


@pytest.mark.parametrize("mode", ("interpret", pytest.param("cuda", marks=pytest.mark.cuda)))
def test_scene_admission_requires_open_not_merely_unclosed_workspace(mode, monkeypatch):
    from types import SimpleNamespace

    from phymeo.backends.mujoco_area_pressure_scene import MuJoCoTyxAreaPressureScene

    models, worlds, assets = _declarations()
    with ExitStack() as stack:
        query = stack.enter_context(TyxAreaPressureIntersector(mode=mode, capacity=8))
        batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(models, worlds, mode=mode))
        leaf = batch._buckets[batch.layout[41].bucket]
        before = batch.checkpoint().to_bytes()
        # Simulate resource-ledger state only; never inject a real native free
        # failure or claim this establishes recovery from one.
        for state in ("UNKNOWN_RELEASE", "CLOSING", "CLOSED"):
            with monkeypatch.context() as patch:
                patch.setattr(query, "_resources", SimpleNamespace(state=state, closed=False))
                with pytest.raises(ValueError, match="open"):
                    MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets, intersector=query)
                with pytest.raises(ValueError, match="open"):
                    MuJoCoTyxAreaPressureScene(leaf, assets["soft"], intersector=query)
            assert batch.checkpoint().to_bytes() == before
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets, intersector=query)
        scene.step(H, expected_revision=batch.checkpoint_revision)
        assert scene.last_geometry
