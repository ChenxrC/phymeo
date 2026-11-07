"""C5 accepted full-affine material -> resident current-pose broad phase.

These are not CCD/contact-response tests. UNEXECUTED by the author.
"""

import copy
from dataclasses import replace
from itertools import combinations, product
from math import pi

import numpy as np
import pytest
from test_resident_affine_step import (
    WINDOW_ATOL,
    H,
    _action,
    _close,
    _json,
    _load,
    _ref,
    _save,
    _world,
)

import phymeo as ph
from phymeo.backends import tyx_world_batch as resident

IDS = {11: (101,), 12: (201, 202), 13: tuple(range(301, 309)), 14: (401, 402, 403, 404, 405)}


def _box(lo, hi):
    return np.asarray(list(product(*zip(lo, hi, strict=True))), float)


def _materials():
    R = _ref().rotation([0, 0, 1], pi / 4)
    return (
        (0, 11, "sphere", np.array([[0.25, -0.125, 0.0]]), 1 / 16),
        (1, 12, "capsule", np.array([[-0.5, 0, 0], [0.5, 0, 0]]), 1 / 32),
        (1, 13, "hull", _box([-0.25, -0.125, -0.0625], [0.25, 0.125, 0.0625]) @ R.T, 0.0),
        (
            0,
            14,
            "hull",
            np.array(
                [
                    [-0.25, -0.125, -0.0625],
                    [-0.25, 0.125, -0.0625],
                    [0.25, -0.125, -0.0625],
                    [0.25, 0.125, -0.0625],
                    [0, 0, 0.25],
                ]
            ),
            0.0,
        ),
    )


def _vertices(data, body, local, full_q):
    truth = _ref().oracle(data["model"], full_q, np.zeros(len(full_q)))
    return truth["position"][body] + np.asarray(local) @ truth["rotation"][body].T


def _bounds(points, radius):
    return np.asarray(points).min(0) - radius, np.asarray(points).max(0) + radius


def _contains(result, key, lo, hi):
    row = result.bounds[result.shape_keys.index(key)]
    assert np.isfinite(row).all()
    assert np.all(row[:3] <= lo) and np.all(row[3:] >= hi)


def _result(result):
    return {
        name: _json(getattr(result, name))
        for name in (
            "topology_generation",
            "state_epoch",
            "world_keys",
            "ticks",
            "shape_keys",
            "shape_revisions",
            "bounds",
            "registered_pairs",
            "candidates",
        )
    }


def _scene(
    asset, data, checkpoint, *, world=7, materials=None, wall_lo=(20, 20, 20), wall_hi=(21, 21, 21)
):
    shapes, bindings, refs = [], [], {}
    for body, shape_id, kind, local, radius in _materials() if materials is None else materials:
        site = asset.tree.sites[data["model"]["names"][body]]
        group = site.link_index
        group_local = (
            np.asarray(site.local_position_m)
            + local @ _ref().quaternion_matrix(site.local_orientation_xyzw).T
        )
        key = ph.CollisionKey(world, 100 + group, shape_id)
        shapes.append(ph.CollisionShape(key, kind, IDS[shape_id], radius, revision=7))
        bindings.append(
            ph.ResidentAffineTreeBinding(
                key, world, group, group_local, topology=checkpoint.topology
            )
        )
        refs[key] = (body, local.copy(), radius)
    wall = ph.CollisionKey(world, 900, 1)
    shapes.append(ph.CollisionShape(wall, "hull", tuple(range(8)), revision=3))
    bindings.append(ph.ResidentStaticBinding(wall, ph.CollisionMotion(_box(wall_lo, wall_hi))))
    return (
        ph.CollisionModel(tuple(shapes), tuple(combinations(refs, 2))),
        tuple(bindings),
        refs,
        wall,
    )


@pytest.mark.tyx
@pytest.mark.parametrize("s", (-256 * pi, -2 * pi, 0.0, 0.37, 2 * pi, 256 * pi))
@pytest.mark.parametrize("placement", ("base", "cycle", "large"))
def test_a08_all_materials_use_full_accepted_mimic_pose(s, placement, tyx_mode, tmp_path):
    data = _ref().transformed(_ref().case("A04"), placement)
    data["s"][:], data["u"][:] = s, 0
    data["lower"][:], data["upper"][:] = -4096, 4096
    asset = _load(tmp_path, data)
    with resident.TyxWorldBatch([(7, _world(asset, data))], tyx_mode, cache_root=tmp_path) as batch:
        checkpoint = batch.snapshot()[7]
        center = np.array([4096, -2048, 1024]) if placement == "large" else np.zeros(3)
        model, bindings, refs, _ = _scene(
            asset, data, checkpoint, wall_lo=center + 20, wall_hi=center + 21
        )
        with batch.prepare_collision(model, bindings) as plan:
            result = plan.query()
            for key, (body, local, radius) in refs.items():
                _contains(
                    result,
                    key,
                    *_bounds(_vertices(data, body, local, checkpoint.full_position), radius),
                )
                assert (
                    next(shape for shape in model.shapes if shape.key == key).vertex_ids
                    == IDS[key.shape]
                )
                assert result.shape_revisions[result.shape_keys.index(key)] == 7
            assert result.candidates == () and result.ticks.tolist() == [0]
            assert batch.snapshot()[7] == checkpoint
            _save(
                tmp_path,
                "all-shapes",
                dict(
                    input=data,
                    checkpoint=checkpoint,
                    materials=refs,
                    model=model,
                    result=_result(result),
                    stats=plan.stats(),
                ),
            )


@pytest.mark.tyx
@pytest.mark.parametrize("gap", (-1 / 65536, 0.25))
def test_a08_predeclared_current_box_near_touch_and_clear_miss(gap, tyx_mode, tmp_path):
    data = _ref().case("A04")
    data["s"][:] = 0.37
    asset = _load(tmp_path, data)
    with resident.TyxWorldBatch([(7, _world(asset, data))], tyx_mode, cache_root=tmp_path) as batch:
        state = batch.snapshot()[7]
        chosen = _materials()[2]
        lo, hi = _bounds(_vertices(data, chosen[0], chosen[3], state.full_position), 0)
        wall_lo, wall_hi = lo.copy() - 0.25, hi.copy() + 0.25
        wall_lo[0], wall_hi[0] = hi[0] + gap, hi[0] + gap + 0.125
        model, bindings, refs, wall = _scene(
            asset, data, state, materials=(chosen,), wall_lo=wall_lo, wall_hi=wall_hi
        )
        with batch.prepare_collision(model, bindings) as plan:
            result = plan.query()
            key = next(iter(refs))
            _contains(result, key, lo, hi)
            assert result.candidates == (((key, wall),) if gap < 0 else ())
            _save(
                tmp_path,
                "near-miss",
                dict(
                    gap_m=gap,
                    state=state,
                    materials=refs,
                    oracle_bounds=[lo, hi],
                    result=_result(result),
                ),
            )


@pytest.mark.tyx
@pytest.mark.parametrize("zone", ("axis", "end_skin", "beyond_end"))
def test_a08_capsule_axis_is_finite_and_keeps_both_skin_ends(zone, tyx_mode, tmp_path):
    data = _ref().case("A04")
    data["u"][:] = 0
    asset = _load(tmp_path, data)
    with resident.TyxWorldBatch([(7, _world(asset, data))], tyx_mode, cache_root=tmp_path) as batch:
        state = batch.snapshot()[7]
        local, radius = np.array([[-8.0, 0, 0], [8.0, 0, 0]]), 1 / 1024
        points = _vertices(data, 1, local, state.full_position)
        direction = (points[1] - points[0]) / 16
        center = (
            points.mean(0)
            if zone == "axis"
            else points[1] + direction * (radius if zone == "end_skin" else 0.25)
        )
        model, bindings, refs, wall = _scene(
            asset,
            data,
            state,
            materials=((1, 12, "capsule", local, radius),),
            wall_lo=center - radius / 2,
            wall_hi=center + radius / 2,
        )
        with batch.prepare_collision(model, bindings) as plan:
            result = plan.query()
            key = next(iter(refs))
            _contains(result, key, *_bounds(points, radius))
            assert result.candidates == (() if zone == "beyond_end" else ((key, wall),))
            _save(
                tmp_path,
                "finite-capsule",
                dict(
                    zone=zone,
                    state=state,
                    original_points=local,
                    oracle_world=points,
                    result=_result(result),
                ),
            )


@pytest.mark.tyx
@pytest.mark.parametrize("name", ("A03", "A05T"))
def test_a05_original_asset_sites_and_fixed_tool_materials_survive_projection(
    name, tyx_mode, tmp_path
):
    data = _ref().case(name)
    asset = _load(tmp_path, data)
    world = _world(asset, data)
    with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: _action(world, asset, data)})
        shapes, bindings, refs = [], [], {}
        for i, col in enumerate(asset.tree.collisions):
            site = asset.tree.sites[col.link_name]
            assert site.link_index is not None
            key = ph.CollisionKey(7, 100 + site.link_index, i + 1)
            shapes.append(replace(col.shape, key=key))
            bindings.append(
                ph.ResidentAffineTreeBinding.from_asset(asset, col.shape.key, key=key, world=7)
            )
            body = data["model"]["names"].index(col.link_name)
            # Fixed URDF values, independent of importer output.
            local = (
                _box([-0.25, -0.125, -0.0625], [0.25, 0.125, 0.0625])
                @ _ref().rotation([0, 0, 1], pi / 4).T
                + [0.25, -0.125, 0]
                if col.link_name == "tool"
                else np.array([[0.25, -0.125, 0]])
            )
            refs[key] = (body, local, 0 if col.link_name == "tool" else 1 / 16)
        moving_keys = tuple(refs)
        wall = ph.CollisionKey(7, 900, 1)
        shapes.append(ph.CollisionShape(wall, "hull", tuple(range(8))))
        bindings.append(
            ph.ResidentStaticBinding(wall, ph.CollisionMotion(_box([20] * 3, [21] * 3)))
        )
        model = ph.CollisionModel(tuple(shapes), tuple(combinations(moving_keys, 2)))
        trace = []
        with batch.prepare_collision(model, tuple(bindings)) as plan:
            for tick in range(1, 4):
                assert batch.step(H) == {7: True}
                state = batch.snapshot()[7]
                result = plan.query()
                for key, (body, local, radius) in refs.items():
                    _contains(
                        result,
                        key,
                        *_bounds(_vertices(data, body, local, state.full_position), radius),
                    )
                trace.append(dict(state=state, result=_result(result), stats=plan.stats()))
            assert result.candidates == ()
        _save(
            tmp_path,
            "asset-source",
            dict(
                input=data,
                source_urdf=(tmp_path / "mechanism.urdf").read_text(),
                original_material=refs,
                trace=trace,
            ),
        )


@pytest.mark.tyx
@pytest.mark.parametrize(
    "fault",
    ("wrong_graph", "wrong_model", "revision", "vertex_ids", "body_alias", "missing_exclusion"),
)
def test_a07_affine_graph_owner_and_source_identity_rejected_before_allocation(
    fault, tyx_mode, tmp_path
):
    data = _ref().case("A03")
    asset = _load(tmp_path, data)
    with resident.TyxWorldBatch([(7, _world(asset, data))], tyx_mode, cache_root=tmp_path) as batch:
        col = asset.tree.collisions[0]
        key = ph.CollisionKey(7, 100, 1)
        source = asset
        if fault == "wrong_graph":
            source = _load(tmp_path / "flat", _ref().case("A03_flat"))
        elif fault == "wrong_model":
            different = copy.deepcopy(data)
            different["model"]["mass"][0] = 3
            source = _load(tmp_path / "mass", different)
        binding = ph.ResidentAffineTreeBinding.from_asset(
            source, source.tree.collisions[0].shape.key, key=key, world=7
        )
        shape = replace(col.shape, key=key)
        if fault == "revision":
            shape = replace(shape, revision=shape.revision + 1)
        if fault == "vertex_ids":
            shape = replace(shape, vertex_ids=(777,))
        shapes, bindings, excluded = [shape], [binding], ()
        if fault in ("body_alias", "missing_exclusion"):
            col2 = asset.tree.collisions[1]
            key2 = ph.CollisionKey(7, 100 if fault == "body_alias" else 101, 2)
            shapes.append(replace(col2.shape, key=key2))
            bindings.append(
                ph.ResidentAffineTreeBinding.from_asset(asset, col2.shape.key, key=key2, world=7)
            )
            if fault == "body_alias":
                excluded = ((key, key2),)
        before = batch.stats()["device_allocations"]
        with pytest.raises(
            (ValueError, TypeError),
            match="binding|group|body|source|shape|revision|topology|model|graph|mimic|pair|dynamic|self",
        ):
            batch.prepare_collision(ph.CollisionModel(tuple(shapes), excluded), tuple(bindings))
        assert batch.stats()["device_allocations"] == before
        assert batch.stats()["collision_plan_live_count"] == 0


def _matrix(count, path):
    kinds = (
        ("A04",)
        if count == 1
        else ("sphere", "A01", "A03", "A05")
        if count == 4
        else ("sphere", "slider", "rotor", "ordinary", "A02", "A03", "N9", "N16") * 4
    )
    worlds, controls, rows = [], {}, {}
    for i, kind in enumerate(kinds):
        key = 7 + 13 * i
        if kind == "sphere":
            spec = resident.FreeSphere(2, 1 / 16, gravity=(0, 0, 0))
            action, data, asset = (1 / 4, 0, 0), None, None
        elif kind in ("slider", "rotor", "ordinary"):
            if kind == "ordinary":
                data = _ref().case("A04")
                m = data["model"]
            else:
                m = _ref()._model(
                    [-1],
                    ["prismatic" if kind == "slider" else "revolute"],
                    [[0, 0, 0]],
                    [[1, 0, 0] if kind == "slider" else [0, 0, 1]],
                    [2],
                    [[0, 0, 0] if kind == "slider" else [0.25, 0, 0]],
                    [[1 / 8] * 3],
                )
                data = dict(model=m)
            n = len(m["mass"])
            model = ph.RobotModel(
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
            program = ph.DrivenRobotProgram(
                model, kp=np.zeros(n), kd=np.zeros(n), effort_limit=np.full(n, 16.0)
            )
            initial = program.initial_state(ph.RobotState(np.full(n, 1 / 16), np.full(n, 1 / 32)))
            spec = (
                resident.PrismaticSlider
                if kind == "slider"
                else resident.RevoluteRotor
                if kind == "rotor"
                else resident.FixedRobotTree
            )(program, initial)
            action = program.inputs(
                np.zeros(n),
                loads=ph.RobotInputs(np.full(n, 1 / 4), np.zeros((n, 3)), np.zeros((n, 3))),
            )
            asset = None
        else:
            data = _ref().case(kind)
            asset = _load(path / str(key), data)
            spec = _world(asset, data)
            action = _action(spec, asset, data)
        worlds.append((key, spec))
        controls[key] = action
        rows[key] = dict(kind=kind, data=data, asset=asset, near=i % 2 == 0)
    return list(reversed(worlds)), controls, rows


def _matrix_scene(batch, rows):
    initial, shapes, bindings, refs = batch.snapshot(), [], [], {}
    for world, row in rows.items():
        state = initial[world]
        moving, wall = ph.CollisionKey(world, 100, 11), ph.CollisionKey(world, 900, 1)
        kind, radius = "sphere", 1 / 16
        local, ids = np.array([[0.25, -0.125, 0.0]]), (101,)
        if row["kind"] in ("A03", "rotor"):
            kind, radius, ids = "capsule", 1 / 32, (201, 202)
            local = np.array([[-0.5, 0, 0], [0.5, 0, 0]])
        elif row["kind"] in ("A02", "N16", "ordinary"):
            kind, radius, ids = "hull", 0.0, tuple(range(301, 309))
            local = (
                _box([-0.25, -0.125, -0.0625], [0.25, 0.125, 0.0625])
                @ _ref().rotation([0, 0, 1], pi / 4).T
            )
        if row["kind"] == "sphere":
            local = np.zeros((1, 3))
            points = np.asarray([state.position])
            bindings.append(ph.ResidentSphereBinding(moving, world))
            body = None
        else:
            data = row["data"]
            body = _ref().coordinates(data["model"])[-1]
            q = state.full_position if row["asset"] is not None else state.position
            points = _vertices(data, body, local, q)
            if row["asset"] is not None:
                asset = row["asset"]
                site = asset.tree.sites[data["model"]["names"][body]]
                group_local = (
                    np.asarray(site.local_position_m)
                    + local @ _ref().quaternion_matrix(site.local_orientation_xyzw).T
                )
                bindings.append(
                    ph.ResidentAffineTreeBinding(
                        moving, world, site.link_index, group_local, topology=state.topology
                    )
                )
            elif row["kind"] == "ordinary":
                bindings.append(
                    ph.ResidentTreeBinding(moving, world, body, local, topology=state.topology)
                )
            else:
                bound = (
                    ph.ResidentSliderBinding
                    if row["kind"] == "slider"
                    else ph.ResidentRevoluteBinding
                )
                zero = _vertices(data, 0, local, np.zeros(1))
                m = data["model"]
                bindings.append(
                    bound(
                        moving,
                        world,
                        ph.CollisionMotion(zero),
                        origin_m=m["offsets"][0],
                        orientation_xyzw=m["orientations"][0],
                        axis_local=m["axes"][0],
                    )
                )
        shapes.append(ph.CollisionShape(moving, kind, ids, radius, revision=7))
        center = points.mean(0) + (0 if row["near"] else 1000)
        fixed = _box(center - 4, center + 4)
        shapes.append(ph.CollisionShape(wall, "hull", tuple(range(8)), revision=3))
        bindings.append(ph.ResidentStaticBinding(wall, ph.CollisionMotion(fixed)))
        refs[world] = dict(
            moving=moving, wall=wall, body=body, local=local, radius=radius, fixed=fixed
        )
    return ph.CollisionModel(tuple(reversed(shapes))), tuple(reversed(bindings)), refs, initial


def _check_matrix(result, rows, refs, states):
    hits = set()
    for world, row in rows.items():
        ref, state = refs[world], states[world]
        q = state.full_position if row["asset"] is not None else state.position
        points = (
            np.asarray([q])
            if row["kind"] == "sphere"
            else _vertices(row["data"], ref["body"], ref["local"], q)
        )
        _contains(result, ref["moving"], *_bounds(points, ref["radius"]))
        _contains(result, ref["wall"], *_bounds(ref["fixed"], 0))
        if row["near"]:
            hits.add((ref["moving"], ref["wall"]))
    assert set(result.candidates) == hits
    assert len(result.registered_pairs) == len(rows)
    assert all(first.world == second.world for first, second in result.registered_pairs)


@pytest.mark.tyx
@pytest.mark.parametrize("count", (1, 4, 32))
def test_a08_complete_heterogeneous_forty_step_output(count, tyx_mode, tmp_path):
    worlds, actions, rows = _matrix(count, tmp_path)
    report = dict(
        h_s=H,
        inputs={key: value["data"] for key, value in rows.items()},
        commands=actions,
        trace=[],
    )
    with resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls(actions)
        model, bindings, refs, initial = _matrix_scene(batch, rows)
        report.update(initial=initial, model=model, materials=refs)
        affine_keys = [key for key, row in rows.items() if row["asset"] is not None]
        expected = {
            key: (np.asarray(initial[key].position), np.asarray(initial[key].velocity))
            for key in affine_keys
        }
        with batch.prepare_collision(model, bindings) as plan:
            try:
                for tick in range(1, 41):
                    accepted = batch.step(H)
                    actual, result = batch.snapshot(), plan.query()
                    read, diag = (
                        batch.tree_kinematics(affine_keys),
                        batch.tree_diagnostics(affine_keys),
                    )
                    report["trace"].append(
                        dict(
                            tick=tick,
                            accepted=accepted,
                            actual=actual,
                            kinematics=read,
                            diagnostic=diag,
                            result=_result(result),
                            batch_stats=batch.stats(),
                            plan_stats=plan.stats(),
                        )
                    )
                    assert all(accepted.values()) and result.ticks.tolist() == [tick] * count
                    _check_matrix(result, rows, refs, actual)
                    for key in affine_keys:
                        s, u, _ = _ref().advance(rows[key]["data"], *expected[key])
                        expected[key] = s, u
                        _close(actual[key].position, s, atol=WINDOW_ATOL)
                        _close(actual[key].velocity, u, atol=WINDOW_ATOL)
                    report["trace"][-1]["independent_affine_s_u"] = copy.deepcopy(expected)
                assert {frame["batch_stats"]["current_buffer"] for frame in report["trace"]} == {
                    0,
                    1,
                }
            finally:
                _save(tmp_path, f"full-w{count}", report)


@pytest.mark.tyx
@pytest.mark.parametrize("count", (1, 4, 32))
def test_a08_warm_no_snapshot_envelope_upload_or_device_allocation(
    count, tyx_mode, tmp_path, monkeypatch
):
    worlds, actions, rows = _matrix(count, tmp_path)
    with resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls(actions)
        model, bindings, refs, initial = _matrix_scene(batch, rows)
        with batch.prepare_collision(model, bindings) as plan:
            for _ in range(2):
                assert all(batch.step(H).values())
                plan.query()
            entry, before, plan_before = batch.snapshot(), batch.stats(), plan.stats()
            old, old_bounds = plan.last_result, plan.last_result.bounds.copy()
            report = dict(
                input={key: row["data"] for key, row in rows.items()},
                commands=actions,
                initial=initial,
                actual_warm_entry=entry,
                materials=refs,
                batch_before=before,
                plan_before=plan_before,
                trace=[],
            )

            def forbidden(*args, **kwargs):
                raise AssertionError(
                    "warm path used HOST current state/envelope or new device allocation"
                )

            try:
                with monkeypatch.context() as patch:
                    for method in (
                        "snapshot",
                        "kinematics",
                        "tree_kinematics",
                        "tree_diagnostics",
                        "_buffer",
                    ):
                        patch.setattr(batch, method, forbidden)
                    patch.setattr(plan._engine, "query", forbidden)
                    if tyx_mode == "cuda":
                        import tyx as tx

                        patch.setattr(tx, "cuda_array", forbidden)
                        patch.setattr(tx, "cuda_empty", forbidden)
                    for tick in range(3, 43):
                        accepted = batch.step(H)
                        result = plan.query()
                        report["trace"].append(
                            dict(
                                tick=tick,
                                accepted=accepted,
                                result=_result(result),
                                batch_stats=batch.stats(),
                                plan_stats=plan.stats(),
                            )
                        )
                        assert all(accepted.values()) and result.ticks.tolist() == [tick] * count
                        expected = {
                            (refs[key]["moving"], refs[key]["wall"])
                            for key, row in rows.items()
                            if row["near"]
                        }
                        assert set(result.candidates) == expected
                actual_end = batch.snapshot()
                report["actual_warm_end"] = actual_end
                _check_matrix(result, rows, refs, actual_end)
                after, plan_after = (
                    report["trace"][-1]["batch_stats"],
                    report["trace"][-1]["plan_stats"],
                )
                assert after["device_allocations"] == before["device_allocations"]
                assert plan_after["device_allocations"] == plan_before["device_allocations"]
                assert plan_after["dynamic_envelope_uploaded_bytes"] == 0
                for name in (
                    "snapshot_downloaded_bytes",
                    "kinematics_downloaded_bytes",
                    "tree_diagnostic_reads",
                    "tree_state_downloaded_bytes",
                ):
                    assert after[name] == before[name]
                for name in before:
                    if name.endswith("_downloaded_bytes") and any(
                        tag in name for tag in ("state", "snapshot", "kinematic")
                    ):
                        assert after[name] == before[name]
                assert plan_after["copied_to_host_bytes"] > plan_before["copied_to_host_bytes"]
                assert plan_after["state_buffer_directions"] == (0, 1)
                np.testing.assert_array_equal(old.bounds, old_bounds)
                for value in (result.bounds, result.ticks):
                    with pytest.raises(ValueError):
                        value.setflags(write=True)
            finally:
                report["final_stats"] = batch.stats()
                _save(tmp_path, f"warm-w{count}", report)


@pytest.mark.tyx
@pytest.mark.parametrize("warm_steps", (0, 1))
def test_a07_a08_epoch_owner_restore_reset_and_numeric_rejection(warm_steps, tyx_mode, tmp_path):
    data = _ref().case("A01")
    asset = _load(tmp_path, data)
    world = _world(asset, data)
    with resident.TyxWorldBatch(
        [(7, world), (23, resident.FreeSphere(2, 1 / 16, gravity=(0, 0, 0)))],
        tyx_mode,
        cache_root=tmp_path,
    ) as batch:
        batch.set_controls({7: _action(world, asset, data), 23: (1.0, 0.0, 0.0)})
        for _ in range(warm_steps):
            assert all(batch.step(H).values())
        checkpoint = batch.snapshot()
        model, bindings, _, _ = _scene(asset, data, checkpoint[7])
        with (
            batch.prepare_collision(model, bindings) as plan,
            batch.prepare_collision(model, bindings) as other,
        ):
            result = plan.query()
            with pytest.raises(ValueError):
                other.require_current(result)
            batch.restore(checkpoint)
            with pytest.raises(ValueError):
                plan.require_current(result)
            restored = plan.query()
            assert (
                restored.state_epoch > result.state_epoch
                and restored.ticks.tolist() == result.ticks.tolist()
            )
            bad = replace(
                checkpoint[7],
                position=(7 / 2,),
                velocity=(8.0,),
                full_position=(7 / 2, 57 / 8),
                full_velocity=(8.0, 16.0),
            )
            batch.restore({7: bad})
            before = batch.snapshot()
            accepted = batch.step(1 / 8)
            assert accepted == {7: False, 23: True}
            assert batch.snapshot()[7] == before[7]
            assert batch.snapshot()[23].tick == before[23].tick + 1
            assert plan.query().ticks.tolist() == [bad.tick]
            spectator = batch.snapshot()[23]
            batch.reset([7])
            assert batch.snapshot()[23] == spectator
            assert batch.snapshot()[7].tick == 0
            _save(
                tmp_path,
                "epoch",
                dict(
                    input=data,
                    checkpoint=checkpoint,
                    rejected=bad,
                    accepted=accepted,
                    final=batch.snapshot(),
                    result=_result(plan.query()),
                    stats=batch.stats(),
                ),
            )


@pytest.mark.tyx
def test_a08_capacity_and_failed_query_publish_no_partial_result(tyx_mode, tmp_path, monkeypatch):
    data = _ref().case("A01")
    asset = _load(tmp_path, data)
    with resident.TyxWorldBatch([(7, _world(asset, data))], tyx_mode, cache_root=tmp_path) as batch:
        state = batch.snapshot()[7]
        model, bindings, _, _ = _scene(asset, data, state, wall_lo=(-4, -4, -4), wall_hi=(4, 4, 4))
        with batch.prepare_collision(model, bindings, candidate_capacity=0) as small:
            with pytest.raises(ValueError, match="capacity"):
                small.query()
            assert small.last_result is None and batch.snapshot()[7] == state
        with batch.prepare_collision(model, bindings) as plan:
            old = plan.query()
            original = batch._launch

            def fail(*args, **kwargs):
                original(*args, **kwargs)
                raise RuntimeError("injected affine bounds failure")

            with monkeypatch.context() as patch:
                patch.setattr(batch, "_launch", fail)
                with pytest.raises(RuntimeError, match="bounds failure"):
                    plan.query()
            assert plan.last_result is old and batch.snapshot()[7] == state
            plan.require_current(old)
            _save(
                tmp_path,
                "query-failure",
                dict(actual=state, retained_result=_result(old), stats=plan.stats()),
            )


@pytest.mark.tyx
def test_a08_close_callback_failure_still_releases_other_plans_and_runtime(
    tyx_mode, tmp_path, monkeypatch
):
    data = _ref().case("A01")
    asset = _load(tmp_path, data)
    batch = resident.TyxWorldBatch([(7, _world(asset, data))], tyx_mode, cache_root=tmp_path)
    model, bindings, _, _ = _scene(asset, data, batch.snapshot()[7])
    first, second = (
        batch.prepare_collision(model, bindings),
        batch.prepare_collision(model, bindings),
    )
    original = first.close

    def close_then_raise():
        original()
        raise RuntimeError("injected close callback")

    monkeypatch.setattr(first, "close", close_then_raise)
    with pytest.raises(RuntimeError, match="close callback"):
        batch.close()
    for plan in (first, second):
        with pytest.raises(RuntimeError, match="closed"):
            plan.query()
    assert (
        batch.stats()["device_owned_bytes"] == 0 and batch.stats()["collision_plan_live_count"] == 0
    )
    _save(tmp_path, "closed", batch.stats())


@pytest.mark.tyx
def test_a08_empty_batch_and_empty_collision_plan_own_no_device_state(tyx_mode, tmp_path):
    with resident.TyxWorldBatch([], tyx_mode, cache_root=tmp_path) as batch:
        with batch.prepare_collision(ph.CollisionModel(()), ()) as plan:
            assert batch.step(H) == {} and batch.snapshot() == {}
            result = plan.query()
            assert (
                result.candidates == () and result.shape_keys == () and result.ticks.shape == (0,)
            )
            assert batch.stats()["device_allocations"] == 0
            _save(
                tmp_path,
                "empty",
                dict(result=_result(result), batch_stats=batch.stats(), plan_stats=plan.stats()),
            )


@pytest.mark.tyx
@pytest.mark.parametrize("direction", (-1, 1))
def test_a07_a08_exact_full_coordinate_domain_edge_and_candidate_rejection(
    direction, tyx_mode, tmp_path
):
    data = _ref().case("A01")
    data["lower"][:], data["upper"][:] = -4096, 4096
    data["s"][:] = (direction * 4096 - 1 / 8) / 2
    data["u"][:] = direction
    asset = _load(tmp_path, data)
    with resident.TyxWorldBatch([(7, _world(asset, data))], tyx_mode, cache_root=tmp_path) as batch:
        before = batch.snapshot()[7]
        assert before.full_position[1] == direction * 4096
        model, bindings, refs, _ = _scene(asset, data, before)
        with batch.prepare_collision(model, bindings) as plan:
            entry = plan.query()
            for key, (body, local, radius) in refs.items():
                _contains(
                    entry, key, *_bounds(_vertices(data, body, local, before.full_position), radius)
                )
            assert batch.step(H) == {7: False}
            assert batch.snapshot()[7] == before and batch.tree_diagnostics() == {7: None}
            after = plan.query()
            np.testing.assert_array_equal(after.bounds, entry.bounds)
            _save(
                tmp_path,
                "exact-domain-edge",
                dict(
                    input=data,
                    before=before,
                    entry=_result(entry),
                    after=_result(after),
                    stats=batch.stats(),
                ),
            )
