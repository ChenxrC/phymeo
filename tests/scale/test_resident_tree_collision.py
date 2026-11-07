"""T07/T08 current accepted full-tree FK -> resident finite-shape broad phase.

AUTHORING ONLY / UNEXECUTED. This is neither CCD nor a contact-response test.
Expected geometry uses independent authored-frame derivatives, never product FK.
"""

from dataclasses import replace
from itertools import combinations, product
from math import pi

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends import tyx_world_batch as resident
from phymeo.backends._resident_collision_plan import ResidentSphereBinding, ResidentStaticBinding

from test_resident_tree_step import (
    _action, _asset, _close, _json, _ordinary_model, _program, _reference, _save, _tree,
)


H = 1 / 1024
IDS = {11: (101,), 12: (201, 202), 13: tuple(range(301, 309)), 14: tuple(range(401, 406))}


def _api():
    from phymeo.backends._resident_tree_binding import ResidentTreeBinding
    return ResidentTreeBinding


def _box(low, high):
    return np.array(list(product(*zip(low, high, strict=True))), dtype=float)


def _materials():
    r = _reference().rotation([0, 0, 1], pi / 4)
    return (
        (0, 11, "sphere", np.array([[0.25, -0.125, 0.0]]), 1 / 16),
        (1, 12, "capsule", np.array([[-0.5, 0, 0], [0.5, 0, 0]]), 1 / 32),
        (1, 13, "hull", _box([-0.25, -0.125, -0.0625], [0.25, 0.125, 0.0625]) @ r.T, 0.0),
        (0, 14, "hull", np.array([[-0.25, -0.125, -0.0625], [-0.25, 0.125, -0.0625],
            [0.25, -0.125, -0.0625], [0.25, 0.125, -0.0625], [0, 0, 0.25]]), 0.0),
    )


def _world_vertices(data, group, local, q):
    pose = _reference().oracle(data, q, np.zeros(len(q)))
    return pose["position"][group] + np.asarray(local) @ pose["rotation"][group].T


def _bounds(points, radius):
    return np.asarray(points).min(0) - radius, np.asarray(points).max(0) + radius


def _contains(row, low, high):
    assert np.isfinite(row).all()
    assert np.all(np.asarray(row[:3], dtype=float) <= low)
    assert np.all(np.asarray(row[3:], dtype=float) >= high)


def _row(result, key):
    return result.bounds[result.shape_keys.index(key)]


def _result(result):
    return {name: _json(getattr(result, name)) for name in (
        "topology_generation", "state_epoch", "world_keys", "ticks", "shape_keys",
        "shape_revisions", "bounds", "registered_pairs", "candidates")}


def _declared_scene(topology, *, world=7, materials=None, wall_low=(20, 20, 20), wall_high=(21, 21, 21)):
    materials = _materials() if materials is None else materials
    shapes, bindings, refs = [], [], {}
    for group, shape_id, kind, local, radius in materials:
        key = ph.CollisionKey(world, 100 + group, shape_id)
        ids = IDS.get(shape_id, tuple(range(len(local))))
        shapes.append(ph.CollisionShape(key, kind, ids, radius, revision=7))
        bindings.append(_api()(key, world, group, local, topology=topology))
        refs[key] = (group, np.asarray(local).copy(), radius)
    excluded = tuple(combinations(refs, 2))  # explicit self/moving-moving exclusions
    wall = ph.CollisionKey(world, 900, 1)
    shapes.append(ph.CollisionShape(wall, "hull", tuple(range(8)), revision=3))
    bindings.append(ResidentStaticBinding(wall, ph.CollisionMotion(_box(wall_low, wall_high))))
    return ph.CollisionModel(tuple(shapes), excluded), tuple(bindings), refs, wall


@pytest.mark.tyx
@pytest.mark.parametrize("q", ((0.0, 0.0), (0.37, -0.51), (-0.51, 0.37),
    (4 * pi + 0.37, -12 * pi - 0.51), (4096.0, -4096.0), (-4096.0, 4096.0)))
@pytest.mark.parametrize("placement", ("base", "cycle", "large_translation"))
def test_t07_all_finite_materials_multiturn_installation_and_current_bounds(q, placement, tyx_mode, tmp_path):
    data = _reference().model_data("T01")
    translation = (4096, -2048, 1024) if placement == "large_translation" else (0.5, -0.25, 0.75) if placement == "cycle" else (0, 0, 0)
    data, _ = _reference().transformed(data, cycled=placement != "base", translation=translation)
    program, initial = _program(_ordinary_model(data), q=q)
    with resident.TyxWorldBatch([(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        actual = batch.snapshot()[7]
        model, bindings, refs, _ = _declared_scene(actual.topology,
            wall_low=np.array(translation) + 20, wall_high=np.array(translation) + 21)
        with batch.prepare_collision(model, bindings) as plan:
            result = plan.query()
            for key, (group, local, radius) in refs.items():
                points = _world_vertices(data, group, local, actual.position)
                _contains(_row(result, key), *_bounds(points, radius))
                assert next(s for s in model.shapes if s.key == key).vertex_ids == IDS[key.shape]
                assert result.shape_revisions[result.shape_keys.index(key)] == 7
            assert result.candidates == ()
            assert result.ticks.tolist() == [0]
            assert batch.snapshot()[7] == actual
            _save(tmp_path, f"t07-all-{placement}-{q}-{tyx_mode}", {"input": data, "actual": actual,
                "local_materials": refs, "model": model, "result": _result(result), "stats": plan.stats()})


@pytest.mark.tyx
@pytest.mark.parametrize("gap", (-1 / 65536, 1 / 4))
def test_t07_predeclared_near_face_overlap_and_clear_miss(gap, tyx_mode, tmp_path):
    data = _reference().model_data("T01")
    q = np.array([0.37, -0.51], np.float32).astype(float)
    chosen = _materials()[2]  # finite, locally rotated box on the second real link
    low, high = _bounds(_world_vertices(data, chosen[0], chosen[3], q), chosen[4])
    wall_low = np.array([high[0] + gap, low[1] - 0.25, low[2] - 0.25])
    wall_high = np.array([high[0] + gap + 0.125, high[1] + 0.25, high[2] + 0.25])
    program, initial = _program(_ordinary_model(data), q=q)
    with resident.TyxWorldBatch([(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        actual = batch.snapshot()[7]
        model, bindings, refs, wall = _declared_scene(actual.topology, materials=(chosen,), wall_low=wall_low, wall_high=wall_high)
        with batch.prepare_collision(model, bindings) as plan:
            result = plan.query()
            key = next(iter(refs))
            _contains(_row(result, key), low, high)
            assert result.candidates == (((key, wall),) if gap < 0 else ())
            _save(tmp_path, f"t07-near-{gap}-{tyx_mode}", {"q": actual.position, "gap_m": gap,
                "oracle_low": low, "oracle_high": high, "result": _result(result)})


@pytest.mark.tyx
@pytest.mark.parametrize("zone", ("axis", "end_skin", "beyond_end"))
def test_t07_capsule_uses_finite_axis_and_skin_not_infinite_line(zone, tyx_mode, tmp_path):
    data = _reference().model_data("T01")
    q = np.array([0.0, pi / 2], np.float32).astype(float)
    local = np.array([[-8.0, 0, 0], [8.0, 0, 0]])
    radius = 1 / 1024
    points = _world_vertices(data, 1, local, q)
    direction = (points[1] - points[0]) / 16
    center = (points[0] + points[1]) / 2 if zone == "axis" else points[1] + direction * (radius if zone == "end_skin" else 0.25)
    half = np.full(3, radius / 2)
    program, initial = _program(_ordinary_model(data), q=q)
    with resident.TyxWorldBatch([(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        top = batch.snapshot()[7].topology
        model, bindings, refs, wall = _declared_scene(top,
            materials=((1, 12, "capsule", local, radius),), wall_low=center-half, wall_high=center+half)
        with batch.prepare_collision(model, bindings) as plan:
            result = plan.query()
            key = next(iter(refs))
            _contains(_row(result, key), *_bounds(points, radius))
            assert result.candidates == (() if zone == "beyond_end" else ((key, wall),))
            _save(tmp_path, f"t07-finite-{zone}-{tyx_mode}", {"actual": batch.snapshot()[7],
                "oracle_endpoints": points, "result": _result(result)})


@pytest.mark.tyx
@pytest.mark.parametrize("name", ("T01", "T03"))
@pytest.mark.parametrize("pose", ((0.0, 0.0), (0.37, -0.51), (-0.51, 0.37)))
def test_t07_asset_groups_fixed_tools_original_material_and_owned_source(name, pose, tyx_mode, tmp_path):
    data = _reference().model_data(name)
    asset = _asset(tmp_path, data)
    program, initial = _program(asset.model, q=pose, v=(0.25, -0.125))
    shapes, bindings, refs = [], [], {}
    for col in asset.collisions:
        group = asset.sites[col.link_name].link_index if hasattr(asset, "sites") else col.link_index
        key = ph.CollisionKey(7, 100 + group, col.shape.key.shape)
        shape = replace(col.shape, key=key)
        shapes.append(shape)
        bindings.append(_api().from_asset(asset, col.shape.key, key=key, world=7))
        body = data["names"].index(col.link_name)
        # Actual source shape IDs and original authored vertices are independent
        # of the principal-frame transform used by folded dynamics.
        if col.link_name == "tool":
            local = _box([-0.25, -0.125, -0.0625], [0.25, 0.125, 0.0625]) + [0.25, -0.125, 0]
            assert shape.vertex_ids == tuple(range(8)) and shape.kind == "hull"
        else:
            local = np.array([[0.25, -0.125, 0]])
            assert shape.vertex_ids == (0,) and shape.kind == "sphere" and shape.radius == 1 / 16
        refs[key] = (body, local, shape.radius)
    wall = ph.CollisionKey(7, 900, 1)
    shapes.append(ph.CollisionShape(wall, "hull", tuple(range(8))))
    bindings.append(ResidentStaticBinding(wall, ph.CollisionMotion(_box([20]*3, [21]*3))))
    model = ph.CollisionModel(tuple(shapes), tuple(combinations(refs, 2)))
    with resident.TyxWorldBatch([(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        with batch.prepare_collision(model, bindings) as plan:
            start = plan.query()
            assert batch.step(H) == {7: True}
            actual = batch.snapshot()[7]
            result = plan.query()
            for key, (source_body, local, radius) in refs.items():
                points = _world_vertices(data, source_body, local, actual.position)
                _contains(_row(result, key), *_bounds(points, radius))
            assert result.candidates == () and result.ticks.tolist() == [1]
            assert any(not np.array_equal(_row(start, key), _row(result, key)) for key in refs)
            saved = result.bounds.copy()
            asset.model.mass.setflags(write=True)
            asset.model.mass[:] = 999
            np.testing.assert_array_equal(plan.query().bounds, saved)
            _save(tmp_path, f"t07-source-{name}-{pose}-{tyx_mode}", {"authored": data,
                "model": model, "source_material": refs, "actual": actual, "result": _result(result)})


@pytest.mark.tyx
@pytest.mark.parametrize("problem", ("missing", "group", "body_alias", "source_radius", "source_ids", "source_revision", "foreign_model", "unexcluded_dynamic"))
def test_t07_registration_rejects_ambiguous_or_changed_sources_before_allocating(problem, tyx_mode, tmp_path):
    data = _reference().model_data("T01")
    asset = _asset(tmp_path, data)
    program, initial = _program(asset.model)
    with resident.TyxWorldBatch([(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        top = batch.snapshot()[7].topology
        model, bindings, _, _ = _declared_scene(top)
        shapes, bindings = list(model.shapes), list(bindings)
        excluded = model.excluded
        if problem == "missing":
            bindings.pop(0)
        elif problem == "group":
            bad = bindings[0]
            bindings[0] = _api()(bad.key, 7, 2, np.array([[0.25, -0.125, 0]]), topology=top)
        elif problem == "body_alias":
            bad = bindings[1]
            alias = ph.CollisionKey(7, 100, bad.key.shape)
            shapes = [replace(s, key=alias) if s.key == bad.key else s for s in shapes]
            bindings[1] = _api()(alias, 7, 1, _materials()[1][3], topology=top)
            excluded = tuple((alias if a == bad.key else a, alias if b == bad.key else b) for a, b in excluded)
        elif problem == "unexcluded_dynamic":
            excluded = ()
        else:
            col = asset.collisions[0]
            key = ph.CollisionKey(7, 100, 11)
            shapes = [replace(col.shape, key=key)]
            bindings = [_api().from_asset(asset, col.shape.key, key=key, world=7)]
            excluded = ()
            if problem == "source_radius":
                shapes[0] = replace(shapes[0], radius=1 / 8)
            elif problem == "source_ids":
                shapes[0] = replace(shapes[0], vertex_ids=(99,))
            elif problem == "source_revision":
                shapes[0] = replace(shapes[0], revision=1)
            else:
                foreign = replace(asset, model=replace(asset.model, mass=np.array([2.0, 1.0])))
                bindings[0] = _api().from_asset(foreign, col.shape.key, key=key, world=7)
        before = batch.stats()["device_allocations"]
        with pytest.raises((ValueError, TypeError), match="binding|group|body|source|shape|revision|topology|model|pair|dynamic|self"):
            batch.prepare_collision(ph.CollisionModel(tuple(shapes), excluded), tuple(bindings))
        assert batch.stats()["device_allocations"] == before
        assert not batch._collision_plans


def _matrix_worlds(count, tmp_path):
    ref = _reference()
    if count == 1:
        kinds = ("T01",)
    elif count == 4:
        kinds = ("sphere", "slider", "rotor", "T02")
    else:
        kinds = ("sphere", "slider", "rotor", "T01", "T02", "T03", "N9", "N16") * 4
    worlds, controls, rows = [], {}, {}
    for i in reversed(range(count)):
        world, kind = 7 + 13*i, kinds[i]
        if kind == "sphere":
            spec = resident.FreeSphere(2, 1 / 16, gravity=(0, 0, 0))
            controls[world] = (1 / 4, 0, 0)
            row = dict(kind=kind, data=None, shape_group=None, authored_body=None,
                local=np.zeros((1, 3)), radius=1 / 16, shape_kind="sphere", vertex_ids=(101,),
                effort=np.array([1 / 4, 0, 0]), force=None, torque=None)
        else:
            if kind in ("slider", "rotor"):
                data = ref._model([-1], ["prismatic" if kind == "slider" else "revolute"],
                    [[0, 0, 0]], [[1, 0, 0] if kind == "slider" else [0, 0, 1]],
                    [2], [[0, 0, 0] if kind == "slider" else [1 / 4, 0, 0]], [[1 / 8]*3])
                q, v, effort, force, torque = np.zeros(1), np.zeros(1), np.array([1 / 4]), np.zeros((1, 3)), np.zeros((1, 3))
            else:
                data, q, v, effort, force, torque = ref.case_input(kind)
            folder = tmp_path / f"asset-{world}"
            folder.mkdir()
            asset = _asset(folder, data) if kind == "T03" else None
            model = asset.model if asset is not None else _ordinary_model(data)
            program, initial = _program(model, q=q, v=v)
            spec = resident.PrismaticSlider(program, initial) if kind == "slider" else resident.RevoluteRotor(program, initial) if kind == "rotor" else _tree(program, initial)
            controls[world] = _action(program, effort=effort,
                force=None if asset is not None else force, torque=None if asset is not None else torque)
            row = dict(kind=kind, data=data, program=program, asset=asset,
                shape_group=model.dof_count-1, authored_body=len(data["mass"])-1,
                local=np.array([[0.25, -0.125, 0.0]]), radius=1 / 16,
                shape_kind="sphere", vertex_ids=(101,), effort=effort, force=force, torque=torque)
            if asset is not None:
                col = next(c for c in asset.collisions if c.link_name == "tool")
                row.update(source=col, local=np.asarray(col.local_vertices), radius=col.shape.radius,
                    shape_kind=col.shape.kind, vertex_ids=col.shape.vertex_ids)
        row["nearby"] = i % 2 == 0
        rows[world] = row
        worlds.append((world, spec))
    return worlds, controls, rows


def _matrix_scene(batch, rows):
    from phymeo.backends._resident_revolute_binding import ResidentRevoluteBinding
    from phymeo.backends._resident_slider_binding import ResidentSliderBinding
    snapshots = batch.snapshot()
    shapes, bindings = [], []
    for world, row in rows.items():
        group = 0 if row["shape_group"] is None else row["shape_group"]
        moving, wall = ph.CollisionKey(world, 100 + group, 11), ph.CollisionKey(world, 900, 1)
        row["moving"], row["wall"] = moving, wall
        shapes.append(ph.CollisionShape(moving, row["shape_kind"], row["vertex_ids"], row["radius"], revision=0 if row.get("asset") is not None else 7))
        state = snapshots[world]
        if row["kind"] == "sphere":
            bindings.append(ResidentSphereBinding(moving, world))
            points = np.asarray([state.position])
        else:
            data = row["data"]
            points = _world_vertices(data, row["authored_body"], row["local"], state.position)
            if row["kind"] in ("slider", "rotor"):
                bound_type = ResidentSliderBinding if row["kind"] == "slider" else ResidentRevoluteBinding
                zero = _world_vertices(data, 0, row["local"], np.zeros(1))
                bindings.append(bound_type(moving, world, ph.CollisionMotion(zero),
                    origin_m=data["offsets"][0], orientation_xyzw=data["orientations"][0], axis_local=data["axes"][0]))
            elif row.get("asset") is not None:
                bindings.append(_api().from_asset(row["asset"], row["source"].shape.key, key=moving, world=world))
            else:
                bindings.append(_api()(moving, world, group, row["local"], topology=state.topology))
        center = points.mean(0) + (0 if row["nearby"] else 1000)
        row["fixed_points"] = _box(center - 4, center + 4)
        shapes.append(ph.CollisionShape(wall, "hull", tuple(range(8)), revision=3))
        bindings.append(ResidentStaticBinding(wall, ph.CollisionMotion(row["fixed_points"])))
    return ph.CollisionModel(tuple(reversed(shapes))), tuple(reversed(bindings)), snapshots


def _advance_independent(rows, oracle):
    for world, row in rows.items():
        q, v = oracle[world]
        if row["kind"] == "sphere":
            v = v + H * row["effort"] / 2
            q = q + H*v
        else:
            q, v, _, _, _ = _reference().reference_step(row["data"], q, v,
                row["effort"], row["force"], row["torque"], H)
        oracle[world] = (q, v)


def _check_matrix_result(result, rows, states, *, oracle_only=False):
    expected = set()
    for world, row in rows.items():
        q = states[world][0] if oracle_only else states[world].position
        points = np.asarray([q]) if row["kind"] == "sphere" else _world_vertices(row["data"], row["authored_body"], row["local"], q)
        low, high = _bounds(points, row["radius"])
        bounds = _row(result, row["moving"])
        if oracle_only:
            # Warm interval has no q readback: this is only a coarse motion check
            # at an independent trajectory. Strict containment is separately
            # checked on EVERY actual q in the full-output 40-step matrix.
            budget = 2e-4
            if row["data"] is not None:
                data, ancestor = row["data"], row["authored_body"]
                chain = []
                while ancestor >= 0:
                    chain.append(ancestor)
                    ancestor = int(data["parents"][ancestor])
                columns = {body: j for j, body in enumerate(_reference().coordinates(data))}
                prismatic = [i for i in chain if data["kinds"][i] == "prismatic"]
                rotations = sum(data["kinds"][i] == "revolute" for i in chain)
                reach = sum(np.linalg.norm(data["offsets"][i]) for i in chain)
                reach += max(np.linalg.norm(p) for p in row["local"])
                reach += sum(abs(q[columns[i]]) + 2e-4 for i in prismatic)
                budget *= len(prismatic) + rotations * reach
            assert np.all(bounds[:3] <= low + budget) and np.all(bounds[3:] >= high - budget)
        else:
            _contains(bounds, low, high)
        _contains(_row(result, row["wall"]), * _bounds(row["fixed_points"], 0.0))
        if row["nearby"]:
            expected.add((row["moving"], row["wall"]))
    assert set(result.candidates) == expected
    assert len(result.registered_pairs) == len(rows)
    assert set(result.world_keys) == set(rows)
    assert all(a.world == b.world for a, b in result.registered_pairs)


@pytest.mark.tyx
@pytest.mark.parametrize("count", (1, 4, 32))
def test_t08_complete_actual_heterogeneous_forty_steps(count, tyx_mode, tmp_path):
    worlds, controls, rows = _matrix_worlds(count, tmp_path)
    report = {"h_s": H, "world_inputs": rows, "controls": controls, "trace": []}
    with resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls(controls)
        model, bindings, start = _matrix_scene(batch, rows)
        report["actual_initial"] = start
        report["model"] = model
        oracle = {w: (np.asarray(s.position), np.asarray(s.velocity)) for w, s in start.items()}
        tree_keys = [w for w, r in rows.items() if r["kind"] not in ("sphere", "slider", "rotor")]
        with batch.prepare_collision(model, bindings) as plan:
            report["initial_result"] = _result(plan.query())
            try:
                for step in range(1, 41):
                    accepted = batch.step(H)
                    actual = batch.snapshot()
                    kinematics = batch.tree_kinematics(tree_keys)
                    diagnostics = batch.tree_diagnostics(tree_keys)
                    result = plan.query()
                    _advance_independent(rows, oracle)
                    report["trace"].append({"step": step, "accepted": accepted, "actual": actual,
                        "kinematics": kinematics, "diagnostics": diagnostics, "result": _result(result),
                        "oracle_q_v": {w: [q.copy(), v.copy()] for w, (q, v) in oracle.items()},
                        "plan_stats": plan.stats(), "batch_stats": batch.stats()})
                    assert all(accepted.values())
                    for world, state in actual.items():
                        assert state.tick == step
                        _close(state.position, oracle[world][0], atol=2e-4)
                        _close(state.velocity, oracle[world][1], atol=2e-4)
                    assert result.ticks.tolist() == [step] * count
                    _check_matrix_result(result, rows, actual)
                assert {x["batch_stats"]["current_buffer"] for x in report["trace"]} == {0, 1}
            finally:
                _save(tmp_path, f"t08-full-w{count}-{tyx_mode}", report)


@pytest.mark.tyx
@pytest.mark.parametrize("count", (1, 4, 32))
def test_t08_warm_resident_queries_have_no_state_snapshot_or_dynamic_upload(count, tyx_mode, tmp_path, monkeypatch):
    worlds, controls, rows = _matrix_worlds(count, tmp_path)
    with resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls(controls)
        model, bindings, start = _matrix_scene(batch, rows)
        tree_keys = [w for w, r in rows.items() if r["kind"] not in ("sphere", "slider", "rotor")]
        with batch.prepare_collision(model, bindings) as plan:
            # Both buffer-specialized dynamics/FK/bounds paths are genuinely used
            # before the allocation gate; these are not fake prewarm counts.
            bootstrap = []
            for _ in range(2):
                assert all(batch.step(H).values())
                bootstrap.append(_result(plan.query()))
            entry = batch.snapshot()
            before, batch_before = plan.stats(), batch.stats()
            oracle = {w: (np.asarray(s.position), np.asarray(s.velocity)) for w, s in entry.items()}
            trace = {"world_inputs": rows, "controls": controls, "actual_initial": start,
                "actual_warm_entry": entry, "bootstrap": bootstrap, "plan_before": before,
                "batch_before": batch_before, "warm": []}
            held_result = plan.last_result
            held_bounds = held_result.bounds.copy()
            def forbidden(*args, **kwargs):
                raise AssertionError("warm tree path attempted snapshot, HOST envelope upload or new device allocation")
            try:
                with monkeypatch.context() as patch:
                    for name in ("snapshot", "kinematics", "tree_kinematics", "tree_diagnostics"):
                        patch.setattr(batch, name, forbidden)
                    patch.setattr(plan._engine, "query", forbidden)
                    patch.setattr(batch, "_buffer", forbidden)
                    if tyx_mode == "cuda":
                        import tyx as tx
                        patch.setattr(tx, "cuda_array", forbidden)
                        patch.setattr(tx, "cuda_empty", forbidden)
                    for step in range(3, 43):
                        accepted = batch.step(H)
                        result = plan.query()
                        _advance_independent(rows, oracle)
                        trace["warm"].append({"step": step, "accepted": accepted, "result": _result(result),
                            "oracle_only_q_v_not_device_readback": {w: [q.copy(), v.copy()] for w, (q, v) in oracle.items()},
                            "plan_stats": plan.stats(), "batch_stats": batch.stats()})
                        assert all(accepted.values()) and result.ticks.tolist() == [step] * count
                        _check_matrix_result(result, rows, oracle, oracle_only=True)
                actual_end = batch.snapshot()
                trace["actual_warm_end"] = actual_end
                trace["actual_end_fk"] = batch.tree_kinematics(tree_keys)
                _check_matrix_result(result, rows, actual_end)
                for world in rows:
                    _close(actual_end[world].position, oracle[world][0], atol=2e-4)
                    _close(actual_end[world].velocity, oracle[world][1], atol=2e-4)
                last_batch = trace["warm"][-1]["batch_stats"]
                after = trace["warm"][-1]["plan_stats"]
                assert after["device_allocations"] == before["device_allocations"]
                assert last_batch["device_allocations"] == batch_before["device_allocations"]
                assert after["dynamic_envelope_uploaded_bytes"] == 0
                assert after["tree_bounds_kernel_launches"] - before["tree_bounds_kernel_launches"] == 40
                assert after["state_buffer_directions"] == (0, 1)
                assert after["copied_to_host_bytes"] > before["copied_to_host_bytes"]
                for key in ("snapshot_downloaded_bytes", "kinematics_downloaded_bytes", "tree_diagnostic_reads"):
                    assert last_batch[key] == batch_before[key]
                np.testing.assert_array_equal(held_result.bounds, held_bounds)
                for array in (result.bounds, result.ticks):
                    with pytest.raises(ValueError):
                        array.setflags(write=True)
            finally:
                trace["final_batch_stats"] = batch.stats()
                trace["final_plan_stats"] = plan.stats()
                _save(tmp_path, f"t08-warm-w{count}-{tyx_mode}", trace)


@pytest.mark.tyx
def test_t08_tree_owner_epoch_same_tick_restore_numeric_reject_and_reset(tyx_mode, tmp_path):
    data = _reference().model_data("T01")
    program, initial = _program(_ordinary_model(data), delay=1)
    with resident.TyxWorldBatch([(7, _tree(program, initial)), (91, _tree(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        start = batch.snapshot()
        a, ab, refs, _ = _declared_scene(start[7].topology)
        b, bb, _, _ = _declared_scene(start[91].topology, world=91)
        model = ph.CollisionModel(a.shapes + b.shapes, a.excluded + b.excluded)
        with batch.prepare_collision(model, ab+bb) as plan, batch.prepare_collision(model, ab+bb) as other:
            old = plan.query()
            with pytest.raises(ValueError, match="registration|different|owner"):
                plan.require_current(other.query())
            batch.restore({7: replace(start[7], position=(0.37, -0.51))})
            changed = plan.query()
            assert changed.ticks.tolist() == old.ticks.tolist()
            assert changed.state_epoch > old.state_epoch
            assert any(not np.array_equal(_row(changed, key), _row(old, key)) for key in refs)
            with pytest.raises(ValueError, match="stale|epoch"):
                plan.require_current(old)
            invalid = replace(batch.snapshot()[7], tick=2**31-1)
            batch.restore({7: invalid})
            before = plan.query()
            batch.set_controls({7: _action(program, ff=(0.25, -0.125)),
                                91: _action(program, effort=(1, 1 / 6))})
            rejected = batch.snapshot()[7]
            assert batch.step(H) == {7: False, 91: True}
            result = plan.query()
            assert batch.snapshot()[7] == rejected
            assert batch.snapshot()[91].tick == 1
            for key in refs:
                np.testing.assert_array_equal(_row(result, key), _row(before, key))
            kept = batch.snapshot()[91]
            batch.reset([7])
            assert batch.snapshot()[7] == start[7] and batch.snapshot()[91] == kept
            reset = plan.query()
            for key in refs:
                np.testing.assert_array_equal(_row(reset, key), _row(old, key))
            _save(tmp_path, f"t08-epoch-{tyx_mode}", {"old": _result(old), "restored": _result(changed),
                "rejected": _result(result), "reset": _result(reset), "actual": batch.snapshot()})


@pytest.mark.tyx
def test_t08_capacity_and_query_failures_never_publish_partial_or_change_physics(tyx_mode, tmp_path, monkeypatch):
    data = _reference().model_data("T01")
    program, initial = _program(_ordinary_model(data))
    with resident.TyxWorldBatch([(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        start = batch.snapshot()[7]
        model, bindings, _, _ = _declared_scene(start.topology, wall_low=(-10, -10, -10), wall_high=(10, 10, 10))
        with batch.prepare_collision(model, bindings, candidate_capacity=1) as limited:
            with pytest.raises((OverflowError, ValueError, RuntimeError), match="capacity|candidate"):
                limited.query()
            assert limited.last_result is None
            assert batch.snapshot()[7] == start
        with batch.prepare_collision(model, bindings) as plan:
            old = plan.query()
            physical = batch.snapshot()
            original_epoch = batch.stats()["state_epoch"]
            with monkeypatch.context() as patch:
                def failed(**kwargs):
                    raise RuntimeError("injected tree BVH readback failure")
                patch.setattr(plan._engine, "query_device_bounds", failed)
                with pytest.raises(RuntimeError, match="tree BVH"):
                    plan.query()
            assert plan.last_result is old
            assert batch.snapshot() == physical and batch.stats()["state_epoch"] == original_epoch
            assert plan.require_current(old) is old
            retry = plan.query()
            np.testing.assert_array_equal(retry.bounds, old.bounds)
            assert retry.candidates == old.candidates
            _save(tmp_path, f"t08-query-failure-{tyx_mode}", {"result": _result(retry), "physical": physical,
                "stats": plan.stats(), "batch_stats": batch.stats()})


@pytest.mark.tyx
def test_t08_close_callback_cleans_all_tree_plans_and_runtime(tyx_mode, tmp_path, monkeypatch):
    data = _reference().model_data("T01")
    program, initial = _program(_ordinary_model(data))
    batch = resident.TyxWorldBatch([(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path)
    plans = []
    try:
        start = batch.snapshot()[7]
        model, bindings, _, _ = _declared_scene(start.topology)
        plans = [batch.prepare_collision(model, bindings) for _ in range(2)]
        results = [p.query() for p in plans]
        original = plans[0].close
        def failed_close():
            original()
            raise RuntimeError("injected tree close callback")
        monkeypatch.setattr(plans[0], "close", failed_close)
        with pytest.raises(RuntimeError, match="tree close"):
            batch.close()
        assert all(p.stats()["closed"] and p.stats()["device_owned_bytes"] == 0 for p in plans)
        assert batch.stats()["device_owned_bytes"] == 0
        with pytest.raises(RuntimeError, match="closed"):
            plans[1].query()
        _save(tmp_path, f"t08-close-{tyx_mode}", {"results": [_result(r) for r in results],
            "plan_stats": [p.stats() for p in plans], "batch_stats": batch.stats()})
    finally:
        batch.close()


@pytest.mark.tyx
def test_t08_empty_selection_and_empty_geometry_are_real_no_work(tyx_mode, tmp_path):
    program, initial = _program(_rr_model_for_empty())
    with resident.TyxWorldBatch([(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        assert batch.tree_kinematics([]) == {} and batch.tree_diagnostics([]) == {}
        with batch.prepare_collision(ph.CollisionModel(()), ()) as plan:
            result = plan.query()
            assert result.shape_keys == result.candidates == result.registered_pairs == ()
            assert result.bounds.shape == (0, 6)
            assert batch.step(H) == {7: True}
            current = plan.query()
            assert current.state_epoch > result.state_epoch
            assert current.world_keys == () and current.ticks.size == 0
            _save(tmp_path, f"t08-empty-{tyx_mode}", {"result": _result(current), "plan_stats": plan.stats()})
    with resident.TyxWorldBatch([], tyx_mode, cache_root=tmp_path / "empty") as batch:
        assert batch.tree_kinematics([]) == {} and batch.tree_diagnostics([]) == {}
        with batch.prepare_collision(ph.CollisionModel(()), ()) as plan:
            result = plan.query()
            assert result.world_keys == () and result.ticks.shape == (0,)
            assert batch.stats()["device_allocations"] == 0


def _rr_model_for_empty():
    return _ordinary_model(_reference().model_data("T01"))


@pytest.mark.tyx
def test_t07_large_actual_q_minus_to_q_plus_uses_accepted_fk_not_entry(tyx_mode, tmp_path):
    data = _reference().model_data("T01")
    program, initial = _program(_ordinary_model(data), v=(2*pi, 0))
    chosen = (1, 11, "sphere", np.array([[0.25, -0.125, 0.0]]), 1 / 16)
    expected_q = np.array([np.float32(2*pi) / 4, 0], dtype=float)
    center = _world_vertices(data, 1, chosen[3], expected_q)[0]
    with resident.TyxWorldBatch([(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        start = batch.snapshot()[7]
        model, bindings, refs, wall = _declared_scene(start.topology, materials=(chosen,),
            wall_low=center-1/32, wall_high=center+1/32)
        with batch.prepare_collision(model, bindings) as plan:
            entry = plan.query()
            assert entry.candidates == ()
            assert batch.step(1 / 4) == {7: True}
            actual = batch.snapshot()[7]
            _close(actual.position, expected_q)
            result = plan.query()
            key = next(iter(refs))
            assert result.candidates == ((key, wall),) and result.ticks.tolist() == [1]
            _contains(_row(result, key), *_bounds(_world_vertices(data, 1, chosen[3], actual.position), 1 / 16))
            assert np.linalg.norm(_row(result, key) - _row(entry, key)) > 1
            _save(tmp_path, f"t07-current-not-entry-{tyx_mode}", {"entry_state": start,
                "actual": actual, "entry_bounds": _result(entry), "accepted_bounds": _result(result)})


@pytest.mark.tyx
def test_t07_separate_prismatic_branches_and_fixed_root_static_material(tyx_mode, tmp_path):
    import xml.etree.ElementTree as ET
    data = _reference().model_data("T02")
    root = ET.fromstring(_reference().urdf_text(data))
    base = next(link for link in root.findall("link") if link.get("name") == "base")
    collision = ET.SubElement(base, "collision", name="fixed_root_ball")
    ET.SubElement(collision, "origin", xyz="-4 0 0")
    ET.SubElement(ET.SubElement(collision, "geometry"), "sphere", radius="0.0625")
    path = tmp_path / "branch_with_root.urdf"
    path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    asset = ph.load_urdf(path, gravity=(0, 0, 0))
    program, initial = _program(asset.model, q=(0.37, -1/16, 3/16), v=(0.5, 0.25, -0.125))
    shapes, bindings, refs, dynamic = [], [], {}, []
    for col in asset.collisions:
        key = ph.CollisionKey(7, 900 if col.link_index < 0 else 100+col.link_index, col.shape.key.shape)
        shapes.append(replace(col.shape, key=key))
        if col.link_index < 0:
            with pytest.raises((ValueError, TypeError), match="root|active|group|static"):
                _api().from_asset(asset, col.shape.key, key=key, world=7)
            bindings.append(ResidentStaticBinding(key, ph.CollisionMotion([[-4.0, 0, 0]])))
        else:
            bindings.append(_api().from_asset(asset, col.shape.key, key=key, world=7))
            refs[key] = col.link_index
            dynamic.append(key)
    model = ph.CollisionModel(tuple(shapes), tuple(combinations(dynamic, 2)))
    with resident.TyxWorldBatch([(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        with batch.prepare_collision(model, bindings) as plan:
            before = plan.query()
            assert batch.step(H) == {7: True}
            actual = batch.snapshot()[7]
            after = plan.query()
            for key, group in refs.items():
                points = _world_vertices(data, group, [[0.25, -0.125, 0]], actual.position)
                _contains(_row(after, key), *_bounds(points, 1 / 16))
            root_key = next(s.key for s in model.shapes if s.key.body == 900)
            np.testing.assert_array_equal(_row(after, root_key), _row(before, root_key))
            assert after.candidates == () and len(after.registered_pairs) == 3
            _save(tmp_path, f"t07-branches-root-{tyx_mode}", {"actual": actual, "source": data,
                "before": _result(before), "after": _result(after)})


@pytest.mark.tyx
def test_t08_distinct_batch_owner_and_changed_generation_are_rejected(tyx_mode, tmp_path):
    data = _reference().model_data("T01")
    program, initial = _program(_ordinary_model(data))
    with resident.TyxWorldBatch([(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path / "first") as first:
        model, bindings, _, _ = _declared_scene(first.snapshot()[7].topology)
        with first.prepare_collision(model, bindings) as plan:
            current = plan.query()
            with pytest.raises(ValueError, match="topology|generation"):
                plan.require_current(replace(current, topology_generation=current.topology_generation+1))
            with resident.TyxWorldBatch([(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path / "second") as second:
                with pytest.raises(ValueError, match="owner|different"):
                    second.collision(plan)
                with second.prepare_collision(model, bindings) as foreign:
                    with pytest.raises(ValueError, match="owner|different|registration"):
                        foreign.require_current(current)
                    assert foreign.query().state_epoch == current.state_epoch
            assert plan.require_current(current) is current
