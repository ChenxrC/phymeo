"""C3 current accepted rotor pose to finite-shape broad phase, NOT CCD/response.

Independent FP64 Rodrigues oracles use the actual stored angle. These cases do
not grant a formal all-real-number interval certificate. See the frozen casebook
for the engineering padding, finite angle domain, timing, and transfer contract.
"""

import json
from dataclasses import asdict, replace
from itertools import product
from math import cos, pi, sin

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends import tyx_world_batch as resident
from phymeo.backends._resident_collision_plan import ResidentStaticBinding


IDS = {11: (41,), 12: (9, 73), 13: (102, 18, 39, 44, 60, 3, 81, 97), 14: (71, 12, 82, 31, 64)}


def _api():
    from phymeo.backends._resident_revolute_binding import ResidentRevoluteBinding
    return ResidentRevoluteBinding


def _program(*, q=0.0, v=0.0, delay=0, origin=(0, 0, 0), orientation=(0, 0, 0, 1), axis=(0, 0, 1)):
    model = ph.RobotModel(np.array([-1]), ("revolute",), np.array([origin], dtype=float),
        np.array([orientation], dtype=float), np.array([axis], dtype=float), np.array([2.0]),
        np.array([[1 / 4, 0, 0]]), np.array([[1 / 8, 1 / 8, 1 / 8]]), np.zeros(3))
    program = ph.DrivenRobotProgram(model, kp=[0.0], kd=[0.0], effort_limit=[8.0], delay_steps=delay)
    return program, program.initial_state(ph.RobotState(np.array([q]), np.array([v])))


def _quat_matrix(xyzw):
    x, y, z, w = xyzw
    return np.array([[1 - 2 * (y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def _rotate(points, axis, q):
    a, p = np.asarray(axis, dtype=float), np.asarray(points, dtype=float)
    return p * cos(q) + np.cross(a, p) * sin(q) + (p @ a)[:, None] * a * (1 - cos(q))


def _box(low, high):
    return np.array(list(product(*zip(low, high, strict=True))), dtype=float)


def _world_points(program, local, q):
    model = program.model
    return _rotate(local, model.joint_axis[0], q) @ _quat_matrix(model.joint_orientation[0]).T + model.joint_position[0]


def _binding(key, program, local):
    model = program.model
    return _api()(key, key.world, ph.CollisionMotion(_world_points(program, local, 0.0)),
        origin_m=model.joint_position[0], orientation_xyzw=model.joint_orientation[0], axis_local=model.joint_axis[0])


def _shape_bounds(program, local, radius, q):
    points = _world_points(program, local, q)
    return points.min(0) - radius, points.max(0) + radius


def _row(result, key):
    return result.bounds[result.shape_keys.index(key)]


def _contains(row, low, high):
    # Engineering envelope, tested against independent FP64 geometry at stored q.
    assert np.all(np.asarray(row[:3], dtype=float) <= np.asarray(low, dtype=float))
    assert np.all(np.asarray(row[3:], dtype=float) >= np.asarray(high, dtype=float))


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _save(path, name, value):
    (path / f"{name}.json").write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False), encoding="utf-8")


def _result(result):
    return {"state_epoch": result.state_epoch, "topology_generation": result.topology_generation,
            "world_keys": result.world_keys, "ticks": result.ticks, "bounds": result.bounds,
            "shape_keys": [asdict(k) for k in result.shape_keys], "shape_revisions": result.shape_revisions,
            "registered_pairs": [[asdict(a), asdict(b)] for a, b in result.registered_pairs],
            "candidates": [[asdict(a), asdict(b)] for a, b in result.candidates]}


def _scene(program, *, world=7, local=((1, 0, 0),), radius=1 / 16, kind="sphere", ids=(41,), fixed_center=(1, 0, 0)):
    moving, wall = ph.CollisionKey(world, 3, 11), ph.CollisionKey(world, 9, 4)
    center = np.array(fixed_center, dtype=float)
    model = ph.CollisionModel((ph.CollisionShape(moving, kind, ids, radius, revision=7),
                              ph.CollisionShape(wall, "hull", tuple(range(8)))))
    fixed = ph.CollisionMotion(_box(center - 1 / 32, center + 1 / 32))
    return model, (_binding(moving, program, local), ResidentStaticBinding(wall, fixed)), moving, wall


@pytest.mark.tyx
@pytest.mark.parametrize("q", (0.0, pi / 2, -pi / 2, pi, 4 * pi + 0.37, -12 * pi - 0.51, 4096.0, -4096.0))
@pytest.mark.parametrize("installed", (False, True))
def test_r04_eccentric_sphere_positive_negative_multi_turn_bounds(q, installed, tyx_mode, tmp_path):
    program, initial = _program(q=q, origin=(0.5, -0.25, 0.75) if installed else (0, 0, 0),
        orientation=(0.5, 0.5, 0.5, 0.5) if installed else (0, 0, 0, 1),
        axis=(3 / 5, 0, 4 / 5) if installed else (0, 0, 1))
    model, bindings, moving, _ = _scene(program, fixed_center=(20, 20, 20))
    with resident.TyxWorldBatch([(7, resident.RevoluteRotor(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        actual = batch.snapshot()[7]
        with batch.prepare_collision(model, bindings) as plan:
            result = plan.query()
            low, high = _shape_bounds(program, [(1, 0, 0)], 1 / 16, actual.position[0])
            _contains(_row(result, moving), low, high)
            assert result.candidates == ()
            assert result.ticks.tolist() == [0]
            assert batch.snapshot()[7] == actual  # querying creates no force or advancement
            _save(tmp_path, f"r04-pose-{q:.6f}-{installed}-{tyx_mode}", {"actual": asdict(actual), "result": _result(result), "oracle_low_m": low, "oracle_high_m": high})


@pytest.mark.tyx
@pytest.mark.parametrize("zone", ("axis", "end", "beyond_finite_end"))
def test_r04_capsule_includes_axis_and_end_skin_but_not_infinite_line(zone, tyx_mode, tmp_path):
    program, initial = _program(q=pi / 2)
    center = {"axis": (0, 1, 0), "end": (0, 1.5 + 1 / 16, 0), "beyond_finite_end": (0, 2, 0)}[zone]
    local = [(0.5, 0, 0), (1.5, 0, 0)]
    model, bindings, moving, wall = _scene(program, local=local, kind="capsule", ids=(9, 73), fixed_center=center)
    with resident.TyxWorldBatch([(7, resident.RevoluteRotor(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        q = batch.snapshot()[7].position[0]
        with batch.prepare_collision(model, bindings) as plan:
            result = plan.query()
            _contains(_row(result, moving), *_shape_bounds(program, local, 1 / 16, q))
            assert result.candidates == (() if zone == "beyond_finite_end" else ((moving, wall),))


def _four_shapes(program, world=7):
    # A rotated box and a non-box five-vertex pyramid keep their original IDs.
    box = _box((-1 / 4, -1 / 32, -1 / 16), (1 / 4, 1 / 32, 1 / 16))
    box = _rotate(box, (0, 0, 1), pi / 4) + (0.75, 0.25, 0)
    specs = ((11, "sphere", [(1, 0, 0)], 1 / 16),
             (12, "capsule", [(0.5, 0, 0), (1.5, 0, 0)], 1 / 16),
             (13, "hull", box, 0.0),
             (14, "hull", [(0, 0, 0), (1 / 4, 0, 0), (1 / 4, 1 / 8, 0), (0, 1 / 8, 0), (1 / 16, 1 / 32, 1 / 4)], 0.0))
    shapes, bindings, refs = [], [], {}
    for number, kind, local, radius in specs:
        key = ph.CollisionKey(world, 3, number)
        shapes.append(ph.CollisionShape(key, kind, IDS[number], radius, revision=7))
        bindings.append(_binding(key, program, local))
        refs[key] = (np.array(local), radius)
    wall = ph.CollisionKey(world, 9, 4)
    shapes.append(ph.CollisionShape(wall, "hull", tuple(range(8))))
    bindings.append(ResidentStaticBinding(wall, ph.CollisionMotion(_box((-5, -5, -5), (5, 5, 5)))))
    return ph.CollisionModel(tuple(reversed(shapes))), tuple(reversed(bindings)), refs


@pytest.mark.tyx
def test_r04_same_body_four_shapes_keep_local_rotation_ids_and_one_accepted_tick(tyx_mode, tmp_path):
    program, initial = _program(q=-0.51, v=1 / 4, origin=(0.5, -0.25, 0.75), orientation=(0.5, 0.5, 0.5, 0.5))
    model, bindings, refs = _four_shapes(program)
    with resident.TyxWorldBatch([(7, resident.RevoluteRotor(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        with batch.prepare_collision(model, bindings) as plan:
            assert batch.step(1 / 4) == {7: True}
            state = batch.snapshot()[7]
            result = plan.query()
            assert result.ticks.tolist() == [1]
            assert len(result.candidates) == len(result.registered_pairs) == 4
            for key, (local, radius) in refs.items():
                shape = next(s for s in model.shapes if s.key == key)
                assert shape.vertex_ids == IDS[key.shape]
                assert result.shape_revisions[result.shape_keys.index(key)] == 7
                _contains(_row(result, key), *_shape_bounds(program, local, radius, state.position[0]))
            _save(tmp_path, f"r04-material-{tyx_mode}", {"checkpoint": asdict(state), "result": _result(result)})


@pytest.mark.tyx
@pytest.mark.parametrize("q", (0.37, -0.51, 8 * pi + 0.37, -4096.0))
@pytest.mark.parametrize("translated", (False, True))
@pytest.mark.parametrize("near_touch", (False, True))
def test_r04_slender_finite_shape_near_touch_and_fixed_miss_with_large_common_translation(q, translated, near_touch, tyx_mode, tmp_path):
    origin = (4096, -2048, 1024) if translated else (0, 0, 0)
    program, initial = _program(q=q, origin=origin)
    local, radius = [(-8, 0, 0), (8, 0, 0)], 1 / 1024
    # q oracle is the actual FP32 state representation, not a HOST requested angle.
    stored_q = float(np.float32(q))
    low, high = _shape_bounds(program, local, radius, stored_q)
    gap = -1 / 65536 if near_touch else 1 / 4
    fixed_low = np.array([high[0] + gap, low[1] - 1 / 8, low[2] - 1 / 8])
    fixed_high = np.array([high[0] + gap + 1 / 16, high[1] + 1 / 8, high[2] + 1 / 8])
    moving, wall = ph.CollisionKey(7, 3, 11), ph.CollisionKey(7, 9, 4)
    model = ph.CollisionModel((ph.CollisionShape(moving, "capsule", (9, 73), radius),
                              ph.CollisionShape(wall, "hull", tuple(range(8)))))
    bindings = (_binding(moving, program, local), ResidentStaticBinding(wall, ph.CollisionMotion(_box(fixed_low, fixed_high))))
    with resident.TyxWorldBatch([(7, resident.RevoluteRotor(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        assert batch.snapshot()[7].position == (stored_q,)
        with batch.prepare_collision(model, bindings) as plan:
            result = plan.query()
            _contains(_row(result, moving), low, high)
            assert result.candidates == (((moving, wall),) if near_touch else ())
            _save(tmp_path, f"r04-slender-{q:.6f}-{translated}-{near_touch}-{tyx_mode}",
                  {"stored_q_rad": stored_q, "gap_m": gap, "result": _result(result),
                   "oracle_low_m": low, "oracle_high_m": high, "stats": plan.stats()})


@pytest.mark.tyx
@pytest.mark.parametrize("delay", (0, 1))
def test_r05_forty_real_accepted_rotations_have_a_current_pose_candidate_window(delay, tyx_mode, tmp_path):
    program, initial = _program(delay=delay)
    model, bindings, moving, wall = _scene(program, fixed_center=(0, 1, 0))
    with resident.TyxWorldBatch([(7, resident.RevoluteRotor(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: program.inputs(np.zeros(1), feedforward=np.array([1 / 4]))})
        with batch.prepare_collision(model, bindings) as plan:
            trace, hits = [], []
            for n in range(1, 41):
                accepted = batch.step(1 / 16)
                state = batch.snapshot()[7]
                result = plan.query()
                row = {"step": n, "acceptance": accepted, "actual": asdict(state),
                       "result": _result(result), "plan_stats": plan.stats(), "batch_stats": batch.stats()}
                trace.append(row)
                _save(tmp_path, f"r05-forty-delay{delay}-{tyx_mode}", trace)
                m = n - delay
                assert accepted == {7: True}
                assert state.position == (m * (m + 1) / 512,)
                assert state.velocity == (m / 16,)
                _contains(_row(result, moving), *_shape_bounds(program, [(1, 0, 0)], 1 / 16, state.position[0]))
                assert result.ticks.tolist() == [n]
                if result.candidates:
                    assert result.candidates == ((moving, wall),)
                    hits.append(n)
            # Only a sampled current-pose interval; this is not a first-TOI assertion.
            assert hits == [28 + delay]
            assert batch.snapshot()[7] == state


def _world_matrix(count):
    worlds, shapes, bindings, controls, refs = [], [], [], {}, {}
    for i in reversed(range(count)):
        world, sign, delay = 7 + 13 * i, (-1 if i % 2 else 1), i % 2
        program, initial = _program(delay=delay)
        model, bound, moving, wall = _scene(program, world=world, fixed_center=(1 if i % 2 == 0 else 10, 0, 0))
        worlds.append((world, resident.RevoluteRotor(program, initial)))
        shapes.extend(model.shapes)
        bindings.extend(bound)
        controls[world] = program.inputs(np.zeros(1), feedforward=np.array([sign / 4]))
        refs[world] = (program, moving, wall, sign, delay, i % 2 == 0)
    return worlds, ph.CollisionModel(tuple(reversed(shapes))), tuple(reversed(bindings)), controls, refs


@pytest.mark.tyx
@pytest.mark.parametrize("count", (1, 4, 32))
def test_r05_warm_w1_w4_w32_two_buffers_full_results_without_state_snapshots(count, tyx_mode, tmp_path, monkeypatch):
    worlds, model, bindings, controls, refs = _world_matrix(count)
    with resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls(controls)
        with batch.prepare_collision(model, bindings) as plan:
            boot = plan.query()
            assert all(batch.step(1 / 64).values())
            first = plan.query()
            checkpoints = batch.snapshot()
            before, batch_before = plan.stats(), batch.stats()
            trace = {"bootstrap": [_result(boot), _result(first)],
                     "actual_warm_entry": {w: asdict(s) for w, s in checkpoints.items()},
                     "before": before, "batch_before": batch_before, "steps": []}
            _save(tmp_path, f"r05-w{count}-{tyx_mode}", trace)
            def forbidden(*args, **kwargs):
                raise AssertionError("warm rotor query may not snapshot, upload dynamic envelopes, or allocate")
            monkeypatch.setattr(batch, "snapshot", forbidden)
            monkeypatch.setattr(batch, "kinematics", forbidden)
            monkeypatch.setattr(plan._engine, "query", forbidden)
            if tyx_mode == "cuda":
                import tyx as tx
                monkeypatch.setattr(tx, "cuda_array", forbidden)
                monkeypatch.setattr(tx, "cuda_empty", forbidden)
            retained = first.bounds.copy()
            for n in range(2, 7):
                accepted = batch.step(1 / 64)
                result = plan.query()
                expected, oracle = set(), {}
                for world, (program, moving, wall, sign, delay, nearby) in refs.items():
                    m = n - delay
                    q, v = sign * m * (m + 1) / 8192, sign * m / 64
                    oracle[world] = {"q_rad": q, "qd_rad_s": v, "source": "analytic, not device readback"}
                    _contains(_row(result, moving), *_shape_bounds(program, [(1, 0, 0)], 1 / 16, q))
                    if nearby:
                        expected.add((moving, wall))
                trace["steps"].append({"step": n, "accepted": accepted, "result": _result(result),
                    "oracle_only_q_v": oracle, "plan_stats": plan.stats(), "batch_stats": batch.stats()})
                _save(tmp_path, f"r05-w{count}-{tyx_mode}", trace)
                assert all(accepted.values()) and result.ticks.tolist() == [n] * count
                assert len(result.registered_pairs) == count and set(result.candidates) == expected
                assert all(a.world == b.world for a, b in result.candidates)
                np.testing.assert_array_equal(first.bounds, retained)
            after = plan.stats()
            assert after["device_allocations"] == before["device_allocations"]
            assert after["dynamic_envelope_uploaded_bytes"] == 0
            assert after["revolute_bounds_kernel_launches"] - before["revolute_bounds_kernel_launches"] == 5
            assert after["state_buffer_directions"] == (0, 1)
            assert after["copied_to_host_bytes"] > before["copied_to_host_bytes"]
            assert batch.stats()["snapshot_downloaded_bytes"] == batch_before["snapshot_downloaded_bytes"]
            assert batch.stats()["kinematics_downloaded_bytes"] == batch_before["kinematics_downloaded_bytes"]
            for array in (result.bounds, result.ticks):
                with pytest.raises(ValueError):
                    array.setflags(write=True)


@pytest.mark.tyx
def test_r06_same_tick_restore_owner_reset_and_failed_world_do_not_publish_wrong_bounds(tyx_mode, tmp_path):
    program, initial = _program(delay=1)
    model, bindings, moving, _ = _scene(program)
    with resident.TyxWorldBatch([(7, resident.RevoluteRotor(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: program.inputs(np.zeros(1), feedforward=np.array([1 / 4]))})
        initial_checkpoint = batch.snapshot()[7]
        with batch.prepare_collision(model, bindings) as first, batch.prepare_collision(model, bindings) as second:
            old = first.query()
            foreign = second.query()
            with pytest.raises(ValueError, match="registration|different"):
                first.require_current(foreign)
            batch.restore({7: replace(initial_checkpoint, position=(1.0,))})
            moved = first.query()
            assert moved.ticks.tolist() == old.ticks.tolist()
            assert moved.state_epoch > old.state_epoch
            assert not np.array_equal(_row(moved, moving), _row(old, moving))
            with pytest.raises(ValueError, match="stale|epoch"):
                first.require_current(old)
            batch.reset([7])
            reset = first.query()
            np.testing.assert_array_equal(_row(reset, moving), _row(old, moving))
            batch.restore({7: replace(initial_checkpoint, position=(4096.0,), velocity=(1.0,))})
            before = first.query()
            physical = batch.snapshot()[7]
            assert batch.step(1 / 64) == {7: False}
            current = first.query()
            assert batch.snapshot()[7] == physical
            np.testing.assert_array_equal(current.bounds, before.bounds)
            assert current.ticks.tolist() == before.ticks.tolist()
            assert first.require_current(current) is current
            _save(tmp_path, f"r06-epoch-{tyx_mode}", {"old": _result(old), "restored": _result(moved), "rejected": _result(current)})


def _asset(tmp_path, folded):
    fixed = """<link name="tool"><inertial><origin xyz="0 0 0"/><mass value="1"/>
<inertia ixx="0.03125" iyy="0.03125" izz="0.03125" ixy="0" ixz="0" iyz="0"/></inertial>
<collision name="tool_box"><origin rpy="0 0 0.37"/>
<geometry><box size="0.25 0.0625 0.125"/></geometry></collision></link>
<joint name="fixed_tool" type="fixed"><parent link="arm"/><child link="tool"/>
<origin xyz="0.5 0.125 0"/></joint>""" if folded else ""
    path = tmp_path / "one_revolute.urdf"
    path.write_text("""<robot name="one_revolute_asset"><link name="base">
<collision name="root_ball"><geometry><sphere radius="0.0625"/></geometry></collision></link>
<link name="arm"><inertial><origin xyz="0.25 0 0"/><mass value="2"/>
<inertia ixx="0.125" iyy="0.125" izz="0.125" ixy="0" ixz="0" iyz="0"/></inertial>
<collision name="offset_ball"><origin xyz="1 0 0"/>
<geometry><sphere radius="0.0625"/></geometry></collision></link>
<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>
<origin xyz="0.5 -0.25 0.75" rpy="0 0 1.5707963267948966"/><axis xyz="0 0 1"/>
<limit lower="-4096" upper="4096" effort="8" velocity="4"/></joint>""" + fixed + "</robot>", encoding="utf-8")
    return ph.load_urdf(path, gravity=(0, 0, 0), collapse_fixed_joints=folded)


@pytest.mark.tyx
@pytest.mark.parametrize("folded", (False, True))
def test_r04_from_asset_binds_actual_active_group_and_preserves_original_material(folded, tyx_mode, tmp_path):
    asset = _asset(tmp_path, folded)
    program = ph.DrivenRobotProgram(asset.model, kp=[0.0], kd=[0.0], effort_limit=[8.0])
    initial = program.initial_state(ph.RobotState(np.array([-0.37]), np.array([0.25])))
    originals = asset.collision_motion(np.zeros(1))
    shapes, bindings, originals_by_key = [], [], {}
    frame = _quat_matrix(asset.model.joint_orientation[0])
    origin = asset.model.joint_position[0]
    for col in asset.collisions:
        group = asset.sites[col.link_name].link_index if folded else col.link_index
        static = group is None or group == -1
        key = ph.CollisionKey(7, 9 if static else 3, col.shape.key.shape)
        shapes.append(replace(col.shape, key=key))
        if static:
            with pytest.raises(ValueError, match="root|active|group"):
                _api().from_asset(asset, col.shape.key, key=key, world=7)
            bindings.append(ResidentStaticBinding(key, originals[col.shape.key]))
        else:
            assert group == 0
            bound = _api().from_asset(asset, col.shape.key, key=key, world=7)
            np.testing.assert_array_equal(bound.zero_motion.start, originals[col.shape.key].start)
            bindings.append(bound)
            local = (originals[col.shape.key].start - origin) @ frame
            originals_by_key[key] = (col, local)
    with resident.TyxWorldBatch([(7, resident.RevoluteRotor(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        with batch.prepare_collision(ph.CollisionModel(tuple(shapes)), tuple(bindings)) as plan:
            assert batch.step(1 / 4) == {7: True}
            accepted = batch.snapshot()[7]
            result = plan.query()
            for key, (source, local) in originals_by_key.items():
                registered = next(s for s in shapes if s.key == key)
                assert registered == replace(source.shape, key=key)
                _contains(_row(result, key), *_shape_bounds(program, local, source.shape.radius, accepted.position[0]))
            _save(tmp_path, f"r04-asset-folded{folded}-{tyx_mode}", {"checkpoint": asdict(accepted), "result": _result(result),
                "source_shapes": [{"link": c.link_name, "key": asdict(c.shape.key), "vertex_ids": c.shape.vertex_ids, "revision": c.shape.revision} for c in asset.collisions]})


@pytest.mark.tyx
@pytest.mark.parametrize("problem", ("frame_origin", "frame_axis", "source_ids", "source_revision"))
def test_r06_binding_mismatch_fails_before_device_plan_allocation(problem, tyx_mode, tmp_path):
    asset = _asset(tmp_path, False)
    col = next(c for c in asset.collisions if c.link_name == "arm")
    key = ph.CollisionKey(7, 3, 11)
    bound = _api().from_asset(asset, col.shape.key, key=key, world=7)
    shape = replace(col.shape, key=key)
    if problem == "frame_origin":
        bound = _api()(key, 7, bound.zero_motion, origin_m=(0, 0, 0), orientation_xyzw=bound.orientation_xyzw, axis_local=bound.axis_local)
    elif problem == "frame_axis":
        bound = _api()(key, 7, bound.zero_motion, origin_m=bound.origin_m, orientation_xyzw=bound.orientation_xyzw, axis_local=(1, 0, 0))
    elif problem == "source_ids":
        shape = replace(shape, vertex_ids=(999,))
    else:
        shape = replace(shape, revision=shape.revision + 1)
    program = ph.DrivenRobotProgram(asset.model, kp=[0.0], kd=[0.0], effort_limit=[8.0])
    initial = program.initial_state(ph.RobotState(np.zeros(1), np.zeros(1)))
    with resident.TyxWorldBatch([(7, resident.RevoluteRotor(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        before = batch.stats()["device_allocations"]
        with pytest.raises(ValueError, match="frame|origin|axis|source|material|revision"):
            batch.prepare_collision(ph.CollisionModel((shape,)), (bound,))
        assert batch.stats()["device_allocations"] == before
        assert not batch._collision_plans


@pytest.mark.tyx
def test_r06_query_failure_no_new_result_and_complete_plan_cleanup(tyx_mode, tmp_path, monkeypatch):
    program, initial = _program()
    model, bindings, _, _ = _scene(program)
    batch = resident.TyxWorldBatch([(7, resident.RevoluteRotor(program, initial))], tyx_mode, cache_root=tmp_path)
    plans = []
    try:
        plans = [batch.prepare_collision(model, bindings) for _ in range(2)]
        before = batch.snapshot()
        old = plans[0].query()
        with monkeypatch.context() as patch:
            def fail_query(**kwargs):
                raise RuntimeError("injected rotor BVH failure")
            patch.setattr(plans[0]._engine, "query_device_bounds", fail_query)
            with pytest.raises(RuntimeError, match="rotor BVH"):
                plans[0].query()
        assert plans[0].last_result is old and batch.snapshot() == before
        assert plans[0].require_current(old) is old
        plans[1].query()
        close_original = plans[0].close
        def failed_close():
            close_original()
            raise RuntimeError("injected rotor close callback")
        monkeypatch.setattr(plans[0], "close", failed_close)
        with pytest.raises(RuntimeError, match="rotor close callback"):
            batch.close()
        assert all(p.stats()["closed"] and p.stats()["device_owned_bytes"] == 0 for p in plans)
        assert batch.stats()["device_owned_bytes"] == 0
        _save(tmp_path, f"r06-close-{tyx_mode}", {"before": _result(old), "plans": [p.stats() for p in plans], "batch": batch.stats()})
    finally:
        batch.close()


@pytest.mark.tyx
def test_r06_capacity_empty_and_plan_close_preserve_borrowed_runtime(tyx_mode, tmp_path):
    program, initial = _program(q=pi / 2)
    model, bindings, moving, wall = _scene(program)
    with resident.TyxWorldBatch([(7, resident.RevoluteRotor(program, initial))], tyx_mode, cache_root=tmp_path) as batch:
        initial_checkpoint = batch.snapshot()[7]
        small = batch.prepare_collision(model, bindings, candidate_capacity=0)
        old = small.query()
        assert old.candidates == ()
        batch.restore({7: replace(initial_checkpoint, position=(0.0,))})
        with pytest.raises(ValueError, match="capacity"):
            small.query()
        assert small.last_result is old
        full = batch.prepare_collision(model, bindings, candidate_capacity=1)
        assert full.query().candidates == ((moving, wall),)
        small.close()
        assert batch.step(1 / 64) == {7: True}
        assert full.query().ticks.tolist() == [1]
        full.close()
        assert batch.step(1 / 64) == {7: True}
    with resident.TyxWorldBatch([], tyx_mode, cache_root=tmp_path / "empty") as empty:
        with empty.prepare_collision(ph.CollisionModel(()), ()) as plan:
            result = plan.query()
            assert result.bounds.shape == (0, 6) and result.candidates == ()
            assert plan.stats()["device_allocations"] == 0
            assert plan.stats()["revolute_bounds_kernel_launches"] == 0

